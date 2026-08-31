#!/usr/bin/env bash
# One PRIDE seed on an OpenAI Gym MuJoCo v2 env. Submit through
# scripts/openai_gym/submit_openai_gym_pride.sh.
#SBATCH --job-name=pride_openai_gym
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

set -euo pipefail

: "${SEED:?SEED must be exported by the submitter}"
: "${OUTPUT_NAME:?OUTPUT_NAME must be exported by the submitter}"
: "${ENV:?ENV must be exported by the submitter}"
: "${FEED_TYPE:?FEED_TYPE must be exported by the submitter}"
RETRAIN_DIFFUSION_EVERY="${RETRAIN_DIFFUSION_EVERY:-}"

# shellcheck source=config/hpc/slurm_preamble.sh
source "${PRIDE_ROOT:-${SLURM_SUBMIT_DIR}}/config/hpc/slurm_preamble.sh"
pride_slurm_job_init train_PRIDE.py

echo "method=PRIDE env=${ENV} model_terminals=true seed=${SEED} steps=1000000 feedback=2000"
if [[ -n "${RETRAIN_DIFFUSION_EVERY}" ]]; then
    echo "feed_type=${FEED_TYPE} retrain_diffusion_every=${RETRAIN_DIFFUSION_EVERY} (explicit override)"
else
    echo "feed_type=${FEED_TYPE} retrain_diffusion_every=10000 diffusion_sample_ratio=0.5 (YAML defaults)"
fi
echo "output=outputs/${OUTPUT_NAME}"

# Gym v2 names contain a hyphen; keep the override as one argv token.
train_args=(
    "env=${ENV}"
    model_terminals=true
    "seed=${SEED}"
    "device=${PRIDE_DEVICE}"
    agent.params.actor_lr=0.0001
    agent.params.critic_lr=0.0001
    gradient_update=1
    activation=tanh
    num_unsup_steps=9000
    num_train_steps=1000000
    num_interact=30000
    max_feedback=2000
    reward_batch=200
    reward_update=50
    "feed_type=${FEED_TYPE}"
    teacher_beta=-1
    teacher_gamma=1
    teacher_eps_mistake=0
    teacher_eps_skip=0
    teacher_eps_equal=0
    save_synthetic_transitions=true
)
if [[ -n "${RETRAIN_DIFFUSION_EVERY}" ]]; then
    train_args+=("retrain_diffusion_every=${RETRAIN_DIFFUSION_EVERY}")
fi

train_args+=("hydra.run.dir=outputs/${OUTPUT_NAME}")

python train_PRIDE.py "${train_args[@]}"
