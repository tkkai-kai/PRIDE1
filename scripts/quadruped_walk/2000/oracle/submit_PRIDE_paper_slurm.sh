#!/usr/bin/env bash
# One strict paper-release PRIDE seed. Submit through submit_paper_baseline.sh.
#SBATCH --job-name=pride_qw_paper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

set -euo pipefail

: "${SEED:?SEED must be exported by the submitter}"
: "${OUTPUT_NAME:?OUTPUT_NAME must be exported by the submitter}"
RETRAIN_DIFFUSION_EVERY="${RETRAIN_DIFFUSION_EVERY:-}"
SAVE_SNAPSHOTS="${SAVE_SNAPSHOTS:-1}"
# Query-selection scheme: 0 uniform, 1 disagreement, 2 entropy, 3 k-center,
# 4 k-center+disagreement, 5 k-center+entropy. Default 2 matches submit_paper_baseline.sh.
FEED_TYPE="${FEED_TYPE:-0}"

# shellcheck source=config/hpc/slurm_preamble.sh
source "${PRIDE_ROOT:-${SLURM_SUBMIT_DIR}}/config/hpc/slurm_preamble.sh"
pride_slurm_job_init train_PRIDE.py

echo "method=PRIDE env=quadruped_walk seed=${SEED} steps=1000000 feedback=2000"
if [[ -n "${RETRAIN_DIFFUSION_EVERY}" ]]; then
    echo "feed_type=${FEED_TYPE} retrain_diffusion_every=${RETRAIN_DIFFUSION_EVERY} (explicit override)"
else
    echo "feed_type=${FEED_TYPE} retrain_diffusion_every=10000 diffusion_sample_ratio=0.5 (YAML defaults)"
fi
echo "output=outputs/${OUTPUT_NAME}_seed${SEED}"

train_args=(
    env=quadruped_walk
    model_terminals=false
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
)
if [[ -n "${RETRAIN_DIFFUSION_EVERY}" ]]; then
    train_args+=("retrain_diffusion_every=${RETRAIN_DIFFUSION_EVERY}")
fi

# Snapshot steps are derived from the retrain interval rather than hard-coded:
# a step that is an exact multiple of the interval always reports
# synthetic_age=0, because the retrain fires at the top of the same loop
# iteration. Each checkpoint therefore samples the synthetic buffer just after
# regeneration, mid-cycle, and immediately before the next regeneration.
if [[ "${SAVE_SNAPSHOTS}" == "1" ]]; then
    interval="${RETRAIN_DIFFUSION_EVERY:-10000}"
    steps=()
    for base in 100000 300000 500000; do
        steps+=("$((base + 1))" "$((base + interval / 2))" "$((base + interval - 1))")
    done
    steps_csv="$(IFS=,; echo "${steps[*]}")"
    echo "snapshots=on steps=[${steps_csv}]"
    train_args+=(
        save_transition_snapshots=true
        "snapshot_steps=[${steps_csv}]"
    )
fi

train_args+=("hydra.run.dir=outputs/${OUTPUT_NAME}_seed${SEED}")

python train_PRIDE.py "${train_args[@]}"
