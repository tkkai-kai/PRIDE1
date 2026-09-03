#!/usr/bin/env bash
# One PRIDE seed on batch_relabel_synthetic_r, hyperparameter-matched to
# pride-pilot (retrain 10000). Submit through scripts/hpc/submit_paper_batchrelabel_2k.sh.
#SBATCH --job-name=pride_qw_batchrelabel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

set -euo pipefail

: "${SEED:?SEED must be exported by the submitter}"
: "${OUTPUT_NAME:?OUTPUT_NAME must be exported by the submitter}"
RETRAIN_DIFFUSION_EVERY="${RETRAIN_DIFFUSION_EVERY:-10000}"
MAX_FEEDBACK="${MAX_FEEDBACK:-2000}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-1000000}"
DIFFUSION_SAMPLE_RATIO="${DIFFUSION_SAMPLE_RATIO:-0.5}"
DIFFUSION_WARM_START="${DIFFUSION_WARM_START:-false}"
DIFFUSION_FINETUNE_STEPS="${DIFFUSION_FINETUNE_STEPS:-5000}"
DIFFUSION_FINETUNE_LR="${DIFFUSION_FINETUNE_LR:-1e-4}"
# Query-selection scheme: 0 uniform, 1 disagreement, 2 entropy, 3 k-center,
# 4 k-center+disagreement, 5 k-center+entropy. Default 0 keeps the paper runs.
FEED_TYPE="${FEED_TYPE:-2}"

# shellcheck source=config/hpc/slurm_preamble.sh
source "${PRIDE_ROOT:-${SLURM_SUBMIT_DIR}}/config/hpc/slurm_preamble.sh"
pride_slurm_job_init train_PRIDE.py

echo "method=PRIDE branch=batch_relabel_synthetic_r env=quadruped_walk seed=${SEED}"
echo "steps=${NUM_TRAIN_STEPS} feedback=${MAX_FEEDBACK} feed_type=${FEED_TYPE}"
echo "retrain_diffusion_every=${RETRAIN_DIFFUSION_EVERY} diffusion_sample_ratio=${DIFFUSION_SAMPLE_RATIO}"
echo "warm_start=${DIFFUSION_WARM_START} finetune_steps=${DIFFUSION_FINETUNE_STEPS} finetune_lr=${DIFFUSION_FINETUNE_LR}"
echo "adaptive_ratio=false diffusion_trainer_kwargs_from_cfg=false"
echo "output=outputs/${OUTPUT_NAME}_seed${SEED}"

python train_PRIDE.py \
    env=quadruped_walk \
    model_terminals=false \
    "seed=${SEED}" \
    "device=${PRIDE_DEVICE}" \
    agent.params.actor_lr=0.0001 \
    agent.params.critic_lr=0.0001 \
    gradient_update=1 \
    activation=tanh \
    num_unsup_steps=9000 \
    "num_train_steps=${NUM_TRAIN_STEPS}" \
    num_interact=30000 \
    "max_feedback=${MAX_FEEDBACK}" \
    reward_batch=200 \
    reward_update=50 \
    "feed_type=${FEED_TYPE}" \
    teacher_beta=-1 \
    teacher_gamma=1 \
    teacher_eps_mistake=0 \
    teacher_eps_skip=0 \
    teacher_eps_equal=0 \
    "retrain_diffusion_every=${RETRAIN_DIFFUSION_EVERY}" \
    "diffusion_sample_ratio=${DIFFUSION_SAMPLE_RATIO}" \
    "diffusion_warm_start=${DIFFUSION_WARM_START}" \
    "diffusion_finetune_steps=${DIFFUSION_FINETUNE_STEPS}" \
    "diffusion_finetune_lr=${DIFFUSION_FINETUNE_LR}" \
    adaptive_diffusion_ratio=false \
    diffusion_trainer_kwargs_from_cfg=false \
    "hydra.run.dir=outputs/${OUTPUT_NAME}_seed${SEED}"
