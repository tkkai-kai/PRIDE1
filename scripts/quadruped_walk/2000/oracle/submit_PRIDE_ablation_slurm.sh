#!/bin/bash
# Single PRIDE ablation group job. Submit via scripts/hpc/submit_ablation_groups.sh.

#SBATCH --job-name=pride_ablation
#SBATCH --output=slurm_output/slurm-%j-ablation.out
#SBATCH --error=slurm_output/slurm-%j-ablation.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kqian8@sheffield.ac.uk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

set -euo pipefail

FEED_TYPE="${FEED_TYPE:-0}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
GRP_LABEL="${GRP_LABEL:-X}"
DIFFUSION_SAMPLE_RATIO="${DIFFUSION_SAMPLE_RATIO:-0.5}"
USE_UNCERTAINTY_FILTER="${USE_UNCERTAINTY_FILTER:-true}"
UNCERTAINTY_KEEP_QUANTILE="${UNCERTAINTY_KEEP_QUANTILE:-0.7}"
MAX_FEEDBACK="${MAX_FEEDBACK:-700}"
RETRAIN_DIFFUSION_EVERY="${RETRAIN_DIFFUSION_EVERY:-10000}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-1000000}"
DIFFUSION_WARM_START="${DIFFUSION_WARM_START:-false}"
DIFFUSION_FINETUNE_STEPS="${DIFFUSION_FINETUNE_STEPS:-5000}"
DIFFUSION_FINETUNE_LR="${DIFFUSION_FINETUNE_LR:-1e-4}"
ADAPTIVE_DIFFUSION_RATIO="${ADAPTIVE_DIFFUSION_RATIO:-false}"
ADAPTIVE_RATIO_MAX="${ADAPTIVE_RATIO_MAX:-0.5}"
ADAPTIVE_RATIO_WARMUP="${ADAPTIVE_RATIO_WARMUP:-0}"
ADAPTIVE_RATIO_RAMP_END="${ADAPTIVE_RATIO_RAMP_END:-200000}"
SEED_LIST="${SEED_LIST:-12345 23451 34512 45123 51234}"
OUTPUT_NAME="${OUTPUT_NAME:-pride_${RUN_DATE}_grp${GRP_LABEL}}"

# shellcheck source=config/hpc/slurm_preamble.sh
source "${PRIDE_ROOT:-${SLURM_SUBMIT_DIR}}/config/hpc/slurm_preamble.sh"
pride_slurm_batch_timer_start "PRIDE group ${GRP_LABEL}"
pride_slurm_job_init train_PRIDE.py

    echo "  diffusion_sample_ratio=${DIFFUSION_SAMPLE_RATIO}"
echo "  use_uncertainty_filter=${USE_UNCERTAINTY_FILTER}"
echo "  uncertainty_keep_quantile=${UNCERTAINTY_KEEP_QUANTILE}"
    echo "  max_feedback=${MAX_FEEDBACK}"
echo "  retrain_diffusion_every=${RETRAIN_DIFFUSION_EVERY}"
echo "  train_batch_size=${TRAIN_BATCH_SIZE}"
echo "  num_train_steps=${NUM_TRAIN_STEPS}"
echo "  diffusion_warm_start=${DIFFUSION_WARM_START}"
echo "  diffusion_finetune_steps=${DIFFUSION_FINETUNE_STEPS}"
echo "  diffusion_finetune_lr=${DIFFUSION_FINETUNE_LR}"
echo "  adaptive_diffusion_ratio=${ADAPTIVE_DIFFUSION_RATIO}"
echo "  adaptive_ratio_max=${ADAPTIVE_RATIO_MAX}"
echo "  adaptive_ratio_warmup=${ADAPTIVE_RATIO_WARMUP}"
echo "  adaptive_ratio_ramp_end=${ADAPTIVE_RATIO_RAMP_END}"
echo "  seeds=${SEED_LIST}"
echo "  output_name=${OUTPUT_NAME}"

for seed in ${SEED_LIST}; do
    python train_PRIDE.py \
        env=quadruped_walk \
        seed="${seed}" \
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
        "train_batch_size=${TRAIN_BATCH_SIZE}" \
        "diffusion_sample_ratio=${DIFFUSION_SAMPLE_RATIO}" \
        "use_uncertainty_filter=${USE_UNCERTAINTY_FILTER}" \
        "uncertainty_keep_quantile=${UNCERTAINTY_KEEP_QUANTILE}" \
        "diffusion_warm_start=${DIFFUSION_WARM_START}" \
        "diffusion_finetune_steps=${DIFFUSION_FINETUNE_STEPS}" \
        "diffusion_finetune_lr=${DIFFUSION_FINETUNE_LR}" \
        "adaptive_diffusion_ratio=${ADAPTIVE_DIFFUSION_RATIO}" \
        "adaptive_ratio_max=${ADAPTIVE_RATIO_MAX}" \
        "adaptive_ratio_warmup=${ADAPTIVE_RATIO_WARMUP}" \
        "adaptive_ratio_ramp_end=${ADAPTIVE_RATIO_RAMP_END}" \
        "hydra.run.dir=outputs/${OUTPUT_NAME}_seed${seed}"
done
