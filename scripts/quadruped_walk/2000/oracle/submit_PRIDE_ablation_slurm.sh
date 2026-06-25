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
USE_SYNTHETIC_REWARD_DATA="${USE_SYNTHETIC_REWARD_DATA:-false}"
SYNTHETIC_REWARD_RATIO="${SYNTHETIC_REWARD_RATIO:-0.0}"
MAX_FEEDBACK="${MAX_FEEDBACK:-700}"
RETRAIN_DIFFUSION_EVERY="${RETRAIN_DIFFUSION_EVERY:-10000}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-1000000}"
DIFFUSION_WARM_START="${DIFFUSION_WARM_START:-false}"
DIFFUSION_FINETUNE_STEPS="${DIFFUSION_FINETUNE_STEPS:-5000}"
DIFFUSION_FINETUNE_LR="${DIFFUSION_FINETUNE_LR:-1e-4}"
SEED_LIST="${SEED_LIST:-12345 23451 34512 45123 51234}"
OUTPUT_NAME="${OUTPUT_NAME:-pride_${RUN_DATE}_grp${GRP_LABEL}}"

# shellcheck source=config/hpc/slurm_preamble.sh
source "${PRIDE_ROOT:-${SLURM_SUBMIT_DIR}}/config/hpc/slurm_preamble.sh"
pride_slurm_batch_timer_start "PRIDE group ${GRP_LABEL}"
pride_slurm_job_init train_PRIDE.py

echo "  diffusion_sample_ratio=${DIFFUSION_SAMPLE_RATIO}"
echo "  use_synthetic_reward_data=${USE_SYNTHETIC_REWARD_DATA}"
echo "  synthetic_reward_ratio=${SYNTHETIC_REWARD_RATIO}"
echo "  max_feedback=${MAX_FEEDBACK}"
echo "  retrain_diffusion_every=${RETRAIN_DIFFUSION_EVERY}"
echo "  train_batch_size=${TRAIN_BATCH_SIZE}"
echo "  num_train_steps=${NUM_TRAIN_STEPS}"
echo "  diffusion_warm_start=${DIFFUSION_WARM_START}"
echo "  diffusion_finetune_steps=${DIFFUSION_FINETUNE_STEPS}"
echo "  diffusion_finetune_lr=${DIFFUSION_FINETUNE_LR}"
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
        "use_synthetic_reward_data=${USE_SYNTHETIC_REWARD_DATA}" \
        "synthetic_reward_ratio=${SYNTHETIC_REWARD_RATIO}" \
        "diffusion_warm_start=${DIFFUSION_WARM_START}" \
        "diffusion_finetune_steps=${DIFFUSION_FINETUNE_STEPS}" \
        "diffusion_finetune_lr=${DIFFUSION_FINETUNE_LR}" \
        "hydra.run.dir=outputs/${OUTPUT_NAME}_seed${seed}"
done
