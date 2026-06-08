#!/bin/bash
# Single PRIDE ablation job. Submit via submit_PRIDE_ablation_groups.sh or directly:
#   cd /users/acp26kq/projects/PRIDE1 && mkdir -p outputs slurm_output
#   sbatch --export=ALL,GRP_LABEL=A,DIFFUSION_SAMPLE_RATIO=0.25,USE_SYNTHETIC_REWARD_DATA=false,SYNTHETIC_REWARD_RATIO=0.0,MAX_FEEDBACK=700 \
#     scripts/quadruped_walk/2000/oracle/submit_PRIDE_ablation_slurm.sh
#
# Only these four params vary across groups; all others match run_PRIDE.sh.
# Optional: SEED_LIST="12345 23451 34512 45123 51234" FEED_TYPE=0 CONDA_ENV=synprefenv
# IMPORTANT: All #SBATCH lines must come before any shell assignments (Slurm stops parsing there).

#SBATCH --job-name=pride_ablation
#SBATCH --output=slurm_output/slurm-%j-ablation.out
#SBATCH --error=slurm_output/slurm-%j-ablation.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kqian8@sheffield.ac.uk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH -p gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1

SUBMIT_SCRIPT="scripts/quadruped_walk/2000/oracle/submit_PRIDE_ablation_slurm.sh"
RUN_TAG="PRIDE"

set -euo pipefail

FEED_TYPE="${FEED_TYPE:-0}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
GRP_LABEL="${GRP_LABEL:-X}"
DIFFUSION_SAMPLE_RATIO="${DIFFUSION_SAMPLE_RATIO:-0.5}"
USE_SYNTHETIC_REWARD_DATA="${USE_SYNTHETIC_REWARD_DATA:-false}"
SYNTHETIC_REWARD_RATIO="${SYNTHETIC_REWARD_RATIO:-0.0}"
MAX_FEEDBACK="${MAX_FEEDBACK:-700}"
SEED_LIST="${SEED_LIST:-12345 23451 34512 45123 51234}"
OUTPUT_NAME="${OUTPUT_NAME:-pride_${RUN_DATE}_grp${GRP_LABEL}}"

BATCH_START_EPOCH=$(date +%s)
BATCH_START_TIME=$(date '+%Y-%m-%d %H:%M:%S %z')
log_batch_finish() {
    local ec=$?
    local end_epoch end_time elapsed h m s
    end_epoch=$(date +%s)
    end_time=$(date '+%Y-%m-%d %H:%M:%S %z')
    elapsed=$((end_epoch - BATCH_START_EPOCH))
    h=$((elapsed / 3600))
    m=$(((elapsed % 3600) / 60))
    s=$((elapsed % 60))
    echo "=== ${RUN_TAG} group ${GRP_LABEL} end: ${end_time} (exit=${ec}) ==="
    echo "Elapsed seconds: ${elapsed}"
    printf 'Elapsed (H:M:S): %d:%02d:%02d\n' "${h}" "${m}" "${s}"
}
trap log_batch_finish EXIT

echo "=== ${RUN_TAG} group ${GRP_LABEL} start: ${BATCH_START_TIME} ==="
echo "  diffusion_sample_ratio=${DIFFUSION_SAMPLE_RATIO}"
echo "  use_synthetic_reward_data=${USE_SYNTHETIC_REWARD_DATA}"
echo "  synthetic_reward_ratio=${SYNTHETIC_REWARD_RATIO}"
echo "  max_feedback=${MAX_FEEDBACK}"
echo "  seeds=${SEED_LIST}"
echo "  output_name=${OUTPUT_NAME}"

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cd "${SLURM_SUBMIT_DIR}"
elif [[ -n "${PRIDE_ROOT:-}" ]]; then
    cd "${PRIDE_ROOT}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    cd "${SCRIPT_DIR}/../../../.."
fi
mkdir -p slurm_output outputs
if [[ ! -f "train_PRIDE.py" ]]; then
    echo "ERROR: train_PRIDE.py not found under $(pwd)." >&2
    echo "Submit from the PRIDE1 repo root, e.g.: cd /path/to/PRIDE1 && sbatch ${SUBMIT_SCRIPT}" >&2
    exit 1
fi

module purge
module load Anaconda3/2024.02-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-synprefenv}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PRIDE_DEVICE="${PRIDE_DEVICE:-cuda}"

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
        num_train_steps=1000000 \
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
        retrain_diffusion_every=2000 \
        "diffusion_sample_ratio=${DIFFUSION_SAMPLE_RATIO}" \
        "use_synthetic_reward_data=${USE_SYNTHETIC_REWARD_DATA}" \
        "synthetic_reward_ratio=${SYNTHETIC_REWARD_RATIO}" \
        "hydra.run.dir=outputs/${OUTPUT_NAME}_seed${seed}"
done
