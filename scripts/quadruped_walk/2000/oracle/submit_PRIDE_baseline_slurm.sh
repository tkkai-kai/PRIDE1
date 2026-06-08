#!/bin/bash
# PRIDE baseline — same train command as run_PRIDE.sh, 5 seeds.
#
# Submit from repo root:
#   cd /users/acp26kq/projects/PRIDE1 && mkdir -p outputs slurm_output
#   sbatch scripts/quadruped_walk/2000/oracle/submit_PRIDE_baseline_slurm.sh
#   sbatch scripts/quadruped_walk/2000/oracle/submit_PRIDE_baseline_slurm.sh 0
#
# IMPORTANT: All #SBATCH lines must come before any shell assignments (Slurm stops parsing there).

#SBATCH --job-name=pride_baseline
#SBATCH --output=slurm_output/slurm-%j-baseline.out
#SBATCH --error=slurm_output/slurm-%j-baseline.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kqian8@sheffield.ac.uk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH -p gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1

set -euo pipefail

FEED_TYPE="${FEED_TYPE:-${1:-0}}"

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
    exit 1
fi

module purge
module load Anaconda3/2024.02-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-synprefenv}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PRIDE_DEVICE="${PRIDE_DEVICE:-cuda}"

# Same as run_PRIDE.sh; only seed loop uses 5 seeds.
: "${OUTPUT_ROOT:=outputs}"
RUN_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUTPUT_ROOT}"
for seed in 12345 23451 34512 45123 51234; do
    python train_PRIDE.py env=quadruped_walk seed=$seed "device=${PRIDE_DEVICE}" agent.params.actor_lr=0.0001 agent.params.critic_lr=0.0001 gradient_update=1 activation=tanh num_unsup_steps=9000 num_train_steps=1000000 num_interact=30000 max_feedback=2000 reward_batch=200 reward_update=50 feed_type="${FEED_TYPE}" teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0 retrain_diffusion_every=2000 hydra.run.dir="${OUTPUT_ROOT}/pride_${RUN_ID}_seed${seed}"
done
