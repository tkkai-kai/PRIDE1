#!/bin/bash
# Submit from PRIDE1 repo root, e.g.:
#   cd /users/acp26kq/projects/PRIDE1 && mkdir -p slurm_output
#   sbatch scripts/quadruped_walk/2000/oracle/submit_PRIDE_slurm.sh
#   sbatch scripts/quadruped_walk/2000/oracle/submit_PRIDE_slurm.sh 0
# Optional: FEED_TYPE=0 CONDA_ENV=synprefenv PRIDE_DEVICE=cuda
# (Slurm copies this script to spool; paths are resolved from SLURM_SUBMIT_DIR.)
# IMPORTANT: All #SBATCH lines must come before any shell assignments (Slurm stops parsing there).

#SBATCH --job-name=pride_qw_oracle
#SBATCH --output=slurm_output/slurm-%j.out
#SBATCH --error=slurm_output/slurm-%j.err
# Set --mail-user to an address Slurm can reach (see `man sbatch` / cluster docs).
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kqian8@sheffield.ac.uk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
# Stanage: default partition (sheffield) has no GPUs — use a GPU partition + QoS + GRES.
#SBATCH -p gpu-h100
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
## A100 instead (4 GPUs max per node on gpu):
##SBATCH -p gpu
##SBATCH --qos=gpu
##SBATCH --gres=gpu:1

RUN_SCRIPT="./scripts/quadruped_walk/2000/oracle/run_PRIDE.sh"
SUBMIT_SCRIPT="scripts/quadruped_walk/2000/oracle/submit_PRIDE_slurm.sh"
RUN_TAG="PRIDE"

set -euo pipefail
FEED_TYPE="${FEED_TYPE:-${1:-0}}"

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
    echo "=== ${RUN_TAG} batch end: ${end_time} (exit=${ec}) ==="
    echo "Elapsed seconds: ${elapsed}"
    printf 'Elapsed (H:M:S): %d:%02d:%02d\n' "${h}" "${m}" "${s}"
}
trap log_batch_finish EXIT
echo "=== ${RUN_TAG} batch start: ${BATCH_START_TIME} (feed_type=${FEED_TYPE}) ==="

# Slurm runs a *copy* of this script from /var/spool/... — BASH_SOURCE is not under PRIDE1.
# SLURM_SUBMIT_DIR is the directory you were in when you ran sbatch (use: cd .../PRIDE1 && sbatch ...).
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cd "${SLURM_SUBMIT_DIR}"
elif [[ -n "${PRIDE_ROOT:-}" ]]; then
    cd "${PRIDE_ROOT}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    cd "${SCRIPT_DIR}/../../../.."
fi
mkdir -p slurm_output
if [[ ! -f "${RUN_SCRIPT}" ]]; then
    echo "ERROR: ${RUN_SCRIPT} not found under $(pwd)." >&2
    echo "Submit from the PRIDE1 repo root, e.g.: cd /path/to/PRIDE1 && sbatch ${SUBMIT_SCRIPT}" >&2
    exit 1
fi

# Same setup as interactive GPU srun: clean modules, then Anaconda + conda env + EGL + cuda device.
module purge
module load Anaconda3/2024.02-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-synprefenv}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PRIDE_DEVICE="${PRIDE_DEVICE:-cuda}"

bash "${RUN_SCRIPT}" "${FEED_TYPE}"
