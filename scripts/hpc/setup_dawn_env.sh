#!/usr/bin/env bash
# One-time Dawn env setup (run on login node — no srun required).
#
#   cd /path/to/PRIDE1-1 && bash scripts/hpc/setup_dawn_env.sh
#
# Optional XPU check on pvc9:
#   PRIDE_VERIFY_XPU=1 srun --account=airr-p97-dawn-gpu -p pvc9 --qos=gpu1 \
#     --nodes=1 --ntasks=1 --gres=gpu:1 --time=00:10:00 \
#     bash -lc 'cd $PWD && PRIDE_VERIFY_XPU=1 bash scripts/hpc/setup_dawn_env.sh'
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p slurm_output
SETUP_LOG="${SETUP_LOG:-${REPO_ROOT}/slurm_output/setup_dawn_env.log}"
exec > >(tee -a "${SETUP_LOG}") 2>&1
echo "=== setup_dawn_env $(date -Is) host=$(hostname) ==="

export PRIDE_CLUSTER=dawn
export PRIDE_CREATE_DAWN_ENV=1
# shellcheck disable=SC1091
source "${REPO_ROOT}/config/hpc/common.sh"
pride_load_hpc_config
pride_activate_runtime

python -c "import torch; import intel_extension_for_pytorch as ipex; print('torch', torch.__version__, 'ipex ok')"

pride_dawn_install_mesalib
pride_setup_osmesa_symlinks
pride_dawn_pip_install
pip install -e . --no-deps
cd custom_dmc2gym && pip install -e . --no-deps && cd ..

python -c "import gym, dmc2gym, dm_control, intel_extension_for_pytorch as ipex; print('deps ok')"

if [[ "${PRIDE_VERIFY_XPU:-0}" == "1" ]]; then
    python -c "import torch; import intel_extension_for_pytorch as ipex; xpu=getattr(torch,'xpu',None); print('xpu', xpu.is_available() if xpu else False)"
else
    echo "XPU check skipped (optional: PRIDE_VERIFY_XPU=1 on pvc9)."
fi

echo "Dawn env ready (${CONDA_ENV}). Log: ${SETUP_LOG}"
echo "Submit: PRIDE_CLUSTER=dawn bash config/hpc/sbatch.sh scripts/quadruped_walk/2000/oracle/submit_PRIDE_baseline_slurm.sh"
