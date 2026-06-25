#!/usr/bin/env bash
# Spool-safe entry for Slurm job scripts (Slurm copies the job script under /var/spool/...).
# Do not use BASH_SOURCE-relative paths to find the repo — use PRIDE_ROOT or SLURM_SUBMIT_DIR.
set -euo pipefail

_pride_root="${PRIDE_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${_pride_root}" || ! -f "${_pride_root}/config/hpc/common.sh" ]]; then
    _here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    _pride_root="${_here}"
    while [[ "${_pride_root}" != "/" && ! -f "${_pride_root}/config/hpc/common.sh" ]]; do
        _pride_root="$(dirname "${_pride_root}")"
    done
fi
if [[ ! -f "${_pride_root}/config/hpc/common.sh" ]]; then
    echo "ERROR: cannot find config/hpc/common.sh (PRIDE_ROOT=${PRIDE_ROOT:-} SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-})." >&2
    echo "Submit from repo root: PRIDE_CLUSTER=... bash config/hpc/sbatch.sh <job.sh>" >&2
    exit 1
fi
# shellcheck source=common.sh
source "${_pride_root}/config/hpc/common.sh"
