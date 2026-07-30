#!/usr/bin/env bash
set -euo pipefail

_pride_root="${PRIDE_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${_pride_root}" || ! -f "${_pride_root}/config/hpc/common.sh" ]]; then
    echo "ERROR: PRIDE_ROOT or SLURM_SUBMIT_DIR must point to the repository root" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "${_pride_root}/config/hpc/common.sh"
