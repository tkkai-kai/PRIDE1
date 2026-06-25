#!/usr/bin/env bash
# Submit one Slurm job with cluster partition/account/QoS from config/hpc/*.env
#
#   PRIDE_CLUSTER=dawn   bash config/hpc/sbatch.sh scripts/.../submit_PRIDE_ablation_slurm.sh
#   PRIDE_CLUSTER=sheffield bash config/hpc/sbatch.sh scripts/.../submit_PRIDE_ablation_slurm.sh
#
# Multi-job submitters (login node): scripts/hpc/submit_*.sh
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: PRIDE_CLUSTER={sheffield|dawn} $0 [sbatch options...] <job.sh>" >&2
    exit 1
fi

HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIDE_ROOT="$(cd "${HPC_DIR}/../.." && pwd)"
# shellcheck source=common.sh
source "${HPC_DIR}/common.sh"
pride_load_hpc_config

job_script="${@: -1}"
extra_args=("${@:1:$#-1}")

mapfile -t cluster_args < <(pride_sbatch_cluster_args)
sbatch \
    "${cluster_args[@]}" \
    --export="ALL,PRIDE_CLUSTER=${PRIDE_CLUSTER},PRIDE_ROOT=${PRIDE_ROOT}" \
    "${extra_args[@]}" \
    "${job_script}"
