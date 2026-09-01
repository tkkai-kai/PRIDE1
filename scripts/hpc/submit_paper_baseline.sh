#!/usr/bin/env bash
# Stage the paper-aligned Quadruped-walk PRIDE/PEBBLE comparison.
#
# Usage:
#   DRY_RUN=1 bash scripts/hpc/submit_paper_baseline.sh pebble
#   DRY_RUN=1 bash scripts/hpc/submit_paper_baseline.sh pride-pilot
#   FEED_TYPE=0 bash scripts/hpc/submit_paper_baseline.sh pride-pilot
#   DRY_RUN=1 bash scripts/hpc/submit_paper_baseline.sh pride-remaining
#   DRY_RUN=1 bash scripts/hpc/submit_paper_baseline.sh pride-2k-pilot
#
# FEED_TYPE (default 2 entropy): 0 uniform, 1 disagreement, 2 entropy,
# 3 k-center, 4 k-center+disagreement, 5 k-center+entropy.
set -euo pipefail

MODE="${1:-}"
case "${MODE}" in
    pebble|pride-pilot|pride-remaining|pride-2k-pilot) ;;
    *)
        echo "Usage: $0 {pebble|pride-pilot|pride-remaining|pride-2k-pilot}" >&2
        exit 2
        ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/config/hpc/common.sh"

DRY_RUN="${DRY_RUN:-0}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
RUN_TAG="${RUN_TAG:-paper_qw_${RUN_DATE}}"
FEED_TYPE="${FEED_TYPE:-2}"
case "${FEED_TYPE}" in
    0) SAMPLING="uniform" ;;
    1) SAMPLING="disagreement" ;;
    2) SAMPLING="entropy" ;;
    3) SAMPLING="k-center" ;;
    4) SAMPLING="k-center + disagreement" ;;
    5) SAMPLING="k-center + entropy" ;;
    *) echo "ERROR: FEED_TYPE must be 0-5, got ${FEED_TYPE}" >&2; exit 2 ;;
esac
LOG_DIR="${REPO_ROOT}/slurm_output/${RUN_TAG}_ft${FEED_TYPE}_${RUN_DATE}"

case "${MODE}" in
    pebble)
        METHOD="PEBBLE"
        SEEDS=(12345 23451 34512 45123 51234)
        JOB_BODY="scripts/quadruped_walk/2000/oracle/submit_PEBBLE_paper_slurm.sh"
        OUTPUT_NAME="pebble_${RUN_DATE}_${RUN_TAG}_mf2000"
        SLURM_TIME="48:00:00"
        RETRAIN_DIFFUSION_EVERY=""
        ;;
    pride-pilot)
        METHOD="PRIDE"
        SEEDS=(12345)
        JOB_BODY="scripts/quadruped_walk/2000/oracle/submit_PRIDE_paper_slurm.sh"
        OUTPUT_NAME="pride_${RUN_DATE}_${RUN_TAG}_yaml_ret10000_mf2000"
        SLURM_TIME="48:00:00"
        RETRAIN_DIFFUSION_EVERY=""
        ;;
    pride-remaining)
        METHOD="PRIDE"
        SEEDS=(23451 34512 45123 51234)
        JOB_BODY="scripts/quadruped_walk/2000/oracle/submit_PRIDE_paper_slurm.sh"
        OUTPUT_NAME="pride_${RUN_DATE}_${RUN_TAG}_yaml_ret10000_mf2000"
        SLURM_TIME="48:00:00"
        RETRAIN_DIFFUSION_EVERY=""
        ;;
    pride-2k-pilot)
        METHOD="PRIDE"
        SEEDS=(12345)
        JOB_BODY="scripts/quadruped_walk/2000/oracle/submit_PRIDE_paper_slurm.sh"
        OUTPUT_NAME="pride_${RUN_DATE}_${RUN_TAG}_ret2000_mf2000"
        SLURM_TIME="4-00:00:00"
        RETRAIN_DIFFUSION_EVERY="2000"
        ;;
esac
OUTPUT_NAME="${OUTPUT_NAME}_ft${FEED_TYPE}"

# gpu-h100 was retired; H100s now sit in the general gpu partition.
export SLURM_PARTITION="${SLURM_PARTITION:-gpu}"
export SLURM_QOS="${SLURM_QOS:-gpu}"
export SLURM_GRES="${SLURM_GRES:-gpu:h100:1}"
export SLURM_EXCLUDE="${SLURM_EXCLUDE:-}"
export SLURM_TIME
export SLURM_MAIL_USER="${SLURM_MAIL_USER:-${USER}@sheffield.ac.uk}"
export SLURM_MAIL_TYPE="${SLURM_MAIL_TYPE:-BEGIN,END,FAIL,TIME_LIMIT}"
pride_hpc_init "${REPO_ROOT}"
mkdir -p "${LOG_DIR}"

echo "=== ${METHOD} paper baseline: ${MODE} ==="
echo "env=quadruped_walk seeds=${SEEDS[*]} steps=1000000 feedback=2000"
echo "oracle=true feed_type=${FEED_TYPE} (${SAMPLING}) segment=50 unsupervised_steps=9000"
if [[ "${METHOD}" == "PRIDE" ]]; then
    echo "diffusion_ratio=0.5 retrain_every=${RETRAIN_DIFFUSION_EVERY:-10000} warm_start=false"
fi
echo "partition=${SLURM_PARTITION} gres=${SLURM_GRES} time=${SLURM_TIME} exclude=${SLURM_EXCLUDE:-none}"
echo "mail=${SLURM_MAIL_USER} type=${SLURM_MAIL_TYPE}"
echo "output=outputs/${OUTPUT_NAME}_seed<seed>"
echo "logs=${LOG_DIR}"

for seed in "${SEEDS[@]}"; do
    output_dir="${REPO_ROOT}/outputs/${OUTPUT_NAME}_seed${seed}"
    if [[ -e "${output_dir}" && "${ALLOW_EXISTING:-0}" != "1" ]]; then
        echo "ERROR: output already exists: ${output_dir}" >&2
        echo "Set a different RUN_TAG/RUN_DATE, or ALLOW_EXISTING=1 intentionally." >&2
        exit 1
    fi

    job_name="$(printf '%s_%s_s%s' "${METHOD,,}" "${MODE}" "${seed}" | tr - _)"
    stdout="${LOG_DIR}/${METHOD,,}_${MODE}_s${seed}_%j.out"
    stderr="${LOG_DIR}/${METHOD,,}_${MODE}_s${seed}_%j.err"
    # NONE avoids leaking submit-shell conda/PATH state into the batch job.
    export_vars="NONE,PRIDE_ROOT=${REPO_ROOT},SEED=${seed},OUTPUT_NAME=${OUTPUT_NAME},FEED_TYPE=${FEED_TYPE}"
    if [[ -n "${RETRAIN_DIFFUSION_EVERY}" ]]; then
        export_vars+=",RETRAIN_DIFFUSION_EVERY=${RETRAIN_DIFFUSION_EVERY}"
    fi

    if [[ "${DRY_RUN}" == "1" ]]; then
        printf 'DRY_RUN: sbatch --partition=%q --qos=%q --gres=%q --time=%q --exclude=%q --mail-user=%q --mail-type=%q --job-name=%q --output=%q --error=%q --export=%q %q\n' \
            "${SLURM_PARTITION}" "${SLURM_QOS}" "${SLURM_GRES}" "${SLURM_TIME}" \
            "${SLURM_EXCLUDE}" "${SLURM_MAIL_USER}" "${SLURM_MAIL_TYPE}" \
            "${job_name}" "${stdout}" "${stderr}" "${export_vars}" "${JOB_BODY}"
    else
        pride_sbatch_submit \
            --job-name="${job_name}" \
            --output="${stdout}" \
            --error="${stderr}" \
            --export="${export_vars}" \
            "${JOB_BODY}"
    fi
done

echo "Prepared ${#SEEDS[@]} ${METHOD} job(s). DRY_RUN=${DRY_RUN}."
