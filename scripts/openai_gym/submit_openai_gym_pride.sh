#!/usr/bin/env bash
# Stage a 1-seed PRIDE run on an OpenAI Gym MuJoCo v2 env.
#
# Usage:
#   ENV=Walker2d-v2 bash scripts/openai_gym/submit_openai_gym_pride.sh
#   ENV=Hopper-v2 RUN_TAG=hopper_v2_pilot bash scripts/openai_gym/submit_openai_gym_pride.sh
#   DRY_RUN=1 ENV=HalfCheetah-v2 bash scripts/openai_gym/submit_openai_gym_pride.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/config/hpc/common.sh"

ENV="${ENV:-Walker2d-v2}"
case "${ENV}" in
    HalfCheetah-v2|Walker2d-v2|Hopper-v2) ;;
    *)
        echo "ERROR: ENV must be HalfCheetah-v2, Walker2d-v2, or Hopper-v2 (got: ${ENV})" >&2
        exit 1
        ;;
esac

env_slug="$(printf '%s' "${ENV}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"

DRY_RUN="${DRY_RUN:-0}"
FEED_TYPE="${FEED_TYPE:-2}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
RUN_TAG="${RUN_TAG:-${env_slug}}"
SEED="${SEED:-12345}"
JOB_BODY="${REPO_ROOT}/scripts/openai_gym/openai_gym_pride_job.sh"
OUTPUT_NAME="pride_${RUN_DATE}_${RUN_TAG}_mf2000_seed${SEED}_ft${FEED_TYPE}"
LOG_DIR="${REPO_ROOT}/slurm_output/${RUN_TAG}_${RUN_DATE}"

export SLURM_PARTITION="${SLURM_PARTITION:-gpu}"
export SLURM_QOS="${SLURM_QOS:-gpu}"
export SLURM_GRES="${SLURM_GRES:-gpu:1}"
export SLURM_EXCLUDE="${SLURM_EXCLUDE:-}"
export SLURM_TIME="${SLURM_TIME:-48:00:00}"
export SLURM_MAIL_USER="${SLURM_MAIL_USER:-${USER}@sheffield.ac.uk}"
export SLURM_MAIL_TYPE="${SLURM_MAIL_TYPE:-BEGIN,END,FAIL,TIME_LIMIT}"
pride_hpc_init "${REPO_ROOT}"
mkdir -p "${LOG_DIR}"

echo "=== PRIDE OpenAI Gym pilot ==="
echo "env=${ENV} model_terminals=true seed=${SEED} steps=1000000 feedback=2000"
echo "feed_type=${FEED_TYPE} segment=50 unsupervised_steps=9000"
echo "partition=${SLURM_PARTITION} gres=${SLURM_GRES} time=${SLURM_TIME} exclude=${SLURM_EXCLUDE:-none}"
echo "mail=${SLURM_MAIL_USER} type=${SLURM_MAIL_TYPE}"
echo "output=outputs/${OUTPUT_NAME}"
echo "logs=${LOG_DIR}"

output_dir="${REPO_ROOT}/outputs/${OUTPUT_NAME}"
if [[ -e "${output_dir}" && "${ALLOW_EXISTING:-0}" != "1" ]]; then
    echo "ERROR: output already exists: ${output_dir}" >&2
    echo "Set a different RUN_TAG/RUN_DATE, or ALLOW_EXISTING=1 intentionally." >&2
    exit 1
fi

job_name="pride_${env_slug}_s${SEED}"
stdout="${LOG_DIR}/pride_${env_slug}_s${SEED}_%j.out"
stderr="${LOG_DIR}/pride_${env_slug}_s${SEED}_%j.err"
export_vars="NONE,PRIDE_ROOT=${REPO_ROOT},SEED=${SEED},OUTPUT_NAME=${OUTPUT_NAME},ENV=${ENV},FEED_TYPE=${FEED_TYPE}"

if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY_RUN: sbatch --partition=%q --qos=%q --gres=%q --time=%q' \
        "${SLURM_PARTITION}" "${SLURM_QOS}" "${SLURM_GRES}" "${SLURM_TIME}"
    [[ -n "${SLURM_EXCLUDE}" ]] && printf ' --exclude=%q' "${SLURM_EXCLUDE}"
    [[ -n "${SLURM_MAIL_USER}" ]] && printf ' --mail-user=%q --mail-type=%q' \
        "${SLURM_MAIL_USER}" "${SLURM_MAIL_TYPE}"
    printf ' --job-name=%q --output=%q --error=%q --export=%q %q\n' \
        "${job_name}" "${stdout}" "${stderr}" "${export_vars}" "${JOB_BODY}"
else
    pride_sbatch_submit \
        --job-name="${job_name}" \
        --output="${stdout}" \
        --error="${stderr}" \
        --export="${export_vars}" \
        "${JOB_BODY}"
fi

echo "Prepared 1 PRIDE job. DRY_RUN=${DRY_RUN}."
