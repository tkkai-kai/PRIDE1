#!/usr/bin/env bash
# =============================================================================
# PEBBLE baseline (quadruped_walk, oracle) — companion comparison for PRIDE.
# =============================================================================
# Submits one Slurm job per seed. Hyperparameters match the recent PRIDE runs;
# PEBBLE has no diffusion, so only the shared RL/reward args apply.
#
#   fixed:  num_train_steps=1e6  max_feedback=1400  feed_type=0 (oracle)
#   seeds:  12345 23451 34512 45123 51234  (5 jobs)
#
# Submit:
#   PRIDE_CLUSTER=sheffield bash scripts/hpc/submit_pebble_baseline.sh
#   DRY_RUN=1 PRIDE_CLUSTER=sheffield bash scripts/hpc/submit_pebble_baseline.sh
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/config/hpc/common.sh"
pride_hpc_init "${REPO_ROOT}"

# Wall-time: 1e6 steps is long; default 48h. Override with PRIDE_SLURM_TIME.
SLURM_TIME="${PRIDE_SLURM_TIME:-48:00:00}"
export SLURM_TIME

SUBMIT_SCRIPT="scripts/quadruped_walk/2000/oracle/submit_PEBBLE_slurm.sh"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
RUN_TAG="${RUN_TAG:-pebble_baseline}"
SEED_LIST="${SEED_LIST:-12345 23451 34512 45123 51234}"
FEED_TYPE="${FEED_TYPE:-0}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-1000000}"
MAX_FEEDBACK="${MAX_FEEDBACK:-1400}"
DRY_RUN="${DRY_RUN:-0}"

OUTPUT_NAME_BASE="${OUTPUT_NAME:-pebble_${RUN_DATE}_qw_oracle_mf${MAX_FEEDBACK}}"
LOG_DIR="slurm_output/${RUN_TAG}_${RUN_DATE}"
mkdir -p "${LOG_DIR}"

echo "=== PEBBLE baseline | cluster=${PRIDE_CLUSTER} | tag=${RUN_TAG} ==="
echo "    fixed: steps=${NUM_TRAIN_STEPS} mf=${MAX_FEEDBACK} feed_type=${FEED_TYPE}"
echo "    seeds: ${SEED_LIST}"
echo "    time:  ${SLURM_TIME} per job (one seed per job)"
echo "    out:   outputs/${OUTPUT_NAME_BASE}_seed<seed>"
echo "    logs:  ${LOG_DIR}"

n_jobs=0
for seed in ${SEED_LIST}; do
    n_jobs=$((n_jobs + 1))
    job_name="${RUN_TAG}_mf${MAX_FEEDBACK}_s${seed}"
    export_vars="ALL,PRIDE_CLUSTER=${PRIDE_CLUSTER},SEED_LIST=${seed},FEED_TYPE=${FEED_TYPE},NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS},MAX_FEEDBACK=${MAX_FEEDBACK},RUN_DATE=${RUN_DATE},OUTPUT_NAME=${OUTPUT_NAME_BASE}"

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "DRY_RUN: pride_sbatch_submit --job-name=${job_name} --output=${LOG_DIR}/mf${MAX_FEEDBACK}_s${seed}_%j.out --error=${LOG_DIR}/mf${MAX_FEEDBACK}_s${seed}_%j.err --export=${export_vars} ${SUBMIT_SCRIPT}"
    else
        pride_sbatch_submit \
            --job-name="${job_name}" \
            --output="${LOG_DIR}/mf${MAX_FEEDBACK}_s${seed}_%j.out" \
            --error="${LOG_DIR}/mf${MAX_FEEDBACK}_s${seed}_%j.err" \
            --export="${export_vars}" \
            "${SUBMIT_SCRIPT}"
    fi
done

echo "Done. submitted ${n_jobs} jobs${DRY_RUN:+ (DRY_RUN=${DRY_RUN})}."
