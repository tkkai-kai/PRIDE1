#!/usr/bin/env bash
# =============================================================================
# uncertainty-filter ablation  (quadruped_walk)
# =============================================================================
# Filters diffusion samples before they enter the diffusion replay buffer by the
# reward-ensemble prediction std: keep only the lowest-uncertainty fraction
# (uncertainty_keep_quantile) each retrain. Everything else matches the baseline
# cell already running (pride_baseline_*), so the only variable is the filter.
#
#   group   use_uncertainty_filter  uncertainty_keep_quantile   notes
#   ------  ----------------------  -------------------------   -------------------
#   UF70    true                    0.70                        keep lowest-unc 70%
#   UF50    true                    0.50                        keep lowest-unc 50%
#
# (Baseline B0 = use_uncertainty_filter=false is submitted separately.)
#
# This run: 2 groups x 7 seeds = 14 Slurm jobs (one seed per job).
#
# Submit:
#   PRIDE_CLUSTER=dawn bash scripts/hpc/submit_uncertainty_filter.sh
#   DRY_RUN=1 PRIDE_CLUSTER=dawn bash scripts/hpc/submit_uncertainty_filter.sh
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/config/hpc/common.sh"
pride_hpc_init "${REPO_ROOT}"

# Wall-time: default to 36h (nowarm 300k needs ~28h). Override with PRIDE_SLURM_TIME.
# NOTE: do not rely on the cluster default here -- it can be as low as 12h, which
# is nowhere near enough for a from-scratch (nowarm) diffusion retrain run.
SLURM_TIME="${PRIDE_SLURM_TIME:-36:00:00}"
export SLURM_TIME

SUBMIT_SCRIPT="scripts/quadruped_walk/2000/oracle/submit_PRIDE_ablation_slurm.sh"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
RUN_TAG="${RUN_TAG:-uncertainty_filter_0.5&0.7}"
SEED_LIST="${SEED_LIST:-12345 23451 34512 45123 51234 67890 78906}"
FEED_TYPE="${FEED_TYPE:-0}"
DRY_RUN="${DRY_RUN:-0}"

# --- FIXED hyperparameters (identical across groups, matching the baseline) ---
MAX_FEEDBACK="${MAX_FEEDBACK:-1400}"
RETRAIN_DIFFUSION_EVERY="${RETRAIN_DIFFUSION_EVERY:-10000}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-300000}"
DIFFUSION_SAMPLE_RATIO="${DIFFUSION_SAMPLE_RATIO:-0.5}"
DIFFUSION_WARM_START="${DIFFUSION_WARM_START:-false}"

# --- Groups: label|uncertainty_keep_quantile --------------------------------
# Override via GROUP_SPECS, e.g. GROUP_SPECS="UF70|0.7 UF50|0.5 UF30|0.3".
GROUP_SPECS="${GROUP_SPECS:-UF70|0.7 UF50|0.5}"

LOG_DIR="slurm_output/pride_${RUN_DATE}_${RUN_TAG}"
mkdir -p "${LOG_DIR}"

echo "=== uncertainty-filter ablation | cluster=${PRIDE_CLUSTER} | tag=${RUN_TAG} ==="
echo "    groups: ${GROUP_SPECS}"
echo "    fixed:  mf=${MAX_FEEDBACK} retrain=${RETRAIN_DIFFUSION_EVERY} batch=${TRAIN_BATCH_SIZE} steps=${NUM_TRAIN_STEPS} dsr=${DIFFUSION_SAMPLE_RATIO} warm=${DIFFUSION_WARM_START} nosyn"
echo "    seeds:  ${SEED_LIST}"
echo "    time:   ${SLURM_TIME:-<cluster default>} per job (one seed per job)"
echo "    logs:   ${LOG_DIR}"

n_jobs=0
for spec in ${GROUP_SPECS}; do
    IFS='|' read -r grp_label keep_q <<< "${spec}"
    IFS=$' \t\n'
    q_tag="${keep_q/./p}"
    output_name="pride_${RUN_DATE}_${RUN_TAG}_${grp_label}_q${q_tag}_nosyn_mf${MAX_FEEDBACK}"

    for seed in ${SEED_LIST}; do
        n_jobs=$((n_jobs + 1))
        job_name="pride_${RUN_TAG}_${grp_label}_mf${MAX_FEEDBACK}_s${seed}"
        export_vars="ALL,PRIDE_CLUSTER=${PRIDE_CLUSTER},GRP_LABEL=${grp_label},DIFFUSION_SAMPLE_RATIO=${DIFFUSION_SAMPLE_RATIO},MAX_FEEDBACK=${MAX_FEEDBACK},RETRAIN_DIFFUSION_EVERY=${RETRAIN_DIFFUSION_EVERY},TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE},NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS},DIFFUSION_WARM_START=${DIFFUSION_WARM_START},USE_UNCERTAINTY_FILTER=true,UNCERTAINTY_KEEP_QUANTILE=${keep_q},SEED_LIST=${seed},FEED_TYPE=${FEED_TYPE},RUN_DATE=${RUN_DATE},OUTPUT_NAME=${output_name}"

        if [[ "${DRY_RUN}" == "1" ]]; then
            echo "DRY_RUN: pride_sbatch_submit --job-name=${job_name} --output=${LOG_DIR}/${grp_label}_mf${MAX_FEEDBACK}_s${seed}_%j.out --error=${LOG_DIR}/${grp_label}_mf${MAX_FEEDBACK}_s${seed}_%j.err --export=${export_vars} ${SUBMIT_SCRIPT}"
        else
            pride_sbatch_submit \
                --job-name="${job_name}" \
                --output="${LOG_DIR}/${grp_label}_mf${MAX_FEEDBACK}_s${seed}_%j.out" \
                --error="${LOG_DIR}/${grp_label}_mf${MAX_FEEDBACK}_s${seed}_%j.err" \
                --export="${export_vars}" \
                "${SUBMIT_SCRIPT}"
        fi
    done
done

echo "Done. submitted ${n_jobs} jobs${DRY_RUN:+ (DRY_RUN=${DRY_RUN})}."
