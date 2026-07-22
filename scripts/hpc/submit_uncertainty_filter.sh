#!/usr/bin/env bash
# =============================================================================
# uncertainty-filter ablation  (quadruped_walk)
# =============================================================================
# Filters diffusion samples before they enter the diffusion replay buffer by the
# reward-ensemble prediction std: keep only the lowest-uncertainty fraction
# (uncertainty_keep_quantile) each retrain. Everything else is pinned to match the
# gpu-h100 baseline (pride_*_baseline_*_warm_dsr0p5_nosyn_mf1400): warm-start on,
# 1M steps, dsr=0.5, mf=1400 -- so the only variable is the filter.
#
#   group   use_uncertainty_filter  uncertainty_keep_quantile   notes
#   ------  ----------------------  -------------------------   -------------------
#   UF80    true                    0.80                        keep lowest-unc 80%
#   UF90    true                    0.90                        keep lowest-unc 90%
#   UF95    true                    0.95                        keep lowest-unc 95%
#
# (Earlier sweep, already run 20260721: UF70/0.70, UF50/0.50, UF30/0.30.)
#
# (Baseline B0 = use_uncertainty_filter=false is submitted separately.)
#
# This run: 3 groups x 5 seeds = 15 Slurm jobs (one seed per job).
#
# Submit (compare against the gpu-h100 baseline):
#   PRIDE_CLUSTER=sheffield SLURM_PARTITION=gpu-h100 bash scripts/hpc/submit_uncertainty_filter.sh
#   DRY_RUN=1 PRIDE_CLUSTER=sheffield SLURM_PARTITION=gpu-h100 bash scripts/hpc/submit_uncertainty_filter.sh
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/config/hpc/common.sh"
pride_hpc_init "${REPO_ROOT}"

# Wall-time: default to 36h. Warm-start 1M-step runs finish well within this on
# gpu-h100 (baseline took ~9h). Override with PRIDE_SLURM_TIME. Do not rely on the
# cluster default here -- it can be as low as 12h.
SLURM_TIME="${PRIDE_SLURM_TIME:-36:00:00}"
export SLURM_TIME

SUBMIT_SCRIPT="scripts/quadruped_walk/2000/oracle/submit_PRIDE_ablation_slurm.sh"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
RUN_TAG="${RUN_TAG:-uncertainty_filter}"
SEED_LIST="${SEED_LIST:-12345 23451 34512 45123 51234}"
FEED_TYPE="${FEED_TYPE:-0}"
DRY_RUN="${DRY_RUN:-0}"

# --- FIXED hyperparameters (identical across groups, matching the gpu-h100 baseline) ---
MAX_FEEDBACK="${MAX_FEEDBACK:-1400}"
RETRAIN_DIFFUSION_EVERY="${RETRAIN_DIFFUSION_EVERY:-10000}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-1000000}"
DIFFUSION_SAMPLE_RATIO="${DIFFUSION_SAMPLE_RATIO:-0.5}"
DIFFUSION_WARM_START="${DIFFUSION_WARM_START:-true}"
DIFFUSION_FINETUNE_STEPS="${DIFFUSION_FINETUNE_STEPS:-5000}"
DIFFUSION_FINETUNE_LR="${DIFFUSION_FINETUNE_LR:-1e-4}"

# --- Groups: label|uncertainty_keep_quantile --------------------------------
# Override via GROUP_SPECS, e.g. GROUP_SPECS="UF80|0.8 UF90|0.9 UF95|0.95".
GROUP_SPECS="${GROUP_SPECS:-UF80|0.8 UF90|0.9 UF95|0.95}"

LOG_DIR="slurm_output/pride_${RUN_DATE}_${RUN_TAG}"
mkdir -p "${LOG_DIR}"

echo "=== uncertainty-filter ablation | cluster=${PRIDE_CLUSTER} | tag=${RUN_TAG} ==="
echo "    groups: ${GROUP_SPECS}"
echo "    fixed:  mf=${MAX_FEEDBACK} retrain=${RETRAIN_DIFFUSION_EVERY} batch=${TRAIN_BATCH_SIZE} steps=${NUM_TRAIN_STEPS} dsr=${DIFFUSION_SAMPLE_RATIO} warm=${DIFFUSION_WARM_START} finetune=${DIFFUSION_FINETUNE_STEPS}@${DIFFUSION_FINETUNE_LR} nosyn"
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
        export_vars="ALL,PRIDE_CLUSTER=${PRIDE_CLUSTER},GRP_LABEL=${grp_label},DIFFUSION_SAMPLE_RATIO=${DIFFUSION_SAMPLE_RATIO},MAX_FEEDBACK=${MAX_FEEDBACK},RETRAIN_DIFFUSION_EVERY=${RETRAIN_DIFFUSION_EVERY},TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE},NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS},DIFFUSION_WARM_START=${DIFFUSION_WARM_START},DIFFUSION_FINETUNE_STEPS=${DIFFUSION_FINETUNE_STEPS},DIFFUSION_FINETUNE_LR=${DIFFUSION_FINETUNE_LR},USE_UNCERTAINTY_FILTER=true,UNCERTAINTY_KEEP_QUANTILE=${keep_q},SEED_LIST=${seed},FEED_TYPE=${FEED_TYPE},RUN_DATE=${RUN_DATE},OUTPUT_NAME=${output_name}"

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
