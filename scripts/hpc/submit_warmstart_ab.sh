#!/usr/bin/env bash
# =============================================================================
# warm-start A/B  (quadruped_walk, 1M steps)  -- clean single-variable control
# =============================================================================
# Two groups, IDENTICAL in every hyperparameter except diffusion_warm_start:
#
#   group   warm_start   dsr    steps   retrain   mf     adaptive  synthetic
#   ------  -----------  -----  ------  --------  -----  --------  ---------
#   nowarm  false        0.50   1e6     10000     1400   off       off
#   warm    true         0.50   1e6     10000     1400   off       off
#
#   nowarm (from-scratch retrain each time) == original PRIDE behaviour.
#   warm   (build once, then fine-tune diffusion_finetune_steps per retrain).
#   diffusion_warm_start is therefore the ONLY variable between the two arms.
#
# This run: 2 groups x 7 seeds = 14 Slurm jobs (one seed per job).
#
# Submit:
#   PRIDE_CLUSTER=sheffield bash scripts/hpc/submit_warmstart_ab.sh
#   DRY_RUN=1 PRIDE_CLUSTER=sheffield bash scripts/hpc/submit_warmstart_ab.sh
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/config/hpc/common.sh"
pride_hpc_init "${REPO_ROOT}"

# 1M @ retrain10000: no-warm (from-scratch) ~20h; warm ~3-4h. 36h covers both
# with margin (Stanage cap is 4 days). Override with PRIDE_SLURM_TIME if needed.
SLURM_TIME="${PRIDE_SLURM_TIME:-36:00:00}"
export SLURM_TIME

SUBMIT_SCRIPT="scripts/quadruped_walk/2000/oracle/submit_PRIDE_ablation_slurm.sh"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
RUN_TAG="${RUN_TAG:-warmab}"
SEED_LIST="${SEED_LIST:-12345 23451 34512 45123 51234}"
FEED_TYPE="${FEED_TYPE:-0}"
DRY_RUN="${DRY_RUN:-0}"

# --- FIXED hyperparameters (identical across both arms) ----------------------
DIFFUSION_SAMPLE_RATIO="${DIFFUSION_SAMPLE_RATIO:-0.5}"
MAX_FEEDBACK="${MAX_FEEDBACK:-1400}"
RETRAIN_DIFFUSION_EVERY="${RETRAIN_DIFFUSION_EVERY:-10000}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-1000000}"
DIFFUSION_FINETUNE_STEPS="${DIFFUSION_FINETUNE_STEPS:-5000}"
DIFFUSION_FINETUNE_LR="${DIFFUSION_FINETUNE_LR:-1e-4}"
# Held off for both arms so warm-start stays the only variable.
ADAPTIVE_DIFFUSION_RATIO="false"

# --- Groups: label|warm_start ------------------------------------------------
# Override with GROUP_SPECS_OVERRIDE (space-separated), e.g.
#   GROUP_SPECS_OVERRIDE="nowarm|false"        -> only the nowarm arm
#   GROUP_SPECS_OVERRIDE="nowarm|false warm|true"
if [[ -n "${GROUP_SPECS_OVERRIDE:-}" ]]; then
    read -r -a GROUP_SPECS <<< "${GROUP_SPECS_OVERRIDE}"
else
    GROUP_SPECS=(
        "nowarm|false"
        "warm|true"
    )
fi

LOG_DIR="slurm_output/pride_${RUN_DATE}_${RUN_TAG}"
mkdir -p "${LOG_DIR}"

ratio_tag() { echo "${1/./p}"; }
dsr_tag="$(ratio_tag "${DIFFUSION_SAMPLE_RATIO}")"

echo "=== warm-start A/B | cluster=${PRIDE_CLUSTER} | tag=${RUN_TAG} ==="
echo "    fixed: dsr=${DIFFUSION_SAMPLE_RATIO} retrain=${RETRAIN_DIFFUSION_EVERY} batch=${TRAIN_BATCH_SIZE} steps=${NUM_TRAIN_STEPS} mf=${MAX_FEEDBACK} nosyn adaptive=off"
echo "    warm:  finetune_steps=${DIFFUSION_FINETUNE_STEPS} finetune_lr=${DIFFUSION_FINETUNE_LR}"
echo "    seeds: ${SEED_LIST}"
echo "    time:  ${SLURM_TIME} per job (one seed per job)"
echo "    logs:  ${LOG_DIR}"

n_jobs=0
for spec in "${GROUP_SPECS[@]}"; do
    IFS='|' read -r grp_label warm <<< "${spec}"
    IFS=$' \t\n'
    output_name="pride_${RUN_DATE}_${RUN_TAG}_${grp_label}_dsr${dsr_tag}_nosyn_mf${MAX_FEEDBACK}"

    for seed in ${SEED_LIST}; do
        n_jobs=$((n_jobs + 1))
        job_name="pride_${RUN_TAG}_${grp_label}_s${seed}"
        export_vars="ALL,PRIDE_CLUSTER=${PRIDE_CLUSTER},GRP_LABEL=${grp_label},DIFFUSION_SAMPLE_RATIO=${DIFFUSION_SAMPLE_RATIO},ADAPTIVE_DIFFUSION_RATIO=${ADAPTIVE_DIFFUSION_RATIO},MAX_FEEDBACK=${MAX_FEEDBACK},RETRAIN_DIFFUSION_EVERY=${RETRAIN_DIFFUSION_EVERY},TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE},NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS},DIFFUSION_WARM_START=${warm},DIFFUSION_FINETUNE_STEPS=${DIFFUSION_FINETUNE_STEPS},DIFFUSION_FINETUNE_LR=${DIFFUSION_FINETUNE_LR},SEED_LIST=${seed},FEED_TYPE=${FEED_TYPE},RUN_DATE=${RUN_DATE},OUTPUT_NAME=${output_name}"

        if [[ "${DRY_RUN}" == "1" ]]; then
            echo "DRY_RUN: pride_sbatch_submit --time=${SLURM_TIME} --job-name=${job_name} --output=${LOG_DIR}/${grp_label}_s${seed}_%j.out --error=${LOG_DIR}/${grp_label}_s${seed}_%j.err --export=${export_vars} ${SUBMIT_SCRIPT}"
        else
            pride_sbatch_submit \
                --job-name="${job_name}" \
                --output="${LOG_DIR}/${grp_label}_s${seed}_%j.out" \
                --error="${LOG_DIR}/${grp_label}_s${seed}_%j.err" \
                --export="${export_vars}" \
                "${SUBMIT_SCRIPT}"
        fi
    done
done

echo "Done. ${#GROUP_SPECS[@]} groups x $(echo ${SEED_LIST} | wc -w) seeds = ${n_jobs} jobs (${SLURM_TIME} each)."
