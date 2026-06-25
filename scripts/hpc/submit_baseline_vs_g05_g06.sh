#!/usr/bin/env bash
# =============================================================================
# warm_start x sample_ratio ablation  (quadruped_walk, 300k steps, nosyn)
# =============================================================================
# Full 2x2(warm_start x sample_ratio) x 2(max_feedback) design:
#
#   group     warm_start  sample_ratio  max_feedback   status
#   --------  ----------  ------------  ------------   --------------------
#   baseline  false       0.50          1000 / 1400    DONE (pride_*_cmp_baseline_*)
#   g05/g06   true        0.75          1000 / 1400    DONE (pride_*_cmp_g05/g06_*)
#   g05nw     false       0.75          1000           <-- this run
#   g06nw     false       0.75          1400           <-- this run
#
# The g05nw/g06nw cells disentangle the two knobs that baseline<->g05/g06 moved together:
#   baseline (dsr0.5) vs g05nw/g06nw (dsr0.75)  -> isolates sample_ratio (no warm-start)
#   g05nw/g06nw (nowarm)  vs g05/g06 (warm)     -> isolates warm-start  (at dsr0.75)
#
# This run: 2 groups x 7 seeds = 14 Slurm jobs (one seed per job).
# Everything else is identical across groups (see FIXED block below).
#
# Submit:
#   PRIDE_CLUSTER=dawn bash scripts/hpc/submit_baseline_vs_g05_g06.sh
#   DRY_RUN=1 PRIDE_CLUSTER=dawn bash scripts/hpc/submit_baseline_vs_g05_g06.sh
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/config/hpc/common.sh"
pride_hpc_init "${REPO_ROOT}"

# 36h cap: warm groups finish ~9h; no-warm baseline (from-scratch retrain) ~32h.
SLURM_TIME="${PRIDE_SLURM_TIME:-36:00:00}"
export SLURM_TIME

SUBMIT_SCRIPT="scripts/quadruped_walk/2000/oracle/submit_PRIDE_ablation_slurm.sh"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
RUN_TAG="${RUN_TAG:-cmp}"
SEED_LIST="${SEED_LIST:-12345 23451 34512 45123 51234 67890 78906}"
FEED_TYPE="${FEED_TYPE:-0}"
DRY_RUN="${DRY_RUN:-0}"

# --- FIXED hyperparameters (identical across all groups) ---------------------
RETRAIN_DIFFUSION_EVERY="${RETRAIN_DIFFUSION_EVERY:-10000}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-300000}"
DIFFUSION_FINETUNE_STEPS="${DIFFUSION_FINETUNE_STEPS:-5000}"
DIFFUSION_FINETUNE_LR="${DIFFUSION_FINETUNE_LR:-1e-4}"

# --- Groups: label|warm_start|sample_ratio|max_feedback ----------------------
# (synthetic reward data is off for every group)
# Completed earlier (kept for the record; uncomment to re-run):
#   "baseline|false|0.5|1000"
#   "baseline|false|0.5|1400"
#   "g05|true|0.75|1000"
#   "g06|true|0.75|1400"
GROUP_SPECS=(
    "g05nw|false|0.75|1000"
    "g06nw|false|0.75|1400"
)

LOG_DIR="slurm_output/pride_${RUN_DATE}_${RUN_TAG}"
mkdir -p "${LOG_DIR}"

ratio_tag() { echo "${1/./p}"; }

echo "=== nowarm x dsr0.75 (g05nw/g06nw) | cluster=${PRIDE_CLUSTER} | tag=${RUN_TAG} ==="
echo "    fixed: retrain=${RETRAIN_DIFFUSION_EVERY} batch=${TRAIN_BATCH_SIZE} steps=${NUM_TRAIN_STEPS} nosyn"
echo "    warm:  finetune_steps=${DIFFUSION_FINETUNE_STEPS} finetune_lr=${DIFFUSION_FINETUNE_LR}"
echo "    seeds: ${SEED_LIST}"
echo "    time:  ${SLURM_TIME} per job (one seed per job)"
echo "    logs:  ${LOG_DIR}"

n_jobs=0
for spec in "${GROUP_SPECS[@]}"; do
    IFS='|' read -r grp_label warm dsr mf <<< "${spec}"
    IFS=$' \t\n'
    dsr_tag="$(ratio_tag "${dsr}")"
    warm_tag=$([[ "${warm}" == "true" ]] && echo "warm" || echo "nowarm")
    output_name="pride_${RUN_DATE}_${RUN_TAG}_${grp_label}_${warm_tag}_dsr${dsr_tag}_nosyn_mf${mf}"

    for seed in ${SEED_LIST}; do
        n_jobs=$((n_jobs + 1))
        job_name="pride_${RUN_TAG}_${grp_label}_mf${mf}_s${seed}"
        export_vars="ALL,PRIDE_CLUSTER=${PRIDE_CLUSTER},GRP_LABEL=${grp_label},DIFFUSION_SAMPLE_RATIO=${dsr},USE_SYNTHETIC_REWARD_DATA=false,SYNTHETIC_REWARD_RATIO=0.0,MAX_FEEDBACK=${mf},RETRAIN_DIFFUSION_EVERY=${RETRAIN_DIFFUSION_EVERY},TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE},NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS},DIFFUSION_WARM_START=${warm},DIFFUSION_FINETUNE_STEPS=${DIFFUSION_FINETUNE_STEPS},DIFFUSION_FINETUNE_LR=${DIFFUSION_FINETUNE_LR},SEED_LIST=${seed},FEED_TYPE=${FEED_TYPE},RUN_DATE=${RUN_DATE},OUTPUT_NAME=${output_name}"

        if [[ "${DRY_RUN}" == "1" ]]; then
            echo "DRY_RUN: pride_sbatch_submit --time=${SLURM_TIME} --job-name=${job_name} --output=${LOG_DIR}/${grp_label}_mf${mf}_s${seed}_%j.out --error=${LOG_DIR}/${grp_label}_mf${mf}_s${seed}_%j.err --export=${export_vars} ${SUBMIT_SCRIPT}"
        else
            pride_sbatch_submit \
                --job-name="${job_name}" \
                --output="${LOG_DIR}/${grp_label}_mf${mf}_s${seed}_%j.out" \
                --error="${LOG_DIR}/${grp_label}_mf${mf}_s${seed}_%j.err" \
                --export="${export_vars}" \
                "${SUBMIT_SCRIPT}"
        fi
    done
done

echo "Done. ${#GROUP_SPECS[@]} groups x 7 seeds = ${n_jobs} jobs (${SLURM_TIME} each)."
