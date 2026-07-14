#!/usr/bin/env bash
# =============================================================================
# adaptive diffusion sample ratio sweep  (quadruped_walk, 300k steps, nosyn)
# =============================================================================
# Compares two curriculum caps for the per-step diffusion sample ratio. With
# adaptive_diffusion_ratio=true the ratio ramps linearly 0 -> adaptive_ratio_max
# over the first adaptive_ratio_ramp_end steps, then holds. adaptive_ratio_max
# replaces the static diffusion_sample_ratio.
#
#   group   adaptive_ratio_max  warm_start  max_feedback   notes
#   ------  ------------------  ----------  ------------   ---------------------
#   ar05    0.50                true        1400           cap = 0.50
#   ar075   0.75                true        1400           cap = 0.75
#
# Everything else matches the g06 (warm) cell: warm_start=true, finetune 5000/1e-4.
#
# This run: 2 groups x 7 seeds = 14 Slurm jobs (one seed per job).
#
# Submit:
#   PRIDE_CLUSTER=dawn bash scripts/hpc/submit_adaptive_ratio.sh
#   DRY_RUN=1 PRIDE_CLUSTER=dawn bash scripts/hpc/submit_adaptive_ratio.sh
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/config/hpc/common.sh"
pride_hpc_init "${REPO_ROOT}"

# 36h cap: warm groups finish ~9h; headroom for the from-scratch first retrain.
SLURM_TIME="${PRIDE_SLURM_TIME:-36:00:00}"
export SLURM_TIME

SUBMIT_SCRIPT="scripts/quadruped_walk/2000/oracle/submit_PRIDE_ablation_slurm.sh"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
RUN_TAG="${RUN_TAG:-adaptive}"
SEED_LIST="${SEED_LIST:-12345 23451 34512 45123 51234 67890 78906}"
FEED_TYPE="${FEED_TYPE:-0}"
DRY_RUN="${DRY_RUN:-0}"

# --- FIXED hyperparameters (identical across all groups, matching g06) -------
MAX_FEEDBACK="${MAX_FEEDBACK:-1400}"
RETRAIN_DIFFUSION_EVERY="${RETRAIN_DIFFUSION_EVERY:-10000}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-300000}"
DIFFUSION_WARM_START="${DIFFUSION_WARM_START:-true}"
DIFFUSION_FINETUNE_STEPS="${DIFFUSION_FINETUNE_STEPS:-5000}"
DIFFUSION_FINETUNE_LR="${DIFFUSION_FINETUNE_LR:-1e-4}"

# --- Adaptive ratio (fixed schedule shape; only the cap differs per group) ---
ADAPTIVE_RATIO_WARMUP="${ADAPTIVE_RATIO_WARMUP:-0}"
ADAPTIVE_RATIO_RAMP_END="${ADAPTIVE_RATIO_RAMP_END:-200000}"

# --- Groups: one adaptive_ratio_max cap per group ----------------------------
# Override the caps via ARMAX_LIST, e.g. ARMAX_LIST="0.3 0.4 0.6".
# Group label is derived from the cap: 0.5 -> ar05, 0.75 -> ar075, 0.3 -> ar03.
ARMAX_LIST="${ARMAX_LIST:-0.5 0.75}"

ratio_tag() { echo "${1/./p}"; }
ratio_label() { echo "ar0${1#*.}"; }

GROUP_SPECS=()
for armax in ${ARMAX_LIST}; do
    GROUP_SPECS+=("$(ratio_label "${armax}")|${armax}")
done

LOG_DIR="slurm_output/pride_${RUN_DATE}_${RUN_TAG}"
mkdir -p "${LOG_DIR}"

echo "=== adaptive ratio sweep (caps: ${ARMAX_LIST}) | cluster=${PRIDE_CLUSTER} | tag=${RUN_TAG} ==="
echo "    fixed: warm=${DIFFUSION_WARM_START} mf=${MAX_FEEDBACK} retrain=${RETRAIN_DIFFUSION_EVERY} batch=${TRAIN_BATCH_SIZE} steps=${NUM_TRAIN_STEPS} nosyn"
echo "    warm:  finetune_steps=${DIFFUSION_FINETUNE_STEPS} finetune_lr=${DIFFUSION_FINETUNE_LR}"
echo "    ramp:  warmup=${ADAPTIVE_RATIO_WARMUP} ramp_end=${ADAPTIVE_RATIO_RAMP_END}"
echo "    seeds: ${SEED_LIST}"
echo "    time:  ${SLURM_TIME} per job (one seed per job)"
echo "    logs:  ${LOG_DIR}"

n_jobs=0
for spec in "${GROUP_SPECS[@]}"; do
    IFS='|' read -r grp_label armax <<< "${spec}"
    IFS=$' \t\n'
    armax_tag="$(ratio_tag "${armax}")"
    output_name="pride_${RUN_DATE}_${RUN_TAG}_${grp_label}_warm_armax${armax_tag}_nosyn_mf${MAX_FEEDBACK}"

    for seed in ${SEED_LIST}; do
        n_jobs=$((n_jobs + 1))
        job_name="pride_${RUN_TAG}_${grp_label}_mf${MAX_FEEDBACK}_s${seed}"
        export_vars="ALL,PRIDE_CLUSTER=${PRIDE_CLUSTER},GRP_LABEL=${grp_label},DIFFUSION_SAMPLE_RATIO=${armax},USE_SYNTHETIC_REWARD_DATA=false,SYNTHETIC_REWARD_RATIO=0.0,MAX_FEEDBACK=${MAX_FEEDBACK},RETRAIN_DIFFUSION_EVERY=${RETRAIN_DIFFUSION_EVERY},TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE},NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS},DIFFUSION_WARM_START=${DIFFUSION_WARM_START},DIFFUSION_FINETUNE_STEPS=${DIFFUSION_FINETUNE_STEPS},DIFFUSION_FINETUNE_LR=${DIFFUSION_FINETUNE_LR},ADAPTIVE_DIFFUSION_RATIO=true,ADAPTIVE_RATIO_MAX=${armax},ADAPTIVE_RATIO_WARMUP=${ADAPTIVE_RATIO_WARMUP},ADAPTIVE_RATIO_RAMP_END=${ADAPTIVE_RATIO_RAMP_END},SEED_LIST=${seed},FEED_TYPE=${FEED_TYPE},RUN_DATE=${RUN_DATE},OUTPUT_NAME=${output_name}"

        if [[ "${DRY_RUN}" == "1" ]]; then
            echo "DRY_RUN: pride_sbatch_submit --time=${SLURM_TIME} --job-name=${job_name} --output=${LOG_DIR}/${grp_label}_mf${MAX_FEEDBACK}_s${seed}_%j.out --error=${LOG_DIR}/${grp_label}_mf${MAX_FEEDBACK}_s${seed}_%j.err --export=${export_vars} ${SUBMIT_SCRIPT}"
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

echo "Done. ${#GROUP_SPECS[@]} groups x $(echo ${SEED_LIST} | wc -w) seeds = ${n_jobs} jobs (${SLURM_TIME} each)."
