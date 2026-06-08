#!/bin/bash
set -euo pipefail

# Submit all PRIDE ablation groups (A–I) from repo root:
#   bash scripts/quadruped_walk/2000/oracle/submit_PRIDE_ablation_groups.sh
#
# Optional overrides:
#   SEED_LIST="12345 23451 34512 45123 51234" RUN_DATE=20260608 FEED_TYPE=0 \
#   bash scripts/quadruped_walk/2000/oracle/submit_PRIDE_ablation_groups.sh
#
# Dry run (print sbatch commands without submitting):
#   DRY_RUN=1 bash scripts/quadruped_walk/2000/oracle/submit_PRIDE_ablation_groups.sh
#
# | Group | diffusion_sample_ratio | use_synthetic_reward_data | synthetic_reward_ratio | max_feedback |
# | ----- | ---------------------: | ------------------------- | ---------------------: | -----------: |
# | A     |                   0.25 | false                     |                    0.0 |          700 |
# | B     |                    0.5 | false                     |                    0.0 |          700 |
# | C     |                   0.75 | false                     |                    0.0 |          700 |
# | D     |                   0.25 | true                      |                   0.25 |          700 |
# | E     |                    0.5 | true                      |                   0.25 |          700 |
# | F     |                   0.75 | true                      |                   0.25 |          700 |
# | G     |                   0.25 | true                      |                    0.5 |          700 |
# | H     |                    0.5 | true                      |                    0.5 |          700 |
# | I     |                   0.75 | true                      |                    0.5 |          700 |

SUBMIT_SCRIPT="scripts/quadruped_walk/2000/oracle/submit_PRIDE_ablation_slurm.sh"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
SEED_LIST="${SEED_LIST:-12345 23451 34512 45123 51234}"
FEED_TYPE="${FEED_TYPE:-0}"
MAX_FEEDBACK="${MAX_FEEDBACK:-700}"
DRY_RUN="${DRY_RUN:-0}"

LOG_DIR="slurm_output/pride_ablation_${RUN_DATE}"
mkdir -p "${LOG_DIR}"

# grp_label | diffusion_sample_ratio | use_synthetic_reward_data | synthetic_reward_ratio
ABLATION_GROUPS=(
    "A|0.25|false|0.0"
    "B|0.5|false|0.0"
    "C|0.75|false|0.0"
    "D|0.25|true|0.25"
    "E|0.5|true|0.25"
    "F|0.75|true|0.25"
    "G|0.25|true|0.5"
    "H|0.5|true|0.5"
    "I|0.75|true|0.5"
)

ratio_tag() {
    local ratio="$1"
    echo "${ratio/./p}"
}

syn_tag() {
    local use_syn="$1"
    local syn_ratio="$2"
    if [[ "${use_syn}" == "true" ]]; then
        echo "syn$(ratio_tag "${syn_ratio}")"
    else
        echo "nosyn"
    fi
}

echo "=== Submitting PRIDE ablation groups A–I ==="
echo "  run_date=${RUN_DATE}"
echo "  seeds=${SEED_LIST}"
echo "  max_feedback=${MAX_FEEDBACK}"
echo "  log_dir=${LOG_DIR}"
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "  mode=DRY_RUN (no jobs will be submitted)"
fi

for entry in "${ABLATION_GROUPS[@]}"; do
    IFS='|' read -r grp_label dsr use_syn syn_ratio <<< "${entry}"

    dsr_tag="$(ratio_tag "${dsr}")"
    syn_part="$(syn_tag "${use_syn}" "${syn_ratio}")"
    output_name="pride_${RUN_DATE}_grp${grp_label}_dsr${dsr_tag}_${syn_part}_mf${MAX_FEEDBACK}"
    job_name="pride_${grp_label}_dsr${dsr_tag}"

    echo "Group ${grp_label}: dsr=${dsr} use_syn=${use_syn} syn_ratio=${syn_ratio} -> ${output_name}"

    export_vars="ALL,GRP_LABEL=${grp_label},DIFFUSION_SAMPLE_RATIO=${dsr},USE_SYNTHETIC_REWARD_DATA=${use_syn},SYNTHETIC_REWARD_RATIO=${syn_ratio},MAX_FEEDBACK=${MAX_FEEDBACK},SEED_LIST=${SEED_LIST},FEED_TYPE=${FEED_TYPE},RUN_DATE=${RUN_DATE},OUTPUT_NAME=${output_name}"

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "  sbatch --job-name=${job_name} --output=${LOG_DIR}/grp${grp_label}_%j.out --error=${LOG_DIR}/grp${grp_label}_%j.err --export=${export_vars} ${SUBMIT_SCRIPT}"
    else
        sbatch \
            --job-name="${job_name}" \
            --output="${LOG_DIR}/grp${grp_label}_%j.out" \
            --error="${LOG_DIR}/grp${grp_label}_%j.err" \
            --export="${export_vars}" \
            "${SUBMIT_SCRIPT}"
    fi
done

echo "All ablation groups submitted."
