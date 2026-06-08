#!/bin/bash
set -euo pipefail

# Submit two PRIDE experiment groups from repo root:
#   bash scripts/quadruped_walk/2000/oracle/submit_PRIDE_group_experiments.sh
#
# Optional overrides:
#   SEED_LIST="12345 23451 34512" RUN_DATE=20260603 \
#   bash scripts/quadruped_walk/2000/oracle/submit_PRIDE_group_experiments.sh

SUBMIT_SCRIPT="scripts/quadruped_walk/2000/oracle/submit_PRIDE_slurm.sh"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
SEED_LIST="${SEED_LIST:-12345 23451 34512}"
FEED_TYPE="${FEED_TYPE:-0}"

RETRAIN_VALUES=(2500 5000 7500 10000)
DIFFUSION_RATIO_VALUES=(0.25 0.5 0.75 1.0)

LOG_DIR="slurm_output/pride_groups_${RUN_DATE}"
mkdir -p "${LOG_DIR}"

echo "=== Submit group 1: retrain_diffusion_every sweep ==="
for retrain in "${RETRAIN_VALUES[@]}"; do
    output_name="pride_${RUN_DATE}_retrain${retrain}"
    echo "Submitting retrain_diffusion_every=${retrain} | seeds=${SEED_LIST}"
    sbatch \
        --job-name="pride_ret${retrain}" \
        --output="${LOG_DIR}/retrain${retrain}_%j.out" \
        --error="${LOG_DIR}/retrain${retrain}_%j.err" \
        --export=ALL,SEED_LIST="${SEED_LIST}",FEED_TYPE="${FEED_TYPE}",RETRAIN_DIFFUSION_EVERY="${retrain}",OUTPUT_NAME="${output_name}" \
        "${SUBMIT_SCRIPT}"
done

echo "=== Submit group 2: diffusion_sample_ratio sweep ==="
for ratio in "${DIFFUSION_RATIO_VALUES[@]}"; do
    ratio_tag="${ratio/./p}"
    output_name="pride_${RUN_DATE}_dsr${ratio_tag}"
    echo "Submitting diffusion_sample_ratio=${ratio} | seeds=${SEED_LIST}"
    sbatch \
        --job-name="pride_dsr${ratio_tag}" \
        --output="${LOG_DIR}/dsr${ratio_tag}_%j.out" \
        --error="${LOG_DIR}/dsr${ratio_tag}_%j.err" \
        --export=ALL,SEED_LIST="${SEED_LIST}",FEED_TYPE="${FEED_TYPE}",DIFFUSION_SAMPLE_RATIO="${ratio}",OUTPUT_NAME="${output_name}" \
        "${SUBMIT_SCRIPT}"
done

echo "All group experiments submitted."
