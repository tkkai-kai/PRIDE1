#!/bin/bash
set -euo pipefail

# Submit original PRIDE + synthetic ratios 0.1/0.2/0.3/0.5.
# Run from repo root:
#   bash scripts/quadruped_walk/2000/oracle/submit_PRIDE_syn_sweep.sh
# Optional:
#   RANDOM_SEED=23451 bash scripts/quadruped_walk/2000/oracle/submit_PRIDE_syn_sweep.sh

ORIGINAL_SCRIPT="scripts/quadruped_walk/2000/oracle/submit_PRIDE_slurm.sh"
SYN_SCRIPT="scripts/quadruped_walk/2000/oracle/submit_PRIDE_syn_slurm.sh"
RATIOS=(0.1 0.2 0.3 0.5)
RANDOM_SEED="${RANDOM_SEED:-12345}"
RUN_DATE="$(date +%Y%m%d)"
LOG_DIR="slurm_output/${RANDOM_SEED}_${RUN_DATE}"

mkdir -p "${LOG_DIR}"

echo "Submitting original PRIDE (no synthetic), seed=${RANDOM_SEED}"
sbatch \
    --job-name=pride_qw_oracle_orig \
    --output="${LOG_DIR}/slurm_%j.out" \
    --error="${LOG_DIR}/slurm_%j.err" \
    --export=ALL,SEED_LIST="${RANDOM_SEED}",OUTPUT_NAME="pride_${RUN_DATE}" \
    "${ORIGINAL_SCRIPT}"

for ratio in "${RATIOS[@]}"; do
    echo "Submitting PRIDE with SYNTHETIC_REWARD_RATIO=${ratio}, seed=${RANDOM_SEED}"
    sbatch \
        --job-name="pride_qw_oracle_syn${ratio/./}" \
        --output="${LOG_DIR}/slurm_%j.out" \
        --error="${LOG_DIR}/slurm_%j.err" \
        --export=ALL,SYNTHETIC_REWARD_RATIO="${ratio}",SEED_LIST="${RANDOM_SEED}",OUTPUT_NAME="pride_${RUN_DATE}_syn${ratio/./}" \
        "${SYN_SCRIPT}"
done
