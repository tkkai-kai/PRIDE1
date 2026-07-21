#!/bin/bash
# PEBBLE baseline (one seed per job) via run_PEBBLE.sh.
#
# Tunables (env, forwarded to run_PEBBLE.sh):
#   SEED_LIST NUM_TRAIN_STEPS MAX_FEEDBACK FEED_TYPE OUTPUT_NAME RUN_DATE
#
# Submit one job directly:
#   PRIDE_CLUSTER=sheffield bash config/hpc/sbatch.sh scripts/quadruped_walk/2000/oracle/submit_PEBBLE_slurm.sh
# Submit all seeds (recommended): scripts/hpc/submit_pebble_baseline.sh

#SBATCH --job-name=pride_pebble_qw
#SBATCH --output=slurm_output/slurm-%j.out
#SBATCH --error=slurm_output/slurm-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kqian8@sheffield.ac.uk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

set -euo pipefail

FEED_TYPE="${FEED_TYPE:-${1:-0}}"
RUN_SCRIPT="scripts/quadruped_walk/2000/oracle/run_PEBBLE.sh"

# shellcheck source=config/hpc/slurm_preamble.sh
source "${PRIDE_ROOT:-${SLURM_SUBMIT_DIR}}/config/hpc/slurm_preamble.sh"
pride_slurm_batch_timer_start "PEBBLE (feed_type=${FEED_TYPE} seeds=${SEED_LIST:-default} steps=${NUM_TRAIN_STEPS:-default} mf=${MAX_FEEDBACK:-default})"
pride_slurm_job_init train_PEBBLE.py

bash "${RUN_SCRIPT}" "${FEED_TYPE}"
