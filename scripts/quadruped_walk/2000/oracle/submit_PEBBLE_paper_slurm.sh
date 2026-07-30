#!/usr/bin/env bash
# One paper-aligned PEBBLE seed. Submit through submit_paper_baseline.sh.
#SBATCH --job-name=pebble_qw_paper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

set -euo pipefail

: "${SEED:?SEED must be exported by the submitter}"
: "${OUTPUT_NAME:?OUTPUT_NAME must be exported by the submitter}"

# shellcheck source=config/hpc/slurm_preamble.sh
source "${PRIDE_ROOT:-${SLURM_SUBMIT_DIR}}/config/hpc/slurm_preamble.sh"
pride_slurm_job_init train_PEBBLE.py

echo "method=PEBBLE env=quadruped_walk seed=${SEED} steps=1000000 feedback=2000"
echo "feed_type=0 output=outputs/${OUTPUT_NAME}_seed${SEED}"

python train_PEBBLE.py \
    env=quadruped_walk \
    seed="${SEED}" \
    "device=${PRIDE_DEVICE}" \
    agent.params.actor_lr=0.0001 \
    agent.params.critic_lr=0.0001 \
    gradient_update=1 \
    activation=tanh \
    num_unsup_steps=9000 \
    num_train_steps=1000000 \
    num_interact=30000 \
    max_feedback=2000 \
    reward_batch=200 \
    reward_update=50 \
    feed_type=0 \
    teacher_beta=-1 \
    teacher_gamma=1 \
    teacher_eps_mistake=0 \
    teacher_eps_skip=0 \
    teacher_eps_equal=0 \
    "hydra.run.dir=outputs/${OUTPUT_NAME}_seed${SEED}"
