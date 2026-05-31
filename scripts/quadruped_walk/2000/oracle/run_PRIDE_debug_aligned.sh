#!/bin/bash
# Aligned debug run:
# Keep parameter structure close to submit_PRIDE_syn_slurm.sh,
# but scale workload down for fast validation of terminal behavior.
#
# Usage:
#   bash scripts/quadruped_walk/2000/oracle/run_PRIDE_debug_aligned.sh 0
# Optional:
#   PRIDE_DEVICE=cuda bash .../run_PRIDE_debug_aligned.sh 0
# Note: terminal threshold only matters when model_terminals=true.

set -euo pipefail

: "${PRIDE_DEVICE:=cpu}"
: "${TERMINAL_THRESHOLD:=0.5}"
FEED_TYPE="${1:-0}"

for seed in 12345; do
    python train_PRIDE.py \
        env=quadruped_walk \
        seed="${seed}" \
        "device=${PRIDE_DEVICE}" \
        agent.params.actor_lr=0.0001 \
        agent.params.critic_lr=0.0001 \
        gradient_update=1 \
        activation=tanh \
        num_unsup_steps=9000 \
        num_train_steps=12000 \
        num_interact=400 \
        max_feedback=200 \
        reward_batch=64 \
        reward_update=10 \
        feed_type="${FEED_TYPE}" \
        teacher_beta=-1 \
        teacher_gamma=1 \
        teacher_eps_mistake=0 \
        teacher_eps_skip=0 \
        teacher_eps_equal=0 \
        retrain_diffusion_every=2000 \
        train_num_steps=2000 \
        num_samples=20000 \
        num_sample_steps=128 \
        print_buffer_stats=false \
        use_synthetic_reward_data=true \
        synthetic_reward_ratio=0.5 \
        synthetic_chunk_size=2000 \
        model_terminals=false \
        "terminal_threshold=${TERMINAL_THRESHOLD}"
done
