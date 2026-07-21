#!/bin/bash
# PEBBLE baseline for quadruped_walk (oracle teacher).
# Params aligned with the recent PRIDE runs so results are directly comparable
# (PEBBLE has no diffusion, so diffusion-only args are omitted).
#
# Default: GPU. For CPU-only nodes: export PRIDE_DEVICE=cpu before running.
: "${PRIDE_DEVICE:=cuda}"
: "${FEED_TYPE:=${1:-0}}"
: "${OUTPUT_ROOT:=outputs}"
: "${NUM_TRAIN_STEPS:=1000000}"
: "${MAX_FEEDBACK:=1400}"
: "${SEED_LIST:=12345 23451 34512 45123 51234}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
OUTPUT_NAME="${OUTPUT_NAME:-pebble_${RUN_DATE}_qw_oracle_mf${MAX_FEEDBACK}}"
mkdir -p "${OUTPUT_ROOT}"

for seed in ${SEED_LIST}; do
    python train_PEBBLE.py \
        env=quadruped_walk \
        seed="${seed}" \
        "device=${PRIDE_DEVICE}" \
        agent.params.actor_lr=0.0001 \
        agent.params.critic_lr=0.0001 \
        gradient_update=1 \
        activation=tanh \
        num_unsup_steps=9000 \
        "num_train_steps=${NUM_TRAIN_STEPS}" \
        num_interact=30000 \
        "max_feedback=${MAX_FEEDBACK}" \
        reward_batch=200 \
        reward_update=50 \
        "feed_type=${FEED_TYPE}" \
        teacher_beta=-1 \
        teacher_gamma=1 \
        teacher_eps_mistake=0 \
        teacher_eps_skip=0 \
        teacher_eps_equal=0 \
        "hydra.run.dir=${OUTPUT_ROOT}/${OUTPUT_NAME}_seed${seed}"
done
