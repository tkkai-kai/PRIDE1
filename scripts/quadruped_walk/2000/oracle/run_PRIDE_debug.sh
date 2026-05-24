#!/bin/bash
# Short debug run. For CPU-only nodes: export PRIDE_DEVICE=cpu before running.
: "${PRIDE_DEVICE:=cpu}"
#for seed in 12345 23451 34512 45123 51234 67890 78906 89067 90678 6789; do
for seed in 12345; do
    python train_PRIDE.py env=quadruped_walk seed=$seed "device=${PRIDE_DEVICE}" agent.params.actor_lr=0.0001 agent.params.critic_lr=0.0001 gradient_update=1 activation=tanh segment=5 num_seed_steps=8 num_unsup_steps=0 num_train_steps=12 num_interact=6 max_feedback=20 reward_batch=8 reward_update=2 eval_frequency=1000000 num_eval_episodes=1 log_frequency=2 feed_type=$1 teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0 retrain_diffusion_every=6 train_num_steps=20 num_samples=2000 num_sample_steps=4 print_buffer_stats=false model_terminals=true
done