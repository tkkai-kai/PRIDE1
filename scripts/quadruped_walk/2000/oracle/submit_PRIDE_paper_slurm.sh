#!/usr/bin/env bash
# One strict paper-release PRIDE seed. Submit through submit_paper_baseline.sh.
#SBATCH --job-name=pride_qw_paper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

set -euo pipefail

: "${SEED:?SEED must be exported by the submitter}"
: "${OUTPUT_NAME:?OUTPUT_NAME must be exported by the submitter}"
RETRAIN_DIFFUSION_EVERY="${RETRAIN_DIFFUSION_EVERY:-}"

# shellcheck source=config/hpc/slurm_preamble.sh
source "${PRIDE_ROOT:-${SLURM_SUBMIT_DIR}}/config/hpc/slurm_preamble.sh"
pride_slurm_job_init train_PRIDE.py

echo "method=PRIDE env=quadruped_walk seed=${SEED} steps=1000000 feedback=2000"
if [[ -n "${RETRAIN_DIFFUSION_EVERY}" ]]; then
    echo "feed_type=0 retrain_diffusion_every=${RETRAIN_DIFFUSION_EVERY} (explicit override)"
else
    echo "feed_type=0 retrain_diffusion_every=10000 diffusion_sample_ratio=0.5 (YAML defaults)"
fi
echo "output=outputs/${OUTPUT_NAME}_seed${SEED}"

train_args=(
    env=quadruped_walk
    "seed=${SEED}"
    "device=${PRIDE_DEVICE}"
    agent.params.actor_lr=0.0001
    agent.params.critic_lr=0.0001
    gradient_update=1
    activation=tanh
    num_unsup_steps=9000
    num_train_steps=1000000
    num_interact=30000
    max_feedback=2000
    reward_batch=200
    reward_update=50
    feed_type=0
    teacher_beta=-1
    teacher_gamma=1
    teacher_eps_mistake=0
    teacher_eps_skip=0
    teacher_eps_equal=0
)
if [[ -n "${RETRAIN_DIFFUSION_EVERY}" ]]; then
    train_args+=("retrain_diffusion_every=${RETRAIN_DIFFUSION_EVERY}")
fi
train_args+=("hydra.run.dir=outputs/${OUTPUT_NAME}_seed${SEED}")

python train_PRIDE.py "${train_args[@]}"
if false; then
#!/usr/bin/env bash
# One strict paper-release PRIDE seed. Submit through submit_paper_baseline.sh.
#SBATCH --job-name=pride_qw_paper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

set -euo pipefail

: "${SEED:?SEED must be exported by the submitter}"
: "${OUTPUT_NAME:?OUTPUT_NAME must be exported by the submitter}"
RETRAIN_DIFFUSION_EVERY="${RETRAIN_DIFFUSION_EVERY:-}"

# shellcheck source=config/hpc/slurm_preamble.sh
source "${PRIDE_ROOT:-${SLURM_SUBMIT_DIR}}/config/hpc/slurm_preamble.sh"
pride_slurm_job_init train_PRIDE.py

echo "method=PRIDE env=quadruped_walk seed=${SEED} steps=1000000 feedback=2000"
if [[ -n "${RETRAIN_DIFFUSION_EVERY}" ]]; then
    echo "feed_type=0 retrain_diffusion_every=${RETRAIN_DIFFUSION_EVERY} (explicit override)"
else
    echo "feed_type=0 retrain_diffusion_every=10000 diffusion_sample_ratio=0.5 (YAML defaults)"
fi
echo "output=outputs/${OUTPUT_NAME}_seed${SEED}"

train_args=(
    env=quadruped_walk
    "seed=${SEED}"
    "device=${PRIDE_DEVICE}"
    agent.params.actor_lr=0.0001
    agent.params.critic_lr=0.0001
    gradient_update=1
    activation=tanh
    num_unsup_steps=9000
    num_train_steps=1000000
    num_interact=30000
    max_feedback=2000
    reward_batch=200
    reward_update=50
    feed_type=0
    teacher_beta=-1
    teacher_gamma=1
    teacher_eps_mistake=0
    teacher_eps_skip=0
    teacher_eps_equal=0
)
if [[ -n "${RETRAIN_DIFFUSION_EVERY}" ]]; then
    train_args+=("retrain_diffusion_every=${RETRAIN_DIFFUSION_EVERY}")
fi
train_args+=("hydra.run.dir=outputs/${OUTPUT_NAME}_seed${SEED}")

python train_PRIDE.py "${train_args[@]}"
fi
