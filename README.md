# PRIDE

This project is implemented based on [B-Pref](https://github.com/rll-research/BPref)


## Install

```
conda env create -f conda_env.yml
pip install -e .[docs,tests,extra]
pip install dm_control
cd custom_dmc2gym
pip install -e .
pip install git+https://github.com/rlworkgroup/metaworld.git@v2.0.0#egg=metaworld
pip install pybullet
pip install torchdiffeq
```

## Run experiments on irrational teacher

To design more realistic models of human teachers, we consider a common stochastic model and systematically manipulate its terms and operators:

```
teacher_beta: rationality constant of stochastic preference model (default: -1 for perfectly rational model)
teacher_gamma: discount factor to model myopic behavior (default: 1)
teacher_eps_mistake: probability of making a mistake (default: 0)
teacher_eps_skip: hyperparameters to control skip threshold (\in [0,1])
teacher_eps_equal: hyperparameters to control equal threshold (\in [0,1])
```

In B-Pref, we tried the following teachers:

`Oracle teacher`: (teacher_beta=-1, teacher_gamma=1, teacher_eps_mistake=0, teacher_eps_skip=0, teacher_eps_equal=0)

`Mistake teacher`: (teacher_beta=-1, teacher_gamma=1, teacher_eps_mistake=0.1, teacher_eps_skip=0, teacher_eps_equal=0)

`Noisy teacher`: (teacher_beta=1, teacher_gamma=1, teacher_eps_mistake=0, teacher_eps_skip=0, teacher_eps_equal=0)

`Skip teacher`: (teacher_beta=-1, teacher_gamma=1, teacher_eps_mistake=0, teacher_eps_skip=0.1, teacher_eps_equal=0)

`Myopic teacher`: (teacher_beta=-1, teacher_gamma=0.9, teacher_eps_mistake=0, teacher_eps_skip=0, teacher_eps_equal=0)

`Equal teacher`: (teacher_beta=-1, teacher_gamma=1, teacher_eps_mistake=0, teacher_eps_skip=0, teacher_eps_equal=0.1)


### PEBBLE

Experiments can be reproduced with the following:

```
./scripts/[env_name]/[max_budget]/[teacher_type]/run_PEBBLE.sh [sampling_scheme: 0=uniform, 1=disagreement, 2=entropy]
```

### PRIDE

Experiments can be reproduced with the following:

```
./scripts/[env_name]/[max_budget]/[teacher_type]/run_PRIDE.sh [sampling_scheme: 0=uniform, 1=disagreement, 2=entropy]

# for example
./scripts/quadruped_walk/2000/oracle/run_PRIDE.sh 0
```

Example script code can be found under `scripts/quadruped_walk/2000/oracle/run_PRIDE.sh`.

The `retrain_diffusion_every` hyperparameter is specifically used to set the interval (in steps) at which the diffusion model is retrained.
