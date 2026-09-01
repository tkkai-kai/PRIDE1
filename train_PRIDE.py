#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl
import tqdm

from logger import Logger
from replay_buffer import ReplayBuffer, DiffReplayBuffer
from reward_model import RewardModel, NoQueryableTrajectories
from collections import deque

import utils
import hydra

from diffusion.elucidated_diffusion import REDQTrainer
from diffusion.train_diffuser import SimpleDiffusionGenerator, ConditionalDiffusionGenerator
from diffusion.utils import construct_diffusion_model

class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        self.logger = Logger(
            self.work_dir,
            save_tb=cfg.log_save_tb,
            log_frequency=cfg.log_frequency,
            agent=cfg.agent.name)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.log_success = False
        self.last_eval_step = 0
        
        # make env
        if 'metaworld' in cfg.env:
            self.env = utils.make_metaworld_env(cfg)
            self.log_success = True
        else:
            self.env = utils.make_env(cfg)

        self.max_episode_steps = getattr(self.env, "_max_episode_steps", None)
        if self.max_episode_steps is None and getattr(self.env, "spec", None) is not None:
            self.max_episode_steps = getattr(self.env.spec, "max_episode_steps", None)
        if self.max_episode_steps is None:
            self.max_episode_steps = 1000
        
        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity),
            self.device)

        self.diffusion_buffer_size = int(1e6)
        self.diffusion_replay_buffer = DiffReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            self.diffusion_buffer_size,
            self.device)
        
        # for logging
        self.total_feedback = 0
        self.labeled_feedback = 0
        self.step = 0

        # synthetic transition snapshot analysis
        self.snapshot_steps = set(int(x) for x in cfg.snapshot_steps)
        self.snapshot_sample_size = int(cfg.snapshot_sample_size)
        self.snapshot_recent_window = int(cfg.snapshot_recent_window)
        self.snapshot_dir = os.path.join(self.work_dir, str(cfg.snapshot_dir))
        self.synthetic_sample_size = int(cfg.synthetic_sample_size)
        self.synthetic_dir = os.path.join(
            self.work_dir, "synthetic_transitions", str(self.cfg.env))
        self.analysis_dir = os.path.join(self.work_dir, "analysis", str(self.cfg.env))
        # records when the current synthetic buffer was generated
        self.last_diffusion_retrain_step = None

        # instantiating the reward model
        self.reward_model = RewardModel(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            ensemble_size=cfg.ensemble_size,
            size_segment=cfg.segment,
            activation=cfg.activation, 
            lr=cfg.reward_lr,
            mb_size=cfg.reward_batch, 
            large_batch=cfg.large_batch, 
            label_margin=cfg.label_margin, 
            teacher_beta=cfg.teacher_beta, 
            teacher_gamma=cfg.teacher_gamma, 
            teacher_eps_mistake=cfg.teacher_eps_mistake, 
            teacher_eps_skip=cfg.teacher_eps_skip, 
            teacher_eps_equal=cfg.teacher_eps_equal)
            
        
    def evaluate(self):
        average_episode_reward = 0
        average_true_episode_reward = 0
        success_rate = 0
        
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            done = False
            episode_reward = 0
            true_episode_reward = 0
            if self.log_success:
                episode_success = 0

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, extra = self.env.step(action)
                
                episode_reward += reward
                true_episode_reward += reward
                if self.log_success:
                    episode_success = max(episode_success, extra['success'])
                
            average_episode_reward += episode_reward
            average_true_episode_reward += true_episode_reward
            if self.log_success:
                success_rate += episode_success
            
        average_episode_reward /= self.cfg.num_eval_episodes
        average_true_episode_reward /= self.cfg.num_eval_episodes
        if self.log_success:
            success_rate /= self.cfg.num_eval_episodes
            success_rate *= 100.0
        
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.log('eval/true_episode_reward', average_true_episode_reward,
                        self.step)
        if self.log_success:
            self.logger.log('eval/success_rate', success_rate,
                    self.step)
            self.logger.log('train/true_episode_success', success_rate,
                        self.step)
        self.logger.dump(self.step)
    
    def learn_reward(self, first_flag=0):
                
        # get feedbacks
        labeled_queries, noisy_queries = 0, 0
        try:
            if first_flag == 1:
                # if it is first time to get feedback, need to use random sampling
                labeled_queries = self.reward_model.uniform_sampling()
            else:
                if self.cfg.feed_type == 0:
                    labeled_queries = self.reward_model.uniform_sampling()
                elif self.cfg.feed_type == 1:
                    labeled_queries = self.reward_model.disagreement_sampling()
                elif self.cfg.feed_type == 2:
                    labeled_queries = self.reward_model.entropy_sampling()
                elif self.cfg.feed_type == 3:
                    labeled_queries = self.reward_model.kcenter_sampling()
                elif self.cfg.feed_type == 4:
                    labeled_queries = self.reward_model.kcenter_disagree_sampling()
                elif self.cfg.feed_type == 5:
                    labeled_queries = self.reward_model.kcenter_entropy_sampling()
                else:
                    raise NotImplementedError
        except NoQueryableTrajectories as e:
            print("Skip reward update: %s" % e)
            return
        
        self.total_feedback += self.reward_model.mb_size
        self.labeled_feedback += labeled_queries
        
        train_acc = 0
        if self.labeled_feedback > 0:
            # update reward
            for epoch in range(self.cfg.reward_update):
                if self.cfg.label_margin > 0 or self.cfg.teacher_eps_equal > 0:
                    train_acc = self.reward_model.train_soft_reward()
                else:
                    train_acc = self.reward_model.train_reward()
                total_acc = np.mean(train_acc)
                
                if total_acc > 0.97:
                    break;
                    
        print("Reward function is updated!! ACC: " + str(total_acc))

    def _get_recent_buffer_indices(self, buffer, recent_window):
        """Indices of the most recently inserted transitions.

        buffer.idx points at the NEXT insertion slot, so the newest samples
        are the ones immediately before it, wrapping around capacity once
        the buffer is full.
        """
        buffer_size = len(buffer)
        if buffer_size == 0:
            return np.array([], dtype=np.int64)

        recent_window = min(int(recent_window), buffer_size)

        if not buffer.full:
            start = max(0, buffer.idx - recent_window)
            return np.arange(start, buffer.idx, dtype=np.int64)

        start = (buffer.idx - recent_window) % buffer.capacity
        if start < buffer.idx:
            return np.arange(start, buffer.idx, dtype=np.int64)

        return np.concatenate([
            np.arange(start, buffer.capacity, dtype=np.int64),
            np.arange(0, buffer.idx, dtype=np.int64),
        ])

    def _meta(self, value):
        return np.array([value], dtype=np.int64)

    def save_synthetic_transitions(self, retrain_step):
        syn_size = len(self.diffusion_replay_buffer)
        if syn_size == 0:
            print(f"[SYN] step={retrain_step}: synthetic buffer empty, skip.")
            return

        rng = np.random.default_rng(int(self.cfg.seed) + int(retrain_step) + 7654321)
        n = min(int(self.synthetic_sample_size), syn_size)
        indices = rng.choice(syn_size, size=n, replace=False)
        syn_obs = self.diffusion_replay_buffer.obses[indices].astype(np.float32)
        syn_actions = self.diffusion_replay_buffer.actions[indices].astype(np.float32)
        syn_rewards = self.diffusion_replay_buffer.rewards[indices].astype(np.float32)
        syn_next_obs = self.diffusion_replay_buffer.next_obses[indices].astype(np.float32)

        os.makedirs(self.synthetic_dir, exist_ok=True)
        save_path = os.path.join(
            self.synthetic_dir, f"synthetic_transitions_step_{retrain_step}.npz")

        np.savez_compressed(
            save_path,
            step=self._meta(retrain_step),
            seed=self._meta(int(self.cfg.seed)),
            env=np.array([str(self.cfg.env)]),
            synthetic_buffer_size=self._meta(syn_size),
            synthetic_obs=syn_obs,
            synthetic_actions=syn_actions,
            synthetic_rewards=syn_rewards,
            synthetic_next_obs=syn_next_obs,
        )
        print(f"[SYN] saved {n}/{syn_size} transitions: {save_path}")

    def save_transition_snapshot(self, snapshot_step):
        """Dump samples of real, recent-real and synthetic transitions to npz.

        1. random samples from full REAL replay buffer
        2. random samples from RECENT REAL transitions
        3. random samples from current SYNTHETIC replay buffer

        No plotting is done here.
        """
        real_size = len(self.replay_buffer)
        syn_size = len(self.diffusion_replay_buffer)

        if real_size == 0:
            print(f"[SNAPSHOT] step={snapshot_step}: real buffer empty, skip.")
            return

        if syn_size == 0:
            print(f"[SNAPSHOT] step={snapshot_step}: synthetic buffer empty, skip.")
            return

        # a private RNG keeps snapshot sampling from advancing the global numpy
        # stream, which would otherwise perturb the RL run
        rng = np.random.default_rng(int(self.cfg.seed) + int(snapshot_step) + 1234567)

        def take(buffer, indices):
            return tuple(
                getattr(buffer, name)[indices].astype(np.float32)
                for name in ('obses', 'actions', 'rewards', 'next_obses'))

        # full real replay buffer
        real_n = min(self.snapshot_sample_size, real_size)
        real_obs, real_actions, real_rewards, real_next_obs = take(
            self.replay_buffer, rng.choice(real_size, size=real_n, replace=False))

        # most recent real transitions
        recent_pool = self._get_recent_buffer_indices(
            self.replay_buffer, self.snapshot_recent_window)
        recent_n = min(self.snapshot_sample_size, len(recent_pool))
        if recent_n > 0:
            recent_indices = recent_pool[
                rng.choice(len(recent_pool), size=recent_n, replace=False)]
            (recent_real_obs, recent_real_actions,
                recent_real_rewards, recent_real_next_obs) = take(
                    self.replay_buffer, recent_indices)
        else:
            obs_dim = self.env.observation_space.shape[0]
            act_dim = self.env.action_space.shape[0]
            recent_real_obs = np.empty((0, obs_dim), dtype=np.float32)
            recent_real_actions = np.empty((0, act_dim), dtype=np.float32)
            recent_real_rewards = np.empty((0, 1), dtype=np.float32)
            recent_real_next_obs = np.empty((0, obs_dim), dtype=np.float32)

        # current synthetic replay buffer
        syn_n = min(self.snapshot_sample_size, syn_size)
        syn_obs, syn_actions, syn_rewards, syn_next_obs = take(
            self.diffusion_replay_buffer,
            rng.choice(syn_size, size=syn_n, replace=False))

        # -1 marks a snapshot taken before any diffusion model was trained
        last_retrain = self.last_diffusion_retrain_step
        synthetic_age = (
            -1 if last_retrain is None
            else int(snapshot_step) - int(last_retrain))

        os.makedirs(self.snapshot_dir, exist_ok=True)
        save_path = os.path.join(
            self.snapshot_dir, f"snapshot_step_{snapshot_step}.npz")


        np.savez_compressed(
            save_path,
            step=self._meta(snapshot_step),
            seed=self._meta(int(self.cfg.seed)),
            real_buffer_size=self._meta(real_size),
            synthetic_buffer_size=self._meta(syn_size),
            last_diffusion_retrain_step=self._meta(
                -1 if last_retrain is None else last_retrain),
            synthetic_age=self._meta(synthetic_age),
            retrain_diffusion_every=self._meta(int(self.cfg.retrain_diffusion_every)),
            real_obs=real_obs,
            real_actions=real_actions,
            real_rewards=real_rewards,
            real_next_obs=real_next_obs,
            recent_real_obs=recent_real_obs,
            recent_real_actions=recent_real_actions,
            recent_real_rewards=recent_real_rewards,
            recent_real_next_obs=recent_real_next_obs,
            syn_obs=syn_obs,
            syn_actions=syn_actions,
            syn_rewards=syn_rewards,
            syn_next_obs=syn_next_obs,
        )

        print(f"[SNAPSHOT] saved: {save_path}")
        print(f"[SNAPSHOT] step={snapshot_step} real={real_n}/{real_size} "
              f"recent={recent_n} synthetic={syn_n}/{syn_size} "
              f"synthetic_age={synthetic_age}")
    
    def reset_diffusion_buffer(self):
        self.diffusion_replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            self.diffusion_buffer_size,
            self.device)

    def analyze_diffusion_model(self, diffusion_trainer, retrain_step):
        """Condition on real (s, a) and dump s'_syn vs s'_real for offline MSE."""
        real_size = len(self.replay_buffer)
        if real_size == 0:
            print(f"[ANALYSIS] step={retrain_step}: real buffer empty, skip.")
            return

        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]
        D = obs_dim + act_dim + 1 + obs_dim
        if self.cfg.model_terminals:
            D += 1

        n = min(int(self.cfg.analysis_sample_size), real_size)
        rng = np.random.default_rng(int(self.cfg.seed) + int(retrain_step) + 2468013)
        indices = rng.choice(real_size, size=n, replace=False)
        real_obs = self.replay_buffer.obses[indices].astype(np.float32)
        real_actions = self.replay_buffer.actions[indices].astype(np.float32)
        real_next_obs = self.replay_buffer.next_obses[indices].astype(np.float32)

        known_mask = torch.zeros(D, dtype=torch.float32)
        known_mask[:obs_dim + act_dim] = 1.0
        known_values = torch.zeros((n, D), dtype=torch.float32)
        known_values[:, :obs_dim] = torch.from_numpy(real_obs)
        known_values[:, obs_dim:obs_dim + act_dim] = torch.from_numpy(real_actions)

        analysis_gen = ConditionalDiffusionGenerator(
            self.cfg, env=self.env, ema_model=diffusion_trainer.ema.ema_model)
        _, _, syn_next_obs = analysis_gen.conditional_sample(
            known_values=known_values,
            known_mask=known_mask,
        )

        os.makedirs(self.analysis_dir, exist_ok=True)
        save_path = os.path.join(
            self.analysis_dir, f"conditional_next_obs_step_{retrain_step}.npz")
        np.savez_compressed(
            save_path,
            step=self._meta(retrain_step),
            seed=self._meta(int(self.cfg.seed)),
            env=np.array([str(self.cfg.env)]),
            real_obs=real_obs,
            real_actions=real_actions,
            real_next_obs=real_next_obs,
            syn_next_obs=syn_next_obs,
        )
        print(f"[ANALYSIS] saved {n}/{real_size} pairs: {save_path}")

    def run(self):
        episode, episode_reward, done = 0, 0, True
        if self.log_success:
            episode_success = 0
        true_episode_reward = 0
        
        # store train returns of recent 10 episodes
        avg_train_true_return = deque([], maxlen=10) 
        start_time = time.time()
        fixed_start_time = time.time()

        interact_count = 0
        
        ################ diffusion model ##########################
        ###########################################################
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]
        # set up diffusion model
        diff_dims = obs_dim + act_dim + 1 + obs_dim
        if self.cfg.model_terminals:
            diff_dims += 1
        inputs = torch.zeros((128, diff_dims)).float()
        if self.cfg.skip_reward_norm:
            skip_dims = [obs_dim + act_dim]
        else:
            skip_dims = []
            
        retrain_diffusion_step = self.cfg.retrain_diffusion_every
        ###########################################################



        while self.step < self.cfg.num_train_steps:
            ################ diffusion model ##########################
            ###########################################################
            if (self.step + 1) % retrain_diffusion_step == 0 and (self.step + 1) >= self.cfg.diffusion_start and self.step + 1 < self.cfg.num_train_steps:
                print(f'Retraining diffusion model at step {self.step + 1}')

                # Train new diffusion model
                diffusion_trainer = REDQTrainer(
                    self.cfg,
                    construct_diffusion_model(
                        self.cfg,
                        inputs=inputs,
                        skip_dims=skip_dims,
                        disable_terminal_norm=self.cfg.model_terminals,  # No terminals in DMC(False), OpenAI(True)
                    ),
                    results_folder=self.work_dir,
                    model_terminals=self.cfg.model_terminals,
                )
                diffusion_trainer.update_normalizer(self.replay_buffer, device=self.device)
                diffusion_trainer.train_from_redq_buffer(self.replay_buffer)
                self.reset_diffusion_buffer()
                
                cpu_rng = torch.get_rng_state()
                cuda_rng = torch.cuda.get_rng_state_all()
                # analysis of diffusion model
                self.analyze_diffusion_model(diffusion_trainer, self.step + 1)

                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state_all(cuda_rng)
                
                # Add samples to agent replay buffer
                generator = SimpleDiffusionGenerator(self.cfg, env=self.env, ema_model=diffusion_trainer.ema.ema_model)
                observations, actions, rewards, next_observations, terminals = generator.sample(num_samples=self.cfg.num_samples)

                print(f'Adding {self.cfg.num_samples} samples to replay buffer.')
                for o, a, r, o2, term in zip(observations, actions, rewards, next_observations, terminals):
                    self.diffusion_replay_buffer.add(o, a, r, o2, term, term)
                
                if self.cfg.save_synthetic_transitions :
                    self.save_synthetic_transitions(self.step + 1)
                # record the last diffusion retrain step
                self.last_diffusion_retrain_step = self.step + 1

                if self.cfg.print_buffer_stats:
                    ptr_location = self.replay_buffer.idx
                    real_observations = self.replay_buffer.obses[:ptr_location]
                    real_actions = self.replay_buffer.actions[:ptr_location]
                    # real_next_observations = self.replay_buffer.next_obses[:ptr_location]
                    real_rewards = self.replay_buffer.rewards[:ptr_location]
                    # Print min, max, mean, std of each dimension in the obs, rew and action
                    print('Buffer stats:')
                    for i in range(observations.shape[1]):
                        print(f'Diffusion Obs {i}: {np.mean(observations[:, i]):.2f} {np.std(observations[:, i]):.2f}')
                        print(
                            f'     Real Obs {i}: {np.mean(real_observations[:, i]):.2f} {np.std(real_observations[:, i]):.2f}')
                    for i in range(actions.shape[1]):
                        print(f'Diffusion Action {i}: {np.mean(actions[:, i]):.2f} {np.std(actions[:, i]):.2f}')
                        print(
                            f'     Real Action {i}: {np.mean(real_actions[:, i]):.2f} {np.std(real_actions[:, i]):.2f}')
                    print(f'Diffusion Reward: {np.mean(rewards):.2f} {np.std(rewards):.2f}')
                    print(f'     Real Reward: {np.mean(real_rewards):.2f} {np.std(real_rewards):.2f}')
                    print(f'Replay buffer size: {ptr_location}')
                    print(f'Diffusion buffer size: {self.diffusion_replay_buffer.idx}')
                
                if self.cfg.diffusion_schedule:
                    retrain_diffusion_step = self.step + self.cfg.retrain_diffusion_every * self.cfg.num_train_steps / (self.cfg.num_train_steps-self.step +1)
            ###########################################################            
            if done:
                if self.step > 0:
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    self.logger.log('train/total_duration', time.time() - fixed_start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                # OpenAI gym tasks vs. DMC tasks                    
                if self.cfg.env in ["HalfCheetah-v2", "Walker2d-v2", "Hopper-v2"]:
                    if self.step > 0 and self.step - self.last_eval_step >= self.cfg.eval_openai_gym_frequency:
                        self.logger.log('eval/episode', episode, self.step)
                        self.evaluate()
                        self.last_eval_step = self.step
                else:
                    if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                        self.logger.log('eval/episode', episode, self.step)
                        self.evaluate()

                self.logger.log('train/episode_reward', episode_reward, self.step)
                self.logger.log('train/true_episode_reward', true_episode_reward, self.step)
                self.logger.log('train/total_feedback', self.total_feedback, self.step)
                self.logger.log('train/labeled_feedback', self.labeled_feedback, self.step)
                
                if self.log_success:
                    self.logger.log('train/episode_success', episode_success,
                        self.step)
                    self.logger.log('train/true_episode_success', episode_success,
                        self.step)
                
                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                avg_train_true_return.append(true_episode_reward)
                true_episode_reward = 0
                if self.log_success:
                    episode_success = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)
                        
            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update                
            if self.step == (self.cfg.num_seed_steps + self.cfg.num_unsup_steps):
                # update schedule
                if self.cfg.reward_schedule == 1:
                    frac = (self.cfg.num_train_steps-self.step) / self.cfg.num_train_steps
                    if frac == 0:
                        frac = 0.01
                elif self.cfg.reward_schedule == 2:
                    frac = self.cfg.num_train_steps / (self.cfg.num_train_steps-self.step +1)
                else:
                    frac = 1
                self.reward_model.change_batch(frac)
                
                # update margin --> not necessary / will be updated soon
                new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.max_episode_steps)
                self.reward_model.set_teacher_thres_skip(new_margin)
                self.reward_model.set_teacher_thres_equal(new_margin)
                
                # first learn reward
                self.learn_reward(first_flag=1)
                
                # relabel buffer    
                self.replay_buffer.relabel_with_predictor(self.reward_model)
                
                # reset Q due to unsuperivsed exploration
                self.agent.reset_critic()
                
                # update agent
                self.agent.update_after_reset(
                    self.replay_buffer, self.logger, self.step, 
                    gradient_update=self.cfg.reset_update, 
                    policy_update=True)
                
                # reset interact_count
                interact_count = 0
            elif self.step > self.cfg.num_seed_steps + self.cfg.num_unsup_steps:
                # update reward function
                if self.total_feedback < self.cfg.max_feedback:
                    if interact_count == self.cfg.num_interact:
                        # update schedule
                        if self.cfg.reward_schedule == 1:
                            frac = (self.cfg.num_train_steps-self.step) / self.cfg.num_train_steps
                            if frac == 0:
                                frac = 0.01
                        elif self.cfg.reward_schedule == 2:
                            frac = self.cfg.num_train_steps / (self.cfg.num_train_steps-self.step +1)
                        else:
                            frac = 1
                        self.reward_model.change_batch(frac)
                        
                        # update margin --> not necessary / will be updated soon
                        new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.max_episode_steps)
                        self.reward_model.set_teacher_thres_skip(new_margin * self.cfg.teacher_eps_skip)
                        self.reward_model.set_teacher_thres_equal(new_margin * self.cfg.teacher_eps_equal)
                        
                        # corner case: new total feed > max feed
                        if self.reward_model.mb_size + self.total_feedback > self.cfg.max_feedback:
                            self.reward_model.set_batch(self.cfg.max_feedback - self.total_feedback)
                            
                        self.learn_reward()
                        self.replay_buffer.relabel_with_predictor(self.reward_model)
                        interact_count = 0
                        
                self.agent.update(self.replay_buffer, self.logger, self.step, 1, True, self.diffusion_replay_buffer, self.cfg.diffusion_sample_ratio)
                
            # unsupervised exploration
            elif self.step > self.cfg.num_seed_steps:
                self.agent.update_state_ent(self.replay_buffer, self.logger, self.step, 
                                            gradient_update=1, K=self.cfg.topK)
                
            next_obs, reward, done, extra = self.env.step(action)
            reward_hat = self.reward_model.r_hat(np.concatenate([obs, action], axis=-1))

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.max_episode_steps else done
            episode_reward += reward_hat
            true_episode_reward += reward
            
            if self.log_success:
                episode_success = max(episode_success, extra['success'])
                
            # adding data to the reward training data
            self.reward_model.add_data(obs, action, reward, done)
            self.replay_buffer.add(
                obs, action, reward_hat, 
                next_obs, done, done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1
            interact_count += 1


            if self.cfg.save_transition_snapshots and self.step in self.snapshot_steps:
                self.save_transition_snapshot(self.step)
            
        # self.agent.save(self.work_dir, self.step)
        # self.reward_model.save(self.work_dir, self.step)
        
@hydra.main(config_path='config/train_PRIDE.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()

if __name__ == '__main__':
    main()
