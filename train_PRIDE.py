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
from reward_model import RewardModel
from collections import deque

import utils
import hydra

from diffusion.elucidated_diffusion import REDQTrainer
from diffusion.train_diffuser import SimpleDiffusionGenerator
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
        self.device = utils.resolve_torch_device(cfg.device)
        self.log_success = False
        
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
            teacher_eps_equal=cfg.teacher_eps_equal,
            use_synthetic_reward_data=cfg.use_synthetic_reward_data,
            synthetic_reward_ratio=cfg.synthetic_reward_ratio,
            device=self.device)

    def _log_time(self, stage, start_time, **metrics):
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        metrics_str = " ".join([f"{k}={v}" for k, v in metrics.items()])
        suffix = f" | {metrics_str}" if metrics_str else ""
        print(f"[TIME][PRIDE] {stage} {elapsed_ms:.2f}ms{suffix}")
            
        
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
        learn_reward_start = time.perf_counter()
                
        # get feedbacks
        labeled_queries, noisy_queries = 0, 0
        if first_flag == 1:
            # if it is first time to get feedback, need to use random sampling
            sample_start = time.perf_counter()
            labeled_queries = self.reward_model.uniform_sampling()
            print(f"[TIME][PRIDE] learn_reward.sampling {((time.perf_counter() - sample_start) * 1000.0):.2f}ms | strategy=uniform_first")
        else:
            if self.cfg.feed_type == 0:
                sample_start = time.perf_counter()
                labeled_queries = self.reward_model.uniform_sampling()
                print(f"[TIME][PRIDE] learn_reward.sampling {((time.perf_counter() - sample_start) * 1000.0):.2f}ms | strategy=uniform")
            elif self.cfg.feed_type == 1:
                sample_start = time.perf_counter()
                labeled_queries = self.reward_model.disagreement_sampling()
                print(f"[TIME][PRIDE] learn_reward.sampling {((time.perf_counter() - sample_start) * 1000.0):.2f}ms | strategy=disagreement")
            elif self.cfg.feed_type == 2:
                sample_start = time.perf_counter()
                labeled_queries = self.reward_model.entropy_sampling()
                print(f"[TIME][PRIDE] learn_reward.sampling {((time.perf_counter() - sample_start) * 1000.0):.2f}ms | strategy=entropy")
            elif self.cfg.feed_type == 3:
                sample_start = time.perf_counter()
                labeled_queries = self.reward_model.kcenter_sampling()
                print(f"[TIME][PRIDE] learn_reward.sampling {((time.perf_counter() - sample_start) * 1000.0):.2f}ms | strategy=kcenter")
            elif self.cfg.feed_type == 4:
                sample_start = time.perf_counter()
                labeled_queries = self.reward_model.kcenter_disagree_sampling()
                print(f"[TIME][PRIDE] learn_reward.sampling {((time.perf_counter() - sample_start) * 1000.0):.2f}ms | strategy=kcenter_disagree")
            elif self.cfg.feed_type == 5:
                sample_start = time.perf_counter()
                labeled_queries = self.reward_model.kcenter_entropy_sampling()
                print(f"[TIME][PRIDE] learn_reward.sampling {((time.perf_counter() - sample_start) * 1000.0):.2f}ms | strategy=kcenter_entropy")
            else:
                raise NotImplementedError
        
        self.total_feedback += self.reward_model.mb_size
        self.labeled_feedback += labeled_queries
        
        train_acc = 0
        if self.labeled_feedback > 0:
            # update reward
            for epoch in range(self.cfg.reward_update):
                train_start = time.perf_counter()
                if self.cfg.label_margin > 0 or self.cfg.teacher_eps_equal > 0:
                    train_acc = self.reward_model.train_soft_reward()
                else:
                    train_acc = self.reward_model.train_reward()
                print(f"[TIME][PRIDE] learn_reward.train_epoch {((time.perf_counter() - train_start) * 1000.0):.2f}ms | epoch={epoch}")
                total_acc = np.mean(train_acc)
                
                if total_acc > 0.97:
                    break;
                    
        print("Reward function is updated!! ACC: " + str(total_acc))
        print(
            f"[TIME][PRIDE] learn_reward.total {((time.perf_counter() - learn_reward_start) * 1000.0):.2f}ms "
            f"| labeled_queries={labeled_queries} labeled_feedback={self.labeled_feedback}"
        )


    def reset_diffusion_buffer(self):
        self.diffusion_replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            self.diffusion_buffer_size,
            self.device)


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
        # Warm-start: persistent diffusion trainer reused across retrains.
        # None -> built (from scratch) on the first retrain.
        diffusion_trainer = None
        diffusion_warm_start = bool(getattr(self.cfg, "diffusion_warm_start", False))
        ###########################################################



        while self.step < self.cfg.num_train_steps:
            ################ diffusion model ##########################
            ###########################################################
            if (self.step + 1) % retrain_diffusion_step == 0 and (self.step + 1) >= self.cfg.diffusion_start and self.step + 1 < self.cfg.num_train_steps:
                diffusion_total_start = time.perf_counter()
                print(f'Retraining diffusion model at step {self.step + 1}')

                # Warm-start: reuse the existing trainer (fine-tune) once it exists;
                # otherwise build a fresh model and train from scratch (original behavior).
                warm_reuse = diffusion_warm_start and (diffusion_trainer is not None)
                diffusion_build_mode = "warm" if warm_reuse else "scratch"

                construct_start = time.perf_counter()
                if not warm_reuse:
                    diffusion_trainer = REDQTrainer(
                        self.cfg,
                        construct_diffusion_model(
                            self.cfg,
                            inputs=inputs,
                            skip_dims=skip_dims,
                            disable_terminal_norm=self.cfg.model_terminals,  # No terminals in DMC(False), OpenAI(True)
                        ),
                        # diffusion trainer arguments
                        train_batch_size=self.cfg.train_batch_size,
                        train_lr=self.cfg.train_lr,
                        lr_scheduler=self.cfg.lr_scheduler,
                        train_num_steps=self.cfg.train_num_steps,
                        save_and_sample_every=self.cfg.save_and_sample_every,
                        weight_decay=self.cfg.weight_decay,
                        results_folder=self.work_dir,
                        model_terminals=self.cfg.model_terminals,
                    )
                self._log_time(
                    "diffusion_retrain.construct",
                    construct_start,
                    train_num_steps=self.cfg.train_num_steps,
                    mode=diffusion_build_mode,
                )

                normalizer_start = time.perf_counter()
                diffusion_trainer.update_normalizer(self.replay_buffer, device=self.device)
                self._log_time("diffusion_retrain.update_normalizer", normalizer_start)

                train_start = time.perf_counter()
                if warm_reuse:
                    finetune_steps = int(getattr(self.cfg, "diffusion_finetune_steps", 5000))
                    finetune_lr = float(getattr(self.cfg, "diffusion_finetune_lr", 1e-4))
                    diffusion_trainer.set_constant_lr(finetune_lr)
                    diffusion_trainer.train_from_redq_buffer(self.replay_buffer, num_steps=finetune_steps)
                    train_steps_done = finetune_steps
                else:
                    diffusion_trainer.train_from_redq_buffer(self.replay_buffer)
                    train_steps_done = self.cfg.train_num_steps
                self._log_time(
                    "diffusion_retrain.train",
                    train_start,
                    train_num_steps=train_steps_done,
                    mode=diffusion_build_mode,
                )

                reset_start = time.perf_counter()
                self.reset_diffusion_buffer()
                self._log_time("diffusion_retrain.reset_buffer", reset_start)

                sample_start = time.perf_counter()
                generator = SimpleDiffusionGenerator(
                    self.cfg,
                    env=self.env,
                    ema_model=diffusion_trainer.ema.ema_model,
                    sample_batch_size=min(100000, int(self.cfg.num_samples)),
                )
                observations, actions, rewards, next_observations, terminals = generator.sample(num_samples=self.cfg.num_samples)
                self._log_time(
                    "diffusion_retrain.sample",
                    sample_start,
                    num_samples=self.cfg.num_samples,
                    num_sample_steps=self.cfg.num_sample_steps,
                )
                # print(f'Diffusion Observations: {observations.shape}, Actions: {actions.shape}, Rewards: {rewards.shape}, Next Observations: {next_observations.shape}, Terminals: {terminals.shape}')
                # print(f'Diffusion Observations: {observations[0]}, Actions: {actions[0]}, Rewards: {rewards[0]}, Next Observations: {next_observations[0]}, Terminals: {terminals[0]}')
                # Debug terminal behavior: check whether sampled terminals are continuous.
                terminals_float = np.asarray(terminals, dtype=np.float32).reshape(-1)
                terminal_threshold = float(getattr(self.cfg, "terminal_threshold", 0.5))
                continuous_mask = (terminals_float > 1e-6) & (terminals_float < 1.0 - 1e-6)
                done_binary = (terminals_float > terminal_threshold).astype(np.float32)
                print(
                    f"Diffusion Terminals stats: min={terminals_float.min():.4f}, "
                    f"max={terminals_float.max():.4f}, mean={terminals_float.mean():.4f}, "
                    f"std={terminals_float.std():.4f}"
                )
                print(
                    f"Diffusion Terminals debug: continuous_ratio={continuous_mask.mean():.4f}, "
                    f"binary_done_ratio@{terminal_threshold}={done_binary.mean():.4f}, "
                    f"sample={terminals_float[:10]}"
                )
                # add sample to reward model's self inputs and targets
                print(f'Adding {self.cfg.num_samples} samples to replay buffer.')
                integrate_start = time.perf_counter()
                synthetic_chunk_size = int(getattr(self.cfg, "synthetic_chunk_size", 2000))
                total_synthetic = len(terminals)
                for idx, (o, a, r, o2, term) in enumerate(zip(observations, actions, rewards, next_observations, terminals)):
                    # Match the per-step types used by add_data(obs, action, reward, done).
                    o = np.asarray(o, dtype=np.float32)
                    a = np.asarray(a, dtype=np.float32)
                    r = float(np.asarray(r).item())
                    done = float(np.asarray(term).item())
                    # Only store synthetic trajectories when enabled by config.
                    if self.cfg.use_synthetic_reward_data:
                        # When model_terminals is disabled, sampled terminals are all zero.
                        # Force-close synthetic trajectories every fixed chunk to avoid one
                        # unbounded trajectory growing forever in reward-model storage.
                        synthetic_done = done
                        if (not self.cfg.model_terminals) and synthetic_chunk_size > 0:
                            is_chunk_end = ((idx + 1) % synthetic_chunk_size) == 0
                            is_last = (idx + 1) == total_synthetic
                            synthetic_done = float(is_chunk_end or is_last)
                        self.reward_model.add_data(
                            o, a, r, synthetic_done, synthetic=True
                        )
                    # relabel difussion buffer with reward model
                    sa = np.concatenate([o, a], axis=-1)
                    r_hat = self.reward_model.r_hat(sa)
                    self.diffusion_replay_buffer.add(o, a, r_hat, o2, term, term)

                self._log_time(
                    "diffusion_retrain.integrate_samples",
                    integrate_start,
                    num_samples=total_synthetic,
                    use_synthetic=self.cfg.use_synthetic_reward_data,
                )
                self._log_time(
                    "diffusion_retrain.total",
                    diffusion_total_start,
                    step=self.step + 1,
                    retrain_every=retrain_diffusion_step,
                )

                # Synthetic-vs-real drift score (logged for warm-start calibration; no
                # auto-reset yet). Standardized mean gap + |log std ratio|, per group,
                # with reward weighted x2 since reward fidelity matters most for RL.
                def _gap(synth, real):
                    eps = 1e-6
                    synth = np.asarray(synth, dtype=np.float64).reshape(len(synth), -1)
                    real = np.asarray(real, dtype=np.float64).reshape(len(real), -1)
                    r_mean, r_std = real.mean(axis=0), real.std(axis=0)
                    s_mean, s_std = synth.mean(axis=0), synth.std(axis=0)
                    mean_gap = float(np.mean(np.abs(s_mean - r_mean) / (r_std + eps)))
                    std_gap = float(np.mean(np.abs(np.log((s_std + eps) / (r_std + eps)))))
                    return mean_gap, std_gap

                ptr_drift = self.replay_buffer.idx
                obs_mg, obs_sg = _gap(observations, self.replay_buffer.obses[:ptr_drift])
                act_mg, act_sg = _gap(actions, self.replay_buffer.actions[:ptr_drift])
                rew_mg, rew_sg = _gap(rewards, self.replay_buffer.rewards[:ptr_drift])
                drift_total = (obs_mg + obs_sg) + (act_mg + act_sg) + 2.0 * (rew_mg + rew_sg)
                print(
                    f"[DRIFT] step={self.step + 1} mode={diffusion_build_mode} "
                    f"total={drift_total:.4f} "
                    f"obs(mean={obs_mg:.3f},std={obs_sg:.3f}) "
                    f"act(mean={act_mg:.3f},std={act_sg:.3f}) "
                    f"rew(mean={rew_mg:.3f},std={rew_sg:.3f})"
                )

                if self.cfg.print_buffer_stats:
                    ptr_location = self.replay_buffer.idx
                    real_observations = self.replay_buffer.obses[:ptr_location]
                    real_actions = self.replay_buffer.actions[:ptr_location]
                    real_next_observations = self.replay_buffer.next_obses[:ptr_location]
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
                
                if self.diffusion_replay_buffer.idx > 0:
                    self.diffusion_replay_buffer.relabel_with_predictor(self.reward_model)

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
                        # relabel reward model's self inputs and targets
                        self.replay_buffer.relabel_with_predictor(self.reward_model)
                        # relabel difussion buffer with reward model
                        if self.diffusion_replay_buffer.idx > 0:
                            self.diffusion_replay_buffer.relabel_with_predictor(self.reward_model)
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
            
        # self.agent.save(self.work_dir, self.step)
        # self.reward_model.save(self.work_dir, self.step)
        
@hydra.main(config_path='config/train_PRIDE.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()

if __name__ == '__main__':
    main()
