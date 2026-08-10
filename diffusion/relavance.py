from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import torch

from diffusion.curiosity import (
    CuriosityModel,
    score_transition_arrays,
)


ConditionMode = Literal["top", "bottom", "random"]


@dataclass
class CuriosityRelevance:
    """
    Store one curiosity relevance label for every REAL transition.
    """

    raw_scores: np.ndarray          # (N, 1)
    normalized_scores: np.ndarray   # (N, 1)

    score_mean: float
    score_std: float

    top_indices: np.ndarray
    bottom_indices: np.ndarray

    buffer_size: int
    top_frac: float

    @classmethod
    def build(
        cls,
        replay_buffer,
        curiosity_model: CuriosityModel,
        device: torch.device,
        top_frac: float = 0.25,
        score_batch_size: int = 8192,
    ) -> "CuriosityRelevance":
        if not 0.0 < top_frac <= 1.0:
            raise ValueError(
                f"top_frac must be in (0, 1], got {top_frac}."
            )

        buffer_size = len(replay_buffer)

        if buffer_size <= 0:
            raise ValueError("Replay buffer is empty.")

        # The first buffer_size entries are valid.
        # Order is not important for this offline training.
        observations = replay_buffer.obses[:buffer_size]
        actions = replay_buffer.actions[:buffer_size]
        next_observations = replay_buffer.next_obses[:buffer_size]

        raw_scores = score_transition_arrays(
            model=curiosity_model,
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            device=device,
            batch_size=score_batch_size,
        ).astype(np.float32)

        score_mean = float(raw_scores.mean())
        score_std = float(raw_scores.std())

        # Standardise condition so the diffusion network receives a
        # numerically well-scaled scalar.
        normalized_scores = (
            raw_scores - score_mean
        ) / (score_std + 1e-6)

        flat_scores = raw_scores.reshape(-1)
        sorted_indices = np.argsort(flat_scores)

        num_selected = max(
            1,
            int(round(buffer_size * top_frac)),
        )

        bottom_indices = sorted_indices[:num_selected]
        top_indices = sorted_indices[-num_selected:]

        print(
            f"[RELEVANCE] buffer_size={buffer_size} "
            f"top_frac={top_frac:.3f} "
            f"num_selected={num_selected}"
        )
        print(
            f"[RELEVANCE] "
            f"mean={score_mean:.6f} "
            f"std={score_std:.6f} "
            f"min={raw_scores.min():.6f} "
            f"median={np.median(raw_scores):.6f} "
            f"max={raw_scores.max():.6f}"
        )
        print(
            f"[RELEVANCE] "
            f"bottom_mean={raw_scores[bottom_indices].mean():.6f} "
            f"top_mean={raw_scores[top_indices].mean():.6f}"
        )

        return cls(
            raw_scores=raw_scores,
            normalized_scores=normalized_scores.astype(np.float32),
            score_mean=score_mean,
            score_std=score_std,
            top_indices=top_indices,
            bottom_indices=bottom_indices,
            buffer_size=buffer_size,
            top_frac=top_frac,
        )

    def sample_training_batch(
        self,
        replay_buffer,
        batch_size: int,
        model_terminals: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Uniformly sample the FULL replay buffer.

        Returns:
            transition_data: (B, transition_dim)
            condition:       (B, 1)
            indices:         (B,)
        """
        indices = np.random.randint(
            0,
            self.buffer_size,
            size=batch_size,
        )

        obs = replay_buffer.obses[indices].astype(np.float32)
        actions = replay_buffer.actions[indices].astype(np.float32)
        rewards = replay_buffer.rewards[indices].astype(np.float32)
        next_obs = replay_buffer.next_obses[indices].astype(np.float32)

        transition_parts = [
            obs,
            actions,
            rewards,
            next_obs,
        ]

        if model_terminals:
            terminals = (
                1.0 - replay_buffer.not_dones_no_max[indices]
            ).astype(np.float32)
            transition_parts.append(terminals)

        transition_data = np.concatenate(
            transition_parts,
            axis=-1,
        ).astype(np.float32)

        condition = self.normalized_scores[indices]

        return transition_data, condition, indices

    def sample_conditions(
        self,
        batch_size: int,
        mode: ConditionMode,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            normalized condition: (B, 1)
            raw condition:        (B, 1)
            source indices:       (B,)
        """
        if mode == "top":
            candidate_indices = self.top_indices
        elif mode == "bottom":
            candidate_indices = self.bottom_indices
        elif mode == "random":
            candidate_indices = np.arange(self.buffer_size)
        else:
            raise ValueError(f"Unknown condition mode: {mode}")

        sampled_indices = np.random.choice(
            candidate_indices,
            size=batch_size,
            replace=True,
        )

        return (
            self.normalized_scores[sampled_indices],
            self.raw_scores[sampled_indices],
            sampled_indices,
        )