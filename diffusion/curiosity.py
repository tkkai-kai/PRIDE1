import torch
import torch.nn as nn
from typing import Dict, Tuple
import numpy as np

class CuriosityModel(nn.Module):
    def __init__(self, 
                obs_dim: int,
                action_dim: int,
                hidden_dim: int = 256,
    ) -> None:
        super(CuriosityModel, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # h(s): encode state into latent feature space.
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # g(s, a): encode state and action into latent feature space.
        self.forward_model = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # inverse model
        # predict action from h(s) and h(s').
        self.inverse_model = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
    def forward(self,
            obs: torch.Tensor,
            next_obs: torch.Tensor,
            action: torch.Tensor,
            ) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            next_feature:           (B, hidden_dim)
            predicted_next_feature: (B, hidden_dim)
            predicted_action:       (B, action_dim)
        """
        obs_feature = self.encoder(obs)
        next_feature = self.encoder(next_obs)

        forward_input = torch.cat([obs_feature, action], dim=-1)
        predicted_next_feature = self.forward_model(forward_input)
        
        # add inverse
        inverse_input = torch.cat([obs_feature, next_feature],dim=-1)
        predicted_action = self.inverse_model(inverse_input)

        return next_feature, predicted_next_feature, predicted_action

    def compute_loss(self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        forward_coef: float = 1.0,
        inverse_coef: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        next_feature, predicted_next_feature, predicted_action = self.forward(obs, next_obs, action)
        forward_loss = F.mse_loss(predicted_next_feature, next_feature.detach())
        inverse_loss = F.mse_loss(action, predicted_action)

        total_loss = (
            forward_coef * forward_loss
            + inverse_coef * inverse_loss
        )

        metrics = {
            "curiosity_total_loss": total_loss.detach(),
            "curiosity_forward_loss": forward_loss.detach(),
            "curiosity_inverse_loss": inverse_loss.detach(),
        }

        return total_loss, metrics

    @torch.no_grad()
    def compute_curiosity(self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action:torch.Tensor,
    ) -> torch.Tensor:
        next_feature, predicted_next_feature, _ = self.forward(obs, next_obs, action)
        curiosity = F.mse_loss(
            next_feature, 
            predicted_next_feature,
            reduction="none",
        ).mean(dim=-1,keepdim=True)
        return curiosity

def train_curiosity_from_buffer(
        curiosity_model: CuriosityModel,
        optimizer: torch.optim.Optimizer,
        replay_buffer,
        num_steps: int,
        batch_size: int,
        log_every: int=500,
) -> Dict[str, float]:
    """
    Train curiosity model only on the REAL replay buffer.
    """
    if len(replay_buffer) < batch_size:
        raise ValueError(f"Replay buffer has less than {batch_size} samples.")

    curiosity_model.train()

    total_loss = 0.0
    forward_loss = 0.0
    inverse_loss = 0.0

    for step in range(num_steps):
        (      
            obs, 
            action, 
            _reward, 
            next_obs, 
            _not_done, 
            _not_done_no_max,
        ) = replay_buffer.sample(batch_size)

        optimizer.zero_grad(set_to_none=True)

        loss, metrics = curiosity_model.compute_loss(
            obs=obs,
            next_obs=next_obs,
            action=action,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(curiosity_model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += metrics["curiosity_total_loss"].item()
        forward_loss += metrics["curiosity_forward_loss"].item()
        inverse_loss += metrics["curiosity_inverse_loss"].item()

        if step % log_every == 0 or step == num_steps - 1:
            print(
                f"[CURIOSITY] step={step}/{num_steps} "
                f"total={np.mean(total_loss[-log_every:]):.6f} "
                f"forward={np.mean(forward_loss[-log_every:]):.6f} "
                f"inverse={np.mean(inverse_loss[-log_every:]):.6f}"
            )

    return {
        "total_loss": float(np.mean(total_loss[-100:])),
        "forward_loss": float(np.mean(forward_loss[-100:])),
        "inverse_loss": float(np.mean(inverse_loss[-100:])),
    }

@torch.no_grad()
def label_curiosity_for_transitions(
    curiosity_model: CuriosityModel,
    obs: np.ndarray,
    next_obs: np.ndarray,
    action: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """
    Label curiosity for transitions.

    Returns:
        scores: (N, 1)
    """
    curiosity_model.eval()

    observations = np.asarray(obs).astype(np.float32)
    next_observations = np.asarray(next_obs).astype(np.float32)
    actions = np.asarray(action).astype(np.float32)

    if not (
            len(observations)
            == len(actions)
            == len(next_observations)
        ):
            raise ValueError("Transition arrays have inconsistent lengths.")

    scores = []

    for start_idx in range(0, len(observations), batch_size):
        end_idx = min(start_idx + batch_size, len(observations))

        obs_batch = torch.from_numpy(
            observations[start_idx:end_idx]
        ).to(device=device, dtype=torch.float32)

        next_obs_batch = torch.from_numpy(
            next_observations[start_idx:end_idx]
        ).to(device=device, dtype=torch.float32)

        action_batch = torch.from_numpy(
            actions[start_idx:end_idx]
        ).to(device=device, dtype=torch.float32)
        
        scores_batch = curiosity_model.compute_curiosity(
            obs=obs_batch,
            next_obs=next_obs_batch,
            action=action_batch,
        )

        scores.append(scores_batch.cpu().numpy())

    return np.concatenate(scores, axis=0)