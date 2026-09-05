"""Actor-critic network.

Separate trunks for policy and value. Shared trunks save a little compute but
couple the value error into the policy gradient, and Notebooks 02 and 06 both
reason about V and pi as separate objects -- keeping them separate keeps that
reasoning honest.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

_ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def _mlp(in_dim: int, hidden: Sequence[int], out_dim: int, activation: str, out_std: float) -> nn.Sequential:
    act = _ACTIVATIONS[activation]
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [layer_init(nn.Linear(prev, h)), act()]
        prev = h
    layers += [layer_init(nn.Linear(prev, out_dim), std=out_std)]
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: str = "tanh",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_sizes = tuple(hidden_sizes)
        self.activation = activation

        # out_std=0.01 on the policy head -> near-uniform initial policy.
        self.actor = _mlp(obs_dim, hidden_sizes, n_actions, activation, out_std=0.01)
        self.critic = _mlp(obs_dim, hidden_sizes, 1, activation, out_std=1.0)

    # -- basic heads -------------------------------------------------------- #

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x).squeeze(-1)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor(x)

    def distribution(self, x: torch.Tensor) -> Categorical:
        return Categorical(logits=self.actor(x))

    # -- rollout / update interfaces ---------------------------------------- #

    @torch.no_grad()
    def act(self, x: torch.Tensor):
        """Sample an action. Returns (action, logprob, value, probs)."""
        logits = self.actor(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), self.value(x), dist.probs

    def evaluate_actions(self, x: torch.Tensor, actions: torch.Tensor):
        """Returns (logprob, entropy, value) with gradients attached."""
        dist = Categorical(logits=self.actor(x))
        return dist.log_prob(actions), dist.entropy(), self.value(x)

    # -- convenience for NB02-06 -------------------------------------------- #

    @torch.no_grad()
    def action_probs(self, x: torch.Tensor) -> torch.Tensor:
        return Categorical(logits=self.actor(x)).probs

    def config_dict(self) -> dict:
        return {
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "hidden_sizes": list(self.hidden_sizes),
            "activation": self.activation,
        }

    @classmethod
    def from_config_dict(cls, d: dict) -> "ActorCritic":
        return cls(d["obs_dim"], d["n_actions"], d["hidden_sizes"], d["activation"])
