"""Rollout buffer + GAE.

Two details that matter for this project specifically:

1. TERMINATION vs TRUNCATION are kept separate. A TimeLimit cutoff is not an MDP
   terminal state, so the value target must bootstrap through it. Collapsing the
   two into a single `done` flag makes V(s) systematically wrong near the time
   limit -- and NB02's oracle Q_CF(s,a) = r + gamma*V(s') is built directly on V.

2. `advantage_transform` is the single hook Notebook 06 will use. It receives the
   full rollout and returns a modified advantage array. Notebook 01 leaves it
   None. Nothing else in the training loop needs to change for CF-PPO.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch


class RolloutBuffer:
    """Arrays are (n_steps, n_envs, ...)."""

    def __init__(self, n_steps: int, n_envs: int, obs_dim: int, n_actions: int, device: str = "cpu"):
        self.n_steps, self.n_envs = n_steps, n_envs
        self.obs_dim, self.n_actions = obs_dim, n_actions
        self.device = device
        self.reset()

    def reset(self) -> None:
        T, N = self.n_steps, self.n_envs
        self.obs = np.zeros((T, N, self.obs_dim), dtype=np.float32)        # scaled
        self.raw_obs = np.zeros((T, N, self.obs_dim), dtype=np.float32)    # unscaled
        self.sim_state = np.zeros((T, N, self.obs_dim), dtype=np.float64)
        self.next_raw_obs = np.zeros((T, N, self.obs_dim), dtype=np.float32)
        self.next_sim_state = np.zeros((T, N, self.obs_dim), dtype=np.float64)
        self.actions = np.zeros((T, N), dtype=np.int64)
        self.logprobs = np.zeros((T, N), dtype=np.float32)
        self.probs = np.zeros((T, N, self.n_actions), dtype=np.float32)
        self.rewards = np.zeros((T, N), dtype=np.float32)
        self.values = np.zeros((T, N), dtype=np.float32)
        self.terminated = np.zeros((T, N), dtype=bool)
        self.truncated = np.zeros((T, N), dtype=bool)
        self.episode_id = np.zeros((T, N), dtype=np.int64)
        self.episode_t = np.zeros((T, N), dtype=np.int64)
        self.ptr = 0

    def add(self, **kw) -> None:
        t = self.ptr
        for k, v in kw.items():
            getattr(self, k)[t] = v
        self.ptr += 1

    # -- GAE ---------------------------------------------------------------- #

    def compute_gae(
        self,
        last_values: np.ndarray,
        last_terminated: np.ndarray,
        next_values_at_boundary: np.ndarray,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (advantages, returns), both (n_steps, n_envs).

        `next_values_at_boundary[t, i]` is V(next_obs) for steps where the episode
        ended, i.e. the value of the TRUE successor state. It is used only when
        `truncated` is True (bootstrap through the time limit) and is ignored when
        `terminated` is True (V of a terminal state is 0 by definition).
        """
        T, N = self.n_steps, self.n_envs
        adv = np.zeros((T, N), dtype=np.float32)
        last_gae = np.zeros(N, dtype=np.float32)

        for t in reversed(range(T)):
            term = self.terminated[t]
            trunc = self.truncated[t]
            ended = term | trunc

            if t == T - 1:
                next_value_cont = last_values
                next_nonterminal_cont = (~last_terminated).astype(np.float32)
            else:
                next_value_cont = self.values[t + 1]
                next_nonterminal_cont = np.ones(N, dtype=np.float32)

            # Value of the state we transition INTO.
            next_value = np.where(ended, next_values_at_boundary[t], next_value_cont)
            # Zero it out for true MDP terminals.
            next_value = np.where(term, 0.0, next_value)

            delta = self.rewards[t] + gamma * next_value - self.values[t]

            # The GAE recursion must not carry across an episode boundary of any
            # kind, because the trajectory after the boundary belongs to a
            # different episode.
            cont = np.where(ended, 0.0, next_nonterminal_cont)
            last_gae = delta + gamma * gae_lambda * cont * last_gae
            adv[t] = last_gae

        returns = adv + self.values
        return adv, returns

    # -- flattening --------------------------------------------------------- #

    def flat_tensors(self, advantages: np.ndarray, returns: np.ndarray) -> dict[str, torch.Tensor]:
        d = self.device
        f = lambda a, dt: torch.as_tensor(a.reshape((-1,) + a.shape[2:]), dtype=dt, device=d)
        return {
            "obs": f(self.obs, torch.float32),
            "actions": f(self.actions, torch.int64),
            "logprobs": f(self.logprobs, torch.float32),
            "values": f(self.values, torch.float32),
            "advantages": f(advantages, torch.float32),
            "returns": f(returns, torch.float32),
        }


AdvantageTransform = Callable[["RolloutBuffer", np.ndarray], np.ndarray]
"""Signature used by Notebook 06:  (buffer, advantages) -> advantages."""
