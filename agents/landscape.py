"""Recent CF-label replay and a detached landscape-student ensemble for E2/E3.

The expensive teacher labels only a small subset of PPO rollout states.  In E2
that subset is uniform.  In E3 the same budget is allocated by ensemble
uncertainty, optionally multiplied by the policy-relevant leverage of the
student mean.  PPO consumes only detached predictions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from agents.networks import LandscapeNet


class LandscapeReplay:
    """Fixed-size FIFO of recent detached counterfactual labels."""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        if self.capacity < 1:
            raise ValueError("landscape replay capacity must be positive")
        self._obs = self._pi = self._target = None
        self._next = 0
        self.size = 0
        self.total_added = 0

    def add(self, obs: np.ndarray, pi: np.ndarray, target: np.ndarray) -> None:
        obs = np.asarray(obs, dtype=np.float32)
        pi = np.asarray(pi, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        if obs.ndim != 2 or pi.ndim != 2 or target.ndim != 2:
            raise ValueError("landscape labels must be rank-2 arrays")
        if not (len(obs) == len(pi) == len(target)):
            raise ValueError("landscape label arrays disagree on batch size")
        if pi.shape != target.shape:
            raise ValueError("landscape pi and target shapes must agree")
        if self._obs is None:
            self._obs = np.zeros((self.capacity, obs.shape[1]), dtype=np.float32)
            self._pi = np.zeros((self.capacity, pi.shape[1]), dtype=np.float32)
            self._target = np.zeros((self.capacity, target.shape[1]), dtype=np.float32)
        if obs.shape[1:] != self._obs.shape[1:] or pi.shape[1:] != self._pi.shape[1:]:
            raise ValueError("landscape label shape changed within one run")
        if len(obs) > self.capacity:
            obs, pi, target = obs[-self.capacity:], pi[-self.capacity:], target[-self.capacity:]
        n = len(obs)
        first = min(n, self.capacity - self._next)
        self._obs[self._next:self._next + first] = obs[:first]
        self._pi[self._next:self._next + first] = pi[:first]
        self._target[self._next:self._next + first] = target[:first]
        rest = n - first
        if rest:
            self._obs[:rest] = obs[first:]
            self._pi[:rest] = pi[first:]
            self._target[:rest] = target[first:]
        self._next = (self._next + n) % self.capacity
        self.size = min(self.capacity, self.size + n)
        self.total_added += n

    def sample(self, batch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.size == 0:
            raise RuntimeError("cannot sample an empty landscape replay")
        idx = rng.integers(0, self.size, size=int(batch_size))
        return self._obs[idx], self._pi[idx], self._target[idx]


@dataclass
class LandscapeTrainStats:
    loss: float = float("nan")
    n_steps: int = 0


class LandscapeEnsemble:
    """Independent bootstrap students for mean prediction and E3 uncertainty."""

    def __init__(self, *, obs_dim: int, n_actions: int, hidden_sizes, activation: str,
                 encoder: str, obs_shape, n_members: int, learning_rate: float,
                 device: str, seed: int):
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed + 37_211)
        # Separate initialisation is intentional: prediction variance becomes
        # E3's epistemic uncertainty without changing E2's code path.
        self.models = []
        self.optimizers = []
        for member in range(int(n_members)):
            # Do not disturb PPO's global sampling stream: with beta=0 the
            # distillation arm must reproduce a paired PPO run exactly.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + 101 * (member + 1))
                model = LandscapeNet(obs_dim, n_actions, hidden_sizes, activation,
                                     encoder, obs_shape).to(self.device)
            self.models.append(model)
            self.optimizers.append(torch.optim.Adam(model.parameters(), lr=learning_rate))

    @torch.no_grad()
    def predict_members(self, obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        preds = [m(x) for m in self.models]
        return torch.stack(preds, dim=0).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def predict(self, obs: np.ndarray) -> np.ndarray:
        return self.predict_members(obs).mean(axis=0).astype(np.float32)

    @torch.no_grad()
    def statistics(
        self, obs: np.ndarray, pi: np.ndarray, *, leverage_eps: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return centred mean, epistemic uncertainty, and leverage per state.

        The contraction with C_pi keeps both uncertainty and leverage in the
        action-space directions the PPO objective can actually move.
        """
        pi = np.asarray(pi, dtype=np.float32)
        members = self.predict_members(obs)
        centred = members - (pi[None, :, :] * members).sum(axis=2, keepdims=True)
        mean = centred.mean(axis=0)

        if centred.shape[0] <= 1:
            uncertainty = np.zeros(len(pi), dtype=np.float32)
        else:
            diff = centred - mean[None, :, :]
            denom = float(centred.shape[0] - 1)
            contracted = (
                (pi[None, :, :] * (diff * diff)).sum(axis=2)
                - (pi[None, :, :] * diff).sum(axis=2) ** 2
            ).sum(axis=0) / denom
            uncertainty = np.sqrt(np.maximum(contracted, 0.0)).astype(np.float32)

        lev = (
            (pi * (mean * mean)).sum(axis=1)
            - (pi * mean).sum(axis=1) ** 2
        )
        leverage = np.sqrt(np.maximum(lev, 0.0) + float(leverage_eps)).astype(np.float32)
        return mean.astype(np.float32), uncertainty, leverage

    def train(self, replay: LandscapeReplay, *, steps: int, batch_size: int) -> LandscapeTrainStats:
        if replay.size == 0 or steps <= 0:
            return LandscapeTrainStats()
        losses: list[float] = []
        for _ in range(int(steps)):
            # Each head receives an independently bootstrapped sample.
            for model, optimizer in zip(self.models, self.optimizers):
                obs, pi, target = replay.sample(batch_size, self.rng)
                x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                p = torch.as_tensor(pi, dtype=torch.float32, device=self.device)
                y = torch.as_tensor(target, dtype=torch.float32, device=self.device)
                err = model(x) - y
                # e^T C_pi e = sum_a pi_a e_a^2 - (sum_a pi_a e_a)^2.
                loss = ((p * err.square()).sum(-1) - (p * err).sum(-1).square()).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
        return LandscapeTrainStats(loss=float(np.mean(losses)), n_steps=len(losses))
