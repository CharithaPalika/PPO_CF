"""Observation scaling.

The whole pipeline needs ONE unambiguous map from an environment state to the
tensor the policy network consumes. Notebook 02 restores simulator states and
queries the policy; Notebook 05 learns B(s); Notebook 06 evaluates B(s) inside
the PPO update. If that map drifts during training (as running mean/std does),
"the policy at the 30% checkpoint" stops being a well-defined object.

`FixedScaler` is therefore the default: a constant affine map derived from the
declared observation-space bounds. `RunningScaler` is provided for comparison
but its statistics must be checkpointed and restored, and NB02+ must use the
frozen copy from the checkpoint.
"""

from __future__ import annotations

import numpy as np


class BaseScaler:
    kind: str = "base"

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def update(self, obs: np.ndarray) -> None:
        """Optional: consume a batch of raw observations. No-op for stateless scalers."""
        return None

    def state_dict(self) -> dict:
        return {"kind": self.kind}

    def load_state_dict(self, d: dict) -> None:
        return None


class IdentityScaler(BaseScaler):
    kind = "none"

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32)


class FixedScaler(BaseScaler):
    """Affine map from [low, high] onto [-1, 1], componentwise.

    Requires finite bounds. MountainCar-v0:
        position in [-1.2, 0.6], velocity in [-0.07, 0.07].
    """

    kind = "fixed"

    def __init__(self, low: np.ndarray, high: np.ndarray):
        low = np.asarray(low, dtype=np.float64)
        high = np.asarray(high, dtype=np.float64)
        if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
            raise ValueError(
                "FixedScaler needs a bounded observation space; got "
                f"low={low}, high={high}. Use obs_norm='running' or 'none'."
            )
        self.low = low
        self.high = high
        self.center = 0.5 * (high + low)
        self.half_range = 0.5 * (high - low)
        self.half_range[self.half_range == 0.0] = 1.0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64)
        return ((obs - self.center) / self.half_range).astype(np.float32)

    def inverse(self, scaled: np.ndarray) -> np.ndarray:
        scaled = np.asarray(scaled, dtype=np.float64)
        return (scaled * self.half_range + self.center).astype(np.float32)

    def state_dict(self) -> dict:
        return {"kind": self.kind, "low": self.low, "high": self.high}

    def load_state_dict(self, d: dict) -> None:
        self.__init__(d["low"], d["high"])


class RunningScaler(BaseScaler):
    """Welford running mean/std, VecNormalize style."""

    kind = "running"

    def __init__(self, shape, epsilon: float = 1e-8, clip: float = 10.0):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4
        self.epsilon = epsilon
        self.clip = clip

    def update(self, obs: np.ndarray) -> None:
        obs = np.atleast_2d(np.asarray(obs, dtype=np.float64))
        batch_mean = obs.mean(axis=0)
        batch_var = obs.var(axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / tot
        self.mean, self.var, self.count = new_mean, m2 / tot, tot

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64)
        z = (obs - self.mean) / np.sqrt(self.var + self.epsilon)
        return np.clip(z, -self.clip, self.clip).astype(np.float32)

    def state_dict(self) -> dict:
        return {
            "kind": self.kind,
            "mean": self.mean,
            "var": self.var,
            "count": self.count,
            "epsilon": self.epsilon,
            "clip": self.clip,
        }

    def load_state_dict(self, d: dict) -> None:
        self.mean = np.asarray(d["mean"], dtype=np.float64)
        self.var = np.asarray(d["var"], dtype=np.float64)
        self.count = float(d["count"])
        self.epsilon = float(d["epsilon"])
        self.clip = float(d["clip"])


def make_scaler(kind: str, observation_space) -> BaseScaler:
    if kind == "fixed":
        return FixedScaler(observation_space.low, observation_space.high)
    if kind == "running":
        return RunningScaler(observation_space.shape)
    if kind == "none":
        return IdentityScaler()
    raise ValueError(f"unknown obs_norm: {kind!r}")


def scaler_from_state_dict(d: dict, observation_space=None) -> BaseScaler:
    kind = d["kind"]
    if kind == "fixed":
        return FixedScaler(d["low"], d["high"])
    if kind == "running":
        s = RunningScaler(np.asarray(d["mean"]).shape)
        s.load_state_dict(d)
        return s
    if kind == "none":
        return IdentityScaler()
    raise ValueError(f"unknown scaler kind: {kind!r}")
