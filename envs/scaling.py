"""Observation scaling: the map from an environment state to a network input.

The whole pipeline needs ONE unambiguous version of this map. Notebook 02
restores simulator states and queries the policy; NB05 learns B(s); NB06
evaluates B(s) inside the PPO update. If the map drifts during training (as
running mean/std does), "the policy at the 30% checkpoint" stops being a
well-defined object.

Two shapes of observation are supported:

  Box(d)        -> FixedScaler (default): constant affine map from the declared
                   bounds onto [-1, 1]. MountainCar-v0's velocity has range
                   +/-0.07, which a tanh MLP effectively cannot see unscaled.
  Discrete(n)   -> OneHotScaler: the integer state becomes an n-dim indicator.
                   Taxi-v4 is Discrete(500).

Every scaler is stateless unless you explicitly ask for `running`, and every
scaler's state is written into each checkpoint, so NB02+ reconstruct the exact
map used during training.
"""

from __future__ import annotations

import numpy as np


class BaseScaler:
    kind: str = "base"
    #: dimension of the network input this scaler produces
    out_dim: int = 0
    #: dimension of the raw observation vector it consumes
    in_dim: int = 0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def update(self, obs: np.ndarray) -> None:
        """Consume a batch of raw observations. No-op for stateless scalers."""
        return None

    def state_dict(self) -> dict:
        return {"kind": self.kind}

    def load_state_dict(self, d: dict) -> None:
        return None


class IdentityScaler(BaseScaler):
    kind = "none"

    def __init__(self, dim: int):
        self.in_dim = self.out_dim = int(dim)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32).reshape(-1, self.in_dim)

    def state_dict(self) -> dict:
        return {"kind": self.kind, "dim": self.in_dim}


class FixedScaler(BaseScaler):
    """Affine map from [low, high] onto [-1, 1], componentwise. Requires finite bounds."""

    kind = "fixed"

    def __init__(self, low: np.ndarray, high: np.ndarray):
        low = np.asarray(low, dtype=np.float64).ravel()
        high = np.asarray(high, dtype=np.float64).ravel()
        if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
            raise ValueError(
                "FixedScaler needs a bounded observation space; got "
                f"low={low}, high={high}. Use obs_norm='running' or 'none'."
            )
        self.low, self.high = low, high
        self.center = 0.5 * (high + low)
        self.half_range = 0.5 * (high - low)
        self.half_range[self.half_range == 0.0] = 1.0
        self.in_dim = self.out_dim = len(low)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64).reshape(-1, self.in_dim)
        return ((obs - self.center) / self.half_range).astype(np.float32)

    def inverse(self, scaled: np.ndarray) -> np.ndarray:
        scaled = np.asarray(scaled, dtype=np.float64).reshape(-1, self.in_dim)
        return (scaled * self.half_range + self.center).astype(np.float32)

    def state_dict(self) -> dict:
        return {"kind": self.kind, "low": self.low, "high": self.high}


class OneHotScaler(BaseScaler):
    """Discrete(n) -> n-dimensional indicator vector.

    Taxi-v4's 500 states carry no usable metric structure (state 314 is not
    "between" 313 and 315 in any sense the dynamics respect), so feeding the raw
    integer to an MLP would invent an ordering that does not exist. One-hot is
    the honest encoding, and at n=500 with a 64-unit first layer it costs 32k
    parameters, which is nothing.

    `decode_fn`, when the environment provides one, is carried for plotting and
    analysis only -- it never touches the network input.
    """

    kind = "onehot"

    def __init__(self, n: int):
        self.n = int(n)
        self.in_dim = 1
        self.out_dim = self.n

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        idx = np.asarray(obs, dtype=np.int64).reshape(-1)
        if idx.min() < 0 or idx.max() >= self.n:
            raise ValueError(f"state index outside [0, {self.n})")
        out = np.zeros((len(idx), self.n), dtype=np.float32)
        out[np.arange(len(idx)), idx] = 1.0
        return out

    def state_dict(self) -> dict:
        return {"kind": self.kind, "n": self.n}


class ImageScaler(BaseScaler):
    """Flat grid encoding -> float32, unscaled.

    MiniGrid's encoding is three small-integer channels (object index, colour
    index, state), so the reference implementation feeds them to the CNN as raw
    floats. Mapping them through the declared Box bounds [0, 255] instead would
    squash every value into a sliver of [-1, 1] and throw away the resolution
    the network needs.
    """

    kind = "image"

    def __init__(self, dim: int, divisor: float = 1.0):
        self.in_dim = self.out_dim = int(dim)
        self.divisor = float(divisor)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32).reshape(-1, self.in_dim)
        return x if self.divisor == 1.0 else x / self.divisor

    def state_dict(self) -> dict:
        return {"kind": self.kind, "dim": self.in_dim, "divisor": self.divisor}


class RunningScaler(BaseScaler):
    """Welford running mean/std, VecNormalize style. Box spaces only."""

    kind = "running"

    def __init__(self, dim: int, epsilon: float = 1e-8, clip: float = 10.0):
        self.in_dim = self.out_dim = int(dim)
        self.mean = np.zeros(self.in_dim, dtype=np.float64)
        self.var = np.ones(self.in_dim, dtype=np.float64)
        self.count = 1e-4
        self.epsilon = epsilon
        self.clip = clip

    def update(self, obs: np.ndarray) -> None:
        obs = np.asarray(obs, dtype=np.float64).reshape(-1, self.in_dim)
        batch_mean, batch_var, batch_count = obs.mean(0), obs.var(0), obs.shape[0]
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m2 = self.var * self.count + batch_var * batch_count + delta**2 * self.count * batch_count / tot
        self.mean, self.var, self.count = new_mean, m2 / tot, tot

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64).reshape(-1, self.in_dim)
        z = (obs - self.mean) / np.sqrt(self.var + self.epsilon)
        return np.clip(z, -self.clip, self.clip).astype(np.float32)

    def state_dict(self) -> dict:
        return {"kind": self.kind, "mean": self.mean, "var": self.var,
                "count": self.count, "epsilon": self.epsilon, "clip": self.clip}

    def load_state_dict(self, d: dict) -> None:
        self.mean = np.asarray(d["mean"], dtype=np.float64)
        self.var = np.asarray(d["var"], dtype=np.float64)
        self.count = float(d["count"])
        self.epsilon = float(d["epsilon"])
        self.clip = float(d["clip"])


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #

def _is_discrete(space) -> bool:
    return space.__class__.__name__ == "Discrete"


def make_scaler(kind: str, observation_space) -> BaseScaler:
    """`kind` is honoured for Box spaces; a Discrete space is always one-hot.

    There is no meaningful 'fixed affine' or 'running mean/std' encoding of a
    categorical state, so the requested kind is overridden rather than silently
    producing a 1-D integer input.
    """
    if _is_discrete(observation_space):
        return OneHotScaler(int(observation_space.n))

    dim = int(np.prod(observation_space.shape))
    if kind == "image":
        return ImageScaler(dim)
    if kind == "fixed":
        return FixedScaler(observation_space.low, observation_space.high)
    if kind == "running":
        return RunningScaler(dim)
    if kind == "none":
        return IdentityScaler(dim)
    raise ValueError(f"unknown obs_norm: {kind!r}")


def scaler_from_state_dict(d: dict) -> BaseScaler:
    kind = d["kind"]
    if kind == "fixed":
        return FixedScaler(d["low"], d["high"])
    if kind == "onehot":
        return OneHotScaler(int(d["n"]))
    if kind == "image":
        return ImageScaler(int(d["dim"]), float(d.get("divisor", 1.0)))
    if kind == "running":
        s = RunningScaler(len(np.asarray(d["mean"])))
        s.load_state_dict(d)
        return s
    if kind == "none":
        return IdentityScaler(int(d["dim"]))
    raise ValueError(f"unknown scaler kind: {kind!r}")
