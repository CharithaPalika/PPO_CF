"""A small, explicit synchronous env pool.

Why not `gymnasium.vector.SyncVectorEnv`? Because its autoreset semantics
(next-step autoreset in Gymnasium >= 1.0) hide the true terminal observation
behind an extra step, and this project's whole output is a per-step trajectory
dataset in which `next_obs` must be the ACTUAL successor state. Getting that
subtly wrong would poison Notebooks 02-06 in a way that is very hard to detect.

So the pool is written out longhand: ~100 lines, no hidden buffering.

Contract of `step()`:
    returns a dict of arrays, all length n_envs, where
      next_obs   -- the true successor observation (terminal obs on termination)
      reset_obs  -- what the agent actually observes next (reset obs if the
                    episode ended, otherwise identical to next_obs)
      terminated -- MDP termination (goal reached). Bootstrapping must NOT
                    continue past this.
      truncated  -- TimeLimit cutoff. Bootstrapping MUST continue past this.
"""

from __future__ import annotations

from typing import Any, Callable

import gymnasium as gym
import numpy as np


# --------------------------------------------------------------------------- #
# Simulator state access (needed by Notebook 02's counterfactual oracle)
# --------------------------------------------------------------------------- #

def get_sim_state(env: gym.Env) -> np.ndarray:
    """Return the underlying simulator state as a float array.

    For classic-control envs this is `env.unwrapped.state`. It is stored in the
    trajectory dataset alongside `obs` so that NB02 can restore an exact state
    without having to trust that obs == state.
    """
    state = getattr(env.unwrapped, "state", None)
    if state is None:
        raise AttributeError(
            f"{type(env.unwrapped).__name__} exposes no `.state`; "
            "NB02's oracle needs an explicit state-restore path for this env."
        )
    return np.asarray(state, dtype=np.float64).copy()


def set_sim_state(env: gym.Env, state: np.ndarray, elapsed_steps: int = 0) -> np.ndarray:
    """Force the simulator into `state` and return the corresponding observation.

    `elapsed_steps` resets the TimeLimit counter so that a restored state can be
    stepped without an unexpected truncation. NB02 uses elapsed_steps=0 because
    it only ever takes a single step.
    """
    env.reset()  # allocates np_random / internal buffers if not already done
    env.unwrapped.state = np.asarray(state, dtype=np.float64).copy()

    # Walk the wrapper chain and reset any TimeLimit counter we find.
    w = env
    while w is not env.unwrapped:
        if hasattr(w, "_elapsed_steps"):
            w._elapsed_steps = elapsed_steps
        w = getattr(w, "env", env.unwrapped)

    return _obs_from_state(env)


def _obs_from_state(env: gym.Env) -> np.ndarray:
    u = env.unwrapped
    if hasattr(u, "_get_obs"):
        return np.asarray(u._get_obs(), dtype=np.float32)
    return np.asarray(u.state, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #

def make_env(env_id: str, max_episode_steps: int | None = None) -> gym.Env:
    kwargs: dict[str, Any] = {}
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = max_episode_steps
    return gym.make(env_id, **kwargs)


# --------------------------------------------------------------------------- #
# Pool
# --------------------------------------------------------------------------- #

class EnvPool:
    def __init__(
        self,
        env_id: str,
        n_envs: int,
        seed: int,
        max_episode_steps: int | None = None,
        env_fn: Callable[[], gym.Env] | None = None,
    ):
        fn = env_fn or (lambda: make_env(env_id, max_episode_steps))
        self.envs = [fn() for _ in range(n_envs)]
        self.n_envs = n_envs
        self.env_id = env_id

        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space
        self.obs_dim = int(np.prod(self.single_observation_space.shape))
        self.n_actions = int(self.single_action_space.n)

        # Deterministic, distinct stream per env.
        self._seeds = [seed * 10_000 + i for i in range(n_envs)]
        for e, s in zip(self.envs, self._seeds):
            e.action_space.seed(s)

        # Episode bookkeeping. `episode_id` is globally unique within a run.
        self._next_episode_id = 0
        self.episode_id = np.zeros(n_envs, dtype=np.int64)
        self.episode_t = np.zeros(n_envs, dtype=np.int64)
        self.episode_return = np.zeros(n_envs, dtype=np.float64)
        self.episode_length = np.zeros(n_envs, dtype=np.int64)

        self._obs = np.zeros((n_envs, self.obs_dim), dtype=np.float32)

    # -- lifecycle ---------------------------------------------------------- #

    def reset(self) -> np.ndarray:
        for i, (e, s) in enumerate(zip(self.envs, self._seeds)):
            obs, _ = e.reset(seed=s)
            self._obs[i] = obs
            self.episode_id[i] = self._new_episode_id()
        self.episode_t[:] = 0
        self.episode_return[:] = 0.0
        self.episode_length[:] = 0
        return self._obs.copy()

    def _new_episode_id(self) -> int:
        eid = self._next_episode_id
        self._next_episode_id += 1
        return eid

    def sim_states(self) -> np.ndarray:
        return np.stack([get_sim_state(e) for e in self.envs])

    @property
    def obs(self) -> np.ndarray:
        return self._obs.copy()

    # -- step --------------------------------------------------------------- #

    def step(self, actions: np.ndarray) -> dict[str, np.ndarray]:
        n = self.n_envs
        next_obs = np.zeros_like(self._obs)
        reset_obs = np.zeros_like(self._obs)
        next_sim = np.zeros((n, self.obs_dim), dtype=np.float64)
        reward = np.zeros(n, dtype=np.float32)
        terminated = np.zeros(n, dtype=bool)
        truncated = np.zeros(n, dtype=bool)

        step_episode_id = self.episode_id.copy()
        step_t = self.episode_t.copy()

        finished: list[dict] = []

        for i, env in enumerate(self.envs):
            o, r, term, trunc, _info = env.step(int(actions[i]))
            next_obs[i] = o
            next_sim[i] = get_sim_state(env)
            reward[i] = r
            terminated[i] = term
            truncated[i] = trunc

            self.episode_return[i] += r
            self.episode_length[i] += 1

            if term or trunc:
                finished.append(
                    {
                        "env_id": i,
                        "episode_id": int(self.episode_id[i]),
                        "return": float(self.episode_return[i]),
                        "length": int(self.episode_length[i]),
                        "success": bool(term),  # MountainCar terminates only on goal
                    }
                )
                ro, _ = env.reset()
                reset_obs[i] = ro
                self.episode_id[i] = self._new_episode_id()
                self.episode_t[i] = 0
                self.episode_return[i] = 0.0
                self.episode_length[i] = 0
            else:
                reset_obs[i] = o
                self.episode_t[i] += 1

        self._obs = reset_obs.copy()

        return {
            "next_obs": next_obs,
            "next_sim_state": next_sim,
            "reset_obs": reset_obs,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "episode_id": step_episode_id,
            "episode_t": step_t,
            "finished": finished,
        }

    def close(self) -> None:
        for e in self.envs:
            e.close()
