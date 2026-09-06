"""A small, explicit synchronous env pool, plus simulator state get/restore.

Why not `gymnasium.vector.SyncVectorEnv`? Because its autoreset semantics
(next-step autoreset in Gymnasium >= 1.0) hide the true terminal observation
behind an extra step, and this project's whole output is a per-step trajectory
dataset in which `next_obs` must be the ACTUAL successor state. Getting that
subtly wrong would poison Notebooks 02-06 in a way that is hard to detect.

Contract of `step()`:
    next_obs   -- the true successor observation (terminal obs on termination)
    reset_obs  -- what the agent observes next (reset obs if the episode ended)
    terminated -- MDP termination. Bootstrapping must NOT continue past this.
    truncated  -- TimeLimit cutoff. Bootstrapping MUST continue past this.

Two observation shapes are supported and kept deliberately separate:
    raw_obs_dim   -- width of the raw observation vector (2 for MountainCar-v0,
                     1 for Taxi-v4, whose observation is a single integer)
    input_dim     -- width of the network input after the scaler (2, or 500 for
                     Taxi's one-hot)
Everything stored in trajectories and landscapes uses the RAW form; the scaler
is the single place the conversion happens.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import gymnasium as gym
import numpy as np


# --------------------------------------------------------------------------- #
# Simulator state access (needed by Notebook 02's counterfactual oracle)
# --------------------------------------------------------------------------- #

def is_tabular(env: gym.Env) -> bool:
    """True for the classic discrete envs that expose a state index `s` and a
    transition table `P` (Taxi, FrozenLake, CliffWalking)."""
    u = env.unwrapped
    return hasattr(u, "P") and hasattr(u, "s")


def _minigrid(env: gym.Env) -> bool:
    from envs.minigrid_env import is_minigrid
    return is_minigrid(env)


def get_sim_state(env: gym.Env) -> np.ndarray:
    """Underlying simulator state as a float array, always 1-D.

    Classic-control envs expose `unwrapped.state`; the tabular envs expose an
    integer `unwrapped.s`. Stored in the trajectory dataset alongside the
    observation so NB02 can restore an exact state without assuming obs == state.
    """
    u = env.unwrapped
    state = getattr(u, "state", None)
    if state is not None:
        return np.asarray(state, dtype=np.float64).ravel().copy()
    if hasattr(u, "s"):
        return np.array([u.s], dtype=np.float64)
    if _minigrid(env):
        from envs.minigrid_env import get_minigrid_state
        return get_minigrid_state(env)
    raise AttributeError(
        f"{type(u).__name__} exposes neither `.state` nor `.s`; NB02's oracle "
        "needs an explicit state-restore path for this environment."
    )


def set_sim_state(env: gym.Env, state: np.ndarray, elapsed_steps: int = 0) -> np.ndarray:
    """Force the simulator into `state` and return the corresponding RAW observation.

    `elapsed_steps` resets the TimeLimit counter so a restored state can be
    stepped without an unexpected truncation. NB02 uses 0 because it only ever
    takes a single step.
    """
    env.reset()  # allocates np_random and internal buffers
    u = env.unwrapped
    state = np.asarray(state, dtype=np.float64).ravel()

    if getattr(u, "state", None) is not None:
        u.state = state.copy()
    elif hasattr(u, "s"):
        u.s = int(round(float(state[0])))
        if hasattr(u, "lastaction"):
            u.lastaction = None
    elif _minigrid(env):
        from envs.minigrid_env import set_minigrid_state
        set_minigrid_state(env, state, elapsed_steps=elapsed_steps)
    else:
        raise AttributeError(f"cannot restore state on {type(u).__name__}")

    # Reset any TimeLimit counter in the wrapper chain.
    w = env
    while w is not u:
        if hasattr(w, "_elapsed_steps"):
            w._elapsed_steps = elapsed_steps
        w = getattr(w, "env", u)

    return _obs_from_state(env)


def _obs_from_state(env: gym.Env) -> np.ndarray:
    """The observation the agent would see, AFTER the wrapper chain.

    This has to run the wrappers, not just read the raw simulator state: with
    FullyObsWrapper + ImgObsWrapper the network input is a transformed view of
    the grid, and NB02 must query the policy on exactly what training used.
    """
    u = env.unwrapped

    if hasattr(u, "gen_obs"):                       # MiniGrid
        obs = u.gen_obs()
    elif hasattr(u, "_get_obs"):
        obs = u._get_obs()
    elif getattr(u, "state", None) is not None:
        obs = np.asarray(u.state, dtype=np.float32)
    else:
        obs = u.s

    # Re-apply observation wrappers from the inside out.
    chain = []
    w = env
    while w is not u:
        chain.append(w)
        w = getattr(w, "env", u)
    for wrapper in reversed(chain):
        if hasattr(wrapper, "observation"):
            obs = wrapper.observation(obs)

    return np.asarray(obs, dtype=np.float32).ravel()


def assert_deterministic(env: gym.Env) -> None:
    """The oracle assumes deterministic dynamics. Taxi-v4 is deterministic only
    with is_rainy=False and fickle_passenger=False, which are the defaults --
    but they are constructor arguments, so check rather than assume."""
    u = env.unwrapped
    for flag in ("is_rainy", "fickle_passenger"):
        if getattr(u, flag, False):
            raise ValueError(
                f"{type(u).__name__} was built with {flag}=True, which makes the "
                "dynamics stochastic. The one-step oracle in NB02 assumes a "
                "deterministic simulator."
            )
    if hasattr(u, "P"):
        for s, row in u.P.items():
            for a, outcomes in row.items():
                if len(outcomes) > 1:
                    raise ValueError(
                        f"transition ({s}, {a}) has {len(outcomes)} outcomes; "
                        "the environment is stochastic."
                    )


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #

def make_env(env_id: str, max_episode_steps: int | None = None,
             fully_observable: bool = True, **kwargs) -> gym.Env:
    if env_id.startswith("MiniGrid"):
        from envs.minigrid_env import make_minigrid_env
        return make_minigrid_env(env_id, fully_observable=fully_observable,
                                 max_episode_steps=max_episode_steps, **kwargs)
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = max_episode_steps
    return gym.make(env_id, **kwargs)


def env_dims(env: gym.Env) -> tuple[int, int]:
    """(raw_obs_dim, n_actions) for an env instance."""
    space = env.observation_space
    raw = 1 if space.__class__.__name__ == "Discrete" else int(np.prod(space.shape))
    return raw, int(env.action_space.n)


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
        env_kwargs: dict[str, Any] | None = None,
        layout_seeds: Sequence[int] | None = None,
        layout_seed_mode: str = "cycle",
        reward_cfg: Any = None,
        gamma: float = 0.99,
    ):
        fn = env_fn or (lambda: make_env(env_id, max_episode_steps, **(env_kwargs or {})))
        self.envs = [fn() for _ in range(n_envs)]
        self.n_envs = n_envs
        self.env_id = env_id

        # --- layout control ------------------------------------------------ #
        # MiniGrid regenerates the layout on EVERY reset unless a seed is given,
        # so with layout_seeds=None the agent faces a new maze each episode and
        # PPO is implicitly being asked to generalise. A fixed pool turns the
        # task back into a single MDP, which is all NB02-06 needs.
        self.layout_seeds = None if layout_seeds is None else [int(s) for s in layout_seeds]
        self.layout_seed_mode = layout_seed_mode
        self._layout_rng = np.random.default_rng(seed)
        self._layout_cursor = np.arange(n_envs)

        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space
        self.raw_obs_dim, self.n_actions = env_dims(self.envs[0])
        self.is_tabular = is_tabular(self.envs[0])
        # The simulator state is not always the observation. MiniGrid's fully
        # observable image is 8*8*3 = 192, while its restorable state is 198
        # (grid + agent pos/dir + carried object + step count), so the two
        # widths are tracked separately.
        self.envs[0].reset(seed=seed)          # populate internal state before probing
        self.sim_state_dim = len(get_sim_state(self.envs[0]))

        self._seeds = [seed * 10_000 + i for i in range(n_envs)]
        for e, s in zip(self.envs, self._seeds):
            e.action_space.seed(s)

        self._next_episode_id = 0
        self.episode_id = np.zeros(n_envs, dtype=np.int64)
        self.episode_t = np.zeros(n_envs, dtype=np.int64)
        self.episode_return = np.zeros(n_envs, dtype=np.float64)
        self.episode_length = np.zeros(n_envs, dtype=np.int64)

        self._obs = np.zeros((n_envs, self.raw_obs_dim), dtype=np.float32)

        # --- MiniGrid sub-goal probes and reward shaping -------------------- #
        self.is_minigrid_pool = _minigrid(self.envs[0])
        self.probes = None
        self.shapers = None
        if self.is_minigrid_pool:
            from envs.shaping import MiniGridProbe
            self.probes = [MiniGridProbe(e) for e in self.envs]
        if reward_cfg is not None and getattr(reward_cfg, "active", False):
            from envs.shaping import RewardShaper
            self.shapers = [
                RewardShaper(reward_cfg, self.probes[i] if self.probes else None, gamma)
                for i in range(n_envs)
            ]
        # per-episode sub-goal latches, mirrored from the probes
        self.episode_key = np.zeros(n_envs, dtype=bool)
        self.episode_door = np.zeros(n_envs, dtype=bool)

    # -- layout ------------------------------------------------------------- #

    def _next_layout_seed(self, i: int) -> int | None:
        """Seed for env `i`'s next reset, or None for a fresh random layout."""
        if not self.layout_seeds:
            return None
        if self.layout_seed_mode == "random":
            return int(self._layout_rng.choice(self.layout_seeds))
        s = self.layout_seeds[int(self._layout_cursor[i]) % len(self.layout_seeds)]
        self._layout_cursor[i] += self.n_envs
        return int(s)

    def _reset_env(self, i: int, seed: int | None) -> np.ndarray:
        env = self.envs[i]
        obs, _ = env.reset(seed=seed)
        if self.probes is not None:
            self.probes[i].reset_probe()
        if self.shapers is not None:
            self.shapers[i].on_reset()
        self.episode_key[i] = False
        self.episode_door[i] = False
        return np.asarray(obs, dtype=np.float32).reshape(-1)

    def set_progress(self, frac: float) -> None:
        """Fraction of training elapsed. Drives the count-bonus anneal."""
        if self.shapers is not None:
            for s in self.shapers:
                s.progress = float(frac)

    @property
    def bonus_coef(self) -> float:
        return 0.0 if self.shapers is None else float(self.shapers[0].count_coef)

    # -- lifecycle ---------------------------------------------------------- #

    def reset(self) -> np.ndarray:
        for i, s in enumerate(self._seeds):
            layout = self._next_layout_seed(i)
            self._obs[i] = self._reset_env(i, layout if layout is not None else s)
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
        n, d = self.n_envs, self.raw_obs_dim
        next_obs = np.zeros((n, d), dtype=np.float32)
        reset_obs = np.zeros((n, d), dtype=np.float32)
        next_sim = np.zeros((n, self.sim_state_dim), dtype=np.float64)
        reward = np.zeros(n, dtype=np.float32)
        terminated = np.zeros(n, dtype=bool)
        truncated = np.zeros(n, dtype=bool)

        step_episode_id = self.episode_id.copy()
        step_t = self.episode_t.copy()
        finished: list[dict] = []

        for i, env in enumerate(self.envs):
            o, r, term, trunc, _info = env.step(int(actions[i]))
            next_obs[i] = np.asarray(o, dtype=np.float32).reshape(-1)
            next_sim[i] = get_sim_state(env)

            # Sub-goal latches BEFORE any reset, so they describe this episode.
            if self.probes is not None:
                k, d = self.probes[i].observe()
                self.episode_key[i] |= k
                self.episode_door[i] |= d

            # The buffer sees the shaped reward (what the critic is trained on);
            # `episode_return` accumulates the RAW environment reward, so the
            # reported return and success rate stay comparable across configs
            # with and without shaping.
            shaped = r if self.shapers is None else self.shapers[i].on_step(float(r), bool(term))
            reward[i] = shaped
            terminated[i] = term
            truncated[i] = trunc

            self.episode_return[i] += r
            self.episode_length[i] += 1

            if term or trunc:
                finished.append({
                    "env_id": i,
                    "episode_id": int(self.episode_id[i]),
                    "return": float(self.episode_return[i]),
                    "length": int(self.episode_length[i]),
                    # `success` == MDP termination. MountainCar terminates only
                    # at the goal; Taxi terminates only on a correct dropoff;
                    # DoorKey terminates only on reaching the goal square.
                    "success": bool(term),
                    # DoorKey sub-goals. The rungs between "did nothing" and
                    # "solved", which is the only way to tell the failure modes
                    # apart while success is pinned at zero.
                    "picked_key": bool(self.episode_key[i]),
                    "opened_door": bool(self.episode_door[i]),
                })
                reset_obs[i] = self._reset_env(i, self._next_layout_seed(i))
                self.episode_id[i] = self._new_episode_id()
                self.episode_t[i] = 0
                self.episode_return[i] = 0.0
                self.episode_length[i] = 0
            else:
                reset_obs[i] = next_obs[i]
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
