"""MiniGrid support: construction, and simulator state get/restore.

Two things need care here.

OBSERVABILITY. MiniGrid's default observation is a 7x7 egocentric view, which
makes it a POMDP. That is fine for plain PPO but it breaks the counterfactual
oracle: A_CF(s,a) = r + gamma*V(s') - V_pi(s) is only well-defined in an MDP,
and under partial observability the critic learns V(o) rather than V(s) while
two different simulator states can produce byte-identical observations. So
`fully_observable=True` (the default here) stacks FullyObsWrapper, giving the
full 8x8x3 grid encoding and keeping the problem Markov.

STATE. MiniGrid exposes no `.state` or `.s`, so `envs.env_pool.get_sim_state`
delegates here. The state is packed into a FIXED-WIDTH vector so it fits the
existing trajectory and landscape schemas unchanged:

    [ grid.encode().ravel()   (W*H*3) ,  agent_col, agent_row, agent_dir,
      carried_type, carried_color, step_count ]

`grid.encode()` already carries door open/closed/locked in its third channel,
which is exactly the state bit DoorKey turns on. The carried object is NOT in
the grid once picked up, hence the two extra slots.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

_STATE_TAIL = 6  # agent_col, agent_row, agent_dir, carried_type, carried_color, step_count


def is_minigrid(env: gym.Env) -> bool:
    u = env.unwrapped
    return hasattr(u, "grid") and hasattr(u, "agent_pos") and hasattr(u, "agent_dir")


def make_minigrid_env(
    env_id: str,
    fully_observable: bool = True,
    max_episode_steps: int | None = None,
    **kwargs,
) -> gym.Env:
    """MiniGrid env with the dict observation reduced to a plain image array.

    Note MiniGrid sets its own step limit internally (`unwrapped.max_steps`,
    640 for DoorKey-8x8) rather than through the registration's
    `max_episode_steps`, so leaving `max_episode_steps=None` keeps the
    environment's own limit rather than removing it.
    """
    from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper

    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = max_episode_steps
    env = gym.make(env_id, **kwargs)
    if fully_observable:
        env = FullyObsWrapper(env)
    return ImgObsWrapper(env)


def sim_state_dim(env: gym.Env) -> int:
    u = env.unwrapped
    return int(np.prod(u.grid.encode().shape)) + _STATE_TAIL


def get_minigrid_state(env: gym.Env) -> np.ndarray:
    u = env.unwrapped
    grid = u.grid.encode().ravel().astype(np.float64)

    carried = u.carrying
    if carried is None:
        c_type, c_color = -1.0, -1.0
    else:
        from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX
        c_type = float(OBJECT_TO_IDX[carried.type])
        c_color = float(COLOR_TO_IDX[carried.color])

    pos = u.agent_pos
    tail = np.array(
        [float(pos[0]), float(pos[1]), float(u.agent_dir), c_type, c_color, float(u.step_count)],
        dtype=np.float64,
    )
    return np.concatenate([grid, tail])


def set_minigrid_state(env: gym.Env, state: np.ndarray, elapsed_steps: int | None = None) -> None:
    """Restore an exact MiniGrid state. Caller is responsible for env.reset() first."""
    from minigrid.core.constants import IDX_TO_COLOR, IDX_TO_OBJECT
    from minigrid.core.grid import Grid
    from minigrid.core.world_object import WorldObj

    u = env.unwrapped
    shape = u.grid.encode().shape
    n_grid = int(np.prod(shape))

    state = np.asarray(state, dtype=np.float64).ravel()
    grid_arr = state[:n_grid].reshape(shape).astype(np.uint8)
    tail = state[n_grid:]

    grid, _vis = Grid.decode(grid_arr)
    u.grid = grid
    u.agent_pos = (int(tail[0]), int(tail[1]))
    u.agent_dir = int(tail[2])

    c_type, c_color = int(tail[3]), int(tail[4])
    if c_type < 0:
        u.carrying = None
    else:
        obj = WorldObj.decode(c_type, c_color, 0)
        u.carrying = obj
        if obj is not None:
            obj.cur_pos = np.array([-1, -1])

    u.step_count = int(tail[5]) if elapsed_steps is None else int(elapsed_steps)


def decode_summary(env: gym.Env) -> dict:
    """Human-readable state, for notebook inspection only."""
    u = env.unwrapped
    return {
        "agent_pos": tuple(int(v) for v in u.agent_pos),
        "agent_dir": int(u.agent_dir),
        "carrying": None if u.carrying is None else u.carrying.type,
        "step_count": int(u.step_count),
    }


ACTION_NAMES = [
    "0 turn left", "1 turn right", "2 forward",
    "3 pickup", "4 drop", "5 toggle", "6 done",
]
