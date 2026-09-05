"""Choosing the evaluation states the oracle is built on.

The plan asks for "500 states from the ~30% checkpoint". That phrase has to be
read carefully, because it names a *policy* and the states are a consequence of
it: the evaluation set should be states the policy at that checkpoint actually
visits. Sampling uniformly over all 400k trajectory rows would mix in states
visited by every other policy during training, and A_CF is policy-dependent.

`window` (the default) therefore restricts sampling to rows collected near the
checkpoint's own global_step. The same `Trajectories.window_around_step` helper
is what Notebook 03 should use to keep COCOA's data matched to the same policy,
so that Notebook 04's comparison is apples-to-apples.
"""

from __future__ import annotations

import numpy as np

from dataio.trajectory import Trajectories


def sample_states(
    traj: Trajectories,
    n_states: int,
    center_step: int | None = None,
    half_width_steps: int | None = None,
    strategy: str = "window",
    seed: int = 0,
) -> dict:
    """Return a dict with the sampled states and a record of how they were chosen.

    strategy:
      "window"     -- uniform over rows within +/- half_width_steps of center_step.
                      This is the policy-matched choice; use it.
      "all"        -- uniform over every recorded row, ignoring which policy
                      produced it. Provided for contrast, not recommended.
      "stratified" -- window, then spread the picks over a grid of the visited
                      state space so tails are represented. Better coverage for
                      NB04's scatter plots, but no longer distributed as the
                      policy's own visitation.
    """
    rng = np.random.default_rng(seed)

    if strategy == "all":
        pool = np.arange(len(traj))
    else:
        if center_step is None or half_width_steps is None:
            raise ValueError("window/stratified sampling needs center_step and half_width_steps")
        g = traj.global_step
        pool = np.flatnonzero((g >= center_step - half_width_steps) & (g <= center_step + half_width_steps))
        if len(pool) < n_states:
            raise ValueError(
                f"only {len(pool)} rows within +/-{half_width_steps:,} steps of "
                f"{center_step:,}; widen the window or lower n_states"
            )

    if strategy == "stratified":
        idx = _stratified_pick(traj.raw_obs[pool], n_states, rng)
        chosen = pool[idx]
    else:
        chosen = rng.choice(pool, size=n_states, replace=False)

    chosen = np.sort(chosen)
    return {
        "index": chosen,
        "sim_state": traj.sim_state[chosen],
        "raw_obs": traj.raw_obs[chosen],
        "global_step": traj.global_step[chosen],
        "episode_id": traj.episode_id[chosen],
        "episode_t": traj.episode_t[chosen],
        "behaviour_probs": traj.probs[chosen],   # pi at COLLECTION time, not at the checkpoint
        "strategy": strategy,
        "center_step": center_step,
        "half_width_steps": half_width_steps,
        "pool_size": len(pool),
    }


def _stratified_pick(obs: np.ndarray, n: int, rng: np.random.Generator, bins: int = 24) -> np.ndarray:
    """Spread picks across occupied cells of a 2-D grid, round-robin."""
    lo, hi = obs.min(0), obs.max(0)
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    cell = np.clip(((obs - lo) / span * bins).astype(int), 0, bins - 1)
    key = cell[:, 0] * bins + (cell[:, 1] if obs.shape[1] > 1 else 0)

    order = rng.permutation(len(obs))
    buckets: dict[int, list[int]] = {}
    for i in order:
        buckets.setdefault(int(key[i]), []).append(int(i))

    picked: list[int] = []
    keys = list(buckets)
    rng.shuffle(keys)
    while len(picked) < n:
        progressed = False
        for k in keys:
            if buckets[k]:
                picked.append(buckets[k].pop())
                progressed = True
                if len(picked) == n:
                    break
        if not progressed:
            raise ValueError("not enough distinct rows to stratify")
    return np.array(picked)
