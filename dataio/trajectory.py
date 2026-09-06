"""Trajectory dataset: schema, recorder, loader, validator.

This dataset is the ONLY thing Notebook 03 (COCOA) is allowed to see. It is also
the input to Notebook 02's state sampling. Because six notebooks pass it around,
the schema is pinned here and `validate()` is run at the end of NB01.

Schema (one row per environment step, N = total recorded steps):

    raw_obs        (N, obs_dim) f32   unscaled observation s_t
    sim_state      (N, obs_dim) f64   underlying simulator state at s_t.
                                      NB02 restores from THIS, not from raw_obs.
    action         (N,)         i8    a_t
    reward         (N,)         f32   r_t
    next_raw_obs   (N, obs_dim) f32   TRUE successor observation s_{t+1}
                                      (the terminal obs when the episode ended,
                                      never the post-reset obs)
    next_sim_state (N, obs_dim) f64   simulator state at s_{t+1}
    terminated     (N,)         bool  MDP terminal (goal reached)
    truncated      (N,)         bool  TimeLimit cutoff -- NOT an MDP terminal
    logprob        (N,)         f32   log pi_theta_t(a_t | s_t) at collection time
    probs          (N, K)       f32   full action distribution at collection time
    value          (N,)         f32   V_theta_t(s_t) at collection time
    episode_id     (N,)         i32   globally unique within the run
    episode_t      (N,)         i16   step index within the episode
    global_step    (N,)         i32   environment steps elapsed when collected
    env_id         (N,)         i8    which parallel env produced the row

NOTE on `logprob` / `probs` / `value`: these are BEHAVIOUR-POLICY quantities,
recorded under whatever theta was current at that moment. They are not the
policy at any checkpoint. Notebook 04 compares A_COCOA against an oracle
computed at ONE fixed checkpoint, so anything policy-dependent must either be
recomputed from that checkpoint or the rows must be restricted to a window of
`global_step` around it. Use `window_around_step()` for the latter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1

FIELDS = {
    "raw_obs": np.float32,
    "sim_state": np.float64,
    "action": np.int8,
    "reward": np.float32,
    "next_raw_obs": np.float32,
    "next_sim_state": np.float64,
    "terminated": np.bool_,
    "truncated": np.bool_,
    "logprob": np.float32,
    "probs": np.float32,
    "value": np.float32,
    "episode_id": np.int32,
    "episode_t": np.int16,
    "global_step": np.int32,
    "env_id": np.int8,
}


# --------------------------------------------------------------------------- #
# Recorder
# --------------------------------------------------------------------------- #

#: Narrow dtypes for environments whose observations and simulator states are
#: small integers. This is a memory necessity, not a micro-optimisation: a
#: 1M-step MiniGrid run at the default widths is
#:     (192 + 192) * 4 B  +  (198 + 198) * 8 B  ~= 4.7 KB/row  ->  4.7 GB in RAM
#: before anything is written to disk. Compact dtypes plus a stride of 4 bring
#: that to ~300 MB. Grid codes fit in uint8; MiniGrid's step_count (up to 640 on
#: DoorKey-8x8) needs int16.
COMPACT_FIELDS = dict(FIELDS)
COMPACT_FIELDS.update({
    "raw_obs": np.uint8,
    "next_raw_obs": np.uint8,
    "sim_state": np.int16,
    "next_sim_state": np.int16,
})


class TrajectoryRecorder:
    def __init__(self, stride: int = 1, compact: bool = False):
        self.stride = max(1, int(stride))
        self.dtypes = COMPACT_FIELDS if compact else FIELDS
        self.compact = compact
        self._chunks: dict[str, list[np.ndarray]] = {k: [] for k in FIELDS}
        self._seen = 0

    def add_rollout(self, buf, global_step_start: int, n_envs: int) -> None:
        """Flatten one rollout (n_steps, n_envs, ...) into rows, in time order."""
        T = buf.ptr
        step_idx = np.arange(T, dtype=np.int64)
        gstep = (global_step_start + step_idx * n_envs).astype(np.int32)

        gstep_b = np.repeat(gstep[:, None], n_envs, axis=1)
        env_b = np.tile(np.arange(n_envs, dtype=np.int8)[None, :], (T, 1))

        raw = {
            "raw_obs": buf.raw_obs[:T],
            "sim_state": buf.sim_state[:T],
            "action": buf.actions[:T],
            "reward": buf.rewards[:T],
            "next_raw_obs": buf.next_raw_obs[:T],
            "next_sim_state": buf.next_sim_state[:T],
            "terminated": buf.terminated[:T],
            "truncated": buf.truncated[:T],
            "logprob": buf.logprobs[:T],
            "probs": buf.probs[:T],
            "value": buf.values[:T],
            "episode_id": buf.episode_id[:T],
            "episode_t": buf.episode_t[:T],
            "global_step": gstep_b,
            "env_id": env_b,
        }

        for k, arr in raw.items():
            flat = arr.reshape((-1,) + arr.shape[2:])
            if self.stride > 1:
                flat = flat[:: self.stride]
            self._chunks[k].append(flat.astype(self.dtypes[k], copy=False))

        self._seen += T * n_envs

    def finalize(self) -> dict[str, np.ndarray]:
        return {k: np.concatenate(v, axis=0) for k, v in self._chunks.items() if v}

    def save(self, path: Path, meta: dict | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.finalize()
        data["_schema_version"] = np.array([SCHEMA_VERSION], dtype=np.int32)
        data["_meta_compact_dtypes"] = np.array(int(self.compact), dtype=np.int32)
        if meta:
            for k, v in meta.items():
                data[f"_meta_{k}"] = np.array(v)
        np.savez_compressed(path, **data)
        return path


# --------------------------------------------------------------------------- #
# Loading / querying
# --------------------------------------------------------------------------- #

@dataclass
class Trajectories:
    data: dict[str, np.ndarray]

    def __getattr__(self, name: str) -> np.ndarray:
        try:
            return self.data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __len__(self) -> int:
        return len(self.data["action"])

    @property
    def n_actions(self) -> int:
        return self.data["probs"].shape[1]

    @property
    def obs_dim(self) -> int:
        return self.data["raw_obs"].shape[1]

    def select(self, mask_or_idx) -> "Trajectories":
        return Trajectories({k: v[mask_or_idx] for k, v in self.data.items()})

    def window_around_step(self, center_step: int, half_width_steps: int) -> "Trajectories":
        """Rows collected within +/- half_width_steps of `center_step`.

        Use this when a downstream comparison must hold the behaviour policy
        roughly fixed (e.g. matching COCOA's data to the checkpoint at which the
        NB02 oracle was computed).
        """
        g = self.data["global_step"]
        m = (g >= center_step - half_width_steps) & (g <= center_step + half_width_steps)
        return self.select(m)

    def episode_returns(self) -> tuple[np.ndarray, np.ndarray]:
        eid = self.data["episode_id"]
        r = self.data["reward"]
        uniq, inv = np.unique(eid, return_inverse=True)
        sums = np.zeros(len(uniq), dtype=np.float64)
        np.add.at(sums, inv, r)
        return uniq, sums


def load_trajectories(path: Path) -> Trajectories:
    with np.load(Path(path), allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    return Trajectories(data)


# --------------------------------------------------------------------------- #
# Validation  (Notebook 01's "data reloads correctly" CHECK)
# --------------------------------------------------------------------------- #

def validate(traj: Trajectories, n_actions: int, verbose: bool = True) -> list[str]:
    """Returns a list of problems. Empty list == dataset is structurally sound."""
    problems: list[str] = []
    d = traj.data
    n = len(traj)

    def bad(msg: str) -> None:
        problems.append(msg)

    for k, _dt in FIELDS.items():
        if k not in d:
            bad(f"missing field: {k}")
        elif len(d[k]) != n:
            bad(f"length mismatch on {k}: {len(d[k])} != {n}")

    if problems:
        return problems

    if not np.all(np.isfinite(d["raw_obs"])):
        bad("raw_obs contains non-finite values")
    if not np.all(np.isfinite(d["next_raw_obs"])):
        bad("next_raw_obs contains non-finite values")
    if not np.all(np.isfinite(d["value"])):
        bad("value contains non-finite values")

    if d["action"].min() < 0 or d["action"].max() >= n_actions:
        bad(f"actions outside [0, {n_actions})")

    p = d["probs"].astype(np.float64)
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-4):
        bad("probs rows do not sum to 1")
    if p.min() < 0:
        bad("probs contains negative entries")

    # logprob must equal log probs[action]
    lp = np.log(np.clip(p[np.arange(n), d["action"].astype(np.int64)], 1e-12, None))
    if not np.allclose(lp, d["logprob"], atol=1e-3):
        worst = float(np.max(np.abs(lp - d["logprob"])))
        bad(f"logprob disagrees with log probs[action] (max abs err {worst:.2e})")

    # terminated and truncated must not both be set
    if np.any(d["terminated"] & d["truncated"]):
        bad("some rows have terminated AND truncated set")

    # within an episode, next_raw_obs[t] must equal raw_obs[t+1] except at boundaries
    ended = d["terminated"] | d["truncated"]

    order = np.lexsort((d["episode_t"], d["episode_id"]))
    eid_o = d["episode_id"][order]
    obs_o = d["raw_obs"][order]
    nobs_o = d["next_raw_obs"][order]
    ended_o = ended[order]
    same_ep = eid_o[:-1] == eid_o[1:]
    interior = same_ep & ~ended_o[:-1]
    if interior.any():
        gap = np.abs(nobs_o[:-1][interior] - obs_o[1:][interior]).max()
        if gap > 1e-5:
            bad(f"next_raw_obs[t] != raw_obs[t+1] inside episodes (max gap {gap:.2e})")

    # every episode should end with at most one terminated-or-truncated row
    # (the final, still-running episode of each env has none, which is fine)
    _uniq, counts = np.unique(d["episode_id"][ended], return_counts=True)
    if counts.size and counts.max() > 1:
        bad("some episode_id has more than one terminal row")

    if verbose:
        n_ep = len(np.unique(d["episode_id"]))
        n_succ = int(d["terminated"].sum())
        print(f"  rows            {n:,}")
        print(f"  episodes        {n_ep:,}")
        print(f"  terminated rows {n_succ:,}   (MDP termination = task solved)")
        print(f"  truncated rows  {int(d['truncated'].sum()):,}")
        print(f"  problems        {len(problems)}")

    return problems
