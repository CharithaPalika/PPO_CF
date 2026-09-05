"""Landscape dataset: the artifact Notebook 02 produces and 03-06 consume.

Schema (N evaluation states, K actions, d state dims):

    sim_state       (N, d)     f64   simulator state -- restore from this
    raw_obs         (N, d)     f32   observation the policy sees in that state
    q_cf            (N, K)     f32   r(s,a) + gamma * V(s')
    a_cf            (N, K)     f32   q_cf - v_pi          <- THE reference landscape
    v_pi            (N,)       f32   sum_a pi(a|s) q_cf(s,a)
    v_critic        (N,)       f32   the critic's own V(s), for comparison only
    pi              (N, K)     f32   pi(.|s) at the oracle's checkpoint
    reward          (N, K)     f32   one-step reward
    terminated      (N, K)     bool
    truncated       (N, K)     bool
    next_raw_obs    (N, K, d)  f32
    next_sim_state  (N, K, d)  f64
    v_next          (N, K)     f32   V(s') used in q_cf
    index           (N,)       i64   row in the source trajectory file
    global_step     (N,)       i32   when that row was collected

Metadata (stored as `_meta_*`): seed, checkpoint fraction and step, gamma,
env_id, max_episode_steps, sampling strategy and window.

NB03 must not read `a_cf`, `q_cf`, `pi`, `v_pi` or `v_critic` -- those are the
labels it is supposed to recover from trajectories alone. It may read
`sim_state` / `raw_obs` / `index` to know WHICH states to produce a landscape
for. `evaluation_states_only()` returns exactly that subset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

LANDSCAPE_SCHEMA_VERSION = 1

LABEL_FIELDS = ("q_cf", "a_cf", "v_pi", "v_critic", "pi", "v_next", "reward",
                "terminated", "truncated", "next_raw_obs", "next_sim_state")


@dataclass
class Landscape:
    data: dict[str, np.ndarray]
    meta: dict

    def __getattr__(self, name: str) -> np.ndarray:
        try:
            return self.data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __len__(self) -> int:
        return len(self.data["a_cf"])

    @property
    def n_actions(self) -> int:
        return self.data["a_cf"].shape[1]

    def best_action(self) -> np.ndarray:
        return self.data["a_cf"].argmax(axis=1)

    def spread(self) -> np.ndarray:
        """max_a Q - min_a Q. How much the actions differ at all."""
        q = self.data["q_cf"]
        return q.max(axis=1) - q.min(axis=1)

    def centering_error(self) -> np.ndarray:
        return np.abs((self.data["pi"] * self.data["a_cf"]).sum(axis=1))

    def evaluation_states_only(self) -> dict:
        """What Notebook 03 is allowed to see: which states, and nothing else."""
        return {k: self.data[k] for k in ("index", "sim_state", "raw_obs", "global_step")
                if k in self.data}


def save_landscape(path: Path, data: dict, meta: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {k: np.asarray(v) for k, v in data.items()}
    blob["_schema_version"] = np.array([LANDSCAPE_SCHEMA_VERSION], dtype=np.int32)
    for k, v in meta.items():
        blob[f"_meta_{k}"] = np.array(v)
    np.savez_compressed(path, **blob)
    return path


def load_landscape(path: Path) -> Landscape:
    with np.load(Path(path), allow_pickle=False) as z:
        data, meta = {}, {}
        for k in z.files:
            if k.startswith("_meta_"):
                v = z[k]
                meta[k[6:]] = v.item() if v.ndim == 0 else v
            elif not k.startswith("_"):
                data[k] = z[k]
    return Landscape(data=data, meta=meta)


def validate_landscape(land: Landscape, gamma: float, verbose: bool = True) -> list[str]:
    """Structural checks. Empty list means the landscape is internally consistent."""
    problems: list[str] = []
    d = land.data
    n, k = d["a_cf"].shape

    def bad(m: str) -> None:
        problems.append(m)

    for f in ("sim_state", "raw_obs", "q_cf", "a_cf", "v_pi", "pi"):
        if f not in d:
            bad(f"missing field: {f}")
    if problems:
        return problems

    if not np.all(np.isfinite(d["q_cf"])):
        bad("q_cf contains non-finite values")
    if not np.all(np.isfinite(d["a_cf"])):
        bad("a_cf contains non-finite values")

    p = d["pi"].astype(np.float64)
    if not np.allclose(p.sum(1), 1.0, atol=1e-4):
        bad("pi rows do not sum to 1")

    # a_cf must equal q_cf - v_pi
    recon = d["q_cf"] - d["v_pi"][:, None]
    if not np.allclose(recon, d["a_cf"], atol=1e-4):
        bad(f"a_cf != q_cf - v_pi (max err {np.abs(recon - d['a_cf']).max():.2e})")

    # v_pi must equal sum_a pi * q_cf
    v_recon = (p * d["q_cf"]).sum(1)
    if not np.allclose(v_recon, d["v_pi"], atol=1e-3):
        bad(f"v_pi != sum_a pi*q_cf (max err {np.abs(v_recon - d['v_pi']).max():.2e})")

    # the centering identity Gate 2 asks for
    cerr = float(np.abs((p * d["a_cf"]).sum(1)).max())
    if cerr > 1e-4:
        bad(f"policy-centering violated: max |sum_a pi*A_CF| = {cerr:.2e}")

    # q_cf must be reconstructible from its own stored parts
    if "reward" in d and "v_next" in d and "terminated" in d:
        q_recon = d["reward"] + gamma * d["v_next"] * (~d["terminated"])
        if not np.allclose(q_recon, d["q_cf"], atol=1e-3):
            bad("q_cf != reward + gamma*v_next*(1-terminated)")

    if verbose:
        print(f"  states            {n:,}   actions {k}")
        print(f"  max |sum pi*A_CF| {cerr:.2e}")
        print(f"  mean |A_CF|       {np.abs(d['a_cf']).mean():.4f}")
        print(f"  median Q spread   {np.median(d['q_cf'].max(1) - d['q_cf'].min(1)):.4f}")
        print(f"  problems          {len(problems)}")

    return problems
