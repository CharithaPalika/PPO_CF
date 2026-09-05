"""Notebook 02's engine: build the explicit counterfactual landscape.

    from scripts.build_oracle import build_all
    landscapes = build_all(cfg)

or:
    python -m scripts.build_oracle --checkpoint-fraction 0.75 --n-states 500
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

import numpy as np

from config import ExperimentConfig, seed_dir
from dataio import (
    checkpoint_path,
    load_checkpoint,
    load_trajectories,
    save_landscape,
)
from oracle.counterfactual import (
    check_determinism,
    compute_landscape,
    mc_reference,
    validate_restore,
)
from oracle.sampling import sample_states


def landscape_path(cfg: ExperimentConfig, seed: int, tag: str = "oracle") -> Path:
    """`tag` keeps side experiments from clobbering the main landscape file.

    The checkpoint sweep in NB02 section 8 rebuilds a small 150-state landscape at
    every fraction; without a distinct tag it would silently overwrite the
    500-state file that NB03 is supposed to consume, at the same checkpoint.
    """
    frac = int(round(cfg.oracle.checkpoint_fraction * 100))
    return seed_dir(cfg.run.run_name, seed) / "landscapes" / f"{tag}_ckpt{frac:03d}.npz"


def build_seed(cfg: ExperimentConfig, seed: int, verbose: bool = True, tag: str = "oracle") -> dict:
    o = cfg.oracle
    sd = seed_dir(cfg.run.run_name, seed)

    ck_path = checkpoint_path(sd / "checkpoints", o.checkpoint_fraction)
    if not ck_path.exists():
        raise FileNotFoundError(f"{ck_path} not found -- run Notebook 01 first")
    ck = load_checkpoint(ck_path)
    traj = load_trajectories(sd / "trajectories.npz")

    if verbose:
        print(f"\n=== seed {seed} " + "=" * 50)
        print(f"  checkpoint {ck_path.name}  (fraction {ck.fraction:.0%}, step {ck.global_step:,})")

    # --- 1. choose evaluation states ------------------------------------- #
    half_width = int(o.state_window_frac * cfg.ppo.total_timesteps)
    picked = sample_states(
        traj,
        n_states=o.n_states,
        center_step=ck.global_step,
        half_width_steps=half_width,
        strategy=o.sampling,
        seed=o.seed_for_states + seed,
    )
    if verbose:
        print(f"  states     {o.n_states} via '{o.sampling}' from a pool of "
              f"{picked['pool_size']:,} rows within +/-{half_width:,} steps")

    # --- 2. restore/step correctness, BEFORE trusting any oracle number --- #
    restore = validate_restore(traj, cfg.env.env_id, cfg.env.max_episode_steps, n=300, seed=seed)
    determinism = check_determinism(
        picked["sim_state"], cfg.env.env_id, cfg.env.max_episode_steps, traj.n_actions
    )
    if verbose:
        print(f"  restore    replayed {restore['n']} recorded transitions: "
              f"state err {restore['max_state_err']:.2e}, reward err {restore['max_reward_err']:.2e}, "
              f"terminated agreement {restore['terminated_agreement']:.3f} "
              f"-> {'PASS' if restore['passed'] else 'FAIL'}")
        print(f"  determinism {'PASS' if determinism else 'FAIL'}")
    if not restore["passed"]:
        raise RuntimeError(
            "restore/step validation FAILED -- the oracle would be measuring the wrong "
            f"transitions. Details: {restore}"
        )

    # --- 3. the oracle ----------------------------------------------------- #
    t0 = time.time()
    land = compute_landscape(
        ck,
        picked["sim_state"],
        env_id=cfg.env.env_id,
        max_episode_steps=cfg.env.max_episode_steps,
        gamma=cfg.ppo.gamma,
        n_actions=traj.n_actions,
    )
    land["index"] = picked["index"].astype(np.int64)
    land["global_step"] = picked["global_step"].astype(np.int32)
    if verbose:
        print(f"  landscape  {len(land['a_cf'])} states x {traj.n_actions} actions "
              f"in {time.time() - t0:.1f}s")

    # --- 4. Monte-Carlo diagnostic ----------------------------------------- #
    mc = None
    if o.run_mc_check:
        t0 = time.time()
        m = min(o.mc_n_states, len(picked["sim_state"]))
        q_mc, q_se = mc_reference(
            ck,
            picked["sim_state"][:m],
            env_id=cfg.env.env_id,
            max_episode_steps=cfg.env.max_episode_steps,
            gamma=cfg.ppo.gamma,
            n_actions=traj.n_actions,
            n_rollouts=o.mc_rollouts,
            horizon=o.mc_horizon,
            seed=seed,
            return_se=True,
        )
        pi_m = land["pi"][:m]
        a_mc = q_mc - (pi_m * q_mc).sum(1, keepdims=True)
        land["mc_index"] = np.arange(m, dtype=np.int64)
        land["q_mc"] = q_mc
        land["q_mc_se"] = q_se
        land["a_mc"] = a_mc.astype(np.float32)
        mc = {
            "n_states": m,
            "mean_abs_q_gap": float(np.abs(q_mc - land["q_cf"][:m]).mean()),
            "mean_abs_a_gap": float(np.abs(a_mc - land["a_cf"][:m]).mean()),
            # Ratio of mean magnitudes, NOT a least-squares slope: the slope is
            # dominated by a handful of large-|A_MC| states and reads ~0.12 even
            # when the two landscapes have essentially the same scale.
            "magnitude_ratio": float(
                np.abs(land["a_cf"][:m]).mean() / max(np.abs(a_mc).mean(), 1e-9)
            ),
            "pearson": float(np.corrcoef(a_mc.ravel(), land["a_cf"][:m].ravel())[0, 1]),
            "best_action_agreement": float(
                (a_mc.argmax(1) == land["a_cf"][:m].argmax(1)).mean()
            ),
            "seconds": time.time() - t0,
        }
        if verbose:
            print(f"  MC check   {m} states x {o.mc_rollouts} CRN rollouts: "
                  f"magnitude ratio {mc['magnitude_ratio']:.2f}, "
                  f"Pearson {mc['pearson']:.2f}, "
                  f"best-action agreement {mc['best_action_agreement']:.2f} "
                  f"({mc['seconds']:.0f}s)")

    # --- 5. save ------------------------------------------------------------ #
    meta = {
        "seed": seed,
        "run_name": cfg.run.run_name,
        "checkpoint_fraction": float(ck.fraction),
        "checkpoint_step": int(ck.global_step),
        "gamma": float(cfg.ppo.gamma),
        "env_id": cfg.env.env_id,
        "max_episode_steps": int(cfg.env.max_episode_steps or 0),
        "n_states": int(o.n_states),
        "sampling": o.sampling,
        "state_half_width_steps": int(half_width),
        "n_actions": int(traj.n_actions),
    }
    path = save_landscape(landscape_path(cfg, seed, tag), land, meta)
    if verbose:
        print(f"  saved      {path.relative_to(path.parents[3])}  "
              f"({path.stat().st_size / 1e3:.0f} KB)")

    return {"path": path, "landscape": land, "meta": meta,
            "restore": restore, "determinism": determinism, "mc": mc, "picked": picked}


def build_all(cfg: ExperimentConfig, verbose: bool = True, tag: str = "oracle") -> dict[int, dict]:
    t0 = time.time()
    out = {s: build_seed(cfg, s, verbose=verbose, tag=tag) for s in cfg.run.seeds}
    if verbose:
        print(f"\nall seeds done in {time.time() - t0:.1f}s")
    return out


def _cli() -> None:
    base = ExperimentConfig()
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default=base.run.run_name)
    p.add_argument("--seeds", type=int, nargs="+", default=list(base.run.seeds))
    p.add_argument("--checkpoint-fraction", type=float, default=base.oracle.checkpoint_fraction)
    p.add_argument("--n-states", type=int, default=base.oracle.n_states)
    p.add_argument("--sampling", default=base.oracle.sampling, choices=["window", "all", "stratified"])
    p.add_argument("--no-mc", action="store_true")
    a = p.parse_args()

    cfg = ExperimentConfig(
        env=base.env,
        ppo=base.ppo,
        run=dataclasses.replace(base.run, run_name=a.run_name, seeds=tuple(a.seeds)),
        oracle=dataclasses.replace(
            base.oracle,
            checkpoint_fraction=a.checkpoint_fraction,
            n_states=a.n_states,
            sampling=a.sampling,
            run_mc_check=not a.no_mc,
        ),
    )
    build_all(cfg)


if __name__ == "__main__":
    _cli()
