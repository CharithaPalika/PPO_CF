#!/usr/bin/env python3
"""The smallest end-to-end unit this project has, for slurm/00_verify.sbatch.

An import check passes on a machine where the first backward pass segfaults,
so this builds a real config, runs real PPO updates WITH the counterfactual
oracle (the expensive, native, restore-heavy path -- the one that actually
breaks), and asserts the losses are finite.

    python -m pipeline.smoke --steps 1
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1, help="PPO updates to run")
    ap.add_argument("--env", default="redbluedoors6x6_cf")
    ap.add_argument("--pg_mode", default="cf_all_action")
    a = ap.parse_args()

    from config import make_config
    from agents.ppo import PPOTrainer

    n_envs, n_steps = 4, 64                       # batch 256, divisible by 8 minibatches
    overrides = {
        "env.n_envs": n_envs,
        "ppo.n_steps": n_steps,
        "ppo.total_timesteps": n_envs * n_steps * a.steps,
        "ppo.pg_mode": a.pg_mode,
        "ppo.cf_subsample": 0.02,
        "run.run_name": "_smoke",
        "run.seeds": (0,),
        "run.record_trajectories": False,
        "run.log_every_updates": 1,
        "run.checkpoint_fractions": (1.0,),
    }
    if a.pg_mode == "landscape_distill":
        # Make the two-update verify actually exercise an all-state student
        # payload rather than spending the whole smoke run in E2's real 1,024
        # label warm-up.
        overrides.update({"distill.min_labels_before_use": 4,
                          "distill.beta_ramp_updates": 0,
                          "distill.train_steps_per_rollout": 1})
    cfg = make_config(a.env, **overrides)
    print(cfg.summary(), flush=True)

    trainer = PPOTrainer(cfg, seed=0, progress=True)
    trainer.train()

    rows = trainer.logger.rows
    if not rows:
        print("FAILED: no scalar rows logged", file=sys.stderr)
        return 1
    last = rows[-1]
    for k in ("pg_loss", "v_loss", "entropy"):
        v = float(last.get(k, float("nan")))
        if not math.isfinite(v):
            print(f"FAILED: {k} is not finite ({v})", file=sys.stderr)
            return 1
        print(f"  {k:12s} {v:+.6f}")
    if a.pg_mode.startswith("cf") or a.pg_mode == "landscape_distill":
        cf = {k: v for k, v in last.items() if k.startswith("cf_")}
        print(f"  cf diagnostics: {cf}")
        if not cf:
            print("FAILED: cf arm produced no cf_* diagnostics", file=sys.stderr)
            return 1
    if a.pg_mode in ("cf_queried_perturb", "landscape_distill"):
        payload = float(last.get("perturb_payload_rms", float("nan")))
        if not math.isfinite(payload):
            print(f"FAILED: E2 perturbation payload is not finite ({payload})", file=sys.stderr)
            return 1
        print(f"  perturb rms   {payload:+.6f}")
    print("smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
