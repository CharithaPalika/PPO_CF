"""Train PPO on any configured environment, from the shell or a notebook.

The point of this script is that TUNING NEVER REQUIRES EDITING CODE. Pick an
environment config and override any field on the command line:

    # list what is configured
    python -m scripts.train --list

    # rung 1 of the curriculum, exactly as the YAML specifies it
    python -m scripts.train --env doorkey5x5

    # a 200k-frame smoke test before committing to the full run
    python -m scripts.train --env doorkey5x5 --frames 200000 --run-name smoke5x5

    # override anything, dotted, repeatable
    python -m scripts.train --env doorkey8x8 \
        --set ppo.ent_coef=0.02 ppo.n_steps=256 env.layout_seeds=[0,1,2,3]

    # rung 2, warm-started from rung 1 (already the default in the YAML)
    python -m scripts.train --env doorkey6x6

Everything the run used is written to runs/<run_name>/config.json, so a result
is always traceable back to the exact configuration that produced it.

From a notebook:

    from config import make_config
    from scripts.train import run_seeds
    cfg = make_config("doorkey5x5", **{"ppo.ent_coef": 0.02})
    results = run_seeds(cfg)
"""

from __future__ import annotations

import argparse
import ast
import json
import time
from pathlib import Path

from agents.ppo import PPOTrainer
from config import ExperimentConfig, RUNS_DIR, make_config, list_env_configs


def run_seed(cfg: ExperimentConfig, seed: int, progress: bool = True) -> dict:
    print(f"\n=== seed {seed} " + "=" * 52)
    trainer = PPOTrainer(cfg, seed=seed, progress=progress)
    artifacts = trainer.train()
    print(
        f"  done in {artifacts['wall_time_s']:.1f}s | "
        f"{artifacts['n_episodes']:,} episodes | "
        f"first success at step {artifacts['first_success_step']}"
    )
    return artifacts


def run_seeds(cfg: ExperimentConfig, progress: bool = True) -> dict[int, dict]:
    t0 = time.time()
    out: dict[int, dict] = {}
    for seed in cfg.run.seeds:
        out[seed] = run_seed(cfg, seed, progress=progress)

    run_dir = RUNS_DIR / cfg.run.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.json")
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_name": cfg.run.run_name,
                "config": cfg.source,
                "env_id": cfg.env.env_id,
                "seeds": list(cfg.run.seeds),
                "total_wall_time_s": time.time() - t0,
                "per_seed": {
                    str(s): {
                        "wall_time_s": a["wall_time_s"],
                        "n_episodes": a["n_episodes"],
                        "first_success_step": a["first_success_step"],
                        "n_checkpoints": len(a["checkpoints"]),
                    }
                    for s, a in out.items()
                },
            },
            indent=2,
        )
    )
    print(f"\nall seeds done in {time.time() - t0:.1f}s -> {run_dir}")
    return out


def _parse_set(items: list[str]) -> dict:
    """['ppo.ent_coef=0.02', 'env.layout_seeds=[0,1]'] -> dotted override dict."""
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, _, raw = item.partition("=")
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw          # plain strings need no quoting
        out[key.strip()] = value
    return out


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default="doorkey5x5",
                   help="environment config: a file stem in config/envs/, an env_id, or a path")
    p.add_argument("--list", action="store_true", help="list configured environments and exit")
    p.add_argument("--run-name", default=None, help="override run.run_name")
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="override run.seeds")
    p.add_argument("--frames", type=int, default=None, help="override ppo.total_timesteps")
    p.add_argument("--init-from", default=None, help="override run.init_from (curriculum warm start)")
    p.add_argument("--no-trajectories", action="store_true",
                   help="skip the trajectory dataset (much less RAM; NB03+ then has no input)")
    p.add_argument("--set", nargs="+", default=None, metavar="KEY=VALUE",
                   help="override any config field, e.g. ppo.ent_coef=0.02")
    p.add_argument("--dry-run", action="store_true", help="print the resolved config and exit")
    a = p.parse_args()

    if a.list:
        for stem, path in list_env_configs().items():
            cfg = make_config(stem)
            print(f"  {stem:<14} {cfg.env.env_id:<28} {cfg.ppo.total_timesteps:>10,} frames")
        return

    over = _parse_set(a.set)
    if a.run_name is not None:
        over["run.run_name"] = a.run_name
    if a.seeds is not None:
        over["run.seeds"] = tuple(a.seeds)
    if a.frames is not None:
        over["ppo.total_timesteps"] = a.frames
    if a.init_from is not None:
        over["run.init_from"] = None if a.init_from.lower() == "none" else a.init_from
    if a.no_trajectories:
        over["run.record_trajectories"] = False

    cfg = make_config(a.env, **over)
    print(cfg.summary())
    if a.dry_run:
        return
    run_seeds(cfg)


if __name__ == "__main__":
    _cli()
