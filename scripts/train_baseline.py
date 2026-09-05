"""Notebook 01's engine. Runs PPO for one or more seeds.

Usable from the notebook:
    from scripts.train_baseline import run_seeds
    results = run_seeds(cfg)

or from the shell:
    python -m scripts.train_baseline --seeds 0 1 2 --total-timesteps 400000
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

from agents.ppo import PPOTrainer
from config import ExperimentConfig, RUNS_DIR, seed_dir


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


def _cli() -> None:
    base = ExperimentConfig()
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default=base.run.run_name)
    p.add_argument("--seeds", type=int, nargs="+", default=list(base.run.seeds))
    p.add_argument("--total-timesteps", type=int, default=base.ppo.total_timesteps)
    p.add_argument("--n-envs", type=int, default=base.env.n_envs)
    p.add_argument("--ent-coef", type=float, default=base.ppo.ent_coef)
    p.add_argument("--no-trajectories", action="store_true")
    a = p.parse_args()

    cfg = ExperimentConfig(
        env=dataclasses.replace(base.env, n_envs=a.n_envs),
        ppo=dataclasses.replace(base.ppo, total_timesteps=a.total_timesteps, ent_coef=a.ent_coef),
        run=dataclasses.replace(
            base.run,
            run_name=a.run_name,
            seeds=tuple(a.seeds),
            record_trajectories=not a.no_trajectories,
        ),
    )
    print(cfg.summary())
    run_seeds(cfg)


if __name__ == "__main__":
    _cli()
