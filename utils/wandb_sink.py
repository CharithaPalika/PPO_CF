"""Weights & Biases sink for ScalarLogger. Entirely optional, never fatal.

OFF unless the environment says otherwise, so notebooks and laptop runs are
completely unaffected:

    PPO_CF_WANDB=1              turn it on
    WANDB_MODE=offline          the default this module sets (see below)
    WANDB_DIR=<path>            where offline runs are written
    PPO_CF_WANDB_PROJECT        project name        (default "ppo-cf")
    PPO_CF_WANDB_GROUP          run group           (default the run name)
    PPO_CF_WANDB_JOB_TYPE       arm / job type
    PPO_CF_WANDB_TAGS           comma-separated

WHY OFFLINE BY DEFAULT. wandb starts a sidecar process per run and talks to it
over a socket under a shared home. Sixteen array tasks starting in the same
second race for it; when that race is lost the run dies before its first step,
records nothing, and the analyse job requeues the same work forever. Offline
mode removes the race instead of tolerating it: rows are written to a local
directory and uploaded later by `slurm/04_sync_wandb.sbatch`.

Every method here swallows its own exceptions. A missing wandb package, a bad
key, a full disk -- all of them print a line and training continues, because
`scalars.csv` already has the data.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


class WandbSink:
    """One wandb run, fed from ScalarLogger rows."""

    def __init__(self, run):
        self._run = run

    def log(self, row: dict) -> None:
        if self._run is None:
            return
        step = row.get("global_step")
        payload = {}
        for k, v in row.items():
            if k == "global_step":
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if fv != fv:                                # drop NaN, wandb charts hate it
                continue
            payload[f"train/{k}"] = fv
        if payload:
            self._run.log(payload, step=int(step) if step is not None else None)

    def close(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None


def make_sinks(cfg, seed: int) -> list:
    """Build the sink list for one training run. Returns [] when disabled."""
    if not _truthy("PPO_CF_WANDB"):
        return []
    try:
        import wandb
    except Exception as e:
        print(f"[wandb] not available ({e!r}) -- CSV only", flush=True)
        return []

    # Offline unless the caller deliberately overrode it.
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_SILENT", "true")
    os.environ.setdefault("WANDB__SERVICE_WAIT", "300")

    run_name = cfg.run.run_name
    tags = [t for t in os.environ.get("PPO_CF_WANDB_TAGS", "").split(",") if t.strip()]
    try:
        run = wandb.init(
            project=os.environ.get("PPO_CF_WANDB_PROJECT", "ppo-cf"),
            group=os.environ.get("PPO_CF_WANDB_GROUP", run_name),
            job_type=os.environ.get("PPO_CF_WANDB_JOB_TYPE", cfg.ppo.pg_mode),
            name=f"{run_name}_seed{seed}",
            tags=tags or None,
            dir=os.environ.get("WANDB_DIR") or None,
            config={**cfg.to_dict(), "seed": seed},
            reinit=True,
        )
    except Exception as e:
        print(f"[wandb] init failed ({e!r}) -- CSV only", flush=True)
        return []

    print(f"[wandb] {os.environ.get('WANDB_MODE')} run {run_name}_seed{seed}", flush=True)
    return [WandbSink(run)]
