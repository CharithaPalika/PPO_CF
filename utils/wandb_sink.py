"""Weights & Biases sink for ScalarLogger. Entirely optional, never fatal.

OFF unless the environment says otherwise, so notebooks and laptop runs are
completely unaffected.  When enabled, it defaults to offline; a launcher may
explicitly request online monitoring (E2 does):

    PPO_CF_WANDB=1              turn it on
    WANDB_MODE=offline|online   offline is the default this module sets
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
directory and uploaded later by `slurm/04_sync_wandb.sbatch`.  Online is
appropriate only when live monitoring is worth the cluster's supported network
and sidecar cost; it is still non-fatal here.

Every method here swallows its own exceptions. A missing wandb package, a bad
key, a full disk -- all of them print a line and training continues, because
`scalars.csv` already has the data.
"""

from __future__ import annotations

import hashlib
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
    # A Slurm retry must attach to the same live curve rather than making a
    # second run with the same display name.  The pipeline's unit identity is
    # stage/arm in `run_name` plus seed, and the digest keeps W&B's run ID
    # syntax independent of path separators used by smoke runs.
    resume = os.environ.get("PPO_CF_WANDB_RESUME", "").strip() or None
    run_id = None
    if resume:
        # W&B does not permit a deleted run ID to be created again. A fixed
        # namespace keeps retries of THIS sweep resumable while allowing a
        # deliberately abandoned/deleted sweep to start with fresh IDs.
        namespace = os.environ.get("PPO_CF_WANDB_RUN_NAMESPACE", "default")
        run_id = hashlib.sha1(
            f"{namespace}:{run_name}:seed:{seed}".encode("utf-8")
        ).hexdigest()[:20]
    # Keep the nested experiment config for reproducibility, and also expose
    # the exact dotted fields that the W&B table groups/filters on. Depending
    # on W&B version, nested dictionaries may otherwise appear only as `ppo`
    # and `env` objects, leaving the `ppo.pg_mode` / `env.env_id` columns null.
    wandb_config = {
        **cfg.to_dict(),
        "seed": seed,
        "ppo.pg_mode": str(cfg.ppo.pg_mode),
        "env.env_id": str(cfg.env.env_id),
        "distill.beta": float(cfg.distill.beta),
        "wandb.run_namespace": os.environ.get("PPO_CF_WANDB_RUN_NAMESPACE", "default"),
    }
    try:
        run = wandb.init(
            project=os.environ.get("PPO_CF_WANDB_PROJECT", "ppo-cf"),
            group=os.environ.get("PPO_CF_WANDB_GROUP", run_name),
            job_type=os.environ.get("PPO_CF_WANDB_JOB_TYPE", cfg.ppo.pg_mode),
            name=f"{run_name}_seed{seed}",
            tags=tags or None,
            dir=os.environ.get("WANDB_DIR") or None,
            config=wandb_config,
            id=run_id,
            resume=resume,
            reinit=True,
        )
    except Exception as e:
        print(f"[wandb] init failed ({e!r}) -- CSV only", flush=True)
        return []

    print(f"[wandb] {os.environ.get('WANDB_MODE')} run {run_name}_seed{seed}", flush=True)
    return [WandbSink(run)]
