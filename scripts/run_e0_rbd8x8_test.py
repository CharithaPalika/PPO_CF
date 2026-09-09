#!/usr/bin/env python3
"""E0 -- RedBlueDoors-8x8, PPO-CF, two seeds, the config exactly as it was run.

WHAT THIS IS

`01_ppo_baseline_redbluedoor.ipynb` ran RedBlueDoors-8x8 with
`ENV_CONFIG = "redbluedoors8x8_cf"` and `OVERRIDES = {}` -- the YAML verbatim --
and produced `runs/rbd8x8`, where seed 0 reached success 0.72. That is the
setting this script re-tests, on seed 0 (reproduction) and seed 2 (fresh).

NOTHING IS RETYPED HERE. Every hyperparameter is read from
`config/envs/redbluedoors8x8_cf.yaml` through `pipeline.stages.build_config`,
which is the same function the Slurm pipeline calls. A value copied into this
file would be a value that can drift from the one the cluster runs; the only
things set are the arm (`pg_mode`), the run name and the seed.

    cf_horizon 32   cf_rollouts 2   cf_subsample 0.05   -> branch budget 3.20
    alpha_gae 1.0   alpha_cf 0.5    n_steps 512         2,000,000 frames

USAGE

    python -m scripts.run_e0_rbd8x8_test                 # both seeds, in order
    python -m scripts.run_e0_rbd8x8_test --seeds 2       # one seed
    python -m scripts.run_e0_rbd8x8_test --dry-run       # print the config, run nothing

Runs unchanged on a laptop or inside a Slurm job; `slurm/10_e0_rbd8x8_test.sbatch`
is the cluster entry point and gives one seed per array task.

Results land in runs/e0_reddoorbluedoor_test_cf/seed_<n>/ and one JSON row per
finished seed is appended to artifacts/chunks/, so the ordinary analyse job can
merge them into a ledger.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.stages import ART, STAGES, build_config, execute   # noqa: E402

STAGE = "E0_reddoorbluedoor_test"
ARM = "cf"                       # PPO-CF only; "gae"/"shuf" exist but are not this test


def main() -> int:
    spec = STAGES[STAGE]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=spec["seeds"],
                    help=f"default: {spec['seeds']}")
    ap.add_argument("--arm", default=ARM, choices=("gae", "cf", "shuf"))
    ap.add_argument("--dry-run", action="store_true",
                    help="print the resolved config and exit without training")
    a = ap.parse_args()

    print(f"stage   {STAGE}")
    print(f"config  config/envs/{spec['env_config']}.yaml   (overrides: "
          f"{spec['overrides'] or 'none -- the YAML verbatim'})")
    print(f"arm     {a.arm}")
    print(f"seeds   {list(a.seeds)}\n")
    print(build_config(STAGE, a.arm, a.seeds[0]).summary())

    if a.dry_run:
        return 0

    out = ART / "chunks" / f"{STAGE}_manual.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    failed = []
    for seed in a.seeds:
        print(f"\n{'=' * 70}\n{STAGE}  arm={a.arm}  seed={seed}\n{'=' * 70}", flush=True)
        try:
            row = execute({"stage": STAGE, "group": a.arm, "seed": int(seed)})
        except Exception:
            failed.append(seed)
            traceback.print_exc()
            # Keep going: one bad seed must not cost the other one's slot.
            continue
        with open(out, "a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
            fh.flush()
        print(f"\n  seed {seed}: success {row.get('final_success')}  "
              f"AUC {row.get('success_auc')}  "
              f"cf_reward_coverage(mean) {row.get('mean_cf_reward_coverage')}  "
              f"{row.get('wall_time_s')} s", flush=True)

    print(f"\nall seeds done in {(time.time() - t0) / 60:.1f} min -> {out.relative_to(ROOT)}")
    if failed:
        print(f"FAILED seeds: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
