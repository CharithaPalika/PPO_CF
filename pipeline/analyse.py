#!/usr/bin/env python3
"""Merge the chunks, then either requeue the stage or summarise it and chain on.

The two limits at the top are what stop a requeue loop from running away.
MAX_PASSES is a coarse backstop. STALL_LIMIT is the one that saves you: a pass
that completed nothing will not complete anything next time either. The requeue
loop exists for walltime cuts and dead nodes, where each pass finishes some
work and the remainder shrinks. It is the wrong response to a fault that hits
every unit identically -- a missing dependency, a broken restore, a config that
raises for every arm.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from pipeline.stages import (ANALYSIS_ONLY, ART, CHAIN_NEXT, GROUPS_ALL,   # noqa: E402
                             LEDGER, STAGES, art_key, merge_chunks, pending,
                             run_list, stage_groups)

MAX_PASSES = 40
STALL_LIMIT = 2          # consecutive zero-progress passes tolerated

#: A bootstrap CI over one or two paired seeds is not evidence -- resampling a
#: single difference reproduces that difference, so the CI excludes zero
#: whenever the point estimate is nonzero and every check reads PASS. Below this
#: many pairs the gate reports `insufficient` instead of a verdict. E1 runs 10
#: paired seeds on 6x6 and 5 on 8x8.
MIN_PAIRS = 5


# --------------------------------------------------------------------------- #
# Submission (mirrors slurm/_submit_lib.sh)
# --------------------------------------------------------------------------- #

def submit_stage(stage: str, groups: str, n_chunks: int, part_end: str, pass_no: int) -> None:
    """manifest -> array -> analyse, exactly as slurm/_submit_lib.sh does it.

    THE VARIABLES GO IN THE CHILD ENVIRONMENT, NOT IN sbatch's --export.
    `--export` is a comma-separated list with no escaping, so a value containing
    commas is read as several items and silently truncated: `PROJ_GROUPS=gae,cf,shuf`
    would arrive as `PROJ_GROUPS=gae` with `cf` and `shuf` read as names of variables
    to propagate. They do not exist, so they vanish without a warning and the
    stage runs one arm out of three and exits 0.
    """
    child = dict(os.environ)
    n_groups = len([g for g in groups.split(",") if g.strip()])
    child.update(STAGE=stage, PROJ_GROUPS=groups, N_CHUNKS=str(n_chunks),
                 PROJ_N_GROUPS=str(n_groups), PART_END=part_end or "", PASS_NO=str(pass_no))
    concurrency = os.environ.get("CONCURRENCY", "16")

    def sb(*args: str) -> str:
        r = subprocess.run(["sbatch", "--parsable", "--export=ALL", *args],
                           capture_output=True, text=True, check=True, env=child)
        return r.stdout.strip().split(";")[0]

    if stage in ANALYSIS_ONLY:
        print(f"  submitted analyse {stage} -> {sb('slurm/03_analyse.sbatch')}", flush=True)
        return
    mj = sb("slurm/01_manifest.sbatch")
    rj = sb(f"--dependency=afterok:{mj}", f"--array=0-{n_chunks - 1}%{concurrency}",
            "slurm/02_run_chunks.sbatch")
    # afterany, NOT afterok: a task that hit the walltime still left good rows
    # behind, and noticing what is left is this job's entire purpose.
    aj = sb(f"--dependency=afterany:{rj}", "slurm/03_analyse.sbatch")
    print(f"  submitted {stage}: manifest={mj} runs={rj} analyse={aj}", flush=True)


# --------------------------------------------------------------------------- #
# The summary: Experiments.pdf section 9 "primary learning" + Eq (32)
# --------------------------------------------------------------------------- #

def _paired_diff(df, a: str, b: str, col: str, higher_is_better: bool) -> dict:
    """Seed-level paired difference a - b on `col`, with a bootstrap CI.

    Paired on seed, because the arms share seeds by construction. Pre-registered
    here rather than chosen after looking at the numbers.
    """
    import numpy as np

    # A pooled E2 ledger contains the same seeds for multiple environments and
    # beta values; pair within each (environment, beta) condition rather than
    # collapsing them into an arbitrary cross-condition mean.
    if "beta" in df.columns and df["beta"].nunique() > 1:
        index = ["env_tag", "beta", "seed"]
    elif "env_tag" in df.columns and df["env_tag"].nunique() > 1:
        index = ["env_tag", "seed"]
    else:
        index = "seed"
    wide = df.pivot_table(index=index, columns="group", values=col, aggfunc="first")
    if a not in wide.columns or b not in wide.columns:
        return {"n_pairs": 0}
    pair = wide[[a, b]].dropna()
    if len(pair) == 0:
        return {"n_pairs": 0}
    d = (pair[a] - pair[b]).to_numpy(dtype=float)
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(d, size=len(d), replace=True).mean() for _ in range(10000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "n_pairs": int(len(d)),
        "mean_diff": float(d.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "wins": int((d > 0).sum() if higher_is_better else (d < 0).sum()),
        "better": bool(d.mean() > 0) if higher_is_better else bool(d.mean() < 0),
    }


def summarise(stage: str, groups: list[str]) -> dict:
    """Write the per-arm table, the paired tests, and the success-curve figure."""
    import numpy as np
    import pandas as pd

    led = LEDGER[stage]
    if not led.exists():
        raise FileNotFoundError(f"no ledger at {led}")
    df = pd.read_csv(led)

    cols = [c for c in ("success_auc", "final_success", "frames_to_50", "frames_to_90",
                        "first_success_step", "wall_time_s", "final_entropy",
                        "mean_cf_reward_coverage", "mean_cf_corr_gae")
            if c in df.columns]
    pooled_envs = "env_tag" in df.columns and df["env_tag"].nunique() > 1
    beta_sweep = "beta" in df.columns and df["beta"].nunique() > 1
    summary_groups = (["env_tag", "beta", "group"] if pooled_envs and beta_sweep else
                      ["env_tag", "group"] if pooled_envs else "group")
    per_arm = df.groupby(summary_groups)[cols].agg(["median", "mean", "std", "count"])

    tbl = ART / "tables" / f"{art_key(stage)}_summary.csv"
    tbl.parent.mkdir(parents=True, exist_ok=True)
    per_arm.to_csv(tbl)
    print(f"\n[{stage}] per-arm summary -> {tbl.relative_to(ROOT)}\n", flush=True)
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(per_arm.to_string(), flush=True)

    # ---- Predeclared pairings --------------------------------------------- #
    gates: dict[str, dict] = {}
    if {"distill", "queried"}.issubset(groups):
        comparisons = [("distill", "queried")]
    else:
        comparisons = [("cf", other) for other in ("gae", "shuf")]
    if pooled_envs and beta_sweep:
        gate_frames = [((str(env), float(beta)), sub)
                       for (env, beta), sub in df.groupby(["env_tag", "beta"])]
    elif pooled_envs:
        gate_frames = [((str(tag), None), sub) for tag, sub in df.groupby("env_tag")]
    else:
        gate_frames = [(("", None), df)]
    for (tag, beta), sub in gate_frames:
        prefix = f"{tag}_beta_{beta:g}_" if beta is not None else f"{tag}_" if tag else ""
        if "success_auc" in sub.columns:
            for better, other in comparisons:
                if other in groups and better in groups:
                    r = _paired_diff(sub, better, other, "success_auc", higher_is_better=True)
                    r["insufficient"] = r["n_pairs"] < MIN_PAIRS
                    r["pass"] = (not r["insufficient"] and bool(r.get("better"))
                                 and r.get("ci_lo", 0.0) > 0.0)
                    gates[f"{prefix}{better}_beats_{other}_success_auc"] = r
        if "frames_to_50" in sub.columns:
            for better, other in comparisons:
                if other in groups and better in groups:
                    r = _paired_diff(sub, better, other, "frames_to_50", higher_is_better=False)
                    r["insufficient"] = r["n_pairs"] < MIN_PAIRS
                    r["pass"] = (not r["insufficient"] and bool(r.get("better"))
                                 and r.get("ci_hi", 0.0) < 0.0)
                    gates[f"{prefix}{better}_faster_than_{other}_to_50pct"] = r

    status = {
        "stage": stage,
        "groups": groups,
        "n_units": int(len(df)),
        # E1's gates are REPORTED, not blocking: a failed ordering is a finding
        # about the method, not a reason to skip the next environment.
        "blocking": False,
        "gates": gates,
    }
    sp = ART / "status" / f"{art_key(stage)}.json"
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(status, indent=2))

    print(f"\n[{stage}] paired checks (95% bootstrap CI):", flush=True)
    for name, g in gates.items():
        if not g.get("n_pairs"):
            print(f"    {name:38s} no pairs", flush=True)
            continue
        verdict = ("n<%d" % MIN_PAIRS) if g.get("insufficient") else ("PASS" if g["pass"] else "fail")
        print(f"    {name:38s} {verdict:4s}  "
              f"mean {g['mean_diff']:+.4g}  CI [{g['ci_lo']:+.4g}, {g['ci_hi']:+.4g}]  "
              f"{g['wins']}/{g['n_pairs']} seeds", flush=True)

    _figure(stage, df)
    return status


def _figure(stage: str, df) -> None:
    """Mean success curve per arm, split by E2 environment and beta."""
    import matplotlib
    matplotlib.use("Agg")                      # compute nodes have no display
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    def one(env_name: str, beta: float | None, subdf) -> None:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        grid = None
        for group, sub in subdf.groupby("group"):
            curves = []
            for _, r in sub.iterrows():
                ep_path = ROOT / str(r["out_dir"]) / "episodes.csv"
                if not ep_path.exists():
                    continue
                ep = pd.read_csv(ep_path).sort_values("global_step")
                if not len(ep):
                    continue
                roll = ep["success"].astype(bool).astype(float).rolling(100, min_periods=1).mean()
                if grid is None:
                    grid = np.linspace(0, float(r.get("total_timesteps", ep["global_step"].max())), 400)
                curves.append(np.interp(grid, ep["global_step"].to_numpy(float), roll.to_numpy()))
            if not curves:
                continue
            arr = np.vstack(curves)
            m = arr.mean(0)
            ax.plot(grid, m, label=f"{group} (n={len(curves)})")
            if len(curves) > 1:
                ax.fill_between(grid, np.percentile(arr, 25, axis=0),
                                np.percentile(arr, 75, axis=0), alpha=0.15)

        ax.set_xlabel("environment frames")
        ax.set_ylabel("success rate (100-episode rolling mean)")
        title = f"{stage}: {env_name}" if env_name else stage
        if beta is not None:
            title += f", beta={beta:g}"
        ax.set_title(title)
        ax.set_ylim(-0.02, 1.02)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        suffix = f"_{env_name}" if env_name else ""
        if beta is not None:
            suffix += f"_beta_{beta:g}"
        out = ART / "figures" / f"{art_key(stage)}{suffix}_success.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"\n[{stage}] figure -> {out.relative_to(ROOT)}", flush=True)

    if ("env_tag" in df.columns and df["env_tag"].nunique() > 1
            and "beta" in df.columns and df["beta"].nunique() > 1):
        for (env_name, beta), subdf in df.groupby(["env_tag", "beta"]):
            one(str(env_name), float(beta), subdf)
    elif "env_tag" in df.columns and df["env_tag"].nunique() > 1:
        for env_name, subdf in df.groupby("env_tag"):
            one(str(env_name), None, subdf)
    else:
        one("", None, df)


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--groups", default=",".join(GROUPS_ALL))
    ap.add_argument("--n_chunks", type=int, default=16)
    ap.add_argument("--part_end", default="")
    ap.add_argument("--pass_no", type=int, default=0)
    ap.add_argument("--no_submit", action="store_true",
                    help="never call sbatch (for running this by hand off-cluster)")
    a = ap.parse_args()

    if a.stage not in STAGES:
        print(f"unknown stage {a.stage!r}; known: {sorted(STAGES)}", file=sys.stderr)
        return 1
    glist = [g.strip() for g in a.groups.split(",") if g.strip()]

    # ---- 1. anything left? ------------------------------------------------ #
    if a.stage not in ANALYSIS_ONLY:
        merge_chunks(a.stage)
        left = pending(a.stage, run_list(a.stage, glist))
        print(f"[{a.stage}] {len(left)} units still pending after merge", flush=True)

        sp = ART / "status" / f".{art_key(a.stage)}_pending"
        prev = json.loads(sp.read_text()) if sp.exists() else None
        stalls = (prev["stalls"] + 1) if (prev and prev.get("left") == len(left)) else 0
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps({"left": len(left), "stalls": stalls}))

        if left and stalls >= STALL_LIMIT:
            print(f"[{a.stage}] STOPPING: {len(left)} units pending for {stalls + 1} "
                  f"consecutive passes with ZERO completed in between. That is a fault "
                  f"affecting every unit, not a walltime cut -- requeueing cannot fix it.\n"
                  f"  Read: tail -40 logs/run_*_0.err\n"
                  f"  Nothing is lost: finished work is in the ledger and a resubmission "
                  f"picks up from there once the fault is fixed.", flush=True)
            return 1

        if left:
            nxt = a.pass_no + 1
            if nxt > MAX_PASSES:
                print(f"[{a.stage}] STOPPING after {nxt} passes with {len(left)} pending.",
                      flush=True)
                return 1
            if a.no_submit:
                print(f"[{a.stage}] {len(left)} pending, --no_submit set: not requeueing.",
                      flush=True)
                return 0
            print(f"[{a.stage}] requeueing (pass {nxt})", flush=True)
            submit_stage(a.stage, a.groups, a.n_chunks, a.part_end, nxt)
            return 0

    (ART / "status" / f".{art_key(a.stage)}_pending").unlink(missing_ok=True)

    # ---- 2. the summary --------------------------------------------------- #
    print(f"[{a.stage}] complete -- summarising", flush=True)
    try:
        summarise(a.stage, stage_groups(a.stage, glist))
    except Exception:
        traceback.print_exc()
        print(f"\n[{a.stage}] SUMMARY FAILED. The results are safe in the ledger. Fix it, "
              f"then rerun only this job:\n"
              f"    export STAGE={a.stage} PROJ_GROUPS='{a.groups}' PROJ_N_GROUPS={len(glist)} "
              f"N_CHUNKS={a.n_chunks} PART_END={a.part_end} PASS_NO={a.pass_no}\n"
              f"    sbatch --export=ALL slurm/03_analyse.sbatch", flush=True)
        return 1

    # ---- 3. the next stage ------------------------------------------------ #
    nxt = CHAIN_NEXT.get(a.stage)
    if not nxt or a.stage == a.part_end:
        print(f"[{a.stage}] this part is finished.", flush=True)
        return 0
    if a.no_submit:
        print(f"[{a.stage}] --no_submit set: not chaining to {nxt}.", flush=True)
        return 0
    print(f"[{a.stage}] -> submitting {nxt}", flush=True)
    submit_stage(nxt, a.groups, a.n_chunks, a.part_end, 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
