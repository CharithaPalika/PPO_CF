#!/usr/bin/env python3
"""Build artifacts/manifests/<STAGE>.json -- the agreed run list for one stage.

Its own job, rather than logic inside each array task, so that all N tasks
partition ONE list written once. If each task enumerated the work itself, a
result landing between two reads would shift every task's slice: two tasks
claim one unit, or the last unit is claimed by nobody and the stage never
finishes.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.stages import (ANALYSIS_ONLY, ART, GROUPS_ALL, STAGES,   # noqa: E402
                             art_key, merge_chunks, pending, run_list)


def interleave(tasks: list[dict], groups: list[str]) -> list[dict]:
    """Round-robin across arms so every chunk gets a comparable workload.

    The natural list is grouped -- all of `gae`, then all of `cf`, then all of
    `shuf`. A contiguous split of that hands every 13-minute gae unit to one
    task and every 5.5-hour cf unit to another, so one task runs for a day
    while another finishes in an hour and its node sits idle. Interleaving
    first lets the chunks stay simple contiguous slices and still carry
    similar loads.

    Bounded `for i in range(max_len)`, not `while any(...)`: the obvious
    while-loop is an infinite loop the moment it forgets to drain, and the
    symptom is a manifest job that hangs until the walltime kills it.
    """
    by = {g: [t for t in tasks if t["group"] == g] for g in groups}
    leftover = [t for t in tasks if t["group"] not in by]
    out: list[dict] = []
    for i in range(max((len(v) for v in by.values()), default=0)):
        for g in groups:
            if i < len(by[g]):
                out.append(by[g][i])
    assert len(out) + len(leftover) == len(tasks), "interleave dropped work"
    return out + leftover


def manifest_path(stage: str) -> Path:
    return ART / "manifests" / f"{art_key(stage)}.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--groups", default=",".join(GROUPS_ALL))
    ap.add_argument("--n_chunks", type=int, default=16)
    a = ap.parse_args()

    if a.stage not in STAGES:
        print(f"unknown stage {a.stage!r}; known: {sorted(STAGES)}", file=sys.stderr)
        return 1

    groups = [g.strip() for g in a.groups.split(",") if g.strip()]
    if a.stage in ANALYSIS_ONLY:
        tasks: list[dict] = []
    else:
        merge_chunks(a.stage)                      # fold in anything a previous pass did
        tasks = interleave(pending(a.stage, run_list(a.stage, groups)), groups)

    p = manifest_path(a.stage)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "stage": a.stage,
        "groups": groups,
        "n_chunks": a.n_chunks,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_tasks": len(tasks),
        "tasks": tasks,
    }, indent=1))

    # READ THIS LINE before walking away: it is the receipt for the --export
    # guard in _prelude.sh. It must say the number of arms you meant to submit.
    print(f"[{a.stage}] {len(groups)} groups: {', '.join(groups)}")
    print(f"[{a.stage}] {len(tasks)} units pending -> {p.relative_to(ROOT)}")
    for g in groups:
        print(f"    {g:14s} {sum(1 for t in tasks if t['group'] == g):4d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
