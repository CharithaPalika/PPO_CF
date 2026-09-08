#!/usr/bin/env python3
"""Process one chunk of a stage manifest. One invocation per Slurm array task.

Three rules are encoded here.

  JSON Lines, not CSV.  Rows from different arms do not carry the same columns
  (only CF arms emit cf_* diagnostics). Appending to a shared CSV with
  header=False writes values positionally under whatever header the first row
  created, so a row with a missing column shifts every later field one column
  left. Nothing errors; it surfaces weeks later as a string in a numeric column.

  One file per task, never a shared one.  Sixteen processes appending to a
  single file is how you get interleaved half-lines.

  Never fail the task for one bad unit, and exit 0 regardless.  Unfinished work
  is normal and is the analyse job's business; a non-zero exit here only makes
  sacct look alarming without changing what happens next.
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

from pipeline.manifest import manifest_path        # noqa: E402
from pipeline.stages import ART, art_key, execute  # noqa: E402


def contiguous_slice(items: list, chunk_id: int, n_chunks: int) -> list:
    """Even split, remainder spread over the first chunks. Every task computes
    the same sizes from the same list, so the slices partition it exactly."""
    base, rem = divmod(len(items), n_chunks)
    sizes = [base + (1 if i < rem else 0) for i in range(n_chunks)]
    start = sum(sizes[:chunk_id])
    return items[start:start + sizes[chunk_id]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--chunk_id", type=int, required=True)
    ap.add_argument("--n_chunks", type=int, required=True)
    ap.add_argument("--minutes", type=float, default=690.0,
                    help="stop STARTING new work after this long (walltime minus margin)")
    ap.add_argument("--margin_min", type=float, default=360.0,
                    help="do not start a unit unless this much budget remains; sized "
                         "from the SLOWEST unit in the stage (RBD-8x8 CF, ~5.5 h), "
                         "not the average")
    a = ap.parse_args()

    mp = manifest_path(a.stage)
    if not mp.exists():
        print(f"no manifest at {mp} -- run slurm/01_manifest.sbatch first", file=sys.stderr)
        return 1
    mine = contiguous_slice(json.loads(mp.read_text())["tasks"], a.chunk_id, a.n_chunks)

    out = ART / "chunks" / f"{art_key(a.stage)}_chunk_{a.chunk_id:03d}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{a.stage} chunk {a.chunk_id}/{a.n_chunks}] {len(mine)} units mine", flush=True)
    if not mine:
        out.touch()
        return 0

    t0 = time.time()
    done = failed = skipped = 0
    with open(out, "a") as fh:                     # append: a requeued task adds to it
        for k, task in enumerate(mine, 1):
            left = a.minutes - (time.time() - t0) / 60.0
            if left < a.margin_min:
                skipped = len(mine) - k + 1
                print(f"[{a.stage} chunk {a.chunk_id}] {left:.1f} min left, {skipped} not "
                      f"started -- the analyse job will requeue them", flush=True)
                break
            print(f"\n[{a.stage} chunk {a.chunk_id}] ({k}/{len(mine)}) {task} "
                  f"({left:.0f} min left)", flush=True)
            try:
                fh.write(json.dumps(execute(task), default=str) + "\n")
                fh.flush()                          # a task killed later keeps what it finished
                done += 1
            except Exception:
                failed += 1
                print(f"*** FAILED {task} ***", flush=True)
                traceback.print_exc()
                # Keep going. One bad unit must not cost the other work in this
                # slot; its row is simply absent and the next pass picks it up.

    print(f"\n[{a.stage} chunk {a.chunk_id}] done={done} failed={failed} skipped={skipped} "
          f"in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0                                        # always 0: see the module docstring


if __name__ == "__main__":
    raise SystemExit(main())
