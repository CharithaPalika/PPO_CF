"""Minimal CSV scalar logger.

The CSV is the GROUND TRUTH and it is written first, unconditionally. Optional
`sinks` (see `utils.wandb_sink`) get a copy of every row afterwards, each call
wrapped so a sink can never raise into the training loop.

That asymmetry is deliberate. A tracking sidecar that can fail a run turns a
lost upload into a lost experiment -- and on a cluster, into an array task that
records nothing, is seen as pending by the analyse job, and is requeued
forever. A monitoring tool must never be able to fail the thing it monitors.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Protocol, Sequence


class Sink(Protocol):
    """Anything that wants a copy of each logged row."""

    def log(self, row: dict) -> None: ...
    def close(self) -> None: ...


class ScalarLogger:
    def __init__(self, path: Path, sinks: Sequence[Any] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = None
        self._w = None
        self.rows: list[dict] = []
        self.sinks: list[Any] = list(sinks or [])

    def log(self, row: dict) -> None:
        self.rows.append(row)
        if self._w is None:
            self._f = open(self.path, "w", newline="")
            self._w = csv.DictWriter(self._f, fieldnames=list(row.keys()))
            self._w.writeheader()
        self._w.writerow(row)
        self._f.flush()

        for s in self.sinks:
            try:
                s.log(row)
            except Exception as e:                      # never fail a run
                print(f"[logging] sink {type(s).__name__} failed: {e!r}", flush=True)

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None
            self._w = None
        for s in self.sinks:
            try:
                s.close()
            except Exception as e:
                print(f"[logging] sink {type(s).__name__} close failed: {e!r}", flush=True)
        self.sinks = []

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.rows)


def read_scalars(path: Path):
    import pandas as pd
    return pd.read_csv(path)
