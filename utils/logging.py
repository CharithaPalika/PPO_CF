"""Minimal CSV scalar logger. No tensorboard, no wandb -- six notebooks on a
laptop do not need a tracking server, and a CSV is trivially re-readable."""

from __future__ import annotations

import csv
from pathlib import Path


class ScalarLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = None
        self._w = None
        self.rows: list[dict] = []

    def log(self, row: dict) -> None:
        self.rows.append(row)
        if self._w is None:
            self._f = open(self.path, "w", newline="")
            self._w = csv.DictWriter(self._f, fieldnames=list(row.keys()))
            self._w.writeheader()
        self._w.writerow(row)
        self._f.flush()

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None
            self._w = None

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.rows)


def read_scalars(path: Path):
    import pandas as pd
    return pd.read_csv(path)
