"""Checkpoint save/load.

A checkpoint must be a self-contained answer to the question Notebook 02 asks:
"given a simulator state s, what is pi(.|s) and V(s)?" That requires the network
weights AND the observation scaler, because the scaler sits between the state and
the network. Saving one without the other is the classic silent-corruption bug in
this kind of pipeline, so `Checkpoint` bundles them and exposes exactly the two
methods NB02+ should call: `probs(states)` and `values(states)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from envs.scaling import BaseScaler, scaler_from_state_dict

if TYPE_CHECKING:  # `agents` imports `dataio`, so keep this out of runtime scope
    from agents.networks import ActorCritic


def checkpoint_path(dirpath: Path, fraction: float) -> Path:
    return Path(dirpath) / f"ckpt_{int(round(fraction * 100)):03d}.pt"


def save_checkpoint(
    path: Path,
    model: "ActorCritic",
    scaler: BaseScaler,
    optimizer: torch.optim.Optimizer | None,
    global_step: int,
    fraction: float,
    extra: dict | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.config_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "global_step": int(global_step),
            "fraction": float(fraction),
            "extra": extra or {},
        },
        path,
    )
    return path


@dataclass
class Checkpoint:
    model: "ActorCritic"
    scaler: BaseScaler
    global_step: int
    fraction: float
    extra: dict
    path: Path

    # -- the interface NB02-06 should use ----------------------------------- #

    def scale(self, raw_obs: np.ndarray) -> torch.Tensor:
        x = self.scaler(np.atleast_2d(np.asarray(raw_obs, dtype=np.float64)))
        return torch.as_tensor(x, dtype=torch.float32)

    @torch.no_grad()
    def probs(self, raw_obs: np.ndarray) -> np.ndarray:
        """pi(.|s) for a batch of RAW observations. Returns (B, K)."""
        return self.model.action_probs(self.scale(raw_obs)).cpu().numpy()

    @torch.no_grad()
    def values(self, raw_obs: np.ndarray) -> np.ndarray:
        """V(s) for a batch of RAW observations. Returns (B,)."""
        return self.model.value(self.scale(raw_obs)).cpu().numpy()

    @torch.no_grad()
    def logits(self, raw_obs: np.ndarray) -> np.ndarray:
        return self.model.logits(self.scale(raw_obs)).cpu().numpy()


def load_checkpoint(path: Path, device: str = "cpu") -> Checkpoint:
    from agents.networks import ActorCritic

    path = Path(path)
    blob = torch.load(path, map_location=device, weights_only=False)
    model = ActorCritic.from_config_dict(blob["model_config"])
    model.load_state_dict(blob["model_state_dict"])
    model.to(device).eval()
    scaler = scaler_from_state_dict(blob["scaler_state_dict"])
    return Checkpoint(
        model=model,
        scaler=scaler,
        global_step=blob["global_step"],
        fraction=blob["fraction"],
        extra=blob.get("extra", {}),
        path=path,
    )


def list_checkpoints(dirpath: Path) -> list[Path]:
    return sorted(Path(dirpath).glob("ckpt_*.pt"))
