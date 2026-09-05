"""Plot helpers for Notebook 01.

Deliberately plain matplotlib, one chart per figure, no seaborn dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED_COLORS = ["#3b6ea5", "#c4622d", "#4f8a5b", "#8a5fa8", "#a8a03f"]


def _smooth(y: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(y) < w:
        return y
    k = np.ones(w) / w
    return np.convolve(y, k, mode="valid")


def plot_learning_curves(
    scalars: dict[int, pd.DataFrame],
    y: str = "mean_return_100",
    ylabel: str | None = None,
    title: str | None = None,
    hline: float | None = None,
    ax: plt.Axes | None = None,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    for i, (seed, df) in enumerate(sorted(scalars.items())):
        ax.plot(
            df["global_step"], df[y],
            color=SEED_COLORS[i % len(SEED_COLORS)],
            lw=1.4, label=f"seed {seed}",
        )
    if hline is not None:
        ax.axhline(hline, color="#888", ls="--", lw=1, zorder=0)
    ax.set_xlabel("environment steps")
    ax.set_ylabel(ylabel or y)
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    return ax


def plot_episode_returns(
    episodes: dict[int, pd.DataFrame],
    window: int = 50,
    ax: plt.Axes | None = None,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    for i, (seed, df) in enumerate(sorted(episodes.items())):
        y = _smooth(df["return"].to_numpy(), window)
        x = df["global_step"].to_numpy()[len(df) - len(y):]
        ax.plot(x, y, color=SEED_COLORS[i % len(SEED_COLORS)], lw=1.3, label=f"seed {seed}")
    ax.set_xlabel("environment steps")
    ax.set_ylabel(f"episode return (mean of {window})")
    ax.set_title("Return curve")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    return ax


def plot_grid(scalars: dict[int, pd.DataFrame], keys: Sequence[str], ncols: int = 3, figsize=(14, 7)):
    n = len(keys)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()
    for ax, k in zip(axes, keys):
        plot_learning_curves(scalars, y=k, ax=ax)
        ax.set_title(k, fontsize=10)
        ax.get_legend().remove()
    for ax in axes[n:]:
        ax.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def plot_state_visitation(raw_obs: np.ndarray, ax: plt.Axes | None = None, bins: int = 80):
    """2-D histogram of visited (position, velocity). Sanity check for NB02's
    state sampling: the 500 evaluation states must come from a region the policy
    actually visits."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4))
    h = ax.hist2d(raw_obs[:, 0], raw_obs[:, 1], bins=bins, cmap="viridis", norm="log")
    plt.colorbar(h[3], ax=ax, label="visits (log)")
    ax.set_xlabel("position")
    ax.set_ylabel("velocity")
    ax.set_title("State visitation")
    return ax


def savefig(fig, path: Path, dpi: int = 150) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


# --------------------------------------------------------------------------- #
# Notebook 02 -- counterfactual landscape
# --------------------------------------------------------------------------- #

ACTION_COLORS = ["#3b6ea5", "#c4622d", "#4f8a5b", "#8a5fa8"]


def plot_landscape_distributions(land, action_names=None, figsize=(13, 3.6)):
    """Plan output #2: A_CF distributions and the per-state advantage spread."""
    a_cf = land.data["a_cf"]
    K = a_cf.shape[1]
    names = action_names or [f"action {a}" for a in range(K)]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for a in range(K):
        axes[0].hist(a_cf[:, a], bins=60, histtype="step", lw=1.6,
                     color=ACTION_COLORS[a % len(ACTION_COLORS)], label=names[a])
    axes[0].axvline(0, color="#888", lw=1, ls="--")
    axes[0].set_xlabel("A_CF(s, a)"); axes[0].set_ylabel("states")
    axes[0].set_title("Per-action advantage")
    axes[0].legend(frameon=False, fontsize=8)

    spread = land.spread()
    axes[1].hist(spread, bins=60, color="#4f6d7a")
    axes[1].set_xlabel("max_a Q - min_a Q"); axes[1].set_ylabel("states")
    axes[1].set_title(f"Advantage spread (median {np.median(spread):.3f})")

    best = land.best_action()
    counts = np.bincount(best, minlength=K)
    axes[2].bar(range(K), counts / counts.sum(),
                color=[ACTION_COLORS[a % len(ACTION_COLORS)] for a in range(K)])
    axes[2].axhline(1 / K, color="#888", lw=1, ls="--")
    axes[2].set_xticks(range(K)); axes[2].set_xticklabels(names, fontsize=8)
    axes[2].set_ylabel("fraction of states")
    axes[2].set_title(f"argmax A_CF  (chance = {1/K:.2f})")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_state_space_landscape(land, figsize=(13, 3.6)):
    """Where in the state space the landscape has structure."""
    obs = land.data["raw_obs"]
    spread = land.spread()
    best = land.best_action()

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    s0 = axes[0].scatter(obs[:, 0], obs[:, 1], c=spread, s=12, cmap="viridis")
    plt.colorbar(s0, ax=axes[0], label="Q spread")
    axes[0].set_title("Advantage spread over state space")

    s1 = axes[1].scatter(obs[:, 0], obs[:, 1], c=best, s=12,
                         cmap=plt.matplotlib.colors.ListedColormap(
                             ACTION_COLORS[: land.n_actions]))
    plt.colorbar(s1, ax=axes[1], label="argmax A_CF", ticks=range(land.n_actions))
    axes[1].set_title("Best counterfactual action")

    s2 = axes[2].scatter(obs[:, 0], obs[:, 1], c=land.data["v_pi"], s=12, cmap="magma")
    plt.colorbar(s2, ax=axes[2], label="V_pi(s)")
    axes[2].set_title("Policy value")

    for ax in axes:
        ax.set_xlabel("position"); ax.set_ylabel("velocity")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_mc_comparison(land, figsize=(12, 3.8)):
    """One-step oracle vs Monte-Carlo reference: is A_CF magnitude trustworthy?"""
    d = land.data
    if "q_mc" not in d:
        raise KeyError("no MC diagnostic in this landscape (oracle.run_mc_check was False)")
    m = len(d["q_mc"])
    q_cf, q_mc = d["q_cf"][:m], d["q_mc"]
    a_cf, a_mc = d["a_cf"][:m], d["a_mc"]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].scatter(q_mc.ravel(), q_cf.ravel(), s=10, alpha=0.6, color="#3b6ea5")
    lim = [min(q_mc.min(), q_cf.min()), max(q_mc.max(), q_cf.max())]
    axes[0].plot(lim, lim, color="#888", lw=1, ls="--")
    axes[0].set_xlabel("Q_MC (rollout)"); axes[0].set_ylabel("Q_CF (one-step)")
    axes[0].set_title("Action values")

    for a in range(land.n_actions):
        axes[1].scatter(a_mc[:, a], a_cf[:, a], s=10, alpha=0.6,
                        color=ACTION_COLORS[a % len(ACTION_COLORS)], label=f"action {a}")
    lim = [min(a_mc.min(), a_cf.min()), max(a_mc.max(), a_cf.max())]
    axes[1].plot(lim, lim, color="#888", lw=1, ls="--")
    axes[1].set_xlabel("A_MC"); axes[1].set_ylabel("A_CF")
    axes[1].set_title("Advantages")
    axes[1].legend(frameon=False, fontsize=8)

    pi = d["pi"][:m]
    axes[2].scatter((pi * q_mc).sum(1), d["v_critic"][:m], s=12, alpha=0.7, color="#c4622d")
    lim = [min((pi * q_mc).sum(1).min(), d["v_critic"][:m].min()),
           max((pi * q_mc).sum(1).max(), d["v_critic"][:m].max())]
    axes[2].plot(lim, lim, color="#888", lw=1, ls="--")
    axes[2].set_xlabel("V^pi from rollouts"); axes[2].set_ylabel("V_critic(s)")
    axes[2].set_title("Critic calibration")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
