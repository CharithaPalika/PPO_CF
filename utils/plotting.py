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


# --------------------------------------------------------------------------- #
# MiniGrid-specific views
# --------------------------------------------------------------------------- #
#
# The simulator state packed by envs/minigrid_env.py is
#     [ grid.encode().ravel() ,  agent_col, agent_row, agent_dir,
#       carried_type, carried_color, step_count ]
# so the last six entries are readable without decoding the grid.

MG_COL, MG_ROW, MG_DIR, MG_CARRY = -6, -5, -4, -3


def _bin_by_step(step: np.ndarray, values: np.ndarray, n_bins: int):
    """Mean of `values` in `n_bins` equal-width bins of `step`."""
    edges = np.linspace(step.min(), step.max() + 1, n_bins + 1)
    idx = np.clip(np.digitize(step, edges) - 1, 0, n_bins - 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    out = np.full(n_bins, np.nan)
    for b in range(n_bins):
        m = idx == b
        if m.any():
            out[b] = values[m].mean()
    return centres, out


def minigrid_episode_progress(traj) -> pd.DataFrame:
    """Per-episode sub-goal progress for DoorKey.

    Success alone is a terrible progress signal on a sparse task: it sits at zero
    for hundreds of thousands of frames while the policy is in fact getting
    closer. DoorKey has a hard prerequisite -- the agent cannot possibly finish
    without first picking up the key -- so `picked_up_key` moves long before
    `success` does, and a run where neither moves is failing for a different
    reason than a run where only `success` is flat.
    """
    d = traj.data
    eid = d["episode_id"]
    order = np.argsort(eid, kind="stable")
    eid_s = eid[order]
    carrying = (d["sim_state"][order, MG_CARRY] >= 0).astype(np.float64)
    term = d["terminated"][order].astype(np.float64)
    step = d["global_step"][order].astype(np.float64)

    uniq, start = np.unique(eid_s, return_index=True)
    end = np.append(start[1:], len(eid_s))
    rows = []
    for u, a, b in zip(uniq, start, end):
        rows.append({
            "episode_id": int(u),
            "global_step": float(step[a]),
            "picked_up_key": float(carrying[a:b].max()),
            "success": float(term[a:b].max()),
            "length": int(b - a),
        })
    return pd.DataFrame(rows)


def plot_minigrid_progress(traj, n_bins: int = 25, figsize=(12, 3.8)):
    prog = minigrid_episode_progress(traj)
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for ax, key, title, color in [
        (axes[0], "picked_up_key", "Episodes where the key was picked up", "#3b6ea5"),
        (axes[1], "success", "Episodes solved", "#4f8a5b"),
    ]:
        x, y = _bin_by_step(prog["global_step"].to_numpy(), prog[key].to_numpy(), n_bins)
        ax.plot(x, y, lw=1.8, color=color, marker="o", ms=3)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("environment steps")
        ax.set_ylabel("fraction of episodes")
        ax.set_title(title, fontsize=10)

    x, y = _bin_by_step(prog["global_step"].to_numpy(), prog["length"].to_numpy(), n_bins)
    axes[2].plot(x, y, lw=1.8, color="#c4622d", marker="o", ms=3)
    axes[2].set_xlabel("environment steps")
    axes[2].set_ylabel("steps")
    axes[2].set_title("Episode length", fontsize=10)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig, prog


def plot_action_usage(traj, action_names, n_bins: int = 25, figsize=(12, 4)):
    """Mean pi(a) over training, per action.

    Worth watching closely: on Taxi-v4 the policy drove pi(PICKUP) to 0.0016 and
    pi(DROPOFF) below 0.007 while total entropy still read a healthy 1.23, which
    made the failure invisible to every aggregate metric. A per-action view is
    the only place that shows up.
    """
    d = traj.data
    step = d["global_step"].astype(np.float64)
    probs = d["probs"]
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for a, name in enumerate(action_names):
        x, y = _bin_by_step(step, probs[:, a], n_bins)
        axes[0].plot(x, y, lw=1.5, label=name)
    axes[0].axhline(1.0 / probs.shape[1], color="#888", ls=":", lw=1, label="uniform")
    axes[0].set_xlabel("environment steps"); axes[0].set_ylabel("mean pi(a)")
    axes[0].set_title("Action probability over training", fontsize=10)
    axes[0].legend(frameon=False, fontsize=7, ncol=2)

    taken = np.bincount(d["action"].astype(np.int64), minlength=probs.shape[1])
    axes[1].bar(range(len(action_names)), taken / taken.sum(), color="#4f6d7a")
    axes[1].set_xticks(range(len(action_names)))
    axes[1].set_xticklabels(action_names, rotation=45, ha="right", fontsize=7)
    axes[1].set_ylabel("fraction of steps")
    axes[1].set_title("Actions actually taken (whole run)", fontsize=10)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_minigrid_visitation(traj, grid_shape=None, figsize=(12, 3.6)):
    """`grid_shape=None` infers (W, H) from the packed sim_state width.

    The state is [grid.encode().ravel() (W*H*3), col, row, dir, carry_type,
    carry_color, step_count], so W*H = (len(sim_state) - 6) / 3, and MiniGrid
    grids are square. Inferring it keeps this function correct across the
    5x5 / 6x6 / 8x8 curriculum instead of silently mis-binning.
    """
    d = traj.data
    col = d["sim_state"][:, MG_COL].astype(int)
    row = d["sim_state"][:, MG_ROW].astype(int)
    carrying = d["sim_state"][:, MG_CARRY] >= 0
    if grid_shape is None:
        n_cells = (d["sim_state"].shape[1] - 6) // 3
        side = int(round(n_cells ** 0.5))
        grid_shape = (side, side)
    w, h = grid_shape

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, mask, title in [
        (axes[0], np.ones(len(col), bool), "All steps"),
        (axes[1], ~carrying, "Before key pickup"),
        (axes[2], carrying, "Carrying the key"),
    ]:
        if mask.sum() == 0:
            ax.text(0.5, 0.5, "no steps", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, fontsize=10)
            continue
        grid = np.zeros((h, w))
        np.add.at(grid, (row[mask], col[mask]), 1.0)
        im = ax.imshow(grid / grid.sum(), cmap="viridis", origin="upper")
        plt.colorbar(im, ax=ax, label="visit fraction")
        ax.set_title(f"{title}  ({int(mask.sum()):,} steps)", fontsize=10)
        ax.set_xlabel("col"); ax.set_ylabel("row")
    fig.tight_layout()
    return fig


def plot_subgoal_ladder(scalars: dict, figsize=(13, 3.8)):
    """key_rate -> door_rate -> success_rate, from scalars.csv.

    On a sparse task success rate is a terrible progress signal: it can sit at
    exactly zero for millions of frames while the policy is either improving or
    dying, and the two look identical. DoorKey has hard prerequisites -- no goal
    without an open door, no open door without the key -- so the ladder says
    WHICH rung the policy is stuck on, and the rungs need opposite responses:

        all three flat            -> no signal at all; check adv_std_raw
        key rising, door flat     -> stuck at the door (toggle suppressed, or
                                     the door is never approached with the key)
        door rising, success flat -> stuck between the door and the goal;
                                     usually just undertrained
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    panels = [("key_rate_100", "Picked up the key", "#3b6ea5"),
              ("door_rate_100", "Opened the door", "#8a6d3b"),
              ("success_rate_100", "Solved", "#4f8a5b")]
    for ax, (key, title, color) in zip(axes, panels):
        for s, d in scalars.items():
            if key not in d.columns:
                continue
            ax.plot(d["global_step"], d[key], lw=1.5, color=color, label=f"seed {s}")
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("environment steps")
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("fraction of last 100 episodes")
    fig.tight_layout()
    return fig


def plot_signal_diagnostics(scalars: dict, figsize=(13, 3.8)):
    """adv_std_raw, explained_variance and entropy on one row.

    `adv_std_raw` is the spread of the advantages BEFORE whitening. It answers
    the question every other diagnostic confounds: was there anything to learn
    from this rollout? If it sits at the `norm_adv_min_std` floor, the policy is
    not learning slowly -- it is receiving no signal, and any movement in
    entropy, approx_kl or clipfrac during that stretch is numerical noise.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for s, d in scalars.items():
        if "adv_std_raw" in d.columns:
            axes[0].semilogy(d["global_step"], d["adv_std_raw"].clip(lower=1e-20),
                             lw=1.4, label=f"seed {s}")
        axes[1].plot(d["global_step"], d["explained_variance"], lw=1.4, label=f"seed {s}")
        axes[2].plot(d["global_step"], d["entropy"], lw=1.4, label=f"seed {s}")
    for ax, title, ylab in [
        (axes[0], "Advantage spread (raw)", "std before whitening"),
        (axes[1], "Explained variance", "EV"),
        (axes[2], "Policy entropy", "nats"),
    ]:
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("environment steps")
        ax.set_ylabel(ylab)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig
