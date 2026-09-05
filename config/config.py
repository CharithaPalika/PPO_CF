"""Central configuration for the counterfactual-PPO feasibility pipeline.

Every notebook imports from here. Nothing else in the codebase is allowed to
hard-code a hyperparameter, a path, or a checkpoint fraction -- if you find one,
that is a bug.

Notebook 01 uses EnvConfig + PPOConfig + RunConfig.
Notebooks 02-06 will add their own dataclasses to this module as they are built.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Sequence


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

# Project root = the directory containing this `config` package's parent.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RUNS_DIR = PROJECT_ROOT / "runs"
FIGURES_DIR = PROJECT_ROOT / "figures"


def seed_dir(run_name: str, seed: int) -> Path:
    """runs/<run_name>/seed_<seed>/"""
    return RUNS_DIR / run_name / f"seed_{seed}"


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class EnvConfig:
    env_id: str = "MountainCar-v0"
    n_envs: int = 16                 # parallel copies used for rollout collection

    # 500, NOT the registered 200. This is an exploration knob, not a change to
    # the task. Measured on this codebase (see experiment_plan.md, Run Log):
    #   limit 200  -> first goal reach at ~506k steps, 38 successes in 1M
    #   limit 500  -> first goal reach at ~133k steps, 100% success by 400k
    #   limit 1000 -> never reaches the goal (entropy collapses first)
    # The resulting policy is then evaluated under the STANDARD 200-step limit
    # and scores 100/100 with return -135, so the benchmark is not being softened.
    max_episode_steps: int | None = 500

    # Observation scaling. MountainCar's velocity component has range +/-0.07,
    # which a tanh MLP effectively cannot see unless it is rescaled. Options:
    #   "fixed"   -> affine map from observation_space.low/high onto [-1, 1].
    #                Deterministic, has no running state, and is therefore safe
    #                to reuse verbatim in Notebooks 02-06.
    #   "running" -> running mean/std (classic VecNormalize). Learns better in
    #                some envs but its statistics drift during training, which
    #                makes "the policy at the 30% checkpoint" ambiguous.
    #   "none"    -> raw observations.
    # Default is "fixed" precisely because the counterfactual oracle in NB02
    # needs one unambiguous state -> network-input map.
    obs_norm: str = "fixed"

    # Reward normalisation is deliberately OFF. NB02 computes
    # Q_CF(s,a) = r + gamma * V(s'), which requires r and V to be in the SAME
    # units. Normalising rewards puts V on a rescaled axis and silently
    # corrupts the oracle. Do not turn this on without fixing NB02.
    normalize_reward: bool = False


# --------------------------------------------------------------------------- #
# PPO
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PPOConfig:
    total_timesteps: int = 400_000   # environment steps, summed over all n_envs
    n_steps: int = 32                # rollout length per env -> batch = n_envs * n_steps
    n_epochs: int = 4
    n_minibatches: int = 4

    gamma: float = 0.99
    gae_lambda: float = 0.98

    clip_coef: float = 0.2
    clip_vloss: bool = True
    # 0.003, tuned. MountainCar is unusually sharp about this:
    #   ent_coef >= 0.01 pins the policy near uniform, and a uniform policy has
    #     a MEASURED zero probability of ever reaching the goal (0/2000 episodes,
    #     best position -0.17 vs goal +0.50) -- exploration is not slow, it is
    #     impossible, because success needs temporally correlated actions.
    #   ent_coef = 0.0 collapses to a deterministic bad policy (entropy ~1e-3)
    #     before it ever sees a reward.
    # 0.003 sits in the window where the policy is correlated enough to build
    # momentum but still stochastic enough to keep searching.
    ent_coef: float = 0.003
    vf_coef: float = 0.5

    # --- entropy control -------------------------------------------------- #
    # A FIXED ent_coef is bimodal on MountainCar and that bimodality is fatal,
    # not merely noisy. Measured over 3 seeds at ent_coef=0.003, 400k steps:
    #   seed 0 -> entropy settles near 0.3-1.0, builds momentum, solves (100%)
    #   seed 1 -> entropy collapses to 0.004 by 100k, never sees a reward
    #   seed 2 -> entropy collapses to 0.001 by 200k, never sees a reward
    # Collapse before the first reward is irrecoverable: a deterministic bad
    # policy generates no reward signal, so nothing can push it back out.
    #
    # "adaptive" runs a multiplicative controller on ent_coef to hold the mean
    # policy entropy near a target that anneals from `target_entropy_start` down
    # to `target_entropy_end`. The floor keeps the policy temporally correlated
    # but not degenerate during the reward-free phase; the anneal lets it sharpen
    # once it has something to sharpen towards.
    # The anneal is EVENT-based, not time-based: the entropy floor is held at
    # `target_entropy_start` until the agent reaches the goal for the first time,
    # and only then decays to `target_entropy_end`. A time-based anneal sharpens
    # the policy on a schedule that has nothing to do with whether it has found
    # anything worth sharpening towards, which is precisely the mistake that
    # loses seeds 1 and 2.
    ent_mode: str = "adaptive"            # "fixed" | "adaptive"
    target_entropy_start: float = 0.60    # log(3) = 1.0986 is uniform
    target_entropy_end: float = 0.02
    # Fraction of total training over which the floor decays, once the first
    # goal reach has happened.
    ent_anneal_frac: float = 0.35
    ent_coef_min: float = 1e-5
    ent_coef_max: float = 0.05
    ent_adapt_rate: float = 0.05
    max_grad_norm: float = 0.5

    # Clip actor and critic gradients as two independent groups.
    #
    # This is NOT cosmetic on MountainCar. Every reward is -1, so returns sit
    # around -60 to -100 and the value loss is O(10^3) from the first update.
    # With a single global grad-norm clip the critic's gradient dominates the
    # norm, the whole vector is rescaled by ~1/40, and the policy effectively
    # stops moving (entropy pinned at log 3, approx_kl ~ 1e-6). Clipping the two
    # heads separately keeps V's units intact -- which NB02's oracle depends on
    # -- while letting the policy actually update.
    separate_grad_clip: bool = True

    learning_rate: float = 7e-4
    anneal_lr: bool = True
    target_kl: float | None = None   # None -> never early-stop an epoch

    hidden_sizes: Sequence[int] = (64, 64)
    activation: str = "tanh"

    device: str = "cpu"              # tiny MLP; MPS/CUDA transfer overhead dominates

    @property
    def batch_size_per_update(self) -> int:
        raise NotImplementedError  # needs n_envs; see PPOTrainer


# --------------------------------------------------------------------------- #
# Run / experiment
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RunConfig:
    run_name: str = "nb01_baseline"
    seeds: Sequence[int] = (0, 1, 2)

    # Plan: save checkpoints at ~10%, 30%, 50%, 75%, 100% of training.
    # Fractions are of `total_timesteps`, not of update count.
    checkpoint_fractions: Sequence[float] = (0.10, 0.30, 0.50, 0.75, 1.00)

    # Log a scalar row every N updates.
    log_every_updates: int = 10

    # Record the full per-step trajectory dataset (needed by NB03 / COCOA).
    record_trajectories: bool = True
    # Keep every step. Set >1 to subsample if disk becomes a problem.
    trajectory_stride: int = 1

    torch_deterministic: bool = True


# --------------------------------------------------------------------------- #
# Notebook 02 -- explicit counterfactual oracle
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class OracleConfig:
    # WHICH CHECKPOINT THE ORACLE IS BUILT ON. This is the single most
    # consequential choice in NB02 and it is also overridable at the top of the
    # notebook.
    #
    # The plan says ~30%. NB01 measured that at 30% the policy has learned
    # nothing on any seed (0/50 greedy success; first goal reach at 39-55% of
    # training), and the resulting landscape is 20-200x flatter than at 100%:
    #
    #   mean |A_CF|   10%      30%      50%      75%     100%
    #   seed 0      0.0039   0.0021   0.2396   0.3646   0.4416
    #   seed 1      0.0221   0.0419   0.3694   0.4416   0.4522
    #   seed 2      0.0053   0.0102   0.0108   0.1779   0.2547
    #
    # 0.75 is the first checkpoint with a competent policy and a landscape with
    # real structure. Set this back to 0.30 to reproduce the plan literally --
    # Gate 2 will then report a near-degenerate landscape, which is a finding
    # about the checkpoint, not about the method.
    checkpoint_fraction: float = 0.75

    n_states: int = 500
    seed_for_states: int = 0

    # "window" restricts the evaluation states to rows collected near the
    # checkpoint's own global_step, so the states belong to the policy the
    # landscape is centered by. See oracle/sampling.py.
    sampling: str = "window"                # "window" | "all" | "stratified"
    state_window_frac: float = 0.05         # half-width, as a fraction of total_timesteps

    # Monte-Carlo diagnostic. Not part of the plan's oracle -- it measures how
    # far the one-step Q_CF sits from an estimate that only leans on the critic
    # at the tail, which is what tells you whether a flat landscape means "the
    # actions are equivalent" or "the critic is flat".
    # Rollouts use COMMON RANDOM NUMBERS across actions (see mc_reference), which
    # is what makes 32 enough: the estimate is converged from ~16 (ratio of mean
    # magnitudes 0.900 at R=16 vs 0.891 at R=128; Pearson 0.164 vs 0.149), and
    # two independent CRN seeds at R=64 agree at Pearson 0.976. Without CRN the
    # per-action noise exceeds the advantage being measured and the diagnostic
    # reports disagreement that is entirely its own.
    run_mc_check: bool = True
    mc_n_states: int = 60
    mc_rollouts: int = 32
    mc_horizon: int = 200

    # Gate 2 thresholds. Pre-registered so the verdict is not decided after
    # looking at the numbers.
    spread_threshold: float = 0.01          # |max_a Q - min_a Q| counted as "actions differ"
    min_frac_states_with_spread: float = 0.20


@dataclass(frozen=True)
class ExperimentConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    run: RunConfig = field(default_factory=RunConfig)
    oracle: OracleConfig = field(default_factory=OracleConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))

    def summary(self) -> str:
        b = self.env.n_envs * self.ppo.n_steps
        n_updates = self.ppo.total_timesteps // b
        lines = [
            f"env                {self.env.env_id}  (n_envs={self.env.n_envs}, obs_norm={self.env.obs_norm})",
            f"total_timesteps    {self.ppo.total_timesteps:,}  per seed",
            f"rollout batch      {b}  ({self.env.n_envs} envs x {self.ppo.n_steps} steps)",
            f"updates            {n_updates:,}",
            f"minibatch size     {b // self.ppo.n_minibatches}  x {self.ppo.n_epochs} epochs",
            f"gamma / lambda     {self.ppo.gamma} / {self.ppo.gae_lambda}",
            f"lr                 {self.ppo.learning_rate}  (anneal={self.ppo.anneal_lr})",
            f"ent_coef           {self.ppo.ent_coef}",
            f"seeds              {list(self.run.seeds)}",
            f"checkpoints at     {[f'{f:.0%}' for f in self.run.checkpoint_fractions]}",
        ]
        return "\n".join(lines)


DEFAULT = ExperimentConfig()
