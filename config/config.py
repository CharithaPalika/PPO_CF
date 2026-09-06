"""Central configuration for the counterfactual-PPO pipeline.

Structure
---------
This module defines the *schema* (dataclasses) and the *loader*. It contains no
per-environment values at all -- those live one file per environment in
`config/envs/*.yaml`, so adding or tuning an environment never means editing
Python.

    config/config.py        <- schema + loader (this file)
    config/envs/*.yaml      <- one file per environment

Usage
-----
    from config import make_config, list_env_configs

    cfg = make_config("doorkey5x5")                       # by file stem
    cfg = make_config("MiniGrid-DoorKey-5x5-v0")          # or by env_id
    cfg = make_config("doorkey5x5", ppo={"ent_coef": 0.02})
    cfg = make_config("doorkey5x5", **{"ppo.ent_coef": 0.02})   # dotted form

Nothing else in the codebase is allowed to hard-code a hyperparameter, a path,
or a checkpoint fraction. If you find one, that is a bug.
"""

from __future__ import annotations

import dataclasses as _dc
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Sequence


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RUNS_DIR = PROJECT_ROOT / "runs"
FIGURES_DIR = PROJECT_ROOT / "figures"
ENV_CONFIG_DIR = Path(__file__).resolve().parent / "envs"


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

    # None keeps the environment's own limit. MiniGrid sets `unwrapped.max_steps`
    # internally rather than through registration, so None is correct there.
    # On MountainCar 500 (not the registered 200) is an exploration knob; the
    # resulting policy is still evaluated at 200.
    max_episode_steps: int | None = 500

    # Observation scaling: "fixed" | "running" | "onehot" | "image" | "none".
    # The oracle needs ONE unambiguous state -> network-input map, which is why
    # the default is stateless.
    obs_norm: str = "fixed"

    # MiniGrid only. False gives the default 7x7 egocentric view (a POMDP).
    # True stacks FullyObsWrapper and keeps the problem Markov, which the
    # counterfactual oracle requires.
    fully_observable: bool = True

    # --- layout control (MiniGrid) ---------------------------------------- #
    # MiniGrid regenerates the layout on every reset, so plain PPO is being
    # asked to GENERALISE across layouts. Nothing in NB02-06 needs that: the
    # oracle restores specific states under a specific policy.
    #
    #   layout_seeds = None        -> a fresh random layout every episode
    #   layout_seeds = [0]         -> one fixed layout, forever
    #   layout_seeds = [0, ..., 7] -> a fixed pool of 8 layouts
    #
    # `layout_seed_mode` decides how the pool is drawn from:
    #   "cycle"  -> each env walks the pool in order (deterministic coverage)
    #   "random" -> uniform draw per episode
    layout_seeds: Sequence[int] | None = None
    layout_seed_mode: str = "cycle"

    # Reward normalisation is deliberately OFF. NB02 computes
    # Q_CF(s,a) = r + gamma * V(s'), which needs r and V in the SAME units.
    normalize_reward: bool = False


# --------------------------------------------------------------------------- #
# Reward shaping / exploration bonuses
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RewardConfig:
    """Optional additions to the environment reward. Both default to OFF.

    ORACLE WARNING. The critic learns V for whatever reward it was trained on,
    and A_CF = r + gamma*V(s') - V(s) is defined against that same reward.

      * Potential-based shaping is RECOVERABLE: with F = gamma*Phi(s') - Phi(s),
        V_shaped(s) = V_true(s) - Phi(s) exactly, so the oracle can undo it.
      * A count bonus is NOT recoverable. It changes the MDP. Only use it if it
        is fully annealed to zero before the checkpoint the oracle is built on;
        `count_bonus_anneal_frac` enforces the anneal, and the trainer asserts
        the bonus is zero at checkpoint time when `assert_zero_at_checkpoints`.
    """

    # Potential-based shaping, Phi(s) = w_key*carrying_key + w_door*door_is_open.
    # Both sub-goals are hard prerequisites in DoorKey, which is what makes a
    # potential over them well-motivated rather than arbitrary.
    potential_shaping: bool = False
    potential_key: float = 0.0
    potential_door: float = 0.0

    # Count-based novelty bonus: coef / sqrt(N(s_abstract)), where the abstract
    # state is (agent_col, agent_row, agent_dir, carrying_key, door_open).
    count_bonus_coef: float = 0.0
    # Linear decay of the coefficient to zero by this fraction of training.
    count_bonus_anneal_frac: float = 0.5
    assert_zero_at_checkpoints: bool = True

    @property
    def active(self) -> bool:
        return bool(self.potential_shaping) or self.count_bonus_coef > 0.0


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
    ent_coef: float = 0.003
    vf_coef: float = 0.5

    # --- advantage normalisation ------------------------------------------ #
    # "minibatch" (the CleanRL default), "batch" (normalise once over the whole
    # rollout), or "none".
    #
    # `norm_adv_min_std` is not cosmetic. On a sparse task the critic converges
    # to V = 0 everywhere and the advantages of a reward-free rollout are ~1e-10
    # of pure float noise. Dividing by their own std rescales that noise to unit
    # variance and feeds it to the policy gradient -- which is exactly what
    # random-walked the DoorKey-8x8 3M run (clipfrac ~0.10 with v_loss ~1e-19).
    # Below this threshold the batch is treated as having no signal and the
    # advantages are left alone.
    norm_adv: str = "batch"
    norm_adv_min_std: float = 1e-6

    # --- policy gradient: baseline PPO vs the counterfactual oracle -------- #
    # "gae"           -> textbook PPO on the sampled GAE advantage.
    # "cf_all_action" -> PPO-CF. Every rollout state is restored in a probe
    #   environment and stepped through ALL K actions to get the exact one-step
    #   Q_CF(s,a) = r(s,a) + gamma*V_phi(s'_a); the policy loss is then a sum
    #   over all actions weighted by pi, so the variance from action sampling
    #   disappears. This is the CEILING of the whole research direction: if the
    #   exact oracle does not beat GAE, no learned approximation of it can.
    #
    # Cost: K restores + K env steps per collected transition. Measured on
    # DoorKey-5x5 (K=7): ~220 collected steps/s with cf_restore="exact",
    # ~380 with "fast", against ~1,900 for plain PPO.
    pg_mode: str = "gae"
    # "exact" restores via envs.env_pool.set_sim_state, the path NB02 validated.
    # "fast" skips the env.reset() inside it (~2x quicker, MiniGrid only);
    # oracle.online.check_restore_equivalence asserts the two are identical.
    cf_restore: str = "exact"
    # Replay the recorded action from the recorded state on the first rollout
    # and assert it reproduces the recorded reward and successor exactly. Costs
    # one rollout's worth of nothing, and catches the one failure mode that is
    # otherwise invisible: a broken restore yields plausible, wrong A_CF.
    cf_validate: bool = True

    # --- entropy control -------------------------------------------------- #
    # "fixed" uses `ent_coef` unchanged. "adaptive" runs a multiplicative
    # controller holding mean policy entropy near a target that anneals from
    # `target_entropy_frac_start` to `target_entropy_frac_end` IN PROPORTION TO
    # MEASURED SUCCESS, not on a wall-clock schedule -- a single lucky success
    # must not be allowed to sharpen the policy (that is how the Taxi run died).
    # Targets are fractions of log(K) so the same value means the same thing at
    # K = 3, 6 and 7.
    ent_mode: str = "adaptive"
    target_entropy_frac_start: float = 0.55
    target_entropy_frac_end: float = 0.02
    ent_anneal_full_rate: float = 0.80

    # --- per-action probability floor ------------------------------------- #
    # pi = (1 - eps)*softmax + eps/K, so pi(a|s) >= eps/K for EVERY action.
    # Entropy regularisation cannot express this constraint: on Taxi the policy
    # held entropy 1.23 of log 6 while pi(PICKUP) averaged 0.0016, which makes
    # success impossible. Rides the same success-proportional schedule.
    prob_floor_start: float = 0.0
    prob_floor_end: float = 0.0
    ent_success_ema: float = 0.02
    ent_coef_min: float = 1e-5
    ent_coef_max: float = 0.05
    ent_adapt_rate: float = 0.05

    max_grad_norm: float = 0.5
    # Clip actor and critic gradients as two independent groups. On MountainCar
    # every reward is -1, the value loss is O(10^3) from the first update, and
    # under a single global clip the critic's gradient dominates the norm and
    # the policy stops moving entirely. Ignored when `share_encoder` is True.
    separate_grad_clip: bool = True

    learning_rate: float = 7e-4
    anneal_lr: bool = True
    target_kl: float | None = None   # None -> never early-stop an epoch

    hidden_sizes: Sequence[int] = (64, 64)
    activation: str = "tanh"
    # "mlp" -> hidden_sizes MLP on the scaled observation.
    # "cnn" -> the reference MiniGrid encoder (agents.networks.MiniGridCNN),
    #          after which hidden_sizes is the head, i.e. (64,).
    encoder: str = "mlp"
    # Share one encoder between actor and critic (what rl-starter-files does)
    # instead of one per head. Sharing roughly halves the parameter count and
    # gives the policy features shaped by the value loss too, which helps when
    # reward is sparse; separate trunks keep V and pi cleanly independent, which
    # is what NB02/NB06 reason about. Forces a single global grad clip.
    share_encoder: bool = False
    optim_eps: float = 1e-5

    device: str = "cpu"


# --------------------------------------------------------------------------- #
# Run / experiment
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RunConfig:
    run_name: str = "nb01_baseline"
    seeds: Sequence[int] = (0, 1, 2)

    checkpoint_fractions: Sequence[float] = (0.10, 0.30, 0.50, 0.75, 1.00)
    log_every_updates: int = 10

    record_trajectories: bool = True
    trajectory_stride: int = 1
    compact_trajectory_dtypes: bool = False

    # --- curriculum -------------------------------------------------------- #
    # Path to a checkpoint to warm-start from, e.g. the 5x5 run's ckpt_100.pt
    # when training 6x6. Relative paths resolve against PROJECT_ROOT.
    # "{seed}" in the path is substituted with the current seed.
    init_from: str | None = None
    init_load_optimizer: bool = False
    # Warm-starting across grid sizes changes the encoder's output width, so
    # incompatible tensors are skipped rather than raising. The trainer prints
    # exactly which tensors were loaded and which were reinitialised.
    init_strict: bool = False

    # Periodic greedy evaluation logged into scalars.csv. 0 disables it.
    # Costs `eval_episodes` episodes of wall time every N updates.
    eval_every_updates: int = 0
    eval_episodes: int = 20

    torch_deterministic: bool = True


# --------------------------------------------------------------------------- #
# Notebook 02 -- explicit counterfactual oracle
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class OracleConfig:
    # The single most consequential choice in NB02. The plan says ~30%; NB01/02
    # measured that at 30% the MountainCar critic is a literal constant and the
    # landscape is zero to four decimal places, so 0.75 is the first checkpoint
    # with real structure. Set back to 0.30 to reproduce the plan literally.
    checkpoint_fraction: float = 0.75

    n_states: int = 500
    seed_for_states: int = 0

    sampling: str = "window"                # "window" | "all" | "stratified"
    state_window_frac: float = 0.05

    # Monte-Carlo diagnostic with common random numbers across actions. Without
    # CRN the per-action noise exceeds the advantage being measured.
    run_mc_check: bool = True
    mc_n_states: int = 60
    mc_rollouts: int = 32
    mc_horizon: int = 200

    # Gate 2 thresholds, pre-registered.
    spread_threshold: float = 0.01
    min_frac_states_with_spread: float = 0.20


@dataclass(frozen=True)
class ExperimentConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    run: RunConfig = field(default_factory=RunConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    oracle: OracleConfig = field(default_factory=OracleConfig)
    #: name of the YAML file this came from, for provenance in config.json
    source: str = "defaults"

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))

    @property
    def batch_size(self) -> int:
        return self.env.n_envs * self.ppo.n_steps

    def summary(self) -> str:
        b = self.batch_size
        n_updates = self.ppo.total_timesteps // b
        layout = ("random every episode" if self.env.layout_seeds is None
                  else f"{len(self.env.layout_seeds)} fixed ({self.env.layout_seed_mode})")
        lines = [
            f"config             {self.source}",
            f"env                {self.env.env_id}  (n_envs={self.env.n_envs}, obs_norm={self.env.obs_norm})",
            f"layouts            {layout}",
            f"total_timesteps    {self.ppo.total_timesteps:,}  per seed",
            f"rollout batch      {b}  ({self.env.n_envs} envs x {self.ppo.n_steps} steps)",
            f"updates            {n_updates:,}",
            f"minibatch size     {b // self.ppo.n_minibatches}  x {self.ppo.n_epochs} epochs",
            f"gamma / lambda     {self.ppo.gamma} / {self.ppo.gae_lambda}",
            f"lr                 {self.ppo.learning_rate}  (anneal={self.ppo.anneal_lr})",
            f"encoder            {self.ppo.encoder}  (shared={self.ppo.share_encoder})",
            f"entropy            {self.ppo.ent_mode}, coef {self.ppo.ent_coef}",
            f"prob floor         {self.ppo.prob_floor_start} -> {self.ppo.prob_floor_end}  (0 = off)",
            f"adv norm           {self.ppo.norm_adv}  (min_std {self.ppo.norm_adv_min_std:g})",
            f"policy gradient    {self.ppo.pg_mode}"
            + (f"  (oracle restore={self.ppo.cf_restore}, validate={self.ppo.cf_validate})"
               if self.ppo.pg_mode == "cf_all_action" else ""),
            f"reward shaping     {'on' if self.reward.active else 'off'}",
            f"warm start         {self.run.init_from or 'none'}",
            f"seeds              {list(self.run.seeds)}",
            f"checkpoints at     {[f'{f:.0%}' for f in self.run.checkpoint_fractions]}",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# YAML loading
# --------------------------------------------------------------------------- #

_SECTIONS = {"env": EnvConfig, "ppo": PPOConfig, "run": RunConfig,
             "reward": RewardConfig, "oracle": OracleConfig}


def _load_yaml(path: Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def list_env_configs() -> dict[str, Path]:
    """{stem: path} for every environment config file."""
    return {p.stem: p for p in sorted(ENV_CONFIG_DIR.glob("*.yaml"))}


def find_env_config(name: str) -> Path:
    """Resolve `name` to a YAML path, by file stem or by the env_id inside it."""
    available = list_env_configs()
    if name in available:
        return available[name]
    p = Path(name)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return p
    for stem, path in available.items():
        blob = _load_yaml(path)
        if blob.get("env", {}).get("env_id") == name:
            return path
    raise KeyError(
        f"no environment config for {name!r}. Available: {sorted(available)} "
        f"(or pass a path to a .yaml file)"
    )


def _split_dotted(overrides: dict[str, Any]) -> dict[str, dict]:
    """{'ppo.ent_coef': 0.02, 'ppo': {...}} -> {'ppo': {...merged...}}"""
    out: dict[str, dict] = {k: {} for k in _SECTIONS}
    for key, value in overrides.items():
        if "." in key:
            section, _, fieldname = key.partition(".")
            if section not in out:
                raise KeyError(f"unknown config section {section!r} in override {key!r}")
            out[section][fieldname] = value
        elif key in out:
            if not isinstance(value, dict):
                raise TypeError(f"override {key!r} must be a dict of fields")
            out[key].update(value)
        else:
            raise KeyError(f"unknown config section {key!r}")
    return out


def _coerce(cls, values: dict) -> dict:
    """Light type coercion so YAML strings/lists land in the right shapes."""
    types = {f.name: f.type for f in _dc.fields(cls)}
    out = {}
    for k, v in values.items():
        if k not in types:
            raise KeyError(f"{cls.__name__} has no field {k!r}")
        if isinstance(v, list):
            v = tuple(v)
        out[k] = v
    return out


def make_config(name: str = "mountaincar", **overrides) -> ExperimentConfig:
    """Build an ExperimentConfig from `config/envs/<name>.yaml`.

    `overrides` accepts either section dicts or dotted keys:
        make_config("doorkey5x5", ppo={"ent_coef": 0.02}, run={"seeds": (0, 1)})
        make_config("doorkey5x5", **{"ppo.ent_coef": 0.02})
    """
    path = find_env_config(name)
    blob = _load_yaml(path)
    over = _split_dotted(overrides)

    kwargs: dict[str, Any] = {}
    for section, cls in _SECTIONS.items():
        merged = {**(blob.get(section) or {}), **over[section]}
        kwargs[section] = cls(**_coerce(cls, merged))

    cfg = ExperimentConfig(source=path.stem, **kwargs)
    _validate(cfg)
    return cfg


def _validate(cfg: ExperimentConfig) -> None:
    b = cfg.batch_size
    if b % cfg.ppo.n_minibatches != 0:
        raise ValueError(
            f"batch {b} (n_envs {cfg.env.n_envs} x n_steps {cfg.ppo.n_steps}) is not "
            f"divisible by n_minibatches {cfg.ppo.n_minibatches}"
        )
    if cfg.ppo.total_timesteps < b:
        raise ValueError(f"total_timesteps {cfg.ppo.total_timesteps} < one batch ({b})")
    if cfg.ppo.norm_adv not in ("minibatch", "batch", "none"):
        raise ValueError(f"norm_adv must be minibatch|batch|none, got {cfg.ppo.norm_adv!r}")
    if cfg.ppo.pg_mode not in ("gae", "cf_all_action"):
        raise ValueError(f"pg_mode must be gae|cf_all_action, got {cfg.ppo.pg_mode!r}")
    if cfg.ppo.cf_restore not in ("exact", "fast"):
        raise ValueError(f"cf_restore must be exact|fast, got {cfg.ppo.cf_restore!r}")
    if cfg.ppo.pg_mode == "cf_all_action" and cfg.reward.active:
        raise ValueError(
            "pg_mode='cf_all_action' with reward shaping on: the oracle would compute "
            "Q_CF from the RAW environment reward while the critic was trained on the "
            "shaped one, so A_CF would mix two different reward functions."
        )
    if cfg.ppo.ent_mode not in ("fixed", "adaptive"):
        raise ValueError(f"ent_mode must be fixed|adaptive, got {cfg.ppo.ent_mode!r}")
    if cfg.env.layout_seed_mode not in ("cycle", "random"):
        raise ValueError(f"layout_seed_mode must be cycle|random, got {cfg.env.layout_seed_mode!r}")
    if cfg.ppo.encoder == "cnn" and cfg.env.obs_norm != "image":
        raise ValueError("encoder='cnn' expects obs_norm='image'")


def config_from_json(path: Path) -> ExperimentConfig:
    """Rebuild a config from the config.json a run wrote. Used by NB02+."""
    blob = json.loads(Path(path).read_text())
    kwargs = {s: cls(**_coerce(cls, blob.get(s, {}))) for s, cls in _SECTIONS.items()}
    return ExperimentConfig(source=blob.get("source", str(path)), **kwargs)
