"""E1 on RedBlueDoors -- the only file in the pipeline that knows the experiment.

WHAT E1 IS  (Experiments.pdf, Table 3 and section 8.2)

    Three arms, identical in every hyperparameter except the policy gradient:

        gae   plain PPO                        ppo.pg_mode = gae
        cf    PPO-CF, full direct oracle       ppo.pg_mode = cf_all_action
        shuf  action-permuted CF control       ppo.pg_mode = cf_shuffled

    The required ordering is BOTH

        cf > gae        and        cf > shuf                        Eq (32)

    The second half is the point of the `shuf` arm: the same oracle vectors
    assigned to DIFFERENT states. If the permuted control matches the true CF
    arm, the benefit was never state-local counterfactual credit -- it was a
    generic action prior or an entropy effect, and must be described as one.
    On DoorKey-6x6 at n=1 the permuted arm reached 90% success FIRST, which is
    why it is a first-class arm here and not an afterthought.

ONE UNIT OF WORK = one (stage, arm, seed) training run.

    Resumption is a set difference against the stage ledger on exactly those
    keys, so a walltime cut, a dead node and a cancelled submission all cost
    the same thing: the runs that were in flight. There is no mid-run resume in
    PPOTrainer, so a unit is atomic -- size the array walltime from the SLOWEST
    unit (RedBlueDoors-8x8 CF, ~5.5 h/seed measured).

WHAT IS DELIBERATELY NOT HERE

    No notebook execution. The analysis is done off-cluster: the merge step
    writes a self-describing ledger CSV plus summary figures, and those are
    what you copy home. See RUNBOOK.md.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ART_BASE = ROOT / "artifacts"

# --------------------------------------------------------------------------- #
# The experiment
# --------------------------------------------------------------------------- #

#: arm -> the ONE config field that differs between arms
ARM_PG_MODE = {
    "gae":  "gae",
    "cf":   "cf_all_action",
    "shuf": "cf_shuffled",
    # E2: same uniform CF-label budget; inject the exact perturbation locally
    # or a distilled prediction at every rollout state.
    "queried": "cf_queried_perturb",
    "distill": "landscape_distill",
}
GROUPS_ALL = ["gae", "cf", "shuf"]

ALGO_TAG = {
    "gae": "ppo",
    "cf": "ppo-cf",
    "shuf": "ppo-cf-shuffled",
    "queried": "ppo-cf-queried-perturb",
    "distill": "ppo-cf-distilled",
}

#: Trajectory datasets are 25 MB/seed and feed NB03+ (E2/E3 distillation), not
#: E1. Off here takes a seed from ~25 MB to ~4 MB -- 30 runs, ~110 MB instead
#: of ~1.5 GB. Set PPO_CF_RECORD_TRAJECTORIES=1 to turn them back on.
RECORD_TRAJECTORIES = os.environ.get("PPO_CF_RECORD_TRAJECTORIES", "0") == "1"

#: The default `log_every_updates: 10` writes a scalars row every 81,920 frames
#: on these configs. The E1 statistic is "frames to 50% / 90% success", and the
#: gaps in play are ~40k frames, so that resolution would quantise the answer.
#: Metrics are computed from episodes.csv (full resolution) and this only makes
#: the live log and the wandb curves usable.
LOG_EVERY_UPDATES = 2

#: Default paired seeds for cluster sweeps.
SEEDS = list(range(5))

#: E2 perturbation-strength sweep. Beta=0 is already represented by the PPO
#: baseline in E1, so E2 evaluates three non-zero strengths only.
E2_BETAS = (0.25, 0.75, 1.5)

#: Frame override used only by the historical RedBlue E1 stages below. The new
#: EXP1_* stages use their YAML budgets as the source of truth.
FRAMES = 2_000_000

STAGES: dict[str, dict] = {
    # RedBlueDoors-6x6. Measured: gae ~25 min, cf/shuf ~90 min per seed at 2M.
    # The YAML's own budget is 1M (rl-baselines3-zoo's published entry for this
    # env); 2M is a deliberate override to match 8x8. NOTE this env already
    # saturates -- rbd6x6_gae/seed_0 reached success 1.0 well before 1M -- so
    # the second half of the run is mostly flat, which dilutes success AUC
    # without changing frames_to_50 / frames_to_90.
    "E1_RBD6": {
        "env_config": "redbluedoors6x6_cf",
        "seeds": SEEDS,
        # cf_horizon 64, cf_subsample 0.025, alpha_cf 0.1 all come from the YAML.
        "overrides": {"ppo.total_timesteps": FRAMES},
    },
    # RedBlueDoors-8x8. cf_horizon comes from config/envs/redbluedoors8x8_cf.yaml
    # (32) -- it was briefly overridden to 64 here, which doubled the branch
    # budget from 3.2 to 6.4 and the cost from ~2.75 h to ~5.5 h per CF seed.
    # Reverted 2026-09-09: the YAML is the single source of truth for the
    # estimator, so the cluster, the notebooks and `scripts/train.py` all agree.
    "E1_RBD8": {
        "env_config": "redbluedoors8x8_cf",
        "seeds": SEEDS,
        "overrides": {"ppo.total_timesteps": FRAMES},
    },

    # ---- E0: a two-seed re-test of the setting that produced 0.72 ---------
    # DELIBERATELY EMPTY OVERRIDES. This is the config as
    # `01_ppo_baseline_redbluedoor.ipynb` ran it -- ENV_CONFIG
    # "redbluedoors8x8_cf" with OVERRIDES = {} -- and `runs/rbd8x8/config.json`
    # confirms it byte for byte. Anything typed here instead of read from the
    # YAML is a chance for the two to drift, which is the whole point of the
    # stage. PPO-CF only; submit with groups "cf".
    #
    # Seeds 0 and 2: seed 0 reached success 0.72 and is the reproduction check,
    # seed 2 is fresh. Seed 1 is skipped -- it reached 0.00 with
    # cf_reward_coverage 0.000 at this same horizon.
    "E0_reddoorbluedoor_test": {
        "env_config": "redbluedoors8x8_cf",
        "seeds": [0, 2],
        "overrides": {},
    },

    # ---- Experiment 1: seven environments, PPO vs PPO-CF -----------------
    #
    # Submitted by ./experiment_1_6_envs. Despite the historical filename, this
    # sweep has seven environments: DoorKey 5x5, DoorKey 6x6, LavaGap, Unlock,
    # UnlockPickup, RedBlueDoors 6x6, and Taxi.
    #
    # Each stage uses the *_cf YAML as the single source of truth. The array
    # overrides only ppo.pg_mode per group, plus seed/run-name/logging fields,
    # so the PPO and PPO-CF arms differ only in the policy gradient.
    "EXP1_DK5": {
        "env_config": "doorkey5x5_cf",
        "env_tag": "doorkey5x5",
        "seeds": SEEDS,
        "overrides": {},
    },
    "EXP1_DK6": {
        "env_config": "doorkey6x6_cf",
        "env_tag": "doorkey6x6",
        "seeds": SEEDS,
        "overrides": {},
    },
    "EXP1_LAVAGAP": {
        "env_config": "lavagap_cf",
        "env_tag": "lavagap",
        "seeds": SEEDS,
        "overrides": {},
    },
    "EXP1_UNLOCK": {
        "env_config": "unlock_cf",
        "env_tag": "unlock",
        "seeds": SEEDS,
        "overrides": {},
    },
    "EXP1_UNLOCKPICKUP": {
        "env_config": "unlockpickup_cf",
        "env_tag": "unlockpickup",
        "seeds": SEEDS,
        "overrides": {},
    },
    "EXP1_RBD6": {
        "env_config": "redbluedoors6x6_cf",
        "env_tag": "redbluedoors6x6",
        "seeds": SEEDS,
        "overrides": {},
    },
    "EXP1_TAXI": {
        "env_config": "taxi_cf",
        "env_tag": "taxi",
        "seeds": SEEDS,
        "overrides": {},
    },

    # ---- E2: CF advantage perturbation, local vs amortised ---------------
    # Both arms query exactly 2% of rollout states and use five paired seeds.
    # `queried` injects the exact A_CF only at a queried state; `distill`
    # trains on those labels and injects its prediction at every rollout state.
    # E1 already contains the GAE and all-action-PPO-CF comparisons.
    "E2_TAXI": {
        "env_config": "taxi_cf",
        "env_tag": "taxi",
        "seeds": SEEDS,
        "overrides": {"ppo.cf_subsample": 0.02, "ppo.norm_adv": "batch"},
    },
    "E2_DK6": {
        "env_config": "doorkey6x6_cf",
        "env_tag": "doorkey6x6",
        "seeds": SEEDS,
        "overrides": {"ppo.cf_subsample": 0.02, "ppo.norm_adv": "batch"},
    },
    "E2_UNLOCKPICKUP": {
        "env_config": "unlockpickup_cf",
        "env_tag": "unlockpickup",
        "seeds": SEEDS,
        "overrides": {"ppo.cf_subsample": 0.02, "ppo.norm_adv": "batch"},
    },
    "E2_RBD6": {
        "env_config": "redbluedoors6x6_cf",
        "env_tag": "redbluedoors6x6",
        "seeds": SEEDS,
        "overrides": {"ppo.cf_subsample": 0.02, "ppo.norm_adv": "batch"},
    },
    # One pooled submission expands to the four environment stages above:
    # 4 environments x 5 seeds x 3 betas x 2 arms = 120 units. Its manifest is
    # globally interleaved, avoiding a sequential environment chain and idle
    # workers.
    "E2_ALL": {
        "env_tag": "mixed",
        "seeds": SEEDS,
        "overrides": {},
    },
}

#: E1_RBD6 finishes -> E1_RBD8 is submitted automatically. `PART_END` in the
#: submit script is what stops the chain.
CHAIN_NEXT = {
    "E1_RBD6": "E1_RBD8",
    "E1_RBD8": None,
    "E0_reddoorbluedoor_test": None,
    "EXP1_DK5": "EXP1_DK6",
    "EXP1_DK6": "EXP1_LAVAGAP",
    "EXP1_LAVAGAP": "EXP1_UNLOCK",
    "EXP1_UNLOCK": "EXP1_UNLOCKPICKUP",
    "EXP1_UNLOCKPICKUP": "EXP1_RBD6",
    "EXP1_RBD6": "EXP1_TAXI",
    "EXP1_TAXI": None,
    "E2_TAXI": None,
    "E2_DK6": None,
    "E2_UNLOCKPICKUP": None,
    "E2_RBD6": None,
    "E2_ALL": None,
}

#: Stages with no parallel work at all (none in E1).
ANALYSIS_ONLY: set[str] = set()

#: Notebook execution is off; see the module docstring.
NOTEBOOK: dict[str, str] = {}

#: SMOKE MODE. Set PPO_CF_SMOKE_FRAMES to exercise the whole chain -- manifest,
#: a real array, merge, requeue, summary -- in minutes instead of ~74 core-hours.
#: It shrinks every unit AND moves the ledger, chunks, manifest and figures into
#: a parallel "<STAGE>_smoke" namespace, so a smoke pass can never mark a real
#: unit as done or pollute a real result. Unset it and the real run starts clean.
SMOKE_FRAMES = int(os.environ.get("PPO_CF_SMOKE_FRAMES", "0"))

#: Every artifact path in the pipeline derives from this one name, so smoke mode
#: needs to redirect exactly one thing. Everything a smoke pass writes then lands
#: under artifacts/_smoke/ and runs/_smoke/ -- two directories you can delete
#: without reading them, and which no real result is ever mixed into.
ART = ART_BASE / "_smoke" if SMOKE_FRAMES else ART_BASE


def art_key(stage: str) -> str:
    """The name a stage's artifacts are filed under."""
    return stage


LEDGER = {s: ART / "tables" / f"{art_key(s)}.csv" for s in STAGES}

#: The columns that uniquely identify one unit of work. `pending()` uses these
#: to decide what is already done -- no more (or resumption misses work) and no
#: fewer (or it redoes work). `stage` is implicit except in E2_ALL, whose one
#: ledger pools four environment stages and three beta settings.
KEYS = {s: ["group", "seed"] for s in STAGES}
KEYS["E2_ALL"] = ["stage", "group", "seed", "beta"]


def beta_slug(beta: float) -> str:
    """Stable, filesystem/W&B-safe label for one fixed beta condition."""
    return "b" + f"{float(beta):g}".replace("-", "m").replace(".", "p")


def run_name(stage: str, group: str, beta: float | None = None) -> str:
    """runs/<run_name>/seed_<seed>/ -- e.g. runs/e1_rbd6_cf/seed_3/."""
    prefix = "_smoke/" if SMOKE_FRAMES else ""
    beta_part = f"_{beta_slug(beta)}" if beta is not None else ""
    return f"{prefix}{stage.lower()}_{group}{beta_part}"


def env_tag(stage: str) -> str:
    spec = STAGES[stage]
    return str(spec.get("env_tag") or spec["env_config"].replace("_cf", ""))


def run_list(stage: str, groups: list[str]) -> list[dict]:
    """Every unit this stage needs -- the FULL sweep. `pending()` subtracts."""
    if stage not in STAGES:
        raise KeyError(f"unknown stage {stage!r}; known: {sorted(STAGES)}")
    if stage == "E2_ALL":
        # This order, followed by `pipeline.manifest.interleave`, makes each
        # preassigned pair one matched (environment, seed, beta) comparison
        # while cycling environments and beta settings across queued chunks.
        env_stages = ("E2_TAXI", "E2_DK6", "E2_UNLOCKPICKUP", "E2_RBD6")
        return [{"stage": env_stage, "group": group, "seed": seed, "beta": beta}
                for seed in SEEDS for env_stage in env_stages for beta in E2_BETAS
                for group in groups]
    seeds = STAGES[stage]["seeds"]
    return [{"stage": stage, "group": g, "seed": s} for g in groups for s in seeds]


def stage_groups(stage: str, groups: list[str]) -> list[str]:
    """Narrow a stage to a subset of arms. E1 runs all three everywhere."""
    unknown = [g for g in groups if g not in ARM_PG_MODE]
    if unknown:
        raise KeyError(f"unknown arm(s) {unknown}; known: {sorted(ARM_PG_MODE)}")
    return list(groups)


def build_config(stage: str, group: str, seed: int, beta: float | None = None):
    """The exact ExperimentConfig one unit runs. Importable for a dry run."""
    from config import make_config

    if stage == "E2_ALL":
        raise ValueError("E2_ALL is a pooled manifest, not an executable environment stage")
    spec = STAGES[stage]
    over = dict(spec["overrides"])
    over["ppo.pg_mode"] = ARM_PG_MODE[group]
    if stage.startswith("E2_"):
        beta = E2_BETAS[0] if beta is None else float(beta)
        if beta not in E2_BETAS:
            raise ValueError(f"E2 beta must be one of {E2_BETAS}, got {beta}")
        over["distill.beta"] = beta
    over["run.run_name"] = run_name(stage, group, beta)
    over["run.seeds"] = (int(seed),)
    over["run.record_trajectories"] = RECORD_TRAJECTORIES
    over["run.log_every_updates"] = LOG_EVERY_UPDATES
    if SMOKE_FRAMES:
        # A few updates on a small batch: enough to prove the chain moves, cheap
        # enough to run before every real submission.
        over["env.n_envs"] = 4
        over["ppo.n_steps"] = 64
        over["ppo.total_timesteps"] = SMOKE_FRAMES
        over["run.log_every_updates"] = 1
        over["run.checkpoint_fractions"] = (1.0,)
    return make_config(spec["env_config"], **over)


# --------------------------------------------------------------------------- #
# One unit of work
# --------------------------------------------------------------------------- #

def execute(task: dict) -> dict:
    """Train ONE (stage, arm, seed) and return one flat, self-describing row.

    Heavy outputs (checkpoints, scalars.csv, episodes.csv) are written by the
    trainer under runs/<run_name>/seed_<seed>/. What comes back here is the
    small row that lands in the ledger and drives the analysis.
    """
    stage = task["stage"]
    group = task["group"]
    seed = int(task["seed"])

    from scripts.train import run_seed

    beta = task.get("beta")
    cfg = build_config(stage, group, seed, beta=beta)

    # W&B, if enabled, is grouped by environment stage so the two E2 arms sit
    # in one comparison. The launcher selects online/offline mode; telemetry
    # never controls whether a training unit succeeds.
    os.environ.setdefault("PPO_CF_WANDB_PROJECT", "ppo-cf")
    algo_tag = ALGO_TAG.get(group, group)
    beta_tag = f"beta-{cfg.distill.beta:g}"
    os.environ["PPO_CF_WANDB_GROUP"] = (
        f"{stage}_{beta_tag}" if group in ("queried", "distill") else stage
    )
    os.environ["PPO_CF_WANDB_JOB_TYPE"] = algo_tag
    os.environ["PPO_CF_WANDB_TAGS"] = f"{env_tag(stage)},{algo_tag},{beta_tag}"

    print(cfg.summary(), flush=True)
    t0 = time.time()
    art = run_seed(cfg, seed, progress=True)
    wall = time.time() - t0

    out_dir = Path(art["out_dir"])
    row = {
        "stage": stage,
        "group": group,
        "env_tag": env_tag(stage),
        "algo_tag": algo_tag,
        "seed": seed,
        "beta": cfg.distill.beta if group in ("queried", "distill") else float("nan"),
        "run_name": cfg.run.run_name,
        "env_id": cfg.env.env_id,
        "config_source": cfg.source,
        "pg_mode": cfg.ppo.pg_mode,
        "alpha_gae": cfg.ppo.alpha_gae,
        "alpha_cf": cfg.ppo.alpha_cf,
        "cf_horizon": cfg.ppo.cf_horizon,
        "cf_subsample": cfg.ppo.cf_subsample,
        "cf_branch_budget": cfg.ppo.cf_horizon * cfg.ppo.cf_rollouts * cfg.ppo.cf_subsample,
        "perturb_beta": cfg.distill.beta if group in ("queried", "distill") else float("nan"),
        "total_timesteps": cfg.ppo.total_timesteps,
        "wall_time_s": round(wall, 1),
        "n_episodes": art["n_episodes"],
        "first_success_step": art["first_success_step"],
        "out_dir": str(out_dir.relative_to(ROOT)),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_sha": _git_sha(),
    }
    row.update(unit_metrics(out_dir, cfg.ppo.total_timesteps))
    return row


def _git_sha() -> str:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# Metrics -- section 9 of Experiments.pdf, "primary learning"
# --------------------------------------------------------------------------- #

def unit_metrics(out_dir: Path, total_timesteps: int) -> dict:
    """Success AUC, final success, frames to 50%/90%, and PPO dynamics.

    The success curve comes from episodes.csv (one row per episode) rather than
    scalars.csv, so its resolution is one episode rather than one log interval.
    `success_rate_100` in scalars.csv is a 100-episode rolling mean; this
    reproduces it exactly, at full resolution.
    """
    import numpy as np
    import pandas as pd

    out: dict = {}
    ep_path = Path(out_dir) / "episodes.csv"
    if ep_path.exists():
        ep = pd.read_csv(ep_path).sort_values("global_step")
        if len(ep):
            succ = ep["success"].astype(bool).astype(float)
            roll = succ.rolling(100, min_periods=1).mean().to_numpy()
            step = ep["global_step"].to_numpy(dtype=float)

            out["final_success"] = float(roll[-1])
            out["n_success_episodes"] = int(succ.sum())
            # Normalised AUC in [0, 1]: area under the success curve divided by
            # the frame budget, so arms with different wall clock but the same
            # frame budget are directly comparable.
            # np.trapezoid is numpy>=2.0; np.trapz is the <2.0 spelling and is
            # removed in 2.x. requirements.txt allows both.
            trapz = getattr(np, "trapezoid", None) or np.trapz
            if len(step) > 1:
                out["success_auc"] = float(trapz(roll, step) / max(total_timesteps, 1))
            else:
                out["success_auc"] = 0.0
            for thr in (0.5, 0.9):
                hit = np.flatnonzero(roll >= thr)
                out[f"frames_to_{int(thr * 100)}"] = (
                    float(step[hit[0]]) if hit.size else float("nan"))
            for col, name in (("subgoal1", "final_subgoal1"), ("subgoal2", "final_subgoal2")):
                if col in ep.columns:
                    out[name] = float(ep[col].astype(bool).astype(float)
                                      .rolling(100, min_periods=1).mean().iloc[-1])
            out["final_return_100"] = float(ep["return"].rolling(100, min_periods=1).mean().iloc[-1])
            out["final_length_100"] = float(ep["length"].rolling(100, min_periods=1).mean().iloc[-1])

    sc_path = Path(out_dir) / "scalars.csv"
    if sc_path.exists():
        sc = pd.read_csv(sc_path)
        if len(sc):
            last = sc.iloc[-1]
            for c in ("entropy", "approx_kl", "clipfrac", "explained_variance",
                      "pg_loss", "v_loss", "sps"):
                if c in sc.columns:
                    out[f"final_{c}"] = _f(last.get(c))
            # Budget honesty + perturbation/student quality, averaged over
            # training and at the end.
            for c in [c for c in sc.columns if c.startswith(("cf_", "perturb_", "distill_"))]:
                out[f"mean_{c}"] = _f(sc[c].mean())
                out[f"final_{c}"] = _f(last.get(c))
    return out


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


# --------------------------------------------------------------------------- #
# Ledger: merge and resume
# --------------------------------------------------------------------------- #

def merge_chunks(stage: str) -> None:
    """Fold artifacts/chunks/<STAGE>_chunk_*.jsonl into the stage ledger BY NAME.

    Chunks are JSON Lines, not appended CSV, because rows from different arms
    do not all carry the same columns (only CF arms emit cf_* diagnostics) and
    a positional CSV append would shift every later field one column left,
    silently. Aligning by column name and de-duplicating on KEYS is what makes
    a re-run of a unit harmless rather than a duplicate row.
    """
    import pandas as pd

    rows = []
    for p in sorted((ART / "chunks").glob(f"{art_key(stage)}_chunk_*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        return

    led = LEDGER[stage]
    new = pd.DataFrame(rows)
    if led.exists():
        new = pd.concat([pd.read_csv(led), new], ignore_index=True)
    new = new.drop_duplicates(subset=KEYS[stage], keep="last")
    new = new.sort_values(KEYS[stage])
    led.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(led, index=False)


def pending(stage: str, wanted: list[dict]) -> list[dict]:
    """The subset of `wanted` with no ledger row. This function IS the resume
    logic; there is no state anywhere else."""
    import pandas as pd

    led = LEDGER[stage]
    if not led.exists():
        return list(wanted)
    df = pd.read_csv(led)
    keys = KEYS[stage]
    if any(k not in df.columns for k in keys):
        return list(wanted)
    have = {tuple(str(r[k]) for k in keys) for _, r in df.iterrows()}
    return [t for t in wanted if tuple(str(t[k]) for k in keys) not in have]
