"""PPO with GAE, written from scratch.

Deliberately plain. The one non-obvious piece of machinery is
`advantage_transform`, which is the hook Notebook 06 will use to add lambda*B(s)
to the advantages. In Notebook 01 it is None and the algorithm is textbook PPO.

Correctness notes that matter downstream:
  * Termination and truncation are handled separately (see buffer.py).
  * V(s) at an episode boundary is computed on the TRUE successor state, not on
    the post-reset observation.
  * The observation scaler is applied once, at collection time, and its state is
    written into every checkpoint.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from agents.buffer import RolloutBuffer
from agents.networks import ActorCritic
from config import ExperimentConfig, seed_dir
from dataio.checkpoint import checkpoint_path, save_checkpoint
from dataio.trajectory import TrajectoryRecorder
from envs.env_pool import EnvPool
from envs.scaling import make_scaler
from utils.logging import ScalarLogger
from utils.seeding import set_global_seed


class PPOTrainer:
    def __init__(
        self,
        cfg: ExperimentConfig,
        seed: int,
        out_dir: Optional[Path] = None,
        advantage_transform: Optional[Callable] = None,
        progress: bool = True,
    ):
        self.cfg = cfg
        self.seed = seed
        self.out_dir = Path(out_dir) if out_dir else seed_dir(cfg.run.run_name, seed)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.advantage_transform = advantage_transform
        self.progress = progress

        set_global_seed(seed, deterministic=cfg.run.torch_deterministic)

        self.pool = EnvPool(
            cfg.env.env_id,
            cfg.env.n_envs,
            seed=seed,
            max_episode_steps=cfg.env.max_episode_steps,
        )
        self.scaler = make_scaler(cfg.env.obs_norm, self.pool.single_observation_space)

        self.device = torch.device(cfg.ppo.device)
        self.model = ActorCritic(
            self.pool.obs_dim,
            self.pool.n_actions,
            cfg.ppo.hidden_sizes,
            cfg.ppo.activation,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg.ppo.learning_rate, eps=1e-5
        )

        self.batch = cfg.env.n_envs * cfg.ppo.n_steps
        self.n_updates = cfg.ppo.total_timesteps // self.batch
        self.minibatch = self.batch // cfg.ppo.n_minibatches

        self.buffer = RolloutBuffer(
            cfg.ppo.n_steps, cfg.env.n_envs, self.pool.obs_dim, self.pool.n_actions,
            device=cfg.ppo.device,
        )
        self.recorder = (
            TrajectoryRecorder(cfg.run.trajectory_stride) if cfg.run.record_trajectories else None
        )
        self.logger = ScalarLogger(self.out_dir / "scalars.csv")

        # rolling episode stats
        self._ep_returns: deque[float] = deque(maxlen=100)
        self._ep_lengths: deque[int] = deque(maxlen=100)
        self._ep_success: deque[float] = deque(maxlen=100)
        self.episode_log: list[dict] = []
        self.first_success_step: int | None = None

        # live entropy coefficient (mutated by the adaptive controller)
        self.ent_coef = cfg.ppo.ent_coef

        self.global_step = 0
        self._ckpt_targets = sorted(
            {int(f * cfg.ppo.total_timesteps): f for f in cfg.run.checkpoint_fractions}.items()
        )
        self._ckpt_done: set[float] = set()

    # ---------------------------------------------------------------- helpers #

    def _scale(self, raw: np.ndarray) -> np.ndarray:
        return self.scaler(raw)

    @torch.no_grad()
    def _values_np(self, raw: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(self._scale(raw), dtype=torch.float32, device=self.device)
        return self.model.value(x).cpu().numpy()

    # ------------------------------------------------------------- collection #

    def collect_rollout(self) -> None:
        cfg = self.cfg
        buf = self.buffer
        buf.reset()

        for _ in range(cfg.ppo.n_steps):
            raw_obs = self.pool.obs
            sim_state = self.pool.sim_states()

            if self.scaler.kind == "running":
                self.scaler.update(raw_obs)
            scaled = self._scale(raw_obs)

            x = torch.as_tensor(scaled, dtype=torch.float32, device=self.device)
            action, logprob, value, probs = self.model.act(x)
            action_np = action.cpu().numpy()

            out = self.pool.step(action_np)

            buf.add(
                obs=scaled,
                raw_obs=raw_obs,
                sim_state=sim_state,
                next_raw_obs=out["next_obs"],
                next_sim_state=out["next_sim_state"],
                actions=action_np,
                logprobs=logprob.cpu().numpy(),
                probs=probs.cpu().numpy(),
                rewards=out["reward"],
                values=value.cpu().numpy(),
                terminated=out["terminated"],
                truncated=out["truncated"],
                episode_id=out["episode_id"],
                episode_t=out["episode_t"],
            )

            self.global_step += cfg.env.n_envs

            for ep in out["finished"]:
                self._ep_returns.append(ep["return"])
                self._ep_lengths.append(ep["length"])
                self._ep_success.append(float(ep["success"]))
                self.episode_log.append({**ep, "global_step": self.global_step})
                if ep["success"] and self.first_success_step is None:
                    self.first_success_step = self.global_step

    # ------------------------------------------------------------------ update #

    def compute_advantages(self) -> tuple[np.ndarray, np.ndarray]:
        cfg, buf = self.cfg, self.buffer
        T = cfg.ppo.n_steps

        # V of the true successor state, needed only on rows where an episode
        # ended. Computing it for the whole rollout is cheap and keeps the code
        # branch-free; compute_gae only reads it at boundaries.
        boundary_v = self._values_np(buf.next_raw_obs[:T].reshape(-1, self.pool.obs_dim))
        boundary_v = boundary_v.reshape(T, cfg.env.n_envs)

        last_values = self._values_np(self.pool.obs)
        last_terminated = np.zeros(cfg.env.n_envs, dtype=bool)

        adv, ret = buf.compute_gae(
            last_values=last_values,
            last_terminated=last_terminated,
            next_values_at_boundary=boundary_v,
            gamma=cfg.ppo.gamma,
            gae_lambda=cfg.ppo.gae_lambda,
        )

        # --- Notebook 06 hook ------------------------------------------------
        if self.advantage_transform is not None:
            adv = self.advantage_transform(buf, adv)
            adv = np.asarray(adv, dtype=np.float32)
            if adv.shape != (T, cfg.env.n_envs):
                raise ValueError(f"advantage_transform returned shape {adv.shape}")
        # ---------------------------------------------------------------------

        return adv, ret

    def update(self, advantages: np.ndarray, returns: np.ndarray) -> dict[str, float]:
        cfg = self.cfg
        t = self.buffer.flat_tensors(advantages, returns)
        idx = np.arange(self.batch)

        clipfracs: list[float] = []
        approx_kls: list[float] = []
        pg_losses: list[float] = []
        v_losses: list[float] = []
        ent_losses: list[float] = []
        grad_norms: list[float] = []          # critic (or joint, if not separate)
        grad_norms_actor: list[float] = []

        stop = False
        for _epoch in range(cfg.ppo.n_epochs):
            np.random.shuffle(idx)
            for start in range(0, self.batch, self.minibatch):
                mb = idx[start : start + self.minibatch]

                newlogprob, entropy, newvalue = self.model.evaluate_actions(
                    t["obs"][mb], t["actions"][mb]
                )
                logratio = newlogprob - t["logprobs"][mb]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean().item()
                    approx_kls.append(approx_kl)
                    clipfracs.append(
                        ((ratio - 1.0).abs() > cfg.ppo.clip_coef).float().mean().item()
                    )

                mb_adv = t["advantages"][mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - cfg.ppo.clip_coef, 1 + cfg.ppo.clip_coef),
                ).mean()

                if cfg.ppo.clip_vloss:
                    v_unclipped = (newvalue - t["returns"][mb]) ** 2
                    v_clipped = t["values"][mb] + torch.clamp(
                        newvalue - t["values"][mb], -cfg.ppo.clip_coef, cfg.ppo.clip_coef
                    )
                    v_loss = 0.5 * torch.max(v_unclipped, (v_clipped - t["returns"][mb]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((newvalue - t["returns"][mb]) ** 2).mean()

                ent_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * ent_loss + cfg.ppo.vf_coef * v_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.ppo.separate_grad_clip:
                    gn_a = nn.utils.clip_grad_norm_(self.model.actor.parameters(), cfg.ppo.max_grad_norm)
                    gn_c = nn.utils.clip_grad_norm_(self.model.critic.parameters(), cfg.ppo.max_grad_norm)
                    grad_norms_actor.append(float(gn_a))
                    grad_norms.append(float(gn_c))
                else:
                    gn = nn.utils.clip_grad_norm_(self.model.parameters(), cfg.ppo.max_grad_norm)
                    grad_norms.append(float(gn))
                    grad_norms_actor.append(float("nan"))
                self.optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                ent_losses.append(ent_loss.item())

            if cfg.ppo.target_kl is not None and np.mean(approx_kls[-cfg.ppo.n_minibatches:]) > cfg.ppo.target_kl:
                stop = True
            if stop:
                break

        y_pred = t["values"].cpu().numpy()
        y_true = t["returns"].cpu().numpy()
        var_y = float(np.var(y_true))
        ev = float("nan") if var_y == 0 else 1.0 - float(np.var(y_true - y_pred)) / var_y

        return {
            "pg_loss": float(np.mean(pg_losses)),
            "v_loss": float(np.mean(v_losses)),
            "entropy": float(np.mean(ent_losses)),
            "approx_kl": float(np.mean(approx_kls)),
            "clipfrac": float(np.mean(clipfracs)),
            "grad_norm_critic": float(np.mean(grad_norms)),
            "grad_norm_actor": float(np.mean(grad_norms_actor)),
            "explained_variance": ev,
            "early_stopped": float(stop),
            "ent_coef": float(self.ent_coef),
        }

    # ------------------------------------------------------------------- train #

    def _adapt_entropy(self, observed_entropy: float, update: int) -> None:
        """Multiplicative controller holding mean policy entropy near a target.

        target(t) anneals linearly from target_entropy_start to
        target_entropy_end across training. If the policy is more deterministic
        than the target, ent_coef is raised; if it is more random, lowered.
        """
        cfg = self.cfg.ppo
        if cfg.ent_mode != "adaptive":
            return
        if self.first_success_step is None:
            frac = 0.0          # nothing found yet -> hold the exploration floor
        else:
            elapsed = self.global_step - self.first_success_step
            frac = min(1.0, elapsed / max(cfg.ent_anneal_frac * self.cfg.ppo.total_timesteps, 1.0))
        target = cfg.target_entropy_start + frac * (cfg.target_entropy_end - cfg.target_entropy_start)
        self._entropy_target = target
        err = target - observed_entropy
        self.ent_coef = float(
            np.clip(self.ent_coef * np.exp(cfg.ent_adapt_rate * np.sign(err) * min(abs(err) / max(target, 1e-8), 1.0) * 4.0),
                    cfg.ent_coef_min, cfg.ent_coef_max)
        )

    def _maybe_checkpoint(self) -> None:
        for target_step, frac in self._ckpt_targets:
            if frac in self._ckpt_done:
                continue
            if self.global_step >= target_step:
                p = checkpoint_path(self.out_dir / "checkpoints", frac)
                save_checkpoint(
                    p, self.model, self.scaler, self.optimizer,
                    self.global_step, frac,
                    extra={
                        "seed": self.seed,
                        "env_id": self.cfg.env.env_id,
                        "mean_return_100": float(np.mean(self._ep_returns)) if self._ep_returns else float("nan"),
                        "success_rate_100": float(np.mean(self._ep_success)) if self._ep_success else 0.0,
                    },
                )
                self._ckpt_done.add(frac)

    def train(self) -> dict:
        cfg = self.cfg
        self.pool.reset()
        t0 = time.time()

        for update in range(1, self.n_updates + 1):
            if cfg.ppo.anneal_lr:
                frac = 1.0 - (update - 1.0) / self.n_updates
                for g in self.optimizer.param_groups:
                    g["lr"] = frac * cfg.ppo.learning_rate

            gstep_start = self.global_step
            self.collect_rollout()
            adv, ret = self.compute_advantages()

            if self.recorder is not None:
                self.recorder.add_rollout(self.buffer, gstep_start, cfg.env.n_envs)

            stats = self.update(adv, ret)
            self._adapt_entropy(stats["entropy"], update)
            self._maybe_checkpoint()

            if update % cfg.run.log_every_updates == 0 or update == self.n_updates:
                row = {
                    "update": update,
                    "global_step": self.global_step,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "mean_return_100": float(np.mean(self._ep_returns)) if self._ep_returns else float("nan"),
                    "mean_length_100": float(np.mean(self._ep_lengths)) if self._ep_lengths else float("nan"),
                    "success_rate_100": float(np.mean(self._ep_success)) if self._ep_success else 0.0,
                    "n_episodes": len(self.episode_log),
                    "entropy_target": float(getattr(self, "_entropy_target", float("nan"))),
                    "sps": int(self.global_step / max(time.time() - t0, 1e-9)),
                    **stats,
                }
                self.logger.log(row)
                if self.progress:
                    print(
                        f"  upd {update:>5}/{self.n_updates}  step {self.global_step:>8,}  "
                        f"ret {row['mean_return_100']:>8.1f}  succ {row['success_rate_100']:.2f}  "
                        f"ent {row['entropy']:.3f}  kl {row['approx_kl']:.4f}  "
                        f"ev {row['explained_variance']:>6.3f}  {row['sps']:,} sps",
                        flush=True,
                    )

        # make sure the 100% checkpoint exists even with integer-division slack
        if 1.0 not in self._ckpt_done:
            save_checkpoint(
                checkpoint_path(self.out_dir / "checkpoints", 1.0),
                self.model, self.scaler, self.optimizer, self.global_step, 1.0,
                extra={"seed": self.seed, "env_id": cfg.env.env_id},
            )
            self._ckpt_done.add(1.0)

        self.logger.close()

        artifacts: dict = {
            "out_dir": self.out_dir,
            "scalars_csv": self.logger.path,
            "checkpoints": sorted((self.out_dir / "checkpoints").glob("ckpt_*.pt")),
            "wall_time_s": time.time() - t0,
            "first_success_step": self.first_success_step,
            "n_episodes": len(self.episode_log),
        }

        if self.recorder is not None:
            traj_path = self.recorder.save(
                self.out_dir / "trajectories.npz",
                meta={"seed": self.seed, "total_timesteps": self.global_step},
            )
            artifacts["trajectories"] = traj_path

        # per-episode log, useful for return curves that are not smoothed by
        # the 100-episode window
        if self.episode_log:
            import csv
            ep_path = self.out_dir / "episodes.csv"
            with open(ep_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(self.episode_log[0].keys()))
                w.writeheader()
                w.writerows(self.episode_log)
            artifacts["episodes_csv"] = ep_path

        self.cfg.save(self.out_dir / "config.json")
        self.pool.close()
        return artifacts
