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
from envs.env_pool import EnvPool, make_env
from envs.scaling import make_scaler
from utils.logging import ScalarLogger
from utils.wandb_sink import make_sinks
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

        env_kwargs = {}
        if cfg.env.env_id.startswith("MiniGrid"):
            env_kwargs["fully_observable"] = cfg.env.fully_observable
        self.pool = EnvPool(
            cfg.env.env_id,
            cfg.env.n_envs,
            seed=seed,
            max_episode_steps=cfg.env.max_episode_steps,
            env_kwargs=env_kwargs,
            layout_seeds=cfg.env.layout_seeds,
            layout_seed_mode=cfg.env.layout_seed_mode,
            success_on=cfg.env.success_on,
            reward_cfg=cfg.reward,
            gamma=cfg.ppo.gamma,
        )
        self.env_kwargs = env_kwargs
        self.scaler = make_scaler(cfg.env.obs_norm, self.pool.single_observation_space)
        self.input_dim = self.scaler.out_dim
        self.raw_obs_dim = self.pool.raw_obs_dim

        self.device = torch.device(cfg.ppo.device)
        self.model = ActorCritic(
            self.input_dim,
            self.pool.n_actions,
            cfg.ppo.hidden_sizes,
            cfg.ppo.activation,
            prob_floor=cfg.ppo.prob_floor_start,
            encoder=cfg.ppo.encoder,
            obs_shape=(tuple(self.pool.single_observation_space.shape)
                       if cfg.ppo.encoder == "cnn" else None),
            share_encoder=cfg.ppo.share_encoder,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg.ppo.learning_rate, eps=cfg.ppo.optim_eps
        )
        self.separate_grad_clip = cfg.ppo.separate_grad_clip and self.model.can_clip_separately
        if cfg.ppo.separate_grad_clip and not self.model.can_clip_separately:
            print("  note: share_encoder=True -> using a single global grad clip")
        if cfg.run.init_from:
            self._warm_start(cfg.run.init_from)

        self.batch = cfg.env.n_envs * cfg.ppo.n_steps
        self.n_updates = cfg.ppo.total_timesteps // self.batch
        self.minibatch = self.batch // cfg.ppo.n_minibatches

        self.buffer = RolloutBuffer(
            cfg.ppo.n_steps, cfg.env.n_envs,
            input_dim=self.input_dim, raw_obs_dim=self.raw_obs_dim,
            sim_state_dim=self.pool.sim_state_dim,
            n_actions=self.pool.n_actions, device=cfg.ppo.device,
        )
        self.recorder = (
            TrajectoryRecorder(cfg.run.trajectory_stride, cfg.run.compact_trajectory_dtypes)
            if cfg.run.record_trajectories else None
        )

        # --- PPO-CF: the counterfactual oracle, evaluated every rollout ----- #
        self.oracle = None
        self.landscape = None
        self._distill_start_update: int | None = None
        self._cf_validated = True
        if cfg.ppo.pg_mode.startswith("cf") or cfg.ppo.pg_mode == "landscape_distill":
            from oracle.online import OnlineOracle
            self.oracle = OnlineOracle(
                cfg.env.env_id, self.pool.n_actions, cfg.ppo.gamma,
                env_kwargs=env_kwargs, max_episode_steps=cfg.env.max_episode_steps,
                restore=cfg.ppo.cf_restore, seed=seed,
            )
            self._cf_validated = not cfg.ppo.cf_validate
            self._cf_rng = np.random.default_rng(seed + 7919)
        if cfg.ppo.pg_mode == "landscape_distill":
            from agents.landscape import LandscapeEnsemble, LandscapeReplay
            self.landscape_replay = LandscapeReplay(cfg.distill.replay_capacity)
            self.landscape = LandscapeEnsemble(
                obs_dim=self.input_dim,
                n_actions=self.pool.n_actions,
                hidden_sizes=cfg.distill.hidden_sizes,
                activation=cfg.ppo.activation,
                encoder=cfg.ppo.encoder,
                obs_shape=(tuple(self.pool.single_observation_space.shape)
                           if cfg.ppo.encoder == "cnn" else None),
                n_members=cfg.distill.ensemble_size,
                learning_rate=cfg.distill.learning_rate,
                device=cfg.ppo.device,
                seed=seed,
            )
        # The oracle restores simulator states, so they must be captured even
        # when no trajectory dataset is being written.
        self._needs_sim_state = self.recorder is not None or self.oracle is not None
        self._zero_sim = np.zeros((cfg.env.n_envs, self.pool.sim_state_dim), dtype=np.float64)
        # `make_sinks` returns [] unless PPO_CF_WANDB=1, so notebooks are
        # unaffected. A sink can never raise into the loop; see utils/logging.py.
        self.logger = ScalarLogger(self.out_dir / "scalars.csv",
                                   sinks=make_sinks(cfg, seed))

        # rolling episode stats
        self._ep_returns: deque[float] = deque(maxlen=100)
        self._ep_lengths: deque[int] = deque(maxlen=100)
        self._ep_success: deque[float] = deque(maxlen=100)
        # DoorKey sub-goals. Success alone is useless as a progress signal on a
        # sparse task -- it can sit at zero for millions of frames. These say
        # WHICH rung the policy is stuck on.
        # What the two sub-goals MEAN is environment-specific (DoorKey: key
        # picked up / door opened. RedBlueDoors: red opened / blue opened), so
        # the names are generic and the notebooks supply the labels.
        self._ep_subgoal1: deque[float] = deque(maxlen=100)
        self._ep_subgoal2: deque[float] = deque(maxlen=100)
        self.episode_log: list[dict] = []
        self.first_success_step: int | None = None
        self._success_ema: float = 0.0

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
            # Snapshotting the simulator state is only needed for the trajectory
            # dataset NB02+ consume; on MiniGrid it means encoding the grid for
            # every env on every step, so skip it when nothing will read it.
            sim_state = (self.pool.sim_states() if self._needs_sim_state
                         else self._zero_sim)

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
                self._ep_subgoal1.append(float(ep.get("subgoal1", float("nan"))))
                self._ep_subgoal2.append(float(ep.get("subgoal2", float("nan"))))
                self.episode_log.append({**ep, "global_step": self.global_step})
                if ep["success"] and self.first_success_step is None:
                    self.first_success_step = self.global_step

    # ---------------------------------------------------- counterfactuals #

    @torch.no_grad()
    def _probs_np(self, raw: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(self._scale(raw), dtype=torch.float32, device=self.device)
        return self.model.action_probs(x).cpu().numpy()

    def compute_counterfactual(self) -> tuple[np.ndarray, np.ndarray, dict]:
        """Q_g on a SUBSAMPLE of rollout states. Returns (q, mask, diagnostics).

        `q` is (B, K) with rows outside the subsample left at zero; `mask` says
        which rows are real. Subsampling is forced by cost: Eq (3) needs
        K * cf_rollouts branches of up to cf_horizon steps PER STATE, so
        computing it for all 2048 rollout states would be ~40x slower than
        training. The alpha blend applies the counterfactual loss only on the
        masked rows and plain PPO everywhere, which is what makes a 5% sample
        affordable.
        """
        cfg = self.cfg.ppo
        T, N = cfg.n_steps, self.cfg.env.n_envs
        B = T * N
        sims = self.buffer.sim_state[:T].reshape(B, -1)

        n_sub = max(1, int(round(cfg.cf_subsample * B)))
        idx = self._cf_rng.choice(B, size=n_sub, replace=False)

        q_sub, diag = self.oracle.q_g(
            sims[idx], self._probs_np, self._values_np,
            horizon=cfg.cf_horizon, n_rollouts=cfg.cf_rollouts,
            seed=int(self._cf_rng.integers(1 << 30)),
            bootstrap_tail=cfg.cf_bootstrap_tail, chunk=cfg.cf_branch_envs,
        )

        q = np.zeros((B, self.pool.n_actions), dtype=np.float32)
        mask = np.zeros(B, dtype=bool)
        q[idx] = q_sub
        mask[idx] = True

        if cfg.pg_mode == "cf_shuffled":
            # The plan's P1 control: same oracle vectors, different states. If
            # this reproduces the benefit, the benefit was not counterfactual
            # information -- it was the extra gradient structure.
            perm = self._cf_rng.permutation(n_sub)
            q[idx] = q_sub[perm]

        diag["n_cf_states"] = n_sub
        return q, mask, diag

    # ---------------------------------------------------- E2 distillation #

    def _distill_beta(self, update: int) -> float:
        """A deterministic warm-up, deliberately not an uncertainty gate."""
        assert self.landscape is not None
        cfg = self.cfg.distill
        if self.landscape_replay.size < cfg.min_labels_before_use:
            return 0.0
        if self._distill_start_update is None:
            self._distill_start_update = int(update)
        if cfg.beta_ramp_updates <= 0:
            return float(cfg.beta)
        frac = min(1.0, (update - self._distill_start_update + 1) / cfg.beta_ramp_updates)
        return float(cfg.beta * frac)

    def predict_distilled_landscape(self, update: int) -> tuple[np.ndarray, float]:
        """Detached student landscape for the *current* rollout.

        This runs before the current rollout's labels are added to replay.  The
        resulting prediction can therefore only use information from earlier
        rollouts, which makes E2 a genuine unqueried-state amortisation test.
        """
        assert self.landscape is not None
        T, N = self.cfg.ppo.n_steps, self.cfg.env.n_envs
        B = T * N
        obs = self.buffer.obs[:T].reshape(B, -1)
        pi = self.buffer.probs[:T].reshape(B, -1)
        raw = self.landscape.predict(obs)
        centred = raw - (pi * raw).sum(axis=1, keepdims=True)
        return centred.astype(np.float32), self._distill_beta(update)

    @staticmethod
    def _policy_direction(v: np.ndarray, pi: np.ndarray) -> np.ndarray:
        """C_pi v, the action-space direction PPO can actually use."""
        centred = v - (pi * v).sum(axis=1, keepdims=True)
        return pi * centred

    def learn_distilled_landscape(
        self, q_cf: np.ndarray, cf_mask: np.ndarray, prediction: np.ndarray
    ) -> dict[str, float]:
        """Audit fresh unseen labels, then add them and train for next rollout."""
        assert self.landscape is not None
        T, N = self.cfg.ppo.n_steps, self.cfg.env.n_envs
        B = T * N
        flat = lambda a: a[:T].reshape((B,) + a.shape[2:])
        mask = np.asarray(cf_mask, dtype=bool)
        obs = flat(self.buffer.obs)[mask]
        pi = flat(self.buffer.probs)[mask]
        q = np.asarray(q_cf, dtype=np.float32)[mask]
        target = q - (pi * q).sum(axis=1, keepdims=True)
        pred = np.asarray(prediction, dtype=np.float32)[mask]

        target_centering = float(np.abs((pi * target).sum(axis=1)).max()) if len(target) else 0.0
        pred_centering = float(np.abs((pi * pred).sum(axis=1)).max()) if len(pred) else 0.0
        # These are evaluated BEFORE the fresh labels enter replay.
        d_pred = self._policy_direction(pred, pi)
        d_target = self._policy_direction(target, pi)
        heldout_pg_error = float(np.linalg.norm(d_pred - d_target, axis=1).mean()) if len(target) else float("nan")
        denom = np.linalg.norm(d_pred, axis=1) * np.linalg.norm(d_target, axis=1)
        cosine = np.divide((d_pred * d_target).sum(axis=1), denom,
                           out=np.zeros_like(denom), where=denom > 1e-12)
        heldout_cos = float(cosine.mean()) if len(cosine) else float("nan")
        best_agreement = float((pred.argmax(axis=1) == target.argmax(axis=1)).mean()) if len(target) else float("nan")

        self.landscape_replay.add(obs, pi, target)
        train = self.landscape.train(
            self.landscape_replay,
            steps=self.cfg.distill.train_steps_per_rollout,
            batch_size=self.cfg.distill.train_batch_size,
        )
        return {
            "distill_replay_size": float(self.landscape_replay.size),
            "distill_total_labels": float(self.landscape_replay.total_added),
            "distill_student_loss": train.loss,
            "distill_student_steps": float(train.n_steps),
            "distill_target_centering": target_centering,
            "distill_prediction_centering": pred_centering,
            "distill_heldout_pg_error": heldout_pg_error,
            "distill_heldout_direction_cos": heldout_cos,
            "distill_heldout_best_action_agreement": best_agreement,
        }

    def _validate_oracle(self, q_cf: np.ndarray, mask: np.ndarray,
                         n_sample: int = 96) -> None:
        """Run once, on the first rollout. Cheap, and the failure it catches is
        otherwise undetectable: a broken restore produces A_CF values that look
        entirely reasonable and are wrong."""
        from oracle.online import check_centering, check_replay

        T, N = self.cfg.ppo.n_steps, self.cfg.env.n_envs
        flat = lambda a: a[:T].reshape((T * N,) + a.shape[2:])
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(T * N, size=min(n_sample, T * N), replace=False)

        probs = flat(self.buffer.probs)[mask]
        q_m = q_cf[mask]
        centering = check_centering(q_m - (probs * q_m).sum(-1, keepdims=True), probs)
        if not centering["ok"]:
            raise RuntimeError(
                f"A_CF is not policy-centred: max |sum_a pi*A_CF| = "
                f"{centering['max_abs']:.3e}. The all-action gradient would be biased."
            )

        if self.cfg.reward.active:
            print("  oracle check: replay skipped (reward shaping alters recorded rewards); "
                  f"centering max {centering['max_abs']:.2e}")
            return

        rep = check_replay(
            self.oracle,
            flat(self.buffer.sim_state)[idx],
            flat(self.buffer.actions)[idx],
            flat(self.buffer.rewards)[idx],
            flat(self.buffer.next_sim_state)[idx],
        )
        if not rep["exact"]:
            raise RuntimeError(
                f"oracle restore is not exact over {rep['n']} sampled transitions: "
                f"max reward error {rep['max_reward_error']:.3e}, max state error "
                f"{rep['max_state_error']:.3e}. Every A_CF in this run would be wrong."
            )
        print(f"  oracle check: replay exact over {rep['n']} transitions "
              f"(reward err 0, state err 0); centering max {centering['max_abs']:.2e}")

    # ------------------------------------------------------------------ update #

    def compute_advantages(self) -> tuple[np.ndarray, np.ndarray]:
        cfg, buf = self.cfg, self.buffer
        T = cfg.ppo.n_steps

        # V of the true successor state, needed only on rows where an episode
        # ended. Computing it for the whole rollout is cheap and keeps the code
        # branch-free; compute_gae only reads it at boundaries.
        boundary_v = self._values_np(buf.next_raw_obs[:T].reshape(-1, self.raw_obs_dim))
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

    @staticmethod
    def _normalise(adv: torch.Tensor, min_std: float) -> tuple[torch.Tensor, float]:
        """Whiten `adv`, but ONLY if it carries signal.

        This guard is the single most important correctness fix in this file.
        On a sparse task the critic converges to V = 0 everywhere and a
        reward-free rollout's advantages are ~1e-10 of pure floating-point
        noise. The textbook `(a - a.mean()) / (a.std() + 1e-8)` then rescales
        that noise to unit variance and hands it to the policy gradient, so the
        policy is actively random-walked by numerical error whenever there is
        nothing to learn. Measured on the DoorKey-8x8 3M run: clipfrac held
        0.05-0.18 and approx_kl ~0.01 while v_loss was 1e-19 and explained
        variance was exactly 0 -- i.e. every one of those updates was noise.

        Returns (possibly unchanged advantages, the raw std) so the raw std can
        be logged; watching it is how you tell "no signal" from "not learning".
        """
        std = float(adv.std())
        if std < min_std:
            return adv, std
        return (adv - adv.mean()) / (std + 1e-8), std

    def update(self, advantages: np.ndarray, returns: np.ndarray,
               q_cf: np.ndarray | None = None,
               cf_mask: np.ndarray | None = None,
               distill_landscape: np.ndarray | None = None,
               perturb_beta: float = 0.0) -> dict[str, float]:
        cfg = self.cfg
        cf = cfg.ppo.pg_mode in ("cf_all_action", "cf_shuffled") and q_cf is not None
        queried = cfg.ppo.pg_mode == "cf_queried_perturb" and q_cf is not None
        distill = cfg.ppo.pg_mode == "landscape_distill" and distill_landscape is not None
        t = self.buffer.flat_tensors(advantages, returns, q_cf if cf else None)
        idx = np.arange(self.batch)

        # Advantage normalisation scope. "batch" whitens once over the whole
        # rollout, which preserves the RELATIVE size of advantages between
        # minibatches -- on a sparse task that matters, because per-minibatch
        # whitening makes a minibatch containing the one rewarded transition
        # look exactly as informative as a minibatch of pure zeros.
        adv_std_raw = float(t["advantages"].std())
        if cfg.ppo.norm_adv == "batch":
            t["advantages"], adv_std_raw = self._normalise(
                t["advantages"], cfg.ppo.norm_adv_min_std
            )

        # E2 injects a detached counterfactual landscape through the ordinary
        # sampled-action PPO loss. `queried` uses the exact oracle landscape on
        # its uniform 2% mask; `distill` uses the student prediction everywhere.
        # Both use the SAME raw-GAE scale and are never independently whitened.
        perturb_centering = perturb_mean_abs = perturb_payload_rms = float("nan")
        perturb_scope_fraction = 0.0
        distill_centering = distill_mean_abs = float("nan")
        if queried:
            b = torch.as_tensor(q_cf, dtype=torch.float32, device=self.device)
            m = torch.as_tensor(cf_mask, dtype=torch.bool, device=self.device)
            if b.shape != t["probs"].shape or m.shape != t["actions"].shape:
                raise ValueError("queried CF labels do not match the rollout batch")
            b = b - (t["probs"] * b).sum(-1, keepdim=True)
            b = b * m[:, None]
            with torch.no_grad():
                perturb_centering = float((t["probs"][m] * b[m]).sum(-1).abs().max()) if bool(m.any()) else 0.0
                perturb_mean_abs = float(b[m].abs().mean()) if bool(m.any()) else 0.0
                perturb_scope_fraction = float(m.float().mean())
            taken = b.gather(1, t["actions"][:, None]).squeeze(1)
            payload = float(perturb_beta) * taken / max(adv_std_raw, cfg.ppo.norm_adv_min_std)
            t["advantages"] = t["advantages"] + payload.detach()
            with torch.no_grad():
                perturb_payload_rms = float(payload.square().mean().sqrt())

        if distill:
            b = torch.as_tensor(distill_landscape, dtype=torch.float32, device=self.device)
            if b.shape != t["probs"].shape:
                raise ValueError(f"distill landscape has shape {tuple(b.shape)}, expected {tuple(t['probs'].shape)}")
            with torch.no_grad():
                distill_centering = float((t["probs"] * b).sum(-1).abs().max())
                distill_mean_abs = float(b.abs().mean())
                perturb_centering = distill_centering
                perturb_mean_abs = distill_mean_abs
                perturb_scope_fraction = 1.0
            taken = b.gather(1, t["actions"][:, None]).squeeze(1)
            payload = float(perturb_beta) * taken / max(adv_std_raw, cfg.ppo.norm_adv_min_std)
            t["advantages"] = t["advantages"] + payload.detach()
            with torch.no_grad():
                perturb_payload_rms = float(payload.square().mean().sqrt())

        # --- counterfactual advantages, centred on the BEHAVIOUR policy ---- #
        # Eq (4): A_CF(s,a) = Q_g(s,a) - sum_b pi_old(b|s) Q_g(s,b), so that
        # sum_a pi_old(a|s) A_CF(s,a) = 0 exactly, per state. That identity is
        # what makes the all-action gradient unbiased w.r.t. action sampling,
        # so the scaling below is SCALE-ONLY -- subtracting a batch mean would
        # destroy it.
        #
        # SCALE. Plan rule 4 requires identical advantage-scale conventions
        # across the paired arms, and Eq (6) adds L_PPO and L_CF together, so
        # they must live in the same units. A_CF is therefore divided by the
        # SAME number the GAE advantages were divided by -- not by its own RMS.
        # Whitening A_CF separately inflates a weak counterfactual signal to
        # full gradient strength, which is the same failure as the original
        # advantage-normalisation bug.
        cf_centering = cf_mean_abs = cf_corr_gae = float("nan")
        cf_frac = 0.0
        if cf:
            t["cf_mask"] = torch.as_tensor(cf_mask, dtype=torch.bool, device=self.device)
            v_pi = (t["probs"] * t["q_cf"]).sum(-1, keepdim=True)
            a_cf = (t["q_cf"] - v_pi) / max(adv_std_raw, cfg.ppo.norm_adv_min_std)
            a_cf = a_cf * t["cf_mask"][:, None]
            t["a_cf"] = a_cf
            with torch.no_grad():
                m = t["cf_mask"]
                cf_frac = float(m.float().mean())
                cf_centering = float((t["probs"][m] * a_cf[m]).sum(-1).abs().max())
                cf_mean_abs = float(a_cf[m].abs().mean())
                taken = a_cf[m].gather(1, t["actions"][m, None]).squeeze(1)
                g_adv = t["advantages"][m]
                if len(taken) > 2 and float(taken.std()) > 0 and float(g_adv.std()) > 0:
                    cf_corr_gae = float(torch.corrcoef(torch.stack([taken, g_adv]))[0, 1])

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

                if cf:
                    # ---- Eq (5): all-action counterfactual surrogate ------- #
                    #   L_CF = -mean_s sum_a pi_old(a|s)
                    #            min( rho_a A_CF, clip(rho_a, 1+-eps) A_CF )
                    # with rho_a = pi_theta(a|s) / pi_old(a|s). Every action
                    # contributes weighted by its probability, so the variance
                    # from ACTION SAMPLING is gone. Only defined on the
                    # subsampled rows; plain PPO carries the rest.
                    logp_all, entropy, newvalue = self.model.evaluate_all_actions(t["obs"][mb])
                    newlogprob = logp_all.gather(1, t["actions"][mb, None]).squeeze(1)
                    m = t["cf_mask"][mb]
                    if bool(m.any()):
                        pi_b = t["probs"][mb][m]
                        a_cf_mb = t["a_cf"][mb][m]
                        ratio_all = (logp_all[m] - torch.log(pi_b.clamp_min(1e-12))).exp()
                        surr = torch.min(
                            ratio_all * a_cf_mb,
                            torch.clamp(ratio_all, 1 - cfg.ppo.clip_coef,
                                        1 + cfg.ppo.clip_coef) * a_cf_mb,
                        )
                        cf_row = -(pi_b * surr).sum(-1)      # per-row, not meaned
                    else:
                        cf_row = None
                else:
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
                if cfg.ppo.norm_adv == "minibatch":
                    mb_adv, _ = self._normalise(mb_adv, cfg.ppo.norm_adv_min_std)

                ppo_row = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - cfg.ppo.clip_coef, 1 + cfg.ppo.clip_coef),
                )

                #     L_actor = alpha_gae * L_PPO + alpha_cf * L_CF
                #
                # Independent weights rather than the convex (1 - alpha, alpha)
                # of Eq (6), so each term can be moved on its own; the defaults
                # 0.5 / 0.5 reproduce the plan exactly.
                #
                # The weighting is applied PER ROW, only where a counterfactual
                # exists. Eq (5) averages L_CF over the whole batch; we can only
                # afford it on a subsample, and weighting globally would instead
                # scale the PPO term on every row while the counterfactual
                # informed 10% of them -- handicapping the treatment arm relative
                # to the control for a reason that has nothing to do with the
                # hypothesis. Rows without a counterfactual get plain PPO at full
                # weight, so the two arms are identical wherever the oracle is
                # silent.
                if cf and cf_row is not None:
                    row = ppo_row.clone()
                    row[m] = cfg.ppo.alpha_gae * ppo_row[m] + cfg.ppo.alpha_cf * cf_row
                    pg_loss = row.mean()
                else:
                    pg_loss = ppo_row.mean()

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
                if self.separate_grad_clip:
                    gn_a = nn.utils.clip_grad_norm_(self.model.actor_parameters(), cfg.ppo.max_grad_norm)
                    gn_c = nn.utils.clip_grad_norm_(self.model.critic_parameters(), cfg.ppo.max_grad_norm)
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
            # The raw spread of the advantages before any whitening. If this is
            # at the 1e-6 floor, the rollout contained no learning signal at
            # all, and any movement in entropy/clipfrac that update is noise.
            "adv_std_raw": adv_std_raw,
            # PPO-CF only. cf_centering must stay at floating-point zero; if it
            # drifts, the all-action gradient has acquired a state-dependent
            # bias and the run is invalid.
            "cf_centering": cf_centering,
            "cf_mean_abs": cf_mean_abs,
            # Correlation between A_CF on the taken action and the GAE
            # advantage. Near zero means the counterfactual estimate carries no
            # information the sampled advantage does not -- the symptom that
            # killed the one-step version (measured 0.098).
            "cf_corr_gae": cf_corr_gae,
            "cf_frac_states": cf_frac,
            # E2 only. The two arms differ solely in perturbation coverage:
            # queried=2% exact teacher labels; distill=all-state student output.
            # The absence of uncertainty here is intentional; E3 changes only
            # the query selector.
            "perturb_beta": float(perturb_beta) if (queried or distill) else float("nan"),
            "perturb_scope_fraction": perturb_scope_fraction,
            "perturb_centering": perturb_centering,
            "perturb_mean_abs": perturb_mean_abs,
            "perturb_payload_rms": perturb_payload_rms,
            "distill_centering": distill_centering,
            "distill_mean_abs": distill_mean_abs,
        }

    # ------------------------------------------------------------------- train #

    def _update_schedules(self, observed_entropy: float, update: int) -> None:
        """Advance the success-proportional schedules.

        Two things ride the same schedule, and they are independent of each
        other: the per-action probability floor (always) and the adaptive
        entropy target (only when ent_mode == "adaptive"). The floor used to be
        updated inside the adaptive branch, which silently pinned it at
        `prob_floor_start` forever whenever ent_mode was "fixed" -- so a floor
        set for DoorKey never annealed.

        The anneal is proportional to MEASURED SUCCESS, smoothed, not to
        wall-clock:  target = start + (end - start) * clip(success/full, 0, 1).
        A single lucky success moves it barely at all, which matters: on Taxi an
        event-based anneal fired on a random delivery at ~6k steps, entropy fell
        to ~0.1, and the policy sat in the trap for the remaining 950k steps.
        """
        cfg = self.cfg.ppo
        rate = float(np.mean(self._ep_success)) if len(self._ep_success) >= 20 else 0.0
        self._success_ema += cfg.ent_success_ema * (rate - self._success_ema)
        frac = float(np.clip(self._success_ema / max(cfg.ent_anneal_full_rate, 1e-9), 0.0, 1.0))

        self.model.prob_floor = float(
            cfg.prob_floor_start + frac * (cfg.prob_floor_end - cfg.prob_floor_start)
        )
        if cfg.ent_mode != "adaptive":
            return

        max_ent = float(np.log(self.pool.n_actions))
        t0 = cfg.target_entropy_frac_start * max_ent
        t1 = cfg.target_entropy_frac_end * max_ent
        target = t0 + frac * (t1 - t0)
        self._entropy_target = target
        err = target - observed_entropy
        self.ent_coef = float(
            np.clip(self.ent_coef * np.exp(cfg.ent_adapt_rate * np.sign(err) * min(abs(err) / max(target, 1e-8), 1.0) * 4.0),
                    cfg.ent_coef_min, cfg.ent_coef_max)
        )

    # -------------------------------------------------------- warm start #

    def _warm_start(self, spec: str) -> None:
        """Load weights from an earlier run's checkpoint (the curriculum path).

        Tensors whose shape does not match the current model are SKIPPED rather
        than raising, because moving between grid sizes can change the encoder's
        output width. Exactly what was and was not loaded is printed, so a
        silently-empty warm start is impossible to miss.
        """
        from config import PROJECT_ROOT

        path = Path(str(spec).format(seed=self.seed))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(
                f"run.init_from points at {path}, which does not exist. Train the "
                "previous rung of the curriculum first, or set init_from: null."
            )

        blob = torch.load(path, map_location=self.device, weights_only=False)
        src = blob["model_state_dict"]
        dst = self.model.state_dict()
        loaded, skipped = [], []
        for k, v in src.items():
            if k in dst and dst[k].shape == v.shape:
                dst[k] = v
                loaded.append(k)
            else:
                skipped.append(k)
        self.model.load_state_dict(dst)
        if self.cfg.run.init_load_optimizer and blob.get("optimizer_state_dict"):
            try:
                self.optimizer.load_state_dict(blob["optimizer_state_dict"])
            except ValueError as e:
                print(f"  warm start: optimizer state not loaded ({e})")
        if self.cfg.run.init_strict and skipped:
            raise ValueError(f"warm start skipped {len(skipped)} tensors: {skipped}")
        print(f"  warm start from {path}")
        print(f"    loaded {len(loaded)} tensors, skipped {len(skipped)}"
              + (f": {skipped}" if skipped else ""))
        if not loaded:
            raise ValueError("warm start loaded nothing -- the architectures are incompatible")

    # ------------------------------------------------------- greedy eval #

    @torch.no_grad()
    def _greedy_eval(self, n_episodes: int) -> tuple[float, float]:
        """(mean return, success rate) of the argmax policy on held-out seeds.

        Training success rate is measured under the stochastic policy, which on
        a sparse task can hide a policy that has actually learned something.
        Eval seeds start at 900_000 so they never collide with training layouts.
        """
        env = make_env(self.cfg.env.env_id, self.cfg.env.max_episode_steps, **self.env_kwargs)
        rets, succ = [], 0
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=900_000 + ep)
            total = 0.0
            while True:
                x = torch.as_tensor(self._scale(np.asarray(obs).ravel()[None, :]),
                                    dtype=torch.float32, device=self.device)
                a = int(self.model.action_probs(x)[0].argmax())
                obs, r, term, trunc, _ = env.step(a)
                total += float(r)
                if term:
                    succ += 1
                    break
                if trunc:
                    break
            rets.append(total)
        env.close()
        return float(np.mean(rets)), succ / max(n_episodes, 1)

    def _maybe_checkpoint(self) -> None:
        for target_step, frac in self._ckpt_targets:
            if frac in self._ckpt_done:
                continue
            if self.global_step >= target_step:
                # A count bonus is not recoverable from V, so a checkpoint the
                # oracle might use must not have been trained under one.
                if (self.cfg.reward.assert_zero_at_checkpoints
                        and self.pool.bonus_coef > 0.0):
                    raise RuntimeError(
                        f"checkpoint at {frac:.0%} would be saved while the count "
                        f"bonus is still {self.pool.bonus_coef:.4g}. The critic it "
                        "stores is V for the bonus-augmented MDP, which NB02 cannot "
                        "undo. Lower reward.count_bonus_anneal_frac below "
                        f"{frac:.2f}, or set reward.assert_zero_at_checkpoints: false "
                        "if you accept that this checkpoint is unusable for the oracle."
                    )
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
            self.pool.set_progress(self.global_step / max(cfg.ppo.total_timesteps, 1))
            self.collect_rollout()
            adv, ret = self.compute_advantages()

            if self.recorder is not None:
                self.recorder.add_rollout(self.buffer, gstep_start, cfg.env.n_envs)

            q_cf = cf_mask = None
            cf_diag = {}
            distill_landscape = None
            # The exact queried arm uses its teacher perturbation immediately.
            # The student arm has its fixed label warm-up/ramp because it must
            # learn from earlier rollouts before predicting this one.
            perturb_beta = (float(cfg.distill.beta)
                              if cfg.ppo.pg_mode == "cf_queried_perturb" else 0.0)
            # Predict before querying/learning current labels.  This ordering
            # is the E2 non-leakage contract.
            if self.landscape is not None:
                distill_landscape, perturb_beta = self.predict_distilled_landscape(update)
            if self.oracle is not None:
                q_cf, cf_mask, cf_diag = self.compute_counterfactual()
                if not self._cf_validated:
                    self._validate_oracle(q_cf, cf_mask)
                    self._cf_validated = True

            stats = self.update(
                adv, ret, q_cf, cf_mask,
                distill_landscape=distill_landscape,
                perturb_beta=perturb_beta,
            )
            stats.update({f"cf_{k}": float(v) for k, v in cf_diag.items()})
            if self.landscape is not None:
                assert q_cf is not None and cf_mask is not None and distill_landscape is not None
                stats.update(self.learn_distilled_landscape(q_cf, cf_mask, distill_landscape))
            self._update_schedules(stats["entropy"], update)
            self._maybe_checkpoint()

            if cfg.run.eval_every_updates and update % cfg.run.eval_every_updates == 0:
                self._eval_return, self._eval_success = self._greedy_eval(cfg.run.eval_episodes)

            if update % cfg.run.log_every_updates == 0 or update == self.n_updates:
                row = {
                    "update": update,
                    "global_step": self.global_step,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "mean_return_100": float(np.mean(self._ep_returns)) if self._ep_returns else float("nan"),
                    "mean_length_100": float(np.mean(self._ep_lengths)) if self._ep_lengths else float("nan"),
                    "success_rate_100": float(np.mean(self._ep_success)) if self._ep_success else 0.0,
                    # Sub-goal rates: the rungs between "did nothing" and
                    # "solved". On DoorKey, key_rate rising while door_rate
                    # stays flat is a completely different failure from both
                    # staying flat, and they need opposite responses.
                    "subgoal1_rate_100": (float(np.mean(self._ep_subgoal1))
                                          if self._ep_subgoal1 else float("nan")),
                    "subgoal2_rate_100": (float(np.mean(self._ep_subgoal2))
                                          if self._ep_subgoal2 else float("nan")),
                    "n_episodes": len(self.episode_log),
                    "entropy_target": float(getattr(self, "_entropy_target", float("nan"))),
                    "success_ema": float(self._success_ema),
                    "prob_floor": float(self.model.prob_floor),
                    "bonus_coef": float(self.pool.bonus_coef),
                    "eval_return": float(getattr(self, "_eval_return", float("nan"))),
                    "eval_success": float(getattr(self, "_eval_success", float("nan"))),
                    "sps": int(self.global_step / max(time.time() - t0, 1e-9)),
                    **stats,
                }
                self.logger.log(row)
                if self.progress:
                    print(
                        f"  upd {update:>5}/{self.n_updates}  step {self.global_step:>8,}  "
                        f"ret {row['mean_return_100']:>7.3f}  succ {row['success_rate_100']:.2f}  "
                        f"sg1 {row['subgoal1_rate_100']:.2f} sg2 {row['subgoal2_rate_100']:.2f}  "
                        f"ent {row['entropy']:.3f}  advstd {row['adv_std_raw']:.2e}  "
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
        if self.oracle is not None:
            self.oracle.close()
        return artifacts
