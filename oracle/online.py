"""The counterfactual oracle, evaluated INSIDE the training loop.

`oracle/counterfactual.py` computes the same quantities against a FROZEN
checkpoint, for NB02's landscape. This module computes them against the LIVE
critic every rollout, which is what PPO-CF needs.

    Q_CF(s,a) = r(s,a) + gamma * V_phi(s'_a) * (1 - terminated_a)
    V_pi(s)   = sum_a pi_behav(a|s) Q_CF(s,a)
    A_CF(s,a) = Q_CF(s,a) - V_pi(s)

Three things worth knowing before trusting a number that comes out of here.

1. THE CENTERING USES THE ORACLE'S OWN V_pi, NOT THE CRITIC'S V(s). That makes
   sum_a pi(a|s) A_CF(s,a) = 0 exact by construction, which is what lets the
   all-action policy gradient be unbiased with respect to action sampling. It
   also means A_CF is only the true A^pi to the extent the critic is good --
   every deviation is critic error, not oracle error.

2. STEP COUNT MUST BE RESTORED, NOT ZEROED. `oracle/counterfactual.py` restores
   with `elapsed_steps=0`. On MountainCar that is harmless because every reward
   is -1. On MiniGrid it is NOT: DoorKey pays `1 - 0.9 * (step_count/max_steps)`
   on success, so zeroing the step count makes a counterfactual success at step
   200 look worth 1.0 instead of 0.72, and it also disables MiniGrid's internal
   truncation. This module restores the state's own step count.

3. THE ORACLE ASSUMES DETERMINISTIC DYNAMICS. `envs.env_pool.assert_deterministic`
   checks that rather than assuming it. MiniGrid is deterministic.

Cost. K restores + K steps per collected transition. Measured on
DoorKey-5x5 (single env, 7 actions):

    plain env.step                6,045 /s
    reset+restore ("exact")       1,733 /s   -> ~220 collected steps/s
    direct restore ("fast")       3,368 /s   -> ~380 collected steps/s

"exact" is the path NB02 validated. "fast" skips the `env.reset()` that
`set_sim_state` performs before every restore; `check_restore_equivalence`
below asserts the two produce bit-identical transitions before you rely on it.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from envs.env_pool import get_sim_state, make_env, set_sim_state


class OnlineOracle:
    """All-action one-step counterfactuals against the live critic."""

    def __init__(
        self,
        env_id: str,
        n_actions: int,
        gamma: float,
        env_kwargs: dict | None = None,
        max_episode_steps: int | None = None,
        restore: str = "exact",
        seed: int = 0,
    ):
        if restore not in ("exact", "fast"):
            raise ValueError(f"restore must be 'exact' or 'fast', got {restore!r}")
        self.env_id = env_id
        self.n_actions = int(n_actions)
        self.gamma = float(gamma)
        self.restore = restore
        self._env_kwargs = dict(env_kwargs or {})
        self._max_steps = max_episode_steps
        self.env = make_env(env_id, max_episode_steps, **self._env_kwargs)
        self.env.reset(seed=seed)          # allocate grid / internal buffers once
        self.obs_dim = int(np.prod(self.env.observation_space.shape))
        self._is_minigrid = env_id.startswith("MiniGrid")
        if restore == "fast" and not self._is_minigrid:
            raise ValueError("restore='fast' is implemented for MiniGrid only")

    # ------------------------------------------------------------ restore #

    @staticmethod
    def _step_count_of(sim_state: np.ndarray) -> int:
        """Last entry of the packed MiniGrid state is step_count."""
        return int(round(float(sim_state[-1])))

    def _restore(self, sim_state: np.ndarray) -> None:
        self._restore_into(self.env, sim_state)

    def _restore_into(self, env, sim_state: np.ndarray) -> None:
        elapsed = self._step_count_of(sim_state) if self._is_minigrid else 0
        if self.restore == "exact":
            set_sim_state(env, sim_state, elapsed_steps=elapsed)
            return
        # fast: skip the env.reset() that set_sim_state does before every restore
        from envs.minigrid_env import set_minigrid_state
        set_minigrid_state(env, sim_state, elapsed_steps=elapsed)
        u = env.unwrapped
        w = env
        while w is not u:                       # keep any TimeLimit counter in sync
            if hasattr(w, "_elapsed_steps"):
                w._elapsed_steps = elapsed
            w = getattr(w, "env", u)

    # --------------------------------------------------------- transitions #

    def transitions(self, sim_states: np.ndarray) -> dict[str, np.ndarray]:
        """(M, K) reward / terminated / truncated and (M, K, D) successor obs.

        The environment work. No network is touched here, so the caller can
        evaluate the critic on all M*K successors in ONE batched forward pass
        instead of M*K small ones.
        """
        sim_states = np.asarray(sim_states, dtype=np.float64)
        M, K, D = len(sim_states), self.n_actions, self.obs_dim

        next_obs = np.zeros((M, K, D), dtype=np.float32)
        reward = np.zeros((M, K), dtype=np.float32)
        terminated = np.zeros((M, K), dtype=bool)
        truncated = np.zeros((M, K), dtype=bool)

        for m in range(M):
            s = sim_states[m]
            for a in range(K):
                # Restore before EVERY action: the previous one moved the sim.
                self._restore(s)
                o, r, term, trunc, _ = self.env.step(a)
                next_obs[m, a] = np.asarray(o, dtype=np.float32).ravel()
                reward[m, a] = r
                terminated[m, a] = term
                truncated[m, a] = trunc

        return {"next_obs": next_obs, "reward": reward,
                "terminated": terminated, "truncated": truncated}

    # ------------------------------------------------------------ the maths #

    def q_cf(self, sim_states: np.ndarray,
             value_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """(M, K) counterfactual action-values under the current critic.

        `value_fn` maps raw observations (N, D) -> values (N,). Note the
        bootstrap is killed on TERMINATION only: truncation is not an MDP
        terminal, so V(s') still applies there.
        """
        t = self.transitions(sim_states)
        M, K = t["reward"].shape
        v_next = np.asarray(value_fn(t["next_obs"].reshape(M * K, -1)),
                            dtype=np.float32).reshape(M, K)
        return t["reward"] + self.gamma * v_next * (~t["terminated"])

    def a_cf(self, sim_states: np.ndarray, pi: np.ndarray,
             value_fn: Callable[[np.ndarray], np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """(A_CF, Q_CF), each (M, K). `pi` is the BEHAVIOUR policy at collection."""
        q = self.q_cf(sim_states, value_fn)
        v_pi = (np.asarray(pi, dtype=np.float32) * q).sum(axis=1)
        return q - v_pi[:, None], q

    # ------------------------------------------- Eq (3): the specified Q_g #

    def _branch_pool(self, n: int):
        """Persistent pool of envs, so branches step in lockstep and the policy
        is evaluated in ONE batched forward per timestep instead of one per
        branch. Creating envs is not free, so the pool is reused."""
        have = len(getattr(self, "_pool", []))
        if have < n:
            if not hasattr(self, "_pool"):
                self._pool = []
            for _ in range(n - have):
                e = make_env(self.env_id, self._max_steps, **self._env_kwargs)
                e.reset(seed=0)
                self._pool.append(e)
        return self._pool[:n]

    def q_g(
        self,
        sim_states: np.ndarray,
        probs_fn: Callable[[np.ndarray], np.ndarray],
        value_fn: Callable[[np.ndarray], np.ndarray],
        horizon: int = 16,
        n_rollouts: int = 2,
        seed: int = 0,
        bootstrap_tail: bool = True,
        chunk: int = 256,
    ) -> tuple[np.ndarray, dict]:
        """Q_g(s,a) = E[ sum_l gamma^l r_{t+l} | do(a_t = a), pi thereafter ].

        This is Equation (3) of the plan: branch from the saved state, FORCE
        action a, then follow the current policy for `horizon` steps, averaging
        over `n_rollouts` continuations. The one-step form r + gamma*V(s') is
        NOT this quantity -- on a task whose reward is purely terminal it
        contains no reward at all outside the states adjacent to the goal
        (measured on DoorKey-5x5: 99.46% of states), which makes A_CF pure
        critic difference.

        COMMON RANDOM NUMBERS. The quantity that matters is the DIFFERENCE
        Q_g(s,a) - Q_g(s,a'), and the apples-and-noise variance of the return
        itself is far larger than that difference. So the uniform used to sample
        the policy at step t of rollout m from state i is the SAME for every
        action a: `U[i, m, t]`, drawn once. The branches still diverge, but from
        a shared source of randomness, which is what makes the paired difference
        low-variance at small `n_rollouts`.

        The tail is bootstrapped with the critic on branches that neither
        terminated nor truncated within the horizon, and on truncation (a time
        limit is not an MDP terminal). Termination gets no bootstrap.
        """
        sim_states = np.asarray(sim_states, dtype=np.float64)
        M, K, R, H = len(sim_states), self.n_actions, int(n_rollouts), int(horizon)
        g = self.gamma

        U = np.random.default_rng(seed).random((M, R, H))
        items = [(i, a, m) for i in range(M) for a in range(K) for m in range(R)]
        total = np.zeros(len(items))
        terminated_in_h = np.zeros(len(items), dtype=bool)
        got_reward = np.zeros(len(items), dtype=bool)
        branch_transitions = 0

        for start in range(0, len(items), chunk):
            block = items[start : start + chunk]
            B = len(block)
            envs = self._branch_pool(B)
            obs = np.zeros((B, self.obs_dim), dtype=np.float32)
            disc = np.full(B, g)
            run = np.ones(B, dtype=bool)      # still stepping
            boot = np.ones(B, dtype=bool)     # eligible for a critic tail
            acc = np.zeros(B)

            # forced first action
            for j, (i, a, _m) in enumerate(block):
                self._restore_into(envs[j], sim_states[i])
                o, r, term, trunc, _ = envs[j].step(a)
                branch_transitions += 1
                obs[j] = np.asarray(o, dtype=np.float32).ravel()
                acc[j] = r
                got_reward[start + j] = r != 0.0
                if term:
                    run[j] = boot[j] = False
                    terminated_in_h[start + j] = True
                elif trunc:
                    run[j] = False

            for t in range(H - 1):
                idx = np.flatnonzero(run)
                if idx.size == 0:
                    break
                p = np.asarray(probs_fn(obs[idx]), dtype=np.float64)
                u = np.array([U[block[j][0], block[j][2], t] for j in idx])
                a_t = (np.cumsum(p, axis=1) < u[:, None]).sum(axis=1).clip(0, K - 1)
                for n, j in enumerate(idx):
                    o, r, term, trunc, _ = envs[j].step(int(a_t[n]))
                    branch_transitions += 1
                    obs[j] = np.asarray(o, dtype=np.float32).ravel()
                    acc[j] += disc[j] * r
                    if r != 0.0:
                        got_reward[start + j] = True
                    disc[j] *= g
                    if term:
                        run[j] = boot[j] = False
                        terminated_in_h[start + j] = True
                    elif trunc:
                        run[j] = False

            if bootstrap_tail and boot.any():
                sel = np.flatnonzero(boot)
                acc[sel] += disc[sel] * np.asarray(value_fn(obs[sel]), dtype=np.float64)
            total[start : start + B] = acc

        q = total.reshape(M, K, R).mean(axis=2).astype(np.float32)
        diag = {
            # The informativeness gate: what fraction of branches actually saw
            # reward? If this is ~0 the estimator has degenerated back to a
            # critic difference and no amount of tuning will help.
            "reward_coverage": float(got_reward.mean()),
            "terminated_within_horizon": float(terminated_in_h.mean()),
            "frac_states_any_reward": float(
                got_reward.reshape(M, K, R).any(axis=(1, 2)).mean()),
            # Budget honesty: terminal branches cost fewer than H steps, so
            # this observed count, not K*R*H, is the authoritative total.
            "branch_transitions": float(branch_transitions),
        }
        return q, diag

    def a_g(self, sim_states: np.ndarray, pi: np.ndarray, probs_fn, value_fn,
            **kw) -> tuple[np.ndarray, np.ndarray, dict]:
        """(A_CF, Q_g, diagnostics), centred on the behaviour policy per Eq (4)."""
        q, diag = self.q_g(sim_states, probs_fn, value_fn, **kw)
        v_pi = (np.asarray(pi, dtype=np.float32) * q).sum(axis=1)
        return q - v_pi[:, None], q, diag

    def close(self) -> None:
        self.env.close()
        for e in getattr(self, "_pool", []):
            e.close()
        self._pool = []


# --------------------------------------------------------------------------- #
# CHECKS. Run these before trusting anything above.
# --------------------------------------------------------------------------- #

def check_replay(oracle: OnlineOracle, sim_states: np.ndarray, actions: np.ndarray,
                 rewards: np.ndarray, next_sim_states: np.ndarray) -> dict:
    """Restore each recorded state, replay the action that was actually taken,
    and compare against what the training run recorded.

    This is the check that matters. If it fails, every A_CF in this project is
    meaningless, and nothing downstream can detect it -- the numbers will look
    perfectly reasonable and be wrong.
    """
    n = len(sim_states)
    r_err = np.zeros(n)
    s_err = np.zeros(n)
    for i in range(n):
        oracle._restore(sim_states[i])
        _o, r, _t, _tr, _ = oracle.env.step(int(actions[i]))
        # Rollout rewards are stored and consumed as float32.  Comparing the
        # raw Python reward against that stored representation falsely rejects
        # a bit-exact restore whenever a MiniGrid success reward rounds by one
        # float32 ulp (observed on Unlock CF seed 2).
        r_err[i] = abs(float(np.float32(r)) - float(np.float32(rewards[i])))
        s_err[i] = np.abs(get_sim_state(oracle.env) - next_sim_states[i]).max()
    return {
        "n": n,
        "max_reward_error": float(r_err.max()),
        "max_state_error": float(s_err.max()),
        "exact": bool(r_err.max() == 0.0 and s_err.max() == 0.0),
    }


def check_restore_equivalence(env_id: str, n_actions: int, gamma: float,
                              sim_states: np.ndarray, env_kwargs: dict | None = None,
                              max_episode_steps: int | None = None) -> dict:
    """Do restore='fast' and restore='exact' produce identical transitions?

    'fast' is ~2x quicker because it skips an env.reset() per restore. That is
    only worth having if it is bit-identical, so verify rather than assume.
    """
    out = {}
    for kind in ("exact", "fast"):
        o = OnlineOracle(env_id, n_actions, gamma, env_kwargs, max_episode_steps, restore=kind)
        out[kind] = o.transitions(sim_states)
        o.close()
    return {
        "max_reward_diff": float(np.abs(out["exact"]["reward"] - out["fast"]["reward"]).max()),
        "max_obs_diff": float(np.abs(out["exact"]["next_obs"] - out["fast"]["next_obs"]).max()),
        "terminated_agreement": float((out["exact"]["terminated"] == out["fast"]["terminated"]).mean()),
        "identical": bool(
            np.array_equal(out["exact"]["reward"], out["fast"]["reward"])
            and np.array_equal(out["exact"]["next_obs"], out["fast"]["next_obs"])
            and np.array_equal(out["exact"]["terminated"], out["fast"]["terminated"])
        ),
    }


def check_centering(a_cf: np.ndarray, pi: np.ndarray) -> dict:
    """sum_a pi(a|s) A_CF(s,a) must be 0 to floating-point precision.

    If it is not, the all-action policy gradient has a state-dependent bias
    term and the whole construction is unsound.
    """
    c = np.abs((np.asarray(pi) * np.asarray(a_cf)).sum(axis=1))
    return {"max_abs": float(c.max()), "mean_abs": float(c.mean()),
            "ok": bool(c.max() < 1e-4)}


def landscape_summary(a_cf: np.ndarray, q_cf: np.ndarray, pi: np.ndarray,
                      spread_threshold: float = 0.01,
                      useless_actions: tuple[int, ...] = ()) -> dict:
    """Is the landscape non-degenerate, and does it say sensible things?

    `useless_actions` is a sanity probe rather than a gate: in DoorKey, `drop`
    (4) and `done` (6) can never help, so a correct oracle should rank them
    below average almost everywhere. If it does not, suspect the oracle before
    suspecting the environment.
    """
    a_cf, q_cf, pi = np.asarray(a_cf), np.asarray(q_cf), np.asarray(pi)
    spread = q_cf.max(axis=1) - q_cf.min(axis=1)
    out = {
        "n_states": int(len(a_cf)),
        "mean_abs_a_cf": float(np.abs(a_cf).mean()),
        "max_abs_a_cf": float(np.abs(a_cf).max()),
        "frac_states_with_spread": float((spread > spread_threshold).mean()),
        "mean_q_spread": float(spread.mean()),
        "finite": bool(np.all(np.isfinite(a_cf)) and np.all(np.isfinite(q_cf))),
        "best_action_counts": np.bincount(a_cf.argmax(axis=1),
                                          minlength=a_cf.shape[1]).tolist(),
    }
    if useless_actions:
        frac = float(np.mean([(a_cf[:, a] < 0).mean() for a in useless_actions]))
        out["useless_actions_negative_frac"] = frac
    return out
