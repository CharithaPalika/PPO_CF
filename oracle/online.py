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
        self.env = make_env(env_id, max_episode_steps, **(env_kwargs or {}))
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
        elapsed = self._step_count_of(sim_state) if self._is_minigrid else 0
        if self.restore == "exact":
            set_sim_state(self.env, sim_state, elapsed_steps=elapsed)
            return
        # fast: skip the env.reset() that set_sim_state does before every restore
        from envs.minigrid_env import set_minigrid_state
        set_minigrid_state(self.env, sim_state, elapsed_steps=elapsed)
        u = self.env.unwrapped
        w = self.env
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

    def close(self) -> None:
        self.env.close()


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
        r_err[i] = abs(float(r) - float(rewards[i]))
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
