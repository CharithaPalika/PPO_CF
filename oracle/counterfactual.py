"""The explicit, simulator-based counterfactual oracle (Notebook 02).

For each evaluation state s, restore the simulator into s, execute every action
a in {0..K-1} once, and form

    Q_CF(s,a) = r(s,a) + gamma * V(s')          with V = 0 at a terminal s'
    V_pi(s)   = sum_a pi(a|s) Q_CF(s,a)
    A_CF(s,.) = Q_CF(s,.) - V_pi(s)

Two things about this definition are worth being explicit about, because
everything in Notebooks 03-06 is measured against it.

1. The centering uses V_pi built from the oracle's own Q_CF, NOT the critic's
   V(s). That makes  sum_a pi(a|s) A_CF(s,a) = 0  true by construction, which is
   the identity Gate 2 checks. Centering on the critic instead would leave a
   per-state offset equal to the critic's error.

2. It is a ONE-STEP oracle: exact for the reward and the transition, approximate
   in V(s'). So A_CF inherits the critic's error at the successor states, and it
   is not the true A^pi unless the critic is good. `mc_reference` below exists to
   measure that gap rather than assume it away.
"""

from __future__ import annotations

import numpy as np

from dataio.checkpoint import Checkpoint
from envs.env_pool import get_sim_state, make_env, set_sim_state


# --------------------------------------------------------------------------- #
# The oracle
# --------------------------------------------------------------------------- #

def compute_landscape(
    ck: Checkpoint,
    sim_states: np.ndarray,
    env_id: str,
    max_episode_steps: int | None,
    gamma: float,
    n_actions: int,
    progress_every: int = 0,
) -> dict:
    """All-action counterfactual landscape for a batch of simulator states."""
    sim_states = np.asarray(sim_states, dtype=np.float64)
    N, d = sim_states.shape
    K = n_actions

    env = make_env(env_id, max_episode_steps)

    obs0 = np.zeros((N, d), dtype=np.float32)
    next_obs = np.zeros((N, K, d), dtype=np.float32)
    next_sim = np.zeros((N, K, d), dtype=np.float64)
    reward = np.zeros((N, K), dtype=np.float32)
    terminated = np.zeros((N, K), dtype=bool)
    truncated = np.zeros((N, K), dtype=bool)

    # WARNING -- elapsed_steps=0 IS WRONG ON MINIGRID.
    # It was correct for MountainCar, where every reward is -1 regardless of
    # when it is collected. DoorKey pays 1 - 0.9*(step_count/max_steps) on
    # success, so zeroing the step count inflates every counterfactual success
    # (measured on DoorKey-5x5: 0.9964 instead of 0.5698, 1.75x) and it also
    # disables MiniGrid's internal truncation. `oracle/online.py` restores the
    # state's own step count and should be followed here before NB02 is re-run
    # on any MiniGrid environment. Left as-is for now so the MountainCar Gate 2
    # results stay reproducible.
    for i in range(N):
        # Restore once to read the observation the policy would see in s.
        obs0[i] = set_sim_state(env, sim_states[i], elapsed_steps=0)
        for a in range(K):
            # Restore before EVERY action: the previous action moved the sim.
            set_sim_state(env, sim_states[i], elapsed_steps=0)
            o, r, term, trunc, _ = env.step(a)
            next_obs[i, a] = o
            next_sim[i, a] = get_sim_state(env)
            reward[i, a] = r
            terminated[i, a] = term
            truncated[i, a] = trunc
        if progress_every and (i + 1) % progress_every == 0:
            print(f"    {i + 1}/{N} states", flush=True)

    env.close()

    # Batched value / policy evaluation -- one forward pass instead of 3N.
    v_next = ck.values(next_obs.reshape(-1, d)).reshape(N, K)
    q_cf = reward + gamma * v_next * (~terminated)          # V(terminal) = 0
    pi = ck.probs(obs0)
    v_pi = (pi * q_cf).sum(axis=1)
    a_cf = q_cf - v_pi[:, None]

    return {
        "sim_state": sim_states,
        "raw_obs": obs0,
        "q_cf": q_cf.astype(np.float32),
        "a_cf": a_cf.astype(np.float32),
        "v_pi": v_pi.astype(np.float32),
        "v_critic": ck.values(obs0).astype(np.float32),
        "pi": pi.astype(np.float32),
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "next_raw_obs": next_obs,
        "next_sim_state": next_sim,
        "v_next": v_next.astype(np.float32),
    }


# --------------------------------------------------------------------------- #
# CHECK: is the restore/step logic actually correct?
# --------------------------------------------------------------------------- #

def validate_restore(
    traj,
    env_id: str,
    max_episode_steps: int | None,
    n: int = 300,
    seed: int = 0,
) -> dict:
    """Replay recorded transitions through the restore path and compare.

    This is the strongest available check on `set_sim_state`, because the
    trajectory dataset holds ground truth: for each recorded row we know the
    simulator state, the action taken, and the resulting state and reward as
    they actually happened during training. Restoring and re-stepping must
    reproduce them exactly (MountainCar's dynamics are deterministic).

    `truncated` is deliberately NOT compared: restoring resets the TimeLimit
    counter to 0, so a transition that was truncated during training is not
    truncated here. That difference is by design, not a bug.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(traj), size=min(n, len(traj)), replace=False)

    env = make_env(env_id, max_episode_steps)
    d_state = np.zeros(len(idx))
    d_obs = np.zeros(len(idx))
    d_reward = np.zeros(len(idx))
    term_match = np.zeros(len(idx), dtype=bool)

    for j, i in enumerate(idx):
        set_sim_state(env, traj.sim_state[i], elapsed_steps=0)
        o, r, term, _trunc, _ = env.step(int(traj.action[i]))
        d_state[j] = np.abs(get_sim_state(env) - traj.next_sim_state[i]).max()
        d_obs[j] = np.abs(np.asarray(o) - traj.next_raw_obs[i]).max()
        d_reward[j] = abs(float(r) - float(traj.reward[i]))
        term_match[j] = bool(term) == bool(traj.terminated[i])
    env.close()

    return {
        "n": len(idx),
        "max_state_err": float(d_state.max()),
        "max_obs_err": float(d_obs.max()),
        "max_reward_err": float(d_reward.max()),
        "terminated_agreement": float(term_match.mean()),
        "passed": bool(d_state.max() < 1e-9 and d_reward.max() < 1e-6 and term_match.all()),
    }


def check_determinism(sim_states: np.ndarray, env_id: str, max_episode_steps, n_actions: int,
                      repeats: int = 3) -> bool:
    """Restoring the same state and taking the same action must be reproducible."""
    env = make_env(env_id, max_episode_steps)
    ok = True
    for s in sim_states[: min(50, len(sim_states))]:
        for a in range(n_actions):
            outs = []
            for _ in range(repeats):
                set_sim_state(env, s, elapsed_steps=0)
                o, r, t, _, _ = env.step(a)
                outs.append((np.asarray(o, dtype=np.float64).copy(), float(r), bool(t)))
            ok &= all(np.array_equal(outs[0][0], o) and outs[0][1] == r and outs[0][2] == t
                      for o, r, t in outs[1:])
    env.close()
    return bool(ok)


# --------------------------------------------------------------------------- #
# Diagnostic: how much of A_CF is critic error?
# --------------------------------------------------------------------------- #

def mc_reference(
    ck: Checkpoint,
    sim_states: np.ndarray,
    env_id: str,
    max_episode_steps: int | None,
    gamma: float,
    n_actions: int,
    n_rollouts: int = 16,
    horizon: int = 200,
    seed: int = 0,
    bootstrap_tail: bool = True,
    chunk: int = 1024,
    return_se: bool = False,
):
    """Monte-Carlo estimate of Q^pi(s,a), returned as (N, K).

    Take action `a` from `s`, then follow pi for up to `horizon` steps, averaging
    over `n_rollouts` sampled trajectories. This does NOT replace the one-step
    oracle -- the plan's oracle is the one-step form. It exists so Gate 2 can
    report how far the one-step Q_CF sits from an estimate that leans on the
    critic only at the tail.

    COMMON RANDOM NUMBERS. The quantity of interest is a DIFFERENCE,
    Q(s,a) - Q(s,a'), and on MountainCar that difference is order 1 while the
    return itself varies by 5-10 across rollouts. Estimating the two terms with
    independent randomness gives a difference whose standard error is larger
    than the signal, and the resulting "disagreement" with the one-step oracle is
    pure noise. So rollout m from state i uses the SAME random stream for every
    action a: `np.random.default_rng([seed, i, m])`, independent of a. The
    trajectories still diverge, but they diverge from a shared source of
    randomness, which is what makes the paired difference low-variance.

    With return_se=True, also returns the per-(s,a) standard error of the mean,
    so you can check the estimate is powered enough to resolve the advantages
    rather than assuming it.
    """
    sim_states = np.asarray(sim_states, dtype=np.float64)
    N, d = sim_states.shape
    K = n_actions

    items = [(i, a, m) for i in range(N) for a in range(K) for m in range(n_rollouts)]
    returns = np.zeros(len(items), dtype=np.float64)

    for start in range(0, len(items), chunk):
        block = items[start : start + chunk]
        B = len(block)
        envs = [make_env(env_id, max_episode_steps) for _ in range(B)]
        # CRN: the stream depends on (state, rollout) but NOT on the action.
        rngs = [np.random.default_rng([seed, i, m]) for (i, _a, m) in block]

        obs = np.zeros((B, d), dtype=np.float32)
        alive = np.ones(B, dtype=bool)
        total = np.zeros(B, dtype=np.float64)
        disc = np.ones(B, dtype=np.float64)

        for j, (i, a, _m) in enumerate(block):
            set_sim_state(envs[j], sim_states[i], elapsed_steps=0)
            o, r, term, trunc, _ = envs[j].step(a)
            total[j] += r
            disc[j] *= gamma
            obs[j] = o
            alive[j] = not (term or trunc)
            if term:
                pass                       # V(terminal) = 0
            elif trunc and bootstrap_tail:
                total[j] += disc[j] * float(ck.values(o)[0])

        for _t in range(horizon):
            if not alive.any():
                break
            live = np.flatnonzero(alive)
            probs = ck.probs(obs[live]).astype(np.float64)
            cum = probs.cumsum(axis=1)
            for k, j in enumerate(live):
                u = rngs[j].random()
                a = int(min((cum[k] < u).sum(), K - 1))
                o, r, term, trunc, _ = envs[j].step(a)
                total[j] += disc[j] * r
                disc[j] *= gamma
                obs[j] = o
                if term:
                    alive[j] = False
                elif trunc:
                    if bootstrap_tail:
                        total[j] += disc[j] * float(ck.values(o)[0])
                    alive[j] = False

        if bootstrap_tail and alive.any():
            live = np.flatnonzero(alive)
            total[live] += disc[live] * ck.values(obs[live]).astype(np.float64)

        returns[start : start + B] = total
        for e in envs:
            e.close()

    g = returns.reshape(N, K, n_rollouts)
    q_mc = g.mean(axis=2).astype(np.float32)
    if not return_se:
        return q_mc
    se = (g.std(axis=2, ddof=1) / np.sqrt(n_rollouts)).astype(np.float32) if n_rollouts > 1 \
        else np.zeros_like(q_mc)
    return q_mc, se


# --------------------------------------------------------------------------- #
# Gate 2 report
# --------------------------------------------------------------------------- #

def gate2_report(land: dict, spread_threshold: float = 0.01, frac_threshold: float = 0.20) -> dict:
    """Numbers behind 'A_CF is nontrivial, numerically stable, policy-centered'."""
    a_cf, pi, q = land["a_cf"], land["pi"], land["q_cf"]

    centering = np.abs((pi * a_cf).sum(axis=1))
    spread = q.max(axis=1) - q.min(axis=1)
    best = a_cf.argmax(axis=1)
    counts = np.bincount(best, minlength=a_cf.shape[1])   # length K, zeros included

    return {
        "n_states": int(len(a_cf)),
        "centering_max_abs": float(centering.max()),
        "centering_mean_abs": float(centering.mean()),
        "finite": bool(np.all(np.isfinite(a_cf)) and np.all(np.isfinite(q))),
        "mean_abs_a_cf": float(np.abs(a_cf).mean()),
        "median_spread": float(np.median(spread)),
        "mean_spread": float(spread.mean()),
        "frac_states_spread_above_thr": float((spread > spread_threshold).mean()),
        "best_action_distribution": (counts / counts.sum()).round(3).tolist(),
        "n_terminal_transitions": int(land["terminated"].sum()),
        "critic_vs_vpi_mean_abs": float(np.abs(land["v_critic"] - land["v_pi"]).mean()),
        "checks": {
            "policy_centered (max |sum pi*A| < 1e-4)": bool(centering.max() < 1e-4),
            "numerically finite": bool(np.all(np.isfinite(a_cf))),
            f"nontrivial (>{frac_threshold:.0%} of states have spread > {spread_threshold})":
                bool((spread > spread_threshold).mean() > frac_threshold),
        },
    }
