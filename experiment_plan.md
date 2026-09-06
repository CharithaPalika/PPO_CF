# Trajectory-Derived Counterfactual Action Landscapes for PPO
## MacBook Feasibility Experiment Plan — living document

**Source of truth for the design:** `counterfactual_ppo_macbook_experiment_plan.pdf`.
This file does not change that plan. It restates it, tracks status, and records
what was actually measured. Update the Status table and the Run Log as each
notebook lands.

**Goal:** verify the complete pipeline on MountainCar before scaling. This is a
feasibility/mechanism test, not the final paper experiment.

**Pipeline:** PPO trajectories → explicit counterfactual reference → COCOA
landscape → state→landscape model B(s) → PPO regularization → stability/
performance checks.

---

## Status

| # | Notebook | Purpose | Status | Date |
|---|----------|---------|--------|------|
| 01 | `01_train_ppo_baseline` | Working PPO + trajectory/checkpoint generation | **Gate 1 passed on MountainCar; ported to DoorKey-8x8, smoke test passed** | 2026-09-06 |
| 02 | `02_counterfactual_oracle` | Explicit all-action reference landscape | **Written — Gate 2 passes at the 75% checkpoint** | 2026-09-05 |
| 03 | `03_cocoa_landscape` | Trajectory-only counterfactual landscape | Not started | — |
| 04 | `04_validate_landscape` | COCOA vs explicit reference | Not started | — |
| 05 | `05_state_to_landscape` | Learn B(s) ≈ A_CF(s,·) | Not started | — |
| 06 | `06_cf_ppo_smoke_test` | Landscape-based PPO regularization | Not started | — |

## Environment history

The plan's design is environment-agnostic; the target environment has changed
twice for reasons that were measured, not assumed. All three configs remain in
`config.ENV_PRESETS` and the codebase runs any of them.

| env | why it was tried | why it was left | random-policy success |
|---|---|---|---|
| MountainCar-v0 | the plan's original choice | the critic is a constant (V = −99.99, sd 0.00) until the first goal reach, so **A_CF is exactly zero during the phase where the task is hard**. The oracle cannot help with exploration, which is the entire difficulty. | **0 / 2000** |
| Taxi-v4 | dense −10 penalties differentiate actions from step 1, so the critic is informative immediately | PPO falls into Taxi's trap: an illegal pickup costs −10 while idling costs −1, so π(PICKUP) → 0.0016 and π(DROPOFF) peaks at 0.007 across all 500 states. Return pinned at −200.0. Entropy regularisation cannot fix this — it is maximised just as well by spreading mass over the four movement actions. Fixed by a per-action probability floor (`PPOConfig.prob_floor_*`), but the env was superseded first. | 94 / 2000 (4.7%) |
| **MiniGrid-DoorKey-8x8-v0** | **current target.** Sparse but reachable reward, 7 actions, deterministic, exact state restore. | — | **42 / 2000 (2.1%)** |

**Why DoorKey-8x8 is used fully observable.** MiniGrid's default 7×7 egocentric
view makes it a POMDP, and `A_CF = r + γV(s′) − V_π(s)` is only well-defined in
an MDP: the critic would learn V(o) rather than V(s), and two distinct simulator
states can produce byte-identical observations. `FullyObsWrapper` (8×8×3 grid
encoding) keeps it Markov. This diverges from published partial-observation
baselines and is a deliberate trade.

## Go / No-Go gates

| Gate | Must observe before continuing | Status |
|------|-------------------------------|--------|
| 1. PPO | Baseline learns and trajectory files are valid | **PASS** (3/3 seeds) |
| 2. Explicit CF | A_CF is nontrivial, numerically stable, policy-centered | **PASS at 75%** (fails at 30%) |
| 3. COCOA | Trajectory-only estimator produces an all-action signal | pending |
| 4. Landscape agreement | COCOA has meaningful agreement with explicit CF | pending |
| 5. Amortization | B(s) predicts held-out landscapes and agrees with explicit CF | pending |
| 6. PPO intervention | CF-PPO changes update direction/stability coherently; shuffled control does not replicate it | pending |

**Feasibility criterion.** The first pass succeeds if the chain
trajectory → COCOA → action landscape → B(s) → PPO intervention executes and the
intermediate signals are coherent. A performance improvement is desirable but not
required for this first debugging pass.

**Do not implement yet.** LunarLander/Pendulum, large seed counts, hyperparameter
sweeps, online B updates, adaptive λ, directional rotation, complex triggers,
paper-quality statistics.

---

## Repository layout

```
PPO_CF/
├── experiment_plan.md          <- this file
├── requirements.txt
├── config/config.py            <- ALL hyperparameters and paths. Single source of truth.
├── envs/
│   ├── env_pool.py             <- explicit sync env pool; sim-state get/set for NB02
│   └── scaling.py              <- state -> network-input map (fixed affine by default)
├── agents/
│   ├── networks.py             <- ActorCritic (separate actor/critic trunks)
│   ├── buffer.py               <- rollout buffer + GAE + NB06 advantage hook
│   └── ppo.py                  <- PPOTrainer
├── dataio/
│   ├── trajectory.py           <- trajectory schema, recorder, loader, validator
│   └── checkpoint.py           <- Checkpoint bundle: weights + scaler together
├── utils/                      <- seeding, CSV logging, plotting
├── scripts/train_baseline.py   <- NB01 engine (also a CLI)
├── oracle/
│   ├── sampling.py             <- policy-matched evaluation-state selection
│   └── counterfactual.py       <- the oracle, restore validation, MC diagnostic
├── dataio/landscape.py         <- landscape schema + validator
├── scripts/build_oracle.py     <- NB02 engine (also a CLI)
├── notebooks/
│   ├── 01_train_ppo_baseline.ipynb
│   └── 02_counterfactual_oracle.ipynb
└── runs/<run_name>/seed_<n>/   <- outputs (git-ignorable)
    ├── trajectories.npz        <- ~25 MB/seed
    ├── checkpoints/ckpt_{010,030,050,075,100}.pt
    ├── scalars.csv, episodes.csv, config.json
```

Everything is a `.py` module; notebooks are thin drivers. Nothing outside
`config/config.py` may hard-code a hyperparameter.

---

## Notebook 01 — `train_ppo_baseline`

Train standard PPO and create the reusable trajectory/checkpoint dataset.

- Inputs: MountainCar-v0, fixed PPO config, 3 seeds.
- Save observations, actions, rewards, next states, done flags, log-probs, policy
  probabilities, values, episode/timestep IDs.
- Save checkpoints at ~10%, 30%, 50%, 75%, 100%.
- Outputs: return curve, entropy, KL, losses, trajectories, checkpoints.
- **CHECK:** PPO learns at least partially; data and checkpoints reload correctly.

### Configuration as run

| | |
|---|---|
| env | MountainCar-v0, 16 parallel envs, `max_episode_steps=500` |
| obs scaling | fixed affine from `observation_space` bounds onto [−1, 1] |
| reward scaling | none (V must stay in true reward units for NB02) |
| total steps | 400,000 per seed |
| rollout | 16 envs × 32 steps = 512 per update, 781 updates |
| epochs / minibatches | 4 / 4 (minibatch 128) |
| γ / λ | 0.99 / 0.98 |
| clip | 0.2, value clipping on |
| lr | 7e-4, linearly annealed |
| entropy | adaptive floor, target 0.60 → 0.02, event-annealed |
| grad clip | 0.5, **actor and critic clipped separately** |
| seeds | 0, 1, 2 |

### Four implementation decisions that were forced by measurement

These are not stylistic. Each one was the difference between the baseline
learning and not learning, and each is documented inline at its definition.

1. **Separate actor/critic gradient clipping.** Every MountainCar reward is −1,
   so returns sit near −60 to −100 and the value loss is O(10³) from the first
   update. Under a single global grad-norm clip the critic dominates the norm,
   the whole vector is rescaled by ~1/40, and the policy stops moving entirely
   (entropy pinned at log 3 = 1.0986, approx_kl ~1e-6). Clipping the two heads
   independently keeps V in its true units — which NB02's oracle needs — while
   letting the policy update.

2. **`max_episode_steps = 500` during training.** The registered limit is 200.
   Measured on this codebase: at limit 200 the first goal reach happens at
   ~506k steps and only 38 of 4,992 episodes ever succeed within 1M steps. At
   limit 500 the first goal reach happens at ~130–220k. At limit 1000 the policy
   never succeeds (entropy collapses before it finds anything). The learned
   policy is then **evaluated under the standard 200-step limit**, where seeds 0
   and 1 score 50/50, so the benchmark is not being softened.

3. **Adaptive entropy floor rather than a fixed `ent_coef`.** MountainCar is
   unusually sharp here, and the failure is structural rather than slow:

   | exploration | successes / 2000 episodes | best position (goal = +0.50) |
   |---|---|---|
   | uniform random | **0** | −0.17 |
   | sticky, repeat p=0.8 | 42 | +0.53 |
   | sticky, repeat p=0.95 | 204 | +0.55 |
   | energy pumping (known solution) | 2000 | +0.53 |

   Reaching the goal requires *temporally correlated* actions. A high `ent_coef`
   (≥0.01) pins the policy near uniform, where the success probability is not
   small but zero. A low one (0.0) collapses it to a deterministic bad policy
   before it ever sees a reward, which is irrecoverable — no reward means no
   signal to push it back out. With a fixed `ent_coef=0.003`, 1 of 3 seeds
   solved and 2 collapsed. The adaptive controller holds mean policy entropy
   near a floor and gives 3/3.

4. **The entropy anneal is event-based, not time-based.** The floor is held at
   0.60 until the agent reaches the goal for the first time, and only then
   decays. A schedule that sharpens the policy on wall-clock has no idea whether
   there is yet anything worth sharpening towards; that is exactly what lost
   seeds 1 and 2.

### Results

| seed | first goal reach | final return (500-limit) | final success rate | explained variance |
|------|------------------|--------------------------|--------------------|--------------------|
| 0 | 164,464 | −134.6 | 1.00 | 0.948 |
| 1 | 157,712 | −136.1 | 1.00 | 0.968 |
| 2 | 220,496 | −229.7 | 1.00 | 0.955 |

Greedy evaluation under the **standard 200-step** MountainCar-v0, 50 episodes:

| seed | 10% | 30% | 50% | 75% | 100% |
|------|-----|-----|-----|-----|------|
| 0 | 0/50 | 0/50 | 1/50 | 49/50 | **50/50** (−132.2) |
| 1 | 0/50 | 0/50 | 12/50 | 50/50 | **50/50** (−133.5) |
| 2 | 0/50 | 0/50 | 0/50 | 0/50 | 3/50 (−199.1) |

Seed 2 solves the 500-step task reliably but is too slow for the 200-step
benchmark. That is acceptable for a feasibility pass — Gate 1 asks that the
baseline learn, not that it be optimal.

Trajectory validation: **0 problems on all 3 seeds**, 399,872 rows each, ~25 MB.
Validated properties: probs sum to 1; `logprob == log probs[action]`;
`next_obs[t] == obs[t+1]` inside episodes; no row both terminated and truncated;
at most one terminal row per episode; all values finite.

Wall time: ~85 s/seed, ~4.5 min for all three (CPU).

### Open issue this notebook creates for Notebook 02

**The plan specifies 500 states from the ~30% checkpoint. At 30% the policy has
not learned anything** — 0/50 greedy success on every seed, and the first goal
reach happens at 39–55% of training. Q_CF(s,a) = r + γV(s′) evaluated there
rests on a critic that has never seen a reward, so V is close to constant and
A_CF will be close to zero for all three actions. Gate 2 ("A_CF is nontrivial")
would then fail for a reason that has nothing to do with the method.

NB01 measures this directly. It runs a 300-state mini-oracle at every checkpoint
and reports `mean |A_CF|`, the spread between the three actions. Since every
non-terminal MountainCar reward is −1, that spread is exactly the headroom Gate 2
has to work with:

| seed | 10% | 30% | 50% | 75% | 100% |
|------|-----|-----|-----|-----|------|
| 0 | 0.0039 | **0.0021** | 0.2396 | 0.3646 | 0.4416 |
| 1 | 0.0221 | **0.0419** | 0.3694 | 0.4416 | 0.4522 |
| 2 | 0.0053 | **0.0102** | 0.0108 | 0.1779 | 0.2547 |

At 30% the landscape is 20–200× flatter than at 100%. The centering identity
Σ_a π·A_CF = 0 holds to ~1e-8 everywhere, so the machinery is correct — there is
simply nothing to measure yet at that checkpoint.

Options, to be decided before NB02 is written:

- **(a)** Use the 75% checkpoint instead of 30%. Cheapest fix; 75% is the first
  checkpoint with a competent policy on seeds 0 and 1.
- **(b)** Keep 30% but extend training so that 30% lands after the first goal
  reach (e.g. 1M steps, making 30% ≈ 300k).
- **(c)** Keep 30% and accept a near-degenerate A_CF as the documented Gate 2
  result, then proceed at a later checkpoint.

Related: A_CF is **policy-dependent** — it is centered by π(a|s) at whichever
checkpoint the oracle uses. NB03 runs COCOA on trajectories from *all* of
training, i.e. a mixture of policies. NB04 compares the two on identical states,
so the behaviour policy has to be matched or the comparison is not
apples-to-apples. `Trajectories.window_around_step(center, half_width)` exists
for this: restrict COCOA's data to a window around the oracle's checkpoint.

---

## Notebook 02 — `counterfactual_oracle`

Create a small simulator-based reference landscape.

- Use 500 states from the ~30% checkpoint. *(See open issue above — the notebook
  defaults to 75% and the choice is a constant at the top of the notebook.)*
- For every state, restore simulator state and execute actions 0, 1, 2.
- Compute Q_CF(s,a) = r + γV(s′); terminal V = 0.
- Compute V_π(s) = Σ_a π(a|s) Q_CF(s,a), then A_CF = Q_CF − V_π.
- **CHECK:** Σ_a π·A ≈ 0 per state; action values differ on a meaningful subset;
  restore/step logic is correct.

### Status: Gate 2 PASSES at the 75% checkpoint, 3/3 seeds

| check | result |
|---|---|
| restore/step reproduces recorded transitions | exact — state err 0.00e+00, reward err 0.00e+00, terminated agreement 1.000 over 300 replayed transitions per seed |
| simulator restore deterministic | PASS |
| policy-centered, max \|Σ_a π·A_CF\| | 1.3e-05 |
| landscape self-consistency (a_cf, v_pi, q_cf) | PASS |
| nontrivial (frac. states with Q spread > 0.01) | 0.98 / 0.96 / 0.93 |
| mean \|A_CF\| | 0.58 / 0.58 / 0.32 |

The restore check is the important one: rather than checking the oracle against
itself, it replays transitions recorded during training — restore the stored
`sim_state`, take the stored action, compare against the `next_sim_state` and
`reward` that actually occurred. It matches to the last bit.

Checkpoint sweep (mean |A_CF|, 150 states, all five checkpoints):

| seed | 10% | 30% | 50% | 75% | 100% |
|------|-----|-----|-----|-----|------|
| 0 | 0.0004 | **0.0000** | 0.2773 | 0.5638 | 0.5918 |
| 1 | 0.0007 | **0.0004** | 0.4758 | 0.6514 | 0.6235 |
| 2 | 0.0004 | **0.0001** | 0.0003 | 0.3127 | 0.5084 |

At 30% the landscape is zero to four decimal places on every seed. The critic
there is a literal constant — V(s) = −99.99 with standard deviation 0.00, which
is exactly −1/(1−γ), the value of never terminating. Gate 2 cannot pass at the
checkpoint the plan names, and this is why.

### New open issue this notebook creates for Notebook 04

**The one-step oracle is only weakly aligned with the true A^π.**

If V were the true V^π, then A_CF = r + γV(s′) − V_π(s) would be *exactly* A^π —
the one-step form is the definition of the advantage, not an approximation of it.
So every deviation is critic error. NB02 measures how much there is, using
rollouts with common random numbers (paired across actions, so the difference is
low-variance):

| seed | mean \|A_CF\| | mean \|A_MC\| | magnitude ratio | Pearson | best-action agreement | critic bias V−V^π |
|---|---|---|---|---|---|---|
| 0 | 0.543 | 1.053 | 0.52 | 0.41 | 0.48 | −12.1 |
| 1 | 0.608 | 1.301 | 0.47 | 0.42 | 0.58 | −14.1 |
| 2 | 0.268 | 0.461 | 0.58 | 0.57 | 0.55 | −15.3 |

Chance best-action agreement is 1/3. So A_CF is positively but weakly aligned
with the true advantage, and about half its magnitude.

This is not Monte-Carlo noise, and that was checked rather than assumed: two
independent CRN seeds at 32 rollouts give landscapes correlated at **Pearson
0.995**, and the Pearson against A_CF is flat at ~0.40 from 8 rollouts to 128.
Without CRN the same diagnostic reported agreement 0.35 and a magnitude ratio of
0.13 — pure noise, and it would have been believed.

Why it matters for Gate 4: NB04 scores COCOA by "meaningful positive agreement
with the explicit CF". If the explicit CF is itself only ~0.4-correlated with
A^π, a *correct* COCOA can score badly and a COCOA that mimics critic error can
score well. Options:

- **(a)** Report NB04 against **both** references, one-step A_CF and rollout
  A_MC. Cheap, informative, does not change the plan's oracle. **Recommended.**
- **(b)** Fit the critic further on the frozen checkpoint (value-only epochs
  against rollout returns) before building the oracle. Removes the confound.
- **(c)** Keep the one-step oracle and state explicitly that Gate 4 measures
  agreement-with-the-critic, not agreement-with-A^π.

### Interfaces for Notebook 03

```python
from dataio import load_landscape
land = load_landscape(".../landscapes/oracle_ckpt075.npz")
allowed = land.evaluation_states_only()   # index, sim_state, raw_obs, global_step
# FORBIDDEN in NB03: land.a_cf, land.q_cf, land.pi, land.v_pi, land.v_critic
```

`evaluation_states_only()` exists so the "trajectories only" constraint is
enforced by code rather than by memory. The evaluation states were drawn from a
±5% window around the checkpoint step (`meta["state_half_width_steps"]`); use the
same window for COCOA's training data via
`Trajectories.window_around_step(center, half_width)` so NB04 compares two
objects defined against the same policy.

## Notebook 03 — `cocoa_landscape`

Infer the counterfactual action landscape using only ordinary PPO trajectories.

- Input only the trajectories from Notebook 01.
- Run/adapt COCOA to produce an all-action counterfactual/contribution signal.
- Do not provide explicit counterfactual labels.
- Save A_COCOA(s,·) for the same evaluation states.
- **CHECK:** no extra counterfactual simulator calls are required for
  estimation; a K-dimensional signal is produced for each state.

## Notebook 04 — `validate_landscape`

Test whether COCOA recovers the explicit action landscape.

- Compare A_COCOA with A_CF on identical states.
- Metrics: vector MSE, Pearson/Spearman correlation, best-action agreement,
  magnitude ratio.
- Per-action scatter plots and several individual-state comparisons.
- Also compare policy-gradient directions if feasible.
- **CHECK:** meaningful positive agreement. If COCOA is near random, stop and
  debug before proceeding.

Note: with K = 3 and Σ_a π·A_CF = 0, the landscape has 2 degrees of freedom and
random best-action agreement is 1/3, not 1/2. Compute the null empirically by
shuffling state↔landscape pairs rather than assuming it.

## Notebook 05 — `state_to_landscape`

Test the amortized state → action-landscape model.

- Dataset: (s, A_COCOA(s,·)); 70% train / 30% held-out.
- Models: Ridge first; tiny MLP second.
- Metrics: R² per action, vector MSE, best-action agreement.
- Critical: evaluate B(s) against explicit A_CF on held-out states.
- Control: shuffle state-target pairs; predictive performance should collapse.
- **CHECK:** B(s) generalizes and retains agreement with explicit CF.

B is trained on A_COCOA targets but evaluated against A_CF, so the error is
COCOA error + regression error compounded. Report the two legs separately.

## Notebook 06 — `cf_ppo_smoke_test`

Test whether the learned landscape can regularize PPO.

- Conditions: standard PPO; PPO + λ·B(s); PPO + shuffled B(s).
- Start with λ = 0.1 only; 3 seeds.
- Use A_new = A_GAE + λ·B(s), with consistent centering/scaling of B.
- Log return, environment steps, entropy, KL, gradient norm, and directional
  alignment.
- **CHECK:** B changes updates coherently; shuffled B does not reproduce the
  same effect; no catastrophic instability.

The hook is already in place: `PPOTrainer(..., advantage_transform=fn)` where
`fn(buffer, advantages) -> advantages` receives the full rollout. No other part
of the training loop needs to change.

Note on shapes: A_GAE is a scalar per (s, a_taken) while B(s) is a K-vector, so
`A_GAE + λ·B(s)` needs a stated reading — `B(s)[a_taken]`, or the full vector via
an all-action auxiliary term. Decide and write it down before running NB06,
otherwise the result is not interpretable either way.

## Minimum outputs to inspect

1. PPO return vs environment steps.
2. Explicit A_CF distributions and per-state advantage spread.
3. Explicit vs COCOA scatter plots for each action.
4. COCOA vs explicit best-action agreement.
5. Held-out R²/MSE for B(s), plus B(s) vs explicit CF.
6. PPO vs CF-PPO vs shuffled return curves.
7. Gradient alignment with the counterfactual reference.
8. KL per update and policy entropy.

---

## Run log

| date | run | what | outcome |
|------|-----|------|---------|
| 2026-09-05 | implementation self-test | PPO on CartPole-v1, 150k steps | return 20 → 462. Confirms the trainer is correct before blaming the environment. |
| 2026-09-05 | exploration measurement | 2000 episodes/policy, MountainCar | uniform random 0/2000, best position −0.17. Sticky p=0.95 204/2000. Success needs temporal correlation. |
| 2026-09-05 | ent_coef sweep | {0, 0.001, 0.003, 0.005, 0.01} × n_steps {16, 32}, 200k | 0 successes in every cell. Fixed ent_coef is not the answer. |
| 2026-09-05 | time-limit test | limit ∈ {200, 500, 1000} @ 400k | 500 works (first success 133k); 200 needs >500k; 1000 never succeeds. |
| 2026-09-05 | fixed ent 0.003, 3 seeds | 400k | 1/3 seeds solved; seeds 1–2 entropy-collapsed pre-reward. |
| 2026-09-05 | adaptive entropy, time-anneal | 400k, 3 seeds | 3/3 reach goal but final policies stay soft (0.71–1.00 success). |
| 2026-09-05 | **adaptive entropy, event-anneal** | **400k, 3 seeds — current baseline** | **3/3 at 100% success, EV 0.95+, trajectories validate clean.** |
| 2026-09-05 | NB02 oracle | 500 states, 75% checkpoint, 3 seeds | Gate 2 PASS. Restore replay exact; centering 1.3e-05; mean \|A_CF\| 0.58/0.58/0.32. |
| 2026-09-05 | NB02 checkpoint sweep | 150 states x 5 checkpoints x 3 seeds | mean \|A_CF\| at 30% is 0.0000/0.0004/0.0001. Critic there is constant at −99.99 (sd 0.00) = −1/(1−γ). |
| 2026-09-05 | NB02 MC diagnostic | 60 states, 32 CRN rollouts | A_CF vs A^π: Pearson 0.41/0.42/0.57, magnitude ratio 0.52/0.47/0.58, critic bias −12 to −15. Stable from R=8 to R=128; CRN self-consistency 0.995. |
| 2026-09-05 | bug found and fixed | NB02 section 8 | The checkpoint sweep was writing to the same filename as the main landscape and silently overwriting the 500-state file NB03 consumes. Sweep now writes `sweep_ckpt*.npz`. It also corrupted an earlier measurement of the MC gap. |
