# The counterfactual oracle on DoorKey-8x8 — design, mathematics, algorithms

Companion to `experiment_plan.md`. This document covers the oracle path **without
COCOA**: build the exact landscape, check whether it points anywhere useful, and
test whether PPO can use it. COCOA (NB03/04) is a later replacement for one box
in this pipeline, not a prerequisite.

---

## 0. Notation

| symbol | meaning |
|---|---|
| `s` | simulator state. On MiniGrid: grid encoding + agent pos/dir + carried object |
| `a ∈ {0..K-1}` | action. DoorKey-8x8 has **K = 7** |
| `π_θ(a\|s)` | policy (actor) |
| `V_φ(s)` | critic |
| `γ = 0.99` | discount |
| `r(s,a)`, `s'(s,a)` | reward and successor — **exact**, from the deterministic simulator |
| `d(s,a)` | 1 if `s'(s,a)` is terminal, else 0 |

---

## 1. The object we are building

### 1.1 Definition

```
Q_CF(s,a) = r(s,a) + γ · (1 − d(s,a)) · V_φ(s'(s,a))        (1)
V_π(s)    = Σ_a π_θ(a|s) · Q_CF(s,a)                         (2)
A_CF(s,a) = Q_CF(s,a) − V_π(s)                               (3)
```

`A_CF(s,·)` is a K-vector: the **action landscape** at `s`.

### 1.2 In plain English

- `Q_CF(s,a)` — "what I get if I take `a` right now, then go back to behaving
  normally." The *right now* part is exact (real simulator). The *then* part is
  the critic's estimate.
- `V_π(s)` — "what I get if I just behave normally from here." It is the average
  of the seven `Q` values, weighted by how often the policy would actually pick
  each action.
- `A_CF(s,a)` — the difference. Positive means better than the policy's habit,
  negative means worse.

### 1.3 Four properties that make it checkable

**(P1) Exact centering.**
```
Σ_a π(a|s) · A_CF(s,a) = Σ_a π·Q − V_π·Σ_a π = V_π − V_π = 0
```
This holds *by construction*, for every state, to numerical precision. It is the
cheapest possible correctness test: if it fails, the code is wrong.

**(P2) If the critic were perfect, the oracle would be exact.**
The one-step Bellman identity says `A^π(s,a) = r + γV^π(s') − V^π(s)`. So (1)–(3)
with `V_φ = V^π` gives `A_CF = A^π` **exactly** — the formula is the *definition*
of the advantage, not an approximation of it. Therefore **every error in `A_CF`
is critic error and nothing else.** That is a strong statement and it is what
makes Stage B (§4) meaningful.

**(P3) A constant offset in the critic does not matter.**
Replace `V_φ → V_φ + c`. Then every `Q_CF(s,a) → Q_CF + γc`, and `V_π → V_π + γc`,
so `A_CF` is unchanged. **Critic bias cancels; only the critic's local *shape*
matters.** (The exception is terminal transitions, where `V(s') = 0` is used
regardless of `c`, so states near termination do feel the offset.)

This is worth knowing: on MountainCar the critic was biased by −12 to −15 and the
landscape was *still* usable, because the bias cancelled. The 0.41 correlation
came from shape error, not bias.

**(P4) The landscape has K−1 degrees of freedom.**
(P1) is one linear constraint, so `A_CF(s,·)` lives in a 6-dimensional subspace
for K = 7. Random best-action agreement is **1/7 ≈ 0.143**, not 1/3 and not 1/2.

### 1.4 A fifth property, specific to MiniGrid

**(P5) No-op actions must tie exactly.**
Many MiniGrid actions leave the world unchanged: `forward` into a wall, `pickup`
with nothing in front, `drop` with nothing held, `done` always. For any such
action, `s'(s,a) = s`, hence

```
Q_CF(s,a) = 0 + γ·V_φ(s)      — identical for every no-op action in that state
```

So in any state with two or more no-ops, those entries of `Q_CF` must agree **to
the last bit**. This is a far sharper restore/step test than anything MountainCar
offered, and it costs nothing to check.

### 1.5 What the landscape actually measures on DoorKey

DoorKey pays nothing until the task is finished, so `r(s,a) = 0` for almost every
pair and (3) collapses to

```
A_CF(s,a) ≈ γ · [ V_φ(s'_a) − Σ_b π(b|s)·V_φ(s'_b) ]
```

**The landscape is entirely a comparison of where the seven actions land you, as
scored by the critic.** This is structurally the same as MountainCar, which is
why the MountainCar failure can recur here: if the critic has not learned, all
seven successors look alike, `A_CF ≈ 0`, and Gate 2 fails for reasons that have
nothing to do with the method. Hence the checkpoint sweep in §3.

---

## 2. Three questions, deliberately kept separate

Conflating these is the main way this kind of experiment goes wrong.

| | question | answered by | failure means |
|---|---|---|---|
| **Q1 Existence** | Is `A_CF` non-degenerate? | Stage A, Gate 2 | the critic is flat — pick a later checkpoint or train longer |
| **Q2 Fidelity** | Does `A_CF ≈ A^π`? | Stage B, vs rollouts | the critic's *shape* is wrong — the landscape misleads |
| **Q3 Utility** | Does using `A_CF` improve PPO? | Stage C | the signal is real but PPO cannot exploit it |

Q1 can pass while Q2 fails (MountainCar: Gate 2 passed at 75%, but Pearson was
only 0.41). Q2 can pass while Q3 fails. Each stage is cheap relative to the next,
so run them in order.

---

## 3. Stage A — build the landscape

### 3.1 Algorithm

```
INPUT: checkpoint (θ, φ) at step c, trajectory dataset D, N states, γ, window w
OUTPUT: A_CF ∈ R^{N×K}, plus Q_CF, V_π, π, and provenance

1  # policy-matched state selection
   pool ← { rows of D : |global_step − c| ≤ w }
   S    ← N states sampled uniformly from pool          # sim_state, not observation

2  # exhaustive one-step expansion  (N×K simulator calls)
   for each s in S:
       obs0[s] ← restore(s); read observation
       for a in 0..K−1:
           restore(s)                                    # the previous action moved the sim
           (r, s', term) ← step(a)
           R[s,a], NS[s,a], T[s,a] ← r, s', term

3  # one batched forward pass each, not N×K of them
   Vn  ← V_φ( obs(NS) )                                  # (N,K)
   Q   ← R + γ · Vn · (1 − T)
   P   ← π_θ( · | obs0 )                                 # (N,K)
   Vpi ← rowsum(P ⊙ Q)                                   # (N,)
   A   ← Q − Vpi[:,None]

4  run the Gate-2 checks of §3.3
5  save landscape + metadata (checkpoint step, γ, window, sampling strategy)
```

**Why step 1 uses a window.** `A_CF` is centered by `π` at *this* checkpoint, so
the evaluation states should be ones that policy actually visits. Sampling from
all 3M rows would mix in states visited by every other policy during training.
`Trajectories.window_around_step` exists for exactly this.

**Why step 2 restores before every action.** The previous action already moved
the simulator. This is the single easiest place to write a silent bug.

### 3.2 Cost

Measured on this codebase: **371 µs** per restore+step (deepcopy snapshot), 630 µs
(encode/decode). So `N=500, K=7` → 3,500 calls → **~1.3–2.2 s per checkpoint per
seed**. The five-checkpoint sweep is under 15 seconds. Stage A is free.

### 3.3 What we compute, and the pre-registered thresholds

Pre-registered means: decided *before* looking at the numbers, so the verdict
cannot be rationalised afterwards.

| metric | definition | threshold | what a failure means |
|---|---|---|---|
| centering error | `max_s \|Σ_a π·A_CF\|` | < 1e-4 | code bug (P1 is an identity) |
| finiteness | all `Q`, `A` finite | required | NaN in critic or restore |
| **no-op tie error** | `max` over states of `max\|Q(a₁)−Q(a₂)\|` over no-op pairs | < 1e-5 | restore/step is broken (P5) |
| spread | `spread(s) = max_a Q − min_a Q` | median > 0.005 | landscape is degenerate (critic flat) |
| nontrivial fraction | `frac{ s : spread(s) > τ }`, τ = 0.005 | > 0.20 | as above |
| argmax concentration | `max_a frac{ argmax A = a }` | < 0.95 | landscape collapsed onto one action |
| dead-action check | `frac{ argmax A ∈ {drop, done} }` | ≈ 0 expected | if high, something is wrong — these are no-ops |

Note the τ = 0.005 threshold is tied to DoorKey's reward scale: return lies in
[0,1], so 0.005 is 0.5% of the maximum achievable return. On MountainCar, where
returns were −60 to −100, the equivalent threshold was 0.01. **This is a config
knob, not a universal constant** — it has to be re-derived per environment.

### 3.4 The checkpoint sweep

Run §3.1 at all five checkpoints and tabulate `mean |A_CF|`. On MountainCar this
was decisive: at the plan's nominal 30% checkpoint the landscape was 0.0000 to
four decimal places, because the critic was a literal constant (V = −99.99,
sd 0.00 = −1/(1−γ), the value of never terminating). **Do not pick a checkpoint
before seeing this table.**

---

## 4. Stage B — is the landscape pointing the right way?

This is the cheap kill test. It runs **before** any retraining.

### 4.1 The intuition

The policy gradient theorem has two equivalent forms:

```
∇J = E_{s~d^π} [ Σ_a ∇π_θ(a|s) · Q^π(s,a) ]          all-action form      (4)
   = E_{s,a~π} [ ∇log π_θ(a|s) · A^π(s,a) ]          sampled form         (5)
```

PPO estimates (5): one sampled action per state, weighted by a noisy GAE
advantage. A *landscape* gives you (4) directly — the same gradient, with the
action-sampling variance removed entirely, because you sum over all seven actions
instead of sampling one.

**That is the whole mathematical case for this project.** Not a heuristic bonus:
the identical gradient, computed with less noise. What stands between the ideal
and reality is that we have `A_CF` (critic-based) rather than `A^π`.

So the question is precisely: **does `A_CF` plugged into (4) point closer to the
true gradient than PPO's sampled estimate does?**

### 4.2 Algorithm

```
INPUT: checkpoint (θ,φ), one rollout batch B, evaluation states S
1  g_GAE ← (1/|B|) Σ_{(s,a)∈B} ∇_θ log π_θ(a|s) · Â_GAE(s,a)
2  g_CF  ← (1/|S|) Σ_{s∈S} Σ_a ∇_θ π_θ(a|s) · A_CF(s,a)
3  A_MC  ← rollout estimate of A^π on S, with common random numbers
   g_MC  ← (1/|S|) Σ_{s∈S} Σ_a ∇_θ π_θ(a|s) · A_MC(s,a)
4  report cos(g_CF, g_MC)  vs  cos(g_GAE, g_MC)
```

All three are flattened parameter vectors, so cosine similarity is well-defined.
Also report it per block (CNN encoder vs policy head), because the two can
disagree.

### 4.3 Why common random numbers are non-negotiable in step 3

The quantity of interest is a **difference**, `Q(s,a) − Q(s,a')`, which on these
tasks is order 1 while the episode return itself varies by far more across
rollouts. With independent randomness per action, the noise in the difference
exceeds the signal and the diagnostic reports disagreement that is entirely its
own.

So rollout `m` from state `s` uses the **same random stream for every action**:
`rng = default_rng([seed, i, m])`, independent of `a`. The trajectories still
diverge, but from a shared source of randomness, which is what makes the paired
difference low-variance.

This was measured on MountainCar, and the difference is not subtle:

| | magnitude ratio | Pearson | best-action agreement |
|---|---|---|---|
| without CRN | 0.13 | 0.16 | 0.35 |
| **with CRN** | **0.52** | **0.41** | **0.48** |

Two independent CRN seeds at 32 rollouts produced landscapes correlated at
**Pearson 0.995**, and the Pearson against `A_CF` was flat from 8 to 128 rollouts
— so the remaining disagreement is real, not sampling noise. Without CRN I would
have reported a 10× magnitude compression that did not exist.

### 4.4 Metrics and decision rule

| metric | meaning | good |
|---|---|---|
| `cos(g_CF, g_MC)` | does the oracle gradient point the right way | high |
| `cos(g_GAE, g_MC)` | does plain PPO point the right way | the bar to beat |
| `‖g_CF‖ / ‖g_MC‖` | is the step the right size | ≈ 1 |
| Pearson(`A_CF`, `A_MC`) | landscape fidelity, state-wise | high |
| ratio of mean \|·\| | magnitude fidelity | ≈ 1 |
| best-action agreement | ordinal fidelity | ≫ 1/7 = 0.143 |
| MC standard error | is the reference itself powered | ≪ signal |

**Stop rule (pre-registered): if `cos(g_CF, g_MC) ≤ cos(g_GAE, g_MC)` at two or
more of three checkpoints, do not run Stage C.** The oracle carries no
directional advantage over what PPO already computes, and no amount of
engineering downstream will change that.

Use the **ratio of mean magnitudes**, not a least-squares slope through the
origin — the slope is dominated by a handful of large-`|A_MC|` states and read
0.12 against a true scale ratio near 0.5 on MountainCar, a metric that looks
rigorous and misleads by a factor of four.

### 4.5 Cost

`A_MC` dominates: `n_states × K × n_rollouts × horizon` environment steps.
At 100 states, 8 CRN rollouts, horizon 128: 100 × 7 × 8 × 128 ≈ **717k steps ≈
4 minutes** per checkpoint. Three checkpoints ≈ 12 minutes. The gradients
themselves are three backward passes — negligible.

---

## 5. Stage C — can PPO use it?

### 5.1 The injection form, and why the obvious one is a trap

The natural thing to write is

```
Â_new(s,a_taken) = Â_GAE(s,a_taken) + λ · A_CF(s,a_taken)        (6)
```

This is close to a no-op, for two compounding reasons.

1. `A_CF` at the taken action is almost TD(0): `r + γV(s') − V_π(s)` versus GAE's
   `r + γV(s') − V(s)`, differing only in the centering term. So (6) blends
   GAE(0.95) with GAE(0) — a bias/variance knob on a quantity PPO already
   computes, not new information.
2. PPO standardises advantages per minibatch, which divides much of the intended
   effect straight back out.

Worse, (6) uses **one column** of a K-column object. The five or six columns
describing actions that were *not* taken — the entire reason to have a landscape
— are discarded.

### 5.2 The form that carries new information

From (4), the all-action policy gradient is obtained from the auxiliary loss

```
L_aux(θ) = − (1/|M|) Σ_{s∈M} Σ_a π_θ(a|s) · Ã_CF(s,a).detach()    (7)
```

Differentiating through `π_θ` gives exactly `Σ_a ∇π(a|s)·A_CF(s,a)`, i.e. (4).
Total objective:

```
L = L_PPO-clip + c_v·L_value − c_e·H[π] + λ·L_aux
```

**Normalisation matters.** `Ã_CF` is `A_CF` standardised over the subset,
`Ã_CF = A_CF / (std(A_CF) + ε)`, so that λ means the same thing across states,
updates and runs. Without this, λ = 0.1 is a different intervention at every
update and the results are uninterpretable.

### 5.3 Algorithm

```
each PPO update:
  1  collect rollout                                  (T×N states, as usual)
  2  compute GAE advantages                           (as usual)
  3  M ← random subset of size M from the batch       # M = 256
  4  for s in M: A_CF(s,·) ← §3.1 step 2–3 with the LIVE (θ,φ)
  5  Ã_CF ← A_CF / (std + ε)
  6  minimise  L_PPO + c_v L_v − c_e H + λ·L_aux(M, Ã_CF)
```

Step 4 uses the *current* critic, recomputed every update — that is the honest,
deployable version. Step 3 subsamples because the full batch is unaffordable
(§5.5); a uniform random subset of the batch is an unbiased sample of the same
state distribution.

### 5.4 Conditions and controls

| condition | purpose |
|---|---|
| baseline PPO | reference |
| **λ = 0 through the oracle code path** | **must reproduce baseline exactly** — proves the plumbing is inert by itself |
| λ ∈ {0.1, 0.3, 1.0}, all-action | the intervention |
| shuffled `A_CF` (permuted across states) | the plan's required control |
| norm-matched random landscape | rules out "any signal of this magnitude perturbs training" |
| scalar form (6), λ = 0.3 | shows *why* the all-action form is the one that matters |

The λ = 0 identity check is the one people skip and the one that catches the most
bugs.

### 5.5 Cost, measured

| oracle coverage | per update | extra over 3M frames |
|---|---|---|
| full batch, 2048 × 7, encode/decode | 9.04 s | 221 min |
| full batch, 2048 × 7, deepcopy | 5.32 s | 130 min |
| **256-state subset × 7, deepcopy** | **0.66 s** | **16 min** |

Full batch is unaffordable; the 256-state subset is not. Baseline 3M ≈ 20 min, so
an oracle condition is ≈ 36 min.

**Recommended design: warm start.** Begin every condition from the *same*
checkpoint — one where the critic is already informative — and train 1M further
frames. This isolates the regime where the oracle has content (it is provably
near-zero before the critic learns), and cuts the cost to ≈ 12 min per run.
Five conditions × 3 seeds = 15 runs ≈ **3 hours**, paired on seeds.

### 5.6 Metrics

| metric | why |
|---|---|
| success rate and return vs frames | the primary outcome |
| frames to first success; frames to 50% success | sample efficiency, which is where an advantage should appear |
| entropy, approx-KL, clipfrac, grad norms | stability — the plan asks for "no catastrophic instability" |
| `cos(actual update, g_MC)` over training | did the intervention change the *direction*, or only the step size |
| `π(pickup)`, `π(toggle)` per-action | the Taxi lesson: a mandatory action collapsing to zero is invisible to entropy |
| key-pickup rate | sub-goal progress, moves long before success on a sparse task |

**Statistical honesty.** 3 seeds detects "this breaks training". It does not
establish "this improves training" against DoorKey's seed variance. Treat Stage C
as a mechanism test — direction changed coherently, control did not replicate it,
nothing exploded — and leave performance claims to a larger run.

---

## 6. Summary of the ladder

| stage | question | cost | stop if |
|---|---|---|---|
| A | does a non-degenerate landscape exist? | ~15 s | spread is ~0 at every checkpoint |
| B | does it point better than GAE? | ~12 min | `cos(g_CF,g_MC) ≤ cos(g_GAE,g_MC)` |
| C | can PPO use it? | ~3 h | shuffled control reproduces the effect |

Each rung is cheap relative to the next and each isolates one failure mode. COCOA
later replaces the *source* of the landscape in Stage C — everything else in this
document stays as written.
