# Tuning guide

What to turn, in what order, and which diagnostic justifies turning it.

Two rules that make the difference between tuning and thrashing:

1. **Change one thing at a time**, in `config/envs/<env>.yaml` or via `--set`.
   Never in the notebook.
2. **Read the diagnostic before choosing the knob.** Most PPO tuning advice is
   written for dense-reward tasks and is actively wrong on a sparse one — for
   example "raise `ent_coef` to explore more" does nothing if the rollouts
   carry no advantage signal at all.

---

## Step 0 — is there a signal? (`adv_std_raw`)

This is a new column in `scalars.csv` and the first thing to look at. It is the
spread of the advantages **before** whitening.

| what you see | meaning |
|---|---|
| `adv_std_raw` ≥ ~1e-2 | there is real signal; hyperparameters are the right lever |
| `adv_std_raw` at the 1e-6 floor | the rollout contained nothing to learn from |

If it is at the floor, **no hyperparameter will help**. Entropy, `approx_kl` and
`clipfrac` all still move in that regime — that movement is numerical noise, not
exploration, and it is what made the failed 8x8 run look like it was doing
something. Fix the reward reachability instead: go down a curriculum rung, fix
the layout pool, or turn on shaping (steps 1–3).

---

## Step 1 — the environment itself (biggest lever by far)

| knob | file | effect |
|---|---|---|
| `ENV_CONFIG` / `--env` | — | `doorkey5x5` → `doorkey6x6` → `doorkey8x8` |
| `env.layout_seeds` | yaml | `null` = new layout every episode. `[0]` = one fixed layout. `[0,1,2,3]` = a pool of 4 |
| `env.layout_seed_mode` | yaml | `cycle` (deterministic coverage) or `random` |

`layout_seeds` is the highest-value knob in this file. With `null`, PPO is being
asked to *generalise across layouts* before it has ever seen a reward. Nothing in
NB02–06 needs that — the oracle restores specific states under a specific policy
— so fixing the pool removes a difficulty the project does not care about.

**Suggested ladder if 8x8 stalls:** `[0]` → `[0..7]` → `[0..31]` → `null`.

---

## Step 2 — batch geometry

| knob | default | when to change |
|---|---|---|
| `ppo.n_steps` | 128 | raise to 256/512 when episodes are long relative to the rollout. At 640-step episodes and a 2048-step rollout only ~3 episodes land per update, so one rewarded episode gets diluted across 2048 samples |
| `env.n_envs` | 16 | raise for more diverse rollouts, at proportional wall-clock cost |
| `ppo.n_minibatches` | 8 | minibatch = `n_envs·n_steps / n_minibatches`; keep it ≥ 256 |
| `ppo.n_epochs` | 4 | 8 squeezes more from rare rewarded transitions but raises the off-policy error; watch `approx_kl` |

`n_steps` and `n_envs` both raise the number of episodes per update, which is the
quantity that actually matters when successes are rare.

---

## Step 3 — reward reachability

Both off by default. **Read `envs/shaping.py` before turning either on** — they
have different consequences for NB02.

| knob | suggested start | oracle-safe? |
|---|---|---|
| `reward.potential_shaping` + `potential_key` / `potential_door` | `true`, 0.2 / 0.4 | **yes** — `V_true = V_shaped + Φ` exactly, so NB02 can undo it |
| `reward.count_bonus_coef` + `count_bonus_anneal_frac` | 0.005, 0.5 | **no** — changes the MDP; the trainer refuses to save a checkpoint while the bonus is nonzero |

Potential shaping over `(has_key, door_open)` is well-motivated here rather than
arbitrary: both are hard prerequisites for the goal, and Ng et al. (1999)
guarantees the optimal policy is unchanged.

---

## Step 4 — exploration and action collapse

| knob | default | when |
|---|---|---|
| `ppo.ent_coef` | 0.01 | raise to 0.02–0.05 if entropy collapses before any success; lower to 0.005 if entropy stays pinned near log K and nothing commits |
| `ppo.ent_mode` | `fixed` | switch to `adaptive` if a fixed coefficient is bimodal across seeds. The target anneals *in proportion to measured success*, not on a clock |
| `ppo.prob_floor_start` → `prob_floor_end` | 0.05 → 0 on 8x8 | **use this, not `ent_coef`, when a specific action collapses.** `π(a) ≥ eps/K` for every action |

The per-action floor exists because entropy cannot express the constraint that
matters. Measured on Taxi: mean entropy 1.231 of log 6 = 1.792 (healthy) while
`π(PICKUP)` averaged 0.0016 and `π(DROPOFF)` peaked at 0.007 — success was
impossible and every aggregate metric looked fine. DoorKey has the same shape:
`pickup` (3) and `toggle` (5) are mandatory. Watch §8 of the notebook, not
entropy.

---

## Step 5 — optimisation

| knob | default | notes |
|---|---|---|
| `ppo.learning_rate` | 1e-3 | the rl-starter-files value. Halve it if `approx_kl` regularly exceeds ~0.03 |
| `ppo.anneal_lr` | false | the reference uses a constant lr |
| `ppo.target_kl` | null | set 0.03 to early-stop an epoch when updates get too large |
| `ppo.gae_lambda` | 0.95 | lower (0.9) reduces variance and helps when reward is very sparse and late |
| `ppo.gamma` | 0.99 | 640-step episodes have a horizon of ~100 at 0.99; 0.995 if the goal is consistently reached late |
| `ppo.vf_coef` | 0.5 | raise to 1.0 if `explained_variance` stays low while the policy is learning |
| `ppo.norm_adv` | `batch` | `batch` preserves the relative size of advantages *between* minibatches, which matters when only one minibatch contains the rewarded transition. `minibatch` is the CleanRL default and makes an all-zero minibatch look as informative as a rewarded one |
| `ppo.norm_adv_min_std` | 1e-6 | the zero-variance guard. Do not set to 0 — that reintroduces the noise-amplification bug |

---

## Step 6 — architecture

| knob | default | notes |
|---|---|---|
| `ppo.share_encoder` | true (MiniGrid) | one conv trunk for both heads, as in rl-starter-files. Halves the parameters and lets the value loss shape the policy's features, which helps when the policy gradient carries almost nothing. `false` keeps V and π cleanly independent, which is what NB02/NB06 reason about |
| `ppo.hidden_sizes` | `[64]` | head after the encoder. `[128, 64]` if the encoder is clearly saturating |
| `ppo.separate_grad_clip` | false when shared | automatically disabled with a shared trunk; a shared parameter belongs to neither head alone |
| `env.fully_observable` | true | **do not change.** `A_CF = r + γV(s′) − V_π(s)` is only defined in an MDP; the 7×7 egocentric view makes DoorKey a POMDP |

---

## Diagnostic → knob, at a glance

| what the plots show | most likely fix |
|---|---|
| `adv_std_raw` at the floor, everything else flat | step 1 or 3 — nothing else can help |
| key rate rises, door rate flat | `prob_floor_start` (is `toggle` alive?), then `potential_door` |
| door rate rises, success flat | more frames; `gamma` 0.995 |
| success rises then collapses to 0 | `ent_coef` down, or `ent_mode: adaptive` |
| entropy pinned near log K, `clipfrac` ~0 | lr up, or `ent_coef` down |
| entropy → 0 before any success | `ent_coef` up, or `ent_mode: adaptive` (irrecoverable once it happens) |
| `explained_variance` low while policy improves | `vf_coef` up, `n_epochs` up — but read the **median**, not the last value: once the task is solved the return variance collapses and EV becomes noisy by construction. Cross-check with `sqrt(2 * v_loss)`, the critic's error in reward units |
| `approx_kl` > 0.03 routinely | lr down, or set `target_kl` |
| a specific `π(a)` below 0.01 | `prob_floor_start` — entropy will not fix this |

---

## Wall clock

Measured ~1,700 steps/s for the MiniGrid CNN config on CPU. So roughly:

| run | frames | time |
|---|---|---|
| 5x5 smoke | 200k | ~2 min |
| 5x5 full | 500k | ~5 min |
| 6x6 full | 1.5M | ~15 min |
| 8x8 full | 3M | ~30 min |

The curriculum end to end is well under an hour, which is why it is worth
running rungs in order rather than gambling 3M frames on 8x8 directly.

---

## PPO-CF (notebook 02)

`ppo.pg_mode` switches the policy gradient. Everything else is shared, so a
control/treatment pair differs only in this one field.

| knob | default | notes |
|---|---|---|
| `ppo.pg_mode` | `gae` | `cf_all_action` turns on the counterfactual oracle |
| `ppo.cf_restore` | `exact` | `fast` is ~2.5x quicker (240 -> 625 collected steps/s on 5x5) and verified bit-identical by notebook 02 section 1.5. Switch after that check passes |
| `ppo.cf_validate` | `true` | replays recorded actions on the first rollout and asserts the restore is exact. Leave on; it costs nothing |
| `ppo.target_kl` | `null` | set `0.03` on **both** arms if the CF arm's `approx_kl` runs much larger than the control's — the all-action gradient moves all K logits per state, so at equal lr it takes bigger steps, which would confound a win |

New scalars: `cf_scale` (RMS of the pi-weighted advantage, the scale-only
normaliser), `cf_centering` (must stay ~0 — above 1e-4 the all-action gradient
is biased and the run is invalid), `cf_mean_abs`.

Reward shaping and `cf_all_action` are mutually exclusive and `config._validate`
refuses the pair: the oracle reads the raw environment reward while the critic
would have been trained on the shaped one, so `A_CF` would mix two different
reward functions.
