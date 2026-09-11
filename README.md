# PPO-CF

PPO baseline + explicit counterfactual oracle, for testing whether a
counterfactual advantage signal helps PPO.

## Layout

```
config/
  config.py            schema (dataclasses) + YAML loader. NO per-env values here.
  envs/*.yaml          one file per environment -- this is where you tune
agents/
  networks.py          ActorCritic, MiniGridCNN (grid-size agnostic)
  buffer.py            rollout buffer + GAE
  ppo.py               PPOTrainer
envs/
  env_pool.py          synchronous env pool, exact sim-state get/restore
  minigrid_env.py      MiniGrid wrappers + state packing
  shaping.py           optional potential shaping / count bonus (both OFF)
  scaling.py           observation scalers
dataio/                trajectories, checkpoints, landscapes
oracle/                NB02's counterfactual machinery (not run yet)
scripts/
  train.py             the runner: python -m scripts.train --env <name>
  build_oracle.py      NB02's builder
notebooks/
  01_ppo_baseline.ipynb   env-agnostic; one knob picks the environment
utils/                 logging, plotting, seeding
runs/                  outputs (gitignored)
to_delete/             archived work, safe to remove
```

## Run something

```bash
pip install -r requirements.txt

python -m scripts.train --list                      # what is configured
python -m scripts.train --env doorkey5x5            # rung 1 of the curriculum
python -m scripts.train --env doorkey5x5 --frames 200000 --run-name smoke5x5
python -m scripts.train --env doorkey6x6            # warm-starts from 5x5
python -m scripts.train --env doorkey8x8            # warm-starts from 6x6

# override any field without editing anything
python -m scripts.train --env doorkey8x8 \
    --set ppo.ent_coef=0.02 env.layout_seeds=[0,1,2,3] --dry-run
```

Every run writes `runs/<run_name>/config.json` — the exact configuration used —
so a result is always traceable to what produced it.

## Adding an environment

Copy a file in `config/envs/`, change `env.env_id`, and run it. No Python
changes are needed unless the environment needs a new observation scaler or a
new simulator-state accessor.

## Tuning

`TUNING.md` lists the knobs in the order worth trying, each tied to the
diagnostic that justifies pulling it.

---

## Running the experiments on the cluster

- **`RUNBOOK.md`** — start here. Clone, build, verify, smoke, submit, collect
  results, troubleshoot. Written for someone setting this up from scratch.
- **`slurm/README.md`** — why the pipeline is shaped the way it is; no commands.

### E2: queried versus distilled CF-advantage perturbation

E2 does **not** rerun PPO-GAE or direct all-action PPO-CF: E1 already answers
that comparison. E2 uses the same uniform 2% CF-label budget in two paired
perturbation arms:

```text
queried   inject exact A_CF(s, a_t) only at the queried 2% of states
distill   train on those same labels, then inject predicted A_CF(s, a_t) at all states
```

The selected environments are Taxi, DoorKey-6x6, UnlockPickup, and
RedBlueDoors-6x6: 4 environments × 5 paired seeds × 3 beta values
(`0.25`, `0.75`, `1.5`) × 2 arms = 120 runs. Run a
local experiment through `notebooks/03_e2_landscape_distillation.ipynb`, or on
the cluster through:

```bash
./submit_e2.sh
```

This creates one globally interleaved `E2_ALL` manifest, not a sequential
environment chain. Its 20 preassigned six-run chunks contain matched
`(environment, seed, beta)` pairs; the array is throttled to 12 workers and
starts queued chunks as slots free up. It preserves completed `(environment
stage, arm, seed, beta)` rows and requeues only outstanding units. E2 sets W&B
to `online`; beta is stored in W&B config, run group, and tags, and a retry
attaches to the existing curve.
See `RUNBOOK.md` for the upload, verification, and launch commands.

E2 deliberately does not use uncertainty for gating or state selection. The
only E2 question is whether uniform sparse CF labels can be amortised over
unqueried states:

```text
distill > queried
```

### E3: uncertainty-driven distillation

E3 keeps the E2 distilled perturbation setup but changes which states receive
the 2% CF-label budget. It runs Taxi, DoorKey-6x6, UnlockPickup, and
RedBlueDoors-6x6 for three paired seeds (`0`, `1`, `2`) with beta fixed at
`0.75`:

```text
uniform      uniform label selection
uncertainty top ensemble-uncertainty states
active      top uncertainty-times-leverage states
```

Launch the cluster sweep with:

```bash
./submit_e3.sh
```

For a local sanity run, use `notebooks/04_e3_uncertainty_distillation.ipynb`.
The pooled `E3_ALL` manifest contains 36 runs. The primary comparison is:

```text
active > uniform
```
