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
