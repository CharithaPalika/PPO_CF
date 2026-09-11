# RUNBOOK - Cluster Runs

## E2: queried vs distilled counterfactual-advantage perturbation

E2 has **two** arms, not a new PPO/GAE comparison. E1 already provides the
PPO-GAE and direct all-action PPO-CF evidence. E2 asks whether the same sparse
counterfactual labels are more useful as a local perturbation or as supervision
for an all-state student.

| E2 arm | Exact CF labels | PPO advantage |
|---|---:|---|
| `queried` | Uniform 2% of rollout states | \(A_{GAE}+\beta A_{CF}(s,a_t)\) only on queried states |
| `distill` | Same uniform 2% | \(A_{GAE}+\beta\hat A_{CF,\phi}(s,a_t)\) at every state |

The pooled sweep contains four independent 30-run sets—five paired seeds, three
fixed beta values, and two E2 arms each—for **120 training runs total**:

| environment | stage inside the pooled ledger | units |
|---|---|---:|
| Taxi | `E2_TAXI` | 30 |
| DoorKey 6x6 | `E2_DK6` | 30 |
| UnlockPickup | `E2_UNLOCKPICKUP` | 30 |
| RedBlueDoors 6x6 | `E2_RBD6` | 30 |

The registered sweep is \(\beta\in\{0.25, 0.75, 1.5\}\). Beta zero is not
repeated: it is the PPO-GAE baseline already measured in E1.

`./submit_e2.sh` submits one globally interleaved `E2_ALL` manifest rather than
four sequential stages. Its 20 array chunks each receive three matched two-arm
`(environment, seed, beta)` pairs; queued chunks cycle through environments and
beta values. Slurm runs at most **12 concurrent workers** and starts queued
chunks as slots free up. This avoids waiting for an entire environment set to
finish.

### Upload the updated repository

From the local repository directory, first make the destination once if needed,
then copy the repository contents using this jump-host command:

```bash
ssh -J aniket@stujump.comp.nus.edu.sg aniket@xlogin.comp.nus.edu.sg 'mkdir -p ~/ppo_cf'
scp -J aniket@stujump.comp.nus.edu.sg -r ./* aniket@xlogin.comp.nus.edu.sg:~/ppo_cf/
```

Then connect and prepare the copied scripts:

```bash
ssh -J aniket@stujump.comp.nus.edu.sg aniket@xlogin.comp.nus.edu.sg
cd ~/ppo_cf
bash slurm/fix_line_endings.sh
chmod +x submit_e2.sh slurm/*.sh slurm/*.sbatch
```

### Build, authenticate, and verify

Do not train on `xlogin`. Create the project virtual environment on a compute
node, authenticate W&B once, and verify both new E2 modes:

```bash
mkdir -p logs
sbatch slurm/00_create_venv.sbatch
cat logs/venv_<jobid>.out

source .venv/bin/activate
wandb login
deactivate

sbatch slurm/00_verify.sbatch
cat logs/verify_<jobid>.out
```

The verification log must end with `verify OK` before the real submission.

### Submit E2

```bash
cd ~/ppo_cf
./submit_e2.sh
```

This uses 20 chunks by default (six preassigned runs / three matched pairs per
chunk), submitted as a single array throttled to 12 runners. To use a smaller
array while preserving the same 120-run manifest, pass a chunk count:

```bash
./submit_e2.sh 12
```

Check the initial manifest receipt before leaving:

```bash
cat logs/manifest_<jobid>.out
```

It must report exactly:

```text
[E2_ALL] 2 groups: queried, distill
[E2_ALL] 120 units pending
```

E2 forces W&B **online**. Each run has `distill.beta` in its W&B config, a
`beta-<value>` tag, and an environment/beta W&B group (for example,
`E2_TAXI_beta-0.75`). Retries of the same `(environment, arm, seed, beta)` use
a stable W&B run ID, while the persistent `E2_ALL` ledger makes completed units
skip automatically on requeue or resubmission.

If you deliberately delete an E2 run in W&B, do **not** reuse its run-ID
namespace: W&B reserves deleted IDs permanently. Before the next fresh sweep,
change `PPO_CF_WANDB_RUN_NAMESPACE` in `submit_e2.sh` (for example, from
`e2_beta_sweep_v2` to `e2_beta_sweep_v3`). Keep it unchanged while a sweep is
active so interrupted units still resume their own W&B curves.

### Monitor and collect E2

```bash
squeue -u $USER
tail -f logs/run_<arrayjob>_0.out
grep -l Traceback logs/run_*_*.err | head
wc -l artifacts/chunks/E2_ALL_chunk_*.jsonl
cat artifacts/status/E2_ALL.json
```

When complete, the pooled ledger is `artifacts/tables/E2_ALL.csv`; use its
`stage` and `beta` columns to separate the twelve environment/beta conditions.
The paired E2 result is `distill` versus `queried` within each environment,
seed, and beta. No `wandb sync` is needed because these E2 runs are online.

This runbook is for launching Experiment 1 on the Slurm cluster. The entry point
is:

```bash
./experiment_1_6_envs
```

Despite the filename, this experiment currently contains seven environments.

## Experiment

Experiment 1 compares two algorithms on each environment:

| group | algorithm | `ppo.pg_mode` | wandb algo tag |
|---|---|---|---|
| `gae` | PPO | `gae` | `ppo` |
| `cf` | PPO-CF | `cf_all_action` | `ppo-cf` |

Each stage runs 5 seeds: `0, 1, 2, 3, 4`.

| stage | environment | config source | wandb env tag | units |
|---|---|---|---|---|
| `EXP1_DK5` | DoorKey 5x5 | `doorkey5x5_cf` | `doorkey5x5` | 10 |
| `EXP1_DK6` | DoorKey 6x6 | `doorkey6x6_cf` | `doorkey6x6` | 10 |
| `EXP1_LAVAGAP` | LavaGap S6 | `lavagap_cf` | `lavagap` | 10 |
| `EXP1_UNLOCK` | Unlock | `unlock_cf` | `unlock` | 10 |
| `EXP1_UNLOCKPICKUP` | UnlockPickup | `unlockpickup_cf` | `unlockpickup` | 10 |
| `EXP1_RBD6` | RedBlueDoors 6x6 | `redbluedoors6x6_cf` | `redbluedoors6x6` | 10 |
| `EXP1_TAXI` | Taxi | `taxi_cf` | `taxi` | 10 |

Total: 70 training units.

The pipeline uses each `*_cf` YAML as the single source of truth. For the PPO arm
it overrides only `ppo.pg_mode=gae`; for the PPO-CF arm it uses
`ppo.pg_mode=cf_all_action`. The submit path also overrides seed, run name,
trajectory recording, and log frequency.

## Cluster Setup

SSH to the cluster login node and enter the repo:

```bash
ssh -J <you>@stujump.comp.nus.edu.sg <you>@xlogin.comp.nus.edu.sg
cd ~/ppo_cf
```

After copying files from a non-git source, fix line endings and executable bits:

```bash
bash slurm/fix_line_endings.sh
chmod +x experiment_1_6_envs submit_*.sh slurm/*.sh slurm/*.sbatch
```

Do not run training on the login node. The only login-node actions should be
`sbatch`, `squeue`, `sacct`, `scancel`, reading logs, and submitting this wrapper.

## Virtualenv

The Slurm jobs use the project virtualenv at `./.venv`, not the local conda
environment. Build it on a compute node:

```bash
mkdir -p logs
sbatch slurm/00_create_venv.sbatch
squeue -u $USER
cat logs/venv_<jobid>.out
```

The verify section of that log should show successful imports.

## Wandb

Slurm enables wandb through `slurm/_prelude.sh`. Runs are written in offline mode
under `artifacts/wandb/` and uploaded later.

If wandb needs credentials, login once from the repo after the venv exists:

```bash
source .venv/bin/activate
wandb login
deactivate
```

Each run gets exactly two wandb tags:

```text
<env tag>,<algo tag>
```

Examples:

```text
doorkey5x5,ppo
doorkey5x5,ppo-cf
unlockpickup,ppo
unlockpickup,ppo-cf
```

To disable wandb for a submission:

```bash
export PPO_CF_WANDB=0
```

## Verify

Before launching the real experiment:

```bash
sbatch slurm/00_verify.sbatch
cat logs/verify_<jobid>.out
```

It should end with `verify OK`.

## Smoke Run

Smoke mode shrinks every unit and writes only under `runs/_smoke/` and
`artifacts/_smoke/`.

```bash
export PPO_CF_SMOKE_FRAMES=8192
./experiment_1_6_envs 6
cat logs/manifest_<jobid>.out
squeue -u $USER
```

The manifest log must say:

```text
2 groups: gae, cf
```

When the smoke run is done:

```bash
unset PPO_CF_SMOKE_FRAMES
rm -rf runs/_smoke artifacts/_smoke
```

## Real Run

Submit the full chain:

```bash
cd ~/ppo_cf
./experiment_1_6_envs
```

Use fewer chunks if your submit cap is tight:

```bash
./experiment_1_6_envs 8
```

The first stage is `EXP1_DK5`. When a stage completes, `03_analyse.sbatch`
submits the next stage automatically:

```text
EXP1_DK5 -> EXP1_DK6 -> EXP1_LAVAGAP -> EXP1_UNLOCK -> EXP1_UNLOCKPICKUP -> EXP1_RBD6 -> EXP1_TAXI
```

Before walking away, check the manifest log:

```bash
cat logs/manifest_<jobid>.out
```

It must say `2 groups: gae, cf`. If it says only one group, stop and resubmit
through `./experiment_1_6_envs`; do not hand-write `sbatch --export` with a
comma-separated group value.

## Monitoring

Useful commands:

```bash
squeue -u $USER
squeue -u $USER --start
squeue -j <jobid> -o "%.18i %.9P %.8T %R"
tail -f logs/run_<arrayjob>_0.out
grep -l Traceback logs/run_*_*.err | head
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ExitCode
```

Progress from artifacts:

```bash
wc -l artifacts/chunks/EXP1_*_chunk_*.jsonl
ls artifacts/tables/EXP1_*.csv
```

Canceling is safe. Completed units are recorded in the ledger and skipped on the
next submission:

```bash
scancel <jobid>
```

## Results

Each stage writes:

```text
artifacts/tables/<STAGE>.csv
artifacts/tables/<STAGE>_summary.csv
artifacts/status/<STAGE>.json
artifacts/figures/<STAGE>_success.png
```

The CSV ledger has one row per `(group, seed)` unit and includes `env_tag` and
`algo_tag`.

Upload offline wandb runs after the Slurm jobs finish:

```bash
sbatch slurm/04_sync_wandb.sbatch
```

Copy results off the cluster:

```bash
tar --exclude='checkpoints' --exclude='*.npz' -czf ~/experiment_1_results.tgz \
    artifacts runs/*/*/scalars.csv runs/*/*/episodes.csv runs/*/*/config.json
```

Then from your local machine:

```bash
scp -J <you>@stujump.comp.nus.edu.sg \
    <you>@xlogin.comp.nus.edu.sg:~/experiment_1_results.tgz .
```

## Common Failures

| symptom | fix |
|---|---|
| `sbatch: error: Batch script contains DOS line breaks` | `bash slurm/fix_line_endings.sh` |
| `REFUSING: not inside a Slurm job` | submit with `sbatch` or `./experiment_1_6_envs`; do not run `.sbatch` files directly |
| `no venv at ...` | run `sbatch slurm/00_create_venv.sbatch` |
| manifest shows one group | resubmit through `./experiment_1_6_envs`; do not hand-write comma exports |
| results look tiny | `PPO_CF_SMOKE_FRAMES` is still set; run `unset PPO_CF_SMOKE_FRAMES` |
| analyse stops with zero progress | read `logs/run_*_*.err`; a fault is hitting every unit |
