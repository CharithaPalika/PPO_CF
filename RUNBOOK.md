# RUNBOOK — running the PPO-CF experiments on the cluster

Written for someone who has just cloned this repository and has an account on the
**NUS SoC compute cluster** (`xlogin.comp.nus.edu.sg`). Every command you type is
in this file. `slurm/README.md` explains *why* the pipeline is shaped the way it
is and contains nothing to copy.

## What you will be running

| stage | environment | arms | seeds | frames | cost |
|---|---|---|---|---|---|
| `E1_RBD6` | RedBlueDoors-6x6 | gae / cf / shuf | 0–4 | 2M | ~17 core-h |
| `E1_RBD8` | RedBlueDoors-8x8 | gae / cf / shuf | 0–4 | 2M | ~30 core-h |
| `E0_reddoorbluedoor_test` | RedBlueDoors-8x8 | cf only | 0, 2 | 2M | ~5.5 core-h |

The three arms differ in exactly one config field, `ppo.pg_mode`:

| arm | `pg_mode` | what it is |
|---|---|---|
| `gae` | `gae` | plain PPO |
| `cf` | `cf_all_action` | PPO-CF, full direct counterfactual oracle |
| `shuf` | `cf_shuffled` | the same oracle vectors on *different* states — the control |

E1 is the experiment (§7). E0 is a small standalone re-test on 8x8 (§8) and is
the cheapest thing to run first if you just want to see the machinery work on
something real.

---

## 1. Get the code onto the cluster

```bash
ssh -J <you>@stujump.comp.nus.edu.sg <you>@xlogin.comp.nus.edu.sg
git clone <repo-url> ~/ppo_cf
cd ~/ppo_cf
```

A one-time `~/.ssh/config` on your own machine saves typing the jump host every
time:

```
Host stujump
    HostName stujump.comp.nus.edu.sg
    User <you>

Host xlogin
    HostName xlogin.comp.nus.edu.sg
    User <you>
    ProxyJump stujump
```

Then `ssh xlogin` is enough.

### If you copied files instead of cloning

`.gitattributes` pins `eol=lf`, so a **clone** is always correct. A copy from a
Windows machine is not: `sbatch` refuses a script with CRLF line endings, and a
*sourced* file (`_prelude.sh`, `_submit_lib.sh`, `env.sh`) is worse — bash does
not refuse it, the `\r` just joins the last word on every line and fails
somewhere that looks unrelated. After any copy:

```bash
cd ~/ppo_cf
bash slurm/fix_line_endings.sh
chmod +x submit_*.sh slurm/*.sh slurm/*.sbatch      # scp drops the executable bit
```

Copying with `scp` through the jump host, if you need it:

```cmd
scp -J <you>@stujump.comp.nus.edu.sg -r <path>\PPO_CF <you>@xlogin.comp.nus.edu.sg:~/ppo_cf
```

---

## 2. Nothing runs on the login node

This is the rule the whole setup is built around, and it is **enforced**: every
job script exits immediately if it is not inside a Slurm job. Running
`bash slurm/02_run_chunks.sbatch` on xlogin prints

```
REFUSING: not inside a Slurm job. Batch scripts are submitted, not run.
```

and stops. (`PPO_CF_ALLOW_LOGIN=1` is the escape hatch if you ever need it.)

| Runs on **xlogin** | Runs on a **compute node** |
|---|---|
| `sbatch ...`, `./submit_e1.sh` | training, analysis, `pip install`, wandb upload |
| `squeue` / `sacct` / `scancel` / `tail` on a log | everything else, without exception |
| `bash slurm/fix_line_endings.sh` (a `sed` over ~70 small files) | |

**You can close your SSH session.** `sbatch` hands the job to Slurm and returns;
the job does not belong to your shell. The chain — manifest → array → analyse →
requeue-or-next-stage — is carried by Slurm job dependencies and by the analyse
job calling `sbatch` from inside its own compute-node job. No `tmux`, no `nohup`.

**Do not use `salloc` / `srun` for setup.** An interactive allocation dies when
your SSH session drops, and `srun pip install ...` runs the *system* python and
fails with `externally-managed-environment`. The venv build is a batch job (§4).

---

## 3. Cluster settings

Measured on xlogin, already applied in the `#SBATCH` headers:

| Partition | MaxTime | Priority factor | Used for |
|---|---|---|---|
| `normal` (default) | **3:00:00** | 4 | venv, verify, manifest, analyse, wandb sync |
| `long` | 3-00:00:00 | 1 | `02_run_chunks` (12 h array), `10_e0_rbd8x8_test` (8 h) |

A single RedBlueDoors-8x8 CF unit runs ~2.75 h, so the arrays cannot use
`normal`. Nothing to edit here.

**One thing you must check for your own account:**

```bash
sacctmgr show assoc user=$USER format=account,maxjobs,maxsubmit
```

Put those two numbers into `MAX_SUBMIT` and `CONCURRENCY` at the top of
`slurm/_submit_lib.sh` (they default to 32 and 16). Exceeding the running-job cap
does not get you more nodes; it gets submissions rejected or your priority cut.

---

## 4. Build the environment

The virtualenv lives **inside the project** at `~/ppo_cf/.venv`, so the checkout
carries its own interpreter and two checkouts can never share one.

```bash
cd ~/ppo_cf
mkdir -p logs                       # Slurm will NOT create it, and --output needs it
sbatch slurm/00_create_venv.sbatch
squeue -u $USER                     # wait for it to disappear
cat logs/venv_<jobid>.out
```

Read the whole log — every import under "4. verify" must pass. If several native
packages fail identically the venv itself is damaged; just resubmit, the script
uses `--clear`.

### Weights & Biases (optional)

Compute nodes here have outbound network, so a token in `~/.netrc` is enough and
every node shares it:

```bash
source .venv/bin/activate && wandb login && deactivate
grep -s api.wandb.ai ~/.netrc       # already logged in? then nothing else to do
```

To run without wandb entirely: `export PPO_CF_WANDB=0` before submitting.
`scalars.csv` is written first and unconditionally, so nothing is ever lost to a
tracking failure.

---

## 5. Verify

```bash
sbatch slurm/00_verify.sbatch
cat logs/verify_<jobid>.out
```

Installs nothing. Checks, in order: no CRLF anywhere; every repository module
imports; every stage config builds; the MiniGrid simulator restore is bit-exact
on both RedBlueDoors variants (the failure mode that yields plausible but
**wrong** `A_CF`); and one real PPO-CF update completes with finite losses. It
must end with `verify OK`.

---

## 6. Smoke the whole chain

`PPO_CF_SMOKE_FRAMES` shrinks every unit to a few updates, so manifest → array →
merge → requeue → summary runs in minutes instead of ~47 core-hours. It writes
only into `runs/_smoke/` and `artifacts/_smoke/`, a parallel namespace, so a
smoke pass can never mark a real unit as done.

```bash
export PPO_CF_SMOKE_FRAMES=8192
./submit_e1.sh 6

cat logs/manifest_<jobid>.out        # MUST say "3 groups: gae, cf, shuf"
squeue -u $USER
```

Then cancel it mid-array on purpose and resubmit — the only way to know the
resume path works is to interrupt it once while the units are cheap:

```bash
scancel <arrayjobid>
./submit_e1.sh 6                     # finished units are skipped; only the rest rerun
```

When you are satisfied:

```bash
unset PPO_CF_SMOKE_FRAMES            # REQUIRED, or the real run finishes in seconds
rm -rf runs/_smoke artifacts/_smoke
```

---

## 7. Run E1

```bash
cd ~/ppo_cf
./submit_e1.sh          # 16 chunks; E1_RBD6 then E1_RBD8, automatically
./submit_e1.sh 12       # fewer chunks if your submit cap is tight
```

**Before you walk away, read one line:**

```bash
cat logs/manifest_<jobid>.out        # must say "3 groups: gae, cf, shuf"
```

That is the receipt for two variable-mangling guards — `sbatch --export` splits
on commas with no escaping, and `GROUPS` is a bash built-in array (which is why
the variable is called `PROJ_GROUPS`). If it says 1 group, stop and read §11.

To run 6x6 only and look at the result before paying for 8x8, change the last
argument of `submit_stage` in `submit_e1.sh` from `"E1_RBD8"` to `"E1_RBD6"`.

30 units, ~47 core-hours, ~110 MB on disk. At 16-way concurrency, roughly 6–8 h
wall clock per stage.

---

## 8. Run E0 — the RedBlueDoors-8x8 two-seed re-test

Standalone, separate from E1: **PPO-CF only, seeds 0 and 2, 2M frames,
`config/envs/redbluedoors8x8_cf.yaml` with no overrides at all.** This is the
configuration that produced `runs/rbd8x8`, where seed 0 reached success 0.72.
Seed 1 is skipped — it reached 0.00 with `cf_reward_coverage` 0.000 at the same
settings.

```bash
sbatch slurm/10_e0_rbd8x8_test.sbatch     # array 0-1, one seed per task, in parallel
tail -f logs/e0test_<jobid>_0.out
```

~2.75 h per seed on `long`. Two things differ from E1 deliberately: it logs
**online** to wandb (project `ppo-cf-sweeps`) because only two tasks start at
once, and it **keeps `trajectories.npz`** (~25 MB/seed) because the YAML says
`record_trajectories: true` and this stage reproduces that config exactly.

The same thing without Slurm, one seed at a time:

```bash
python -m scripts.run_e0_rbd8x8_test --dry-run     # print the config, run nothing
python -m scripts.run_e0_rbd8x8_test --seeds 2
```

**Watch `cf_reward_coverage`, not just success.** It was 0.078 on the seed that
learned and 0.000 on the seed that did not. Near zero means the counterfactual
teacher is empty and the arm is running on critic noise, whatever the success
curve is doing. RedBlueDoors-8x8 has large seed variance — do not read one seed
as a result.

---

## 9. Watch it

```bash
squeue -u $USER                                       # queued and running
squeue -u $USER --start                               # estimated start times
squeue -j <id> -o "%.18i %.9P %.8T %R"                # WHY a job is still pending
tail -f logs/run_<A>_0.out                            # follow one array task
grep -l Traceback logs/run_*_*.err | head             # which tasks raised
sacct -j <id> --format=JobID,State,Elapsed,MaxRSS,ExitCode
scancel <id>            # or: scancel -u $USER        # safe: the ledger keeps what finished
scancel <arrayid>_[5-15]                              # cancel part of an array
```

Progress without reading logs:

```bash
wc -l artifacts/chunks/E1_RBD6_chunk_*.jsonl          # units finished, per chunk
column -s, -t artifacts/tables/E1_RBD6.csv | cut -c1-150
```

---

## 10. Results

The analyse job writes, per stage:

```
artifacts/tables/<STAGE>.csv            the ledger: one row per (arm, seed)
artifacts/tables/<STAGE>_summary.csv    per-arm median / mean / std / n
artifacts/status/<STAGE>.json           the paired Eq (32) tests
artifacts/figures/<STAGE>_success.png   mean success curve per arm, IQR band
```

The Eq (32) checks (`cf > gae` **and** `cf > shuf`) are **reported, not
blocking** — a failed ordering is a finding about the method, not a reason to
skip the next environment. They are paired on seed with a 95% bootstrap CI, and
report `n<5` rather than a verdict when fewer than 5 paired seeds finished.

### Copy results off the cluster

```cmd
scp -J <you>@stujump.comp.nus.edu.sg -r <you>@xlogin.comp.nus.edu.sg:~/ppo_cf/artifacts "%USERPROFILE%\Downloads"
```

That is everything the analysis reads. For the per-run curves as well, pack on
the cluster first so you move one file instead of thousands (`tar` on the login
node is a file copy, not compute — the same class of action as `scp` itself):

```bash
cd ~/ppo_cf && tar --exclude='checkpoints' --exclude='*.npz' -czf ~/results.tgz \
    artifacts runs/*/*/scalars.csv runs/*/*/episodes.csv runs/*/*/config.json
```
```cmd
scp -J <you>@stujump.comp.nus.edu.sg <you>@xlogin.comp.nus.edu.sg:~/results.tgz "%USERPROFILE%\Downloads"
```

Analysis happens off the cluster from `artifacts/tables/*.csv` plus each run's
`scalars.csv` / `episodes.csv`. The notebooks in `notebooks/` read exactly those.

### Upload the offline wandb runs (E1 only)

```bash
sbatch slurm/04_sync_wandb.sbatch
```

Safe to resubmit; already-synced runs are skipped. E0 logs online and needs
nothing here. Do **not** run `wandb sync` on xlogin — a long upload started in an
SSH session dies with the session.

---

## 11. When something goes wrong

| Symptom | Fix |
|---|---|
| `sbatch: error: Batch script contains DOS line breaks` | `bash slurm/fix_line_endings.sh`, then resubmit |
| `REFUSING: not inside a Slurm job` | you ran a batch script instead of submitting it: `sbatch slurm/<script>.sbatch` |
| `No such file or directory: 'requirements.txt'` in a job log | submitted from the wrong directory; `cd ~/ppo_cf` first |
| `no venv at ...` | `sbatch slurm/00_create_venv.sbatch` (it builds `./.venv`) |
| `Permission denied` on `./submit_e1.sh` | `chmod +x submit_e1.sh`, or just `bash submit_e1.sh` |
| Slurm error about the output file | `mkdir -p logs` — Slurm does not create the `--output` directory |
| manifest log says fewer than 3 groups | submit through `./submit_e1.sh`, never a hand-written `--export=ALL,PROJ_GROUPS=...` |
| analyse says `STOPPING ... ZERO completed` | a fault hitting every unit, not a walltime cut. `tail -40 logs/run_*_0.err`. Nothing is lost — the ledger keeps what finished |
| a unit failed but others finished | normal. Its row is absent; the next pass picks it up. `grep -A20 FAILED logs/run_*.out` |
| Eq (32) checks all say `n<5` | fewer than 5 paired seeds finished; resubmit — the ledger skips what is already done |
| `Segmentation fault` / `SIGILL` in a run log | `sbatch slurm/09_diagnose.sbatch` — one process per import names the culprit |
| results look like a smoke run | `PPO_CF_SMOKE_FRAMES` was still exported. `unset` it |
| want to redo one unit | delete its row from `artifacts/tables/<STAGE>.csv` and its line from the matching `artifacts/chunks/<STAGE>_chunk_*.jsonl`, then resubmit |
| want to rerun only the summary | `export STAGE=E1_RBD6 PROJ_GROUPS="gae,cf,shuf" PROJ_N_GROUPS=3 N_CHUNKS=16 PART_END=E1_RBD8 PASS_NO=0` then `sbatch --export=ALL slurm/03_analyse.sbatch` |

---

## 12. Things this setup does NOT do, on purpose

- **No notebook execution on the cluster.** The notebooks train inside
  themselves, so running them here would duplicate the array's work. The analyse
  job writes the ledger and figures; you analyse locally from those.
- **No trajectory datasets in E1.** `trajectories.npz` is ~25 MB/seed and feeds
  the later distillation work, not E1. Set `PPO_CF_RECORD_TRAJECTORIES=1` before
  submitting if you want them. E0 keeps them, as described in §8.
- **No mid-run resume.** One unit = one `(stage, arm, seed)` training run, and it
  is atomic. A killed unit is redone from zero, which is why the array walltime
  is sized from the slowest unit rather than the average.
- **A fresh clone has no `runs/`.** It is gitignored. That is fine for
  RedBlueDoors — every `redbluedoors*_cf.yaml` has `init_from: null`. The DoorKey
  curriculum configs do warm-start from an earlier rung's checkpoint, so those
  need the previous rung trained first.
