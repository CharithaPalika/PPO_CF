# RUNBOOK — E1 on the cluster

Every command you type is in this file. `slurm/README.md` explains *why* and contains
nothing to copy.

Experiment: **E1** (`All_docs/Experiments.pdf` §8.2) on RedBlueDoors, three arms —
`gae` (plain PPO), `cf` (PPO-CF), `shuf` (action-permuted CF control).

---

## 0. Get the repo onto the cluster

```bash
git clone <your remote> ppo_cf && cd ppo_cf
# or: rsync -av --exclude runs --exclude figures --exclude '*.npz' ./ user@cluster:~/ppo_cf/
```

Nothing is path-dependent: `slurm/_prelude.sh` and `config/config.py` both derive the
project root from their own file location, so the checkout runs wherever it lands.

---

## 1. Bring-up — first hour on a cluster you have not used

Each step is cheap and rules out a class of failure that is much harder to diagnose later.

| # | Run | What you are reading off it |
|---|---|---|
| 1 | `sinfo -s` | partition names, per-partition walltime limits, node counts |
| 2 | `scontrol show partition <name>` | the walltime cap. `02_run_chunks.sbatch` asks for **12 h** |
| 3 | `sacctmgr show assoc user=$USER format=account,maxjobs,maxsubmit` | your caps → `MAX_SUBMIT` and `CONCURRENCY` |
| 4 | `sinfo -o "%20N %10c %10m %25f"` | node families and features for `--constraint` |
| 5 | `df -h ~` and your site's quota command | home is usually NFS and small. This experiment writes ~160 MB |
| 6 | your site's docs | priority bands — many clusters give a large bonus below a walltime threshold |

Then **edit two things**:

1. `slurm/_submit_lib.sh` — `MAX_SUBMIT`, `CONCURRENCY`, and `PARTITION` if you need one.
2. If you set a partition, uncomment the `#SBATCH --partition=` line in each
   `slurm/*.sbatch`.

If your walltime cap is below 12 h, lower `--time` in `02_run_chunks.sbatch` **and**
lower `--minutes` / `--margin_min` in the same file to match — `minutes` is walltime
minus roughly 30 min, and `margin_min` must stay above the slowest unit (~5.5 h = 330 min).

---

## 2. Build the environment

```bash
sbatch slurm/00_create_venv.sbatch
tail -f logs/venv_<jobid>.out
```

Read the **whole** log. Every import must pass. If several native packages fail
identically the venv itself is damaged — rerun this script (it uses `--clear`) before
investigating anything else.

Optional, for wandb:

```bash
source ~/.venvs/ppo_cf/bin/activate && wandb login    # writes ~/.netrc, shared by compute nodes
grep -s api.wandb.ai ~/.netrc                          # already logged in? then env.sh needs no key
```

To turn wandb off entirely: `export PPO_CF_WANDB=0` before submitting. `scalars.csv`
is unaffected either way.

---

## 3. Verify

```bash
sbatch slurm/00_verify.sbatch
cat logs/verify_<jobid>.out
```

Installs nothing. Checks, in order: every repo module imports; every stage config
builds; the MiniGrid simulator restore is bit-exact on both RedBlueDoors variants
(the failure mode that yields plausible, *wrong* `A_CF`); and one real PPO-CF update
completes with finite losses.

---

## 4. Smoke the whole chain before the real thing

`PPO_CF_SMOKE_FRAMES` shrinks every unit to a few updates on a small batch, so the
whole chain — manifest → a real array → merge → requeue → summary — runs in minutes
instead of ~74 core-hours.

```bash
export PPO_CF_SMOKE_FRAMES=8192      # propagates to every job via --export=ALL
./submit_e1.sh 6
```

It writes **nowhere near your real results**: smoke output goes to `runs/_smoke/` and
`artifacts/_smoke/`, a parallel namespace, so a smoke pass can never mark a real unit
as done. Delete those two directories whenever you like.

Then check the one line that matters and stop:

```bash
cat logs/manifest_<jobid>.out        # MUST say "3 groups: gae, cf, shuf"
squeue -u $USER
```

Cancel it mid-array on purpose and resubmit — the only way to know your resume path
works is to interrupt it once while the units are cheap:

```bash
scancel <arrayjobid>
./submit_e1.sh 6                     # the ledger keeps what finished; only the rest reruns
```

When you are satisfied:

```bash
unset PPO_CF_SMOKE_FRAMES            # REQUIRED before the real run
```

## 5. Run it

```bash
./submit_e1.sh          # 16 chunks; 6x6 then 8x8, automatically
./submit_e1.sh 12       # fewer chunks if your submit cap is tight
```

**Read `logs/manifest_<jobid>.out` before you walk away.** It prints the number of arms
it actually saw. It must say 3. That line is the receipt for the two variable-mangling
guards (`--export` splitting on commas, and `GROUPS` being a bash built-in — which is
why the variable is called `PROJ_GROUPS`).

Check `PPO_CF_SMOKE_FRAMES` is unset first, or you will queue 45 two-second jobs.

To run 6x6 only and look at the result before paying for 8x8, change the last argument
of `submit_stage` in `submit_e1.sh` from `"E1_RBD8"` to `"E1_RBD6"`.

### What it costs

Measured from this repo's own `runs/*/summary.json`:

| stage | arm | frames | per seed | seeds | subtotal |
|---|---|---|---|---|---|
| `E1_RBD6` | gae | 1M | ~13 min | 10 | 2.2 h |
| `E1_RBD6` | cf | 1M | ~45 min | 10 | 7.5 h |
| `E1_RBD6` | shuf | 1M | ~45 min | 10 | 7.5 h |
| `E1_RBD8` | gae | 2M | ~25 min | 5 | 2.1 h |
| `E1_RBD8` | cf | 2M | ~5.5 h | 5 | 27.5 h |
| `E1_RBD8` | shuf | 2M | ~5.5 h | 5 | 27.5 h |

**45 units, ~74 core-hours, ~160 MB on disk.** At 16-way concurrency, roughly 6–8 h
wall clock. The 8x8 CF estimate assumes the doubled branch budget (`cf_horizon` 64 at
`cf_subsample` 0.05 = 6.4, against 3.2 in the committed YAML).

---

## 6. Watch it

```bash
squeue -u $USER                                       # what is queued and running
squeue -u $USER --start                               # estimated start times
squeue -j <id> -o "%.18i %.9P %.8T %R"                # WHY a job is still pending
tail -f logs/run_<A>_0.out                            # follow one array task
grep -l Traceback logs/run_*_*.err | head             # which tasks raised
sacct -j <id> --format=JobID,State,Elapsed,MaxRSS,ExitCode
scancel <id>            # or: scancel -u $USER        # safe: the ledger keeps what finished
scancel <arrayid>_[5-15]                              # cancel part of an array
scontrol show job <id>                                # incl. the resolved --export environment
```

Progress without reading logs:

```bash
wc -l artifacts/chunks/E1_RBD6_chunk_*.jsonl          # units finished, per chunk
column -s, -t artifacts/tables/E1_RBD6.csv | cut -c1-150
```

---

## 7. When it finishes

`03_analyse` writes, per stage:

```
artifacts/tables/<STAGE>.csv            the ledger: one row per (arm, seed)
artifacts/tables/<STAGE>_summary.csv    per-arm median / mean / std / n
artifacts/status/<STAGE>.json           the Eq (32) paired tests
artifacts/figures/<STAGE>_success.png   mean success curve per arm, IQR band
```

The Eq (32) checks are **reported, not blocking** — a failed ordering is a finding
about the method, not a reason to skip the next environment. They are paired on seed
with a 95% bootstrap CI, pre-registered in `pipeline/analyse.py` rather than chosen
after looking at the numbers.

### Copy the results home

```bash
# from your laptop. ~50 MB: everything except checkpoints.
rsync -av --exclude 'checkpoints' --exclude '*.npz' \
    user@cluster:~/ppo_cf/artifacts/ ./artifacts/
rsync -av --include '*/' --include 'scalars.csv' --include 'episodes.csv' \
    --include 'config.json' --exclude '*' \
    user@cluster:~/ppo_cf/runs/ ./runs/

# checkpoints only for the seeds you actually want to probe (~3 MB each)
rsync -av user@cluster:~/ppo_cf/runs/e1_rbd6_cf/seed_0/checkpoints/ \
    ./runs/e1_rbd6_cf/seed_0/checkpoints/
```

Then open the notebooks locally: everything they need is `artifacts/tables/*.csv` plus
each run's `scalars.csv` / `episodes.csv`.

### Upload the wandb runs

```bash
sbatch slurm/04_sync_wandb.sbatch
```

If your compute nodes have no outbound network that job fails harmlessly (the offline
directories stay on disk). Run it from wherever you do have network:

```bash
source ~/.venvs/ppo_cf/bin/activate
wandb sync artifacts/wandb/offline-run-*
```

---

## 8. When something goes wrong

| Symptom | Do |
|---|---|
| manifest log says fewer than 3 groups | `PROJ_GROUPS` was mangled. Submit through `./submit_e1.sh`, never a hand-written `--export=ALL,PROJ_GROUPS=gae,cf,shuf` |
| analyse says `STOPPING ... ZERO completed` | a fault hitting every unit, not a walltime cut. `tail -40 logs/run_*_0.err`. Nothing is lost |
| `Segmentation fault` / `SIGILL` in a run log | `sbatch slurm/09_diagnose.sbatch` — one process per import names the culprit |
| a unit failed but others finished | normal. Its row is absent; the next pass picks it up. `grep -A20 'FAILED' logs/run_*.out` |
| `no venv at ...` | `sbatch slurm/00_create_venv.sbatch` |
| want to redo one unit | delete its row from `artifacts/tables/<STAGE>.csv` and its `<STAGE>_chunk_*.jsonl` line, then resubmit |
| every Eq (32) check says `n<5` | fewer than 5 paired seeds finished. A bootstrap over 1–2 pairs is not evidence, so the gate refuses a verdict rather than printing a flattering one |
| results look like a smoke run | `PPO_CF_SMOKE_FRAMES` was still exported. `unset` it; real and smoke never share a file |
| want to rerun the summary only | `export STAGE=E1_RBD6 PROJ_GROUPS="gae,cf,shuf" PROJ_N_GROUPS=3 N_CHUNKS=16 PART_END=E1_RBD8 PASS_NO=0` then `sbatch --export=ALL slurm/03_analyse.sbatch` |

---

## 9. Things this setup does NOT do, on purpose

- **No notebook execution on the cluster.** The notebooks currently train inside
  themselves; running them here would duplicate the array's work. `03_analyse` writes
  the ledger and the figures, and you analyse locally from those.
- **No trajectory datasets.** `trajectories.npz` is 25 MB/seed and feeds NB03+ (E2/E3
  distillation), not E1. Set `PPO_CF_RECORD_TRAJECTORIES=1` before submitting if you
  want them — it costs ~1.1 GB instead of ~160 MB.
- **No mid-run resume.** A unit is atomic. A killed unit is redone from zero, which is
  why the walltime is sized from the slowest one.
