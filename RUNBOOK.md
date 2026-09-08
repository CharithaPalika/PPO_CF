# RUNBOOK — E1 on the cluster

Every command you type is in this file. `slurm/README.md` explains *why* and contains
nothing to copy. **`HOW_TO_RUN.md` is the short path** — the nine commands from a
fresh copy to 30 finished runs; come here for the detail behind any of them.

Experiment: **E1** (`All_docs/Experiments.pdf` §8.2) on RedBlueDoors, three arms —
`gae` (plain PPO), `cf` (PPO-CF), `shuf` (action-permuted CF control).

---

## 0. Copy the code from Windows to the cluster

### One-time: an SSH config so you never type the jump host again

Create `%USERPROFILE%\.ssh\config` (Notepad is fine):

```
Host stujump
    HostName stujump.comp.nus.edu.sg
    User aniket

Host xlogin
    HostName xlogin.comp.nus.edu.sg
    User aniket
    ProxyJump stujump
```

Then `ssh xlogin` and `scp ... xlogin:~/...` both work directly.

### First copy

Windows `tar` (shipped with Windows 10+) makes an archive without the heavy
directories, which `scp` alone cannot exclude:

```cmd
cd C:\path\to\Personal Projects
tar --exclude=runs --exclude=figures --exclude=.venv --exclude=.git ^
    --exclude=to_delete --exclude=artifacts --exclude=__pycache__ ^
    -czf ppo_cf.tgz PPO_CF

scp ppo_cf.tgz xlogin:~/
ssh xlogin "mkdir -p ~/ppo_cf && tar xzf ~/ppo_cf.tgz -C ~/ppo_cf --strip-components=1 && rm ~/ppo_cf.tgz"
```

(In PowerShell use a backtick `` ` `` instead of `^` for line continuation, or put
it all on one line.)

### Updating a few files afterwards

```cmd
scp .\slurm\*.sbatch .\slurm\*.sh xlogin:~/ppo_cf/slurm/
scp .\pipeline\*.py                xlogin:~/ppo_cf/pipeline/
scp .\submit_e1.sh .\RUNBOOK.md    xlogin:~/ppo_cf/
```

### ALWAYS fix line endings after copying

Windows writes `\r\n`. `sbatch` refuses such a script outright —

```
sbatch: error: Batch script contains DOS line breaks (\r\n)
```

— and a **sourced** file (`_prelude.sh`, `_submit_lib.sh`, `env.sh`) is worse: bash
does not refuse it, it just takes the `\r` as part of the last word on every line
and fails somewhere confusing. Run this on the cluster after every copy:

```bash
cd ~/ppo_cf && bash slurm/fix_line_endings.sh
```

It converts only the project's own text files. Do **not** use
`find . -type f -exec dos2unix {} +` — that walks `.venv/` (thousands of package
files) and `runs/`, which is slow and pointless. The script falls back to `sed -i
's/\r$//'` when `dos2unix` is not installed.

If you use git, `.gitattributes` (already committed) pins `eol=lf` so a clone on
the cluster is correct without any of this.

---

## 1. Nothing runs on the login node

This is the rule the whole setup is built around, and it is enforced, not just
documented: **every job script exits immediately if it is not inside a Slurm job**
(the check is at the top of `slurm/_prelude.sh`, and repeated in the two scripts
that do not source it). Running `bash slurm/02_run_chunks.sbatch` on xlogin prints

```
REFUSING: not inside a Slurm job. Batch scripts are submitted, not run.
```

and stops.

| Runs on **xlogin** | Runs on a **compute node** |
|---|---|
| `./submit_e1.sh` — calls `sbatch` and exits | training, analysis, `pip install`, wandb upload |
| `sbatch slurm/*.sbatch` | everything else without exception |
| `squeue` / `sacct` / `scancel` / `tail` on a log | |
| `bash slurm/fix_line_endings.sh` — `sed` over ~60 small files | |

**You can close your SSH session.** `sbatch` hands the job to Slurm and returns;
the job does not belong to your shell. The whole chain — manifest → array →
analyse → requeue-or-next-stage — is carried by Slurm job dependencies and by
`03_analyse` calling `sbatch` from inside its own compute-node job, so it keeps
going with nobody logged in. You do not need `tmux`, `nohup` or `screen`.

**Do not use `salloc` / `srun` for setup.** An interactive allocation dies when
your SSH session drops, and `srun pip install ...` runs the *system* python, which
is why it failed with `externally-managed-environment`. The venv build is a batch
job: `sbatch slurm/00_create_venv.sbatch`.

---

## 2. Cluster settings (NUS SoC — already applied)

Measured on xlogin, 2026-09-08:

| Partition | MaxTime | Priority factor | Used for |
|---|---|---|---|
| `normal` (default) | **3:00:00** | 4 | venv, verify, manifest, analyse, wandb sync |
| `long` | 3-00:00:00 | 1 | **`02_run_chunks`** — the 12 h array |

The 8x8 CF units take ~5.5 h, so the array cannot run on `normal`; its header says
`--partition=long`. Everything else is well under 3 h and takes `normal`'s higher
priority factor. These are already set in the `#SBATCH` headers — nothing to edit.

One thing is still unknown and worth 10 seconds:

```bash
sacctmgr show assoc user=$USER format=account,maxjobs,maxsubmit
```

Put those two numbers into `MAX_SUBMIT` and `CONCURRENCY` at the top of
`slurm/_submit_lib.sh` (currently 32 and 16). Exceeding the running-job cap does
not get you more nodes; it gets submissions rejected or your priority reduced.

---

## 3. Build the environment

The venv lives **inside the project** at `~/ppo_cf/.venv`, not in `$HOME/.venvs`,
so the checkout carries its own interpreter: move or delete the project and the
environment goes with it, and two checkouts can never share one by accident.

```bash
cd ~/ppo_cf
mkdir -p logs                       # Slurm will not create it, and --output needs it
sbatch slurm/00_create_venv.sbatch
tail -f logs/venv_<jobid>.out
```

Read the **whole** log. Every import must pass. If several native packages fail
identically the venv itself is damaged — rerun this script (it uses `--clear`)
before investigating anything else.

Optional, for wandb — compute nodes here have outbound network, so a token in
`~/.netrc` is all that is needed and it is shared with every compute node:

```bash
source .venv/bin/activate && wandb login
grep -s api.wandb.ai ~/.netrc        # already logged in? then slurm/env.sh needs no key
```

To turn wandb off entirely: `export PPO_CF_WANDB=0` before submitting.
`scalars.csv` is unaffected either way.

---

## 4. Verify

```bash
sbatch slurm/00_verify.sbatch
cat logs/verify_<jobid>.out
```

Installs nothing. Checks, in order: no stray CRLF line endings; every repo module
imports; every stage config builds; the MiniGrid simulator restore is bit-exact on
both RedBlueDoors variants (the failure mode that yields plausible, *wrong*
`A_CF`); and one real PPO-CF update completes with finite losses.

---

## 5. Smoke the whole chain before the real thing

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

## 6. Run it

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

Measured from this repo's own `runs/*/summary.json`, scaled to 2M frames:

| stage | arm | frames | per seed | seeds | subtotal |
|---|---|---|---|---|---|
| `E1_RBD6` | gae | 2M | ~25 min | 5 | 2.1 h |
| `E1_RBD6` | cf | 2M | ~1.5 h | 5 | 7.5 h |
| `E1_RBD6` | shuf | 2M | ~1.5 h | 5 | 7.5 h |
| `E1_RBD8` | gae | 2M | ~25 min | 5 | 2.1 h |
| `E1_RBD8` | cf | 2M | ~5.5 h | 5 | 27.5 h |
| `E1_RBD8` | shuf | 2M | ~5.5 h | 5 | 27.5 h |

**30 units, ~74 core-hours, ~110 MB on disk.** 15 units per stage, so the default 16
chunks gives one unit per task. At 16-way concurrency, roughly 6–8 h wall clock per
stage. The 8x8 CF estimate assumes the doubled branch budget (`cf_horizon` 64 at
`cf_subsample` 0.05 = 6.4, against 3.2 in the committed YAML).

---

## 7. Watch it

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

## 8. When it finishes

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

`rsync` is not on stock Windows, and `scp` cannot exclude anything, so pack on the
cluster and pull one file. **`tar` here runs on the login node deliberately** — it is
a file copy, not compute, the same class of action as `scp` itself.

```bash
# on xlogin
cd ~/ppo_cf
tar --exclude='checkpoints' --exclude='*.npz' -czf ~/e1_results.tgz \
    artifacts runs/e1_rbd*/*/scalars.csv runs/e1_rbd*/*/episodes.csv \
    runs/e1_rbd*/*/config.json
ls -lh ~/e1_results.tgz          # expect a few tens of MB
```

```cmd
:: on Windows
scp xlogin:~/e1_results.tgz .
tar xzf e1_results.tgz
```

Checkpoints (~3 MB each) only for the seeds you actually want to probe:

```cmd
scp -r xlogin:~/ppo_cf/runs/e1_rbd6_cf/seed_0/checkpoints .\runs\e1_rbd6_cf\seed_0\
```

Then open the notebooks locally: everything they need is `artifacts/tables/*.csv`
plus each run's `scalars.csv` / `episodes.csv`.

### Upload the wandb runs

```bash
sbatch slurm/04_sync_wandb.sbatch
```

Compute nodes here have outbound network — `00_create_venv` pulls torch and wandb
from PyPI on one — so this is an ordinary batch job on `normal`. Safe to resubmit:
already-synced runs are skipped. If an upload fails, nothing is lost; the offline
directories stay on disk.

Do **not** run `wandb sync` on xlogin as a workaround. It is not needed, and a long
upload started in an SSH session dies with the session.

---

## 9. When something goes wrong

| Symptom | Do |
|---|---|
| manifest log says fewer than 3 groups | `PROJ_GROUPS` was mangled. Submit through `./submit_e1.sh`, never a hand-written `--export=ALL,PROJ_GROUPS=gae,cf,shuf` |
| analyse says `STOPPING ... ZERO completed` | a fault hitting every unit, not a walltime cut. `tail -40 logs/run_*_0.err`. Nothing is lost |
| `Segmentation fault` / `SIGILL` in a run log | `sbatch slurm/09_diagnose.sbatch` — one process per import names the culprit |
| a unit failed but others finished | normal. Its row is absent; the next pass picks it up. `grep -A20 'FAILED' logs/run_*.out` |
| `no venv at ...` | `sbatch slurm/00_create_venv.sbatch` (it builds `./.venv`) |
| `sbatch: error: Batch script contains DOS line breaks` | `bash slurm/fix_line_endings.sh`, then resubmit |
| `No such file or directory: 'requirements.txt'` in a job log | the job was submitted from somewhere other than the project root; `cd ~/ppo_cf` first |
| `REFUSING: not inside a Slurm job` | you ran a batch script instead of submitting it: `sbatch slurm/<script>.sbatch` |
| want to redo one unit | delete its row from `artifacts/tables/<STAGE>.csv` and its `<STAGE>_chunk_*.jsonl` line, then resubmit |
| every Eq (32) check says `n<5` | fewer than 5 paired seeds finished. A bootstrap over 1–2 pairs is not evidence, so the gate refuses a verdict rather than printing a flattering one |
| results look like a smoke run | `PPO_CF_SMOKE_FRAMES` was still exported. `unset` it; real and smoke never share a file |
| want to rerun the summary only | `export STAGE=E1_RBD6 PROJ_GROUPS="gae,cf,shuf" PROJ_N_GROUPS=3 N_CHUNKS=16 PART_END=E1_RBD8 PASS_NO=0` then `sbatch --export=ALL slurm/03_analyse.sbatch` |

---

## 10. Things this setup does NOT do, on purpose

- **No notebook execution on the cluster.** The notebooks currently train inside
  themselves; running them here would duplicate the array's work. `03_analyse` writes
  the ledger and the figures, and you analyse locally from those.
- **No trajectory datasets.** `trajectories.npz` is 25 MB/seed and feeds NB03+ (E2/E3
  distillation), not E1. Set `PPO_CF_RECORD_TRAJECTORIES=1` before submitting if you
  want them — it costs ~1.5 GB instead of ~110 MB.
- **No mid-run resume.** A unit is atomic. A killed unit is redone from zero, which is
  why the walltime is sized from the slowest one.
