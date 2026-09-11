# `slurm/` — why each decision is what it is

No commands here. Everything you type lives in `RUNBOOK.md`, in one place, because
a command that appears in two documents will be wrong in one of them within a month.

## The one rule

**Nothing runs on the login node.** Not training, not analysis, not `pip install`.
Submitting a job is a control-plane action; everything else is compute and belongs
inside an `.sbatch`.

This is **enforced, not just documented**. `_prelude.sh` exits immediately when
`SLURM_JOB_ID` is unset, and the two scripts that do not source it carry the same
check, so `bash slurm/02_run_chunks.sbatch` on xlogin prints `REFUSING: not inside a
Slurm job` and stops. `PPO_CF_ALLOW_LOGIN=1` is the escape hatch. The only things
that belong on the login node are `sbatch`, `squeue`/`sacct`/`scancel`, `tail` on a
log, and `fix_line_endings.sh` (a `sed` over ~60 small files — the same order of work
as `ls`, and a prerequisite for submitting anything at all).

A consequence worth stating: **the SSH session can be closed.** `sbatch` hands the
job to Slurm and returns, and the chain continues because `03_analyse` calls `sbatch`
from inside its own compute-node job. No `tmux`, no `nohup`.

`wandb sync` also runs as a batch job here: compute nodes on this cluster have
outbound network — `00_create_venv` pulls torch and wandb from PyPI on one.

## The shape that rule forces

```
./submit_e1.sh  →  01_manifest  →  02_run_chunks         →  03_analyse
 (login node,      builds the       array 0..N-1%K            merge chunks
  sbatch only)     agreed run       one contiguous slice      │
                   list for one     each, one JSONL file      ├─ units left? resubmit all three
                   stage            each                      └─ complete?  summarise, then
                                                                            submit the next stage
```

Three properties, and every other decision serves one of them.

| Property | What it buys |
|---|---|
| **One agreed list** | All array tasks partition the same manifest, written once by one job. If each task enumerated the work itself, a result landing between two reads shifts every task's slice: two tasks claim one unit, or the last is claimed by nobody and the stage never finishes. |
| **Append-only ledger** | Every finished unit appends one self-describing JSONL row. Work is redone only if its row is absent — which makes a walltime cut, a dead node and a cancelled submission all cost the same thing: the units in flight. |
| **The unit is atomic** | One unit = one `(stage, arm, seed)` training run. `PPOTrainer` has no mid-run resume, so the array walltime is sized from the *slowest* unit (RedBlueDoors-8x8 CF, ~5.5 h), not the average. |

## What this experiment is

E1 from `All_docs/Experiments.pdf` §8.2, on RedBlueDoors only. Three arms differing
in exactly one config field, `ppo.pg_mode`:

| arm | `pg_mode` | what it is |
|---|---|---|
| `gae` | `gae` | plain PPO |
| `cf` | `cf_all_action` | PPO-CF, full direct oracle |
| `shuf` | `cf_shuffled` | the same oracle vectors on *different* states (Eq 29) |

The required ordering is **both** `cf > gae` **and** `cf > shuf` (Eq 32). The second
half is why `shuf` is a first-class arm and not an afterthought: on DoorKey-6x6 at
n=1 the permuted control reached 90% success *first*. If it matches the true CF arm
again here, the benefit was never state-local counterfactual credit and must not be
described as such.

## Stages

| stage | env config | frames | seeds | units |
|---|---|---|---|---|
| `E1_RBD6` | `redbluedoors6x6_cf.yaml` | 2M | 0–4 | 15 |
| `E1_RBD8` | `redbluedoors8x8_cf.yaml` | 2M | 0–4 | 15 |

Seeds and frame budget are matched across the two stages on purpose, so the only
thing that differs is the environment. Note `redbluedoors6x6_cf.yaml`'s own budget is
1M (rl-baselines3-zoo's entry for that env) and 2M is a deliberate override.

`E1_RBD6` chains into `E1_RBD8` automatically; `PART_END` is where it stops.

Only three things are overridden per stage, all in `pipeline/stages.py` and nowhere
else: `ppo.total_timesteps: 2_000_000` (matching the two stages), `ppo.cf_horizon: 64`
on 8x8 (the 6x6 sweep measured coverage 0.002 at H=32, i.e. the degenerate
critic-difference estimator), and `run.log_every_updates: 2` (the default 10 writes
one scalars row every 82k frames, which would quantise a "frames to 50%" statistic
whose gaps of interest are ~40k). Everything else comes from the YAML as committed.

## What does not go in here

- **No experiment logic.** If an `.sbatch` file knows what a hyperparameter is, that
  knowledge is in the wrong file. These scripts know: which venv, which resources,
  which Python entry point.
- **No secrets in tracked files.** `slurm/env.sh` holds credentials and is in
  `.gitignore`.
- **No output.** Logs to `logs/`, results to `runs/` and `artifacts/`.
- **No absolute paths.** `_prelude.sh` resolves `PROJECT_ROOT` from its own location
  and `config/config.py` does the same, so the checkout runs wherever it lands.

## The guards, and what each one guards against

**1. `sbatch --export` splits on commas, and there is no escaping.**
`--export=ALL,STAGE=E1_RBD6,PROJ_GROUPS=gae,cf,shuf` is read as six items; `PROJ_GROUPS`
arrives as `gae`, and the bare words `cf` and `shuf` are read as names of variables
to propagate. They do not exist, so they vanish silently, the stage runs one arm out
of three, and it **exits 0**. Guard, in three parts: `_submit_lib.sh` exports in the
submitting shell and passes `--export=ALL` alone; `PROJ_N_GROUPS` travels alongside as a
plain integer that cannot be mangled the same way; `_prelude.sh` refuses to run if
the two disagree. `01_manifest` prints the arm count it actually saw — that line is
the human-readable receipt.

**1b. `GROUPS` is a bash built-in.** It holds the invoking user's group ids, so
`export GROUPS="gae,cf,shuf"` assigns to element 0 of that array, bash does not export
arrays at all, and the receiving job reads the builtin instead — observed delivering the
literal string `1046`. Same silent corruption as guard 1, from the other direction, and
the reason the variable is `PROJ_GROUPS` everywhere.

**1c. `$0` is not the script under `sbatch`.** Slurm copies the batch script into
the node's spool directory and executes the copy, so inside a job `$0` is
`/var/spool/slurmd/job<N>/slurm_script`. Every path resolved from `dirname "$0"`
then lands outside the repository — which is how `pip install -r requirements.txt`
came back with "No such file or directory" while the job's own `--output=logs/...`
worked fine (Slurm resolves that one against the submission directory). Every
`.sbatch` now starts from `${SLURM_SUBMIT_DIR:-$PWD}` and sources `slurm/_prelude.sh`
by relative path; `_prelude.sh` then resolves `PROJECT_ROOT` from `BASH_SOURCE[0]`,
which *is* its real path because it was sourced from the real file.

**2. Telemetry must never be able to fail an experiment.** wandb starts a sidecar per
run over a socket under an NFS home. Sixteen array tasks starting in the same second
race for it, and a lost race kills the run before its first step — which, because
nothing was recorded, the analyse job sees as pending and requeues, forever. Guard:
`scalars.csv` is written first and unconditionally; `WANDB_MODE=offline` in every job
removes the race rather than tolerating it; `utils/wandb_sink.py` swallows every
exception it can produce; `STALL_LIMIT` is the backstop.

**3. A stalled stage is not a slow stage.** Requeueing is correct for walltime cuts
and dead nodes, where each pass finishes some work. It is wrong for a fault that hits
every unit identically, and it will happily retry that fault forty times. Two
consecutive passes that completed nothing → stop and print what to read.

**4. Positional CSV appends corrupt data silently.** Only the CF arms emit `cf_*`
diagnostics, so rows do not share columns. Chunks are JSON Lines; the merge aligns by
column *name*; the ledger is rewritten whole, never appended to positionally. This is
the guard on this list that produces a wrong number in a paper rather than a failed job.

**5. Walltime margin sized from the slowest unit.** A task that starts a 5.5 h unit
with 3 h left loses both the unit and the slot. `chunk.py` stops *starting* work at
`--minutes 690` minus `--margin_min 360`.

## Smoke mode

`PPO_CF_SMOKE_FRAMES=8192 ./submit_e1.sh 6` shrinks every unit to a few updates and
redirects **all** output into `runs/_smoke/` and `artifacts/_smoke/`. One environment
variable moves one path (`ART` in `pipeline/stages.py`) and one run-name prefix, so
there is no code path where a smoke pass can write a row into a real ledger.

## Environment and partitions

The virtualenv lives at **`<project>/.venv`**, not `$HOME/.venvs`. A checkout then
carries its own interpreter: moving the project moves the environment with it,
deleting the project leaves nothing behind, and two checkouts cannot share a venv by
accident. `PROJ_VENV=/some/path` overrides it.

Partitions are set in the `#SBATCH` headers from the measured limits on NUS SoC:

| Partition | MaxTime | Priority factor | Jobs |
|---|---|---|---|
| `normal` | 3:00:00 | 4 | venv, verify, manifest, analyse, wandb sync |
| `long` | 3-00:00:00 | 1 | `02_run_chunks` — 12 h array, units up to ~5.5 h |

The array cannot run on `normal`: a single RedBlueDoors-8x8 CF unit is longer than
that partition's entire cap. Everything else is well inside 3 h and takes `normal`'s
higher priority factor.

## Line endings

The repo is copied from Windows, so CRLF is a recurring hazard. `sbatch` refuses a
CRLF *script* outright, which is the friendly case; a **sourced** file
(`_prelude.sh`, `_submit_lib.sh`, `env.sh`) is not refused by anything — the `\r`
just becomes part of the last word on every line and fails somewhere that looks
unrelated. Three layers: `.gitattributes` pins `eol=lf`, `fix_line_endings.sh`
repairs a copy, and `00_verify.sbatch` section 0 refuses to pass with any CRLF left.

## Files

| file | what it is |
|---|---|
| `env.sh` | credentials. **gitignored**, never `cat`'d |
| `fix_line_endings.sh` | CRLF → LF over the project's own text files, after a copy from Windows |
| `_prelude.sh` | sourced by every `.sbatch`: the login-node guard, root, venv, threads, headless backend, wandb mode, the `PROJ_GROUPS` guard |
| `_submit_lib.sh` | sourced by `experiment_1_6_envs`, `submit_e2.sh`, and `submit_e3.sh`: how a stage is submitted. **Your account's job caps go at the top of this file.** |
| `00_create_venv.sbatch` | builds the project-owned venv on a compute node |
| `00_verify.sbatch` | installs nothing; proves the venv, the configs, the simulator restore, and real PPO-CF/E2/E3 updates |
| `01_manifest.sbatch` | builds the run list for one stage |
| `02_run_chunks.sbatch` | the array |
| `03_analyse.sbatch` | merge → requeue, or summarise and chain on |
| `04_sync_wandb.sbatch` | uploads the offline wandb runs; safe to rerun |
| `09_diagnose.sbatch` | one process per import, for when something crashes natively |
