# How to run E1 on the cluster

The short path: nine commands from a Windows laptop to 30 finished runs.
`RUNBOOK.md` is the reference — why each thing is the way it is, every watch and
recovery command, what to do when something breaks. This file is just the sequence.

Cluster: NUS SoC, `xlogin.comp.nus.edu.sg` via the `stujump` jump host.
Experiment: **E1** — PPO vs PPO-CF vs action-permuted CF, on RedBlueDoors 6x6 and
8x8, 5 paired seeds and 2M frames each. **30 runs, ~74 core-hours, ~110 MB.**

---

## 0. One-time: SSH config

Create `%USERPROFILE%\.ssh\config` on Windows so you never type the jump host again:

```
Host stujump
    HostName stujump.comp.nus.edu.sg
    User aniket

Host xlogin
    HostName xlogin.comp.nus.edu.sg
    User aniket
    ProxyJump stujump
```

Test: `ssh xlogin` should land you on the login node after two passphrase prompts.

---

## 1. Copy the code up  *(Windows)*

`scp` cannot exclude directories, so pack first. `tar` ships with Windows 10+.

```cmd
cd C:\path\to\Personal Projects

tar --exclude=runs --exclude=figures --exclude=.venv --exclude=.git --exclude=to_delete --exclude=artifacts --exclude=__pycache__ -czf ppo_cf.tgz PPO_CF

scp ppo_cf.tgz xlogin:~/
ssh xlogin "mkdir -p ~/ppo_cf && tar xzf ~/ppo_cf.tgz -C ~/ppo_cf --strip-components=1 && rm ~/ppo_cf.tgz"
```

Later, to push just the files you changed:

```cmd
scp .\slurm\*.sbatch .\slurm\*.sh xlogin:~\ppo_cf\slurm\
scp .\pipeline\*.py xlogin:~\ppo_cf\pipeline\
```

---

## 2. Fix line endings and permissions  *(on xlogin)*

Windows writes `\r\n`. `sbatch` refuses such a script outright; a **sourced** `.sh`
is not refused and fails somewhere that looks unrelated. `scp` also drops the
executable bit.

```bash
ssh xlogin
cd ~/ppo_cf
bash slurm/fix_line_endings.sh
chmod +x submit_e1.sh slurm/*.sh slurm/*.sbatch
mkdir -p logs
```

Run `fix_line_endings.sh` after **every** copy from Windows. Do not use
`find . -type f -exec dos2unix {} +` — it walks `.venv/` and `runs/`.

---

## 3. Build the environment  *(one Slurm job, ~15 min)*

```bash
sbatch slurm/00_create_venv.sbatch
squeue -u $USER                      # wait for it to disappear
cat logs/venv_<jobid>.out
```

The venv is built **inside the project** at `~/ppo_cf/.venv`. Read the whole log:
every import under "4. verify" must pass. If several native packages fail
identically the venv is damaged — just resubmit, it uses `--clear`.

Optional, for Weights & Biases (compute nodes here have network, so a token in
`~/.netrc` is enough and every node shares it):

```bash
source .venv/bin/activate && wandb login && deactivate
```

To skip wandb entirely: `export PPO_CF_WANDB=0` before submitting anything.

---

## 4. Set your job caps  *(30 seconds)*

```bash
sacctmgr show assoc user=$USER format=account,maxjobs,maxsubmit
nano slurm/_submit_lib.sh     # put those two numbers in MAX_SUBMIT / CONCURRENCY
```

Partitions are already set: `long` for the 12 h array (RedBlueDoors-8x8 CF units run
~5.5 h and `normal` caps at 3 h), `normal` for everything else.

---

## 5. Verify  *(one job, ~5 min)*

```bash
sbatch slurm/00_verify.sbatch
cat logs/verify_<jobid>.out
```

Installs nothing. Checks: no CRLF anywhere; every module imports; all six stage
configs build; the MiniGrid simulator restore is bit-exact on both RedBlueDoors
variants; and one real PPO-CF update finishes with finite losses. It must end with
`verify OK`.

---

## 6. Smoke the whole chain  *(~10 min, cheap)*

Shrinks every unit to a few updates so the full chain — manifest → array → merge →
requeue → summary — runs in minutes. Writes only into `runs/_smoke/` and
`artifacts/_smoke/`, so it can never touch a real result.

```bash
export PPO_CF_SMOKE_FRAMES=8192
./submit_e1.sh 6

squeue -u $USER
cat logs/manifest_<jobid>.out         # MUST say "3 groups: gae, cf, shuf"
```

Then interrupt it on purpose and resubmit — the only way to know the resume path
works is to break it once while the units are cheap:

```bash
scancel <arrayjobid>
./submit_e1.sh 6                      # finished units are skipped; only the rest rerun
```

When satisfied:

```bash
unset PPO_CF_SMOKE_FRAMES             # REQUIRED, or you queue 30 two-second jobs
```

---

## 7. Run it

```bash
cd ~/ppo_cf
./submit_e1.sh
```

That is the whole thing. It submits `E1_RBD6` (6x6) and chains into `E1_RBD8` (8x8)
automatically when the first stage finishes.

**Before you walk away, check one line:**

```bash
cat logs/manifest_<jobid>.out         # must say "3 groups: gae, cf, shuf"
```

That is the receipt for the two variable-mangling guards. If it says 1 group,
something ate the commas — stop and read `RUNBOOK.md` §9.

**You can now close SSH.** `sbatch` hands the jobs to Slurm and returns; the chain
continues because the analyse job calls `sbatch` from inside its own compute-node
job. Nothing depends on your shell — no `tmux`, no `nohup`.

---

## 8. Watch it  *(whenever you feel like it)*

```bash
ssh xlogin && cd ~/ppo_cf

squeue -u $USER                              # queued and running
squeue -u $USER --start                      # estimated start times
wc -l artifacts/chunks/E1_RBD6_chunk_*.jsonl # units finished so far
tail -f logs/run_<arrayjobid>_0.out          # follow one task
grep -l Traceback logs/run_*_*.err           # anything that raised
```

Expect roughly 6–8 h wall clock per stage at 16-way concurrency, longer if the
queue is busy.

---

## 9. Get the results  *(when it finishes)*

The analyse job writes, per stage: the ledger `artifacts/tables/<STAGE>.csv`, the
per-arm summary, the Eq (32) paired tests in `artifacts/status/<STAGE>.json`, and
`artifacts/figures/<STAGE>_success.png`.

```bash
# on xlogin
cd ~/ppo_cf
tar --exclude='checkpoints' --exclude='*.npz' -czf ~/e1_results.tgz \
    artifacts runs/e1_rbd*/*/scalars.csv runs/e1_rbd*/*/episodes.csv runs/e1_rbd*/*/config.json
ls -lh ~/e1_results.tgz

sbatch slurm/04_sync_wandb.sbatch      # upload the offline wandb runs
```

```cmd
:: on Windows
scp xlogin:~/e1_results.tgz .
tar xzf e1_results.tgz
```

Analysis happens locally from `artifacts/tables/*.csv` and each run's
`scalars.csv` / `episodes.csv`. Pull checkpoints only for seeds you want to probe:

```cmd
scp -r xlogin:~/ppo_cf/runs/e1_rbd6_cf/seed_0/checkpoints .\runs\e1_rbd6_cf\seed_0\
```

---

## The whole thing, once you have done it once

```bash
cd ~/ppo_cf
bash slurm/fix_line_endings.sh && chmod +x submit_e1.sh slurm/*.sh
sbatch slurm/00_create_venv.sbatch     # only after a fresh copy
sbatch slurm/00_verify.sbatch          # read the log before continuing
./submit_e1.sh
cat logs/manifest_<jobid>.out          # "3 groups: gae, cf, shuf"
```

---

## If something goes wrong

| Message | Fix |
|---|---|
| `sbatch: error: Batch script contains DOS line breaks` | `bash slurm/fix_line_endings.sh` |
| `REFUSING: not inside a Slurm job` | you ran a batch script instead of submitting it: `sbatch slurm/<script>.sbatch` |
| `No such file or directory: 'requirements.txt'` in a job log | submitted from the wrong directory: `cd ~/ppo_cf` first |
| `no venv at ...` | `sbatch slurm/00_create_venv.sbatch` |
| `Permission denied` on `./submit_e1.sh` | `chmod +x submit_e1.sh`, or just `bash submit_e1.sh` |
| manifest log says fewer than 3 groups | submit through `./submit_e1.sh`, never a hand-written `--export=ALL,PROJ_GROUPS=...` |
| analyse says `STOPPING ... ZERO completed` | a fault hitting every unit. `tail -40 logs/run_*_0.err`. Nothing is lost |
| Eq (32) checks all say `n<5` | fewer than 5 paired seeds finished; rerun the missing ones (resubmit — the ledger skips what is done) |
| `Segmentation fault` / `SIGILL` in a run log | `sbatch slurm/09_diagnose.sbatch` — one process per import names the culprit |

Anything not here is in `RUNBOOK.md` §9, with more context.

---

## Two things to decide before spending 74 core-hours

1. **`norm_adv: batch`** in every RedBlueDoors config contradicts Experiments.pdf
   §3.1 Remark 1 ("actor advantage normalization is disabled in every arm"). It is a
   one-line change now and a full rerun later.
2. **RedBlueDoors-8x8 barely learns.** Plain PPO reached 0.04 success at 2M frames,
   and the two CF seeds gave 0.72 and 0.00. `frames_to_50` / `frames_to_90` will be
   undefined for at least one arm, so the 8x8 result rests on success AUC against a
   baseline that does not work. The 8x8 config's own note suggests raising `n_steps`
   to 1024 before anything else.
