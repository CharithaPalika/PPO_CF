#!/bin/bash
# Sourced at the top of every .sbatch. Never executed directly.
#
# Everything that must be true before any job of this project does work. One
# copy, so a fix lands everywhere at once.

set -euo pipefail

# ---- 1. project root -------------------------------------------------------
# Resolved from this file's own location, so the scripts work from any cwd and
# the path appears exactly once in the repository. NOTHING here is hard-coded
# to a machine: config/config.py derives PROJECT_ROOT the same way, so runs/,
# artifacts/ and logs/ follow the checkout wherever it lives.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p logs artifacts/{manifests,chunks,tables,figures,status} runs

# ---- 2. the project's OWN virtualenv ---------------------------------------
# Not a shared or lab-wide venv: installing these pins into a shared one moves
# a transitive dependency underneath somebody else's project and theirs moves
# one underneath yours. Override with PROJ_VENV=/some/path.
VENV="${PROJ_VENV:-$HOME/.venvs/ppo_cf}"
if [ ! -f "$VENV/bin/activate" ]; then
    echo "no venv at $VENV -- run: sbatch slurm/00_create_venv.sbatch" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# ---- 3. optional secrets ---------------------------------------------------
[ -f slurm/env.sh ] && source slurm/env.sh || true

# ---- 4. a compute node is not a workstation --------------------------------
export CUDA_VISIBLE_DEVICES=""                 # CPU project: no accidental CUDA init
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLBACKEND=Agg                          # no display; matplotlib must not look
export PYTHONUNBUFFERED=1                      # a killed job must not lose its last 4 KB

# Thread counts matter more than they look. The numeric libraries default to
# "one thread per core the NODE has", not per core Slurm gave you. On a 128-core
# node a 4-cpu task spawns 128 threads that fight over 4 cores and everything
# runs several times slower while looking busy.

# ---- 5. wandb ---------------------------------------------------------------
# OFFLINE on compute nodes, always. wandb starts a sidecar per run over a socket
# under an NFS home; sixteen array tasks starting in the same second race for it,
# and when that race is lost the run dies before its first step, records nothing,
# and the analyse job requeues the same work forever. Offline removes the race.
# Upload afterwards with slurm/04_sync_wandb.sbatch.
#
# scalars.csv is written first and unconditionally, so nothing here can lose data.
export PPO_CF_WANDB="${PPO_CF_WANDB:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-$PROJECT_ROOT/artifacts/wandb}"
mkdir -p "$WANDB_DIR"

# ---- 6. did the submitted variables survive the trip? ----------------------
# NOT called GROUPS. `GROUPS` is a bash BUILT-IN array holding the invoking
# user's group ids, so `export GROUPS="gae,cf,shuf"` assigns to element 0 of
# that array, bash does not export arrays, and the receiving job reads the
# builtin instead -- observed delivering the literal string "1046". The same
# class of silent corruption as the --export comma bug below, from a different
# direction.
# PROJ_GROUPS contains commas; PROJ_N_GROUPS is a plain integer that cannot be mangled the
# same way. If they disagree, the values were damaged in transit and the job
# must refuse rather than silently run a fraction of the experiment.
PROJ_GROUPS="${PROJ_GROUPS:-}"
if [ -n "$PROJ_GROUPS" ]; then
    N_GOT=$(awk -F, '{print NF}' <<< "$PROJ_GROUPS")
    echo "STAGE=${STAGE:-<unset>} N_CHUNKS=${N_CHUNKS:-<unset>}"
    echo "PROJ_GROUPS (${N_GOT}): ${PROJ_GROUPS}"
    if [ -n "${PROJ_N_GROUPS:-}" ] && [ "$N_GOT" -ne "$PROJ_N_GROUPS" ]; then
        echo >&2 "REFUSING: submitter meant ${PROJ_N_GROUPS} groups; this job received ${N_GOT}."
        echo >&2 "PROJ_GROUPS was mangled in transit -- almost certainly an"
        echo >&2 "  --export=ALL,PROJ_GROUPS=gae,cf,shuf call. Submit via slurm/_submit_lib.sh."
        exit 1
    fi
fi

echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?}   job=${SLURM_JOB_ID:-?}"
echo "root=$PROJECT_ROOT"
echo "start $(date -Is)"
