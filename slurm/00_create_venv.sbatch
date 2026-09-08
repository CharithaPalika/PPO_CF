#!/bin/bash
#SBATCH --job-name=ppocf_venv
#SBATCH --output=logs/venv_%j.out
#SBATCH --error=logs/venv_%j.err
#SBATCH --time=00:50:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=normal

#   sbatch slurm/00_create_venv.sbatch
#   tail -f logs/venv_<jobid>.out
#
# On a COMPUTE node, because a large install is exactly the load a login node
# cannot absorb, and because the node that installs the wheels should be the
# same kind of node that imports them -- a wheel compiled for instructions the
# compute nodes lack fails at import with SIGILL, and you want that now.

# NOT `set -e`: a failed check here is data to report, not a reason to abort
# the report.
set -uo pipefail

# NOTHING RUNS ON THE LOGIN NODE. If this script is executed directly on xlogin
# instead of submitted, it exits here rather than loading the machine everyone
# shares -- and rather than dying the moment the SSH session drops.
if [ -z "${SLURM_JOB_ID:-}" ] && [ "${PPO_CF_ALLOW_LOGIN:-0}" != "1" ]; then
    echo "REFUSING: this is a batch script, not a command." >&2
    echo "  sbatch slurm/$(basename "$0")" >&2
    exit 1
fi

# $0 IS NOT THIS FILE under sbatch: the script is copied into the node's spool
# directory and the copy is executed, so `dirname "$0"` is /var/spool/slurmd/...
# and every relative path resolved from it lands outside the repository. That is
# what made `pip install -r requirements.txt` fail with "No such file".
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_ROOT" || exit 1
if [ ! -f requirements.txt ]; then
    echo "no requirements.txt under $PWD -- submit from the project root:" >&2
    echo "    cd ~/ppo_cf && sbatch slurm/00_create_venv.sbatch" >&2
    exit 1
fi
mkdir -p logs

# INSIDE the project, not $HOME/.venvs. One checkout carries its own interpreter,
# so moving or copying the project moves the environment with it, `rm -rf` of the
# project leaves nothing behind, and two checkouts cannot share a venv by accident.
VENV="${PROJ_VENV:-$PROJECT_ROOT/.venv}"
PYBIN="${PROJ_PYTHON:-python3}"

echo "=================== node ==================="
echo "host   : $(hostname)"
echo "os     : $(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME" || uname -sr)"
echo "arch   : $(uname -m)"
echo "glibc  : $(ldd --version | head -1)"
echo "python : $($PYBIN -V 2>&1)   ($(command -v $PYBIN))"
echo "target : $VENV"

echo; echo "=================== 1. create ==================="
# --clear so a rerun after a failed attempt starts from nothing rather than
# layering onto a half-built environment.
"$PYBIN" -m venv --clear "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip setuptools wheel 2>&1 | tail -2

echo; echo "============ 2. torch, FIRST and ALONE, CPU index ============"
# Pinned to the CPU index deliberately. If torch is resolved as a side effect
# of another requirement, pip takes the default CUDA wheel and drags in ~3 GB
# of nvidia-* packages this project cannot use (ppo.device is "cpu" in every
# config) plus a CUDA init path that can fail on a node with no driver.
pip install --index-url https://download.pytorch.org/whl/cpu torch 2>&1 | tail -3

echo; echo "=================== 3. everything else ==================="
pip install -r "$PROJECT_ROOT/requirements.txt" 2>&1 | tail -4
# wandb is a pipeline dependency, not a project one: nothing in agents/ or
# oracle/ imports it, and utils/wandb_sink.py degrades to CSV-only without it.
pip install wandb 2>&1 | tail -2

echo; echo "============ 4. verify, ONE PROCESS PER IMPORT ============"
# A SIGSEGV or a dynamic-linker assertion kills the interpreter outright. No
# `except` catches either, because neither is an exception. If every import
# shares one `python -c`, the first native crash ends the process and the log
# says "Segmentation fault" without naming which package caused it. One
# process each means the crash names the culprit and the loop continues.
export PYTHONUNBUFFERED=1
FAILED=""
check() {
    local label="$1" code="$2" rc out
    printf "  %-22s " "$label"
    out=$(mktemp); ( python -u -c "$code" ) >"$out" 2>&1; rc=$?
    tail -2 "$out"; rm -f "$out"
    case $rc in
        0)   return 0 ;;
        132) echo "      ^^ SIGILL: wheel wants CPU instructions this node lacks" ;;
        134) echo "      ^^ SIGABRT: a native library aborted" ;;
        137) echo "      ^^ SIGKILL: out of memory" ;;
        139) echo "      ^^ SIGSEGV: native crash" ;;
        *)   echo "      ^^ exit $rc" ;;
    esac
    FAILED="$FAILED $label"; return 1
}

check numpy      'import numpy; print(numpy.__version__)'
check pandas     'import pandas; print(pandas.__version__)'
check scipy      'import scipy; print(scipy.__version__)'
check sklearn    'import sklearn; print(sklearn.__version__)'
check torch      'import torch; print(torch.__version__, torch.get_num_threads())'
check matplotlib 'import matplotlib; matplotlib.use("Agg"); print(matplotlib.__version__)'
check gymnasium  'import gymnasium; print(gymnasium.__version__)'
check minigrid   'import minigrid; print(minigrid.__version__)'
check yaml       'import yaml; print(yaml.__version__)'
check wandb      'import wandb; print(wandb.__version__)'
check make_env   'import gymnasium, minigrid; e=gymnasium.make("MiniGrid-RedBlueDoors-8x8-v0"); e.reset(seed=0); print("env ok", e.observation_space["image"].shape)'

echo
if [ -n "$FAILED" ]; then
    echo "FAILED:$FAILED"
    echo "If several native packages fail identically, the venv itself is damaged --"
    echo "rerun this script (it uses --clear) before investigating anything else."
    exit 1
fi
echo "venv OK: $VENV"
echo "next: sbatch slurm/00_verify.sbatch"
