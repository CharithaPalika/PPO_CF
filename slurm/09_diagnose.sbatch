#!/bin/bash
#SBATCH --job-name=ppocf_diag
#SBATCH --output=logs/diag_%j.out
#SBATCH --error=logs/diag_%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
# Pin a node family to test the "is it this hardware?" hypothesis:
# #SBATCH --constraint=<feature>          (see: sinfo -o "%20N %10c %25f")

# For when something crashes NATIVELY. Numbered out of the way because you will
# need it rarely and urgently. It answers three questions the normal logs
# cannot: which import crashes, what the node actually is, and whether the
# crash reproduces on a different node family.

set -uo pipefail                    # not -e: collect every result, then report

# NOTHING RUNS ON THE LOGIN NODE. If this script is executed directly on xlogin
# instead of submitted, it exits here rather than loading the machine everyone
# shares -- and rather than dying the moment the SSH session drops.
if [ -z "${SLURM_JOB_ID:-}" ] && [ "${PPO_CF_ALLOW_LOGIN:-0}" != "1" ]; then
    echo "REFUSING: this is a batch script, not a command." >&2
    echo "  sbatch slurm/$(basename "$0")" >&2
    exit 1
fi

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"      # $0 is the spool copy under sbatch
cd "$PROJECT_ROOT" || exit 1
source "${PROJ_VENV:-$PROJECT_ROOT/.venv}/bin/activate"

echo "host=$(hostname) arch=$(uname -m) kernel=$(uname -r)"
echo "cpu flags: $(grep -m1 ^flags /proc/cpuinfo | tr ' ' '\n' | \
                   grep -E '^(avx|avx2|avx512f|sse4_2)$' | tr '\n' ' ')"
echo "glibc: $(ldd --version | head -1)"
echo "venv : $(which python)"
echo

# One process per import: a SIGSEGV kills only its own probe, so the log names
# the culprit.
for m in numpy pandas scipy sklearn matplotlib torch gymnasium minigrid wandb; do
    printf "%-14s " "$m"
    python -u -c "import $m; print('ok', getattr($m,'__version__','?'))" 2>&1 | tail -1
    echo "               exit=$?"
done

echo
echo "--- library resolution for the module that crashes ---"
python -c "import numpy, os; print(os.path.dirname(numpy.__file__))"
python -c "import torch, os; print(os.path.dirname(torch.__file__))"
# ldd on the offending .so says whether it is a MISSING system library or a
# DAMAGED one:
#   ldd $(python -c "import numpy,glob,os;
#         print(glob.glob(os.path.dirname(numpy.__file__)+'/_core/_multiarray_umath*.so')[0])")

echo
echo "READING A NATIVE CRASH"
echo "  ld.so ... Assertion failed  -> the .so files are damaged: rebuild with 00_create_venv (--clear)"
echo "  SIGILL  (exit 132)          -> wheel needs CPU instructions this node lacks: pin newer nodes"
echo "                                 with --constraint, or install a lower-baseline wheel"
echo "  SIGSEGV (exit 139)          -> native crash: the loop above names which import"
echo "  SIGKILL (exit 137)          -> OOM or walltime: sacct -j <id> --format=MaxRSS,Elapsed,State"
