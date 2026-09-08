#!/bin/bash
#SBATCH --job-name=ppocf_run
#SBATCH --output=logs/run_%A_%a.out
#SBATCH --error=logs/run_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-15%16
#SBATCH --partition=long

# Runs one contiguous slice of artifacts/manifests/<STAGE>.json. One unit of
# work = one (stage, arm, seed) training run.
#
# THE WALLTIME. 12 h is sized from the SLOWEST unit, not the average:
# RedBlueDoors-8x8 with pg_mode=cf_all_action at cf_horizon=64 / subsample 0.05
# is a branch budget of 6.4 and measures ~5.5 h/seed. Two of those fit in one
# slot. Look up your cluster's priority bands before raising it -- many
# clusters give a large bonus below a threshold, and because the analyse job
# requeues whatever is unfinished, the walltime limits how long one task holds
# a node, not how much work the stage can do.
#
# THE %K THROTTLE. --array=0-15%16 submits 16 tasks and runs at most 16 at
# once. Set K to your account's concurrent-job cap in slurm/_submit_lib.sh, not
# above it: exceeding it does not get you more nodes, it gets your submissions
# rejected or your priority reduced. The --array line here is overridden by
# _submit_lib.sh's --array flag; it is kept so a bare
# `sbatch slurm/02_run_chunks.sbatch` still does something sane.
#
# CPUS. EnvPool steps its 16 envs in one process, so this is a single-process
# job; the cpus are for torch's numeric libraries (batch 8192, shared CNN).
# 4 is measured-adequate. Check MaxRSS with sacct before changing --mem.

# $0 IS NOT THIS FILE. sbatch copies the script into the node's spool directory
# and runs the copy, so under Slurm $0 is /var/spool/slurmd/job<N>/slurm_script
# and `dirname "$0"` points at the spool, not the repository. SLURM_SUBMIT_DIR is
# the directory sbatch was called from, which is the project root by contract.
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
if [ ! -f slurm/_prelude.sh ]; then
    echo "no slurm/_prelude.sh under $PWD -- submit from the project root:" >&2
    echo "    cd ~/ppo_cf && sbatch slurm/<script>.sbatch" >&2
    exit 1
fi
source slurm/_prelude.sh

echo "stage=${STAGE} chunk=${SLURM_ARRAY_TASK_ID}/${N_CHUNKS:-16}"

# --minutes against the wall with a margin: 690 of 720. The MARGIN is sized
# from the slowest unit (~5.5 h = 330 min, rounded to 360), not the average --
# its purpose is that a task never starts something it cannot finish and
# record. A unit killed mid-way leaves no ledger row, so it is simply redone;
# but it also wasted the whole slot it was killed in.
python pipeline/chunk.py \
    --stage    "${STAGE}" \
    --chunk_id "${SLURM_ARRAY_TASK_ID}" \
    --n_chunks "${N_CHUNKS:-16}" \
    --minutes 690 --margin_min 360

echo "end $(date -Is)"
