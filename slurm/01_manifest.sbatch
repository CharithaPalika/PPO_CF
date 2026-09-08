#!/bin/bash
#SBATCH --job-name=ppocf_manifest
#SBATCH --output=logs/manifest_%j.out
#SBATCH --error=logs/manifest_%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --partition=normal

# Writes artifacts/manifests/<STAGE>.json: the flat, ordered list of every unit
# this stage still needs, minus anything already in its ledger. Pure
# bookkeeping -- and still on a compute node, because nothing runs on the login
# node.
#
# Its own job, rather than logic inside each array task, so that all N tasks
# partition ONE agreed list.
#
# Submitted by slurm/_submit_lib.sh, which exports the variables and passes
# --export=ALL alone. Never hand-write `sbatch --export=ALL,PROJ_GROUPS=gae,cf,shuf`
# -- sbatch splits that on its own commas and delivers PROJ_GROUPS=gae.

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

python pipeline/manifest.py \
    --stage    "${STAGE}" \
    --groups   "${PROJ_GROUPS}" \
    --n_chunks "${N_CHUNKS:-16}"
