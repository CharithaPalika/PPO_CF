#!/bin/bash
#SBATCH --job-name=ppocf_manifest
#SBATCH --output=logs/manifest_%j.out
#SBATCH --error=logs/manifest_%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
# #SBATCH --partition=<your_partition>

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

source "$(dirname "$0")/_prelude.sh"

python pipeline/manifest.py \
    --stage    "${STAGE}" \
    --groups   "${PROJ_GROUPS}" \
    --n_chunks "${N_CHUNKS:-16}"
