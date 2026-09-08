#!/bin/bash
#SBATCH --job-name=ppocf_analyse
#SBATCH --output=logs/analyse_%j.out
#SBATCH --error=logs/analyse_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=normal

# Merges the chunk files into the stage ledger, then does one of two things:
#
#   work remains   -> resubmit manifest + array + this job, and exit. That is
#                     what makes the pipeline survive a walltime cut, a dead
#                     node or a bad seed: the ledger only grows, so nothing is
#                     redone and nothing is lost.
#   stage complete -> write the per-arm summary table, the Eq (32) paired
#                     tests and the success-curve figure, then submit the next
#                     stage if this part has one.
#
# More memory than the array tasks: it holds every unit's episodes.csv to build
# the figure.
#
# To rerun this job alone (after fixing the summary, say):
#   export STAGE=E1_RBD6 PROJ_GROUPS="gae,cf,shuf" PROJ_N_GROUPS=3 N_CHUNKS=16 PART_END=E1_RBD8 PASS_NO=0
#   sbatch --export=ALL slurm/03_analyse.sbatch
# NOT `sbatch --export=ALL,PROJ_GROUPS=gae,cf,shuf ...` -- sbatch splits on its own commas.

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

python pipeline/analyse.py \
    --stage    "${STAGE}" \
    --groups   "${PROJ_GROUPS}" \
    --n_chunks "${N_CHUNKS:-16}" \
    --part_end "${PART_END:-}" \
    --pass_no  "${PASS_NO:-0}"
