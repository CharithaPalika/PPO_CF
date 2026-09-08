#!/bin/bash
#SBATCH --job-name=ppocf_wbsync
#SBATCH --output=logs/wbsync_%j.out
#SBATCH --error=logs/wbsync_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
# #SBATCH --partition=<your_partition>

# Uploads the offline wandb runs written by the array tasks. Separate from
# training on purpose: an upload that hangs or fails must not be able to cost a
# training slot, and this can be rerun any number of times (already-synced runs
# are skipped).
#
#   sbatch slurm/04_sync_wandb.sbatch
#
# IF YOUR COMPUTE NODES HAVE NO OUTBOUND NETWORK this job will fail with a
# connection error and nothing is lost -- the offline directories stay on disk.
# Run the same command from wherever you do have network:
#     source ~/.venvs/ppo_cf/bin/activate
#     wandb sync artifacts/wandb/offline-run-*
# That is a file transfer, not compute, which is why it is the one thing in
# this project allowed off a compute node.

source "$(dirname "$0")/_prelude.sh"

shopt -s nullglob
runs=(artifacts/wandb/offline-run-*)
if [ ${#runs[@]} -eq 0 ]; then
    echo "no offline runs under artifacts/wandb/ -- nothing to sync"
    exit 0
fi
echo "${#runs[@]} offline run(s) to sync"

# Never fail the job on one bad upload.
set +e
for r in "${runs[@]}"; do
    echo "--- $r"
    WANDB_MODE=online wandb sync "$r" 2>&1 | tail -3
done
echo "sync pass complete (re-run safely; synced runs are skipped)"
exit 0
