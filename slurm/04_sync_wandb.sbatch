#!/bin/bash
#SBATCH --job-name=ppocf_wbsync
#SBATCH --output=logs/wbsync_%j.out
#SBATCH --error=logs/wbsync_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --partition=normal

# Uploads the offline wandb runs written by the array tasks. Separate from
# training on purpose: an upload that hangs or fails must not be able to cost a
# training slot, and this can be rerun any number of times (already-synced runs
# are skipped).
#
#   sbatch slurm/04_sync_wandb.sbatch
#
# Compute nodes on this cluster DO have outbound network -- 00_create_venv pulls
# torch and wandb from PyPI on one -- so this runs as an ordinary batch job like
# everything else. If an upload fails, nothing is lost: the offline directories
# stay on disk and the job is safe to resubmit (synced runs are skipped).
#
# Do NOT run `wandb sync` on xlogin as a workaround. It is not needed here, and
# a long upload started in an SSH session dies with the session.

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
