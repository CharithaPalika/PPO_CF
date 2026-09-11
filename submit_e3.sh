#!/bin/bash
# E3: uncertainty-driven CF-label allocation for distilled perturbation.
#
# Arms:
#   uniform      uniform 2% CF-label selection, the E2 distilled control
#   uncertainty top 2% by ensemble epistemic uncertainty
#   active      top 2% by uncertainty times policy leverage
#
# The pooled sweep covers Taxi, DoorKey-6x6, UnlockPickup and
# RedBlueDoors-6x6: 4 envs x 3 seeds x 3 arms = 36 runs. Beta is fixed at 0.75.
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p logs artifacts/{manifests,chunks,tables,figures,status} runs
command -v sbatch >/dev/null || { echo "sbatch not found -- run this on the login node" >&2; exit 1; }

source slurm/_submit_lib.sh
if [ "$#" -eq 0 ]; then
    set -- 12
fi
parse_chunks "$@" || exit 1

# E3 runs the same oracle-heavy label budget as E2 but only 36 total units.
export CONCURRENCY=12

export PPO_CF_WANDB=1
export WANDB_MODE=online
export PPO_CF_WANDB_RESUME=allow
export PPO_CF_WANDB_RUN_NAMESPACE="${PPO_CF_WANDB_RUN_NAMESPACE:-e3_active_query_v1}"
export PPO_CF_WANDB_PROJECT="${PPO_CF_WANDB_PROJECT:-ppo-cf}"

submit_stage "E3_ALL" "uniform,uncertainty,active" "$N_CHUNKS" "E3_ALL"
