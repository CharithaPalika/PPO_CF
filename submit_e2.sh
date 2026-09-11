#!/bin/bash
# E2: uniform 2%-budget CF advantage perturbation.
#
# Arms: exact teacher perturbation at queried states and a student perturbation
# at every state. The pooled sweep covers Taxi, DoorKey-6x6, UnlockPickup and
# RedBlueDoors-6x6: 4 envs x 5 seeds x 3 betas x 2 arms = 120 runs.
#
# Default: 20 preassigned chunks x six units (three matched beta pairs). The
# array is throttled to 12 concurrent runners, so a completed chunk immediately
# makes room for the next queued chunk. The global manifest staggers fast and
# slow environments.
#
# This launcher deliberately differs from E1 only in telemetry: E2 publishes
# live W&B curves.  The variables are exported before submission, so they are
# inherited by the initial manifest/array/analyse jobs *and* by the analyse
# job's automatic requeues and next-stage submissions.
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p logs artifacts/{manifests,chunks,tables,figures,status} runs
command -v sbatch >/dev/null || { echo "sbatch not found -- run this on the login node" >&2; exit 1; }

source slurm/_submit_lib.sh
if [ "$#" -eq 0 ]; then
    set -- 20
fi
parse_chunks "$@" || exit 1

# E2 is intentionally capped at 12 workers even though E1's global default is
# 16: each worker runs three oracle-heavy matched beta pairs, and 12 concurrent
# processes is the requested cluster footprint. This value is inherited by
# automatic requeues as well.
export CONCURRENCY=12

# E2 needs live monitoring.  `PPO_CF_WANDB_RESUME=allow` combines retries of
# the same atomic (stage, arm, seed) unit into one W&B run; the pipeline ledger
# still decides which completed units must not be rerun.
export PPO_CF_WANDB=1
export WANDB_MODE=online
export PPO_CF_WANDB_RESUME=allow
# v2 avoids the deterministic IDs of the aborted/deleted E2 attempt. Keep this
# value unchanged during this sweep: retries then resume the same W&B run.
export PPO_CF_WANDB_RUN_NAMESPACE="${PPO_CF_WANDB_RUN_NAMESPACE:-e2_beta_sweep_v2}"
export PPO_CF_WANDB_PROJECT="${PPO_CF_WANDB_PROJECT:-ppo-cf}"

submit_stage "E2_ALL" "queried,distill" "$N_CHUNKS" "E2_ALL"
