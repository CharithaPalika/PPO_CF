#!/bin/bash
# Submit the editable matrix in slurm/all_experiment_config.yaml.
#
# The YAML decides which environments, seeds, and algorithms run. This wrapper
# only chooses the Slurm chunk count and passes the supported internal groups to
# the shared manifest/chunk/analyse pipeline.
set -euo pipefail
cd "$(dirname "$0")"

CONFIG_FILE="slurm/all_experiment_config.yaml"
[ -f "$CONFIG_FILE" ] || { echo "missing $CONFIG_FILE" >&2; exit 1; }

mkdir -p logs artifacts/{manifests,chunks,tables,figures,status} runs
command -v sbatch >/dev/null || { echo "sbatch not found -- run this on the login node" >&2; exit 1; }

source slurm/_submit_lib.sh
if [ "$#" -eq 0 ]; then
    set -- 16
fi
parse_chunks "$@" || exit 1

export CONCURRENCY="${PPO_CF_CONCURRENCY:-12}"

export PPO_CF_WANDB=1
export WANDB_MODE=online
export PPO_CF_WANDB_RESUME=allow
export PPO_CF_WANDB_RUN_NAMESPACE="${PPO_CF_WANDB_RUN_NAMESPACE:-all_experiment_config_v1}"
export PPO_CF_WANDB_PROJECT="${PPO_CF_WANDB_PROJECT:-ppo-cf}"

submit_stage \
    "ALL_EXPERIMENT_CONFIG" \
    "gae,cf,queried,distill,uncertainty,active" \
    "$N_CHUNKS" \
    "ALL_EXPERIMENT_CONFIG"
