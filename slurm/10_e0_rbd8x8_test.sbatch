#!/bin/bash
#SBATCH --job-name=ppocf_e0test
#SBATCH --output=logs/e0test_%A_%a.out
#SBATCH --error=logs/e0test_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-1
#SBATCH --partition=long

# E0 -- RedBlueDoors-8x8, PPO-CF, seeds 0 and 2, the config as
# `01_ppo_baseline_redbluedoor.ipynb` ran it (redbluedoors8x8_cf.yaml, no
# overrides). One seed per array task, so both run in parallel.
#
#   sbatch slurm/10_e0_rbd8x8_test.sbatch
#   tail -f logs/e0test_<jobid>_0.out
#
# `long`, not `normal`: at branch budget 3.20 a seed measures ~2.75 h and
# `normal` caps at 3 h. 8 h leaves room for a slow node without asking for the
# 12 h the E1 array needs.

# $0 IS NOT THIS FILE under sbatch -- see slurm/README.md. SLURM_SUBMIT_DIR is
# the directory sbatch was called from, i.e. the project root.
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
if [ ! -f slurm/_prelude.sh ]; then
    echo "no slurm/_prelude.sh under $PWD -- submit from the project root:" >&2
    echo "    cd ~/ppo_cf && sbatch slurm/10_e0_rbd8x8_test.sbatch" >&2
    exit 1
fi

# ---- set BEFORE the prelude: it reads both with ${VAR:-default} ------------
# ONLINE, unlike E1. Only two tasks start here, so the sidecar race that forces
# E1 offline (slurm/README.md guard 2) cannot happen, and live curves are the
# point of a param sweep. scalars.csv is still written first and every wandb
# call is still wrapped, so a failed upload cannot fail a run.
export WANDB_MODE=online
export PPO_CF_WANDB=1
export PPO_CF_WANDB_PROJECT="${PPO_CF_WANDB_PROJECT:-ppo-cf-sweeps}"

# The YAML says record_trajectories: true and this stage is "exactly as it ran",
# so trajectories are ON here even though the E1 pipeline defaults them off.
# ~25 MB per seed.
export PPO_CF_RECORD_TRAJECTORIES=1

source slurm/_prelude.sh

SEEDS=(0 2)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
echo "E0_reddoorbluedoor_test  arm=cf  seed=${SEED}  wandb=${WANDB_MODE} project=${PPO_CF_WANDB_PROJECT}"

python -m scripts.run_e0_rbd8x8_test --seeds "${SEED}" --arm cf

echo "end $(date -Is)"
