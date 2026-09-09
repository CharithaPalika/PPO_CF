#!/bin/bash
# Optional. Sourced by _prelude.sh if present, ignored if not. Nothing may
# depend on this file existing, or a fresh clone will not run.
#
# The ONLY thing that belongs here is a credential, because it is the one
# setting that cannot live in the repository. Paths, thread counts and backends
# are set in _prelude.sh.
#
# GITIGNORED. Never `cat` this file into a log or a chat window.

# Not needed if you have already run `wandb login` on this cluster: the token
# lands in ~/.netrc and compute nodes share your home directory. Check with
#     grep -s api.wandb.ai ~/.netrc
# export WANDB_API_KEY="..."

# `:-` so a caller can pick a different project without editing this file --
# slurm/10_e0_rbd8x8_test.sbatch sends its sweeps to "ppo-cf-sweeps".
export PPO_CF_WANDB_PROJECT="${PPO_CF_WANDB_PROJECT:-ppo-cf}"
export WANDB_SILENT="true"           # compute nodes have no interactive terminal
export WANDB__SERVICE_WAIT="300"     # a slow upload must not hold a training job open
