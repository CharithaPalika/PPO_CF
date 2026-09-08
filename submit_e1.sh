#!/bin/bash
# E1 (Experiments.pdf section 8.2) on RedBlueDoors.
#
#   ./submit_e1.sh          # 16 chunks (default)
#   ./submit_e1.sh 8        # 8 chunks
#
# Submits E1_RBD6 (RedBlueDoors-6x6) and chains automatically into E1_RBD8
# (RedBlueDoors-8x8) when the first stage completes. Both stages: 2M frames,
# seeds 0-4, three arms each (gae / cf / shuf) -- 15 units per stage.
#
# PART_END (the last argument to submit_stage) is the safety catch: the analyse
# job chains to the next stage automatically and PART_END is where it stops.
# Set it to E1_RBD6 to run 6x6 only and look at the result before paying for
# 8x8 -- which is the whole point of having stages.
#
# Submits and exits. Nothing computes here: this runs on the login node.
set -euo pipefail
cd "$(dirname "$0")"
source slurm/_submit_lib.sh

parse_chunks "$@" || exit 1
submit_stage "E1_RBD6" "$(default_groups)" "$N_CHUNKS" "E1_RBD8"
