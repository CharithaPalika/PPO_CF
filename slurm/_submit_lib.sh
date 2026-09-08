#!/bin/bash
# Sourced by submit_*.sh at the repo root. Never executed directly.
#
# The one place that knows how a stage is submitted, so three parts of the
# project cannot drift into submitting the same pipeline three slightly
# different ways.

# ===========================================================================
# CLUSTER SETTINGS. Measured on NUS SoC (xlogin) 2026-09-08:
#
#   sinfo -s                -> normal: MaxTime 3:00:00  (PriorityJobFactor 4)
#                              long:   MaxTime 3-00:00:00 (PriorityJobFactor 1)
#   scontrol show partition -> normal caps at 3 h, which is why the 12 h array
#                              job runs on `long` and every short job on `normal`
#
# The partitions live in the #SBATCH headers (02_run_chunks -> long, everything
# else -> normal). Only the job caps are here.
#
# STILL TO CONFIRM on your account -- run this and correct the two numbers:
#     sacctmgr show assoc user=$USER format=account,maxjobs,maxsubmit
# ===========================================================================
MAX_SUBMIT=32          # your account's submitted-job cap
CONCURRENCY=16         # your account's RUNNING-job cap -> the "%K" array throttle
export CONCURRENCY     # pipeline/analyse.py reads this when it requeues
# ===========================================================================

default_groups() { echo "gae,cf,shuf"; }

# Parse and validate IN THE CALLER'S SHELL -- not in a $(...) helper.
#
# A helper called as N=$(parse "$@") runs in a subshell, so its `exit 1` on a
# bad value kills only the subshell and the caller carries on and submits
# anyway. A guard that does not guard is worse than no guard.
parse_chunks() {
    N_CHUNKS=16
    for arg in "$@"; do
        if [[ "$arg" =~ ^[0-9]+$ ]]; then
            N_CHUNKS="$arg"
        else
            echo "Unrecognized argument: $arg (expected a chunk count)" >&2
            return 1
        fi
    done
    [ "$N_CHUNKS" -ge 1 ] || { echo "N_CHUNKS must be >= 1" >&2; return 1; }
    if [ "$((N_CHUNKS + 2))" -gt "$MAX_SUBMIT" ]; then
        echo "Refusing: N_CHUNKS=$N_CHUNKS plus the manifest and analyse jobs exceeds the" \
             "${MAX_SUBMIT}-job submission cap." >&2
        return 1
    fi
}

submit_stage() {
    local stage="$1" groups="$2" n_chunks="$3" part_end="$4"
    local last=$((n_chunks - 1))

    # THE VARIABLES ARE EXPORTED HERE, NOT LISTED INSIDE --export=...
    #
    # sbatch's --export takes a COMMA-SEPARATED list of NAME=VALUE pairs and
    # offers no way to escape a comma inside a value, so
    #     --export=ALL,STAGE=E1_RBD6,PROJ_GROUPS=gae,cf,shuf,N_CHUNKS=16
    # is read as SIX items: ALL | STAGE=E1_RBD6 | PROJ_GROUPS=gae | cf | shuf |
    # N_CHUNKS=16. `cf` and `shuf` are taken as names of variables to
    # propagate; they do not exist, so they vanish without a warning, PROJ_GROUPS
    # arrives as "gae", and the stage runs one arm out of three and exits 0.
    #
    # --export=ALL on its own propagates this shell's whole environment,
    # commas intact.
    export STAGE="$stage" PROJ_GROUPS="$groups" N_CHUNKS="$n_chunks" PART_END="$part_end" PASS_NO=0
    PROJ_N_GROUPS=$(awk -F, '{print NF}' <<< "$groups"); export PROJ_N_GROUPS

    echo "stage=${stage}   chunks=${n_chunks}   groups(${PROJ_N_GROUPS})=${groups}"

    echo "1/3 manifest..."
    local mjob; mjob=$(sbatch --parsable --export=ALL slurm/01_manifest.sbatch)
    echo "    -> job $mjob"

    echo "2/3 ${n_chunks}-task array (0-${last}%${CONCURRENCY}, after manifest)..."
    local rjob; rjob=$(sbatch --parsable \
        --dependency=afterok:${mjob} \
        --array=0-${last}%${CONCURRENCY} \
        --export=ALL \
        slurm/02_run_chunks.sbatch)
    echo "    -> job $rjob"

    # afterany, NOT afterok. A task that hit the walltime or died on one bad
    # unit still left good rows behind, and noticing what is left is this job's
    # entire purpose.
    echo "3/3 analyse (after every task finishes, however it finishes)..."
    local ajob; ajob=$(sbatch --parsable \
        --dependency=afterany:${rjob} \
        --export=ALL \
        slurm/03_analyse.sbatch)
    echo "    -> job $ajob"

    echo
    echo "Submitted ${stage}: manifest=${mjob} runs=${rjob} analyse=${ajob}"
    echo "Track: squeue -u \$USER"
    echo
    echo "READ THE MANIFEST LOG FIRST -- it prints how many arms it actually saw:"
    echo "    cat logs/manifest_${mjob}.out"
    echo "It must say ${PROJ_N_GROUPS}."
}
