#!/bin/bash
#SBATCH --job-name=ppocf_verify
#SBATCH --output=logs/verify_%j.out
#SBATCH --error=logs/verify_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=normal

# Installs nothing. Proves this cluster can run this project, without spending
# a queue slot on real work. Run it after the venv build, after any dependency
# change, and any time a stage fails in a way you cannot explain.

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

echo "--- 0. no CRLF line endings ---"
# sbatch refuses a CRLF *script*, so if you are reading this the .sbatch files are
# fine. Sourced .sh files and .py files are not checked by anything else, and a
# stray \r there fails in a way that looks unrelated to line endings.
bad=$(grep -rlU $'\r' --include='*.sh' --include='*.py' --include='*.yaml' \
        --include='*.sbatch' slurm pipeline config agents envs oracle utils dataio scripts \
        2>/dev/null || true)
if [ -n "$bad" ]; then
    echo "CRLF found in:" >&2; echo "$bad" >&2
    echo "fix with: bash slurm/fix_line_endings.sh" >&2
    exit 1
fi
echo "  all shell/python/yaml files are LF"

echo "--- 1. every repository module imports ---"
# A typo in an import statement discovered by the array at 3 a.m. is an
# expensive way to find a typo.
python - <<'PY'
import importlib, pkgutil, sys
bad = []
for pkg in ("config", "agents", "envs", "oracle", "dataio", "utils", "pipeline", "scripts"):
    try:
        m = importlib.import_module(pkg)
    except Exception as e:
        bad.append((pkg, repr(e))); continue
    for mod in pkgutil.walk_packages(m.__path__, pkg + "."):
        try:
            importlib.import_module(mod.name)
        except Exception as e:
            bad.append((mod.name, repr(e)))
print("FAILED:", bad) if bad else print("all modules import")
sys.exit(1 if bad else 0)
PY

echo
echo "--- 2. every stage config resolves ---"
python - <<'PY'
import sys
from pipeline.stages import STAGES, GROUPS_ALL, build_config
for stage in STAGES:
    for group in GROUPS_ALL:
        cfg = build_config(stage, group, 0)
        print(f"  {stage:10s} {group:5s} {cfg.env.env_id:32s} "
              f"pg={cfg.ppo.pg_mode:14s} H={cfg.ppo.cf_horizon} "
              f"sub={cfg.ppo.cf_subsample} a_cf={cfg.ppo.alpha_cf} "
              f"frames={cfg.ppo.total_timesteps:,}")
print("all stage configs build")
PY

echo
echo "--- 3. the counterfactual restore is exact ---"
# The one failure mode that is otherwise invisible: a broken set_sim_state
# yields plausible, WRONG A_CF. RedBlueDoors is the env where this bit --
# Grid.decode makes new WorldObj instances, leaving env.red_door / env.blue_door
# pointing at stale objects, and the winning move measured 0.9938 live and
# 0.0000 after a restore.
python - <<'RESTORE'
import numpy as np
from envs.env_pool import make_env, get_sim_state, set_sim_state

def img(o):
    return np.asarray(o["image"] if isinstance(o, dict) else o)

for env_id in ("MiniGrid-RedBlueDoors-6x6-v0", "MiniGrid-RedBlueDoors-8x8-v0"):
    env = make_env(env_id, max_episode_steps=None)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    n = 0
    for _ in range(40):
        s = get_sim_state(env)
        a = int(rng.integers(env.action_space.n))
        o1, r1, t1, tr1, _ = env.step(a)
        set_sim_state(env, s)
        o2, r2, t2, tr2, _ = env.step(a)
        assert r1 == r2 and t1 == t2 and np.array_equal(img(o1), img(o2)), (
            f"restore mismatch on {env_id} after action {a}")
        n += 1
        if t1 or tr1:
            env.reset(seed=0)
    print(f"  {env_id}: restore exact over {n} transitions")
RESTORE
echo
echo "--- 4. one real PPO-CF update, end to end ---"
# An import check passes on a machine where the first backward pass segfaults.
python -m pipeline.smoke --steps 2 --pg_mode cf_all_action
rm -rf runs/_smoke || true

echo
echo "verify OK -- safe to submit ./experiment_1_6_envs"
