#!/usr/bin/env bash
# Run a ManiSkill planner across N seeds and list the log dirs that failed.
# Usage: analyze.sh <planner_module> <seed_count> [logs_dir] [cwd]
#   planner_module  Python module under planners/, e.g. myrobocasa_fridge_veggies_planner
#   seed_count      number of seeds to run (1..N)
#   logs_dir        where run logs live (default ./logs)
#   cwd             project root to run uv from (default .)
set -uo pipefail

# Accept <planner> <num_runs> (skill contract), but be forgiving if a caller
# passes <num_runs> <planner>: swap whenever the first arg is purely numeric.
if [[ $# -lt 2 ]]; then
    echo "usage: analyze.sh <planner_module> <seed_count> [logs_dir] [cwd]" >&2
    exit 2
fi
if [[ ${1} =~ ^[0-9]+$ ]]; then
    SEED_COUNT=$1
    PLANNER_NAME=$2
else
    PLANNER_NAME=$1
    SEED_COUNT=$2
fi
LOGS_PATH=${3:-./logs}
WD_PATH=${4:-.}

cd "$WD_PATH" || exit 1

# Only run dirs created after this batch started count as "this run".
THRESHOLD_START=$(date +%s)

for seed in $(seq 1 "$SEED_COUNT"); do
    echo "==> ${PLANNER_NAME} seed ${seed} (running)"
    # Planner's own artifacts (events.jsonl / csv / console.log via capture_stdout)
    # are written by its logger, not by shell redirection. Its leftover solver
    # noise on stdout/stderr cannot be captured, so silence it here.
    uv run python -m "planners.${PLANNER_NAME}" --seed "$seed" >/dev/null 2>&1
    echo "    exit $?"
done

echo
echo "==> failed runs (result event success=false) created since start:"
cd "$LOGS_PATH" || exit 1

# -newermt "@epoch" selects files written after this batch began (GNU find).
find . -maxdepth 2 -type f -name '*_events.jsonl' -newermt "@$THRESHOLD_START" 2>/dev/null |
    while IFS= read -r f; do
        if grep -q '"success": *false' "$f"; then
            echo "$(dirname "$f")"
        fi
    done | sort -u
echo "==> done"
