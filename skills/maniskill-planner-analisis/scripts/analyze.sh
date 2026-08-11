#!/usr/bin/env bash

THRESHOLD=$(date +"%Y%m%d_%H%M%S")
PLANNER_NAME=${1}
SEED_COUNT=${2}
LOGS_PATH=${3:-./logs}
WD_PATH=${4:-.}

cd $WD_PATH
for seed in $(seq 1 "$SEED_COUNT"); do
  uv run python -m planners.myrobocasa_takeitback_planner --seed "$seed" &> /dev/null
done

pwd
cd $LOGS_PATH
grep -l '"success": false' */*_events.jsonl | awk -F'_' -v th="$THRESHOLD" '$3"_"$4 > th' | awk -F'/' '{print $1}'
