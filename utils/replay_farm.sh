#!/usr/bin/env bash
# Local replay farm: replay successful state-only demos on upstream fetch.
# Usage: bash utils/replay_farm.sh <workers> [start] [end]
set -u
WORKERS=${1:-6}
START=${2:-1}
END=${3:-1000}

FAILED_SEEDS="86 182 199 216 239 430 438 451 454 472 488 548 570 580 581 618 620 645 672 708 709 713 855 883 897 975 110 241 406 940"

is_failed() {
  for s in $FAILED_SEEDS; do [ "$1" = "$s" ] && return 0; done
  return 1
}

mkdir -p tmp_test/farm_logs tmp_test/replay_dataset
active=0
run_one() {
  local src="$1"
  local seed id
  seed=$(basename "$src" | sed 's/.*seed\([0-9]*\)_.*/\1/')
  id="r${seed}"
  uv run python -m utils.replay_rgb "$src" \
    --output-dir "tmp_test/replay_dataset/seed${seed}" \
    --robot fetch --stride 10 \
    > "tmp_test/farm_logs/${id}.log" 2>&1
}

for seed in $(seq "$START" "$END"); do
  is_failed "$seed" && continue
  src=$(ls -d logs/takeitback_tray_seed${seed}_20260910_1* 2>/dev/null | head -1)
  [ -z "$src" ] && continue
  [ -f "tmp_test/replay_dataset/seed${seed}/trajectory.h5" ] && continue
  while [ "$(jobs -rp | wc -l)" -ge "$WORKERS" ]; do sleep 10; done
  run_one "$src" &
done
wait
echo "FARM-DONE"
