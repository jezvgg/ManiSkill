#!/usr/bin/env bash
set -euo pipefail

SEED_COUNT=${SEED_COUNT:-100}
WORKERS=${WORKERS:-10}
PLANNER_NAME=${PLANNER_NAME:-myrobocasa_fridge_veggies_planner}
LOG_PREFIX=${LOG_PREFIX:-fridgeveggies}
export SEED_COUNT WORKERS PLANNER_NAME LOG_PREFIX

mkdir -p logs
find logs -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
# remote_analyze seeds run dirs from a persistent code cache; remove only stale
# cached logs so this iteration cannot count another run's terminal events.
ssh -o BatchMode=yes -o ConnectTimeout=15 gangway 'rm -rf ~/gw-runs/src/logs' >/dev/null 2>&1 || true

SCRIPT="$(git rev-parse --show-toplevel)/.auto/remote_analyze_reliable.sh"
NO_VIDEO=1 bash "$SCRIPT" "$PLANNER_NAME" "$SEED_COUNT" "$WORKERS"

python3 - <<'PY'
import json
import os
import re
from pathlib import Path

expected = set(range(1, int(os.environ["SEED_COUNT"]) + 1))
runs = {}
for path in Path("logs").glob(f'{os.environ["LOG_PREFIX"]}_seed*_*/{os.environ["LOG_PREFIX"]}_seed*_events.jsonl'):
    match = re.search(rf'{re.escape(os.environ["LOG_PREFIX"])}_seed(\d+)_', path.name)
    if not match:
        continue
    seed = int(match.group(1))
    if seed in expected and (seed not in runs or path.stat().st_mtime > runs[seed].stat().st_mtime):
        runs[seed] = path

completed = set()
successes = 0
error_events = 0
stage_counts = {3: 0, 5: 0, 12: 0}
for seed, event_path in runs.items():
    records = []
    with event_path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    error_events += sum(record.get("event") == "error" for record in records)
    terminal = [record for record in records if record.get("event") == "result"]
    if terminal:
        completed.add(seed)
        if bool(terminal[-1].get("success")):
            successes += 1
    console = event_path.parent / "console.log"
    text = console.read_text(encoding="utf-8", errors="replace") if console.exists() else ""
    for stage in stage_counts:
        if f"[STAGE] {stage} " in text:
            stage_counts[stage] += 1

print(f"SUMMARY runs={len(runs)} completed={len(completed)} successes={successes} missing={len(expected - completed)} errors={error_events}")
print(f"METRIC successes={successes}")
print(f"METRIC completed={len(completed)}")
if os.environ["LOG_PREFIX"] == "takeitback":
    print(f"METRIC stage3={stage_counts[3]}")
    print(f"METRIC stage5={stage_counts[5]}")
    print(f"METRIC stage12={stage_counts[12]}")
print(f"METRIC missing={len(expected - completed)}")
PY
