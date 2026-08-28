#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
find logs -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
# remote_analyze seeds run dirs from a persistent code cache; remove only stale
# cached logs so this iteration cannot count another run's terminal events.
ssh -o BatchMode=yes -o ConnectTimeout=15 gangway 'rm -rf ~/gw-runs/src/logs' >/dev/null 2>&1 || true

SCRIPT=$(find "$(git rev-parse --show-toplevel)/skills" -name remote_analyze.sh | head -1)
NO_VIDEO=1 bash "$SCRIPT" myrobocasa_takeitback_planner 50 10

python3 - <<'PY'
import json
import re
from pathlib import Path

expected = set(range(1, 51))
runs = {}
for path in Path("logs").glob("takeitback_seed*_*/takeitback_seed*_events.jsonl"):
    match = re.search(r"takeitback_seed(\d+)_", path.name)
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

print(f"SUMMARY runs={len(runs)} completed={len(completed)} successes={successes} missing={50-len(completed)} errors={error_events}")
print(f"METRIC successes={successes}")
print(f"METRIC completed={len(completed)}")
print(f"METRIC stage3={stage_counts[3]}")
print(f"METRIC stage5={stage_counts[5]}")
print(f"METRIC stage12={stage_counts[12]}")
print(f"METRIC missing={50-len(completed)}")
PY
