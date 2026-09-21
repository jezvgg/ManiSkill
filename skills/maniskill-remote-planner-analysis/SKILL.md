---
name: maniskill-remote-planner-analysis
description: Runs the ManiSkill planner benchmark remotely on gangway (ssh) with parallel workers and pulls the logs back. Read maniskill-planner-analysis, but use this skill's script for the remote computation.
---

Read **maniskill-planner-analysis**, but run the benchmark with this script instead of its local one.

**Where the script is** — do not hunt for it. Resolve it once (works from any directory inside this repo/worktree):

```bash
SCRIPT=$(find "$(git rev-parse --show-toplevel)/skills" -name remote_analyze.sh | head -1)
```

Then run the benchmark with it:

```bash
bash "$SCRIPT" {planner} {num_runs} {workers}
```

Rendering is by far the biggest CPU cost; for large batches (10+ workers) run video-less:

```bash
NO_VIDEO=1 bash "$SCRIPT" {planner} {num_runs} {workers}
```

## Failed-seed videos: on demand by subagents

Do **not** re-run all failed seeds locally upfront — re-rendering every failed seed costs about as much as the batch itself. Instead, let the analysis subagents decide (per `maniskill-planner-analysis`): a subagent locally re-runs **a specific** failed seed **with video only when the video is actually needed** to confirm an anomaly, watches the resulting mp4, and reports.

This overrides the read-only rule of `maniskill-logs-analysis` for exactly this case: rerun is allowed on demand, not as a mass pre-pass. Local re-runs stay **single-worker** (one seed, no parallel batch):

```bash
uv run python -m planners.{planner} --seed {failed_seed}
```

The subagent then inspects `logs/<run>/video_*.mp4` around the flagged steps.
