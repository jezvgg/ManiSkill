# Autoresearch: TakeItBack planner geometry

## Objective

Improve `myrobocasa_takeitback_planner` until all 100 evaluation seeds succeed (`100/100`). The task moves the cup from its counter position onto the tray, regrips it, and returns it to the original counter position. Prefer a better geometric strategy (reachable poses, approach corridor, base/arm/torso configuration, and transport geometry) over piles of seed-specific retries or small threshold tweaks. Do not overfit to the current seed batch and do not alter the task/evaluation to fake success.

## Metrics

- **Primary**: `successes` (count, higher is better), from 100 seeds run remotely with 25 workers.
- **Secondary**: `completed` terminal result logs, `stage3`, `stage5`, `stage12`, `missing`.

## How to Run

`./.auto/measure.sh` runs the prescribed remote benchmark:
`NO_VIDEO=1 SEED_COUNT=100 WORKERS=25 PLANNER_NAME=myrobocasa_takeitback_planner LOG_PREFIX=takeitback bash .auto/measure.sh`

The script clears local generated `logs/`, runs seeds 1..100, and emits `METRIC name=value` lines. A result counts only when that seed's terminal event has `success: true`; missing or abruptly terminated runs do not count as success.

## Files in Scope

- `planners/myrobocasa_takeitback_planner.py` — primary planner. Geometry and stage sequencing belong here.
- `utils/planners_utils.py` — shared base/velocity helpers only if the geometric planner needs a general fix.
- `.auto/*` — experiment prompt, measurement, and notes.

Read-only context:

- `my_scenes/my_robocasa_takeitback.py` — task geometry and success predicate.
- ManiSkill/mplib upstream code — do not edit.

## Off Limits

- Never modify upstream `mani_skill/`, tests, task scene definitions, evaluation predicates, or dependencies.
- Preserve unrelated pre-existing changes in `planners/myrobocasa_fridge_veggies_planner.py`.
- No seed-specific branches, no changing seeds/workers, no disabling physics/collision checks, no changing success criteria, and no mass video reruns.
- Do not call a run successful unless the terminal result event reports `success: true`.

## Constraints

- Use `uv` for Python/package execution; no dependency additions.
- Keep planner deterministic for a given seed where possible.
- Every code change must be tested by the 100-seed metric at 25 workers before keeping it.
- Favor one coherent geometric plan over local patches. Simplify/remove contradictory old paths when replacing them.
- If a benchmark is incomplete, diagnose measurement/runner integrity separately; do not tune planner against missing data.

## Current evidence and hypotheses

The initial partial 50/25 run was not a valid baseline: only 7 seeds produced trajectory files, 18 directories were empty, seeds 26..50 were absent, and event files were empty because workers were interrupted before logger close. The first measured 50/25 run reached 32/50 successes, with 45 terminal results and 5 missing. In its usable traces, six seeds stopped around pre-grasp: TCP got near the cup but orientation stayed about 121..131 degrees away and `grasped` remained false; seed 8 also printed IK failures. Seed 5 reached grasp/lift but its transport jumped 17..35 cm and missed the tray.

The current planner is over-constrained by a fixed straight-arm configuration, pan target near 85 degrees, torso-only height control, and base motion doing nearly all horizontal positioning. This can put the TCP near the object while leaving an invalid grasp orientation and little IK margin. A strong candidate direction is to choose a collision-free reachable grasp pose first, move/turn the base and torso to a robust pre-grasp configuration, then use a short arm motion for the final grasp. Transport should use the held-object transform and bounded, geometry-aware waypoints rather than relying on an ideal fixed arm offset. Reuse existing verification gates, but do not add blind retries as the main solution.

## What's Been Tried

- Straight-arm transport geometry with one base heading and torso ramp.
- Axis-aligned base drives and live cup-offset correction.
- Two-step final grasp motion, live cup remeasurement, grasp/lift/cup-gap checks.
- Multiple torso/grasp heights and collision-matrix allowances.
- Slow torso lowering and stationary release.

These mechanisms are useful diagnostics, but the benchmark evidence points to a geometric redesign rather than more attempts at the same pose. Update this section after each meaningful experiment with the result and the general lesson, including discarded ideas.

Latest nonholonomic TakeItBack experiments: e4 turn-drive-turn control reached 96/100 at 25 workers. Full Stage12 opening regressed to 95/100. Closed-gripper 12-step settling before initial/tray lifts improved to 97/100 and is kept. Low-torso recovery and pre-grasp arm reset regressed and were discarded. Remaining failures span initial grasp/lift, tray transport/regrasp, and final release; do not tune individual seeds. User requires global geometry, 100 seeds, 25 workers.
