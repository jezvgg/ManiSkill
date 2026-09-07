# Autoresearch: TakeItBack planner geometry

## Objective

> **Amendment 2026-09-07 (user directive, overrides the scene read-only rule for
> exactly this change):** scene layout v2 in `my_scenes/my_robocasa_fridge_veggies.py`
> moves objects inward from the counter front edge. `LIFT_CLEAR_GAP` changes
> 0.08→0.02 to extend the placement strip north; vegetables now spawn around
> y=-0.494..-0.514 (13.6–15.6 cm from edge y=-0.65, roughly 1–2 cm more inset).
> The pot moves from y≈-0.56 to y≈-0.51; its 9.6 cm rim is fully on the counter
> instead of overhanging the edge. Benchmark distribution changed: metrics are
> re-baselined from the first v2 run with committed planner `37892bc4`; old
> 94/99 results are historical only. No further scene/evaluation edits without
> a new user directive.

Improve `myrobocasa_takeitback_planner` until all 50 evaluation seeds succeed (`50/50`). The task moves the cup from its counter position onto the tray, regrips it, and returns it to the original counter position. Prefer a better geometric strategy (reachable poses, approach corridor, base/arm/torso configuration, and transport geometry) over piles of seed-specific retries or small threshold tweaks. Do not overfit to the current seed batch and do not alter the task/evaluation to fake success.

## Metrics

- **Primary**: `successes` (count, higher is better), from 50 seeds run remotely with 10 workers.
- **Secondary**: `completed` terminal result logs, `stage3`, `stage5`, `stage12`, `missing`.

## How to Run

`./.auto/measure.sh` runs the prescribed remote benchmark:
`NO_VIDEO=1 bash skills/maniskill-remote-planner-analysis/scripts/remote_analyze.sh myrobocasa_takeitback_planner 50 10`

The script clears local generated `logs/`, runs seeds 1..50, and emits `METRIC name=value` lines. A result counts only when that seed's terminal event has `success: true`; missing or abruptly terminated runs do not count as success.

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
- Every code change must be tested by the 50-seed metric before keeping it.
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

Latest experiments: 10 workers produced a complete 36/50 baseline. A 65-degree pan redesign collapsed to 12/50 and was discarded. A fixed-arm screw transport route reached 42/50; fixed-arm screw staging plus live pre-grasp parking reached 45/50; live front-facing initial fallback reached 38/50 when tested alone. Closing from the release pose and validating an immediate vertical lift reached 48/50, then skipping the redundant Stage 9 lift check after that verified lift reached 50/50. Independent confirmation also reached 50/50 with 10 workers; target is met.
