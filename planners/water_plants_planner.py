"""Oracle for MikasaWaterPlants-v0, and its blind twin.

Written against the amd64 CPU container (`tools/docker/planner.sh run …`) on
kitchen 102 — mplib has no macOS wheel, so nothing here was ever "verified by
reading". The shared stage kit is `oracle_common.py`; the staging follows
`burner_planner.py` (carry an object across the kitchen) and
`station_checklist_planner.py` (drive a multi-stop route).

There is **no cue phase** in this task, so there is no `wait_cue`: at t = 0 the
answer does not exist yet. What the oracle demonstrates is the *accumulation* of
a self-generated set, and the memory it uses is the single line
`choose_next_plant` — the one line `--blind` replaces.

Opening, once:

0. **Drive onto the refill dock**, with an empty gripper and the arm at rest. The
   reset parks the base near the dock with up to 0.08 m and 0.10 rad of jitter, and
   the grasp that follows is the tightest reach in the episode.
1. **Grasp the cup**, from the top, with the torso lowered first, and **lift**. The
   cup is grasped once and never released: every predicate requires
   `agent.is_grasping(cup)`.

Then one trip, three times:

2. **Dip.** The cup into the bucket: origin within `pour_xy_radius` of the bucket's
   xy and inside `pour_min/max_clearance` above its mesh top, approached from a hover
   above the band. The torso has to come **down** — measured, W1: forward reach at
   z = 0.25 m falls from 0.980 m (torso 0.000) to 0.549 m (torso 0.386), and the
   bucket is 0.80 m out from the dock. `dip_done` is a latch the env clears the moment
   the base leaves the dock, so it cannot be banked for the next trip.
3. **Fold to the rest keyframe.** The refill only counts with the arm back in the
   carry pose — every joint of `_carry_qpos_idx` within `carry_pose_tol_rad`
   (0.15 rad) of the `rest` keyframe. That is a *joint-space* goal, so it goes to
   `plan_qpos` (`fold_to_rest`) rather than to `static_manipulation`, which takes a
   TCP pose and would satisfy it at some other elbow.
4. **Dwell.** `refill_dwell[trip]` consecutive steps with the base parked, the cup
   grasped, `dip_done` set and the arm in the carry pose. Then `has_water`.
5. **Choose the next plant** — `choose_next_plant`, the memory, one line.
6. **Drive, then pour.** The cup over the pot within `pour_xy_radius` in xy, inside the
   clearance band above the pot's mesh top, tipped at least `pour_tilt_deg` off
   vertical, held `hold_steps` consecutive steps with the base in that plant's zone.
7. **Carry pose, drive back** for the next dose. The forced return is what stops the
   base pose from being the progress counter, so it is a stage and not an economy —
   and it is skipped only after the last dose, when the budget is spent and the
   verdict has already latched.

Every stage that reaches into a corner of the workspace is a **ladder**, and every rung
is a different attempt rather than another draw: grasp heights, dip postures, drive
recoveries, pour clearances x commanded tilts x tilt axes. Each ladder exists because
its stage was measured to be refused, and the constants say which refusal. A refused
plan executes nothing, so the ladders cost planning and not horizon.



TWO SOLVER DEFECTS THIS ORACLE WORKS AROUND, AND HOW
----------------------------------------------------
Both are in `extand.py`, both were measured here, and neither is fixed at its source
in this branch — that file is shared by `MikasaBurner-v0`, `MikasaSeasonDish-v0` and
`MikasaStationChecklist-v0`, whose published numbers and recorded demonstrations were
taken with its current behaviour.

**1. The base plan moves the arm.** `move_base_forward` asks the planner for a TCP pose
one base-delta away with `BASE_PLAN_MASK`, which frees all fifteen joints — the request
never says "with the base only". `docs/solver-delta-primer.md` §3 makes exactly this
argument for base *rotation* and answers it with `base_yaw.py`; translation kept the old
shape. Measured: every drive refusal on this task was a **wrist** joint limit inside a
plan whose only job was to move the base. Handled by an **opt-in** parameter, added in
this branch: `drive_base(..., freeze_arm=True)` selects `BASE_ONLY_PLAN_MASK`. The
default is untouched, so nothing else plans differently.

**2. Every executed plan zeroes the head.** See `hold_rest_body`, which is the one place
in this oracle that builds an action by hand, and whose docstring says at length why
that is a motor competence rather than a privilege. All three of
`extand.py`'s path followers — `follow_forward_path_w_refinement`,
`follow_rotation`, `follow_moving_forward` — write the body action as
`body_action[0] = body_action[1] = 0.0`, i.e. they command **head_pan and
head_tilt to zero on every executed plan**, and no primitive ever commands them
back (`idle_steps` and `change_gripper_state` hold the *current* body qpos). The
task's carry pose is read off the `rest` keyframe, where `head_tilt = 0.562`.
Measured in the container, seed 11:

    after the fold, cup in the fingers:
        max|d| over the seven arm joints and the torso = 0.0076 rad
        head_tilt                                       = 0.5619 rad  <-- over tol
        carry_error = 0.5619, carry_ok = False

So the arm **can** return to the rest keyframe while gripping the cup — the risk
`task-6-context.md` §3 names is answered, and answered yes — but `carry_ok` is
still false, for a reason that has nothing to do with the arm. `hold_rest_body`
sends the head back to its rest angle during the dwell. The head stays in the
predicate — it is the "body as memory" channel the refill exists to close, and
weakening it would open the hole the task is built to shut. The fix belongs in
the three path followers, and goes in separately; this function retires with it.

WHAT THE BLIND CONTROL DOES NOT PROVE
------------------------------------
`choose_next_plant` is the whole memory of this task and `blind_histogram` is what says
its draw is uniform. A reviewer checked both by **mutation** rather than by reading — a
constant arm fails uniformity, a without-replacement arm passes every per-site check and
gives itself away only on a permutation rate of 1.0000, a 50/25/25 bias fails uniformity
— and in doing so established three things the histogram does *not* establish. They are
here because they change how a sweep's numbers must be read, and that is exactly the kind
of caveat that gets lost between a review and a paper.

1. **The histogram cannot see a blind arm that filters.** Its stand-in task carries a
   `watered_mask` that is all-False and is never mutated, so a blind branch that skipped
   already-watered sites would still print a perfect 0.2222. What catches that is a
   *different* test — `test_the_blind_arm_ignores_what_it_has_already_watered`, which
   draws against a mask with sites already watered and requires the same answers. The
   histogram is not the guard; that test is.

2. **The histogram measures the function, not the episode.** It calls
   `choose_next_plant` directly. `water()`'s `watered_before` and `solve`'s
   `already_watered=` are log fields today and nothing enforces that they stay inert; an
   edit that made either load-bearing would hand the blind arm memory with every offline
   check still green.

3. **0.2218 is the design floor, not the realised distribution of blind episodes.** The
   histogram always takes exactly `n_plants` draws. `solve` takes one per trip and stops
   at the first stage that misses, so a blind episode can end after one or two draws.
   The floor is what the draw *would* give over a full episode; what a sweep measures is
   that convolved with the motor stack.

And a fourth, from the same review: **the blind arm has never been run against the
simulator.** Its motor rate is assumed equal to the sighted arm's because the two share
every stage but one line — but a revisit is not the same drive as a first visit, so a
blind rate that comes in low could be motor cost rather than memory. Task 7 has to
separate those.

Contract, as `template_planner.py:32-45`: the solution owns the reset; `-1` only
for a planning or grasp refusal and the gym 5-tuple otherwise; do not catch
exceptions; do not decide success; do not call `env.evaluate()` — read `info`.
"""

from __future__ import annotations

import argparse
import math
import sys

import gymnasium as gym
import os
import numpy as np
import sapien

from mani_skill.utils.wrappers import RecordEpisode

from utils.mikasa_oracle.planners import oracle_common as common
from utils.mikasa.seeding import seed_everything

# The mplib-dependent imports stay deferred inside oracle_common's factories, so
# this module imports on a Mac and the blind histogram below runs there.

WHO = "water_plants_planner"

# Every "measured" in the constants below was taken in the amd64 CPU container on
# kitchen 102, on the calibration seeds (11, 12 — the sweep's 0-9 are reserved), and is
# written up with its trace in
# `.superpowers/sdd/structured-frolicking-riddle/task-6-report.md`.

# Torso height the station work is done at — the grasp, the hover and the dip.
#
# **It was 0.05, and that was the tightest constraint in the task.** On the floor the
# dip sits at z ~ 0.27 and the refill dock is 0.55 m out, and W1's table gives forward
# reach at that height as 0.980 m at torso 0.000 falling to **0.549 m at torso 0.386**
# — so 0.55 m was inside the envelope only with the torso against the bottom of its
# travel. 0.05 rather than 0.0 bought five centimetres of somewhere-to-go, because at
# the hard limit every screw for a downward TCP move dies on `joint limit at index [3]`
# before it starts (measured, seed 11). Five centimetres is not much somewhere to go:
# that same index [3] is the largest single cause of screw refusal in the K59 sweep —
# **397, against shoulder_lift's 209 and wrist_flex's 162 together** — and each refusal
# is a straight line replaced by an RRT path of median 83 knots against the screw's 20.
#
# K60 lifts the station to 0.45 (`WaterPlantsConfig.station_z_lift`), which puts the
# dip at z ~ 0.72 and the grasp at z ~ 0.54. W1's table at those heights reads
# 1.021-1.126 m and 0.928-1.100 m — inside the envelope at **every** torso height, with
# roughly half a metre to spare. The dock distance has stopped constraining the torso,
# so the torso can sit where it is most useful to a screw rather than where reach
# forces it, and that is the middle of its travel: 0.193 is equidistant from both stops
# (0.193 below, 0.193 above) and is a row W1 measured directly rather than interpolated.
TORSO_DOWN = 0.193

# The torso height used for the *cup grasp* specifically, split from `TORSO_DOWN`
# (which the dip posture also uses) so the two can be measured apart. K57: at 0.05 the
# pre-grasp hover's screw plan still dies on `joint limit at index [3] after 1 step(s),
# 1.581 of the twist left` — index 3 is `torso_lift_joint`, limits [0, 0.386] — and the
# fallback is a 70-knot RRT whose randomised path is the visible up-and-down before the
# grasp. W1 measured this top grasp reachable at torso 0.000/0.097/0.193 and no higher,
# so there is headroom above 0.05 to give the screw somewhere to go.
TORSO_GRASP = float(os.environ.get("MIKASA_WP_TORSO_GRASP", TORSO_DOWN))

# Legs for the pre-grasp *reach* (the hover 0.12 m above the cup). K57: this is the
# stage that produces the visible up-and-down before the grasp. Its screw plan dies
# with `joint limit at index [3] after 1 step(s), 1.581 of the twist left` — 1.581 rad
# is ~90 deg of TCP reorientation asked for in a single constant twist — and the
# fallback is a 70-knot RRTConnect path whose randomised detour is what you see.
# Splitting the approach into legs makes each twist smaller, which is the same lever
# that fixed the carry (`move_via`), applied to the stage that actually needs it.
REACH_LEGS = int(os.environ.get("MIKASA_WP_REACH_LEGS", "1"))

# Freeze the torso for the pre-grasp reach, the way the descent below it already does.
# K57: the reach's screw dies on `joint limit at index [3]` — index 3 IS
# `torso_lift_joint` (verified against `move_group_joint_indices`) — and the code one
# stage lower already knows the cure: "with it free the screw reports joint limit at
# index [3] before it has moved ... Frozen, the same descent is the arm's to make."
# The same argument applies to the reach and was simply never applied there.
REACH_FREEZE_TORSO = os.environ.get("MIKASA_WP_REACH_FREEZE", "0") != "0"

# Longest RRT path the pour tip is willing to execute, and how many draws to take
# looking for one. **Measured, K58.** Over the 30-seed fleet, 85 tips ran with the cup
# actually in the gripper and 5 of them threw it out. Every one of those 5 was a plan
# of >= 173 knots; not one of the 73 tips under 173 knots ever dropped the cup:
#
#     tip plan length     tips    dropped the cup
#     < 173 knots           73          0
#     >= 173 knots          12          5   (42 %)
#
# So a long path is *necessary but not sufficient* — 12 long tips held — and the cure
# is not to forbid the goal but to ask RRTConnect for a different path to it. Draws are
# free in simulation steps, so the tip takes several and keeps the shortest. This is
# also the "simple trajectory" the VLA evaluation wants: a 231-knot tip swings the cup
# through ~100-152 deg of intermediate roll to reach a 65 deg target.
POUR_MAX_KNOTS = int(os.environ.get("MIKASA_WP_POUR_MAX_KNOTS", "150"))
POUR_KNOT_DRAWS = int(os.environ.get("MIKASA_WP_POUR_KNOT_DRAWS", "4"))

# The same cap, for every other arm move made *while the cup is in the fingers*.
# K58 second wave: capping only the pour tip moved the drops rather than removing
# them — the 30-seed sweep came back 26/30 with all five original failures fixed, but
# three of the four new failures were cup drops at the **dip** and **carry** stages
# (seeds 2 and 10 at ~step 350, seed 8 at 1700), which had no cap. The mechanism does
# not care which stage it is: a long RRT path swings the cup through extreme
# intermediate orientations and gravity takes it out of a 2.75 cm rim grasp. Draws
# cost no simulation steps, so every held-cup move now asks for a short path.
HELD_MAX_KNOTS = int(os.environ.get("MIKASA_WP_HELD_MAX_KNOTS", "150"))
HELD_KNOT_DRAWS = int(os.environ.get("MIKASA_WP_HELD_KNOT_DRAWS", "4"))

# TCP heights for the top grasp, **below the top of the cup's world collision AABB**,
# in order. Three measurements set this, and two of them refuted an earlier guess:
#
#   * `agent.tcp` IS the point between the fingers — the offset from `gripper_link` to
#     `(finger1 + finger2) / 2` is 0.0000 m — so the TCP's height is the height the
#     fingers straddle at.
#   * **The cup sinks 0.04 m in the first ten steps of every episode**, on the Mac and
#     in the container alike: its origin goes 0.0775 -> 0.0375 and its 0.115 m hull
#     settles spanning z -0.020 to 0.095, i.e. 2 cm *through* the
#     floor it was spawned 0.02 m above. Nothing in the task reads the cup's resting
#     height, so this breaks no predicate, but it means the graspable part of the cup
#     is only its top ~3 cm and that anything keyed to `cup.pose.p[2] + _cup_rest_lift`
#     (0.135) aims 4 cm above the cup. **The AABB top, read after the settle, is the
#     only honest anchor.**
#   * An earlier version of this comment claimed the hull differed between mani_skill
#     3.0.1 and 3.0.0b22 (0.135 vs 0.095). **Refuted**: the hull is 0.115 m tall on
#     both; one reading was taken at t = 0 and the other after the settle.
#
# The window is narrow and was measured to be narrow, all on seed 11 in the container:
# the three heights that have held the cup are 0.0200, 0.0225 and 0.0275 below the
# settled top; 0.0175 below it closes on air; 0.045, 0.0625 and 0.08 below it are
# refused as `collision gripper_link<->cup`. The target is *inside* the cup's convex
# hull by construction (as the burner's is), so whether mplib calls it a collision is
# a knife-edge — the same pose planned on one run and was refused on the next. Hence
# several draws per pose. A close that misses knocks the cup over and moves it ~0.10 m,
# so the mesh is re-read on every rung.
GRASP_DEPTHS = (0.0275, 0.0225, 0.0250, 0.0200)

# Where the cup origin goes above the target's mesh top. The task's band is
# `pour_min_clearance` 0.05 to `pour_max_clearance` 0.30. The dip takes the middle of
# it; the pour takes a **ladder from the bottom up**, because the pot top is at
# z = 1.22 and the pour pose is the highest, furthest reach in the episode: at
# +0.15 (cup origin z = 1.37) it is refused outright — `joint limit at index [3]`,
# the torso already at its 0.38615 ceiling, then `IK Failed` (measured, seed 11,
# plant 0). W1's "the pour band is reachable at every torso height" was a pure-FK
# upper bound and its own docstring says planability is a different question; this is
# that question answered.
DIP_CLEARANCE = 0.20

#: Where the dip puts the cup above the bucket's mesh top, tried in order. The task's
#: band is `pour_min_clearance` 0.05 to `pour_max_clearance` 0.30 and every entry is
#: inside it, so which one lands does not change whether the refill counts.
#:
#: K59. The dip used to be a single height with a two-rung torso ladder behind it, and
#: when both rungs refused the oracle threw away a hover it had *already reached* and
#: restarted the whole posture from the rest keyframe — which, holding a cup at floor
#: height, drops it back to the floor and makes the next hover's IK unsolvable. That
#: cascade is the whole of seeds 6 and 10, both of them episodes where the hover had
#: succeeded and only the 18 cm descent behind it had not. Three heights from a good
#: hover is strictly more than one, and costs nothing when the first works: a refused
#: plan executes no steps.
#:
#: 0.26 before 0.14 on purpose. The refusals measured at 0.20 are
#: `collision scene-0_cup<->scene-0_floor` — a verdict the cup's own geometry says is
#: impossible at that height (its base sits ~0.227 m up) and which TODOS carries a card
#: to explain — so the first retry moves *away* from the floor rather than toward it.
DIP_CLEARANCES = (DIP_CLEARANCE, 0.26, 0.14)

POUR_CLEARANCES = (0.08, 0.13, 0.20)

# Approach heights: the cup is brought to this much above the target's mesh top,
# *outside* the 0.05-0.30 band, before it descends into it. Measured why: with the
# cup grasped on the floor beside the bucket, a direct move to the dip pose is a
# straight line through the bucket wall and `plan_screw` refuses it —
# `collision scene-0_cup<->scene-0_water_bucket` — with the RRT fallback's IK finding
# nothing either (seed 11).
DIP_HOVER = 0.38
LIFT_AFTER_GRASP = 0.20

# How far the dip and pour targets are pulled from the bucket's / the pot's centre
# *toward the base*. The predicate admits the cup anywhere within `pour_xy_radius` =
# 0.10 m of the target's xy, so this is reach bought inside the tolerance rather than
# against it — and both targets need it: the bucket is 0.80 m out from the dock and
# the pot 0.874 m out from the plant dock (measured, plant 0), both at the edge of the
# arm. The pour's is the smaller of the two because the pour is also *committed* on
# that distance: `d_pot <= 0.10` must hold with the solver's tracking error (up to
# ~0.02 m) on top.
DIP_PULLBACK = 0.07
POUR_PULLBACK = 0.06

# How far past the plant's dock, toward the counter, the base is parked for a pour.
# `at_plant` admits the base anywhere within `dock_radius` = 0.20 m of the dock, so this
# is the same trade as the pullbacks above — reach bought inside a tolerance. It is
# worth 0.13 m of a reach the arm does not have: the pot is 0.874 m out from the dock
# itself and the pour pose was refused there at every clearance in the band.
POUR_DOCK_ADVANCE = 0.13

# There is deliberately **no** equivalent at the refill dock, and the reason is a
# measurement rather than an omission. Until the station was relocated the dock stood
# 0.80 m from the bucket, past what the arm plans a dip at, and the oracle advanced
# 0.13 m to close the gap. The bucket is now `cfg.dip_distance` = 0.55 m away by
# construction, so no advance is needed — and one would be actively unsafe: 0.13 m
# forward puts the base **0.4200 m** from the bucket, inside the measured 0.4132-0.4156 m
# no-go band below.

# The Fetch base's collision hull, measured (probe block 7), and the two radii are not
# interchangeable — which one applies depends on how tall the obstacle is:
#
#   BASE_HULL_XY            the hull's greatest reach in xy, at any height. The number
#                           for anything the base can hit at hub height.
#   BASE_HULL_BELOW_BUCKET  its reach below 0.0736 m, the bucket's full height. The
#                           bucket is a shallow bowl, so this is the only part of the
#                           base that can ever touch it, and it is the one used here.
#
# The oracle previously assumed 0.28 m and said so; the truth is 8 mm more.
BASE_HULL_XY = 0.2876
BASE_HULL_BELOW_BUCKET = 0.2853

# Distance between the base's and the bucket's centres at which they collide: the hull
# radius below the bucket's height plus the bucket's own half-extent. Measured, and it
# **predicts** the refusals rather than being fitted to them — the oracle's observed
# boundary (0.3645 collided, 0.4560 drove) brackets this band.
STATION_NOGO = (0.4132, 0.4156)

# How far the cup is *asked* to tip for the pour, in order, and the margin the tilt it
# actually reaches must clear the predicate by.
#
# The commands are far past `pour_tilt_deg` = 55 on purpose, and the reason is measured:
# **the cup rotates back inside the fingers when it is tipped.** On seed 11, plant 0, a
# 75-degree command was executed with the TCP landing 0.004 m / 0.7 deg from the goal —
# and the *cup* came out at 53.9 deg, 1.1 deg under the threshold, so nothing committed
# with every other term of the predicate satisfied (`d_pot` 0.060, clearance 0.075,
# `at_plant_id` 0, `has_water`, `is_grasped`). That ~21 deg is not tracking error; it is
# the cup turning in a top grasp once its weight is off the grip axis, and it is exactly
# the kind of thing a return code cannot tell you. So the tilt is **verified from
# `info["tilt_rad"]` after the move** and a command that lands short is followed by a
# bigger one.
# K55: the reference is now `level_quat`, so a command is an ABSOLUTE tilt rather
# than an increment on an unknown one — and that turns out to remove the slip the
# paragraph above describes, rather than merely accounting for it. Measured on
# seeds 0 and 1 with the absolute reference: `commanded 80.0 -> reached 80.1`,
# `lost_deg -0.1` and `-0.2`. The ~21 deg of "the cup turns back inside the
# fingers" was the *relative* reference compounding an unknown starting tilt, not
# the grip giving way. So the first rung is set just clear of the predicate
# (`pour_tilt_deg` 55 + `POUR_TILT_MARGIN_DEG` 4 = 59, plus 6 deg of headroom)
# instead of 45 deg past it. The ladder still escalates for the case where the
# grip really does slip, and the tilt that counts is still read back from
# `info["tilt_rad"]` after the move, never inferred from the plan.
#: Interpolated waypoints each long arm move is broken into. 1 restores the old
#: single-plan behaviour. See `oracle_common.move_via` for why: a long plan lets
#: RRT wander (2557 deg travelled against 858 deg of net change, and a hover that
#: ends 1.3 deg from where it started after 269 deg of travel), and short hops let
#: the straight-line screw plan carry more of the path.
#: Do the torso's share of a joint-space move first and alone — jezv's insight, that
#: an arm turning through a large angle to change height can often just be the torso
#: going up or down. `torso_lift_joint` is prismatic, so its share of a move costs
#: **zero** hand rotation, and the fold carries it 0.05 -> 0.386 m while spending
#: 182 deg of rotation doing so.
#:
#: **Off, and now for a measured reason rather than a blocked one.** At n=6 against
#: the same seeds: 4/6 success and 2653 deg of hand rotation with it on, against
#: **6/6 and 2433** with it off. It looked like a 20% win at n=2 — seed 0 went
#: 2198 -> 1752 — and that was noise: the same seed at n=6 comes back 3280. The idea
#: is still sound in principle (a prismatic joint changes height for free) and the
#: enabling fix below is kept; what the numbers say is that splitting the move hands
#: the arm a worse starting configuration more often than it saves rotation.
#:
#: The blocker that kept this off is fixed. It was not the torso leg at all: splitting
#: the move makes the *second* plan a no-op whenever the first covered it (a
#: torso-only target leaves the arm nothing to do), and a zero-knot plan came back
#: `Success` and then died in `follow_forward_path_w_refinement` on
#: `result["position"][-1]`. That is a solver robustness gap of its own — nothing in
#: the shipped staging ever produced an empty plan — and it is guarded there now.
#:
#: Not the same as freeing the torso inside a single plan, which *was* measured and is
#: worse (2557 -> 2616 deg, 1/2, tilt 8.9% -> 20.9%): there the planner may spend the
#: torso *and* the arm on one move, where this spends the torso first and alone.
#: How far ahead of the base centre the *first* tuck attempt pulls the TCP, in
#: metres. The shipped 0.25 is inside the base footprint; rung 2 falls back to it.
#: Control steps the gripper is closed over when the cup starts in it, with the
#: cup re-seated each step so it cannot fall out before the fingers meet.
#: The tip is deliberately **not** waypointed, and that is measured twice. It is the
#: last stage where a single plan still wanders — every pour executes one tip
#: (`tips=1`, the ladder never escalates) and three of them still travel 309-389 deg
#: for a net 65, carrying the cup through 174 deg on the way. Breaking it into 3 legs
#: fixes none of that and costs two episodes: 4/6 against 6/6, mean 1411 against
#: 1082 deg, and the two moves past a full turn are still there. The wander is in the
#: reach to the pot, not in the tip, and waypoints do not reach it.

GRIP_CLOSE_STEPS = 12

#: K57, ported from the season-dish branch as a *question*, not an answer. There the
#: tuck was deleted after 30 interleaved seeds per arm showed it never once refused
#: (0/30) and nothing ever failed without it (0/30 at the drive), while costing 27% of
#: the executed motion. The reason it was safe there is stated explicitly — "this
#: oracle's two docks sit on the same counter run, so its turn is small" — and that
#: does **not** transfer: water-plants shuttles between a refill dock at yaw 290 deg
#: and plant docks at 90 deg, and every drive turns more than 30 deg (measured here,
#: 0 skips when a low-turn bypass was tried). So the tuck's stated purpose — guarding
#: `rotate_base_z`'s swept arc against the held cup (K51) — is live here in a way it
#: was not there. Knob so the same A/B can be run rather than argued.
TUCK = os.environ.get("MIKASA_WP_TUCK", "1") != "0"

CARRY_AHEAD_SHALLOW = 0.40
#: The shipped depth, used on the retry. Named rather than read off
#: `carry_pose.__defaults__`, which indexes the first *defaulted* parameter — that
#: is `who`, so the earlier version silently passed the string "oracle" as a
#: distance and every episode died in numpy.
CARRY_AHEAD_DEEP = 0.25

#: Split a joint-space move into a torso-only leg and then the arm's share, instead of
#: asking one `plan_qpos` for both (`plan_to_joint_targets`, which explains the mechanism).
#:
#: Shipped off, and until K59 with no recorded reason for that. K59 gives it one to be
#: measured against: **every remaining `RRTConnect Failed. Timeout` in the sweep is a
#: joint plan whose only real content is a torso move.** Seed 8 asks the dip posture for
#: `{"torso_lift_joint": TORSO_DOWN}` with `gap=0.17286` — 17 cm of a single prismatic
#: joint — and RRTConnect times out sampling the whole arm space to get there. It is not
#: a near-miss that costs a stage: the refused dip posture falls through to the `rest arm`
#: rung, which puts the held cup back at floor height, and the hover after it is then
#: `collision cup<->floor` + `IK Failed`. That is seeds 6, 8 and 10 — the whole of what
#: is left.
#:
#: The torso leg plans with `fixed_joint_indices=[0, 1, 2]` and one moving DOF, so it is
#: the kind of request RRT cannot time out on. It also composes with the "already there"
#: guard below: on the dip's `as it stands` rung the *only* target is the torso, so the
#: leg does the whole move and the arm plan behind it becomes a no-op the guard skips.
#:
#: A knob, not a flip, because it changes every `plan_to_joint_targets` caller — the rest
#: fold included — and because this file has twice this session measured a change that
#: reasoned well and behaved worse. Set `MIKASA_WP_TORSO_FIRST=1` to run the other arm.
TORSO_FIRST = os.environ.get("MIKASA_WP_TORSO_FIRST", "0") != "0"

#: Metres of torso travel below which the split is not worth its own plan.
TORSO_FIRST_MIN_M = 0.05

#: How close a joint has to be to a commanded value for `plan_to_joint_targets` to call
#: it arrived. 1e-3 is 1 mm on the prismatic torso and 0.06 deg on a revolute joint —
#: two orders under the solver's own goal tolerance (0.02 m / 0.10 rad) and far under
#: anything a stage depends on, so it can only fire on a target the joint is already
#: holding, never on one the arm still has to travel to.
#:
#: It was 1e-4 for the first four episodes of the v4 sweep and fired **zero** times, so
#: the torso is evidently not sitting exactly on `TORSO_DOWN` when the dip asks for it
#: again. `plan refused` now carries `gap=` — the largest distance any named joint still
#: had to cover — so the next sweep says what the real number is instead of this comment
#: guessing at it.
JOINT_AT_TARGET_RAD = 1e-3

#: 3, and the trade is measured in both directions. The owner watched a clip at 3 and
#: said the base bobs up and down too much, which is real — each leg is a separately
#: time-parameterised trajectory that decelerates to a stop, so every extra waypoint
#: is another stop and another torso correction. But 2 is worse on both things the
#: waypoints exist for: hand rotation 2272 -> 2787 deg and the cup past the spill line
#: 6.3% -> 18.2% of the carry. Smoother to watch, worse to measure. Left at 3.
WAYPOINT_LEGS = 3

#: Offsets above the angle the predicate needs, not absolute angles. The shipped
#: task needs `pour_tilt_deg` 55 + `POUR_TILT_MARGIN_DEG` 4 = 59, so these reproduce
#: the measured (65, 85, 110) exactly — but they now *follow* the threshold instead
#: of restating it, so lowering `pour_tilt_deg` actually lowers what is commanded
#: rather than leaving the oracle over-tipping by the difference.
POUR_TILT_OFFSETS = (6.0, 26.0, 51.0)


def pour_tilt_commands(task) -> tuple:
    """The tilt ladder for this task's own pour threshold, shallowest first.

    Args:
        task: `env.unwrapped`.

    Returns:
        Absolute commanded tilts in degrees.

    Example:
        >>> pour_tilt_commands(task)          # doctest: +SKIP
        (65.0, 85.0, 110.0)
    """
    need = float(task.cfg.pour_tilt_deg) + float(POUR_TILT_MARGIN_DEG)
    return tuple(need + d for d in POUR_TILT_OFFSETS)
POUR_TILT_MARGIN_DEG = 4.0

#: Cap on tips actually *executed* before the pour is called a miss. Refused plans cost
#: no steps, executed ones cost ~70, and the horizon is the scarce thing here.
#:
#: **It must stay strictly greater than `len(POUR_TILT_OFFSETS)`**, and it did not: at 4
#: it equalled `len(POUR_TILT_AXES)`, so four axes that all planned and executed at the
#: first (clearance, command) pair exhausted the budget and broke out of all three loops
#: before 120 or 140 degrees was ever asked for. The escalation the constant above says
#: was measured to be needed could then never happen. Seed 11 hid it because the first
#: tip succeeded; a sweep would not have. `test_the_tip_budget_leaves_room_to_escalate`
#: pins the relation.
POUR_MAX_TIPS = 6

#: Degrees per in-place bite of the split pour (K61). Inside K58's measured window —
#: every 21-33 deg wrist correction planned, every 65-70 deg one refused — and sized so
#: the first commanded tilt (need + 4 = ~63-65 deg) is exactly two bites. Shared with
#: nothing: `level_in_place`'s 30 is the same physics but its own measured constant.
POUR_BITE_DEG = 35.0


# Which way to tip it, as a rotation about a horizontal axis in the base's frame:
# +/- the base's lateral axis (the cup tips away from / toward the robot) and +/- its
# facing (the cup tips to one side). The innermost of the pour's three ladders, and the
# cheapest: each is a different turn for the wrist, and a refused one executes nothing.
#
# **`lateral-` leads because it is the only one that ever plans.** K59, measured over
# the 30-seed v3 sweep by attributing each `[static_manipulation]` line to the tip
# candidate that preceded it: `lateral-` at the first commanded tilt won **81 of its 82**
# attempts, and `lateral+` won **0 of 246** — every seed, every clearance, all three
# commanded tilts. It is not a weak branch, it is a dead one: tipping the cup *away*
# from the base at full pour extension puts the wrist past `joint limit at index [11]`,
# and the RRT fallback's IK then finds nothing (`IK Failed`, 821 of the sweep's 823
# refusals). Leading with it cost three guaranteed-`IK Failed` plans before every pour
# — 246 of the sweep's refusals, ~30% of all wasted planning.
#
# Reordered rather than pruned: `lateral+` and the two `facing` axes are kept as
# fallbacks because the measurement is one kitchen (scene_idx 0) and a different pot
# geometry could invert it. When the lead axis plans, the tail costs nothing.
POUR_TILT_AXES = ("lateral-", "lateral+", "facing+", "facing-")

# Steps to stand still after a drive before reading anything: the base has to be
# under `base_static_speed` before either the refill or the pour will count.
SETTLE_AFTER_DRIVE = 12

# Slack over the task's own counters before a stage is called a miss. The dwell is
# drawn per trip from [20, 60] and needs the head to physically arrive first; the
# pour needs `hold_steps` = 15 consecutive steps.
REFILL_EXTRA_STEPS = 60
POUR_EXTRA_STEPS = 25


def _np(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def _f(info: dict, key: str, idx: int = 0) -> float:
    return float(_np(info[key]).reshape(-1)[idx])


def _b(info: dict, key: str, idx: int = 0) -> bool:
    return bool(_np(info[key]).reshape(-1)[idx])


def say(env, stage: str, **extra) -> None:
    """`oracle_common.say` tagged `[water_plants_planner]`, with the step count first.

    Every line carries `t=<elapsed_steps>`: this task's schedule is three round trips
    across a kitchen and the horizon is the number Task 7 has to measure, so a trace
    that does not say *when* a stage happened cannot answer the question the run was
    for.
    """
    task = getattr(env, "unwrapped", env)
    t = getattr(task, "elapsed_steps", None)
    if t is not None:
        extra = {"t": int(_np(t).reshape(-1)[0]), **extra}
    common.say(env, WHO, stage, **extra)


def fail(env, stage: str, **extra):
    """Return -1 for a failed stage, saying which one — never a silent -1."""
    return common.fail(env, WHO, stage, **extra)


def default_planner_factory(env, debug: bool, vis: bool):
    """The real Fetch solver at the oracle refinement cap (60). Needs mplib."""
    return common.default_planner_factory(env, debug, vis)


# --------------------------------------------------------------- the memory --


def choose_next_plant(task, blind: bool, rng: np.random.Generator) -> int:
    """The site to water next. **This one function is the whole memory of the task.**

    Sighted: the lowest-indexed occupied site that has not been watered — a
    privileged read of `watered_mask`, which a demonstrator is allowed and a policy
    is not (`_get_obs_extra` never emits it).

    Blind: a **uniform draw from the occupied sites, with replacement**. With
    replacement is the point, not a slip: it is what makes the memoryless floor
    `3!/3**3 = 0.222`. Drawing *without* replacement would be a memory of its own —
    the arm would never repeat a plant, which is exactly the ability under test —
    and the control would be worth nothing. The blind arm still knows *where* the
    plants are (a plant is visible; a watered one is deliberately indistinguishable
    from a dry one, design hole A); what it does not know is which it has already
    served.

    `MikasaStationChecklist-v0`'s blind arm returned the constant `[0, 1, 2]` for
    six months and both of its published numbers turned out not to be controls
    (found 2026-08-21). Hence `blind_histogram` below, and hence the rule that the
    histogram is printed before any sweep.

    Args:
        task: `env.unwrapped` — read for `occupied_mask` and `watered_mask` only.
        blind: True for the memory-free control arm.
        rng: `np.random.default_rng(seed)`, drawn from once per trip.

    Returns:
        A site index in `range(cfg.k_sites)` that holds a plant.

    Example:
        >>> import types, numpy as np
        >>> t = types.SimpleNamespace(
        ...     occupied_mask=np.array([[False, True, True, False, True]]),
        ...     watered_mask=np.array([[False, False, True, False, False]]))
        >>> choose_next_plant(t, blind=False, rng=np.random.default_rng(0))
        1
        >>> rng = np.random.default_rng(0)
        >>> sorted({choose_next_plant(t, blind=True, rng=rng) for _ in range(200)})
        [1, 2, 4]
    """
    occupied = np.flatnonzero(_np(task.occupied_mask).reshape(-1).astype(bool))
    if blind:
        return int(occupied[rng.integers(len(occupied))])
    watered = _np(task.watered_mask).reshape(-1).astype(bool)
    for s in occupied:
        if not watered[s]:
            return int(s)
    return int(occupied[0])


def blind_histogram(n_seeds: int = 20000, occupied=(1, 2, 4), n_plants: int = 3):
    """Draw a whole blind episode's choices on each of `n_seeds` seeds and count them.

    Reproduces `solve(blind=True)` exactly: one `np.random.default_rng(seed)` per
    episode and `n_plants` calls to `choose_next_plant` off it, in order, against the
    same `occupied` set. Nothing here simulates anything, so it runs on a Mac.

    The three endpoints the design names are recomputed from the draws with the
    bookkeeping of `WaterPlantsTask._commit_pour`, not with a shortcut: a repeat
    latches `repeat_seen`, and after it no later pour counts as distinct — which is
    why `distinct_before_repeat` is {1: 1/3, 2: 4/9, 3: 2/9} and *not* the
    distribution of `len(set(draws))` ({1: 1/9, 2: 2/3, 3: 2/9}). They agree only on
    the third bin, which is the floor, and it is easy to check the wrong one.

    Args:
        n_seeds: episodes to draw.
        occupied: the sites holding a plant.
        n_plants: doses per episode; also the number of draws.

    Returns:
        `(first, allc, dbr, first_ok, perms)` — `first[s]` counts of the trip-0 draw
        per site, `allc[s]` counts over all trips, `dbr[d]` how many episodes reached
        `distinct_before_repeat == d`, `first_ok` how many got the *second* pour right
        (the first cannot be wrong), `perms` how many drew a permutation (the floor).

    Example:
        >>> first, allc, dbr, first_ok, perms = blind_histogram(600, (0, 3, 4))
        >>> sorted(first), sum(first.values())
        ([0, 3, 4], 600)
        >>> 0.15 < perms / 600 < 0.30      # 3!/3**3 = 0.2222
        True
        >>> 0.55 < first_ok / 600 < 0.78   # 2/3
        True
    """
    import types

    k = max(occupied) + 1
    occ = np.zeros((1, k), dtype=bool)
    occ[0, list(occupied)] = True
    task = types.SimpleNamespace(occupied_mask=occ, watered_mask=np.zeros((1, k), dtype=bool))
    first = {int(s): 0 for s in occupied}
    allc = {int(s): 0 for s in occupied}
    dbr = {d: 0 for d in range(1, n_plants + 1)}
    first_ok = 0
    perms = 0
    for seed in range(int(n_seeds)):
        rng = np.random.default_rng(seed)
        draws = [choose_next_plant(task, True, rng) for _ in range(n_plants)]
        first[draws[0]] += 1
        for d in draws:
            allc[d] += 1
        watered, repeat_seen, distinct = set(), False, 0
        for i, d in enumerate(draws):
            already = d in watered
            if not already and not repeat_seen:
                distinct += 1
            if already:
                repeat_seen = True
            if i == 1:
                first_ok += int(not already)
            watered.add(d)
        dbr[distinct] += 1
        perms += int(len(set(draws)) == n_plants)
    return first, allc, dbr, first_ok, perms


def print_blind_histogram(n_seeds: int = 20000, n_plants: int = 3) -> None:
    """Print `blind_histogram` for three different occupied sets, with the expectations.

    The check `MikasaStationChecklist-v0` did not have. A constant arm, an arm keyed
    to the site *values*, or one that draws without replacement all show up here and
    nowhere else — the last of them as a permutation rate of 1.0 instead of 0.222.

    Example:
        >>> print_blind_histogram(600)  # doctest: +ELLIPSIS
        [blind draw] ...
    """
    exp_dbr = {1: 1 / 3, 2: 4 / 9, 3: 2 / 9}
    for occupied in ((0, 1, 2), (1, 2, 4), (2, 3, 4)):
        first, allc, dbr, first_ok, perms = blind_histogram(n_seeds, occupied, n_plants)
        n_all = sum(allc.values())
        print(f"[blind draw] occupied={list(occupied)}  seeds={n_seeds}", flush=True)
        print("             trip-0 draw  " + "  ".join(
            f"site {s}: {c:6d} ({c / n_seeds:.4f})" for s, c in sorted(first.items())
        ) + f"   expected {1 / len(occupied):.4f}", flush=True)
        print("             all trips    " + "  ".join(
            f"site {s}: {c:6d} ({c / n_all:.4f})" for s, c in sorted(allc.items())
        ) + f"   expected {1 / len(occupied):.4f}", flush=True)
        print("             distinct_before_repeat  " + "  ".join(
            f"{d}: {c:6d} ({c / n_seeds:.4f} vs {exp_dbr.get(d, 0.0):.4f})"
            for d, c in sorted(dbr.items())
        ), flush=True)
        print(f"             first_decision_ok  {first_ok:6d} ({first_ok / n_seeds:.4f})   "
              f"expected {2 / 3:.4f}", flush=True)
        print(f"             permutations       {perms:6d} ({perms / n_seeds:.4f})   "
              f"expected {math.factorial(n_plants) / n_plants ** n_plants:.4f}  <- THE FLOOR",
              flush=True)


# ------------------------------------------------------------ pure geometry --


def dock_frame(dock_xyyaw):
    """`(xyz, face, yaw)` of a dock stored as `(x, y, yaw)`.

    `rotate_base_z` asserts the view vector is horizontal, and an assertion is not
    a `-1`, so the z of `face` is zeroed by construction here rather than hoped for.

    Args:
        dock_xyyaw: `(x, y, yaw)` — a row of `task.site_dock` or `task.refill_dock`.

    Returns:
        `(xyz, face, yaw)`: float64 `(x, y, 0)`, the unit facing `(cos, sin, 0)`, and
        the yaw in radians.

    Example:
        >>> p, f, y = dock_frame([2.8614, -2.5068, np.pi / 2])
        >>> p.round(4).tolist(), f.round(6).tolist(), round(y, 4)
        ([2.8614, -2.5068, 0.0], [0.0, 1.0, 0.0], 1.5708)
    """
    d = np.asarray(dock_xyyaw, dtype=np.float64).reshape(-1)
    return (
        np.array([d[0], d[1], 0.0]),
        np.array([np.cos(d[2]), np.sin(d[2]), 0.0]),
        float(d[2]),
    )


def segment_clearance(a, b, point) -> float:
    """Shortest distance from `point` to the segment `a`-`b`, in the xy plane.

    Args:
        a, b: segment ends; only the first two components are read.
        point: the obstacle's centre.

    Example:
        >>> round(segment_clearance([0, 0], [2, 0], [1, 0.5]), 3)
        0.5
        >>> round(segment_clearance([0, 0], [2, 0], [3, 0]), 3)
        1.0
    """
    a = np.asarray(a, dtype=np.float64)[:2]
    b = np.asarray(b, dtype=np.float64)[:2]
    s = np.asarray(point, dtype=np.float64)[:2]
    d = b - a
    L = float(np.linalg.norm(d))
    if L < 1e-9:
        return float(np.linalg.norm(s - a))
    t = float(np.clip(np.dot(s - a, d) / (L * L), 0.0, 1.0))
    return float(np.linalg.norm(s - (a + t * d)))


def station_clearance_ok(here, target, station_xy) -> tuple[bool, float]:
    """`(clears, distance)` for a straight base leg passing the water station.

    A **tripwire, not a router.** With the station where it now stands every leg in
    this task clears the bucket comfortably (0.5465-0.5500 m pre-K60; K60 moved the
    station 0.20 m further out, so today's legs clear by ~0.75 m at their closest
    approach), so nothing needs to steer around it; this exists so that if the
    station, the docks or the site layout ever move back into a bad arrangement, the
    drive says so in the trace instead of coming back as an unexplained
    `collision base_link<->water_bucket`.

    Known blind spot, found by the K61 review and left open on a measurement: this
    measures against the **bucket centre**, and since K60 the bucket stands on a
    stand whose AABB corner reaches 0.3846 m from the station centre — so a leg
    passing 0.42-0.67 m from the station toward a corner would satisfy the tripwire
    and still clip the stand. No leg in the shipped layout does (minimum
    leg-to-stand distance 0.3149 m against the 0.2876 m hull happens at the dock
    endpoint, not mid-leg, and every mid-leg approach is ~0.75 m); if the layout
    ever moves, widen this check to the stand's rectangle, not just the bucket.

    What it replaced, and why the measurement is kept: until the station was relocated
    the bucket sat between the refill dock and the counters, `drive_base` moves in a
    straight line, and two legs of five passed **below** the collision band —

        plant    0       1       2       3       4
        before   0.456   0.365   0.231   0.429   0.498    two below 0.4132
        after    0.550   0.550   0.550   0.550   0.546    none

    **Below, not inside**, and the preposition is load-bearing: 0.4132-0.4156 m is the
    *threshold's* uncertainty, not a region legs sit in. Zero of the five lie within it;
    two lie under it — which is what collides, and what `station_clearance_ok` and the
    test's `v < _NOGO_LO` both actually test.

    **Two, not three.** Three was the count under the retired `STATION_CLEARANCE = 0.45`,
    a threshold chosen before the base hull was measured; against the measured band only
    plants 1 and 2 fall below it, and two is also what was actually observed to collide
    (seed 12 on plant 1, seed 11 on plant 2). Plant 3 at 0.429 sits between the old
    threshold and the band and was never driven under the old placement.

    — so the oracle carried a `station_detour` that inserted a sidestep waypoint. That
    machinery is gone rather than left inert: it was a workaround for a station point
    chosen without asking whether the robot could drive away from it, and the point was
    fixed instead. Its own defect is recorded too, because it is the kind that repeats:
    the recovery rungs recomputed the detour from an already-sidestepped base, so each
    rung stepped further aside and walked `d_dock` from 1.260 to 1.439 to 1.625.

    Args:
        here: the base's current position.
        target: where the drive is going.
        station_xy: the bucket's centre.

    Returns:
        `(clears, distance)` — `clears` is False once the leg passes closer than the
        upper edge of `STATION_NOGO`.

    Example:
        >>> station_clearance_ok([2.9619, -1.6732], [2.4542, -1.27], [3.15, -2.19])[0]
        True
        >>> ok, d = station_clearance_ok([2.8614, -2.3768], [2.4542, -1.27], [2.8614, -1.7068])
        >>> ok, round(d, 3)          # plant 2, under the placement this replaced
        (False, 0.231)
    """
    d = segment_clearance(here, target, station_xy)
    return d >= STATION_NOGO[1], d


def tilt_axis_vec(name: str, face) -> np.ndarray:
    """The horizontal world axis named by an entry of `POUR_TILT_AXES`.

    Args:
        name: `"lateral+"`, `"lateral-"`, `"facing+"` or `"facing-"`.
        face: the base's unit facing, z = 0.

    Example:
        >>> tilt_axis_vec("lateral+", np.array([0.0, 1.0, 0.0])).round(6).tolist()
        [-1.0, 0.0, 0.0]
        >>> tilt_axis_vec("facing-", np.array([0.0, 1.0, 0.0])).round(6).tolist()
        [-0.0, -1.0, -0.0]
    """
    f = np.asarray(face, dtype=np.float64)
    f = np.array([f[0], f[1], 0.0])
    f = f / np.linalg.norm(f)
    lateral = np.array([-f[1], f[0], 0.0])
    base = {"lateral": lateral, "facing": f}[name[:-1]]
    return base * (1.0 if name[-1] == "+" else -1.0)


def level_quat(q) -> np.ndarray:
    """`q` rotated by the smallest arc that puts its body +Z back on world +Z.

    K55. The pour used to be built on the cup's *current* orientation under the name
    `q_upright`, so every tip was measured from whatever tilt the cup had already
    accumulated rather than from level. Two consequences, both measured on seed 0:
    the tilt ratcheted (0.0 -> 10.9 -> 31.7 -> 49.0 deg across the stages, nothing
    ever removing it) and the commanded tip had to overshoot to compensate
    (`POUR_TILT_COMMANDS` starts at 100 deg for a 55 deg threshold, and 140 deg
    landed at 108 deg). Against a genuinely level reference the tip is absolute and
    the overshoot is unnecessary.

    Minimal arc, so the cup's yaw about its own axis is left alone — the pour
    predicate does not care about it and turning it costs rotation for nothing.

    Args:
        q: `(w, x, y, z)` of the cup as it is now.

    Returns:
        float64 `(w, x, y, z)` with the body +Z vertical.

    Example:
        >>> tipped = tilt_quat([1.0, 0, 0, 0], [1.0, 0, 0], 40.0)
        >>> R = sapien.Pose(q=level_quat(tipped)).to_transformation_matrix()[:3, :3]
        >>> float(np.round(R[2, 2], 6))       # body +Z is back on world +Z
        1.0
        >>> float(np.round(sapien.Pose(q=level_quat([1.0, 0, 0, 0])).q[0], 6))
        1.0
    """
    q = np.asarray(q, dtype=np.float64)
    R = sapien.Pose(q=q).to_transformation_matrix()[:3, :3]
    v = R[:, 2]
    v = v / np.linalg.norm(v)
    up = np.array([0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(v, up), -1.0, 1.0))
    if dot > 1.0 - 1e-12:
        return q
    axis = np.cross(v, up)
    n = np.linalg.norm(axis)
    if n < 1e-9:
        # Exactly inverted: the minimal arc is undefined, every horizontal axis is
        # equally short. Pick one rather than divide by zero.
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = axis / n
    half = math.acos(dot) / 2.0
    q_corr = np.array([np.cos(half), *(np.sin(half) * axis)], dtype=np.float64)
    return np.asarray(
        (sapien.Pose(q=q_corr) * sapien.Pose(q=q)).q, dtype=np.float64
    )


def tilt_quat(q_upright, axis, degrees: float) -> np.ndarray:
    """`q_upright` tipped `degrees` about the world axis `axis`.

    The pour predicate reads the cup's own +Z against world up (`pour_axis_body`,
    measured in Task 4), so a pre-rotation in the world frame is what tips it.

    Args:
        q_upright: `(w, x, y, z)` of the cup standing upright.
        axis: a unit world axis, horizontal for a tip.
        degrees: how far to tip.

    Returns:
        float64 `(w, x, y, z)`.

    Example:
        >>> q = tilt_quat([1.0, 0, 0, 0], [1.0, 0, 0], 90.0)
        >>> R = sapien.Pose(q=q).to_transformation_matrix()[:3, :3]
        >>> (np.round(R[:, 2], 6) + 0.0).tolist()   # the body +Z now lies in the horizon
        [0.0, -1.0, 0.0]
    """
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    half = np.radians(float(degrees)) / 2.0
    q_rot = np.array([np.cos(half), *(np.sin(half) * a)], dtype=np.float64)
    return np.asarray((sapien.Pose(q=q_rot) * sapien.Pose(q=np.asarray(q_upright, dtype=np.float64))).q,
                      dtype=np.float64)


def target_tcp(cup_pose: sapien.Pose, t_tcp_cup: sapien.Pose) -> sapien.Pose:
    """"Put the cup origin here, in this orientation" as a TCP goal.

    Args:
        cup_pose: the pose the *cup* must take.
        t_tcp_cup: `tcp.pose.inv() * cup.pose`, read at a still moment after the drive
            (the burner measured the cup shifting in the fingers during base rotations).

    Example:
        >>> p = target_tcp(sapien.Pose(p=[1.0, 0.0, 1.0]), sapien.Pose(p=[0.1, 0, 0]))
        >>> np.round(p.p.astype(np.float64), 4).tolist()
        [0.9, 0.0, 1.0]
    """
    return cup_pose * t_tcp_cup.inv()


# --------------------------------------------------------------- primitives --


def plan_to_joint_targets(env, planner, task, targets: dict, *, label: str,
                          tries: int = 2, planning_time: float = 8.0):
    """Plan and execute a **joint-space** move to `targets`, the base pinned. -1 or the 5-tuple.

    `static_manipulation` takes a TCP pose, and two of this task's stages are
    conditions on the *configuration* rather than on where the hand is: the carry
    pose the refill is served in (every joint within 0.15 rad of the `rest`
    keyframe) and the torso height the floor work needs. IK would satisfy a TCP goal
    at some other elbow, so those go straight to `plan_qpos` — the primitive
    `plan_pose` itself ends in — with the three root joints fixed exactly as
    `static_manipulation` fixes them.

    Two draws by default: RRTConnect is randomized and refuses the same reachable
    goal on one draw and finds it on the next (measured on the season dish's seed
    12). Nothing is executed twice — the first draw that plans is the one that runs.

    Args:
        env, planner: as everywhere.
        task: `env.unwrapped`.
        targets: `{joint_name: value}`; every other joint keeps its current value.
        label: stage name for the refusal line.
        tries: draws before it is booked as a refusal.
        planning_time: RRT budget per draw, seconds.

    Returns:
        The 5-tuple, or -1 after the last refused draw.

    Example:
        >>> res = plan_to_joint_targets(env, planner, task,
        ...     {"torso_lift_joint": 0.0}, label="lower the torso")   # doctest: +SKIP
    """
    if getattr(planner, "truncated", False):
        return planner.idle_steps(t=1)
    p = planner.planner
    robot = task.agent.robot

    # K55 / jezv. Do the vertical part with the torso, on its own, before asking the
    # arm for anything. `torso_lift_joint` is **prismatic**: it moves the hand up and
    # down at exactly zero rotation cost, where the arm can only change height by
    # turning. Asked for both at once the planner happily spends the arm on height —
    # the fold carries the torso 0.05 -> 0.386 m and costs 182 deg of hand rotation
    # doing it. Split, the torso leg is free and the arm leg only has to cover what
    # is left.
    #
    # This is not the same as freeing the torso inside a single plan, which was tried
    # and is worse (2557 -> 2616 deg, 1/2, tilt 8.9% -> 20.9%): there the planner may
    # spend the torso *and* the arm on the same move. Here the torso goes first and
    # alone, so the arm's share is strictly smaller.
    if TORSO_FIRST and "torso_lift_joint" in targets:
        i_torso = robot.active_joints_map["torso_lift_joint"].active_index[0].item()
        now = float(_np(robot.get_qpos()).reshape(-1)[i_torso])
        want = float(targets["torso_lift_joint"])
        if abs(want - now) > TORSO_FIRST_MIN_M:
            cur = _np(robot.get_qpos()).reshape(-1).astype(np.float64)
            goal = cur.copy()
            goal[i_torso] = want
            first = p.plan_qpos(
                [p.fold_qpos(p.pad_move_group_qpos(goal))],
                p.fold_qpos(p.pad_move_group_qpos(cur)),
                time_step=task.control_timestep,
                planning_time=planning_time,
                rrt_range=0.1,
                simplify=True,
                fixed_joint_indices=[0, 1, 2],
                ref_yaw=float(cur[2]),
            )
            # A "Success" with no knots is possible (the torso is already there to
            # within the planner's tolerance) and following it indexes [-1] into an
            # empty array. Cheap to check, and the whole leg is optional anyway.
            if first["status"] == "Success" and len(np.asarray(first["position"])) > 1:
                say(env, f"{label}: torso first", frm=round(now, 3), to=round(want, 3))
                # No refinement on this leg: the refiner nudges toward the *arm's*
                # goal and a torso-only path gives it nothing to work with (it
                # indexes [-1] into an empty target array). The arm's own plan below
                # is refined as before, so nothing is lost.
                r = planner.follow_forward_path_w_refinement(first, refine=False)
                if r != -1 and getattr(planner, "truncated", False):
                    return r

    for i in range(int(tries)):
        cur = _np(robot.get_qpos()).reshape(-1).astype(np.float64)
        goal = cur.copy()
        for n, v in targets.items():
            goal[robot.active_joints_map[n].active_index[0].item()] = float(v)
        # Already there: say so and step once, rather than asking RRTConnect for a path
        # from a configuration to itself.
        #
        # K59. `goal` starts as a copy of `cur` and only the *named* joints are written,
        # so when every named joint already holds its target the two are bitwise equal
        # and there is nothing to plan. The dip's "as it stands" rung asks for exactly
        # one joint — `{"torso_lift_joint": TORSO_DOWN}` — and the torso was already put
        # there by "lower the torso for the floor work", so on the first trip this is a
        # no-op every time. It was not free: RRTConnect samples the *whole* arm space
        # to reach a goal that constrains one joint, and from the contorted
        # configuration a 179-knot RRT lift leaves behind it can time out. That is both
        # of the sweep's two `RRTConnect Failed. Timeout`s (seed 6), and the cost was
        # the episode — a refused dip posture falls through to the `rest arm` rung,
        # which puts the held cup back on the floor and makes the hover's IK
        # unsolvable (`collision cup<->floor`, then `IK Failed`).
        #
        # `idle_steps(t=1)` is what the truncation guard above already returns for "no
        # motion needed": a real 5-tuple, one control step, no plan.
        if np.allclose(goal, cur, rtol=0.0, atol=JOINT_AT_TARGET_RAD):
            say(env, f"{label}: already there", joints=sorted(targets))
            return planner.idle_steps(t=1)
        # K61: the straight line in joint space first. For a joint-space goal the line
        # is the minimum-rotation path outright — box limits cannot be left on a
        # segment between two in-limit points, and no joint can wind — where RRTConnect
        # merely finds *a* path, and the w10 meter priced that difference at 16 159 deg
        # of tip rotation on the fold alone across the v9 sweep. Collision-checked knot
        # by knot (attached cup included); a blocked line refuses without executing and
        # the RRT below still runs, so nothing reachable becomes unreachable.
        line = p.plan_qpos_line(
            goal,
            cur,
            time_step=task.control_timestep,
            ref_yaw=float(cur[2]),
        )
        if line["status"] == "Success" and len(np.asarray(line["position"])) > 1:
            say(env, f"{label}: joint line", knots=int(np.asarray(line["position"]).shape[0]))
            return planner.follow_forward_path_w_refinement(line, refine=True)
        if line["status"] != "Success":
            # One line, because K62's seed-25 fold died with the *reason invisible*:
            # the blocked pair is the whole diagnosis (what the straight line sweeps
            # through decides whether RRT can plausibly do better) and it was being
            # swallowed on the way to the RRT fallback.
            say(env, f"{label}: {line['status']}")
        result = p.plan_qpos(
            [p.fold_qpos(p.pad_move_group_qpos(goal))],
            p.fold_qpos(p.pad_move_group_qpos(cur)),
            time_step=task.control_timestep,
            planning_time=planning_time,
            rrt_range=0.1,
            simplify=True,
            fixed_joint_indices=[0, 1, 2],
            ref_yaw=float(cur[2]),
        )
        if result["status"] == "Success":
            return planner.follow_forward_path_w_refinement(result, refine=True)
        say(env, f"{label}: plan refused", status=result["status"], draw=i + 1,
            gap=round(float(np.max(np.abs(goal - cur))), 5))
    return -1


def rest_arm_targets(task) -> dict:
    """`{joint_name: rest keyframe value}` for the seven arm joints and the torso.

    Not the head: `head_pan` and `head_tilt` are not in the planner's move group, so
    no plan can move them at all — see `hold_rest_body` and this module's docstring.

    Example:
        >>> sorted(rest_arm_targets(task))[:2]        # doctest: +SKIP
        ['elbow_flex_joint', 'forearm_roll_joint']
    """
    # K55. The base values come from the keyframe at **float64**, and only the
    # `carry_overrides` are layered on from the config — not from `task._carry_qpos`,
    # which is the same numbers stored as float32. That distinction is not pedantry:
    # reading the float32 copy shifts each goal by ~1e-8, and the RRT downstream is
    # chaotic enough that it re-rolls into a different path and a different verdict
    # (measured — seeds 0-1 went 2/2 to 1/2 on that change alone, with nothing else
    # different). Same source of truth for the *overrides*, full precision for the
    # values, and byte-identical behaviour to the pre-K55 oracle while overrides are
    # empty.
    rest = np.asarray(task.agent.keyframes["rest"].qpos, dtype=np.float64)
    joints = task.agent.robot.active_joints_map
    names = list(task.agent.controller.controllers["arm"].config.joint_names) + ["torso_lift_joint"]
    out = {n: float(rest[joints[n].active_index[0].item()]) for n in names}
    for _name, _val in task.cfg.carry_overrides:
        if _name in out:
            out[_name] = float(_val)
    return out


def fold_to_rest(env, planner, task, **kwargs):
    """The seven arm joints and the torso back to the `rest` keyframe. -1 or the 5-tuple.

    Example:
        >>> res = fold_to_rest(env, planner, task)                            # doctest: +SKIP
        >>> if res != -1 and common.stopped_by_horizon(planner): return res   # doctest: +SKIP
        >>> if res == -1: return fail(env, "fold to the rest keyframe")       # doctest: +SKIP
    """
    return plan_to_joint_targets(env, planner, task, rest_arm_targets(task),
                                 label="fold to the rest keyframe", **kwargs)


def hold_rest_body(env, planner, task, n_steps: int):
    """Hold the arm where it is and command head + torso to the `rest` keyframe.

    **Why this is not privileged, and not cheating.** It is the one hand-built action in
    this file, so the question deserves a straight answer rather than a footnote. The
    head is part of `ds_fetch`'s **action space** — `head_pan_joint` and
    `head_tilt_joint` are two of the three joints of the `body` controller, and every
    policy that drives this robot emits targets for them on every step. There is a
    precedent in this repository doing exactly this and doing it from inside a *task*:
    `station_checklist.py`'s `head_script` writes `act[:, self._head_pan_i]` and
    `act[:, self._head_tilt_i]` directly. Nothing here reads state a policy cannot read,
    and nothing here is easier for the oracle than for an agent: an agent that wants
    `carry_ok` commands its head back, exactly as this does. What the function supplies
    is a **motor competence the inherited solver lacks**, not information.

    **Why the solver lacks it.** `extand.py`'s three path followers —
    `follow_forward_path_w_refinement`, `follow_rotation`, `follow_moving_forward` — all
    write the body row as `body_action[0] = body_action[1] = 0.0`, so head_pan and
    head_tilt are driven to zero by **every executed plan**; `idle_steps` and
    `change_gripper_state` then hold whatever the head currently is, and the head joints
    are not in the planner's move group, so no plan can reach them either. The task's
    carry pose is read off the `rest` keyframe, where `head_tilt = 0.562`. Measured, seed
    11: one joint-space move of the torso alone, with nothing in the gripper, is enough
    to leave `carry_error = 0.5621, carry_ok = False`; after a full fold with the cup
    held, the seven arm joints and the torso are within **0.0076 rad** of rest and the
    head is the only thing over tolerance. Without this function `refilling` is never
    true, `has_water` is never set, and every episode of this task fails identically and
    silently with `budget_left = 3`.

    **When it retires.** The followers' zeroing is a defect and the fix belongs there —
    two lines in each of three methods. It is deliberately not made here: `extand.py` is
    shared by `MikasaBurner-v0`, `MikasaSeasonDish-v0` and `MikasaStationChecklist-v0`,
    and changing what a plan *executes* would change their trajectories and invalidate
    their published numbers and recorded demonstrations. That is a separate piece of
    work. **When it lands, delete this function** — the head will already be where the
    predicate wants it, and `planner.idle_steps` will serve the dwell.

    The action is built the way `idle_steps` builds it — arm targets from the arm
    controller's current qpos, the planner's own latched gripper state, base held — with
    only the body row changed, and it goes through the solver's own `_step`, so the
    horizon guard counts and latches these steps like any other.

    Args:
        env: the (possibly wrapped) env, for `say`.
        planner: the solver.
        task: `env.unwrapped`.
        n_steps: control steps to hold.

    Returns:
        The last 5-tuple, or None if `n_steps <= 0` or the horizon already struck.

    Example:
        >>> res = hold_rest_body(env, planner, task, 20)     # doctest: +SKIP
        >>> if res is not None and _b(res[-1], "carry_ok"): ...   # doctest: +SKIP
    """
    n = int(n_steps)
    if n <= 0 or getattr(planner, "truncated", False):
        return None
    ctrl = task.agent.controller.controllers
    robot = task.agent.robot
    rest = np.asarray(task.agent.keyframes["rest"].qpos, dtype=np.float64)
    body_action = np.array(
        [rest[robot.active_joints_map[j].active_index[0].item()]
         for j in ctrl["body"].config.joint_names],
        dtype=np.float64,
    )
    arm_action = _np(ctrl["arm"].qpos)[0].astype(np.float64)
    base_action = np.array([0.0, 0.0])
    res = None
    for _ in range(n):
        if planner.truncated:
            break
        res = planner._step(np.hstack([arm_action, planner.gripper_state, body_action, base_action]))
    return res


#: Yaw order `carry_pose` is re-tried with when a drive is refused. Rotating the list
#: is what makes the retry a *different* configuration rather than the same one drawn
#: again: `carry_pose` executes the first candidate that plans, and yaw 0 (the held
#: orientation) plans from the rest keyframe, so an unrotated retry reproduces the tuck
#: that just failed — measured, seed 11, where both attempts died identically at
#: `d_dock = 1.511, dyaw = 42.9`.
CARRY_RETRY_YAWS = (90.0, -90.0, 180.0, 0.0)


def drive_to(env, planner, task, dock_xyyaw, label: str, *, carry: bool = True,
             advance: float = 0.0, **extra):
    """Drive to a dock, with a three-rung recovery ladder. `(res, parked)`.

    **On the carry pose.** K52 makes `carry_pose` obligatory before a drive that carries
    something, and the reason is specific: `rotate_base_z` sweeps each turn's arc against
    the *attached* object, and an object on an outstretched arm genuinely sweeps through
    furniture. **Every drive in this oracle that carries the cup tucks it in first.**

    `carry=False` is passed exactly once, by the opening drive onto the refill dock, and
    it is not a K52 exception: the gripper is empty there, so there is nothing to sweep,
    and `carry_pose` cannot be used on an empty hand at all — it verifies the grasp after
    moving and reports a slipped object when it finds none. K52 has nothing to say about
    an empty hand. Every rung of the ladder is gated on `carry` for the same reason.

    (History, because it was measured and then measured away: for one round the drive to
    a plant also skipped the tuck, since tucking from the rest keyframe left `wrist_flex`
    on its stop and `move_base_forward` refused the drive — reproducibly, the identical
    retry reproducing `d_dock = 1.511, dyaw = 42.9`. That turned out to be the free-arm
    base plan and not the tuck: `BASE_PLAN_MASK` handed a base translation all fifteen
    joints. With `freeze_arm=True` the wrist is not the plan's to move, and the tuck is
    back on every carrying drive. `oracle_common.capture_refusal` still reads the honest
    K51 `rotation sweep hits …` text out of a refusal and chains it into rung 2.)

    **On the ladder.** Each rung is a *different* attempt, not another draw of the same
    one — measured, because an identical retry was measured to reproduce the failure
    exactly:

      1. as asked (tucked or not);
      2. `carry_pose` with the yaw candidates rotated (`CARRY_RETRY_YAWS`), then drive
         — **only when something is being carried**; with an empty gripper this rung is
         a fresh draw of the same route, which is worth having because the base plan is
         randomized;
      3. halve the leg.

    Rung 2 used to re-tuck regardless of `carry`, which broke the ladder for the one
    drive that passes `carry=False`: `carry_pose` moved the empty hand, failed its own
    `is_grasping` check, and returned "carry pose unreachable: object slipped out on the
    way in" — so a base-planning refusal was reported in the trace as a dropped cup, and
    rung 3 was unreachable.

    **Every rung plans from the pose rung 1 started at**, not from wherever the previous
    rung left the base. The retired `station_detour` did the opposite and it is exactly
    how a recovery becomes a liability: recomputing a route from an already-sidestepped
    base stepped further aside each time and walked `d_dock` from 1.260 to 1.439 to
    1.625 while the target never moved.

    The route is also checked against the water station before rung 1
    (`station_clearance_ok`) — a tripwire only. Since the station was relocated every
    leg clears the bucket by ~0.55 m; when it did not, two legs of five passed below the
    collision band and both were observed to drive into it.

    Rung 3 halves the translation. It was introduced against `move_base_forward`'s screw
    plan giving up part way — `joint limit at index [11]`, "0.790 of the twist left" —
    and that particular cause is now fixed at its source by `freeze_arm=True`, which
    takes the wrist out of a plan that only ever needed the base. The rung is kept
    because a shorter leg is a genuinely different problem for anything else that can
    stop a drive, and it costs nothing until rungs 1 and 2 have both failed.

    **On `advance`.** The base is driven `advance` metres *past* the dock, along what
    the dock faces, and `parked` is still judged against the dock itself. Since
    `at_plant` admits anything within `dock_radius` = 0.20 m, a small advance is reach
    bought inside the task's own tolerance rather than against it — and the pour needs
    it (see `POUR_DOCK_ADVANCE`).

    **A rung is spent on an under-shoot as well as on a refusal.** `move_base_forward`
    executes only the base part of its screw plan and deliberately passes no
    `goal_tolerance`, so a plan that stops short still returns Success; measured on both
    calibration seeds, the drive to the trip-1 plant came back non-`-1` having stopped
    0.427 m and 0.651 m from its dock. Parking is therefore checked from the state after
    every rung, and a short stop advances to the next rung instead of being booked as a
    miss on the spot.

    Returns `(res, parked)`: `res` is -1 on a refused carry or drive (already said
    why), else the last 5-tuple; `parked` is `d_dock <= dock_radius and
    |dyaw| <= dock_heading_deg` — read from the state, not from the return code. A base
    that never reaches the zone in three rungs returns `(last 5-tuple, False)`, which is
    a physical miss (D6) and not a failed plan.

    Example:
        >>> res, parked = drive_to(env, planner, task, dock, "refill dock")  # doctest: +SKIP
        >>> if res == -1: return res                                          # doctest: +SKIP
    """
    xyz, face, yaw = dock_frame(dock_xyyaw)
    # `advance` drives past the dock toward whatever it faces. `parked` is still judged
    # against the dock itself, so a target inside the zone stays inside the zone.
    target = xyz + face * float(advance)
    # Read ONCE, before rung 1. Every rung plans its route from here rather than from
    # wherever the previous rung left the base: a retry should retry the same route with
    # a different tuck, not a progressively worse one.
    start = _np(task.agent.base_link.pose.p).reshape(-1, 3)[0].astype(np.float64)
    station = _np(task.station_pos).reshape(-1, 2)[0].astype(np.float64)
    refusal = None
    if not TUCK:
        carry = False
    for rung in (1, 2, 3):
        if carry and rung in (1, 2):
            yaws = common.CARRY_YAWS_DEG if rung == 1 else CARRY_RETRY_YAWS
            # Tuck depth is the shipped 0.25 m on every rung. A shallower first tuck
            # (0.40 m) was the obvious economy — the tuck is the arm's largest single
            # cost, 119 deg each and twice per plant — and the base's rotation sweep
            # is genuinely collision-checked (K51), so it looked free. Measured at
            # n=6 it is **2/6 against 6/6**: the cup rides outside the base footprint
            # and the sweep refuses, or the drive arrives and the pour cannot reach.
            # Its apparently-low rotation figure is an artefact of short failed
            # episodes. `CARRY_AHEAD_SHALLOW` is kept so the experiment is repeatable.
            ahead = CARRY_AHEAD_DEEP
            # Not waypointed, and that is measured: the tuck is a large reorientation
            # and breaking it up executes more motion through tilted intermediates,
            # which loses the cup out of the fingers ("carry pose unreachable:
            # object slipped out on the way in", seed 0). Long moves benefit from
            # waypoints; this one does not.
            res = common.carry_pose(env, planner, task, task.cup, who=WHO,
                                    after=refusal, yaws_deg=yaws, upright=True,
                                    ahead=ahead, max_knots=HELD_MAX_KNOTS,
                                    knot_draws=HELD_KNOT_DRAWS)
            if res != -1 and common.stopped_by_horizon(planner):
                return res, False
            if res == -1:
                return res, False  # carry_pose said why, chained to the refusal
            planner.planner.update_from_simulation()

        legs = [target]
        if rung == 3:
            # Halve the leg. A shorter translation is a different problem for anything
            # the first two rungs could not get past — and it is halved from `start`,
            # the pose the *first* rung began at, not from wherever a failed rung left
            # the base. Recomputing a route from a drifted base is how the retired
            # detour walked `d_dock` from 1.260 to 1.439 to 1.625 across three rungs.
            mid = np.array([(start[0] + target[0]) / 2.0, (start[1] + target[1]) / 2.0, 0.0])
            legs = [mid, target]

        clears, d_station = station_clearance_ok(start, target, station)
        say(env, f"drive to {label}", dock=[round(float(v), 3) for v in target],
            rung=rung, legs=len(legs), d_station=round(d_station, 3), **extra)
        if not clears:
            # Loud, and not fatal: the planner is the authority on whether it collides.
            # This says the leg was already known to be a bad one before it refused.
            say(env, f"WARNING: the leg to {label} passes {d_station:.3f} m from the "
                     f"bucket, inside the measured no-go band {STATION_NOGO}", rung=rung)

        res = -1
        with common.capture_refusal() as cap:
            for i, leg in enumerate(legs):
                # freeze_arm=True (extand.py, BASE_ONLY_PLAN_MASK): a base translation
                # is spanned by root_x and root_y alone, and with the arm free every
                # drive refusal measured on this task was a wrist joint limit inside a
                # plan that never needed the arm. Opt-in, so the three shipped oracles
                # keep planning exactly as their recordings were made.
                res = planner.drive_base(
                    target_pos=leg,
                    target_view_vec=face if i + 1 == len(legs) else None,
                    freeze_arm=True,
                )
                if res == -1:
                    break
                planner.planner.update_from_simulation()
        if res != -1 and common.stopped_by_horizon(planner):
            return res, False

        if res == -1:
            d_dock, dyaw = common.dock_error(task, (xyz[0], xyz[1], yaw))
            refusal = cap.refusal
            say(env, f"drive to {label} refused", rung=rung, d_dock=round(d_dock, 3),
                dyaw_deg=round(dyaw, 1), swept_arc=refusal)
            if rung == 3:
                return fail(env, f"drive to {label}", d_dock=round(d_dock, 3),
                            dyaw_deg=round(dyaw, 1), swept_arc=refusal), False
            continue

        planner.planner.update_from_simulation()
        res = planner.idle_steps(t=SETTLE_AFTER_DRIVE)
        if res != -1 and common.stopped_by_horizon(planner):
            return res, False
        if res == -1:
            return fail(env, f"settle at {label}"), False

        # **The state, not the return code.** `move_base_forward` executes only the base
        # part of its screw plan, so its plan's FK endpoint says nothing about where the
        # robot stopped, and it deliberately passes no `goal_tolerance` for that reason
        # — a plan that ends short still reports Success. Measured on both calibration
        # seeds: the drive to the trip-1 plant came back non-`-1` having stopped 0.427 m
        # (seed 11) and 0.651 m (seed 12) from its dock, outside the 0.20 m zone. The
        # oracle used to book that as a miss immediately, which was honest but idle: an
        # under-shoot is exactly what the next rung is for.
        d_dock, dyaw = common.dock_error(task, (xyz[0], xyz[1], yaw))
        parked = d_dock <= task.cfg.dock_radius and dyaw <= task.cfg.dock_heading_deg
        say(env, f"parked at {label}", rung=rung, d_dock=round(d_dock, 3),
            dyaw_deg=round(dyaw, 1), in_zone=parked)
        if parked:
            return res, True
        if rung < 3:
            say(env, f"drive to {label} stopped short of the zone; next rung",
                rung=rung, d_dock=round(d_dock, 3), dyaw_deg=round(dyaw, 1))
            continue
        # D6: the drive ran and the base is where it is. A physical miss, not a failed
        # plan — the caller returns the last 5-tuple and the sweep books it `missed`.
        return res, False
    return res, False


def grasp_the_cup(env, planner, task):
    """Torso down, then a top grasp of the cup standing on the floor. `(res, grasped)`.

    Not the burner's OBB thin-side grasp: that one approaches horizontally, and this
    cup's collision hull tops out 0.095 m off the floor, where a horizontal wrist has
    nowhere to be — measured, seed 11: its reach plan is refused on the torso limit
    and IK finds nothing. W1 measured a *top* grasp of this same cup instance reachable at torso
    0.000/0.097/0.193 and at no higher torso, which is why `TORSO_DOWN` comes first
    and is not an economy: from the rest posture (torso 0.386) the grasp-pose plan is
    refused at one depth and the fingers close on air at the next.

    `Fetch.build_grasp_pose` stacks `[ortho, closing, approaching]`, so the approach
    axis is the TCP's +Z; the closing axis is put across the base's facing, and for a
    cup of revolution any horizontal closing does. The height is `GRASP_DEPTHS` below
    the top of the mesh's **world AABB, read after the settle** — see that constant for
    the three measurements behind it — and not `obb.primitive.extents[2]`, which is the
    OBB's third axis and for this mesh is not the vertical one (measured: OBB centre
    z 0.038, "half-height" 0.0595).

    Example:
        >>> res, grasped = grasp_the_cup(env, planner, task)   # doctest: +SKIP
    """
    say(env, "lower the torso for the floor work", torso=TORSO_GRASP)
    res = plan_to_joint_targets(env, planner, task, {"torso_lift_joint": TORSO_GRASP},
                                label="lower the torso")
    if res != -1 and common.stopped_by_horizon(planner):
        return res, False
    if res == -1:
        return fail(env, "lower the torso"), False
    planner.planner.update_from_simulation()

    approach = np.array([0.0, 0.0, -1.0])
    face = _np(task.agent.base_link.pose.to_transformation_matrix())[0][:3, 0]
    face = np.array([face[0], face[1], 0.0])
    face = face / np.linalg.norm(face)
    closing = np.cross(approach, face)
    closing = closing / np.linalg.norm(closing)

    for depth in GRASP_DEPTHS:
        # Read the mesh fresh each time: a refused attempt may have knocked the cup
        # over and moved it (measured — a failed close pushed it 0.10 m and dropped
        # its AABB top from 0.095 to 0.053), and the t = 0 read is not the one that
        # matters either, since the cup is dropped onto the floor at reset.
        mesh = task.cup.get_first_collision_mesh(to_world_frame=True)
        if mesh is None:
            return fail(env, "grasp the cup: no collision mesh"), False
        bounds = np.asarray(mesh.bounds, dtype=np.float64)
        cup_p = _np(task.cup.pose.p).reshape(-1, 3)[0].astype(np.float64)
        z_top = float(bounds[1][2])
        centre = np.array([cup_p[0], cup_p[1], z_top - float(depth)])
        grasp = task.agent.build_grasp_pose(approach, closing, centre)
        reach = grasp * sapien.Pose([0, 0, -0.12])
        say(env, "grasp the cup", depth=round(float(depth), 4),
            cup_z=round(float(cup_p[2]), 4), z_top=round(z_top, 4),
            d_base=round(float(np.linalg.norm(
                _np(task.agent.base_link.pose.p).reshape(-1, 3)[0][:2] - cup_p[:2])), 3),
            grasp=[round(float(v), 3) for v in grasp.p])

        res = common.arm_move(env, planner, reach, who=WHO,
                              stage=f"reach the cup (depth {depth})", tries=3,
                              legs=REACH_LEGS, disable_lift_joint=REACH_FREEZE_TORSO)
        if res != -1 and common.stopped_by_horizon(planner):
            return res, False
        if res == -1:
            say(env, "reach refused", depth=round(float(depth), 4))
            continue

        # The descent from the reach pose is tried with the torso FROZEN first. A
        # vertical TCP move is what `plan_screw` spends the torso on, and the torso is
        # near its floor here, so with it free the screw reports `joint limit at
        # index [3]` before it has moved and falls through to a randomized IK that has
        # also refused at this pose. Frozen, the same descent is the arm's to make.
        planned = False
        for frozen in (True, False):
            res = common.arm_move(env, planner, grasp, who=WHO,
                                  stage=f"grasp pose (depth {depth}, torso "
                                        f"{'frozen' if frozen else 'free'})",
                                  disable_lift_joint=frozen, tries=3)
            if res != -1 and common.stopped_by_horizon(planner):
                return res, False
            if res != -1:
                planned = True
                break
        if not planned:
            say(env, "grasp pose refused with the torso both frozen and free",
                depth=round(float(depth), 4))
            planner.planner.update_from_simulation()
            continue

        res = planner.close_gripper()
        if res != -1 and common.stopped_by_horizon(planner):
            return res, False
        grasped = bool(_np(task.agent.is_grasping(task.cup)).any())
        if grasped:
            say(env, "cup in the gripper", depth=round(float(depth), 4))
            return res, True
        moved = float(np.linalg.norm(
            _np(task.cup.pose.p).reshape(-1, 3)[0][:2] - cup_p[:2]))
        tcp_p = _np(task.agent.tcp.pose.p).reshape(-1, 3)[0]
        say(env, "grasp attempt did not hold: fingers closed on air",
            depth=round(float(depth), 4), cup_moved=round(moved, 4),
            d_tcp_cup=round(float(np.linalg.norm(
                tcp_p - _np(task.cup.pose.p).reshape(-1, 3)[0])), 4))
        planner.open_gripper()
        planner.planner.update_from_simulation()
    return fail(env, "grasp the cup: no height held the cup",
                tried=[round(float(v), 4) for v in GRASP_DEPTHS]), False


def refill(env, planner, task, info, t_tcp_cup, trip: int):
    """Dip, fold to the rest keyframe, dwell. `(res, filled)`.

    Three separate moments and the task tells them apart: `dip_done` latches only at
    the dock and is cleared the moment the base leaves it, so a dip cannot be banked
    for the next trip; the dwell is redrawn per trip; and the fill only lands with
    the arm back in the carry pose.

    Example:
        >>> res, filled = refill(env, planner, task, info, t_tcp_cup, trip=0)  # doctest: +SKIP
    """
    bucket_p = _np(task.bucket.pose.p).reshape(-1, 3)[0].astype(np.float64)
    bucket_top = float(bucket_p[2] + task._bucket_top_lift)
    base_p = _np(task.agent.base_link.pose.p).reshape(-1, 3)[0].astype(np.float64)
    toward_base = base_p[:2] - bucket_p[:2]
    toward_base = toward_base / max(float(np.linalg.norm(toward_base)), 1e-9)
    aim = bucket_p[:2] + toward_base * DIP_PULLBACK

    # Two starting postures, tried in order, because **which arm configuration the trip
    # arrives in decides whether the bucket can be reached at all** — measured on seed
    # 11, and measured in both directions, which is why this is a ladder and not a
    # choice. Trip 0 arrives from the lift and its hover plans; trip 1 arrives after a
    # drive whose carry tuck came back at yaw 90 with the wrist turned, and the dip is
    # refused with the torso equally low. Normalising every trip to the rest arm fixes
    # trip 1 and *breaks trip 0*, whose hover is then refused. So: keep the arm as it
    # is and only lower the torso first, and fall back to the rest arm if that fails.
    #
    # The torso must be down either way — forward reach at z = 0.25 m is 0.980 m at
    # torso 0.000 and 0.549 m at 0.386 (W1), and the bucket is 0.80 m out from the dock.
    postures = ("as it stands", "rest arm")
    dipped = False
    for posture in postures:
        targets = {"torso_lift_joint": TORSO_DOWN}
        if posture == "rest arm":
            targets = {**rest_arm_targets(task), "torso_lift_joint": TORSO_DOWN}
        say(env, "into the dip posture", trip=trip, posture=posture, torso=TORSO_DOWN)
        res = plan_to_joint_targets(env, planner, task, targets,
                                    label=f"the dip posture ({posture})")
        if res != -1 and common.stopped_by_horizon(planner):
            return res, False
        if res == -1:
            say(env, "dip posture refused", trip=trip, posture=posture)
            continue
        planner.planner.update_from_simulation()

        cup_q = _np(task.cup.pose.q).reshape(-1, 4)[0].astype(np.float64)
        # Above the band first, then straight down into it: a direct move from the floor
        # beside the bucket is a straight line through its wall (measured, seed 11).
        hover_cup = sapien.Pose(p=np.array([aim[0], aim[1], bucket_top + DIP_HOVER]), q=cup_q)
        say(env, "hover over the bucket", trip=trip, posture=posture,
            cup_target=[round(float(v), 3) for v in hover_cup.p],
            bucket_top=round(bucket_top, 3),
            d_base=round(float(np.linalg.norm(base_p[:2] - aim)), 3))
        res = common.arm_move(env, planner, target_tcp(hover_cup, t_tcp_cup), who=WHO,
                              max_knots=HELD_MAX_KNOTS, knot_draws=HELD_KNOT_DRAWS,
                              stage=f"hover over the bucket ({posture})",
                              legs=WAYPOINT_LEGS)
        if res != -1 and common.stopped_by_horizon(planner):
            return res, False
        if res == -1:
            say(env, "hover over the bucket refused", trip=trip, posture=posture)
            continue
        # Possession again, for the same D6 reason as the lift. K60, seed 19: the cup
        # left the fingers *during the hover*, so the lift's guard had already passed;
        # the dip then refused at all three clearances with `cup_z_now=0.483` against a
        # `tcp_z_now=0.88` — the cup back on the stand and the gripper empty 0.4 m above
        # it — and the episode was booked `no plan found (-1)`. A dropped cup is a miss
        # at whatever stage it is noticed, and the last stage that touched it is the
        # honest place to say so.
        if not _b(res[-1], "is_grasped"):
            cup_p = _np(task.cup.pose.p).reshape(-1, 3)[0]
            tcp_p = _np(task.agent.tcp.pose.p)[0]
            say(env, "MISSED: the cup left the gripper during the hover", trip=trip,
                posture=posture, cup_z=round(float(cup_p[2]), 3),
                tcp_z=round(float(tcp_p[2]), 3),
                d_tcp=round(float(np.linalg.norm(cup_p - tcp_p)), 3))
            return res, False
        planner.planner.update_from_simulation()

        for clearance in DIP_CLEARANCES:
            dip_cup = sapien.Pose(p=np.array([aim[0], aim[1], bucket_top + float(clearance)]),
                                  q=cup_q)
            say(env, "dip into the bucket", trip=trip, posture=posture,
                clearance=round(float(clearance), 3),
                cup_target=[round(float(v), 3) for v in dip_cup.p])
            # Torso frozen for the descent, then free: a vertical TCP move is what
            # plan_screw spends the torso on, and the torso is at its floor here.
            for frozen in (True, False):
                res = common.arm_move(env, planner, target_tcp(dip_cup, t_tcp_cup), who=WHO,
                                      max_knots=HELD_MAX_KNOTS, knot_draws=HELD_KNOT_DRAWS,
                                      stage=f"dip {clearance} (torso {'frozen' if frozen else 'free'})",
                                      disable_lift_joint=frozen, legs=WAYPOINT_LEGS)
                if res != -1 and common.stopped_by_horizon(planner):
                    return res, False
                if res != -1:
                    dipped = True
                    break
            if dipped:
                break
            # What the *simulator* says the cup is doing, next to the goal the planner
            # just refused. The refusals here name `collision cup<->floor`, and whether
            # that verdict is real is the whole question: with the cup ~9.5 cm tall, a
            # goal of `bucket_top + 0.20` puts its base ~0.227 m up and it cannot touch
            # the floor — unless the cup is not where the goal says, i.e. the arm never
            # left the floor (real) or the planning world's attached-cup transform has
            # drifted from the simulator's (phantom). One line settles which, and all
            # three of the v6 sweep's losses were here. TODOS carries the card.
            _cp = _np(task.cup.pose.p).reshape(-1, 3)[0]
            _tcp = _np(task.agent.tcp.pose.p)[0]
            say(env, "dip refused with the torso both frozen and free", trip=trip,
                posture=posture, clearance=round(float(clearance), 3),
                cup_z_now=round(float(_cp[2]), 3),
                cup_z_goal=round(float(bucket_top + clearance), 3),
                tcp_z_now=round(float(_tcp[2]), 3),
                d_bucket=round(float(np.linalg.norm(_cp[:2] - bucket_p[:2])), 3),
                grasped=_b(res[-1], "is_grasped") if res != -1 else None)
        if dipped:
            break
    if not dipped:
        return fail(env, "dip into the bucket", trip=trip, postures=list(postures),
                    clearances=list(DIP_CLEARANCES)), False

    res = planner.idle_steps(t=6)
    if res != -1 and common.stopped_by_horizon(planner):
        return res, False
    if res == -1:
        return fail(env, "settle over the bucket", trip=trip), False
    info = res[-1]
    cup_p = _np(task.cup.pose.p).reshape(-1, 3)[0]
    say(env, "dipped", trip=trip,
        d_bucket=round(float(np.linalg.norm(cup_p[:2] - bucket_p[:2])), 3),
        clearance=round(float(cup_p[2] - bucket_top), 3),
        over_bucket=_b(info, "over_bucket"), dip_done=_b(info, "dip_done"),
        at_station=_b(info, "at_station"), is_grasped=_b(info, "is_grasped"))
    if not _b(info, "dip_done"):
        # A physical miss, not a plan failure: the arm went where it was sent.
        say(env, "MISSED: the cup reached the dip pose and dip_done did not latch", trip=trip)
        return res, False

    planner.planner.update_from_simulation()
    say(env, "fold to the rest keyframe", trip=trip)
    res = fold_to_rest(env, planner, task)
    if res != -1 and common.stopped_by_horizon(planner):
        return res, False
    if res == -1:
        return fail(env, "fold to the rest keyframe", trip=trip), False
    planner.planner.update_from_simulation()
    if not bool(_np(task.agent.is_grasping(task.cup)).any()):
        # A physical miss, not a plan failure — the same call as `dip_done` fourteen
        # lines above, and `res` here is the valid 5-tuple `fold_to_rest` returned: the
        # fold planned and executed, and the cup came out of the fingers anyway. `-1`
        # would print "(planner returned -1)", which is untrue, and book the episode as
        # `no_plan`; `evaluate_planner.py:704-707` appends to `infos` only for 5-tuples,
        # so the episode's `distinct_before_repeat`, `first_decision_ok` and
        # `double_water_count` would be deleted from the very tally the floor comparison
        # is built on (this file's own argument at the pour, D6). No published sweep
        # reached this branch, so no measured number depends on which way it went — but
        # it is latent censorship of the memory statistic, and that is why it is `res`.
        say(env, "MISSED: the cup left the gripper during the fold", trip=trip,
            is_grasped=_b(res[-1], "is_grasped"), at_station=_b(res[-1], "at_station"))
        return res, False

    dwell = int(_np(info["refill_dwell"]).reshape(-1)[trip])
    budget = dwell + REFILL_EXTRA_STEPS
    say(env, "dwell in the carry pose", trip=trip, dwell=dwell, budget=budget)
    spent = 0
    while spent < budget:
        step = hold_rest_body(env, planner, task, min(10, budget - spent))
        if step is None:
            break
        spent += min(10, budget - spent)
        res = step
        if common.stopped_by_horizon(planner):
            return res, False
        if _b(res[-1], "has_water"):
            say(env, "filled", trip=trip, steps=spent,
                carry_error=round(_f(res[-1], "carry_error"), 4))
            return res, True
    last = res[-1]
    say(env, "MISSED: the dwell ran out with the cup still empty", trip=trip,
        carry_error=round(_f(last, "carry_error"), 4), carry_ok=_b(last, "carry_ok"),
        at_station=_b(last, "at_station"), dip_done=_b(last, "dip_done"),
        refill_timer=int(_f(last, "refill_timer")), dwell=dwell,
        is_grasped=_b(last, "is_grasped"))
    return res, False


def water(env, planner, task, s: int, t_tcp_cup, trip: int):
    """Tip the cup over plant `s` and hold. `(res, poured)`.

    The pose is built on the *cup*, not on the hand: origin over the pot's mesh top
    inside `pour_min/max_clearance`, within `pour_xy_radius` in xy, and the cup's own
    +Z tipped at least `cfg.pour_tilt_deg` off vertical (the axis the tilt is measured
    on is the cup's own +Z, `pour_axis_body`, Task 4).

    All three ladders were measured to be needed: the clearance one because the pose at
    +0.15 is refused with the torso at its ceiling, the tilt-axis one because a tip about
    the wrong horizontal axis asks the wrist for a turn it does not have, and the
    commanded-tilt one because **the cup turns back inside the fingers as it tips** — see
    `POUR_TILT_COMMANDS`. The tilt that counts is read from `info["tilt_rad"]` after the
    move, never inferred from the plan.

    **The two ways this stage ends without a pour are not the same thing** (D6). No
    combination *planning* is an honest `-1`: the oracle never got the cup over the pot
    and never reached a decision. A tip that plans, executes and leaves the cup short of
    `pour_tilt_deg` is a **miss** — the arm went exactly where it was sent and the grasp
    lost the tilt — so it returns the last 5-tuple with a `MISSED:` row. The branch at
    the end of this function says why that distinction decides what a sweep can measure.

    Example:
        >>> res, poured = water(env, planner, task, 2, t_tcp_cup, trip=0)  # doctest: +SKIP
    """
    # Read before anything moves. The tip's own settle can be long enough for
    # `pour_hold` to reach `hold_steps` and commit, so reading this after the ladder
    # reports every first pour as a repeat — which is worse than not reporting it.
    watered_before = bool(_np(task.watered_mask).reshape(-1)[s])
    plant_p = _np(task.plants[s].pose.p).reshape(-1, 3)[0].astype(np.float64)
    pot_top = float(plant_p[2] + task._plant_top_lift)
    base_p = _np(task.agent.base_link.pose.p).reshape(-1, 3)[0].astype(np.float64)
    face = _np(task.agent.base_link.pose.to_transformation_matrix())[0][:3, 0]
    face = np.array([face[0], face[1], 0.0])
    face = face / np.linalg.norm(face)
    # K55: genuinely upright, not "however the cup is sitting now". With the old
    # reading the commanded tip was relative to an accumulated tilt, so the same
    # command meant a different absolute angle on every trip.
    q_upright = level_quat(_np(task.cup.pose.q).reshape(-1, 4)[0].astype(np.float64))

    toward_base = base_p[:2] - plant_p[:2]
    toward_base = toward_base / max(float(np.linalg.norm(toward_base)), 1e-9)
    aim = plant_p[:2] + toward_base * POUR_PULLBACK

    # Straight to the tipped pose — there is no upright hover, and that is a measured
    # choice rather than an economy. An upright hover asks the gripper to point
    # straight **down** at the far end of the arm's reach, which is the hardest
    # orientation there is: on seed 11 it was refused at every clearance in the band
    # (`joint limit at index [3]`, the torso at its ceiling, then `IK Failed`). Tipped
    # the gripper points nearly horizontally, which is what a Fetch arm at extension
    # does naturally. No predicate asks for an upright hover, and the RRT
    # collision-checks the path with the cup attached, so the pot is protected by the
    # planner rather than by a waypoint.
    #
    # Three ladders, and each rung answers a different question: the clearance (the
    # lowest legal pose is the nearest and so the most likely to plan), the tilt axis
    # (each asks the wrist for a different turn), and the commanded tilt (the cup turns
    # in the fingers, so what is asked for is not what is reached).
    #
    # **The command is innermost, and the order is the fix for a real defect.** The two
    # failures have different remedies: a tip that will not *plan* wants a different axis
    # or clearance, and a tip that plans, executes and comes back *short* wants a bigger
    # command on the same axis. With the command outermost, four axes that all executed
    # at command 100 spent the whole tip budget before 120 was ever tried. A refused plan
    # executes nothing, so only executed tips count against `POUR_MAX_TIPS`.
    need = task.cfg.pour_tilt_deg + POUR_TILT_MARGIN_DEG
    target_p, tipped, executed, reached = None, False, 0, -1.0
    # The last 5-tuple from a tip that actually ran. `res` cannot serve: a later
    # combination that refuses leaves it at `-1`, and the miss diagnostics below would
    # then subscript an int. Kept separately so "what the cup did" survives a refusal
    # that happened after it.
    last_ok = None
    # K61: the reach and the tilt are two moves, not one. The single combined goal —
    # pour position *and* 65+ deg of reorientation — is exactly the request K57
    # measured the screw refusing (`joint limit at index [3]`, 1.581 rad of twist in
    # one screw), so nearly every tip fell to RRT, and the w10 meter priced that at
    # 2153 deg executed for 585 deg net on the matched pilot — the reach to the pot
    # wandering, not the tilt. Split, each half is a request the screw accepts:
    #
    #   1. **Reach level.** The cup goes to the pour position still upright — the
    #      same kind of move as the hover over the bucket, which plans at a 15 %
    #      refusal rate against the tip's 75 %.
    #   2. **Tilt in place.** Pure rotation at a fixed TCP, walked in
    #      `POUR_BITE_DEG` bites re-derived from the cup's measured orientation —
    #      K58's own numbers: every 21-33 deg correction planned, every 65-70 deg
    #      one refused. `level_in_place` has poured this exact foundation since K58;
    #      this is the same mechanism run in the other direction.
    #
    # A refused level reach falls through to the old combined goal, so nothing that
    # planned before stops planning; the ladder below is unchanged as the fallback.
    def _settle_and_read(after, command):
        """Settle, verify possession, read the cup's tilt. One tip's bookkeeping.

        Shared by the bite ladder and the combined-goal fallback so the two paths
        cannot drift apart on the K58 possession guard or the miss diagnostics.
        Returns ("cut"|"drop"|"fail", res) to bubble up, or ("read", res) with
        `reached` refreshed.
        """
        nonlocal last_ok, reached
        r = planner.idle_steps(t=4)
        if r != -1 and common.stopped_by_horizon(planner):
            return "cut", r
        if r == -1:
            return "fail", fail(env, "settle at the pour pose", site=s, trip=trip)
        last_ok = r
        # **Possession before tilt** (K58): `tilt_rad` is read from the cup actor
        # wherever it is — a dropped cup lying on its side reads ~86 deg and sails
        # past `need`. The fingerprint of the four K58 episodes was `lost_deg=-21.3`;
        # a cup cannot over-rotate its command, it can only be on the floor.
        if not _b(r[-1], "is_grasped"):
            cup_p = _np(task.cup.pose.p).reshape(-1, 3)[0]
            say(env, "MISSED: the cup left the gripper during the tip",
                site=s, trip=trip, tips=executed, commanded_deg=command,
                cup_z=round(float(cup_p[2]), 3),
                d_pot=round(float(np.linalg.norm(cup_p[:2] - plant_p[:2])), 3),
                knots_hint="a long RRT tip path is the measured risk factor")
            return "drop", r
        reached = float(np.degrees(_f(r[-1], "tilt_rad")))
        say(env, "tipped", site=s, trip=trip, commanded_deg=command,
            reached_deg=round(reached, 1), need_deg=round(need, 1),
            lost_deg=round(command - reached, 1), tips=executed)
        return "read", r

    # K61: the reach and the tilt are two moves, not one. The single combined goal —
    # pour position *and* 65+ deg of reorientation — is exactly the request K57
    # measured the screw refusing (1.581 rad of twist in one screw), so nearly every
    # tip fell to RRT, and the w10 meter priced that at 2153 deg executed for 585 deg
    # net on the matched pilot: the *reach* wanders, not the tilt. Split, each half is
    # a request the screw accepts — the level reach is the same kind of move as the
    # bucket hover (15 % refusal against the tip's 75 %), and the in-place tilt is
    # walked in `POUR_BITE_DEG` bites, K58's measured window (21-33 deg planned,
    # 65-70 refused) run in the other direction. The combined ladder below survives
    # as the fallback, so nothing that planned before stops planning.
    for clearance in POUR_CLEARANCES:
        if tipped or executed >= POUR_MAX_TIPS:
            break
        target_p = np.array([aim[0], aim[1], pot_top + float(clearance)])
        say(env, "reach the pour position (level)", site=s, trip=trip,
            cup_target=[round(float(v), 3) for v in target_p],
            clearance=round(float(clearance), 3))
        res = common.arm_move(
            env, planner, target_tcp(sapien.Pose(p=target_p, q=q_upright), t_tcp_cup),
            who=WHO, stage=f"pour reach ({clearance})", tries=1,
            max_knots=POUR_MAX_KNOTS, knot_draws=POUR_KNOT_DRAWS)
        if res != -1 and common.stopped_by_horizon(planner):
            return res, False
        if res == -1:
            continue
        if not _b(res[-1], "is_grasped"):
            cup_p = _np(task.cup.pose.p).reshape(-1, 3)[0]
            say(env, "MISSED: the cup left the gripper during the pour reach",
                site=s, trip=trip, cup_z=round(float(cup_p[2]), 3))
            return res, False
        for axis_name in POUR_TILT_AXES:
            tilted_far = 0.0
            for command in pour_tilt_commands(task):
                while tilted_far < command:
                    step_to = min(tilted_far + POUR_BITE_DEG, command)
                    q = tilt_quat(q_upright, tilt_axis_vec(axis_name, face), step_to)
                    say(env, "tip the cup over the plant", site=s, trip=trip,
                        cup_target=[round(float(v), 3) for v in target_p],
                        pot_top=round(pot_top, 3), clearance=round(float(clearance), 3),
                        tilt_axis=axis_name, commanded_deg=step_to,
                        d_base=round(float(np.linalg.norm(base_p[:2] - aim)), 3))
                    bite = common.arm_move(
                        env, planner, target_tcp(sapien.Pose(p=target_p, q=q), t_tcp_cup),
                        who=WHO, stage=f"pour bite ({clearance}, {step_to}, {axis_name})",
                        tries=1, disable_lift_joint=True,
                        max_knots=POUR_MAX_KNOTS, knot_draws=POUR_KNOT_DRAWS)
                    if bite != -1 and common.stopped_by_horizon(planner):
                        return bite, False
                    if bite == -1:
                        break
                    res = bite
                    tilted_far = step_to
                if tilted_far < command:
                    break  # a bite refused; try the next axis (absolute goals, so
                           # whatever tilt this axis left is absorbed by the next)
                executed += 1
                verdict, res = _settle_and_read(res, command)
                if verdict == "cut":
                    return res, False
                if verdict == "fail":
                    return res, False
                if verdict == "drop":
                    return res, False
                if reached >= need:
                    tipped = True
                    break
                if executed >= POUR_MAX_TIPS:
                    break
            if tipped or executed >= POUR_MAX_TIPS:
                break
        if tipped or executed >= POUR_MAX_TIPS:
            break

    # The combined-goal ladder, now the fallback: exactly the pre-K61 behaviour, run
    # only when the split above neither tipped nor spent the budget.
    if not tipped and executed < POUR_MAX_TIPS:
        for clearance in POUR_CLEARANCES:
            for axis_name in POUR_TILT_AXES:
                for command in pour_tilt_commands(task):
                    target_p = np.array([aim[0], aim[1], pot_top + float(clearance)])
                    q = tilt_quat(q_upright, tilt_axis_vec(axis_name, face), command)
                    say(env, "tip the cup over the plant", site=s, trip=trip,
                        cup_target=[round(float(v), 3) for v in target_p],
                        pot_top=round(pot_top, 3), clearance=round(float(clearance), 3),
                        tilt_axis=axis_name, commanded_deg=command,
                        d_base=round(float(np.linalg.norm(base_p[:2] - aim)), 3))
                    res = common.arm_move(
                        env, planner, target_tcp(sapien.Pose(p=target_p, q=q), t_tcp_cup),
                        who=WHO, stage=f"pour ({clearance}, {command}, {axis_name})",
                        tries=1,
                        max_knots=POUR_MAX_KNOTS, knot_draws=POUR_KNOT_DRAWS)
                    if res != -1 and common.stopped_by_horizon(planner):
                        return res, False
                    if res == -1:
                        continue
                    executed += 1
                    verdict, res = _settle_and_read(res, command)
                    if verdict in ("cut", "fail", "drop"):
                        return res, False
                    if reached >= need:
                        tipped = True
                        break
                    if executed >= POUR_MAX_TIPS:
                        break
                if tipped or executed >= POUR_MAX_TIPS:
                    break
            if tipped or executed >= POUR_MAX_TIPS:
                break
    if not tipped:
        # **Two different worlds, and `-1` is right in only one of them (D6).**
        #
        # `executed == 0`: every combination was *refused*. The oracle never got the cup
        # over the pot at all, so it never reached a decision — a planning refusal, `-1`.
        #
        # `executed > 0`: a tip ran, the TCP went exactly where it was sent, and the cup
        # came back under the threshold anyway, because a top grasp loses tilt as the
        # cup turns in the fingers. That is the task's own outcome and it is `missed`.
        # Measured: a 75-degree command executed with the TCP 0.004 m / 0.7 deg from its
        # goal and the *cup* at 53.9 deg against a 55 deg floor, every other term of the
        # predicate satisfied.
        #
        # Getting this wrong is not a misfiled bucket. `evaluate_planner.py:704-707`
        # appends to `infos` **only for 5-tuples**, so a `-1` episode contributes nothing
        # to `tally_info_keys` — a blind episode that poured a degree short would have
        # its `distinct_before_repeat`, `first_decision_ok` and `double_water_count`
        # deleted from the very tally the floor comparison is built on. The blind arm's
        # honest wrong answers would vanish from the statistic that measures whether the
        # task needs memory at all.
        if executed == 0 or last_ok is None:
            return fail(env, "pour pose unreachable: no clearance, tilt or axis planned",
                        site=s, trip=trip,
                        clearances=list(POUR_CLEARANCES),
                        commands=list(pour_tilt_commands(task)),
                        axes=list(POUR_TILT_AXES)), False
        cup_p = _np(task.cup.pose.p).reshape(-1, 3)[0]
        last = last_ok[-1]
        say(env, "MISSED: the cup reached the pour pose and would not stay tipped",
            site=s, trip=trip, tips=executed,
            reached_deg=None if reached < 0 else round(reached, 1),
            need_deg=round(need, 1),
            tilt_deg=round(np.degrees(_f(last, "tilt_rad")), 1),
            d_pot=round(float(np.linalg.norm(cup_p[:2] - plant_p[:2])), 3),
            clearance=round(float(cup_p[2] - pot_top), 3),
            at_plant_id=int(_f(last, "at_plant_id")),
            has_water=_b(last, "has_water"), is_grasped=_b(last, "is_grasped"))
        return last_ok, False

    budget = task.cfg.hold_steps + POUR_EXTRA_STEPS
    spent = 0
    while spent < budget:
        chunk = min(10, budget - spent)
        res = planner.idle_steps(t=chunk)
        spent += chunk
        if res != -1 and common.stopped_by_horizon(planner):
            return res, False
        if res == -1:
            return fail(env, "hold the pour", site=s, trip=trip), False
        if int(_f(res[-1], "budget_left")) < task.cfg.n_plants - trip:
            cup_p = _np(task.cup.pose.p).reshape(-1, 3)[0]
            say(env, "poured", site=s, trip=trip, steps=spent,
                repeat=watered_before,
                tilt_deg=round(np.degrees(_f(res[-1], "tilt_rad")), 1),
                d_pot=round(float(np.linalg.norm(cup_p[:2] - plant_p[:2])), 3))
            # **No untip here, and the whole family is measured dead (K61).** Three
            # variants were built, piloted and swept, each killed by its own number:
            #
            #   1. *Single mirror* — same TCP position, `q_upright` back in one move:
            #      refused 12/12 across two pilots (`joint limit at index [3]` with
            #      the torso free, `[11]` with it frozen, ~0.75 of the twist left).
            #      The wrist does not have the 65 deg return in one move from the
            #      pose the pour leaves it in — K58's `level_in_place` finding,
            #      arrived at from the other side.
            #   2. *Bite walk to level* — 35 deg in-place bites, the split pour's own
            #      mechanism reversed: the 65 -> 30 bite planned 4/6 but the final
            #      30 -> 0 bite refused 4/4, and the walk cost ~490 deg of executed
            #      rotation to save ~280 at the later levelling.
            #   3. *Parked single bite* — stop at ~30 deg and hand `level_in_place` a
            #      start inside its measured window: at n=30, concurrent A/B against
            #      K60, the untip stage cost 2 571 deg and the carry grew ~3 500 —
            #      the bite leaves the wrist in a configuration the tuck plans worse
            #      from — for a net **+14.5 %** on the meter (2 720 -> 3 115).
            #
            # The standing conclusion is bigger than the untip: across eight
            # measurements the same code varies by ±300 deg between runs and every
            # path-level treatment lands inside that band. The per-episode total is
            # pinned at ~2 700-3 200 deg by RRT's own stochasticity on held-cup
            # moves, and the ~300 deg target is not reachable by reordering stages —
            # it needs RRT out of the held-cup pipeline entirely. The K61 review's
            # doses>1 hazard (a re-armed pour band during a near-pot rotation) also
            # applies to any future attempt here; see the journal.
            return res, True
    last = res[-1]
    cup_p = _np(task.cup.pose.p).reshape(-1, 3)[0]
    say(env, "MISSED: the cup reached the pour pose and nothing committed", site=s, trip=trip,
        d_pot=round(float(np.linalg.norm(cup_p[:2] - plant_p[:2])), 3),
        clearance=round(float(cup_p[2] - pot_top), 3),
        tilt_deg=round(np.degrees(_f(last, "tilt_rad")), 1),
        at_plant_id=int(_f(last, "at_plant_id")),
        has_water=_b(last, "has_water"), is_grasped=_b(last, "is_grasped"),
        pour_hold=int(_np(last["pour_hold"]).reshape(-1)[s]))
    return res, False


# ---------------------------------------------------------------------- solve --


def solve(env, seed=None, debug=False, vis=False, blind=False, *,
          planner_factory=default_planner_factory):
    """Solve one episode. `-1` on a planning or grasp refusal, the gym 5-tuple otherwise.

    `planner_factory` exists so a test can hand in `tools/stub_planner.py`; it
    defaults to the real solver and `run.py`-style drivers never pass it.
    """
    obs, info = env.reset(seed=seed)
    # Seeds Python, numpy, torch AND mplib's C++ RNG (utils.mikasa/seeding.py). Per
    # attempt, next to the reset, so an episode belongs to its seed.
    if seed is not None:
        seed_everything(seed)

    assert env.unwrapped.control_mode in (
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ), env.unwrapped.control_mode

    planner = planner_factory(env, debug, vis)
    task = env.unwrapped
    rng = np.random.default_rng(seed)

    say(env, "episode", blind=bool(blind),
        occupied=np.flatnonzero(_np(task.occupied_mask).reshape(-1)).tolist(),
        dwells=_np(task.refill_dwell).reshape(-1).astype(int).tolist())

    # -- STAGE 0: onto the refill dock, before anything is touched ------------------
    # The reset parks the base near the dock with up to `start_jitter_xy` = 0.08 m and
    # `start_jitter_yaw` = 0.10 rad of jitter, which on seed 11 is 0.098 m off — and
    # the cup on the floor is already 0.62 m out from the dock itself, near the far end
    # of the band W1 measured a floor grasp in (0.250-0.772 m forward, |lateral| <=
    # 0.15 m). Squaring up first takes the jitter out of the hardest reach in the
    # episode; it also makes trip 0 the same shape as trips 1 and 2, each of which
    # begins by driving onto this dock. No advance: the bucket is `cfg.dip_distance`
    # = 0.55 m from the dock by construction now, and driving 0.13 m onto it would put
    # the base 0.4200 m from the bucket, inside the measured no-go band.
    if task.cfg.cup_starts_full:
        # K55. Nothing to fetch: the cup is in the gripper and already holding its
        # doses, so the opening drive to the dock, the floor grasp and the lift all
        # have no work to do. Those three are 185 + 178 + 547 deg of hand rotation
        # on the shipped task, and they exist only because the cup and the water sit
        # on the floor. The gripper is closed here so the contact is real before the
        # first drive rather than a pose that happens to coincide.
        say(env, "cup starts in the gripper, already full; skipping the fetch")
        # The scene already spawned the cup inside open fingers, so this closes
        # onto it directly.
        # Close, settle, and check; on a miss re-seat the cup between the fingers and
        # close again. The cup is dynamic and the fingers take a few frames to meet,
        # so whether contact registers on the first try depends on the spawn jitter —
        # measured, seed 1 caught it and seed 0 did not.
        # Pin the cup to the TCP for the duration of the close. It is a dynamic body
        # and the fingers take ~6 control steps to meet: left to itself it falls out
        # in that window (measured — 4 of 6 seeds refused inside 2 s, and a retry that
        # re-opens the gripper loses it for good). Re-seating it every step costs
        # nothing and stops the moment the fingers actually touch it, after which
        # physics holds it.
        for _ in range(int(GRIP_CLOSE_STEPS)):
            task.cup.set_pose(task.agent.tcp.pose)
            planner.close_gripper(t=1)
            if bool(_np(task.agent.is_grasping(task.cup)).any()):
                break
        planner.idle_steps(t=SETTLE_AFTER_DRIVE)
        planner.planner.update_from_simulation()
        if not bool(_np(task.agent.is_grasping(task.cup)).any()):
            return fail(env, "cup_starts_full: the cup is not in the gripper after closing")
        res = planner.idle_steps(t=1)
        common.hold_object_in_planner(env, planner, task, task.cup, held=True, who=WHO)
    else:
        res, parked = drive_to(env, planner, task, _np(task.refill_dock)[0],
                               "the refill dock (opening)", carry=False)
        if res == -1:
            return res
        if common.stopped_by_horizon(planner):
            say(env, "stopped by the horizon on the opening drive")
            return res
        if not parked:
            say(env, "MISSED: could not square up on the refill dock before the grasp")
            return res

        # -- STAGE 1: the cup, once. It is never put down again. -------------------
        res, grasped = grasp_the_cup(env, planner, task)
        if res != -1 and common.stopped_by_horizon(planner):
            say(env, "stopped by the horizon during the grasp")
            return res
        if res == -1:
            return res  # grasp_the_cup said why
        if not grasped:
            return fail(env, "grasp the cup: fingers closed but agent.is_grasping(cup) is False")
        planner.planner.update_from_simulation()
        common.hold_object_in_planner(env, planner, task, task.cup, held=True, who=WHO)

        # -- STAGE 2: straight up, torso frozen ----------------------------------------
        # The cup stands right beside the bucket, so anything that moves it sideways at
        # floor height drags it through the bucket wall. Up first.
        lift = sapien.Pose(p=_np(task.agent.tcp.pose.p)[0].astype(np.float64)
                           + np.array([0.0, 0.0, LIFT_AFTER_GRASP]),
                           q=_np(task.agent.tcp.pose.q)[0].astype(np.float64))
        say(env, "lift", height=LIFT_AFTER_GRASP)
        res = common.arm_move(env, planner, lift, who=WHO, stage="lift", disable_lift_joint=True,
                          max_knots=HELD_MAX_KNOTS, knot_draws=HELD_KNOT_DRAWS)
        if res != -1 and common.stopped_by_horizon(planner):
            return res
        if res == -1:
            return fail(env, "lift the cup off the floor")

        # **Possession after the lift.** K59: without this a cup that leaves the fingers
        # on the way up is not noticed until the dip, which then refuses — correctly,
        # because the planning world is being asked to drag a body lying on the floor
        # half a metre away down into a bucket — and the episode is booked
        # `no plan found (-1)`. Seed 10 measured exactly that: `cup in the gripper
        # t=255` with `agent.is_grasping` true, and by the dip at t=398
        # `cup_z_now=0.013, tcp_z_now=0.489, d_bucket=0.504`. All three of the v6
        # sweep's losses read as planning failures and at least one was this.
        #
        # That mis-labelling is a contract breach, not just a confusing log:
        # `oracle_common`'s D6 says `-1` means the oracle gave up *before a decision* —
        # a planning or grasp failure — and a physical miss must come back as the gym
        # 5-tuple so the sweep books it `missed`. A dropped cup is a miss. The fold and
        # the tip have had this guard since K58; the lift, which is where the 2.75 cm
        # rim hold is most likely to fail, did not.
        if not _b(res[-1], "is_grasped"):
            cup_p = _np(task.cup.pose.p).reshape(-1, 3)[0]
            tcp_p = _np(task.agent.tcp.pose.p)[0]
            say(env, "MISSED: the cup left the gripper during the lift",
                cup_z=round(float(cup_p[2]), 3), tcp_z=round(float(tcp_p[2]), 3),
                d_tcp=round(float(np.linalg.norm(cup_p - tcp_p)), 3),
                height=LIFT_AFTER_GRASP,
                depth_hint="the rim hold is 2.75 cm on a cup that stands on the floor")
            return res
        planner.planner.update_from_simulation()

    for trip in range(task.cfg.n_plants):
        # -- the refill --------------------------------------------------------
        # Every trip reaches this line with the base squared up on the refill dock:
        # trip 0 by the opening drive, trips 1 and 2 by the drive back at the end of
        # the trip before. `t_tcp_cup` is read here, from a still moment, and not once
        # at the grasp: the burner measured the cup shifting in the fingers during the
        # base rotations.
        t_tcp_cup = (task.agent.tcp.pose[0].inv() * task.cup.pose[0]).sp
        # K55. Skip the whole refill errand while the cup still holds water. At the
        # shipped `doses_per_fill = 1` this is never true after a pour and the loop
        # is unchanged; above 1 it is what the knob buys — the drive back, the dip,
        # the fold to the dwell pose and the dwell itself all disappear for the
        # trips that do not need them, and those are the task's most expensive
        # stages in hand rotation.
        if bool(_np(task.has_water).reshape(-1)[0]):
            say(env, "cup still holds water; skipping the refill", trip=trip,
                doses_left=int(_np(task.doses_left).reshape(-1)[0]))
            filled = True
        else:
            res, filled = refill(env, planner, task, res[-1], t_tcp_cup, trip)
            if res == -1:
                return res
            if common.stopped_by_horizon(planner):
                say(env, "stopped by the horizon during the refill", trip=trip)
                return res
        if not filled:
            return res  # refill already said MISSED with the diagnostic fields

        # -- the memory --------------------------------------------------------
        s = choose_next_plant(task, blind, rng)
        say(env, "plant chosen", trip=trip, site=s, blind=bool(blind),
            already_watered=bool(_np(task.watered_mask).reshape(-1)[s]))

        # -- carry, drive, pour -------------------------------------------------
        # `carry=True` (the default), as K52 requires. It was skipped here for one
        # round: the tuck from the rest keyframe left `wrist_flex` on its stop and
        # `move_base_forward` then refused the drive, reproducibly. That was a symptom
        # of the free-arm base plan, not of the tuck — with `freeze_arm=True` the wrist
        # is not the plan's to move — so the cup is carried in as the primer says.
        res, parked = drive_to(env, planner, task, _np(task.site_dock)[0, s],
                               f"plant {s}", advance=POUR_DOCK_ADVANCE,
                               site=s, trip=trip)
        if res == -1:
            return res
        if common.stopped_by_horizon(planner):
            say(env, "stopped by the horizon on the way to the plant", trip=trip, site=s)
            return res
        if not parked:
            say(env, "MISSED: parked outside the plant's dock zone", trip=trip, site=s)
            return res
        planner.planner.update_from_simulation()

        t_tcp_cup = (task.agent.tcp.pose[0].inv() * task.cup.pose[0]).sp
        res, poured = water(env, planner, task, s, t_tcp_cup, trip)
        if res == -1:
            return res
        if common.stopped_by_horizon(planner):
            say(env, "stopped by the horizon during the pour", trip=trip, site=s)
            return res
        if not poured:
            return res  # water() already said MISSED

        # -- back for the next dose --------------------------------------------
        # The forced return is the design (it is what stops the base pose from being
        # the progress counter), so it is a stage even on the last trip — except that
        # after the last dose the budget is spent and the verdict has already latched.
        if trip + 1 < task.cfg.n_plants:
            planner.planner.update_from_simulation()
            res, parked = drive_to(env, planner, task, _np(task.refill_dock)[0],
                                   "the refill dock", trip=trip)
            if res == -1:
                return res
            if common.stopped_by_horizon(planner):
                say(env, "stopped by the horizon on the way back", trip=trip)
                return res
            if not parked:
                say(env, "MISSED: parked outside the refill dock zone", trip=trip)
                return res

    say(env, "episode over", watered=_np(res[-1]["watered_mask"]).reshape(-1).astype(int).tolist(),
        occupied=_np(res[-1]["occupied_mask"]).reshape(-1).astype(int).tolist(),
        success=_b(res[-1], "success"), doubles=int(_f(res[-1], "double_water_count")))
    return res


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--scene-idx", type=int, default=0)
    p.add_argument("--output-dir", default="videos/water_plants")
    p.add_argument("--render-mode", default="rgb_array")
    p.add_argument("--render-width", type=int, default=512)
    p.add_argument("--render-height", type=int, default=512)
    p.add_argument("--max-steps-per-video", type=int, default=None)
    p.add_argument("--max-episode-steps", type=int, default=None,
                   help="override the task horizon; for calibration only (Task 7 measures it)")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--no-trajectory", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--histogram", type=int, default=None, metavar="N",
                   help="print the blind draw histogram over N seeds and exit (no simulator)")
    p.add_argument(
        "--blind",
        action="store_true",
        help="memory-free control: draw the next plant uniformly, WITH replacement",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.histogram is not None:
        print_blind_histogram(args.histogram)
        return 0

    kwargs = dict(
        num_envs=1,
        render_mode=args.render_mode,
        robot_uids="mikasa_ds_fetch",
        control_mode="pd_joint_pos",
        scene_idx=args.scene_idx,
        human_render_camera_configs=dict(width=args.render_width, height=args.render_height),
    )
    if args.max_episode_steps is not None:
        kwargs["max_episode_steps"] = args.max_episode_steps
    env = gym.make("MikasaWaterPlants-v0", **kwargs)
    env = RecordEpisode(
        env,
        output_dir=args.output_dir,
        save_video=not args.no_video,
        save_trajectory=not args.no_trajectory,
        video_fps=30,
        save_on_reset=True,
        max_steps_per_video=args.max_steps_per_video,
    )

    res = solve(env, seed=args.seed, debug=args.debug, vis=False, blind=args.blind)
    if res == -1:
        print("failed_motion_plan")
    else:
        info = res[-1]
        print(
            "success:", _b(info, "success"),
            "| watered:", int(_f(info, "watered_count")),
            "| omissions:", int(_f(info, "omission_count")),
            "| doubles:", int(_f(info, "double_water_count")),
            "| distinct_before_repeat:", int(_f(info, "distinct_before_repeat")),
            "| first_decision_ok:", _b(info, "first_decision_ok"),
            "| budget_left:", int(_f(info, "budget_left")),
        )
    env.close()
    return res


if __name__ == "__main__":
    sys.exit(0 if main() != -1 else 1)
