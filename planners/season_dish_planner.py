"""Oracle for MikasaSeasonDish-v0, and its blind twin.

Written against `tools/stub_planner.py` on a Mac and run for real in the amd64 CPU
container (`tools/docker/planner.sh run …`) since 2026-08-18 (T5) — the geometry
below is what the container measured. The stage kit is `oracle_common.py` (D7);
the four poses that were `TODO(owner)` here are now built from it: the grasp from
`grasp_geometry`/`try_grasp`, the drive to the task's own bowl dock, the hover
from `pose_over`, and the pour from `pour_pose_for` (this file's one piece of
geometry of its own, pure and tested against the task's predicate arithmetic).

What the oracle does, in order, and why:

1. **Waits the cue out** (`wait_cue`, to `cfg.cue_steps` — K36; the repo rule: an
   oracle does not act through the cue) and only then reads the answer once,
   `info["target_is_shaker"]` — privileged, as `template_planner.py` allows a
   demonstrator; `--blind` replaces that read with a **uniform draw** of the
   condiment from `np.random.default_rng(seed)` (the burner's convention) and
   changes nothing else, so the arms differ only in memory (blind ≈ 0.5× sighted).
   (Until the T5 fix round it took "always the shaker" — the harder grasp — and
   scored structurally below the floor; both numbers are in the journal.)
2. **Slides the base along the counter** until the target is straight ahead
   (`dock_for_target`, K55). The station dock is drawn midway between the two
   condiments, so the target sits `station_spacing / 2` = 12 cm to one side and the
   reach is a diagonal; the slide costs a ~12 cm drive and leaves the standoff
   alone. A refusal is not fatal — the grasp from the dock as drawn is what the
   oracle did before.

3. **Grasps the target**, and keeps asking when refused (K55). First the inherited
   attempts: OBB thin side, one more draw of the same grasp (RRT is randomized),
   then the closing direction reversed. Then a ladder over the three things that
   are free to choose, cheapest first — **wrist yaw** (both condiments are bodies
   of revolution, and the shaker's 0.051 x 0.052 m footprint means the yaw the OBB
   picks is decided by a millimetre), **grip height** (+2.5 cm lifts the wrist out
   of the counter it was sweeping through), and finally **stance** (±10 cm toward
   the counter, when the refusal is a bare `IK Failed` and no wrist angle can help).
   A refused plan costs no episode steps, so the ladder is nearly free in horizon —
   only in wall clock. Verifies with `agent.is_grasping`, then reads
   `distractor_ok` off the step's info — a nudged distractor is a **miss** (D6:
   the 5-tuple, not `-1`).
4. **Attaches the object in the planning world** (`hold_object_in_planner`;
   honest since K53), **lifts** with the torso frozen to `lift_z = max(grasp.z +
   0.15, tallest condiment top + 0.10)` (K40 — the base turn carries the object
   over its neighbour), takes the object's pose in the base frame, and **drives**
   straight to the bowl dock, facing the counter — no carry pose (K57: measured
   free, 30 seeds per arm, and a quarter of the motion cheaper); `d_dock`/`dyaw`
   in the trace (K26); `distractor_ok` again.
5. **Hovers** the object origin `HOVER_ABOVE` over the bowl, upright as it was
   grasped relative to the base (`object_q_from_base` — the base turns under it),
   using `T_tcp_obj` read *after* the drive (the emulated burner showed the object
   shifting in the fingers during base turns).
6. **Pours**: `pour_pose_for` puts the object origin `POUR_ABOVE` over the bowl,
   tilted about a horizontal world axis — the candidates in order: the base's
   facing (+tilt, −tilt), then along the counter (±), first at `POUR_TILT_DEG` and
   then at `POUR_TILT_STRONG_DEG`. The second rung is for the arm that reaches the
   pose but not the angle: the predicate wants 55°, the first rung commands 65°,
   and seed 2 came to rest at 47.5° — a 10° margin against a 17.5° tracking error.
   Each executed candidate is given `POUR_SETTLE_STEPS` to come to rest before its
   angle is believed, because a tilt read mid-motion is the commanded pose's error
   rather than the object's attitude. After each the predicate's own flags are read
   from info; a plan that ran
   but did not satisfy them moves on to the next; `grasp_target` gone → a miss
   ("dropped during the tilt"); none satisfied after all candidates → a miss
   ("no pour pose reached", with the last `tilt_rad`); `-1` only if **no**
   candidate planned at all (D6).
7. **Holds**, polled in `HOLD_POLL_CHUNK` steps to `HOLD_POLL_BUDGET` and breaking
   as soon as `success` latches — `pour_hold` resets on any single step where
   `pour_now` is false, so one flicker inside a fixed window loses an episode that
   would have latched a few steps later. Returns the last 5-tuple. Every stage checks
   `stopped_by_horizon(planner)` before interpreting its result (K23).

Contract, as `template_planner.py:32-45`: the solution owns the reset; return `-1`
on a failed plan and the gym 5-tuple otherwise; do not catch exceptions; do not
decide success; do not call `env.evaluate()` — read `info` from the tuple.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys

import gymnasium as gym
import numpy as np
import sapien

from mani_skill.utils.wrappers import RecordEpisode

from utils.mikasa_oracle.planners import oracle_common as common
from utils.mikasa.seeding import seed_everything

# The mplib-dependent imports live inside oracle_common's factories, so this module
# imports on a Mac and tests/test_season_dish_oracle.py runs the *real* solve()
# against tools/stub_planner.py (K38).

WHO = "season_dish_planner"

# Distance the fingers close over, used to sink the grasp into the object.
FINGER_LENGTH = common.FINGER_LENGTH

# Object origin over the bowl origin: the hover, and the pour (inside the predicate's
# clearance band `pour_min_clearance` 0.05 … `pour_max_clearance` 0.30).
HOVER_ABOVE = 0.20
HOVER_RUNGS = (
    (0.0, 0.0, 0.0), (0.06, 0.0, 0.0), (0.0, 0.06, 0.0), (-0.05, 0.0, 0.0),
    (0.0, 0.0, 30.0), (0.0, 0.0, -30.0), (0.0, 0.0, 60.0),
)
"""(extra height, metres pulled back toward the base, degrees of spin) for the hover.

`arm_move` already draws the same pose twice, which answers RRT's randomness but not a
pose the arm cannot hold: held-out seed 24 refused both draws with `joint limit at index
[11]` — the wrist, 0.185 of the twist short. A different height or a hover pulled in
toward the base is a different arm configuration for the same job. So is a spin: both
condiments are bodies of revolution, so turning one about its own vertical axis is
physically nothing — the same shaker over the same bowl — while the hand that holds it
orbits to a different wrist angle. The pour inherits whichever orientation the hover
settled on, since that is where the object actually is. The first rung is the shipped
hover, so a seed that hovers first time is unchanged."""
POUR_ABOVE = 0.20

# The hover is approached through a waypoint this far back toward the base along its
# facing (same height, same orientation): from the tucked arm the direct hover was one
# long RRT motion and came back `Approximate solution` / `IK Failed` on seed 12 (a
# refused waypoint is skipped, not fatal — the hover is then asked for directly). That
# measurement predates K57, which deleted the tuck: the arm now arrives at the dock
# extended, so the waypoint may have become unnecessary. It is kept because it is free
# when refused, and re-measuring it is its own experiment.
PRE_HOVER_BACK = 0.35

# Tilt asked of the object at the pour: the predicate wants ≥ 55° (`pour_tilt_deg`);
# 65° leaves 10° for the plan's orientation error. May be lowered toward 60° by
# measurement, never the predicate's bound widened.
POUR_SETTLE_STEPS = 4
"""Steps to let the object come to rest before its tilt is believed."""

HOLD_POLL_CHUNK = 5
HOLD_POLL_BUDGET = 60
"""The hold is polled in `HOLD_POLL_CHUNK` steps up to `HOLD_POLL_BUDGET`, breaking as
soon as `success` latches. `cfg.hold_steps` is 15, so the first three chunks are the
old single block; the rest is slack for a counter that reset once on a flicker."""

# A stance rung — re-driving the base 10 cm and walking the ladder again — was built
# and withdrawn (K55). It does not pay: `drive_base` refused the 10 cm shift outright
# on seed 7 (a base translation planned with all fifteen joints), and the opposite
# shift spent what was left of the horizon on turn-drive-turn, converting a `no plan`
# into a `truncated`. A flat 0.15 m advance applied from the start measures worse
# still (2/10). The reach that stance was meant to buy is bought in the task instead,
# by standing the stations clear of the sink — see `station_along` in scenes/.

# Gripping at the centre of mass, floored by clearance over the counter
# (`min(top - 0.01, max(centre, counter_top + 0.06))`), was built and withdrawn (K55).
# The reasoning is sound and the lever arm is real — `max(centre, top - GRASP_BELOW_TOP)`
# raises the 15.7 cm bottle to 4.9 cm above its centre of mass, which is what pulls it out
# of the hand during the tuck on held-out seed 29 (K57 has since deleted that tuck, so
# this particular failure may no longer exist). But gripping 2-5 cm lower costs more
# than it buys, measured: eval seed 3 then fails the *lift* (`joint limit at index [3]`,
# the torso, which the lift freezes — from a lower grip the arm alone must span more), and
# held-out seed 27 fails the grasp on `wrist_flex_link <-> counter_main`, the very
# collision GRASP_BELOW_TOP exists to avoid. Both are new failures on seeds that passed.
# The lever arm wants a fix that does not move the hand down, and that is not this one.
#
# **Re-measured on the randomized layout and still true** (K79r). K79p quantified the lever
# the analyst blamed for the topple — `max(centre, top - 0.03)` puts the pads **48.5 mm
# above the centre of mass** on the 157 mm bottle, whose whole footprint then rides above a
# free-standing object held down by friction, while the 94 mm shaker's pads straddle its CoM
# at 16.5 mm and it is never ejected that way. Gripping at the CoM instead (floored at the
# counter clearance, so it can only move *down toward* the CoM) measured **161/180 against
# 174/180** — thirteen seeds worse, grasp failures 5 -> 15. The mechanism is real and the
# remedy is worse, on the new population as on the old.

ARC_RETREAT_M = 0.10

GRASP_IK_GIVE_UP = 3
"""Consecutive `IK Failed` refusals after which the wrist ladder is abandoned (K79b).

mplib reports two different refusals and the ladder used to treat them alike. A tree
search that times out (`RRTConnect Failed`) says the pose is reachable and the corridor
is thin — worth another draw, another yaw. `IK Failed! Cannot find valid solution` says
no collision-free arm configuration puts the TCP there at all; it is iteration-bounded,
not time-bounded, and **no wrist yaw and no grip height can answer it**. Measured on a
failing seed-80 capture: 7 of 13 refusals were hard IK failures, and the ladder answered
them by trying six yaws, two closings and three top-down poses — all refused the same way
— while the one attempt that changes base-to-object distance (the arc) ran last, after the
200-step budget was already spent (203/200).

So: count them, and when three come in a row stop turning the wrist and go to the arc with
budget left. Only reachable on the failure path, so no episode that currently succeeds can
change."""

IK_REFUSAL = "IK Failed"
"""The substring `capture_refusal` watches for; mplib prints it from `plan_pose`'s IK."""

ARC_KEEPOUT_PAD = 0.08
"""Keep-out inflation for the base-and-arm arc when the hand could not back off (K79).

Only for that case. `GRASP_KEEPOUT_PAD` (0.03) is right for an approach leg, which must
close on the target; this leg moves the base and every arm joint at once at ~0.8 m
extension, where the measured Cartesian tracking error is 7-9 cm. On a failing seed-80
capture it swept the gripper through the distractor and moved it 0.1227 m, 23% past
`distractor_move_tol` — inside a keep-out that was nominally guarding it."""
"""Metres the hand backs off before the base-and-arm plan, to clear its own start state."""

RESYNC_BEFORE_GRASP = os.environ.get("MIKASA_RESYNC_GRASP", "1") == "1"
"""Re-sync the planning world between the approach and the grasp leg (K79i).

The approach leg **executes**, and executing is what knocks the object; without this the
grasp leg is planned against the pre-approach world. Measured directly, by comparing every
`static_manipulation` call's `PlanningWorld` pose against SAPIEN's on a failing seed-71
episode: drift is **0.0 mm at ten of eleven calls** and **88.9 mm at the one rung** that
reported `reached=True` to 2 mm and then closed its fingers 9.7 cm from the object — the
same rung an independent per-seed post-mortem had flagged, with matching magnitudes.

**SR-neutral: 173/180 against 174/180, inside noise.** Kept as a correctness fix, not as a
gain, and said so here so the next reader does not credit it. It cannot rescue that grasp —
the commanded pose is still derived from the stale mesh read, and K55 measured that
re-aiming after the fact does not help — but planning a collision check against a world
known to be 8.9 cm out of date is wrong on its own terms, and a sync costs no episode steps.

The same probe also **refuted** the theory that motivated the search: every `IK Failed`
refusal in that episode happened at **0.0 mm** drift. Those refusals are honest; the
planning world and the simulator agree at exactly the moments the planner says no."""

GRASP_N_INIT_QPOS = (lambda v: None if v <= 0 else v)(
    int(os.environ.get("MIKASA_N_INIT_QPOS", "0")))
"""IK seed configurations for the grasp legs (None = mplib's default of 20; K79m).

The last unexplained thing after K79c-K79l: IK reports **no solution** at a pose whose
neighbour one wrist rotation away plans to 3 mm, with the planning world verified in sync
(0.0 mm drift at every refusal, K79i) and the arm demonstrably able to reach the point.
`IK Failed! Cannot find valid solution` means all `n_init_qpos` seeds were rejected — so
the refusal may be about how many starting configurations the solver tried, not about
whether a solution exists. The solver already raises this elsewhere for the same reason
(`move_base_forward(..., n_init_qpos=100)`).

**Two companion fixes were tried; neither works** (K79n, K79u). And the combination this
journal proposed as its own handoff is now measured and **wrong**: K79s removed the drops
(its control arm shows zero), so raising `n_init_qpos` on top of it should have kept the
recovered grasps without the cost. It does not — **169/180 against 174/180, five seeds
worse, and the drops stay at five**. The extra drops are not the crush-past-contact kind
K79s fixes; they are genuinely marginal grasps, and dropping is simply how that fragility
shows. Stopping the close cannot make a marginal IK solution hold.

**The first companion fix** (K79n). Raising this converts
grasp failures but adds `dropped during the lift` ones, so the natural next move is to ask
IK for the solution *nearest the current configuration* rather than any solution
(`SapienPlannerV2.IK(..., return_closest=True)`), on the theory that the extra solutions
are contorted arm poses. Plumbed end to end and measured: **both seeds that pass at
n_init_qpos=100 fail with it on.** The reason is in the API, not the physics —
`return_closest` collapses IK's *list* of goal configurations to a single one, so
RRTConnect is handed one goal state instead of many and plans worse. Withdrawn, and the
shared-solver plumbing reverted with it.

**So was the second companion fix** (K79o). If the extra solutions give bad *grips*, score the
grip: the two prismatic finger joints open to 0.05 each, so their sum is the aperture, and a
real hold on a 5.1 cm shaker keeps them near 0.051 while a hold on the lip closes them much
further. `is_grasping` reports that contact exists, not that the hold is good. Measured on the
seeds that drop at `n_init_qpos=100`: on seed 6 the gate **fires** and the ladder still finds
nothing better; on seed 55 it **does not fire** and the object drops anyway. Aperture is
neither sufficient nor actionable here — a drop during the lift is not predicted by how wide
the fingers ended up."""

CLOSE_ON_CONTACT = os.environ.get("MIKASA_CLOSE_CONTACT", "1") == "1"
"""Stop the gripper close at first contact rather than driving it fully shut (K79s).

The last untried lead from K79p, and the second change all session to actually move the
rate. The fingers are **position**-commanded: they reach exactly 0.0000 aperture in both
measured drop runs and never reopen, so a contact that begins to slip simply lets them close
further — on seed 6 a 45 mm bottle ends pinched at 21 mm, mid-topple, levered over by the
closing motion itself while the arm moves 0.4 mm. Polling `is_grasping` after each step and
stopping there ends the close typically at step 2 of 6.

**The other half of that recommendation — *slow* the closure — was tried and is worse**
(K79w): ramping the command toward CLOSED over 6 steps instead of stepping to it measures
174/180 against 176/180, and puts a drop back. The contact test fires while the ramp is a
third of the way in, so the grip at the stop is lighter; gentleness costs more than the slam
did. This change already has the useful half.

Measured, 180 seeds per arm interleaved, **two independent runs** (the standard K79's budget
had to meet — one favourable sweep is what produced the K79j disaster):

| | fixed close | stop on contact |
|---|---|---|
| run 1 | 173/180, 7 failures | **175/180, 5** |
| run 2 | 171/180, 9 failures | **174/180, 6** |

Same direction both times, and the gain lands where the mechanism predicts: one `dropped
during the lift` per run against two or more in the control.

**Partial by construction, and deliberately left so** (K79t). `self.gripper_state` stays
CLOSED afterwards — which on this `PDJointPosMimicController` targets `lower = -0.01 m`,
commanding the fingers *past shut* for every later step. Completing the fix by re-aiming that
standing command at the width the fingers actually reached (minus a 2 mm squeeze) was built
and measured: **173/180 against 175/180**, two seeds worse, and it grows failures the control
does not have — `FAILED: drive to bowl dock` and `carry pose unreachable`, which is what a
looser grip does to an object being carried. The control arm shows **zero** drops, so this
change already captures the whole effect; the residual squeeze was not costing anything."""
GRASP_STOP_ON_TOUCH = os.environ.get("MIKASA_GRASP_STOP_ON_TOUCH", "0") == "1"
"""Stop the standoff->grasp leg at the first robot-target contact and close right there
(K102, `stop_on_touch`). g66 @0.15 (2026-09-06): on 1679 and 1957 the approach's executed
draw had no knot within 6 cm of the shaker and the shaker still lay on its side at the
close — the fingertip topples it on the LAST 10 cm, the grasp leg, not on the transit.
Default off until measured."""
GRASP_LEG_STRETCH = int(os.environ.get("MIKASA_GRASP_LEG_STRETCH", "1"))
"""Slow the standoff->grasp leg this many times (rows interpolated, K84's mechanism). The
contact traces of 1679/1957 (2026-09-06): the finger never reaches the shaker — at 3.9 cm
from its side, closing at 0.16 m/s, PhysX's contact solver (contact_offset 0.02 on both
shapes) kicks the 16 g shaker to 36 rad/s in one step. 1 = as shipped."""
APPROACH_BY_LINE = os.environ.get("MIKASA_APPROACH_BY_LINE", "1") == "1"
"""The approach leg first as a straight JOINT line to the standoff's nearest IK solution
(collision-checked, cannot wind), the screw and the RRT only when no line plans. From the
rest keyframe the screw is refused 200/200 at the shoulder_lift stop and the RRT wanders
80-110 knots (HP6, 1500-1699) — the arm's "twisting" at every episode's start. 0 = off."""

STRETCH_MAX_CLEARANCE = float(os.environ.get("MIKASA_STRETCH_CLEARANCE", "0"))
"""Apply `APPROACH_STRETCH` only to grasps this close to the worktop (K87). 0 = all grasps.

The shaker topples in all five stable failures and the condiment bottle in none of them,
and the reason is not tippiness — the bottle is the more slender of the two (15.7 cm on a
4.55 cm base against 9.4 on 5.1). It is **grasp clearance**. `max(centre, top - 0.03)`
puts the shaker's grasp at 0.984 and the bottle's at 1.047, against a worktop at 0.920:
**6.4 cm against 12.7 cm**. A low grasp holds the wrist near the worktop, which is what
makes the approach a constrained RRT rather than a screw, and RRT approaches own 4 of the
5 first slides (K84).

So the slowdown can be spent where the risk is instead of on every leg. An object at rest
sits on the worktop, so `mesh.bounds[0][2]` is the counter and no new plumbing is needed —
the same self-gating trick K81 used."""

APPROACH_STRETCH_TAIL = int(os.environ.get("MIKASA_APPROACH_STRETCH_TAIL", "0")) or None
"""Slow only the last N knots of the approach instead of all of them (K86). 0 = all.

`APPROACH_STRETCH=2` over the whole leg converts core seeds 71, 80 and 177 by preventing
the topple outright (`drop` 2.5 cm -> 0.0 cm) and still nets 172/180 against 174, because
a 128-252 knot approach doubled costs 128-252 episode steps and those come out of
`GRASP_LADDER_STEP_BUDGET`: out-of-budget deaths 6 -> 12 at budget 200, and raising the
budget only trades them for truncations (172 at 450, 171 at 900).

The contact that topples the object happens where the hand arrives, not on the transit to
it, so most of that cost buys nothing. This slows the arrival alone."""

APPROACH_APERTURE = float(os.environ.get("MIKASA_APPROACH_APERTURE", "-1"))
"""Normalised finger command during the approach when `NARROW_APPROACH` is on (K93).

-1 shut, +1 fully open (what the oracle ships). K90 measured only the endpoints and
concluded that "every hand configuration puts a different five seeds inside the margin" —
a claim resting on **two samples**. This is the third. A middle aperture is not obviously
worse than either: full open sweeps the widest hand through the corridor that clips, and
fully shut has to open at the standoff, where the opening fingers can collide instead."""

ABORT_ON_TOUCH = os.environ.get("MIKASA_ABORT_ON_TOUCH", "0") == "1"
"""Stop the approach the moment a gripper link touches the target (K92). 0 = shipped.

The only warning the oracle can act on. K90 measured that a **fingertip** is what topples
the object and that first contact leads the object's first movement by **2-15 steps**;
K91's prediction test then showed the warning cannot come earlier — the planned path's
intrusion on the target is **0 knots on every seed that topples**, so nothing before
execution distinguishes them. The contact itself is the signal, and SAPIEN reports it.

On abort the rung is handed back to the ladder rather than continued, because continuing
means driving the last 10 cm and closing on a grasp pose computed before the nudge."""

NARROW_APPROACH = os.environ.get("MIKASA_NARROW_APPROACH", "0") == "1"
"""Travel to the standoff with the fingers shut and open them there (K90). 0 = shipped.

TRIED ON 2026-09-05 AND WITHDRAWN — the measurement that seemed to carry it was mine and
it was invalid. The FINAL run over 200 unseen seeds lost six, and
four of them are this exact mechanism: `the object has moved since the first grasp`, by
7 to 12 cm, after which no wrist yaw and no grip height has a solution. Probed on all six
failing seeds, three ways:

    as shipped (open hand)          0 / 6
    fingers shut for the transit    6 / 6
    abort on first touch            0 / 6

Six of six — and the six were chosen BECAUSE they failed with the open hand, so any change
that perturbs the trajectory was going to recover some of them. That is not validation,
it is the selection speaking.

The paired run says so plainly. Same 200 unseen seeds, same commit, the flag the only
difference (`runs/2026-09-05-final2` against `runs/2026-09-05-sd-paired`):

    open hand   195/200      b = 6 seeds only the open hand wins
    shut hand   194/200      c = 5 seeds only the shut hand wins

A wash, and the six it fixes are not the six it breaks. It also costs a close and an open
per rung. So the shipped default stays as it was, and what remains true from the probe is
narrower than it looked: the shut hand changes WHICH seeds fail, not how many.

`ABORT_ON_TOUCH` was probed in the same run and recovered 0 of 6 — that one is a real
negative, and the reason is in the mechanism: the contact IS the nudge, so aborting on it
is always too late.

Measured from SAPIEN's own contact reports rather than inferred from trajectories: on
seeds 61, 129, 136 and 177 the first robot-shaker contact is `r_gripper_finger_link` or
`l_gripper_finger_link` **every time**, 1-15 steps before the object starts moving, with
peak impulses 0.005-0.248. Never the wrist, never the forearm, never the gripper body
first. The thing that topples the shaker is a fingertip.

The oracle holds the hand **fully open** — 0.05 per finger, 10 cm of aperture — for the
whole approach, though the fingers only need to be open at the standoff, 10 cm short of
the target. Closing for the transit halves the swept width exactly where it clips. Costs
one `close_gripper` and one `open_gripper` per rung (about 12 steps) and moves no target
pose, so it is neither a geometry change nor a timing one."""

APPROACH_STRETCH = int(os.environ.get("MIKASA_APPROACH_STRETCH", "1"))
"""Command the approach leg this many times slower along the identical path (K84). 1 = shipped.

The experiment K79g named and declared impossible. Its reasoning was right about mplib —
`joint_vel_limits` are constructor-only and no per-call override exists — but wrong that
this closes the question: the follower sends one row of `result["position"]` per control
step, so interpolating rows halves the commanded joint delta per step without changing a
single waypoint the planner chose. Tracking lag is set by that delta.

Why the approach leg specifically. K81's trajectories show the object's first slide, and
**4 of the 5 stable failures have it owned by a long RRT approach** (128-252 knots; the
fifth is a 26-knot screw). That leg is planned with the condiments inflated 3 cm, so its
*plan* clears the object — what reaches it is the execution, measured at up to 6.4 cm of
TCP error (K79c). K79f slowed every leg and lost 1 -> 11 truncations; K79h repeated it
with the horizon raised and lost four seeds anyway. Neither has ever slowed this leg
alone, which is the version whose cost is bounded: `(factor - 1) * knots` steps on one
leg, against ~500 of 1100 an episode uses.

**MEASURED 2026-09-05 AND IT LOSES.** The bounded version was finally run, paired, 200
seeds per arm on the same commit (`runs/2026-09-05-stretch`, seeds 300-499, incomplete 0):

    stretch 1   194/200    missed 3  no plan 3  truncated 0
    stretch 2   190/200    missed 2  no plan 5  truncated 3
    b = 7 seeds only the normal speed wins, c = 3 only the slow one

The reasoning was sound and the number says no. The horizon is where it goes: three
episodes truncate that never truncated before, and the leg it slows is the long one.
Whatever the executed 6.4 cm of deviation costs, buying it back with steps costs more."""

APPROACH_DRAWS = int(os.environ.get("MIKASA_APPROACH_DRAWS", "1"))
"""RRT draws the approach leg takes before choosing one (K80). 1 = shipped; **3 is worse**.

The defect this was built for is K67's, and it is still there. mplib gives OMPL one
weight-1 subspace per joint, so `maxExtent` is the plain sum of the joint ranges
(**132.01**, of which `root_x`/`root_y` contribute 80 at +/-20 m) and
`longestValidSegment = maxExtent * 0.01 = 1.32` rad — longer than every edge RRTConnect
draws at `rrt_range=0.1`, so **the interior of a path is validated at its endpoints
alone**. Measured again here on the shipped configuration, with `MIKASA_SKIM_REPORT=1`
and the folded-qpos convention `check_for_env_collision` actually wants: **48% of
approach legs command a trajectory that intrudes on the world**, mid-path rather than at
either end (held-out seed 83: knots 31-111 of 181, `l_gripper_finger<->condiment_bottle`
on 49 of them and `<->keepout_1` on 58). K76's keep-out is in the world and the path goes
straight through it. The furniture is hit more often than the condiments are.

Draws to the same goal differ, so selection has something real to act on — unlike K65,
which selected on knot count and so on the simplifier's B-spline subdivision. A clean
draw existed for **74%** of legs, and taking it cut mean intruded knots 7.9 -> 3.3.

**And the rate went down.** 180 seeds per arm, interleaved in one pool:

| | dev 0-79 | held 80-179 | total | new modes |
|---|---|---|---|---|
| **1 draw (shipped)** | **78/80** | 96/100 | **174/180** | - |
| 3 draws | 75/80 | **97/100** | 172/180 | 2 dropped in lift, 1 in tilt |

Two seeds worse and +24% episode wall clock (79 s -> 98 s), with three drop failures the
control arm does not have. This is the **third** independent way of removing the strike
(K66 `simplify=False`, K67 verify-then-replan, this) and the third to move the rate by
nothing or less. The strike is real, quantified and not the binding constraint.

Left at 1 — byte-identical to the code before K80 — with the override kept so the sweep
is cheap to repeat, as `MIKASA_GRASP_PAD` and `MIKASA_RRT_RANGE` are. The measurement
half (`MIKASA_SKIM_REPORT`, `MikasaFetchSolver.path_env_collisions`) is
worth more than the lever and stays: it is the only instrument in the repo that reads
what the arm is *about* to do rather than what it did.
"""

DISTRACTOR_KEEPOUT_PAD = float(os.environ.get("MIKASA_DISTRACTOR_PAD", "0.03"))
"""Keep-out inflation for the condiment that is *not* the target (K82).

Equal to `GRASP_KEEPOUT_PAD` by default, which makes the per-actor list identical to the
scalar the oracle passed before. K79c built the per-actor plumbing for this and then did
not use it, with the reason recorded: the standoff ceiling that caps the pad binds only
the object being approached, so nothing caps the distractor's — but seed 61 showed the
opposite cost at the same site (its rungs refused `gripper_link<->keepout_1` on the hand's
own **start** state), the two effects run opposite ways, and it was left unmeasured.

It is worth measuring now because K82's re-read has exactly one measured cost: it converts
core seed 71 and loses seed 17 to `distractor moved during the grasp`, which is the mode a
wider distractor pad exists to prevent."""

GRASP_KEEPOUT_PAD = float(os.environ.get("MIKASA_GRASP_PAD", "0.03"))
APPROACH_TARGET_PAD = float(os.environ.get("MIKASA_APPROACH_TARGET_PAD",
                                           os.environ.get("MIKASA_GRASP_PAD", "0.03")))
"""The TARGET's own proxy pad on the approach leg (the distractor's and the lift/hover
pads are GRASP_KEEPOUT_PAD / DISTRACTOR_KEEPOUT_PAD). g61 @0.15 (2026-09-06): at 0.05 the
fingertip-topple seeds 1679 and 1957 (and 1916) grasped, and 1664 lost its ladder to IK
refusals — a wider pad keeps the fingertips off a 16 g shaker and shrinks the reachable
set. Measured per value; the default is the shipped 0.03."""
APPROACH_CLEARANCE_PAD = (None if os.environ.get("MIKASA_APPROACH_CLEARANCE", "0.06") in ("", "0")
                          else float(os.environ.get("MIKASA_APPROACH_CLEARANCE", "0.06")))
"""The clearance knife on the approach leg: among the RRT draws that are clean by the
shipped 3 cm proxy, execute the one with the fewest knots inside a proxy THIS wide
around the target, then the shortest. K90/K91: the draws that topple the 16 g shaker
are clean by the intrusion metric — a fingertip passes within tracking error of it.
Ranking only; no refusal changes. "" or 0 = off (clean-then-short, as shipped)."""
GRASP_LEG_MAX_KNOTS = (None if os.environ.get("MIKASA_GRASP_LEG_MAX_KNOTS", "60") == "" else
                       int(os.environ.get("MIKASA_GRASP_LEG_MAX_KNOTS", "60")))
GRASP_LEG_KNOT_DRAWS = int(os.environ.get("MIKASA_GRASP_LEG_KNOT_DRAWS", "3"))
"""The grasp leg (standoff -> grasp, ~10 cm) under a knot cap WITH refusal: 1653 @0.15
(2026-09-06) had its screw refuse and RRT answer with 209 knots, tcp_err 0.196 m — the
arm swept the bottle 33 cm before the fingers moved. Refused, the ladder re-aims with the
object where it stands; the knot knife alone was wrong on the APPROACH (W29: the shortest
path hugs obstacles), which keeps its skim ranking. "" disables."""
APPROACH_MAX_KNOTS = (None if os.environ.get("MIKASA_APPROACH_MAX_KNOTS", "120") == "" else
                      int(os.environ.get("MIKASA_APPROACH_MAX_KNOTS", "120")))
"""Knot cap on the APPROACH leg (standoff reach), on top of the skim knife: 1581 @0.15
executed a clean 143-knot approach that swung through the bottle (moved 20 cm) before
the grasp leg ran; 1679's good reach was 89 knots. Empty = no cap (as shipped before
2026-09-06). Over the cap the leg is refused and the ladder re-draws or re-aims."""
"""Metres the condiments are inflated by while the approach is planned (K76).

Overridable with `MIKASA_GRASP_PAD` so it can be swept without editing code, exactly as
`MIKASA_PLANNING_TIME` is (`extand.py`) — K79c needs an A/B of this constant and the two
arms must otherwise run byte-identical code.

Must exceed the controller's tracking error and stay under the 0.10 m standoff. The
measured error is 2-7 deg of joint lag, 7-9 cm of Cartesian deviation at a 0.7 m
extension, against approach paths OMPL itself reports as "slightly touching ... but it
was successfully fixed" — 0.9 cm of skin clearance on one measured seed. 3 cm invalidates
that homotopy while leaving 7 cm of standoff, so the pre-grasp pose stays reachable.

**Widening it to 5 cm was measured and is worse** (K79c). The argument for widening is
sound and came from two per-seed post-mortems: the proxy radius at 3 cm is `ext/2 + pad`
= 5.6 cm against executed deviations of 6.4 cm, which is how the gripper clips a *standing*
shaker on seed 61 and knocks it flat — after which no IK exists for the fallen body and
every later refusal is honest. 5 cm keeps the radius (7.6 cm) inside the 10 cm standoff, so
it is legal. Measured anyway, 180 seeds per arm interleaved in one pool:

| | dev 0-79 | held 80-179 | total |
|---|---|---|---|
| **0.03 (shipped)** | **77/80** | **95/100** | **172/180** |
| 0.05 | 74/80 | 93/100 | 167/180 |

Worse, and it grows failure modes the 3 cm arm does not have: 2x `FAILED: drive to bowl
dock` and 2x `FAILED: carry pose unreachable`. A wider proxy picks a different approach
path, which lands a different grasp, which cascades into stages downstream of it — K65's
finding that perturbing this planner's path reshuffles rather than reduces. The clipping is
real; inflating the obstacle is not the answer to it."""

GRASP_REREAD_REPORT_M = 0.02
"""Object drift, in metres, past which the ladder says out loud that it is re-aiming.

Below this the re-read is a no-op worth no trace line; above it the object was knocked
by the attempts that failed, and the trace is how a sweep shows which seeds were
reaching for a stale pose (K63: seeds 45 and 51 measured 10.4 cm and 17.3 cm)."""

PLANNING_TIME_S = 6.0
"""RRT wall-clock budget for this oracle only (`common.planning_budget`).

The solver ships 2 s (`extand.PLANNING_TIME`, shared by all eight oracles). Measured
here on the randomized task, 100 held-out seeds per cell, two independent runs each:

| budget | run a | run b | seeds failing in either |
|---|---|---|---|
| 2 (solver default) | 93/100 | 94/100 | 8 |
| **6** | **99/100** | **99/100** | **1** |
| 20 | 97/100 | 95/100 | 5 |

Not "more is better" — 20 is worse than 6 and less stable. What 6 buys is *variance*:
`seed_everything` already seeds mplib's C++ RNG, so the draw is fixed per seed and the
only thing load changes is how many iterations fit in the budget. At 2 s that decides
eight seeds; at 6 s it decides none, and both runs fail the same single seed (80, a
reachability failure that no budget fixes -- K79). Costs +4.5% median episode wall clock
and halves the refusal count (648 -> 363 over 180 episodes), which also cuts grasp-ladder
retry rungs by 71% (221 -> 63)."""

GRASP_LADDER_STEP_BUDGET = int(os.environ.get("MIKASA_LADDER_BUDGET", "400"))
"""Steps the grasp ladder may spend before it gives up and says so.

A refused plan is free, but a plan that runs and then fails to grasp is not, and
neither is the `open_gripper` before each attempt. On held-out seed 16 the ladder
walked far enough to spend the episode: the verdict came back `truncated`, which
books a planning problem as a horizon problem and is exactly the confusion D6 exists
to prevent. With the budget the same episode fails as `no plan`, which is what it is.

K58 measured 200 against 450 and found it **never the binding constraint** — byte-identical
scores, the same four failing seeds. That was without `APPROACH_STRETCH`. With the approach
commanded at half speed every executed rung costs twice as many steps and out-of-budget
deaths double (6 -> 12 over 180 episodes) while truncations stay at 1, so the budget is what
the stretch spends and the two have to be measured together. Overridable with
`MIKASA_LADDER_BUDGET`. **400 since 2026-09-07 (W30d)**: the four-rung ladder's executed
approaches ran two seeds out of a 200-step budget before the top rung; at 400 the medians
of successful episodes are unchanged (449 steps)."""

GRASP_LIFT_RUNGS = tuple(
    float(v) for v in os.environ.get("MIKASA_GRASP_LIFT_RUNGS", "0,0.01,0.02,0.025").split(","))
"""Extra grip heights tried after every wrist yaw at the rung below has been refused.

**(0, 0.01, 0.02, 0.025) since 2026-09-07 (W30d), was (0, 0.025).** On a grip 3 cm below
the top these are 3.0 / 2.0 / 1.0 / 0.5 cm below it; the old pair jumped straight from
the body to the lip, and the lip grip on the 9.4 cm shaker dropped the object on the
lift in a quarter of the episodes that reached it (W30: 4 of 4 population losses on
2026-09-06). Measured on 3400-3599, one pool, paired (the scene is the same in every
arm): (0,0.025)/200 **193/200**, (0,0.0125,0.025)/300 **195/200**, this ladder/400
**196/200**, b=0 against both, cleanliness medians identical (steps 449, RRT share
0.48, roll 542 in all three). The middle rungs took every grasp the lip used to take
(9 vs 8) and dropped none; the lip stays as the last resort. The ladder budget goes
with it (`GRASP_LADDER_STEP_BUDGET`): at 200 the extra rungs' executed approaches
(65-91 knots each) ran two seeds out of budget before the top rung (3113, 3131);
at 300/400 none. Sweepable through `MIKASA_GRASP_LIFT_RUNGS`.

Seed 7's refusals are mostly `collision wrist_flex_link <-> counter_main`: the shaker
is 9.4 cm tall on a 92 cm counter, so a level reach puts the wrist within a couple of
centimetres of the worktop and the link sweeps through it. Gripping 2.5 cm higher on
the same body lifts the whole hand by the same amount. Refused plans cost no episode
steps — only wall clock — so the rung is close to free when it is not needed."""
GRIP_MIN_DEPTH = float(os.environ.get("MIKASA_GRIP_MIN_DEPTH", "0.0"))
"""The ladder's raised rungs never put the TCP closer than this to the object's live top
(1916 @0.15: a grip on the cap's last 5 mm slid out on a straight lift). Measured: 0.02
was neutral on 1900-2099 (HP2, b=0 c=0) and LOST 1671 on 1500-1699 (HP5: the dz=+0.025
rung at 0.994 instead of 1.009 never grasped and the ladder ran out of budget), while
1916 itself is carried by the stove back-dock. 0 = no clamp (the default since 2026-09-06
evening); >0 clamps."""

REREAD_BEFORE_RETRY = os.environ.get("MIKASA_REREAD_RUNGS", "0") == "1"
"""Re-read the object before the *same-closing* and *flipped-closing* retries (K82).

The same oversight K76 records one block later, and the trajectories say it still bites.
Cross-referencing `shaker_trajectory.csv` against the event log on the five stable
failures, the slide and the topple are **different events, 5 to 75 steps apart**:

| seed | first slide > 1 cm | topple |
|---|---|---|
| 61 | the first attempt, step 165 | `turned and raised`, 185 |
| 71 | `flipped closing`, 485 | `turned and raised`, 560 |
| 129 | `same closing`, 230 | that rung, 235 |
| 136 | the first attempt, step 230 | `turned and raised`, 250 |
| 177 | the first attempt, step 515 | `flipped closing`, 570 |

In three of five the object is nudged by the **first** attempt and only tipped by a later
rung — and rungs 2 and 3 reach for where it used to be, because `obb`/`mesh`/`z_grasp` are
read once before rung 1 and the first live re-read is at the top of the yaw ladder (K63),
with the from-above rung fixed separately (K76). Closing on a 5.1 cm object a centimetre
or more off-centre is what levers it over, so the stale rungs are not merely wasted, they
are plausibly the cause of the state nothing can recover from (K81).

Distinct from K55's withdrawn `try_grasp_adaptive`, which re-read *inside* one attempt,
after the reach and before the close, and could not help because by then the hand was
already committed. This re-reads at a rung boundary, where K63 and K76 both measured it
paying. Byte-identical whenever the object has not moved: the same OBB yields the same
pose and the same plan."""

SKIP_TOPDOWN_RESCUE = os.environ.get("MIKASA_SKIP_TOPDOWN", "0") == "1"
"""Skip the from-above rescue rung entirely (K81). 0 = shipped.

The rung has **never** rescued an episode: 0 successes in 60 attempts across three
180-seed sweeps, against 231 grasps the side ladder wins. Measured why, at the recorded
toppled poses of seeds 129/136/177 and with the oracle's own
`grasp_geometry`/`raise_grasp_to`/`grasp_yaw_candidates`: over six wrist yaws at heights
from 2 to 16 cm above the worktop, from the episode-start posture and with
`n_init_qpos=100`, the family has **0 of 12 poses with IK at every height**. Horizontal
grasps of a toppled shaker have none either. There is no height to move it to.

Four ways of making the rung work were built and all refuse — raising it into the band
above the object, dropping the toppled target's own keep-out (which blocks the start
state, `gripper_link<->keepout_0 after 1 step`), 100 IK seeds, and re-homing the arm to
its start posture through the new `move_to_qpos`. Kept as a flag rather than a deletion
because the rung costs episode steps in seeds that are budget-bound, which is the one
thing about it still worth measuring."""

GRASP_YAWS_DEG = (0.0, 30.0, -30.0, 60.0, -60.0, 90.0)
"""Wrist yaws tried at the grasp, in order; 0 is the OBB's own choice."""

POUR_TILT_DEG = float(os.environ.get("MIKASA_POUR_TILT_CMD_DEG", "165.0"))
POUR_TILT_STRONG_DEG = float(os.environ.get("MIKASA_POUR_TILT_STRONG_DEG", "175.0"))
"""The tilt ladder's two rungs. 65 / 90 until 2026-09-08, when the predicate went from
55 to 155 deg (the owner: a pour is an inverted shaker): the same 10 deg margin over
the predicate, and the strong rung near the ceiling (180 = exactly upside down).

The original reasoning, still the reasoning: the second rung is tried only after all
four first-rung candidates.

The predicate wants 55 deg (`cfg.pour_tilt_deg`) and the oracle commands 65, a 10 deg
margin — smaller than the tracking error the arm actually shows at full stretch. On
seed 2 a candidate planned and executed and the bottle came to rest at 47.5 deg: the
pose was reached as well as the arm could reach it, and the episode was lost by 7.5
deg. Commanding 90 asks for a pose no predicate needs, so that falling as short as
seed 2 fell still clears 55. Appended rather than substituted: the four 65 deg
candidates are what every passing seed uses, and the loop breaks on the first one
that satisfies the flags."""

# Clearance the lift buys over the tallest condiment (K40): the base turn carries the
# held object over its neighbour.
LIFT_BY_TORSO = os.environ.get("MIKASA_LIFT_BY_TORSO", "1") == "1"
LIFT_MAX_KNOTS = (None if os.environ.get("MIKASA_LIFT_MAX_KNOTS", "100") == "" else
                  int(os.environ.get("MIKASA_LIFT_MAX_KNOTS", "100")))
LIFT_KNOT_DRAWS = int(os.environ.get("MIKASA_LIFT_KNOT_DRAWS", "4"))
HOVER_MAX_KNOTS = (None if os.environ.get("MIKASA_HOVER_MAX_KNOTS", "100") == "" else
                   int(os.environ.get("MIKASA_HOVER_MAX_KNOTS", "100")))
HOVER_KNOT_DRAWS = int(os.environ.get("MIKASA_HOVER_KNOT_DRAWS", "3"))
CARRY_MAX_KNOTS = (None if os.environ.get("MIKASA_CARRY_MAX_KNOTS", "100") == "" else
                   int(os.environ.get("MIKASA_CARRY_MAX_KNOTS", "100")))
CARRY_KNOT_DRAWS = int(os.environ.get("MIKASA_CARRY_KNOT_DRAWS", "3"))
HOVER_CORRECT = os.environ.get("MIKASA_HOVER_CORRECT", "1") == "1"
PRE_HOVER_UP = float(os.environ.get("MIKASA_PRE_HOVER_UP", "0.10"))
PRE_HOVER_REFUSE = os.environ.get("MIKASA_PRE_HOVER_REFUSE", "1") == "1"
"""The pre-hover waypoints are REFUSED over the knot cap (not executed as the shortest of
the draws): 1507 @0.15 executed a 291-knot raised waypoint and swept the bowl off the
counter. Refused, the ladder tries the level waypoint, then the hover directly."""
RUNG_BACK_OFF = os.environ.get("MIKASA_RUNG_BACK_OFF", "1") == "1"
"""Before each grasp-ladder rung that follows an EXECUTED attempt, open and back the hand
off the object (up ARC_RETREAT_M, else along the approach) — 1679 @0.15: three reached
closes missed and nine rungs were then refused `gripper_link <-> keepout_0 after 1 step`,
the hand's start inside the target's own approach proxy. 0 = rungs plan from where the
last close left the hand, as shipped before 2026-09-06."""
DRIVE_ALLOW_HELD_TOUCH = os.environ.get("MIKASA_DRIVE_ALLOW_HELD_TOUCH", "1") == "1"
"""After the carry tuck, a drive refused `<arm link> <-> <held object>` at its first step
re-attaches the object with that link allowed and drives once more (2025 @0.15: the
tucked bottle rests on the shoulder in the model). 0 = the refusal stands."""
STOVE_BACKDOCK_M = float(os.environ.get("MIKASA_STOVE_BACKDOCK", "0.15"))
STOVE_BACKDOCK_MIN_HITS = int(os.environ.get("MIKASA_STOVE_BACKDOCK_HITS", "2"))
"""When a grasp attempt is refused against the STOVE (`forearm_roll_link <-> stove`), back
the base straight off by this much once, re-aim from the live object and retry both
closings before the ladder goes on: 1916 @0.15 (2026-09-06) — the shaker against the stove,
every level grasp leg refused there, the same seed at a dock 0.15 m further back grasped
and poured (g62a). 0 = off. The rescue costs a drive and two closings, so it waits for
STOVE_BACKDOCK_MIN_HITS refusals naming the stove (HP4 @0.15, 1652: one incidental stove
refusal fired it on a seed the ladder was carrying, and the budget ran out at 298)."""
GRASP_TORSO_RESERVE = float(os.environ.get("MIKASA_GRASP_TORSO_RESERVE", "0.0"))
"""EXPERIMENT (off): torso left under its stop before the grasp approach, frozen through
the reach and grasp legs, so the lift can raise the torso as a joint line. 0 = the shipped
grasp (the approach may spend the torso to its stop)."""
LIFT_HANG_AWARE = os.environ.get("MIKASA_LIFT_HANG_AWARE", "1") == "1"
"""The lift also clears the held object's BOTTOM over the tallest neighbour (see the lift):
K40's 0.10 margin was taken from the TCP, and a 16 cm bottle hangs 13 cm under it. 0 = the
TCP-only margin, as shipped before 2026-09-06."""
LIFT_BOTTOM_CLEAR = float(os.environ.get("MIKASA_LIFT_BOTTOM_CLEAR", "0.05"))
"""Clearance of the held object's bottom over the tallest neighbour at the lift target (the
rungs go down to 0.02). 1847 swept the neighbour with the bottom 1.6 cm over its top."""
"""The pre-hover waypoint sits this much ABOVE the hover height (metres; 0 = level with
it, as before 2026-09-06). Measured with the bowl read after the drive, after the
pre-hover and after the hover (1619, 1867 @0.15): the bowl moved 12-15 cm during the
PRE-HOVER and not after — the transit swings the held condiment through the bowl's rim
although the bowl sits in that leg's keepout (the executed path is not the planned one).
A transit 10 cm higher keeps the bottle's bottom above the rim; the hover then descends."""
"""After the hover, when the object is NOT over the bowl although the TCP reached its
target: re-read the object's pose in the hand and re-aim the hover once with it, and
pour with that live offset. Measured (1619, 1867 @0.15; 452 @0.0): tcp_err 0.009-0.017 m
on the hover and the object 12-15 cm from the bowl — the offset read after the drive was
stale by the time the arm got there (the pre-hover/hover swings shift the object in the
fingers). All eight pour candidates then miss the bowl by construction (1867 rode the
horizon out on them)."""
"""The carry-pose tuck's knot cap (the recovery when the drive to the bowl dock refuses
with the arm out). 2026-09-06-hovercap @0.15, seed 1595: the tuck to yaw 90 was a 206-knot
RRT path (10 s) with the condiment in hand and the object was on the floor before the
hover — the same K59 class as the pour tips, the lift and the hover. `carry_pose`
already takes the cap; SeasonDish now passes it. "" disables."""
"""The pre-hover and hover moves' knot cap and draws — the same K59 medicine as the pour
tips'. 2026-09-05-dock2 @0.15, seed 1515: the pre-hover was a 139-knot RRT path (7 s) with
the condiment in hand, and the object was on the floor before the tilt ("dropped during
the tilt" was only the checkpoint that noticed: hovering clearance -0.954). "" disables."""
"""The RRT lift's knot cap and draws when neither a screw rung nor the torso can lift
(377 @0.15: the torso already at 0.368 of 0.386 at the grasp). K59: every measured drop
rode a path of >= 173 knots and none under; the cap draws up to LIFT_KNOT_DRAWS paths,
takes a draw under LIFT_MAX_KNOTS at once, else the shortest. "" disables the cap."""
"""When no screw-reachable lift rung exists, raise the TORSO by the rise as a joint line
before falling to the RRT. Measured (2026-09-05-dock, D15 @0.15): seeds 363 and 377 lost
the condiment during the lift — the probe exhausted, the RRT lift swung the arm and the
object hit the counter and flew (object_z 0.003) — the owner's "rotates the arm a lot
lifting". The torso moves the hand straight up with the arm as it stands. K79q measured
that REFUSING the RRT lift costs more than it saves (166/180 vs 173/180); this adds a
cleaner channel in front of it and leaves the RRT as the fallback."""
LIFT_OVER_NEIGHBOUR = 0.10
LIFT_ABOVE_GRASP = 0.15

# The grasp sits this far below the object's top rather than at the OBB centre: a
# horizontal grasp at the shaker's centre (z 0.967, 4.7 cm over the counter) has the
# wrist links inside the counter for every IK solution (measured, seed 11 — `IK
# Failed`); at top − 0.03 (0.984) the reach and the grasp both plan, and the bottle
# (top 1.077) grasps at 1.047. Applied as `max(centre, top − GRASP_BELOW_TOP)`.
GRASP_BELOW_TOP = 0.03


def _np(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def say(env, stage: str, **extra) -> None:
    """`oracle_common.say` tagged with this oracle's name."""
    common.say(env, WHO, stage, **extra)


def fail(env, stage: str, **extra):
    """`oracle_common.fail` tagged with this oracle's name."""
    return common.fail(env, WHO, stage, **extra)


# Scoping the TOPP slowdown to the approach leg alone was attempted and is IMPOSSIBLE
# through mplib (K79g). `self.joint_vel_limits` is read only inside `setup_planner`, which
# hands the values to the mplib planner at construction; `plan_qpos`/`plan_pose` take no
# per-call limits and mplib exposes no setter. A context manager mutating the Python
# attribute mid-episode is a **no-op** — it produced byte-identical results across a
# 360-episode A/B (same score, same failing seeds, same truncations), which is how it was
# caught. Removed rather than left in place, because a scoping helper that silently does
# nothing is worse than no helper. Per-leg limits would need the planner rebuilt mid-episode
# or a patch to mplib.
#
# Lowering this oracle's TOPP limits *globally* was built and withdrawn (K79f). It is the only lever
# tried all session aimed at the *execution* error rather than the path, and the K79c
# post-mortems' own numbers point at it: 2-7 deg of joint lag gives 7-9 cm of Cartesian
# deviation at 0.8 m extension, against collision margins of 3-6 cm — which is how an
# approach the planner believes is collision-free clips a standing object. Slower
# following really does mean less lag. Measured anyway, 180 seeds per arm interleaved:
#
#   | limits | dev   | held   | total   | truncated |
#   |--------|-------|--------|---------|-----------|
#   | 0.9    | 77/80 | 96/100 | 173/180 |  1/180    |
#   | 0.5    | 74/80 | 87/100 | 161/180 | 11/180    |
#
# Much worse, and the mechanism is the last column rather than the grasp: slowing the
# trajectory spends the horizon. The median success uses 323 of 1100 steps, but the
# episodes that are already long are exactly the ones halving the speed pushes over.
# That is K55's turn-drive-turn objection in a new place.
# `common.default_planner_factory` keeps its `joint_vel_limits`/`joint_acc_limits`
# pass-through (default None -> a byte-identical solver) so the A/B stays cheap to repeat.
JOINT_LIMIT_SCALE = float(os.environ.get("MIKASA_JOINT_LIMITS", "0.9"))
"""TOPP limits for this oracle's solver; 0.9 is the solver's own default (K79f/K79g)."""


def default_planner_factory(env, debug: bool, vis: bool):
    """The real Fetch solver at the oracle refinement cap (`oracle_common`)."""
    return common.default_planner_factory(
        env, debug, vis,
        joint_vel_limits=JOINT_LIMIT_SCALE, joint_acc_limits=JOINT_LIMIT_SCALE,
    )


def default_grasp_info(obb, ee_direction, target_closing) -> dict:
    """OBB thin-side grasp frame (`oracle_common.default_grasp_info`)."""
    return common.default_grasp_info(obb, ee_direction, target_closing, FINGER_LENGTH)


CYL_FOOTPRINT_TOL = 0.15
"""Footprint aspect ratio inside which the wrist yaw is treated as free (K59).

Measured on the assets this task loads: the shaker's OBB footprint is 0.0509 x 0.0517 m
(1.5% apart), the bottle's 0.0455 x 0.0458 m (0.8%). At that margin the `np.argsort`
in `compute_box_grasp_thin_side_info` is resolving sub-millimetre mesh noise, not a
shape, and the axis it picks rides the object's uniformly random spawn yaw. Anything
more elongated than this keeps the OBB thin-side frame, where the jaws genuinely must
close across the short axis and the yaw is not ours to spend."""


def approach_aligned_grasp_info(obb, ee_direction, target_closing) -> dict:
    """Grasp frame for a body of revolution: approach along `ee_direction`, jaws across it.

    Why (K59): a parallel jaw closing on a 5.1 cm circle grips identically at every
    yaw, so for these condiments the wrist yaw is a free parameter. `default_grasp_info`
    spends it anyway — it aims the approach down the OBB's *longer* horizontal axis,
    which for a near-round footprint is decided by mesh noise and tracks the spawn yaw.
    Measured over 30 seeds x 2 objects, the angle between that axis and the base->object
    ray is uniform: mean 46.7 deg, median 46.7, 73% above 30 deg, 8% below 10. The arm
    was being sent to a wrist yaw ~47 deg off the line it reaches along, for nothing.

    Falls back to `default_grasp_info` when the footprint is elongated (the yaw is not
    free) or when `ee_direction` has no horizontal part — the top-down rescue rung
    passes ``[0, 0, -1]`` and must keep the frame it was measured with.

    The grip depth uses the footprint's mean radius rather than the OBB's support width
    along `approaching`: for a near-circular footprint that width grows toward the OBB
    corners, which would grip a chord instead of the diameter at 45 deg.

    Example:
        >>> import numpy as np, trimesh
        >>> obb = trimesh.primitives.Box(extents=[0.051, 0.052, 0.094])
        >>> info = approach_aligned_grasp_info(obb, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        >>> np.round(info["approaching"], 3).tolist()
        [1.0, 0.0, 0.0]
        >>> np.round(info["closing"], 3).tolist()
        [-0.0, 1.0, 0.0]

        An elongated footprint keeps the OBB frame:
        >>> box = trimesh.primitives.Box(extents=[0.03, 0.09, 0.10])
        >>> a = approach_aligned_grasp_info(box, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        >>> b = default_grasp_info(box, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        >>> bool(np.allclose(a["approaching"], b["approaching"]))
        True
    """
    extents = np.asarray(obb.primitive.extents, dtype=np.float64)
    T = np.asarray(obb.primitive.transform, dtype=np.float64)
    short, long_ = float(min(extents[:2])), float(max(extents[:2]))
    a = np.asarray(ee_direction, dtype=np.float64) * np.array([1.0, 1.0, 0.0])
    n = float(np.linalg.norm(a))
    if long_ - short > CYL_FOOTPRINT_TOL * long_ or n < 1e-6:
        return default_grasp_info(obb, ee_direction, target_closing)
    approaching = a / n
    closing = np.array([-approaching[1], approaching[0], 0.0])
    if target_closing is not None and float(np.asarray(target_closing, dtype=np.float64) @ closing) < 0:
        closing = -closing  # parallel jaws: theta and theta+180 close identically
    radius = 0.25 * (float(extents[0]) + float(extents[1]))
    center = T[:3, 3] + approaching * (-radius + min(FINGER_LENGTH, radius))
    return dict(approaching=approaching, closing=closing, center=center, extents=extents)


def wait_cue(env, planner, info):
    return common.wait_cue(env, planner, info, who=WHO)


def choose_target(info, blind: bool, rng: np.random.Generator) -> bool:
    """`target_is_shaker`: the answer from `info`, or arm B's uniform draw.

    The one line the blind arm replaces (the burner's `choose_target` pattern):
    arm B ignores the cue and grasps a uniformly random condiment —
    `rng.integers(2)` from `np.random.default_rng(seed)`, so the draw is
    deterministic per seed and both objects appear over the eval seeds — and it
    never touches the key. Called only after `wait_cue`.

    Example:
        >>> choose_target({"target_is_shaker": np.array([False])}, blind=False, rng=np.random.default_rng(0))
        False
        >>> [choose_target({}, blind=True, rng=np.random.default_rng(s)) for s in range(4)]
        [True, False, True, True]
    """
    if blind:
        return bool(rng.integers(2))
    return bool(_np(info["target_is_shaker"]).reshape(-1)[0])


def grasp_geometry(task, obb, ee_direction, target_closing, grasp_info=default_grasp_info):
    return common.grasp_geometry(task, obb, ee_direction, target_closing, grasp_info)


def stretch_for(obj, grasp) -> int:
    """`APPROACH_STRETCH`, or 1 when the grasp is too high off the worktop to need it."""
    if APPROACH_STRETCH <= 1 or STRETCH_MAX_CLEARANCE <= 0.0:
        return APPROACH_STRETCH
    mesh = obj.get_first_collision_mesh(to_world_frame=True)
    if mesh is None:
        return APPROACH_STRETCH
    return APPROACH_STRETCH if (float(grasp.p[2]) - float(mesh.bounds[0][2])) <= STRETCH_MAX_CLEARANCE else 1


FULL_DOF_REACH = os.environ.get("MIKASA_FULL_DOF_REACH", "0") == "1"
"""Let the BASE join the reach when the arm alone cannot plan it. 0 until measured.

Aimed at the measured root of this task's remaining losses rather than at their symptom.
Four of the five failures on 200 unseen seeds begin with `the object has moved` — and the
plan is not what touches it: the approach is planned with the condiments inflated 3 cm,
giving 5.6 cm of proxy radius, against an EXECUTED deviation of 6.4 cm (K79c). That
deviation is joint lag, and joint lag grows with extension: the 6.4 cm was measured at a
0.7 m reach. A base that steps in does not need the extension.

Every lever that attacks the symptom has now been measured and none pays: keepout pad at
5 cm worse (K79c), three approach draws worse (K80), narrow approach a wash, abort on
touch 0 of 6, the skim refusal keeps only the gross cases, the slowed approach worse and
truncating. This one attacks the cause — and it is a wash too.

**MEASURED 2026-09-05** (`runs/2026-09-05-fulldof`, paired, 200 seeds per arm, one commit):

    off   194/200    missed 3  no plan 3  truncated 0
    on    194/200    missed 2  no plan 3  truncated 1
    b = 1  c = 1

The reasoning holds and the lever does not reach it: `full_dof_reach` fires only AFTER
the arm-only reach has refused, and the losses here are not refusals — they are reaches
that succeed and knock the object on the way. To attack the extension it would have to be
the FIRST reach, not the fallback, and that is a different change with a different cost.
Left off, and the finding recorded so the next attempt starts from it."""

APPROACH_SKIM_CAP = float(os.environ.get("MIKASA_SD_SKIM_CAP", "0.05"))
"""Refuse an approach draw that is inside the scene for more than this share of its knots.

The two condiments stand on a counter with nothing to catch them, and an approach that
runs through the scene does not merely look bad: it MOVES them, and after that no grasp
exists at all. Measured on HELD seed 204 — the executed draw had 98 of its 191 knots
intruding (two of the three draws did not plan, so the knife had nothing better to pick),
the bottle ended 11.2 cm away, every wrist yaw and grip height then refused, and the
episode was lost with `no plan`. With the refusal the ladder gets an intact scene, plans
a clean 55-knot path, and the grasp holds.

Asked for HERE and nowhere else. As a global default the same rule refused 257 plans over
30 DepthRecall episodes and took that task from 26/30 to 13/30: an arm working inside a
shelf intrudes by this metric as a matter of course, and there the intrusion is the job.

**0.05, not the third I first chose.** A third was read off the gross case — 98 knots of
191 — and the gross case is not what costs this task its seeds. On the 200-seed verdict
run EVERY ONE of the five failures executed an intruding path, at 0.12, 0.13, 0.29, 0.05
and 0.11 of its knots, all comfortably under a third; and only twelve paths in 200
episodes intrude at all, so the signal sits in five failures of five against seven other
episodes. Swept paired, 200 seeds per arm on one commit (`runs/2026-09-05-skimcap`,
`-skimcap2`):

    cap 1/3     194/200      b = 0   c = 2   against 0.05
    cap 0.05    196/200
    cap 0.0     195/200      b = 1   c = 0   against 0.05

0.05 takes two seeds and gives none back. Refusing EVERYTHING is worse by one: some
paths graze the target legitimately at the very end, and refusing those spends a rung for
nothing."""


def held_touch_link(refusal: str, held_stem: str) -> str | None:
    """The robot link a refusal names against the HELD object, as a name stem, or None.

    `refusal` is the solver's printed status from `capture_refusal("collision ")`, e.g.
    `collision scene-0-ds_fetch_shoulder_lift_link<->scene-0_condiment_bottle_112 after
    1 step(s)`; `held_stem` names the held actor ("shaker" / "condiment_bottle").
    Returns "shoulder_lift_link" when one side is a robot link and the other the held
    actor; None for any other pair (a wall, a fixture, the distractor).

    Example:
        >>> held_touch_link("collision scene-0-ds_fetch_shoulder_lift_link<->scene-0_condiment_bottle_112 after 1 step(s)", "condiment_bottle")
        'shoulder_lift_link'
        >>> held_touch_link("collision scene-0-ds_fetch_l_gripper_finger_link<->scene-0_wall_left_room_0_31 after", "condiment_bottle") is None
        True
    """
    m = re.search(r"collision\s+(\S+?)<->(\S+?)(?:,|\s|$)", str(refusal))
    if not m or not held_stem:
        return None
    for a, b in ((m.group(1), m.group(2)), (m.group(2), m.group(1))):
        if "ds_fetch_" in a and held_stem in b and "ds_fetch_" not in b:
            return a.split("ds_fetch_", 1)[1]
    return None


def back_off_from_the_object(env, planner, task, *, why: str):
    """Open and back the hand off the object it stands against; the retreat's result.

    Up by ARC_RETREAT_M first (`lift_hand`, a pure screw), else along the hand's own
    approach axis through `static_manipulation` (which has the RRT fallback). The grasp
    ladder leaves the hand where its last close missed — against the object — and mplib
    will not plan out of a start it considers in collision: with the target's own proxy
    inflated for the approach leg (K76), every later rung is refused at its first step
    (1679 @0.15, 2026-09-06). The arc always backed off first (K79); the rungs do too.

    Example:
        >>> r = back_off_from_the_object(env, planner, task, why="a missed close")  # doctest: +SKIP
        >>> if r != -1 and common.stopped_by_horizon(planner): return r          # doctest: +SKIP
    """
    say(env, "backing the hand off the object", why=why)
    planner.open_gripper()
    retreat = planner.lift_hand(delta_h=ARC_RETREAT_M)
    if retreat == -1:
        tcp_now = task.agent.tcp.pose[0].sp
        back = sapien.Pose(p=tcp_now.p, q=tcp_now.q) * sapien.Pose([0, 0, -ARC_RETREAT_M])
        say(env, "retreat refused upward; backing off along the approach", why=why)
        retreat = planner.static_manipulation(back, disable_lift_joint=False)
    return retreat


def _say_close_miss(env, task, obj):
    """Instrument: the fingers closed and `is_grasping` is False — where did they close?

    Prints the finger gap and the TCP against the object's live centre and top. Quiet
    on a double without joints or meshes.

    Example:
        >>> _say_close_miss(env, task, task.shaker)  # doctest: +SKIP
    """
    try:
        robot = task.agent.robot
        jm = robot.active_joints_map
        q = _np(robot.get_qpos()).reshape(-1)
        gap = sum(float(q[int(jm[n].active_index[0])])
                  for n in ("l_gripper_finger_joint", "r_gripper_finger_joint"))
        tcp = _np(task.agent.tcp.pose.p).reshape(-1)[:3].astype(np.float64)
        mesh = obj.get_first_collision_mesh(to_world_frame=True)
        c = np.asarray(mesh.bounding_box_oriented.center_mass, dtype=np.float64)
        say(env, "close missed", finger_gap=round(gap, 3),
            tcp_minus_centre=[round(float(v), 3) for v in (tcp - c)],
            tcp_below_top=round(float(mesh.bounds[1][2]) - float(tcp[2]), 3))
    except Exception as exc:      # a double without joints/meshes: the instrument is quiet
        say(env, "close missed", instrument=f"unavailable ({type(exc).__name__})")


def try_grasp(env, planner, task, obj, grasp, reach, target_pad: float | None = None,
              by_line: bool | None = None):
    """`common.try_grasp`, with both condiments inflated for the approach leg (K76).

    Both, not just the distractor: six of nine failing seeds on the randomized layout
    were an approach path skimming one of them, and on seed 78 the *distractor* was
    toppled first and then fell into the corridor to the target.
    """
    # A wider pad for the distractor alone was written and NOT adopted (K79c). The
    # argument for it is real — `keepout`'s standoff ceiling binds only the object being
    # approached, and on seed 71 a leg with `tcp_err=0.064 m` pushed the distractor
    # 0.1136 m inside a 0.03 m pad. But seed 61 shows the opposite cost at the same site:
    # its ladder rungs were refused `gripper_link<->keepout_1` **after 1 step**, i.e. the
    # hand's own START state sat inside the distractor's proxy, and mplib will not plan
    # out of a start it considers in collision. The two effects run opposite ways and it
    # is unmeasured, so the single pad stays. `common.keepout` accepts per-actor pads now.
    # An "above-then-descend" approach shape was built and withdrawn (K79e). Both K79c
    # post-mortems blame the path *shape*: RRTConnect lifts the TCP high, descends onto the
    # worktop, then sweeps sideways past the target — seed 61's TCP passes 9.7-13.4 cm from
    # the shaker's axis and a finger clips it. So aim the search 10 cm ABOVE the standoff,
    # where a sideways sweep is harmless, and close the last stretch with `lift_hand` — a
    # pure vertical `plan_screw`, no RRT fallback, geometrically unable to sweep sideways.
    # Measured, 180 seeds per arm interleaved: shipped **175/180**, above-then-descend
    # **172/180**, with an almost disjoint failing set and two new stage failures
    # (`no pour pose reached`, `distractor moved during the drive`). Withdrawn.
    res, grasped = common.try_grasp(
        env, planner, task, obj, grasp, reach,
        keepout_actors=[task.shaker, task.condiment_bottle],
        # Per-actor since K79c; the target keeps the pad an approach leg can live with,
        # the other condiment may take a wider one (it is never approached).
        keepout_pad=[(APPROACH_TARGET_PAD if target_pad is None else float(target_pad))
                     if a is obj else DISTRACTOR_KEEPOUT_PAD
                     for a in (task.shaker, task.condiment_bottle)],
        resync_before_grasp=RESYNC_BEFORE_GRASP, n_init_qpos=GRASP_N_INIT_QPOS,
        close_on_contact=CLOSE_ON_CONTACT, approach_draws=APPROACH_DRAWS,
        grasp_max_knots=GRASP_LEG_MAX_KNOTS, grasp_knot_draws=GRASP_LEG_KNOT_DRAWS,
        approach_max_knots=APPROACH_MAX_KNOTS, approach_clearance_pad=APPROACH_CLEARANCE_PAD,
        grasp_stop_on_touch=GRASP_STOP_ON_TOUCH, grasp_stretch=GRASP_LEG_STRETCH,
        approach_by_line=(APPROACH_BY_LINE if by_line is None else bool(by_line)),
        freeze_torso=GRASP_TORSO_RESERVE > 0.0,
        approach_stretch=stretch_for(obj, grasp), approach_stretch_tail=APPROACH_STRETCH_TAIL,
        narrow_approach=NARROW_APPROACH, abort_on_touch=ABORT_ON_TOUCH,
        approach_aperture=APPROACH_APERTURE, approach_skim_cap=APPROACH_SKIM_CAP,
        full_dof_reach=FULL_DOF_REACH,
    )
    if res != -1 and not grasped and not common.stopped_by_horizon(planner):
        _say_close_miss(env, task, obj)
    return res, grasped


_try_grasp_with_pad = try_grasp      # the stage shadows `try_grasp` with its own pad


def hold_object_in_planner(env, planner, task, obj, held: bool) -> None:
    common.hold_object_in_planner(env, planner, task, obj, held, who=WHO)


def dock_for_target(dock_xyyaw, target_xyz):
    """`common.dock_for_target` — the dock slid along the counter to the target."""
    return common.dock_for_target(dock_xyyaw, target_xyz)


def quat_about(axis, angle_rad: float) -> np.ndarray:
    """wxyz quaternion for a rotation of `angle_rad` about the unit vector `axis`.

    Example:
        >>> np.round(quat_about([0, 0, 1], np.pi / 2), 4).tolist()
        [0.7071, 0.0, 0.0, 0.7071]
    """
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    h = float(angle_rad) / 2.0
    return np.array([math.cos(h), *(math.sin(h) * a)], dtype=np.float64)


def raise_grasp_to(grasp: sapien.Pose, reach: sapien.Pose, z: float) -> tuple[sapien.Pose, sapien.Pose]:
    """The same grasp and reach poses with the grasp point moved to height `z` (both
    shifted by the same vertical offset, so the reach stays 0.1 m back along the
    approach).

    Example:
        >>> g = sapien.Pose(p=[1.0, 2.0, 0.967]); r = g * sapien.Pose([0, 0, -0.1])
        >>> g2, r2 = raise_grasp_to(g, r, 0.984)
        >>> round(float(g2.p[2]), 3), round(float(r2.p[2] - r.p[2]), 3)
        (0.984, 0.017)
    """
    dz = float(z) - float(grasp.p[2])
    up = np.array([0.0, 0.0, dz])
    return sapien.Pose(p=np.asarray(grasp.p) + up, q=grasp.q), sapien.Pose(p=np.asarray(reach.p) + up, q=reach.q)


# `try_grasp_adaptive` — re-read the object after the reach and re-aim the grasp before
# closing — was built and withdrawn (K55). The mechanism it was built for is real and
# measured: on held-out seed 16 the reach displaces the shaker 0.108 m (matching a
# contact probe that found the gripper closing 11.4 cm away with zero finger contact),
# against 0.0009 m on seed 29. But compensating after the fact does not rescue seed 16 —
# by then the object is somewhere the arm cannot take it from — and it *cost* eval seed
# 7, which grasps through the same ladder. No benefit, measured cost. The lever that
# remains is not to knock the object at all, which is a path-following question in the
# solver rather than a grasp-geometry one.

def grasp_yaw_candidates(grasp, reach, centre_xy, angles_deg=GRASP_YAWS_DEG):
    """The same grasp, turned about the object's own vertical axis.

    Both condiments are bodies of revolution — the shaker's OBB is 0.051 x 0.052 in
    plan, a millimetre apart, so which side `compute_grasp_info_by_obb` calls "thin"
    is decided by noise, and the one wrist yaw that falls out of it is arbitrary. When
    that yaw is the unreachable one the whole episode is lost at the reach, which is
    what seeds 7 and 8 do: `IK Failed! Cannot find valid solution`, identically, on
    every approach direction tried.

    For a cylinder the grasp is a one-parameter family, so the arbitrary choice can be
    a ladder instead. Rotating about the object's vertical axis keeps the fingers at
    the same height on the same body and only turns the wrist.

    Example:
        >>> import sapien, numpy as np
        >>> g = sapien.Pose(p=[1.0, 0.0, 1.0]); r = sapien.Pose(p=[0.9, 0.0, 1.0])
        >>> cands = grasp_yaw_candidates(g, r, np.array([1.0, 0.0]), (0.0, 90.0))
        >>> np.round(np.asarray(cands[1][2].p, dtype=float), 3).tolist()
        [1.0, -0.1, 1.0]
    """
    out = []
    c = np.array([float(centre_xy[0]), float(centre_xy[1]), 0.0])
    for a in angles_deg:
        qz = quat_about(np.array([0.0, 0.0, 1.0]), math.radians(float(a)))
        R = sapien.Pose(q=qz)
        turned = []
        for pose in (grasp, reach):
            p_rel = np.asarray(pose.p) - c
            p_new = np.asarray((R * sapien.Pose(p=p_rel)).p) + c
            turned.append(sapien.Pose(p=p_new, q=np.asarray((R * sapien.Pose(q=pose.q)).q)))
        out.append((float(a), turned[0], turned[1]))
    return out


def pour_pose_for(bowl_p, above: float, obj_q, T_tcp_obj: sapien.Pose, axis_world, tilt_deg: float) -> sapien.Pose:
    """TCP pose that puts the held object's origin `above` metres over `bowl_p`, tilted
    `tilt_deg` about the horizontal world axis `axis_world` from its upright `obj_q`.

    By construction the object origin lands at `bowl_p + [0, 0, above]` (the tilt is a
    rotation about the object's own origin), so the task's `over_bowl` and
    `height_ok` hold if the plan reaches the pose, and `tilted` holds when
    `tilt_deg ≥ pour_tilt_deg` and `obj_q` is upright (yaw only — the task spawns
    yaw only). `T_tcp_obj` is the grasp transform read after the drive: "object
    origin at X" is "TCP at X · T_tcp_obj⁻¹".

    Args:
        bowl_p: the bowl's origin, world (3,).
        above: metres above it for the object origin (inside the predicate's band).
        obj_q: the object's upright world orientation, wxyz.
        T_tcp_obj: sapien.Pose, TCP → object.
        axis_world: horizontal unit vector to tilt about (world), e.g. the base's facing.
        tilt_deg: the tilt (positive or negative — two candidates per axis).

    Example:
        >>> T = sapien.Pose(p=[0.0, 0.0, -0.05])            # object 5 cm below the TCP
        >>> pose = pour_pose_for(np.array([1.0, 2.0, 0.9]), 0.2, np.array([1.0, 0, 0, 0]), T, [1, 0, 0], 65.0)
        >>> obj = pose * T
        >>> [round(float(v), 3) for v in obj.p]   # the object origin: over the bowl, `above` up
        [1.0, 2.0, 1.1]
        >>> R = obj.to_transformation_matrix()[:3, :3]
        >>> round(float(np.degrees(np.arccos(R[2, 2]))), 1)   # the object's +Z, tilted from world +Z
        65.0
    """
    q_tilt = quat_about(axis_world, math.radians(float(tilt_deg)))
    q_obj = (sapien.Pose(q=q_tilt) * sapien.Pose(q=np.asarray(obj_q, dtype=np.float64))).q
    obj_pose = sapien.Pose(
        p=np.asarray(bowl_p, dtype=np.float64) + np.array([0.0, 0.0, float(above)]), q=q_obj
    )
    return obj_pose * T_tcp_obj.inv()


def pour_candidates(face_xy, along_xy, tilt_deg: float = POUR_TILT_DEG,
                    strong_deg: float = POUR_TILT_STRONG_DEG):
    """The (axis_world, tilt_deg) candidates in the order the oracle tries them: about
    the base's facing (+, −), then about the counter's along axis (+, −) — and then the
    same four again at `strong_deg`, for the arm that reaches the pose but not the angle.

    Example:
        >>> [(a, t) for a, t in pour_candidates([0, 1, 0], [1, 0, 0])][:4]
        [([0, 1, 0], 65.0), ([0, 1, 0], -65.0), ([1, 0, 0], 65.0), ([1, 0, 0], -65.0)]
        >>> [t for _a, t in pour_candidates([0, 1, 0], [1, 0, 0])][4:]
        [90.0, -90.0, 90.0, -90.0]
    """
    rung = lambda d: [(face_xy, float(d)), (face_xy, -float(d)),
                      (along_xy, float(d)), (along_xy, -float(d))]
    return rung(tilt_deg) + rung(strong_deg)


def _flag(info, key) -> bool:
    return bool(_np(info[key]).reshape(-1)[0])


def _pour_flags(info) -> dict:
    return {k: _flag(info, k) for k in ("tilted", "over_bowl", "height_ok", "grasp_target", "distractor_ok")}


def solve(
    env,
    seed=None,
    debug=False,
    vis=False,
    blind=False,
    *,
    planner_factory=default_planner_factory,
    grasp_info=approach_aligned_grasp_info,
):
    """`_solve` under this oracle's own RRT budget (`PLANNING_TIME_S`).

    A thin wrapper rather than a `with` inside `_solve`, because `_solve` returns from
    two dozen places and the budget must be restored on every one of them.
    """
    with common.planning_budget(PLANNING_TIME_S):
        return _solve(
            env, seed=seed, debug=debug, vis=vis, blind=blind,
            planner_factory=planner_factory, grasp_info=grasp_info,
        )


def _solve(
    env,
    seed=None,
    debug=False,
    vis=False,
    blind=False,
    *,
    planner_factory=default_planner_factory,
    grasp_info=approach_aligned_grasp_info,
):
    """Season the dish with the condiment the recipe asked for. `-1` on a failed
    plan, the gym 5-tuple otherwise (a physical miss included, D6).

    The two keyword-only hooks exist for the offline test: they default to the
    real solver and the real grasp geometry.
    """
    obs, info = env.reset(seed=seed)
    if seed is not None:
        seed_everything(seed)  # Python, numpy, torch and mplib's C++ RNG (seeding.py)

    assert env.unwrapped.control_mode in (
        "pd_joint_pos",
        "pd_joint_pos_vel",
        "pd_joint_delta_pos",
    ), env.unwrapped.control_mode

    planner = planner_factory(env, debug, vis)
    task = env.unwrapped
    rng = np.random.default_rng(seed)  # the blind arm's draw, deterministic per seed

    # -- STAGE 0: sit through the cue -------------------------------------------
    info = wait_cue(env, planner, info)
    if info == -1:
        return info

    # -- STAGE 1: the privileged read (or the blind draw) ---------------------------
    target_is_shaker = choose_target(info, blind, rng)
    target = task.shaker if target_is_shaker else task.condiment_bottle
    say(env, "target chosen", target_is_shaker=target_is_shaker, blind=bool(blind))

    # The base used to slide along the counter here, to stand in front of the target's
    # own station (K55). Removed on 2026-08-24 after a 75-seed-per-arm A/B at matched
    # load: **60/75 with it, 59/75 without** — the stage buys nothing, and it is the
    # first thing anyone watching a recording sees the robot do. It earned its keep only
    # while the left condiment sat seven centimetres from the sink; `station_along`
    # moved the station clear of the sink and the reach stopped needing help.
    # `dock_for_target` stays in `oracle_common` — the burner oracle uses it.

    # -- STAGE 2: grasp the target ---------------------------------------------------
    say(env, "grasp the target")
    mesh = target.get_first_collision_mesh(to_world_frame=True)
    if mesh is None:
        return fail(env, "grasp the target: no collision mesh")
    obb = mesh.bounding_box_oriented
    tallest_top = max(
        float(a.get_first_collision_mesh(to_world_frame=True).bounds[1][2])
        for a in (task.shaker, task.condiment_bottle)
    )

    # The horizontal base->object ray: the line the arm actually extends along, which
    # is the approach `approach_aligned_grasp_info` aims down (K59). Taken from the base
    # and not the TCP so it does not drift with whatever pose the hand happens to be in.
    if GRASP_TORSO_RESERVE > 0.0:
        # EXPERIMENT (2026-09-06, off by default): leave the lift some torso. At the shipped
        # standoff the grasp spends the torso to its stop (377, 1802: 0.368 of 0.386) and the
        # lift then has neither a screw rung nor torso room, only an RRT swing that drops the
        # condiment. Set the torso GRASP_TORSO_RESERVE under its stop before the approach and
        # freeze it for the reach and grasp legs; the arm reaches on its own.
        jm = getattr(task.agent.robot, "active_joints_map", None)
        if jm is not None and "torso_lift_joint" in jm:
            t_idx = int(jm["torso_lift_joint"].active_index[0])
            t_now = float(_np(task.agent.robot.get_qpos()).reshape(-1)[t_idx])
            t_max = float(_np(jm["torso_lift_joint"].limits).reshape(-1)[-1])
            t_goal = min(t_now, t_max - GRASP_TORSO_RESERVE)
            if t_now - t_goal > 0.01:
                say(env, "torso reserve for the lift", torso=round(t_now, 3), to=round(t_goal, 3))
                r_t = common.plan_joints(env, planner, task, {"torso_lift_joint": t_goal},
                                         label="torso reserve", tries=1, who=WHO, line_only=True)
                if r_t != -1 and common.stopped_by_horizon(planner):
                    return r_t
                planner.planner.update_from_simulation()
    base_pos = _np(task.agent.base_link.pose.p)[0]
    ee_direction = np.asarray(obb.center_mass, dtype=np.float64) - base_pos
    ee_direction[2] = 0.0
    ee_direction = ee_direction / np.linalg.norm(ee_direction)
    target_closing = _np(task.agent.tcp.pose.to_transformation_matrix())[0, :3, 1]

    z_grasp = max(float(obb.center_mass[2]), float(mesh.bounds[1][2]) - GRASP_BELOW_TOP)
    grasp, reach = raise_grasp_to(*grasp_geometry(task, obb, ee_direction, target_closing, grasp_info), z_grasp)
    stove_backed = False
    stove_hits: list = []           # refusal texts naming the stove, by attempt
    attempt_res: list = []          # the first closings' results (-1 = refused, nothing executed)
    # The target's approach pad for this stage: APPROACH_TARGET_PAD for the first closings
    # (a wide pad keeps a fingertip off a 16 g shaker — g61: 1679, 1957), the shipped
    # GRASP_KEEPOUT_PAD once those are refused and for every rung after (1664: the wide pad
    # refused the one approach that grasps). Refused plans cost no steps.
    pad_now = [APPROACH_TARGET_PAD]
    # The joint-line approach for the FIRST closings only (HP7, 1652 @0.15, 2026-09-06: a
    # stove-side target had the line above the standoff execute 92 knots on rung after
    # rung, each descent refused at the stove, and the episode ran out of horizon at the
    # hover). The line's worth is the episode's opening; the rungs keep the old approach.
    line_now = [APPROACH_BY_LINE]

    def try_grasp(env_, planner_, task_, obj_, grasp_, reach_):     # the stage's pad and line
        return _try_grasp_with_pad(env_, planner_, task_, obj_, grasp_, reach_,
                                   target_pad=pad_now[0], by_line=line_now[0])

    def stove_backdock(res_now, why: str):
        """Back the base straight off STOVE_BACKDOCK_M and retry both closings; once.

        1916 @0.15 (2026-09-06): the shaker stood against the stove and every level grasp
        leg was refused `forearm_roll_link <-> stove`; at a dock 0.15 m further back the
        same seed grasped and poured (g62a). A straight reverse is one motion for the
        differential base (no turning); the aim is rebuilt from the live object with the
        base where it now stands. Returns `(res, grasped, ran)`; `ran` False when the
        rescue did not apply (already used, no `drive_straight`, or off).
        """
        nonlocal ee_direction, grasp, reach, stove_backed
        drive = getattr(planner, "drive_straight", None)
        if stove_backed or not callable(drive) or STOVE_BACKDOCK_M <= 0.0:
            return res_now, False, False
        stove_backed = True
        say(env, "the stove blocks the reach; backing the base off", m=STOVE_BACKDOCK_M,
            refusal=str(why)[:70])
        if res_now != -1:
            r_back = back_off_from_the_object(env, planner, task, why="re-docking further back")
            if r_back != -1 and common.stopped_by_horizon(planner):
                return r_back, False, True
        planner.open_gripper()
        r_drv = drive(-STOVE_BACKDOCK_M)
        if r_drv != -1 and common.stopped_by_horizon(planner):
            return r_drv, False, True
        planner.planner.update_from_simulation()
        base_now = _np(task.agent.base_link.pose.p)[0]
        live = target.get_first_collision_mesh(to_world_frame=True)
        obb_live = live.bounding_box_oriented if live is not None else obb
        d = np.asarray(obb_live.center_mass, dtype=np.float64) - np.asarray(base_now, dtype=np.float64)
        d[2] = 0.0
        ee_direction = d / max(float(np.linalg.norm(d)), 1e-9)
        z_live = (max(float(obb_live.center_mass[2]), float(live.bounds[1][2]) - GRASP_BELOW_TOP)
                  if live is not None else z_grasp)
        r_last, g_last = -1, False
        for sign, name in ((1.0, "as seeded"), (-1.0, "flipped")):
            g_try, r_try = raise_grasp_to(
                *grasp_geometry(task, obb_live, ee_direction, sign * target_closing, grasp_info), z_live)
            say(env, "grasp after backing off", closing=name,
                grasp=[round(float(v), 3) for v in g_try.p])
            planner.open_gripper()
            planner.planner.update_from_simulation()
            r_last, g_last = try_grasp(env, planner, task, target, g_try, r_try)
            if r_last != -1 and common.stopped_by_horizon(planner):
                return r_last, g_last, True
            if r_last != -1 and g_last:
                grasp, reach = g_try, r_try
                break
            if r_last != -1:
                rb = back_off_from_the_object(env, planner, task, why="the last close missed")
                if rb != -1 and common.stopped_by_horizon(planner):
                    return rb, False, True
        return r_last, g_last, True
    say(env, "grasp geometry", extents=[round(float(v), 3) for v in obb.primitive.extents],
        top=round(float(mesh.bounds[1][2]), 3), grasp=[round(float(v), 3) for v in grasp.p],
        approach_deg=round(math.degrees(math.atan2(float(ee_direction[1]), float(ee_direction[0]))), 1))

    def reaim_from_live(closing_sign: float):
        """The yaw ladder's live re-read (K63), one block earlier. None if unreadable."""
        live = target.get_first_collision_mesh(to_world_frame=True)
        if live is None:
            return None
        obb_now = live.bounding_box_oriented
        moved = float(np.linalg.norm(
            np.asarray(obb_now.center_mass, dtype=np.float64)[:2]
            - np.asarray(obb.center_mass, dtype=np.float64)[:2]))
        if moved > GRASP_REREAD_REPORT_M:
            say(env, "the object moved during an earlier attempt; re-aiming the retry",
                moved_cm=round(moved * 100, 1))
        z_now = max(float(obb_now.center_mass[2]),
                    float(live.bounds[1][2]) - GRASP_BELOW_TOP)
        return raise_grasp_to(
            *grasp_geometry(task, obb_now, ee_direction, closing_sign * target_closing, grasp_info),
            z_now)

    with common.capture_refusal("stove") as stove_cap:
        res, grasped = try_grasp(env, planner, task, target, grasp, reach)
    stove_hits.append(stove_cap.refusal)
    attempt_res.append(res)
    if res != -1 and common.stopped_by_horizon(planner):
        say(env, "stopped by the horizon during the grasp")
        return res
    if res == -1:
        # The reach from the rest posture is an RRT plan (the screw hits the
        # shoulder_lift limit) and RRTConnect is randomized: on seed 12 it returned
        # `Approximate solution` twice at the same reachable pose (IK found it). One
        # more draw of the same grasp before flipping the closing.
        say(env, "grasp retry, same closing (a refused plan; RRT is randomized)")
        planner.planner.update_from_simulation()
        if REREAD_BEFORE_RETRY:
            aimed = reaim_from_live(1.0)
            if aimed is not None:
                grasp, reach = aimed
        with common.capture_refusal("stove") as stove_cap:
            res, grasped = try_grasp(env, planner, task, target, grasp, reach)
        stove_hits.append(stove_cap.refusal)
        attempt_res.append(res)
        if res != -1 and common.stopped_by_horizon(planner):
            say(env, "stopped by the horizon during the grasp retry")
            return res
    if res == -1 or not grasped:
        say(env, "grasp retry with flipped closing", plan_failed=bool(res == -1), grasped=grasped)
        planner.open_gripper()
        planner.planner.update_from_simulation()
        aimed = reaim_from_live(-1.0) if REREAD_BEFORE_RETRY else None
        grasp, reach = aimed if aimed is not None else raise_grasp_to(
            *grasp_geometry(task, obb, ee_direction, -target_closing, grasp_info), z_grasp)
        with common.capture_refusal("stove") as stove_cap:
            res, grasped = try_grasp(env, planner, task, target, grasp, reach)
        stove_hits.append(stove_cap.refusal)
        attempt_res.append(res)
        if res != -1 and common.stopped_by_horizon(planner):
            say(env, "stopped by the horizon during the grasp retry")
            return res
        if res == -1 or not grasped:
            # Both closings are spent. Turn the wrist and ask again rather than give the
            # episode up: seeds 7 and 8 refuse every approach direction at yaw 0 with the
            # same `IK Failed`, and the object does not care which way the fingers lie
            # across it.
            #
            # `not grasped` is here since K62. Until then the ladder was entered only on
            # a refused *plan*, so an episode whose plan succeeded and whose **grip**
            # failed got exactly two attempts and stopped — `fingers closed but
            # agent.is_grasping(target) is False after both closing directions`. That is
            # two of the five remaining held-out failures (seeds 45 and 51), and a failed
            # close on a body of revolution is exactly the case a different yaw or a
            # different grip height can answer. The budget (`GRASP_LADDER_STEP_BUDGET`)
            # still bounds it, and an episode that reaches here was failing anyway.
            climbed = False
            if pad_now[0] > GRASP_KEEPOUT_PAD and all(r_ == -1 for r_ in attempt_res):
                # Every closing was REFUSED under the wide pad (nothing executed): the pad,
                # not the scene, is what refused. Ask both closings once more under the
                # shipped pad before the rungs (1664 @0.15: its grasp is the flipped
                # closing at 0.03; at 0.05 the ladder burned its budget on IK refusals).
                pad_now[0] = GRASP_KEEPOUT_PAD
                say(env, "every closing refused under the wide approach pad; the shipped pad",
                    pad=GRASP_KEEPOUT_PAD)
                for sign, name in ((1.0, "as seeded"), (-1.0, "flipped")):
                    aimed = reaim_from_live(sign) if REREAD_BEFORE_RETRY else None
                    g_try, r_try = aimed if aimed is not None else raise_grasp_to(
                        *grasp_geometry(task, obb, ee_direction, sign * target_closing, grasp_info), z_grasp)
                    say(env, "grasp under the shipped pad", closing=name)
                    planner.open_gripper()
                    planner.planner.update_from_simulation()
                    with common.capture_refusal("stove") as stove_cap:
                        res, grasped = try_grasp(env, planner, task, target, g_try, r_try)
                    stove_hits.append(stove_cap.refusal)
                    if res != -1 and common.stopped_by_horizon(planner):
                        return res
                    if res != -1 and grasped:
                        grasp, reach = g_try, r_try
                        climbed = True
                        say(env, "grasped under the shipped pad", closing=name)
                        break
                    if res != -1:
                        rb = back_off_from_the_object(env, planner, task, why="the last close missed")
                        if rb != -1 and common.stopped_by_horizon(planner):
                            return rb
            pad_now[0] = GRASP_KEEPOUT_PAD          # the rungs plan under the shipped pad
            line_now[0] = False                     # and approach as before the line
            n_stove = sum(1 for h in stove_hits if h)
            stove_why = (next((h for h in stove_hits if h), None)
                         if (not climbed and n_stove >= STOVE_BACKDOCK_MIN_HITS) else None)
            if stove_why is not None:
                # The stove named in a refusal: back the base off before the rungs spend
                # the budget on reaches the stove refuses one by one (1916: 232 steps).
                res, grasped, ran = stove_backdock(res, stove_why)
                if ran and res != -1 and common.stopped_by_horizon(planner):
                    return res
                if ran and res != -1 and grasped:
                    say(env, "grasped after backing the base off the stove")
                    climbed = True
            spent_at_start = int(planner.elapsed_steps)
            out_of_budget = False
            ik_fails = 0            # consecutive hard IK refusals at one grip height (K79b)
            ik_exhausted = False
            hand_at_object = res != -1      # an executed close that missed leaves it there
            for dz in (GRASP_LIFT_RUNGS if not climbed else ()):
                # K63: re-read the object before building this rung's candidates. The
                # attempts that got us here **touched** it: measured on held-out seeds
                # 45 and 51, the object had already been pushed 10.4 cm and 17.3 cm from
                # where the original OBB put it, and every one of the twelve rungs was
                # reaching for the place it used to be. That is why those seeds exhaust
                # the ladder rather than recover on some other yaw. Re-reading costs no
                # episode steps — it is a mesh read, not a motion.
                live = target.get_first_collision_mesh(to_world_frame=True)
                obb_now = live.bounding_box_oriented if live is not None else obb
                centre_xy = np.asarray(obb_now.center_mass, dtype=np.float64)[:2]
                moved = float(np.linalg.norm(
                    centre_xy - np.asarray(obb.center_mass, dtype=np.float64)[:2]))
                if moved > GRASP_REREAD_REPORT_M:
                    say(env, "the object has moved since the first grasp; re-aiming",
                        moved_cm=round(moved * 100, 1),
                        centre=[round(float(v), 3) for v in centre_xy])
                # The grip height is re-derived from the live mesh too: a knocked object
                # may also have tipped, and `z_grasp` was measured off the old bounds.
                z_now = max(float(obb_now.center_mass[2]),
                            float(live.bounds[1][2]) - GRASP_BELOW_TOP) if live is not None else z_grasp
                # The raised rung exists to lift the wrist out of the counter (K79c), but
                # on a 9.4 cm shaker `top - 0.03 + 0.025` is a grip on the cap's last 5 mm:
                # 1916 @0.15 (2026-09-06) "grasped" there and the shaker slid out of the
                # pads on a straight 31-knot screw lift. Never raise the grip within
                # GRIP_MIN_DEPTH of the live top; a rung that would is clamped to it.
                z_rung = z_now + dz
                if GRIP_MIN_DEPTH > 0.0 and live is not None:
                    z_cap = float(live.bounds[1][2]) - GRIP_MIN_DEPTH
                    if z_rung > z_cap:
                        say(env, "rung clamped to the grip depth", asked=round(float(z_rung), 3),
                            clamped=round(float(z_cap), 3), dz=round(float(dz), 3))
                        z_rung = z_cap
                base_grasp, base_reach = raise_grasp_to(
                    *grasp_geometry(task, obb_now, ee_direction, target_closing, grasp_info),
                    z_rung,
                )
                cands = grasp_yaw_candidates(base_grasp, base_reach, centre_xy)
                # K79c: the IK counter resets per grip height. As first written it was
                # global, and on seed 61 it abandoned the ladder at rung 3 of 12 with
                # `out_of_budget=False` — skipping yaw -60, yaw +90 **and the whole
                # dz = +0.025 rung, which exists precisely to lift the wrist out of the
                # counter that every one of those refusals named**. A hard IK failure says
                # this pose family has no solution *at this height*; it says nothing about
                # the next one.
                ik_fails = 0
                # yaw 0 at the first rung is the grasp both closings already refused.
                for yaw_deg, g_try, r_try in (cands[1:] if dz == 0.0 else cands):
                    if int(planner.elapsed_steps) - spent_at_start > GRASP_LADDER_STEP_BUDGET:
                        say(env, "grasp ladder out of step budget",
                            spent=int(planner.elapsed_steps) - spent_at_start)
                        out_of_budget = True
                        break
                    say(env, "grasp retry, turned and raised", yaw_deg=yaw_deg,
                        dz=round(float(dz), 3), grasp=[round(float(v), 3) for v in g_try.p])
                    if RUNG_BACK_OFF and hand_at_object:
                        r_back = back_off_from_the_object(env, planner, task, why="the last close missed")
                        if r_back != -1 and common.stopped_by_horizon(planner):
                            return r_back
                        hand_at_object = False
                    planner.open_gripper()
                    planner.planner.update_from_simulation()
                    with common.capture_refusal("stove") as stove_cap, \
                            common.capture_refusal(IK_REFUSAL) as cap:
                        res, grasped = try_grasp(env, planner, task, target, g_try, r_try)
                    hand_at_object = res != -1
                    stove_hits.append(stove_cap.refusal)
                    if ((res == -1 or not grasped) and stove_cap.refusal is not None and not stove_backed
                            and sum(1 for h in stove_hits if h) >= STOVE_BACKDOCK_MIN_HITS):
                        res, grasped, ran = stove_backdock(res, stove_cap.refusal)
                        if ran and res != -1 and common.stopped_by_horizon(planner):
                            return res
                        hand_at_object = ran and res != -1
                        if ran and res != -1 and grasped:
                            say(env, "grasped after backing the base off the stove")
                            climbed = True
                            break
                    if res == -1 and cap.refusal is not None:
                        ik_fails += 1
                        if ik_fails >= GRASP_IK_GIVE_UP:
                            say(env, "no IK at this grip height; trying the next rung",
                                ik_fails=ik_fails, yaw_deg=yaw_deg, dz=round(float(dz), 3))
                            ik_exhausted = True
                            break   # this height's yaws only — the dz loop continues
                    else:
                        ik_fails = 0
                    if res != -1 and common.stopped_by_horizon(planner):
                        say(env, "stopped by the horizon during the grasp ladder")
                        return res
                    if res != -1 and grasped:
                        grasp, reach = g_try, r_try
                        say(env, "grasped after turning the wrist", yaw_deg=yaw_deg,
                            dz=round(float(dz), 3))
                        climbed = True
                        break
                if climbed or out_of_budget:
                    break
            if not climbed:
                # Last resort, and the only stage here that moves the base and the arm
                # in one plan: `move_base_x_and_manipulation` frees root_x and every arm
                # joint at once, so the base slides while the hand reaches instead of
                # parking first and reaching after. Its mask is world-x, which the
                # inherited comment calls "the robot's lateral axis" at yaw pi/2 — true,
                # and on kitchen 102 the counter runs along world x, so lateral *is* the
                # useful direction. Worth one plan: every refusal above was found with
                # the base pinned, and freeing it is a degree of freedom the whole ladder
                # never had.
                # Before the arc: come at the object from above instead of level with
                # it. Every refusal so far is the hand working in the plane of the
                # counter — `wrist_flex_link <-> counter_main` — and a vertical approach
                # puts the wrist above the worktop rather than through it. Tried as a
                # global setting earlier and it was no better across the eval set, which
                # is why it is a rung and not the default.
                # Running this rescue EARLY — at the top of the ladder, while the object is
                # still standing rather than ninth after it is flat — was built and withdrawn
                # (K79k). The gap was real: on seed 71 the three top-down rungs aim at z 0.942,
                # 2.2 cm over the worktop, because the object is already toppled by then, and
                # top-down at its *standing* pose (z 0.984, 6.4 cm clearance) had never been
                # tried. It fires (the trace shows `from above (early)`) and **both persistent
                # failures still fail**, so no sweep was spent on it: a rescue that does not
                # convert the two seeds it was built for has no mechanism left to help a
                # population. **And tried again, conditionally,
                # in K79j** — top-down first only when the grip sits under 9 cm over the
                # worktop, which selects the shaker (6.45 cm) and not the bottle (12.67 cm).
                # The motivation was sound: both persistent failures target the shaker, 0 of
                # 12 bottle-target episodes fail, and five `wrist_flex_link<->counter_main`
                # refusals say the level approach drags the wrist through the counter. Seed
                # 71 — which had resisted every other intervention that day — passed with it.
                # The population did not: dev 78->67/80, held 95->89/100, **173 -> 156/180**,
                # 24 failures against 7. One seed passing is not evidence. This stays a rung,
                # and the sentence above was already warning about it.
                # Deliberately the *lowest* height, not the highest: the +2.5 cm rung
                # exists to lift the wrist out of the counter, and a hand coming straight
                # down is already clear of it. Higher here only moves the fingers toward
                # the rim, and seed 16 showed what that costs — all three top-down poses
                # were reached to within 5 mm (`reached=True`) and the gripper still
                # closed on nothing. Grip the body, not the lip.
                # K76: read the object HERE, not from the `obb`/`mesh` captured before the
                # first attempt. K63 put the live re-read at the top of each `dz` rung and
                # this block was missed, so the top-down rescue — the last thing tried
                # before the arc — aimed wherever the object used to be. Three independent
                # per-seed post-mortems measured the same signature: seed 66 planned 10.3 cm
                # from the bottle, seed 54's three rungs missed by 10.1-12.7 cm, and both
                # reported `reached=True` while closing on bare counter. That is also the
                # real mechanism behind the seed-16 note below, which blamed the grip height.
                live_top = target.get_first_collision_mesh(to_world_frame=True)
                obb_top = live_top.bounding_box_oriented if live_top is not None else obb
                mesh_top = live_top if live_top is not None else mesh
                centre_top = np.asarray(obb_top.center_mass, dtype=np.float64)[:2]
                # K81: the note above is right, and the rung it defends has still never
                # worked — 0 rescues in 60 attempts over three 180-seed sweeps, against
                # 231 grasps the side ladder wins. Probed at the recorded toppled poses of
                # seeds 129/136/177 with this same geometry: **0 of 12 poses have IK at
                # every height from 2 to 16 cm over the worktop**, so there is no height
                # to move it to and nothing here to tune. `MIKASA_SKIP_TOPDOWN=1` drops it.
                top_grasp, top_reach = raise_grasp_to(
                    *grasp_geometry(task, obb_top, np.array([0.0, 0.0, -1.0]), target_closing, grasp_info),
                    max(float(obb_top.center_mass[2]), float(mesh_top.bounds[1][2]) - 2 * FINGER_LENGTH),
                )
                yaws = () if SKIP_TOPDOWN_RESCUE else GRASP_YAWS_DEG[:3]
                if SKIP_TOPDOWN_RESCUE:
                    say(env, "skipping the from-above rescue (0 of 60 historic successes)")
                # The flipped closing direction was tried here too and withdrawn (K55):
                # `target_closing` is seeded from the gripper's level-approach axis, so a
                # vertical approach wanting it turned was the obvious next suspect after
                # the grip height. All five top-down attempts on seed 16 — three at the
                # seeded closing, two flipped — reached and closed on nothing. Whatever
                # that grip misses, it is not the closing axis.
                for yaw_deg, g_try, r_try in grasp_yaw_candidates(
                        top_grasp, top_reach, centre_top, angles_deg=yaws):
                    say(env, "grasp retry, from above", yaw_deg=yaw_deg,
                        grasp=[round(float(v), 3) for v in g_try.p])
                    if RUNG_BACK_OFF and hand_at_object:
                        r_back = back_off_from_the_object(env, planner, task, why="the last close missed")
                        if r_back != -1 and common.stopped_by_horizon(planner):
                            return r_back
                        hand_at_object = False
                    planner.open_gripper()
                    planner.planner.update_from_simulation()
                    res, grasped = try_grasp(env, planner, task, target, g_try, r_try)
                    hand_at_object = res != -1
                    if res != -1 and common.stopped_by_horizon(planner):
                        return res
                    if res != -1 and grasped:
                        grasp, reach = g_try, r_try
                        climbed = True
                        say(env, "grasped from above", yaw_deg=yaw_deg)
                        break
            if not climbed:
                say(env, "grasp ladder exhausted; one plan for base and arm together",
                    out_of_budget=out_of_budget)
                # The ladder leaves the hand wherever its last attempt stopped, which is
                # against the object: the arc's first refusal was
                # `l_gripper_finger_link <-> shaker` on the *start* state, and mplib will
                # not plan out of a start it considers in collision. Open and back off
                # first — MIKASA-Robo-VLA's oracle does the same thing before its retry.
                planner.open_gripper()
                # K79: the retreat's result is *checked*. `lift_hand` plans with
                # `plan_screw` alone — no RRT fallback — so a saturated torso is an
                # outright -1 that executes zero steps. Measured on a failing seed-80
                # capture: `joint limit at index [3] after 1 step(s), 0.000 of the twist
                # left`, the hand never backed off, and the 26-step sweep below then drove
                # through the distractor at counter height and moved it 0.1227 m — 23%
                # past `distractor_move_tol`. It cost nothing there only because that
                # episode was already -1; on an episode whose grasp succeeds it voids the
                # run on `distractor_ok`.
                retreat = planner.lift_hand(delta_h=ARC_RETREAT_M)
                if retreat == -1:
                    # Up is blocked by the torso, which says nothing about the arm's
                    # reach: back off along the hand's own approach axis instead, through
                    # `static_manipulation`, which does have the RRT fallback `lift_hand`
                    # lacks.
                    tcp_now = task.agent.tcp.pose[0].sp
                    back = sapien.Pose(p=tcp_now.p, q=tcp_now.q) * sapien.Pose([0, 0, -ARC_RETREAT_M])
                    say(env, "arc retreat refused upward; backing off along the approach")
                    retreat = planner.static_manipulation(back, disable_lift_joint=False)
                if retreat == -1:
                    # Both retreats refused. Do NOT skip the arc: that was tried and
                    # measured worse (dev 77->76, held 96->95, and seed 80 came back).
                    # The arc from a colliding start is usually refused too, but not
                    # always, and refusing it outright converts a chance into a certain
                    # failure. Widen the keep-out for that leg instead — the hand is
                    # against the object and this leg's tracking error is 7-9 cm against
                    # a 3 cm pad — so the attempt survives and the damage does not.
                    say(env, "arc retreat refused both ways; widening the keep-out")
                    arc_pad = ARC_KEEPOUT_PAD
                else:
                    arc_pad = GRASP_KEEPOUT_PAD
                if common.stopped_by_horizon(planner):
                    say(env, "stopped by the horizon during the arc retreat")
                    return retreat
                planner.planner.update_from_simulation()
                # K77: the arc frees root_x *and* every arm joint at once, so it sweeps
                # the longest path across the counter — and it was the last unguarded
                # caller. On seed 71 it swung the gripper body and wrist through the
                # distractor at counter height, moving it 0.131 m (31% past tolerance)
                # while the episode was booked as a plain `no plan`.
                #
                # Its poses were stale too: `grasp`/`reach` are only reassigned on a
                # *successful* rung, so after a ladder that changed nothing they still
                # aim where the object was before the first attempt — 0.228 m away on
                # this seed. K76 fixed exactly this for the top-down rung and stopped one
                # block short.
                live_arc = target.get_first_collision_mesh(to_world_frame=True)
                if live_arc is not None:
                    obb_arc = live_arc.bounding_box_oriented
                    grasp, reach = raise_grasp_to(
                        *grasp_geometry(task, obb_arc, ee_direction, target_closing, grasp_info),
                        max(float(obb_arc.center_mass[2]),
                            float(live_arc.bounds[1][2]) - GRASP_BELOW_TOP),
                    )
                with common.keepout(planner, [task.shaker, task.condiment_bottle],
                                    pad=arc_pad):
                    arc = planner.move_base_x_and_manipulation(reach)
                if arc != -1 and common.stopped_by_horizon(planner):
                    return arc
                if arc != -1:
                    planner.planner.update_from_simulation()
                    with common.keepout(planner, [task.shaker, task.condiment_bottle],
                                        pad=arc_pad):
                        arc = planner.static_manipulation(grasp, disable_lift_joint=False)
                    if arc != -1 and common.stopped_by_horizon(planner):
                        return arc
                    if arc != -1:
                        arc = planner.close_gripper()
                        grasped = bool(_np(task.agent.is_grasping(target)).any())
                        if arc != -1 and grasped:
                            res = arc
                            climbed = True
                            say(env, "grasped after moving the base and the arm together")
            # A re-dock rung — drive the base to the target's own station after the arc
            # fails, then retry the grasp — was built and withdrawn (K79b). K55 withdrew
            # the same idea for two reasons and BOTH are now measurably gone: `drive_base`
            # refused because it planned a pure base translation with fifteen joints
            # (`freeze_arm=True`, K74, fixes exactly that), and turn-drive-turn "spent what
            # was left of the horizon" when episodes used ~382 of 1100 — these end at 387
            # with 713 unused. So it was rebuilt on a genuinely changed premise, and it
            # still does not pay: it fired on **all five** remaining grasp failures and
            # converted **none**. The drive itself is not the problem — `rotate_base_z`
            # reports `achieved=-1.5731 residual=-0.0001 jammed=False` out and back, so the
            # base goes where it is sent, and `out_of_budget=False` throughout. The grasp
            # is refused from the new stance too. Whatever these seeds need, it is not a
            # base pose the arm can be driven to. Cost if kept: two ~56-knot rotations on
            # a path that already failed.
            if not climbed:
                why = ("ran out of step budget" if out_of_budget
                       else "every wrist yaw and grip height refused, the arc failed, "
                            "and so did the re-dock")
                return fail(env, f"grasp the target (retry): {why}", out_of_budget=out_of_budget)
        if not grasped:
            return fail(env, "grasp the target: fingers closed but agent.is_grasping(target) is False "
                             "after both closing directions")
    # D6: a nudged distractor voids the episode — a physical miss, not a plan failure.
    if not _flag(res[-1], "distractor_ok"):
        say(env, "missed: distractor moved during the grasp", distractor_ok=False,
            distractor_moved=round(float(_np(res[-1]["distractor_moved"]).reshape(-1)[0]), 3))
        return res
    say(env, "grasped", distractor_ok=True)
    planner.planner.update_from_simulation()
    hold_object_in_planner(env, planner, task, target, held=True)

    # The object the episode is voided for touching. Named once, guarded at every plan
    # that runs after the grasp (K77) — but never the target, which is held.
    distractor = task.condiment_bottle if target is task.shaker else task.shaker

    # -- STAGE 3: lift, torso frozen, over the neighbour (K40) --------------------------
    # The held object HANGS below the TCP — grasped GRASP_BELOW_TOP under its top, its
    # bottom is (height - GRASP_BELOW_TOP) lower: 6 cm on the 9 cm shaker, 13 cm on the 16 cm
    # bottle. K40's margin over the neighbour was taken from the TCP, so the object's bottom
    # sat AT the neighbour's top through the base's 180-degree turn to the bowl dock and
    # swept it (1847 @0.15, 2026-09-06: distractor moved 0.18 m during the drive). Measure
    # the hang from the object's own mesh and lift the BOTTOM over the neighbour.
    hang = 0.0
    if LIFT_HANG_AWARE:
        hang = max(0.0, float(grasp.p[2]) - float(mesh.bounds[0][2]))
    lift_z = max(float(grasp.p[2]) + LIFT_ABOVE_GRASP, tallest_top + LIFT_OVER_NEIGHBOUR,
                 tallest_top + hang + LIFT_BOTTOM_CLEAR)
    say(env, "lift", lift_z=round(lift_z, 3), tallest_top=round(tallest_top, 3), hang=round(hang, 3))
    # Planned with the same keep-out as the approach (K77). It was missing here, and the
    # lift is where an unguarded RRT does the most damage: the payload hangs up to
    # 12.8 cm below the TCP, so a lateral excursion drags a lever of glass across the
    # counter. Measured on seed 63 — the *guarded* approach RRT left the distractor at
    # 0.0001 m, the *unguarded* lift RRT moved it 0.158 m against a 0.10 m tolerance,
    # after flying a 1.5 m loop for a 15 cm vertical move. The object is already attached
    # in the planning world here, so mplib checks the payload, not merely the links.
    # Prefer a height the screw can actually reach (K77). The lift is a pure vertical
    # translation, and handing it to RRTConnect is where the worst damage in this oracle
    # happens: measured, the screw was discarded with 0.026 of the twist left (5 mm of a
    # 193 mm lift) and the fallback drove the TCP to 0.21 m *below* the counter, raking
    # the object out of the fingers (seed 30); on another it flew a 1.78 m path for a
    # 0.19 m move and knocked the distractor onto the floor (seed 8). Shaving a couple of
    # centimetres off the target is free — the height only has to clear the neighbour —
    # so probe downwards and take the first height that plans straight. A probe executes
    # nothing, so a rejected rung costs no episode steps.
    lift_floor = max(float(grasp.p[2]) + 0.02, tallest_top + LIFT_OVER_NEIGHBOUR * 0.5,
                     tallest_top + hang + 0.02)
    lift_rungs = [lift_z] + [z for z in (lift_z - 0.02, lift_z - 0.04, lift_z - 0.06)
                             if z >= lift_floor]
    lift_target = sapien.Pose(np.array([grasp.p[0], grasp.p[1], lift_z]), grasp.q)
    lift_freeze = True
    found = False
    # Two passes: torso frozen first, then torso free. K40 freezes it so the lift cannot
    # spend the torso, and that stands as the default — but the reason is keeping the
    # object clear of its neighbour during the *base turn*, not the lift itself, and the
    # torso is exactly the joint that makes vertical motion cheap. Measured: with it
    # frozen the refusal moves from `forearm_roll` to `joint limit at index [7]`
    # (`shoulder_lift`) at every height offered — the arm alone cannot span the rise from
    # those configurations, so the choice is a torso-driven lift or an RRT that rakes the
    # counter. A probe executes nothing, so the whole search is free in episode steps.
    for freeze in (True, False):
        for z_try in lift_rungs:
            cand = sapien.Pose(np.array([grasp.p[0], grasp.p[1], z_try]), grasp.q)
            if common.screw_plans(planner, cand, disable_lift_joint=freeze):
                lift_target, lift_freeze, found = cand, freeze, True
                if z_try != lift_z or not freeze:
                    say(env, "lift takes a screw-reachable rung",
                        asked=round(float(lift_z), 3), taking=round(float(z_try), 3),
                        torso_frozen=freeze)
                break
        if found:
            break
    if not found:
        # K79q: the exhausted probe was silent — its only trace was the *absence* of the
        # line above, and it is the precise predictor of the RRT dive that follows (the
        # analyst on seed 171: eight rungs tried, none screw-reachable, no event emitted).
        # Refusing the RRT lift here was built and withdrawn (K79q). The mechanism is real
        # — on seed 171 RRTConnect answered a *pure vertical* lift with a 142-knot path
        # that descends 13.6 cm, puts the gripper ~7 cm below the counter surface and
        # back-drives the frozen torso to a 72 N tracking error, raking the object out of
        # the fingers. But it is the exception: this probe exhausts on **10 of 180**
        # episodes and the RRT lift usually works, so refusing it measured **166/180
        # against 173/180**, seven seeds worse, with nine new `FAILED: lift`. Same shape as
        # the arc-refusal mistake — removing a path that mostly works costs more than the
        # harm it prevents. The diagnostic below stays, because the silence was the only
        # predictor of the dive.
        say(env, "no screw-reachable lift rung; the lift falls to RRT",
            asked=round(float(lift_z), 3), rungs=len(lift_rungs) * 2,
            floor=round(float(lift_floor), 3))
    # **Only the distractor.** The target is attached to the gripper by now, and padding
    # it too gives the held object a free-standing proxy to collide with — its own. That
    # mistake cost 34 of 80 seeds a `FAILED: lift` in one measured sweep (39/80 against a
    # 75/80 baseline); the approach in `try_grasp` pads both only because nothing is held
    # there yet.
    res = -1
    if not found and LIFT_BY_TORSO:
        # The torso is the one joint that moves the hand straight up with the arm as it
        # stands — no swing, no dive. Between the exhausted screw probe and the RRT
        # (which on 377 @0.15 answered a 0.19 m rise with a path that flung the
        # condiment to the floor — the owner's "rotates the arm a lot lifting"): raise
        # the torso by the rise, as a joint LINE, highest rung first; only what fits
        # under the torso's stop. Line only — a blocked line falls to the RRT as before.
        jm = getattr(task.agent.robot, "active_joints_map", None)
        if jm is None or "torso_lift_joint" not in jm:
            lift_rungs_torso: list = []     # a robot without the joint map: the RRT as before
            say(env, "no torso joint map; the RRT")
        else:
            lift_rungs_torso = list(lift_rungs)
            t_idx = int(jm["torso_lift_joint"].active_index[0])
            t_now = float(_np(task.agent.robot.get_qpos()).reshape(-1)[t_idx])
            t_max = float(_np(jm["torso_lift_joint"].limits).reshape(-1)[-1])
        for z_try in lift_rungs_torso:
            dz = float(z_try) - float(grasp.p[2])
            if dz < 0.05 or t_now + dz > t_max + 1e-6:
                continue
            with common.keepout(planner, [distractor], pad=GRASP_KEEPOUT_PAD):
                r = common.plan_joints(env, planner, task, {"torso_lift_joint": t_now + dz},
                                       label="lift by the torso", tries=1, who=WHO,
                                       line_only=True)
            if r != -1:
                say(env, "lift by the torso", rise=round(dz, 3), to_z=round(float(z_try), 3),
                    torso=round(t_now + dz, 3))
                res = r
                break
        if res == -1:
            say(env, "no torso line fits the lift; the RRT")
    if res == -1:
        with common.keepout(planner, [distractor], pad=GRASP_KEEPOUT_PAD):
            if not found and LIFT_MAX_KNOTS is not None:
                # No screw rung and no torso room (377 @0.15: torso 0.368 of 0.386 at
                # the grasp): the RRT lift, drawn up to LIFT_KNOT_DRAWS times and the
                # shortest taken, a draw under LIFT_MAX_KNOTS accepted at once — K59's
                # cap: every drop it measured rode a path of >= 173 knots, none under.
                say(env, "lift by RRT under a knot cap", max_knots=LIFT_MAX_KNOTS,
                    draws=LIFT_KNOT_DRAWS)
                res = common.arm_move(env, planner, lift_target, who=WHO, stage="lift",
                                      disable_lift_joint=lift_freeze, tries=1,
                                      max_knots=LIFT_MAX_KNOTS, knot_draws=LIFT_KNOT_DRAWS)
            else:
                res = planner.static_manipulation(lift_target, disable_lift_joint=lift_freeze)
    if res != -1 and common.stopped_by_horizon(planner):
        return res
    if res == -1:
        return fail(env, "lift")

    # Post-conditions the lift never had (K77). There are `distractor_ok` checkpoints
    # after the grasp and after the drive but none here, and `distractor_moved` is an
    # absolute distance from home rather than a per-stage delta — so a knock during the
    # lift was reported as "distractor moved during the drive", 131 steps after it
    # happened (seed 63; the drive moved it 0.0000 m). Every stage label in this oracle
    # is only ever "which checkpoint noticed first", which is exactly the D6 confusion
    # this ledger exists to prevent.
    #
    # The grasp check is the other half: on seed 30 the screw was discarded 5 mm short of
    # a 193 mm lift, the RRT fallback drove the TCP to z=0.710 — 0.21 m *below* the
    # counter — raked the shaker out of the fingers, and because nothing checked, the next
    # 270 steps planned against a phantom object still attached in the planning world.
    if not _flag(res[-1], "distractor_ok"):
        say(env, "missed: distractor moved during the lift", distractor_ok=False,
            distractor_moved=round(float(_np(res[-1]["distractor_moved"]).reshape(-1)[0]), 3))
        return res
    if not bool(_np(task.agent.is_grasping(target)).any()):
        say(env, "missed: dropped during the lift", lift_z=round(float(lift_z), 3),
            object_z=round(float(_np(target.pose.p)[0][2]), 3))
        return res
    planner.planner.update_from_simulation()
    # The object's orientation relative to the base, read before the drive turns it.
    obj_rel = common.object_pose_in_base(task, target)

    # There is no carry pose here (K57). The tuck was inherited from the burner oracle,
    # where `rotate_base_z` sweeps the turn's arc against the *attached* object (K51)
    # and a stretched-out arm makes the turn unplannable. This task's two docks sit on
    # the same counter run, so that turn is small: measured over 30 seeds per arm at
    # matched load, **no episode failed the drive without the tuck** and the tuck itself
    # never once refused — it was buying nothing and costing a quarter of the motion.
    # jezv's planners have no equivalent stage at all.

    # -- STAGE 4: drive to the bowl dock -----------------------------------------------------
    dock = _np(task._bowl_dock_np)[0].astype(np.float64)
    face = np.array([math.cos(dock[2]), math.sin(dock[2]), 0.0])
    dock_xyz = np.array([dock[0], dock[1], 0.0])
    say(env, "drive to bowl dock", dock=[round(float(v), 3) for v in dock_xyz])
    # `freeze_arm=True` plans the translation with the base's three joints and nothing
    # else (`BASE_ONLY_PLAN_MASK`). Without it `move_base_forward` asks fifteen joints to
    # produce a pure base translation and runs one of them into a limit: measured here on
    # the randomized layout (K74), 12 of 14 failures were `FAILED: drive to bowl dock`
    # with `joint limit at index [10]` — the forearm_roll, inside a plan whose only job
    # was to move the base. The mask's own docstring records the same disease on
    # water-plants seed 11. It was never needed while the drive was a fixed 0.39 m; the
    # drawn layout makes it up to 2.2 m.
    res = planner.drive_base(target_pos=dock_xyz, target_view_vec=face, freeze_arm=True)
    if res != -1 and common.stopped_by_horizon(planner):
        return res
    if res == -1:
        # The drive refused with the arm out. Pull the object in over the base and ask
        # again (K74). This is `carry_pose` earning its place back: K57 deleted it as a
        # stage because on a fixed 0.39 m drive it never once fired and cost a quarter of
        # the motion — true then, and still true for short drives, which is why this is a
        # recovery and not a stage. The drawn layout puts the dock as far as x = 0.33,
        # against `wall_left_room_0_31`, and the refusals name it: `rotation sweep hits
        # gripper_link<->wall` from `rotate_base_z`, `screw plan failed: collision
        # ...finger_link<->wall, shaker<->wall` from `move_base_forward`. An arm folded
        # over the base sweeps a far smaller volume past that wall.
        say(env, "drive refused with the arm out; tucking and retrying")
        tucked = common.carry_pose(env, planner, task, target, who=WHO,
                                   max_knots=CARRY_MAX_KNOTS, knot_draws=CARRY_KNOT_DRAWS)
        if tucked != -1 and common.stopped_by_horizon(planner):
            return tucked
        if tucked == -1:
            return fail(env, "drive to bowl dock")
        planner.planner.update_from_simulation()
        with common.capture_refusal("collision ") as cap:
            res = planner.drive_base(target_pos=dock_xyz, target_view_vec=face, freeze_arm=True)
        if res == -1 and DRIVE_ALLOW_HELD_TOUCH and cap.refusal is not None:
            # 2025 @0.15 (2026-09-06): the tuck (yaw 90) executed to 2 mm and the drive's
            # screw was then refused `shoulder_lift_link <-> condiment_bottle after 1
            # step` — the held bottle rests against the shoulder in the MODEL, and mplib
            # will not plan out of a start it considers in collision. The object is held;
            # that contact is harmless. Allow it for the attachment and drive once more.
            stem = "shaker" if target is task.shaker else "condiment_bottle"
            link = held_touch_link(cap.refusal, stem)
            if link is not None:
                say(env, "the tucked object touches the arm in the model; allowing that contact",
                    link=link, refusal=cap.refusal[:80])
                common.hold_object_in_planner(env, planner, task, target, held=True, who=WHO,
                                              extra_touch=(link,))
                planner.planner.update_from_simulation()
                res = planner.drive_base(target_pos=dock_xyz, target_view_vec=face, freeze_arm=True)
        if res != -1 and common.stopped_by_horizon(planner):
            return res
        if res == -1:
            return fail(env, "drive to bowl dock (after the tuck)")
    planner.planner.update_from_simulation()
    d_dock, dyaw = common.dock_error(task, (dock[0], dock[1], dock[2]))
    say(env, "parked at the dock", d_dock=round(d_dock, 3), dyaw_deg=round(dyaw, 1))
    if not _flag(res[-1], "distractor_ok"):
        say(env, "missed: distractor moved during the drive", distractor_ok=False,
            distractor_moved=round(float(_np(res[-1]["distractor_moved"]).reshape(-1)[0]), 3))
        return res

    # World-frame, read *after* the drive: the object as it was grasped relative to
    # the base — the base has turned underneath it — with the base parked; and the
    # grasp transform from the same still moment.
    obj_q = common.object_q_from_base(task, obj_rel)
    T_tcp_obj = (task.agent.tcp.pose[0].inv() * target.pose[0]).sp
    bowl_p = _np(task.bowl.pose.p)[0].astype(np.float64)
    say(env, "bowl at", when="after the drive", p=[round(float(v), 3) for v in bowl_p])

    # -- STAGE 5: hover over the bowl, through a waypoint back toward the base ----------------
    hovered = False
    obj_q_held = obj_q
    for extra, back, spin in HOVER_RUNGS:
        aim = bowl_p - back * face
        obj_q_try = obj_q if spin == 0.0 else np.asarray(
            (sapien.Pose(q=quat_about(np.array([0.0, 0.0, 1.0]), math.radians(float(spin))))
             * sapien.Pose(q=obj_q)).q
        )
        hover = common.pose_over(aim, HOVER_ABOVE + extra, obj_q_try) * T_tcp_obj.inv()
        if not hovered and (extra, back, spin) == HOVER_RUNGS[0]:
            # The waypoint exists to break one long RRT motion into two (seed 12 came
            # back `Approximate solution` / `IK Failed` without it). If the hover itself
            # plans by a **straight** screw there is nothing to break up, and the
            # waypoint is a detour the video shows as the arm swinging back before it
            # goes forward — so probe first and skip it when the direct move is straight
            # (K61). That measurement was taken from the tucked arm K57 deleted, which is
            # the other reason not to pay for it unconditionally.
            if common.screw_plans(planner, hover, disable_lift_joint=False):
                say(env, "hover plans straight; skipping the pre-hover waypoint")
            else:
                # A ladder of waypoints, each asked under the knot cap WITH refusal
                # (1507 @0.15, 2026-09-06: the RAISED waypoint's best draw was 291
                # knots; executed under a cap without refusal it swept the bowl off the
                # counter, on a seed the level waypoint had carried in 80 s). A refused
                # waypoint falls to the next; the last falls to the hover itself, which
                # plans from wherever the arm stands.
                ups = [PRE_HOVER_UP] + ([0.0] if PRE_HOVER_UP > 0.0 else [])
                res = -1
                for up in ups:
                    pre = sapien.Pose(p=np.asarray(hover.p) - PRE_HOVER_BACK * face
                                      + np.array([0.0, 0.0, up]), q=hover.q)
                    say(env, "pre-hover", pre=[round(float(v), 3) for v in pre.p],
                        up=round(float(up), 3))
                    # The bowl is guarded for this transit leg only: seed 62's pre-hover RRT
                    # travelled 0.73 m for a 0.35 m goal and flipped the bowl upside down,
                    # 0.177 m from home, which alone loses the episode (`over_bowl` is read
                    # against the *live* bowl). It must NOT be guarded for the hover and pour
                    # below, which aim at it deliberately.
                    with common.keepout(planner, [distractor, task.bowl], pad=GRASP_KEEPOUT_PAD):
                        res = common.arm_move(env, planner, pre, who=WHO, stage="pre-hover",
                                              disable_lift_joint=False,
                                              max_knots=HOVER_MAX_KNOTS, knot_draws=HOVER_KNOT_DRAWS,
                                              knot_refuse=PRE_HOVER_REFUSE)
                    if res != -1 and common.stopped_by_horizon(planner):
                        return res
                    if res != -1:
                        break
                    say(env, "pre-hover refused at this height", up=round(float(up), 3))
                if res == -1:
                    say(env, "pre-hover refused; asking for the hover directly")
                say(env, "bowl at", when="after the pre-hover",
                    p=[round(float(v), 3) for v in _np(task.bowl.pose.p)[0]])
        say(env, "hover over bowl", hover=[round(float(v), 3) for v in hover.p],
            extra=round(float(extra), 3), back=round(float(back), 3), spin_deg=float(spin))
        with common.keepout(planner, [distractor], pad=GRASP_KEEPOUT_PAD):
            res = common.arm_move(env, planner, hover, who=WHO, stage="hover over bowl",
                                  disable_lift_joint=False,
                                  max_knots=HOVER_MAX_KNOTS, knot_draws=HOVER_KNOT_DRAWS)
        if res != -1 and common.stopped_by_horizon(planner):
            return res
        if res != -1:
            hovered = True
            obj_q_held = obj_q_try
            break
        say(env, "hover refused at this rung", extra=round(float(extra), 3),
            back=round(float(back), 3), spin_deg=float(spin))
    if not hovered:
        return fail(env, "hover over bowl: every rung refused")
    hinfo = res[-1]
    say(env, "bowl at", when="after the hover", p=[round(float(v), 3) for v in _np(task.bowl.pose.p)[0]])
    say(env, "hovering", xy_to_bowl=round(float(_np(hinfo["xy_to_bowl"]).reshape(-1)[0]), 3),
        clearance=round(float(_np(hinfo["clearance"]).reshape(-1)[0]), 3),
        height_ok=_flag(hinfo, "height_ok"), over_bowl=_flag(hinfo, "over_bowl"))
    if HOVER_CORRECT and not _flag(hinfo, "over_bowl"):
        # The TCP reached its target (tcp_err 1-2 cm) and the object is 12-15 cm from the
        # bowl (1619, 1867 @0.15; 452 @0.0): the object moved in the fingers on the way
        # here, so the offset the hover was aimed with — T_tcp_obj, read after the drive
        # — is stale, and every pour candidate below would be aimed with it too. Read the
        # object as it sits in the hand NOW, re-aim the hover once with it, and pour with
        # the live offset. One correction: a second miss says the object is loose.
        T_live = (task.agent.tcp.pose[0].inv() * target.pose[0]).sp
        # INSTRUMENT (2026-09-06): which frame is off? The real TCP against the hover
        # target, the planner's believed TCP (FK of the qpos it plans from) against the
        # real one, the object against the bowl, and the in-hand offset stale vs live.
        try:
            _tcp_real = _np(task.agent.tcp.pose.sp.p).reshape(-1)[:3]
            _obj_real = _np(target.pose.sp.p).reshape(-1)[:3]
            _pw = planner.planner
            _pw.update_from_simulation() if False else None
            _q = _np(task.agent.robot.get_qpos()).reshape(-1)
            _pw.pinocchio_model.compute_forward_kinematics(_q)
            _tcp_model = _pw.pinocchio_model.get_link_pose(_pw.link_name_2_idx[_pw.move_group])
            _tcp_model_p = np.asarray(_tcp_model.p, dtype=np.float64).reshape(-1)[:3]
            _base_model = np.asarray(_pw.base_pose.p if hasattr(_pw, "base_pose") else [0, 0, 0], dtype=np.float64)
            say(env, "hover instrument",
                tcp_real=[round(float(v), 3) for v in _tcp_real],
                hover_target=[round(float(v), 3) for v in np.asarray(hover.p)],
                tcp_minus_target=[round(float(v), 3) for v in (_tcp_real - np.asarray(hover.p))],
                tcp_model_local=[round(float(v), 3) for v in _tcp_model_p],
                base_model=[round(float(v), 3) for v in _base_model],
                base_real=[round(float(v), 3) for v in _np(task.agent.base_link.pose.sp.p).reshape(-1)[:3]],
                obj_minus_bowl=[round(float(v), 3) for v in (_obj_real - bowl_p)],
                bowl_stale=[round(float(v), 3) for v in bowl_p],
                bowl_live=[round(float(v), 3) for v in _np(task.bowl.pose.p)[0].astype(np.float64)],
                T_stale=[round(float(v), 3) for v in np.asarray(T_tcp_obj.p)],
                T_live=[round(float(v), 3) for v in np.asarray(T_live.p)])
        except Exception as exc:  # diagnostic only
            say(env, "hover instrument failed", why=f"{type(exc).__name__}: {exc}")
        extra0, back0, _spin0 = HOVER_RUNGS[0]
        # The BOWL as it stands now, too: measured on 1619 @0.15 (2026-09-06) the object
        # hovered 8 mm from where the bowl had been read after the drive, and the bowl
        # itself stood 15 cm away — pushed on the way in. The task judges against the
        # live bowl; so does this re-aim, and so do the pour candidates after it.
        bowl_live = _np(task.bowl.pose.p)[0].astype(np.float64)
        bowl_moved = float(np.linalg.norm(bowl_live[:2] - bowl_p[:2]))
        aim0 = bowl_live - back0 * face
        hover2 = common.pose_over(aim0, HOVER_ABOVE + extra0, obj_q_held) * T_live.inv()
        say(env, "hover corrected for the object as held and the bowl as it stands",
            xy_before=round(float(_np(hinfo["xy_to_bowl"]).reshape(-1)[0]), 3),
            bowl_moved=round(bowl_moved, 3),
            shift=[round(float(v), 3) for v in (np.asarray(hover2.p) - np.asarray(hover.p))])
        with common.keepout(planner, [distractor], pad=GRASP_KEEPOUT_PAD):
            r2 = common.arm_move(env, planner, hover2, who=WHO, stage="hover (corrected)",
                                 disable_lift_joint=False,
                                 max_knots=HOVER_MAX_KNOTS, knot_draws=HOVER_KNOT_DRAWS)
        if r2 != -1:
            res = r2
            if common.stopped_by_horizon(planner):
                return res
            T_tcp_obj = T_live
            bowl_p = bowl_live
            hinfo = res[-1]
            say(env, "hovering", xy_to_bowl=round(float(_np(hinfo["xy_to_bowl"]).reshape(-1)[0]), 3),
                clearance=round(float(_np(hinfo["clearance"]).reshape(-1)[0]), 3),
                height_ok=_flag(hinfo, "height_ok"), over_bowl=_flag(hinfo, "over_bowl"),
                corrected=True)
        else:
            say(env, "the corrected hover refused; pouring as hovered")

    # -- STAGE 6: pour — the tilt candidates -------------------------------------------------
    along = _np(task._along_np)[0].astype(np.float64)
    along = np.array([along[0], along[1], 0.0])
    planned_any = False
    last = res
    # K61: order the candidates so the ones a **straight** screw can reach come first.
    # `static_manipulation` falls back to RRTConnect silently, and an RRT path wanders
    # in joint space — on seed 0 the pour came back `plan=rrt knots=115` after the screw
    # refused on joint 3 (the torso), and that wander is what the video shows as the arm
    # rotating twice and rising before it tips. Seed 1's pour planned by screw (37 knots)
    # and looks smooth. The probe plans only, so a candidate it rejects costs no episode
    # steps; the order is a permutation, so every candidate is still tried and the
    # measured tilt-ladder semantics are unchanged.
    cands = list(pour_candidates(face, along))
    # Indices, not membership: a candidate is `(numpy axis, tilt)` and `in` would compare
    # the arrays element-wise and raise on the ambiguous truth value.
    straight = [i for i, (axis, tilt) in enumerate(cands)
                if common.screw_plans(
                    planner, pour_pose_for(bowl_p, POUR_ABOVE, obj_q_held, T_tcp_obj, axis, tilt))]
    if straight:
        rest = [i for i in range(len(cands)) if i not in straight]
        cands = [cands[i] for i in straight + rest]
    say(env, "pour candidates ordered", straight=len(straight), total=len(cands))
    for axis, tilt in cands:
        pour = pour_pose_for(bowl_p, POUR_ABOVE, obj_q_held, T_tcp_obj, axis, tilt)
        say(env, "pour", axis=[round(float(v), 2) for v in axis], tilt_deg=tilt,
            tcp=[round(float(v), 3) for v in pour.p])
        with common.keepout(planner, [distractor], pad=GRASP_KEEPOUT_PAD):
            res = planner.static_manipulation(pour, disable_lift_joint=False)
        if res != -1 and common.stopped_by_horizon(planner):
            return res
        if res == -1:
            continue
        planned_any = True
        # The angle is read after the arm settles: a tilt read mid-motion is the
        # commanded pose's error, not the object's rest attitude.
        settled = planner.idle_steps(t=POUR_SETTLE_STEPS)
        if settled != -1:
            res = settled
        if res != -1 and common.stopped_by_horizon(planner):
            return res
        last = res
        flags = _pour_flags(res[-1])
        say(env, "pour candidate executed", tilt_rad=round(float(_np(res[-1]["tilt_rad"]).reshape(-1)[0]), 3), **flags)
        if not flags["grasp_target"]:
            say(env, "missed: dropped during the tilt")
            return res
        if not flags["distractor_ok"]:
            say(env, "missed: distractor moved during the tilt")
            return res
        if flags["tilted"] and flags["over_bowl"] and flags["height_ok"]:
            break
    else:
        if not planned_any:
            return fail(env, "pour: no candidate planned")
        say(env, "missed: no pour pose reached after all candidates",
            tilt_rad=round(float(_np(last[-1]["tilt_rad"]).reshape(-1)[0]), 3))
        return last

    # -- STAGE 7: hold ---------------------------------------------------------------------
    # Polled rather than one block: `pour_hold` resets on any single step where
    # `pour_now` is false (season_dish.py), so one flicker inside a 20-step window
    # loses an episode that would have latched at step 30. Verdicts land around step
    # 550-815 of an 1100 horizon, so the slack is there to spend.
    say(env, "hold", steps=int(task.cfg.hold_steps) + 5, budget=HOLD_POLL_BUDGET)
    spent = 0
    while True:
        res = planner.idle_steps(t=HOLD_POLL_CHUNK)
        if res == -1:
            return fail(env, "hold")
        spent += HOLD_POLL_CHUNK
        if common.stopped_by_horizon(planner):
            say(env, "stopped by the horizon during the hold")
            return res
        if _flag(res[-1], "success"):
            break
        if spent >= HOLD_POLL_BUDGET:
            break
    say(env, "held", success=_flag(res[-1], "success"), pour_hold=int(_np(res[-1]["pour_hold"]).reshape(-1)[0]))
    return res


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--scene-idx", type=int, default=0)
    parser.add_argument("--output-dir", default="videos/season_dish")
    parser.add_argument("--render-mode", default="rgb_array")
    parser.add_argument("--render-width", type=int, default=512)
    parser.add_argument("--render-height", type=int, default=512)
    parser.add_argument("--max-steps-per-video", type=int, default=None)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-trajectory", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--blind",
        action="store_true",
        help="arm B of the control experiment: ignore the cue, take a uniformly random condiment (seeded)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    env = gym.make(
        "MikasaSeasonDish-v0",
        num_envs=1,
        render_mode=args.render_mode,
        robot_uids="mikasa_ds_fetch",
        control_mode="pd_joint_pos",
        scene_idx=args.scene_idx,
        human_render_camera_configs=dict(
            width=args.render_width, height=args.render_height
        ),
    )
    env = RecordEpisode(
        env,
        output_dir=args.output_dir,
        save_video=not args.no_video,
        save_trajectory=not args.no_trajectory,
        video_fps=30,
        save_on_reset=True,
        max_steps_per_video=args.max_steps_per_video,
    )

    # solve() seeds everything (Python, numpy, torch, mplib) next to its reset.
    res = solve(env, seed=args.seed, debug=args.debug, vis=False, blind=args.blind)
    if res == -1:
        print("failed_motion_plan")
    else:
        print("success:", bool(res[-1]["success"][0]))
    env.close()
    return res


if __name__ == "__main__":
    sys.exit(0 if main() != -1 else 1)
