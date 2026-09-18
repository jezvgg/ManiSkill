"""Scripted oracle for MikasaCabinetRetrieval-v0: side-grasp the cup off the cabinet
shelf through the open door, carry it down, stand it on the counter.

Every stage is the W13 chain (tools/probes/w13_cabinet_take.py, journal 2026-08-28),
which is the measurement this oracle exists to reproduce under the task's own reset:

- the VERTICAL top grasp never plans through the opening (0/9 honest cells), so the
  grasp here is the horizontal side grasp — approach along +y into the cabinet,
  closing across, TCP a few mm short of the cup's axis;
- from the shelf the exit is +0.03 m (a +0.10 lift refuses — the wrist is already
  high), then straight back through the opening, then down to the counter;
- mplib's IK seeds from the *current* configuration plus random inits, so every
  cartesian stage keeps its `tries` and the stage order is load-bearing.

Motor-only: `blind` is accepted (run_sweep requires the literal parameter) and
ignored — there is no cue to blind. The memory variant will make it real.

Return contract (D6): `-1` only for a planning or grasp refusal before physics has
committed a decision; a physical miss (the cup slipped, the place did not settle)
returns the last gym 5-tuple so the sweep books it `missed`.

The door stages (`open_the_door`, `close_the_door`, `door_rad_now`) take a `DoorSpec`
(K113): every world number and every `task.cfg` read they used to make goes through
it, so MikasaCabinetSearch-v0 drives three leaves with the same measured code. With
`door=None` the spec is built from this task's config — the Closed oracle's path,
byte-identical to K105–K108.
"""

from __future__ import annotations

import dataclasses
import math
import os
import types

import numpy as np
import sapien

from utils.mikasa_oracle.planners import oracle_common as common
from utils.mikasa_oracle.planners.oracle_common import fail as _fail
from utils.mikasa_oracle.planners.oracle_common import say as _say
from utils.mikasa.seeding import seed_everything

WHO = "cabinet_retrieval_planner"

#: TCP offsets along the approach past the cup's axis for the side grasp, in metres
#: (negative = short of the axis). W13 measured -0.005 planning and holding (pads at
#: 0.0337/0.0307 around the body); the K62 review's corrected palm arithmetic says
#: -0.005 clears the tapered wall by ~3.5 mm and -0.010 by ~8.5 mm, while 0.0 puts
#: the palm 1.5 mm inside it — kept last as the knife-edge rung.
SIDE_GRASP_DEPTHS = (-0.005, -0.010, 0.0)

#: Where the base parks, metres south of the cabinet front. W13's grid: from
#: y = -1.05 every spawn depth in the task's band plans; -1.15 only the deepest.
WORK_DOCK_Y = -1.05

#: The exit ladder after the grasp. `UNSHELVE_DZ` clears the shelf; the retract is
#: a REACH from the base, not a world y — the first run refused `joint limit at
#: index [7]` because a fixed y=-0.70 from the -1.05 dock is a 0.35 m reach at
#: 1.5 m of height, the same close-and-high pose the vertical grasp died on (W13
#: retracted to the same -0.70 happily from its -1.30 dock: 0.65 m of reach).
UNSHELVE_DZ = 0.03
RETRACT_REACH = 0.55
HOVER_Z = 1.12

#: How far above the place target the cup's MESH BOTTOM ends the descent. 2 cm, the
#: same figure every spawn in this repo uses; the release drops the cup the rest.
PLACE_DROP = 0.02

#: Steps of stillness after the release, so `settled` can latch before the episode
#: is read. The predicate needs is_static, and a cup released 2 cm up wobbles.
SETTLE_STEPS = 30

#: The closed-door variant (MikasaCabinetRetrievalClosed-v0), K103/K104 numbers.
#: HANDLE_BAR is the CLOSED right door's bar, world — W12 scanned it off the door
#: link's collision shapes (a vertical bar r=0.013 standing 0.05 m off the panel);
#: valid ONLY at door qpos 0, which is why validate() refuses the ajar band.
#: HANDLE_DOCK is where every arc-pull number was measured from (W12/W14).
HANDLE_BAR = (2.302, -0.430, 1.592)
HANDLE_DOCK = (2.30, -1.30)
DOOR_OPEN_TARGET = 1.75
"""Past the task's 1.6 on purpose, for two measured reasons (sweep 2): the freed
door drifts back ~0.05 rad after the release (1.601 -> 1.552), and at 1.55 the
panel's free edge sits 2.5 cm deeper in the grasp corridor than the 1.6 every
v0 margin was measured at. Past ~1.6 the panel swings BEYOND the x=2.75 hinge
line, out of the work lane; the hinge stop is 3.0."""
ARM_PASS_RAD = 0.9
"validate()'s floor: below this angle the opening does not admit the arm (W12)."
PULL_MAX_STEPS = 1100
"""K104's rest-grip form took 591 steps to full open at v=0.05; the DRIVEN
grip form was slower still (1.44-1.48 rad at a 700 cap, why=max-steps,
fingers alive). At PULL_V_HANDLE=0.20 the measured pull is ~144 steps, so
this cap is now a 7x cushion — kept, because the cap is paid only by
progress: a true stall ends the pull early via stall_window."""
PULL_V_HANDLE = 0.20
"""W16 (K107): the sweep 0.05/0.10/0.15/0.20 all reached 1.75 with the grip
untouched (26.1->25.5 mm at every rung) in 655/311/194/144 steps — linear in
1/v, ceiling above the grid — and the settled angle at speed is HIGHER
(1.700 vs 1.649: less drift-back window). 0.20 is the user's pick of the two
measured-clean ship values. The CLOSING push keeps its own measured 0.05
(fist contact, not a grip — the speed transfer is unmeasured there)."""

#: The closing stage (require_door_closed): W15's DOCK_GRID as a LADDER, in the
#: order the FULL ORACLE measured, not W15's lever order. W15 teleported the
#: base and crowned (3.4, -0.9) (fist lever a_x = 0.62, closing arc 1.503 ->
#: 0.149 rad in 293 monotonic steps) — but the oracle has to DRIVE there from
#: the retrieval's parking, and that screw clipped `base_link<->stove` on 8 of
#: 8 attempts (cycle sweep 5): the W15 winner is unreachable by drive, parked
#: last. (3.2, -1.1) planned 3/3 and closed the door to 0.116-0.12 rad on
#: every retrieval that delivered — the measured first rung.
CLOSE_DOCKS = ((3.2, -1.1), (3.2, -0.9), (3.0, -1.1), (3.4, -0.9))
LADDER_REACH_CHECK = os.environ.get("MIKASA_CLOSE_LADDER_REACH", "1") == "1"
"""Whether a rung has to afford the PUSH, not merely the drive.

Measured 2026-09-04 on `runs/2026-09-03-fix-e` (three arms, 30 seeds each):
`pre-contact the panel` is the TERMINAL stage of all 13 seeds that main solves and
the arm+reverse arm does not, its counter runs 6 -> 14 -> 28, its refusal is the same
28/28 (`screw plan failed: collision` + `IK Failed`) — and every one of those 28 is at
RUNG 0. The remaining three rungs were never tried with the push, because the stage
returned `fail` the moment the pre-contact plan refused. The ladder was choosing a dock
by whether the DRIVE plans and never asking whether the panel is reachable from it.

On: a refused pre-contact hands control back to the ladder and the next rung is driven.
The fist is made once, at the first rung that is taken, so the later rungs are driven
with a closed hand — a smaller sweep than the stowed one, never a larger one.
Off: the measured K106 ladder and the K109 acceptance, byte for byte."""
CLOSE_R_PUSH = 0.35
"Push-point radius on the panel's mid-plane arc from the hinge (W15's winner)."
CLOSE_PUSH_Z = 1.55
"Panel z-span is 1.39..2.31 (W12); lower third, inside the arm's reach."
CLOSE_PUSH_GRID = (
    ((CLOSE_R_PUSH, CLOSE_PUSH_Z), (CLOSE_R_PUSH, 1.85), (0.43, 1.70), (0.25, 1.55))
    if os.environ.get("MIKASA_CLOSE_PUSH_GRID", "0") == "1"
    else ((CLOSE_R_PUSH, CLOSE_PUSH_Z),))
"""`(radius, z)` push points to try, in order; the first is W15's measured winner, so
a run in which it plans is today's run.

DEFAULT OFF, and the default is a measurement, not a preference — see the verdict at
the end of this docstring.

Why a ladder here at all (measured, runs/2026-09-04-fix-f): once the closing rungs were
made accountable to the push, `pre-contact the panel` stayed the terminal stage of 20 of
CabinetSearch's 21 losses, and its refusal is 12x `IK Failed` against 10x `screw plan
failed: collision` — no collision-free IK solution EXISTS for this one pose from any of
the four rungs, not a planner that ran out of budget. Folding the arm is already tried
and does not help (the pass-B fold runs 47 times), so the pose is what has to move.

Where the alternatives come from, not invented: the panel spans z 1.39..2.31 (W12), so
1.85 is mid-panel rather than lower-third; 0.43 is the handle bar's own radius from the
hinge (`handle_bar` x 2.302 against hinge 2.735), i.e. a radius the leaf demonstrably
carries; 0.25 is inboard, for the case where the leaf's outer half is what is blocked.
Conclusion 5 of the protocol was the reason to expect the higher point to help: the arm
package reaches by EXTENDING rather than by lowering the body, and an extended arm is
high and far, which is where 1.85 sits and 1.55 does not.

VERDICT (runs/2026-09-04-fix-g, three arms, CabinetSearch, 30 seeds, incomplete=0):
**it buys nothing.** The grid fired — 189 `no IK at this push point` lines against 0 on
both other arms — walked all four points every time, and not ONE alternative point ever
planned. The per-seed string is identical to the arm without it, character for character
(000010100000001010110101100000), because the grid always exhausts and falls through.
So the push POINT is not the binding constraint; where the base stands relative to the
open leaf is. The expectation from conclusion 5 is hereby measured and refuted. Kept in
the tree, off by default: the mechanism and its tests are the record of that measurement,
and turning it on costs planning time for nothing."""
CLOSE_RETRACE = os.environ.get("MIKASA_CLOSE_RETRACE", "1") == "1"
"""Close the leaf by UNDOING what the robot did after it opened it, instead of driving
to a closing dock on the far side and pushing the panel with a fist.

Owner's idea, 2026-09-04, and the measurements point the same way. Two fixes aimed at
the fist path returned 2 seeds and 0 seeds of the 13 (runs 2026-09-04-fix-f and -fix-g),
and the second closed the question of where the constraint lives: the push POINT is not
it — where the base stands relative to the open leaf is. The retrace never stands
anywhere new. It backs up to the pose the pull ended in, takes the bar again, and rides
the opening arc backwards.

The one thing that makes this sound rather than hopeful: `follow_arc` never re-orients
the arm — every step it re-commands the arm at its MEASURED qpos and gives (v, omega) to
the BASE alone (`extand.py:follow_arc`). So across the pull the gripper's world
orientation changed only by the base's own yaw change, and the hand that held the bar is
one rigid rotation about the hinge away from the hand that can hold it now. That is why
the grasp is RESTORED from the recorded TCP pose and merely re-positioned for the drift,
never recomputed from the live angle: recomputing gets the approach direction wrong by
most of the door's swing.

Off: not one byte of the shipped closing path changes."""
PUSH_HERE_R = float(os.environ.get("MIKASA_PUSH_HERE_R", "0.35"))
PUSH_HERE_Z = float(os.environ.get("MIKASA_PUSH_HERE_Z", "1.59"))
LAST_PUSH: dict = {}
"""What the last push in `close_the_door` reached (`pushed`: the leaf's angle when the
arc stopped), for the caller's verdict: a leaf that reached the closed band and then
reads open again SPRANG BACK — the forearm was round it (cab_2: 1177, 1485, 1681) — and
a second push from the same place wraps it again; the re-push goes by the ladder's rung."""
PUSH_HERE_R_FALLBACK = 0.40
"""The bar's own radius: the shortest reach from the opening pose, tried when PUSH_HERE_R
is not it. Measured (2026-09-05, seed 1177 g28-g30): at 0.40 the push closed cab_2's left
leaf to 0.116 and the forearm, round the leaf's free edge, sprang it back to 0.47 — twice;
at 0.35 (the ladder's CLOSE_R_PUSH) the same leaf closed at the first attempt. But 0.35
alone refused the reach on four leaves that 0.40 had planned (g31) and the ladder drove.
So the from-here grid is PUSH_HERE_R first, this radius second, the ladder's point last."""
"""Where the push-from-here reaches first: the bar's own radius and height (HANDLE_BAR z
= 1.592, r ~ 0.40 from the hinge), i.e. just past the outer side of the leaf, next to the
hand that has just let go of the bar. The ladder's point (r 0.35, z 1.55) was drawn for a
base that has driven round to the hinge side and refused on IK from the opening pose
(`joint limit at index [3]`)."""
PUSH_FROM_HERE = os.environ.get("MIKASA_PUSH_FROM_HERE", "1") == "1"
"""Try the fist push from where the base stands BEFORE the ladder of closing docks.

The owner's fallback for the closing (2026-09-05, after the 1105/1250 recordings): if
the reversal cannot take the bar again, do not drive round the leaf — extend the arm to
its OUTER face from the opening pose and push it shut, which is what the ladder does at
the end of its drive anyway. `close_the_door` never tried it: pass A of the ladder is
"straight from the parking" only in the sense of not folding first, and still drives to
`door.close_docks[0]` before the first reach. Here the reach is tried first; a refusal
executes nothing and leaves the ladder its usual pose. 0 until measured."""
RETRACE_CARRY = os.environ.get("MIKASA_RETRACE_CARRY", "0") == "1"
"""Carry the recorded stance round the hinge by the leaf's settle before the reversal. OFF.

**MEASURED 2026-09-05, AND THE CARRY WAS THE STRIKE.** With the look a turn and no fold
(base never moves), seeds 1105 and 1250:

    carry on    along -60 / -58 mm   keep_side -61 / +62 mm   keep_hit finger<->neighbour leaf
    carry off   along  -1 /  -3 mm   keep_side  +1 /  +1 mm   no hit, bar HELD, arc riding

It followed the opened leaf's 3-4 degree settle with ~7 cm of base motion, toward a
neighbouring leaf that had not moved and whose edge is 5 cm from the bar. The settle
itself costs 1-3 mm and the pads forgive it; the cure cost 60. Kept switchable because
the measurement is worth repeating, shipped off.

The reasoning that put it in (2026-09-05, one day earlier): with the look a turn and no fold, the base never moves
and the re-grasp still misses by ~60 mm along the approach, with `keep_hit` naming the
NEIGHBOURING leaf and the shove starting in the second quarter — on the approach itself.
The carry moves the base ~7 cm to follow the OPENED leaf's 3-4 deg settle; the neighbour
did not settle, and the bar stands 5 cm from its edge. The carry may be exactly what puts
the open finger into it. 0 = reach for the bar where it was recorded, base untouched."""
RETRACE_ARC = os.environ.get("MIKASA_CLOSE_RETRACE_ARC", "0") == "1"
"""Whether the retrace tries to CLOSE the leaf, or only walks the base back.

The question the first measurement forced (runs/2026-09-04-fix-h, CabinetSearch, 30
seeds): the retrace scores 27/30 against main's 20/30 — and closes the door itself
exactly ZERO times. It catches the bar 26 times, the arc stalls 26 times, and 26 times
the leaf is ridden back OPEN and the ladder finishes the job. What the ladder gets that
it did not have before is the POSE: `pre-contact the panel` refusals fall 22 -> 2 and
closing-dock refusals 284 -> 34, because the base leg walks the robot out of the corridor
between the two leaves where every rotation was refused.

VERDICT, three arms and 122 retraced rounds later: **the arc does not pay, and it ships
OFF.** It was tried three ways and none of them closed a single door.

| when the pads close | grip rate | doors closed by the arc | outcome |
| :--- | ---: | ---: | ---: |
| on arrival (fix-i, R) | 26/40 | 0 | 27/30, median 4433 steps |
| at 0.6 of the way in (fix-j, G) | 0/40 | 0 | 26/30, median 3954 |
| at 0.9, keeping its progress (fix-k, K) | 13/41 | 0 | 27/30, longer than B |

The base leg ALONE scores the same 27/30 at median 3908 steps (fix-i and -fix-k, arm B),
so every step the arc spends is spent for nothing. What earns the seeds is the base leg
walking the robot out of the corridor between the two leaves, where the fist path's
rotations were refused 212 times: `pre-contact the panel` falls 22 -> 2 and closing-dock
refusals 284 -> 34.

Kept in the tree, off, with its flags: three measurements are worth more as code that can
be switched back on than as a paragraph. The one thing the arc DID prove is that it is
not idle when the grip holds — it carries the leaf from ~1.86 rad to a median of 0.817 —
so if it is ever revisited, the question is why it stalls at half a swing, not whether
the grasp works."""
RETRACE_DRIFT_MAX_RAD = 0.25
"""How far the leaf may have settled from the recorded angle and still be retraced.
0.25 rad is 109 mm of travel at the bar's radius; beyond it the rigid rotation stops
being a small correction and the arm is no longer demonstrably able to reach."""
RETRACE_BASE_XY_TOL = 0.004
RETRACE_BASE_TRIES = 4
RETRACE_BASE_FAST_M = 0.05
"""How close in metres the base leg has to get, and how many looks it may take.

One aim-and-drive leaves 12-15 mm every time, and always the same way: `drive_straight`
stops on `travelled >= dist`, which overruns by up to one step (5 mm at 0.10 m/s), and
under the old 10 mm entry gate a short leg was not driven at all. Measured over the
rounds that DID grip, `d_base` was 12.6 to 16.6 mm and the miss along the approach 11 to
15 mm — the grip was living on the last few millimetres of a 37 mm allowance. Three looks
cost a few tens of steps and take the residue to the tolerance."""
RETRACE_BASE_TOL = 0.005
"""How exactly the base has to be back before the arm's own reversal is believed.

`turn_in_place` defaults to `tol=0.03` rad — 1.7 deg — and stops the moment it is inside
that, which is why the base came back to a median of 1.5 deg: the tolerance WAS the
error. The arm is reversed at the precision of its controller and the base at 1.7 deg,
and at the 0.76 m the hand stands from the base axis that is 20 mm of the roughly 37 mm
the gripper has to spare (pads 100 mm, bar 26 mm). Measured over 38 retraces: d_dock
median 0.018 m, dyaw median 1.5 deg, and the pads then close on nothing 32 times in 38.

0.005 rad is 0.29 deg, about 4 mm at that lever. Not a tuned number — the point is that
the base's reversal should not be the loosest link in a chain the rest of which is exact.
It costs steps, and `turn_in_place` slows inside 4x tol, so the cost is bounded."""
RETRACE_BASE_SKIP_M = float(os.environ.get("MIKASA_RETRACE_BASE_SKIP_M", "0.05"))
RETRACE_BASE_SKIP_RAD = float(os.environ.get("MIKASA_RETRACE_BASE_SKIP_RAD", "0.20"))
"""With the arc OFF, the base leg is skipped when the base already stands within this of
the recorded pose (metres, radians); 0 = never skip.

The base leg's millimetre precision (RETRACE_BASE_XY_TOL, RETRACE_BASE_TOL) was built for
the arc: reversing a recorded grasp needs the base back where the record was made. With
the arc off (RETRACE_ARC, measured and shipped off) what follows is the from-here push,
which reads the leaf's live frame and reaches from wherever the base stands; and since the
look is a turn (LOOK_BY_TURN) the base never leaves the opening pose — the leg was
correcting the 1-2 cm the look's two turns leave, as a turn-drive-turn: on 1105 (2026-09-07,
integration 399d6df) +14 deg, 1.4 cm in reverse, -14 deg, three motions the owner saw as
"turns away, turns back, and only then closes". The skip is coarse on purpose: the push
does not need the record, only a base near enough that its 0.35-0.40 m reach is the one
that was measured. The drive-to-look case (LOOK_BY_TURN=0, 0.4-0.5 m away) still drives."""
RETRACE_TOUCH_RAD = 0.02
"""A base leg that moves ANY joint of the cabinet by this much stops on that step.
The base legs are unplanned (that is the point — the planned ones are what refuse), so
the leaf is watched by state instead of by a collision sweep. Well under the task's own
`theta_count` = 0.05 rad, which is what latches a compartment as opened."""
RETREAT_LIFT_M = float(os.environ.get("MIKASA_RETREAT_LIFT_M", "0.0"))
RETREAT_ALONG_FACE = os.environ.get("MIKASA_RETREAT_ALONG_FACE", "1") == "1"
"""The retreat after the release runs along the LEAF's plane — the hand's own axis
projected onto it — instead of along the hand's axis as it stands. Measured
(2026-09-05-repro-cs, 1500-1699): the retreat along the hand's axis dragged the leaf
toward closed by 0.10-0.15 rad on 7 of 251 cab_2 leaves and 16 of 259 cab_main ones —
the inner pad, in the 3 cm between bar and face, presses the face while it slides
because the hand's axis is a few degrees into the panel. On cab_2 a dragged leaf
(1.60-1.63 instead of 1.72-1.75) is what makes the push from here wrap the forearm
round the leaf and the leaf spring back (1177, 1485, 1681). Only when a retrace record
is kept (the search task); the tape is not replayed with the reversal off."""
"""Lift of the OPEN hand off the leaf's face, along the live normal, before the taped
retreat (only when a retrace is recorded and RETRACE_ARC is on; 0 = the shipped retreat).
The pads close along what was the closed leaf's x and still is, 100 deg of arc later —
a friction grip lets the bar turn in the hand — so at the pull's end the retreat along
the hand's own z is a SLIDE along the face (hand_vs_face_tail, g22: d_n 0.02-0.04 the whole
way while d_t runs 0.18 -> 0.02). Its reversal strikes the face whenever the leaf has
settled less than the 2-3 deg the old look-turn used to push it (g20 landed at drift
-0.062, g21 struck at -0.028, same seed, base back to 3 mm). The bar has 37 mm of play
between open pads; lifting 20 mm off the face first was to give the slide, and its
reversal, that much clearance.

MEASURED OFF (g25, 2026-09-05, six leaves): the lift moved the hand +3..+10 mm relative to
the bar and the LEAF +0.019..+0.021 rad — the pads close along the normal at the pull's
end and the inner pad stands in the 3 cm between bar and face, so a lift along the normal
drags the bar with it instead of clearing the face. The play is not 37 mm here; it is the
gap minus the pad. With the pads on the normal, the slide in the gap is the only way in
or out, and its clearance (~1 cm) is inside the landing's own scatter plus the settle:
the reversal is marginal by construction, and g16-g20's landings rode on the look-turn
nudging the leaf 2-3 deg away. Kept as a knob for the measurement; shipped at 0."""
RETRACE_SHOVE_RAD = 0.05
"""The reversal is BLIND — recorded actions, nothing planned, nothing refuses — so the one
sign that the hand has struck the leaf instead of finding the bar is the leaf moving under
a base that is being held still. Measured: on cab_main the leaf sits within +-0.013 rad of
where the replay found it (seeds 1105/1110/1177/1199/1250); on cab_2's left leaf the finger
met the panel, the replay carried on for its remaining 55 steps and pushed the leaf 0.35 rad
further open (1150/1177/1199), past any reach the push from here has, and the base 43 mm
sideways with it. The replay stops once the leaf is RETRACE_SHOVE_RAD MORE OPEN than the record — signed:
the settle after the release runs toward closed (-0.03..-0.08 and more) and the pads pull a
settled bar back only as far as the grip; both read as shoves under the two unsigned
criteria tried first (g18, g19) — and hands over with the leaf where the opening left it,
which is the leaf the ladder was measured on.
cab_2 from the opening pose: the outer face points AWAY from the base, and every IK
solution to the push point sweeps wrist_roll/gripper through the leaf (1150/1177, g18),
so facing the point cannot help there (measured, 2/2) — the first rung is the least drive."""
RETRACE_UNDO_ON_STALL = os.environ.get("MIKASA_RETRACE_UNDO_STALL", "0") == "1"
"""Whether a stalled arc rides the leaf back OPEN before handing over.

It used to, on an argument: the fist path's push point recedes from every rung as the
leaf closes, so a half-shut leaf looked like a harder start than an open one. The
argument was never measured, and the measurement says it was expensive. When the grip
holds, the arc is NOT idle — it carries the leaf from ~1.86 rad to a median of 0.817
(n=26, runs/2026-09-04-fix-i), better than half the swing — and riding it back to 1.75
throws every bit of that away and hands the ladder a fully open door. Off by default:
keep what the arc won and let the ladder finish from there."""
RETRACE_HORIZON_FRAC = float(os.environ.get("MIKASA_RETRACE_HORIZON_FRAC", "0.90"))
"""Refuse the retrace once the episode has spent this much of its horizon.

Was 0.55, on the reasoning that a round which retraces AND hands over pays for both
paths. The reasoning is right and the number was wrong: MEASURED, refusing costs more
than paying. The two CabinetSearch seeds that stood between this task and 30/30 both
died here and nowhere else —

    retrace: handing over to the ladder reason=no step budget spent=4098 horizon=7100

— and both are solved at 0.90 (seeds 8 and 12, probed both ways, then confirmed on the
full pool: 28/30 -> 30/30, and the per-seed string differs in exactly those two).

The mechanism is that the gate switches off the CHEAPER path on a long episode and
leaves the expensive fist path to run alone, which is also the path that flings the leaf
past 2.5 rad and then cannot close it. Seed 8's verdict at 0.55 was `MISSED: the door did
not close door_rad=2.535`.

0.90 still leaves a tenth of the horizon, and the episode-length guard is separate: the
cleanliness threshold is steps/horizon <= 0.7 and it is measured, not assumed."""
RETRACE_SETTLE_STEPS = 10
RETRACE_SETTLE_TRIES = 6
"""The arc leaves the leaf turning at ~0.23 rad/s against the env's 0.02 rad/s stillness
term, so the stage idles until the env's own latch fires rather than until the angle
looks right. Six rounds of ten steps is 3 s."""
RETRACE_GRIP_MIN = common.FINGER_EMPTY_M
RETRACE_GRIP_MAX = 0.045
"""Finger aperture that means the bar is held, read after the pads have SETTLED.

The bar is 26 mm; a grip on it measured 26.1 -> 25.5 mm through the pull, a hand that
never closed reads 0.100. The floor is `oracle_common.FINGER_EMPTY_M` and not a number
of this module's own, because `pull_hinge_arc` refuses on exactly that threshold a
moment later: a gate that disagreed with it would announce a grip and then hand the arc
an empty hand, which is what one smoke test did (`the bar is held` said twice on seed 2,
then `nothing between the pads fingers=[0.0054, 0.0011]`). The lesson was the READ, not
the threshold — `close_gripper` returns while the pads are still moving, so the aperture
is settled first."""
RETRACE_GRIP_SETTLE = 6
"Steps to let the pads finish closing before the aperture is believed."
RETRACE_STRETCH = 1
"""How many times each recorded action is re-issued. MEASURED AT 1: slower is WORSE.

Once the base's along-heading and heading terms are held (1.4 mm, 0.2 deg), what is left
of the miss is entirely SIDEWAYS in the base frame — 41 to 54 mm on the rounds where the
pads shut on nothing against 4 to 11 mm on the rounds that grip. A forward-only base
cannot answer that term at all: the controller's own action is `[v, 0]` rotated into the
world (`PDBaseForwardVelController.set_action`), so the lateral velocity target is
structurally zero and the joint is left with damping alone. Damping is not a position
loop: a standing force gives a standing creep.

The hypothesis this constant tested was that the creep is the arm's INERTIAL reaction,
which would fall with speed. It does not. At `stretch=3` the sideways slide GREW, 41-54
mm -> 65-70 mm on the same four seeds, while `keep_fwd` stayed at 0.0 and one seed still
gripped. Slower means longer under the standing force, so the answer is the opposite of
slowing down. Kept as a parameter because the measurement is worth being able to repeat;
shipped at 1, which is byte for byte the recorded rate."""
RETRACE_HOLD_STEPS = 40
RETRACE_HOLD_TOL = 2e-4
"""How long the reversal may hold its FINAL command, and when it stops holding.

A tape frame is a target, not a position. Playing the window backwards issues the grip
frame as its last action and returns immediately, while the controller is still a step
or two behind it — and backwards that lag points AWAY from the bar, because frame 0 is
now approached from frame 1, i.e. from the retreat. Measured on seed 2 (2026-09-04, four
retraces): the base came back to 0.012-0.016 m and 0.0-0.1 deg, and the hand still
stopped 20/24/67/71 mm short of the recorded grasp — the pads then closed on nothing.
Holding the same recorded action until nothing outside the base moves any more is still
the recording, not a new plan, and it is also what gives the fingers their closing
steps: 40 steps is 2 s, an order more than the arm needs from 70 mm out."""
CLOSE_MAX_STEPS = 500
"W15 measured 293 steps for the full 1.5-rad close, plus ~70% margin."
CLOSE_MARGIN_RAD = 0.03
"""Stop the push this far BELOW door_closed_rad: the verdict reads the hinge
after the fist retreats, and W15's stop-at-threshold left only 1 mrad of
slack (0.149 against 0.15)."""


@dataclasses.dataclass(frozen=True)
class DoorSpec:
    """One hinged leaf of a wall cabinet, as the door stages need it (K113).

    Everything `open_the_door` / `close_the_door` / `door_rad_now` used to read
    from `task.cfg` or from this module's constants, in one frozen record, so the
    same measured stages serve a second and a third leaf (MikasaCabinetSearch-v0's
    compartments) without a second copy. World frame, kitchen 102, closed leaf.

    Attributes:
        stem: the articulation's exact stem (`cfg.cabinet_name` style — the scene
            key carries a suffix and is resolved by substring, as the task does).
        hinge: the leaf's active-joint name (`"rightdoorhinge"` / `"leftdoorhinge"`).
        open_dir: +1 when opening INCREASES qpos (limits [0, 3]), -1 when it
            DECREASES it (limits [-3, 0]). `door_rad_now` returns `open_dir * qpos`,
            so "how open" reads >= 0 for either leaf; `pull_hinge_arc` gets it.
        panel_dir: world-x direction from the hinge to the CLOSED leaf's free edge:
            -1 for a right leaf (hinge east, edge west), +1 for a left leaf. Sets
            the push point and, with `open_dir`, the closing arc's sense.
        handle_bar: the closed leaf's bar, world xyz (valid at qpos 0 only — the
            reason validate() refuses the ajar band).
        handle_dock: base xy for the pull, facing +y.
        close_docks: the closing ladder, base xy, on the hinge's side of the leaf.
            EMPTY means the leaf has no push ladder and `close_the_door` refuses
            it before spending a step — the W24 case (cab_1's ladder mirrors
            behind the room's west wall), where the task closes the round with a
            wall park instead. Every leaf that IS pushed shut carries rungs.
        closed_rad: the task's "closed" threshold on `open_dir * qpos`.

    Example:
        >>> CAB_MAIN_RIGHT.handle_bar == HANDLE_BAR and CAB_MAIN_RIGHT.open_dir == 1
        True
        >>> CAB_MAIN_LEFT.open_dir, CAB_MAIN_LEFT.panel_dir
        (-1, 1)
        >>> CAB_1.close_docks
        ()
    """

    stem: str
    hinge: str
    open_dir: int
    panel_dir: int
    handle_bar: tuple
    handle_dock: tuple
    close_docks: tuple
    closed_rad: float

    def __post_init__(self):
        assert self.open_dir in (1, -1), self.open_dir
        assert self.panel_dir in (1, -1), self.panel_dir
        assert len(self.handle_bar) == 3, self.handle_bar
        assert len(self.handle_dock) == 2, self.handle_dock
        # An EMPTY ladder is legal and means "this leaf is never pushed shut"
        # (cab_1 — `close_the_door` refuses it by name); a ladder with a
        # malformed rung is still a bug.
        assert all(len(d) == 2 for d in self.close_docks), self.close_docks
        # the scene's own band: 0 is unreachable (a free hinge holds where it is
        # released), ARM_PASS_RAD is the opening's floor, not a closed door
        assert 0.0 < float(self.closed_rad) < ARM_PASS_RAD, self.closed_rad


#: The scene's `door_closed_rad` (cabinet_retrieval_base.py): ~8.6 deg, a 7.3 cm
#: gap at the free edge; the push stops CLOSE_MARGIN_RAD under it and lands
#: 0.116-0.120 (K106).
DOOR_CLOSED_RAD = 0.15

CAB_MAIN_RIGHT = DoorSpec(
    stem="cab_main_main_group", hinge="rightdoorhinge", open_dir=1, panel_dir=-1,
    handle_bar=HANDLE_BAR, handle_dock=HANDLE_DOCK, close_docks=CLOSE_DOCKS,
    closed_rad=DOOR_CLOSED_RAD)
"""The measured leaf: hinge (2.735, -0.400) axis +z (K103), every K103-K108
number, the module constants by identity. `door_from_cfg` builds the same spec
from the Closed task's config; this constant is for tasks that name their doors."""

CAB_2_RIGHT = DoorSpec(
    stem="cab_2_main_group", hinge="rightdoorhinge", open_dir=1, panel_dir=-1,
    handle_bar=(1.302, -0.430, 1.592), handle_dock=(1.30, -1.30),
    close_docks=((2.2, -1.1), (2.2, -0.9), (2.0, -1.1), (2.4, -0.9)),
    closed_rad=DOOR_CLOSED_RAD)
"""`cab_main`'s right leaf shifted by exactly -1.000 m in x, a derivation, not a
scan: one_wall_small.yaml chains cab_2 -> cab_main (`align_to`, side right) with
the same `hinge_cabinet` [1, 0.40, 0.92], and the panel arithmetic
(robocasa cabinet_panels.py: hpad 0.05, vpad 0.20) reproduces the measured
(2.302, -0.430, 1.592) bar to 0.5 mm, so the shifted bar carries the same
construction. Hinge (1.735, -0.400). PROVISIONAL until W21b scans the bar."""

CAB_MAIN_LEFT = DoorSpec(
    stem="cab_main_main_group", hinge="leftdoorhinge", open_dir=-1, panel_dir=1,
    handle_bar=(2.199, -0.430, 1.592), handle_dock=(2.20, -1.30),
    close_docks=((1.30, -1.1), (1.30, -0.9), (1.50, -1.1), (1.10, -0.9)),
    closed_rad=DOOR_CLOSED_RAD)
"""The MIRROR leaf, not a shift: hinge (1.765, -0.400), the same +z axis, limits
[-3, 0] — it opens by DECREASING qpos, hence open_dir -1 and a pull target of
-DOOR_OPEN_TARGET. Bar 10.4 cm west of the right leaf's, by the same panel
arithmetic (2.1985). The closing ladder is CLOSE_DOCKS mirrored about the hinge
(hinge-relative x offsets +0.465/+0.465/+0.265/+0.665 -> 1.30/1.30/1.50/1.10),
i.e. WEST of the leaf, on the free floor in front of the sink. PROVISIONAL: the
mirror is a hypothesis over an asymmetric arm (rest pan/roll are not zero); W21
measures it, with the right leaf through the same code as the control."""

CAB_2_LEFT = DoorSpec(
    stem="cab_2_main_group", hinge="leftdoorhinge", open_dir=-1, panel_dir=1,
    handle_bar=(1.20, -0.430, 1.592), handle_dock=(1.20, -1.30),
    close_docks=((0.70, -1.3), (0.50, -1.3), (0.50, -1.1), (0.30, -1.1)),
    closed_rad=DOOR_CLOSED_RAD)
"""The west box's LEFT leaf — the search family's fourth compartment.

MEASURED (W20's articulation dump, 2026-09-02, Mac and container identical to
the digit): the hinge anchor (0.765, -0.400), axis +z, `hinge_anchor` sense +1,
limits [-3, 0] — it opens by DECREASING qpos, hence `open_dir` -1, `panel_dir`
+1 and a pull target of -DOOR_OPEN_TARGET, exactly as CAB_MAIN_LEFT.

ARITHMETIC, not scanned — the bar: `cab_2`'s box spans x [0.75, 1.75] (centre
1.25, the W20 census) and a top-row bar stands `handle_hpad` = 0.05 inside the
leaf's free edge (robocasa cabinet_panels.py: hpad 0.05, vpad 0.20). That is the
construction which reproduced the SCANNED `cab_main` R bar (2.302, -0.430,
1.592) at 2.3015 — 0.5 mm — so this leaf's bar is 1.25 - 0.05 = **1.20**. (The
panel formula's own figure, mirroring CAB_MAIN_LEFT's 2.1985 by the -1.000 m
`align_to` shift, is 1.1985; the two agree to 1.5 mm, an order inside the
+-0.030 m bar-offset band W21 measured as holding.) The dock stands under the
bar at the measured y = -1.30, as every other leaf's does.

MEASURED, and it took a sweep and a probe to get here — the closing ladder. The
mirror of CAB_MAIN_LEFT's rungs lands at x = 0.30/0.30/0.50/0.10 about this
hinge, and the first N=4 sweep measured what that costs: this leaf OPENED on 13
of 13 sighted rounds and CLOSED on **0**, every rung refusing the drive with
`gripper/wrist_flex/wrist_roll <-> wall_left_room` — the stowed hand reaches
0.55-0.70 m ahead of a base standing half a metre from the west wall. W25
(tools/probes/w25_cab2_left_close.py) then drove fourteen cells from the leaf's
own pull-end pose, two arm shapes (folded via TCP, and the family's stow) across
five docks: **every dock at x <= 0.6 refuses the drive itself** (the base stops
0.45-0.81 m short), and **(0.70, -1.3) arrives to 2-3 mm and closes the door to
0.117 with EITHER arm shape** — so the ladder leads with the measured winner and
keeps the mirrored rungs behind it for the wall-clock coin. The control in the
same run: CAB_MAIN_LEFT's own first rung (1.30, -1.1) closes to 0.118, K106's
number. The sink (x [0.79, 1.71], top 1.085, faucet ~z 1.22) sits under this
ladder and the push point rides above it at z = CLOSE_PUSH_Z = 1.55."""

CAB_1 = DoorSpec(
    stem="cab_1_main_group", hinge="doorhinge", open_dir=-1, panel_dir=1,
    handle_bar=(0.698, -0.435, 1.592), handle_dock=(0.698, -1.30),
    close_docks=(),
    closed_rad=DOOR_CLOSED_RAD)
"""The WEST wall unit — the search family's optional fifth compartment (W24).

MEASURED (W24 scan, 2026-09-02, tools/probes/w24_cab1_wall.py): hinge anchor
(0.265, -0.400), axis +z, sense +1, limits [-3, 0] — a left leaf, so `open_dir`
-1 and `panel_dir` +1, exactly as CAB_MAIN_LEFT. The joint is named `doorhinge`,
NOT `leftdoorhinge`: cab_1 carries a SINGLE door over the whole box. Its door
link is `hingedoor` (8 collision shapes), the exact mirror of `cab_main`'s right
leaf shifted by -1.604, and its bar was SCANNED at **(0.698, -0.435, 1.592)** —
cab_main's measured (2.302, -0.435, 1.592) minus 1.604. The scene's own bar
arithmetic (`_bar_x`: hpad inside the free edge of a box spanning x [0.25, 0.75])
gives 0.700, 2 mm away; the scan is what this spec carries. The y differs by
5 mm from the family's `handle_y = -0.430`, which is inside every standoff this
stage works to.

The dock stands under the bar at the measured y = -1.30, as every other leaf's
does; that is a DERIVATION, not a drive — no sweep has docked here.

NO CLOSING LADDER, and that is the point. CAB_MAIN_LEFT's rungs sit at
hinge-relative x offsets +0.465/+0.465/+0.265/+0.665, which about THIS hinge are
x = 0.73/0.73/0.53/0.93 — but the push side of an opened LEFT leaf is west of its
hinge at 0.265, and the room's west wall is at x = 0, so the fist's own standing
room is inside the wall. Hence `close_docks = ()`: `close_the_door` refuses this
leaf by name, and the task closes its round with the wall park instead
(`cabinet_search_base.Compartment.close_policy == "wall_park"`,
`open_the_door(park_rad=...)`).

One more thing this leaf needs that no other does: until the scene clears ignore
bit 26 on its door shapes, the door is INTANGIBLE to this robot and the pads pass
straight through the bar (W24; `CabinetSearchConfig.untangle_stems`). Every
number above was taken with that bit cleared."""


def door_from_cfg(task) -> DoorSpec:
    """The leaf a CabinetRetrieval-family config names, as a spec.

    `cfg.cabinet_name` / `cfg.door_hinge` / `cfg.door_closed_rad` plus this
    module's measured constants — exactly what the stages read before the spec
    existed, so `door=None` is today's behaviour.

    Args:
        task: `env.unwrapped`, with a `cfg` carrying the three fields.

    Returns:
        A `DoorSpec` for the right leaf (`open_dir` +1, `panel_dir` -1).

    Example:
        >>> door_from_cfg(task) == CAB_MAIN_RIGHT   # the Closed task's default cfg  # doctest: +SKIP
        True
    """
    cfg = task.cfg
    return DoorSpec(stem=str(cfg.cabinet_name), hinge=str(cfg.door_hinge),
                    open_dir=1, panel_dir=-1, handle_bar=HANDLE_BAR,
                    handle_dock=HANDLE_DOCK, close_docks=CLOSE_DOCKS,
                    closed_rad=float(cfg.door_closed_rad))


def _art_key(task, door: DoorSpec) -> str:
    """The scene key for the leaf's articulation (the stem plus a suffix)."""
    return [k for k in task.scene.articulations if door.stem in k][0]


def _hinge_qpos(task, door: DoorSpec) -> float:
    """The RAW hinge angle from the simulator (signed as the joint is)."""
    art = task.scene.articulations[_art_key(task, door)]
    names = [j.name for j in art.get_active_joints()]
    return float(_np(art.get_qpos()).reshape(-1)[names.index(door.hinge)])


def hand_vs_face(task, door: DoorSpec, anchor_xy, sense) -> tuple:
    """Where the hand stands relative to the leaf, in the LEAF's frame: TCP minus the
    live bar on the face's outward normal (+ = in front of the face), on the face's
    radial tangent from the hinge, and on z — plus the pads' span. The bar is
    `door.handle_bar` carried round the hinge by the live angle (finish_by_the_handle's
    formula). The instrument that answered g22 (the reversal slides ALONG the face).

    Args:
        task: `env.unwrapped`.
        door: the leaf's spec.
        anchor_xy: the hinge axis, world xy.
        sense: +1 when +qpos is CCW about world +z.

    Returns:
        `(d_n, d_t, d_z, span)`, metres, rounded to mm.

    Example:
        >>> hand_vs_face(task, CAB_MAIN_RIGHT, np.array([2.735, -0.4]), 1)  # doctest: +SKIP
        (0.035, 0.02, -0.011, 0.1)
    """
    anchor_xy = np.asarray(anchor_xy, dtype=np.float64).reshape(-1)[:2]
    bar0 = np.asarray(door.handle_bar, dtype=np.float64).reshape(-1)[:3]
    q_h = _hinge_qpos(task, door)
    phi = float(sense) * q_h
    c, s_ = float(np.cos(phi)), float(np.sin(phi))
    xy = bar0[:2] - anchor_xy
    bar = np.array([anchor_xy[0] + c * xy[0] - s_ * xy[1],
                    anchor_xy[1] + s_ * xy[0] + c * xy[1], bar0[2]])
    _, n = push_frame(anchor_xy, sense, q_h, door)
    t = bar[:2] - anchor_xy
    t = t / max(float(np.linalg.norm(t)), 1e-6)
    tcp = _np(task.agent.tcp.pose.sp.p).reshape(-1)[:3]
    d = tcp - bar
    q = _np(task.agent.robot.get_qpos()).reshape(-1)
    return (round(float(d[:2] @ n), 3), round(float(d[:2] @ t), 3),
            round(float(d[2]), 3), round(float(q[-1] + q[-2]), 3))


def push_frame(anchor_xy, sense, qpos: float, door: DoorSpec, r: float | None = None):
    """The fist's push point on the leaf and the leaf's outward face normal, world xy.

    At qpos 0 the leaf lies along world x from its hinge toward `panel_dir` with
    its outer face toward -y (both leaves of a wall cabinet face the room); the
    hinge turns it by `sense * qpos` (CCW positive). The push point is
    `CLOSE_R_PUSH` out along the leaf's mid-plane — W15's winner on the right leaf
    — and the normal is the rotated -y. For the right leaf (`panel_dir` -1,
    `open_dir` +1) this is the K106 formula verbatim; for the left leaf
    (`panel_dir` +1, `open_dir` -1) at the same opening angle it is that geometry's
    mirror image about the hinge's own x — the hypothesis W21 measures.

    Derivation, with phi = open_dir * qpos >= 0 the opening angle and R the
    rotation by sense * qpos = sense * open_dir * phi (sense +1 on both leaves):
        right:  R(+phi) @ (-1, 0) = (-cos phi, -sin phi);  n = R(+phi) @ (0, -1) = ( sin phi, -cos phi)
        left:   R(-phi) @ (+1, 0) = ( cos phi, -sin phi);  n = R(-phi) @ (0, -1) = (-sin phi, -cos phi)
    Each left vector is the right one with x negated: the leaf swings south
    either way and its outer face turns toward the hinge's side, which is where the
    closing ladder stands. The closing tangent therefore runs opposite to the
    opening one — `-(sense * open_dir)` for `follow_arc`: CW (-1) for the right leaf
    (today's `-sense`), CCW (+1) for the left.

    Args:
        anchor_xy: the hinge axis, world xy (`oracle_common.hinge_anchor`).
        sense: +1 when +qpos is CCW about world +z.
        qpos: the RAW hinge angle, not `door_rad_now`'s signed reading.
        door: the leaf's spec (`panel_dir`).
        r: how far out along the leaf's mid-plane the push point sits; None =
            `CLOSE_R_PUSH`, W15's winner, so every existing caller is unchanged.

    Returns:
        `(centre_xy, n_xy)`: push point and outward face normal, float64.

    Example:
        >>> c, n = push_frame(np.array([2.735, -0.4]), 1, 0.0, CAB_MAIN_RIGHT)
        >>> c.round(3).tolist(), n.round(3).tolist()
        ([2.385, -0.4], [0.0, -1.0])
    """
    a = sense * float(qpos)
    rot = np.array([[np.cos(a), -np.sin(a)],
                    [np.sin(a), np.cos(a)]])
    n = rot @ np.array([0.0, -1.0])
    centre = np.asarray(anchor_xy, dtype=np.float64) + (
        CLOSE_R_PUSH if r is None else float(r)) * (
        rot @ np.array([float(door.panel_dir), 0.0]))
    return centre, n


def say(env, stage: str, **extra) -> None:
    """`oracle_common.say` with this oracle's tag.

    Args:
        env: the (possibly wrapped) env.
        stage: one line, greppable, per stage.
        **extra: key=value pairs appended to the line.

    Example:
        >>> say(env, "dock at the cabinet", dock=[2.5, -1.05])   # doctest: +SKIP
    """
    _say(env, WHO, stage, **extra)


def fail(env, stage: str, **extra):
    """`oracle_common.fail` with this oracle's tag; returns -1.

    Args:
        env: the (possibly wrapped) env.
        stage: what refused, for the trace.
        **extra: diagnostics.

    Returns:
        -1, always — the sweep books it `no_plan`.

    Example:
        >>> return fail(env, "grasp the cup")                     # doctest: +SKIP
    """
    return _fail(env, WHO, stage, **extra)


def _np(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def _b(info, key) -> bool:
    return bool(_np(info[key]).reshape(-1)[0])


def side_grasp_pose(task, depth: float):
    """The side-grasp TCP pose for the cup where it stands NOW, plus its pre-grasp.

    Approach along +y (into the cabinet), closing across, TCP at the cup's
    mid-height `depth` metres past/short of the axis — the W13 winning geometry.
    The mesh is read fresh on every call: a refused close can move the cup.

    Args:
        task: `env.unwrapped`.
        depth: metres along the approach past the cup's axis (negative = short).

    Returns:
        `(grasp, pre)` sapien Poses, or `(None, None)` if the cup has no mesh.

    Example:
        >>> g, pre = side_grasp_pose(task, -0.005)                # doctest: +SKIP
    """
    mesh = task.cup.get_first_collision_mesh(to_world_frame=True)
    if mesh is None:
        return None, None
    b = np.asarray(mesh.bounds, dtype=np.float64)
    cup_p = _np(task.cup.pose.p).reshape(-1, 3)[0].astype(np.float64)
    centre = np.array([cup_p[0], cup_p[1] + float(depth), (b[0][2] + b[1][2]) / 2.0])
    grasp = task.agent.build_grasp_pose(
        np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]), centre
    )
    return grasp, grasp * sapien.Pose([0, 0, -0.12])


#: The torso height for the WHOLE episode, and it is one number on two measurements.
#: At the rest keyframe's 0.386 the upperarm rides the open panel's bottom edge
#: (1.391) and the dock approach was refused `collision upperarm_roll<->
#: hingerightdoor` on three of ten sweep seeds — start jitter decided which. Ducked
#: to 0.20 the whole arm passes under the panel. Raising it back AT the dock then
#: refused too — `collision forearm_roll<->cab object` on the line's knot 2/3: the
#: vertical ride lifts the forearm into the cabinet box. And the raise buys
#: nothing: the side grasp at shelf height plans from 0.20 (measured, this seed
#: set), so the torso ducks once and stays ducked.
TORSO_DRIVE = 0.20


def plan_joints(env, planner, task, targets: dict, *, label: str, tries: int = 2,
                line_only: bool = False):
    """Plan and execute a joint-space move, straight line first, RRT second.

    A trimmed copy of water_plants_planner.plan_to_joint_targets — that one
    carries WP-specific env knobs and importing it would couple two oracles' RNG
    phasing. The third copy (DepthRecall's) has since been promoted as
    `oracle_common.plan_joints`; this one stays so the Closed oracle's measured
    path is byte-identical (the promoted one says one more trace line on the RRT
    branch). Known and filed (K111): that RRT branch executes as a no-op.

    Args:
        env, planner: as everywhere.
        task: `env.unwrapped`.
        targets: `{joint_name: value}`; every other joint keeps its value.
        label: stage name for the trace.
        tries: RRT draws after a blocked line.

    Returns:
        The gym 5-tuple, or -1 after the last refused draw.

    Example:
        >>> res = plan_joints(env, planner, task, {"torso_lift_joint": 0.2},
        ...                   label="duck the torso")             # doctest: +SKIP
    """
    p = planner.planner
    robot = task.agent.robot
    for i in range(int(tries)):
        cur = _np(robot.get_qpos()).reshape(-1).astype(np.float64)
        goal = cur.copy()
        for n, v in targets.items():
            goal[robot.active_joints_map[n].active_index[0].item()] = float(v)
        # mplib's time parameterization can RAISE instead of returning a status:
        # `TOPP` -> RuntimeError("Fail to parameterize path"), measured by W25
        # (2026-09-02) folding the arm beside the west wall. A planner that cannot
        # time a path has refused this leg — that is a refusal, not an episode-
        # killing exception (a sweep would book `errored`, and the oracle's own
        # D6 contract would never get to speak).
        try:
            line = p.plan_qpos_line(goal, cur, time_step=task.control_timestep,
                                    ref_yaw=float(cur[2]))
        except RuntimeError as e:
            line = {"status": f"mplib raised: {e}", "position": []}
        if line["status"] == "Success" and len(np.asarray(line["position"])) > 1:
            say(env, f"{label}: joint line",
                knots=int(np.asarray(line["position"]).shape[0]))
            return planner.follow_forward_path_w_refinement(line, refine=True)
        if line["status"] != "Success":
            say(env, f"{label}: {line['status']}")
        if line_only:
            # The caller wants the one channel that measurably executes (K111: the
            # RRT branch ran 185 knots and moved the arm by nothing) or nothing.
            say(env, f"{label}: line only; no RRT draw")
            return -1
        try:
            result = p.plan_qpos(
                [p.fold_qpos(p.pad_move_group_qpos(goal))],
                p.fold_qpos(p.pad_move_group_qpos(cur)),
                time_step=task.control_timestep, planning_time=8.0, rrt_range=0.1,
                simplify=True, fixed_joint_indices=[0, 1, 2], ref_yaw=float(cur[2]),
            )
        except RuntimeError as e:
            result = {"status": f"mplib raised: {e}", "position": []}
        if result["status"] == "Success":
            return planner.follow_forward_path_w_refinement(result, refine=True)
        say(env, f"{label}: plan refused", status=result["status"], draw=i + 1)
    return -1


def door_rad_now(task, door: DoorSpec | None = None) -> float:
    """How far open the leaf stands, radians >= 0, read from the simulator.

    `open_dir * qpos`: for the measured right leaf (open_dir +1) exactly the raw
    hinge angle every K103-K108 number is in; for a left leaf (limits [-3, 0]) the
    negated qpos, so "how open" carries one sign for either. The same
    stem-resolution the task itself uses (the stem is exact; the scene key carries
    a suffix), so the oracle judges the opening by STATE — the pull's return code
    says nothing about how far the door got.

    Args:
        task: `env.unwrapped`.
        door: the leaf; None = the task's configured leaf (`door_from_cfg`).

    Returns:
        `open_dir * qpos` in radians.

    Example:
        >>> door_rad_now(task)                                    # doctest: +SKIP
        1.601
        >>> door_rad_now(task, CAB_MAIN_LEFT)   # qpos -1.6 reads 1.6  # doctest: +SKIP
        1.6
    """
    door = door_from_cfg(task) if door is None else door
    return door.open_dir * _hinge_qpos(task, door)


def stow_hand(env, planner, task, z: float = 1.10, aheads=(0.70, 0.55, 0.85)) -> None:
    """Pull the empty hand in over the base before the drive to the cup dock.

    A trimmed copy of same_drawer_planner.stow_arm — that one carries SameDrawer's
    REACH_N_INIT and importing it would couple two oracles' RNG phasing (the
    plan_joints precedent above; promotion into oracle_common once a third copy
    appears). Why here: after the pull the arm is stretched to the swung bar at
    ~1.6 m height next to the open panel; the drive to the cup dock passes under
    that panel (its bottom edge is at 1.391), so the hand comes down to 1.10 over
    the base first. Non-fatal: a refusal still tries the drive (refusals cost no
    steps).

    Args:
        env, planner: as everywhere.
        task: `env.unwrapped`.
        z: stow height — under the open panel's bottom edge.
        aheads: metres ahead of the base, tightest-that-plans wins.

    Example:
        >>> stow_hand(env, planner, task)                         # doctest: +SKIP
    """
    planner.close_gripper()
    base = _np(task.agent.base_link.pose.p).reshape(-1)[:3]
    q = _np(task.agent.base_link.pose.raw_pose).reshape(-1)[3:]
    yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
    face = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    tcp_q = _np(task.agent.tcp.pose.raw_pose).reshape(-1)[3:]
    for ahead in aheads:
        target = sapien.Pose(p=base + face * ahead + np.array([0.0, 0.0, z]), q=tcp_q)
        if planner.static_manipulation(target, disable_lift_joint=False) != -1:
            say(env, "hand stowed for the drive", ahead=ahead)
            return
    say(env, "stow refused; driving with the hand as it is")


def grasp_the_bar(env, planner, task, door: DoorSpec, *, bar, dock):
    """Dock, pre-grasp and stroke onto a leaf's bar. `(res, done)`.

    Extracted from `open_the_door` UNCHANGED, so a second caller gets the measured
    sequence itself rather than a copy of it. The second caller is the finisher: a leaf
    that is nearly shut has carried its bar round the hinge, so it needs these same three
    legs at a DIFFERENT bar position, and every comment below was paid for once already.

    `done=True` means the caller must return `res` immediately — the horizon ran out, or
    the stage refused and has already said so.

    Args:
        env, planner, task: as everywhere.
        door: the leaf, for its stem (the contact stroke's substring) and nothing else.
        bar: world xyz of the bar AS IT STANDS. `door.handle_bar` is the closed-leaf
            value and is valid at qpos 0 only.
        dock: base pose to pull from, as `[x, y, 0.0]` — `drive_base` wants three
            components and a bare xy pair raises `operands could not be broadcast`,
            which is exactly how the finisher's first run died.

    Returns:
        `(res, done)`.

    Example:
        >>> res, done = grasp_the_bar(env, planner, task, door,        # doctest: +SKIP
        ...                           bar=door.handle_bar, dock=door.handle_dock)
    """
    A = task.agent
    say(env, "drive to the handle dock", dock=[round(float(v), 3) for v in dock])
    res = planner.drive_base(target_pos=dock, target_view_vec=np.array([0.0, 1.0, 0.0]))
    if res != -1 and common.stopped_by_horizon(planner):
        return res, True
    if res == -1:
        return fail(env, "drive to the handle dock"), True
    planner.planner.update_from_simulation()

    # The closing axis is +x on BOTH leaves, measured (W21): the left bar holds
    # with the right leaf's own frame (fingers 0.026 on 8 of 8 cells of the
    # offset grid, the neighbour leaf never moving), while MIRRORING the axis
    # asks the wrist for a 180 deg flip the screw cannot take from a driven
    # posture (`joint limit at index [7]`, 5.158 rad of twist left). The bars of
    # one cabinet straddle the box centre 10.3 cm apart, so what killed the
    # search's first solo was not the axis but a WOUND wrist after the drive —
    # the caller normalizes the roll joints before this stage.
    grasp = A.build_grasp_pose(np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]),
                               np.array(bar))
    pre = grasp * sapien.Pose([0, 0, -0.15])
    # The PRE-grasp plans WITH the doors in the world: it has 15 cm of clearance
    # off the bar, and a door-free plan is licensed to route THROUGH the panel —
    # sweep 5, seed 9 measured exactly that: the executed pre-grasp swept the
    # door to 0.55 rad before the grip ever closed.
    res = common.arm_move(env, planner, pre, who=WHO, stage="pre-grasp the handle",
                          tries=3)
    if res != -1 and common.stopped_by_horizon(planner):
        return res, True
    if res == -1:
        return fail(env, "pre-grasp the handle"), True
    # K100 contact stroke on the SHORT grasp leg only: the bar stands 5 cm off
    # the panel and 5 cm from the LEFT door's edge, so the open fingers around
    # it collide with the doors in the planning world by construction — sweep 3
    # measured this leg refusing `finger<->hingeleftdoor` on 6 of 10 seeds (and
    # flipping between sweeps: wall-clock RRT). The fixture being grasped is the
    # goal, not an obstacle (same_drawer's bar reach does exactly this); and
    # K102 caps the unchecked contact: the touch IS the arrival — stop at first
    # gripper contact with the cabinet and close there. `gripper_touching`
    # matches entity names by substring; the stem names the whole assembly.
    touch = types.SimpleNamespace(name=door.stem)
    say(env, "grasp the handle", stop_on="first touch of the cabinet")
    with common.contact_stroke(planner, [door.stem]):
        res = -1
        for _ in range(3):
            res = planner.static_manipulation(grasp, stop_on_touch=touch)
            if res != -1:
                break
            say(env, "grasp the handle: plan refused, one more draw")
    planner.planner.update_from_simulation()
    if res != -1 and common.stopped_by_horizon(planner):
        return res, True
    if res == -1:
        return fail(env, "grasp the handle"), True
    return res, False


def open_the_door(env, planner, task, *, door: DoorSpec | None = None, anchor=None,
                  park_rad: float | None = None, record: dict | None = None,
                  fold: bool = True):
    """The closed-door variant's opening stage: grasp the bar, arc-pull it open.

    The K104 chain from the W14 stand, driven instead of teleported: dock at the
    handle, pre-grasp + grasp the CLOSED door's bar (the W12-scanned pose — the
    reason the ajar band is refused by validate()), close, `pull_hinge_arc` to
    `DOOR_OPEN_TARGET` in the leaf's opening sense, release, fold the arm for the
    drive. The caller judges the achieved angle by state (`door_rad_now`) — per
    D6 the pull returns a 5-tuple however far the door got, and -1 only before
    any step.

    Parametrized by `door` (K113): the dock, the bar, the stem the contact stroke
    and the touch-stop name, the hinge and the pull's sign all come from the spec;
    `door=None` builds it from the task's config — the Closed oracle's path,
    byte-identical.

    Args:
        env: the (possibly wrapped) env.
        planner: the solver (or the stub).
        task: `env.unwrapped`.
        door: the leaf; None = `door_from_cfg(task)`.
        anchor: optional `(anchor_xy, sense)` handed to `pull_hinge_arc`, so an
            offline stub can stage the whole opening on a fake task.
        park_rad: None (default) ends the pull at `DOOR_OPEN_TARGET` — the
            K104-K108 path, byte-identical. A float continues the SAME pull, on
            the SAME grip, to that larger angle before the hand lets go: the W24
            wall park for cab_1, whose leaf cannot be pushed shut and is instead
            driven against the room's west wall (`close_docks = ()`,
            `Compartment.close_policy == "wall_park"`). It is a second
            `pull_hinge_arc` call rather than a bigger first one because the
            caller reads its reveal verdict off the first target and because a
            pull that stalls at 1.6 should still be reported as an opening that
            failed to park, not as an opening that never happened. Post-commit:
            the fist is already on the bar, so a refused park is SAID and the
            opening's tuple comes back — the task judges the park by state
            (`door_rad_now >= cfg.wall_park_rad`), as it judges everything else.
            PROVISIONAL: no sweep has driven the arc past 1.75 rad.

    Returns:
        -1 (a `fail(...)` said why), or the last gym 5-tuple.

    Example:
        >>> res = open_the_door(env, planner, task)               # doctest: +SKIP
        >>> if res == -1: return res                              # doctest: +SKIP
        >>> res = open_the_door(env, planner, task, door=CAB_2_RIGHT)  # doctest: +SKIP
        >>> res = open_the_door(env, planner, task, door=CAB_1, park_rad=2.0)  # doctest: +SKIP
    """
    door = door_from_cfg(task) if door is None else door
    A = task.agent
    dock = np.array([door.handle_dock[0], door.handle_dock[1], 0.0])
    res, done = grasp_the_bar(env, planner, task, door,
                              bar=door.handle_bar, dock=dock)
    if done:
        return res
    res = planner.close_gripper(t=12)
    if res != -1 and common.stopped_by_horizon(planner):
        return res

    # The empty-hand gate is pull_hinge_arc's own (FINGER_EMPTY_M — the W12 trap).
    # The target is SIGNED by the leaf: +1.75 raises a right leaf's qpos, -1.75
    # lowers a left leaf's; the primitive compares in the opening sense.
    res = common.pull_hinge_arc(env, planner, task, _art_key(task, door), door.hinge,
                                target_rad=door.open_dir * DOOR_OPEN_TARGET, who=WHO,
                                v_handle=PULL_V_HANDLE,
                                max_steps=PULL_MAX_STEPS, anchor=anchor,
                                open_dir=door.open_dir)
    if res == -1:
        return res  # pull_hinge_arc already said why
    if common.stopped_by_horizon(planner):
        return res

    # THE WALL PARK (W24), on the same grip, before the hand lets go — the only
    # place it can happen: `pull_hinge_arc` refuses an empty hand (FINGER_EMPTY_M)
    # and the closed-door bar this stage grasped has swung metres away by now, so
    # a park attempted after the release would have to re-find a bar nothing has
    # measured. The leaf is already at ~1.7; this rides the same arc on to
    # `park_rad` — reachable, because the untangled leaf walks freely to exactly
    # 2.0 rad, with the west wall refusing it from 2.10 on as the backstop (host
    # re-measurement 2026-09-02, `CabinetSearchConfig.wall_park_rad`).
    if park_rad is not None:
        say(env, "park the door at the wall", park_rad=round(float(park_rad), 3),
            door_rad=round(door_rad_now(task, door), 3))
        parked = common.pull_hinge_arc(
            env, planner, task, _art_key(task, door), door.hinge,
            target_rad=door.open_dir * float(park_rad), who=WHO,
            v_handle=PULL_V_HANDLE, max_steps=PULL_MAX_STEPS, anchor=anchor,
            open_dir=door.open_dir)
        if parked == -1:
            # Post-commit: the door IS open, the caller judges the park by state.
            say(env, "the wall park refused; verdict by state",
                door_rad=round(door_rad_now(task, door), 3))
        else:
            res = parked
            if common.stopped_by_horizon(planner):
                return res
        say(env, "door at the wall", door_rad=round(door_rad_now(task, door), 3))

    if record is not None:
        # Captured HERE, with the fingers still on the bar and before the release:
        # this is the one instant at which the arm demonstrably holds this leaf at
        # this angle from this base pose, and `retrace_the_door` has nothing else to
        # stand on. Pure reads; nothing below depends on it. `tcp_q` is the field
        # that matters most and the easiest to omit: the pull never re-orients the
        # hand, so the recorded orientation is the only one known to work.
        _pose = task.agent.base_link.pose.sp
        _bx = _np(_pose.to_transformation_matrix()).reshape(-1, 4, 4)[0][:3, 0]
        _tcp = task.agent.tcp.pose.sp
        record.update(
            stem=door.stem, hinge=door.hinge,
            base_xy=_np(_pose.p).reshape(-1)[:2].astype(np.float64).copy(),
            base_yaw=float(np.arctan2(float(_bx[1]), float(_bx[0]))),
            hinge_qpos=_hinge_qpos(task, door),
            door_rad=door_rad_now(task, door),
            anchor=(anchor if anchor is not None
                    else common.hinge_anchor(task, _art_key(task, door), door.hinge)),
            tcp_p=np.asarray(_tcp.p, dtype=np.float64).copy(),
            tcp_q=np.asarray(_tcp.q, dtype=np.float64).copy(),
            arm_qpos=_np(task.agent.robot.get_qpos()).reshape(-1).astype(np.float64).copy(),
        )
        # From here the window that the retrace will UNDO: release, retreat, fold. The
        # base does not move in it, so the tape is arm-and-gripper only and replaying it
        # backwards is the exact reverse of all of them at once — the fingers re-close
        # precisely where they opened, with no moment for anyone to choose.
        record["tape"] = planner.start_tape()
        # One held step BEFORE the release, so the tape's first entry carries the hand
        # on the bar with the fingers SHUT. Without it the earliest thing recorded is
        # already the opening, and a reversal would end with the pads open — the one
        # state the retrace must not finish in. One step buys the anchor.
        planner.idle_steps(t=1)
        say(env, "recorded the grip for a retrace",
            door_rad=round(float(record["door_rad"]), 3),
            base=[round(float(v), 3) for v in record["base_xy"]])
    planner.open_gripper()
    planner.planner.update_from_simulation()
    if record is not None and RETRACE_ARC and RETREAT_LIFT_M > 0:
        # Off the face first, along the LIVE normal (RETREAT_LIFT_M says why); taped, so
        # the reversal undoes it last, with the hand already at the bar's radius.
        a_xy, a_sense = record["anchor"]
        _, n_xy = push_frame(np.asarray(a_xy, dtype=np.float64).reshape(-1)[:2],
                             int(a_sense), _hinge_qpos(task, door), door)
        tcp = task.agent.tcp.pose.sp
        lift_to = sapien.Pose(p=np.asarray(tcp.p, dtype=np.float64).reshape(-1)[:3]
                              + RETREAT_LIFT_M * np.array([n_xy[0], n_xy[1], 0.0]),
                              q=tcp.q)
        say(env, "lift the open hand off the face", m=RETREAT_LIFT_M,
            normal=[round(float(v), 3) for v in n_xy])
        # The open pads still touch the bar, so the planning world sees the start in
        # collision with the leaf and refuses every draw (g23, 4/4 leaves). The leaf
        # is what the hand is leaving, not an obstacle: out of the world for this leg,
        # as every contact stroke at it is (contact_stroke, K100).
        before = hand_vs_face(task, door, a_xy, a_sense)
        rad_before = float(door_rad_now(task, door))
        with common.contact_stroke(planner, [door.stem]):
            lifted = common.arm_move(env, planner, lift_to, who=WHO,
                                     stage="lift the open hand off the face", tries=2)
        if lifted != -1:
            res = lifted
            if common.stopped_by_horizon(planner):
                return res
            planner.planner.update_from_simulation()
        after = hand_vs_face(task, door, a_xy, a_sense)
        say(env, "lifted", hand_before=before, hand_after=after,
            d_n=round(after[0] - before[0], 3), d_t=round(after[1] - before[1], 3),
            leaf_moved=round(float(door_rad_now(task, door)) - rad_before, 4))
    # Retreat off the bar BEFORE the stow: stow_hand opens with a fist, and a
    # fist closed back onto the bar would drag the door shut during the next
    # drive — after the caller's door_rad_now verdict, where nothing re-checks
    # it (same_drawer's close_drawer retreats by BAR_STANDOFF for this exact
    # reason). Non-fatal: a refused retreat still tries the stow.
    tcp = task.agent.tcp.pose.sp
    back_to = sapien.Pose(p=tcp.p, q=tcp.q) * sapien.Pose([0, 0, -0.15])
    if record is not None and RETREAT_ALONG_FACE and not RETRACE_ARC:
        # Along the leaf's plane, 0.15 m: the hand's axis projected onto it, so the
        # inner pad slides beside the face instead of pressing it (RETREAT_ALONG_FACE).
        a_xy, a_sense = record["anchor"]
        _, n_xy = push_frame(np.asarray(a_xy, dtype=np.float64).reshape(-1)[:2],
                             int(a_sense), _hinge_qpos(task, door), door)
        n3 = np.array([float(n_xy[0]), float(n_xy[1]), 0.0])
        hz = _np(tcp.to_transformation_matrix()).reshape(4, 4)[:3, 2]
        away = -hz - float(np.dot(-hz, n3)) * n3          # the retreat's own sense, in-plane
        norm = float(np.linalg.norm(away))
        if norm > 0.3:                                    # the axis is not the normal itself
            away = away / norm
            back_to = sapien.Pose(p=np.asarray(tcp.p, dtype=np.float64).reshape(-1)[:3]
                                  + 0.15 * away, q=tcp.q)
            say(env, "retreat along the leaf's plane", away=[round(float(v), 3) for v in away],
                off_normal=round(float(np.dot(-hz, n3)), 3))
    back = common.arm_move(env, planner, back_to,
                           who=WHO, stage="retreat off the bar", tries=2)
    if back != -1:
        res = back
        if common.stopped_by_horizon(planner):
            return res
        planner.planner.update_from_simulation()
    # Fold the arm back to the REST arm pose (wrist to 1.7, off its stop) — the
    # posture every v0 number was measured from. The first sweep tried the
    # same_drawer forward stow here and it was the wrong shape for this dock:
    # a fist 0.7 m ahead at z 1.10 juts over the counter at y=-1.05 and the
    # dock screw died on `upperarm_roll<->stove/counter` (seed 0). Non-fatal:
    # a refused fold still tries the drive, stow_hand as the fallback.
    if fold:
        say(env, "fold the arm to rest for the drive")
        res_fold = fold_arm_to_rest(env, planner, task)
        if res_fold != -1:
            res = res_fold
            if common.stopped_by_horizon(planner):
                return res
        else:
            stow_hand(env, planner, task)
    else:
        # No drive follows (the look is a turn), so there is nothing to fold for — and
        # the fold is what the reversal has to UNFOLD, which is the leg where the finger
        # was measured striking the neighbouring leaf (keep_hit at 62-68 % of the
        # replay). Left out, the tape is release + retreat and its reverse is approach
        # + close, with nothing between the hand and the bar but the 0.15 m it backed off.
        say(env, "arm left out: no drive follows")
    if record is not None:
        planner.stop_tape()
        # Where the forward window LEFT the base. The base's lateral joint has no
        # position loop at all, so the same standing force that creeps the base during
        # the reversal crept it during the recording too — and this is that creep,
        # measured for free. If the two match, the reversal can start mirrored about the
        # recorded pose and arrive at it.
        _end = task.agent.base_link.pose.sp
        _ex = _np(_end.to_transformation_matrix()).reshape(-1, 4, 4)[0][:3, 0]
        record["base_xy_end"] = _np(_end.p).reshape(-1)[:2].astype(np.float64).copy()
        record["base_yaw_end"] = float(np.arctan2(float(_ex[1]), float(_ex[0])))
    return res


def fold_arm_to_rest(env, planner, task, *, label: str = "fold to rest"):
    """Fold the arm to the REST pose (wrist at 1.7, off its stop) for a drive.

    The posture every v0 number was measured from, and the only arm shape that
    survives base rotations deep in the east alcove: the stowed fist (0.7 m
    ahead at z 1.10) is a boom whose rotate sweeps hit
    `elbow_flex<->stove/counter` on every W15 closing dock (cycle diag 7).
    Returns the executor result, or -1 when every draw refused (non-fatal at
    every call site — the caller keeps whatever posture it has).

    Example:
        >>> fold_arm_to_rest(env, planner, task)                  # doctest: +SKIP
    """
    rest_q = np.asarray(task.agent.keyframes["rest"].qpos, dtype=np.float64).reshape(-1)
    jm = task.agent.robot.active_joints_map
    arm_names = list(task.agent.controller.controllers["arm"].config.joint_names)
    targets = {n: float(rest_q[jm[n].active_index[0].item()]) for n in arm_names}
    targets["wrist_flex_joint"] = 1.7
    return plan_joints(env, planner, task, targets, label=label, tries=3)


def _turn_cost(heading, direction) -> float:
    """Absolute in-place turn, radians, from `heading` to `direction` in the xy plane."""
    h = np.asarray(heading, dtype=np.float64).reshape(-1)[:2]
    d = np.asarray(direction, dtype=np.float64).reshape(-1)[:2]
    nh, nd = float(np.linalg.norm(h)), float(np.linalg.norm(d))
    if nh < 1e-9 or nd < 1e-9:
        return 0.0
    h, d = h / nh, d / nd
    return abs(float(np.arctan2(h[0] * d[1] - h[1] * d[0], float(np.dot(h, d)))))


def _leaf_joints(task, door: DoorSpec) -> np.ndarray:
    """Every joint angle of the leaf's articulation — both leaves of a two-door box.

    The retrace's base legs are UNPLANNED, which is the whole point (the planned ones
    are what refuse), so the cabinet is watched by state instead of by a collision
    sweep. Watching the whole articulation and not just `door.hinge` is deliberate:
    nudging the SIBLING leaf past the task's `theta_count` latches it as opened and
    loses the episode, and the two bars of one box straddle the centre 10.3 cm apart.
    """
    art = task.scene.articulations[_art_key(task, door)]
    return _np(art.get_qpos()).reshape(-1).astype(np.float64)


def retrace_the_door(env, planner, task, *, door: DoorSpec, record: dict, res,
                     anchor=None):
    """Close the leaf by undoing what the robot did after it opened it.

    The round's own actions after the pull are: release, retreat 0.15 m off the bar,
    fold to rest, drive to the look dock, idle. This runs them backwards — drive back
    to the pull-end pose, take the bar again, ride the opening arc in reverse — and
    then ends the way the env needs it ended: hand off the bar, arm folded, leaf still.

    Why this can work where the fist path does not. The fist path has to reach a NEW
    Cartesian pose from a NEW dock, and the measurement says no IK solution exists for
    it (12 of 22 refusals `IK Failed`, and a grid of four alternative push points never
    planned once in 189 tries). The retrace asks for a pose the arm held seconds ago,
    from a base pose it stood in seconds ago, with the leaf within a quarter radian of
    where it was. Reachability is not argued; it is remembered.

    Every verdict here is a state read — the hinge angles, the finger aperture, the
    env's own latch — with ONE deliberate exception: a refused pre-grasp returns -1 and
    hands over BEFORE the contact stroke, because `contact_stroke` removes the whole
    cabinet from the planning world and the return code is the only signal that exists
    before that commit.

    Args:
        env, planner, task: as everywhere.
        door: the leaf to close.
        record: what `open_the_door(record=...)` captured with the fingers on the bar.
        res: the last 5-tuple the caller holds; returned when nothing here steps.
        anchor: optional `(anchor_xy, sense)`; None takes the recorded one.

    Returns:
        `(res, closed)`. `closed` is the env's own verdict, never the angle alone.
        A False hands the caller straight to the measured fist-and-ladder path, from a
        defined pose: hand open, arm folded to rest.

    Example:
        >>> res, closed = retrace_the_door(env, planner, task, door=door, record=rec)  # doctest: +SKIP
    """
    def give_up(reason, **extra):
        say(env, "retrace: handing over to the ladder", reason=reason, **extra)
        return res, False

    # -- R0: the gates, all pure reads --------------------------------------------
    need = ("base_xy", "base_yaw", "hinge_qpos", "door_rad", "anchor", "tcp_p", "tcp_q")
    if not record or any(k not in record for k in need):
        return give_up("no record")
    anchor = record["anchor"] if anchor is None else anchor
    anchor_xy = np.asarray(anchor[0], dtype=np.float64).reshape(-1)[:2]
    sense = int(anchor[1])
    rad_live = door_rad_now(task, door)
    if rad_live < ARM_PASS_RAD:
        return give_up("the leaf is not open enough to retrace",
                       door_rad=round(float(rad_live), 3))
    drift = float(rad_live) - float(record["door_rad"])
    if abs(drift) > RETRACE_DRIFT_MAX_RAD:
        return give_up("the leaf drifted too far from the grip",
                       drift=round(drift, 3), cap=RETRACE_DRIFT_MAX_RAD)
    spent = float(_np(getattr(task, "elapsed_steps", 0)).reshape(-1)[0])
    horizon = float(getattr(task.cfg, "horizon", 0) or 0)
    if horizon and spent > RETRACE_HORIZON_FRAC * horizon:
        # A round that retraces AND hands over pays for both paths.
        return give_up("no step budget", spent=int(spent), horizon=int(horizon))

    say(env, "retrace the opening", door_rad=round(float(rad_live), 3),
        drift=round(drift, 3), back_to=[round(float(v), 3) for v in record["base_xy"]])

    # -- R1: drive back to the pose the pull ended in, unplanned -------------------
    # `turn_in_place` and `drive_straight` build the action array and step; neither
    # calls mplib, so neither can be refused by the rotation sweep that names
    # `forearm_roll_link<->hingerightdoor`. That sweep is what the fist path dies on.
    leaf0 = _leaf_joints(task, door)

    def touched() -> bool:
        now = _leaf_joints(task, door)
        n = min(len(now), len(leaf0))
        return bool(np.any(np.abs(now[:n] - leaf0[:n]) > RETRACE_TOUCH_RAD))

    # The ONE motion the reversal cannot undo, because it is not ours: after the fingers
    # let go, the leaf settles. Measured at the gate above it is -0.017 to -0.032 rad,
    # and at the bar's radius that is 8 to 13 mm — moved along the arc, whose tangent at
    # a bar mounted on the leaf's face IS the gripper's approach, the one axis with no
    # allowance. It is also why a MORE exact return made things worse: with the base
    # dead on the recorded pose the hand follows the recorded path exactly, and the leaf
    # has swung into it, so the hand strikes the panel and shoves the base sideways.
    # The answer needs no IK and no new plan. The arm comes back to the same joints, so
    # carrying the whole stance around the hinge by the angle the leaf turned carries
    # the hand exactly as the bar went.
    goal_xy, goal_yaw, turned = _stance_carried_with_the_leaf(task, door, record)
    face = np.array([float(np.cos(goal_yaw)), float(np.sin(goal_yaw)), 0.0])

    def aim_again():
        """Re-read the hinge and re-aim. The leaf does not hold still while we drive.

        Measured on seed 1 (2026-09-04): the settle at the gate was -0.032 rad and by the
        time the hand reached the bar the leaf stood at +0.036 from the record — about
        four degrees of travel between taking the measurement and needing it. A carry
        computed once, before a leg that costs a hundred-odd steps, is stale exactly when
        it matters. Re-read per cycle and the last cycle is aimed at the leaf as it is."""
        nonlocal goal_xy, goal_yaw, turned, face
        goal_xy, goal_yaw, turned = _stance_carried_with_the_leaf(task, door, record)
        face = np.array([float(np.cos(goal_yaw)), float(np.sin(goal_yaw)), 0.0])

    def square():
        """Turn to the heading the opening was done from.

        Returns "" when it may go on, "touch" when the leaf moved under it, "horizon"
        when the episode ran out — the caller answers each differently."""
        r = planner.turn_in_place(face, tol=RETRACE_BASE_TOL, stop_when=touched)
        if r != -1:
            out[0] = r
        if touched():
            return "touch"
        return "horizon" if common.stopped_by_horizon(planner) else ""

    out = [res]
    if not RETRACE_ARC and RETRACE_BASE_SKIP_M > 0.0:
        here = _np(task.agent.base_link.pose.sp.p).reshape(-1)[:2].astype(np.float64)
        bx = _np(task.agent.base_link.pose.sp.to_transformation_matrix()
                 ).reshape(-1, 4, 4)[0][:3, 0]
        yaw_here = float(np.arctan2(bx[1], bx[0]))
        d_skip = float(np.linalg.norm(goal_xy - here))
        dyaw_skip = float(abs((yaw_here - goal_yaw + np.pi) % (2 * np.pi) - np.pi))
        if d_skip <= RETRACE_BASE_SKIP_M and dyaw_skip <= RETRACE_BASE_SKIP_RAD:
            say(env, "retrace: the base stands at the opening pose; no base leg",
                d_dock=round(d_skip, 3), dyaw_deg=round(float(np.degrees(dyaw_skip)), 1))
            return give_up("arc disabled; the base leg is not needed")
    # AIM, DRIVE, SQUARE — and measure only between whole cycles. The base's rotation
    # axis is the root_z joint and `base_link` hangs off it, so a turn SLIDES base_link
    # along a circle about that axis. One aim-drive-square is therefore not a
    # translation, and the drive was aimed at a pose the closing turn then moved: that
    # is where the 12-15 mm came from, every round, whatever the tolerances were. But a
    # cycle that ENDS on the recorded heading leaves the next measurement standing on
    # it, so from the second cycle on each one IS a pure translation and they converge.
    # The first cycle is the coarse leg back from the viewing dock and squares up at its
    # end, which is what the leg always did — no turn is added to the shipped path.
    for cycle in range(RETRACE_BASE_TRIES):
        aim_again()
        here = _np(task.agent.base_link.pose.sp.p).reshape(-1)[:2].astype(np.float64)
        d = goal_xy - here
        dist = float(np.linalg.norm(d))
        if dist <= RETRACE_BASE_XY_TOL:
            if cycle:            # already squared by the cycle before
                break
            stopped = square()   # in place, but not yet facing the way it opened from
            res = out[0]
            if stopped == "touch":
                return give_up("the base leg touched the cabinet", leg="square")
            if stopped:
                return res, False
            break
        bx = _np(task.agent.base_link.pose.sp.to_transformation_matrix()
                 ).reshape(-1, 4, 4)[0][:3, 0]
        heading = np.array([float(bx[0]), float(bx[1]), 0.0])
        aim = np.array([float(d[0]), float(d[1]), 0.0])
        # Forward or reverse, by whichever needs the smaller in-place turn. Reverse
        # costs nothing here: `drive_straight` takes the sign on the distance.
        back = _turn_cost(heading, aim) > _turn_cost(heading, -aim)
        r = planner.turn_in_place(-aim if back else aim, tol=RETRACE_BASE_TOL,
                                  stop_when=touched)
        if r != -1:
            res = r
        if touched():
            return give_up("the base leg touched the cabinet", leg="turn")
        if common.stopped_by_horizon(planner):
            return res, False
        # A short leg is driven slowly: `drive_straight` stops on `travelled >= dist`,
        # so its overshoot is one step, and one step is 5 mm at 0.10 m/s against 1 mm
        # at 0.02. The coarse leg back from the viewing dock keeps the fast speed.
        r = planner.drive_straight(-dist if back else dist, stop_when=touched,
                                   v=0.10 if dist > RETRACE_BASE_FAST_M else 0.02)
        if r != -1:
            res = r
        if touched():
            return give_up("the base leg touched the cabinet", leg="drive")
        if common.stopped_by_horizon(planner):
            return res, False
        out[0] = res
        stopped = square()
        res = out[0]
        if stopped == "touch":
            return give_up("the base leg touched the cabinet", leg="square")
        if stopped:
            return res, False
    planner.planner.update_from_simulation()
    d_dock, dyaw = common.dock_error(task, (float(goal_xy[0]), float(goal_xy[1]),
                                            float(goal_yaw)))
    say(env, "retrace: back at the opening pose", d_dock=round(d_dock, 3),
        dyaw_deg=round(dyaw, 1), carried=round(float(np.degrees(turned)), 2))
    if not RETRACE_ARC:
        # The base leg alone. Measured question, not caution: see RETRACE_ARC.
        return give_up("arc disabled; the base leg is the whole stage")

    # -- R2: take the bar again, by UNDOING the letting go ---------------------------
    # THE REVERSAL. Not a plan back to a remembered pose: the recorded window played
    # backwards, every channel at once. In `pd_joint_pos` the arm and body slots are
    # position targets, so reversing their order retraces the path the hand took; the
    # gripper slot carries its own state, so the fingers re-close exactly where they
    # opened. Nothing is timed by hand and nothing is re-planned, which also means
    # nothing can refuse — there is no IK branch to re-pick and no sweep to hallucinate.
    tape = record.get("tape")
    if not tape:
        return give_up("no tape of the opening window")
    say(env, "retrace: playing the opening window backwards", steps=len(tape))
    # SIGNED, against the RECORD. A strike pushes the leaf further OPEN than it was
    # gripped (+0.35 on cab_2). Everything benign runs the other way or stops at the
    # record: the leaf settles toward closed after the release (-0.03..-0.08, on some
    # leaves more), and the closing pads pull the bar back up to the grip and no
    # further. Measured both wrong ways first (g18: from the replay's first frame, the
    # pull-back read as a shove; g19: unsigned from the record, the settle alone read
    # as one and stopped the replay 16 cm short on every cab_main leaf). The sibling
    # leaf has no grasp on it and no business moving: unsigned from the start.
    rad_ref = float(record["door_rad"])
    sib_ref = _leaf_joints(task, door)
    _art = task.scene.articulations[_art_key(task, door)]
    _names = [j.name for j in _art.get_active_joints()]
    _hi = _names.index(door.hinge) if door.hinge in _names else -1

    # INSTRUMENT (2026-09-05): where the hand is relative to the leaf's face on every
    # replayed step — TCP minus the LIVE bar, resolved on the face's outward normal
    # (+ = in front of the face), the face's radial tangent from the hinge, and z —
    # plus the pads' span. The strike question (g20 landed, g21 struck, same seed,
    # same base pose to 3 mm) is answered by this tail, not by the base numbers.
    replay_tail: list = []

    def _hand_vs_face():
        return hand_vs_face(task, door, anchor_xy, sense)

    def shoved() -> bool:
        """This leaf RETRACE_SHOVE_RAD more open than the record, or its sibling moved
        that much either way. Polled after every replayed step."""
        replay_tail.append(_hand_vs_face())
        if door_rad_now(task, door) - rad_ref > RETRACE_SHOVE_RAD:
            return True
        now = _leaf_joints(task, door)
        n = min(len(now), len(sib_ref))
        d = np.abs(now[:n] - sib_ref[:n])
        if 0 <= _hi < n:
            d[_hi] = 0.0
        return bool(np.any(d > RETRACE_SHOVE_RAD))

    r = planner.replay_tape(tape, reverse=True, freeze_base=True, keep_base=True,
                            stretch=RETRACE_STRETCH, stop_when=shoved,
                            hold=RETRACE_HOLD_STEPS, hold_tol=RETRACE_HOLD_TOL)
    planner.planner.update_from_simulation()
    if r != -1:
        res = r
        if common.stopped_by_horizon(planner):
            return res, False
    if shoved():
        # The hand is ON the leaf. Parking from here drags it through the leaf (g21/g22:
        # the guard stopped at +0.05 and the park left the leaf at +0.15..+0.33), so first
        # back out exactly the way it came in — the frames just stepped, played forward —
        # then let go and hand over from the parked posture, with the leaf a few
        # degrees from where the opening left it.
        frames = int(getattr(planner, "replay_frames", 0))
        if frames > 0:
            say(env, "retrace: the hand backs out the way it came", frames=frames)
            planner.replay_tape(tape[-frames:], reverse=False, freeze_base=True,
                                keep_base=True, hold=10, hold_tol=RETRACE_HOLD_TOL)
            planner.planner.update_from_simulation()
        tcp_now = _np(task.agent.tcp.pose.sp.p).reshape(-1)[:3]
        d_tcp = float(np.linalg.norm(tcp_now - np.asarray(record["tcp_p"], dtype=np.float64)))
        planner.open_gripper()
        _park_arm(env, planner, task)
        return give_up("the reversal shoved the leaf", d_tcp=round(d_tcp, 4),
                       **_miss_in_the_grasp_frame(task, record, tcp_now),
                       **_landed(planner, task, record, door),
                       hand_vs_face_tail=replay_tail[::2][:20])
    # How well the reversal actually landed, in the one number that decides it: the
    # gripper spans 100 mm on a 26 mm bar, so the bar has to end within about 37 mm of
    # the pads' centreline. Said on every retrace, held or not, because "the pads closed
    # on nothing" alone cannot tell a near miss from a wild one.
    tcp_now = _np(task.agent.tcp.pose.sp.p).reshape(-1)[:3]
    d_tcp = float(np.linalg.norm(tcp_now - np.asarray(record["tcp_p"], dtype=np.float64)))
    miss = _miss_in_the_grasp_frame(task, record, tcp_now)
    ap = _settled_aperture(planner, task)
    if not (RETRACE_GRIP_MIN < ap <= RETRACE_GRIP_MAX):
        planner.open_gripper()
        _park_arm(env, planner, task)
        return give_up("the pads closed on nothing", aperture=round(ap, 4),
                       touching=planner.touching_now(),
                       d_tcp=round(d_tcp, 4), **miss, **_landed(planner, task, record, door),
                       hand_vs_face_tail=replay_tail[::2][:20])
    say(env, "retrace: the bar is held", aperture=round(ap, 4), d_tcp=round(d_tcp, 4),
        **miss, **_landed(planner, task, record, door),
        hand_vs_face_tail=replay_tail[::2][:20])

    # -- R3: ride the opening arc backwards ----------------------------------------
    # `door_rad_now` is `open_dir * qpos`, so the raw target for "how open = 0.12" is
    # `open_dir * 0.12`; `open_dir=-door.open_dir` points the primitive's own progress
    # test the other way. This is the same arc the opening rode, in reverse.
    stop_at = float(door.closed_rad) - CLOSE_MARGIN_RAD
    say(env, "retrace: pull the door shut", target_rad=round(stop_at, 3),
        start_rad=round(float(door_rad_now(task, door)), 3))
    r = common.pull_hinge_arc(env, planner, task, _art_key(task, door), door.hinge,
                              target_rad=door.open_dir * stop_at, who=WHO,
                              v_handle=PULL_V_HANDLE, max_steps=PULL_MAX_STEPS,
                              anchor=(anchor_xy, sense), open_dir=-door.open_dir)
    if r != -1:
        res = r
        if common.stopped_by_horizon(planner):
            return res, False
    rad_after = float(door_rad_now(task, door))
    say(env, "retrace: arc done", door_rad=round(rad_after, 3))
    if rad_after > float(door.closed_rad):
        # Stalled short. Hand the LADDER back the angle it was measured at: the push
        # point recedes from every rung as the leaf closes, so a half-shut leaf is a
        # harder start than the open one, not an easier one. The grip is still live,
        # which is the only moment this can be undone.
        if RETRACE_UNDO_ON_STALL:
            say(env, "retrace: the arc stalled; riding the leaf back open",
                door_rad=round(rad_after, 3))
            common.pull_hinge_arc(env, planner, task, _art_key(task, door), door.hinge,
                                  target_rad=door.open_dir * DOOR_OPEN_TARGET, who=WHO,
                                  v_handle=PULL_V_HANDLE, max_steps=PULL_MAX_STEPS,
                                  anchor=(anchor_xy, sense), open_dir=door.open_dir)
        planner.open_gripper()
        _park_arm(env, planner, task)
        return give_up("the arc stalled short",
                       door_rad=round(float(door_rad_now(task, door)), 3))

    # -- R4: end it the way the env credits it -------------------------------------
    # The env's rule is a conjunction, and the two terms the arc leaves false are the
    # hand being far from the bar and the hinge being still: the arc's last commanded
    # step leaves the leaf turning about ten times the stillness threshold.
    planner.open_gripper()
    tcp = task.agent.tcp.pose.sp
    r = common.arm_move(env, planner,
                        sapien.Pose(p=tcp.p, q=tcp.q) * sapien.Pose([0, 0, -0.15]),
                        who=WHO, stage="retrace: retreat off the bar", tries=2)
    if r != -1:
        res = r
        if common.stopped_by_horizon(planner):
            return res, False
    _park_arm(env, planner, task)
    closed = False
    for _ in range(RETRACE_SETTLE_TRIES):
        r = planner.idle_steps(t=RETRACE_SETTLE_STEPS)
        if r != -1:
            res = r
        info = res[-1] if isinstance(res, tuple) else {}
        if isinstance(info, dict) and "any_open" in info:
            closed = not _b(info, "any_open")
        else:
            closed = float(door_rad_now(task, door)) <= float(task.cfg.theta_closed)
        if closed or common.stopped_by_horizon(planner):
            break
    say(env, "retrace: done", closed=bool(closed),
        door_rad=round(float(door_rad_now(task, door)), 3))
    return res, bool(closed)


def _miss_in_the_grasp_frame(task, record: dict, tcp_now) -> dict:
    """Split the reversal's miss into the directions the gripper forgives and does not.

    `build_grasp_pose` stacks the TCP frame as [ortho, closing, approaching]
    (mani_skill/agents/robots/fetch/fetch.py:417-419), so in the RECORDED grasp frame
    the error's y is across the pads — where a 100 mm span on a 26 mm bar forgives about
    37 mm — and its z is along the approach, where nothing is forgiven at all: a hand
    that stops short leaves the bar in FRONT of the fingertips and the pads shut on air.
    A single `d_tcp` cannot tell those apart, and the first two smoke tests read 19 mm
    and 66 mm with the same verdict."""
    q = np.asarray(record["tcp_q"], dtype=np.float64)
    R = np.asarray(sapien.Pose(p=[0.0, 0.0, 0.0], q=q).to_transformation_matrix()
                   )[:3, :3]
    e = np.asarray(tcp_now, dtype=np.float64) - np.asarray(record["tcp_p"], dtype=np.float64)
    return dict(along=round(float(e @ R[:, 2]), 4),   # approaching: unforgiving
                across=round(float(e @ R[:, 1]), 4),  # closing: ~37 mm of room
                up=round(float(e @ R[:, 0]), 4))      # ortho, along the bar


def _landed(planner, task, record: dict, door: DoorSpec) -> dict:
    """Where the reversal actually put each channel, so one number cannot hide another.

    `q_err` is the arm against the target the tape last gave it, in joints; `d_base` and
    `dyaw` are the base against the pose it opened from; `d_leaf` is how far the door has
    moved since — the one term the reversal cannot undo, because that motion is not ours."""
    base = task.agent.base_link.pose.sp
    bx = _np(base.to_transformation_matrix()).reshape(-1, 4, 4)[0][:3, 0]
    yaw = float(np.arctan2(float(bx[1]), float(bx[0])))
    d_base = float(np.linalg.norm(
        _np(base.p).reshape(-1)[:2].astype(np.float64)
        - np.asarray(record["base_xy"], dtype=np.float64).reshape(-1)[:2]))
    return dict(q_err=round(float(getattr(planner, "replay_qerr", 0.0)), 4),
                held=int(getattr(planner, "replay_held", 0)),
                d_base=round(d_base, 4),
                dyaw=round(float(np.degrees(_wrap(yaw - float(record["base_yaw"])))), 2),
                d_leaf=round(float(door_rad_now(task, door)) - float(record["door_rad"]), 4),
                **{f"keep_{k}": v for k, v in
                   _keep_terms(getattr(planner, "replay_keep", {})).items()},
                **_window_creep(record))


def _stance_carried_with_the_leaf(task, door: DoorSpec, record: dict):
    """The recorded base pose, rigidly carried around the hinge by the leaf's own turn.

    The reversal restores the arm to the joints it was recorded in, so wherever the base
    stands, the hand stands in the same place RELATIVE TO THE BASE. The bar, meanwhile,
    has ridden the leaf: after the release it settles, measured -0.017 to -0.032 rad. A
    rotation of the whole stance about the hinge axis by exactly that angle moves the
    hand exactly as the bar moved — and moves the recorded approach path with it, which
    is the part that matters, because that path is what the settled leaf swings into.

    No IK, no re-planning, no branch to re-pick: one rigid rotation of a target pose.

    Returns:
        `(goal_xy, goal_yaw, turned)` — where the base should stand, facing where, and
        the angle it was carried by (radians, world CCW), for the trace.
    """
    xy = np.asarray(record["base_xy"], dtype=np.float64).reshape(-1)[:2]
    yaw = float(record["base_yaw"])
    anchor = record.get("anchor")
    if not RETRACE_CARRY or anchor is None or record.get("hinge_qpos") is None:
        return xy, yaw, 0.0
    anchor_xy, sense = anchor
    anchor_xy = np.asarray(anchor_xy, dtype=np.float64).reshape(-1)[:2]
    phi = float(sense) * (_hinge_qpos(task, door) - float(record["hinge_qpos"]))
    c, s_ = float(np.cos(phi)), float(np.sin(phi))
    rot = np.array([[c, -s_], [s_, c]])
    return anchor_xy + rot @ (xy - anchor_xy), yaw + phi, phi


def _keep_terms(keep: dict) -> dict:
    """The station-keeper's terms, with the sideways creep sampled along the way.

    WHEN the creep happens says WHAT it is: spread evenly over the replay it is a
    standing force against a joint that has only damping; arriving in one burst it is a
    contact, and then the recorded path is no longer clear and no base term can save it."""
    trail = keep.get("trail") or []
    out = {k: v for k, v in keep.items() if k != "trail"}
    if trail:
        out["q"] = [trail[min(len(trail) - 1, (len(trail) * i) // 4)] for i in (1, 2, 3)]
    return out


def _window_creep(record: dict) -> dict:
    """How far the base drifted DURING the recorded window, in the recorded base frame.

    Not a diagnostic for its own sake: the reversal's whole remaining miss is a sideways
    creep the base cannot correct, and if the recording drifted the same way then the
    correction is free — start the reversal mirrored about the recorded pose."""
    if record.get("base_xy_end") is None:
        return {}
    yaw = float(record["base_yaw"])
    fwd = np.array([np.cos(yaw), np.sin(yaw)])
    side = np.array([-np.sin(yaw), np.cos(yaw)])
    d = (np.asarray(record["base_xy_end"], dtype=np.float64).reshape(-1)[:2]
         - np.asarray(record["base_xy"], dtype=np.float64).reshape(-1)[:2])
    return dict(win_fwd=round(float(d @ fwd), 4), win_side=round(float(d @ side), 4),
                win_yaw=round(float(np.degrees(
                    _wrap(float(record["base_yaw_end"]) - yaw))), 2))


def _wrap(a: float) -> float:
    "Angle to (-pi, pi]."
    return float(np.arctan2(np.sin(a), np.cos(a)))


def _settled_aperture(planner, task) -> float:
    """Finger opening in metres, read after the pads have stopped moving.

    `close_gripper` returns while they are still closing, so the naive read reports a
    grip that a moment later is nothing — measured on seed 0 (2026-09-04): the retrace
    said `the bar is held` and `pull_hinge_arc` then refused with fingers at 6.5 mm."""
    planner.idle_steps(t=RETRACE_GRIP_SETTLE)
    return float(_np(task.agent.robot.get_qpos()).reshape(-1)[-2:].sum())


def _park_arm(env, planner, task) -> None:
    """Leave the arm in the posture the fist-and-ladder path was measured from.

    Every handover out of `retrace_the_door` goes through here, so the fallback always
    starts from one pose instead of from wherever the retrace stopped."""
    fold_arm_to_rest(env, planner, task, label="retrace: fold to rest")
    planner.planner.update_from_simulation()


FINISH_BY_HANDLE = os.environ.get("MIKASA_FINISH_BY_HANDLE", "1") == "1"
FINISH_MAX_RAD = float(os.environ.get("MIKASA_FINISH_MAX_RAD", str(ARM_PASS_RAD)))
"""Pull a nearly-shut leaf the last few degrees by its BAR, when the fist has run out.

Both CabinetSearch losses on the second 200-seed verdict sample are this and nothing else:
the leaf stops at 0.155 and 0.233 against a closed band of 0.150 — two millimetres at the
bar on the first — and `pre-contact the panel` refuses at every rung of the ladder. That
is not bad luck: the closer the leaf gets, the worse the fist's geometry becomes, because
the push point rides round onto the cabinet face where the arm cannot stand.

The handle goes the other way. On a nearly shut leaf it is as reachable as it ever is —
it stands where a CLOSED leaf's bar stands, carried round the hinge by the leaf's own
angle — and `pull_hinge_arc` is measured to carry a leaf from 1.8 rad to 0.62 when the
grip holds, so a tenth of a radian is nothing to it.

`FINISH_MAX_RAD` is the band this may be tried in, and it is `ARM_PASS_RAD` on purpose
rather than by coincidence: that constant is validate()'s floor, "below this angle the
opening does not admit the arm" (W12). Above it the arm can work IN the opening and the
fist ladder is the measured path; below it the opening is shut to the arm and only the
bar, which faces outward, is left. The two stages meet exactly where the geometry
changes hands.

**MEASURED AND SHIPPED ON 2026-09-05.** Paired, 200 seeds per arm on one commit
(`runs/2026-09-05-finish`, seeds 300-499, incomplete 0):

    without   199/200    missed 1  no plan 0  truncated 0
    with      200/200    missed 0  no plan 0  truncated 0
    b = 0     c = 1

It never costs a seed, and it cannot: it runs only after the fist ladder has already
failed, so its worst case is the steps it spends, and truncations stay at zero. On the
three known losses it was probed directly and took all three, the leaf reaching
`door_rad=-0.0` each time — but those three were SELECTED for failing, so what this rests
on is the paired run."""


def finish_by_the_handle(env, planner, task, *, door: DoorSpec, anchor=None, res=None):
    """Take the bar of a nearly-shut leaf and ride the arc the rest of the way. `(res, closed)`.

    Post-commit like every closing stage: a refusal is said and the last tuple kept, never
    -1 — physics has already moved the door and the verdict is read from its state.

    Example:
        >>> res, closed = finish_by_the_handle(env, planner, task, door=door, res=res)  # doctest: +SKIP
    """
    rad = float(door_rad_now(task, door))
    if rad > FINISH_MAX_RAD:
        return res, False
    if anchor is None:
        try:
            anchor = common.hinge_anchor(task, _art_key(task, door), door.hinge)
        except Exception as exc:
            say(env, "finish by the handle: no anchor", why=type(exc).__name__)
            return res, False
    anchor_xy, sense = anchor
    # Where the bar stands NOW. `door.handle_bar` is the closed-leaf value and is valid at
    # qpos 0 only; the leaf has carried it round the hinge by its own angle, which at 0.16
    # rad is already 6 cm — more than the gripper forgives across the pads.
    phi = float(sense) * _hinge_qpos(task, door)
    c, s_ = float(np.cos(phi)), float(np.sin(phi))
    xy = np.asarray(door.handle_bar, dtype=np.float64)[:2] - np.asarray(anchor_xy, dtype=np.float64)
    bar = (float(anchor_xy[0] + c * xy[0] - s_ * xy[1]),
           float(anchor_xy[1] + s_ * xy[0] + c * xy[1]),
           float(door.handle_bar[2]))
    say(env, "finish by the handle", door_rad=round(rad, 3),
        bar=[round(v, 3) for v in bar], carried_deg=round(float(np.degrees(phi)), 1))
    out, done = grasp_the_bar(
        env, planner, task, door, bar=bar,
        dock=np.array([door.handle_dock[0], door.handle_dock[1], 0.0]))
    if done:
        return (out if out != -1 else res), False
    res = out
    r = planner.close_gripper(t=12)
    if r != -1:
        res = r
    if common.stopped_by_horizon(planner):
        return res, False
    # The arc, in the CLOSING sense: `pull_hinge_arc` rides `sense * open_dir`, so the
    # close runs its negative, and the target is signed by the leaf the same way the
    # opening's is.
    stop_at = float(door.closed_rad) - CLOSE_MARGIN_RAD
    r = common.pull_hinge_arc(env, planner, task, _art_key(task, door), door.hinge,
                              target_rad=door.open_dir * stop_at, who=WHO,
                              v_handle=PULL_V_HANDLE, max_steps=PULL_MAX_STEPS,
                              anchor=anchor, open_dir=-door.open_dir)
    if r != -1:
        res = r
    planner.open_gripper()
    if common.stopped_by_horizon(planner):
        return res, False
    r = fold_arm_to_rest(env, planner, task, label="finish by the handle: fold to rest")
    if r != -1:
        res = r
    planner.planner.update_from_simulation()
    now = float(door_rad_now(task, door))
    say(env, "finish by the handle: done", door_rad=round(now, 3),
        closed=bool(now <= float(task.cfg.theta_closed)))
    return res, bool(now <= float(task.cfg.theta_closed))


def close_the_door(env, planner, task, *, door: DoorSpec | None = None, anchor=None,
                   arrive_tol: float | None = None, from_here: bool | None = None):
    """Close the opened door by pushing a fist along its arc (W15, the probe run).

    There is no handle on this side of a door standing at ~1.7 rad, so the fist
    PUSHES the panel's outer face and `follow_arc` with the closing sense rides
    the closing tangent — at this geometry the tangent equals minus the face
    normal, so the fist drives normal-first into the face and the panel yields by
    rotating. The stage: fold the arm, drive to the W15 dock on the hinge's side of
    the panel, make a fist, plan the contact (pre-contact WITH the doors in the
    world — a door-free plan is licensed to route through the panel, the sweep-5
    lesson — then the short stroke under the K100 contact stroke with the K102
    touch-stop), ride the arc down, retreat. The caller judges the result by state
    (`door_rad_now`), per D6: after the push has stepped, the return is a 5-tuple
    however far the door got.

    Parametrized by `door` (K113): the ladder, the stem, the hinge, the push point's
    side (`panel_dir`) and the closing sense (`-(sense * open_dir)`, see
    `push_frame`) come from the spec; `door=None` builds it from the task's config
    — the Closed oracle's path, byte-identical.

    Args:
        env: the (possibly wrapped) env.
        planner: the solver (or the stub).
        task: `env.unwrapped`.
        door: the leaf; None = `door_from_cfg(task)`.
        anchor: optional `(anchor_xy, sense)` to skip `hinge_anchor` — for an
            offline stub on a fake task, or a caller that already measured it.
        arrive_tol: None (default) keeps the measured K106 ladder exactly as it was —
            a rung counts only when `drive_base` returns a step. A float turns on the
            K109 ARRIVED-despite-refusal acceptance measured by W20a for the SEARCH
            round: the base reaches the rung (0.012-0.017 m on 4/4 seeds) while the
            drive's own view rotation refuses on the rotate-sweep phantom, so the
            rung is accepted within `arrive_tol` metres and the aim is taken by
            `turn_in_place` — the unplanned spin, because the refusal IS the sweep.
            Without it the fist ends ~100 deg off the panel and the push moves the
            door 0.05 rad (W20a, run 6).

    Returns:
        -1 (a `fail(...)` said why, before any push), or the last gym 5-tuple.

    Example:
        >>> res = close_the_door(env, planner, task)              # doctest: +SKIP
        >>> if res == -1: return res                              # doctest: +SKIP
        >>> res = close_the_door(env, planner, task, door=CAB_MAIN_LEFT)  # doctest: +SKIP
    """
    door = door_from_cfg(task) if door is None else door
    A = task.agent
    if not door.close_docks:
        # A leaf with no ladder is never pushed shut (cab_1: its rungs would
        # stand inside the room's west wall — W24). Refuse before the lift and
        # the two stow drives spend steps on a stage that cannot finish; the
        # caller's own rule closes that round (`Compartment.close_policy`).
        return fail(env, "close the door: this leaf has no closing ladder",
                    stem=door.stem, hinge=door.hinge)
    if anchor is None:
        try:
            anchor = common.hinge_anchor(task, _art_key(task, door), door.hinge)
        except (ValueError, KeyError, IndexError) as e:
            return fail(env, f"close the door: anchor unreadable ({e})")
    anchor_xy, sense = anchor
    anchor_xy = np.asarray(anchor_xy, dtype=np.float64)

    def lift_and_stow():
        """The arm's preparation for a DRIVE to a closing dock — the ladder's. Lift
        the hand off the counter, then stow it for the drive east. Not run before
        the push from here (the owner's note on the clips, 2026-09-05: the hand
        rose, dropped and swung before the fist reached the panel — this pair, for
        a drive that never came); run only when the ladder is about to drive.
        Returns the horizon tuple when the episode ran out, else None."""
        # Lift the hand off the counter BEFORE folding: the place stages leave the
        # TCP a few cm over the countertop, and the straight joint line to rest from
        # there drags a finger through the counter (cycle sweep 1, seed 1: blocked
        # at knot 11/35, all RRT draws timed out, and the closing drive then refused
        # with the arm still extended). Non-fatal, like every posture nicety.
        tcp = task.agent.tcp.pose.sp
        lift = common.arm_move(
            env, planner,
            sapien.Pose(p=[float(tcp.p[0]), float(tcp.p[1]), float(tcp.p[2]) + 0.20],
                        q=tcp.q),
            who=WHO, stage="lift the hand before the fold", tries=2)
        if lift != -1 and common.stopped_by_horizon(planner):
            return lift
        planner.planner.update_from_simulation()
        stow_hand(env, planner, task, z=1.30)
        planner.planner.update_from_simulation()
        return None

    use_here = PUSH_FROM_HERE if from_here is None else bool(from_here)
    if not use_here:
        stopped = lift_and_stow()
        if stopped is not None:
            return stopped

    # Stow for the drive east, at z=1.30 — ABOVE the placed cup, not beside it.
    # NOT the joint-line fold straight from the place pose — the arm stands
    # right over the PLACED CUP and the line to rest drags the gripper through
    # it (cycle sweep 2). And not the default z=1.10 either: the 0.70-ahead
    # fist at 1.10 is a real 2-3 cm near-contact with the cup standing at the
    # place target (every "phantom cup" refusal in this stage's history was
    # this, no mplib artifact involved), and sweep 8 measured the landmine
    # springing — doors closed at 0.118/0.119 rad while the cup ended 0.16 and
    # 0.26 m off the target, swept en route. 1.30 puts the fist 15+ cm over
    # the cup's top and still 9 cm under the open panel's bottom edge (1.391).
    # Shorter aheads at 1.10 (0.40/0.45/0.55) all IK-fail (`joint limit [9]`).
    # Non-fatal. (Executed inside `lift_and_stow` above.)
    # Back the base 0.30 m off the counter before folding: from the parking the
    # joint line to rest dips through the counter edge / stove hood (sweep 5:
    # blocked at knot 43/59 and 17/63 — the fold never planned pre-close).
    # freeze_arm on the ladder drives: with the default BASE_PLAN_MASK the
    # forward screw is free to fold the arm toward the goal, and it folded the
    # stowed fist straight through the PLACED CUP (`gripper<->cup` at step 1 of
    # the twist — the documented BASE_ONLY_PLAN_MASK disease, see extand.py).
    # A base translation moves the TCP by exactly that translation.
    # `push` carries what the reach found back out of the ladder: the fist is made
    # once across rungs, `goal` is the contact pose of the rung that worked, and
    # `stop` is a horizon hit that must end the stage instead of the next rung.
    push: dict = {}

    def _reach(_res):
        """Fist and pre-contact from the rung just taken; `-1` = this rung does not
        afford the push. Nothing executes when the plan refuses, so the arm is left
        exactly as the drive left it and the next rung starts clean."""
        if not push.get("fisted"):
            r = planner.close_gripper(t=12)  # the fist is the tool, not a failed grasp
            push["fisted"] = True
            if r != -1 and common.stopped_by_horizon(planner):
                push["stop"] = r
                return -1
        # The contact, rebuilt from the LIVE angle (the drive never touches the door,
        # but the pull left it anywhere in ~1.65-1.75): push point on the panel's
        # mid-plane arc, approach into the outer face, closing along z (a wrist roll
        # a fist ignores). `push_frame` carries the leaf's side and the mirror.
        # CLOSE_PUSH_GRID walks alternative points when the measured one has no IK.
        grid = list(CLOSE_PUSH_GRID)
        if push.get("from_here"):
            # The owner's fallback, literally: from the opening pose "extend the arm a
            # little past the OUTER side of the leaf". The hand is at the bar's height
            # and radius already; the ladder's push point (r=0.35, z=1.55) was drawn for
            # a base that has driven round to the hinge side. Try the bar's own radius
            # and height first — the shortest reach there is.
            grid = [(PUSH_HERE_R, PUSH_HERE_Z)] + grid
            if abs(PUSH_HERE_R - PUSH_HERE_R_FALLBACK) > 1e-6:
                grid.insert(1, (PUSH_HERE_R_FALLBACK, PUSH_HERE_Z))
        for i, (rad, z) in enumerate(grid):
            centre_xy, n = push_frame(anchor_xy, sense, _hinge_qpos(task, door), door,
                                      r=rad)
            if push.get("from_here") and i == 0:
                _b = task.agent.base_link.pose.sp
                _bx = _np(_b.to_transformation_matrix()).reshape(-1, 4, 4)[0][:3, 0]
                say(env, "push from here: the target", point=[round(float(v), 3) for v in
                    (*centre_xy, z)], normal=[round(float(v), 3) for v in n],
                    base=[round(float(v), 3) for v in _np(_b.p).reshape(-1)[:2]],
                    heading_deg=round(float(np.degrees(np.arctan2(float(_bx[1]),
                                                                   float(_bx[0])))), 1),
                    door_rad=round(float(door_rad_now(task, door)), 3))
            centre = np.array([*centre_xy, z])
            approach = np.array([-n[0], -n[1], 0.0])
            if push.get("from_here"):
                # From the opening pose the panel's normal points ~110 deg off the base
                # heading; a fist that must face it needs 0.83 m of reach at z 1.59 and
                # refuses on the torso limit (measured, seeds 1105/1250). A push does not
                # need the normal: the stroke stops at first touch and follow_arc then
                # drives the BASE along the closing tangent, so the hand only has to be
                # on the outer face. Aim it the way the arm naturally reaches there.
                _b = _np(task.agent.base_link.pose.sp.p).reshape(-1)[:2]
                d = np.array([centre_xy[0] - _b[0], centre_xy[1] - _b[1], 0.0])
                approach = d / max(float(np.linalg.norm(d)), 1e-6)
            goal = A.build_grasp_pose(approach, np.array([0.0, 0.0, 1.0]), centre)
            pre = goal * sapien.Pose([0, 0, -0.15])
            if push.get("from_here"):
                # The stroke runs pre-contact -> goal along the FIST'S axis, and from
                # here that axis lies along the panel (toward the hinge), so a
                # pre-contact behind the fist sits past the leaf's free edge and the
                # stroke grazes the face: first touch is the EDGE, the leaf opens
                # 1-2 deg and the arc then slides the fist off (measured, seeds
                # 1105/1250: TCP 0.05 m/s, leaf still). Back off along the face
                # normal instead, fist orientation unchanged: the stroke enters the
                # outer face square, as the ladder's does.
                pre = sapien.Pose(p=centre + 0.15 * np.array([n[0], n[1], 0.0]), q=goal.q)
            got = common.arm_move(env, planner, pre,
                                  who=WHO, stage="pre-contact the panel",
                                  tries=3 if i == 0 else 2)
            if got != -1:
                if common.stopped_by_horizon(planner):
                    push["stop"] = got
                    return -1
                push["goal"] = goal
                return got
            if i + 1 < len(grid):
                say(env, "no IK at this push point; next point on the panel",
                    r=round(float(rad), 2), z=round(float(z), 2))
        return -1

    def _try_ladder(reach=None):
        aim, _ = push_frame(anchor_xy, sense, _hinge_qpos(task, door), door)
        out = -1
        for dock_xy in door.close_docks:
            dock = np.array([dock_xy[0], dock_xy[1], 0.0])
            view = np.array([aim[0] - dock_xy[0], aim[1] - dock_xy[1], 0.0])
            view = view / np.linalg.norm(view)
            say(env, "drive to the closing dock", dock=[round(float(v), 3) for v in dock])
            out = planner.drive_base(target_pos=dock, target_view_vec=view, freeze_arm=True)
            took = out != -1
            if not took and arrive_tol is not None:
                # W20a: the drive REACHED the rung and only its view rotation
                # refused (the sweep phantom). Accept the arrival by state, then
                # take the aim with the unplanned spin.
                base = common._np(task.agent.base_link.pose.sp.p).reshape(-1)
                d = float(np.hypot(base[0] - dock_xy[0], base[1] - dock_xy[1]))
                if d <= float(arrive_tol):
                    say(env, "closing dock: refused rotate but ARRIVED",
                        dock=[round(float(v), 3) for v in dock], d=round(d, 3))
                    turned = planner.turn_in_place(view)
                    planner.planner.update_from_simulation()
                    out = turned if turned != -1 else planner.idle_steps(t=1)
                    took = True
            if took:
                if reach is None:
                    return out
                got = reach(out)
                if got != -1:
                    return got
                if push.get("stop") is not None:
                    return -1
                say(env, "the rung does not afford the push; next rung of the W15 ladder",
                    dock=[round(float(v), 3) for v in dock])
                out = -1
                continue
            say(env, "closing dock refused, next rung of the W15 ladder")
        return out

    # TWO ladder passes, because sweeps 6 and 7 measured two DISJOINT winning
    # shapes and each one's fix breaks the other's seeds. Pass A, straight from
    # the parking with the stowed boom (sweep 6: 3 first-rung drives, all three
    # closed; but deep parkings lose every rotate on `elbow<->stove`, seeds
    # 6/8). Pass B, only when A exhausts: back the base 0.30 m off and fold to
    # rest, then re-ladder (the back-off is what finally lets the fold plan —
    # sweep 7, knots 134/140 — but from the pulled-back pose OTHER seeds lose
    # their rotates on `gripper<->counter_right`, so it cannot be the default).
    reach = _reach if LADDER_REACH_CHECK else None
    res = -1
    if use_here:
        # The owner's fallback (2026-09-05): before driving anywhere, try the push
        # from where the base already stands — the fist reaches over to the leaf's
        # OUTER face and rides the arc, exactly what the ladder does after its drive,
        # minus the drive. Nothing executes when the reach refuses, so a refusal
        # hands the ladder the same pose it always had.
        say(env, "push from here first", door_rad=round(float(door_rad_now(task, door)), 3))
        push["from_here"] = True
        res = _reach(res)
        push["from_here"] = False
        if push.get("stop") is not None:
            return push["stop"]
        if res == -1:
            say(env, "the panel is out of reach from here; the ladder")
            stopped = lift_and_stow()      # the drive's preparation, only now
            if stopped is not None:
                return stopped
    if res == -1:
        res = _try_ladder(reach)
    if res == -1 and push.get("stop") is None:
        say(env, "ladder exhausted from the parking; back off, fold, retry")
        res_back = planner.move_forward_delta(-0.30)
        if res_back != -1 and common.stopped_by_horizon(planner):
            return res_back
        planner.planner.update_from_simulation()
        fold_arm_to_rest(env, planner, task, label="fold to rest for the close")
        planner.planner.update_from_simulation()
        res = _try_ladder(reach)
    if push.get("stop") is not None:
        return push["stop"]
    if res != -1 and common.stopped_by_horizon(planner):
        return res
    if res == -1:
        # Name the stage that actually refused. A rung was taken and the panel was
        # out of reach from every one of them -> the push, not the drive.
        return fail(env, "pre-contact the panel" if push.get("fisted")
                    else "drive to the closing dock")
    planner.planner.update_from_simulation()

    if reach is None:
        res = _reach(res)
        if push.get("stop") is not None:
            return push["stop"]
        if res == -1:
            return fail(env, "pre-contact the panel")
    goal = push["goal"]
    touch = types.SimpleNamespace(name=door.stem)
    say(env, "stroke to the panel", stop_on="first touch of the cabinet")
    with common.contact_stroke(planner, [door.stem]):
        res = -1
        for _ in range(3):
            res = planner.static_manipulation(goal, stop_on_touch=touch)
            if res != -1:
                break
            say(env, "stroke to the panel: plan refused, one more draw")
    planner.planner.update_from_simulation()
    if res != -1 and common.stopped_by_horizon(planner):
        return res
    if res == -1:
        return fail(env, "stroke to the panel")

    stop_at = float(door.closed_rad) - CLOSE_MARGIN_RAD
    say(env, "push the door closed", target_rad=round(stop_at, 3),
        start_rad=round(door_rad_now(task, door), 3))
    hist: list = []

    def stop() -> bool:
        # `door_rad_now` is open_dir * qpos: "how open", one sign for either leaf.
        hist.append(door_rad_now(task, door))
        if hist[-1] <= stop_at:
            return True
        return len(hist) >= 50 and abs(hist[-1] - hist[-50]) < 0.01
    # The closing tangent is the opening one reversed: the opening arc runs
    # `sense * open_dir` (pull_hinge_arc), so the close runs its negative — for the
    # right leaf `-sense`, exactly K106's call. Derivation in `push_frame`.
    res = planner.follow_arc(anchor_xy, -(sense * door.open_dir), v_handle=0.05,
                             max_steps=CLOSE_MAX_STEPS, stop_when=stop)
    if res == -1:
        return fail(env, "push the door closed: no step was taken")
    if common.stopped_by_horizon(planner):
        return res
    say(env, "door pushed", rad=round(door_rad_now(task, door), 3))
    LAST_PUSH["pushed"] = float(door_rad_now(task, door))

    # Retreat the fist off the panel so the settle reads a free door.
    tcp = task.agent.tcp.pose.sp
    back = common.arm_move(env, planner,
                           sapien.Pose(p=tcp.p, q=tcp.q) * sapien.Pose([0, 0, -0.15]),
                           who=WHO, stage="retreat off the panel", tries=2)
    if back != -1:
        res = back
        if common.stopped_by_horizon(planner):
            return res
    return res


def solve(env, seed=None, debug=False, vis=False, blind=False,
          planner_factory=common.default_planner_factory):
    """Solve one episode. `-1` on a planning/grasp refusal, the gym 5-tuple otherwise.

    Args:
        env: the (possibly wrapped) MikasaCabinetRetrieval-v0 env, or the
            closed-door variant MikasaCabinetRetrievalClosed-v0 (the oracle reads
            `cfg.door_open_rad` and opens the door itself below the passable
            angle — stage 1b).
        seed: episode seed; the solution owns the reset.
        debug, vis: passed to the solver factory.
        blind: accepted because `run_sweep` requires the literal parameter; ignored
            — this task has no cue phase yet. The memory variant makes it real.
        planner_factory: seam for `tools/stub_planner.py`; defaults to the real
            solver.

    Returns:
        -1, or the last gym 5-tuple.

    Example:
        >>> res = solve(env, seed=0)                              # doctest: +SKIP
    """
    obs, info = env.reset(seed=seed)
    if seed is not None:
        seed_everything(seed)
    assert env.unwrapped.control_mode in ("pd_joint_pos", "pd_joint_pos_vel", "pd_joint_delta_pos"), \
        env.unwrapped.control_mode

    task = env.unwrapped
    planner = planner_factory(env, debug, vis)
    res = -1

    cup_p = _np(task.cup.pose.p).reshape(-1, 3)[0]
    say(env, "episode", cup=[round(float(v), 3) for v in cup_p],
        door_rad=float(task.cfg.door_open_rad))

    # -- STAGE 1: duck the torso, dock in front of the cabinet ---------------------
    # Ducked drive: see TORSO_DRIVE — at the rest height the upperarm rides the
    # open panel's bottom edge and three of ten seeds refused the approach on
    # `collision upperarm_roll<->hingerightdoor`.
    # wrist_flex comes off its stop in the same move: the rest keyframe parks it at
    # 2.077 with the stop at 2.16, and the dock drive's base screw died on
    # `joint limit at index [11]` with the wrist pinned there (seed 8, 2 steps in).
    say(env, "duck the torso for the drive", torso=TORSO_DRIVE, wrist_flex=1.7)
    res = plan_joints(env, planner, task,
                      {"torso_lift_joint": TORSO_DRIVE, "wrist_flex_joint": 1.7},
                      label="duck the torso")
    if res != -1 and common.stopped_by_horizon(planner):
        return res
    if res == -1:
        return fail(env, "duck the torso for the drive")
    planner.planner.update_from_simulation()

    # -- STAGE 1b (closed variant only): open the door yourself --------------------
    # Gated on the config so the v0 path is byte-identical when the door starts
    # open. The duck above serves this drive too: the dock screw died on the rest
    # wrist's joint stop (seed 8) before the duck existed, door open or closed.
    if float(task.cfg.door_open_rad) < ARM_PASS_RAD:
        res = open_the_door(env, planner, task)
        if res == -1:
            return res
        if common.stopped_by_horizon(planner):
            return res
        door_now = door_rad_now(task)
        if door_now < ARM_PASS_RAD:
            # The pull stepped, so this is physics, not a refusal (D6): the
            # opening does not admit the arm and the retrieval cannot follow.
            say(env, "MISSED: the door did not open enough",
                door_rad=round(door_now, 3))
            return res
        say(env, "door opened", door_rad=round(door_now, 3))
        planner.planner.update_from_simulation()
        # Re-establish stage 1's invariant: the grasp/pull left the torso and
        # wrist wherever planning put them, and the dock drive below was
        # measured to need the duck (3/10 seeds refused it at torso 0.386,
        # panel now open — the exact collision the duck exists to avoid).
        say(env, "re-duck for the dock drive", torso=TORSO_DRIVE, wrist_flex=1.7)
        res = plan_joints(env, planner, task,
                          {"torso_lift_joint": TORSO_DRIVE, "wrist_flex_joint": 1.7},
                          label="re-duck after the door")
        if res != -1 and common.stopped_by_horizon(planner):
            return res
        if res == -1:
            return fail(env, "re-duck after the door")
        planner.planner.update_from_simulation()
        # Rejoin v0's measured approach lane before the dock: v0 drives to the
        # dock straight from the SOUTH at the cup's x, and the pull leaves the
        # base south-WEST of it — the diagonal approach swept the folded arm's
        # leading edge past the open panel and the dock screw refused
        # `forearm/wrist<->hingerightdoor` (sweep 2, seeds 4/8). Non-fatal: a
        # refused waypoint still tries the dock directly.
        lane = np.array([cup_p[0], -1.55, 0.0])
        say(env, "rejoin the dock lane", lane=[round(float(v), 3) for v in lane])
        res = planner.drive_base(target_pos=lane,
                                 target_view_vec=np.array([0.0, 1.0, 0.0]))
        if res != -1 and common.stopped_by_horizon(planner):
            return res
        planner.planner.update_from_simulation()

    dock = np.array([cup_p[0], WORK_DOCK_Y, 0.0])
    say(env, "dock at the cabinet", dock=[round(float(v), 3) for v in dock])
    res = planner.drive_base(target_pos=dock, target_view_vec=np.array([0.0, 1.0, 0.0]))
    if res != -1 and common.stopped_by_horizon(planner):
        return res
    if res == -1:
        return fail(env, "dock at the cabinet")
    planner.planner.update_from_simulation()
    d_dock, dyaw = common.dock_error(task, (dock[0], dock[1], math.pi / 2))
    say(env, "parked", d_dock=round(d_dock, 3), dyaw_deg=round(dyaw, 1))

    # No torso raise here, and the absence is measured — see TORSO_DRIVE.

    # -- STAGE 2: side-grasp the cup off the shelf ---------------------------------
    grasped = False
    for depth in SIDE_GRASP_DEPTHS:
        grasp, pre = side_grasp_pose(task, depth)
        if grasp is None:
            return fail(env, "grasp: the cup has no collision mesh")
        say(env, "reach the shelf", depth=depth,
            grasp=[round(float(v), 3) for v in grasp.p])
        res = common.arm_move(env, planner, pre, who=WHO,
                              stage=f"pre-grasp (depth {depth})", tries=3)
        if res != -1 and common.stopped_by_horizon(planner):
            return res
        if res == -1:
            say(env, "pre-grasp refused", depth=depth)
            continue
        res = common.arm_move(env, planner, grasp, who=WHO,
                              stage=f"grasp (depth {depth})", tries=3)
        if res != -1 and common.stopped_by_horizon(planner):
            return res
        if res == -1:
            say(env, "grasp pose refused", depth=depth)
            planner.planner.update_from_simulation()
            continue
        res = planner.close_gripper(t=12)
        if res != -1 and common.stopped_by_horizon(planner):
            return res
        if bool(_np(task.agent.is_grasping(task.cup)).any()):
            say(env, "cup in the gripper", depth=depth)
            grasped = True
            break
        say(env, "close missed", depth=depth)
        planner.open_gripper()
        planner.planner.update_from_simulation()
    if not grasped:
        return fail(env, "grasp the cup off the shelf",
                    tried=[float(d) for d in SIDE_GRASP_DEPTHS])
    planner.planner.update_from_simulation()
    common.hold_object_in_planner(env, planner, task, task.cup, held=True, who=WHO)

    # -- STAGE 3: unshelve, retract through the opening, lower to the counter ------
    # The W13 exit profile verbatim. Each leg re-reads the TCP: the previous leg's
    # refinement decides where this one starts.
    tcp = task.agent.tcp.pose.sp
    res = common.arm_move(
        env, planner, sapien.Pose(p=np.asarray(tcp.p) + [0, 0, UNSHELVE_DZ], q=tcp.q),
        who=WHO, stage="unshelve", tries=3)
    if res != -1 and common.stopped_by_horizon(planner):
        return res
    if res == -1:
        return fail(env, "unshelve the cup")
    if not _b(res[-1], "is_grasped"):
        say(env, "MISSED: the cup left the gripper at the unshelve")
        return res
    planner.planner.update_from_simulation()

    tcp = task.agent.tcp.pose.sp
    # base_link, not robot.pose: ds_fetch's root pose is the identity — the base
    # lives in qpos[0:2] — and reading robot.pose sent the first retract to
    # y=+0.55, through the kitchen wall, with the attached cup reported colliding
    # against it. base_link is where the oracles read the base everywhere.
    base_y = float(_np(task.agent.base_link.pose.p).reshape(-1, 3)[0][1])
    res = common.arm_move(
        env, planner,
        sapien.Pose(p=[float(tcp.p[0]), base_y + RETRACT_REACH, float(tcp.p[2])],
                    q=tcp.q),
        who=WHO, stage="retract out of the cabinet", tries=3)
    if res != -1 and common.stopped_by_horizon(planner):
        return res
    if res == -1:
        return fail(env, "retract out of the cabinet")
    planner.planner.update_from_simulation()

    # Back the base off before the hover: the place target sits 0.52 m from the
    # grasp dock, and a horizontal grip at that reach re-runs the close-and-high
    # refusal (`joint limit at index [3]` measured on the first run). Backing to
    # ~0.78 m of reach reproduces W13's hover geometry exactly, and a base move
    # under a live grasp is the solver's normal execution path (W12: the gripper
    # state is re-emitted every step of `follow_moving_forward`).
    say(env, "back the base off", delta=-0.25)
    res = planner.move_forward_delta(-0.25)
    if res != -1 and common.stopped_by_horizon(planner):
        return res
    if res == -1:
        # The planned back-off is a screw over base and arm together, and it is
        # refused when the grasp left a wrist joint on its limit (`joint limit at
        # index [11]`, 1101/1102 under pd_joint_delta_pos, 2026-09-09) — then the hover
        # from the close dock runs into the close-and-high refusal this back-off
        # exists to avoid. Backing up needs no plan: the base drives straight back
        # with the arm held, the cup rides along in the gripper (`drive_straight`).
        say(env, "back-off refused; driving straight back instead", delta=-0.25)
        res = planner.drive_straight(-0.25, v=0.10)
        if res != -1 and common.stopped_by_horizon(planner):
            return res
        if res == -1:
            say(env, "the straight back-off refused too; trying the hover from here")
    planner.planner.update_from_simulation()
    if not bool(_np(task.agent.is_grasping(task.cup)).any()):
        say(env, "MISSED: the cup left the gripper during the back-off")
        return res if res != -1 else planner.idle_steps(t=1)

    target = _np(task.place_target).reshape(-1, 3)[0]
    tcp = task.agent.tcp.pose.sp
    res = common.arm_move(
        env, planner,
        sapien.Pose(p=[float(target[0]), float(target[1]), HOVER_Z], q=tcp.q),
        who=WHO, stage="hover over the place target", tries=3)
    if res != -1 and common.stopped_by_horizon(planner):
        return res
    if res == -1:
        return fail(env, "hover over the place target")
    if not bool(_np(task.agent.is_grasping(task.cup)).any()):
        say(env, "MISSED: the cup left the gripper on the way down")
        return res
    planner.planner.update_from_simulation()

    # Descend until the cup's mesh bottom is PLACE_DROP above the counter. The
    # cup-to-TCP offset is read here, from the still moment, not assumed.
    cup_now = _np(task.cup.pose.p).reshape(-1, 3)[0]
    mesh = task.cup.get_first_collision_mesh(to_world_frame=True)
    bottom = float(np.asarray(mesh.bounds)[0][2])
    tcp = task.agent.tcp.pose.sp
    drop = bottom - (float(target[2]) + PLACE_DROP)
    res = common.arm_move(
        env, planner,
        sapien.Pose(p=[float(tcp.p[0]), float(tcp.p[1]), float(tcp.p[2]) - drop],
                    q=tcp.q),
        who=WHO, stage="descend to the counter", tries=3)
    if res != -1 and common.stopped_by_horizon(planner):
        return res
    if res == -1:
        return fail(env, "descend to the counter")

    # -- STAGE 4: release, back off, settle ----------------------------------------
    res = planner.open_gripper()
    if res != -1 and common.stopped_by_horizon(planner):
        return res
    common.hold_object_in_planner(env, planner, task, task.cup, held=False, who=WHO)
    tcp = task.agent.tcp.pose.sp
    back = common.arm_move(
        env, planner,
        sapien.Pose(p=[float(tcp.p[0]), float(tcp.p[1]) - 0.15, float(tcp.p[2]) + 0.10],
                    q=tcp.q),
        who=WHO, stage="back off", tries=2)
    if back != -1:
        res = back
        if common.stopped_by_horizon(planner):
            return res

    # -- STAGE 5 (require_door_closed only): close the door behind you -------------
    # Gated on the config like stage 1b, so tasks without the requirement run a
    # byte-identical path. The success predicate needs the door at or under
    # door_closed_rad AT the moment the place predicates hold, so the close comes
    # after the place and before the settle that lets the verdict latch.
    if bool(getattr(task.cfg, "require_door_closed", False)):
        res5 = close_the_door(env, planner, task)
        if res5 == -1:
            # NOT a `return -1` (D6): by this stage the whole retrieval has
            # committed physics (cup grasped, carried, released), and sweep 5
            # measured most closing refusals as downstream symptoms of a
            # physical miss — the cup still wedged in the open fingers, the
            # drive honestly refusing through it. Fall through to the settle
            # and let the verdict book the episode by the scene's state.
            say(env, "the closing stage refused; the verdict books the episode")
        else:
            res = res5
            if common.stopped_by_horizon(planner):
                return res
        door_now = door_rad_now(task)
        if door_now > float(task.cfg.door_closed_rad):
            # The push stepped, so this is physics, not a refusal (D6); the
            # settle below still runs and the verdict read stays honest.
            say(env, "the door did not close enough", door_rad=round(door_now, 3))
        else:
            say(env, "door closed", door_rad=round(door_now, 3))

    res = planner.idle_steps(t=SETTLE_STEPS)
    if res == -1:
        return fail(env, "settle wait")

    info = res[-1]
    ok = _b(info, "success")
    say(env, "episode over", success=ok,
        on_counter=_b(info, "on_counter"), settled=_b(info, "settled"),
        xy=round(float(_np(info["xy_distance"]).reshape(-1)[0]), 3))
    if not ok:
        say(env, "MISSED: the cup did not end standing on the counter")
    return res


if __name__ == "__main__":
    import argparse

    import gymnasium as gym

    import mani_skill  # noqa: F401  (registers my_scenes and utils.mikasa)

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    env = gym.make("MikasaCabinetRetrieval-v0", num_envs=1, robot_uids="mikasa_ds_fetch",
                   control_mode="pd_joint_pos", obs_mode="state", sim_backend="cpu")
    out = solve(env, seed=args.seed)
    print("result:", "no_plan" if out == -1 else f"success={bool(_np(out[-1]['success']).any())}")
    env.close()
