"""Scripted oracle for MikasaCabinetSearch-v0: find the cube behind one of N
wall-cabinet doors — close behind you, home between openings, never twice.

The route (design blank I, plan step 4), one ROUND per compartment:

  S0  drive posture (the cabinet line's duck: torso 0.20, wrist 1.7) -> home
      (1.80, -1.90, facing +y) -> settle. The FIRST decision is made from home
      too — the start stands outside the home disk and `passed_home` is False
      at reset, so an opening without this leg is `skipped_home`.
  S1  pick the compartment. Sighted: the k-th entry of a SEEDED random
      permutation (never "left to right" — a fixed order is a memory the
      policy would not need). Blind: a fresh coin every round, uniform WITH
      replacement — the one line that differs (`pick_compartment`); a repeat
      is `reopened` and the env terminates the episode, which is exactly the
      memoryless floor the blank states (`memoryless_search_floor`: 17/27 at
      N=3, 71/128 at the shipped N=4).
  S2  `open_the_door(door=spec)` — the shipped K104-K108 stage (it drives to
      the leaf's handle dock itself, grasps the bar, arc-pulls to 1.75,
      releases, folds). The verdict is STATE: `door_rad_now >= theta_reveal`,
      else MISSED. Then a short look-idle and ONLY `info["success"]` /
      `info["fail"]` are read — `cube_cab` is never touched (the AST test in
      tests/test_cabinet_search.py says so). Success -> settle and return.
  S3  the compartment was empty: `close_the_door(door=spec)` (K106's fist
      push), verdict by state (`<= theta_closed`), one re-push, else MISSED.
      The env's detent zeroes the hinge once the hand is away.
      EXCEPT for a `close_policy="wall_park"` compartment (cab_1 in
      `MikasaCabinetSearch5-v0`, W24), which has no closing ladder at all: its
      leaf was already driven against the west wall by S2 on the same grip
      (`open_the_door(park_rad=cfg.wall_park_rad)`), so S3 only reads the
      verdict — `door_rad_now >= wall_park_rad` — and the round goes on. The
      park cannot wait until S3 because `pull_hinge_arc` refuses an empty hand
      and by then the fist has released, retreated, folded and driven away.
      Every other compartment's round is byte-identical to what it was.
  S4  the tail nobody had driven before W20a: `drive_straight(-0.35)` (an
      UNPLANNED reverse — every planned retreat off the closing dock refused
      8/8 in K106), `fold_via_tcp` (the only fold verified end to end, K111),
      a south waypoint so the final turn at home is ~0, home, settle, read
      `info["passed_home"]`; one re-drive (or one re-push if a door still
      stands open), else MISSED.

W20a (tools/probes/w20_cabinet_round_trip.py, logs/w20, 2026-09-02, four seeds
11-14 on the measured `cab_main` R leaf): the duck posture + straight reverse
(73 steps for 0.36 m) + fold (arm_vs_rest 0.562, the K111 residual) + south
waypoint + home lands `at_home=True` 4/4 with d <= 0.012 m and dyaw 0.0 deg;
the LEFT hinge never moved during the right leaf's grasp (min/max -0.0/0.0 —
no brush). The same logs record the K106 closing stage REFUSING before any
push (`close.ok False`, door still ~1.7) in every approach variant tried, so
S3's re-push and the MISSED it books are the honest reading until that leg is
fixed; the tail was measured with the door still open.

D6 (return contract): -1 ONLY for a planning/grasp refusal before the first
fist closed on a bar (a refused duck, home drive, dock drive, pre-grasp or
stroke — `fail(...)` says which); from the first `close_gripper` on, physics
has committed and every failure returns the last gym 5-tuple with a printed
`MISSED: ...` line, a horizon cut the truncated tuple. `stopped_by_horizon` is
consulted after every primitive.

The oracle's coins come from `np.random.default_rng(seed + RNG_OFFSET)`, NEVER
from the global numpy stream: after `seed_everything(seed)` that stream IS
`RandomState(seed)`, the episode ticket's own generator, and
`np.random.randint(N)` there returns `cube_cab` bit for bit (100 % agreement
over seeds 0-199 — the coin-aliasing test's known-bad control).

Run (container; mplib):
  tools/docker/planner.sh run --timeout 5400 python -m utils.mikasa_oracle.evaluate_planner \\
      -s MikasaCabinetSearch-v0 -p cabinet_search_planner -n 10 --start-seed 0 \\
      --scene-idx 0 --obs-mode state --sim-backend cpu --log-dir logs/search-sweep \\
      --info-keys revealed,reopened,two_open,skipped_home,foreign_moved,n_decisions,second_decision_made,second_decision_ok,retry_count
"""

from __future__ import annotations

import dataclasses
import math
import os

import numpy as np

from utils.mikasa_oracle.planners import oracle_common as common
from planners import cabinet_retrieval_planner as _crp
from planners.cabinet_retrieval_planner import (
    FINISH_BY_HANDLE,
    finish_by_the_handle,
    CAB_1,
    CAB_2_LEFT,
    CAB_2_RIGHT,
    CAB_MAIN_LEFT,
    CAB_MAIN_RIGHT,
    CLOSE_RETRACE,
    RETRACE_ARC,
    RETRACE_TOUCH_RAD,
    TORSO_DRIVE,
    DoorSpec,
    _leaf_joints,
    close_the_door,
    door_rad_now,
    open_the_door,
    plan_joints,
    retrace_the_door,
)
from utils.mikasa_oracle.planners.oracle_common import normalize_base_yaw
from utils.mikasa_oracle.planners.oracle_common import fail as _fail
from utils.mikasa_oracle.planners.oracle_common import say as _say
from utils.mikasa.seeding import seed_everything

WHO = "cabinet_search_planner"

RNG_OFFSET = 10_007
"""Added to the episode seed for the oracle's own generator. The ticket is
`RandomState(seed)` (BatchedRNG.from_seeds, mani_skill batched_rng.py:20) and
`seed_everything(seed)` puts the GLOBAL numpy stream on that same state, so an
unoffset coin — or any `np.random.*` draw — aliases the answer (the 100 %
control in tests/test_cabinet_search.py)."""

#: The leaf each compartment is opened by, keyed by (stem, open hinge). Four
#: entries for `DEFAULT_COMPARTMENTS` (N=4: both leaves of both boxes) plus the
#: optional fifth (`COMPARTMENTS_WITH_CAB_1`). The right leaves are K103-K108's
#: measured stage shifted (CAB_2_RIGHT) or verbatim (CAB_MAIN_RIGHT);
#: `cab_main`'s left leaf is the W21 mirror, measured on its grasp grid;
#: `cab_2`'s left leaf (CAB_2_LEFT) is the same mirror translated west, and its
#: closing ladder is derived, never driven. CAB_1 is the W24 scan and carries NO
#: ladder at all — its round ends at the wall, not shut.
DOOR_SPECS = {
    ("cab_1_main_group", "doorhinge"): CAB_1,
    ("cab_2_main_group", "leftdoorhinge"): CAB_2_LEFT,
    ("cab_2_main_group", "rightdoorhinge"): CAB_2_RIGHT,
    ("cab_main_main_group", "leftdoorhinge"): CAB_MAIN_LEFT,
    ("cab_main_main_group", "rightdoorhinge"): CAB_MAIN_RIGHT,
}

DRIVE_POSTURE = {"torso_lift_joint": TORSO_DRIVE, "wrist_flex_joint": 1.7}
"""The cabinet line's duck (cabinet_retrieval_planner.TORSO_DRIVE: at the rest
0.386 the upperarm rides an open panel's bottom edge, 3/10 dock refusals; the
wrist comes off its 2.16 stop for the dock screw). W20a drove every round-trip
leg from it (4/4 home); the pure-rest drive posture was never run. Re-applied
at the start of EVERY round, because `fold_via_tcp` un-ducks the torso to
REST_TORSO in the tail and the measured dock approach starts ducked."""

CLOSE_ARRIVE_TOL = 0.15
"""The K109 ARRIVED-despite-refusal tolerance handed to `close_the_door` for the
search round. W20a run 6, seeds 11-14: from the search's entry state the shipped
ladder books EVERY rung refused, yet the base stands 0.012-0.017 m from the first
rung (3.2, -1.1) — the drive arrived and only its view rotation refused on the
rotate-sweep phantom (`forearm_roll/wrist_flex <-> microwave door`, K109/K111
item 6, filed). Without the acceptance the closing stage returns -1 on 4/4 (run
2-5: close 0/4); with it the rung is taken and the aim goes to `turn_in_place`.
0.15 m is the rung tolerance, an order over the measured arrival error and well
under the 0.20 m rung spacing of the ladder."""

LOOK_DOCK_Y = -1.05
"""Where the round reads its answer: (the leaf's bar x, -1.05), facing +y — W13's
work dock, and the only pose W22 measured the cube visible from (72 px on a base
camera at 1.75 rad, 19-34 px at 0.9). The pull's own end pose sees 0 px at every
angle. The env judges the reveal from the hinge angle either way; this is what
makes the DEMONSTRATION honest."""

BACK_OFF_M = 0.35
"""The unplanned reverse off the closing dock (W20a: 73 steps, 0.361-0.362 m
travelled at 0.10 m/s on all four seeds). 0.35 is the K106 pass-B back-off
(0.30) plus margin: from there the fold planned 4/4."""

SOUTH_WAYPOINT_DY = -0.40
"""The waypoint (home_x, home_y - 0.40) = (1.80, -2.30) on the free floor: the
approach to home from due south leaves the final rotate ~0 (W20a dyaw 0.0 deg
on all four seeds) instead of a turn next to the counter, where rotates die on
planning-world phantoms (K106 diag 7, K109, K111 item 6)."""

HOME_SETTLE_STEPS = 5
"`at_home` needs base_static (< 0.08); W20a idled 5 after the home drive."
LOOK_STEPS = 10
LOOK_BY_TURN = os.environ.get("MIKASA_LOOK_BY_TURN", "1") == "1"
LOOK_TURN_TOL = 0.005
LOOK_TURN_RAD = float(os.environ.get("MIKASA_LOOK_TURN_RAD", "0.26"))
"""Look by TURNING toward the look point instead of driving to it. 0 until measured.

The owner's version (2026-09-05, after the 1105/1250 recordings): after the pull the
robot may at most turn toward where it now drives, read the verdict, and turn back —
never drive. Two things ride on it. The base then stays exactly where the closing has
to start, so the retrace has no leg to walk back and no cycles to square. And the
demonstration loses the 0.42-0.52 m drive with its 82 deg median turn (5.3's measure).

What it costs is the ONE reason the drive exists: W22 measured the cube at 0 px in the
robot's own cameras from the pull-end pose — at the arc's own yaw, which it never swept.
tools/probes/w22b_turn_only_look.py swept it (2026-09-05, /workspace/w22b, same pixel
count as W22: present minus hidden, the robot's rig only). Cube pixels in the better
base camera by base yaw, degrees off the arc's own yaw toward the cabinet, door at
1.75, torso and head as the pull leaves them (0.386, tilt 0):

    off arc:   -17  -9  -1    +7  +15  +23  +31  +39  +47  +56  +64  +72  +80
    cab_main_R:  0   0   0   163  115   73   63   52   42   42   41   42   43
    cab_main_L (mirrored, same numbers), cab_2_R (same numbers)

The cliff is at the arc's yaw exactly — one degree either way is 0 or 163 px — and past
it there is a plateau above W22's own 30 px reveal floor all the way to +80. The look
dock the drive went to reads 72 px. So a 15 deg turn, eight degrees clear of the cliff
with a 0.3 deg tolerance, reads 115 px with no head tilt and no torso change: better
than the drive, for a turn instead of a 0.42-0.52 m leg with an 82 deg median swing.
Aiming at the look POINT would be an ~86 deg turn and 43 px — the plateau's far end."""
LOOK_MODE = os.environ.get("MIKASA_LOOK_MODE", "head")
"""The button (the owner, 2026-09-07): `head` — pan the head at the compartment's spawn
point, the base never turns (W22c, be1ac9b/cc16b13); `base` — the 15 deg base turn of
W22b (as shipped 2026-09-05 at 187f4eb/399d6df, `LOOK_TURN_RAD`) with the head at zero,
which puts the cube in BOTH base cameras, in the corner. The scene reads the same
variable for `look_cone_rad` (0.40 for head, 0.70 for base), so the W22d terminal holds
in either mode. `MIKASA_LOOK_BY_HEAD=0` is the older spelling of `base`."""
LOOK_BY_HEAD = (LOOK_MODE == "head") and os.environ.get("MIKASA_LOOK_BY_HEAD", "1") == "1"
LOOK_HEAD_TILT = (lambda v: None if v == "" else float(v))(os.environ.get("MIKASA_LOOK_HEAD_TILT", ""))
LOOK_HEAD_CAM_PITCH = 0.3
LOOK_HEAD_TILT_MIN, LOOK_HEAD_TILT_MAX = -0.76, 1.45
"""Head tilt for the look: by geometry unless MIKASA_LOOK_HEAD_TILT pins it. The base cameras
carry a 0.3 rad downward pitch of their own and sit ~1.1 m up, the shelf is at 1.42 m, so
centring the spawn point vertically needs tilt = -(0.3 + elevation) — about -0.55 rad at
1.5 m (negative is up; joint limits -0.76..1.45)."""
LOOK_HEAD_CAMS = {"left_base_camera_link": (-0.5, +0.5, -0.2), "right_base_camera_link": (-0.5, -0.5, +0.2)}
"""The base cameras in the head frame at pan 0: (back, left, yaw) — 0.5 m behind the head
link, 0.5 m to either side, toed IN by 0.2 rad (measured from `get_sensor_params`
2026-09-07: optical yaw -11.5 / +11.5 deg off the heading, 17 deg of downward pitch)."""
LOOK_HEAD_PAN_MAX = 1.2
LOOK_HEAD_RETURN_STEPS = 10
LOOK_HEAD_SETTLE_STEPS = 8
LOOK_HEAD_RAMP_STEPS = int(os.environ.get("MIKASA_LOOK_HEAD_RAMP", "16"))
"""Steps over which the head turns toward the look (and back): 0.8 s at 20 Hz. 0 = the
one-step snap of W22c. Owner's request 2026-09-08: the look was too fast to read."""
LOOK_HEAD_DWELL_EXTRA = int(os.environ.get("MIKASA_LOOK_HEAD_DWELL_EXTRA", "20"))
"""Extra steps the head stays on an EMPTY compartment before turning back (1 s): the
look no longer ends the instant the settle and the dwell budget are spent. On the
compartment with the cube the look ends when success latches, as before."""
"""Look with the HEAD, not the base (W22c, 2026-09-07). The base cameras ride on
`head_camera_link`, so panning the head aims them at the compartment while the base stays
where the pull ended and the tucked arm stays out of the picture — the owner's two asks
in one: look at where the cube would be, and stop turning before the closing.

W22b maximised cube PIXELS, and pixels grow toward the edge of an 86 deg rectilinear
frame: the shipped 15 deg base turn read 115 px with the cabinet in the top-left corner
and the tucked arm across the middle. W22c (`tools/probes/w22c_head_look.py`) swept head
pan and tilt at the pull-end pose, base unturned, and read where the cube lands:
pan = bearing to the compartment's spawn point minus the near camera's own 0.2 rad yaw
puts it at the frame's centre (cx 0.48-0.52, cy 0.43-0.45) on all three compartments,
tilt -0.25 (the cameras carry a 0.3 rad downward pitch of their own; negative tilt is up).
38-41 px there — an honest size for a 6 cm cube a metre away, above W22's 30 px floor.
The head goes back to 0 before the closing, which every executed plan does anyway."""


def head_pan_to_centre(head_xy, heading: float, target_xy, cams=LOOK_HEAD_CAMS,
                       pan_max: float = LOOK_HEAD_PAN_MAX) -> tuple[float, str]:
    """The head pan that puts `target_xy` on the optical axis of the nearer base camera.

    The cameras ride 0.5 m behind and 0.5 m beside the head link and swing with the pan,
    so the bearing from the CAMERA changes with the pan itself; three fixed-point steps
    settle it. The camera is the one that needs the smaller pan. Returns (pan, camera).

    Example:
        >>> pan, cam = head_pan_to_centre((0.0, 0.0), np.pi / 2, (1.2, 1.2))
        >>> cam, round(float(np.degrees(pan)))
        ('right_base_camera_link', -43)
    """
    hx, hy = float(head_xy[0]), float(head_xy[1]); tx, ty = float(target_xy[0]), float(target_xy[1])
    best = None
    for cam, (back, left, yaw_c) in cams.items():
        pan = 0.0
        for _ in range(4):
            a = heading + pan
            cx = hx + back * np.cos(a) - left * np.sin(a)
            cy = hy + back * np.sin(a) + left * np.cos(a)
            bearing = float(np.arctan2(ty - cy, tx - cx))
            pan = float((bearing - heading - yaw_c + np.pi) % (2 * np.pi) - np.pi)
        if best is None or abs(pan) < abs(best[0]):
            best = (pan, cam)
    return float(np.clip(best[0], -pan_max, pan_max)), best[1]


LOOK_BACK_M = float(os.environ.get("MIKASA_LOOK_BACK_M", "0.30"))
"""Straight back-off before the look turn (LOOK_BY_TURN only), metres; 0 = none.
The owner's ask (2026-09-05, the NEW clips of 1105/1250): turning in place with the
arm still out at the bar, the hand caught the leaf (drift -0.08 on 1250's left leaf
against the -0.02..-0.03 settle). So: back off this far, turn to look, turn back,
drive the same distance forward, and only then close — the retrace's own base leg
takes out the millimetres that are left. Guarded by the leaf's angle like the
retrace's legs: a back-off that moves the leaf stops where it is."""
LOOK_TOUCH_RAD = 0.05
"The leaf travel that stops the look's back-off / return: above the settle, below a catch."
FOLD_ONE_LINE = os.environ.get("MIKASA_FOLD_ONE_LINE", "1") == "1"
"""After the close, ONE joint line from where the push left the arm straight to the
drive posture (the rest arm ducked), instead of the shipped four plans (un-duck,
TCP screw to the rest point, a 113-knot branch line to the rest configuration,
and the next round's re-duck) — the owner's ask (2026-09-05): the hand visibly
wandered between one door and the next. Continuous joints are wrapped toward
rest first so the line takes the short way (K111's 2*pi winding). Line only: the
RRT branch is the channel measured executing knots and moving nothing. A blocked
line backs off the old 0.35 m and tries once more, then the old TCP fold.
SHIPS ON (2026-09-05-traj, 1300-1499, 200 seeds): with LOOK_BY_TURN and
PUSH_FROM_HERE, 200/200 against the old path's 194/195, a third fewer steps,
half the base turning, a quarter less wrist roll. MIKASA_FOLD_ONE_LINE=0 is the
old path."""
REDUCK_SKIP_TOL = 0.05
SPRANG_BACK_TOL = 0.02
"Above theta_closed by no more than this after the arc = the push had closed the leaf; an open reading after that is a spring-back."
"Joint-space distance from the drive posture under which the next round skips its re-duck."
"""Steps of stillness after the pull before `info` is read: the freed door
drifts back ~0.05 rad after the release (sweep 2), and the verdict must be
read on the settled angle, not the pull's last frame."""
SUCCESS_SETTLE_STEPS = 30

# -- the touch (owner, 2026-09-08: "a small kick of the red cube"; cfg.terminal == "nudge") --
NUDGE_PRE_M = float(os.environ.get("MIKASA_NUDGE_PRE_M", "0.10"))
"""Metres in FRONT of the cube's near face where the closed hand starts the push."""
NUDGE_PAST_M = float(os.environ.get("MIKASA_NUDGE_PAST_M", "0.01"))
"""Metres PAST the cube's spawn centre the TCP is driven to: with the cube's half-size
0.03 that pushes it ~4 cm, twice `cfg.cube_nudge_m`."""
NUDGE_Z_OFFSET = float(os.environ.get("MIKASA_NUDGE_Z_OFFSET", "0.025"))
"""Metres ABOVE the cube's centre the TCP pushes at. At the centre (3 cm over the shelf)
the palm, thicker than the fingers, met the shelf 5 % short of the pose on every seed
(`gripper_link <-> cab_main object`, 2026-09-08); Retrieval's cup grasp rides 5.7 cm over
the shelf and plans. 2.5 cm puts the fingertip 5.5 cm up a 6 cm cube — a kick that may
tip it, which is still a touch (the predicate is horizontal displacement)."""
NUDGE_TRIES = 3
"The cabinet line's SETTLE_STEPS after the find, so the recording shows it."
NUDGE_TORSO = float(os.environ.get("MIKASA_NUDGE_TORSO", "0.386"))
"""The torso height the touch reaches FROM (owner's suggestion, 2026-09-08: rise, then
reach). The three losses of the 10 cm cube's 200 seeds were all the reach: after the look
the hand hangs 2 cm from the own open leaf's plane (2171: gripper x 0.773, leaf x 0.75)
and the pre-touch plan swings it up through the leaf, 1.72 -> 1.04 rad, under
`theta_reveal`; the same swing opened the double door's other leaf 0.59 rad (2183,
`two_open`). Raising the torso as the arm stands is refused in a band (0.26-0.30 here):
the forearm lies under the cabinet's bottom edge (1.39 m) at y -0.55. So, in order: the
arm to `NUDGE_READY_POSTURE` (a joint line: sagittal, the hand hanging on the base's
centreline, 0.25 m from either leaf); the raise to NUDGE_TORSO (the fingertips end
1.45 m up, in front of the cabinet); `NUDGE_LEVEL_POSTURE` (a joint line: the hand
level, the elbow-down family); the screw down to the pre-touch pose and the push with
the lift frozen, RRT capped. Negative disables the raise and both postures (the old
plan)."""
NUDGE_READY_POSTURE = {
    "shoulder_pan_joint": 0.0, "shoulder_lift_joint": -1.1, "upperarm_roll_joint": 0.0,
    "elbow_flex_joint": 0.2, "forearm_roll_joint": 0.0, "wrist_flex_joint": 2.1,
    "wrist_roll_joint": 0.0,
}
"""The arm the touch reaches from, in JOINT space: sagittal (pan 0, rolls 0), the upper
arm steep up, the forearm forward-down, the hand hanging from the wrist, fingertips
(the TCP) 0.60 m ahead of the base centre on its centreline — y -0.45 at the touch dock,
6.5 cm before the cabinet's front plane — and 1.45 m up at the top of the lift, ABOVE
the cabinet's bottom edge (1.39): the pre-touch move from here is the hand swinging
level about the wrist into the open compartment's air and a 4.5 cm descent, no rise.
Chosen over an IK-picked ready pose: IK from the hanging hand keeps the elbow out on
the leaf's side (2171's solution panned the shoulder 83 deg into the open leaf), and
`static_manipulation` takes the nearest solution, so the reach inherits whatever the
look left. Two neighbours measured: lift -1.2 / elbow 0.4 / wrist 2.0 (198/200 on
2100-2299; 2144 and 2259 hit the lift's limit, 0.02 away, on the screw's first step and
RRT's 160-knot detour swept both leaves of cab_2) and lift -1.1 / elbow 0.7 / wrist 1.97
(fingertips at 1.31, under the bottom edge: every line to the pre-touch pose swept the
edge, 0/5). This one keeps 0.12 rad from the lift's limit; the wrist at 2.1 is 0.06 from
its own but the swing to level UNflexes it. Clear at the docks of the losses at both
the standing torso and 0.386; elbow at z 1.49, y -0.75, 0.25 m from either leaf."""
NUDGE_LEVEL_POSTURE = {
    "shoulder_pan_joint": 0.0, "shoulder_lift_joint": -0.3, "upperarm_roll_joint": 0.0,
    "elbow_flex_joint": -1.7, "forearm_roll_joint": 0.0, "wrist_flex_joint": 1.9,
    "wrist_roll_joint": 0.0,
}
"""The level hand, in JOINT space, taken as a joint line from `NUDGE_READY_POSTURE` at
the top of the lift: the upper arm near horizontal, the forearm UP (elbow -1.7, the
elbow-down family), the hand flexed level along +y with the fingers closing across x —
the pre-touch pose's orientation exactly, TCP 0.65 m ahead on the centreline (y -0.40,
1.5 cm before the cabinet's front plane) and 1.60 m up, so the screw to the pre-touch
pose is a 10 cm descent into the open compartment. Why a second posture: from the
hanging hand (elbow-up family, lift -1.1) the screw levels the hand by lifting the
upper arm, and the lift's limit (-1.221) stops it after three steps on every seed
measured (2144, 2259, then 2171/2183/2219 with the posture two notches lower); the
elbow-down family levels the hand with the lift at -0.3. The two families are separated
by the elbow's sign, which a screw cannot cross, hence the joint line. Grid at 2171's
dock (approach = the TCP's z-axis, closing its y-axis, both measured, not assumed): 9 of
9 level postures collision-free, this one with the most room (lift 0.9, elbow 0.55,
wrist 0.26 from their limits)."""
NUDGE_ROOM_M = 0.20
"""How far the base backs off when the ready posture's joint line is blocked at the
dock (the forearm under the cabinet's bottom edge after the look)."""
NUDGE_REACH_MAX_KNOTS = 60
"""The RRT cap on the pre-touch reach (K59's `max_knots`, refusing over it): the reach
from the ready posture is a screw of a few knots; a long RRT path there is the arm
wandering across the cabinet's front, and 2144's 160-knot one opened the other leaf."""

_CLOSED = -1
"The solver's `gripper_state` for a fist (extand.CLOSED; the stub mirrors it)."


def say(env, stage: str, **extra) -> None:
    """`oracle_common.say` with this oracle's tag.

    Args:
        env: the (possibly wrapped) env.
        stage: one line, greppable, per stage.
        **extra: key=value pairs appended to the line.

    Example:
        >>> say(env, "round", k=0, compartment="cab_2_main_group:rightdoorhinge")  # doctest: +SKIP
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
        >>> return fail(env, "duck for the drive")                 # doctest: +SKIP
    """
    return _fail(env, WHO, stage, **extra)


def _np(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def _b(info, key) -> bool:
    return bool(_np(info[key]).reshape(-1)[0])


def _i(info, key) -> int:
    return int(_np(info[key]).reshape(-1)[0])


def _latches(info) -> dict:
    """The fail latches and counters a MISSED line names, from the last info."""
    keys = ("n_decisions", "reopened", "two_open", "skipped_home", "foreign_moved",
            "passed_home", "any_open")
    out = {k: (_i(info, k) if k == "n_decisions" else _b(info, k))
           for k in keys if k in info}
    # Every compartment's angle, every time a latch line is printed: the search's
    # rounds refuse each other through the WORLD (a leaf left ajar juts into the
    # next bar's corridor and `stop_on_touch` stops the stroke on it), so the
    # trace has to carry what the doors were actually doing, not just the latches.
    if "door_theta_max" in info:
        out["theta"] = [round(float(v), 4)
                        for v in _np(info["door_theta_max"]).reshape(-1)]
    return out


def oracle_rng(seed) -> np.random.Generator:
    """The oracle's own generator: `default_rng((seed or 0) + RNG_OFFSET)`.

    Not the global numpy stream (see `RNG_OFFSET`): after `seed_everything` that
    stream is the ticket's, and a coin drawn from it is the answer.

    Args:
        seed: the episode seed; None counts as 0.

    Returns:
        A `numpy.random.Generator`, fresh — the first draw is always the same
        for a given seed.

    Example:
        >>> int(oracle_rng(0).integers(3)) == int(oracle_rng(0).integers(3))
        True
    """
    return np.random.default_rng((0 if seed is None else int(seed)) + RNG_OFFSET)


def search_plan(seed, n: int):
    """The oracle's generator and its sighted route for an episode.

    `order` is a seeded random permutation of the `n` compartments — the sighted
    arm walks it; the blind arm ignores it and draws from the same `rng` at every
    decision (`pick_compartment`). Both arms draw the permutation, so the coins
    the blind arm sees are the ones AFTER it — the coin-aliasing test replicates
    the sequence through this function, not by hand.

    Args:
        seed: the episode seed.
        n: the number of compartments.

    Returns:
        `(rng, order)`: the generator (already past the permutation) and the
        permutation as a numpy int array.

    Example:
        >>> rng, order = search_plan(0, 3)
        >>> sorted(int(v) for v in order)
        [0, 1, 2]
    """
    rng = oracle_rng(seed)
    order = rng.permutation(int(n))
    return rng, order


def pick_compartment(rng, order, k: int, *, blind: bool) -> int:
    """Which compartment round `k` opens — the ONE line the two arms differ on.

    Sighted: `order[k]`, the seeded route (no repeats by construction — the
    memory of what was opened is the permutation itself). Blind: a fresh draw
    from `rng`, uniform over the compartments WITH replacement; a repeat walks
    into `reopened` and the env terminates the episode — the memoryless floor.

    Args:
        rng: the generator from `search_plan`.
        order: the permutation from `search_plan`.
        k: the round index.
        blind: the design blank's control arm.

    Returns:
        The compartment index into `cfg.compartments`.

    Example:
        >>> rng, order = search_plan(3, 3)
        >>> [pick_compartment(rng, order, k, blind=False) for k in range(3)] == [int(v) for v in order]
        True
    """
    n = len(order)
    if blind:
        return int(rng.integers(n))   # BLIND: the memoryless coin — the one differing line
    return int(order[k])


def door_spec_for(compartment, *, closed_rad: float | None = None) -> DoorSpec:
    """The `DoorSpec` that opens a `Compartment`, keyed by (stem, open hinge).

    The spec's `open_dir` must agree with the compartment's sign for that hinge
    (both say whether opening raises or lowers qpos); `closed_rad` is rebased
    onto the TASK's `theta_closed` when given, so the push stops under the
    threshold the verdict reads — every `cfg` read goes through the spec.

    Args:
        compartment: a `cabinet_search_base.Compartment`.
        closed_rad: the task's `theta_closed`; None keeps the spec's own.

    Returns:
        A `DoorSpec`.

    Raises:
        KeyError: no measured leaf for that (stem, hinge).

    Example:
        >>> from utils.mikasa.scenes.cabinet_search_base import DEFAULT_COMPARTMENTS
        >>> door_spec_for(DEFAULT_COMPARTMENTS[0]).hinge, door_spec_for(DEFAULT_COMPARTMENTS[0]).open_dir
        ('leftdoorhinge', -1)
    """
    key = (str(compartment.stem), str(compartment.open_hinge))
    if key not in DOOR_SPECS:
        raise KeyError(f"no DoorSpec for compartment {key}; have {sorted(DOOR_SPECS)}")
    spec = DOOR_SPECS[key]
    want = int(compartment.open_dir[list(compartment.hinges).index(compartment.open_hinge)])
    assert spec.open_dir == want, (compartment.name, spec.open_dir, want)
    if closed_rad is not None and float(closed_rad) != spec.closed_rad:
        spec = dataclasses.replace(spec, closed_rad=float(closed_rad))
    return spec


def arm_vs_rest(task) -> float:
    """|q - q_rest| over the torso and the arm, the roll joints wrapped toward
    rest (a wound wrist reads 0, not 2*pi) — the W20a/K111 number.

    Args:
        task: `env.unwrapped`.

    Returns:
        The Euclidean residual in radians (the torso's metres count as is).

    Example:
        >>> round(arm_vs_rest(task), 3)                          # doctest: +SKIP
        0.562
    """
    rest_q = np.asarray(task.agent.keyframes["rest"].qpos, dtype=np.float64).reshape(-1)
    q = _np(task.agent.robot.get_qpos()).reshape(-1).astype(np.float64)
    jm = task.agent.robot.active_joints_map
    names = ["torso_lift_joint"] + list(task.agent.controller.controllers["arm"].config.joint_names)
    d = []
    for n in names:
        i = int(jm[n].active_index[0])
        v = float(q[i] - rest_q[i])
        lim = _np(jm[n].limits).reshape(-1)
        if not (np.isfinite(lim).all() and float(lim[-1] - lim[0]) < 4 * np.pi - 0.1):
            v = (v + np.pi) % (2 * np.pi) - np.pi
        d.append(v)
    return float(np.linalg.norm(d))


def drive_posture_targets(task) -> dict:
    """The rest arm, ducked: every arm joint at the rest keyframe, then the duck's
    torso and wrist on top (`DRIVE_POSTURE`). The configuration every measured
    dock approach starts from, as one target for a single joint line.

    Args:
        task: `env.unwrapped`.

    Returns:
        `{joint_name: value}` for the torso and every arm joint.

    Example:
        >>> t = drive_posture_targets(task)                        # doctest: +SKIP
        >>> t["torso_lift_joint"], t["wrist_flex_joint"]           # doctest: +SKIP
        (0.2, 1.7)
    """
    rest_q = np.asarray(task.agent.keyframes["rest"].qpos, dtype=np.float64).reshape(-1)
    jm = task.agent.robot.active_joints_map
    arm_names = list(task.agent.controller.controllers["arm"].config.joint_names)
    targets = {n: float(rest_q[jm[n].active_index[0].item()]) for n in arm_names}
    targets.update({k: float(v) for k, v in DRIVE_POSTURE.items()})
    return targets


def arm_vs_drive_posture(task) -> float:
    """Joint-space distance from `drive_posture_targets`, so the next round can
    tell "already ducked" (skip the re-duck) from "folded to rest" (re-duck).

    Args:
        task: `env.unwrapped`.

    Returns:
        The Euclidean residual in radians (the torso's metres count as is).

    Example:
        >>> round(arm_vs_drive_posture(task), 3)                  # doctest: +SKIP
        0.01
    """
    q = _np(task.agent.robot.get_qpos()).reshape(-1).astype(np.float64)
    jm = task.agent.robot.active_joints_map
    d = [q[jm[n].active_index[0].item()] - v for n, v in drive_posture_targets(task).items()]
    return float(np.linalg.norm(d))


def drive_posture(env, planner, task, *, label: str):
    """The duck (`DRIVE_POSTURE`) as a joint line — the posture every measured
    dock approach starts from. Returns the executor result or -1.

    Example:
        >>> res = drive_posture(env, planner, task, label="duck")  # doctest: +SKIP
    """
    say(env, label, **{k: round(float(v), 3) for k, v in DRIVE_POSTURE.items()})
    res = plan_joints(env, planner, task, dict(DRIVE_POSTURE), label=label)
    if res != -1:
        planner.planner.update_from_simulation()
    return res


def drive_home(env, planner, task, *, via_south: bool):
    """Drive to the home mark facing +y and settle; the W20a tail's last legs.

    Base yaw normalized first (a wound yaw feeds the rotate-sweep phantoms,
    K109), the arm frozen on every drive (the BASE_PLAN_MASK disease: an
    unfrozen base screw folds the arm through whatever is near). With
    `via_south` the base first goes to (home_x, home_y + SOUTH_WAYPOINT_DY) so
    the final turn is ~0; a refused waypoint is non-fatal. The home drive's
    own refusal is -1 (the caller decides whether that is pre-commit).

    Args:
        env, planner: as everywhere.
        task: `env.unwrapped`.
        via_south: approach from the south waypoint.

    Returns:
        The settle's 5-tuple, the truncated tuple, or -1 (home drive refused).

    Example:
        >>> res = drive_home(env, planner, task, via_south=True)   # doctest: +SKIP
        >>> if res == -1: return fail(env, "drive home")           # doctest: +SKIP
    """
    cfg = task.cfg
    hx, hy = float(cfg.home_xy[0]), float(cfg.home_xy[1])
    view = np.array([0.0, 1.0, 0.0])
    normalize_base_yaw(env, planner, task)
    planner.planner.update_from_simulation()
    if via_south:
        wp = np.array([hx, hy + SOUTH_WAYPOINT_DY, 0.0])
        say(env, "drive to the south waypoint", wp=[round(float(v), 3) for v in wp])
        res = planner.drive_base(target_pos=wp, target_view_vec=view, freeze_arm=True)
        if res != -1 and common.stopped_by_horizon(planner):
            return res
        if res == -1:
            say(env, "south waypoint refused; driving home directly")
        planner.planner.update_from_simulation()
    say(env, "drive home", home=[hx, hy])
    res = planner.drive_base(target_pos=np.array([hx, hy, 0.0]), target_view_vec=view,
                             freeze_arm=True)
    if res == -1:
        return -1
    if common.stopped_by_horizon(planner):
        return res
    planner.planner.update_from_simulation()
    res = planner.idle_steps(t=HOME_SETTLE_STEPS)
    d, dyaw = common.dock_error(task, (hx, hy, math.radians(float(cfg.home_yaw_deg))))
    say(env, "parked at home", d_dock=round(d, 3), dyaw_deg=round(dyaw, 1))
    return res


def close_and_verify(env, planner, task, door: DoorSpec, res):
    """S3: `close_the_door`, verdict by state, ONE re-push. Post-commit by
    construction (a door is only closed after it was pulled), so a refused
    stage is said, never -1.

    Args:
        env, planner: as everywhere.
        task: `env.unwrapped`.
        door: the leaf.
        res: the last 5-tuple held by the caller (returned when nothing steps).

    Returns:
        `(res, closed)`: the last 5-tuple and whether `door_rad_now <= theta_closed`.

    Example:
        >>> res, closed = close_and_verify(env, planner, task, door, res)  # doctest: +SKIP
    """
    thr = float(task.cfg.theta_closed)
    from_here = None
    for attempt in range(2):
        if attempt == 1 and _crp.LAST_PUSH.get("pushed", 9.0) <= thr + SPRANG_BACK_TOL:
            # The first push reached the closed band and the leaf reads open again: it
            # sprang back off the forearm that had wrapped round it (cab_2, 1177/1485/
            # 1681). A second push from the same place wraps it the same way; the
            # ladder's rung stands on the hinge side, where the fist never wraps.
            say(env, "the leaf sprang back after a closed push; the re-push from the ladder's rung",
                pushed_to=round(float(_crp.LAST_PUSH["pushed"]), 3))
            from_here = False
        _crp.LAST_PUSH.clear()
        r = close_the_door(env, planner, task, door=door,
                           arrive_tol=CLOSE_ARRIVE_TOL, from_here=from_here)
        if r == -1:
            # NOT -1 (D6): physics has committed; sweep 5 of the cabinet line
            # measured closing refusals as downstream symptoms of physics.
            say(env, "the closing stage refused; verdict by state", attempt=attempt)
        else:
            res = r
            if common.stopped_by_horizon(planner):
                return res, False
        planner.planner.update_from_simulation()
        rad = door_rad_now(task, door)
        if rad <= thr:
            say(env, "door closed", door_rad=round(rad, 3), attempt=attempt)
            return res, True
        say(env, "door not closed" + ("; one re-push" if attempt == 0 else ""),
            door_rad=round(rad, 3), theta_closed=thr)
    # The fist has run out. On a nearly shut leaf that is not bad luck — the push point
    # rides onto the cabinet face where the arm cannot stand — while the HANDLE is as
    # reachable as it ever gets. Both losses on the second verdict sample end here, at
    # 0.155 and 0.233 against a band of 0.150.
    if FINISH_BY_HANDLE:
        res, closed = finish_by_the_handle(env, planner, task, door=door, res=res)
        if closed:
            return res, True
    return res, False


def go_home(env, planner, task, rest_tcp, res):
    """S4: back off the closing dock unplanned, fold, drive home via the south
    waypoint, settle. Post-commit: a refusal is said and the last tuple kept.

    Args:
        env, planner: as everywhere.
        task: `env.unwrapped`.
        rest_tcp: the rest TCP in the base frame (captured at solve start).
        res: the last 5-tuple held by the caller.

    Returns:
        `(res, info)`: the last 5-tuple and its info (None on a refused drive).

    Example:
        >>> res, info = go_home(env, planner, task, rest_tcp, res)   # doctest: +SKIP
    """
    folded = False
    backed_off = False
    if FOLD_ONE_LINE:
        common.normalize_continuous_arm_joints(env, planner, task, who=WHO)
        planner.planner.update_from_simulation()
        for attempt in range(2):
            r = plan_joints(env, planner, task, drive_posture_targets(task),
                            label="fold to the drive posture", tries=1, line_only=True)
            if r != -1:
                res = r
                if common.stopped_by_horizon(planner):
                    return res, res[-1]
                folded = True
                say(env, "in the drive posture", d=round(arm_vs_drive_posture(task), 3))
                break
            if attempt == 0:
                # The line is checked from where the base stands, and the closing pose
                # can leave the BASE inside a neighbour's margin (1105/1177, g21:
                # base_link<->stack_2's leaf, the dishwasher) — every arm plan is then
                # "in collision" before it moves. The shipped back-off, then the line
                # once more; the TCP fold only after that.
                say(env, "the one-line fold is blocked from here; back off and try the line again")
                say(env, "back off the closing dock", distance=-BACK_OFF_M)
                r = planner.drive_straight(-BACK_OFF_M)
                if r != -1:
                    res = r
                    if common.stopped_by_horizon(planner):
                        return res, res[-1]
                backed_off = True
                planner.planner.update_from_simulation()
        if not folded:
            say(env, "the one-line fold is blocked; the shipped fold")
    if not folded and not backed_off:
        say(env, "back off the closing dock", distance=-BACK_OFF_M)
        r = planner.drive_straight(-BACK_OFF_M)
        if r == -1:
            say(env, "back-off took no step; folding from here")
        else:
            res = r
            if common.stopped_by_horizon(planner):
                return res, res[-1]
        planner.planner.update_from_simulation()
    if not folded:
        r = common.fold_via_tcp(env, planner, task, rest_tcp, stage="fold for the drive",
                                who=WHO)
        if r != -1:
            res = r
            if common.stopped_by_horizon(planner):
                return res, res[-1]
        say(env, "arm_vs_rest after the fold", d=round(arm_vs_rest(task), 3))
    planner.planner.update_from_simulation()
    r = drive_home(env, planner, task, via_south=True)
    if r == -1:
        say(env, "MISSED: the drive home refused")
        return res, None
    res = r
    return res, res[-1]


def nudge_the_cube(env, planner, task, res, cab=None):
    """Touch the revealed cube: close the hand, reach in front of it, push it `NUDGE_PAST_M`
    past its centre along +y (into the cabinet), pull back. Returns the last 5-tuple or -1.

    The pose construction is Retrieval's side grasp (`build_grasp_pose` approaching along
    +y, closing across x) at the cube's centre height, from the same door dock, with the
    torso as it is. The predicate reads the cube's displacement from its spawn
    (`cfg.cube_nudge_m`), so the push is checked in `info["nudged"]`, not assumed. The
    cube's position comes from `task.revealed_target()` — NaN until the compartment is
    open, which is the only way the oracle may learn it (the source guard).

    Example:
        >>> res = nudge_the_cube(env, planner, task, res)     # doctest: +SKIP
        >>> bool(res[-1]["nudged"])                            # doctest: +SKIP
        True
    """
    import sapien

    target_p = _np(task.revealed_target()).reshape(-1, 3)[0].astype(np.float64)
    if not np.all(np.isfinite(target_p)):
        say(env, "MISSED: nudge asked before the compartment was revealed")
        return res
    half = float(task.cfg.cube_half)
    say(env, "nudge the cube", target=[round(float(v), 3) for v in target_p])
    # From the handle dock (the bar's x, y=-1.30) the pre-touch pose has no IK and the
    # screw sweeps through the open leaf; Retrieval reaches the same shelf from a dock
    # straight in front of the object at y=LOOK_DOCK_Y (-1.05). Drive there, arm frozen
    # (tucked from the look), facing the cabinet.
    # Dock at the COMPARTMENT's spawn-centre x, not the cube's: at the cube's x the
    # tucked wrist met the microwave beside cab_main R on four of 200 seeds (2026-09-08),
    # and the side reach covers the +-6 cm spawn jitter. Arm in the drive posture first —
    # the look's tuck swept a neighbouring door during the turn (two_open on two seeds).
    centres = getattr(task, "_spawn_centre_x", None)
    dock_x = float(target_p[0])
    if centres is not None and cab is not None:
        dock_x = float(_np(centres).reshape(-1, int(task.layout.n_compartments))[0][cab])
    dock = np.array([dock_x, LOOK_DOCK_Y, 0.0])
    # Two postures for the drive, in order: as the arm stands after the look (its tuck
    # cleared 2100-2109 and the own leaf), then the drive posture (which cleared the six
    # 200-seed losses of the tuck — a neighbour's door swept, the microwave — but meets
    # the own open leaf on some seeds). Never push from the handle dock: the pose has no
    # IK there and one attempt took pinocchio down (log3 assertion) on 2102-2104.
    docked = False
    for posture in ("as-is", "drive"):
        if posture == "drive":
            r = drive_posture(env, planner, task, label="duck for the touch drive")
            if r != -1:
                res = r
                if common.stopped_by_horizon(planner):
                    return res
        say(env, "drive to the touch dock", dock=[round(float(v), 3) for v in dock[:2]], posture=posture)
        r = planner.drive_base(target_pos=dock, target_view_vec=np.array([0.0, 1.0, 0.0]), freeze_arm=True)
        if r != -1:
            res = r
            if common.stopped_by_horizon(planner):
                return res
            docked = True
            break
        say(env, "the touch-dock drive refused", posture=posture)
    planner.planner.update_from_simulation()
    if not docked:
        say(env, "MISSED: no drive to the touch dock; the cube was not nudged")
        return res
    r = planner.close_gripper(t=8)
    if r != -1:
        res = r
        if common.stopped_by_horizon(planner):
            return res
    planner.planner.update_from_simulation()
    approach, closing = np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])
    # Ready, rise, then reach (NUDGE_TORSO): the arm to the sagittal ready posture (a
    # joint line, the torso as it stands); the torso up; then the level reach with the
    # lift frozen. Never a climb under the leaf, never an elbow on the leaf's side.
    raised = False
    if NUDGE_TORSO >= 0:
        r = plan_joints(env, planner, task, dict(NUDGE_READY_POSTURE), label="ready posture for the touch")
        if r == -1:
            # The line to the ready posture sweeps the cabinet's bottom edge when the
            # look left the forearm under it (2110 in pd_joint_delta_pos: `forearm_roll
            # <-> cab object` at knot 3/12). Room is the cure: back off NUDGE_ROOM_M
            # with the arm as it stands, take the posture there, come back to the dock.
            say(env, "the ready posture refused at the dock; backing off for room", m=NUDGE_ROOM_M)
            r = planner.drive_straight(-NUDGE_ROOM_M, v=0.10)
            if r != -1 and common.stopped_by_horizon(planner):
                return r
            planner.planner.update_from_simulation()
            r = plan_joints(env, planner, task, dict(NUDGE_READY_POSTURE), label="ready posture for the touch (with room)")
            if r != -1 and common.stopped_by_horizon(planner):
                return r
            planner.planner.update_from_simulation()
            r2 = planner.drive_base(target_pos=dock, target_view_vec=np.array([0.0, 1.0, 0.0]), freeze_arm=True)
            if r2 != -1:
                r = r2
                if common.stopped_by_horizon(planner):
                    return r
        if r != -1:
            res = r
        else:
            say(env, "the ready posture refused; the raise from where the arm stands")
        planner.planner.update_from_simulation()
        r = plan_joints(env, planner, task, {"torso_lift_joint": NUDGE_TORSO},
                        label="raise the torso for the touch")
        if r != -1:
            res = r
            if common.stopped_by_horizon(planner):
                return res
            raised = True
        else:
            say(env, "the torso raise refused; reaching from where it stands")
        planner.planner.update_from_simulation()
        # The level hand (the elbow-down family) as a joint line; a blocked line is
        # left to the screw from the ready posture, never an RRT draw (K111). The cube
        # is out of the planning world for it: the swing's fingertips stay 8 cm short
        # of its near face by the numbers, but the planner booked `finger <-> cube` at
        # knot 7/21 on 2144 (cube 6 cm off the dock's x) and refused the whole line —
        # and a brush of the cube here is the touch itself, not a loss.
        with common.touchable(planner, "cube"):
            r = plan_joints(env, planner, task, dict(NUDGE_LEVEL_POSTURE),
                            label="level the hand for the touch", line_only=True)
        if r != -1:
            res = r
            if common.stopped_by_horizon(planner):
                return res
        else:
            say(env, "the level posture refused; the screw from the ready posture")
        planner.planner.update_from_simulation()
    for attempt in range(NUDGE_TRIES):
        target_p = _np(task.revealed_target()).reshape(-1, 3)[0].astype(np.float64)
        if not np.all(np.isfinite(target_p)):
            # The compartment stopped counting as revealed (the arm pushed the leaf back
            # under theta_reveal on 2193): a NaN pose here took pinocchio down (log3
            # assertion) on the second attempt. Stop, cleanly.
            say(env, "MISSED: the compartment is no longer revealed; the cube was not nudged", attempt=attempt)
            return res
        z = target_p[2] + NUDGE_Z_OFFSET
        pre_c = np.array([target_p[0], target_p[1] - half - NUDGE_PRE_M, z])
        push_c = np.array([target_p[0], target_p[1] + NUDGE_PAST_M, z])
        pre = task.agent.build_grasp_pose(approach, closing, pre_c)
        push = task.agent.build_grasp_pose(approach, closing, push_c)
        # The lift frozen when it was raised (the level reach); the plain plan, torso
        # free, as the fallback of the same attempt.
        for frozen in ((True, False) if raised else (False,)):
            r = common.arm_move(env, planner, pre, who=WHO, stage=f"pre-touch (try {attempt})",
                                disable_lift_joint=frozen, tries=2,
                                max_knots=NUDGE_REACH_MAX_KNOTS, knot_draws=2, knot_refuse=True)
            if r != -1:
                break
            say(env, "pre-touch refused", attempt=attempt, torso_frozen=frozen)
            planner.planner.update_from_simulation()
        if r != -1 and common.stopped_by_horizon(planner):
            return r
        if r == -1:
            continue
        res = r
        with common.touchable(planner, "cube"):
            r = common.arm_move(env, planner, push, who=WHO, stage=f"push (try {attempt})",
                                disable_lift_joint=raised, tries=2)
        if r != -1 and common.stopped_by_horizon(planner):
            return r
        if r != -1:
            res = r
        planner.planner.update_from_simulation()
        info = res[-1]
        moved = float(_np(info.get("moved_m", 0.0)).reshape(-1)[0]) if isinstance(info, dict) else 0.0
        say(env, "pushed", attempt=attempt, moved_m=round(moved, 3), nudged=_b(info, "nudged"))
        # pull back so the hand is clear whatever happens next
        r = common.arm_move(env, planner, pre, who=WHO, stage="retract", disable_lift_joint=raised, tries=2)
        if r != -1:
            res = r
            if common.stopped_by_horizon(planner):
                return res
        planner.planner.update_from_simulation()
        if _b(res[-1], "success") or _b(res[-1], "nudged"):
            return res
    say(env, "MISSED: the cube was not nudged")
    return res


def solve(env, seed=None, debug=False, vis=False, blind=False,
          planner_factory=common.default_planner_factory):
    """Solve one episode. `-1` only for a refusal before the first fist closed
    on a bar; the gym 5-tuple otherwise (MISSED lines say what went wrong).

    Args:
        env: the (possibly wrapped) MikasaCabinetSearch-v0 env.
        seed: episode seed; the solution owns the reset.
        debug, vis: passed to the solver factory.
        blind: the design blank's control arm — every pick is a coin
            (`pick_compartment`), nothing else changes. Floor:
            `memoryless_search_floor(N)` x the sighted rate — 17/27 at N=3,
            71/128 at the shipped N=4.
        planner_factory: seam for `tools/stub_planner.py`.

    Returns:
        -1, or the last gym 5-tuple.

    Example:
        >>> res = solve(env, seed=0)                              # doctest: +SKIP
        >>> res = solve(env, seed=0, blind=True)                  # doctest: +SKIP
    """
    obs, info = env.reset(seed=seed)
    if seed is not None:
        seed_everything(seed)
    assert env.unwrapped.control_mode in ("pd_joint_pos", "pd_joint_pos_vel", "pd_joint_delta_pos"), \
        env.unwrapped.control_mode
    task = env.unwrapped
    planner = planner_factory(env, debug, vis)
    cfg = task.cfg
    comps = tuple(cfg.compartments)
    n = len(comps)
    rng, order = search_plan(seed, n)
    say(env, "episode", n=n, blind=bool(blind),
        route=("coin" if blind else [int(v) for v in order]))
    # The rest TCP in the base frame while the arm still sits on the reset
    # keyframe — fold_via_tcp's target, exact under any dock yaw.
    rest_tcp = task.agent.base_link.pose.sp.inv() * task.agent.tcp.pose.sp
    res = -1
    committed = False   # D6: True from the first fist on a bar

    # -- S0: drive posture, then home --------------------------------------------
    res = drive_posture(env, planner, task, label="duck for the drive")
    if res != -1 and common.stopped_by_horizon(planner):
        return res
    if res == -1:
        return fail(env, "duck for the drive")
    res = drive_home(env, planner, task, via_south=False)
    if res == -1:
        return fail(env, "drive home from the start")
    if common.stopped_by_horizon(planner):
        return res
    if not _b(res[-1], "passed_home"):
        say(env, "home not credited; one re-drive", **_latches(res[-1]))
        res = drive_home(env, planner, task, via_south=True)
        if res == -1:
            return fail(env, "re-drive home from the start")
        if common.stopped_by_horizon(planner):
            return res
        if not _b(res[-1], "passed_home"):
            # The base drove and parked: a physical miss, not a refusal.
            say(env, "MISSED: home never credited before the first opening",
                **_latches(res[-1]))
            return res
    say(env, "home credited", **_latches(res[-1]))

    for k in range(n):
        # -- S1: pick ---------------------------------------------------------------
        cab = pick_compartment(rng, order, k, blind=blind)
        comp = comps[cab]
        door = door_spec_for(comp, closed_rad=cfg.theta_closed)
        wall_park = str(getattr(comp, "close_policy", "push")) == "wall_park"
        say(env, "round", k=k, compartment=comp.name, pick=cab,
            close_policy=("wall_park" if wall_park else "push"),
            dock=[float(v) for v in door.handle_dock])
        if k > 0 and FOLD_ONE_LINE and arm_vs_drive_posture(task) < REDUCK_SKIP_TOL:
            # The one-line fold left the arm in the drive posture; nothing to re-duck.
            say(env, "already in the drive posture; no re-duck",
                d=round(arm_vs_drive_posture(task), 3))
        elif k > 0:
            # The tail un-ducked the torso (fold_via_tcp -> REST_TORSO); the
            # measured dock approach starts ducked. Post-commit: non-fatal.
            r = drive_posture(env, planner, task, label="re-duck for the dock drive")
            if r != -1:
                res = r
                if common.stopped_by_horizon(planner):
                    return res
            else:
                say(env, "re-duck refused; driving with the posture as it is")

        # A representation change, not a motion (K55/K111): the drives leave the
        # continuous roll joints wound (the previous round's grasp + pull + push +
        # three drives), and a wound wrist is what the bar screw then refuses —
        # W21 measured the LEFT leaf holding on 8 of 8 offset cells from a clean
        # rest pose (fingers 0.026, pull -1.755) while the same leaf, entered from
        # a driven posture, refused with `joint limit at index [7]`, 5.158 rad of
        # twist left, and the pads on nothing. Free, and it cannot make the arm
        # move: it rewrites qpos into the branch nearest rest.
        # OPEN THE HAND. The closing stage's fist is a TOOL (`close_gripper(t=12)`,
        # cabinet_retrieval_planner) and nothing reopens it — in the Closed variant
        # the episode ends there, so it never mattered. Here the next round's bar
        # grasp then approaches with the pads shut: `stop_on_touch` fires when the
        # fist touches the cabinet 5.2 cm short of the bar and `close_gripper`
        # finds nothing between the pads (`pull hinge arc: nothing between the
        # pads`, fingers 0.0/0.0). Measured on seed 11: the pre-open qpos of round
        # 0 ends `0.05, 0.05` and of round 1 `0.0, 0.0` — the ONLY difference
        # between a round that opens its door and one that refuses it.
        if k > 0:
            planner.open_gripper()
        # A representation change, not a motion (K55/K111): three drives and a
        # push leave the continuous roll joints wound, and a wound wrist is what
        # the bar screw refuses. Free, and it cannot make the arm move.
        common.normalize_continuous_arm_joints(env, planner, task, who=WHO)

        # -- S2: open (the stage drives to the leaf's handle dock itself) -----------
        # A wall-park compartment (cab_1, W24) is driven straight on to
        # `cfg.wall_park_rad` on the same grip — see S3 for why the park cannot
        # wait until after the look. Every other compartment passes park_rad=None
        # and its round is byte-identical to the four-compartment task's.
        park = float(cfg.wall_park_rad) if wall_park else None
        # `rec` is filled with the fingers still on the bar and is what lets S3 close
        # the leaf by undoing this stage instead of driving around it. Per round, and
        # dead at the end of it: this is a memory task, and no state may outlive a round.
        rec: dict = {}
        r = open_the_door(env, planner, task, door=door, park_rad=park, record=rec,
                          fold=not LOOK_BY_TURN)
        committed = committed or int(getattr(planner, "gripper_state", 1)) == _CLOSED
        if r == -1:
            if not committed:
                return fail(env, "open the compartment", compartment=comp.name)
            say(env, "MISSED: the opening stage refused after the fist had closed",
                compartment=comp.name)
            return res if res != -1 else planner.idle_steps(t=1)
        committed = True
        res = r
        if common.stopped_by_horizon(planner):
            return res
        rad = door_rad_now(task, door)
        if rad < float(cfg.theta_reveal):
            say(env, "MISSED: the door did not open to the reveal angle",
                compartment=comp.name, door_rad=round(rad, 3),
                theta_reveal=float(cfg.theta_reveal), **_latches(res[-1]))
            return res
        say(env, "door open", compartment=comp.name, door_rad=round(rad, 3))
        planner.planner.update_from_simulation()
        # LOOK. W22 (2026-09-02, pixel counts on the ds_fetch rig, cube present
        # vs hidden): from the pose the ARC PULL leaves the base in — measured
        # (2.504, -1.514, 0.482 rad) — the cube is **0 px in every camera at
        # every angle**, because the arc walks the base south-west and turns it
        # ~62 deg off the cabinet. From the row dock (x_bar, -1.05) facing +y the
        # same cube reads 72 px at 1.75 and 19-34 px already at 0.9. The verdict
        # is state (the hinge angle), so the episode would be credited either
        # way — but a demonstration that never puts the cube in frame teaches a
        # policy to open a door and walk away, and no camera policy could tell an
        # empty compartment from the one it is looking for. So the round drives
        # to where the answer is visible before reading it. Non-fatal: a refused
        # look drive leaves the verdict exactly as it was.
        look = np.array([float(door.handle_bar[0]), LOOK_DOCK_Y, 0.0])
        heading_before = None
        backed = 0.0
        leaf_moved = None
        if LOOK_BY_TURN:
            # The owner's version (2026-09-05): at most TURN toward the look point,
            # never drive to it. The base stays where the arc left it — which is also
            # where the closing has to start, so nothing is walked back afterwards.
            # `turn_in_place` plans nothing, so it cannot be refused by the sweep.
            here = _np(task.agent.base_link.pose.sp.p).reshape(-1)[:2]
            bx = _np(task.agent.base_link.pose.sp.to_transformation_matrix()
                     ).reshape(-1, 4, 4)[0][:3, 0]
            heading_before = np.array([float(bx[0]), float(bx[1]), 0.0])
            # Not toward the look POINT (that is ~86 deg round and reads 43 px) but a
            # fixed LOOK_TURN_RAD toward the cabinet, i.e. toward facing +y: W22b's
            # plateau. The sign is whichever way brings the heading closer to +y, so
            # a right leaf's 28 deg and a left leaf's 152 deg both turn inward.
            leaf0 = _leaf_joints(task, door)

            def leaf_moved() -> bool:
                now = _leaf_joints(task, door)
                n = min(len(now), len(leaf0))
                return bool(np.any(np.abs(now[:n] - leaf0[:n]) > LOOK_TOUCH_RAD))

            tucked = False
            if FOLD_ONE_LINE and not RETRACE_ARC:
                # No reversal is coming, so there is no tape to keep: tuck the arm —
                # the rest arm with the wrist ducked, torso as it stands — in one joint
                # line right after the retreat, and look with it tucked. The owner's
                # note on the clips (2026-09-05): the hand, left out at the bar
                # through the back-off and the return, brushed the leaf. The fist then
                # reaches the panel from the tucked arm. Non-fatal: a blocked line
                # leaves the arm where the retreat left it.
                tuck = drive_posture_targets(task)
                tuck.pop("torso_lift_joint", None)
                common.normalize_continuous_arm_joints(env, planner, task, who=WHO)
                planner.planner.update_from_simulation()
                r = plan_joints(env, planner, task, tuck, label="tuck the arm for the look",
                                tries=1, line_only=True)
                if r != -1:
                    res = r
                    if common.stopped_by_horizon(planner):
                        return res
                    planner.planner.update_from_simulation()
                    tucked = True
            if LOOK_BACK_M > 0 and not tucked:
                # The back-off exists for a hand left out at the bar (the reversal's
                # tape). Tucked, the arm stands below the leaf and the turn clears it —
                # the owner's call on the clips (2026-09-05): no drive away and back.
                # Back off first so the hand, still out where the bar was, clears the
                # leaf while the base turns (LOOK_BACK_M). Measured by base pose, so
                # the return leg drives exactly what was driven, stopped early or not.
                say(env, "back off before the look", distance=-LOOK_BACK_M)
                r = planner.drive_straight(-LOOK_BACK_M, stop_when=leaf_moved)
                if r != -1:
                    res = r
                    if common.stopped_by_horizon(planner):
                        return res
                    planner.planner.update_from_simulation()
                    now_xy = _np(task.agent.base_link.pose.sp.p).reshape(-1)[:2]
                    backed = float(np.linalg.norm(now_xy - here))
                    if leaf_moved():
                        say(env, "the back-off moved the leaf; stopped", backed=round(backed, 3))
            theta = float(np.arctan2(heading_before[1], heading_before[0]))
            if LOOK_BY_HEAD:
                # Aim the head at the compartment's spawn point — where the cube would
                # stand if this were the one — and leave the base alone (W22c).
                centres = getattr(task, "_spawn_centre_x", None)
                spawn_x = (float(_np(centres).reshape(-1, int(task.layout.n_compartments))[0][cab])
                           if centres is not None else float(door.handle_bar[0]))
                spawn_y = float(task.cfg.spawn_depth)
                head_link = getattr(task.agent.robot, "links_map", {}).get("head_camera_link")
                head_xy = (_np(head_link.pose.sp.p).reshape(-1)[:2] if head_link is not None else here)
                pan, cam = head_pan_to_centre(head_xy, theta, (spawn_x, spawn_y))
                if LOOK_HEAD_TILT is None:
                    cam_z = (float(_np(head_link.pose.sp.p).reshape(-1)[2]) if head_link is not None else 1.2)
                    spawn_z = float(task.cfg.shelf_top_z) + float(task.cfg.cube_half)
                    dist = float(np.hypot(spawn_x - float(head_xy[0]), spawn_y - float(head_xy[1]))) + 0.5
                    tilt = float(np.clip(-(LOOK_HEAD_CAM_PITCH + np.arctan2(spawn_z - cam_z, dist)),
                                         LOOK_HEAD_TILT_MIN, LOOK_HEAD_TILT_MAX))
                else:
                    tilt = float(LOOK_HEAD_TILT)
                say(env, "look with the head", pan_deg=round(float(np.degrees(pan)), 1),
                    tilt=round(tilt, 3), camera=cam[:5], at=[round(spawn_x, 3), round(spawn_y, 3)])
                # The look lasts the head's settle plus the task's dwell (W22d: success
                # needs the revealed cube seen for `reveal_dwell_steps`), and stops the
                # moment success latches.
                dwell = int(getattr(task.cfg, "reveal_dwell_steps", 0) or 0)
                r = planner.hold_head(pan, tilt, stop_on_success=True, ramp=LOOK_HEAD_RAMP_STEPS,
                                      t=LOOK_HEAD_RAMP_STEPS + LOOK_STEPS + LOOK_HEAD_SETTLE_STEPS + dwell
                                      + LOOK_HEAD_DWELL_EXTRA)
            else:
                theta += LOOK_TURN_RAD * (1.0 if theta < np.pi / 2 else -1.0)
                aim = np.array([float(np.cos(theta)), float(np.sin(theta)), 0.0])
                say(env, "turn toward the cabinet to look",
                    by_deg=round(float(np.degrees(LOOK_TURN_RAD)), 1),
                    aim_deg=round(float(np.degrees(theta)), 1))
                # Guarded by the leaf's own angle, like the retrace's legs: on cab_2 the turn
                # in place from the opening pose brushed the leaf even with the arm tucked
                # (1177, g28: drift -0.155 by the close, the push then wrapped the arm round
                # the leaf and the leaf sprang back to 0.47). cab_main never trips it.
                r = planner.turn_in_place(aim, tol=LOOK_TURN_TOL, stop_when=leaf_moved)
                if leaf_moved():
                    say(env, "the look turn moved the leaf; stopped short of the aim")
        else:
            say(env, "drive to the look dock", dock=[round(float(v), 3) for v in look])
            r = planner.drive_base(target_pos=look, target_view_vec=np.array([0.0, 1.0, 0.0]),
                                   freeze_arm=True)
        if r != -1:
            res = r
            if common.stopped_by_horizon(planner):
                return res
            planner.planner.update_from_simulation()
        else:
            say(env, "the look drive refused; reading the verdict where we stand")
        res = planner.idle_steps(t=LOOK_STEPS)
        if common.stopped_by_horizon(planner):
            return res
        info = res[-1]
        if _b(info, "success"):
            say(env, "FOUND", compartment=comp.name, k=k, **_latches(info))
            res = planner.idle_steps(t=SUCCESS_SETTLE_STEPS)
            say(env, "episode over", success=_b(res[-1], "success"))
            return res
        if getattr(task.cfg, "terminal", "seen") == "nudge" and _b(info, "revealed"):
            # The cube's compartment is open and (with the head look) seen; the touch
            # terminal wants it pushed before the episode counts.
            say(env, "FOUND, not yet touched", compartment=comp.name, k=k, **_latches(info))
            if LOOK_BY_TURN and LOOK_BY_HEAD and heading_before is not None:
                r = planner.hold_head(0.0, 0.0, t=LOOK_HEAD_RAMP_STEPS + LOOK_HEAD_RETURN_STEPS,
                                      ramp=LOOK_HEAD_RAMP_STEPS)
                if r != -1:
                    res = r
                    if common.stopped_by_horizon(planner):
                        return res
                    planner.planner.update_from_simulation()
            r = nudge_the_cube(env, planner, task, res, cab=cab)
            if r != -1:
                res = r
            if common.stopped_by_horizon(planner):
                return res
            res = planner.idle_steps(t=SUCCESS_SETTLE_STEPS)
            say(env, "episode over", success=_b(res[-1], "success"), **_latches(res[-1]))
            return res
        if _b(info, "fail"):
            say(env, "MISSED: the search failed", compartment=comp.name, k=k,
                **_latches(info))
            return res
        say(env, "empty", compartment=comp.name, k=k, **_latches(info))
        if LOOK_BY_TURN and LOOK_BY_HEAD and heading_before is not None:
            r = planner.hold_head(0.0, 0.0, t=LOOK_HEAD_RAMP_STEPS + LOOK_HEAD_RETURN_STEPS,
                                  ramp=LOOK_HEAD_RAMP_STEPS)
            if r != -1:
                res = r
                if common.stopped_by_horizon(planner):
                    return res
                planner.planner.update_from_simulation()
        elif LOOK_BY_TURN and heading_before is not None:
            r = planner.turn_in_place(heading_before, tol=LOOK_TURN_TOL, stop_when=leaf_moved)
            if r != -1:
                res = r
                if common.stopped_by_horizon(planner):
                    return res
                planner.planner.update_from_simulation()
            if backed > 0:
                say(env, "return from the look back-off", distance=round(backed, 3))
                r = planner.drive_straight(backed, stop_when=leaf_moved)
                if r != -1:
                    res = r
                    if common.stopped_by_horizon(planner):
                        return res
                    planner.planner.update_from_simulation()

        # -- S3: close behind you -----------------------------------------------------
        # ... unless the compartment's rule is the wall park (cab_1, W24). Then
        # there is nothing to do here: the leaf was already driven to the wall by
        # the opening stage, on the same grip, and it STAYS there — that is the
        # rule and the mark the corrected floor is priced on
        # (`self_marking_search_floor`). The park has to happen inside the
        # opening stage rather than here because `pull_hinge_arc` refuses an
        # empty hand and by this line the fist has released, retreated and folded
        # and the base has driven to the look dock. So this branch only checks
        # the state the env will judge, and the round goes straight to S4.
        if wall_park:
            rad = door_rad_now(task, door)
            if rad < float(cfg.wall_park_rad):
                say(env, "MISSED: the door did not reach the wall",
                    compartment=comp.name, door_rad=round(rad, 3),
                    wall_park_rad=float(cfg.wall_park_rad), **_latches(info))
                return res
            say(env, "door parked at the wall", compartment=comp.name,
                door_rad=round(rad, 3))
        else:
            closed = False
            if CLOSE_RETRACE and rec:
                # Undo the opening rather than drive around the leaf. A False hands
                # over to the measured ladder from a defined pose (hand open, arm at
                # rest), so this can only add a path, never remove one.
                res, closed = retrace_the_door(env, planner, task, door=door,
                                               record=rec, res=res)
                if common.stopped_by_horizon(planner):
                    return res
            if not closed:
                res, closed = close_and_verify(env, planner, task, door, res)
            if common.stopped_by_horizon(planner):
                return res
            if not closed:
                say(env, "MISSED: the door did not close", compartment=comp.name,
                    door_rad=round(door_rad_now(task, door), 3))
                return res

        # -- S4: back off, fold, home -------------------------------------------------
        res, info = go_home(env, planner, task, rest_tcp, res)
        if common.stopped_by_horizon(planner) or info is None:
            return res
        if not _b(info, "passed_home"):
            if _b(info, "any_open") and wall_park:
                # There is no re-push for this leaf and no second way to deal
                # with it: the env clears a wall-park compartment only when its
                # hinge stands at `wall_park_rad`, still, with the hand away. If
                # it still reads open here, the park did not take (or the leaf
                # swung back off the wall), and the round has nothing left to
                # try — book it by state.
                say(env, "MISSED: the parked door still reads open at home",
                    compartment=comp.name,
                    door_rad=round(door_rad_now(task, door), 3),
                    wall_park_rad=float(cfg.wall_park_rad), **_latches(info))
                return res
            if _b(info, "any_open"):
                # The detent never fired: the door drifted back over
                # theta_closed (or never got under it). One re-push, then the
                # same tail again.
                say(env, "a door still stands open at home; one re-push",
                    door_rad=round(door_rad_now(task, door), 3), **_latches(info))
                res, closed = close_and_verify(env, planner, task, door, res)
                if common.stopped_by_horizon(planner):
                    return res
                if not closed:
                    say(env, "MISSED: the door did not close on the re-push",
                        compartment=comp.name)
                    return res
                res, info = go_home(env, planner, task, rest_tcp, res)
            else:
                say(env, "home not credited; one re-drive", **_latches(info))
                r = drive_home(env, planner, task, via_south=True)
                if r == -1:
                    say(env, "MISSED: the re-drive home refused")
                    return res
                res, info = r, r[-1]
            if common.stopped_by_horizon(planner) or info is None:
                return res
            if not _b(info, "passed_home"):
                say(env, "MISSED: home never credited after the round", k=k,
                    **_latches(info))
                return res
        say(env, "home credited", k=k, **_latches(info))

    # Every compartment opened and closed without a find: the env disagrees
    # with the ticket's own construction (the cube is in one of them) — book
    # it by state, say so.
    say(env, "MISSED: no compartment revealed the cube", rounds=n)
    return res


if __name__ == "__main__":
    import argparse

    import gymnasium as gym

    import mani_skill  # noqa: F401  (registers my_scenes and utils.mikasa)

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blind", action="store_true")
    args = ap.parse_args()
    env = gym.make("MikasaCabinetSearch-v0", num_envs=1, robot_uids="mikasa_ds_fetch",
                   control_mode="pd_joint_pos", obs_mode="state", scene_idx=0,
                   sim_backend="cpu")
    out = solve(env, seed=args.seed, blind=args.blind)
    print("result:", "no_plan" if out == -1 else f"success={_b(out[-1], 'success')}")
    env.close()
