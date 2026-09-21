"""Oracle for MikasaDepthRecall-v1, built from extraction and restoration flows.

Each transfer is one of the two strokes this branch measured at 100/100:

- **extract** (shelf -> counter slot) is `cabinet_retrieval_planner`'s straight flow,
- **restore** (counter slot -> shelf) uses the shelf-placement helpers extracted
  from `cabinet_stow_planner` into `planners.oracle.cabinet_place`.

The default four-prop row requires seven transfers: four extractions and three
restorations. The separate CabinetStow environment is not needed by this oracle.

Nothing is re-derived: the docks, `GRASP_LIFT`, the depth ladder, `ready_by_line` and
its pre-checks, the loaded hold and the lift-off all come from those files. What is new
here is only the sequencing and the one lateral fact that makes it work — with the base
docked at the object's own x, the arm reaches sideways far enough that NO transfer ever
carries a prop while the base moves (measured 2026-09-10: onto the counter 36 cm toward
smaller x, into the shelf 36 cm toward larger x, both 5/5).

**Where the answer is read, and it is one line.** The extraction order is taken from the
props' own poses — front first, which physics enforces anyway — so it needs no
privileged information. The only privileged read is the RESTORE assignment: which staged
prop belongs in which row slot. `choose_assignment` reads `task.original_slots` when
sighted and draws when blind, and `--blind` swaps exactly that. The coins come from
`default_rng(seed + 10_007)`, never from the episode RNG, or they alias the ticket
(CabinetSearch's recorded trap).

Contract: `solve(env, seed=None, debug=False, vis=False, blind=False)` returning -1 (no
plan) or the gym 5-tuple, per `template_planner.py`.
"""

from __future__ import annotations

import contextlib
import inspect
import os

import numpy as np
import sapien

from planners import cabinet_retrieval_planner as retr
from planners.oracle import cabinet_place as sp
from planners.oracle import oracle_common as common
from utils.mikasa.seeding import seed_everything

WHO = "depth_recall_v1_planner"

#: Straight-up clearance off the shelf after a grasp, and off the counter after one.
#: Imported, not re-derived: this is the retrieval flow's own number.
UNSHELVE_DZ = retr.UNSHELVE_DZ

#: Metres the unshelve ALSO travels back along the approach axis, so the leg separates
#: the fingers from the next prop in the row instead of sliding up its face.
#:
#: Measured 2026-09-10, first run of this task: the row pitch is 5.5 cm and a prop is
#: 4.5 cm across, so 1 cm of air separates two bodies — and the finger plates reach
#: ~3 cm past the TCP along the approach. A pure +z unshelve was refused at step 1 with
#: `l_gripper_finger_link <-> prop_1`, `0.005 of the twist left`, on every draw. Going
#: up AND back moves the fingers off the neighbour's face from the first knot.
UNSHELVE_BACK = float(os.environ.get("MIKASA_DR1_UNSHELVE_BACK", "0.06"))

#: Where the TCP sits above a prop's mid-height, tried in order.
#:
#: The retrieval family's single `GRASP_LIFT` (0.02) exists to keep `wrist_flex_link`,
#: which hangs 6.7 cm under the hand's axis, off the surface — and on a 12 cm prop it
#: leaves only 1.3 cm. Measured 2026-09-10, seed 9: every one of 75 IK solutions for a
#: counter grasp was refused with `gripper_link <-> counter_main`, because the props
#: already standing beside it had taken away the branches that clear. The higher rungs
#: are `depth_recall_planner`'s own measured top band (its `GRASP_Z_LADDER` is
#: 0.045/0.055/0.035), where the prop's own body is the support under the pads.
GRASP_LIFTS = tuple(
    float(v) for v in os.environ.get("MIKASA_DR1_GRASP_LIFTS", "0.02,0.035,0.045").split(",")
)

#: Torso heights the lateral cross is tried at, in order. `None` means "wherever the
#: carry-out left it" — the first rung is free, the rest cost a joint line.
#:
#: The ladder stops at 0.20 on purpose: there the carried prop's base rides at 1.25,
#: which is 21 cm over the tops of the props already standing on the slots. Going lower
#: buys reach and starts sweeping them.
CROSS_TORSO_RUNGS = (None, 0.30, 0.20)

#: Joint-space step the TORSO lines are planned with, metres.
#:
#: `plan_qpos_line`'s default is 0.1, which for a 0.386 m torso ride is four knots and
#: therefore four control steps — about 9.6 cm asked per step. Under
#: `pd_joint_delta_pos` the channel caps at 0.1 per step, so the command SATURATES:
#: measured 2026-09-10 on seed 0, the torso action sat at |a| = 1.000 on 1.1 % of the
#: episode's steps while the joint actually moved at most 3.35 cm. That is the "base
#: dropping sharply" in the clip, and at the 10 Hz recording clock it gets worse, not
#: better. 0.02 asks ~2 cm a step, a fifth of the channel.
#:
#: The arm never clips at either step size (0.000 % on all seven channels, same run).
TORSO_LINE_STEP = float(os.environ.get("MIKASA_DR1_TORSO_STEP", "0.02"))

#: Largest torso move issued as ONE joint line, metres; longer rides are chunked.
#:
#: The line's own knots are already fine (22-26 for a 0.386 m ride, ~1.5 cm each), so
#: the per-step speed is set by the follower, not the plan — and the follower is shared
#: with every other task here. Chunking is the part this oracle owns: a short line
#: cannot build up the same advance.
TORSO_CHUNK = float(os.environ.get("MIKASA_DR1_TORSO_CHUNK", "0.08"))

#: Torso heights the RESTORE's lateral cross is tried at, in order.
#:
#: The cross has to thread a band, and both walls are measured. Too HIGH and it hits
#: the closed LEFT door: seed 18 crossed 36 cm west with the torso at its 0.386 stop,
#: which puts the hand near z 1.5, and the door's panel occupies z 1.39-2.31 below
#: x = 2.25 — every posture was refused with
#: `l_gripper_finger_link <-> cab_main..._hingeleftdoor`. Too LOW and it sweeps the
#: props already standing on the slots, whose tops are 12 cm above the counter: seed 6
#: tipped the prop it was on its way to. The middle rungs put the hand between the two,
#: and the previous height is kept as the last resort so nothing that used to plan
#: stops planning.
CROSS_SAFE_TORSO = tuple(
    float(v) for v in os.environ.get("MIKASA_DR1_CROSS_TORSO", "0.15,0.20,0.10,0.25").split(",")
)

#: Roll joints, by name. Their travel is what `roll_rank` minimises.
ROLL_JOINTS = ("upperarm_roll_joint", "forearm_roll_joint", "wrist_roll_joint")

#: Torso rungs, imported from the stow flow rather than copied: the counter reach and
#: the loaded shelf reach are the same two ladders it measured.
COUNTER_TORSO_RUNGS = sp.GRASP_TORSO_RUNGS

#: The counter GRASP gets two rungs above the stow flow's, because it has an obstacle
#: the stow flow never has: the props already standing on the other slots. Seed 6 puts
#: one in the dock's own column, and the reach to a prop 24 cm west then has no IK at
#: 0.0/0.10/0.20 — the arm has to come OVER it, not around it.
COUNTER_GRASP_TORSO_RUNGS = tuple(COUNTER_TORSO_RUNGS) + (0.30, retr.READY_TORSO)
SHELF_TORSO_RUNGS = sp.RISE_TORSO_RUNGS

#: Coin offset for the oracle's own draws. NEVER the episode RNG: `seed_everything`
#: puts numpy's global stream in the ticket's state, so an unoffset draw aliases the
#: answer (CabinetSearch, blank I).
COIN_OFFSET = 10_007


def say(env, stage: str, **extra):
    """Trace one stage to stdout and the episode log."""
    return common.say(env, WHO, stage, **extra)


def fail(env, stage: str, **extra):
    """Print the refusal and return -1 — the contract's no-plan sentinel."""
    return common.fail(env, WHO, stage, **extra)


def _np(x) -> np.ndarray:
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _b(info, key: str) -> bool:
    return bool(_np(info[key]).reshape(-1)[0])


def roll_rank(task):
    """An IK ranking that puts the least ROLL travel first, `goal_order` as the tiebreak.

    Why this exists, measured 2026-09-10 on seed 0: the planning roll window is +-3.44
    rad, which is NARROWER than a full turn, so when a solution sits near the far edge
    the joint can only get there the long way round. The wrist rolled **-5.83 rad over
    166 steps** on one lateral cross — a near-complete revolution of the gripper with
    the hand translating, which is the "spinning like a wheel" in the clip — and its
    range over the whole episode was only [-3.15, 3.07], i.e. it was bouncing between
    the window's two ends. `goal_order` cannot see this: it ranks by the largest single
    joint's travel, and a 5.8 rad roll loses to nothing when the alternatives move the
    elbow further.

    Returns a callable for `ready_by_line(rank=)`.

    Example:
        >>> res = retr.ready_by_line(..., rank=roll_rank(task))   # doctest: +SKIP
    """
    amap = task.agent.robot.active_joints_map
    idx = []
    for name in ROLL_JOINTS:
        joint = amap.get(name)
        if joint is not None:
            idx.append(int(np.asarray(joint.active_index).reshape(-1)[0]))

    def rank(goals, cur):
        cur = np.asarray(cur, dtype=float).reshape(-1)

        def cost(g):
            g = np.asarray(g, dtype=float).reshape(-1)
            n = min(len(g), len(cur))
            roll = sum(abs(float(g[j]) - float(cur[j])) for j in idx if j < n)
            return (round(roll, 3), float(np.abs(g[:n] - cur[:n]).max()))

        return sorted(np.atleast_2d(goals), key=cost)

    return rank


def census(env, task, stage: str):
    """Trace every prop's position and tilt. A tilt that appears between two of these
    names the leg that caused it; a tilt read only at the end names nothing.

    Example:
        >>> census(env, task, "after the lift-off")    # doctest: +SKIP
    """
    say(env, f"census: {stage}", **{
        q.name: [round(float(v), 3) for v in _np(q.pose.p).reshape(-1)[:3]]
                + [round(common._tilt_deg(q.pose.sp.q), 1)]
        for q in task.props})


def _straight(env, planner, pose, *, stage: str, tries: int = 1):
    """The screw with the torso held, never an RRT path. -1 or the 5-tuple.

    Example:
        >>> res = _straight(env, planner, up, stage="unshelve")   # doctest: +SKIP
    """
    return common.arm_move(env, planner, pose, who=WHO, stage=stage, tries=tries,
                           disable_lift_joint=True, max_knots=1, knot_refuse=True)


def _drive_kw(planner) -> dict:
    """`hold_pose=True` where the solver has it — a loaded drive must not sag.

    Probed rather than assumed so the stub and older trees keep working; the measurement
    behind it is in `drive_straight`'s own docstring (3.9 cm of creep carrying 243 g).

    Example:
        >>> planner.drive_straight(d, v=0.10, **_drive_kw(planner))   # doctest: +SKIP
    """
    if "hold_pose" in inspect.signature(planner.drive_straight).parameters:
        return {"hold_pose": True}
    return {}


def row_neighbours(task, prop):
    """The other props standing in the cabinet row right now, nearest first.

    They are what the fingers have to share 1 cm of air with (see `UNSHELVE_BACK`), and
    the legs that cannot avoid them take them out of the planning world for the stroke
    — honest, because those strokes are straight lines that move AWAY from the
    neighbour, and a straight screw cannot detour through the gap it opens.

    Example:
        >>> with neighbours_touchable(planner, row_neighbours(task, prop)):  # doctest: +SKIP
        ...     res = _straight(env, planner, up, stage="unshelve")
    """
    out = []
    for other in task.props:
        if other is prop:
            continue
        q = _np(other.pose.p).reshape(-1)
        # In the cabinet at all: above the shelf plane and north of its front face.
        if float(q[2]) > task.cfg.shelf_top_z - 0.05 and float(q[1]) > -0.40:
            out.append(other)
    return out


@contextlib.contextmanager
def neighbours_touchable(planner, props):
    """Nest `oracle_common.touchable` over several actors for one stroke."""
    with contextlib.ExitStack() as stack:
        for prop in props:
            stack.enter_context(common.touchable(planner, prop.name))
        yield


def fold_arm(env, planner, task):
    """Un-wind the continuous joints, then fold to rest with the retrieval flow's fold.

    Every transfer has to start from the same posture, and a release leaves the arm
    EXTENDED over the counter. Ducking does not retract it — the duck moves the torso
    and the wrist and nothing else — so the fourth dock of the first full run drove that
    extended hand over a prop already staged on the counter and the torso descent was
    refused with `gripper_link <-> prop_0`. Folding first is what makes a five-transfer
    episode look like five copies of a one-transfer episode.

    Both halves are imported rather than written here, and both for a measured reason:

    - `normalize_continuous_arm_joints` first, because a roll joint sitting at
      `rest ± 2pi` makes the fold's joint line a full visible revolution (K111 measured
      exactly that, `arm_vs_rest 6.283` with the arm physically at rest). This oracle
      folds five times an episode, so the winding is five times as visible.
    - `retr.fold_arm_to_rest` for the fold, because it reads the joint names off the arm
      controller instead of hardcoding them AND parks `wrist_flex_joint` at 1.7. The
      rest keyframe leaves the wrist at 2.077 against a 2.16 stop, and a dock drive from
      there was refused on `joint limit at index [11]` two steps in.

    Example:
        >>> fold_arm(env, planner, task)     # doctest: +SKIP
    """
    common.normalize_continuous_arm_joints(env, planner, task, who=WHO)
    return retr.fold_arm_to_rest(env, planner, task)


def dock_at(env, planner, task, x: float, *, stage: str):
    """Put the base in front of `x`, empty-handed, facing the counter. -1 or the tuple.

    Example:
        >>> res = dock_at(env, planner, task, 2.41, stage="dock at slot 2")  # doctest: +SKIP
    """
    # No fold and no duck. Both were needed when this oracle re-docked before every
    # transfer; with ONE dock they are a round trip to nowhere. The robot spawns with the
    # torso already at its 0.386 stop and the arm at the rest keyframe, and the first
    # thing the first extraction asks for is 0.386 — so ducking to 0.20 and back is the
    # "torso goes down and up again" a viewer sees at the top of the clip, and the fold
    # has nothing to fold.
    dock = np.array([float(x), retr.READY_DOCK_Y, 0.0])
    say(env, stage, dock=[round(float(v), 3) for v in dock])
    res = planner.drive_base(target_pos=dock, target_view_vec=np.array([0.0, 1.0, 0.0]))
    if res != -1:
        planner.planner.update_from_simulation()
        d_dock, dyaw = common.dock_error(task, (float(x), retr.READY_DOCK_Y, np.pi / 2))
        # With ONE dock per episode this is the only place a docking defect can be seen.
        say(env, "parked", d_dock=round(d_dock, 3), dyaw_deg=round(dyaw, 1))
    return res


def torso_to(env, planner, task, rungs, *, label: str):
    """First rung that plans, reached in `TORSO_CHUNK` steps. Returns (res, rung)."""
    amap = task.agent.robot.active_joints_map
    j = amap.get("torso_lift_joint")
    ji = int(np.asarray(j.active_index).reshape(-1)[0]) if j is not None else None
    for torso in rungs:
        here = (float(_np(task.agent.robot.get_qpos()).reshape(-1)[ji])
                if ji is not None else None)
        if here is None or abs(torso - here) <= TORSO_CHUNK:
            waypoints = [torso]
        else:
            n = int(np.ceil(abs(torso - here) / TORSO_CHUNK))
            waypoints = [here + (torso - here) * (k + 1) / n for k in range(n)]
        res = -1
        for w in waypoints:
            res = retr.plan_joints(env, planner, task, {"torso_lift_joint": float(w)},
                                   label=f"{label} {round(float(w), 3)}", line_only=True,
                                   qpos_step=TORSO_LINE_STEP)
            if res == -1:
                break
            planner.planner.update_from_simulation()
            if common.stopped_by_horizon(planner):
                return res, torso
        if res != -1:
            return res, torso
    return -1, None


def grasp_ladder(env, planner, task, prop, *, where: str, lift: float = None):
    """Screw to the side grasp over `SIDE_GRASP_DEPTHS`, close, verify. (res, ok).

    The depth ladder and `GRASP_LIFT` are the retrieval flow's measured ones; the mesh
    is re-read on every rung because a refused close can have moved the prop.
    """
    res = -1
    lift = retr.GRASP_LIFT if lift is None else float(lift)
    for depth in retr.SIDE_GRASP_DEPTHS:
        grasp, _pre = retr.side_grasp_pose(task, depth, lift=lift, obj=prop)
        if grasp is None:
            return fail(env, f"{where}: {prop.name} has no collision mesh"), False
        say(env, f"{where}: grasp stroke", prop=prop.name, depth=depth, lift=lift,
            grasp=[round(float(v), 3) for v in grasp.p])
        res = _straight(env, planner, grasp, stage=f"{where} grasp (depth {depth})")
        if res == -1:
            # The same straight stroke with the TARGET out of the planning world. This
            # is `oracle_common.try_grasp`'s middle rung and it exists because the
            # fingers have to close AROUND the object: mplib sees the approach as
            # `l_gripper_finger_link <-> prop_N` and refuses a grasp that is correct.
            # Measured 2026-09-10 on the four-prop row: that refusal is the whole of the
            # counter-grasp failure class, 4 of 30 episodes.
            with common.touchable(planner, prop.name):
                res = _straight(env, planner, grasp,
                                stage=f"{where} grasp (depth {depth}, target touchable)",
                                tries=2)
        if res != -1 and common.stopped_by_horizon(planner):
            return res, False
        if res == -1:
            planner.planner.update_from_simulation()
            continue
        res = planner.close_gripper(t=12)
        if res != -1 and common.stopped_by_horizon(planner):
            return res, False
        if bool(task.agent.is_grasping(prop).any()):
            say(env, f"{where}: in the gripper", prop=prop.name, depth=depth,
                tilt=round(common._tilt_deg(prop.pose.sp.q), 1))
            planner.planner.update_from_simulation()
            return res, True
        say(env, f"{where}: close missed", prop=prop.name, depth=depth)
        planner.open_gripper()
        planner.planner.update_from_simulation()
    return fail(env, f"{where}: grasp {prop.name}", tried=list(retr.SIDE_GRASP_DEPTHS)), False


def release_and_clear(env, planner, task, prop, place):
    """Open over `place`, detach, lift off, drive the base clear, then tidy the arm.

    Order matters and each step of it was measured. The rise before the base moves is
    the stow flow's cure for the hollow success: a 4.5 cm prop between 10 cm pads has
    2.75 cm a side, and a screw carries a wrist rotation that eats it, so the pads leave
    UPWARD and the base does the horizontal part. The base then retreats BEFORE the
    torso rises or the arm folds, because after a row placement the hand is inside the
    cabinet, where both are refused against its shell — that chain cost 5 of the first
    10 episodes, every one of them as `dock at the slot ... returned -1` two stages
    later, once the un-folded arm made the next dock's rotation sweep hit a door.

    The base always returns to the ONE dock line the episode is run from, so there is no
    per-caller distance to pass.
    """
    res = planner.open_gripper(t=retr.RELEASE_RAMP + 6, ramp=retr.RELEASE_RAMP)
    if res != -1 and common.stopped_by_horizon(planner):
        return res
    common.hold_object_in_planner(env, planner, task, prop, held=False, who=WHO)
    say(env, "released", prop=prop.name,
        at=[round(float(v), 3) for v in _np(prop.pose.p).reshape(-1)[:3]],
        tilt=round(common._tilt_deg(prop.pose.sp.q), 1))
    planner.planner.update_from_simulation()

    tcp = task.agent.tcp.pose.sp
    others = row_neighbours(task, prop)
    for dz in sp.RELEASE_LIFT_RUNGS:
        with common.touchable(planner, prop.name), neighbours_touchable(planner, others):
            up = _straight(env, planner,
                           sapien.Pose(p=[tcp.p[0], tcp.p[1], tcp.p[2] + dz], q=tcp.q),
                           stage=f"hand off {prop.name} ({dz} m)", tries=2)
        if up != -1:
            res = up
            say(env, "hand off", prop=prop.name, rise=dz,
                tilt=round(common._tilt_deg(prop.pose.sp.q), 1))
            break
    else:
        say(env, "no rise planned; backing the base out anyway", prop=prop.name)
    if common.stopped_by_horizon(planner):
        return res
    planner.planner.update_from_simulation()

    # Torso, then base, then fold — and the order is the whole lesson of this stage.
    #
    # The TORSO first, because it is the only one of the three that is safe in both
    # places: after a row placement the hand is inside the cabinet, where the torso is
    # already at its stop and the raise is a no-op; after a counter placement it lifts
    # the hand clear of props whose tops stand 12 cm above the surface.
    #
    # The BASE second. Moving it before the raise drags the still-low arm across the
    # counter: measured, that knocked the TARGET prop flat and the episode came back
    # `target_on_a_slot=False tilt_max=90.0` with every transfer otherwise clean.
    #
    # The FOLD last, and only once the base is back on the dock line, because folding
    # from inside the cabinet is refused against its shell.
    up, torso = torso_to(env, planner, task, SHELF_TORSO_RUNGS, label="torso up to")
    if up != -1:
        res = up
        if common.stopped_by_horizon(planner):
            return res
    planner.planner.update_from_simulation()

    base_y = float(_np(task.agent.base_link.pose.p).reshape(-1)[1])
    to_line = float(retr.READY_DOCK_Y - base_y)
    back = planner.drive_straight(to_line, v=0.10)
    if back != -1:
        res = back
        if common.stopped_by_horizon(planner):
            return res
    planner.planner.update_from_simulation()

    # No fold. It existed to make the NEXT dock's rotation sweep clear the doors, and
    # there is no next dock: the base stays on this line for the whole episode.
    # `ready_by_line` starts the following transfer from wherever the arm is and picks
    # the nearest IK branch, so a fold buys nothing and costs a visible trip to rest —
    # and it refused often enough to fold between some transfers and not others, which
    # is what made the clip look inconsistent.
    say(env, "cleared", prop=prop.name, to_line=round(to_line, 3), torso=torso)
    return res


def extract_one(env, planner, task, prop, slot_xyz):
    """Shelf -> counter slot: the retrieval straight flow, aimed at a chosen slot.

    `(res, ok)`. `ok` False with -1 is a refusal; with a 5-tuple it is a miss.
    """
    res, torso = torso_to(env, planner, task, SHELF_TORSO_RUNGS, label="torso up to")
    if res == -1:
        return fail(env, "raise the torso to the shelf"), False
    if common.stopped_by_horizon(planner):
        return res, False
    say(env, "torso up", torso=torso)

    # The posture the drive-in ends at, with the grasp screw pre-checked from there.
    base_p = _np(task.agent.base_link.pose.p).reshape(-1)
    drive = float(retr.WORK_DOCK_Y - base_p[1])
    chosen_lift = None
    for lift in GRASP_LIFTS:
        grasp, pre = retr.side_grasp_pose(task, retr.SIDE_GRASP_DEPTHS[0],
                                          lift=lift, obj=prop)
        if grasp is None:
            return fail(env, f"{prop.name} has no collision mesh"), False
        ready = sapien.Pose(p=[pre.p[0], pre.p[1] - drive, pre.p[2]], q=pre.q)
        say(env, "ready posture", prop=prop.name, drive=round(drive, 3), lift=lift)
        res = retr.ready_by_line(env, planner, task, ready, drive=drive, grasp=grasp,
                                 label=f"ready posture ({prop.name}, lift {lift})",
                                 touch_needle=prop.name, rank=roll_rank(task))
        if res != -1 and common.stopped_by_horizon(planner):
            return res, False
        if res != -1:
            chosen_lift = lift
            break
        planner.planner.update_from_simulation()
    if chosen_lift is None:
        return fail(env, f"ready posture for {prop.name}", lifts=list(GRASP_LIFTS)), False
    planner.planner.update_from_simulation()

    say(env, "drive into the cabinet", distance=round(drive, 3))
    res = planner.drive_straight(drive, v=0.10)
    if res != -1 and common.stopped_by_horizon(planner):
        return res, False
    if res == -1:
        return fail(env, "drive into the cabinet"), False
    planner.planner.update_from_simulation()

    res, ok = grasp_ladder(env, planner, task, prop, where="shelf", lift=chosen_lift)
    if not ok:
        return res, False
    common.hold_object_in_planner(env, planner, task, prop, held=True, who=WHO)

    tcp = task.agent.tcp.pose.sp
    others = row_neighbours(task, prop)
    # Up AND back: the fingers leave the neighbour's face from the first knot. The
    # neighbours come out of the planning world for this one straight stroke; see
    # `row_neighbours` for why that is honest and `UNSHELVE_BACK` for the measurement.
    up = sapien.Pose(p=[tcp.p[0], tcp.p[1] - UNSHELVE_BACK, tcp.p[2] + UNSHELVE_DZ], q=tcp.q)
    say(env, "unshelve", prop=prop.name, back=UNSHELVE_BACK, up=UNSHELVE_DZ,
        neighbours=[o.name for o in others])
    with neighbours_touchable(planner, others):
        res = _straight(env, planner, up, stage="unshelve", tries=2)
        if res == -1:
            plain = sapien.Pose(p=[tcp.p[0], tcp.p[1], tcp.p[2] + UNSHELVE_DZ], q=tcp.q)
            res = _straight(env, planner, plain, stage="unshelve (straight up)", tries=2)
    if res != -1 and common.stopped_by_horizon(planner):
        return res, False
    if res == -1:
        return fail(env, f"unshelve {prop.name}"), False
    if not bool(task.agent.is_grasping(prop).any()):
        say(env, f"MISSED: {prop.name} left the gripper at the unshelve")
        return res, False
    planner.planner.update_from_simulation()

    # Carry out: the base backs off until the prop stands over the counter band. Loaded,
    # so the hold is latched (see `_drive_kw`).
    prop_y = float(_np(prop.pose.p).reshape(-1)[1])
    back = float(prop_y - float(np.asarray(slot_xyz).reshape(3)[1]))
    say(env, "carry out", distance=round(back, 3))
    res = planner.drive_straight(-back, v=0.10, **_drive_kw(planner))
    if res != -1 and common.stopped_by_horizon(planner):
        return res, False
    if res == -1:
        return fail(env, "carry out of the cabinet"), False
    if not bool(task.agent.is_grasping(prop).any()):
        say(env, f"MISSED: {prop.name} left the gripper on the way out")
        return res, False
    planner.planner.update_from_simulation()

    # ACROSS to the slot's column, before the torso comes down.
    #
    # Two constraints pull against each other and `CROSS_TORSO_RUNGS` is where they
    # meet. High is safe — a prop carried at the shelf height clears the ones already
    # standing on the slots, whose tops are 12 cm above the counter — but the far slot
    # is 36 cm sideways and 0.82 m forward, 0.90 m of radius, and at the carry-out
    # height the IK simply has no solution (measured seed 3: `IK Failed` on the line,
    # the screw and the RRT alike). Lower is reachable but eventually sweeps the staged
    # props. So the cross walks DOWN a ladder and stops at the first rung that solves,
    # and the ladder stops at 0.20, where the carried prop's base still rides 1.25 —
    # 21 cm over the tallest thing on the counter.
    #
    # Coming south of the counter first and crossing down there was tried and is worse:
    # the move back onto the slot line was refused on 3 of 10 seeds (6/10 overall).
    slot = np.asarray(slot_xyz, dtype=np.float64).reshape(3)
    crossed = False
    for torso in CROSS_TORSO_RUNGS:
        if torso is not None:
            lowered = retr.plan_joints(env, planner, task, {"torso_lift_joint": torso},
                                       label=f"torso to {torso} for the cross",
                                       line_only=True, qpos_step=TORSO_LINE_STEP)
            if lowered == -1:
                continue
            res = lowered
            if common.stopped_by_horizon(planner):
                return res, False
            planner.planner.update_from_simulation()
        tcp = task.agent.tcp.pose.sp
        across = sapien.Pose(p=[float(slot[0]), float(slot[1]), float(tcp.p[2])], q=tcp.q)
        say(env, "across to the slot's column", prop=prop.name, torso=torso,
            dx=round(float(slot[0]) - float(tcp.p[0]), 3))
        # A joint LINE to the nearest IK branch, not a screw: this is the longest
        # Cartesian stroke of the episode and `plan_screw` is a least-norm Jacobian walk
        # that spreads it over every joint, which is the arm rotation the owner saw.
        res = retr.ready_by_line(env, planner, task, across, drive=0.0, grasp=across,
                                 label=f"across to the slot ({prop.name}, torso {torso})",
                                 slack=(0.0, 0.0, -0.005), touch_needle=None,
                                 rank=roll_rank(task))
        if res == -1:
            res = _straight(env, planner, across, stage="across to the slot", tries=2)
        if res != -1 and common.stopped_by_horizon(planner):
            return res, False
        if res != -1:
            crossed = True
            break
        planner.planner.update_from_simulation()
    if not crossed:
        return fail(env, f"cross to the slot column for {prop.name}",
                    rungs=list(CROSS_TORSO_RUNGS)), False
    if not bool(task.agent.is_grasping(prop).any()):
        say(env, f"MISSED: {prop.name} left the gripper crossing to the slot")
        return res, False
    planner.planner.update_from_simulation()

    down, torso = torso_to(env, planner, task, COUNTER_TORSO_RUNGS, label="torso down to")
    if down != -1:
        res = down
        if common.stopped_by_horizon(planner):
            return res, False
        say(env, "torso down", torso=torso)
    if not bool(task.agent.is_grasping(prop).any()):
        say(env, f"MISSED: {prop.name} left the gripper on the way down")
        return res, False

    place = sp.place_pose_for(task, obj=prop, target=slot_xyz, compensate_xy=True)
    if place is None:
        return fail(env, f"{prop.name} has no collision mesh"), False
    say(env, "descend to the slot", prop=prop.name,
        place=[round(float(v), 3) for v in place.p])
    res = _straight(env, planner, place, stage="descend to the slot", tries=2)
    if res == -1:
        res = common.arm_move(env, planner, place, who=WHO,
                              stage="descend to the slot (any plan)", tries=2)
    if res != -1 and common.stopped_by_horizon(planner):
        return res, False
    if res == -1:
        return fail(env, f"stand {prop.name} on the slot"), False
    planner.planner.update_from_simulation()

    res = release_and_clear(env, planner, task, prop, place)
    return res, not common.stopped_by_horizon(planner)


def restore_one(env, planner, task, prop, row_xyz):
    """Counter slot -> its row slot: the stow flow, aimed at a chosen row point."""
    # ACROSS FIRST, at the height the previous clearing left, and only then down. Going
    # straight to the pre-grasp posture sweeps the hand sideways across the counter AT
    # grasp height, and a line that passes a standing prop by a centimetre is one the
    # tracking error can touch: measured on seed 6, the prop the arm was on its way to
    # TIPPED OVER during the approach, and the grasp that followed aimed at a mid-height
    # of 0.942 — a lying prop — and drove the palm into it. The extraction already
    # crosses high and descends; this is the same move on the way back.
    p0 = _np(prop.pose.p).reshape(-1)[:3]
    moved = -1
    for torso in tuple(CROSS_SAFE_TORSO) + (None,):
        if torso is not None:
            lowered = retr.plan_joints(env, planner, task, {"torso_lift_joint": torso},
                                       label=f"torso to {torso} for the cross",
                                       line_only=True, qpos_step=TORSO_LINE_STEP)
            if lowered == -1:
                continue
            res = lowered
            if common.stopped_by_horizon(planner):
                return res, False
            planner.planner.update_from_simulation()
        tcp0 = task.agent.tcp.pose.sp
        over = sapien.Pose(p=[float(p0[0]), float(tcp0.p[1]), float(tcp0.p[2])], q=tcp0.q)
        say(env, "across to the prop's column", prop=prop.name, torso=torso,
            dx=round(float(p0[0]) - float(tcp0.p[0]), 3),
            tcp_z=round(float(tcp0.p[2]), 3))
        moved = retr.ready_by_line(env, planner, task, over, drive=0.0, grasp=over,
                                   label=f"across to {prop.name}'s column (torso {torso})",
                                   slack=(0.0, 0.0, -0.005), touch_needle=None,
                                   rank=roll_rank(task))
        if moved == -1:
            moved = _straight(env, planner, over, stage="across to the prop's column",
                              tries=2)
        if moved != -1 and common.stopped_by_horizon(planner):
            return moved, False
        if moved != -1:
            res = moved
            planner.planner.update_from_simulation()
            break
        planner.planner.update_from_simulation()
    if moved == -1:
        say(env, "cross refused at every height; approaching from where we are",
            prop=prop.name, rungs=list(CROSS_SAFE_TORSO))

    # THREE ladders, outermost first, each bought by a measured failure: the TORSO,
    # because the props already staged are obstacles for the reach to the ones behind
    # them (seed 6); the LIFT, because the wrist hangs 6.7 cm under the hand and the
    # counter is right below (seed 9: 75 IK solutions, all refused against the counter);
    # the DEPTH, the retrieval flow's own. The FOLD is a recovery outside all three, not
    # a routine — folding between transfers was removed, but a posture the previous
    # release happened to leave can still refuse.
    res, ok = -1, False
    chosen = None
    for attempt in ("as we are", "after a fold"):
        if attempt != "as we are":
            say(env, "counter grasp refused; folding to rest and trying once more",
                prop=prop.name)
            if fold_arm(env, planner, task) == -1:
                break
            planner.planner.update_from_simulation()
        for torso in COUNTER_GRASP_TORSO_RUNGS:
            lowered = retr.plan_joints(env, planner, task, {"torso_lift_joint": torso},
                                       label=f"torso to {torso} for the counter grasp",
                                       line_only=True, qpos_step=TORSO_LINE_STEP)
            if lowered == -1:
                continue
            res = lowered
            if common.stopped_by_horizon(planner):
                return res, False
            planner.planner.update_from_simulation()
            for lift in GRASP_LIFTS:
                for depth in retr.SIDE_GRASP_DEPTHS:
                    grasp, pre = retr.side_grasp_pose(task, depth, lift=lift, obj=prop)
                    if grasp is None:
                        return fail(env, f"{prop.name} has no collision mesh"), False
                    res = retr.ready_by_line(
                        env, planner, task, pre, drive=0.0, grasp=grasp,
                        label=f"grasp posture ({prop.name}, torso {torso}, lift {lift}, "
                              f"depth {depth})",
                        touch_needle=prop.name, rank=roll_rank(task))
                    if res != -1 and common.stopped_by_horizon(planner):
                        return res, False
                    planner.planner.update_from_simulation()
                    if res != -1:
                        chosen = (torso, lift)
                        break
                if chosen is not None:
                    break
            if chosen is None:
                continue
            say(env, "counter grasp posture", prop=prop.name, torso=chosen[0],
                lift=chosen[1])
            res, ok = grasp_ladder(env, planner, task, prop, where="counter",
                                   lift=chosen[1])
            if ok:
                break
            if res != -1 and common.stopped_by_horizon(planner):
                return res, False
            chosen = None
        if ok:
            break
    if not ok:
        return fail(env, f"take {prop.name} off the counter",
                    torsos=list(COUNTER_GRASP_TORSO_RUNGS),
                    lifts=list(GRASP_LIFTS)), False
    common.hold_object_in_planner(env, planner, task, prop, held=True, who=WHO)

    tcp = task.agent.tcp.pose.sp
    up = sapien.Pose(p=[tcp.p[0], tcp.p[1], tcp.p[2] + UNSHELVE_DZ], q=tcp.q)
    res = _straight(env, planner, up, stage="lift off the counter")
    if res == -1:
        res = common.arm_move(env, planner, up, who=WHO,
                              stage="lift off the counter (any plan)", tries=2)
    if res != -1 and common.stopped_by_horizon(planner):
        return res, False
    if res == -1:
        return fail(env, f"lift {prop.name} off the counter"), False
    planner.planner.update_from_simulation()

    # The stow flow's rise, imported whole: it tries the same ladder and, when nothing
    # plans, retreats along the approach axis and tries it again — the difference
    # between a hard -1 and a second chance. Its body never names the object.
    res = sp.rise_with_the_cup(env, planner, task)
    if res == -1:
        return res, False
    if common.stopped_by_horizon(planner):
        return res, False
    say(env, "torso up", tilt=round(common._tilt_deg(prop.pose.sp.q), 1))
    if not bool(task.agent.is_grasping(prop).any()):
        say(env, f"MISSED: {prop.name} left the gripper on the way up")
        return res, False

    entry = sp.place_pose_for(task, rise=sp.ENTRY_RISE, obj=prop, target=row_xyz,
                              compensate_xy=True)
    if entry is None:
        return fail(env, f"{prop.name} has no collision mesh"), False
    base_p = _np(task.agent.base_link.pose.p).reshape(-1)
    drive = float(retr.WORK_DOCK_Y - base_p[1])
    ready = sapien.Pose(p=[entry.p[0], entry.p[1] - drive, entry.p[2]], q=entry.q)
    say(env, "entry posture", prop=prop.name, drive=round(drive, 3),
        entry=[round(float(v), 3) for v in entry.p])
    # The held prop stays in the planning world for the descent pre-check: it is exactly
    # what must not hit the shelf.
    res = retr.ready_by_line(env, planner, task, ready, drive=drive, grasp=entry,
                             label=f"entry posture ({prop.name})",
                             slack=sp.PLACE_PROBE_SLACK, touch_needle=None)
    if res != -1 and common.stopped_by_horizon(planner):
        return res, False
    if res == -1:
        return fail(env, f"entry posture for {prop.name}"), False
    planner.planner.update_from_simulation()

    # Let the arm settle under the load and correct what the joint line could not see.
    # Measured on this exact 4.5x4.5x12 cm 243 g prop (`cabinet_stow_planner`'s
    # ENTRY_SAG_TOL): the entry came in 4.4 cm low against 2 mm with an 8 g cup, the
    # object crossed the shelf's edge with 4 mm to spare, and on one seed it arrived
    # tilted 21.6 deg in the fingers and fell flat when they opened.
    settle = planner.idle_steps(t=sp.ENTRY_SETTLE_STEPS)
    if settle != -1:
        res = settle
        if common.stopped_by_horizon(planner):
            return res, False
    tcp_now = task.agent.tcp.pose.sp
    sag = float(ready.p[2]) - float(tcp_now.p[2])
    if abs(sag) > sp.ENTRY_SAG_TOL:
        say(env, "the entry sagged under the load; correcting", prop=prop.name,
            sag=round(sag, 4))
        fix = _straight(env, planner, ready, stage="correct the entry height", tries=2)
        if fix != -1:
            res = fix
            if common.stopped_by_horizon(planner):
                return res, False
            planner.planner.update_from_simulation()
        else:
            say(env, "entry correction refused; driving in as we are")

    say(env, "drive into the cabinet, loaded", distance=round(drive, 3))
    res = planner.drive_straight(drive, v=0.10, **_drive_kw(planner))
    if res != -1 and common.stopped_by_horizon(planner):
        return res, False
    if res == -1:
        return fail(env, "drive into the cabinet, loaded"), False
    if not bool(task.agent.is_grasping(prop).any()):
        say(env, f"MISSED: {prop.name} left the gripper on the way in")
        return res, False
    planner.planner.update_from_simulation()

    place = sp.place_pose_for(task, obj=prop, target=row_xyz, compensate_xy=True)
    others = row_neighbours(task, prop)
    say(env, "descend onto the row slot", prop=prop.name,
        place=[round(float(v), 3) for v in place.p],
        neighbours=[o.name for o in others])
    with neighbours_touchable(planner, others):
        res = _straight(env, planner, place, stage="descend onto the row slot", tries=2)
        if res == -1:
            # The stow flow's middle rung: the last 5 mm inside a box is where a screw
            # refuses on clearance, and an RRT with a prop in the fingers inside a
            # cabinet is the worst plan in this file. Defer it by one rung.
            higher = sapien.Pose(p=[place.p[0], place.p[1], place.p[2] + 0.005], q=place.q)
            res = _straight(env, planner, higher,
                            stage="descend onto the row slot (5 mm up)", tries=2)
        if res == -1:
            res = common.arm_move(env, planner, place, who=WHO,
                                  stage="descend onto the row slot (any plan)", tries=2)
    if res != -1 and common.stopped_by_horizon(planner):
        return res, False
    if res == -1:
        return fail(env, f"stand {prop.name} back in the row"), False
    planner.planner.update_from_simulation()

    # Out of the cabinet on the drive it came in on, before anything else moves.
    res = release_and_clear(env, planner, task, prop, place)
    return res, not common.stopped_by_horizon(planner)


def choose_assignment(task, blind: bool, rng):
    """Which staged prop goes into which row slot — the ONE privileged read.

    Sighted: the episode's own permutation. Blind: a draw over the same two slots, which
    is the memoryless arm and must land on the 0.5 floor over a seed sweep.

    Returns a list of `(prop_index, slot_index)`, deepest empty slot first, because
    nothing can be placed behind an occupied slot.

    Example:
        >>> plan = choose_assignment(task, blind=False, rng=rng)   # doctest: +SKIP
    """
    slots = _np(task.original_slots).reshape(-1)
    deep = int(task.cfg.n_props - 1)
    movables = [i for i in range(len(slots)) if int(slots[i]) != deep]
    targets = sorted((int(slots[i]) for i in movables), reverse=True)
    if blind:
        order = list(rng.permutation(len(movables)))
        return [(movables[order[k]], targets[k]) for k in range(len(movables))]
    return [(i, int(slots[i])) for i in sorted(movables, key=lambda i: -int(slots[i]))]


def solve(env, seed=None, debug=False, vis=False, blind=False,
          planner_factory=common.default_planner_factory):
    """Clear the row onto the counter, keep the deepest prop, put the rest back.

    Example:
        >>> res = solve(env, seed=0)   # doctest: +SKIP
    """
    obs, info = env.reset(seed=seed)
    if seed is not None:
        seed_everything(seed)
    assert env.unwrapped.control_mode in (
        "pd_joint_pos", "pd_joint_pos_vel", "pd_joint_delta_pos"
    ), env.unwrapped.control_mode
    task = env.unwrapped
    planner = planner_factory(env, debug, vis)
    rng = np.random.default_rng((0 if seed is None else int(seed)) + COIN_OFFSET)

    props = list(task.props)
    slot_pts = _np(task._slot_points).reshape(-1, 4, 3)[0]
    row_pts = _np(task._row_points).reshape(-1, task.cfg.n_props, 3)[0]
    say(env, "episode", blind=bool(blind),
        slots=[round(float(v), 3) for v in slot_pts[:, 0]],
        row_x=round(float(row_pts[0][0]), 3))

    # ONE dock for the whole episode, at the row's column. Everything after it moves on
    # straight lines: the base only ever drives along its own axis, into the cabinet and
    # back out, and the arm reaches sideways to the counter slots. `drive_base` turns the
    # base, so calling it per transfer is what made the base hop and spin.
    # Retrieval ducks the torso before its dock drive; this does not, and the difference
    # is that this dock is the FIRST thing the episode does — the robot is already in the
    # posture retrieval ducks into position for. `MIKASA_DR1_DUCK=1` puts it back.
    if os.environ.get("MIKASA_DR1_DUCK", "0") == "1":
        res = retr.plan_joints(env, planner, task,
                               {"torso_lift_joint": retr.TORSO_DRIVE,
                                "wrist_flex_joint": 1.7},
                               label="duck the torso")
        if res != -1 and common.stopped_by_horizon(planner):
            return res
        if res == -1:
            return fail(env, "duck the torso for the drive")
        planner.planner.update_from_simulation()

    row_x = float(row_pts[0][0])
    res = dock_at(env, planner, task, row_x, stage="dock at the row (once)")
    if res != -1 and common.stopped_by_horizon(planner):
        return res
    if res == -1:
        return fail(env, "dock at the row")

    # --- take the row apart, front first (read off the poses, not the answer) ---
    order = sorted(range(len(props)), key=lambda i: float(_np(props[i].pose.p).reshape(-1)[1]))
    free = list(range(4))
    staged = {}
    for prop_i in order:
        slot_k = int(rng.choice(free))
        free.remove(slot_k)
        staged[prop_i] = slot_k
        say(env, "extract", prop=props[prop_i].name, to_slot=slot_k,
            where={q.name: [round(float(v), 3) for v in _np(q.pose.p).reshape(-1)[:3]]
                   for q in props})
        res, ok = extract_one(env, planner, task, props[prop_i], slot_pts[slot_k])
        if not ok:
            return res
    say(env, "row cleared", staged={props[i].name: k for i, k in staged.items()})

    # --- put the two movables back, deepest empty slot first -------------------
    for prop_i, slot_idx in choose_assignment(task, blind, rng):
        say(env, "restore", prop=props[prop_i].name, to_row_slot=slot_idx)
        res, ok = restore_one(env, planner, task, props[prop_i], row_pts[slot_idx])
        if not ok:
            return res

    settled = planner.idle_steps(t=retr.SETTLE_STEPS)
    if settled == -1:
        return fail(env, "settle wait")
    res = settled
    info = res[-1]
    ok = _b(info, "success")
    tilt_max = float(_np(info["tilt_max_deg"]).reshape(-1)[0])
    if tilt_max > sp.TILT_WARN_DEG:
        # The line a sweep greps to tell an honest run from a hollow one.
        say(env, "KNOCKED OVER", tilt_deg=round(tilt_max, 1),
            props=[round(float(v), 1) for v in _np(info["prop_tilt_deg"]).reshape(-1)])
    say(env, "episode over", success=ok,
        restored=_np(info["restored"]).reshape(-1).tolist(),
        target_on_a_slot=_b(info, "target_on_a_slot"),
        wrong_assign=_b(info, "wrong_assign"),
        target_returned=_b(info, "target_returned"),
        two_in_slot=_b(info, "two_in_slot"),
        tilt_max=round(float(_np(info["tilt_max_deg"]).reshape(-1)[0]), 1),
        steps=int(getattr(planner, "elapsed_steps", -1)))
    return res


if __name__ == "__main__":
    import argparse

    import gymnasium as gym

    import mani_skill  # noqa: F401  (registers my_scenes and utils.mikasa)

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blind", action="store_true")
    args = ap.parse_args()
    e = gym.make("MikasaDepthRecall-v1", num_envs=1, robot_uids="ds_fetch",
                 control_mode="pd_joint_pos", obs_mode="state", sim_backend="cpu")
    out = solve(e, seed=args.seed, blind=args.blind)
    print("no_plan" if isinstance(out, int) else f"success={_b(out[-1], 'success')}")
    e.close()
