"""Shelf-placement helpers used by the DepthRecall-v1 oracle.

Extracted from planners/cabinet_stow_planner.py at pa40l/ManiSkill
3cba2399851828645788431536e85f8efd25e82d. No environment is registered here.
The movement helpers and their constants are preserved from that source.
"""
from __future__ import annotations

import os

import numpy as np
import sapien

from planners import cabinet_retrieval_planner as retr
from utils.mikasa_oracle.planners import oracle_common as common

WHO = "cabinet_stow_planner"

GRASP_TORSO_RUNGS = (0.0, 0.10, 0.20)

RISE_TORSO_RUNGS = (retr.READY_TORSO, 0.34, 0.30)

ENTRY_RISE = float(os.environ.get("MIKASA_STOW_ENTRY_RISE", "0.04"))

ENTRY_SETTLE_STEPS = int(os.environ.get("MIKASA_STOW_ENTRY_SETTLE", "6"))

PRE_RISE_BACK = float(os.environ.get("MIKASA_STOW_PRE_RISE_BACK", "0.12"))

PLACE_DROP = float(os.environ.get("MIKASA_STOW_PLACE_DROP", "0.008"))

RELEASE_LIFT_RUNGS = tuple(
    float(v) for v in os.environ.get("MIKASA_STOW_RELEASE_LIFT", "0.06,0.045,0.03").split(",")
)

TILT_WARN_DEG = 15.0

ENTRY_SAG_TOL = float(os.environ.get("MIKASA_STOW_ENTRY_SAG_TOL", "0.005"))

PLACE_PROBE_SLACK = (0.0, 0.0, -0.005)

def say(env, stage: str, **extra):
    """Trace one stage to stdout and to the episode log. See `oracle_common.say`."""
    return common.say(env, WHO, stage, **extra)

def fail(env, stage: str, **extra):
    """Print the refusal and return -1 — the no-plan sentinel of the contract."""
    return common.fail(env, WHO, stage, **extra)

def _np(x) -> np.ndarray:
    """A torch tensor or numpy array as numpy, on the host."""
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

def _straight_move(env, planner, pose, *, stage: str, tries: int = 1):
    """`arm_move` restricted to the straight channels: the screw with the torso held,
    never an RRT path. -1 or the 5-tuple.

    Args:
        pose: the TCP target.
        stage: what to call this leg in the trace.

    Returns:
        -1 when nothing planned (nothing is stepped), else the gym 5-tuple.

    Example:
        >>> res = _straight_move(env, planner, up, stage="lift off the counter")  # doctest: +SKIP
    """
    return common.arm_move(env, planner, pose, who=WHO, stage=stage, tries=tries,
                           disable_lift_joint=True, max_knots=1, knot_refuse=True)

def place_pose_for(task, *, rise: float = 0.0, obj=None, target=None,
                   compensate_xy: bool = False):
    """The TCP pose that stands the HELD cup on `task.place_target`, `rise` metres high.

    The arithmetic is retrieval's descent (`take_and_place_straight`), unchanged: keep
    the hand's current orientation, snap x and y to the target, and lower z by however
    much the cup's LIVE mesh bottom overshoots the target plane. Reading the live mesh
    rather than a cached rest-lift is what makes it correct for a cup that is tilted in
    the fingers.

    Args:
        task: the unwrapped env.
        rise: metres above the final placement (the drive-in clearance).
        obj: the held actor; None = `task.cup`.
        target: the (x, y, z) the object's BASE must end on; None = `task.place_target`.
        compensate_xy: aim so the OBJECT lands on the target, not the TCP.

            Default False, which is this family's measured behaviour and correct for a
            cup grasped about its axis. It is wrong for anything that hangs off-axis in
            the fingers: measured 2026-09-10 on `MikasaDepthRecall-v1`, a prop aimed at
            a row slot came to rest 1.7-2.9 cm away from it, against a 3.5 cm slot
            tolerance — inside, but with almost nothing left. With this on, the live
            offset between the object and the TCP is subtracted from the goal, which is
            what `depth_recall_planner.seat_prop` has always done (W19).

    Returns:
        `sapien.Pose` for the TCP, or None when the cup has no collision mesh.

    Example:
        >>> place = place_pose_for(task)                      # doctest: +SKIP
        >>> entry = place_pose_for(task, rise=ENTRY_RISE)     # doctest: +SKIP
    """
    obj = task.cup if obj is None else obj
    mesh = obj.get_first_collision_mesh(to_world_frame=True)
    if mesh is None:
        return None
    bottom = float(np.asarray(mesh.bounds)[0][2])
    target = (_np(task.place_target).reshape(-1, 3)[0] if target is None
              else np.asarray(target, dtype=np.float64).reshape(3))
    tcp = task.agent.tcp.pose.sp
    drop = bottom - (float(target[2]) + PLACE_DROP)
    x, y = float(target[0]), float(target[1])
    if compensate_xy:
        off = _np(obj.pose.p).reshape(-1)[:3] - np.asarray(tcp.p, dtype=np.float64).reshape(3)
        x -= float(off[0])
        y -= float(off[1])
    return sapien.Pose(p=[x, y, float(tcp.p[2]) - drop + float(rise)], q=tcp.q)

def rise_with_the_cup(env, planner, task):
    """The loaded torso rise toward the shelf: the rung ladder, then a retreat and
    the ladder again. -1 or the 5-tuple.

    This is the stage that has no counterpart in retrieval (there the raise happens
    with an empty hand), so it is also the one that carries the fallback.
    """
    for attempt in ("as grasped", "after a retreat"):
        for torso in RISE_TORSO_RUNGS:
            res = retr.plan_joints(env, planner, task, {"torso_lift_joint": torso},
                                   label=f"raise the torso to {torso}", line_only=True)
            if res != -1 and common.stopped_by_horizon(planner):
                return res
            if res != -1:
                say(env, "torso up", torso=torso, how=attempt)
                planner.planner.update_from_simulation()
                return res
        if attempt != "as grasped":
            break
        # Nothing planned from where the grasp left the hand. Withdraw along the
        # approach axis — south, away from the counter — and try the ladder again.
        tcp = task.agent.tcp.pose.sp
        back = sapien.Pose(p=[tcp.p[0], tcp.p[1] - PRE_RISE_BACK, tcp.p[2]], q=tcp.q)
        say(env, "the loaded rise refused; retreating first", back=PRE_RISE_BACK)
        moved = _straight_move(env, planner, back, stage="retreat before the rise", tries=2)
        if moved != -1 and common.stopped_by_horizon(planner):
            return moved
        if moved == -1:
            return fail(env, "retreat before the rise")
        planner.planner.update_from_simulation()
    return fail(env, "raise the torso with the cup", rungs=list(RISE_TORSO_RUNGS))
