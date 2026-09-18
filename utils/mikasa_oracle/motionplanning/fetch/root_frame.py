"""The planning world's robot frame: the robot's world pose folded into its root joints (K53).

Why this exists (K53 / D13, docs/lab-journal.md 2026-08-18 T5)
----------------------------------------------------------------
mplib 0.2.1 draws an attached object at `link_pose_in_base_frame * attached_pose`
and captures `attached_pose` as `link_pose_in_base_frame^-1 * object_world_pose`.
The robot's own collision links get the articulation's base pose folded in
(`base_pose * link_pose`); the attached body does not. So the attachment is exact
at the configuration it was captured in and wrong everywhere else — the held object
orbits *the world origin* instead of the hand as soon as a link rotates: measured
3.296 m from the hand at shoulder_pan +90 deg with the burner's robot standing at
(1.92, −1.40, yaw 90 deg). No pose we could hand `attach_object` fixes that: mplib
composes it in the base-frame link, and a constant transform in that frame is right
in one configuration only unless the base pose is the identity.

So the base pose is made the identity. Fetch's kinematic chain starts with three
planar root joints — `root_x_axis_joint`, `root_y_axis_joint` (prismatic) and
`root_z_rotation_joint` (revolute about z) — and the base link's world pose is
`B * Trans(x, y) * RotZ(yaw)`. For a planar `B = Trans(bx, by, 0) * RotZ(byaw)`
that is `Trans(bx + R(byaw)[x, y]) * RotZ(byaw + yaw)`: the same configuration,
expressed as root-joint values against an identity base. `SapienPlanningWorldV2`
keeps the planned articulation that way at every sync, and `SapienPlannerV2` folds
every qpos it hands mplib and unfolds every plan it gets back, so nothing outside
`utils.py` sees the change — except that an attached body is now drawn where the
hand is, in every configuration.

Everything here is pure numpy (no mplib), so it is unit-tested on a Mac
(`tests/test_root_frame.py`) against a fake forward kinematics.

Example:
    >>> import numpy as np
    >>> base_p, base_q = [1.92, -1.40, 0.0], [0.7071068, 0, 0, 0.7071068]   # yaw +90 deg
    >>> q = fold_root(base_p, base_q, np.zeros(15))
    >>> np.round(q[:3], 4).tolist()
    [1.92, -1.4, 1.5708]
    >>> np.allclose(unfold_root(base_p, base_q, q, ref_yaw=0.0), np.zeros(15))
    True
    >>> is_planar_root(["a_root_x_axis_joint", "a_root_y_axis_joint", "a_root_z_rotation_joint", "torso"])
    True
"""

from __future__ import annotations

import math

import numpy as np

#: The three joints, in chain order, that a planar mobile base has at the root of its
#: kinematic chain (ManiSkill's Fetch/ds_fetch): x, y prismatic then yaw about z.
ROOT_JOINT_SUFFIXES = ("root_x_axis_joint", "root_y_axis_joint", "root_z_rotation_joint")


def is_planar_root(joint_names) -> bool:
    """True when the first three joints are the planar root joints, in chain order.

    Args:
        joint_names: the articulation's active joint names in user order.

    Example:
        >>> is_planar_root(["s0_root_x_axis_joint", "s0_root_y_axis_joint", "s0_root_z_rotation_joint"])
        True
        >>> is_planar_root(["panda_joint1", "panda_joint2", "panda_joint3"])
        False
    """
    names = list(joint_names)
    return len(names) >= 3 and all(str(n).endswith(s) for n, s in zip(names[:3], ROOT_JOINT_SUFFIXES))


def yaw_of_quat(q) -> float:
    """Yaw (rad) of a wxyz quaternion.

    Example:
        >>> round(yaw_of_quat([0.7071068, 0, 0, 0.7071068]), 4)
        1.5708
    """
    w, x, y, z = (float(v) for v in np.asarray(q, dtype=np.float64).reshape(4))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def planar_base(base_p, base_q, tol: float = 1e-6) -> tuple[float, float, float]:
    """`(bx, by, byaw)` of a base pose that is a planar (z = 0, yaw-only) transform.

    Raises ValueError otherwise: the fold cannot express a base height or a tilt in
    the three planar root joints, and a residual base pose would bring the phantom
    back (a z offset alone displaces a held object by `2·sin(θ/2)·z` for a rotation
    θ about a horizontal axis — the pour tilt).

    Args:
        base_p, base_q: the world pose of the articulation's root, wxyz quaternion.
        tol: tolerance on the height and on the roll/pitch quaternion components.

    Example:
        >>> tuple(round(v, 4) for v in planar_base([1.92, -1.40, 0.0], [0.7071068, 0, 0, 0.7071068]))
        (1.92, -1.4, 1.5708)
        >>> planar_base([0, 0, 0.1], [1, 0, 0, 0])  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: root pose is not planar
    """
    p = np.asarray(base_p, dtype=np.float64).reshape(3)
    q = np.asarray(base_q, dtype=np.float64).reshape(4)
    if abs(p[2]) > tol or abs(q[1]) > tol or abs(q[2]) > tol:
        raise ValueError(
            f"root pose is not planar (z={p[2]:.4g}, roll/pitch quat components "
            f"{q[1]:.3g}/{q[2]:.3g}); the base pose cannot be folded into the root joints"
        )
    return float(p[0]), float(p[1]), yaw_of_quat(q)


def _wrap_pi(a: float) -> float:
    return float((a + math.pi) % (2.0 * math.pi) - math.pi)


def fold_root(base_p, base_q, qpos) -> np.ndarray:
    """The configuration `qpos` re-expressed against an identity base pose.

    Root joints `[x, y, yaw]` become `[bx + c·x − s·y, by + s·x + c·y, byaw + yaw]`
    (yaw wrapped to (−π, π]); every other joint is copied. Accepts a 1-D qpos or a
    2-D array of rows.

    Args:
        base_p, base_q: the world pose of the articulation's root (planar).
        qpos: joint positions in user order, root joints first (1-D) or (N, dof).

    Example:
        >>> q = fold_root([1.0, 2.0, 0.0], [1, 0, 0, 0], [0.5, 0.0, 0.1, 9.0])  # yaw 0: a shift
        >>> np.round(q, 4).tolist()
        [1.5, 2.0, 0.1, 9.0]
        >>> q = fold_root([0.0, 0.0, 0.0], [0.7071068, 0, 0, 0.7071068], [1.0, 0.0, 0.0])
        >>> (np.round(q, 4) + 0.0).tolist()  # the root's x axis is the world's y
        [0.0, 1.0, 1.5708]
    """
    bx, by, byaw = planar_base(base_p, base_q)
    c, s = math.cos(byaw), math.sin(byaw)
    q = np.array(qpos, dtype=np.float64, copy=True)
    flat = q.ndim == 1
    rows = q.reshape(1, -1) if flat else q
    x, y, yaw = rows[:, 0].copy(), rows[:, 1].copy(), rows[:, 2].copy()
    rows[:, 0] = bx + c * x - s * y
    rows[:, 1] = by + s * x + c * y
    rows[:, 2] = [_wrap_pi(byaw + v) for v in yaw]
    return rows[0] if flat else rows


def unfold_root(base_p, base_q, qpos, ref_yaw: float | None = None) -> np.ndarray:
    """Inverse of `fold_root`: root joints back into the root's own frame.

    The yaw comes back on the 2π branch nearest `ref_yaw` when one is given (the
    simulator's current yaw joint value: mplib wraps revolute joints into
    `[q_min, q_min + 2π)` inside its IK, so a plan's yaw column may come back a
    turn away from where the robot is), else as `yaw − byaw`. For rows (a plan)
    the yaw column is first made **continuous** along the rows (`np.unwrap`) and
    then the whole column is moved to the branch nearest `ref_yaw` at its first
    row — a per-row choice would put a 2π jump into a trajectory that crosses
    ±π relative to the reference.

    Args:
        base_p, base_q: the world pose of the articulation's root (planar).
        qpos: folded joint positions, 1-D or (N, dof).
        ref_yaw: the yaw joint value to bring the result next to (within π).

    Example:
        >>> base_p, base_q = [1.0, 2.0, 0.0], [0.7071068, 0, 0, 0.7071068]
        >>> q0 = np.array([0.3, -0.2, 0.4, 7.0])
        >>> np.allclose(unfold_root(base_p, base_q, fold_root(base_p, base_q, q0), ref_yaw=0.4), q0)
        True
        >>> round(float(unfold_root(base_p, base_q, [1.0, 2.0, -4.71, 0.0], ref_yaw=0.0)[2]), 2)  # a turn away
        0.0
    """
    bx, by, byaw = planar_base(base_p, base_q)
    c, s = math.cos(byaw), math.sin(byaw)
    q = np.array(qpos, dtype=np.float64, copy=True)
    flat = q.ndim == 1
    rows = q.reshape(1, -1) if flat else q
    dx, dy, yaw = rows[:, 0] - bx, rows[:, 1] - by, rows[:, 2] - byaw
    rows[:, 0] = c * dx + s * dy
    rows[:, 1] = -s * dx + c * dy
    if ref_yaw is not None:
        yaw = np.unwrap(yaw) if len(yaw) > 1 else yaw  # continuous along the plan
        yaw = yaw + 2.0 * math.pi * np.round((float(ref_yaw) - yaw[0]) / (2.0 * math.pi))
    rows[:, 2] = yaw
    return rows[0] if flat else rows


def unfold_root_rates(base_q, rates) -> np.ndarray:
    """Joint rates (or accelerations) of a folded plan, back into the root's frame.

    Only the two prismatic root joints rotate (`R(byaw)^T` on columns 0–1); the yaw
    rate and every other joint are unchanged. 1-D or (N, dof).

    Args:
        base_q: the root's world orientation, wxyz.
        rates: `velocity` / `acceleration` rows of a plan made against the identity base.

    Example:
        >>> v = unfold_root_rates([0.7071068, 0, 0, 0.7071068], [0.0, 1.0, 0.2, 0.0])
        >>> (np.round(v, 4) + 0.0).tolist()  # world +y is the root's +x
        [1.0, 0.0, 0.2, 0.0]
    """
    byaw = yaw_of_quat(base_q)
    c, s = math.cos(byaw), math.sin(byaw)
    r = np.array(rates, dtype=np.float64, copy=True)
    flat = r.ndim == 1
    rows = r.reshape(1, -1) if flat else r
    vx, vy = rows[:, 0].copy(), rows[:, 1].copy()
    rows[:, 0] = c * vx + s * vy
    rows[:, 1] = -s * vx + c * vy
    return rows[0] if flat else rows


def link_local_pose(link_world: np.ndarray, obj_world: np.ndarray) -> np.ndarray:
    """`T_link_obj = link_world^-1 · obj_world` — the transform an attachment must store.

    Homogeneous 4×4 matrices. This is the frame mplib composes an attached body in
    *once the planned articulation's base pose is the identity* (then its
    "link pose in the base frame" is the world link pose); with a non-identity base
    pose mplib stores `link_in_base^-1 · obj_world` instead, which is why the fold
    exists. Given a rigid grasp, `link_world_new · T_link_obj` is where the object
    is after any motion — the check `tests/test_solver_in_container.py` runs against
    the real planner.

    Example:
        >>> L = np.eye(4); L[:3, 3] = [1.0, 2.0, 3.0]
        >>> O = np.eye(4); O[:3, 3] = [1.0, 2.5, 3.0]
        >>> np.round(link_local_pose(L, O)[:3, 3], 6).tolist()
        [0.0, 0.5, 0.0]
    """
    L = np.asarray(link_world, dtype=np.float64).reshape(4, 4)
    O = np.asarray(obj_world, dtype=np.float64).reshape(4, 4)
    return np.linalg.inv(L) @ O
