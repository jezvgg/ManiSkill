"""The base turn as a yaw-only path: a trapezoid over one joint, and its swept arc checked.

Why this exists (K51 / D12, docs/lab-journal.md 2026-08-18)
--------------------------------------------------------------
`FetchMotionPlanningSapienSolver.rotate_base_z` used to ask `plan_screw` for a
whole-body plan that rotates the TCP about the base's z axis and then executed only
the base-yaw velocity of it (`follow_rotation` writes `qvel[2]` and holds the arm).
The pseudo-inverse put about a third of the turn into the base and the rest into
the arm — an arm motion the robot never made — and the collision check ran over
that phantom motion: with the torso free it hid a real cup-against-the-wall sweep,
with the torso masked it refused a drive on a collision the robot would not have
had. Nothing about a base turn needs a whole-body plan: the path is one joint, its
timing follows that joint's velocity/acceleration limits, and what must be checked
for collision is the robot as it is, swept along the arc, with whatever it holds.

Everything here is mplib-free and pure, so it can be tested on a Mac against a
stubbed collision checker; the solver supplies the real one (the planning world's
`check_robot_collision` + `check_self_collision` on the current full qpos with the
yaw substituted). `yaw_trapezoid` gives the timing, `sweep_yaw` chooses the side and
names the obstacle when neither side is free, and `HeldBodies` puts whatever the
robot is holding where physics has it before either of them means anything.

Example:
    >>> t, off, rate, T = yaw_trapezoid(0.9, v_max=0.9, a_max=0.9, dt=0.05)
    >>> round(T, 2), round(float(off[-1]), 6), float(rate[-1])
    (2.0, 0.9, 0.0)
    >>> sweep_yaw(1.0, lambda off: None)                       # nothing in the way
    (1.0, None)
    >>> sweep_yaw(1.0, lambda off: "cup<->wall" if off > 0 else None)[0]   # go round the other way
    -5.283185307179586
    >>> sweep_yaw(1.0, lambda off: "cup<->wall")[0] is None       # both ways blocked
    True
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

#: Fewest poses checked along a candidate arc (excluding the start, which is the robot
#: as it stands and is not this function's business to judge). A floor, not a count:
#: `sweep_samples` adds poses on a long arc so the spacing stays `SWEEP_STEP`.
SWEEP_SAMPLES = 16

#: Yaw between checked poses on an arc long enough to need more than `SWEEP_SAMPLES`.
#: At the burner's 0.9 m grasp radius 0.15 rad is 13 cm of cup travel between poses;
#: a fixed 16 per arc would have been 0.29 rad ≈ 26 cm on the long way round (4.66 rad),
#: which is exactly the arc taken when something is already in the way of the short one.
SWEEP_STEP = 0.15

# The one prefix every swept-arc refusal starts with. `rotate_base_z` prints the
# refusal and returns a bare -1, so an oracle that wants the *reason* reads it off
# stdout (`planners.oracle_common.capture_refusal`, D13) — the constant lives here
# so a rewording cannot silently disable that recovery.
SWEEP_REFUSAL = "rotation sweep hits"


def sweep_samples(angle: float, n_min: int = SWEEP_SAMPLES, step: float = SWEEP_STEP) -> int:
    """Poses to check on an arc of `angle` radians: `n_min`, or more to keep `step`.

    Args:
        angle: signed arc in radians.
        n_min: the floor (short arcs are sampled at least this finely).
        step: the coarsest yaw allowed between two checked poses, in radians.

    Returns:
        The number of poses (>= `n_min`).

    Example:
        >>> sweep_samples(1.62)          # 0.10 rad apart: the floor already suffices
        16
        >>> sweep_samples(4.66)          # the long way round, 0.15 rad apart
        32
    """
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")
    return max(int(n_min), int(math.ceil(abs(float(angle)) / step)))


def new_contacts(pairs, baseline=(), limit: int = 2) -> str | None:
    """Name the colliding pairs that were not already in contact before the motion.

    A swept-arc check has to subtract its start pose. The robot standing still can
    already register contacts — a held cup grazing the counter it was lifted from, a
    self-touch the SRDF does not list — and those are not the turn's doing; left in,
    every sample of both arcs "collides" and every turn the robot ever asks for comes
    back as a named refusal.

    Args:
        pairs: collision pair names at the sampled pose.
        baseline: pair names already in contact at the start pose.
        limit: how many names to put in the description.

    Returns:
        `"<pair>, <pair>"` for the new contacts, or None when there are none.

    Example:
        >>> new_contacts({"cup_110<->wall_left_room"})
        'cup_110<->wall_left_room'
        >>> new_contacts({"cup_110<->counter"}, baseline={"cup_110<->counter"}) is None
        True
        >>> new_contacts({"b<->c", "a<->b", "c<->d"}, limit=2)
        'a<->b, b<->c'
    """
    new = sorted(set(pairs) - set(baseline))
    return ", ".join(new[:limit]) if new else None


def yaw_trapezoid(angle: float, v_max: float, a_max: float, dt: float):
    """Rest-to-rest turn of `angle` under `|rate| <= v_max`, `|accel| <= a_max`, sampled every `dt`.

    A trapezoidal (triangular when the turn is too short to reach `v_max`) velocity
    profile sampled **exactly `dt` apart**, from 0 to the first multiple of `dt` at
    or past `duration` (`N = ceil(duration / dt) + 1` instants), so a follower that
    spends one control step per sample turns through `angle` — trapezoid-rule error
    only. The spacing has to be `dt` and not `duration / (N - 1)`: with `duration =
    2.8` and `dt = 0.05`, `int(2.8 / 0.05)` is 55 (the quotient is 55.999999999999993),
    so evenly dividing the duration spaced the samples 0.0509 s apart and the executed
    turn came out 1.8 % short — 0.03 rad of the burner's 1.62 rad drive, left for the
    residual pass to clear. Samples past `duration` sit at rest at the goal, and the
    last sample is exactly (`angle`, 0).

    Args:
        angle: signed turn in radians (0 gives a single sample at rest).
        v_max, a_max: positive limits in rad/s and rad/s².
        dt: control step in seconds.

    Returns:
        `(t, offsets, rates, duration)`: times (N,), yaw offsets from the start (N,),
        signed yaw rates (N,), and the duration in seconds.

    Example:
        >>> t, off, rate, T = yaw_trapezoid(-1.62, v_max=0.9, a_max=0.9, dt=0.05)
        >>> round(T, 3), round(float(off[-1]), 6), round(float(rate.min()), 3)
        (2.8, -1.62, -0.9)
        >>> t2, off2, rate2, T2 = yaw_trapezoid(0.4, v_max=0.9, a_max=0.9, dt=0.05)  # triangular
        >>> round(T2, 4), round(float(rate2.max()), 3), round(float(t2[1] - t2[0]), 3)
        (1.3333, 0.585, 0.05)
    """
    if v_max <= 0 or a_max <= 0 or dt <= 0:
        raise ValueError(f"limits and dt must be positive, got v_max={v_max} a_max={a_max} dt={dt}")
    theta = abs(float(angle))
    sign = 1.0 if angle >= 0 else -1.0
    if theta == 0.0:
        return np.zeros(1), np.zeros(1), np.zeros(1), 0.0
    t_ramp = v_max / a_max
    d_ramp = 0.5 * a_max * t_ramp**2
    if theta <= 2 * d_ramp:  # never reaches v_max
        t_ramp = math.sqrt(theta / a_max)
        v_peak = a_max * t_ramp
        t_flat = 0.0
    else:
        v_peak = v_max
        t_flat = (theta - 2 * d_ramp) / v_max
    duration = 2 * t_ramp + t_flat
    n = int(math.ceil(duration / dt - 1e-9)) + 1
    t = np.arange(max(n, 2), dtype=np.float64) * dt
    rates = np.empty_like(t)
    offsets = np.empty_like(t)
    for i, ti in enumerate(t):
        if ti >= duration:  # the tail past the last whole control step: at rest, arrived
            rates[i] = 0.0
            offsets[i] = theta
        elif ti < t_ramp:
            rates[i] = a_max * ti
            offsets[i] = 0.5 * a_max * ti**2
        elif ti <= t_ramp + t_flat:
            rates[i] = v_peak
            offsets[i] = 0.5 * a_max * t_ramp**2 + v_peak * (ti - t_ramp)
        else:
            td = duration - ti
            rates[i] = a_max * td
            offsets[i] = theta - 0.5 * a_max * td**2
    rates[-1] = 0.0
    offsets[-1] = theta
    return t, sign * offsets, sign * rates, float(duration)


def sweep_yaw(
    angle: float,
    colliding_at: Callable[[float], str | None],
    n_samples: int | None = None,
):
    """Pick the way round: the short way (`angle`), else the long way (`angle − 2π·sign`).

    Each candidate arc is sampled at yaw offsets from the start (the end included) and
    `colliding_at(offset)` is asked about each; it answers None for a free pose or a
    short description of the first colliding pair (`"<link>↔<object>"`). The first arc
    with no collision is chosen. Each arc gets its own sample count (`sweep_samples`),
    so the long way round is checked as finely as the short one, not 3x coarser.

    Args:
        angle: the signed short-way turn in radians.
        colliding_at: the checker (see `rotate_base_z` for the real one).
        n_samples: poses per arc; None (the default) asks `sweep_samples` per arc.

    Returns:
        `(chosen_angle, None)` on success, or `(None, reason)` where `reason` starts
        with `rotation sweep hits` and names the pair and yaw offset for both ways.

    Example:
        >>> sweep_yaw(0.5, lambda off: None)
        (0.5, None)
        >>> blocked = lambda off: "gripper_link<->wall_left_room" if 0 < off < 0.4 else None
        >>> chosen, why = sweep_yaw(0.5, blocked)
        >>> round(chosen, 4), why
        (-5.7832, None)
        >>> sweep_yaw(0.5, lambda off: "cup_110<->wall")[1][:35]
        'rotation sweep hits cup_110<->wall '
    """
    if n_samples is not None and n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    sign = 1.0 if angle >= 0 else -1.0
    hits = []
    for candidate in (float(angle), float(angle) - 2 * math.pi * sign):
        n = sweep_samples(candidate) if n_samples is None else int(n_samples)
        for k in range(1, n + 1):
            offset = candidate * k / n
            hit = colliding_at(offset)
            if hit:
                hits.append((hit, offset))
                break
        else:
            return candidate, None
    (hit, off), (hit2, off2) = hits
    return None, (
        f"{SWEEP_REFUSAL} {hit} at yaw={off:+.2f} rad (the other way round: "
        f"{hit2} at yaw={off2:+.2f} rad)"
    )


def yaw_path(qpos, yaw_index: int, angle: float, v_max: float, a_max: float, dt: float, rate_scale: float = 1.0):
    """The yaw turn as a `plan_screw`-shaped result: every joint frozen but one.

    Args:
        qpos: the move-group configuration to start from (1-D; copied, not changed).
        yaw_index: which entry is the base yaw (2 for ds_fetch's move group).
        angle: signed turn in radians.
        v_max, a_max, dt: as `yaw_trapezoid`.
        rate_scale: what the follower's velocity channel multiplies the value it is
            handed by. `follow_rotation` writes `velocity[i][2]` into the base
            controller's normalized rotation channel, whose action 1.0 means
            `upper[1]` rad/s (3.14 for ds_fetch, `agents/ds_fetch/ds_fetch.py`),
            so the stored rate is `yaw_rate / rate_scale`; `position` stays in rad.

    Returns:
        dict with `status="Success"`, `time` (N,), `position` (N, dof), `velocity`
        (N, dof), `duration`, `angle`.

    Example:
        >>> res = yaw_path(np.zeros(11), 2, 0.9, v_max=0.9, a_max=0.9, dt=0.05, rate_scale=3.14)
        >>> res["position"].shape, round(float(res["position"][-1, 2]), 6), round(float(res["velocity"][:, 2].max()), 4)
        ((41, 11), 0.9, 0.2866)
        >>> bool((res["position"][:, [0, 1] + list(range(3, 11))] == 0).all())   # only the yaw moves
        True
    """
    q0 = np.asarray(qpos, dtype=np.float64).reshape(-1)
    t, offsets, rates, duration = yaw_trapezoid(angle, v_max, a_max, dt)
    position = np.tile(q0, (len(t), 1))
    position[:, yaw_index] = q0[yaw_index] + offsets
    velocity = np.zeros_like(position)
    velocity[:, yaw_index] = rates / float(rate_scale)
    return {
        "status": "Success",
        "time": t,
        "position": position,
        "velocity": velocity,
        "duration": duration,
        "angle": float(angle),
    }


class HeldBodies:
    """The planning world's attached bodies, drawn where physics actually has them.

    A swept-arc check is only worth as much as the pose it draws the held object at,
    and mplib 0.2.1 gets that pose wrong in two independent ways (both measured in the
    container on 2026-08-18, docs/lab-journal.md):

    1. **Frame.** An attached body is placed at `link_pose_in_base_frame *
       attached_pose` — the articulation's base pose is left out, while the robot's own
       FCL links do get it. With the robot spawned away from the world origin the held
       cup therefore orbits *the origin* as the base yaws: gripper at (2.79, −1.16),
       cup drawn at (0.00, −1.96), a phantom `cup<->wall_left_room`.
    2. **Staleness.** `attached_pose` is `link.inv() * object_world_pose` captured by
       the last `update_from_simulation`, so it is a rigid link→object transform only
       at the configuration it was captured in. Read it after the base has moved and
       `get_global_pose()` returns `link_new * link_old.inv() * object_world_old` —
       displaced by `2·sin(Δθ/2)·|base translation|`, metres for any real drive. This
       is why `sync` is a required argument and is called before anything is read: the
       correction below cannot be built from a stale snapshot, and the caller that
       most needs it (the residual turn of `rotate_base_z`, right after
       `follow_rotation` has turned the base ~1.6 rad) is exactly the one whose
       planning world is out of date.

    `place()` redraws every held body for the articulation's current configuration and
    `restore()` puts mplib's own `attached_pose` back, so the planner is left as found.
    Duck-typed on mplib's `Pose` (`inv()`, `*`) and `PlanningWorld` / `AttachedBody`,
    so it is testable on a Mac against fakes — see `tests/test_solver_stepping.py`.

    Since K53 (T5, `root_frame.py`): `SapienPlanningWorldV2` folds the robot's world
    pose into its root joints, so the planned articulation's base pose *is* the
    identity, mplib's own composition is right in every configuration, and the
    frame correction (1) here reduces to a no-op (`true_rel` == mplib's attached
    pose, `place()` redraws the body where it already is). The sync (2) is what
    still matters. Callers pass `art.get_base_pose()` — the identity when folded,
    the real base pose for a world that cannot fold (fixed-base arm, non-planar
    root) — so this stays correct either way.

    Args:
        world: the planning world (`get_object_names`, `is_object_attached`,
            `get_attached_object`).
        base_pose: the planned articulation's base pose *as the planning world holds
            it* (`art.get_base_pose()`).
        sync: the planner's `update_from_simulation`, called **first**. It keeps every
            attachment (mplib only rewrites `attached_body.pose` from the simulator's
            entity pose), so nothing has to be re-attached afterwards.

    Example:
        >>> held = HeldBodies(world, base_pose, planner.update_from_simulation)  # doctest: +SKIP
        >>> art.set_qpos(q_with_the_yaw_substituted, True); held.place()         # doctest: +SKIP
        >>> pairs = world.check_robot_collision() + world.check_self_collision() # doctest: +SKIP
        >>> held.restore()                                                      # doctest: +SKIP
        >>> held.names                                                          # doctest: +SKIP
        ['scene-0_cup_110']
    """

    def __init__(self, world, base_pose, sync: Callable[[], None]):
        sync()  # before any read: see (2) above
        self._base = base_pose
        self._held = []
        for name in world.get_object_names():
            if not world.is_object_attached(name):
                continue
            body = world.get_attached_object(name)
            link = body.get_attached_link_global_pose()  # mplib's: in the base frame
            # `get_global_pose()` is `link * attached_pose`, which right after the sync
            # is the simulator's world pose of the object; against the link's *world*
            # pose (`base_pose * link`) that gives the true link→object transform.
            self._held.append((body, body.pose, link.inv() * base_pose.inv() * body.get_global_pose()))

    @property
    def names(self) -> list[str]:
        """Names of the attached bodies, for the diagnostic line."""
        return [body.get_name() for body, _, _ in self._held]

    def __len__(self) -> int:
        return len(self._held)

    def place(self) -> None:
        """Redraw every held body at `base_pose * link * true_rel` for the current qpos."""
        for body, _, true_rel in self._held:
            link = body.get_attached_link_global_pose()
            body.set_pose(link.inv() * self._base * link * true_rel)
            body.update_pose()

    def restore(self) -> None:
        """Put mplib's own attached poses back, so the next planner call sees no change."""
        for body, pose0, _ in self._held:
            body.set_pose(pose0)
            body.update_pose()
