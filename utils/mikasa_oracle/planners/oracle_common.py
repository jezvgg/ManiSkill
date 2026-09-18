"""The stage kit shared by the memory-task oracles, extracted from burner_planner.py.

One module because the three oracles (burner, season dish, station checklist) repeat
the same staging skeleton: announce a stage, wait the cue out, grasp something,
tell the planning world about it, and stop the moment the horizon truncates the
episode. Everything here is generalised over *which* object is held — the burner
passes `task.cup`, season dish its bottle — and everything mplib-flavoured is
imported inside the function that needs it, so this module imports on a Mac and the
offline stage tests run the real staging against `tools/stub_planner.py`.

Contract reminders that bind every caller (template_planner.py, and D6 of the
2026-08-18 plan): `-1` means the oracle gave up before a decision — a planning or
grasp failure only. A physical miss (the object landed on the wrong target, a
distractor moved, a commit did not latch) returns the last gym 5-tuple, so the
sweep books it as `missed`, not `no_plan`. And the horizon is checked with
`stopped_by_horizon(planner)` after every primitive, *before* interpreting its
result — a stage cut off by the horizon must be returned as the truncated tuple,
never misread as a failed plan.

Kept on purpose, unused in production since K53: `capture_refusal` / `_Tee` and
`carry_pose(after=...)` were the D13 seam ("carry only after a named drive refusal")
and no shipped oracle calls them now that the carry is a full stage (K52). They stay
because the seam is the documented fallback if a task ever needs the conditional
carry again, and `tests/test_oracle_common.py` keeps them honest; delete both together
with those tests if that day never comes.
"""

from __future__ import annotations

import contextlib
import inspect as _inspect
import os
import io
import sys
import types

import numpy as np
import sapien
import torch

from utils.mikasa_oracle.motionplanning.fetch.base_yaw import SWEEP_REFUSAL  # mplib-free

# Distance the fingers close over, used to sink the grasp into the object.
FINGER_LENGTH = 0.025

# Refinement cap the oracles run the solver with: ~3 s of nudging at the control
# rate. The solver's own default is 200 (the inherited planners rely on it); an
# oracle would rather fail a stage inside the horizon than spend 10 s converging.
ORACLE_MAX_REFINE_STEPS = 60


def _np(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def say(env, who: str, stage: str, **extra) -> None:
    """Announce a stage on stdout and, when `env` is a PlannerLogger, in events.jsonl.

    The oracle's stdout is what the next debugging turn reads
    (docs/coding-agent-primer.md §2-3). One line per stage, `[{who}] {stage} ...`,
    so it can be grepped out of the solver's IK/RRT prints; and the same line as a
    `phase` event when the sweep wrapped the env with
    utils.mikasa_oracle.planner_log.PlannerLogger, so `--log-dir` traces name the stage
    next to the step.

    Args:
        env: the (possibly wrapped) env passed to `solve`.
        who: the oracle's tag, e.g. `"burner_planner"`.
        stage: short stage name, e.g. `"reach cup"`.
        **extra: JSON-serialisable fields for the event line (target burner, ...).

    Example:
        >>> say(env, "burner_planner", "drive to stove", target=2)  # doctest: +SKIP
        [burner_planner] drive to stove target=2
    """
    tail = " ".join(f"{k}={v}" for k, v in extra.items())
    print(f"[{who}] {stage}" + (f" {tail}" if tail else ""), flush=True)
    log = getattr(env, "log_event", None)
    if callable(log):
        log("phase", stage, **extra)


def normalize_base_yaw(env, planner, task, say=None) -> None:
    """Rewrite the base yaw into (-pi, pi] — a representation change of the
    SAME physical pose (the K55 precedent); a wound yaw feeds the rotate
    sweep hallucination (K109, filed).

    Shared by the cabinet planners (DepthRecall, CabinetSearch). It lived in
    `depth_recall_planner` until 2026-09-07, when the publish generator showed that
    one import dragged a whole task into a three-task PR (`tools/publish_to_fork.py`
    closes the selection over task-to-task imports). `say(env, stage, **extra)` is the
    caller's announcer; the default prints the `[solve] ...` line the old one printed.

    Example:
        >>> normalize_base_yaw(env, planner, task)  # doctest: +SKIP
        [solve] base yaw normalized {'was': 3.4, 'now': -2.883}
    """
    q = task.agent.robot.get_qpos()
    q = q.clone() if torch.is_tensor(q) else torch.as_tensor(q).clone()
    yaw = float(q.reshape(-1)[2])
    wrapped = (yaw + np.pi) % (2 * np.pi) - np.pi
    if abs(wrapped - yaw) > 1e-9:
        q.reshape(-1)[2] = wrapped
        task.agent.robot.set_qpos(q)
        planner.planner.update_from_simulation()
        announce = say or (lambda env, stage, **extra: print(f"[solve] {stage}" + (f" {extra}" if extra else "")))
        announce(env, "base yaw normalized", was=round(yaw, 3), now=round(wrapped, 3))


def fail(env, who: str, stage: str, **extra):
    """Return -1 for a failed stage, saying which one — never a silent -1.

    Example:
        >>> if res == -1: return fail(env, "burner_planner", "reach cup")  # doctest: +SKIP
    """
    say(env, who, f"FAILED: {stage} (planner returned -1)", **extra)
    return -1


def stopped_by_horizon(planner) -> bool:
    """True once the episode was truncated under this planner's feet.

    Reads `planner.truncated` — the latch `StepGuard` keeps on the real solver and
    `tools/stub_planner.py` mirrors. Check it after **every** primitive, before
    interpreting the primitive's result (K23): once the horizon has struck, a
    still-open gripper or an unread `-1` is the horizon's doing, and the honest
    return value is the truncated 5-tuple the primitive handed back — the sweep
    then books the episode as `truncated`, not as `no_plan`.

    Example:
        >>> res, grasped = try_grasp(env, planner, task, task.cup, grasp, reach)  # doctest: +SKIP
        >>> if res != -1 and stopped_by_horizon(planner):  # doctest: +SKIP
        ...     return res  # truncated mid-grasp: not a failed plan
    """
    return bool(getattr(planner, "truncated", False))


#: The grasp leg tries its straight plans first and, refused, the same straight plan
#: with the target out of the planning world, before it lets RRT answer (see
#: `try_grasp`). Env `MIKASA_GRASP_TOUCH_RETRY=0` restores the plain call.
GRASP_TOUCH_RETRY = os.environ.get("MIKASA_GRASP_TOUCH_RETRY", "1") != "0"

#: TOPP limits the solver plans the ARM under when the env is in `pd_joint_delta_pos`
#: (`default_planner_factory`), for a probe: 0 (the default) keeps the solver's own.
#: Control step at or above which the clock counts as SLOW and plans are timed down (see
#: `default_planner_factory`): 0.09 s catches the dataset's 10 Hz and leaves the tasks' own
#: 20 Hz (0.05 s) exactly as measured.
SLOW_CLOCK_TIMESTEP = float(os.environ.get("MIKASA_SLOW_CLOCK_TIMESTEP", "0.09"))
SLOW_CLOCK_VEL_LIMIT = float(os.environ.get("MIKASA_SLOW_CLOCK_VEL", "0.45"))
#: `MikasaFetchSolver`'s own default, for the cap above to compare against.
SOLVER_DEFAULT_VEL_LIMIT = 0.9

DELTA_JOINT_VEL_LIMIT = float(os.environ.get("MIKASA_DELTA_VEL", "0"))
DELTA_JOINT_ACC_LIMIT = float(os.environ.get("MIKASA_DELTA_ACC", "0.9"))


def default_planner_factory(env, debug: bool, vis: bool, *, max_refine_steps: int = ORACLE_MAX_REFINE_STEPS,
                            joint_vel_limits=None, joint_acc_limits=None):
    """The real Fetch solver, capped at the oracle refinement budget. Needs mplib.

    Imported here, not at the top, so this module (and the oracles that defer to
    it) import on a Mac.

    Args:
        env: the (possibly wrapped) env.
        debug, vis: passed through to the solver.
        max_refine_steps: per-instance cap on the post-path refinement loop
            (`ORACLE_MAX_REFINE_STEPS` = 60 ≈ 3 s); the solver's class default of
            200 is for the inherited planners.
        joint_vel_limits, joint_acc_limits: per-instance TOPP limits (solver default
            0.9 each). `None` passes nothing, so every existing caller builds a
            byte-identical solver — the only form in which touching shared solver
            configuration is defensible (K65). Lowering them trades episode steps for
            tracking accuracy, which is the quantity the K79c post-mortems measured as
            the root cause: 2-7 deg of joint lag, 7-9 cm of Cartesian error at 0.8 m
            extension, against collision margins of 3-6 cm.

    Example:
        >>> planner = default_planner_factory(env, debug=False, vis=False)  # doctest: +SKIP
    """
    from utils.mikasa_oracle.motionplanning.fetch.extand import MikasaFetchSolver

    if joint_vel_limits is None and DELTA_JOINT_VEL_LIMIT > 0 \
            and getattr(env.unwrapped, "control_mode", None) == "pd_joint_delta_pos":
        # Optional, measured OFF (2026-09-09): planning the arm slower under the delta
        # controller (whose 0.1 rad step caps the PD torque and so the speed) moved the
        # end configurations of Retrieval's stages onto joint limits (1101, 1102 refused
        # at 0.6 and 0.75 where 0.9 passed). The lag is handled where it arises instead:
        # `follow_forward_path_w_refinement` gates the knot on the arm's lag.
        joint_vel_limits = DELTA_JOINT_VEL_LIMIT
        if joint_acc_limits is None:
            joint_acc_limits = DELTA_JOINT_ACC_LIMIT
    if getattr(env.unwrapped, "control_mode", None) == "pd_joint_delta_pos" \
            and float(getattr(env.unwrapped, "control_timestep", 0.05)) >= SLOW_CLOCK_TIMESTEP:
        # A SLOW control clock — the VLA dataset's 10 Hz (`evaluate_planner --control-freq
        # 10`). An action is a delta per control STEP, bounded at 0.1 rad however long the
        # step lasts, so a path timed for the solver's default 0.9 asks the channel for
        # more than it can carry and the recorded actions sit at the clip. Measured
        # 2026-09-10 at 10 Hz, arm clip fraction / episode steps:
        #   Retrieval 1100     0.9 -> 0.107      0.6 -> 0.000 (327)   0.45 -> 0.000 (341)
        #   SeasonDish 3600    0.9 -> 0.200      0.6 -> 0.005 (436)   0.45 -> 0.000 (490)
        #   CabinetSearch 2367 0.9 -> 0.028      0.6 -> 0.028 (1850)  0.45 -> 0.000 (2012)
        # At 0.6 the clipped steps were genuine motion — the plan asking 0.1 rad where the
        # PD delivers 0.063 — so the cap is 0.45: every task clips nothing and the
        # episodes grow 4-9 %. A CAP rather than a
        # default, because the oracles set the limit themselves (SeasonDish passes 0.9 —
        # `JOINT_LIMIT_SCALE` — and kept clipping a fifth of its steps until this capped
        # it). Keyed on the clock, not on the mode, so the tasks' own 20 Hz (0.05 s,
        # 0.038 rad/step, no clip) is untouched — and the 2026-09-09 finding above (0.6 at
        # 20 Hz put Retrieval's stage ends on joint limits) is about a step half as long.
        joint_vel_limits = min(SOLVER_DEFAULT_VEL_LIMIT if joint_vel_limits is None
                               else float(joint_vel_limits), SLOW_CLOCK_VEL_LIMIT)
    return MikasaFetchSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        max_refine_steps=max_refine_steps,
        **({} if joint_vel_limits is None else {"joint_vel_limits": joint_vel_limits}),
        **({} if joint_acc_limits is None else {"joint_acc_limits": joint_acc_limits}),
    )


def default_grasp_info(obb, ee_direction, target_closing, finger_length: float = FINGER_LENGTH) -> dict:
    """OBB thin-side grasp frame, `utils.mikasa_oracle.motionplanning.fetch.utils`. Needs mplib.

    Example:
        >>> info = default_grasp_info(obb, ee_direction, target_closing)  # doctest: +SKIP
        >>> grasp = task.agent.build_grasp_pose(
        ...     info["approaching"], info["closing"], info["center"])     # doctest: +SKIP
    """
    from utils.mikasa_oracle.motionplanning.fetch.utils import compute_box_grasp_thin_side_info

    return compute_box_grasp_thin_side_info(
        obb, ee_direction=ee_direction, target_closing=target_closing, depth=finger_length, ortho=True
    )


def wait_cue(env, planner, info, who: str = "oracle"):
    """Idle until `elapsed_steps >= cue_steps` — the CUE phase only, not the delay.

    An honest agent may act as soon as the marker is gone; the road eats the delay.
    The cue length is resolved from the task in the two shapes the memory tasks
    use (K36): `task.cue_steps` — a per-env tensor on the burner (`burner.py`,
    drawn per episode) — when the attribute exists, else `int(task.cfg.cue_steps)`
    (season dish / station checklist, a config scalar).

    Chunked so a `-1` from the planner surfaces between chunks. Returns the last
    info dict, or -1.

    Example:
        >>> info = wait_cue(env, planner, info, who="burner_planner")  # doctest: +SKIP
        >>> if info == -1: return info                                  # doctest: +SKIP
    """
    task = env.unwrapped
    cue = getattr(task, "cue_steps", None)
    cue_steps = int(task.cfg.cue_steps) if cue is None else int(_np(cue).reshape(-1)[0])
    remaining = cue_steps - int(_np(task.elapsed_steps).reshape(-1)[0])
    say(env, who, "wait out the cue", steps=remaining)
    while remaining > 0:
        chunk = min(remaining, 20)
        res = planner.idle_steps(t=chunk)
        if res == -1:
            return fail(env, who, "wait out the cue")
        remaining -= chunk
        info = res[-1]
    return info


def grasp_geometry(task, obb, ee_direction, target_closing, grasp_info=default_grasp_info, standoff: float = 0.1):
    """(grasp_pose, reach_pose) for the object, with the far-side flip of myrobocasa_planner.py:100-120.

    `standoff`: metres the reach (pre-grasp) pose stands back from the grasp along the
    approach axis. 0.10 as inherited; SeasonDish passes 0.15 since 2026-09-09 — the
    approach to the standoff is a joint-space line whose hand path curves past the
    object, and at 10 cm the delta controller's tracking put a finger on the shaker
    (3608, step 107); the last stretch into the grasp is a screw, straight by construction.

    Example:
        >>> grasp, reach = grasp_geometry(task, obb, ee_dir, closing)  # doctest: +SKIP
        >>> res, grasped = try_grasp(env, planner, task, task.cup, grasp, reach)  # doctest: +SKIP
    """
    info = grasp_info(obb, ee_direction, target_closing)
    grasp = task.agent.build_grasp_pose(info["approaching"], info["closing"], info["center"])
    reach = grasp * sapien.Pose([0, 0, -float(standoff)])

    base_pos = _np(task.agent.base_link.pose.p)[0]
    obj_center = np.asarray(obb.center_mass)
    if np.dot(reach.p - obj_center, base_pos - obj_center) < 0:
        # The reach pose is on the far side of the object from the base: flip.
        info = grasp_info(obb, -ee_direction, target_closing)
        grasp = task.agent.build_grasp_pose(info["approaching"], info["closing"], info["center"])
        reach = grasp * sapien.Pose([0, 0, -float(standoff)])
    return grasp, reach


@contextlib.contextmanager
def keepout(planner, actors, pad=0.03, prefix: str = "keepout"):
    """Inflate `actors` in the planning world for the duration of the block (K76).

    Why this exists, measured. An approach RRTConnect returns is collision-free only in
    the binary sense, and `simplify=True` shortcutting pulls the path **taut against**
    the obstacle set — OMPL says so itself on the offending plan: ``The solution path
    was slightly touching on an invalid region of the state space, but it was
    successfully fixed``. Per-seed post-mortems put numbers on it: 0.9 cm of skin
    clearance at the closest sampled state on one seed, against a pd_joint_pos
    controller carrying 2-7 deg of tracking lag — 7-9 cm of Cartesian error at a 0.7 m
    extension. A path legal by millimetres executes as a collision, and a 9.4 cm shaker
    on a 4.5 cm base topples rather than slides. Six of the nine failing seeds on the
    randomized layout are that one mechanism.

    Padding defeats it without depending on the checker's resolution: an inflated proxy
    makes the whole marginal homotopy invalid, so the planner routes wide instead of
    skimming. `pad` must exceed the tracking error and stay under the standoff, or the
    pre-grasp pose itself becomes unreachable.

    **That standoff ceiling binds only the actor being approached** (K79c). With a 0.10 m
    standoff and 7-9 cm of tracking error there is no legal pad for the *target* — but the
    distractor is never approached, so nothing caps its pad, and it is the object whose
    displacement voids the episode (`distractor_ok`). Hence `pad` accepts a sequence
    parallel to `actors` as well as a single float: measured on seed 71, an approach leg
    executed with `tcp_err=0.064 m` inside a 0.03 m pad and pushed the distractor 0.1136 m,
    past `distractor_move_tol` at step 420 — 424 steps before the episode ended, so every
    grasp attempt after that was already playing for nothing.

    Removed on the way out, exceptions included — a leaked keep-out would silently make
    every later plan in the episode harder.

    Args:
        planner: the solver (needs `.planner.planning_world`); one without it yields
            unchanged, so the offline stub and the other oracles are untouched.
        actors: ManiSkill Actors to inflate.
        pad: metres added to the radius and to each end of the height.
        prefix: name prefix for the proxies.

    Example:
        >>> with keepout(planner, [task.shaker], pad=0.03):   # doctest: +SKIP
        ...     res = planner.static_manipulation(reach)      # routes wide of the shaker
    """
    pads = ([float(pad)] * len(list(actors))) if np.isscalar(pad) else [float(x) for x in pad]
    world = getattr(getattr(planner, "planner", None), "planning_world", None)
    added: list[str] = []
    if world is None:
        yield
        return
    try:
        import mplib
        from mplib.collision_detection import fcl

        for i, a in enumerate(actors):
            mesh = a.get_first_collision_mesh(to_world_frame=True)
            if mesh is None:
                continue
            obb = mesh.bounding_box_oriented
            ext = np.asarray(obb.primitive.extents, dtype=np.float64)
            centre = np.asarray(obb.centroid, dtype=np.float64)
            shape = fcl.Cylinder(
                float(max(ext[0], ext[1]) / 2.0 + pads[i]), float(ext[2] + 2.0 * pads[i])
            )
            name = f"{prefix}_{i}"
            world.add_object(name, fcl.CollisionObject(
                shape, mplib.Pose(p=centre, q=[1.0, 0.0, 0.0, 0.0])))
            added.append(name)
        yield
    except Exception:
        yield
    finally:
        for name in added:
            try:
                planner.planner.remove_object(name)
            except Exception:
                pass


@contextlib.contextmanager
def touchable(planner, needle: str):
    """Take every planning-world object whose name contains `needle` OUT of the world for
    the block, and put it back after — for a stroke whose GOAL is contact.

    mplib plans collision-free by construction; a pose inside the cube, which is what a
    push commands, is refused (`finger_link <-> cube` in the screw's report, 2026-09-08).
    The sim still collides — that is the push — only the planner stops objecting.
    Restored exceptions included, like `keepout`; the handle `get_object` returns is
    re-added under the same name so later plans see the object where the sim left it
    after the next `update_from_simulation`.

    Example:
        >>> with touchable(planner, "cube"):                     # doctest: +SKIP
        ...     res = planner.static_manipulation(push)          # the pose inside the cube plans
    """
    world = getattr(getattr(planner, "planner", None), "planning_world", None)
    taken: list = []
    if world is None:
        yield
        return
    try:
        for name in list(world.get_object_names()):
            if needle in name:
                obj = world.get_object(name)
                planner.planner.remove_object(name)
                taken.append((name, obj))
        yield
    finally:
        for _name, obj in taken:
            try:
                world.add_object(obj)  # an FCLObject carries its own name
            except Exception:
                pass


GRASP_ARRIVE_M = float(os.environ.get("MIKASA_GRASP_ARRIVE_M", "0.03"))
"""The stroke's tracking error above which `try_grasp` does not close the fingers (the
pads have 37 mm of play on a 26 mm bar; a 20 cm miss is a shove, not a grasp)."""


def try_grasp(env, planner, task, obj, grasp, reach, *, keepout_actors=None, keepout_pad: float = 0.03,
              resync_before_grasp: bool = False, n_init_qpos=None,
              close_on_contact: bool = False, approach_draws: int = 1,
              approach_stretch: int = 1, approach_stretch_tail=None,
              narrow_approach: bool = False, abort_on_touch: bool = False,
              approach_aperture: float = -1.0, full_dof_reach: bool = False,
              grasp_stop_on_touch: bool = False, approach_skim_cap: float | None = None,
              grasp_max_knots: int | None = None, grasp_knot_draws: int = 3,
              freeze_torso: bool = False, approach_max_knots: int | None = None,
              approach_clearance_pad: float | None = None, grasp_stretch: int = 1,
              approach_by_line: bool = False):
    """reach -> grasp -> close. Returns (res, grasped) with res == -1 on a failed plan.

    `grasped` is read from `task.agent.is_grasping(obj)` — the state, not the
    return code. Interpret the pair only after `stopped_by_horizon(planner)`: a
    grasp cut off by the horizon comes back as (truncated 5-tuple, False) and must
    not be retried or booked as a failure.

    Example:
        >>> res, grasped = try_grasp(env, planner, task, task.cup, grasp, reach)  # doctest: +SKIP
        >>> if res != -1 and stopped_by_horizon(planner): return res              # doctest: +SKIP
    """
    # The approach is planned with the loose objects inflated (K76); the final
    # standoff->grasp screw is not, because that leg must legitimately close on the
    # target. `keepout_actors=None` leaves every existing caller planning as before.
    # K79m: `n_init_qpos` is how many seed configurations mplib's IK tries before it
    # reports `IK Failed! Cannot find valid solution`. Passed only when a caller asks for
    # it, so the offline stub planner and every existing caller stay untouched.
    ik = {} if n_init_qpos is None else {"n_init_qpos": int(n_init_qpos)}
    # K80: draw the approach more than once and execute the draw that skims least.
    # Applied to the approach leg **only** — the standoff->grasp leg legitimately closes
    # on the target, so knots that "collide" there are the grasp, not a defect. Probed
    # rather than assumed, like `close_on_contact`, so the offline stub planner and the
    # other seven oracles keep calling the signature they were written against.
    draws = {}
    _params = _inspect.signature(planner.static_manipulation).parameters
    if int(approach_draws) > 1 and "draws" in _params:
        draws = {"draws": int(approach_draws)}
    # K84: slow the *approach* leg alone. The leg that topples the object is the approach
    # (measured: 4 of the 5 stable failures have their first slide owned by a long RRT
    # approach), and K79f/K79h only ever slowed every leg at once, losing on the horizon
    # the first time and on the grasp the second. `approach_stretch=1` leaves every
    # existing caller byte-identical.
    # A leg that ends AT a loose object may refuse a draw that runs through the scene.
    # Not a default anywhere: as one it cost DepthRecall half its seeds, because an arm
    # reaching inside a shelf intrudes by this metric routinely. See `path_is_a_plan`.
    if approach_skim_cap is not None and "skim_cap" in _params:
        draws["skim_cap"] = float(approach_skim_cap)
    # The approach leg under a knot cap as well (SeasonDish 1581, 2026-09-06: its best
    # skim-ranked draw was clean and 143 knots long, executed to reached=False at 0.094 m
    # and moved the bottle 20 cm before the grasp leg ever ran). The cap lives on the
    # skim-ranked branch, so the skim knife is kept; over the cap the leg is refused and
    # the caller's ladder re-draws or re-aims. Probed: a stub has no such parameter.
    if approach_max_knots is not None and "skim_max_knots" in _params:
        draws["skim_max_knots"] = int(approach_max_knots)
    # The clearance knife (SeasonDish 1679/1957, 2026-09-06): rank the approach's draws
    # by how many knots fall inside a WIDER proxy of the object — the topple draws were
    # clean by the 3 cm intrusion metric, a fingertip passed within tracking error of a
    # 16 g shaker. The wider proxy exists only while a draw is scored (prefix
    # "clearance"), so the planner's own world and every refusal are unchanged.
    # The approach as a straight joint line to the standoff's nearest IK solution,
    # before the screw and the RRT (SeasonDish 2026-09-06: every first approach from the
    # rest keyframe is an RRT detour of 80-110 knots — the screw is refused at the
    # shoulder_lift stop). Probed; the grasp leg never takes it (it filters `draws`).
    if approach_by_line and "by_line" in _params:
        draws["by_line"] = True
    if approach_clearance_pad is not None and "clearance_scorer" in _params:
        def _near(position, _obj=obj, _pad=float(approach_clearance_pad)):
            fn = getattr(planner, "path_env_collisions", None)
            if not callable(fn):
                return 0                      # a double without a planning world: no opinion
            with keepout(planner, [_obj], pad=_pad, prefix="clearance"):
                out = fn(position, names=True)
            if not isinstance(out, tuple) or int(out[0]) < 0:
                return 0
            return int(sum(v for k, v in out[1].items() if "clearance" in k))
        draws["clearance_scorer"] = _near
    if int(approach_stretch) > 1 and "stretch" in _params:
        draws["stretch"] = int(approach_stretch)
        if approach_stretch_tail and "stretch_tail" in _params:
            draws["stretch_tail"] = int(approach_stretch_tail)
    # K90: travel to the standoff with the fingers shut, and open them there.
    # The contact that knocks the object over is a **fingertip** — measured directly from
    # SAPIEN's contact reports on seeds 61/129/136/177: the first robot-shaker contact is
    # `r_` or `l_gripper_finger_link` every time, 1-15 steps before the object starts
    # moving, never the wrist or the forearm. The oracle holds the hand fully open for the
    # whole approach, which is 10 cm of aperture swept through a corridor the fingers only
    # need to be open at the *end* of. Closing for the transit halves the hand's width
    # where it clips and opens it again at the standoff, 10 cm short of the object.
    if narrow_approach and callable(getattr(planner, "change_gripper_state", None)):
        # `approach_aperture` is the normalised finger command for the transit: -1 shut,
        # +1 fully open (what the oracle ships). K90 measured the two endpoints and found
        # them to fail on disjoint sets of five; a middle value is untested and there is
        # a reason to expect it to differ from both — wide fingers clip on the way in,
        # shut fingers have to open at the standoff where they can collide instead.
        planner.change_gripper_state(gripper_state=float(approach_aperture))
        _resync(planner)
    elif narrow_approach and callable(getattr(planner, "close_gripper", None)):
        planner.close_gripper()
        _resync(planner)
    touch = {}
    if abort_on_touch and "stop_on_touch" in _params:
        touch = {"stop_on_touch": obj}
    with keepout(planner, keepout_actors or [], pad=keepout_pad):
        res = planner.static_manipulation(reach, disable_lift_joint=bool(freeze_torso), **ik, **draws, **touch)
        if res == -1 and full_dof_reach and callable(
                getattr(planner, "move_to_pose_with_RRTConnect", None)):
            # K101: mplib's IK restarts are local to the current posture, so an
            # arm-only reach can read `IK Failed` from one arrival family and plan
            # cleanly from another. Base freedom bridges it (measured at the drawer
            # column and again at the apple station). Default off: every existing
            # caller keeps the arm-only contract.
            res = planner.move_to_pose_with_RRTConnect(reach, **ik)
    # K92: the approach touched the target on the way in. The object has been nudged but
    # not necessarily knocked over — first contact leads its first movement by 2-15 steps
    # (K90) — so the one thing not to do now is drive the last 10 cm and close on a pose
    # read before the nudge. Hand back to the ladder, which re-reads and re-aims.
    if (abort_on_touch and res != -1 and callable(getattr(planner, "gripper_touching", None))
            and planner.gripper_touching(obj)):
        return res, False
    if narrow_approach and callable(getattr(planner, "open_gripper", None)):
        planner.open_gripper()
        _resync(planner)
    if res == -1:
        return res, False
    if resync_before_grasp:
        # K79i: the approach leg **executes**, and executing is what knocks the object.
        # Without this the grasp leg is planned against the pre-approach world. Measured
        # on seed 71 by comparing every `static_manipulation` call's PlanningWorld pose
        # against SAPIEN's: drift is 0.0 mm at ten of eleven calls and **88.9 mm** at the
        # one rung that reported `reached=True` to 2 mm and then closed its fingers 9.7 cm
        # from the object. A synced world cannot make that grasp succeed — the commanded
        # pose is still stale — but it makes the leg *refuse* instead of executing into
        # space that is empty only in the planner's model, which hands the step budget
        # back to the next rung instead of spending it on a certain miss.
        planner.planner.update_from_simulation()
    gtouch = {}
    if grasp_stop_on_touch and "stop_on_touch" in _params:
        # K102: for a ROLLING object the descend itself is the hazard — a grazing
        # fingertip accelerates a sphere out of the cage before the close. Stop the
        # grasp leg at first contact and close right there: the touch IS arrival.
        # Default off; the K92 abort-on-touch is the opposite policy (hand back) and
        # stays for approaches, where contact means a mistake rather than arrival.
        gtouch = {"stop_on_touch": obj}
    # The standoff->grasp leg gets the skim cap too, when the caller asked for one. The
    # comment above says knots that "collide" on this leg are the grasp rather than a
    # defect, and that holds for a leg that IS a short closing move — but when the screw
    # refuses and RRT answers with a wandering path, it is not that leg any more.
    # Measured over 60 SeasonDish episodes: 75 executed plans intrude at zero knots and
    # the seven that intrude are 98, 37, 31, 19, 5, 4, 3 — and exactly ONE of them stands
    # immediately before a grasp verdict, the 98-of-191 that shoved the bottle 11.2 cm.
    # The grasp leg is a short descent; a wandering RRT answer to it sweeps the object
    # (SeasonDish 1653, 2026-09-06: 209 knots, tcp_err 0.196 m, the bottle 33 cm away).
    # Under a knot cap with refusal the leg is refused instead and the caller's ladder
    # re-aims with the object untouched. Probed: a stub has no such parameters.
    gknots = {}
    if grasp_max_knots is not None and "max_knots" in _params:
        gknots = {"max_knots": int(grasp_max_knots), "knot_draws": int(grasp_knot_draws)}
        if "knot_refuse" in _params:
            gknots["knot_refuse"] = True
    # The standoff->grasp leg slowed `grasp_stretch` times (the K84 mechanism, on the
    # last 10 cm instead of the approach): SeasonDish 1679/1957 (2026-09-06, contact
    # traces) — the 16 g shaker is kicked (36 rad/s) the step the finger comes within
    # PhysX's contact offset of it at 0.16 m/s; the arm itself never touches it.
    gstretch = {}
    if int(grasp_stretch) > 1 and "stretch" in _params:
        gstretch = {"stretch": int(grasp_stretch)}
    skim = {k: v for k, v in draws.items() if k == "skim_cap"}
    res = -1
    if GRASP_TOUCH_RETRY and "max_knots" in _params and "knot_refuse" in _params:
        # The grasp leg's screw ends IN contact with the object by construction (the
        # fingers close around it), and the planning world refuses the last hair of it
        # when the finger's box meets the object's hull a millimetre early — then RRT
        # answers with a detour under the knot cap and the detour sweeps the object
        # (SeasonDish 3608 in pd_joint_delta_pos, 2026-09-09: 35 knots, the shaker
        # knocked 12 cm; the same seed's screw passes in pd_joint_pos by a hair).
        # So: the straight plans only (no RRT); refused, the same straight plan with
        # the object out of the planning world (`touchable`) — the sim still collides,
        # that IS the grasp, and `stop_on_touch` still stops the stroke; only then the
        # plain call with its RRT fallback as before.
        straight = dict(gknots); straight.update(max_knots=1, knot_draws=1, knot_refuse=True)
        res = planner.static_manipulation(grasp, disable_lift_joint=bool(freeze_torso), **ik, **gtouch,
                                          **straight, **gstretch, **skim)
        if res == -1 and getattr(planner, "planner", None) is not None:
            say(env, "oracle", "grasp stroke refused straight; retrying with the object touchable",
                obj=getattr(obj, "name", "?"))
            with touchable(planner, str(getattr(obj, "name", ""))):
                res = planner.static_manipulation(grasp, disable_lift_joint=bool(freeze_torso), **ik, **gtouch,
                                                  **straight, **gstretch, **skim)
        if res != -1 and stopped_by_horizon(planner):
            return res, False
    if res == -1:
        res = planner.static_manipulation(grasp, disable_lift_joint=bool(freeze_torso), **ik, **gtouch, **gknots,
                                          **gstretch, **skim)
    if res == -1 and full_dof_reach and callable(
            getattr(planner, "move_to_pose_with_RRTConnect", None)):
        # K101 applies to the descend too: after a planning reach, the short grasp
        # leg still refuses from some arrival families; base freedom bridges it.
        res = planner.move_to_pose_with_RRTConnect(grasp, **ik)
    if res == -1:
        return res, False
    # The stroke must have ARRIVED before the fingers close: on SeasonDish 1653 the stroke
    # executed with `reached=False, tcp_err=0.196 m` and the closing knocked the bottle
    # 33 cm away (the same knock on 1508/1585 at the other standoff). The planner records
    # its last move's tracking error; a stub has none, so the gate is probed.
    err = getattr(planner, "last_tcp_err", None)
    if err is not None and not gtouch and float(err) > GRASP_ARRIVE_M:
        # (a contact stroke, `gtouch`, stops short by design — first touch — and is
        # judged by the touch, not by the pose it was aimed at)
        say(env, "oracle", "grasp stroke fell short; not closing on air",
            tcp_err=round(float(err), 3), tol=GRASP_ARRIVE_M)
        return res, False
    # K79s: stop the closing motion at first contact instead of driving the fingers to a
    # fixed fully-closed position. The offline stage tests drive a stub planner whose
    # `close_gripper` takes no `stop_when` (tools/stub_planner.py), so the capability is
    # probed rather than assumed — same contract as `hold_object_in_planner`'s no-op when
    # there is no planning world.
    supports_stop = "stop_when" in _inspect.signature(planner.close_gripper).parameters \
        if callable(getattr(planner, "close_gripper", None)) else False
    if close_on_contact and supports_stop:
        res = planner.close_gripper(
            stop_when=lambda: bool(_np(task.agent.is_grasping(obj)).any()))
    else:
        res = planner.close_gripper()
    if res == -1:
        return res, False
    grasped = bool(_np(task.agent.is_grasping(obj)).any())
    return res, grasped


def _resync(planner) -> None:
    """`update_from_simulation` when there is a planning world; a no-op for the stub."""
    world = getattr(getattr(planner, "planner", None), "update_from_simulation", None)
    if callable(world):
        world()


@contextlib.contextmanager
def contact_stroke(planner, articulation_names):
    """Remove named articulations from the planning world for the block (K100).

    A drawer or a door can only be moved by TOUCHING it, and every planning leg in
    this solver is collision-checked — `SapienPlannerV2.plan_screw` walks the stroke
    against the world and RRT goals inside the handle geometry have no valid IK — so
    a push or a pull aimed AT an articulated fixture refuses by construction (the
    upstream Panda pushes only because its vanilla `plan_screw` never checks). For a
    contact stroke the fixture being moved is the goal, not an obstacle: take exactly
    it out of the world for the leg, and leave every OTHER fixture in, so a stroke at
    drawer 3 still plans around drawers 1, 2 and 4.

    Restored on every exit path, then re-synced by the caller's next
    `update_from_simulation` (re-adding uses the model as it stood; the sync puts its
    links back where the simulator has them). A planner with no planning world (the
    offline stub) yields unchanged.

    Args:
        planner: the solver (needs `.planner.planning_world`).
        articulation_names: names to remove; matched by substring against
            `get_articulation_names()`, because the world prefixes scene names.

    Example:
        >>> with contact_stroke(planner, ["stack_1_main_group_3_0"]):   # doctest: +SKIP
        ...     res = planner.static_manipulation(push_pose)            # may touch it
    """
    world = getattr(getattr(planner, "planner", None), "planning_world", None)
    if world is None:
        yield
        return
    removed = []
    try:
        for full in list(world.get_articulation_names()):
            if any(want in full for want in articulation_names):
                model = world.get_articulation(full)
                world.remove_articulation(full)
                removed.append(model)
        yield
    finally:
        for model in removed:
            try:
                world.add_articulation(model)
            except Exception:
                pass


#: `fingers.sum()` at or below this is an empty hand — the W12 threshold: a closed
#: mimic gripper bottoms out at 0.000 on nothing, and 8 mm between the pads is less
#: than any handle bar or edge in kitchen 102 (the bar alone is 26 mm).
FINGER_EMPTY_M = 0.008


def hinge_anchor(task, articulation_name, joint_name):
    """World axis of a vertical hinge and the sense of +qpos, from the raw frames.

    Reads the joint's own frame — `parent_link.entity_pose * pose_in_parent`, the
    parent being the fixture's body so the anchor does not move with the door — and
    takes the axis from the frame's x column (a SAPIEN revolute rotates about its
    frame's x). W14 verified this read against an independent three-point circle fit
    of the scanned handle bar: 0.0 cm apart, same sense.

    Args:
        task: `env.unwrapped`.
        articulation_name: key into `task.scene.articulations`.
        joint_name: the hinge's active-joint name, e.g. `"rightdoorhinge"`.

    Returns:
        `(anchor_xy, sense)`: the axis' world xy as float64, and +1 when +qpos is
        CCW about world +z (-1 otherwise).

    Raises:
        ValueError: the axis is not vertical — the arc-pull law is planar and has
            no meaning for such a hinge.

    Example:
        >>> H, s = hinge_anchor(task, "cab_main_main_group_0", "rightdoorhinge")  # doctest: +SKIP
        >>> H.round(3), s                                                         # doctest: +SKIP
        (array([ 2.735, -0.4  ]), 1)
    """
    art = task.scene.articulations[articulation_name]
    joint = [j for j in art.get_active_joints() if j.name == joint_name][0]
    raw = joint._objs[0]
    anchor = raw.get_parent_link().entity_pose * raw.get_pose_in_parent()
    T = anchor.to_transformation_matrix()
    axis_world = T[:3, 0]
    if abs(float(axis_world[2])) < 0.99:
        raise ValueError(
            f"{articulation_name}/{joint_name}: hinge axis {np.round(axis_world, 3)} "
            "is not vertical; the arc-pull law is planar")
    sense = 1 if float(axis_world[2]) > 0 else -1
    return T[:2, 3].astype(np.float64), sense


def pull_hinge_arc(env, planner, task, articulation_name, joint_name, *, target_rad,
                   who: str = "oracle", v_handle: float = 0.05, max_steps: int = 400,
                   stall_window: int = 50, stall_eps: float = 0.01, anchor=None,
                   open_dir: int = 1):
    """Open a hinged fixture by riding the base along the handle's arc (K103, W14).

    The caller has already closed the fingers on the handle; this stage measures the
    hinge anchor, then drives `planner.follow_arc` — the per-step unicycle law that
    keeps the TCP on the handle's circle — until the joint reaches `target_rad`, the
    hand empties, the joint stalls, or the step cap runs out. W14 measured the law
    on the kitchen-102 wall cabinet: 1.356 rad with a live grip against 0.55 for the
    straight pull, and the stall was the palm riding the panel, not the grip.

    D6 contract: `-1` only before any step is taken (the anchor is unreadable, or
    nothing is between the pads). Once the pull has stepped, the last 5-tuple comes
    back whatever the angle — how far the door got is state (`joint qpos`), and the
    caller judges it there, not from the return code. A horizon cut comes back as
    the truncated tuple, unread.

    Args:
        env: the (possibly wrapped) env, for `say`/`fail`.
        planner: the solver (or the stub — which records the call and steps nothing).
        task: `env.unwrapped`.
        articulation_name: key into `task.scene.articulations`.
        joint_name: the hinge's active-joint name.
        target_rad: stop once the joint reaches this (the caller's opening goal).
        who: the oracle's tag.
        v_handle: handle speed along its tangent, m/s.
        max_steps: hard step cap for the pull.
        stall_window: steps of joint history for the stall check.
        stall_eps: less joint progress than this over the window is a stall.
        anchor: optional `(anchor_xy, sense)` to skip the measurement — for offline
            stubs and for callers that already measured it.
        open_dir: +1 when opening INCREASES the hinge qpos (every leaf measured
            through K108: `cab_main`'s right leaf, limits [0, 3]); -1 when opening
            DECREASES it (a left leaf, limits [-3, 0] — `cab_main/leftdoorhinge`).
            Every angle comparison — the "target ahead of the start" gate, the
            target stop, the stall window — is made on `open_dir * qpos`, and the
            arc is ridden with `sense * open_dir`, so `target_rad` is passed SIGNED
            (-1.75 for a left leaf). +1 is the K104–K108 path unchanged.

    Returns:
        The last gym 5-tuple, or -1 (a `fail(...)`) before any step was taken.

    Raises:
        ValueError: `open_dir` is not +1 or -1 — a caller's bug, not a stage result.

    Example:
        >>> res = pull_hinge_arc(env, planner, task, "cab_main_main_group_0",   # doctest: +SKIP
        ...                      "rightdoorhinge", target_rad=1.2, who="cab")   # doctest: +SKIP
        >>> if res == -1: return res  # pull_hinge_arc already said why         # doctest: +SKIP
        >>> if res != -1 and stopped_by_horizon(planner): return res            # doctest: +SKIP
        >>> res = pull_hinge_arc(env, planner, task, "cab_main_main_group_0",   # doctest: +SKIP
        ...                      "leftdoorhinge", target_rad=-1.75, open_dir=-1,
        ...                      who="cab")                                     # doctest: +SKIP
    """
    if int(open_dir) not in (1, -1):
        raise ValueError(f"pull hinge arc: open_dir must be +1 or -1, got {open_dir!r}")
    open_dir = int(open_dir)
    if anchor is None:
        try:
            anchor = hinge_anchor(task, articulation_name, joint_name)
        except (ValueError, KeyError, IndexError) as e:
            return fail(env, who, f"pull hinge arc: anchor unreadable ({e})")
    anchor_xy, sense = anchor

    fingers = _np(task.agent.robot.get_qpos()).reshape(-1)[-2:]
    if float(fingers.sum()) <= FINGER_EMPTY_M:
        # A pull with an empty hand reads a free hinge as "did not move" — the
        # W12 trap. A missing grasp is a grasp failure before any decision (D6).
        return fail(env, who, "pull hinge arc: nothing between the pads",
                    fingers=[round(float(f), 4) for f in fingers])

    try:
        art = task.scene.articulations[articulation_name]
        names = [j.name for j in art.get_active_joints()]
        idx = names.index(joint_name)
    except (KeyError, ValueError) as e:
        return fail(env, who, f"pull hinge arc: joint unresolvable ({e})")

    def read_rad() -> float:
        return float(_np(art.get_qpos()).reshape(-1)[idx])

    start_rad = read_rad()
    if open_dir * float(target_rad) <= open_dir * start_rad:
        # The stop conditions assume opening moves qpos in the `open_dir` sense; a
        # target at or behind the start in that sense would read as "reached" on
        # the first check without a single pull — refuse before stepping instead.
        return fail(env, who, "pull hinge arc: target not ahead of the start",
                    target_rad=round(float(target_rad), 3),
                    start_rad=round(start_rad, 3), open_dir=open_dir)

    say(env, who, "pull hinge arc", target_rad=round(float(target_rad), 3),
        anchor=[round(float(v), 3) for v in anchor_xy], sense=sense,
        open_dir=open_dir, start_rad=round(start_rad, 3))

    hist: list = []
    fing: list = []   # aperture per step, so a slip can say WHEN the bar left the pads
    # "planner-stop" survives when the executor stopped for its own reason (lever
    # collapse, horizon, a stub that steps nothing) — saying "max-steps" for those
    # would contradict the executor's own report one line above.
    reason = ["planner-stop"]

    def stop() -> bool:
        theta = read_rad()
        hist.append(theta)
        if open_dir * theta >= open_dir * float(target_rad):
            reason[0] = "target"
            return True
        f = _np(task.agent.robot.get_qpos()).reshape(-1)[-2:]
        fing.append(float(f.sum()))
        if float(f.sum()) <= FINGER_EMPTY_M:
            reason[0] = "slipped"
            return True
        if (len(hist) >= stall_window
                and open_dir * (hist[-1] - hist[-stall_window]) < stall_eps):
            reason[0] = "stalled"
            return True
        return False

    # `sense` is the hinge's own (+qpos = CCW); the opening arc is CCW only when
    # opening also raises qpos, so the ridden sense is their product.
    res = planner.follow_arc(anchor_xy, sense * open_dir, v_handle=v_handle,
                             max_steps=max_steps, stop_when=stop)
    if res == -1:
        # The executor took no step at all (D6) — a refusal, said as one.
        return fail(env, who, "pull hinge arc: no step was taken")
    if stopped_by_horizon(planner):
        return res
    if reason[0] == "planner-stop" and len(hist) >= max_steps:
        reason[0] = "max-steps"
    say(env, who, "pull hinge arc done", rad=round(read_rad(), 3), why=reason[0],
        **({"door_tail": [round(v, 3) for v in hist[-30:]],
            "aperture_tail": [round(v, 4) for v in fing[-30:]]}
           if reason[0] in ("slipped", "stalled") else {}))
    return res


@contextlib.contextmanager
def planning_budget(seconds):
    """Temporarily widen `extand.PLANNING_TIME` for one oracle's `solve()`.

    `PLANNING_TIME` is module state shared by all eight oracles, so raising the
    module default would silently re-plan the burner and water-plants runs too —
    the thing K65 says may not be done to shared solver code. A context manager
    scopes it to one caller and restores it on every exit path.

    RRTConnect stops the moment its two trees meet and never improves the path
    afterwards, so a wider budget costs nothing on a plan that succeeds; it is
    paid only by plans that would otherwise have been refused. That is why the
    wall-clock cost of widening it is small (K79: +4.5% median episode) while the
    refusal count halves.

    `None` is a no-op, so a caller can pass a config value straight through. Also
    a no-op when mplib is absent, which keeps the offline stage tests importable
    on a Mac.

    Example:
        >>> with planning_budget(None):   # a no-op, and does not need mplib
        ...     pass
    """
    if seconds is None:
        yield
        return
    try:
        from utils.mikasa_oracle.motionplanning.fetch import extand
    except Exception:
        yield
        return
    old = extand.PLANNING_TIME
    extand.PLANNING_TIME = float(seconds)
    try:
        yield
    finally:
        extand.PLANNING_TIME = old


def hold_object_in_planner(env, planner, task, obj, held: bool, who: str = "oracle",
                           extra_touch=()) -> None:
    """Tell the planning world that `obj` is (or no longer is) part of the hand.

    Without this, the first emulated burner run died one step into the drive:
    `plan_screw` reported `collision scene-0-ds_fetch_gripper_link<->scene-0_cup_110`
    — the cup it had just grasped was still an obstacle in the planning world, so
    every state with the fingers closed on it "collides". `attach_object`
    (utils.mikasa/motionplanning/fetch/utils.py) moves the object into the
    gripper's frame and allows its contact with the touching links; `detach_object`
    puts it back as an obstacle once released, so the retract plan sees it where it
    landed. The planning world is synced right after the attach, so the stored
    link->object transform is taken at the configuration the hand is actually in
    (K52); and since K53 the planning world holds the robot with its base pose
    folded into the root joints, so that transform is the rigid `T_link_obj` and
    the held object is drawn where the hand is in every configuration — a hover,
    a tuck (`carry_pose`) and a pour tilt are collision-checked against the object
    itself, not against the 3 m phantom of before.

    A no-op on planners without a planning world (tools/stub_planner.py), so the
    offline stage tests keep running without mplib.

    Args:
        env: the (possibly wrapped) env, for `say`.
        planner: the MikasaFetchSolver (or the stub).
        task: `env.unwrapped`.
        obj: the ManiSkill Actor being held (e.g. `task.cup`).
        held: True after a verified grasp, False after the release.
        who: the oracle's tag, for the trace.
        extra_touch: link-name stems (e.g. "shoulder_lift") whose contact with the held
            object is allowed in the model, on top of the whole hand.

    Example:
        >>> hold_object_in_planner(env, planner, task, task.cup, held=True, who="burner_planner")  # doctest: +SKIP
        >>> # ... drive, hover, lower, open_gripper
        >>> hold_object_in_planner(env, planner, task, task.cup, held=False, who="burner_planner")  # doctest: +SKIP
    """
    world = getattr(getattr(planner, "planner", None), "planning_world", None)
    if world is None:
        say(env, who, "planning world: none (stub) — object attach skipped")
        return
    from utils.mikasa_oracle.motionplanning.fetch.utils import attach_object, convert_object_name

    entity = obj._objs[0]
    robot = task.agent.robot._objs[0]
    if held:
        link = next(l for l in robot.links if l.name.endswith("gripper_link"))
        # touch_links spelled out: the planning world holds the object as a convex
        # hull (utils.convert_actor_convex_mesh_to_fcl), which fills a cup's
        # opening, so the palm above the rim is "inside" it and mplib's
        # auto-detected touch set (links colliding at attach time) still left
        # gripper_link<->object as a collision on the very next plan. Allow the
        # whole hand.
        touch = [l for l in robot.links if any(k in l.name for k in ("gripper", "wrist"))]
        # `extra_touch`: link-name stems a caller has measured the held object to touch
        # harmlessly (SeasonDish 2025, 2026-09-06: the carry tuck rests a 16 cm bottle
        # against `shoulder_lift_link`, and the drive's start state is then a collision
        # in the model though the object is held). Allowed for this attachment only.
        touch += [l for l in robot.links
                  if any(k in l.name for k in extra_touch) and l not in touch]
        attach_object(world, entity, robot, link, touch_links=touch)
        planner.planner.update_from_simulation()  # the transform at the hand's true configuration
        say(env, who, "object attached to gripper_link in the planning world",
            touch_links=[l.name.split("_", 1)[-1] for l in touch])
    else:
        world.detach_object(convert_object_name(entity))
        say(env, who, "object detached in the planning world")


def pose_over(xyz, above: float, q) -> sapien.Pose:
    """The pose of an object origin `above` metres over the point `xyz`, upright as `q`.

    Example:
        >>> float(pose_over(np.array([0.0, 0.0, 1.0]), 0.5, np.array([1.0, 0, 0, 0])).p[2])
        1.5
    """
    return sapien.Pose(
        p=np.asarray(xyz, dtype=np.float64) + np.array([0.0, 0.0, float(above)]), q=q
    )


class _Tee(io.TextIOBase):
    """Write-through to several streams; `capture_refusal`'s plumbing.

    `encoding`, `fileno`, `isatty` and `errors` pass through to the first (real)
    stream, so libraries that inspect stdout while a base primitive runs
    (tqdm, logging handlers, `os.write(sys.stdout.fileno(), …)`) see a normal
    text stream and not a bare `TextIOBase`.
    """

    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self._streams:
            st.flush()

    @property
    def encoding(self):
        return getattr(self._streams[0], "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self._streams[0], "errors", None)

    def fileno(self):
        return self._streams[0].fileno()

    def isatty(self):
        return bool(getattr(self._streams[0], "isatty", lambda: False)())

    def writable(self):
        return True


def roll_room(planner):
    """`planner.roll_room()` — the roll joints' planning limits widened to the simulator's
    for the plans inside — or a no-op for a solver (a test double) without it.

    Open it only for a motion that IS a roll, and only when every plan after it is
    under it too: a leg planned inside the room can leave a roll joint outside the
    window, and every later plan then clips the start qpos back into it, so the
    model's forward kinematics disagree with the simulator (SeasonDish 3811,
    2026-09-09: 0.2 m and 4 deg apart, and the next drive chased an unreachable goal).

    Example:
        >>> with roll_room(planner):                       # doctest: +SKIP
        ...     res = common.arm_move(env, planner, pour_pose, who=WHO, stage="pour")
    """
    ctx = getattr(planner, "roll_room", None)
    return ctx() if callable(ctx) else contextlib.nullcontext()


def head_look_at(planner, point):
    """`planner.look_at(point)` — the drives inside hold the head toward `point` (world
    xyz) — or a no-op for a solver (a test double) without it.

    Example:
        >>> with head_look_at(planner, bowl_p):                                    # doctest: +SKIP
        ...     res = planner.drive_base(target_pos=dock_xyz, target_view_vec=face, freeze_arm=True)
    """
    ctx = getattr(planner, "look_at", None)
    return ctx(point) if callable(ctx) else contextlib.nullcontext()


@contextlib.contextmanager
def screw_tolerance(planner, tol):
    """The screw's arrival gate (`ARM_SCREW_GOAL_TOLERANCE`, metres and radians) set on
    THIS solver instance for the plans inside — `static_manipulation` and the K61 probe
    both read the attribute, so the probe still mirrors the execution. For a leg whose
    target has slack by construction: SeasonDish's lift (2026-09-09, 3811) only has to
    clear the neighbour, and its screw was refused with the knot before the stop 2.5 cm
    short of a target 3 cm above the floor. Restored on exit.

    Args:
        tol: `(metres, radians)` — the arrival gate for the plans inside.

    Example:
        >>> with screw_tolerance(planner, (0.03, 0.10)):    # doctest: +SKIP
        ...     ok = screw_plans(planner, rung_pose, disable_lift_joint=True)
    """
    had = "ARM_SCREW_GOAL_TOLERANCE" in vars(planner)
    saved = vars(planner).get("ARM_SCREW_GOAL_TOLERANCE")
    try:
        planner.ARM_SCREW_GOAL_TOLERANCE = tuple(tol)
    except Exception:  # a double that forbids attributes
        yield
        return
    try:
        yield
    finally:
        if had:
            planner.ARM_SCREW_GOAL_TOLERANCE = saved
        else:
            try:
                del planner.ARM_SCREW_GOAL_TOLERANCE
            except AttributeError:
                pass


@contextlib.contextmanager
def allow_held_contacts(env, planner, stems, who: str, stage: str):
    """Allow, in the planning world's collision matrix, the contacts a HELD object makes at
    the start with non-robot objects — the counter it still stands on — for the plans
    inside; removed on exit.

    mplib will not plan out of a start it calls colliding: the RRT prints `Invalid start
    state!`, perturbs the start and answers with a two-knot path (SeasonDish 3811,
    2026-09-09: the bottle, just grasped, rests on the counter in the model; the lift's
    four RRT draws all started that way and the executed one twisted the forearm 0.7 rad
    while the torso rose, ripping the bottle out of the pinch). The contact is real and
    harmless — the object is in the hand and about to leave the surface — the same
    reasoning as `hold_object_in_planner(extra_touch=)` for the object against the arm.
    Pairs are found by asking the planning world what collides NOW, so nothing is allowed
    that the start does not already touch; robot-link pairs are left alone.

    Args:
        stems: substrings naming the held object(s) in the planning world (e.g. "shaker").

    Yields:
        The pairs allowed, as `(name_a, name_b)` — empty when the solver is a double
        or nothing collides at the start.

    Example:
        >>> with allow_held_contacts(env, planner, ("shaker",), who=WHO, stage="lift") as pairs:  # doctest: +SKIP
        ...     res = common.arm_move(env, planner, lift_pose, who=WHO, stage="lift")
    """
    inner = getattr(planner, "planner", None)
    world = getattr(inner, "planning_world", None)
    if world is None or not callable(getattr(world, "check_collision", None)):
        yield []
        return
    pairs = []
    try:
        inner.update_from_simulation()
        for c in world.check_collision():
            n1, n2 = str(c.link_name1), str(c.link_name2)
            if ("fetch" in n1) or ("fetch" in n2):
                continue
            if any(s in n1 or s in n2 for s in stems):
                pairs.append((n1, n2))
        acm = world.get_allowed_collision_matrix()
        for n1, n2 in pairs:
            acm.set_entry(n1, n2, True)
    except Exception as exc:  # the probe must never lose a leg
        say(env, who, f"{stage}: could not read the model's contacts", why=f"{type(exc).__name__}: {exc}"[:120])
        pairs = []
    if pairs:
        say(env, who, f"{stage}: the held object touches the model's world at the start; allowing that contact",
            pairs=[f"{a.split('_', 1)[-1]}<->{b.split('_', 1)[-1]}" for a, b in pairs])
    try:
        yield pairs
    finally:
        if pairs:
            try:
                acm = world.get_allowed_collision_matrix()
                for n1, n2 in pairs:
                    acm.remove_entry(n1, n2)
            except Exception:
                pass


def screw_plans(planner, target_tcp_pose, *, disable_lift_joint: bool = False, want_result: bool = False):
    """Would `static_manipulation(target)` reach this pose by a **straight** screw? (K61)

    `static_manipulation` tries `plan_screw` and falls back to RRTConnect without
    saying so until afterwards, and an RRT path wanders in joint space — it is what
    makes an episode's video show the arm rotating and rising on its way to a pose a
    straight move would have reached. This probes the screw with the *same* arguments
    `static_manipulation` uses, so a True here means that call will take the screw
    branch. **It plans only — nothing is executed, so it costs no episode steps**
    (wall clock only), which is what makes it affordable to probe a whole candidate
    list before committing to one.

    Mirrors `MikasaPandaArmSolverV2.static_manipulation`: the same
    `only_manipulate` mask (base x/y/yaw always fixed, the torso lift fixed only when
    `disable_lift_joint`), the same `time_step`, the same `ARM_SCREW_GOAL_TOLERANCE`.
    If that method's screw call ever changes, this must change with it — pinned by
    `tests/test_oracle_common.py::test_k61_the_screw_probe_mirrors_static_manipulation`.

    Args:
        planner: the solver (needs `.planner`, `.robot`, `.base_env`).
        target_tcp_pose: the TCP pose to test, `sapien.Pose`.
        disable_lift_joint: as passed to `static_manipulation`.
        want_result: return the screw's result dict (its `position` knots included)
            instead of a bool — None when it did not plan.

    Returns:
        True if `plan_screw` reports `Success`, False otherwise (a refusal, or any
        solver that does not expose the mplib planner — the caller then just runs its
        candidates in their normal order); with `want_result`, the dict or None.

    Example:
        >>> ordered = sorted(cands, key=lambda c: not screw_plans(planner, pose_of(c)))  # doctest: +SKIP
    """
    inner = getattr(planner, "planner", None)
    robot = getattr(planner, "robot", None)
    base_env = getattr(planner, "base_env", None)
    if inner is None or robot is None or base_env is None:
        return None if want_result else False
    try:
        import mplib  # lazy: this module stays importable without mplib (offline tests)

        only_manipulate = [True, True, True, bool(disable_lift_joint)] + [False] * 11
        result = inner.plan_screw(
            mplib.Pose(p=np.asarray(target_tcp_pose.p), q=np.asarray(target_tcp_pose.q)),
            robot.get_qpos().cpu().numpy()[0],
            time_step=base_env.control_timestep,
            masked_joints=~np.array(only_manipulate),
            goal_tolerance=getattr(planner, "ARM_SCREW_GOAL_TOLERANCE", None),
        )
        ok = str(result.get("status", "")) == "Success"
        if not ok:
            # The solver's unjam retry (ARM_SCREW_UNJAM): a screw stopped by a joint's stop
            # is planned once more with that joint held. Mirrored here since 2026-09-09 —
            # without it the probe called 3608's pour "no straight candidate" while the
            # execution planned it by screw (`unjam=7`), and the ordering was blind.
            from utils.mikasa_oracle.motionplanning.fetch.extand import ARM_SCREW_UNJAM, screw_jammed_joints
            jammed = [j for j in screw_jammed_joints(result.get("status", ""))
                      if 0 <= j < len(only_manipulate)]
            if ARM_SCREW_UNJAM and jammed:
                held = list(only_manipulate)
                for j in jammed:
                    held[j] = True
                retry = inner.plan_screw(
                    mplib.Pose(p=np.asarray(target_tcp_pose.p), q=np.asarray(target_tcp_pose.q)),
                    robot.get_qpos().cpu().numpy()[0],
                    time_step=base_env.control_timestep,
                    masked_joints=~np.array(held),
                    goal_tolerance=getattr(planner, "ARM_SCREW_GOAL_TOLERANCE", None),
                )
                if str(retry.get("status", "")) == "Success":
                    result, ok = retry, True
        if want_result:
            return result if ok else None
        return ok
    except Exception:  # a stub planner, or an mplib that refuses the probe
        return None if want_result else False


@contextlib.contextmanager
def capture_refusal(pattern: str = SWEEP_REFUSAL):
    """Watch stdout for the K51 swept-arc refusal while a base primitive runs.

    The solver's contract for an honest "the held object really sweeps into
    something" is a printed status line starting with `rotation sweep hits
    <link>↔<obj> at yaw=…` (`base_yaw.sweep_yaw`, printed by `rotate_base_z` and
    echoed in the `[rotate_base_z] … refused=…` report line). The solver returns
    a bare `-1`, so this is the seam an oracle reads the *reason* from without
    reaching into solver internals. Everything printed still reaches the real
    stdout (a tee, not a redirect), so traces and debugging turns lose nothing.
    (The burner's D13 recovery tuck read it until K53 made the carry a full stage
    again; kept as the seam for any oracle that wants the refusal's text.)

    Yields an object whose `.refusal` is, after the block exits, the refusal text
    from the first matching line (sliced from the pattern), or None.

    Example:
        >>> with capture_refusal() as cap:                       # doctest: +SKIP
        ...     res = planner.drive_base(target_pos=p, target_view_vec=v)
        >>> if res == -1 and cap.refusal is not None:            # doctest: +SKIP
        ...     say(env, who, "drive refused by the swept arc", refusal=cap.refusal)
    """
    holder = types.SimpleNamespace(refusal=None)
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = _Tee(old, buf)
    try:
        yield holder
    finally:
        sys.stdout = old
        holder.refusal = next(
            (l[l.index(pattern):].strip() for l in buf.getvalue().splitlines() if pattern in l),
            None,
        )


#: The tucks `carry_pose` tries, in order: the TCP yawed about the world's z axis by
#: this much (degrees) from its current orientation — the object stays upright under
#: each. 0 = orientation held (least motion). Measured for exactly this set at the
#: shipped 0.25 m ahead (`tools/probes/t5_carry_yaws.py`, burner seed 3 post-lift,
#: journal 2026-08-18): 0 → screw `joint limit at index [7]` (a real shoulder_lift
#: limit) + `IK Failed`; +90, −90 and 180 → RRT `Success` (the screw refuses all
#: three on the same limit).
CARRY_YAWS_DEG = (0.0, 90.0, -90.0, 180.0)
"""Upright reorientations for the tuck, in the order the T5 probe measured them.

Half-turns (±45, ±135) were inserted ahead of 180 on the theory that the slip on
held-out seeds 17 and 29 came from 180 being the most violent candidate, and were
**withdrawn**: seed 29 then executed a *45 deg* tuck and lost the object anyway, so the
sweep angle is not the mechanism, and the held-out rate did not move (17/20 either way)
while eval seed 3 picked up a physical miss on the drive that follows. Whatever pulls
the object out of the fingers there, it is not how far the wrist turns."""


CARRY_LIFTS_M = (0.0, 0.08)
"""Extra height for the tuck, tried only after every yaw at the height below it failed.

The tuck pulls the TCP in to `base + ahead·face` at the height the lift left it, and the
held object hangs below and swings with the yaw. When it catches something on the way in
the refusal names the object — `shaker <-> counter_main`, `shaker <-> bowl` on held-out
seeds 17 and 29 — and no further yaw at that height helps, because the height is what is
wrong. Fallback-only, so a tuck that plans at the first candidate is unchanged."""



def level_correction(q) -> np.ndarray:
    """The world-frame rotation that puts body +Z of `q` back on world +Z, minimal arc.

    Returned as a *correction* rather than a corrected pose because the thing that
    must end up level (the held object) is not the thing being commanded (the TCP).
    They are rigidly attached, so the same world-frame pre-rotation levels both:
    ``q_tcp_new = correction(q_obj) * q_tcp``. Levelling the TCP instead would leave
    the object at whatever angle it happens to sit in the fingers.

    Minimal arc, so spin about the vertical is left alone — it costs rotation and
    changes nothing about whether the object is level.

    Args:
        q: `(w, x, y, z)` of the object as it is now.

    Returns:
        float64 `(w, x, y, z)`, a world-frame rotation to pre-multiply.

    Example:
        >>> import numpy as np, sapien
        >>> half = np.radians(40.0) / 2.0
        >>> tipped = np.array([np.cos(half), np.sin(half), 0.0, 0.0])   # 40 deg about X
        >>> fixed = (sapien.Pose(q=level_correction(tipped)) * sapien.Pose(q=tipped)).q
        >>> R = sapien.Pose(q=fixed).to_transformation_matrix()[:3, :3]
        >>> float(np.round(R[2, 2], 6))
        1.0
        >>> # an already-level pose needs no correction
        >>> float(np.round(level_correction(np.array([1.0, 0, 0, 0]))[0], 6))
        1.0
    """
    q = np.asarray(q, dtype=np.float64)
    v = sapien.Pose(q=q).to_transformation_matrix()[:3, 2]
    v = v / np.linalg.norm(v)
    dot = float(np.clip(v[2], -1.0, 1.0))
    if dot > 1.0 - 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = np.cross(v, np.array([0.0, 0.0, 1.0]))
    n = float(np.linalg.norm(axis))
    axis = np.array([1.0, 0.0, 0.0]) if n < 1e-9 else axis / n
    half = np.arccos(dot) / 2.0
    return np.array([np.cos(half), *(np.sin(half) * axis)], dtype=np.float64)


def slerp(q0, q1, t: float) -> np.ndarray:
    """Shortest-arc interpolation between two `(w, x, y, z)` quaternions.

    Args:
        q0, q1: the endpoints, `(w, x, y, z)`.
        t: 0 gives `q0`, 1 gives `q1`.

    Returns:
        float64 `(w, x, y, z)`, unit length.

    Example:
        >>> import numpy as np
        >>> np.round(slerp([1.0, 0, 0, 0], [0.0, 1, 0, 0], 0.5), 4).tolist()
        [0.7071, 0.7071, 0.0, 0.0]
    """
    a = np.asarray(q0, dtype=np.float64)
    b = np.asarray(q1, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    d = float(np.dot(a, b))
    if d < 0.0:                      # take the short way round
        b, d = -b, -d
    if d > 0.9995:                   # nearly identical: lerp and renormalise
        out = a + t * (b - a)
        return out / np.linalg.norm(out)
    th = np.arccos(np.clip(d, -1.0, 1.0))
    s = np.sin(th)
    return (np.sin((1 - t) * th) / s) * a + (np.sin(t * th) / s) * b


def _knot_kw(max_knots, knot_draws, knot_refuse: bool = False) -> dict:
    """The knot-cap kwargs, or nothing at all.

    Passed only when a caller asked for a cap: the oracles' test doubles implement
    `static_manipulation(pose, disable_lift_joint=...)` and nothing more, and widening
    the call unconditionally breaks them for a feature they never use. `knot_refuse`
    rides along only when asked for (SeasonDish's pre-hover waypoint, 1507 @0.15).
    """
    if max_knots is None:
        return {}
    kw = dict(max_knots=max_knots, knot_draws=knot_draws)
    if knot_refuse:
        kw["knot_refuse"] = True
    return kw


def move_via(env, planner, task, target, *, who: str, stage: str, legs: int = 2,
             disable_lift_joint: bool = False, bow: float = 0.0,
             max_knots=None, knot_draws: int = 1):
    """Reach `target` through `legs` interpolated waypoints instead of in one plan.

    K55, and the reason is the shape of the waste rather than its size. A single
    long plan lets RRT wander: over a water-plants pair the hand travels 2557 deg
    against 858 deg of net per-stage change, a 3.0x detour, and `hover over the
    bucket` ends **1.3 deg** from where it started having travelled **269**. Broken
    into short hops the planner has far less room to wander between the ends, and —
    more importantly — `plan_screw`, which is a straight line in SE(3) and barely
    rotates at all, succeeds much more often on a short move than a long one, so
    more of the path is straight by construction rather than by luck.

    Position is interpolated linearly and orientation by shortest-arc slerp, so the
    waypoints lie on the direct path; they add no distance, only structure.

    Falls back to the single direct move the moment a leg refuses, so this can only
    change *how* a reachable target is reached, never *whether* it is. A refused leg
    executes nothing and costs no step.

    Args:
        env, planner, task: as everywhere.
        target: the final TCP pose.
        who, stage: for the trace.
        legs: waypoints including the target. 1 is the old behaviour. Each leg is a
            separately time-parameterised trajectory that decelerates to a stop, so
            more legs means more stop-start and more torso correction — 2 is the
            measured compromise between that and letting a long plan wander.
        bow: how far the curve is lifted off the straight chord, as a fraction of
            the chord length. **0, and measured.** A curved path was tried at 0.12
            on the hover and the dip and took the pair to 0/2: those two moves are
            near-vertical, so bowing them upward fights the motion instead of
            rounding it, and the arc puts the cup where the bucket rim is. The
            parameter stays because the argument for curving is sound for a long
            *horizontal* transfer — it is the wrong tool for these two stages.
        disable_lift_joint: passed through to `static_manipulation`.

    Returns:
        The 5-tuple, or -1 if both the waypointed path and the direct move refuse.

    Example:
        >>> res = move_via(env, planner, task, pose, who=WHO, stage="hover")  # doctest: +SKIP
    """
    start = task.agent.tcp.pose.sp
    p0, q0 = np.asarray(start.p, dtype=np.float64), np.asarray(start.q, dtype=np.float64)
    p1 = np.asarray(target.p, dtype=np.float64)
    q1 = np.asarray(target.q, dtype=np.float64)
    n = max(1, int(legs))

    # A quadratic Bezier through a control point lifted off the chord, rather than
    # the chord itself. Two reasons, both from watching it run: a straight chord
    # between a low pose and a high one drags the cup along the shortest line, which
    # is exactly where the counter edge and the bucket rim are; and a path that
    # curves lets the torso make one monotone move instead of correcting at every
    # waypoint. `bow` is a fraction of the chord length, so it scales with the move
    # and is nothing at all for a short one.
    chord = p1 - p0
    span = float(np.linalg.norm(chord))
    lift = np.array([0.0, 0.0, 1.0]) * (bow * span)
    ctrl = p0 + chord * 0.5 + lift

    res = -1
    for i in range(1, n + 1):
        t = i / n
        # Bezier at t; at t=1 this is exactly p1, so the target is never approximated.
        pos = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * ctrl + t ** 2 * p1
        leg = sapien.Pose(p=pos, q=slerp(q0, q1, t))
        res = planner.static_manipulation(leg, disable_lift_joint=disable_lift_joint,
                                          **_knot_kw(max_knots, knot_draws))
        if res != -1 and stopped_by_horizon(planner):
            return res
        if res == -1:
            if i == 1:
                break            # the first hop refused: nothing executed, fall back
            say(env, who, f"{stage}: leg {i}/{n} refused, finishing direct")
            return planner.static_manipulation(target, disable_lift_joint=disable_lift_joint,
                                               **_knot_kw(max_knots, knot_draws))
    if res == -1:
        return planner.static_manipulation(target, disable_lift_joint=disable_lift_joint,
                                           **_knot_kw(max_knots, knot_draws))
    return res


def level_in_place(env, planner, task, obj, who: str = "oracle", *, tol_deg: float = 3.0,
                   max_step_deg: float = 30.0, max_bites: int = 3) -> bool:
    """Rotate the held object back to level without moving the TCP's position.

    The cheapest possible correction: the hand stays where it is and only turns, by
    the minimal arc that puts the object's +Z back on world +Z. Used before a tuck
    or a drive so the object is not carried at an angle, and after a pour so the
    tip does not persist into the next leg.

    Best effort by design. A refused plan executes nothing and costs no steps, so
    the caller carries on with the object still tilted rather than losing the
    episode to a cosmetic stage — the tilt is a quality measure here, not a success
    condition.

    Args:
        env: the (possibly wrapped) env, for `say`.
        planner: the solver (or the stub).
        task: `env.unwrapped`.
        obj: the held Actor.
        who: the oracle's tag.
        tol_deg: skip the move when the object is already this close to level.
        max_step_deg: largest single correction attempted, in degrees. 30 is under
            the 33 deg that was observed to plan and over half the 65 deg that was
            observed never to.

            **This defaulted to 180.0 until K58, which is the number the sentence
            above says not to use.** At 180 every real tilt gives
            `frac = min(1.0, 180/tilt) = 1.0`, so the first bite asks for the *entire*
            correction — the 65 deg case documented as never planning — and the
            `if res == -1 and frac >= 1.0: break` guard below then exits before the
            halving path or `max_bites` can be reached. The adaptive machinery was
            unreachable from the default. Measured over 60 seeds (412 attempts):
            **216 of 412 levellings, 52.4 %, came back `level fell short` with zero
            progress, at a median tilt of 65.1 deg.** So the object was carried at ~65
            deg through most return legs, and — because a 65 deg object makes the
            `levelled` carry candidates a large reorientation *and* translation — it
            also pushed the tuck onto its harder `as held` fallbacks (seed 56 lost the
            episode there).
        max_bites: cap on corrections, so a pose that cannot be levelled costs a
            bounded number of refused plans rather than the episode's horizon.

    Returns:
        True if the object ended level to within `tol_deg`.

    Example:
        >>> level_in_place(env, planner, task, task.cup, who="water_plants_planner")  # doctest: +SKIP
        True
    """
    before = _tilt_deg(_np(obj.pose.q).reshape(-1, 4)[0].astype(np.float64))
    if before <= tol_deg:
        return True
    say(env, who, "level the cup in place", tilt_deg=round(float(before), 1))

    # One big correction is refused where several small ones are not. Measured over
    # a water-plants pair: every attempt from 21-33 deg planned and landed the cup
    # at 0.3-0.5 deg, and **every attempt from 65-70 deg was refused** — the wrist
    # simply does not have that turn available in one move at the pose the pour
    # leaves it in. Walking the correction in `max_step_deg` bites turns the same
    # total into moves the arm will accept, and each bite is re-derived from the
    # cup's *measured* orientation rather than from the plan, so a bite that lands
    # short or slips in the fingers is corrected by the next one instead of
    # accumulating (that accumulation is the ratchet this whole change removes).
    for _ in range(max_bites):
        now_q = _np(obj.pose.q).reshape(-1, 4)[0].astype(np.float64)
        now = _tilt_deg(now_q)
        if now <= tol_deg:
            break
        frac = min(1.0, max_step_deg / now)
        corr = level_correction(now_q)
        if frac < 1.0:
            # Same axis, a fraction of the angle: slerp the correction toward identity.
            w = float(np.clip(corr[0], -1.0, 1.0))
            ang = 2.0 * np.arccos(w)
            axis = corr[1:]
            n = float(np.linalg.norm(axis))
            if n > 1e-9:
                axis = axis / n
                half = (ang * frac) / 2.0
                corr = np.array([np.cos(half), *(np.sin(half) * axis)], dtype=np.float64)
        tcp = task.agent.tcp.pose
        tcp_p = _np(tcp.p)[0].astype(np.float64)
        tcp_q = _np(tcp.q)[0].astype(np.float64)
        q = np.asarray((sapien.Pose(q=corr) * sapien.Pose(q=tcp_q)).q, dtype=np.float64)
        # Torso frozen, and that is measured, not inherited. Freeing it — the
        # obvious move, since a prismatic joint lifts the cup at zero rotation cost
        # where the arm can only do it by turning — was tried on seeds 0-1 and is
        # worse on every axis: hand rotation 2557 -> 2616 deg, **1/2 instead of
        # 2/2**, and the cup past the spill line for 20.9% of the carry against
        # 8.9%. K52's "the tuck must not spend the torso" holds, and
        # `test_carry_pose_tucks_the_tcp_inside_the_base_footprint_with_the_torso_frozen`
        # is guarding a real property rather than pinning a preference.
        # **Do not execute the part of a refused correction that did plan.** K59 tried
        # exactly that, on the reasoning that a partial screw *is* the smaller bite this
        # loop's halving spends its time hunting for — 219 of the v3 sweep's levellings
        # were refused, `shoulder_lift` 122 of them with a median 0.385 of the twist
        # left, so a third of each correction was already in hand and was being thrown
        # away. Measured over a full 30-seed sweep it converted **8 of 91** attempts
        # here, and cost exactly what the contract above exists to protect: the arm ends
        # the partial standing *on* the joint limit that stopped it, so the RRT that
        # follows has to climb out of that corner and comes back **longer** (median 100
        # knots against 79 for the same stage without it), and two episodes lost the cup
        # outright — `object slipped out on the way in`, both previously green — to
        # partial rotations executed at the pour pose on a 2.75 cm rim grasp.
        #
        # The reasoning that was wrong is worth naming: "re-deriving from the new
        # configuration gives the solver a fresh Jacobian" ignores that the binding
        # constraint travels with the arm. A refused bite costs no steps. Keep it so.
        res = planner.static_manipulation(sapien.Pose(p=tcp_p, q=q), disable_lift_joint=True)
        if res != -1 and stopped_by_horizon(planner):
            break
        after_bite = _tilt_deg(_np(obj.pose.q).reshape(-1, 4)[0].astype(np.float64))
        # No progress on this bite and nothing left to try smaller: give up rather
        # than spend the horizon on refusals.
        if res == -1 and frac >= 1.0:
            break
        if res == -1 and abs(after_bite - now) < 0.5:
            max_step_deg = max_step_deg / 2.0
            if max_step_deg < 5.0:
                break

    after_deg = _tilt_deg(_np(obj.pose.q).reshape(-1, 4)[0].astype(np.float64))
    say(env, who, "levelled" if after_deg <= tol_deg else "level fell short",
        tilt_deg=round(float(after_deg), 1), was=round(float(before), 1))
    return bool(after_deg <= tol_deg)


def _tilt_deg(q) -> float:
    """Angle in degrees between the pose's body +Z and world +Z."""
    R = sapien.Pose(q=np.asarray(q, dtype=np.float64)).to_transformation_matrix()[:3, :3]
    return float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))


def carry_pose(
    env, planner, task, obj, who: str = "oracle", *, ahead: float = 0.25, after: str | None = None,
    yaws_deg=CARRY_YAWS_DEG, lifts_m=CARRY_LIFTS_M,
    upright: bool = False, legs: int = 1, max_knots=None, knot_draws: int = 1,
    knot_refuse: bool = False, by_line: bool = False,
):
    """Bring the held object into a carry pose over the base before driving (K52/D12).

    Why: `rotate_base_z` sweeps the turn's arc for collisions **with the attached
    object** (K51), and with the arm stretched out after a lift the held object
    can genuinely sweep through furniture. The object is pulled in: TCP at
    `base + ahead·face` (inside the base's footprint), at the TCP's current height,
    the object kept upright — its current orientation first, then the TCP yawed
    about the world's z axis by each entry of `yaws_deg` (the object stays vertical
    under a yaw); one `static_manipulation` per candidate with
    `disable_lift_joint=True` so the torso is not spent on the tuck; the first
    candidate that plans is executed. A candidate that does not plan costs no
    step (the solver executes nothing on a refused plan).

    History: until K53 (T5, 2026-08-18) this was a *recovery* only (D13) — under
    mplib 0.2.1's attached-body frame error (the held cup drawn 3.296 m off at
    shoulder_pan+90°) the tuck itself was unplannable on both calibration seeds:
    the held orientation on a real `shoulder_lift` limit, every upright
    reorientation on phantom collisions. With the frame fixed the reorientations
    plan (measured for the shipped set at 0.25 m ahead, `tools/probes/t5_carry_yaws.py`:
    RRT Success at yaw +90 / −90 / 180, the held orientation still on the real
    limit; `t3_carry_reach.py`'s +90/+135/+180 at 0.3–0.4 m agree), so the carry is
    a full stage again in the burner and the season-dish oracles: between the lift
    and the drive.

    After the move the grasp is verified with `task.agent.is_grasping(obj)`; no
    candidate planning, or a slipped object, is an honest `-1` ("carry pose
    unreachable") — a grasp/plan failure before any decision, per D6. A horizon cut
    is returned as the truncated tuple.

    Args:
        env: the (possibly wrapped) env, for `say`.
        planner: the solver (or the stub).
        task: `env.unwrapped`.
        obj: the held Actor.
        who: the oracle's tag.
        ahead: metres ahead of the base centre along its facing (default 0.25 —
            inside the Fetch base's ~0.3 m footprint).
        after: a refusal this tuck is recovering from, e.g. the `rotation sweep
            hits …` text; when given, the *one* `FAILED:` line a refused tuck emits
            is the chained `carry pose unreachable after: …`.
        yaws_deg: the upright candidates, in order (`CARRY_YAWS_DEG`).
        lifts_m: TCP heights above the current one to try, in order
            (`CARRY_LIFTS_M`); each height gets the full yaw ladder.
        upright: try a levelled orientation basis before the as-held one (K55,
            water-plants); False keeps the single-basis behaviour byte for byte.
        legs: above 1 the tuck is broken into short hops via `move_via`.
        max_knots: the K58 redraw knife — forwarded to the solver.
        knot_draws: the K58 redraw knife — forwarded to the solver.
        knot_refuse: with `max_knots`, refuse an RRT answer over the cap instead of
            executing the shortest draw (2026-09-09: a straight-only tuck pass).
        by_line: ask the solver for a joint LINE to the nearest IK solution first
            (`static_manipulation(by_line=True)`): the tuck as one monotone joint
            motion, no roll winding, where the screw to the same pose wound the
            upperarm and forearm rolls by +2 rad each (SeasonDish, the yaw-90 carry).

    Example:
        >>> res = carry_pose(env, planner, task, task.cup, who="burner_planner")  # doctest: +SKIP
        >>> if res != -1 and stopped_by_horizon(planner): return res               # doctest: +SKIP
        >>> if res == -1: return res  # carry_pose already said why               # doctest: +SKIP
    """
    if upright:
        # Level the object **where it stands** before pulling it in. Asking for the
        # correction and the translation in one move is what made every candidate
        # refuse: after a pour the cup is 80 deg over, and "level it and tuck it"
        # is a much harder request than either half (measured, water-plants seeds
        # 0 and 1 — all four yaws refused, `carry pose unreachable`). A rotation
        # about the object in place keeps the wrist near where it already is.
        # Best effort: a refusal here costs no steps and the tuck below still runs.
        level_in_place(env, planner, task, obj, who=who)

    base = task.agent.base_link.pose
    base_p = _np(base.p)[0].astype(np.float64)
    face = _np(base.to_transformation_matrix())[0][:3, 0].astype(np.float64)
    face[2] = 0.0
    face = face / np.linalg.norm(face)
    tcp = task.agent.tcp.pose
    tcp_z = float(_np(tcp.p)[0][2])
    tcp_q = _np(tcp.q)[0].astype(np.float64)

    # K55. The docstring has always said "the object kept upright", and what the
    # code did was keep the object's *current* orientation — only upright if it
    # already was. Measured on water-plants seed 0 it was not: the tilt ratcheted
    # 0.0 -> 10.9 -> 31.7 -> 49.0 deg across the stages, each one preserving the
    # last one's error, and after a pour the cup was carried on at 108 deg.
    #
    # With `upright` the levelled orientation is tried **first** and the preserved
    # one is kept as a fallback rather than dropped. That ordering is the whole
    # design: asking for a large correction and the tuck translation in one move
    # is much harder than either half, and demanding it cost both seeds their
    # episode (`carry pose unreachable`, all four yaws, after an 80 deg pour). The
    # in-place levelling above removes most of the correction; this fallback means
    # that even when it does not, the tuck still plans and the episode survives
    # with a tilted cup — a worse number, not a lost run.
    #
    # `upright=False` keeps the old single-basis behaviour byte for byte: the
    # burner and season-dish oracles have published numbers measured against it.
    bases = [tcp_q]
    if upright:
        obj_q = _np(obj.pose.q).reshape(-1, 4)[0].astype(np.float64)
        corr = level_correction(obj_q)
        levelled = np.asarray(
            (sapien.Pose(q=corr) * sapien.Pose(q=tcp_q)).q, dtype=np.float64
        )
        bases = [levelled, tcp_q]

    base_target = np.array([base_p[0], base_p[1], 0.0]) + ahead * face + np.array([0.0, 0.0, tcp_z])
    chain = f" after: {after}" if after else ""
    tried = []
    # **Every levelled candidate before any as-held one, and the order is load-bearing.**
    #
    # K59 tried interleaving them, on the measurement that over the v3 sweep the
    # levelled basis planned 86/139 at yaw 0 and then **0 of 100** at yaw -90 and 180,
    # while as-held planned 36/49 at yaw 0 — so three of the four levelled rungs looked
    # like pure waste standing in front of a rung that works two thirds of the time.
    # That measurement was of the wrong thing. Plan rate is not the outcome; keeping the
    # cup is. Interleaved, the sweep produced **2 `object slipped out on the way in` in
    # 16 episodes where the ordered ladder produced 0 in 30** — seeds 0 and 25, both
    # previously successful, both lost at a tuck that planned.
    #
    # The mechanism: a refused rung costs no steps, so reordering cannot change the pose
    # the surviving rung starts from — what it changes is *which* rung survives. The
    # rare levelled winner (yaw 90 planned 3 times in 30 episodes) is the one that tucks
    # a **levelled** cup; promoting as-held substitutes a rung that carries the cup's
    # pour tilt in through the base, and a 2.75 cm rim grasp does not survive that. The
    # dead rungs are the price of reaching the safe winner, not waste to be optimised
    # out.
    #
    # Inside a basis, each `lifts_m` height gets the full yaw ladder — a refused
    # rung is free, and the raised tuck is the season-dish held-out fallback that
    # runs only after every yaw at the current height has refused. The failure
    # form `tried_yaw_lift` counts lifts × yaws per basis.
    for basis_i, basis in enumerate(bases):
        for lift in lifts_m:
            target_p = base_target + np.array([0.0, 0.0, float(lift)])
            for yaw in yaws_deg:
                half = np.radians(float(yaw)) / 2.0
                q = (sapien.Pose(q=np.array([np.cos(half), 0.0, 0.0, np.sin(half)])) * sapien.Pose(q=basis)).q
                target = sapien.Pose(p=target_p, q=q)
                say(env, who, "carry pose", target=[round(float(v), 3) for v in target.p],
                    yaw_deg=float(yaw), lift=round(float(lift), 3),
                    basis="levelled" if (upright and basis_i == 0) else "as held")
                # K55: hold the cup level for the whole tuck, not just at its ends.
                # Only on the levelled basis — constraining the fallback would ask the
                # planner to preserve a tilt it is meant to be escaping.
                # K55: waypointed, for the same reason the other long arm moves are —
                # the tuck is a large reorientation and a single plan wanders through it.
                # K55: `legs > 1` breaks the tuck into short hops (`move_via`), for the
                # same reason the other long arm moves are broken up. Default 1 keeps
                # the burner and season-dish oracles on the single plan their published
                # numbers and their tests were measured against.
                res = (
                    move_via(env, planner, task, target, who=who, stage="carry pose",
                             legs=legs, disable_lift_joint=True,
                             max_knots=max_knots, knot_draws=knot_draws)
                    if legs > 1
                    else planner.static_manipulation(
                        target, disable_lift_joint=True,
                        **_knot_kw(max_knots, knot_draws, knot_refuse),
                        **({"by_line": True} if by_line else {}),
                    )
                )
                if res != -1 and stopped_by_horizon(planner):
                    return res
                if res == -1:
                    tried.append((float(yaw), round(float(lift), 3)))
                    continue
                if not bool(_np(task.agent.is_grasping(obj)).any()):
                    return fail(env, who, f"carry pose unreachable: object slipped out on the way in{chain}")
                if upright:
                    # Second attempt at levelling, now that the arm is tucked. The
                    # first one runs before the tuck, where a large correction is
                    # cheapest to plan *if* it plans at all — and after a pour it does
                    # not: the arm is at full extension over the pot and an 80 deg
                    # wrist correction is refused there (measured, both seeds). Pulled
                    # in over the base the same correction is an ordinary move, and
                    # leaving it undone is what carries a tipped cup through the whole
                    # drive back. Best effort, exactly as above.
                    level_in_place(env, planner, task, obj, who=who)
                return res
    return fail(env, who, f"carry pose unreachable{chain}", tried_yaw_lift=tried)


def arm_move(env, planner, pose, *, who: str, stage: str, disable_lift_joint: bool = False,
             tries: int = 2, legs: int = 1, max_knots: int | None = None,
             knot_draws: int = 1, knot_refuse: bool = False):
    """`static_manipulation(pose)` with up to `tries` draws of the plan.

    A refused plan returns `-1` without stepping the env, and the solver's RRT
    fallback (`RRTConnect`, 4 s under emulation) is randomized: on the season dish's
    seed 12 the same reachable pose (IK found it) came back `RRTConnect Failed.
    Approximate solution` on one draw and as a 94-knot path on the next. So a stage
    that is RRT-bound — a hover from the tucked arm, a reach from the rest posture —
    gets one more draw before it is booked as a failed plan. Nothing is executed
    twice: the first draw that plans is the one that runs. Check the horizon after
    the call as usual (K23).

    Args:
        env, planner, who: as `say`.
        pose: the TCP target (sapien.Pose).
        stage: the stage's name for the retry line.
        disable_lift_joint: passed to `static_manipulation`.
        tries: draws in total (default 2).
        max_knots: if set, prefer an RRT path no longer than this many knots.
        knot_draws: how many RRT draws to take when `max_knots` is set; the shortest
            is kept. Draws cost no simulation steps, only planner time.

    Returns:
        the primitive's result: the 5-tuple, or -1 after the last refused draw.

    Example:
        >>> res = arm_move(env, planner, hover, who=WHO, stage="hover over bowl")  # doctest: +SKIP
        >>> if res != -1 and stopped_by_horizon(planner): return res              # doctest: +SKIP
        >>> if res == -1: return fail(env, WHO, "hover over bowl")                # doctest: +SKIP
    """
    res = -1
    for i in range(int(tries)):
        if legs > 1:
            # K55: go through interpolated waypoints. Short hops give RRT far less
            # room to wander and let the straight-line screw plan succeed more
            # often; `move_via` falls back to this same direct call if a leg
            # refuses, so nothing reachable becomes unreachable.
            res = move_via(env, planner, env.unwrapped, pose, who=who, stage=stage,
                           legs=legs, disable_lift_joint=disable_lift_joint,
                           max_knots=max_knots, knot_draws=knot_draws)
        else:
            # Only pass the knot-cap kwargs when a caller actually asked for them:
            # the oracles' test doubles implement `static_manipulation(pose,
            # disable_lift_joint=...)` and nothing else, and widening the call for
            # every caller would break them for a feature they do not use.
            res = planner.static_manipulation(pose, disable_lift_joint=disable_lift_joint,
                                              **_knot_kw(max_knots, knot_draws, knot_refuse))
        if res != -1:
            return res
        if i + 1 < int(tries):
            say(env, who, f"{stage}: plan refused, one more draw (RRT is randomized)")
    return res


def object_pose_in_base(task, obj) -> sapien.Pose:
    """The held object's pose in the base link's frame — its orientation *relative
    to the robot*, taken before the carry tuck turns the hand.

    Why: the carry pose (K52) may yaw the hand about world z to get the object over
    the base, and the base drive turns everything with it. The place-or-pour poses
    after the drive want the object oriented as it was when it was grasped —
    hand along the base's facing — not as the tuck left it: ask for
    `object_q_from_base(task, rel)` after the drive, which composes this relative
    pose with the base *as it is parked* (world-frame, read after the drive, as
    the burner's `cup_q` invariant demands).

    Args:
        task: `env.unwrapped`.
        obj: the held Actor.

    Example:
        >>> # after the lift:
        >>> cup_rel = object_pose_in_base(task, task.cup)      # doctest: +SKIP
        >>> # ... carry_pose, drive_base
        >>> # after the drive:
        >>> cup_q = object_q_from_base(task, cup_rel)          # doctest: +SKIP
    """
    return (task.agent.base_link.pose[0].inv() * obj.pose[0]).sp


def object_q_from_base(task, rel: sapien.Pose) -> np.ndarray:
    """World orientation (wxyz) of an object held in relation `rel` to the base, with
    the base where it stands now. See `object_pose_in_base`.

    Example:
        >>> q = object_q_from_base(task, cup_rel)  # doctest: +SKIP
        >>> hover = pose_over(burner_xyz, 0.154, q) * T_tcp_cup.inv()  # doctest: +SKIP
    """
    return np.asarray((task.agent.base_link.pose[0].sp * rel).q, dtype=np.float64)


def dock_for_target(dock_xyyaw, target_xyz) -> tuple[np.ndarray, np.ndarray]:
    """Slide a dock along the fixture's face until the target is straight ahead.

    A dock is a standoff pose in front of a fixture, and it is normally drawn for
    the fixture as a whole — the middle of the stove, the middle of the counter
    between two stations. The target sits to one side of it, so reaching from the
    dock as drawn is a diagonal: out, across, and usually down. Projecting the
    dock onto the fixture's `along` axis at the target's own offset turns that
    into a straight reach, at the cost of a short drive, and keeps the standoff
    exactly as the dock had it — only the sideways component moves.

    Args:
        dock_xyyaw: `(x, y, yaw)` of the dock as drawn. `yaw` faces the fixture.
        target_xyz: world position of the thing to stand in front of; z ignored.

    Returns:
        `(dock_xyz, face)`: the slid dock position (z = 0) and the unit facing
        vector, ready for `planner.drive_base(target_pos=, target_view_vec=)`.

    Example:
        >>> p, face = dock_for_target(np.array([3.134, -1.46, np.pi / 2]), np.array([2.9036, -0.216, 0.92]))
        >>> np.round(p, 4).tolist(), np.round(face, 6).tolist()
        ([2.9036, -1.46, 0.0], [0.0, 1.0, 0.0])
    """
    d = np.asarray(dock_xyyaw, dtype=np.float64)
    yaw = float(d[2])
    face = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    along = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    dock = np.array([d[0], d[1], 0.0])
    b = np.array([float(target_xyz[0]), float(target_xyz[1]), 0.0])
    p = dock + along * float(np.dot(b - dock, along))
    return p, face


def dock_error(task, dock_xyyaw) -> tuple[float, float]:
    """`(d_dock, dyaw_deg)`: how far the base parked from a dock, in xy metres and yaw degrees.

    The one parking measurement the memory tasks share (K26): the burner logs it
    after every drive, and it is the number that decides the station task's zone
    mode (D5: `radius` if the parking error stays well under 0.10 m, `nearest`
    otherwise). Reads `task.agent.base_link.pose`; nothing else.

    Args:
        task: `env.unwrapped`.
        dock_xyyaw: `(x, y, yaw)` of the dock the drive aimed at.

    Returns:
        `(d_dock, dyaw_deg)`: xy distance in metres, absolute heading error in
        degrees (wrapped to [0, 180]).

    Example:
        >>> d_dock, dyaw = dock_error(task, (3.134, -1.21, math.pi / 2))  # doctest: +SKIP
        >>> say(env, "burner_planner", "parked at the dock", d_dock=round(d_dock, 3), dyaw_deg=round(dyaw, 1))  # doctest: +SKIP
    """
    d = np.asarray(dock_xyyaw, dtype=np.float64).reshape(-1)
    pose = task.agent.base_link.pose
    base_p = _np(pose.p).reshape(-1, 3)[0]
    base_x = _np(pose.to_transformation_matrix()).reshape(-1, 4, 4)[0][:3, 0]
    d_dock = float(np.linalg.norm(base_p[:2] - d[:2]))
    yaw = float(np.arctan2(base_x[1], base_x[0]))
    dyaw = float(abs((yaw - float(d[2]) + np.pi) % (2 * np.pi) - np.pi))
    return d_dock, float(np.degrees(dyaw))


# ------------------------------------------ the fold that executes (K111, DepthRecall) --

REST_TORSO = 0.386
"""The rest keyframe's torso lift (the drive duck is 0.20): the un-duck target of
`fold_via_tcp`. Promoted with it from depth_recall_planner."""


def plan_joints(env, planner, task, targets: dict, *, label: str, tries: int = 2,
                who: str = "oracle", line_only: bool = False):
    """Plan and execute a joint-space move, straight line first, RRT second.

    The water_plants_planner.plan_to_joint_targets shape, promoted here from
    depth_recall_planner when that became its third copy (cabinet_retrieval_planner
    keeps its own: the Closed oracle's path is measured territory and stays
    byte-identical). Known and filed (K111): the RRT branch has been measured
    executing 185 knots and moving the arm by NOTHING (arm_vs_rest 0.562 bit-for-bit
    before and after, solos 5/6/12) — the LINE branch and `static_manipulation` are
    the two channels this repo has verified end to end, which is why `fold_via_tcp`
    exists.

    Args:
        env, planner: as everywhere.
        task: `env.unwrapped`.
        targets: `{joint_name: value}`; every other joint keeps its value.
        label: stage name for the trace.
        tries: RRT draws after a blocked line.
        who: the oracle's tag for `say`.

    Returns:
        The gym 5-tuple, or -1 after the last refused draw.

    Example:
        >>> res = plan_joints(env, planner, task, {"torso_lift_joint": 0.2},
        ...                   label="duck", who=WHO)               # doctest: +SKIP
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
            say(env, who, f"{label}: joint line",
                knots=int(np.asarray(line["position"]).shape[0]))
            return planner.follow_forward_path_w_refinement(line, refine=True)
        if line_only:
            # The caller wants the one channel that measurably executes or nothing
            # (K111: the RRT branch ran 185 knots and moved the arm by nothing).
            say(env, who, f"{label}: line only; no RRT draw", status=line["status"])
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
            say(env, who, f"{label}: rrt joints",
                knots=int(np.asarray(result["position"]).shape[0]))
            return planner.follow_forward_path_w_refinement(result, refine=True)
        say(env, who, f"{label}: plan refused", status=result["status"], draw=i + 1)
    return -1


def normalize_continuous_arm_joints(env, planner, task, *, who: str = "oracle"):
    """Rewrite every CONTINUOUS arm joint's qpos into (rest - pi, rest + pi].

    The K55 representation-change precedent, arm edition: solo 9 (K111) measured
    the wrist winding a full 2*pi across the restore seat (arm_vs_rest 6.283 with
    the arm physically at rest) and the next drive's rotate sweep refusing 3/3 —
    the wound-representation disease K109 filed for the base yaw, verbatim. The
    wrap is TOWARD the rest value, not toward zero: solo 19 measured a zero-centred
    wrap putting forearm_roll on the far side of rest (-3.06 vs rest ~0) and the
    fold's branch line dying of the long way round. A state write, so the planning
    world is re-synced afterwards; a no-op (no write, no line) when nothing wraps.

    Args:
        env, planner: as everywhere.
        task: `env.unwrapped`.
        who: the oracle's tag for `say`.

    Example:
        >>> normalize_continuous_arm_joints(env, planner, task, who=WHO)  # doctest: +SKIP
    """
    jm = task.agent.robot.active_joints_map
    arm_names = list(task.agent.controller.controllers["arm"].config.joint_names)
    rest_q = np.asarray(task.agent.keyframes["rest"].qpos,
                        dtype=np.float64).reshape(-1)
    q = task.agent.robot.get_qpos()
    q = q.clone() if torch.is_tensor(q) else torch.as_tensor(q).clone()
    flat = q.reshape(-1)
    changed = []
    for n in arm_names:
        j = jm[n]
        lim = _np(j.limits).reshape(-1)
        # the roll joints carry FINITE +-2*pi limits here, not inf: treat a
        # span of >= 4*pi - eps as a wrappable revolute
        if np.isfinite(lim).all() and float(lim[-1] - lim[0]) < 4 * np.pi - 0.1:
            continue
        i = j.active_index[0].item()
        v = float(flat[i])
        rest_v = float(rest_q[i])
        w = rest_v + ((v - rest_v + np.pi) % (2 * np.pi) - np.pi)
        if abs(w - v) > 1e-9:
            flat[i] = w
            changed.append((n, round(v, 3), round(w, 3)))
    if changed:
        task.agent.robot.set_qpos(q)
        planner.planner.update_from_simulation()
        say(env, who, "continuous arm joints normalized", joints=changed)


def fold_via_tcp(env, planner, task, rest_tcp, *, stage: str, who: str = "oracle",
                 rest_torso: float = REST_TORSO):
    """Return to the rest CARRIAGE via the channels that measurably execute.

    A torso LINE first (the duck's own form, reversed), then a Cartesian move
    (`static_manipulation`, torso frozen) to the episode-start rest TCP point
    re-based onto the current base pose, then the continuous joints wrapped and
    a branch LINE to the exact rest configuration. Solo 5/6 (K111) measured
    `plan_joints`' RRT branch executing 185 knots and moving the arm by NOTHING
    (arm_vs_rest 0.562 before and after, bit-for-bit) — the line branch and
    `static_manipulation` are the two channels this repo has verified end to end,
    so the fold uses only those. Stage 2 exists because the screw lands the TCP on
    the rest POINT while the arm can sit on another IK branch (solo 11:
    arm_vs_rest ~4 rad with the fold "ok"), and a frozen weird-branch arm honestly
    blocks the next drive's rotate; a branch line that flakes (arm_vs_rest > 1.0
    after it) gets one full retry round from a re-screwed start. Non-fatal, like
    the fold it replaces: the return value is the screw's, and the caller reads
    `arm_vs_rest` from qpos if it needs the truth.

    Args:
        env, planner: as everywhere.
        task: `env.unwrapped`.
        rest_tcp: the rest TCP pose IN THE BASE FRAME, captured at solve start
            (`base_link.pose.sp.inv() * tcp.pose.sp` with the arm still on the
            reset keyframe), so the re-basing is exact under any dock yaw.
        stage: prefix for the trace lines.
        who: the oracle's tag for `say`.
        rest_torso: the un-duck target (`REST_TORSO`, the rest keyframe's lift).

    Returns:
        The screw's result: the gym 5-tuple, or -1 when both draws refused.

    Example:
        >>> rest_tcp = task.agent.base_link.pose.sp.inv() * task.agent.tcp.pose.sp  # doctest: +SKIP
        >>> ...                                                                   # doctest: +SKIP
        >>> fold_via_tcp(env, planner, task, rest_tcp, stage="fold", who=WHO)     # doctest: +SKIP
    """
    plan_joints(env, planner, task, {"torso_lift_joint": float(rest_torso)},
                label=f"{stage}: un-duck", who=who)
    goal = task.agent.base_link.pose.sp * rest_tcp
    r = planner.static_manipulation(goal, disable_lift_joint=True)
    if r == -1:
        r = planner.static_manipulation(goal, disable_lift_joint=True)
    say(env, who, f"{stage}: fold via tcp {'ok' if r != -1 else 'REFUSED'}")
    normalize_continuous_arm_joints(env, planner, task, who=who)
    rest_q = np.asarray(task.agent.keyframes["rest"].qpos,
                        dtype=np.float64).reshape(-1)
    jm = task.agent.robot.active_joints_map
    arm_names = list(task.agent.controller.controllers["arm"].config.joint_names)
    targets = {n: float(rest_q[jm[n].active_index[0].item()])
               for n in arm_names}
    plan_joints(env, planner, task, targets, label=f"{stage}: branch line",
                tries=1, who=who)
    q_now = _np(task.agent.robot.get_qpos()).reshape(-1)
    d_rest = float(np.linalg.norm(q_now[3:11] - rest_q[3:11]))
    say(env, who, f"{stage}: post-branch arm_vs_rest", d=round(d_rest, 3))
    if d_rest > 1.0:
        # the tf-sweep S2/S6 mechanism: a flaked branch line leaves the arm
        # on a weird branch and every following fetch pre refuses. One full
        # retry round — the TCP screw moves the arm first, so the line
        # starts from a different config.
        say(env, who, f"{stage}: branch line flaked — one retry round")
        goal2 = task.agent.base_link.pose.sp * rest_tcp
        r2 = planner.static_manipulation(goal2, disable_lift_joint=True)
        if r2 == -1:
            planner.static_manipulation(goal2, disable_lift_joint=True)
        normalize_continuous_arm_joints(env, planner, task, who=who)
        plan_joints(env, planner, task, targets,
                    label=f"{stage}: branch line retry", tries=1, who=who)
        q_now = _np(task.agent.robot.get_qpos()).reshape(-1)
        say(env, who, f"{stage}: post-retry arm_vs_rest",
            d=round(float(np.linalg.norm(q_now[3:11] - rest_q[3:11])), 3))
    return r
