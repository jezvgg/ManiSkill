import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import cast

import gymnasium as gym
import numpy as np
import sapien
import torch
from trimesh.primitives import Box

from mani_skill.agents.robots import Fetch
from mani_skill.envs.tasks import MyRoboCasaSceneTakeItBack
from mani_skill.examples.motionplanning.fetch.extand import (
    FetchMotionPlanningSapienSolver,
)
from mani_skill.examples.motionplanning.fetch.utils import (
    compute_box_grasp_thin_side_info,
)
from utils.logging_utils import PlannerLogger, StreamingVideoRecorder, capture_stdout
from utils.planners_utils import (
    _rotate_base_to,
    _screw_base_translate,
    _velocity_segment,
    lower_torso_smooth,
    retract_arm_lift_torso,
    drive_base_to_position,
    _base_cmd,
)

FINGER_LENGTH = 0.025
# max horizontal base->cup distance at which the fallback arm re-grasp drives
# the base so the cup is inside the arm's reachable workspace. The straight-arms
# TCP sits ARM_OFFSET (1.128 m) north of the base but mplib's real reachable
# horizontal is ~1.086 m, so the gripper-at-cup park (base->cup = ARM_OFFSET)
# leaves the cup ~2-5 cm past reach on some seeds (11/32/41/47 -> "IK Failed").
GRASP_STANDOFF = 1.00
# lift offsets (world), tried in order: lifting straight up swings the elbow
# into the fixture stack and the cup into the cabinet door behind the counter,
# so prefer up + back toward the base (south)
LIFT_OFFSETS = [
    np.array([0.0, -0.20, 0.10]),
    np.array([0.0, -0.22, 0.08]),
    np.array([0.0, -0.18, 0.13]),
    np.array([0.0, -0.15, 0.15]),
    np.array([0.0, -0.12, 0.13]),
    np.array([0.0, -0.10, 0.15]),
]
# straight-up lift offsets (world): used at the tray so the cup KEEPS its y
# (the whole task uses one release line, and a pull-back would push the cup
# 0.2 m south, outside the return tolerance). The cup is held high so the
# return transport's fixed-arm screw keeps its swept volume clear.
LIFT_OFFSETS_STRAIGHT = [
    np.array([0.0, 0.0, 0.15]),
    np.array([0.0, 0.0, 0.12]),
    np.array([0.0, 0.0, 0.18]),
    np.array([0.0, -0.05, 0.15]),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Motion planner for MyRoboCasa_TakeItBack-v1 scene")
    parser.add_argument("--seed", type=int, default=3, help="Random seed (default: 3)")
    parser.add_argument("--render-mode", type=str, default="rgb_array",
                        choices=["rgb_array", "human", "sensors"],
                        help="Render mode (default: rgb_array)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode in planner")
    parser.add_argument("--info", action="store_true", help="Print environment info in planner")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for log output (default: logs)")
    parser.add_argument("--log-freq", type=int, default=10, help="Write trajectory rows every N steps (default: 10)")
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable mp4 video recording (skip StreamingVideoRecorder: no render, faster)",
    )
    return parser.parse_args()


def _attach_cup(planner, cup):
    """Register the cup as attached to the gripper in the mplib planning world
    so lift/transport plans stop treating the grasped cup as an obstacle."""
    from mani_skill.examples.motionplanning.fetch.utils import attach_object
    from mplib.sapien_utils.conversion import convert_object_name

    gripper_link = next(
        l for l in planner.robot._objs[0].get_links() if l.name.endswith("gripper_link")
    )
    attach_object(
        planner.planner.planning_world,
        cup._objs[0],
        planner.robot._objs[0],
        gripper_link,
    )
    # Allow the attached cup to collide with EVERYTHING else in the planning
    # world. The cup rests on the counter (contact) and mplib treats resting
    # contact as a collision, so without this every lift/transport plan starts
    # in an invalid state and fails ("IK Failed"). The physical sim still
    # enforces the real contacts, and the cup only ever moves through free
    # space above the counter.
    acm = planner.planner.planning_world.get_allowed_collision_matrix()
    cup_name = convert_object_name(cup._objs[0])
    acm.set_default_entry(cup_name, True)


def _detach_cup(planner):
    """Release the cup from the planning world (after the gripper is opened)."""
    planner.planner.detach_object()


# grasp height raises (m) above the cup center, cycled across attempts: the
# cup rests ~2 cm above the counter/dishwasher and mplib's collision margin
# rejects candidates whose fingers graze them; a taller grasp clears the
# fixtures but has its own reach trade-offs, so several heights are tried
GRASP_RAISES = [0.04, 0.07, 0.02, 0.06]


def _grasp_pose(agent, obb, cup_center, ee_direction, target_closing, raise_z=0.04,
                back_off=0.1, lift_over=0.0, force_front=False):
    """Compute grasp + pre-grasp poses; flip the approach if the target ends up
    on the far side of the cup relative to the base. back_off is the pre-grasp
    standoff along the approach; lift_over raises the pre-grasp pose above the
    grasp (the approach is horizontal, so without it the pre-grasp sits at the
    cup's height: on the TRAY that puts the forearm at the tray's top edge and
    the IK rejects the pose - seeds 3/14/15 with the cup at the tray center).
    The regrasp passes lift_over ~0.12 so the arm approaches from above."""
    base_pos = agent.base_link.pose.p[0].cpu().numpy()
    grasp_info = compute_box_grasp_thin_side_info(
        obb,
        ee_direction=ee_direction,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
        ortho=True,
    )
    # the ROUND cup's OBB long axis is numerically arbitrary and often points
    # SIDEWAYS, so the "thin-side" grasp approaches the cup from the side
    # (seen on video, seed 14). The REGRASP forces the approach along the
    # base->cup direction (the FRONT, exactly like the first grasp) and
    # re-orthogonalizes the closing against it; the initial grasp keeps the
    # OBB-based approach (proven there).
    if force_front:
        approaching = ee_direction / np.linalg.norm(ee_direction)
        closing = grasp_info["closing"]
        closing = closing - (approaching @ closing) * approaching
        closing = closing / np.linalg.norm(closing)
        grasp_info["approaching"] = approaching
        grasp_info["closing"] = closing
        grasp_info["center"] = obb.center_mass.copy()
    grasp_pose = agent.build_grasp_pose(
        grasp_info["approaching"], grasp_info["closing"], grasp_info["center"]
    )
    grasp_pose.p[2] += raise_z
    # the offset is in the GRASP frame: z = approach (horizontal), y = closing
    # (vertical) - so +lift_over on y raises the pre-grasp above the tray
    reach_pose = grasp_pose * sapien.Pose([0, lift_over, -back_off])
    if np.dot(reach_pose.p - cup_center, base_pos - cup_center) < 0:
        print("Validation failed: grasp is diametrically opposite. Flipping approaching direction...")
        if force_front:
            approaching = -np.asarray(ee_direction, dtype=float)
            approaching /= np.linalg.norm(approaching)
            closing = closing - (approaching @ closing) * approaching
            closing = closing / np.linalg.norm(closing)
            grasp_pose = agent.build_grasp_pose(approaching, closing, obb.center_mass.copy())
        else:
            grasp_info = compute_box_grasp_thin_side_info(
                obb,
                ee_direction=-ee_direction,
                target_closing=target_closing,
                depth=FINGER_LENGTH,
                ortho=True,
            )
            grasp_pose = agent.build_grasp_pose(
                grasp_info["approaching"], grasp_info["closing"], grasp_info["center"]
            )
        grasp_pose.p[2] += raise_z
        reach_pose = grasp_pose * sapien.Pose([0, lift_over, -back_off])
    return grasp_pose, reach_pose


def _tcp_at(agent, target_pose, tol=0.03, z_only=False):
    """True if the TCP actually reached target_pose (within tol meters; when
    z_only, only the height must match).

    static_manipulation returns a success tuple even when its refinement gave
    up ("Robot is stuck") or the plan was only approximate, so the executed
    pose must be verified before a motion counts as done."""
    tcp = agent.tcp.pose.p[0].cpu().numpy()
    target = np.asarray(target_pose.p, dtype=float)
    if z_only:
        # pi-lens-ignore: unchecked-throwing-call-python
        return float(abs(tcp[2] - target[2])) <= tol
    # pi-lens-ignore: unchecked-throwing-call-python
    return float(np.linalg.norm(tcp - target)) <= tol


def _reach_and_grasp(env, planner, agent, obb, cup_center, ee_direction,
                      target_closing, unwenv, raises=GRASP_RAISES, back_off=0.1,
                      lift_over=0.0, use_fallback=True, allow_tray=False,
                      force_front=False):
    from mplib.sapien_utils.conversion import convert_object_name

    if allow_tray:
        # the tray regrasp: the arm reaches OVER the tray to the cup; without
        # this the mplib IK rejects every candidate as a tray collision
        # ("IK Failed" on every seed with the frontal approach). Allow
        # gripper/arm vs the tray BEFORE the reach so the IK can solve.
        try:
            acm = planner.planner.planning_world.get_allowed_collision_matrix()
            acm.set_default_entry(convert_object_name(unwenv.tray._objs[0]), True)
        except Exception:
            pass
    """Reach the pre-grasp pose, execute the grasp and close the gripper.
    The RRT planner is stochastic, so retry with flipped approach/closing and
    repeated attempts. The cup is re-measured AFTER every reach: the executed
    (often approximate) reach can knock the cup several cm, and grasping at a
    stale pose closes on empty space. When the arm is already AT the cup (the
    sink regrasp: the transport leaves the TCP next to the released cup) the
    reach is skipped and the grasp simply lowers onto the cup with the CURRENT
    gripper orientation (the OBB long-side axis of the round cup is
    numerically unstable and can demand an unreachable orientation). Returns
    the executed grasp pose, or None if all attempts failed."""
    for attempt in range(6):
        ed = ee_direction if attempt % 2 == 0 else -ee_direction
        tc = target_closing if attempt < 2 else -target_closing
        # re-read the cup pose (the previous attempt's reach may have knocked it)
        mesh = unwenv.cup.get_first_collision_mesh(to_world_frame=True)
        if mesh is not None:
            obb = mesh.bounding_box_oriented
            cup_center = obb.center_mass.copy()
        if use_fallback and attempt == 0 and _tcp_to(agent, cup_center) <= 0.25:
            # already at the cup: skip the reach, lower with the current
            # orientation (raise heights are cycled for fixture clearance).
            # ONLY on the first attempt: a transport-left orientation can be
            # misaligned with the cup and the closing fingers then PUSH the
            # cup across the tray instead of clamping it (verified seed 5: 6
            # fallback grasps all shoved the cup ~0.12 m); the later attempts
            # use the OBB-based grasp pose for a proper approach.
            raise_z = raises[attempt % len(raises)]
            grasp_pose = sapien.Pose(
                p=[cup_center[0], cup_center[1], cup_center[2] + raise_z],
                q=agent.tcp.pose.q[0].cpu().numpy(),
            )
        else:
            grasp_pose, reach_pose = _grasp_pose(
                agent, obb, cup_center, ed, tc,
                raise_z=raises[attempt % len(raises)], back_off=back_off,
                lift_over=lift_over, force_front=force_front,
            )
            res = env.log_motion("Reach cup", planner.static_manipulation, reach_pose,
                                 n_init_qpos=100, disable_lift_joint=False)
            if res == -1:
                continue
            # re-measure the cup AFTER the reach (it may have been knocked) and
            # re-aim the grasp at its actual position
            mesh = unwenv.cup.get_first_collision_mesh(to_world_frame=True)
            if mesh is not None:
                obb = mesh.bounding_box_oriented
                cup_center = obb.center_mass.copy()
            if _tcp_to(agent, cup_center) > 0.15:
                continue
            grasp_pose, _ = _grasp_pose(
                agent, obb, cup_center, ed, tc,
                raise_z=raises[attempt % len(raises)], back_off=back_off,
                lift_over=lift_over, force_front=force_front,
            )
        # Allow the gripper to touch the cup: the grasp pose inherently has the
        # fingers/gripper in contact with the cup, and mplib would otherwise
        # reject every IK candidate as a collision (the reach keeps the cup as
        # an obstacle, so the approach path avoids knocking it).
        acm = planner.planner.planning_world.get_allowed_collision_matrix()
        acm.set_default_entry(convert_object_name(unwenv.cup._objs[0]), True)
        # TWO-STEP grasp: first an intermediate pose 5 cm short of the cup,
        # then the final pose. The single ~10 cm closing motion repeatedly got
        # stuck in the mplib refinement ("Robot is stuck" - the TCP ended 10
        # cm short, the close missed; seeds 14/15 with the frontal approach).
        # Short 5 cm motions complete the refinement reliably.
        inter_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        for grasp_try in range(3):
            res = env.log_motion("Grasp cup", planner.static_manipulation, inter_pose,
                                 n_init_qpos=100, disable_lift_joint=False)
            if res == -1:
                continue
            res = env.log_motion("Grasp cup", planner.static_manipulation, grasp_pose,
                                 n_init_qpos=100, disable_lift_joint=False)
            if res != -1 and _tcp_to(agent, grasp_pose.p) <= 0.05 and abs(
                    agent.tcp.pose.p[0][2] - grasp_pose.p[2]) <= 0.02:
                planner.close_gripper()
                if bool(unwenv.agent.is_grasping(unwenv.cup).item()):
                    return grasp_pose
                # the motion executed but the fingers did not clamp the cup:
                # reopen and retry
                planner.open_gripper()
    return None


def _tcp_to(agent, point):
    """Distance from the TCP to a point (m)."""
    tcp = agent.tcp.pose.p[0].cpu().numpy()
    # pi-lens-ignore: unchecked-throwing-call-python
    return float(np.linalg.norm(tcp - np.asarray(point, dtype=float)))


def _base_near(env, target_xy, tol=0.2):
    """True if the base is within tol (m) of target_xy (xy only)."""
    base = env.unwrapped.agent.base_link.pose.p[0].cpu().numpy()
    # pi-lens-ignore: unchecked-throwing-call-python
    return float(np.linalg.norm(np.asarray(target_xy, dtype=float)[:2] - base[:2])) <= tol


def _base_heading(agent):
    """World yaw of the base x-axis (rad)."""
    xa = agent.base_link.pose.sp.to_transformation_matrix()[:3, 0]
    return np.arctan2(xa[1], xa[0])


def _lift_cup(env, planner, agent, grasp_pose, offsets=LIFT_OFFSETS, cup=None):
    """Lift the grasped cup, trying several offsets (up + back first), each
    twice: the RRT planner is stochastic and the workspace is tight. A motion
    only counts once the TCP actually reached the target HEIGHT (the solver
    reports success even when its refinement gave up half-way, and the executed
    lift can drift laterally by several cm while the height is correct)."""
    res = -1
    # pi-lens-ignore: unchecked-throwing-call-python
    cup_z0 = float(cup.pose.p[0][2]) if cup is not None else 0.0
    for _ in range(2):
        for off in offsets:
            lift_pose = sapien.Pose(grasp_pose.p + off, grasp_pose.q)
            res = env.log_motion("Lift cup", planner.static_manipulation, lift_pose,
                                 n_init_qpos=100, disable_lift_joint=False)
            if res != -1 and _tcp_at(agent, lift_pose, tol=0.04, z_only=True):
                if cup is not None:
                    # pi-lens-ignore: unchecked-throwing-call-python
                    cz = float(cup.pose.p[0][2])
                    tcp_p = agent.tcp.pose.p[0].cpu().numpy()
                    cup_p = cup.pose.p[0].cpu().numpy()
                    # pi-lens-ignore: unchecked-throwing-call-python
                    gap = float(np.linalg.norm(tcp_p - cup_p))
                    if cz < cup_z0 + 0.10 or gap > 0.12:
                        # POST-LIFT GRASP CONDITION: the cup must have risen
                        # with the TCP (>= 0.10 m) AND stay near it (TCP-cup
                        # gap <= 0.12 m; a held cup rides at ~0.05-0.11, a
                        # slipped one lags at 0.13+ - verified seed 17: the
                        # second lift lost the cup at gap 0.135 m and the
                        # transport then ran empty, and seed 18's loose grip
                        # lifted the cup only ~6 cm). Report a failure so the
                        # caller re-grasps locally instead of transporting a
                        # barely-held cup.
                        print(f"[INFO] lift: cup not held (cup z {cz:.3f} vs "
                              f"{cup_z0:.3f}, TCP-cup gap {gap:.3f}); re-grasping")
                        continue
                return res
    return res


def _transport_cup(env, planner, agent, aim_xy, arm_action, body_action,
                   gripper_action, target_yaw, margin=0.05):
    """Axis-aligned transport: the base heading is FIXED (the counter-facing
    yaw set by the approach), so the held cup's world offset from the base is
    constant. The cup reaches aim_xy via an L path - the y-drive first
    (perpendicular to the counter, forward/backward), then the x-drive (along
    the counter). No rotations, no diagonals, no screw plans: the closed-loop
    velocity segment steers each axis. Returns 0 on convergence, -1 otherwise."""
    unwenv = env.unwrapped
    base = agent.base_link.pose.p[0].cpu().numpy()
    cup = unwenv.cup.pose.p[0].cpu().numpy()
    # the fixed cup offset in the world (heading fixed, arm rigid)
    off = cup[:2] - base[:2]
    base_aim = np.asarray(aim_xy, dtype=float) - off
    for tgt in ([base[0], base_aim[1], 0.0], [base_aim[0], base_aim[1], 0.0]):
        # the cup-attach monitor: the regrasp's grip can be weak and a lost
        # cup would leave the base driving on with the cup behind
        tcp = agent.tcp.pose.p[0].cpu().numpy()
        cup = unwenv.cup.pose.p[0].cpu().numpy()
        # pi-lens-ignore: unchecked-throwing-call-python
        if float(np.linalg.norm(tcp - cup)) > 0.20:
            print(f"[INFO] transport: cup lost (TCP-cup "
                  f"{np.linalg.norm(tcp - cup):.2f} m); aborting")
            return -1
        if np.linalg.norm(agent.base_link.pose.p[0].cpu().numpy()[:2] - tgt[:2]) < 0.12:
            continue
        res = _velocity_segment(env, planner, tgt, arm_action, body_action,
                                gripper_action, target_yaw=target_yaw)
        if res == -1:
            return -1
    return 0


def planning(env, seed, debug=False, vis=None, info=False):
    vis = vis or env.unwrapped.render_mode == "human"

    unwenv: MyRoboCasaSceneTakeItBack = env.unwrapped
    obs, _ = env.reset(seed=seed, options={"reconfigure": True})
    agent: Fetch = cast(Fetch, unwenv.agent)  # captured after reconfigure reset

    tray_center = unwenv.tray.pose.p[0].cpu().numpy()
    init_cup = unwenv.cup_pos[0]

    planner = FetchMotionPlanningSapienSolver(
        env,
        base_pose=agent.robot.pose.sp,
        vis=vis,
        print_env_info=info,
        debug=debug,
    )

    def _sync():
        getattr(planner.planner, "update_from_simulation")()
    env.track_object(unwenv.cup, "cup")
    env.track_object(unwenv.tray, "tray")
    env.track_object(agent.tcp, "robot_tcp")
    env.track_object(agent.base_link, "robot_base")
    env.log_event("start", "Planning started")

    # ==================================================================== #
    # STRAIGHT-ARM GEOMETRY: the arm stays in the home (straight) config for
    # the WHOLE task. The TCP rides at (1.128, 0, 0.786 + torso) in the base
    # frame (measured), so the cup offset from the base is constant and all
    # horizontal positioning is done with the base (forward/backward =
    # north/south, sideways = east/west), the vertical with the torso only.
    # The robot rotates ONCE to face north (the arm into the counter) with
    # the empty gripper, then never rotates again. No arm reconfiguration,
    # no mplib arm motions, no sharp moves: every stage is a smooth scripted
    # drive / torso ramp, verified (cup z, TCP-cup gap) before the next one.
    # ==================================================================== #
    ARM_OFFSET = 1.128    # straight-arm TCP offset from the base center (m);
    # the arm points NORTH (into the counter) via the shoulder pan, while
    # the robot itself faces EAST (parallel to the counter)
    TORSO_TRANSPORT = 0.386  # the torso MAX: the arm ~1.19 m high - well above
    # the stove top (1.08) and the stack cabinets (0.89) that the straight
    # arm sweeps past during the base drives (the mplib collision models are
    # conservative, so the extra height reduces the false positives)
    TORSO_GRASP = 0.21      # the jaws' mid at the cup center (~1.0 m)
    TORSO_LOW = 0.02        # the torso floor for the placement ramps

    def hold_a():
        return getattr(agent.controller, "controllers")["arm"].qpos[0].cpu().numpy()

    def hold_b():
        return getattr(agent.controller, "controllers")["body"].qpos[0].cpu().numpy().copy()

    def step_hold(torso_target=None):
        a = np.zeros(14)
        a[:7] = hold_a()
        a[7] = planner.gripper_state
        b = hold_b()
        if torso_target is not None:
            b[2] = torso_target
        a[8:11] = b
        env.step(a)

    def ramp_torso(target, steps=150):
        # ramp the torso TARGET gradually: a fixed target makes the PD snap
        # the torso (and the arm with it) fast, which shoves the base around;
        # stepping the target by small increments keeps the motion slow
        start = hold_b()[2]
        for i in range(steps):
            b = hold_b()
            b[2] = start + (target - start) * ((i + 1) / steps)
            env.step(np.hstack([hold_a(), planner.gripper_state, b, _base_cmd()]))
        _sync()
        # pi-lens-ignore: unchecked-throwing-call-python
        return float(agent.tcp.pose.p[0][2])

    def ramp_arm(target, steps=120):
        """Move arm to a joint waypoint without teleporting its target."""
        target = np.asarray(target, dtype=float)
        start = hold_a()
        for i in range(steps):
            arm = start + (target - start) * ((i + 1) / steps)
            env.step(np.hstack([arm, planner.gripper_state, hold_b(), _base_cmd()]))
        _sync()

    def descend_to_grasp(target_pose, xy_tol=0.05):
        """Use torso for final vertical closure when arm IK stalls."""
        target = np.asarray(target_pose.p, dtype=float)
        tcp = agent.tcp.pose.p[0].cpu().numpy()
        if np.linalg.norm(tcp[:2] - target[:2]) > xy_tol:
            return False
        # pi-lens-ignore: unchecked-throwing-call-python
        drop = float(tcp[2] - target[2])
        if drop > 0.015:
            body_z = max(0.0, hold_b()[2] - drop)
            ramp_torso(body_z, steps=30)
        return _tcp_to(agent, target) <= 0.05

    def drive_to(tgt, tol=0.04, min_improve=0.02):
        """Axis-aligned base drive, holding the arm + current torso,
        heading fixed north. Returns 0 on convergence (final dist < tol)."""
        return _velocity_segment(
            env, planner, np.asarray(tgt, dtype=float), hold_a(), hold_b(),
            planner.gripper_state, target_yaw=0.0, tol=tol,
            min_improve=min_improve,
        )

    def l_drive(aim_xy, tol=0.04):
        """Move held cup with fixed-arm screw segments and a 5 cm detour."""
        aim = np.asarray(aim_xy, dtype=float)
        base = agent.base_link.pose.p[0].cpu().numpy()
        targets = (
            np.array([base[0], aim[1] + 0.05, 0.0]),
            np.array([aim[0], aim[1] + 0.05, 0.0]),
            np.array([aim[0], aim[1], 0.0]),
        )
        for target in targets:
            current = agent.base_link.pose.p[0].cpu().numpy()
            if np.linalg.norm(target[:2] - current[:2]) <= tol:
                continue
            if _screw_base_translate(planner, target) != 0:
                return -1
            _sync()
        final = agent.base_link.pose.p[0].cpu().numpy()[:2]
        return 0 if np.linalg.norm(final - aim[:2]) <= max(tol, 0.06) else -1

    def lower_torso_until_cup_rests(surface_top_z):
        """Slowly lower the torso (grasped cup descends with the jaws) until
        the cup rests on the surface below (cup z stops decreasing)."""
        # pi-lens-ignore: unchecked-throwing-call-python
        rest_z = float(surface_top_z + unwenv.cup_half[2])
        for _ in range(300):
            # pi-lens-ignore: unchecked-throwing-call-python
            cz = float(unwenv.cup.pose.p[0][2])
            if cz <= rest_z + 0.01:
                break
            b = hold_b()
            if b[2] <= TORSO_LOW + 0.001:
                # No further torso motion is possible; avoid repeating no-op
                # steps when a held cup cannot reach the surface.
                break
            b[2] -= 0.005
            a = np.zeros(14)
            a[:7] = hold_a()
            a[7] = planner.gripper_state
            a[8:11] = b
            env.step(a)
        _sync()
        # pi-lens-ignore: unchecked-throwing-call-python
        return float(unwenv.cup.pose.p[0][2])

    def cup_held():
        return bool(unwenv.agent.is_grasping(unwenv.cup).item())

    def tcp_cup_gap():
        t = agent.tcp.pose.p[0].cpu().numpy()
        c = unwenv.cup.pose.p[0].cpu().numpy()
        # pi-lens-ignore: unchecked-throwing-call-python
        return float(np.linalg.norm(t - c))

    def report_stage(name):
        b = agent.base_link.pose.p[0].cpu().numpy()
        c = unwenv.cup.pose.p[0].cpu().numpy()
        print(f"[STAGE] {name}: base ({b[0]:.3f},{b[1]:.3f}) "
              f"cup ({c[0]:.3f},{c[1]:.3f},{c[2]:.3f}) gap {tcp_cup_gap():.3f} "
              f"grasped={cup_held()}")

    # ------------------------------------------------------------------ #
    # STAGE 0: raise torso above counter, rotate ONCE to face north, then
    # fold elbow slightly. The bent carry pose shortens the arm offset while
    # keeping elbow and gripper above fixtures.
    # ------------------------------------------------------------------ #
    env.log_event("phase", "Stage 0: raise torso, align along the counter")
    ramp_torso(TORSO_TRANSPORT, steps=150)
    # the robot faces EAST (parallel to the counter front edge): its
    # forward/backward drives then run along the counter (left/right of the
    # countertop) and its side faces the counter. The straight arm is then
    # swung 90 deg at the shoulder (pan) so it points INTO the counter
    # (north), then bend the elbow in a high, compact carry pose.
    _rotate_base_to(env, planner, np.array([1.0, 0.0, 0.0]))
    _sync()
    # 85 deg, NOT 90: the shoulder pan limit is +-1.6056 rad (+-92 deg), and
    # a pan pinned at exactly 90 deg leaves the IK no room to swing the arm
    # to the cup (the align failed with "IK Failed" - the pan was at the
    # joint limit). 85 deg keeps the arm pointing into the counter while
    # giving the IK ~7 deg of swing room. The pan target is ramped gradually
    # so the arm swing is slow and does not shove the base.
    _pan1 = np.deg2rad(85)
    _bent_arm = hold_a()
    _bent_arm[0] = _pan1
    _bent_arm[1] = -0.40
    _bent_arm[3] = 0.80
    _bent_arm[5] = -0.40
    # Pan and bend together: same safe high-torso corridor, one arm path
    # instead of two sequential ramps.
    ramp_arm(_bent_arm, steps=180)
    report_stage("0 raise+align+bend")

    # ------------------------------------------------------------------ #
    # STAGE 1: drive sideways (east-west, along the counter) to the cup's x
    # at the safe south line (y = cup_y - ARM_OFFSET - 0.4), then STAGE 2:
    # lower the torso to the grasp height and drive to the PRE-GRASP along a
    # SAFE corridor. The jaws' plates span +-6.45 cm from the gripper, so any
    # drive that passes closer than ~10 cm to the cup pushes it with a plate
    # edge (verified: the cup slid 0.2 m). The offset corridor (cup_x + 0.15)
    # keeps the plates clear during the y-leg and the x-leg.
    # The screw drive (drive_base_to_position) is used instead of the closed-
    # loop velocity segment: it converges reliably (~0.15 m, the arm's fine
    # alignment covers the rest) and keeps the heading fixed.
    # ------------------------------------------------------------------ #
    env.log_event("phase", "Stage 1: drive to the cup x")
    cup_xy = unwenv.cup.pose.p[0].cpu().numpy()[:2]
    south_line = cup_xy[1] - ARM_OFFSET - 0.40
    # the base parks WEST of the cup: the pan-85 arm puts the gripper 10 cm
    # west of the base, so the y-leg must pass WEST of the cup (the plates
    # bracket it with 2.6-3.1 cm clearance); an east corridor would drive
    # the plates through the cup (verified: the cup was knocked over).
    # L-path: drive SOUTH first (away from the counter) to the line, then
    # EAST/WEST to the cup's x - a direct diagonal let the base cross the
    # y=-0.95 counter guard on seed 9 (the screw's rotate-retry slides)
    _b1 = agent.base_link.pose.p[0].cpu().numpy()
    # y_guard=False for the south leg: the robot can SPAWN north of the
    # counter line (y > -0.95) and must be allowed to drive away from the
    # counter first (verified: seed 9 aborted before moving - the guard fired
    # on the starting position)
    res = env.log_motion(
        "Stage 1 drive", l_drive, np.array([_b1[0], south_line])
    )
    if res == 0:
        res = env.log_motion(
            "Stage 1 drive", l_drive,
            np.array([cup_xy[0] - 0.15, south_line])
        )
    if res != 0:
        print("Stage 1 drive failed; aborting")
        env.log_event("error", "Stage 1 drive failed")
        success = bool(unwenv.evaluate()["success"].item())
        env.log_event("result", "Task aborted", success=success)
        env.reset()
        return success
    report_stage("1 at cup x")

    env.log_event("phase", "Stage 2: pre-grasp position")
    # allow the cup in the planning world from here on: the screw drives near
    # the cup fail their start-state collision check (gripper <-> cup) and
    # fling the base around (verified: the correction drive ended 0.5 m away)
    from mplib.sapien_utils.conversion import convert_object_name
    _acm = planner.planner.planning_world.get_allowed_collision_matrix()
    _acm.set_default_entry(convert_object_name(unwenv.cup._objs[0]), True)
    # allow the straight arm vs the counter fixtures (the stack doors, the
    # dishwasher, the stove...): the arm rides ~1.19 m high - physically
    # above them - but the mplib collision models are conservative and flag
    # false positives that fail every screw plan near the counter (verified:
    # seeds 9/17 - the gripper vs the stack hingedoor / the dishwasher).
    # NOT the walls/floor - those collisions are real.
    for _nm, _act in unwenv.scene.actors.items():
        if any(_k in _nm for _k in ("counter", "stack", "stove", "dishwasher",
                                    "sink", "cab", "fridge", "paper_towel",
                                    "tray")):
            try:
                _acm.set_default_entry(convert_object_name(_act._objs[0]), True)
            except Exception:
                pass
    # drive with arm HIGH (the plates cannot touch cup), lower torso to grasp
    # height only AFTER base is parked. Recompute pre-grasp from live bent-arm
    # TCP offset; fixed straight-arm geometry is no longer valid.
    cc = unwenv.cup.pose.p[0].cpu().numpy()
    arm_xy = (
        agent.tcp.pose.p[0].cpu().numpy()[:2]
        - agent.base_link.pose.p[0].cpu().numpy()[:2]
    )
    pre = np.r_[cc[:2] - arm_xy, 0.0]
    res = env.log_motion("Stage 2 pre-grasp", l_drive, pre)
    _sync()
    # correction loop: the screw drive lands ~15 cm off, but the arm's align
    # (pan-85, near the +-92 deg joint limit) can only cover ~8-10 cm, so
    # re-aim the base until the GRIPPER is within ~5 cm of the cup
    for _c in range(3):
        _g = agent.tcp.pose.p[0].cpu().numpy()[:2]
        _cc2 = unwenv.cup.pose.p[0].cpu().numpy()[:2]
        # pi-lens-ignore: unchecked-throwing-call-python
        if float(np.linalg.norm(_g - _cc2)) <= 0.05:
            break
        _b = agent.base_link.pose.p[0].cpu().numpy()[:2]
        # aim the gripper 6 cm west-south of the cup: the drives' overshoot
        # keeps the plates clear of the cup (a 3 cm aim let the plates push
        # the cup ~12 cm)
        _aim2 = np.array([_b[0] + (_cc2[0] - 0.06 - _g[0]),
                          _b[1] + (_cc2[1] - 0.06 - _g[1]), 0.0])
        env.log_motion("Stage 2 correction", l_drive, _aim2)
        _sync()
    # Leave torso high; first Stage 3 screw can lower torso and align arm in
    # one geometry path. Failed reach falls back to the old torso ramp.
    report_stage("2 high pre-grasp")

    # ------------------------------------------------------------------ #
    # STAGE 3: grasp - a SHORT two-step straight-arm motion (inter 5 cm
    # above the cup, then the final 5 cm drop; the arm stays straight, a
    # micro-translation) aligns the jaws on the cup's measured position,
    # then close. Retry with height variations and re-measurement.
    # ------------------------------------------------------------------ #
    env.log_event("phase", "Stage 3: grasp")
    def run_grasp():
        got = False
        torso_ready = False
        for attempt in range(10):
            cc = unwenv.cup.pose.p[0].cpu().numpy()
            raise_z = [0.02, 0.06, 0.02, 0.08, 0.04, 0.0, 0.05, 0.03, 0.07, 0.01][attempt]
            q_now = agent.tcp.pose.q[0].cpu().numpy()
            final = sapien.Pose(p=[cc[0], cc[1], cc[2] + raise_z], q=q_now)
            inter = sapien.Pose(p=[cc[0], cc[1], cc[2] + raise_z + 0.05], q=q_now)
            r1 = env.log_motion("Stage 3 align", planner.static_manipulation,
                                inter, n_init_qpos=100, disable_lift_joint=False)
            _sync()
            if not torso_ready and (r1 == -1 or not _tcp_at(agent, inter, tol=0.05)):
                # The high-to-intermediate screw was not accurate; restore the
                # proven low-torso contract before retrying live geometry.
                ramp_torso(TORSO_GRASP, steps=120)
                torso_ready = True
                continue
            torso_ready = True
            r2 = -1 if r1 == -1 else env.log_motion(
                "Stage 3 align", planner.static_manipulation, final,
                n_init_qpos=100, disable_lift_joint=False)
            _sync()
            aligned = r2 != -1 and _tcp_to(agent, final.p) <= 0.04
            if not aligned:
                cc_live = unwenv.cup.pose.p[0].cpu().numpy()
                live_final = sapien.Pose(
                    p=[cc_live[0], cc_live[1], cc_live[2] + raise_z],
                    q=agent.tcp.pose.q[0].cpu().numpy(),
                )
                aligned = descend_to_grasp(live_final)
            if aligned:
                planner.close_gripper()
                _sync()
                if cup_held():
                    got = True
                    break
                planner.open_gripper()
                _sync()
        return got

    def fallback_grasp():
        # drive the base closer (arm HIGH - the mid-grasp low-torso drive near
        # the counter fails to move the base) so the cup is inside the
        # workspace, then re-run the arm align. Returns True if the cup is held.
        ramp_torso(TORSO_TRANSPORT, steps=150)
        _b = agent.base_link.pose.p[0].cpu().numpy()[:2]
        _cc = unwenv.cup.pose.p[0].cpu().numpy()[:2]
        # pi-lens-ignore: unchecked-throwing-call-python
        if float(np.linalg.norm(_cc - _b)) > GRASP_STANDOFF:
            _aim = np.array([_cc[0] + 0.10, _cc[1] - GRASP_STANDOFF, 0.0])
            env.log_motion("fallback reach", l_drive, _aim[:2], 0.04)
            _sync()
        # Move the base using the measured TCP/cup offset, then close from the
        # already aligned carry orientation before attempting a new IK pose.
        for _ in range(2):
            tcp_xy = agent.tcp.pose.p[0].cpu().numpy()[:2]
            cup_xy = unwenv.cup.pose.p[0].cpu().numpy()[:2]
            # pi-lens-ignore: unchecked-throwing-call-python
            if float(np.linalg.norm(tcp_xy - cup_xy)) <= 0.05:
                break
            base_xy = agent.base_link.pose.p[0].cpu().numpy()[:2]
            env.log_motion(
                "fallback center", l_drive,
                base_xy + cup_xy - tcp_xy, 0.04
            )
            _sync()
        ramp_torso(TORSO_GRASP, steps=120)
        for _ in range(3):
            planner.close_gripper()
            _sync()
            if cup_held() and tcp_cup_gap() <= 0.08:
                return True
            planner.open_gripper()
            _sync()
        return run_grasp()

    def alternate_grasp():
        """Try a live front-facing grasp after repeated vertical stalls."""
        mesh = unwenv.cup.get_first_collision_mesh(to_world_frame=True)
        if mesh is None:
            return False
        obb = mesh.bounding_box_oriented
        cc = obb.center_mass.copy()
        ed = cc - agent.tcp.pose.p[0].cpu().numpy()
        ed[2] = 0.0
        if np.linalg.norm(ed) < 1e-6:
            ed = np.array([0.0, 1.0, 0.0])
        ed /= np.linalg.norm(ed)
        closing = np.cross(np.array([0.0, 0.0, 1.0]), ed)
        closing /= np.linalg.norm(closing)
        for raise_z in (0.02, 0.06, 0.04):
            grasp_pose, reach_pose = _grasp_pose(
                agent, obb, cc, ed, closing,
                raise_z=raise_z, back_off=0.06, force_front=True,
            )
            r1 = env.log_motion(
                "Stage 3 alternate approach", planner.static_manipulation,
                reach_pose, n_init_qpos=100, disable_lift_joint=False,
            )
            _sync()
            r2 = -1 if r1 == -1 else env.log_motion(
                "Stage 3 alternate grasp", planner.static_manipulation,
                grasp_pose, n_init_qpos=100, disable_lift_joint=False,
            )
            _sync()
            if r2 != -1 and _tcp_at(agent, grasp_pose, tol=0.05):
                planner.close_gripper()
                _sync()
                if cup_held():
                    return True
                planner.open_gripper()
                _sync()
        return False

    got = run_grasp()
    if not got:
        got = alternate_grasp()
    if not got:
        # FALLBACK: the primary straight-arm grasp could not reach the cup - it
        # sits just past the arm's reachable envelope (base->cup = ARM_OFFSET
        # ~2-5 cm beyond mplib's real reach). Drive the base closer so the cup
        # is inside the workspace, then re-run the arm align. Only fires when
        # the primary grasp failed, so seeds that grasp fine are untouched.
        env.log_event("phase", "Stage 3: fallback arm regrasp")
        got = fallback_grasp()
    if not got:
        print("Grasp failed after retries; aborting")
        env.log_event("error", "Grasp failed")
        success = bool(unwenv.evaluate()["success"].item())
        env.log_event("result", "Task aborted", success=success)
        env.reset()
        return success
    report_stage("3 grasped")

    # ------------------------------------------------------------------ #
    # STAGE 4: lift - raise the torso to the transport height, verify the
    # cup rose with the jaws (>= 0.08 m) and stays near the TCP.
    # ------------------------------------------------------------------ #
    env.log_event("phase", "Stage 4: lift")
    # pi-lens-ignore: unchecked-throwing-call-python
    cup_z0 = float(unwenv.cup.pose.p[0][2])
    ramp_torso(TORSO_GRASP + 0.04, steps=40)
    # pi-lens-ignore: unchecked-throwing-call-python
    probe_cz = float(unwenv.cup.pose.p[0][2])
    if probe_cz >= cup_z0 + 0.015 and tcp_cup_gap() <= 0.12:
        ramp_torso(TORSO_TRANSPORT, steps=110)
    else:
        ramp_torso(TORSO_GRASP, steps=40)
    # pi-lens-ignore: unchecked-throwing-call-python
    cz = float(unwenv.cup.pose.p[0][2])
    if cz < cup_z0 + 0.05 or tcp_cup_gap() > 0.12:
        # RE-GRASP FALLBACK: the cup didn't ride up with the jaws - a
        # false-positive grasp (`is_grasping` fired but the cup never clamped,
        # slipping out on the raise) - seed 48. Instead of aborting, re-grasp
        # (drive base closer + re-run the align) and retry the lift once.
        env.log_event("phase", "Stage 4: re-grasp (lift detect)")
        ramp_torso(TORSO_GRASP, steps=80)
        if fallback_grasp():
            # pi-lens-ignore: unchecked-throwing-call-python
            cup_z0 = float(unwenv.cup.pose.p[0][2])
            ramp_torso(TORSO_TRANSPORT, steps=150)
            # pi-lens-ignore: unchecked-throwing-call-python
            cz = float(unwenv.cup.pose.p[0][2])
    if cz < cup_z0 + 0.05 or tcp_cup_gap() > 0.12:
        print(f"Lift failed (cup z {cz:.3f} vs {cup_z0 + 0.05:.3f}, "
              f"gap {tcp_cup_gap():.3f}); aborting")
        env.log_event("error", "Lift failed")
        success = bool(unwenv.evaluate()["success"].item())
        env.log_event("result", "Task aborted", success=success)
        env.reset()
        return success
    report_stage("4 lifted")

    # ------------------------------------------------------------------ #
    # STAGE 5: transport to the tray - back up (south) for clearance, then
    # the L-drive to (tray_x, tray_y - ARM_OFFSET): the cup over the tray
    # center. Cup-attach monitor after each leg.
    # ------------------------------------------------------------------ #
    env.log_event("phase", "Stage 5: transport to tray")
    b = agent.base_link.pose.p[0].cpu().numpy()
    res = env.log_motion("Stage 5 back up", drive_to,
                         np.array([b[0], b[1] - 0.15, 0.0]), 0.10, 0.01)
    # the cup's z must stay near the stage-4 reference (the arm already
    # lifted it; the back-up does not change the z - the old check compared
    # against a pre-raise reference and falsely fired after the arm lift)
    # pi-lens-ignore: unchecked-throwing-call-python
    if res != 0 or tcp_cup_gap() > 0.15 or float(unwenv.cup.pose.p[0][2]) < cup_z0 + 0.04:
        print("Stage 5 back-up failed / cup lost; aborting")
        env.log_event("error", "Stage 5 back-up failed")
        success = bool(unwenv.evaluate()["success"].item())
        env.log_event("result", "Task aborted", success=success)
        env.reset()
        return success
    # aim the BASE so the CUP (at its live offset from the base - the arm
    # config after the grasp differs from the ideal straight offset) lands
    # on the tray center
    _off = unwenv.cup.pose.p[0].cpu().numpy()[:2] - agent.base_link.pose.p[0].cpu().numpy()[:2]
    aim_tray = np.array([tray_center[0] - _off[0], tray_center[1] - _off[1]])
    res = env.log_motion("Stage 5 drive to tray", l_drive, aim_tray, 0.10)
    # closed-loop correction: the base drives land 3-30 cm off, but the cup's
    # base must sit fully on the tray (center within ~0.105 m); re-aim at the
    # CUP's live error and re-drive up to twice
    for _c in range(2):
        # pi-lens-ignore: unchecked-throwing-call-python
        if float(np.linalg.norm(unwenv.cup.pose.p[0].cpu().numpy()[:2] - tray_center[:2])) <= 0.02:
            break
        _off = unwenv.cup.pose.p[0].cpu().numpy()[:2] - agent.base_link.pose.p[0].cpu().numpy()[:2]
        _aim2 = np.array([tray_center[0] - _off[0], tray_center[1] - _off[1]])
        res = env.log_motion("Stage 5 correction", l_drive, _aim2, 0.10)
        _sync()
    # pi-lens-ignore: unchecked-throwing-call-python
    if res != 0 or tcp_cup_gap() > 0.15 or float(unwenv.cup.pose.p[0][2]) < cup_z0 + 0.04:
        print("Stage 5 drive to tray failed / cup lost; aborting")
        env.log_event("error", "Stage 5 drive to tray failed")
        success = bool(unwenv.evaluate()["success"].item())
        env.log_event("result", "Task aborted", success=success)
        env.reset()
        return success
    report_stage("5 at tray")

    # ------------------------------------------------------------------ #
    # STAGE 6: place - lower the torso SLOWLY until the cup rests on the
    # tray (the cup z stops decreasing).
    # ------------------------------------------------------------------ #
    env.log_event("phase", "Stage 6: lower onto tray")
    # pi-lens-ignore: unchecked-throwing-call-python
    tray_top = float(unwenv.tray.pose.p[0][2] + unwenv.tray_half[2])
    lower_torso_until_cup_rests(tray_top)
    report_stage("6 on tray")

    # ------------------------------------------------------------------ #
    # STAGE 7: release - partial open (the jaws just off the cup), no motion.
    # ------------------------------------------------------------------ #
    env.log_event("phase", "Stage 7: release")
    # VERTICAL release: open the jaws slightly (the squeeze released), then
    # LIFT the torso so the plates rise off the cup. A lateral jaw-open pops
    # the round cup sideways and spins it (measured: 6 cm pop, av ~2.6 rad/s
    # that never decays on the smooth tray - the is_static latch can never
    # pass), because the plates' edges wedge the off-center cup. The vertical
    # lift presses the cup DOWN onto the tray (no lateral kick) and leaves it
    # free, so the cup stays put and quiet.
    for _i in range(30):
        _frac = (_i + 1) / 30
        planner.change_gripper_state(t=1, gripper_state=-1.0 + _frac * 1.85)  # pyright: ignore[reportArgumentType]
    _sync()
    # NO torso lift here: lifting the plates catches the cup's rim and drags
    # it up (measured: the cup rode up to z 1.09 with the rising plates and
    # stayed there, precariously held - the is_static latch failed). The jaw
    # open alone frees the cup at the rest height (measured pop ~1 cm, cup
    # quiet - the same behaviour as the old design's release that latched
    # with av 0.002).
    if cup_held():
        print("Release failed (still grasping); aborting")
    if cup_held():
        print("Release failed (still grasping); aborting")
        env.log_event("error", "Release failed")
        success = bool(unwenv.evaluate()["success"].item())
        env.log_event("result", "Task aborted", success=success)
        env.reset()
        return success
    # let the cup settle (the latch needs is_static); the release's pop can
    # spin the cup and the spin decays slowly (measured: av 0.14 after 125 s),
    # so wait long enough for the is_static latch
    for _ in range(2500):
        env.step(np.hstack([hold_a(), planner.gripper_state, hold_b(), _base_cmd()]))
        # pi-lens-ignore: ast-grep:unchecked-throwing-call-python
        if (float(torch.linalg.norm(unwenv.cup.linear_velocity, dim=1)[0]) <= 0.1
                # pi-lens-ignore: ast-grep:unchecked-throwing-call-python
                and float(torch.linalg.norm(unwenv.cup.angular_velocity, dim=1)[0]) <= 0.2):
            break
    unwenv.evaluate()
    report_stage("7 released")

    # ------------------------------------------------------------------ #
    # STAGE 8: regrasp - close the jaws (no motion), verify. The cup is
    # still between the jaws (partial open), so the close always catches it.
    # ------------------------------------------------------------------ #
    env.log_event("phase", "Stage 8: regrasp")
    # Close at release pose first. If contact is real, a short lift test
    # proves the cup is supported before any re-localization can disturb it.
    got8 = False
    lifted_from_tray = False
    # pi-lens-ignore: unchecked-throwing-call-python
    cup_z_before_regrasp = float(unwenv.cup.pose.p[0][2])
    planner.close_gripper()
    _sync()
    if cup_held():
        tcp8 = agent.tcp.pose.p[0].cpu().numpy()
        q8 = agent.tcp.pose.q[0].cpu().numpy()
        lift8 = sapien.Pose(p=tcp8 + np.array([0.0, 0.0, 0.07]), q=q8)
        r8 = env.log_motion(
            "Stage 8 lift test", planner.static_manipulation, lift8,
            n_init_qpos=100, disable_lift_joint=False,
        )
        _sync()
        # pi-lens-ignore: unchecked-throwing-call-python
        lifted_from_tray = (
            r8 != -1
            # pi-lens-ignore: unchecked-throwing-call-python
            and float(unwenv.cup.pose.p[0][2]) >= cup_z_before_regrasp + 0.05
            and tcp_cup_gap() <= 0.12
        )
        got8 = lifted_from_tray
        if not got8:
            planner.open_gripper()
            _sync()

    if not got8:
        # Re-localize only after the in-place lift test fails.
        ramp_torso(TORSO_GRASP, steps=100)
        for _a in range(3):
            planner.close_gripper()
            _sync()
            if cup_held():
                got8 = True
                break
            planner.open_gripper()
            _sync()
            cc8 = unwenv.cup.pose.p[0].cpu().numpy()
            q8 = agent.tcp.pose.q[0].cpu().numpy()
            fin8 = sapien.Pose(p=[cc8[0], cc8[1], cc8[2] + 0.02], q=q8)
            int8 = sapien.Pose(p=[cc8[0], cc8[1], cc8[2] + 0.07], q=q8)
            r1 = env.log_motion(
                "Stage 8 align", planner.static_manipulation, int8,
                n_init_qpos=100, disable_lift_joint=False,
            )
            _sync()
            r2 = -1 if r1 == -1 else env.log_motion(
                "Stage 8 align", planner.static_manipulation, fin8,
                n_init_qpos=100, disable_lift_joint=False,
            )
            _sync()
    if not got8:
        print("Regrasp failed; aborting")
        env.log_event("error", "Regrasp failed")
        success = bool(unwenv.evaluate()["success"].item())
        env.log_event("result", "Task aborted", success=success)
        env.reset()
        return success
    report_stage("8 regrasped")

    # ------------------------------------------------------------------ #
    # STAGE 9: lift from the tray, unless Stage 8 already proved attachment.
    # ------------------------------------------------------------------ #
    env.log_event("phase", "Stage 9: lift from tray")
    if lifted_from_tray:
        # The lift test already cleared the tray; a second lift adds no signal
        # and can turn a supported contact into a slip.
        # pi-lens-ignore: unchecked-throwing-call-python
        cz = float(unwenv.cup.pose.p[0][2])
        if tcp_cup_gap() > 0.12:
            print(f"Lift from tray failed (cup gap {tcp_cup_gap():.3f}); aborting")
            env.log_event("error", "Lift from tray failed")
            success = bool(unwenv.evaluate()["success"].item())
            env.log_event("result", "Task aborted", success=success)
            env.reset()
            return success
    else:
        # pi-lens-ignore: unchecked-throwing-call-python
        cup_z0 = float(unwenv.cup.pose.p[0][2])
        ramp_torso(TORSO_TRANSPORT, steps=150)
        # pi-lens-ignore: unchecked-throwing-call-python
        cz = float(unwenv.cup.pose.p[0][2])
        if cz < cup_z0 + 0.05 or tcp_cup_gap() > 0.12:
            print(f"Lift from tray failed (cup z {cz:.3f}); aborting")
            env.log_event("error", "Lift from tray failed")
            success = bool(unwenv.evaluate()["success"].item())
            env.log_event("result", "Task aborted", success=success)
            env.reset()
            return success
    if lifted_from_tray:
        # Stage 8 lift used arm IK; raise torso before base return so bent
        # forearm clears the fixture stack.
        ramp_torso(TORSO_TRANSPORT, steps=150)
    report_stage("9 lifted from tray")

    # ------------------------------------------------------------------ #
    # STAGE 10: return - back up, then the L-drive to the initial cup spot
    # (init_cup_x, init_cup_y - ARM_OFFSET): the cup over its start point.
    # ------------------------------------------------------------------ #
    env.log_event("phase", "Stage 10: drive back to initial spot")
    b = agent.base_link.pose.p[0].cpu().numpy()
    res = env.log_motion("Stage 10 back up", drive_to,
                         np.array([b[0], b[1] - 0.15, 0.0]), 0.10, 0.01)
    if res != 0 or tcp_cup_gap() > 0.15:
        print("Stage 10 back-up failed / cup lost; aborting")
        env.log_event("error", "Stage 10 back-up failed")
        success = bool(unwenv.evaluate()["success"].item())
        env.log_event("result", "Task aborted", success=success)
        env.reset()
        return success
    _off = unwenv.cup.pose.p[0].cpu().numpy()[:2] - agent.base_link.pose.p[0].cpu().numpy()[:2]
    aim_init = np.array([init_cup[0] - _off[0], init_cup[1] - _off[1]])
    res = env.log_motion("Stage 10 drive to initial", l_drive, aim_init, 0.10)
    for _c in range(2):
# pi-lens-ignore: ast-grep:unchecked-throwing-call-python
        if float(np.linalg.norm(unwenv.cup.pose.p[0].cpu().numpy()[:2] - init_cup[:2])) <= 0.06:
            break
        _off = unwenv.cup.pose.p[0].cpu().numpy()[:2] - agent.base_link.pose.p[0].cpu().numpy()[:2]
        _aim2 = np.array([init_cup[0] - _off[0], init_cup[1] - _off[1]])
        res = env.log_motion("Stage 10 correction", l_drive, _aim2, 0.10)
        _sync()
    if res != 0 or tcp_cup_gap() > 0.15:
        print("Stage 10 drive to initial failed / cup lost; aborting")
        env.log_event("error", "Stage 10 drive to initial failed")
        success = bool(unwenv.evaluate()["success"].item())
        env.log_event("result", "Task aborted", success=success)
        env.reset()
        return success
    report_stage("10 at initial spot")

    # ------------------------------------------------------------------ #
    # STAGE 11: place on the counter - lower slowly until the cup rests.
    # ------------------------------------------------------------------ #
    env.log_event("phase", "Stage 11: lower onto counter")
# pi-lens-ignore: ast-grep:unchecked-throwing-call-python
    counter_top = float(unwenv.counter_pos[2] + unwenv.counter_size[2] / 2)
    lower_torso_until_cup_rests(counter_top)
    report_stage("11 on counter")

    # ------------------------------------------------------------------ #
    # STAGE 12: release - full open (no motion), let the cup settle.
    # ------------------------------------------------------------------ #
    env.log_event("phase", "Stage 12: release")
    # vertical release (same as the tray release): open the jaws, lift the
    # torso - the plates rise off the cup, the cup stays put and quiet
    for _i in range(30):
        _frac = (_i + 1) / 30
        planner.change_gripper_state(t=1, gripper_state=-1.0 + _frac * 1.85)  # pyright: ignore[reportArgumentType]
    _sync()
    for _ in range(2500):
        env.step(np.hstack([hold_a(), planner.gripper_state, hold_b(), _base_cmd()]))
# pi-lens-ignore: ast-grep:unchecked-throwing-call-python
        if (float(torch.linalg.norm(unwenv.cup.linear_velocity, dim=1)[0]) <= 0.1
# pi-lens-ignore: ast-grep:unchecked-throwing-call-python
                and float(torch.linalg.norm(unwenv.cup.angular_velocity, dim=1)[0]) <= 0.2):
# pi-lens-ignore: ast-grep:unchecked-throwing-call-python
            break
# pi-lens-ignore: ast-grep:unchecked-throwing-call-python
    _av12 = float(torch.linalg.norm(unwenv.cup.angular_velocity, dim=1)[0])
    print(f"[INFO] stage 12 settle done, cup av={_av12:.3f} rad/s")
    report_stage("12 released")

    print("Task completed. Closing env...")
    ev = unwenv.evaluate()
    success = bool(ev["success"].item())
    print("Success:", success,
          "| placed_on_tray:", bool(unwenv.placed_on_tray.item()),
          "| cup xy:", np.round(unwenv.cup.pose.p[0].cpu().numpy()[:2], 3),
          # pi-lens-ignore: ast-grep:unchecked-throwing-call-python
          "z:", round(float(unwenv.cup.pose.p[0][2]), 3),
          # pi-lens-ignore: ast-grep:unchecked-throwing-call-python
          "| v:", round(float(torch.linalg.norm(unwenv.cup.linear_velocity, dim=1)[0]), 4),
          # pi-lens-ignore: ast-grep:unchecked-throwing-call-python
          "av:", round(float(torch.linalg.norm(unwenv.cup.angular_velocity, dim=1)[0]), 4))
    env.log_event("result", "Task completed", success=success)
    env.reset()
    return success


if __name__ == "__main__":
    args = parse_args()
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    # the mplib IK/RRT samples random initial configurations from an
    # UNSEEDED C++ RNG (verified: two runs of the same seed diverged at the
    # reach phase - the "IK results" candidates differed -> different arm
    # paths -> different outcomes = the batch "seed flips"). Seed it so the
    # same run seed reproduces exactly.
    from mplib.pymp import set_global_seed
    set_global_seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print(f"[INFO] seed={SEED}, render_mode='{args.render_mode}', "
          f"debug={args.debug}, info={args.info}, log_dir='{args.log_dir}'")

    run_id = f"takeitback_seed{SEED}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.log_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        "MyRoboCasa_TakeItBack-v1",
        num_envs=1,
        render_mode=args.render_mode,
        obs_mode="rgb",
        robot_uids="ds_fetch",
        control_mode="pd_joint_pos",
        # the PhysX CPU solver's parallel contact ordering is non-deterministic
        # by default (two runs of the same seed diverged by 0.0001 m at the
        # approach, amplified by the closed-loop base drives into different
        # reach inputs -> the "seed flips" between batches). eENHANCED_DETERMINISM
        # pins the pair processing order so the same seed reproduces exactly.
        sim_config=dict(scene_config=dict(cpu_workers=1, enable_enhanced_determinism=True)),
    )
    # Video-only recording: frames stream straight into ffmpeg, so RAM stays at
    # a single frame instead of RecordEpisode's whole-episode frame buffer
    # (~12 GB at the 2048x2048 render resolution).
    if args.no_video:
        print("[INFO] video recording disabled (--no-video)")
    else:
        env = StreamingVideoRecorder(env, output_dir=str(run_dir), video_fps=30)
    env = PlannerLogger(env, log_dir=str(run_dir), name=f"takeitback_seed{SEED}", log_freq=args.log_freq, run_dir=run_dir)

    env.action_space.seed(SEED)
    with capture_stdout(env.dir / "console.log"):
        planning(env, SEED, debug=args.debug, info=args.info)
    env.close()
    if not args.no_video:
        print(f"[INFO] Video recording saved in '{run_dir}/'")
