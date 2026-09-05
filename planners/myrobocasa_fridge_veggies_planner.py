import argparse
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import mplib
import sapien
import torch
from trimesh.primitives import Box
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import Fetch
from mani_skill.envs.tasks import MyRoboCasaFridgeVeggies
from mani_skill.examples.motionplanning.fetch.extand import (
    FetchMotionPlanningSapienSolver,
)
from utils.logging_utils import PlannerLogger, StreamingVideoRecorder, capture_stdout
from utils.planners_utils import (
    lower_torso_smooth,
    lower_torso_until_rest,
)

# lift offsets (world), tried in order: lifting straight up swings the elbow
# into the fixture stack and the vegetable into the cabinet door behind the
# counter, so prefer up + back toward the base (south)
LIFT_OFFSETS = [
    np.array([0.0, -0.20, 0.10]),
    np.array([0.0, -0.22, 0.08]),
    np.array([0.0, -0.18, 0.13]),
    np.array([0.0, -0.15, 0.15]),
    np.array([0.0, -0.15, 0.18]),
    np.array([0.0, -0.12, 0.20]),
]

# Render cameras for the recorded video, takeitback-style: named static
# cameras, one per view the benchmark needs. The CameraConfig poses are
# PLACEHOLDERS: the pot and the vegetable are only placed at episode start
# (inside env.reset), so _pose_render_cameras re-points every camera after
# the reset. The env's inherited single diagnostic "render_camera" (fridge
# picture scene) is replaced wholesale by _install_render_cameras.
RENDER_CAMERA_NAMES = ("main_camera", "fridge_camera", "pot_camera", "veg_camera")


def _render_camera_configs():
    """The four video cameras, one per view:
      * main_camera: the whole counter band (robot + pot + vegetable);
      * fridge_camera: the fridge door with the picture;
      * pot_camera: the pot, from the side;
      * veg_camera: the target vegetable, from the side above."""
    from mani_skill.sensors.camera import CameraConfig
    from mani_skill.utils import sapien_utils

    placeholder = sapien_utils.look_at(
        np.array([0.0, -1.5, 1.5]), np.array([0.0, 0.0, 0.5])
    )
    return [
        CameraConfig(name, placeholder, 1600, 1600, 70 * np.pi / 180, 0.01, 100)
        for name in RENDER_CAMERA_NAMES
    ]


def _install_render_cameras(env):
    """Replace the scene's single diagnostic render camera with the four
    video cameras. MUST run before env.reset(): the camera configs are
    consumed during the reset's reconfigure."""
    type(env.unwrapped)._default_human_render_camera_configs = property(
        lambda self: _render_camera_configs()
    )


def _pose_render_cameras(env, target_veg):
    """Point the four render cameras at their per-episode targets. Static
    framing (like takeitback): main frames the counter band, fridge frames
    the door picture, pot frames the pot, veg frames the vegetable's initial
    spot (the grasp/lift happen there; the release is on the pot camera)."""
    from mani_skill.utils import sapien_utils

    unwenv = env.unwrapped
    cam = unwenv._human_render_cameras
    cx, cy = unwenv.counter_pos[:2]
    # main: elevated south view of the whole counter band (takeitback's
    # main_camera framing)
    main = sapien_utils.look_at(
        np.array([cx + 0.7, cy - 1.8, 1.6]),
        np.array([cx + 0.7, cy - 0.025, 0.9]),
    )
    # fridge: the picture is on the door face. The robot spawns 0.9 m in
    # front of the door (robot_spawn_pos = picture_center + front * 0.9)
    # with its folded gripper at picture height right at the door, which
    # physically occludes the picture's lower half from every outside view;
    # frame the door from in front, offset sideways and up so the sight line
    # passes over the arm and the picture's upper part stays readable
    pc = np.asarray(unwenv.picture_center, dtype=float)
    front = np.asarray(unwenv.robot_spawn_pos, dtype=float) - pc
    front = front / np.linalg.norm(front)
    width = np.cross(np.array([0.0, 0.0, 1.0]), front)
    fridge = sapien_utils.look_at(
        pc + front * 1.1 + width * 0.85 + np.array([0.0, 0.0, 0.35]),
        pc + np.array([0.0, 0.0, -0.05]),
    )
    # pot: east-side profile (flat tilt like the takeitback east camera; the
    # release happens here)
    plate = unwenv.plate.pose.p[0].cpu().numpy().copy()
    pot = sapien_utils.look_at(
        plate + np.array([0.6, 0.0, 0.15]),
        plate + np.array([0.0, 0.0, 0.03]),
    )
    # vegetable: elevated south-east view (side from above)
    veg = target_veg.pose.p[0].cpu().numpy().copy()
    veggie = sapien_utils.look_at(
        veg + np.array([0.8, -0.75, 0.95]),
        veg + np.array([0.0, 0.0, -0.03]),
    )
    for name, pose in (
        ("main_camera", main),
        ("fridge_camera", fridge),
        ("pot_camera", pot),
        ("veg_camera", veggie),
    ):
        cam[name].camera.set_local_pose(pose.sp)


def _fit_rim_circle(xy):
    """RANSAC circle fit of the pot's rim wall vertices (xy). Returns
    (center_xy, R) or None. Deterministic (fixed RNG) so the measured bowl
    center is stable across runs with the same geometry."""
    rng = np.random.default_rng(0)
    best = None
    for _ in range(400):
        idx = rng.choice(len(xy), 3, replace=False)
        a, b, c = xy[idx]
        d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
        if abs(d) < 1e-9:
            continue
        ux = (
            (a[0] ** 2 + a[1] ** 2) * (b[1] - c[1])
            + (b[0] ** 2 + b[1] ** 2) * (c[1] - a[1])
            + (c[0] ** 2 + c[1] ** 2) * (a[1] - b[1])
        ) / d
        uy = (
            (a[0] ** 2 + a[1] ** 2) * (c[0] - b[0])
            + (b[0] ** 2 + b[1] ** 2) * (a[0] - c[0])
            + (c[0] ** 2 + c[1] ** 2) * (b[0] - a[0])
        ) / d
        R = np.hypot(ux - a[0], uy - a[1])
        if not (0.03 < R < 0.35):
            continue
        n = int(np.sum(np.abs(np.hypot(xy[:, 0] - ux, xy[:, 1] - uy) - R) < 0.015))
        if best is None or n > best[0]:
            best = (n, np.array([ux, uy]), R)
    return best


def _pot_bowl_center(env):
    """World xyz of the pot's BOWL center (the aim point for transport and
    descent), not the actor origin. The mjcf loader recenters the actor to
    its AABB center; for a pan with a long handle that center sits 70-91% of
    the rim radius toward the handle (measured 6.7-8.6 cm at rim R ~9.5 cm
    in this pot family). Aiming the descent at the origin drops the
    vegetable on the rim wall - observed: veg pressed against the wall
    during the descend (IK failure -> release from height -> wedged ->
    flung), and a drop test where the veg at the origin tipped over the rim
    and fell off the counter. Fit a circle to the pot's upper wall vertices
    and aim at its center. Falls back to the origin if the fit fails."""
    import transforms3d.quaternions as tq
    import trimesh

    unwenv = env.unwrapped
    plate = unwenv.plate
    origin = plate.pose.p[0].cpu().numpy()
    q = plate.pose.q[0].cpu().numpy()
    Rm = tq.quat2mat(q)
    mesh = plate.get_collision_meshes(to_world_frame=True)
    full = trimesh.util.concatenate(mesh) if isinstance(mesh, list) else mesh
    local = (np.asarray(full.vertices) - origin) @ Rm.T
    zmin, zmax = local[:, 2].min(), local[:, 2].max()
    wall = local[local[:, 2] > zmin + 0.5 * (zmax - zmin)][:, :2]
    fit = _fit_rim_circle(wall) if len(wall) >= 10 else None
    if fit is None:
        print("[WARN] pot rim circle fit failed; aiming at the actor origin")
        return origin.copy()
    _, c, R = fit
    bowl = origin + Rm @ np.array([c[0], c[1], 0.0])
    print(f"[INFO] pot bowl center {np.round(bowl, 3)} (rim R={R:.3f}, "
          f"offset {np.linalg.norm(c):.3f} m from the actor origin)")
    return bowl


def parse_args():
    parser = argparse.ArgumentParser(
        description="Motion planner for MyRoboCasa_FridgeVeggies-v1 scene"
    )
    parser.add_argument("--seed", type=int, default=3, help="Random seed (default: 3)")
    parser.add_argument(
        "--render-mode",
        type=str,
        default="rgb_array",
        choices=["rgb_array", "human", "sensors"],
        help="Render mode (default: rgb_array)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode in planner"
    )
    parser.add_argument(
        "--info", action="store_true", help="Print environment info in planner"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for log output (default: logs)",
    )
    parser.add_argument(
        "--log-freq",
        type=int,
        default=10,
        help="Write trajectory rows every N steps (default: 10)",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable mp4 recording for batch runs",
    )
    return parser.parse_args()


def _attach_object(planner, obj):
    """Register the grasped object as attached to the gripper in the mplib
    planning world so lift/transport plans stop treating it as an obstacle."""
    from mani_skill.examples.motionplanning.fetch.utils import attach_object

    gripper_link = next(
        l for l in planner.robot._objs[0].get_links() if l.name.endswith("gripper_link")
    )
    attach_object(
        planner.planner.planning_world,
        obj._objs[0],
        planner.robot._objs[0],
        gripper_link,
    )


def _detach_object(planner):
    """Release the object from the planning world (after the gripper is opened)."""
    planner.planner.detach_object()


def _top_down_grasp_pose(agent, obj_center, closing, raise_z, pre_clear=0.14):
    """Top-down grasp pose: the gripper approaches straight down (-z world)
    and the fingers close along the horizontal `closing` axis. The grasp
    center is the vegetable's xy at its center height, raised so the fingers
    (which hang ~6 cm below the TCP) wrap the vegetable's upper body. The
    pre-grasp is 18 cm above the grasp (the approach direction is -z)."""
    approaching = np.array([0.0, 0.0, -1.0])
    grasp_pose = agent.build_grasp_pose(approaching, closing, obj_center.copy())
    # sapien.Pose.p returns a COPY: in-place `grasp_pose.p[2] += raise_z`
    # silently does nothing (the raise_z sweep was dead without this), so the
    # pose must be rebuilt
    grasp_pose = sapien.Pose(
        p=grasp_pose.p + np.array([0.0, 0.0, raise_z]), q=grasp_pose.q
    )
    # HIGH pre-grasp: 14 cm above the grasp (was 10 cm, tried 18 cm). The
    # fingers hang ~6 cm below the TCP, so at 10 cm the fingers arrived at
    # the vegetable's TOP height and the reach path swept them through it
    # (the knock, seeds 30/42/47). At 14 cm the fingers end ~4-6 cm above the
    # vegetable's top; the final approach is the base-fixed vertical descent
    # (see _reach_and_grasp), so nothing sweeps at the vegetable's height.
    # 18 cm was too high: the arm's horizontal reach shrinks with altitude
    # (~0.60 m at z+0.18), and bases at 0.62-0.65 m (yaw drift) could not
    # reach it (seeds 28/34/46/50); 14 cm keeps the over-the-top clearance
    # while staying inside the reachable envelope from those bases.
    reach_pose = grasp_pose * sapien.Pose([0, 0, -pre_clear])
    return grasp_pose, reach_pose


def _drive_base(
    env, planner, target_pos, max_chunks=30, chunk=0.7, align_deg=3.0, yaw=True
):
    """Drive the base to target_pos (xy) WITHOUT teleportation.

    The ds_fetch base only translates along its current facing direction
    (move_base_forward's follow projects the screw velocity onto the base
    x-axis and never commands yaw), so the loop is: yaw-align in place via
    velocity control (unless yaw=False, which drives in the CURRENT facing -
    forward or backward - without turning), resync the planning world, then
    drive a short screw chunk. Re-aligning each chunk corrects the rotation
    drift. Returns the final base position."""
    unwenv = env.unwrapped
    agent = unwenv.agent
    arm_action = agent.controller.controllers["arm"].qpos[0].cpu().numpy()
    body_action = agent.controller.controllers["body"].qpos[0].cpu().numpy().copy()
    body_action[:2] = 0.0
    target = np.asarray(target_pos, dtype=float).copy()
    target[2] = 0.0

    def yaw_align(max_rot=500):
        for _ in range(max_rot):
            sp = agent.base_link.pose.sp
            p = sp.p.copy()
            p[2] = 0.0
            delta = target - p
            dist = np.linalg.norm(delta)
            if dist < 1e-4:
                return
            xa = sp.to_transformation_matrix()[:3, 0]
            xa[2] = 0.0
            nx = np.linalg.norm(xa)
            if nx < 1e-6:
                return
            xa /= nx
            dt = delta / dist
            he = float(np.arctan2(np.cross(xa, dt)[2], np.dot(xa, dt)))
            if abs(he) < np.deg2rad(align_deg):
                break
            base_action = np.array([0.0, 0.0, float(np.clip(1.5 * he, -0.6, 0.6))])
            env.step(np.hstack([arm_action, 1, body_action, base_action]))
        planner.planner.update_from_simulation()

    cur = agent.base_link.pose.p[0].cpu().numpy().copy()
    cur[2] = 0.0
    fails = 0
    for _ in range(max_chunks):
        dist = float(np.linalg.norm(target - cur))
        if dist < 0.10:
            break
        if yaw:
            yaw_align()
        way = cur + (target - cur) / dist * min(dist, chunk)
        res = planner.move_base_forward(way, n_init_qpos=100)
        if res == -1:
            # screw plans fail when an object is attached to the gripper
            # (mplib treats it as colliding with the robot); fall back to the
            # closed-loop velocity drive, which the takeitback transport uses
            from utils.planners_utils import _velocity_segment

            res = _velocity_segment(
                env,
                planner,
                way,
                arm_action,
                body_action,
                1,
                max_bursts=250,
            )
        planner.planner.update_from_simulation()
        cur = agent.base_link.pose.p[0].cpu().numpy().copy()
        cur[2] = 0.0
        if res == -1:
            fails += 1
            if fails > 3:
                print(f"[WARN] base drive stuck at {np.round(cur, 3)}")
                break
    return cur


def _yaw_base_to(env, planner, target_pos, max_rot=500, align_deg=3.0):
    """Rotate the base in place (velocity yaw) until its x-axis points at
    target_pos. Only the yaw joint is commanded; the arm is expected to be in
    a folded/safe pose. Rotation drifts the base slightly, which is fine for
    stance positioning."""
    unwenv = env.unwrapped
    agent = unwenv.agent
    arm_action = agent.controller.controllers["arm"].qpos[0].cpu().numpy()
    body_action = agent.controller.controllers["body"].qpos[0].cpu().numpy().copy()
    body_action[:2] = 0.0
    target = np.asarray(target_pos, dtype=float).copy()
    target[2] = 0.0
    for _ in range(max_rot):
        sp = agent.base_link.pose.sp
        p = sp.p.copy()
        p[2] = 0.0
        delta = target - p
        dist = np.linalg.norm(delta)
        if dist < 1e-4:
            break
        xa = sp.to_transformation_matrix()[:3, 0]
        xa[2] = 0.0
        nx = np.linalg.norm(xa)
        if nx < 1e-6:
            break
        xa /= nx
        he = float(np.arctan2(np.cross(xa, delta / dist)[2], np.dot(xa, delta / dist)))
        if abs(he) < np.deg2rad(align_deg):
            break
        base_action = np.array([0.0, 0.0, float(np.clip(1.5 * he, -0.6, 0.6))])
        env.step(np.hstack([arm_action, 1, body_action, base_action]))
    planner.planner.update_from_simulation()


def _safe_manipulation(env, planner, pose, **kwargs):
    """planner.static_manipulation that never crashes: the solver's refinement
    path indexes result['position'][-1] and throws IndexError on degenerate
    empty trajectories; treat that as a failed motion (-1)."""
    try:
        return planner.static_manipulation(pose, **kwargs)
    except (IndexError, KeyError, ValueError, RuntimeError) as e:
        print(f"[WARN] static_manipulation raised {type(e).__name__}: {e}")
        return -1


def _tcp_at(agent, target_pose, tol=0.04, z_only=False):
    """True if the TCP actually reached target_pose (within tol meters; when
    z_only, only the height must match).

    static_manipulation returns a success tuple even when its refinement gave
    up ("Robot is stuck") or the plan was only approximate, so the executed
    pose must be verified before a motion counts as done."""
    tcp = agent.tcp.pose.p[0].cpu().numpy()
    target = np.asarray(target_pose.p, dtype=float)
    if z_only:
        return float(abs(tcp[2] - target[2])) <= tol
    return float(np.linalg.norm(tcp - target)) <= tol


def _veg_world_name(veg):
    """The target vegetable's name in the mplib planning world (world objects
    are named f"{name}_{per_scene_id}", e.g. veg_carrot_4_110)."""
    from mplib.sapien_utils.conversion import convert_object_name

    import sapien.physx as physx

    component = veg._objs[0].find_component_by_type(physx.PhysxRigidBaseComponent)
    return convert_object_name(component.entity)


def _exclude_target_collision(planner, veg_name):
    """Allow the robot to collide with the target vegetable in the mplib
    planning world (allowed-collision matrix) so the grasp refinement can
    close the fingers AROUND it. Without this a flat vegetable on the counter
    is ungraspable: any collision-free pose keeps the fingers too high, and
    any pose that reaches the vegetable intersects the still-obstacle object.

    NOTE: the ACM starts EMPTY (entries are created on demand), so the robot
    link names must be enumerated explicitly - get_all_entry_names() returns
    [] and a loop over it would set nothing. The collision object names are
    the raw sapien link names (see mplib conversion: FCLObject(comp.name))."""
    acm = planner.planner.planning_world.get_allowed_collision_matrix()
    robot = planner.base_env.agent.robot._objs[0]  # raw sapien articulation
    for link in robot.get_links():
        acm.set_entry(link.name, veg_name, True)


def _restore_target_collision(planner, veg_name):
    """Undo _exclude_target_collision: make the vegetable a collision obstacle
    again (called only when ALL grasp attempts failed, so the next stance's
    drives/reaches still avoid it)."""
    acm = planner.planner.planning_world.get_allowed_collision_matrix()
    robot = planner.base_env.agent.robot._objs[0]  # raw sapien articulation
    for link in robot.get_links():
        acm.set_entry(link.name, veg_name, False)


def _exclude_veggies_collision(env, planner):
    """Allow the robot to collide with ALL the vegetables on the counter (the
    target AND the distractors) in the planning world. The reach's
    collision-aware IK rejects every pre-grasp whose arm passes over or near
    the other vegetables (the counter holds 5), producing the repeated
    'IK Failed! Cannot find valid solution' failures; the distractors are
    restored right after the grasp."""
    acm = planner.planner.planning_world.get_allowed_collision_matrix()
    robot = planner.base_env.agent.robot._objs[0]  # raw sapien articulation
    for veg in env.unwrapped.veggies:
        name = _veg_world_name(veg)
        for link in robot.get_links():
            acm.set_entry(link.name, name, True)


def _restore_veggies_collision(env, planner, keep_target=None):
    """Undo _exclude_veggies_collision; optionally keep the target excluded
    (the caller attaches it right after the grasp)."""
    acm = planner.planner.planning_world.get_allowed_collision_matrix()
    robot = planner.base_env.agent.robot._objs[0]
    for veg in env.unwrapped.veggies:
        name = _veg_world_name(veg)
        if name == keep_target:
            continue
        for link in robot.get_links():
            acm.set_entry(link.name, name, False)


def _descend_tcp_step(env, planner, dz, n_init_qpos=200):
    """One small VERTICAL TCP descent planned with the base joints FIXED.

    static_manipulation leaves the base free - its IK solutions swing the
    whole robot sideways mid-descent, flinging the held vegetable off the
    plate. Fixing root_x/y/z here keeps the descent strictly vertical."""
    agent = env.unwrapped.agent
    tcp_now = agent.tcp.pose.sp
    tgt = sapien.Pose(p=tcp_now.p - np.array([0.0, 0.0, dz]), q=tcp_now.q)

    # The pot (the "plate") is a CONTAINER: the descend must lower the
    # vegetable INTO it, so the fingers and the held vegetable are allowed to
    # touch it during the descent. Without this the collision-aware IK stops
    # the vegetable ~10 cm above the pot's bottom and the release's drop
    # bounces it out. Restore the collision after the step.
    import sapien.physx as physx
    from mplib.sapien_utils.conversion import convert_object_name
    acm = planner.planner.planning_world.get_allowed_collision_matrix()
    robot = planner.base_env.agent.robot._objs[0]  # raw sapien articulation
    pot_comp = env.unwrapped.plate._objs[0].find_component_by_type(
        physx.PhysxRigidBaseComponent
    )
    pot_name = convert_object_name(pot_comp.entity)
    target_comp = env.unwrapped.veggies[
        env.unwrapped._picture_target
    ]._objs[0].find_component_by_type(physx.PhysxRigidBaseComponent)
    target_name = convert_object_name(target_comp.entity)
    pot_links = [link.name for link in robot.get_links()]
    for ln in pot_links:
        acm.set_entry(ln, pot_name, True)
    # The held target is attached in the planning world during transport.
    # Allow it to enter the pot as well; otherwise the robot-link allowance
    # below is insufficient and release IK stops above the container floor.
    acm.set_entry(target_name, pot_name, True)

    # move-group order: root_x, root_y, root_z, torso, shoulder_pan,
    # shoulder_lift, upperarm_roll, elbow_flex, forearm_roll, wrist_flex,
    # wrist_roll - fix the first three (base translation); mask=True means
    # the joint is NOT used by the IK (fixed)
    mask = [True, True, True] + [False] * 12
    res = planner.planner.plan_pose(
        tgt, planner.robot.get_qpos().cpu().numpy()[0],
        time_step=env.unwrapped.control_timestep, wrt_world=True,
        planning_time=4, rrt_range=0.1, simplify=True, mask=mask,
        fixed_joint_indices=[0, 1, 2], n_init_qpos=n_init_qpos,
    )
    for ln in pot_links:
        acm.set_entry(ln, pot_name, False)
    acm.set_entry(target_name, pot_name, False)
    if not str(res.get("status", "")).startswith("Success"):
        print(f"[WARN] descend step failed: {res.get('status')}")
        return -1
    pos = res.get("position", np.zeros((1, 11)))
    print(f"[DBG] descend plan: dz_req={dz:.3f} tcp_z0={tcp_now.p[2]:.3f} "
          f"plan_first={np.round(pos[0][3:5], 3).tolist()} "
          f"plan_last={np.round(pos[-1][3:5], 3).tolist()} n={len(pos)}")
    try:
        planner.follow_path(res)
    except AssertionError as e:
        import numpy as _np
        print("[DESCEND-DEBUG]",
              "arm", agent.controller.controllers["arm"].qpos[0].shape,
              "gs", repr(planner.gripper_state), np.shape(planner.gripper_state),
              "body", agent.controller.controllers["body"].qpos[0].shape,
              "| err:", e)
        return -1
    return 0


def _descend_tcp_abs(env, planner, tgt, n_init_qpos=100):
    """Reach grasp height with a fixed-base plan and a vertical fallback.

    If the fixed-base RRT cannot solve the final drop, lower the torso slowly
    while holding arm/base targets. This keeps the already aligned TCP on a
    vertical corridor instead of retrying a blocked long plan.
    """
    agent = env.unwrapped.agent
    mask = [True, True, True] + [False] * 12
    res = planner.planner.plan_pose(
        tgt, planner.robot.get_qpos().cpu().numpy()[0],
        time_step=env.unwrapped.control_timestep, wrt_world=True,
        planning_time=4, rrt_range=0.1, simplify=True, mask=mask,
        fixed_joint_indices=[0, 1, 2], n_init_qpos=n_init_qpos,
    )
    if str(res.get("status", "")).startswith("Success"):
        try:
            planner.follow_path(res)
            return 0
        except AssertionError as exc:
            print(f"[INFO] grasp descend path failed: {exc}")

    current_z = float(agent.tcp.pose.sp.p[2])
    drop = current_z - float(tgt.p[2])
    if drop > 0.01:
        print(f"[INFO] grasp descend fallback: torso drop {drop:.3f} m")
        arm_action = agent.controller.controllers["arm"].qpos[0].cpu().numpy()
        lower_torso_smooth(
            env,
            planner,
            target_drop=min(drop, 0.16),
            total_steps=50,
            arm_action=arm_action,
            gripper_action=planner.gripper_state,
        )
        if _tcp_at(agent, tgt, tol=0.06):
            return 0
    print(f"[WARN] grasp descend failed: {res.get('status')}")
    return -1


def _reach_and_grasp(env, planner, agent, pre_clear=0.14):
    """Reach the pre-grasp pose then execute the grasp. ONLY top-down grasps
    are attempted: the gripper approaches straight down (-z) and the fingers
    close along a horizontal world axis (x or y). Straight/side grasps are
    never tried (the low flat vegetables cannot clear the counter edge for a
    side grasp). The grasp is a FINGERTIP grasp: the target poses the object
    at the finger tips (the sweep raise_z ~ finger length), and the target
    vegetable is excluded from the planning-world collisions during the final
    close so the fingers can close around a flat object. Each motion is
    VERIFIED with _tcp_at (static_manipulation can report success while its
    refinement got stuck), and the vegetable is re-measured after every reach
    (an approximate reach knocks it). The pre-grasp is HIGH (18 cm above the
    grasp) and the reach is planned WITH the vegetables as collision
    obstacles, so the arm routes over them; the final approach to the grasp
    is a base-fixed VERTICAL descent (no lateral sweep at the vegetable's
    height - the knock). Returns the executed grasp pose, or None if all
    failed."""
    # close ACROSS the vegetable's SHORT horizontal dimension: the fingers
    # must wrap the narrow side. World-axis closings (x/y) align with the
    # LONG axis of a diagonal vegetable, so the fingers only graze it and the
    # lift slips (the "Grasp invalid" failures). Compute the OBB's shortest
    # horizontal axis; world x/y remain as fallbacks. The sign is symmetric
    # for a parallel gripper (the grasp frame rotates 180 deg about the
    # approach axis).
    closings = []
    veg = env.unwrapped.veggies[env.unwrapped._picture_target]
    mesh = veg.get_first_collision_mesh(to_world_frame=True)
    if mesh is not None:
        obb = mesh.bounding_box_oriented
        T = np.asarray(obb.transform)[:3, :3]  # OBB axes (columns)
        world_ext = np.asarray(obb.extents)
        local_ext = np.asarray(obb.primitive.extents)
        # Horizontal axes only (small world-z component). Correct the extent
        # ranking only for near-equivalent widths; dramatic orientation swaps
        # regressed otherwise-valid elongated grasps.
        horiz = [i for i in range(3) if abs(T[2, i]) < 0.5]
        ext = world_ext
        if len(horiz) > 1:
            old_i = min(horiz, key=lambda i: world_ext[i])
            new_i = min(horiz, key=lambda i: local_ext[i])
            if local_ext[old_i] <= 1.2 * local_ext[new_i]:
                ext = local_ext
        horiz.sort(key=lambda i: ext[i])
        for i in horiz:
            ax = np.asarray(T[:, i], dtype=float).copy()
            ax[2] = 0.0
            n = float(np.linalg.norm(ax))
            if n > 1e-4:
                closings.append(ax / n)
    closings += [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ]
    veg_name = _veg_world_name(veg)
    for tc in closings:
        # re-read the vegetable pose: the previous reach may have knocked it
        obj_center = (
            env.unwrapped.veggies[env.unwrapped._picture_target]
            .pose.p[0]
            .cpu()
            .numpy()
            .copy()
        )
        # the fingers are HORIZONTAL plates (the grip point is at the TCP's
        # height, not below it): raise_z is the clamp height on the vegetable.
        # Values 0.03-0.07 clamp ABOVE a flat vegetable's center (the fingers
        # graze the top edge and the lift slips); start at 0.0 = clamp at the
        # vegetable's middle. The collision-aware IK rejects the values that
        # would push the plates into the counter.
        for raise_z in (0.0, 0.02, 0.04, 0.06):
            grasp_pose, reach_pose = _top_down_grasp_pose(
                agent, obj_center, tc, raise_z, pre_clear
            )
            res = env.log_motion(
                "Reach veggie",
                _safe_manipulation,
                env,
                planner,
                reach_pose,
                disable_lift_joint=False,
                n_init_qpos=100,
            )
            if res == -1 or not _tcp_at(agent, reach_pose, tol=0.10):
                if res == -1:
                    base_p = agent.base_link.pose.p[0].cpu().numpy()
                    print(
                        f"[DBG] reach fail: base={np.round(base_p[:2],2)} "
                        f"veg={np.round(obj_center[:2],2)} "
                        f"target={np.round(reach_pose.p,2)} "
                        f"closing={np.round(tc,2)} raise_z={raise_z}"
                    )
                continue
            # corrective second reach: the first RRT refinement often gives up
            # a few cm short; re-planning the same target from the approximate
            # pose converges to the exact pose
            res = env.log_motion(
                "Reach veggie (corrective)",
                _safe_manipulation,
                env,
                planner,
                reach_pose,
                disable_lift_joint=False,
                n_init_qpos=100,
            )
            if res == -1 or not _tcp_at(agent, reach_pose):
                continue
            # re-measure the vegetable after the reach and re-aim the grasp
            obj_center = (
                env.unwrapped.veggies[env.unwrapped._picture_target]
                .pose.p[0]
                .cpu()
                .numpy()
                .copy()
            )
            grasp_pose, _ = _top_down_grasp_pose(agent, obj_center, tc, raise_z, pre_clear)
            # fingertip grasp: allow the fingers to touch the target vegetable
            # in the planning world, otherwise the collision-aware IK refuses
            # every pose that reaches a flat vegetable on the counter. The
            # exclusion applies ONLY now (before the close): the reaches above
            # are planned with the vegetables as collision obstacles, so the
            # arm routes OVER them instead of sweeping through (the knock).
            _exclude_veggies_collision(env, planner)
            # the grasp is a BASE-FIXED VERTICAL descent to the grasp pose
            # (see _descend_tcp_abs): no lateral swing at the vegetable's
            # height, so the fingers descend straight onto it and the close
            # wraps it
            res = _descend_tcp_abs(env, planner, grasp_pose)
            if res != -1 and _tcp_at(agent, grasp_pose):
                # hold check BEFORE returning: the plan/execution can succeed
                # while the fingers only graze a flat vegetable (the caller's
                # lift-verify would then reject it and waste the whole stance).
                # Close the fingers and confirm contact; if the vegetable is
                # not held, open and try the next raise_z/closing instead.
                planner.close_gripper()
                planner.planner.update_from_simulation()
                if agent.is_grasping(env.unwrapped.veggies[env.unwrapped._picture_target]):
                    # keep the TARGET excluded (the caller attaches it right
                    # after); restore the distractors
                    _restore_veggies_collision(env, planner, keep_target=veg_name)
                    return grasp_pose
                planner.open_gripper()
                planner.planner.update_from_simulation()
            # every grasp attempt (failed, or executed but not holding) leaves
            # the vegetable excluded from the planning world for the fingers
            # to close; restore it immediately so the NEXT reach is planned
            # with the vegetable as a collision obstacle again - otherwise the
            # arm passes through it and knocks it away (observed: lemon
            # knocked over the counter edge by the reach after a failed grasp)
            _restore_veggies_collision(env, planner)
    # all attempts failed: the next stance must still AVOID the vegetables, so
    # restore them as collision obstacles
    _restore_veggies_collision(env, planner)
    return None


def _lift_veggie(env, planner, agent, grasp_pose):
    """Lift the grasped vegetable, trying several offsets (up + back first),
    each three times: the RRT planner is stochastic and the workspace is
    tight. The TCP must actually RISE to the lift height (static_manipulation
    can report success while its refinement got stuck at the old height).
    The lifted height must be HIGH: the transport yaw-swing only clears the
    counter front wall if the vegetable is well above it (bottom > ~1.0)."""
    res = -1
    # first pass: require a real high lift (z >= grasp_z + 0.14)
    for off in sorted(LIFT_OFFSETS, key=lambda o: -o[2]):
        for _ in range(3):
            lift_pose = sapien.Pose(grasp_pose.p + off, grasp_pose.q)
            res = env.log_motion(
                "Lift veggie",
                _safe_manipulation,
                env,
                planner,
                lift_pose,
                disable_lift_joint=False,
                n_init_qpos=100,
            )
            if (
                res != -1
                and (agent.tcp.pose.p[0].cpu().numpy()[2] - grasp_pose.p[2]) >= 0.12
            ):
                return res
    # fallback: any verified rise
    for off in LIFT_OFFSETS:
        for _ in range(3):
            lift_pose = sapien.Pose(grasp_pose.p + off, grasp_pose.q)
            res = env.log_motion(
                "Lift veggie",
                _safe_manipulation,
                env,
                planner,
                lift_pose,
                disable_lift_joint=False,
                n_init_qpos=100,
            )
            if res != -1 and _tcp_at(agent, lift_pose, z_only=True, tol=0.06):
                return res
    return res


def _transport_veg(env, planner, agent, target_veg, aim_xy, margin=0.05,
                max_iter=30):
    """Drive the base so the held vegetable reaches aim_xy (world xy).

    Port of the takeitback _transport_cup pattern (which converges on the
    fork's base): each iteration measures the vegetable's CURRENT local offset
    in the base frame, scans the full circle of headings for a feasible drive
    direction (the base target must stay south of the counter front y=-1.0),
    rotates the base there SLOWLY (rot_cap=0.06 - a fast turn swings the held
    vegetable on its ~0.6 m orbit and the lateral force drops it), then drives
    a chunk with the fallback chain fixed-arm screw -> move_base_forward ->
    closed-loop velocity segment. Convergence is measured on the actual
    vegetable position, not on a calibrated map, so the final centimetres do
    not stall (the P-drive stalled at 0.14-0.20 m; the map-based velocity
    drive diverged after yaw drift). Returns 0 on convergence, -1 otherwise."""
    from utils.planners_utils import (
        _screw_base_translate,
        _rotate_base_to,
        _velocity_segment,
    )
    unwenv = env.unwrapped

    def heading():
        xa = agent.base_link.pose.sp.to_transformation_matrix()[:3, 0]
        return np.arctan2(xa[1], xa[0])

    for _ in range(max_iter):
        veg = target_veg.pose.p[0].cpu().numpy()
        # veg-attach monitor: a weak grip can drop the vegetable during the
        # first rotation; the held veg rides ~0.1-0.15 m from the TCP, a
        # larger gap means it is lost
        tcp = agent.tcp.pose.p[0].cpu().numpy()
        if float(np.linalg.norm(tcp - veg)) > 0.25:
            print(f"[INFO] transport: vegetable lost (TCP-veg "
                  f"{np.linalg.norm(tcp - veg):.2f} m); aborting")
            return -1
        rem = np.asarray(aim_xy, dtype=float) - veg[:2]
        if float(np.linalg.norm(rem)) < margin:
            return 0
        base = agent.base_link.pose.p[0].cpu().numpy()
        h = heading()
        # the vegetable's local offset in the base frame (the arm is rigid
        # during the transport): re-measured every iteration
        dx, dy = veg[0] - base[0], veg[1] - base[1]
        cos_h, sin_h = np.cos(h), np.sin(h)
        veg_local = np.array([cos_h * dx + sin_h * dy, -sin_h * dx + cos_h * dy])

        def base_target(ang):
            c, s = np.cos(ang), np.sin(ang)
            return np.array([aim_xy[0] - (c * veg_local[0] - s * veg_local[1]),
                             aim_xy[1] - (s * veg_local[0] + c * veg_local[1])])

        d = np.arctan2(rem[1], rem[0])
        # scan the full circle; pick the heading whose drive direction
        # (forward or backward) is closest to the residual
        h_drive, best_err, h_back = None, np.inf, False
        for h in np.arange(-np.pi, np.pi, np.deg2rad(6.0)):
            # the base-end must stay south of the drive's actual guard
            # (y > -0.95 aborts in _velocity_segment), NOT the conservative
            # -1.0 line: the bowl-center aim sits up to ~1 cm north of what
            # -1.0 allows (observed: veg_local 0.44 -> min base-end y -0.993,
            # 7 mm over the line, "no feasible drive heading" -> run lost)
            if base_target(h)[1] <= -0.95:
                for sign, drive in ((False, h), (True, h + np.pi)):
                    err = abs(((drive - d + np.pi) % (2 * np.pi)) - np.pi)
                    if err < best_err:
                        best_err, h_drive, h_back = err, h, sign
        if h_drive is None:
            print(f"[INFO] transport: no feasible drive heading at base "
                  f"{np.round(base, 3)}, veg {np.round(veg[:2], 3)} "
                  f"h={np.degrees(h):.1f} veg_local={np.round(veg_local, 3)} "
                  f"rem={np.round(rem, 3)}")
            return -1
        # slow rotation to face the drive direction; the veg swings on its
        # orbit, so the rem below is re-measured AFTER the swing
        err = ((h_drive - h + np.pi) % (2 * np.pi)) - np.pi
        if abs(err) > np.deg2rad(2):
            _rotate_base_to(env, planner,
                            np.array([np.cos(h_drive), np.sin(h_drive), 0.0]),
                            rot_cap=0.06)
            planner.planner.update_from_simulation()
        veg = target_veg.pose.p[0].cpu().numpy()
        rem = np.asarray(aim_xy, dtype=float) - veg[:2]
        if float(np.linalg.norm(rem)) < margin:
            return 0
        base = agent.base_link.pose.p[0].cpu().numpy()
        step = min(1.0, 0.4 / max(float(np.linalg.norm(rem)), 1e-6))
        waypoint = base + np.array([rem[0] * step, rem[1] * step, 0.0])
        waypoint[1] = min(waypoint[1], -1.0)
        res = _screw_base_translate(planner, waypoint)
        if res == -1:
            # the fixed-arm screw is picky about start states; the yaw-free
            # move_base_forward replans the arm (safe: the veg is north of the
            # base, the arm stays above the fixtures)
            res = planner.move_base_forward(waypoint, n_init_qpos=100)
        if res == -1:
            arm_action = agent.controller.controllers["arm"].qpos[0].cpu().numpy()
            body_action = agent.controller.controllers["body"].qpos[0].cpu().numpy().copy()
            # pi-lens-ignore: unchecked-throwing-call-python
            body_action[0] = body_action[1] = 0.0
            # slow: the default 0.35 m/s bursts jerk a smooth vegetable out
            # of the fingertip grip (observed drops); 0.18 is gentle enough
            res = _velocity_segment(env, planner, waypoint, arm_action,
                                    body_action, planner.gripper_state,
                                    speed=0.18, x_min=0.05)
        planner.planner.update_from_simulation()
    print(f"[INFO] transport: did not converge after {max_iter} iterations")
    return -1


def planning(env, seed, debug=False, vis=None, info=False):
    vis = vis or env.unwrapped.render_mode == "human"

    unwenv: MyRoboCasaFridgeVeggies = env.unwrapped
    if unwenv.render_mode is not None:
        _install_render_cameras(env)  # configs are read during reconfigure
    obs, _ = env.reset(seed=seed, options={"reconfigure": True})
    # ManiSkill resolves agent/controller types dynamically at runtime.
    agent: Any = unwenv.agent  # must be captured AFTER the reconfigure reset

    # --- identify the target from the fridge picture -----------------------
    # The scene shows the pictured vegetable's own photo texture; the planner
    # reads the scene state directly (the picture actor shown this episode).
    target_idx = unwenv._picture_target
    target_veg = unwenv.veggies[target_idx]
    print(f"Target vegetable (from fridge picture): {target_veg.name}")
    if unwenv.render_mode is not None:
        _pose_render_cameras(env, target_veg)  # per-episode camera poses

    # --- Approach the target vegetable --------------------------------------
    # No teleportation: the base DRIVES. The robot may spawn right next to the
    # fridge with the extended arm poking INTO it (which makes every screw plan
    # fail), so fold the arm to the rest config first, drive, then unfold.
    mesh = target_veg.get_first_collision_mesh(to_world_frame=True)
    severe_grasp_geometry = False
    if mesh is not None:
        obb: Box = mesh.bounding_box_oriented
        veg_center = obb.center_mass.copy()
        axes = np.asarray(obb.transform)[:3, :3]
        world_ext = np.asarray(obb.extents)
        local_ext = np.asarray(obb.primitive.extents)
        horiz = [i for i in range(3) if abs(axes[2, i]) < 0.5]
        if len(horiz) > 1:
            old_i = min(horiz, key=lambda i: world_ext[i])
            new_i = min(horiz, key=lambda i: local_ext[i])
            severe_grasp_geometry = (
                local_ext[old_i] > 0.10 and local_ext[new_i] < 0.035
            )

    home_qpos = agent.robot.qpos[0].cpu().numpy().copy()
    arm_idx = agent.controller.controllers["arm"].active_joint_indices.cpu().numpy()
    rest_qpos = np.asarray(agent.keyframes["rest"].qpos)
    # fold ONLY the arm joints: the rest keyframe's base joints are zero, so
    # set_qpos(rest) would teleport the base back to the spawn (a hidden base
    # teleportation that also breaks every subsequent drive)
    q = agent.robot.qpos[0].cpu().numpy().copy()
    q[arm_idx] = rest_qpos[arm_idx]
    agent.robot.set_qpos(q)

    planner: Any = FetchMotionPlanningSapienSolver(
        env,
        base_pose=agent.robot.pose,
        vis=vis,
        print_env_info=info,
        debug=debug,
    )
    planner.MAX_REFINE_STEPS = 100
    for i, veg in enumerate(unwenv.veggies):
        env.track_object(veg, f"veg_{i}")
    env.track_object(unwenv.plate, "plate")
    env.track_object(agent.tcp, "robot_tcp")
    env.track_object(agent.base_link, "robot_base")
    env.log_event("start", "Planning started")
    env.log_event("info", "Target identified", target=target_veg.name)

    grasp_pose = None
    # straight-south stances can put the base in the dishwasher/stack-cabinet
    # corridor; diagonal stances approach the same strip from ±40 deg. The
    # reach is a top-down grasp: the arm's effective reach at the grasp
    # height is ~0.75 m, and the base cannot cross y=-1.0 (the counter front
    # face), so stances closer than 0.6 m give the IK a much bigger feasible
    # region (the 0.6-0.7 m band was marginal - the same geometry succeeded
    # or failed depending on the mplib RNG). 0.8 m stances never succeeded
    # (beyond the reach) and were dropped.
    straight = np.array([0.0, -1.0])
    diagonals = [
        np.array([0.5, -0.8660254]),
        np.array([-0.5, -0.8660254]),
    ]
    stance_dirs = diagonals + [straight] if severe_grasp_geometry else (
        [straight] + diagonals
    )
    for sdir in stance_dirs:
        for dist in (0.5, 0.6):
            # fold the arm again before any drive: after a failed grasp the arm
            # can be in a pose that blocks screw planning (and the spawn is
            # inside the fridge with the arm extended). Arm joints only - the
            # base must NOT be teleported.
            q = agent.robot.qpos[0].cpu().numpy().copy()
            q[arm_idx] = rest_qpos[arm_idx]
            agent.robot.set_qpos(q)
            planner.planner.update_from_simulation()
            # re-observe the target right before each stance: it may have been
            # knocked or shifted during earlier attempts / base drive-in
            veg_center = target_veg.pose.p[0].cpu().numpy().copy()
            approach_pos = veg_center.copy()
            approach_pos[:2] += sdir * dist
            counter_front_y = float(
                unwenv.counter_pos[1] - unwenv.counter_size[1] / 2
            )
            approach_pos[1] = min(approach_pos[1], counter_front_y - 0.02)
            approach_pos[2] = 0.0
            staging_pos = approach_pos.copy()
            staging_pos[1] = min(staging_pos[1], counter_front_y - 0.35)
            if np.linalg.norm(
                agent.base_link.pose.p[0].cpu().numpy()[:2] - staging_pos[:2]
            ) >= 0.10:
                print(f"Staging base in open corridor at {np.round(staging_pos, 3)}")
                _drive_base(env, planner, staging_pos)
                planner.planner.update_from_simulation()
            print(f"Approaching target: driving base to {np.round(approach_pos, 3)}")
            env.log_event("phase", "Approach target")
            _drive_base(env, planner, approach_pos)
            planner.planner.update_from_simulation()

            actual = agent.base_link.pose.p[0].cpu().numpy()
            err = float(np.linalg.norm(actual[:2] - approach_pos[:2]))
            if err >= 0.12:
                print(f"[WARN] drive to stance failed (err {err:.2f} m), trying next")
                env.log_event("warn", "Drive to stance failed", error_m=round(err, 3))
                continue

            # face the vegetable: the drive leaves the base yawed along the
            # travel direction, which can put the garlic outside the arm's
            # forward workspace; rotate in place (folded arm) toward it
            _yaw_base_to(env, planner, veg_center)
            planner.planner.update_from_simulation()
            # The in-place yaw rotation DRIFTS the base (PhysX slide,
            # pi-lens-ignore: unchecked-throwing-call-python
            # measured up to 0.26 m - seed 5), pushing it beyond the arm's
            # reach. The reachable envelope is SMALLER than the flat ~0.75 m:
            # the pre-grasp is HIGH (18 cm above the grasp) and the grasp is
            # a base-fixed vertical descent, so the base must stay close
            # (observed: bases at 0.60-0.65 m made the high reach / fixed
            # descend IK-infeasible, seeds 32/47). Re-measure after the yaw
            # and re-drive toward the vegetable when out of reach. yaw=False:
            # the base already faces it, and a second yaw near the vegetable
            # would sweep the 0.5 m folded arm through it (observed: cucumber
            # knocked under the cabinet overhang).
            base_p = agent.base_link.pose.p[0].cpu().numpy()
            b2v = float(np.linalg.norm(base_p[:2] - veg_center[:2]))
            if b2v > 0.68:
                print(f"[INFO] yaw drift: base {b2v:.2f} m from vegetable "
                      f"(reach ~0.60 m at the high pre-grasp), re-driving closer")
                _drive_base(
                    env, planner,
                    base_p + (veg_center - base_p) / b2v * (b2v - 0.55),
                    yaw=False,
                )
                planner.planner.update_from_simulation()
            # NOTE: the arm stays folded; the reach plans FROM the folded pose
            # pi-lens-ignore: unchecked-throwing-call-python
            # (a set_qpos unfold would push the TCP into the counter volume)

            print("Reaching + grasping target vegetable")
            env.log_event("phase", "Reaching target vegetable")
            # Keep proven high clearance first; a lower pre-grasp can recover
            # poses whose high target is outside the arm's collision-free
            # envelope without disturbing seeds that already work.
            grasp_pose = None
            for pre_clear in (0.14, 0.10):
                grasp_pose = _reach_and_grasp(
                    env, planner, agent, pre_clear=pre_clear
                )
                planner.planner.update_from_simulation()
                if grasp_pose is not None:
                    break
            if grasp_pose is None:
                continue

            # grasp-validity gate: a stuck reach still returns non-(-1), so
            # close the gripper, LIFT, and verify the vegetable actually rose
            # with the gripper (the weak tcp-distance test passes when the TCP
            # merely hovers near the vegetable without holding it). An empty
            # grasp is rejected here and the next stance is tried.
            print("Grasp vegetable")
            env.log_event("phase", "Grasp target vegetable")
            planner.close_gripper()
            planner.planner.update_from_simulation()
            veg_z0 = float(target_veg.pose.p[0].cpu().numpy()[2])

            _attach_object(planner, target_veg)
            planner.planner.update_from_simulation()

            print("Lift vegetable")
            env.log_event("phase", "Lift vegetable")
            _lift_veggie(env, planner, agent, grasp_pose)
            planner.planner.update_from_simulation()

            tcp_pos = agent.tcp.pose.p[0].cpu().numpy()
            veg_now = target_veg.pose.p[0].cpu().numpy()
            # Require both physical contact and a measured lift. Contact
            # alone can be a finger/counter collision that falsely licenses
            # transport with an empty gripper.
            grasped = bool(agent.is_grasping(target_veg)) and (
                np.linalg.norm(tcp_pos - veg_now) < 0.08
                and (veg_now[2] - veg_z0) > 0.03
            )
            if not grasped:
                print(
                    f"Grasp invalid: vegetable did not rise with gripper "
                    f"(z {veg_z0:.3f} -> {veg_now[2]:.3f}), trying next stance."
                )
                env.log_event("error", "Grasp invalid (vegetable not lifted)")
                planner.open_gripper()
                _detach_object(planner)
                grasp_pose = None
                continue
            # validated grasp: stop trying stances
            break
        if grasp_pose is not None:
            break

    if grasp_pose is None:
        print("Grasping failed entirely.")
        success = bool(unwenv.evaluate()["success"].item())  # pyright: ignore[reportAttributeAccessIssue]
        env.log_event("result", "Task failed at grasp", success=success)
        env.reset()
        return success

    # --- transport to the plate ---------------------------------------------
    # The takeitback-proven transport: rotate toward the transfer direction,
    # then drive velocity bursts with the arm frozen (the fork's burst driver
    # measures the actual motion and rotates only to correct it). The close
    # verified high lift keeps the vegetable clear of the counter front wall.
    # aim the transport/centering/descent at the pot's BOWL center, not the
    # actor origin: the origin is the AABB center (incl. the long handle),
    # which sits 70-91% of the rim radius toward the handle (see
    # _pot_bowl_center)
    plate_center = _pot_bowl_center(env)
    print("\n--- Drive base toward plate ---")
    env.log_event("phase", "Drive base toward plate")
    for drive_attempt in range(3):
        veg_now = target_veg.pose.p[0].cpu().numpy()
        dxy = np.linalg.norm(veg_now[:2] - plate_center[:2])
        # drive until the vegetable is actually over the plate center (0.05):
        # with yaw_sweep=False the transport translates DIRECTLY (no rotation
        # swings the held veg away), so the final few centimetres are safe to
        # close by driving. The old 0.60 break belonged to the orbit-swung
        # transport, where driving closer only arced the veg away.
        if dxy < 0.05:
            break
        if veg_now[2] < 1.0:
            print("Transport: vegetable dropped, aborting transport.")
            env.log_event("error", "Vegetable dropped during transport")
            break
        res = env.log_motion(
            "Drive base to plate",
            _transport_veg,
            env,
            planner,
            agent,
            target_veg,
            plate_center[:2],
            # tight: the raise-then-open release drops the veg ~5 cm and the
            # bounce rolls it up to ~5 cm further, so release near the rim
            # (0.05) ends up oscillating on it. Deliver to 0.03.
            margin=0.03,
        )
        planner.planner.update_from_simulation()
        veg_now = target_veg.pose.p[0].cpu().numpy()
        dxy = np.linalg.norm(veg_now[:2] - plate_center[:2])
        print(f"[WARN] veggie {dxy:.2f} m from plate after transport")
        env.log_event("warn", "Veggie not over plate", dxy=round(float(dxy), 3))

    # --- center the vegetable over the plate (base rotation ONLY) ---------
    # The arm-alignment loop (small TCP steps toward the plate) is REMOVED:
    # it pushes the arm to the workspace edge, after which the base-fixed
    # vertical descent cannot find IK solutions and the vegetable releases
    # from >15 cm (bounces off the plate). Centering is done by ROTATING the
    # base: the held veg rides a ~0.5 m orbit around the base, so a yaw sweep
    # swings it onto the plate without touching the arm. Center to <=0.04 m
    # (a long vegetable released at 0.06-0.10 off-center has its end past the
    # 0.116 m rim and tips off during the settle).
    veg_now = target_veg.pose.p[0].cpu().numpy()
    # pi-lens-ignore: unchecked-throwing-call-python
    dxy = np.linalg.norm(veg_now[:2] - plate_center[:2])
    if 0.05 <= dxy < 0.35 and veg_now[2] >= 1.0:
        # If the pot lies on the held payload's current orbit around the base,
        # yaw alone can finish a stalled transport without moving the arm.
        base_xy = agent.base_link.pose.p[0].cpu().numpy()[:2]
        veg_from_base = veg_now[:2] - base_xy
        pot_from_base = plate_center[:2] - base_xy
        radius_error = abs(
            np.linalg.norm(veg_from_base) - np.linalg.norm(pot_from_base)
        )
        if radius_error <= 0.04:
            heading = np.arctan2(
                agent.base_link.pose.sp.to_transformation_matrix()[1, 0],
                agent.base_link.pose.sp.to_transformation_matrix()[0, 0],
            )
            local_angle = np.arctan2(veg_from_base[1], veg_from_base[0]) - heading
            goal_heading = (
                np.arctan2(pot_from_base[1], pot_from_base[0]) - local_angle
            )
            from utils.planners_utils import _yaw_sweep_with_pass_check

            _yaw_sweep_with_pass_check(
                env,
                planner,
                np.array([np.cos(goal_heading), np.sin(goal_heading)]),
                plate_center,
                target_obj=target_veg,
                rot_cap=0.06,
                pass_dxy=0.04,
            )
            planner.planner.update_from_simulation()

    veg_now = target_veg.pose.p[0].cpu().numpy()
    dxy = np.linalg.norm(veg_now[:2] - plate_center[:2])
    if 0.05 <= dxy < 0.60 and veg_now[2] >= 1.0:
        # arm alignment: step the TCP toward the plate center in small
        # verified motions (a single long arm motion at the workspace edge
        # swings the vegetable out of the fingers). The prop-drive transport
        # reliably delivers the veg to ~0.13 m; these steps close the gap to
        # the release gate (0.05).
        for _ in range(6):
            veg_now = target_veg.pose.p[0].cpu().numpy()
            dxy = np.linalg.norm(veg_now[:2] - plate_center[:2])
            if dxy < 0.05 or veg_now[2] < 1.0:
                break
            tcp_now = agent.tcp.pose.sp
            residual = plate_center[:2] - veg_now[:2]
            rdist = float(np.linalg.norm(residual))
            if rdist < 0.01:
                break
            step = residual / rdist * min(rdist, 0.10)
            target_tcp = sapien.Pose(
                p=tcp_now.p + np.array([step[0], step[1], 0.0]), q=tcp_now.q
            )
            res = env.log_motion(
                "Align veggie over plate",
                _safe_manipulation,
                env,
                planner,
                # pi-lens-ignore: unchecked-throwing-call-python
                target_tcp,
                disable_lift_joint=False,
                n_init_qpos=300,
            )
            if res != -1 and _tcp_at(agent, target_tcp, tol=0.10):
                # corrective second motion: converges from the approximate pose
                res = env.log_motion(
                    "Align veggie over plate (corrective)",
                    _safe_manipulation,
                    env,
                    planner,
                    target_tcp,
                    disable_lift_joint=False,
                    n_init_qpos=300,
                )
            elif res == -1:
                # one retry: a single IK miss at the workspace edge often
                # succeeds on re-planning from the same pose
                res = env.log_motion(
                    "Align veggie over plate (retry)",
                    _safe_manipulation,
                    env,
                    planner,
                    target_tcp,
                    disable_lift_joint=False,
                    n_init_qpos=300,
                )
            planner.planner.update_from_simulation()
            if res == -1 or not _tcp_at(agent, target_tcp, tol=0.05):
                print("[WARN] arm alignment step failed")
                break
        veg_now = target_veg.pose.p[0].cpu().numpy()
        dxy = np.linalg.norm(veg_now[:2] - plate_center[:2])
        print(f"Aligned veggie over plate (residual {dxy:.3f} m)")
        env.log_event("info", "Veggie aligned over plate")

    # gate: only lower/release when the held veggie is actually OVER the plate
    # (well within its radius - a long vegetable released near the rim tips
    # off during the settle); a looser gate releases the veg beside the plate
    # and it falls off the counter
    veg_now = target_veg.pose.p[0].cpu().numpy()
    dxy = np.linalg.norm(veg_now[:2] - plate_center[:2])
    if dxy < 0.05:
        print("Lower vegetable onto plate (smooth vertical movement)")
        env.log_event("phase", "Lower vegetable onto plate")
        # drop the vegetable to just above the plate: a fixed 0.10 m drop is
        # not enough after a high lift (the release then bounces the vegetable
        # off the plate). Compute the drop from the held height to the plate
        # top plus the vegetable's LYING half height plus a small gap.
        veg_z = float(target_veg.pose.p[0].cpu().numpy()[2])
        plate_z = float(unwenv.plate.pose.p[0].cpu().numpy()[2])
        # the vegetable lies FLAT once released: its resting half-height is
        # the MINIMUM half-extent (for an elongated carrot the max is its
        # half-length, which would leave it hanging centimetres above the
        # plate - it then stays wedged in the open fingers and gets shaken
        # off during retract)
        half = float(np.min(obb.extents) / 2) if obb is not None else 0.02
        target_drop = max(0.005, veg_z - (plate_z + half + 0.005))
        # lower with small VERTICAL BASE-FIXED TCP steps: a torso drop swings
        # the near-full-extension arm like a pendulum - the object lands
        # centimetres off target; a free-base descent swings the whole robot.
        # pi-lens-ignore: unchecked-throwing-call-python
        # The follow_path executor writes ABSOLUTE plan qpos into the delta
        # pi-lens-ignore: unchecked-throwing-call-python
        # arm controller, so each step lands ~50% of the requested dz; loop
        # on the MEASURED vegetable height instead of a fixed step count.
        n_desc = 40
        # raise the TORSO fully BEFORE the descent: the base-fixed descend
        # executes via the torso (the IK keeps the arm angles ~fixed), so a
        # raised torso gives the descent the full travel to lower the
        # pi-lens-ignore: unchecked-throwing-call-python
        # vegetable deep into the pot. The pot's walls hold it even if the
        # release bounces it off the fingers.
        arm_action = agent.controller.controllers["arm"].qpos[0].cpu().numpy()
        body_action = agent.controller.controllers["body"].qpos[0].cpu().numpy().copy()
        body_action[0] = body_action[1] = 0.0
        body_action[2] += 0.06  # torso UP (absolute target)
        for _ in range(30):
            a = np.hstack([arm_action, planner.gripper_state, body_action,
                           np.array([0.0, 0.0, 0.0])])
            env.step(a)
        planner.planner.update_from_simulation()
        # the base-fixed plan descends via the TORSO (the IK keeps the arm
        # angles ~fixed) and the veg drops only ~40% of the requested dz, so
        # request a chunk (target_drop/6) and loop on the MEASURED height
        dz_step = max(target_drop / 6, 0.02)
        # the pot's bottom (the kinematic pot keeps its origin at the spawn
        # height; its half-extent places the bottom at the counter top), plus
        # the vegetable's lying half-height and a small gap - the vegetable
        # rests INSIDE the pot with its top below the rim
        # pi-lens-ignore: unchecked-throwing-call-python
        release_z = plate_z - float(unwenv.plate_half[2]) + half + 0.005
        fail_streak = 0
        stall_streak = 0
        for _k in range(n_desc):
            # pi-lens-ignore: unchecked-throwing-call-python
            veg_z_now = float(target_veg.pose.p[0].cpu().numpy()[2])
            if veg_z_now <= release_z + 0.01:
                break
            res = _descend_tcp_step(env, planner, dz_step)
            if res == -1:
                # descent limit reached - work with the current height
                fail_streak += 1
                if fail_streak >= 2:
                    break
            else:
                fail_streak = 0
            planner.planner.update_from_simulation()
            # pi-lens-ignore: unchecked-throwing-call-python
            veg_z_after = float(target_veg.pose.p[0].cpu().numpy()[2])
            print(f"[INFO] descend step {_k}: veg z {veg_z_now:.3f} -> {veg_z_after:.3f}")
            if veg_z_now - veg_z_after < 0.005:
                # a single short step is a plan/measurement transient; only
                # give up after 3 consecutive no-progress steps
                stall_streak += 1
                if stall_streak >= 3:
                    print("[WARN] descent stalled, releasing at current height")
                    break
            else:
                stall_streak = 0

        # low-orbit sweep: at this height a gentle yaw carries the vegetable
        # along its LOW orbit; when it passes over the plate, release there.
        # Trigger only when clearly off the plate (beyond radius+margin): a
        # sweep started ON the plate swings it off the rim instead.
        veg_now = target_veg.pose.p[0].cpu().numpy()
        # pi-lens-ignore: unchecked-throwing-call-python
        dxy_low = float(np.linalg.norm(veg_now[:2] - plate_center[:2]))
        if dxy_low > 0.045:
            base_xy = agent.base_link.pose.p[0].cpu().numpy()[:2]
            from utils.planners_utils import _yaw_sweep_with_pass_check
            _yaw_sweep_with_pass_check(
                env, planner,
                np.asarray(plate_center)[:2] - base_xy,
                plate_center,
                target_obj=target_veg,
                rot_cap=0.10, pass_dxy=0.04,
            )
            planner.planner.update_from_simulation()
            veg_now = target_veg.pose.p[0].cpu().numpy()
            # pi-lens-ignore: unchecked-throwing-call-python
            dxy_low = float(np.linalg.norm(veg_now[:2] - plate_center[:2]))
            print(f"[INFO] low-orbit sweep done: {dxy_low:.3f} m from plate")

        print("Release vegetable")
        env.log_event("phase", "Release vegetable")
        # The fingers are horizontal PLATES (±3 cm around the TCP) that wrap
        # the vegetable's middle. If the descend stalled above the pot floor,
        # the vegetable hangs between the plates and OPENING it lets the
        # plates' bottom edges catch its top as it falls - the vegetable is
        # LAUNCHED out of the pot (observed: veg flung 0.7 m after a descend
        # that stalled 13 cm above the floor). First LOWER the torso to press
        # the vegetable onto the pot floor (small increments, measured), then
        # open: with the vegetable supported, the open is safe and the walls
        # contain it.
        arm_action = agent.controller.controllers["arm"].qpos[0].cpu().numpy()
        body_action = agent.controller.controllers["body"].qpos[0].cpu().numpy().copy()
        body_action[0] = body_action[1] = 0.0
        for _ in range(15):
            # pi-lens-ignore: unchecked-throwing-call-python
            veg_z_now = float(target_veg.pose.p[0].cpu().numpy()[2])
            if veg_z_now <= release_z + 0.02:
                break
            body_action[2] -= 0.02  # absolute torso target, small steps
            for _ in range(6):
                a = np.hstack([arm_action, planner.gripper_state, body_action,
                               np.array([0.0, 0.0, 0.0])])
                env.step(a)
        planner.planner.update_from_simulation()
        # the torso alone cannot always press the vegetable onto the floor
        # (the arm's lever is ~0.4, the torso hits its limit with the veg
        # still above the pot bottom): retry the base-fixed vertical descent
        # from the pressed pose - the arm can now flex to reach the floor
        for _ in range(20):
            # pi-lens-ignore: unchecked-throwing-call-python
            veg_z_now = float(target_veg.pose.p[0].cpu().numpy()[2])
            if veg_z_now <= release_z + 0.02:
                break
            res = _descend_tcp_step(env, planner, 0.03)
            planner.planner.update_from_simulation()
            if res == -1:
                break
        veg_z_now = float(target_veg.pose.p[0].cpu().numpy()[2])
        if veg_z_now > release_z + 0.02:
            remaining = veg_z_now - release_z - 0.01
            print(f"[INFO] release descend fallback: torso drop {remaining:.3f} m")
            lower_torso_smooth(
                env,
                planner,
                target_drop=min(remaining, 0.16),
                total_steps=60,
                arm_action=arm_action,
                gripper_action=planner.gripper_state,
            )
        planner.planner.update_from_simulation()
        arm_action = agent.controller.controllers["arm"].qpos[0].cpu().numpy()
        body_action = agent.controller.controllers["body"].qpos[0].cpu().numpy().copy()
        body_action[0] = body_action[1] = 0.0
        gs = float(planner.gripper_state)  # -1 while holding
        # pi-lens-ignore: unchecked-throwing-call-python
        for frac in np.linspace(0.0, 1.0, 20):
            a = np.hstack([arm_action, gs + (0.6 - gs) * frac,
                           body_action, np.array([0.0, 0.0, 0.0])])
            env.step(a)
        planner.gripper_state = 0.6
        _detach_object(planner)
        planner.planner.update_from_simulation()
        # break any remaining finger contact: a WIDE vegetable (e.g. the
        # sweet potato, ~5 cm) stays squeezed between the OPEN fingers at
        # their max opening (~5 cm < the vegetable's width), so is_grasping
        # never clears and the static check never passes (observed:
        # pi-lens-ignore: unchecked-throwing-call-python
        # dbg_grasped=True with the veg on the pot floor). Retreat the base
        # (arm frozen, gripper open) until the fingers slide off the
        # vegetable; the pot walls hold it inside.
        if agent.is_grasping(target_veg):
            print("[INFO] release: vegetable still touching the gripper; "
                  "retreating the base to break contact")
            from utils.planners_utils import _velocity_segment
            base_now = agent.base_link.pose.p[0].cpu().numpy()
            retreat = base_now.copy()
            retreat[1] -= 0.40  # south, away from the counter
            arm_a = agent.controller.controllers["arm"].qpos[0].cpu().numpy()
            body_a = agent.controller.controllers["body"].qpos[0].cpu().numpy().copy()
            body_a[0] = body_a[1] = 0.0
            _velocity_segment(env, planner, retreat, arm_a, body_a,
                              planner.gripper_state, speed=0.18, tol=0.06)
            planner.planner.update_from_simulation()
    else:
        print(f"Transport failed: veggie {dxy:.2f} m from plate, aborting.")
        env.log_event(
            "error", "Transport did not reach plate", dxy=round(float(dxy), 3)
        )
        # same open-in-place release as the success path (see above).
        arm_action = agent.controller.controllers["arm"].qpos[0].cpu().numpy()
        body_action = agent.controller.controllers["body"].qpos[0].cpu().numpy().copy()
        body_action[0] = body_action[1] = 0.0
        gs = float(planner.gripper_state)
        for frac in np.linspace(0.0, 1.0, 20):
            a = np.hstack([arm_action, gs + (0.6 - gs) * frac,
                           body_action, np.array([0.0, 0.0, 0.0])])
            env.step(a)
        planner.gripper_state = 0.6
        # pi-lens-ignore: unchecked-throwing-call-python
        _detach_object(planner)
        planner.planner.update_from_simulation()

    print("Retract arm (Bypassing planner to lift torso back up)")
    env.log_event("phase", "Retract arm (bypassing planner to lift torso up)")
    # NO arm motion after the release: the open gripper hovers a couple cm
    # pi-lens-ignore: unchecked-throwing-call-python
    # above the released vegetable, and ANY arm motion clips it - observed
    # flings: torso lift sweeping the gripper across the plate (carrot flicked
    # 25 cm), and a straight-up lift dragging the fingertips through the veg.
    # The episode ends after the settle below, so the arm stays where it is;
    # the frozen open gripper does not disturb the resting vegetable.

    # settle: wait until the released vegetable STOPS MOVING (max ~300 steps;
    # a round vegetable spins in place long after its position freezes, and
    # an early evaluate fails the angular-velocity part of the static check).
    # The settle action must keep the GRIPPER OPEN and the TORSO UP: an
    # all-zero action drives the mimic gripper target to 0 (fully closed, the
    # closing fingers clamp the released vegetable and the stored squeeze
    # LAUNCHES it), AND the body controller is NOT delta (use_delta=False) -
    # a zero body action is the ABSOLUTE target 0, so the torso drops from
    # 0.34 to 0 and the arm collapses onto the plate (observed: torso 0.34 ->
    # 0.17, arm folded, veg flicked 40 cm). Hold the torso at its current
    # position and the gripper open.
    settle_act = np.zeros(env.action_space.shape)
    # hold the ARM at its current absolute targets too: the arm controller
    # is ABSOLUTE (pd_joint_pos, use_delta=False), so zero arm slots would
    # FOLD the arm and swing the open gripper through the released
    # vegetable (observed: wedged veg dragged 58 cm off the pot)
    settle_act[:7] = agent.controller.controllers["arm"].qpos[0].cpu().numpy()
    settle_act[7] = planner.gripper_state  # gripper slot: hold it open
    settle_act[8:11] = agent.controller.controllers["body"].qpos[0].cpu().numpy()
    for _ in range(600):
        vv = target_veg.linear_velocity[0].cpu().numpy()
        av = target_veg.angular_velocity[0].cpu().numpy()
        if float(np.linalg.norm(vv)) <= env.unwrapped.STATIC_V_MAX and float(
            np.linalg.norm(av)
        ) <= env.unwrapped.STATIC_AV_MAX:
            break
        env.step(settle_act)

    print("Task completed. Closing env...")
    ev = unwenv.evaluate()
    success = bool(ev["success"].item())  # pyright: ignore[reportAttributeAccessIssue]
    print("Success:", success,
          {k: v for k, v in ev.items() if k.startswith("dbg")})
    print("Success:", success)
    # pi-lens-ignore: unchecked-throwing-call-python
    env.log_event("result", "Task completed", success=success)
    env.reset()
    return success


if __name__ == "__main__":
    args = parse_args()
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    # seed mplib's global RNG (std::srand, OMPL and FCL): without it the
    # collision-aware IK / RRT outcomes vary run-to-run for the SAME seed -
    # the same geometry succeeded or failed 0 vs 81 times depending on the
    # unseeded draw sequence. With the seed the reach/grasp/transport are
    # reproducible and the batch flakiness disappears.
    import mplib.pymp as mplib_pymp
    mplib_pymp.set_global_seed(SEED)

    print(
        f"[INFO] seed={SEED}, render_mode='{args.render_mode}', "
        f"debug={args.debug}, info={args.info}, log_dir='{args.log_dir}'"
    )

    run_id = f"fridgeveggies_seed{SEED}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.log_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        "MyRoboCasa_FridgeVeggies-v1",
        num_envs=1,
        render_mode=None if args.no_video else args.render_mode,
        obs_mode="state" if args.no_video else "rgb",
        robot_uids="ds_fetch",
        control_mode="pd_joint_pos",
        sim_config=dict(scene_config=dict(cpu_workers=1, enable_enhanced_determinism=True)),
    )
    # Video-only recording: frames stream straight into ffmpeg, so RAM stays at
    # a single frame instead of RecordEpisode's whole-episode frame buffer
    # (~12 GB at the 2048x2048 render resolution).
    if not args.no_video:
        env = StreamingVideoRecorder(env, output_dir=str(run_dir), video_fps=30)
    env = PlannerLogger(
        env,
        log_dir=str(run_dir),
        name=f"fridgeveggies_seed{SEED}",
        log_freq=args.log_freq,
        run_dir=str(run_dir),
    )
    env.action_space.seed(SEED)
    with capture_stdout(env.dir / "console.log"):
        planning(env, SEED, debug=args.debug, info=args.info)
    env.close()
    if not args.no_video:
        print(f"[INFO] Video recording saved in '{run_dir}/'")
