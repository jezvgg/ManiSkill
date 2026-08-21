import gymnasium as gym
import numpy as np
import sapien
import torch
import mplib

def lower_torso_smooth(env, planner, target_drop=0.17, total_steps=100, vis=False, arm_action=None, gripper_action=None):
    """
    Lower the torso slowly and smoothly by interpolating the height index.
    """
    unw_env = env.unwrapped
    if arm_action is None:
        arm_action = unw_env.agent.controller.controllers["arm"].qpos[0].cpu().numpy()
    start_body_action = unw_env.agent.controller.controllers["body"].qpos[0].cpu().numpy().copy()
    base_action = np.array([0.0, 0.0])
    if gripper_action is None:
        gripper_action = planner.gripper_state

    for step in range(total_steps):
        fraction = (step + 1) / total_steps
        current_drop = fraction * target_drop

        body_action = start_body_action.copy()
        body_action[2] -= current_drop

        action = np.hstack([arm_action, gripper_action, body_action, base_action])
        env.step(action)

        if vis and hasattr(unw_env, "render_human"):
            unw_env.render_human()

    planner.planner.update_from_simulation()

def retract_arm_lift_torso(env, planner, lift_amount=0.15, total_steps=40, vis=False, arm_action=None, gripper_action=None):
    """
    Lifts torso back up by bypassing the planner.
    """
    unw_env = env.unwrapped
    if arm_action is None:
        arm_action = unw_env.agent.controller.controllers["arm"].qpos[0].cpu().numpy()
    body_action = unw_env.agent.controller.controllers["body"].qpos[0].cpu().numpy().copy()
    body_action[2] += lift_amount
    base_action = np.array([0.0, 0.0])
    if gripper_action is None:
        gripper_action = planner.gripper_state

    action = np.hstack([arm_action, gripper_action, body_action, base_action])

    for _ in range(total_steps):
        env.step(action)
        if vis and hasattr(unw_env, "render_human"):
            unw_env.render_human()

    planner.planner.update_from_simulation()


def move_base_backward_smooth(env, planner, target_base_pos, max_steps=200, eps=0.015, vis=False):
    """
    Smoothly move the base backward toward target_base_pos using simple velocity control.
    """
    unwenv = env.unwrapped
    agent = unwenv.agent
    arm_action = unwenv.agent.controller.controllers["arm"].qpos[0].cpu().numpy()
    body_action = unwenv.agent.controller.controllers["body"].qpos[0].cpu().numpy().copy()
    body_action[0] = body_action[1] = 0.0
    gripper_action = planner.gripper_state

    print(f"[INFO] Smooth base control to target pos: {target_base_pos}")
    for step in range(max_steps):
        planner.planner.update_from_simulation()
        cur_base_p = unwenv.agent.base_link.pose.sp.p.copy()
        cur_base_dir = agent.base_link.pose.sp.to_transformation_matrix()[:3, 0]

        back_delta_world = target_base_pos - cur_base_p
        back_delta_world[2] = 0.0
        dist_to_target = np.linalg.norm(back_delta_world)

        if dist_to_target < eps:
            print(f"[INFO] Reached target base position: {cur_base_p} (dist={dist_to_target:.4f} m, step={step})")
            break

        rem_dist = float(np.dot(back_delta_world, cur_base_dir))
        vel = np.clip(rem_dist * 2.5, -0.6, 0.6)
        base_action = np.array([vel, 0.0])

        action = np.hstack([arm_action, gripper_action, body_action, base_action])
        env.step(action)

        if vis and hasattr(unwenv, "render_human"):
            unwenv.render_human()

    planner.planner.update_from_simulation()

def align_arm_over_target(env, planner, source_pos, target_pos, vis=False):
    """
    Align arm horizontally by calculating delta dx, dy between source and target positions.
    """
    unwenv = env.unwrapped
    agent = unwenv.agent
    dx = target_pos[0] - source_pos[0]
    dy = target_pos[1] - source_pos[1]

    if np.hypot(dx, dy) > 0.005:
        print(f"Correction needed: dx={dx:.4f}, dy={dy:.4f}")
        current_tcp_pose = agent.tcp.pose.sp
        target_tcp_p = current_tcp_pose.p.copy()
        target_tcp_p[0] += dx
        target_tcp_p[1] += dy

        result = planner.planner.plan_screw(
            mplib.Pose(target_tcp_p, current_tcp_pose.q),
            planner.robot.get_qpos().cpu().numpy()[0],
            time_step=planner.base_env.control_timestep,
            masked_joints=[True, True, True, False] + [False]*11
        )
        if result["status"] == "Success":
            planner.follow_path(result)
        else:
            print("[WARNING] Alignment screw failed:", result["status"])
        planner.planner.update_from_simulation()

    if hasattr(planner, "render_wait"):
        planner.render_wait()

def drive_base_to_position(env, planner, target_pos, chunk=0.5, max_rot=300,
                           rot_gain=1.2, rot_cap=0.25, align_deg=4):
    """Drive the base to an arbitrary floor position.

    The ds_fetch base translation joints respond unreliably to velocity commands
    (direction/magnitude vary with yaw and drift during rotation), while the yaw
    joint is linear and the screw-based `move_base_forward` translates reliably
    at low heading error. So: (1) yaw-align with velocity control, (2) translate
    with short chunked screw moves, re-aligning between chunks.
    """
    unwenv = env.unwrapped
    agent = unwenv.agent
    target = np.asarray(target_pos, dtype=float).copy()
    target[2] = 0.0
    arm_action = agent.controller.controllers["arm"].qpos[0].cpu().numpy()
    body_action = agent.controller.controllers["body"].qpos[0].cpu().numpy().copy()
    body_action[0] = body_action[1] = 0.0
    gripper_action = planner.gripper_state

    def heading_error():
        sp = agent.base_link.pose.sp
        base_p = sp.p.copy()
        base_p[2] = 0.0
        delta = target - base_p
        dist = np.linalg.norm(delta)
        if dist < 1e-6:
            return 0.0, dist, base_p
        xa = sp.to_transformation_matrix()[:3, 0]
        xa[2] = 0.0
        nx = np.linalg.norm(xa)
        if nx < 1e-6:
            return 0.0, dist, base_p
        xa /= nx
        dt = delta / dist
        return np.arctan2(np.cross(xa, dt)[2], np.dot(xa, dt)), dist, base_p

    # --- phase 1: yaw-align (velocity rotation; yaw joint is reliable) ---
    for _ in range(max_rot):
        he, dist, _ = heading_error()
        if abs(he) < np.deg2rad(align_deg):
            break
        ba = np.array([0.0, float(np.clip(rot_gain * he, -rot_cap, rot_cap))])
        env.step(np.hstack([arm_action, gripper_action, body_action, ba]))
    planner.planner.update_from_simulation()

    # --- phase 2: chunked screw forward translation, velocity fallback ---
    for _ in range(20):
        he, dist, base_p = heading_error()
        if dist < 0.03:
            break
        # re-align before each chunk (rotation drift / small heading errors)
        for _ in range(60):
            he, dist, base_p = heading_error()
            if abs(he) < np.deg2rad(align_deg):
                break
            ba = np.array([0.0, float(np.clip(rot_gain * he, -rot_cap, rot_cap))])
            env.step(np.hstack([arm_action, gripper_action, body_action, ba]))
        planner.planner.update_from_simulation()
        he, dist, base_p = heading_error()
        if dist < 0.03:
            break
        delta = target - base_p
        waypoint = base_p + delta * min(1.0, chunk / max(dist, 1e-6))
        res = planner.move_base_forward(waypoint, n_init_qpos=100)
        if res == -1:
            print("[INFO] drive_base_to_position: screw segment failed, trying shorter")
            waypoint = base_p + delta * min(1.0, 0.25 / max(dist, 1e-6))
            res = planner.move_base_forward(waypoint, n_init_qpos=100)
        if res == -1:
            # screw translation is unreliable with the arm extended: fall back
            # to direct velocity control (base yaw joint is linear, x/y joints
            # respond at most yaws)
            print("[INFO] drive_base_to_position: using velocity fallback")
            res = _velocity_segment(env, planner, waypoint, arm_action, body_action,
                                    gripper_action, rot_gain, rot_cap, align_deg)
            if res == -1:
                print("[INFO] drive_base_to_position: giving up at", base_p)
                break
        planner.planner.update_from_simulation()
    return 0


def _velocity_segment(env, planner, target_pos, arm_action, body_action,
                      gripper_action, rot_gain, rot_cap, align_deg,
                      max_steps=300, speed_gain=1.0, max_speed=0.3):
    """Translate the base toward target_pos with direct velocity commands;
    returns 0 if it got within 0.05 m, -1 otherwise. The ds_fetch x/y joints
    respond unreliably at some yaws, so this is a best-effort fallback."""
    unwenv = env.unwrapped
    agent = unwenv.agent
    target = np.asarray(target_pos, dtype=float).copy()
    target[2] = 0.0
    last = None
    stalled = 0
    for _ in range(max_steps):
        sp = agent.base_link.pose.sp
        base_p = sp.p.copy()
        base_p[2] = 0.0
        delta = target - base_p
        dist = np.linalg.norm(delta)
        if dist < 0.05:
            return 0
        xa = sp.to_transformation_matrix()[:3, 0]
        xa[2] = 0.0
        nx = np.linalg.norm(xa)
        if nx < 1e-6:
            return -1
        xa /= nx
        dt = delta / dist
        he = np.arctan2(np.cross(xa, dt)[2], np.dot(xa, dt))
        ba = np.array([0.0, float(np.clip(rot_gain * he, -rot_cap, rot_cap))])
        if abs(he) < np.deg2rad(align_deg):
            ba[0] = float(np.clip(dist * speed_gain, -max_speed, max_speed))
        env.step(np.hstack([arm_action, gripper_action, body_action, ba]))
        moved = last is not None and dist < last - 0.01
        if not moved:
            stalled += 1
            if stalled > 80:
                break
        else:
            stalled = 0
        last = dist
    planner.planner.update_from_simulation()
    return -1


def drive_base_to_object_target(env, planner, current_obj_pos, target_obj_pos, margin=0.04):
    """
    Calculates displacement required to transport object from current to target position.
    Translates robot base by this displacement (with margin).
    Uses linear forward movement if heading misalignment is small, else drives base generally.
    """
    unwenv = env.unwrapped
    agent = unwenv.agent
    base_pos_world = agent.base_link.pose.sp.p.copy()
    base_tf = agent.base_link.pose.sp.to_transformation_matrix()

    delta_world = target_obj_pos - current_obj_pos
    delta_world[2] = 0.0
    dist = np.linalg.norm(delta_world)

    if dist > 1e-3:
        dir_world = delta_world / dist
        delta_world += dir_world * margin

    base_target_pos = base_pos_world + delta_world

    print(f"[INFO] Helper driving base. Current pos: {base_pos_world}, Target: {base_target_pos}")
    print(f"[INFO] Delta world vector: {delta_world}, distance: {dist:.4f} m")

    if dist > 1e-3:
        base_x_axis_world = base_tf[:3, 0]
        base_x_axis_world = base_x_axis_world / np.linalg.norm(base_x_axis_world)
        transfer_direction_world = delta_world / np.linalg.norm(delta_world)
        dot_prod = np.clip(
            np.dot(base_x_axis_world, transfer_direction_world), -1.0, 1.0
        )
        base_turn_angle = np.arccos(dot_prod)
        print(f"[INFO] Base forward axis (X): {base_x_axis_world}")
        print(f"[INFO] Base turn angle: {np.rad2deg(base_turn_angle):.2f} deg")

        # long transports are driven with direct velocity control
        drive_base_to_position(env, planner, base_target_pos)

        planner.planner.update_from_simulation()
