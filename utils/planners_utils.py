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

        if np.cross(base_x_axis_world, transfer_direction_world)[2] < 0:
            base_turn_angle = -base_turn_angle

        if abs(base_turn_angle) < np.deg2rad(2.0) or abs(abs(base_turn_angle) - np.pi) < np.deg2rad(2.0):
            forward_dist = np.dot(delta_world, base_x_axis_world)
            print(f"[INFO] Moving base forward by scalar dist = {forward_dist:.4f} m")
            base_forward_target = base_pos_world + base_x_axis_world * forward_dist
            planner.move_base_forward(base_forward_target, n_init_qpos=100)
        else:
            print(f"[INFO] Driving base to target_pos: {base_target_pos}")
            planner.drive_base(target_pos=base_target_pos)

        planner.planner.update_from_simulation()
