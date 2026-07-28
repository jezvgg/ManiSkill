import argparse
import random

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
from mani_skill.utils.wrappers.record import RecordEpisode


def parse_args():
    parser = argparse.ArgumentParser(description="Motion planner for MyRoboCasa_TakeItBack-v1 scene")
    parser.add_argument("--seed", type=int, default=3, help="Random seed (default: 3)")
    parser.add_argument("--output-dir", type=str, default="videos",
                        help="Directory for video output (default: videos)")
    parser.add_argument("--render-mode", type=str, default="rgb_array",
                        choices=["rgb_array", "human", "sensors"],
                        help="Render mode (default: rgb_array)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode in planner")
    parser.add_argument("--info", action="store_true", help="Print environment info in planner")
    return parser.parse_args()


def move_base_backward_smooth(env, planner, target_base_pos, max_steps=200, eps=0.015, vis=False):
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
        obs, reward, terminated, truncated, info = env.step(action)

        if vis and hasattr(unwenv, "render_human"):
            unwenv.render_human()

    planner.planner.update_from_simulation()


def planning(env, seed, debug=False, vis=None, info=False):
    vis = vis or env.unwrapped.render_mode == "human"

    FINGER_LENGTH = 0.025
    unwenv: MyRoboCasaSceneTakeItBack = env.unwrapped
    agent: Fetch = unwenv.agent
    obs, _ = env.reset(seed=seed, options={"reconfigure": True})
    planner = FetchMotionPlanningSapienSolver(
        env,
        base_pose=agent.robot.pose,
        vis=vis,
        print_env_info=info,
        debug=debug,
    )
    print("Calculate grasp position")
    mesh = unwenv.cup.get_first_collision_mesh(to_world_frame=True)
    if mesh is not None:
        obb: Box = mesh.bounding_box_oriented
        cup_center = obb.center_mass.copy()

    planner.planner.update_from_simulation()

    tcp_pos = agent.tcp.pose.p[0].cpu().numpy()
    ee_direction = obb.center_mass - tcp_pos
    ee_direction = ee_direction / np.linalg.norm(ee_direction)
    target_closing = agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()

    grasp_info = compute_box_grasp_thin_side_info(
        obb,
        ee_direction=ee_direction,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
        ortho=True,
    )
    closing, center, approaching = (
        grasp_info["closing"],
        grasp_info["center"],
        grasp_info["approaching"],
    )
    grasp_pose = agent.build_grasp_pose(approaching, closing, center)
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.1])

    base_pos = agent.base_link.pose.p[0].cpu().numpy()
    target_to_cup = reach_pose.p - cup_center
    base_to_cup = base_pos - cup_center

    print("Reaching cup")
    res = planner.static_manipulation(reach_pose, disable_lift_joint=False)
    planner.planner.update_from_simulation()

    print("Grasp cup")
    grasp_cup = grasp_pose
    planner.static_manipulation(grasp_cup, disable_lift_joint=False)
    planner.close_gripper()
    planner.planner.update_from_simulation()

    print("Lift cup")
    lift_pose = sapien.Pose(grasp_pose.p + np.array([0, 0, 0.15]), grasp_pose.q)
    planner.static_manipulation(lift_pose, disable_lift_joint=False)
    planner.planner.update_from_simulation()

    print("\n--- Phase 1: Drive base toward sink ---")
    initial_base_pos = agent.base_link.pose.sp.p.copy()
    base_tf = agent.base_link.pose.sp.to_transformation_matrix()
    base_pos_world = planner.base_env.agent.base_link.pose.sp.p.copy()
    delta_world = unwenv.cup_pos_sink - cup_center
    delta_world[2] = 0.0
    dist = np.linalg.norm(delta_world)
    if dist > 1e-3:
        dir_world = delta_world / dist
        delta_world += dir_world * 0.04
    base_target_pos = base_pos_world + delta_world

    print(f"[INFO] Initial base position: {initial_base_pos}")
    print(f"[INFO] Target position at sink: {base_target_pos}")
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

        if abs(base_turn_angle) < np.deg2rad(2.0):
            forward_dist = np.dot(delta_world, base_x_axis_world)
            print(f"[INFO] Moving base forward by scalar dist = {forward_dist:.4f} m")
            base_forward_target = base_pos_world + base_x_axis_world * forward_dist
            planner.move_base_forward(base_forward_target, n_init_qpos=100)
        else:
            print(f"[INFO] Driving base to target_pos: {base_target_pos}")
            planner.drive_base(target_pos=base_target_pos)
        planner.planner.update_from_simulation()
        print(f"[INFO] Base position after driving to sink: {planner.base_env.agent.base_link.pose.sp.p}")
    print("Lower cup (Smooth vertical movement)")
    arm_action = unwenv.agent.controller.controllers["arm"].qpos[0].cpu().numpy()
    start_body_action = (
        unwenv.agent.controller.controllers["body"].qpos[0].cpu().numpy().copy()
    )

    base_action = np.array([0.0, 0.0])
    gripper_action = planner.gripper_state

    total_steps = 100
    target_drop = 0.17

    for step in range(total_steps):
        fraction = (step + 1) / total_steps
        current_drop = fraction * target_drop

        body_action = start_body_action.copy()
        body_action[2] -= current_drop

        action = np.hstack([arm_action, gripper_action, body_action, base_action])
        obs, reward, terminated, truncated, info = env.step(action)

        if vis and hasattr(unwenv, "render_human"):
            unwenv.render_human()

    planner.planner.update_from_simulation()

    print("Release cup")
    planner.open_gripper()
    planner.planner.update_from_simulation()

    print("Retract arm (Bypassing planner to lift torso back up)")
    body_action[2] += 0.15
    action = np.hstack([arm_action, planner.gripper_state, body_action, base_action])

    for _ in range(40):
        env.step(action)
        if vis and hasattr(unwenv, "render_human"):
            unwenv.render_human()

    planner.planner.update_from_simulation()

    print("Calculate grasp position")
    mesh = unwenv.cup.get_first_collision_mesh(to_world_frame=True)
    if mesh is not None:
        obb: Box = mesh.bounding_box_oriented
        cup_center = obb.center_mass.copy()

    planner.planner.update_from_simulation()

    tcp_pos = agent.tcp.pose.p[0].cpu().numpy()
    ee_direction = obb.center_mass - tcp_pos
    ee_direction = ee_direction / np.linalg.norm(ee_direction)
    target_closing = agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()

    grasp_info = compute_box_grasp_thin_side_info(
        obb,
        ee_direction=ee_direction,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
        ortho=True,
    )
    closing, center, approaching = (
        grasp_info["closing"],
        grasp_info["center"],
        grasp_info["approaching"],
    )
    grasp_pose = agent.build_grasp_pose(approaching, closing, center)
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.1])

    base_pos = agent.base_link.pose.p[0].cpu().numpy()
    target_to_cup = reach_pose.p - cup_center
    base_to_cup = base_pos - cup_center

    print("Reaching cup")
    res = planner.static_manipulation(reach_pose, disable_lift_joint=False)
    planner.planner.update_from_simulation()

    print("Grasp cup")
    grasp_cup = grasp_pose
    planner.static_manipulation(grasp_cup, disable_lift_joint=False)
    planner.close_gripper()
    planner.planner.update_from_simulation()

    print("Lift cup")
    lift_pose = sapien.Pose(grasp_pose.p + np.array([0, 0, 0.15]), grasp_pose.q)
    planner.static_manipulation(lift_pose, disable_lift_joint=False)
    planner.planner.update_from_simulation()

    print("\n--- Phase 2: Drive base backward to initial position (table) ---")
    planner.planner.update_from_simulation()
    cur_base_p = planner.base_env.agent.base_link.pose.sp.p.copy()
    move_base_backward_smooth(env, planner, initial_base_pos, vis=vis)
    start_body_action = (
        unwenv.agent.controller.controllers["body"].qpos[0].cpu().numpy().copy()
    )

    base_action = np.array([0.0, 0.0])
    gripper_action = planner.gripper_state

    total_steps = 100
    target_drop = 0.17

    for step in range(total_steps):
        fraction = (step + 1) / total_steps
        current_drop = fraction * target_drop

        body_action = start_body_action.copy()
        body_action[2] -= current_drop

        action = np.hstack([arm_action, gripper_action, body_action, base_action])
        obs, reward, terminated, truncated, info = env.step(action)

        if vis and hasattr(unwenv, "render_human"):
            unwenv.render_human()

    planner.planner.update_from_simulation()

    print("Release cup")
    planner.open_gripper()
    planner.planner.update_from_simulation()

    print("Retract arm (Bypassing planner to lift torso back up)")
    body_action[2] += 0.15
    action = np.hstack([arm_action, planner.gripper_state, body_action, base_action])

    for _ in range(40):
        env.step(action)
        if vis and hasattr(unwenv, "render_human"):
            unwenv.render_human()

    planner.planner.update_from_simulation()

    print("Task completed. Closing env...")
    env.reset()
    return True


if __name__ == "__main__":
    args = parse_args()
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print(f"[INFO] seed={SEED}, output_dir='{args.output_dir}', render_mode='{args.render_mode}', "
          f"debug={args.debug}, info={args.info}")

    env = gym.make(
        "MyRoboCasa_TakeItBack-v1",
        num_envs=1,
        render_mode=args.render_mode,
        obs_mode="rgb",
        robot_uids="ds_fetch",
        control_mode="pd_joint_pos",
    )
    env = RecordEpisode(
        env,
        output_dir=args.output_dir,
        save_trajectory=False,
        save_video=True,
        video_fps=30,
        trajectory_name="take_it_back",
    )

    env.action_space.seed(SEED)
    planning(env, SEED, debug=args.debug, info=args.info)
    env.close()
    print(f"[INFO] Video recording saved in '{args.output_dir}/'")
