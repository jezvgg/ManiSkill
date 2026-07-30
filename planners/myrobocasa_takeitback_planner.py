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
from utils.planners_utils import (
    lower_torso_smooth,
    retract_arm_lift_torso,
    move_base_backward_smooth,
    drive_base_to_object_target,
)


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
    drive_base_to_object_target(env, planner, cup_center, unwenv.cup_pos_sink, margin=0.04)
    print("Lower cup (Smooth vertical movement)")
    lower_torso_smooth(env, planner, target_drop=0.17, total_steps=100, vis=vis)

    print("Release cup")
    planner.open_gripper()
    planner.planner.update_from_simulation()

    print("Retract arm (Bypassing planner to lift torso back up)")
    retract_arm_lift_torso(env, planner, lift_amount=0.15, total_steps=40, vis=vis)

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
    print("Lower cup (Smooth vertical movement)")
    lower_torso_smooth(env, planner, target_drop=0.17, total_steps=100, vis=vis)

    print("Release cup")
    planner.open_gripper()
    planner.planner.update_from_simulation()

    print("Retract arm (Bypassing planner to lift torso back up)")
    retract_arm_lift_torso(env, planner, lift_amount=0.15, total_steps=40, vis=vis)

    print("Task completed. Closing env...")
    success = bool(unwenv.evaluate()["success"].item())
    print("Success:", success)
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
