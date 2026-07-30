import argparse
import os
import random

import gymnasium as gym
import numpy as np
import sapien
import torch
from trimesh.primitives import Box

from mani_skill.agents.robots import Fetch
from mani_skill.envs.tasks import MyRoboCasaScene
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
    align_arm_over_target,
    drive_base_to_object_target,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Motion planner for MyRoboCasa-v1 scene")
    parser.add_argument("--seed", type=int, default=3, help="Random seed (default: 3)")
    parser.add_argument("--output-dir", type=str, default=os.path.join("videos", "my_robocasa"),
                        help="Directory for video output (default: videos/my_robocasa)")
    parser.add_argument("--render-mode", type=str, default="rgb_array",
                        choices=["rgb_array", "human", "sensors"],
                        help="Render mode (default: rgb_array)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode in planner")
    parser.add_argument("--info", action="store_true", help="Print environment info in planner")
    return parser.parse_args()


def planning(env, seed, debug=False, vis=None, info=False) -> bool:
    if vis is None:
        vis = (env.unwrapped.render_mode == "human")
    unwenv: MyRoboCasaScene = env.unwrapped
    agent: Fetch = unwenv.agent
    FINGER_LENGTH = 0.025
    obs, _ = env.reset(seed=seed, options={"reconfigure": True})
    planner = FetchMotionPlanningSapienSolver(
        env,
        base_pose=agent.robot.pose,
        vis=vis,
        print_env_info=info,
        debug=debug,
    )

    mesh = unwenv.cup.get_first_collision_mesh(to_world_frame=True)
    if mesh is not None:
        obb: Box = mesh.bounding_box_oriented
        cup_center = obb.center_mass.copy()

    bowl_mesh = unwenv.bowl.get_first_collision_mesh(to_world_frame=True)
    if bowl_mesh is not None:
        bowl_obb: Box = bowl_mesh.bounding_box_oriented
        bowl_center = bowl_obb.center_mass.copy()

    planner.planner.update_from_simulation()


    print("Calculate grasp position")
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

    # Validation: if the target is on the far side of the cup relative to the robot base
    if np.dot(target_to_cup, base_to_cup) < 0:
        print("Validation failed: grasp is diametrically opposite. Flipping approaching direction...")
        grasp_info = compute_box_grasp_thin_side_info(
            obb,
            ee_direction=-ee_direction,
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

    print("Reaching cup")
    res = planner.static_manipulation(reach_pose, disable_lift_joint=False)

    if res == -1:
        print("Reaching cup failed, trying opposite closing direction as fallback...")
        grasp_info = compute_box_grasp_thin_side_info(
            obb,
            ee_direction=-ee_direction if np.dot(target_to_cup, base_to_cup) < 0 else ee_direction,
            target_closing=-target_closing,
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

    print("Drive base toward bowl")
    drive_base_to_object_target(env, planner, cup_center, bowl_center, margin=0.04)


    print("Aligning arm over the bowl (transit step)")
    cup_pos = unwenv.cup.pose.p[0].cpu().numpy()
    bowl_pos = unwenv.bowl.pose.p[0].cpu().numpy()
    align_arm_over_target(env, planner, cup_pos, bowl_pos, vis=vis)

    print("Lower cup (Smooth vertical movement)")
    arm_action = unwenv.agent.controller.controllers["arm"].qpos[0].cpu().numpy()
    lower_torso_smooth(env, planner, target_drop=0.17, total_steps=100, vis=vis, arm_action=arm_action)

    # Синхронизируем планер
    planner.planner.update_from_simulation()

    print("Release cup")
    planner.open_gripper()
    planner.planner.update_from_simulation()

    print("Retract arm (Bypassing planner to lift torso back up)")
    retract_arm_lift_torso(env, planner, lift_amount=0.15, total_steps=40, vis=vis, arm_action=arm_action)

    planner.planner.update_from_simulation()

    print("Task completed. Closing env...")
    success = unwenv.evaluate()["success"]
    print("Success:", success[0])
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
        "MyRoboCasa-v1",
        num_envs=1,
        render_mode=args.render_mode,
        robot_uids="ds_fetch",
        control_mode="pd_joint_pos",
    )
    env = RecordEpisode(
        env,
        output_dir=args.output_dir,
        save_video=True,
        video_fps=30,
        save_on_reset=True,
    )
    env.action_space.seed(SEED)
    planning(env, SEED, debug=args.debug, info=args.info)
    env.close()
