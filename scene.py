import os

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

if __name__ == "__main__":
    env = gym.make(
        "MyRoboCasa-v1",
        num_envs=1,
        render_mode="rgb_array",
        robot_uids="ds_fetch",
        control_mode="pd_joint_pos",
    )
    env = RecordEpisode(
        env,
        output_dir=os.path.join("videos", "my_robocasa"),
        save_video=True,
        video_fps=30,
        save_on_reset=True,
    )

    unwenv: MyRoboCasaScene = env.unwrapped
    agent: Fetch = unwenv.agent
    FINGER_LENGTH = 0.025

    obs, _ = env.reset(options={"reconfigure": True})
    planner = FetchMotionPlanningSapienSolver(
        env, base_pose=agent.robot.pose, vis=True, print_env_info=True
    )

    mesh = unwenv.cup.get_first_collision_mesh(to_world_frame=True)
    if mesh is not None:
        obb: Box = mesh.bounding_box_oriented

    bowl_mesh = unwenv.bowl.get_first_collision_mesh(to_world_frame=True)
    if bowl_mesh is not None:
        bowl_obb: Box = bowl_mesh.bounding_box_oriented

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

    print("Reaching cup")
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.1])
    planner.static_manipulation(reach_pose, disable_lift_joint=False)
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

    print("Go to bowl")
    bowl_over_pos = bowl_obb.center_mass.copy()
    bowl_over_pos[2] = lift_pose.p[2]
    bowl_over_pose = sapien.Pose(bowl_over_pos, grasp_pose.q)
    planner.static_manipulation(bowl_over_pose, disable_lift_joint=False)
    planner.planner.update_from_simulation()

    print("Lower cup")
    bowl_lower_pos = bowl_obb.center_mass.copy()
    bowl_lower_pos[2] += 0.08
    bowl_lower_pose = sapien.Pose(bowl_lower_pos, grasp_pose.q)
    planner.static_manipulation(bowl_lower_pose, disable_lift_joint=False)
    planner.planner.update_from_simulation()

    print("Release cup")
    planner.open_gripper()
    planner.planner.update_from_simulation()

    print("Retract arm")
    retract_pose = bowl_lower_pose * sapien.Pose([0, 0, -0.2])
    planner.static_manipulation(retract_pose, disable_lift_joint=False)
    planner.planner.update_from_simulation()

    print("Task completed. Closing env...")
    env.reset()
    env.close()
