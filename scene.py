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
        render_mode="human",
        robot_uids="ds_fetch",
        control_mode="pd_joint_pos",
    )
    env = RecordEpisode(
        env,
        output_dir=os.path.join("videos", "my_robocasa"),
        save_video=True,
        video_fps=30,
        save_on_reset=False,
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

    print("Calculate rotating position")
    cup_pos = obb.center_mass
    robot_pos = agent.base_link.pose.p[0].cpu().numpy()
    direction = cup_pos - robot_pos
    direction[2] = 0

    print("Rotating base to a cup")
    planner.rotate_base_z(direction)
    planner.planner.update_from_simulation()

    tcp_pos = agent.tcp.pose.p[0].cpu().numpy()
    ee_direction = obb.center_mass - tcp_pos
    ee_direction = ee_direction / np.linalg.norm(ee_direction)
    target_closing = agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()

    print("Calculate grasp position")
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
    base_mask = [True, True, True] + [False] * 12
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.2])
    planner.static_manipulation(reach_pose, disable_lift_joint=False)
    planner.planner.update_from_simulation()

    print("Grasp cup")
    grasp_cup = grasp_pose * sapien.Pose([-0.03, 0, -0.02])
    planner.static_manipulation(grasp_cup, disable_lift_joint=False)
    planner.close_gripper()
    planner.planner.update_from_simulation()

    current_arm_pos = agent.controller.controllers["arm"].qpos[0].cpu().numpy()
    current_gripper_pos = agent.controller.controllers["gripper"].qpos[0].cpu().numpy()
    static_action = np.zeros(13)
    static_action[0:7] = current_arm_pos
    static_action[7] = current_gripper_pos[0]
    static_action_tensor = torch.as_tensor(static_action)

    while True:
        obs, rew, terminated, truncated, info = env.step(static_action_tensor)
        done = (terminated | truncated).any()

    env.reset()
    env.close()
