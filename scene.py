import os

import gymnasium as gym
import numpy as np
import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.robocasa.scene_builder import RoboCasaSceneBuilder
from mani_skill.utils.building import actors
from mani_skill.utils.structs import Pose, Actor
from mani_skill.utils import sapien_utils
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.utils.geometry.rotation_conversions import axis_angle_to_quaternion
from mani_skill.examples.motionplanning.fetch.extand import FetchMotionPlanningSapienSolver
from mani_skill import ASSET_DIR
import sapien.physx as physx

from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from mani_skill.examples.motionplanning.fetch.utils import compute_box_grasp_thin_side_info

if __name__ == '__main__':


    env = gym.make("MyRoboCasa-v1", 
            num_envs=1, 
            render_mode="human", 
            robot_uids='ds_fetch',
            control_mode='pd_joint_pos'
            )


    env = RecordEpisode(
        env, 
        output_dir=os.path.join("videos", "my_robocasa"), 
        save_video=True, 
        video_fps=30,
        save_on_reset=False,
    )

    print("Resetting environment...")
    obs, _ = env.reset(options={"reconfigure": True})

    print("Creating planner...")
    planner = FetchMotionPlanningSapienSolver(
        env,
        base_pose=env.unwrapped.agent.robot.pose,
        vis=True,
        print_env_info=True
        )
    print("Planner created!")

    FINGER_LENGTH = 0.025

    # Подходим к чашке (cup) сверху
    mesh = env.unwrapped.cup.get_first_collision_mesh(to_world_frame=False)
    if mesh is not None:
        obb = mesh.bounding_box_oriented
    print(obb)
    target_closing = env.unwrapped.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    ee_direction = env.unwrapped.agent.tcp.pose.to_transformation_matrix()[0, :3, 2].cpu().numpy()

    grasp_info = compute_box_grasp_thin_side_info(
        obb,
        ee_direction=ee_direction,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )

    closing, center, approaching = grasp_info["closing"], grasp_info["center"], grasp_info["approaching"]
    grasp_pose = env.unwrapped.agent.build_grasp_pose(approaching, closing, center)
    print(grasp_pose.p)

    print("Reaching cup")
    reach_pose = grasp_pose
    planner.static_manipulation(reach_pose)
    planner.planner.update_from_simulation()

    print("Grasp cup")
    # planner.move_to_pose_with_RRTConnect(grasp_pose)
    # planner.close_gripper()
    # planner.planner.update_from_simulation()

    while True:
        obs, rew, terminated, truncated, info = env.step(torch.as_tensor([0]*13))
        done = (terminated | truncated).any()

    env.reset()
    env.close()
