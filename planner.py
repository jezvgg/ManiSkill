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

if __name__ == "__main__":
    SEED = 3
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    env = gym.make(
        "MyRoboCasa-v1",
        num_envs=1,
        render_mode="rgb_array",
        robot_uids="ds_fetch",
        control_mode="pd_joint_pos",
        render_backend="pci:0000:00:00.0",
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

    env.action_space.seed(SEED)
    obs, _ = env.reset(seed=SEED, options={"reconfigure": True})
    planner = FetchMotionPlanningSapienSolver(
        env, base_pose=agent.robot.pose, vis=False, print_env_info=True, debug=True
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

    print("Drive base toward bowl")
    base_tf = agent.base_link.pose.sp.to_transformation_matrix()
    world_to_base_rot = base_tf[:3, :3].T
    cup_center_local = world_to_base_rot @ (cup_center - base_tf[:3, 3])
    bowl_center_local = world_to_base_rot @ (bowl_center - base_tf[:3, 3])
    base_local_delta = bowl_center_local - cup_center_local
    base_local_delta[2] = 0.0
    base_transfer_delta = base_tf[:3, :3] @ base_local_delta
    print("base transfer world:", np.round(base_transfer_delta, 4))
    print("base transfer local:", np.round(base_local_delta, 4))
    print("base +X world:", np.round(base_tf[:3, 0], 4))
    print("base +Y world:", np.round(base_tf[:3, 1], 4))
    if np.linalg.norm(base_transfer_delta) > 1e-3:
        base_pos_for_drive = planner.base_env.agent.base_link.pose.sp.p.copy()
        base_x_axis = base_tf[:3, 0]
        base_x_axis = base_x_axis / np.linalg.norm(base_x_axis)
        transfer_direction = base_transfer_delta / np.linalg.norm(base_transfer_delta)
        base_turn_angle = np.arccos(
            np.clip(np.dot(base_x_axis, transfer_direction), -1.0, 1.0)
        )
        if np.cross(base_x_axis, transfer_direction)[2] < 0:
            base_turn_angle = -base_turn_angle

        if abs(base_turn_angle) < np.deg2rad(2.0):
            base_forward_delta = base_x_axis * base_local_delta[0]
            base_target_pos = base_pos_for_drive + base_forward_delta
            print(
                "scene base move mode: forward",
                "angle deg:",
                np.round(np.rad2deg(base_turn_angle), 4),
            )
            print("scene base_pos_for_drive:", np.round(base_pos_for_drive, 4))
            print("scene base_target_pos:", np.round(base_target_pos, 4))
            planner.move_base_forward(base_target_pos, n_init_qpos=100)
        else:
            base_target_pos = base_pos_for_drive + base_transfer_delta
            print(
                "scene base move mode: drive_base",
                "angle deg:",
                np.round(np.rad2deg(base_turn_angle), 4),
            )
            print("scene base_pos_for_drive:", np.round(base_pos_for_drive, 4))
            print("scene base_target_pos:", np.round(base_target_pos, 4))
            planner.drive_base(target_pos=base_target_pos)
        planner.planner.update_from_simulation()

    planner.render_wait()

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
