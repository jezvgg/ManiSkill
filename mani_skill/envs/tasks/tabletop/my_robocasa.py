import os
import gymnasium as gym
import numpy as np
import sapien
import torch

from mani_skill import ASSET_DIR
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.geometry.rotation_conversions import axis_angle_to_quaternion
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.robocasa.scene_builder import RoboCasaSceneBuilder
from mani_skill.utils.structs import Actor, Pose


def get_actor_size(actor: Actor):
    bodies = np.array([body.get_global_aabb_fast() for body in actor._bodies])
    return (bodies.max(axis=1) - bodies.min(axis=1))[0]


def degree_to_quaternion(x: int = 0, y: int = 0, z: int = 0):
    return axis_angle_to_quaternion(
        torch.Tensor([x * torch.pi / 180, y * torch.pi / 180, z * torch.pi / 180])
    )


@register_env("MyRoboCasa-v1", asset_download_ids=["RoboCasa"])
class MyRoboCasaScene(BaseEnv):
    SUPPORTED_ROBOTS = ["fetch", "none"]
    SUPPORTED_REWARD_MODES = ["none"]

    main_counter: object
    bowl_pos: tuple[int, int, int]
    cup_pos: tuple[int, int, int]

    camera_pos: tuple[int, int, int]

    cup: Actor
    bowl: Actor

    def __init__(self, robot_uids="fetch", *args, **kwargs):
        super().__init__(robot_uids=robot_uids, *args, **kwargs)

    def _load_agent(self, options: dict):
        # Set a safe initial pose far from the center to prevent default init collision
        safe_pose = sapien.Pose(p=[0.0, -3.0, 0.0])
        super()._load_agent(options, initial_agent_poses=[safe_pose])

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        # Initialize scene builder centered on the main counter and check for build_config_idxs override
        build_config_idxs = options.get("build_config_idxs", None) if options else None
        self.scene_builder = RoboCasaSceneBuilder(self, init_robot_base_pos="counter_main")
        self.scene_builder.build(build_config_idxs=build_config_idxs)

        fixtures = self.scene_builder.scene_data[0]["fixtures"]
        self.main_counter = self.scene_builder.get_fixture(fixtures, "counter_main")
        
        self.counter_pos = self.main_counter.pos
        self.counter_size = self.main_counter.size

        # Create counter-aligned local coordinate frame
        counter_pose = sapien.Pose(p=self.counter_pos, q=self.main_counter.quat)

        # Load bowl and cup assets
        bowl_path = os.path.join(
            ASSET_DIR,
            "scene_datasets/robocasa_dataset/assets/objects/objaverse/bowl/bowl_2/model.xml",
        )
        cup_path = os.path.join(
            ASSET_DIR,
            "scene_datasets/robocasa_dataset/assets/objects/objaverse/cup/cup_2/model.xml",
        )
        loader = self.scene.create_mjcf_loader()
        loader.visual_groups = [1]

        # Parse and build bowl (set approximate initial pose to avoid warning)
        builder = loader.parse(str(bowl_path), package_dir=os.path.dirname(bowl_path))["actor_builders"][0]
        # We calculate an approximate height for the bowl's initial pose to prevent default [0,0,0] collision
        approx_bowl_pos = (counter_pose * sapien.Pose(p=[0.0, self.counter_size[1] / 6, self.counter_size[2] / 2 + 0.1])).p
        builder.initial_pose = sapien.Pose(p=approx_bowl_pos, q=self.main_counter.quat)
        self.bowl = builder.build(name="bowl")

        # Place bowl near the center of the main counter top (slightly towards back)
        bowl_local_pos = [0.0, self.counter_size[1] / 6, self.counter_size[2] / 2 + get_actor_size(self.bowl)[2] / 2]
        self.bowl_pos = (counter_pose * sapien.Pose(p=bowl_local_pos)).p
        self.bowl.set_pose(Pose.create_from_pq(p=self.bowl_pos, q=self.main_counter.quat))

        # Parse and build cup (approximate spawn 0.3m above counter, then correct)
        builder = loader.parse(str(cup_path), package_dir=os.path.dirname(cup_path))["actor_builders"][0]
        cup_spawn_local = [self.counter_size[0] / 6, self.counter_size[1] / 6, self.counter_size[2] / 2 + 0.3]
        builder.initial_pose = counter_pose * sapien.Pose(p=cup_spawn_local)
        self.cup = builder.build_dynamic(name="cup")

        # Position cup on counter top using precise mesh size
        cup_local_pos = [
            self.counter_size[0] / 6,
            self.counter_size[1] / 6,
            self.counter_size[2] / 2 + get_actor_size(self.cup)[2] / 2 + 0.02
        ]
        self.cup_pos = (counter_pose * sapien.Pose(p=cup_local_pos)).p
        cup_pose = Pose.create_from_pq(p=self.cup_pos, q=self.main_counter.quat)
        self.cup.initial_pose = cup_pose
        self.cup.set_pose(cup_pose)

        # Place camera facing the center of the counter top
        camera_local = [self.counter_size[0] / 2, 0.0, self.counter_size[2]]
        self.camera_pos = (counter_pose * sapien.Pose(p=camera_local)).p

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        # Set initial tuck qpos to prevent start collisions for ds_fetch or fetch
        if "rest" in self.agent.keyframes:
            self.agent.robot.set_qpos(self.agent.keyframes["rest"].qpos)
        # Manually orient and position the robot base relative to the main counter and cup
        counter_pose = sapien.Pose(p=self.counter_pos, q=self.main_counter.quat)
        robot_local_pos = [
            self.counter_size[0] / 12, # Center between bowl (0.0) and cup (counter_size[0]/6)
            -self.counter_size[1] * 1.30,  # 1.30m back (0.845m from center) for perfect workspace reach and cabinet clearance
            0.0
        ]
        agent_pos = (counter_pose * sapien.Pose(p=robot_local_pos)).p
        agent_pos[2] = 0.0
        agent_quat = (counter_pose * sapien.Pose(q=degree_to_quaternion(z=90))).q
        agent_pose = Pose.create_from_pq(p=agent_pos, q=agent_quat)
        self.agent.robot.set_pose(agent_pose)

    def evaluate(self):
        bowl_radius = get_actor_size(self.bowl)[0] / 2
        
        # Calculate horizontal (XY) distance between the geometric center of the bowl and cup
        diff_xy = self.cup.pose.p[:, :2] - self.bowl.pose.p[:, :2]
        dist_xy = torch.linalg.norm(diff_xy, dim=1)
        
        # Calculate vertical (Z) distance: cup should be sitting inside the bowl
        # i.e., cup bottom Z should be at or slightly below/above the bowl top Z.
        dist_z = self.cup.pose.p[:, 2] - self.bowl.pose.p[:, 2]
        
        # We check that the robot is not grasping the cup, the cup is horizontally close to the bowl center,
        # and vertically it is placed inside the bowl.
        is_xy_close = dist_xy <= bowl_radius
        is_z_inside = (dist_z > -0.05) & (dist_z < 0.15)
        is_not_grasped = ~self.agent.is_grasping(self.cup)
        
        success = is_xy_close & is_z_inside & is_not_grasped
        return dict(success=success)

    def compute_dense_reward(self, obs: object = None, action: torch.Tensor = None, info: dict = None):
        bowl_size = get_actor_size(self.bowl)
        cup_size = get_actor_size(self.cup)

        tcp_to_obj_dist = torch.linalg.norm(
            self.cup.pose.p - self.agent.tcp_pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(6 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = self.agent.is_grasping(self.cup)
        reward += is_grasped

        upper_bound = self.bowl.pose.p + bowl_size[2] / 2 + cup_size[2] / 2
        reward += upper_bound

        bowl_radius = get_actor_size(self.bowl)[0] / 2
        xy_distance = torch.linalg.norm(self.bowl.pose.p - self.cup.pose.p, dim=1)
        place_reward = 1 - torch.tanh(6 * xy_distance)
        reward += place_reward * is_grasped * (upper_bound | xy_distance < bowl_radius)

        z_distance = self.bowl.pose.p[0][2] - self.cup.pose.p[0][2]
        place_into_reward = 1 - torch.tanh(6 * z_distance)
        reward += place_into_reward * is_grasped * xy_distance < bowl_radius

        z_tres = self.bowl.pose.p[0][2] <= self.cup.pose.p[0][2]
        reward += z_tres & xy_distance < bowl_radius

        return reward

    def compute_normalized_dense_reward(
        self, obs: object, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 6

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(self.camera_pos, self.bowl_pos)
        return [
            CameraConfig("base_camera", pose, 128, 128, 60 * np.pi / 180, 0.01, 100)
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(self.camera_pos, self.bowl_pos)
        return CameraConfig(
            "render_camera", pose, 512, 512, 60 * np.pi / 180, 0.01, 100
        )
