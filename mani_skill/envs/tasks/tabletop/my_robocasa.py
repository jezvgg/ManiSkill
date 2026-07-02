import os

import gymnasium as gym
import numpy as np
import sapien
import sapien.physx as physx
import torch

from mani_skill import ASSET_DIR
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.geometry.rotation_conversions import axis_angle_to_quaternion
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.robocasa.scene_builder import RoboCasaSceneBuilder
from mani_skill.utils.structs import Actor, Pose
from mani_skill.utils.wrappers.record import RecordEpisode


def get_actor_size(actor: Actor):
    bodies = np.array([body.get_global_aabb_fast() for body in actor._bodies])
    return (bodies.max(axis=1) - bodies.min(axis=1))[0]


def degree_to_quanterion(x: int = 0, y: int = 0, z: int = 0):
    return axis_angle_to_quaternion(
        torch.Tensor([x * torch.pi / 180, y * torch.pi / 180, z * torch.pi / 180])
    )


@register_env("MyRoboCasa-v1", asset_download_ids=["RoboCasa"])
class MyRoboCasaScene(BaseEnv):
    SUPPORTED_ROBOTS = ["fetch", "none"]
    SUPPORTED_REWARD_MODES = ["none"]

    fxtr_placements: dict[str, dict[str, object]]
    bowl_pos: tuple[int, int, int]
    cup_pos: tuple[int, int, int]

    camera_pos: tuple[int, int, int]
    agent_pose: sapien.Pose

    cup: Actor
    bowl: Actor

    def __init__(self, robot_uids="fetch", *args, **kwargs):
        super().__init__(robot_uids=robot_uids, *args, **kwargs)
        self.fxtr_placements = {}

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.scene_builder = RoboCasaSceneBuilder(self)
        self.scene_builder.build()
        self.fixture_placements = {
            config["name"]
            + "_"
            + getattr(
                getattr(config["model"], "__class__", None), "__name__", "Wrong"
            ): {
                "pos": getattr(config["model"], "pos", None),
                "quat": getattr(config["model"], "quat", None),
                "size": getattr(config["model"], "size", None),
            }
            for config in self.scene_builder.scene_data[0].get("fixture_cfgs")
        }

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

        builder = loader.parse(str(bowl_path), package_dir=os.path.dirname(bowl_path))[
            "actor_builders"
        ][0]
        self.bowl = builder.build(name="bowl")

        self.counter_pos = self.fixture_placements["counter_main_main_group_Counter"][
            "pos"
        ]
        self.counter_size = self.fixture_placements["counter_main_main_group_Counter"][
            "size"
        ]

        self.bowl_pos = self.counter_pos.copy()
        self.bowl_pos[0] += self.counter_size[0] / 3
        self.bowl_pos[1] -= self.counter_size[1] / 4
        self.bowl_pos[2] += self.counter_size[2] / 2 + get_actor_size(self.bowl)[2] / 2
        self.bowl.set_pose(Pose.create_from_pq(p=self.bowl_pos))

        builder = loader.parse(str(cup_path), package_dir=os.path.dirname(cup_path))[
            "actor_builders"
        ][0]
        cup_initial_pos = self.counter_pos.copy()
        cup_initial_pos[0] += self.counter_size[0] / 6
        cup_initial_pos[2] += self.counter_size[2] / 2 + 0.3
        cup_initial_pos[1] -= self.counter_size[1] / 4
        builder.initial_pose = sapien.Pose(p=cup_initial_pos)
        self.cup = builder.build_dynamic(name="cup")

        self.cup_pos = self.counter_pos.copy()
        self.cup_pos[2] += (
            self.counter_size[2] / 2 + get_actor_size(self.cup)[2] / 2 + 0.02
        )
        self.cup_pos[0] += self.counter_size[0] / 6
        self.cup_pos[1] -= self.counter_size[1] / 4
        cup_pose = Pose.create_from_pq(p=self.cup_pos)
        self.cup.initial_pose = cup_pose
        self.cup.set_pose(cup_pose)

        self.camera_pos = self.bowl_pos.copy()
        self.camera_pos[0] -= self.counter_size[0] / 4
        self.camera_pos[1] -= self.counter_size[1] / 2
        self.camera_pos[2] += self.counter_size[2] / 2

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        agent_pos = self.agent.robot.pose.p[0]
        agent_pos[0] = self.cup_pos[0]
        agent_pos[1] = self.cup_pos[1]
        agent_pos[1] -= self.counter_size[1] * 1.6

        q = degree_to_quanterion(z=180)

        agent_pose = Pose.create_from_pq(p=agent_pos, q=q)
        self.agent.robot.set_pose(agent_pose)

    def evaluate(self):
        """
        Сделаю чекер, что робот разместил нижнюю часть кружки, в
        центральную позицию чашки. По идее это значит, что кружка в чашке.
        """
        bowl_radius = get_actor_size(self.bowl)[0] / 2
        xy_distance = (
            torch.linalg.norm(self.bowl.pose.p - self.cup.pose.p, dim=1) <= bowl_radius
        )
        z_distance = self.bowl.pose.p[0][2] >= self.cup.pose.p[0][2]
        return dict(success=xy_distance & z_distance)

    def compute_dense_reward(self):
        """
        Награда состоит из расстояния до кружки, из того, взял ли он кружку, из расстояния от кружки до чашки.
        """
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
            "render_camera", pose, 2048, 2048, 60 * np.pi / 180, 0.01, 100
        )
