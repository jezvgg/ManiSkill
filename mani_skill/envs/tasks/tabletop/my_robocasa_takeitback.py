import os

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

from .my_robocasa import get_actor_size, degree_to_quanterion


@register_env("MyRoboCasa_TakeItBack-v1", asset_download_ids=["RoboCasa"])
class MyRoboCasaSceneTakeItBack(BaseEnv):
    SUPPORTED_ROBOTS = ["fetch", "none"]
    SUPPORTED_REWARD_MODES = ["none"]

    fxtr_placements: dict[str, dict[str, object]]
    cup_pos: tuple[int, int, int]
    cup_pos_sink: tuple[int, int, int]

    camera_pos: tuple[int, int, int]
    agent_pose: sapien.Pose

    cup: Actor


    def __init__(self, robot_uids="fetch", *args, **kwargs):
        super().__init__(robot_uids=robot_uids, *args, **kwargs)
        self.fxtr_placements = {}


    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self._set_episode_rng(102, torch.arange(self.num_envs))
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

        cup_path = os.path.join(
            ASSET_DIR,
            "scene_datasets/robocasa_dataset/assets/objects/objaverse/cup/cup_2/model.xml",
        )
        loader = self.scene.create_mjcf_loader()
        loader.visual_groups = [1]

        self.counter_pos = self.fixture_placements["counter_main_main_group_Counter"][
            "pos"
        ]
        self.counter_size = self.fixture_placements["counter_main_main_group_Counter"][
            "size"
        ]

        self.sink_pos = self.fixture_placements["sink_main_group_Sink"][
            "pos"
        ]
        self.sink_size = self.fixture_placements["sink_main_group_Sink"][
            "size"
        ]

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

        self.cup_pos_sink = self.sink_pos.copy()
        self.cup_pos_sink[0] -= self.sink_size[0] / 6
        self.cup_pos_sink[2] = self.cup_pos[2]
        self.cup_pos_sink[1] = self.cup_pos[1]

        self.camera_pos = self.cup_pos_sink.copy()
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

        self.agent_pose = Pose.create_from_pq(p=agent_pos, q=q)
        self.agent.robot.set_pose(self.agent_pose)


    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(self.camera_pos, self.cup_pos_sink)
        return [
            CameraConfig("base_camera", pose, 128, 128, 60 * np.pi / 180, 0.01, 100)
        ]


    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(self.camera_pos, self.cup_pos_sink)
        return CameraConfig(
            "render_camera", pose, 2048, 2048, 60 * np.pi / 180, 0.01, 100
        )
