import os
import numpy as np
import sapien
import torch
from mani_skill import ASSET_DIR
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Actor, Pose
from .base_robocasa import BaseRoboCasaScene
from utils.scene_utils import get_actor_size, degree_to_quanterion

@register_env("MyRoboCasa_TakeItBack-v1", asset_download_ids=["RoboCasa"])
class MyRoboCasaSceneTakeItBack(BaseRoboCasaScene):
    FIXTURE_SEED = 102
    fxtr_placements: dict[str, dict[str, object]]
    cup_pos: tuple[int, int, int]
    cup_pos_sink: tuple[int, int, int]

    camera_pos: tuple[int, int, int]
    agent_pose: sapien.Pose

    cup: Actor

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        cup_path = os.path.join(
            ASSET_DIR,
            "scene_datasets/robocasa_dataset/assets/objects/objaverse/cup/cup_2/model.xml",
        )
        loader = self.scene.create_mjcf_loader()
        loader.visual_groups = [1]

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
        with torch.device(self.device):
            if not hasattr(self, "placed_on_sink") or self.placed_on_sink.shape[0] != self.num_envs:
                self.placed_on_sink = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.placed_on_sink[env_idx] = False

    def evaluate(self):
        cup_pos = self.cup.pose.p
        is_grasped = self.agent.is_grasping(self.cup)

        cup_pos_sink = torch.as_tensor(self.cup_pos_sink, device=self.device)
        xy_dist_sink = torch.linalg.norm(cup_pos[:, :2] - cup_pos_sink[:2], dim=1)
        z_dist_sink = torch.abs(cup_pos[:, 2] - cup_pos_sink[2])

        is_on_sink = (xy_dist_sink <= 0.15) & (z_dist_sink <= 0.10) & (~is_grasped)
        self.placed_on_sink = self.placed_on_sink | is_on_sink

        cup_pos_init = torch.as_tensor(self.cup_pos, device=self.device)
        xy_dist_init = torch.linalg.norm(cup_pos[:, :2] - cup_pos_init[:2], dim=1)
        z_dist_init = torch.abs(cup_pos[:, 2] - cup_pos_init[2])

        returned_to_initial = (xy_dist_init <= 0.15) & (z_dist_init <= 0.10) & (~is_grasped)

        success = self.placed_on_sink & returned_to_initial
        return dict(success=success)

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
