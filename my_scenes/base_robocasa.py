import numpy as np
import sapien
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.scene_builder.robocasa.scene_builder import RoboCasaSceneBuilder
from mani_skill.utils.structs import Pose
from utils.scene_utils import degree_to_quanterion

class BaseRoboCasaScene(BaseEnv):
    SUPPORTED_ROBOTS = ["fetch", "none"]
    SUPPORTED_REWARD_MODES = ["none"]
    FIXTURE_SEED: int = None

    fixture_placements: dict[str, dict[str, object]]
    counter_pos: np.ndarray
    counter_size: np.ndarray
    cup_pos: np.ndarray

    def __init__(self, robot_uids="fetch", *args, **kwargs):
        super().__init__(robot_uids=robot_uids, *args, **kwargs)
        self.fixture_placements = {}

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        if self.FIXTURE_SEED is not None:
            self._set_episode_rng(self.FIXTURE_SEED, torch.arange(self.num_envs, device=self.device))
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

        self.counter_pos = self.fixture_placements["counter_main_main_group_Counter"][
            "pos"
        ]
        self.counter_size = self.fixture_placements["counter_main_main_group_Counter"][
            "size"
        ]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        agent_pos = self.agent.robot.pose.p[0]
        agent_pos[0] = self.cup_pos[0]
        agent_pos[1] = self.cup_pos[1]
        agent_pos[1] -= self.counter_size[1] * 1.6

        q = degree_to_quanterion(z=180)

        self.agent_pose = Pose.create_from_pq(p=agent_pos, q=q)
        self.agent.robot.set_pose(self.agent_pose)
