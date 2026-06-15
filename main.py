import os
import random

import gymnasium as gym
import numpy as np
import torch

from mani_skill.envs.tasks import MyRoboCasaScene
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
    )
    env = RecordEpisode(
        env,
        output_dir=os.path.join("videos", "my_robocasa"),
        save_video=True,
        video_fps=30,
        save_on_reset=True,
    )

    unwenv: MyRoboCasaScene = env.unwrapped
