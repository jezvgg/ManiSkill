import os

import gymnasium as gym
import matplotlib.pyplot as plt
import torch

from mani_skill.envs.tasks import MyRoboCasaScene

if __name__ == "__main__":
    env = gym.make(
        "MyRoboCasa-v1",
        num_envs=1,
        render_mode="rgb",
        obs_mode="rgb",
        robot_uids="ds_fetch",
        control_mode="pd_joint_pos",
    )

    unwenv: MyRoboCasaScene = env.unwrapped

    for i in range(1, 17):
        plt.subplot(4, 4, i)
        obs, _ = env.reset(options={"reconfigure": True})
        plt.imshow(obs["sensor_data"]["base_camera"]["rgb"][0].detach().cpu())

    plt.show()
