# main_grid_test.py
import os
import random

import gymnasium as gym
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

# Импортируем твой модуль с функцией planning(env, seed)
from planner import planning

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    env = gym.make(
        "MyRoboCasa_TakeItBack-v1",
        num_envs=1,
        render_mode="human",
        obs_mode="rgb",
        robot_uids="ds_fetch",
        control_mode="pd_joint_pos",
    )

    obs, info = env.reset()

    try:
        while True:
            # Отрисовываем кадр
            env.render()

            # Если нужно, можно подавать нулевые/пустые действия,
            # чтобы физический движок продолжал работать:
            # action = env.action_space.sample() * 0
            # env.step(action)

    except KeyboardInterrupt:
        print("\nЗакрываем среду...")
    finally:
        env.close()
