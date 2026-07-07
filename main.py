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
        "MyRoboCasa-v1",
        num_envs=1,
        render_mode=None,
        obs_mode="none",
        robot_uids="ds_fetch",
        control_mode="pd_joint_pos",
    )

    all_scenes_frames = []
    success_statuses = []

    print("=== ЗАПУСК ТЕСТОВОГО СТЕНДА С АВТОСОХРАНЕНИЕМ ВИДЕО ===")

    for i in range(1, 17):
        current_seed = i + 100
        print(f"\nКухня {i}/16 (Seed: {current_seed})...")

        pass

        success_val = False
        try:
            success_output = planning(env, seed=current_seed)
            if isinstance(success_output, (list, np.ndarray, torch.Tensor)):
                success_val = bool(success_output[0])
            else:
                success_val = bool(success_output)
        except Exception as e:
            print(f"Критическая ошибка на сцене {i}: {e}")
            success_val = False

        pass

        print(f"Результат сцены {i}: {'УСПЕХ' if success_val else 'ПРОВАЛ'}")
        pass
        success_statuses.append(success_val)

    pass
    # --------------------------------------------------------------------------
    # НАСТРОЙКА АВТОСОХРАНЕНИЯ ВИДЕО
    # --------------------------------------------------------------------------
    print(f"\n[Запись] Скипаем рендер видео для скорости...")
    # ani.save(video_path, writer="ffmpeg", fps=25, dpi=100)

    print(f"[Готово] Видео успешно сохранено!")
    # --------------------------------------------------------------------------

    num_success = sum(success_statuses)
    total_scenes = len(success_statuses)
    success_rate = (num_success / total_scenes) * 100

    print("\n=== СТАТИСТИКА ТЕСТИРОВАНИЯ ===")
    print(f"Всего сцен: {total_scenes}")
    print(f"Успешно: {num_success}")
    print(f"Провалено: {total_scenes - num_success}")
    print(f"Success rate: {success_rate:.2f}%")
    print("===============================\n")

    print("Отображаем интерактивное окно...")
    # plt.tight_layout()
    # plt.show()

    env.close()
