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
        render_mode="rgb_array",
        obs_mode="rgb",
        robot_uids="ds_fetch",
        control_mode="pd_joint_pos",
    )

    all_scenes_frames = []
    success_statuses = []

    print("=== ЗАПУСК ТЕСТОВОГО СТЕНДА С АВТОСОХРАНЕНИЕМ ВИДЕО ===")

    for i in range(1, 17):
        current_seed = i + 100
        print(f"\nКухня {i}/16 (Seed: {current_seed})...")

        scene_frames = []

        original_reset = env.reset

        def custom_reset(*args, **kwargs):
            obs, info = original_reset(*args, **kwargs)
            frame = obs["sensor_data"]["base_camera"]["rgb"][0].detach().cpu().numpy()
            scene_frames.append(frame)
            return obs, info

        original_step = env.step

        def custom_step(action):
            obs, r, t, tr, info = original_step(action)
            frame = obs["sensor_data"]["base_camera"]["rgb"][0].detach().cpu().numpy()
            scene_frames.append(frame)
            return obs, r, t, tr, info

        env.reset = custom_reset
        env.step = custom_step

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

        env.reset = original_reset
        env.step = original_step

        print(f"Результат сцены {i}: {'УСПЕХ' if success_val else 'ПРОВАЛ'}")
        all_scenes_frames.append(scene_frames)
        success_statuses.append(success_val)

    # Наращивание кадров (Padding)
    max_len = max(len(frames) for frames in all_scenes_frames)
    for i in range(16):
        while len(all_scenes_frames[i]) < max_len:
            all_scenes_frames[i].append(all_scenes_frames[i][-1])

    # Сборка сетки
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    im_objects = []

    for i in range(16):
        ax = axes[i]
        im = ax.imshow(all_scenes_frames[i][0])
        ax.axis("off")

        title_color = "green" if success_statuses[i] else "red"
        status_text = "SUCCESS" if success_statuses[i] else "FAILED"
        ax.set_title(f"Scene {i + 1}: {status_text}", color=title_color, fontsize=10)
        im_objects.append(im)

    def update_grid(frame_idx):
        for i in range(16):
            im_objects[i].set_data(all_scenes_frames[i][frame_idx])
        return im_objects

    ani = animation.FuncAnimation(
        fig, update_grid, frames=max_len, interval=40, blit=True, repeat=True
    )

    # --------------------------------------------------------------------------
    # НАСТРОЙКА АВТОСОХРАНЕНИЯ ВИДЕО
    # --------------------------------------------------------------------------
    output_dir = os.path.join("videos", "my_robocasa")
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, "grid_validation.mp4")

    print(f"\n[Запись] Сборка кадров и кодирование видео в {video_path}...")

    # Использованием ffmpeg для сборки MP4 (25 кадров в секунду)
    # Если хочешь гифку, замени расширение на .gif и убавь fps до 15-20, чтоб файл не весил гигабайт
    ani.save(video_path, writer="ffmpeg", fps=25, dpi=100)

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
