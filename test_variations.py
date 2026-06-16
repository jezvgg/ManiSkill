import os
if "VK_ICD_FILENAMES" not in os.environ and os.path.exists("/home/jezv/vulkan_user/lvp_icd_user.json"):
    os.environ["VK_ICD_FILENAMES"] = "/home/jezv/vulkan_user/lvp_icd_user.json"

import gymnasium as gym
import numpy as np
import torch
import imageio
import mani_skill.envs.tasks.tabletop.my_robocasa

os.makedirs("videos/my_robocasa", exist_ok=True)

print("Starting variations test verification (10 configs)...")
# 10 configurations across different layouts / designs
configs = [0, 12, 24, 36, 48, 60, 72, 84, 96, 108]
for i, config_idx in enumerate(configs):
    print(f"\n--- VARIATION {i} (Config Override: {config_idx}) ---")
    env = gym.make(
        "MyRoboCasa-v1",
        num_envs=1,
        obs_mode="state",
        render_mode="rgb_array",
        render_backend="pci:0000:00:00.0"
    )
    env.reset(seed=i, options={"reconfigure": True, "build_config_idxs": [config_idx]})
    unw = env.unwrapped
    
    # Print locations
    print(f"Main Counter Pos: {unw.main_counter.pos}")
    print(f"Main Counter Size: {unw.main_counter.size}")
    print(f"Bowl Pos: {unw.bowl.pose.p[0].cpu().numpy()}")
    print(f"Cup Pos: {unw.cup.pose.p[0].cpu().numpy()}")
    print(f"Robot Base Pos: {unw.agent.robot.pose.p[0].cpu().numpy()}")
    
    # Render and save frame
    img = env.render()
    img_np = img[0].cpu().numpy()
    out_path = f"videos/my_robocasa/variation_{i}.png"
    imageio.imwrite(out_path, img_np)
    print(f"Saved screenshot: {out_path}")
    env.close()

print("Variations test verification finished!")
