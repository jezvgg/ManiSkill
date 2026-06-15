import os
import gymnasium as gym
import numpy as np
import torch
import imageio
import mani_skill.envs.tasks.tabletop.my_robocasa

os.makedirs("videos/my_robocasa", exist_ok=True)

print("Starting variations test verification...")
for i in range(5):
    print(f"\n--- VARIATION {i} ---")
    env = gym.make(
        "MyRoboCasa-v1",
        num_envs=1,
        obs_mode="state",
        render_mode="rgb_array",
        render_backend="pci:0000:00:00.0"
    )
    env.reset(seed=i)
    unw = env.unwrapped
    
    # Print locations
    print(f"Main Counter Pos: {unw.main_counter.pos}")
    print(f"Main Counter Size: {unw.main_counter.size}")
    print(f"Bowl Pos: {unw.bowl.pose.p[0].cpu().numpy()}")
    print(f"Cup Pos: {unw.cup.pose.p[0].cpu().numpy()}")
    print(f"Robot Base Pos: {unw.agent.robot.pose.p[0].cpu().numpy()}")
    print(f"Robot Base Quat: {unw.agent.robot.pose.q[0].cpu().numpy()}")
    
    # Verify positions on main counter
    # Y-axis bounds of main counter: main_counter.pos[1] - size[1]/2 to main_counter.pos[1] + size[1]/2
    # Since robot is 1.6m away, etc.
    
    # Render and save frame
    img = env.render()
    img_np = img[0].cpu().numpy()
    out_path = f"videos/my_robocasa/variation_{i}.png"
    imageio.imwrite(out_path, img_np)
    print(f"Saved screenshot: {out_path}")
    env.close()

print("Variations test verification finished!")
