import numpy as np
import gymnasium as gym
import mani_skill  # noqa: F401  (must init before my_scenes; tabletop imports my_scenes)
import my_scenes  # noqa: F401  (registers envs)
from my_scenes.my_robocasa_fridge_picture import MyRoboCasaFridgePicture

MyRoboCasaFridgePicture.PICTURE_PATH = "my_scenes/area_check.png"

env = gym.make(
    "MyRoboCasa_FridgePicture-v1",
    num_envs=1,
    obs_mode="state",
    reward_mode="none",
)
obs, _ = env.reset(seed=0)
scene = env.unwrapped

# robot in front of fridge door, facing it
robot_pos = scene.agent.robot.pose.p[0].cpu().numpy()
robot_q = scene.agent.robot.pose.q[0].cpu().numpy()  # (w, x, y, z)
forward = np.array(
    [
        1 - 2 * (robot_q[2] ** 2 + robot_q[3] ** 2),
        2 * (robot_q[1] * robot_q[2] + robot_q[0] * robot_q[3]),
    ]
)
fridge_front = scene.picture_center - scene.robot_spawn_pos
fridge_front = fridge_front[:2] / np.linalg.norm(fridge_front[:2])
print("robot pos:", robot_pos[:2])
print("picture center:", scene.picture_center)
print("forward dot fridge_front:", float(forward @ fridge_front))
assert float(forward @ fridge_front) > 0.9, "robot not facing the fridge"
dist = np.linalg.norm(robot_pos[:2] - scene.robot_spawn_pos[:2])
print("spawn position error:", dist)
assert dist < 1e-3, "robot not at spawn position"

# picture exists, visible, placed on the door
assert scene.picture is not None, "picture not built"
assert not scene.picture.hidden, "picture hidden at start"
pic_pos = scene.picture.pose.p[0].cpu().numpy()
print("picture pose:", pic_pos)
off = np.linalg.norm(pic_pos - scene.picture_center)
print("picture offset from door center:", off)
assert off < 0.01, "picture not on door face"

# visible before step 100, hidden at step 100
env.step(env.action_space.sample())
assert not scene.picture.hidden
for _ in range(98):
    env.step(env.action_space.sample())
assert not scene.picture.hidden, "picture hidden too early"
env.step(env.action_space.sample())
assert scene.picture.hidden, "picture not hidden at step 100"
print("hidden at step 100 OK")

# reset restores visibility
env.reset(seed=0)
assert not scene.picture.hidden, "picture not restored after reset"
print("restored after reset OK")

print("SMOKE PASS")
