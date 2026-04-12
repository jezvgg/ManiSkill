import os
import gymnasium as gym
import numpy as np
import sapien
import torch
from mani_skill.examples.motionplanning.fetch.extand import (
    FetchMotionPlanningSapienSolver,
)
from mani_skill.examples.motionplanning.fetch.utils import get_fcl_object_name
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.utils.structs import Pose


def run_motion_planning(env: gym.Env):
    """Запускает motion planning для перемещения чашки в миску."""
    print("Creating planner...")
    env_unwrapped = env.unwrapped
    # Используем FetchMotionPlanningSapienSolver
    planner = FetchMotionPlanningSapienSolver(
        env_unwrapped,
        base_pose=env_unwrapped.agent.robot.pose,
        vis=True,
        print_env_info=True,
        verbose=True,
    )
    print("Planner created!")

    robot_name = env_unwrapped.agent.robot.name

    cup = env_unwrapped.cup
    bowl = env_unwrapped.bowl

    # Настройка ACM (Allowed Collision Matrix) для избежания коллизий
    planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
        f"scene-0-{robot_name}_r_gripper_finger_link", get_fcl_object_name(cup), True
    )
    planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
        f"scene-0-{robot_name}_l_gripper_finger_link", get_fcl_object_name(cup), True
    )
    planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
        f"scene-0-{robot_name}_wrist_flex_link", get_fcl_object_name(cup), True
    )
    planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
        f"scene-0-{robot_name}_wrist_roll_link", get_fcl_object_name(cup), True
    )
    planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
        f"scene-0-{robot_name}_upperarm_roll_link",
        "scene-0_counter_1_front_group_0_119",
        True,
    )
    planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
        f"scene-0-{robot_name}_elbow_flex_link",
        "scene-0_counter_1_front_group_0_119",
        True,
    )
    planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
        f"scene-0-{robot_name}_forearm_roll_link",
        "scene-0_counter_1_front_group_0_119",
        True,
    )
    planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
        f"scene-0-{robot_name}_wrist_flex_link",
        "scene-0_counter_1_front_group_0_119",
        True,
    )

    cup_pos = cup.pose.p[0]
    cup_quat = cup.pose.q[0]
    bowl_pos = bowl.pose.p[0]
    bowl_quat = bowl.pose.q[0]

    # Позиции для манипуляций
    approach_height = 0.25  # Увеличиваем высоту подъезда
    grasp_height = 0.05  # Поднимаем высоту захвата, чтобы не задеть стол

    approach_pose = sapien.Pose(
        p=np.array(
            [cup_pos[0], cup_pos[1], cup_pos[2] + approach_height], dtype=np.float32
        ),
        q=np.array(cup_quat, dtype=np.float32),
    )

    grasp_pose = sapien.Pose(
        p=np.array(
            [cup_pos[0], cup_pos[1], cup_pos[2] + grasp_height], dtype=np.float32
        ),
        q=np.array(cup_quat, dtype=np.float32),
    )

    above_bowl_pose = sapien.Pose(
        p=np.array(
            [bowl_pos[0], bowl_pos[1], bowl_pos[2] + approach_height], dtype=np.float32
        ),
        q=np.array(bowl_quat, dtype=np.float32),
    )

    # 1. Подъезд и захват
    print("=== Moving to cup ===")
    planner.static_manipulation(approach_pose, n_init_qpos=100)
    planner.static_manipulation(grasp_pose, n_init_qpos=100)

    print("=== Grasping cup ===")
    planner.close_gripper(t=10)

    # Прикрепление чашки
    # get_fcl_object_name возвращает имя, которое планировщик использует во внутренней мапе.
    # Проверим, что это имя есть в планировщике, используя имя объекта напрямую, если нужно.
    cup_name = get_fcl_object_name(cup)
    print(f"Attaching object: {cup_name}")

    # Можно попробовать найти объект по имени, если attach_object кидает IndexError
    # Возможно, нам нужно использовать имя объекта из env.unwrapped

    kwargs = {
        "name": cup_name,
        "art_name": f"scene-0_{robot_name}",
        "link_id": planner.planner.move_group_link_id,
    }
    try:
        planner.planner.planning_world.attach_object(**kwargs)
    except IndexError:
        print(f"Could not attach object {cup_name}. Skipping attachment.")

    # 2. Перемещение к миске
    print("=== Moving to bowl ===")
    planner.static_manipulation(above_bowl_pose, n_init_qpos=100)

    # 3. Опускание
    print("=== Placing cup ===")
    planner.static_manipulation(
        sapien.Pose(p=bowl_pos + np.array([0, 0, 0.05]), q=bowl_quat), n_init_qpos=100
    )
    planner.open_gripper(t=10)

    print("Motion planning completed!")


if __name__ == "__main__":
    env = gym.make(
        "MyRoboCasa-v1",
        num_envs=1,
        render_mode="human",
        robot_uids="ds_fetch",
        control_mode="pd_joint_pos",
    )
    env = RecordEpisode(
        env,
        output_dir=os.path.join("videos", "my_robocasa"),
        save_video=True,
        video_fps=30,
        save_on_reset=False,
    )
    env.reset(options={"reconfigure": True})
    run_motion_planning(env)
