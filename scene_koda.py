import os

import gymnasium as gym
import numpy as np
import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.robocasa.scene_builder import RoboCasaSceneBuilder
from mani_skill.utils.building import actors
from mani_skill.utils.structs import Pose, Actor
from mani_skill.utils import sapien_utils
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.utils.geometry.rotation_conversions import axis_angle_to_quaternion
from mani_skill.examples.motionplanning.fetch.extand import FetchMotionPlanningSapienSolver
from mani_skill import ASSET_DIR
import sapien.physx as physx


def get_actor_size(actor: Actor):
    bodies = np.array([body.get_global_aabb_fast() for body in actor._bodies])
    return (bodies.max(axis=1) - bodies.min(axis=1))[0]


def degree_to_quanterion(x: int = 0, y: int = 0, z: int = 0): 
    return axis_angle_to_quaternion(
        torch.Tensor([
            x * torch.pi / 180, 
            y * torch.pi / 180,
            z * torch.pi / 180
            ]))



@register_env(
    "MyRoboCasa-v1", asset_download_ids=['RoboCasa']
)
class MyRoboCasaScene(BaseEnv):
    SUPPORTED_ROBOTS = ["ds_fetch", "none"]
    SUPPORTED_REWARD_MODES = ["none"]

    fxtr_placements: dict[str, dict[str,object]]
    bowl_pos: tuple[int, int, int]
    cup_pos: tuple[int, int, int]

    camera_pos: tuple[int, int, int]
    agent_pose: sapien.Pose

    cup: Actor
    bowl: Actor
    

    def __init__(self, robot_uids='ds_fetch', *args, **kwargs): 
        super().__init__(robot_uids=robot_uids, *args, **kwargs)
        self.fxtr_placements = {}

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.scene_builder = RoboCasaSceneBuilder(self)
        self.scene_builder.build()
        self.fixture_placements = {config['name'] + '_' + getattr(
                                                        getattr(config['model'], '__class__', None),
                                                        '__name__', 'Wrong'): 
                                    {'pos':getattr(config['model'], 'pos', None), 
                                     'quat':getattr(config['model'], 'quat', None),
                                     'size':getattr(config['model'], 'size', None)}
                                for config in self.scene_builder.scene_data[0].get('fixture_cfgs')}
        for name, cfg in self.fixture_placements.items():
            print(f"{name}:pq({cfg['pos']}, {cfg['quat']}), size({cfg['size']})")

        bowl_path = os.path.join(ASSET_DIR, "scene_datasets/robocasa_dataset/assets/objects/objaverse/bowl/bowl_2/model.xml")
        cup_path = os.path.join(ASSET_DIR, "scene_datasets/robocasa_dataset/assets/objects/objaverse/cup/cup_2/model.xml")
        loader = self.scene.create_mjcf_loader()
        loader.visual_groups = [1]
        
        builder = loader.parse(str(bowl_path), package_dir=os.path.dirname(bowl_path))['actor_builders'][0]
        self.bowl = builder.build(name="bowl")

        self.counter_pos = self.fixture_placements['counter_1_front_group_Counter']['pos']
        self.counter_size = self.fixture_placements['counter_1_front_group_Counter']['size']

        self.bowl_pos = self.counter_pos.copy()
        self.bowl_pos[2] += self.counter_size[2] / 2 + get_actor_size(self.bowl)[2] / 2
        self.bowl.set_pose(Pose.create_from_pq(p=self.bowl_pos))

        builder = loader.parse(str(cup_path), package_dir=os.path.dirname(cup_path))['actor_builders'][0]
        self.cup = builder.build(name="cup")

        self.cup_pos = self.counter_pos.copy()
        self.cup_pos[2] += self.counter_size[2] / 2 + get_actor_size(self.cup)[2] / 2
        self.cup_pos[0] += self.counter_size[0] / 6
        self.cup.set_pose(Pose.create_from_pq(p=self.cup_pos))

        self.camera_pos = self.bowl_pos.copy()
        self.camera_pos[0] += (self.counter_size[0] / 2)
        self.camera_pos[2] += self.counter_size[2]


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        agent_pos = self.agent.robot.pose.p[0]
        agent_pos[0] = self.counter_pos[0]
        agent_pos[1] = self.counter_pos[1]
        agent_pos[1] += self.counter_size[1] * 1.5

        q = degree_to_quanterion(z=180)

        agent_pose = Pose.create_from_pq(p=agent_pos, q=q)
        self.agent.robot.set_pose(agent_pose)


    def evaluate(self):
        '''
        Сделаю чекер, что робот разместил нижнюю часть кружки, в
        центральную позицию чашки. По идее это значит, что кружка в чашке.
        '''
        bowl_radius = get_actor_size(self.bowl)[0] / 2
        xy_distance = (torch.linalg.norm(self.bowl.pose.p - self.cup.pose.p, dim=1) <= bowl_radius)
        z_distance = (self.bowl.pose.p[0][2] >= self.cup.pose.p[0][2])
        return dict(success = xy_distance & z_distance)


    def compute_dense_reward(self):
        '''
        Награда состоит из расстояния до кружки, из того, взял ли он кружку, из расстояния от кружки до чашки.
        '''
        bowl_size = get_actor_size(self.bowl)
        cup_size = get_actor_size(self.cup)

        tcp_to_obj_dist = torch.linalg.norm(
            self.cup.pose.p - self.agent.tcp_pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(6 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = self.agent.is_grasping(self.cup)
        reward += is_grasped

        upper_bound = self.bowl.pose.p + bowl_size[2] / 2 + cup_size[2] / 2
        reward += upper_bound

        bowl_radius = get_actor_size(self.bowl)[0] / 2
        xy_distance = torch.linalg.norm(self.bowl.pose.p - self.cup.pose.p, dim=1)
        place_reward = 1 - torch.tanh(6 * xy_distance)
        reward += place_reward * is_grasped * (upper_bound | xy_distance < bowl_radius)

        z_distance = self.bowl.pose.p[0][2] - self.cup.pose.p[0][2]
        place_into_reward = 1 - torch.tanh(6 * z_distance)
        reward += place_into_reward * is_grasped * xy_distance < bowl_radius

        z_tres = self.bowl.pose.p[0][2] <= self.cup.pose.p[0][2]
        reward += z_tres & xy_distance < bowl_radius

        return reward

    def compute_normalized_dense_reward(
        self, obs: object, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 6

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(self.camera_pos, self.bowl_pos)
        return [
            CameraConfig("base_camera", pose, 128, 128, 60 * np.pi / 180, 0.01, 100)
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(self.camera_pos, self.bowl_pos)
        return CameraConfig(
            "render_camera", pose, 2048, 2048, 60 * np.pi / 180, 0.01, 100
        )


# =============================================================================
# Motion Planning для Koda
# =============================================================================


def run_motion_planning(env: gym.Env):
    """Запускает motion planning для перемещения чашки в миску."""
    from mani_skill.examples.motionplanning.fetch.utils import get_fcl_object_name
    
    print("Creating planner...")
    env_unwrapped = env.unwrapped
    planner = FetchMotionPlanningSapienSolver(
        env,
        base_pose=env_unwrapped.agent.robot.pose,
        vis=True,
        print_env_info=True,
        verbose=True
    )
    print("Planner created!")

    # Настройка ACM (Allowed Collision Matrix) для избежания коллизий с окружением
    robot_name = env_unwrapped.agent.robot.name
    
    # Разрешаем коллизии между колесами робота и полом
    planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
        f'scene-0-{robot_name}_r_wheel_link', 'scene-0_ground_31', True
    )
    planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
        f'scene-0-{robot_name}_l_wheel_link', 'scene-0_ground_31', True
    )
    # Разрешаем коллизии между базой робота и столешницей
    planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
        f'scene-0-{robot_name}_base_link', 'scene-0_table-workspace_30', True
    )
    planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
        f'scene-0-{robot_name}_laser_link', 'scene-0_table-workspace_30', True
    )

    cup = env_unwrapped.cup
    bowl = env_unwrapped.bowl
    
    cup_pos = np.array(cup.pose.sp.p, dtype=np.float32)
    cup_quat = cup.pose.sp.q
    bowl_pos = np.array(bowl.pose.sp.p, dtype=np.float32)
    bowl_quat = bowl.pose.sp.q
    
    # Вычисляем позиции для grasp
    # Приближаемся к чашке сверху
    approach_height = 0.15
    grasp_height = 0.02
    
    # Позиция над чашкой для подъезда
    approach_pose = sapien.Pose(
        p=np.array([cup_pos[0], cup_pos[1], cup_pos[2] + approach_height], dtype=np.float32),
        q=np.array(cup_quat, dtype=np.float32)
    )
    
    # Позиция для захвата
    grasp_pose = sapien.Pose(
        p=np.array([cup_pos[0], cup_pos[1], cup_pos[2] + grasp_height], dtype=np.float32),
        q=np.array(cup_quat, dtype=np.float32)
    )
    
    # Позиция подъема чашки
    lift_pose = sapien.Pose(
        p=np.array([cup_pos[0], cup_pos[1], cup_pos[2] + approach_height], dtype=np.float32),
        q=np.array(cup_quat, dtype=np.float32)
    )
    
    # Позиция над миской
    above_bowl_pose = sapien.Pose(
        p=np.array([bowl_pos[0], bowl_pos[1], bowl_pos[2] + approach_height], dtype=np.float32),
        q=np.array(bowl_quat, dtype=np.float32)
    )
    
    # Позиция опускания в миску
    place_pose = sapien.Pose(
        p=np.array([bowl_pos[0], bowl_pos[1], bowl_pos[2] + 0.05], dtype=np.float32),
        q=np.array(bowl_quat, dtype=np.float32)
    )

    tcp_quat = env_unwrapped.agent.tcp.pose.sp.q
    
    # =============================================================================
    # Фаза 1: Подъезд к чашке (используем drive_base + static_manipulation)
    # =============================================================================
    print("=== Phase 1: Driving to cup ===")
    
    # Вычисляем позицию для подъезда (сзади от чашки)
    drive_pos = sapien.Pose(
        p=np.array([cup_pos[0] - 0.5, cup_pos[1], cup_pos[2]], dtype=np.float32),
        q=np.array(tcp_quat, dtype=np.float32)
    )
    planner.drive_base(drive_pos, approach_pose)
    planner.planner.update_from_simulation()
    
    # Подъезжаем к позиции над чашкой
    print("Moving to approach pose above cup...")
    planner.static_manipulation(approach_pose, n_init_qpos=100)
    planner.planner.update_from_simulation()
    
    # =============================================================================
    # Фаза 2: Захват чашки
    # =============================================================================
    print("=== Phase 2: Grasping cup ===")
    
    # Опускаемся к чашке
    print("Moving to grasp pose...")
    planner.static_manipulation(grasp_pose, n_init_qpos=100)
    planner.planner.update_from_simulation()
    
    # Закрываем gripper
    print("Closing gripper...")
    res = planner.close_gripper(t=10)
    planner.planner.update_from_simulation()
    
    # Прикрепляем чашку к gripper
    kwargs = {"name": get_fcl_object_name(cup), "art_name": f'scene-0_{robot_name}', "link_id": planner.planner.move_group_link_id}
    planner.planner.planning_world.attach_object(**kwargs)
    planner.planner.update_from_simulation()
    
    # Разрешаем коллизии между пальцами и чашкой
    planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
        f'scene-0-{robot_name}_r_gripper_finger_link', get_fcl_object_name(cup), True
    )
    planner.planner.planning_world.get_allowed_collision_matrix().set_entry(
        f'scene-0-{robot_name}_l_gripper_finger_link', get_fcl_object_name(cup), True
    )
    
    # =============================================================================
    # Фаза 3: Подъем чашки
    # =============================================================================
    print("=== Phase 3: Lifting cup ===")
    planner.static_manipulation(lift_pose, n_init_qpos=100)
    planner.planner.update_from_simulation()
    
    # =============================================================================
    # Фаза 4: Перемещение к миске
    # =============================================================================
    print("=== Phase 4: Moving to bowl ===")
    
    # Вычисляем позицию для подъезда к миске
    drive_to_bowl = sapien.Pose(
        p=np.array([bowl_pos[0] - 0.5, bowl_pos[1], bowl_pos[2]], dtype=np.float32),
        q=np.array(tcp_quat, dtype=np.float32)
    )
    planner.drive_base(drive_to_bowl, above_bowl_pose)
    planner.planner.update_from_simulation()
    
    # Подъезжаем к позиции над миской
    print("Moving to above bowl...")
    planner.static_manipulation(above_bowl_pose, n_init_qpos=100)
    planner.planner.update_from_simulation()
    
    # =============================================================================
    # Фаза 5: Опускание чашки в миску
    # =============================================================================
    print("=== Phase 5: Placing cup ===")
    planner.static_manipulation(place_pose, n_init_qpos=100)
    planner.planner.update_from_simulation()
    
    # Открываем gripper
    print("Opening gripper...")
    planner.open_gripper(t=10)
    planner.planner.update_from_simulation()
    
    # =============================================================================
    # Фаза 6: Отъезд
    # =============================================================================
    print("=== Phase 6: Retreating ===")
    planner.static_manipulation(above_bowl_pose, n_init_qpos=100)
    planner.planner.update_from_simulation()
    
    print("Motion planning completed!")
    planner.render_wait()


if __name__ == '__main__':
    env = gym.make("MyRoboCasa-v1", num_envs=1, render_mode="human", robot_uids='ds_fetch')
    print(env.unwrapped.robot_uids)
    env = RecordEpisode(
        env, 
        output_dir=os.path.join("videos", "my_robocasa"), 
        save_video=True, 
        video_fps=30,
        save_on_reset=False,
    )
    print("Resetting environment...")
    obs, _ = env.reset(options={"reconfigure": True})

    print(env.unwrapped.agent)
    print(env.unwrapped.robot_uids)

    # Запускаем motion planning
    run_motion_planning(env)

    # Оставляем среду работать для визуализации
    while True:
        obs, rew, terminated, truncated, info = env.step(torch.from_numpy(env.action_space.sample()))
        done = (terminated | truncated).any()
