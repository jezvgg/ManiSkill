import os
from typing import Optional

import numpy as np
import sapien
import sapien.physx as physx
import torch
from PIL import Image
from transforms3d.quaternions import mat2quat, quat2mat

from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.geometry.rotation_conversions import axis_angle_to_quaternion
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Actor, Pose

from .base_robocasa import BaseRoboCasaSimple


@register_env("MyRoboCasa_FridgePicture-v1", asset_download_ids=["RoboCasa"])
class MyRoboCasaFridgePicture(BaseRoboCasaSimple):
    """Robot spawns in front of the fridge facing it; a user image is displayed
    on the fridge door for the first PICTURE_STEPS steps of each episode."""

    #: Path to any image file (PNG/JPG) shown on the fridge door. None disables the picture.
    PICTURE_PATH: Optional[str] = None
    PICTURE_WIDTH = 0.3  # picture width in meters (height from image aspect)
    PICTURE_STEPS = 100  # steps the picture stays visible
    ROBOT_FRONT_DISTANCE = 0.9  # robot base distance from the fridge door face (m)

    fridge: Actor
    picture: Optional[Actor]
    picture_center: np.ndarray
    robot_spawn_pos: np.ndarray
    agent_pose: sapien.Pose

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        fridge_key = next(k for k in self.fixture_placements if k.endswith("_Fridge"))
        self.fridge = self.scene.actors[fridge_key[: -len("_Fridge")] + "_0"]

        # fridge geometry from the built actor's world AABB
        aabbs = np.array([b.get_global_aabb_fast() for b in self.fridge._bodies])
        lower, upper = aabbs[:, 0].min(axis=0), aabbs[:, 1].max(axis=0)
        center = (lower + upper) / 2

        quat = np.asarray(self.fixture_placements[fridge_key]["quat"])
        # sapien quat (w, x, y, z) -> rotation matrix
        front = quat2mat(quat) @ np.array([0.0, -1.0, 0.0])  # door direction
        half_depth = ((upper - lower) * np.abs(front)).sum() / 2
        door_center = center + front * half_depth
        self.picture_center = door_center.copy()

        # robot spawn: in front of the door, facing back toward the fridge
        self.robot_spawn_pos = center + front * (half_depth + self.ROBOT_FRONT_DISTANCE)
        yaw = float(np.arctan2(-front[1], -front[0]))
        self._robot_yaw_quat = axis_angle_to_quaternion(torch.tensor([0.0, 0.0, yaw]))

        if self.PICTURE_PATH is None:
            self.picture = None
            return
        if not os.path.exists(self.PICTURE_PATH):
            raise FileNotFoundError(
                f"PICTURE_PATH image not found: {self.PICTURE_PATH}"
            )

        # picture size from image aspect ratio, clamped to the door face
        try:
            with Image.open(self.PICTURE_PATH) as im:
                iw, ih = im.size
        except Exception:
            iw = ih = 1
        w = self.PICTURE_WIDTH
        h = w * ih / iw
        face_w = (
            (upper - lower) * np.abs(quat2mat(quat) @ np.array([1.0, 0.0, 0.0]))
        ).sum()
        face_h = upper[2] - lower[2]
        scale = min(1.0, 0.9 * face_w / w, 0.85 * face_h / h)
        w, h = w * scale, h * scale

        mat = sapien.render.RenderMaterial()
        mat.base_color_texture = sapien.render.RenderTexture2D(
            filename=self.PICTURE_PATH, mipmap_levels=1
        )
        width_dir = np.cross(np.array([0.0, 0.0, 1.0]), front)
        rot = np.column_stack([front, width_dir, np.array([0.0, 0.0, 1.0])])
        pose = sapien.Pose(p=door_center + front * 0.005, q=mat2quat(rot))

        builder = self.scene.create_actor_builder()
        builder.add_plane_visual(scale=[1.0, w / 2, h / 2], material=mat)
        builder.initial_pose = pose
        self.picture = builder.build_dynamic(name="fridge_picture")
        for obj in self.picture._objs:
            obj.find_component_by_type(
                physx.PhysxRigidDynamicComponent
            ).set_disable_gravity(True)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # NOTE: BaseRoboCasaScene._initialize_episode needs cup_pos, which this
        # scene never sets; BaseEnv's implementation is a no-op, so it is skipped.
        agent_pos = self.agent.robot.pose.p[0]
        agent_pos[0] = self.robot_spawn_pos[0]
        agent_pos[1] = self.robot_spawn_pos[1]
        self.agent_pose = Pose.create_from_pq(
            p=agent_pos, q=self._robot_yaw_quat
        )
        self.agent.robot.set_pose(self.agent_pose)
        self._picture_steps = 0
        if self.picture is not None:
            self.picture.show_visual()

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self.picture is not None:
            self._picture_steps += 1
            if self._picture_steps == self.PICTURE_STEPS:
                self.picture.hide_visual()
        return obs, reward, terminated, truncated, info

    @property
    def _default_human_render_camera_configs(self):
        camera_pos = self.robot_spawn_pos + np.array([0.0, 0.0, 1.3])
        pose = sapien_utils.look_at(camera_pos, self.picture_center)
        return CameraConfig(
            "render_camera", pose, 2048, 2048, 60 * np.pi / 180, 0.01, 100
        )
