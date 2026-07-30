import numpy as np
import torch
from mani_skill.utils.geometry.rotation_conversions import axis_angle_to_quaternion
from mani_skill.utils.structs import Actor

def get_actor_size(actor: Actor):
    bodies = np.array([body.get_global_aabb_fast() for body in actor._bodies])
    return (bodies.max(axis=1) - bodies.min(axis=1))[0]

def degree_to_quanterion(x: int = 0, y: int = 0, z: int = 0):
    return axis_angle_to_quaternion(
        torch.Tensor([x * torch.pi / 180, y * torch.pi / 180, z * torch.pi / 180])
    )
