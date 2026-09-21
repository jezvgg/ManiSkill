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


def subtract_rect(regions, blocker):
    """Subtract axis-aligned ``[x0, x1, y0, y1]`` blocker from rects."""
    out = []
    bx0, bx1, by0, by1 = blocker
    for r in regions:
        x0, x1, y0, y1 = r
        ix0, ix1 = max(x0, bx0), min(x1, bx1)
        iy0, iy1 = max(y0, by0), min(y1, by1)
        if ix0 >= ix1 or iy0 >= iy1:
            out.append(np.asarray(r, dtype=np.float64))
            continue
        if x0 < ix0:
            out.append(np.asarray([x0, ix0, y0, y1], dtype=np.float64))
        if ix1 < x1:
            out.append(np.asarray([ix1, x1, y0, y1], dtype=np.float64))
        if y0 < iy0:
            out.append(np.asarray([ix0, ix1, y0, iy0], dtype=np.float64))
        if iy1 < y1:
            out.append(np.asarray([ix0, ix1, iy1, y1], dtype=np.float64))
    return out
