"""Shared grasp checks for TakeItBack planners."""

import numpy as np
import sapien

from robots.fetch.utils import compute_box_grasp_thin_side_info

FINGER_LENGTH = 0.025


def _grasp_pose(agent, obb, cup_center, ee_direction, target_closing, raise_z=0.04,
                back_off=0.1, lift_over=0.0, force_front=False):
    """Compute grasp and pre-grasp poses for the shared cup grasp ladder."""
    base_pos = agent.base_link.pose.p[0].cpu().numpy()
    grasp_info = compute_box_grasp_thin_side_info(
        obb,
        ee_direction=ee_direction,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
        ortho=True,
    )
    if force_front:
        approaching = ee_direction / np.linalg.norm(ee_direction)
        closing = grasp_info["closing"]
        closing = closing - (approaching @ closing) * approaching
        closing = closing / np.linalg.norm(closing)
        grasp_info["approaching"] = approaching
        grasp_info["closing"] = closing
        grasp_info["center"] = obb.center_mass.copy()
    grasp_pose = agent.build_grasp_pose(
        grasp_info["approaching"], grasp_info["closing"], grasp_info["center"]
    )
    grasp_pose.p[2] += raise_z
    reach_pose = grasp_pose * sapien.Pose([0, lift_over, -back_off])
    if np.dot(reach_pose.p - cup_center, base_pos - cup_center) < 0:
        print("Validation failed: grasp is diametrically opposite. Flipping approaching direction...")
        if force_front:
            approaching = -np.asarray(ee_direction, dtype=float)
            approaching /= np.linalg.norm(approaching)
            closing = closing - (approaching @ closing) * approaching
            closing = closing / np.linalg.norm(closing)
            grasp_pose = agent.build_grasp_pose(approaching, closing, obb.center_mass.copy())
        else:
            grasp_info = compute_box_grasp_thin_side_info(
                obb,
                ee_direction=-ee_direction,
                target_closing=target_closing,
                depth=FINGER_LENGTH,
                ortho=True,
            )
            grasp_pose = agent.build_grasp_pose(
                grasp_info["approaching"], grasp_info["closing"], grasp_info["center"]
            )
        grasp_pose.p[2] += raise_z
        reach_pose = grasp_pose * sapien.Pose([0, lift_over, -back_off])
    return grasp_pose, reach_pose


def _tcp_at(agent, target_pose, tol=0.03, z_only=False):
    """True if TCP reached target pose within ``tol`` metres."""
    tcp = agent.tcp.pose.p[0].cpu().numpy()
    target = np.asarray(target_pose.p, dtype=float)
    if z_only:
        return float(abs(tcp[2] - target[2])) <= tol
    return float(np.linalg.norm(tcp - target)) <= tol


def _tcp_to(agent, point):
    """Distance from TCP to point in metres."""
    tcp = agent.tcp.pose.p[0].cpu().numpy()
    return float(np.linalg.norm(tcp - np.asarray(point, dtype=float)))
