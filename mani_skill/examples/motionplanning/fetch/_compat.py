"""Shims over ManiSkill API drift.

``build_two_finger_gripper_grasp_pose_visual`` lives in
``mani_skill.examples.motionplanning.two_finger_gripper`` in recent ManiSkill, but
older releases (e.g. 3.0.0b15) only ship the panda-specific
``build_panda_gripper_grasp_pose_visual``. Both only draw the debug marker for a
grasp pose, so either satisfies the planner.
"""

try:
    from mani_skill.examples.motionplanning.two_finger_gripper.motionplanner import (
        build_two_finger_gripper_grasp_pose_visual,
    )
except ImportError:  # older mani_skill
    from mani_skill.examples.motionplanning.panda.motionplanner import (
        build_panda_gripper_grasp_pose_visual as build_two_finger_gripper_grasp_pose_visual,
    )

__all__ = ["build_two_finger_gripper_grasp_pose_visual"]
