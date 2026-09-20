from copy import deepcopy
from pathlib import Path

import os

import numpy as np
import sapien
from transforms3d import euler

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.fetch.fetch import Fetch
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.structs import Pose

# The URDF ships with this package. extand.py derives the SRDF path from it by
# swapping the suffix, so both files must stay side by side.
_URDF_PATH = Path(__file__).resolve().parent / "fetch.urdf"  # beside the robot; meshes are his

# pd_ee_delta_pose (ee_base.py): the frame and the per-step bounds. Environment
# overrides exist for measurement, not for shipping a different robot.
EE_ROOT_LINK = os.environ.get("MIKASA_EE_ROOT_LINK", "base_link")
EE_POS_STEP = float(os.environ.get("MIKASA_EE_POS_STEP", "0.2"))  # metres per control step
EE_ROT_STEP = float(os.environ.get("MIKASA_EE_ROT_STEP", "0.5"))  # radians per control step


@register_agent()
class MikasaDSFetch(Fetch):
    uid = "mikasa_ds_fetch"
    urdf_path = str(_URDF_PATH)


    def before_simulation_step(self):
        """Apply interpolated joint targets on the GPU, every physics sub-step.

        The body controller (torso, head) runs with `interpolate=True`: it writes a new
        drive target in `before_simulation_step` on each of the five sub-steps of a
        control step. On the CPU those writes reach PhysX at once. On the GPU, ManiSkill
        3.0.0b22 copies the target buffer to the simulator ONCE, right after
        `set_action` (sapien_env.py:1106-1117) — the sub-step writes land in a buffer
        nobody applies, the torso and head drives keep their reset targets (0) and the
        torso sinks to the floor within ten steps (measured 2026-09-08, W32: 0.386 → 0.000
        under every control mode; the arm, which does not interpolate, tracks fine).
        One extra apply per sub-step fixes it; on the CPU this is a no-op.
        """
        super().before_simulation_step()
        if self.scene.gpu_sim_enabled:
            self.scene.px.gpu_apply_articulation_target_position()

    @property
    def _sensor_configs(self):
        """The real Fetch's cameras (2026-09-10, the owner: "откати их к реальному fetch"):
        one camera ON the head — `fetch_head`, at the head camera frame's origin, looking
        along it, turning with head_pan / head_tilt — and the wrist camera `fetch_hand` on
        the gripper. The two "base" cameras this robot carried since the fork
        (`left/right_base_camera_link`: on the head link but 0.5 m behind and 0.5 m beside
        it, pitched down, turned inward) looked at the robot from outside — its own torso
        and raised arm filled their frames; they stay only behind MIKASA_SHOULDER_CAMERAS=1
        for reading old recordings side by side.

        Both cameras are 224 x 224 at fov 2 rad (115 deg), set by the owner on 2026-09-12.
        224 is what openpi feeds the model, so nothing is resized on the way in and no detail
        is paid for twice. Before this the head was 256 at fov 1.5 (86 deg) and the wrist 128
        at fov 2, so the wrist only gains pixels while the head also gains width of view. Note
        for anyone comparing against hardware: the real Fetch's head camera is 640 x 480 at
        about 54 x 45 deg, much narrower than fov 2 -- this is a deliberately wide view, not a
        model of that lens. Changing either number invalidates every rgb dataset recorded
        before it -- re-render from the state recordings, do not re-plan."""
        cams = [
            CameraConfig(
                uid="fetch_head",
                pose=Pose.create_from_pq([0, 0, 0], [1, 0, 0, 0]),
                width=224,
                height=224,
                fov=float(os.environ.get("MIKASA_HEAD_CAMERA_FOV", "2")),
                near=0.01,
                far=100,
                entity_uid="head_camera_link",
            ),
            CameraConfig(
                uid="fetch_hand",
                pose=Pose.create_from_pq(
                    [0.1, 0, -0.1], euler.euler2quat(np.pi, -np.pi / 2, 0)
                ),
                width=224,
                height=224,
                fov=2,
                near=0.01,
                far=100,
                entity_uid="gripper_link",
            ),
        ]
        if os.environ.get("MIKASA_SHOULDER_CAMERAS", "0") == "1":
            cams += [
                CameraConfig(
                    uid="left_base_camera_link",
                    pose=Pose.create_from_pq(
                        [-0.5, 0.5, 0], euler.euler2quat(0, 0.3, -0.2)
                    ),
                    width=256, height=256, fov=1.5, near=0.01, far=100,
                    entity_uid="head_camera_link",
                ),
                CameraConfig(
                    uid="right_base_camera_link",
                    pose=Pose.create_from_pq(
                        [-0.5, -0.5, 0], euler.euler2quat(0, 0.3, 0.2)
                    ),
                    width=256, height=256, fov=1.5, near=0.01, far=100,
                    entity_uid="head_camera_link",
                ),
            ]
        return cams

    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
        )
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        # PD ee position
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )

        # EE delta pose in the frame of the MOBILE BASE (ee_base.py). The inherited
        # config above is in the articulation-root frame, i.e. the dock the robot was
        # spawned at, which is meaningless once the base drives. Bounds per control
        # step (0.05 s), sized from the recorded demonstrations (journal 2026-09-08,
        # W32): the lift of the cup out of the cabinet moves the hand 0.136 m in one
        # step, so 0.1 m clipped it; 0.2 m and 0.5 rad leave every recorded step
        # unclipped.
        # Imported here, not at module import: ee_base leans on b22 internals of
        # `Kinematics`; a host with another ManiSkill must still import the agent for
        # every other mode (review, 2026-09-08). The env is built long after import.
        from .ee_base import PDEEPoseBaseControllerConfig

        arm_pd_ee_delta_pose_base = PDEEPoseBaseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-EE_POS_STEP,
            pos_upper=EE_POS_STEP,
            rot_lower=-EE_ROT_STEP,
            rot_upper=EE_ROT_STEP,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            root_link_name=EE_ROOT_LINK,
            frame="root_translation:root_aligned_body_rotation",
        )

        # The remaining EE modes (pd_ee_delta_pos, the two *target* variants, *_align)
        # still derive from the INHERITED root-frame config and its Euler/0.1 bounds:
        # nothing in the repo uses them, and they are left as ManiSkill defines them
        # rather than silently re-based. pd_ee_target_delta_pose is therefore NOT the
        # use_target variant of pd_ee_delta_pose.
        arm_pd_ee_target_delta_pos = deepcopy(arm_pd_ee_delta_pos)
        arm_pd_ee_target_delta_pos.use_target = True
        arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose.use_target = True

        # PD ee position (for human-interaction/teleoperation)
        arm_pd_ee_delta_pose_align = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_delta_pose_align.frame = "ee_align"

        # PD joint velocity
        arm_pd_joint_vel = PDJointVelControllerConfig(
            self.arm_joint_names,
            -1.0,
            1.0,
            self.arm_damping,  # this might need to be tuned separately
            self.arm_force_limit,
        )

        # PD joint position and velocity
        arm_pd_joint_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=True,
        )
        arm_pd_joint_delta_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
        )

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #
        # NOTE(jigu): IssacGym uses large P and D but with force limit
        # However, tune a good force limit to have a good mimic behavior
        # Diagnostic only (K99): cap the finger drives' force. The inherited limit is
        # 100 N against condiments that tip at 0.0038 N*s (K94) — during the approach
        # and grasp legs the fingers hold position, so a finger dragged into a standing
        # object pushes with the arm's full authority and the object always loses
        # (K90: every topple is a fingertip). A single-digit limit lets the finger
        # *yield* on contact instead; holding the 15.8 g shaker needs ~0.2 N of grip.
        # Unset (the default) leaves the agent byte-identical.
        _fl = os.environ.get("MIKASA_GRIPPER_FORCE_LIMIT")
        if _fl is not None:
            self.gripper_force_limit = float(_fl)
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            -0.01,  # a trick to have force when the object is thin
            0.05,
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
        )

        # -------------------------------------------------------------------------- #
        # Body
        # -------------------------------------------------------------------------- #
        # The DELTA modes' body (2026-09-09, the supervisor's format note): head pan,
        # head tilt and torso as increments of +-0.1 rad / +-0.1 m per step, normalized
        # to [-1, 1] by ManiSkill like the arm — one scale across the whole vector. The
        # fork had `body_pd_joint_pos` (absolute, raw radians and metres) in every mode,
        # so `pd_joint_delta_pos` carried a mixed vector: arm in [-1, 1], body raw.
        body_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.body_joint_names,
            -0.1,
            0.1,
            self.body_stiffness,
            self.body_damping,
            self.body_force_limit,
            use_delta=True,
        )

        body_pd_joint_pos = PDJointPosControllerConfig(
            self.body_joint_names,
            None,
            None,
            self.body_stiffness,
            self.body_damping,
            self.body_force_limit,
            use_delta=False,
            normalize_action=False,
            interpolate=True,
        )

        # useful to keep body unmoving from passed position
        stiff_body_pd_joint_pos = PDJointPosControllerConfig(
            self.body_joint_names,
            None,
            None,
            1e5,
            1e5,
            1e5,
            normalize_action=False,
        )

        # -------------------------------------------------------------------------- #
        # Base
        # -------------------------------------------------------------------------- #
        base_pd_joint_vel = PDBaseForwardVelControllerConfig(
            self.base_joint_names,
            lower=[-1, -3.14],
            upper=[1, 3.14],
            damping=1000,
            force_limit=500,
            normalize_action=True,
        )

        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos,
                gripper=gripper_pd_joint_pos,
                body=body_pd_joint_delta_pos,
                base=base_pd_joint_vel,
            ),
            pd_joint_pos=dict(
                arm=arm_pd_joint_pos,
                gripper=gripper_pd_joint_pos,
                body=body_pd_joint_pos,
                base=base_pd_joint_vel,
            ),
            pd_ee_delta_pos=dict(
                arm=arm_pd_ee_delta_pos,
                gripper=gripper_pd_joint_pos,
                body=body_pd_joint_delta_pos,
                base=base_pd_joint_vel,
            ),
            # pd_ee_delta_pose: arm deltas in the BASE frame (ee_base.py); the inherited
            # articulation-root-frame semantics stay reachable as pd_ee_delta_pose_root.
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose_base,
                gripper=gripper_pd_joint_pos,
                body=body_pd_joint_delta_pos,
                base=base_pd_joint_vel,
            ),
            pd_ee_delta_pose_root=dict(
                arm=arm_pd_ee_delta_pose,
                gripper=gripper_pd_joint_pos,
                body=body_pd_joint_delta_pos,
                base=base_pd_joint_vel,
            ),
            pd_ee_delta_pose_align=dict(
                arm=arm_pd_ee_delta_pose_align,
                gripper=gripper_pd_joint_pos,
                body=body_pd_joint_delta_pos,
                base=base_pd_joint_vel,
            ),
            # TODO(jigu): how to add boundaries for the following controllers
            pd_joint_target_delta_pos=dict(
                arm=arm_pd_joint_target_delta_pos,
                gripper=gripper_pd_joint_pos,
                body=body_pd_joint_pos,
                base=base_pd_joint_vel,
            ),
            pd_ee_target_delta_pos=dict(
                arm=arm_pd_ee_target_delta_pos,
                gripper=gripper_pd_joint_pos,
                body=body_pd_joint_pos,
                base=base_pd_joint_vel,
            ),
            pd_ee_target_delta_pose=dict(
                arm=arm_pd_ee_target_delta_pose,
                gripper=gripper_pd_joint_pos,
                body=body_pd_joint_pos,
                base=base_pd_joint_vel,
            ),
            # Caution to use the following controllers
            pd_joint_vel=dict(
                arm=arm_pd_joint_vel,
                gripper=gripper_pd_joint_pos,
                body=body_pd_joint_pos,
                base=base_pd_joint_vel,
            ),
            pd_joint_pos_vel=dict(
                arm=arm_pd_joint_pos_vel,
                gripper=gripper_pd_joint_pos,
                body=body_pd_joint_pos,
                base=base_pd_joint_vel,
            ),
            pd_joint_delta_pos_vel=dict(
                arm=arm_pd_joint_delta_pos_vel,
                gripper=gripper_pd_joint_pos,
                body=body_pd_joint_pos,
                base=base_pd_joint_vel,
            ),
            pd_joint_delta_pos_stiff_body=dict(
                arm=arm_pd_joint_delta_pos,
                gripper=gripper_pd_joint_pos,
                body=stiff_body_pd_joint_pos,
                base=base_pd_joint_vel,
            ),
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)
