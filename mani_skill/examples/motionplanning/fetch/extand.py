import os
from collections import deque

import mplib
import numpy as np
import sapien
import trimesh
from transforms3d.euler import euler2mat, euler2quat

from mani_skill.agents.base_agent import BaseAgent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.utils.structs.pose import to_sapien_pose

from ._compat import build_two_finger_gripper_grasp_pose_visual
from .base_yaw import HeldBodies, new_contacts, sweep_yaw, yaw_path
from .stepping import StepGuard, find_log_event, pose_error, refine_should_stop
from .utils import SapienPlannerV2, SapienPlanningWorldV2

#: Seconds RRTConnect may spend before it gives up (`plan_pose`). Overridable with
#: `MIKASA_PLANNING_TIME` so the budget can be swept without editing four call
#: sites. Note what this does and does not buy: RRTConnect stops the moment its two
#: trees meet and never improves the path afterwards, so a larger budget lowers the
#: rate of `no plan` refusals and does **not** straighten the trajectory. The one
#: site that used double this value keeps doing so.
PLANNING_TIME = float(os.environ.get("MIKASA_PLANNING_TIME", "2"))

OPEN = 1
CLOSED = -1

# `masked_joints` for move_base_forward's screw plans (its own mask, unchanged since
# the inherited code): the 15 user joints of ds_fetch — base x, y, yaw free, torso_lift
# (index 3) masked, head/arm/gripper free. rotate_base_z no longer plans with
# plan_screw at all (K51/D12, see rotate_base_z), so it takes no mask.
BASE_PLAN_MASK = [True, True, True, False] + [True] * 11

# The opt-in mask for the same screw plans: base x, y, yaw free and **everything else
# frozen**. `masked_joints[i] is False` zeroes that joint's Jacobian column
# (`SapienPlannerV2.plan_screw`), so the joint cannot move in the plan.
#
# Why it exists (`docs/solver-delta-primer.md` §3, the same disease in translation).
# `move_base_forward` asks the planner for a TCP pose one base-delta away and, with
# BASE_PLAN_MASK, hands it fifteen joints to get there — the request never says "with
# the base only". §3 makes exactly this argument for base *rotation* and answers it
# with `base_yaw.py`, which turns one joint and freezes the rest; translation kept the
# old shape. The cost is measured: on `MikasaWaterPlants-v0`, seed 11, the drive to a
# plant was refused with `joint limit at index [11]` — the wrist_flex, inside a plan
# whose only job was to move the base — reproducibly, twice, from the same start pose,
# while the same leg from a different arm configuration parked 0.009 m from its dock.
# A base translation moves the TCP by exactly that translation, so root_x and root_y
# alone span the required twist; the arm was never needed.
#
# **Opt-in, and it must stay opt-in.** `MikasaBurner-v0`, `MikasaSeasonDish-v0` and
# `MikasaStationChecklist-v0` have published numbers and recorded demonstrations taken
# with BASE_PLAN_MASK; changing the default would change which plans they find and
# invalidate both. `freeze_arm=True` is passed by `water_plants_solution.py` and by
# nothing else.
#
# Note what it does *not* change: `follow_moving_forward` sends the arm's **current**
# qpos as the arm action on every step whatever the plan says, so the mask decides
# whether a base move plans at all, never what the arm executes.
BASE_ONLY_PLAN_MASK = [True, True, True] + [False] * 12


class PandaArmMotionPlanningSolverV2(PandaArmMotionPlanningSolver):
    def __init__(
        self,
        env: BaseEnv,
        debug: bool = False,
        vis: bool = True,
        base_pose: sapien.Pose = None,  # TODO mplib doesn't support robot base being anywhere but 0
        visualize_target_grasp_pose: bool = True,
        print_env_info: bool = True,
        joint_vel_limits=0.9,
        joint_acc_limits=0.9,
        objects=[],
    ):
        self.env = env
        self.base_env: BaseEnv = env.unwrapped
        self.env_agent: BaseAgent = self.base_env.agent
        self._sim_scene: sapien.Scene = self.base_env.scene.sub_scenes[0]
        self.robot = self.env_agent.robot
        self.joint_vel_limits = joint_vel_limits
        self.joint_acc_limits = joint_acc_limits

        self.base_pose = to_sapien_pose(base_pose)

        self.planner = self.setup_planner(objects)
        self.control_mode = self.base_env.control_mode

        self.debug = debug
        self.vis = vis
        self.print_env_info = print_env_info
        self.visualize_target_grasp_pose = visualize_target_grasp_pose
        self.gripper_state = OPEN
        self.grasp_pose_visual = None
        if self.vis and self.visualize_target_grasp_pose:
            if "grasp_pose_visual" not in self.base_env.scene.actors:
                self.grasp_pose_visual = build_two_finger_gripper_grasp_pose_visual(
                    self.base_env.scene
                )
            else:
                self.grasp_pose_visual = self.base_env.scene.actors["grasp_pose_visual"]
            self.grasp_pose_visual.set_pose(self.base_env.agent.tcp.pose)
        self.elapsed_steps = 0

        self.use_point_cloud = False
        self.collision_pts_changed = False
        self.all_collision_pts = None

    def setup_planner(self, objects=[]):
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        planner = mplib.Planner(
            urdf=self.env_agent.urdf_path,
            srdf=self.env_agent.urdf_path.replace(".urdf", ".srdf"),
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group="panda_hand_tcp",
            joint_vel_limits=np.ones(7) * self.joint_vel_limits,
            joint_acc_limits=np.ones(7) * self.joint_acc_limits,
            objects=objects,
        )
        planner.set_base_pose(mplib.Pose(self.base_pose.p, self.base_pose.q))
        return planner

    def move_to_pose_with_RRTConnect(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0, mask=None
    ):
        pose = to_sapien_pose(pose)
        if self.grasp_pose_visual is not None:
            self.grasp_pose_visual.set_pose(pose)
        pose = mplib.Pose(p=pose.p, q=pose.q)
        result = self.planner.plan_pose(
            pose,
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
            # use_point_cloud=self.use_point_cloud,
            wrt_world=True,
            verbose=True,
            planning_time=PLANNING_TIME,
            rrt_range=0.1,
            simplify=True,
            mask=mask,
        )
        if result["status"] != "Success":
            print(result["status"])
            self.render_wait()
            return -1
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    def move_to_pose_with_screw(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):
        pose = to_sapien_pose(pose)
        # try screw two times before giving up
        if self.grasp_pose_visual is not None:
            self.grasp_pose_visual.set_pose(pose)
        pose = sapien.Pose(p=pose.p, q=pose.q)
        result = self.planner.plan_screw(
            mplib.Pose(pose.p, pose.q),
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
            verbose=True,
            # use_point_cloud=self.use_point_cloud,
        )
        if result["status"] != "Success":
            result = self.planner.plan_screw(
                mplib.Pose(pose.p, pose.q),
                self.robot.get_qpos().cpu().numpy()[0],
                time_step=self.base_env.control_timestep,
                # # use_point_cloud=self.use_point_cloud,
            )
            if result["status"] != "Success":
                print(result["status"])
                self.render_wait()
                return -1
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    def open_gripper(self):
        self.gripper_state = OPEN
        qpos = self.robot.get_qpos()[0, :-2].cpu().numpy()
        for i in range(6):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state])
            else:
                action = np.hstack([qpos, qpos * 0, self.gripper_state])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def close_gripper(self, t=6, gripper_state=CLOSED):
        self.gripper_state = gripper_state
        qpos = self.robot.get_qpos()[0, :-2].cpu().numpy()
        for i in range(t):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state])
            else:
                action = np.hstack([qpos, qpos * 0, self.gripper_state])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def add_box_collision(
        self, extents: np.ndarray, pose: sapien.Pose, name="scene_pcd"
    ):
        self.use_point_cloud = True
        box = trimesh.creation.box(extents, transform=pose.to_transformation_matrix())
        pts, _ = trimesh.sample.sample_surface(box, 500)
        if self.all_collision_pts is None:
            self.all_collision_pts = {name: pts}
        else:
            self.all_collision_pts[name] = pts
        self.planner.update_point_cloud(
            self.all_collision_pts[name], resolution=1e-2, name=name
        )

    def remove_collision_pts(self, name):
        del self.all_collision_pts[name]
        self.planner.remove_point_cloud(name)

    def add_collision_pts(self, pts: np.ndarray, name="scene_pcd"):
        if self.all_collision_pts is None:
            self.all_collision_pts = {name: pts}
        else:
            # self.all_collision_pts = np.vstack([self.all_collision_pts, pts])
            self.all_collision_pts[name] = pts
        self.planner.update_point_cloud(
            self.all_collision_pts[name], resolution=1e-2, name=name
        )

    def get_all_collision_pts(self):
        all_points = [pts for pts in self.all_collision_pts.values()]
        return np.vstack(all_points)

    def clear_collisions(self):
        self.all_collision_pts = None
        self.use_point_cloud = False

    def close(self):
        pass


class PandaArmMotionPlanningSapienSolver(PandaArmMotionPlanningSolverV2):
    def __init__(
        self,
        env: BaseEnv,
        debug: bool = False,
        vis: bool = True,
        base_pose: sapien.Pose = None,  # TODO mplib doesn't support robot base being anywhere but 0
        visualize_target_grasp_pose: bool = True,
        print_env_info: bool = True,
        joint_vel_limits=0.9,
        joint_acc_limits=0.9,
        objects=[],
        disable_actors_collision=False,
        verbose=True,
    ):
        self.verbose = verbose
        self.disable_actors_collision = disable_actors_collision
        super().__init__(
            env,
            debug,
            vis,
            base_pose,
            visualize_target_grasp_pose,
            print_env_info,
            joint_vel_limits,
            joint_acc_limits,
            objects,
        )

    def setup_planner(self, objects=[]):
        # raise NotImplementedError
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]

        planned_articulation = self._sim_scene.get_all_articulations()[0]
        planning_world = SapienPlanningWorldV2(
            self._sim_scene,
            [planned_articulation],
            disable_actors_collision=self.disable_actors_collision,
        )
        planner = SapienPlannerV2(
            planning_world,
            "scene-0-panda_wristcam_panda_hand_tcp",
            joint_vel_limits=np.ones(7) * self.joint_vel_limits,
            joint_acc_limits=np.ones(7) * self.joint_acc_limits,
        )

        planner.set_base_pose(mplib.Pose(self.base_pose.p, self.base_pose.q))
        return planner

    def move_to_pose_with_RRTConnect(
        self,
        pose: sapien.Pose,
        dry_run: bool = False,
        refine_steps: int = 0,
        mask=None,
        n_init_qpos=20,
    ):
        pose = to_sapien_pose(pose)
        if self.grasp_pose_visual is not None:
            self.grasp_pose_visual.set_pose(pose)
        pose = mplib.Pose(p=pose.p, q=pose.q)
        result = self.planner.plan_pose(
            pose,
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
            # use_point_cloud=self.use_point_cloud,
            wrt_world=True,
            verbose=True,
            planning_time=PLANNING_TIME,
            rrt_range=0.1,
            simplify=True,
            mask=mask,
            n_init_qpos=n_init_qpos,
        )
        if result["status"] != "Success":
            print(result["status"])
            self.render_wait()
            return -1
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)


class FetchStaticArmMotionPlanningSapienSolver(PandaArmMotionPlanningSapienSolver):
    def setup_planner(self, *args, **kwargs):
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]

        planned_articulation = self._sim_scene.get_all_articulations()[0]
        planning_world = SapienPlanningWorldV2(
            self._sim_scene,
            [planned_articulation],
            disable_actors_collision=self.disable_actors_collision,
        )
        planner = SapienPlannerV2(
            planning_world,
            f"scene-0-{self.robot.name}_gripper_link",
            joint_vel_limits=np.ones(8) * self.joint_vel_limits,
            joint_acc_limits=np.ones(8) * self.joint_acc_limits,
        )

        planner.set_base_pose(mplib.Pose(self.base_pose.p, self.base_pose.q))
        return planner

    def move_to_pose_with_screw_static_body(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):
        pose = to_sapien_pose(pose)
        # try screw two times before giving up
        if self.grasp_pose_visual is not None:
            self.grasp_pose_visual.set_pose(pose)
        pose = sapien.Pose(p=pose.p, q=pose.q)
        result = self.planner.plan_screw(
            mplib.Pose(pose.p, pose.q),
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
            verbose=True,
            masked_joints=[False] + [True] * 11,
            # use_point_cloud=self.use_point_cloud,
        )
        if result["status"] != "Success":
            result = self.planner.plan_screw(
                mplib.Pose(pose.p, pose.q),
                self.robot.get_qpos().cpu().numpy()[0],
                time_step=self.base_env.control_timestep,
                masked_joints=[False] + [True] * 11,
                # # use_point_cloud=self.use_point_cloud,
            )
            if result["status"] != "Success":
                print(result["status"])
                self.render_wait()
                return -1
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    def follow_path(self, result, refine_steps: int = 0, refine: bool = False):
        return self.follow_forward_path_w_refinement(result, refine)

    def lift_hand(self, delta_h=0.0, dry_run: bool = False, refine_steps: int = 0):
        cur_pose = self.base_env.agent.tcp.pose.sp
        taget_pose = mplib.Pose(
            p=cur_pose.p + np.array([0.0, 0.0, delta_h]), q=cur_pose.q
        )
        result = self.planner.plan_screw(
            taget_pose,
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
            verbose=True,
            # use_point_cloud=self.use_point_cloud,
        )
        if result["status"] != "Success":
            print(result["status"])
            self.render_wait()
            return -1
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    def follow_forward_path_w_refinement(
        self, result, refine: bool = False, static=False
    ):
        qpos_final = result["position"][-1]
        qpos_dict_final = {}
        for idx, q in zip(self.planner.move_group_joint_indices, qpos_final):
            joint_name = self.planner.user_joint_names[idx]
            qpos_dict_final[joint_name] = q

        n_step = result["position"].shape[0]

        for i in range(n_step):
            arm_action = (
                self.env_agent.controller.controllers["arm"].qpos[0].cpu().numpy()
            )

            qpos = result["position"][min(i, n_step - 1)]
            qvel = result["velocity"][min(i, n_step - 1)]

            qpos_dict = {}

            for idx, q in zip(self.planner.move_group_joint_indices, qpos):
                joint_name = self.planner.user_joint_names[idx]
                qpos_dict[joint_name] = q

            for n, joint_name in enumerate(
                self.env_agent.controller.controllers["arm"].config.joint_names
            ):
                arm_action[n] = qpos_dict[f"scene-0-{self.robot.name}_{joint_name}"]

            assert self.control_mode == "pd_joint_pos"

            body_action = np.zeros_like(
                self.env_agent.controller.controllers["body"].qpos[0].cpu().numpy()
            )
            body_action[2] = qpos_dict[f"scene-0-{self.robot.name}_torso_lift_joint"]

            # base_action = np.array([0., 0.])
            # base_action[0] =  np.sqrt(qvel[0] ** 2 + qvel[1] ** 2)

            action = np.hstack([arm_action, self.gripper_state, body_action])
            print("arm Action:", np.round(arm_action, 4))
            print("body Action:", np.round(body_action, 4))
            # print("base Action:", np.round(base_action, 4))
            print("Full: ", np.round(self.robot.get_qpos().cpu().numpy()[0], 4))
            obs, reward, terminated, truncated, info = self.env.step(action)

            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()

        if refine:
            # REFINEMENT!
            passed_refine_steps = 0
            last_lift_poses = deque(maxlen=10)
            last_x_base_poses = deque(maxlen=10)
            last_lift_vels = deque(maxlen=10)
            last_x_base_vels = deque(maxlen=10)
            print("==== REFINEMENT ====")

            while not self.check_body_close_to_target(qpos_dict_final):
                if (
                    (len(last_lift_vels) > 4 and np.std(last_lift_vels) < 1e-3)
                    and (len(last_x_base_vels) > 4 and np.std(last_x_base_vels) < 1e-3)
                    and (len(last_lift_poses) > 4 and np.std(last_lift_poses) < 1e-3)
                    and (
                        len(last_x_base_poses) > 4 and np.std(last_x_base_poses) < 1e-3
                    )
                ):
                    # robot is stuck
                    print("Robot is stuck")
                    break

                body_action = np.zeros_like(
                    self.env_agent.controller.controllers["body"].qpos[0].cpu().numpy()
                )
                body_action[2] = qpos_dict_final[
                    f"scene-0-{self.robot.name}_torso_lift_joint"
                ]
                body_action[0] = body_action[1] = 0.0

                # base_action = np.array([0., 0.])

                last_lift_poses.append(
                    self.env_agent.controller.controllers["body"]
                    .qpos[0]
                    .cpu()
                    .numpy()[2]
                )
                last_lift_vels.append(
                    self.env_agent.controller.controllers["body"]
                    .qvel[0]
                    .cpu()
                    .numpy()[2]
                )

                action = np.hstack([arm_action, self.gripper_state, body_action])
                print("arm Action:", np.round(arm_action, 4))
                print("body Action:", np.round(body_action, 4))
                # print("base Action:", np.round(base_action, 4))
                print("Full: ", np.round(self.robot.get_qpos().cpu().numpy()[0], 4))
                obs, reward, terminated, truncated, info = self.env.step(action)
                passed_refine_steps += 1
                self.elapsed_steps += 1
                if self.print_env_info:
                    print(
                        f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                    )
                if self.vis:
                    self.base_env.render_human()

        return obs, reward, terminated, truncated, info

    def check_body_close_to_target(self, target_dict, eps=1e-3):
        body_qpos = (
            self.env_agent.controller.controllers["body"].qpos[0].cpu().numpy()[2]
        )
        target_lift_joint_height = target_dict[
            f"scene-0-{self.robot.name}_torso_lift_joint"
        ]

        # base_xy = self.env_agent.controller.controllers['base'].qpos[0].cpu().numpy()[0:2]
        # target_base = np.array([
        #     target_dict[f'scene-0-{self.robot.name}_root_x_axis_joint'],
        #     target_dict[f'scene-0-{self.robot.name}_root_y_axis_joint']
        # ])

        robot_qpos = self.robot.get_qpos().cpu().numpy()[0]
        arm_pos = robot_qpos[
            self.env_agent.controller.controllers["arm"]
            .active_joint_indices.cpu()
            .numpy()
        ]
        target_arm_pos = np.array(
            [
                target_dict[f"scene-0-{self.robot.name}_shoulder_pan_joint"],
                target_dict[f"scene-0-{self.robot.name}_shoulder_lift_joint"],
                target_dict[f"scene-0-{self.robot.name}_upperarm_roll_joint"],
                target_dict[f"scene-0-{self.robot.name}_elbow_flex_joint"],
                target_dict[f"scene-0-{self.robot.name}_forearm_roll_joint"],
                target_dict[f"scene-0-{self.robot.name}_wrist_flex_joint"],
                target_dict[f"scene-0-{self.robot.name}_wrist_roll_joint"],
            ]
        )
        return np.allclose(
            body_qpos, target_lift_joint_height, atol=eps
        ) and np.allclose(arm_pos, target_arm_pos, atol=eps)

    def open_gripper(self):
        self.gripper_state = OPEN
        qpos = self.robot.get_qpos()[0, :-2].cpu().numpy()
        for i in range(6):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state])
            else:
                action = np.hstack([qpos, qpos * 0, self.gripper_state])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def close_gripper(self, t=6, gripper_state=CLOSED):
        self.gripper_state = gripper_state
        arm_action = self.env_agent.controller.controllers["arm"].qpos[0].cpu().numpy()
        body_action = (
            self.env_agent.controller.controllers["body"].qpos[0].cpu().numpy()
        )
        base_vel = np.array([0, 0])

        for i in range(t):
            if self.control_mode == "pd_joint_pos":
                # action = np.hstack([arm_action, self.gripper_state, body_action, base_vel])
                action = np.hstack([arm_action, self.gripper_state, body_action])
            else:
                raise NotImplementedError
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info


class FetchQuasiStaticArmMotionPlanningSapienSolver(PandaArmMotionPlanningSapienSolver):
    def setup_planner(self, *args, **kwargs):
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]

        planned_articulation = self._sim_scene.get_all_articulations()[0]
        planning_world = SapienPlanningWorldV2(
            self._sim_scene,
            [planned_articulation],
            disable_actors_collision=self.disable_actors_collision,
        )
        planner = SapienPlannerV2(
            planning_world,
            "scene-0-ds_fetch_quasi_static_gripper_link",
            joint_vel_limits=np.ones(9) * self.joint_vel_limits,
            joint_acc_limits=np.ones(9) * self.joint_acc_limits,
        )

        planner.set_base_pose(mplib.Pose(self.base_pose.p, self.base_pose.q))
        return planner

    def follow_path(self, result, refine_steps: int = 0):
        qpos_final = result["position"][-1]
        qpos_dict_final = {}
        for idx, q in zip(self.planner.move_group_joint_indices, qpos_final):
            joint_name = self.planner.user_joint_names[idx]
            qpos_dict_final[joint_name] = q

        n_step = result["position"].shape[0]
        for i in range(n_step + refine_steps):
            arm_action = (
                self.env_agent.controller.controllers["arm"].qpos[0].cpu().numpy()
            )

            qpos = result["position"][min(i, n_step - 1)]

            qpos_dict = {}

            for idx, q in zip(self.planner.move_group_joint_indices, qpos):
                joint_name = self.planner.user_joint_names[idx]
                qpos_dict[joint_name] = q

            for n, joint_name in enumerate(
                self.env_agent.controller.controllers["arm"].config.joint_names
            ):
                arm_action[n] = qpos_dict[f"scene-0-ds_fetch_quasi_static_{joint_name}"]

            assert self.control_mode == "pd_joint_pos"

            body_action = (
                self.env_agent.controller.controllers["body"].qpos[0].cpu().numpy()
            )
            body_action[2] = qpos_dict["scene-0-ds_fetch_quasi_static_torso_lift_joint"]
            body_action[0] = body_action[1] = 0.0

            base_action = (
                self.env_agent.controller.controllers["base"].qpos[0].cpu().numpy()
            )
            base_action[0] = qpos_dict[
                "scene-0-ds_fetch_quasi_static_root_x_axis_joint"
            ]

            action = np.hstack(
                [arm_action, self.gripper_state, body_action, base_action]
            )
            print("arm Action:", np.round(arm_action, 4))
            print("body Action:", np.round(body_action, 4))
            print("base Action:", np.round(base_action, 4))
            print("Full: ", np.round(self.robot.get_qpos().cpu().numpy()[0], 4))
            obs, reward, terminated, truncated, info = self.env.step(action)

            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()

        # REFINEMENT!
        # We refine only x position and lift at the end of the trajectory
        passed_refine_steps = 0
        last_lift_poses = deque(maxlen=10)
        last_x_base_poses = deque(maxlen=10)
        last_lift_vels = deque(maxlen=10)
        last_x_base_vels = deque(maxlen=10)
        print("==== REFINEMENT ====")
        while not self.check_body_base_close_to_target(qpos_dict_final):
            if (
                (len(last_lift_vels) > 4 and np.std(last_lift_vels) < 1e-3)
                and (len(last_x_base_vels) > 4 and np.std(last_x_base_vels) < 1e-3)
                and (len(last_lift_poses) > 4 and np.std(last_lift_poses) < 1e-3)
                and (len(last_x_base_poses) > 4 and np.std(last_x_base_poses) < 1e-3)
            ):
                # robot is stuck
                print("Robot is stuck")
                break

            body_action = (
                self.env_agent.controller.controllers["body"].qpos[0].cpu().numpy()
            )
            body_action[2] = qpos_dict_final[
                "scene-0-ds_fetch_quasi_static_torso_lift_joint"
            ]
            body_action[0] = body_action[1] = 0.0

            base_action = (
                self.env_agent.controller.controllers["base"].qpos[0].cpu().numpy()
            )
            base_action[0] = qpos_dict_final[
                "scene-0-ds_fetch_quasi_static_root_x_axis_joint"
            ]

            last_lift_poses.append(
                self.env_agent.controller.controllers["body"].qpos[0].cpu().numpy()[2]
            )
            last_x_base_poses.append(
                self.env_agent.controller.controllers["base"].qpos[0].cpu().numpy()[0]
            )

            last_lift_vels.append(
                self.env_agent.controller.controllers["body"].qvel[0].cpu().numpy()[2]
            )
            last_x_base_vels.append(
                self.env_agent.controller.controllers["base"].qvel[0].cpu().numpy()[0]
            )

            action = np.hstack(
                [arm_action, self.gripper_state, body_action, base_action]
            )
            print("arm Action:", np.round(arm_action, 4))
            print("body Action:", np.round(body_action, 4))
            print("base Action:", np.round(base_action, 4))
            print("Full: ", np.round(self.robot.get_qpos().cpu().numpy()[0], 4))
            obs, reward, terminated, truncated, info = self.env.step(action)
            passed_refine_steps += 1
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()

        return obs, reward, terminated, truncated, info

    def check_body_base_close_to_target(self, target_dict, eps=1e-2):
        body_qpos = (
            self.env_agent.controller.controllers["body"].qpos[0].cpu().numpy()[2]
        )
        target_lift_joint_height = target_dict[
            "scene-0-ds_fetch_quasi_static_torso_lift_joint"
        ]

        base_x = self.env_agent.controller.controllers["base"].qpos[0].cpu().numpy()[0]
        target_base_x = target_dict["scene-0-ds_fetch_quasi_static_root_x_axis_joint"]

        robot_qpos = self.robot.get_qpos().cpu().numpy()[0]
        arm_pos = robot_qpos[
            self.env_agent.controller.controllers["arm"]
            .active_joint_indices.cpu()
            .numpy()
        ]
        target_arm_pos = np.array(
            [
                target_dict["scene-0-ds_fetch_quasi_static_shoulder_pan_joint"],
                target_dict["scene-0-ds_fetch_quasi_static_shoulder_lift_joint"],
                target_dict["scene-0-ds_fetch_quasi_static_upperarm_roll_joint"],
                target_dict["scene-0-ds_fetch_quasi_static_elbow_flex_joint"],
                target_dict["scene-0-ds_fetch_quasi_static_forearm_roll_joint"],
                target_dict["scene-0-ds_fetch_quasi_static_wrist_flex_joint"],
                target_dict["scene-0-ds_fetch_quasi_static_wrist_roll_joint"],
            ]
        )
        return (
            np.allclose(body_qpos, target_lift_joint_height, atol=eps)
            and np.allclose(base_x, target_base_x, atol=eps)
            and np.allclose(arm_pos, target_arm_pos, atol=eps)
        )

    def open_gripper(self):
        self.gripper_state = OPEN
        qpos = self.robot.get_qpos()[0, :-2].cpu().numpy()
        for i in range(6):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state])
            else:
                action = np.hstack([qpos, qpos * 0, self.gripper_state])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def close_gripper(self, t=6, gripper_state=CLOSED):
        self.gripper_state = gripper_state
        arm_action = self.env_agent.controller.controllers["arm"].qpos[0].cpu().numpy()
        body_action = (
            self.env_agent.controller.controllers["body"].qpos[0].cpu().numpy()
        )
        base_vel = np.array([0, 0])
        base_action = np.zeros_like(
            self.env_agent.controller.controllers["base"].qpos[0].cpu().numpy()
        )

        for i in range(t):
            if self.control_mode == "pd_joint_pos":
                # action = np.hstack([arm_action, self.gripper_state, body_action, base_vel])
                action = np.hstack(
                    [arm_action, self.gripper_state, body_action, base_action]
                )
            else:
                raise NotImplementedError
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info


class FetchMotionPlanningSapienSolver(PandaArmMotionPlanningSapienSolver):
    # Default cap on the refinement loop of follow_forward_path_w_refinement; the
    # inherited cup/takeitback planners rely on it. An oracle that would rather fail
    # a stage than spend 200 steps converging passes `max_refine_steps=` instead.
    MAX_REFINE_STEPS = 200

    # Continuous joints in Fetch robot (indices relative to move_group_joint_indices)
    # These will be fixed during planning to avoid "continuous revolute joint" error
    CONTINUOUS_JOINT_NAMES = [
        "root_z_rotation_joint",
        "upperarm_roll_joint",
        "forearm_roll_joint",
        "wrist_roll_joint",
    ]

    # How far the executed screw plan may end from its goal in static_manipulation
    # and lift_hand before plan_screw reports a miss and static_manipulation falls
    # back to plan_pose: (metres, radians). Not passed by the base primitives, whose
    # screw plans are executed only in part (see rotate_base_z).
    ARM_SCREW_GOAL_TOLERANCE = (0.02, 0.10)

    def __init__(self, *args, max_refine_steps: int | None = None, **kwargs):
        """As the parent, plus `max_refine_steps` (keyword-only).

        Args:
            max_refine_steps: cap on the refinement loop after a manipulation path,
                per instance; None keeps the class default `MAX_REFINE_STEPS` (200).
                The memory-task oracles pass a small number so a stage that cannot
                converge fails inside the horizon instead of eating it.
        """
        env = args[0] if args else kwargs["env"]
        # Before super().__init__: the parent assigns `self.elapsed_steps = 0`
        # (PandaArmMotionPlanningSolverV2.__init__), which the property below routes
        # to the guard, so the guard has to exist first.
        self._guard = StepGuard(env)
        super().__init__(*args, **kwargs)
        self.max_refine_steps = (
            self.MAX_REFINE_STEPS if max_refine_steps is None else int(max_refine_steps)
        )

    @property
    def elapsed_steps(self) -> int:
        """Control steps this solver has taken through `env.step`, all of them."""
        return self._guard.elapsed_steps

    @elapsed_steps.setter
    def elapsed_steps(self, value: int) -> None:
        self._guard.elapsed_steps = int(value)

    @property
    def truncated(self) -> bool:
        """True once any `env.step` reported `truncated` — the episode is over."""
        return self._guard.truncated

    def _step(self, action):
        """The one `env.step` of this class: counted, latched, printed, rendered."""
        obs, reward, terminated, truncated, info = self._guard.step(action)
        if self.print_env_info:
            print(f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}")
        if self.vis:
            self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def _stopped_by_horizon(self, where: str) -> bool:
        """After a step: True (and one line on stdout) if the episode just ended."""
        if not self._guard.truncated:
            return False
        print(f"[solver] episode truncated at step {self.elapsed_steps}; stopping {where}", flush=True)
        return True

    def _final_qpos_dict(self, result) -> dict:
        """The plan's last knot as `{user_joint_name: q}` over the move group."""
        return {
            self.planner.user_joint_names[idx]: q
            for idx, q in zip(self.planner.move_group_joint_indices, result["position"][-1])
        }

    def _report(self, stage: str, **fields) -> None:
        """One diagnostic line per executed plan, on stdout and — when a `PlannerLogger`
        is anywhere in the env chain — as a `solver` event in events.jsonl (with
        `stage=`), so the trace names the plan next to the step. Deliberately not the
        oracle's `say()`: the solver must not import from the task planners."""
        line = f"[{stage}] " + " ".join(f"{k}={v}" for k, v in fields.items())
        print(line, flush=True)
        log_event = find_log_event(self.env)
        if log_event is not None:
            log_event("solver", line, stage=stage, **fields)

    def setup_planner(self, *args, **kwargs):
        planned_articulation = self._sim_scene.get_all_articulations()[0]
        planning_world = SapienPlanningWorldV2(
            self._sim_scene,
            [planned_articulation],
            disable_actors_collision=self.disable_actors_collision,
        )

        # Get joint info for debugging
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]

        # Create planner first to get joint indices
        planner = SapienPlannerV2(
            planning_world,
            f"scene-0-{self.robot.name}_gripper_link",
            joint_vel_limits=np.ones(11) * self.joint_vel_limits,
            joint_acc_limits=np.ones(11) * self.joint_acc_limits,
        )

        # Find indices of continuous joints in move_group_joint_indices
        user_joint_names = planner.user_joint_names
        move_group_joint_indices = planner.move_group_joint_indices

        fixed_joint_indices = []
        for i, joint_idx in enumerate(move_group_joint_indices):
            if user_joint_names[joint_idx] in self.CONTINUOUS_JOINT_NAMES:
                fixed_joint_indices.append(i)
                print(
                    f"Fixed continuous joint: {user_joint_names[joint_idx]} at index {i}"
                )

        # Store for later use in planning
        self._fixed_joint_indices = fixed_joint_indices

        planner.set_base_pose(mplib.Pose(self.base_pose.p, self.base_pose.q))
        return planner

    def rotate_base_z(
        self,
        new_direction,
        n_init_qpos=20,
        dry_run=False,
        rotate_recalculation_enabled=True,
    ):
        """Turn the base to face `new_direction` (world xy); -1 or the 5-tuple.

        A yaw-only path (K51/D12): a trapezoid over the base yaw joint from the
        planner's velocity/acceleration limits for that joint, every other joint
        frozen where it is, and the robot *as it stands* — arm posture and whatever is
        attached in the planning world — checked for collision at `SWEEP_SAMPLES`
        poses along the arc, the short way first, then the long way round; both
        blocked → a named `-1` (`rotation sweep hits <link>↔<obj> at yaw=…`).

        Why not `plan_screw` (as this method did until 2026-08-18): the screw plan
        rotated the TCP with the whole body and `follow_rotation` executed only its
        base-yaw velocity, so two-thirds of the planned turn was arm motion the robot
        never made — and the collision check ran on that phantom motion (torso free:
        hid a real cup-against-the-wall sweep; torso masked: refused a drive on a
        collision the robot would not have had). The base channel is normalized
        (`pd_base_vel`, action 1 = `upper[1]` rad/s), so `velocity[:, 2]` is stored in
        channel units — see `base_yaw.yaw_path`. `follow_rotation` is unchanged. A
        second, residual turn is planned only if the executed one missed by ≥ 1e-2 rad.

        `n_init_qpos` is kept for signature compatibility; nothing samples IK here.
        """
        if self.truncated:
            return self._guard.last_step
        assert np.isclose(new_direction[2], 0)
        angle = self._yaw_to(new_direction)
        if np.abs(angle) < 1e-2:
            return self.idle_steps(t=1)

        result = self._plan_base_yaw(angle)
        if result["status"] != "Success":
            print(result["status"])
            self.render_wait()
            return -1
        if dry_run:
            return result
        self.render_wait()
        res = self.follow_rotation(result)
        if self.truncated or not rotate_recalculation_enabled:
            return res

        # The executed turn is a velocity command tracked by a PD controller; take
        # up the residual, if any, with one more yaw-only path.
        residual = self._yaw_to(new_direction)
        if np.abs(residual) < 1e-2:
            return res
        result = self._plan_base_yaw(residual)
        if result["status"] != "Success":
            print(result["status"])
            self.render_wait()
            return -1
        return self.follow_rotation(result)

    def _yaw_to(self, new_direction) -> float:
        """Signed angle (rad) from the base's x axis to `new_direction`, world xy."""
        base_x_axis = self.base_env.agent.base_link.pose.sp.to_transformation_matrix()[:3, 0]
        cosang = np.dot(new_direction, base_x_axis) / np.linalg.norm(base_x_axis) / np.linalg.norm(new_direction)
        angle = float(np.arccos(np.clip(cosang, -1, 1)))
        if np.cross(base_x_axis, new_direction)[2] < 0:
            angle = -angle
        return angle

    def _plan_base_yaw(self, angle: float) -> dict:
        """The yaw-only path for a turn of `angle`, or a `status` naming the obstacle.

        Timing from the planner's limits for the yaw joint (`joint_vel_limits[2]`,
        `joint_acc_limits[2]` — 0.9 rad/s, 0.9 rad/s² as constructed); rate stored in
        the base controller's normalized channel units. Collision: the current full
        qpos with the yaw substituted, the planning world's robot↔env and self checks
        at `sweep_samples(arc)` poses per candidate arc, with whatever is held drawn
        where physics has it — `HeldBodies` (`base_yaw.py`) both re-syncs the planning
        world from the simulator and undoes mplib 0.2.1's missing base pose; without
        the sync the residual turn below, planned after `follow_rotation` has already
        swung the base, would sweep the arc against a cup metres from the hand.

        Pairs already in contact at the start pose (the held cup grazing a counter, a
        self-touch the SRDF does not list) are subtracted: they are not this turn's
        doing, and left in they would refuse every turn the robot ever asks for.
        """
        planner = self.planner
        world = planner.planning_world
        art = world.get_planned_articulations()[0]
        # Sync first — `HeldBodies` calls `planner.update_from_simulation()` before it
        # reads a single pose (why: its docstring; attachments survive the sync). The
        # base pose is the planning articulation's own: the identity when the world
        # folds the robot's pose into the root joints (K53, `SapienPlanningWorldV2`),
        # in which case HeldBodies' frame correction is a no-op and only its sync
        # matters; mplib's base pose otherwise.
        held = HeldBodies(world, art.get_base_pose(), planner.update_from_simulation)
        qpos = self.robot.get_qpos().cpu().numpy()[0].astype(np.float64)
        yaw_index = list(planner.move_group_joint_indices).index(2)  # root_z_rotation_joint

        def pairs_at(offset: float) -> set:
            q = qpos.copy()
            q[2] += offset
            art.set_qpos(planner.fold_qpos(q), True)
            held.place()
            pairs = list(world.check_robot_collision()) + list(world.check_self_collision())
            return {f"{c.link_name1}<->{c.link_name2}" for c in pairs}

        def colliding_at(offset: float):
            return new_contacts(pairs_at(offset), baseline)

        try:
            baseline = pairs_at(0.0)  # the robot as it stands: not the turn's doing
            chosen, why = sweep_yaw(angle, colliding_at)
        finally:
            art.set_qpos(planner.fold_qpos(qpos), True)
            held.restore()
        if chosen is None:
            if baseline:
                why += f" (already touching before the turn, ignored: {', '.join(sorted(baseline)[:2])})"
            self._report("rotate_base_z", short_way=round(float(angle), 3), chosen=None, refused=why)
            return {"status": why}

        base_controller = self.env_agent.controller.controllers["base"]
        rate_scale = (
            float(base_controller.config.upper[1]) if base_controller.config.normalize_action else 1.0
        )
        result = yaw_path(
            qpos[planner.move_group_joint_indices],
            yaw_index,
            chosen,
            v_max=float(planner.joint_vel_limits[yaw_index]),
            a_max=float(planner.joint_acc_limits[yaw_index]),
            dt=self.base_env.control_timestep,
            rate_scale=rate_scale,
        )
        result["short_way"] = float(angle)
        self._report(
            "rotate_base_z",
            short_way=round(float(angle), 3),
            chosen=round(float(chosen), 3),
            knots=int(result["position"].shape[0]),
            dur=round(float(result["duration"]), 2),
            held=held.names,
        )
        return result

    def drive_base(self, target_pos=None, target_view_vec=None, freeze_arm: bool = False):
        """Turn toward `target_pos`, drive to it, then turn to face `target_view_vec`.

        `freeze_arm` is passed straight to `move_base_forward`; see
        `BASE_ONLY_PLAN_MASK`. Default False, so every existing caller plans exactly
        as before.
        """
        if self.truncated:
            return self._guard.last_step
        if target_pos is None and target_view_vec is None:
            # Used to fall through to `return res` with `res` unbound
            # (UnboundLocalError); a call with nothing to do is a failed plan.
            print("[solver] drive_base: neither target_pos nor target_view_vec given; nothing to plan")
            return -1
        if not target_pos is None:
            moving_direction = target_pos - self.base_env.agent.base_link.pose.sp.p
            moving_direction[2] = 0.0

            if np.linalg.norm(moving_direction) < 1e-2:
                res = self.idle_steps(t=1)
                if res == -1:
                    return res
                self.planner.update_from_simulation()

            else:
                res = self.rotate_base_z(moving_direction)
                if res == -1:
                    return res
                self.planner.update_from_simulation()

                res = self.move_base_forward(target_pos, n_init_qpos=100, freeze_arm=freeze_arm)
                if res == -1:
                    return res
                self.planner.update_from_simulation()

        # view_direction = target_view_pos.p - self.base_env.agent.base_link.pose.sp.p
        if not target_view_vec is None:
            res = self.rotate_base_z(target_view_vec)
        return res

    def move_base_forward(self, new_base_pose, n_init_qpos=20, dry_run=False,
                          freeze_arm: bool = False):
        """Drive the base to `new_base_pose` (xy); -1 or the 5-tuple.

        `freeze_arm=True` swaps the screw plans' `masked_joints` from
        `BASE_PLAN_MASK` to `BASE_ONLY_PLAN_MASK` — base x, y, yaw free and every
        other joint frozen. Opt-in on purpose: the default is what the three shipped
        oracles' numbers and recordings were taken with. See `BASE_ONLY_PLAN_MASK`
        for the measurement that motivates it.
        """
        if self.truncated:
            return self._guard.last_step
        mask = BASE_ONLY_PLAN_MASK if freeze_arm else BASE_PLAN_MASK
        tcp_pose = self.base_env.agent.tcp.pose.sp
        base_link_pose = self.base_env.agent.base_link.pose.sp
        delta = new_base_pose - base_link_pose.p
        delta[2] = 0.0
        target_tcp_pose = sapien.Pose(p=tcp_pose.p + delta, q=tcp_pose.q)

        if self.grasp_pose_visual is not None:
            self.grasp_pose_visual.set_pose(target_tcp_pose)
        target_tcp_pose = mplib.Pose(p=target_tcp_pose.p, q=target_tcp_pose.q)
        # No goal_tolerance: executed in part — follow_moving_forward drives the base
        # forward only — so the plan's FK endpoint is not what the robot will do
        # (measured in the container: ~4 cm of drift while the base still arrived).
        result = self.planner.plan_screw(
            mplib.Pose(p=target_tcp_pose.p, q=target_tcp_pose.q),
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
            masked_joints=mask,
        )

        self.render_wait()

        if result["status"] != "Success":
            print(result["status"])
            self.render_wait()
            return -1
        res = self.follow_moving_forward(result)
        if self.truncated:
            return res

        result = self.planner.plan_screw(
            mplib.Pose(p=target_tcp_pose.p, q=target_tcp_pose.q),
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
            masked_joints=mask,
        )

        if result["status"] != "Success":
            print(result["status"])
            self.render_wait()
            return -1

        if dry_run:
            return result

        return self.follow_moving_forward(result)

    def move_base_x_and_manipulation(self, target_tcp_pose, n_init_qpos=20):
        # Axis semantics under the K53 fold: the planning articulation's root
        # joints are folded into WORLD axes (root_x/root_y are world x/y, the
        # base pose is the identity), so the mask below frees index 0 = world x
        # and `fixed_joint_indices=[1]` pins world y — not the robot's own
        # forward/lateral axes as the method name suggests. Identical only when
        # the base yaw is 0; at the memory tasks' yaw = pi/2 "x" here is the
        # robot's lateral axis. Inherited, unused by the memory-task oracles,
        # left as is (T5 report; converting the mask by the base yaw is the fix
        # if anyone starts calling it).
        if self.truncated:
            return self._guard.last_step
        if self.grasp_pose_visual is not None:
            self.grasp_pose_visual.set_pose(target_tcp_pose)
        target_tcp_pose = mplib.Pose(p=target_tcp_pose.p, q=target_tcp_pose.q)

        move_x_and_manipulate = [
            False,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ]
        result = self.planner.plan_pose(
            target_tcp_pose,
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
            # use_point_cloud=self.use_point_cloud,
            wrt_world=True,
            verbose=True,
            planning_time=PLANNING_TIME,
            rrt_range=0.1,
            simplify=True,
            mask=move_x_and_manipulate,
            fixed_joint_indices=[1],
            n_init_qpos=n_init_qpos,
        )

        if result["status"] != "Success":
            print(result["status"])
            self.render_wait()
            return -1
        self.render_wait()

        res = self.follow_forward_path_w_refinement(result)
        self.planner.update_from_simulation()
        return self.static_manipulation(target_tcp_pose, n_init_qpos=n_init_qpos)

    def static_manipulation(
        self, target_tcp_pose, n_init_qpos=20, disable_lift_joint: bool = False
    ):
        """Move the TCP to `target_tcp_pose` with the base held still; -1 or the 5-tuple.

        Screw plan first, gated on its FK goal error (`ARM_SCREW_GOAL_TOLERANCE`) —
        the seed-3 hover of the burner oracle got a "Success" screw plan that ended
        12.9 cm from the goal and then spent 2000 steps refining towards it — then
        RRTConnect (`plan_pose`) as the fallback. After execution one diagnostic
        line, `[static_manipulation] plan=… iters=… knots=… dur=… exec=… refine=…
        reached=… tcp_err=… goal_error=…`, on stdout and in events.jsonl (`iters`
        is the screw plan's Jacobian steps, None for rrt; `knots` the control steps
        the TOPP trajectory takes, `dur` its seconds).
        """
        if self.truncated:
            return self._guard.last_step
        if self.grasp_pose_visual is not None:
            self.grasp_pose_visual.set_pose(
                sapien.Pose(p=target_tcp_pose.p, q=target_tcp_pose.q)
            )
        target_tcp_pose = mplib.Pose(p=target_tcp_pose.p, q=target_tcp_pose.q)
        only_manipulate = [
            True,
            True,
            True,
            disable_lift_joint,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ]
        fixed_joint_indices = [0, 1, 2, 3] if disable_lift_joint else [0, 1, 2]

        plan = "screw"
        result = self.planner.plan_screw(
            mplib.Pose(p=target_tcp_pose.p, q=target_tcp_pose.q),
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
            masked_joints=~np.array(only_manipulate),
            goal_tolerance=self.ARM_SCREW_GOAL_TOLERANCE,
        )

        screw_status = result["status"]
        if screw_status != "Success":
            plan = "rrt"
            result = self.planner.plan_pose(
                target_tcp_pose,
                self.robot.get_qpos().cpu().numpy()[0],
                time_step=self.base_env.control_timestep,
                # use_point_cloud=self.use_point_cloud,
                wrt_world=True,
                verbose=self.verbose,
                planning_time=2 * PLANNING_TIME,
                rrt_range=0.1,
                simplify=True,
                mask=only_manipulate,
                fixed_joint_indices=fixed_joint_indices,
                n_init_qpos=n_init_qpos,
            )

            if result["status"] != "Success":
                # Nothing executed; say why both planners refused, in the same
                # place the executed plans report, so the trace names the reason
                # next to the oracle's "FAILED: <stage>".
                self._report("static_manipulation", plan=None, screw=screw_status, rrt=result["status"])
                self.render_wait()
                return -1

        self.render_wait()

        knots = int(result["position"].shape[0])
        before = self.elapsed_steps
        out = self.follow_forward_path_w_refinement(result, refine=True)
        # Path following stops early only on truncation, so what was executed is the
        # smaller of the path and the steps taken; the rest was refinement.
        executed = min(knots, self.elapsed_steps - before)
        tcp = self.base_env.agent.tcp.pose.sp
        tcp_pos, tcp_rot = pose_error(target_tcp_pose.p, target_tcp_pose.q, tcp.p, tcp.q)
        goal_error = result.get("goal_error")
        duration = result.get("duration")
        self._report(
            "static_manipulation",
            plan=plan,
            iters=result.get("iterations"),
            knots=knots,
            dur=None if duration is None else round(float(duration), 2),
            exec=executed,
            refine=self.elapsed_steps - before - executed,
            reached=bool(self.check_body_base_close_to_target(self._final_qpos_dict(result))),
            tcp_err=f"{tcp_pos:.3f}m/{np.degrees(tcp_rot):.1f}deg",
            goal_error=(
                f"{goal_error[0]:.3f}m/{np.degrees(goal_error[1]):.1f}deg" if goal_error else None
            ),
            **({"screw": screw_status} if plan == "rrt" else {}),
        )
        return out

    def move_to_pose_with_screw_static_body(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):
        if self.truncated:
            return self._guard.last_step
        pose = to_sapien_pose(pose)
        # try screw two times before giving up
        if self.grasp_pose_visual is not None:
            self.grasp_pose_visual.set_pose(pose)
        pose = sapien.Pose(p=pose.p, q=pose.q)
        result = self.planner.plan_screw(
            mplib.Pose(pose.p, pose.q),
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
            verbose=True,
            masked_joints=[False, False, False, False] + [True] * 11,
            # use_point_cloud=self.use_point_cloud,
        )
        if result["status"] != "Success":
            result = self.planner.plan_screw(
                mplib.Pose(pose.p, pose.q),
                self.robot.get_qpos().cpu().numpy()[0],
                time_step=self.base_env.control_timestep,
                masked_joints=[False, False, False, False] + [True] * 11,
                # # use_point_cloud=self.use_point_cloud,
            )
            if result["status"] != "Success":
                print(result["status"])
                self.render_wait()
                return -1
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    def lift_hand(self, delta_h=0.0, dry_run: bool = False, refine_steps: int = 0):
        if self.truncated:
            return self._guard.last_step
        cur_pose = self.base_env.agent.tcp.pose.sp
        taget_pose = mplib.Pose(
            p=cur_pose.p + np.array([0.0, 0.0, delta_h]), q=cur_pose.q
        )
        # The whole plan is executed (follow_path), so the FK goal gate applies.
        result = self.planner.plan_screw(
            taget_pose,
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
            verbose=True,
            goal_tolerance=self.ARM_SCREW_GOAL_TOLERANCE,
            # use_point_cloud=self.use_point_cloud,
        )
        if result["status"] != "Success":
            print(result["status"])
            self.render_wait()
            return -1
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    def move_forward_delta(self, delta=0.0, dry_run: bool = False):
        cur_pose = self.base_env.agent.base_link.pose.sp
        direction = cur_pose.to_transformation_matrix()[:3, 0]
        direction[2] = 0.0
        shift = direction * delta
        taget_pose = mplib.Pose(p=cur_pose.p + shift, q=cur_pose.q)
        result = self.move_base_forward(taget_pose.p, dry_run=dry_run)
        return result

    def rotate_z_delta(
        self,
        delta=0.0,
        dry_run: bool = False,
        rotate_recalculation_enabled: bool = True,
    ):
        cur_pose = self.base_env.agent.base_link.pose.sp
        direction = cur_pose.to_transformation_matrix()[:3, 0]
        direction[2] = 0.0

        rot_matrix = euler2mat(0, 0, delta)

        new_direction = rot_matrix @ direction

        result = self.rotate_base_z(
            new_direction,
            dry_run=dry_run,
            rotate_recalculation_enabled=rotate_recalculation_enabled,
        )

        return result

    def follow_rotation(self, result, refine_steps: int = 0):
        n_step = result["position"].shape[0]
        for i in range(n_step + refine_steps):
            arm_action = (
                self.env_agent.controller.controllers["arm"].qpos[0].cpu().numpy()
            )
            body_action = (
                self.env_agent.controller.controllers["body"].qpos[0].cpu().numpy()
            )
            body_action[0] = body_action[1] = 0.0
            base_action = np.array([0.0, 0.0])

            qvel = result["velocity"][min(i, n_step - 1)]

            base_action[1] = qvel[2]

            action = np.hstack(
                [arm_action, self.gripper_state, body_action, base_action]
            )
            if self.verbose:
                print("base Action:", np.round(base_action, 4))
                print("Full: ", np.round(self.robot.get_qpos().cpu().numpy()[0], 4))
            obs, reward, terminated, truncated, info = self._step(action)
            if self._stopped_by_horizon("follow_rotation"):
                break

        return obs, reward, terminated, truncated, info

    def follow_moving_forward(self, result, refine_steps: int = 0):
        n_step = result["position"].shape[0]
        base_direction = self.env_agent.base_link.pose.sp.to_transformation_matrix()[
            :3, 0
        ]
        root_to_world = self.env_agent.robot.root_pose.sp.to_transformation_matrix()[
            :3, :3
        ]
        for i in range(n_step + refine_steps):
            arm_action = (
                self.env_agent.controller.controllers["arm"].qpos[0].cpu().numpy()
            )
            body_action = (
                self.env_agent.controller.controllers["body"].qpos[0].cpu().numpy()
            )
            body_action[0] = body_action[1] = 0.0
            base_action = np.array([0.0, 0.0])

            qvel = result["velocity"][min(i, n_step - 1)]
            base_vel = np.array([qvel[0], qvel[1], 0.0])
            base_vel_wrt_world = root_to_world @ base_vel
            is_forward = np.dot(base_vel_wrt_world, base_direction)
            base_action[0] = is_forward

            action = np.hstack(
                [arm_action, self.gripper_state, body_action, base_action]
            )
            if self.verbose:
                print("base Action:", np.round(base_action, 4))
                print("Full: ", np.round(self.robot.get_qpos().cpu().numpy()[0], 4))
            obs, reward, terminated, truncated, info = self._step(action)
            if self._stopped_by_horizon("follow_moving_forward"):
                break

        return obs, reward, terminated, truncated, info

    def follow_path(self, result, refine_steps: int = 0, refine: bool = False):
        return self.follow_forward_path_w_refinement(result, refine)

    def follow_forward_path_w_refinement(
        self, result, refine: bool = False, static=False
    ):
        qpos_dict_final = self._final_qpos_dict(result)
        n_step = result["position"].shape[0]

        for i in range(n_step):
            arm_action = (
                self.env_agent.controller.controllers["arm"].qpos[0].cpu().numpy()
            )

            qpos = result["position"][min(i, n_step - 1)]
            qvel = result["velocity"][min(i, n_step - 1)]

            qpos_dict = {}

            for idx, q in zip(self.planner.move_group_joint_indices, qpos):
                joint_name = self.planner.user_joint_names[idx]
                qpos_dict[joint_name] = q

            for n, joint_name in enumerate(
                self.env_agent.controller.controllers["arm"].config.joint_names
            ):
                arm_action[n] = qpos_dict[f"scene-0-{self.robot.name}_{joint_name}"]

            assert self.control_mode == "pd_joint_pos"

            body_action = np.zeros_like(
                self.env_agent.controller.controllers["body"].qpos[0].cpu().numpy()
            )
            body_action[2] = qpos_dict[f"scene-0-{self.robot.name}_torso_lift_joint"]

            base_direction = (
                self.env_agent.base_link.pose.sp.to_transformation_matrix()[:3, 0]
            )
            root_to_world = (
                self.env_agent.robot.root_pose.sp.to_transformation_matrix()[:3, :3]
            )
            base_vel = np.array([qvel[0], qvel[1], 0.0])
            base_vel_wrt_world = root_to_world @ base_vel
            is_forward = np.dot(base_vel_wrt_world, base_direction)

            base_action = np.array([0.0, 0.0])
            base_action[0] = is_forward

            action = np.hstack(
                [arm_action, self.gripper_state, body_action, base_action]
            )
            if self.verbose:
                print("arm Action:", np.round(arm_action, 4))
                print("body Action:", np.round(body_action, 4))
                print("base Action:", np.round(base_action, 4))
                print("qpos: ", np.round(self.robot.get_qpos().cpu().numpy()[0], 4))
            obs, reward, terminated, truncated, info = self._step(action)
            if self._stopped_by_horizon("follow_forward_path_w_refinement"):
                break

        if refine and not self.truncated:
            # REFINEMENT!
            passed_refine_steps = 0
            last_lift_poses = deque(maxlen=10)
            last_x_base_poses = deque(maxlen=10)
            last_lift_vels = deque(maxlen=10)
            last_x_base_vels = deque(maxlen=10)
            if self.verbose:
                print("==== REFINEMENT ====")

            while not self.check_body_base_close_to_target(qpos_dict_final):
                why = refine_should_stop(
                    passed_refine_steps,
                    self.max_refine_steps,
                    last_lift_poses,
                    last_lift_vels,
                    last_x_base_poses,
                    last_x_base_vels,
                )
                if why == "stuck":
                    print("Robot is stuck")
                    break
                if why == "max":
                    print(f"Reached max refining steps ({self.max_refine_steps})!")
                    break

                body_action = np.zeros_like(
                    self.env_agent.controller.controllers["body"].qpos[0].cpu().numpy()
                )
                body_action[2] = qpos_dict_final[
                    f"scene-0-{self.robot.name}_torso_lift_joint"
                ]
                body_action[0] = body_action[1] = 0.0

                base_action = np.array([0.0, 0.0])

                last_lift_poses.append(
                    self.env_agent.controller.controllers["body"]
                    .qpos[0]
                    .cpu()
                    .numpy()[2]
                )
                last_x_base_poses.append(
                    self.env_agent.controller.controllers["base"]
                    .qpos[0]
                    .cpu()
                    .numpy()[0]
                )

                last_lift_vels.append(
                    self.env_agent.controller.controllers["body"]
                    .qvel[0]
                    .cpu()
                    .numpy()[2]
                )
                last_x_base_vels.append(
                    self.env_agent.controller.controllers["base"]
                    .qvel[0]
                    .cpu()
                    .numpy()[0]
                )

                action = np.hstack(
                    [arm_action, self.gripper_state, body_action, base_action]
                )
                if self.verbose:
                    print("arm Action:", np.round(arm_action, 4))
                    print("body Action:", np.round(body_action, 4))
                    print("base Action:", np.round(base_action, 4))
                    print("Full: ", np.round(self.robot.get_qpos().cpu().numpy()[0], 4))
                obs, reward, terminated, truncated, info = self._step(action)
                passed_refine_steps += 1
                if self._stopped_by_horizon("refinement"):
                    break

        return obs, reward, terminated, truncated, info

    def check_body_base_close_to_target(self, target_dict, eps=1e-2):
        body_qpos = (
            self.env_agent.controller.controllers["body"].qpos[0].cpu().numpy()[2]
        )
        target_lift_joint_height = target_dict[
            f"scene-0-{self.robot.name}_torso_lift_joint"
        ]

        base_xy = (
            self.env_agent.controller.controllers["base"].qpos[0].cpu().numpy()[0:2]
        )
        target_base = np.array(
            [
                target_dict[f"scene-0-{self.robot.name}_root_x_axis_joint"],
                target_dict[f"scene-0-{self.robot.name}_root_y_axis_joint"],
            ]
        )

        robot_qpos = self.robot.get_qpos().cpu().numpy()[0]
        arm_pos = robot_qpos[
            self.env_agent.controller.controllers["arm"]
            .active_joint_indices.cpu()
            .numpy()
        ]
        target_arm_pos = np.array(
            [
                target_dict[f"scene-0-{self.robot.name}_shoulder_pan_joint"],
                target_dict[f"scene-0-{self.robot.name}_shoulder_lift_joint"],
                target_dict[f"scene-0-{self.robot.name}_upperarm_roll_joint"],
                target_dict[f"scene-0-{self.robot.name}_elbow_flex_joint"],
                target_dict[f"scene-0-{self.robot.name}_forearm_roll_joint"],
                target_dict[f"scene-0-{self.robot.name}_wrist_flex_joint"],
                target_dict[f"scene-0-{self.robot.name}_wrist_roll_joint"],
            ]
        )
        return (
            np.allclose(body_qpos, target_lift_joint_height, atol=eps)
            and np.allclose(base_xy, target_base, atol=eps)
            and np.allclose(arm_pos, target_arm_pos, atol=eps)
        )

    def change_gripper_state(self, t=6, gripper_state=OPEN):
        if self.truncated:
            return self._guard.last_step
        self.gripper_state = gripper_state
        arm_action = self.env_agent.controller.controllers["arm"].qpos[0].cpu().numpy()
        body_action = (
            self.env_agent.controller.controllers["body"].qpos[0].cpu().numpy()
        )
        base_action = np.array([0, 0])

        for i in range(t):
            if self.control_mode == "pd_joint_pos":
                # action = np.hstack([arm_action, self.gripper_state, body_action, base_vel])
                action = np.hstack(
                    [arm_action, self.gripper_state, body_action, base_action]
                )
            else:
                raise NotImplementedError
            obs, reward, terminated, truncated, info = self._step(action)
            if self._stopped_by_horizon("change_gripper_state"):
                break
        return obs, reward, terminated, truncated, info

    def close_gripper(self, t=6):
        return self.change_gripper_state(t=t, gripper_state=CLOSED)

    def open_gripper(self, t=6):
        return self.change_gripper_state(t=t, gripper_state=OPEN)

    def idle_steps(self, t=20):
        if self.truncated:
            return self._guard.last_step
        arm_action = self.env_agent.controller.controllers["arm"].qpos[0].cpu().numpy()
        body_action = (
            self.env_agent.controller.controllers["body"].qpos[0].cpu().numpy()
        )
        base_action = np.array([0, 0])
        for i in range(t):
            if self.control_mode == "pd_joint_pos":
                # action = np.hstack([arm_action, self.gripper_state, body_action, base_vel])
                action = np.hstack(
                    [arm_action, self.gripper_state, body_action, base_action]
                )
            else:
                raise NotImplementedError
            obs, reward, terminated, truncated, info = self._step(action)
            if self._stopped_by_horizon("idle_steps"):
                break
        return obs, reward, terminated, truncated, info
