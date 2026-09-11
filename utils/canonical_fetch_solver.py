from collections import deque

import numpy as np

from mani_skill.examples.motionplanning.fetch.extand import (
    OPEN,
    FetchMotionPlanningSapienSolver as _FetchMotionPlanningSapienSolver,
)
from utils.canonical_fetch import ACTION_DIM, _base_cmd


class FetchMotionPlanningSapienSolver(_FetchMotionPlanningSapienSolver):
    """Fetch solver emitting canonical 13D actions only.

    Layout: 7 raw arm joint deltas/targets, gripper, 3 body targets,
    base forward velocity, base yaw velocity.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        shape = self.base_env.single_action_space.shape
        if shape != (ACTION_DIM,):
            raise ValueError(
                f"Canonical Fetch solver requires {ACTION_DIM}D actions, got {shape}"
            )

    def _action(self, arm_action, body_action, forward=0.0, yaw=0.0):
        action = np.hstack(
            [arm_action, self.gripper_state, body_action, _base_cmd(forward, yaw)]
        )
        assert action.shape == (ACTION_DIM,)
        return action

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
            yaw = result["velocity"][min(i, n_step - 1)][2]
            action = self._action(arm_action, body_action, yaw=yaw)
            if self.verbose:
                print("base Action:", np.round(action[-2:], 4))
                print("Full: ", np.round(self.robot.get_qpos().cpu().numpy()[0], 4))
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def follow_moving_forward(self, result, refine_steps: int = 0):
        n_step = result["position"].shape[0]
        base_direction = (
            self.env_agent.base_link.pose.sp.to_transformation_matrix()[:3, 0]
        )
        root_to_world = (
            self.env_agent.robot.root_pose.sp.to_transformation_matrix()[:3, :3]
        )
        for i in range(n_step + refine_steps):
            arm_action = (
                self.env_agent.controller.controllers["arm"].qpos[0].cpu().numpy()
            )
            body_action = (
                self.env_agent.controller.controllers["body"].qpos[0].cpu().numpy()
            )
            body_action[0] = body_action[1] = 0.0
            qvel = result["velocity"][min(i, n_step - 1)]
            world_velocity = root_to_world @ np.array([qvel[0], qvel[1], 0.0])
            forward = float(np.dot(world_velocity, base_direction))
            action = self._action(arm_action, body_action, forward=forward)
            if self.verbose:
                print("base Action:", np.round(action[-2:], 4))
                print("Full: ", np.round(self.robot.get_qpos().cpu().numpy()[0], 4))
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def follow_forward_path_w_refinement(
        self, result, refine: bool = False, static=False
    ):
        qpos_final = result["position"][-1]
        qpos_dict_final = {
            self.planner.user_joint_names[idx]: q
            for idx, q in zip(self.planner.move_group_joint_indices, qpos_final)
        }
        n_step = result["position"].shape[0]

        for i in range(n_step):
            arm_action = (
                self.env_agent.controller.controllers["arm"]
                .qpos[0]
                .cpu()
                .numpy()
                .copy()
            )
            qpos = result["position"][i]
            qvel = result["velocity"][i]
            qpos_dict = {
                self.planner.user_joint_names[idx]: q
                for idx, q in zip(self.planner.move_group_joint_indices, qpos)
            }
            for n, joint_name in enumerate(
                self.env_agent.controller.controllers["arm"].config.joint_names
            ):
                arm_action[n] = qpos_dict[f"scene-0-{self.robot.name}_{joint_name}"]

            assert self.control_mode == "pd_joint_pos"
            body_action = np.zeros_like(
                self.env_agent.controller.controllers["body"].qpos[0].cpu().numpy()
            )
            body_action[2] = qpos_dict[
                f"scene-0-{self.robot.name}_torso_lift_joint"
            ]
            root_to_world = (
                self.env_agent.robot.root_pose.sp.to_transformation_matrix()[:3, :3]
            )
            base_direction = (
                self.env_agent.base_link.pose.sp.to_transformation_matrix()[:3, 0]
            )
            world_velocity = root_to_world @ np.array([qvel[0], qvel[1], 0.0])
            forward = float(np.dot(world_velocity, base_direction))
            action = self._action(arm_action, body_action, forward=forward)
            if self.verbose:
                print("arm Action:", np.round(arm_action, 4))
                print("body Action:", np.round(body_action, 4))
                print("base Action:", np.round(action[-2:], 4))
                print("qpos: ", np.round(self.robot.get_qpos().cpu().numpy()[0], 4))
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()

        if refine:
            passed_refine_steps = 0
            last_lift_poses = deque(maxlen=10)
            last_x_base_poses = deque(maxlen=10)
            last_lift_vels = deque(maxlen=10)
            last_x_base_vels = deque(maxlen=10)
            if self.verbose:
                print("==== REFINEMENT ====")

            while not self.check_body_base_close_to_target(qpos_dict_final):
                if (
                    len(last_lift_vels) > 4
                    and np.std(last_lift_vels) < 1e-3
                    and len(last_x_base_vels) > 4
                    and np.std(last_x_base_vels) < 1e-3
                    and len(last_lift_poses) > 4
                    and np.std(last_lift_poses) < 1e-3
                    and len(last_x_base_poses) > 4
                    and np.std(last_x_base_poses) < 1e-3
                ):
                    print("Robot is stuck")
                    break
                if passed_refine_steps > self.MAX_REFINE_STEPS:
                    print("Reached max refining steps!")
                    break

                body_controller = self.env_agent.controller.controllers["body"]
                base_controller = self.env_agent.controller.controllers["base"]
                body_action = np.zeros_like(body_controller.qpos[0].cpu().numpy())
                body_action[2] = qpos_dict_final[
                    f"scene-0-{self.robot.name}_torso_lift_joint"
                ]
                last_lift_poses.append(body_controller.qpos[0].cpu().numpy()[2])
                last_x_base_poses.append(base_controller.qpos[0].cpu().numpy()[0])
                last_lift_vels.append(body_controller.qvel[0].cpu().numpy()[2])
                last_x_base_vels.append(base_controller.qvel[0].cpu().numpy()[0])

                action = self._action(arm_action, body_action)
                if self.verbose:
                    print("arm Action:", np.round(arm_action, 4))
                    print("body Action:", np.round(body_action, 4))
                    print("base Action:", np.round(action[-2:], 4))
                    print(
                        "Full: ", np.round(self.robot.get_qpos().cpu().numpy()[0], 4)
                    )
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

    def change_gripper_state(self, t=6, gripper_state=OPEN):
        self.gripper_state = gripper_state
        arm_action = self.env_agent.controller.controllers["arm"].qpos[0].cpu().numpy()
        body_action = (
            self.env_agent.controller.controllers["body"].qpos[0].cpu().numpy()
        )
        for _ in range(t):
            if self.control_mode != "pd_joint_pos":
                raise NotImplementedError
            obs, reward, terminated, truncated, info = self.env.step(
                self._action(arm_action, body_action)
            )
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def idle_steps(self, t=20):
        arm_action = self.env_agent.controller.controllers["arm"].qpos[0].cpu().numpy()
        body_action = (
            self.env_agent.controller.controllers["body"].qpos[0].cpu().numpy()
        )
        for _ in range(t):
            if self.control_mode != "pd_joint_pos":
                raise NotImplementedError
            obs, reward, terminated, truncated, info = self.env.step(
                self._action(arm_action, body_action)
            )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info
