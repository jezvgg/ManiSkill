"""End-effector delta-pose control in the frame of the MOBILE BASE.

Why ManiSkill's `pd_ee_delta_pose` is the wrong tool for `ds_fetch` (journal W32, 2026-09-08):
`PDEEPoseController` expresses the end-effector in the frame of `articulation.root`, and for a
robot on the three virtual base joints (`root_x_axis`, `root_y_axis`, `root_z_rotation`) that
root is the WORLD-FIXED dummy link at the dock where the robot was spawned. Once the base drives,
"move the hand 5 cm forward" means +x of wherever the robot started (measured: base turned 91°,
the inherited mode moved the hand along world x). Its IK (SAPIEN's pinocchio wrapper) ignores
joint limits and reports failure at millimetre residuals, and on the GPU its Jacobian solver dies
on a chain with uncontrolled ancestor joints (3.0.0b22).

`PDEEPoseBaseController` keeps the parent's plumbing (kinematics objects, action space, reset,
`use_target`, state) and replaces the rest:

* the reference frame is the link named by `root_link_name` (`base_link`; `torso_lift_link`
  would give arm-mount-relative control) — the target is composed there and transformed to the
  articulation-root frame for the IK (`to_root_frame`), which the parent never did;
* the rotation delta is a rotation VECTOR (axis-angle, radians) scaled by `rot_upper`; the
  translation is scaled by `pos_upper` and clipped per axis, the rotation by its norm — the plain
  inverse of what `ee_convert` writes;
* the IK is its own: weighted damped least squares within the joint limits (an active set for
  joints on a stop), `IK_ITERS` Newton steps from the current joints, the best iterate kept, plus
  a null-space posture term (`_nullspace_step`) toward `posture_hint` when one is set (the
  converter sets the source demonstration's joints) or toward the rest posture (`_rest_qpos`)
  otherwise — that is what resolves the 7-DoF arm's redundancy deterministically. CPU:
  `_solve_ik_pinocchio` (num_envs == 1, like upstream's CPU path); GPU: `_solve_ik_batched`.
  An unreachable target yields the closest feasible joints and a count in `ik_short`;
  a non-finite solution is refused and counted in `ik_failures` (the hand holds still).

Action (6): `[dx, dy, dz, rx, ry, rz]` in `[-1, 1]`, meaning `pos_upper * d` metres and
`rot_upper * r` radians per control step (0.05 s), in the base frame; the new target is
`p + dp`, `R_delta @ R` ("root_translation:root_aligned_body_rotation"). Near the joint limits the
effective semantics is "the closest feasible pose", not the delta — see `ik_short`.

Example:
    >>> env = gym.make("MikasaCabinetRetrieval-v0", control_mode="pd_ee_delta_pose")  # doctest: +SKIP
    >>> env.action_space.shape                                                        # doctest: +SKIP
    (12,)
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from transforms3d import quaternions as tq

from mani_skill.agents.controllers.pd_ee_pose import (
    PDEEPoseController,
    PDEEPoseControllerConfig,
)
from mani_skill.utils import common, gym_utils
from mani_skill.utils.geometry.rotation_conversions import (
    axis_angle_to_quaternion,
    matrix_to_axis_angle,
    quaternion_multiply,
    quaternion_to_matrix,
)
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array


#: IK residuals accepted as a solution; see `_solve_ik`.
IK_ACCEPT_POS = 0.003  # m
IK_ACCEPT_ROT = 0.01  # rad
#: The solver: weighted damped least squares, `IK_ITERS` Newton steps from the current
#: joints, damping `IK_DAMPING`; joints whose name contains "roll" cost `ROLL_WEIGHT`
#: times more to move. Why (journal 2026-09-08, W32): with the wrist straight the
#: forearm-roll and wrist-roll axes coincide, the end-effector pose does not tell them
#: apart, and an unweighted solver traded 4 rad of one for the other over a 500-step
#: approach while the hand tracked to a millimetre — until the rolled forearm met the
#: cabinet. The weighted minimum-norm step keeps the posture the demonstration had.
IK_ITERS = 20
IK_DAMPING = 1e-4
#: Null-space posture term: each iteration also moves the joints toward `posture_hint`
#: (or, without a hint, the arm's joints at the first reset) by this fraction of the gap, projected
#: into the null space of the task Jacobian, so the hand does not move. This is what
#: resolves a 7-DoF arm's redundancy deterministically (robosuite's OSC does the same
#: toward the initial configuration). A joint-space planner's demonstrations carry
#: null-space motion the end-effector pose cannot show — on Retrieval seed 0 the forearm
#: rolls 4 rad against the wrist with the hand still — and `ee_convert` hands the
#: source's joints in as the hint so a converted episode keeps the demonstrated posture.
#: The hint is NOT part of the action; a policy at deployment gets the rest posture.
NULLSPACE_GAIN = float(os.environ.get("MIKASA_EE_NULLSPACE_GAIN", "1.0"))  # toward a consistent hint (conversion)
NULLSPACE_GAIN_REST = float(os.environ.get("MIKASA_EE_NULLSPACE_GAIN_REST", "0.05"))  # toward the rest posture
# (deployment): gentle, or the pull toward the tucked start fights every reach — at 1.0 the
# open-loop replay of 20 demonstrations dropped from 19 to 14 successes (W32)
NULLSPACE_TOL = 1e-3  # rad: the posture gap under which the iteration may stop (a 0.04 rad
# residual left by an early exit put the two robots' drives on different transients and the
# hand 1.5 cm short in a 0.4 m/s approach, W32)
IK_MAX_STEP = 0.15  # rad per joint per iteration: a full Gauss-Newton step overshoots on a 0.1 m reach (W32)
PROJECTOR_DAMPING = 1e-9  # the null-space projector is (all but) undamped: with IK_DAMPING it leaked
# lambda*(J W^-1 J^T)^-1 J want into the task space — 0.2 % at the rest posture, ~100 % near a
# singularity (review, 2026-09-08)
ROLL_WEIGHT = float(os.environ.get("MIKASA_EE_ROLL_WEIGHT", "10.0"))


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)


def _rotvec(R: np.ndarray) -> np.ndarray:
    """Rotation vector of a rotation matrix (the short one)."""
    q = tq.mat2quat(R)
    if q[0] < 0:
        q = -q
    axis, angle = tq.quat2axangle(q)
    return np.asarray(axis, dtype=np.float64) * float(angle)


def _nullspace_step(J: np.ndarray, w_inv: np.ndarray, want: np.ndarray, damping: float) -> np.ndarray:
    """`want` projected into the null space of J (W-weighted): N want = want - W^-1 J^T (J W^-1 J^T)^-1 J want."""
    JW = J * w_inv[None, :]
    A = JW @ J.T + damping * np.eye(J.shape[0])
    return want - JW.T @ np.linalg.solve(A, J @ want)


def _cap(dq: np.ndarray, step: float) -> np.ndarray:
    big = float(np.abs(dq).max()) if dq.size else 0.0
    return dq * (step / big) if big > step else dq


def _cap_t(dq: torch.Tensor, step: float) -> torch.Tensor:
    big = dq.abs().amax(dim=1, keepdim=True).clamp(min=step)
    return dq * (step / big)


def _weighted_dls(J: np.ndarray, delta: np.ndarray, w_inv: np.ndarray, damping: float) -> np.ndarray:
    """Minimum-norm step in the metric W (w_inv = 1/diag(W)): dq = W^-1 J^T (J W^-1 J^T + damping I)^-1 delta."""
    JW = J * w_inv[None, :]
    A = JW @ J.T + damping * np.eye(J.shape[0])
    return JW.T @ np.linalg.solve(A, delta)


class PDEEPoseBaseController(PDEEPoseController):
    config: "PDEEPoseBaseControllerConfig"

    ik_failures: int = 0  # steps on which the solver returned no finite joint target (the hand holds still)
    ik_short: int = 0  # steps on which the target was out of reach and the closest feasible joints were used
    posture_hint = None  # (n,) joints the null space is pulled toward this step; None = the rest posture
    _rest_qpos = None  # the null-space anchor without a hint: the arm's joints at the FIRST reset (see reset)

    def reset(self):
        super().reset()
        # The rest posture is anchored ONCE, at the first reset, and never moved: under GPU
        # partial resets `controller.reset()` runs with the reset mask already all-ones
        # (sapien_env.py, b22), so re-anchoring here would give the envs still running
        # their mid-episode joints as the posture reference (measured: 0.37 rad shift on
        # three of four envs when one reset — review, 2026-09-08). The tasks start every
        # episode from the same arm posture, so the first reset is the reference.
        if self._rest_qpos is None:
            self._rest_qpos = self.qpos[:1].clone()
        self.posture_hint = None
        self.ik_failures = 0
        self.ik_short = 0

    def _check_gpu_sim_works(self):
        # The parent refuses every frame but root-translation on the GPU because it
        # never transforms; this class always solves in the articulation-root frame.
        pass

    @property
    def ee_pose_at_root(self) -> Pose:
        """The end-effector in the ARTICULATION-ROOT frame — the frame the IK solves in
        (pinocchio and the pytorch-kinematics chain both start at the URDF root)."""
        return self.articulation.pose.inv() * self.ee_pose

    def to_root_frame(self, pose_at_base: Pose) -> Pose:
        return self.articulation.pose.inv() * (self.root_link.pose * pose_at_base)

    def _joint_weights(self):
        if not hasattr(self, "_w_inv"):
            w = np.array([ROLL_WEIGHT if "roll" in j.name else 1.0 for j in self.joints], dtype=np.float64)
            names = [j.name for j in self.joints]
            for item in filter(None, (x.strip() for x in os.environ.get("MIKASA_EE_JOINT_WEIGHTS", "").split(","))):
                name, sep, val = item.partition(":")  # "wrist_roll_joint:10,..." — exact joint names
                assert sep and name.strip() in names, f"MIKASA_EE_JOINT_WEIGHTS: expected <joint name>:<weight> with a name in {names}, got {item!r}"
                w[names.index(name.strip())] = float(val)
            self._w_inv = 1.0 / w
            lim = np.array([[float(j.limits[0, 0]), float(j.limits[0, 1])] for j in self.joints], dtype=np.float64)
            self._qlim = lim
        return self._w_inv, self._qlim

    def _solve_ik(self, target_at_root: Pose):
        """Joint targets for `target_at_root`.

        CPU: `_solve_ik_pinocchio`; GPU: `_solve_ik_batched`. Both are the same weighted
        damped-least-squares scheme within the joint limits (see the module constants);
        an unreachable target yields the closest feasible joints and a count in
        `ik_short`. None only if the solver itself fails, which the caller turns into
        "hold still" and counts in `ik_failures`.
        """
        q0 = self.articulation.get_qpos()
        if self.kinematics.use_gpu_ik:
            q = self._solve_ik_batched(target_at_root, q0)
        else:
            q = self._solve_ik_pinocchio(target_at_root, q0)
        if q is None or not bool(torch.isfinite(q).all()):
            return None
        return q

    def _solve_ik_pinocchio(self, target_at_root: Pose, q0: torch.Tensor):
        k = self.kinematics
        pm = k.pmodel
        assert q0.shape[0] == 1, "the pinocchio path solves one env, like upstream's CPU IK; use the GPU backend for batches"
        ctrl = common.to_numpy(k.pmodel_controlled_joint_indices)
        q = common.to_numpy(q0)[0][common.to_numpy(k.pmodel_active_joint_indices)].astype(np.float64)
        w_inv, qlim = self._joint_weights()
        tp = np.asarray(target_at_root.sp.p, dtype=np.float64)
        Rt = quaternion_to_matrix(target_at_root.q)[0].cpu().numpy().astype(np.float64)
        hint = self.posture_hint if self.posture_hint is not None else self._rest_qpos
        gain = NULLSPACE_GAIN if self.posture_hint is not None else NULLSPACE_GAIN_REST
        hint = None if hint is None else common.to_numpy(hint).reshape(-1)[-len(ctrl):].astype(np.float64)

        def residual(qv):
            pm.compute_forward_kinematics(qv)
            pose = pm.get_link_pose(k.end_link_idx)
            p = np.asarray(pose.p, dtype=np.float64)
            R = tq.quat2mat(np.asarray(pose.q, dtype=np.float64))
            return p, tp - p, _rotvec(Rt @ R.T)

        best_q, best_err = q.copy(), np.inf
        for _ in range(IK_ITERS):
            p, dp, drot = residual(q)
            err = np.linalg.norm(dp) + 0.3 * np.linalg.norm(drot)  # 0.3 m per rad: the arm's reach
            if err < best_err:
                best_q, best_err = q.copy(), err
            task_done = np.linalg.norm(dp) <= IK_ACCEPT_POS * 0.3 and np.linalg.norm(drot) <= IK_ACCEPT_ROT * 0.3
            posture_done = self.posture_hint is None or np.abs(hint - q[ctrl]).max() <= NULLSPACE_TOL
            if task_done and posture_done:
                break
            pm.compute_full_jacobian(q)
            J = np.asarray(pm.get_link_jacobian(k.end_link_idx, local=False), dtype=np.float64)
            # pinocchio's world Jacobian is the spatial twist at the origin; the point's
            # linear velocity is v - p x w
            Jp = J[:3] - _skew(p) @ J[3:]
            Jc = np.vstack([Jp, J[3:]])[:, ctrl]  # (6, n)
            delta = np.r_[dp, drot]
            qc = q[ctrl]
            want = None if hint is None else gain * (hint - qc)
            # Active set over the TOTAL step: a joint on a stop that the step (task OR posture
            # part) would push outward is frozen — its column dropped from both solves and its
            # posture wish zeroed — and the step re-solved, up to a few rounds. Clipping the
            # posture part alone would take it out of the null space and inject a task error
            # every iteration (the review's stall, 2026-09-08); clipping the task part alone
            # wasted the whole step on a stop (W32).
            blocked = np.zeros(len(ctrl), dtype=bool)
            for _round in range(4):
                wi = np.where(blocked, 0.0, w_inv)
                dq_task = _weighted_dls(Jc, delta, wi, IK_DAMPING)
                dq_null = np.zeros_like(dq_task) if want is None else _nullspace_step(Jc, wi, np.where(blocked, 0.0, want), PROJECTOR_DAMPING)
                dq = _cap(dq_task, IK_MAX_STEP) + _cap(dq_null, IK_MAX_STEP)
                push = ((qc >= qlim[:, 1] - 1e-9) & (dq > 0)) | ((qc <= qlim[:, 0] + 1e-9) & (dq < 0))
                if not (push & ~blocked).any():
                    break
                blocked |= push
            q[ctrl] = np.clip(qc + dq, qlim[:, 0], qlim[:, 1])
        p, dp, drot = residual(q)
        acceptable = np.linalg.norm(dp) <= IK_ACCEPT_POS and np.linalg.norm(drot) <= IK_ACCEPT_ROT
        err = np.linalg.norm(dp) + 0.3 * np.linalg.norm(drot)
        if not acceptable and err > best_err:
            # Only when the final iterate is NOT within tolerance: never return something
            # worse than the best seen (or the start). An acceptable final iterate is kept
            # even if an earlier one had a smaller task residual — the later one has the
            # converged posture, and with a hint that is what the drive transients follow
            # (falling back unconditionally cost two of nineteen conversions, 2026-09-08).
            q = best_q
            p, dp, drot = residual(q)
            acceptable = np.linalg.norm(dp) <= IK_ACCEPT_POS and np.linalg.norm(drot) <= IK_ACCEPT_ROT
        if not acceptable:
            # Unreachable within the joint limits (pinocchio's own IK would return a
            # target past a limit, which the drive then cannot follow). Best effort: the
            # closest feasible joints, and the step is counted in `ik_short`.
            self.ik_short += 1
        return common.to_tensor([q[ctrl]], device=self.device)

    def _solve_ik_batched(self, target_at_root: Pose, q0: torch.Tensor) -> torch.Tensor:
        """The same scheme on the pytorch-kinematics chain, batched over envs.

        Not `Kinematics.compute_ik`: in ManiSkill 3.0.0b22 its Levenberg-Marquardt step
        regularises with an identity the size of the WHOLE chain (11 on Fetch) against a
        Jacobian masked to the 7 controlled joints and fails with a shape mismatch.
        Returns the controlled joints in chain order, which is `active_joint_indices` order.
        Stops early once every env's task residual is under tolerance (and, with a hint,
        its posture gap): 20 full iterations cost 430 ms per step at any batch size
        (kernel-launch bound; review, 2026-09-08). No best-iterate bookkeeping here: the
        batched loop is what the GPU evaluation runs, and a per-env argmin would double
        the FK calls; the CPU path (the converter) keeps the best iterate.
        """
        k = self.kinematics
        qa = q0[:, k.active_ancestor_joint_idxs].clone()
        mask = k.qmask
        n = int(mask.sum())
        w_inv, qlim = self._joint_weights()
        w_inv_t = torch.tensor(w_inv, dtype=qa.dtype, device=self.device)
        lo = torch.tensor(qlim[:, 0], dtype=qa.dtype, device=self.device)
        hi = torch.tensor(qlim[:, 1], dtype=qa.dtype, device=self.device)
        eye6 = torch.eye(6, device=self.device, dtype=qa.dtype)
        R_t = quaternion_to_matrix(target_at_root.q)
        hint_t = self.posture_hint if self.posture_hint is not None else self._rest_qpos
        gain = NULLSPACE_GAIN if self.posture_hint is not None else NULLSPACE_GAIN_REST
        hint_t = None if hint_t is None else common.to_tensor(hint_t, device=self.device).to(qa.dtype).reshape(-1, n)
        B = qa.shape[0]

        def solve(J, W, rhs):  # W^-1 J^T (J W^-1 J^T + damping I)^-1 rhs; W: (B, n) = diag of W^-1
            JW = J * W.unsqueeze(1)  # (B, 6, n)
            A = JW @ J.transpose(1, 2)
            return (JW.transpose(1, 2) @ torch.linalg.solve(A + IK_DAMPING * eye6, rhs)).squeeze(-1)

        def project(J, W, want):  # (I - W^-1 J^T (J W^-1 J^T)^-1 J) want
            JW = J * W.unsqueeze(1)
            A = JW @ J.transpose(1, 2)
            return want - (JW.transpose(1, 2) @ torch.linalg.solve(A + PROJECTOR_DAMPING * eye6, J @ want.unsqueeze(-1))).squeeze(-1)

        for _ in range(IK_ITERS):
            T = k.pk_chain.forward_kinematics(qa).get_matrix()
            dp = target_at_root.p - T[:, :3, 3]
            drot = matrix_to_axis_angle(R_t @ T[:, :3, :3].transpose(1, 2))
            qc = qa[:, mask]
            task_done = (dp.norm(dim=1) <= IK_ACCEPT_POS * 0.3) & (drot.norm(dim=1) <= IK_ACCEPT_ROT * 0.3)
            posture_done = torch.ones(B, dtype=torch.bool, device=self.device) if self.posture_hint is None else (hint_t - qc).abs().amax(dim=1) <= NULLSPACE_TOL
            if bool((task_done & posture_done).all()):
                break
            delta = torch.cat([dp, drot], dim=1).unsqueeze(-1)  # (B, 6, 1)
            J = k.pk_chain.jacobian(qa)[:, :, mask]  # (B, 6, n)
            want = None if hint_t is None else gain * (hint_t - qc)
            blocked = torch.zeros((B, n), dtype=torch.bool, device=self.device)
            for _round in range(4):
                W = torch.where(blocked, torch.zeros((B, n), dtype=qa.dtype, device=self.device), w_inv_t.expand(B, n))
                dq_task = _cap_t(solve(J, W, delta), IK_MAX_STEP)
                dq_null = torch.zeros_like(dq_task) if want is None else _cap_t(project(J, W, torch.where(blocked, torch.zeros_like(want), want)), IK_MAX_STEP)
                dq = dq_task + dq_null
                push = ((qc >= hi - 1e-6) & (dq > 0)) | ((qc <= lo + 1e-6) & (dq < 0))
                if not bool((push & ~blocked).any()):
                    break
                blocked |= push
            qa[:, mask] = torch.clamp(qc + dq, lo, hi)
        return qa[:, mask]

    def _clip_and_scale_action(self, action):
        pos = gym_utils.clip_and_scale_action(
            action[:, :3], self.action_space_low[:3], self.action_space_high[:3]
        )
        rot = action[:, 3:].clone()
        norm = torch.linalg.norm(rot, dim=1)
        over = norm > 1
        rot[over] = rot[over] / norm[over, None]
        rot = rot * self._rot_step()
        return torch.hstack([pos, rot])

    def _rot_step(self) -> float:
        r = np.broadcast_to(np.asarray(self.config.rot_upper, dtype=np.float64), 3)
        assert np.allclose(r, r[0]), "rot_upper must be one number: the rotation delta is a vector clipped by norm"
        return float(r[0])

    def compute_target_pose(self, prev_ee_pose_at_base: Pose, action):
        assert self.config.use_delta, "this controller is a delta controller"
        delta_pos, delta_rotvec = action[:, 0:3], action[:, 3:6]
        delta_q = axis_angle_to_quaternion(delta_rotvec)
        return Pose.create_from_pq(
            prev_ee_pose_at_base.p + delta_pos,
            quaternion_multiply(delta_q, prev_ee_pose_at_base.q),
        )

    def set_action(self, action: Array):
        action = self._preprocess_action(action)
        self._step = 0
        self._start_qpos = self.qpos
        prev = self._target_pose if self.config.use_target else self.ee_pose_at_base
        self._target_pose = self.compute_target_pose(prev, action)  # base frame
        self._target_qpos = self._solve_ik(self.to_root_frame(self._target_pose))
        if self._target_qpos is None:
            self.ik_failures += 1  # counted, not silent: the hand holds still this step
            self._target_qpos = self._start_qpos
        if self.config.interpolate:
            self._step_size = (self._target_qpos - self._start_qpos) / self._sim_steps
        else:
            self.set_drive_targets(self._target_qpos)


@dataclass
class PDEEPoseBaseControllerConfig(PDEEPoseControllerConfig):
    controller_cls = PDEEPoseBaseController
