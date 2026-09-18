"""Convert a recorded `pd_joint_pos` trajectory to `pd_ee_delta_pose` — the BASE-FRAME mode.

Why not ManiSkill's `from_pd_joint_pos_to_ee` (journal 2026-09-07, five findings on the
Retrieval demonstration seed 0): it takes the hand's delta in the WORLD frame and feeds it to a
controller that adds it in its root frame — right for a fixed arm at the origin, wrong for a
base that has driven and turned; when a step exceeds the 0.1 m bound it clips and re-issues the
step up to four times, and each re-issue repeats the base velocity and the body targets, so the
base runs 0.67 m past where it should be; and the extra sub-steps ate the horizon (616 source
steps became 900). This converter:

* expresses the target in the frame of the SOURCE robot's base (`base_link`) and applies it in
  the frame of the target robot's base — what `PDEEPoseBaseController` expects;
* uses the source controller's TARGET pose (forward kinematics of its `_target_qpos`), not the
  achieved one, so both arms are driven toward the same pose each step and track alike;
* never sub-steps: one source step is one target step, base/body/gripper pass through
  unchanged; a delta beyond the bound is clipped and COUNTED (`stats`), so a lossy conversion is
  visible rather than silent. Size the bounds (`ds_fetch.EE_POS_STEP`/`EE_ROT_STEP`) so the
  count is zero.

Installed by `utils.mikasa.replay` when `--target-control-mode pd_ee_delta_pose` is asked for:

    python -m utils.mikasa.replay --traj-path X.h5 --target-control-mode pd_ee_delta_pose \\
        --obs-mode rgb --save-traj --allow-failure --use-first-env-state

Example:
    >>> info = from_pd_joint_pos_to_ee_base("pd_ee_delta_pose", actions, ori_env, env)  # doctest: +SKIP
    >>> stats["clipped_pos"], stats["clipped_rot"]                                     # doctest: +SKIP
    (0, 0)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import sapien
from transforms3d import quaternions as tq

from mani_skill.utils import common
from utils.mikasa.agents.ds_fetch.ee_base import PDEEPoseBaseController

#: Filled by the last conversion (one episode per process in the recording pipeline).
stats: dict = {}


def rotvec_of(dq: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle, radians) of a wxyz quaternion — the SHORT one.

    q and -q are the same rotation; `quat2axangle` of the one with w < 0 returns the long
    way round (up to 2*pi). The first conversion run wrote 163 such steps (2026-09-08).
    """
    dq = np.asarray(dq, dtype=np.float64)
    dq = dq / np.linalg.norm(dq)
    if dq[0] < 0:
        dq = -dq
    axis, angle = tq.quat2axangle(dq)
    return np.asarray(axis, dtype=np.float64) * float(angle)


def _robot_contacts(env) -> list:
    """Robot links in contact with anything but the robot, with the impulse norm (CPU sim)."""
    try:
        robot = env.unwrapped.agent.robot
        links = {l._objs[0].entity.name if hasattr(l._objs[0], "entity") else l.name: l for l in robot.links}
        out = {}
        for contact in env.unwrapped.scene.px.get_contacts():
            names = [b.entity.name for b in contact.bodies]
            robot_side = [n for n in names if n in links]
            other = [n for n in names if n not in links]
            if not robot_side or not other:
                continue
            imp = float(np.linalg.norm(sum(np.asarray(pt.impulse) for pt in contact.points))) if contact.points else 0.0
            if imp > 1e-6:
                key = f"{robot_side[0]}~{other[0]}"
                out[key] = round(out.get(key, 0.0) + imp, 5)
        return sorted(out.items(), key=lambda kv: -kv[1])[:4]
    except Exception as e:  # a diagnostic must not kill a conversion
        return [("ERR", type(e).__name__)]


def from_pd_joint_pos_to_ee_base(output_mode, ori_actions, ori_env, env, render=False, pbar=None, verbose=False):
    """ManiSkill's conversion signature (replay_trajectory calls it through `from_pd_joint_pos`).

    Returns the last step's `info`, like upstream, so `--allow-failure` bookkeeping works.
    """
    ori_c = ori_env.agent.controller
    c = env.agent.controller
    ori_arm, arm = ori_c.controllers["arm"], c.controllers["arm"]
    assert isinstance(arm, PDEEPoseBaseController), type(arm).__name__
    pin = ori_c.articulation.create_pinocchio_model()
    ee_idx = arm.ee_link.index
    pos_step, rot_step = float(arm.config.pos_upper), float(arm.config.rot_upper)

    req_pos, req_rot, tcp_err, base_err = [], [], [], []
    info = {}
    rows = []  # per-step trace, written when MIKASA_EE_TRACE names a file
    trace = os.environ.get("MIKASA_EE_TRACE")
    trace_contacts = bool(trace)
    use_hint = os.environ.get("MIKASA_EE_HINT", "1") != "0"  # MIKASA_EE_HINT=0: the deployment-time null space (rest posture)
    # Everything that changes the meaning of the written actions travels with them, or a
    # dataset converted under an override replays as clipped elsewhere (review, 2026-09-08).
    settings = dict(pos_step=pos_step, rot_step=rot_step, root_link=arm.root_link.name, hint=use_hint,
                    **{k: v for k, v in os.environ.items() if k.startswith("MIKASA_EE_")})
    arm.ik_failures = 0
    arm.ik_short = 0
    if pbar is not None:
        pbar.reset(total=len(ori_actions))
    try:
        info = _convert_loop(ori_actions, ori_env, env, ori_c, c, ori_arm, arm, pin, ee_idx, pos_step, rot_step,
                             use_hint, trace_contacts, render, pbar, verbose, req_pos, req_rot, tcp_err, base_err, rows, info)
    finally:
        if trace:
            Path(trace).write_text(json.dumps(dict(settings=settings, rows=rows)))
    clipped_pos = sum(1 for r in rows if r["clip_pos"])
    clipped_rot = sum(1 for r in rows if r["clip_rot"])
    stats.clear()
    stats.update(
        steps=len(ori_actions), clipped_pos=clipped_pos, clipped_rot=clipped_rot, ik_failures=arm.ik_failures, ik_short=arm.ik_short,
        max_req_pos_m=round(max(req_pos), 4), max_req_rot_rad=round(max(req_rot), 4),
        p99_req_pos_m=round(float(np.percentile(req_pos, 99)), 4), p99_req_rot_rad=round(float(np.percentile(req_rot, 99)), 4),
        tcp_err_final_m=round(tcp_err[-1], 4), tcp_err_max_m=round(max(tcp_err), 4),
        base_err_final_m=round(base_err[-1], 4), base_err_max_m=round(max(base_err), 4),
        success=bool(common.to_numpy(info["success"]).reshape(-1)[0]) if "success" in info else None,
        settings=json.dumps(settings, separators=(",", ":")),
    )
    print("EE-CONVERT " + " ".join(f"{k}={v}" for k, v in stats.items()), flush=True)
    return info


def _convert_loop(ori_actions, ori_env, env, ori_c, c, ori_arm, arm, pin, ee_idx, pos_step, rot_step,
                  use_hint, trace_contacts, render, pbar, verbose, req_pos, req_rot, tcp_err, base_err, rows, info):
    for t in range(len(ori_actions)):
        a = common.to_tensor(ori_actions[t], device=ori_env.unwrapped.device)
        out = dict(common.to_tensor(ori_c.to_action_dict(a), device=env.unwrapped.device))
        # The source's body slot is an absolute target (pd_joint_pos); the EE modes' body is
        # a delta controller since 2026-09-09 (head, torso: ±0.1 per step from the measured
        # pose, like the arm's joints in pd_joint_delta_pos). Same rule as the solver's
        # `_compose`: the increment toward the absolute target, clipped to one step.
        body = c.controllers.get("body")
        if body is not None and getattr(body.config, "use_delta", False):
            step = float(np.asarray(body.config.upper, dtype=np.float64).reshape(-1)[0])
            measured = body.qpos.to(out["body"].device).reshape(-1)
            out["body"] = ((out["body"].reshape(-1) - measured) / step).clamp(-1.0, 1.0)
        ori_env.step(a)
        if pbar is not None:
            pbar.update()

        # the SOURCE's commanded hand pose, in the SOURCE base frame
        full = ori_c.articulation.get_qpos().clone()
        full[:, ori_arm.active_joint_indices] = ori_arm._target_qpos
        pin.compute_forward_kinematics(full.cpu().numpy()[0])
        target_world = ori_c.articulation.pose.sp * pin.get_link_pose(ee_idx)
        target_at_base = ori_env.agent.base_link.pose.sp.inv() * target_world

        # the source's joints as the null-space hint (ee_base.py: NULLSPACE_GAIN) — not an
        # action, only what resolves the redundancy the way the demonstration did
        arm.posture_hint = ori_arm._target_qpos.clone() if use_hint else None

        # the delta from the TARGET robot's current hand, in its own base frame
        cur = arm.ee_pose_at_base.sp
        dp = target_at_base.p - cur.p
        dq = tq.qmult(target_at_base.q, tq.qinverse(cur.q))
        rv = rotvec_of(dq)
        req_pos.append(float(np.linalg.norm(dp)))
        req_rot.append(float(np.linalg.norm(rv)))

        a_pos = dp / pos_step
        clip_pos = bool(np.abs(a_pos).max() > 1)
        if clip_pos:
            if verbose:
                print(f"[ee_convert] step {t}: position delta clipped {a_pos.round(2)}")
            a_pos = np.clip(a_pos, -1, 1)
        a_rot = rv / rot_step
        n = np.linalg.norm(a_rot)
        clip_rot = bool(n > 1)
        if clip_rot:
            if verbose:
                print(f"[ee_convert] step {t}: rotation delta clipped |{n:.2f}|")
            a_rot = a_rot / n
        out["arm"] = common.to_tensor(np.r_[a_pos, a_rot].astype(np.float32), device=env.unwrapped.device)

        _, _, _, _, info = env.step(c.from_action_dict(out))
        if render:
            env.render_human()

        tcp_err.append(float(np.linalg.norm(env.agent.tcp_pose.sp.p - ori_env.agent.tcp_pose.sp.p)))
        base_err.append(float(np.linalg.norm(env.agent.base_link.pose.sp.p[:2] - ori_env.agent.base_link.pose.sp.p[:2])))
        q_src = common.to_numpy(ori_c.articulation.get_qpos())[0][common.to_numpy(ori_arm.active_joint_indices)]
        q_tgt = common.to_numpy(c.articulation.get_qpos())[0][common.to_numpy(arm.active_joint_indices)]
        track_src = float(np.abs(common.to_numpy(ori_arm._target_qpos)[0] - q_src).max())
        track_tgt = float(np.abs(common.to_numpy(arm._target_qpos)[0] - q_tgt).max())
        tgt_diff = np.abs(common.to_numpy(arm._target_qpos)[0] - common.to_numpy(ori_arm._target_qpos)[0])
        contacts = _robot_contacts(env) if trace_contacts else None
        rows.append(dict(t=t, req_pos=round(req_pos[-1], 4), req_rot=round(req_rot[-1], 4),
                         track_src=round(track_src, 4), track_tgt=round(track_tgt, 4), contacts=contacts,
                         tgt_diff_max=round(float(tgt_diff.max()), 4), tgt_diff_argmax=int(tgt_diff.argmax()),
                         q_diff_max=round(float(np.abs(q_src - q_tgt).max()), 4), q_diff_argmax=int(np.abs(q_src - q_tgt).argmax()),
                         clip_pos=clip_pos, clip_rot=clip_rot,
                         ik_fail=arm.ik_failures, ik_short=arm.ik_short, tcp_err=round(tcp_err[-1], 4), base_err=round(base_err[-1], 4),
                         gripper=float(common.to_numpy(out["gripper"]).reshape(-1)[0]),
                         base_cmd=[round(float(v), 3) for v in common.to_numpy(out["base"]).reshape(-1).tolist()]))

    return info
