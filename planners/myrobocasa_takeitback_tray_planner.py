import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import cast

import gymnasium as gym
import numpy as np
import sapien
import torch
from mani_skill.agents.robots import Fetch
from my_scenes.my_robocasa_takeitback_tray import MyRoboCasaSceneTakeItBackTray
from mani_skill.utils.wrappers import RecordEpisode
from robots.fetch.extand import FetchMotionPlanningSapienSolver
from planners.takeitback_common import _grasp_pose, _tcp_at, _tcp_to
from utils.logging_utils import PlannerLogger, capture_stdout
from utils.planners_utils import (
    _rotate_base_to,
    _screw_base_translate,
    _velocity_segment,
    _base_cmd,
)

# max horizontal base->cup distance at which the fallback arm re-grasp drives
# the base so the cup is inside the arm's reachable workspace. The straight-arms
# TCP sits ARM_OFFSET (1.128 m) north of the base but mplib's real reachable
# horizontal is ~1.086 m, so the gripper-at-cup park (base->cup = ARM_OFFSET)
# leaves the cup ~2-5 cm past reach on some seeds (11/32/41/47 -> "IK Failed").
GRASP_STANDOFF = 1.00


def _repair_trajectory_metadata(run_dir):
    """Make RoboCasa reconfigure seeds replayable by ManiSkill's checker."""
    path = Path(run_dir) / "trajectory.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for episode in data.get("episodes", []):
        seed = episode.get("reset_kwargs", {}).get("seed")
        if isinstance(seed, list) and len(seed) == 1:
            seed = seed[0]
        if seed is not None and episode.get("episode_seed") != seed:
            episode["episode_seed"] = int(seed)
            changed = True
    if changed:
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Motion planner for MyRoboCasa_TakeItBackTray-v1 scene")
    parser.add_argument("--seed", type=int, default=3, help="Random seed (default: 3)")
    parser.add_argument("--render-mode", type=str, default="rgb_array",
                        choices=["rgb_array", "human", "sensors"],
                        help="Render mode (default: rgb_array)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode in planner")
    parser.add_argument("--info", action="store_true", help="Print environment info in planner")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for log output (default: logs)")
    parser.add_argument("--log-freq", type=int, default=10, help="Write trajectory rows every N steps (default: 10)")
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable mp4 video recording (skip StreamingVideoRecorder: no render, faster)",
    )
    return parser.parse_args()

def planning(env, seed, debug=False, vis=None, info=False):
    vis = vis or env.unwrapped.render_mode == "human"

    unwenv: MyRoboCasaSceneTakeItBackTray = env.unwrapped
    obs, _ = env.reset(seed=seed, options={"reconfigure": True})
    agent: Fetch = cast(Fetch, unwenv.agent)  # captured after reconfigure reset

    tray_center = unwenv.tray.pose.p[0].cpu().numpy()

    planner = FetchMotionPlanningSapienSolver(
        env,
        base_pose=agent.robot.pose.sp,
        vis=vis,
        print_env_info=info,
        debug=debug,
    )

    # mplib emits absolute arm targets. Convert only those seven slots to raw
    # joint deltas; remaining canonical action layout stays unchanged.
    planner.control_mode = "pd_joint_pos"
    _step_absolute = env.step

    def _step_delta(action):
        action = np.asarray(action)
        if action.shape == (13,):
            action = action.copy()
            arm_qpos = agent.controller.controllers["arm"].qpos[0].cpu().numpy()
            action[:7] -= arm_qpos
        return _step_absolute(action)

    env.step = _step_delta

    def _sync():
        getattr(planner.planner, "update_from_simulation")()
    env.track_object(unwenv.cup, "cup")
    env.track_object(unwenv.tray, "tray")
    env.track_object(agent.tcp, "robot_tcp")
    env.track_object(agent.base_link, "robot_base")
    env.log_event("start", "Planning started")

    # ==================================================================== #
    # STRAIGHT-ARM GEOMETRY: the arm stays in the home (straight) config for
    # the WHOLE task. The TCP rides at (1.128, 0, 0.786 + torso) in the base
    # frame (measured), so the cup offset from the base is constant and all
    # horizontal positioning uses turn-then-forward base motion; vertical
    # positioning uses the torso only.
    # The robot rotates ONCE to face north (the arm into the counter) with
    # the empty gripper, then never rotates again. No arm reconfiguration,
    # no mplib arm motions, no sharp moves: every stage is a smooth scripted
    # drive / torso ramp, verified (cup z, TCP-cup gap) before the next one.
    # ==================================================================== #
    ARM_OFFSET = 1.128    # straight-arm TCP offset from the base center (m);
    # the arm points NORTH (into the counter) via the shoulder pan, while
    # the robot itself faces EAST (parallel to the counter)
    TORSO_TRANSPORT = 0.386  # the torso MAX: the arm ~1.19 m high - well above
    # the stove top (1.08) and the stack cabinets (0.89) that the straight
    # arm sweeps past during the base drives (the mplib collision models are
    # conservative, so the extra height reduces the false positives)
    TORSO_GRASP = 0.21      # the jaws' mid at the cup center (~1.0 m)
    TORSO_LOW = 0.02        # the torso floor for the placement ramps

    def hold_a():
        return getattr(agent.controller, "controllers")["arm"].qpos[0].cpu().numpy()

    def hold_b():
        return getattr(agent.controller, "controllers")["body"].qpos[0].cpu().numpy().copy()

    def step_hold(torso_target=None):
        a = np.zeros(13)
        a[:7] = hold_a()
        a[7] = planner.gripper_state
        b = hold_b()
        if torso_target is not None:
            b[2] = torso_target
        a[8:11] = b
        env.step(a)

    def ramp_torso(target, steps=150):
        # ramp the torso TARGET gradually: a fixed target makes the PD snap
        # the torso (and the arm with it) fast, which shoves the base around;
        # stepping the target by small increments keeps the motion slow
        start = hold_b()[2]
        for i in range(steps):
            b = hold_b()
            b[2] = start + (target - start) * ((i + 1) / steps)
            env.step(np.hstack([hold_a(), planner.gripper_state, b, _base_cmd()]))
        _sync()
        # pi-lens-ignore: unchecked-throwing-call-python
        return float(agent.tcp.pose.p[0][2])

    def ramp_arm(target, steps=120):
        """Move arm to a joint waypoint without teleporting its target."""
        target = np.asarray(target, dtype=float)
        start = hold_a()
        for i in range(steps):
            arm = start + (target - start) * ((i + 1) / steps)
            env.step(np.hstack([arm, planner.gripper_state, hold_b(), _base_cmd()]))
        _sync()

    def settle_gripper(steps=12):
        """Let a closed contact settle before testing or lifting it."""
        for _ in range(steps):
            step_hold()
        _sync()

    def descend_to_grasp(target_pose, xy_tol=0.05):
        """Use torso for final vertical closure when arm IK stalls."""
        target = np.asarray(target_pose.p, dtype=float)
        tcp = agent.tcp.pose.p[0].cpu().numpy()
        if np.linalg.norm(tcp[:2] - target[:2]) > xy_tol:
            return False
        # pi-lens-ignore: unchecked-throwing-call-python
        drop = float(tcp[2] - target[2])
        if drop > 0.015:
            body_z = max(0.0, hold_b()[2] - drop)
            ramp_torso(body_z, steps=30)
        return _tcp_to(agent, target) <= 0.05

    def drive_to(tgt, tol=0.04, min_improve=0.02):
        """Axis-aligned base drive, holding the arm + current torso,
        heading fixed north. Returns 0 on convergence (final dist < tol)."""
        return _velocity_segment(
            env, planner, np.asarray(tgt, dtype=float), hold_a(), hold_b(),
            planner.gripper_state, target_yaw=0.0, tol=tol,
            min_improve=min_improve,
        )

    def l_drive(aim_xy, tol=0.04):
        """Use one straight payload leg; keep L corridor for empty approach."""
        aim = np.asarray(aim_xy, dtype=float)
        base = agent.base_link.pose.p[0].cpu().numpy()
        if cup_held():
            # A held cup should cross the transfer in one orbit rather than
            # sweeping around the base at every L-corner.
            targets = (np.array([aim[0], aim[1], 0.0]),)
        else:
            targets = (
                np.array([base[0], aim[1] + 0.05, 0.0]),
                np.array([aim[0], aim[1] + 0.05, 0.0]),
                np.array([aim[0], aim[1], 0.0]),
            )
        for target in targets:
            current = agent.base_link.pose.p[0].cpu().numpy()
            if np.linalg.norm(target[:2] - current[:2]) <= tol:
                continue
            if _screw_base_translate(planner, target) != 0:
                return -1
            _sync()
        final = agent.base_link.pose.p[0].cpu().numpy()[:2]
        return 0 if np.linalg.norm(final - aim[:2]) <= max(tol, 0.06) else -1

    def lower_torso_until_cup_rests(surface_top_z):
        """Slowly lower the torso (grasped cup descends with the jaws) until
        the cup rests on the surface below (cup z stops decreasing)."""
        # pi-lens-ignore: unchecked-throwing-call-python
        rest_z = float(surface_top_z + unwenv.cup_half[2])
        for _ in range(300):
            # pi-lens-ignore: unchecked-throwing-call-python
            cz = float(unwenv.cup.pose.p[0][2])
            if cz <= rest_z + 0.01:
                break
            b = hold_b()
            if b[2] <= TORSO_LOW + 0.001:
                # No further torso motion is possible; avoid repeating no-op
                # steps when a held cup cannot reach the surface.
                break
            b[2] -= 0.005
            a = np.zeros(13)
            a[:7] = hold_a()
            a[7] = planner.gripper_state
            a[8:11] = b
            env.step(a)
        _sync()
        # pi-lens-ignore: unchecked-throwing-call-python
        return float(unwenv.cup.pose.p[0][2])

    def cup_held():
        return bool(unwenv.agent.is_grasping(unwenv.cup).item())

    def tcp_cup_gap():
        t = agent.tcp.pose.p[0].cpu().numpy()
        c = unwenv.cup.pose.p[0].cpu().numpy()
        # pi-lens-ignore: unchecked-throwing-call-python
        return float(np.linalg.norm(t - c))

    def report_stage(name):
        b = agent.base_link.pose.p[0].cpu().numpy()
        c = unwenv.cup.pose.p[0].cpu().numpy()
        print(f"[STAGE] {name}: base ({b[0]:.3f},{b[1]:.3f}) "
              f"cup ({c[0]:.3f},{c[1]:.3f},{c[2]:.3f}) gap {tcp_cup_gap():.3f} "
              f"grasped={cup_held()}")

    # ------------------------------------------------------------------ #
    # STAGE 0: raise torso above counter, rotate ONCE to face north, then
    # fold elbow slightly. The bent carry pose shortens the arm offset while
    # keeping elbow and gripper above fixtures.
    # ------------------------------------------------------------------ #
    env.log_event("phase", "Stage 0: raise torso, align along the counter")
    ramp_torso(TORSO_TRANSPORT, steps=150)
    # the robot faces EAST (parallel to the counter front edge): its
    # forward/backward drives then run along the counter (left/right of the
    # countertop) and its side faces the counter. The straight arm is then
    # swung 90 deg at the shoulder (pan) so it points INTO the counter
    # (north), then bend the elbow in a high, compact carry pose.
    _rotate_base_to(env, planner, np.array([1.0, 0.0, 0.0]))
    _sync()
    # 85 deg, NOT 90: the shoulder pan limit is +-1.6056 rad (+-92 deg), and
    # a pan pinned at exactly 90 deg leaves the IK no room to swing the arm
    # to the cup (the align failed with "IK Failed" - the pan was at the
    # joint limit). 85 deg keeps the arm pointing into the counter while
    # giving the IK ~7 deg of swing room. The pan target is ramped gradually
    # so the arm swing is slow and does not shove the base.
    _pan1 = np.deg2rad(85)
    _bent_arm = hold_a()
    _bent_arm[0] = _pan1
    _bent_arm[1] = -0.40
    _bent_arm[3] = 0.80
    _bent_arm[5] = -0.40
    # Pan and bend together: same safe high-torso corridor, one arm path
    # instead of two sequential ramps.
    ramp_arm(_bent_arm, steps=180)
    report_stage("0 raise+align+bend")

    # ------------------------------------------------------------------ #
    # STAGE 1: turn and drive along the counter to the cup's x
    # at the safe south line (y = cup_y - ARM_OFFSET - 0.4), then STAGE 2:
    # lower the torso to the grasp height and drive to the PRE-GRASP along a
    # SAFE corridor. The jaws' plates span +-6.45 cm from the gripper, so any
    # drive that passes closer than ~10 cm to the cup pushes it with a plate
    # edge (verified: the cup slid 0.2 m). The offset corridor (cup_x + 0.15)
    # keeps the plates clear during the y-leg and the x-leg.
    # The screw drive (drive_base_to_position) is used instead of the closed-
    # loop velocity segment: it converges reliably (~0.15 m, the arm's fine
    # alignment covers the rest) and keeps the heading fixed.
    # ------------------------------------------------------------------ #
    env.log_event("phase", "Stage 1: drive to the cup x")
    cup_xy = unwenv.cup.pose.p[0].cpu().numpy()[:2]
    south_line = cup_xy[1] - ARM_OFFSET - 0.40
    # the base parks WEST of the cup: the pan-85 arm puts the gripper 10 cm
    # west of the base, so the y-leg must pass WEST of the cup (the plates
    # bracket it with 2.6-3.1 cm clearance); an east corridor would drive
    # the plates through the cup (verified: the cup was knocked over).
    # L-path: drive SOUTH first (away from the counter) to the line, then
    # EAST/WEST to the cup's x - a direct diagonal let the base cross the
    # y=-0.95 counter guard on seed 9 (the screw's rotate-retry slides)
    _b1 = agent.base_link.pose.p[0].cpu().numpy()
    # y_guard=False for the south leg: the robot can SPAWN north of the
    # counter line (y > -0.95) and must be allowed to drive away from the
    # counter first (verified: seed 9 aborted before moving - the guard fired
    # on the starting position)
    res = env.log_motion(
        "Stage 1 drive", l_drive, np.array([_b1[0], south_line])
    )
    if res == 0:
        res = env.log_motion(
            "Stage 1 drive", l_drive,
            np.array([cup_xy[0] - 0.15, south_line])
        )
    if res != 0:
        print("Stage 1 drive failed; aborting")
        env.log_event("error", "Stage 1 drive failed")
        success = bool(unwenv.evaluate()["success"].item())
        env.log_event("result", "Task aborted", success=success)
        env.reset()
        return success
    # Turn-drive-turn preserves base heading, but the absolute arm controller
    # can retain yaw-induced tracking error; restore existing bent carry pose
    # before computing the next live TCP offset.
    ramp_arm(_bent_arm, steps=60)
    report_stage("1 at cup x")

    env.log_event("phase", "Stage 2: pre-grasp position")
    # allow the cup in the planning world from here on: the screw drives near
    # the cup fail their start-state collision check (gripper <-> cup) and
    # fling the base around (verified: the correction drive ended 0.5 m away)
    from mplib.sapien_utils.conversion import convert_object_name
    _acm = planner.planner.planning_world.get_allowed_collision_matrix()
    _acm.set_default_entry(convert_object_name(unwenv.cup._objs[0]), True)
    # allow the straight arm vs the counter fixtures (the stack doors, the
    # dishwasher, the stove...): the arm rides ~1.19 m high - physically
    # above them - but the mplib collision models are conservative and flag
    # false positives that fail every screw plan near the counter (verified:
    # seeds 9/17 - the gripper vs the stack hingedoor / the dishwasher).
    # NOT the walls/floor - those collisions are real.
    for _nm, _act in unwenv.scene.actors.items():
        if any(_k in _nm for _k in ("counter", "stack", "stove", "dishwasher",
                                    "sink", "cab", "fridge", "paper_towel",
                                    "tray")):
            try:
                _acm.set_default_entry(convert_object_name(_act._objs[0]), True)
            except Exception:
                pass
    # drive with arm HIGH (the plates cannot touch cup), lower torso to grasp
    # height only AFTER base is parked. Recompute pre-grasp from live bent-arm
    # TCP offset; fixed straight-arm geometry is no longer valid.
    cc = unwenv.cup.pose.p[0].cpu().numpy()
    arm_xy = (
        agent.tcp.pose.p[0].cpu().numpy()[:2]
        - agent.base_link.pose.p[0].cpu().numpy()[:2]
    )
    pre = np.r_[cc[:2] - arm_xy, 0.0]
    res = env.log_motion("Stage 2 pre-grasp", l_drive, pre)
    _sync()
    # correction loop: the screw drive lands ~15 cm off, but the arm's align
    # (pan-85, near the +-92 deg joint limit) can only cover ~8-10 cm, so
    # re-aim the base until the GRIPPER is within ~5 cm of the cup
    for _c in range(3):
        _g = agent.tcp.pose.p[0].cpu().numpy()[:2]
        _cc2 = unwenv.cup.pose.p[0].cpu().numpy()[:2]
        # pi-lens-ignore: unchecked-throwing-call-python
        if float(np.linalg.norm(_g - _cc2)) <= 0.05:
            break
        _b = agent.base_link.pose.p[0].cpu().numpy()[:2]
        # aim the gripper 6 cm west-south of the cup: the drives' overshoot
        # keeps the plates clear of the cup (a 3 cm aim let the plates push
        # the cup ~12 cm)
        _aim2 = np.array([_b[0] + (_cc2[0] - 0.06 - _g[0]),
                          _b[1] + (_cc2[1] - 0.06 - _g[1]), 0.0])
        env.log_motion("Stage 2 correction", l_drive, _aim2)
        _sync()
    # Leave torso high; first Stage 3 screw can lower torso and align arm in
    # one geometry path. Failed reach falls back to the old torso ramp.
    report_stage("2 high pre-grasp")

    # ------------------------------------------------------------------ #
    # STAGE 3: grasp - a SHORT two-step straight-arm motion (inter 5 cm
    # above the cup, then the final 5 cm drop; the arm stays straight, a
    # micro-translation) aligns the jaws on the cup's measured position,
    # then close. Retry with height variations and re-measurement.
    # ------------------------------------------------------------------ #
    env.log_event("phase", "Stage 3: grasp")
    def run_grasp():
        got = False
        torso_ready = False
        for attempt in range(10):
            cc = unwenv.cup.pose.p[0].cpu().numpy()
            raise_z = [0.02, 0.06, 0.02, 0.08, 0.04, 0.0, 0.05, 0.03, 0.07, 0.01][attempt]
            q_now = agent.tcp.pose.q[0].cpu().numpy()
            final = sapien.Pose(p=[cc[0], cc[1], cc[2] + raise_z], q=q_now)
            inter = sapien.Pose(p=[cc[0], cc[1], cc[2] + raise_z + 0.05], q=q_now)
            r1 = env.log_motion("Stage 3 align", planner.static_manipulation,
                                inter, n_init_qpos=100, disable_lift_joint=False)
            _sync()
            if not torso_ready and (r1 == -1 or not _tcp_at(agent, inter, tol=0.05)):
                # The high-to-intermediate screw was not accurate; restore the
                # proven low-torso contract before retrying live geometry.
                ramp_torso(TORSO_GRASP, steps=120)
                torso_ready = True
                continue
            torso_ready = True
            r2 = -1 if r1 == -1 else env.log_motion(
                "Stage 3 align", planner.static_manipulation, final,
                n_init_qpos=100, disable_lift_joint=False)
            _sync()
            aligned = r2 != -1 and _tcp_to(agent, final.p) <= 0.04
            if not aligned:
                cc_live = unwenv.cup.pose.p[0].cpu().numpy()
                live_final = sapien.Pose(
                    p=[cc_live[0], cc_live[1], cc_live[2] + raise_z],
                    q=agent.tcp.pose.q[0].cpu().numpy(),
                )
                aligned = descend_to_grasp(live_final)
            if aligned:
                planner.close_gripper()
                settle_gripper()
                if cup_held():
                    got = True
                    break
                planner.open_gripper()
                _sync()
        return got

    def verify_load_bearing_grasp():
        """Reject contact that closes around the cup but cannot lift it."""
        if not cup_held() or tcp_cup_gap() > 0.08:
            return False
        torso_z = float(hold_b()[2])
        cup_z = float(unwenv.cup.pose.p[0][2])
        ramp_torso(torso_z + 0.04, steps=40)
        valid = (
            cup_held()
            and float(unwenv.cup.pose.p[0][2]) >= cup_z + 0.015
            and tcp_cup_gap() <= 0.12
        )
        ramp_torso(torso_z, steps=40)
        return valid and cup_held() and tcp_cup_gap() <= 0.12

    def fallback_grasp():
        # drive the base closer (arm HIGH - the mid-grasp low-torso drive near
        # the counter fails to move the base) so the cup is inside the
        # workspace, then re-run the arm align. Returns True only after a
        # measured lift proves the fallback contact is load-bearing.
        ramp_torso(TORSO_TRANSPORT, steps=150)
        _b = agent.base_link.pose.p[0].cpu().numpy()[:2]
        _cc = unwenv.cup.pose.p[0].cpu().numpy()[:2]
        # pi-lens-ignore: unchecked-throwing-call-python
        if float(np.linalg.norm(_cc - _b)) > GRASP_STANDOFF:
            _aim = np.array([_cc[0] + 0.10, _cc[1] - GRASP_STANDOFF, 0.0])
            env.log_motion("fallback reach", l_drive, _aim[:2], 0.04)
            _sync()
        # Move the base using the measured TCP/cup offset, then close from the
        # already aligned carry orientation before attempting a new IK pose.
        for _ in range(2):
            tcp_xy = agent.tcp.pose.p[0].cpu().numpy()[:2]
            cup_xy = unwenv.cup.pose.p[0].cpu().numpy()[:2]
            # pi-lens-ignore: unchecked-throwing-call-python
            if float(np.linalg.norm(tcp_xy - cup_xy)) <= 0.05:
                break
            base_xy = agent.base_link.pose.p[0].cpu().numpy()[:2]
            env.log_motion(
                "fallback center", l_drive,
                base_xy + cup_xy - tcp_xy, 0.04
            )
            _sync()
        ramp_torso(TORSO_GRASP, steps=120)
        for _ in range(3):
            planner.close_gripper()
            settle_gripper()
            if verify_load_bearing_grasp():
                return True
            planner.open_gripper()
            _sync()
        if run_grasp():
            if verify_load_bearing_grasp():
                return True
            planner.open_gripper()
            _sync()
        if alternate_grasp():
            if verify_load_bearing_grasp():
                return True
            planner.open_gripper()
            _sync()
        return False

    def alternate_grasp():
        """Try a live front-facing grasp after repeated vertical stalls."""
        mesh = unwenv.cup.get_first_collision_mesh(to_world_frame=True)
        if mesh is None:
            return False
        obb = mesh.bounding_box_oriented
        cc = obb.center_mass.copy()
        ed = cc - agent.tcp.pose.p[0].cpu().numpy()
        ed[2] = 0.0
        if np.linalg.norm(ed) < 1e-6:
            ed = np.array([0.0, 1.0, 0.0])
        ed /= np.linalg.norm(ed)
        closing = np.cross(np.array([0.0, 0.0, 1.0]), ed)
        closing /= np.linalg.norm(closing)
        for raise_z in (0.02, 0.06, 0.04):
            grasp_pose, reach_pose = _grasp_pose(
                agent, obb, cc, ed, closing,
                raise_z=raise_z, back_off=0.06, force_front=True,
            )
            r1 = env.log_motion(
                "Stage 3 alternate approach", planner.static_manipulation,
                reach_pose, n_init_qpos=100, disable_lift_joint=False,
            )
            _sync()
            r2 = -1 if r1 == -1 else env.log_motion(
                "Stage 3 alternate grasp", planner.static_manipulation,
                grasp_pose, n_init_qpos=100, disable_lift_joint=False,
            )
            _sync()
            if r2 != -1 and _tcp_at(agent, grasp_pose, tol=0.05):
                planner.close_gripper()
                _sync()
                if cup_held():
                    return True
                planner.open_gripper()
                _sync()
        return False

    got = run_grasp()
    if not got:
        got = alternate_grasp()
    if not got:
        # FALLBACK: the primary straight-arm grasp could not reach the cup - it
        # sits just past the arm's reachable envelope (base->cup = ARM_OFFSET
        # ~2-5 cm beyond mplib's real reach). Drive the base closer so the cup
        # is inside the workspace, then re-run the arm align. Only fires when
        # the primary grasp failed, so seeds that grasp fine are untouched.
        env.log_event("phase", "Stage 3: fallback arm regrasp")
        got = fallback_grasp()
    if not got:
        print("Grasp failed after retries; aborting")
        env.log_event("error", "Grasp failed")
        success = bool(unwenv.evaluate()["success"].item())
        env.log_event("result", "Task aborted", success=success)
        env.reset()
        return success
    report_stage("3 grasped")

    # ------------------------------------------------------------------ #
    # STAGE 4: lift - raise the torso to the transport height, verify the
    # cup rose with the jaws (>= 0.08 m) and stays near the TCP.
    # ------------------------------------------------------------------ #
    env.log_event("phase", "Stage 4: lift")
    # pi-lens-ignore: unchecked-throwing-call-python
    cup_z0 = float(unwenv.cup.pose.p[0][2])
    ramp_torso(TORSO_GRASP + 0.04, steps=40)
    # pi-lens-ignore: unchecked-throwing-call-python
    probe_cz = float(unwenv.cup.pose.p[0][2])
    if probe_cz >= cup_z0 + 0.015 and tcp_cup_gap() <= 0.12:
        ramp_torso(TORSO_TRANSPORT, steps=110)
    else:
        ramp_torso(TORSO_GRASP, steps=40)
    # pi-lens-ignore: unchecked-throwing-call-python
    cz = float(unwenv.cup.pose.p[0][2])
    if cz < cup_z0 + 0.05 or tcp_cup_gap() > 0.12:
        # RE-GRASP FALLBACK: the cup didn't ride up with the jaws - a
        # false-positive grasp (`is_grasping` fired but the cup never clamped,
        # slipping out on the raise) - seed 48. Instead of aborting, re-grasp
        # (drive base closer + re-run the align) and retry the lift once.
        env.log_event("phase", "Stage 4: re-grasp (lift detect)")
        ramp_torso(TORSO_GRASP, steps=80)
        if fallback_grasp():
            # pi-lens-ignore: unchecked-throwing-call-python
            cup_z0 = float(unwenv.cup.pose.p[0][2])
            ramp_torso(TORSO_TRANSPORT, steps=150)
            # pi-lens-ignore: unchecked-throwing-call-python
            cz = float(unwenv.cup.pose.p[0][2])
    if cz < cup_z0 + 0.05 or tcp_cup_gap() > 0.12:
        print(f"Lift failed (cup z {cz:.3f} vs {cup_z0 + 0.05:.3f}, "
              f"gap {tcp_cup_gap():.3f}); aborting")
        env.log_event("error", "Lift failed")
        success = bool(unwenv.evaluate()["success"].item())
        env.log_event("result", "Task aborted", success=success)
        env.reset()
        return success
    report_stage("4 lifted")

    # ------------------------------------------------------------------ #
    # STAGE 5: transport to the tray - back up (south) for clearance, then
    # the L-drive to (tray_x, tray_y - ARM_OFFSET): the cup over the tray
    # center. Cup-attach monitor after each leg.
    # ------------------------------------------------------------------ #
    env.log_event("phase", "Stage 5: transport to tray")
    b = agent.base_link.pose.p[0].cpu().numpy()
    res = env.log_motion("Stage 5 back up", drive_to,
                         np.array([b[0], b[1] - 0.15, 0.0]), 0.10, 0.01)
    # the cup's z must stay near the stage-4 reference (the arm already
    # lifted it; the back-up does not change the z - the old check compared
    # against a pre-raise reference and falsely fired after the arm lift)
    # pi-lens-ignore: unchecked-throwing-call-python
    if res != 0 or tcp_cup_gap() > 0.15 or float(unwenv.cup.pose.p[0][2]) < cup_z0 + 0.04:
        print("Stage 5 back-up failed / cup lost; aborting")
        env.log_event("error", "Stage 5 back-up failed")
        success = bool(unwenv.evaluate()["success"].item())
        env.log_event("result", "Task aborted", success=success)
        env.reset()
        return success
    # aim the BASE so the CUP (at its live offset from the base - the arm
    # config after the grasp differs from the ideal straight offset) lands
    # on the tray center
    _off = unwenv.cup.pose.p[0].cpu().numpy()[:2] - agent.base_link.pose.p[0].cpu().numpy()[:2]
    aim_tray = np.array([tray_center[0] - _off[0], tray_center[1] - _off[1]])
    res = env.log_motion("Stage 5 drive to tray", l_drive, aim_tray, 0.10)
    # closed-loop correction: the base drives land 3-30 cm off, but the cup's
    # base must sit fully on the tray (center within ~0.105 m); re-aim at the
    # CUP's live error and re-drive up to twice
    for _c in range(2):
        # pi-lens-ignore: unchecked-throwing-call-python
        if float(np.linalg.norm(unwenv.cup.pose.p[0].cpu().numpy()[:2] - tray_center[:2])) <= 0.02:
            break
        _off = unwenv.cup.pose.p[0].cpu().numpy()[:2] - agent.base_link.pose.p[0].cpu().numpy()[:2]
        _aim2 = np.array([tray_center[0] - _off[0], tray_center[1] - _off[1]])
        res = env.log_motion("Stage 5 correction", l_drive, _aim2, 0.10)
        _sync()
    # pi-lens-ignore: unchecked-throwing-call-python
    if res != 0 or tcp_cup_gap() > 0.15 or float(unwenv.cup.pose.p[0][2]) < cup_z0 + 0.04:
        print("Stage 5 drive to tray failed / cup lost; aborting")
        env.log_event("error", "Stage 5 drive to tray failed")
        success = bool(unwenv.evaluate()["success"].item())
        env.log_event("result", "Task aborted", success=success)
        env.reset()
        return success
    report_stage("5 at tray")

    # ------------------------------------------------------------------ #
    # STAGE 6: place - lower the torso SLOWLY until the cup rests on the
    # tray (the cup z stops decreasing).
    # ------------------------------------------------------------------ #
    env.log_event("phase", "Stage 6: lower onto tray")
    # pi-lens-ignore: unchecked-throwing-call-python
    tray_top = float(unwenv.tray.pose.p[0][2] + unwenv.tray_half[2])
    lower_torso_until_cup_rests(tray_top)
    report_stage("6 on tray")

    # ------------------------------------------------------------------ #
    # STAGE 7: release - partial open (the jaws just off the cup), no motion.
    # ------------------------------------------------------------------ #
    env.log_event("phase", "Stage 7: release")
    # VERTICAL release: open the jaws slightly (the squeeze released), then
    # LIFT the torso so the plates rise off the cup. A lateral jaw-open pops
    # the round cup sideways and spins it (measured: 6 cm pop, av ~2.6 rad/s
    # that never decays on the smooth tray - the is_static latch can never
    # pass), because the plates' edges wedge the off-center cup. The vertical
    # lift presses the cup DOWN onto the tray (no lateral kick) and leaves it
    # free, so the cup stays put and quiet.
    for _i in range(30):
        _frac = (_i + 1) / 30
        planner.change_gripper_state(t=1, gripper_state=-1.0 + _frac * 1.85)  # pyright: ignore[reportArgumentType]
    _sync()
    # NO torso lift here: lifting the plates catches the cup's rim and drags
    # it up (measured: the cup rode up to z 1.09 with the rising plates and
    # stayed there, precariously held - the is_static latch failed). The jaw
    # open alone frees the cup at the rest height (measured pop ~1 cm, cup
    # quiet - the same behaviour as the old design's release that latched
    # with av 0.002).
    if cup_held():
        print("Release failed (still grasping); aborting")
    if cup_held():
        print("Release failed (still grasping); aborting")
        env.log_event("error", "Release failed")
        success = bool(unwenv.evaluate()["success"].item())
        env.log_event("result", "Task aborted", success=success)
        env.reset()
        return success
    # let the cup settle (the latch needs is_static); the release's pop can
    # spin the cup and the spin decays slowly (measured: av 0.14 after 125 s),
    # so wait long enough for the is_static latch
    for _ in range(2500):
        env.step(np.hstack([hold_a(), planner.gripper_state, hold_b(), _base_cmd()]))
        # pi-lens-ignore: ast-grep:unchecked-throwing-call-python
        if (float(torch.linalg.norm(unwenv.cup.linear_velocity, dim=1)[0]) <= 0.1
                # pi-lens-ignore: ast-grep:unchecked-throwing-call-python
                and float(torch.linalg.norm(unwenv.cup.angular_velocity, dim=1)[0]) <= 0.2):
            break
    unwenv.evaluate()
    report_stage("7 released")

    print("Task completed. Closing env...")
    ev = unwenv.evaluate()
    success = bool(ev["success"].item())
    print("Success:", success,
          "| cup xy:", np.round(unwenv.cup.pose.p[0].cpu().numpy()[:2], 3),
          "z:", round(float(unwenv.cup.pose.p[0][2]), 3),
          "| v:", round(float(torch.linalg.norm(unwenv.cup.linear_velocity, dim=1)[0]), 4),
          "av:", round(float(torch.linalg.norm(unwenv.cup.angular_velocity, dim=1)[0]), 4))
    env.log_event("result", "Task completed", success=success)
    env.reset()
    return success


if __name__ == "__main__":
    args = parse_args()
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    from mplib.pymp import set_global_seed
    set_global_seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print(f"[INFO] seed={SEED}, render_mode='{args.render_mode}', "
          f"debug={args.debug}, info={args.info}, log_dir='{args.log_dir}'")

    run_id = f"takeitback_tray_seed{SEED}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.log_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        "MyRoboCasa_TakeItBackTray-v1",
        num_envs=1,
        render_mode=None if args.no_video else args.render_mode,
        obs_mode="state" if args.no_video else "rgb",
        robot_uids="ds_fetch",
        control_mode="pd_joint_delta_pos",
        sim_config=dict(scene_config=dict(cpu_workers=1, enable_enhanced_determinism=True)),
    )
    # mp4 side-videos from the three EXTERNAL scene cameras are NOT recorded:
    # trajectory weight. The h5 keeps RGB observations from the robot-mounted
    # cameras (obs_mode="rgb"), which is what the LeRobot converter encodes
    # into per-camera videos.
    if args.no_video:
        print("[INFO] video recording disabled (--no-video)")
    env = RecordEpisode(
        env,
        output_dir=str(run_dir),
        trajectory_name="trajectory",
        save_trajectory=True,
        save_video=False,
        source_type="motionplanning",
        source_desc="TakeItBack tray Fetch motion-planning demonstration",
    )
    env = PlannerLogger(
        env,
        log_dir=str(run_dir),
        name=f"takeitback_tray_seed{SEED}",
        log_freq=args.log_freq,
        run_dir=run_dir,
    )

    env.action_space.seed(SEED)
    with capture_stdout(env.dir / "console.log"):
        planning(env, SEED, debug=args.debug, info=args.info)
    env.close()
    _repair_trajectory_metadata(run_dir)
    if not args.no_video:
        print(f"[INFO] Video recording saved in '{run_dir}/'")
