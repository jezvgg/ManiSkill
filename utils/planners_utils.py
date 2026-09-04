import gymnasium as gym
import numpy as np
import sapien
import torch
import mplib


# persistent cmd->world map estimate for the holonomic base drive: a robot
# property (slowly varying with config), shared across all drive calls so the
# transport's repeated segment fallbacks do not re-calibrate from scratch
# persistent EGO-frame cmd->world map (per unit command per step, resolved
# at heading 0): a robot property, invariant to the base's current heading.
# At each drive call it is rotated into the world frame by the current yaw,
# so heading changes between calls (or the screw planner's own turns) do not
# stale the estimate. It is CALIBRATED by measuring the response to commands
# along the ego axes (no model guessing); recalibrated whenever the heading
# moves more than 25 deg from the heading it was measured at.
_BASE_MAP_EGO = None
_CAL_HEADING = None
_BASE_MAP_DET = None

def _base_cmd(vx=0.0, vy=0.0, w=0.0):
    """Base velocity command for PDBaseVelController: [vx_fwd, vy_left, w_yaw]"""
    return np.array([vx, vy, w])


def lower_torso_smooth(env, planner, target_drop=0.17, total_steps=100, vis=False, arm_action=None, gripper_action=None):
    """
    Lower the torso slowly and smoothly by interpolating the height index.
    """
    unw_env = env.unwrapped
    if arm_action is None:
        arm_action = unw_env.agent.controller.controllers["arm"].qpos[0].cpu().numpy()
    start_body_action = unw_env.agent.controller.controllers["body"].qpos[0].cpu().numpy().copy()
    base_action = _base_cmd()
    if gripper_action is None:
        gripper_action = planner.gripper_state

    for step in range(total_steps):
        fraction = (step + 1) / total_steps
        current_drop = fraction * target_drop

        body_action = start_body_action.copy()
        body_action[2] -= current_drop

        action = np.hstack([arm_action, gripper_action, body_action, base_action])
        env.step(action)

        if vis and hasattr(unw_env, "render_human"):
            unw_env.render_human()

    planner.planner.update_from_simulation()

def retract_arm_lift_torso(env, planner, lift_amount=0.15, total_steps=40, vis=False, arm_action=None, gripper_action=None):
    """
    Lifts torso back up by bypassing the planner.
    """
    unw_env = env.unwrapped
    if arm_action is None:
        arm_action = unw_env.agent.controller.controllers["arm"].qpos[0].cpu().numpy()
    body_action = unw_env.agent.controller.controllers["body"].qpos[0].cpu().numpy().copy()
    body_action[2] += lift_amount
    base_action = _base_cmd()
    if gripper_action is None:
        gripper_action = planner.gripper_state

    action = np.hstack([arm_action, gripper_action, body_action, base_action])

    for _ in range(total_steps):
        env.step(action)
        if vis and hasattr(unw_env, "render_human"):
            unw_env.render_human()

    planner.planner.update_from_simulation()


def move_base_backward_smooth(env, planner, target_base_pos, max_steps=200, eps=0.015, vis=False):
    """
    Smoothly move the base backward toward target_base_pos using simple velocity control.
    """
    unwenv = env.unwrapped
    agent = unwenv.agent
    arm_action = unwenv.agent.controller.controllers["arm"].qpos[0].cpu().numpy()
    body_action = unwenv.agent.controller.controllers["body"].qpos[0].cpu().numpy().copy()
    body_action[0] = body_action[1] = 0.0
    gripper_action = planner.gripper_state

    print(f"[INFO] Smooth base control to target pos: {target_base_pos}")
    for step in range(max_steps):
        planner.planner.update_from_simulation()
        cur_base_p = unwenv.agent.base_link.pose.sp.p.copy()
        cur_base_dir = agent.base_link.pose.sp.to_transformation_matrix()[:3, 0]

        back_delta_world = target_base_pos - cur_base_p
        back_delta_world[2] = 0.0
        dist_to_target = np.linalg.norm(back_delta_world)

        if dist_to_target < eps:
            print(f"[INFO] Reached target base position: {cur_base_p} (dist={dist_to_target:.4f} m, step={step})")
            break

        rem_dist = float(np.dot(back_delta_world, cur_base_dir))
        vel = np.clip(rem_dist * 2.5, -0.6, 0.6)
        base_action = np.array([vel, 0.0, 0.0])

        action = np.hstack([arm_action, gripper_action, body_action, base_action])
        env.step(action)

        if vis and hasattr(unwenv, "render_human"):
            unwenv.render_human()

    planner.planner.update_from_simulation()

def align_arm_over_target(env, planner, source_pos, target_pos, vis=False):
    """
    Align arm horizontally by calculating delta dx, dy between source and target positions.
    """
    unwenv = env.unwrapped
    agent = unwenv.agent
    dx = target_pos[0] - source_pos[0]
    dy = target_pos[1] - source_pos[1]

    if np.hypot(dx, dy) > 0.005:
        print(f"Correction needed: dx={dx:.4f}, dy={dy:.4f}")
        current_tcp_pose = agent.tcp.pose.sp
        target_tcp_p = current_tcp_pose.p.copy()
        target_tcp_p[0] += dx
        target_tcp_p[1] += dy

        result = planner.planner.plan_screw(
            mplib.Pose(target_tcp_p, current_tcp_pose.q),
            planner.robot.get_qpos().cpu().numpy()[0],
            time_step=planner.base_env.control_timestep,
            masked_joints=[True, True, True, False] + [False]*11
        )
        if result["status"] == "Success":
            planner.follow_path(result)
        else:
            print("[WARNING] Alignment screw failed:", result["status"])
        planner.planner.update_from_simulation()

    if hasattr(planner, "render_wait"):
        planner.render_wait()

def _rotate_base_to(env, planner, dir_world, max_rot=300, rot_gain=1.2,
                    rot_cap=0.25, align_deg=4):
    """Rotate the base in place until its x-axis aligns with dir_world.
    Velocity control of the yaw joint is linear and reliable (unlike the
    x/y translation joints)."""
    unwenv = env.unwrapped
    agent = unwenv.agent
    arm_action = agent.controller.controllers["arm"].qpos[0].cpu().numpy()
    body_action = agent.controller.controllers["body"].qpos[0].cpu().numpy().copy()
    body_action[0] = body_action[1] = 0.0
    gripper_action = planner.gripper_state
    dt = np.asarray(dir_world, dtype=float).copy()
    dt[2] = 0.0
    n = np.linalg.norm(dt)
    if n < 1e-6:
        return
    dt /= n
    for _ in range(max_rot):
        xa = agent.base_link.pose.sp.to_transformation_matrix()[:3, 0]
        xa[2] = 0.0
        nx = np.linalg.norm(xa)
        if nx < 1e-6:
            return
        xa /= nx
        # UNWRAPPED error: the shortest wrapped path can jam the yaw joint at
        # its +-180 deg seam (root frame) and freeze the base mid-rotation
        # (verified: the -164 deg path from world -166 froze the base at +164).
        # Monotonic rotation stays in-range and always reaches the target.
        he = np.arctan2(dt[1], dt[0]) - np.arctan2(xa[1], xa[0])
        if abs((he + np.pi) % (2 * np.pi) - np.pi) < np.deg2rad(align_deg):
            return
        ba = _base_cmd(w=float(np.clip(rot_gain * he, -rot_cap, rot_cap)))
        env.step(np.hstack([arm_action, gripper_action, body_action, ba]))
    planner.planner.update_from_simulation()


def drive_base_to_position(env, planner, target_pos, chunk=0.5, max_rot=300,
                           rot_gain=1.2, rot_cap=0.25, align_deg=4,
                           y_guard=True):
    """Drive the base to an arbitrary floor position.

    Navigation primitives, verified empirically on this fork:
    * The yaw joint is linear and reliable, but ANY yaw activity makes the
      base slide +x-world at ~0.003 m/step (a PhysX artifact), and a FAST yaw
      rotation also swings the arm joints (inertia) so hard that the arm can
      collide with nearby fixtures (e.g. the stove), which makes every screw
      plan fail on its start-state check.
    * move_base_forward's screw translates reliably when the base is aligned
      and the start state is collision-free.

    So: rotate with a SHORT burst + HOLD cadence (the hold lets the arm PD
    re-center so the arm never swings into a fixture), re-measuring the
    heading every cadence (the +x slide changes the bearing), then translate
    with short screw chunks, re-aligning between chunks. The +x slide is
    absorbed by the re-measure + the screw chunks.
    """
    unwenv = env.unwrapped
    agent = unwenv.agent
    target = np.asarray(target_pos, dtype=float).copy()
    target[2] = 0.0
    arm_action = agent.controller.controllers["arm"].qpos[0].cpu().numpy()
    body_action = agent.controller.controllers["body"].qpos[0].cpu().numpy().copy()
    body_action[0] = body_action[1] = 0.0
    gripper_action = planner.gripper_state

    def heading_error():
        sp = agent.base_link.pose.sp
        base_p = sp.p.copy()
        base_p[2] = 0.0
        delta = target - base_p
        dist = np.linalg.norm(delta)
        if dist < 1e-6:
            return 0.0, dist, base_p
        xa = sp.to_transformation_matrix()[:3, 0]
        xa[2] = 0.0
        nx = np.linalg.norm(xa)
        if nx < 1e-6:
            return 0.0, dist, base_p
        xa /= nx
        dt = delta / dist
        return np.arctan2(np.cross(xa, dt)[2], np.dot(xa, dt)), dist, base_p

    for _ in range(60):  # outer loop: rotate-cadence or screw chunk
        he, dist, base_p = heading_error()
        if dist < 0.15:
            # close enough: further rotation would only slide the base (the
            # +x slide during yaw activity) and the reach re-aims from the
            # actual position anyway
            return 0
        if y_guard and base_p[1] > -0.95:
            # the rotate-cadence's +x slide can walk an east-facing spawn
            # north of the counter line (verified: seed 18 ended at y=-0.87);
            # give up instead of driving into the counter
            print(f"[INFO] drive_base_to_position: base crossed north of the counter "
                  f"(y={base_p[1]:.2f}); aborting")
            return -1
        # NO rotation for the screw drive: the base is holonomic
        # (PDBaseVelController), so move_base_forward drives straight toward
        # the waypoint from any heading. Rotating first (a forward-only-era
        # leftover) fights the screw: the yaw-joint motion rotates the base
        # link mid-drive and the plan's base-joint frame conversion drifts,
        # so a big rotation (seed 18 spawned 0.27 m from the target and tried
        # a +127 deg turn) sent the base off at a fixed local bearing instead
        # of toward the target. The heading for the REACH is set later by the
        # heading fix, not here.
        delta = target - base_p
        waypoint = base_p + delta * min(1.0, chunk / max(dist, 1e-6))
        # the mplib screw fails EXACTLY on pure -x motions (verified: bearing
        # 180 deg fails, 175/185 deg succeed), so perturb the waypoint in y to
        # break the degeneracy; the 5 cm offset is re-measured next iteration
        waypoint[1] += 0.05
        res = planner.move_base_forward(waypoint, n_init_qpos=100)
        if res == -1:
            _screw_translate_debug(planner, waypoint)
            print("[INFO] drive_base_to_position: screw segment failed, trying shorter")
            waypoint = base_p + delta * min(1.0, 0.25 / max(dist, 1e-6))
            waypoint[1] += 0.05
            res = planner.move_base_forward(waypoint, n_init_qpos=100)
            # leave the dead state and retry
            print("[INFO] drive_base_to_position: screw failed, rotating and retrying")
            # gentle dead-state break: slow rotation (fast rotations near
            # fixtures fling the base - verified: seed 17 flew 3+ m south)
            for _ in range(12):
                env.step(np.hstack([arm_action, gripper_action, body_action,
                                    _base_cmd(w=0.08)]))
            for _ in range(30):
                env.step(np.hstack([arm_action, gripper_action, body_action,
                                    _base_cmd()]))
            continue
        planner.planner.update_from_simulation()
    he, dist, base_p = heading_error()
    if dist < 0.15:
        return 0
    print(f"[INFO] drive_base_to_position: did not converge, {dist:.2f} m from target "
          f"at {np.round(base_p, 3)}")
    return -1


def _screw_base_translate(planner, target_base_pos):
    """Translate the base to target_base_pos with a screw plan that keeps the
    ARM FIXED: move_base_forward frees the arm joints, so its screw replans
    the arm and swings a held object behind the base (which then cannot place
    it). The fixed-arm screw needs the arm held HIGH (above the fixtures) so
    the swept volume stays collision-free. Returns 0 on success, -1 on
    failure."""
    agent = planner.base_env.agent
    tcp_pose = agent.tcp.pose.sp
    base_link_pose = agent.base_link.pose.sp
    delta = np.asarray(target_base_pos, dtype=float) - base_link_pose.p
    delta[2] = 0.0
    target_tcp = mplib.Pose(p=tcp_pose.p + delta, q=tcp_pose.q)
    try:
        result = planner.planner.plan_screw(
            target_tcp,
            planner.robot.get_qpos().cpu().numpy()[0],
            time_step=planner.base_env.control_timestep,
            # base x/y/yaw + torso free, ARM FIXED (masked_joints=True = free)
            masked_joints=[True, True, True, True] + [False] * 11,
        )
    except Exception as e:
        # TOPP-Ra can RAISE (e.g. FailUncontrollable on degenerate paths)
        # instead of returning a failure status; degrade to the caller's
        # velocity fallback like a returned failure
        print(f"[INFO] Transport: fixed-arm screw raised {type(e).__name__}")
        return -1
    if result["status"] != "Success":
        return -1
    # follow_moving_forward, NOT follow_path: the ds_fetch base x/y joints are
    # driven by a VELOCITY controller, so the screw path's position targets
    # (follow_path) never move the base - the path must be executed through
    # the base velocity action (the same executor move_base_forward uses to
    # drive the base to the stances). The arm joints are masked (fixed) in
    # the screw, so the held object stays rigid in the base frame.
    planner.follow_moving_forward(result)
    return 0


def _current_object_pos(env, planner):
    """Re-read the tracked object's live world position from the simulation.
    Uses the PlannerLogger registry (env._objs: name -> (handle, file));
    returns None if nothing is tracked."""
    objs = getattr(env, "_objs", None)
    if not objs:
        return None
    for handle, _f in objs.values():
        try:
            return np.asarray(handle.pose.sp.p, dtype=float).copy()
        except Exception:
            continue
    return None


def _velocity_segment(env, planner, target_pos, arm_action, body_action,
                      gripper_action, speed=0.18, max_bursts=80,
                      burst_steps=6, dead_move=0.01,
                      max_steps=3000, min_improve=0.08, stall_bursts=10,
                      initial_backward=False, target_yaw=None, tol=0.12,
                      y_guard=True):
    """Omnidirectional closed-loop base drive toward target_pos.

    The base chassis is HOLONOMIC (independent root x/y prismatic joints),
    but its response to an ego-frame velocity command is a fixed-but-unknown
    linear map M (URDF axis conventions + PhysX friction). Instead of guessing
    headings and signs, this driver COMMANDS a velocity vector, MEASURES the
    resulting world displacement, incrementally learns M (rank-1 LMS), and
    then steers straight at the target through M^-1. Works through every
    frame/sign quirk of the fork; never rotates in place while translating.

    Guards: abort with -1 on no-progress (dead zone / limit cycle), on the
    base crossing north of the counter front line, or on exhausting the step
    budget. initial_backward is accepted for compatibility and ignored."""
    unwenv = env.unwrapped
    agent = unwenv.agent
    target = np.asarray(target_pos, dtype=float).copy()
    target[2] = 0.0

    def pose_xy():
        p = agent.base_link.pose.p[0].cpu().numpy()[:2]
        q = agent.base_link.pose.q[0].cpu().numpy()
        # z-yaw from quaternion (w,x,y,z): atan2(2(wz+xy), 1-2(y^2+z^2)).
        # NOTE: the naive (q3*q2 + q0*q1)/(1-2(q1^2+q2^2)) variant is WRONG
        # for z-rotations (returns 0 always) and silently broke the map
        # rotation + recalibration trigger; keep this exact form
        yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]),
                         1 - 2 * (q[2] ** 2 + q[3] ** 2))
        return p, yaw

    arm_n = len(np.atleast_1d(arm_action))
    grip_n = len(np.atleast_1d(gripper_action))
    body_n = len(np.atleast_1d(body_action))

    def burst(vx, vy, n, w=0.0):
        a = np.zeros(arm_n + grip_n + body_n + 3)
        a[:arm_n] = arm_action
        a[arm_n:arm_n + grip_n] = gripper_action
        a[arm_n + grip_n:arm_n + grip_n + body_n] = body_action
        a[-3:] = [vx, vy, w]
        p0 = agent.base_link.pose.p[0].cpu().numpy()[:2]
        for _ in range(n):
            env.step(a)
        return agent.base_link.pose.p[0].cpu().numpy()[:2] - p0

    global _BASE_MAP_EGO, _CAL_HEADING, _BASE_MAP_DET
    yaw0 = pose_xy()[1]
    yaw_prev = yaw0
    if _BASE_MAP_EGO is None or abs(yaw0 - _CAL_HEADING) > np.deg2rad(25):
        print(f"[INFO] _velocity_segment: calibrating base map at yaw "
              f"{np.degrees(yaw0):.1f} deg")
        Mc = np.zeros((2, 2))
        # command at FULL amplitude: the base has a static-friction dead zone
        # at low velocities (a 0.3 cmd measured ~0 motion near the fixtures),
        # which makes the calibrated map garbage; 0.8 reliably breaks free
        for j, (ax, ay) in enumerate([(0.6, 0.0), (0.0, 0.6)]):
            Mc[:, j] = burst(ax, ay, 4, w=0.0) / 4 / 0.6
            if np.linalg.norm(Mc[:, j]) < 0.005:
                Mc[:, j] = burst(ax, ay, 8, w=0.0) / 8 / 0.6  # retry, longer burst
            if np.linalg.norm(Mc[:, j]) < 0.005:
                # the FORWARD command is dead: the base's joints can be dead
                # in one direction at some configs (verified seed 9: the x
                # joint barely moved forward at the spawn, yaw 169 deg). The
                # REVERSE response is often alive (the old code's
                # "reverse-burst"); fold the measured reverse into the column
                # so the steering can drive that axis backward.
                Mc[:, j] = -burst(-ax, -ay, 8, w=0.0) / 8 / 0.6
                print(f"[INFO] _velocity_segment: axis {j} dead forward, "
                      f"reverse response {np.round(Mc[:, j], 4)}")
        if np.linalg.det(Mc) < 1e-4:
            # the forward probes are nearly PARALLEL (dead-zone map, seed 9:
            # the spawn at yaw 169, det ~1e-5): the forward response is dead
            # but the REVERSE is often alive there. Probe the reverse of both
            # axes and keep whichever column is stronger.
            for j, (ax, ay) in enumerate([(0.6, 0.0), (0.0, 0.6)]):
                r = -burst(-ax, -ay, 8, w=0.0) / 8 / 0.6
                if np.linalg.norm(r) > np.linalg.norm(Mc[:, j]):
                    Mc[:, j] = r
            print(f"[INFO] _velocity_segment: singular map, reverse-probed "
                  f"Mc={np.round(Mc, 4).tolist()}")
        _CAL_HEADING = yaw0
        c, s = np.cos(yaw0), np.sin(yaw0)
        _BASE_MAP_EGO = np.array([[c, s], [-s, c]]) @ Mc
        global _BASE_MAP_DET
        _BASE_MAP_DET = float(np.linalg.det(Mc))
        print(f"[INFO] _velocity_segment: calibrated Mc={np.round(Mc, 4).tolist()} "
              f"det={_BASE_MAP_DET:.2e}")
    steps = 0
    # initial world map from the start heading; the loop recomputes it from
    # the live yaw every burst
    c0, s0 = np.cos(yaw0), np.sin(yaw0)
    M = np.array([[c0, -s0], [s0, c0]]) @ _BASE_MAP_EGO
    best_dist = float(np.linalg.norm(target[:2] - pose_xy()[0]))
    stalled = 0
    for _ in range(max_bursts):
        base0, yaw_b = pose_xy()
        dvec = target[:2] - base0
        dist = float(np.linalg.norm(dvec))
        if dist < tol and (target_yaw is None or
                            abs((target_yaw - yaw_b + np.pi) % (2 * np.pi) - np.pi)
                            < np.deg2rad(8)):
            # BRAKE: the last burst's velocity target persists in the PD
            # controller, so the base coasts up to ~0.2 m past the target
            # before the next phase's action zeroes it (verified seed 18: the
            # base overshot the stance by 0.16 m and the reach then knocked
            # the cup off the counter edge). Hold zero velocity for a few
            # steps to actually stop.
            a0 = np.zeros(arm_n + grip_n + body_n + 3)
            a0[:arm_n] = arm_action
            a0[arm_n:arm_n + grip_n] = gripper_action
            a0[arm_n + grip_n:arm_n + grip_n + body_n] = body_action
            for _ in range(30):
                env.step(a0)
            return 0
        if dist > best_dist + 1.0:
            print(f"[INFO] _velocity_segment: diverging (dist {dist:.2f} m); aborting")
            break
        if dist < best_dist - min_improve:
            best_dist = dist
            stalled = 0
        else:
            stalled += 1
            if stalled >= stall_bursts:
                print(f"[INFO] _velocity_segment: no progress toward target "
                      f"(best {best_dist:.2f} m, now {dist:.2f} m); aborting")
                break
        _, yaw = pose_xy()
        if yaw > -np.deg2rad(2) or yaw < -np.deg2rad(178):
            pass  # heading free: no in-place rotations during translation
        base_y = agent.base_link.pose.p[0].cpu().numpy()[1]
        if y_guard and base_y > -0.95:
            if dvec[1] < -0.02:
                # north of the counter line, but the TARGET is south of the
                # base: the closed loop would steer back south on its own,
                # but the guard fires before it can move. Allow the drive -
                # the transport's west translations drift north on a
                # near-singular base map, and aborting here deadlocks it
                # (observed: 13+ retries stuck at y=-0.94, each aborting on
                # the first loop check before any motion).
                pass
            else:
                print(f"[INFO] _velocity_segment: base crossed north of the counter "
                      f"(y={base_y:.2f}); aborting")
                break
        _bx = agent.base_link.pose.p[0].cpu().numpy()[0]
        # the base SPAWNS at x~3.4 (east of the counter); allow the spawn
        # corridor but still catch the transport's eastward wander (x=5.1)
        if _bx > 3.6 or _bx < 0.35:
            print(f"[INFO] _velocity_segment: base left the driving corridor "
                  f"(x={_bx:.2f}); aborting")
            break

        # steer: solve the ego command that produces world motion along dvec;
        # scale the speed down with the remaining distance so short final
        # hops land inside the convergence radius instead of overshooting
        wdir = dvec / dist
        v_des = wdir * min(speed, 1.5 * dist)
        # the world map = R(current yaw) @ ego map, recomputed EVERY burst:
        # the base's yaw DRIFTS during the drive (PhysX slide, up to 40 deg)
        # and a map frozen at the start heading steers into a circle
        cb, sb = np.cos(yaw_b), np.sin(yaw_b)
        M = np.array([[cb, -sb], [sb, cb]]) @ _BASE_MAP_EGO
        try:
            # damped pseudo-inverse: the measured map is near-SINGULAR in the
            # PhysX dead zones (det ~2e-5 vs ~5e-4 healthy), where the exact
            # solve explodes and the clamped command drives the base the
            # WRONG way. The ridge keeps the command along the alive axis.
            gram = M @ M.T + 1e-4 * np.eye(2)
            cmd = M.T @ np.linalg.solve(gram, v_des)
        except np.linalg.LinAlgError:
            cmd = wdir
        n_cmd = float(np.linalg.norm(cmd))
        if n_cmd > 1.0:
            cmd /= n_cmd

        n_burst = max(2, min(burst_steps, int(burst_steps * dist / 0.25)))
        # rotate toward the target heading WHILE translating: the in-place
        # rotation slides the base into dead zones near the fixtures, but the
        # same rotation spread over the drive is absorbed by the closed loop
        w = 0.0
        if _BASE_MAP_DET is not None and _BASE_MAP_DET < 1e-4:
            # singular dead-zone map: the base's translation is 1-D, and the
            # alive axis ROTATES with the heading (measured: world-north at
            # yaw 33.6, world-east at yaw 128.6). A constant spin sweeps the
            # alive axis so every direction becomes reachable; the closed-loop
            # steering + the per-burst map rotation absorb it.
            w = 0.25
        elif target_yaw is not None:
            # UNWRAPPED error: the yaw JOINT's travel is limited to +-180 deg
            # (root frame), so the shortest wrapped path can jam the joint at
            # the seam and freeze the base mid-rotation (verified: the wrapped
            # -164 deg path from world -166 froze the base at world +164).
            # Monotonic rotation toward the target stays in-range and always
            # reaches it, at the cost of taking the long way when needed.
            he = target_yaw - pose_xy()[1]
            w = float(np.clip(1.5 * he, -0.2, 0.2))
        moved = burst(float(cmd[0]), float(cmd[1]), n_burst, w=w)
        steps += n_burst
        m = float(np.linalg.norm(moved))


        # the base heading DRIFTS while translating (PhysX artifact, ~0.35
        # deg/step): rotate the world-frame map to the newly measured heading
        # so the steering does not lag the rotating frame
        yaw_now = pose_xy()[1]
        dy = yaw_now - yaw_prev
        if abs(dy) > 1e-6:
            cd, sd = np.cos(dy), np.sin(dy)
            M = np.array([[cd, -sd], [sd, cd]]) @ M
            yaw_prev = yaw_now

        # incremental rank-1 LMS update of M from the observed response
        denom = float(cmd @ cmd)
        if denom > 1e-9 and m > 1e-6:
            per_step = moved / n_burst
            errv = per_step - M @ cmd
            M += 0.2 * np.outer(errv, cmd) / denom

        if steps < 24:
            stalled = 0  # warmup: let the map calibration converge first
        if m < dead_move:
            stalled += 2  # dead burst counts double toward the stall guard
        if steps > max_steps:
            print(f"[INFO] _velocity_segment: step budget exhausted; aborting")
            break
    yaw1 = pose_xy()[1]
    c1, s1 = np.cos(yaw1), np.sin(yaw1)
    np.copyto(_BASE_MAP_EGO, np.array([[c1, s1], [-s1, c1]]) @ M)
    planner.planner.update_from_simulation()
    base = agent.base_link.pose.p[0].cpu().numpy()[:2]
    ok = float(np.linalg.norm(target[:2] - base)) < tol
    print(f"[INFO] _velocity_segment done: {'OK' if ok else 'FAIL'} "
          f"final dist {float(np.linalg.norm(target[:2] - base)):.3f} m")
    return 0 if ok else -1

def lower_torso_until_rest(env, planner, target_drop, *, chunk=0.02,
                           steps_per_chunk=12, rest_tol=0.002, vis=False):
    """Lower the torso toward target_drop in small chunks, stopping as soon
    as the held vegetable STOPS DESCENDING (it rests on the surface below -
    typically the plate top).

    A single fixed-size drop overshoots once the object touches down: the
    position-controlled gripper keeps pressing it into the plate, and the
    stored squeeze launches the object when the gripper finally opens."""
    unwenv = env.unwrapped
    agent = unwenv.agent
    arm_action = agent.controller.controllers["arm"].qpos[0].cpu().numpy()
    body_action = agent.controller.controllers["body"].qpos[0].cpu().numpy().copy()
    base_action = _base_cmd()
    gripper_action = planner.gripper_state
    obj = None
    for handle, _f in getattr(env, "_objs", {}).values():
        obj = handle
        break

    def veg_z():
        unwenv_ = env.unwrapped
        scene = getattr(unwenv_, "scene", None)
        if scene is not None and getattr(scene, "gpu_sim_enabled", False):
            scene._gpu_fetch_all()
        return float(obj.pose.p[0].cpu().numpy()[2])

    lowered = 0.0
    prev_z = veg_z()
    while lowered < target_drop - 1e-6:
        step_drop = min(chunk, target_drop - lowered)
        for _ in range(steps_per_chunk):
            ba = body_action.copy()
            ba[2] -= step_drop * (1.0 / steps_per_chunk)
            env.step(np.hstack([arm_action, gripper_action, ba, base_action]))
            if vis and hasattr(unwenv, "render_human"):
                unwenv.render_human()
        lowered += step_drop
        planner.planner.update_from_simulation()
        z_now = veg_z()
        if abs(prev_z - z_now) < rest_tol:
            break  # vegetable rests on the surface below
        prev_z = z_now
    planner.planner.update_from_simulation()


def _yaw_sweep_with_pass_check(env, planner, bearing, plate_center, *,
                               target_obj=None, rot_cap=0.12,
                               align_deg=6.0, pass_dxy=0.09,
                               max_steps=300):
    """Rotate the base toward `bearing` in small increments, checking the held
    vegetable's horizontal distance to the plate after EVERY increment.

    The vegetable rides on an orbit of ~0.58 m around the base; whenever the
    plate lies close to that orbit, the yaw sweep carries the vegetable right
    over it. Catching the pass here avoids long arm alignments at the
    workspace edge later.

    Returns True if the sweep caught a veg-over-plate pass (dxy <= pass_dxy),
    False otherwise (aligned without a pass, or the vegetable dropped)."""
    unwenv = env.unwrapped
    agent = unwenv.agent
    arm_action = agent.controller.controllers["arm"].qpos[0].cpu().numpy()
    body_action = agent.controller.controllers["body"].qpos[0].cpu().numpy().copy()
    body_action[0] = body_action[1] = 0.0
    gripper_action = planner.gripper_state

    def hd():
        m = agent.base_link.pose.sp.to_transformation_matrix()[:3, 0]
        return float(np.arctan2(m[1], m[0]))

    def veg_plate_dxy():
        o = (
            np.asarray(target_obj.pose.sp.p, dtype=float)
            if target_obj is not None
            else _current_object_pos(env, planner)
        )
        if o is None:
            return None
        return float(np.linalg.norm(np.asarray(plate_center)[:2] - o[:2]))

    tgt = float(np.arctan2(bearing[1], bearing[0]))
    for _ in range(max_steps):
        dxy = veg_plate_dxy()
        if dxy is not None and dxy <= pass_dxy:
            return True
        e = ((tgt - hd() + np.pi) % (2 * np.pi)) - np.pi
        if abs(e) < np.deg2rad(align_deg):
            return False
        va = float(np.clip(0.9 * e, -rot_cap, rot_cap))
        env.step(np.hstack([arm_action, gripper_action, body_action,
                            _base_cmd(w=va)]))
    return False


def _drive_base_chunk(planner, target_base_pos, *, speed_scale=0.35):
    """One base-translation chunk planned as a TCP screw (base + torso + arm
    free - the arm tracks the target smoothly so the held object rides
    stably) and executed SLOWLY through the base velocity channel.

    follow_moving_forward feeds the path velocities straight into the base
    velocity controller; unscaled TOPP profiles reach ~1 m/s, and a jolt like
    that rips a shallow fingertip-pinch vegetable out of the gripper."""
    agent = planner.base_env.agent
    tcp_pose = agent.tcp.pose.sp
    base_link_pose = agent.base_link.pose.sp
    delta = np.asarray(target_base_pos, dtype=float) - base_link_pose.p
    delta[2] = 0.0
    target_tcp = mplib.Pose(p=tcp_pose.p + delta, q=tcp_pose.q)
    mask = [True, True, True] + [False] + [True] * 11
    try:
        result = planner.planner.plan_screw(
            target_tcp, planner.robot.get_qpos().cpu().numpy()[0],
            time_step=planner.base_env.control_timestep, masked_joints=mask)
    except Exception:
        return -1
    if not str(result.get("status", "")).startswith("Success"):
        return -1
    if "velocity" in result:
        v = result["velocity"] * speed_scale
        # smooth the TOPP start/stop spikes (moving average, sum preserved):
        # raw profiles jump ~1 m/s between steps and jolt the held vegetable
        k = np.ones(5) / 5.0
        for c in range(v.shape[1]):
            v[:, c] = np.convolve(v[:, c], k, mode="same")
        if len(v) > 1:  # convolution sags the edges - restore them
            v[0] = v[1]
            v[-1] = v[-2]
        result["velocity"] = v
    planner.follow_moving_forward(result)
    return 0


def _prop_forward_transport(env, planner, plate_center, *,
                            stop_dist=0.10, max_cycles=120, fwd_steps=16,
                            k_gain=1.2, v_max=0.18, align_deg=20.0,
                            stall_cycles=6):
    """Polar differential-drive transport for the held vegetable.

    Control law per cycle (all quantities re-MEASURED from the sim):
      e   = plate_pos - veg_pos          error in the vegetable's position
      rho = |e|                          remaining distance
      alpha = wrap(bearing(e) - heading) misalignment of the base heading

      rho > 0.30: if |alpha| large -> gentle yaw toward the bearing;
                  else slow forward burst (speed ~ min(v_max, rho - 0.25))
      rho <= 0.30: pure gentle yaw - the vegetable rides an orbit around the
                  base, so yawing swings it onto the plate without any
                  translation (the translation primitive is unreliable)

    Slow speeds everywhere: a shallow fingertip pinch holds statically but a
    fast yaw/translation jolt shakes flat vegetables loose.

    Returns 0 when the vegetable is within stop_dist of the plate center,
    -1 on stall (no progress over stall_cycles) or vegetable drop.
    """
    unwenv = env.unwrapped
    agent = unwenv.agent
    arm_action = agent.controller.controllers["arm"].qpos[0].cpu().numpy()
    body_action = agent.controller.controllers["body"].qpos[0].cpu().numpy().copy()
    body_action[0] = body_action[1] = 0.0
    gripper_action = planner.gripper_state

    backward = False
    stall = 0
    prev_dist = None
    for _ in range(max_cycles):
        obj_now = _current_object_pos(env, planner)
        if obj_now is None:
            break
        rem = np.asarray(plate_center, dtype=float) - obj_now
        rem[2] = 0.0
        dist = float(np.linalg.norm(rem))
        if dist <= stop_dist:
            return 0
        if obj_now[2] < 0.5:
            print("[INFO] prop drive: vegetable dropped")
            return -1
        # polar control: alpha = bearing error of the base heading relative
        # to the veg->plate direction
        evec = np.asarray(plate_center, dtype=float)[:2] - agent.base_link.pose.p[0].cpu().numpy()[:2]
        bearing = float(np.arctan2(evec[1], evec[0]))
        h = float(np.arctan2(
            agent.base_link.pose.sp.to_transformation_matrix()[1, 0],
            agent.base_link.pose.sp.to_transformation_matrix()[0, 0]))
        alpha = (bearing - h + np.pi) % (2 * np.pi) - np.pi

        if dist > 0.30 and abs(alpha) > np.deg2rad(align_deg):
            # far and misaligned: gentle yaw toward the bearing
            va = float(np.clip(1.5 * alpha, -0.22, 0.22))
            env.step(np.hstack([arm_action, gripper_action, body_action,
                                _base_cmd(w=va)]))
        elif dist > 0.30:
            # aligned: slow proportional forward burst toward the plate
            for _ in range(fwd_steps):
                base = agent.base_link.pose.p[0].cpu().numpy()[:2]
                to_plate = np.asarray(plate_center, dtype=float)[:2] - base
                hd = agent.base_link.pose.sp.to_transformation_matrix()[:3, 0][:2]
                hd = hd / max(float(np.linalg.norm(hd)), 1e-6)
                fwd_left = float(np.dot(to_plate, hd))
                vel = float(np.clip(k_gain * fwd_left, -v_max, v_max))
                if abs(vel) < 0.02:
                    break
                env.step(np.hstack([arm_action, gripper_action, body_action,
                                    np.array([-vel if backward else vel,
                                              0.0, 0.0])]))
        else:
            # near: pure gentle yaw - the veg rides its orbit onto the plate
            va = float(np.clip(1.5 * alpha, -0.18, 0.18))
            env.step(np.hstack([arm_action, gripper_action, body_action,
                                _base_cmd(w=va)]))
        planner.planner.update_from_simulation()

        obj_now = _current_object_pos(env, planner)
        if obj_now is None:
            break
        dist_now = float(np.linalg.norm(np.asarray(plate_center)[:2] - obj_now[:2]))
        if dist_now <= stop_dist:
            return 0
        progressed = prev_dist is not None and dist_now < prev_dist - 0.005
        prev_dist = dist_now
        if progressed:
            stall = 0
            continue
        stall += 1
        if stall == 1:
            # flip direction once per stall: some root-z configs only
            # translate backwards
            backward = not backward
        elif stall >= stall_cycles:
            print(f"[INFO] prop drive: stalled at {dist_now:.2f} m from plate")
            return -1
    obj_now = _current_object_pos(env, planner)
    if obj_now is not None:
        d = float(np.linalg.norm(np.asarray(plate_center)[:2] - obj_now[:2]))
        return 0 if d <= stop_dist else -1
    return -1


def drive_base_to_object_target(env, planner, current_obj_pos, target_obj_pos,
                                margin=0.04, fixed_arm=False, yaw_sweep=True,
                                screw=True, vtol=0.12):
    """Transport a grasped object from current_obj_pos to target_obj_pos.

    The carried object is rigid in the base frame, so ANY base rotation sweeps
    it along an arc. The safe sequence is therefore: (1) rotate the base in
    place to face the transfer direction (the object only spins), then
    (2) translate straight forward while re-measuring the remaining offset -
    no rotation happens while translating.
    """
    unwenv = env.unwrapped
    agent = unwenv.agent
    base_pos_world = agent.base_link.pose.sp.p.copy()

    delta_world = np.asarray(target_obj_pos, dtype=float) - np.asarray(current_obj_pos, dtype=float)
    delta_world[2] = 0.0
    dist = np.linalg.norm(delta_world)
    if dist <= 1e-3:
        return
    dir_world = delta_world / dist
    base_target_pos = base_pos_world + dir_world * (dist + margin)

    print(f"[INFO] Transport: base {np.round(base_pos_world, 3)} -> "
          f"{np.round(base_target_pos, 3)} (object move {dist:.3f} m)")

    # 1) fixed-arm transport: closed-loop proportional drive (yaw-align +
    #    proportional forward bursts). The screw path cannot translate the
    #    ds_fetch base at all (follow_path sends position targets to a base
    #    that is velocity-controlled), and the axis-velocity fallback stalls;
    #    the P-drive with per-cycle yaw re-alignment is the only primitive
    #    that reliably closes the gap. Falls back to the screw chunks below
    #    if the P-drive stalls.
    if fixed_arm:
        if _prop_forward_transport(env, planner,
                                   np.asarray(target_obj_pos, dtype=float)) == 0:
            return

    # 2) rotate in place toward the transfer direction, sweeping the held
    #    object on its orbit: if the sweep carries it over the plate, stop
    #    immediately (the caller lowers and releases without further driving)
    #    yaw_sweep=False skips these rotations: the holonomic base controller
    #    translates in any direction directly, and every base rotation swings
    #    the held vegetable (inertia pulls it out of the fingers - observed
    #    mid-transport drops).
    if yaw_sweep and _yaw_sweep_with_pass_check(env, planner, dir_world, target_obj_pos):
        planner.planner.update_from_simulation()
        return
    planner.planner.update_from_simulation()

    # 2) translate straight forward; re-measure the object offset each chunk so
    #    small drifts are corrected without any further base rotation
    unwenv = env.unwrapped
    agent = unwenv.agent
    arm_action = agent.controller.controllers["arm"].qpos[0].cpu().numpy()
    body_action = agent.controller.controllers["body"].qpos[0].cpu().numpy().copy()
    body_action[0] = body_action[1] = 0.0
    gripper_action = planner.gripper_state
    for _ in range(20):
        obj_now = _current_object_pos(env, planner)
        if obj_now is None:
            break
        rem = np.asarray(target_obj_pos, dtype=float) - obj_now
        rem[2] = 0.0
        rem_dist = np.linalg.norm(rem)
        if rem_dist < 0.04:
            break
        # the held veg orbits the base and repeatedly sweeps past the target
        # mid-drive; stop as soon as it is over the plate (within the release
        # radius) - overshoot beyond this is caught by the yaw-sweep pass
        # check below, so driving closer is safe
        if rem_dist < 0.10:
            break
        # re-align the base heading to the CURRENT bearing with a yaw sweep
        # that watches for a veg-over-plate pass on every increment (the
        # sweep stops the moment the veg passes over the plate)
        if yaw_sweep and _yaw_sweep_with_pass_check(env, planner, rem / rem_dist,
                                                    target_obj_pos):
            break
        sp = agent.base_link.pose.sp
        base_p = sp.p.copy()
        base_p[2] = 0.0
        # drive the base along the remaining object direction (the held object
        # is rigid in the base frame); the screw translates reliably when the
        # heading is aligned, the axis velocity drive is the fallback. The
        # base's lateral motion is unreliable (controller-frame quirk), so the
        # waypoint is clamped to stay south of the counter front (front face at
        # y=-0.65 minus the ~0.35 m base radius) or the base wedges into it.
        waypoint = base_p + rem / rem_dist * min(rem_dist, 0.5)
        # the mplib screw fails EXACTLY on pure -x motions: perturb the
        # waypoint in y to break the degeneracy (5 cm, re-measured next chunk).
        # The perturb must be applied BEFORE the clamp - applied after, it
        # pushes the waypoint north of the -1.0 guard and wedges the base into
        # the counter front (face at y=-0.65).
        waypoint[1] = min(waypoint[1] + 0.05, -1.0)
        if screw:
            res = _drive_base_chunk(planner, waypoint)
            if res == -1:
                print("[INFO] Transport: screw segment failed, using axis velocity drive")
                res = _velocity_segment(env, planner, waypoint, arm_action,
                                        body_action, gripper_action, speed=0.18,
                                        tol=vtol)
                if res == -1:
                    print("[INFO] Transport: giving up at", base_p)
                    break
        else:
            # velocity-only: the screw replans the ARM while the base moves,
            # and the moving fingers let a smooth vegetable slip out (observed
            # mid-transport drops). The velocity drive keeps the arm frozen.
            res = _velocity_segment(env, planner, waypoint, arm_action,
                                    body_action, gripper_action, speed=0.18,
                                    tol=vtol)
            if res == -1:
                print("[INFO] Transport: giving up at", base_p)
                break
        planner.planner.update_from_simulation()

def _screw_translate_debug(planner, waypoint):
    """DIAGNOSTIC: instrumented copy of SapienPlannerV2.plan_screw for base
    translation. Runs the same damped-IK screw loop as move_base_forward but
    reports WHICH termination condition fires (collide / joint_limit /
    zero_twist) instead of the generic 'screw plan failed'. Plan-only: the
    planning world is restored afterwards."""
    pl = planner.planner
    tcp = planner.base_env.agent.tcp.pose.sp
    base_link = planner.base_env.agent.base_link.pose.sp
    delta = np.asarray(waypoint, dtype=float) - base_link.p
    delta[2] = 0.0
    target = mplib.Pose(p=tcp.p + delta, q=tcp.q)

    masked_joints = [True, True, True] + [False] + [True] * 11

    pl_robot = pl.robot
    world = pl.planning_world
    orig_full = np.array(pl_robot.get_qpos()).reshape(-1).copy()
    current_qpos = pl.pad_move_group_qpos(orig_full.copy())
    move_joint_idx = pl.move_group_joint_indices
    orig_move_qpos = current_qpos[move_joint_idx].copy()
    def restore():
        world.set_qpos_all(orig_move_qpos)
        pl_robot.set_qpos(orig_full, True)

    def skew(vec):
        return np.array([[0, -vec[2], vec[1]],
                         [vec[2], 0, -vec[0]],
                         [-vec[1], vec[0], 0]])

    def rot2so3(R):
        tr = float(R.trace())
        if np.isclose(tr, 3.0):
            return np.zeros(3), 1.0
        if np.isclose(tr, -1.0):
            return np.zeros(3), -1e6
        theta = np.arccos((tr - 1) / 2)
        return (np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0],
                          R[1, 0] - R[0, 1]]).T / (2 * np.sin(theta))), theta

    def pose2exp(pose):
        M = pose.to_transformation_matrix()
        omega, theta = rot2so3(M[:3, :3])
        ss = skew(omega)
        inv_left_jac = (np.eye(3) / theta - 0.5 * ss
                        + (1.0 / theta - 0.5 / np.tan(theta / 2)) * ss @ ss)
        v = inv_left_jac @ M[:3, 3]
        return np.concatenate([v, omega]), theta

    pl_robot.set_qpos(current_qpos, True)
    pm = pl.pinocchio_model
    goal = pl._transform_goal_to_wrt_base(target)
    ee = pl.link_name_2_idx[pl.move_group]
    pm.compute_forward_kinematics(current_qpos)
    rel = goal * pm.get_link_pose(ee).inv()
    omega, theta = pose2exp(rel)
    if theta < -1e4:
        print("[DEBUG-screw] FAIL: rotation singularity (pose2exp theta)")
        restore()
        return
    omega = omega.reshape((-1, 1)) * theta

    stats = {"ok_steps": 0, "collide": 0, "joint_limit": 0, "zero_twist": 0}
    reasons = []
    while True:
        pm.compute_full_jacobian(current_qpos)
        J = pm.get_link_jacobian(ee, local=False)
        J = J * np.tile(np.asarray(masked_joints), (J.shape[0], 1)).astype(np.int32)
        delta_q = np.linalg.pinv(J) @ omega
        n = float(np.linalg.norm(delta_q))
        if n < 1e-9:
            reasons.append("pinv_zero_delta")
            break
        delta_q *= 0.1 / n
        delta_twist = J @ delta_q
        flag = False
        if np.linalg.norm(delta_twist) > np.linalg.norm(omega):
            ratio = np.linalg.norm(omega) / np.linalg.norm(delta_twist)
            delta_q = delta_q * ratio
            delta_twist = delta_twist * ratio
            flag = True
        current_qpos += delta_q.reshape(-1)
        omega -= delta_twist
        within = bool(np.all((current_qpos >= pl.joint_limits[:, 0] - 1e-3)
                             & (current_qpos <= pl.joint_limits[:, 1] + 1e-3)))
        world.set_qpos_all(current_qpos[move_joint_idx])
        collide = bool(world.is_state_colliding())
        pairs = ""
        if collide:
            try:
                res = world.check_robot_collision()
                names = sorted({f"{r.object_name1}:{r.link_name1} <-> {r.object_name2}:{r.link_name2}"
                                for r in res})
                pairs = " PAIRS[" + "; ".join(names[:5]) + "]"
            except Exception as exc:
                pairs = f" PAIRS[query failed: {exc}]"
        if float(np.linalg.norm(delta_twist)) < 1e-4:
            stats["zero_twist"] += 1
            reason = "zero_twist"
        elif collide:
            stats["collide"] += 1
            reason = "collide"
        elif not within:
            stats["joint_limit"] += 1
            reason = "joint_limit"
        else:
            stats["ok_steps"] += 1
            if flag:
                break
            continue
        viol = [(i, float(current_qpos[i])) for i in range(len(current_qpos))
                if current_qpos[i] < pl.joint_limits[i][0] - 1e-3
                or current_qpos[i] > pl.joint_limits[i][1] + 1e-3]
        worst = ",".join(f"j{i}={v:.2f}" for i, v in viol[:4]) if viol else ""
        reasons.append(f"{reason}@step{stats['ok_steps'] + 1}{pairs}"
                       + (f" viol[{worst}]" if worst else ""))
        break

    restore()
    print(f"[DEBUG-screw] stats={stats} | "
          f"{'; '.join(reasons) if reasons else 'SUCCESS'} | "
          f"start base qpos xy={np.round(orig_move_qpos[[0, 1]], 3)}")
