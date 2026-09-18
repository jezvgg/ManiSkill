import os
import re
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

from .._compat import build_two_finger_gripper_grasp_pose_visual
from .base_yaw import (HeldBodies, arc_pull_cmd, jammed_turn, new_contacts,
                       sweep_yaw, yaw_path)
from .stepping import (StepGuard, draw_rank, path_is_a_plan, pose_error,
                       refine_should_stop, report)
from .utils import SapienPlannerV2, SapienPlanningWorldV2

#: Seconds RRTConnect may spend before it gives up (`plan_pose`). Overridable with
#: `MIKASA_PLANNING_TIME` so the budget can be swept without editing four call
#: sites. Note what this does and does not buy: RRTConnect stops the moment its two
#: trees meet and never improves the path afterwards, so a larger budget lowers the
#: rate of `no plan` refusals and does **not** straighten the trajectory. The one
#: site that used double this value keeps doing so.
PLANNING_TIME = float(os.environ.get("MIKASA_PLANNING_TIME", "2"))

#: RRTConnect's maximum edge length. Raw tree nodes are this far apart (K70), and an edge
#: is collision-checked at `longestValidSegment` granularity, so a shorter edge is more
#: likely to be checked through its interior rather than only at its ends — which is the
#: mechanism K76's keep-out exists to defeat (paths returned "collision-free" while
#: skimming an obstacle). Overridable with `MIKASA_RRT_RANGE`; hardcoded 0.1 at four call
#: sites before K79v, and never measured.
#:
#: **Measured now, and 0.05 does not pay** (K79v). 180 seeds per arm, interleaved:
#: 0.1 -> dev 76/80, held 96/100; 0.05 -> dev 74/80, held 98/100. **172/180 either way** —
#: the best held-out number of the session and the worst dev number, cancelling exactly.
#: The mechanism shows in the refusal count: 0.05 gives **507 refusals against 374** (+36%)
#: for the same mean path length (55 vs 57 knots), because a finer tree needs more nodes to
#: span the same distance and so connects less often inside the wall-clock budget. The
#: ladder recovers most of them, which is why the score nets flat while episodes run slower.
#: Left at 0.1; the override stays so the sweep is cheap to repeat.
RRT_RANGE = float(os.environ.get("MIKASA_RRT_RANGE", "0.1"))

#: Measure-only. When `MIKASA_SCREW_SPLIT_PROBE` is set, every RRT fallback in
#: `static_manipulation` first asks whether the SAME target would plan as two short
#: screws through its midpoint, and prints `[screw_split]`. It plans and restores;
#: nothing is executed, no episode step is spent, and with the variable unset not a
#: line of it runs — so a sweep without it is byte-identical to one on main.
SCREW_SPLIT_PROBE = bool(os.environ.get("MIKASA_SCREW_SPLIT_PROBE"))

#: Draws RRTConnect is given in `static_manipulation`'s fallback when the caller asked
#: for no knife of its own — the shortest is executed. RRTConnect is randomized, so two
#: draws for the same reachable goal come back with wildly different path lengths (K58
#: measured it on the water plants' pour), and W29 measured what that costs when nobody
#: chooses: four of the five benchmark tasks execute the FIRST path returned, and an RRT
#: path runs 76-111 knots against a screw plan's 19-32. A draw executes nothing, so the
#: price is wall clock and never simulation steps; `plan=None` is 96 % `IK Failed`
#: (W29), which returns before RRTConnect starts, so a refused goal stays nearly free.
#: 1 restores the pre-W29 behaviour exactly. Overridable with `MIKASA_RRT_ARM_DRAWS`.
RRT_ARM_DRAWS = int(os.environ.get("MIKASA_RRT_ARM_DRAWS", "3"))
#: The joint-line approach's second try: a line to the pose this far ABOVE the target,
#: then the screw down (SeasonDish 2026-09-06, g74). 0 = direct lines only.
LINE_VIA_UP = float(os.environ.get("MIKASA_LINE_VIA_UP", "0.15"))
#: Largest share of a drawn path that may be inside the environment and still be executed
#: BY A CALLER THAT ASKS. Never a global default, and that is a measured decision: as a
#: default it refused 257 plans across 30 DepthRecall episodes and took that task from
#: 26/30 to 13/30, because an arm working inside a shelf intrudes by this metric as a
#: matter of course. Where it does belong is a leg that ends at a loose object, which is
#: what it was measured on: SeasonDish seed 204 executed a path with 98 of its 191 knots
#: inside the scene, shoved the bottle 11.2 cm, and lost the episode with `no plan`.
SKIM_REFUSE_FRAC = float(os.environ.get("MIKASA_SKIM_REFUSE_FRAC", str(1 / 3)))

#: Retry a refused arm screw once with the joint it jammed on held still (`0` disables).
#: W29 measured this over 404 jams on the 2026-09-03 baseline: 36 % of them plan on the
#: retry, and the plan that comes back runs 39-59 knots where the RRT fallback it
#: replaces runs 76-111. The refusal names the joint, the arm has seven for a twist that
#: needs six, and the retry is gated on the same `ARM_SCREW_GOAL_TOLERANCE` as the first
#: screw — so nothing is silenced and a Success is a Success by the standard in force.
ARM_SCREW_UNJAM = os.environ.get("MIKASA_ARM_SCREW_UNJAM", "1") == "1"

#: May `drive_base` approach a dock **backwards** when that turns the base less
#: (`0` disables). The primitive already drives either way and always has:
#: `follow_moving_forward` commands the base with `dot(planned world velocity, the base's
#: own +x)`, so a robot standing with its back to the target gets a negative speed and
#: reverses. What was missing is that `drive_base` only ever aimed the base AT the target,
#: which costs a turn of `pi - theta` where `theta` would have done.
#:
#: The owner allowed reverse driving on 2026-09-03; W29's A/B then measured the price of
#: not having it. CabinetSearch's closing drive refuses on the swept arc
#: (`forearm_roll<->hingerightdoor`) 126 times on main and 432 with the step-3 arm fixes,
#: and those refused turns have a median of 82-93 deg with 49-56 % of them over 90 —
#: the half that stops existing if the base does not turn round at all.
BASE_REVERSE = os.environ.get("MIKASA_BASE_REVERSE", "1") == "1"

#: How much less the base must turn, in DEGREES, before backwards is offered at all.
#: Not a taste setting — a correctness one, and it is what a per-task survey of the five
#: oracles turned up. Several docks are laid out so the two candidates cost exactly the
#: same: the season dish's station and bowl docks carry the same yaw and differ only
#: along the counter (`scenes/season_dish.py:492-497`), so the heading change is 90.000
#: deg and the sum is 180 either way — the choice would then be made by the last bit of a
#: float and by a degree or two of parking error, seed to seed, for zero saving, while
#: mirroring which quadrant the loaded arm sweeps past a distractor checked at 0.10 m.
#: The same shape appears at `same_drawer_planner.py:533` (a pure sideways slide) and
#: `:438` (a base slide with the apple in the gripper). A margin makes those cases keep
#: doing exactly what they did. It costs nothing where reverse is actually worth it:
#: CabinetSearch's closing ladder and its south waypoint each save over 100 deg.
BASE_REVERSE_MARGIN_DEG = float(os.environ.get("MIKASA_BASE_REVERSE_MARGIN", "30"))


def screw_reason(status) -> str:
    """A `plan_screw` status as one countable token.

    `screw plan failed: joint limit at index [7] after 7 step(s), 2.752 of the twist
    left` becomes `limit7` — the distinction that matters when counting why the arm
    fell back to RRT, with the per-call numbers dropped so the counts group.
    """
    s = str(status)
    if s == "Success":
        return "ok"
    m = re.search(r"joint limit at index \[(\d+)\]", s)
    if m:
        return f"limit{m.group(1)}"
    for needle, name in (("collision", "collision"), ("converged", "goal"),
                         ("no convergence", "iters"), ("IK Failed", "ik")):
        if needle in s:
            return name
    return s.split(":")[-1].strip()[:24].replace(" ", "_") or "?"


def screw_left(status) -> float | None:
    """How much of the twist a jammed `plan_screw` still had left, or None.

    From `… after 7 step(s), 2.752 of the twist left`. Near zero means the screw all
    but arrived and a shorter leg would plan; a large value means it jammed early.
    """
    m = re.search(r"([\d.]+) of the twist left", str(status))
    return round(float(m.group(1)), 3) if m else None


def screw_jammed_joints(status) -> list[int]:
    """Which joints the screw walked into their stops; empty if that is not why it failed.

    From `screw plan failed: joint limit at index [7] after 7 step(s), …`, and the
    bracket is a *list* — `[3, 7]` when the same iteration put two joints over at once.
    Indices are into the full simulator qpos: 3 torso_lift, 7 shoulder_lift, 11
    wrist_flex are the three W29 counted (194, 360 and 115 of 967 refusals).

    Empty for every other refusal, which is what keeps the unjam retry off a collision:
    a naive parse there would find no bracket, and a naive default would freeze joint 0.
    """
    m = re.search(r"joint limit at index \[([\d,\s]+)\]", str(status))
    return [int(x) for x in m.group(1).replace(",", " ").split()] if m else []


#: Report `skim=` — how many knots of the trajectory about to be executed the planning
#: world calls env-colliding — on every RRTConnect leg, without changing which plan is
#: chosen (`MIKASA_SKIM_REPORT=1`). Off by default and behaviour-free either way: the
#: measurement runs after the plan is picked. It exists because K67's central number
#: (58 of 80 episodes carrying a path whose interior collides) was taken *before* K76
#: added the keep-out, and nothing has re-measured it since.
SKIM_REPORT = os.environ.get("MIKASA_SKIM_REPORT", "0") == "1"

OPEN = 1

#: `plan_pose`'s `mask` marks joints the IK may NOT use, and `fixed_joint_indices` the ones
#: RRT may not move along the way. Both cover the three virtual base joints here, because a
#: reach is the arm's job: with `mask=None` mplib excludes nothing (`mplib/planner.py:643`),
#: and the instrument measured what that costs — on SameDrawer seed 0 two reaches whose IK
#: GOAL stood the base 0.50 m and 0.70 m away and turned it 69 and 67 degrees, which is the
#: 0.81 m of base travel W27 saw with no base primitive called at all.
#:
#: This is the same pair `static_manipulation` has always passed (`only_manipulate` plus
#: `fixed_joint_indices=[0, 1, 2]`) and the pair the container test plans with
#: (`tests/test_solver_in_container.py:217-218`); the direct callers were the exception, not
#: the rule. `base_free=True` restores the old behaviour for a caller that wants the base to
#: shuffle — nobody passes it today, and the six sites that used to rely on it
#: (`oracle_common.try_grasp`, `same_drawer_planner`'s four reach/push/pull legs) are an
#: opt-in A/B of their own, not a silent inheritance.
RRT_BASE_MASK = [True, True, True, False] + [False] * 11

#: How far from `target_pos` the base may stand and still have its drive counted as
#: arrived, when the drive itself succeeded and only the closing view turn refused.
#:
#: The refusal is the rotate-sweep phantom (K109/K111): `rotate_base_z` sweeps the arc for
#: collisions and hallucinates one against a fixture a metre away, so a drive that has
#: physically ARRIVED is booked `-1` and the whole leg is thrown away. W20a measured the
#: base 0.012-0.017 m from its rung on 4/4 seeds while every rung was booked refused, and
#: the instrument then counted it at sweep scale: CabinetSearch books 46 such refusals
#: across 20 successful episodes, at distances of 2.5 mm, 6.8 mm, 16 mm from target.
#:
#: 0.15 m is not a new number — it is `CLOSE_ARRIVE_TOL` from
#: `cabinet_search_planner.py:130`, measured and shipped for ONE dock in K113.
#:
#: **Opt-in, and that is a measured decision, not caution.** Making it the default lost
#: CabinetSearch 20/30 -> 17/30 (2026-09-03 A/B, band 0) while firing perfectly — 43 of 43
#: refusals accepted, every one of them 2.5-20 mm from target. The mechanism worked and the
#: task got worse, because the premise was wrong: **the sweep refusal is not always a
#: phantom.** On seed 0 it named `wrist_flex_link <-> hingerightdoor` — the arm genuinely
#: colliding with the door the robot had just opened — and `turn_in_place` ignores the sweep
#: by design, so accepting drove the arm through it and every later stage failed from there.
#: Worse, the oracle already had a better answer than the primitive can invent: "the look
#: drive refused; reading the verdict where we stand" — it does not need the turn at all,
#: and that seed succeeds in the arm that lets it do so.
#:
#: So a caller that KNOWS its posture is safe passes `arrive_tol=` (the closing dock, where
#: K113 measured it, is such a caller); the primitive does not decide that for everyone.
DRIVE_ARRIVE_TOL = 0.15

#: Distinguishes "caller said nothing" from "caller asked for no mask at all".
_MASK_UNSET = object()


def _rrt_base_defaults(mask, base_free: bool):
    """(mask, fixed_joint_indices) for a `plan_pose` reach — see `RRT_BASE_MASK`."""
    if mask is _MASK_UNSET:
        return (None, None) if base_free else (RRT_BASE_MASK, [0, 1, 2])
    return mask, (None if base_free else [0, 1, 2])


def _report_rrt_pose(solver, result, mask) -> None:
    """Instrument: how far an RRTConnect plan's own goal moves the BASE (2026-09-03).

    Behaviour-free, and deliberately so: it runs after the plan has been chosen and
    prints without tracing (`stepping.report`'s `to_trace=False`), so an `events.jsonl`
    diffed byte for byte is unchanged. It exists because `mask=None` means mplib excludes
    no joint from IK sampling (`mplib/planner.py:643`), so the three virtual base joints
    are free to carry a reach the arm could not make — measured on SameDrawer seed 0 as
    0.81 m of base travel with no base primitive called at all (W27). Without this line
    that only shows in a 15-episode probe; with it, in every sweep.

    A failure here must never cost an episode, hence the bare except: the instrument is
    not allowed to be the reason a run dies.
    """
    try:
        pos = np.asarray(result["position"])
        q0 = solver.robot.get_qpos().cpu().numpy()[0]
        report(
            solver.env,
            "rrt_pose",
            {
                "mask": "none" if mask is None else "set",
                "base_goal_dxy": round(float(np.linalg.norm(pos[-1][:2] - q0[:2])), 4),
                "base_goal_dyaw": round(float(abs(pos[-1][2] - q0[2])), 4),
                "knots": int(pos.shape[0]),
            },
            to_trace=False,
        )
    except Exception as exc:  # noqa: BLE001 - an instrument may not change an episode
        print(f"[rrt_pose] instrument failed: {type(exc).__name__}: {exc}")
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
# invalidate both. `freeze_arm=True` is passed by `water_plants_planner.py` and by
# nothing else.
#
# Note what it does *not* change: `follow_moving_forward` sends the arm's **current**
# qpos as the arm action on every step whatever the plan says, so the mask decides
# whether a base move plans at all, never what the arm executes.
BASE_ONLY_PLAN_MASK = [True, True, True] + [False] * 12


class MikasaPandaArmSolverV2(PandaArmMotionPlanningSolver):
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
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0,
        mask=_MASK_UNSET, base_free: bool = False,
    ):
        """RRTConnect to `pose`, with the base masked out of the reach by default.

        See `RRT_BASE_MASK`: `base_free=True` is the old behaviour, and an explicit `mask=`
        still wins over both.
        """
        mask, fixed = _rrt_base_defaults(mask, base_free)
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
            rrt_range=RRT_RANGE,
            simplify=True,
            mask=mask,
            fixed_joint_indices=fixed,
        )
        if result["status"] != "Success":
            print(result["status"])
            self.render_wait()
            return -1
        _report_rrt_pose(self, result, mask)
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


class MikasaPandaArmSapienSolver(MikasaPandaArmSolverV2):
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
        mask=_MASK_UNSET,
        n_init_qpos=20,
        base_free: bool = False,
    ):
        """RRTConnect to `pose`, with the base masked out of the reach by default.

        See `RRT_BASE_MASK`: `base_free=True` is the old behaviour, and an explicit `mask=`
        still wins over both.
        """
        mask, fixed = _rrt_base_defaults(mask, base_free)
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
            rrt_range=RRT_RANGE,
            simplify=True,
            mask=mask,
            n_init_qpos=n_init_qpos,
            fixed_joint_indices=fixed,
        )
        if result["status"] != "Success":
            print(result["status"])
            self.render_wait()
            return -1
        _report_rrt_pose(self, result, mask)
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)


class MikasaFetchSolver(MikasaPandaArmSapienSolver):
    RESIDUAL_ACCEPT_RAD = 0.25
    """A refused RESIDUAL yaw path this small (rad) does not fail the turn already made
    (`rotate_base_z`). Measured 2026-09-05: 1392 of ~1600 CabinetSearch logs carry a
    refused residual — the base standing inside an obstacle's margin after a 90 %-executed
    turn — nearly all absorbed by the caller, two episodes lost to the -1 (1485, 1199)."""

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

    #: Index of `root_z_rotation_joint` in ds_fetch's full qpos. The same 2 that
    #: `_plan_base_yaw` substitutes the swept offset into; named here so the two
    #: cannot drift apart.
    YAW_JOINT_INDEX = 2

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
        # (MikasaPandaArmSolverV2.__init__), which the property below routes
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

    # -- the action vector, in the env's control mode -------------------------------
    #: Modes this solver composes actions for. `pd_joint_pos`: the arm and body slots
    #: are absolute targets. `pd_joint_delta_pos`: they are increments from the MEASURED
    #: pose, normalized by the controller's step (0.1 rad / 0.1 m), which is the format a
    #: VLA dataset is recorded in (the supervisor's note, 2026-09-08). The gripper (an
    #: absolute mimic target) and the base (velocities) read the same in both.
    COMPOSE_MODES = ("pd_joint_pos", "pd_joint_delta_pos")

    @staticmethod
    def _delta_step(ctrl) -> np.ndarray:
        """The per-step increment one unit of a normalized delta action commands."""
        cfg = ctrl.config
        lo = np.asarray(cfg.lower, dtype=np.float64).reshape(-1)
        hi = np.asarray(cfg.upper, dtype=np.float64).reshape(-1)
        assert getattr(cfg, "use_delta", False) and getattr(cfg, "normalize_action", True), cfg
        assert np.allclose(lo, -hi), (lo, hi)
        n = len(ctrl.config.joint_names)
        return np.broadcast_to(hi, (n,)) if hi.shape[0] != n else hi

    def _compose(self, arm_target, body_target, base_action):
        """One env action from ABSOLUTE arm and body targets, in the env's control mode.

        Every primitive of this class computes what it wants the joints AT — the next
        knot of a plan, the measured pose to hold, a head target — and hands it here.
        In `pd_joint_pos` that is the action. In `pd_joint_delta_pos` the action is
        `(target - measured) / step`, clipped to [-1, 1]: holding is a zero, a knot
        within one step of the arm is reached exactly, a farther one is approached at
        the controller's rate (the caller's refinement loop keeps re-issuing it). The
        measured pose is read here, at the step, never cached (the delta controller
        anchors to the pose of the step it is applied on). The absolute form is kept
        for the tape (`start_tape`), whatever the mode.

        Args:
            arm_target: (7,) absolute arm joint targets, in the arm controller's order.
            body_target: (3,) absolute [head_pan, head_tilt, torso_lift].
            base_action: (2,) normalized base velocities [forward, yaw].

        Returns:
            the (13,) action for `env.step`.

        Example:
            >>> a = planner._compose(arm_q, np.array([0.0, 0.0, torso]), np.zeros(2))  # doctest: +SKIP
        """
        arm_target = np.asarray(arm_target, dtype=np.float64).reshape(-1)
        body_target = np.asarray(body_target, dtype=np.float64).reshape(-1)
        base_action = np.asarray(base_action, dtype=np.float64).reshape(-1)
        self._last_abs = np.hstack([arm_target, self.gripper_state, body_target, base_action])
        return self._from_abs(self._last_abs)

    def _from_abs(self, vec):
        """The action for the env's control mode from the ABSOLUTE form
        `[arm targets, gripper, body targets, base velocities]` (a tape entry)."""
        vec = np.asarray(vec, dtype=np.float64).reshape(-1).copy()
        # The base slots are normalized velocities in every mode, and the planned base
        # speed can exceed the controller's 1 m/s (mplib's TOPP on the base joints:
        # 1.16 on CabinetSearch seed 2100). ManiSkill clips them on the way in; clipping
        # here makes the RECORDED action the executed one (the dataset check reads
        # `|a| <= 1`), nothing else changes.
        vec[-2:] = np.clip(vec[-2:], -1.0, 1.0)
        mode = self.control_mode
        if mode == "pd_joint_pos":
            return vec
        if mode == "pd_joint_delta_pos":
            arm_ctrl = self.env_agent.controller.controllers["arm"]
            body_ctrl = self.env_agent.controller.controllers["body"]
            n_arm = len(arm_ctrl.config.joint_names)
            n_body = len(body_ctrl.config.joint_names)
            arm_target, grip = vec[:n_arm], vec[n_arm:n_arm + 1]
            body_target, base = vec[n_arm + 1:n_arm + 1 + n_body], vec[n_arm + 1 + n_body:]
            arm_q = arm_ctrl.qpos[0].cpu().numpy().astype(np.float64)
            body_q = body_ctrl.qpos[0].cpu().numpy().astype(np.float64)
            arm = np.clip((arm_target - arm_q) / self._delta_step(arm_ctrl), -1.0, 1.0)
            body = np.clip((body_target - body_q) / self._delta_step(body_ctrl), -1.0, 1.0)
            return np.hstack([arm, grip, body, base])
        raise NotImplementedError(f"{type(self).__name__} composes actions for "
                                  f"{self.COMPOSE_MODES}, not {mode!r}")

    #: `pd_joint_delta_pos` path following: the knot advances once every arm joint is
    #: within this of it (rad); under the controller's 0.1 step, so the delta never
    #: saturates and the arm follows the plan's geometry, not a chord to a knot ahead.
    DELTA_LAG_GATE = float(os.environ.get("MIKASA_DELTA_LAG_GATE", "0.08"))
    #: The most steps a single knot may be re-issued for before the clock moves on
    #: regardless (a knot the arm cannot reach — contact, a limit — must not hang).
    DELTA_LAG_MAX_STALL = int(os.environ.get("MIKASA_DELTA_LAG_MAX_STALL", "10"))

    def _arm_lag(self, arm_target) -> float:
        """max |target - measured| over the arm's joints, rad."""
        q = self.env_agent.controller.controllers["arm"].qpos[0].cpu().numpy().astype(np.float64)
        return float(np.max(np.abs(np.asarray(arm_target, dtype=np.float64) - q)))

    def _hold_targets(self, head_zero: bool = True):
        """The measured arm pose and body pose, as ABSOLUTE targets to hold; the head
        at zero when `head_zero` (the drives' convention since the fork: the head is
        parked while the base moves)."""
        arm = self.env_agent.controller.controllers["arm"].qpos[0].cpu().numpy().astype(np.float64)
        body = self.env_agent.controller.controllers["body"].qpos[0].cpu().numpy().astype(np.float64)
        if head_zero:
            body[0] = body[1] = 0.0
        return arm, body

    # -- the base's dropped lateral velocity (the supervisor's item 3, 2026-09-08) ----
    def _note_lateral(self, base_vel_world, is_forward, base_direction):
        """Keep the norm of what the forward projection threw away this step: mplib
        plans the base holonomically, the differential base executes the projection
        on its heading only. A Fetch cannot move sideways, so the plan must not ask
        it to; this measures whether it does."""
        lateral = np.asarray(base_vel_world, dtype=np.float64) - float(is_forward) * np.asarray(base_direction, dtype=np.float64)
        self._lateral = getattr(self, "_lateral", [])
        self._lateral.append(float(np.linalg.norm(lateral[:2])))

    def _report_lateral(self, stage: str):
        """One line per drive leg: the dropped lateral speed, max and mean (m/s)."""
        lat = getattr(self, "_lateral", None)
        if lat:
            self._report(stage, to_trace=False, lateral_max=round(max(lat), 4),
                         lateral_mean=round(float(np.mean(lat)), 4), steps=len(lat))
        self._lateral = []

    def _step(self, action):
        """The one `env.step` of this class: counted, latched, printed, rendered."""
        entry, self._last_abs = getattr(self, "_last_abs", None), None
        obs, reward, terminated, truncated, info = self._guard.step(action, tape_entry=entry)
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
        """The plan's last knot as `{user_joint_name: q}` over the move group.

        An EMPTY plan is a real return from the solver — mplib can hand back a
        "success" whose `position` array has zero rows, and W25 (2026-09-02) hit
        one folding the arm beside the west wall. Indexing `[-1]` there raised
        `IndexError` from inside the diagnostic line and took the whole episode
        with it (the sweep would book `errored`, not `no plan`). An empty plan
        has no final knot, so it reports as unreached rather than as a crash.
        """
        pos = result.get("position") if isinstance(result, dict) else None
        if pos is None or len(pos) == 0:
            return {}
        return {
            self.planner.user_joint_names[idx]: q
            for idx, q in zip(self.planner.move_group_joint_indices, pos[-1])
        }

    def _report(self, stage: str, *, to_trace: bool = True, **fields) -> None:
        """One diagnostic line per executed plan, on stdout and — when a `PlannerLogger`
        is anywhere in the env chain — as a `solver` event in events.jsonl (with
        `stage=`), so the trace names the plan next to the step. Deliberately not the
        oracle's `say()`: the solver must not import from `utils.mikasa_oracle.planners`.

        `to_trace=False` prints without tracing, for an instrument that runs on every
        turn and must not add a line to traces that are diffed byte for byte; see
        `stepping.report`, which does the work and explains the split."""
        report(self.env, stage, fields, to_trace=to_trace)

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
        yaw_before = self._yaw_joint()
        res = self.follow_rotation(result)

        # What the turn actually did, before any early return: `achieved` is the yaw
        # joint's own delta (not `chosen − residual`, which is measured against the
        # world direction asked for and reads the same whether the base moved or the
        # goal was already met), and `yaw_joint` is its absolute value, which makes
        # the accumulated winding readable straight off the line. `jammed` is the
        # K54/D14 guard; it is traced only when it fires, so the three other tasks'
        # events.jsonl stay byte-identical — see `_report` and `base_yaw.jammed_turn`.
        yaw_joint = self._yaw_joint()
        achieved = yaw_joint - yaw_before
        residual = self._yaw_to(new_direction)
        jammed = jammed_turn(achieved, result["angle"], yaw_joint, self._yaw_joint_limits())
        self._report(
            "rotate_base_z",
            achieved=round(float(achieved), 4),
            yaw_joint=round(float(yaw_joint), 4),
            residual=round(float(residual), 4),
            jammed=jammed,
            to_trace=jammed,
        )
        if self.truncated or not rotate_recalculation_enabled:
            return res

        # The executed turn is a velocity command tracked by a PD controller; take
        # up the residual, if any, with one more yaw-only path.
        if np.abs(residual) < 1e-2:
            return res
        result = self._plan_base_yaw(residual)
        if result["status"] != "Success":
            print(result["status"])
            self.render_wait()
            if np.abs(residual) <= self.RESIDUAL_ACCEPT_RAD:
                # The turn was made; only its last few degrees have no planned path
                # (the sweep sees the base already inside an obstacle's margin at
                # yaw 0 of the residual). Refusing the WHOLE turn here lost seed 1485
                # (2026-09-05: 2.01 of 2.20 rad executed, the 0.19 refused on
                # `base_link<->stack_1 inner_box`, the drive to the handle dock failed).
                # The caller's next leg plans from the heading as it stands and takes
                # the residual with it; a large residual is still the refusal it was.
                self._report("rotate_base_z", residual_left=round(float(residual), 4),
                             accepted="below RESIDUAL_ACCEPT_RAD; the executed turn stands")
                return res
            return -1
        return self.follow_rotation(result)

    def _yaw_joint(self) -> float:
        """The base-yaw joint's current value in radians, read from the simulator.

        Not the base link's world heading: that one is the joint plus whatever the
        root pose was set to at reset, and it wraps at ±π. The joint is what has a
        stop (`fetch.urdf:40`, ±6.28 rad) and what accumulates across a whole
        episode.
        """
        return float(self.robot.get_qpos().cpu().numpy()[0][self.YAW_JOINT_INDEX])

    def _yaw_joint_limits(self) -> tuple:
        """`(lower, upper)` of the base-yaw joint, from the articulation.

        Read rather than written down: the ±6.28 rad in `fetch.urdf` is the source,
        and a literal here would go on agreeing with a URDF that had changed.
        """
        lo, hi = self.robot.get_qlimits()[0][self.YAW_JOINT_INDEX]
        return float(lo), float(hi)

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
        yaw_index = list(planner.move_group_joint_indices).index(self.YAW_JOINT_INDEX)

        def pairs_at(offset: float) -> set:
            q = qpos.copy()
            q[self.YAW_JOINT_INDEX] += offset
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

    @staticmethod
    def turn_cost(heading, direction) -> float:
        """Smallest absolute yaw change, radians, that points `heading` along `direction`.

        Both are world xy vectors; z is ignored. Returns 0 for a degenerate input, which
        is the right answer for "the target is where we stand" and keeps the caller from
        having to special-case it.
        """
        h = np.asarray(heading, dtype=float).reshape(-1)[:2]
        d = np.asarray(direction, dtype=float).reshape(-1)[:2]
        nh, nd = float(np.linalg.norm(h)), float(np.linalg.norm(d))
        if nh < 1e-9 or nd < 1e-9:
            return 0.0
        h, d = h / nh, d / nd
        return abs(float(np.arctan2(h[0] * d[1] - h[1] * d[0], float(np.dot(h, d)))))

    def approach_aims(self, target_pos, target_view_vec, reverse_ok: bool = True):
        """How to reach a dock, cheapest first: backwards then forwards, or forwards alone.

        A differential-drive base cannot go sideways, so `drive_base` is turn, drive,
        turn. Aiming the nose at the dock costs `theta`; aiming the back at it costs
        `pi - theta`, and the closing turn to `target_view_vec` differs by the same pi.
        So the criterion is the SUM of the absolute turns, not the opening one: facing +x
        with the dock at +y and the view wanted at -y, the nose costs exactly 90 deg and
        then another 180 on arrival, against 90 in total for the back.

        Backwards is offered **only when it saves more than `BASE_REVERSE_MARGIN_DEG`**,
        and then it is offered *with* the forward candidate behind it rather than instead
        of it. Two consequences, both deliberate:

        - When forward wins, the list has one entry and `drive_base` does exactly what it
          did before any of this existed. No leg that used to drive nose-first can now be
          reversed by a tie, by float noise, or by a refusal.
        - When backwards wins, a refused turn falls back to the forward aim, so choosing
          the cheaper way round can never lose a leg the old code would have won. That
          fallback is not K113's silencing of the sweep refusal: the second aim is a
          different motion sweeping a different volume, and when both refuse the leg
          returns -1 as before.

        Returns `(direction, is_reverse, total_turn_radians)` entries, cheapest first.
        """
        d = np.asarray(target_pos, dtype=float).reshape(-1)[:3] - self.base_env.agent.base_link.pose.sp.p
        d = np.array([d[0], d[1], 0.0])
        heading = self.base_env.agent.base_link.pose.sp.to_transformation_matrix()[:3, 0]

        def total(aim):
            c = self.turn_cost(heading, aim)
            return c if target_view_vec is None else c + self.turn_cost(aim, target_view_vec)

        forward = (d, False, total(d))
        if not (BASE_REVERSE and reverse_ok):
            return [forward]
        rev_cost = total(-d)
        if forward[2] - rev_cost <= np.radians(BASE_REVERSE_MARGIN_DEG):
            return [forward]
        return [(-d, True, rev_cost), forward]

    def drive_base(self, target_pos=None, target_view_vec=None, freeze_arm: bool = False,
                   arrive_tol: float | None = None, reverse_ok: bool = True):
        """Turn toward `target_pos`, drive to it, then turn to face `target_view_vec`.

        `freeze_arm` is passed straight to `move_base_forward`; see
        `BASE_ONLY_PLAN_MASK`. Default False, so every existing caller plans exactly
        as before.

        `arrive_tol` (default None — see `DRIVE_ARRIVE_TOL` for why it is opt-in): when
        the drive succeeded and only the closing view turn refused, accept the arrival by
        STATE if the base is within this many metres of `target_pos`, and take the aim with
        the unplanned `turn_in_place` rather than throwing the whole leg away. Pass it only
        from a caller that knows the arm's posture is safe to spin — the refusal is
        sometimes a real collision, and an oracle often has a better recovery than a turn.
        Only the closing turn is forgiven: a refused OPENING turn returns -1 as before,
        because then the base has not moved and there is nothing to have arrived at.

        `reverse_ok=False` forbids the backwards approach for this leg (see
        `approach_aims`). Nothing passes it yet and nothing should until a measurement
        asks for it — but the case it exists for is already written down, in
        `cabinet_search_planner`'s look drive: that leg exists *only* so the recorded
        demonstration puts the cube in frame (W22, its own comment), and a drive recorded
        backwards teaches a policy to approach a target it cannot see. The verdicts in
        these sweeps are state, so nothing here is affected; step 6 is where it will be.
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
                # W29/step 4: aim the base the cheaper way round and let the follower's
                # sign do the rest. If the cheaper aim's turn refuses on the swept arc,
                # try the other one — that is a genuinely different motion with a
                # different swept volume, not the refusal being silenced (K113), and if
                # both refuse the leg still returns -1 exactly as it always did.
                cands = self.approach_aims(target_pos, target_view_vec,
                                           reverse_ok=reverse_ok)
                res, chosen = -1, None
                for i, (aim, rev, cost) in enumerate(cands):
                    res = self.rotate_base_z(aim)
                    if res != -1:
                        chosen = (rev, cost, i)
                        break
                    if self.truncated:
                        return res
                if chosen is None:
                    return res
                self._report("drive_base", to_trace=False, reverse=chosen[0],
                             turn_deg=round(float(np.degrees(chosen[1])), 1),
                             second_aim=bool(chosen[2]),
                             cost_deg=",".join(f"{np.degrees(c[2]):.0f}" for c in cands))
                self.planner.update_from_simulation()

                res = self.move_base_forward(target_pos, n_init_qpos=100, freeze_arm=freeze_arm)
                if res == -1:
                    return res
                self.planner.update_from_simulation()

        # view_direction = target_view_pos.p - self.base_env.agent.base_link.pose.sp.p
        if not target_view_vec is None:
            res = self.rotate_base_z(target_view_vec)
            if res == -1 and target_pos is not None:
                here = self.base_env.agent.base_link.pose.sp.p
                d = float(np.linalg.norm(np.asarray(target_pos, dtype=float)[:2] - here[:2]))
                # Instrument (2026-09-03): the drive ARRIVED and only the view turn refused,
                # on the rotate-sweep phantom (K109/K111). Printed, never traced.
                self._report("drive_base", to_trace=False, view_refused=True, d_target=round(d, 4))
                if arrive_tol is not None and d <= float(arrive_tol) and not self.truncated:
                    # Accept by state and take the aim with the spin, because the sweep is
                    # exactly what hallucinates (K113). Traced: it fires only on this path,
                    # so it cannot add a line to a trace that used to be diffed byte for byte.
                    turned = self.turn_in_place(target_view_vec)
                    self.planner.update_from_simulation()
                    self._report("drive_base", accepted_by_state=True, d_target=round(d, 4),
                                 turned=(turned != -1))
                    res = turned if turned != -1 else self.idle_steps(t=1)
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
            rrt_range=RRT_RANGE,
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

    @staticmethod
    def stretch_path(result, factor: int, tail: int | None = None):
        """Resample a TOPP trajectory so the same path is commanded `factor` times slower.

        K79g concluded that scoping a slowdown to one leg "cannot be done", and that is
        true of mplib: `joint_vel_limits` reach the planner at construction and
        `plan_qpos`/`plan_pose` take no per-call override. But the trajectory that comes
        back is an array, and the follower sends one row per control step — so
        interpolating rows halves the joint delta per step without touching the geometry
        the planner chose. The tracking lag a `pd_joint_pos` controller carries is set by
        that per-step delta, which is the quantity K79c measured at 2-7 deg (7-9 cm of
        Cartesian error at a 0.7 m extension).

        Costs `(factor - 1) * knots` episode steps on the leg it is applied to, and
        nothing anywhere else. `factor <= 1` returns the result untouched.
        """
        if factor is None or int(factor) <= 1:
            return result
        k = int(factor)
        out = dict(result)
        for key in ("position", "velocity", "acceleration"):
            a = result.get(key)
            if a is None:
                continue
            a = np.asarray(a)
            if a.ndim != 2 or a.shape[0] < 2:
                continue
            # `tail` slows only the last stretch of the path. The contact that topples the
            # object happens where the hand arrives, not on the long transit to it, so
            # paying for slowness over the whole leg buys nothing over most of it and the
            # steps come out of the grasp ladder's budget (K85: out-of-budget 6 -> 12).
            cut = 0 if tail is None else max(0, a.shape[0] - int(tail) - 1)
            head, seg = a[:cut], a[cut:]
            if seg.shape[0] < 2:
                continue
            src_i = np.arange(seg.shape[0])
            dst_i = np.linspace(0.0, seg.shape[0] - 1, (seg.shape[0] - 1) * k + 1)
            grid = np.stack([np.interp(dst_i, src_i, seg[:, j]) for j in range(seg.shape[1])], axis=1)
            if key != "position":
                grid = grid / k
            out[key] = np.vstack([head, grid]) if head.shape[0] else grid
        return out

    def gripper_touching(self, actor, threshold: float = 1e-6) -> bool:
        """Is a gripper link in contact with `actor` right now, per SAPIEN (K92)?

        Not the planner's opinion — the simulator's. K90 measured that what topples a
        standing object is a **fingertip**, and that first contact precedes the object's
        first movement by 2-15 steps. That gap is the only warning available: the planned
        path is clean (target intrusion 0 on every seed that topples, measured), so
        nothing before execution can predict it, but the contact itself is observable
        while there is still time to stop.
        """
        name = getattr(actor, "name", None)
        if name is None:
            return False
        try:
            for c in self._sim_scene.get_contacts():
                ns = [b.entity.name for b in c.bodies]
                if not any(name in n for n in ns):
                    continue
                other = ns[0] if name in ns[1] else ns[1]
                if "gripper" not in other and "finger" not in other:
                    continue
                if float(sum(np.linalg.norm(pt.impulse) for pt in c.points)) > threshold:
                    return True
        except Exception:
            return False
        return False

    def path_env_collisions(self, position, names: bool = False):
        """Knots of a planned trajectory the planning world calls env-colliding (K80).

        OMPL validates an edge at its endpoints alone whenever the edge is shorter
        than `longestValidSegment`, which on this robot is **1.32 rad** of summed
        joint travel — larger than every edge RRTConnect draws at `rrt_range=0.1`,
        so the interior of a returned path is never looked at (K67 measured 58 of 80
        episodes carrying a path whose interior collides). This looks at it, after
        the fact, on the TOPP trajectory that will actually be commanded.

        `plan_pose` hands its knots back in the **simulator's** root frame (the
        unfold happens before TOPP, K53), so each is folded again for the check.
        Returns -1 if the planning world cannot answer, which callers read as
        "no opinion" rather than as "clean".
        """
        fold = getattr(self.planner, "fold_qpos", None)
        n = 0
        who: dict[str, int] = {}
        at: list[int] = []
        try:
            traj = np.asarray(position)
            for i, q in enumerate(traj):
                s = fold(q) if callable(fold) else q
                hits = self.planner.check_for_env_collision(np.asarray(s))
                if hits:
                    n += 1
                    at.append(i)
                    for h in hits:
                        key = f"{h.link_name1}<->{h.link_name2}"
                        who[key] = who.get(key, 0) + 1
        except Exception:
            return (-1, {}, "") if names else -1
        span = f"{at[0]}-{at[-1]}/{len(traj)}" if at else ""
        return (n, who, span) if names else n

    @staticmethod
    def _slerp(q0, q1, t: float) -> np.ndarray:
        """Shortest-arc interpolation between two `(w, x, y, z)` quaternions.

        A line-for-line copy of `utils.mikasa_oracle.planners.oracle_common.slerp`, which is
        what `move_via` places its legs with. It is copied rather than imported because
        the solver must not import from `utils.mikasa_oracle.planners` (see `_report`), and it
        must be the same function rather than merely a correct one: the probe measures
        the ceiling for a fix built out of those legs, so a midpoint that differs from
        `move_via`'s would measure a different thing. Pinned by
        `tests/test_solver_stepping.py::test_the_probes_slerp_is_the_one_move_via_interpolates_its_legs_with`.
        """
        a = np.asarray(q0, dtype=np.float64)
        b = np.asarray(q1, dtype=np.float64)
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        d = float(np.dot(a, b))
        if d < 0.0:                      # take the short way round
            b, d = -b, -d
        if d > 0.9995:                   # nearly identical: lerp and renormalise
            out = a + t * (b - a)
            return out / np.linalg.norm(out)
        th = np.arccos(np.clip(d, -1.0, 1.0))
        s = np.sin(th)
        return (np.sin((1 - t) * th) / s) * a + (np.sin(t * th) / s) * b

    def _probe_screw_split(self, target_tcp_pose, only_manipulate, screw_status,
                           fractions=(0.5, 0.25)) -> None:
        """Would this refused screw plan as two short screws through a waypoint?

        Measurement only (`SCREW_SPLIT_PROBE`), and worth measuring before writing the
        fix because of what the baseline says the refusals *are*: 74 % of the RRT
        fallbacks on 2026-09-03 read `screw plan failed: joint limit at index [7|3|11]`
        — the resolved-rate integration walks a joint into its stop partway through one
        long twist, which is not the same thing as a target the arm cannot reach. A
        shorter twist re-linearises from a different configuration; that is K55's
        argument for `move_via`'s legs, and because `plan_screw` takes its start qpos as
        an argument, both legs can be planned before anything is executed.

        The two fractions are the difference between a hypothesis and a number. If a
        jam is *proportional* — the twist runs a joint out of range only because it is
        long — a shorter first leg plans, and shorter still plans more often. If the
        jam is at the start, because the arm is already against that stop and the twist
        pushes into it, no fraction plans and legs cannot be the lever whatever K55
        measured on other stages. `a_at` is the largest fraction that planned.

        Costs wall clock and no episode steps, and restores the planning model's qpos
        so a probed run plans exactly as an unprobed one does — without that the
        counts it prints could not be compared with the baseline's.
        """
        try:
            keep = self.planner.robot.get_qpos()
            start = self.base_env.agent.tcp.pose.sp
            q_now = self.robot.get_qpos().cpu().numpy()[0]
            p0 = np.asarray(start.p, dtype=np.float64)
            p1 = np.asarray(target_tcp_pose.p, dtype=np.float64)
            kw = dict(
                time_step=self.base_env.control_timestep,
                masked_joints=~np.array(only_manipulate),
                goal_tolerance=self.ARM_SCREW_GOAL_TOLERANCE,
            )
            fields = {
                "why": screw_reason(screw_status),
                "left": screw_left(screw_status),
                "span_m": round(float(np.linalg.norm(p1 - p0)), 3),
            }
            # The other recovery the refusal itself names. A jam is one joint running out
            # of range; the arm has seven and the twist needs six, so the same straight
            # line may still be realisable with the offending joint held still. One extra
            # deterministic plan, and it addresses 69 % of the refusals (W29) rather than
            # the 74 % a shorter leg could in principle address.
            jammed = [j for j in screw_jammed_joints(screw_status)
                      if 0 <= j < len(only_manipulate)]
            if jammed:
                frozen = list(only_manipulate)
                for j in jammed:
                    frozen[j] = True           # True here means "hold this joint still"
                f = self.planner.plan_screw(
                    mplib.Pose(p=target_tcp_pose.p, q=target_tcp_pose.q), q_now,
                    **{**kw, "masked_joints": ~np.array(frozen)})
                fields["jam"] = ",".join(str(j) for j in jammed)
                fields["frozen"] = screw_reason(f["status"])
                if f["status"] == "Success":
                    fields["frozen_knots"] = int(f["position"].shape[0])

            a = None
            for t in fractions:
                way = mplib.Pose(p=(1.0 - t) * p0 + t * p1,
                                 q=self._slerp(start.q, target_tcp_pose.q, t))
                r = self.planner.plan_screw(way, q_now, **kw)
                fields[f"a{t}"] = screw_reason(r["status"])
                if r["status"] == "Success":
                    a, fields["a_at"] = r, t
                    break
            if a is not None:
                # `plan_screw` hands its trajectory back in the simulator's root frame
                # (K53, and its own docstring), so the last knot is a start qpos for the
                # second leg — written into a copy of the full simulator qpos by move
                # group index, because the trajectory carries only the move group.
                q_mid = q_now.copy()
                q_mid[list(self.planner.move_group_joint_indices)] = a["position"][-1]
                b = self.planner.plan_screw(
                    mplib.Pose(p=target_tcp_pose.p, q=target_tcp_pose.q), q_mid, **kw)
                fields["b"] = screw_reason(b["status"])
                fields["both"] = bool(b["status"] == "Success")
                if fields["both"]:
                    fields["knots"] = int(a["position"].shape[0]) + int(b["position"].shape[0])
            else:
                fields["both"] = False
            self.planner.robot.set_qpos(keep, True)
            report(self.env, "screw_split", fields, to_trace=False)
        except Exception as exc:  # a probe must never cost an episode
            print(f"[screw_split] probe failed: {type(exc).__name__}: {exc}")

    #: IK solutions tried, nearest first, for the joint-line approach.
    LINE_IK_GOALS = 3

    def _approach_by_line(self, target_tcp_pose, only_manipulate, n_init_qpos,
                          stretch: int = 1, stretch_tail=None, stop_on_touch=None):
        """Reach `target_tcp_pose` by a straight joint line to its nearest IK solution.

        The IK is mplib's, masked like the screw (base fixed, torso per the caller);
        its solutions are unwrapped toward the current posture and ordered nearest
        first (`unwrap_toward`, `goal_order`, as `plan_pose` does). Each is turned into
        a FULL simulator qpos by keeping the root columns of the current pose and taking
        every other joint from the solution, and handed to `plan_qpos_line`, which
        collision-checks the line against the planning world (keepout proxies included)
        and refuses at the first colliding knot. The first line that plans is executed
        with the same follower and report as any other plan (`plan=line`).

        Returns the follower's 5-tuple, or None when no line planned — the caller's
        screw and RRT then run unchanged. A planner without `IK` (a double) gets None.
        """
        from utils.mikasa_oracle.motionplanning.fetch.utils import goal_order, unwrap_toward
        p = self.planner
        if not callable(getattr(p, "IK", None)) or not callable(getattr(p, "plan_qpos_line", None)):
            return None
        cur = self.robot.get_qpos().cpu().numpy()[0].astype(np.float64)
        ref_yaw = float(cur[2]) if len(cur) > 2 else 0.0
        try:
            cur_f = p.fold_qpos(cur)
            goal_b = p._transform_goal_to_wrt_base(mplib.Pose(p=target_tcp_pose.p, q=target_tcp_pose.q))
            status, goal_qpos = p.IK(goal_b, cur_f, only_manipulate, n_init_qpos=n_init_qpos)
        except Exception as e:  # pragma: no cover - the solver's own failure modes
            self._report("static_manipulation", plan="line", status=f"IK raised {type(e).__name__}: {e}"[:140])
            return None
        if status != "Success" or goal_qpos is None or len(np.atleast_2d(goal_qpos)) == 0:
            self._report("static_manipulation", plan="line", status=f"IK: {status}"[:140])
            return None
        return self._line_to_ik_goals(target_tcp_pose, goal_qpos, cur, cur_f, ref_yaw,
                                      stretch, stretch_tail, stop_on_touch, tag="direct")

    def _line_to_ik_goals(self, target_tcp_pose, goal_qpos, cur, cur_f, ref_yaw,
                          stretch, stretch_tail, stop_on_touch, tag: str, precheck=None):
        """The nearest-first joint lines to `goal_qpos`; the follower's tuple or None.

        `precheck(goal_full)` — when given — must plan the leg that FOLLOWS the line from
        its end configuration; a line whose follow-up does not plan is not executed
        (HP7, 1652 @0.15, 2026-09-06: a 92-knot line above the standoff ran on every
        attempt while the descent from there was refused at the stove, and the episode
        ran out of horizon — a refusal that used to cost no steps became a paid leg).
        """
        from utils.mikasa_oracle.motionplanning.fetch.utils import goal_order, unwrap_toward
        p = self.planner
        goals = [unwrap_toward(np.asarray(g, dtype=float), cur_f, p.joint_limits)
                 for g in np.atleast_2d(goal_qpos)]
        order = goal_order(goals, cur_f)
        root = set(p._root_cols() or [])
        tried = []
        for k, g in enumerate(order[:self.LINE_IK_GOALS]):
            goal_full = cur.copy()
            for j in range(min(len(goal_full), len(g))):
                if j not in root:
                    goal_full[j] = float(g[j])
            try:
                line = p.plan_qpos_line(goal_full, cur, time_step=self.base_env.control_timestep, ref_yaw=ref_yaw)
            except RuntimeError as e:
                line = {"status": f"mplib raised: {e}"}
            tried.append(str(line.get("status"))[:70])
            if line.get("status") != "Success" or len(np.asarray(line.get("position", []))) < 2:
                continue
            if precheck is not None and not precheck(goal_full):
                tried.append("line plans but its descent does not")
                continue
            self.render_wait()
            result = self.stretch_path(line, stretch, tail=stretch_tail)
            knots = int(result["position"].shape[0])
            before = self.elapsed_steps
            out = self.follow_forward_path_w_refinement(
                result, refine=True,
                stop_when=(None if stop_on_touch is None
                           else lambda: self.gripper_touching(stop_on_touch)))
            executed = min(knots, self.elapsed_steps - before)
            tcp = self.base_env.agent.tcp.pose.sp
            tcp_pos, tcp_rot = pose_error(target_tcp_pose.p, target_tcp_pose.q, tcp.p, tcp.q)
            duration = result.get("duration")
            self.last_tcp_err = float(tcp_pos)
            self.last_tcp_rot_err = float(tcp_rot)
            self._report(
                "static_manipulation",
                plan="line",
                iters=result.get("iterations"),
                knots=knots,
                dur=None if duration is None else round(float(duration), 2),
                exec=executed,
                refine=self.elapsed_steps - before - executed,
                reached=bool(self.check_body_base_close_to_target(self._final_qpos_dict(result))),
                tcp_err=f"{tcp_pos:.3f}m/{np.degrees(tcp_rot):.1f}deg",
                ik_goal=k, ik_goals=len(order), line=tag,
                **({"stretch": int(stretch)} if int(stretch) > 1 else {}),
            )
            return out
        self._report("static_manipulation", plan="line", line=tag,
                     status="no line: " + " | ".join(tried), ik_goals=len(order))
        return None

    def _approach_via_above(self, target_tcp_pose, only_manipulate, n_init_qpos):
        """A joint line to the pose LINE_VIA_UP above `target_tcp_pose`, or None.

        The direct lines from the rest keyframe to a standoff at counter height are
        blocked by the hand itself sweeping through the object's proxy or the counter
        (g74, 2026-09-06). A point above the standoff is reachable by a line that lifts
        and extends the arm over the worktop; from there the caller's screw descends
        straight to the standoff with the shoulder off its stop. Executes the line and
        returns the follower's tuple; None when no line plans.
        """
        if LINE_VIA_UP <= 0.0:
            return None
        p = self.planner
        above = sapien.Pose(p=np.asarray(target_tcp_pose.p) + np.array([0.0, 0.0, LINE_VIA_UP]),
                            q=target_tcp_pose.q)
        cur = self.robot.get_qpos().cpu().numpy()[0].astype(np.float64)
        ref_yaw = float(cur[2]) if len(cur) > 2 else 0.0
        try:
            cur_f = p.fold_qpos(cur)
            goal_b = p._transform_goal_to_wrt_base(mplib.Pose(p=above.p, q=above.q))
            status, goal_qpos = p.IK(goal_b, cur_f, only_manipulate, n_init_qpos=n_init_qpos)
        except Exception as e:  # pragma: no cover
            self._report("static_manipulation", plan="line", line="above", status=f"IK raised {type(e).__name__}"[:140])
            return None
        if status != "Success" or goal_qpos is None or len(np.atleast_2d(goal_qpos)) == 0:
            self._report("static_manipulation", plan="line", line="above", status=f"IK: {status}"[:140])
            return None
        def descends(goal_full):
            # the screw from the line's end to the pose, planned only — no steps
            try:
                res = p.plan_screw(
                    mplib.Pose(p=target_tcp_pose.p, q=target_tcp_pose.q), np.asarray(goal_full, dtype=np.float64),
                    time_step=self.base_env.control_timestep,
                    masked_joints=~np.array(only_manipulate),
                    goal_tolerance=self.ARM_SCREW_GOAL_TOLERANCE)
            except Exception:  # pragma: no cover
                return False
            return res.get("status") == "Success"
        return self._line_to_ik_goals(above, goal_qpos, cur, cur_f, ref_yaw, 1, None, None,
                                      tag="above", precheck=descends)

    def static_manipulation(
        self, target_tcp_pose, n_init_qpos=20, disable_lift_joint: bool = False,
        max_knots: int | None = None, knot_draws: int = 1, knot_refuse: bool = False,
        draws: int = 1, stretch: int = 1, stretch_tail: int | None = None,
        skim_cap: float | None = None,
        skim_max_knots: int | None = None,
        clearance_scorer=None,
        stop_on_touch=None,
        by_line: bool = False,
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

        line_probe = getattr(self, "_approach_by_line", None)   # a double may lack it
        if by_line and callable(line_probe):
            # SeasonDish, 2026-09-06: from the rest keyframe the straight screw to a side
            # grasp at counter height is refused 200/200 times at the shoulder_lift stop
            # (11 deg away at rest), and the RRT that replaces it wanders 80-110 knots —
            # the "arm twisting" every recording opens with. A straight JOINT line to
            # the pose's nearest IK solution is collision-checked knot by knot (K61) and
            # cannot wind; when it plans, it is the approach. When it does not, the
            # screw and the RRT below proceed exactly as before.
            out = line_probe(target_tcp_pose, only_manipulate, n_init_qpos,
                             stretch, stretch_tail, stop_on_touch)
            if out is not None:
                return out
            via = getattr(self, "_approach_via_above", None)
            if callable(via):
                up = via(target_tcp_pose, only_manipulate, n_init_qpos)
                if up is not None and self.truncated:
                    return up
                # the arm now stands above the standoff (or where it was): the screw
                # below descends to the pose, the RRT only if the screw refuses
        plan = "screw"
        result = self.planner.plan_screw(
            mplib.Pose(p=target_tcp_pose.p, q=target_tcp_pose.q),
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
            masked_joints=~np.array(only_manipulate),
            goal_tolerance=self.ARM_SCREW_GOAL_TOLERANCE,
        )

        screw_status = result["status"]
        # W29 / fix 3b. A `joint limit at index [n]` refusal is a fact about joint n
        # running out of range partway through the twist, not about the goal: the same
        # straight line may still be realisable with that joint held still, and 36 % of
        # the baseline's 404 jams are. Deliberately *not* the K59 mistake of executing
        # the knots the jammed screw got through and re-deriving from there — that left
        # the arm standing on the limit that stopped it and made the fallback longer.
        # This executes nothing until a whole plan to the whole goal exists.
        unjam: list[int] = []
        if ARM_SCREW_UNJAM and screw_status != "Success":
            jammed = [j for j in screw_jammed_joints(screw_status)
                      if 0 <= j < len(only_manipulate)]
            if jammed:
                held = list(only_manipulate)
                for j in jammed:
                    held[j] = True             # True here means "hold this joint still"
                retry = self.planner.plan_screw(
                    mplib.Pose(p=target_tcp_pose.p, q=target_tcp_pose.q),
                    self.robot.get_qpos().cpu().numpy()[0],
                    time_step=self.base_env.control_timestep,
                    masked_joints=~np.array(held),
                    goal_tolerance=self.ARM_SCREW_GOAL_TOLERANCE,
                )
                if retry["status"] == "Success":
                    result, unjam = retry, jammed

        redraws = 0
        skim = None
        skim_who: dict[str, int] = {}
        skim_at = ""
        skim_draws = ""
        if result["status"] != "Success":
            plan = "rrt"
            if SCREW_SPLIT_PROBE:
                self._probe_screw_split(target_tcp_pose, only_manipulate, screw_status)

            def _draw():
                return self.planner.plan_pose(
                    target_tcp_pose,
                    self.robot.get_qpos().cpu().numpy()[0],
                    time_step=self.base_env.control_timestep,
                    # use_point_cloud=self.use_point_cloud,
                    wrt_world=True,
                    verbose=self.verbose,
                    planning_time=2 * PLANNING_TIME,
                    rrt_range=RRT_RANGE,
                    simplify=True,
                    mask=only_manipulate,
                    fixed_joint_indices=fixed_joint_indices,
                    n_init_qpos=n_init_qpos,
                )

            # Two redraw knives arrived on parallel branches and disagree on both
            # the selection criterion (shortest-by-knots vs least-skim) and the
            # failed-draw policy (continue vs break), so they stay two explicit
            # loops; the knot knife wins when both are asked for (no caller
            # passes both today).
            #
            # W29: a caller that asked for neither used to execute the first path
            # RRTConnect happened to return, and that is four of the five benchmark
            # tasks. It now gets the knot knife's default draws. A caller that asked
            # for the skim knife (`draws`) keeps it — that one is about where the path
            # goes, not how long it is, and the two must not be silently swapped.
            no_knife = int(draws) <= 1 and int(knot_draws) <= 1 and max_knots is None
            if no_knife:
                # BOTH knives, in the order that matters: of the draws that do not
                # intrude on the environment, execute the shortest.
                #
                # The knot knife alone was wrong here, and it was wrong in a way that
                # bites exactly where it hurts. The shortest path is the most DIRECT
                # one, and the most direct one hugs the obstacles it has to avoid: the
                # planner's "collision-free" is a claim about edge endpoints, so a path
                # that is clean by that test can still sweep an object on the way in.
                # Measured on SeasonDish seed 6, reproduced in three pools: `main` takes
                # a 146-knot path and grasps the bottle; with the knot knife the arm
                # takes a 112-knot path, the fingers close on nothing, and the bottle is
                # 18.8 cm away — the approach itself swept it, before the pads moved.
                #
                # Skim first, length second. A draw executes nothing, so both scores
                # cost wall clock and never simulation steps.
                best = None
                for _ in range(max(1, int(RRT_ARM_DRAWS))):
                    result = _draw()
                    if result["status"] != "Success":
                        continue
                    score = int(self.path_env_collisions(result["position"]))
                    n = int(result["position"].shape[0])
                    key = draw_rank(score, n)
                    if clearance_scorer is not None:
                        # The clearance knife (SeasonDish 1679/1957, 2026-09-06): every
                        # draw that toppled the 16 g shaker was CLEAN by the intrusion
                        # metric — the fingertip passed inside the tracking error of
                        # the object. Among clean draws prefer the one whose knots stay
                        # out of a WIDER proxy of the target (the caller's scorer), and
                        # only then the shortest. Refusals are unchanged.
                        try:
                            near = max(0, int(clearance_scorer(result["position"])))
                        except Exception:
                            near = 0
                        key = (key[0], key[1], near, key[2])
                    if best is None or key < best[0]:
                        best = (key, result)
                    redraws += 1
                if best is not None:
                    _clean, skim, _n = best[0][0], best[0][1], best[0][-1]
                    result = best[1]
                    skim_draws = f"skim{skim}" + (f" near{best[0][2]}" if len(best[0]) == 4 else "")
                    if skim_cap is not None and not path_is_a_plan(skim, _n, skim_cap):
                        # Better no plan than this one: the caller's own ladder gets a
                        # scene it can still work in. See `path_is_a_plan`.
                        result = dict(result)
                        result["status"] = f"path intrudes at {skim}/{_n} knots"
                    elif skim_max_knots is not None and _n > int(skim_max_knots):
                        # The cleanest draw is still too LONG to execute beside a loose
                        # object: "clean" is a claim about edge endpoints, and a long
                        # path sweeps between them (SeasonDish 1581, 2026-09-06: the
                        # approach's best draw, 143 knots and clean by this metric,
                        # swung through the bottle and moved it 20 cm; reached=False
                        # at 0.094 m). Refused, like an intruding draw: the caller's
                        # ladder re-draws (RRTConnect is randomized) or re-aims.
                        result = dict(result)
                        result["status"] = (f"path too long: {_n} knots over the cap of "
                                            f"{int(skim_max_knots)} (skim-ranked)")
            elif max_knots is not None or int(knot_draws) > 1:
                # `plan_pose` is RRTConnect and RRTConnect is *randomized*: two draws for
                # the same reachable goal come back with wildly different path lengths. A
                # draw costs no simulation steps, so when the caller says how long a path
                # it is willing to execute, take more draws and keep the shortest.
                #
                # K58, water-plants: measured over 85 pour tips executed with the cup in
                # hand, every one of the 5 that threw the cup out of the gripper was a plan
                # of >= 173 knots, and none of the 73 tips under 173 knots ever dropped it
                # (12 long tips held, so length is necessary and not sufficient — the drop
                # rate above the line is 5/12). A long path swings the cup through extreme
                # intermediate orientations on its way to a modest final tilt, and gravity
                # levers it out of a 2.75 cm top grasp.
                best = None
                for _ in range(max(1, int(knot_draws))):
                    result = _draw()
                    if result["status"] != "Success":
                        continue
                    n = int(result["position"].shape[0])
                    if best is None or n < int(best["position"].shape[0]):
                        best = result
                    if max_knots is not None and n <= int(max_knots):
                        break        # under the cap the caller named; the rest is waste
                    redraws += 1
                if best is not None:
                    result = best
                    n_best = int(result["position"].shape[0])
                    if knot_refuse and max_knots is not None and n_best > int(max_knots):
                        # The caller would rather have no plan than this one: a short
                        # leg (a 10 cm descent onto a grasp) answered with a wandering
                        # path sweeps what it was aimed at (SeasonDish 1653: 209 knots,
                        # tcp_err 0.196 m, the bottle 33 cm away). Its ladder re-aims.
                        result = dict(result)
                        result["status"] = f"path too long: {n_best} knots over the cap of {int(max_knots)}"
                if SKIM_REPORT and result["status"] == "Success":
                    skim, skim_who, skim_at = self.path_env_collisions(result["position"], names=True)
            else:
                # `draws=1` takes the first plan RRTConnect returns, which is what every
                # caller did before K80 and is byte-identical to it. Above 1, the same
                # goal is drawn up to `draws` times and the draw whose *commanded*
                # trajectory intrudes on the environment at fewest knots is the one
                # executed — the planner's own "collision-free" is a claim about edge
                # endpoints only (see `path_env_collisions`). A clean draw short-circuits,
                # so the extra wall clock is paid only where the first draw skims. No
                # episode steps are spent either way: refused and rejected plans are free.
                best = None
                tried: list[int] = []
                for _ in range(max(1, int(draws))):
                    result = _draw()
                    if int(draws) <= 1 or result["status"] != "Success":
                        if SKIM_REPORT and result["status"] == "Success":
                            skim, skim_who, skim_at = self.path_env_collisions(result["position"], names=True)
                        break
                    score = self.path_env_collisions(result["position"])
                    tried.append(int(score))
                    if best is None or (0 <= score < best[0]):
                        best = (score, result)
                    if best[0] <= 0:
                        break
                if best is not None:
                    skim, result = best
                    skim_draws = ",".join(str(t) for t in tried)

            if result["status"] != "Success":
                # Nothing executed; say why both planners refused, in the same
                # place the executed plans report, so the trace names the reason
                # next to the oracle's "FAILED: <stage>".
                self._report("static_manipulation", plan=None, screw=screw_status, rrt=result["status"])
                self.render_wait()
                return -1

        self.render_wait()

        result = self.stretch_path(result, stretch, tail=stretch_tail)
        knots = int(result["position"].shape[0])
        before = self.elapsed_steps
        out = self.follow_forward_path_w_refinement(
            result, refine=True,
            stop_when=(None if stop_on_touch is None
                       else lambda: self.gripper_touching(stop_on_touch)))
        # Path following stops early only on truncation, so what was executed is the
        # smaller of the path and the steps taken; the rest was refinement.
        executed = min(knots, self.elapsed_steps - before)
        tcp = self.base_env.agent.tcp.pose.sp
        tcp_pos, tcp_rot = pose_error(target_tcp_pose.p, target_tcp_pose.q, tcp.p, tcp.q)
        goal_error = result.get("goal_error")
        duration = result.get("duration")
        # The last executed move's tracking error, for callers that must not act on a
        # pose the arm did not reach: try_grasp closed its fingers after a stroke that
        # reported tcp_err 0.196 m (SeasonDish 1653, 2026-09-06) and knocked the bottle
        # 33 cm. A stub planner has no such attribute; callers probe for it.
        self.last_tcp_err = float(tcp_pos)
        self.last_tcp_rot_err = float(tcp_rot)
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
            # The token, not the sentence: `screw=` may end a line with spaces in it,
            # but this sits mid-line and every parser splits the report on whitespace.
            **({"unjam": ",".join(str(j) for j in unjam),
                "jammed": screw_reason(screw_status)} if unjam else {}),
            **({"redraws": redraws} if redraws else {}),
            **({"skim": skim} if skim is not None else {}),
            **({"skim_who": ",".join(
                f"{k}:{v}" for k, v in sorted(skim_who.items(), key=lambda kv: -kv[1])[:4])}
               if skim_who else {}),
            **({"skim_at": skim_at} if skim_at else {}),
            **({"skim_draws": skim_draws} if skim_draws else {}),
            **({"stretch": int(stretch)} if int(stretch) > 1 else {}),
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
            arm_action, body_action = self._hold_targets()
            base_action = np.array([0.0, 0.0])

            qvel = result["velocity"][min(i, n_step - 1)]

            base_action[1] = qvel[2]

            action = self._compose(arm_action, body_action, base_action)
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
            arm_action, body_action = self._hold_targets()
            base_action = np.array([0.0, 0.0])

            qvel = result["velocity"][min(i, n_step - 1)]
            base_vel = np.array([qvel[0], qvel[1], 0.0])
            base_vel_wrt_world = root_to_world @ base_vel
            is_forward = np.dot(base_vel_wrt_world, base_direction)
            base_action[0] = is_forward
            self._note_lateral(base_vel_wrt_world, is_forward, base_direction)

            action = self._compose(arm_action, body_action, base_action)
            if self.verbose:
                print("base Action:", np.round(base_action, 4))
                print("Full: ", np.round(self.robot.get_qpos().cpu().numpy()[0], 4))
            obs, reward, terminated, truncated, info = self._step(action)
            if self._stopped_by_horizon("follow_moving_forward"):
                break

        self._report_lateral("follow_moving_forward")
        return obs, reward, terminated, truncated, info

    def follow_arc(self, anchor_xy, sense, v_handle: float = 0.05,
                   max_steps: int = 400, radial_bias: float = 0.0, stop_when=None):
        """Drive the base so the TCP rides the tangent of a circle about a vertical
        hinge (K103, W14) — the arc pull that opens a door the straight pull cannot.

        Unplanned by design: the pull is a velocity-followed contact stroke, like the
        straight pull W12 measured, not an mplib trajectory — the door it moves could
        never be planned against anyway. The arm and torso are commanded at their
        *measured* qpos each step (the `follow_moving_forward` pattern: the target
        follows the arm, it does not spring back — unlike the W14 probe's fixed
        hold, and the `--use-primitive` re-measurement is what says whether that
        difference costs anything), and `self.gripper_state` is re-emitted every
        step, so a live grasp stays the solver's normal execution path.

        Per step `arc_pull_cmd` gives physical (v, omega); the omega action channel is
        normalized ([-1, 1] -> +-3.14 rad/s per the ds_fetch base controller config),
        so the slot value is omega divided by that scale — read from the controller
        rather than hardcoded, because a silently wrong scale here is a 3.14x spin.

        Stops on: `max_steps`; the episode's horizon; `stop_when()` returning truthy
        (checked after each step — the caller watches the door joint, the fingers,
        whatever its contract needs); or the forward lever collapsing (the held arm
        pose is gone — reported, never divided by).

        Args:
            anchor_xy: hinge axis, world xy.
            sense: +1 when +qpos is CCW about world +z.
            v_handle: handle speed along its tangent, m/s (0.05 = the W12 pull speed).
            max_steps: hard cap on steps.
            radial_bias: optional outward preload, m/s.
            stop_when: nullary callable; truthy stops the pull after that step.

        Returns:
            The last gym 5-tuple (or the guard's last step when already truncated);
            -1 only when no step was ever taken (D6: a refusal before any commit).
        """
        if self.truncated:
            return self._guard.last_step
        cfg = self.env_agent.controller.controllers["base"].config
        # Half-reading the config is the silent 3.14x spin this docstring warns
        # about: the omega scale below is only right for a normalized channel, and
        # the v slot is passed through raw only because upper[0] is 1 m/s.
        assert getattr(cfg, "normalize_action", False) and float(cfg.upper[0]) == 1.0, (
            f"follow_arc assumes the ds_fetch base channel shape; got "
            f"normalize_action={getattr(cfg, 'normalize_action', None)} upper={cfg.upper}")
        omega_scale = float(cfg.upper[1])
        out = self._guard.last_step
        reason = "max-steps"
        v = omega = 0.0
        lever_tail, speed_tail = [], []   # a_x and |tcp velocity| per step, said at the end
        last_p = None
        dt = float(self.base_env.control_timestep)
        i = -1
        for i in range(int(max_steps)):
            arm_action, body_action = self._hold_targets()
            q = self.robot.get_qpos().cpu().numpy()[0]
            tcp = self.base_env.agent.tcp.pose.sp
            v, omega, a_x = arc_pull_cmd(
                tcp.p[:2], q[:2], float(q[2]), anchor_xy, sense,
                v_handle, radial_bias,
            )
            if last_p is not None and len(speed_tail) < 60:
                speed_tail.append(round(float(np.linalg.norm(np.asarray(tcp.p[:2]) - last_p)) / dt, 4))
                lever_tail.append(round(float(a_x), 3))
            last_p = np.asarray(tcp.p[:2], dtype=np.float64).copy()
            if a_x <= 0.3:
                reason = "lever-collapsed"
                print(f"[follow_arc] a_x={a_x:.3f} — the TCP lever collapsed; "
                      "the held arm pose is gone", flush=True)
                break
            base_action = np.array([
                np.clip(v, -1.0, 1.0),
                np.clip(omega, -omega_scale, omega_scale) / omega_scale,
            ])
            action = self._compose(arm_action, body_action, base_action)
            out = self._step(action)
            if self._stopped_by_horizon("follow_arc"):
                reason = "horizon"
                break
            if stop_when is not None and stop_when():
                reason = "stop_when"
                break
        self._report("follow_arc", plan=None, steps=i + 1, reason=reason,
                     v=round(v, 4), omega=round(omega, 4),
                     lever_tail=lever_tail[:30], tcp_speed_tail=speed_tail[:30])
        if out is None:
            # No step was ever taken (the lever was collapsed on entry, or
            # max_steps=0) — that is a refusal before any physical commit (D6).
            return -1
        return out

    def drive_straight(self, distance, v: float = 0.10, max_steps=None, stop_when=None):
        """Drive the base straight by `distance` metres (negative = reverse), unplanned.

        The `follow_arc` template with the arc law replaced by a constant: no
        mplib leg at all, because the one place this is needed is the leg nothing
        plans from — backing off a closing dock with the fist still up against a
        panel, where every planned retreat refused (`elbow/forearm/wrist<->
        hingerightdoor`, 8/8, K106) and the folded arm cannot be reached without
        first moving away. The arm and torso are re-commanded at their *measured*
        qpos each step (the compliant hold, K104), `self.gripper_state` is re-emitted,
        and the base channel carries `[v, 0]` with the sign of `distance` — the
        ds_fetch base controller's forward slot is normalized with upper[0] = 1 m/s,
        asserted below exactly as `follow_arc` asserts it, so the slot value IS the
        speed in m/s.

        Stops on: the base_link having travelled `|distance|` from where it stood
        on entry (Euclidean xy — a drift sideways counts, a straight drive is what
        is commanded); `stop_when()` returning truthy (checked after each step);
        the episode's horizon; or `max_steps`, by default twice the nominal step
        count plus 20 (the controller ramps, and a 0.10 m/s ask lands a few steps
        late).

        Args:
            distance: metres along the base's own x axis; negative reverses.
            v: speed magnitude, m/s (0.10 default: half the pull speed, well
                inside the 1 m/s slot).
            max_steps: hard cap; None = `ceil(|distance| / (v * control_dt)) * 2 + 20`.
            stop_when: nullary callable; truthy stops the drive after that step.

        Returns:
            The last gym 5-tuple (or the guard's last step when already truncated);
            -1 only when no step was ever taken (D6: a refusal before any commit —
            `max_steps=0` is the only way to get one).

        Example:
            >>> res = solver.drive_straight(-0.35)              # doctest: +SKIP
            >>> if res == -1: return res                        # doctest: +SKIP
            >>> if solver.truncated: return res                 # doctest: +SKIP
        """
        if self.truncated:
            return self._guard.last_step
        cfg = self.env_agent.controller.controllers["base"].config
        assert getattr(cfg, "normalize_action", False) and float(cfg.upper[0]) == 1.0, (
            f"drive_straight assumes the ds_fetch base channel shape; got "
            f"normalize_action={getattr(cfg, 'normalize_action', None)} upper={cfg.upper}")
        dist = abs(float(distance))
        sign = -1.0 if float(distance) < 0 else 1.0
        speed = abs(float(v))
        if max_steps is None:
            dt = float(self.base_env.control_timestep)
            max_steps = int(np.ceil(dist / max(speed * dt, 1e-9))) * 2 + 20
        start_xy = np.array(self.env_agent.base_link.pose.sp.p[:2], dtype=np.float64)
        out = self._guard.last_step
        reason = "max-steps"
        travelled = 0.0
        i = -1
        for i in range(int(max_steps)):
            arm_action, body_action = self._hold_targets()
            base_action = np.array([np.clip(sign * speed, -1.0, 1.0), 0.0])
            action = self._compose(arm_action, body_action, base_action)
            out = self._step(action)
            travelled = float(np.linalg.norm(
                np.array(self.env_agent.base_link.pose.sp.p[:2], dtype=np.float64)
                - start_xy))
            if self._stopped_by_horizon("drive_straight"):
                reason = "horizon"
                break
            if travelled >= dist:
                reason = "distance"
                break
            if stop_when is not None and stop_when():
                reason = "stop_when"
                break
        self._report("drive_straight", plan=None, steps=i + 1, reason=reason,
                     distance=round(float(distance), 4), travelled=round(travelled, 4))
        if i < 0:
            return -1
        return out

    def turn_in_place(self, target_view_vec, omega: float = 0.6, tol: float = 0.03,
                      max_steps=None, stop_when=None):
        """Spin the base to face `target_view_vec`, unplanned — the twin of `drive_straight`.

        The closing dock needs this and nothing else does: W20a measured the search
        round's ladder ARRIVING at the measured rung (3.2, -1.1) on 4/4 seeds while
        `drive_base`'s own view rotation refused every time on the rotate-sweep
        phantom (`forearm_roll/wrist_flex <-> microwave door` a metre away, the
        K109/K111-filed hallucination), leaving the fist pointing 100 deg off the
        panel — the push then moved the door by 0.05 rad instead of closing it.
        `rotate_base_z` cannot help: its sweep is exactly what hallucinates.

        So: no plan, no sweep. Each step re-commands the arm and torso at their
        *measured* qpos (the compliant hold, K104), re-emits `self.gripper_state`,
        and drives the base's omega channel toward the shortest wrapped angle to the
        goal heading, slowing inside 4x`tol` so the stop is not overshot. The base
        rotates about its own axis, so a folded arm sweeps a fixed circle the caller
        has already measured clear.

        Args:
            target_view_vec: world direction to face (xy used; z ignored).
            omega: spin rate magnitude, rad/s (0.6 ~ 34 deg/s; the channel is
                normalized to +-3.14 rad/s and the scale is read from the config).
            tol: stop once the heading is within this many radians of the goal.
            max_steps: hard cap; None = `ceil(pi / (omega * control_dt)) * 2 + 20`
                (a half turn is the worst case, twice over, plus the ramp).
            stop_when: nullary callable; truthy stops the turn after that step.

        Returns:
            The last gym 5-tuple (or the guard's last step when already truncated);
            -1 only when no step was ever taken.

        Example:
            >>> res = solver.turn_in_place(np.array([0.0, 1.0, 0.0]))   # doctest: +SKIP
            >>> if res == -1: return res                                # doctest: +SKIP
        """
        if self.truncated:
            return self._guard.last_step
        cfg = self.env_agent.controller.controllers["base"].config
        assert getattr(cfg, "normalize_action", False) and float(cfg.upper[0]) == 1.0, (
            f"turn_in_place assumes the ds_fetch base channel shape; got "
            f"normalize_action={getattr(cfg, 'normalize_action', None)} upper={cfg.upper}")
        omega_scale = float(cfg.upper[1])
        goal = np.asarray(target_view_vec, dtype=np.float64).reshape(-1)[:2]
        if float(np.linalg.norm(goal)) < 1e-9:
            print("[turn_in_place] target_view_vec is zero; nothing to face")
            return -1
        goal_yaw = float(np.arctan2(goal[1], goal[0]))
        speed = abs(float(omega))
        if max_steps is None:
            dt = float(self.base_env.control_timestep)
            max_steps = int(np.ceil(np.pi / max(speed * dt, 1e-9))) * 2 + 20
        out = self._guard.last_step
        reason = "max-steps"
        err = np.pi
        i = -1
        for i in range(int(max_steps)):
            yaw = float(self.robot.get_qpos().cpu().numpy()[0][2])
            err = float((goal_yaw - yaw + np.pi) % (2 * np.pi) - np.pi)
            if abs(err) <= tol:
                reason = "aimed"
                break
            arm_action, body_action = self._hold_targets()
            # slow down inside 4*tol so the stop is not overshot by a whole step
            scale = min(1.0, abs(err) / max(4.0 * tol, 1e-9))
            w = np.sign(err) * speed * max(scale, 0.15)
            base_action = np.array([0.0, np.clip(w, -omega_scale, omega_scale) / omega_scale])
            action = self._compose(arm_action, body_action, base_action)
            out = self._step(action)
            if self._stopped_by_horizon("turn_in_place"):
                reason = "horizon"
                break
            if stop_when is not None and stop_when():
                reason = "stop_when"
                break
        self._report("turn_in_place", plan=None, steps=i + 1, reason=reason,
                     goal_yaw=round(goal_yaw, 4), err=round(float(err), 4))
        if i < 0:
            return -1
        return out

    def follow_path(self, result, refine_steps: int = 0, refine: bool = False):
        return self.follow_forward_path_w_refinement(result, refine)

    def follow_forward_path_w_refinement(
        self, result, refine: bool = False, static=False, stop_when=None
    ):
        # K55. A plan can come back `Success` with **no knots**: the goal was already
        # satisfied to within the planner's tolerance, so there is nothing to
        # interpolate. `_final_qpos_dict` then indexes `[-1]` into an empty array and
        # the episode dies with `IndexError: index -1 is out of bounds for axis 0
        # with size 0`. Nothing in the shipped staging produced one, because every
        # move it makes has something to do; splitting a move in two (do the torso's
        # share first, then the arm's — `water_plants_planner.TORSO_FIRST`) makes the
        # second half a no-op whenever the first half covered it, which is exactly the
        # case for a torso-only target. Executing nothing is the correct behaviour
        # here, not a crash.
        if np.asarray(result["position"]).shape[0] == 0:
            self._report("follow", plan="empty", exec=0, reached=True)
            return self.idle_steps(t=1)

        qpos_dict_final = self._final_qpos_dict(result)
        n_step = result["position"].shape[0]

        # In `pd_joint_delta_pos` the knot only advances once the arm is within
        # `DELTA_LAG_GATE` of the current one (`_arm_lag`): the delta controller caps the
        # PD error at one step (0.1 rad), so it caps the torque and the speed, and an arm
        # that falls behind an open-loop clock is pulled toward knots AHEAD of it — a
        # straight line in joint space through whatever the plan went around (SeasonDish
        # 3608, 2026-09-09: the approach cut a corner and knocked the shaker 32 cm). The
        # stall re-issues the same knot with the BASE HELD, so the base still integrates
        # exactly the plan's velocities; `DELTA_LAG_MAX_STALL` bounds it per knot.
        gate = self.control_mode == "pd_joint_delta_pos"
        i, stalled, stalls_total = 0, 0, 0
        while i < n_step:
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

            assert self.control_mode in self.COMPOSE_MODES, self.control_mode

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
            self._note_lateral(base_vel_wrt_world, is_forward, base_direction)

            base_action = np.array([0.0, 0.0])
            base_action[0] = is_forward

            stall = gate and stalled < self.DELTA_LAG_MAX_STALL and self._arm_lag(arm_action) > self.DELTA_LAG_GATE
            if stall:
                base_action[:] = 0.0
            action = self._compose(arm_action, body_action, base_action)
            if self.verbose:
                print("arm Action:", np.round(arm_action, 4))
                print("body Action:", np.round(body_action, 4))
                print("base Action:", np.round(base_action, 4))
                print("qpos: ", np.round(self.robot.get_qpos().cpu().numpy()[0], 4))
            obs, reward, terminated, truncated, info = self._step(action)
            if stall:
                stalled += 1
                stalls_total += 1
            else:
                i += 1
                stalled = 0
            if self._stopped_by_horizon("follow_forward_path_w_refinement"):
                break
            if stop_when is not None and stop_when():
                self._report("follow_path", stopped="touch", at=i, of=n_step)
                return obs, reward, terminated, truncated, info
        if stalls_total:
            self._report("follow_path", to_trace=False, knots=n_step, lag_stalls=stalls_total)

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

                action = self._compose(arm_action, body_action, base_action)
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
        # An empty dict means the plan had no final knot (`_final_qpos_dict` on an
        # empty `position` array — W25, 2026-09-02): nothing to be close to, so it
        # is not reached. Without this the diagnostic line raised KeyError and took
        # the episode with it.
        if not target_dict:
            return False
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

    def change_gripper_state(self, t=6, gripper_state=OPEN, stop_when=None):
        """Drive the gripper to `gripper_state` for `t` steps.

        `stop_when`, when given, is polled after every step and ends the motion early.
        `None` (the default) leaves every existing caller byte-identical. It exists because
        the fingers are **position**-commanded: they reach exactly 0.0000 aperture in both
        measured `dropped during the lift` runs and never reopen, so a contact that begins
        to slip simply lets them close further — on one seed a 45 mm bottle ends pinched at
        21 mm, mid-topple, having been levered over by the closing motion itself while the
        arm moved 0.4 mm."""
        if self.truncated:
            return self._guard.last_step
        self.gripper_state = gripper_state
        arm_action = self.env_agent.controller.controllers["arm"].qpos[0].cpu().numpy()
        body_action = (
            self.env_agent.controller.controllers["body"].qpos[0].cpu().numpy()
        )
        base_action = np.array([0, 0])

        for i in range(t):
            action = self._compose(arm_action, body_action, base_action)
            obs, reward, terminated, truncated, info = self._step(action)
            if self._stopped_by_horizon("change_gripper_state"):
                break
            if stop_when is not None and stop_when():
                self._report("change_gripper_state", to_trace=False, stopped_at=i + 1, of=t)
                break
        return obs, reward, terminated, truncated, info

    def close_gripper(self, t=6, stop_when=None):
        return self.change_gripper_state(t=t, gripper_state=CLOSED, stop_when=stop_when)

    def open_gripper(self, t=6):
        return self.change_gripper_state(t=t, gripper_state=OPEN)

    def start_tape(self):
        """Begin keeping every action this solver emits, and return the tape.

        The tape is how a stage UNDOES itself. Restoring a recorded pose with a fresh
        plan is not the reverse of a motion — it re-picks the IK branch, it re-decides
        the path, and it leaves any channel the pose does not describe (the fingers)
        with no reversal at all. Replaying the recorded actions backwards is the reverse
        of the motion, exactly, for every channel at once: the tape keeps the ABSOLUTE
        form of every action (`_compose`: arm and body slots as position targets,
        whatever the env's control mode), so reversing their order retraces the path,
        and the gripper slot carries its own state, so it re-closes precisely where it
        opened. Nothing has to be timed by hand. In `pd_joint_delta_pos` the replay
        re-derives each increment from the measured pose at the step (`_from_abs`).

        The tape lives on `self._guard`, which is where this class's primitives actually
        reach the env — the guard captured its own reference at construction, so a
        wrapper around `self.env` records nothing at all (measured the hard way). Cheap:
        one array copy per step, and nothing when no tape is running.

        Example:
            >>> tape = planner.start_tape()      # doctest: +SKIP
            >>> ...                              # doctest: +SKIP
            >>> planner.stop_tape()              # doctest: +SKIP
            >>> planner.replay_tape(tape, reverse=True, freeze_base=True)  # doctest: +SKIP
        """
        if self._guard.tape is None:
            self._guard.tape = []
        return self._guard.tape

    def stop_tape(self):
        """Stop keeping actions; returns the tape (or None if none was running)."""
        tape, self._guard.tape = self._guard.tape, None
        return tape

    KEEP_BASE_KV = 4.0      # m/s per metre of along-heading error
    KEEP_BASE_KI = 8.0      # m/s per metre-second: kills the standing offset
    KEEP_BASE_VMAX = 0.15   # m/s, the cap drive_straight uses 0.10 of
    KEEP_BASE_KW = 4.0      # rad/s per radian of heading error
    KEEP_BASE_KIW = 8.0     # rad/s per radian-second
    KEEP_BASE_WMAX = 0.5    # rad/s, below turn_in_place's 0.6
    KEEP_BASE_REPORT_M = 0.005  # sideways slide at which the obstacle is named

    def touching_now(self, limit: int = 4) -> list:
        """Which link pairs the planning world says are in contact, as things stand.

        A NAMED obstacle instead of a guess. The reversal replays recorded actions, so
        nothing along it is planned and nothing refuses — when the hand strikes
        something the only evidence is the base being shoved, and that says how much,
        never what. This asks the same world the planners ask, at the state physics is
        actually in.

        Two honest limits. The world is synced from the simulator first, so it reports
        where things ARE; but a stage that removed an articulation from the planning
        world (`contact_stroke` does, by substring) leaves it unreported here too, and
        pairs already touching before the motion are not distinguished from new ones.
        Read it as a name for what is there, not as a verdict.

        Example:
            >>> solver.touching_now()                       # doctest: +SKIP
            ['ds_fetch_l_gripper_finger_link<->cab_2_hingerightdoor']
        """
        try:
            planner = self.planner
            planner.update_from_simulation()
            world = planner.planning_world
            pairs = list(world.check_robot_collision()) + list(world.check_self_collision())
        except Exception as exc:                      # a read must never end an episode
            return [f"unreadable ({type(exc).__name__})"]
        return sorted({f"{c.link_name1}<->{c.link_name2}" for c in pairs})[:limit]

    def _base_station(self):
        """A closure that returns the two base slots holding the base where it is now.

        The base channel is normalised VELOCITY — slot 0 in m/s (`cfg.upper[0]` is 1.0,
        asserted by `drive_straight`) and slot 1 scaled by `cfg.upper[1]` the way
        `turn_in_place` scales it. Commanding zeros therefore asks for no motion, which
        is not the same as staying put: with the arm swinging out, the base slid up to
        64 mm backwards under exactly that command. A differential base cannot correct
        sideways without turning, and it does not need to — the slide is along its own
        heading, which is also the gripper's approach.

        The term that matters is the INTEGRAL one. A proportional hold against a steady
        push settles at an offset of push / (gain * dt) and stays there: at a first
        attempt's gains that offset was 20 mm, most of the miss it was meant to remove.
        The integral term takes it to zero, and the pair is damped (zeta ~ 0.7) and
        settles well inside the window the replay has.

        Example:
            >>> keep = solver._base_station()   # doctest: +SKIP
            >>> action[-2:] = keep()            # doctest: +SKIP
        """
        cfg = self.env_agent.controller.controllers["base"].config
        w_scale = float(cfg.upper[1])
        pose0 = self.env_agent.base_link.pose.sp
        p0 = np.array(pose0.p[:2], dtype=np.float64)
        yaw0 = float(self.robot.get_qpos().cpu().numpy()[0][2])
        dt = float(self.base_env.control_timestep)
        acc = [0.0, 0.0]
        v_cap = self.KEEP_BASE_VMAX / max(self.KEEP_BASE_KI, 1e-9)
        w_cap = self.KEEP_BASE_WMAX / max(self.KEEP_BASE_KIW, 1e-9)

        def keep():
            pose = self.env_agent.base_link.pose.sp
            here = np.array(pose.p[:2], dtype=np.float64)
            T = np.asarray(pose.to_transformation_matrix(), dtype=np.float64)
            bx, by = T[:3, 0], T[:3, 1]
            forward = float((p0 - here) @ bx[:2])
            # A forward-only base can answer `forward`. It cannot answer `sideways`
            # without turning, so that term is reported rather than corrected: if the
            # miss lives there, the station-keeper is the wrong instrument for it.
            side = float((p0 - here) @ by[:2])
            trail = self.replay_keep.get("trail")
            if trail is None:
                trail = self.replay_keep["trail"] = []
            trail.append(round(side, 4))
            self.replay_keep.update(fwd=round(forward, 4), side=round(side, 4),
                                    n=len(trail))
            # Name the obstacle AT THE MOMENT the base starts to give, not afterwards.
            # Read at the end it comes back empty every time, and no wonder: the base
            # yields until the contact is relieved, so by then there is nothing left to
            # see. The first millimetres of the slide are the only window.
            if abs(side) > self.KEEP_BASE_REPORT_M and "hit" not in self.replay_keep:
                self.replay_keep["hit"] = self.touching_now()
                self.replay_keep["hit_at"] = len(trail)
            yaw = float(self.robot.get_qpos().cpu().numpy()[0][2])
            err = float((yaw0 - yaw + np.pi) % (2 * np.pi) - np.pi)
            acc[0] = float(np.clip(acc[0] + forward * dt, -v_cap, v_cap))
            acc[1] = float(np.clip(acc[1] + err * dt, -w_cap, w_cap))
            v = float(np.clip(self.KEEP_BASE_KV * forward + self.KEEP_BASE_KI * acc[0],
                              -self.KEEP_BASE_VMAX, self.KEEP_BASE_VMAX))
            w = float(np.clip(self.KEEP_BASE_KW * err + self.KEEP_BASE_KIW * acc[1],
                              -self.KEEP_BASE_WMAX, self.KEEP_BASE_WMAX))
            return np.array([v, np.clip(w, -w_scale, w_scale) / w_scale])

        return keep

    def replay_tape(self, tape, *, reverse: bool = True, freeze_base: bool = True,
                    stop_when=None, hold: int = 0, hold_tol: float = 2e-4,
                    keep_base: bool = False, stretch: int = 1,
                    gripper_cap: float | None = None):
        """Step the recorded actions, backwards by default.

        `freeze_base` zeroes the base's two slots, which is what a window whose base
        stood still needs: those slots are the base controller's NORMALIZED VELOCITY
        (see `drive_straight`), and a velocity does not reverse by reversing the order
        of the recording, it reverses by changing sign. Rather than guess a sign per
        primitive, a stage that moved its base reverses that part with the base's own
        primitives and hands this method the arm-and-gripper window only.

        Args:
            tape: the list `start_tape` filled.
            reverse: play it backwards (the default and the point).
            freeze_base: zero the last two slots.
            stop_when: nullary callable polled after every step; truthy ends the replay.
            hold: how many extra steps the LAST emitted action may be re-issued for,
                so the position loop can catch up with it. 0 keeps the old behaviour
                byte for byte. See the note below for why a replay needs this.
            hold_tol: the hold stops early once no joint outside the base moves more
                than this (radians) in a step.
            keep_base: hold the base where it stood when the replay began, instead of
                merely commanding it nothing. Zero on a VELOCITY channel is not a brake:
                measured (2026-09-04, seven retraces) the base slid 17 to 64 mm backwards
                while the arm unfolded and reached, and that slide was the whole miss —
                the hand landed its joints to 1e-4 rad and still stopped 17 to 61 mm
                short of the bar ALONG THE APPROACH, the one direction the gripper does
                not forgive. The correction is the base's own two slots, proportional
                plus integral, and it plans nothing. It answers the along-heading and
                heading terms completely (measured: 1.4 mm and 0.2 deg) and the SIDEWAYS
                one not at all, because a forward-only base cannot — see `stretch`.
            stretch: emit each recorded action this many times, so the reversal runs at
                1/stretch of the speed it was recorded at. Still exactly the recording —
                the same targets in the same order — but the reaction the arm's own
                motion puts into the base falls with the speed, and what is left of the
                miss once the base is held is that reaction coming out SIDEWAYS: 41 to
                54 mm of it on the rounds that fail, 4 to 11 mm on the rounds that grip.
                1 replays at the recorded rate, byte for byte.
            gripper_cap: clamp the gripper slot of every replayed frame to at most this
                normalised command (+1 fully open, -1 shut). The recorded targets for
                the arm are untouched; only how wide the hand is while it retraces them.
                Why it exists: the reversed retreat approaches the bar with the pads at
                +1, a 100 mm span, and the bar stands 5 cm from the neighbouring leaf's
                edge — the outer finger meets that edge on the way in (keep_hit names it,
                the shove starts in the second quarter). The original grasp survives the
                same geometry only because its last leg is a contact stroke that stops
                at first touch. None = the recording as it is.

        Returns:
            the last gym 5-tuple, or `self._guard.last_step` when nothing was stepped.

        Attributes set for the caller: `replay_held` (steps the hold actually spent)
        and `replay_residual` (the last per-step motion it saw, radians).
        """
        if self.truncated or not tape:
            return self._guard.last_step
        was, self._guard.tape = self._guard.tape, None   # a replay does not tape itself
        out, last = self._guard.last_step, None
        self.replay_keep = {}
        self.replay_frames = 0   # frames actually stepped, so a caller can undo exactly them
        keep = self._base_station() if keep_base else None
        stop = False
        for a in (reversed(tape) if reverse else tape):
            if self.truncated or stop:
                break
            self.replay_frames += 1
            action = np.asarray(a, dtype=np.float64).copy()
            if gripper_cap is not None:
                n_arm = len(self.env_agent.controller.controllers["arm"].config.joint_names)
                if action.shape[0] > n_arm:
                    action[n_arm] = min(float(action[n_arm]), float(gripper_cap))
            for _ in range(max(1, int(stretch))):
                if self.truncated:
                    break
                if action.shape[0] >= 2:
                    if keep is not None:
                        action[-2:] = keep()
                    elif freeze_base:
                        action[-2:] = 0.0
                # the tape holds the ABSOLUTE form (see `_compose`); the env gets it
                # in its own mode, and a nested tape keeps the absolute entry
                out = self._guard.step(self._from_abs(action), tape_entry=action)
                self.elapsed_steps += 1
                if stop_when is not None and stop_when():
                    stop = True
                    break
            last = action
        # Hold the last command until the loop has caught up with it. A recorded
        # action is a TARGET, and the controller trails its target by a step or two:
        # forward, frame k was approached from frame k-1, so the hand arrived at the
        # grip already converged; backwards, frame 0 is approached from frame 1, and
        # the replay ends the moment the target is ISSUED, not when it is reached.
        # That lag is not a small correction — measured on seed 2 (2026-09-04) the
        # base came back to 12 mm while the hand stopped 20 to 71 mm short of the pose
        # it was reversing to. Re-issuing the same action costs nothing but steps and
        # is still the recording, not a new plan. It ends when the arm AND the fingers
        # have stopped, which is also what gives the pads their closing steps.
        self.replay_held, self.replay_residual, self.replay_qerr = 0, 0.0, 0.0
        if last is not None and hold > 0 and not self.truncated:
            prev = self.robot.get_qpos().cpu().numpy()[0][3:]
            for _ in range(int(hold)):
                if keep is not None and last.shape[0] >= 2:
                    last[-2:] = keep()
                out = self._guard.step(self._from_abs(last), tape_entry=last)
                self.elapsed_steps += 1
                self.replay_held += 1
                now = self.robot.get_qpos().cpu().numpy()[0][3:]
                self.replay_residual = float(np.max(np.abs(now - prev)))
                prev = now
                if self.truncated or (stop_when is not None and stop_when()):
                    break
                if self.replay_residual < hold_tol:
                    break
        self._guard.tape = was
        # Adopt the gripper the replay just commanded. Every primitive re-emits
        # `self.gripper_state`, so leaving it stale undoes the replay on the very next
        # held step: measured — the tape brought the pads onto the bar and the settle
        # that followed re-opened them to 0.0977, the aperture of a hand that never
        # closed. The slot is the one after the arm's own joints.
        if last is not None:
            arm = self.env_agent.controller.controllers["arm"]
            n_arm = len(arm.config.joint_names)
            # How far the ARM is from the target it was last given, in its own joints.
            # This is what separates "the loop had not caught up" from "the loop caught
            # up and the miss is somewhere else" — measured in the same units the tape
            # is written in, so no frame or IK branch can confuse it.
            self.replay_qerr = float(np.max(np.abs(
                np.asarray(arm.qpos[0].cpu().numpy(), dtype=np.float64)
                - np.asarray(last[:n_arm], dtype=np.float64))))
            if last.shape[0] > n_arm:
                self.gripper_state = float(last[n_arm])
        return out

    def hold_head(self, pan: float, tilt: float, t: int = 10, stop_on_success: bool = False,
                  ramp: int = 0):
        """Hold the base and the arm and command the HEAD to (pan, tilt) for `t` steps.

        The base cameras ride on `head_camera_link`, so this aims them without moving
        the base or the arm — a look that leaves the base where the closing has to start
        (W22c, 2026-09-07). Every plan the solver executes writes the head back to 0,
        which is what the caller relies on afterwards; `idle_steps` holds whatever the
        body controller last targeted, so a look is: hold_head(pan, tilt), read the
        verdict, hold_head(0, 0).

        `ramp` > 0 spreads the turn over that many steps (a linear ramp of the target
        from the head's current angles), then holds for the rest of `t`. Without it the
        body controller reaches the target inside one control step, 0.05 s — the owner
        called that look "very fast" (2026-09-08).
        """
        if self.truncated:
            return self._guard.last_step
        arm_action = self.env_agent.controller.controllers["arm"].qpos[0].cpu().numpy()
        body_action = (
            self.env_agent.controller.controllers["body"].qpos[0].cpu().numpy().copy()
        )
        start = body_action[:2].copy()                                 # the head now
        goal = np.array([float(pan), float(tilt)])                      # head_pan, head_tilt
        base_action = np.array([0, 0])
        out = self._guard.last_step
        for i in range(int(t)):
            frac = min(1.0, (i + 1) / ramp) if ramp > 0 else 1.0
            body_action[:2] = start + frac * (goal - start)
            action = self._compose(arm_action, body_action, base_action)
            out = self._step(action)
            if self._stopped_by_horizon("hold_head"):
                break
            if stop_on_success:
                info = out[-1] if isinstance(out, tuple) else {}
                ok = info.get("success", False) if isinstance(info, dict) else False
                if bool(ok[0] if hasattr(ok, "__len__") else ok):
                    break
        return out

    def idle_steps(self, t=20):
        if self.truncated:
            return self._guard.last_step
        arm_action = self.env_agent.controller.controllers["arm"].qpos[0].cpu().numpy()
        body_action = (
            self.env_agent.controller.controllers["body"].qpos[0].cpu().numpy()
        )
        base_action = np.array([0, 0])
        for i in range(t):
            action = self._compose(arm_action, body_action, base_action)
            obs, reward, terminated, truncated, info = self._step(action)
            if self._stopped_by_horizon("idle_steps"):
                break
        return obs, reward, terminated, truncated, info
