"""SHARED base for the cabinet-retrieval family — registers NOTHING.

The convention is one scene file = one registered task, and a task file may
not import another task's file (the registration side effect —
docs/review-inherited-code.md §12). A variant family therefore keeps its
machinery here, in a module named in SHARED_SCENE_MODULES, and each variant's
own file holds only its config override and its @register_env line.

MikasaCabinetRetrieval-v0: take the cup out of the opened wall cabinet, put it down.

The first motor-only task in this package — every other registered task is a memory
task (AGENTS.md: "memory is the point"). This one is deliberately built as the
*substrate* for a memory variant rather than an exception to the rule: the door
angle is a config field (a later variant closes it and adds the arc-pull), the
place target is a named per-episode point (a later variant can make it "where it
was"), and nothing in the observation names the answer-shaped facts. The design
blank naming those hooks is docs/task-designs/F-cabinetretrieval-v0.md.

Everything geometric here is measured, not assumed, and the measurement is W13
(tools/probes/w13_cabinet_take.py, journal 2026-08-28): the door holds where it is
put; the cabinet floor is real at z = 1.4200; a vertical top grasp NEVER plans
through the opening (0/9 — the first grid said otherwise and every 'plan' was a
phantom return from the solver's truncation guard); the horizontal side grasp
plans, holds, and the full chain ends with the cup standing on the counter.

Scene: kitchen 102 (scene_idx=0, pinned). The wall cabinet `cab_main` hangs over
`counter_main` at x [1.75, 2.75], box z [1.39, 2.31]; its right door — the one W12
measured opening — exposes the right half, x [2.25, 2.75], which sits over the
sink-free counter span. The cup spawns on the cabinet's interior floor inside that
half; the robot starts on the open floor south of the counter row.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.scene_builder.robocasa.scene_builder import RoboCasaSceneBuilder
from mani_skill.utils.structs import Actor, Pose

from .robocasa_utils import (
    parking_pose,
    load_objaverse_actor,
    require_get_fixture,
    restore_task_tensor,
)

#: Keys `get_state_dict` adds beyond the simulator state. One place, so the state
#: tests and the implementation cannot drift apart.
TASK_STATE_KEYS = ("has_been_lifted", "succeeded", "place_target")

INSTRUCTIONS = (
    "Take the cup out of the open cabinet and put it on the counter.",
    "Get the cup down from the cabinet onto the counter.",
    "Fetch the cup from the cabinet shelf and set it on the counter.",
)


@dataclass
class CabinetRetrievalConfig:
    """Every threshold named; magic numbers in evaluate() are how failures go mute."""

    horizon: int = 900
    """Episode length, by the repo's horizon rule, measured 2026-08-28 against the
    SHIPPED oracle (the duck-and-drive one — an earlier draft measured 800 and the
    duck stage moved the calibration): seeds 11 and 12 (never the eval seeds 0-9)
    finished at steps 453 and 556; no cue phase, so 556 x 1.5 = 834, rounded up to
    a multiple of 50 then 100 -> 900. Must equal max_episode_steps in the
    decorator, which reads this constant. Expires with the oracle."""

    # -- the cabinet -----------------------------------------------------------
    cabinet_name: str = "cab_main_main_group"
    """The wall cabinet's fixture/articulation name stem. Exact stem, not a loose
    substring: `get_fixture` falls back to random choice among substring matches."""
    counter_name: str = "counter_main_main_group"
    open_door: str = "right"
    """Which door starts open. v0 supports only "right" (the door W12/W13 measured;
    it exposes the half over the sink-free counter span) — validate() enforces it.
    "left"/"both" are a later variant's knobs, listed here so the field's meaning
    is fixed before they exist."""
    door_hinge: str = "rightdoorhinge"
    door_open_rad: float = 1.6
    """The hinge angle the episode starts at. W13: set once at reset, it holds at
    exactly this value through physics (free hinge, damping 2), and the open
    panel's shapes reach only y = -0.844 — clear of the robot's floor. 0.0 means
    CLOSED: since K103/K104 the oracle opens the door itself (the arc pull reaches
    the full 1.6), and `MikasaCabinetRetrievalClosed-v0` starts there. The ajar
    band (0, 0.9) stays refused — see validate(). The upper bound is the hinge
    stop at 3.0."""
    require_door_closed: bool = False
    """Success additionally requires door_angle <= door_closed_rad at the moment
    the five place-predicates hold — the variant where the robot must close the
    door behind itself. False here: v0's published 9/10 was measured without the
    requirement and stays byte-identical (evaluate() multiplies the raw verdict
    by all-ones when this is off)."""
    door_closed_rad: float = 0.15
    """The "closed" threshold on the hinge angle. Not 0.0: the free hinge with
    damping 2 holds wherever it is left (W13), so the last few degrees depend on
    where the fist released the panel — demanding exact zero would fail honest
    closes. 0.15 rad ~ 8.6 deg leaves a ~7.3 cm gap at the free edge, which the
    arm cannot pass (W12: even 0.53 rad refused the arm), so a door under this
    angle is closed for every purpose the task cares about."""

    # -- the object and its spawn ---------------------------------------------
    object_category: str = "cup"
    object_name: str = "cup"
    object_index: int = 0
    """The same cup instance as burner and water_plants (index 0), so its mesh,
    settle behaviour and is_grasping thresholds are already measured territory."""
    shelf_top_z: float = 1.4200
    "The cabinet's interior floor. Measured by W13 (cup mesh bottom at settle)."
    spawn_clearance: float = 0.02
    """The cup's mesh BOTTOM starts this far above the shelf — never the origin
    (season_dish.py records PhysX pushing an origin-spawned bottle through the
    counter into the cabinet below)."""
    spawn_depth: float = -0.19
    spawn_jitter_depth: float = 0.09
    """Cup y (depth into the cabinet): spawn_depth +/- jitter spans [-0.28, -0.10],
    exactly the band W13's grid measured graspable from the work dock."""
    spawn_margin_x: float = 0.10
    spawn_jitter_x: float = 0.06
    """Cup x inside the exposed half [2.25, 2.75]: margin off both edges (the door
    panel swings near the x=2.75 hinge side; the closed left door walls off x<2.25),
    jitter within what remains. validate() keeps margin+jitter inside the half."""

    # -- the place -------------------------------------------------------------
    place_across: float = -0.525
    """World y of the place target: the open-sky band of the counter (wall cabinets
    hang forward to y=-0.448, the counter reaches y=-0.65; -0.525 is the middle —
    the same band water_plants derived for its pots)."""
    place_radius: float = 0.15
    max_height_above_counter: float = 0.06
    "Measured from the cup's MESH BOTTOM, not the origin (the origin sits 0.0575 up)."
    settle_lin_speed: float = 0.05
    settle_ang_speed: float = 0.2

    # -- the robot start -------------------------------------------------------
    start_xy: tuple = (2.55, -1.90)
    "Open floor south of the counter row, one honest drive from the cabinet."
    start_yaw_deg: float = 90.0
    start_jitter_xy: float = 0.08
    start_jitter_yaw: float = 0.10

    def validate(self) -> None:
        """Refuse to build on self-contradictory numbers. Called from __init__."""
        assert self.horizon > 0
        assert self.open_door == "right", (
            f"open_door={self.open_door!r}: v0 supports only the right door — the "
            "one whose opening, handle and panel sweep W12/W13 measured. A left or "
            "both-doors variant re-measures the spawn half and the dock first."
        )
        assert self.door_open_rad == 0.0 or 0.9 <= self.door_open_rad <= 2.9, (
            f"door_open_rad={self.door_open_rad}: 0.0 means CLOSED — the oracle "
            "opens it itself with the arc pull (K103/K104 measured the full 1.6 "
            "reachable). Between 0 and ~0.9 is refused twice over: the opening "
            "does not admit the arm (W12: 0.53 is all a straight pull achieves, "
            "and the arm was refused at that angle), and the measured handle "
            "grasp (W12's approach/closing at the CLOSED bar pose) does not apply "
            "to an ajar door's rotated bar. 3.0 is the hinge stop."
        )
        assert 0 < self.door_closed_rad < 0.9, (
            f"door_closed_rad={self.door_closed_rad}: must be positive (exact "
            "zero is unreachable — the free hinge holds where the fist released "
            "it, W13) and under 0.9, the ARM_PASS boundary (W12): at 0.9 the "
            "opening starts admitting the arm, and a threshold that calls an "
            "arm-passable door 'closed' contradicts what closed is for."
        )
        assert 1.39 <= self.shelf_top_z <= 2.31, (
            f"shelf_top_z={self.shelf_top_z} is outside the cabinet box z-range"
        )
        assert self.spawn_clearance > 0
        assert self.spawn_jitter_depth >= 0 and self.spawn_jitter_x >= 0
        assert -0.37 <= self.spawn_depth - self.spawn_jitter_depth and \
               self.spawn_depth + self.spawn_jitter_depth <= 0.0, (
            "the spawn depth band leaves the cabinet box (y in [-0.37, 0])"
        )
        assert self.spawn_margin_x + self.spawn_jitter_x < 0.25, (
            "spawn margin + jitter exceed the exposed half's 0.25 m half-width"
        )
        assert self.place_radius > 0 and self.max_height_above_counter > 0
        assert self.settle_lin_speed > 0 and self.settle_ang_speed > 0
        assert -0.65 < self.place_across < -0.448, (
            f"place_across={self.place_across} is outside the open-sky counter band "
            "(-0.65 counter edge, -0.448 wall-cabinet overhang)"
        )


class CabinetRetrievalTaskBase(BaseEnv):
    """Take the cup out of the opened wall cabinet and stand it on the counter."""

    SUPPORTED_ROBOTS = ["mikasa_ds_fetch", "fetch"]
    SUPPORTED_REWARD_MODES = ["sparse", "none"]

    cfg = CabinetRetrievalConfig()

    cup: Actor

    def __init__(self, *args, robot_uids="mikasa_ds_fetch", scene_idx: int | None = 0, **kwargs):
        # Kitchen 102 pinned by default: every number in the config was measured
        # there (W12/W13), and a drawn kitchen would place the cabinet elsewhere
        # or nowhere.
        self.cfg.validate()
        self.scene_idx = scene_idx
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # ---------------------------------------------------------------- sensors --

    @property
    def _default_sensor_configs(self):
        # ds_fetch carries its own head and wrist rigs; a task-level camera would
        # double-render. Same call water_plants makes.
        return []

    @property
    def _default_human_render_camera_configs(self):
        # Framed for THIS task: the open door, the cabinet interior, the counter
        # below and the robot in one 512^2 frame. The eye stays inside the room
        # (floor runs to y=-3.04). Checked on a pilot frame, not by a threshold.
        pose = sapien_utils.look_at(eye=[2.25, -2.75, 2.05], target=[2.35, -0.40, 1.30])
        return CameraConfig("render_camera", pose, 512, 512, 1.2, 0.01, 100)

    # ------------------------------------------------------------------ load --

    def _load_agent(self, options: dict):
        super()._load_agent(options, parking_pose(self))

    def _load_scene(self, options: dict):
        """Geometry and initial poses only — never set_pose (discarded by _setup)."""
        super()._load_scene(options)

        self.scene_builder = RoboCasaSceneBuilder(self)
        if self.scene_idx is None:
            self.scene_builder.build()
        else:
            self.scene_builder.build([self.scene_idx] * self.num_envs)
        self._fix_ds_fetch_collision_bits()

        # Fixtures per env: the counter for the place plane, the cabinet for the
        # spawn half. Exact-stem lookup; get_fixture's substring fallback draws
        # RANDOMLY among matches, which is why the stems are full group names.
        self.counters = [
            require_get_fixture(
                self.scene_builder, self.scene_builder.scene_data[i]["fixtures"],
                self.cfg.counter_name, scene_idx=self.scene_idx,
            )
            for i in range(self.num_envs)
        ]
        self.cabinets = [
            require_get_fixture(
                self.scene_builder, self.scene_builder.scene_data[i]["fixtures"],
                self.cfg.cabinet_name, scene_idx=self.scene_idx,
            )
            for i in range(self.num_envs)
        ]

        # The door articulation, resolved ONCE. This task is the first in the repo
        # to touch kitchen-articulation qpos, so the resolution is deliberately
        # strict: every articulation whose name carries the cabinet stem, exactly
        # one per sub-scene, cached as (articulation, hinge_index_in_qpos).
        matches = [
            (name, art) for name, art in self.scene.articulations.items()
            if self.cfg.cabinet_name in name
        ]
        assert len(matches) == self.num_envs, (
            f"expected one {self.cfg.cabinet_name!r} articulation per env, found "
            f"{[n for n, _ in matches]!r} for num_envs={self.num_envs}. RoboCasa "
            "builds fixtures per sub-scene; if this ever merges into one batched "
            "articulation, the per-env door write below needs the batched form."
        )
        self._door = []
        for _name, art in sorted(matches):
            names = [j.name for j in art.get_active_joints()]
            assert self.cfg.door_hinge in names, (_name, names)
            self._door.append((art, names.index(self.cfg.door_hinge)))

        # The cup, on the cabinet's interior floor. Initial pose only; the real
        # per-episode spawn happens in _initialize_episode by mesh bottom.
        self.cup = load_objaverse_actor(
            self, self.cfg.object_category, self.cfg.object_name,
            sapien.Pose(p=[2.50, float(self.cfg.spawn_depth),
                           float(self.cfg.shelf_top_z) + 0.10]),
            index=self.cfg.object_index,
        )

        # Per-env derived geometry, cached as device tensors once.
        half_lo, half_hi = self._exposed_half()
        centre_x = (half_lo + half_hi) / 2.0
        counter_tops = np.array(
            [float(np.asarray(c.pos)[2] + np.asarray(c.size)[2] / 2.0)
             for c in self.counters], dtype=np.float32,
        )
        self._counter_top_z = torch.as_tensor(counter_tops, device=self.device)
        self._spawn_centre_x = torch.full(
            (self.num_envs,), float(centre_x), device=self.device
        )
        self.place_target = torch.zeros((self.num_envs, 3), device=self.device)

    def _exposed_half(self) -> tuple[float, float]:
        """The x-span the open right door exposes, shrunk by the spawn margin.

        Derived from the cabinet fixture's own box rather than written down: the
        right door covers the right half of the cabinet's along-extent, so the
        exposed span is [centre, right edge], inset by `spawn_margin_x` on both
        sides (the closed left door walls off the centre; the open panel swings
        near the right edge).
        """
        cab = self.cabinets[0]
        pos = np.asarray(cab.pos, dtype=np.float64)
        size = np.asarray(cab.size, dtype=np.float64)
        lo = float(pos[0])                      # the cabinet's x centre
        hi = float(pos[0] + size[0] / 2.0)      # its right edge
        return lo + self.cfg.spawn_margin_x, hi - self.cfg.spawn_margin_x

    def _fix_ds_fetch_collision_bits(self):
        """Restore the wheel/base exemption scene_builder.py:490 skips for our uid."""
        if self.robot_uids == "fetch" or self.agent is None:
            return
        for link in self.agent.robot.links:
            for body in link._bodies:
                for shape in body.get_collision_shapes():
                    groups = shape.get_collision_groups()
                    for bit in range(25, 30):
                        groups[2] |= 1 << bit
                    shape.set_collision_groups(groups)

    # ------------------------------------------------------------ initialize --

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.scene_builder.initialize(env_idx)
            self._restore_robot(env_idx)

            # First task in the repo to set kitchen-articulation qpos at reset.
            # RoboCasaSceneBuilder.initialize() restores ONLY the robot
            # (scene_builder.py:563-577) and nothing else touches fixture joints,
            # so the door is this task's to set every episode — otherwise episode
            # 2 starts wherever episode 1's contacts left the hinge. W13: a free
            # hinge holds a set angle exactly.
            for i in env_idx.tolist():
                art, j = self._door[int(i)]
                q = art.get_qpos()
                q = q.clone() if torch.is_tensor(q) else torch.as_tensor(q).clone()
                q.reshape(-1)[j] = self.cfg.door_open_rad
                art.set_qpos(q)

            # The episode ticket, from the batched episode RNG — torch's generator
            # is only seeded when a seed is passed, and sweeps, calibration and
            # replay all key on seeds (water_plants' measured lesson).
            jx = np.stack([self._batched_episode_rng[i].uniform(-1.0, 1.0, size=3)
                           for i in env_idx.tolist()])
            jx_t = torch.as_tensor(jx, dtype=torch.float32)

            # The cup, by MESH BOTTOM on the shelf. `_rest_lift` is origin-to-mesh-
            # bottom, measured once (the mesh does not change between episodes).
            spawn = torch.zeros((b, 3))
            spawn[:, 0] = self._spawn_centre_x[env_idx] \
                + jx_t[:, 0] * self.cfg.spawn_jitter_x
            spawn[:, 1] = self.cfg.spawn_depth + jx_t[:, 1] * self.cfg.spawn_jitter_depth
            spawn[:, 2] = self.cfg.shelf_top_z + self.cfg.spawn_clearance \
                + self._cup_rest_lift
            self.cup.set_pose(Pose.create_from_pq(p=spawn))

            # The place target: under the cup's spawn x, in the open-sky band, on
            # the counter top. A named per-episode point — the memory variant's
            # future hook ("put it back where it was").
            tgt = torch.zeros((b, 3))
            tgt[:, 0] = spawn[:, 0]
            tgt[:, 1] = self.cfg.place_across
            tgt[:, 2] = self._counter_top_z[env_idx]
            self.place_target[env_idx] = tgt

            # The robot start: jittered dock on the open floor.
            base = torch.zeros((b, 3))
            base[:, 0] = self.cfg.start_xy[0] + jx_t[:, 0] * self.cfg.start_jitter_xy
            base[:, 1] = self.cfg.start_xy[1] + jx_t[:, 1] * self.cfg.start_jitter_xy
            base[:, 2] = math.radians(self.cfg.start_yaw_deg) \
                + jx_t[:, 2] * self.cfg.start_jitter_yaw
            qpos = self.agent.robot.get_qpos()
            qpos[env_idx, 0] = base[:, 0]
            qpos[env_idx, 1] = base[:, 1]
            qpos[env_idx, 2] = base[:, 2]
            self.agent.robot.set_qpos(qpos[env_idx])

            self.has_been_lifted[env_idx] = False
            self.succeeded[env_idx] = False

    def _restore_robot(self, env_idx: torch.Tensor):
        """The rest keyframe for every robot uid (scene_builder only restores
        uid == "fetch", and to its own drawn dock — water_plants' measured note)."""
        keyframe = self.agent.keyframes["rest"]
        qpos = torch.as_tensor(np.asarray(keyframe.qpos, dtype=np.float32))
        self.agent.robot.set_qpos(qpos.unsqueeze(0).repeat(len(env_idx), 1))
        self.agent.robot.set_root_pose(sapien.Pose())

    def _after_reconfigure(self, options: dict):
        self.has_been_lifted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.succeeded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Origin-to-mesh-bottom, once: the spawn arithmetic is by mesh bottom.
        mesh = self.cup.get_first_collision_mesh(to_world_frame=False)
        self._cup_rest_lift = float(-np.asarray(mesh.bounds)[0][2])
        return super()._after_reconfigure(options)

    # -------------------------------------------------------------- evaluate --

    def evaluate(self) -> dict:
        cup_pos = self.cup.pose.p

        xy_distance = torch.linalg.norm(
            cup_pos[:, :2] - self.place_target[:, :2], dim=1
        )
        # Mesh bottom against the counter top: the origin rides 0.0575 up the cup,
        # and the template's origin-height check would pass a cup lying on its side.
        height_above = (cup_pos[:, 2] - self._cup_rest_lift) - self.place_target[:, 2]
        is_grasped = self.agent.is_grasping(self.cup)

        on_counter = xy_distance <= self.cfg.place_radius
        low_enough = height_above <= self.cfg.max_height_above_counter
        settled = self.cup.is_static(
            lin_thresh=self.cfg.settle_lin_speed, ang_thresh=self.cfg.settle_ang_speed
        )
        # Still inside the cabinet box (diagnostic, and the future memory variant's
        # phase marker): the cup counts as inside while over the shelf plane.
        in_cabinet = (cup_pos[:, 2] > self.cfg.shelf_top_z - 0.05) \
            & (cup_pos[:, 1] > -0.45)
        door_angle = torch.stack([
            self._door[i][0].get_qpos().reshape(-1)[self._door[i][1]]
            for i in range(self.num_envs)
        ]).to(cup_pos.device)

        # "Door closed" against the named threshold (see door_closed_rad's
        # docstring for why 0.15 and not 0.0). Always computed and always
        # reported — diagnostic like in_cabinet — but it gates success only
        # when the config asks: the all-ones multiplier keeps the
        # require_door_closed=False verdict byte-identical to the pre-field
        # task on every input (v0's published 9/10 is measured territory).
        door_closed = door_angle <= self.cfg.door_closed_rad
        door_ok = door_closed if self.cfg.require_door_closed \
            else torch.ones_like(door_closed)

        self.has_been_lifted = self.has_been_lifted | is_grasped
        raw = on_counter & low_enough & settled & ~is_grasped \
            & self.has_been_lifted & door_ok
        # Latched: the owner's chosen semantics — once placed, a later nudge does
        # not un-succeed the episode.
        self.succeeded = self.succeeded | raw

        return {
            "success": self.succeeded,
            "on_counter": on_counter,
            "low_enough": low_enough,
            "settled": settled,
            "is_grasped": is_grasped,
            "has_been_lifted": self.has_been_lifted,
            "in_cabinet": in_cabinet,
            "door_angle": door_angle,
            "door_closed": door_closed,
            "xy_distance": xy_distance,
            "height_above": height_above,
        }

    # ------------------------------------------------------------------- obs --

    def get_language_instruction(self, **kwargs):
        """The task text a language-conditioned policy is given (the VLA dataset's
        `task`). One phrasing for every episode: the text carries nothing per-episode
        here — the cup, the cabinet and the counter are the same every time."""
        return [INSTRUCTIONS[0]] * self.num_envs

    def _get_obs_extra(self, info: dict) -> dict:
        # Fixed key order; latched progress (has_been_lifted, succeeded) is NEVER
        # emitted — that discipline is what the memory variant will lean on.
        obs = dict(
            tcp_pose=self.agent.tcp_pose.raw_pose,
            base_pose=self.agent.robot.pose.raw_pose,
            is_grasped=info["is_grasped"],
        )
        if self.obs_mode_struct.use_state:
            obs.update(
                cup_pose=self.cup.pose.raw_pose,
                place_target=self.place_target.clone(),
                cup_to_target=self.place_target - self.cup.pose.p,
                door_qpos=info["door_angle"].unsqueeze(-1),
            )
        return obs

    # ----------------------------------------------------------------- state --

    def get_state_dict(self) -> dict:
        state = super().get_state_dict()
        state["has_been_lifted"] = self.has_been_lifted.clone()
        state["succeeded"] = self.succeeded.clone()
        state["place_target"] = self.place_target.clone()
        return state

    def set_state_dict(self, state: dict, env_idx: torch.Tensor = None):
        # Numpy-tolerant (a replayed h5 hands numpy scalars back — that is what
        # `restore_task_tensor` absorbs); ours first, then super, the water_plants
        # order. Full-buffer restore, like every task here: a partial env_idx
        # restore of latched memory has no user yet, and pretending to support it
        # untested would be worse than not.
        for key in TASK_STATE_KEYS:
            setattr(self, key, restore_task_tensor(
                getattr(self, key), state[key], self.device
            ))
        super().set_state_dict(state, env_idx)
