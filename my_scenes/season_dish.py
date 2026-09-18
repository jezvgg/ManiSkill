"""Season the dish — a memory task in a RoboCasa kitchen, driven by Fetch.

The chore: a bowl waits on the counter and two condiments stand at a station
further along it. A recipe marker shows which condiment the dish needs, then
disappears. The robot must fetch **that** condiment, hold it over the bowl,
tip it past an angle, and keep it there.

What makes it a memory task rather than a pick-and-place with a colour:

- the answer is drawn per episode and is visible only while `elapsed_steps <
  cue_steps`. After that the marker is at z = 1000 and the post-cue observation
  is *identical* under swapping the answer;
- the two condiments swap sides per episode, so "always go left" is worth 50 %,
  not 100 %;
- success requires the wrong condiment to be left untouched at its spawn. Without
  that clause a memoryless policy seasons with both and wins every time. That one
  predicate — `distractor_ok` in evaluate() — is what makes the 50 % ceiling true,
  and it is the first thing to check if a blind baseline ever scores higher.

**Not simulated: any substance.** SAPIEN has no granular or fluid physics here, so
"seasoning" is a pose predicate — the condiment held above the bowl and tilted.
That is a proxy and the task says so rather than implying otherwise.

The memoryless baseline is not an argument, it is an experiment: run the same
scripted solution twice, once reading `env.unwrapped.target_is_shaker` and once
always taking the shaker. The task is only validated when the second scores about
half the first at the same failed_motion_plan_rate.

Conventions and the reasons for them: docs/writing-tasks.md and AGENTS.md
"Writing a memory task". Where this file departs from template_task.py, the
comment says why.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.robocasa.scene_builder import RoboCasaSceneBuilder
from mani_skill.utils.structs import Actor, Pose

from utils.mikasa.scenes.robocasa_utils import (
    parking_pose,
    counter_frame,
    dock_pose_for,
    load_objaverse_actor,
    require_get_fixture,
    restore_task_tensor,
    point_in_regions,
    sample_in_regions,
    usable_counter_regions,
)

#: The task text a language-conditioned policy is given (the VLA dataset's `task`).
#: It names the chore, never the answer: which condiment the dish needs is shown by
#: the recipe marker during the cue and is the memory content. No numerals.
INSTRUCTIONS = (
    "Season the dish with the condiment the recipe card showed, then hold it "
    "tipped over the bowl.",
)

# Where the cue marker goes while it must not be seen. The constant, and the
# teleport-in-evaluate idiom, are MIKASA-Robo's: memory_envs/chain_rule_transfer.py:47
# names it HIDDEN_Z, remember_color.py:203 does the lift.
HIDDEN_Z = 1000.0

WALL_LEFT_FACE_X = 0.10
"""World x of the room's left wall face (K76).

`wall_left_room` is centred at x = 0; this is the effective face two independent
FK reconstructions of colliding configurations agreed on (0.02-0.10, taken
conservatively). Only used to keep a drawn dock out of the corner."""


@dataclass
class SeasonDishConfig:
    """Every threshold the task uses, named once.

    A magic number inside evaluate() is how a failed run becomes unattributable —
    you can see `success=False` but not which of six criteria missed.
    """

    horizon: int = int(os.environ.get("MIKASA_HORIZON", "1100"))
    """Episode length. Must equal max_episode_steps in the decorator.

    Overridable with `MIKASA_HORIZON` (default unchanged, so the registered task is
    byte-identical unless it is set). K79g needs the horizon out of the way to test the
    one hypothesis still standing — that slower, more accurate following prevents the
    approach from clipping a standing object — because K79f's attempt at it lost to
    `truncated` 1/180 -> 11/180 rather than to the grasp.

    From the measured demo (K22, T5 2026-08-18, container, CPU, emulated, kitchen 102,
    calibration seeds 11 and 12 — not the eval seeds 0–9): the oracle's verdict landed
    at steps 568 and 670 (cue 40 + grasp with a retry, lift, carry, drive, pre-hover,
    hover, pour, hold); `manip = max(568, 670) − cue 40 = 630`, × 1.5 = 945 → 950 (a
    multiple of 50); + cue 40 + delay 40 = 1030 → **1100** (a multiple of 100). Was 600
    (a guess): seed 12 truncated at 600 inside the hover with the pour still to come."""

    cue_steps: int = 40
    "The marker is visible while elapsed_steps < cue_steps."

    delay_steps: int = 40
    "Blank window after the cue. Earliest possible success is cue_steps + delay_steps."

    hold_steps: int = 15
    "Consecutive steps the pour pose must hold. ~0.75 s at the default control rate."

    # --- placement, in the counter's own frame -----------------------------
    bowl_along: float = 0.85
    station_along: float = 0.46
    """Along the counter from its centre, metres. Both moved on 2026-08-18 (T5) from
    0.22 / -0.28: on the pinned kitchen 102 the counter's centre (x = 1.5) is the
    sink (`sink_main_group`, x 0.79-1.71) — the old station stood over the basin
    and the bottle fell into it every episode (measured: origin -12.7 cm, its top
    2.9 cm below the counter, on 3.0.1 CPU and in the b22 container alike; the
    bowl sat on the sink's rim and rocked). Now both are on the free surface right
    of the sink, over the dishwasher's counter (x 1.71-2.75): station at x = 1.96
    (condiments 1.84 / 2.08), bowl at 2.35 — where the inherited cup task keeps
    its bowl (2.333).

    0.40 -> 0.46 on 2026-08-23 (K55), for the same reason as the first move and
    measured the same way. At 0.40 the left condiment stood at x = 1.78, seven
    centimetres from the sink block's edge, in a pocket the arm could not plan
    into: seed 7 refused every wrist yaw at every grip height, nine of eleven with
    a bare `IK Failed` and the rest `wrist_flex_link <-> counter_main`. It was the
    last seed of ten to fail and no planner-side ladder reached it. Six centimetres
    right puts the clearance at thirteen and changes nothing the task measures —
    same two condiments, same cue, same one bit."""
    across: float = -0.10
    "Toward the robot side of the counter. Negative is 'in front'."
    station_spacing: float = 0.24
    "Separation of the two condiments along the counter."
    spawn_clearance: float = 0.02
    "Spawn height above the work surface; physics settles the rest."
    spawn_jitter_xy: float = 0.02

    # --- per-episode placement (K74) --------------------------------------------
    randomize_placements: bool = True
    """Draw the bowl and the two seasoning stations somewhere on the counter each
    episode, instead of at fixed `bowl_along` / `station_along` offsets.

    The free area comes from jezv's placement work (`base_robocasa.py`, `bba1117`),
    ported to `robocasa_utils.usable_counter_regions`: the counter top inset from its
    edges minus every fixture standing on it. Measured on kitchen 102 that is 0.62 m2
    in four rects, split by the sink at x 0.76-1.74.

    Set False to recover the fixed layout every number before K74 was measured on."""

    reach_band: tuple[float, float] = tuple(
        float(x) for x in os.environ.get("MIKASA_REACH_BAND", "0.74,0.88").split(","))
    """Metres from the robot's dock to a drawn object, min and max.

    Not decoration — the binding constraint. The counter's usable band runs from
    y = -0.57 (front edge) to -0.08 (back), and the dock stands at y = -1.25: an object
    at the back edge is a **1.17 m** reach, far outside what the arm plans. The fixed
    layout every pre-K74 number was measured on sits at 0.833 m with a 4 cm spread
    (K71), so this band is that reach widened, not a new regime."""

    min_object_gap: float = 0.04
    """Clear space required between two objects' **surfaces**, metres (K76).

    Not centre-to-centre any more. The test is
    `dist >= r_a + r_b + contact_band + min_object_gap`, with radii measured off the
    collision meshes (`_obj_radius_np`: bowl 0.1299, shaker 0.0262, bottle 0.0235).

    A scalar centre-to-centre gap was the wrong shape and the numbers say so: at 0.16 m
    it bought 4 mm over bare bowl+shaker non-overlap (0.1561 m) and nothing at all
    against the contact band. The equivalent scalar would have to be >= 0.26 m."""

    dock_wall_clear: float = float(os.environ.get("MIKASA_DOCK_WALL_CLEAR", "0.46"))
    """Least clear floor between a drawn object's dock and the room's end wall, metres.
    **0.46 since 2026-09-06 (W30), was 0.55.** The 9 cm margin K76 added below was chosen,
    not measured; three geometry-selected probes (200 episodes, `runs/2026-09-06-wall`)
    put the wall's real threshold at dock x 0.53 — every failure below it is K76's
    `drive to bowl dock (after the tuck)` against `wall_left_room`, the last at 0.524,
    and 0.53–0.65 is 52/52. 0.46 keeps 3.6 cm over that and measured 199/200 against
    197/200 for 0.55 on 3000–3199. On kitchen 102 the knob decides whether the segment
    left of the sink (x 0.33–0.76) is a spawn region: at 0.55 the bowl had an 11 cm
    pocket there and the station pair never fit (0 of 400 draws); at 0.46 the pocket is
    20 cm and the pair lands there in ~20 % of draws. Every SeasonDish number before W30
    was measured at 0.55. Sweepable through `MIKASA_DOCK_WALL_CLEAR`.

    `_draw_placement` constrains reach, separation, free space and station pairing — and
    said nothing about walls, so the bowl could be drawn at the counter's far end with
    its dock in the corner. Measured on two seeds: `wall_left_room` sits at x = 0 with
    its effective face at x ~ 0.02-0.10, and FK on the logged qpos puts a **tucked**
    arm's leading link (the elbow, not the gripper) 0.461 m ahead of `base_link`. Seed 2
    docked at x = 0.504, leaving ~0.40 m, and `move_base_forward` refused with
    `0.000 of the twist left` — the goal configuration itself in collision, which no
    path planner can rescue. Seed 26 was the same 8 cm short.

    0.461 m plus ~9 cm of margin. It costs the leftmost ~20 cm of a 2.5 m counter."""
    station_dock_wall_clear: float = float(os.environ.get("MIKASA_STATION_DOCK_WALL_CLEAR", "0.36"))
    """The same clearance for the STATION dock — where the robot starts — metres.
    Smaller than `dock_wall_clear`, because the wall binds on the arrival and not on the
    wall: the bowl dock is reached by a drive along the counter, nose-first, with the tucked
    arm's elbow leading (K76), while at the station dock the robot is placed facing the
    counter and its only drive from there goes to the bowl, which cannot also be on the
    left (the segment is 43 cm; a pair and a bowl need 52). Measured 2026-09-06 (W30):
    station docks at x 0.46–0.65 are 70/70. 0.36 puts the base's edge ~8 cm off the wall
    face; on kitchen 102 the region's own edge (pair midpoint ≥ 0.46) binds first, so the
    pair's window left of the sink is the full 17 cm instead of 7. Sweepable."""


    contact_band: float = 0.04
    """Twice PhysX's `contact_offset` (0.02), metres.

    Contacts are generated while surfaces are within `contact_offset` of each other, so
    two objects that merely fail to overlap can still be pushed apart on the first step.
    Seed 79 spawned the shaker 31 mm clear of the bowl — inside this band — and PhysX
    ejected it at 7.3 m/s, 6.5 m across the kitchen, before the robot moved."""

    placement_draw_tries: int = 2000
    """Rejection-sampling attempts before `_draw_placement` gives up and returns the
    fixed layout (`fell_back=True`).

    Was 200, which fell back on **15 of 180 seeds (8.3%)** — measured by resetting every
    seed and reading `_placement_fell_back`. That is a benchmark-integrity bug, not a
    tuning question: one episode in twelve was silently running the *pre-randomisation*
    layout, and because the fallback is a single fixed arrangement those seeds were all
    byte-identical to each other (80, 135, 169 and 176 draw the same 0.8337 m stations).
    Fallback seeds also fail more often than drawn ones (~7-12% against 0-4%), so they
    were over-represented among the failures that made the "failing set" look unstable.

    Three objects must land in the same 14 cm reach band, clear each other radius-aware,
    keep both docks off the end wall and stay inside the usable region, which works out
    at roughly 1.3% acceptance per draw — hence 0.987**200 = 8% exhaustion. At 2000 the
    same arithmetic gives e**-25, and measured over the same 180 seeds it is **0 of 180**.
    Costs only rejected numpy draws at reset (tens of ms worst case), and nothing at all
    on the draws that already succeeded, since the loop returns on first acceptance."""
    """Rejection-sampling attempts before falling back to the fixed layout.

    A fallback is not silent: `_placement_fell_back` goes into `info` so a sweep can
    show how often the draw gave up rather than quietly measuring the old task."""
    placement_draw: str = os.environ.get("MIKASA_PLACEMENT_DRAW", "joint")
    """How the three objects are drawn (W30e, 2026-09-07). `joint` — one draw proposes the
    pair's midpoint AND the bowl and is accepted only when every constraint holds for all
    three: uniform over feasible LAYOUTS, which on kitchen 102 puts the pair left of the
    sink in 36 % of episodes (the pair-right layouts lose more draws to the bowl's overlap)
    against 20 % by window length. `pair-first` — the pair is drawn and accepted on its own
    constraints, then the bowl is drawn given the pair: uniform over the pair's POSITION,
    and the bowl over what is left. Which uniformity the task wants is the owner's call;
    `joint` stays the default until it is made."""

    # --- the robot's docks -----------------------------------------------------
    dock_toward: float = float(os.environ.get("MIKASA_DOCK_TOWARD", "0.15"))
    bowl_dock_toward: float | None = (
        None if os.environ.get("MIKASA_BOWL_DOCK_TOWARD", "") == ""
        else float(os.environ["MIKASA_BOWL_DOCK_TOWARD"]))
    """The BOWL dock's own standoff toward the counter, metres; None = `dock_toward`.
    The owner's diagnosis (2026-09-05) was about the grasp and the lift at the station:
    too close to the counter, the arm works folded, rotates a lot and knocks the
    condiment off. Standing 0.15 further back there (dock_toward 0.0) cured both
    fly-offs on 300-499 (363, 377) — and cost 452 at the bowl: from the further bowl
    dock the hover stopped 14.8 cm short of the bowl (`over_bowl=False` on all eight
    pour candidates). The two docks answer different reaches; this lets the bowl dock
    keep its own. Same bound as `dock_toward` in `validate`."""
    """How much closer than RoboCasa's standoff (0.8 m from the counter's near edge for
    ds_fetch) the two docks stand, in metres toward the counter. Two docks are the
    task's own (T5, 2026-08-18): the **station dock** — the counter dock slid along
    the counter to `station_along`, where the robot starts (`_robot_start`) and grasps
    — and the **bowl dock** at `bowl_along`, where it pours. Before, the robot started
    at whatever dock the scene builder drew (`robot_poses`), often at another
    fixture. `dock_pose_for(..., offset=(0.0, dock_toward))`; `validate` caps it at
    0.4 (the Fetch base's footprint must stay clear of the counter)."""

    # --- the pour predicate ------------------------------------------------
    pour_xy_radius: float = 0.10
    pour_min_clearance: float = 0.05
    pour_max_clearance: float = 0.30
    pour_tilt_deg: float = float(os.environ.get("MIKASA_POUR_TILT_DEG", "155.0"))
    """The tilt the pour must reach, degrees of the object's axis from world +Z (180 = upside
    down). 55 until 2026-09-08; the owner raised it to 155 "for realism" — a shaker pours
    inverted, not tipped. The oracle's rungs (`season_dish_planner.POUR_TILT_DEG`) follow it."""
    pour_axis_body: tuple = (0.0, 0.0, 1.0)
    "Body axis treated as 'up' when upright. Settle it with diagnose_task --task season."

    grasp_min_force: float = 0.5
    distractor_move_tol: float = 0.10
    "How far the wrong condiment may drift before the episode is void."

    marker_height: float = 0.14
    marker_radius: float = 0.025

    require_return: bool = False
    "v2 switch: also require the condiment to be put back. Not implemented in v1."

    def validate(self) -> None:
        """Every constraint the thresholds must satisfy, checked once at construction
        (`SeasonDishTask.__init__`, as `BurnerConfig.validate` is, K37).

        Example:
            >>> SeasonDishConfig().validate()
            >>> SeasonDishConfig(dock_toward=0.5).validate()  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
            AssertionError
        """
        assert 0.0 <= self.dock_toward <= 0.4, self.dock_toward
        if self.bowl_dock_toward is not None:
            assert 0.0 <= self.bowl_dock_toward <= 0.4, self.bowl_dock_toward
        assert self.pour_min_clearance < self.pour_max_clearance, (
            self.pour_min_clearance, self.pour_max_clearance)
        assert 0.0 <= self.pour_xy_radius, self.pour_xy_radius
        assert 0.0 < self.pour_tilt_deg < 180.0, self.pour_tilt_deg
        assert self.cue_steps > 0 and self.delay_steps >= 0 and self.hold_steps > 0, (
            self.cue_steps, self.delay_steps, self.hold_steps)
        assert self.hold_steps <= self.horizon - self.cue_steps - self.delay_steps, (
            f"hold_steps={self.hold_steps} cannot fit after the cue ({self.cue_steps}) and the "
            f"delay ({self.delay_steps}) in a {self.horizon}-step episode"
        )
        assert self.station_spacing > 0 and self.spawn_jitter_xy >= 0
        assert 0.0 < self.reach_band[0] < self.reach_band[1], self.reach_band
        assert self.min_object_gap >= 0.0 and self.contact_band >= 0.0
        assert self.placement_draw_tries > 0
        assert self.dock_wall_clear >= 0.0 and self.station_dock_wall_clear >= 0.0
        assert self.placement_draw in ("joint", "pair-first"), self.placement_draw
        assert self.distractor_move_tol > 0 and self.grasp_min_force > 0
        assert self.marker_radius > 0 and self.marker_height > 0


@register_env(
    "MikasaSeasonDish-v0",
    max_episode_steps=SeasonDishConfig.horizon,
    asset_download_ids=["RoboCasa"],
)
class SeasonDishTask(BaseEnv):
    """Fetch the condiment the recipe asked for, and season the dish with it."""

    # ds_fetch is what the planner in this repo actually drives. "none" is
    # deliberately absent: it sets self.agent = None and every method below
    # dereferences self.agent.
    SUPPORTED_ROBOTS = ["mikasa_ds_fetch", "fetch"]

    # No compute_dense_reward in v1, so "dense" must not appear here — listing a
    # mode you did not implement raises on the first step, and implementing one
    # you did not list means it never runs.
    SUPPORTED_REWARD_MODES = ["sparse", "none"]

    cfg = SeasonDishConfig()

    bowl: Actor
    shaker: Actor
    condiment_bottle: Actor

    def __init__(self, *args, robot_uids="mikasa_ds_fetch", scene_idx: int | None = 0, **kwargs):
        # scene_idx defaults to 0 (K43, 2026-08-18): the pinned kitchen 102 (layout
        # one_wall_small, style industrial) — the same default as the burner and the
        # station tasks, so the three memory tasks are one-kitchen benchmarks by
        # default and `--scene-idx` remains the diversity knob (None = a random
        # kitchen per env). Set before super().__init__: that call runs a full
        # reset(reconfigure=True), which reaches _load_scene, which reads self.scene_idx.
        self.scene_idx = scene_idx
        self.cfg.validate()
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # ---------------------------------------------------------------- sensors --

    def _focus_point(self) -> np.ndarray:
        """Where the cameras look: the counter top halfway between the station and the
        bowl (`along = (station_along + bowl_along) / 2`), so the objects — 0.63–0.68 m
        along the counter from its centre since T5 moved them off the sink — sit in
        the middle of both frames instead of at the right edge (the frame reader on
        the 2026-08-18 clip: bowl cut by the right edge on `base_camera`, marker and
        bowl behind the robot on `render_camera`). Decoration, not predicate: no
        threshold reads a camera."""
        # With `randomize_placements` the objects are drawn anywhere on the counter
        # (along -1.17 to +1.17 on kitchen 102), so a focus at the fixed layout's
        # midpoint leaves half the draws outside the frame — the user's report:
        # "I can't see, camera doesn't capture the full room". Aim at the counter's
        # own centre instead, which is `along = 0` by construction of `counter_frame`.
        along = (
            0.0 if self.cfg.randomize_placements
            else 0.5 * (self.cfg.station_along + self.cfg.bowl_along)
        )
        return self._counter_top_np[0] + self._along_np[0] * along

    def _view_point(self, distance: float, height: float, along: float = 0.0) -> np.ndarray:
        """A camera position `distance` out from the counter, on the working side,
        `along` metres along the counter from the focus point.

        The sign is taken from `cfg.across` rather than written separately, and that
        is the whole point of this helper. `across` is a unit vector whose direction
        depends on the counter's yaw, so "+across" is not reliably "in front" —
        in the pinned kitchen it points at +y while the objects, and the robot, are
        at -y. A camera placed at +across looks at the counter from behind the wall
        and renders a view with nothing in it.

        Found by running the probe: the cue marker moved to HIDDEN_Z on schedule and
        the recorded video did not change by a single pixel, because neither camera
        could see it.
        """
        side = math.copysign(1.0, self.cfg.across) if self.cfg.across else -1.0
        return (
            self._focus_point()
            + self._across_np[0] * (side * distance)
            + self._along_np[0] * along
            + np.array([0.0, 0.0, height], dtype=np.float32)
        )

    @property
    def _default_sensor_configs(self):
        # Read during _reconfigure, after _load_scene but before any
        # _initialize_episode — so it may use geometry cached in _load_scene, and
        # must not use anything drawn per episode.
        target = self._focus_point() + np.array([0, 0, 0.05])
        pose = sapien_utils.look_at(
            eye=self._view_point(1.6, 1.1) if self.cfg.randomize_placements
            else self._view_point(0.9, 0.55),
            target=target,
        )
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        """The clip's viewpoint. Decoration — no threshold reads a camera.

        The eye stands off to the sink's side of the focus: an oblique view along
        the counter, so the robot — which works between the docks and the counter,
        0.8 m in front of the objects — does not stand between the camera and the
        bowl or the marker (the frame reader's finding on the first clip: bowl and
        marker behind the robot's shoulder).

        Raised 2026-08-19 from (1.3 m out, 1.4 m up, −1.4 m along) to (1.5, 1.9,
        −1.2), look-at lifted 0.15 m: at the old height the robot's torso filled the
        right third of the frame from the moment it docked, and the spice bottles —
        the thing a viewer has to tell apart — sat behind it. From higher up the
        whole counter is visible past the robot. Chosen by rendering the recorded
        episode from four eyes at four moments, not by arithmetic.

        512 is the *recording* default (`RecordEpisode` holds a whole episode of
        frames in host RAM; 931 steps × 512² = 0.73 GB, × 2048² = 11.7 GB). Clips
        are drawn one at a time by `utils.mikasa.replay --render-size`, which is
        where a bigger number belongs.
        """
        # The (1.5, 1.9, -1.2) eye below was chosen for the fixed layout, where every
        # object sat within 20 cm of one point. A drawn layout spans the whole 2.5 m
        # counter, so the randomized eye is **straight on** (along=0) and squared to the
        # counter's centre: the oblique offset that kept the robot clear of one fixed
        # spot just swings the frame off one end when the objects can be anywhere.
        #
        # Chosen by reading frames, then by geometry (K75). Two attempts were measured
        # and rejected by a frame reader before this one:
        #
        #   (2.9, 2.5, along=-0.5) — half the pixels blank brick and bare floor, and the
        #     seasonings only ~20 px wide in 1600.
        #   (2.6, 1.65, along=0)   — tighter, but it put the camera on the robot's
        #     shoulder: the cue sphere fell to a 19x7 px sliver on one seed and to
        #     **zero pixels** on the other, and the bottle it cues vanished entirely.
        #
        # The camera and the robot are both in front of the counter, and the robot docks
        # *directly* in front of whichever objects it is working on with its rest pose
        # parking the arm vertically across them. So no eye on that side wins: a third
        # attempt at (2.2, 2.6) rendered the cue sphere perfectly on one seed (22x22 px,
        # clean) and at **zero pixels** on the other, purely because of where the robot
        # happened to stand. Clearing it is a height problem — a ray from (out=D, up=h)
        # passes the robot at `0.92 + (0.83/D)·h` against its ~1.55 m — and the cheapest
        # height is bought by coming *in* rather than going out: (1.2, 3.0) crosses at
        # 3.00 m, nearly twice the clearance of (2.2, 2.6)'s 1.90 m, from a closer eye
        # (3.23 m vs 3.41) that therefore also frames the counter slightly larger.
        # Near-overhead costs the 3D read of the scene and buys the one thing the clip
        # exists for: the cue, and which container it marks, visible on every draw.
        #
        # (1.2, 3.0) got the cue clean on both extremes — 389 and 416 px, full discs, on
        # seeds docking at opposite ends — but the occlusion only *moved*: the gripper,
        # which at rest hovers over the counter exactly where the robot docks, then ate
        # ~85% of the bottle the cue points at. The robot docks at the stations, so no
        # eye behind it escapes that. 0.7 m out puts the lens **in front of** the robot
        # (it stands 0.83 m out), so the base is behind the camera entirely and only the
        # arm reaching for the counter can enter frame.
        #
        # The cooktop and cabinets beyond the marble do run off the right edge. That is
        # left alone deliberately: objects are only ever drawn on the marble, which is
        # fully in frame, and pulling back to include the range would shrink the
        # seasonings — the one thing a viewer has to tell apart — for no task content.
        eye = (
            self._view_point(0.7, 3.2, along=0.0) if self.cfg.randomize_placements
            else self._view_point(1.5, 1.9, along=-1.2)
        )
        pose = sapien_utils.look_at(
            eye=eye,
            target=self._focus_point() + np.array([0.0, 0.0, 0.15], dtype=np.float32),
        )
        return CameraConfig("render_camera", pose, 512, 512, 1.05, 0.01, 100)

    # ------------------------------------------------------------------- load --

    def _load_agent(self, options: dict):
        # Spawn clear of the kitchen so gpu_init() is not resolving an
        # interpenetration on frame one. The scene builder overwrites this pose
        # with the RoboCasa standoff pose during build() anyway.
        super()._load_agent(options, parking_pose(self))

    def _load_scene(self, options: dict):
        """Geometry and initial poses only.

        There is no `set_pose` anywhere in this method, and that is not style:
        `_reconfigure` runs `scene._setup()` immediately after, which re-applies
        `initial_pose` to every non-static actor and silently discards anything
        positioned with `set_pose`. The lint checks for it by walking this
        method's AST (tests/test_task_conventions.py:139).
        """
        super()._load_scene(options)

        self.scene_builder = RoboCasaSceneBuilder(self)
        if self.scene_idx is None:
            self.scene_builder.build()
        else:
            self.scene_builder.build([self.scene_idx] * self.num_envs)

        self._fix_ds_fetch_collision_bits()

        # One get_fixture call per env, cached. Calling it repeatedly is not free:
        # a substring match draws from self.env._episode_rng (scene_builder.py:656),
        # so it can return a different counter each time *and* it shifts the shared
        # episode stream. Per env, because each sub-scene draws its own layout —
        # deriving one position from scene_data[0] puts objects inside walls in
        # envs 1..N-1, silently (review §5).
        tops, alongs, acrosses = [], [], []
        station_docks, bowl_docks, robot_starts = [], [], []
        dock_ps, dock_yaws, bowl_dock_ps = [], [], []
        for i in range(self.num_envs):
            fixtures = self.scene_builder.scene_data[i]["fixtures"]
            counter = require_get_fixture(
                self.scene_builder, fixtures, "counter_main_main_group", scene_idx=self.scene_idx,
            )
            top, along, across = counter_frame(counter)
            tops.append(top)
            alongs.append(along)
            acrosses.append(across)
            # The robot's two docks (T5): the counter's front-facing standoff pose,
            # `dock_toward` closer, slid along the counter to the station (where it
            # starts and grasps) and to the bowl (where it pours) — as burner.py does
            # for its cup and stove docks. Each is (x, y, yaw); the start pose is the
            # station dock as a 7-vector for `_restore_robot`.
            dock_p, dock_yaw = dock_pose_for(
                self.scene_builder, fixtures, "counter_main_main_group",
                offset=(0.0, self.cfg.dock_toward),
            )
            sta = dock_p + along * self.cfg.station_along
            bowl_toward = (self.cfg.dock_toward if self.cfg.bowl_dock_toward is None
                           else self.cfg.bowl_dock_toward)
            bowl_dock_p = dock_p
            if bowl_toward != self.cfg.dock_toward:
                bowl_dock_p, _ = dock_pose_for(
                    self.scene_builder, fixtures, "counter_main_main_group",
                    offset=(0.0, bowl_toward),
                )
            bwl = np.asarray(bowl_dock_p, dtype=np.float64) + along * self.cfg.bowl_along
            dock_ps.append(np.asarray(dock_p, dtype=np.float64))
            bowl_dock_ps.append(np.asarray(bowl_dock_p, dtype=np.float64))
            dock_yaws.append(float(dock_yaw))
            station_docks.append(np.array([sta[0], sta[1], dock_yaw], dtype=np.float32))
            bowl_docks.append(np.array([bwl[0], bwl[1], dock_yaw], dtype=np.float32))
            robot_starts.append(np.array(
                [sta[0], sta[1], 0.0, math.cos(dock_yaw / 2), 0.0, 0.0, math.sin(dock_yaw / 2)],
                dtype=np.float32,
            ))

        # Per-env inputs for the placement draw (K74). Per env, not `scene_data[0]`
        # for all of them: review §5 is exactly this bug, and jezv's own version reads
        # env 0 only.
        self._usable_regions = []
        self._counter_top_z = []
        for i in range(self.num_envs):
            cfgs = self.scene_builder.scene_data[i].get("fixture_cfgs") or []
            counter_model = None
            for c in cfgs:
                if str(c.get("name", "")).startswith("counter_main_main_group"):
                    counter_model = c.get("model")
                    break
            if counter_model is None or getattr(counter_model, "pos", None) is None:
                self._usable_regions.append([])
                self._counter_top_z.append(float(tops[i][2]))
                continue
            c_pos = np.asarray(counter_model.pos, dtype=np.float64)
            c_size = np.asarray(counter_model.size, dtype=np.float64)
            self._usable_regions.append(
                usable_counter_regions(cfgs, c_pos, c_size)
            )
            self._counter_top_z.append(float(c_pos[2] + c_size[2] / 2.0))

        self._counter_top_np = np.stack(tops)
        self._along_np = np.stack(alongs)
        self._across_np = np.stack(acrosses)
        self._station_dock_np = np.stack(station_docks).astype(np.float32)
        self._bowl_dock_np = np.stack(bowl_docks).astype(np.float32)
        self._robot_start_np = np.stack(robot_starts).astype(np.float32)
        # The fixed layout, kept whole: `randomize_placements=False` restores it, and a
        # draw that cannot satisfy its constraints falls back to it (K74).
        self._fixed_station_dock_np = self._station_dock_np.copy()
        self._fixed_bowl_dock_np = self._bowl_dock_np.copy()
        self._fixed_robot_start_np = self._robot_start_np.copy()
        self._dock_p_np = np.stack(dock_ps)
        self._bowl_dock_p_np = np.stack(bowl_dock_ps)   # the bowl dock's own standoff
        self._dock_yaw_np = np.asarray(dock_yaws, dtype=np.float64)

        bowl_home = self._counter_point(self.cfg.bowl_along)
        station_mid = self._counter_point(self.cfg.station_along)
        half = self._along_np * (self.cfg.station_spacing / 2.0)
        station_left = station_mid + half
        station_right = station_mid - half

        # Objects come from the registry, at the scale RoboCasa declares — see
        # robocasa_utils.load_objaverse_actor for why both of those matter.
        self.bowl = load_objaverse_actor(
            self, "bowl", "bowl", sapien.Pose(p=bowl_home[0]), index=0
        )
        self.shaker = load_objaverse_actor(
            self, "shaker", "shaker", sapien.Pose(p=station_left[0]), index=0
        )
        self.condiment_bottle = load_objaverse_actor(
            self, "condiment_bottle", "condiment_bottle", sapien.Pose(p=station_right[0]), index=0
        )
        # Diagnostic only (K98): scale both condiments' density by MIKASA_CONDIMENT_MASS_X.
        # The asset default weighs 15.8 g and tips at 0.0038 N*s — below the gentlest
        # contact the oracle has ever been measured making (K94). Mass and inertia scale
        # together, so this is a density change, not an unphysical one. 1.0 (default) is
        # byte-identical: the branch is not taken.
        _mx = float(os.environ.get("MIKASA_CONDIMENT_MASS_X", "1.0"))
        if _mx != 1.0:
            for _a in (self.shaker, self.condiment_bottle):
                for _b in _a._bodies:
                    _b.mass = float(_b.mass) * _mx
                    _b.inertia = _b.inertia * _mx
        # Each object's rest lift: how far its origin sits above the lowest point of
        # its collision mesh (yaw-invariant, so read once at load). Spawning at
        # `top + spawn_clearance` by *origin* put the mesh bottoms 1.7 / 2.7 / 5.9 cm
        # (bowl / shaker / bottle) inside the counter slab; PhysX pushed the first two
        # up and the bottle *through* — origin 12.7 cm down, its top 2.9 cm below the
        # counter, inside the cabinet, unreachable — measured on 3.0.1 CPU and in the
        # b22 container alike (journal 2026-08-18 T5). `_initialize_episode` adds
        # these so every mesh bottom starts `spawn_clearance` above the surface.
        #
        # Read in the actor's **own** frame, not the world's. The quantity wanted is
        # origin-to-mesh-bottom, which is a property of the mesh and needs no pose at
        # all; taking it as `pose.p[2] - world_bounds[0][2]` did need one, and
        # `Actor.pose` under GPU sim reads `cuda_rigid_body_data`, which does not exist
        # while `_load_scene` is still running. That is the whole of
        # `IndexError: basic_string::at` on `--sim-backend gpu` (K55). The two forms
        # agree to 2e-8 on CPU for all three actors — the pose here is yaw-only, and a
        # yaw does not move a z-extent.
        # True xy collision radius per object, for the separation test (K76). A scalar
        # `min_object_gap` is centre-to-centre and knows nothing about the fact that the
        # bowl is 26 cm across and a shaker 5 cm: measured, bowl r = 0.1299 m and
        # shaker r = 0.0262 m, so bare non-overlap already needs 0.1561 m and the 0.16 m
        # threshold bought 4 mm. PhysX generates contacts inside `contact_offset` (0.02)
        # of each surface, i.e. a 40 mm band, so the real requirement is ~0.196 m.
        self._obj_radius_np = np.array(
            [
                float(np.linalg.norm(
                    a.get_first_collision_mesh(to_world_frame=False).vertices[:, :2], axis=1).max())
                for a in (self.bowl, self.shaker, self.condiment_bottle)
            ],
            dtype=np.float32,
        )
        self._rest_lift_np = np.array(
            [
                float(-a.get_first_collision_mesh(to_world_frame=False).bounds[0][2])
                for a in (self.bowl, self.shaker, self.condiment_bottle)
            ],
            dtype=np.float32,
        )

        # One marker. `recipe_marker` is phase-teleported to HIDDEN_Z in evaluate(),
        # which is the only hiding mechanism that can be a function of elapsed_steps
        # and so the only one that can express "visible until step 40".
        #
        # A second, magenta `recipe_ghost` used to sit above it in `self._hidden_objects`
        # — hidden from every sensor capture (sapien_env.py:603) but left in the human
        # render (:1375), so a recording showed what the answer had been while the policy
        # could never see it. Removed on request: it reads as a second cue that never goes
        # away, which is exactly what a viewer should not see in a memory task's clip. It
        # was a debug annotation only — invisible to every sensor, absent from `evaluate`
        # and from `_get_obs_extra` — so nothing about the task, the predicate or the
        # policy's observations changes with it gone.
        #
        # add_collision=False (hide_visual asserts no collision shapes,
        # structs/actor.py:184) and kinematic, not static (the pose setter asserts
        # non-static under GPU sim, :367).
        marker_pose = sapien.Pose(p=station_left[0] + np.array([0, 0, self.cfg.marker_height]))
        self.recipe_marker = actors.build_sphere(
            self.scene,
            radius=self.cfg.marker_radius,
            color=[0.95, 0.85, 0.1, 1.0],
            name="recipe_marker",
            body_type="kinematic",
            add_collision=False,
            initial_pose=marker_pose,
        )

        # Cache the homes as device tensors once. Rebuilding numpy every step costs
        # a host-to-device copy, and float64 numpy silently promotes the whole
        # comparison to float64.
        t = lambda a: torch.tensor(a, dtype=torch.float32, device=self.device)  # noqa: E731
        # The fixed layout in numpy, for `_draw_placement`'s fallback (K74).
        self._bowl_home_fixed = np.asarray(bowl_home, dtype=np.float64)
        self._station_left_fixed = np.asarray(station_left, dtype=np.float64)
        self._station_right_fixed = np.asarray(station_right, dtype=np.float64)
        self._bowl_home = t(bowl_home)
        self._station_left = t(station_left)
        self._station_right = t(station_right)
        self._robot_start = t(self._robot_start_np)
        self._rest_lift = t(self._rest_lift_np)  # (3,): bowl, shaker, bottle

    def _counter_point(self, along_offset: float) -> np.ndarray:
        """A point on the work surface, `along_offset` along the counter. (N, 3)."""
        return (
            self._counter_top_np
            + self._along_np * along_offset
            + self._across_np * self.cfg.across
            + np.array([0.0, 0.0, self.cfg.spawn_clearance], dtype=np.float32)
        )

    def _fix_ds_fetch_collision_bits(self):
        """Restore the wheel/base collision exemption the scene builder skips.

        scene_builder.py:490 guards it with `robot_uids == "fetch"`, so a Fetch
        subclass with a different uid silently keeps colliding with the kitchen
        floor and walls. Fetch._after_init already sets bits 30 and 31
        (fetch.py:352-359); what is lost is the builder's own 25-29.
        """
        if self.robot_uids == "fetch" or self.agent is None:
            return
        for link in self.agent.robot.links:
            for body in link._bodies:
                for shape in body.get_collision_shapes():
                    groups = shape.get_collision_groups()
                    for bit in range(25, 30):
                        groups[2] |= 1 << bit
                    shape.set_collision_groups(groups)

    # ------------------------------------------------------------- initialize --

    def _after_reconfigure(self, options: dict):
        # Task buffers are allocated here, once, at full num_envs width. Allocating
        # them lazily in _initialize_episode (as MyRoboCasa_TakeItBack does, with a
        # hasattr guard) means get_state_dict can be called before they exist —
        # and it is, at sapien_env.py:332 and by RecordEpisode on every reset.
        n = self.num_envs
        dev = self.device
        self.target_is_shaker = torch.zeros(n, dtype=torch.bool, device=dev)
        self.station_left_is_shaker = torch.zeros(n, dtype=torch.bool, device=dev)
        self.pour_hold = torch.zeros(n, dtype=torch.int32, device=dev)
        self._pour_last_eval_step = torch.full((n,), -1, dtype=torch.int32, device=dev)
        self._distractor_home = torch.zeros((n, 3), dtype=torch.float32, device=dev)
        self._marker_home = torch.zeros((n, 3), dtype=torch.float32, device=dev)
        # The bowl where it comes to rest, captured at `cue_steps` (K73).
        self._bowl_settled = torch.zeros((n, 2), dtype=torch.float32, device=dev)
        # Did this episode's placement draw give up and use the fixed layout? (K74)
        self._placement_fell_back = torch.zeros(n, dtype=torch.bool, device=dev)
        return super()._after_reconfigure(options)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Runs on every reset, for the envs in env_idx only."""
        with torch.device(self.device):
            b = len(env_idx)

            # Restores fixtures and, for robot_uids == "fetch", the robot. The lint
            # walks this method's own source for the call (test_task_conventions.py:133),
            # so it must appear here literally and not behind a helper.
            self.scene_builder.initialize(env_idx)

            # The answer comes from the *episode* RNG, not torch. sapien_env.py:948-953
            # only seeds the torch generator when a seed was passed, so torch.rand is
            # not reproducible from a seed alone — fine for jitter, wrong for the one
            # bit the whole benchmark rests on.
            rng = self._batched_episode_rng[env_idx]
            target_bits = torch.as_tensor(rng.randint(0, 2), dtype=torch.bool)
            side_bits = torch.as_tensor(rng.randint(0, 2), dtype=torch.bool)
            self.target_is_shaker[env_idx] = target_bits
            self.station_left_is_shaker[env_idx] = side_bits

            if self.cfg.randomize_placements:
                # Draw per env, and rebuild that env's two docks from what was drawn:
                # the docks are the counter's standoff slid along to the objects, so a
                # moved object with a stale dock is simply out of reach (K74).
                fell = []
                # `self._batched_episode_rng` draws one value per env at a time, so its
                # `uniform` returns arrays; `_draw_placement` works on one env. Give it a
                # plain Generator seeded from the batched draw, which keeps the placement
                # reproducible per (seed, env) without reshaping every call site.
                seeds = np.asarray(rng.randint(0, 2**31 - 1)).reshape(-1)
                for j, e in enumerate(env_idx.tolist()):
                    sub = np.random.default_rng(int(seeds[j % len(seeds)]) + int(e))
                    bowl_j, left_j, right_j, fb = self._draw_placement(e, sub)
                    fell.append(fb)
                    self._bowl_home[e] = torch.as_tensor(
                        bowl_j, dtype=self._bowl_home.dtype, device=self._bowl_home.device)
                    self._station_left[e] = torch.as_tensor(
                        left_j, dtype=self._station_left.dtype, device=self._station_left.device)
                    self._station_right[e] = torch.as_tensor(
                        right_j, dtype=self._station_right.dtype, device=self._station_right.device)
                    yaw = float(self._dock_yaw_np[e])
                    # A dock is the counter's standoff **slid along the counter** to the
                    # object — `dock_p + along * offset`, as `_load_scene` builds it and
                    # as `oracle_common.dock_for_target` does. Using the objects' own
                    # midpoint instead puts the base on the counter top: measured, the
                    # robot spawned at y = -0.50 (on the surface) rather than -1.25 (in
                    # front of it), and every episode died at the grasp.
                    dp = self._dock_p_np[e][:2]
                    av = self._along_np[e][:2].astype(np.float64)
                    av = av / max(float(np.linalg.norm(av)), 1e-9)
                    mid = 0.5 * (left_j + right_j)
                    sta = dp + av * float(np.dot(mid[:2] - dp, av))
                    # The bowl dock slides along from ITS standoff (`bowl_dock_toward`).
                    dpb = self._bowl_dock_p_np[e][:2]
                    bwl = dpb + av * float(np.dot(bowl_j[:2] - dpb, av))
                    self._station_dock_np[e] = np.array([sta[0], sta[1], yaw], dtype=np.float32)
                    self._bowl_dock_np[e] = np.array([bwl[0], bwl[1], yaw], dtype=np.float32)
                    start = np.array(
                        [sta[0], sta[1], 0.0, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)],
                        dtype=np.float32)
                    self._robot_start_np[e] = start
                    # `_restore_robot` reads the **tensor**, which `_load_scene` built
                    # once. Updating only the numpy copy leaves the robot spawning at the
                    # old dock while the objects move away — which is what 14 of the
                    # first 30 randomized episodes failed on, at the grasp.
                    self._robot_start[e] = torch.as_tensor(
                        start, dtype=self._robot_start.dtype, device=self._robot_start.device)
                self._placement_fell_back[env_idx] = torch.as_tensor(
                    fell, dtype=torch.bool, device=self._placement_fell_back.device)
            else:
                self._placement_fell_back[env_idx] = False

            # After the draw, never before: `_restore_robot` puts the base at
            # `_robot_start`, and the station dock the draw just computed is where that
            # has to be. Restoring first placed the robot at the previous episode's dock
            # and then moved the objects away from it — every one of the first 30
            # randomized episodes failed, 14 of them at the grasp, on exactly that.
            self._restore_robot(env_idx)

            left = self._station_left[env_idx]
            right = self._station_right[env_idx]
            side = side_bits.unsqueeze(-1)
            shaker_pos = torch.where(side, left, right)
            bottle_pos = torch.where(side, right, left)

            jitter = lambda p: p + torch.cat(  # noqa: E731
                [(torch.rand((b, 2)) - 0.5) * 2 * self.cfg.spawn_jitter_xy, torch.zeros((b, 1))],
                dim=1,
            )
            # Mesh bottoms `spawn_clearance` above the surface (see _rest_lift_np).
            up = lambda p, k: p + torch.tensor([0.0, 0.0, 1.0]) * self._rest_lift[k]  # noqa: E731
            # `jitter` only when the fixed layout is in play. With `randomize_placements`
            # the draw already applied it *inside* its acceptance loop (K76), and
            # re-jittering here would throw the certified separation away again — which
            # is the exact bug that ejected seed 79's shaker across the kitchen.
            jz = (lambda q: q) if self.cfg.randomize_placements else jitter
            bowl_pos = up(jz(self._bowl_home[env_idx]), 0)
            shaker_pos = up(jz(shaker_pos), 1)
            bottle_pos = up(jz(bottle_pos), 2)

            # Yaw only. This is a correctness requirement, not a style choice: the
            # tilt predicate in evaluate() compares the body's +Z against world +Z,
            # and that is only a statement about "upright" if the spawn never rolls
            # or pitches the object.
            self.bowl.set_pose(Pose.create_from_pq(p=bowl_pos, q=self._yaw_quat(b)))
            self.shaker.set_pose(Pose.create_from_pq(p=shaker_pos, q=self._yaw_quat(b)))
            self.condiment_bottle.set_pose(
                Pose.create_from_pq(p=bottle_pos, q=self._yaw_quat(b))
            )

            target_pos = torch.where(target_bits.unsqueeze(-1), shaker_pos, bottle_pos)
            distractor_pos = torch.where(target_bits.unsqueeze(-1), bottle_pos, shaker_pos)
            self._distractor_home[env_idx] = distractor_pos

            marker_pos = target_pos + torch.tensor([0.0, 0.0, self.cfg.marker_height])
            self._marker_home[env_idx] = marker_pos
            self.recipe_marker.set_pose(Pose.create_from_pq(p=marker_pos))

            # Task state is not sim state — mask it by hand.
            self.pour_hold[env_idx] = 0
            # Seeded with the drawn pose so `bowl_moved` is defined before `cue_steps`;
            # overwritten with the settled pose at that step (K73).
            self._bowl_settled[env_idx] = bowl_pos[..., :2]
            self._pour_last_eval_step[env_idx] = -1

    def _draw_placement(self, i: int, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """Draw (bowl, station_left, station_right) on env `i`'s counter (K74).

        Rejection sampling over `usable_counter_regions` — jezv's free area, ported —
        under three constraints, all of which are load-bearing:

        1. **Reach.** `cfg.reach_band` from the dock. The counter's usable band reaches
           1.17 m from the dock at its back edge, which no plan survives; the fixed
           layout sat at 0.833 m (K71). The dock slides along the counter to the object,
           so what this really bounds is how deep on the counter a thing may sit.
        2. **Separation.** `cfg.min_object_gap` between every pair, so the bowl (25.6 cm
           across) never overlaps a seasoning and the gripper can get between them.
        3. **Station pairing.** The two seasonings stay `cfg.station_spacing` apart on
           the counter's `along` axis, because the task's whole point is a cue that
           picks one of two *interchangeable* stations; scattering them independently
           would make "which one" a different question.

        Returns `(bowl, left, right, fell_back)`. On exhaustion it returns the fixed
        layout with `fell_back=True` rather than an out-of-reach draw — and says so in
        `info`, because a silent fallback would quietly measure the old task.
        """
        regions = self._usable_regions[i] if i < len(self._usable_regions) else []
        along = self._along_np[i].astype(np.float64)
        av = along[:2] / max(float(np.linalg.norm(along[:2])), 1e-9)
        dock_p = self._dock_p_np[i]
        top_z = float(self._counter_top_z[i])
        lo, hi = float(self.cfg.reach_band[0]), float(self.cfg.reach_band[1])
        half = along * (self.cfg.station_spacing / 2.0)

        # The dock is the counter's standoff **slid along the counter** to the object,
        # so what reach bounds is the perpendicular depth onto the counter, not the raw
        # distance to the un-slid standoff. Measuring the latter rejects everything more
        # than a few centimetres along the counter and silently falls back to the fixed
        # layout — which is exactly what the first version of this did.
        across = self._across_np[i].astype(np.float64)[:2]
        n = float(np.linalg.norm(across))
        across = across / n if n > 1e-9 else np.array([0.0, 1.0])

        def dock_clear(xy, clear: float) -> bool:
            """Would the dock for this object leave `clear` metres to the end wall?"""
            q = np.asarray(xy, dtype=np.float64)[:2]
            dock = dock_p[:2] + av * float(np.dot(q - dock_p[:2], av))
            return bool(dock[0] >= WALL_LEFT_FACE_X + clear)

        def reach_ok(xy) -> bool:
            d = abs(float(np.dot(np.asarray(xy)[:2] - dock_p[:2], across)))
            return lo <= d <= hi

        j = self.cfg.spawn_jitter_xy
        rad = self._obj_radius_np
        gap = self.cfg.contact_band + self.cfg.min_object_gap

        def jit(q):
            return q + np.concatenate([rng.uniform(-j, j, size=2), [0.0]])

        def apart(a, ra, b, rb) -> bool:
            return bool(np.linalg.norm(a[:2] - b[:2]) >= ra + rb + gap)

        if regions and self.cfg.placement_draw == "pair-first":
            # W30e: the pair on its own constraints first, the bowl given the pair. The same
            # tests as the joint loop below, in two stages, so the pair's midpoint is uniform
            # over its feasible windows instead of weighted by the room the bowl finds.
            tries = int(self.cfg.placement_draw_tries)
            for _ in range(tries):
                mid = sample_in_regions(regions, rng, z=top_z)
                left, right = jit(mid + half), jit(mid - half)
                if not apart(left, float(rad[1]), right, float(rad[2])):
                    continue
                if not (reach_ok(left) and reach_ok(right)):
                    continue
                if not dock_clear(0.5 * (left + right), self.cfg.station_dock_wall_clear):
                    continue
                if not all(point_in_regions(regions, q, margin=0.01) for q in (left[:2], right[:2])):
                    continue
                for _ in range(max(20, tries // 20)):
                    bowl = jit(sample_in_regions(regions, rng, z=top_z))
                    if not (apart(bowl, float(rad[0]), left, float(rad[1]))
                            and apart(bowl, float(rad[0]), right, float(rad[2]))):
                        continue
                    if not reach_ok(bowl):
                        continue
                    if not dock_clear(bowl, self.cfg.dock_wall_clear):
                        continue
                    if not point_in_regions(regions, bowl[:2], margin=0.01):
                        continue
                    return bowl, left, right, False
                # no bowl fits beside this pair: draw another pair

        if regions:
            for _ in range(int(self.cfg.placement_draw_tries)):
                bowl = sample_in_regions(regions, rng, z=top_z)
                mid = sample_in_regions(regions, rng, z=top_z)
                left, right = mid + half, mid - half
                # Every acceptance test below runs on the **jittered** points, which is
                # the whole point of drawing the jitter inside this loop. The station
                # checks matter as much as the midpoint's: a legal midpoint at x 0.69 put
                # a station at 0.81 — on the sink — and another put one at x 0.28, off
                # the counter's left end.
                # The jitter is drawn HERE, not in `_initialize_episode` (K76). Applying
                # it after the acceptance tests threw away up to 2*0.02*sqrt(2) = 56.6 mm
                # of a gap this loop had just certified, so the guaranteed separation was
                # 0.1034 m and not the 0.16 m the config promises. Measured: seed 3 spawned
                # a bottle 0.1419 m from the bowl centre — below the sampler's own
                # threshold — with 1.8 mm of mesh clearance, and it toppled at reset.
                j = self.cfg.spawn_jitter_xy
                jit = lambda q: q + np.concatenate(  # noqa: E731
                    [rng.uniform(-j, j, size=2), [0.0]])
                bowl, left, right = jit(bowl), jit(left), jit(right)

                # Radius-aware, not a scalar centre-to-centre gap: the bowl is 26 cm
                # across and a seasoning 5 cm, so `min_object_gap` alone bought 4 mm over
                # bare non-overlap and nothing at all against PhysX's contact band
                # (`contact_offset` 0.02 either side). Seed 79 drew a legal 0.1813 m with
                # 31 mm of real clearance, inside that band, and PhysX ejected the shaker
                # at 7.3 m/s to a resting place 6.5 m away on the floor.
                rad = self._obj_radius_np
                pts = [(bowl[:2], float(rad[0])), (left[:2], float(rad[1])), (right[:2], float(rad[2]))]
                if any(np.linalg.norm(a[0] - b[0]) < a[1] + b[1] + self.cfg.contact_band + self.cfg.min_object_gap
                       for a, b in ((pts[0], pts[1]), (pts[0], pts[2]), (pts[1], pts[2]))):
                    continue
                if not (reach_ok(bowl) and reach_ok(left) and reach_ok(right)):
                    continue
                # Both docks must be somewhere the robot can actually stand while
                # holding something — the corner is not (K76).
                # The bowl dock is an ARRIVAL (drive along the counter, elbow leading); the
                # station dock is where the robot is placed — two clearances (W30).
                if not (dock_clear(bowl, self.cfg.dock_wall_clear)
                        and dock_clear(0.5 * (left + right), self.cfg.station_dock_wall_clear)):
                    continue
                if not all(point_in_regions(regions, q, margin=0.01)
                           for q in (bowl[:2], left[:2], right[:2])):
                    continue
                return bowl, left, right, False

        return (
            self._bowl_home_fixed[i].copy(),
            self._station_left_fixed[i].copy(),
            self._station_right_fixed[i].copy(),
            True,
        )

    def _yaw_quat(self, b: int) -> torch.Tensor:
        yaw = torch.rand(b) * 2 * math.pi
        return torch.stack(
            [torch.cos(yaw / 2), torch.zeros(b), torch.zeros(b), torch.sin(yaw / 2)], dim=1
        )

    def _restore_robot(self, env_idx: torch.Tensor):
        """The rest keyframe and the task's start pose at the station dock, for every robot.

        `scene_builder.initialize` restores the robot only when uid == "fetch"
        (scene_builder.py:563-577 compares a literal), and even then to *its* pose —
        `robot_poses[env_idx]`, a dock at a fixture it picked by `rng.choice`
        (scene_builder.py:528-551) — not to the task's `_robot_start`, the counter
        dock slid in front of the condiment station (T5, K38). Until 2026-08-18 this
        method returned early for uid "fetch" and restored `ds_fetch` to
        `robot_poses` — a random dock of the builder's, often at another fixture.
        No uid test now (as burner.py): this runs after `scene_builder.initialize`
        and simply overwrites it, which is safe and idempotent. (The uid test in
        `_fix_ds_fetch_collision_bits` is a different matter and must stay.)
        """
        if self.agent is None:
            return
        self.agent.robot.set_qpos(self.agent.keyframes["rest"].qpos)
        self.agent.robot.set_pose(Pose.create(self._robot_start[env_idx]))

    # --------------------------------------------------------------- evaluate --

    def evaluate(self) -> dict:
        """Runs every step — and once inside reset(), at t=0, before any action.

        Two consequences the code below has to survive:
          - the phase teleport must be a no-op at t=0 (it is: 0 < cue_steps);
          - the hold counter would tick on that call too, which is why the success
            expression also gates on elapsed_steps, and why the counter is guarded
            against advancing twice within one simulation step.
        """
        # --- the cue, on its schedule --------------------------------------
        cue_visible = self.elapsed_steps < self.cfg.cue_steps
        marker = self._marker_home.clone()
        marker[:, 2] = torch.where(
            cue_visible, marker[:, 2], torch.full_like(marker[:, 2], HIDDEN_Z)
        )
        self.recipe_marker.set_pose(Pose.create_from_pq(p=marker))

        # --- select target vs distractor by mask, never by indexing actors ---
        m = self.target_is_shaker
        m1 = m.unsqueeze(-1)
        shaker_p, bottle_p = self.shaker.pose.p, self.condiment_bottle.pose.p
        shaker_R = self.shaker.pose.to_transformation_matrix()[:, :3, :3]
        bottle_R = self.condiment_bottle.pose.to_transformation_matrix()[:, :3, :3]

        target_p = torch.where(m1, shaker_p, bottle_p)
        target_R = torch.where(m1.unsqueeze(-1), shaker_R, bottle_R)
        distractor_p = torch.where(m1, bottle_p, shaker_p)

        grasp_shaker = self.agent.is_grasping(self.shaker, min_force=self.cfg.grasp_min_force)
        grasp_bottle = self.agent.is_grasping(
            self.condiment_bottle, min_force=self.cfg.grasp_min_force
        )
        grasp_target = torch.where(m, grasp_shaker, grasp_bottle)
        grasp_distractor = torch.where(m, grasp_bottle, grasp_shaker)

        # --- tilt ------------------------------------------------------------
        axis = torch.tensor(self.cfg.pour_axis_body, dtype=torch.float32, device=self.device)
        axis_world = target_R @ axis
        cos_tilt = axis_world[:, 2].clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        tilt_rad = torch.arccos(cos_tilt)
        tilted = tilt_rad >= math.radians(self.cfg.pour_tilt_deg)

        # --- position over the bowl -----------------------------------------
        bowl_p = self.bowl.pose.p
        xy_to_bowl = torch.linalg.norm(target_p[:, :2] - bowl_p[:, :2], dim=1)
        clearance = target_p[:, 2] - bowl_p[:, 2]
        over_bowl = xy_to_bowl <= self.cfg.pour_xy_radius
        height_ok = (clearance >= self.cfg.pour_min_clearance) & (
            clearance <= self.cfg.pour_max_clearance
        )

        # --- the clause that makes the 50 % ceiling true ---------------------
        # Without it, "pick up both and wave both over the bowl" succeeds every
        # episode with no memory at all.
        distractor_moved = torch.linalg.norm(
            distractor_p[:, :2] - self._distractor_home[:, :2], dim=1
        )
        distractor_ok = (distractor_moved <= self.cfg.distractor_move_tol) & ~grasp_distractor

        # How far the bowl has been shoved **by the robot**. Measured, not gated (K73):
        # the user's frame-by-frame read of seed 53 — "the hand with the salt pushed the
        # bowl" — and nothing in this task had ever looked.
        #
        # The reference is the bowl where it comes to **rest**, not where it was
        # commanded to spawn. That distinction is the whole measurement: against
        # `_bowl_home` the metric reads 1.38 cm on average and >=1 cm on 54 of 75
        # episodes, which looks like the robot barging around and is not — it is the
        # bowl settling out of its drawn pose under gravity. Measured with the robot
        # held completely idle, seeds 3 and 27 settle 2.53 and 2.55 cm, exactly their
        # full-episode figures, and settling is complete by step 40. Against the settled
        # pose the robot's own contribution is median **0.00 cm**, above 1 cm on
        # **1 of 75** episodes, and that one is seed 53 at 2.97 cm.
        #
        # `cue_steps` is the capture point because the cue phase is dead time by
        # construction — the oracle only idles through it — so the bowl is at rest and
        # the robot has not yet moved. The `torch.where` is idempotent, which it must be:
        # `evaluate()` is called out of band by three sites (see the hold counter below).
        settled = torch.where(
            (self.elapsed_steps == self.cfg.cue_steps).unsqueeze(-1),
            self.bowl.pose.p[:, :2],
            self._bowl_settled,
        )
        self._bowl_settled = settled
        bowl_moved = torch.linalg.norm(self.bowl.pose.p[:, :2] - settled, dim=1)

        pour_now = grasp_target & over_bowl & height_ok & tilted & distractor_ok

        # --- the hold counter, idempotent within a step ----------------------
        # evaluate() mutates, and three call sites invoke it out of band:
        # myrobocasa_planner.py:284 and diagnose_task.py:58,108. Review §11 names
        # exactly this hazard in TakeItBack, where placed_on_sink latches on any
        # stray get_info(). Advancing only when elapsed_steps has actually moved
        # makes a second call within the same step a no-op.
        step = self.elapsed_steps.to(torch.int32)
        advance = step != self._pour_last_eval_step
        self.pour_hold = torch.where(
            advance & pour_now,
            self.pour_hold + 1,
            torch.where(advance, torch.zeros_like(self.pour_hold), self.pour_hold),
        )
        self._pour_last_eval_step = torch.where(advance, step, self._pour_last_eval_step)

        earliest = self.cfg.cue_steps + self.cfg.delay_steps
        success = (self.pour_hold >= self.cfg.hold_steps) & (self.elapsed_steps >= earliest)

        # A brace literal, not dict(success=...): the convention check is
        # `"dict(success=" in src.replace(" ", "")` (test_task_conventions.py:161),
        # which a multi-line dict(...) call trips even when it returns intermediates.
        return {
            "success": success,
            "pour_now": pour_now,
            "pour_hold": self.pour_hold,
            "tilted": tilted,
            "tilt_rad": tilt_rad,
            "over_bowl": over_bowl,
            "xy_to_bowl": xy_to_bowl,
            "height_ok": height_ok,
            "clearance": clearance,
            "grasp_target": grasp_target,
            "grasp_distractor": grasp_distractor,
            "is_grasping_shaker": grasp_shaker,
            "is_grasping_bottle": grasp_bottle,
            "distractor_ok": distractor_ok,
            "distractor_moved": distractor_moved,
            "bowl_moved": bowl_moved,
            "placement_fell_back": self._placement_fell_back,
            "cue_visible": cue_visible,
            # Ground truth, for the trajectory file and for the oracle arm of the
            # control experiment. info is not an observation: only what
            # _get_obs_extra copies out of it reaches a policy.
            "target_is_shaker": self.target_is_shaker,
        }

    # -------------------------------------------------------------------- obs --

    def get_language_instruction(self, **kwargs):
        """`INSTRUCTIONS[0]` for every env: the text is the chore, the answer is the cue."""
        return [INSTRUCTIONS[0]] * self.num_envs

    def _get_obs_extra(self, info: dict) -> dict:
        """The most dangerous method in the file. Three ways it can leak the answer.

        1. **Key order.** In state mode the dict is flattened in insertion order,
           so emitting `target_pose` before `distractor_pose` puts the correct
           object at a fixed slice index — a perfect memoryless signal. The two
           condiments are therefore always emitted as `shaker_pose`, `bottle_pose`,
           in that order, and no key is ever named for the target.
        2. **Conditional keys.** The observation space is frozen from the t=0
           observation (sapien_env.py:330, :377). Emitting the cue only during the
           cue phase changes the dict shape at step 40. It is always emitted, and
           masked to zeros instead.
        3. **Obs mode.** _get_obs_extra runs in every mode, so an ungated cue hands
           an image policy the answer in clean symbolic form and the marker becomes
           decorative. Ground truth is gated on use_state.

        `phase` stays ungated: it carries time, not the answer, and a policy can
        count steps anyway.
        """
        cue_visible = info["cue_visible"]
        obs = dict(
            tcp_pose=self.agent.tcp_pose.raw_pose,
            phase=torch.stack(
                [
                    cue_visible,
                    (~cue_visible) & (self.elapsed_steps < self.cfg.cue_steps + self.cfg.delay_steps),
                    self.elapsed_steps >= self.cfg.cue_steps + self.cfg.delay_steps,
                ],
                dim=1,
            ).to(torch.float32),
        )
        if self.obs_mode_struct.use_state:
            zero = torch.zeros_like(self.recipe_marker.pose.p)
            obs.update(
                bowl_pose=self.bowl.pose.raw_pose,
                shaker_pose=self.shaker.pose.raw_pose,
                bottle_pose=self.condiment_bottle.pose.raw_pose,
                # Masked, not omitted. After the cue phase this is exactly zeros,
                # which is the same tensor in both halves of the answer symmetry.
                recipe_cue=torch.where(cue_visible.unsqueeze(-1), self.recipe_marker.pose.p, zero),
            )
        return obs

    # ------------------------------------------------------------------ state --

    def get_state_dict(self) -> dict:
        """The episode's answer has to survive a checkpoint.

        For a memory benchmark this is the state that matters most: without it
        env.set_state(env.get_state()) drops the answer, and a recorded successful
        episode replays as a failure.
        """
        state = super().get_state_dict()
        state["target_is_shaker"] = self.target_is_shaker.clone()
        state["station_left_is_shaker"] = self.station_left_is_shaker.clone()
        state["pour_hold"] = self.pour_hold.clone()
        state["pour_last_eval_step"] = self._pour_last_eval_step.clone()
        state["distractor_home"] = self._distractor_home.clone()
        state["marker_home"] = self._marker_home.clone()
        return state

    def set_state_dict(self, state: dict, env_idx: torch.Tensor = None):
        """Tolerates a state dict that has no task keys, because one is routine.

        BaseEnv.set_state(flat) rebuilds a dict holding only "actors" and
        "articulations" (sapien_env.py:1309-1327), so every task-added key is gone
        by the time it reaches here. template_task.py:276 indexes its key
        unconditionally and would raise KeyError on exactly the
        env.set_state(env.get_state()) round trip AGENTS.md calls for.

        super() first, not last: it restores sim state, and the task's flags should
        land on a consistent scene.
        """
        super().set_state_dict(state, env_idx)
        restore = {
            "target_is_shaker": "target_is_shaker",
            "station_left_is_shaker": "station_left_is_shaker",
            "pour_hold": "pour_hold",
            "pour_last_eval_step": "_pour_last_eval_step",
            "distractor_home": "_distractor_home",
            "marker_home": "_marker_home",
        }
        for key, attr in restore.items():
            if key in state:
                setattr(self, attr, restore_task_tensor(getattr(self, attr), state[key], self.device))
