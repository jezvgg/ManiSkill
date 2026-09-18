"""SHARED base for the cabinet-search family — registers NOTHING.

MikasaCabinetSearch-v0: a red cube is hidden in ONE of N wall-cabinet
compartments. Find it. Close every cabinet you open, go back to the marked
home spot before opening another, and never open a compartment twice.

The memory type is the agent's OWN ACTION HISTORY — the set of compartments
it has already opened (design blank I, docs/task-designs/I-cabinetsearch-v0.md).
There is no cue phase and no clock: the trace is erased by the agent itself
(closing the door) and by the env's detent (a closed door snaps to exactly
0.000 once the hand has left its bar, so a visited compartment and an
untouched one are identical by construction — the SameDrawer detent, applied
to hinges). What remains of the history exists only in the policy.

Success = REVEAL: the cube's compartment door stands open at or past
`theta_reveal` with no fail latch (the owner's 2026-09-01 decision; the found
cabinet is NOT closed). Failure is instant (`fail` -> `terminated`,
sapien_env.py:1055-1056) on any of four sticky latches:

  reopened      a decision on a compartment already opened this episode
  two_open      a door rising while another compartment still stands open
  skipped_home  a decision on a DIFFERENT compartment without a home visit
                in between (the first opening counts: the start is outside
                the home disk and `passed_home` is False at reset)
  foreign_moved a non-compartment kitchen joint drifted past
                `foreign_drift_tol` (enters `fail` only once
                `foreign_drift_fails` is switched on after the pilot)

A "decision" is the rising edge of a compartment (any of its hinges past
`theta_count`, and either past `theta_closed` or with the hand at the bar —
a still, hand-away door inside the closed band is a brush: the eraser zeroes
it, it is never a decision) while `passed_home` is True. The same compartment
rising again WITHOUT a home visit is a RETRY (a slipped grasp knocked the door
shut) — no latch, counted in `retry_count`. Every latch is written by the
pure function `step_search_latches`, which the offline tests drive on
synthetic traces.

Memoryless floor (uniform with replacement, a repeat = fail),
`memoryless_search_floor`: 17/27 of the motor rate at N=3, **71/128 = 0.5547 at
the shipped N=4**; with memory 1.0 at either. The headline memory endpoint is
the repeat rate on the SECOND decision (`second_decision_ok` conditional on
`second_decision_made`): memoryless 1/N (1/4 here), memory 0.

That floor holds only while EVERY compartment is erased behind the agent. The
optional fifth compartment (`CAB_1`, `COMPARTMENTS_WITH_CAB_1`) is not: the
owner's rule for it is "drive the door to the wall" (W24 — its closing ladder
mirrors behind the room's west wall, so the family's fist push has nowhere to
stand), and a leaf parked at the wall is a MARK THAT SURVIVES THE ROUND. A
memoryless agent can rule that compartment out on sight, so the floor for a set
containing it is NOT the plain formula — see `self_marking_search_floor`, which
derives it (0.5442 for the five-compartment set, against 0.5021 plain). That is
the price of the fifth door, stated rather than hidden, and it is why cab_1 is
OFF by default: `MikasaCabinetSearch-v0` ships the four erased compartments.

Motor substrate, all measured on kitchen 102 (K103-K108, W20 dump 2026-09-01):
`pull_hinge_arc` opens a right door to 1.75 in ~144 steps at v=0.20 (9/10);
the fist push closes to 0.116-0.120 (9/9). Hinge anchors: cab_2 L/R at x
0.765/1.735, cab_main L/R at 1.765/2.735 (all y=-0.400, sense +1). Left doors
open by DECREASING qpos (limits [-3, 0]) — hence `open_dir` per hinge.

Skeleton: cabinet_retrieval_base.py (fixture and articulation resolution,
reset qpos write after scene_builder.initialize) + depth_recall.py (latches,
state dict) + same_drawer.py (detent, advance guard) + station_checklist.py
(home predicate on base_link, state-dict guard).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import sapien
import torch
from transforms3d import euler

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.scene_builder.robocasa.scene_builder import RoboCasaSceneBuilder
from mani_skill.utils.structs import Actor, Pose

from .robocasa_utils import require_get_fixture, restore_task_tensor, parking_pose

#: Keys `get_state_dict` adds beyond the simulator state — every one is task
#: MEMORY or the ticket. One place, so the state tests and `SearchLatches`
#: cannot drift apart (`test_latch_buffers_are_exactly_the_state_keys`).
TASK_STATE_KEYS = (
    "cube_cab", "instruction_idx",
    "opened_count", "is_open", "passed_home", "last_opened",
    "reopened", "two_open", "skipped_home", "foreign_moved",
    "second_decision_made", "second_decision_ok", "retry_count",
    "succeeded", "articulation_home", "last_eval_step", "seen_count", "cube_spawn",
)
#: Keys a recording may lack and still replay: added after the key was frozen, and
#: restoring them as zero changes nothing but the step the success latches on
#: (`cube_spawn` at zero makes every cube "nudged" — a recording from before the
#: touch terminal replays under the seen terminal, MIKASA_SEARCH_TERMINAL=seen).
TASK_STATE_OPTIONAL = ("seen_count", "cube_spawn")

#: No numerals, no "twice"/"second"/"first": the count of compartments and the
#: count of allowed openings must not reach a language-conditioned policy
#: through the text (station_checklist's trap 5). The home rule is STATED —
#: an unstated rule with an instant fail would fail an agent that follows the
#: instruction literally (review finding).
INSTRUCTIONS = (
    "Find the cube hidden in the wall cabinets. Go to the marked spot on the "
    "floor before opening any cabinet, close each cabinet you open, and "
    "return to the mark before opening the next. Never open a cabinet you "
    "have already opened.",
    "A cube is hidden in the wall cabinets. Starting from the floor mark, "
    "open cabinets to look for it; close the cabinet behind you and come "
    "back to the mark before trying another. Do not open a cabinet you "
    "already opened.",
    "Search the wall cabinets for the cube. Every search starts at the "
    "marked spot: shut each cabinet after looking and head back to the mark "
    "before the next. Opening the same cabinet again fails the task.",
)

#: The same three under the touch terminal (`cfg.terminal == "nudge"`): the goal
#: is STATED — the episode ends on a push of the cube, and a policy that only
#: looks would wait out the horizon without being told why. Same word rules.
INSTRUCTIONS_NUDGE = (
    "Find the cube hidden in the wall cabinets and nudge it when you find it. "
    "Go to the marked spot on the floor before opening any cabinet, close "
    "each cabinet you open, and return to the mark before opening the next. "
    "Never open a cabinet you have already opened.",
    "A cube is hidden in the wall cabinets. Starting from the floor mark, "
    "open cabinets to look for it and give it a push when you see it; close "
    "the cabinet behind you and come back to the mark before trying another. "
    "Do not open a cabinet you already opened.",
    "Search the wall cabinets for the cube and push it when you find it. "
    "Every search starts at the marked spot: shut each cabinet after looking "
    "and head back to the mark before the next. Opening the same cabinet "
    "again fails the task.",
)

#: The W13-measured graspable depth band on the cabinet shelf (y, world). The
#: cube is never grasped here, but the band is also where W22 will measure
#: visibility, so the spawn stays inside it.
SHELF_DEPTH_BAND = (-0.28, -0.10)

#: How a round on a compartment ends (`Compartment.close_policy`). "push" is the
#: family's rule and the only one any shipped default uses; "wall_park" is the
#: W24 rule for cab_1 and it costs the task the plain memoryless floor
#: (`self_marking_search_floor`).
CLOSE_POLICIES = ("push", "wall_park")

#: cab_1's stem. Named because three places have to recognise exactly this box:
#: the collision-mask untangle in `_load_scene`, the single-door bar arithmetic,
#: and `validate`'s refusal to build it while its stem is not untangled.
CAB_1_STEM = "cab_1_main_group"

UNTANGLE_BITS = {
    "cab_1_main_group": 26,
    "cab_micro_main_group": 25,
    "sink_main_group": 25,
    "stack_4_main_group_1": 25,
    "stack_4_main_group_2": 25,
}
"""Every kitchen-102 ARTICULATION this robot cannot touch, and the bit why.

MEASURED, not assumed. A SAPIEN collision shape carries four group words; a bit
shared in the THIRD word filters the pair out of contact entirely.
`_fix_ds_fetch_collision_bits` sets bits 25-29 on all 20 shapes of `ds_fetch`
(the wheel/floor exemption the RoboCasa builder skips for our uid), so a
fixture sharing a bit with the robot's word 2 is intangible to this robot and
only to this robot.

HOW IT WAS SCANNED, and why over the WHOLE word. The table was produced by
walking the live `MikasaCabinetSearch-v0` scene (scene_idx=0, one env) shape by
shape and printing `fixture_word2 & robot_word2_union` across all 32 bits —
never a 25-29 window and never from prose. The window would not have been
enough to support the word "every": the robot's word-2 UNION is bits 0-9 and
**25-31**, and only its INTERSECTION — what all 20 shapes share — is 25-29.
`base_link` additionally carries bit 31 and both wheel links carry bit 30,
which is exactly how `floor_room` (25-31) and the three walls (25-30) are
filtered out of contact with the base. A fixture carrying only 30 or 31 would
therefore be a ghost to the base and wheels and a 25-29 scan would have missed
it. Scanned over the full word: none is, and the five stems below are the
complete set. Every value here still lands in the 25-29 intersection, which is
the stronger statement — those bits are ignored by every robot shape, the
fingers included, so they are what makes a door ungraspable.

What that scan printed, in full:

    articulation                  bit   links carrying it
    cab_1_main_group               26   object (6 shapes), hingedoor (8)
    cab_micro_main_group           25   object (6), hingeleftdoor (6),
                                        hingerightdoor (6)
    sink_main_group                25   object (9), spout (4), handle (2)
    stack_4_main_group_1           25   object (6), hingedoor (8)
    stack_4_main_group_2           25   object (5), inner_box (12)
    cab_2/cab_4/cab_main            -   none (cab_2 carries 22, cab_main 23 —
                                        bits the robot does NOT ignore)
    microwave/stove/stack_1_*       -   none
    stack_2_main_group_1            -   none
    stack_3_main_group_1            -   none
    ds_fetch (the robot)      25...29   every one of its 20 shapes (union
                                        0-9 + 25-31; see above)

Two entries have NO link named `*door*` — they are `UNTANGLE_DOORLESS`, and
both `validate()` and `_untangle_fixture_door` refuse them. They are listed
anyway because the table's job is to say which fixtures are intangible, not
only which ones the door rule can fix.

STATIC fixtures carry the same exemption and are NOT in this table, because
they are actors, not articulations: they have no door link and nothing here can
untangle them. Measured in the same scan (bits SHARED with the robot):
`cab_3_main_group` (3 shapes, bit 26), `counter_main_main_group` (4, bit 26),
`dishwasher_main_group` (1, bit 26), `stack_4_main_group_base` (1, bit 25),
`floor_room` (1 shape, bits 25-31) and the three walls `wall_room` /
`wall_left_room` / `wall_right_room` (1 shape each, bits 25-30) — which is why
the base drives THROUGH the west wall, W24 run 1.

Kitchen 102 only (`scene_idx=0`, the pinned kitchen). Under any other layout
these stems and bits are unmeasured, and `_untangle_fixture_door` will refuse
rather than guess."""

UNTANGLE_DOORLESS = frozenset({"sink_main_group", "stack_4_main_group_2"})
"""The `UNTANGLE_BITS` entries the door-only rule can NEVER make tangible.

Measured in the same scan, and the reason this is a separate name: being
intangible and being untangleable are two different sets. These two carry a
robot ignore bit like the other three, but neither has a link with `door` in
its name — `sink_main_group` is object (9) / spout (4) / handle (2) and
`stack_4_main_group_2` is object (5) / inner_box (12), a drawer. Clearing a
bit on "the door links" of either would clear nothing.

`validate()` refuses them OFFLINE for that reason, with the same fact the build
would state ~4 s later inside `_untangle_fixture_door`: this is knowable
without loading a kitchen, so a container sweep should not spend a run
discovering it. Making either fixture solid needs a second, separately measured
rule (which links, and what it costs to make a carcass tangible) — not this
one."""


@dataclass(frozen=True)
class Compartment:
    """One searchable unit: a cabinet stem and the hinges that open into it.

    `theta` per hinge is `open_dir * qpos`, which is >= 0 while opening for
    both door handednesses (left doors run qpos from 0 down to -3, right doors
    from 0 up to 3 — the W20 dump). A compartment's opening angle is the max
    over its hinges. A HingeCabinet is ONE box with no interior wall
    (cabinet.py:562-600), so a stem may host two compartments only with the
    kinematic `partition` that splits it (`CabinetSearchConfig.validate`);
    without one, both leaves of a box belong to a SINGLE two-hinge compartment
    (the N=2 fallback), because either door exposes the whole interior.
    `open_hinge` is the door the oracle pulls and the one the reveal is
    measured on; its handedness names the spawn half.

    `close_policy` is how a ROUND on this compartment ends — what makes it
    count as dealt with (`step_search_latches`'s falling edge):

      "push"       the family's rule: every hinge back under `theta_closed`,
                   still, hand away; the detent then snaps it to exactly 0.000
                   and a visited compartment is identical to an untouched one.
      "wall_park"  the W24 rule for cab_1: the open hinge reaches
                   `cfg.wall_park_rad` against the room's west wall. NOT
                   snapped — the leaf stays where it is, which is the whole
                   point of the rule and the whole cost of it (the mark
                   survives; see `self_marking_search_floor`).

    Example:
        >>> c = Compartment("cab_2_main_group", ("leftdoorhinge", "rightdoorhinge"),
        ...                 "rightdoorhinge", (-1, 1))
        >>> c.half, c.name, c.close_policy
        ('right', 'cab_2_main_group:rightdoorhinge', 'push')
        >>> CAB_1.half, CAB_1.close_policy
        ('whole', 'wall_park')
    """

    stem: str
    hinges: tuple
    open_hinge: str
    open_dir: tuple
    close_policy: str = "push"

    @property
    def half(self) -> str:
        """Which part of the fixture box the open hinge exposes.

        "left"/"right" for a two-door box, whose leaf covers [edge, centre] or
        [centre, edge]; **"whole"** for a SINGLE-door cabinet (RoboCasa names
        its one hinge `doorhinge`, not `*doorhinge`), whose leaf exposes the
        entire box — cab_1. The spawn band is centred on that part, so a
        single-door compartment's cube sits on the box centre.
        """
        if self.open_hinge == "doorhinge":
            return "whole"
        return "left" if self.open_hinge.startswith("left") else "right"

    @property
    def name(self) -> str:
        return f"{self.stem}:{self.open_hinge}"

    def validate(self) -> None:
        assert self.open_hinge in ("leftdoorhinge", "rightdoorhinge", "doorhinge"), (
            f"{self.name}: open_hinge must be a RoboCasa door hinge name"
        )
        assert self.close_policy in CLOSE_POLICIES, (
            f"{self.name}: close_policy {self.close_policy!r} is not one of "
            f"{sorted(CLOSE_POLICIES)}"
        )
        if self.open_hinge == "doorhinge":
            assert self.hinges == ("doorhinge",), (
                f"{self.name}: `doorhinge` is a SINGLE-door cabinet's only hinge; "
                f"it cannot share its box with another leaf: {self.hinges}"
            )
        assert len(self.hinges) >= 1 and len(set(self.hinges)) == len(self.hinges), (
            f"{self.name}: hinges must be non-empty and unique: {self.hinges}"
        )
        assert self.open_hinge in self.hinges, (
            f"{self.name}: open_hinge {self.open_hinge!r} not among {self.hinges}"
        )
        assert len(self.open_dir) == len(self.hinges), (
            f"{self.name}: one open_dir per hinge: {self.open_dir} vs {self.hinges}"
        )
        assert all(int(d) in (-1, 1) for d in self.open_dir), (
            f"{self.name}: open_dir entries must be -1 or +1: {self.open_dir}"
        )


#: N = 4: BOTH boxes split into their two halves by a kinematic partition at the
#: box centre (see `partition`) — cab_2 at x=1.25, cab_main at x=2.25 (the W20
#: census). Every compartment owns exactly ONE hinge, so the hinge index and the
#: compartment index coincide in `compartment_layout`. Supersedes the N=3 form
#: (2026-09-01 - 2026-09-02), where cab_2 was one undivided two-hinge box; the
#: floor moves 17/27 -> 71/128 and every leaf needs its own `DoorSpec`
#: (`cabinet_search_planner.DOOR_SPECS`).
DEFAULT_COMPARTMENTS = (
    Compartment("cab_2_main_group", ("leftdoorhinge",), "leftdoorhinge", (-1,)),
    Compartment("cab_2_main_group", ("rightdoorhinge",), "rightdoorhinge", (1,)),
    Compartment("cab_main_main_group", ("leftdoorhinge",), "leftdoorhinge", (-1,)),
    Compartment("cab_main_main_group", ("rightdoorhinge",), "rightdoorhinge", (1,)),
)

CAB_1 = Compartment(CAB_1_STEM, ("doorhinge",), "doorhinge", (-1,),
                    close_policy="wall_park")
"""The optional FIFTH compartment — the west wall unit, W24 (2026-09-02).

MEASURED: hinge (0.265, -0.400), axis +z, sense +1, joint named `doorhinge` —
NOT `leftdoorhinge` like its neighbours, because the box carries a single door
that exposes the whole interior (hence `half == "whole"` and no partition). The
limits are [-3, 0], so it opens by DECREASING qpos: `open_dir` -1, a left leaf.
Its door link is `hingedoor`, the exact mirror of `cab_main`'s right leaf
shifted by -1.604, and its bar was scanned at (0.698, -0.435, 1.592).

Two things make it different from every other compartment here.

1. Its door is INTANGIBLE to this robot until the scene is untangled.
   `_fix_ds_fetch_collision_bits` puts ignore bits 25-29 on every robot shape
   (the wheel/floor exemption) and cab_1's door shapes carry bit 26, so the
   pads pass straight through the bar — W24 measured 20/20 grasp-grid cells
   closing on nothing with the TCP 2-17 mm from the bar. Putting this stem in
   `cfg.untangle_stems` clears that one bit on this door (see `_load_scene`
   and `UNTANGLE_BITS`); `validate()` refuses to build cab_1 without it,
   because an untouchable compartment is a task that cannot be solved.

2. It CANNOT be closed the family's way. The push side of an opened left leaf
   lies WEST of its hinge — inside the room's west wall at x=0 — so the fist
   has nowhere to stand. The owner's rule is instead "drive the door to the
   wall", and once the door is untangled the wall is a real backstop: the leaf
   walks freely to -2.0000 and is refused from -2.10 on (settling at -2.0526),
   while the same writes pass straight through in the four-compartment env.
   Hence `close_policy="wall_park"` — and hence the corrected floor. See
   `CabinetSearchConfig.wall_park_rad` for the numbers and for the one probe
   caveat that comes with them.
"""

COMPARTMENTS_WITH_CAB_1 = (CAB_1,) + DEFAULT_COMPARTMENTS
"""N = 5: cab_1 plus the four erased compartments — `MikasaCabinetSearch5-v0`.

cab_1 goes FIRST so the compartment order stays monotone west to east (spawn
centres 0.50, 1.00, 1.50, 2.00, 2.50 — the Mac smoke reads them back in that
order). The consequence, stated because it will bite anyone comparing sweeps:
the compartment INDEX means something different here than in the N=4 task —
index 0 is cab_1, not `cab_2`'s left leaf — so `cube_cab` from one task's
trajectories must never be read against the other's compartment list.

NOT a default. The four-compartment task is the one whose numbers are measured,
and this set trades a lower plain floor for a self-marking door: see
`self_marking_search_floor`."""


@dataclass
class CabinetSearchConfig:
    """Every threshold named; magic numbers in evaluate() are how failures go mute.

    Example:
        >>> cfg = CabinetSearchConfig()
        >>> cfg.validate()
        >>> len(cfg.compartments), cfg.theta_reveal > cfg.theta_count
        (4, True)
    """

    horizon: int = 7100
    """K22 against the SHIPPED four-compartment oracle, measured 2026-09-02 on
    calibration seeds 11-22 (never the eval seeds 0-9), container CPU, 7 of 12
    succeeding: the longest successful episode is seed 15 at 4706 steps (the cube
    late in a four-round search), 4706 x 1.5 = 7059 -> 7100. The shapes it covers:
    a one-round find lands at 717-989, a two-round find at 2901-3073, the long
    ones at 4353-4706. Nothing truncated in that sweep (0/12), so this is a cushion
    over a measured worst case, not a bound the sweep leaned on. Supersedes both
    the 6500 of the three-compartment task and the 10700 this field carried while
    the fourth compartment was arithmetic. Must equal max_episode_steps in the
    decorator, which reads this constant. Expires with the oracle."""

    # -- the compartments ------------------------------------------------------
    compartments: tuple = DEFAULT_COMPARTMENTS
    """Searchable units in a fixed order (the compartment index the latches
    and the ticket use). Two compartments on one stem are legal only with the
    partition (validate())."""
    counter_name: str = "counter_main_main_group"
    "Exact stem (get_fixture's substring fallback draws RANDOMLY among matches)."
    partition: bool = True
    """Build a kinematic wall inside every stem that hosts two compartments,
    at the box's x centre, so the two halves are separate interiors (a
    HingeCabinet is one box: cabinet.py:562-600). Without it, opening one
    half exposes the other and N collapses. At the shipped N=4 that is TWO
    walls — cab_2 at x=1.25 and cab_main at x=2.25 — built by the same loop
    over the shared stems (`_load_scene`)."""
    partition_y: tuple = (-0.36, 0.0)
    """Wall depth span, world y: inside the box (y in [-0.37, 0]) and clear of
    the door panels at y ~ -0.385..-0.415, which swing SOUTH, away from it."""
    partition_z: tuple = (1.42, 2.30)
    "Shelf floor (W13: 1.4200) to just under the box top (2.31)."
    partition_half_thickness: float = 0.005

    # -- cab_1, the optional fifth compartment (W24) ----------------------------
    wall_park_rad: float = 1.40
    """A `close_policy="wall_park"` compartment counts as dealt with once its open
    hinge reaches this angle — the owner's "drive the door to the wall", capped by
    a measurement at the angle where the ROBOT still fits.

    The literal wall is at 2.05 rad (W24: writing -2.10 settles at -2.0526 once the
    leaf is untangled), and the oracle does reach it — the first five-compartment
    episode in the container drove cab_1's door to **2.003**. What it also did was
    drive the BASE into the west wall: the arc pull keeps the TCP on the handle's
    circle, so the base rides that circle too, and past ~1.8 rad it stands at
    x < 0.15 where the room's wall is at x = 0. Physically it passes through (the
    robot ignores the wall, bits 25-29), but the PLANNING world does not, and every
    plan after the park refused `base_link<->wall_left_room` — the round parked the
    door and ended the episode with it.

    So the ORACLE pulls this leaf to `DOOR_OPEN_TARGET` (1.75) like every other —
    the last angle measured to leave the base clear; the pull to 1.751 planned on
    afterwards, the extra 25 steps to 2.003 did not — and the LATCH threshold sits
    below where the freed hinge settles. The drift is measured, and it is bigger
    here than the 0.05 the cabinet line records for its right leaf: the same
    container run parked at 1.751 and 1.755 and read back **1.636 and 1.452** once
    the hand was away. 1.40 is under that band and still over `theta_reveal` (1.2),
    so a parked leaf is by construction a revealed one, and a leaf merely opened
    and abandoned is not parked.

    cab_1's individual rule therefore reads "opened as far as any other door and
    left standing", not "flat against the wall": the door is out of the way, the
    robot is not in the wall, and the leaf still marks itself — the cost the floor
    arithmetic already prices in.

    `validate()` refuses anything outside [1.6, 2.05]: under 1.6 the door has not
    left the working corridor and the park would fire on an ordinary open; over
    2.05 it is past where the wall starts holding — and, as the container run
    showed, past where the robot's own base still fits beside it."""
    untangle_stems: frozenset = frozenset()
    """Fixture stems whose DOOR shapes lose their robot-shared ignore bit at scene
    build, so the robot can touch them (`_untangle_fixture_door`, `UNTANGLE_BITS`).

    Empty by default: the four-compartment task touches no mask at all, which is
    the state every number in this repo was measured under. The five-compartment
    task sets exactly `{CAB_1_STEM}` (`CabinetSearch5Config`), because cab_1's
    door is intangible without it and `validate()` refuses that combination.

    Every stem must be a key of `UNTANGLE_BITS` — a measured fixture — and must
    not be one of `UNTANGLE_DOORLESS`. There is no silent no-op here: an unknown
    stem, a doorless one, or a stem whose door shapes do not actually carry the
    bit all raise. The first two are knowable from the table, so `validate()`
    refuses them offline; the third needs the live scene and raises at build.

    A `frozenset`, enforced, not merely annotated: a bare string would iterate as
    characters and untangle nothing, and a plain set on a CLASS attribute
    (`CabinetSearchTask.cfg`) could be added to after validation passed.

    It is a change to the SCENE's collision mask, on named doors and nowhere else
    — never to the robot's own mask, which every measured number in this repo was
    taken under. W24 ran both variants and shipped this one for that reason."""

    @property
    def untangle_cab_1(self) -> bool:
        """Back-compatible read of the boolean this field replaced (W24 shipped a
        single-purpose `untangle_cab_1: bool`). True iff cab_1's stem is in
        `untangle_stems`. Read-only — the set is the field; constructing a config
        with `untangle_cab_1=...` is a TypeError, deliberately loud."""
        return CAB_1_STEM in self.untangle_stems

    # -- door thresholds (theta = open_dir * qpos, >= 0) -----------------------
    theta_count: float = 0.05
    """A door counts as OPENED at this angle (with the hand at its bar, or past
    theta_closed) — the peek floor, measured by W22 (2026-09-02, cube-present vs
    cube-hidden pixel diffs on the ds_fetch rig over a floor scan of the whole
    corridor): a leaf cracked to 0.05 rad leaks at most **4 px** of cube from one
    grazing floor spot, 0.10 leaks 12-14, 0.15 leaks 20-31 and 0.30 leaks 25-51.
    A closed door leaks 0 px from everywhere (the control). So 0.05 is the
    largest angle a memoryless agent cannot read the answer through, and the
    4 px that survive are the recorded residual — closing them further would put
    the threshold inside the hinge's own numerical noise.

    The ordering against `theta_closed` is deliberately NOT load-bearing: the
    rules read `theta_count` for what counts as an opening and `theta_closed`
    for what counts as shut, with the hand-away eraser covering the band between
    them whichever way round they sit (see `step_search_latches`)."""
    theta_closed: float = 0.15
    """A compartment whose hinges all sit at or under this, still, with the
    hand away, counts as closed and snaps to 0.000 (the detent). 0.15 rad ~
    8.6 deg — the cabinet family's `door_closed_rad` (W13: a free hinge holds
    where the fist released it, so exact zero would fail honest closes; K106's
    push lands at 0.116-0.120, and the search oracle reproduces 0.119 on every
    round it closes). Inside this band a door with no hand at its bar is never
    an opening (see theta_count)."""
    theta_reveal: float = 1.2
    """PROVISIONAL (W22): the cube's open hinge at or past this = revealed =
    success. Must be past 0.9 (ARM_PASS, W12) and reachable by the pull
    (1.75 target, settles ~1.70, K108). At 0.9 the panel still covers a cube
    by the outer wall from the pull-end pose (review finding)."""
    terminal: str = os.environ.get("MIKASA_SEARCH_TERMINAL", "nudge")
    """What ends the episode on the revealed cube. `nudge` (owner, 2026-09-08): the cube
    has been TOUCHED — pushed at least `cube_nudge_m` from where it spawned, with its
    compartment open; a policy has to reach into the cabinet, not only open it and look.
    `seen`: the W22d terminal — the revealed cube inside a base camera's cone for
    `reveal_dwell_steps` steps with the base still. Both keep every fail latch."""
    cube_nudge_m: float = 0.02
    """The push that counts as a touch: the cube's HORIZONTAL displacement from its spawn
    (the settle is vertical, 2 cm of `spawn_clearance`); 2 cm is above any jitter and
    below a fall off the shelf."""
    reveal_dwell_steps: int = 10
    """The cheap terminal (the owner's choice, 2026-09-07, W22d): success needs the
    revealed cube SEEN — its centre inside `look_cone_rad` of a base camera's optical
    axis, in front of it and within `look_range_m` — with the base static, for this
    many consecutive steps. Reveal alone let a policy that opens doors in the right
    order succeed without ever looking, and gave a demonstration no terminal action;
    a touch would price the answer in motor skill (that task exists: Retrieval).
    Geometry only, no render: the camera pose is the head link's pose composed with
    the rig's mount (`ds_fetch._sensor_configs`), which matches the render params
    to the millimetre (measured 2026-09-07). 0 = the old reveal-only success."""
    look_cone_rad: float = float(os.environ.get(
        "MIKASA_LOOK_CONE_RAD",
        "0.70" if (os.environ.get("MIKASA_LOOK_MODE", "head") == "base"
                   or os.environ.get("MIKASA_LOOK_BY_HEAD", "1") == "0") else "0.40"))
    """Half-angle of the "seen" cone about a base camera's optical axis, radians. The
    rig's FOV is 1.5 rad (43 deg half). 0.40 (23 deg) is the frame's middle half — the
    head look centres the spawn point at ~2 deg off axis (W22c). The oracle's 15 deg
    base-turn look (`MIKASA_LOOK_MODE=base`, the W22b shape) leaves it at ~36 deg, in
    the corner, so that mode defaults the cone to 0.70 (40 deg): the same variable sets
    the oracle's look and the task's idea of "seen", and the two must agree or the
    demonstrations would not pass their own terminal. `MIKASA_LOOK_CONE_RAD` pins it."""
    look_range_m: float = 3.0
    "Farther than this from the camera is not 'seen' (the kitchen is 3 m across)."
    door_still_vel: float = 0.02
    "rad/s under which a hinge is 'still' (same_drawer's 0.02)."
    hand_far_m: float = 0.25
    """The TCP farther than this from every bar of a compartment = the hand
    is away. Gates all three things the machine does to a door inside the
    closed band: the eraser (an untouched door snapped to 0), the detent (a
    closed door snapped to 0 — a bar held in the fingers must not be
    teleported into the panel: 0.13 rad x 0.435 m = 5.7 cm, review finding)
    and the closed-band rising (a door under theta_closed is an opening only
    while the hand is at its bar). The bar is the CLOSED-door bar
    (`handle_*`): every gate fires on near-closed doors, where the live bar
    is within ~6 cm of it."""

    # -- the handle bars (for hand_far_m), by construction ---------------------
    handle_hpad: float = 0.05
    """The bar stands this far inside the door's FREE edge along x
    (cabinet_panels.py:168-220, hpad); bar_x = box centre +- hpad. The
    arithmetic reproduces W12's scanned cab_main R bar (2.302, -0.430, 1.592)
    at 2.3015 — 0.5 mm."""
    handle_y: float = -0.430
    handle_z: float = 1.592
    "World y/z of every top-row bar on kitchen 102 (W12 scan, W20 census)."
    handle_dock_y: float = -1.30
    "The measured pull dock's y (HANDLE_DOCK = (2.30, -1.30), K103-K108)."
    home_dock_min_m: float = 0.50
    "Home must be at least this far from every handle dock (checked at build)."

    # -- the home spot ---------------------------------------------------------
    home_xy: tuple = (1.80, -1.90)
    """Free floor south of the counter row, between the cab_2 dock (x 1.30)
    and the cab_main docks (2.20/2.30), 0.6 m south of the dock line."""
    home_yaw_deg: float = 90.0
    "Facing the cabinets (+y)."
    home_radius: float = 0.20
    "The disk (station_checklist's dock_radius; K26 parking error 0.008 m)."
    home_yaw_tol_deg: float = 35.0
    base_static_speed: float = 0.08
    "|root x/y/yaw velocity| under this = the base is parked (station_checklist)."
    home_marker_z: float = 0.002
    """The visual disk's centre height. The RoboCasa floor RENDERS as a box
    of half-size 0.02 centred at z=-0.02 (fixture `floor_room`: top face at
    z=0.0); its collision is a plane at z=-0.02 that nothing here touches —
    ds_fetch's base is carried by its root joints at z=0. The disk
    (half-length 0.001) sits 1 mm above the top face so it draws on the
    floor, not in it (probe 2026-09-02)."""

    # -- the cube --------------------------------------------------------------
    cube_half: float = float(os.environ.get("MIKASA_CUBE_HALF", "0.05"))
    """Half-side of the cube (10 cm cube by default, 2026-09-08; the owner asked for
    a bigger cube than the first 6 cm one so it reads in the cameras and the touch
    has more to hit). `MIKASA_CUBE_HALF` overrides for a probe; the spawn asserts in
    `validate` (compartment band, cabinet height) and `_build_layout` bound it."""
    cube_color: tuple = (1.0, 0.1, 0.1, 1.0)
    shelf_top_z: float = 1.4200
    "The cabinet's interior floor (W13, mesh-bottom at settle)."
    spawn_clearance: float = 0.02
    "The cube's BOTTOM starts this far above the shelf (never the origin)."
    spawn_jitter_x: float = 0.06
    "Cube x: the compartment half's centre +- this."
    spawn_depth: float = -0.26
    spawn_jitter_depth: float = 0.02
    """Cube y: the FRONT of W13's band (-0.28..-0.24) — nearest the door
    opening and the cameras; W22 measures visibility across this band."""

    # -- the robot start -------------------------------------------------------
    start_xy: tuple = (1.80, -2.40)
    """0.5 m SOUTH of home — outside the home disk by construction, so the
    first decision needs a real home visit (validate() checks the margin)."""
    start_yaw_deg: float = 90.0
    start_jitter_xy: float = 0.08
    start_jitter_yaw: float = 0.10

    # -- foreign kitchen joints ------------------------------------------------
    foreign_drift_tol: float = 0.10
    """PROVISIONAL: |q - articulation_home| over this on any non-compartment
    kitchen joint latches `foreign_moved`. Calibrated by the pilot's
    `foreign_drift_max` print (step 5 of the plan)."""
    foreign_drift_fails: bool = False
    """Whether `foreign_moved` enters `fail`. Off until the pilot has
    calibrated the tolerance and excluded knobs the drives themselves bump
    (the plan: «включается после пилота»); the latch and `foreign_drift_max`
    are always reported."""
    require_found_closed: bool = False
    "Owner decision 2026-09-01: the found cabinet is NOT closed. Fixed False."

    def validate(self) -> None:
        """Refuse to build on self-contradictory numbers. Called from __init__.

        Example:
            >>> CabinetSearchConfig().validate()
            >>> CabinetSearchConfig(theta_reveal=0.5).validate()  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
            AssertionError
        """
        assert self.terminal in ("nudge", "seen"), self.terminal
        assert 0.0 < self.cube_nudge_m < 0.10, self.cube_nudge_m

        assert self.horizon > 0
        n = len(self.compartments)
        assert n in (2, 3, 4, 5), (
            f"{n} compartments: the family is floored for N=2 (0.75x), N=3 "
            "(17/27), N=4 (71/128 = 0.5547, the shipped default) and N=5 (the "
            "cab_1 set, 0.5442 by `self_marking_search_floor`); another N needs "
            "its own floor, its own docks and its own probes"
        )
        for c in self.compartments:
            c.validate()
        names = [c.name for c in self.compartments]
        assert len(set(names)) == n, f"duplicate compartments: {names}"
        pairs = [(c.stem, h) for c in self.compartments for h in c.hinges]
        assert len(set(pairs)) == len(pairs), (
            f"a hinge belongs to two compartments: {pairs}"
        )
        stems = [c.stem for c in self.compartments]
        shared = {s for s in stems if stems.count(s) > 1}
        if shared:
            assert self.partition, (
                f"stems {sorted(shared)} host two compartments each: a HingeCabinet "
                "is one box, so the halves need the kinematic partition"
            )
        # -- cab_1 and the wall park (W24) ------------------------------------
        if any(c.close_policy == "wall_park" for c in self.compartments):
            assert 1.3 <= self.wall_park_rad <= 2.05, (
                f"wall_park_rad={self.wall_park_rad}: the west wall stops the "
                "leaf at ~2.05 rad (W24: -2.10 settles at -2.0526) and 1.6 is "
                "the pull's own working angle — a park target outside "
                "[1.6, 2.05] is either unreachable or fires on an ordinary open"
            )
            assert self.wall_park_rad > self.theta_reveal, (
                f"wall_park_rad={self.wall_park_rad} is not past "
                f"theta_reveal={self.theta_reveal}: the park would land before "
                "the reveal and a found cube could be booked as dealt with"
            )
        assert isinstance(self.untangle_stems, frozenset), (
            f"untangle_stems must be a frozenset of stems, not "
            f"{type(self.untangle_stems).__name__} {self.untangle_stems!r}: a "
            "bare string would iterate as characters and untangle nothing, and "
            "a plain set is mutable — these configs are CLASS attributes "
            "(`CabinetSearchTask.cfg`), so one could be added to after "
            "validate() had already passed on it"
        )
        for stem in sorted(self.untangle_stems):
            assert stem in UNTANGLE_BITS, (
                f"untangle_stems: {stem!r} is not a measured fixture. "
                f"`UNTANGLE_BITS` lists every kitchen-102 articulation that "
                f"carries one of the robot's ignore bits 25-29: "
                f"{sorted(UNTANGLE_BITS)}"
            )
            assert stem not in UNTANGLE_DOORLESS, (
                f"untangle_stems: {stem!r} carries ignore bit "
                f"{UNTANGLE_BITS[stem]} but has NO link named *door* in the "
                "measured scan, so the door-only rule clears nothing on it "
                "(`UNTANGLE_DOORLESS`). Refused here rather than at scene "
                "build, where the same fact costs a kitchen load first"
            )
        if any(c.stem == CAB_1_STEM for c in self.compartments):
            assert CAB_1_STEM in self.untangle_stems, (
                f"{CAB_1_STEM} is among the compartments with its stem not in "
                "untangle_stems: its door shapes carry ignore bit 26 and the "
                "robot ignores bits 25-29, so the pads pass through the bar "
                "(W24) — that is an unreachable compartment, i.e. a task that "
                "cannot be solved"
            )

        assert self.partition_y[0] < self.partition_y[1] <= 0.0
        assert -0.37 <= self.partition_y[0], "the partition leaves the box (y)"
        assert 1.39 <= self.partition_z[0] < self.partition_z[1] <= 2.31
        assert self.partition_half_thickness > 0

        assert 0 < self.theta_count < self.theta_reveal, (
            f"theta_count={self.theta_count}: must be positive and under the "
            "reveal angle"
        )
        assert 0 < self.theta_closed <= 0.15, (
            f"theta_closed={self.theta_closed}: over 0.15 (~8.6 deg) the gap at "
            "the free edge is no longer closed for any purpose the task has "
            "(cabinet family's door_closed_rad)"
        )
        assert 0.9 <= self.theta_reveal <= 2.9, (
            f"theta_reveal={self.theta_reveal}: under 0.9 (ARM_PASS, W12) the "
            "panel covers the opening; 3.0 is the hinge stop"
        )
        assert self.door_still_vel > 0 and self.hand_far_m > 0
        assert self.handle_hpad > 0 and self.home_dock_min_m > 0

        assert self.home_radius > 0 and 0 < self.home_yaw_tol_deg < 90
        assert self.base_static_speed > 0
        assert self.reveal_dwell_steps >= 0
        assert 0.0 < self.look_cone_rad < math.pi / 2 and self.look_range_m > 0.5, (
            self.look_cone_rad, self.look_range_m)
        assert self.start_jitter_xy >= 0 and self.start_jitter_yaw >= 0
        d_start = math.hypot(self.start_xy[0] - self.home_xy[0],
                             self.start_xy[1] - self.home_xy[1])
        assert d_start > self.home_radius + math.sqrt(2.0) * self.start_jitter_xy, (
            f"the start ({d_start:.2f} m from home) can land inside the home "
            "disk: the first decision would need no home visit"
        )

        assert 0 < self.cube_half <= 0.10
        assert 1.39 <= self.shelf_top_z <= 2.31
        assert self.spawn_clearance > 0
        assert self.shelf_top_z + self.spawn_clearance + 2 * self.cube_half < 2.31
        assert self.spawn_jitter_x >= 0 and self.spawn_jitter_depth >= 0
        assert self.spawn_jitter_x + self.cube_half < 0.25, (
            "the cube band leaves its 0.5 m compartment half"
        )
        lo, hi = SHELF_DEPTH_BAND
        assert lo <= self.spawn_depth - self.spawn_jitter_depth and \
            self.spawn_depth + self.spawn_jitter_depth <= hi, (
                f"the spawn depth band leaves W13's measured band {SHELF_DEPTH_BAND}"
            )
        assert self.foreign_drift_tol > 0
        assert not self.require_found_closed, (
            "owner decision 2026-09-01: the found cabinet is not closed"
        )


# ------------------------------------------------------------ pure functions --


def memoryless_search_floor(n: int, r_nc: float = 1.0) -> float:
    """Success probability of a memoryless uniform search with replacement.

    Each round picks a compartment uniformly among N; a repeat is an instant
    fail; the cube is found the round its compartment is opened. `r_nc` is
    the motor success rate of a no-cube round (open, close, get home) —
    each empty round the search survives costs that factor; the cube round's
    own motor rate multiplies both arms equally and is left out.

    P(N) = sum_{j<N} (1/N) * prod_{i<j} ((N-1-i)/N * r_nc).

    Example:
        >>> memoryless_search_floor(2)
        0.75
        >>> from fractions import Fraction
        >>> Fraction(memoryless_search_floor(3)).limit_denominator(100)
        Fraction(17, 27)
        >>> Fraction(memoryless_search_floor(4)).limit_denominator(1000)
        Fraction(71, 128)
        >>> round(memoryless_search_floor(3, r_nc=0.73)
        ...       / search_success_with_memory(3, r_nc=0.73), 2)
        0.71
    """
    return float(_plain_floor_exact(int(n), r_nc))


def _plain_floor_exact(n: int, r_nc) -> Fraction:
    """`memoryless_search_floor` without the float cast — the exact rational.

    Both floors need it: `self_marking_search_floor` folds it into a second
    exact walk, and rounding to float in between would put noise in the middle
    of an arithmetic the docstrings state as fractions.
    """
    assert n >= 1
    r = Fraction(r_nc) if not isinstance(r_nc, Fraction) else r_nc
    total = Fraction(0)
    survive = Fraction(1)
    for j in range(n):
        total += Fraction(1, n) * survive
        survive *= Fraction(n - 1 - j, n) * r
    return total


def self_marking_search_floor(n: int, r_nc: float = 1.0) -> float:
    """`memoryless_search_floor` when ONE of the N compartments marks itself.

    The corrected floor for a set containing cab_1. Its round ends with the leaf
    parked at the wall (`close_policy="wall_park"`), and that leaf is never
    snapped back to 0 — so a memoryless agent, which cannot tell the other N-1
    visited compartments from untouched ones, CAN see that this one is done. The
    model: at every decision the agent draws uniformly among the compartments it
    cannot rule out, and the only one it can rule out is the mark, once visited.
    A draw of an already-opened compartment is still an instant fail; `r_nc` is
    the motor rate of a round that finds nothing, paid on every round survived
    (the mark's own round included — it is opened and parked like any other).

    DERIVATION. Write m for the self-marking compartment and condition on where
    the cube is.

    (A) The cube is IN m, probability 1/N. The mark is useless here: it only
        starts marking once it has been opened, and opening it ends the episode
        with a find. So until then the agent draws uniformly over all N and a
        repeat among the other N-1 is fatal — exactly the plain search:

            P(success | cube in m) = memoryless_search_floor(N, r_nc)

    (B) The cube is NOT in m, probability (N-1)/N. Let j count the DECOYS
        already opened (the compartments that are neither m nor the cube's;
        there are D = N-2 of them) and let b say whether m has been opened.
        With a_j = P(success | j decoys open, m still in the pool) and
        b_j = P(success | j decoys open, m ruled out):

            b_j = 1/(N-1) + r*(D-j)/(N-1) * b_{j+1},          b_D = 1/(N-1)
            a_j = 1/N + r/N * b_j + r*(D-j)/N * a_{j+1},      a_D = 1/N + r/N * b_D

        b unrolls to the plain floor of the N-1 compartments that are left once
        m is ruled out; a is the same walk with m still drawable, where drawing
        it costs a round and rules it out. The answer is a_0.

    So the floor is (1/N)*(A) + ((N-1)/N)*(B). At r_nc = 1 the two arms of (B)
    coincide — a_j = b_j exactly, because drawing m only re-runs the same
    lottery conditioned on not drawing m — and the whole thing collapses to

        floor_marked(N) = [ floor(N) + (N-1) * floor(N-1) ] / N          (r=1)

    i.e. the mark is FREE except when the cube is behind it. For N = 5:
    (1569/3125 + 4 * 71/128) / 5 = **272083/500000 = 0.544166**, against the
    plain 1569/3125 = 0.50208 — the fifth door buys less than a fifth door
    should, and that gap IS what the wall park costs the benchmark. Under motor
    attrition the collapse fails (a_j < b_j: the mark's round is paid for) and
    the recursion above is the honest number; the tests drive it against a
    brute-force enumeration at r_nc = 1, 0.73 and 0.5.

    The second-decision endpoint moves too, by the same argument: the first
    decision is the mark with probability 1/N given that it missed, and then the
    second draw cannot repeat at all, so a memoryless agent repeats on the
    second decision with probability (N-1)/N**2 — 4/25 = 0.16 here, not 1/5.

    Args:
        n: the number of compartments, at least 2 (a mark needs a non-mark).
        r_nc: motor success rate of a round that finds nothing.

    Returns:
        The memoryless success probability, relative to the motor rate of the
        finding round (which multiplies both arms and is left out, as in
        `memoryless_search_floor`).

    Example:
        >>> from fractions import Fraction
        >>> Fraction(self_marking_search_floor(5)).limit_denominator(10 ** 7)
        Fraction(272083, 500000)
        >>> round(self_marking_search_floor(5), 6)
        0.544166
        >>> self_marking_search_floor(5) > memoryless_search_floor(5)
        True
        >>> self_marking_search_floor(5) < memoryless_search_floor(4)
        True
        >>> # the r=1 collapse, on the numbers the blank names
        >>> round((memoryless_search_floor(5) + 4 * memoryless_search_floor(4)) / 5, 9)
        0.544166
        >>> # one compartment that marks itself out of two: draw, then no repeat
        >>> self_marking_search_floor(2)
        0.875
        >>> round(self_marking_search_floor(5, r_nc=0.73), 6)
        0.405905
    """
    n = int(n)
    assert n >= 2, "a self-marking compartment needs at least one other to hide among"
    r = Fraction(r_nc) if not isinstance(r_nc, Fraction) else r_nc
    d = n - 2
    b = [Fraction(0)] * (d + 2)
    for j in range(d, -1, -1):
        b[j] = Fraction(1, n - 1) + Fraction(d - j, n - 1) * r * b[j + 1]
    a = [Fraction(0)] * (d + 2)
    for j in range(d, -1, -1):
        a[j] = (Fraction(1, n) + Fraction(1, n) * r * b[j]
                + Fraction(d - j, n) * r * a[j + 1])
    plain = _plain_floor_exact(n, r)
    return float(Fraction(1, n) * plain + Fraction(n - 1, n) * a[0])


def search_success_with_memory(n: int, r_nc: float = 1.0) -> float:
    """The same search by an agent that never repeats — the sighted ceiling
    under per-round motor attrition `r_nc`: sum_{j<N} (1/N) * r_nc^j.

    Example:
        >>> search_success_with_memory(3)
        1.0
        >>> round(search_success_with_memory(3, r_nc=0.5), 4)
        0.5833
    """
    n = int(n)
    assert n >= 1
    r = Fraction(r_nc) if not isinstance(r_nc, Fraction) else r_nc
    return float(sum(Fraction(1, n) * r ** j for j in range(n)))


def _leaf_span(half: str, lo: float, mid: float, hi: float) -> tuple:
    """The world-x span one leaf covers, from its box's `lo`/`mid`/`hi`.

    "left" and "right" split a two-door box at its centre; "whole" is a
    SINGLE-door cabinet, whose one leaf covers the box (cab_1).
    """
    if half == "whole":
        return (lo, hi)
    return (lo, mid) if half == "left" else (mid, hi)


def _bar_x(hinge: str, span_lo: float, span_hi: float, hpad: float) -> float:
    """World x of a closed leaf's bar: `hpad` inside the FREE edge of its span.

    A right leaf hinges at its span's east end, so its bar is at
    `span_lo + hpad`; a left leaf — `leftdoorhinge`, and cab_1's single
    `doorhinge`, whose hinge W24 measured at (0.265, -0.400), the west end of
    its box — carries it at `span_hi - hpad`.
    """
    return span_lo + hpad if hinge.startswith("right") else span_hi - hpad


def hinge_theta(qpos: torch.Tensor, open_dir: torch.Tensor) -> torch.Tensor:
    """Opening angle per hinge, >= 0 while opening for either handedness.

    Example:
        >>> hinge_theta(torch.tensor([[-1.75, 1.75]]), torch.tensor([-1.0, 1.0]))
        tensor([[1.7500, 1.7500]])
    """
    return qpos * open_dir


def at_home_predicate(base_xy: torch.Tensor, base_yaw: torch.Tensor,
                      base_speed: torch.Tensor, cfg: CabinetSearchConfig) -> torch.Tensor:
    """Inside the home disk, facing home_yaw within the tolerance, parked.

    Example:
        >>> cfg = CabinetSearchConfig()
        >>> xy = torch.tensor([[1.85, -1.95], [1.85, -1.95], [2.5, -1.9]])
        >>> yaw = torch.tensor([math.radians(90), math.radians(90 - 360), math.radians(90)])
        >>> v = torch.tensor([0.0, 0.0, 0.0])
        >>> at_home_predicate(xy, yaw, v, cfg).tolist()
        [True, True, False]
    """
    home = torch.tensor(cfg.home_xy, dtype=base_xy.dtype, device=base_xy.device)
    d_xy = torch.linalg.norm(base_xy - home, dim=-1)
    dyaw = base_yaw - math.radians(cfg.home_yaw_deg)
    dyaw = torch.atan2(torch.sin(dyaw), torch.cos(dyaw)).abs()
    return (d_xy <= cfg.home_radius) \
        & (dyaw <= math.radians(cfg.home_yaw_tol_deg)) \
        & (base_speed < cfg.base_static_speed)


@dataclass(frozen=True)
class SearchLayout:
    """The hinge<->compartment wiring the latch step needs, as tensors.

    Hinges are flattened in compartment order; `hinge_cab[h]` is the
    compartment of hinge h, `reveal_hinge[c]` the global index of compartment
    c's open hinge, `open_dir[h]` its sign. `wall_park[c]` says whether
    compartment c ends its round at the wall instead of shut
    (`Compartment.close_policy`) — the one bit `step_search_latches` needs to
    tell the two closing rules apart.

    Example:
        >>> lay = compartment_layout(DEFAULT_COMPARTMENTS)
        >>> lay.hinge_cab.tolist(), lay.reveal_hinge.tolist(), lay.open_dir.tolist()
        ([0, 1, 2, 3], [0, 1, 2, 3], [-1.0, 1.0, -1.0, 1.0])
        >>> lay.wall_park.tolist()
        [False, False, False, False]
        >>> compartment_layout(COMPARTMENTS_WITH_CAB_1).wall_park.tolist()
        [True, False, False, False, False]
    """

    hinge_cab: torch.Tensor
    reveal_hinge: torch.Tensor
    open_dir: torch.Tensor
    hinge_names: tuple
    wall_park: torch.Tensor

    @property
    def n_compartments(self) -> int:
        return int(self.reveal_hinge.numel())

    @property
    def n_hinges(self) -> int:
        return int(self.hinge_cab.numel())


def compartment_layout(compartments, device="cpu") -> SearchLayout:
    """Flatten the compartments' hinges into a `SearchLayout`.

    Example:
        >>> compartment_layout(DEFAULT_COMPARTMENTS).hinge_names[2]
        ('cab_main_main_group', 'leftdoorhinge')
    """
    hinge_cab, open_dir, names, reveal = [], [], [], []
    for c_i, c in enumerate(compartments):
        for h, d in zip(c.hinges, c.open_dir):
            if h == c.open_hinge:
                reveal.append(len(hinge_cab))
            hinge_cab.append(c_i)
            open_dir.append(float(d))
            names.append((c.stem, h))
    return SearchLayout(
        hinge_cab=torch.tensor(hinge_cab, dtype=torch.long, device=device),
        reveal_hinge=torch.tensor(reveal, dtype=torch.long, device=device),
        open_dir=torch.tensor(open_dir, dtype=torch.float32, device=device),
        hinge_names=tuple(names),
        wall_park=torch.tensor([c.close_policy == "wall_park" for c in compartments],
                               dtype=torch.bool, device=device),
    )


@dataclass
class SearchLatches:
    """The task's per-env memory — exactly the `TASK_STATE_KEYS` buffers.

    The env carries these as attributes on itself; the offline tests carry
    them on an instance of this class. Both are driven by
    `step_search_latches`, which only ever reads and REASSIGNS the
    attributes (no in-place tensor writes), so a checkpoint clone never
    aliases live state. Every buffer is rank <= 2 (the flatten rule).

    Example:
        >>> z = SearchLatches.zeros(2, n_compartments=3, n_foreign=4)
        >>> z.is_open.shape, int(z.last_opened[0]), int(z.last_eval_step[0])
        (torch.Size([2, 3]), -1, -1)
    """

    cube_cab: torch.Tensor
    instruction_idx: torch.Tensor
    opened_count: torch.Tensor
    is_open: torch.Tensor
    passed_home: torch.Tensor
    last_opened: torch.Tensor
    reopened: torch.Tensor
    two_open: torch.Tensor
    skipped_home: torch.Tensor
    foreign_moved: torch.Tensor
    second_decision_made: torch.Tensor
    second_decision_ok: torch.Tensor
    retry_count: torch.Tensor
    succeeded: torch.Tensor
    articulation_home: torch.Tensor
    last_eval_step: torch.Tensor
    seen_count: torch.Tensor
    cube_spawn: torch.Tensor

    @classmethod
    def zeros(cls, n: int, n_compartments: int, n_foreign: int, device="cpu"):
        b = lambda *s: torch.zeros(s, dtype=torch.bool, device=device)  # noqa: E731
        i32 = lambda *s: torch.zeros(s, dtype=torch.int32, device=device)  # noqa: E731
        return cls(
            cube_cab=torch.zeros((n,), dtype=torch.long, device=device),
            instruction_idx=i32(n),
            opened_count=i32(n, n_compartments),
            is_open=b(n, n_compartments),
            passed_home=b(n),
            last_opened=torch.full((n,), -1, dtype=torch.int32, device=device),
            reopened=b(n), two_open=b(n), skipped_home=b(n), foreign_moved=b(n),
            second_decision_made=b(n), second_decision_ok=b(n),
            retry_count=i32(n),
            succeeded=b(n),
            articulation_home=torch.zeros((n, n_foreign), dtype=torch.float32,
                                          device=device),
            last_eval_step=torch.full((n,), -1, dtype=torch.int32, device=device),
            seen_count=i32(n),
            cube_spawn=torch.zeros((n, 3), dtype=torch.float32, device=device),
        )


def step_search_latches(latches, *, step: torch.Tensor, theta: torch.Tensor,
                        hinge_still: torch.Tensor, tcp_far: torch.Tensor,
                        at_home: torch.Tensor, foreign_hit: torch.Tensor,
                        layout: SearchLayout, cfg: CabinetSearchConfig,
                        cube_seen: torch.Tensor = None, base_static: torch.Tensor = None,
                        moved_m: torch.Tensor = None):
    """One evaluate() tick of the search latches, with no simulator.

    Reads and reassigns the `TASK_STATE_KEYS` attributes of `latches` (the
    env, or a `SearchLatches`). Every edge event is gated on `advance =
    step != last_eval_step`, so a second call in the same step changes
    nothing (evaluate() runs at t=0 inside reset and out of band from three
    call sites). Returns `(info, snap)`: `info` is the evaluate() dict
    (`success`, `fail` and the diagnostics), `snap` an `(n, H)` bool mask of
    hinges the caller must write to qpos = qvel = 0 — the detent. Inside,
    `theta` is treated as already zeroed on those hinges, so the sim write
    and the verdict agree by construction.

    The three door rules, the same under either ordering of theta_count and
    theta_closed (the owner's open decision, see `theta_count`):

      rising   ~is_open & th_max >= theta_count & (th_max > theta_closed
               | hand at the bar) — a door inside the closed band is an
               opening only while the hand holds it; a hand-away door
               there is a brush.
      eraser   ~is_open & 0 < th_max < theta_count | th_max <= theta_closed,
               still, hand away — the brush (and any sub-count mark) is
               zeroed; a moving brush waits until it is still.
      falling  is_open & th_max <= theta_closed, still, hand away — the
               detent never teleports a bar out of the fingers.

    A `close_policy="wall_park"` compartment (cab_1, W24) replaces the falling
    rule and only that one: it falls when its hinge reaches `cfg.wall_park_rad`
    — still, hand away, so the pull that drove it there has let go — and it is
    NOT put in the snap mask. Its leaf stays against the wall, which is the
    rule the owner asked for and the mark the corrected floor is priced on
    (`self_marking_search_floor`). REPLACES, not adds: a wall-park compartment
    driven back into the closed band is NOT dealt with — it stays `is_open`,
    home stays uncredited, and the round has to park it properly. Because that
    leaf never comes back through
    `theta_count`, the rising rule would otherwise re-fire on it every tick
    after the fall (and `reopened` on the next home visit), so a wall-parked
    compartment that has ALREADY been counted is barred from rising for exactly
    as long as it stands parked. Pull it back off the wall and it can rise
    again — and then it is a reopen like any other, which is the honest
    reading: the mark informs only while it stands.

    Args:
        step: `(n,)` int32 elapsed steps.
        theta: `(n, H)` opening angle per hinge (`hinge_theta`).
        hinge_still: `(n, H)` |qvel| < door_still_vel.
        tcp_far: `(n, H)` TCP farther than hand_far_m from the hinge's bar.
        at_home: `(n,)` the home predicate.
        foreign_hit: `(n,)` some non-compartment joint past foreign_drift_tol.
        layout: `compartment_layout(cfg.compartments)`.
        cfg: the thresholds.

    Example:
        >>> cfg = CabinetSearchConfig(); lay = compartment_layout(cfg.compartments)
        >>> z = SearchLatches.zeros(1, 4, 0); z.cube_cab[:] = 3
        >>> f = torch.zeros((1, 4), dtype=torch.bool); t = ~f
        >>> kw = dict(hinge_still=t, tcp_far=t, foreign_hit=f[:, 0], layout=lay, cfg=cfg)
        >>> th = torch.zeros((1, 4))
        >>> _ = step_search_latches(z, step=torch.tensor([1]), theta=th, at_home=t[:, 0], **kw)
        >>> th[0, 3] = 1.3   # cab_main R past theta_reveal, after a home visit
        >>> info, snap = step_search_latches(z, step=torch.tensor([2]), theta=th,
        ...                                  at_home=f[:, 0], moved_m=torch.tensor([0.0]), **kw)
        >>> bool(info["revealed"][0]), bool(info["fail"][0]), int(info["n_decisions"][0])
        (True, False, 1)
        >>> bool(info["success"][0])      # revealed, but not yet TOUCHED (the touch terminal)
        False
        >>> info, _ = step_search_latches(z, step=torch.tensor([3]), theta=th, at_home=f[:, 0],
        ...                               moved_m=torch.tensor([0.03]), **kw)
        >>> bool(info["nudged"][0]), bool(info["success"][0])
        (True, True)
        >>> seen = CabinetSearchConfig(terminal="seen"); kw2 = dict(kw, cfg=seen)   # the W22d terminal
        >>> z2 = SearchLatches.zeros(1, 4, 0); z2.cube_cab[:] = 3; th2 = torch.zeros((1, 4))
        >>> _ = step_search_latches(z2, step=torch.tensor([1]), theta=th2, at_home=t[:, 0], **kw2)
        >>> th2[0, 3] = 1.3
        >>> for k in range(2, 2 + seen.reveal_dwell_steps):
        ...     info, _ = step_search_latches(z2, step=torch.tensor([k]), theta=th2,
        ...                                   at_home=f[:, 0], **kw2)
        >>> bool(info["success"][0]), int(info["seen_count"][0])
        (True, 10)
    """
    n, H = theta.shape
    N = layout.n_compartments
    dev = theta.device
    step = step.to(torch.int32)
    advance = step != latches.last_eval_step
    adv = advance.unsqueeze(-1)

    # hinge -> compartment membership, (H, N)
    member = layout.hinge_cab.unsqueeze(-1) == torch.arange(N, device=dev).unsqueeze(0)
    neg_inf = torch.tensor(float("-inf"), device=dev, dtype=theta.dtype)
    th_max = theta.unsqueeze(-1).masked_fill(~member, neg_inf).amax(dim=1)     # (n, N)
    still_c = (hinge_still.unsqueeze(-1) | ~member).all(dim=1)               # (n, N)
    far_c = (tcp_far.unsqueeze(-1) | ~member).all(dim=1)                     # (n, N)

    is_open = latches.is_open
    passed_home = latches.passed_home
    opened_count = latches.opened_count
    last_opened = latches.last_opened

    closed_band = th_max <= cfg.theta_closed
    # cab_1's rule (W24). `parked` is the compartment's own mark: it stands at
    # the wall, and unlike a shut door it stays there.
    wall = layout.wall_park.to(theta.device).unsqueeze(0)                    # (1, N)
    parked = wall & (th_max >= cfg.wall_park_rad)                            # (n, N)
    # A door inside the closed band with no hand at its bar is a brush, not
    # an opening; past the band, or held, it is the rising edge. Under
    # theta_count > theta_closed the gate is implied by the count itself.
    #
    # A parked leaf sits far past theta_count for the rest of the episode, so
    # without a guard it would rise again on the very next tick — booking
    # retries on the way home and `reopened` the moment home is re-armed. The
    # guard is `parked AND already counted`, not `parked`: a compartment whose
    # FIRST opening is the one that reaches the wall (a leaf that flew past
    # wall_park_rad between two ticks) must still be counted, or opening cab_1
    # fast enough would be free. The only way to stand parked with nothing
    # counted is to have opened it without a home visit, which is `skipped_home`
    # and has already ended the episode.
    spent = parked & (opened_count >= 1)
    rising = (adv & ~is_open & (th_max >= cfg.theta_count)
              & (~closed_band | ~far_c) & ~spent)
    # The detent waits for the hand: the snap moves the bar ~0.435 m x theta.
    falling_push = adv & is_open & closed_band & still_c & far_c & ~wall
    falling_wall = adv & is_open & parked & still_c & far_c
    falling = falling_push | falling_wall
    # The eraser's band is the union of the sub-count band and the closed
    # band, so the brush is erased whichever threshold is the larger.
    erase_band = (th_max > 0) & ((th_max < cfg.theta_count) | closed_band)
    eraser = adv & ~is_open & erase_band & still_c & far_c
    # ... and the wall park is NOT snapped: the leaf stays where it is.
    snap_c = falling_push | eraser                                           # (n, N)
    snap = snap_c[:, layout.hinge_cab]                                       # (n, H)
    theta_after = theta.masked_fill(snap, 0.0)
    th_max_after = theta_after.unsqueeze(-1).masked_fill(~member, neg_inf).amax(dim=1)

    same_as_last = torch.arange(N, device=dev).unsqueeze(0) == last_opened.unsqueeze(-1)
    decision = rising & passed_home.unsqueeze(-1)
    retry = rising & ~passed_home.unsqueeze(-1) & same_as_last
    repeat_now = (decision & (opened_count >= 1)).any(dim=-1)
    decision_any = decision.any(dim=-1)

    reopened = latches.reopened | repeat_now
    two_open = latches.two_open | (
        rising & (is_open & ~rising).any(dim=-1, keepdim=True)).any(dim=-1)
    skipped_home = latches.skipped_home | (
        rising & ~passed_home.unsqueeze(-1) & ~same_as_last).any(dim=-1)
    foreign_moved = latches.foreign_moved | foreign_hit

    second = decision_any & (opened_count.sum(dim=-1) == 1) & ~latches.second_decision_made
    second_decision_ok = torch.where(second, ~repeat_now, latches.second_decision_ok)
    second_decision_made = latches.second_decision_made | second

    retry_count = latches.retry_count + retry.any(dim=-1).to(torch.int32)
    opened_count = opened_count + decision.to(torch.int32)
    last_opened = torch.where(decision_any, decision.to(torch.int32).argmax(dim=-1).to(torch.int32),
                              last_opened)
    is_open = (is_open | rising) & ~falling
    passed_home = (passed_home & ~decision_any) | (advance & at_home & ~is_open.any(dim=-1))

    fail_latched = reopened | two_open | skipped_home
    if cfg.foreign_drift_fails:
        fail_latched = fail_latched | foreign_moved
    rows = torch.arange(n, device=dev)
    reveal_h = layout.reveal_hinge[latches.cube_cab.long()]
    revealed = theta_after[rows, reveal_h] >= cfg.theta_reveal
    # The cheap terminal (W22d): the revealed cube has to be SEEN — inside a base
    # camera's cone with the base still — for `reveal_dwell_steps` consecutive steps.
    # `cube_seen` / `base_static` None (the offline traces) read as True; the dwell at 0
    # is the old reveal-only success.
    seen_now = revealed
    if cube_seen is not None:
        seen_now = seen_now & cube_seen
    if base_static is not None:
        seen_now = seen_now & base_static
    seen_count = torch.where(advance & seen_now, latches.seen_count + 1,
                             torch.where(advance & ~seen_now, torch.zeros_like(latches.seen_count),
                                         latches.seen_count))
    dwelt = seen_count >= int(cfg.reveal_dwell_steps)
    # The touch terminal: the cube has moved `cube_nudge_m` from its spawn. `moved_m`
    # None (the offline traces) reads as the displacement being enough.
    nudged = revealed if moved_m is None else (revealed & (moved_m >= float(cfg.cube_nudge_m)))
    done = nudged if getattr(cfg, "terminal", "seen") == "nudge" else dwelt
    succeeded = latches.succeeded | (revealed & done & ~fail_latched)

    latches.reopened = reopened
    latches.two_open = two_open
    latches.skipped_home = skipped_home
    latches.foreign_moved = foreign_moved
    latches.second_decision_ok = second_decision_ok
    latches.second_decision_made = second_decision_made
    latches.retry_count = retry_count
    latches.opened_count = opened_count
    latches.last_opened = last_opened
    latches.is_open = is_open
    latches.passed_home = passed_home
    latches.succeeded = succeeded
    latches.seen_count = seen_count
    latches.last_eval_step = torch.where(advance, step, latches.last_eval_step)

    info = {
        "success": succeeded,
        "fail": fail_latched & ~succeeded,
        "revealed": revealed,
        "seen": seen_now,
        "seen_count": seen_count,
        "nudged": nudged,
        "moved_m": moved_m if moved_m is not None else torch.zeros_like(theta[:, 0]),
        "at_home": at_home,
        "passed_home": passed_home,
        "any_open": is_open.any(dim=-1),
        "n_decisions": opened_count.sum(dim=-1).to(torch.int32),
        "decision_now": decision_any,
        "retry_now": retry.any(dim=-1),
        "snapped_now": snap_c.any(dim=-1),
        "second_decision_made": second_decision_made,
        "second_decision_ok": second_decision_ok,
        "reopened": reopened,
        "two_open": two_open,
        "skipped_home": skipped_home,
        "foreign_moved": foreign_moved,
        "retry_count": retry_count,
        "door_theta_max": th_max_after,
        "is_open": is_open,
    }
    return info, snap


# ------------------------------------------------------------------ the task --


class CabinetSearchTaskBase(BaseEnv):
    """Find the hidden cube; close behind you; home between openings; never twice.

    Example:
        >>> import gymnasium as gym  # doctest: +SKIP
        >>> env = gym.make("MikasaCabinetSearch-v0", num_envs=1, robot_uids="mikasa_ds_fetch",
        ...                control_mode="pd_joint_pos", obs_mode="state",
        ...                scene_idx=0, sim_backend="cpu")  # doctest: +SKIP
    """

    SUPPORTED_ROBOTS = ["mikasa_ds_fetch", "fetch"]
    SUPPORTED_REWARD_MODES = ["sparse", "none"]

    cfg = CabinetSearchConfig()

    cube: Actor

    def __init__(self, *args, robot_uids="mikasa_ds_fetch", scene_idx: int | None = 0, **kwargs):
        # Kitchen 102 pinned by default: every number in the config was
        # measured there (W12-W20).
        self.cfg.validate()
        self.scene_idx = scene_idx
        self.layout = compartment_layout(self.cfg.compartments)
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # ---------------------------------------------------------------- sensors --

    @property
    def _default_sensor_configs(self):
        # ds_fetch carries its own head and wrist rigs (the family's call).
        return []

    @property
    def _default_human_render_camera_configs(self):
        # Both cabinet boxes (x 0.75-2.75) and the home disk in one frame;
        # the eye stays inside the room (floor runs to y=-3.04).
        pose = sapien_utils.look_at(eye=[1.8, -3.0, 2.2], target=[1.8, -0.4, 1.4])
        return CameraConfig("render_camera", pose, 512, 512, 1.2, 0.01, 100)

    # ------------------------------------------------------------------ load --

    def _load_agent(self, options: dict):
        super()._load_agent(options, parking_pose(self))

    def _load_scene(self, options: dict):
        """Geometry and initial poses only — never set_pose (discarded by _setup)."""
        super()._load_scene(options)
        cfg = self.cfg
        n = self.num_envs

        self.scene_builder = RoboCasaSceneBuilder(self)
        if self.scene_idx is None:
            self.scene_builder.build()
        else:
            self.scene_builder.build([self.scene_idx] * n)
        self._fix_ds_fetch_collision_bits()

        stems = sorted({c.stem for c in cfg.compartments})
        self.counters, self.cabinets = [], []
        for i in range(n):
            fixtures = self.scene_builder.scene_data[i]["fixtures"]
            self.counters.append(require_get_fixture(
                self.scene_builder, fixtures, cfg.counter_name, scene_idx=self.scene_idx))
            self.cabinets.append({
                s: require_get_fixture(self.scene_builder, fixtures, s,
                                       scene_idx=self.scene_idx)
                for s in stems
            })

        # Articulations, resolved ONCE per env by exact key: RoboCasa names a
        # fixture's articulation `<stem>_<env index>` (fixture.py:107). The
        # robot is an articulation too (key "mikasa_ds_fetch") and is never a
        # kitchen joint.
        robot = self.agent.robot
        self._hinges: list[list[tuple]] = []
        self._foreign: list[list] = []
        for i in range(n):
            arts = {}
            for s in stems:
                key = f"{s}_{i}"
                assert key in self.scene.articulations, (
                    f"no articulation {key!r} for env {i}; have "
                    f"{sorted(self.scene.articulations)}"
                )
                arts[s] = self.scene.articulations[key]
            row = []
            for stem, hinge in self.layout.hinge_names:
                names = [j.name for j in arts[stem].get_active_joints()]
                assert hinge in names, (stem, hinge, names)
                row.append((arts[stem], names.index(hinge)))
            self._hinges.append(row)
            foreign = []
            for key in sorted(self.scene.articulations):
                art = self.scene.articulations[key]
                if art is robot or art.name == robot.name:
                    continue
                head, _, tail = key.rpartition("_")
                if not tail.isdigit() or int(tail) != i:
                    continue
                if head in stems:
                    continue
                foreign.append(art)
            self._foreign.append(foreign)
        dofs = [sum(len(a.get_active_joints()) for a in row) for row in self._foreign]
        assert len(set(dofs)) == 1, f"foreign joint layouts differ per env: {dofs}"
        self._n_foreign = dofs[0]

        for stem in sorted(cfg.untangle_stems):
            self._untangle_fixture_door(stem)

        # Per-compartment spawn half and per-hinge handle bar, from the fixture
        # box (the cabinet_retrieval `_exposed_half` arithmetic): a two-door box
        # is split at its centre — the right door covers [centre, right edge],
        # the left [left edge, centre] — while a SINGLE-door box (cab_1, whose
        # one hinge is named `doorhinge`) is covered whole by its one leaf, so
        # that compartment's span is the box itself and its cube band is centred
        # on the box centre.
        #
        # The bar stands `handle_hpad` inside the leaf's FREE edge, which is the
        # end of the leaf's own span away from its hinge: a right leaf hinges at
        # its span's east end and carries its bar at span_lo + hpad, a left leaf
        # (`leftdoorhinge` and cab_1's `doorhinge` alike) hinges at the west end
        # and carries it at span_hi - hpad. On a two-door box both reduce to the
        # centre +- hpad this code used before — cab_main R 2.25 + 0.05 = 2.30
        # against W12's scanned 2.302 — and on cab_1 it gives 0.75 - 0.05 = 0.70
        # against W24's scanned 0.698.
        centres = np.zeros((n, self.layout.n_compartments), dtype=np.float32)
        bars = np.zeros((n, self.layout.n_hinges, 3), dtype=np.float32)
        docks = np.zeros((n, self.layout.n_compartments, 2), dtype=np.float32)
        for i in range(n):
            for c_i, c in enumerate(cfg.compartments):
                box = self.cabinets[i][c.stem]
                pos = np.asarray(box.pos, dtype=np.float64)
                size = np.asarray(box.size, dtype=np.float64)
                lo = float(pos[0] - size[0] / 2.0)
                mid = float(pos[0])
                hi = float(pos[0] + size[0] / 2.0)
                h_lo, h_hi = _leaf_span(c.half, lo, mid, hi)
                centre = (h_lo + h_hi) / 2.0
                band = cfg.spawn_jitter_x + cfg.cube_half
                assert h_lo + 0.02 < centre - band and centre + band < h_hi - 0.02, (
                    f"{c.name}: cube band leaves its half [{h_lo:.2f}, {h_hi:.2f}]"
                )
                centres[i, c_i] = centre
                bar_x = _bar_x(c.open_hinge, h_lo, h_hi, cfg.handle_hpad)
                docks[i, c_i] = (bar_x, cfg.handle_dock_y)
                d_home = math.hypot(bar_x - cfg.home_xy[0], cfg.handle_dock_y - cfg.home_xy[1])
                assert d_home >= cfg.home_dock_min_m, (
                    f"{c.name}: home is {d_home:.2f} m from its handle dock "
                    f"({bar_x:.2f}, {cfg.handle_dock_y})"
                )
            for h, (stem, hinge) in enumerate(self.layout.hinge_names):
                box = self.cabinets[i][stem]
                pos = np.asarray(box.pos, dtype=np.float64)
                size = np.asarray(box.size, dtype=np.float64)
                lo = float(pos[0] - size[0] / 2.0)
                mid = float(pos[0])
                hi = float(pos[0] + size[0] / 2.0)
                half = "whole" if hinge == "doorhinge" else (
                    "left" if hinge.startswith("left") else "right")
                h_lo, h_hi = _leaf_span(half, lo, mid, hi)
                bars[i, h] = (_bar_x(hinge, h_lo, h_hi, cfg.handle_hpad),
                              cfg.handle_y, cfg.handle_z)
        self._spawn_centre_x = torch.as_tensor(centres, device=self.device)
        self._bar_xyz = torch.as_tensor(bars, device=self.device)
        self._handle_dock = torch.as_tensor(docks, device=self.device)
        self._open_dir = self.layout.open_dir.to(self.device)

        # The partition: a kinematic wall at the box centre of every stem that
        # hosts two compartments (the StackRecall plinth idiom, build_box
        # kinematic). Constant pose -> initial_pose only.
        self.partitions = []
        shared = sorted({c.stem for c in cfg.compartments
                         if sum(1 for d in cfg.compartments if d.stem == c.stem) > 1})
        if cfg.partition:
            y0, y1 = cfg.partition_y
            z0, z1 = cfg.partition_z
            for s in shared:
                mid = float(np.asarray(self.cabinets[0][s].pos, dtype=np.float64)[0])
                self.partitions.append(actors.build_box(
                    self.scene,
                    half_sizes=[cfg.partition_half_thickness, (y1 - y0) / 2.0, (z1 - z0) / 2.0],
                    color=[0.5, 0.5, 0.5, 1.0],
                    name=f"partition_{s}",
                    body_type="kinematic",
                    initial_pose=sapien.Pose(p=[mid, (y0 + y1) / 2.0, (z0 + z1) / 2.0]),
                ))
                for c in cfg.compartments:
                    if c.stem != s:
                        continue
                    band = cfg.spawn_jitter_x + cfg.cube_half + cfg.partition_half_thickness
                    assert abs(float(centres[0, cfg.compartments.index(c)]) - mid) > band + 0.02, (
                        f"{c.name}: the cube band touches the partition at x={mid:.2f}"
                    )

        # The cube: dynamic, red, parked in the air at load; per-episode spawn
        # by MESH BOTTOM in _initialize_episode.
        self.cube = actors.build_cube(
            self.scene,
            half_size=cfg.cube_half,
            color=list(cfg.cube_color),
            name="cube",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[float(centres[0, 0]), cfg.spawn_depth,
                                        cfg.shelf_top_z + 0.20]),
        )

        # The home marker: a visual-only disk on the floor (no collision shapes
        # — `add_collision=False`; kinematic, not static: the pose setter
        # asserts non-static under GPU sim, season_dish's marker note). A
        # cylinder's axis is x, so it is stood up by a 90-degree pitch.
        self.home_marker = actors.build_cylinder(
            self.scene,
            radius=cfg.home_radius,
            half_length=0.001,
            color=[0.95, 0.85, 0.1, 1.0],
            name="home_marker",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(
                p=[cfg.home_xy[0], cfg.home_xy[1], cfg.home_marker_z],
                q=euler.euler2quat(0.0, math.pi / 2.0, 0.0),
            ),
        )

    def _untangle_fixture_door(self, stem: str):
        """Make one fixture's DOOR touchable by clearing its ignore bit.

        This is a change to the SCENE's collision mask, applied to the named
        doors and to nothing else. `stem` must be a key of `UNTANGLE_BITS`,
        which says which bit that fixture carries; anything else raises.

        Why it is needed (W24, 2026-09-02, measured on cab_1).
        `_fix_ds_fetch_collision_bits` sets ignore bits 25-29 on every shape of
        the robot — the wheel/base exemption the RoboCasa scene builder skips
        for our uid. The kitchen's own masks put bit 26 on `floor_room`, the
        three walls, `counter_main`, `dishwasher`, `cab_3` — and on `cab_1`.
        Robot and cab_1 therefore share an ignore bit and are filtered out of
        contact with each other: the door is a ghost to this robot and only to
        this robot. Measured: a bar grasp staged over a 20-cell offset grid put
        the TCP 2-17 mm from cab_1's bar and closed the pads to 0.0000 on every
        cell, while the same code on cab_main's right bar held at 0.0261.

        Why THIS fix and not the other one. W24 ran both controls. Clearing bit
        26 on the ROBOT's 20 shapes also works, but the robot's mask is what
        every measured number in this repo was taken under. Clearing it on
        cab_1's 8 door shapes instead leaves the robot's mask and every other
        fixture's exemption untouched: the leaf then holds in the fingers at
        0.0262, `pull_hinge_arc(open_dir=-1)` opens it to -1.605 `why=target`,
        and the west wall becomes a real backstop — which is what makes the wall
        park a rule rather than a wish. Re-measured on the host 2026-09-02, with
        the four-compartment env as the control: the untangled leaf is refused
        from -2.10 (settling at -2.0526) while the same write in the default env
        goes straight through to -2.1000. The box itself keeps bit 26, so
        nothing about how cab_1's carcass sits in the room changes.

        What the edit does, read off both live builds (Mac, 2026-09-02): in
        `MikasaCabinetSearch-v0` this method is never called and cab_1's eight
        `hingedoor` shapes keep word 2 = 0x4000000; in
        `MikasaCabinetSearch5-v0` the same eight go 0x4000000 -> 0x0 (bit 26 was
        their only ignore bit) while the carcass's six shapes stay at 0x4200000
        (bits 26 and 21) and every robot shape stays at 0b11111 for bits 25-29.

        DOOR LINKS ONLY IS A CHOICE, and here is its price. A link whose name
        has no `door` in it keeps the exemption, so the fixture's CARCASS stays
        intangible: the robot can still drive its base straight through the
        untangled fixture's body, and a motion plan that refuses such a pose is
        refusing on the PLANNING world, not on physics (W24 run 1 parked cab_1's
        leaf at 2.003 by standing the base inside the west wall, which the
        planner then refused as `base_link<->wall_left_room`). The choice is
        deliberate: it changes exactly the surface a hand has to grasp, and
        leaves every measured base path in this repo alone. A fixture whose body
        must also become solid needs a second, separately measured rule — not
        this one.

        Refuses loudly rather than doing nothing: an unknown stem, a missing
        articulation, a fixture with no `*door*` link (`UNTANGLE_DOORLESS` —
        `sink_main_group` and `stack_4_main_group_2` are both in the table and
        both have none), or door shapes that do not actually carry the bit. A
        silent no-op here reads downstream as "the hand missed the handle" and
        cost W24 three probe runs before the mask was suspected at all.

        The first two of those are also refused OFFLINE by `validate()`, off the
        table, before a kitchen is loaded. The checks here are kept anyway and
        are not the same check: these read the LIVE scene, so they still catch a
        fixture whose links or masks moved under a table that says otherwise —
        which is the failure a stale table would actually produce.
        """
        assert stem in UNTANGLE_BITS, (
            f"untangle: {stem!r} is not a measured fixture; `UNTANGLE_BITS` has "
            f"{sorted(UNTANGLE_BITS)}. Add it only with a live-scene scan of "
            "which of bits 25-29 its shapes carry"
        )
        bit = UNTANGLE_BITS[stem]
        cleared = carried = 0
        for i in range(self.num_envs):
            key = f"{stem}_{i}"
            assert key in self.scene.articulations, (
                f"untangle {stem!r}: no articulation {key!r}; have "
                f"{sorted(self.scene.articulations)}"
            )
            for link in self.scene.articulations[key].links:
                if "door" not in link.name.lower():
                    continue          # the carcass keeps its exemption
                for body in link._bodies:
                    for shape in body.get_collision_shapes():
                        groups = shape.get_collision_groups()
                        carried += bool(groups[2] & (1 << bit))
                        groups[2] &= ~(1 << bit)
                        shape.set_collision_groups(groups)
                        cleared += 1
        assert cleared, (
            f"untangle {stem!r}: no collision shape on a link named *door* — "
            "W24 found 8 on cab_1 (link `hingedoor`); without them the bar "
            "stays intangible and the fixture cannot be handled"
        )
        assert carried == cleared, (
            f"untangle {stem!r}: {carried} of {cleared} door shapes carried "
            f"ignore bit {bit}. `UNTANGLE_BITS` says this fixture is exempt "
            "from the robot on that bit; if the live scene disagrees the table "
            "is stale — re-scan it rather than clearing a bit that is not there"
        )

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

    def _after_reconfigure(self, options: dict):
        # The layout built in __init__ lives on the CPU (no device yet); the latch
        # step compares it with door angles read from the simulation, which under
        # GPU sim are CUDA tensors — the second thing that kept `num_envs > 1` from
        # running here (2026-09-08; the first was the single parking pose). On the
        # CPU backend this is the same tensor on the same device.
        self.layout = compartment_layout(self.cfg.compartments, device=self.device)
        z = SearchLatches.zeros(self.num_envs, self.layout.n_compartments,
                                self._n_foreign, device=self.device)
        for key in TASK_STATE_KEYS:
            setattr(self, key, getattr(z, key))
        joints = self.agent.robot.active_joints_map
        self._finger_qpos_idx = [
            joints[j].active_index[0].item()
            for j in ("l_gripper_finger_joint", "r_gripper_finger_joint")
        ]
        return super()._after_reconfigure(options)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            cfg = self.cfg
            self.scene_builder.initialize(env_idx)
            self._restore_robot(env_idx)

            # EVERY kitchen articulation to its zero: the scene builder
            # restores only the robot (scene_builder.py:563-577), and episode
            # 2 would otherwise start wherever episode 1's contacts left the
            # hinges. Velocities too — a door still swinging at the cut would
            # carry momentum into the next episode. All zeros are inside the
            # W20-dumped limits (microjoint [-1.57, 0], sink handle [0, 0.52],
            # knobs unbounded).
            for i in env_idx.tolist():
                arts = {art for art, _ in self._hinges[int(i)]} | set(self._foreign[int(i)])
                for art in arts:
                    q = art.get_qpos()
                    q = q.clone() if torch.is_tensor(q) else torch.as_tensor(q).clone()
                    art.set_qpos(torch.zeros_like(q))
                    art.set_qvel(torch.zeros_like(q))
            # The foreign snapshot AFTER the zeroing — what "untouched" means.
            self.articulation_home[env_idx] = self._read_foreign()[env_idx]

            # The episode ticket, from the batched episode RNG (torch's
            # generator is only seeded when a seed is passed). Four
            # INDEPENDENT draws: the cube's jitter must not correlate with
            # the start's (the family's shared-jitter shortcut was a review
            # finding), and the instruction is its own draw.
            n_comp = self.layout.n_compartments
            cabs, cube_j, start_j, instr = [], [], [], []
            for i in env_idx.tolist():
                rng = self._batched_episode_rng[i]
                cabs.append(rng.randint(n_comp))
                cube_j.append(rng.uniform(-1.0, 1.0, size=2))
                start_j.append(rng.uniform(-1.0, 1.0, size=3))
                instr.append(rng.randint(len(INSTRUCTIONS)))
            cab_t = torch.as_tensor(np.asarray(cabs), dtype=torch.long)
            cj = torch.as_tensor(np.stack(cube_j), dtype=torch.float32)
            sj = torch.as_tensor(np.stack(start_j), dtype=torch.float32)
            self.cube_cab[env_idx] = cab_t
            self.instruction_idx[env_idx] = torch.as_tensor(np.asarray(instr), dtype=torch.int32)

            # The cube by MESH BOTTOM on the shelf, in its compartment's half.
            spawn = torch.zeros((b, 3))
            spawn[:, 0] = self._spawn_centre_x[env_idx, cab_t] + cj[:, 0] * cfg.spawn_jitter_x
            spawn[:, 1] = cfg.spawn_depth + cj[:, 1] * cfg.spawn_jitter_depth
            spawn[:, 2] = cfg.shelf_top_z + cfg.spawn_clearance + cfg.cube_half
            self.cube.set_pose(Pose.create_from_pq(p=spawn))
            self.cube_spawn[env_idx] = spawn.to(self.cube_spawn.device)
            self.cube.set_linear_velocity(torch.zeros((b, 3)))
            self.cube.set_angular_velocity(torch.zeros((b, 3)))

            # The robot start: south of home, outside the disk (validate()).
            base = torch.zeros((b, 3))
            base[:, 0] = cfg.start_xy[0] + sj[:, 0] * cfg.start_jitter_xy
            base[:, 1] = cfg.start_xy[1] + sj[:, 1] * cfg.start_jitter_xy
            base[:, 2] = math.radians(cfg.start_yaw_deg) + sj[:, 2] * cfg.start_jitter_yaw
            qpos = self.agent.robot.get_qpos()
            qpos[env_idx, 0] = base[:, 0]
            qpos[env_idx, 1] = base[:, 1]
            qpos[env_idx, 2] = base[:, 2]
            self.agent.robot.set_qpos(qpos[env_idx])

            n_c = self.layout.n_compartments
            self.opened_count[env_idx] = torch.zeros((b, n_c), dtype=torch.int32)
            self.is_open[env_idx] = False
            self.passed_home[env_idx] = False
            self.last_opened[env_idx] = -1
            self.reopened[env_idx] = False
            self.two_open[env_idx] = False
            self.skipped_home[env_idx] = False
            self.foreign_moved[env_idx] = False
            self.second_decision_made[env_idx] = False
            self.second_decision_ok[env_idx] = False
            self.retry_count[env_idx] = 0
            self.succeeded[env_idx] = False
            self.last_eval_step[env_idx] = -1
            self.seen_count[env_idx] = 0

    def _restore_robot(self, env_idx: torch.Tensor):
        """The rest keyframe for every robot uid (scene_builder only restores
        uid == "fetch", and to its own drawn dock)."""
        keyframe = self.agent.keyframes["rest"]
        qpos = torch.as_tensor(np.asarray(keyframe.qpos, dtype=np.float32))
        self.agent.robot.set_qpos(qpos.unsqueeze(0).repeat(len(env_idx), 1))
        self.agent.robot.set_root_pose(sapien.Pose())

    # ------------------------------------------------------------ sim reads --

    def _read_hinges(self) -> tuple[torch.Tensor, torch.Tensor]:
        """`(qpos, qvel)`, each `(n, H)`, in layout hinge order."""
        n, H = self.num_envs, self.layout.n_hinges
        q = torch.zeros((n, H), dtype=torch.float32, device=self.device)
        v = torch.zeros((n, H), dtype=torch.float32, device=self.device)
        for i in range(n):
            for h, (art, j) in enumerate(self._hinges[i]):
                q[i, h] = art.get_qpos().reshape(-1)[j]
                v[i, h] = art.get_qvel().reshape(-1)[j]
        return q, v

    def _read_foreign(self) -> torch.Tensor:
        """`(n, F)` qpos of every non-compartment kitchen joint, sorted by key."""
        rows = []
        for i in range(self.num_envs):
            arts = self._foreign[i]
            if arts:
                rows.append(torch.cat([a.get_qpos().reshape(-1) for a in arts]))
            else:
                rows.append(torch.zeros((0,), device=self.device))
        return torch.stack(rows).to(torch.float32)

    def _snap_hinges(self, snap: torch.Tensor):
        """The detent: qpos = qvel = 0 on the masked hinges ONLY (columns of
        the articulation — a compartment's snap never touches its neighbour
        door on the same box). Written from evaluate(), the repo's blessed
        slot for phase side effects."""
        for i in range(self.num_envs):
            for h, (art, j) in enumerate(self._hinges[i]):
                if not bool(snap[i, h]):
                    continue
                q = art.get_qpos()
                q = q.clone() if torch.is_tensor(q) else torch.as_tensor(q).clone()
                q.reshape(-1)[j] = 0.0
                art.set_qpos(q)
                v = art.get_qvel()
                v = v.clone() if torch.is_tensor(v) else torch.as_tensor(v).clone()
                v.reshape(-1)[j] = 0.0
                art.set_qvel(v)
        if self.gpu_sim_enabled:
            # UNVERIFIED on GPU (the repo's standing caveat): without the
            # apply/fetch pair the writes are dead on the GPU tier and the
            # detent silently does not happen (stack_recall's note).
            self.scene._gpu_apply_all()
            self.scene._gpu_fetch_all()

    def _base_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # agent.base_link, not agent.robot: Fetch drives through root joints
        # (fetch.urdf:21-35), so the articulation's root pose never moves.
        pose = self.agent.base_link.pose
        q = pose.q
        yaw = torch.atan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                          1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2))
        speed = torch.linalg.norm(self.agent.robot.get_qvel()[:, :3], dim=-1)
        return pose.p[:, :2], yaw, speed

    # ------------------------------------------------------------- the look --

    def _look_cameras(self) -> list:
        """(uid, mount pose) of the base cameras on the head link, from the agent's rig."""
        out = []
        for c in getattr(self.agent, "_sensor_configs", []) or []:
            if getattr(c, "entity_uid", None) == "head_camera_link" and "base_camera" in c.uid:
                out.append((c.uid, c.pose))
        return out

    def revealed_target(self) -> torch.Tensor:
        """The cube's position (n, 3) for the envs whose compartment is REVEALED, NaN
        elsewhere — what a policy could see, and the oracle's only sanctioned read of
        the answer (the source guard forbids the name `cube` in the planner). Read from
        the last `evaluate()`: `revealed` is what `step_search_latches` decided this step.
        """
        pos = self.cube.pose.p.clone()
        revealed = getattr(self, "_revealed_now", None)
        if revealed is None:
            revealed = torch.zeros(pos.shape[0], dtype=torch.bool, device=pos.device)
        pos[~revealed] = float("nan")
        return pos

    def _cube_seen(self) -> torch.Tensor:
        """Is the cube's centre inside `look_cone_rad` of a base camera's optical axis?

        Kinematic, no render: camera = head link pose * mount; the optical axis is the
        camera frame's +x (SAPIEN's convention; equal to the render params' forward,
        measured 2026-09-07). No occlusion test — the reveal condition already puts the
        leaf out of the way, and the head look keeps the arm out of the middle."""
        n = self.num_envs
        cfg = self.cfg
        cams = self._look_cameras()
        head = getattr(getattr(self.agent, "robot", None), "links_map", {}).get("head_camera_link")
        if not cams or head is None:
            return torch.ones((n,), dtype=torch.bool, device=self.device)
        cube_p = self.cube.pose.p
        seen = torch.zeros((n,), dtype=torch.bool, device=self.device)
        cos_cone = math.cos(cfg.look_cone_rad)
        for _uid, mount in cams:
            M = (head.pose * mount).to_transformation_matrix()
            pos, axis = M[:, :3, 3], M[:, :3, 0]
            v = cube_p - pos
            dist = torch.linalg.norm(v, dim=-1).clamp_min(1e-6)
            cos_a = (v * axis).sum(dim=-1) / dist
            seen = seen | ((cos_a >= cos_cone) & (dist <= cfg.look_range_m))
        return seen

    # -------------------------------------------------------------- evaluate --

    def evaluate(self) -> dict:
        cfg = self.cfg
        step = self.elapsed_steps.to(torch.int32)
        qpos, qvel = self._read_hinges()
        theta = hinge_theta(qpos, self._open_dir)
        still = qvel.abs() < cfg.door_still_vel
        tcp = self.agent.tcp_pose.p
        tcp_far = torch.linalg.norm(tcp.unsqueeze(1) - self._bar_xyz, dim=-1) > cfg.hand_far_m
        base_xy, base_yaw, base_speed = self._base_state()
        at_home = at_home_predicate(base_xy, base_yaw, base_speed, cfg)
        foreign_q = self._read_foreign()
        drift = (foreign_q - self.articulation_home).abs()
        if drift.shape[1]:
            foreign_hit = (drift > cfg.foreign_drift_tol).any(dim=-1)
            foreign_drift_max = drift.amax(dim=-1)
        else:
            foreign_hit = torch.zeros_like(at_home)
            foreign_drift_max = torch.zeros_like(base_speed)

        info, snap = step_search_latches(
            self, step=step, theta=theta, hinge_still=still, tcp_far=tcp_far,
            at_home=at_home, foreign_hit=foreign_hit, layout=self.layout, cfg=cfg,
            cube_seen=self._cube_seen(), base_static=base_speed < cfg.base_static_speed,
            # horizontal only: the spawn is `spawn_clearance` (2 cm) above the shelf and the
            # settle drop alone would clear the threshold (the first smoke: 10/10 "nudged"
            # with no push at all)
            moved_m=torch.linalg.norm((self.cube.pose.p - self.cube_spawn)[..., :2], dim=-1),
        )
        self._revealed_now = info["revealed"]  # read by revealed_target()
        if bool(snap.any()):
            self._snap_hinges(snap)
        info["foreign_drift_max"] = foreign_drift_max
        # Mirrors the sim write: the obs sees exactly what the detent left.
        info["door_qpos"] = qpos.masked_fill(snap, 0.0)
        # Per-env scalars are the sweep keys (`--info-keys`: revealed,
        # reopened, two_open, skipped_home, foreign_moved, n_decisions,
        # second_decision_*, retry_count, foreign_drift_max); is_open (n, N),
        # door_theta_max (n, N) and door_qpos (n, H) are rank-2 diagnostics
        # for traces and the Mac smoke — evaluate_planner's tally folds only
        # scalars (`test_sweep_info_keys_are_per_env_scalars`).
        return info

    # ------------------------------------------------------------------- obs --

    def _get_obs_extra(self, info: dict) -> dict:
        """What the policy may see. Fixed key order; NEVER the cube (its pose
        IS the answer), never `cube_cab`, never a counter or a latch. Under
        use_state only the four hinge angles — after the detent a closed
        door carries no history."""
        qpos = self.agent.robot.get_qpos()
        obs = dict(
            tcp_pose=self.agent.tcp_pose.raw_pose,
            base_pose=self.agent.base_link.pose.raw_pose,
            gripper_open=qpos[:, self._finger_qpos_idx].sum(dim=-1),
        )
        if self.obs_mode_struct.use_state:
            obs.update(door_qpos=info["door_qpos"])
        return obs

    def get_language_instruction(self, **kwargs):
        texts = INSTRUCTIONS_NUDGE if self.cfg.terminal == "nudge" else INSTRUCTIONS
        return [texts[int(i)] for i in self.instruction_idx.tolist()]

    # ----------------------------------------------------------------- state --

    def get_state_dict(self) -> dict:
        state = super().get_state_dict()
        for key in TASK_STATE_KEYS:
            state[key] = getattr(self, key).clone()
        return state

    def set_state_dict(self, state: dict, env_idx: torch.Tensor = None):
        """No task key present -> simulator state only (BaseEnv.set_state(flat)
        rebuilds a dict with just actors/articulations, sapien_env.py:1309);
        some but not all -> ValueError, because a half-restored memory would
        replay as a different episode. Numpy-tolerant via restore_task_tensor.
        Full-buffer restore (env_idx ignored for the task tensors), like every
        task here."""
        present = [k for k in TASK_STATE_KEYS if k in state]
        if present:
            for k in TASK_STATE_OPTIONAL:            # a recording from before the key
                if k not in state:
                    state = dict(state); state[k] = torch.zeros_like(getattr(self, k))
            present = [k for k in TASK_STATE_KEYS if k in state]
        if present and len(present) != len(TASK_STATE_KEYS):
            missing = [k for k in TASK_STATE_KEYS if k not in state]
            raise ValueError(
                f"partial task state: {present} present, {missing} missing — "
                "a half-restored search memory replays as a different episode"
            )
        super().set_state_dict(state, env_idx)
        for key in present:
            setattr(self, key, restore_task_tensor(
                getattr(self, key), state[key], self.device
            ))
