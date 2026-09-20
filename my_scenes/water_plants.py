"""Water every plant exactly once — the first task here whose answer does not exist at t = 0.

A bucket of water stands on the floor in the middle of the room with a cup beside
it. The robot picks the cup up, fills it, drives to one plant on the counter,
empties the cup over it, and drives back for the next dose. Every plant must be
watered, and none of them twice.

**Why this is a memory task, and how it differs from the other four.** There is no
cue phase. At t = 0 the set of watered plants is *empty*; it is filled only by the
agent's own actions, and nothing in the scene records them. The construct is
`action_history` — the same label `MikasaStationChecklist-v0` carries — but the
measured object is different: StationChecklist hands the answer over as beacons in
an encoding phase and measures *retention* of an externally given set, while here
there is nothing to retain at t = 0 and what is measured is *accumulation* of a
self-generated one. A failure here cannot be confused with a failure to perceive a
cue, because there is no cue.

Two escapes had to be closed, and both are structural rather than cosmetic
(`docs/task-designs/F-waterplants-v1.md`, holes A and B):

- **A watered plant must be indistinguishable from a dry one.** There is no fluid
  simulation in SAPIEN here, so "watering" is a *pose* predicate, as "seasoning" is
  in `season_dish.py:20`. The plants are `body_type="kinematic"` so a brush of the
  cup's rim cannot nudge one into becoming a bookmark, all of them are the same
  asset variant, and the pour predicate never touches the pot. Add any trace and a
  memoryless "water the nearest dry one" policy scores 100 %.
- **The robot's own pose must not work as the counter.** This is the harder one. A
  memoryless sweep — "water the first plant to my right, then move right" — reads
  the whole task state off one frame, because the base pose *is* the progress
  counter. The closure is the forced return: the cup holds one dose, so after every
  plant the robot must come back to the bucket, and the refill only counts with the
  base in the station dock and the arm back in its carry pose. At the start of every
  trip the observation is therefore **uninformative up to the predicates'
  tolerances** whether nothing, one plant or two have been watered.

  "Up to the tolerances" is not a hedge, and the earlier wording here said
  "identical", which was an overclaim in the one sentence the whole task rests on.
  The refill admits the base anywhere within `dock_radius = 0.20 m` of the dock and
  any heading within `dock_heading_deg = 35°` of it, and the arm anywhere within
  `carry_pose_tol_rad = 0.15` of the rest keyframe. **The residual channel is real
  and is disclosed rather than papered over:** 70° of heading freedom on a
  `base_pose` this task *emits* is somewhere a policy could park its progress —
  stand at +30° with no plant done, −30° with one — and read its own heading back as
  the counter. That is the "body as memory" hole the refill exists to close,
  squeezing back through a tolerance. It is not closed here, and the reason is that
  closing it means tightening numbers that hold the motor stack together, which
  cannot be done without measuring. A scripted oracle parks the same way every trip
  so it will not exercise the channel, and a blind run will not find it either; an
  RL policy could. Recorded in the design form's self-check row as well
  (`docs/task-designs/F-waterplants-v1.md`, «тело вместо памяти»).

Memoryless floor, stated so a success rate means something
----------------------------------------------------------
A policy with no memory draws uniformly *with replacement* from the three plants on
each trip, so it waters all three exactly once with probability

    3! / 3**3  =  6 / 27  =  0.222

and the **episode floor is 0.222 x the motor success rate**, the motor rate being what
the oracle's sighted sweep reports. **The floor rests on N = 3 alone.** `k_sites`
changes the number of arrangements — C(5,3) = 10 on the adopted layout B, C(6,3) = 20
on the rejected layout A — and **not the floor**, which is worth saying out loud
because the count and the floor look like the same kind of number and are not. Unlike
`MikasaStationChecklist-v0`, where C(6,3) *is* the answer, the arrangement here is a
row of the episode ticket: the answer does not exist at t = 0 and depends on `k_sites`
not at all. The site row has been cut from six to five and the station relocated
twice since the design was written, and 0.222 has not moved once.

A second, graduated endpoint is mandatory at ten seeds, where 0.222 is under two
expected hits and drowns in binomial noise: `distinct_before_repeat` (blind
distribution {1: 1/3, 2: 4/9, 3: 2/9}, mean 1.89) and `first_decision_ok` (blind 2/3;
the *second* pour, because the first cannot be wrong when every plant is dry).

Measured 2026-08-22, both arms on one commit, in the amd64 CPU container on kitchen
102 over eval seeds 0-9 (`docs/task-cards.md` card 6 and `docs/lab-journal.md` carry
the provenance and the caveats):

    sighted   8/10   (missed 0, no plan 2)   exact_cover 8/8, distinct mean 3.000
    blind     0/10   (missed 10, no plan 0)  exact_cover 0/10, distinct mean 1.600
    floor     0.222 x 8/10 x 10  =  1.8 expected hits          -> blind got 0

So the blind arm is **below** its floor rather than on it, and it is not a leak: the
leak signature is a blind arm at or above the sighted one, and 0/10 against 8/10 is
the other end of the scale. Three of the ten blind episodes never reached a second
decision at all — they were lost to a drive that stopped short of the plant's dock
zone — and over the seven that did run three pours `distinct_before_repeat` averaged
1.857 against the design's 1.889 — **the mean agreed; the shape of the distribution
is not testable at n = 7** — and no permutation came up at all, which at p = 2/9 has
probability (1 - 2/9)**7 = 0.17.

Geometry is measured, not derived on paper: `tools/probes/w1_sites_and_reach.py` on
the assembled kitchen 102, written up in the plan's `geometry-decided.md`. Nothing
below recomputes it. The adopted layout is **B**: five sites over the two free spans
right of the sink. The station has since moved **twice**, both times on a measurement,
and the current pair is the bucket at (3.1500, −2.1900) with the refill dock at
(2.9619, −1.6732) on the counter side of it, 0.55 m out — **`WaterPlantsConfig.station_xy`
and `refill_dock_xy` are the source of truth and carry the argument**, and this paragraph
is a summary of them rather than a second place to maintain. The superseded pairs
(2.7935, −1.7563) / (2.7935, −2.5563) and (2.8614, −1.7068) / (2.8614, −2.5068) still
appear in the design form, which keeps its own history on purpose. What that form marks
is enumerated there rather than claimed here: `F-waterplants-v1.md`, Changelog v1.3
(the v1.2 pair, blanket) and v1.4 item 2, which lists the five places the final review
found still reading as live — the deviations section's items 4 and 5, the alternatives
row, the ticket row and the state table. **One** section is deliberately not corrected
and is labelled as such: «Принятые решения» is the measurement protocol for the rejected
layout A. § 2 used to be the other one, and this paragraph went on saying so after it had
stopped being true: v1.3 item 2 rewrote § 2 to carry the live pair, and its layout-A
numbers survive there only as explicitly-marked history. Corrected 2026-08-22 in the
form's own changelog (v1.5) as well as here — a stale claim about the form, inside the
docstring that exists to keep the form's claims straight.

Layout A's sixth site stood against
the room's left wall, where a drive-in is exactly what `MikasaStationChecklist-v0`
already met a refusal on, and stretching the retention interval is the kitchen
tier's job rather than the site row's.

Conventions: `docs/writing-tasks.md`. Structural reference: `template_task.py`; the
pour predicate is `season_dish.py:605-629`, the exactly-once bookkeeping is
`station_checklist.py:786-894`, and the pure-verdict-function shape is
`burner.py:310-349`.
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
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.robocasa.scene_builder import RoboCasaSceneBuilder
from mani_skill.utils.structs import Actor, Pose

from utils.mikasa.scenes.robocasa_utils import (
    parking_pose,
    counter_frame,
    dock_pose_for,
    fixture_frame,
    load_accessory_actor,
    load_objaverse_actor,
    require_get_fixture,
    restore_task_tensor,
)

# Where an unoccupied site's plant is parked. MIKASA-Robo's constant
# (memory_envs/chain_rule_transfer.py:47) and idiom (remember_color.py:203), reached
# here through season_dish.py:63 and burner.py:149.
HIDDEN_Z = 1000.0

# `spikey_in_white_pot` reads as "a plant in a pot" better than the other three, and
# the four accessories are one asset *set*, not a per-plant variant: all N plants in
# an episode are the same asset, or "wilted vs fresh" becomes a texture cue for
# "not watered yet" (design, hole A). The registry's fourth key is misspelled on
# both sides — see `robocasa_utils.load_accessory_actor`.
PLANT_REL_XML = "fixtures/accessories/plants/spikey_in_white_pot"

#: The water surface drawn inside the bucket (`WaterPlantsConfig.water_visual`).
#: Inset from the rim so it reads as water in a vessel; thin enough that it is a
#: surface and not a block; and a colour that is water rather than paint.
WATER_COLOR = (0.05, 0.28, 0.45, 1.0)
WATER_INSET = 0.012
WATER_HALF_THICK = 0.002

#: The station stand (K60), when `station_z_lift` > 0. Colour only — a plain matte grey
#: so it reads as a utility stand and not as another piece of the kitchen's cabinetry.
STAND_COLOR = (0.42, 0.42, 0.44, 1.0)
#: Metres of stand surface beyond everything standing on it, so a cup jittered by
#: `cup_jitter_xy` still has surface under it with room to spare. Kept small on
#: purpose — the robot parks 0.75 m away and every centimetre here is a centimetre
#: closer to its base. Not a cosmetic
#: border: a cup that spawns over the edge falls to the floor and the episode is lost
#: before it starts. (`base_cup_min_clearance` does not enter — that guard moves the
#: *base* away from the cup, never the cup.)
STAND_MARGIN = 0.08

# No digits and no number words. "Water all three" would hand a language-conditioned
# policy the size of the answer space; "twice" and "a second time" are numerals too,
# so the prohibition is expressed as "no repeats" (burner.py:157 is the rule).
INSTRUCTIONS = (
    "Water every plant. No repeats.",
    "Every plant must get water. Do not water any plant again.",
    "Give water to all the plants, without repeating any.",
)

# How far the arm has been measured to plan a dip from, in metres. **A measured
# ceiling, not a swept envelope**: 0.67 m planned and 0.80 m was refused with the screw
# converging 11.2 cm short, both in the amd64 CPU container on kitchen 102. FK says the
# arm reaches 0.980 m at z = 0.25 m with the torso down, and that number is deliberately
# NOT used here — reachability under forward kinematics is not planability.
#
# **K60: that ceiling was a ceiling at z = 0.25 m, and the dip is no longer there.**
# Every number above was measured with the bucket on the floor, where W1's reach table
# runs 0.980 m at torso 0.000 down to 0.549 m at torso 0.386 — the arm is at the edge of
# its envelope and 0.80 m falls off it. With `station_z_lift = 0.45` the dip sits at
# z ~ 0.72, where the same table reads 1.021-1.126 m across the *whole* torso range. The
# cap is raised to 0.85 to admit the 0.75 m the raised station needs, and 0.75 is not
# taken on the table's word: `logs/raise*` is the dip planning at that distance and
# height, and if it had not planned this constant would have gone back down rather than
# the evidence being explained away.
MAX_DIP_DISTANCE = 0.85

# What `get_state_dict` adds on top of the sim state: the task's entire memory.
# `trip_index` is here and is deliberately NOT in `_get_obs_extra` — see that
# method's docstring.
TASK_STATE_KEYS = (
    "watered_mask",
    "occupied_mask",
    "has_water",
    "budget_left",
    "trip_index",
    "refill_dwell",
    "refill_timer",
    "dip_done",
    "pour_hold",
    "double_water_count",
    "distinct_before_repeat",
    "repeat_seen",
    "first_decision_made",
    "first_decision_ok",
    "scene_untampered",
    "instruction_idx",
    "plant_home",
    "articulation_home",
    "last_eval_step",
)


@dataclass
class WaterPlantsConfig:
    """Every threshold the task uses, named once, so a failed run can say which missed.

    A literal inside `evaluate()` is how a failed episode becomes unattributable.
    Thresholds are config and are restored with the code; per-episode draws and
    latches are checkpoint state and are restored from the state dict (`TASK_STATE_KEYS`).
    """

    horizon: int = 5200
    """Episode length. Must equal `max_episode_steps` in the decorator.

    **Measured (K22), not guessed.** The shipped placeholder was 2400, set before
    anything drove, and it was low by more than a third. The formula is the same one
    the other three tasks used — the steps to the *last* verdict of a successful demo
    on **out-of-eval calibration seeds**, times 1.5 — and this task has neither a cue
    phase nor a pause, so it has none of their addends:

        seed 11  success=True at t = 3257
        seed 12  success=True at t = 3433
        max(3257, 3433) = 3433  ×1.5 = 5149.5 → 5150 (to 50) → **5200** (to 100)

    Seeds **11 and 12**, never 0-9: the sweep seeds are where the memoryless floor is
    measured, and tuning a horizon on them is the sin `MikasaBurner-v0`'s card admits
    to (its 800 and its 0.25 m dock were fitted on seeds 3 and 1, both inside its own
    eval set). The two runs above are bit-identical on re-run, so one run per seed is
    the whole measurement — but that claim is **one seed, one commit, one machine and
    image**, and is not to be quoted wider.

    Three round trips across a kitchen is the longest motion schedule in the
    portfolio, and **~1260 of these steps are a solver defect rather than the task**.
    `move_base_forward` returns `Success` having stopped 0.37-0.65 m short, so three
    of the six plant drives take a second rung at ~280 steps each — but those ~840
    steps were measured *inside* the 3433-step calibration demo, and the demo is what
    gets multiplied, so the defect propagates through the ×1.5 and costs **1260** of
    this horizon rather than 840. Fixing it would take the demo to ~2593 and this
    horizon to ~3900: a drop of ~1300. The defect is filed separately; until it is
    fixed the horizon carries it, and by half again as much as the raw count suggests.

    **This number expires when the solver does.** A refused plan costs no control
    step but still advances the planner's generator, so **an edit that changes the
    number of planner draws taken *before* a later draw re-phases every plan after
    it.** That is the established form and it is narrower than it first looks: the
    wide version — "any edit to the retry ladder moves the horizon" — is **untrue**,
    because a rung nobody reaches takes no draws, and neither does a rung below the
    winning one. What does invalidate this measurement wholesale is a change of solver
    or of mplib, which re-phases from the first plan onward; the horizon must then be
    derived again.
    """

    # --- the puzzle ----------------------------------------------------------
    n_plants: int = 3
    """Plants per episode. A family constant, not a knob: the 0.222 floor is 3!/3**3.

    Also the bucket's capacity in doses and the length of the `refill_dwell` buffer.
    """
    k_sites: int = 5
    """Candidate sites the plants are drawn into. Sets C(5, 3) = 10 arrangements.

    Measured (Task 2, layout B): five sites at 0.3104 m of arc across the two right
    free spans, 0.1004 m of pot-to-pot gap. **The memoryless floor does not move
    with this number**: 3!/3**3 = 0.222 is set by `n_plants` alone, and unlike
    StationChecklist — where C(6,3) = 20 *is* the answer — the arrangement here is a
    row of the episode ticket and not the answer at all. The answer is generated by
    the agent during the episode and does not depend on `k_sites`.

    `k_sites` alone does **not** choose the layout, and a previous version of this
    comment wrongly implied it did: the *spans* are chosen by `site_spans` below.
    Layout A (six sites over all three spans, step 0.3067 m) is that field plus this
    one, and the pair is what the test suite pins.
    """

    # --- where the plants can stand ------------------------------------------
    site_spans: tuple = (
        ("counter_main_main_group", 1),
        ("counter_right_main_group", 0),
    )
    """Which derived free span of which counter carries sites, in order.

    `(exact fixture name, index of that counter's free span counting from the low
    along end)`. The spans themselves are still **derived from the assembled
    kitchen** by `_free_spans` — the sink's extent is measured, not declared — but
    *which* of the derived spans the task uses is a decision, and decisions belong in
    the config rather than in a threshold that happens to exclude one of them.

    Layout B, adopted after the probe's block 6 (`geometry-decided.md`): span 1 of
    `counter_main` is the surface right of the sink, and `counter_right` has a single
    free span. Span **0** of `counter_main` — the 0.542 m left of the sink, whose
    single site would sit at x = 0.375 — is deliberately unused: it stands against
    the room's left wall at x = 0, and that is the wall where
    `MikasaStationChecklist-v0` already met a refused approach (`docs/task-cards.md`,
    card 3). Standing there is fine and was measured (0.284 m of arm clearance,
    0.091 m of base clearance); the *drive in* is what nothing on this machine can
    settle, and layout B removes the risk rather than mitigating it. The trip lengths
    also even out, 1.404 max/min against 2.032.

    Layout A is one line: put `("counter_main_main_group", 0)` first and set
    `k_sites = 6`.

    Exact fixture names, never substrings: `scene_builder.get_fixture` resolves an
    ambiguous substring with `self.env._episode_rng.choice` (scene_builder.py:656),
    which both randomises the pick and shifts the stream the ticket is drawn from.
    """
    site_across: float = -0.200
    """Offset from the counter's top centre along its `across` unit vector, metres.

    Measured, and it is the *only* depth that works on kitchen 102 (Task 2): the wall
    cabinets hang down to z = 1.390 and forward to across −0.400, so the band with
    open sky above it is across ∈ [−0.650, −0.400], 0.250 m deep. −0.200 from the
    counter's centre (across −0.325) puts the pot at across −0.525, the middle of
    that band, where a 0.21 m pot keeps 0.040 m of slack on each side. A plant under
    a cabinet leaves 0.17 m of headroom and cannot be poured into at all.
    """
    pot_xy: float = 0.21
    "Pot footprint, `spikey_in_white_pot` at scale 0.3 (Task 3 measured 0.211 x 0.206)."
    pot_height: float = 0.30
    """Plant height at scale 0.3 (Task 3 measured 0.300).

    Used only to decide what counts as a *surface obstacle* when the free spans are
    derived: something whose bottom is above a pot's head is a wall cabinet, not a
    thing in the pot's way. The pour predicate reads the pot's real mesh top instead.
    """
    pot_edge_clear: float = 0.02
    "Extra gap between a pot and the end of a free span, on top of half a pot."
    site_jitter_along: float = 0.0251
    """Independent per-pot jitter along the counter, metres. A quarter of the gap.

    Not half of it. Two neighbours each draw their own jitter and close the gap by
    **2j**, so +-gap/2 is the *touching* limit and +-gap/4 is the budget: on layout
    B's 0.1004 m gap that is +-0.0251 m, leaving 0.0502 m between neighbours in the
    worst case. The correction is recorded in the plan's `geometry-decided.md`; the
    first version of that file, and the probe's own printed line, both doubled it.
    (Layout A's 0.097 m gap gives +-0.0242 m by the same arithmetic.)
    """
    site_jitter_across: float = 0.020
    """Independent per-pot jitter across the counter, metres.

    Half of the arithmetic above is different here and the difference is worth saying
    out loud: across the counter a pot moves against the *fixed* edges of the
    open-sky band (0.250 m band, 0.21 m pot, 0.040 m slack), not against a neighbour
    that moves too, so closure is j rather than 2j and +-0.020 m is a real budget.
    """
    plant_rel_xml: str = PLANT_REL_XML
    plant_scale: float = 0.3

    # --- the water station ----------------------------------------------------
    station_xy: tuple = (3.2184, -2.3779)
    """World xy of the bucket, on the floor. Measured, and **relocated** (probe block 7).

    Two criteria, and the second one was missing for a round.

    *Every site must be out of the arm's reach from the bucket*, or a plant could be
    watered without driving and the trip — and with it the holding interval this task
    measures — would collapse. The nearest site is **1.781 m** away, against an
    `fk_envelope` of 1.10. (The geometric middle of the room, (2.750, −1.520), is
    refuted on exactly this: two sites fall inside the envelope from there.)

    *The robot must be able to **drive** from the refill dock to every plant dock*, and
    nobody asked that until `MikasaWaterPlants-v0`'s oracle could not. The old pair —
    bucket (2.8614, −1.7068), dock 0.80 m behind it toward the room — put the bucket
    **between the dock and the counters**, so every leg outward passed it. `drive_base`
    turns toward its target and moves in a straight line, so the perpendicular distance
    from a leg to the bucket is pure geometry:

        plant   0       1       2       3       4
        old     0.456   0.365   0.231   0.429   0.498     two of five below the band
        new     0.550   0.550   0.550   0.550   0.550     none

    The forbidden band is **0.4132-0.4156 m** between centres, and it is measured, not
    assumed: the Fetch base hull reaches 0.2876 m in xy, but the bucket is a shallow
    bowl only **0.0736 m** tall, so only the part of the base below that height can
    touch it and there the radius is 0.2853 m. The oracle's observed boundary — 0.3645
    colliding, 0.456 driving — brackets that band, so the number predicts the refusals
    rather than being fitted to them.

    The feasible region is broad rather than a knife edge (110 598 valid combinations
    over 2 414 bucket positions, x in [0.200, 5.250], y in [−2.840, −0.990]); this point
    is the one that minimises the longest leg, which falls from 1.6559 m to 1.2008 m.
    """
    refill_dock_xy: tuple = (2.9619, -1.6732)
    """World xy of the refill dock: `dip_distance` = 0.55 m from the bucket, facing it.

    The dock is now on the **counter side** of the bucket and looks back at it, which is
    what takes the bucket out of every outbound leg (see `station_xy`). Swapping the two
    costs nothing in turning: in-place rotation is exactly 360° per trip in *every*
    arrangement, because the four headings close a loop and
    `|a| + |a+180| = |b| + |180−b| = 180` whatever a and b are. That is an assert in the
    probe now, not prose.

    0.55 m, not the old 0.80 m, and the reach is still the binding constraint: the dip
    is inside the arm's forward envelope at the bucket's height **only with the torso
    down** — reach at z = 0.25 m falls from 0.980 m (torso 0.000) to 0.549 m (torso
    0.386). "Lower the torso to fill" is a property of the task, not a taste of the
    oracle's. What has changed is the margin: 0.80 m was measured to refuse (the screw
    plan converged 11.2 cm short), 0.67 m to work, and 0.55 m is inside both.
    """
    dock_yaw_deg: float = 290.0
    """Heading of the refill dock, in degrees. **Points at the bucket**, and `validate()`
    checks that it does — this is the one dock in the task that does not face a counter.
    """
    fk_envelope: float = 1.10
    "Arm reach at hover height (`tools/probes/t3_fk_envelope.py`). Every site must be outside it."
    cup_gap: float = 0.02
    "Clear space between the bucket's and the cup's meshes at spawn."
    cup_jitter_xy: float = 0.02
    "Cup spawn jitter. `season_dish.py`'s `spawn_jitter_xy`; the cup's own has never been measured."
    spawn_clearance: float = 0.02
    "How far a mesh *bottom* starts above the surface it will settle onto (season_dish)."
    start_jitter_xy: float = 0.08

    base_cup_min_clearance: float = 0.33
    """Metres the robot's spawned base centre must clear the cup's centre.

    **The base knocks its own cup over before the arm has moved, and nothing checked.**
    The opening manoeuvre parks at the refill dock with a pure *in-place* rotation —
    deliberately, so the base cannot advance into the bucket — and that rotation can be
    ~280 deg (measured, seed 25: -137.8 then +140.8 to net 3 deg). The base hull is not
    circular: `w1_sites_and_reach.py:165-170` measures `base_link` at 0.2876 m, estop
    0.2827, wheels 0.232, so a spin sweeps a fat sector through anything inside 0.2876 m
    plus the cup's ~0.039 m half-extent.

    Measured over seeds 0-29 (K58): every cup at **>= 0.3232 m** from the base centre was
    untouched; **0.3159 m** was flattened — seed 25, shoved 0.1016 m and toppled to 86.3
    deg, after which all four `GRASP_DEPTHS` refused (24 of 24 draws) because the ladder
    is keyed to an upright cup's AABB top and that had collapsed 0.0964 -> 0.0534 — and
    **0.3041 m** was nudged 0.0258 m and survived by luck (seed 16). The boundary is
    therefore between 0.3159 and 0.3232; 0.33 takes it with margin.

    `start_jitter_xy` is 0.08 m against a nominal base-to-cup clearance of only ~0.364 m,
    so the draw alone can breach this. `validate()` has always guarded the dock against
    the **bucket** (`STATION_NOGO`); the **cup** had no such guard. Rather than shrink the
    jitter — which would narrow the start distribution the task deliberately randomises —
    the reset pushes the base radially out to this clearance only when the draw breaches
    it."""

    start_jitter_yaw: float = 0.10
    "Base start jitter at the refill dock. StationChecklist's viewpoint jitter, same purpose."

    # --- docks ----------------------------------------------------------------
    dock_radius: float = 0.20
    dock_heading_deg: float = 35.0
    "D5 rung, `StationChecklistConfig`: the burner oracle parks within 0.008 m (K26)."
    min_dock_separation: float = 0.45
    """How far the refill dock must stand from every plant dock, in xy.

    **Read the next paragraph before reusing this number.** In
    `StationChecklistConfig` the identically-named field is checked between *every
    pair* of station docks, because there a commitment has to name one station and
    overlapping zones would make it ambiguous. That rule cannot hold here and must
    not: the measured site spacing is 0.3067 m, so three adjacent pairs of plant
    docks are closer than 2 x `dock_radius` and their zones **do** overlap by design.
    Nothing is ambiguous anyway — a commitment here also requires the cup within
    `pour_xy_radius` = 0.10 m of a pot centre, and 0.3067 > 2 x 0.10, so the cup can
    be over at most one pot. What this field does check is the separation that would
    genuinely break the task: the refill dock must not sit inside a plant's zone, or
    a trip could be served without moving. Measured slack there is large (the nearest
    plant dock is 1.16 m away).
    """
    base_static_speed: float = 0.08
    "Base speed below which the robot counts as parked, as in StationChecklist."

    # --- the pour --------------------------------------------------------------
    pour_xy_radius: float = 0.10
    pour_min_clearance: float = 0.05
    pour_max_clearance: float = 0.30
    "Cup origin above the pot's top. The pour is non-contact; the pot must not be touched."
    pour_tilt_deg: float = 55.0
    pour_axis_body: tuple = (0.0, 0.0, 1.0)
    """The cup's own axis that points up when it stands upright.

    **Measured, not assumed** (Task 4, `diagnose_task --task pour-axis`): world-up
    component 1.00000 at rest on nine runs across four build configs, corroborated by
    the mesh's body-frame AABB (7.30 x 7.30 x 11.50 cm, so the symmetry axis is Z)
    and by the origin resting half of the *z* extent above the surface rather than
    half the cross-section. A rotation matrix alone cannot settle this: objects spawn
    with an identity quaternion, so `R` at t = 0 is the identity by construction.
    """
    hold_steps: int = 15
    "Consecutive steps the pour pose must hold, as in season_dish. ~0.75 s."
    grasp_min_force: float = 0.5

    # --- the refill ------------------------------------------------------------
    refill_dwell_range: tuple = (20, 60)
    """Per-trip dwell, drawn on reset for all `n_plants` trips at once.

    A constant dwell would make `elapsed_steps` a usable trip counter (primer
    principle 4). Drawn once per episode into an `(n,)` buffer rather than trip by
    trip, so the RNG consumption of an episode does not depend on how many trips the
    agent managed — the same reason `station_checklist.py:576-589` draws one
    permutation instead of several independent throws.
    """
    carry_pose_tol_rad: float = 0.15
    """Per-joint tolerance on the carry pose the refill dwell is served in.

    **Subject to calibration on the first run**, exactly as written in the brief: no
    oracle has driven this pose yet, so nothing has measured how tightly a planner
    can hold it while gripping the cup.
    """

    water_visual: bool = True
    """Draw the water the task is about: a surface inside the bucket.

    Believability, and it was a real hole — the task is *watering plants* and until
    now nothing in the scene was water. Read off a rendered frame: the robot dipped
    an empty cup into an empty dish and tipped nothing onto a plant, and the "water
    bucket" (a `bowl` at RoboCasa's scale 2.0) reads as a red plate on the floor.

    **It cannot leak the answer**, which is the constraint any visual change here has
    to clear (`tools/probes/w8_watered_is_invisible.py`): this surface is built once,
    static, collision-free, and its pose and colour depend on nothing but the
    station's own geometry. `watered_mask` is not consulted, so a watered plant still
    renders byte-for-byte identically to a dry one. It is deliberately in the *source*
    and not on the plants for exactly that reason.

    Off restores the previous appearance for anyone re-measuring a published RGB
    number, since this does change what an `obs_mode="rgb"` policy sees."""

    water_fill: float = 0.72
    """Where the surface sits, as a fraction of the bucket's own mesh height."""

    cup_starts_full_grip_open: float = 0.05
    """Finger opening the cup is placed inside when `cup_starts_full`, in metres."""

    cup_starts_full: bool = False
    """Start the episode with the cup already in the gripper and already full.

    K55, opt-in, and the largest reduction in hand rotation available anywhere in
    the task — because it deletes the errand that costs it. Measured on the shipped
    task, the refill errand is `grasp the cup` 185 + `hover over the bucket` 178 +
    `fold to the rest keyframe` 547 of a 2198 deg episode, and all three exist only
    because the cup and the water are on the **floor**: the arm has to go down to
    fetch them and come back up, three times over.

    What it removes from the *task*, and it is not nothing: there is no bucket trip,
    so `trip_index`, the refill dwell and the carry predicate stop doing anything,
    and the episode becomes "water each of these plants exactly once, from a cup you
    are already holding". **The memory demand survives** — the agent still cannot see
    which plants it has watered and still has to hold that set across the drives —
    but the holding interval is plant-to-plant, and the task no longer measures
    anything about the refill. Treat it as a motion-economy variant of the task, not
    as the task; do not compare its success rate to a published one.

    Default False: the shipped benchmark is unchanged.
    """

    doses_per_fill: int = 1
    """Plants one fill of the cup can water before the cup must be refilled.

    K55, opt-in. 1 is the shipped task: one trip to the bucket per plant, which is
    what makes `trip_index` a trip counter and what the refill dwell is timed
    against. Raising it to `n_plants` collapses the three refill cycles into one and
    is by far the largest single reduction available in hand rotation — `fold to the
    rest keyframe` and `hover over the bucket` together are 1851 deg of a 2557 deg
    episode, and two thirds of both are the second and third refills.

    **The memory demand survives it**, which is why it is offered at all: the agent
    still drives between plants, still cannot see which are already watered, and
    still has to hold the set it has done. What changes is the *interval* over which
    it holds — plant-to-plant instead of plant-to-bucket-to-plant — so the floor
    (`3!/3^3`) is unaffected but the task gets easier in a way that has not been
    measured. Do not compare a sweep at 3 against a published number at 1.
    """

    station_z_lift: float = 0.45
    """Metres the water station is raised above the floor. 0.0 = on the floor.

    K55. The refill is the arm's most expensive errand in the task, and the reason
    is height, not distance: the bucket's mesh bottom sits on the floor, and
    `refill_dock_xy` already records that "the dip is inside the arm's forward
    envelope at the bucket's height **only with the torso down**" — reach at
    z = 0.25 m falls from 0.980 m at torso 0.000 to 0.549 m at torso 0.386. So every
    trip drives the whole arm down to the floor and hauls it back, three times an
    episode, and `hover over the bucket` plus `fold to the rest keyframe` are 1851 of
    2557 deg of hand rotation between them.

    Raising the bucket is available for nothing because it is **kinematic** — see
    `_initialize_episode`, which places its mesh bottom exactly rather than dropping
    it — so a lift needs no support geometry and no physics. It is still a change to
    the scene, hence a knob with a default of 0.0 that reproduces the shipped task
    exactly rather than a silent edit.

    **0.45 since K60, and the 0/2 that stood against it was measuring something else.**
    K55 tried `station_z_lift = 0.45` on seeds 0-1 and got **0/2, both `no plan`** at
    `dip into the bucket` with both dip postures exhausted. That reading is real and the
    conclusion drawn from it — "lifting the bucket takes it out of reach from above" —
    was not: this knob lifted only the *bucket*, and `_cup_home_np` went on placing the
    cup on the floor. The oracle therefore grasped at floor height and then asked for a
    dip half a metre above it, in one move, from a posture chosen for floor work. It was
    never a test of a raised station; it was a test of a station split across two
    heights.

    What the reach table actually says, read at the right height: the dip at
    `station_z_lift = 0.45` sits at z ~ 0.72, where W1 measures forward reach of
    **1.021 m at torso 0.386 and 1.126 m at torso 0.000** — against a `refill_dock_xy`
    distance of 0.55 m. It is inside the envelope at *every* torso height, with 0.47 m
    to spare, so the dock does **not** have to move and the no-go-band arithmetic is
    untouched (`_check_station_clears_the_sites` compares dock and site in xy only, and
    neither moved).

    On the floor the same table reads 0.980 m at torso 0.000 falling to 0.549 m at
    0.386, so 0.55 m is inside the envelope only with the torso at the bottom — which is
    what pins `TORSO_DOWN` 5 cm off its stop and makes `joint limit at index [3]` the
    largest single cause of screw refusal in the sweep (397, against shoulder_lift's 209
    and wrist_flex's 162). Taking the torso off its stop is the point of this change.

    Both the bucket (kinematic, placed by `_initialize_episode`) and the cup (dynamic,
    `_cup_home_np`) follow this number now, and `_build_station_stand` puts a static
    surface under them — the cup needs real geometry to stand on, which is the piece the
    K55 attempt did not have.

    **It is not the fix for the dropped cups.** The grasp is capped at 2.75 cm below the
    cup top because deeper poses are refused `collision gripper_link<->cup` — the gripper
    body against the cup's own hull, measured at 0.045, 0.0625 and 0.08 — which is the
    same at any height.

    *(A first attempt at this measurement was void and is recorded so nobody repeats
    it: it patched `WaterPlantsConfig.station_z_lift`, and `cfg` is a class-level
    dataclass **instance** created at import, so the class attribute is never read.
    The run was byte-identical to the unlifted one — which is how it was caught. Set
    `WaterPlantsTask.cfg.station_z_lift`, or pass a config in.)*
    """

    carry_overrides: tuple[tuple[str, float], ...] = ()
    """Joints where the carry pose departs from the `rest` keyframe, and to what.

    K55, and the reason is measured rather than aesthetic. The dwell pose exists to
    make the observation *uninformative* — the same posture whichever plant is next
    — and **any** fixed posture does that equally well; `rest` was chosen because it
    is also the t=0 pose, which is a tidiness property, not a requirement of the
    memory design. What `rest` is not is *near the work*. Read off a successful
    episode, on the two joints that dominate:

        joint            dipped   tipped   rest
        elbow_flex       -0.903   -1.659   +0.949
        shoulder_lift    +0.600   -0.796   -1.032

    `elbow_flex` sits on the **opposite side of zero** from both postures the fold
    connects, so every trip swings it ~2 rad out to the dwell and ~2 rad back —
    three times an episode, for nothing. `fold to the rest keyframe` was accordingly
    the single largest consumer of hand rotation in the whole task (1245 deg of
    2557, 49%), and it is not planner waste: a collision-checked straight line
    through joint space costs 1232 deg, within noise of RRT's 1245.

    **Empty, and that is a measured result, not an oversight.** Putting those two
    joints between the postures the fold connects (`elbow_flex -1.20`,
    `shoulder_lift -0.20`) was tried on seeds 0-1 and does exactly what it was meant
    to — the fold falls 1245 -> 826 deg and the pour 930 -> 394 — and it is still not
    worth taking, because the cost moves rather than disappears: `carry pose` rises
    749 -> 1627 deg and the cup sits at **178 deg** at the dwell instead of 32, so
    23.2% of the water-carrying steps are past the spill line against 8.9%. Net hand
    rotation 2557 -> 2509, i.e. nothing, for a much worse cup.

    **Why it cannot simply be fixed by picking better numbers.** The cup's tilt at a
    fixed arm pose is set by `shoulder_lift + elbow_flex + wrist_flex`. Holding the
    cup level while moving `elbow_flex` by the -2.15 rad that brings it near the work
    needs `wrist_flex` at about 3.39 rad to compensate, and its range is +-2.16. So
    for *this* grasp there is no carry pose that is both near the work and level, and
    the fold's cost is bought back at the cup's expense wherever it is put. Moving it
    for real needs the grasp to change, or the bucket to come off the floor.

    Kept as a knob because the experiment is worth being able to repeat, and because
    a different grasp would change the arithmetic above.
    """

    # --- the world must not be used as a notebook --------------------------------
    tamper_move_tol: float = 0.10
    """How far a plant may drift, or a kitchen joint may move, before the episode is void.

    Borrowed from `season_dish.py`'s `distractor_move_tol`, which is the predicate
    this one is modelled on, and used for both because a drawer's joint travel and a
    pot's displacement are both metres-or-radians of "somebody wrote the answer into
    the furniture". Also subject to calibration: nothing has measured how far a
    kitchen joint drifts on its own over a 5200-step episode.
    """

    instructions: tuple = INSTRUCTIONS

    @property
    def dip_distance(self) -> float:
        """Metres from the refill dock to the bucket — how far the arm must reach to dip.

        Derived, never declared: the dock and the bucket are both world points, so a
        stored copy could disagree with them. It is the number the dip's reachability
        is argued from (`refill_dock_xy`), and `validate()` bounds it.

        Example:
            >>> round(WaterPlantsConfig().dip_distance, 4)
            0.55
        """
        return float(
            math.dist(tuple(self.station_xy)[:2], tuple(self.refill_dock_xy)[:2])
        )

    @property
    def dock_faces_station_error_deg(self) -> float:
        """Degrees between `dock_yaw_deg` and the direction from the dock to the bucket.

        Zero when the robot parked on the dock looks straight at what it has to dip
        into. `validate()` requires it to be small; nothing else in the task reads a
        heading except `evaluate`'s `at_station`, which compares the *base's* yaw to
        `dock_yaw_deg`.

        Example:
            >>> round(WaterPlantsConfig().dock_faces_station_error_deg, 4)
            0.0
        """
        dx = float(self.station_xy[0]) - float(self.refill_dock_xy[0])
        dy = float(self.station_xy[1]) - float(self.refill_dock_xy[1])
        want = math.degrees(math.atan2(dy, dx))
        return abs((float(self.dock_yaw_deg) - want + 180.0) % 360.0 - 180.0)

    @property
    def site_counters(self) -> tuple:
        """The distinct counters `site_spans` names, in first-use order.

        Derived rather than declared, so the two can never drift apart.

        Example:
            >>> WaterPlantsConfig().site_counters
            ('counter_main_main_group', 'counter_right_main_group')
        """
        seen = []
        for name, _which in self.site_spans:
            if name not in seen:
                seen.append(name)
        return tuple(seen)

    def validate(self) -> None:
        """Every constraint the thresholds must satisfy, checked at construction.

        Example:
            >>> WaterPlantsConfig().validate()
            >>> WaterPlantsConfig(n_plants=6).validate()  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
            AssertionError
        """
        assert 0 < self.n_plants < self.k_sites, (self.n_plants, self.k_sites)
        assert len(self.site_spans) >= 1, self.site_spans
        assert all(
            isinstance(n, str) and isinstance(i, int) and i >= 0
            for n, i in self.site_spans
        ), self.site_spans
        assert self.pot_xy > 0 and self.pot_edge_clear >= 0
        assert self.site_jitter_along >= 0 and self.site_jitter_across >= 0
        assert self.dock_radius > 0 and 0 < self.dock_heading_deg <= 180.0
        assert self.min_dock_separation > 2 * self.dock_radius, (
            "the refill dock could then sit inside a plant's zone and a trip would "
            "not have to move"
        )
        assert self.pour_min_clearance < self.pour_max_clearance, (
            self.pour_min_clearance, self.pour_max_clearance)
        assert self.pour_xy_radius > 0 and 0.0 < self.pour_tilt_deg < 180.0
        # The spacing is `pot_xy + gap`, and the gap is four times the jitter budget
        # (`site_jitter_along`, which is gap/4 — see its docstring). Deriving it that
        # way is the point: this line held a bare 0.097, layout A's gap, and went on
        # holding it after the layout moved to B's 0.1004. It passed either way, and
        # a geometric literal in `validate()` that does not track `site_spans` is the
        # kind of green that stops meaning anything.
        site_pitch = self.pot_xy + 4 * self.site_jitter_along
        assert 2 * self.pour_xy_radius < site_pitch, (
            f"the pour radius {self.pour_xy_radius} is wider than half the measured "
            f"pot-to-pot spacing (pitch {site_pitch:.4f}, half {site_pitch / 2:.4f}), "
            "so one cup pose could be over two pots at once"
        )
        assert self.hold_steps > 0 and self.grasp_min_force > 0
        lo, hi = self.refill_dwell_range
        assert 0 < lo <= hi, self.refill_dwell_range
        assert self.carry_pose_tol_rad > 0 and self.tamper_move_tol > 0
        # A negative lift would bury the bucket in the floor; a large one would put
        # it out of the arm's envelope from the other side. Both fail as a refused
        # dip several stages later, which is the kind of thing validate() exists for.
        assert 0.0 <= self.station_z_lift <= 1.5, (
            f"station_z_lift={self.station_z_lift} is outside [0, 1.5] m"
        )
        # The surface is placed as a fraction of the bucket's own mesh height, so
        # outside [0, 1] it renders below the floor or floating above the rim —
        # cosmetic, but the kind of thing that is only ever noticed in a clip.
        assert 0.0 <= self.water_fill <= 1.0, (
            f"water_fill={self.water_fill} is outside [0, 1] of the bucket's height"
        )
        assert 1 <= self.doses_per_fill <= self.n_plants, (
            f"doses_per_fill={self.doses_per_fill} must be in [1, n_plants={self.n_plants}]"
        )
        assert self.fk_envelope > 0 and self.cup_gap >= 0
        # Below the measured 0.3232 m the base's own spin sweeps the cup over
        # (K58); above the dock radius the push would eject the base from its
        # own start zone.
        assert 0.30 <= self.base_cup_min_clearance <= 0.60, (
            f"base_cup_min_clearance={self.base_cup_min_clearance} is outside [0.30, 0.60] m"
        )

        # The refill dock, against the bucket it exists to serve. Both of these were
        # violable before the station was relocated and neither was checked: the dock
        # faced the counters and the dip was 0.80 m, which is past what the arm plans
        # to (measured: the screw plan converged 11.2 cm short).
        assert self.dock_faces_station_error_deg <= self.dock_heading_deg, (
            f"the refill dock's heading is {self.dock_faces_station_error_deg:.1f} deg "
            f"off the bucket, more than dock_heading_deg={self.dock_heading_deg}: a "
            "robot parked inside the station zone would not be facing what it dips into"
        )
        assert self.dip_distance <= MAX_DIP_DISTANCE, (
            f"the bucket is {self.dip_distance:.3f} m from the refill dock, past the "
            f"{MAX_DIP_DISTANCE} m the arm has been measured to plan a dip at "
            "(0.67 m worked, 0.80 m refused with the screw plan 11.2 cm short)"
        )
        assert self.dip_distance > self.dock_radius, (
            "the bucket is inside the dock's own tolerance radius; the base could be "
            "parked on top of it"
        )
        floor = self.n_plants * (hi + self.hold_steps)
        assert floor < self.horizon, (
            f"the dwells and the pour holds alone need {floor} steps of a "
            f"{self.horizon}-step episode, before any driving"
        )


# ------------------------------------------------------------ pure helpers --
# Module-level and free of `self`, so the tests can run them without a simulator and
# so `evaluate()` stays a thin geometry layer over arithmetic that can be checked on
# paper. `burner.py:310-349` is the shape.


def subtract_spans(span, blocked):
    """`span` minus every interval in `blocked`, as a list of surviving intervals.

    Args:
        span: `(lo, hi)`.
        blocked: iterable of `(lo, hi)` intervals to remove.

    Returns:
        The surviving intervals, in order, dropping anything of zero width.

    Example:
        >>> subtract_spans((0.0, 3.0), [(1.0, 2.0)])
        [(0.0, 1.0), (2.0, 3.0)]
        >>> subtract_spans((0.0, 1.0), [(-1.0, 5.0)])
        []
    """
    free = [tuple(float(v) for v in span)]
    for lo, hi in sorted(tuple(float(v) for v in b) for b in blocked):
        out = []
        for a, b in free:
            if hi <= a or lo >= b:
                out.append((a, b))
                continue
            if a < lo:
                out.append((a, lo))
            if hi < b:
                out.append((hi, b))
        free = out
    return [(a, b) for a, b in free if b - a > 1e-6]


def lay_out_sites(spans, n: int, inset: float):
    """`n` sites at equal arc length along a concatenated list of free spans.

    The sites are not at round coordinates and are not meant to be: they are whatever
    equal spacing along the usable centre-line gives, with the first and last landing
    on its ends. Equal *arc* length rather than equal world distance is what lets a
    site row cross a gap — on kitchen 102 the step between the site left of the sink
    and the next one is 0.3067 m of arc and 1.47 m of floor.

    Args:
        spans: `[(key, lo, hi), ...]` free intervals in each surface's own along
            coordinate, in the order they should be concatenated. `key` is passed
            through untouched.
        n: how many sites to place. Must be at least 2.
        inset: how far a site centre must stay from a span's end — half a pot plus
            the edge clearance.

    Returns:
        `(sites, step, total)` where `sites` is `[(key, along, arc), ...]`, `step` is
        the arc spacing and `total` the usable arc length. A span shorter than
        `2 * inset` contributes nothing.

    Raises:
        ValueError: if the usable spans cannot hold `n` sites at all.

    Example:
        >>> sites, step, total = lay_out_sites([("a", 0.0, 1.0)], 3, 0.1)
        >>> [(k, round(x, 3)) for k, x, _ in sites], round(step, 3)
        ([('a', 0.1), ('a', 0.5), ('a', 0.9)], 0.4)
    """
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")
    usable = [
        (key, float(lo) + inset, float(hi) - inset)
        for key, lo, hi in spans
        if float(hi) - float(lo) > 2 * inset
    ]
    lengths = [hi - lo for _, lo, hi in usable]
    total = float(sum(lengths))
    if total <= 0.0:
        raise ValueError(
            f"no usable counter surface for {n} sites: every free span is shorter "
            f"than {2 * inset:.3f} m (spans={list(spans)})"
        )
    step = total / (n - 1)
    out = []
    for i in range(n):
        arc = i * step
        rest = arc
        for (key, lo, hi), length in zip(usable, lengths):
            if rest <= length + 1e-9:
                out.append((key, lo + rest, arc))
                break
            rest -= length
        else:  # pragma: no cover - float slop at the far end
            key, lo, hi = usable[-1]
            out.append((key, hi, arc))
    return out, step, total


def score_watering(
    watered_mask: torch.Tensor,
    occupied_mask: torch.Tensor,
    budget_left: torch.Tensor,
    scene_untampered: torch.Tensor,
) -> dict:
    """The verdict, as arithmetic over booleans. No simulator, no `self`.

    The episode is judged **when the doses are gone**, not when the last plant is
    watered: that is what makes "exactly once" a priced mistake rather than a
    preference. A repeat pour spends a dose on a plant that already had one, so after
    the third dose one plant is still dry, the cover is not exact, and no amount of
    further driving can fix it — the same budget-and-exact-cover pair
    `station_checklist.py:786-824` uses, and the reason a memoryless policy cannot
    simply try all three plants in turn.

    Args:
        watered_mask: `(b, k)` bool — sites that have received a dose.
        occupied_mask: `(b, k)` bool — sites that hold a plant this episode.
        budget_left: `(b,)` int — doses still in the bucket.
        scene_untampered: `(b,)` bool — the latch that says the world was not used
            as a notebook.

    Returns:
        A dict of batched tensors: `success`, `fail`, `episode_over`, `exact_cover`,
        `watered_count`, `omission_count`, `stray_count`.

    Example:
        >>> import torch
        >>> occ = torch.tensor([[True, True, False]])
        >>> ok = torch.tensor([[True, True, False]])
        >>> v = score_watering(ok, occ, torch.tensor([0]), torch.tensor([True]))
        >>> bool(v["success"][0]), int(v["watered_count"][0])
        (True, 2)
    """
    episode_over = budget_left <= 0
    exact_cover = (watered_mask == occupied_mask).all(dim=-1)
    watered_count = (watered_mask & occupied_mask).sum(dim=-1).to(torch.int32)
    omission_count = (occupied_mask & ~watered_mask).sum(dim=-1).to(torch.int32)
    stray_count = (watered_mask & ~occupied_mask).sum(dim=-1).to(torch.int32)
    clean = exact_cover & scene_untampered
    return {
        "success": episode_over & clean,
        "fail": episode_over & ~clean,
        "episode_over": episode_over,
        "exact_cover": exact_cover,
        "watered_count": watered_count,
        "omission_count": omission_count,
        "stray_count": stray_count,
    }


def memoryless_floor(cfg: WaterPlantsConfig) -> float:
    """Chance of covering every plant exactly once by drawing uniformly with replacement.

    `n! / n**n`. Episode success is this times the motor success rate. It depends on
    `n_plants` only — `k_sites` sets how many arrangements exist, not how hard the
    bookkeeping is.

    Example:
        >>> round(memoryless_floor(WaterPlantsConfig()), 4)
        0.2222
    """
    n = cfg.n_plants
    return float(math.factorial(n)) / float(n**n)


@register_env(
    "MikasaWaterPlants-v0",
    max_episode_steps=WaterPlantsConfig.horizon,
    asset_download_ids=["RoboCasa"],
)
class WaterPlantsTask(BaseEnv):
    """Water every plant on the counter, and none of them twice."""

    # `ds_fetch` is what the planner in this repo drives. "none" is deliberately
    # absent: it sets `self.agent = None` and every method below dereferences it.
    SUPPORTED_ROBOTS = ["mikasa_ds_fetch", "fetch"]

    # No `compute_dense_reward` in v1, so no dense mode may be listed: a listed mode
    # without an implementation raises on the first step, and an implementation
    # without the listing never runs.
    SUPPORTED_REWARD_MODES = ["sparse", "none"]

    cfg = WaterPlantsConfig()

    cup: Actor
    bucket: Actor

    def __init__(self, *args, robot_uids="mikasa_ds_fetch", scene_idx: int | None = 0, **kwargs):
        # scene_idx defaults to 0 — the same pinned kitchen as the other three memory
        # tasks. It is a RoboCasa *build config* index, not a kitchen number
        # (`layout = idx // 12`, `style = idx % 12`, measured in Task 4), so 0..11 are
        # one kitchen in twelve textures and anyone varying the *kitchen* must step in
        # multiples of 12. Set before super().__init__, which runs a full reconfigure
        # and reaches _load_scene.
        self.scene_idx = scene_idx
        self.cfg.validate()
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # ---------------------------------------------------------------- sensors --

    @property
    def _default_sensor_configs(self):
        """None of the task's own — `ds_fetch` already carries the head and wrist rigs.

        A room camera would be a privileged viewpoint for a task that is almost
        entirely about driving, so it lives only in the human render below. This is
        the same choice `station_checklist.py:330` makes and it differs from the
        burner's and the season dish's, which do add a `base_camera`. That
        divergence is an open question in `docs/envs-for-supervisor.md`; it is named
        here rather than quietly settled in one direction.
        """
        return []

    @property
    def _default_human_render_camera_configs(self):
        """The clip's viewpoint. Decoration — no threshold reads a camera.

        Stands behind and above the refill dock looking at the counter row, so one
        frame holds the bucket, the robot and all the plants: the thing a viewer has
        to read is which plant the robot is driving to, and whether it went back for
        water in between.

        512 is the *recording* default. `RecordEpisode` keeps a whole episode of
        frames in host RAM, and this task has by some way the longest horizon in the
        repo (5200 steps = 3.9 GB at 512**2, 65 GB at 2048**2 — a full episode at the
        clip resolution does not fit on this machine at all). Clips get their size from
        `utils.mikasa.replay --render-size` instead, one at a time.
        """
        station = np.asarray(self.cfg.station_xy, dtype=np.float32)
        # Behind and above the station, wide enough to hold the counter row *and* the
        # station in one frame — the earlier narrower framing pushed the station into
        # the corner, so the dip half of the episode played off-camera. The eye stays
        # clamped inside the room: the floor runs to y = −3.04, and an eye past the
        # south wall renders the wall, not the kitchen. Decoration either way; no
        # threshold reads a camera.
        eye = np.array(
            [station[0] - 2.3, max(float(station[1]) - 0.41, -2.95), 2.10], dtype=np.float32
        )
        target = np.array([station[0] - 0.95, -1.2, 0.85], dtype=np.float32)
        pose = sapien_utils.look_at(eye=eye, target=target)
        return CameraConfig("render_camera", pose, 512, 512, 1.20, 0.01, 100)

    # ------------------------------------------------------------------- load --

    def _load_agent(self, options: dict):
        # Clear of the kitchen, so gpu_init() is not resolving an interpenetration on
        # frame one. The scene builder overwrites this during build() anyway.
        super()._load_agent(options, parking_pose(self))

    def _load_scene(self, options: dict):
        """Kitchen, sites, plants, bucket and cup. Poses through `initial_pose` only.

        There is no `set_pose` in this method and that is not style: `_reconfigure`
        runs `scene._setup()` straight after, which re-applies `initial_pose` to every
        non-static actor and silently discards anything positioned with `set_pose`.
        The lint walks this method's AST (`tests/test_task_conventions.py`).

        Everything scene-derived is derived **per env**. RoboCasa draws a layout per
        sub-scene, so taking one counter's frame from `scene_data[0]` and using it
        everywhere puts objects inside walls in envs 1..N-1 without a crash.
        """
        super()._load_scene(options)

        self.scene_builder = RoboCasaSceneBuilder(self)
        idx = 0 if self.scene_idx is None else self.scene_idx
        self.scene_builder.build([idx] * self.num_envs)

        self._fix_ds_fetch_collision_bits()

        k = self.cfg.k_sites
        site_pos, site_dock, site_names, steps = [], [], [], []
        along_vecs, across_vecs = [], []
        for i in range(self.num_envs):
            fixtures = self.scene_builder.scene_data[i]["fixtures"]
            spans = self._free_spans(fixtures, i)
            inset = self.cfg.pot_xy / 2.0 + self.cfg.pot_edge_clear
            laid, step, _total = lay_out_sites(spans, k, inset)
            steps.append(step)

            frames, docks = {}, {}
            for name in self.cfg.site_counters:
                counter = require_get_fixture(
                    self.scene_builder, fixtures, name, scene_idx=self.scene_idx
                )
                frames[name] = counter_frame(counter)
                docks[name] = dock_pose_for(self.scene_builder, fixtures, name)

            pos_i, dock_i, names_i = [], [], []
            for name, along_off, _arc in laid:
                top, along, across = frames[name]
                pos_i.append(top + along * along_off + across * self.cfg.site_across)
                dock_p, dock_yaw = docks[name]
                slid = dock_p + along * along_off
                dock_i.append(np.array([slid[0], slid[1], dock_yaw], dtype=np.float32))
                names_i.append(name)
            site_pos.append(np.stack(pos_i).astype(np.float32))
            site_dock.append(np.stack(dock_i).astype(np.float32))
            site_names.append(names_i)
            along_vecs.append(frames[self.cfg.site_counters[0]][1])
            across_vecs.append(frames[self.cfg.site_counters[0]][2])

        if any(n != site_names[0] for n in site_names):
            raise RuntimeError(
                "parallel environments laid the sites out on different counters, so "
                f"site index s means different things per env: {site_names}"
            )
        self.site_counter_names = site_names[0]
        self.site_step = float(np.mean(steps))

        self._site_pos_np = np.stack(site_pos).astype(np.float32)
        self._site_dock_np = np.stack(site_dock).astype(np.float32)
        self._along_np = np.stack(along_vecs).astype(np.float32)
        self._across_np = np.stack(across_vecs).astype(np.float32)

        station = np.asarray(self.cfg.station_xy, dtype=np.float32)
        dock = np.asarray(self.cfg.refill_dock_xy, dtype=np.float32)
        yaw = math.radians(self.cfg.dock_yaw_deg)
        self._refill_dock_np = np.tile(
            np.array([dock[0], dock[1], yaw], dtype=np.float32), (self.num_envs, 1)
        )
        self._check_station_clears_the_sites()

        # --- the plants ------------------------------------------------------
        # One actor per *site*, not per plant: the draw parks the unoccupied ones at
        # HIDDEN_Z, which is also what makes the pour predicate self-limiting — a
        # cup cannot be 0.05-0.30 m above a pot that is a kilometre up.
        # `body_type="kinematic"`: their poses are a row of the episode ticket, so
        # they are re-posed every reset, and `Actor.pose`'s setter asserts a
        # non-static body under GPU sim (structs/actor.py:344-347). Kinematic also
        # means a brush of the cup's rim cannot move one into being a bookmark,
        # which is half of closing hole A.
        self.plants = [
            load_accessory_actor(
                self,
                self.cfg.plant_rel_xml,
                f"plant_{s}",
                sapien.Pose(p=self._site_pos_np[0, s]),
                scale=self.cfg.plant_scale,
                body_type="kinematic",
            )
            for s in range(k)
        ]

        # --- the bucket and the cup ------------------------------------------
        # The bucket is `bowl` at the scale RoboCasa declares for it (2.0), standing
        # on the floor: Task 2's probe took the stand out of the design by measuring
        # that the TCP reaches z = 0.25 m at 0.30-0.50 m in front of the base at every
        # torso height. Kinematic rather than dynamic so a knock cannot slide the
        # station, and rather than static so the pose setter stays available.
        #
        # The category stays `bowl` and that was **measured, not assumed** (K56). A
        # bowl reads as a red plate, so the believable-looking alternatives were
        # loaded and their meshes compared against it (xy half-extent, height):
        #
        #     bowl   0.1279  0.0736   <- shipped
        #     pot    0.1840  0.0784   renders as a flat blue pan lid, and its dark
        #                             rim hides the water surface completely
        #     jug    0.0522  0.1954   too narrow to dip a cup into
        #     kettle 0.0832  0.1420   lidded; nothing to dip into
        #     tray   0.1991  0.0642   flat, no basin
        #
        # `pot` was the only real candidate and it looks *worse* rendered, so the
        # swap buys nothing — and it is not free. The no-go band between base and
        # bucket centres is `base_radius_at_bucket_height + bucket_half`
        # = 0.2853 + 0.1279 = 0.4132 m, hardcoded in three places
        # (`water_plants_planner.STATION_NOGO`, `tests/test_water_plants._NOGO_LO`,
        # `w1_sites_and_reach.py`); a pot moves it to ~0.4695 and would put the
        # planner's 0.4200 m approach pose on the wrong side of it. What actually
        # made the station read as water was putting water in it — `water_visual`.
        self.bucket = load_objaverse_actor(
            self,
            "bowl",
            "water_bucket",
            sapien.Pose(p=[station[0], station[1], self.cfg.spawn_clearance]),
            index=0,
            body_type="kinematic",
        )
        # The same cup asset and instance as the burner task (`burner.py:629`,
        # index=0), so its grasp geometry and the `carry_pose` yaw candidates are
        # already measured. Dynamic: it is the one thing the robot carries.
        self.cup = load_objaverse_actor(
            self, "cup", "cup", sapien.Pose(p=[station[0], station[1] - 0.3, 0.2]), index=0
        )

        # Mesh measurements taken once, after the build, in the actors' rest poses.
        # `_rest_lift` is origin-to-mesh-bottom and exists because spawning by origin
        # at `surface + clearance` drove three of season_dish's meshes 1.7-5.9 cm into
        # the counter slab and PhysX pushed one of them *through* it, into the cabinet
        # (journal 2026-08-18). Spawn by mesh bottom, never by origin.
        self._bucket_rest_lift = self._rest_lift(self.bucket)
        self._cup_rest_lift = self._rest_lift(self.cup)
        self._plant_rest_lift = self._rest_lift(self.plants[0])
        self._plant_top_lift = self._top_lift(self.plants[0])
        self._bucket_top_lift = self._top_lift(self.bucket)
        bucket_half = float(
            np.max(self.bucket.get_first_collision_mesh(to_world_frame=False).extents[:2]) / 2.0
        )
        cup_half = float(
            np.max(self.cup.get_first_collision_mesh(to_world_frame=False).extents[:2]) / 2.0
        )
        if self.cfg.water_visual:
            self._build_water_surface(station, bucket_half)
        # The cup starts between the dock and the bucket, on the base's centre line:
        # the probe's block 5 measured a near-vertical top grasp of a cup on the floor
        # to be reachable at dock distances 0.251-0.770 m with lateral offset within
        # 0.15 m, and the bucket is `dip_distance` = 0.55 m out, which puts the cup
        # ~0.37 m out. Its distance from the bucket is measured, not chosen: two
        # half-footprints plus `cup_gap`.
        #
        # The direction is **derived from the dock**, not written down. It used to be
        # the literal (0, −1, 0), which was true only while the dock happened to sit
        # due south of the bucket; after the station moved (probe block 7) that literal
        # would have spawned the cup on the far side of the bucket from the robot, out
        # of reach, and nothing in the task would have said so.
        to_dock = dock - station
        toward_dock = np.zeros(3, dtype=np.float32)
        toward_dock[:2] = to_dock / max(float(np.linalg.norm(to_dock)), 1e-9)
        self._cup_home_np = np.tile(
            np.array(
                [
                    station[0] + toward_dock[0] * (bucket_half + cup_half + self.cfg.cup_gap),
                    station[1] + toward_dock[1] * (bucket_half + cup_half + self.cfg.cup_gap),
                    # K60: the cup stands on whatever the station stands on. The bucket
                    # follows `station_z_lift` in `_initialize_episode` and this did not,
                    # which is why the one previous attempt at a raised station (0.45,
                    # 0/2, recorded under `station_z_lift`) was never a test of a raised
                    # station at all — it lifted the bucket and left the cup on the floor,
                    # so the oracle grasped at the floor and dipped half a metre up.
                    float(self.cfg.station_z_lift)
                    + self.cfg.spawn_clearance
                    + self._cup_rest_lift,
                ],
                dtype=np.float32,
            ),
            (self.num_envs, 1),
        )
        if self.cfg.station_z_lift > 0.0:
            self._build_station_stand(station, bucket_half, cup_half, toward_dock)

        # Kitchen articulations, minus the robot: cabinet doors and drawers. Their
        # qpos is the other half of "the world must not become a notebook" — a policy
        # that cannot mark a pot can still leave a drawer open as a tally.
        self._kitchen_articulations = [
            art
            for art in self.scene.articulations.values()
            if self.agent is None or art.name != self.agent.robot.name
        ]

        dev = self.device
        t = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)  # noqa: E731
        self.site_pos = t(self._site_pos_np)  # (n_envs, k, 3)
        self.site_dock = t(self._site_dock_np)  # (n_envs, k, 3): x, y, yaw
        self.refill_dock = t(self._refill_dock_np)  # (n_envs, 3): x, y, yaw
        self.station_pos = t(np.tile(station, (self.num_envs, 1)))
        self._cup_home = t(self._cup_home_np)
        self._along = t(self._along_np)
        self._across = t(self._across_np)

    def _free_spans(self, fixtures: dict, env_i: int):
        """The spans `cfg.site_spans` names, in that order, in each counter's frame.

        Two steps that are deliberately separate. `_counter_free_spans` **derives**
        every free span of a counter from the assembled kitchen — nothing about the
        sink's extent is declared anywhere. This method then **selects** the ones the
        layout uses, because which of them to use is a decision and not a
        measurement.

        Keeping them apart is the fix for a real defect: before layout B this method
        returned every derived span and `k_sites` alone was documented as the knob
        that switched layouts. It was not — dropping `k_sites` to 5 kept the
        left-of-wall span and gave 0.3833 m of spacing rather than the intended
        0.3104 m, and the test that appeared to prove otherwise hand-fed a span list
        `_free_spans` never returned.
        """
        out = []
        for name, which in self.cfg.site_spans:
            free = self._counter_free_spans(fixtures, name, env_i)
            if which >= len(free):
                raise RuntimeError(
                    f"env {env_i}: cfg.site_spans asks for free span {which} of "
                    f"{name!r}, which has only {len(free)}: {free}. Either the "
                    f"kitchen is not the one this layout was measured on, or "
                    f"something new is standing on that counter."
                )
            lo, hi = free[which]
            out.append((name, lo, hi))
        return out

    def _counter_free_spans(self, fixtures: dict, name: str, env_i: int):
        """Every free along-span of one counter, low end first, in its own frame.

        Derived, not declared. Every fixture whose box stands **on** the counter
        (its top above the counter top, its bottom below a pot's height) and whose
        across extent overlaps where a pot would stand is subtracted from the
        counter's along extent. On kitchen 102 that is the sink, and it is the only
        one: the paper towel holder stands at across −0.166, clear of the pot band,
        and the wall cabinets start 0.47 m above a pot's head. The overhead cabinets
        are not handled here — they are why `site_across` is what it is.
        """
        counter = require_get_fixture(
            self.scene_builder, fixtures, name, scene_idx=self.scene_idx
        )
        top, along, across, size = fixture_frame(counter)
        half = float(size[0]) / 2.0
        extent = (-half, half)
        top_z = float(top[2])
        band = (
            self.cfg.site_across - self.cfg.pot_xy / 2.0,
            self.cfg.site_across + self.cfg.pot_xy / 2.0,
        )
        blocked = []
        for other_name, other in fixtures.items():
            if other_name == name:
                continue
            box = self._fixture_box(other)
            if box is None:
                continue
            lo_z, hi_z = box[2]
            if hi_z <= top_z + 1e-3 or lo_z >= top_z + self.cfg.pot_height:
                continue
            a_lo, a_hi = self._project(box, along, origin=top)
            c_lo, c_hi = self._project(box, across, origin=top)
            if not (c_lo < band[1] and band[0] < c_hi):
                continue
            if not (a_lo < extent[1] and extent[0] < a_hi):
                continue
            blocked.append((max(a_lo, extent[0]), min(a_hi, extent[1])))
        free = subtract_spans(extent, blocked)
        if not free:
            raise RuntimeError(
                f"env {env_i}: counter {name!r} has no free surface left after "
                f"subtracting {blocked}; the sites cannot be laid out"
            )
        return free

    @staticmethod
    def _fixture_box(fixture):
        """`((x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi))` world box, or None for a marker.

        Walls, floors and wall accessories are excluded: RoboCasa stores half sizes
        for the first two and a zero size for the third, so their boxes would be
        wrong in a way that quietly eats counter surface.
        """
        if type(fixture).__name__ in ("Wall", "Floor", "WallAccessory"):
            return None
        pos = np.asarray(fixture.pos, dtype=np.float64)
        size = np.asarray(fixture.size, dtype=np.float64)
        if size.shape != (3,) or not np.all(np.isfinite(size)):
            return None
        yaw = float(getattr(fixture, "rot", 0.0) or 0.0)
        c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
        hx = 0.5 * (size[0] * c + size[1] * s)
        hy = 0.5 * (size[0] * s + size[1] * c)
        return (
            (pos[0] - hx, pos[0] + hx),
            (pos[1] - hy, pos[1] + hy),
            (pos[2] - size[2] / 2.0, pos[2] + size[2] / 2.0),
        )

    @staticmethod
    def _project(box, axis, origin):
        """Min/max of a world box's footprint along `axis`, relative to `origin`."""
        axis = np.asarray(axis, dtype=np.float64)[:2]
        base = float(np.dot(np.asarray(origin, dtype=np.float64)[:2], axis))
        corners = np.array(
            [
                [box[0][0], box[1][0]],
                [box[0][0], box[1][1]],
                [box[0][1], box[1][0]],
                [box[0][1], box[1][1]],
            ]
        )
        v = corners @ axis - base
        return float(v.min()), float(v.max())

    def _build_station_stand(self, station, bucket_half: float, cup_half: float,
                             toward_dock) -> None:
        """The surface the bucket and the cup stand on when `station_z_lift` > 0.

        K60. Raising the station is what takes the torso off its stop. On the floor the
        dip sits at z ~ 0.27 and `refill_dock_xy` is 0.55 m out, and W1's reach table
        says forward reach at that height runs 0.980 m at torso 0.000 down to **0.549 m
        at torso 0.386** — so the dock distance is only inside the envelope with the
        torso all the way down, and `TORSO_DOWN` is pinned 5 cm off its lower limit.
        That single fact is why `joint limit at index [3]` — the torso — is the largest
        cause of screw refusal in the sweep (397, more than shoulder_lift's 209 and
        wrist_flex's 162 together), and therefore why 71 % of executed motion is an RRT
        detour rather than a straight line. Lifted to 0.45 the dip is at z ~ 0.72, where
        the same table gives 1.021-1.126 m at *every* torso height: the dock distance
        stops constraining the torso and the torso is free to be spent on height.

        Static, so nothing can nudge the station; a box rather than a loaded asset
        because its only job is to be a surface at a height, and an asset would bring a
        mesh whose top has to be measured before it can be trusted. It is sized from the
        two things standing on it — never from constants — so it follows `station_xy`,
        the cup's own offset and the jitter budget without anyone having to remember it.

        **This is not what fixes the dropped cups**, and the distinction is worth
        keeping straight: the grasp is capped at 2.75 cm below the cup's top because
        deeper poses are refused `collision gripper_link<->cup` (measured, seed 11 —
        0.045, 0.0625 and 0.08 all collide), which is the gripper body against the cup's
        own hull and has nothing to do with what the cup stands on. What this buys is
        the torso, the trajectories that follow from it, and a watering station that
        does not stand on the kitchen floor.
        """
        lift = float(self.cfg.station_z_lift)
        # Sized to the two things standing on it and no larger, and **rectangular, not
        # square**: the bucket is 0.128 m of half-width against the cup's 0.037, so a
        # square sized for the bucket wastes 9 cm on the cup's side — and that side is
        # the side the robot parks on. The first version was square with a generous
        # margin, came out 0.320 m of half-extent, and put its near face 0.138 m from
        # the base centre: every episode died at t=0 with the rotation sweep reporting
        # `base_link<->station_stand` *already touching before the turn*.
        #
        # Along the station-to-dock axis the box runs from behind the bucket to beyond
        # the cup; across it, only as wide as the wider of the two.
        cup_out = float(bucket_half + cup_half + self.cfg.cup_gap)
        pad = float(self.cfg.cup_jitter_xy) + STAND_MARGIN
        # **Axis-aligned with the kitchen, not with the dock ray.** The first version
        # yawed the box onto `toward_dock`, which runs ~110 deg off the room's axes on
        # kitchen 102 — so the stand sat visibly askew from the counter row in every
        # frame, and the owner read it as a randomised orientation. It was not random,
        # just wrong: nothing else in the room is oriented to the dock ray, and a piece
        # of furniture aligns with the room it stands in. The box is now the
        # axis-aligned bounding rectangle of its two occupants — the bucket at the
        # station and the cup one `cup_out` along `toward_dock` — plus the pad. Covers
        # the same contents at any dock bearing, with no yaw to read as a tilt.
        cup_c = np.array(
            [
                float(station[0]) + float(toward_dock[0]) * cup_out,
                float(station[1]) + float(toward_dock[1]) * cup_out,
            ],
            dtype=np.float64,
        )
        lo = np.minimum(
            np.array([float(station[0]), float(station[1])]) - bucket_half,
            cup_c - cup_half,
        ) - pad
        hi = np.maximum(
            np.array([float(station[0]), float(station[1])]) + bucket_half,
            cup_c + cup_half,
        ) + pad
        half_x, half_y = float(hi[0] - lo[0]) / 2.0, float(hi[1] - lo[1]) / 2.0
        centre = np.array(
            [float(lo[0] + hi[0]) / 2.0, float(lo[1] + hi[1]) / 2.0, lift / 2.0],
            dtype=np.float64,
        )
        self.station_stands = []
        for i in range(self.num_envs):
            mat = sapien.render.RenderMaterial(
                base_color=STAND_COLOR, roughness=0.8, metallic=0.0
            )
            b = self.scene.create_actor_builder()
            b.add_box_collision(half_size=[half_x, half_y, lift / 2.0])
            b.add_box_visual(half_size=[half_x, half_y, lift / 2.0], material=mat)
            b.set_scene_idxs([i])
            b.initial_pose = sapien.Pose(p=centre)
            self.station_stands.append(b.build_static(name=f"station_stand_env{i}"))

    def _build_water_surface(self, station, bucket_half: float) -> None:
        """A still water surface inside the bucket. Visual only, static, per env.

        Sized and placed from the bucket's own measured mesh rather than from
        constants, so it follows `station_xy` and `station_z_lift` without anyone
        having to remember to move it. Inset from the rim by `WATER_INSET` so it
        reads as water *in* a vessel rather than a disc balanced on one.

        Static and collision-free on purpose, the same reasoning `emissive.BeaconGrid`
        gives: it must not be something the cup can knock, and it must not be
        something whose pose could ever encode task state. One actor and one material
        per env, because `ActorBuilder` shares a `RenderMaterial` across sub-scenes.
        """
        height = float(self._bucket_rest_lift + self._bucket_top_lift)
        z = float(self.cfg.station_z_lift) + self.cfg.water_fill * height
        radius = max(float(bucket_half) - WATER_INSET, 0.02)
        self.water_surfaces = []
        for i in range(self.num_envs):
            mat = sapien.render.RenderMaterial(
                base_color=WATER_COLOR, roughness=0.05, metallic=0.0
            )
            b = self.scene.create_actor_builder()
            b.add_cylinder_visual(radius=radius, half_length=WATER_HALF_THICK, material=mat)
            b.set_scene_idxs([i])
            # `add_cylinder_visual` extends along local +X; stand it up.
            b.initial_pose = sapien.Pose(
                p=[float(station[0]), float(station[1]), z],
                q=[0.70710678, 0.0, 0.70710678, 0.0],
            )
            self.water_surfaces.append(b.build_static(name=f"water_surface_env{i}"))

    @staticmethod
    def _rest_lift(actor: Actor) -> float:
        """How far an actor's origin sits above the lowest point of its collision mesh."""
        mesh = actor.get_first_collision_mesh(to_world_frame=True)
        return float(np.asarray(actor.pose.p[0])[2] - mesh.bounds[0][2])

    @staticmethod
    def _top_lift(actor: Actor) -> float:
        """How far the highest point of an actor's collision mesh sits above its origin."""
        mesh = actor.get_first_collision_mesh(to_world_frame=True)
        return float(mesh.bounds[1][2] - np.asarray(actor.pose.p[0])[2])

    def _check_station_clears_the_sites(self) -> None:
        """Refuse to build a kitchen where a plant could be watered without driving.

        Two separate distances, both measured on 102 and both load-bearing, and both
        measured from the **refill dock**: no site may be within the arm's envelope of
        it (else the trip, and with it the holding interval, collapses), and no plant
        dock may be within `min_dock_separation` of it (else a trip could be served
        without leaving the station's zone).

        The first one measured from `station_xy` until the final review, and that was
        not a typo — it was correct under the geometry it was written for. The arm
        reaches from the base and the base stands on the dock; while the dock sat
        0.80 m *behind* the bucket relative to the counters, site-to-bucket was the
        shorter of the two and measuring it was conservative. Block 7 of
        `geometry-decided.md` moved the dock to the counter side and inverted that
        without anyone touching this line. On the shipped layout the site-to-bucket
        minimum is 1.7811 m (+0.681 over `fk_envelope = 1.10`) and the site-to-dock
        minimum is 1.2555 m (+0.156), so the guard would not have fired until the real
        margin had gone 0.53 m negative — backwards from the collapse it exists to
        prevent, and that collapse is what would silently destroy the holding interval
        this task measures. Neither number moved when the guard was corrected; the
        margin it reports did.
        """
        reach = self._site_pos_np[:, :, :2] - np.asarray(
            self.cfg.refill_dock_xy, dtype=np.float32
        )
        d_reach = np.linalg.norm(reach, axis=-1)
        if float(d_reach.min()) < self.cfg.fk_envelope:
            raise RuntimeError(
                f"a candidate site is {d_reach.min():.3f} m from the refill dock, "
                f"inside the {self.cfg.fk_envelope} m arm envelope: that plant could be "
                "watered from the refill dock without driving, and the holding "
                "interval this task measures would not exist. Move refill_dock_xy (with "
                "station_xy following it), or pin a kitchen where the counters are "
                "further away."
            )
        d_dock = np.linalg.norm(
            self._site_dock_np[:, :, :2]
            - np.asarray(self.cfg.refill_dock_xy, dtype=np.float32),
            axis=-1,
        )
        if float(d_dock.min()) < self.cfg.min_dock_separation:
            raise RuntimeError(
                f"the refill dock is {d_dock.min():.3f} m from a plant dock, below "
                f"min_dock_separation={self.cfg.min_dock_separation}: the robot could "
                "stand in both zones at once and a trip would not have to move."
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

    # ------------------------------------------------------------- initialize --

    def _after_reconfigure(self, options: dict):
        """Task buffers, allocated once at full width.

        Not lazily in `_initialize_episode`: `get_state_dict` runs before the first
        one (sapien_env.py:332, and RecordEpisode on every reset), so a buffer that
        does not exist yet is an AttributeError on a routine path.
        """
        n, k, dev = self.num_envs, self.cfg.k_sites, self.device
        z = lambda *shape, dtype=torch.bool: torch.zeros(shape, dtype=dtype, device=dev)  # noqa: E731

        self.watered_mask = z(n, k)
        self.occupied_mask = z(n, k)
        self.has_water = z(n)
        # Doses left in the cup. Only meaningful when `cfg.doses_per_fill` > 1;
        # at the default of 1 it is exactly `has_water` as an integer and the
        # arithmetic below reduces to the shipped behaviour.
        self.doses_left = torch.zeros(n, dtype=torch.int32, device=dev)
        # Set by a pour, cleared once the cup has left every pot. Only consulted when
        # `cfg.doses_per_fill` > 1: at 1 the cup is empty after a pour and
        # `has_water` already blocks a second commit, which is the only reason the
        # shipped task never had to think about it. With doses left over, the cup
        # lingers tilted over the same pot during the tuck, `pour_hold` re-arms and
        # the same plant commits twice — spending two doses for one pour.
        self.poured_since_left_pot = torch.zeros(n, dtype=torch.bool, device=dev)
        self.dip_done = z(n)
        self.repeat_seen = z(n)
        self.first_decision_made = z(n)
        self.first_decision_ok = z(n)
        self.scene_untampered = torch.ones(n, dtype=torch.bool, device=dev)

        self.budget_left = torch.full((n,), self.cfg.n_plants, dtype=torch.int32, device=dev)
        self.trip_index = z(n, dtype=torch.int32)
        self.refill_dwell = z(n, self.cfg.n_plants, dtype=torch.int32)
        self.refill_timer = z(n, dtype=torch.int32)
        self.pour_hold = z(n, k, dtype=torch.int32)
        self.double_water_count = z(n, dtype=torch.int32)
        self.distinct_before_repeat = z(n, dtype=torch.int32)
        self.instruction_idx = z(n, dtype=torch.int32)

        self.plant_home = torch.zeros((n, k, 3), dtype=torch.float32, device=dev)
        self._last_eval_step = torch.full((n,), -1, dtype=torch.int32, device=dev)

        # The carry pose the refill dwell is served in: the arm, the torso and the
        # head at the `rest` keyframe — the pose `_restore_robot` puts the robot in at
        # t = 0. Reading it from the keyframe rather than naming ten joint angles is
        # what keeps the target of the carry predicate *the same object* as the t = 0
        # pose, so the two cannot drift apart when the keyframe changes. It does not
        # make the observation identical, and the earlier wording here said it did —
        # contradicting this file's own module docstring, which retracts "identical"
        # as an overclaim. What holds is the module docstring's sentence:
        # **uninformative up to the predicates' tolerances**, `carry_pose_tol_rad`
        # being 0.15 rad of it. The gripper is excluded: it is holding the cup. The
        # base is excluded too, and is covered separately by the dock predicate.
        ctrl = self.agent.controller
        names = list(ctrl.controllers["arm"].config.joint_names) + list(
            ctrl.controllers["body"].config.joint_names
        )
        joints = self.agent.robot.active_joints_map
        self._carry_qpos_idx = [joints[j].active_index[0].item() for j in names]
        self._carry_joint_names = names
        rest = torch.as_tensor(
            np.asarray(self.agent.keyframes["rest"].qpos, dtype=np.float32), device=dev
        )
        self._carry_qpos = rest[self._carry_qpos_idx].clone()
        # K55: the dwell posture departs from `rest` on the joints named in
        # `carry_overrides` — see that field for the measurement and the cost.
        for _name, _val in self.cfg.carry_overrides:
            if _name not in names:
                raise KeyError(
                    f"carry_overrides names {_name!r}, which is not one of the carry "
                    f"joints {names}. A typo here would silently do nothing, so it raises."
                )
            self._carry_qpos[names.index(_name)] = float(_val)
        # K55. `carry_error` compares angles, and an angle is only defined modulo a
        # full turn: a joint at -3.14 rad and a target at +3.14 rad are the *same
        # pose* but differ by 6.28 under plain subtraction. That never had to be
        # faced while the rolling joints stopped at +-pi, because the two ends were
        # barely reachable in one episode; with the stops at +-2pi (fetch.urdf, K55)
        # a legitimately-wrapped joint reads as a 6.28 rad error and the refill
        # dwell never opens — measured on seed 1, `carry_error=6.2801`, the cup
        # still in the gripper and the pose visually correct. The wrap is applied
        # per joint because `torso_lift_joint` is **prismatic**: metres do not wrap,
        # and folding them would silently accept a torso a full "turn" out of place.
        # `getattr` rather than `.type`: the conventions tests drive this method with
        # SimpleNamespace joint doubles that carry only `active_index`. Defaulting an
        # unknown joint to angular is the safe direction — wrapping is the identity
        # for any joint whose range is under a full turn, which every joint here but
        # the rolling ones is, while *failing* to wrap a rolling joint is the bug
        # this mask exists to prevent.
        self._carry_wraps = torch.as_tensor(
            [getattr(joints[j], "type", None) != "prismatic" for j in names],
            dtype=torch.bool,
            device=dev,
        )

        self.articulation_home = torch.zeros(
            (n, self._articulation_qpos().shape[-1]), dtype=torch.float32, device=dev
        )
        return super()._after_reconfigure(options)

    def _articulation_qpos(self) -> torch.Tensor:
        """`(n_envs, d)` — every kitchen joint's position, the robot's excluded."""
        parts = [art.get_qpos() for art in self._kitchen_articulations]
        if not parts:
            return torch.zeros((self.num_envs, 0), dtype=torch.float32, device=self.device)
        return torch.cat([p.to(torch.float32) for p in parts], dim=-1)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Runs on every reset, for the envs in `env_idx` only."""
        with torch.device(self.device):
            b = len(env_idx)
            k = self.cfg.k_sites
            n = self.cfg.n_plants

            # Restores fixtures, and the robot only when uid == "fetch"
            # (scene_builder.py:566 compares a literal), hence _restore_robot after.
            # The lint walks this method's own source for the call, so it must appear
            # here literally rather than behind a helper.
            self.scene_builder.initialize(env_idx)
            self._restore_robot(env_idx)

            # The whole ticket from the *episode* RNG. torch's generator is only
            # seeded when a seed was passed (sapien_env.py:948-953), so a
            # torch-drawn answer is not reproducible from a seed; torch.rand is for
            # nothing here. One permutation gives the occupancy, and every dwell is
            # drawn now rather than trip by trip, so an episode's RNG consumption
            # does not depend on how far the agent got.
            rng = self._batched_episode_rng[env_idx]
            perm = np.asarray(rng.permutation(k)).reshape(b, k)
            lo, hi = self.cfg.refill_dwell_range
            dwell = np.asarray(rng.randint(lo, hi + 1, n)).reshape(b, n)
            site_jit = np.asarray(rng.uniform(-1.0, 1.0, (k, 2))).reshape(b, k, 2)
            cup_jit = np.asarray(rng.uniform(-1.0, 1.0, 2)).reshape(b, 2)
            base_jit = np.asarray(rng.uniform(-1.0, 1.0, 3)).reshape(b, 3)
            instr = np.asarray(rng.randint(0, len(self.cfg.instructions))).reshape(b)

            occupied = torch.zeros((b, k), dtype=torch.bool)
            rows = torch.arange(b).unsqueeze(-1)
            occupied[rows, torch.as_tensor(perm[:, :n])] = True

            # Certification hook: an override replaces the draw but does not skip it,
            # so the random stream advances identically either way.
            #
            # The assert is the whole reason `_occupied_plant_poses` may claim that no
            # `HIDDEN_Z` ever reaches the observation. That method sorts the occupied
            # sites forward and slices `[:n]`, which is correct only when exactly `n`
            # are occupied; the draw above guarantees it, an override does not.
            # Reproduced with two occupied out of five: the emitted z came back
            # `[1.1, 1.1, 1000.0]`, and `z > 100` reconstructs `occupied_mask` exactly,
            # under an honest key, in a form no camera has an analogue for. This path is
            # exactly how the paired memoryless control is built.
            ov = (options or {}).get("water_plants") or {}
            if "occupied" in ov:
                occupied = torch.as_tensor(ov["occupied"], dtype=torch.bool).reshape(b, k)
                assert int(occupied.sum(-1).min()) == int(occupied.sum(-1).max()) == n, (
                    f"the water_plants['occupied'] override must hold exactly n_plants={n} "
                    f"sites in every row; got counts {occupied.sum(-1).tolist()}. A shorter "
                    "row makes _occupied_plant_poses emit HIDDEN_Z, which reconstructs "
                    "occupied_mask from the observation."
                )

            self.occupied_mask[env_idx] = occupied
            self.refill_dwell[env_idx] = torch.as_tensor(dwell, dtype=torch.int32)
            self.instruction_idx[env_idx] = torch.as_tensor(instr, dtype=torch.int32)

            # --- the plants ---------------------------------------------------
            jit = torch.as_tensor(site_jit, dtype=torch.float32)
            along = self._along[env_idx].unsqueeze(1)  # (b, 1, 3)
            across = self._across[env_idx].unsqueeze(1)
            pos = (
                self.site_pos[env_idx]
                + along * (jit[:, :, :1] * self.cfg.site_jitter_along)
                + across * (jit[:, :, 1:2] * self.cfg.site_jitter_across)
            )
            # Mesh bottom exactly on the counter, no settling clearance: a kinematic
            # body never falls, so a clearance would leave every pot hovering. The
            # pot top then lands at counter + 0.30 m, which is the height Task 2's
            # probe measured the pour pose's reachability against.
            pos[:, :, 2] = pos[:, :, 2] + self._plant_rest_lift
            hidden = pos.clone()
            hidden[:, :, 2] = HIDDEN_Z
            pos = torch.where(occupied.unsqueeze(-1), pos, hidden)
            self.plant_home[env_idx] = pos
            for s, plant in enumerate(self.plants):
                plant.set_pose(Pose.create_from_pq(p=self.plant_home[env_idx, s]))

            # --- the cup ------------------------------------------------------
            cup_pos = self._cup_home[env_idx].clone()
            cup_pos[:, :2] += torch.as_tensor(cup_jit, dtype=torch.float32) * self.cfg.cup_jitter_xy
            # Yaw only: the tilt predicate compares the cup's body +Z against world
            # +Z, and that is a statement about "upright" only if the spawn never
            # rolls or pitches it.
            self.cup.set_pose(Pose.create_from_pq(p=cup_pos, q=self._yaw_quat(b)))

            # The bucket is kinematic, so its mesh bottom sits exactly on the floor
            # rather than being dropped onto it — and re-posing it every reset is what
            # makes the station a constant of the episode rather than of the build.
            bucket_pos = self.station_pos[env_idx].clone()
            bucket_z = torch.full(
                (b, 1), self._bucket_rest_lift + float(self.cfg.station_z_lift)
            )
            self.bucket.set_pose(
                Pose.create_from_pq(p=torch.cat([bucket_pos[:, :2], bucket_z], dim=1))
            )

            # --- the robot's start pose ---------------------------------------
            base_j = torch.as_tensor(base_jit, dtype=torch.float32)
            dock = self.refill_dock[env_idx]
            p = torch.zeros((b, 3))
            p[:, :2] = dock[:, :2] + base_j[:, :2] * self.cfg.start_jitter_xy
            # Push the base out of the cup's rotation footprint if the draw put it
            # inside — see `base_cup_min_clearance`. Radially, away from the cup, so
            # the jittered heading and the dock zone are both preserved: the shift is
            # at most `start_jitter_xy` and only ever increases the distance to the
            # cup the robot is about to pick up.
            _to_base = p[:, :2] - cup_pos[:, :2]
            _d = torch.linalg.norm(_to_base, dim=-1, keepdim=True)
            _need = float(self.cfg.base_cup_min_clearance)
            _unit = _to_base / _d.clamp_min(1e-6)
            p[:, :2] = torch.where(_d < _need, cup_pos[:, :2] + _unit * _need, p[:, :2])
            p[:, 2] = self.agent.robot.pose.p[env_idx][:, 2]
            yaw = dock[:, 2] + base_j[:, 2] * self.cfg.start_jitter_yaw
            q = torch.stack(
                [torch.cos(yaw / 2), torch.zeros(b), torch.zeros(b), torch.sin(yaw / 2)], dim=1
            )
            self.agent.robot.set_pose(Pose.create_from_pq(p=p, q=q))

            # --- task state is not sim state; mask it by hand -----------------
            self.watered_mask[env_idx] = False
            self.has_water[env_idx] = False
            self.doses_left[env_idx] = 0
            self.poured_since_left_pot[env_idx] = False
            self.dip_done[env_idx] = False
            self.repeat_seen[env_idx] = False
            self.first_decision_made[env_idx] = False
            self.first_decision_ok[env_idx] = False
            self.scene_untampered[env_idx] = True
            self.budget_left[env_idx] = self.cfg.n_plants

            if self.cfg.cup_starts_full:
                # After the robot's pose and after the state reset, both deliberately:
                # the TCP is only in its start position once the base is placed, and
                # the reset above clears `has_water`/`doses_left`, so setting them
                # earlier is silently undone.
                #
                # The fingers are opened in qpos rather than driven open, and the cup
                # is placed between them in the same instant. The cup is a *dynamic*
                # body: open the gripper and step, and it falls out before anything
                # can close on it (measured — `is_grasping` False, the stage refused
                # in 1.0 s). Placed inside already-open fingers it has nowhere to go
                # in the frame before the oracle closes them.
                grip = [
                    self.agent.robot.active_joints_map[n].active_index[0].item()
                    for n in ("l_gripper_finger_joint", "r_gripper_finger_joint")
                ]
                qpos = self.agent.robot.get_qpos()
                qpos[env_idx[:, None], torch.as_tensor(grip, device=qpos.device)] = (
                    self.cfg.cup_starts_full_grip_open
                )
                self.agent.robot.set_qpos(qpos)
                tcp = self.agent.tcp.pose.p[env_idx]
                self.cup.set_pose(Pose.create_from_pq(p=tcp, q=self._yaw_quat(b)))
                self.has_water[env_idx] = True
                self.doses_left[env_idx] = int(self.cfg.doses_per_fill)
                # No bucket trip will happen, so the refill counter starts spent.
                self.trip_index[env_idx] = self.cfg.n_plants
            self.trip_index[env_idx] = 0
            self.refill_timer[env_idx] = 0
            self.pour_hold[env_idx] = 0
            self.double_water_count[env_idx] = 0
            self.distinct_before_repeat[env_idx] = 0
            self._last_eval_step[env_idx] = -1
            self.articulation_home[env_idx] = self._articulation_qpos()[env_idx]

    def _yaw_quat(self, b: int) -> torch.Tensor:
        yaw = torch.rand(b) * 2 * math.pi
        return torch.stack(
            [torch.cos(yaw / 2), torch.zeros(b), torch.zeros(b), torch.sin(yaw / 2)], dim=1
        )

    def _restore_robot(self, env_idx: torch.Tensor):
        """The rest keyframe, for every robot uid.

        `scene_builder.initialize` restores the robot only when uid == "fetch"
        (scene_builder.py:563-577 compares a literal), and even then to *its* pose —
        a dock at a fixture it picked with `rng.choice` — not to the refill dock this
        task starts at. No uid test, as in `burner.py` and `season_dish.py`: this runs
        after `scene_builder.initialize` and simply overwrites it, which is safe and
        idempotent. The base pose is written by `_initialize_episode` right after,
        with the episode's jitter. (The uid test in `_fix_ds_fetch_collision_bits`
        is a different matter and must stay.)
        """
        if self.agent is None:
            return
        self.agent.robot.set_qpos(self.agent.keyframes["rest"].qpos)

    # --------------------------------------------------------------- evaluate --

    def evaluate(self) -> dict:
        """Runs every step, once inside `reset()` at t = 0, and out of band from the
        diagnostics. Every latch below advances only when `elapsed_steps` has actually
        moved, so a second call within one simulation step returns the same dict —
        `season_dish.py`'s hazard, and review §11's.

        Nothing here moves an actor. The plants are kinematic and are posed once per
        episode; there is no phase teleport to schedule, because there is no cue.
        """
        step = self.elapsed_steps.to(torch.int32)
        advance = step != self._last_eval_step
        rows = torch.arange(self.num_envs, device=self.device)

        # --- where the base is ----------------------------------------------
        # `agent.base_link`, not `agent.robot`: Fetch drives through root_x/root_y/
        # root_z_rotation joints, so the articulation's root pose stays at the spawn
        # point for the whole episode.
        base_pose = self.agent.base_link.pose
        base_p = base_pose.p
        base_yaw = torch.atan2(
            2.0 * (base_pose.q[:, 0] * base_pose.q[:, 3] + base_pose.q[:, 1] * base_pose.q[:, 2]),
            1.0 - 2.0 * (base_pose.q[:, 2] ** 2 + base_pose.q[:, 3] ** 2),
        )
        heading = math.radians(self.cfg.dock_heading_deg)

        d_plant_dock = torch.linalg.norm(
            base_p[:, None, :2] - self.site_dock[:, :, :2], dim=-1
        )
        dyaw_plant = torch.atan2(
            torch.sin(base_yaw.unsqueeze(-1) - self.site_dock[:, :, 2]),
            torch.cos(base_yaw.unsqueeze(-1) - self.site_dock[:, :, 2]),
        ).abs()
        at_plant = (d_plant_dock <= self.cfg.dock_radius) & (dyaw_plant <= heading)

        d_station = torch.linalg.norm(base_p[:, :2] - self.refill_dock[:, :2], dim=-1)
        dyaw_station = torch.atan2(
            torch.sin(base_yaw - self.refill_dock[:, 2]),
            torch.cos(base_yaw - self.refill_dock[:, 2]),
        ).abs()
        at_station = (d_station <= self.cfg.dock_radius) & (dyaw_station <= heading)
        base_static = (
            torch.linalg.norm(self.agent.robot.get_qvel()[:, :3], dim=-1)
            < self.cfg.base_static_speed
        )

        # --- the cup ---------------------------------------------------------
        grasped = self.agent.is_grasping(self.cup, min_force=self.cfg.grasp_min_force)
        cup_p = self.cup.pose.p
        cup_R = self.cup.pose.to_transformation_matrix()[:, :3, :3]
        axis = torch.tensor(self.cfg.pour_axis_body, dtype=torch.float32, device=self.device)
        cos_tilt = (cup_R @ axis)[:, 2].clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        tilt_rad = torch.arccos(cos_tilt)
        tilted = tilt_rad >= math.radians(self.cfg.pour_tilt_deg)

        # --- the pour, per site ----------------------------------------------
        plant_p = torch.stack([p.pose.p for p in self.plants], dim=1)  # (b, k, 3)
        pot_top = plant_p[:, :, 2] + self._plant_top_lift
        d_pot = torch.linalg.norm(cup_p[:, None, :2] - plant_p[:, :, :2], dim=-1)
        clearance = cup_p[:, None, 2] - pot_top
        over_pot = (
            (d_pot <= self.cfg.pour_xy_radius)
            & (clearance >= self.cfg.pour_min_clearance)
            & (clearance <= self.cfg.pour_max_clearance)
        )
        # Re-arm once the cup has left every pot, not once it is level again: the
        # levelling is best effort and measurably refuses at the ~65 deg the pour
        # leaves behind, and tying the re-arm to it strands the remaining doses
        # (measured — seed 1 reached the second pot with water, upright tolerance
        # unmet, and committed nothing).
        self.poured_since_left_pot = self.poured_since_left_pot & over_pot.any(dim=-1)
        rearmed = (
            ~self.poured_since_left_pot
            if int(self.cfg.doses_per_fill) > 1
            else torch.ones_like(self.has_water)
        )
        ready = (grasped & tilted & self.has_water & base_static & rearmed).unsqueeze(-1)
        pour_now = over_pot & at_plant & self.occupied_mask & ready

        # --- the refill ------------------------------------------------------
        bucket_p = self.bucket.pose.p
        d_bucket = torch.linalg.norm(cup_p[:, :2] - bucket_p[:, :2], dim=-1)
        bucket_clearance = cup_p[:, 2] - (bucket_p[:, 2] + self._bucket_top_lift)
        over_bucket = (
            (d_bucket <= self.cfg.pour_xy_radius)
            & (bucket_clearance >= self.cfg.pour_min_clearance)
            & (bucket_clearance <= self.cfg.pour_max_clearance)
        )
        # Per-joint difference, wrapped into [-pi, pi] on the revolute joints only
        # (see `_carry_wraps`). `torch.remainder` is used rather than `%` on a
        # negative float for the same reason numpy's `mod` is preferred there: the
        # sign of the result must follow the divisor, not the dividend.
        _delta = self.agent.robot.get_qpos()[:, self._carry_qpos_idx] - self._carry_qpos
        _wrapped = torch.remainder(_delta + math.pi, 2 * math.pi) - math.pi
        carry_error = torch.where(self._carry_wraps, _wrapped, _delta).abs().amax(dim=-1)
        carry_ok = carry_error <= self.cfg.carry_pose_tol_rad
        can_refill = (
            at_station
            & grasped
            & ~self.has_water
            & (self.budget_left > 0)
            & (self.trip_index < self.cfg.n_plants)
        )
        dipping = can_refill & over_bucket
        refilling = can_refill & self.dip_done & carry_ok & base_static

        # --- the world as a notebook -----------------------------------------
        plant_drift = torch.linalg.norm(plant_p[:, :, :2] - self.plant_home[:, :, :2], dim=-1)
        plant_moved = (plant_drift > self.cfg.tamper_move_tol).any(dim=-1)
        joint_drift = (self._articulation_qpos() - self.articulation_home).abs()
        joints_moved = (
            (joint_drift > self.cfg.tamper_move_tol).any(dim=-1)
            if joint_drift.shape[-1]
            else torch.zeros_like(plant_moved)
        )
        tampered = plant_moved | joints_moved

        if bool(advance.any()):
            self._advance(advance, dipping, at_station, refilling, pour_now, tampered, rows)

        verdict = score_watering(
            self.watered_mask, self.occupied_mask, self.budget_left, self.scene_untampered
        )
        at_plant_id = torch.where(
            at_plant.any(dim=-1),
            at_plant.to(torch.int64).argmax(dim=-1),
            torch.full((self.num_envs,), -1, dtype=torch.int64, device=self.device),
        )
        return {
            "success": verdict["success"],
            "fail": verdict["fail"],
            "episode_over": verdict["episode_over"],
            "exact_cover": verdict["exact_cover"],
            "watered_count": verdict["watered_count"],
            "omission_count": verdict["omission_count"],
            "stray_count": verdict["stray_count"],
            "double_water_count": self.double_water_count,
            "distinct_before_repeat": self.distinct_before_repeat,
            "first_decision_made": self.first_decision_made,
            "first_decision_ok": self.first_decision_ok,
            "budget_left": self.budget_left,
            "has_water": self.has_water,
            "trip_index": self.trip_index,
            "refill_timer": self.refill_timer,
            "dip_done": self.dip_done,
            "at_station": at_station,
            "at_plant_id": at_plant_id,
            "at_plant_any": at_plant.any(dim=-1),
            "carry_ok": carry_ok,
            "carry_error": carry_error,
            "is_grasped": grasped,
            "tilt_rad": tilt_rad,
            "tilted": tilted,
            "over_bucket": over_bucket,
            "pour_hold": self.pour_hold,
            "scene_untampered": self.scene_untampered,
            "plant_drift": plant_drift,
            # Ground truth for the oracle and the trajectory file. `info` is not an
            # observation: only what `_get_obs_extra` copies out reaches a policy
            # (station_checklist.py:842-850), and a sighted oracle has to be able to
            # read the answer from somewhere.
            "occupied_mask": self.occupied_mask,
            "watered_mask": self.watered_mask,
            "refill_dwell": self.refill_dwell,
        }

    def _advance(self, advance, dipping, at_station, refilling, pour_now, tampered, rows):
        """Every mutation `evaluate()` makes, in one place and behind one guard."""
        self.scene_untampered = self.scene_untampered & ~(advance & tampered)

        # The dip arms the refill; leaving the station disarms it, so a dip cannot be
        # banked and spent on the next trip.
        self.dip_done = torch.where(
            advance, (self.dip_done | dipping) & at_station, self.dip_done
        )

        self.refill_timer = torch.where(
            advance & refilling,
            self.refill_timer + 1,
            torch.where(advance, torch.zeros_like(self.refill_timer), self.refill_timer),
        )
        # `refill_dwell[:, trip_index]` would be torch advanced indexing on a (b,)
        # index and would give a (b, b) matrix — every env's dwell crossed with every
        # other's, silently. It is a gather.
        dwell = self.refill_dwell.gather(
            1, self.trip_index.clamp(0, self.cfg.n_plants - 1).long().unsqueeze(-1)
        ).squeeze(-1)
        filled = advance & refilling & (self.refill_timer >= dwell)
        if bool(filled.any()):
            self.has_water = self.has_water | filled
            self.doses_left = torch.where(
                filled,
                torch.full_like(self.doses_left, int(self.cfg.doses_per_fill)),
                self.doses_left,
            )
            self.trip_index = self.trip_index + filled.to(torch.int32)
            self.refill_timer = torch.where(
                filled, torch.zeros_like(self.refill_timer), self.refill_timer
            )
            self.dip_done = self.dip_done & ~filled

        self.pour_hold = torch.where(
            advance.unsqueeze(-1) & pour_now,
            self.pour_hold + 1,
            torch.where(advance.unsqueeze(-1), torch.zeros_like(self.pour_hold), self.pour_hold),
        )
        done = self.pour_hold >= self.cfg.hold_steps
        commit = advance & done.any(dim=-1)
        if bool(commit.any()):
            self._commit_pour(commit, done, rows)

        self._last_eval_step = torch.where(
            advance, self.elapsed_steps.to(torch.int32), self._last_eval_step
        )

    def _commit_pour(self, commit: torch.Tensor, done: torch.Tensor, rows: torch.Tensor):
        """Spend a dose on one plant. This is the event the endpoints are measured on.

        Which plant cannot be ambiguous: `pour_xy_radius` is 0.10 m and the measured
        site spacing is 0.3067 m, so the cup is over at most one pot. `argmax` over an
        all-False row would answer 0, which is why the caller gates on `.any()` first.
        """
        s = done.to(torch.int64).argmax(dim=-1)
        already = self.watered_mask[rows, s]
        correct = ~already

        repeat = commit & already
        fresh = commit & correct & ~self.repeat_seen
        self.distinct_before_repeat = self.distinct_before_repeat + fresh.to(torch.int32)
        self.repeat_seen = self.repeat_seen | repeat
        self.double_water_count = self.double_water_count + repeat.to(torch.int32)

        # The first pour cannot be wrong — every plant is dry — so "the first decision
        # with a wrong option available" is the *second* one, and the blind rate on it
        # is 2/3. The semantics differ from StationChecklist's on purpose; both are
        # reported and neither is a substitute for the other.
        second = commit & (self.budget_left == self.cfg.n_plants - 1)
        self.first_decision_ok = torch.where(second, correct, self.first_decision_ok)
        self.first_decision_made = self.first_decision_made | second

        self.watered_mask[rows[commit], s[commit]] = True
        self.doses_left = torch.clamp(self.doses_left - commit.to(torch.int32), min=0)
        self.has_water = self.has_water & (self.doses_left > 0)
        self.poured_since_left_pot = self.poured_since_left_pot | commit
        self.budget_left = self.budget_left - commit.to(torch.int32)
        self.pour_hold = torch.where(
            commit.unsqueeze(-1), torch.zeros_like(self.pour_hold), self.pour_hold
        )

    # -------------------------------------------------------------------- obs --

    def _get_obs_extra(self, info: dict) -> dict:
        """What the policy is allowed to see. **The omissions are the design.**

        Never emitted, in any observation mode: `watered_mask`, `occupied_mask`,
        `has_water`, `budget_left`, **`trip_index`**, `refill_dwell`, `refill_timer`,
        `dip_done`, `pour_hold`, `double_water_count`, `distinct_before_repeat`,
        `repeat_seen`, `first_decision_*`, `scene_untampered`.

        `trip_index` deserves its own sentence, because the design document listed it
        as checkpoint state and forgot to list it here. It is the purest progress
        counter in the task: it *is* how many plants have been watered. Emitting it
        would not weaken the measurement, it would end it — the construct under test
        is the agent's own action history, and handing an agent its own history
        switches off exactly the thing being measured. `station_checklist.py:906-911`
        is the model for that strictness, and `AGENTS.md` records that nothing
        enforces it automatically; `tests/test_water_plants.py` adds a parsed check so
        that stops being true for this task.

        `info` is **not** an observation. `evaluate()` returns the ground truth
        because a sighted oracle has to read it from somewhere; only what this method
        copies out reaches a policy.

        What *is* emitted: proprioception, and under `use_state` the poses of things a
        camera can already see. Two facts about the plants have to be told apart and
        this method is where the distinction is enforced:

        - **which sites hold a plant is perceptible.** A camera sees the pots, and
          the 0.222 floor is stated for an agent that can see them, so withholding it
          would make a state-mode run measure perception rather than memory. It goes
          out — as `n_plants` poses of the occupied sites, in site order
          (`_occupied_plant_poses`), never as `occupied_mask` and never as `k_sites`
          poses with `HIDDEN_Z` standing in for the empties, which is the same fact
          in a form no camera has an analogue for.
        - **whether a plant has been watered is not perceptible, and is the
          measurement.** Nothing in the scene records it: the pour is non-contact,
          the plants are kinematic, and the one piece of water in the scene — the
          fixed surface inside the refill basin (`water_visual`) — is static scenery
          that never reads `watered_mask` or `has_water`. That is the whole reason
          the cup is *not* rendered with a fill level and the pour has no stream:
          `has_water` is on the withheld list above, so a visible fill would hand a
          vision policy the very latch this method omits, and a stream that only
          appears when a dose commits would mark a real pour apart from a tip of an
          empty cup. `tools/probes/w8_watered_is_invisible.py` re-proves the render
          is byte-identical dry vs wet on all four cameras. Nothing here emits it
          either, in any form.

        Key order is fixed and no key is named for a target: in state mode the dict is
        flattened in insertion order, so a role-ordered key would put the answer at a
        constant index. Nothing here is conditional on time either — the observation
        space is frozen from the t = 0 dict (sapien_env.py:330).
        """
        obs = dict(
            tcp_pose=self.agent.tcp_pose.raw_pose,
            base_pose=self.agent.base_link.pose.raw_pose,
            base_qvel=self.agent.robot.get_qvel()[:, :3],
            joint_qpos=self.agent.robot.get_qpos()[:, self._carry_qpos_idx],
            instruction_id=torch.nn.functional.one_hot(
                self.instruction_idx.long(), num_classes=len(self.cfg.instructions)
            ).to(torch.float32),
        )
        if self.obs_mode_struct.use_state:
            obs.update(
                cup_pose=self.cup.pose.raw_pose,
                plant_pose=self._occupied_plant_poses(),
                station_pos=self.station_pos,
                dock_pos=self.site_dock.reshape(self.num_envs, -1),
            )
        return obs

    def _occupied_plant_poses(self) -> torch.Tensor:
        """`(b, n_plants * 7)` — the poses of the sites that hold a plant, by site index.

        Exactly `n_plants` of them, so the shape is fixed (N is a family constant and
        the observation space is frozen from the t = 0 dict), and **no `HIDDEN_Z`
        ever reaches the observation**. That sentinel is the reason this method
        exists: emitting all `k_sites` poses would put z = 1000 in the vector for
        every empty site, and `z > 100` reconstructs `occupied_mask` exactly — under
        an honest-looking key, in a form no camera has any analogue for.

        **That sentence is a check, not a habit**, and it was a habit until the final
        review: the `[:n]` slice below drops a `HIDDEN_Z` only while exactly `n` sites
        are occupied, and `occupied_mask` has two writers. The permutation draw in
        `_initialize_episode` sets exactly `n` by construction; the certification
        override beside it sets whatever the caller passed, and a two-of-five mask was
        reproduced emitting z `[1.1, 1.1, 1000.0]`. The override now asserts the count
        (`_initialize_episode`, the `"occupied" in ov` branch), which is what this
        paragraph rests on — `test_the_occupancy_override_guard_runs_on_a_short_mask`
        executes that very assert, and
        `test_a_short_occupancy_mask_would_put_HIDDEN_Z_in_the_observation` is the
        control that shows what it is buying.

        The information itself is not the problem and is not withheld. Which sites
        hold plants is **perceptible**: a camera sees the pots, and the 0.222 floor
        assumes an agent that can see them, so hiding it would make a state-mode run
        measure perception instead of memory. What is *not* perceptible, and is the
        whole measurement, is whether a plant has already been watered — nothing in
        the scene records that, and nothing here emits it.

        The order is **strictly by site index**, and that is load-bearing rather than
        tidy: an order correlated with anything about watering — most watered first,
        most recent last — would hand over the answer in the permutation.
        """
        n, k = self.cfg.n_plants, self.cfg.k_sites
        idx = torch.arange(k, device=self.device)
        # Occupied sites keep their own index, empty ones sort to the end. Exactly n
        # sites are in the first group — the draw by construction, the certification
        # override by assert — so the ties among the empty ones never reach the slice
        # and the result does not depend on how they break.
        rank = torch.where(self.occupied_mask, idx, torch.full_like(idx, k))
        order = rank.sort(dim=-1).indices[:, :n]
        poses = torch.stack([p.pose.raw_pose for p in self.plants], dim=1)  # (b, k, 7)
        chosen = poses.gather(1, order.unsqueeze(-1).expand(-1, -1, poses.shape[-1]))
        return chosen.reshape(self.num_envs, -1)

    def get_language_instruction(self, **kwargs):
        return [self.cfg.instructions[int(i)] for i in self.instruction_idx.tolist()]

    # ------------------------------------------------------------------ state --

    def _task_state(self) -> dict:
        """The task's memory, keyed exactly as `TASK_STATE_KEYS`."""
        return {
            "watered_mask": self.watered_mask.clone(),
            "occupied_mask": self.occupied_mask.clone(),
            "has_water": self.has_water.clone(),
            "budget_left": self.budget_left.clone(),
            "trip_index": self.trip_index.clone(),
            "refill_dwell": self.refill_dwell.clone(),
            "refill_timer": self.refill_timer.clone(),
            "dip_done": self.dip_done.clone(),
            "pour_hold": self.pour_hold.clone(),
            "double_water_count": self.double_water_count.clone(),
            "distinct_before_repeat": self.distinct_before_repeat.clone(),
            "repeat_seen": self.repeat_seen.clone(),
            "first_decision_made": self.first_decision_made.clone(),
            "first_decision_ok": self.first_decision_ok.clone(),
            "scene_untampered": self.scene_untampered.clone(),
            "instruction_idx": self.instruction_idx.clone(),
            # Flattened, and it has to be: `get_state()` runs the dict through
            # `common.flatten_state_dict`, which ends in `torch.hstack` — a rank-3
            # buffer raises "Tensors must have same number of dimensions: got 2 and
            # 3" there and nowhere near this line (measured 2026-08-22, .venv-cpu).
            # `restore_task_tensor` reshapes it back from the attribute's own shape.
            "plant_home": self.plant_home.reshape(self.num_envs, -1).clone(),
            "articulation_home": self.articulation_home.clone(),
            "last_eval_step": self._last_eval_step.clone(),
        }

    def _restore_task_state(self, state: dict) -> None:
        """Tolerates a dict with no task keys, because one is routine.

        `BaseEnv.set_state(flat)` rebuilds a dict holding only "actors" and
        "articulations" (sapien_env.py:1309-1327), so every task key is gone by the
        time it arrives here. `restore_task_tensor` is what makes a *recorded*
        trajectory restorable too: replay hands back numpy, and a `(1,)` batch that
        has been indexed comes back as a bare `numpy.int64`.
        """
        attrs = {name: name for name in TASK_STATE_KEYS}
        attrs["last_eval_step"] = "_last_eval_step"
        for key, attr in attrs.items():
            if key in state:
                setattr(
                    self, attr, restore_task_tensor(getattr(self, attr), state[key], self.device)
                )

    def get_state_dict(self) -> dict:
        """Every latch and every draw, so `env.set_state(env.get_state())` is lossless.

        For a memory benchmark this is the state that matters most: the whole answer
        is generated during the episode, so a dropped buffer turns a recorded success
        into a replayed failure. `refill_dwell` and `trip_index` are both here on
        purpose — restoring mid-refill has to resume the *same* dwell, and without the
        index the replay cannot know which one is running.
        """
        state = super().get_state_dict()
        state.update(self._task_state())
        return state

    def set_state_dict(self, state: dict, env_idx: torch.Tensor = None):
        # super() first: sim state lands, then the task's flags on a consistent scene.
        super().set_state_dict(state, env_idx)
        self._restore_task_state(state)
