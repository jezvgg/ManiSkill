"""Helpers for building on top of RoboCasa kitchens, with the sharp edges wrapped.

Deliberately imports no task. `docs/review-inherited-code.md` §12 records that
`my_robocasa_takeitback.py` importing helpers out of `my_robocasa.py` drags an
unrelated `@register_env` along as an import side effect; a task that wants a
helper should not have to register a second environment to get one. That is why
`get_actor_size` and `degree_to_quanterion` live here rather than in the task
that happened to define them first — §12's "they belong in a non-task module",
done. `scene_file_imports_no_other_task` in tests/test_task_conventions.py is
what stops the next one drifting back.

Everything here exists because the upstream API has a trap at that spot. Each
function's docstring names the trap and cites it.
"""

from __future__ import annotations

import os

import numpy as np
import sapien
import torch

# ObjCat instances are built by walking the asset tree at import time
# (kitchen_object_utils.py:99-111). Without the 3.7 GB dataset every mjcf_paths
# is [], which is why objaverse_mjcf() below raises rather than returning "".
from mani_skill.utils.geometry.rotation_conversions import axis_angle_to_quaternion
from mani_skill.utils.scene_builder.robocasa.objects.kitchen_objects import OBJ_CATEGORIES

# The one root both asset trees hang off: `BASE_ASSET_ZOO_PATH`, which every
# `ObjCat.mjcf_paths` above is built from, is `ROBOCASA_ASSET_DIR / "objects"`
# (kitchen_object_utils.py:19). Importing it is how `load_accessory_actor` stays
# on the same source of truth as `objaverse_mjcf` instead of pasting `$HOME`.
from mani_skill.utils.scene_builder.robocasa.utils.scene_utils import ROBOCASA_ASSET_DIR
from mani_skill.utils.structs import Actor, Pose


def parking_pose(env) -> Pose:
    """Where the robot waits while the kitchen is built: 5 m up, one pose PER ENV.

    RoboCasa's scene builder writes each env's dock into
    `env.agent.robot.initial_pose.raw_pose[scene_idx]` (scene_builder.py:394), so the
    initial pose must carry `num_envs` rows. Passing a single `sapien.Pose` — what all
    eleven scenes did until 2026-09-08 — gives one row, and `num_envs > 1` dies with
    `IndexError: index 1 is out of bounds` while the second kitchen is built. That is
    why GPU-parallel simulation never ran here: it was not the physics, it was this one
    argument. ManiSkill's own RoboCasa kitchen builds the batch the same way
    (`kitchen.py:278-281`). For `num_envs == 1` the tensor is the old pose exactly.

    Example:
        >>> def _load_agent(self, options):                       # doctest: +SKIP
        ...     super()._load_agent(options, parking_pose(self))
    """
    p = torch.zeros((env.num_envs, 3), device=env.device)
    p[:, 2] = 5.0
    return Pose.create_from_pq(p=p)


def get_actor_size(actor: Actor):
    """The actor's world-space AABB extent, from sub-scene zero.

    Inherited verbatim from `my_robocasa.py`, where it was defined next to a
    `@register_env`; moved here by §12. `_bodies` is per sub-scene and the `[0]`
    picks the first, so this is one env's answer, not a batched one.

    Example:
        >>> bowl = loader.parse(bowl_xml)["actor_builders"][0].build(name="bowl")
        >>> pos[2] += counter_size[2] / 2 + get_actor_size(bowl)[2] / 2  # sit it on top
    """
    bodies = np.array([body.get_global_aabb_fast() for body in actor._bodies])
    return (bodies.max(axis=1) - bodies.min(axis=1))[0]


def degree_to_quanterion(x: int = 0, y: int = 0, z: int = 0):
    """Euler degrees to a quaternion. Misspelling inherited, and load-bearing.

    Renaming it is a public-API change (`scenes/__init__.py:__all__`), so it is
    left as §12 found it. Same caveat as there: it builds a CPU tensor and
    returns `(4,)` — one shared orientation, no per-env variation.

    Example:
        >>> q = degree_to_quanterion(z=180)  # turn it to face the other way
        >>> self.cup.set_pose(Pose.create_from_pq(p=self.cup_pos, q=q))
    """
    return axis_angle_to_quaternion(
        torch.Tensor([x * torch.pi / 180, y * torch.pi / 180, z * torch.pi / 180])
    )


def require_get_fixture(scene_builder, fixtures: dict, ident, scene_idx=None):
    """`scene_builder.get_fixture` with an error you can act on.

    Three upstream behaviours this wraps, all in
    mani_skill/utils/scene_builder/robocasa/scene_builder.py:

    - a miss is a bare `assert len(matches) > 0` (:653) with no message, so the
      traceback never says which fixture or which kitchen;
    - `ref=` is dead: :686 calls `self.rng`, and neither RoboCasaSceneBuilder nor
      the SceneBuilder base ever defines that attribute. Never pass it;
    - a substring matching several fixtures returns a *random* one via
      `self.env._episode_rng.choice` (:656), which also perturbs the shared
      episode stream. Call this once per env and cache the result.

    Args:
        scene_builder: the `RoboCasaSceneBuilder` of the task (`self.scene_builder`).
        fixtures: one env's fixture dict, `scene_builder.scene_data[i]["fixtures"]`.
        ident: fixture name or unique substring, e.g. `"stove"`, `"counter_main"`.
        scene_idx: only for the error message.

    Returns:
        The fixture object (a `Counter`, `Stove`, ... instance).

    Example:
        >>> fixtures = self.scene_builder.scene_data[0]["fixtures"]
        >>> stove = require_get_fixture(self.scene_builder, fixtures, "stove", scene_idx=0)
        >>> top, along, across, size = fixture_frame(stove)
    """
    try:
        return scene_builder.get_fixture(fixtures, ident)
    except AssertionError:
        where = "" if scene_idx is None else f" (scene_idx={scene_idx})"
        raise KeyError(
            f"no fixture matching {ident!r} in this kitchen{where}. "
            f"RoboCasa draws a layout per env, and not every layout has every "
            f"fixture — this is what colab.find_kitchens() probes for. "
            f"Available: {sorted(fixtures)}"
        ) from None


def counter_frame(counter) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(top_centre, along, across)` for a counter fixture, in world coordinates.

    `top_centre` is the middle of the work surface: the fixture's origin is at its
    centre, so the top is `pos[2] + size[2] / 2` — the same arithmetic
    `Counter.get_reset_regions` uses (fixtures/counter.py:653).

    `along` and `across` are unit vectors in the counter's own yaw frame, so a
    placement expressed as "0.28 m along, 0.10 m toward the robot" survives a
    kitchen whose counter is rotated. `Fixture.rot` is a Z euler angle
    (fixtures/fixture.py:127).

    Everything is copied: `Fixture.pos` is a live array, and `my_robocasa.py:114`
    aliases it — review §12.

    Args:
        counter: a RoboCasa `Counter` fixture (from `require_get_fixture`).

    Returns:
        `(top_centre, along, across)`, three float32 arrays of shape (3,); `along` and
        `across` are unit vectors in the world frame.

    Example:
        >>> top, along, across = counter_frame(counter)
        >>> spawn = top + 0.28 * along - 0.10 * across + np.array([0, 0, 0.05])
    """
    pos = np.asarray(counter.pos, dtype=np.float64).copy()
    size = np.asarray(counter.size, dtype=np.float64).copy()
    yaw = float(getattr(counter, "rot", 0.0) or 0.0)

    top_centre = pos.copy()
    top_centre[2] += size[2] / 2.0

    along = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    across = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    return top_centre.astype(np.float32), along.astype(np.float32), across.astype(np.float32)


def fixture_frame(fixture) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`(top_centre, along, across, size)` for any fixture, in world coordinates.

    `counter_frame` above is this for counters; this one is the general case and
    also hands back the extent, which a task needs to place something on the near
    edge rather than in the middle.

    Everything is copied — `Fixture.pos` is a live array and aliasing it is review
    §12.

    Args:
        fixture: any RoboCasa fixture (`Stove`, `Counter`, `Sink`, ...).

    Returns:
        `(top_centre, along, across, size)`; `size` is the fixture's (x, y, z) extent in
        its own frame, so `top_centre - 0.5 * size[1] * across` is the near edge.

    Example:
        >>> top, along, across, size = fixture_frame(stove)
        >>> near_edge = top - 0.5 * size[1] * across
    """
    pos = np.asarray(fixture.pos, dtype=np.float64).copy()
    size = np.asarray(fixture.size, dtype=np.float64).copy()
    yaw = float(getattr(fixture, "rot", 0.0) or 0.0)

    top_centre = pos.copy()
    top_centre[2] += size[2] / 2.0

    along = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    across = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    return (
        top_centre.astype(np.float32),
        along.astype(np.float32),
        across.astype(np.float32),
        size.astype(np.float32),
    )


# The fixture classes RoboCasa itself considers worth parking a robot in front of
# (scene_builder.py:531-546). Hood and Oven are dropped: a Hood hangs above head
# height and an Oven's front face is at floor level, so neither gives a service
# point an arm can reach from a standing dock pose.
STATION_FIXTURE_CLASSES = (
    "CoffeeMachine",
    "Toaster",
    "Stove",
    "Stovetop",
    "SingleCabinet",
    "HingeCabinet",
    "OpenCabinet",
    "Drawer",
    "Microwave",
    "Sink",
    "Fridge",
    "Dishwasher",
)


def station_candidates(
    fixtures: dict,
    classes: tuple[str, ...] = STATION_FIXTURE_CLASSES,
    min_top_z: float = 0.30,
    max_top_z: float = 1.60,
) -> list[str]:
    """Names of fixtures usable as service stations, in a deterministic order.

    Sorted by name and nothing else. This is load-bearing: any ordering that
    depends on geometry would make the oracle's "visit stations in index order"
    a spatial policy, which is trap 3 of the spec — the visit order would leak
    into the demonstration data. And nothing here may touch `_episode_rng`,
    because consuming it would let station choice correlate with the draw of which
    stations are completed (spec §8.3).

    Args:
        fixtures: one env's fixture dict.
        classes: fixture class names that count as stations.
        min_top_z, max_top_z: keep fixtures whose top surface lies in this band (m).

    Returns:
        Sorted list of fixture names.

    Example:
        >>> names = station_candidates(fixtures)
        >>> docks = np.stack([dock_pose_for(sb, fixtures, n)[0] for n in names])
        >>> chosen = select_spread_stations(names, docks, k=6)
    """
    out = []
    for name, fixture in fixtures.items():
        if type(fixture).__name__ not in classes:
            continue
        top_z = float(np.asarray(fixture.pos)[2]) + float(np.asarray(fixture.size)[2]) / 2.0
        if min_top_z <= top_z <= max_top_z:
            out.append(name)
    return sorted(out)


def dock_pose_for(
    scene_builder, fixtures: dict, name: str, offset=None
) -> tuple[np.ndarray, float]:
    """`(base_position, base_yaw)` for docking at the named fixture.

    Wraps `RoboCasaSceneBuilder.compute_robot_base_placement_pose`
    (scene_builder.py:688-761), which stands the robot `front_face_size` back from
    the near edge of whatever counter the fixture sits on — 0.8 m for `ds_fetch`
    once `agents/ds_fetch/__init__.py` has registered it.

    `offset=(along, toward)` is added to the dock in the *base fixture's* frame
    before the rotation to world (scene_builder.py:743-745:
    `base_to_edge[1] = cntr_y - front_face_size + offset[1]`, then the fixture's
    `rot` is applied) — so `offset[1] > 0` moves the base **toward** the fixture,
    shrinking the standoff, and `offset[0]` slides it along the fixture's face.
    The sign is read from the source, identical in the b22 fork, 3.0.1 and b14
    (K18); the first container run of T3 asserted it live (journal 2026-08-18).
    Note the branch right below it: for HousingCabinet/Fridge and any fixture
    whose base counter has "stack" in its name, upstream subtracts another
    0.10 m from the standoff — part of why station docks measure closer than the
    nominal 0.8 m on some fixtures.

    Note the returned yaw is the *counter's* facing plus 90°, so two stations on
    the same counter get the same heading and differ only by position along it.
    That is exactly why the task offers a `nearest` zone mode as well as a radius.

    Args:
        scene_builder: the task's `RoboCasaSceneBuilder`.
        fixtures: one env's fixture dict.
        name: exact fixture name (a key of `fixtures`).
        offset: optional `(along, toward)` metres in the base fixture's frame;
            None keeps upstream's dock untouched.

    Returns:
        `(base_position, base_yaw)`: float32 (3,) world position of the robot base and
        the yaw in radians.

    Example:
        >>> pos, yaw = dock_pose_for(self.scene_builder, fixtures, "stove_main_group_Stove",
        ...                          offset=(0.0, 0.25))  # 0.25 m closer to the stove
        >>> planner.drive_base(target_pos=pos)   # then rotate_base_z toward the fixture
    """
    pos, ori = scene_builder.compute_robot_base_placement_pose(
        fixtures, fixtures[name], offset=offset
    )
    return np.asarray(pos, dtype=np.float32).copy(), float(np.asarray(ori)[2])


# The classes upstream treats as something a fixture can sit on/in
# (scene_builder.py:703-711, `compute_robot_base_placement_pose` step 1).
BASE_FIXTURE_CLASSES = ("Counter", "Stove", "Stovetop", "HousingCabinet", "Fridge")


def _inside_2d(point_xy, fixture) -> bool:
    """Is `point_xy` inside the fixture's footprint (its `pos ± size/2` box, yawed by `rot`)?"""
    pos = np.asarray(fixture.pos, dtype=np.float64)
    size = np.asarray(fixture.size, dtype=np.float64)
    yaw = float(getattr(fixture, "rot", 0.0) or 0.0)
    d = np.asarray(point_xy, dtype=np.float64)[:2] - pos[:2]
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
    return bool(abs(local[0]) <= size[0] / 2.0 and abs(local[1]) <= size[1] / 2.0)


def base_fixture_for(fixtures: dict, name: str):
    """The counter/stove/housing a fixture sits in — the same lookup the robot dock uses.

    Mirrors step 1 of `RoboCasaSceneBuilder.compute_robot_base_placement_pose`
    (scene_builder.py:700-720): the first `Counter`/`Stove`/`Stovetop`/
    `HousingCabinet`/`Fridge` whose 2-D footprint contains the fixture's `pos`,
    else the fixture itself. Written here (footprint from `pos`/`size`/`rot`,
    class matched by name as `station_candidates` does) so it runs on fakes
    without the scene builder; on kitchen 102 the answers agree with upstream's
    (T4 probe, journal 2026-08-18: drawers/cabinets/dishwasher/sink →
    `counter_*`, stove → itself).

    Why a task wants it: a drawer's own top is 0.47 m but the countertop over
    it is 0.92 m — anything meant to be *seen* at that station (the beacon) has
    to clear the base fixture's top, not the drawer's.

    Args:
        fixtures: one env's fixture dict.
        name: exact fixture name (a key of `fixtures`).

    Returns:
        The base fixture object (possibly `fixtures[name]` itself).

    Example:
        >>> base = base_fixture_for(fixtures, "stack_1_main_group_2")   # doctest: +SKIP
        >>> counter_top_z = float(fixture_frame(base)[0][2])           # doctest: +SKIP
    """
    ref = fixtures[name]
    point = np.asarray(ref.pos, dtype=np.float64)[:2]
    for fx in fixtures.values():
        if type(fx).__name__ not in BASE_FIXTURE_CLASSES:
            continue
        if _inside_2d(point, fx):
            return fx
    if type(ref).__name__ not in BASE_FIXTURE_CLASSES:
        # Say so: on another kitchen this puts the beacon back at the fixture's
        # own top, i.e. possibly inside a counter body again, and only a frame
        # read would show it (T4, journal 2026-08-18).
        print(
            f"[robocasa_utils] base_fixture_for: no Counter/Stove/HousingCabinet/Fridge "
            f"footprint contains {name!r} ({type(ref).__name__}); using the fixture "
            f"itself as its base — heights derived from it may sit inside a counter"
        )
    return ref


def dedup_docks(
    names: list[str], docks: np.ndarray, tol: float = 0.05
) -> tuple[list[str], np.ndarray, dict[str, list[str]]]:
    """Drop fixtures whose dock coincides (within `tol` m in xy) with an earlier one.

    Why: `compute_robot_base_placement_pose` docks the robot in front of the
    *base counter* a fixture sits on, so the three drawers of one cabinet stack
    (`stack_1_main_group_1/2/3` on kitchen 102) share one dock to the millimetre,
    and a sink shares its dock with the cabinet under it. Two stations with one
    dock can never be told apart by the base's position — every zone mode makes
    them one zone — so the task keeps one station per dock, and the
    `len(cand) < k` check then counts *distinct* docks rather than fixtures
    (measured on 102, T4 probe, journal 2026-08-18: candidates share docks at
    0.00 m; the closest distinct pair is 0.50 m apart).

    Deterministic and non-geometric in *which* survives: `names` arrive sorted
    (`station_candidates`) and the first name of a coincident group is kept, so
    the station index still encodes nothing spatial (spec trap 3). Called once,
    in `_load_scene`, before the `len(cand) < k` check (K28);
    `select_spread_stations` is unchanged and runs on the survivors.

    Args:
        names: candidate fixture names, aligned with the rows of `docks`.
        docks: array (n, 3) of dock positions (from `dock_pose_for`); only xy is compared.
        tol: two docks closer than this in xy are one dock.

    Returns:
        `(kept_names, kept_docks, merged)`: the survivors (order preserved), their
        dock rows, and `{survivor: [dropped names]}` for every group that lost a
        member — the log line in `_load_scene` prints it.

    Example:
        >>> names = ["sink", "stack_1_a", "stack_1_b", "stove"]
        >>> docks = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.01, 0.0], [2.0, 0.0, 0.0]])
        >>> kept, kd, merged = dedup_docks(names, docks)
        >>> kept, kd.shape, merged
        (['sink', 'stack_1_a', 'stove'], (3, 3), {'stack_1_a': ['stack_1_b']})
    """
    docks = np.asarray(docks, dtype=np.float64)
    assert len(names) == len(docks), (len(names), docks.shape)
    kept_idx: list[int] = []
    merged: dict[str, list[str]] = {}
    for i, name in enumerate(names):
        owner = None
        for j in kept_idx:
            if float(np.linalg.norm(docks[i, :2] - docks[j, :2])) < tol:
                owner = j
                break
        if owner is None:
            kept_idx.append(i)
        else:
            merged.setdefault(names[owner], []).append(name)
    kept_names = [names[i] for i in kept_idx]
    kept_docks = docks[kept_idx].astype(np.float32) if kept_idx else np.zeros((0, 3), dtype=np.float32)
    return kept_names, kept_docks, merged


def select_spread_stations(names: list[str], dock_pos: np.ndarray, k: int) -> list[str]:
    """Pick `k` stations whose dock poses are as far apart as possible.

    Farthest-point selection seeded with the most distant pair, ties broken by
    name so the result is reproducible. Returns names sorted alphabetically, not in
    selection order: the station *index* the oracle iterates must not encode
    anything geometric.

    Args:
        names: candidate fixture names, aligned with the rows of `dock_pos`.
        dock_pos: array (n, 3) of dock positions (from `dock_pose_for`).
        k: how many to pick; raises `ValueError` if `n < k`.

    Returns:
        `k` names, sorted alphabetically.

    Example:
        >>> select_spread_stations(["a", "b", "c"], np.array([[0, 0, 0], [1, 0, 0], [5, 0, 0]]), 2)
        ['a', 'c']
    """
    n = len(names)
    if n < k:
        raise ValueError(f"need {k} stations, found {n}: {names}")
    d = np.linalg.norm(dock_pos[:, None, :2] - dock_pos[None, :, :2], axis=-1)

    i, j = np.unravel_index(int(np.argmax(d)), d.shape)
    chosen = [int(i), int(j)] if i != j else [0, 1 % n]
    while len(chosen) < k:
        rest = [x for x in range(n) if x not in chosen]
        # Maximise the distance to the nearest already-chosen station.
        best = max(rest, key=lambda x: (float(d[x, chosen].min()), -x))
        chosen.append(best)
    return sorted(names[c] for c in chosen)


def objaverse_mjcf(category: str, index: int = 0) -> str:
    """Absolute path to one objaverse instance of `category`, from the registry.

    Do not build this path by hand. The obvious pattern
    `objects/objaverse/<category>/<category>_<n>/model.xml` is wrong for at least
    one category we use: `condiment_bottle` declares
    `model_folders=["objaverse/condiment"]` (kitchen_objects.py:458-461), so its
    instances live under a folder that is not its own name. The registry walks the
    real tree, so reading `mjcf_paths` is correct by construction.

    `exclude` is honoured: `bowl` excludes `bowl_21` because you can see through
    its bottom (kitchen_objects.py:214-216). Nothing in this repo honoured it
    before.

    Args:
        category: RoboCasa object category, e.g. `"cup"`, `"bowl"`, `"condiment_bottle"`.
        index: which instance; wraps around modulo the number available.

    Returns:
        Absolute path to a `model.xml`.

    Example:
        >>> path = objaverse_mjcf("cup", index=0)
        >>> path.endswith("model.xml")
        True
    """
    entry = OBJ_CATEGORIES.get(category)
    if entry is None:
        raise KeyError(f"{category!r} is not a RoboCasa object category")
    cat = entry["objaverse"]

    excluded = set(cat.exclude or [])
    paths = [p for p in cat.mjcf_paths if os.path.basename(os.path.dirname(p)) not in excluded]

    if not paths:
        raise FileNotFoundError(
            f"no objaverse MJCF for {category!r}. The registry walks the asset tree "
            f"at import, so an empty list means the RoboCasa dataset is missing or "
            f"partial — run colab.ensure_assets() / `python -m mani_skill.utils."
            f"download_asset -y RoboCasa`. If the assets ARE on disk, this process "
            f"imported the package before they were downloaded and read the tree "
            f"while it was empty — downloading again cannot fix a live process: "
            f"restart the interpreter (on Colab: Runtime → Restart session). This "
            f"cannot be checked on a machine without the 3.7 GB download."
        )
    return paths[index % len(paths)]


# The three body types both loaders below dispatch on, in the order upstream does
# (actors/common.py:28-35). "kinematic" is the one a two-valued dynamic/static
# flag cannot express: immovable, but still poseable under GPU sim.
ACCESSORY_BODY_TYPES = ("dynamic", "kinematic", "static")


def load_objaverse_actor(
    env,
    category: str,
    name: str,
    initial_pose: sapien.Pose,
    index: int = 0,
    dynamic: bool = True,
    body_type: str | None = None,
):
    """Load one RoboCasa object at the scale RoboCasa intends it to have.

    Two things nothing else in this repo does:

    - **`loader.scale` is applied.** `ObjCat.scale` is 2.0 for `bowl` and 1.05 for
      `condiment_bottle` (kitchen_objects.py:203, :459). `template_task.py:140-144`
      and both inherited tasks call `loader.parse` without it, so every object in
      this package is currently loaded at native MJCF size — the bowl at half the
      intended size.
    - **the path comes from the registry**, see `objaverse_mjcf`.

    The pose is set through `builder.initial_pose`, never `set_pose`: `_load_scene`
    is followed by `scene._setup()`, which re-applies `initial_pose` to every
    non-static actor and discards anything set with `set_pose`.

    Args:
        env: the task (`self` inside `_load_scene`); needs `env.scene`.
        category: RoboCasa object category (see `objaverse_mjcf`).
        name: actor name, unique within the scene.
        initial_pose: `sapien.Pose` applied by `scene._setup()`.
        index: which instance of the category.
        dynamic: `True` for a movable body, `False` for a static one. Ignored when
            `body_type` is given.
        body_type: `"dynamic"`, `"kinematic"` or `"static"`, overriding `dynamic`.
            The third option exists because `dynamic` is a two-valued flag and an
            object that is *immovable but re-posed* is neither of the two: a static
            body works at `num_envs=1` on CPU and raises the first time anyone runs a
            batch, because `Actor.pose`'s setter asserts a non-static body under GPU
            sim (`structs/actor.py:344-347`). `load_accessory_actor` takes the same
            argument for the same reason; `MikasaWaterPlants-v0`'s bucket is the
            objaverse-side case.

    Returns:
        The built `Actor`.

    Example:
        >>> self.cup = load_objaverse_actor(
        ...     self, "cup", "cup", sapien.Pose(p=[1.9, -0.49, 1.0]), dynamic=True)
        >>> self.bucket = load_objaverse_actor(
        ...     self, "bowl", "water_bucket", sapien.Pose(p=[2.79, -1.76, 0.02]),
        ...     body_type="kinematic")
    """
    path = objaverse_mjcf(category, index)
    scale = float(getattr(OBJ_CATEGORIES[category]["objaverse"], "scale", 1.0) or 1.0)

    if body_type is None:
        body_type = "dynamic" if dynamic else "static"
    if body_type not in ACCESSORY_BODY_TYPES:
        raise ValueError(
            f"body_type must be one of {ACCESSORY_BODY_TYPES}, got {body_type!r}"
        )

    loader = env.scene.create_mjcf_loader()
    loader.visual_groups = [1]
    loader.scale = scale

    builder = loader.parse(path, package_dir=os.path.dirname(path))["actor_builders"][0]
    builder.initial_pose = initial_pose
    # Explicit, the way upstream dispatches the same three (actors/common.py:28-35).
    if body_type == "dynamic":
        return builder.build_dynamic(name=name)
    if body_type == "kinematic":
        return builder.build_kinematic(name=name)
    return builder.build_static(name=name)



def accessory_mjcf(rel_xml: str) -> str:
    """Absolute path to one RoboCasa *fixture accessory* MJCF, from the asset root.

    `objaverse_mjcf`'s counterpart for the other tree. Objects live under
    `objects/objaverse/<category>/...` and are indexed by `OBJ_CATEGORIES`;
    accessories live under `fixtures/accessories/...` and are not in that registry
    at all, which is why `objaverse_mjcf` cannot reach them — see
    `load_accessory_actor` for the case that forced this.

    `rel_xml` is relative to the RoboCasa **asset root**, in the same form the
    fixture registry yamls store it — `"fixtures/accessories/plants/aloe"`, a
    *directory*. A path that already names the `model.xml` works too. Both shapes
    are accepted because upstream accepts both at exactly this spot
    (`mujoco_object.py:46-49` tries `<rel>/model.xml` and falls back to `<rel>`).

    The root is not pasted together from `$HOME`: it is `ROBOCASA_ASSET_DIR`, the
    same constant `objaverse_mjcf`'s paths are built from, so it follows
    `$MANISKILL_ASSET_DIR` wherever the dataset was put.

    Args:
        rel_xml: path under the RoboCasa asset root — a directory holding a
            `model.xml`, or the `model.xml` itself.

    Returns:
        Absolute path to a `model.xml`.

    Example:
        >>> path = accessory_mjcf("fixtures/accessories/plants/spikey_in_white_pot")
        >>> path.endswith("model.xml")
        True
    """
    candidate = ROBOCASA_ASSET_DIR / rel_xml
    path = candidate / "model.xml" if candidate.is_dir() else candidate
    if not path.is_file():
        raise FileNotFoundError(
            f"no RoboCasa accessory MJCF at {str(path)!r} (asked for {rel_xml!r} "
            f"under {str(ROBOCASA_ASSET_DIR)!r}). Accessories ship with the 3.7 GB "
            f"RoboCasa dataset — run colab.ensure_assets() / `python -m "
            f"mani_skill.utils.download_asset -y RoboCasa`. If the dataset IS on "
            f"disk, check the spelling against the folder, not against the fixture "
            f"registry: plant.yaml keys the fourth plant 'dodakaeder' and points at "
            f"'.../plants/dodakaeder', while the folder is 'dodekaeder'. This cannot "
            f"be checked on a machine without the download."
        )
    return str(path)


def load_accessory_actor(
    env,
    rel_xml: str,
    name: str,
    initial_pose: sapien.Pose,
    *,
    scale: float = 0.3,
    body_type: str = "kinematic",
):
    """Load one RoboCasa fixture accessory as a plain actor.

    The small brother of `load_objaverse_actor`: same MJCF machinery
    (`create_mjcf_loader`, and `visual_groups = [1]` because RoboCasa puts visuals in
    `group=1` and collisions in `group=0`, `mujoco_object.py:41-44`), but a
    different place on disk and a different registry — and the registry is the whole
    reason this exists. `load_objaverse_actor` walks `OBJ_CATEGORIES`, the *object*
    registry, and there is no `plant` object category. The plants are accessories:
    `fixtures/accessories/plants/{aloe,senecio,spikey_in_white_pot,dodekaeder}`.

    **Data trap.** `assets/fixtures/fixture_registry/plant.yaml` keys the fourth
    plant `dodakaeder` and its `xml:` points at `.../plants/dodakaeder`, but the
    folder on disk is `dodekaeder`. Reaching for that variant by its *registry*
    name is a `FileNotFoundError`; pass the disk spelling. The other three keys
    match their folders.

    **Why `body_type` and not `load_objaverse_actor`'s `dynamic=` flag.** A
    dynamic/static pair is not enough for an accessory whose pose is drawn per
    episode. Such an actor must be re-posed in `_initialize_episode`, and
    `Actor.pose`'s setter asserts a non-static body under GPU sim
    (`structs/actor.py:344-347`): a static body works at `num_envs=1` on CPU and
    raises the first time anyone runs a batch. Kinematic is poseable, immovable, and
    unmoved by contact — what a decoration that *carries a position* wants;
    `burner.py:638-645` builds its marker kinematic for the same reason and says so.
    `"static"` stays available for scenery nothing ever re-poses.

    The pose goes through `builder.initial_pose`, never `set_pose`, for the reason
    `load_objaverse_actor` gives: `scene._setup()` re-applies `initial_pose` to every
    non-static actor after `_load_scene` and discards anything set with `set_pose`.

    Args:
        env: the task (`self` inside `_load_scene`); needs `env.scene`.
        rel_xml: path under the RoboCasa asset root — see `accessory_mjcf`.
        name: actor name, unique within the scene.
        initial_pose: `sapien.Pose` applied by `scene._setup()`.
        scale: `loader.scale`. `0.3` is what `plant.yaml` declares for the plants
            (`spikey_in_white_pot` then measures roughly 0.21 x 0.21 x 0.30 m, with
            a single box collision shape).
        body_type: `"kinematic"` (default), `"dynamic"` or `"static"`.

    Returns:
        The built `Actor`.

    Example:
        >>> self.plant = load_accessory_actor(
        ...     self, "fixtures/accessories/plants/spikey_in_white_pot", "plant_0",
        ...     sapien.Pose(p=[2.79, -1.76, 0.95]), scale=0.3, body_type="kinematic")
    """
    if body_type not in ACCESSORY_BODY_TYPES:
        raise ValueError(
            f"body_type must be one of {ACCESSORY_BODY_TYPES}, got {body_type!r}"
        )
    path = accessory_mjcf(rel_xml)

    loader = env.scene.create_mjcf_loader()
    loader.visual_groups = [1]
    loader.scale = float(scale)

    builder = loader.parse(path, package_dir=os.path.dirname(path))["actor_builders"][0]
    builder.initial_pose = initial_pose

    # Explicit, the way upstream dispatches the same three (actors/common.py:28-35).
    if body_type == "dynamic":
        return builder.build_dynamic(name=name)
    if body_type == "kinematic":
        return builder.build_kinematic(name=name)
    return builder.build_static(name=name)


def restore_task_tensor(current, value, device):
    """Restore one task buffer from a state dict that may not hold torch tensors.

    `set_state_dict` is fed from two different places and only one of them was ever
    exercised. In-process — `env.set_state_dict(env.get_state_dict())`, which is what
    the offline tests and the paired memory-free control do — every value is a torch
    tensor of the right shape, and `value.clone().to(device)` is correct. Restoring a
    **recorded** trajectory is the other place: `RecordEpisode` writes the state dict
    to HDF5 (`record.py:153-185`) and `replay_trajectory` reads it back and slices one
    timestep out of it (`trajectory_utils.dict_to_list_of_dicts` / `index_dict`), so
    what arrives is numpy — and a `(1,)` batch that has been indexed comes back as a
    bare `numpy.int64`, which has no `.clone()` and no batch dimension. Replaying our
    own demos died on exactly that (2026-08-18), which also means nothing could have
    replayed a recorded episode of these tasks before.

    Widening rather than converting at the call sites keeps the in-process path
    byte-identical: `torch.as_tensor` on a tensor of the right dtype and device is the
    same tensor, the shape check is a no-op, and the `.clone()` that prevents aliasing
    the caller's buffer still happens.

    Args:
        current: the attribute being replaced — its shape and dtype are the template.
        value: the stored value: a torch tensor, a numpy array, or a numpy/python scalar.
        device: the env's device.

    Returns:
        A tensor with `current`'s shape and dtype, on `device`, sharing storage with
        neither `value` nor `current`.

    Raises:
        ValueError: if `value` cannot be reshaped to `current`'s shape — a real
            mismatch (a trajectory from a differently configured task) rather than the
            missing batch dimension this exists to absorb.

    Example:
        >>> import numpy as np, torch
        >>> cur = torch.zeros(1, dtype=torch.long)
        >>> restore_task_tensor(cur, np.int64(3), torch.device("cpu"))
        tensor([3])
        >>> restore_task_tensor(cur, torch.tensor([7]), torch.device("cpu"))
        tensor([7])
        >>> restore_task_tensor(torch.zeros(2, 3), np.zeros(6), torch.device("cpu")).shape
        torch.Size([2, 3])
    """
    import torch

    if torch.is_tensor(value):
        # np.asarray on a CUDA tensor raises; the tensor is already what we
        # want, minus device/dtype/shape normalization below.
        tensor = value.detach()
    else:
        tensor = torch.as_tensor(np.asarray(value))
    if tensor.shape != current.shape:
        if tensor.numel() != current.numel():
            raise ValueError(
                f"cannot restore a buffer of shape {tuple(current.shape)} from a "
                f"stored value of shape {tuple(tensor.shape)}: the trajectory was "
                "recorded from a differently configured task"
            )
        tensor = tensor.reshape(current.shape)
    return tensor.to(dtype=current.dtype, device=device).clone()


# --- free space on a counter top -------------------------------------------------
#
# Ported from jezv's `my_scenes/base_robocasa.py` (`bba1117`, "Added new functionals
# for placemant", 2026-08-21): the usable area of a counter is its top inset from the
# edges, minus every fixture that occupies the top plane. Kept as free functions taking
# explicit arguments rather than as his mixin's methods, for two reasons — a task here
# already has its counter through `counter_frame`/`require_get_fixture` and should not
# have to inherit a base to place an object, and functions with explicit inputs are
# testable without a GPU or the 3.7 GB asset tree.
#
# One deliberate difference: he reads `scene_data[0]["fixture_cfgs"]` once. Doing that
# here would reintroduce the bug `docs/review-inherited-code.md` §5 names — deriving a
# position from env 0 silently puts objects inside walls in envs 1..N-1 — so
# `counter_blockers` takes one env's `fixture_cfgs` and the caller loops.

COUNTER_EDGE_INSET = 0.08
"""Metres the usable area is inset from the counter's edges (jezv's `EDGE_INSET`)."""

COUNTER_BLOCKER_GAP = 0.03
"""Extra clearance left around a blocking fixture (jezv's `BLOCKER_GAP`)."""

COUNTER_Z_TOL = 0.02
"""Vertical slack for deciding a fixture reaches the top plane (jezv's `Z_TOL`)."""

COUNTER_MIN_OVERLAP = 0.005
"""Least x/y overlap with the counter before a fixture counts as a blocker."""


def subtract_rect(regions, blocker) -> list[np.ndarray]:
    """Subtract one axis-aligned rect from a list of them; returns up to 4 per input.

    Rects are `[x0, x1, y0, y1]`. A rect the blocker misses passes through untouched;
    one it covers entirely disappears.

    Args:
        regions: iterable of `[x0, x1, y0, y1]`.
        blocker: `[x0, x1, y0, y1]` to remove.

    Returns:
        The remaining rects, as a list of `np.ndarray`.

    Example:
        A blocker biting the middle out of a strip leaves the two ends:

        >>> out = subtract_rect([np.array([0.0, 3.0, 0.0, 1.0])], [1.0, 2.0, -1.0, 2.0])
        >>> [[round(float(v), 2) for v in r] for r in out]
        [[0.0, 1.0, 0.0, 1.0], [2.0, 3.0, 0.0, 1.0]]

        A blocker that misses changes nothing:

        >>> out = subtract_rect([np.array([0.0, 1.0, 0.0, 1.0])], [5.0, 6.0, 5.0, 6.0])
        >>> [[round(float(v), 2) for v in r] for r in out]
        [[0.0, 1.0, 0.0, 1.0]]

        A blocker that covers everything leaves nothing:

        >>> subtract_rect([np.array([0.0, 1.0, 0.0, 1.0])], [-1.0, 2.0, -1.0, 2.0])
        []
    """
    out: list[np.ndarray] = []
    bx0, bx1, by0, by1 = (float(v) for v in blocker)
    for r in regions:
        x0, x1, y0, y1 = (float(v) for v in r)
        ix0, ix1 = max(x0, bx0), min(x1, bx1)
        iy0, iy1 = max(y0, by0), min(y1, by1)
        if ix0 >= ix1 or iy0 >= iy1:
            out.append(np.asarray([x0, x1, y0, y1], dtype=np.float64))
            continue
        if x0 < ix0:
            out.append(np.asarray([x0, ix0, y0, y1], dtype=np.float64))
        if ix1 < x1:
            out.append(np.asarray([ix1, x1, y0, y1], dtype=np.float64))
        if y0 < iy0:
            out.append(np.asarray([ix0, ix1, y0, iy0], dtype=np.float64))
        if iy1 < y1:
            out.append(np.asarray([ix0, ix1, iy1, y1], dtype=np.float64))
    return out


def counter_blockers(
    fixture_cfgs,
    counter_pos,
    counter_size,
    *,
    gap: float = COUNTER_BLOCKER_GAP,
    z_tol: float = COUNTER_Z_TOL,
    min_overlap: float = COUNTER_MIN_OVERLAP,
) -> list[np.ndarray]:
    """XY rects of the fixtures standing on (or flush with) this counter's top.

    A fixture blocks only if its vertical span reaches the top plane — the sink is
    flush with the surface, accessories stand on it, and a cabinet underneath must not
    count. Other `*_Counter` fixtures are skipped: they are edge-aligned neighbours,
    never obstacles on this top.

    Args:
        fixture_cfgs: one env's `scene_data[i]["fixture_cfgs"]` — dicts with `name` and
            a `model` carrying `pos` and `size`. **One env's**, not env 0's for all of
            them (review §5).
        counter_pos: the counter's centre `(x, y, z)`.
        counter_size: its full extents `(sx, sy, sz)`.
        gap: clearance added around each blocker.
        z_tol: vertical slack when testing whether a fixture reaches the top.
        min_overlap: least x/y overlap with the counter to count at all.

    Returns:
        `[x0, x1, y0, y1]` rects, already widened by `gap` and clipped to the counter.

    Example:
        A sink flush with the top blocks; a cabinet below it does not:

        >>> class M:
        ...     def __init__(self, pos, size): self.pos, self.size = pos, size
        >>> cfgs = [
        ...     {"name": "sink", "model": M((0.5, 0.0, 1.0), (0.4, 0.4, 0.05))},
        ...     {"name": "cab",  "model": M((0.0, 0.0, 0.4), (0.4, 0.4, 0.5))},
        ... ]
        >>> b = counter_blockers(cfgs, (0.0, 0.0, 0.5), (4.0, 1.0, 1.0), gap=0.0)
        >>> len(b), [round(float(v), 2) for v in b[0]]
        (1, [0.3, 0.7, -0.2, 0.2])
    """
    c_pos = np.asarray(counter_pos, dtype=np.float64)
    c_size = np.asarray(counter_size, dtype=np.float64)
    top_z = c_pos[2] + c_size[2] / 2.0
    x0, x1 = c_pos[0] - c_size[0] / 2.0, c_pos[0] + c_size[0] / 2.0
    y0, y1 = c_pos[1] - c_size[1] / 2.0, c_pos[1] + c_size[1] / 2.0

    blockers: list[np.ndarray] = []
    for cfg in fixture_cfgs or []:
        name = cfg.get("name", "") if isinstance(cfg, dict) else ""
        if str(name).endswith("_Counter") or str(name).endswith("counter_main_main_group"):
            continue
        model = cfg.get("model") if isinstance(cfg, dict) else None
        pos = getattr(model, "pos", None)
        size = getattr(model, "size", None)
        if pos is None or size is None:
            continue
        pos = np.asarray(pos, dtype=np.float64)
        size = np.asarray(size, dtype=np.float64)
        if not (pos[2] - size[2] / 2.0 - z_tol <= top_z <= pos[2] + size[2] / 2.0 + z_tol):
            continue
        fx0, fx1 = pos[0] - size[0] / 2.0, pos[0] + size[0] / 2.0
        fy0, fy1 = pos[1] - size[1] / 2.0, pos[1] + size[1] / 2.0
        ox0, ox1 = max(x0, fx0), min(x1, fx1)
        oy0, oy1 = max(y0, fy0), min(y1, fy1)
        if ox1 - ox0 > min_overlap and oy1 - oy0 > min_overlap:
            blockers.append(np.asarray([ox0 - gap, ox1 + gap, oy0 - gap, oy1 + gap]))
    return blockers


def usable_counter_regions(
    fixture_cfgs, counter_pos, counter_size, *, inset: float = COUNTER_EDGE_INSET, **kw
) -> list[np.ndarray]:
    """The counter top inset from its edges, minus everything standing on it.

    Args:
        fixture_cfgs: one env's fixture configs (see `counter_blockers`).
        counter_pos: the counter's centre.
        counter_size: its full extents.
        inset: metres to pull in from each edge.
        **kw: forwarded to `counter_blockers` (`gap`, `z_tol`, `min_overlap`).

    Returns:
        Free `[x0, x1, y0, y1]` rects. Possibly empty — a caller must say so rather
        than sampling from nothing.

    Example:
        >>> class M:
        ...     def __init__(self, pos, size): self.pos, self.size = pos, size
        >>> cfgs = [{"name": "sink", "model": M((0.0, 0.0, 1.0), (1.0, 2.0, 0.05))}]
        >>> r = usable_counter_regions(cfgs, (0.0, 0.0, 0.5), (4.0, 1.0, 1.0),
        ...                            inset=0.0, gap=0.0)
        >>> [[round(float(v), 2) for v in x] for x in r]
        [[-2.0, -0.5, -0.5, 0.5], [0.5, 2.0, -0.5, 0.5]]
    """
    c_pos = np.asarray(counter_pos, dtype=np.float64)
    c_size = np.asarray(counter_size, dtype=np.float64)
    regions = [
        np.asarray([
            c_pos[0] - c_size[0] / 2.0 + inset, c_pos[0] + c_size[0] / 2.0 - inset,
            c_pos[1] - c_size[1] / 2.0 + inset, c_pos[1] + c_size[1] / 2.0 - inset,
        ])
    ]
    for b in counter_blockers(fixture_cfgs, counter_pos, counter_size, **kw):
        regions = subtract_rect(regions, b)
    return regions


def point_in_regions(regions, xy, *, margin: float = 0.0) -> bool:
    """Is `xy` inside the union of `regions`, with `margin` to spare on every side?

    `sample_in_regions` guarantees only the point it returns. A caller that derives
    *other* points from it — two stations either side of a drawn midpoint, say — must
    test those too, or they land wherever the arithmetic puts them: measured on
    season-dish, a midpoint drawn legally at x 0.69 put its stations at 0.57 and 0.81,
    the second of which is on the sink.

    Args:
        regions: `[x0, x1, y0, y1]` rects.
        xy: the point to test; only its first two components are read.
        margin: metres of clearance required inside the rect's edge.

    Returns:
        True if some rect contains the point with that margin.

    Example:
        >>> r = [np.array([0.0, 1.0, 0.0, 1.0])]
        >>> point_in_regions(r, [0.5, 0.5]), point_in_regions(r, [1.5, 0.5])
        (True, False)
        >>> point_in_regions(r, [0.98, 0.5], margin=0.05)
        False
    """
    x, y = float(np.asarray(xy)[0]), float(np.asarray(xy)[1])
    for r in regions:
        r = np.asarray(r, dtype=np.float64)
        if (r[0] + margin <= x <= r[1] - margin) and (r[2] + margin <= y <= r[3] - margin):
            return True
    return False


def sample_in_regions(regions, rng, *, z: float | None = None) -> np.ndarray:
    """Uniform point over the union of `regions`, area-weighted between them.

    Area weighting is what makes it uniform over the *union*: picking a rect uniformly
    would over-sample the small ones.

    Args:
        regions: `[x0, x1, y0, y1]` rects, as from `usable_counter_regions`.
        rng: a `np.random.Generator` or `RandomState`.
        z: if given, returned as the third component.

    Returns:
        `[x, y]`, or `[x, y, z]` when `z` is given.

    Raises:
        ValueError: if `regions` is empty — there is no honest sample to return.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> p = sample_in_regions([np.array([0.0, 1.0, 2.0, 3.0])], rng, z=0.9)
        >>> bool(0.0 <= p[0] <= 1.0), bool(2.0 <= p[1] <= 3.0), round(float(p[2]), 2)
        (True, True, 0.9)
    """
    regions = [np.asarray(r, dtype=np.float64) for r in regions]
    if not regions:
        raise ValueError("no usable region to sample from")
    areas = np.array([(r[1] - r[0]) * (r[3] - r[2]) for r in regions], dtype=np.float64)
    if areas.sum() <= 0:
        raise ValueError("usable regions have zero area")
    r = regions[int(rng.choice(len(areas), p=areas / areas.sum()))]
    xy = np.array([rng.uniform(r[0], r[1]), rng.uniform(r[2], r[3])], dtype=np.float64)
    return xy if z is None else np.array([xy[0], xy[1], float(z)])
