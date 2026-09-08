import os

import numpy as np
import sapien
import sapien.physx as physx
import torch
from transforms3d.quaternions import mat2quat, quat2mat

from mani_skill.utils import common
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.robocasa.objects.kitchen_objects import (
    OBJ_CATEGORIES,
    OBJ_GROUPS,
)
from mani_skill.utils.structs import Actor, Pose

from utils.scene_utils import degree_to_quanterion, get_actor_size

from .my_robocasa_fridge_picture import MyRoboCasaFridgePicture


@register_env("MyRoboCasa_FridgeVeggies-v1", asset_download_ids=["RoboCasa"])
class MyRoboCasaFridgeVeggies(MyRoboCasaFridgePicture):
    """Robot spawns in front of the fridge; the fridge door shows a photo of
    one of the NUM_VEGGIES different vegetables sitting on the counter (the
    target, re-picked every episode). A pot also lies on the counter."""

    NUM_VEGGIES = 3  # three different vegetables (kinds sampled from the pool)

    #: Static check thresholds (evaluate + planner settle). The angular
    #: threshold is deliberately loose: the pot walls contain the vegetable,
    #: and a released vegetable in the pot keeps reporting contact micro-
    #: jitter (av 0.2-0.35) long after its pose freezes (observed: cucumber
    #: av=0.35, pose frozen, failed the old 0.2 threshold and burned the
    #: whole settle budget). 0.5 covers the jitter class; a genuinely moving
    #: vegetable (rolling, launched) reads far above it.
    STATIC_V_MAX = 0.1
    STATIC_AV_MAX = 0.5
    VEG_GAP = 0.03  # min center distance between vegetables (m)
    PLATE_MARGIN = 0.03  # extra min distance from vegetables to the plate (m)
    MAX_SAMPLE_ATTEMPTS = 500

    # Wall cabinets overhang the counter's back (front face at CAB_FRONT_Y), so
    # anything that must be lifted is confined to the front strip; same fix as
    # MyRoboCasaSceneTakeItBack.
    CAB_FRONT_Y = -0.45  # front face of the wall cabinets
    LIFT_CLEAR_GAP = 0.08  # clearance from the cabinet front for the lift
    FRONT_GAP = 0.03  # inset from the counter front edge for the strip

    #: Categories excluded from the pool. Potato is excluded: the ds_fetch
    #: gripper's max opening (~5 cm) cannot wrap a potato's 5-6 cm width, so
    #: the fingertip grasp is marginal and the lift repeatedly fails
    #: ("Grasp invalid (vegetable not lifted)" across many seeds).
    EXCLUDED_CATEGORIES: set[str] = {"potato"}

    #: Specific vegetable INSTANCES excluded from the pool (identified by the
    #: model directory name). carrot_4: the fingertip grasp consistently fails
    #: to RAISE it (Grasp invalid in 3/3 runs - the fingers touch it but the
    #: lift never comes, unlike carrot_0/1/2/3/6/7 which grasp fine).
    EXCLUDED_INSTANCES: set[str] = {"carrot_4"}

    veggies: list[Actor]
    veggie_halves: np.ndarray  # [NUM_VEGGIES, 3] half extents
    veg_quats: list[np.ndarray]  # per-veg [4] wxyz; lying quat or identity
    veggie_pos: np.ndarray  # [N, NUM_VEGGIES, 3] per-env vegetable positions
    plate: Actor
    plate_half: np.ndarray  # [3]
    plate_pos: np.ndarray  # [N, 3]
    plate_quat: np.ndarray  # [4] wxyz; identity or 90-deg (handle along x)
    _picture_actors: list[Actor]  # one fridge picture per vegetable
    _picture_target: int

    # ------------------------------------------------------------------ #
    # Scene build
    # ------------------------------------------------------------------ #

    @staticmethod
    def _vegetable_pool():
        """(category, ObjCat, [(mjcf_path, image0.png)]) for every vegetable
        category that has a photo texture usable as the fridge picture and is
        graspable by the Fetch motion planner."""
        pool = []
        for cat in OBJ_GROUPS["vegetable"]:
            if "objaverse" not in OBJ_CATEGORIES[cat]:
                continue
            if cat in MyRoboCasaFridgeVeggies.EXCLUDED_CATEGORIES:
                continue
            objcat = OBJ_CATEGORIES[cat]["objaverse"]
            pairs = [
                (p, os.path.join(os.path.dirname(p), "visual", "image0.png"))
                for p in objcat.mjcf_paths
                if os.path.exists(
                    os.path.join(os.path.dirname(p), "visual", "image0.png")
                )
                and os.path.basename(os.path.dirname(p))
                not in MyRoboCasaFridgeVeggies.EXCLUDED_INSTANCES
            ]
            if pairs:
                pool.append((cat, objcat, pairs))
        return pool

    def _counter_top(self) -> float:
        return self.counter_pos[2] + self.counter_size[2] / 2

    def _load_scene(self, options: dict):
        super()._load_scene(
            options
        )  # fridge, robot spawn, picture_center; PICTURE_PATH None -> no picture
        rng = self._main_rng
        pool = self._vegetable_pool()
        chosen = [
            pool[i] for i in rng.choice(len(pool), self.NUM_VEGGIES, replace=False)
        ]

        loader = self.scene.create_mjcf_loader()
        loader.visual_groups = [1]
        top = self._counter_top()
        spawn = [self.counter_pos[0], self.counter_pos[1], top + 1.0]

        self.veggies, self.veggie_halves, textures = [], [], []
        self.veg_quats = []
        for cat, objcat, pairs in chosen:
            path, image = pairs[rng.randint(len(pairs))]
            loader.scale = objcat.get_mjcf_kwargs()["scale"]
            if cat in (
                "cucumber", "squash", "eggplant", "sweet_potato", "onion",
                "corn",
            ):
                # these categories have instances at or above the ds_fetch
                # gripper's max opening (~5 cm), so the fingertip grasp is
                # marginal and the lift fails ("Grasping failed entirely" on
                # squash/eggplant/sweet_potato instances up to 9.8 cm wide;
                # the corn cob's 4.8 cm thickness squeezes at the max opening
                # and slips out during the lift - "vegetable did not rise");
                # reduce them ~20% so the fingers can wrap them
                loader.scale *= 0.8
            builder = loader.parse(path, package_dir=os.path.dirname(path))[
                "actor_builders"
            ][0]
            builder.initial_pose = sapien.Pose(p=spawn)
            veg = builder.build_dynamic(
                name=f"veg_{os.path.basename(os.path.dirname(path))}"
            )
            self.veggies.append(veg)
            self.veggie_halves.append(get_actor_size(veg) / 2)
            # Elongated vegetables SPAWN STANDING ON END (their long axis is
            # the model's Z - e.g. the carrot's half is [0.016, 0.015,
            # 0.069]): when released they fall over and ROLL 5-8 cm north,
            # often under the cabinet overhang where the arm cannot reach
            # them (observed: cucumber rolled to y=-0.41, grasp IK failed on
            # every stance). Lay them down - rotate the long axis along the
            # counter line (x) - and re-measure the halves so the placement
            # z uses the lying thickness.
            h = self.veggie_halves[-1]
            if h[2] > h[0] and h[2] > h[1]:
                self.veg_quats.append(degree_to_quanterion(y=90))
                veg.set_pose(
                    sapien.Pose(p=spawn, q=self.veg_quats[-1])
                )
                self.veggie_halves[-1] = get_actor_size(veg) / 2
            elif h[1] > h[0] and h[1] > h[2]:
                # Long axis along the model's Y (e.g. corn): it lies with the
                # long axis along the counter DEPTH. The strip sampling pins
                # the center at r[2] + half_y, so the half_y margin shoves the
                # center NORTH to the wall-cabinet line (y=-0.44) and the
                # reach/grasp becomes infeasible (forearm vs the base cabinet,
                # seeds 32/47). Rotate 90 deg about z: the long axis runs
                # along the counter line (x) and the center stays south of the
                # cabinet face (same fix as the pot handle).
                self.veg_quats.append(degree_to_quanterion(z=90))
                veg.set_pose(
                    sapien.Pose(p=spawn, q=self.veg_quats[-1])
                )
                self.veggie_halves[-1] = get_actor_size(veg) / 2
            else:
                self.veg_quats.append(np.array([1.0, 0.0, 0.0, 0.0]))
            textures.append(image)

        # the "plate" is a POT (the pot category - a frying-pan-style pan with
        # walls): round vegetables bounced off the flat plate during the
        # settle and rolled away (lemon 0.6 m, onion off the plate); the
        # pot's walls keep the released vegetable inside. The objaverse pot's
        # collision is a convex-decomposed SHELL that works as a container
        # (verified: a vegetable dropped into it rests inside, ~1 cm above
        # the pot origin), unlike the plate's broken 33-cm block, so the
        # collision is kept as-is.
        pl_objcat = OBJ_CATEGORIES["pot"]["objaverse"]
        # pan_22 is EXCLUDED: its bowl has an internal handle/structure
        # crossing the interior at ~2/3 of the pot's depth (measured: 37
        # collision vertices inside the bowl at world z 0.997-1.040). Large
        # vegetables cannot descend past it - they rest on it and the release
        # launches them out (observed: lemon stuck at z=1.048, flung 18 cm
        # off the pot). The other two pan models have a clear bowl.
        pot_paths = [p for p in pl_objcat.mjcf_paths if "pan_22" not in p]
        pl_path = pot_paths[rng.randint(len(pot_paths))]
        loader.scale = pl_objcat.get_mjcf_kwargs()["scale"]
        plb = loader.parse(pl_path, package_dir=os.path.dirname(pl_path))[
            "actor_builders"
        ][0]
        plb.initial_pose = sapien.Pose(p=spawn)
        # The pot is KINEMATIC (static): it has no movement physics - the
        # released vegetable's impact must not shove it around (the light
        # dynamic pot slid 0.6 m under the lemon's drop). The walls keep the
        # vegetable inside; it does not need to be pushed by anything.
        self.plate = plb.build_kinematic(name="plate")
        self.plate_half = get_actor_size(self.plate) / 2
        # Pot handles: the mjcf loader recenters the actor to its AABB
        # center, so the BOWL is offset from the actor origin toward the
        # handle (measured 6.7-8.6 cm = 70-91% of the rim radius in this
        # pot family). A handle pointing across the counter depth (south)
        # puts the bowl under the wall-cabinet overhang, where the arm
        # cannot descend at all (the bowl center lands at the cabinet face).
        # Rotate such pots 90 deg so the handle runs along the counter line:
        # the bowl offset then lies along x (unconstrained for the base
        # drive) and the bowl stays clear of the cabinets. The planner aims
        # the transport/descent at the measured bowl center.
        if self.plate_half[1] > self.plate_half[0]:
            self.plate_quat = degree_to_quanterion(z=90)
            self.plate.set_pose(sapien.Pose(p=spawn, q=self.plate_quat))
            self.plate_half = get_actor_size(self.plate) / 2
        else:
            self.plate_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # one picture per vegetable, all on the fridge door; hidden until chosen
        self._picture_actors = [
            self._build_picture(t, f"fridge_picture_{i}")
            for i, t in enumerate(textures)
        ]
        for p in self._picture_actors:
            p.hide_visual()
        self.picture = None
        self._picture_target = 0

    def _build_picture(self, texture_path: str, name: str) -> Actor:
        """Fridge-door picture showing a photo texture (same as the parent's
        picture, but parameterized by texture file)."""
        fridge_key = next(k for k in self.fixture_placements if k.endswith("_Fridge"))
        quat = np.asarray(self.fixture_placements[fridge_key]["quat"])
        front = quat2mat(quat) @ np.array([0.0, -1.0, 0.0])  # door direction

        aabbs = np.array([b.get_global_aabb_fast() for b in self.fridge._bodies])
        lower, upper = aabbs[:, 0].min(axis=0), aabbs[:, 1].max(axis=0)
        center = (lower + upper) / 2
        half_depth = ((upper - lower) * np.abs(front)).sum() / 2
        door_center = center + front * half_depth

        w = self.PICTURE_WIDTH
        h = w  # vegetable textures are square
        face_w = (
            (upper - lower) * np.abs(quat2mat(quat) @ np.array([1.0, 0.0, 0.0]))
        ).sum()
        face_h = upper[2] - lower[2]
        scale = min(1.0, 0.9 * face_w / w, 0.85 * face_h / h)
        w, h = w * scale, h * scale

        mat = sapien.render.RenderMaterial()
        mat.base_color_texture = sapien.render.RenderTexture2D(
            filename=texture_path, mipmap_levels=1
        )
        width_dir = np.cross(np.array([0.0, 0.0, 1.0]), front)
        rot = np.column_stack([front, width_dir, np.array([0.0, 0.0, 1.0])])
        pose = sapien.Pose(p=door_center + front * 0.005, q=mat2quat(rot))

        builder = self.scene.create_actor_builder()
        builder.add_plane_visual(scale=[1.0, w / 2, h / 2], material=mat)
        builder.initial_pose = pose
        pic = builder.build_dynamic(name=name)
        for obj in pic._objs:
            obj.find_component_by_type(
                physx.PhysxRigidDynamicComponent
            ).set_disable_gravity(True)
        return pic

    # ------------------------------------------------------------------ #
    # Episode setup
    # ------------------------------------------------------------------ #

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # re-pick the target vegetable and show only its fridge picture
        self._picture_target = int(self._main_rng.randint(self.NUM_VEGGIES))
        for i, p in enumerate(self._picture_actors):
            if i == self._picture_target:
                p.show_visual()
            else:
                p.hide_visual()
        self.picture = self._picture_actors[self._picture_target]
        super()._initialize_episode(
            env_idx, options
        )  # robot pose + picture steps reset

        # place vegetables and plate at random separated spots on the counter's
        # FRONT STRIP (clear of the wall-cabinet overhang, so the arm can lift)
        top = self._counter_top()
        cy0 = self.counter_pos[1] - self.counter_size[1] / 2
        y0 = cy0 + self.FRONT_GAP
        y1 = self.CAB_FRONT_Y - self.LIFT_CLEAR_GAP
        strip = [
            np.array([r[0], r[1], max(r[2], y0), min(r[3], y1)])
            for r in self.usable_regions
        ]
        strip = [r for r in strip if r[3] - r[2] > 0 and r[1] - r[0] >= 0.08]
        halves = [h[:2] + self.VEG_GAP for h in self.veggie_halves] + [
            self.plate_half[:2] + self.PLATE_MARGIN
        ]
        idxs = common.to_numpy(env_idx)
        batched = self._batched_episode_rng
        veg_pos, plate_pos = [], []
        for i in range(len(idxs)):
            rng = batched[idxs[i]] if batched is not None else self._main_rng
            spots = self._sample_separated_many(rng, halves, rects=strip)
            veg_pos.append(
                [
                    # place the vegetable ALMOST resting (5 mm drop): the
                    # old +0.02 drop let round vegetables roll 5-8 cm north
                    # on impact - often under the cabinet overhang, where
                    # the arm cannot reach them
                    [s[0], s[1], top + self.veggie_halves[j][2] + 0.005]
                    for j, s in enumerate(spots[: self.NUM_VEGGIES])
                ]
            )
            plate_pos.append(
                [spots[-1][0], spots[-1][1], top + self.plate_half[2] + 0.01]
            )

        self.veggie_pos = np.asarray(veg_pos)
        self.plate_pos = np.asarray(plate_pos)
        z3 = torch.zeros(len(idxs), 3, device=self.device)
        for j, veg in enumerate(self.veggies):
            # the placement must reuse the load-time rotation (the lying
            # quat for standing vegetables): an identity quat here would
            # STAND the vegetable back up while veggie_halves (measured
            # with the rotation) place it too low - the collision then
            # penetrates the counter and the solver pops it away
            # (observed: carrot launched with -7.6 m/s)
            veg.set_pose(
                Pose.create_from_pq(p=self.veggie_pos[:, j], q=self.veg_quats[j])
            )
            # the vegetable spent the reconfigure FALLING from the spawn
            # (~1 m above the counter) and set_pose teleports it WITHOUT
            # clearing the velocity - the accumulated drop momentum then
            # rolls it away (observed: carrot rolled 9 m off the counter).
            # Zero the velocities so the vegetable rests where it is placed.
            veg.set_linear_velocity(z3)
            veg.set_angular_velocity(z3)
        self.plate.set_pose(
            Pose.create_from_pq(p=self.plate_pos, q=self.plate_quat)
        )

    # ------------------------------------------------------------------ #
    # Task logic
    # ------------------------------------------------------------------ #

    def evaluate(self):
        """Success: the pictured (target) vegetable rests on the plate, not
        grasped, and nearly static."""
        target = self.veggies[self._picture_target]
        target_pos = target.pose.p
        is_grasped = self.agent.is_grasping(target)

        v = torch.linalg.norm(target.linear_velocity, dim=1)
        av = torch.linalg.norm(target.angular_velocity, dim=1)
        is_static = (v <= self.STATIC_V_MAX) & (av <= self.STATIC_AV_MAX)

        plate_pos = self.plate.pose.p
        xy_dist = torch.linalg.norm(target_pos[..., :2] - plate_pos[..., :2], dim=1)
        z_diff = target_pos[..., 2] - plate_pos[..., 2]

        # The "plate" is a POT (a container): the vegetable counts as placed
        # when it is INSIDE it.
        # - The containment radius is the pot's inner wall (~0.15 m, the pan
        #   without the handle), NOT the AABB's x-half (0.09 - the pan's
        #   width, which rejected vegetables that were well inside).
        # - The vegetable rests on the pot's BOTTOM, ~pot_half_z below the
        #   pot's origin (the pot is kinematic and keeps its spawn height),
        #   so allow z_diff down to the bottom; the old -0.02 rejected the
        #   resting vegetable by 1-3 cm.
        plate_radius = 0.15
        on_plate = (xy_dist <= plate_radius) & (z_diff >= -0.06) & (z_diff <= 0.10)
        success = on_plate & (~is_grasped) & is_static
        return dict(
            success=success,
            dbg_grasped=bool(is_grasped[0]),
            dbg_static=bool(is_static[0]),
            dbg_on_plate=bool(on_plate[0]),
            dbg_xy=float(xy_dist[0]),
            dbg_zd=float(z_diff[0]),
            dbg_v=float(v[0]),
            dbg_av=float(av[0]),
        )

    def _sample_separated_many(self, rng, halves, rects=None) -> list[np.ndarray]:
        """One random [x, y] spot per object in the given rects (default: the
        usable counter regions).

        The front strip is thin (h ~0.04 m), so y is pinned to the rect's strip
        center (proven clear of the counter edge and cabinet overhang, same as
        MyRoboCasaSceneTakeItBack._sample_cup) and only x is sampled. Each
        object gets its own rect when one fits; objects sharing a rect are
        retried with pairwise separation by their inflated radii."""
        if rects is None:
            rects = self.usable_regions
        radii = [float(np.linalg.norm(h[:2])) for h in halves]
        order = sorted(range(len(halves)), key=lambda k: -radii[k])
        placed: list[tuple[int, int, np.ndarray]] = []  # (rect idx, object idx, xy)
        for k in order:
            h = halves[k]
            fit = [i for i, r in enumerate(rects) if r[1] - r[0] >= 2 * h[0]]
            if not fit:
                raise RuntimeError(f"Object {k} does not fit any counter region")
            free = [i for i in fit if all(ri != i for ri, _, _ in placed)]
            cand = free if free else fit
            spot = None
            for _ in range(self.MAX_SAMPLE_ATTEMPTS):
                i = cand[rng.randint(len(cand))]
                r = rects[i]
                # y: the round vegetables ROLL north ~5-8 cm when they drop
                # onto the counter, and the strip's north edge is only
                # LIFT_CLEAR_GAP from the cabinet overhang - a vegetable
                # sampled in the north half ends up UNDER the cabinet, where
                # the arm cannot reach it (observed: cucumber rolled to
                # y=-0.41 under the overhang, grasp IK failed on every
                # stance). Sample the vegetables in the SOUTH part of the
                # strip. The pot is kinematic (does not roll), but its BOWL
                # must also stay south of the cabinet clear line; its AABB
                # (with the long handle) is taller than the strip, so it
                # overhangs the north no matter what - pin the ORIGIN (=
                # the bowl center for the rotated pots) to the strip's south
                # band, unconstrained by the AABB margin.
                is_pot = k == len(halves) - 1
                y_lo = r[2] if is_pot else r[2] + h[1]
                y_hi = r[2] + (r[3] - r[2]) * 0.4
                c = np.array([
                    rng.uniform(r[0] + h[0], r[1] - h[0]),
                    rng.uniform(y_lo, max(y_hi, y_lo + 0.005)),
                ])
                if all(
                    np.linalg.norm(c - s) >= radii[k] + radii[j]
                    for ri, j, s in placed
                    if ri == i
                ):
                    spot = (i, c)
                    break
            if spot is None:
                raise RuntimeError("Could not place all objects on the counter")
            placed.append((spot[0], k, spot[1]))
        return [s for _, _, s in sorted(placed, key=lambda p: p[1])]
