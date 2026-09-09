import os
import tempfile

import numpy as np
import sapien
import torch
import trimesh
from mani_skill import ASSET_DIR
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Actor, Pose
from .base_robocasa import BaseRoboCasaSimple
from utils.scene_utils import get_actor_size


@register_env("MyRoboCasa_TakeItBackTray-v1", asset_download_ids=["RoboCasa"])
class MyRoboCasaSceneTakeItBackTray(BaseRoboCasaSimple):
    """Place a randomly positioned cup onto a randomly positioned baking tray."""

    # baking tray asset (repo assets/), recentered + scaled at load time
    TRAY_PATH = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "baking_tray.glb")
    )
    TRAY_SCALE = 0.1  # raw glb is 2.82 m -> ~0.28 m real-world tray

    # counter zones: the wall cabinets overhang the counter's back (y > -0.45),
    # so the cup (which must be lifted ~0.15 m) is confined to the front strip;
    # the flat tray is fine under the overhang
    CAB_FRONT_Y = -0.45  # front face of the wall cabinets
    CUP_CLEAR_GAP = 0.08  # cup clearance from the cabinet front for the lift
    CUP_FRONT_GAP = 0.03  # cup clearance from the counter front face

    # tray band (world y): the held cup rides ~0.65 m north of the base (base
    # line y <= -1.0, arm ~perpendicular to the counter), so it reaches
    # y = base_y + R in [-0.50, -0.20]. The south edge clears the cup strip
    # (-0.53) by 3 cm; the north edge needs R = 0.80 (grasp config). The band
    # is exactly one tray tall (0.282 m), so the tray's y is pinned to its
    # center like the cup's.
    TRAY_BAND_Y = (-0.50, -0.20)

    fxtr_placements: dict[str, dict[str, object]]
    cup_pos: np.ndarray  # [N, 3] initial cup position per env
    camera_pos: np.ndarray
    agent_pose: sapien.Pose

    cup: Actor
    tray: Actor
    tray_half: np.ndarray  # [3] half extents of the scaled tray
    cup_half: np.ndarray  # [3] half extents of the cup
    cup_zone: np.ndarray  # counter rects reserved for the cup
    tray_zone: np.ndarray  # counter rects reserved for the tray

    def _load_scene(self, options: dict):
        super()._load_scene(options)

        # --- cup (same objaverse asset as before) ---
        cup_path = os.path.join(
            ASSET_DIR,
            "scene_datasets/robocasa_dataset/assets/objects/objaverse/cup/cup_2/model.xml",
        )
        loader = self.scene.create_mjcf_loader()
        loader.visual_groups = [1]
        builder = loader.parse(str(cup_path), package_dir=os.path.dirname(cup_path))[
            "actor_builders"
        ][0]
        spawn = self.counter_pos.copy()
        spawn[2] = self._counter_top() + 1.0  # above the counter; episodes re-place it
        builder.initial_pose = sapien.Pose(p=spawn)
        self.cup = builder.build_dynamic(name="cup")
        self.cup_half = get_actor_size(self.cup) / 2

        # --- baking tray (visual + box collision matching its size) ---
        glb_path, self.tray_half = self._prepare_tray()
        tb = self.scene.create_actor_builder()
        tb.add_visual_from_file(str(glb_path))
        tb.add_box_collision(half_size=self.tray_half.tolist())
        tb.initial_pose = sapien.Pose(p=spawn)
        self.tray = tb.build_dynamic(name="baking_tray")

        self.cup_zone, self.tray_zone = self._split_zones()

        self.camera_pos = self.counter_pos.copy()
        self.camera_pos[0] -= self.counter_size[0] / 4
        self.camera_pos[1] -= self.counter_size[1] / 2
        self.camera_pos[2] += self.counter_size[2] / 2

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)  # random free robot spawn
        with torch.device(self.device):
            if not hasattr(self, "placed_on_tray") or self.placed_on_tray.shape[0] != self.num_envs:
                self.placed_on_tray = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.placed_on_tray[env_idx] = False

        # free placement: cup on the counter's front strip, tray in the rest,
        # re-sampled every episode (zones are disjoint -> no initial overlap).
        # NOTE: use _main_rng (per-seed), not _batched_episode_rng: reconfigure
        # resets the episode rng to FIXTURE_SEED, so it never varies by seed.
        top = self._counter_top()
        cup_xy_half = self.cup_half[:2] + 0.005
        tray_xy_half = self.tray_half[:2] + 0.02
        cup_pos, tray_pos = [], []
        for _ in range(len(env_idx)):
            c = self._sample_cup(self._main_rng, cup_xy_half)
            t = self._sample_tray(self._main_rng, tray_xy_half)
            if c is None or t is None:
                c, t = self._sample_separated(self._main_rng, cup_xy_half, tray_xy_half)
            cup_pos.append([c[0], c[1], top + self.cup_half[2] + 0.02])
            tray_pos.append([t[0], t[1], top + self.tray_half[2] + 0.01])

        self.cup_pos = np.asarray(cup_pos)
        self.cup.set_pose(Pose.create_from_pq(p=self.cup_pos))
        self.tray.set_pose(Pose.create_from_pq(p=np.asarray(tray_pos)))

    # ------------------------------------------------------------------ #
    # Tray asset
    # ------------------------------------------------------------------ #

    def _prepare_tray(self):
        """Return (cached glb path, half extents) for the tray: recentered,
        scaled by TRAY_SCALE, and rotated so the thin axis is vertical (z)."""
        key = (
            f"baking_tray_{os.path.getsize(self.TRAY_PATH)}_{int(os.path.getmtime(self.TRAY_PATH))}"
            f"_{self.TRAY_SCALE}.glb"
        )
        cache = os.path.join(tempfile.gettempdir(), "maniskill_" + key)
        if not os.path.exists(cache):
            mesh = trimesh.load(self.TRAY_PATH, force="mesh")
            verts = (mesh.vertices - mesh.bounds.mean(axis=0)) * self.TRAY_SCALE
            # raw glb: thin axis along y, face in xz -> rotate to flat in xy
            verts = np.stack([verts[:, 0], -verts[:, 2], verts[:, 1]], axis=1)
            out = trimesh.Trimesh(vertices=verts, faces=mesh.faces, process=False)
            out.visual = mesh.visual
            out.export(cache)
        half = trimesh.load(cache, force="mesh").extents / 2
        return cache, half

    # ------------------------------------------------------------------ #
    # Counter zones for cup and tray
    # ------------------------------------------------------------------ #

    def _counter_top(self) -> float:
        return self.counter_pos[2] + self.counter_size[2] / 2

    def _split_zones(self):
        """Split the usable counter area into a cup zone (front strip, clear of
        the wall-cabinet overhang so the cup can be lifted) and a tray zone (the
        reachable band north of the cup strip, TRAY_BAND_Y)."""
        cy0 = self.counter_pos[1] - self.counter_size[1] / 2
        cup_min = cy0 + self.CUP_FRONT_GAP
        cup_max = self.CAB_FRONT_Y - self.CUP_CLEAR_GAP
        tray_min, tray_max = self.TRAY_BAND_Y
        cup_zone, tray_zone = [], []
        # drop the west pocket (x < ~0.8, between the room wall and the sink):
        # a base facing west there drags its 1.06 m home-pose arm into the left
        # wall (x=0), so the approach staging point and long west transports
        # are geometrically unreachable for this robot
        regions = [r for r in self.usable_regions if r[1] >= 1.20]
        for r in regions:
            cup = list(r)
            cup[2], cup[3] = max(r[2], cup_min), min(r[3], cup_max)
            if cup[3] - cup[2] >= 0.03 and cup[1] - cup[0] >= 0.1:  # strip exists
                cup_zone.append(cup)
            tray = list(r)
            tray[2], tray[3] = max(r[2], tray_min), min(r[3], tray_max)
            if tray[3] - tray[2] >= 0.30:  # fits the tray + margins
                tray_zone.append(tray)
        return tuple(np.asarray(z) if z else None for z in (cup_zone, tray_zone))

    def _sample_cup(self, rng, half):
        """Sample x in the cup zone; y is pinned to the strip center (the front
        strip is too thin for free y sampling, and its center is proven clear
        of both the counter front and the cabinet overhang)."""
        if self.cup_zone is None:
            return None
        ok = [r for r in self.cup_zone if r[1] - r[0] >= 2 * half[0]]
        if not ok:
            return None
        areas = np.array([r[1] - r[0] for r in ok])
        r = ok[rng.choice(len(ok), p=areas / areas.sum())]
        return np.array(
            [rng.uniform(r[0] + half[0], r[1] - half[0]), (r[2] + r[3]) / 2]
        )

    @staticmethod
    def _sample_in_rects(rects, rng, half):
        """Uniform [x, y] in a random rect (area-weighted) that fits half extents."""
        if rects is None:
            return None
        ok = [
            r
            for r in rects
            if r[1] - r[0] >= 2 * half[0] and r[3] - r[2] >= 2 * half[1]
        ]
        if not ok:
            return None
        areas = np.array([(r[1] - r[0]) * (r[3] - r[2]) for r in ok])
        r = ok[rng.choice(len(ok), p=areas / areas.sum())]
        return np.array(
            [
                rng.uniform(r[0] + half[0], r[1] - half[0]),
                rng.uniform(r[2] + half[1], r[3] - half[1]),
            ]
        )

    def _sample_tray(self, rng, half):
        """Sample the tray in the reachable band: y pinned to the band center
        (the band is exactly one tray tall), x uniform across the zone rect."""
        if self.tray_zone is None:
            return None
        ok = [r for r in self.tray_zone if r[1] - r[0] >= 2 * half[0]]
        if not ok:
            return None
        areas = np.array([r[1] - r[0] for r in ok])
        r = ok[rng.choice(len(ok), p=areas / areas.sum())]
        return np.array(
            [rng.uniform(r[0] + half[0], r[1] - half[0]), (r[2] + r[3]) / 2]
        )

    def _sample_separated(self, rng, half_a, half_b, min_dist=0.35):
        """Fallback when a zone is too small: sample both in the whole usable
        area, keeping them at least min_dist apart (different spots)."""
        a = b = None
        for _ in range(200):
            a = self._sample_in_rects(self.usable_regions, rng, half_a)
            b = self._sample_in_rects(self.usable_regions, rng, half_b)
            if a is not None and b is not None and np.linalg.norm(a - b) >= min_dist:
                return a, b
        if a is None or b is None:
            raise RuntimeError("Usable counter area too small for cup and tray")
        return a, b

    # ------------------------------------------------------------------ #
    # Task logic
    # ------------------------------------------------------------------ #

    def evaluate(self):
        cup_pos = self.cup.pose.p
        tray_pos = self.tray.pose.p
        is_grasped = self.agent.is_grasping(self.cup)
        is_static = (
            torch.linalg.norm(self.cup.linear_velocity, dim=1) <= 0.1
        ) & (torch.linalg.norm(self.cup.angular_velocity, dim=1) <= 0.2)
        xy_off = torch.abs(cup_pos[:, :2] - tray_pos[:, :2])
        tray_xy_tol = torch.as_tensor(self.tray_half[:2], device=self.device) + 0.05
        on_tray_xy = (xy_off[:, 0] <= tray_xy_tol[0]) & (xy_off[:, 1] <= tray_xy_tol[1])
        tray_top = tray_pos[:, 2] + self.tray_half[2]
        on_tray_z = torch.abs(cup_pos[:, 2] - tray_top - self.cup_half[2]) <= 0.10
        return dict(success=on_tray_xy & on_tray_z & ~is_grasped & is_static)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(self.camera_pos, self.counter_pos)
        return [
            CameraConfig("base_camera", pose, 128, 128, 60 * np.pi / 180, 0.01, 100)
        ]

    @property
    def _default_human_render_camera_configs(self):
        """Three static video cameras for the takeitback task:
          * main_camera: FULL REAR VIEW from the south (behind the robot),
            elevated and wide (fov 100 deg) - the whole countertop plus the
            robot's floor drive area, so base driving is always visible;
          * tray_camera: SIDE view from the east, aimed at the tray band
            (y in [-0.50, -0.20]) - profile of the tray regrasp / placement;
          * cup_camera: WEST 3/4 view aimed at the cup's initial strip
            (front counter edge) - the first grasp, from the opposite side
            of the tray camera.
        The wall cabinets overhang the counter back (y > -0.45), so no camera
        looks straight down from above the counter; the rear view looks over
        the counter from the south and stays clear of the overhang."""
        cx, cy = self.counter_pos[:2]
        return [
            CameraConfig(
                "main_camera",
                sapien_utils.look_at(
                    np.array([cx + 0.3, cy - 1.575, 2.2]),
                    np.array([cx + 0.3, cy - 0.175, 0.7]),
                ),
                1600, 1600, 100 * np.pi / 180, 0.01, 100,
            ),
            CameraConfig(
                "tray_camera",
                sapien_utils.look_at(
                    np.array([cx + 1.85, cy - 0.525, 1.35]),
                    np.array([cx + 0.7, cy - 0.025, 0.95]),
                ),
                1600, 1600, 55 * np.pi / 180, 0.01, 100,
            ),
            CameraConfig(
                "cup_camera",
                sapien_utils.look_at(
                    np.array([cx - 0.25, cy - 1.225, 1.35]),
                    np.array([cx + 0.65, cy - 0.225, 1.0]),
                ),
                1600, 1600, 55 * np.pi / 180, 0.01, 100,
            ),
        ]
