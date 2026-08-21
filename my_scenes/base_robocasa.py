import numpy as np
import sapien
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.scene_builder.robocasa.scene_builder import RoboCasaSceneBuilder
from mani_skill.utils.structs import Pose
from utils.scene_utils import degree_to_quanterion

class BaseRoboCasaScene(BaseEnv):
    SUPPORTED_ROBOTS = ["fetch", "none"]
    SUPPORTED_REWARD_MODES = ["none"]
    FIXTURE_SEED: int = None

    fixture_placements: dict[str, dict[str, object]]
    counter_pos: np.ndarray
    counter_size: np.ndarray
    cup_pos: np.ndarray

    def __init__(self, robot_uids="fetch", *args, **kwargs):
        super().__init__(robot_uids=robot_uids, *args, **kwargs)
        self.fixture_placements = {}

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        if self.FIXTURE_SEED is not None:
            self._set_episode_rng(self.FIXTURE_SEED, torch.arange(self.num_envs, device=self.device))
        self.scene_builder = RoboCasaSceneBuilder(self)
        self.scene_builder.build()
        self.fixture_placements = {
            config["name"]
            + "_"
            + getattr(
                getattr(config["model"], "__class__", None), "__name__", "Wrong"
            ): {
                "pos": getattr(config["model"], "pos", None),
                "quat": getattr(config["model"], "quat", None),
                "size": getattr(config["model"], "size", None),
            }
            for config in self.scene_builder.scene_data[0].get("fixture_cfgs")
        }

        self.counter_pos = self.fixture_placements["counter_main_main_group_Counter"][
            "pos"
        ]
        self.counter_size = self.fixture_placements["counter_main_main_group_Counter"][
            "size"
        ]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        agent_pos = self.agent.robot.pose.p[0]
        agent_pos[0] = self.cup_pos[0]
        agent_pos[1] = self.cup_pos[1]
        agent_pos[1] -= self.counter_size[1] * 1.6

        q = degree_to_quanterion(z=180)

        self.agent_pose = Pose.create_from_pq(p=agent_pos, q=q)
        self.agent.robot.set_pose(self.agent_pose)


class BaseRoboCasaSimple(BaseRoboCasaScene):
    """RoboCasa kitchen base with a computed usable area on the main counter and
    free robot placement on the floor.

    Usable area: the main counter top inset from the edges (EDGE_INSET), minus
    the footprints of fixtures occupying the counter surface (e.g. the sink set
    flush into the counter, counter accessories standing on it). Fixtures are
    detected generically from fixture_placements: any fixture whose vertical
    span reaches the counter top plane and whose footprint overlaps the counter
    is subtracted. Results: self.usable_regions (N, 4) rects [x0, x1, y0, y1]
    at counter top height, self.usable_area (m^2), self.blockers.

    Free placement: the robot is spawned at a random free floor spot, computed
    once at load time as the floor area minus every fixture/object footprint
    (collision AABBs, so rotated fixture groups are handled), inflated by
    ROBOT_RADIUS + FIXTURE_MARGIN. The spawn is re-sampled every episode with a
    random yaw. Candidates: self.free_cells (N, 2), self.floor_bounds.
    """

    FIXTURE_SEED = 102

    # usable area
    EDGE_INSET = 0.08  # inset of the usable area from the counter edges (m)
    BLOCKER_GAP = 0.03  # extra margin around fixtures blocking the counter (m)
    Z_TOL = 0.02  # z tolerance for a fixture to occupy the counter top plane (m)
    MIN_OVERLAP = 0.005  # min x/y overlap with the counter to be a blocker (m)

    # free placement
    ROBOT_RADIUS = 0.35  # fetch base ~0.54 m wide -> half-width + slack
    FIXTURE_MARGIN = 0.15  # min distance between robot base edge and fixtures
    GRID_CELL = 0.1  # free-space grid resolution

    cup_pos: np.ndarray
    usable_regions: np.ndarray
    usable_area: float
    blockers: list[np.ndarray]
    free_cells: np.ndarray  # [N, 2] candidate robot (x, y) positions on the floor
    floor_bounds: np.ndarray  # [x0, x1, y0, y1] of the floor in world coords
    agent_pose: sapien.Pose

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        # NOTE: fixture_placements is cleared after __init__ (see BaseRoboCasaScene),
        # so everything derived from it must be captured here.
        self.blockers = self._counter_blockers()
        self.usable_regions = self._compute_usable_regions()
        self.usable_area = float(
            sum((r[1] - r[0]) * (r[3] - r[2]) for r in self.usable_regions)
        )
        # default focus point on the counter (task scenes may override)
        self.cup_pos = self.counter_pos.copy()
        self.cup_pos[1] -= self.counter_size[1] / 4
        self.free_cells = self._compute_free_cells()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # random free floor spawn (BaseRoboCasaScene places the robot at a fixed spot)
        pos = self.free_cells[self._main_rng.randint(len(self.free_cells))]
        agent_pos = self.agent.robot.pose.p[0]
        agent_pos[0] = pos[0]
        agent_pos[1] = pos[1]
        q = degree_to_quanterion(z=int(self._main_rng.uniform(0, 360)))
        self.agent_pose = Pose.create_from_pq(p=agent_pos, q=q)
        self.agent.robot.set_pose(self.agent_pose)

    # ------------------------------------------------------------------ #
    # Usable area on the main counter
    # ------------------------------------------------------------------ #

    def _compute_usable_regions(self) -> np.ndarray:
        """Main counter top inset by EDGE_INSET, minus blocking fixtures."""
        cx, cy = self.counter_pos[:2]
        sx, sy = self.counter_size[:2]
        i = self.EDGE_INSET
        regions = [
            np.array(
                [cx - sx / 2 + i, cx + sx / 2 - i, cy - sy / 2 + i, cy + sy / 2 - i]
            )
        ]
        for b in self._counter_blockers():
            gap = self.BLOCKER_GAP
            blocker = np.array([b[0] - gap, b[1] + gap, b[2] - gap, b[3] + gap])
            regions = self._subtract_rects(regions, blocker)
        return np.asarray(regions)

    def _counter_blockers(self) -> list[np.ndarray]:
        """XY rects (clipped to the counter) of fixtures occupying the counter top."""
        c_pos, c_size = self.counter_pos, self.counter_size
        top_z = c_pos[2] + c_size[2] / 2
        x0, x1 = c_pos[0] - c_size[0] / 2, c_pos[0] + c_size[0] / 2
        y0, y1 = c_pos[1] - c_size[1] / 2, c_pos[1] + c_size[1] / 2
        blockers = []
        for name, fp in self.fixture_placements.items():
            if name.endswith("_Counter"):
                continue  # other counters are edge-aligned, never on the top
            pos, size = fp["pos"], fp["size"]
            if pos is None or size is None:
                continue
            pos = np.asarray(pos, dtype=float)
            size = np.asarray(size, dtype=float)
            # vertical span must reach the counter top plane: the sink is flush
            # with the surface, accessories stand on it
            if not (
                pos[2] - size[2] / 2 - self.Z_TOL
                <= top_z
                <= pos[2] + size[2] / 2 + self.Z_TOL
            ):
                continue
            fx0, fx1 = pos[0] - size[0] / 2, pos[0] + size[0] / 2
            fy0, fy1 = pos[1] - size[1] / 2, pos[1] + size[1] / 2
            ox0, ox1 = max(x0, fx0), min(x1, fx1)
            oy0, oy1 = max(y0, fy0), min(y1, fy1)
            if ox1 - ox0 > self.MIN_OVERLAP and oy1 - oy0 > self.MIN_OVERLAP:
                blockers.append(np.array([ox0, ox1, oy0, oy1]))
        return blockers

    @staticmethod
    def _subtract_rects(regions, blocker):
        """Subtract blocker [x0, x1, y0, y1] from a list of rects (axis-aligned)."""
        out = []
        bx0, bx1, by0, by1 = blocker
        for r in regions:
            x0, x1, y0, y1 = r
            ix0, ix1 = max(x0, bx0), min(x1, bx1)
            iy0, iy1 = max(y0, by0), min(y1, by1)
            if ix0 >= ix1 or iy0 >= iy1:
                out.append(r)
                continue
            if x0 < ix0:
                out.append([x0, ix0, y0, y1])
            if ix1 < x1:
                out.append([ix1, x1, y0, y1])
            if y0 < iy0:
                out.append([ix0, ix1, y0, iy0])
            if iy1 < y1:
                out.append([ix0, ix1, iy1, y1])
        return out

    def sample_placement_pos(self, rng: np.random.RandomState = None) -> np.ndarray:
        """Sample a random [x, y, z] inside the usable area (z = counter top surface)."""
        if len(self.usable_regions) == 0:
            raise RuntimeError("No usable area on the main counter")
        if rng is None:
            rng = np.random.default_rng()
        areas = np.array(
            [(r[1] - r[0]) * (r[3] - r[2]) for r in self.usable_regions]
        )
        r = self.usable_regions[rng.choice(len(areas), p=areas / areas.sum())]
        z = self.counter_pos[2] + self.counter_size[2] / 2
        return np.array([rng.uniform(r[0], r[1]), rng.uniform(r[2], r[3]), z])

    # ------------------------------------------------------------------ #
    # Free robot placement on the floor
    # ------------------------------------------------------------------ #

    @staticmethod
    def _body_aabbs(body):
        """World AABB [lower, upper] of an Actor or Articulation, merged over all bodies.
        Returns None if the body has no collision shapes (visual-only, cannot collide)."""
        if hasattr(body, "links"):
            bodies = [b for link in body.links for b in link._bodies]
        else:
            bodies = body._bodies
        aabbs = []
        for b in bodies:
            if not b.get_collision_shapes():
                continue
            aabbs.append(b.get_global_aabb_fast())
        if not aabbs:
            return None
        aabbs = np.array(aabbs)  # [n, 2, 3]
        return aabbs[:, 0].min(axis=0), aabbs[:, 1].max(axis=0)

    def _compute_free_cells(self) -> np.ndarray:
        floor_cfg = next(
            cfg
            for name, cfg in self.fixture_placements.items()
            if name.endswith("_Floor") and "backing" not in name
        )
        fpos, fsize = np.asarray(floor_cfg["pos"]), np.asarray(floor_cfg["size"])
        self.floor_bounds = np.array(
            [
                fpos[0] - fsize[0] / 2,
                fpos[0] + fsize[0] / 2,
                fpos[1] - fsize[1] / 2,
                fpos[1] + fsize[1] / 2,
            ]
        )
        xs = np.arange(
            self.floor_bounds[0] + self.ROBOT_RADIUS,
            self.floor_bounds[1] - self.ROBOT_RADIUS,
            self.GRID_CELL,
        )
        ys = np.arange(
            self.floor_bounds[2] + self.ROBOT_RADIUS,
            self.floor_bounds[3] - self.ROBOT_RADIUS,
            self.GRID_CELL,
        )
        occupied = np.zeros((len(xs), len(ys)), dtype=bool)
        inflate = self.ROBOT_RADIUS + self.FIXTURE_MARGIN
        for name, body in {**self.scene.actors, **self.scene.articulations}.items():
            if name.startswith("floor") or body is self.agent.robot:
                continue
            aabb = self._body_aabbs(body)
            if aabb is None:
                continue
            lower, upper = aabb
            mask = (
                (xs >= lower[0] - inflate) & (xs <= upper[0] + inflate)
            )[:, None] & (
                (ys >= lower[1] - inflate) & (ys <= upper[1] + inflate)
            )[None, :]
            occupied |= mask
        cells = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1)[~occupied]
        if len(cells) == 0:
            raise RuntimeError(
                f"No free floor spot for the robot in {self.__class__.__name__}"
            )
        return cells
