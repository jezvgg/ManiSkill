import multiprocessing as mp
import os
from copy import deepcopy
import time
import argparse
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import os.path as osp
import numpy as np
from transforms3d.euler import euler2quat
from typing import Callable
import toppra as ta
import mplib
from mplib.sapien_utils.conversion import convert_object_name
from mplib.collision_detection.fcl import CollisionGeometry
from mplib.sapien_utils import SapienPlanner, SapienPlanningWorld
from mplib.collision_detection.fcl import Convex, CollisionObject, FCLObject
from mplib.collision_detection import fcl
from mplib.sapien_utils.urdf_exporter import export_kinematic_chain_urdf
from mplib.sapien_utils.srdf_exporter import export_srdf

import sapien
import sapien.physx as physx
from sapien import Entity
from sapien.physx import (
    PhysxArticulation,
    PhysxArticulationLinkComponent,
    PhysxCollisionShapeConvexMesh
)


from typing import Literal, Optional, Sequence, Union
import sys
import trimesh
from mani_skill.utils.structs.pose import to_sapien_pose
from mani_skill.utils.wrappers.record import RecordEpisode
from .stepping import pose_error
from .root_frame import (
    fold_root,
    is_planar_root,
    planar_base,
    unfold_root,
    unfold_root_rates,
)
from mani_skill.trajectory.merge_trajectory import merge_trajectories
from mani_skill.examples.motionplanning.panda.solutions import solvePushCube, solvePickCube, solveStackCube, solvePegInsertionSide, solvePlugCharger, solvePullCubeTool, solveLiftPegUpright, solvePullCube
from mani_skill.envs.tasks import PickCubeEnv
from mani_skill.utils.geometry.trimesh_utils import get_component_mesh
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver

from mani_skill.utils import common
from mani_skill.utils.structs import Actor

BAD_ENV_ERROR_CODE = -1234

def attach_object(  # type: ignore
    planning_world: SapienPlanningWorld,
    obj: Union[Entity, str],
    articulation: Union[PhysxArticulation, str],
    link: Union[PhysxArticulationLinkComponent, int],
    pose: Optional[mplib.Pose] = None,
    *,
    touch_links: Optional[list[Union[PhysxArticulationLinkComponent, str]]] = None,
    obj_geom: Optional[CollisionGeometry] = None,
) -> None:
    """
    Attaches given non-articulated object to the specified link of articulation.

    Updates ``acm_`` to allow collisions between attached object and touch_links.

    Frame (K53): mplib composes the attached body as ``link_pose_in_base_frame *
    pose`` and, with ``pose=None``, stores ``link_pose_in_base_frame^-1 *
    object_pose`` — the articulation's base pose is left out, so with a robot
    standing away from the world origin the body is drawn right only in the
    configuration it was attached in (measured 3.296 m off at shoulder_pan +90 deg).
    ``SapienPlanningWorldV2`` therefore keeps the planned mobile base at the
    identity base pose with the robot's world pose folded into its root joints
    (`root_frame.py`); against that world the stored transform is the true
    ``T_link_obj = link_world^-1 * obj_world`` and mplib draws the body where the
    hand is, in every configuration. Nothing here changes; call it on a
    ``SapienPlanningWorldV2`` and sync (`update_from_simulation`) afterwards.

    :param obj: the non-articulated object (or its name) to attach
    :param articulation: the planned articulation (or its name) to attach to
    :param link: the link of the planned articulation (or its index) to attach to
    :param pose: attached pose (relative pose from attached link to object, in
        the planning world's frame of that link). If ``None``, attach the object
        at its current pose.
    :param touch_links: links (or their names) that the attached object touches.
        When ``None``,

        * if the object is not currently attached, touch_links are set to the name
        of articulation links that collide with the object in the current state.

        * if the object is already attached, touch_links of the attached object
        is preserved and ``acm_`` remains unchanged.
    :param obj_geom: a CollisionGeometry object representing the attached object.
        If not ``None``, pose must be not ``None``.

    .. raw:: html

        <details>
        <summary><a>Overloaded
        <code class="docutils literal notranslate">
        <span class="pre">PlanningWorld.attach_object()</span>
        </code>
        methods</a></summary>
    .. automethod:: mplib.PlanningWorld.attach_object
        :no-index:
    .. raw:: html
        </details>
    """
    kwargs = {"name": obj, "art_name": articulation, "link_id": link}
    if pose is not None:
        kwargs["pose"] = pose
    if touch_links is not None:
        kwargs["touch_links"] = [
            l.name if isinstance(l, PhysxArticulationLinkComponent) else l
            for l in touch_links  # noqa: E741
        ]
    if obj_geom is not None:
        kwargs["obj_geom"] = obj_geom

    if isinstance(obj, Entity):
        kwargs["name"] = convert_object_name(obj)
    if isinstance(articulation, PhysxArticulation):
        kwargs["art_name"] = articulation = convert_object_name(articulation)
    if isinstance(link, PhysxArticulationLinkComponent):
        kwargs["link_id"] = (
            planning_world.get_articulation(articulation)
            .get_pinocchio_model()
            .get_link_names()
            .index(link.name)
        )

    planning_world.attach_object(**kwargs)


def get_fcl_object_name(entity):
    component = entity._objs[0].find_component_by_type(physx.PhysxRigidBaseComponent)
    return convert_object_name(component.entity)


def compute_box_grasp_thin_side_info(
    obb: trimesh.primitives.Box,
    target_closing=None,
    ee_direction=None,
    depth=0.0,
    ortho=True,
):
    """Compute grasp info given an oriented bounding box.
    The grasp info includes axes to define grasp frame, namely approaching, closing, orthogonal directions and center.

    Args:
        obb: oriented bounding box to grasp
        approaching: direction to approach the object
        target_closing: target closing direction, used to select one of multiple solutions
        depth: displacement from hand to tcp along the approaching vector. Usually finger length.
        ortho: whether to orthogonalize closing  w.r.t. approaching.
    """
    # NOTE(jigu): DO NOT USE `x.extents`, which is inconsistent with `x.primitive.transform`!
    extents = np.array(obb.primitive.extents)
    T = np.array(obb.primitive.transform)

    inds = np.argsort(extents[:2])
    short_base_side_ind = inds[0]
    long_base_side_ind = inds[1]

    height = extents[2]

    approaching = np.array(T[:3, long_base_side_ind])
    approaching = common.np_normalize_vector(approaching)

    if ee_direction @ approaching < 0:
        approaching = -approaching

    closing = np.array(T[:3, short_base_side_ind])

    if target_closing is not None and target_closing @ closing < 0:
        closing = -closing

    if ortho:
        closing = closing - (approaching @ closing) * approaching
        closing = common.np_normalize_vector(closing)

    # Find the origin on the surface
    center = T[:3, 3]
    half_size = extents[long_base_side_ind] / 2
    center = center + approaching * (-half_size + min(depth, half_size))

    grasp_info = dict(
        approaching=approaching, closing=closing, center=center, extents=extents
    )
    return grasp_info

def convert_actor_convex_mesh_to_fcl(actor: Actor):
    component = actor._objs[0].find_component_by_type(physx.PhysxRigidBaseComponent)
    assert component is not None, (
        f"No PhysxRigidBaseComponent found in {actor.name}: "
        f"{actor.components=}"
    )
    assert len(component.collision_shapes) == 1
    shape = component.collision_shapes[0]
    assert isinstance(shape, physx.PhysxCollisionShapeConvexMesh)

    # tranform vertices, so that scale == 1.0
    vertices = shape.vertices
    vertices[:, 0] *= shape.scale[0]
    vertices[:, 1] *= shape.scale[1]
    vertices[:, 2] *= shape.scale[2]
    c_geom = Convex(vertices=vertices, faces=shape.triangles)
    collision_shape = CollisionObject(c_geom)

    return FCLObject(
        convert_object_name(component.entity),
        component.entity.pose,
        [collision_shape],
        [mplib.Pose(shape.local_pose)],
    )

def is_mesh_cylindrical(actor, to_world_frame=True, thresh=5e-3):
    mesh = get_component_mesh(
        actor._objs[0].find_component_by_type(physx.PhysxRigidDynamicComponent),
        to_world_frame=to_world_frame,
    )
    assert mesh is not None, "can not get actor mesh for {}".format(actor)

    obb: trimesh.primitives.Box = mesh.bounding_box_oriented
    cylinder: trimesh.primitives.Cylinder = mesh.bounding_cylinder
    cylinder_obb: trimesh.primitives.Box = cylinder.bounding_box_oriented

    h_obb, w_obb = obb.primitive.extents[:2]
    h_c_obb, w_c_obb = cylinder_obb.primitive.extents[:2]

    #if extents are equal up to the permutation then the mesh is cylindrical
    if np.abs(h_obb * w_obb - h_c_obb * w_c_obb) < thresh and \
        np.abs(h_obb + w_obb - h_c_obb - w_c_obb) < thresh:
        return True
    return False
    

class SapienPlanningWorldV2(SapienPlanningWorld):
    """
    Patched version of SapienPlanningWorld for meshes with scale — and, for a planned
    articulation with a planar mobile base, the robot's world pose folded into its
    root joints (K53, `root_frame.py`).

    mplib 0.2.1 draws an attached body at ``link_in_base_frame * attached_pose`` —
    the articulation's base pose is left out — so with the robot standing away from
    the world origin a held object orbits the origin as soon as a link rotates
    (measured 3.296 m from the hand at shoulder_pan +90 deg; docs/lab-journal.md
    2026-08-18). Here the planned articulation's base pose is kept the identity and
    ``root_pose * Trans(x, y) * RotZ(yaw)`` is expressed as root-joint values instead,
    at construction and at every ``update_from_simulation``; then mplib's own
    ``link.inv() * entity.pose`` *is* the rigid link->object transform and the body is
    drawn where the hand is in every configuration. ``fold_qpos`` / ``unfold_qpos``
    are the two-way translation ``SapienPlannerV2`` applies at its boundary, so
    callers keep passing and receiving simulator qpos. (``unfold_rates`` and
    ``root_frame.unfold_root_rates`` are kept as tested helpers but are unused since
    the unfold moved *before* TOPP in ``plan_pose``/``plan_qpos`` — the rates come
    out of TOPP already in the simulator's frame.)

    ``root_folded`` names the planned articulation that is folded, or None (no
    planar root joints — a fixed-base arm — or a root pose with height/tilt, which
    the three planar joints cannot express; then mplib's own framing is kept and
    ``attach_object`` stays phantom-prone, as before).
    """
    def __init__(
        self,
        sim_scene: sapien.Scene,
        planned_articulations: list[PhysxArticulation] = [],  # noqa: B006
        disable_actors_collision=False,
    ):
        """
        Creates an mplib.PlanningWorld from a sapien.Scene.

        :param planned_articulations: list of planned articulations.
        """
        mplib.PlanningWorld.__init__(self, [])
        self._sim_scene = sim_scene
        self.disable_actors_collision = disable_actors_collision
        # name -> the simulator articulation whose root pose is folded into its
        # root joints (K53); at most the planned articulation.
        self._folded: dict[str, PhysxArticulation] = {}

        articulations: list[PhysxArticulation] = sim_scene.get_all_articulations()
        actors: list[Entity] = sim_scene.get_all_actors()

        for articulation in articulations:
            if not self.disable_actors_collision or articulation in planned_articulations:
                urdf_str = export_kinematic_chain_urdf(articulation)
                srdf_str = export_srdf(articulation)

                # Convert all links to FCLObject
                collision_links = [
                    fcl_obj
                    for link in articulation.links
                    if (fcl_obj := self.convert_physx_component(link)) is not None
                ]

                joint_names = [j.name for j in articulation.active_joints]
                articulated_model = mplib.ArticulatedModel.create_from_urdf_string(
                    urdf_str,
                    srdf_str,
                    collision_links=collision_links,
                    gravity=sim_scene.get_physx_system().config.gravity,  # type: ignore
                    link_names=[link.name for link in articulation.links],
                    joint_names=joint_names,
                    verbose=False,
                )
                if articulation in planned_articulations and self._can_fold(articulation, joint_names):
                    self._folded[convert_object_name(articulation)] = articulation
                self._place(articulated_model, articulation)
                self.add_articulation(articulated_model)

        for articulation in planned_articulations:
            self.set_articulation_planned(convert_object_name(articulation), True)

        # if not self.disable_actors_collision:
        for entity in actors:
            if self.disable_actors_collision and 'food' in entity.name:
                continue
            component = entity.find_component_by_type(sapien.physx.PhysxRigidBaseComponent)
            assert component is not None, (
                f"No PhysxRigidBaseComponent found in {entity.name}: "
                f"{entity.components=}"
            )

            # Convert collision shapes at current global pose
            if (fcl_obj := self.convert_physx_component(component)) is not None:  # type: ignore
                self.add_object(fcl_obj)

    # ------------------------------------------------------------ K53: the fold --

    @staticmethod
    def _can_fold(articulation: PhysxArticulation, joint_names) -> bool:
        """Planar root joints first in the chain, and a planar root pose.

        ``MIKASA_DISABLE_ROOT_FOLD=1`` forces this to False. It exists to isolate what
        K53 costs and buys: with the fold off, the planning world keeps mplib's own base
        representation, which is what jezv's solver uses. **Only safe where nothing is
        attached** — the fold is what puts a held object in the hand rather than 3.3 m
        away, so switching it off on a task that carries something reinstates the
        phantom. The inherited planners call ``attach_object`` zero times, which is why
        the comparison against them may use it.
        """
        if os.environ.get("MIKASA_DISABLE_ROOT_FOLD") == "1":
            print("[planning world] MIKASA_DISABLE_ROOT_FOLD=1: base pose NOT folded "
                  "(diagnostic; reinstates mplib's frame error for attached bodies)", flush=True)
            return False
        if not is_planar_root(joint_names):
            return False
        try:
            planar_base(articulation.root_pose.p, articulation.root_pose.q)
        except ValueError as e:
            print(f"[planning world] {articulation.name}: base pose NOT folded into the root joints "
                  f"({e}); attached bodies keep mplib's frame error (K53)", flush=True)
            return False
        return True

    @property
    def root_folded(self) -> str | None:
        """Name of the planned articulation whose root pose is folded, or None."""
        return next(iter(self._folded), None)

    def _root_pose(self):
        """`(p, q)` of the folded articulation's simulator root pose, read live."""
        art = self._folded[self.root_folded]
        return np.asarray(art.root_pose.p, dtype=np.float64), np.asarray(art.root_pose.q, dtype=np.float64)

    def _place(self, model, articulation: PhysxArticulation) -> None:
        """Put the planning model where the simulator has the articulation: folded for
        the planned mobile base, mplib's own `base_pose + qpos` for everything else."""
        if convert_object_name(articulation) in self._folded:
            model.set_base_pose(mplib.Pose())
            model.set_qpos(fold_root(articulation.root_pose.p, articulation.root_pose.q, articulation.qpos), full=True)
        else:
            model.set_base_pose(articulation.root_pose)  # type: ignore
            model.set_qpos(articulation.qpos, full=True)  # type: ignore

    def fold_qpos(self, qpos):
        """Simulator qpos (root joints first) -> the planning model's; identity if not folded."""
        if self.root_folded is None:
            return np.asarray(qpos, dtype=np.float64)
        p, q = self._root_pose()
        return fold_root(p, q, qpos)

    def unfold_qpos(self, qpos, ref_yaw: float | None = None):
        """The planning model's qpos (1-D or rows) -> simulator qpos; identity if not folded."""
        if self.root_folded is None:
            return np.asarray(qpos, dtype=np.float64)
        p, q = self._root_pose()
        return unfold_root(p, q, qpos, ref_yaw=ref_yaw)

    def unfold_rates(self, rates):
        """The planning model's joint rates/accelerations -> the simulator's root frame.

        Unused in production since ``plan_qpos`` unfolds the knots before TOPP; kept
        (with its test) as the helper for anyone who time-parameterises a folded path
        themselves.
        """
        if self.root_folded is None:
            return np.asarray(rates, dtype=np.float64)
        _, q = self._root_pose()
        return unfold_root_rates(q, rates)

    def update_from_simulation(self, *, update_attached_object: bool = True) -> None:
        """mplib's sync, with the planned mobile base kept folded (K53).

        Articulations: base pose + qpos from the simulator — through `_place`, so the
        planned one stays at the identity base with its root pose in the root joints.
        Attached bodies: mplib's `attached_body.pose = link.inv() * entity.pose`,
        which against the folded articulation is the true link->object transform.
        Free actors: overwritten at their simulator pose. Same contract and same
        errors as `SapienPlanningWorld.update_from_simulation`.
        """
        for articulation in self._sim_scene.get_all_articulations():
            if art := self.get_articulation(convert_object_name(articulation)):
                self._place(art, articulation)
            else:
                raise RuntimeError(
                    f"Articulation {articulation.name} not found in PlanningWorld! "
                    "The scene might have changed since last update."
                )

        for entity in self._sim_scene.get_all_actors():
            object_name = convert_object_name(entity)
            if attached_body := self.get_attached_object(object_name):
                if update_attached_object:
                    attached_body.pose = (
                        attached_body.get_attached_link_global_pose().inv() * entity.pose  # type: ignore
                    )
                attached_body.update_pose()
            elif fcl_obj := self.get_object(object_name):
                self.add_object(
                    FCLObject(object_name, entity.pose, fcl_obj.shapes, fcl_obj.shape_poses)  # type: ignore
                )
            elif (
                len(entity.find_component_by_type(physx.PhysxRigidBaseComponent).collision_shapes) > 0  # type: ignore
            ):
                raise RuntimeError(
                    f"Entity {entity.name} not found in PlanningWorld! "
                    "The scene might have changed since last update."
                )

    @staticmethod
    def convert_physx_component(comp: physx.PhysxRigidBaseComponent) -> FCLObject | None:
        """
        Converts a SAPIEN physx.PhysxRigidBaseComponent to an FCLObject.
        All shapes in the returned FCLObject are already set at their world poses.

        :param comp: a SAPIEN physx.PhysxRigidBaseComponent.
        :return: an FCLObject containing all collision shapes in the Physx component.
            If the component has no collision shapes, return ``None``.
        """
        shapes: list[CollisionObject] = []
        shape_poses: list[mplib.Pose] = []
        for shape in comp.collision_shapes:
            shape_poses.append(mplib.Pose(shape.local_pose))  # type: ignore

            if isinstance(shape, physx.PhysxCollisionShapeBox):
                c_geom = fcl.Box(side=shape.half_size * 2)
            elif isinstance(shape, physx.PhysxCollisionShapeCapsule):
                c_geom = fcl.Capsule(radius=shape.radius, lz=shape.half_length * 2)
                # NOTE: physx Capsule has x-axis along capsule height
                # FCL Capsule has z-axis along capsule height
                shape_poses[-1] *= mplib.Pose(q=euler2quat(0, np.pi / 2, 0))
            elif isinstance(shape, PhysxCollisionShapeConvexMesh):
                # assert np.allclose(
                #     shape.scale, 1.0
                # ), f"Not unit scale {shape.scale}, need to rescale vertices?"

                # Scale vertices!
                vertices = shape.vertices
                vertices[:, 0] *= shape.scale[0]
                vertices[:, 1] *= shape.scale[1]
                vertices[:, 2] *= shape.scale[2]
                c_geom = Convex(vertices=vertices, faces=shape.triangles)
            elif isinstance(shape, physx.PhysxCollisionShapeCylinder):
                c_geom = fcl.Cylinder(radius=shape.radius, lz=shape.half_length * 2)
                # NOTE: physx Cylinder has x-axis along cylinder height
                # FCL Cylinder has z-axis along cylinder height
                shape_poses[-1] *= mplib.Pose(q=euler2quat(0, np.pi / 2, 0))
            elif isinstance(shape, physx.PhysxCollisionShapePlane):
                # PhysxCollisionShapePlane are actually a halfspace
                # https://nvidia-omniverse.github.io/PhysX/physx/5.3.1/docs/Geometry.html#planes
                # PxPlane's Pose determines its normal and offert (normal is +x)
                n = shape_poses[-1].to_transformation_matrix()[:3, 0]
                d = n.dot(shape_poses[-1].p)
                c_geom = fcl.Halfspace(n=n, d=d)
                shape_poses[-1] = mplib.Pose()
            elif isinstance(shape, physx.PhysxCollisionShapeSphere):
                c_geom = fcl.Sphere(radius=shape.radius)
            elif isinstance(shape, physx.PhysxCollisionShapeTriangleMesh):
                c_geom = fcl.BVHModel()
                c_geom.begin_model()
                c_geom.add_sub_model(vertices=shape.vertices, faces=shape.triangles)  # type: ignore
                c_geom.end_model()
            else:
                raise TypeError(f"Unknown shape type: {type(shape)}")
            shapes.append(CollisionObject(c_geom))
            
        if len(shapes) == 0:
            return None

        return FCLObject(
            comp.name
            if isinstance(comp, PhysxArticulationLinkComponent)
            else convert_object_name(comp.entity),
            comp.entity.pose,  # type: ignore
            shapes,
            shape_poses,
        )
    
class SapienPlannerV2(SapienPlanner):
    """mplib's SapienPlanner with the repo's `plan_screw`/`plan_pose`, and — when the
    planning world folds the robot's world pose into its root joints (K53,
    `SapienPlanningWorldV2`) — the two-way translation at the boundary: every qpos a
    caller passes is the simulator's and is folded before mplib sees it; every plan
    is unfolded — the path knots back into the root's frame, the yaw on the 2π branch
    next to the simulator's — **before** TOPP times them (`plan_screw` and
    `plan_qpos` alike), so the joint limits bind on the root's own axes and
    `follow_*` in extand.py executes exactly what it did before. `set_base_pose` is a
    check, not a setting, when folded: the world pose lives in the joints."""

    def set_base_pose(self, pose: mplib.Pose):
        """Where the base is w.r.t. the world — mplib's, unless the world folds it (K53).

        Folded: the planning articulation's base pose stays the identity; ``pose``
        must be the simulator's root pose (checked to 1 mm / 1e-3 in the
        quaternion), and the model is re-placed from the simulator.
        """
        world = self.planning_world
        if getattr(world, "root_folded", None) is None:
            return super().set_base_pose(pose)
        p, q = world._root_pose()
        given = mplib.Pose(pose.p, pose.q) if not isinstance(pose, mplib.Pose) else pose
        gp, gq = np.asarray(given.p, dtype=np.float64), np.asarray(given.q, dtype=np.float64)
        if np.linalg.norm(gp - p) > 1e-3 or min(np.linalg.norm(gq - q), np.linalg.norm(gq + q)) > 1e-3:
            raise ValueError(
                f"set_base_pose({gp.round(4).tolist()}, {gq.round(4).tolist()}) differs from the "
                f"simulator's root pose ({p.round(4).tolist()}, {q.round(4).tolist()}); with the base "
                "pose folded into the root joints (K53) the planner reads it from the simulator"
            )
        world._place(self.robot, world._folded[world.root_folded])

    # -- K53: simulator qpos <-> the planning model's ---------------------------------

    def fold_qpos(self, qpos):
        """Simulator qpos (full or move-group, root joints first) -> the planning model's."""
        return self.planning_world.fold_qpos(qpos) if hasattr(self.planning_world, "fold_qpos") else np.asarray(qpos, dtype=np.float64)

    def unfold_qpos(self, qpos, ref_yaw: float | None = None):
        """The planning model's qpos (1-D or rows) -> simulator qpos."""
        return self.planning_world.unfold_qpos(qpos, ref_yaw=ref_yaw) if hasattr(self.planning_world, "unfold_qpos") else np.asarray(qpos, dtype=np.float64)

    def _root_cols(self) -> list[int] | None:
        """Move-group columns of the three root joints, or None if not folded / not all in the group."""
        if getattr(self.planning_world, "root_folded", None) is None:
            return None
        mgi = list(self.move_group_joint_indices)
        if not all(i in mgi for i in (0, 1, 2)):
            return None
        return [mgi.index(i) for i in (0, 1, 2)]

    def plan_qpos(
        self,
        goal_qposes,
        current_qpos,
        *,
        time_step: float = 0.1,
        rrt_range: float = 0.1,
        planning_time: float = 1,
        fix_joint_limits: bool = True,
        fixed_joint_indices=None,
        simplify: bool = True,
        constraint_function=None,
        constraint_jacobian=None,
        constraint_tolerance: float = 1e-3,
        verbose: bool = False,
        ref_yaw: float = 0.0,
    ) -> dict:
        """mplib 0.2.1's `Planner.plan_qpos` (RRTConnect over the move group, then
        TOPP), with one insertion: the RRT path is unfolded into the simulator's root
        frame **before** it is timed (K53) — as `plan_screw` does — so the joint
        velocity/acceleration limits bind on the root's own x/y axes and the returned
        `position`/`velocity`/`acceleration` are the simulator's directly. Every
        argument means what it means in mplib; `qpos` in is the planning model's
        (folded) — `plan_pose` folds before calling — and out is the simulator's.
        `ref_yaw` pins the unfolded yaw branch (see `root_frame.unfold_root`).
        """
        from mplib.planning.ompl import FixedJoint

        if fixed_joint_indices is None:
            fixed_joint_indices = []
        if fix_joint_limits:
            current_qpos = np.clip(current_qpos, self.joint_limits[:, 0], self.joint_limits[:, 1])
        current_qpos = self.pad_move_group_qpos(current_qpos)

        self.robot.set_qpos(current_qpos, True)
        collisions = self.planning_world.check_collision()
        if len(collisions) > 0:
            print("Invalid start state!")
            for collision in collisions:
                print(f"{collision.link_name1} and {collision.link_name2} collide!")

        move_joint_idx = self.move_group_joint_indices
        goal_qpos_ = [goal_qposes[i][move_joint_idx] for i in range(len(goal_qposes))]
        fixed_joints = set()
        for joint_idx in fixed_joint_indices:
            fixed_joints.add(FixedJoint(0, joint_idx, current_qpos[joint_idx]))
        assert len(current_qpos[move_joint_idx]) == len(goal_qpos_[0])
        status, path = self.planner.plan(
            current_qpos[move_joint_idx],
            goal_qpos_,
            time=planning_time,
            range=rrt_range,
            fixed_joints=fixed_joints,
            simplify=simplify,
            constraint_function=constraint_function,  # type: ignore
            constraint_jacobian=constraint_jacobian,  # type: ignore
            constraint_tolerance=constraint_tolerance,
            verbose=verbose,
        )
        if status != "Exact solution":
            return {"status": f"RRTConnect Failed. {status}"}
        if verbose:
            ta.setup_logging("INFO")
        else:
            ta.setup_logging("WARNING")
        knots = np.array(path, dtype=np.float64)
        cols = self._root_cols()
        if cols is not None:  # K53: the simulator's root frame before timing (as plan_screw)
            knots[:, cols] = self.planning_world.unfold_qpos(knots[:, cols], ref_yaw=ref_yaw)
        times, pos, vel, acc, duration = self.TOPP(knots, time_step)
        return {
            "status": "Success",
            "time": times,
            "position": pos,
            "velocity": vel,
            "acceleration": acc,
            "duration": duration,
        }

    # plan_screw ankor
    def plan_screw(
        self,
        goal_pose: mplib.Pose,
        current_qpos: np.ndarray,
        *,
        qpos_step: float = 0.1,
        time_step: float = 0.1,
        wrt_world: bool = True,
        masked_joints: list = None,
        verbose: bool = False,
        max_iters: int = 200,
        goal_tolerance: tuple[float, float] | None = None,
    ) -> dict[str, str | np.ndarray | np.float64]:
        # plan_screw ankor end
        """
        Plan from a start configuration to a goal pose of the end-effector using
        screw motion

        Args:
            goal_pose: pose of the goal
            current_qpos: current joint configuration — the FULL simulator qpos (all
                joints, root joints first); the clip against ``joint_limits`` below
                needs the full length, a move-group-length vector broadcasts wrongly
            qpos_step: size of the random step
            time_step: time step for the discretization
            wrt_world: if True, interpret the target pose with respect to the
                world frame instead of the base frame
            verbose: if True, will print the log of TOPPRA
            max_iters: give up after this many Jacobian steps
                (``screw plan failed: no convergence after N step(s), ...``). One
                step is 0.1 in joint-space norm; the inherited cup planner's
                alignment screw ran 377 before it collided (docs/lab-journal.md,
                2026-08-18), so 200 is a bound on the pathological case, not a
                budget the good ones approach.
            goal_tolerance: ``(metres, radians)``. When given, a plan whose forward
                kinematics ends further than this from ``goal_pose`` is reported as
                ``screw plan failed: converged X cm / Y deg from the goal ...`` instead
                of ``Success``. Off by default (``None``): the screw integration is a
                first-order approximation and can "converge" far from the goal (the
                burner oracle's hover on seed 3: Success, 12.9 cm off), so callers that
                execute the whole plan and have a fallback — ``static_manipulation``
                (then ``plan_pose``) and ``lift_hand`` — pass a tolerance;
                ``move_base_forward``, which executes only the base part of the plan
                (``follow_moving_forward`` drives the base and holds the arm) and whose
                FK endpoint therefore says nothing about what the robot will do, must
                not. ``rotate_base_z`` is not on either list any more: since K51/D12 it
                does not plan with a screw at all (``base_yaw.py``).

        Returns:
            dict with ``status``; on ``Success`` also the TOPP trajectory
            (``time``, ``position``, ``velocity``, ``acceleration``, ``duration``),
            ``goal_error=(metres, radians)`` of the plan's last knot against
            ``goal_pose`` and ``iterations``. Every failure status starts with
            ``screw plan failed``.
        """
        # Into the limits first, as plan_pose does (fix_joint_limits): a start qpos a
        # hair past a stop — the torso parked on its 0.386 m limit — otherwise fails
        # check_joint_limit on the very first iteration.
        current_qpos = np.clip(
            current_qpos, self.joint_limits[:, 0], self.joint_limits[:, 1]
        )
        current_qpos = self.pad_move_group_qpos(current_qpos)
        # K53: the simulator's qpos into the planning model's frame (the robot's
        # world pose folded into the root joints); the plan is unfolded again before
        # TOPP, so the trajectory handed back is in the simulator's root frame as
        # before. `ref_yaw` pins the unfolded yaw to the simulator's 2π branch.
        ref_yaw = float(current_qpos[2]) if len(current_qpos) > 2 else 0.0
        current_qpos = self.fold_qpos(current_qpos)
        self.robot.set_qpos(current_qpos, True)

        if wrt_world:
            goal_pose = self._transform_goal_to_wrt_base(goal_pose)

        def skew(vec):
            return np.array([
                [0, -vec[2], vec[1]],
                [vec[2], 0, -vec[0]],
                [-vec[1], vec[0], 0],
            ])

        def pose2exp_coordinate(pose: mplib.Pose) -> tuple[np.ndarray, float]:
            def rot2so3(rotation: np.ndarray):
                assert rotation.shape == (3, 3)
                if np.isclose(rotation.trace(), 3):
                    return np.zeros(3), 1
                if np.isclose(rotation.trace(), -1):
                    return np.zeros(3), -1e6
                theta = np.arccos((rotation.trace() - 1) / 2)
                omega = (
                    1
                    / 2
                    / np.sin(theta)
                    * np.array([
                        rotation[2, 1] - rotation[1, 2],
                        rotation[0, 2] - rotation[2, 0],
                        rotation[1, 0] - rotation[0, 1],
                    ]).T
                )
                return omega, theta

            pose_mat = pose.to_transformation_matrix()
            omega, theta = rot2so3(pose_mat[:3, :3])
            if theta < -1e5:
                return omega, theta
            ss = skew(omega)
            inv_left_jacobian = (
                np.eye(3) / theta
                - 0.5 * ss
                + (1.0 / theta - 0.5 / np.tan(theta / 2)) * ss @ ss
            )
            v = inv_left_jacobian @ pose_mat[:3, 3]
            return np.concatenate([v, omega]), theta

        self.pinocchio_model.compute_forward_kinematics(current_qpos)
        ee_index = self.link_name_2_idx[self.move_group]
        # relative_pose = T_base_goal * T_base_link.inv()
        relative_pose = goal_pose * self.pinocchio_model.get_link_pose(ee_index).inv()
        omega, theta = pose2exp_coordinate(relative_pose)

        if theta < -1e4:
            return {"status": "screw plan failed."}
        omega = omega.reshape((-1, 1)) * theta

        move_joint_idx = self.move_group_joint_indices
        path = [np.copy(current_qpos[move_joint_idx])]

        while True:
            self.pinocchio_model.compute_full_jacobian(current_qpos)
            J = self.pinocchio_model.get_link_jacobian(ee_index, local=False)
            mask = np.ones_like(J)
            if masked_joints is not None:
                mask = np.tile(masked_joints, (mask.shape[0], 1)).astype(np.int32)
            J *= mask
            delta_q = np.linalg.pinv(J) @ omega
            delta_q *= qpos_step / (np.linalg.norm(delta_q))
            delta_twist = J @ delta_q

            flag = False
            if np.linalg.norm(delta_twist) > np.linalg.norm(omega):
                ratio = np.linalg.norm(omega) / np.linalg.norm(delta_twist)
                delta_q = delta_q * ratio
                delta_twist = delta_twist * ratio
                flag = True

            current_qpos += delta_q.reshape(-1)
            omega -= delta_twist

            def check_joint_limit(q):
                n = len(q)
                for i in range(n):
                    if (
                        q[i] < self.joint_limits[i][0] - 1e-3
                        or q[i] > self.joint_limits[i][1] + 1e-3
                    ):
                        return False
                return True

            within_joint_limit = check_joint_limit(current_qpos)
            self.planning_world.set_qpos_all(current_qpos[move_joint_idx])
            collide = self.planning_world.is_state_colliding()

            # Three different failures used to share one message. Name the reason:
            # the caller's stdout is what the next debugging turn reads, and "screw
            # plan failed" alone cost a run of MikasaBurner-v0 a whole iteration
            # (docs/lab-journal.md, 2026-08-17). Status still starts with
            # "screw plan failed" so every `!= "Success"` check upstream is unchanged.
            if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:
                if collide:
                    why = "collision"
                    try:
                        pairs = self.planning_world.check_collision()
                        names = sorted({f"{c.link_name1}<->{c.link_name2}" for c in pairs})
                        if names:
                            why += " " + ", ".join(names[:4])
                    except Exception:  # pragma: no cover - diagnostic only
                        pass
                elif not within_joint_limit:
                    over = [
                        i for i, q in enumerate(current_qpos)
                        if q < self.joint_limits[i][0] - 1e-3 or q > self.joint_limits[i][1] + 1e-3
                    ]
                    why = f"joint limit at index {over}"
                else:
                    why = "twist stalled (goal not reachable along the screw)"
                return {
                    "status": f"screw plan failed: {why} after {len(path)} step(s), "
                              f"{np.linalg.norm(omega):.3f} of the twist left"
                }

            path.append(np.copy(current_qpos[move_joint_idx]))
            iterations = len(path) - 1

            if flag:
                # The twist is integrated to first order in a frame fixed at the
                # start, so "no twist left" is not "at the goal": say how far off
                # the plan's last knot really is (FK, both poses in the base frame),
                # and refuse it when the caller asked for a tolerance.
                self.pinocchio_model.compute_forward_kinematics(current_qpos)
                ee_pose = self.pinocchio_model.get_link_pose(ee_index)
                goal_error = pose_error(goal_pose.p, goal_pose.q, ee_pose.p, ee_pose.q)
                if goal_tolerance is not None and (
                    goal_error[0] > goal_tolerance[0] or goal_error[1] > goal_tolerance[1]
                ):
                    return {
                        "status": f"screw plan failed: converged {goal_error[0] * 100:.1f} cm / "
                                  f"{np.degrees(goal_error[1]):.1f} deg from the goal after "
                                  f"{iterations} step(s)",
                        "goal_error": goal_error,
                        "iterations": iterations,
                    }
                if verbose:
                    ta.setup_logging("INFO")
                else:
                    ta.setup_logging("WARNING")
                knots = np.vstack(path)
                cols = self._root_cols()
                if cols is not None:  # K53: back into the simulator's root frame before timing
                    knots[:, cols] = self.planning_world.unfold_qpos(knots[:, cols], ref_yaw=ref_yaw)
                times, pos, vel, acc, duration = self.TOPP(knots, time_step)
                return {
                    "status": "Success",
                    "time": times,
                    "position": pos,
                    "velocity": vel,
                    "acceleration": acc,
                    "duration": duration,
                    "goal_error": goal_error,
                    "iterations": iterations,
                }

            if iterations >= max_iters:
                return {
                    "status": f"screw plan failed: no convergence after {iterations} step(s), "
                              f"{np.linalg.norm(omega):.3f} of the twist left"
                }


    def plan_pose(
        self,
        goal_pose: mplib.Pose,
        current_qpos: np.ndarray,
        mask: Optional[list[bool] | np.ndarray] = None,
        *,
        time_step: float = 0.1,
        rrt_range: float = 0.1,
        planning_time: float = 1,
        fix_joint_limits: bool = True,
        fixed_joint_indices: Optional[list[int]] = None,
        wrt_world: bool = True,
        simplify: bool = True,
        constraint_function: Optional[Callable] = None,
        constraint_jacobian: Optional[Callable] = None,
        constraint_tolerance: float = 1e-3,
        verbose: bool = False,
        n_init_qpos: int = 20
    ) -> dict[str, str | np.ndarray | np.float64]:
        """
        plan from a start configuration to a goal pose of the end-effector

        Args:
            goal_pose: pose of the goal
            current_qpos: current joint configuration — the FULL simulator qpos (all
                joints, root joints first); the clip against ``joint_limits`` needs
                the full length
            mask: if the value at a given index is True, the joint is *not* used in the
                IK
            time_step: time step for TOPPRA (time parameterization of path)
            rrt_range: step size for RRT
            planning_time: time limit for RRT
            fix_joint_limits: if True, will clip the joint configuration to be within
                the joint limits
            wrt_world: if true, interpret the target pose with respect to
                the world frame instead of the base frame
            verbose: if True, will print the log of OMPL and TOPPRA
        """
        if mask is None:
            mask = []

        if fix_joint_limits:
            current_qpos = np.clip(
                current_qpos, self.joint_limits[:, 0], self.joint_limits[:, 1]
            )
        current_qpos = self.pad_move_group_qpos(current_qpos)
        # K53: into the planning model's frame (see plan_screw); `plan_qpos` unfolds
        # the RRT path before timing it.
        ref_yaw = float(current_qpos[2]) if len(current_qpos) > 2 else 0.0
        current_qpos = self.fold_qpos(current_qpos)

        if wrt_world:
            goal_pose = self._transform_goal_to_wrt_base(goal_pose)

        # we need to take only the move_group joints when planning
        # idx = self.move_group_joint_indices

        ik_status, goal_qpos = self.IK(goal_pose, current_qpos, mask, n_init_qpos=n_init_qpos, verbose=True)
        if ik_status != "Success":
            return {"status": ik_status}

        if verbose:
            print("IK results:")
            for i in range(len(goal_qpos)):  # type: ignore
                print(goal_qpos[i])  # type: ignore

        return self.plan_qpos(
            goal_qpos,  # type: ignore
            current_qpos,
            time_step=time_step,
            rrt_range=rrt_range,
            planning_time=planning_time,
            fix_joint_limits=fix_joint_limits,
            fixed_joint_indices=fixed_joint_indices,
            simplify=simplify,
            constraint_function=constraint_function,
            constraint_jacobian=constraint_jacobian,
            constraint_tolerance=constraint_tolerance,
            verbose=verbose,
            ref_yaw=ref_yaw,
        )
