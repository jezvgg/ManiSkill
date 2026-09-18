"""Beacons: actors whose appearance switches without their geometry ever moving.

A memory task needs a cue that can be taken away. The obvious ways of taking one
away are all wrong here:

- deleting the actor changes shadows and occlusion, and the change gives the answer
  away to anything looking at the floor instead of the beacon;
- `_hidden_objects` is implemented on GPU by moving the actor 99999 m away
  (`mani_skill/utils/structs/actor.py:187-194`) — that is a geometry change wearing
  a disguise;
- moving the beacon out of frame is the same thing, said honestly.

What does work, verified by running it on CPU mid-episode: mutate the
`RenderMaterial`. `set_emission` and `set_base_color` take effect at the next
render, the geometry is untouched, and switching back reproduces the frame
bit-for-bit.

Two facts make the implementation less obvious than the API suggests:

1. **One material is shared by every parallel environment.** `ActorBuilder` reuses
   the same `RenderMaterial` object for every sub-scene it builds
   (`mani_skill/utils/building/actor_builder.py:234-245`), so mutating it changes
   the beacon in all of them at once. Independent per-env answers therefore need
   one actor *and one material* per (env, beacon). This module builds them that way
   and hands back the material handles.
2. **Static, not kinematic.** `Actor.pose`'s setter asserts a non-static body under
   GPU sim (`structs/actor.py:369-371`), so building the beacons static makes "the
   beacon never moves" a property of the type rather than a rule someone has to
   remember. They carry no collision either — they are pure signal.
"""

from __future__ import annotations

import numpy as np
import sapien
import sapien.render


class BeaconGrid:
    """`n_envs × n_beacons` beacons, each independently switchable.

    Built once in `_load_scene`; after that only `set_lit` is ever called, and it
    touches nothing but material parameters.
    """

    def __init__(
        self,
        env,
        positions: np.ndarray,
        radius: float,
        on_color=(1.0, 0.85, 0.15, 1.0),
        on_emission=(1.0, 0.85, 0.15, 6.0),
        off_color=(0.22, 0.22, 0.24, 1.0),
        off_emission=(0.0, 0.0, 0.0, 1.0),
        name: str = "beacon",
    ):
        """positions: `(n_envs, n_beacons, 3)` world positions, one beacon each."""
        positions = np.asarray(positions, dtype=np.float32)
        assert positions.ndim == 3 and positions.shape[-1] == 3, positions.shape
        self.n_envs, self.n_beacons = positions.shape[:2]

        self.on_color = list(on_color)
        self.on_emission = list(on_emission)
        self.off_color = list(off_color)
        self.off_emission = list(off_emission)

        # [env][beacon] -> RenderMaterial. Not a tensor: these are render handles,
        # they never take part in any computation.
        self._materials: list[list[sapien.render.RenderMaterial]] = []
        self.actors: list[list] = []

        for i in range(self.n_envs):
            row_mats, row_actors = [], []
            for s in range(self.n_beacons):
                # A fresh material per beacon per env — see note 1 in the module
                # docstring. Sharing one would make every environment answer alike.
                mat = sapien.render.RenderMaterial(
                    base_color=self.on_color, emission=self.on_emission
                )
                builder = env.scene.create_actor_builder()
                builder.add_sphere_visual(radius=radius, material=mat)
                builder.set_scene_idxs([i])
                builder.initial_pose = sapien.Pose(p=positions[i, s])
                # Name carries both indices: actor names are dict keys in
                # `scene.actors` (actor_builder.py:259) and a duplicate silently
                # replaces the earlier entry.
                row_actors.append(builder.build_static(name=f"{name}_s{s}_env{i}"))
                row_mats.append(mat)
            self._materials.append(row_mats)
            self.actors.append(row_actors)

    def set_lit(self, lit) -> None:
        """Apply a `(n_envs, n_beacons)` boolean pattern of lit/dark.

        Writes both emission and base colour. Emission alone leaves the beacon a
        bright yellow ball in ambient light — a residual glow, which the spec calls
        out as a leak channel of its own.
        """
        lit = np.asarray(lit)
        assert lit.shape == (self.n_envs, self.n_beacons), lit.shape
        for i in range(self.n_envs):
            for s in range(self.n_beacons):
                mat = self._materials[i][s]
                if lit[i, s]:
                    mat.set_base_color(self.on_color)
                    mat.set_emission(self.on_emission)
                else:
                    mat.set_base_color(self.off_color)
                    mat.set_emission(self.off_emission)

    def material(self, env_i: int, beacon_s: int) -> sapien.render.RenderMaterial:
        """Handle for tests and diagnostics."""
        return self._materials[env_i][beacon_s]
