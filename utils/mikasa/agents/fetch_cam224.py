"""This checkout's standard Fetch with 224 x 224 head and wrist cameras.

The class inherits Fetch's URDF, controllers, camera mounts and field of view.
Only the camera dimensions and registered UID change. It is intended for RGB
replay and policy evaluation; the MIKASA oracle uses ``mikasa_ds_fetch`` with a
separate planning URDF. Public ``ds_fetch`` is a third, incompatible controller
configuration, not an alias for the MIKASA robot.

Environment code may branch on the robot UID, so class-level inheritance alone
does not establish identical scene initialization. See ../ROBOTS.md for the
controller comparison and the RoboCasa integration limits.
"""
from dataclasses import replace

from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.fetch import Fetch

#: What the dataset holds, and what openpi's pi0.5 takes without resizing.
CAMERA_PX = 224


@register_agent()
class FetchOurCameras(Fetch):
    """`fetch`, unchanged, except that both of its cameras render at `CAMERA_PX`."""

    uid = "fetch_cam224"

    @property
    def _sensor_configs(self):
        # `CameraConfig` is a dataclass: copy each of the robot's own cameras and move the two
        # numbers. Everything else — the mount link, the pose, the fov, the near and far planes
        # — stays whatever the upstream robot declares, so this file does not go stale when it
        # changes.
        return [replace(cam, width=CAMERA_PX, height=CAMERA_PX) for cam in super()._sensor_configs]
