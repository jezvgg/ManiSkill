import numpy as np

from mani_skill.agents.controllers import PDBaseForwardVelControllerConfig
from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.ds_fetch import DSFetch

ACTION_DIM = 13


def _base_cmd(forward=0.0, yaw=0.0):
    """Canonical Fetch base action: forward velocity and yaw velocity."""
    return np.array([forward, yaw])


@register_agent()
class CanonicalDSFetch(DSFetch):
    """DSFetch with native 13D actions and no lateral base command."""

    uid = "ds_fetch_canonical"

    @property
    def _controller_configs(self):
        configs = super()._controller_configs
        for config in configs.values():
            old = config["base"]
            config["base"] = PDBaseForwardVelControllerConfig(
                self.base_joint_names,
                lower=[old.lower[0], old.lower[-1]],
                upper=[old.upper[0], old.upper[-1]],
                damping=old.damping,
                force_limit=old.force_limit,
                friction=old.friction,
                normalize_action=old.normalize_action,
                drive_mode=old.drive_mode,
            )
        for mode in (
            "pd_joint_delta_pos",
            "pd_joint_target_delta_pos",
            "pd_joint_delta_pos_stiff_body",
        ):
            configs[mode]["arm"].normalize_action = False
        return configs
