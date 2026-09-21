import torch

from mani_skill.utils.registration import register_env

from .my_robocasa_takeitback import MyRoboCasaSceneTakeItBack


@register_env("MyRoboCasa_TakeItBackTray-v1", asset_download_ids=["RoboCasa"])
class MyRoboCasaSceneTakeItBackTray(MyRoboCasaSceneTakeItBack):
    """Place a randomly positioned cup onto a randomly positioned baking tray."""

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
        # Trajectory collection records robot-mounted cameras only.
        return []
