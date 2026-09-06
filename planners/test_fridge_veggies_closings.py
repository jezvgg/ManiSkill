"""CPU-only check: uv run python -m planners.test_fridge_veggies_closings."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from planners import myrobocasa_fridge_veggies_planner as planner_module


def test_closing_order():
    veg = Mock(pose=SimpleNamespace(p=torch.zeros((1, 3))))
    veg.get_first_collision_mesh.return_value = None
    env = Mock(unwrapped=SimpleNamespace(
        veggies=[veg], _picture_target=0, control_timestep=0.05
    ))
    planner = Mock()
    planner.robot.get_qpos.return_value = torch.zeros((1, 15))
    env.log_motion.side_effect = RuntimeError("stop before execution")

    def pose_for_closing(agent, center, closing, *args):
        pose = planner_module.sapien.Pose(p=closing)
        return pose, pose

    long = {"status": "Success", "position": [[0, 0], [3, 4]]}
    short = {"status": "Success", "position": [[0, 0], [0, 1]]}
    failed = {"status": "screw plan failed"}
    with patch.object(planner_module, "_veg_world_name", return_value="target"), \
         patch.object(planner_module, "_top_down_grasp_pose", pose_for_closing):
        for ranked, paths, expected in (
            (True, [long, short], [0, 1, 0]),
            (True, [failed, short], [0, 1, 0]),
            (True, [failed, failed], [1, 0, 0]),
            (False, [], [1, 0, 0]),
        ):
            planner.planner.plan_screw.reset_mock(side_effect=True)
            planner.planner.plan_screw.side_effect = paths
            try:
                planner_module._reach_and_grasp(
                    env, planner, Mock(), rank_closings=ranked
                )
            except RuntimeError as exc:
                assert str(exc) == "stop before execution"
            else:
                raise AssertionError("Reach was not attempted")
            np.testing.assert_allclose(env.log_motion.call_args.args[4].p, expected)
            assert planner.planner.plan_screw.call_count == (2 if ranked else 0)
            for call in planner.planner.plan_screw.call_args_list:
                np.testing.assert_array_equal(call.args[1], np.zeros(15))
                np.testing.assert_array_equal(
                    call.kwargs["masked_joints"], [False] * 3 + [True] * 12
                )


if __name__ == "__main__":
    test_closing_order()
    print("closing order check passed")
