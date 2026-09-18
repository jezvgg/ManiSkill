from mani_skill.utils.scene_builder.robocasa.scene_builder import ROBOT_FRONT_FACING_SIZE

from .ds_fetch import MikasaDSFetch


def _register_front_facing_size():
    """Tell RoboCasa how far to stand ds_fetch back from a fixture.

    `ROBOT_FRONT_FACING_SIZE` (scene_builder.py:99) is keyed on
    `env.agent.robot.name`, and `robot.name` is the agent's uid because
    `base_agent.py:171` does `loader.name = self.uid`. So `"mikasa_ds_fetch"` misses the
    dict, `compute_robot_base_placement_pose` logs a warning and falls back to
    0.7 m (:731-737) — 10 cm closer to the counter than Fetch is meant to stand.

    setdefault, not assignment: if a future ManiSkill ships its own entry, it wins.

    This moves MyRoboCasa-v1 and MyRoboCasa_TakeItBack-v1 too. That is a fix, but
    it is a behaviour change — run colab.run_cup_task before and after so any shift
    in success rate is attributed to this and not to something else.
    """
    ROBOT_FRONT_FACING_SIZE.setdefault("mikasa_ds_fetch", 0.8)


_register_front_facing_size()

__all__ = ["MikasaDSFetch"]
