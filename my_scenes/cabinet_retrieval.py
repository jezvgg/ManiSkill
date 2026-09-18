"""MikasaCabinetRetrieval-v0: take the cup out of the OPENED wall cabinet.

The whole task lives in `cabinet_retrieval_base.py` (shared with the closed-door
variant); this file is the v0 registration — door open at 1.6 rad from reset,
the measured W12/W13 configuration with the 9/10 oracle.
"""
from mani_skill.utils.registration import register_env

from utils.mikasa.scenes.cabinet_retrieval_base import (  # noqa: F401  (re-exported names)
    INSTRUCTIONS,
    TASK_STATE_KEYS,
    CabinetRetrievalConfig,
    CabinetRetrievalTaskBase,
)


@register_env(
    "MikasaCabinetRetrieval-v0",
    max_episode_steps=CabinetRetrievalConfig.horizon,
    asset_download_ids=["RoboCasa"],
)
class CabinetRetrievalTask(CabinetRetrievalTaskBase):
    """Take the cup out of the opened wall cabinet and stand it on the counter."""

    cfg = CabinetRetrievalConfig()
