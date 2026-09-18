"""MikasaCabinetSearch-v0: find the cube hidden in one of the wall cabinets —
close behind you, home between openings, never open a compartment twice.

The whole task lives in `cabinet_search_base.py` (shared machinery, registers
nothing); this file is the v0 registration: N = 4 compartments on kitchen 102 —
both wall boxes, `cab_2` and `cab_main`, each split into its left and right half
by a kinematic partition at the box centre (`DEFAULT_COMPARTMENTS`) — success =
reveal, the found cabinet not closed.
"""
from mani_skill.utils.registration import register_env

from utils.mikasa.scenes.cabinet_search_base import (  # noqa: F401  (re-exported names)
    DEFAULT_COMPARTMENTS,
    INSTRUCTIONS,
    INSTRUCTIONS_NUDGE,
    TASK_STATE_KEYS,
    UNTANGLE_BITS,
    UNTANGLE_DOORLESS,
    CabinetSearchConfig,
    CabinetSearchTaskBase,
    Compartment,
    SearchLatches,
    SearchLayout,
    at_home_predicate,
    compartment_layout,
    hinge_theta,
    memoryless_search_floor,
    search_success_with_memory,
    self_marking_search_floor,
    step_search_latches,
)

__all__ = [
    "DEFAULT_COMPARTMENTS", "INSTRUCTIONS", "INSTRUCTIONS_NUDGE", "TASK_STATE_KEYS",
    "UNTANGLE_BITS", "UNTANGLE_DOORLESS",
    "CabinetSearchConfig", "CabinetSearchTask", "CabinetSearchTaskBase",
    "Compartment", "SearchLatches", "SearchLayout",
    "at_home_predicate", "compartment_layout", "hinge_theta",
    "memoryless_search_floor", "search_success_with_memory",
    "self_marking_search_floor", "step_search_latches",
]


@register_env(
    "MikasaCabinetSearch-v0",
    max_episode_steps=CabinetSearchConfig.horizon,
    asset_download_ids=["RoboCasa"],
)
class CabinetSearchTask(CabinetSearchTaskBase):
    """Find the hidden cube across four compartments without repeating one."""

    cfg = CabinetSearchConfig()
