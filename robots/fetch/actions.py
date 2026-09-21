"""Fetch action contract shared by planners and task-level helpers."""

import numpy as np


def base_cmd(forward: float = 0.0, yaw: float = 0.0) -> np.ndarray:
    """Return the nonholonomic Fetch base command ``[forward, yaw]``."""
    return np.array([forward, yaw])
