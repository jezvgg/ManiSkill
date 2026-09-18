"""The action that keeps the robot exactly where it is, and a loop that repeats it.

Lifted out of `diagnose_task.py`, which is where it was written and where it is now
imported from, because three unrelated things need it: the diagnostics, the state
diff (`utils.mikasa/state_diff.py`) and any oracle for a memory task, which has to
sit still through the cue phase rather than act during it (`docs/writing-tasks.md`,
the memory checklist).

Nothing here builds an env or imports a simulator at module level, so it can be
imported and read on a laptop.
"""

from __future__ import annotations

import numpy as np


def hold_action(env):
    """The action that tells the robot to stay exactly where it is.

    Zero is not that action. Under `pd_joint_pos` the arm and body controllers take
    **absolute** joint targets with `normalize_action=False` (ds_fetch.py:81-89,
    191-201), so a zero action commands qpos 0 and drops the torso through the
    floor. The identity action is the current qpos of each position-controlled
    joint, and zero only for the velocity-controlled base.

    Built from the controller's own active-joint indices rather than a hardcoded
    7+1+3+2 split, and checked against the action space, so a controller change
    fails loudly here instead of quietly steering the robot somewhere.

    Args:
        env: a `gym.Env` wrapping a ManiSkill `BaseEnv` (or the base env itself);
            only `env.unwrapped.agent` and `env.unwrapped.action_space` are used.
            Any `num_envs` works — the returned action is batched to match qpos.

    Returns:
        torch.Tensor of shape `(num_envs, action_dim)`, dtype float32, on the same
        device as `agent.robot.get_qpos()`. Feed it straight to `env.step`.

    Raises:
        AssertionError: if the assembled action does not match
            `env.action_space.shape[-1]` — meaning the controller layout changed
            and this function, not the caller, is what needs fixing.

    Example:
        >>> import gymnasium as gym, utils.mikasa  # doctest: +SKIP
        >>> env = gym.make("MyRoboCasa-v1", num_envs=1, robot_uids="mikasa_ds_fetch",
        ...                control_mode="pd_joint_pos")  # doctest: +SKIP
        >>> env.reset(seed=0)  # doctest: +SKIP
        >>> a = hold_action(env)  # doctest: +SKIP
        >>> a.shape[-1] == env.action_space.shape[-1]  # doctest: +SKIP
        True
    """
    import torch

    base_env = env.unwrapped
    agent = base_env.agent
    qpos = agent.robot.get_qpos()
    parts = []
    for name, ctrl in agent.controller.controllers.items():
        dim = int(np.prod(ctrl.single_action_space.shape))
        if "base" in name or "vel" in type(ctrl).__name__.lower():
            parts.append(torch.zeros((qpos.shape[0], dim), device=qpos.device))
        elif getattr(ctrl.config, "use_delta", False):
            # a delta controller: "stay" is a zero increment from the measured pose
            parts.append(torch.zeros((qpos.shape[0], dim), device=qpos.device))
        else:
            # A mimic gripper drives two joints from one number, so take the first
            # `dim` of its active joints rather than assuming one-to-one.
            idx = list(ctrl.active_joint_indices)[:dim]
            parts.append(qpos[:, idx])
    action = torch.cat(parts, dim=1)
    expected = base_env.action_space.shape[-1]
    assert action.shape[-1] == expected, (
        f"hold action is {action.shape[-1]}-dim, action space wants {expected}. "
        "The controller layout changed; fix hold_action rather than padding."
    )
    return action


def hold_still(env, n_steps: int):
    """Step the hold action `n_steps` times and return the last transition.

    The hold action is computed **once**, before the first step, and reused. That is
    deliberate and it is the difference between "stay put" and "drift": recomputing
    it every step feeds back whatever error the controller has accumulated, so each
    step re-targets the joints at where they *ended up* rather than where they were
    told to be, and the arm sags a little further every time. One fixed target is
    also what a real wait-for-the-cue phase looks like.

    Nothing here decides success and nothing calls `evaluate()`: the `info` returned
    is the one the env handed back from `step`, which is the only one whose
    `elapsed_steps` is the step you are actually on. Calling `env.evaluate()` out of
    band advances latches in tasks that hold one (`season_dish.py`,
    `station_checklist.py:565`), so read `info`, never re-evaluate.

    Args:
        env: a `gym.Env` wrapping a ManiSkill `BaseEnv`, already `reset`.
        n_steps: how many control steps to hold for. `<= 0` steps nothing.

    Returns:
        The gym 5-tuple `(obs, reward, terminated, truncated, info)` from the final
        step, or `None` when `n_steps <= 0` (nothing was stepped, so there is no
        transition to report — an all-zeros tuple would be a lie).

    Example:
        >>> import gymnasium as gym, utils.mikasa  # doctest: +SKIP
        >>> env = gym.make("MikasaSeasonDish-v0", num_envs=1, robot_uids="mikasa_ds_fetch",
        ...                control_mode="pd_joint_pos", obs_mode="state")  # doctest: +SKIP
        >>> obs, info = env.reset(seed=0)  # doctest: +SKIP
        >>> obs, rew, term, trunc, info = hold_still(env, 30)  # wait out the cue
        ... # doctest: +SKIP
        >>> bool(info["cue_visible"][0])  # doctest: +SKIP
        False
    """
    if n_steps <= 0:
        return None
    action = hold_action(env)
    result = None
    for _ in range(n_steps):
        result = env.step(action)
    return result
