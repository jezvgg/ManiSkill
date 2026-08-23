"""Step accounting for the Fetch solver, and two pure helpers it decides with.

Why this exists
---------------
`FetchMotionPlanningSapienSolver` (extand.py) drives the env from six loops and
none of them looked at `truncated`: the burner oracle's first emulated run went to
4935 control steps in an episode whose horizon is 400 (docs/lab-journal.md,
2026-08-17). `StepGuard` is the one place every `env.step` now goes through — it
counts, it latches `truncated`, and it keeps the last 5-tuple so a primitive that
is called after the horizon can return it instead of stepping again.

Nothing here imports mplib, so the guard and the two helpers can be unit-tested on a
Mac against a fake env; the solver that uses them cannot. No wall-clock either — a
hang inside mplib is caught outside the process (`tools/docker/planner.sh run
--timeout N`), not by the solver.

Example:
    >>> class Env:  # a fake that truncates on its 3rd step
    ...     n = 0
    ...     def step(self, action):
    ...         self.n += 1
    ...         return None, 0.0, False, self.n >= 3, {"success": False}
    >>> g = StepGuard(Env())
    >>> for _ in range(3):
    ...     _ = g.step(None)
    >>> g.elapsed_steps, g.truncated
    (3, True)
    >>> refine_should_stop(201, 200, [0.1] * 5, [0.0] * 5, [0.5] * 5, [0.0] * 5)
    'stuck'
    >>> round(pose_error([0, 0, 0], [1, 0, 0, 0], [0.02, 0, 0], [1, 0, 0, 0])[0], 3)
    0.02
"""

from __future__ import annotations

import math

import numpy as np



# --- vendored from mikasa-bench (mikasa_bench/state_diff.py) ------------------
# Ten lines of pure math, inlined so this module has no dependency outside this
# package. Kept identical to the original so a future diff against it is trivial.
def quat_angle_deg(qa, qb):
    """Angle in degrees between two `[w, x, y, z]` quaternions, or None.

    `2 * acos(|qa . qb|)`, with the absolute value because `q` and `-q` are the same
    rotation — without it a settled object reads as having spun 360 degrees the
    moment the solver flips the sign, which is the single most common false finding
    in a naive pose diff.

    Args:
        qa, qb: sequences of 4 floats, `[w, x, y, z]`, need not be normalised.

    Returns:
        float degrees in [0, 180], or None if either input is not a 4-vector or is
        the zero quaternion.

    Example:
        >>> round(quat_angle_deg([1, 0, 0, 0], [0.70710678, 0, 0, 0.70710678]), 3)
        90.0
        >>> quat_angle_deg([1, 0, 0, 0], [-1, 0, 0, 0])  # same rotation, flipped sign
        0.0
        >>> quat_angle_deg([1, 0, 0, 0], None) is None
        True
    """
    if qa is None or qb is None or len(qa) != 4 or len(qb) != 4:
        return None
    na = math.sqrt(sum(float(c) ** 2 for c in qa))
    nb = math.sqrt(sum(float(c) ** 2 for c in qb))
    if na == 0.0 or nb == 0.0:
        return None
    dot = sum(float(x) * float(y) for x, y in zip(qa, qb)) / (na * nb)
    return math.degrees(2.0 * math.acos(min(1.0, abs(dot))))


def _any_true(flag) -> bool:
    """`bool()` of a step flag whether it is a Python bool, a numpy array or a torch
    tensor (batched, possibly on a GPU): `torch.as_tensor(x).any().item()`."""
    if hasattr(flag, "any") and hasattr(flag, "item"):
        import torch

        return bool(torch.as_tensor(flag).any().item())
    return bool(flag)


class StepGuard:
    """Wraps `env.step`: counts steps, latches `truncated`, keeps the last 5-tuple.

    `truncated` is a latch — once any env in the batch reports it, it stays True
    for the life of the guard (the solver is per-episode; a new episode gets a new
    solver). `terminated` is recorded but is *not* a stop condition: the memory
    tasks set `terminated &= ~success`-style flags that the oracle must not act on,
    and the sweep decides success from `info`, not from the flag.

    Args:
        env: anything with `step(action) -> (obs, reward, terminated, truncated, info)`
            — the raw env, or the `PlannerLogger` wrapper the sweep hands the planner.

    Attributes:
        elapsed_steps (int): number of `step` calls so far.
        truncated (bool): latched — True from the first step whose `truncated` was.
        terminated (bool): the last step's flag, reduced with `.any()`.
        last_step (tuple | None): the last 5-tuple returned, None before any step.

    Example:
        >>> class Env:
        ...     def __init__(self, at): self.at, self.n = at, 0
        ...     def step(self, a):
        ...         self.n += 1
        ...         return None, 0.0, False, self.n >= self.at, {}
        >>> g = StepGuard(Env(at=5))
        >>> [g.step(None)[3] for _ in range(6)]
        [False, False, False, False, True, True]
        >>> g.elapsed_steps, g.truncated, g.last_step[3]
        (6, True, True)
    """

    def __init__(self, env):
        self.env = env
        self.elapsed_steps = 0
        self.truncated = False
        self.terminated = False
        self.last_step = None

    def step(self, action):
        """`env.step(action)`, counted and latched. Returns the 5-tuple untouched."""
        out = self.env.step(action)
        obs, reward, terminated, truncated, info = out
        self.elapsed_steps += 1
        self.terminated = _any_true(terminated)
        if _any_true(truncated):
            self.truncated = True
        self.last_step = out
        return out


def refine_should_stop(passed, max_steps, lift_pos, lift_vel, base_pos, base_vel):
    """Why the solver's refinement loop should stop, or None to keep going.

    The four-deque test of `FetchMotionPlanningSapienSolver.follow_forward_path_w_refinement`,
    lifted verbatim so it can be tested without a robot: "stuck" when every deque has
    more than 4 samples and a standard deviation under 1e-3 (the torso and the base
    have not moved and are not moving), else "max" when `passed` is strictly greater
    than `max_steps`, else None. Order matters and is the original's: a stuck robot
    is reported as stuck even on the step it would also hit the cap.

    Args:
        passed: refinement steps taken so far.
        max_steps: the cap (`max_refine_steps` of the solver).
        lift_pos, lift_vel, base_pos, base_vel: the last ≤10 samples of the torso
            lift qpos/qvel and the base x qpos/qvel (any sequence of floats).

    Returns:
        "stuck" | "max" | None

    Example:
        >>> flat, moving = [0.3] * 5, [0.1, 0.2, 0.3, 0.4, 0.5]
        >>> refine_should_stop(10, 200, flat, flat, flat, flat)
        'stuck'
        >>> refine_should_stop(201, 200, moving, flat, flat, flat)
        'max'
        >>> refine_should_stop(200, 200, moving, flat, flat, flat) is None
        True
    """
    settled = all(
        len(samples) > 4 and np.std(samples) < 1e-3
        for samples in (lift_vel, base_vel, lift_pos, base_pos)
    )
    if settled:
        return "stuck"
    if passed > max_steps:
        return "max"
    return None


def pose_error(p_goal, q_goal, p_ee, q_ee):
    """`(metres, radians)` between a goal pose and an end-effector pose, same frame.

    Position is the Euclidean distance; the angle is `state_diff.quat_angle_deg`
    converted to radians, so `q` and `-q` compare as the same rotation
    (`[w, x, y, z]` quaternions, as mplib and sapien hand them out).

    Args:
        p_goal, p_ee: 3-vectors.
        q_goal, q_ee: `[w, x, y, z]` quaternions.

    Returns:
        tuple[float, float]: `(pos_m, rot_rad)`.

    Example:
        >>> pos, rot = pose_error([0, 0, 0], [1, 0, 0, 0], [0.02, 0, 0], [1, 0, 0, 0])
        >>> round(pos, 3), round(rot, 3)
        (0.02, 0.0)
        >>> pose_error([0, 0, 0], [1, 0, 0, 0], [0, 0, 0], [-1, 0, 0, 0])[1]  # -q is q
        0.0
    """
    deg = quat_angle_deg(q_goal, q_ee)
    if deg is None:
        raise ValueError(f"quaternions must be 4-vectors, got {q_goal!r} and {q_ee!r}")
    pos = float(np.linalg.norm(np.asarray(p_goal, dtype=float) - np.asarray(p_ee, dtype=float)))
    return pos, math.radians(deg)


def find_log_event(env, max_depth: int = 16):
    """The `log_event` of the nearest wrapper in `env`'s chain that has one, else None.

    The solver's diagnostic lines (`_report`) go to `events.jsonl` when a
    `PlannerLogger` is somewhere around the env. Finding it has two traps, which is
    why this is a walk and not a `getattr`:

    * `getattr(env, "log_event", None)` on a gymnasium 0.29 wrapper *succeeds* by
      forwarding to the inner env — with a deprecation warning per call, which is
      noise in every run and a warning the container gate counts.
    * `hasattr(type(env), "log_event")` on the outermost class alone silently loses
      every `solver` event as soon as the logger is not the outermost wrapper, which
      is exactly what T2's `RecordEpisode` around it does.

    So: look on each wrapper's *class*, and descend by reading `env` out of the
    instance `__dict__` (where `gymnasium.Wrapper.__init__` puts it), never by
    attribute access that `__getattr__` could answer.

    Args:
        env: the env, wrapped or not.
        max_depth: give up after this many wrappers (a deeper chain is a bug).

    Returns:
        The bound `log_event` method, or None if no wrapper in the chain has one.

    Example:
        >>> class Logger:                       # a PlannerLogger stand-in
        ...     def __init__(self, env): self.env = env
        ...     def log_event(self, event, message="", **extra): return (event, message)
        >>> class Recorder:                     # RecordEpisode around it (T2)
        ...     def __init__(self, env): self.env = env
        >>> find_log_event(Recorder(Logger(object())))("solver", "hello")
        ('solver', 'hello')
        >>> find_log_event(object()) is None
        True
    """
    node = env
    for _ in range(max_depth):
        if node is None:
            return None
        if hasattr(type(node), "log_event"):
            return node.log_event
        node = node.__dict__.get("env") if hasattr(node, "__dict__") else None
    return None
