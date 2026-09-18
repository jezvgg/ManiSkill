"""Step accounting for the Fetch solver, and two pure helpers it decides with.

Why this exists
---------------
`MikasaFetchSolver` (extand.py) drives the env from six loops and
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

from utils.mikasa.state_diff import quat_angle_deg


def path_is_a_plan(skim: int, knots: int, cap: float) -> bool:
    """Whether a drawn path is worth executing at all, by how much of it is inside things.

    OMPL validates an edge at its endpoints, so a returned path can be "collision-free"
    and still run through the furniture between two knots; `path_env_collisions` counts
    the knots where it does. A path that is inside something for a THIRD of its length is
    not a near miss, it is a plan through the scene, and executing it is worse than
    having none: measured on SeasonDish seed 204, a draw with 98 of 191 knots intruding
    was executed because it was the only one that planned, and it shoved the bottle
    11.2 cm — after which no grasp existed at all and the episode was lost with `no plan`.

    Across the 60 HELD episodes 163 of 175 executed paths intrude at zero knots, so this
    refuses a handful of outliers and touches nothing else.

    Args:
        skim: intruding knots; a negative value means the world could not answer.
        knots: total knots on the path.
        cap: the largest intruding fraction still worth executing.

    Returns:
        True when the path may be executed.

    Example:
        >>> path_is_a_plan(0, 191, 1 / 3)      # clean
        True
        >>> path_is_a_plan(98, 191, 1 / 3)     # half of it is inside the scene
        False
        >>> path_is_a_plan(21, 172, 1 / 3)     # a graze near the goal
        True
        >>> path_is_a_plan(-1, 100, 1 / 3)     # no opinion is not an accusation
        True
    """
    if int(skim) <= 0 or int(knots) <= 0:
        return True
    return (int(skim) / int(knots)) <= float(cap)


def draw_rank(skim: int, knots: int) -> tuple:
    """Order two RRT draws of the same goal: clean first, then short.

    RRTConnect is randomized, so drawing the same reachable goal twice gives paths of
    wildly different length AND different clearance. Ranking by length alone picks the
    most DIRECT path, and the most direct path hugs the obstacles it has to avoid —
    the planner's "collision-free" is a claim about edge endpoints, so a path that
    passes that test can still sweep an object between two knots.

    Measured on SeasonDish seed 6 and reproduced in three pools: ranked by length the
    arm takes a 112-knot approach, the fingers close on nothing, and the bottle is
    18.8 cm away — the approach itself swept it. Ranked clean-first the same goal takes
    the 146-knot draw that `main` took, and the grasp holds.

    Args:
        skim: knots at which the commanded path intrudes on the environment.
        knots: length of the path, in knots.

    Returns:
        a tuple that sorts ascending: prefer no intrusion, then least intrusion, then
        fewest knots.

    Example:
        >>> draw_rank(0, 146) < draw_rank(3, 112)      # clean beats short
        True
        >>> draw_rank(0, 112) < draw_rank(0, 146)      # among clean, short wins
        True
        >>> draw_rank(1, 200) < draw_rank(4, 20)       # among dirty, least skim wins
        True
    """
    return (1 if int(skim) > 0 else 0, max(0, int(skim)), int(knots))


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
        #: When a list, every action stepped through here is appended to it. This is
        #: the ONE place the solver's primitives reach the env, which is why the tape
        #: lives here and not on a wrapper around `solver.env`: `StepGuard` captured
        #: its own reference at construction, so wrapping the solver's attribute
        #: records nothing (measured the hard way, 2026-09-04).
        self.tape = None
        self.elapsed_steps = 0
        self.truncated = False
        self.terminated = False
        self.last_step = None

    def step(self, action, tape_entry=None):
        """`env.step(action)`, counted and latched. Returns the 5-tuple untouched.

        When `self.tape` is a list the action is kept, so a stage can later UNDO itself
        by replaying what it did in reverse — see `MikasaFetchSolver.
        start_tape`. Off by default and free when off. `tape_entry`, when given, is
        what goes on the tape INSTEAD of the raw action: the Fetch solver hands the
        absolute joint targets it composed the action from, so a tape reads the same
        in `pd_joint_pos` (where the action IS the targets) and in `pd_joint_delta_pos`
        (where reversing the raw increments would not retrace anything)."""
        if self.tape is not None:
            self.tape.append(np.asarray(action if tape_entry is None else tape_entry,
                                        dtype=np.float64).copy())
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

    The four-deque test of `MikasaFetchSolver.follow_forward_path_w_refinement`,
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


def report(env, stage: str, fields: dict, to_trace: bool = True) -> str:
    """One diagnostic line on stdout; the same line as a `solver` event when asked.

    The solver reports every executed plan here (`extand.MikasaFetchSolver
    ._report`). Two channels, and they are deliberately separable:

    * **stdout, always.** A line per plan is what makes a run readable while it runs,
      and it costs nothing anyone is measuring.
    * **`events.jsonl`, only when `to_trace`.** The trace is compared byte for byte
      across re-runs — that is how this branch proves a change was a no-op for the
      other tasks' published numbers — so a line that would be added to every run of
      every task is a line that invalidates every recorded trace. An instrument that
      fires rarely prints always and traces only when it fires.

    Args:
        env: the (possibly wrapped) env; `find_log_event` walks it for a logger.
        stage: the primitive's name, printed as `[stage]`.
        fields: the diagnostic fields, printed as `k=v` in order and passed through
            to the event as keyword fields.
        to_trace: write the event too (default True — the existing behaviour of every
            caller that does not pass it).

    Returns:
        The printed line.

    Example:
        >>> class Logger:
        ...     def __init__(self): self.seen = []
        ...     def log_event(self, event, message="", **extra): self.seen.append(extra)
        >>> log = Logger()
        >>> report(log, "rotate_base_z", {"short_way": 0.56, "chosen": 0.56})
        [rotate_base_z] short_way=0.56 chosen=0.56
        '[rotate_base_z] short_way=0.56 chosen=0.56'
        >>> [e["stage"] for e in log.seen]
        ['rotate_base_z']
        >>> _ = report(log, "rotate_base_z", {"jammed": False}, to_trace=False)
        [rotate_base_z] jammed=False
        >>> len(log.seen)                     # printed, not traced
        1
    """
    line = f"[{stage}] " + " ".join(f"{k}={v}" for k, v in fields.items())
    print(line, flush=True)
    if to_trace:
        log_event = find_log_event(env)
        if log_event is not None:
            log_event("solver", line, stage=stage, **fields)
    return line
