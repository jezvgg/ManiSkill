"""A text trace of a planner run: phase events as JSON lines, object poses as CSV.

Ported from jezvgg/ManiSkill `utils/logging_utils.py` @ `5e6c775` (jezv, 2026-08);
changes:

1. **`run_dir` is required and explicit.** Upstream derived
   `<log_dir>/<name>_<YYYYmmdd_HHMMSS>/` from the wall clock, so two runs of the
   same seed landed in different folders and no third party could name the one it
   wanted. The caller decides now — `evaluate_planner` uses
   `<--log-dir>/seed_<seed>/`, which is reproducible and diffable across runs.
2. **`track_task_actors()` added.** Upstream expected the planner itself to call
   `track_object` for each thing it cared about; our oracles do not, and the sweep
   cannot reach inside them, so the sweep registers the task's own actors instead.
3. **`finish()` split out of `close()`.** The sweep reuses one env across
   episodes, so closing the log must not close the env.
4. **`reset()` logs a `reset` event** (with the seed, when one was passed) instead
   of only zeroing the step counter.
5. **Docstrings carry `Args:` / `Returns:` / `Example:`** — CaP-X §6 ablates exactly
   this and finds that stripping the examples costs almost every model accuracy
   (`docs/coding-agent-primer.md` §6).
6. No behaviour change, but stated because upstream's docstring implied otherwise:
   **the wrapper never writes `console.log`**. That file exists only if the caller
   wraps the run in `capture_stdout(run_dir / "console.log")`.

Why this exists at all: CaP-X measures that feeding a coding agent raw RGB frames
between turns is *worse* than giving it no images (§3.3, Takeaway 3) and that a
structured **text** description of what changed beats both. An LLM debugging a
planner should therefore read this trace, not watch the mp4 —
`docs/coding-agent-primer.md` §2 and §15(b), which is the item this file closes.

Outputs
-------
Everything lands in the `run_dir` the caller names:

1. ``<name>_events.jsonl`` — one JSON object per line, flushed on write::

       {"step": 0, "time": "2026-08-16T12:00:00", "event": "episode_start",
        "message": "", "seed": 100, "scene": "MikasaSeasonDish-v0"}

   `event` is free-form; the sweep writes `episode_start`, `reset`, `error` and
   `verdict`, and a planner adds its own stage names through `log_event`.
2. ``<object>_trajectory.csv`` — ``step,x,y,z,qx,qy,qz,qw`` per registered object,
   the starting row at registration and one row every `log_freq` sim steps.
   **The CSV is x-y-z-w quaternion order**; the sim hands out `[w,x,y,z]` and this
   file reorders it, once, here.
3. ``console.log`` — only if the caller asked for it, see `capture_stdout`.

How to analyze the logs (for an LLM)
------------------------------------
1. **Did a stage fail, and which one?**
   Grep ``<name>_events.jsonl`` for ``"event": "error"``. `log_motion` writes one
   per failed stage with the stage name in `message` and the solver's own words in
   `detail` (``IK Failed``, ``not reachable``, ``plan failed (returned -1)``). The
   event's `step` says how far into the episode it happened; compare it with the
   task's phase boundaries (`cue_steps`, `delay_steps`) to tell "the oracle acted
   during the cue window" from "the oracle could not reach the object".
2. **Did the episode end the way the planner thought?**
   The last line is the sweep's `verdict` with `success`. A run whose events show
   no `error` and whose verdict is `success: false` is the interesting case: the
   plan executed and the task was still not achieved. Read the trajectories next.
3. **Physical anomalies, from the trajectories.**
   Before a grasp, `robot_tcp_trajectory.csv` should converge smoothly on the
   target object's csv — compute ``sqrt(dx^2+dy^2+dz^2)`` between the two files
   row by row (they share the `step` column). A discontinuity of several
   centimetres in one `log_freq` window is a collision, a physics blow-up or a
   teleport, not motion. A stationary object whose x/y/z drifts is slipping.
   `robot_base_trajectory.csv` jumping is the base planner, not the arm.
4. **Did the object end where the task wants it?**
   Take the last row of the object's csv and compare it against the target region
   the task class defines. That answers "did the cup land in the bowl" without
   opening a video, and it is the same number `utils.mikasa.diagnose_task` prints
   at t=0.
5. **What this cannot tell you.** Nothing here is evidence about *success* —
   success is `info["success"]` from the task's `evaluate()`, and the trace is
   deliberately not allowed to redefine it. And a pose written every `log_freq`
   steps aliases: a fast overshoot-and-return between two samples is invisible.
   Lower `log_freq` before concluding the motion was smooth.

The GPU/CPU asymmetry, because it produced a whole file of frozen poses once:
under GPU sim the torch buffers behind `pose` are only refreshed when something
fetches them, so `_write_row` calls `scene._gpu_fetch_all()` first. CPU sim reads
the live entity and needs no fetch.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# Top-level, and only because `utils.mikasa_oracle.evaluate_planner` — this module's only
# in-repo caller — already imports gymnasium at module level, so nothing new is
# pulled into a process that did not already have it. torch is NOT imported here:
# poses are read through `.cpu().numpy()` on whatever the sim handed back, which
# keeps this module importable (and testable) on a machine with no simulator.
import gymnasium as gym

#: Lines the solver prints when it gives up. `\bik\b` is separate because "IK" is
#: also printed on success ("IK results", "IK solution"), and those must not count.
_FAILURE_RE = re.compile(r"fail|stuck|not reach|unreachable", re.I)
_IK_RE = re.compile(r"\bik\b", re.I)
_IK_OK_RE = re.compile(r"\b(results|solution)\b", re.I)


@contextlib.contextmanager
def capture_stdout(output_path=None):
    """Route ``sys.stdout`` to a file (default: the null device) inside the block.

    Keeps mplib/OMPL chatter off the terminal while preserving it for later
    reading. **stderr is left untouched** so real errors stay visible — that is the
    point of redirecting only one of the two.

    Args:
        output_path (str | os.PathLike | None): file to write stdout into. `None`
            discards it (``os.devnull``). The file is opened in write mode, so an
            existing log at that path is replaced.

    Yields:
        None: the block runs with `sys.stdout` redirected.

    Example:
        >>> with capture_stdout():          # discard
        ...     print("solver chatter nobody needs")
        >>> import tempfile, pathlib
        >>> tmp = pathlib.Path(tempfile.mkdtemp()) / "console.log"
        >>> with capture_stdout(tmp):
        ...     print("stage 1 done")
        >>> tmp.read_text()
        'stage 1 done\\n'
    """
    path = output_path if output_path is not None else os.devnull
    with open(path, "w", encoding="utf-8") as sink, contextlib.redirect_stdout(sink):
        yield


def _looks_like_an_actor(value) -> bool:
    """True for a mani_skill `Actor`/`Link`, or anything shaped like one.

    Duck-typed rather than ``isinstance(value, Actor)`` on purpose: it keeps this
    module importable without a simulator (so the offline tests can exercise the
    scan with a stub), and it accepts `Link` too, which is what `agent.tcp` is.
    Articulations are excluded by their `get_links` method — the robot is tracked
    through its tcp and base link, and a whole articulation's "pose" is its root,
    which is not what anyone reading a trajectory wants.

    Args:
        value: any attribute value off a task instance.

    Returns:
        bool: whether `track_object` can read a pose out of it.

    Example:
        >>> from types import SimpleNamespace
        >>> pose = SimpleNamespace(p=[[0.0, 0.0, 0.0]], q=[[1.0, 0.0, 0.0, 0.0]])
        >>> _looks_like_an_actor(SimpleNamespace(name="cup", pose=pose))
        True
        >>> _looks_like_an_actor(SimpleNamespace(name="robot", pose=pose,
        ...                                      get_links=lambda: []))
        False
        >>> _looks_like_an_actor(42)
        False
    """
    try:
        if not isinstance(getattr(value, "name", None), str):
            return False
        if hasattr(value, "get_links"):  # an Articulation, not an actor
            return False
        pose = getattr(value, "pose", None)
        return hasattr(pose, "p") and hasattr(pose, "q")
    except Exception:
        # A property that raises is not an actor either. Reading arbitrary
        # attributes off a half-built task must never take the sweep down.
        return False


class PlannerLogger(gym.Wrapper):
    """Wrap an env so a planner's run leaves a readable text trace behind.

    Two files, both described at length in the module docstring: an events
    `.jsonl` (phases, errors, verdict) and one `_trajectory.csv` per tracked
    object. Wrapping is what makes the trajectories possible — the row cadence is
    counted in `step` calls, so the planner has to drive *this* object rather than
    `env.unwrapped`.

    Args:
        env (gymnasium.Env): the env to wrap. Not closed by `finish()`.
        run_dir (str | os.PathLike): directory for this run's files. Required and
            created (`parents=True, exist_ok=True`). Deliberately not derived from
            the clock: a run must be nameable by whoever wants to read it again.
        name (str): filename stem for the events log, ``<name>_events.jsonl``.
            Use the planner's name; the sweep passes the planner module's last
            component so a value like ``planners/foo`` cannot become a subdirectory.
        log_freq (int): write a trajectory row every this many `step` calls.
            Clamped to >= 1. Rows are also written at registration time.

    Attributes:
        dir (pathlib.Path): the run directory, after creation.

    Example:
        >>> import tempfile
        >>> from pathlib import Path
        >>> log = PlannerLogger(env, run_dir=Path(tempfile.mkdtemp()),
        ...                     name="season_dish_planner")   # doctest: +SKIP
        >>> log.track_object(env.unwrapped.shaker, "shaker")   # doctest: +SKIP
        >>> log.log_event("stage", "reaching for the shaker")  # doctest: +SKIP
        >>> res = log.log_motion("grasp", planner.static_manipulation, pose)  # doctest: +SKIP
        >>> log.finish()                                       # doctest: +SKIP
    """

    def __init__(self, env, run_dir, name="run", log_freq=10):
        super().__init__(env)
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.log_freq = max(1, int(log_freq or 1))
        self._step = 0
        self._events_f = open(self.dir / f"{name}_events.jsonl", "w", encoding="utf-8")
        self._objs = {}  # tracked name -> (handle, open csv file)
        self._finished = False

    # --- public API -----------------------------------------------------
    def track_object(self, handle, name):
        """Register `handle` so its pose is written to ``<name>_trajectory.csv``.

        Writes the header and the object's current pose immediately, then a row
        every `log_freq` steps. Idempotent: registering the same name twice keeps
        the first handle and the file already being written.

        Args:
            handle: anything with a batched `.pose` (`pose.p` shape `(N, 3)`,
                `pose.q` shape `(N, 4)` in `[w, x, y, z]`). Index 0 is read. A
                `.name` attribute is used to re-resolve the handle after a
                reconfiguring reset — see `_live_handle`.
            name (str): the trajectory's name, used for the filename. Keep it
                short and stable; an LLM reading the run joins files by it.

        Returns:
            pathlib.Path: the CSV being written (also on a repeat registration).

        Example:
            >>> log.track_object(env.unwrapped.agent.tcp, "robot_tcp")  # doctest: +SKIP
            PosixPath('.../seed_100/robot_tcp_trajectory.csv')
        """
        path = self.dir / f"{name}_trajectory.csv"
        if name in self._objs:
            return path
        f = open(path, "w", encoding="utf-8")
        f.write("step,x,y,z,qx,qy,qz,qw\n")
        self._objs[name] = (handle, f)
        self._write_row(name)  # starting pose
        return path

    def track_task_actors(self, task, names=None):
        """Register the task's own actors, plus the robot's tcp and base link.

        The sweep cannot reach inside an oracle to see what it manipulates, so it
        registers everything the task class holds. Only **instance** attributes are
        scanned (`vars(task)`), never `dir(task)`: evaluating arbitrary properties
        on a task mid-episode can be expensive and can raise.

        Call this after a reset, so the handles exist. If the planner then resets
        with `reconfigure=True` the handles go stale — `_live_handle` re-resolves
        them by name on every write, which is why that method exists.

        Args:
            task: the unwrapped task, i.e. `env.unwrapped`.
            names (list[str] | None): attribute names to register instead of
                scanning. Unknown names raise `AttributeError` rather than being
                skipped — a typo'd object name must not read as "nothing to track".

        Returns:
            dict[str, pathlib.Path]: registered name -> trajectory CSV. `agent.tcp`
            and `agent.base_link` are registered as `robot_tcp` and `robot_base`
            whether or not `names` was given; either is skipped if absent.

        Example:
            >>> log.track_task_actors(env.unwrapped)               # doctest: +SKIP
            {'cup': PosixPath('.../cup_trajectory.csv'), ...}
            >>> log.track_task_actors(env.unwrapped, ["shaker"])   # doctest: +SKIP
            {'shaker': ..., 'robot_tcp': ..., 'robot_base': ...}
        """
        tracked = {}
        if names is None:
            candidates = [
                (attr, value)
                for attr, value in vars(task).items()
                if not attr.startswith("_") and _looks_like_an_actor(value)
            ]
        else:
            candidates = [(attr, getattr(task, attr)) for attr in names]

        agent = getattr(task, "agent", None)
        for attr, label in (("tcp", "robot_tcp"), ("base_link", "robot_base")):
            handle = getattr(agent, attr, None) if agent is not None else None
            if handle is not None:
                candidates.append((label, handle))

        for label, handle in candidates:
            tracked[label] = self.track_object(handle, label)
        return tracked

    def log_event(self, event, message="", **extra):
        """Append one JSON line to the events log and flush it.

        Flushed on every write on purpose: the run this is meant to explain is one
        that crashed or was killed, and a buffered last line is the one you wanted.

        Args:
            event (str): the kind — `episode_start`, `reset`, `error`, `verdict`,
                or a stage name of the planner's choosing.
            message (str): free text for a human.
            **extra: any JSON-serialisable fields (`seed`, `stage`, `detail`, ...),
                merged into the record. They keep their names, so an LLM can filter
                on them.

        Returns:
            dict: the record as written, including `step` and `time`.

        Example:
            >>> log.log_event("stage", "driving to station 2", station=2)  # doctest: +SKIP
            {'step': 140, 'time': '2026-08-16T...', 'event': 'stage',
             'message': 'driving to station 2', 'station': 2}
        """
        rec = {
            "step": self._step,
            "time": datetime.now().isoformat(),
            "event": event,
            "message": message,
        }
        rec.update(extra)
        self._events_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._events_f.flush()
        return rec

    def log_motion(self, stage, fn, *args, **kwargs):
        """Run one motion/plan call and log an `error` event if it went wrong.

        Two independent failure signals, because either one alone misses cases:
        the solver *returns* `-1` when it gives up, and it *prints* why (``IK
        Failed``, ``not reachable``, ``RRT stuck``) on its way there. This captures
        stdout for the scan and then replays it to the real stdout, so a caller
        redirecting stdout into `console.log` loses nothing.

        A line counts as a failure if it matches ``fail|stuck|not reach|unreachable``
        (case-insensitive), or if it mentions `IK` without also saying `results` or
        `solution` — the solver prints "IK results" on the happy path too.

        Args:
            stage (str): what was being attempted; appears as `"<stage> failed"`.
            fn (callable): the call to make, e.g. `planner.static_manipulation`.
            *args: passed to `fn`.
            **kwargs: passed to `fn`.

        Returns:
            Whatever `fn` returned, unchanged — including `-1`. This never converts
            a failure into an exception and never decides success; the caller still
            owns `if res == -1: return res`.

        Example:
            >>> log.log_motion("grasp cup", planner.static_manipulation, pose)  # doctest: +SKIP
            -1
            >>> # ... and the events log now holds:
            >>> # {"event": "error", "message": "grasp cup failed", "detail": "IK Failed"}
        """
        real_out = sys.stdout
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = fn(*args, **kwargs)
        out = buf.getvalue()
        if out:
            real_out.write(out)
            real_out.flush()

        hits = []
        for line in out.splitlines():
            s = line.strip()
            if not s:
                continue
            if _FAILURE_RE.search(s) or (_IK_RE.search(s) and not _IK_OK_RE.search(s)):
                hits.append(s)
        if hits or (isinstance(result, (int, float)) and int(result) == -1):
            detail = " | ".join(dict.fromkeys(hits)) or f"plan failed (returned {result})"
            self.log_event("error", f"{stage} failed", detail=detail)
        return result

    def finish(self):
        """Close the log files. **Does not close the env.**

        The sweep builds one env and runs many episodes through it, each with its
        own logger, so closing the log must not take the env down with it.
        Idempotent, and `close()` calls it.

        Returns:
            pathlib.Path: the run directory, so a caller can print where it went.

        Example:
            >>> log.finish()   # doctest: +SKIP
            PosixPath('.../seed_100')
        """
        if not self._finished:
            self._finished = True
            for _, f in self._objs.values():
                f.close()
            self._events_f.close()
        return self.dir

    # --- gym interface --------------------------------------------------
    def step(self, action):
        """Step the env, counting steps and sampling poses every `log_freq` of them.

        Args:
            action: passed straight through to the wrapped env.

        Returns:
            tuple: the wrapped env's `(obs, reward, terminated, truncated, info)`,
            untouched.

        Example:
            >>> obs, rew, term, trunc, info = log.step(action)   # doctest: +SKIP
        """
        self._step += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self._step % self.log_freq == 0:
            for name in list(self._objs):
                self._write_row(name)
        return obs, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        """Reset the env, zero the step counter, and log a `reset` event.

        The counter is per-episode, so the `step` field in every event and every
        CSV row means "steps since the last reset" — the planner owns the reset
        (`template_planner.py`), and it usually resets after this wrapper was
        built, which is exactly why the counter cannot be per-wrapper.

        Args:
            *args: forwarded; a positional first argument is read as the seed,
                which is how ManiSkill's `reset(seed, options)` is called.
            **kwargs: forwarded; `seed=` is read for the event.

        Returns:
            tuple: the wrapped env's `(obs, info)`.

        Example:
            >>> obs, info = log.reset(seed=100)   # doctest: +SKIP
            >>> # events.jsonl gains {"step": 0, "event": "reset", "seed": 100}
        """
        self._step = 0
        result = self.env.reset(*args, **kwargs)
        seed = kwargs.get("seed", args[0] if args else None)
        extra = {} if seed is None else {"seed": seed}
        self.log_event("reset", **extra)
        return result

    def close(self):
        """Close the log files and then the wrapped env.

        Example:
            >>> log.close()   # doctest: +SKIP
        """
        try:
            self.finish()
        finally:
            super().close()

    # --- helpers --------------------------------------------------------
    def _live_handle(self, handle):
        """Re-resolve a tracked object to the live entity of the same name.

        `reset(..., reconfigure=True)` rebuilds the scene: actors and robot links
        are new objects, and the handle captured before the reset now points at a
        dead one whose pose never changes again. That produces a trajectory of
        constant rows — a file that looks fine and says nothing. Looking the object
        up by name each write is the fix.

        Args:
            handle: the registered handle.

        Returns:
            The live counterpart from `env.unwrapped.scene.actors` or the robot's
            links, or `handle` unchanged when it has no name or no match (a link
            that is genuinely gone stays stale rather than raising — the trace is
            never allowed to break the run).
        """
        name = getattr(handle, "name", None)
        if not name:
            return handle
        env = getattr(self.env, "unwrapped", None)
        scene = getattr(env, "scene", None)
        actors = getattr(scene, "actors", None) or {}
        if name in actors:
            return actors[name]
        robot = getattr(getattr(env, "agent", None), "robot", None)
        if robot is not None and hasattr(robot, "get_links"):
            for link in robot.get_links():
                if getattr(link, "name", None) == name:
                    return link
        return handle

    def _write_row(self, name):
        """Append one `step,x,y,z,qx,qy,qz,qw` row for a tracked object, and flush.

        Args:
            name (str): a name previously passed to `track_object`.

        Returns:
            None.
        """
        handle, f = self._objs[name]
        handle = self._live_handle(handle)

        # GPU sim keeps poses in torch buffers that are only refreshed on demand;
        # without this the whole file can be the same row. CPU sim reads the live
        # entity and PhysxCpuSystem has no gpu_fetch_* at all, hence the guard.
        scene = getattr(getattr(self.env, "unwrapped", None), "scene", None)
        if scene is not None and getattr(scene, "gpu_sim_enabled", False):
            fetch = getattr(scene, "_gpu_fetch_all", None)
            if fetch is not None:
                fetch()

        pose = handle.pose
        p, q = pose.p[0], pose.q[0]
        if hasattr(p, "cpu"):
            p = p.cpu().numpy()
        if hasattr(q, "cpu"):
            q = q.cpu().numpy()
        # The sim's quaternion is [w, x, y, z]; the CSV is x, y, z, w. This one
        # reordering is the only place the convention changes.
        qx, qy, qz, qw = q[1], q[2], q[3], q[0]
        f.write(
            f"{self._step},{float(p[0]):.6f},{float(p[1]):.6f},{float(p[2]):.6f},"
            f"{float(qx):.6f},{float(qy):.6f},{float(qz):.6f},{float(qw):.6f}\n"
        )
        f.flush()
