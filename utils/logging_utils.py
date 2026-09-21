import contextlib
import io
import json
import os
import re
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np

from mani_skill.utils import common


def _looks_like_an_actor(value):
    """Return whether a task attribute exposes one batched pose."""
    try:
        if not isinstance(getattr(value, "name", None), str):
            return False
        if hasattr(value, "get_links"):
            return False
        pose = getattr(value, "pose", None)
        return hasattr(pose, "p") and hasattr(pose, "q")
    except Exception:
        return False


"""
Planner logging utilities for multi-step robot manipulation environments.

Overview:
---------
This module provides `PlannerLogger` (a Gymnasium wrapper) and `capture_stdout` (a redirect
helper) to capture both high-level event sequences and low-level physics/motion data as
well as step-by-step console logs during planning scripts run.

Outputs:
--------
Every execution run creates a unique timestamped folder at `<log_dir>/<name>_<timestamp>/` containing:
1. `console.log`:
   The complete stdout/stderr output captured during execution, containing planning print statements,
   MPLib logs, optimization status, and IK solutions.
2. `<name>_events.jsonl` (JSON-lines / LLM-friendly events):
   Tracks discrete events with timestamps (e.g. stage starts, errors, final outcome).
   Format: {"step": <int>, "time": "ISO-8601", "event": "start|error|info", "message": "<desc>", ...optional_args}
3. `<object>_trajectory.csv` (CSV trajectories):
   Position and orientational trajectories of registered objects (e.g., cup, robot hand TCP, base)
   sampled every `log_freq` steps.
   Format: step,x,y,z,qx,qy,qz,qw (where quaternion is in [x, y, z, w] order).

What to look for / How to analyze the logs:
------------------------------------------
1. Detecting Execution Failures (Events & Console):
   * Look at `<name>_events.jsonl` for `event: "error"` keys. This indicates exactly which stage of
     planning failed (e.g., "Grasp cup failed") and extracts the root cause (e.g. "IK Failed", "not reachable").
   * Search `console.log` for standard warning phrases such as "IK Failed", "Motion Failed", "stuck",
     or "solver failed" to trace exact workspace or joint limit boundaries that were violated.
2. Detecting Physical Anomalies and Drift (Trajectories):
   * Compare the `x,y,z` trajectory of the robot TCP against the target object's trajectory. Before grasping,
     TCP coordinates should smoothly converge to target object coordinates.
   * Sudden, large jumps in coordinate values (e.g., discrete jumps in base x/y position or TCP coordinates)
     indicate simulated collisions (glitches), physics engine instabilities, or planner teleportation errors.
   * If the robot base or TCP is stationary, its `x,y,z` coordinates should remain constant. Small drifts
     might indicate physical slippage or forces acting on the joints.
3. LLM Analysis Guidelines:
   * Event logs are compact and structured for fast LLM summarization. LLMs should parse the sequence of
     phases (e.g., start -> reach_cup -> grasp_cup -> lift_cup) and look for the absence of `error` types to confirm success.
   * CSV trajectory columns use the standard `[x,y,z]` position and `[qx,qy,qz,qw]` quaternion formats.
     An LLM can compute distances between coordinates `sqrt(dx^2 + dy^2 + dz^2)` at key step indices (e.g. before/after grasp)
     to verify spatial accuracy of final actions (i.e. did the cup actually land inside the bowl?).
"""


@contextlib.contextmanager
def capture_stdout(output_path=None):
    """Route sys.stdout to a file (default: /dev/null) for the duration of the block.

    Keeps the terminal clean of solver/mplib chatter while preserving the output for
    later inspection. stderr is left untouched so real errors stay visible.
    """
    path = output_path if output_path is not None else os.devnull
    try:
        sink = open(path, "w")
    except OSError as e:
        raise OSError(f"cannot open {path} for console log: {e}") from e
    with sink, contextlib.redirect_stdout(sink):
        yield


class PlannerLogger(gym.Wrapper):
    """Wraps an env so every ``env.step`` is observed by the planners' logging.

    Writes into ``<log_dir>/<name>/``:
      * ``<name>_events.jsonl``      -- one JSON object per line (LLM-friendly event log).
      * ``<object>_trajectory.csv``  -- ``step,x,y,z,qx,qy,qz,qw`` per registered object,
        written every ``log_freq`` sim steps.

    Planners call ``logger.track_object(handle, name)`` to register an object whose
    pose should be tracked, and ``logger.log_event(event, message, **extra)`` to record
    high-level phases.
    """

    def __init__(self, env, log_dir="logs", name="run", log_freq=10, run_dir=None):
        super().__init__(env)
        # Each run gets its own timestamped folder, so re-runs never overwrite (or leave
        # stray files from) a previous run with the same name.
        if run_dir is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = Path(log_dir) / f"{name}_{stamp}"
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        try:
            self.log_freq = max(1, int(log_freq or 1))
        except (TypeError, ValueError) as e:
            raise ValueError(f"invalid log_freq {log_freq!r}") from e
        self._step = 0
        # Flush the event log in batches instead of per event: per-event flush()
        # from W parallel workers is a metadata storm on NFS. 64 events is well
        # under a second of planner output, so nothing observable is lost.
        self._events_written = 0
        self._flush_every = 64
        try:
            self._events_f = open(self.dir / f"{name}_events.jsonl", "w", encoding="utf-8")
        except OSError as e:
            raise OSError(f"cannot create event log in {self.dir}: {e}") from e
        self._objs = {}  # name -> open csv file handle
        self._finished = False

    # --- public API -----------------------------------------------------
    def track_object(self, handle, name):
        """Register ``handle`` (any object with batched ``.pose``) as ``name``.

        Its pose is written to ``<name>_trajectory.csv`` every ``log_freq`` steps.
        """
        if name in self._objs:
            return self.dir / f"{name}_trajectory.csv"
        path = self.dir / f"{name}_trajectory.csv"
        try:
            f = open(path, "w", encoding="utf-8")
        except OSError as e:
            raise OSError(f"cannot create trajectory log {path}: {e}") from e
        f.write("step,x,y,z,qx,qy,qz,qw\n")
        self._objs[name] = (handle, f)
        self._write_row(name)  # starting pose
        return path

    def track_task_actors(self, task, names=None):
        """Register direct task actors plus robot TCP/base trajectories."""
        if names is None:
            candidates = [
                (attr, value)
                for attr, value in vars(task).items()
                if not attr.startswith("_") and _looks_like_an_actor(value)
            ]
        else:
            candidates = [(name, getattr(task, name)) for name in names]

        agent = getattr(task, "agent", None)
        for attr, label in (("tcp", "robot_tcp"), ("base_link", "robot_base")):
            handle = getattr(agent, attr, None) if agent is not None else None
            if handle is not None:
                candidates.append((label, handle))

        return {
            label: self.track_object(handle, label)
            for label, handle in candidates
        }

    def log_event(self, event, message="", **extra):
        rec = {
            "step": self._step,
            "time": datetime.now().isoformat(),
            "event": event,
            "message": message,
        }
        rec.update(extra)
        self._events_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._events_written += 1
        if self._events_written % self._flush_every == 0:
            self._events_f.flush()
        return rec

    def log_motion(self, stage, fn, *args, **kwargs):
        """Run a motion/plan call and log an ``error`` event if it fails.

        The solver prints its failure status (e.g. ``IK Failed``) to stdout and returns
        ``-1`` on failure; this captures that output so the log records both the stage
        that failed and the reason. Returns the call's result unchanged.
        """
        real_out = sys.stdout
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = fn(*args, **kwargs)
        out = buf.getvalue()
        # Forward the captured motion output back to the real stdout so nothing is
        # lost from console.log (which is just sys.stdout redirected to a file).
        if out:
            real_out.write(out)
            real_out.flush()
        hits = []
        for ln in (out or "").splitlines():
            s = ln.strip()
            if s and (re.search(r"fail|stuck|not reach|unreachable", s, re.I) or (re.search(r"\bik\b", s, re.I) and not re.search(r"\b(results|solution)\b", s, re.I))):
                hits.append(s)
        if hits or (isinstance(result, (int, float)) and result == -1):
            detail = " | ".join(dict.fromkeys(hits)) or f"plan failed (returned {result})"
            self.log_event("error", f"{stage} failed", detail=detail)
        return result

    # --- gym interface --------------------------------------------------
    def step(self, action):
        self._step += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.log_freq and self._step % self.log_freq == 0:
            for name in list(self._objs):
                self._write_row(name)
        return obs, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self._step = 0
        result = self.env.reset(*args, **kwargs)
        seed = kwargs.get("seed", args[0] if args else None)
        self.log_event("reset", **({} if seed is None else {"seed": seed}))
        return result

    def finish(self):
        """Close log files without closing the wrapped environment."""
        if not self._finished:
            self._finished = True
            self._events_f.flush()
            for _, f in self._objs.values():
                f.close()
            self._events_f.close()
        return self.dir

    def close(self):
        try:
            self.finish()
        finally:
            super().close()

    # --- helpers --------------------------------------------------------
    def _live_handle(self, handle):
        """Re-resolve a tracked object to its current live counterpart.

        reset(..., reconfigure=True) regenerates the scene: robot links (tcp/base_link)
        and actors are recreated, so a handle captured before the reset references stale
        entities whose poses stay frozen at their initial value. Find the live object by
        name instead.
        """
        name = getattr(handle, "name", None)
        if not name:
            return handle
        env = getattr(self.env, "unwrapped", None)
        scene = getattr(env, "scene", None)
        if scene is not None and name in (getattr(scene, "actors", {}) or {}):
            return scene.actors[name]
        robot = getattr(getattr(env, "agent", None), "robot", None)
        if robot is not None and hasattr(robot, "get_links"):
            for link in robot.get_links():
                if getattr(link, "name", None) == name:
                    return link
        return handle

    def _write_row(self, name):
        handle, f = self._objs[name]
        handle = self._live_handle(handle)
        # Force the latest PhysX rigid/articulation poses into the torch buffers before
        # reading. Only relevant for GPU sim (CPU sim's Link.pose already reads live
        # entity poses and PhysxCpuSystem has no gpu_fetch_* methods).
        scene = getattr(getattr(self.env, "unwrapped", None), "scene", None)
        if scene is not None and scene.gpu_sim_enabled:
            fetch = getattr(scene, "_gpu_fetch_all", None)
            if fetch is not None:
                fetch()
        pose = handle.pose
        p = pose.p[0].cpu().numpy()
        q = pose.q[0].cpu().numpy()  # [w, x, y, z]
        qx, qy, qz, qw = q[1], q[2], q[3], q[0]
        f.write(
            f"{self._step},{p[0]:.6f},{p[1]:.6f},{p[2]:.6f},"
            f"{qx:.6f},{qy:.6f},{qz:.6f},{qw:.6f}\n"
        )
        f.flush()


class StreamingVideoRecorder(gym.Wrapper):
    """Record one mp4 per human-render camera, buffering the encoded stream in
    RAM and writing each mp4 to disk ONCE at the end.

    Frames are piped into one ffmpeg per camera (libx264, same settings as
    before), but ffmpeg writes to its stdout (fragmented mp4, pipe-safe)
    instead of straight to the output file. The encoded bytes are drained into
    an in-memory buffer during the run and flushed to
    ``<output_dir>/video_<camera>.mp4`` in a single sequential write on
    finalize.

    This keeps the exact same encode and quality while removing the per-frame
    NFS write storm that backpressured the sim (75 concurrent streams at 25
    workers): NFS now sees one sequential write per camera at the end. RAM cost
    is the *compressed* stream (~0.1-0.3 GB per worker), not the raw frames
    that RecordEpisode holds in memory.
    """

    def __init__(self, env, output_dir, video_fps=30):
        super().__init__(env)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.video_fps = video_fps
        self._procs: dict[str, subprocess.Popen] = {}
        self._bufs: dict[str, io.BytesIO] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._disabled: set[str] = set()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._write_frame()
        return obs, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self._finalize()
        return self.env.reset(*args, **kwargs)

    def close(self):
        self._finalize()
        return super().close()

    def _write_frame(self):
        env = self.env.unwrapped
        for obj in env._hidden_objects:
            obj.show_visual()
        env.scene.update_render(update_sensors=False, update_human_render_cameras=True)
        render_images = env.scene.get_human_render_camera_images()
        for obj in env._hidden_objects:
            obj.hide_visual()
        for name, img in render_images.items():
            if name in self._disabled:
                continue
            img = common.to_numpy(img)
            if img.ndim < 3 or img.size == 0:
                # e.g. render_mode="human" produces no offscreen images; skip
                continue
            if img.ndim == 4:
                if img.shape[0] != 1:
                    raise ValueError("StreamingVideoRecorder supports num_envs=1 only")
                img = img[0]
            proc = self._procs.get(name)
            if proc is None:
                proc = self._start(name, img.shape[0], img.shape[1])
                self._procs[name] = proc
            stdin = proc.stdin
            if stdin is None:
                self._disabled.add(name)
                continue
            try:
                stdin.write(np.ascontiguousarray(img, dtype=np.uint8).tobytes())
            except BrokenPipeError:
                print(f"[StreamingVideoRecorder] ffmpeg exited for camera '{name}', "
                      "disabling its video recording")
                self._procs.pop(name, None)
                self._bufs.pop(name, None)
                self._threads.pop(name, None)
                self._disabled.add(name)
                continue
            # The ffmpeg stdout is drained continuously by a background thread
            # (started in _start), so ffmpeg never blocks on its own stdout and
            # always consumes our stdin: no pipe deadlock.

    def _start(self, name, height, width):
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(self.video_fps),
            "-i", "-",
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-f", "mp4", "-movflags", "frag_keyframe+empty_moov+default_base_moof",
            "pipe:1",
        ]
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        stdout = proc.stdout
        if stdout is None:
            raise RuntimeError(f"ffmpeg for '{name}' has no stdout pipe")
        buf = io.BytesIO()
        self._bufs[name] = buf
        # A dedicated daemon thread drains ffmpeg's stdout into the RAM buffer.
        # This guarantees ffmpeg never wedges on a full stdout pipe while we
        # block on stdin.write (the classic two-pipe deadlock: we wait to write
        # stdin, ffmpeg waits to write stdout because nobody drains it).
        t = threading.Thread(target=self._drain_loop, args=(name, proc, buf), daemon=True, name=f"ffmpeg-drain-{name}")
        t.start()
        self._threads[name] = t
        return proc

    def _drain_loop(self, name, proc, buf):
        """Continuously move ffmpeg's stdout into the RAM buffer until EOF.

        Runs on its own thread; blocks on stdin-close/EOF only of this thread.
        """
        stdout = proc.stdout
        if stdout is None:
            return
        try:
            while True:
                chunk = stdout.read(1 << 20)
                if not chunk:
                    break
                buf.write(chunk)
        except (ValueError, OSError):
            pass

    def _finalize(self):
        # Close stdin ends so ffmpeg hits EOF and drains its output for good.
        for name, proc in self._procs.items():
            stdin = proc.stdin
            if stdin is not None:
                stdin.close()
        # Let the drain threads hit EOF (ffmpeg closes stdout after stdin EOF)
        # and the children exit; then collect what they wrote.
        for name, t in list(self._threads.items()):
            t.join(timeout=30.0)
            if t.is_alive():
                proc = self._procs.get(name)
                if proc is not None:
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                    t.join(timeout=5.0)
        for name, proc in self._procs.items():
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        # Single sequential write of the whole encoded stream per camera.
        for name in list(self._procs):
            buf = self._bufs.pop(name, None)
            if buf is None:
                continue
            data = buf.getvalue()
            if not data or not re.fullmatch(r"[a-zA-Z0-9_]+", name):
                continue
            path = self.output_dir / f"video_{name}.mp4"
            try:
                with open(path, "wb") as f:
                    f.write(data)
            except OSError as e:
                print(f"[StreamingVideoRecorder] failed to write {path}: {e}")
        self._procs = {}
        self._bufs = {}
