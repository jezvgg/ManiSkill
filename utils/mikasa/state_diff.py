"""What changed between two moments of an episode, as text an LLM agent can read.

    python -m utils.mikasa.state_diff --task MyRoboCasa-v1 --scene-idx 0 --at 0,5,20
    python -m utils.mikasa.state_diff --task MikasaSeasonDish-v0 --at 0,30,60
    python -m utils.mikasa.state_diff --at 0,10 --frames /tmp/sd --objects cup,bowl

This is the repo's answer to CaP-X's Visual Differencing Module, and it exists
because of the one measurement in that paper that changes how we work: feeding a
coding agent raw RGB frames each turn scored *worse* than giving it no images at
all, while a structured **text description of the difference between two
observations** beat both (`docs/coding-agent-primer.md` §2, tier M3 vs M1/M2).
Their VDM is a second VLM captioning frames; ours does not need one, because in
simulation the state is right there — poses, grasp flags, and the `info` dict the
env already hands back. So when you are debugging why an oracle stalls or why
`evaluate()` never flips, read a diff, not a video.

Three lines of house rules are baked in:

- **`agent.base_link.pose`, never `agent.robot.pose`.** Fetch drives through
  root_x/root_y/root_z_rotation joints (fetch.urdf:21,28,35), so the articulation
  root stays at the spawn point all episode and `robot.pose` reports "the base
  never moved" no matter where the robot drove (`station_checklist.py:583-586`).
- **`info` comes from `reset`/`step`, never from calling `env.evaluate()`.** Tasks
  here latch state inside `evaluate()`, and an out-of-band call advances that latch
  a step early — three existing call sites already do this (`AGENTS.md`, the
  season_dish section). `snapshot()` takes `info` as an argument for that reason.
- **`obs_mode="state"`, not `"rgb"`.** `sapien_env.py:589` calls
  `torch.cuda.synchronize()` unconditionally at the end of every sensor read, so
  `obs_mode="rgb"` hard-crashes on a CPU-only torch build. `--frames` gets its
  images from the sensors directly instead, which does not go through that path.

Everything except the CLI is plain Python over plain dicts: `snapshot()` needs a
live env, but `diff()` and `format_snapshot()` do not, so snapshots can be written
to `poses.json` on the GPU box and diffed on a laptop.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

#: Below these, a difference is sensor/solver noise rather than motion. 5 mm is
#: about the resting jitter of a settled RoboCasa object over a few hundred steps.
DEFAULT_POS_TOL = 0.005
DEFAULT_ANG_TOL_DEG = 2.0

#: `info` numbers this close are the same number. Reward is a float that wobbles in
#: the last bits; a real change in any predicate here is orders of magnitude bigger.
INFO_NUM_TOL = 1e-6


# --------------------------------------------------------------------- format --


def _fmt(x) -> str:
    """One number, three decimals — the `_fmt` convention from diagnose_task.py:31."""
    return f"{float(x):.3f}"


def _vec(v) -> str:
    """A position or quaternion as `(0.100, 0.200, 0.300)`."""
    return "(" + ", ".join(_fmt(c) for c in v) + ")"


def _val(v) -> str:
    """An `info` value for a diff line: bools verbatim, ints exact, floats to 3dp."""
    if v is None:
        return "absent"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    return _fmt(v)


# ------------------------------------------------------------------- snapshot --


def _pose_dict(pose, index: int = 0) -> dict:
    """`{"p": [x, y, z], "q": [w, x, y, z]}` for one env of a batched ManiSkill pose."""
    return {
        "p": [round(float(c), 6) for c in pose.p[index]],
        "q": [round(float(c), 6) for c in pose.q[index]],
    }


def _scalar(value):
    """A scalar/bool from an `info` entry, or None if it is not one.

    Tensors are read at env index 0 — this whole module is a single-env debugging
    tool. Anything with more than one axis (a pose, a per-object mask) is skipped
    rather than summarised: a wrong summary is worse than an absent line.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        if len(shape) == 0:
            item = value.item()
        elif len(shape) == 1 and shape[0] >= 1:
            item = value[0].item()
        else:
            return None
    except Exception:
        return None
    if isinstance(item, (bool, int, float)):
        return item
    return None


def _flatten_info(info, prefix: str = "") -> dict:
    """Flatten an `info` dict to scalars, nested keys joined with `.`, sorted."""
    out: dict = {}
    if not info:
        return out
    for key in sorted(info, key=str):
        value = info[key]
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten_info(value, prefix=f"{name}."))
            continue
        scalar = _scalar(value)
        if scalar is not None:
            out[name] = scalar
    return out


def _gripper_indices(agent):
    """Active joint indices of the gripper controller, or of the last controller.

    "Restricted to the gripper joints if easily identifiable" means exactly this:
    the controller dict is keyed by name and every robot in this repo calls that
    one `gripper` (`ds_fetch.py:169-175`). If a future robot does not, the last
    controller's joints are reported instead — for Fetch's layout that is the same
    thing, and the key is named `gripper_qpos` either way, so read the module
    docstring before trusting it on a robot that is not Fetch-shaped.
    """
    controller = getattr(agent, "controller", None)
    controllers = getattr(controller, "controllers", None)
    if not controllers:
        return None
    for name, ctrl in controllers.items():
        if "gripper" in name.lower():
            return list(ctrl.active_joint_indices)
    return list(list(controllers.values())[-1].active_joint_indices)


def _is_grasping(agent, actor):
    """`agent.is_grasping(actor)` for env 0, or None when the agent cannot say.

    Guarded because it is a contact query: a robot with no gripper, an actor with
    no collision shapes (both cue markers in `season_dish.py` are built with
    `add_collision=False`) or a mid-reconfigure scene all raise here, and none of
    those is a reason for a diagnostic to die.
    """
    try:
        return bool(agent.is_grasping(actor)[0])
    except Exception:
        return None


def snapshot(env, info=None, objects=None) -> dict:
    """Everything worth knowing about one moment of an episode, as a plain dict.

    JSON-serialisable by construction (no tensors, no numpy scalars), so a snapshot
    survives being written to a file, mailed to another machine, or diffed months
    later against a run nobody can reproduce.

    Args:
        env: a `gym.Env` wrapping a ManiSkill `BaseEnv`, already `reset`. Only env
            index 0 is read.
        info: the `info` dict returned by `env.reset` or `env.step`, or None. It is
            passed in rather than fetched because `env.evaluate()` is not
            side-effect-free in this repo — tasks that latch state advance the latch
            when it is called out of band (`AGENTS.md`, season_dish section).
        objects: attribute names on `env.unwrapped` to record. Default is every
            attribute holding a `mani_skill.utils.structs.Actor`, skipping names
            that start with `_`. An explicit name that does not exist raises.

    Returns:
        dict with keys:
            `step` (int) — `env.unwrapped.elapsed_steps[0]`;
            `base_link` / `tcp` — `{"p": [x, y, z], "q": [w, x, y, z]}` or None;
            `gripper_qpos` (list[float]) — see `_gripper_indices`;
            `objects` — `{name: {"p", "q", "is_grasping"}}`, keys sorted, where
            `is_grasping` is a bool or None if the query could not be made;
            `info` — the flattened scalar/bool entries of `info`, keys sorted.

    Example:
        >>> import gymnasium as gym, utils.mikasa  # doctest: +SKIP
        >>> env = gym.make("MyRoboCasa-v1", num_envs=1, robot_uids="mikasa_ds_fetch",
        ...                control_mode="pd_joint_pos", obs_mode="state")  # doctest: +SKIP
        >>> obs, info = env.reset(seed=3, options={"reconfigure": True})  # doctest: +SKIP
        >>> s0 = snapshot(env, info)  # doctest: +SKIP
        >>> sorted(s0["objects"])  # doctest: +SKIP
        ['bowl', 'cup']
        >>> s0["step"]  # doctest: +SKIP
        0
    """
    from mani_skill.utils.structs import Actor

    task = env.unwrapped
    agent = getattr(task, "agent", None)

    if objects is None:
        names = sorted(
            name
            for name, value in vars(task).items()
            if not name.startswith("_") and isinstance(value, Actor)
        )
    else:
        names = sorted(objects)

    recorded: dict = {}
    for name in names:
        actor = getattr(task, name, None)
        if actor is None:
            raise AttributeError(
                f"{type(task).__name__} has no attribute {name!r}; "
                f"it holds these actors: {sorted(n for n, v in vars(task).items() if isinstance(v, Actor))}"
            )
        entry = _pose_dict(actor.pose)
        entry["is_grasping"] = _is_grasping(agent, actor) if agent is not None else None
        recorded[name] = entry

    gripper: list = []
    if agent is not None:
        idx = _gripper_indices(agent)
        if idx:
            qpos = agent.robot.get_qpos()
            gripper = [round(float(qpos[0, i]), 6) for i in idx]

    base_link = getattr(agent, "base_link", None) if agent is not None else None
    tcp = getattr(agent, "tcp", None) if agent is not None else None
    return {
        "step": int(task.elapsed_steps[0]),
        "base_link": _pose_dict(base_link.pose) if base_link is not None else None,
        "tcp": _pose_dict(tcp.pose) if tcp is not None else None,
        "gripper_qpos": gripper,
        "objects": recorded,
        "info": _flatten_info(info),
    }


# ----------------------------------------------------------------------- diff --


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


def _distance(pa, pb) -> float:
    return math.sqrt(sum((float(y) - float(x)) ** 2 for x, y in zip(pa, pb)))


def _pose_lines(name: str, a: dict, b: dict, pos_tol: float, ang_tol_deg: float) -> list:
    """`moved`/`rotated` lines for one named pose, in that order."""
    lines = []
    pa, pb = a.get("p"), b.get("p")
    if pa and pb:
        moved = _distance(pa, pb)
        if moved > pos_tol:
            dz = float(pb[2]) - float(pa[2])
            lines.append(f"{name}: moved {_fmt(moved)} m (dz {dz:+.3f}) -> {_vec(pb)}")
    angle = quat_angle_deg(a.get("q"), b.get("q"))
    if angle is not None and angle > ang_tol_deg:
        lines.append(f"{name}: rotated {angle:.1f} deg")
    return lines


def _info_changed(old, new) -> bool:
    if old is None or new is None:
        return not (old is None and new is None)
    if isinstance(old, bool) or isinstance(new, bool):
        return bool(old) != bool(new)
    return abs(float(new) - float(old)) > INFO_NUM_TOL


def _info_pair(old, new) -> str:
    """`old -> new`, widened to 6 decimals when 3 would print the same twice.

    Real case, from `MikasaSeasonDish-v0`: `distractor_moved` crept by a few tens of
    microns and the line came out as `0.000 -> 0.000`, which reads as a bug in the
    diff. Raising the change threshold instead would hide the creep, and the creep
    is the point — so widen the format rather than drop the finding.
    """
    left, right = _val(old), _val(new)
    if left == right:
        left = f"{float(old):.6f}"
        right = f"{float(new):.6f}"
    return f"{left} -> {right}"


def diff(a: dict, b: dict, pos_tol: float = DEFAULT_POS_TOL,
         ang_tol_deg: float = DEFAULT_ANG_TOL_DEG) -> list:
    """What changed between two snapshots, one finding per line.

    Deterministic: objects first, sorted by name, each contributing its `moved`,
    `rotated` and `is_grasping` lines in that order; then `base_link`, then `tcp`
    (both of which also report rotation — the base's yaw is how you tell "drove
    forward" from "spun in place"); then changed `info` keys, sorted; then the
    count of objects that did not change at all.

    That last line is the one design decision worth knowing about: **the
    `"N objects unchanged"` summary is emitted whenever at least one object was
    compared and found unchanged**, so a diff of two identical snapshots of a scene
    with objects is `["2 objects unchanged"]` rather than `[]`. Silence would be
    ambiguous — "nothing moved" and "I was never given any objects" would print the
    same. `[]` therefore means literally nothing was comparable. Objects present in
    only one snapshot get an `appeared`/`disappeared` line instead of being dropped.

    Args:
        a, b: snapshots from `snapshot()` (or hand-built dicts of the same shape),
            `a` being the earlier moment. Missing keys are tolerated.
        pos_tol: metres. A displacement at or below this is not reported.
        ang_tol_deg: degrees. A rotation at or below this is not reported.

    Returns:
        list[str], possibly empty, safe to `print("\\n".join(...))`.

    Example:
        >>> a = {"objects": {"cup": {"p": [0, 0, 0.9], "q": [1, 0, 0, 0],
        ...                          "is_grasping": False}}, "info": {"success": False}}
        >>> b = {"objects": {"cup": {"p": [0, 0, 1.05], "q": [1, 0, 0, 0],
        ...                          "is_grasping": True}}, "info": {"success": True}}
        >>> for line in diff(a, b): print(line)
        cup: moved 0.150 m (dz +0.150) -> (0.000, 0.000, 1.050)
        cup: is_grasping False -> True
        info.success: False -> True
        >>> diff(a, a)
        ['1 object unchanged']
    """
    lines: list = []
    objects_a = a.get("objects") or {}
    objects_b = b.get("objects") or {}
    unchanged = 0

    for name in sorted(set(objects_a) | set(objects_b)):
        if name not in objects_b:
            lines.append(f"{name}: disappeared (present only in the earlier snapshot)")
            continue
        if name not in objects_a:
            lines.append(f"{name}: appeared (present only in the later snapshot)")
            continue
        found = _pose_lines(name, objects_a[name], objects_b[name], pos_tol, ang_tol_deg)
        grasp_a = objects_a[name].get("is_grasping")
        grasp_b = objects_b[name].get("is_grasping")
        if grasp_a != grasp_b:
            found.append(f"{name}: is_grasping {grasp_a} -> {grasp_b}")
        if found:
            lines.extend(found)
        else:
            unchanged += 1

    for link in ("base_link", "tcp"):
        pose_a, pose_b = a.get(link), b.get(link)
        if pose_a and pose_b:
            lines.extend(_pose_lines(link, pose_a, pose_b, pos_tol, ang_tol_deg))

    info_a = a.get("info") or {}
    info_b = b.get("info") or {}
    for key in sorted(set(info_a) | set(info_b), key=str):
        old, new = info_a.get(key), info_b.get(key)
        if _info_changed(old, new):
            lines.append(f"info.{key}: {_info_pair(old, new)}")

    if unchanged:
        lines.append(f"{unchanged} object{'' if unchanged == 1 else 's'} unchanged")
    return lines


def format_snapshot(s: dict) -> str:
    """One snapshot as a block of text.

    The counterpart to `diff`: a diff says what moved, this says where everything
    is. Print it once for the first moment of a run and diff from there — printing
    it at every step is how you end up with a wall of numbers nobody reads, which is
    the failure mode this module exists to avoid.

    Args:
        s: a snapshot dict from `snapshot()`.

    Returns:
        str, multi-line, no trailing newline.

    Example:
        >>> s = {"step": 0, "base_link": None, "tcp": None, "gripper_qpos": [0.05],
        ...      "objects": {"cup": {"p": [0, 0, 0.9], "q": [1, 0, 0, 0],
        ...                          "is_grasping": False}},
        ...      "info": {"success": False}}
        >>> print(format_snapshot(s))
        step 0
          gripper_qpos (0.050)
          objects (1):
            cup  p=(0.000, 0.000, 0.900)  q=(1.000, 0.000, 0.000, 0.000)  is_grasping=False
          info (1):
            success = False
    """
    out = [f"step {s.get('step')}"]
    for link in ("base_link", "tcp"):
        pose = s.get(link)
        if pose:
            out.append(f"  {link:<10} p={_vec(pose['p'])}  q={_vec(pose['q'])}")
    gripper = s.get("gripper_qpos")
    if gripper:
        out.append(f"  gripper_qpos {_vec(gripper)}")

    objects = s.get("objects") or {}
    out.append(f"  objects ({len(objects)}):")
    width = max((len(n) for n in objects), default=0)
    for name in sorted(objects):
        entry = objects[name]
        out.append(
            f"    {name:<{width}}  p={_vec(entry['p'])}  q={_vec(entry['q'])}"
            f"  is_grasping={entry.get('is_grasping')}"
        )

    info = s.get("info") or {}
    out.append(f"  info ({len(info)}):")
    for key in sorted(info, key=str):
        out.append(f"    {key} = {_val(info[key])}")
    return "\n".join(out)


# ------------------------------------------------------------------ live runs --


def backend_kwargs() -> dict:
    """`sim_backend`/`render_backend` kwargs for `gym.make` that work on this machine.

    ManiSkill defaults `render_backend` to `"gpu"`, which on a machine without CUDA
    dies in `sapien.Device("cuda")` (`sapien_env.py:260-263`) before the scene is
    built. Nothing else about the CPU path is exotic: with these two kwargs the
    RoboCasa kitchen builds and renders through MoltenVK on an M2 in about four
    seconds. On a CUDA machine this returns ManiSkill's own defaults.

    Returns:
        `{"sim_backend": ..., "render_backend": ...}` — `("cpu", "cpu")` without
        CUDA, `("auto", "gpu")` with it.

    Example:
        >>> kw = backend_kwargs()
        >>> sorted(kw)
        ['render_backend', 'sim_backend']
        >>> env = gym.make("MyRoboCasa-v1", num_envs=1, **kw)  # doctest: +SKIP
    """
    import torch

    cuda = torch.cuda.is_available()
    return dict(sim_backend="auto" if cuda else "cpu", render_backend="gpu" if cuda else "cpu")


def build_env(task: str, scene_idx=0, render_size: int = 256):
    """Build a single-env, CPU-safe instance of a task, the way that is known to work.

    The exact recipe below is the one verified to run on a CPU-only macOS box, and
    two of its arguments are load-bearing: `obs_mode="state"` because the sensor
    read path ends in an unconditional `torch.cuda.synchronize()`
    (`sapien_env.py:589`) that aborts on a CPU-only torch, and the `cpu` backends
    because `auto` picks a GPU that is not there. `scene_idx` is passed only to
    tasks that accept one — `MyRoboCasa_TakeItBack-v1` pins its own kitchen and
    would reject the kwarg (`evaluate_planner.accepts_scene_idx`).

    Args:
        task: registered env id, e.g. `"MyRoboCasa-v1"`.
        scene_idx: kitchen index, or None to leave the draw random. Ignored with a
            note on stderr for tasks that do not accept it.
        render_size: width and height of the human render camera, in pixels.

    Returns:
        The `gym.Env`. Not reset — the caller owns the seed.

    Example:
        >>> env = build_env("MyRoboCasa-v1", scene_idx=0)  # doctest: +SKIP
        >>> obs, info = env.reset(seed=3, options={"reconfigure": True})  # doctest: +SKIP
        >>> env.close()  # doctest: +SKIP
    """
    import gymnasium as gym

    import mani_skill  # noqa: F401  (registers my_scenes and utils.mikasa)
    from utils.mikasa_oracle.evaluate_planner import accepts_scene_idx

    make_kwargs = dict(
        num_envs=1,
        robot_uids="mikasa_ds_fetch",
        control_mode="pd_joint_pos",
        obs_mode="state",
        render_mode="rgb_array",
        human_render_camera_configs=dict(width=render_size, height=render_size),
        **backend_kwargs(),
    )
    if scene_idx is not None:
        if accepts_scene_idx(task):
            make_kwargs["scene_idx"] = scene_idx
        else:
            print(f"{task} does not take scene_idx; ignoring it.", file=sys.stderr)
    return gym.make(task, **make_kwargs)


def save_frames(env, out_dir, step: int) -> list:
    """Write the human render and every sensor image for the current moment.

    Files are `step_{k:04d}_render_camera.png` and `step_{k:04d}_{sensor_uid}.png`.

    The sensors are read directly — `scene.update_render()`, then `capture()` and
    `get_obs(rgb=True, depth=False, position=False, segmentation=False)` per sensor
    — rather than through `env.get_obs()`, because the env's own path finishes with
    an unconditional `torch.cuda.synchronize()` (`sapien_env.py:589`). Two
    consequences of the bypass, both worth knowing: it does not apply the task's
    `_hidden_objects` masking, so a marker the policy is not supposed to see (the
    magenta ghost in `season_dish.py`) **will** appear in these PNGs; and it costs a
    render pass, so pass `--frames` only when you actually want pictures.

    Args:
        env: a `gym.Env` built with `render_mode="rgb_array"`, already reset.
        out_dir: `pathlib.Path` (or str) of an existing directory.
        step: the step number to put in the filenames.

    Returns:
        list[str] of the paths written, in the order written.

    Example:
        >>> save_frames(env, "/tmp/sd", 10)  # doctest: +SKIP
        ['/tmp/sd/step_0010_render_camera.png', '/tmp/sd/step_0010_fetch_hand.png', ...]
    """
    from pathlib import Path

    from PIL import Image

    out_dir = Path(out_dir)
    task = env.unwrapped
    written = []

    frame = env.render()
    array = frame[0]
    array = array.cpu().numpy() if hasattr(array, "cpu") else array
    path = out_dir / f"step_{step:04d}_render_camera.png"
    Image.fromarray(array).save(path)
    written.append(str(path))

    task.scene.update_render()
    for uid, sensor in task._sensors.items():
        sensor.capture()
        obs = sensor.get_obs(rgb=True, depth=False, position=False, segmentation=False)
        if "rgb" not in obs:
            continue
        rgb = obs["rgb"][0]
        rgb = rgb.cpu().numpy() if hasattr(rgb, "cpu") else rgb
        path = out_dir / f"step_{step:04d}_{uid}.png"
        Image.fromarray(rgb).save(path)
        written.append(str(path))
    return written


def _flag(value) -> bool:
    """`terminated`/`truncated` as a bool, whether it arrived as a tensor or not."""
    try:
        return bool(value[0])
    except (TypeError, IndexError, KeyError):
        return bool(value)


def run(task: str, scene_idx=0, seed: int = 3, at=(0, 10), policy: str = "hold",
        frames=None, objects=None, render_size: int = 256) -> list:
    """Roll one episode out under a fixed policy and print snapshots and diffs.

    Args:
        task: registered env id.
        scene_idx: kitchen index, or None.
        seed: passed to `env.reset`, which also gets `options={"reconfigure": True}`
            so the kitchen is rebuilt rather than reused.
        at: step indices to snapshot at. Sorted and de-duplicated; 0 means "right
            after reset", using the reset's own `info`.
        policy: only `"hold"` today — the identity action from
            `utils.mikasa.hold`. Anything else raises.
        frames: directory to write PNGs and `poses.json` into, or None for text
            only. Created if missing.
        objects: list of attribute names to track, or None for auto-discovery.
        render_size: human render camera size in pixels.

    Returns:
        list[dict] — every snapshot taken, in step order. Also written to
        `<frames>/poses.json` when `frames` is given.

    Example:
        >>> snaps = run("MyRoboCasa-v1", scene_idx=0, seed=3, at=(0, 5))  # doctest: +SKIP
        >>> [s["step"] for s in snaps]  # doctest: +SKIP
        [0, 5]
    """
    from pathlib import Path

    from utils.mikasa.hold import hold_still

    if policy != "hold":
        raise ValueError(f"unknown policy {policy!r}; only 'hold' exists so far")

    targets = sorted({int(k) for k in at})
    out_dir = None
    if frames:
        out_dir = Path(frames)
        out_dir.mkdir(parents=True, exist_ok=True)

    env = build_env(task, scene_idx=scene_idx, render_size=render_size)
    print(
        f"=== state diff: {task} scene_idx={scene_idx} seed={seed} policy={policy} "
        f"at={targets} ===",
        flush=True,
    )
    snapshots = []
    try:
        _, info = env.reset(seed=seed, options={"reconfigure": True})
        current = int(env.unwrapped.elapsed_steps[0])
        stopped = ""
        for target in targets:
            to_go = target - current
            if to_go > 0:
                result = hold_still(env, to_go)
                _, _, terminated, truncated, info = result
                current = int(env.unwrapped.elapsed_steps[0])
                if _flag(terminated) or _flag(truncated):
                    stopped = (
                        f"episode ended at step {current} "
                        f"(terminated={_flag(terminated)}, truncated={_flag(truncated)})"
                    )
            taken = snapshot(env, info, objects=objects)
            if out_dir is not None:
                save_frames(env, out_dir, current)

            # Printed as they are taken, not collected and dumped at the end: a long
            # rollout on CPU is minutes, and output you can read while it runs is the
            # whole reason this is a text tool.
            if not snapshots:
                print()
                print(format_snapshot(taken), flush=True)
            else:
                previous = snapshots[-1]
                print(f"\n--- step {previous['step']} -> step {taken['step']}", flush=True)
                lines = diff(previous, taken)
                if lines:
                    print("\n".join(f"  {line}" for line in lines), flush=True)
                else:
                    print(
                        f"  (nothing changed beyond {DEFAULT_POS_TOL} m / "
                        f"{DEFAULT_ANG_TOL_DEG} deg, and there were no objects to count)",
                        flush=True,
                    )
            snapshots.append(taken)
            if stopped:
                break
        if stopped:
            print(f"\n  {stopped} — no further snapshots taken.")
        if out_dir is not None:
            poses = out_dir / "poses.json"
            poses.write_text(json.dumps(snapshots, indent=2))
            print(f"\n  frames and {poses} written to {out_dir}")
    finally:
        env.close()
    return snapshots


def parse_args(argv=None):
    """Parse the CLI arguments, separately from acting on them.

    Split out so the argument surface can be checked offline — nothing here builds
    an env, so a test can assert the defaults without a simulator.

    Args:
        argv: list of argument strings, or None to read `sys.argv[1:]`.

    Returns:
        `argparse.Namespace` with `task, scene_idx, seed, at, policy, frames,
        objects, render_size`. `at` and `objects` are still raw strings at this
        point; `main` splits them.

    Example:
        >>> args = parse_args(["--task", "MikasaSeasonDish-v0", "--at", "0,30,60"])
        >>> args.task, args.at, args.scene_idx, args.policy
        ('MikasaSeasonDish-v0', '0,30,60', 0, 'hold')
    """
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task", default="MyRoboCasa-v1", help="registered env id")
    p.add_argument("--scene-idx", type=int, default=0, help="pin the kitchen")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--at", default="0,10", help="comma-separated step indices")
    p.add_argument("--policy", default="hold", choices=["hold"])
    p.add_argument("--frames", default=None, help="directory for PNGs and poses.json")
    p.add_argument("--objects", default=None, help="comma-separated attribute names")
    p.add_argument("--render-size", type=int, default=256)
    return p.parse_args(argv)


def main(argv=None) -> int:
    """CLI entry point: parse, roll out, print.

    Args:
        argv: argument strings, or None to read `sys.argv[1:]`.

    Returns:
        int exit code, 0 on success. Anything that goes wrong raises rather than
        returning non-zero — a diagnostic that swallows its own traceback is how you
        lose the one piece of information you ran it for.

    Example:
        >>> main(["--task", "MyRoboCasa-v1", "--at", "0,5"])  # doctest: +SKIP
        0
    """
    args = parse_args(argv)
    at = [int(k) for k in args.at.split(",") if k.strip()]
    objects = (
        [name.strip() for name in args.objects.split(",") if name.strip()]
        if args.objects
        else None
    )
    run(
        task=args.task,
        scene_idx=args.scene_idx,
        seed=args.seed,
        at=at,
        policy=args.policy,
        frames=args.frames,
        objects=objects,
        render_size=args.render_size,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
