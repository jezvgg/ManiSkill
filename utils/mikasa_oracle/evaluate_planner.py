"""Run a scripted planner over many episodes and report its success rate.

    python -m utils.mikasa_oracle.evaluate_planner -n 100 --scene-idx 0
    python -m utils.mikasa_oracle.evaluate_planner -n 25 --start-seed 102 --seed-step 0

Ported from jezvgg/ManiSkill `utils/test_planner.py` (commit `0b0493f`). Renamed
because pytest collects `test_*.py` and this is not a test.

The loop is upstream's, including the line the whole thing rests on:

    ok = bool(planning_fn(env, seed))

The verdict comes from the planner, so the planner's return value has to be real.
It was not: `myrobocasa_takeitback_planner.planning` returned a hardcoded `True`
until the fix that landed with this file, and this harness would have reported
100% on it without simulating anything.

**The rate still depends on how many episodes you run.** Repeating a sweep now
reproduces it — `seeding.seed_everything` seeds mplib per episode, which it did not
before — but the scene is never restored between episodes, so a seed's outcome
depends on its position in the run. Measured on MyRoboCasa-v1, kitchen 0, before
the seeding fix:

    20 episodes from seed 100 ............ 45%
    the same 20 seeds, reversed .......... 15%
    seed 102, repeated 25 times .......... 16%

Episodes 1 and 2 succeeded in every run taken and later ones rarely, because
`scene_builder.initialize(env_idx)` is never called — the standing
`initialize_episode_resets_the_scene` violation. So quote N with any number from
here, and compare two planners only at the same N.

Two departures from upstream, both because upstream's number is hard to read:

- `--scene-idx` pins the kitchen. Without it `RoboCasaSceneBuilder` redraws one per
  episode and the kitchens with no `counter_main_main_group_Counter` raise, which
  upstream books as a failure — mixing "the planner missed" with "this kitchen has
  no counter" (§4). The three memory tasks took the lesson into their own defaults
  (`scene_idx=0`, kitchen 102) and so are pinned whether or not the flag is passed;
  the inherited tasks still draw at random without it.
- Failures are reported split into misses, failed plans and errors, for the same
  reason. A planner that returns the `-1` sentinel found no path; that is not the
  same event as a plan that ran and left the task unsolved.

## `--blind`: the memoryless control arm, over the same sweep

    python -m utils.mikasa_oracle.evaluate_planner -s MikasaBurner-v0 -p burner_planner \
        -n 100 --obs-mode state --blind

The memory tasks state a chance floor, and the only thing that turns that claim into a
measurement is running the planner's memory-free twin over the *same* seed sweep: a
blind arm at 0.25 x the sighted arm's motor rate is the task working, anything above it
is the cue leaking into the act phase. `burner_planner`, `season_dish_planner` and
`station_checklist_planner` all export that twin as a `blind=` parameter of `solve`,
but until this flag existed it was reachable only from each module's own `__main__`,
which runs one episode per process — N first episodes, which is not the same population
as episodes 1..N of a sweep (see the warning above, and compare only at the same N).

`--blind` passes `blind=True` as a keyword. A planner that does not declare the
parameter is a hard error rather than a silently sighted run: the sweep exits with a
message naming the planner.

## `--video-dir`: record the sweep itself, one named clip per episode

    python -m utils.mikasa_oracle.evaluate_planner -s MikasaBurner-v0 -p burner_planner \
        -n 1 --obs-mode state --video-dir videos/burner --render-backend cpu

RecordEpisode wraps the env *inside* the sweep (PlannerLogger stays outside), built
with `save_on_reset=False` on purpose: the planner owns its reset — it happens inside
the planner call — and the clip is named by the sweep once the verdict is known, as
`ep{i:03d}_seed{seed}_{ok|fail}[_blind]`. The episode index is part of the name
because `--seed-step 0` repeats one seed. The flush runs in a `finally`, and any
frames it leaves buffered are then dropped explicitly — RecordEpisode clears its
buffer only *after* a successful encode, and not at all on its one-frame early
return — so an errored episode or a dead encoder costs that clip and nothing
after it: the failure is logged as `video: <exc>` (and into events.jsonl under
`--log-dir`) and the sweep continues with an empty buffer. Do not "harmonize"
this with the planners' own `__main__` recorders, which do use
`save_on_reset=True` — there the env owns the reset.

`--max-steps-per-video` defaults to None because the three memory tasks stop at
their horizon (T1), which bounds the frame buffer per episode. Set it for an env
with no TimeLimit (MyRoboCasa-v1), where a runaway episode would otherwise buffer
frames until `planner.sh run --timeout N` kills the container; when set,
RecordEpisode auto-flushes anonymous chunks (`0.mp4`, `1.mp4`, ...) mid-episode and
the named clip is the tail. In the container pass `--render-backend cpu` (lavapipe);
on Colab the default backend is the right one. There is no `--max-episode-seconds`
and there must not be: hang protection is `planner.sh run --timeout N` (D4).

## `--info-keys`: per-key tallies over the sweep

    python -m utils.mikasa_oracle.evaluate_planner -s MikasaStationChecklist-v0 \
        -p station_checklist_planner -n 10 --obs-mode state \
        --info-keys first_decision_ok,double_service_count

Comma-separated `info` keys to tally across the sweep. Only episodes that returned
a gym 5-tuple carrying the key count toward its `n_valid` — an episode that returned
`-1` (no plan) or errored never produced an info dict to read. Bool keys report
`key: hits/n_valid`; int/float keys (counters like `double_service_count`) report
`sum`, `mean` and `nonzero hits/n_valid` — a counter is never collapsed through
`bool()`, where a count of 2 would masquerade as one hit.

## `--log-dir`: a text trace per episode

    python -m utils.mikasa_oracle.evaluate_planner -p season_dish_planner -n 5 \
        --log-dir logs/season --obs-mode state

Each episode then writes `<log-dir>/seed_<seed>/` — an events `.jsonl` and one
`_trajectory.csv` per tracked object. `utils.mikasa_oracle.planner_log` documents how to
read them; the short version is that this is the artefact an LLM debugging a
planner should be handed, because CaP-X measures raw frames in-context as *worse*
than no images and structured text as better than both (`docs/coding-agent-primer.md`
§2, §15(b)).

**Why the sweep resets before calling the planner when logging is on.** Trajectories
need object handles, and handles only exist after a reset — but the planner owns its
own reset (`template_planner.py`), which happens inside the call, too late to
register anything. So with `--log-dir` the sweep resets once itself, registers the
task's actors, and then hands the wrapper to the planner. Two consequences worth
knowing:

- The planner's own reset may pass `reconfigure=True`, which rebuilds the scene and
  makes those handles stale. `PlannerLogger._live_handle` re-resolves every handle
  by name on each write, which is the entire reason that method exists.
- That extra reset is a real cost: the episode is not bit-identical to the same seed
  run without `--log-dir`. `seed_everything` is called *after* it, so the RNG state
  entering the planner is the same either way, but the simulator has been reset one
  more time — and this scene is not restored between resets (see above). Compare
  logged runs with logged runs.
"""

from __future__ import annotations

import argparse
import inspect
import sys
import time
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

import gymnasium as gym

# Already in sys.modules: importing utils.mikasa pulls this module in transitively,
# so the top-level import costs nothing and keeps run_sweep's seam default honest.
from mani_skill.utils.wrappers.record import RecordEpisode

import mani_skill  # noqa: F401  (registers my_scenes and utils.mikasa)
from utils.mikasa_oracle.planner_log import PlannerLogger
from utils.mikasa.seeding import seed_everything


def planner_module_name(name: str) -> str:
    """Map a CLI planner name to its module path.

    Kept separate from the import so it can be checked without pulling a planner
    into sys.modules — importing one needs mplib, and it would also defeat
    `test_planners_are_not_imported_eagerly`.

    Example:
        >>> planner_module_name("burner_planner")
        'planners.burner_planner'
        >>> planner_module_name("planners/burner_planner.py")
        'planners.burner_planner'
    """
    module = name.replace("/", ".").removesuffix(".py").strip(".")
    if module.startswith("planners."):
        return module
    return f"planners.{module.removeprefix('planners.')}"


def load_planner(name: str):
    """Resolve a planner module to its entry point.

    Two spellings exist in this repo and both are supported: the inherited planners
    export `planning(env, seed)`, the ones written since export
    `solve(env, seed=None, debug=False, vis=False)` — the upstream contract that
    `template_planner.py` documents. `planning` wins when a module has both, so a
    module keeping a legacy wrapper around a new `solve` still behaves as before.

    Args:
        name (str): the CLI planner name, in any spelling `planner_module_name`
            accepts.

    Returns:
        Callable: the module's `planning`, or its `solve`.

    Raises:
        AttributeError: the module has neither.

    Example:
        >>> load_planner("season_dish_planner")   # doctest: +SKIP
        <function solve at 0x...>
    """
    mod = import_module(planner_module_name(name))
    return getattr(mod, "planning", None) or getattr(mod, "solve")


def snapshot_info(info: dict) -> dict:
    """A copy of an episode's last `info` whose values cannot change under later resets.

    Why: a task's `evaluate()` hands back its *live* state tensors (ManiSkill
    convention — `info["first_decision_ok"]` is `self.first_decision_ok` itself),
    and the next episode's `_initialize_episode` zeroes them **in place**. A
    sweep that keeps the info dicts of ten episodes and tallies them at the end
    would then read the first nine through the tenth episode's reset — measured
    on `MikasaStationChecklist-v0` (T4, 2026-08-18): ten successes tallied as
    `first_decision_ok: 1/10`, because only the last dict still held its value.

    The guarantee, exactly: **top-level** tensors are cloned to CPU and top-level
    arrays copied; every other value — scalars, strings, and any nested container
    (a dict or list of tensors) — is kept as is and stays aliased. The memory
    tasks' `info` is flat, which is what `--info-keys` reads; a nested value
    would need its own copy.

    Args:
        info: the info dict of a gym 5-tuple.

    Returns:
        dict: same keys; top-level tensors/arrays detached, the rest as given.

    Example:
        >>> import torch
        >>> live = torch.tensor([True]); snap = snapshot_info({"ok": live})
        >>> live[0] = False; bool(snap["ok"][0])
        True
    """
    out = {}
    for k, v in info.items():
        if hasattr(v, "detach") and hasattr(v, "clone"):
            out[k] = v.detach().clone().cpu()
        elif hasattr(v, "copy") and hasattr(v, "shape"):
            out[k] = v.copy()
        else:
            out[k] = v
    return out


def _scalar(value):
    """Unwrap the num_envs=1 batch and any tensor: `[True]`, `tensor([2])` -> `True`, `2`."""
    try:
        if hasattr(value, "__len__") and not isinstance(value, (str, dict)) and len(value) == 1:
            value = value[0]
    except TypeError:  # a 0-d tensor defines __len__ and refuses it
        pass
    if hasattr(value, "item"):  # torch tensor / numpy scalar -> Python scalar
        value = value.item()
    return value


def classify_result(result) -> str:
    """Sort one episode's return value into `success | no_plan | truncated | missed`.

    Pure, and the single place the sorting happens: `run_sweep`'s buckets and
    `episode_verdict` are both derived from it. The precedence is
    success > no_plan > truncated > missed — an episode that reaches success on the
    very step the horizon truncates it is one hit, never a hit *and* a truncation.

    Three return shapes are in play and `bool()` is wrong for two of them. `-1` is
    the "no plan found" sentinel and `bool(-1)` is `True`; a `solve` planner's gym
    5-tuple is a non-empty tuple, so `bool(...)` is `True` for that too, success or
    not. Reporting either as a success is exactly the failure AGENTS.md records
    against the inherited TakeItBack planner: a sweep that printed 100% without
    simulating anything. A 5-tuple whose info carries no `success` key is read the
    same way: not a success, `truncated` when its truncated flag is set, `missed`
    otherwise.

    `-1` means the planner gave up before a decision — planning or grasp failure
    (D6). A physical miss returns the last 5-tuple and lands in `missed`; the
    oracles narrow their `-1` to that semantics in T3–T5.

    Args:
        result: the planner's return value — `-1`, a gym 5-tuple (`truncated` at
            index 3, `info` at index 4), or the tensor/bool the inherited
            `planning()` returns.

    Returns:
        str: one of `"success"`, `"no_plan"`, `"truncated"`, `"missed"`.

    Example:
        >>> classify_result(-1)
        'no_plan'
        >>> classify_result((None, 0.0, False, False, {"success": [True]}))
        'success'
        >>> classify_result((None, 0.0, False, True, {"success": [False]}))
        'truncated'
        >>> classify_result((None, 0.0, False, True, {"success": [True]}))
        'success'
        >>> classify_result((None, 0.0, False, False, {"success": [False]}))
        'missed'
    """
    if isinstance(result, (int, float)) and not isinstance(result, bool):
        if int(result) == -1:
            return "no_plan"
    if isinstance(result, tuple) and len(result) == 5 and isinstance(result[-1], dict):
        info = result[-1]
        if "success" in info and bool(_scalar(info["success"])):
            return "success"
        if bool(_scalar(result[3])):
            return "truncated"
        return "missed"
    return "success" if bool(result) else "missed"


def episode_verdict(result) -> tuple[bool, bool]:
    """Read one episode's outcome out of whatever the planner returned.

    A thin wrapper over `classify_result` — the sorting logic lives there, this
    keeps the two-flag shape older callers and tests read.

    Args:
        result: the planner's return value — `-1`, a gym 5-tuple, or the tensor/bool
            the inherited `planning()` returns.

    Returns:
        tuple[bool, bool]: `(success, plan_failed)`. `plan_failed` marks the `-1`
        sentinel, which is a miss of a different kind and is counted separately.

    Example:
        >>> episode_verdict(-1)
        (False, True)
        >>> episode_verdict((None, 0.0, False, False, {"success": [True]}))
        (True, False)
        >>> episode_verdict((None, 0.0, False, False, {"success": [False]}))
        (False, False)
        >>> episode_verdict(True)
        (True, False)
    """
    verdict = classify_result(result)
    return verdict == "success", verdict == "no_plan"


def tally_info_keys(infos, keys) -> dict[str, tuple[int, int] | float]:
    """Tally requested info keys over the episodes that produced an info dict.

    Pure: `run_sweep` collects the info dict of every episode that returned a gym
    5-tuple (a `-1` or an errored episode produced none) and hands them here. Per
    key, `n_valid` counts the infos actually carrying it — an absent key is absent,
    not zero.

    Bool keys tally as hits: `per_key[k] = (hits, n_valid)`. Int/float keys are
    counters and are never collapsed through `bool()` — `bool(2)` would turn two
    double-services into one hit — so they tally as `per_key[k] = (nonzero, n_valid)`
    plus `per_key[f"{k}_sum"]` and `per_key[f"{k}_mean"]` as floats.

    Args:
        infos: iterable of per-episode `info` dicts (values may be batched tensors;
            num_envs=1 throughout).
        keys: iterable of key names, e.g. from `--info-keys a,b`.

    Returns:
        dict[str, tuple[int, int] | float]: the `SweepResult.per_key` mapping.

    Example:
        >>> tally_info_keys([{"ok": True, "n": 2}, {"ok": False}], ["ok", "n"])
        {'ok': (1, 2), 'n': (1, 1), 'n_sum': 2.0, 'n_mean': 2.0}
    """
    per_key: dict[str, tuple[int, int] | float] = {}
    for key in keys:
        values = [_scalar(info[key]) for info in infos if key in info]
        n_valid = len(values)
        # Over all values, not values[0]: a key that is a bool in one episode and
        # a count in another is a counter, whichever episode came first.
        numeric = any(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in values
        )
        if numeric:
            total = float(sum(values))
            per_key[key] = (sum(1 for v in values if v != 0), n_valid)
            per_key[f"{key}_sum"] = total
            per_key[f"{key}_mean"] = total / n_valid
        else:
            per_key[key] = (sum(1 for v in values if v), n_valid)
    return per_key


def _format_info_keys(per_key: dict, keys) -> list[str]:
    """One printable line per requested key, in the order they were asked for."""
    lines = []
    for key in keys:
        if f"{key}_sum" in per_key:
            nonzero, n_valid = per_key[key]
            lines.append(
                f"{key}: sum={per_key[f'{key}_sum']:g} "
                f"mean={per_key[f'{key}_mean']:.3f} nonzero {nonzero}/{n_valid}"
            )
        else:
            hits, n_valid = per_key[key]
            lines.append(f"{key}: {hits}/{n_valid}")
    return lines


def accepts_scene_idx(env_id: str) -> bool:
    """Only some envs take one; TakeItBack pins its kitchen with FIXTURE_SEED.

    Example:
        >>> accepts_scene_idx("MyRoboCasa-v1")            # doctest: +SKIP
        True
        >>> accepts_scene_idx("MyRoboCasa_TakeItBack-v1")  # doctest: +SKIP
        False
    """
    from mani_skill.utils.registration import REGISTERED_ENVS

    spec = REGISTERED_ENVS.get(env_id)
    if spec is None:
        return False
    return "scene_idx" in inspect.signature(spec.cls.__init__).parameters


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scene", "-s", default="MyRoboCasa-v1")
    p.add_argument(
        "--planner",
        "-p",
        default="cabinet_retrieval_planner",
        help="module under planners — e.g. myrobocasa_planner, "
        "season_dish_planner, station_checklist_planner, burner_planner. Any "
        "module exporting `planning` or `solve` works; there is no registry to "
        "add a new one to (see planner_module_name)",
    )
    p.add_argument("-n", "--num-episodes", type=int, default=100)
    p.add_argument("--start-seed", type=int, default=100)
    p.add_argument(
        "--seed-step",
        type=int,
        default=1,
        help="0 repeats one seed, which is how the non-reproducibility above was measured",
    )
    p.add_argument(
        "--scene-idx",
        type=int,
        default=None,
        help="pin the kitchen; taken by MyRoboCasa-v1 and by the three memory tasks "
        "(an env without the argument says so and ignores it). Omitting it leaves "
        "each env's own default: a random draw for MyRoboCasa-v1, kitchen 102 "
        "(scene_idx=0) for the memory tasks",
    )
    p.add_argument("--render-mode", default="rgb_array", choices=["rgb_array", "human"])
    p.add_argument(
        "--obs-mode",
        default="rgb",
        help="env obs_mode; use `state` on a CPU-only machine, where rendering RGB "
        "every step costs more than the planner does (default: rgb)",
    )
    p.add_argument(
        "--control-mode",
        default="pd_joint_pos",
        choices=("pd_joint_pos", "pd_joint_delta_pos"),
        help="the env's control mode; the Fetch solver composes its actions for either "
             "(pd_joint_delta_pos is the VLA recording format: every slot in [-1, 1])",
    )
    p.add_argument(
        "--sim-backend",
        default=None,
        help="passed to gym.make when given (e.g. cpu, gpu); default leaves ManiSkill's",
    )
    p.add_argument(
        "--render-backend",
        default=None,
        help="passed to gym.make when given; default leaves ManiSkill's",
    )
    p.add_argument(
        "--blind",
        action="store_true",
        help="run the planner's memory-free control arm over this sweep — passes "
        "blind=True to solve(). Only for planners that declare the parameter "
        "(burner_planner, season_dish_planner, station_checklist_planner); "
        "anything else is an error, not a silently sighted run",
    )
    p.add_argument(
        "--video-dir",
        default=None,
        help="record one mp4 per episode into this directory, named "
        "ep{i:03d}_seed{seed}_{ok|fail}[_blind] (RecordEpisode inside the sweep; "
        "PlannerLogger stays outside). In the container pair it with "
        "--render-backend cpu; see the module docstring",
    )
    p.add_argument(
        "--traj-dir",
        default=None,
        help="save one .h5 + .json of actions and per-step env states into this "
        "directory (no pixels). Use it where the robot cannot be rendered — the "
        "amd64 container — and replay elsewhere: `python -m "
        "mani_skill.trajectory.replay_trajectory --traj-path <h5> --use-env-states "
        "--save-video --allow-failure`. Can be combined with --video-dir or used "
        "alone",
    )
    p.add_argument(
        "--render-size",
        type=int,
        default=512,
        help="human render camera width=height for --video-dir (default: 512; the "
        "scene's own 2048 would buffer every frame in host RAM)",
    )
    p.add_argument(
        "--max-steps-per-video",
        type=int,
        default=None,
        help="RecordEpisode's mid-episode auto-flush cap. Default None: the memory "
        "tasks stop at their horizon, which already bounds the clip. Set it for an "
        "env with no TimeLimit (MyRoboCasa-v1); anonymous chunks (0.mp4, ...) are "
        "then flushed mid-episode and the named clip is the tail",
    )
    p.add_argument(
        "--info-keys",
        default=None,
        help="comma-separated info keys to tally over episodes that returned a "
        "5-tuple carrying the key: bool keys as hits/n_valid; int/float keys "
        "(counters) as sum, mean and nonzero hits/n_valid — never through bool(). "
        "Episodes that returned -1 or errored carry no info and are excluded",
    )
    p.add_argument(
        "--log-dir",
        default=None,
        help="write a per-episode text trace to <log-dir>/seed_<seed>/ "
        "(events.jsonl + trajectory CSVs). Off by default; see the module docstring "
        "for what it costs and utils.mikasa_oracle.planner_log for how to read it",
    )
    p.add_argument(
        "--log-freq",
        type=int,
        default=10,
        help="sim steps between trajectory rows, with --log-dir (default: 10)",
    )
    return p.parse_args(argv)


@dataclass
class SweepResult:
    """What one sweep measured, in the shape `colab.summary_table` reads (K16).

    `per_key` is `tally_info_keys`' mapping for the requested `--info-keys`: bool
    key -> `(hits, n_valid)`; counter key -> `(nonzero, n_valid)` plus the
    `<key>_sum` / `<key>_mean` floats.

    Example:
        >>> SweepResult(task="MikasaBurner-v0", planner="burner_planner",
        ...             blind=False, seeds=[3], hits=1, total=1, misses=0,
        ...             plan_failures=0, errors=0, truncated=0, per_key={}).hits
        1
    """

    task: str
    planner: str
    blind: bool
    seeds: list[int]
    hits: int
    total: int
    misses: int
    plan_failures: int
    errors: int
    truncated: int
    per_key: dict[str, tuple[int, int] | float]


def run_sweep(args, *, make_env=gym.make, recorder_cls=RecordEpisode) -> SweepResult:
    """Run the sweep `parse_args` described and return its tallies.

    The body behind `main` — split out so `colab.run_sweep` (T6) can call it with
    parsed args, and so tests can inject the two things a host cannot build for
    real: `make_env` (a kitchen) and `recorder_cls` (RecordEpisode writing mp4s).
    Prints the per-episode lines and the RESULTS block as it goes; success is
    always quoted as hits/N, never as a percentage.

    With `--video-dir` the recorder wraps the env inside the sweep and a
    `PlannerLogger` (with `--log-dir`) wraps the recorder. The flush is called on
    the recorder variable itself — gymnasium 1.x wrappers do not forward
    attributes, so reaching `flush_video` through an outer wrapper works in the
    container (0.29) and breaks on this Mac.

    Args:
        args (argparse.Namespace): from `parse_args`.
        make_env: `gym.make`-shaped callable building the env.
        recorder_cls: `RecordEpisode`-shaped callable wrapping it when
            `--video-dir` is set.

    Returns:
        SweepResult: the sweep's tallies; episodes sort per `classify_result`.

    Example:
        >>> args = parse_args(["-s", "MikasaBurner-v0", "-p", "burner_planner",
        ...                    "-n", "10", "--start-seed", "0", "--obs-mode", "state"])
        >>> run_sweep(args).hits                              # doctest: +SKIP
        7
    """
    planning_fn = load_planner(args.planner)

    # Checked once, before the env is built: a blind sweep that quietly ran sighted
    # would report the memory floor as the motor rate and look like a passing task.
    planner_kwargs: dict = {}
    if args.blind:
        if "blind" not in inspect.signature(planning_fn).parameters:
            raise SystemExit(
                f"--blind: {args.planner} has no blind arm (its entry point takes no "
                "`blind` parameter). The memory tasks' oracles declare "
                "solve(env, seed, debug, vis, blind=False); the inherited planners do not."
            )
        planner_kwargs["blind"] = True

    make_kwargs = dict(
        num_envs=1,
        render_mode=args.render_mode,
        obs_mode=args.obs_mode,
        robot_uids="mikasa_ds_fetch",
        control_mode=args.control_mode,
    )
    # Only when asked: passing sim_backend=None is not the same as not passing it,
    # and the defaults here have to stay exactly what they were.
    for flag, value in (("sim_backend", args.sim_backend), ("render_backend", args.render_backend)):
        if value is not None:
            make_kwargs[flag] = value
    if args.scene_idx is not None:
        if not accepts_scene_idx(args.scene):
            print(f"{args.scene} does not take scene_idx; ignoring it.", file=sys.stderr)
        else:
            make_kwargs["scene_idx"] = args.scene_idx
    if args.video_dir is not None:
        make_kwargs["human_render_camera_configs"] = dict(
            width=args.render_size, height=args.render_size
        )

    env = make_env(args.scene, **make_kwargs)
    recorder = None
    if args.video_dir is not None or args.traj_dir is not None:
        # save_on_reset=False on purpose: the planner owns its reset (it happens
        # inside the planner call), and the sweep names each clip itself once the
        # verdict is known. Do not "harmonize" with the planners' own __main__
        # recorders — there the env owns the reset and save_on_reset=True is right.
        #
        # The two outputs are independent. `--video-dir` renders here, which is only
        # worth doing on a machine that draws the robot correctly — the amd64
        # container does not (2026-08-18: lavapipe renders the kitchen right and the
        # robot as a flattened pile). `--traj-dir` writes actions plus per-step
        # `get_state_dict()` and no pixels, so the episode can be replayed and
        # rendered elsewhere: `python -m mani_skill.trajectory.replay_trajectory
        # --traj-path <h5> --use-env-states --save-video --allow-failure`, which
        # restores through `set_state_dict` and therefore needs no determinism.
        recorder = recorder_cls(
            env,
            output_dir=str(args.video_dir or args.traj_dir),
            save_trajectory=args.traj_dir is not None,
            save_video=args.video_dir is not None,
            video_fps=30,
            save_on_reset=False,
            max_steps_per_video=args.max_steps_per_video,
            source_type="motionplanning",
        )
        env = recorder
    print(
        f"=== {args.planner}{' [BLIND]' if args.blind else ''} on {args.scene}, "
        f"scene_idx={make_kwargs.get('scene_idx')}, "
        f"{args.num_episodes} episode(s) from seed {args.start_seed} step {args.seed_step} ===",
        flush=True,
    )
    if args.video_dir is not None:
        print(
            f"recording to {args.video_dir}/ep*.mp4 at "
            f"{args.render_size}x{args.render_size}",
            flush=True,
        )

    log_root = Path(args.log_dir) if args.log_dir else None
    # The planner name can arrive as `planners/foo.py`; the log's filename stem must
    # not turn into a path. The module's last component is what identifies the run.
    log_name = planner_module_name(args.planner).rsplit(".", 1)[-1]
    if log_root is not None:
        print(f"logging to {log_root}/seed_<seed>/ every {max(1, args.log_freq)} steps", flush=True)

    info_keys = [k.strip() for k in (args.info_keys or "").split(",") if k.strip()]
    seeds: list[int] = []
    verdicts: list[str] = []
    infos: list[dict] = []
    video_failures = 0
    traj_failures = 0
    for i in range(args.num_episodes):
        seed = args.start_seed + i * args.seed_step
        seeds.append(seed)

        started = time.time()
        note = ""
        logger = None
        verdict = "missed"
        # The logging setup is inside the guard on purpose: one episode failing to
        # set up must cost that episode, not the other 99 and their results.
        try:
            target = env
            if log_root is not None:
                logger = PlannerLogger(
                    env, run_dir=log_root / f"seed_{seed}", name=log_name, log_freq=args.log_freq
                )
                logger.log_event("episode_start", seed=seed, scene=args.scene)
                # Handles exist only after a reset, and the planner's own reset
                # happens inside the call — too late to register anything. See the
                # docstring: this costs one extra reset, and _live_handle covers
                # the staleness it can introduce.
                env.reset(seed=seed)
                target = logger

            # Per episode, not once per run: mplib's generator is global and advances
            # as the planner works, so otherwise an episode depends on its
            # predecessors. After the logging reset above, so the planner starts from
            # the same RNG state whether or not --log-dir was given.
            seed_everything(seed)

            if logger is not None:
                logger.track_task_actors(env.unwrapped)

            result = planning_fn(target, seed, **planner_kwargs)
            verdict = classify_result(result)
            if isinstance(result, tuple) and len(result) == 5 and isinstance(result[-1], dict):
                # Snapshot: the task's next reset would otherwise rewrite this
                # dict's live tensors in place (see snapshot_info).
                infos.append(snapshot_info(result[-1]))
            if verdict == "no_plan":
                note = "  no plan found (-1)"
                if logger is not None:
                    # Otherwise the trace shows a false verdict and no reason: the
                    # planner only logs its own `-1` if it used log_motion, and the
                    # inherited ones do not.
                    logger.log_event("error", "planner returned -1 (no plan found)")
            elif verdict == "truncated":
                note = "  stopped by the horizon"
        except Exception as exc:  # upstream books this as a failure; say which kind
            verdict = "errored"
            note = f"  {type(exc).__name__}: {exc}"[:180]
            if logger is not None:
                logger.log_event("error", f"{type(exc).__name__}: {exc}")
        finally:
            # In a finally so an errored episode's frames never leak into the next
            # clip; on the recorder variable, never through an outer wrapper (see
            # the docstring). The episode index is in the name because
            # --seed-step 0 repeats one seed.
            if recorder is not None:
                clip = f"ep{i:03d}_seed{seed}_{'ok' if verdict == 'success' else 'fail'}"
                if args.blind:
                    clip += "_blind"
                # Only when a video was asked for. Upstream's flush_video would
                # return at its first line anyway (nothing captured when
                # save_video=False, record.py:775-778), but calling a video flush
                # in a trajectory-only run reads as if pixels were expected.
                if args.video_dir is not None:
                    try:
                        recorder.flush_video(name=clip)
                    except Exception as exc:  # an encoder failure costs the clip, not the sweep
                        video_failures += 1
                        note += f"  video: {type(exc).__name__}: {exc}"[:180]
                        if logger is not None:
                            logger.log_event("error", f"video: {type(exc).__name__}: {exc}")
                # Its own try for the same reason as the video's: one unwritable
                # episode must not abort the sweep or leave the .h5 half-written.
                # No `name=` — the trajectories of a run share one file, keyed
                # `traj_<i>`, and the .json carries the reset seed per episode.
                if args.traj_dir is not None:
                    try:
                        recorder.flush_trajectory()
                    except Exception as exc:
                        traj_failures += 1
                        note += f"  traj: {type(exc).__name__}: {exc}"[:180]
                        if logger is not None:
                            logger.log_event("error", f"traj: {type(exc).__name__}: {exc}")
                # Drop whatever the flush left behind: RecordEpisode clears the
                # buffer only *after* a successful encode (record.py:796-804) and
                # not at all on its one-frame early return (record.py:775-778) —
                # without this, a failed episode's frames head the next clip and
                # the buffer grows for every later failure (~0.79 MB/frame at
                # 512²). Attribute reset rather than flush_video(save=False)
                # because the save=False path takes the same early returns and
                # cannot clear a one-frame buffer; both names exist and are the
                # buffer on b14 and b22 (checked in both interpreters).
                if getattr(recorder, "render_images", None):
                    recorder.render_images = []
                    recorder._video_steps = 0
        elapsed = time.time() - started

        ok = verdict == "success"
        if logger is not None:
            logger.log_event("verdict", success=ok)
            logger.finish()

        verdicts.append(verdict)
        print(
            f"EP {i + 1}/{args.num_episodes} seed={seed}  "
            f"{'SUCCESS' if ok else 'FAILED '}  {elapsed:6.1f}s  "
            f"running={verdicts.count('success')}/{len(verdicts)}{note}",
            flush=True,
        )

    env.close()
    hits = verdicts.count("success")
    total = len(verdicts)
    misses = verdicts.count("missed")
    plan_failures = verdicts.count("no_plan")
    errors = verdicts.count("errored")
    truncated = verdicts.count("truncated")
    per_key = tally_info_keys(infos, info_keys)
    print(
        f"\n=== RESULTS === success {hits}/{total}  (missed {misses}, "
        f"no plan {plan_failures}, errored {errors}, truncated {truncated})"
    )
    if video_failures:
        print(f"video: {video_failures} clip(s) failed to encode — see the EP lines above")
    if traj_failures:
        print(f"traj: {traj_failures} episode(s) failed to write — see the EP lines above")
    if args.traj_dir is not None:
        print(
            f"trajectories in {args.traj_dir}/ — replay and render them somewhere that "
            "draws the robot: python -m mani_skill.trajectory.replay_trajectory "
            f"--traj-path {args.traj_dir}/<file>.h5 --use-env-states --save-video "
            "--allow-failure"
        )
    for line in _format_info_keys(per_key, info_keys):
        print(line)
    if log_root is not None:
        print(f"traces in {log_root}/ — see utils.mikasa_oracle.planner_log for how to read them")
    if args.seed_step != 0 and total > 1:
        print(
            f"Quote this as {hits}/{total}, not as a rate: the scene is not restored "
            "between episodes, so the figure falls as N grows. See the module docstring."
        )
    return SweepResult(
        task=args.scene,
        planner=args.planner,
        blind=bool(args.blind),
        seeds=seeds,
        hits=hits,
        total=total,
        misses=misses,
        plan_failures=plan_failures,
        errors=errors,
        truncated=truncated,
        per_key=per_key,
    )


def main(argv=None) -> int:
    run_sweep(parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
