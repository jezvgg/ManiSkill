"""Check that this machine can actually run the benchmark, before you wait on it.

    python -m utils.mikasa.preflight

Every requirement is checked separately and reported on its own line, because the
failure modes are otherwise indistinguishable: a missing Vulkan ICD, an mplib that
is one major version too old, and un-downloaded assets all surface as unrelated
tracebacks deep inside a run that has already burned several minutes.

Exits non-zero if anything required is missing.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

# Files SAPIEN needs to find the NVIDIA Vulkan driver. Colab images ship neither
# the ICD manifest nor the EGL vendor file.
#
# Two ICD paths because the Vulkan loader reads both, while SAPIEN
# (sapien/_vulkan_tricks.py) looks only at the /usr/share one — so a manifest
# installed just in /etc is present for the loader and missing for SAPIEN, which
# then falls back to its own bundled copy. See docs/getting-it-to-run.md §9.
VULKAN_ICD = Path("/usr/share/vulkan/icd.d/nvidia_icd.json")
VULKAN_ICD_ALT = Path("/etc/vulkan/icd.d/nvidia_icd.json")
EGL_VENDOR = Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")

# NVIDIA exports the Vulkan ICD entry points from more than one library.
# libGLX_nvidia is the GLX path and refuses to initialise where there is no X
# server; libEGL_nvidia is the headless one. Both export every vk_icd* symbol, so
# a manifest naming the wrong one loads cleanly and fails at negotiation instead —
# which surfaces much later as ErrorIncompatibleDriver. §9.
ICD_HEADLESS_LIBRARY = "libEGL_nvidia.so.0"

# The exact release this package pins; see pyproject.toml for why.
EXPECTED_MANI_SKILL = "3.0.0b22"

# mplib 0.2 moved everything: mplib.Pose, mplib.sapien_utils.SapienPlanner and
# mplib.collision_detection.fcl.FCLObject do not exist in 0.1.x. Every published
# mani_skill pins 0.1.1, so a plain install lands on the wrong one.
MPLIB_REQUIRED_ATTRS = [
    ("mplib", "Pose"),
    ("mplib.sapien_utils", "SapienPlanner"),
    ("mplib.collision_detection.fcl", "FCLObject"),
]

# (package, requirement) pairs we knowingly violate. mani_skill pins mplib to
# 0.1.1, which predates the API this benchmark's solver is written against, so
# we install 0.2.1 over it on purpose. Every other declared requirement has to
# hold — mplib's own `numpy<2.0` most of all, since violating that one makes
# its bindings segfault with no error at all.
DELIBERATE_OVERRIDES = {("mani_skill", "mplib")}

# Paths under ASSET_DIR/scene_datasets/robocasa_dataset that the cup task opens.
# Listed individually because a partial extract passes an is_dir() check on the
# dataset root and only fails later, deep inside scene building, as a bare
# FileNotFoundError. Shared with utils.mikasa.colab so the two cannot drift.
REQUIRED_ASSET_PARTS = [
    "assets/scenes/kitchen_layouts",
    "assets/scenes/kitchen_styles",
    "assets/objects/objaverse/cup/cup_2/model.xml",
    "assets/objects/objaverse/bowl/bowl_2/model.xml",
    # MikasaSeasonDish-v0. Directories, not a named instance: the season task asks
    # the registry which instances exist rather than hardcoding one, so any
    # instance will do and naming e.g. shaker_0 would invent a guarantee.
    # Note the second path — condiment_bottle's models live under `condiment`,
    # because it declares model_folders=["objaverse/condiment"]
    # (robocasa/objects/kitchen_objects.py:458-461).
    "assets/objects/objaverse/shaker",
    "assets/objects/objaverse/condiment",
]

# Asks an ICD library the first question the Vulkan loader asks it. Run out of
# process for the same reason as RENDER_PROBE: loading a graphics driver can
# abort rather than raise.
ICD_PROBE = """
import ctypes, sys
lib = ctypes.CDLL(sys.argv[1])
negotiate = lib.vk_icdNegotiateLoaderICDInterfaceVersion
negotiate.restype = ctypes.c_int
negotiate.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
version = ctypes.c_uint32(5)
print("ICD_RESULT", negotiate(ctypes.byref(version)), version.value)
"""

RENDER_PROBE = """
import gymnasium as gym
import mani_skill.envs  # noqa: F401
env = gym.make("PickCube-v1", num_envs=1, render_mode="rgb_array")
env.reset(seed=0)
img = env.render()
env.close()
print("PROBE_OK", tuple(img.shape))
"""


class Report:
    def __init__(self) -> None:
        self.failed: list[str] = []
        self.warned: list[str] = []

    def ok(self, name: str, detail: str = "") -> None:
        print(f"  \033[32mOK  \033[0m {name}" + (f" — {detail}" if detail else ""))

    def warn(self, name: str, detail: str) -> None:
        self.warned.append(name)
        print(f"  \033[33mWARN\033[0m {name} — {detail}")

    def fail(self, name: str, detail: str) -> None:
        self.failed.append(name)
        print(f"  \033[31mFAIL\033[0m {name} — {detail}")


def check_platform(r: Report) -> None:
    print("\n[ platform ]")
    r.ok("python", f"{platform.python_version()} on {platform.system()} {platform.machine()}")
    if platform.system() != "Linux":
        r.warn(
            "os",
            f"{platform.system()}: mplib and fast_kinematics publish Linux wheels only, "
            "so the planners cannot run here. Scene code still imports.",
        )


def check_gpu(r: Report) -> None:
    print("\n[ gpu ]")
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        r.fail("nvidia-smi", f"not usable ({type(exc).__name__}). No NVIDIA GPU visible.")
        return
    if out.returncode != 0:
        r.fail("nvidia-smi", f"exit {out.returncode}: {out.stderr.strip()[:200]}")
        return
    r.ok("nvidia-smi", out.stdout.strip().replace("\n", " | "))

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if len(out.stdout.strip().splitlines()) > 1 and not visible:
        r.warn(
            "CUDA_VISIBLE_DEVICES",
            "several GPUs visible and the variable is unset; ManiSkill wants exactly one. "
            "Set CUDA_VISIBLE_DEVICES=0.",
        )


def icd_negotiates(library: str) -> tuple[bool, str]:
    """Ask an ICD library to negotiate, the way the Vulkan loader does.

    The manifest existing says nothing about the driver behind it. The library it
    names can load cleanly, export every vk_icd* symbol, and still refuse to
    initialise — which is exactly what a headless libGLX_nvidia does. Only asking
    it tells the two apart.
    """
    try:
        out = subprocess.run(
            [sys.executable, "-c", ICD_PROBE, library],
            capture_output=True, text=True, timeout=120,
        )
    except subprocess.SubprocessError as exc:
        return False, f"probe could not run: {exc}"
    if "ICD_RESULT" not in out.stdout:
        if out.returncode < 0:
            return False, f"{library} killed the probe with signal {-out.returncode}"
        tail = (out.stderr or out.stdout).strip().splitlines()[-1:]
        return False, f"{library} did not load: {' '.join(tail)[:200]}"
    code, version = out.stdout.split("ICD_RESULT", 1)[1].split()[:2]
    if int(code) != 0:
        return False, f"{library} refused to negotiate (VkResult {code})"
    return True, f"{library}, loader interface v{version}"


def check_vulkan(r: Report) -> None:
    print("\n[ vulkan ]")
    if platform.system() != "Linux":
        r.warn("icd", "skipped: ICD layout check is Linux-specific")
        return

    manifest = next((p for p in (VULKAN_ICD, VULKAN_ICD_ALT) if p.is_file()), None)
    if manifest is None:
        r.fail(
            str(VULKAN_ICD),
            "missing. Rendering will fail. Install the Vulkan ICD; see the Colab "
            "section of the README.",
        )
    else:
        try:
            library = json.loads(manifest.read_text())["ICD"]["library_path"]
        except (OSError, ValueError, KeyError) as exc:
            r.fail(str(manifest), f"not readable as an ICD manifest: {exc}")
            library = None
        if library is not None:
            works, detail = icd_negotiates(library)
            if works:
                r.ok(str(manifest), detail)
                if manifest != VULKAN_ICD:
                    r.fail(
                        str(VULKAN_ICD),
                        f"missing. The working manifest is {manifest}, which the Vulkan "
                        f"loader reads but SAPIEN does not. Fix: copy it to {VULKAN_ICD}.",
                    )
            else:
                fix = ""
                if library != ICD_HEADLESS_LIBRARY:
                    spare, _ = icd_negotiates(ICD_HEADLESS_LIBRARY)
                    if spare:
                        fix = (
                            f' Fix: set "library_path" to {ICD_HEADLESS_LIBRARY}, which '
                            "does negotiate on this machine."
                        )
                r.fail(
                    str(manifest),
                    f"names a driver that will not initialise — {detail}.{fix} "
                    "See docs/getting-it-to-run.md §9.",
                )

    if EGL_VENDOR.is_file():
        r.ok(str(EGL_VENDOR))
    else:
        r.fail(
            str(EGL_VENDOR),
            "missing. Rendering will fail. Install the glvnd EGL vendor file; see "
            "the Colab section of the README.",
        )


def check_render(r: Report) -> None:
    """Render one frame in a subprocess — a broken driver segfaults rather than raising."""
    print("\n[ offscreen render ]")
    try:
        out = subprocess.run(
            [sys.executable, "-c", RENDER_PROBE],
            capture_output=True, text=True, timeout=600,
        )
    except subprocess.SubprocessError as exc:
        r.fail("render probe", f"could not run: {exc}")
        return
    if "PROBE_OK" in out.stdout:
        shape = out.stdout.split("PROBE_OK", 1)[1].strip()
        r.ok("render probe", f"rendered a frame {shape}")
        return
    if out.returncode < 0:
        r.fail(
            "render probe",
            f"killed by signal {-out.returncode} (usually a broken Vulkan driver)",
        )
    else:
        tail = (out.stderr or out.stdout).strip().splitlines()[-3:]
        r.fail("render probe", f"exit {out.returncode}: " + " / ".join(tail)[:400])


def check_mani_skill(r: Report) -> None:
    print("\n[ mani_skill ]")
    try:
        import mani_skill
    except ImportError as exc:
        r.fail("mani_skill", f"not importable: {exc}")
        return
    version = getattr(mani_skill, "__version__", "unknown")
    if version == EXPECTED_MANI_SKILL:
        r.ok("mani_skill", version)
    else:
        r.warn(
            "mani_skill",
            f"{version}, expected {EXPECTED_MANI_SKILL}. Other releases are untested "
            "and move the motionplanning modules around.",
        )
    r.ok("ASSET_DIR", str(mani_skill.ASSET_DIR))


def check_mplib(r: Report) -> None:
    print("\n[ mplib ]")
    try:
        import mplib
    except ImportError as exc:
        if platform.system() != "Linux":
            r.warn("mplib", "not installed (Linux-only); planners unavailable here")
        else:
            r.fail("mplib", f"not importable: {exc}")
        return

    version = getattr(mplib, "__version__", "unknown")
    missing = []
    for module_name, attr in MPLIB_REQUIRED_ATTRS:
        try:
            module = __import__(module_name, fromlist=[attr])
            if not hasattr(module, attr):
                missing.append(f"{module_name}.{attr}")
        except ImportError:
            missing.append(f"{module_name}.{attr}")
    if missing:
        r.fail(
            "mplib",
            f"version {version} lacks {', '.join(missing)} — this is the 0.1.x API. "
            "Fix: pip install --no-deps --force-reinstall mplib==0.2.1",
        )
    else:
        r.ok("mplib", f"{version} (0.2.x API present)")


def check_dependency_pins(r: Report) -> None:
    """Check that installed packages satisfy what the installed packages ask for.

    Normally pip guarantees this. Here it cannot: mplib has to go in with
    --no-deps to get past mani_skill's wrong `mplib==0.1.1`, and --no-deps
    discards mplib's *own* requirements at the same time. mplib 0.2.1 declares
    `numpy<2.0` and its bindings segfault when handed a numpy 2 array, with no
    error message — the process just dies.
    """
    print("\n[ dependency pins ]")
    try:
        from importlib.metadata import PackageNotFoundError, requires
        from packaging.requirements import Requirement
        from packaging.version import Version
    except ImportError as exc:
        r.warn("pins", f"cannot check: {exc}")
        return

    checked = 0
    failures_before = len(r.failed)
    for package in ["mplib", "mani_skill"]:
        try:
            declared = requires(package) or []
        except PackageNotFoundError:
            continue
        for raw in declared:
            req = Requirement(raw)
            if req.marker is not None and not req.marker.evaluate():
                continue
            if (package, req.name) in DELIBERATE_OVERRIDES:
                # Reporting this as a failure would tell the user to undo the
                # override the benchmark depends on.
                r.ok(f"{package} wants {raw}", "deliberately overridden, see README")
                continue
            try:
                installed = Version(__import__(req.name).__version__)
            except (ImportError, AttributeError, ValueError):
                continue
            checked += 1
            if installed not in req.specifier:
                r.fail(
                    f"{package} needs {raw}",
                    f"but {req.name} {installed} is installed. "
                    f"Fix: pip install '{req.name}{req.specifier}'",
                )
    if not checked:
        r.warn("pins", "nothing to check (mplib and mani_skill both absent)")
    elif len(r.failed) == failures_before:
        r.ok("pins", f"{checked} declared requirement(s) satisfied")


def check_assets(r: Report) -> None:
    print("\n[ robocasa assets ]")
    try:
        from mani_skill import ASSET_DIR
    except ImportError:
        r.fail("assets", "cannot resolve ASSET_DIR without mani_skill")
        return
    root = Path(ASSET_DIR) / "scene_datasets" / "robocasa_dataset"
    r.ok("ASSET_DIR resolves to", str(ASSET_DIR))
    if not root.is_dir():
        r.fail(
            "robocasa_dataset",
            f"missing at {root} (~3.7 GB). Fix: "
            "python -m mani_skill.utils.download_asset -y RoboCasa",
        )
        return
    absent = [
        str(root / part)
        for part in REQUIRED_ASSET_PARTS
        if not (root / part).exists()
    ]
    if absent:
        r.fail(
            "robocasa_dataset incomplete",
            f"present at {root} but missing {absent}. The archive did not extract "
            "fully. Fix: delete that directory and re-run "
            "python -m mani_skill.utils.download_asset -y RoboCasa",
        )
    else:
        r.ok("robocasa_dataset", str(root))


def check_package(r: Report) -> None:
    print("\n[ utils.mikasa ]")
    try:
        import mani_skill  # noqa: F401  (registers my_scenes and utils.mikasa)
        from mani_skill.agents.registration import REGISTERED_AGENTS
        from mani_skill.utils.registration import REGISTERED_ENVS
    except ImportError as exc:
        r.fail("import utils.mikasa", str(exc))
        return
    # The three memory tasks are listed by name, not counted: the notebook runs
    # all three in one session, and an env that failed to register surfaces there
    # as a gym.error.NameNotFound several minutes into the first sweep.
    for env_id in [
        "MyRoboCasa-v1",
        "MyRoboCasa_TakeItBack-v1",
        "MikasaBurner-v0",
        "MikasaSeasonDish-v0",
        "MikasaStationChecklist-v0",
    ]:
        if env_id in REGISTERED_ENVS:
            r.ok(f"env {env_id}")
        else:
            r.fail(f"env {env_id}", "not registered")
    if "mikasa_ds_fetch" in REGISTERED_AGENTS:
        r.ok("agent ds_fetch")
    else:
        r.fail("agent ds_fetch", "not registered")

    try:
        import imageio_ffmpeg  # noqa: F401
        r.ok("imageio-ffmpeg", "video encoding available")
    except ImportError:
        r.warn("imageio-ffmpeg", "missing; RecordEpisode cannot write mp4")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Skip the offscreen render probe (it builds a throwaway env, ~30 s)",
    )
    args = parser.parse_args(argv)

    r = Report()
    check_platform(r)
    check_gpu(r)
    check_vulkan(r)
    check_mani_skill(r)
    check_mplib(r)
    check_dependency_pins(r)
    check_assets(r)
    check_package(r)
    if args.skip_render:
        print("\n[ offscreen render ]\n  skipped (--skip-render)")
    else:
        check_render(r)

    print()
    if r.failed:
        print(f"FAILED ({len(r.failed)}): {', '.join(r.failed)}")
        return 1
    if r.warned:
        print(f"Ready, with {len(r.warned)} warning(s): {', '.join(r.warned)}")
    else:
        print("Ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
