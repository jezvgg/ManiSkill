# Repository Guidelines

## Project Overview

This repository is a **fork of ManiSkill 3** repurposed as a **VLA (Vision-Language-Action) memory benchmark** for household robotic tasks. The benchmark tests whether VLA models can remember and execute multi-step manipulation sequences in kitchen environments (RoboCasa scenes). The upstream ManiSkill engine (SAPIEN 3.0+ / NVIDIA PhysX CUDA) is used as-is for GPU-parallelized simulation and Vulkan rendering.

> **Edit boundary**: All development happens exclusively in `my_scenes/` (benchmark scene definitions), `planners/` (motion-planning solvers), and `utils/` (shared utility helpers). **NEVER modify files outside these three directories.** Everything else is upstream ManiSkill infrastructure and must remain untouched.

---

## Architecture & Data Flow

### High-Level Simulation Lifecycle

```
+-------------------------------------------------------------------------+
|                           Gymnasium API / User                          |
|             (gym.make, env.step(action), env.reset(), vector envs)      |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                BaseEnv (mani_skill.envs.sapien_env.BaseEnv)            |
|  - Task logic, step/reset loops, reward & evaluation metrics            |
|  - Manages robot agents (BaseAgent) & sensor pipelines                  |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|               ManiSkillScene (mani_skill.envs.scene)                   |
|  - Manages N sub-scenes (sapien.Scene) for parallel GPU simulation      |
|  - GPU Lifecycle: _gpu_fetch_all() <-> PhysX GPU <-> _gpu_apply_all()   |
|  - Pairwise contact impulse queries & render camera management          |
+-------------------------------------------------------------------------+
         /                           |                           \
        v                            v                            v
+---------------+          +--------------------+       +------------------+
|     Actor     |          |    Articulation    |       |   BaseSensor     |
| (Rigid Bodies)|          |  (Robots/Joints)   |       | (Camera / Depth) |
+---------------+          +--------------------+       +------------------+
        |                            |                            |
        +----------------------------+----------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                  PhysX CUDA Buffers / GPU Memory                        |
|  (px.cuda_rigid_body_data, px.cuda_articulation_qpos, qvel, qf, etc.)   |
+-------------------------------------------------------------------------+
```

### Core Architecture Concepts

1. **Dual Backends**:
   - `sim_backend`: Supports `physx_cpu` (`num_envs=1`) and `physx_cuda` (`num_envs > 1`).
   - `render_backend`: Supports `sapien_cuda` (Vulkan/CUDA hardware rendering) and `cpu`.
2. **GPU Synchronization Loop**:
   - `_gpu_fetch_all()`: Updates PyTorch CUDA tensor views/structs from simulation buffers in CUDA memory.
   - Controller update: Target actions are converted to joint/pose drives.
   - `_gpu_apply_all()`: Writes updated user/controller target tensors back to PhysX CUDA buffers prior to `px.step()`.
3. **Core Struct Abstractions**:
   - `BaseStruct`: Generic dataclass managing SAPIEN objects across parallel sub-scenes (`_objs`, `_scene_idxs`, `device`).
   - `Actor`: Batched wrapper for rigid bodies (`mani_skill/utils/structs/actor.py`), exposing PyTorch tensor getters/setters (`pose`, `linear_velocity`, `set_pose`).
   - `Articulation`: Batched wrapper for articulated robots and objects (`mani_skill/utils/structs/articulation.py`), providing zero-copy tensor properties (`qpos`, `qvel`, `qf`, `target_qpos`).
   - `Link` & `ArticulationJoint`: Sub-structs providing fine-grained link pose/velocity and joint drive controls.
4. **Agent & Control System**:
   - `BaseAgent`: Interface for articulated robots loaded via URDF or MJCF. Manages joint/EE controllers (`PDJointPosController`, `PDEEPoseController`, etc.), sensors, and keyframes.
5. **Zero-Copy Vectorization**:
   - `ManiSkillVectorEnv`: Native Gymnasium `VectorEnv` wrapper passing PyTorch GPU tensors directly without CPU/GPU memory transfer overhead. Supports partial sub-scene auto-resets via boolean masks (`_reset_mask`).

---

## Key Directories

### Active (editable) directories

- `my_scenes/`: **Benchmark scene definitions.** Custom `BaseEnv` subclasses registered as Gymnasium environments for the VLA memory benchmark. Each file defines a RoboCasa-based kitchen task with object placement, evaluation logic, reward shaping, and camera configs.
  - `my_robocasa.py` — `MyRoboCasaScene` (`MyRoboCasa-v1`): base kitchen scene with a Fetch robot, bowl + cup on a counter, and a dense reward for grasping/placing.
  - `my_robocasa_takeitback.py` — `MyRoboCasaSceneTakeItBack` (`MyRoboCasa_TakeItBack-v1`): extended task requiring the robot to move a cup from counter to sink and back, testing spatial memory.
  - `__init__.py` — re-exports both scene classes and helpers (`get_actor_size`, `degree_to_quanterion`).
- `planners/`: **Motion-planning solvers** that drive the benchmark scenes end-to-end using `FetchMotionPlanningSapienSolver` (mplib). Each planner orchestrates grasp planning, IK, path execution, and gripper control for its corresponding scene. Runnable as standalone scripts (`python planners/<file>.py`).
  - `myrobocasa_planner.py` — solver for `MyRoboCasa-v1`.
  - `myrobocasa_takeitback_planner.py` — solver for `MyRoboCasa_TakeItBack-v1` (includes smooth base movement for multi-waypoint tasks).
- `utils/`: **Shared utility helpers.** Custom helper libraries for common tasks.
  - `planners_utils.py` — Contains shared parameters and functions for motion planners (e.g. torso lowering, safe base backing, screw/planar alignment).
  - **Development Guideline**: When creating a new motion planner, agents MUST inspect `utils/planners_utils.py` and existing planners in `planners/` to reuse these custom helper functions rather than duplicating operations or rewriting simulation loops by hand.

### Upstream ManiSkill (read-only reference, DO NOT edit)

- `mani_skill/envs/`: Environment task definitions, `BaseEnv` (`sapien_env.py`), scene manager `ManiSkillScene` (`scene.py`).
- `mani_skill/agents/`: Robot agent implementations (`robots/fetch`, `panda`, `xarm6`, etc.) and controllers.
- `mani_skill/utils/structs/`: PyTorch CUDA tensor struct wrappers (`actor.py`, `articulation.py`, `link.py`, `pose.py`).
- `mani_skill/utils/building/`: Procedural asset loading (`actor_builder.py`, `urdf_loader.py`, `mjcf_loader.py`).
- `mani_skill/utils/scene_builder/`: Scene builders for RoboCasa kitchen fixtures, ReplicaCAD.
- `mani_skill/sensors/`: Vulkan rendering camera sensors and point cloud generators.
- `mani_skill/vector/`: Gymnasium-compatible vector environment wrappers.
- `mani_skill/trajectory/`: Dataset recording and trajectory replay utilities.
- `tests/`, `scripts/`, `examples/`, `mshab/`, `docs/source/`: Tests, data generation, baselines, benchmarks, documentation.

---

## Development Commands

### Environment Setup & Verification
```bash
# Install package in editable mode with development extras (ALWAYS use uv, NEVER standard pip)
uv pip install -e ".[dev]"

# Run random action environment test with interactive GUI
uv run python -m mani_skill.examples.demo_random_action -e PickCube-v1
```

### Testing
```bash
# Run CPU environment tests
pytest tests/test_envs.py
pytest -m "not gpu_sim"

# Run GPU simulation tests (requires NVIDIA GPU)
pytest -m gpu_sim
pytest tests/test_gpu_envs.py

# Exclude slow asset download tests
pytest -m "not slow"

# Run tests in parallel
pytest -n auto --forked tests

# Run full multi-Python Docker test matrix
./tests/run.sh
```

### Code Quality & Formatting
```bash
# Run pre-commit hooks across all files
pre-commit run --all-files

# Format code with Black (line length 88)
black mani_skill tests

# Sort imports with isort (black profile)
isort mani_skill tests

# Remove unused imports
autoflake --remove-all-unused-imports -r mani_skill tests

# Static type checking
pyright
```

---

## Code Conventions & Common Patterns

### Naming & Types
- **Classes**: `PascalCase` (`BaseEnv`, `ActorBuilder`, `PDEEPoseController`).
- **Functions/Variables**: `snake_case` (`get_obs`, `compute_dense_reward`, `set_pose`).
- **Registries**: `UPPERCASE` (`REGISTERED_ENVS`, `REGISTERED_AGENTS`).
- **Task Environment IDs**: `TaskName-v1` (e.g. `PickCube-v1`, `OpenCabinetDoor-v1`).
- **Type Annotations**: Explicit annotations preferred (`torch.Tensor`, `sapien.Pose`, `Array`, `Device`, `SimConfig`).

### GPU vs CPU Vectorization Patterns
- **Batched Tensor Operations**: When `num_envs > 1`, always perform batched operations directly on PyTorch CUDA tensors (`torch.Tensor`). Avoid CPU transfers (`.cpu()`, `.numpy()`) in env step/reward loops.
- **GPU Synchronization**: Flow MUST be: `_gpu_fetch_all()` -> action step / reward calculation -> `_gpu_apply_all()` -> `scene.px.step()`.
- **State Locking Guard**: Use the `@before_gpu_init` decorator on methods that modify immutable physical actor/joint properties before `px.gpu_init()`.
- **Partial Env Resets**: Reset specific sub-scenes in a batched vector environment using sub-scene boolean masks (`env._reset_mask`).
- **Deterministic RNG**: Use `BatchedRNG` (`mani_skill.envs.utils.randomization.batched_rng`) instead of raw `torch.rand` or `np.random` to ensure identical CPU and GPU parallel random seeding.
- **Observation Modes**: Standard keys: `"state"`, `"state_dict"`, `"sensor_data"`, `"rgb"`, `"depth"`, `"segmentation"`, `"pointcloud"`. Offscreen visual rendering is skipped when non-visual observation modes (`"state"`) are active.

---

## Important Files

- `pyproject.toml`: Build backend specification (Setuptools) and pytest marker configurations (`gpu_sim`, `slow`, `serial`).
- `setup.py`: Core setup script defining dependencies, platform-specific binaries (`mplib`, `fast_kinematics`, `SAPIEN`), dynamic versioning, and package extras (`dev`, `docs`).
- `pyrightconfig.json`: Pyright static type checker configuration.
- `mani_skill/envs/sapien_env.py`: `BaseEnv` class managing Gym interface, simulation steps, resets, GPU device initialization, observation/reward pipelines, and episode state.
- `mani_skill/envs/scene.py`: `ManiSkillScene` orchestrating N sub-scenes, PhysX CPU/GPU system backends, and GPU buffer fetch/apply cycles.
- `mani_skill/agents/base_agent.py`: `BaseAgent` abstraction for articulated robots handling URDF/MJCF loading, controller instantiation, sensor registration, and keyframes.
- `mani_skill/utils/structs/actor.py`: PyTorch CUDA tensor struct wrapping static, dynamic, and kinematic rigid bodies.
- `mani_skill/utils/structs/articulation.py`: PyTorch CUDA tensor struct wrapping multi-link articulated bodies and joint target indices.
- `mani_skill/vector/wrappers/gymnasium.py`: Zero-copy GPU vectorized environment wrapper.
- `tests/utils.py`: Shared test environment IDs, observation modes, control modes, low-memory configs, and assertion helpers (`assert_obs_equal`).

---

## Runtime/Tooling Preferences

- **Python**: Python >= 3.9 (Python 3.11 recommended with `uv.lock`).
- **Package Manager**: **NEVER use standard `pip`**. ALWAYS use `uv` for package management, dependency installation, and environment execution (`uv pip install ...`, `uv run ...`).
- **Hardware & Physics Dependencies**:
  - NVIDIA GPU with CUDA support and Vulkan ICD drivers for GPU simulation and hardware rendering (`physx_cuda` + `sapien_cuda`).
  - SAPIEN 3.0+ physics simulation engine.
- **Code Style & Formatters**:
  - Formatter: Black (line length 88), `isort` (profile `black`).
  - Cleaner: `autoflake` for unused imports.
  - Type Checker: Pyright (`basic` mode).
  - Test Framework: `pytest` with `pytest-xdist` and `pytest-forked`.

---

## Testing & QA

### Framework & Execution
- **Pytest**: Primary test suite runner supporting parallel execution with `pytest-xdist` (`pytest -n auto --forked tests`).
- **Docker Multi-Python Runner (`tests/run.sh`)**: Runs test matrices across Python versions (3.8-3.11) inside GPU-enabled Docker containers.

### Pytest Markers
- `@pytest.mark.gpu_sim`: Indicates tests requiring GPU simulation (`PhysX CUDA`). Skip on CPU-only machines with `-m "not gpu_sim"`.
- `@pytest.mark.slow`: Long-running asset download or benchmark tests. Skip with `-m "not slow"`.
- `@pytest.mark.serial`: Tests that cannot run in parallel.

### Common Testing Patterns & Assertions
- **Parametrization**: Heavy use of `@pytest.mark.parametrize` over standard test constants (`ENV_IDS`, `OBS_MODES`, `CONTROL_MODES_STATIONARY_SINGLE_ARM` from `tests/utils.py`).
- **GPU Memory Bounding**: Uses `LOW_MEM_SIM_CONFIG` (`max_rigid_patch_count=81920`, `found_lost_pairs_capacity=262144`) during GPU test environment creation to prevent CUDA OOM in CI environments.
- **Observation Assertions**: Use `assert_obs_equal(obs1, obs2)` from `tests/utils.py` for structural dict comparisons and numerical tolerance checks (`np.testing.assert_allclose`).
