---
name: maniskill-planner-analisis
description: Automates the evaluation and failure diagnostics of the ManiSkill motion planner. Executes multi-seed motion planning benchmarks, orchestrates sub-agents to analyze individual failure logs via a dedicated log-analysis skill, and aggregates all sub-agent findings into a comprehensive diagnostic report with high-level resolution strategies.

# ManiSkill Motion Planner Failure Analysis Skill

## Inputs
* `planner`: Name of the ManiSkill planner to evaluate.
* `num_runs`: Total number of seeds/episodes to run.

---

## Execution Workflow

### Step 1: Execute Benchmark Suite
Execute the multi-seed benchmark script to generate execution traces across distinct scenes/seeds.

1. Run the benchmark controller:
   ```bash
   ./analyze.sh {planner} {num_runs}
   ```

2. Parse the output summary to extract pathes to logs with failed planners.

---

### Step 2: Spawn Log Analysis Sub-Agents

For every identified failure directory:

1. Spawn a dedicated sub-agent for the individual failed seed.
2. Instruct the sub-agent to invoke the **maniskill-log-analisis** targeting path to log with failed planner.

---

### Step 3: Aggregate Reports & Synthesize Output

Collect analysis outputs from all sub-agents and generate a consolidated report using the structured format below.

---

## Final Report Template

```markdown
# ManiSkill Motion Planner Failure Diagnostics Report

**Environment/Scene:** <scene_name>  
**Total Runs:** <num_runs>  
**Success Rate:** <success_count> / <num_runs> (<success_percentage>%)  
**Failed Seeds:** [<seed_1>, <seed_2>, ...]

---

## Summary of Failure Categorization

| Failure Category | Affected Seeds Count | Primary Causes |
| :--- | :--- | :--- |
| <Category Name 1> | X | IK reach limits, singularity points |
| <Category Name 2> | Y | Torso-table clip, gripper-object collision |
| <Category Name 3> | Z | Path planner trapped in local minima |

---

## Detailed Analysis & Solutions by Failure Group

### 1. Category: <Category Name 1>
* **Affected Seeds:** [<seed_1>, <seed_3>, ...]
* **Failure Phase:** [e.g., Pre-grasp Reach, Object Lifting, Navigation]
* **Root Cause Analysis:** Detailed explanation of why the motion planner failed for this group (e.g., arm link collided with table edge due to low torso posture across these runs).
* **High-Level Fixes & Action Plan:**
  * Adjust default robot pose (e.g., raise torso height by `+N` units prior to execution).
  * Tune planner parameters (e.g., adjust collision buffer margins or increase solver iterations).
  * Modify trajectory generation (e.g., insert intermediate lift waypoints before horizontal translation).
  * *[Optional]* Log directories for reference: `<path_to_log_1>`, `<path_to_log_2>`

---

### 2. Category: <Category Name 2>
* **Affected Seeds:** [<seed_2>, <seed_5>, ...]
* **Failure Phase:** [e.g., Navigation, Trajectory Execution]
* **Root Cause Analysis:** Detailed explanation of the shared failure mode across these seeds.
* **High-Level Fixes & Action Plan:**
  * Specific pose/posture adjustment for this issue.
  * Specific planner parameter tweaks.
  * Trajectory/waypoint adjustments.

*(Repeat section for each identified failure category)*

```

## Guidelines for Sub-Agent Delegation

* Ensure sub-agents are executed in parallel if supported by the orchestration frame to speed up batch log evaluation.
* Do not attempt to read raw trajectory dumps directly in the root agent; offload all trace parsing to sub-agents.
* Ensure all high-level fixes are actionable for robot control tuning (focus on spatial offsets, stance tweaks, or path constraints).

```
