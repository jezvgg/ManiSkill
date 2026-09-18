"""Scripted reference solutions: exactly one `<task>_planner.py` per benchmark task.

Each module exposes either the inherited ``planning(env, seed, ...)`` (the two
`myrobocasa_*` planners, returning a bool/tensor) or the upstream-shaped
``solve(env, seed=None, debug=False, vis=False)`` (every solution written here,
returning ``-1`` or the gym 5-tuple — see ``template_planner.py``);
``utils.mikasa_oracle.evaluate_planner`` accepts both. Every one but the ``template_planner.py``
skeleton is runnable standalone:

    python -m utils.mikasa_oracle.planners.myrobocasa_planner --seed 3

Nothing is re-exported here because these modules pull in mplib (Linux-only).
"""
