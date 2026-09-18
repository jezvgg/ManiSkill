"""The Fetch motion-planning solver and its mplib-free step accounting.

Nothing is re-exported here on purpose. `.extand` (the solver) and `.utils` (the
planner) import mplib at module level, which has Linux wheels only; `.stepping`
(`StepGuard`, `refine_should_stop`, `pose_error`) does not, and must stay importable
on a Mac so tests/test_solver_stepping.py can run there. Import the submodule you
mean:

    from utils.mikasa_oracle.motionplanning.fetch.extand import MikasaFetchSolver
    from utils.mikasa_oracle.motionplanning.fetch.stepping import StepGuard
"""
