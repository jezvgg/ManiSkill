"""Seed every RNG that changes a run's outcome — including the one in C++.

Seeding `random`, `numpy` and `torch` is not enough, and that is measured rather
than argued. With all three seeded identically, four fresh processes running the
same seed put the cup in four different places:

    [2.33101153, -0.48897028, 1.00214922]   133 steps
    [2.33311391, -0.49026725, 1.00210857]   141
    [2.32759070, -0.48667774, 1.00176001]   131
    [2.32809997, -0.48597807, 1.00188661]   140

RRTConnect samples configurations at random, and mplib keeps that generator on the
C++ side where `np.random.seed` cannot reach it. Adding `mplib.set_global_seed`
makes the same four runs agree to the last digit — `[1.91649508, -0.48360544,
0.97749990]`, 124 steps, every time — and it reproduces on other seeds too.

Outcomes then belong to the seed instead of the throw. Measured over seeds 100-106
on kitchen 0: 100, 101, 104 succeed; 102, 105, 106 fail; 103 raises
`RuntimeError: Fail to parameterize path` — reproducibly, which is what makes it
worth debugging. See docs/lab-journal.md.

Call this per episode, not once per process: the generator is global and advances
as the planner runs, so an episode's result otherwise depends on how many episodes
preceded it.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, numpy, torch and mplib from one number."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        import mplib
    except ImportError:
        # mplib ships Linux wheels only. Scene code runs without it; the planners
        # do not, and they are the only thing whose determinism depends on this.
        return
    mplib.set_global_seed(seed)
