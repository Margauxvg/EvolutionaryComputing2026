"""Shared experiment configuration for EC Assignment 2 (Group 28).

Everything here is part of the FIXED setup: identical for every variant and
seed you compare. Change a value only before the final runs, never between them.
"""

from ariel.body_phenotypes.robogen_lite.prebuilt_robots.john_set import spider_8

# --- Body and world -------------------------------------------------------- #
BUILD_BODY = spider_8  # John Set body; swap here and nowhere else
SPAWN_POSITION: tuple[float, float, float] = (0.0, 0.0, 0.1)
TARGET_POSITION: tuple[float, float, float] = (2.0, 0.0, 0.1)
SIM_DURATION: float = 15.0  # seconds of simulated time per evaluation

# --- Neural controller ----------------------------------------------------- #
HIDDEN_SIZE: int = 6
CLOCK_FREQ: float = 1.0  # Hz; pick from pilots, then freeze
DIST_SCALE: float = 2.0  # initial spawn-target distance
CONTROL_MODE: str = "direct"  # "direct" or "delta"; pick one, justify it
DELTA_ALPHA: float = 0.05  # only used when CONTROL_MODE == "delta"