"""Every number the experiment depends on, in one place.

Same pattern as A1's config.py, and for the same reason: the brief requires the experiment
to be reproducible and Methods has to state every parameter. If a value lives here, Methods
can be read off this file. If it is a function default somewhere, it will be misreported.

All values below are final for the main experiment (design frozen Sat 3 Oct). Pilots change
them per invocation through run.py's options, never by editing this file.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
#  Paths
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
RESULTS_DIR: Path = HERE / "results"
RESULT_FILE_NAME: str = "fitness_overview.csv"

# --------------------------------------------------------------------------- #
#  The task - fixed for the whole assignment
# --------------------------------------------------------------------------- #
# The brief: keep body, world, SIM_DURATION and fitness identical across everything you
# compare. Changing any of these mid-experiment makes earlier runs incomparable.
SPAWN_POS: list[float] = [0.0, 0.0, 0.1]
TARGET_POSITION: list[float] = [2.0, 0.0, 0.1]

# Re-measured on the John Set gecko (stage1_findings.md section 5): the best pilot controller
# walks at ~12 cm/s and is 0.13 m from the target at 15 s. So almost every run measures PROGRESS
# rather than arrival, and the few runs that get close are handled in the analysis (analyze.py,
# SATURATION_DISTANCE) instead of by changing the task. Every configuration truncates at the
# same point.
SIM_DURATION: float = 15.0

# A LABEL for the results folders and run_info.json only. The body itself is built in
# simulate.build_robot, which always builds john_set.gecko(); change both together or neither.
BODY_NAME: str = "gecko"

# --------------------------------------------------------------------------- #
#  Controller
# --------------------------------------------------------------------------- #
# controller.py reads these; defining them here makes this file authoritative.
HIDDEN_SIZE: int = 6
# Settled by argument, not by sweep. On the John Set gecko: 108 weights (18h, from 12 inputs and
# 6 outputs with no bias vector), 138.9 evaluations per weight at the current budget. Inherited
# from the template and NOT tuned - section 6 gives every reason to expect a sweep would return
# another null, so the compute went into 20 seeds instead. Methods says exactly that.
# Confirmed by `experiments.py hidden` on the John Set gecko, 1 Oct.
CLOCK_FREQ: float = 1.0
# Settled, as a null. 1-6 Hz across 3 seeds (18 runs): F = 0.95 between frequencies - less
# variation than noise alone would give - and a paired permutation test gives p = 0.38. No choice
# is catastrophic (all 18 runs land in 1.42-1.78 m), so this is a nuisance parameter that only has
# to be held identical across variants. See stage1_findings.md section 6.
DIST_SCALE: float = 2.0  # normalises target distance to ~[0, 1] at spawn

CONTROL_ALPHA: float = 0.05  # DELTA step size, template's controller contract
WORST_FITNESS: float = 100.0  # what a NaN/inf individual scores; any value above the
# largest achievable distance works, and distances here cannot exceed a few metres

# --------------------------------------------------------------------------- #
#  Experiment
# --------------------------------------------------------------------------- #
DEFAULT_SEED: int = 1

# 20 seeds, not the 5 the brief requires. Two reasons, and the second is now measured.
#   (a) At 5 vs 5 the smallest attainable two-sided Mann-Whitney p is 0.0079, reached only under
#       complete separation, so the test bottoms out before it can resolve a moderate effect -
#       which is exactly what happened in A1.
#   (b) The pilot (John Set gecko, 150 generations, 8 seeds per variant) measured the
#       seed-to-seed sd of the final distance at 0.29 m. With 20 runs per configuration that
#       gives 80% power for a difference of 0.26 m (the number Methods reports).
# Compute is cheap enough here to buy statistical power instead of scale.
#
# 101-120, NOT 1-20: the pilots used seeds 1-8, and runs are deterministic, so seeds 1-8 here
# would just be the pilot runs continued. sigma_max = 1.0 and the run length were chosen after
# looking at those runs, so the main experiment uses seeds that were never looked at.
SEEDS: tuple[int, ...] = tuple(range(101, 121))

POP_SIZE: int = 50  # from the pilot (Methods, "Population size and run length")
# The brief: "run until your fitness curve plateaus". In the pilot the mean best distance still
# improved by 0.07 m over the last 25 of 150 generations, and that improvement roughly halved
# every 50 generations, so the curve is expected to flatten around 250. Fixed for every run, so
# every configuration gets the same budget: 50 x (250 + 1) = 12,550 evaluations.
NUM_GENERATIONS: int = 250

# The four configurations of the main experiment. "adaptive_cap1" is the 1/5 rule with the upper
# bound on sigma lowered to ADAPTIVE_CAP1_MAX_SIGMA, to check how much the bound matters (the
# pilot hit the 2.0 bound in 7 of 8 runs). ea.py only treats "baseline" differently, so every
# other name runs the same EA and differs only in what mutation.make_mutation returns.
VARIANTS: tuple[str, ...] = ("static", "adaptive", "adaptive_cap1", "baseline")

# --------------------------------------------------------------------------- #
#  EA - identical across static and adaptive. Only the mutation strategy differs.
# --------------------------------------------------------------------------- #
TOURNAMENT_SIZE: int = 3
# No CROSSOVER_PROBABILITY: there is no crossover. Mutation is the only variation operator, so
# every offspring is attributable to it and the 1/5 success rate is measured on all 50 children
# per generation rather than the ~15 that survived A1's exclusion rule. stage1_findings.md 9.
ELITISM_RATIO: float = 0.05  # fraction of the PARENT population carried over unchanged
INIT_WEIGHT_SCALE: float = 0.5  # std of the normal the initial genotypes are drawn from,
# matching the template's make_random_weights

# --------------------------------------------------------------------------- #
#  Mutation - the one thing that differs between the two variants
# --------------------------------------------------------------------------- #
# sigma is the standard deviation of the Gaussian added to each weight - a STEP SIZE, which is
# the parameter the 1/5 rule was derived to control. A1 adapted a mutation PROBABILITY instead,
# and the A1 marker deducted for exactly that conflation. Note what is NOT native here: the rule
# was derived for the (1+1)-ES, and this is a population-based EA with tournament selection and
# elitism. Native parameter, extended algorithm - say both, never "native setting".
STATIC_SIGMA: float = 0.3  # the value the Stage 1 probe used
ADAPTIVE_INITIAL_SIGMA: float = 0.3  # MUST equal STATIC_SIGMA: both variants start identical,
# so any divergence is the controller's doing. A1 nearly shipped this confound.

ADAPTIVE_TARGET_SUCCESS: float = 0.2  # Rechenberg's 1/5
ADAPTIVE_FACTOR: float = 1.22  # ~1/0.817, the reciprocal of the classical constant

# A1 floored the mutation probability at 0.1 and it pinned there once success hit zero. In
# continuous space a shrinking sigma near an optimum is the rule working correctly, so the
# floor is set low enough to be effectively off - a ceiling is the only real guard needed.
ADAPTIVE_MIN_SIGMA: float = 1e-4
ADAPTIVE_MAX_SIGMA: float = 2.0  # four times INIT_WEIGHT_SCALE
ADAPTIVE_CAP1_MAX_SIGMA: float = 1.0  # the upper bound for the "adaptive_cap1" configuration

ADAPTIVE_WINDOW: int = 5  # generations pooled before comparing to the 1/5 target
ADAPTIVE_MIN_SAMPLES: int = 10  # below this many scored mutations in the window, hold sigma

# --------------------------------------------------------------------------- #
#  Logging
# --------------------------------------------------------------------------- #
# Same shape as A1's CSV so analyze.py ports with minimal changes. Three columns are new:
#   sigma                 the quantity the research question is about
#   mean_genotype_spread  A1 discovered its convergence story after the fact; log it from
#                         generation 0 this time
#   num_nan               failed (NaN/inf) controllers among that generation's new evaluations,
#                         so a silent controller blow-up is visible
CSV_COLUMNS: tuple[str, ...] = (
    "generation",
    "variant",
    "seed",
    "best_fitness",
    "mean_fitness",
    "std_fitness",
    "sigma",
    "success_rate",
    "num_scored_mutations",
    "num_mutated",
    "mean_genotype_spread",
    "num_nan",
    "evaluations",
)
