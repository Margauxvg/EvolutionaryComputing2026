"""Every number the experiment depends on, in one place.

Same pattern as A1's config.py, and for the same reason: the brief requires the experiment
to be reproducible and Methods has to state every parameter. If a value lives here, Methods
can be read off this file. If it is a function default somewhere, it will be misreported.

Values marked PROVISIONAL are not yet decided - they are placeholders so the code runs.
Settle them before the design freezes (Sat 3 Oct) and delete the marker.
"""

from pathlib import Path
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.john_set import gecko, spider_8

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

# 15 s does not let the gecko reach the target - the best controller from the Stage 1 probe
# stopped progressing at ~25 s, 1.45 m short (stage1_findings.md section 5). Fitness therefore
# measures PROGRESS, not arrival. That is still a monotone gradient, every configuration
# truncates at the same point, and 30 s would double an already-doubled compute budget.
SIM_DURATION: float = 15.0

BUILD_BODY = gecko  # PROVISIONAL - decided by the body pilot (John Set gecko vs spider_8)
CONTROL_MODE: str = "direct"   # "direct" or "delta"


# --------------------------------------------------------------------------- #
#  Controller
# --------------------------------------------------------------------------- #
# controller.py reads these; defining them here makes this file authoritative.
HIDDEN_SIZE: int = 6
# Settled by argument, not by sweep. 132 weights (22h: 14 inputs, 8 outputs, no bias vector) and
# 113.6 evaluations per weight at the current budget. Inherited from the template and NOT tuned -
# section 6 gives every reason to expect a sweep would return another null, so the compute went
# into 20 seeds instead. Methods says exactly that. See stage1_findings.md section 10.
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
#   (b) The clock sweep measured the seed-to-seed sd of a 20-generation outcome at 0.107 m. At
#       80% power that makes the smallest detectable difference 0.190 m with 5 seeds and 0.095 m
#       with 20. Half the progress a run makes would be invisible at 5.
# Compute is cheap enough here to buy statistical power instead of scale.
SEEDS: tuple[int, ...] = tuple(range(1, 21))

POP_SIZE: int = 50  # PROVISIONAL
NUM_GENERATIONS: int = 300  # PROVISIONAL - must come from the pilot plateau, not a guess.
# The brief: "run until your fitness curve plateaus, and treat that plateau, not a fixed
# generation count, as your stopping criterion."

VARIANTS: tuple[str, ...] = ("static", "adaptive", "baseline")

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
# sigma is the standard deviation of the Gaussian added to each weight. A1 adapted a discrete
# mutation PROBABILITY, which the report noted was "an analogy rather than a transfer" of
# Rechenberg's analysis. Here sigma is a continuous step size on a real-valued genome, which
# is what the 1/5 rule was derived for.
STATIC_SIGMA: float = 0.3  # the value the Stage 1 probe used
ADAPTIVE_INITIAL_SIGMA: float = 0.3  # MUST equal STATIC_SIGMA: both variants start identical,
# so any divergence is the controller's doing. A1 nearly shipped this confound.

ADAPTIVE_TARGET_SUCCESS: float = 0.2  # Rechenberg's 1/5
ADAPTIVE_FACTOR: float = 1.22  # ~1/0.817, the reciprocal of the classical constant

# A1 floored the mutation probability at 0.1 and it pinned there once success hit zero. In
# continuous space a shrinking sigma near an optimum is the rule working correctly, so the
# floor is set low enough to be effectively off - a ceiling is the only real guard needed.
ADAPTIVE_MIN_SIGMA: float = 1e-4
ADAPTIVE_MAX_SIGMA: float = 2.0

ADAPTIVE_WINDOW: int = 5  # generations pooled before comparing to the 1/5 target
ADAPTIVE_MIN_SAMPLES: int = 10  # below this many scored mutations in the window, hold sigma

# --------------------------------------------------------------------------- #
#  Logging
# --------------------------------------------------------------------------- #
# Same shape as A1's CSV so analyze.py ports with minimal changes. Three columns are new:
#   sigma                 the quantity the research question is about
#   mean_genotype_spread  A1 discovered its convergence story after the fact; log it from
#                         generation 0 this time
#   num_nan               NaN/inf individuals, so a silent controller blow-up is visible
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
