from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_DIR: Path = HERE / "results"
RESULT_FILE_NAME: str = "fitness_overview.csv"

# --------------------------------------------------------------------------- #
#  Controller
# --------------------------------------------------------------------------- #
HIDDEN_SIZE: int = 6 # number of neurons in the hidden layer
CLOCK_FREQ: float = 1.0 # controller update frequency (Hz)
DIST_SCALE: float = 2.0  # normalises target distance to ~[0, 1] at spawn

CONTROL_ALPHA: float = 0.05  # DELTA step size, template's controller contract
WORST_FITNESS: float = 100.0  # what a NaN/inf individual scores

# --------------------------------------------------------------------------- #
#  Experiment
# --------------------------------------------------------------------------- #
DEFAULT_SEED: int = 1
SEEDS: tuple[int, ...] = tuple(range(101, 121)) # other seeds than the pilot runs
POP_SIZE: int = 50
NUM_GENERATIONS: int = 250
VARIANTS: tuple[str, ...] = ("static", "adaptive", "adaptive_cap1", "baseline")

# --------------------------------------------------------------------------- #
#  Identical EA settings
# --------------------------------------------------------------------------- #
TOURNAMENT_SIZE: int = 3
ELITISM_RATIO: float = 0.05  # fraction of the parent population carried over unchanged
INIT_WEIGHT_SCALE: float = 0.5  # std of the normal the initial genotypes are drawn from

# --------------------------------------------------------------------------- #
#  Variant-specific mutation settings
# --------------------------------------------------------------------------- #
STATIC_SIGMA: float = 0.3  # fixed mutation step size
ADAPTIVE_INITIAL_SIGMA: float = 0.3  # initial mutation step size for adaptive variant
ADAPTIVE_TARGET_SUCCESS: float = 0.2  # Rechenberg's 1/5 success rate
ADAPTIVE_FACTOR: float = 1.22  # ~1/0.817, the reciprocal of the classical constant
ADAPTIVE_MIN_SIGMA: float = 1e-4 # lower bound for adaptive mutation step size
ADAPTIVE_MAX_SIGMA: float = 2.0  # upper bound for adaptive mutation step size (four times INIT_WEIGHT_SCALE)
ADAPTIVE_CAP1_MAX_SIGMA: float = 1.0  # the upper bound for the "adaptive_cap1" configuration
ADAPTIVE_WINDOW: int = 5  # generations pooled before comparing to the 1/5 target
ADAPTIVE_MIN_SAMPLES: int = 10  # below this many scored mutations in the window, hold sigma

# --------------------------------------------------------------------------- #
#  Logging
# --------------------------------------------------------------------------- #
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
