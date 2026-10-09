from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_DIR: Path = HERE / "results"
RESULT_FILE_NAME: str = "fitness_overview.csv"

# Task: walk from the spawn point to a target 2 m in front of it
SPAWN_POS: list[float] = [0.0, 0.0, 0.1]
TARGET_POSITION: list[float] = [2.0, 0.0, 0.1]
SIM_DURATION: float = 15.0  # seconds per evaluation
BODY_NAME: str = "gecko"  # only used as a folder name, the body itself is chosen in simulation.py

# Controller
HIDDEN_SIZE: int = 6  # neurons in the hidden layer, same as the template
CLOCK_FREQ: float = 1.0  # frequency (Hz) of the sin/cos clock that is fed into the network
DIST_SCALE: float = 2.0  # divides the target distance so it is about 1 at the spawn point
CONTROL_ALPHA: float = 0.05  # step size of the delta control, as in the template
WORST_FITNESS: float = 100.0  # fitness of a controller whose network outputs NaN

# Experiment
DEFAULT_SEED: int = 1
SEEDS: tuple[int, ...] = tuple(range(101, 121))  # not the seeds we used for the pilots
POP_SIZE: int = 50
NUM_GENERATIONS: int = 250
VARIANTS: tuple[str, ...] = ("static", "adaptive", "adaptive_cap1", "baseline")

# Same for every variant
TOURNAMENT_SIZE: int = 3
ELITISM_RATIO: float = 0.05  # fraction of the PARENT population carried over unchanged
INIT_WEIGHT_SCALE: float = 0.5  # std of the normal distribution the first genotypes are drawn from

# Mutation step size (sigma) per variant
STATIC_SIGMA: float = 0.3
ADAPTIVE_INITIAL_SIGMA: float = 0.3  # keep equal to STATIC_SIGMA so both variants start the same
ADAPTIVE_TARGET_SUCCESS: float = 0.2  # Rechenberg's 1/5
ADAPTIVE_FACTOR: float = 1.22  # approximately 1/0.817, the reciprocal of the classical constant
ADAPTIVE_MIN_SIGMA: float = 1e-4
ADAPTIVE_MAX_SIGMA: float = 2.0  # four times INIT_WEIGHT_SCALE
ADAPTIVE_CAP1_MAX_SIGMA: float = 1.0  # upper bound for the "adaptive_cap1" variant

ADAPTIVE_WINDOW: int = 5  # generations pooled together before comparing to the 1/5 target
ADAPTIVE_MIN_SAMPLES: int = 10  # below this many scored mutations in the window, keep sigma

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
