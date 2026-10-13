from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_DIR: Path = HERE / "results"

# Task: walk from the spawn point to a target 2 m away (as in the template)
SPAWN_POS: list[float] = [0.0, 0.0, 0.1]
TARGET_POSITION: list[float] = [2.0, 0.0, 0.1]
SIM_DURATION: float = 15.0  # seconds per evaluation

# Controller
HIDDEN_SIZE: int = 6  # neurons in the hidden layer, as in the template
CLOCK_FREQ: float = 1.0  # frequency (Hz) of the sin/cos clock that is fed into the network
DIST_SCALE: float = 2.0  # the distance input is divided by this, so it is 1 at the spawn point
CONTROL_ALPHA: float = 0.05  # step size of the delta control, as in the template
WORST_FITNESS: float = 100.0  # fitness of a controller whose network outputs NaN

# Experiment
SEEDS: tuple[int, ...] = tuple(range(101, 121))  # the pilots used seeds 1-8
NUM_GENERATIONS: int = 250
VARIANTS: tuple[str, ...] = ("static", "adaptive", "adaptive_cap1", "baseline")

# Evolutionary algorithm (the baseline only uses POP_SIZE and INIT_WEIGHT_SCALE)
POP_SIZE: int = 50
INIT_WEIGHT_SCALE: float = 0.5  # std of the normal distribution the first genotypes are drawn from
TOURNAMENT_SIZE: int = 3
ELITISM_RATIO: float = 0.05  # fraction of the parents copied unchanged into the next generation

# Mutation step size (sigma)
STATIC_SIGMA: float = 0.3
ADAPTIVE_INITIAL_SIGMA: float = STATIC_SIGMA  # so both variants start with the same step size
ADAPTIVE_TARGET_SUCCESS: float = 0.2  # Rechenberg's 1/5
ADAPTIVE_FACTOR: float = 1.22  # about 1/0.82, the factor commonly used in the classical rule
ADAPTIVE_MIN_SIGMA: float = 1e-4
ADAPTIVE_MAX_SIGMA: float = 2.0  # four times INIT_WEIGHT_SCALE
ADAPTIVE_CAP1_MAX_SIGMA: float = 1.0  # upper bound for the "adaptive_cap1" variant
SUCCESS_WINDOW: int = 5  # the success rate is measured over this many generations (also logged for static)
