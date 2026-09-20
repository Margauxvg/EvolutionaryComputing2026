from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET_DIR: Path = HERE.parent / "target_bodies"
RESULTS_DIR: Path = HERE / "results"
RESULT_FILE_NAME: str = "fitness_overview.csv"

NUM_OF_MODULES: int = 20
MAX_TREE_DEPTH: int = 12
MAX_TOTAL_MODULES: int = 2 * NUM_OF_MODULES
MAX_MUTATION_ATTEMPTS: int = 20

DEFAULT_SEED: int = 1
SEEDS: tuple[int, ...] = (1, 2, 3, 4, 5)
POP_SIZE: int = 100
NUM_GENERATIONS: int = 100

TOURNAMENT_SIZE: int = 3
CROSSOVER_PROBABILITY: float = 0.7
ELITISM_RATIO: float = 0.05  # fraction of the PARENT population carried over unchanged

STATIC_MUTATION_PROBABILITY: float = 0.6
ADAPTIVE_INITIAL_PROBABILITY: float = 0.6
ADAPTIVE_TARGET_SUCCESS: float = 0.2  # Rechenberg's 1/5
ADAPTIVE_FACTOR: float = 1.22  # approximately 1/0.817, the reciprocal of the classical constant
ADAPTIVE_MIN_PROBABILITY: float = 0.1
ADAPTIVE_MAX_PROBABILITY: float = 1.0

ADAPTIVE_WINDOW: int = 5  # generations pooled together before comparing to the 1/5 target
ADAPTIVE_MIN_SAMPLES: int = 10  # below this many scored mutations in the window, hold the rate

MUTATION_OPERATORS: tuple[str, ...] = (
    "replace_node",
    "subtree_replacement",
    "shrink",
    "hoist",
)

CSV_COLUMNS: tuple[str, ...] = (
    "generation",
    "variant",
    "operator",
    "seed",
    "best_fitness",
    "mean_fitness",
    "std_fitness",
    "mutation_probability",
    "success_rate",
    "num_scored_mutations",
    "num_mutated",
)
