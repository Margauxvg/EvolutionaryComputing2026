from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
TARGET_DIR: Path = PROJECT_ROOT / "target_bodies"
RESULTS_DIR: Path = HERE / "results"

NUM_OF_MODULES: int = 20
MAX_TREE_DEPTH: int = 12
MAX_TOTAL_MODULES: int = 2 * NUM_OF_MODULES
MAX_MUTATION_ATTEMPTS: int = 20

DEFAULT_SEED: int = 1
VARIANTS: tuple[str, ...] = ("static", "adaptive", "baseline")
POP_SIZE: int = 100
NUM_GENERATIONS: int = 100
EVAL_BUDGET: int = POP_SIZE * NUM_GENERATIONS

TOURNAMENT_SIZE: int = 3
CROSSOVER_PROBABILITY: float = 0.2
ELITISM_RATIO: float = 0.10

STATIC_MUTATION_PROBABILITY: float = 0.1
ADAPTIVE_INITIAL_PROBABILITY: float = 0.3
ADAPTIVE_TARGET_SUCCESS: float = 0.2      # Rechenberg's 1/5
ADAPTIVE_FACTOR: float = 1.22
ADAPTIVE_MIN_PROBABILITY: float = 0.02
ADAPTIVE_MAX_PROBABILITY: float = 1.0

CSV_COLUMNS: tuple[str, ...] = (
    "generation",
    "variant",
    "seed",
    "best_fitness",
    "mean_fitness",
    "std_fitness",
)

RESULT_FILE_NAME: str = "fitness_overview.csv"
