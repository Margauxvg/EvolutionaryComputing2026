from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
TARGET_DIR: Path = HERE.parent / "target_bodies"
RESULTS_DIR: Path = HERE / "results"
RESULT_FILE_NAME: str = "fitness_overview.csv"

# ---------------------------------------------------------------------------
# Genome / tree constraints
# ---------------------------------------------------------------------------

NUM_OF_MODULES: int = 20
MAX_TREE_DEPTH: int = 12
MAX_TOTAL_MODULES: int = 2 * NUM_OF_MODULES
MAX_MUTATION_ATTEMPTS: int = 20

# ---------------------------------------------------------------------------
# Experiment setup
# ---------------------------------------------------------------------------

DEFAULT_SEED: int = 1
VARIANTS: tuple[str, ...] = ("static", "adaptive", "baseline")
POP_SIZE: int = 10
NUM_GENERATIONS: int = 10
EVAL_BUDGET: int = POP_SIZE * NUM_GENERATIONS

# ---------------------------------------------------------------------------
# Selection & reproduction
# ---------------------------------------------------------------------------

TOURNAMENT_SIZE: int = 3
CROSSOVER_PROBABILITY: float = 0.7   # raised from 0.2: too low left most offspring as uncrossed parent
                                     # copies, so the population converged (std=0) by generation ~20-30
                                     # and stayed flat for the rest of the run; 0.7 delayed that by 40-50+
                                     # generations and improved final best_fitness on a seed=1 test run
ELITISM_RATIO: float = 0.05          # lowered from 0.10: fewer frozen elites per generation reduces how
                                     # fast the pop=100 population loses turnover/diversity

# ---------------------------------------------------------------------------
# Mutation (static & adaptive)
# ---------------------------------------------------------------------------
ADAPTIVE_INITIAL_STRENGTH: float = 0.25

ADAPTIVE_TARGET_SUCCESS: float = 0.2    # Rechenberg's 1/5, ongewijzigd
ADAPTIVE_STRENGTH_FACTOR: float = 1.22  # hergebruikt van jullie oude ADAPTIVE_FACTOR
ADAPTIVE_MIN_STRENGTH: float = 0.05     # Lower bound > 0 so that subtree_replacement/hoist never completely fail
ADAPTIVE_MAX_STRENGTH: float = 0.8      # (otherwise, adaptive degenerates into pure replace_node/shrink and you lose
ADAPTIVE_MAX_STRENGTH: float = 0.8      # the ability to recover/explore after convergence—exactly the
                                        # spiral-to-zero problem you already saw with ADAPTIVE_MIN_PROBABILITY).


# Venster i.p.v. reset-per-generatie: middel succes over de laatste ~5
# generaties aan offspring (5 * POP_SIZE events), analoog aan Rechenberg's
# advies om over ~10n mutaties te middelen.
ADAPTIVE_WINDOW_SIZE: int = 5 * POP_SIZE
MUTATION_PROBABILITY: float = 0.6   # tested against 0.1 (matching ADAPTIVE_MIN_PROBABILITY) at 50
                                            # generations: 0.6 reached 12.795 best_fitness vs 0.1's 13.691,
                                            # the latter statistically indistinguishable from adaptive's own
                                            # 13.713 - confirms static's advantage comes from the sustained
                                            # high rate itself, not from being non-adaptive

MUTATION_OPERATOR_WEIGHTS: dict[str, float] = {   # tested a more disruptive mix (35/35/15/15, shifting weight
    "replace_node": 0.60,                         # from replace_node onto subtree_replacement) over 100
    "subtree_replacement": 0.10,                  # generations: it helped adaptive (13.355 -> 13.268) but made
    "shrink": 0.15,                               # static worse and less stable (12.410+/-0.131 -> 12.635+/-0.486),
    "hoist": 0.15,                                # so kept these original weights for the official comparison
}

# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

CSV_COLUMNS: tuple[str, ...] = (
    "generation",
    "variant",
    "seed",
    "best_fitness",
    "mean_fitness",
    "std_fitness",
    "mutation_probability",
    "mutation_strength",
)
