"""Shared constants. Every number that appears in Methods lives here."""

from pathlib import Path

HERE = Path(__file__).parent
TARGET_DIR = HERE.parent / "target_bodies"

# Same convention as the examples: CWD / "__data__" / <name>.
CWD = Path.cwd()
DATA = CWD / "__data__" / "assignment_1"

NUM_OF_MODULES: int = 20

SEEDS: tuple[int, ...] = (1, 2, 3, 4, 5)
POP_SIZE: int = 100
NUM_GENERATIONS: int = 100

# Generations x population, as everywhere in the repo.
# Shared by every variant and the baseline, so the comparison is like for like.
EVAL_BUDGET: int = POP_SIZE * NUM_GENERATIONS


def db_path(variant: str, seed: int) -> Path:
    """One database per independent run.

    EA's `db_handling` defaults to "delete", so two runs pointed at the same
    path leave one database.
    """
    return DATA / variant / f"seed_{seed}" / "database.db"
