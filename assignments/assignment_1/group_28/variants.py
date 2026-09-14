"""EA variants: two configurations of one EA..
You can find operators from  genome.py (SUPPLIED_MUTATIONS / SUPPLIED_CROSSOVER / PRIMITIVES) and fitness
from problem.fitness.

Reference: examples/new_EC_engine_example.py, examples/c_genotypes/1_body_evolution_tree.py

Two settings we need to remember:
    is_maximisation=False      fitness is a minimisation
    db_file_path=db_path       one database per independent run
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Variant:
    name: str


def run(variant: Variant, seed: int, db_path: Path) -> None:
    raise NotImplementedError


VARIANTS: dict[str, Variant] = {}
