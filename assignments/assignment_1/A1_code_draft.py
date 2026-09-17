"""Assignment 1 - Group 28: evolving robot bodies (tree genotype) with ariel.ec.

Scaffolding: config, fitness, genotype helpers, and the EA steps shared by
every variant (evaluate / parent_selection / crossover / survivor_selection)
are implemented. What's left - the three things this assignment actually
compares - are marked TODO and raise NotImplementedError:

    mutate_static              section 4, "static" variant
    AdaptiveMutation.mutate    section 4, "adaptive" variant
    AdaptiveMutation.adapt     section 4, "adaptive" variant (the 1/5 rule)
    baseline_regenerate        section 4, "baseline" variant

    python -m group_28.run --variant static --seeds 1 2 3 4 5
    python -m group_28.run --variant adaptive --seeds 1 2 3 4 5
    python -m group_28.run --variant baseline --seeds 1 2 3 4 5
"""

# Standard library
import argparse
import copy
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Third-party libraries
import networkx as nx
import numpy as np

# Local script (sibling to assignment_1/, see A1_template_2026.py)
from tree_edit_distance import mean_plus_std_tree_edit_distance

# Local libraries (ARIEL)
from ariel.body_phenotypes.robogen_lite.decoders._blueprint import (
    load_graph_from_json,
)
from ariel.ec import EA, EAOperation, Individual, Population
from ariel.ec.genotypes.tree.operators import (
    crossover_subtree,
    mutate_hoist,
    mutate_replace_node,
    mutate_shrink,
    mutate_subtree_replacement,
    random_tree,
    validate_tree_depth,
)
from ariel.ec.genotypes.tree.tree_genome import TreeGenome

type VariantName = Literal["static", "adaptive", "baseline"]

# ============================================================================ #
#  CONFIGURATION
# ============================================================================ #

HERE = Path(__file__).parent
TARGET_DIR: Path = HERE / "target_bodies"
CWD = Path.cwd()
DATA = CWD / "__data__" / "assignment_1"

NUM_OF_MODULES: int = 20  # module budget for a freshly sampled tree
MAX_TREE_DEPTH: int = 12  # depth guard against GP bloat
MAX_TOTAL_MODULES: int = 2 * NUM_OF_MODULES  # size guard against GP bloat

SEEDS: tuple[int, ...] = (1, 2, 3, 4, 5)
POP_SIZE: int = 5
NUM_GENERATIONS: int = 2
EVAL_BUDGET: int = POP_SIZE * NUM_GENERATIONS  # kept equal across every variant

TOURNAMENT_SIZE: int = 3
CROSSOVER_PROBABILITY: float = 0.2

# Mutation - the ONE thing that differs between variants. Both express
# "mutation strength" the same way: how many structural edits a fresh
# offspring receives.
MUTATION_OPS = [
    mutate_replace_node,
    mutate_subtree_replacement,
    mutate_shrink,
    mutate_hoist,
]
STATIC_N_OPS: int = 1  # variant "static": always exactly one structural edit

ADAPTIVE_INITIAL_STRENGTH: float = 1.0
ADAPTIVE_TARGET_SUCCESS: float = 0.2  # Rechenberg's 1/5
ADAPTIVE_FACTOR: float = 1.22  # classic ES step-size update factor
ADAPTIVE_MIN_STRENGTH: float = 0.2
ADAPTIVE_MAX_STRENGTH: float = 5.0


def db_path(variant: str, seed: int) -> Path:
    """One SQLite database per independent run.

    EA's `db_handling` defaults to "delete", so two runs pointed at the same
    path leave only the latest one.
    """
    return DATA / variant / f"seed_{seed}" / "database.db"


# ============================================================================ #
#  1. TARGETS + FITNESS  (as in A1_template_2026.py)
# ============================================================================ #


def load_targets(target_dir: Path = TARGET_DIR) -> list[nx.DiGraph]:
    """Load every target body graph from a directory.

    Raises
    ------
    FileNotFoundError
        If the directory holds no target JSON files.
    """
    paths = sorted(target_dir.glob("*.json"))
    if not paths:
        msg = f"no target bodies found in {target_dir}"
        raise FileNotFoundError(msg)
    return [load_graph_from_json(p) for p in paths]


TARGETS: list[nx.DiGraph] = load_targets()


def fitness_function(body: nx.DiGraph) -> float:
    """Mean + 1 std tree edit distance to every target. LOWER IS BETTER."""
    return mean_plus_std_tree_edit_distance(body, TARGETS)


# ============================================================================ #
#  2. GENOTYPE  (tree encoding; only ariel.ec.genotypes.tree operators)
# ============================================================================ #
# `Individual.genotype_` is a JSON column, so genotypes are stored as
# `TreeGenome.to_dict()`; convert to `TreeGenome` only when an operator needs it.


def random_genotype() -> dict:
    return random_tree(max_modules=NUM_OF_MODULES).to_dict()


def create_individual() -> Individual:
    ind = Individual()
    ind.genotype = random_genotype()
    return ind


def decode(genotype: dict) -> nx.DiGraph:
    return TreeGenome.from_dict(genotype).to_networkx()


# ============================================================================ #
#  3. SHARED EA STEPS  (identical for every variant except mutation)
# ============================================================================ #


def evaluate(population: Population) -> Population:
    for ind in population.unevaluated:
        ind.fitness = fitness_function(decode(ind.genotype))
    return population


def parent_selection(
    population: Population,
    tournament_size: int = TOURNAMENT_SIZE,
) -> Population:
    """k-tournament selection. Winners tagged 'selected'; population unchanged in size."""
    alive = population.alive.to_list()
    for ind in population:
        ind.tags = {"selected": False}
    for _ in range(len(alive)):
        contenders = random.sample(alive, min(tournament_size, len(alive)))
        winner = min(contenders, key=lambda ind: ind.fitness_)  # minimisation
        winner.tags = {"selected": True}
    return population


def crossover(
    population: Population,
    crossover_probability: float = CROSSOVER_PROBABILITY,
) -> Population:
    """Subtree crossover between random pairs of selected parents.

    Each child is tagged with `baseline_fitness` (the better of its two
    parents) and `mutate=True` so the mutation step - and, for the adaptive
    variant, the 1/5-rule bookkeeping - knows what "better than the parent"
    means for this child.
    """
    parents = population.where(
        lambda ind: bool(ind.tags.get("selected", False)),
    ).shuffle()

    children: list[Individual] = []
    for idx in range(0, len(parents) - 1, 2):
        p_a, p_b = parents[idx], parents[idx + 1]
        baseline = min(p_a.fitness_, p_b.fitness_)

        if random.random() < crossover_probability:
            g_a = TreeGenome.from_dict(p_a.genotype)
            g_b = TreeGenome.from_dict(p_b.genotype)
            c_a, c_b = crossover_subtree(g_a, g_b)
            genotypes = (c_a.to_dict(), c_b.to_dict())
        else:
            genotypes = (p_a.genotype, p_b.genotype)

        for genotype in genotypes:
            child = Individual()
            child.genotype = genotype
            child.tags = {"mutate": True, "baseline_fitness": baseline}
            children.append(child)

    population.extend(children)
    return population


def _apply_ops(genome: TreeGenome, n_ops: int) -> TreeGenome:
    """Apply `n_ops` random structural mutations in place.

    Each edit is rejected (and the genome rolled back) if it would break the
    depth or size guard - a cheap defence against the tree bloat the
    assignment brief warns about.
    """
    for _ in range(n_ops):
        before_nodes, before_edges = copy.deepcopy(genome.nodes), copy.deepcopy(genome.edges)

        op = random.choice(MUTATION_OPS)
        if op is mutate_subtree_replacement:
            op(genome, max_modules=NUM_OF_MODULES)
        else:
            op(genome)

        too_deep = not validate_tree_depth(genome, MAX_TREE_DEPTH)
        too_big = len(genome.nodes) > MAX_TOTAL_MODULES
        if too_deep or too_big:
            genome.nodes, genome.edges = before_nodes, before_edges
    return genome


def survivor_selection(
    population: Population,
    target_population_size: int = POP_SIZE,
) -> Population:
    """(mu + lambda) truncation: keep the best `target_population_size`, minimisation."""
    ranked = population.alive.sort(sort="min", attribute="fitness_")
    survivor_ids = {id(ind) for ind in ranked[:target_population_size]}
    for ind in population:
        if id(ind) not in survivor_ids:
            ind.alive = False
    return population


# ============================================================================ #
#  4. MUTATION  (the ONE step that differs between variants)
# ============================================================================ #

def mutate_static(population: Population, n_ops: int = STATIC_N_OPS) -> Population:
    for ind in population:
        if not ind.tags.get("mutate", False):
            continue
        genome = TreeGenome.from_dict(ind.genotype)
        genome = _apply_ops(genome, n_ops)
        ind.genotype = genome.to_dict()
        ind.requires_eval = True
    return population


@dataclass
class AdaptiveMutation:
    """Mutation strength controlled by Rechenberg's 1/5-success rule.

    `strength` (rounded to the nearest int >= 1) is the number of structural
    edits applied per offspring - the same "how much" knob `mutate_static`
    uses, just no longer fixed. Once per generation, after offspring are
    evaluated, `adapt` compares each child's fitness to the better of its two
    parents (tagged by `crossover`): if more than 1/5 of offspring improved
    on their parent, the step widens; otherwise it shrinks.
    """

    strength: float = ADAPTIVE_INITIAL_STRENGTH
    target_success: float = ADAPTIVE_TARGET_SUCCESS
    factor: float = ADAPTIVE_FACTOR
    min_strength: float = ADAPTIVE_MIN_STRENGTH
    max_strength: float = ADAPTIVE_MAX_STRENGTH

    def mutate(self, population: Population) -> Population:
        n_ops = max(1, round(self.strength))
        for ind in population:
            if not ind.tags.get("mutate", False):
                continue
            genome = TreeGenome.from_dict(ind.genotype)
            genome = _apply_ops(genome, n_ops)
            ind.genotype = genome.to_dict()
            ind.requires_eval = True
        return population

    def adapt(self, population: Population) -> Population:
        offspring = [
            ind for ind in population
            if "baseline_fitness" in ind.tags and not ind.requires_eval
        ]
        if not offspring:
            return population  # nothing mutated this generation

        successes = sum(
            1 for ind in offspring if ind.fitness_ < ind.tags["baseline_fitness"]
        )
        success_rate = successes / len(offspring)

        if success_rate > self.target_success:
            self.strength = min(self.max_strength, self.strength * self.factor)
        elif success_rate < self.target_success:
            self.strength = max(self.min_strength, self.strength / self.factor)
        # exactly on target: leave self.strength unchanged

        return population   

def baseline_regenerate(population: Population) -> Population:
    for ind in population:
        ind.alive = False
    fresh = [create_individual() for _ in range(POP_SIZE)]
    population.extend(fresh)
    return population

# ============================================================================ #
#  5. VARIANT PIPELINES
# ============================================================================ #


def build_pipeline(variant: VariantName) -> list[EAOperation]:
    match variant:
        case "baseline":
            return [
                EAOperation(baseline_regenerate),
                EAOperation(evaluate),
            ]
        case "static":
            return [
                EAOperation(parent_selection),
                EAOperation(crossover),
                EAOperation(mutate_static),
                EAOperation(evaluate),
                EAOperation(survivor_selection),
            ]
        case "adaptive":
            adaptive = AdaptiveMutation()
            return [
                EAOperation(parent_selection),
                EAOperation(crossover),
                EAOperation(adaptive.mutate),
                EAOperation(evaluate),
                EAOperation(adaptive.adapt),
                EAOperation(survivor_selection),
            ]


# ============================================================================ #
#  6. ENTRY POINT
# ============================================================================ #


def run(variant: VariantName, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    initial = Population([create_individual() for _ in range(POP_SIZE)])
    initial = evaluate(initial)

    path = db_path(variant, seed)
    path.parent.mkdir(parents=True, exist_ok=True)

    ea = EA(
        initial,
        build_pipeline(variant),
        num_steps=NUM_GENERATIONS,
        is_maximisation=False,
        db_file_path=path,
    )
    ea.run()

    best = ea.get_solution("best", only_alive=False)
    print(f"{variant} seed={seed}  best fitness={best.fitness:.4f}  {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one EA variant across seeds")
    parser.add_argument("--variant", required=True, choices=["static", "adaptive", "baseline"])
    parser.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    args = parser.parse_args()

    for seed in args.seeds:
        started = time.perf_counter()
        run(args.variant, seed)
        print(f"  ({time.perf_counter() - started:.1f}s)")


if __name__ == "__main__":
    main()