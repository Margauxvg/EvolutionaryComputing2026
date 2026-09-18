import random

import config
from ariel.ec.genotypes.tree.operators import (
    random_tree as random_genome,
    crossover_subtree as crossover_operator,
)
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from helpers import load_targets, make_individual
from mutation import MutationVariant, make_mutation
from tree_edit_distance import mean_plus_std_tree_edit_distance as fitness_function

TARGETS = load_targets()


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------

def evaluate(population: list[dict]) -> list[dict]:
    for individual in population:
        if individual.get("fitness") is None:
            body = TreeGenome.from_dict(individual["genotype"]).to_networkx()
            individual["fitness"] = fitness_function(body, TARGETS)
    return population


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------

def init_population(size: int = config.POP_SIZE) -> list[dict]:
    """Uniform random initialization: a fresh random tree genome per individual."""
    return [
        make_individual(random_genome(max_modules=config.NUM_OF_MODULES).to_dict())
        for _ in range(size)
    ]


# ---------------------------------------------------------------------------
# Selection & reproduction
# ---------------------------------------------------------------------------

def parent_selection(population: list[dict], tournament_size: int = config.TOURNAMENT_SIZE) -> list[dict]:
    """Tournament selection: best-of-`tournament_size` sample, one tournament per parent slot, with replacement."""
    candidates = [ind for ind in population if ind.get("fitness") is not None and ind.get("alive", True)]
    parents: list[dict] = []

    for _ in range(len(candidates)):
        contestants = random.sample(candidates, tournament_size)
        winner = min(contestants, key=lambda individual: individual["fitness"])
        parents.append(winner)

    return parents


def reproduction(population: list[dict], crossover_probability: float = config.CROSSOVER_PROBABILITY) -> list[dict]:
    """Subtree crossover, applied per parent pair with `crossover_probability` (else parents pass through
    unchanged); offspring carry "parent_fitness" for the adaptive mutation's success rule."""
    parents = parent_selection(population)
    if len(parents) < 2:
        return []

    random.shuffle(parents)
    offspring: list[dict] = []

    for i in range(0, len(parents) - 1, 2):
        parent_a = parents[i]
        parent_b = parents[i + 1]
        baseline = min(parent_a["fitness"], parent_b["fitness"])

        if random.random() < crossover_probability:
            genome_a = TreeGenome.from_dict(parent_a["genotype"])
            genome_b = TreeGenome.from_dict(parent_b["genotype"])
            child_a, child_b = crossover_operator(genome_a, genome_b)
            genotypes = (child_a.to_dict(), child_b.to_dict())
        else:
            genotypes = (parent_a["genotype"], parent_b["genotype"])

        for genotype in genotypes:
            offspring.append(make_individual(genotype, parent_fitness=baseline))

    return offspring


def survivor_selection(population: list[dict], target_population_size: int = config.POP_SIZE) -> list[dict]:
    """Generational replacement with elitism: the top `ELITISM_RATIO` fraction survives unconditionally,
    the rest of the next generation is filled from the remaining best-ranked individuals."""
    ranking = sorted(population, key=lambda individual: individual["fitness"])
    elite_count = max(1, int(config.ELITISM_RATIO * target_population_size))
    elites = ranking[:elite_count]

    next_generation = elites[:]
    for individual in ranking[elite_count:]:
        if len(next_generation) >= target_population_size:
            break
        next_generation.append(individual)

    for individual in next_generation:
        individual["alive"] = True

    return next_generation[:target_population_size]


# ---------------------------------------------------------------------------
# Generation step
# ---------------------------------------------------------------------------

def baseline_regenerate(population: list[dict]) -> list[dict]:
    """Random-search baseline: the whole population is replaced with fresh random individuals every
    generation (no selection, crossover, or mutation); matches POP_SIZE so the evaluation budget stays
    equal across variants."""
    return init_population()


def run_generation(variant: str, population: list[dict], mutation: MutationVariant | None) -> list[dict]:
    if variant == "baseline":
        return evaluate(baseline_regenerate(population))

    assert mutation is not None
    offspring = reproduction(population)
    offspring = mutation.mutate(offspring)
    offspring = evaluate(offspring)
    offspring = mutation.adapt(offspring)
    return survivor_selection(population + offspring)
