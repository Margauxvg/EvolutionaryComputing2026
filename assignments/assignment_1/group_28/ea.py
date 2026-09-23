import random

import config
from ariel.ec.genotypes.tree.operators import (
    random_tree as random_genome,
    crossover_subtree as crossover_operator,
)
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from helpers import load_targets, make_individual
from mutation import MutationVariant
from tree_edit_distance import mean_plus_std_tree_edit_distance as fitness_function

TARGETS = load_targets()


def evaluate(population: list[dict]) -> list[dict]:
    for individual in population:
        if individual.get("fitness") is None:
            body = TreeGenome.from_dict(individual["genotype"]).to_networkx()
            individual["fitness"] = fitness_function(body, TARGETS)
    return population


def init_population(size: int = config.POP_SIZE) -> list[dict]:
    return [
        make_individual(random_genome(max_modules=config.NUM_OF_MODULES).to_dict())
        for _ in range(size)
    ]


def parent_selection(population: list[dict], tournament_size: int = config.TOURNAMENT_SIZE) -> list[dict]:
    candidates = [ind for ind in population if ind.get("fitness") is not None and ind.get("alive", True)]
    parents: list[dict] = []

    for _ in range(len(candidates)):
        contestants = random.sample(candidates, tournament_size)
        parents.append(min(contestants, key=lambda individual: individual["fitness"]))

    return parents


def reproduction(population: list[dict]) -> list[dict]:
    """Subtree crossover per parent pair. Each offspring also gets two extra labels used later
    to judge whether mutation helped: crossed (did crossover touch it?) and parent_fitness,
    the fitness it needs to beat. If it was crossed, that's the better of its two parents
    (since it's a mix of both). If not, it's just that one parent's own fitness."""
    parents = parent_selection(population)
    random.shuffle(parents) # prevent picking pairs of parents in the same order every generation
    offspring: list[dict] = []

    for i in range(0, len(parents) - 1, 2):
        parent_a, parent_b = parents[i], parents[i + 1]
        crossed = random.random() < config.CROSSOVER_PROBABILITY

        if crossed:
            genome_a = TreeGenome.from_dict(parent_a["genotype"])
            genome_b = TreeGenome.from_dict(parent_b["genotype"])
            child_a, child_b = crossover_operator(genome_a, genome_b)
            genotypes = (child_a.to_dict(), child_b.to_dict())
            fitter_parent = min(parent_a["fitness"], parent_b["fitness"])
            parent_fitnesses = (fitter_parent, fitter_parent)
        else:
            genotypes = (parent_a["genotype"], parent_b["genotype"])
            parent_fitnesses = (parent_a["fitness"], parent_b["fitness"])

        for genotype, parent_fitness in zip(genotypes, parent_fitnesses):
            offspring.append(make_individual(genotype, parent_fitness=parent_fitness, crossed=crossed))

    return offspring


def survivor_selection(parents: list[dict], offspring: list[dict]) -> list[dict]:
    """Generational replacement with elitism."""
    elite_count = max(1, int(config.ELITISM_RATIO * config.POP_SIZE))

    parent_elites = sorted(parents, key=lambda individual: individual["fitness"])[:elite_count]
    ranked_offspring = sorted(offspring, key=lambda individual: individual["fitness"])
    next_generation = parent_elites + ranked_offspring[: config.POP_SIZE - elite_count]

    for individual in next_generation:
        individual["alive"] = True

    return next_generation


def baseline_regenerate(population: list[dict]) -> list[dict]:
    return init_population()


def run_generation(variant: str, population: list[dict], mutation: MutationVariant | None) -> list[dict]:
    if variant == "baseline":
        return evaluate(baseline_regenerate(population))

    offspring = reproduction(population)
    offspring = mutation.mutate(offspring)
    offspring = evaluate(offspring)
    offspring = mutation.adapt(offspring)
    return survivor_selection(population, offspring)
