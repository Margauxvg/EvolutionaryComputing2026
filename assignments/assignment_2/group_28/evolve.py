import statistics

import numpy as np

from ariel.ec import Individual, Population

import brain
import config
import simulation
from mutation import MutationVariant

# Every process (also every worker in main.py) builds its own world when it imports this
# file, because a compiled MuJoCo model cannot be sent to another process.
MODEL, DATA = simulation.build_world()

# How many controllers in the last evaluated batch gave NaN (they got WORST_FITNESS)
num_failed = 0


def genotype_length() -> int:
    return brain.num_weights(MODEL)


def make_individual(genotype: np.ndarray, **tags) -> Individual:
    individual = Individual()
    individual.genotype = genotype.tolist()
    if tags:
        individual.tags = tags
    return individual


def evaluate(population: Population) -> Population:
    global num_failed
    num_failed = 0
    for individual in population.unevaluated:
        fitness = simulation.evaluate(individual.genotype, MODEL, DATA)
        individual.fitness = fitness
        if fitness == config.WORST_FITNESS:
            num_failed += 1
    return population


def init_population(rng: np.random.Generator) -> Population:
    return Population([
        make_individual(rng.normal(scale=config.INIT_WEIGHT_SCALE, size=genotype_length()))
        for _ in range(config.POP_SIZE)
    ])


def parent_selection(population: Population, rng: np.random.Generator) -> list[Individual]:
    """Tournament selection, the lowest distance wins."""
    candidates = [ind for ind in population.alive if ind.fitness_ is not None]
    parents: list[Individual] = []

    for _ in range(len(candidates)):
        picked = rng.choice(len(candidates), config.TOURNAMENT_SIZE, replace=False)
        contestants = [candidates[i] for i in picked]
        parents.append(min(contestants, key=lambda individual: individual.fitness_))

    return parents


def reproduction(population: Population, rng: np.random.Generator) -> Population:
    """No crossover: every offspring is a copy of one parent, labelled with the parent's
    fitness so mutation can check later if it got better."""
    return Population([
        make_individual(np.asarray(parent.genotype, dtype=np.float64), parent_fitness=parent.fitness_)
        for parent in parent_selection(population, rng)
    ])


def survivor_selection(parents: Population, offspring: Population) -> Population:
    """Generational replacement with elitism."""
    elite_count = max(1, int(config.ELITISM_RATIO * config.POP_SIZE))

    parent_elites = sorted(parents, key=lambda individual: individual.fitness_)[:elite_count]
    ranked_offspring = sorted(offspring, key=lambda individual: individual.fitness_)
    next_generation = parent_elites + ranked_offspring[: config.POP_SIZE - elite_count]

    for individual in next_generation:
        individual.alive = True

    return Population(next_generation)


def baseline_regenerate(rng: np.random.Generator) -> Population:
    """Random search: throw the population away and draw a new one."""
    return init_population(rng)


def run_generation(
    variant: str,
    population: Population,
    mutation: MutationVariant | None,
    rng: np.random.Generator,
) -> Population:
    if variant == "baseline":
        return evaluate(baseline_regenerate(rng))

    offspring = reproduction(population, rng)
    offspring = mutation.mutate(offspring)
    offspring = evaluate(offspring)
    offspring = mutation.adapt(offspring)
    return survivor_selection(population, offspring)


def fitness_stats(population: Population) -> tuple[float, float, float]:
    """(best, mean, std) of the fitness in the population."""
    fitnesses = [individual.fitness_ for individual in population if individual.fitness_ is not None]
    if not fitnesses:
        return 0.0, 0.0, 0.0
    return min(fitnesses), statistics.fmean(fitnesses), statistics.pstdev(fitnesses)


def genotype_spread(population: Population) -> float:
    """Diversity: the std of every gene over the population, averaged over all genes."""
    genotypes = np.asarray([individual.genotype for individual in population], dtype=np.float64)
    if len(genotypes) < 2:
        return 0.0
    return float(np.mean(np.std(genotypes, axis=0)))
