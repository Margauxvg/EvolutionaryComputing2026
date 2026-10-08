import statistics

import numpy as np
import numpy.typing as npt

from ariel.ec import Individual, Population

import config
import controller
from simulate import get_simulator

# Built on first use, once per process. MuJoCo models do not pickle).
_SIM = None

# NaN/inf controllers (WORST_FITNESS) in the batch the last evaluate() call scored. Counted here
# and not among the survivors
last_num_failed = 0


def simulator():
    """This process's simulator."""
    global _SIM
    if _SIM is None:
        _SIM = get_simulator(controller.act)
    return _SIM


def genotype_length() -> int:
    """Length of the flat weight vector. Read from the compiled model."""
    return controller.genotype_length(simulator().model)


def evaluate(population: Population) -> Population:
    """Score every individual that does not have a fitness yet. LOWER IS BETTER."""
    global last_num_failed
    sim = simulator()
    failed = 0
    for individual in population.unevaluated:
        genotype = np.asarray(individual.genotype, dtype=np.float64)
        fitness = sim.evaluate(genotype)
        individual.fitness = fitness
        failed += fitness == config.WORST_FITNESS
    last_num_failed = int(failed)
    return population


def count_nan(population: Population) -> int:
    """Individuals in `population` whose controller produced NaN or inf (WORST_FITNESS)."""
    return sum(1 for ind in population if ind.fitness_ == config.WORST_FITNESS)


def genotype_spread(population: Population) -> float:
    """Mean per-gene standard deviation across the population (logged from generation 0)."""
    genotypes = np.asarray(
        [np.asarray(ind.genotype, dtype=np.float64) for ind in population],
    )
    if len(genotypes) < 2:
        return 0.0
    return float(np.mean(np.std(genotypes, axis=0)))


def make_individual(genotype: npt.NDArray[np.float64], **tags) -> Individual:
    """An unevaluated individual holding a flat weight vector."""
    individual = Individual()
    individual.genotype = genotype.tolist()
    if tags:
        individual.tags = tags
    return individual


def init_population(
    rng: np.random.Generator,
    size: int = config.POP_SIZE,
) -> Population:
    """Create random controllers."""
    n = genotype_length()
    return Population([
        make_individual(rng.normal(scale=config.INIT_WEIGHT_SCALE, size=n))
        for _ in range(size)
    ])


def parent_selection(
    population: Population,
    rng: np.random.Generator,
    tournament_size: int = config.TOURNAMENT_SIZE,
) -> list[Individual]:
    """Tournament selection. The lowest distance wins."""
    candidates = [ind for ind in population.alive if ind.fitness_ is not None]
    parents: list[Individual] = []

    for _ in range(len(candidates)):
        contestants = [candidates[i] for i in rng.choice(len(candidates), tournament_size, replace=False)]
        parents.append(min(contestants, key=lambda ind: ind.fitness_))

    return parents


def survivor_selection(parents: Population, offspring: Population) -> Population:
    """Generational replacement with elitism: the best parents plus the best offspring."""
    elite_count = max(1, int(config.ELITISM_RATIO * config.POP_SIZE))

    parent_elites = sorted(parents, key=lambda ind: ind.fitness_)[:elite_count]
    ranked_offspring = sorted(offspring, key=lambda ind: ind.fitness_)
    survivors = parent_elites + ranked_offspring[: config.POP_SIZE - elite_count]

    for individual in survivors:
        individual.alive = True

    return Population(survivors)


def reproduction(population: Population, rng: np.random.Generator) -> Population:
    """Clone the selected parents, and label each child with the fitness it has to beat."""
    offspring = [
        make_individual(
            np.asarray(parent.genotype, dtype=np.float64),
            parent_fitness=parent.fitness_,
        )
        for parent in parent_selection(population, rng)
    ]
    return Population(offspring)


def baseline_regenerate(rng: np.random.Generator) -> Population:
    """Random search: throw the population away and draw a fresh one."""
    return init_population(rng, size=config.POP_SIZE)


def run_generation(
    variant: str,
    population: Population,
    mutation,
    rng: np.random.Generator,
) -> Population:
    """Run one generation."""
    if variant == "baseline":
        return evaluate(baseline_regenerate(rng))

    offspring = reproduction(population, rng)
    offspring = mutation.mutate(offspring)
    offspring = evaluate(offspring)
    offspring = mutation.adapt(offspring)
    return survivor_selection(population, offspring)


def fitness_stats(population: Population) -> tuple[float, float, float]:
    """(best, mean, std) over scored individuals."""
    fitnesses = [ind.fitness_ for ind in population if ind.fitness_ is not None]
    if not fitnesses:
        return 0.0, 0.0, 0.0
    return min(fitnesses), statistics.fmean(fitnesses), statistics.pstdev(fitnesses)
