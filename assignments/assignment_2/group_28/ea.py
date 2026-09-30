"""The evolutionary algorithm. A port of A1's ea.py to real-valued genomes.

WHAT THIS FILE IS
-----------------
One generation of the EA, and the pieces it is built from. Nothing in here imports mujoco -
it asks simulate.py for a fitness and otherwise deals only in numbers, so the whole algorithm
can be tested against a cheap synthetic objective without waiting on rollouts.

Structure follows A1's ea.py function for function, because that structure worked and the
report has to describe it either way:

    A1                               A2
    evaluate(population)             same, via simulate.Simulator
    init_population()                random float vectors instead of random trees
    parent_selection()               same tournament, unchanged
    reproduction()                   clone selected parents; mutation is the only variation
    survivor_selection()             same elitism + generational replacement
    baseline_regenerate()            same
    run_generation()                 same five steps in the same order

WHAT IT NEEDS FROM mutation.py
------------------------------
`mutation.py` owns the one thing the research question is about, and this file only calls it:

    make_mutation(variant, rng) -> MutationVariant | None     None for "baseline"

    MutationVariant:
        sigma                  float, current step size (read before offspring are made,
                               so the logged value is the one that produced them)
        mutate(offspring)      -> Population, Gaussian perturbation at `sigma`
        adapt(population)      -> Population, called AFTER evaluate; measures the success
                               rate and, for the adaptive variant, updates sigma
        last_success_rate      float | None
        last_num_scored        int
        last_num_mutated       int

Same interface as A1's, with `probability` renamed `sigma`. The `parent_fitness` bookkeeping the
1/5 success measure needs is set here in `reproduction` and read there.

WHY ariel.ec.Individual RATHER THAN A1's DICTS
----------------------------------------------
The A2 template says outright: "Build a proper EA on top of ariel.ec. You are expected to use
that module - it gives you the population/individual data model". So the data model is theirs
and the search is ours, which is also what the brief's rules require.

One consequence: `Individual.genotype_` is a JSON column, so genotypes are stored as lists and
converted to arrays at the point of use. The extra per-individual fields A1 hung off its dicts
live in `Individual.tags`, which is also JSON.

NO CROSSOVER
------------
Mutation is the only variation operator. That is a deliberate choice, not an omission, and
report/stage1_findings.md section 9 is the argument: with crossover at p = 0.7 only ~15 of 50
offspring per generation could be scored for the 1/5 success rate, because an improvement over a
parent cannot be attributed to mutation if crossover also touched the child. Removing it takes
that to 50 and cuts the standard error of the measured success rate from 0.046 to 0.025 - against
a target of 0.2, which is the only number the rule reads.

It also puts the rule in its native setting. Rechenberg derived it for evolution strategies, where
mutation IS the step distribution; A1 applied it to a discrete mutation probability in a GA and
the report called that "an analogy rather than a transfer".

SEEDING
-------
Only one generator matters now:
    np.random.default_rng     - all of our sampling, passed explicitly as `rng`
run.py does the seeding; this file takes an `rng` argument so nothing here reads global state.
Nothing we call draws from a package-level RNG. If an ariel.ec operator is ever added back,
ariel.ec.set_seed(seed) has to be called too or every "independent" seed shares its draws.
"""

# Standard library
import statistics

# Third-party libraries
import numpy as np
import numpy.typing as npt

from ariel.ec import Individual, Population

import config
import controller
from simulate import get_simulator

# Built on first use, once per process. MuJoCo models do not pickle, so a multiprocessing
# worker has to compile its own - which is what get_simulator's cache is for.
_SIM = None


def simulator():
    """This process's Simulator, and the genotype length that goes with its body."""
    global _SIM
    if _SIM is None:
        _SIM = get_simulator(controller.act)
    return _SIM


def genotype_length() -> int:
    """Length of the flat weight vector. Read from the compiled model, never hardcoded."""
    return controller.genotype_length(simulator().model)


# --------------------------------------------------------------------------- #
#  Evaluation
# --------------------------------------------------------------------------- #
def evaluate(population: Population) -> Population:
    """Score every individual that does not have a fitness yet. LOWER IS BETTER.

    A1's evaluate() decoded a tree and measured graph distance. Here it runs a 15 s physics
    rollout - about 0.45 s per individual - which is why nothing else in this file is allowed
    to be wasteful about calling it.
    """
    sim = simulator()
    for individual in population.unevaluated:
        genotype = np.asarray(individual.genotype, dtype=np.float64)
        individual.fitness = sim.evaluate(genotype)
    return population


def count_nan(population: Population) -> int:
    """Individuals whose controller produced NaN or inf and scored WORST_FITNESS.

    Worth logging: a rising count means mutation is pushing weights into overflow, which is
    invisible in the fitness curve because those individuals are simply never selected.
    """
    return sum(1 for ind in population if ind.fitness_ == config.WORST_FITNESS)


def genotype_spread(population: Population) -> float:
    """Mean per-gene standard deviation across the population.

    A1 worked out its convergence story after the fact and lost the chance to show it. This is
    logged from generation 0 so that if the population collapses, the collapse is evidence
    rather than inference.
    """
    genotypes = np.asarray(
        [np.asarray(ind.genotype, dtype=np.float64) for ind in population],
    )
    if len(genotypes) < 2:
        return 0.0
    return float(np.mean(np.std(genotypes, axis=0)))


# --------------------------------------------------------------------------- #
#  Initialisation
# --------------------------------------------------------------------------- #
def make_individual(genotype: npt.NDArray[np.float64], **tags) -> Individual:
    """An unevaluated Individual holding a flat weight vector.

    Replaces A1's helpers.make_individual. `genotype_` is a JSON column, hence .tolist().
    """
    individual = Individual()
    individual.genotype = genotype.tolist()
    if tags:
        individual.tags = tags
    return individual


def init_population(
    rng: np.random.Generator,
    size: int = config.POP_SIZE,
) -> Population:
    """Random controllers, drawn like the template's make_random_weights."""
    n = genotype_length()
    return Population([
        make_individual(rng.normal(scale=config.INIT_WEIGHT_SCALE, size=n))
        for _ in range(size)
    ])


# --------------------------------------------------------------------------- #
#  Selection
# --------------------------------------------------------------------------- #
def parent_selection(
    population: Population,
    rng: np.random.Generator,
    tournament_size: int = config.TOURNAMENT_SIZE,
) -> list[Individual]:
    """Tournament selection. Unchanged from A1 except for the container type.

    ariel ships no parent-selection operator - ec/archive.py has a database-backed tournament,
    which is a different thing - so this is ours.

    `min` because fitness is a minimisation: the lowest distance wins.
    """
    candidates = [ind for ind in population.alive if ind.fitness_ is not None]
    parents: list[Individual] = []

    for _ in range(len(candidates)):
        contestants = [candidates[i] for i in rng.choice(len(candidates), tournament_size, replace=False)]
        parents.append(min(contestants, key=lambda ind: ind.fitness_))

    return parents


def survivor_selection(parents: Population, offspring: Population) -> Population:
    """Generational replacement with elitism. Same scheme as A1.

    Deterministic truncation, so no tuning parameter can accidentally differ between the two
    variants - the only thing that differs is the mutation strategy.
    """
    elite_count = max(1, int(config.ELITISM_RATIO * config.POP_SIZE))

    parent_elites = sorted(parents, key=lambda ind: ind.fitness_)[:elite_count]
    ranked_offspring = sorted(offspring, key=lambda ind: ind.fitness_)
    survivors = parent_elites + ranked_offspring[: config.POP_SIZE - elite_count]

    for individual in survivors:
        individual.alive = True

    return Population(survivors)


# --------------------------------------------------------------------------- #
#  Variation
# --------------------------------------------------------------------------- #
def reproduction(population: Population, rng: np.random.Generator) -> Population:
    """Clone the selected parents, and label each child with the fitness it has to beat.

    One label, `parent_fitness`: the fitness of the parent this child was copied from. mutation.py
    compares the child's own fitness against it to decide whether the mutation succeeded. A1
    carried a second label, `crossed`, to exclude recombined children from that measure; with no
    crossover every child is attributable, so the label is gone and so is the exclusion.

    No variation happens here - the children leave this function as exact copies. run_generation
    hands them straight to mutation.mutate, which is where all the variation lives.
    """
    offspring = [
        make_individual(
            np.asarray(parent.genotype, dtype=np.float64),
            parent_fitness=parent.fitness_,
        )
        for parent in parent_selection(population, rng)
    ]
    return Population(offspring)


def baseline_regenerate(rng: np.random.Generator) -> Population:
    """Random search: throw the population away and draw a fresh one.

    The brief requires a baseline at the same evaluation budget. Because this samples
    POP_SIZE new individuals per generation, it spends exactly the same number of evaluations
    as the EA - no separate accounting needed.
    """
    return init_population(rng)


# --------------------------------------------------------------------------- #
#  One generation
# --------------------------------------------------------------------------- #
def run_generation(
    variant: str,
    population: Population,
    mutation,
    rng: np.random.Generator,
) -> Population:
    """One generation. Same five steps, in the same order, as A1.

    The order matters for the research question: `adapt` runs AFTER `evaluate`, because the
    1/5 success rate can only be measured once the offspring have fitnesses to compare
    against their parents'.
    """
    if variant == "baseline":
        return evaluate(baseline_regenerate(rng))

    offspring = reproduction(population, rng)
    offspring = mutation.mutate(offspring)
    offspring = evaluate(offspring)
    offspring = mutation.adapt(offspring)
    return survivor_selection(population, offspring)


# --------------------------------------------------------------------------- #
#  Stats, for the CSV row run.py writes
# --------------------------------------------------------------------------- #
def fitness_stats(population: Population) -> tuple[float, float, float]:
    """(best, mean, std) over scored individuals. Same helper as A1's run.py."""
    fitnesses = [ind.fitness_ for ind in population if ind.fitness_ is not None]
    if not fitnesses:
        return 0.0, 0.0, 0.0
    return min(fitnesses), statistics.fmean(fitnesses), statistics.pstdev(fitnesses)
