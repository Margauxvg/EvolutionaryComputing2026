# Make sure the group_28 folder is in the assignments/assignment_1 folder, and that the project root is the current working directory.
# Run from the project root, e.g.:
#   cd C:(...)EvolutionaryComputing2026
#   uv run assignments\assignment_1\group_28\run.py static --seed 1 (example)

# CHANGES 
#   1. baseline_regenerate now actually regenerates a fresh random population
#      every generation (it was a no-op before -> baseline never did anything
#      after generation 1).
#   2. mutate_adaptive is now genuinely self-adaptive: an AdaptiveMutation
#      object tracks a mutation probability across generations and updates it
#      via Rechenberg's 1/5 success rule after each generation's evaluation.
#   3. reproduction() now records which fitness each offspring should be
#      compared against ("parent_fitness"), needed for the 1/5 rule.
#   4. Only OFFSPRING get mutated -- survivors/elites carried over from the
#      previous generation are left untouched (previously everyone in the
#      combined population got mutated, including individuals selection had
#      already decided to protect). --> need to discuss together whether we want to do this
#   5. mutate() now returns whether it actually changed the genome, instead of
#      silently returning the original genome on failure -- needed to compute
#      an honest success rate.
#   6. Removed the risk of stale "offspring"/tracking data leaking into future
#      generations by always building brand-new dicts for offspring (the old
#      code did `parent_a.copy()` in the no-crossover branch, which carried
#      over whatever transient keys that parent dict happened to have).
#   7. CSV output now includes the mutation probability used in that
#      generation (constant for "static", None for "baseline", live-tracked
#      for "adaptive") so you can plot it against fitness in your report.
#   8. We need to discuss whether we want to adaptive mutation rate entails IF a mutation happens or how MANY mutations happen; and in the script we need to change a standard mutation operator, now there are still four

#
# NEW CONFIG CONSTANTS TO config.py:
#   ADAPTIVE_INITIAL_PROBABILITY, ADAPTIVE_TARGET_SUCCESS, ADAPTIVE_FACTOR,
#   ADAPTIVE_MIN_PROBABILITY, ADAPTIVE_MAX_PROBABILITY
# Also added "mutation_probability" to config.CSV_COLUMNS.

import argparse
import csv
import random
import statistics
import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from ariel.body_phenotypes.robogen_lite.decoders._blueprint import load_graph_from_json
from ariel.ec.genotypes.tree.operators import (
    random_tree as random_genome,
    crossover_subtree as crossover_operator,
    get_tree_depth,
    mutate_replace_node,
    mutate_subtree_replacement,
    mutate_shrink,
    mutate_hoist,
)
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from tree_edit_distance import mean_plus_std_tree_edit_distance as fitness_function

VariantName = config.VARIANTS

# ---------------------------------------------------------------------------
# 1. Configuration helpers
# ---------------------------------------------------------------------------

def run_dir(variant: VariantName, seed: int) -> Path:
    """Return the folder used for one experiment run."""
    path = config.RESULTS_DIR / variant / f"seed_{seed}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv_summary(variant: VariantName, seed: int, rows: list[dict]) -> Path:
    """Store a CSV table with resutls for plotting. """
    folder = run_dir(variant, seed)
    path = folder / config.RESULT_FILE_NAME

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=config.CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return path


# ---------------------------------------------------------------------------
# 2. Targets and actual fitness metric
# ---------------------------------------------------------------------------

def load_targets(target_dir: Path = config.TARGET_DIR) -> list:
    """Load the target robot bodies directly with the library loader."""
    paths = sorted(target_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No target bodies found in {target_dir}")
    return [load_graph_from_json(path) for path in paths]

TARGETS = load_targets()

# ---------------------------------------------------------------------------
# 4. Shared EA steps
# ---------------------------------------------------------------------------
def create_random_individual() -> dict:
    """A single random individual, ready to be evaluated."""
    return {
        "genotype": random_genome(max_modules=config.NUM_OF_MODULES).to_dict(),
        "fitness": None,
        "alive": True,
    }

def evaluate(population: list[dict]) -> list[dict]:
    """Evaluate each individual with the assignment fitness function."""
    for individual in population:
        if individual.get("fitness") is None:
            genotype = individual["genotype"]
            body = TreeGenome.from_dict(genotype).to_networkx()
            individual["fitness"] = fitness_function(body, TARGETS)
    return population


def parent_selection(population: list[dict], tournament_size: int = config.TOURNAMENT_SIZE) -> list[dict]:
    """Select parents via tournament selection from alive individuals."""
    candidates = [ind for ind in population if ind.get("fitness") is not None and ind.get("alive", True)]
    parents: list[dict] = []

    for _ in range(len(candidates)):
        contestants = random.sample(candidates, tournament_size)
        winner = min(contestants, key=lambda individual: individual["fitness"])
        parents.append(winner)

    return parents


def reproduction(population: list[dict], crossover_probability: float = config.CROSSOVER_PROBABILITY) -> list[dict]:
    """Create offspring and keep them in the population for the next generation."""

    parents = parent_selection(population)
    if len(parents) < 2:
        return population

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
            offspring.append({
                "genotype": genotype,
                "fitness": None,
                "alive": True,
                "offspring": True,
                "parent_fitness": baseline,
            })

    population.extend(offspring)
    return population

def mutate(genome: TreeGenome, mutation_probability: float = config.STATIC_MUTATION_PROBABILITY) -> TreeGenome:
    """Apply a valid ARIEL mutation operator chosen by probability."""
    if random.random() >= mutation_probability:
        return genome, False

    for _ in range(config.MAX_MUTATION_ATTEMPTS): # Try to mutate the genome up to MAX_MUTATION_ATTEMPTS times
        candidate = TreeGenome.from_dict(genome.to_dict())

        operator_name = random.choices(
            ["replace_node", "subtree_replacement", "shrink", "hoist"],
            weights=[0.60, 0.10, 0.15, 0.15],
        )[0]

        if operator_name == "replace_node":
            mutate_replace_node(candidate)
        elif operator_name == "subtree_replacement":
            mutate_subtree_replacement(candidate, max_modules=config.NUM_OF_MODULES)
        elif operator_name == "shrink":
            mutate_shrink(candidate)
        elif operator_name == "hoist":
            mutate_hoist(candidate)

        if len(candidate.nodes) <= config.MAX_TOTAL_MODULES and get_tree_depth(candidate) <= config.MAX_TREE_DEPTH:
            return candidate, True

    return genome, False

def survivor_selection(population: list[dict], target_population_size: int = config.POP_SIZE) -> list[dict]:
    """Generational replacement with elitism: keep the best of old + offspring."""

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
# 5. Variants
# ---------------------------------------------------------------------------

def mutate_static(population: list[dict], mutation_probability: float = config.STATIC_MUTATION_PROBABILITY) -> list[dict]:
    mutated: list[dict] = []
    """ Only individuals tagged 'offspring' (fresh
    this generation) are eligible; survivors/elites pass through untouched.
    """
    for individual in population:
        if not individual.get("offspring"):
            mutated.append(individual)
            continue
        
        genome = TreeGenome.from_dict(individual["genotype"])
        new_genome, _was_mutated = mutate(genome, mutation_probability)
        mutated.append({
            "genotype": new_genome.to_dict(),
            "fitness": None,
            "alive": True,
        })
        
    return mutated

class AdaptiveMutation:
    """Mutation PROBABILITY controlled by Rechenberg's 1/5-success rule.
 
    `probability` is the chance that a given offspring gets mutated at all
    (the operator choice and its strength stay fixed -- see `mutate()` --
    this isolates probability as the one thing that differs from the static
    variant). After each generation's offspring are evaluated, `adapt`
    compares each mutated offspring's fitness to the parent fitness recorded
    at reproduction time: if more than 1/5 of mutated offspring improved on
    their parent, the probability grows; otherwise it shrinks.
 
    `_pending` holds direct references to this generation's freshly mutated
    offspring dicts (paired with the parent fitness to compare against).
    Because `evaluate()` fills in `entry["fitness"]` on those SAME dict
    objects afterwards, `adapt()` can just read `entry["fitness"]` straight
    off -- no flags stored on the population data itself, so there is nothing
    that could leak into a later generation.
    """
 
    def __init__(
        self,
        initial_probability: float = config.ADAPTIVE_INITIAL_PROBABILITY,
        target_success: float = config.ADAPTIVE_TARGET_SUCCESS,
        factor: float = config.ADAPTIVE_FACTOR,
        min_probability: float = config.ADAPTIVE_MIN_PROBABILITY,
        max_probability: float = config.ADAPTIVE_MAX_PROBABILITY,
    ):
        self.probability = initial_probability
        self.target_success = target_success
        self.factor = factor
        self.min_probability = min_probability
        self.max_probability = max_probability
        self._pending: list[tuple[dict, float]] = []
 
    def mutate(self, population: list[dict]) -> list[dict]:
        mutated: list[dict] = []
        self._pending = []
 
        for individual in population:
            if not individual.get("offspring"):
                mutated.append(individual)
                continue
 
            genome = TreeGenome.from_dict(individual["genotype"])
            new_genome, was_mutated = mutate(genome, self.probability)
            entry = {
                "genotype": new_genome.to_dict(),
                "fitness": None,
                "alive": True,
            }
            mutated.append(entry)
 
            if was_mutated:
                self._pending.append((entry, individual["parent_fitness"]))
 
        return mutated
 
    def adapt(self, population: list[dict]) -> list[dict]:
        """Call this AFTER evaluate() has scored the new offspring."""
        if not self._pending:
            return population  # nothing was mutated this generation
 
        successes = sum(
            1 for entry, parent_fitness in self._pending
            if entry["fitness"] is not None and entry["fitness"] < parent_fitness  # minimisation
        )
        success_rate = successes / len(self._pending)
 
        if success_rate > self.target_success:
            self.probability = min(self.max_probability, self.probability * self.factor)
        elif success_rate < self.target_success:
            self.probability = max(self.min_probability, self.probability / self.factor)
        # exactly on target: leave probability unchanged
 
        self._pending = []  # reset; nothing carries over to the next generation
        return population

def baseline_regenerate(population: list[dict]) -> list[dict]:
    """Random search: a completely fresh random population every generation,
    at the same population size as the evolutionary variants (so the total
    evaluation budget matches across variants).
    """
    return [create_random_individual() for _ in range(config.POP_SIZE)]


# ---------------------------------------------------------------------------
# 6. Variant pipelines
# ---------------------------------------------------------------------------

def build_pipeline(variant: VariantName) -> list:
    """Return the ordered list of EA operations for each variant."""
    match variant:
        case "baseline":
            return [baseline_regenerate, evaluate], None
        case "static":
            return [reproduction, mutate_static, evaluate, survivor_selection], None
        case "adaptive":
            adaptive = AdaptiveMutation()
            return [reproduction, adaptive.mutate, evaluate, adaptive.adapt, survivor_selection], adaptive
        case _:
            raise ValueError(f"Unknown variant: {variant}")


# ---------------------------------------------------------------------------
# 7. Experiment runner
# ---------------------------------------------------------------------------
def current_mutation_probability(variant: VariantName, adaptive: "AdaptiveMutation | None") -> float | None:
    """What to log in the CSV for this generation's mutation probability."""
    if variant == "static":
        return config.STATIC_MUTATION_PROBABILITY
    if variant == "adaptive":
        return adaptive.probability if adaptive is not None else None
    return None  # baseline: no mutation probability concept


def run(variant: VariantName, seed: int) -> Path:
    """Run one experiment and store per-generation fitness summaries."""
    random.seed(seed)

    population = [create_random_individual() for _ in range(config.POP_SIZE)]
    population = evaluate(population)
 
    pipeline, adaptive = build_pipeline(variant)
    rows: list[dict] = []
    
    for generation in range(1, config.NUM_GENERATIONS + 1):
        for step in pipeline:
            population = step(population)

        fitnesses = [individual["fitness"] for individual in population if individual.get("fitness") is not None]
        if fitnesses:
            best = min(fitnesses)
            mean = sum(fitnesses) / len(fitnesses)
            std = statistics.pstdev(fitnesses)
        else:
            best = mean = std = 0.0

        rows.append(
            {
                "generation": generation,
                "variant": variant,
                "seed": seed,
                "best_fitness": round(best, 3),
                "mean_fitness": round(mean,3),
                "std_fitness": round(std, 3),
                "mutation_probability": current_mutation_probability(variant, adaptive),
            }
        )

    result_path = write_csv_summary(variant, seed, rows)
    print(f"{variant} seed={seed} -> {result_path}")
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an EA experiment for Assignment 1.")
    parser.add_argument("variant", choices=["static", "adaptive", "baseline"], help="EA variant to run.")
    parser.add_argument(
        "--seed",
        type=int,
        nargs="*",
        default=[config.DEFAULT_SEED],
        help="Seed or seeds to use. Defaults to a single seed: %(default)s.",
    )
    args = parser.parse_args()

    seeds = args.seed
    for seed in seeds:
        started = time.perf_counter()
        run(args.variant, seed)
        print(f"  ({time.perf_counter() - started:.1f}s)")


if __name__ == "__main__":
    main()
