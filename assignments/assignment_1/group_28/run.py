# Make sure the group_28 folder is in the assignments/assignment_1 folder, and that the project root is the current working directory.
# Run from the project root, e.g.:
#   cd C:(...)EvolutionaryComputing2026
#   uv run assignments\assignment_1\group_28\run.py static --seed 1 (example)

import argparse
import csv
import random
import statistics
import time
from pathlib import Path

import config
from ariel.body_phenotypes.robogen_lite.decoders._blueprint import load_graph_from_json
from ariel.ec.genotypes.tree.operators import random_tree as random_genome
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

def evaluate(population: list[dict]) -> list[dict]:
    """Evaluate each individual with the assignment fitness function."""
    for individual in population:
        if individual.get("fitness") is None:
            genotype = individual["genotype"]
            body = TreeGenome.from_dict(genotype).to_networkx()
            individual["fitness"] = fitness_function(body, TARGETS)
    return population


def parent_selection(population: list[dict], tournament_size: int = config.TOURNAMENT_SIZE) -> list[dict]:
    """Select parents via tournament selection."""
    parents = []
    for _ in range(len(population)):
        contestants = random.sample(population, tournament_size)
        winner = min(contestants, key=lambda individual: individual["fitness"])
        parents.append(winner)

    return parents

def crossover(population: list[dict], crossover_probability: float = config.CROSSOVER_PROBABILITY) -> list[dict]:
    """Placeholder for crossover step."""
    return population


def survivor_selection(population: list[dict], target_population_size: int = config.POP_SIZE) -> list[dict]:
    """Placeholder for truncation or elitist survivor selection."""
    return population


# ---------------------------------------------------------------------------
# 5. Variants
# ---------------------------------------------------------------------------

def mutate_static(population: list[dict], mutation_probability: float = config.STATIC_MUTATION_PROBABILITY) -> list[dict]:
    # TODO: Implement static mutation logic here
    return population


def mutate_adaptive(population: list[dict], mutation_probability: float = config.ADAPTIVE_MUTATION_PROBABILITY) -> list[dict]:
    # TODO: Implement adaptive mutation logic here (the parameters might have to change based on the success rate of mutations)
    return population


def baseline_regenerate(population: list[dict]) -> list[dict]:
    """Placeholder for random-search baseline."""
    return population


# ---------------------------------------------------------------------------
# 6. Variant pipelines
# ---------------------------------------------------------------------------

def build_pipeline(variant: VariantName) -> list:
    """Return the ordered list of EA operations for each variant."""
    match variant:
        case "baseline":
            return [baseline_regenerate, evaluate]
        case "static":
            return [parent_selection, crossover, mutate_static, evaluate, survivor_selection]
        case "adaptive":
            return [parent_selection, crossover, mutate_adaptive, evaluate, survivor_selection]
        case _:
            raise ValueError(f"Unknown variant: {variant}")


# ---------------------------------------------------------------------------
# 7. Experiment runner
# ---------------------------------------------------------------------------

def run(variant: VariantName, seed: int) -> Path:
    """Run one experiment and store per-generation fitness summaries."""
    random.seed(seed)

    population = [
        {
            "genotype": random_genome(max_modules=config.NUM_OF_MODULES).to_dict(),
            "fitness": None,
            "alive": True,
        }
        for _ in range(config.POP_SIZE)
    ]
    population = evaluate(population)

    pipeline = build_pipeline(variant)
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