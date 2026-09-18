# Make sure the group_28 folder is in the assignments/assignment_1 folder, and that the project root is the current working directory.
# Run from the project root, e.g.:
#   cd C:(...)EvolutionaryComputing2026
#   uv run assignments\assignment_1\group_28\run.py static --seed 1 (example)

import argparse
import csv
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
import ea


# ---------------------------------------------------------------------------
# Experiment I/O
# ---------------------------------------------------------------------------

def run_dir(variant: str, seed: int) -> Path:
    path = config.RESULTS_DIR / variant / f"seed_{seed}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv_summary(variant: str, seed: int, rows: list[dict]) -> Path:
    folder = run_dir(variant, seed)
    path = folder / config.RESULT_FILE_NAME

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=config.CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return path


# ---------------------------------------------------------------------------
# Fitness stats
# ---------------------------------------------------------------------------

def fitness_stats(population: list[dict]) -> tuple[float, float, float]:
    """(best, mean, std) of scored individuals' fitness; 0.0 each if none are scored yet."""
    fitnesses = [individual["fitness"] for individual in population if individual.get("fitness") is not None]
    if not fitnesses:
        return 0.0, 0.0, 0.0
    return min(fitnesses), sum(fitnesses) / len(fitnesses), statistics.pstdev(fitnesses)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run(variant: str, seed: int) -> Path:
    random.seed(seed)

    population = ea.evaluate(ea.init_population())
    mutation = ea.make_mutation(variant)

    rows: list[dict] = []
    for generation in range(1, config.NUM_GENERATIONS + 1):
        population = ea.run_generation(variant, population, mutation)
        best, mean, std = fitness_stats(population)

        rows.append(
    {
        "generation": generation,
        "variant": variant,
        "seed": seed,
        "best_fitness": round(best, 3),
        "mean_fitness": round(mean, 3),
        "std_fitness": round(std, 3),
        "mutation_probability": round(getattr(mutation, "probability", 0.0), 3),
        "mutation_strength": (
            round(mutation.strength, 3)
            if getattr(mutation, "strength", None) is not None
            else None
        ),
    }
)

    result_path = write_csv_summary(variant, seed, rows)
    print(f"{variant} seed={seed} -> {result_path}")
    return result_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run an EA experiment for Assignment 1.")
    parser.add_argument("variant", choices=list(config.VARIANTS), help="EA variant to run.")
    parser.add_argument(
        "--seed",
        type=int,
        nargs="*",
        default=[config.DEFAULT_SEED],
        help="Seed or seeds to use. Defaults to a single seed: %(default)s.",
    )
    args = parser.parse_args()

    for seed in args.seed:
        started = time.perf_counter()
        run(args.variant, seed)
        print(f"  ({time.perf_counter() - started:.1f}s)")


if __name__ == "__main__":
    main()
