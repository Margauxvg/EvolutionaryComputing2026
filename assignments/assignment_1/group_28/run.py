# Run from the project root (after making sure that group_28 is under assignment1 folder), such as:
#   cd C:(...)EvolutionaryComputing2026
#   uv run assignments\assignment_1\group_28\run.py                 (all 5 default seeds)
#   uv run assignments\assignment_1\group_28\run.py --seed 1        (just seed 1)

import argparse
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
import ea
from helpers import variant_dir, write_csv


def run_dir(variant: str, operator: str | None, seed: int) -> Path:
    path = variant_dir(variant, operator) / f"seed_{seed}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv_summary(variant: str, operator: str | None, seed: int, rows: list[dict]) -> Path:
    folder = run_dir(variant, operator, seed)
    return write_csv(rows, folder / config.RESULT_FILE_NAME, fieldnames=config.CSV_COLUMNS)


def fitness_stats(population: list[dict]) -> tuple[float, float, float]:
    """(best, mean, std) of scored individuals' fitness; 0.0 each if none are scored yet."""
    fitnesses = [individual["fitness"] for individual in population if individual.get("fitness") is not None]
    if not fitnesses:
        return 0.0, 0.0, 0.0
    return min(fitnesses), sum(fitnesses) / len(fitnesses), statistics.pstdev(fitnesses)


def make_row(generation: int, variant: str, operator: str | None, seed: int, population: list[dict], probability_used: float, mutation) -> dict:
    best, mean, std = fitness_stats(population)
    success_rate = getattr(mutation, "last_success_rate", None)

    return {
        "generation": generation,
        "variant": variant,
        "operator": operator or "",
        "seed": seed,
        "best_fitness": round(best, 3),
        "mean_fitness": round(mean, 3),
        "std_fitness": round(std, 3),
        "mutation_probability": round(probability_used, 3),
        "success_rate": "nan" if success_rate is None else round(success_rate, 4),
        "num_scored_mutations": getattr(mutation, "last_num_scored", 0),
        "num_mutated": getattr(mutation, "last_num_mutated", 0),
    }


def run(variant: str, operator: str | None, seed: int) -> Path:
    random.seed(seed)

    population = ea.evaluate(ea.init_population())
    mutation = ea.make_mutation(variant, operator)

    # Generation 0 is the evaluated initial population, before any variation, so every variant's curve starts from the same point.
    rows: list[dict] = [
        make_row(0, variant, operator, seed, population, getattr(mutation, "probability", 0.0), mutation)
    ]

    for generation in range(1, config.NUM_GENERATIONS + 1):
        # Get the probability used while the offspring was produced
        probability_used = getattr(mutation, "probability", 0.0)
        population = ea.run_generation(variant, population, mutation)
        rows.append(make_row(generation, variant, operator, seed, population,
                             probability_used, mutation))

    result_path = write_csv_summary(variant, operator, seed, rows)
    print(f"{variant} operator={operator} seed={seed} -> {result_path}")
    return result_path


def run_all(seeds: list[int]) -> None:
    """Run all experiment combinations: static/adaptive x operator x seed, plus the random-search baseline (once
    per seed, since it does not depend on the operator)."""
    jobs: list[tuple[str, str | None]] = [
        (variant, operator)
        for variant in ("static", "adaptive")
        for operator in config.MUTATION_OPERATORS
    ]
    jobs.append(("baseline", None))

    total = len(jobs) * len(seeds)
    started_all = time.perf_counter()

    done = 0
    for variant, operator in jobs:
        for seed in seeds:
            done += 1
            started = time.perf_counter()
            run(variant, operator, seed)
            print(f"  [{done}/{total}] ({time.perf_counter() - started:.1f}s)")

    print(f"Done: {total} runs in {time.perf_counter() - started_all:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full EA experiment grid (static/adaptive x operator x seed, plus baseline)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        nargs="*",
        default=list(config.SEEDS),
        help="Seed or seeds to use. Defaults to: %(default)s.",
    )
    args = parser.parse_args()
    run_all(args.seed)


if __name__ == "__main__":
    main()
