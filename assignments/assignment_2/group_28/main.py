# Run from the project root (EvolutionaryComputing2026), for example:
#   uv run assignments/assignment_2/group_28/main.py --workers 16                  (full experiment: all variants, all 20 seeds)
#   uv run assignments/assignment_2/group_28/main.py --variant static --seed 101   (one run)
#   uv run assignments/assignment_2/group_28/main.py --tag smoke --seed 1 2 --pop-size 6 --generations 3 --workers 8   (quick test)
#
# Every run is saved in results/<tag>/<body>/<variant>/seed_<seed>/: the per-generation csv, the
# best genotype (best.npy) and run_info.json. Runs that already have a run_info.json are skipped.

import argparse
import csv
import json
import multiprocessing as mp
import platform
import random
import time
from pathlib import Path

import mujoco as mj
import numpy as np

from ariel.ec import set_seed

import config
import evolve
from mutation import make_mutation


def apply_overrides(overrides: dict) -> None:
    """Command line overrides of config. Also called inside every worker process, since those
    import a fresh config."""
    if overrides.get("pop_size"):
        config.POP_SIZE = overrides["pop_size"]
    if overrides.get("generations"):
        config.NUM_GENERATIONS = overrides["generations"]
    if overrides.get("sigma") is not None:
        config.STATIC_SIGMA = overrides["sigma"]
        config.ADAPTIVE_INITIAL_SIGMA = overrides["sigma"]


def run_dir(tag: str, variant: str, seed: int) -> Path:
    path = config.RESULTS_DIR / tag / config.BODY_NAME / variant / f"seed_{seed}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_done(tag: str, variant: str, seed: int) -> bool:
    return (run_dir(tag, variant, seed) / "run_info.json").exists()


def make_row(generation: int, variant: str, seed: int, population, sigma_used: float | None, mutation) -> dict:
    best, mean, std = evolve.fitness_stats(population)
    success_rate = getattr(mutation, "last_success_rate", None)

    return {
        "generation": generation,
        "variant": variant,
        "seed": seed,
        "best_fitness": round(best, 6),
        "mean_fitness": round(mean, 6),
        "std_fitness": round(std, 6),
        "sigma": "nan" if sigma_used is None else round(sigma_used, 6),
        "success_rate": "nan" if success_rate is None else round(success_rate, 4),
        "num_scored_mutations": getattr(mutation, "last_num_scored", 0),
        "num_mutated": getattr(mutation, "last_num_mutated", 0),
        "mean_genotype_spread": round(evolve.genotype_spread(population), 6),
        "num_nan": evolve.num_failed,
        "evaluations": config.POP_SIZE * (generation + 1),
    }


def run(variant: str, seed: int, tag: str) -> Path:
    rng = np.random.default_rng(seed)
    random.seed(seed)
    set_seed(seed)
    started = time.perf_counter()

    population = evolve.evaluate(evolve.init_population(rng))
    mutation = make_mutation(variant, rng)
    sigma_start = getattr(mutation, "sigma", None)

    # Generation 0 is the evaluated initial population, so every variant starts from the same point
    rows = [make_row(0, variant, seed, population, sigma_start, mutation)]

    best = min(population, key=lambda individual: individual.fitness_)
    best_genotype = np.asarray(best.genotype, dtype=np.float64)
    best_fitness, best_generation = best.fitness_, 0

    for generation in range(1, config.NUM_GENERATIONS + 1):
        # The sigma used to make this generation's offspring, before adapt() changes it
        sigma_used = getattr(mutation, "sigma", None)
        population = evolve.run_generation(variant, population, mutation, rng)
        rows.append(make_row(generation, variant, seed, population, sigma_used, mutation))

        current = min(population, key=lambda individual: individual.fitness_)
        if current.fitness_ < best_fitness:
            best_fitness, best_generation = current.fitness_, generation
            best_genotype = np.asarray(current.genotype, dtype=np.float64)

        if generation % 10 == 0 or generation == config.NUM_GENERATIONS:
            elapsed = time.perf_counter() - started
            print(f"  {variant} seed={seed} gen {generation}/{config.NUM_GENERATIONS} "
                  f"best {best_fitness:.4f} ({elapsed:.0f}s)", flush=True)

    wall = time.perf_counter() - started
    folder = run_dir(tag, variant, seed)

    with (folder / config.RESULT_FILE_NAME).open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=config.CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    np.save(folder / "best.npy", best_genotype)

    evaluations = config.POP_SIZE * (config.NUM_GENERATIONS + 1)
    info = {
        "tag": tag,
        "body": config.BODY_NAME,
        "hinges": int(evolve.MODEL.nu),
        "genotype_length": int(best_genotype.size),
        "variant": variant,
        "seed": seed,
        "pop_size": config.POP_SIZE,
        "generations": config.NUM_GENERATIONS,
        "evaluations": evaluations,
        "control_alpha": config.CONTROL_ALPHA,
        "clock_freq": config.CLOCK_FREQ,
        "hidden_size": config.HIDDEN_SIZE,
        "sim_duration": config.SIM_DURATION,
        "sigma_start": sigma_start,
        "sigma_end": getattr(mutation, "sigma", None),
        "sigma_max": getattr(mutation, "max_sigma", None),
        "best_fitness": best_fitness,
        "best_found_at_generation": best_generation,
        "wall_seconds": round(wall, 1),
        "seconds_per_evaluation": round(wall / evaluations, 4),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "mujoco": mj.__version__,
        },
    }
    (folder / "run_info.json").write_text(json.dumps(info, indent=2))

    print(f"{variant} seed={seed} -> {folder} ({wall:.0f}s)", flush=True)
    return folder


def run_job(job: tuple) -> None:
    """One (variant, seed) run, in this process or in a worker."""
    variant, seed, tag, overrides, force = job
    apply_overrides(overrides)

    if not force and is_done(tag, variant, seed):
        print(f"{variant} seed={seed} already done, skipped", flush=True)
        return
    run(variant, seed, tag)


def run_all(variants: list[str], seeds: list[int], tag: str, overrides: dict, workers: int, force: bool) -> None:
    jobs = [(variant, seed, tag, overrides, force) for variant in variants for seed in seeds]
    apply_overrides(overrides)
    print(f"{len(jobs)} runs: variants={variants} seeds={seeds} pop={config.POP_SIZE} "
          f"generations={config.NUM_GENERATIONS} workers={workers} tag={tag}", flush=True)

    started = time.perf_counter()
    if workers <= 1:
        for done, job in enumerate(jobs, start=1):
            run_job(job)
            print(f"  [{done}/{len(jobs)}]", flush=True)
    else:
        # "spawn" so every worker starts clean and builds its own MuJoCo world
        with mp.get_context("spawn").Pool(processes=workers) as pool:
            for done, _ in enumerate(pool.imap_unordered(run_job, jobs), start=1):
                print(f"  [{done}/{len(jobs)}]", flush=True)

    print(f"Done: {len(jobs)} runs in {time.perf_counter() - started:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the experiment grid (variant x seed).")
    parser.add_argument("--variant", nargs="*", choices=config.VARIANTS, default=list(config.VARIANTS),
                        help="Variant(s) to run. Defaults to: %(default)s.")
    parser.add_argument("--seed", type=int, nargs="*", default=list(config.SEEDS),
                        help="Seed(s) to run. Defaults to config.SEEDS.")
    parser.add_argument("--pop-size", type=int, help="Overrides config.POP_SIZE.")
    parser.add_argument("--generations", type=int, help="Overrides config.NUM_GENERATIONS.")
    parser.add_argument("--sigma", type=float,
                        help="Overrides both config.STATIC_SIGMA and config.ADAPTIVE_INITIAL_SIGMA.")
    parser.add_argument("--tag", default="main", help="Results subfolder. Defaults to: %(default)s.")
    parser.add_argument("--workers", type=int, default=1, help="Runs in parallel. Defaults to: %(default)s.")
    parser.add_argument("--force", action="store_true", help="Also redo runs that are already done.")
    args = parser.parse_args()

    overrides = {"pop_size": args.pop_size, "generations": args.generations, "sigma": args.sigma}
    run_all(args.variant, args.seed, args.tag, overrides, args.workers, args.force)


if __name__ == "__main__":
    main()
