# Run from the project root (after making sure that group_28 is under assignment2 folder), such as:
#   cd C:(...)EvolutionaryComputing2026
# the full experiment as config.py describes it (all four variants, all 20 seeds): uv run assignments/assignment_2/group_28/run.py --workers 16
# one run: uv run assignments/assignment_2/group_28/run.py --variant static --seed 101
# a quick check before a long batch: tiny budget, kept apart from the real results by --tag
#    uv run assignments/assignment_2/group_28/run.py --tag smoke --seed 1 2 --pop-size 6 --generations 3 --workers 8


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
import ea
import mutation as mutation_module


def apply_overrides(overrides: dict) -> None:
    """Set config values for this process. Must run before the first run starts."""
    if overrides.get("pop_size"):
        config.POP_SIZE = overrides["pop_size"]
    if overrides.get("generations"):
        config.NUM_GENERATIONS = overrides["generations"]
    if overrides.get("sigma") is not None:
        # Both must move together: the two variants only differ by HOW sigma is controlled,
        # never by where it starts. mutation.make_mutation enforces this too, as a second
        # guard for the case config.py itself is edited without going through here.
        config.STATIC_SIGMA = overrides["sigma"]
        config.ADAPTIVE_INITIAL_SIGMA = overrides["sigma"]


def body_name() -> str:
    # The body is fixed (simulate.build_robot builds the John Set gecko), so this is only the
    # folder label. run_info.json also records the hinge count read from the compiled model.
    return config.BODY_NAME


def run_dir(tag: str, variant: str, seed: int) -> Path:
    path = config.RESULTS_DIR / tag / body_name() / variant / f"seed_{seed}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_done(tag: str, variant: str, seed: int) -> bool:
    """True if this (tag, body, variant, seed) already has a complete result.

    Checked against run_info.json rather than the CSV: run_info.json is written only at the
    very end of run(), after the CSV, best.npy and the file itself are all in place. A run
    that crashed or was interrupted mid-write leaves no run_info.json, so it is correctly
    treated as not done and gets re-run.
    """
    return (run_dir(tag, variant, seed) / "run_info.json").exists()


def make_row(
    generation: int,
    variant: str,
    seed: int,
    population,
    sigma_used: float | None,
    mutation,
) -> dict:
    best, mean, std = ea.fitness_stats(population)
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
        "mean_genotype_spread": round(ea.genotype_spread(population), 6),
        # Failed controllers among this generation's NEW evaluations (offspring, or the whole
        # population at generation 0 and for the baseline), not among the survivors.
        "num_nan": ea.last_num_failed,
        # Initial population plus POP_SIZE new individuals per generation. Identical for the
        # EA and the baseline, which is what makes the comparison equal-budget.
        "evaluations": config.POP_SIZE * (generation + 1),
    }


def run(variant: str, seed: int, tag: str) -> Path:
    rng = np.random.default_rng(seed)
    random.seed(seed)
    set_seed(seed)

    started = time.perf_counter()
    population = ea.evaluate(ea.init_population(rng, size=config.POP_SIZE))
    mutation = mutation_module.make_mutation(variant, rng)
    sigma_start = getattr(mutation, "sigma", None)

    # Generation 0: the evaluated initial population, before any variation, so every
    # variant's curve starts from the same point.
    rows = [make_row(0, variant, seed, population, sigma_start, mutation)]

    best = min(population, key=lambda ind: ind.fitness_)
    best_genotype = np.asarray(best.genotype, dtype=np.float64)
    best_fitness, best_generation = best.fitness_, 0

    for generation in range(1, config.NUM_GENERATIONS + 1):
        # The sigma that PRODUCES this generation's offspring, read before adapt() moves it.
        sigma_used = getattr(mutation, "sigma", None)
        population = ea.run_generation(variant, population, mutation, rng)
        rows.append(make_row(generation, variant, seed, population, sigma_used, mutation))

        # Best-so-far, tracked here because the baseline replaces its whole population.
        current = min(population, key=lambda ind: ind.fitness_)
        if current.fitness_ < best_fitness:
            best_fitness, best_generation = current.fitness_, generation
            best_genotype = np.asarray(current.genotype, dtype=np.float64)

        if generation % 10 == 0 or generation == config.NUM_GENERATIONS:
            elapsed = time.perf_counter() - started
            eta = elapsed / generation * (config.NUM_GENERATIONS - generation)
            print(
                f"  {body_name()} {variant} seed={seed} gen {generation}/"
                f"{config.NUM_GENERATIONS} best-so-far {best_fitness:.4f} "
                f"({elapsed:.0f}s, ~{eta:.0f}s left)",
                flush=True,
            )

    wall = time.perf_counter() - started
    folder = run_dir(tag, variant, seed)

    with (folder / config.RESULT_FILE_NAME).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=config.CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    np.save(folder / "best.npy", best_genotype)

    model = ea.simulator().model
    evaluations = config.POP_SIZE * (config.NUM_GENERATIONS + 1)
    info = {
        "tag": tag,
        "body": body_name(),
        "hinges": int(model.nu),
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
        "sigma_max": getattr(mutation, "max_sigma", None),  # None for static and baseline
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

    print(f"{body_name()} {variant} seed={seed} -> {folder}  ({wall:.0f}s)", flush=True)
    return folder


def run_job(job: tuple[str, int, str, dict, bool]) -> str:
    """Entry point for one (variant, seed) job, in this or a worker process."""
    variant, seed, tag, overrides, force = job
    apply_overrides(overrides)

    if not force and is_done(tag, variant, seed):
        folder = run_dir(tag, variant, seed)
        print(f"{body_name()} {variant} seed={seed} -> {folder}  (skipped, already done)",
              flush=True)
        return str(folder)

    return str(run(variant, seed, tag))


def run_all(
    variants: list[str],
    seeds: list[int],
    tag: str,
    overrides: dict,
    workers: int,
    force: bool = False,
) -> None:
    jobs = [(variant, seed, tag, overrides, force) for variant in variants for seed in seeds]
    apply_overrides(overrides)  # so the summary below shows the effective values

    already_done = 0 if force else sum(
        is_done(tag, variant, seed) for variant, seed, *_ in jobs
    )
    print(
        f"{len(jobs)} runs: body={body_name()} variants={variants} seeds={seeds} "
        f"pop={config.POP_SIZE} generations={config.NUM_GENERATIONS} "
        f"workers={workers} tag={tag}"
        + (f" ({already_done} already done, will be skipped)" if already_done else ""),
        flush=True,
    )

    started = time.perf_counter()
    if workers <= 1:
        for done, job in enumerate(jobs, start=1):
            run_job(job)
            print(f"  [{done}/{len(jobs)}]", flush=True)
    else:
        # "spawn" on every platform: it is the macOS default, and using it everywhere means
        # Linux and macOS runs behave the same.
        with mp.get_context("spawn").Pool(processes=workers) as pool:
            for done, _ in enumerate(pool.imap_unordered(run_job, jobs), start=1):
                print(f"  [{done}/{len(jobs)}]", flush=True)

    print(f"Done: {len(jobs)} runs in {time.perf_counter() - started:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the A2 experiment grid (variant x seed).",
    )
    parser.add_argument(
        "--variant",
        nargs="*",
        choices=config.VARIANTS,
        default=list(config.VARIANTS),
        help="Variant(s) to run. Default: %(default)s.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        nargs="*",
        default=list(config.SEEDS),
        help="Seed(s) to run. Default: config.SEEDS.",
    )
    parser.add_argument("--pop-size", type=int, help="Override config.POP_SIZE.")
    parser.add_argument("--generations", type=int, help="Override config.NUM_GENERATIONS.")
    parser.add_argument(
        "--sigma",
        type=float,
        help="Override config.STATIC_SIGMA and config.ADAPTIVE_INITIAL_SIGMA together.",
    )
    parser.add_argument(
        "--tag",
        default="main",
        help="Results subfolder, e.g. 'smoke'. Default: %(default)s.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Runs in parallel, one process each. Use 16 for the main experiment on an "
             "otherwise idle 8-core laptop (report/parallelisation_plan.md). Default: %(default)s.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run every job even if a complete result already exists for it.",
    )
    args = parser.parse_args()

    overrides = {
        "pop_size": args.pop_size,
        "generations": args.generations,
        "sigma": args.sigma,
    }
    run_all(args.variant, args.seed, args.tag, overrides, args.workers, args.force)


if __name__ == "__main__":
    main()
