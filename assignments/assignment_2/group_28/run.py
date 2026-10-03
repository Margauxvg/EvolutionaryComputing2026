"""Run the experiment grid: variant x seed, for one body. A port of A1's run.py.

Run from the project root:

    # the full experiment as config.py describes it (all variants, all seeds)
    uv run assignments/assignment_2/group_28/run.py

    # one run
    uv run assignments/assignment_2/group_28/run.py --variant static --seed 1

    # a pilot: override the body and the budget without editing config.py, and keep the
    # results apart from the real experiment with --tag
    uv run assignments/assignment_2/group_28/run.py --tag body_pilot --body gecko \
        --variant static --seed 1 2 3 --pop-size 30 --generations 40 --workers 3

WHAT IT WRITES
--------------
One folder per run:  results/<tag>/<body>/<variant>/seed_<n>/
    fitness_overview.csv   one row per generation, generation 0 included, columns from
                           config.CSV_COLUMNS - the same shape as A1 so analyze.py ports
    best.npy               the best genotype found in the run (for watch.py --load)
    run_info.json          everything needed to reproduce and report the run: body, sizes,
                           budget, sigma at start and end, best fitness and when it was
                           found, wall time, seconds per evaluation, library versions

OVERRIDES
---------
--body, --pop-size and --generations change config values for this invocation only. They are
re-applied inside every worker process, because a spawned worker imports config.py fresh.
config.py stays the description of the final experiment; pilots never require editing it.

RESUMING
--------
Before running a job, run_job() checks whether that job's run_info.json already exists and
is complete. If it does, the job is skipped rather than re-run. This means a crash, a sleeping
laptop, or a Ctrl-C partway through a long batch costs only the one run that was in progress
when it happened: restarting the exact same command picks up from there instead of redoing
everything. Pass --force to ignore existing results and re-run every job anyway.

SEEDING
-------
Each run seeds one numpy Generator (passed to everything that samples), plus `random` and
ariel.ec.set_seed for safety: nothing in our code draws from them, but if an ariel.ec
operator is ever added, its package-level RNG would otherwise be shared across "independent"
seeds. Evaluation itself is deterministic and draws nothing.

PARALLELISM
-----------
--workers N runs N (variant, seed) jobs at once, one process each. Every process builds its
own Simulator on first use (MuJoCo models cannot be pickled). Results are identical to a
sequential run, because each run depends only on its own seed.
"""

# Standard library
import argparse
import csv
import json
import multiprocessing as mp
import platform
import random
import time
from pathlib import Path

# Third-party libraries
import mujoco as mj
import numpy as np

from ariel.body_phenotypes.robogen_lite.prebuilt_robots import john_set
from ariel.ec import set_seed

import config
import ea
import mutation as mutation_module

BODIES = ("gecko", "spider_8")  # the John Set bodies under consideration


# --------------------------------------------------------------------------- #
#  Overrides
# --------------------------------------------------------------------------- #
def apply_overrides(overrides: dict) -> None:
    """Set config values for this process. Must run before the Simulator is built."""
    if overrides.get("body"):
        config.BUILD_BODY = getattr(john_set, overrides["body"])
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
    return config.BUILD_BODY.__name__


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


# --------------------------------------------------------------------------- #
#  One CSV row
# --------------------------------------------------------------------------- #
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
        "num_nan": ea.count_nan(population),
        # Initial population plus POP_SIZE new individuals per generation. Identical for the
        # EA and the baseline, which is what makes the comparison equal-budget.
        "evaluations": config.POP_SIZE * (generation + 1),
    }


# --------------------------------------------------------------------------- #
#  One run
# --------------------------------------------------------------------------- #
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
        "control_mode": config.CONTROL_MODE,
        "clock_freq": config.CLOCK_FREQ,
        "hidden_size": config.HIDDEN_SIZE,
        "sim_duration": config.SIM_DURATION,
        "sigma_start": sigma_start,
        "sigma_end": getattr(mutation, "sigma", None),
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


# --------------------------------------------------------------------------- #
#  The grid
# --------------------------------------------------------------------------- #
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
        description="Run the A2 experiment grid (variant x seed) for one body.",
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
    parser.add_argument(
        "--body",
        choices=BODIES,
        help="Override config.BUILD_BODY for this invocation.",
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
        help="Results subfolder, e.g. 'body_pilot'. Default: %(default)s.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Runs in parallel, one process each. Default: %(default)s.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run every job even if a complete result already exists for it.",
    )
    args = parser.parse_args()

    overrides = {
        "body": args.body,
        "pop_size": args.pop_size,
        "generations": args.generations,
        "sigma": args.sigma,
    }
    run_all(args.variant, args.seed, args.tag, overrides, args.workers, args.force)


if __name__ == "__main__":
    main()