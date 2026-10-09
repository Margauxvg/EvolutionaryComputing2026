"""Check the EA and the 1/5 rule on a cheap synthetic objective, without any physics.

Run from the project root (takes a few seconds):

    uv run assignments/assignment_2/group_28/extras/sphere_check.py

The objective is a shifted sphere, f(x) = sum((x - 1)^2), the standard problem the 1/5 rule
was analysed on. simulation.evaluate and evolve.genotype_length are swapped for stand-ins, so
evolve.py, mutation.py and the selection code run exactly as in the real experiment - only the
fitness is different. If a check here fails, the bug is in the EA, not in the robot.

What it checks
--------------
1. Same seed -> identical results (seeding is complete).
2. Static and adaptive start from the same generation-0 population.
3. Static sigma never changes; adaptive sigma does.
4. Adaptive sigma shrinks as the population approaches the optimum, and its success rate
   is pulled towards 1/5.
5. On the sphere, adaptive ends clearly better than static (the textbook result).
6. Both beat random search at the same number of evaluations.
7. The baseline draws exactly POP_SIZE new individuals per generation (equal budget).
8. adaptive_cap1 starts like adaptive and never exceeds its own, lower, sigma bound.
"""

import random
import sys
from pathlib import Path

import numpy as np

from ariel.ec import set_seed

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
import evolve
import simulation
from mutation import make_mutation

N_GENES = 30
POP_SIZE = 20
GENERATIONS = 150


def sphere(genotype, model, data) -> float:
    """Stands in for simulation.evaluate: same signature, no MuJoCo."""
    return float(np.sum((np.asarray(genotype, dtype=np.float64) - 1.0) ** 2))


simulation.evaluate = sphere
evolve.genotype_length = lambda: N_GENES
config.POP_SIZE = POP_SIZE


def run(variant: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    random.seed(seed)
    set_seed(seed)

    population = evolve.evaluate(evolve.init_population(rng))
    mutation = make_mutation(variant, rng)
    gen0 = sorted(ind.fitness_ for ind in population)
    best_so_far = min(gen0)
    sigmas, rates = [], []

    for _ in range(GENERATIONS):
        sigmas.append(getattr(mutation, "sigma", None))
        population = evolve.run_generation(variant, population, mutation, rng)
        rates.append(getattr(mutation, "last_success_rate", None))
        best_so_far = min(best_so_far, evolve.fitness_stats(population)[0])

    return {"gen0": gen0, "best": best_so_far, "sigmas": sigmas, "rates": rates,
            "final_size": len(population)}


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{'  ' + detail if detail else ''}")
    return ok


def main() -> None:
    print(f"Shifted sphere, {N_GENES} genes, pop {POP_SIZE}, {GENERATIONS} generations, "
          f"sigma0 = {config.STATIC_SIGMA}, window = {config.ADAPTIVE_WINDOW}\n")

    seeds = [1, 2, 3]
    results = {v: [run(v, s) for s in seeds] for v in config.VARIANTS}
    again = run("adaptive", seeds[0])

    static_best = [r["best"] for r in results["static"]]
    adaptive_best = [r["best"] for r in results["adaptive"]]
    baseline_best = [r["best"] for r in results["baseline"]]
    adaptive_sigmas = results["adaptive"][0]["sigmas"]
    late_rates = [
        r for res in results["adaptive"] for r in res["rates"][GENERATIONS // 2 :]
        if r is not None
    ]

    print(f"final best (lower is better), seeds {seeds}")
    print(f"  static   : {np.round(static_best, 4)}")
    print(f"  adaptive : {np.round(adaptive_best, 4)}")
    print(f"  baseline : {np.round(baseline_best, 2)}")
    print(f"adaptive sigma, seed 1: start {adaptive_sigmas[0]:.3f}, "
          f"middle {adaptive_sigmas[GENERATIONS // 2]:.4f}, end {adaptive_sigmas[-1]:.5f}\n")

    results_ok = [
        check("same seed -> identical run", again["best"] == results["adaptive"][0]["best"]
              and again["sigmas"] == adaptive_sigmas),
        check("static and adaptive share generation 0",
              all(a["gen0"] == s["gen0"]
                  for a, s in zip(results["adaptive"], results["static"]))),
        check("static sigma constant",
              all(set(r["sigmas"]) == {config.STATIC_SIGMA} for r in results["static"])),
        check("adaptive sigma shrinks", adaptive_sigmas[-1] < adaptive_sigmas[0] / 2,
              f"{adaptive_sigmas[0]:.3f} -> {adaptive_sigmas[-1]:.5f}"),
        check("adaptive success rate pulled towards 1/5",
              0.05 < float(np.mean(late_rates)) < 0.35,
              f"mean over 2nd half {np.mean(late_rates):.3f}"),
        check("adaptive beats static on the sphere",
              max(adaptive_best) < min(static_best)),
        check("both EAs beat random search",
              max(static_best + adaptive_best) < min(baseline_best)),
        check("baseline spends POP_SIZE evaluations per generation",
              all(r["final_size"] == POP_SIZE for r in results["baseline"])),
        check("adaptive_cap1 starts like adaptive and respects its bound",
              all(c["gen0"] == a["gen0"] and c["sigmas"][0] == a["sigmas"][0]
                  and max(c["sigmas"]) <= config.ADAPTIVE_CAP1_MAX_SIGMA
                  for c, a in zip(results["adaptive_cap1"], results["adaptive"]))),
    ]
    print("\nALL CHECKS PASSED" if all(results_ok) else "\nSOME CHECKS FAILED")


if __name__ == "__main__":
    main()