# Quick checks of the controller and the 1/5 rule (these used to be at the bottom of
# controller.py and mutation.py). Run from the project root:
#   uv run assignments/assignment_2/group_28/extras/self_tests.py

import sys
import time
from pathlib import Path

import numpy as np

from ariel.ec import Individual, Population

sys.path.append(str(Path(__file__).resolve().parent.parent))

import brain
import config
import simulation
from mutation import AdaptiveMutation, StaticMutation


def check_controller() -> None:
    """Network sizes, output range, time per evaluation, and determinism."""
    model, data = simulation.build_world()
    n = brain.num_weights(model)
    print(f"hinges (outputs) : {model.nu}")
    print(f"inputs           : {brain.num_inputs(model)}")
    print(f"genotype length  : {n}")

    genotype = np.random.default_rng(config.DEFAULT_SEED).normal(scale=0.5, size=n)
    targets = brain.hinge_targets(data, *brain.split_weights(genotype, model))
    print(f"outputs in hinge range: {bool(np.all(np.abs(targets) <= np.pi / 2))}")

    started = time.perf_counter()
    first = simulation.evaluate(genotype, model, data)
    print(f"fitness {first:.6f}, {time.perf_counter() - started:.3f} s per evaluation")
    repeats = [simulation.evaluate(genotype, model, data) for _ in range(2)]
    print(f"same fitness every time: {all(r == first for r in repeats)}\n")


def fake_offspring(n: int, parent_fitness: float, rng: np.random.Generator) -> Population:
    offspring = []
    for _ in range(n):
        individual = Individual()
        individual.genotype = rng.normal(size=8).tolist()
        individual.tags = {"parent_fitness": parent_fitness}
        offspring.append(individual)
    return Population(offspring)


def check_one_fifth_rule() -> None:
    """Fake fitnesses, no simulator: adaptive sigma should grow while more than 1/5 of the
    mutations succeed and shrink below that, static sigma should never move."""
    rng = np.random.default_rng(config.DEFAULT_SEED)

    for label, success_fraction in (("60% of mutations succeed", 0.6), ("0% of mutations succeed", 0.0)):
        adaptive = AdaptiveMutation(np.random.default_rng(1), max_sigma=config.ADAPTIVE_MAX_SIGMA)
        static = StaticMutation(np.random.default_rng(1))
        print(label)
        print(f"{'gen':>5} {'adaptive sigma':>16} {'static sigma':>14} {'success rate':>14}")

        for generation in range(1, 9):
            for variant in (adaptive, static):
                offspring = variant.mutate(fake_offspring(20, parent_fitness=1.0, rng=rng))
                # Instead of evaluate(): a fixed fraction of the children beats the parent
                for i, individual in enumerate(offspring):
                    individual.fitness = 0.9 if i < 20 * success_fraction else 1.1
                variant.adapt(offspring)

            rate = adaptive.last_success_rate
            rate_text = "-" if rate is None else f"{rate:.2f}"
            print(f"{generation:>5} {adaptive.sigma:>16.4f} {static.sigma:>14.4f} {rate_text:>14}")
        print()


if __name__ == "__main__":
    check_controller()
    check_one_fifth_rule()
