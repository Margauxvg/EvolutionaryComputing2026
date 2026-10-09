# Watch the robot in the MuJoCo viewer. Run from the project root:
#   uv run assignments/assignment_2/group_28/extras/watch.py      (random weights, wiggles without direction)
#   uv run assignments/assignment_2/group_28/extras/watch.py --load assignments/assignment_2/group_28/results/main/gecko/adaptive/seed_101/best.npy
#   uv run assignments/assignment_2/group_28/extras/watch.py --quick-evolve 20     (a tiny throwaway search first, NOT the real EA)
# --quick-evolve saves its best genotype to extras/quick_best.npy, so you can --load it later.

import argparse
import sys
import time
from pathlib import Path

import mujoco as mj
import numpy as np
from mujoco import viewer

sys.path.append(str(Path(__file__).resolve().parent.parent))

import brain
import config
import simulation


def quick_evolve(model, data, n: int, generations: int, rng: np.random.Generator) -> np.ndarray:
    """A tiny (5 + 10) search with fixed sigma, only to get something that moves."""
    mu, lam, sigma = 5, 10, 0.3
    parents = [rng.normal(scale=0.5, size=n) for _ in range(mu)]
    fitness = [simulation.evaluate(g, model, data) for g in parents]
    started = time.perf_counter()

    for generation in range(1, generations + 1):
        children = [parents[rng.integers(mu)] + rng.normal(scale=sigma, size=n) for _ in range(lam)]
        pool = parents + children
        pool_fitness = fitness + [simulation.evaluate(g, model, data) for g in children]
        best = np.argsort(pool_fitness)[:mu]
        parents = [pool[i] for i in best]
        fitness = [pool_fitness[i] for i in best]
        print(f"  gen {generation:>3}/{generations}  best distance {fitness[0]:.3f} m "
              f"({time.perf_counter() - started:.0f} s)")

    return parents[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch the robot in the viewer.")
    parser.add_argument("--load", type=Path, help="genotype .npy file to watch")
    parser.add_argument("--quick-evolve", type=int, metavar="GENS", help="run a tiny search for GENS generations first")
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    args = parser.parse_args()

    model, data = simulation.build_world()
    n = brain.num_weights(model)
    rng = np.random.default_rng(args.seed)

    if args.load is not None:
        genotype = np.load(args.load)
        source = f"loaded from {args.load}"
    elif args.quick_evolve:
        genotype = quick_evolve(model, data, n, args.quick_evolve, rng)
        output = Path(__file__).with_name("quick_best.npy")
        np.save(output, genotype)
        source = f"quick search (saved to {output})"
    else:
        genotype = rng.normal(scale=0.5, size=n)
        source = f"random weights (seed {args.seed})"

    if genotype.shape != (n,):
        raise SystemExit(f"Genotype has {genotype.size} values, this body needs {n}.")

    fitness = simulation.evaluate(genotype, model, data)
    print(f"Genotype: {source}")
    print(f"Distance to target after {config.SIM_DURATION:.0f} s: {fitness:.3f} m")

    # Same controller as in the EA. The viewer keeps going after 15 s until you close it.
    control, _ = simulation.make_controller(genotype, model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)
    mj.set_mjcb_control(control)
    try:
        viewer.launch(model=model, data=data)
    finally:
        mj.set_mjcb_control(None)


if __name__ == "__main__":
    main()
