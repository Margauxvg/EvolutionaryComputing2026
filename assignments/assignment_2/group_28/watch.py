"""Watch the robot in the MuJoCo viewer, using simulate.py + controller.py + config.py.

Uses the SAME body, controller and control application (DELTA, clipping, NaN
guard) as the EA, because it goes through simulate.Simulator. What you see here
is exactly what gets scored.

Run from the project root:

    # random weights: rhythmic wiggling, no direction yet
    uv run assignments/assignment_2/group_28/watch.py

    # quick throwaway search first (NOT the real EA), then watch the best
    uv run assignments/assignment_2/group_28/watch.py --quick-evolve 20

    # watch a saved genotype (e.g. the best one from a real EA run)
    uv run assignments/assignment_2/group_28/watch.py --load best.npy

--quick-evolve also saves the best genotype to quick_best.npy, so you can
reload it with --load without searching again. Close the window to exit.
The target is 2 m along the +x axis (red axis in the viewer) from the spawn.
"""

# Standard library
import argparse
import time
from pathlib import Path

# Third-party libraries
import mujoco as mj
import numpy as np
from mujoco import viewer

import config
import controller
from simulate import Simulator


def quick_evolve(
    sim: Simulator,
    n: int,
    generations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """A tiny (5 + 10) mutation-only search, just to get something that moves.

    Throwaway demo code: fixed sigma, no logging, one seed. The real EA and
    its 1/5-rule variant live in their own files.
    """
    mu, lam, sigma = 5, 10, 0.3
    parents = [rng.normal(scale=0.5, size=n) for _ in range(mu)]
    fitness = [sim.evaluate(g) for g in parents]
    started = time.perf_counter()

    for gen in range(1, generations + 1):
        children = [
            parents[rng.integers(mu)] + rng.normal(scale=sigma, size=n)
            for _ in range(lam)
        ]
        child_fitness = [sim.evaluate(g) for g in children]
        pool = parents + children
        pool_fitness = fitness + child_fitness
        best = np.argsort(pool_fitness)[:mu]
        parents = [pool[i] for i in best]
        fitness = [pool_fitness[i] for i in best]
        print(
            f"  gen {gen:>3}/{generations}  best distance {fitness[0]:.3f} m"
            f"  ({time.perf_counter() - started:.0f} s)",
        )

    return parents[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch the robot in the viewer.")
    parser.add_argument("--load", type=Path, help="genotype .npy file to watch")
    parser.add_argument(
        "--quick-evolve",
        type=int,
        metavar="GENS",
        help="run a tiny throwaway search for GENS generations first",
    )
    parser.add_argument("--seed", type=int, default=getattr(config, "DEFAULT_SEED", 0))
    args = parser.parse_args()

    sim = Simulator(controller.act)
    n = controller.genotype_length(sim.model)
    rng = np.random.default_rng(args.seed)
    print(f"hinges {sim.model.nu} | inputs {controller.input_size(sim.model)} | "
          f"genotype length {n}")

    if args.load is not None:
        genotype = np.load(args.load)
        source = f"loaded from {args.load}"
    elif args.quick_evolve:
        print(f"Quick search: {args.quick_evolve} generations x 10 rollouts ...")
        genotype = quick_evolve(sim, n, args.quick_evolve, rng)
        out = Path(__file__).with_name("quick_best.npy")
        np.save(out, genotype)
        source = f"quick search (saved to {out.name})"
    else:
        genotype = rng.normal(scale=0.5, size=n)
        source = f"random weights (seed {args.seed})"

    if genotype.shape != (n,):
        msg = f"Genotype has {genotype.size} values, this body needs {n}."
        raise SystemExit(msg)

    fitness = sim.evaluate(genotype)
    start_distance = np.linalg.norm(np.asarray(config.TARGET_POSITION[:2]))
    print(f"Genotype: {source}")
    print(f"Scored distance after {sim.duration:.0f} s: {fitness:.3f} m "
          f"(started {start_distance:.1f} m away)")

    # Reset and attach the Simulator's own callback, so the viewer applies
    # actions exactly as the EA's rollouts do.
    mj.set_mjcb_control(None)
    mj.mj_resetData(sim.model, sim.data)
    mj.mj_forward(sim.model, sim.data)
    mj.set_mjcb_control(sim._make_callback(genotype))  # noqa: SLF001

    print("Opening viewer. It runs past 15 s until you close the window.")
    try:
        viewer.launch(model=sim.model, data=sim.data)
    finally:
        mj.set_mjcb_control(None)


if __name__ == "__main__":
    main()