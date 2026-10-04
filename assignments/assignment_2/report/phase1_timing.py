"""Phase 1 feasibility probe for Assignment 2. Run this BEFORE writing any EA.

It answers the four questions that decide your whole experimental design:

  1. How big is the genotype? (= your search dimensionality)
  2. How long does ONE evaluation take?
  3. Can you reuse the compiled model between evaluations, and how much does that save?
  4. Is the evaluation deterministic? (if not, the 1/5 success signal is unreliable)

and then prints a budget table so you can pick pop_size / generations / seeds from
measured numbers instead of guessing.

Run from the project root:
    uv run assignments/assignment_2/group_28/phase1_timing.py
"""

import time

import mujoco as mj
import numpy as np

from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.runners import simple_runner

SPAWN_POS = [0.0, 0.0, 0.1]
TARGET_POSITION = [2.0, 0.0, 0.1]
SIM_DURATION = 15.0
HIDDEN_SIZE = 6

N_TIMED = 10  # evaluations per timing measurement


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #

def build_model() -> tuple[mj.MjModel, mj.MjData]:
    """Compile a fresh world + gecko into a MuJoCo model."""
    mj.set_mjcb_control(None)  # the callback is a GLOBAL - always clear it first
    world = SimpleFlatWorld()
    robot = gecko()
    world.spawn(robot.spec, position=SPAWN_POS, correct_collision_with_floor=True)
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)
    return model, data


# --------------------------------------------------------------------------- #
# Controller (the template's version: bare qpos, no oscillator, no target input)
# --------------------------------------------------------------------------- #

def nn_controller(data: mj.MjData, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    layer1 = np.tanh(data.qpos @ w1)
    outputs = np.tanh(layer1 @ w2)
    return outputs * (np.pi / 2)


def evaluate(model: mj.MjModel, data: mj.MjData, w1: np.ndarray, w2: np.ndarray) -> float:
    """One rollout -> fitness (Euclidean distance to target in the ground plane)."""
    mj.set_mjcb_control(None)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    def control_callback(m: mj.MjModel, d: mj.MjData) -> None:
        d.ctrl[:] = nn_controller(d, w1, w2)

    mj.set_mjcb_control(control_callback)
    simple_runner(model, data, duration=SIM_DURATION)
    mj.set_mjcb_control(None)

    final = np.asarray(data.qpos[0:3]).copy()
    target = np.asarray(TARGET_POSITION)
    return float(np.linalg.norm(final[:2] - target[:2]))


def random_weights(rng: np.random.Generator, n_in: int, n_out: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        rng.normal(scale=0.5, size=(n_in, HIDDEN_SIZE)),
        rng.normal(scale=0.5, size=(HIDDEN_SIZE, n_out)),
    )


# --------------------------------------------------------------------------- #
# The four probes
# --------------------------------------------------------------------------- #

def report_sizes() -> tuple[int, int, int]:
    model, data = build_model()
    n_in, n_out = len(data.qpos), model.nu
    genotype_length = n_in * HIDDEN_SIZE + HIDDEN_SIZE * n_out

    print("=== PROBLEM SIZE ===")
    print(f"  controller inputs  (len(data.qpos)) : {n_in}")
    print(f"  controller outputs (model.nu)       : {n_out}")
    print(f"  genotype length (flat weights)      : {genotype_length}")
    print(f"  qvel length (if you add it)         : {len(data.qvel)}")
    print()
    return n_in, n_out, genotype_length


def time_rebuild_each_time(n: int, n_in: int, n_out: int) -> float:
    """Worst case: recompile the world for every evaluation (what the template does)."""
    rng = np.random.default_rng(0)
    started = time.perf_counter()
    for _ in range(n):
        model, data = build_model()
        evaluate(model, data, *random_weights(rng, n_in, n_out))
    return (time.perf_counter() - started) / n


def time_reuse_model(n: int, n_in: int, n_out: int) -> float:
    """Compile once, reset between evaluations. This is what your EA should do."""
    model, data = build_model()
    rng = np.random.default_rng(0)
    started = time.perf_counter()
    for _ in range(n):
        evaluate(model, data, *random_weights(rng, n_in, n_out))
    return (time.perf_counter() - started) / n


def check_determinism(n_in: int, n_out: int) -> None:
    """Evaluate the SAME weights twice. If the fitnesses differ, your evaluation is
    noisy, and an offspring can 'beat its parent' by luck alone -- which corrupts the
    1/5 success signal. Decide what to do about it before you build the EA."""
    model, data = build_model()
    rng = np.random.default_rng(123)
    w1, w2 = random_weights(rng, n_in, n_out)

    a = evaluate(model, data, w1, w2)
    b = evaluate(model, data, w1, w2)

    # Also check it across a fresh model, in case state leaks through the model object.
    model2, data2 = build_model()
    c = evaluate(model2, data2, w1, w2)

    print("=== DETERMINISM ===")
    print(f"  same weights, same model, twice : {a:.9f}  vs  {b:.9f}   (diff {abs(a - b):.2e})")
    print(f"  same weights, fresh model       : {c:.9f}            (diff {abs(a - c):.2e})")
    if abs(a - b) < 1e-9 and abs(a - c) < 1e-9:
        print("  -> deterministic. The 1/5 success signal is trustworthy.")
    else:
        print("  -> NOT deterministic. Either average over k rollouts, or say so in Methods.")
    print()


def budget_table(seconds_per_eval: float, n_cores: int) -> None:
    print("=== BUDGET (wall-clock hours for the FULL experiment) ===")
    print(f"  assuming {seconds_per_eval:.3f} s/eval and {n_cores} parallel processes")
    print(f"  3 configurations (static, adaptive, baseline)\n")
    print(f"  {'pop':>5} {'gens':>6} {'seeds':>6} {'evals/run':>10} {'total evals':>12} {'hours':>8}")
    for pop in (20, 30, 50):
        for gens in (50, 100, 200):
            for seeds in (5, 10):
                evals_per_run = pop * (gens + 1)
                total = evals_per_run * 3 * seeds
                hours = total * seconds_per_eval / 3600 / n_cores
                flag = "  <-- feasible" if hours < 12 else ""
                print(f"  {pop:>5} {gens:>6} {seeds:>6} {evals_per_run:>10} {total:>12} {hours:>8.1f}{flag}")
    print()


def main() -> None:
    import os

    n_in, n_out, _ = report_sizes()

    print("=== TIMING ===")
    per_eval_rebuild = time_rebuild_each_time(N_TIMED, n_in, n_out)
    per_eval_reuse = time_reuse_model(N_TIMED, n_in, n_out)
    print(f"  rebuild world every eval : {per_eval_rebuild:.3f} s/eval")
    print(f"  reuse compiled model     : {per_eval_reuse:.3f} s/eval")
    print(f"  speedup from reuse       : {per_eval_rebuild / per_eval_reuse:.2f}x")
    print(f"  ({SIM_DURATION}s simulated per eval -> "
          f"{SIM_DURATION / per_eval_reuse:.1f}x faster than real time)")
    print()

    check_determinism(n_in, n_out)

    n_cores = max(1, (os.cpu_count() or 4) - 1)
    budget_table(per_eval_reuse, n_cores)


if __name__ == "__main__":
    main()
