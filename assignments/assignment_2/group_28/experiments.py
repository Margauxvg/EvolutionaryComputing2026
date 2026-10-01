"""Small probes that answer the open questions from Stage 1. Not the experiment.

Each subcommand answers one question that is currently an unjustified number in Methods,
or an assumption nobody has tested. They are cheap, they are throwaway, and their results
belong in report/stage1_findings.md as evidence rather than in the final EA.

Run from the project root:

    uv run assignments/assignment_2/group_28/experiments.py duration
    uv run assignments/assignment_2/group_28/experiments.py clock
    uv run assignments/assignment_2/group_28/experiments.py qvel
    uv run assignments/assignment_2/group_28/experiments.py nan
    uv run assignments/assignment_2/group_28/experiments.py cache
    uv run assignments/assignment_2/group_28/experiments.py determ
    uv run assignments/assignment_2/group_28/experiments.py hidden
    uv run assignments/assignment_2/group_28/experiments.py sigma

WHAT EACH ONE IS FOR
--------------------
    duration  Can the gecko reach the target at all, or is 15 s simply too short a window?
              Decides SIM_DURATION or the target distance.
    clock     Is CLOCK_FREQ = 1.0 Hz a good gait frequency, or just the first number we tried?
              Turns an unexplained constant into a measured choice.
    qvel      Does giving the network joint velocities help? Three lines of code, and a
              Methods sentence either way.
    nan       Does the NaN guard in simulate.py actually fire, and does WORST_FITNESS behave?
    cache     What does get_simulator's per-process cache actually buy? Reproduces the
              1.25x model-reuse result through the CURRENT code path.
    determ    Is evaluation bit-identical when the MODEL is rebuilt, and across PROCESSES?
              The second half is load-bearing: run.py evaluates in spawned workers, so if a
              worker disagrees with the parent the 20 seeds are not comparable runs.
    hidden    Prints genotype length against HIDDEN_SIZE. No simulation - this is the
              parameter-count argument for Methods, not a sweep.
    sigma     Which fixed sigma should the static variant use, and does the 1/5 rule end up
              in the same place whatever sigma it starts from? Runs the REAL EA (ea.py and
              mutation.py), not the throwaway ES the other probes use.

Results print as a table and are appended to results/<name>.csv so they can be cited.
"""

# Standard library
import argparse
import csv
import multiprocessing as mp
import os
import statistics
import time
from pathlib import Path

# Third-party libraries
import mujoco as mj
import numpy as np
import numpy.typing as npt

import config
import controller
import ea
from mutation import AdaptiveMutation, StaticMutation
from simulate import Simulator

HERE = Path(__file__).resolve().parent
RESULTS_DIR = getattr(config, "RESULTS_DIR", HERE / "results")


def write_csv(name: str, rows: list[dict]) -> Path:
    """Same shape as A1's helpers.write_csv, kept local so this file stands alone."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n-> {path}")
    return path


def search(
    sim: Simulator,
    n_weights: int,
    generations: int,
    rng: np.random.Generator,
    sigma: float = 0.3,
    mu: int = 5,
    lam: int = 10,
) -> list[float]:
    """The same (5 + 10) throwaway ES as watch.quick_evolve, but quiet and returning history.

    Fixed sigma, mutation only, no logging. Duplicated rather than imported because
    watch.quick_evolve prints per generation, which is unreadable across four sweeps.
    """
    parents = [rng.normal(scale=0.5, size=n_weights) for _ in range(mu)]
    fitness = [sim.evaluate(g) for g in parents]
    history = [min(fitness)]

    for _ in range(generations):
        children = [
            parents[rng.integers(mu)] + rng.normal(scale=sigma, size=n_weights)
            for _ in range(lam)
        ]
        pool = parents + children
        pool_fitness = fitness + [sim.evaluate(g) for g in children]
        best = np.argsort(pool_fitness)[:mu]
        parents = [pool[i] for i in best]
        fitness = [pool_fitness[i] for i in best]
        history.append(fitness[0])

    return history


# --------------------------------------------------------------------------- #
#  duration - is 15 s long enough to reach the target?
# --------------------------------------------------------------------------- #
def experiment_duration(args: argparse.Namespace) -> None:
    """Score one genotype at increasing simulation durations.

    Because evaluation is deterministic, the 15 s rollout is a strict prefix of the 60 s
    one, so scoring the same genotype at 5, 10, ... 60 s traces its whole trajectory. Each
    row is literally "what fitness would be if SIM_DURATION were this", which is the
    decision we are trying to make.

    Read the speed column, not the distance column. Roughly constant speed means the gait is
    sustained. Speed decaying towards zero long before the target means the controller lurches
    once and stalls, which is a controller problem that no amount of simulation time fixes.

    Reaching the target inside the window is NOT the goal. If good controllers already reach it
    within SIM_DURATION, fitness piles up near 0 and stops separating good from better (a floor
    effect). A window the best controller cannot quite finish keeps the comparison informative.
    """
    path = Path(args.genotype)
    if not path.is_absolute():
        path = HERE / path
    if not path.exists():
        raise SystemExit(
            f"No genotype at {path}. Produce one first:\n"
            "  uv run assignments/assignment_2/group_28/watch.py --quick-evolve 20"
        )

    genotype = np.load(path)
    durations = list(range(args.step, args.max + 1, args.step))

    print(f"genotype : {path.name} ({genotype.size} weights)")
    print(f"target   : {config.TARGET_POSITION[:2]}, {np.linalg.norm(config.TARGET_POSITION[:2]):.2f} m from spawn\n")
    print(f"{'sim s':>7} {'distance m':>12} {'travelled m':>12} {'cm/s overall':>14} {'cm/s in slice':>15}")

    start_distance = float(np.linalg.norm(np.asarray(config.TARGET_POSITION[:2])))
    rows: list[dict] = []
    previous_distance, previous_duration = start_distance, 0.0

    for duration in durations:
        sim = Simulator(controller.act, duration=float(duration))
        distance = sim.evaluate(genotype)

        travelled = start_distance - distance
        overall = 100.0 * travelled / duration
        slice_speed = 100.0 * (previous_distance - distance) / (duration - previous_duration)

        print(f"{duration:>7} {distance:>12.3f} {travelled:>12.3f} {overall:>14.2f} {slice_speed:>15.2f}")
        rows.append({
            "sim_duration_s": duration,
            "distance_m": round(distance, 4),
            "travelled_m": round(travelled, 4),
            "speed_overall_cm_s": round(overall, 3),
            "speed_in_slice_cm_s": round(slice_speed, 3),
        })
        previous_distance, previous_duration = distance, float(duration)

    needed = 100.0 * start_distance / config.SIM_DURATION

    # Speed WHILE WALKING: up to the closest approach, not over the whole replay. An earlier
    # version averaged over all 60 s, which counted the time a controller spends standing at
    # the target after arriving - and reported a robot walking at ~7.5 cm/s as "3.2 cm/s".
    closest = min(rows, key=lambda r: r["distance_m"])
    walking = 100.0 * (start_distance - closest["distance_m"]) / closest["sim_duration_s"]
    reached = [r for r in rows if r["distance_m"] <= args.reach]
    in_window = [r for r in rows if r["sim_duration_s"] <= config.SIM_DURATION]
    at_window = in_window[-1] if in_window else None

    print(f"\nclosest approach : {closest['distance_m']:.3f} m at {closest['sim_duration_s']} s")
    print(f"walking speed    : {walking:.1f} cm/s (up to the closest approach)")
    print(f"needed to reach the target within SIM_DURATION={config.SIM_DURATION:.0f}s: {needed:.1f} cm/s")
    if reached:
        print(f"reaches the target (within {args.reach} m) at {reached[0]['sim_duration_s']} s")

    if at_window is None:
        print("-> no duration in this replay is inside SIM_DURATION; use a smaller --step.")
    elif at_window["distance_m"] <= args.reach:
        print(f"-> this controller already reaches the target within {at_window['sim_duration_s']} s. "
              "Fitness can saturate at ~0,\n   which hides differences between good controllers "
              "(a floor effect). Consider a farther target.")
    elif at_window["distance_m"] < args.floor:
        print(f"-> {at_window['distance_m']:.2f} m from the target at {at_window['sim_duration_s']} s: "
              "close enough that better controllers may\n   saturate at ~0 (floor-effect risk). "
              "Re-check with the best controller of a longer run.")
    elif not reached and closest["sim_duration_s"] < durations[-1]:
        print(f"-> stalls {closest['distance_m']:.2f} m short after {closest['sim_duration_s']} s: "
              "a controller problem that more simulation\n   time does not fix.")
    else:
        print(f"-> still {at_window['distance_m']:.2f} m away at {at_window['sim_duration_s']} s, "
              "so fitness cannot saturate at 0 and keeps\n   separating good controllers from "
              "better ones. The current window is fine.")
    write_csv("duration", rows)


# --------------------------------------------------------------------------- #
#  clock - is 1.0 Hz the right gait frequency?
# --------------------------------------------------------------------------- #
def experiment_clock(args: argparse.Namespace) -> None:
    """Run the same short search at several CLOCK_FREQ values.

    CLOCK_FREQ sets how fast the sin/cos inputs oscillate, i.e. roughly how many gait
    cycles per second the network is offered. 1.0 Hz is currently a guess; controller.py
    even says "choose in pilots".

    Every sweep uses the SAME seed, so the initial population and every mutation are
    identical and the only difference is the frequency. controller.CLOCK_FREQ is patched
    at runtime rather than edited, so config.py is left alone.

    Caveat to write down: this is one seed and a 20-generation toy search. It is enough to
    reject an obviously bad frequency, not enough to claim one is optimal.
    """
    original = controller.CLOCK_FREQ
    rows: list[dict] = []
    per_freq: dict[float, list[float]] = {}

    total = len(args.freqs) * len(args.seeds)
    print(f"{args.generations} generations, {len(args.seeds)} seeds, "
          f"{len(args.freqs)} frequencies = {total} runs\n")
    print(f"{'Hz':>6} {'seed':>6} {'final best m':>14} {'improvement m':>15} {'improving gens':>16}")

    try:
        for freq in args.freqs:
            per_freq[freq] = []
            controller.CLOCK_FREQ = float(freq)
            # Rebuilt per frequency but not per seed: the body does not depend on the clock,
            # and compiling once per frequency keeps the sweep honest and cheap.
            sim = Simulator(controller.act)
            n = controller.genotype_length(sim.model)

            for seed in args.seeds:
                rng = np.random.default_rng(seed)
                history = search(sim, n, args.generations, rng)
                improved = sum(1 for a, b in zip(history, history[1:]) if b < a)
                per_freq[freq].append(history[-1])

                print(f"{freq:>6.2f} {seed:>6} {history[-1]:>14.3f} "
                      f"{history[0] - history[-1]:>15.3f} {improved:>16}")
                rows.append({
                    "clock_freq_hz": freq,
                    "seed": seed,
                    "generations": args.generations,
                    "initial_best_m": round(history[0], 4),
                    "final_best_m": round(history[-1], 4),
                    "improvement_m": round(history[0] - history[-1], 4),
                    "improving_generations": improved,
                })
    finally:
        controller.CLOCK_FREQ = original

    # Across seeds is the only comparison worth reading. A single seed cannot separate a real
    # difference from where the initial population happened to land.
    print(f"\n{'Hz':>6} {'mean final m':>14} {'std':>8} {'best seed':>11} {'worst seed':>12}")
    for freq, finals in per_freq.items():
        spread = statistics.pstdev(finals) if len(finals) > 1 else 0.0
        print(f"{freq:>6.2f} {statistics.fmean(finals):>14.3f} {spread:>8.3f} "
              f"{min(finals):>11.3f} {max(finals):>12.3f}")

    ranked = sorted(per_freq.items(), key=lambda kv: statistics.fmean(kv[1]))
    best_freq, best_finals = ranked[0]
    print(f"\nlowest mean: {best_freq} Hz at {statistics.fmean(best_finals):.3f} m")
    if len(ranked) > 1:
        runner_up, runner_finals = ranked[1]
        gap = statistics.fmean(runner_finals) - statistics.fmean(best_finals)
        pooled = statistics.fmean([
            statistics.pstdev(v) for v in per_freq.values() if len(v) > 1
        ]) if len(args.seeds) > 1 else 0.0
        print(f"gap to {runner_up} Hz: {gap:.3f} m, against a typical within-frequency "
              f"spread of {pooled:.3f} m")
        if pooled and gap < pooled:
            print("-> the gap is inside the noise. Do not pick a winner from this.")
    write_csv("clock_freq", rows)


# --------------------------------------------------------------------------- #
#  qvel - does the network benefit from joint velocities?
# --------------------------------------------------------------------------- #
def _act_with_qvel(
    model: mj.MjModel,
    data: mj.MjData,
    genotype: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """controller.act, with data.qvel appended to the inputs.

    Lives here rather than in controller.py because it is a probe, not a decision. It works
    because simulate.Simulator takes the controller as a parameter - this is exactly what
    that injection was for.
    """
    n_in = controller.input_size(model) + len(data.qvel)
    split = n_in * controller.HIDDEN_SIZE
    w1 = genotype[:split].reshape(n_in, controller.HIDDEN_SIZE)
    w2 = genotype[split:].reshape(controller.HIDDEN_SIZE, model.nu)
    inputs = np.concatenate((controller.build_inputs(data), data.qvel))
    return controller.forward(inputs, w1, w2) * controller.HINGE_LIMIT


def experiment_qvel(args: argparse.Namespace) -> None:
    """Same search, with and without joint velocities in the input vector.

    The template mentions qvel as an option and we are not using it. Proprioceptive velocity
    is the usual reason a gait controller can tell "leg moving forward" from "leg moving
    back" at the same angle, so there is a real mechanism to expect a difference.

    Important confound, and it must go in the write-up: adding qvel enlarges the genotype,
    so the qvel variant searches a higher-dimensional space on the same budget. A worse
    result does not cleanly mean the inputs are useless - it may mean the search got harder.
    """
    baseline_sim = Simulator(controller.act)
    n_baseline = controller.genotype_length(baseline_sim.model)

    qvel_sim = Simulator(_act_with_qvel)
    n_qvel = (
        (controller.input_size(qvel_sim.model) + len(qvel_sim.data.qvel))
        * controller.HIDDEN_SIZE
        + controller.HIDDEN_SIZE * qvel_sim.model.nu
    )

    print(f"{args.generations} generations, {len(args.seeds)} seeds\n")
    print(f"{'inputs':>22} {'seed':>6} {'genotype':>10} {'initial m':>11} {'final m':>9}")

    rows: list[dict] = []
    finals: dict[str, list[float]] = {}
    initials: dict[str, list[float]] = {}
    for label, sim, n in (
        ("hinges + extras", baseline_sim, n_baseline),
        ("hinges + extras + qvel", qvel_sim, n_qvel),
    ):
        finals[label], initials[label] = [], []
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            history = search(sim, n, args.generations, rng)
            finals[label].append(history[-1])
            initials[label].append(history[0])
            print(f"{label:>22} {seed:>6} {n:>10} {history[0]:>11.3f} {history[-1]:>9.3f}")
            rows.append({
                "inputs": label,
                "seed": seed,
                "genotype_length": n,
                "generations": args.generations,
                "initial_best_m": round(history[0], 4),
                "final_best_m": round(history[-1], 4),
                "improvement_m": round(history[0] - history[-1], 4),
            })

    # Read the initial column as carefully as the final one. On one seed the qvel variant
    # finished ahead but had also STARTED ahead, so the gain may be in initialisation - random
    # controllers with velocity feedback twitch more - rather than in search.
    print(f"\n{'inputs':>22} {'mean initial':>14} {'mean final':>12} {'mean gain':>11}")
    for label in finals:
        mi, mf = statistics.fmean(initials[label]), statistics.fmean(finals[label])
        print(f"{label:>22} {mi:>14.3f} {mf:>12.3f} {mi - mf:>11.3f}")

    print(f"\nthe qvel variant searches {n_qvel - n_baseline} more dimensions on the same "
          "budget - say so when reporting this")
    write_csv("qvel", rows)


# --------------------------------------------------------------------------- #
#  nan - does the guard fire, and does WORST_FITNESS behave?
# --------------------------------------------------------------------------- #
def experiment_nan(args: argparse.Namespace) -> None:
    """Check the NaN path in simulate.Simulator._make_callback.

    Two things worth knowing, and the second is the interesting one.

    First: a NaN genotype must return WORST_FITNESS rather than a plausible number or None.
    None would crash the min(contestants, key=...) in tournament selection, days later and
    far from the cause.

    Second: finite-but-enormous weights do NOT produce NaN, because tanh saturates. So the
    guard does not fire on "the EA drifted to big weights" - it fires on genuine overflow to
    inf, or nan already in the genotype. Useful to know before writing a Methods sentence
    claiming the guard protects against runaway mutation.
    """
    sim = Simulator(controller.act)
    n = controller.genotype_length(sim.model)
    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []

    cases = {
        "normal weights": rng.normal(scale=0.5, size=n),
        "huge finite weights (1e6)": np.full(n, 1e6),
        "one NaN in the genotype": np.concatenate(([np.nan], rng.normal(scale=0.5, size=n - 1))),
        "one inf in the genotype": np.concatenate(([np.inf], rng.normal(scale=0.5, size=n - 1))),
    }

    print(f"WORST_FITNESS = {config.WORST_FITNESS}\n")
    print(f"{'genotype':>28} {'fitness':>14} {'guard fired':>13}")
    for label, genotype in cases.items():
        fitness = sim.evaluate(genotype)
        fired = fitness == config.WORST_FITNESS
        print(f"{label:>28} {fitness:>14.4f} {str(fired):>13}")
        rows.append({
            "case": label,
            "fitness": fitness,
            "guard_fired": fired,
        })

    # Swapping the sentinel: nothing downstream should care what the number is, only that
    # it is worse than any real distance. A real distance cannot exceed a few metres.
    original = config.WORST_FITNESS
    try:
        config.WORST_FITNESS = 100.0
        swapped = sim.evaluate(cases["one NaN in the genotype"])
        print(f"\nwith WORST_FITNESS = 100.0, a NaN genotype scores {swapped}")
        print("min() over a mixed list still picks a real individual: "
              f"{min([1.62, swapped, 1.95])}")
    finally:
        config.WORST_FITNESS = original

    write_csv("nan_guard", rows)


# --------------------------------------------------------------------------- #
#  cache - what does compiling once actually buy?
# --------------------------------------------------------------------------- #
def experiment_cache(args: argparse.Namespace) -> None:
    """Time N evaluations reusing one Simulator versus building a new one each time.

    This is the 1.25x model-reuse figure from phase1_timing.py, re-measured through the
    CURRENT controller and control-writing path - so it also gives a second, comparable
    seconds-per-evaluation number on this machine.

    It is also the answer to "what is get_simulator's cache for". Delete the cache and every
    evaluation pays the rebuild column.
    """
    sim = Simulator(controller.act)
    n = controller.genotype_length(sim.model)
    rng = np.random.default_rng(args.seed)
    genotypes = [rng.normal(scale=0.5, size=n) for _ in range(args.n)]

    started = time.perf_counter()
    for genotype in genotypes:
        sim.evaluate(genotype)
    reuse = (time.perf_counter() - started) / args.n

    started = time.perf_counter()
    for genotype in genotypes:
        Simulator(controller.act).evaluate(genotype)
    rebuild = (time.perf_counter() - started) / args.n

    print(f"{args.n} evaluations each\n")
    print(f"  reuse one Simulator   : {reuse:.3f} s/eval")
    print(f"  rebuild every time    : {rebuild:.3f} s/eval")
    print(f"  saving from reuse     : {rebuild / reuse:.2f}x")
    print(f"\n  25 Sep, old controller on the 15-core machine: 0.168 / 0.210 s, 1.25x")

    write_csv("cache", [{
        "evaluations": args.n,
        "reuse_s_per_eval": round(reuse, 4),
        "rebuild_s_per_eval": round(rebuild, 4),
        "speedup": round(rebuild / reuse, 3),
    }])


def _child_evaluate(genotype_list: list[float]) -> float:
    """Evaluate one genotype in a spawned worker. Module level, so it can be pickled.

    Each worker compiles its own MjModel - MuJoCo models do not pickle - which is exactly the
    condition being tested. This is the same code path run.py will use.
    """
    sim = Simulator(controller.act)
    return sim.evaluate(np.asarray(genotype_list, dtype=np.float64))


def experiment_determinism(args: argparse.Namespace) -> None:
    """Is one genotype's fitness bit-identical when the model, and then the process, changes?

    Three levels, each strictly harder than the last:

        1. same Simulator, repeated        already known to hold (findings section 2)
        2. fresh Simulator, same process   does rebuilding the model change anything?
        3. fresh PROCESS, spawned          the one that matters

    Level 3 is not a formality. run.py will evaluate in a process pool because
    mj.set_mjcb_control is a global and cannot be shared between threads. Every worker compiles
    its own model. If a worker's fitness differs from the parent's in the last bits, then two
    seeds that ran on different workers are not strictly comparable, and the paired statistics
    in the report rest on sand. Windows only has spawn, so spawn is what is tested.

    Reported as an exact equality, not a tolerance. A difference of 1e-16 is still a difference
    and Methods should not claim bit-identical if it is only nearly so.
    """
    sim = Simulator(controller.act)
    n = controller.genotype_length(sim.model)
    rng = np.random.default_rng(args.seed)
    genotype = rng.normal(scale=config.INIT_WEIGHT_SCALE, size=n)

    same = [sim.evaluate(genotype) for _ in range(args.repeats)]
    fresh = [Simulator(controller.act).evaluate(genotype) for _ in range(args.repeats)]

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=min(args.repeats, 4)) as pool:
        spawned = pool.map(_child_evaluate, [genotype.tolist()] * args.repeats)

    reference = same[0]
    rows = []
    print(f"genotype length {n}, seed {args.seed}, {args.repeats} repeats per level")
    print(f"reference fitness {reference!r}\n")
    print(f"{'level':<28} {'identical':>10} {'max abs diff':>16}")

    for label, values in (
        ("1 same Simulator", same),
        ("2 fresh Simulator", fresh),
        ("3 spawned process", spawned),
    ):
        diff = max(abs(v - reference) for v in values)
        ok = all(v == reference for v in values)
        print(f"{label:<28} {('yes' if ok else 'NO'):>10} {diff:>16.3e}")
        rows.append({
            "level": label,
            "identical": ok,
            "max_abs_diff": f"{diff:.3e}",
            "reference_fitness": repr(reference),
        })

    if all(r["identical"] for r in rows):
        print("\nAll three identical. Methods may say evaluation is deterministic and")
        print("bit-identical across model rebuilds and process boundaries.")
    else:
        print("\nNOT identical at every level. Do not claim bit-identical determinism.")
        print("Report the largest difference above and say runs are reproducible to that")
        print("tolerance instead - and check it is small against the 0.107 m seed-to-seed sd.")

    write_csv("determinism", rows)


def experiment_hidden(args: argparse.Namespace) -> None:
    """Genotype length against HIDDEN_SIZE. No simulation, no sweep.

    Section 6 showed a six-way sweep at three seeds could not separate anything, and there is no
    reason to expect HIDDEN_SIZE to behave differently. Rather than spend a day of compute
    earning another null, Methods states the parameter count and says the value was not tuned.
    This prints the number that argument needs.
    """
    sim = Simulator(controller.act)
    n_in = controller.input_size(sim.model)
    nu = sim.model.nu
    budget = config.POP_SIZE * config.NUM_GENERATIONS

    print(f"inputs {n_in}, outputs {nu}, no bias vector (bias is a constant input)")
    print(f"genotype = inputs*h + h*outputs = {n_in + nu}h\n")
    print(f"{'HIDDEN_SIZE':>12} {'genotype':>10} {'evals per weight':>18}")
    rows = []
    for h in args.sizes:
        length = n_in * h + h * nu
        print(f"{h:>12} {length:>10} {budget / length:>18.1f}")
        rows.append({"hidden_size": h, "genotype_length": length,
                     "evals_per_weight": round(budget / length, 1)})

    current = config.HIDDEN_SIZE
    print(f"\ncurrent HIDDEN_SIZE = {current} -> {(n_in + nu) * current} weights, "
          f"{budget} evaluations at the PROVISIONAL budget")
    print("Methods sentence: state the count, state that it was inherited from the template and")
    print("not tuned, and say why - the compute went into 20 seeds instead. Do not imply it was")
    print("chosen by experiment.")
    write_csv("hidden_size", rows)


# --------------------------------------------------------------------------- #
#  sigma - which fixed sigma, and does the 1/5 rule care where it starts?
# --------------------------------------------------------------------------- #
def _sigma_run(job: tuple[str, float, int, int, int]) -> list[dict]:
    """One full run of the real EA. Module level, so a spawned worker can pickle it.

    Unlike `search`, this is the algorithm Methods describes: ea.run_generation with
    tournament selection, elitism and mutation.py's Static/AdaptiveMutation. So it is also the
    first end-to-end test of ea.py and mutation.py together.

    Returns one row per generation, generation 0 being the initial population.
    """
    variant, sigma0, seed, pop_size, generations = job

    # survivor_selection reads config.POP_SIZE when it is called, so the pilot's smaller
    # population has to be patched in HERE, inside the worker. A spawned process imports
    # config afresh and would never see a patch made in the parent.
    config.POP_SIZE = pop_size

    rng = np.random.default_rng(seed)

    # The initial population is drawn FIRST, before anything else touches the generator, so
    # every variant and every sigma0 starts from the same population at a given seed.
    # `size` is passed explicitly: init_population's default was bound to config.POP_SIZE
    # when ea.py was imported, before the patch above.
    started = time.perf_counter()
    population = ea.evaluate(ea.init_population(rng, size=pop_size))

    if variant == "static":
        mutation = StaticMutation(rng, sigma=sigma0)
    else:
        mutation = AdaptiveMutation(rng, initial_sigma=sigma0)

    def row(generation: int, sigma: float | None, evaluations: int) -> dict:
        best, mean, std = ea.fitness_stats(population)
        return {
            "generation": generation,
            "variant": variant,
            "sigma0": sigma0,
            "seed": seed,
            "best_fitness": round(best, 6),
            "mean_fitness": round(mean, 6),
            "std_fitness": round(std, 6),
            "sigma": sigma,
            "success_rate": mutation.last_success_rate,
            "num_scored_mutations": mutation.last_num_scored,
            "num_mutated": mutation.last_num_mutated,
            "mean_genotype_spread": round(ea.genotype_spread(population), 6),
            "num_nan": ea.count_nan(population),
            "evaluations": evaluations,
            "seconds": round(time.perf_counter() - started, 1),
        }

    evaluations = pop_size
    rows = [row(0, None, evaluations)]

    for generation in range(1, generations + 1):
        # Read BEFORE the offspring are made, so the logged sigma is the one that made them.
        sigma_used = mutation.sigma
        population = ea.run_generation(variant, population, mutation, rng)
        evaluations += pop_size
        rows.append(row(generation, sigma_used, evaluations))

    # Keep each run's best controller, so the simulation-length check (`duration --genotype`)
    # and any video for the report can use a controller evolved on the John Set gecko.
    best = min(population, key=lambda ind: ind.fitness_)
    out = RESULTS_DIR / "sigma_best"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{variant}_sigma{sigma0}_seed{seed}.npy",
            np.asarray(best.genotype, dtype=np.float64))

    return rows


def experiment_sigma(args: argparse.Namespace) -> None:
    """Static AND adaptive at several sigma values, a few seeds each.

    Two questions, one sweep:

    1. WHICH FIXED SIGMA. The static variant is the control, and a badly chosen control makes
       the 1/5 rule look good for free. The fair comparison is the rule against a fixed sigma
       that was itself chosen by experiment - and A1's marker singled out exactly this kind of
       pilot ("what you did for parameter selection is exactly what we are looking for").

    2. DOES THE STARTING POINT MATTER TO THE RULE. If the adaptive variant ends in a similar
       place whatever sigma it starts from, the rule has replaced a tuning step. That is the
       standard argument for parameter control, and part two of the research question.

    Pilot scale by default: population 30 for 40 generations is 1,230 evaluations per run,
    the same budget as the body pilot, so the two are directly comparable. Three seeds can
    show a direction, not prove one - the summary says so where it matters.
    """
    jobs = [
        (variant, sigma0, seed, args.pop, args.generations)
        for seed in args.seeds
        for sigma0 in args.sigmas
        for variant in args.variants
    ]
    workers = max(1, min(args.workers, len(jobs)))
    per_run = args.pop * (args.generations + 1)
    print(f"{len(jobs)} runs: variants {args.variants}, sigma {args.sigmas}, seeds {args.seeds}")
    print(f"population {args.pop}, {args.generations} generations, {per_run} evaluations per run")
    print(f"{workers} worker process(es)\n")

    rows: list[dict] = []
    started = time.perf_counter()

    def report(done: int, run_rows: list[dict]) -> None:
        last = run_rows[-1]
        sigma_note = (
            f"final sigma {last['sigma']:.3f}" if last["variant"] == "adaptive" else ""
        )
        print(f"[{done:>3}/{len(jobs)}] {last['variant']:<8} sigma0 {last['sigma0']:<5} "
              f"seed {last['seed']:<3} best {last['best_fitness']:.3f} m  "
              f"{last['seconds']:>6.0f} s  {sigma_note}", flush=True)

    if workers == 1:
        for done, job in enumerate(jobs, start=1):
            run_rows = _sigma_run(job)
            rows.extend(run_rows)
            report(done, run_rows)
    else:
        # spawn, because that is all Windows has and it is what `determ` verified.
        with mp.get_context("spawn").Pool(processes=workers) as pool:
            for done, run_rows in enumerate(pool.imap_unordered(_sigma_run, jobs), start=1):
                rows.extend(run_rows)
                report(done, run_rows)

    print(f"\ntotal wall time {(time.perf_counter() - started) / 60:.1f} min")
    rows.sort(key=lambda r: (r["variant"], r["sigma0"], r["seed"], r["generation"]))
    write_csv("sigma_sweep", rows)
    _summarise_sigma(rows, args)


def _summarise_sigma(rows: list[dict], args: argparse.Namespace) -> None:
    """Final best per configuration, and what the adaptive sigma did."""
    last = {
        (r["variant"], r["sigma0"], r["seed"]): r
        for r in rows if r["generation"] == args.generations
    }

    def finals(variant: str, sigma0: float) -> list[float]:
        return [last[(variant, sigma0, s)]["best_fitness"] for s in args.seeds
                if (variant, sigma0, s) in last]

    def mean_sd(values: list[float]) -> tuple[float, float]:
        return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)

    print("\nFINAL BEST DISTANCE (m, lower is better)\n")
    print(f"{'variant':<9} {'sigma0':>7} {'mean':>7} {'sd':>7}   per seed")
    for variant in args.variants:
        for sigma0 in args.sigmas:
            values = finals(variant, sigma0)
            m, sd = mean_sd(values)
            print(f"{variant:<9} {sigma0:>7} {m:>7.3f} {sd:>7.3f}   "
                  + "  ".join(f"{v:.3f}" for v in values))

    # ---- question 1: which fixed sigma
    if "static" in args.variants and len(args.sigmas) > 1:
        ranked = sorted(args.sigmas, key=lambda s: statistics.fmean(finals("static", s)))
        best, runner_up = ranked[0], ranked[1]
        gap = statistics.fmean(finals("static", runner_up)) - statistics.fmean(finals("static", best))
        spread = statistics.fmean(mean_sd(finals("static", s))[1] for s in args.sigmas)
        print(f"\n1. WHICH FIXED SIGMA: lowest mean at sigma = {best}, "
              f"{gap:.3f} m ahead of sigma = {runner_up}")
        print(f"   typical spread between seeds at one sigma: {spread:.3f} m")
        if len(args.seeds) < 2:
            print("   -> one seed has no spread to compare against: run at least 2 seeds.")
        elif gap < spread:
            print("   -> the gap is inside the noise. Do not pick a winner on this alone; choose")
            print("      by argument and say in Methods that the pilot could not separate them.")
        else:
            print(f"   -> the gap is larger than the seed-to-seed spread: sigma = {best} is the")
            print("      candidate for STATIC_SIGMA. With 3 seeds that is a direction, not a test.")

    # ---- question 2: does the rule care where it starts
    if "adaptive" in args.variants and len(args.sigmas) > 1:
        print("\n2. WHAT THE 1/5 RULE DID WITH EACH STARTING SIGMA\n")
        print(f"{'sigma0':>7} {'final sigma (geo. mean)':>24} {'range between seeds':>21} "
              f"{'success, last 5 gens':>21}")
        geo_means, within = {}, []
        for sigma0 in args.sigmas:
            runs = [r for r in rows if r["variant"] == "adaptive" and r["sigma0"] == sigma0]
            ends = [r["sigma"] for r in runs if r["generation"] == args.generations]
            late = [r["success_rate"] for r in runs
                    if r["generation"] > args.generations - 5 and r["success_rate"] is not None]
            # Geometric mean, because the rule changes sigma by a FACTOR each generation.
            geo_means[sigma0] = float(np.exp(np.mean(np.log(ends))))
            within.append(max(ends) / min(ends))
            print(f"{sigma0:>7} {geo_means[sigma0]:>24.4f} "
                  f"{min(ends):>10.4f}-{max(ends):<10.4f} "
                  f"{(statistics.fmean(late) if late else float('nan')):>21.3f}")

        start_ratio = max(args.sigmas) / min(args.sigmas)
        between_ratio = max(geo_means.values()) / min(geo_means.values())
        within_ratio = max(within)
        print(f"\n   starting sigmas differ by {start_ratio:.1f}x")
        print(f"   final sigmas differ by {between_ratio:.1f}x between starting values, "
              f"and by up to {within_ratio:.1f}x between seeds of the SAME start")

        # Comparing only the means (what an earlier version did) hides the spread within one
        # starting value. If seeds of the same start end further apart than the starting values
        # were, sigma has not converged anywhere - it is wandering.
        if len(args.seeds) > 1 and within_ratio >= start_ratio:
            print("   -> sigma no longer reflects where it started, but it does not settle either: it")
            print("      wanders. That is NOT evidence that the rule replaces tuning.")
        elif between_ratio < start_ratio / 2:
            print("   -> different starting values end near a common sigma: evidence the rule replaces")
            print("      the tuning step the static variant needed.")
        else:
            print("   -> the final sigma still reflects where it started: within this budget the rule")
            print("      has NOT replaced tuning. That is a finding too - say so.")

        adaptive_runs = {(r["sigma0"], r["seed"]) for r in rows if r["variant"] == "adaptive"}
        capped = {(r["sigma0"], r["seed"]) for r in rows if r["variant"] == "adaptive"
                  and r["sigma"] is not None and r["sigma"] >= config.ADAPTIVE_MAX_SIGMA}
        print(f"\n   sigma reached its upper bound ({config.ADAPTIVE_MAX_SIGMA}) in "
              f"{len(capped)} of {len(adaptive_runs)} adaptive runs")

        if "static" in args.variants and len(args.seeds) > 1:
            def dependence(variant: str) -> float:
                means = [statistics.fmean(finals(variant, s)) for s in args.sigmas]
                return max(means) - min(means)
            spread = statistics.fmean(
                mean_sd(finals(v, s))[1] for v in args.variants for s in args.sigmas)
            print(f"\n   how much final distance depends on sigma0: static {dependence('static'):.3f} m, "
                  f"adaptive {dependence('adaptive'):.3f} m")
            print(f"   typical spread between seeds: {spread:.3f} m")
            if max(dependence("static"), dependence("adaptive")) < spread:
                print("   -> both inside the noise: no evidence either variant is sensitive to sigma0.")

    # ---- question 3: does the success rate respond to sigma at all?
    # The 1/5 rule assumes that larger steps succeed less often. The static variant logs its
    # success rate without acting on it, so it shows directly whether that assumption holds.
    if "static" in args.variants and len(args.sigmas) > 1:
        print("\n3. DOES THE SUCCESS RATE DEPEND ON SIGMA? (static variant, after the first window)\n")
        rates = {}
        for sigma0 in args.sigmas:
            values = [r["success_rate"] for r in rows
                      if r["variant"] == "static" and r["sigma0"] == sigma0
                      and r["generation"] > config.ADAPTIVE_WINDOW and r["success_rate"] is not None]
            rates[sigma0] = statistics.fmean(values) if values else float("nan")
            print(f"   sigma {sigma0:<5} mean success rate {rates[sigma0]:.3f}")
        ordered = [rates[s] for s in sorted(args.sigmas)]
        if max(ordered) - min(ordered) < 0.05:
            print("   -> the success rate barely changes with sigma. The rule's input then says little")
            print("      about whether sigma is too large or too small, so expect the adaptive sigma")
            print("      to wander rather than settle.")
        elif all(x >= y for x, y in zip(ordered, ordered[1:])):
            print("   -> the success rate falls as sigma grows, as the 1/5 rule assumes.")
        else:
            print("   -> the success rate does not fall steadily with sigma, which the 1/5 rule assumes.")

    print("\nPer-generation rows, including sigma and success rate, are in results/sigma_sweep.csv")
    print("for plotting the sigma trajectories. Each run's best controller is in results/sigma_best/.")


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    p = subparsers.add_parser("duration", help="is 15 s long enough to reach the target?")
    p.add_argument("--genotype", default="quick_best.npy")
    p.add_argument("--max", type=int, default=60, help="longest duration to try, seconds")
    p.add_argument("--step", type=int, default=5)
    p.add_argument("--reach", type=float, default=0.05, help="distance (m) that counts as reaching the target")
    p.add_argument("--floor", type=float, default=0.3, help="warn about a floor effect below this distance (m)")
    p.set_defaults(func=experiment_duration)

    p = subparsers.add_parser("clock", help="sweep CLOCK_FREQ")
    p.add_argument("--freqs", type=float, nargs="+", default=[0.5, 1.0, 2.0, 3.0])
    p.add_argument("--generations", type=int, default=20)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3],
                   help="one run per frequency per seed; the mean across seeds is the result")
    p.set_defaults(func=experiment_clock)

    p = subparsers.add_parser("qvel", help="does adding joint velocities help?")
    p.add_argument("--generations", type=int, default=20)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.set_defaults(func=experiment_qvel)

    p = subparsers.add_parser("nan", help="does the NaN guard fire?")
    p.add_argument("--seed", type=int, default=getattr(config, "DEFAULT_SEED", 1))
    p.set_defaults(func=experiment_nan)

    p = subparsers.add_parser("cache", help="what does compiling once buy?")
    p.add_argument("--n", type=int, default=10, help="evaluations per condition")
    p.add_argument("--seed", type=int, default=getattr(config, "DEFAULT_SEED", 1))
    p.set_defaults(func=experiment_cache)

    p = subparsers.add_parser("determ", help="bit-identical across rebuilds and processes?")
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--seed", type=int, default=getattr(config, "DEFAULT_SEED", 1))
    p.set_defaults(func=experiment_determinism)

    p = subparsers.add_parser("hidden", help="genotype length vs HIDDEN_SIZE (no simulation)")
    p.add_argument("--sizes", type=int, nargs="+", default=[2, 4, 6, 8, 12, 16])
    p.set_defaults(func=experiment_hidden)

    p = subparsers.add_parser("sigma", help="fixed-sigma pilot + does the 1/5 rule care where it starts")
    p.add_argument("--sigmas", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    p.add_argument("--variants", nargs="+", default=["static", "adaptive"],
                   choices=["static", "adaptive"])
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--pop", type=int, default=30, help="population size (pilot scale)")
    p.add_argument("--generations", type=int, default=40)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                   help="parallel runs; lower this if the laptop struggles")
    p.set_defaults(func=experiment_sigma)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
