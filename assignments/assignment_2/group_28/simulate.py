"""One rollout of the physics simulation, scored. The EA/physics boundary.

WHAT THIS FILE IS FOR
---------------------
This is the only module in the project that imports MuJoCo. Everything above it
speaks in numbers - a genotype is a flat float vector, a fitness is one float,
lower is better - and everything below it speaks in bodies, hinges and contacts.

    flat float vector  ->  [ physics ]  ->  one fitness number

Keeping that boundary here means ea.py, mutation.py and analyze.py never import
mujoco, so the EA can be unit-tested against a cheap synthetic fitness (a sphere
function, say) in seconds instead of waiting on rollouts. That matters when the
research question is about the 1/5 rule rather than about robots.

HOW TO USE IT
-------------
Build one Simulator per process, then call `evaluate` many times:

    from simulate import get_simulator

    sim = get_simulator(controller.act)      # compiles the model once
    fitness = sim.evaluate(genotype)         # ~0.168 s, deterministic

`get_simulator` caches per process, which is what the multiprocessing runner
needs: MuJoCo models do not pickle, so each worker must build its own. Call it
inside the worker, never in the parent.

In ea.py, keep the A1 population-level shape and let it delegate:

    def evaluate(population: list[dict]) -> list[dict]:
        sim = get_simulator(controller.act)
        for individual in population:
            if individual.get("fitness") is None:
                individual["fitness"] = sim.evaluate(individual["genotype"])
        return population

For the report figures, `rollout(genotype, mode="video")` runs the same physics
through a renderer instead of the headless runner.

WHAT IT EXPECTS FROM controller.py
----------------------------------
A single callable, injected at construction rather than imported, so this module
can be built and tested before controller.py exists:

    controller_fn(model, data, genotype) -> ndarray of length model.nu,
                                            already scaled to [-pi/2, pi/2]

Dependency injection also means the walk smoke test can pass a hand-written
controller straight in without touching this file.

WHAT IT EXPECTS FROM config.py
------------------------------
    SPAWN_POS, TARGET_POSITION, SIM_DURATION, CONTROL_ALPHA, WORST_FITNESS

Run `python simulate.py` for a self-test: it prints the controller sizes, the
genotype length, a fitness from random weights, and the seconds per evaluation.
"""

# Standard library
import time
from typing import Callable, Literal

# Third-party libraries
import mujoco as mj
import numpy as np
import numpy.typing as npt

# Local libraries (ARIEL)
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.environments import SimpleFlatWorld
from ariel.simulation.tasks.targeted_locomotion import distance_to_target
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.video_recorder import VideoRecorder

import config

type RolloutModes = Literal["simple", "video", "frame"]

# (model, data, genotype) -> actions. See "WHAT IT EXPECTS FROM controller.py".
type ControllerFn = Callable[
    [mj.MjModel, mj.MjData, npt.NDArray[np.float64]],
    npt.NDArray[np.float64],
]


# --------------------------------------------------------------------------- #
#  Body and world
# --------------------------------------------------------------------------- #
# Both copied verbatim from A2_template_2026.py:77 and :91. Decided and FIXED
# for the whole assignment: changing either changes model.nu and len(data.qpos),
# and therefore the genotype length, so runs before and after are not comparable.


def build_world() -> SimpleFlatWorld:
    """The environment. A2_template_2026.py:77."""
    return SimpleFlatWorld()


def build_robot() -> CoreModule:
    """The body. A2_template_2026.py:91."""
    return gecko()


def get_core_position(data: mj.MjData) -> npt.NDArray[np.float64]:
    """The core's (x, y, z) world position. A2_template_2026.py:206.

    The robot spawns with a free joint, so data.qpos[0:3] IS the core position -
    no tracker needed. `.copy()` matters: qpos is a live view into the MuJoCo
    state and will change under you on the next step.
    """
    return np.asarray(data.qpos[0:3]).copy()


# --------------------------------------------------------------------------- #
#  The simulator
# --------------------------------------------------------------------------- #


class Simulator:
    """Compiles the world once, then scores genotypes against it.

    The compile-once/evaluate-many split is worth roughly 1.25x (measured in
    phase1_timing.py, 25 Sep). Everything else is physics stepping, so there is
    no further speedup to chase here.
    """

    def __init__(
        self,
        controller_fn: ControllerFn,
        duration: float = config.SIM_DURATION,
        spawn_pos: list[float] | None = None,
        target_position: list[float] | None = None,
    ) -> None:
        self.controller_fn = controller_fn
        self.duration = duration
        self.spawn_pos = spawn_pos if spawn_pos is not None else config.SPAWN_POS
        self.target = np.asarray(
            target_position if target_position is not None else config.TARGET_POSITION
        )

        # MuJoCo's control callback is a GLOBAL. Clear it before building
        # anything. A2_template_2026.py:250 - "DO NOT REMOVE".
        mj.set_mjcb_control(None)

        # Spawn and compile. A2_template_2026.py:253-264.
        world = build_world()
        robot = build_robot()
        world.spawn(
            robot.spec,
            position=self.spawn_pos,
            correct_collision_with_floor=True,
        )
        self.model = world.spec.compile()
        self.data = mj.MjData(self.model)

        # Clean, known state before reading anything. A2_template_2026.py:267.
        mj.mj_resetData(self.model, self.data)
        mj.mj_forward(self.model, self.data)

        # Sizes come from the compiled model, never hardcoded. They depend on
        # the body chosen above. A2_template_2026.py:273-274.
        self.output_size: int = self.model.nu
        self.qpos_size: int = len(self.data.qpos)

        # Set by the control callback when the network emits NaN; see _make_callback.
        self._nan_seen: bool = False

    # -- the hot path -------------------------------------------------------- #

    def evaluate(self, genotype: npt.NDArray[np.float64]) -> float:
        """Score one genotype. LOWER IS BETTER. Called once per individual."""
        return self.rollout(genotype, mode="simple")

    def rollout(
        self,
        genotype: npt.NDArray[np.float64],
        mode: RolloutModes = "simple",
        video_folder: str | None = None,
    ) -> float:
        """One full simulation, scored.

        `mode="simple"` is headless and is what the EA uses. "video" and "frame"
        run identical physics through a renderer - use them for report figures,
        never inside the evolutionary loop.
        """
        genotype = np.asarray(genotype, dtype=np.float64)

        # Reset BEFORE reading the start position, not after.
        #
        # simple_runner calls mj_resetData itself (ariel/utils/runners.py:30), so
        # it is tempting to skip this. Don't: on the second and later calls of a
        # reused Simulator, data still holds the PREVIOUS rollout's end state, and
        # reading initial_position here would silently give you that instead of
        # the spawn. It is finite, plausible, and poisons fitness_delta_distance.
        # Resetting twice is harmless; resetting once, too late, is not.
        #
        # The reset also zeroes data.ctrl and data.time, which DELTA control and
        # any sin(2*pi*f*t) oscillator input both rely on starting from zero.
        mj.set_mjcb_control(None)
        mj.mj_resetData(self.model, self.data)
        mj.mj_forward(self.model, self.data)

        initial_position = get_core_position(self.data)

        self._nan_seen = False
        mj.set_mjcb_control(self._make_callback(genotype))

        try:
            match mode:
                case "simple":
                    # Headless. A2_template_2026.py:304.
                    simple_runner(self.model, self.data, duration=self.duration)
                case "video":
                    recorder = VideoRecorder(output_folder=video_folder or "__videos__")
                    video_renderer(
                        self.model,
                        self.data,
                        duration=self.duration,
                        video_recorder=recorder,
                    )
                case "frame":
                    single_frame_renderer(self.model, self.data, steps=1, show=True)
        finally:
            # Detach again so the next rollout starts clean, even if the run
            # raised. A2_template_2026.py:321.
            mj.set_mjcb_control(None)

        # A NaN controller produced no meaningful motion, so the final position
        # is meaningless too. Score it worst-possible rather than returning a
        # number that looks fine. Never return None: it would crash the
        # `min(contestants, key=...)` in tournament selection (A1 ea.py:35).
        if self._nan_seen:
            return config.WORST_FITNESS

        final_position = get_core_position(self.data)

        # Planar Euclidean distance to the target, lower is better. Imported
        # rather than reimplemented: ariel/simulation/tasks/targeted_locomotion.py:9.
        # The delta-distance variant on :14 is the alternative we considered -
        # if we switch, switch once and say so in Methods.
        return distance_to_target(final_position, self.target)

    # -- internals ----------------------------------------------------------- #

    def _make_callback(self, genotype: npt.NDArray[np.float64]):
        """Build the per-rollout control callback bound to this genotype.

        Built fresh each rollout on purpose. A module-level callback reading a
        shared `weights` variable is the classic way to evolve against the wrong
        genotype without any error being raised.
        """

        def control_callback(m: mj.MjModel, d: mj.MjData) -> None:
            """MuJoCo calls this every physics step. A2_template_2026.py:278."""
            actions = self.controller_fn(m, d, genotype)

            # Blown-up weights write NaN into data.ctrl silently, the simulation
            # carries on, and the run returns a plausible-looking fitness. Flag
            # it and hold still instead. A2_template_2026.py:126 warns about this.
            if not np.all(np.isfinite(actions)):
                self._nan_seen = True
                d.ctrl[:] = 0.0
                return

            # DELTA application, per the controller contract at
            # A2_template_2026.py:121-125: smoother than DIRECT, which can
            # destabilise the sim on large jumps, but it accumulates, so the clip
            # is required rather than optional. Chosen once, used everywhere.
            d.ctrl[:] += actions * config.CONTROL_ALPHA
            d.ctrl[:] = np.clip(d.ctrl, -np.pi / 2, np.pi / 2)

        return control_callback


# --------------------------------------------------------------------------- #
#  Per-process cache
# --------------------------------------------------------------------------- #

_SIMULATOR: Simulator | None = None


def get_simulator(controller_fn: ControllerFn, **kwargs) -> Simulator:
    """Return this process's Simulator, building it on first call.

    MuJoCo models cannot be pickled, so a multiprocessing worker cannot receive
    one from the parent - it has to compile its own. Calling this inside the
    worker gets each process exactly one compile instead of one per individual.
    """
    global _SIMULATOR
    if _SIMULATOR is None:
        _SIMULATOR = Simulator(controller_fn, **kwargs)
    return _SIMULATOR


# --------------------------------------------------------------------------- #
#  Self-test
# --------------------------------------------------------------------------- #


def _demo_controller_factory(qpos_size: int, output_size: int, hidden: int = 6):
    """PLACEHOLDER, for this file's self-test only - controller.py replaces it.

    Reproduces the template's bare architecture (A2_template_2026.py:135): raw
    qpos in, one tanh hidden layer, tanh out, rescaled to the hinge range. The
    real controller adds sin/cos(2*pi*f*t) and the vector to the target, without
    which there is nothing to drive a gait and nothing to steer by.
    """
    length = qpos_size * hidden + hidden * output_size

    def act(m: mj.MjModel, d: mj.MjData, genotype: npt.NDArray[np.float64]):
        assert len(genotype) == length, f"expected {length} weights, got {len(genotype)}"
        split = qpos_size * hidden
        w1 = genotype[:split].reshape(qpos_size, hidden)
        w2 = genotype[split:].reshape(hidden, output_size)
        layer1 = np.tanh(d.qpos @ w1)
        return np.tanh(layer1 @ w2) * (np.pi / 2)

    return act, length


if __name__ == "__main__":
    rng = np.random.default_rng(config.DEFAULT_SEED)

    probe = Simulator(lambda m, d, g: np.zeros(m.nu))
    act, genotype_length = _demo_controller_factory(probe.qpos_size, probe.output_size)

    sim = Simulator(act)
    print(f"controller inputs (len(data.qpos)) : {sim.qpos_size}")
    print(f"controller outputs (model.nu)      : {sim.output_size}")
    print(f"genotype length (total weights)    : {genotype_length}")

    genotype = rng.normal(scale=0.5, size=genotype_length)

    started = time.perf_counter()
    fitness = sim.evaluate(genotype)
    elapsed = time.perf_counter() - started
    print(f"fitness (lower is better)          : {fitness:.4f}")
    print(f"seconds per evaluation             : {elapsed:.3f}")

    # Determinism: the same genotype must score identically on a reused
    # Simulator. If this ever prints a non-zero difference, the reset discipline
    # in rollout() has been broken and the 1/5 success measure is unreliable.
    repeat = sim.evaluate(genotype)
    print(f"determinism (repeat - first)       : {repeat - fitness:.2e}")
