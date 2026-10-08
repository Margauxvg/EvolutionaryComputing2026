import time
from typing import Callable, Literal

import mujoco as mj
import numpy as np
import numpy.typing as npt

from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.prebuilt_robots import john_set
from ariel.simulation.environments import SimpleFlatWorld
from ariel.simulation.tasks.targeted_locomotion import distance_to_target
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.video_recorder import VideoRecorder

import config

type RolloutModes = Literal["simple", "video", "frame"]
type ControllerFn = Callable[
    [mj.MjModel, mj.MjData, npt.NDArray[np.float64]],
    npt.NDArray[np.float64],
]


# --------------------------------------------------------------------------- #
#  Body and world
# --------------------------------------------------------------------------- #

def build_world() -> SimpleFlatWorld:
    """Create the environment the robot lives in. A2_template_2026.py:77."""
    return SimpleFlatWorld()


def build_robot() -> CoreModule:
    """Create the robot body. The body: the John Set gecko. 6 hinges, 12 controller inputs, 108 weights."""
    return john_set.gecko()


def get_core_position(data: mj.MjData) -> npt.NDArray[np.float64]:
    """Return the robot core's current (x, y, z) world position. A2_template_2026.py:206."""
    return np.asarray(data.qpos[0:3]).copy()


# --------------------------------------------------------------------------- #
#  The simulator
# --------------------------------------------------------------------------- #


class Simulator:
    """Compiles the world once, then scores genotypes against it."""

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

        # MuJoCo's control callback is a GLOBAL. Clear it. DO NOT REMOVE.
        mj.set_mjcb_control(None)

        # Wolrd and robot
        world = build_world()
        robot = build_robot()
       
        world.spawn(
            robot.spec,
            position=self.spawn_pos,
            correct_collision_with_floor=True,
        )
        
        # Compile the world into a model
        self.model = world.spec.compile()
        self.data = mj.MjData(self.model)

        # By default MuJoCo resets the simulation when it becomes unstable, which sets
        # data.time back to 0, so simple_runner would replay forever (seen on 1 Oct: the speed
        # test hung on seed 13). With autoreset off the state turns to NaN instead, the NaN
        # guard in the callback catches it, and the controller scores WORST_FITNESS.
        self.model.opt.disableflags |= mj.mjtDisableBit.mjDSBL_AUTORESET


        # Clean, known state before reading anything
        mj.mj_resetData(self.model, self.data)
        mj.mj_forward(self.model, self.data)

        self.output_size: int = self.model.nu
        self.qpos_size: int = len(self.data.qpos)

        # Set by the control callback when the network emits NaN; see _make_callback
        self._nan_seen: bool = False


    def evaluate(self, genotype: npt.NDArray[np.float64]) -> float:
        """Score one genotype. LOWER IS BETTER. Called once per individual."""
        return self.rollout(genotype, mode="simple")

    def rollout(
        self,
        genotype: npt.NDArray[np.float64],
        mode: RolloutModes = "simple",
        video_folder: str | None = None,
    ) -> float:
        """One full simulation, scored."""
        genotype = np.asarray(genotype, dtype=np.float64)

        # Start from a clean state: the reset also zeroes data.ctrl and data.time, which DELTA
        # control and the sin/cos clock input both rely on
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
            mj.set_mjcb_control(None)

        if self._nan_seen:
            return config.WORST_FITNESS

        final_position = get_core_position(self.data)

        return distance_to_target(final_position, self.target)


    def _make_callback(self, genotype: npt.NDArray[np.float64]):
        """Per-rollout control callback bound to this genotype."""

        def control_callback(m: mj.MjModel, d: mj.MjData) -> None:
            """MuJoCo calls this every physics step. A2_template_2026.py:278."""
            actions = self.controller_fn(m, d, genotype)

            # Blown-up weights write NaN into data.ctrl silently. Flag
            # it and hold still instead
            if not np.all(np.isfinite(actions)):
                self._nan_seen = True
                d.ctrl[:] = 0.0
                d.qpos[:] = m.qpos0
                d.qvel[:] = 0.0
                d.time = self.duration
                return

            # DELTA control: smoother than direct targets, but it accumulates, so the clip is
            # required.
            d.ctrl[:] += actions * config.CONTROL_ALPHA
            d.ctrl[:] = np.clip(d.ctrl, -np.pi / 2, np.pi / 2)

        return control_callback


# --------------------------------------------------------------------------- #
#  Per-process cache
# --------------------------------------------------------------------------- #

_SIMULATOR: Simulator | None = None


def get_simulator(controller_fn: ControllerFn, **kwargs) -> Simulator:
    """Return this process's Simulator, building it on first call."""
    global _SIMULATOR
    if _SIMULATOR is None:
        _SIMULATOR = Simulator(controller_fn, **kwargs)
    return _SIMULATOR
