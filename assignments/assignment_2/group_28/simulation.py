import mujoco as mj
import numpy as np

from ariel.body_phenotypes.robogen_lite.prebuilt_robots import john_set
from ariel.simulation.environments import SimpleFlatWorld
from ariel.simulation.tasks.targeted_locomotion import distance_to_target
from ariel.utils.runners import simple_runner

import brain
import config

TARGET = np.asarray(config.TARGET_POSITION)


def build_world() -> tuple[mj.MjModel, mj.MjData]:
    """Spawn the robot in the world and compile it. Done once per process, compiling is slow."""
    mj.set_mjcb_control(None)  # MuJoCo's control callback is global, clear any old one

    world = SimpleFlatWorld()
    # The John Set gecko (6 hinges). Not the gecko from the template, that one has 8 hinges.
    robot = john_set.gecko()
    world.spawn(robot.spec, position=config.SPAWN_POS, correct_collision_with_floor=True)
    model = world.spec.compile()

    # By default MuJoCo resets an unstable simulation, which sets data.time back to 0, so
    # simple_runner never finishes. Without the reset the state just becomes NaN, which the
    # controller below catches.
    model.opt.disableflags |= mj.mjtDisableBit.mjDSBL_AUTORESET

    return model, mj.MjData(model)


def make_controller(genotype: np.ndarray, model: mj.MjModel, duration: float = config.SIM_DURATION):
    """The function MuJoCo calls every physics step, plus a dict that remembers if the
    network ever gave NaN (that happens when mutation makes the weights blow up)."""
    w1, w2 = brain.split_weights(genotype, model)
    status = {"failed": False}

    def control(m: mj.MjModel, d: mj.MjData) -> None:
        targets = brain.hinge_targets(d, w1, w2)

        if not np.all(np.isfinite(targets)):
            # Put the robot back in its start pose and jump to the end so the run stops,
            # otherwise MuJoCo keeps stepping (and warning) on a NaN state
            status["failed"] = True
            d.ctrl[:] = 0.0
            d.qpos[:] = m.qpos0
            d.qvel[:] = 0.0
            d.time = duration
            return

        # Delta control like in the template: move a small step towards the targets. The
        # steps add up, so ctrl has to be clipped to the hinge range.
        d.ctrl[:] += targets * config.CONTROL_ALPHA
        d.ctrl[:] = np.clip(d.ctrl, -np.pi / 2, np.pi / 2)

    return control, status


def evaluate(genotype, model: mj.MjModel, data: mj.MjData, duration: float = config.SIM_DURATION) -> float:
    """Simulate one genotype and return its distance to the target (lower is better)."""
    genotype = np.asarray(genotype, dtype=np.float64)
    control, status = make_controller(genotype, model, duration)

    # simple_runner resets data itself (time and ctrl back to 0), so every run starts the same
    mj.set_mjcb_control(control)
    try:
        simple_runner(model, data, duration=duration)
    finally:
        mj.set_mjcb_control(None)

    if status["failed"]:
        return config.WORST_FITNESS
    return distance_to_target(data.qpos[0:3].copy(), TARGET)
