import math
import time

import mujoco as mj
import numpy as np
import numpy.typing as npt

import config

# --------------------------------------------------------------------------- #
#  Settings
# --------------------------------------------------------------------------- #
N_FREE_JOINT_QPOS: int = 7  # x, y, z + quaternion (w, x, y, z)
N_EXTRA_INPUTS: int = 6  # 2 clock + 2 direction + 1 distance + 1 bias
HINGE_LIMIT: float = np.pi / 2
TARGET_X: float = float(config.TARGET_POSITION[0])
TARGET_Y: float = float(config.TARGET_POSITION[1])
TWO_PI: float = 2.0 * math.pi


# --------------------------------------------------------------------------- #
#  Genotype --> weights
# --------------------------------------------------------------------------- #
def _check_layout(model: mj.MjModel) -> None:
    """qpos must be [free joint (7)] + [one angle per actuated hinge]."""
    expected = N_FREE_JOINT_QPOS + model.nu
    if model.nq != expected:
        msg = (
            f"Unexpected qpos layout: nq={model.nq}, expected 7 + nu = {expected}. "
            "Does the world contain other joints, or is a hinge unactuated?"
        )
        raise ValueError(msg)


def input_size(model: mj.MjModel) -> int:
    """Number of network inputs for the compiled body."""
    _check_layout(model)
    return model.nu + N_EXTRA_INPUTS


def genotype_length(model: mj.MjModel) -> int:
    """Length of the flat weight vector the EA evolves."""
    return input_size(model) * config.HIDDEN_SIZE + config.HIDDEN_SIZE * model.nu


def unpack(
    genotype: npt.NDArray[np.float64],
    model: mj.MjModel,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Reshape the flat genotype into (w1, w2). Views, no copying."""
    n_in = input_size(model)
    expected = n_in * config.HIDDEN_SIZE + config.HIDDEN_SIZE * model.nu
    if genotype.shape != (expected,):
        msg = f"Genotype has shape {genotype.shape}, expected ({expected},)"
        raise ValueError(msg)
    split = n_in * config.HIDDEN_SIZE
    return (
        genotype[:split].reshape(n_in, config.HIDDEN_SIZE),
        genotype[split:].reshape(config.HIDDEN_SIZE, model.nu),
    )


# --------------------------------------------------------------------------- #
#  Inputs and forward pass
# --------------------------------------------------------------------------- #
# Reused across calls so the input vector is not reallocated on every physics step.
_INPUT_BUFFER: npt.NDArray[np.float64] | None = None


def build_inputs(data: mj.MjData) -> npt.NDArray[np.float64]:
    """Assemble the input vector described in the module docstring.

    Returns a SHARED buffer that is overwritten on the next call; copy it if you need to keep it."""
    global _INPUT_BUFFER

    qpos = data.qpos
    n_hinges = len(qpos) - N_FREE_JOINT_QPOS
    size = n_hinges + N_EXTRA_INPUTS

    if _INPUT_BUFFER is None or _INPUT_BUFFER.size != size:
        _INPUT_BUFFER = np.empty(size, dtype=np.float64)
    buffer = _INPUT_BUFFER

    buffer[:n_hinges] = qpos[N_FREE_JOINT_QPOS:]

    # Heading (yaw) of the core, from its quaternion (w, x, y, z)
    w = qpos[3]
    x = qpos[4]
    y = qpos[5]
    z = qpos[6]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    # Vector to the target, rotated into the robot's own frame, so the network sees "ahead-left"
    dx = TARGET_X - qpos[0]
    dy = TARGET_Y - qpos[1]
    c = math.cos(yaw)
    s = math.sin(yaw)
    tx = c * dx + s * dy
    ty = -s * dx + c * dy
    dist = np.hypot(tx, ty)

    # data.time restarts at 0 on every mj_resetData, so the clock starts in the same phase.
    phase = TWO_PI * config.CLOCK_FREQ * data.time

    buffer[n_hinges] = math.sin(phase)
    buffer[n_hinges + 1] = math.cos(phase)
    if dist > 1e-8:
        buffer[n_hinges + 2] = tx / dist
        buffer[n_hinges + 3] = ty / dist
    else:
        buffer[n_hinges + 2] = 0.0
        buffer[n_hinges + 3] = 0.0
    buffer[n_hinges + 4] = min(dist / config.DIST_SCALE, 2.0)
    buffer[n_hinges + 5] = 1.0

    return buffer


def forward(
    inputs: npt.NDArray[np.float64],
    w1: npt.NDArray[np.float64],
    w2: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Network forward pass. Returns values in [-1, 1]."""
    return np.tanh(np.tanh(inputs @ w1) @ w2)


def act(
    model: mj.MjModel,
    data: mj.MjData,
    genotype: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Hinge targets for this step, scaled to [-pi/2, pi/2].

    Pure: writes nothing. NaN weights give NaN actions, which simulate.py scores as
    WORST_FITNESS.
    """
    w1, w2 = unpack(genotype, model)
    return forward(build_inputs(data), w1, w2) * HINGE_LIMIT


# --------------------------------------------------------------------------- #
#  Self-test: python controller.py
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from simulate import Simulator

    sim = Simulator(act)
    n = genotype_length(sim.model)
    print(f"hinges (outputs)            : {sim.model.nu}")
    print(f"inputs                      : {input_size(sim.model)}")
    print(f"genotype length             : {n}")

    rng = np.random.default_rng(config.DEFAULT_SEED)
    genotype = rng.normal(scale=0.5, size=n)

    actions = act(sim.model, sim.data, genotype)
    in_range = bool(np.all(np.abs(actions) <= HINGE_LIMIT))
    print(f"act() output                : shape {actions.shape}, in range {in_range}")

    started = time.perf_counter()
    first = sim.evaluate(genotype)
    elapsed = time.perf_counter() - started
    repeats = [sim.evaluate(genotype) for _ in range(2)]
    print(f"fitness (lower is better)   : {first:.6f}")
    print(f"seconds per evaluation      : {elapsed:.3f}")
    print(f"repeats                     : {[round(r, 6) for r in repeats]}")
    if all(r == first for r in repeats):
        print("determinism                 : OK, identical every time")
    else:
        print("determinism                 : NOT identical, the 1/5 success signal is unreliable"
        )
