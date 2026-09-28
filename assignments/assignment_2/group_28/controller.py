"""Neural controller for EC Assignment 2 (Group 28).

Maps the robot's state to hinge targets.:

    from simulate import get_simulator
    import controller

    sim = get_simulator(controller.act)
    n_weights = controller.genotype_length(sim.model)
    fitness = sim.evaluate(genotype)

WHAT simulate.py EXPECTS (and this file provides)
-------------------------------------------------
    act(model, data, genotype) -> ndarray of length model.nu,
                                  already scaled to [-pi/2, pi/2]

`act` only COMPUTES actions. It never writes data.ctrl: applying them
(DELTA with CONTROL_ALPHA, clipping, the NaN guard) is simulate.py's job, so
there is exactly one place that decides how commands reach the motors.

WHAT THE EA NEEDS FROM HERE
---------------------------
    genotype_length(model) -> int     length of the flat weight vector
    input_size(model)      -> int     number of network inputs

Both read their sizes from the compiled model, so they follow whatever body
simulate.build_robot() returns. Never hardcode them.

NETWORK
-------
    inputs (nu + 6) --tanh--> hidden (HIDDEN_SIZE) --tanh--> nu outputs * pi/2

    genotype = [w1.ravel(), w2.ravel()]. The bias is a constant input of 1.0,
    so there is no separate bias vector in the genotype.

INPUTS, in this order
---------------------
    qpos[7:]            hinge angles (one per hinge)              nu values
    sin(wt), cos(wt)    clock: something to drive rhythmic motion  2 values
    target direction    unit vector, in the robot's own frame      2 values
    target distance     / DIST_SCALE, clipped to [0, 2]            1 value
    1.0                 bias                                       1 value

The brief warns that a controller "with no signal telling it where the target
is, or nothing to drive rhythmic movement with, has very little to work with".
The clock and target inputs are the answer to exactly that.

SETTINGS
--------
TARGET_POSITION comes from config.py, the same value simulate.py scores
against. HIDDEN_SIZE, CLOCK_FREQ and DIST_SCALE are read from config.py if
they are defined there, otherwise the defaults below are used. Move them into
config.py once they are fixed, so every setting lives in one place.
"""

# Standard library
import time

# Third-party libraries
import mujoco as mj
import numpy as np
import numpy.typing as npt

import config

# --------------------------------------------------------------------------- #
#  Settings (config.py wins if it defines them)
# --------------------------------------------------------------------------- #
HIDDEN_SIZE: int = getattr(config, "HIDDEN_SIZE", 6)
CLOCK_FREQ: float = getattr(config, "CLOCK_FREQ", 1.0)  # Hz; choose in pilots
DIST_SCALE: float = getattr(config, "DIST_SCALE", 2.0)  # spawn-target distance
TARGET_XY = np.asarray(config.TARGET_POSITION[:2], dtype=np.float64)

N_FREE_JOINT_QPOS: int = 7  # x, y, z + quaternion (w, x, y, z)
N_EXTRA_INPUTS: int = 6  # 2 clock + 2 direction + 1 distance + 1 bias
HINGE_LIMIT: float = np.pi / 2


# --------------------------------------------------------------------------- #
#  Sizes
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
    return input_size(model) * HIDDEN_SIZE + HIDDEN_SIZE * model.nu


# --------------------------------------------------------------------------- #
#  Genotype -> weights
# --------------------------------------------------------------------------- #
def unpack(
    genotype: npt.NDArray[np.float64],
    model: mj.MjModel,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Reshape the flat genotype into (w1, w2). Views, no copying."""
    n_in = input_size(model)
    expected = n_in * HIDDEN_SIZE + HIDDEN_SIZE * model.nu
    if genotype.shape != (expected,):
        msg = f"Genotype has shape {genotype.shape}, expected ({expected},)"
        raise ValueError(msg)
    split = n_in * HIDDEN_SIZE
    return (
        genotype[:split].reshape(n_in, HIDDEN_SIZE),
        genotype[split:].reshape(HIDDEN_SIZE, model.nu),
    )


# --------------------------------------------------------------------------- #
#  Inputs and forward pass
# --------------------------------------------------------------------------- #
def build_inputs(data: mj.MjData) -> npt.NDArray[np.float64]:
    """Assemble the input vector described in the module docstring."""
    qpos = data.qpos

    # Heading (yaw) of the core, from its quaternion (w, x, y, z).
    w, x, y, z = qpos[3:7]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    # Vector to the target, rotated into the robot's own frame, so the
    # network sees "ahead-left", not world coordinates.
    dx = TARGET_XY[0] - qpos[0]
    dy = TARGET_XY[1] - qpos[1]
    c, s = np.cos(yaw), np.sin(yaw)
    tx = c * dx + s * dy
    ty = -s * dx + c * dy
    dist = np.hypot(tx, ty)
    direction = (tx / dist, ty / dist) if dist > 1e-8 else (0.0, 0.0)

    # data.time restarts at 0 on every mj_resetData, so the clock phase is
    # identical at the start of every rollout.
    phase = 2.0 * np.pi * CLOCK_FREQ * data.time

    return np.concatenate((
        qpos[N_FREE_JOINT_QPOS:],
        (np.sin(phase), np.cos(phase)),
        direction,
        (min(dist / DIST_SCALE, 2.0), 1.0),
    ))


def forward(
    inputs: npt.NDArray[np.float64],
    w1: npt.NDArray[np.float64],
    w2: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Network forward pass. Returns values in [-1, 1]."""
    return np.tanh(np.tanh(inputs @ w1) @ w2)


# --------------------------------------------------------------------------- #
#  The function simulate.py calls every physics step
# --------------------------------------------------------------------------- #
def act(
    model: mj.MjModel,
    data: mj.MjData,
    genotype: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Hinge targets for this step, scaled to [-pi/2, pi/2].

    Pure: reads model and data, writes nothing. NaN weights produce NaN
    actions, which simulate.py detects and scores as WORST_FITNESS.
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

    rng = np.random.default_rng(getattr(config, "DEFAULT_SEED", 0))
    genotype = rng.normal(scale=0.5, size=n)

    # Check act() alone: right shape, right range.
    actions = act(sim.model, sim.data, genotype)
    in_range = bool(np.all(np.abs(actions) <= HINGE_LIMIT))
    print(f"act() output                : shape {actions.shape}, in range {in_range}")

    # Full rollouts through simulate.py.
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
        print(
            "determinism                 : NOT identical. The same genotype scores "
            "differently,\n                              so the 1/5 success signal is "
            "unreliable. See simulate.py (simple_runner starts DELTA from a random ctrl)."
        )