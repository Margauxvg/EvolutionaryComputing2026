import math

import mujoco as mj
import numpy as np

import config

# data.qpos starts with the core's free joint: position (x, y, z) and orientation as a
# quaternion (w, x, y, z). After that comes one angle per hinge.
FREE_JOINT_SIZE: int = 7
EXTRA_INPUTS: int = 6  # sin, cos, direction to target (x, y), distance to target, bias


def num_inputs(model: mj.MjModel) -> int:
    return model.nu + EXTRA_INPUTS


def num_weights(model: mj.MjModel) -> int:
    """Length of the genotype: input->hidden weights plus hidden->output weights."""
    return num_inputs(model) * config.HIDDEN_SIZE + config.HIDDEN_SIZE * model.nu


def split_weights(genotype: np.ndarray, model: mj.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """Cut the flat genotype into the two weight matrices of the network."""
    split = num_inputs(model) * config.HIDDEN_SIZE
    w1 = genotype[:split].reshape(num_inputs(model), config.HIDDEN_SIZE)
    w2 = genotype[split:].reshape(config.HIDDEN_SIZE, model.nu)
    return w1, w2


def sensor_inputs(data: mj.MjData) -> np.ndarray:
    """Hinge angles, a clock, and where the target is as seen from the robot."""
    qpos = data.qpos
    x, y = qpos[0], qpos[1]
    qw, qx, qy, qz = qpos[3], qpos[4], qpos[5], qpos[6]

    # Which way the robot is facing (yaw), from its quaternion:
    # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

    # Rotate the vector to the target into the robot's own frame, so "ahead" means the same
    # thing whichever way the robot is turned.
    dx = config.TARGET_POSITION[0] - x
    dy = config.TARGET_POSITION[1] - y
    ahead = math.cos(yaw) * dx + math.sin(yaw) * dy
    left = -math.sin(yaw) * dx + math.cos(yaw) * dy
    # Keep np.arctan2/np.hypot (and not math.*): they can differ in the last digit, and the
    # results of the main experiment were made with these
    distance = np.hypot(ahead, left)

    if distance > 1e-8:
        direction = [ahead / distance, left / distance]
    else:
        direction = [0.0, 0.0]

    # data.time starts at 0 in every evaluation, so the clock always starts at the same point
    phase = 2.0 * math.pi * config.CLOCK_FREQ * data.time

    extra = [
        math.sin(phase),
        math.cos(phase),
        *direction,
        min(distance / config.DIST_SCALE, 2.0),
        1.0,  # bias
    ]
    return np.concatenate([qpos[FREE_JOINT_SIZE:], extra])


def hinge_targets(data: mj.MjData, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    """One forward pass of the network. Outputs are in [-pi/2, pi/2], the hinge range."""
    hidden = np.tanh(sensor_inputs(data) @ w1)
    return np.tanh(hidden @ w2) * (np.pi / 2)
