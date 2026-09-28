"""Watch the configured body driven by controller.py in the MuJoCo viewer.

Run from the project root:
    uv run assignments/assignment_2/group_28/watch.py            # mode from config
    uv run assignments/assignment_2/group_28/watch.py delta      # force DELTA

Close the window to end. Uses random weights (seed 0) until the EA exists.
"""

import sys

import mujoco as mj
import numpy as np
from mujoco import viewer

from ariel.simulation.environments import SimpleFlatWorld

from config import BUILD_BODY, CONTROL_MODE, SPAWN_POSITION
from controller import genotype_length, make_controller

mode = sys.argv[1] if len(sys.argv) > 1 else CONTROL_MODE

mj.set_mjcb_control(None)
world = SimpleFlatWorld()
world.spawn(
    BUILD_BODY().spec,
    position=list(SPAWN_POSITION),
    correct_collision_with_floor=True,
)
model = world.spec.compile()
data = mj.MjData(model)
mj.mj_resetData(model, data)

genotype = np.random.default_rng(0).normal(scale=0.5, size=genotype_length(model))
mj.set_mjcb_control(make_controller(genotype, model, mode=mode))

print(f"Opening viewer: {BUILD_BODY.__name__}, {mode}. Close the window to exit.")
viewer.launch(model=model, data=data)
mj.set_mjcb_control(None)