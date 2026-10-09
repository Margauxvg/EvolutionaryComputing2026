# extras

Not part of the submission: pilots, checks and tools we used while building and tuning the EA.
Zip `group_28` without this folder for the hand-in.

| file | what |
|---|---|
| `watch.py` | watch a genotype (e.g. a `best.npy`) walk in the MuJoCo viewer |
| `pilot_experiments.py` | the pilot probes (duration, clock, qvel, nan, cache, determ, hidden, sigma) |
| `timing.py` | first timing measurements and budget estimate |
| `sphere_check.py` | runs the real EA on a sphere function, to test the EA without physics |
| `self_tests.py` | quick checks of the controller and the 1/5 rule |
| `docs/` | our notes from the pilots and planning |
| `results/` | results of the pilots and smoke tests (`pilot_experiments.py` writes here); not committed, like every `results/` folder |

The notes in `docs/` were written before the code was renamed:
`run.py` -> `main.py`, `ea.py` -> `evolve.py`, `simulate.py` -> `simulation.py`,
`controller.py` -> `brain.py`, `analyze.py` -> `plots.py` (+ `diagnostics.py`),
`experiments.py` -> `extras/pilot_experiments.py`, `test_sphere.py` -> `extras/sphere_check.py`,
`phase1_timing.py` -> `extras/timing.py`.
