# What each file actually does — A2, group 28

Written to be read next to the code, one section at a time. Every section ends with
**what's load-bearing and what's yours**, because the thing that makes code feel like
someone else's is not knowing which lines you're allowed to break.

---

## The through-line

There is one path through the whole project, and everything else is scaffolding around it:

```
a flat vector of 132 floats          ← the EA invents this
        ↓  controller.unpack
two weight matrices, w1 and w2
        ↓  controller.build_inputs + controller.forward   (every physics step)
8 hinge target angles
        ↓  simulate._make_callback                        (every physics step)
data.ctrl, moved a little towards those angles
        ↓  mujoco steps the physics 15 simulated seconds
the gecko ends up somewhere
        ↓  simulate.rollout
one number: how far it is from (2, 0)
```

That number is the fitness. Lower is better. Everything the EA does — selection, mutation,
the 1/5 rule — is machinery for producing better vectors of 132 floats. If you hold only
one picture of this project, hold that one.

**Where the 132 comes from:** the gecko has 8 actuated hinges (`model.nu = 8`). The network
sees 8 hinge angles + 6 extra inputs = 14 inputs, has 6 hidden units, and emits 8 outputs.
So `14 × 6 + 6 × 8 = 84 + 48 = 132`.

---

## `config.py` — the single source of truth

Six constants, and it's currently the emptiest and most important file you have.

The A1 pattern was *one module holds every number*, and the reason is not tidiness: the
brief asks for the experiment to be reproducible, and Methods has to state every parameter.
If a number lives in `config.py`, you can read Methods off the file. If it's buried in a
function default, you will misreport it.

| constant | what it is | where it came from |
|---|---|---|
| `SPAWN_POS` | where the gecko starts | template line 68 |
| `TARGET_POSITION` | where it should end up, 2 m along +x | template line 69 |
| `SIM_DURATION` | 15 simulated seconds per evaluation | template line 70 |
| `CONTROL_ALPHA` | 0.05, the DELTA step size | template's controller contract, line 124 |
| `WORST_FITNESS` | what a NaN individual scores | our decision, not the template's |
| `DEFAULT_SEED` | seed when none is given | A1 `config.py` had the same |

**Load-bearing:** `SPAWN_POS`, `TARGET_POSITION`, `SIM_DURATION`. The brief says to keep
body, world, `SIM_DURATION` and fitness identical across everything you compare — change
one of these mid-experiment and your runs stop being comparable to each other.

**Yours:** `CONTROL_ALPHA` and `WORST_FITNESS` are judgement calls. `WORST_FITNESS = 1e6`
is arbitrary — the only real requirement is "worse than any real distance", and a real
distance can't exceed a few metres. `100.0` would work identically and read less like a
magic number.

**Obviously missing:** every EA parameter. `POP_SIZE`, `SEEDS`, `NUM_GENERATIONS`,
`TOURNAMENT_SIZE`, `CROSSOVER_PROBABILITY`, `ELITISM_RATIO`, `SIGMA_*`. A1's `config.py`
had all of them and this one has none, because nobody has written `ea.py` yet.

---

## `simulate.py` — the only file that knows physics exists

**In experiment terms:** this is "run the robot and see how far it got". Nothing more.

The reason it's a separate file is that it's the only module importing `mujoco`. That means
`ea.py` can be tested against a fake fitness function — evolve towards a sphere function in
two seconds — instead of waiting on 0.168 s rollouts every time you want to check whether
the 1/5 rule is adapting correctly. When you're debugging the research question rather than
the robot, that matters.

### `build_world()` and `build_robot()`

Copied verbatim from the template, lines 77 and 91. Two lines of actual code. They exist as
functions rather than inline so there's exactly one place the body and world are decided,
and so `phase1_timing.py`'s claims stay true of what you're running.

### `get_core_position(data)`

Template line 206. Returns `data.qpos[0:3].copy()`.

Worth understanding rather than trusting: the gecko is spawned with a **free joint**, which
is MuJoCo's way of saying "this body can move and rotate freely in space". A free joint
occupies the first 7 slots of `qpos` — 3 for position, 4 for the orientation quaternion. So
`qpos[0:3]` *is* the core's world position, with no tracking or bookkeeping.

The `.copy()` is not defensive habit. `data.qpos` is a live view into MuJoCo's memory; without
it you'd hold a reference that silently changes on the next physics step, and your
"initial position" would equal your final position.

### `Simulator.__init__`

Template lines 250–274, wrapped in a class. Spawn, compile, make `MjData`, reset, read the
sizes off the compiled model.

The only reason this is a class rather than a function is the **compile-once** trick: the
template rebuilds the world for every evaluation, and `phase1_timing.py` measured that as
0.210 s against 0.168 s when reused. A class gives somewhere for `model` and `data` to live
between calls.

`self.output_size = model.nu` and `self.qpos_size = len(data.qpos)` are read from the
compiled model rather than hardcoded, per the template's warning at line 271 — change the
body and they change with it.

### `Simulator.rollout` — the important one

This is the whole file. Read it line by line at least once.

1. **Clear the callback, reset, forward.** `mj.set_mjcb_control(None)` first because MuJoCo's
   control callback is a *global variable in C* — not attached to a model, not attached to a
   simulator, one per process. If you don't clear it, whatever ran last is still attached.

2. **Read `initial_position`.** After the reset, never before. The long comment explains why
   and it's the one bug in this file that would actually cost you a day: `simple_runner`
   resets internally (`ariel/utils/runners.py:30`), so it looks safe to skip your own reset —
   but on the second call of a reused `Simulator`, `data` still holds the *previous* rollout's
   end state. You'd read that as your starting point. Finite, plausible, completely wrong.

3. **Build a fresh callback bound to this genotype**, attach, run, detach in a `finally`.

4. **Score.** `distance_to_target(final_position, target)` — imported from
   `ariel/simulation/tasks/targeted_locomotion.py:9`, not reimplemented, so Methods can cite
   the module rather than describing arithmetic.

### `Simulator._make_callback`

The function MuJoCo calls every physics step — roughly 7500 times per rollout at the default
2 ms timestep. It does three things:

**Asks the controller for actions.** `self.controller_fn(m, d, genotype)`.

**Checks for NaN.** Template line 126 warns that blown-up weights write NaN into `data.ctrl`
silently. The simulation continues, the robot does nothing coherent, and you get a
finite-looking fitness. So: flag it, write zeros, and return `WORST_FITNESS` at the end.
Returning `None` instead would crash `min(contestants, key=...)` in tournament selection
three days later, which is exactly the kind of bug that surfaces at 2am on the 8th.

**Applies DELTA.** `d.ctrl[:] += actions * CONTROL_ALPHA`, then clip.

That last line deserves more attention than it usually gets. The template offers two schemes
(lines 121–125): **DIRECT** commands the hinge angle straight, **DELTA** nudges it. DELTA is
smoother but it *accumulates*, which is why the clip isn't optional. It also means the
network isn't choosing an angle — it's choosing a direction to move the angle in. That's a
genuinely different control problem, and it's a Methods sentence.

### Why `controller_fn` is a parameter and not an import

`simulate.py` never imports `controller.py`. You pass the function in. That's what let
`simulate.py` be written and tested before `controller.py` existed, and it's what lets the
walk smoke test pass in a hand-written controller without touching either file.

**Load-bearing:** the reset ordering in `rollout`, clearing the global callback, the
`.copy()` in `get_core_position`, reading sizes from the model.

**Yours:** DELTA vs DIRECT (a real experimental choice, currently DELTA because the game plan
said so — but nobody has compared them). `WORST_FITNESS` handling. Whether `rollout` and
`evaluate` are two methods or one. The `get_simulator` cache is only needed once you
parallelise; delete it if the runner ends up structured differently.

---

## `controller.py` — the robot's brain, and the file your RQ depends on most

**In experiment terms:** given the robot's current state, what angle should each hinge aim
for? The EA's 132 numbers are this file's weights.

The brief says a controller "with no signal telling it where the target is, or nothing to
drive rhythmic movement with, has very little to work with". This file is the answer to that
sentence, and if it's wrong, **no mutation scheme finds anything and the research question
becomes untestable** — which is why the plan made it Stage 1's gate.

### The input vector (14 values)

| what | how many | why it's there |
|---|---|---|
| `qpos[7:]` — hinge angles | 8 | proprioception: where are my legs right now |
| `sin(wt)`, `cos(wt)` | 2 | the clock — something to drive a rhythm |
| target direction, robot frame | 2 | which way to steer |
| target distance / 2, clipped | 1 | how far to go |
| `1.0` | 1 | bias |

Three design decisions in there worth owning:

**The free joint is dropped.** `qpos[7:]` skips the first 7 values — the robot's world
position and orientation. That's deliberate: a network that knows its absolute world
coordinates could learn "walk to x=2" by memorising coordinates rather than by steering, and
it wouldn't transfer if you moved the target. The target-relative inputs carry that
information in a more useful form.

**The target vector is rotated into the robot's frame.** The quaternion-to-yaw maths in
`build_inputs` converts "the target is north-east of me in world coordinates" into "the
target is ahead and slightly left of *me*". Without that rotation the network would have to
learn its own heading compensation, which is a much harder function.

**Bias as a constant input rather than a separate vector.** Appending `1.0` to the inputs
makes the first column of `w1` act as a bias. Fewer moving parts in the genotype, one less
thing to reshape.

### `_check_layout`

Asserts `model.nq == 7 + model.nu`. This is the assumption that `qpos[7:]` gives exactly the
hinge angles. If someone swaps the body for one with a different joint structure, this raises
instead of silently feeding the network garbage.

### `unpack`

The flat-vector-to-matrices reshape, with a shape check. This is the interface `ea.py`
depends on and the one that breaks silently if the input vector changes — hence the explicit
`ValueError` rather than letting numpy broadcast something plausible.

### `act`

What `simulate.py` calls. Pure: reads, computes, returns. It never writes `data.ctrl` — that's
`simulate.py`'s job, so there's exactly one place deciding how commands reach the motors.

**Load-bearing:** `input_size` and `genotype_length` must agree with `unpack`, and all three
must agree with what `ea.py` generates. The `_check_layout` assumption.

**Yours, and these are real experimental choices nobody has justified yet:**

- `HIDDEN_SIZE = 6` — inherited from the template. Why 6? Nobody knows. It sets your search
  dimensionality.
- `CLOCK_FREQ = 1.0` Hz — one gait cycle per second. A gecko might want 2, or 0.5. The
  comment says "choose in pilots" and nobody has.
- `DIST_SCALE = 2.0` — normalises distance to roughly [0,1] at spawn. Arbitrary but harmless.
- `tanh` twice. Could be one layer, could be three, could be ReLU.
- Dropping `qvel` entirely. The template mentions it; you're not using it.

---

## `phase1_timing.py` — the feasibility probe

**In experiment terms:** "before we design anything, how expensive is one evaluation and can
we trust it?" It answers four questions and prints a budget table.

It is deliberately **standalone** — it imports nothing from the other files, reimplements the
model build and the rollout inline, and hardcodes its own constants. That's the right call for
a probe: it can't be broken by a later refactor, and it measures what it says it measures.

`time_rebuild_each_time` vs `time_reuse_model` is the 1.25× comparison. `check_determinism`
evaluates one weight vector twice on the same model and once on a fresh one — the second
check rules out state leaking through the model object, which is a more careful test than
most people would write.

**The thing to know about this file:** it tests a **different controller** from the one you
now run. Line 58 is `d.ctrl[:] = nn_controller(...)` — that's DIRECT, not DELTA — and its
network takes the full bare `qpos`, not the 14-input vector. So the 0.168 s and the
determinism result were measured on a code path that no longer exists.

Probably still true. Not yet verified.

---

## `watch.py` — the debugging tool nobody planned, and it's the best thing here

**In experiment terms:** "show me the robot actually moving".

It matters more than it looks. Everything else in this project reduces a robot to one number,
and one number cannot tell you the difference between *walking slowly towards the target* and
*flailing in place while drifting in the right direction*. `watch.py` is the only way anyone
sees the difference.

The important design point: it goes through `Simulator._make_callback`, so what you watch is
*exactly* what gets scored — same DELTA application, same clipping, same NaN guard. A viewer
that re-implemented the control loop could show you a robot that walks beautifully while the
EA scores something else entirely.

`quick_evolve` is a (5+10) evolution strategy — 5 parents, 10 children, keep the best 5 —
with fixed σ = 0.3 and no logging. It is explicitly *not* the experiment; it exists to get
something moving so you have something to look at.

**Yours:** all of it. This is a tool, not an artefact. Nothing in the report depends on it.

---

## `report/preliminary_measurements.tex`

A Methods subsection, drafted before results exist — which is exactly the sequencing the plan
depends on, and it's ahead of schedule.

It makes three arguments, and the second one is the strongest thing in the project so far:
determinism isn't a convenience, it's a *methodological* result. The 1/5 rule adapts on the
fraction of offspring beating their parent, and under a noisy fitness some of those wins are
measurement error biasing the signal. Most groups won't have checked.

**Every `\TODO{}` in it is a number that needs re-deriving**, and at least one is wrong — see
the next section.

---

## A quick way to make this yours

Pick one and change it today. Ownership comes from having broken something and fixed it.

1. **Change `CLOCK_FREQ` to 2.0 and watch the robot.** One number, immediate visible effect,
   and it turns "6 was in the template" into "we tried 1 and 2 and picked one".
2. **Replace `WORST_FITNESS = 1e6` with `100.0`** and satisfy yourself nothing breaks.
3. **Delete the `get_simulator` cache.** You're not parallelising yet. Put it back when
   `run.py` needs it, and you'll know exactly why it exists.
4. **Add `qvel` to the inputs** and see whether the quick search does better. That's a real
   experiment, it's 3 lines, and it's a Methods sentence either way.

None of these are improvements. They're all ways of finding out what the code does by
changing it.
