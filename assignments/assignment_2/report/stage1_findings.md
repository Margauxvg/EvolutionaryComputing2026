# Stage 1 — what we measured, why, and what it fixes about the design

> **BODY CHANGED 1 OCTOBER — read this before citing any number below.**
>
> Sections 1–8 were measured on `prebuilt_robots.gecko.gecko()`, the body `A2_template_2026.py`
> imports: **8 hinges, 14 inputs, 132 weights**. The brief requires a body from the John Set, and
> `report/body_pilot.md` chose `john_set.gecko()`: **6 hinges, 12 inputs, 108 weights**. ariel
> ships two different functions called `gecko()` in two different modules. `simulate.py` now uses
> the John Set one.
>
> **The arguments survive the switch; the numbers do not.** Still valid as written: section 9
> (crossover — pure arithmetic about sample counts, body-independent), the 5-vs-20-seed power
> reasoning, and determinism as a property of the engine rather than of a body.
>
> **Already re-measured on the John Set gecko (1 Oct):** genotype length and the HIDDEN_SIZE
> table (§1, §10), determinism at all three levels (§2), the simulation length (§5).
>
> **Seconds per evaluation (§3), measured 1 Oct on the laptop:** 0.345 s for a single worker;
> 19.6 evaluations/s with 16 workers (`group_28/parallelisation_plan.md`, speed test).
>
> **Still to re-measure before Methods cites them:** the walk gate (§4) and the 0.107 m
> seed-to-seed sd behind `SEEDS = 20` (§6). Both fall out of the pilot.
>
> Each stale section is flagged inline below.

Working record, not report prose. Every section is *what we did → why → how → result → what it
changes about the experiment*, so the design decisions in Methods can be traced back to the
measurement that forced them.

**Group 28 · Assignment 2 · 25 and 29 September 2026**

---

## Summary

| | Result | Consequence for the design |
|---|---|---|
| Is the search space navigable? | **Yes** — 2.0 → 1.621 m, monotone | The RQ is testable. Build the EA. |
| Is evaluation deterministic? | **Yes** — bit-identical, through the real control path | The 1/5 success signal is noise-free. One rollout per individual. |
| Genotype size | **132 weights** | Search dimensionality. Supersedes two earlier figures. |
| Seconds per evaluation | **0.454 s**, and the cost is **ours, not the machine's** | Budget is 12.7 h — but recoverable to ~4.7 h by optimising `build_inputs`. |
| Can the robot reach the target? | **No** — it stops progressing at ~25 s, 1.45 m short | Keep `SIM_DURATION = 15`. Fitness measures progress, not arrival. |
| Landscape shape | **Long neutral plateaus** | Sharpens the hypothesis — see §4. |
| Gait frequency | **3 Hz beats 1 Hz** by 5× the noise band, on one seed | `CLOCK_FREQ` needs a proper sweep before it goes in Methods. |

The Stage 1 gate is **closed**.

---

## 1. Problem size — re-measured on the John Set gecko, 1 Oct

**What we did.** Ran `controller.py`'s self-test, which builds the compiled model and reads the
network's input and output sizes off it.

**Why.** The genotype length is the dimensionality of the search. It determines what σ means,
how long the EA needs, and it is a number Methods has to state. Two earlier figures were in
circulation and neither matched the code being run.

**How.** `input_size` and `genotype_length` in `controller.py` read `model.nu` from the
compiled model rather than hardcoding it, so they follow whatever body `simulate.build_robot()`
returns.

**Result.** `experiments.py hidden`, on `john_set.gecko()`:

```
hinges (model.nu)   : 6
network inputs      : 12      = 6 hinge angles + 6 extras
genotype length     : 108     = 12 x 6 + 6 x 6        (18h at HIDDEN_SIZE = h)
```

Superseded: the template's `prebuilt_robots.gecko.gecko()` gave 8 hinges, 14 inputs, 132.

The six extras are `sin(wt)`, `cos(wt)`, target direction (2 values, in the robot's own frame),
scaled target distance, and a constant bias.

**What it changes.**

- The Methods table currently says `\TODO{168}`. **It is 108.** The 168 came from an early
  estimate assuming ~20 inputs; 132 was the template's 8-hinge body.
- The 138 figure from 25 September was the template's bare-`qpos` controller (15 inputs, no
  clock, no target signal) and no longer describes anything we run.
- The row labelled "Controller inputs (`len(data.qpos)`)" is mislabelled. The controller feeds
  `qpos[7:]` — hinge angles only — deliberately excluding the free joint's world pose, so the
  network steers by a target-relative vector rather than memorising world coordinates. That
  exclusion is a design decision and belongs in Methods as one.

---

## 2. Determinism — closed at three levels, on both bodies

**What we did.** Evaluated one fixed genotype repeatedly and compared the fitnesses for **exact**
equality, at three levels: a reused `Simulator`, a freshly compiled one in the same process, and
a spawned worker process.

**Why.** This is the most load-bearing property for the research question. The 1/5 rule adapts
sigma from the fraction of offspring that beat their parent. Under a noisy fitness, some of those
"improvements" are measurement error, and the signal driving the controller is partly noise.
Determinism removes that concern, makes repeated rollouts per individual unnecessary, and makes
every run exactly reproducible from its seed.

The third level is not a formality. `run.py` has to evaluate in a process pool, because
`mj.set_mjcb_control` is a global and cannot be shared across threads, so every worker compiles
its own model. If a worker disagreed with the parent even in the last bits, two seeds that ran on
different workers would not be strictly comparable and the paired statistics would rest on sand.
Windows has only `spawn`, so `spawn` is what was tested.

**How.** `experiments.py determ`, 4 repeats per level, seed 1. Compared with `==`, not a
tolerance: a 1e-16 difference is still a difference, and Methods should not claim bit-identical
if it is only nearly so.

**Result.** On the John Set gecko, 1 October:

```
genotype length 108, seed 1, 4 repeats per level
reference fitness 2.1512138839134756

level                         identical     max abs diff
1 same Simulator                    yes        0.000e+00
2 fresh Simulator                   yes        0.000e+00
3 spawned process                   yes        0.000e+00
```

Exact zeros, not small numbers. The same check on the template's 8-hinge body on 30 September
also gave three exact zeros (reference 1.9958052330735918, matching `controller.py`'s self-test
through a different entry point). Determinism holding on two different bodies is good evidence
it is a property of the engine and the control path, not of a particular morphology.

A side observation, not a finding: a random controller on the John Set gecko ends 2.151 m from
the target - *further* than the 2.0 m it started at - where the 8-hinge body's random controller
barely moved (1.996). The John Set gecko moves more under random weights. Consistent with
`body_pilot.md`, where it improved steadily on every seed.

**What it changes.**

- Methods can state, without hedging, that evaluation is deterministic and bit-identical across
  model rebuilds and process boundaries. That is the strongest single claim in the section.
- Budget stays at one rollout per individual rather than k.
- `run.py` is cleared to use a spawn-based process pool. This was the last correctness question
  standing in front of the runner.
- The open item carried since 25 September — whether state leaks through the model object — is
  now closed against the current code path, not an older one.

---

## 3. Evaluation cost — resolved

> **Stale after the 1 Oct body change — re-measure: seconds per evaluation.**

**What we did.** Timed single evaluations in two places, and separately timed the model-compile
step on both machines.

**Why.** Everything downstream — population size, generation count, number of seeds — is sized
from this number, and the decision to spend compute on 20 seeds rather than 5 was justified by
it. The laptop measured 2.7× slower than the 15-core machine, and it mattered whether that was
*our controller* (fixable) or *the machine* (not).

**How.** `experiments.py cache` times N evaluations reusing one `Simulator` against building a
new one each time. The difference between those two columns is the cost of compiling the model,
which is pure CPU work independent of the controller.

**Result.**

| | rollout + compile | rollout only | compile alone | stepping only |
|---|---|---|---|---|
| 25 Sep, 15-core, old controller | 0.210 s | 0.168 s | **0.042 s** | 0.126 s |
| 29 Sep, laptop, new controller | 0.462 s | 0.419 s | **0.043 s** | 0.376 s |

**Compilation takes the same time on both machines.** If the laptop were 2.7× slower overall,
compilation would be 2.7× slower too. It is not. So the machines are comparable, and the gap is
entirely in the stepping — **3.0× slower, caused by our controller.**

`build_inputs` runs a quaternion-to-yaw conversion, a frame rotation, a `hypot` and a fresh
`np.concatenate` on every one of roughly 7 500 physics steps per rollout. The old controller did
a single matrix multiply. The per-step allocation is the likely bulk of it.

**What it changes.** For 3 configurations × 20 seeds × population 50 × 300 generations
(≈903 000 evaluations) on 15 cores at ~60% parallel efficiency:

| scenario | s/eval | wall clock |
|---|---|---|
| as measured today | 0.454 | **12.7 h** |
| `build_inputs` optimised back toward the old cost | ~0.17 | **4.7 h** |
| today's cost at `SIM_DURATION = 30` | 0.91 | 25.3 h |
| optimised, at `SIM_DURATION = 30` | ~0.34 | 9.5 h |

The design survives either way and the 20-seed decision stands. But at 12.7 h the margin is
gone: the final runs launch Saturday 3 October and there is no slot where all three of us are
available until Thursday 8 October.

**Optimising the input assembly is worth roughly eight hours of wall clock**, and it is the only
change that makes a longer `SIM_DURATION` affordable (see §5).

---

## 4. The walk gate

> **Stale after the 1 Oct body change — re-measure: all distances.**

**What we did.** Ran a throwaway (5 + 10) evolution strategy for 20 generations and watched
whether the distance to target fell.

**Why.** This was Stage 1's gate. The brief warns that a controller "with no signal telling it
where the target is, or nothing to drive rhythmic movement with, has very little to work with".
If no controller in this space can walk, the landscape is flat, no mutation scheme finds
anything, and **the research question is untestable** — a failure that would only surface after
the EA was built.

**How.** `watch.py --quick-evolve 20`: 5 parents, 10 children per generation, fixed σ = 0.3,
mutation only, one seed, 205 evaluations in 95 s. Best genotype saved to `quick_best.npy`.

**Result.**

| Generation | Best distance (m) | |
|---|---|---|
| start | 2.000 | spawn distance |
| 1 | 1.904 | ↓ |
| 2 | 1.904 | |
| 3 | 1.876 | ↓ |
| 4 | 1.829 | ↓ |
| 5–11 | 1.829 | **seven generations without improvement** |
| 12 | 1.820 | ↓ |
| 13–14 | 1.820 | |
| 15 | 1.769 | ↓ |
| 16–18 | 1.769 | |
| 19 | 1.621 | ↓ largest single step, 0.148 m |
| 20 | 1.621 | |

Total improvement 0.379 m; six improving generations out of twenty.

**What it changes.** Three things, and the second is the important one.

**The gate is closed — build the EA.** Selection improves the controller monotonically. Nothing
about the representation, the input vector or the fitness prevents search from working.

**The hypothesis gets sharper.** Seven consecutive generations of no improvement at a *fixed*
step size says this landscape has long neutral plateaus. That bears directly on the research
question. Rechenberg's rule shrinks σ when success falls below 1/5 — the right response when
steps are *overshooting* a smooth optimum, and the wrong one on a *flat* plateau, where smaller
steps find no improvement either and the rule shrinks itself into stagnation. A1 saw exactly that
failure and attributed it to applying the rule to a discrete probability; this run suggests the
plateau structure may be the real cause, and that it will still be there in continuous space.

So the Introduction can ask something better than "does the 1/5 rule help". It can ask **whether
the plateaus in this landscape are the kind the rule handles**, and the answer is visible by
plotting σ against the fitness curve rather than only comparing endpoints. That also means
**σ must be logged per generation from generation 0** — without it the result is a number without
a mechanism.

**The pilot must run long.** The largest single improvement was the *last* generation. Twenty is
nowhere near convergence, consistent with the brief's warning. Fix the generation count from a
long pilot curve, not from a guess.

---

## 5. How far can it actually get? (re-measured on the John Set gecko, 1 Oct)

**Re-measured.** We took the best controller from the sigma sweep (static σ=0.5, seed 3: 0.855 m
at 15 s) and replayed it for up to 60 s. Command:
`experiments.py duration --genotype results/sigma_best_pilot1/static_sigma0.5_seed3.npy`.

| sim s | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 45 | 60 |
|---|---|---|---|---|---|---|---|---|---|
| distance m | 1.582 | 1.222 | **0.855** | 0.502 | 0.223 | **0.009** | 0.015 | 0.024 | 0.093 |

- **It walks steadily** at about 7–8 cm/s in every 5 s slice up to 25 s.
- **It reaches the target** at about 30 s, within 1 cm.
- **Then it stays there.** It's within 10 cm for the remaining 30 s and doesn't overshoot or walk
  off. That's good evidence the target-direction and distance inputs are actually used.

Very different from the old 8-hinge body, whose best probe controller stopped at ~25 s, 1.45 m
short.

**Ignore the script's verdict (since fixed).** It printed *"sustains 3.2 cm/s over 60 s → not reachable"*. That
average includes 30 s of standing at the target. The walking speed is ~7 cm/s (7.1 cm/s averaged up to 25 s). Reaching the
target in 15 s would need ~13 cm/s, about twice as fast as this controller after 1,230
evaluations.

**What it changes: keep SIM_DURATION = 15 s, for a stronger reason than cost.** At 15 s even the
best controller is still 0.86 m away, so fitness can't saturate at 0. Differences between
variants stay visible through the whole run. With 30 s, or a closer target, the best runs would
reach the target and pile up near 0 (a floor effect), hiding exactly the differences we want to
measure, and every run would cost twice as much.

Re-check after the pilot (μ = 50, 150 generations, far more evaluations): if the pilot's best
distance at 15 s gets close to 0 (say < 0.3 m), the floor effect becomes a real risk.

*The 29 Sep measurements below are kept for the record. They describe the template's 8-hinge
gecko.*

**What we did.** Scored `quick_best.npy` at durations from 5 to 60 seconds.

**Why.** Because evaluation is deterministic, a 15 s rollout is a strict prefix of a 60 s one, so
scoring one genotype at increasing durations traces its whole trajectory. Each row is literally
"what fitness would be if `SIM_DURATION` were this". The question behind it: is 15 s truncating
progress that is still happening, and can the target be reached at all?

**How.** `experiments.py duration`.

**Result.**

| sim s | distance m | travelled m | cm/s in that slice |
|---|---|---|---|
| 5 | 1.829 | 0.171 | 3.43 |
| 10 | 1.736 | 0.264 | 1.85 |
| 15 | 1.621 | 0.379 | 2.30 |
| 20 | 1.525 | 0.475 | 1.92 |
| 25 | **1.446** | **0.554** | 1.59 |
| 30 | 1.450 | 0.550 | −0.09 |
| 40 | 1.473 | 0.527 | −0.33 |
| 60 | 1.492 | 0.508 | −0.06 |

It makes real progress for about 25 seconds, reaching 0.554 m travelled, then settles and creeps
back roughly 4 cm over the following 30. Not a lurch-and-stall, and not a sustained gait — a gait
that reaches some configuration and stops. Reaching the target inside 15 s would need 13.3 cm/s;
this controller sustains 0.8 cm/s over 60 s.

**What it changes.**

**Keep `SIM_DURATION = 15`.** Going to 25–30 s would capture more of what this controller does,
but at today's cost it takes the budget to 25 h, launching Saturday with nobody available Monday
to Wednesday. If `build_inputs` is optimised first (§3), 30 s becomes affordable at ~9.5 h and
this is worth revisiting.

**Two sentences for Methods.** Fitness measures **progress towards the target, not arrival** —
`2.0 − distance travelled`, which is a perfectly good monotone gradient whether or not the target
is reachable. And all configurations truncate at the same 15 s, so the comparison stays fair even
though the window is short.

**Caveat on n = 1.** This is one genotype from a 20-generation toy search. A properly evolved
controller may well sustain movement past 25 s. Worth re-running this probe on the best
individual from the real pilot before the generation count is fixed.

---

## 6. Gait frequency — resolved, as a null

> **Stale after the 1 Oct body change — re-measure: the 0.107 m sd; the null itself is unlikely to flip.**

**What we did.** Ran the same 20-generation search at six values of `CLOCK_FREQ` (1–6 Hz) across
three seeds. Eighteen runs.

**Why.** `CLOCK_FREQ` sets how fast the `sin`/`cos` inputs oscillate — roughly how many gait cycles
per second the network is offered. 1.0 Hz was inherited from the template; `controller.py` itself
says "choose in pilots". An unexplained constant in Methods is a weakness. A first single-seed
probe had suggested 3 Hz was 0.12 m better than 1 Hz, which is the finding this sweep was built
to check.

**How.** `experiments.py clock --freqs 1 2 3 4 5 6 --seeds 1 2 3`. `controller.CLOCK_FREQ` is
patched at runtime, so `config.py` is untouched and the only difference between runs is the
frequency. The seed fixes the initial population, so each frequency sees the same three starting
populations — a paired design.

**Result.**

| Hz | mean final m | sd | best seed | worst seed |
|---|---|---|---|---|
| 1.0 | 1.601 | 0.085 | 1.488 | 1.695 |
| 2.0 | 1.586 | 0.105 | 1.440 | 1.682 |
| 3.0 | 1.617 | 0.091 | 1.491 | 1.705 |
| 4.0 | 1.695 | 0.069 | 1.609 | 1.778 |
| 5.0 | 1.515 | 0.068 | 1.420 | 1.578 |
| 6.0 | 1.561 | 0.081 | 1.471 | 1.667 |

The single-seed 3 Hz advantage did not survive. Two-way decomposition (frequency × seed, no
replication):

| source | SS | df | MS | F |
|---|---|---|---|---|
| frequency | 0.0542 | 5 | 0.01084 | **0.95** |
| seed | 0.0133 | 2 | 0.00664 | 0.58 |
| residual | 0.1147 | 10 | 0.01147 | |

**F < 1.** The variation between frequencies is smaller than what noise alone would produce. A
paired permutation test on the spread of frequency means (200 000 shuffles of the frequency label
within each seed) gives **p = 0.378**.

Two numbers that look like evidence and are not. Frequency "explains 30% of total variance" — but
frequency has 5 of the 15 degrees of freedom, so 33% is the chance expectation; 30% is slightly
*less* than nothing. And 5 Hz has the lowest mean — but the lowest of six noisy means is the
lowest of six noisy means. With three seeds per group, the smallest two-sided Mann–Whitney p
attainable is 0.100, reached only under complete separation, so no pairwise claim is available
from this design at any effect size.

**What it changes.**

`CLOCK_FREQ` stays at **1.0**, and the PROVISIONAL marker comes off `config.py`. Not because 1 Hz
won — nothing won — but because the sweep's real job was to check that no choice is catastrophic,
and none is. All eighteen runs land in 1.42–1.78 m. Frequency is a nuisance parameter, not the
research question; it only has to be held identical across static, adaptive and baseline, which
`config.py` already guarantees. Methods gets one honest sentence instead of an unexplained
constant, and we spend no more compute on it.

The claim is narrow and should be written that way: no detectable effect **on 20-generation early
search progress**. At this point the gecko is not walking, it is flailing; the best run has closed
0.55 m of 2.0 m. A clock frequency could plausibly matter for a formed gait and not for random
flailing. We are not testing that, and should not imply we are.

**The residual is worth more than the null.** Residual MS gives a seed-to-seed standard deviation
of **0.107 m** for the 20-generation outcome — our own measurement of the noise this experiment
has to see through:

| seeds per variant | smallest difference detectable at 80% power |
|---|---|
| 5 (the brief's minimum) | 0.190 m |
| 10 | 0.134 m |
| **20 (`config.SEEDS`)** | **0.095 m** |
| 30 | 0.077 m |

The brief's five seeds could not resolve a difference smaller than 0.19 m — about half the total
progress a 20-generation run makes. This replaces the a-priori argument for `SEEDS = 20` in
`config.py` with a measured one, and it is a Methods paragraph rather than an assertion. Caveat:
measured at 20 generations; the spread at the pilot's plateau length will differ and should be
re-estimated from the pilot.

**And it confirms the plateau.** Across all eighteen runs, a mean of 4.6 of 20 generations produced
any improvement at all — **77% of generations are flat**, range 2–8. Section 4 saw seven
consecutive flat generations in one run; this says that was typical, not a bad draw. That is the
landscape the 1/5 rule will be asked to handle, and the reason `sigma` must be logged from
generation 0.

---

## 7. Joint velocities as extra inputs

**What we did.** Ran the same search with and without `data.qvel` appended to the input vector.

**Why.** The template mentions `qvel` and we are not using it. Velocity feedback is the usual way
a gait controller distinguishes "leg moving forward" from "leg moving back" at the same angle, so
there is a real mechanism to expect a difference.

**How.** `experiments.py qvel`, using an alternative `act` defined in the probe rather than
editing `controller.py` — `simulate.Simulator` takes the controller as a parameter, which is what
that injection was for.

**Result.**

| inputs | genotype | initial best m | final best m | improvement m |
|---|---|---|---|---|
| hinges + extras | 132 | 1.969 | 1.621 | 0.348 |
| hinges + extras + qvel | 216 | **1.776** | **1.546** | 0.230 |

**What it changes.** Read the *initial* column, not the improvement column. The qvel variant
finished better but improved less, because its random starting population was already 0.19 m
ahead. So the advantage may lie entirely in initialisation rather than in search: velocity
feedback makes random controllers twitch more, which moves them further by accident.

Final fitness is the metric that matters and qvel wins on it — but from one seed, with 84 extra
dimensions searched on the same budget. Same treatment as `CLOCK_FREQ`: worth three seeds before
it becomes a decision, and the confound has to be stated if it goes in the report.

---

## 8. The NaN guard

**What we did.** Fed the simulator four deliberately hostile genotypes.

**Why.** The template warns that blown-up weights write NaN into `data.ctrl` silently, the
simulation carries on, and the run returns a plausible-looking fitness. We wanted to know that the
guard fires, and what it protects against — before writing a Methods sentence claiming it protects
against runaway mutation.

**How.** `experiments.py nan`.

**Result.**

| genotype | fitness | guard fired |
|---|---|---|
| normal weights | 1.996 | no |
| huge finite weights (1e6) | **2.048** | **no** |
| one NaN in the genotype | `WORST_FITNESS` | yes |
| one inf in the genotype | `WORST_FITNESS` | yes |

**What it changes.**

**The guard does not protect against large weights.** `tanh` saturates, so weights of 1e6 produce
perfectly finite outputs. The guard fires on genuine `inf` or `nan` — arithmetic overflow, or a
mutation that produced a non-finite value directly. Do not claim otherwise in Methods.

**Saturated controllers score worse than standing still.** 1e6 weights gave 2.048 against a spawn
distance of 2.0 — the robot was pushed backwards. Useful to recognise during a real run: a fitness
above 2.0 is saturation, not a bug.

**The sentinel value is arbitrary and interchangeable.** Swapping `WORST_FITNESS` to 100.0 changed
nothing downstream; `min()` over a mixed list still selects a real individual. The only requirement
is "worse than any achievable distance", and distances cannot exceed a few metres. 1e6 reads as a
magic number; 100.0 would do.

**One nuisance to fix before the real runs.** The `inf` case emitted a numpy `RuntimeWarning` from
`controller.py:163`. Harmless once, but if `inf` appears mid-run it will emit thousands of lines
and bury the run output. Wrap the forward pass in `np.errstate` or add a warnings filter.

---

### 8b. Physics instability: a hang, found and fixed (1 Oct)

**What happened.** The speed test stopped partway through run 13 with one Python process still
busy. MuJoCo had written to `MUJOCO_LOG.TXT`:
*"Nan, Inf or huge value in QACC at DOF 0. The simulation is unstable. Time = 12.0820."*

**Why it hung.** One controller drove the physics unstable at 12.08 s. By default MuJoCo then
resets the simulation, which puts `data.time` back to 0. ariel's `simple_runner` steps until
`data.time` reaches the duration. The controller is deterministic, so it replays the same motion,
goes unstable at 12.08 s again, resets again, and so on forever. The NaN guard never fired,
because the reset happened before the controller saw a bad value.

**How rare.** It didn't occur once in the sigma sweep's 22,140 evaluations, and the same seed ran
fine on Linux. So it depends on last-digit differences between machines. But one hang blocks a
worker forever, and with hundreds of thousands of evaluations in the final experiment it would
very likely happen at least once.

**The fix (`simulate.py`).** Two changes:
- The automatic reset is switched off when the model is compiled
  (`mjDSBL_AUTORESET`). An unstable state now turns into NaN, the existing NaN guard in the
  control callback catches it, and the controller gets `WORST_FITNESS`, like any failed one.
- When that guard fires, the callback puts back a finite state and moves the clock to the end,
  so the rollout stops after its current batch of steps instead of stepping a NaN state for the
  rest of the 15 s.

**Checked** in a copy of the setup (same ariel source, MuJoCo 3.8.0):
- A forced instability, which hung forever before the fix, now ends in under 0.1 s with fitness
  100 m.
- 20 normal genotypes give exactly the same fitness with the old and new behaviour (max
  difference 0.0), and the seed-1 reference is unchanged. Determinism and all earlier results
  stand.
- A `Simulator` that has just evaluated an unstable controller evaluates the next one normally.
- **Confirmed in a real run on Windows:** in the speed test, the same seed-13 controller went
  unstable at exactly t = 12.0820 s in all five worker settings (2, 4, 8, 12, 16). It's logged
  five times in `MUJOCO_LOG.TXT`, and every setting finished. The fix works, and the instability
  is reproducible regardless of how many workers run.
- **Confirmed on the Windows laptop after the fix:** `experiments.py determ` gives three exact
  zeros and the same reference fitness as before the fix, `2.1512138839134756`.

**Side effect.** In the single step where the physics goes bad, MuJoCo prints about 1,900
`mesh_support could not find support vertex` warnings, to the terminal and to `MUJOCO_LOG.TXT`.
That's harmless and only happens on these rare evaluations. It's also a useful signal: if
`MUJOCO_LOG.TXT` grows during a run, some controllers went unstable. They're counted in the
`num_nan` column ("controllers that failed").

**Methods** now says a controller gets 100 m if it produces invalid outputs *or makes the
simulation unstable*.

---

## 9. Design decision: no crossover

**What we did.** Removed crossover. Mutation is now the EA's only variation operator.
`reproduction` selects parents and clones them; `mutation.mutate` supplies all the variation.

**Why.** Because crossover was eating the measurement the research question runs on. An
improvement over a parent cannot be attributed to mutation if crossover also touched the child, so
recombined children have to be excluded from the 1/5 success rate - A1 made the same exclusion.
At `CROSSOVER_PROBABILITY = 0.7` that discarded ~70% of the evidence.

**How.** Arithmetic, not a run. Population 50 produces 50 offspring per generation, pooled over a
5-generation window.

| p_cross | scorable/gen | per window | sd of the measured rate at 0.2 |
|---|---|---|---|
| 0.7 | 15 | 75 | 0.046 |
| 0.0 | 50 | 250 | **0.025** |

The rule's only decision is whether the true rate sits above or below 0.2. How often it gets that
right:

| true rate | p_c = 0.7 | p_c = 0.0 |
|---|---|---|
| 0.15 | 89% | 99% |
| 0.25 | 84% | 97% |
| 0.30 | 97% | 100% |

At a true rate of 0.25 the old design sent sigma the wrong way one generation in six, purely from
sampling noise.

**The alternative we rejected.** Keep crossover and make recombined children scorable by also
evaluating the pre-mutation intermediate, giving each one its own attributable baseline. That is
one extra evaluation per crossed child: **+70% on the compute budget**. Not affordable.

**What it changes.**

`CROSSOVER_PROBABILITY` is gone from `config.py`, and with it an unjustified constant we would
have had to defend in Methods. The `crossed` tag and the exclusion rule are gone from `ea.py` and
`mutation.py` - a net deletion of around twenty lines.

It also removes a reproducibility trap. `Crossover.uniform` was the only thing drawing from
ariel's package-level RNG, so `ariel.ec.set_seed()` had to be called or all 20 "independent" seeds
would have recombined identically. Nothing we call now reads global random state.

And it tightens the framing. Rechenberg derived the 1/5 rule to control the step size of
Gaussian mutation; sigma here is exactly that. A1 applied it to a mutation probability inside a
GA, its own report concluded "Our results call this analogy into question rather than confirm it." and the A1 marker deducted for
the step-size/probability conflation in the Introduction. A2 removes that conflation.

Be careful not to overclaim in the other direction. The rule was derived for the **(1+1)-ES** —
one parent, one child. This EA has a population of 50, tournament selection and elitism. So the
honest phrasing is *the rule's native parameter inside a population-based extension*, not "the
rule's native setting". Population-based use of success-based step-size control is common, but it
is an extension, and the success definition (each child against its own parent) is our choice,
which Methods must state as one.

**Signed off.** Confirmed with the course staff on 30 September that dropping crossover is
acceptable for this assignment. That closes the one external risk in this decision - the argument
below stood on its own, but a brief that mandated crossover would have overridden it.

**What it costs, honestly.** Recombination genuinely helps on many problems, and a mutation-only
search of a 132-dimensional weight vector may reach a worse absolute fitness than one with
crossover would. That is a real cost and Methods should say so. It does not threaten the result:
the research question is about the *difference* between fixed and adapted sigma, and a weaker EA
weakens both arms equally. We trade absolute performance for a cleaner measurement of the thing
we are actually asking about.

---

## 10. Still open

Stage 1 exploration is finished. Determinism (§2) and `HIDDEN_SIZE` are both closed. What is left
is one optional decision and the pilot - and the pilot is blocked on code, not on experiments.

**`qvel` as extra inputs** (§7) is the last open parameter, and section 6 has changed how to
decide it. Three seeds cannot reach significance - the smallest attainable 3 v 3 Mann-Whitney p is
0.100 - so a sweep cannot *prove* anything here either. Treat it as section 6 treated
`CLOCK_FREQ`: a nuisance parameter where the only question is whether a choice is catastrophic.
There is also a standing argument against it that costs no compute at all: appending hinge velocities
takes the input vector from 12 to 18 and the genotype from 108 to 144 on the John Set gecko, a 33%
larger search space for the same evaluation budget. Default to leaving it out unless a run says otherwise.

**`HIDDEN_SIZE = 6`** is settled by argument rather than by sweep, and the number is measured on
the John Set gecko (1 Oct): `experiments.py hidden` gives 108 weights (18h, from 12 inputs and 6
outputs with no bias vector) and 138.9 evaluations per weight at the provisional 15 000-evaluation
budget.

| HIDDEN_SIZE | genotype | evals per weight |
|---|---|---|
| 2 | 36 | 416.7 |
| 4 | 72 | 208.3 |
| **6** | **108** | **138.9** |
| 8 | 144 | 104.2 |
| 12 | 216 | 69.4 |
| 16 | 288 | 52.1 |

Methods states the count, states that the value was inherited from the template and not tuned,
and says why - the compute went into 20 seeds rather than a network-size sweep that section 6
gives every reason to expect would return another null.

The table raises a question it does not answer: at 77% flat generations (section 6), a *smaller*
network might search this budget better than a larger one. That belongs in the limitations
paragraph, not in an underpowered sweep we cannot afford. Do not let the table tempt a late
change to `HIDDEN_SIZE` - freezing it is worth more than optimising it.

**Re-run the duration probe on the pilot's best individual** before fixing the generation count
(§5). This one cannot move earlier: it is blocked on the pilot existing.

**The actual bottleneck is `run.py`.** `POP_SIZE` and `NUM_GENERATIONS` are the last two
PROVISIONAL values in `config.py` and both have to come out of a pilot run, which needs the
runner. No further experiment shortens that path.

---

## 11. Mutation step-size sweep (the real EA, John Set gecko)

**What we did.** Ran the real EA (`ea.py` + `mutation.py`) for both variants from σ₀ ∈ {0.1, 0.3,
0.5}, 3 seeds each: 18 runs, population 30, 40 generations, 1,230 evaluations per run (the body
pilot's budget). It's the first end-to-end run of the EA on this body. `experiments.py sigma`, 1 Oct.

**Why.** Two reasons:
- The fixed σ is the control. A badly chosen control makes the 1/5 rule look good without
  telling us anything.
- We wanted to know whether the rule ends up in the same place whatever σ it starts from.

**How.** Saved as `results/sigma_sweep_pilot1.csv` and `results/sigma_best_pilot1/`. All six
configurations at a given seed started from the identical population (gen-0 best 1.877 / 1.920 /
1.845 for seeds 1 / 2 / 3), so the pairing works.

**Result 1: final distance (m).** Means over 3 seeds:

| σ₀ | static | adaptive |
|---|---|---|
| 0.1 | 1.317 ± 0.062 | 1.397 ± 0.231 |
| 0.3 | 1.329 ± 0.208 | 1.414 ± 0.258 |
| 0.5 | 1.232 ± 0.338 | 1.225 ± 0.131 |

Every gap is inside the seed-to-seed spread (~0.2 m). Static 0.5 has the best mean, but because of
one seed (0.855 m): the same lottery pattern that ruled out spider_8. Pooled curves are almost
identical up to generation 10, and static is slightly ahead at generation 20 (1.45 vs 1.54).
There's no early advantage for adaptive (H1) at this scale.

**Result 2: the success rate sits at 1/5 whatever σ is.** This is the most important finding.

| | g1 | g10 | g20 | g30 | g40 |
|---|---|---|---|---|---|
| static σ=0.1 | 0.28 | 0.19 | 0.19 | 0.20 | 0.23 |
| static σ=0.3 | 0.20 | 0.22 | 0.21 | 0.20 | 0.19 |
| static σ=0.5 | 0.18 | 0.21 | 0.20 | 0.21 | 0.20 |

Five times the step size, and the same success rate. So the rule's input sits on its own threshold
and carries almost no information about whether σ is too large or too small. Adaptive σ then
wanders: final values ranged **0.05 to 2.0 within the same σ₀**, and σ hit the 2.0 cap in **5 of
9** adaptive runs. The likely cause is the population setting flagged in the Intro: each child is
compared with its own parent, and with tournament selection many parents are mediocre, so roughly
one child in five beats its parent whatever the step size. In the (1+1)-ES the rule was derived
for, the success rate depends strongly on σ.

This parallels A1, where the static variant's success rate followed the adaptive one *"almost
exactly"* and the collapse was *"a property of the search rather than something the controller
caused"*. It's a pilot (3 seeds, 40 generations), so the final experiment has to confirm it. If it
holds, this is the mechanism the second half of the RQ is asking about.

**Correction to the sweep's own printout:** it said the rule *"pulls different starting points
together: evidence it replaces the tuning step"*. That overstates it. The means of final σ differ
by 1.9× against 5× at the start, but within one σ₀ final σ varies 30×. σ forgets its starting
point because it random-walks, not because it converges to a good value. How much final distance
depends on σ₀ was 0.10 m for static and 0.19 m for adaptive, both inside the noise. No evidence yet
that the rule replaces tuning. The printout has since been fixed: it now compares the spread between seeds of
the same σ₀ with the spread between starting values, and adds a check of whether the static
success rate depends on σ at all. On this data it now prints *"it wanders. That is NOT evidence
that the rule replaces tuning"*.

**Result 3: still no plateau, and long flat stretches.**
- **80% of generations** brought no improvement in best distance (old body: 77%).
- The longest flat streaks were 26 and 27 generations.
- Mean curves were still falling 0.05–0.1 m between generations 30 and 40.

The draft's per-run stopping rule (less than 0.01 m improvement over 20 generations) would have
stopped two runs. One of them (adaptive σ₀=0.5, seed 1) was stopped at generation 25
at 1.582 m and then improved by **0.489 m**, its biggest gain, to 1.093 m. That's direct evidence
for a fixed budget over per-run plateau stopping (decision D6).

**Result 4: other observations.**
- Diversity: static σ=0.1 *lost* diversity (genotype spread 0.49 → 0.22), while adaptive runs
  gained a lot (→ 2.0–3.3) because of the large σ values.
- No invalid (NaN) controllers in 22,140 evaluations.

**Result 5: timing.**
- 15 workers on 8 cores: about 0.95 s per evaluation per process, about 15.8 evaluations/s in
  total.
- The last three runs, alone on the machine, still took about 0.63 s per evaluation each, roughly
  single-process throughput.
- 18 jobs on 15 workers meant the second round used only 3 workers. The whole sweep took 34.9 min.
- The speed test (parallelisation plan, step 2) has to explain the slow tail: heat, power mode,
  or something else running.

**What it changes.**

- **D2, fixed σ.** The pilot can't separate 0.1–0.5. Recommendation: keep **0.3**, by argument:
  - It's the middle of a flat range.
  - 0.1 was already losing diversity at 40 generations, a premature-convergence risk in longer
    runs.
  - 0.5's lead rests on one seed (the spider argument).
  - Every earlier probe used 0.3.
  - Draft Methods sentence: *"Fixed σ ∈ {0.1, 0.3, 0.5} gave final distances within the
    seed-to-seed spread (3 seeds, 1,230 evaluations), so we kept σ = 0.3, the middle of this
    range."*
- **D6, stopping rule:** fixed budget. Result 3 is the evidence.
- **Run length:** at least 150–200 generations. To be fixed by the pilot at μ = 50.
- **σ_max = 2.0 is not "just a guard".** It was reached in more than half the adaptive runs. Report
  that, and decide whether it stays at 2.0 (it bounds how far the random walk can go).
- **Budget.** At about 15.8 evaluations/s at full load, 60 runs × 50 × (G+1) evaluations take
  ~5.3 h for G = 100, ~10.6 h for G = 200 and ~16 h for G = 300. The earlier estimate in the
  parallelisation plan (6–7 h for G = 200) was too optimistic. This puts pressure on D5 (20 vs 10
  seeds) and D1 (μ).
- **H2 wording:** supported by 80% flat generations on this body. The Intro can keep it as written.

---

## Provenance

| Date | Machine | Script | What it produced |
|---|---|---|---|
| 25 Sep | 15-core | `phase1_timing.py` | sizes, timing, determinism, budget table |
| 29 Sep | laptop | `controller.py` self-test | sizes, one fitness, timing, determinism |
| 29 Sep | laptop | `watch.py --quick-evolve 20` | the walk gate; `quick_best.npy` |
| 29 Sep | laptop | `experiments.py duration` | `results/duration.csv` |
| 29 Sep | laptop | `experiments.py clock` | `results/clock_freq.csv` (1 seed, 4 freqs) |
| 29 Sep | laptop | `experiments.py clock --freqs 1 2 3 4 5 6 --seeds 1 2 3` | the 18-run sweep, §6 |
| 29 Sep | - | arithmetic, no run | the crossover decision, §9 |
| 30 Sep | laptop | `experiments.py hidden` | `results/hidden_size.csv`, §10 |
| 30 Sep | laptop | `experiments.py determ` | `results/determinism.csv`, §2 (8-hinge body) |
| 1 Oct | laptop | `experiments.py hidden` | §1, §10 on `john_set.gecko` |
| 1 Oct | laptop | `experiments.py determ` | §2 on `john_set.gecko` |
| 1 Oct | laptop, 15 workers | `experiments.py sigma` (defaults) | §11; `results/sigma_sweep_pilot1.csv` |
| 1 Oct | laptop | `experiments.py duration --genotype results/sigma_best_pilot1/static_sigma0.5_seed3.npy` | §5 on `john_set.gecko`; `results/duration.csv` |
| 1 Oct | laptop | speed test (`sigma --workers 1`, seeds 1–12 finished) | 0.345 s/evaluation single-worker; the hang in §8b |
| 1 Oct | laptop | speed test, `--workers 2 4 8 12 16` (after the fix) | speedups up to 6.7× (19.6 evals/s at 16); §8b confirmed 5/5 |
| 29 Sep | laptop | `experiments.py qvel` | `results/qvel.csv` |
| 29 Sep | laptop | `experiments.py nan` | `results/nan_guard.csv` |
| 29 Sep | laptop | `experiments.py cache` | `results/cache.csv` |
