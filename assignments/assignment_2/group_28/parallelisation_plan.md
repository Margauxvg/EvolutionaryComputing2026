# Parallelising the runs: what, why, and how

Working note for Margaux's to-do (1 October). This is advice only; no code has been changed.

---

## What parallelisation is for

It only saves time. Results don't change: every run is deterministic from its seed, and `determ`
showed that a spawned process gives exactly the same fitness.

The final experiment is 3 configurations × 20 seeds = **60 independent runs**, each of
50 × (G + 1) evaluations.

**Updated 1 Oct with the measured throughput.** The sigma sweep ran at about **15.8 evaluations/s
in total** with 15 workers on this laptop (stage1_findings §11). One-at-a-time time uses about
4.3 evaluations/s.

| generations per run | 60 runs, one at a time | 60 runs, full load (~15.8 evals/s) |
|---|---|---|
| 100 | ~19 h | ~5.3 h |
| 200 | ~39 h | ~10.6 h |
| 300 | ~58 h | ~16 h |

The first version of this table said ~6–7 h for G = 200. That was too optimistic.

The final run week is Janna's (Mon 5–Thu 8). One at a time, a single attempt takes most of that
window, and if anything goes wrong there's no time to run it again. Run in parallel, it takes an
afternoon. That margin is the real reason to do it.

---

## Is it expected by the assignment?

No. The brief only warns that physics evaluation is slow and says to *"budget your time
accordingly"*, and the rubric gives no points for it. The rubric also says *"algorithm details are
not mixed with coding details"*, so in the report it's worth one clause at most. Methods already
has the sentence that matters: evaluation gives identical results in a separate process.

---

## Would reviewers think the idea came from Claude?

Not the idea. The course repo has `examples/re_book/1_brain_evolution_multiprocessing.py` (plus a
multithreaded version). It's a brain-evolution example that runs evaluations in parallel with
`ProcessPoolExecutor`, the `spawn` start method and a `--workers` option. So parallel evaluation is
something the course demonstrates for exactly this kind of task, and any group that does the
budget arithmetic ends up there.

To be straight about it: in our sessions, Claude brought it up first. What could look AI-made is
not the idea but the *code*. Copying the pattern from `experiments.py sigma` would carry over its
style. Write `run.py` yourself, using the course example as the reference. And keep the AI-use
statement in the acknowledgements, as in A1. Being open about it is what protects you, not hiding
it.

---

## What to do today

### 1. Count your cores (done)

```
Get-CimInstance Win32_Processor | Select NumberOfCores, NumberOfLogicalProcessors
```

Result on Margaux's laptop: **8 physical cores, 16 logical** (each core runs two hyperthreads).
MuJoCo runs one simulation per core, and hyperthreads add little, so plan around 8.

**Warning:** `experiments.py sigma` defaults to `os.cpu_count() - 1` workers. That counts
*logical* processors, so it would start 15 processes on 8 real cores. Pass `--workers` explicitly.

### 2. Measure the speedup instead of assuming it

Test with at least as many jobs as the largest worker count, here 16 seeds:

```
foreach ($w in 1,2,4,8,12,16) { "workers $w"; uv run assignments/assignment_2/group_28/experiments.py sigma --pop 10 --generations 10 --seeds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 --sigmas 0.3 --variants static --workers $w | Select-String "total wall time" }
```

- About 12–15 minutes in total; most of it is the 1-worker run.
- **Run it before the real sigma sweep.** Each run overwrites `results/sigma_sweep.csv` and
  `results/sigma_best/`.
- Speedup = time with 1 worker ÷ time with N workers. Expect close to N up to about 6–8, then a
  flattening. Hyperthreads (12 and 16) usually add only 10–30%. Heat and the few seconds each
  worker needs to start on Windows (it loads MuJoCo in every new process) cut into it.
- Write the measured times in `stage1_findings.md`.

**Choose a worker count that divides the number of jobs**, so no worker sits idle in the last
round:

| Job | Runs | Options |
|---|---|---|
| Sigma sweep | 18 | 9 workers = 2 rounds; 6 workers = 3 rounds |
| Final experiment | 60 | 10 workers = 6 rounds; 12 workers = 5 rounds |

Take the larger option only if the test shows the hyperthreads help.

### Speed test results (1 Oct)

16 runs of 110 evaluations each (`--pop 10 --generations 10`, 16 seeds). The 1-worker time comes
from the 12 runs that finished before the hang (38 s per run).

| workers | wall time | speedup | evaluations/s |
|---|---|---|---|
| 1 | ~10.1 min | 1.0× | 2.9 |
| 2 | 6.5 min | 1.6× | 4.5 |
| 4 | 4.2 min | 2.4× | 7.0 |
| 8 | 2.3 min | 4.4× | 12.8 |
| 12 | 2.3 min | 4.4× | 12.8 |
| **16** | **1.5 min** | **6.7×** | **19.6** |

**How to read it:**
- **12 workers looks no better than 8 only because of rounding.** 16 jobs on 12 workers still
  take two rounds (12 + 4), the same as on 8.
- **Speedup is below the number of workers at every level.** A laptop runs fastest on one core
  and lowers its clock speed as more cores get busy. One run took 38 s alone, ~69 s with 8
  running and ~90 s with 16.
- **Hyperthreads still help.** 16 workers give about 50% more throughput than 8.

**Choice:**
- **16 workers** when the laptop is otherwise idle (overnight, or Janna's final runs).
- **14–15** if you want to keep using it at the same time.
- Pick job counts that fill the workers: 16 jobs per round.

**Revised budget for the final 60 runs at 16 workers (~19.6 evaluations/s):**

| generations per run | evaluations | wall time |
|---|---|---|
| 100 | 303,000 | ~4.3 h |
| 150 | 453,000 | ~6.4 h |
| 200 | 603,000 | ~8.5 h |
| 300 | 903,000 | ~12.8 h |

These are rough. Per-run cost also depends on the controllers, and longer runs spend
proportionally less time on starting up the workers.

### 3. Choose what runs in parallel: whole runs, not evaluations within a generation

- The course example spreads the individuals of one generation across workers. That speeds up a
  single run, but the workers have to wait for each other at the end of every generation.
- We have 60 independent runs, far more than there are cores. Giving each worker one whole run
  (one variant, one seed) keeps every core busy without that waiting.
- It's also simpler: runs share nothing, and each one is reproducible from its seed.
- Being able to say why we chose this over the course's approach is good evidence we own the
  decision.

### 4. Write `run.py` around one top-level function

For example `run_one(variant, seed)`. The main block builds the list of jobs and hands them to a
process pool. Things that will bite on Windows:

- **The `__main__` guard.** Put the pool inside `if __name__ == "__main__":`. With spawn, each
  worker re-imports the script, and without the guard every worker starts its own pool.
- **Pass everything in the job.** Settings changed in the parent process don't reach the workers.
  Give each job everything it needs.
- **Seed from the run's seed, never from the process ID.** The course example's `_init_worker`
  seeds with `base_seed + os.getpid()`. That makes runs non-reproducible, because process IDs
  differ every time. Spotting that is a nice point to own.
- **Draw the initial population first.** Each run creates its random generator from its seed and
  draws the initial population before anything else uses it. Methods promises that a seed gives
  the same start in every configuration.

### 5. One CSV per run

Use something like `results/adaptive/seed_07.csv`, not one shared file.

- Several processes writing to one file can corrupt it.
- Separate files also allow *resuming*: skip any job whose file already exists. If Janna's laptop
  sleeps at hour 4, she restarts and loses nothing.

### 6. Verify

Run a few jobs one at a time and the same jobs in parallel. The CSVs should be identical apart
from timing columns. This held for the sigma sweep, but check it on your own `run.py`.

### 7. Before long runs

- Turn off Windows sleep.
- Keep the laptop plugged in.
- Print a progress line per finished run, so whoever runs it can see where it is.

---

## Why processes and not threads

In case a reviewer asks: `simulate.py` uses `mj.set_mjcb_control`, which is global. Threads in one
process would share it, so a run could be scored with another run's controller. ariel's
`thread_safe_runner` (`src/ariel/utils/runners.py`) avoids that, and the course's multithreaded
example uses it, but our code doesn't. Separate processes avoid the problem entirely, and that's
what `experiments.py determ` verified.

---

## Time estimate

| Step | Time |
|---|---|
| Steps 1–2 (cores and speedup) | ~20 min; can run alongside the sigma sweep, which already gives real timings |
| Writing `run.py` | 1–2 h |
| Verifying | ~30 min |
