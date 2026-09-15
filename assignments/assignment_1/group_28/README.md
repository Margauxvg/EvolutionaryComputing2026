# Group 28 — Assignment 1

Running the following command needs to be done before committing to ensure we have not made changes to forbidden parts of the code.
```bash
git fetch upstream
git diff --stat upstream/main -- src/ariel     e 
```

Representation chosen: tree (`ariel.ec.genotypes.tree`).

## Layout

Per TA guidance, the whole assignment (config, fitness, genotype and the EA itself) lives
in one file, the way `A1_template_2026.py` and the course examples do it.

| | |
|---|---|
| `run.py` | the whole assignment: config, targets/fitness, tree genotype, shared EA steps (parent/survivor selection, crossover) — all implemented — plus the CLI and the pipeline wiring for the two mutation variants and the baseline, whose bodies are still `TODO`/`NotImplementedError` for us to fill in |
| `plot_fit_per_gen.py` | analysis only — adapted from `examples/z_ec_course/plot_fit_per_gen.py`, reads `run.py`'s databases |

**Still to implement** (see the `TODO` docstrings in `run.py`, section 4):
- `mutate_static` — static mutation variant
- `AdaptiveMutation.mutate` / `AdaptiveMutation.adapt` — adaptive 1/5-rule variant
- `baseline_regenerate` — random-search baseline

The EA is built on `ariel.ec.EA`, so the framework persists every individual to a SQLite
database per run, under `__data__/assignment_1/<variant>/seed_<n>/database.db`. There is no
CSV layer: `plot_fit_per_gen.py` reads those databases, `simulate_from_db.py` in the course
examples can replay the best body from one, and `EA(restart=...)` can resume a run.

## Research question

Parent selection (k-tournament), crossover (subtree) and survivor selection ((mu+lambda)
truncation) are identical across every variant — only the mutation step is varied, so any
difference in outcome is attributable to it:

- `static` — every offspring receives the same, fixed number of structural edits.
- `adaptive` — that number ("strength") is controlled by Rechenberg's 1/5-success rule:
  each generation, if more than 1/5 of offspring improved on the better of their two
  parents, the strength grows; otherwise it shrinks.
- `baseline` — random search at the same evaluation budget (`POP_SIZE * NUM_GENERATIONS`),
  no inheritance and no selection.

## Genotypes are stored as JSON

`Individual.genotype_` is a JSON column, so genotypes live in serialised form
(`TreeGenome.to_dict()`); `run.py` converts to/from `TreeGenome` only where ariel's
operators need it.

Per-individual extras go in `Individual.tags`, which is also a JSON column — e.g.
`ind.tags = {"baseline_fitness": ...}`, used by the adaptive variant's 1/5-rule bookkeeping.

## Two engine settings that bite

- `is_maximisation` defaults to **True**; our fitness is a minimisation. Pass
  `is_maximisation=False` or `get_solution("best")` returns the worst individual.
- `db_handling` defaults to **"delete"**: it wipes an existing database at that path.
  `run.db_path(variant, seed)` gives each run its own.

## Running

```bash
cd assignments/assignment_1
uv run python -m group_28.run --variant baseline --seeds 1 2 3 4 5
uv run python -m group_28.run --variant static --seeds 1 2 3 4 5
uv run python -m group_28.run --variant adaptive --seeds 1 2 3 4 5
uv run python -m group_28.plot_fit_per_gen --variants baseline static adaptive
```

The source script plots one run, and its band is the spread *within* a generation's
population. This one reduces each run to one value per generation, then takes mean and std
*across* runs — which is what the brief asks for.

## Submission

`28.zip` -> folder `28/` -> `28.pdf` + this code. Confirm the exact naming with Andy.
