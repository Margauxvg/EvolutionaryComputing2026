# Group 28 — Assignment 1

All our code lives in this folder. Nothing outside it is edited — `src/ariel/` in particular,
which the brief says is fraud to modify:

```bash
git fetch upstream
git diff --stat upstream/main -- src/ariel     # must print nothing
```

Representation: tree only (`ariel.ec.genotypes.tree`).

## What this assumes

The EA is built on `ariel.ec.EA`, so the framework persists every individual to a SQLite
database per run, under `__data__/assignment_1/<variant>/seed_<n>/database.db`. There is no
CSV layer: `plot_fit_per_gen.py` reads those databases, `simulate_from_db.py` in the course
examples can replay the best body from one, and `EA(restart=...)` can resume a run.

| | |
|---|---|
| `config.py` | shared constants, and the per-run database path |
| `problem.py` | targets and fitness, as defined in `A1_template_2026.py` |
| `genome.py` | the tree encoding, the operators ariel supplies, and JSON conversion |
| `variants.py` | the EA variants — not implemented |
| `baseline.py` | random search at the same budget — not implemented |
| `run.py` | CLI |
| `plot_fit_per_gen.py` | adapted from `examples/z_ec_course/plot_fit_per_gen.py` |

## Genotypes are stored as JSON

`Individual.genotype_` is a JSON column, so genotypes live in serialised form and
`to_native` / `from_native` convert to and from the `TreeGenome` ariel's operators expect.

Per-individual extras go in `Individual.tags`, which is also a JSON column — e.g.
`ind.tags = {"n_nodes": ...}`.

## Two engine settings that bite

- `is_maximisation` defaults to **True**; our fitness is a minimisation. Pass
  `is_maximisation=False` or `get_solution("best")` returns the worst individual.
- `db_handling` defaults to **"delete"**: it wipes an existing database at that path.
  `config.db_path(variant, seed)` gives each run its own.

## Running

```bash
cd assignments/assignment_1
uv run python -m group_28.run --variant baseline --seeds 1 2 3 4 5
uv run python -m group_28.run --variant A --seeds 1 2 3 4 5
uv run python -m group_28.run --variant B --seeds 1 2 3 4 5
uv run python -m group_28.plot_fit_per_gen --variants baseline A B
```

The source script plots one run, and its band is the spread *within* a generation's
population. This one reduces each run to one value per generation, then takes mean and std
*across* runs — which is what the brief asks for.

## Submission

`28.zip` -> folder `28/` -> `28.pdf` + this code. Confirm the exact naming with Andy.
