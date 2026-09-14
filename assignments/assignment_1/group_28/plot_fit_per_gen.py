"""Fitness per generation, aggregated across independent runs.

Adapted from `examples/z_ec_course/plot_fit_per_gen.py`. That script plots one
run: its band is the spread WITHIN a generation's population. The brief asks for
the average and spread OVER the independent runs, so this version reads every
database under a variant's folder, reduces each run to one value per generation,
and takes mean and std across runs.

    python -m group_28.plot_fit_per_gen --variants baseline A B
    python -m group_28.plot_fit_per_gen --variants A B --stat mean
"""

import argparse
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402


def per_generation(path: Path, mode: str, stat: str) -> pd.Series:
    """One value per generation for a single run's database.

    Population membership follows the source script: an individual is alive in
    generation g when time_of_birth <= g < time_of_death.
    """
    best_fn = np.min if mode == "min" else np.max
    data = pd.read_sql("SELECT * FROM individual", sqlite3.connect(str(path)))

    min_gen = int(data["time_of_birth"].min())
    max_gen = int(data["time_of_death"].max())

    population_per_gen = {
        gen: data.loc[
            (data["time_of_birth"] <= gen) & (data["time_of_death"] > gen), "id"
        ].tolist()
        for gen in range(min_gen, max_gen + 1)
    }

    fitness_by_id = data.set_index("id")["fitness_"]

    values = []
    for gen in population_per_gen:
        ids = population_per_gen.get(int(gen), [])
        fits = fitness_by_id.reindex(ids).dropna().astype(float).values
        if fits.size == 0:
            values.append(np.nan)
        else:
            values.append(float(best_fn(fits)) if stat == "best" else float(np.mean(fits)))

    return pd.Series(values, index=list(population_per_gen), name=path.parent.name)


def runs_for(variant: str) -> list[Path]:
    return sorted((config.DATA / variant).glob("*/database.db"))


def aggregate(variant: str, mode: str, stat: str) -> pd.DataFrame:
    """-> DataFrame, one column per run, indexed by generation."""
    series = [per_generation(p, mode, stat) for p in runs_for(variant)]
    if not series:
        msg = f"no databases found under {config.DATA / variant}"
        raise FileNotFoundError(msg)
    return pd.concat(series, axis=1).dropna()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fitness per generation, averaged over independent runs"
    )
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--mode", choices=["min", "max"], default="min",
                        help="Whether lower or higher fitness is better")
    parser.add_argument("--stat", choices=["best", "mean"], default="best",
                        help="Per-generation statistic taken within each run")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    plt.figure(figsize=(10, 5))
    for variant in args.variants:
        df = aggregate(variant, args.mode, args.stat)
        x = df.index
        mean = df.mean(axis=1)
        std = df.std(axis=1, ddof=0)
        line, = plt.plot(x, mean, linewidth=2, label=f"{variant} (n={df.shape[1]})")
        plt.fill_between(x, mean - std, mean + std, color=line.get_color(), alpha=0.25)

    plt.xlabel("Generation")
    plt.ylabel(f"Fitness ({args.stat} per run)")
    plt.title("Fitness per generation, mean ± std over independent runs")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out = Path(args.out) if args.out else config.DATA / "fitness_per_generation.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
