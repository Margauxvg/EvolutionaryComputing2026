import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config


def load_seed_runs(variant: str) -> list[list[dict]]:
    """One list of generation-rows per seed folder found for this variant."""
    runs = []
    for seed_dir in sorted((config.RESULTS_DIR / variant).glob("seed_*")):
        path = seed_dir / config.RESULT_FILE_NAME
        if path.exists():
            with path.open(newline="", encoding="utf-8") as csv_file:
                runs.append(list(csv.DictReader(csv_file)))
    return runs


def column_matrix(runs: list[list[dict]], column: str, cumulative_min: bool = False) -> np.ndarray:
    """(num_seeds, num_generations) array; trims to the shortest run in case seeds
    were produced under different NUM_GENERATIONS. `cumulative_min` takes each row's
    running minimum first - needed for baseline, whose best_fitness is only the best
    of that generation's fresh random batch, not the best found so far."""
    generations = min(len(run) for run in runs)
    matrix = np.array([[float(row[column]) for row in run[:generations]] for run in runs])
    if cumulative_min:
        matrix = np.minimum.accumulate(matrix, axis=1)
    return matrix


def plot_fitness_comparison(output_path: Path = config.RESULTS_DIR / "fitness_comparison.png") -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))

    for variant in config.VARIANTS:
        runs = load_seed_runs(variant)
        if not runs:
            continue

        matrix = column_matrix(runs, "best_fitness", cumulative_min=(variant == "baseline"))
        generations = np.arange(1, matrix.shape[1] + 1)
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)

        ax.plot(generations, mean, label=f"{variant} (n={matrix.shape[0]})")
        ax.fill_between(generations, mean - std, mean + std, alpha=0.2)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness (lower is better)")
    ax.set_title("Fitness over generations (mean ± std across independent runs)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_adaptive_probability(output_path: Path = config.RESULTS_DIR / "adaptive_probability.png") -> Path:
    runs = load_seed_runs("adaptive")
    if not runs:
        raise FileNotFoundError("No adaptive runs found under results/adaptive/.")

    matrix = column_matrix(runs, "mutation_probability")
    generations = np.arange(1, matrix.shape[1] + 1)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(generations, mean, color="tab:orange")
    ax.fill_between(generations, mean - std, mean + std, alpha=0.2, color="tab:orange")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mutation probability")
    ax.set_title("Adaptive mutation probability over generations (mean ± std across independent runs)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> None:
    print(f"Saved: {plot_fitness_comparison()}")
    print(f"Saved: {plot_adaptive_probability()}")


if __name__ == "__main__":
    main()
