import csv
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

import config
from helpers import variant_dir, write_csv

SUMMARY_CSV_NAME = "summary_table.csv"
SIGNIFICANCE_CSV_NAME = "significance_ranking.csv"
SIGNIFICANCE_TEST = "Mann-Whitney U (two-sided) on final best fitness, Holm-corrected"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_seed_runs(variant: str, operator: str | None = None) -> list[list[dict]]:
    """One list of generation-rows per seed folder found for this variant/operator."""
    runs = []
    for seed_dir in sorted(variant_dir(variant, operator).glob("seed_*")):
        path = seed_dir / config.RESULT_FILE_NAME
        if path.exists():
            with path.open(newline="", encoding="utf-8") as csv_file:
                runs.append(list(csv.DictReader(csv_file)))
    return runs


def ea_combinations() -> list[tuple[str, str]]:
    return list(itertools.product(("static", "adaptive"), config.MUTATION_OPERATORS))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def column_matrix(runs: list[list[dict]], column: str, cumulative_min: bool = False) -> np.ndarray:
    """(num_seeds, num_generations) array; trims to the shortest run."""
    generations = min(len(run) for run in runs)
    matrix = np.array([[float(row[column]) for row in run[:generations]] for run in runs])
    if cumulative_min:
        matrix = np.minimum.accumulate(matrix, axis=1)
    return matrix


def _plot_series(ax, variant: str, operator: str | None, column: str, cumulative_min: bool = False, **kwargs) -> None:
    runs = load_seed_runs(variant, operator)
    if not runs:
        return

    matrix = column_matrix(runs, column, cumulative_min=cumulative_min)
    generations = np.array([int(row["generation"]) for row in runs[0][: matrix.shape[1]]]) # generation axis
    # ignore NaN (unscored) cells: https://numpy.org/doc/stable/reference/generated/numpy.nanmean.html
    mean = np.nanmean(matrix, axis=0)
    # ignore NaN cells: https://numpy.org/doc/stable/reference/generated/numpy.nanstd.html
    std = np.nanstd(matrix, axis=0, ddof=1) if matrix.shape[0] > 1 else np.zeros_like(mean)

    ax.plot(generations, mean, label=f"{variant} (n={matrix.shape[0]})", **kwargs)
    ax.fill_between(generations, mean - std, mean + std, alpha=0.2, **kwargs)


def plot_fitness_comparison(operator: str, output_path: Path | None = None) -> Path:
    """Baseline vs static vs adaptive best-so-far fitness, for one mutation operator."""
    output_path = output_path or config.RESULTS_DIR / f"fitness_comparison_{operator}.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_series(ax, "baseline", None, "best_fitness", cumulative_min=True)
    _plot_series(ax, "static", operator, "best_fitness", cumulative_min=True)
    _plot_series(ax, "adaptive", operator, "best_fitness", cumulative_min=True)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best-so-far fitness (lower is better)")
    ax.set_title(f"Fitness over generations - operator: {operator} (mean ± std across seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_probability_comparison(operator: str, output_path: Path | None = None) -> Path:
    """Mutation probability over time, static vs adaptive, for one mutation operator."""
    output_path = output_path or config.RESULTS_DIR / f"probability_comparison_{operator}.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_series(ax, "static", operator, "mutation_probability", color="tab:blue")
    _plot_series(ax, "adaptive", operator, "mutation_probability", color="tab:orange")
    ax.axhline(config.ADAPTIVE_MIN_PROBABILITY, linestyle=":", linewidth=1, color="grey")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Mutation probability")
    ax.set_title(f"Mutation probability over generations - operator: {operator} (mean ± std across seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_success_rate_comparison(operator: str, output_path: Path | None = None) -> Path:
    """Mutation success rate over time, static vs adaptive, for one mutation operator, with
    the 1/5 target marked. The static variant measures this signal but never acts on it."""
    output_path = output_path or config.RESULTS_DIR / f"success_rate_comparison_{operator}.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_series(ax, "static", operator, "success_rate", color="tab:blue")
    _plot_series(ax, "adaptive", operator, "success_rate", color="tab:orange")
    ax.axhline(config.ADAPTIVE_TARGET_SUCCESS, linestyle="--", linewidth=1, color="red")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Mutation success rate")
    ax.set_title(f"Mutation success rate over generations - operator: {operator} (mean ± std across seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def make_plots() -> None:
    for operator in config.MUTATION_OPERATORS:
        print(f"Saved: {plot_fitness_comparison(operator)}")
        print(f"Saved: {plot_probability_comparison(operator)}")
        print(f"Saved: {plot_success_rate_comparison(operator)}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

# Current minimum of best_fitness
def best_so_far(run: list[dict]) -> list[float]:
    running_best = float("inf") # start from infinite and decrease as better fitness values are found
    result = []
    for row in run:
        running_best = min(running_best, float(row["best_fitness"]))
        result.append(running_best)
    return result


def final_best(run: list[dict]) -> float:
    return best_so_far(run)[-1]


def generations_to_threshold(run: list[dict], threshold: float) -> int | None:
    """First generation whose best-so-far reaches a fitness threshold shared across configurations."""
    for row, value in zip(run, best_so_far(run)):
        if value <= threshold:
            return int(row["generation"])
    return None


def convergence_threshold() -> float:
    """Worst final best-fitness across all EA runs, so every configuration reaches it. It's used
    to measure how many generations each configuration needs to converge."""
    finals = [
        final_best(run)
        for variant, operator in ea_combinations()
        for run in load_seed_runs(variant, operator)
    ]
    if not finals:
        raise RuntimeError("No EA runs found; cannot derive a convergence threshold.")
    return max(finals)


def summarize_combination(variant: str, operator: str | None, threshold: float) -> dict | None:
    runs = load_seed_runs(variant, operator)
    if not runs:
        return None

    finals = [final_best(run) for run in runs]
    reached = [generations_to_threshold(run, threshold) for run in runs]
    hit = [g for g in reached if g is not None]

    return {
        "variant": variant,
        "operator": operator or "-",
        "n_seeds": len(runs),
        "best_overall": round(min(finals), 3),
        "mean_final_best": round(float(np.mean(finals)), 3),
        "std_final_best": round(float(np.std(finals, ddof=1)), 3) if len(finals) > 1 else 0.0,
        "gens_to_threshold_mean": round(float(np.mean(hit)), 1) if hit else "",
        "gens_to_threshold_std": round(float(np.std(hit, ddof=1)), 1) if len(hit) > 1 else "",
        "n_reached_threshold": len(hit),
    }


def build_table(threshold: float) -> list[dict]:
    rows = []
    for variant, operator in ea_combinations():
        row = summarize_combination(variant, operator, threshold)
        if row is not None:
            rows.append(row)

    baseline_row = summarize_combination("baseline", None, threshold)
    if baseline_row is not None:
        rows.append(baseline_row)

    return rows


def write_table(rows: list[dict], output_path: Path = config.RESULTS_DIR / SUMMARY_CSV_NAME) -> Path:
    return write_csv(rows, output_path)


# ---------------------------------------------------------------------------
# Significance tests
# ---------------------------------------------------------------------------

def prob_a_better(finals_a: list[float], finals_b: list[float]) -> float:
    """Vargha-Delaney A-measure (oriented for minimisation): fraction of cross-pairs in which
    a run from A beats one from B, ties counting a half. 0.5 = indistinguishable, 1.0 = A
    always wins."""
    wins = 0
    for x in finals_a:
        for y in finals_b:
            if x < y:
                wins += 1
            elif x == y:
                wins += 0.5
    return wins / (len(finals_a) * len(finals_b))


def effect_magnitude(a_measure: float) -> str:
    # Standard Vargha-Delaney A-measure thresholds (A = 0.56/0.64/0.71), as distance from 0.5:
    distance = abs(a_measure - 0.5)
    if distance < 0.06:
        return "negligible"
    if distance < 0.14:
        return "small"
    if distance < 0.21:
        return "medium"
    return "large"


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment, controlling the family-wise error rate across
    repeated tests of the same hypothesis."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0

    for rank, index in enumerate(order):
        running = max(running, min((m - rank) * p_values[index], 1.0))
        adjusted[index] = running

    return adjusted


def _compare(category: str, operator: str, label_a: str, finals_a: list[float], label_b: str, finals_b: list[float]) -> dict:
    statistic, p_value = mannwhitneyu(finals_a, finals_b, alternative="two-sided")
    a_measure = prob_a_better(finals_a, finals_b)

    return {
        "category": category,
        "operator": operator,
        "group_a": label_a,
        "group_b": label_b,
        "test": SIGNIFICANCE_TEST,
        "statistic": round(float(statistic), 3),
        "p_value": round(float(p_value), 4),
        "p_holm": None,  # filled in per family below
        "prob_a_better": round(a_measure, 3),
        "effect": effect_magnitude(a_measure),
        "mean_a": round(float(np.mean(finals_a)), 3),
        "mean_b": round(float(np.mean(finals_b)), 3),
    }


def significance_ranking() -> list[dict]:
    results = []

    for operator in config.MUTATION_OPERATORS:
        static_runs = load_seed_runs("static", operator)
        adaptive_runs = load_seed_runs("adaptive", operator)
        if not static_runs or not adaptive_runs:
            continue

        results.append(_compare(
            "static_vs_adaptive", operator,
            f"static_{operator}", [final_best(run) for run in static_runs],
            f"adaptive_{operator}", [final_best(run) for run in adaptive_runs],
        ))

    baseline_runs = load_seed_runs("baseline", None)
    baseline_finals = [final_best(run) for run in baseline_runs] if baseline_runs else []

    if baseline_finals:
        for variant, operator in ea_combinations():
            combo_runs = load_seed_runs(variant, operator)
            if not combo_runs:
                continue
            results.append(_compare(
                "vs_baseline", operator,
                f"{variant}_{operator}", [final_best(run) for run in combo_runs],
                "baseline", baseline_finals,
            ))

    # Holm-correct within each family of tests
    for category in {row["category"] for row in results}:
        family = [row for row in results if row["category"] == category]
        for row, adjusted in zip(family, holm([row["p_value"] for row in family])):
            row["p_holm"] = round(adjusted, 4)

    return sorted(results, key=lambda r: (r["category"], r["p_value"]))


def write_significance_csv(
    rows: list[dict], output_path: Path = config.RESULTS_DIR / SIGNIFICANCE_CSV_NAME
) -> Path:
    return write_csv(rows, output_path)


def make_summary() -> None:
    rows = build_table(convergence_threshold())
    output_path = write_table(rows)
    print(f"Saved: {output_path}")

    ranking = significance_ranking()
    significance_path = write_significance_csv(ranking)
    print(f"Saved: {significance_path}")


def main() -> None:
    make_plots()
    print()
    make_summary()


if __name__ == "__main__":
    main()
