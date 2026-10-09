# Makes the figures and statistics for the report from main.py's results. Run from the project root:
#   uv run assignments/assignment_2/group_28/plots.py              (the main experiment)
#   uv run assignments/assignment_2/group_28/plots.py --tag smoke
# Everything is written to results/<tag>/<body>/analysis/.

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

import config

VARIANT_ORDER = ("static", "adaptive", "adaptive_cap1", "baseline")
EA_VARIANTS = ("static", "adaptive", "adaptive_cap1")
ADAPTIVE_VARIANTS = ("adaptive", "adaptive_cap1")
COLORS = {
    "static": "#1f77b4",
    "adaptive": "#d62728",
    "adaptive_cap1": "#ff7f0e",
    "baseline": "#7f7f7f",
}
LABELS = {
    "static": rf"fixed $\sigma = {config.STATIC_SIGMA}$",
    "adaptive": rf"1/5 rule, $\sigma_{{\max}} = {config.ADAPTIVE_MAX_SIGMA}$",
    "adaptive_cap1": rf"1/5 rule, $\sigma_{{\max}} = {config.ADAPTIVE_CAP1_MAX_SIGMA}$",
    "baseline": "random search",
}
SATURATION_DISTANCE = 0.15  # a run that ends closer than this (m) has basically reached the target


def ignore_nan_warnings(func, *args, **kwargs):
    """np.nanmean and friends warn on columns that are all NaN, those are expected here."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return func(*args, **kwargs)


def sample_std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_seed_runs(tag: str, body: str, variant: str) -> list[pd.DataFrame]:
    """One dataframe per seed folder, with an extra best_so_far column."""
    variant_dir = config.RESULTS_DIR / tag / body / variant
    if not variant_dir.exists():
        return []

    runs = []
    for seed_dir in sorted(variant_dir.iterdir(), key=lambda path: path.name):
        path = seed_dir / config.RESULT_FILE_NAME
        if seed_dir.is_dir() and path.exists():
            df = pd.read_csv(path).sort_values("generation").reset_index(drop=True)
            df["best_so_far"] = df["best_fitness"].cummin()
            runs.append(df)
    return runs


def is_saturated(df: pd.DataFrame) -> bool:
    return float(df["best_so_far"].iloc[-1]) <= SATURATION_DISTANCE


@dataclass
class Curve:
    """One variant, summarised over its seeds."""
    generations: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    finals: list[float]  # best_so_far at the last generation, one per seed
    seeds: list[int]
    sigma_runs: np.ndarray | None  # sigma per generation, one row per seed
    success_mean: np.ndarray | None


def make_curve(variant: str, runs: list[pd.DataFrame]) -> Curve | None:
    if not runs:
        return None

    # Cut every run to the shortest one, so an unfinished run does not get padded
    length = min(len(df) for df in runs)
    if length < max(len(df) for df in runs):
        print(f"  [{variant}] seeds have different lengths, all cut to {length} rows")

    best = np.stack([df["best_so_far"].to_numpy()[:length] for df in runs])
    std = np.nanstd(best, axis=0, ddof=1) if len(runs) > 1 else np.zeros(length)

    sigma_runs = success_mean = None
    if runs[0]["sigma"].notna().any():
        sigma_runs = np.stack([df["sigma"].to_numpy(dtype=float)[:length] for df in runs])
    if runs[0]["success_rate"].notna().any():
        success = np.stack([df["success_rate"].to_numpy(dtype=float)[:length] for df in runs])
        success_mean = ignore_nan_warnings(np.nanmean, success, axis=0)

    return Curve(
        generations=runs[0]["generation"].to_numpy()[:length],
        mean=np.nanmean(best, axis=0),
        std=std,
        finals=best[:, -1].tolist(),
        seeds=[int(df["seed"].iloc[0]) for df in runs],
        sigma_runs=sigma_runs,
        success_mean=success_mean,
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def save(fig, ax, title: str, output_path: Path) -> None:
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_fitness(curves: dict[str, Curve], output_path: Path, title: str) -> None:
    """Best-so-far fitness per generation, mean +- std over the seeds."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for variant in VARIANT_ORDER:
        if variant not in curves:
            continue
        curve = curves[variant]
        ax.plot(curve.generations, curve.mean, label=f"{LABELS[variant]} (n={len(curve.seeds)})",
                color=COLORS[variant])
        ax.fill_between(curve.generations, curve.mean - curve.std, curve.mean + curve.std,
                        color=COLORS[variant], alpha=0.15)
    ax.set_xlabel("generation")
    ax.set_ylabel("best-so-far fitness (distance to target, m)")
    save(fig, ax, title, output_path)


def plot_sigma(curves: dict[str, Curve], output_path: Path, title: str) -> None:
    """Sigma of every adaptive run (thin) and the median per variant (thick)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for variant in ADAPTIVE_VARIANTS:
        curve = curves.get(variant)
        if curve is None or curve.sigma_runs is None:
            continue
        for run in curve.sigma_runs:
            ax.plot(curve.generations, run, color=COLORS[variant], alpha=0.2, lw=0.7)
        median = ignore_nan_warnings(np.nanmedian, curve.sigma_runs, axis=0)
        ax.plot(curve.generations, median, color=COLORS[variant], lw=2.2,
                label=f"{LABELS[variant]} (median, n={len(curve.seeds)})")
    ax.axhline(config.STATIC_SIGMA, color=COLORS["static"], ls="--", lw=1, label=LABELS["static"])
    ax.set_xlabel("generation")
    ax.set_ylabel(r"$\sigma$")
    ax.set_yscale("log")
    save(fig, ax, title, output_path)


def plot_success_rate(curves: dict[str, Curve], output_path: Path, title: str) -> None:
    """Mutation success rate per generation (mean over seeds) against the 1/5 target."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for variant in EA_VARIANTS:
        curve = curves.get(variant)
        if curve is None or curve.success_mean is None:
            continue
        ax.plot(curve.generations, curve.success_mean, label=f"{LABELS[variant]} (n={len(curve.seeds)})",
                color=COLORS[variant])
    ax.axhline(config.ADAPTIVE_TARGET_SUCCESS, color="grey", ls=":", lw=1, label="1/5 target")
    ax.set_xlabel("generation")
    ax.set_ylabel("success rate (mean over seeds)")
    save(fig, ax, title, output_path)


def plot_final_distance(curves: dict[str, Curve], output_path: Path, title: str) -> None:
    """A box per variant with every run drawn on top of it."""
    present = [variant for variant in VARIANT_ORDER if variant in curves]
    finals = [curves[variant].finals for variant in present]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot(finals, showfliers=False, medianprops={"color": "black"})  # the dots already show outliers
    jitter = np.random.default_rng(0)  # fixed seed, so the figure is the same every time
    for i, (variant, values) in enumerate(zip(present, finals), start=1):
        x = i + jitter.uniform(-0.12, 0.12, size=len(values))
        ax.scatter(x, values, s=14, color=COLORS[variant], alpha=0.8, zorder=3)
    ax.axhline(SATURATION_DISTANCE, color="grey", ls=":", lw=1,
               label=f"{SATURATION_DISTANCE} m (saturated below)")
    ax.set_xticks(range(1, len(present) + 1))
    ax.set_xticklabels([LABELS[variant] for variant in present], fontsize=8)
    ax.set_ylabel("final best distance to target (m)")
    save(fig, ax, title, output_path)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def vargha_delaney_a12(a: list[float], b: list[float]) -> float:
    """Chance that a value from a is lower (= better) than one from b, ties count half."""
    wins = ties = 0
    for x in a:
        for y in b:
            if x < y:
                wins += 1
            elif x == y:
                ties += 1
    return (wins + 0.5 * ties) / (len(a) * len(b))


def effect_size(a12: float) -> str:
    """Vargha-Delaney thresholds 0.56 / 0.64 / 0.71, as distance from 0.5."""
    distance = abs(a12 - 0.5)
    if distance < 0.06:
        return "negligible"
    if distance < 0.14:
        return "small"
    if distance < 0.21:
        return "medium"
    return "large"


def pairwise_tests(values: dict[str, list[float]], measure: str) -> list[dict]:
    """Two-sided Mann-Whitney U test and A12 for every pair of variants."""
    present = [variant for variant in VARIANT_ORDER if variant in values]
    rows = []
    for i, a in enumerate(present):
        for b in present[i + 1:]:
            u_stat, p_value = mannwhitneyu(values[a], values[b], alternative="two-sided")
            a12 = vargha_delaney_a12(values[a], values[b])
            rows.append({
                "measure": measure, "a": a, "b": b, "n_a": len(values[a]), "n_b": len(values[b]),
                "median_a": float(np.median(values[a])), "median_b": float(np.median(values[b])),
                "U": float(u_stat), "p_raw": float(p_value),
                "A12_a_over_b": a12, "effect": effect_size(a12),
            })
    return rows


def holm(p_values: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni corrected p-values, in the original order."""
    adjusted = np.empty(len(p_values))
    running_max = 0.0
    for rank, i in enumerate(p_values.argsort()):
        running_max = max(running_max, min(1.0, (len(p_values) - rank) * p_values[i]))
        adjusted[i] = running_max
    return adjusted


def convergence(runs: dict[str, list[pd.DataFrame]]) -> tuple[pd.DataFrame, float | None, int]:
    """Convergence speed: evaluations until a run first reaches the target, where the target
    is the worst final best-so-far of all EA runs (so every EA run reaches it). Runs that
    never reach it count as budget + 1. Returns (table, target, value for not reached)."""
    ea_finals = [float(df["best_so_far"].iloc[-1]) for variant in EA_VARIANTS for df in runs.get(variant, [])]
    if not ea_finals:
        return pd.DataFrame(), None, 0
    target = max(ea_finals)
    not_reached = max(int(df["evaluations"].iloc[-1]) for dfs in runs.values() for df in dfs) + 1

    rows = []
    for variant, dfs in runs.items():
        for df in dfs:
            reached = df.index[df["best_so_far"] <= target]
            rows.append({
                "variant": variant,
                "seed": int(df["seed"].iloc[0]),
                "target": target,
                "evaluations_to_target": int(df["evaluations"].iloc[reached[0]]) if len(reached) else None,
                "generations_to_target": int(df["generation"].iloc[reached[0]]) if len(reached) else None,
            })
    return pd.DataFrame(rows), target, not_reached


def plateau_generation(generations: np.ndarray, mean_curve: np.ndarray, eps: float, window: int) -> int | None:
    """First generation from which the mean curve never again improves by eps or more within
    `window` generations. None if that never happens."""
    if len(mean_curve) <= window:
        return None

    settled = (mean_curve[:-window] - mean_curve[window:]) < eps
    for i in range(len(settled)):
        if settled[i:].all():
            return int(generations[i + window])
    return None


def summary_table(curves: dict[str, Curve], runs: dict[str, list[pd.DataFrame]], conv: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant in VARIANT_ORDER:
        if variant not in curves:
            continue
        finals = curves[variant].finals
        reached = conv.loc[
            (conv["variant"] == variant) & conv["evaluations_to_target"].notna(), "evaluations_to_target"
        ].astype(float)
        rows.append({
            "variant": variant,
            "n_seeds": len(finals),
            "best_overall": round(min(finals), 4),
            "median_final_best": round(float(np.median(finals)), 4),
            "mean_final_best": round(float(np.mean(finals)), 4),
            "std_final_best": round(sample_std(finals), 4),
            "evals_to_target_mean": round(float(reached.mean()), 1) if len(reached) else "",
            "evals_to_target_std": round(sample_std(reached.tolist()), 1) if len(reached) > 1 else "",
            "n_reached_target": int(len(reached)),
            "n_saturated": sum(is_saturated(df) for df in runs[variant]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze(tag: str, body: str, plateau_eps: float, plateau_window: int) -> None:
    runs: dict[str, list[pd.DataFrame]] = {}
    curves: dict[str, Curve] = {}
    for variant in VARIANT_ORDER:
        variant_runs = load_seed_runs(tag, body, variant)
        curve = make_curve(variant, variant_runs)
        if curve is None:
            print(f"  no seeds found for {variant}")
            continue
        runs[variant] = variant_runs
        curves[variant] = curve
        print(f"  {variant}: {len(curve.seeds)} seed(s) {curve.seeds}")

    if not curves:
        print(f"No results found for tag={tag}, body={body}")
        return

    output_dir = config.RESULTS_DIR / tag / body / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    title = f"{body}, tag={tag}"

    plot_fitness(curves, output_dir / "fitness_curve.png", title)
    plot_sigma(curves, output_dir / "sigma_curve.png", title)
    plot_success_rate(curves, output_dir / "success_rate_curve.png", title)
    plot_final_distance(curves, output_dir / "final_distance.png", title)

    conv, target, not_reached = convergence(runs)
    conv.to_csv(output_dir / "convergence.csv", index=False)

    # Both measures compared for every pair of variants, Holm-corrected over all tests together
    final_values = {variant: curve.finals for variant, curve in curves.items()}
    conv_values = {
        variant: conv.loc[conv["variant"] == variant, "evaluations_to_target"].fillna(not_reached).astype(float).tolist()
        for variant in curves
    }
    stats = pd.DataFrame(pairwise_tests(final_values, "final_distance") + pairwise_tests(conv_values, "evaluations_to_target"))
    if not stats.empty:
        stats["p_holm"] = holm(stats["p_raw"].to_numpy())
    stats.to_csv(output_dir / "stats.csv", index=False)

    table = summary_table(curves, runs, conv)
    table.to_csv(output_dir / "summary_table.csv", index=False)

    plateaus = {
        variant: plateau_generation(curve.generations, curve.mean, plateau_eps, plateau_window)
        for variant, curve in curves.items()
    }

    summary = {
        "tag": tag,
        "body": body,
        "plateau_eps": plateau_eps,
        "plateau_window": plateau_window,
        "plateau_generation": plateaus,
        "saturation_distance": SATURATION_DISTANCE,
        "final_best_so_far": {
            variant: {
                "mean": float(np.mean(curve.finals)),
                "std": sample_std(curve.finals),
                "median": float(np.median(curve.finals)),
                "n_seeds": len(curve.seeds),
            }
            for variant, curve in curves.items()
        },
        "convergence_target": target,
        "convergence_not_reached_counted_as": not_reached,
        "pairwise_stats": stats.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nWritten to {output_dir}\n")
    print(table.to_string(index=False))
    print(f"\nconvergence target: {target}")
    print(f"plateau generation (eps={plateau_eps}, window={plateau_window}): {plateaus}")
    if not stats.empty:
        print(f"\npairwise tests (Holm over all {len(stats)} tests):")
        print(stats.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Figures and statistics for the report.")
    parser.add_argument("--tag", default="main", help="Results tag. Defaults to: %(default)s.")
    parser.add_argument("--body", default=config.BODY_NAME, help="Body subfolder. Defaults to: %(default)s.")
    parser.add_argument("--plateau-eps", type=float, default=0.02,
                        help="Improvement (m) that still counts as progress. Defaults to: %(default)s.")
    parser.add_argument("--plateau-window", type=int, default=20,
                        help="Generations the improvement is measured over. Defaults to: %(default)s.")
    args = parser.parse_args()
    analyze(args.tag, args.body, args.plateau_eps, args.plateau_window)


if __name__ == "__main__":
    main()
