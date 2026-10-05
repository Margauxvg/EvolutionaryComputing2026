import argparse
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

import config

VARIANT_ORDER = ("static", "adaptive", "adaptive_cap1", "baseline")
EA_VARIANTS = ("static", "adaptive", "adaptive_cap1")
ADAPTIVE_VARIANTS = ("adaptive", "adaptive_cap1")
VARIANT_COLOR = {
    "static": "#1f77b4",
    "adaptive": "#d62728",
    "adaptive_cap1": "#ff7f0e",
    "baseline": "#7f7f7f",
}
VARIANT_LABEL = {
    "static": rf"fixed $\sigma = {config.STATIC_SIGMA}$",
    "adaptive": rf"1/5 rule, $\sigma_{{\max}} = {config.ADAPTIVE_MAX_SIGMA}$",
    "adaptive_cap1": rf"1/5 rule, $\sigma_{{\max}} = {config.ADAPTIVE_CAP1_MAX_SIGMA}$",
    "baseline": "random search",
}

SATURATION_DISTANCE = 0.15  # distance below which a run is considered saturated (m)
IMPROVEMENT_EPS = 0.01      # minimum imporvement considered meaningful (m)
LATE_GENERATIONS = 50       # number of final generations used for success rate analysis


def _nan_quietly(func, *args, **kwargs):
    """Apply a NumPy aggregation without warnings for all-NaN slices."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return func(*args, **kwargs)


# --------------------------------------------------------------------------- #
#  Loading
# --------------------------------------------------------------------------- #
def variant_dir(tag: str, body: str, variant: str) -> Path:
    return config.RESULTS_DIR / tag / body / variant


def seed_dirs(tag: str, body: str, variant: str) -> list[Path]:
    base = variant_dir(tag, body, variant)
    if not base.exists():
        return []
    return sorted(
        (p for p in base.iterdir() if p.is_dir() and (p / config.RESULT_FILE_NAME).exists()),
        key=lambda p: p.name,
    )


def load_seed_csv(folder: Path) -> pd.DataFrame:
    """Load one seed's results and add the running best fitness."""
    df = pd.read_csv(folder / config.RESULT_FILE_NAME)
    df = df.sort_values("generation").reset_index(drop=True)
    df["best_so_far"] = df["best_fitness"].cummin()
    return df


def is_saturated(df: pd.DataFrame) -> bool:
    return float(df["best_so_far"].iloc[-1]) <= SATURATION_DISTANCE


# --------------------------------------------------------------------------- #
#  Aggregation
# --------------------------------------------------------------------------- #
@dataclass
class VariantCurve:
    variant: str
    generations: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    per_seed_final: list[float]          # best_so_far at each seed's last generation
    seeds: list[int] = field(default_factory=list)
    sigma_runs: np.ndarray | None = None  # one row per seed, for plot (ii)
    success_mean: np.ndarray | None = None


def aggregate(variant: str, frames: list[pd.DataFrame]) -> VariantCurve | None:
    if not frames:
        return None

    # use the shortest run so interrupted runs are not padded
    min_len = min(len(f) for f in frames)
    if min_len < max(len(f) for f in frames):
        print(f"  [{variant}] seeds have different lengths; truncating all to {min_len} rows")

    stacked = np.stack([f["best_so_far"].to_numpy()[:min_len] for f in frames])
    generations = frames[0]["generation"].to_numpy()[:min_len]
    # sample standard deviation across seeds (ddof=1)
    curve_std = np.nanstd(stacked, axis=0, ddof=1) if stacked.shape[0] > 1 else np.zeros(min_len)

    sigma_runs = success_mean = None
    if frames[0]["sigma"].notna().any():
        sigma_runs = np.stack([f["sigma"].to_numpy(dtype=float)[:min_len] for f in frames])
    if frames[0]["success_rate"].notna().any():
        success_mean = _nan_quietly(
            np.nanmean,
            np.stack([f["success_rate"].to_numpy(dtype=float)[:min_len] for f in frames]),
            axis=0,
        )

    return VariantCurve(
        variant=variant,
        generations=generations,
        mean=np.nanmean(stacked, axis=0),
        std=curve_std,
        per_seed_final=stacked[:, -1].tolist(),
        seeds=[int(f["seed"].iloc[0]) for f in frames],
        sigma_runs=sigma_runs,
        success_mean=success_mean,
    )


def sample_std(values: list[float]) -> float:
    """ddof=1 everywhere (table, json and printout), 0.0 for a single value."""
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


# --------------------------------------------------------------------------- #
#  Plotting
# --------------------------------------------------------------------------- #
def plot_fitness_curve(curves: dict[str, VariantCurve], out_path: Path, title: str) -> None:
    """Plot fitness over generations."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for variant in VARIANT_ORDER:
        curve = curves.get(variant)
        if curve is None:
            continue
        color = VARIANT_COLOR[variant]
        ax.plot(curve.generations, curve.mean,
                label=f"{VARIANT_LABEL[variant]} (n={len(curve.seeds)})", color=color)
        ax.fill_between(curve.generations, curve.mean - curve.std, curve.mean + curve.std,
                         color=color, alpha=0.15)
    ax.set_xlabel("generation")
    ax.set_ylabel("best-so-far fitness (distance to target, m)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_sigma_curve(curves: dict[str, VariantCurve], out_path: Path, title: str) -> None:
    """Plot sigma over generations for the adaptive variants."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for variant in ADAPTIVE_VARIANTS:
        curve = curves.get(variant)
        if curve is None or curve.sigma_runs is None:
            continue
        color = VARIANT_COLOR[variant]
        for run in curve.sigma_runs:
            ax.plot(curve.generations, run, color=color, alpha=0.2, lw=0.7)
        ax.plot(curve.generations, _nan_quietly(np.nanmedian, curve.sigma_runs, axis=0),
                color=color, lw=2.2, label=f"{VARIANT_LABEL[variant]} (median, n={len(curve.seeds)})")
    ax.axhline(config.STATIC_SIGMA, color=VARIANT_COLOR["static"], ls="--", lw=1,
               label=VARIANT_LABEL["static"])
    ax.set_xlabel("generation")
    ax.set_ylabel(r"$\sigma$")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_success_rate_curve(curves: dict[str, VariantCurve], out_path: Path, title: str) -> None:
    """Plot mutation success rate over generations."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for variant in EA_VARIANTS:
        curve = curves.get(variant)
        if curve is None or curve.success_mean is None:
            continue
        ax.plot(curve.generations, curve.success_mean,
                label=f"{VARIANT_LABEL[variant]} (n={len(curve.seeds)})",
                color=VARIANT_COLOR[variant])
    ax.axhline(config.ADAPTIVE_TARGET_SUCCESS, color="grey", ls=":", lw=1, label="1/5 target")
    ax.set_xlabel("generation")
    ax.set_ylabel("success rate (mean over seeds)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_final_distance(curves: dict[str, VariantCurve], out_path: Path, title: str) -> None:
    """Plot distribution of final distances, a box per configuration with every run on top."""
    present = [v for v in VARIANT_ORDER if v in curves]
    data = [curves[v].per_seed_final for v in present]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    # showfliers=False: the points below already show every run
    ax.boxplot(data, showfliers=False, medianprops={"color": "black"})
    jitter = np.random.default_rng(0)   # fixed, so the figure is the same every time
    for i, (variant, values) in enumerate(zip(present, data), start=1):
        x = i + jitter.uniform(-0.12, 0.12, size=len(values))
        ax.scatter(x, values, s=14, color=VARIANT_COLOR[variant], alpha=0.8, zorder=3)
    ax.axhline(SATURATION_DISTANCE, color="grey", ls=":", lw=1,
               label=f"{SATURATION_DISTANCE} m (saturated below)")
    ax.set_xticks(range(1, len(present) + 1))
    ax.set_xticklabels([VARIANT_LABEL[v] for v in present], fontsize=8)
    ax.set_ylabel("final best distance to target (m)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Statistics
# --------------------------------------------------------------------------- #
def vargha_delaney_a12(a: list[float], b: list[float]) -> float:
    """P(a value from `a` < a value from `b`) + 0.5 * P(equal). A12 > 0.5 favours a (lower=better)."""
    wins = ties = 0
    for x in a:
        for y in b:
            if x < y:
                wins += 1
            elif x == y:
                ties += 1
    return (wins + 0.5 * ties) / (len(a) * len(b))


def effect_magnitude(a12: float) -> str:
    """Standard Vargha-Delaney thresholds (A = 0.56 / 0.64 / 0.71), as distance from 0.5."""
    distance = abs(a12 - 0.5)
    if distance < 0.06:
        return "negligible"
    if distance < 0.14:
        return "small"
    if distance < 0.21:
        return "medium"
    return "large"


def pairwise_tests(values: dict[str, list[float]], measure: str) -> list[dict]:
    """Every pair of configurations on one measure (lower is better for both measures)."""
    present = [v for v in VARIANT_ORDER if v in values]
    rows = []
    for i, a in enumerate(present):
        for b in present[i + 1 :]:
            fa, fb = values[a], values[b]
            # two-sided: neither configuration is assumed better going in
            u_stat, p_raw = scipy_stats.mannwhitneyu(fa, fb, alternative="two-sided")
            a12 = vargha_delaney_a12(fa, fb)
            rows.append({
                "measure": measure, "a": a, "b": b, "n_a": len(fa), "n_b": len(fb),
                "median_a": float(np.median(fa)), "median_b": float(np.median(fb)),
                "U": float(u_stat), "p_raw": float(p_raw),
                "A12_a_over_b": a12, "effect": effect_magnitude(a12),
            })
    return rows


def holm(p_values: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, in the original order."""
    order = p_values.argsort()
    m = len(p_values)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p_values[idx])
        running_max = max(running_max, value)
        adjusted[idx] = running_max
    return adjusted


# --------------------------------------------------------------------------- #
#  Convergence
# --------------------------------------------------------------------------- #
def convergence(
    frames: dict[str, list[pd.DataFrame]],
) -> tuple[pd.DataFrame, float | None, int]:
    """Evaluations each run needs to reach the target. Returns (table, target, censored value)."""
    ea_finals = [
        float(df["best_so_far"].iloc[-1])
        for variant in EA_VARIANTS for df in frames.get(variant, [])
    ]
    if not ea_finals:
        return pd.DataFrame(), None, 0
    target = max(ea_finals)
    budget = max(int(df["evaluations"].iloc[-1]) for dfs in frames.values() for df in dfs)
    censored = budget + 1

    rows = []
    for variant, dfs in frames.items():
        for df in dfs:
            reached = df.index[df["best_so_far"] <= target]
            evals = int(df["evaluations"].iloc[reached[0]]) if len(reached) else None
            rows.append({
                "variant": variant,
                "seed": int(df["seed"].iloc[0]),
                "target": target,
                "evaluations_to_target": evals,
                "generations_to_target": int(df["generation"].iloc[reached[0]]) if len(reached) else None,
            })
    return pd.DataFrame(rows), target, censored


# --------------------------------------------------------------------------- #
#  Per-seed analysis
# --------------------------------------------------------------------------- #
def longest_stall(df: pd.DataFrame) -> tuple[int, int, int]:
    """Find the longest run where best_so_far remains exactly unchanged. 
    Unlike an epsilon-based plateau, this requires no fitter individual to be found during the entire run.
    Returns (length_in_generations, start_generation, end_generation).
    """
    values = df["best_so_far"].to_numpy()
    gens = df["generation"].to_numpy()

    best_len, best_start_idx = 0, 0
    run_start_idx = 0
    for i in range(1, len(values)):
        if values[i] != values[run_start_idx]:
            run_len = i - run_start_idx
            if run_len > best_len:
                best_len, best_start_idx = run_len, run_start_idx
            run_start_idx = i
    run_len = len(values) - run_start_idx  # the run reaching to the end of the data
    if run_len > best_len:
        best_len, best_start_idx = run_len, run_start_idx

    return (
        best_len,
        int(gens[best_start_idx]),
        int(gens[best_start_idx + best_len - 1]),
    )


def last_improvement(df: pd.DataFrame, eps: float = IMPROVEMENT_EPS) -> int | None:
    """The last generation in which the best-so-far improved by more than `eps` (Methods).
    None if it never did after generation 0."""
    values = df["best_so_far"].to_numpy()
    gens = df["generation"].to_numpy()
    big = np.nonzero(values[:-1] - values[1:] > eps)[0]
    return int(gens[big[-1] + 1]) if len(big) else None


def per_seed_detail(frames: dict[str, list[pd.DataFrame]]) -> pd.DataFrame:
    """One row per (variant, seed), preserving details hidden by aggregation across seeds."""
    rows = []
    for variant, dfs in frames.items():
        for df in dfs:
            stall_len, stall_start, stall_end = longest_stall(df)
            last = last_improvement(df)
            row = {
                "variant": variant,
                "seed": int(df["seed"].iloc[0]),
                "n_generations": int(df["generation"].iloc[-1]),
                "final_best_so_far": round(float(df["best_so_far"].iloc[-1]), 4),
                "saturated": is_saturated(df),
                "last_improvement_generation": last if last is not None else "",
                "longest_stall_generations": stall_len,
                "longest_stall_start": stall_start,
                "longest_stall_end": stall_end,
            }
            sigma = df["sigma"].to_numpy(dtype=float)
            if variant in EA_VARIANTS and np.isfinite(sigma).any():
                peak = int(np.nanargmax(sigma))
                late = df["success_rate"].to_numpy(dtype=float)[-LATE_GENERATIONS:]
                row.update({
                    "final_sigma": float(sigma[-1]),
                    "peak_sigma": float(sigma[peak]),
                    "peak_sigma_generation": int(df["generation"].iloc[peak]),
                    "late_success_rate": float(_nan_quietly(np.nanmean, late)),
                })
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["variant", "seed"]).reset_index(drop=True)


def sigma_summary(detail: pd.DataFrame) -> pd.DataFrame:
    """Per adaptive configuration, with all runs and without the saturated ones."""
    rows = []
    for variant in ADAPTIVE_VARIANTS:
        runs = detail[detail["variant"] == variant]
        if runs.empty:
            continue
        for subset, part in (("all", runs), ("unsaturated", runs[~runs["saturated"]])):
            rows.append({
                "variant": variant,
                "runs": subset,
                "n": len(part),
                "median_final_sigma": round(float(part["final_sigma"].median()), 4) if len(part) else "",
                "median_peak_sigma": round(float(part["peak_sigma"].median()), 4) if len(part) else "",
                "median_peak_generation": float(part["peak_sigma_generation"].median()) if len(part) else "",
                f"mean_success_last_{LATE_GENERATIONS}": round(float(part["late_success_rate"].mean()), 4) if len(part) else "",
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  Summary table
# --------------------------------------------------------------------------- #
def summary_table(
    curves: dict[str, VariantCurve],
    conv_frame: pd.DataFrame,
    detail: pd.DataFrame,
) -> pd.DataFrame:
    """Create the summary table for each configuration."""
    rows = []
    for variant in VARIANT_ORDER:
        curve = curves.get(variant)
        if curve is None:
            continue
        finals = curve.per_seed_final
        hit = conv_frame.loc[
            (conv_frame["variant"] == variant) & conv_frame["evaluations_to_target"].notna(),
            "evaluations_to_target",
        ].astype(float)
        rows.append({
            "variant": variant,
            "n_seeds": len(finals),
            "best_overall": round(min(finals), 4),
            "median_final_best": round(float(np.median(finals)), 4),
            "mean_final_best": round(float(np.mean(finals)), 4),
            "std_final_best": round(sample_std(finals), 4),
            "evals_to_target_mean": round(float(hit.mean()), 1) if len(hit) else "",
            "evals_to_target_std": round(sample_std(hit.tolist()), 1) if len(hit) > 1 else "",
            "n_reached_target": int(len(hit)),
            "n_saturated": int(detail.loc[detail["variant"] == variant, "saturated"].sum()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  Plateau
# --------------------------------------------------------------------------- #
def plateau_generation(
    generations: np.ndarray,
    mean_curve: np.ndarray,
    eps: float,
    window: int,
) -> int | None:
    """Find the earliest generation where improvement stays within `eps` over the trailing window."""
    if len(mean_curve) <= window:
        return None

    improvement = mean_curve[:-window] - mean_curve[window:]  # non-negative, curve non-increasing
    settled = improvement < eps  # index i here means: generations[i + window] is settled

    for i in range(len(settled)):
        if settled[i:].all():
            return int(generations[i + window])
    return None


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def analyze(tag: str, body: str, plateau_eps: float, plateau_window: int) -> None:
    print(f"tag={tag} body={body}")
    frames: dict[str, list[pd.DataFrame]] = {}
    curves: dict[str, VariantCurve] = {}

    for variant in VARIANT_ORDER:
        dfs = [load_seed_csv(f) for f in seed_dirs(tag, body, variant)]
        curve = aggregate(variant, dfs)
        if curve is None:
            print(f"  (no seeds found for variant={variant!r})")
            continue
        frames[variant] = dfs
        curves[variant] = curve
        print(f"  [{variant}] {len(curve.seeds)} seed(s): {curve.seeds}")

    if not curves:
        print("Nothing to analyze: no results found for this tag/body.")
        return

    out_dir = config.RESULTS_DIR / tag / body / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    title = f"{body}, tag={tag}"

    # per-run detail first: it decides which runs are saturated
    detail = per_seed_detail(frames)
    detail.to_csv(out_dir / "per_seed_detail.csv", index=False)

    plot_fitness_curve(curves, out_dir / "fitness_curve.png", title)
    plot_sigma_curve(curves, out_dir / "sigma_curve.png", title)
    plot_success_rate_curve(curves, out_dir / "success_rate_curve.png", title)
    plot_final_distance(curves, out_dir / "final_distance.png", title)

    # plot_sigma_curve and plot_success_rate_curve again without the saturated runs
    n_saturated_ea = int(detail.loc[detail["variant"].isin(EA_VARIANTS), "saturated"].sum())
    if n_saturated_ea:
        unsat_curves = {}
        for variant in EA_VARIANTS:
            if variant not in frames:
                continue
            kept = [df for df in frames[variant] if not is_saturated(df)]
            curve = aggregate(variant, kept)
            if curve is not None:
                unsat_curves[variant] = curve
        plot_sigma_curve(unsat_curves, out_dir / "sigma_curve_unsaturated.png",
                         f"{title}, without saturated runs")
        plot_success_rate_curve(unsat_curves, out_dir / "success_rate_curve_unsaturated.png",
                                f"{title}, without saturated runs")
    sig_frame = sigma_summary(detail)
    sig_frame.to_csv(out_dir / "sigma_summary.csv", index=False)

    conv_frame, target, censored = convergence(frames)
    conv_frame.to_csv(out_dir / "convergence.csv", index=False)

    final_values = {v: c.per_seed_final for v, c in curves.items()}
    conv_values = {
        v: conv_frame.loc[conv_frame["variant"] == v, "evaluations_to_target"]
              .fillna(censored).astype(float).tolist()
        for v in curves
    }
    stats_frame = pd.DataFrame(
        pairwise_tests(final_values, "final_distance")
        + pairwise_tests(conv_values, "evaluations_to_target")
    )
    if not stats_frame.empty:
        stats_frame["p_holm"] = holm(stats_frame["p_raw"].to_numpy())
    stats_frame.to_csv(out_dir / "stats.csv", index=False)

    table = summary_table(curves, conv_frame, detail)
    table.to_csv(out_dir / "summary_table.csv", index=False)

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
                "mean": float(np.mean(curve.per_seed_final)),
                "std": sample_std(curve.per_seed_final),
                "median": float(np.median(curve.per_seed_final)),
                "n_seeds": len(curve.seeds),
            }
            for variant, curve in curves.items()
        },
        "convergence_target": target,
        "convergence_not_reached_counted_as": censored,
        "pairwise_stats": stats_frame.to_dict(orient="records"),
        "sigma_summary": sig_frame.to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nWritten to {out_dir}")
    print("\nfinal best-so-far (mean +/- sample std across seeds):")
    for variant, curve in curves.items():
        print(f"  {variant:>13}: {np.mean(curve.per_seed_final):.4f} "
              f"+/- {sample_std(curve.per_seed_final):.4f}  (n={len(curve.seeds)})")
    print(f"\nconvergence target (worst final best-so-far of any EA run): {target}")
    print(f"saturated runs (final distance <= {SATURATION_DISTANCE} m): "
          + ", ".join(f"{v} {int(detail.loc[detail['variant'] == v, 'saturated'].sum())}"
                      for v in curves))
    print(f"\nplateau generation (eps = {plateau_eps:.4f}, window = {plateau_window}):")
    for variant, gen in plateaus.items():
        print(f"  {variant:>13}: {gen if gen is not None else 'not reached in this run'}")
    if not stats_frame.empty:
        print(f"\npairwise stats (Holm across all {len(stats_frame)} tests):")
        print(stats_frame.to_string(index=False))
    if not sig_frame.empty:
        print("\nsigma summary:")
        print(sig_frame.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarise run.py's results: the four plots, stats, convergence, plateau.",
    )
    parser.add_argument("--tag", default="main", help="Results tag to analyze. Default: %(default)s.")
    parser.add_argument(
        "--body", default=config.BODY_NAME,
        help="Body subfolder to analyze. Default: %(default)s (config.BODY_NAME).",
    )
    parser.add_argument(
        "--plateau-eps", type=float, default=0.02,
        help="Plateau threshold, in fitness units (metres). Default: %(default)s.",
    )
    parser.add_argument(
        "--plateau-window", type=int, default=20,
        help="Generations over which the improvement must stay below --plateau-eps. "
             "Default: %(default)s.",
    )
    args = parser.parse_args()
    analyze(args.tag, args.body, args.plateau_eps, args.plateau_window)


if __name__ == "__main__":
    main()
