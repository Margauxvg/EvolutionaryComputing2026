# Extra per-seed analysis on top of plots.py (not needed for the report figures). Run from the project root:
#   uv run assignments/assignment_2/group_28/extras/diagnostics.py --tag main
# Writes per_seed_detail.csv, sigma_summary.csv and the sigma / success rate plots without the
# saturated runs to results/<tag>/<body>/analysis/.

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
import plots

IMPROVEMENT_EPS = 0.01  # smallest improvement (m) that counts for last_improvement
LATE_GENERATIONS = 50  # success rate is averaged over this many final generations


def longest_stall(df: pd.DataFrame) -> tuple[int, int, int]:
    """Longest stretch of generations where best_so_far did not change at all.
    Returns (length, first generation, last generation)."""
    values = df["best_so_far"].to_numpy()
    gens = df["generation"].to_numpy()

    best_len, best_start = 0, 0
    start = 0
    for i in range(1, len(values)):
        if values[i] != values[start]:
            if i - start > best_len:
                best_len, best_start = i - start, start
            start = i
    if len(values) - start > best_len:  # the stretch that lasts until the end
        best_len, best_start = len(values) - start, start

    return best_len, int(gens[best_start]), int(gens[best_start + best_len - 1])


def last_improvement(df: pd.DataFrame, eps: float = IMPROVEMENT_EPS) -> int | None:
    """Last generation where best_so_far improved by more than eps."""
    values = df["best_so_far"].to_numpy()
    gens = df["generation"].to_numpy()
    big = np.nonzero(values[:-1] - values[1:] > eps)[0]
    return int(gens[big[-1] + 1]) if len(big) else None


def per_seed_detail(runs: dict[str, list[pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for variant, dfs in runs.items():
        for df in dfs:
            stall_len, stall_start, stall_end = longest_stall(df)
            last = last_improvement(df)
            row = {
                "variant": variant,
                "seed": int(df["seed"].iloc[0]),
                "n_generations": int(df["generation"].iloc[-1]),
                "final_best_so_far": round(float(df["best_so_far"].iloc[-1]), 4),
                "saturated": plots.is_saturated(df),
                "last_improvement_generation": last if last is not None else "",
                "longest_stall_generations": stall_len,
                "longest_stall_start": stall_start,
                "longest_stall_end": stall_end,
            }
            sigma = df["sigma"].to_numpy(dtype=float)
            if variant in plots.EA_VARIANTS and np.isfinite(sigma).any():
                peak = int(np.nanargmax(sigma))
                late = df["success_rate"].to_numpy(dtype=float)[-LATE_GENERATIONS:]
                row.update({
                    "final_sigma": float(sigma[-1]),
                    "peak_sigma": float(sigma[peak]),
                    "peak_sigma_generation": int(df["generation"].iloc[peak]),
                    "late_success_rate": float(plots.ignore_nan_warnings(np.nanmean, late)),
                })
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["variant", "seed"]).reset_index(drop=True)


def sigma_summary(detail: pd.DataFrame) -> pd.DataFrame:
    """Per adaptive variant: medians over all runs, and over the unsaturated runs only."""
    rows = []
    for variant in plots.ADAPTIVE_VARIANTS:
        variant_runs = detail[detail["variant"] == variant]
        if variant_runs.empty:
            continue
        for subset, part in (("all", variant_runs), ("unsaturated", variant_runs[~variant_runs["saturated"]])):
            empty = len(part) == 0
            rows.append({
                "variant": variant,
                "runs": subset,
                "n": len(part),
                "median_final_sigma": "" if empty else round(float(part["final_sigma"].median()), 4),
                "median_peak_sigma": "" if empty else round(float(part["peak_sigma"].median()), 4),
                "median_peak_generation": "" if empty else float(part["peak_sigma_generation"].median()),
                f"mean_success_last_{LATE_GENERATIONS}": "" if empty else round(float(part["late_success_rate"].mean()), 4),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-seed diagnostics.")
    parser.add_argument("--tag", default="main")
    parser.add_argument("--body", default=config.BODY_NAME)
    args = parser.parse_args()

    runs = {variant: plots.load_seed_runs(args.tag, args.body, variant) for variant in plots.VARIANT_ORDER}
    runs = {variant: dfs for variant, dfs in runs.items() if dfs}
    output_dir = config.RESULTS_DIR / args.tag / args.body / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    title = f"{args.body}, tag={args.tag}"

    detail = per_seed_detail(runs)
    detail.to_csv(output_dir / "per_seed_detail.csv", index=False)
    sigmas = sigma_summary(detail)
    sigmas.to_csv(output_dir / "sigma_summary.csv", index=False)

    # The sigma and success rate plots again, without the runs that (almost) reached the target
    if detail.loc[detail["variant"].isin(plots.EA_VARIANTS), "saturated"].any():
        unsaturated = {}
        for variant in plots.EA_VARIANTS:
            kept = [df for df in runs.get(variant, []) if not plots.is_saturated(df)]
            curve = plots.make_curve(variant, kept)
            if curve is not None:
                unsaturated[variant] = curve
        plots.plot_sigma(unsaturated, output_dir / "sigma_curve_unsaturated.png", f"{title}, without saturated runs")
        plots.plot_success_rate(unsaturated, output_dir / "success_rate_curve_unsaturated.png",
                                f"{title}, without saturated runs")

    print(detail.to_string(index=False))
    print()
    print(sigmas.to_string(index=False))


if __name__ == "__main__":
    main()
