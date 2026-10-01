"""Turn run.py's CSVs into the report's figures, stats table and plateau point.

run.py only RUNS experiments, one (tag, body, variant, seed) at a time, each into its own
fitness_overview.csv. Nothing else reads those back and compares them. This file does that:

    uv run assignments/assignment_2/group_28/analyze.py --tag main --body gecko

WHAT IT PRODUCES, under results/<tag>/<body>/analysis/
--------------------------------------------------------
    fitness_curve.png        best-so-far fitness vs generation, mean +/- std across seeds,
                             one line per variant (static / adaptive / baseline) - the
                             brief's required "line-plot across generations, showing the
                             average/std of fitness over your independent runs"
    sigma_curve.png          sigma vs generation, static vs adaptive
    success_rate_curve.png   mutation success rate vs generation, static vs adaptive, with
                             the 1/5 target marked - together with sigma_curve.png, the A1
                             Figure 2 equivalent: shows WHY one variant did or didn't win
    summary_table.csv        one row per variant: n_seeds, best_overall, mean/std final best
                             fitness, mean/std generations-to-threshold, n_reached_threshold -
                             the flat, report-ready table (A1's summarize.py build_table)
    per_seed_detail.csv      one row per (variant, seed): this seed's OWN plateau generation
                             and its longest literal stall (a run of generations with not a
                             single fitter individual found). The mean-across-seeds view in
                             fitness_curve.png and summary_table.csv can hide a seed that
                             stalls badly while others keep improving, because the average
                             keeps moving; this file is where that shows up directly.
    stats.csv                final-fitness comparison: Mann-Whitney U, Holm-corrected p,
                             Vargha-Delaney A12 and its effect-size label, for every pair of
                             variants present
    convergence.csv          per seed, generations needed to reach the shared target fitness
                             (A1's definition: the worst final best-so-far of any compared run,
                             so every configuration can in principle reach it)
    summary.json             everything above as numbers, plus the plateau generation per
                             variant - read this to decide NUM_GENERATIONS for the real runs

A1 wrote its plots and tables flat into config.RESULTS_DIR, because it only ever had one
experimental configuration running at a time. A2 instead has several pilots (body, sigma,
plateau) plus the real experiment, all needing to stay apart, which is what tag and body
nest the output by here.

Holm correction is applied as a SINGLE family across every pairwise comparison this call
produces (at most 3: static-adaptive, static-baseline, adaptive-baseline). A1 corrected per
operator instead, because it ran the same comparison repeatedly across several mutation
operators - several independent families of tests. A2 has no operator axis (mutation.py:
"no crossover" - mutation is the only variation operator, period), so there is only ever one
family here, and a single global correction is the right one, not an oversight of A1's
per-category loop.

BASELINE: WHY best_fitness NEEDS A RUNNING MINIMUM
----------------------------------------------------
ea.run_generation throws the whole population away each generation for "baseline" (random
search) rather than keeping the best found so far. So its best_fitness column, as logged by
run.py, is just THAT generation's minimum - not cumulative, and not monotonic. The EA variants
do not have this problem: elitism keeps the best individual, so their best_fitness column is
already non-increasing by construction. Taking a running (cumulative) minimum per seed, BEFORE
averaging across seeds, makes every variant's curve mean the same thing: "the best this
configuration has found by generation g". This is applied to every variant for safety, even
though it only changes the baseline's numbers.

PLATEAU DETECTION
------------------
A variant is "on its plateau" from the first generation g such that every later improvement,
over the rest of the run, is smaller than --plateau-eps. Concretely: take the running-min best-
so-far curve (mean across seeds), and find the earliest g where
    curve[g] - min(curve[g:]) < plateau-eps
holds for the rest of the curve. This is the automated version of "run until your fitness curve
plateaus" (the brief's own phrase) - eyeballing a plot is still worth doing too, but this number
is what goes into NUM_GENERATIONS so the choice is reproducible rather than a judgement call
made once and forgotten.
"""

# Standard library
import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

# Third-party libraries
import matplotlib
matplotlib.use("Agg")  # headless: never try to open a window, e.g. over SSH or in a pool worker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

import config

VARIANT_ORDER = ("static", "adaptive", "baseline")
VARIANT_COLOR = {"static": "#1f77b4", "adaptive": "#d62728", "baseline": "#7f7f7f"}


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
    """One seed's fitness_overview.csv, with the running-minimum columns added.

    na_values are not passed explicitly: pandas' default na recognition already treats the
    literal string "nan" (written by run.py.make_row for sigma/success_rate on generations
    where neither applies) as NaN, so sigma and success_rate come back as proper floats with
    gaps rather than strings.
    """
    df = pd.read_csv(folder / config.RESULT_FILE_NAME)
    df = df.sort_values("generation").reset_index(drop=True)
    df["best_so_far"] = df["best_fitness"].cummin()
    return df


def load_variant(tag: str, body: str, variant: str) -> list[pd.DataFrame]:
    """One DataFrame per seed. Skips nothing silently: an empty variant is reported, not hidden."""
    frames = [load_seed_csv(folder) for folder in seed_dirs(tag, body, variant)]
    if not frames:
        print(f"  (no seeds found for variant={variant!r})")
    return frames


# --------------------------------------------------------------------------- #
#  Aggregation across seeds
# --------------------------------------------------------------------------- #
@dataclass
class VariantCurve:
    variant: str
    generations: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    per_seed_final: list[float]          # best_so_far at each seed's last generation
    seeds: list[int] = field(default_factory=list)
    sigma_mean: np.ndarray | None = None
    success_mean: np.ndarray | None = None


def aggregate(variant: str, frames: list[pd.DataFrame]) -> VariantCurve | None:
    if not frames:
        return None

    # Seeds can differ in length if a pilot was interrupted; align on the shortest so every
    # seed contributes to every generation plotted, rather than padding with guesses.
    min_len = min(len(f) for f in frames)
    if min_len < max(len(f) for f in frames):
        print(f"  [{variant}] seeds have different lengths; truncating all to {min_len} rows")

    stacked = np.stack([f["best_so_far"].to_numpy()[:min_len] for f in frames])
    generations = frames[0]["generation"].to_numpy()[:min_len]
    # Sample std (ddof=1), matching A1: these seeds are a SAMPLE from which we are estimating
    # the std of the underlying process, not its entire population. ddof=1 is undefined
    # (divides by zero) for a single seed, so fall back to an all-zero band there, as A1 did.
    curve_std = np.nanstd(stacked, axis=0, ddof=1) if stacked.shape[0] > 1 else np.zeros(min_len)

    sigma_mean = success_mean = None
    if frames[0]["sigma"].notna().any():
        sigma_mean = np.nanmean(
            np.stack([f["sigma"].to_numpy()[:min_len] for f in frames]), axis=0,
        )
    if frames[0]["success_rate"].notna().any():
        success_mean = np.nanmean(
            np.stack([f["success_rate"].to_numpy()[:min_len] for f in frames]), axis=0,
        )

    return VariantCurve(
        variant=variant,
        generations=generations,
        mean=np.nanmean(stacked, axis=0),
        std=curve_std,
        per_seed_final=stacked[:, -1].tolist(),
        seeds=[int(f["seed"].iloc[0]) for f in frames],
        sigma_mean=sigma_mean,
        success_mean=success_mean,
    )


# --------------------------------------------------------------------------- #
#  Plots
# --------------------------------------------------------------------------- #
def plot_fitness_curve(curves: dict[str, VariantCurve], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for variant in VARIANT_ORDER:
        curve = curves.get(variant)
        if curve is None:
            continue
        color = VARIANT_COLOR[variant]
        ax.plot(curve.generations, curve.mean, label=f"{variant} (n={len(curve.seeds)})",
                color=color)
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
    """Sigma vs generation, static vs adaptive. Split from the success-rate plot (rather than
    two subplots of one figure) so either can be used alone as the one mutation-mechanism
    figure the report's page budget allows, without cropping - matching A1's one-plot-per-file
    convention.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for variant in ("static", "adaptive"):
        curve = curves.get(variant)
        if curve is None or curve.sigma_mean is None:
            continue
        ax.plot(curve.generations, curve.sigma_mean, label=variant, color=VARIANT_COLOR[variant])
    ax.axhline(config.ADAPTIVE_MIN_SIGMA, color="grey", ls=":", lw=1, label="sigma floor")
    ax.set_xlabel("generation")
    ax.set_ylabel("sigma (mean over seeds)")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_success_rate_curve(curves: dict[str, VariantCurve], out_path: Path, title: str) -> None:
    """Mutation success rate vs generation, static vs adaptive, with the 1/5 target marked.
    The static variant measures this too and never acts on it, which is what makes the two
    curves comparable (mutation.py's own docstring makes this point)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for variant in ("static", "adaptive"):
        curve = curves.get(variant)
        if curve is None or curve.success_mean is None:
            continue
        ax.plot(curve.generations, curve.success_mean, label=variant, color=VARIANT_COLOR[variant])
    ax.axhline(config.ADAPTIVE_TARGET_SUCCESS, color="grey", ls=":", lw=1, label="1/5 target")
    ax.set_xlabel("generation")
    ax.set_ylabel("success rate (mean over seeds)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Statistics: final-fitness comparison
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


def pairwise_stats(curves: dict[str, VariantCurve]) -> pd.DataFrame:
    present = [v for v in VARIANT_ORDER if v in curves]
    pairs = [(a, b) for i, a in enumerate(present) for b in present[i + 1 :]]
    rows = []
    for a, b in pairs:
        fa, fb = curves[a].per_seed_final, curves[b].per_seed_final
        # two-sided: neither variant is assumed better going in
        u_stat, p_raw = scipy_stats.mannwhitneyu(fa, fb, alternative="two-sided")
        a12 = vargha_delaney_a12(fa, fb)
        rows.append({
            "a": a, "b": b, "n_a": len(fa), "n_b": len(fb),
            "mean_a": float(np.mean(fa)), "mean_b": float(np.mean(fb)),
            "U": float(u_stat), "p_raw": float(p_raw),
            "A12_a_over_b": a12, "effect": effect_magnitude(a12),
        })

    frame = pd.DataFrame(rows)
    if not frame.empty:
        # Holm-Bonferroni across the comparisons made on this body/tag together.
        order = frame["p_raw"].to_numpy().argsort()
        m = len(frame)
        adjusted = np.empty(m)
        running_max = 0.0
        for rank, idx in enumerate(order):
            value = min(1.0, (m - rank) * frame["p_raw"].to_numpy()[idx])
            running_max = max(running_max, value)
            adjusted[idx] = running_max
        frame["p_holm"] = adjusted
    return frame


# --------------------------------------------------------------------------- #
#  Convergence speed (A1's definition, ported)
# --------------------------------------------------------------------------- #
# Computed inline in analyze(), where the seed CSVs are already loaded once: generations
# needed to first reach a shared target, where target = the worst (highest) final
# best-so-far value across every seed of every variant being compared, so the target is
# reachable in principle by every configuration. Using each run's own final fitness as its
# target would instead reward a configuration that settled early at a poor value, by making
# it look like it "converged fastest" (A1's mistake, fixed here as it was there).


# --------------------------------------------------------------------------- #
#  Flat summary table (A1's build_table / summarize_combination, ported)
# --------------------------------------------------------------------------- #
def summary_table(curves: dict[str, VariantCurve], conv_frame: pd.DataFrame) -> pd.DataFrame:
    """One row per variant: exactly what goes straight into the report's summary table."""
    rows = []
    for variant in VARIANT_ORDER:
        curve = curves.get(variant)
        if curve is None:
            continue
        finals = curve.per_seed_final
        hit = conv_frame.loc[
            (conv_frame["variant"] == variant) & conv_frame["generations_to_target"].notna(),
            "generations_to_target",
        ]
        rows.append({
            "variant": variant,
            "n_seeds": len(finals),
            "best_overall": round(min(finals), 4),
            "mean_final_best": round(float(np.mean(finals)), 4),
            # sample std: undefined (ddof=1 divides by zero) for a single seed
            "std_final_best": round(float(np.std(finals, ddof=1)), 4) if len(finals) > 1 else 0.0,
            "gens_to_threshold_mean": round(float(hit.mean()), 1) if len(hit) else "",
            "gens_to_threshold_std": round(float(hit.std(ddof=1)), 1) if len(hit) > 1 else "",
            "n_reached_threshold": int(len(hit)),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  Per-seed stalls and per-seed plateau
# --------------------------------------------------------------------------- #
def longest_stall(df: pd.DataFrame) -> tuple[int, int, int]:
    """The longest run of consecutive generations with a literally UNCHANGED best_so_far -
    not "improved by less than eps" (plateau_generation's question), but "found not a single
    fitter individual in this span". This is the harder, more direct diagnostic: it is what a
    run like "frozen at 0.7996 from generation 50 to 80" looks like in the data, and it is
    visible on a single seed even when the mean-across-seeds curve still looks like it is
    moving, because the other seeds are not stalled at the same time.

    Returns (length_in_generations, start_generation, end_generation) for the longest such run.
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


def per_seed_detail(
    seed_folders: dict[str, list[Path]],
    plateau_eps: float,
    plateau_window: int,
) -> pd.DataFrame:
    """One row per (variant, seed): this seed's own plateau generation and longest stall.

    Deliberately separate from VariantCurve, which only holds the mean and std ACROSS seeds -
    exactly the view that hides a case like one seed stalling for 30 generations while another
    is still improving, since the mean keeps moving and nothing in the aggregate flags it.
    """
    rows = []
    for variant, folders in seed_folders.items():
        for folder in folders:
            df = load_seed_csv(folder)
            seed = int(df["seed"].iloc[0])
            gen = plateau_generation(
                df["generation"].to_numpy(), df["best_so_far"].to_numpy(),
                plateau_eps, plateau_window,
            )
            stall_len, stall_start, stall_end = longest_stall(df)
            rows.append({
                "variant": variant,
                "seed": seed,
                "n_generations": int(df["generation"].iloc[-1]),
                "final_best_so_far": round(float(df["best_so_far"].iloc[-1]), 4),
                "plateau_generation": gen if gen is not None else "",
                "longest_stall_generations": stall_len,
                "longest_stall_start": stall_start,
                "longest_stall_end": stall_end,
            })
    return pd.DataFrame(rows).sort_values(["variant", "seed"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
#  Plateau
# --------------------------------------------------------------------------- #
def plateau_generation(
    generations: np.ndarray,
    mean_curve: np.ndarray,
    eps: float,
    window: int,
) -> int | None:
    """Earliest generation g from which the curve stays within `eps` of itself `window`
    generations earlier, all the way to the end of the run. The plan's own definition: "no
    improvement in best fitness over N generations" (N = window here).

    Comparing each point against the GLOBAL minimum of everything after it (an earlier,
    simpler version of this function did that) is wrong: for any curve that is still
    steadily declining, that minimum is always the curve's own final point, so the last
    couple of generations trivially look "settled" just because there is almost nothing left
    of the curve to improve against - regardless of whether it had actually levelled off. A
    FIXED trailing window avoids this: it asks "how much did it improve over the last N
    generations", the same question at every point along the curve, start to finish.

    Returns None if the curve never settles within eps for the rest of the run (still
    improving, by more than eps every window, when the data runs out - a real finding, not a
    bug: it means the budget was too short to see where this configuration levels off).
    """
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
    curves: dict[str, VariantCurve] = {}
    seed_folders: dict[str, list[Path]] = {}

    for variant in VARIANT_ORDER:
        folders = seed_dirs(tag, body, variant)
        frames = [load_seed_csv(f) for f in folders]
        curve = aggregate(variant, frames)
        if curve is not None:
            curves[variant] = curve
            seed_folders[variant] = folders
            print(f"  [{variant}] {len(curve.seeds)} seed(s): {curve.seeds}")

    if not curves:
        print("Nothing to analyze: no results found for this tag/body.")
        return

    out_dir = config.RESULTS_DIR / tag / body / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    title = f"{body}, tag={tag}"
    plot_fitness_curve(curves, out_dir / "fitness_curve.png", title)
    if any(c.sigma_mean is not None for c in curves.values()):
        plot_sigma_curve(curves, out_dir / "sigma_curve.png", title)
    if any(c.success_mean is not None for c in curves.values()):
        plot_success_rate_curve(curves, out_dir / "success_rate_curve.png", title)

    stats_frame = pairwise_stats(curves)
    stats_frame.to_csv(out_dir / "stats.csv", index=False)

    conv_rows = []
    all_finals = [f for curve in curves.values() for f in curve.per_seed_final]
    target = max(all_finals) if all_finals else None
    for variant, folders in seed_folders.items():
        for folder in folders:
            seed_df = load_seed_csv(folder)
            reached = seed_df.index[seed_df["best_so_far"] <= target]
            gen = int(seed_df["generation"].iloc[reached[0]]) if len(reached) else None
            conv_rows.append({
                "variant": variant,
                "seed": int(seed_df["seed"].iloc[0]),
                "target": target,
                "generations_to_target": gen,
            })
    conv_frame = pd.DataFrame(conv_rows)
    conv_frame.to_csv(out_dir / "convergence.csv", index=False)

    table = summary_table(curves, conv_frame)
    table.to_csv(out_dir / "summary_table.csv", index=False)

    detail_frame = per_seed_detail(seed_folders, plateau_eps, plateau_window)
    detail_frame.to_csv(out_dir / "per_seed_detail.csv", index=False)

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
        "final_best_so_far": {
            variant: {
                "mean": float(np.mean(curve.per_seed_final)),
                "std": float(np.std(curve.per_seed_final)),
                "n_seeds": len(curve.seeds),
            }
            for variant, curve in curves.items()
        },
        "pairwise_stats": stats_frame.to_dict(orient="records"),
        "convergence_target": target,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nWritten to {out_dir}")
    print("\nfinal best-so-far (mean +/- std across seeds):")
    for variant, curve in curves.items():
        print(f"  {variant:>9}: {np.mean(curve.per_seed_final):.4f} "
              f"+/- {np.std(curve.per_seed_final):.4f}  (n={len(curve.seeds)})")
    print(f"\nplateau generation (eps = {plateau_eps:.4f}, window = {plateau_window}):")
    for variant, gen in plateaus.items():
        print(f"  {variant:>9}: {gen if gen is not None else 'not reached in this run'}")
    if not stats_frame.empty:
        print("\npairwise stats:")
        print(stats_frame.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarise run.py's results: fitness curve, sigma curve, stats, plateau.",
    )
    parser.add_argument("--tag", default="main", help="Results tag to analyze. Default: %(default)s.")
    parser.add_argument(
        "--body", default=None,
        help="Body subfolder to analyze. Default: config.BUILD_BODY.__name__.",
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
    body = args.body or config.BUILD_BODY.__name__
    analyze(args.tag, body, args.plateau_eps, args.plateau_window)


if __name__ == "__main__":
    main()