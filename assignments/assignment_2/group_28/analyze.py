"""Turn run.py's CSVs into the report's figures, stats table and plateau point.

run.py only RUNS experiments, one (tag, body, variant, seed) at a time, each into its own
fitness_overview.csv. Nothing else reads those back and compares them. This file does that:

    uv run assignments/assignment_2/group_28/analyze.py --tag main

Everything below follows Methods ("Experimental procedure"): four configurations, two outcome
measures (final fitness and convergence speed), four plots, Mann-Whitney U with Holm and
Vargha-Delaney A12, and the separate handling of runs that end close to the target.

WHAT IT PRODUCES, under results/<tag>/<body>/analysis/
--------------------------------------------------------
    fitness_curve.png        plot (i): best-so-far fitness vs generation, mean +/- std across
                             seeds, one line per configuration - the brief's required
                             "line-plot across generations, showing the average/std of fitness
                             over your independent runs"
    sigma_curve.png          plot (ii): sigma vs generation for both adaptive configurations,
                             every run as a thin line plus the median as a thick one
    success_rate_curve.png   plot (iii): mutation success rate vs generation for all three EA
                             configurations, with the 1/5 target marked - together with
                             sigma_curve.png, the A1 Figure 2 equivalent: shows WHY one variant
                             did or didn't win
    final_distance.png       plot (iv): the distribution of final distances per configuration
    *_unsaturated.png        plots (ii) and (iii) again without the saturated runs (see below);
                             only written if there are any
    summary_table.csv        one row per configuration: n_seeds, best_overall, mean/std final
                             best fitness, mean/std evaluations-to-target, n_reached_target,
                             n_saturated - the flat, report-ready table (A1's summarize.py
                             build_table)
    sigma_summary.csv        per adaptive configuration, with and without the saturated runs:
                             median final sigma, median peak sigma and when it happened, and the
                             mean success rate over the last 50 generations
    per_seed_detail.csv      one row per (variant, seed): final distance, whether the run is
                             saturated, the last generation in which its best improved by more
                             than 0.01 m (the per-run number Methods promises), its longest
                             literal stall, and its sigma story. The mean-across-seeds view in
                             fitness_curve.png and summary_table.csv can hide a seed that stalls
                             badly while others keep improving, because the average keeps
                             moving; this file is where that shows up directly.
    stats.csv                both outcome measures, every pair of configurations: Mann-Whitney
                             U, Holm-corrected p, Vargha-Delaney A12 and its effect-size label
    convergence.csv          per seed, evaluations needed to reach the shared target fitness
    summary.json             everything above as numbers, plus the plateau generation per
                             configuration (to check afterwards that 250 generations was enough)

A1 wrote its plots and tables flat into config.RESULTS_DIR, because it only ever had one
experimental configuration running at a time. A2 instead has several pilots plus the real
experiment, all needing to stay apart, which is what tag and body nest the output by here.

STATISTICS
----------
Four configurations give six pairs, which is exactly the list in Methods: fixed vs each
adaptive (2), adaptive vs adaptive (1), each EA vs the baseline (3). Each pair is tested on both
outcome measures, so 12 tests. Holm is applied once across all 12 ("across these tests" in
Methods). A1 corrected per operator instead, because it ran the same comparison repeatedly across
several mutation operators - several independent families of tests. A2 has no operator axis, so
there is one family. Both measures are lower-is-better, so A12 > 0.5 always favours `a`.

CONVERGENCE SPEED
-----------------
The number of EVALUATIONS a run needs to first reach one shared target fitness. The target is
the worst final best-so-far of any EA run (static, adaptive, adaptive_cap1), so every EA run
reaches it by construction. The baseline is left out of the target on purpose: its worst run is
far worse than any EA run, and including it would make the target so easy that every EA run
"converges" in the first few generations. A baseline run that never reaches the target is
counted as worse than any run that did (evaluations = budget + 1), which the rank-based
Mann-Whitney test handles correctly; convergence.csv leaves it empty.
Using each run's own final fitness as its target would instead reward a configuration that
settled early at a poor value, by making it look like it "converged fastest" (A1's mistake,
fixed here as it was there).

SATURATED RUNS
--------------
A run whose final distance is at most SATURATION_DISTANCE (0.15 m) has nearly reached the target
and can hardly improve any more. That by itself lowers its success rate and makes the 1/5 rule
shrink sigma, which would look like the rule "deciding" the search is over. These runs are
counted per configuration, and the sigma and success-rate analysis is repeated without them.

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
Not one of the outcome measures; a check on the run length. A configuration is "on its plateau"
from the first generation g after which the mean best-so-far curve never again improves by more
than --plateau-eps within --plateau-window generations. If that point is well before 250, the
budget was long enough; if it is "not reached", the curves were still going down at the end and
the report has to say so.
"""

# Standard library
import argparse
import json
import warnings
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

SATURATION_DISTANCE = 0.15  # metres; Methods: "Runs that end within 0.15 m of the target"
IMPROVEMENT_EPS = 0.01      # metres; Methods: "the last generation in which its best fitness
                            # improved by more than 0.01 m"
LATE_GENERATIONS = 50       # window for "late" success rate in sigma_summary.csv


def _nan_quietly(func, *args, **kwargs):
    """np.nanmean / np.nanmedian without the 'Mean of empty slice' warning. Generation 0 has no
    success rate in any run (nothing has been mutated yet), so an all-NaN column is expected."""
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


def is_saturated(df: pd.DataFrame) -> bool:
    return float(df["best_so_far"].iloc[-1]) <= SATURATION_DISTANCE


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
    sigma_runs: np.ndarray | None = None  # one row per seed, for plot (ii)
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
#  Plots
# --------------------------------------------------------------------------- #
def plot_fitness_curve(curves: dict[str, VariantCurve], out_path: Path, title: str) -> None:
    """Plot (i)."""
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
    """Plot (ii): sigma vs generation for both adaptive configurations, every run plus the median.

    The median rather than the mean because sigma moves multiplicatively and spans orders of
    magnitude between runs; one run sitting at the bound would drag a mean far away from what
    a typical run does. Split from the success-rate plot (rather than two subplots of one
    figure) so either can be used alone if the page budget only allows one, matching A1's
    one-plot-per-file convention.
    """
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
    """Plot (iii): mutation success rate vs generation, all EA configurations, 1/5 target marked.
    The static variant measures this too and never acts on it, which is what makes the curves
    comparable (mutation.py's own docstring makes this point)."""
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
    """Plot (iv): distribution of final distances, a box per configuration with every run on top."""
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
#  Statistics: pairwise comparisons on both outcome measures
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
#  Convergence speed (A1's definition, ported; see the module docstring)
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
#  Per-seed detail
# --------------------------------------------------------------------------- #
def longest_stall(df: pd.DataFrame) -> tuple[int, int, int]:
    """The longest run of consecutive generations with a literally UNCHANGED best_so_far -
    not "improved by less than eps", but "found not a single fitter individual in this span".
    This is the harder, more direct diagnostic: it is what a run like "frozen at 0.7996 from
    generation 50 to 80" looks like in the data, and it is visible on a single seed even when
    the mean-across-seeds curve still looks like it is moving, because the other seeds are not
    stalled at the same time.

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


def last_improvement(df: pd.DataFrame, eps: float = IMPROVEMENT_EPS) -> int | None:
    """The last generation in which the best-so-far improved by more than `eps` (Methods).
    None if it never did after generation 0."""
    values = df["best_so_far"].to_numpy()
    gens = df["generation"].to_numpy()
    big = np.nonzero(values[:-1] - values[1:] > eps)[0]
    return int(gens[big[-1] + 1]) if len(big) else None


def per_seed_detail(frames: dict[str, list[pd.DataFrame]]) -> pd.DataFrame:
    """One row per (variant, seed). Deliberately separate from VariantCurve, which only holds
    the mean and std ACROSS seeds - exactly the view that hides a case like one seed stalling
    for 30 generations while another is still improving."""
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
#  Flat summary table (A1's build_table / summarize_combination, ported)
# --------------------------------------------------------------------------- #
def summary_table(
    curves: dict[str, VariantCurve],
    conv_frame: pd.DataFrame,
    detail: pd.DataFrame,
) -> pd.DataFrame:
    """One row per configuration: exactly what goes straight into the report's summary table."""
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
#  Plateau (run-length check, not an outcome measure)
# --------------------------------------------------------------------------- #
def plateau_generation(
    generations: np.ndarray,
    mean_curve: np.ndarray,
    eps: float,
    window: int,
) -> int | None:
    """Earliest generation g from which the curve stays within `eps` of itself `window`
    generations earlier, all the way to the end of the run.

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

    # Per-run detail first: it decides which runs are saturated.
    detail = per_seed_detail(frames)
    detail.to_csv(out_dir / "per_seed_detail.csv", index=False)

    # Plots (i)-(iv)
    plot_fitness_curve(curves, out_dir / "fitness_curve.png", title)
    plot_sigma_curve(curves, out_dir / "sigma_curve.png", title)
    plot_success_rate_curve(curves, out_dir / "success_rate_curve.png", title)
    plot_final_distance(curves, out_dir / "final_distance.png", title)

    # (ii) and (iii) again without the saturated runs
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

    # Convergence
    conv_frame, target, censored = convergence(frames)
    conv_frame.to_csv(out_dir / "convergence.csv", index=False)

    # Statistics: both measures, one Holm family
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
