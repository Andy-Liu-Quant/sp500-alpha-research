"""
Tests whether combining multiple weak individual alphas produces a
meaningfully stronger composite signal -- the Fundamental Law of Active
Management logic: individually weak signals can combine into something
better IF they're not too correlated with each other (i.e. each is
capturing at least partly different information).

HORIZON MATTERS: IC should be evaluated at the horizon that matches the
strategy's actual rebalance/holding period, not an arbitrary default.
Testing at the wrong horizon can hide real predictive power (a signal
whose edge builds gradually over a week will look weak under a 1-day
test) or overstate it (a signal with fast-decaying 1-day edge is useless
to a weekly-rebalanced strategy that can't act on it before it decays).
This script defaults to horizon=5 (one trading week) to match
REBALANCE_FREQ='W' in the actual backtest, using genuine cumulative
forward returns (see screen_alphas.compute_cumulative_forward_returns) --
NOT alpha3_backtest.information_coefficient's default horizon=1, which
earlier runs of this script (and screen_alphas.py's non-multi-horizon
path) used without realizing the mismatch. Pass --horizon 1 to reproduce
the old (rebalance-mismatched) behavior if you want to compare.

Two things this script does:
  1. Pairwise alpha correlation matrix -- for every pair of registered
     alphas, the mean daily cross-sectional Spearman correlation between
     their signals. High correlation between two alphas means they're
     largely redundant (combining them won't help much); low/negative
     correlation means they may be capturing different information
     (good ensemble candidates).
  2. Ensemble IC test -- combines alpha signals via cross-sectional
     rank-averaging (robust to each alpha's raw numeric scale) and
     computes the combined signal's gross IC at the specified horizon.

Usage:
    python combine_alphas.py --start 2009-01-01 --end 2026-08-01
    python combine_alphas.py --start 2009-01-01 --end 2026-08-01 --horizon 1
    python combine_alphas.py --start 2009-01-01 --end 2026-08-01 --alphas alpha9,alpha1,alpha4
"""
import argparse

import numpy as np
import pandas as pd

from db_data_loader import load_prices_from_db, load_membership_mask
from screen_alphas import compute_cumulative_forward_returns
from alphas import ALPHA_REGISTRY

DEFAULT_EXCLUDE = ["SIVB", "FRC", "CHK", "WIN", "FTR", "DF", "MNK", "DNR", "BIG", "WFR"]


def compute_ic_series(signal: pd.DataFrame, closes: pd.DataFrame, horizon: int) -> pd.Series:
    """IC of a signal (alpha or combined ensemble) against the genuine
    N-day cumulative forward return at the given horizon."""
    fwd_return = compute_cumulative_forward_returns(closes, horizon)
    ic_series = signal.corrwith(fwd_return, axis=1, method="spearman")
    return ic_series.dropna()


def compute_all_alphas(alpha_names: list, opens, highs, lows, closes, volumes, mask) -> dict:
    """Compute every requested alpha's signal once, reused for both the
    correlation matrix and the ensemble construction."""
    input_frames = {"opens": opens, "highs": highs, "lows": lows,
                     "closes": closes, "volumes": volumes}
    results = {}
    for name in alpha_names:
        spec = ALPHA_REGISTRY[name]
        args = [input_frames[n] for n in spec["inputs"]]
        try:
            results[name] = spec["func"](*args, mask)
        except Exception as e:
            print(f"  {name} failed: {e}")
    return results


def pairwise_alpha_correlation(alpha_signals: dict) -> pd.DataFrame:
    """Mean daily cross-sectional Spearman correlation between every pair
    of alpha signals -- same mechanic as information_coefficient(), just
    between two alphas instead of alpha-vs-forward-returns."""
    names = list(alpha_signals.keys())
    corr_matrix = pd.DataFrame(index=names, columns=names, dtype=float)

    for i, name_i in enumerate(names):
        for name_j in names[i:]:
            if name_i == name_j:
                corr_matrix.loc[name_i, name_j] = 1.0
                continue
            daily_corr = alpha_signals[name_i].corrwith(
                alpha_signals[name_j], axis=1, method="spearman"
            )
            mean_corr = daily_corr.mean()
            corr_matrix.loc[name_i, name_j] = mean_corr
            corr_matrix.loc[name_j, name_i] = mean_corr

    return corr_matrix


def build_rank_averaged_ensemble(alpha_signals: dict, names: list, weights: dict = None) -> pd.DataFrame:
    """Combine alpha signals via cross-sectional rank-averaging: each
    alpha's raw values are converted to a daily cross-sectional percentile
    rank (robust to differing raw numeric scales across alphas -- an alpha
    bounded in [-1,1] and one bounded in [-0.5,0.5] combine meaningfully
    once both are just "percentile among today's eligible universe"), then
    averaged (equal-weight by default, or IC-weighted if weights given)."""
    ranked = {}
    for name in names:
        ranked[name] = alpha_signals[name].rank(axis=1, pct=True)

    if weights is None:
        weights = {name: 1.0 for name in names}

    total_weight = sum(weights[name] for name in names)
    combined = sum(ranked[name] * weights[name] for name in names) / total_weight
    return combined


def run(db_path: str, start: str, end: str, alpha_names: list = None, exclude: list = None,
        horizon: int = 5):
    exclude = exclude if exclude is not None else DEFAULT_EXCLUDE
    alpha_names = alpha_names if alpha_names is not None else list(ALPHA_REGISTRY.keys())

    print(f"Loading data and computing {len(alpha_names)} alpha(s)...")
    print(f"IC horizon: {horizon} trading day(s) forward cumulative return")
    opens, highs, lows, closes, volumes = load_prices_from_db(db_path, start, end, exclude=exclude)
    mask = load_membership_mask(db_path, opens.index, list(opens.columns))

    alpha_signals = compute_all_alphas(alpha_names, opens, highs, lows, closes, volumes, mask)
    valid_names = list(alpha_signals.keys())

    # --- individual IC, for reference and for IC-weighting the ensemble ---
    individual_ic = {}
    for name in valid_names:
        ic_series = compute_ic_series(alpha_signals[name], closes, horizon)
        individual_ic[name] = ic_series.mean() if len(ic_series) > 0 else np.nan

    print("\n--- Pairwise alpha correlation (mean daily cross-sectional Spearman) ---")
    corr_matrix = pairwise_alpha_correlation(alpha_signals)
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")
    pd.set_option("display.width", 200)
    print(corr_matrix)

    # flag highly redundant pairs (potential low value-add from combining)
    print("\nHighly correlated pairs (|corr| > 0.5 -- likely redundant, combining won't add much):")
    found_redundant = False
    for i, name_i in enumerate(valid_names):
        for name_j in valid_names[i + 1:]:
            c = corr_matrix.loc[name_i, name_j]
            if abs(c) > 0.5:
                print(f"  {name_i} <-> {name_j}: {c:.2f}")
                found_redundant = True
    if not found_redundant:
        print("  none found -- alphas appear largely non-redundant with each other")

    # --- ensemble tests ---
    print("\n--- Ensemble IC tests ---")

    configs = {
        "equal-weight, all alphas": (valid_names, None),
    }

    # top-5 by individual |mean IC|, equal-weighted
    top5 = sorted(valid_names, key=lambda n: abs(individual_ic.get(n, 0)), reverse=True)[:5]
    configs["equal-weight, top 5 by |IC|"] = (top5, None)

    # IC-weighted, all alphas (weight = |mean IC|, sign-corrected so each
    # alpha contributes in its "working" direction)
    ic_weights = {name: abs(individual_ic.get(name, 0)) for name in valid_names}
    signed_signals = {
        name: (alpha_signals[name] if individual_ic.get(name, 0) >= 0 else -alpha_signals[name])
        for name in valid_names
    }
    configs["IC-weighted (sign-corrected), all alphas"] = (valid_names, ic_weights, signed_signals)

    results = []
    for label, cfg in configs.items():
        if len(cfg) == 2:
            names, weights = cfg
            signals_to_use = alpha_signals
        else:
            names, weights, signals_to_use = cfg

        combined = build_rank_averaged_ensemble(signals_to_use, names, weights)
        ic_series = compute_ic_series(combined, closes, horizon)
        if len(ic_series) == 0:
            results.append({"config": label, "n_alphas": len(names), "mean_ic": np.nan, "ic_ir": np.nan})
            continue
        mean_ic = ic_series.mean()
        std_ic = ic_series.std()
        ic_ir = mean_ic / std_ic if std_ic > 0 else np.nan
        results.append({
            "config": label, "n_alphas": len(names),
            "mean_ic": mean_ic, "ic_ir": ic_ir,
            "n_days": len(ic_series),
        })

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    print("\n--- Individual alpha IC, for reference ---")
    ind_df = pd.DataFrame({"alpha": list(individual_ic.keys()), "mean_ic": list(individual_ic.values())})
    ind_df["abs_ic"] = ind_df["mean_ic"].abs()
    ind_df = ind_df.sort_values("abs_ic", ascending=False).drop(columns="abs_ic")
    print(ind_df.to_string(index=False))

    corr_matrix.to_csv("alpha_correlation_matrix.csv")
    results_df.to_csv("ensemble_results.csv", index=False)
    print("\nSaved alpha_correlation_matrix.csv and ensemble_results.csv")

    return corr_matrix, results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="sp500_pit.db")
    parser.add_argument("--start", default="2009-01-01")
    parser.add_argument("--end", default="2026-08-01")
    parser.add_argument("--alphas", default=None,
                         help="Comma-separated alpha names. Default: all registered.")
    parser.add_argument("--exclude", default=",".join(DEFAULT_EXCLUDE))
    parser.add_argument("--horizon", type=int, default=5,
                         help="Forward return horizon in trading days for IC evaluation. "
                              "Default 5 (one trading week) to match weekly rebalancing. "
                              "Pass 1 to reproduce the old (rebalance-mismatched) behavior.")
    args = parser.parse_args()

    alpha_names = [a.strip() for a in args.alphas.split(",")] if args.alphas else None
    exclude_list = [t.strip().upper() for t in args.exclude.split(",") if t.strip()]

    run(args.db, args.start, args.end, alpha_names=alpha_names, exclude=exclude_list,
        horizon=args.horizon)
