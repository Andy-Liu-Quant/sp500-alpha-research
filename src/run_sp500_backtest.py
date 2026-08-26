"""
Runs a WorldQuant-101-style alpha backtest using the point-in-time S&P 500
database (sp500_pit.db) instead of the static 30-ticker list / synthetic data.

Supports multiple alphas via --alpha (see alphas.py's ALPHA_REGISTRY).
Masking strategy is applied automatically per-alpha:
  - cross-sectional alphas (e.g. alpha3): inputs masked BEFORE computation
  - time-series-only alphas (e.g. alpha7): computed on full history, output
    masked AFTERWARD (see alphas.py docstring for why these differ)

Pipeline:
  1. Load prices for every ticker that's ever been in the database (prices table)
  2. Load the point-in-time membership mask (membership table)
  3. Compute the selected alpha, with masking applied internally by the
     alpha function itself at each of its formula's cross-sectional steps
     (see alphas.py's module docstring for the unified masking design --
     one mechanism, works for any alpha without per-alpha classification)
  4. Compute beta vs. a benchmark (SPY, downloaded separately)
  5. Run the constrained optimizer (dollar-neutral, beta-neutral, soft
     sector cap) exactly as in alpha3_backtest.py -- reused directly, not
     reimplemented
  6. Backtest, report performance, CAPM-adjusted IR, save chart

Usage:
    python run_sp500_backtest.py --start 2009-01-01 --end 2026-08-01
    python run_sp500_backtest.py --start 2009-01-01 --end 2026-08-01 --alpha alpha7
    python run_sp500_backtest.py --start 2009-01-01 --end 2026-08-01 --neutralize heuristic
"""
import argparse
import sqlite3

import numpy as np
import pandas as pd

from db_data_loader import (
    load_prices_from_db, load_membership_mask, load_sector_map, load_benchmark_returns,
)
from alphas import ALPHA_REGISTRY
from alpha3_backtest import (
    compute_forward_returns, compute_rolling_beta,
    build_optimized_weights, build_long_short_weights, beta_neutralize_weights,
    resample_weights, backtest, information_coefficient, performance_summary,
    capm_information_ratio, plot_results, CORR_WINDOW, REBALANCE_FREQ,
)

DEFAULT_EXCLUDE = ["SIVB", "FRC", "CHK", "WIN", "FTR", "DF", "MNK", "DNR", "BIG", "WFR"]
# confirmed real bankruptcy/severe-loss cases -- see prior discussion for
# the full sourced breakdown of all 37 "market cap change" delistings


def compute_alpha_with_masking(alpha_name: str, opens, highs, lows, closes, volumes, mask):
    """Every alpha function takes `mask` as its last parameter and applies
    it internally, at exactly the point(s) its formula has a cross-sectional
    rank(...)/correlation(...) step -- see alphas.py's module docstring.
    This dispatcher no longer needs to know anything about HOW a given
    alpha uses masking; it just passes mask through uniformly."""
    spec = ALPHA_REGISTRY[alpha_name]
    func = spec["func"]
    input_frames = {"opens": opens, "highs": highs, "lows": lows,
                     "closes": closes, "volumes": volumes}
    args = [input_frames[name] for name in spec["inputs"]]
    return func(*args, mask)


def run(db_path: str, start: str, end: str, benchmark: str = "SPY",
        neutralize: str = "optimizer", exclude: list = None,
        max_position: float = 0.10, gross_target: float = 2.0,
        sector_gross_cap_pct: float = 0.40, beta_window: int = 60,
        turnover_penalty: float = 0.0, alpha_name: str = "alpha3",
        auto_scale_turnover: bool = True, turnover_scale_smoothing_window: int = 10):

    exclude = exclude if exclude is not None else DEFAULT_EXCLUDE

    opens, highs, lows, closes, volumes = load_prices_from_db(db_path, start, end, exclude=exclude)
    mask = load_membership_mask(db_path, opens.index, list(opens.columns))
    sector_map = load_sector_map(db_path)

    market_returns = load_benchmark_returns(db_path, benchmark, start, end)
    if market_returns is None:
        print("Falling back to equal-weighted universe average as market proxy.")
        market_returns = closes.pct_change().mean(axis=1)

    alpha = compute_alpha_with_masking(alpha_name, opens, highs, lows, closes, volumes, mask)
    print(f"Using alpha: {alpha_name}")
    returns = compute_forward_returns(closes)
    beta = compute_rolling_beta(returns, market_returns, window=beta_window)

    if neutralize == "optimizer":
        # build_optimized_weights now solves only on real rebalance dates
        # and returns already-held weights directly -- no separate resample
        # step needed (see alpha3_backtest.py's build_optimized_weights
        # docstring for the bug this fixes: solving daily then resampling
        # meant the turnover penalty was anchored to throwaway intermediate
        # days, never the actually-held previous portfolio)
        weights = build_optimized_weights(
            alpha, beta, sector_map,
            max_position=max_position, gross_target=gross_target,
            sector_gross_cap_pct=sector_gross_cap_pct, turnover_penalty=turnover_penalty,
            auto_scale_turnover=auto_scale_turnover,
            turnover_scale_smoothing_window=turnover_scale_smoothing_window,
            rebalance_freq=REBALANCE_FREQ,
        )
    elif neutralize == "heuristic":
        raw_weights = build_long_short_weights(alpha)
        raw_weights = beta_neutralize_weights(raw_weights, beta)
        weights = resample_weights(raw_weights, REBALANCE_FREQ)
    else:
        raw_weights = build_long_short_weights(alpha)
        weights = resample_weights(raw_weights, REBALANCE_FREQ)

    # sanity check: confirm the optimizer never assigned weight to a
    # ticker on a day it wasn't actually a member (would indicate a bug
    # in the masking, not just a data-quality issue)
    leaked = (weights.abs() > 1e-9) & (~mask)
    n_leaked = leaked.sum().sum()
    if n_leaked > 0:
        print(f"WARNING: {n_leaked} (date, ticker) weight entries fall outside "
              f"point-in-time membership -- investigate before trusting results.")
    else:
        print("Membership mask check passed: no weight assigned outside point-in-time membership.")

    net_pnl, gross_pnl, turnover = backtest(weights, returns)
    ic_series = information_coefficient(alpha, returns)

    weights_lagged = weights.shift(1).fillna(0.0)
    beta_aligned = beta.reindex_like(weights_lagged)
    realized_net_beta = (weights_lagged * beta_aligned).sum(axis=1)
    realized_dollar = weights_lagged.sum(axis=1)
    print(f"\nRealized net beta   - mean: {realized_net_beta.mean():.4f}, std: {realized_net_beta.std():.4f}")
    print(f"Realized net dollar - mean: {realized_dollar.mean():.4f}, std: {realized_dollar.std():.4f}")

    summary = performance_summary(net_pnl, ic_series)
    print("\n--- Performance Summary ---")
    for k, v in summary.items():
        print(f"{k:22s}: {v}")

    capm = capm_information_ratio(net_pnl, market_returns)
    print("\n--- CAPM-Adjusted (true skill-based) ---")
    for k, v in capm.items():
        print(f"{k:28s}: {v:.4f}")

    results = pd.DataFrame({"gross_pnl": gross_pnl, "net_pnl": net_pnl, "turnover": turnover})
    csv_path = f"sp500_backtest_{alpha_name}_daily_results.csv"
    png_path = f"sp500_backtest_{alpha_name}_performance.png"
    results.to_csv(csv_path)
    print(f"Saved daily results to {csv_path}")

    # e.g. 'alpha3' -> 'Alpha#3', 'alpha7' -> 'Alpha#7' for the chart titles
    alpha_label = alpha_name.replace("alpha", "Alpha#") if alpha_name.startswith("alpha") else alpha_name
    plot_results(net_pnl, ic_series, png_path, gross_pnl=gross_pnl, alpha_label=alpha_label)

    return alpha, weights, net_pnl, ic_series, summary, capm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="sp500_pit.db")
    parser.add_argument("--start", default="2009-01-01")
    parser.add_argument("--end", default="2026-08-01")
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--neutralize", default="optimizer",
                         choices=["optimizer", "heuristic", "none"])
    parser.add_argument("--exclude", default=",".join(DEFAULT_EXCLUDE))
    parser.add_argument("--max-position", type=float, default=0.10)
    parser.add_argument("--gross-target", type=float, default=2.0)
    parser.add_argument("--sector-cap", type=float, default=0.40)
    parser.add_argument("--turnover-penalty", type=float, default=0.0,
                         help="Penalizes L1 distance from previous day's weights in the "
                              "optimizer objective. 0 = off (original behavior). Try 0.5-5.0 "
                              "to reduce cost-driven churn. See prior discussion re: gross vs "
                              "net cumulative return divergence.")
    parser.add_argument("--no-auto-scale-turnover", action="store_true",
                         help="Disable auto-scaling of turnover_penalty by a smoothed measure "
                              "of cross-sectional alpha std. By default (auto-scaling ON), the "
                              "same --turnover-penalty value represents a comparable relative "
                              "tradeoff across different alphas with different natural "
                              "dispersion (e.g. Alpha#3's smooth rolling correlation vs. "
                              "Alpha#7's hard binary volume-spike switch). Pass this flag to "
                              "use turnover_penalty as a raw, unscaled value instead.")
    parser.add_argument("--turnover-scale-window", type=int, default=10,
                         help="Trailing rolling-mean window (trading days) used to smooth the "
                              "daily alpha-dispersion scale for auto-scaling, rather than using "
                              "a single day's noisy raw value. Default 10.")
    parser.add_argument("--alpha", default="alpha3", choices=list(ALPHA_REGISTRY.keys()))
    args = parser.parse_args()

    exclude_list = [t.strip().upper() for t in args.exclude.split(",") if t.strip()]

    run(args.db, args.start, args.end, benchmark=args.benchmark,
        neutralize=args.neutralize, exclude=exclude_list,
        max_position=args.max_position, gross_target=args.gross_target,
        sector_gross_cap_pct=args.sector_cap, turnover_penalty=args.turnover_penalty,
        alpha_name=args.alpha, auto_scale_turnover=not args.no_auto_scale_turnover,
        turnover_scale_smoothing_window=args.turnover_scale_window)
