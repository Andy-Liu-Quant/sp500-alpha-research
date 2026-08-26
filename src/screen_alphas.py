"""
Fast IC screening across many alphas -- skips the optimizer/backtest
entirely (the slow part: a QP solve per rebalance) and just computes each
alpha's raw signal + gross Information Coefficient against forward returns.

This is the right tool for "we have ~101 candidate alphas and want to know
which ones are even worth a full backtest" -- screening is ~10-50x faster
per alpha than a full run_sp500_backtest.py pass, since it never touches
beta, the sector map, or cvxpy at all.

MULTI-HORIZON TESTING: also supports testing each alpha against forward
returns over several holding periods (e.g. 1, 5, 10, 20 days), not just
next-day. This uses genuine CUMULATIVE N-day forward returns
(close.pct_change(h).shift(-h)) -- NOT alpha3_backtest.information_coefficient's
built-in `horizon` parameter, which shifts an already-1-day return series
and therefore tests something different (whether alpha predicts the single
1-day return that happens to occur N days later, not the cumulative N-day
move). A signal whose real predictive power plays out gradually over
several days can show near-zero 1-day IC while still having genuine
multi-day predictive power -- this is the right tool to check for that
before concluding an alpha has no edge at all.

Workflow:
    1. Run this to rank ALL registered alphas by gross IC on your universe/period
    2. Only run run_sp500_backtest.py's full pipeline (with the optimizer,
       turnover penalty, cost model, etc.) on whichever alphas actually
       clear a meaningful IC bar here

Usage:
    python screen_alphas.py --start 2009-01-01 --end 2026-08-01
    python screen_alphas.py --start 2009-01-01 --end 2026-08-01 --alphas alpha1,alpha3,alpha7
    python screen_alphas.py --start 2009-01-01 --end 2026-08-01 --horizons 1,5,10,20
"""
import argparse
import sqlite3

import numpy as np
import pandas as pd

from db_data_loader import load_prices_from_db, load_membership_mask
from alpha3_backtest import compute_forward_returns, information_coefficient
from alphas import ALPHA_REGISTRY

DEFAULT_EXCLUDE = ["SIVB", "FRC", "CHK", "WIN", "FTR", "DF", "MNK", "DNR", "BIG", "WFR"]


def compute_cumulative_forward_returns(closes: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Genuine N-day forward CUMULATIVE return, correctly aligned: at time t,
    this gives (close[t+h] - close[t]) / close[t] -- the total return an
    investor would realize holding from t to t+h. This is what should be
    used for multi-horizon IC testing, not a shifted 1-day return series."""
    return closes.pct_change(horizon).shift(-horizon)


def compute_ic_at_horizon(alpha: pd.DataFrame, closes: pd.DataFrame, horizon: int) -> pd.Series:
    """IC of alpha (known at time t) against the genuine N-day cumulative
    forward return starting at t. No further shifting needed -- the
    cumulative-return series is already correctly aligned to t."""
    fwd_return = compute_cumulative_forward_returns(closes, horizon)
    ic_series = alpha.corrwith(fwd_return, axis=1, method="spearman")
    return ic_series.dropna()


def screen_one_alpha_multi_horizon(alpha_name: str, opens, highs, lows, closes, volumes,
                                    mask, horizons: list) -> dict:
    """Compute one alpha's signal once, then test it against several
    different holding-period horizons using genuine cumulative forward
    returns (see compute_cumulative_forward_returns)."""
    spec = ALPHA_REGISTRY[alpha_name]
    input_frames = {"opens": opens, "highs": highs, "lows": lows,
                     "closes": closes, "volumes": volumes}
    args = [input_frames[name] for name in spec["inputs"]]

    try:
        alpha = spec["func"](*args, mask)
    except Exception as e:
        return {"alpha": alpha_name, "status": f"error: {e}"}

    result = {"alpha": alpha_name, "status": "ok"}
    any_valid = False
    for h in horizons:
        ic_series = compute_ic_at_horizon(alpha, closes, h)
        if len(ic_series) == 0:
            result[f"mean_ic_{h}d"] = np.nan
            result[f"ic_ir_{h}d"] = np.nan
            continue
        any_valid = True
        mean_ic = ic_series.mean()
        std_ic = ic_series.std()
        result[f"mean_ic_{h}d"] = mean_ic
        result[f"ic_ir_{h}d"] = mean_ic / std_ic if std_ic > 0 else np.nan

    if not any_valid:
        result["status"] = "no valid IC observations at any horizon"
    return result


def screen_one_alpha(alpha_name: str, opens, highs, lows, closes, volumes, mask, returns) -> dict:
    """Compute just the alpha signal + gross IC stats -- no beta, no
    optimizer, no cost model. This is what makes screening fast."""
    spec = ALPHA_REGISTRY[alpha_name]
    input_frames = {"opens": opens, "highs": highs, "lows": lows,
                     "closes": closes, "volumes": volumes}
    args = [input_frames[name] for name in spec["inputs"]]

    try:
        alpha = spec["func"](*args, mask)
    except Exception as e:
        return {"alpha": alpha_name, "status": f"error: {e}"}

    ic_series = information_coefficient(alpha, returns)
    if len(ic_series) == 0:
        return {"alpha": alpha_name, "status": "no valid IC observations"}

    mean_ic = ic_series.mean()
    std_ic = ic_series.std()
    ic_ir = mean_ic / std_ic if std_ic > 0 else np.nan
    pct_positive = (ic_series > 0).mean()
    coverage = alpha.notna().sum().sum() / alpha.size

    return {
        "alpha": alpha_name,
        "status": "ok",
        "n_days": len(ic_series),
        "mean_ic": mean_ic,
        "ic_ir": ic_ir,
        "pct_positive_days": pct_positive,
        "data_coverage": coverage,
    }


def run(db_path: str, start: str, end: str, alpha_names: list = None, exclude: list = None,
        horizons: list = None):
    exclude = exclude if exclude is not None else DEFAULT_EXCLUDE
    alpha_names = alpha_names if alpha_names is not None else list(ALPHA_REGISTRY.keys())

    print(f"Screening {len(alpha_names)} alpha(s) on {start} to {end}...")
    opens, highs, lows, closes, volumes = load_prices_from_db(db_path, start, end, exclude=exclude)
    mask = load_membership_mask(db_path, opens.index, list(opens.columns))

    if horizons is not None:
        print(f"Multi-horizon mode: testing {horizons}-day forward cumulative returns")
        results = []
        for name in alpha_names:
            if name not in ALPHA_REGISTRY:
                print(f"  skipping '{name}': not in ALPHA_REGISTRY")
                continue
            print(f"  computing {name}...")
            results.append(screen_one_alpha_multi_horizon(
                name, opens, highs, lows, closes, volumes, mask, horizons))

        df = pd.DataFrame(results)
        ok = df[df["status"] == "ok"].copy()
        failed = df[df["status"] != "ok"]

        if not ok.empty:
            pd.set_option("display.float_format", lambda x: f"{x:.4f}")
            pd.set_option("display.width", 200)
            mean_ic_cols = [f"mean_ic_{h}d" for h in horizons]
            print("\n--- Mean IC by horizon (columns = forward holding period) ---")
            print(ok[["alpha"] + mean_ic_cols].to_string(index=False))
            ic_ir_cols = [f"ic_ir_{h}d" for h in horizons]
            print("\n--- IC IR by horizon ---")
            print(ok[["alpha"] + ic_ir_cols].to_string(index=False))

        if not failed.empty:
            print("\n--- Failed / no data ---")
            print(failed[["alpha", "status"]].to_string(index=False))

        out_path = "alpha_screening_multi_horizon_results.csv"
        df.to_csv(out_path, index=False)
        print(f"\nSaved full results to {out_path}")
        return df

    returns = compute_forward_returns(closes)

    results = []
    for name in alpha_names:
        if name not in ALPHA_REGISTRY:
            print(f"  skipping '{name}': not in ALPHA_REGISTRY")
            continue
        print(f"  computing {name}...")
        results.append(screen_one_alpha(name, opens, highs, lows, closes, volumes, mask, returns))

    df = pd.DataFrame(results)
    ok = df[df["status"] == "ok"].copy()
    failed = df[df["status"] != "ok"]

    if not ok.empty:
        ok["abs_mean_ic"] = ok["mean_ic"].abs()
        ok = ok.sort_values("abs_mean_ic", ascending=False).drop(columns="abs_mean_ic")
        pd.set_option("display.float_format", lambda x: f"{x:.4f}")
        print("\n--- Ranked by |mean IC| (best candidates for a full backtest) ---")
        print(ok[["alpha", "mean_ic", "ic_ir", "pct_positive_days", "data_coverage", "n_days"]]
              .to_string(index=False))

    if not failed.empty:
        print("\n--- Failed / no data ---")
        print(failed[["alpha", "status"]].to_string(index=False))

    out_path = "alpha_screening_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved full results to {out_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="sp500_pit.db")
    parser.add_argument("--start", default="2009-01-01")
    parser.add_argument("--end", default="2026-08-01")
    parser.add_argument("--alphas", default=None,
                         help="Comma-separated alpha names to screen. Default: all registered.")
    parser.add_argument("--exclude", default=",".join(DEFAULT_EXCLUDE))
    parser.add_argument("--horizons", default=None,
                         help="Comma-separated forward holding periods in trading days, e.g. "
                              "'1,5,10,20'. When set, tests each alpha's IC against genuine "
                              "cumulative N-day forward returns at each horizon (not just "
                              "next-day) -- useful when a signal's real predictive power might "
                              "play out gradually rather than showing up immediately.")
    args = parser.parse_args()

    alpha_names = [a.strip() for a in args.alphas.split(",")] if args.alphas else None
    exclude_list = [t.strip().upper() for t in args.exclude.split(",") if t.strip()]
    horizons = [int(h.strip()) for h in args.horizons.split(",")] if args.horizons else None

    run(args.db, args.start, args.end, alpha_names=alpha_names, exclude=exclude_list,
        horizons=horizons)
