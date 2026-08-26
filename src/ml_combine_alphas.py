"""
Combines multiple alpha signals into one composite using statistical
learning (Lasso, Random Forest, Gradient Boosting), compared honestly
against the naive rank-averaged ensembles from combine_alphas.py.

CRITICAL METHODOLOGY -- WALK-FORWARD VALIDATION:
The danger with time-series financial data isn't the bootstrap resampling
inside Random Forest (each row here is a (date, ticker) cross-sectional
observation, not a single evolving series -- bagging across stock-days is
a much milder, well-studied caveat: same-day rows share market-wide shocks,
so out-of-bag error estimates run slightly optimistic, but this doesn't
invalidate the method). The REAL danger, for ANY model (Lasso, Random
Forest, boosting, even plain linear regression), is evaluating on a
RANDOM train/test split of dates -- that lets the model train on data
chronologically AFTER what it's tested on, which financial time series
absolutely cannot tolerate (regimes, trends, and autocorrelation mean a
random split leaks future information into training).

This script instead uses walk-forward (expanding-window) validation:
train on all data through year Y, test ONLY on year Y+1 (strictly later,
never seen during training), then expand the training window to include
Y+1 and test on Y+2, repeating forward through the full history. Every
reported "out-of-sample IC" number comes from predictions the model made
on data it could not have seen during that fold's training.

Usage:
    python ml_combine_alphas.py --start 2009-01-01 --end 2026-08-01
    python ml_combine_alphas.py --start 2009-01-01 --end 2026-08-01 --alphas alpha1,alpha7,alpha9
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from db_data_loader import load_prices_from_db, load_membership_mask
from screen_alphas import compute_cumulative_forward_returns
from alphas import ALPHA_REGISTRY

warnings.filterwarnings("ignore")

DEFAULT_EXCLUDE = ["SIVB", "FRC", "CHK", "WIN", "FTR", "DF", "MNK", "DNR", "BIG", "WFR"]


def build_panel(alpha_signals: dict, returns: pd.DataFrame) -> pd.DataFrame:
    """Stack per-alpha (date x ticker) DataFrames into one long panel:
    one row per (date, ticker), one column per alpha, plus the forward
    return target. This is the standard "tabular ML on a panel" shape."""
    frames = []
    for name, alpha_df in alpha_signals.items():
        stacked = alpha_df.stack()
        stacked.name = name
        frames.append(stacked)

    ret_stacked = returns.stack()
    ret_stacked.name = "forward_return"
    frames.append(ret_stacked)

    panel = pd.concat(frames, axis=1)
    panel.index.names = ["date", "ticker"]
    panel = panel.dropna()  # rows need every alpha + a valid forward return
    return panel


def walk_forward_folds(dates: pd.DatetimeIndex, initial_train_years: int = 5):
    """Yields (train_end, test_start, test_end) as calendar-year boundaries
    for expanding-window walk-forward validation."""
    years = sorted(dates.year.unique())
    if len(years) <= initial_train_years:
        raise ValueError(f"Need more than {initial_train_years} years of data for "
                          f"walk-forward validation; only have {len(years)}.")

    for test_year in years[initial_train_years:]:
        train_end = pd.Timestamp(f"{test_year - 1}-12-31")
        test_start = pd.Timestamp(f"{test_year}-01-01")
        test_end = pd.Timestamp(f"{test_year}-12-31")
        yield train_end, test_start, test_end


def daily_cross_sectional_ic(predictions: pd.Series, actual_returns: pd.Series) -> pd.Series:
    """predictions and actual_returns are both indexed by (date, ticker).
    Returns the daily cross-sectional Spearman IC, unstacked back to a
    per-date series."""
    df = pd.DataFrame({"pred": predictions, "actual": actual_returns})
    ic_by_date = df.groupby(level="date").apply(
        lambda g: g["pred"].corr(g["actual"], method="spearman") if len(g) >= 10 else np.nan
    )
    return ic_by_date.dropna()


def run(db_path: str, start: str, end: str, alpha_names: list = None, exclude: list = None,
        initial_train_years: int = 5, horizon: int = 5):
    """
    horizon: forward return window in trading days used as the ML target.
    Default 5 (one trading week) to match REBALANCE_FREQ='W' in the actual
    backtest -- the model should be trained to predict returns over the
    SAME period the strategy actually holds positions for, not an
    arbitrary 1-day return that doesn't match the real holding period.
    Uses genuine cumulative forward returns (close[t+h]-close[t])/close[t],
    correctly forward-shifted -- NOT alpha3_backtest.compute_forward_returns,
    which despite its name returns same-day (backward-looking) daily
    returns; see prior discussion for the bug this fixes.
    """
    exclude = exclude if exclude is not None else DEFAULT_EXCLUDE
    alpha_names = alpha_names if alpha_names is not None else list(ALPHA_REGISTRY.keys())

    print(f"Loading data and computing {len(alpha_names)} alpha(s)...")
    print(f"Target: genuine {horizon}-day forward cumulative return "
          f"(matches weekly rebalancing if horizon=5)")
    opens, highs, lows, closes, volumes = load_prices_from_db(db_path, start, end, exclude=exclude)
    mask = load_membership_mask(db_path, opens.index, list(opens.columns))
    returns = compute_cumulative_forward_returns(closes, horizon)

    input_frames = {"opens": opens, "highs": highs, "lows": lows,
                     "closes": closes, "volumes": volumes}
    alpha_signals = {}
    for name in alpha_names:
        spec = ALPHA_REGISTRY[name]
        args = [input_frames[n] for n in spec["inputs"]]
        try:
            alpha_signals[name] = spec["func"](*args, mask)
        except Exception as e:
            print(f"  {name} failed: {e}")

    print("Building panel (date, ticker) x alphas...")
    panel = build_panel(alpha_signals, returns)
    print(f"Panel shape: {panel.shape[0]:,} rows x {panel.shape[1]} columns")

    feature_cols = list(alpha_signals.keys())
    dates_available = panel.index.get_level_values("date").unique()

    models = {
        "Lasso": lambda: Lasso(alpha=0.00001, max_iter=10000),
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=200, max_depth=5, min_samples_leaf=500,
            n_jobs=-1, random_state=0),
        "GradientBoosting": lambda: HistGradientBoostingRegressor(
            max_depth=4, max_iter=200, learning_rate=0.05,
            min_samples_leaf=500, random_state=0),
    }

    # naive baselines (equal-weight rank average) for direct comparison,
    # evaluated on the SAME out-of-sample folds as the ML models -- not
    # just the full sample -- for a fair apples-to-apples comparison
    ranked_signals = {name: df.rank(axis=1, pct=True) for name, df in alpha_signals.items()}
    equal_weight_baseline = sum(ranked_signals.values()) / len(ranked_signals)
    baseline_stacked = equal_weight_baseline.stack()
    baseline_stacked.name = "baseline"

    fold_results = []
    oos_ic_by_model = {name: [] for name in list(models.keys()) + ["EqualWeightBaseline"]}
    train_ic_by_model = {name: [] for name in models.keys()}

    sorted_dates = np.sort(dates_available)

    for train_end, test_start, test_end in walk_forward_folds(dates_available, initial_train_years):
        # EMBARGO: a training row on trading day t has its target computed
        # from close[t+horizon] -- if t is within `horizon` trading days of
        # train_end, that target reaches into the test period, leaking test
        # data through the label even though the ROW's date is technically
        # in the "training" range. Drop the last `horizon` trading days
        # before train_end from training to eliminate this.
        train_dates_up_to_boundary = sorted_dates[sorted_dates <= np.datetime64(train_end)]
        if len(train_dates_up_to_boundary) <= horizon:
            continue  # not enough history yet to embargo safely; skip this fold
        embargoed_train_end = train_dates_up_to_boundary[-(horizon + 1)]

        train_mask = panel.index.get_level_values("date") <= embargoed_train_end
        test_mask = (panel.index.get_level_values("date") >= test_start) & \
                    (panel.index.get_level_values("date") <= test_end)

        train_panel = panel[train_mask]
        test_panel = panel[test_mask]

        if len(train_panel) < 1000 or len(test_panel) < 100:
            continue

        X_train, y_train = train_panel[feature_cols], train_panel["forward_return"]
        X_test, y_test = test_panel[feature_cols], test_panel["forward_return"]

        fold_label = f"train<={embargoed_train_end.astype('datetime64[D]')} (embargoed {horizon}d before {train_end.date()}) | test={test_start.year}"
        print(f"\nFold: {fold_label}  (train={len(X_train):,} rows, test={len(X_test):,} rows)")

        for model_name, model_factory in models.items():
            if model_name == "Lasso":
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)
            else:
                X_train_s, X_test_s = X_train, X_test

            model = model_factory()
            model.fit(X_train_s, y_train)

            train_pred = pd.Series(model.predict(X_train_s), index=X_train.index)
            test_pred = pd.Series(model.predict(X_test_s), index=X_test.index)

            if train_pred.std() < 1e-12:
                print(f"  {model_name:18s} WARNING: predictions are ~constant "
                      f"(std={train_pred.std():.2e}) -- likely all coefficients/splits "
                      f"collapsed to zero given how weak the signal is. Skipping IC "
                      f"(would be undefined) rather than reporting a misleading NaN silently.")
                continue

            train_ic = daily_cross_sectional_ic(train_pred, y_train)
            test_ic = daily_cross_sectional_ic(test_pred, y_test)

            oos_ic_by_model[model_name].append(test_ic)
            train_ic_by_model[model_name].append(train_ic)

            print(f"  {model_name:18s} train IC: {train_ic.mean():.4f}   "
                  f"OOS IC: {test_ic.mean():.4f}   "
                  f"overfit gap: {train_ic.mean() - test_ic.mean():.4f}")

        # baseline on the exact same test rows
        baseline_test = baseline_stacked.reindex(test_panel.index)
        baseline_ic = daily_cross_sectional_ic(baseline_test, y_test)
        oos_ic_by_model["EqualWeightBaseline"].append(baseline_ic)
        print(f"  {'EqualWeightBaseline':18s}                OOS IC: {baseline_ic.mean():.4f}")

    print("\n" + "=" * 70)
    print("SUMMARY -- Out-of-sample IC aggregated across all walk-forward folds")
    print("=" * 70)
    summary = []
    for name, ic_list in oos_ic_by_model.items():
        if not ic_list:
            continue
        full_oos_ic = pd.concat(ic_list)
        mean_ic = full_oos_ic.mean()
        std_ic = full_oos_ic.std()
        ic_ir = mean_ic / std_ic if std_ic > 0 else np.nan
        train_mean = pd.concat(train_ic_by_model[name]).mean() if name in train_ic_by_model else np.nan
        summary.append({
            "model": name,
            "oos_mean_ic": mean_ic,
            "oos_ic_ir": ic_ir,
            "train_mean_ic": train_mean,
            "overfit_gap": (train_mean - mean_ic) if not np.isnan(train_mean) else np.nan,
            "n_oos_days": len(full_oos_ic),
        })

    summary_df = pd.DataFrame(summary).sort_values("oos_mean_ic", key=abs, ascending=False)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(summary_df.to_string(index=False))

    summary_df.to_csv("ml_combine_results.csv", index=False)
    print("\nSaved ml_combine_results.csv")

    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="sp500_pit.db")
    parser.add_argument("--start", default="2009-01-01")
    parser.add_argument("--end", default="2026-08-01")
    parser.add_argument("--alphas", default=None)
    parser.add_argument("--exclude", default=",".join(DEFAULT_EXCLUDE))
    parser.add_argument("--initial-train-years", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=5,
                         help="Forward return window in trading days for the ML target. "
                              "Default 5 (one trading week) to match weekly rebalancing.")
    args = parser.parse_args()

    alpha_names = [a.strip() for a in args.alphas.split(",")] if args.alphas else None
    exclude_list = [t.strip().upper() for t in args.exclude.split(",") if t.strip()]

    run(args.db, args.start, args.end, alpha_names=alpha_names, exclude=exclude_list,
        initial_train_years=args.initial_train_years, horizon=args.horizon)
