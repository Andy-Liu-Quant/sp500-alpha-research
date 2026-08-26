"""
Alpha#3 Backtest
================
Alpha#3 (WorldQuant 101 Formulaic Alphas):
    alpha = -1 * correlation(rank(open), rank(volume), 10)

Interpretation:
    - rank(open)   : cross-sectional rank of each stock's open price on a given day
    - rank(volume) : cross-sectional rank of each stock's volume on a given day
    - correlation(..., 10): rolling 10-day Pearson correlation of the two rank series
    - -1 *         : sign flip -> contrarian signal

This script:
    1. Downloads OHLCV data for a universe of tickers (via yfinance)
    2. Computes the Alpha#3 signal for every stock, every day
    3. Cross-sectionally ranks stocks by the signal each day
    4. Builds a long/short decile portfolio (long = highest alpha, short = lowest alpha)
    5. Computes daily portfolio returns, cumulative PnL, Sharpe ratio, and the
       Information Coefficient (IC) of the signal
    6. Saves a performance chart and a CSV of daily results

Usage:
    python alpha3_backtest.py

Edit the TICKERS list and date range below to customize your universe.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "UNH",
    "HD", "PG", "MA", "DIS", "BAC", "XOM", "PFE", "KO", "PEP", "CSCO",
    "INTC", "CMCSA", "VZ", "ADBE", "NFLX", "CRM", "ABT", "NKE", "T", "WMT",
]

START_DATE = "2019-01-01"
END_DATE = "2024-12-31"

CORR_WINDOW = 10          # lookback window for the rolling correlation, per the formula
LONG_SHORT_PCT = 0.2      # top/bottom 20% of names go long/short each day
REBALANCE_FREQ = "W"      # 'D' = daily, 'W' = weekly rebalancing (signal is slow-moving)
TRANSACTION_COST_BPS = 5  # round-trip cost assumption per unit turnover, in bps

OUTPUT_DIR = "."


# ----------------------------------------------------------------------
# 1. DATA DOWNLOAD
# ----------------------------------------------------------------------

def download_data(tickers, start, end):
    """Download OHLCV for all tickers, return dict of {field: DataFrame(date x ticker)}."""
    import yfinance as yf

    print(f"Downloading data for {len(tickers)} tickers...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)

    # yfinance returns a MultiIndex column DataFrame: (field, ticker)
    opens = raw["Open"]
    volumes = raw["Volume"]
    closes = raw["Adj Close"]  # use adjusted close for return calculations

    # yfinance can return a tz-aware index for some calls and tz-naive for others
    # depending on ticker/version -- normalize so separate downloads (e.g. universe
    # vs. SPY benchmark) align correctly on merge/rolling-cov operations later
    for df in (opens, volumes, closes):
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index = df.index.normalize()

    # Drop tickers with too much missing data
    good_cols = opens.columns[opens.isna().mean() < 0.05]
    opens, volumes, closes = opens[good_cols], volumes[good_cols], closes[good_cols]

    return opens, volumes, closes


def generate_synthetic_data(tickers, start, end, seed=42):
    """
    Fallback / test-mode data generator, useful for testing the pipeline
    without network access, or for quick experimentation.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n_days, n_assets = len(dates), len(tickers)

    # simulate correlated open/volume series with a mean-reverting relationship
    base_price = rng.uniform(50, 300, n_assets)
    price_paths = base_price * np.exp(
        np.cumsum(rng.normal(0.0002, 0.02, size=(n_days, n_assets)), axis=0)
    )
    opens = pd.DataFrame(price_paths, index=dates, columns=tickers)

    # volume with some structure: noisy, occasionally correlated with price moves
    vol_base = rng.uniform(1e6, 5e7, n_assets)
    shock = rng.normal(0, 1, size=(n_days, n_assets))
    volumes = pd.DataFrame(
        np.abs(vol_base * (1 + 0.3 * shock)), index=dates, columns=tickers
    )

    # close ~ open plus daily drift/noise, used to compute forward returns
    closes = opens * (1 + rng.normal(0.0003, 0.015, size=(n_days, n_assets)))

    return opens, volumes, closes


# ----------------------------------------------------------------------
# 2. ALPHA COMPUTATION
# ----------------------------------------------------------------------

def compute_alpha3(opens: pd.DataFrame, volumes: pd.DataFrame, window: int = CORR_WINDOW):
    """
    alpha = -1 * correlation(rank(open), rank(volume), window)

    rank(.) is a cross-sectional (per-day) rank across all tickers.
    correlation(.,.,window) is a rolling per-ticker Pearson correlation
    of the two ranked time series over `window` days.
    """
    # cross-sectional rank each day, scaled to [0, 1]
    rank_open = opens.rank(axis=1, pct=True)
    rank_volume = volumes.rank(axis=1, pct=True)

    # rolling correlation per column (per ticker), computed on the rank series
    rolling_corr = rank_open.rolling(window).corr(rank_volume)

    # pandas' rolling .corr() can occasionally emit +-inf due to floating-point
    # division by near-zero variance in a window; correlation is mathematically
    # bounded in [-1, 1], so treat these as missing rather than passing bad
    # values downstream (e.g. into the optimizer, where inf breaks the solver)
    rolling_corr = rolling_corr.replace([np.inf, -np.inf], np.nan)

    alpha = -1 * rolling_corr
    return alpha


# ----------------------------------------------------------------------
# 3. PORTFOLIO CONSTRUCTION
# ----------------------------------------------------------------------

def build_long_short_weights(alpha: pd.DataFrame, pct: float = LONG_SHORT_PCT):
    """
    Each day: rank stocks by alpha, go long the top `pct` fraction,
    short the bottom `pct` fraction, equal-weighted, dollar-neutral.
    """
    weights = pd.DataFrame(0.0, index=alpha.index, columns=alpha.columns)

    for date, row in alpha.iterrows():
        row = row.dropna()
        if len(row) < 5:
            continue
        n_select = max(1, int(len(row) * pct))
        sorted_row = row.sort_values()
        shorts = sorted_row.index[:n_select]
        longs = sorted_row.index[-n_select:]
        weights.loc[date, longs] = 1.0 / n_select
        weights.loc[date, shorts] = -1.0 / n_select

    return weights


def compute_rolling_beta(returns: pd.DataFrame, market_returns: pd.Series, window: int = 60):
    """
    Rolling beta of each stock vs. the market index, estimated via
    cov(stock, market) / var(market) over a trailing `window`.
    """
    cov = returns.rolling(window).cov(market_returns)
    market_var = market_returns.rolling(window).var()
    beta = cov.div(market_var, axis=0)
    return beta


def beta_neutralize_weights(weights: pd.DataFrame, beta: pd.DataFrame):
    """
    Rescale each day's weights so the portfolio's net beta exposure is ~0,
    while preserving the long/short structure and roughly the same gross exposure.

    Approach: split into long book and short book, and scale one side so that
        sum(long_w * beta) + sum(short_w * beta) == 0
    This keeps dollar-neutrality intact (longs and shorts still sum to +1/-1 gross)
    while zeroing out net beta.
    """
    beta_aligned = beta.reindex_like(weights)
    neutralized = weights.copy()

    for date in weights.index:
        w = weights.loc[date]
        b = beta_aligned.loc[date]
        longs = w[w > 0]
        shorts = w[w < 0]
        if longs.empty or shorts.empty:
            continue

        long_beta_exposure = (longs * b[longs.index]).sum()
        short_beta_exposure = (shorts * b[shorts.index]).sum()

        # scale the short book so long_beta_exposure + scale*short_beta_exposure = 0
        if short_beta_exposure != 0 and not np.isnan(short_beta_exposure):
            scale = -long_beta_exposure / short_beta_exposure
            scale = np.clip(scale, 0.25, 4.0)  # guardrail against extreme rescaling
            neutralized.loc[date, shorts.index] = shorts * scale

    return neutralized


def sector_neutralize_alpha(alpha: pd.DataFrame, sector_map: dict):
    """
    Demean the alpha within each sector, cross-sectionally, each day.
    sector_map: {ticker: sector_name}. Stocks not in the map are left as-is.
    Removes any systematic sector tilt in the ranking before portfolio construction.
    """
    sectors = pd.Series(sector_map)
    neutralized = alpha.copy()
    for sector in sectors.unique():
        cols = [t for t in sectors[sectors == sector].index if t in alpha.columns]
        if len(cols) < 2:
            continue
        sector_mean = alpha[cols].mean(axis=1)
        neutralized[cols] = alpha[cols].sub(sector_mean, axis=0)
    return neutralized


SECTOR_MAP = {
    "AAPL": "Info Tech", "MSFT": "Info Tech", "GOOGL": "Comm Services", "AMZN": "Consumer Disc",
    "META": "Comm Services", "NVDA": "Info Tech", "TSLA": "Consumer Disc", "JPM": "Financials",
    "V": "Info Tech", "UNH": "Health Care", "HD": "Consumer Disc", "PG": "Consumer Staples",
    "MA": "Info Tech", "DIS": "Comm Services", "BAC": "Financials", "XOM": "Energy",
    "PFE": "Health Care", "KO": "Consumer Staples", "PEP": "Consumer Staples", "CSCO": "Info Tech",
    "INTC": "Info Tech", "CMCSA": "Comm Services", "VZ": "Comm Services", "ADBE": "Info Tech",
    "NFLX": "Comm Services", "CRM": "Info Tech", "ABT": "Health Care", "NKE": "Consumer Disc",
    "T": "Comm Services", "WMT": "Consumer Staples",
}


def optimize_weights_single_day(alpha_row: pd.Series,
                                 beta_row: pd.Series,
                                 sector_map: dict,
                                 max_position: float = 0.15,
                                 gross_target: float = 2.0,
                                 sector_gross_cap_pct: float = 0.40,
                                 risk_aversion: float = 8.0,
                                 prev_weights: pd.Series = None,
                                 turnover_penalty: float = 0.0,
                                 auto_scale_turnover: bool = True,
                                 precomputed_alpha_scale: float = None):
    """
    Returns (weights_or_None, reason_str) where reason_str is one of:
        'ok', 'insufficient_valid_names', 'solver_exception', 'solver_not_optimal'
    Solve a constrained optimization for one day's portfolio weights:

        maximize    alpha^T w  -  risk_aversion * ||w||^2  -  effective_turnover_penalty * ||w - w_prev||_1
        subject to  sum(w) == 0                                 (dollar-neutral, HARD)
                    sum(w * beta) == 0                           (beta-neutral, HARD)
                    sum(|w_i| for i in sector s) <= cap, for each s   (sector concentration, SOFT cap,
                                                                        not forced to net zero)
                    sum(|w|) <= gross_target                     (gross exposure cap)
                    -max_position <= w_i <= max_position         (single-name position cap)

    The turnover_penalty term is what actually controls transaction-cost
    drag: without it, the optimizer re-solves from scratch every rebalance
    with no memory of yesterday's positions, so tiny/noisy changes in the
    alpha ranking can trigger large, costly reshuffling that isn't justified
    by any real new information. Penalizing L1 distance from prev_weights
    makes the optimizer only trade when the alpha benefit clearly outweighs
    the cost of moving -- turnover_penalty=0 recovers the original behavior.

    auto_scale_turnover: different alphas have different natural day-to-day
        dispersion (e.g. Alpha#3's smooth rolling correlation vs. Alpha#7's
        hard binary regime switch on volume spikes), so a fixed raw
        turnover_penalty value doesn't carry the same relative weight
        against the alpha term across different alphas. When True (default),
        turnover_penalty is scaled by a measure of typical alpha dispersion,
        so the same turnover_penalty setting represents roughly the same
        relative tradeoff regardless of which alpha's raw numeric scale
        you're using. Set to False to use turnover_penalty as a raw,
        unscaled value (original behavior).

    precomputed_alpha_scale: the scale factor to use when auto_scale_turnover
        is True. build_optimized_weights supplies this as a SMOOTHED
        (rolling-mean) scale across recent days rather than that single
        day's raw std -- since turnover is inherently a two-day quantity
        (how far today's weights move from yesterday's), anchoring the
        scale to a single day's snapshot means a quiet day right after a
        volatile one could suddenly loosen the penalty and unwind
        yesterday's carefully-anchored positions for no real reason. If
        not provided (e.g. calling this function directly/standalone),
        falls back to that single day's own cross-sectional alpha std.
    """
    import cvxpy as cp

    valid = alpha_row.dropna().index.intersection(beta_row.dropna().index)
    valid = [t for t in valid if t in sector_map]
    if len(valid) < 10:
        return None, "insufficient_valid_names"

    a = alpha_row[valid].values
    b = beta_row[valid].values
    n = len(valid)

    w = cp.Variable(n)

    constraints = [
        cp.sum(w) == 0,
        cp.sum(cp.multiply(w, b)) == 0,
        cp.norm1(w) <= gross_target,
        w >= -max_position,
        w <= max_position,
    ]

    sector_cap = sector_gross_cap_pct * gross_target
    sectors = pd.Series({t: sector_map[t] for t in valid})
    for sector in sectors.unique():
        mask = (sectors.values == sector)
        if mask.sum() >= 2:
            constraints.append(cp.norm1(w[mask]) <= sector_cap)

    objective_expr = a @ w - risk_aversion * cp.sum_squares(w)

    if turnover_penalty > 0 and prev_weights is not None and len(prev_weights) > 0:
        effective_turnover_penalty = turnover_penalty
        if auto_scale_turnover:
            if precomputed_alpha_scale is not None:
                alpha_scale = precomputed_alpha_scale
            else:
                alpha_scale = float(np.std(a))  # fallback: this day's own std
            if not np.isfinite(alpha_scale) or alpha_scale == 0:
                alpha_scale = 1.0  # e.g. every name flat at the same value with no recent history
            effective_turnover_penalty = turnover_penalty * alpha_scale

        # align prev_weights to today's valid universe; names not held
        # yesterday (or no longer valid today) are treated as 0 prior weight
        prev_aligned = prev_weights.reindex(valid).fillna(0.0).values
        objective_expr = objective_expr - effective_turnover_penalty * cp.norm1(w - prev_aligned)

    objective = cp.Maximize(objective_expr)
    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver=cp.CLARABEL)
    except Exception as e:
        return None, f"solver_exception: {e}"

    if w.value is None or problem.status not in ("optimal", "optimal_inaccurate"):
        return None, f"solver_not_optimal (status={problem.status})"

    return pd.Series(w.value, index=valid), "ok"


def build_optimized_weights(alpha: pd.DataFrame, beta: pd.DataFrame, sector_map: dict,
                             max_position: float = 0.15, gross_target: float = 2.0,
                             sector_gross_cap_pct: float = 0.40, risk_aversion: float = 8.0,
                             turnover_penalty: float = 0.0, auto_scale_turnover: bool = True,
                             turnover_scale_smoothing_window: int = 10,
                             rebalance_freq: str = "D"):
    """Loop the single-day optimizer, solving ONLY on actual rebalance dates
    (determined by rebalance_freq), and hold each solve's weights flat until
    the next rebalance -- returning an already-held (ready-to-backtest)
    weight series directly, no separate resample_weights() call needed.

    IMPORTANT: this fixes a real bug that existed when this function used to
    solve every trading day and get resampled to rebalance_freq afterward.
    In that version, prev_weights was chained day-to-day internally (Mon ->
    Tue -> Wed -> ...), but resample_weights() then discarded all but the
    last day of each period -- so e.g. Friday's solve was anchored against
    Thursday's daily solve, which was NEVER actually held/traded (it got
    thrown away), rather than against last Friday's actually-held portfolio.
    The turnover penalty was therefore providing close to zero real
    protection for the actual trade that happens between real rebalances,
    since the optimizer never even saw the real previous state. Solving
    only on real rebalance dates and chaining prev_weights between THOSE
    (and only those) fixes this: the turnover penalty is now correctly
    computed against what was genuinely held before.

    rebalance_freq: 'D' = every trading day (no resampling, matches
        original all-days behavior). Any pandas frequency string (e.g. 'W'
        for weekly) restricts solving to the last trading day of each
        period, holding flat in between -- this should match whatever
        REBALANCE_FREQ your backtest actually uses downstream, so the
        turnover penalty is anchored to the real trading cadence.

    When turnover_penalty > 0, each rebalance's solve is anchored to the
    previous REBALANCE's actual weights (not a throwaway intermediate
    day), so the optimizer only trades when it's worth it relative to what
    is genuinely held, rather than reshuffling on noise. See
    optimize_weights_single_day for auto_scale_turnover -- it normalizes
    the penalty by a measure of typical alpha dispersion so the same
    setting is comparable across different alphas.

    turnover_scale_smoothing_window: rather than scaling by a single day's
    raw cross-sectional alpha std (noisy), this precomputes a trailing
    rolling mean of the daily std across the whole date range up front and
    uses that smoothed value at each rebalance instead. Default 10
    (trading days, not rebalances -- keeps the smoothing window's real-time
    meaning consistent regardless of rebalance_freq).
    """
    from collections import Counter

    # determine the actual rebalance dates -- same logic as resample_weights,
    # kept here so the turnover-penalty anchor matches the real trading cadence
    if rebalance_freq == "D":
        rebalance_dates = list(alpha.index)
    else:
        period_id = alpha.index.to_series().dt.to_period(rebalance_freq)
        rebalance_dates = sorted(pd.Timestamp(d) for d in alpha.index.to_series().groupby(period_id).max().values)

    # precompute the smoothed alpha-dispersion scale for every date up front
    # (daily, not just rebalance dates -- keeps the window's real-time meaning
    # consistent regardless of rebalance_freq)
    daily_alpha_std = alpha.std(axis=1, skipna=True)
    smoothed_alpha_scale = daily_alpha_std.rolling(
        turnover_scale_smoothing_window, min_periods=1
    ).mean()

    weights = pd.DataFrame(0.0, index=alpha.index, columns=alpha.columns)
    fail_reasons = Counter()
    first_examples = {}
    prev_weights = None
    solved_dates = []

    for date in rebalance_dates:
        beta_row = beta.loc[date] if date in beta.index else pd.Series(dtype=float)
        scale_today = smoothed_alpha_scale.loc[date] if date in smoothed_alpha_scale.index else None
        result, reason = optimize_weights_single_day(
            alpha.loc[date], beta_row, sector_map,
            max_position=max_position, gross_target=gross_target,
            sector_gross_cap_pct=sector_gross_cap_pct, risk_aversion=risk_aversion,
            prev_weights=prev_weights, turnover_penalty=turnover_penalty,
            auto_scale_turnover=auto_scale_turnover,
            precomputed_alpha_scale=scale_today,
        )
        if result is None:
            key = reason.split(":")[0]
            fail_reasons[key] += 1
            first_examples.setdefault(key, (date, reason))
            continue
        weights.loc[date, result.index] = result.values
        prev_weights = result
        solved_dates.append(date)

    n_rebalances = len(rebalance_dates)
    n_failed = sum(fail_reasons.values())
    if n_failed:
        print(f"Optimizer: {n_failed}/{n_rebalances} rebalances skipped. Breakdown:")
        for reason, count in fail_reasons.most_common():
            example_date, example_msg = first_examples[reason]
            print(f"  {reason}: {count} rebalances (e.g. {example_date.date()}: {example_msg})")
        if fail_reasons.get("insufficient_valid_names", 0) == n_failed and n_failed == n_rebalances:
            print("  -> ALL rebalances failed on insufficient valid names: this almost always means "
                  "alpha or beta is entirely NaN. Check: (1) beta.notna().sum().sum() -- if 0, "
                  "the market_returns index likely doesn't align with returns.index (timezone or "
                  "date mismatch between separate downloads); (2) alpha.notna().sum().sum() -- if "
                  "0, opens/volumes may not have enough history for the rolling window yet, or the "
                  "DataFrame shape from yfinance wasn't what compute_alpha3 expected.")

    # hold each successful rebalance's weights flat until the next one --
    # same mechanic as resample_weights(), done here directly since we only
    # ever solved at rebalance dates in the first place. Failed rebalances
    # are simply skipped (previous holding carries forward via ffill), same
    # behavior as if that date had never been a rebalance date at all.
    held = pd.DataFrame(np.nan, index=alpha.index, columns=alpha.columns)
    held.loc[solved_dates] = weights.loc[solved_dates]
    held = held.ffill().fillna(0.0)
    return held


def resample_weights(weights: pd.DataFrame, freq: str):
    """Rebalance only on period boundaries; hold weights fixed in between."""
    if freq == "D":
        return weights
    # last actual trading day within each period (e.g. last business day of each week)
    period_id = weights.index.to_series().dt.to_period(freq)
    rebal_dates = weights.index.to_series().groupby(period_id).max().values

    held = weights.copy()
    held.loc[:, :] = np.nan
    held.loc[rebal_dates] = weights.loc[rebal_dates]
    held = held.ffill().fillna(0.0)
    return held


# ----------------------------------------------------------------------
# 4. PERFORMANCE EVALUATION
# ----------------------------------------------------------------------

def compute_forward_returns(closes: pd.DataFrame):
    """Simple daily close-to-close returns, shifted so weights[t] earn return[t+1]."""
    returns = closes.pct_change()
    return returns


def backtest(weights: pd.DataFrame, returns: pd.DataFrame, cost_bps: float = TRANSACTION_COST_BPS):
    """
    Apply weights (known at close of day t) to next-day returns.
    Subtract transaction costs proportional to turnover.
    """
    weights_lagged = weights.shift(1).fillna(0.0)  # avoid lookahead bias
    aligned_returns = returns.reindex_like(weights_lagged).fillna(0.0)

    gross_pnl = (weights_lagged * aligned_returns).sum(axis=1)

    turnover = weights_lagged.diff().abs().sum(axis=1).fillna(0.0)
    costs = turnover * (cost_bps / 1e4)

    net_pnl = gross_pnl - costs
    return net_pnl, gross_pnl, turnover


def information_coefficient(alpha: pd.DataFrame, returns: pd.DataFrame, horizon: int = 1):
    """
    Daily cross-sectional Spearman correlation between the alpha signal (day t)
    and forward returns (day t+horizon). This is the standard IC metric used
    to judge raw signal quality, independent of portfolio construction choices.
    """
    fwd_returns = returns.shift(-horizon)
    ic_series = alpha.corrwith(fwd_returns, axis=1, method="spearman")
    return ic_series.dropna()


def capm_information_ratio(net_pnl: pd.Series, market_returns: pd.Series, ann_factor: int = 252):
    """
    Regress portfolio returns on market returns to isolate the alpha (skill)
    component from beta (market exposure) before computing a risk-adjusted ratio.

        R_p = intercept(alpha) + beta * R_m + residual

    This is the correct way to measure "true" information ratio when a
    portfolio is dollar-neutral but NOT beta-neutral -- it strips out
    return/risk that's just unhedged market exposure, not genuine skill.
    """
    aligned = pd.concat([net_pnl, market_returns], axis=1, join="inner").dropna()
    aligned.columns = ["portfolio", "market"]

    X = np.column_stack([np.ones(len(aligned)), aligned["market"].values])
    y = aligned["portfolio"].values
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha_daily, beta_est = coef[0], coef[1]

    residuals = y - X @ coef
    resid_std = residuals.std(ddof=2)  # ddof=2: two estimated parameters (alpha, beta)

    alpha_annualized = alpha_daily * ann_factor
    residual_risk_annualized = resid_std * np.sqrt(ann_factor)
    true_ir = alpha_annualized / residual_risk_annualized if residual_risk_annualized > 0 else np.nan

    return {
        "estimated_beta": beta_est,
        "annualized_alpha": alpha_annualized,
        "residual_risk_annualized": residual_risk_annualized,
        "true_information_ratio": true_ir,
    }


def performance_summary(net_pnl: pd.Series, ic_series: pd.Series):
    ann_factor = 252
    mean_ret = net_pnl.mean() * ann_factor
    vol = net_pnl.std() * np.sqrt(ann_factor)
    sharpe = mean_ret / vol if vol > 0 else np.nan
    cum_ret = (1 + net_pnl).cumprod().iloc[-1] - 1
    max_dd = ((1 + net_pnl).cumprod() / (1 + net_pnl).cumprod().cummax() - 1).min()

    summary = {
        "Annualized Return": f"{mean_ret:.2%}",
        "Annualized Vol": f"{vol:.2%}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Total Return": f"{cum_ret:.2%}",
        "Max Drawdown": f"{max_dd:.2%}",
        "Mean Daily IC": f"{ic_series.mean():.4f}",
        "IC IR (mean/std)": f"{ic_series.mean() / ic_series.std():.3f}" if ic_series.std() > 0 else "n/a",
    }
    return summary


# ----------------------------------------------------------------------
# 5. PLOTTING
# ----------------------------------------------------------------------

def plot_results(net_pnl: pd.Series, ic_series: pd.Series, out_path: str, gross_pnl: pd.Series = None,
                  alpha_label: str = "Alpha#3"):
    """
    3-panel diagnostic:
      1. Gross vs. net cumulative return -- the gap between these two lines
         IS the cumulative cost drag. A smoothly diverging gap (net trending
         steadily below gross) while IC oscillates near zero is the classic
         signature of transaction costs eroding a near-zero-edge signal --
         NOT genuine negative alpha. Distinguishing these two failure modes
         is the whole point of this panel.
      2. 60-day rolling IC -- signal quality over time, independent of
         costs or position sizing.

    alpha_label: display name shown in the chart titles (e.g. "Alpha#3",
        "Alpha#7") -- purely cosmetic, doesn't affect any computation.
    """
    n_panels = 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(11, 4 * n_panels))

    ax_idx = 0
    if gross_pnl is not None:
        cum_net = (1 + net_pnl).cumprod()
        cum_gross = (1 + gross_pnl.reindex(net_pnl.index).fillna(0)).cumprod()
        axes[ax_idx].plot(cum_gross.index, cum_gross.values, color="#2ea043",
                           linewidth=1.3, label="Gross (before costs)")
        axes[ax_idx].plot(cum_net.index, cum_net.values, color="#1f6feb",
                           linewidth=1.5, label="Net (after costs)")
        axes[ax_idx].fill_between(cum_net.index, cum_gross.values, cum_net.values,
                                   color="#da3633", alpha=0.15, label="Cumulative cost drag")
        axes[ax_idx].set_title(f"{alpha_label} — Gross vs. Net Cumulative Return")
        axes[ax_idx].set_ylabel("Growth of $1")
        axes[ax_idx].legend(loc="upper right", fontsize=9)
        axes[ax_idx].grid(alpha=0.3)

        total_drag = cum_gross.iloc[-1] - cum_net.iloc[-1]
        print(f"Gross final value: {cum_gross.iloc[-1]:.3f}  |  "
              f"Net final value: {cum_net.iloc[-1]:.3f}  |  "
              f"Cumulative cost drag: {total_drag:.3f}")
    else:
        cum = (1 + net_pnl).cumprod()
        axes[ax_idx].plot(cum.index, cum.values, color="#1f6feb", linewidth=1.5)
        axes[ax_idx].set_title(f"{alpha_label} Long/Short Portfolio — Cumulative Return (net of costs)")
        axes[ax_idx].set_ylabel("Growth of $1")
        axes[ax_idx].grid(alpha=0.3)
    ax_idx += 1

    rolling_ic = ic_series.rolling(60).mean()
    axes[ax_idx].plot(rolling_ic.index, rolling_ic.values, color="#da3633", linewidth=1.2)
    axes[ax_idx].axhline(0, color="black", linewidth=0.8)
    axes[ax_idx].set_title(f"{alpha_label} — 60-Day Rolling Information Coefficient")
    axes[ax_idx].set_ylabel("IC (Spearman)")
    axes[ax_idx].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved chart to {out_path}")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def run(use_synthetic: bool = False, neutralize: str = "optimizer", beta_window: int = 60,
        turnover_penalty: float = 0.0, auto_scale_turnover: bool = True,
        turnover_scale_smoothing_window: int = 10):
    """
    neutralize: 'none'      - raw long/short weights, no neutrality constraints
                'heuristic' - beta_neutralize_weights() rescale (dollar-neutral OR beta-neutral, not both)
                'optimizer' - build_optimized_weights() joint constrained optimization
                              (dollar-neutral AND beta-neutral AND sector-neutral simultaneously)
    turnover_penalty: if > 0 and neutralize == 'optimizer', penalizes L1 distance
                from the previous day's weights in the optimizer objective, reducing
                cost-driven churn from noisy day-to-day alpha changes. Try values in
                the range 0.5-5.0; higher = less turnover, more inertia.
    auto_scale_turnover: normalizes turnover_penalty by a smoothed measure of
                cross-sectional alpha std, so the same setting is comparable
                across different alphas with different natural dispersion
                (e.g. Alpha#3's smooth rolling correlation vs. Alpha#7's
                hard binary regime switch). Default True.
    turnover_scale_smoothing_window: trailing rolling-mean window (in trading
                days) used to smooth the daily alpha-dispersion scale, rather
                than using a single day's noisy raw value. Default 10.
    """
    if use_synthetic:
        opens, volumes, closes = generate_synthetic_data(TICKERS, START_DATE, END_DATE)
        market_returns = closes.pct_change().mean(axis=1)
    else:
        try:
            opens, volumes, closes = download_data(TICKERS, START_DATE, END_DATE)
            mkt_opens, mkt_vol, mkt_close = download_data(["SPY"], START_DATE, END_DATE)
            market_returns = mkt_close["SPY"].pct_change()
        except Exception as e:
            print(f"Download failed ({e}), falling back to synthetic data for demonstration.")
            opens, volumes, closes = generate_synthetic_data(TICKERS, START_DATE, END_DATE)
            market_returns = closes.pct_change().mean(axis=1)

    alpha = compute_alpha3(opens, volumes, window=CORR_WINDOW)
    returns = compute_forward_returns(closes)
    beta = compute_rolling_beta(returns, market_returns, window=beta_window)

    if neutralize == "optimizer":
        # build_optimized_weights now solves only on real rebalance dates
        # and returns already-held weights directly -- no separate resample
        # step needed (that was the bug: solving daily then resampling
        # meant the turnover penalty was anchored to throwaway intermediate
        # days, not the actually-held previous portfolio)
        weights = build_optimized_weights(
            alpha, beta, SECTOR_MAP,
            turnover_penalty=turnover_penalty,
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

    net_pnl, gross_pnl, turnover = backtest(weights, returns)
    ic_series = information_coefficient(alpha, returns)

    weights_lagged = weights.shift(1).fillna(0.0)
    beta_aligned = beta.reindex_like(weights_lagged)
    realized_net_beta = (weights_lagged * beta_aligned).sum(axis=1)
    realized_dollar = weights_lagged.sum(axis=1)
    print(f"Realized net beta   - mean: {realized_net_beta.mean():.4f}, std: {realized_net_beta.std():.4f}")
    print(f"Realized net dollar - mean: {realized_dollar.mean():.4f}, std: {realized_dollar.std():.4f}")

    summary = performance_summary(net_pnl, ic_series)
    print("\n--- Performance Summary ---")
    for k, v in summary.items():
        print(f"{k:22s}: {v}")

    results = pd.DataFrame({
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "turnover": turnover,
    })
    results.to_csv(f"{OUTPUT_DIR}/alpha3_daily_results.csv")
    print(f"\nSaved daily results to {OUTPUT_DIR}/alpha3_daily_results.csv")

    plot_results(net_pnl, ic_series, f"{OUTPUT_DIR}/alpha3_performance.png", gross_pnl=gross_pnl)

    return alpha, weights, net_pnl, ic_series, summary


if __name__ == "__main__":
    run(use_synthetic=False, neutralize="optimizer")
