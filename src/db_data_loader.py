"""
Loads data from sp500_pit.db and prepares it for alpha3_backtest.py's
pipeline (compute_alpha3, build_optimized_weights, backtest, etc.).

The critical piece here is point-in-time membership masking: a ticker's
price data is only allowed to participate in the cross-sectional alpha
ranking on days it was ACTUALLY an S&P 500 constituent. Without this,
loading "all tickers that were ever in the index" and just backtesting
across the full date range would let e.g. NVDA's 2015-2019 data (before
it joined the index in 2001... bad example, it's been in a while -- but
e.g. a 2023-added ticker's pre-2023 history) leak into a cross-sectional
ranking on dates before that stock was actually eligible to be traded as
part of this universe. Masking prevents that.

Returns (unmasked, since price-return computation should reflect whatever
actually happened to a position regardless of index membership status):
    - closes (for computing returns / beta) : NOT masked
Returns (masked before computing alpha's rank/correlation):
    - opens, volumes : masked to NaN on non-member days
"""
import sqlite3
import pandas as pd


def load_prices_from_db(db_path: str, start: str, end: str, exclude: list = None):
    """Load prices table, pivot to wide (date x ticker) DataFrames.
    exclude: tickers to drop entirely (e.g. ['SIVB', 'FRC'] -- confirmed bank
    failures where missing post-collapse data would hide a real, large loss;
    see prior discussion. Clean-M&A delistings like FL/HBI/CMA/SEE are NOT
    excluded here -- their truncated-at-zero final return is a much smaller,
    low-risk distortion, as previously discussed, so no special handling.

    Returns (opens, highs, lows, closes, volumes) -- note closes uses
    adj_close (for return computations), while highs/lows are RAW
    (unadjusted) since they're only used by alphas that need intraday
    range, not for return calculations."""
    exclude = exclude or []
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(
        "SELECT ticker, date, open, high, low, close, adj_close, volume FROM prices "
        "WHERE date >= ? AND date <= ?",
        conn, params=(start, end),
    )
    conn.close()

    if exclude:
        before = df["ticker"].nunique()
        df = df[~df["ticker"].isin(exclude)]
        print(f"Excluded {before - df['ticker'].nunique()} ticker(s): {exclude}")

    df["date"] = pd.to_datetime(df["date"])

    opens = df.pivot(index="date", columns="ticker", values="open")
    highs = df.pivot(index="date", columns="ticker", values="high")
    lows = df.pivot(index="date", columns="ticker", values="low")
    volumes = df.pivot(index="date", columns="ticker", values="volume")
    closes = df.pivot(index="date", columns="ticker", values="adj_close")

    print(f"Loaded prices: {len(opens)} dates x {opens.shape[1]} tickers "
          f"({opens.index.min().date()} to {opens.index.max().date()})")

    return opens, highs, lows, closes, volumes


def load_membership_mask(db_path: str, dates: pd.DatetimeIndex, tickers: list) -> pd.DataFrame:
    """Boolean (date x ticker) DataFrame: True where `ticker` was an actual
    S&P 500 constituent on `date`, per the point-in-time membership table."""
    conn = sqlite3.connect(db_path)
    mem = pd.read_sql("SELECT ticker, start_date, end_date FROM membership", conn)
    conn.close()

    mem["start_date"] = pd.to_datetime(mem["start_date"])
    mem["end_date"] = pd.to_datetime(mem["end_date"])

    mask = pd.DataFrame(False, index=dates, columns=tickers)

    # group by ticker for efficiency rather than looping every (date, ticker) pair
    for ticker, rows in mem.groupby("ticker"):
        if ticker not in mask.columns:
            continue
        col_mask = pd.Series(False, index=dates)
        for _, row in rows.iterrows():
            start = row["start_date"] if pd.notna(row["start_date"]) else dates.min()
            end = row["end_date"] if pd.notna(row["end_date"]) else dates.max()
            col_mask |= (dates >= start) & (dates <= end)
        mask[ticker] = col_mask.values

    coverage = mask.sum(axis=1)
    print(f"Point-in-time universe size: min={coverage.min()}, "
          f"max={coverage.max()}, mean={coverage.mean():.0f} tickers/day")

    return mask


def load_sector_map(db_path: str) -> dict:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT ticker, gics_sector FROM constituents_current", conn)
    conn.close()
    return dict(zip(df["ticker"], df["gics_sector"]))


def load_benchmark_returns(db_path: str, benchmark_ticker: str, start: str, end: str):
    """Load a benchmark's returns (e.g. SPY) for beta computation. SPY is not
    an S&P 500 constituent itself, so it must have been downloaded separately
    (download_prices.py --tickers SPY). Falls back to an equal-weighted
    universe average with a warning if unavailable."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(
        "SELECT date, adj_close FROM prices WHERE ticker = ? AND date >= ? AND date <= ? "
        "ORDER BY date",
        conn, params=(benchmark_ticker, start, end),
    )
    conn.close()

    if df.empty:
        print(f"WARNING: no data for benchmark '{benchmark_ticker}' in the database. "
              f"Run: python download_prices.py --tickers {benchmark_ticker} "
              f"--start {start} --end {end}")
        return None

    df["date"] = pd.to_datetime(df["date"])
    series = df.set_index("date")["adj_close"].pct_change()
    print(f"Loaded benchmark '{benchmark_ticker}': {len(series)} days")
    return series
