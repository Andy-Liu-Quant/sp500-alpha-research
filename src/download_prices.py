"""
Downloads daily OHLCV for every ticker that has ever been an S&P 500
constituent (per sp500_pit.db's membership table) and stores it in the
`prices` table of the same database.

WHY THIS SCRIPT IS DESIGNED THIS WAY:
  - 600+ tickers is enough that a single yf.download() call, or a naive
    per-ticker loop with no error handling, WILL partially fail (rate
    limiting, delisted tickers with no data, renamed/reused symbols,
    transient network errors). This script batches requests, retries,
    logs every outcome per ticker, and is safe to re-run: it skips
    tickers already successfully downloaded unless --force is passed.
  - Batching (yf.download accepts a list of tickers per call) is much
    faster than one call per ticker, but a bad ticker in a batch can
    make pandas return partial/misaligned data -- so failed batches
    fall back to downloading that batch's tickers one at a time.

USAGE (run this on a machine with real internet access -- Yahoo Finance
is not reachable from network-restricted sandboxes):

    python download_prices.py                  # download all, skip done
    python download_prices.py --force           # re-download everything
    python download_prices.py --start 2015-01-01 --end 2026-08-01
    python download_prices.py --batch-size 25   # smaller batches if rate-limited
"""
import argparse
import sqlite3
import time
from datetime import datetime, timezone

import pandas as pd

DB_PATH = "sp500_pit.db"

# Known cases where the ticker used in our membership/sector data (matching
# what the company was called *at the time*) no longer resolves on Yahoo,
# because the company did a pure ticker-symbol rename (not an M&A event --
# same listing, same company, just relabeled). Yahoo typically preserves full
# historical data under the NEW symbol back through the rename date, so we
# fetch under the new symbol but store under the old one, keeping this table
# consistent with what membership.py / SECTOR_MAP expect.
SYMBOL_RENAME_OVERRIDES = {
    "GPS": "GAP",    # Gap Inc. changed NYSE ticker GPS -> GAP on 2024-08-22
}


def yf_ticker_symbol(ticker: str) -> str:
    """yfinance uses '-' instead of '.' for share classes, e.g. BRK.B -> BRK-B.
    Also applies any known symbol-rename override (see SYMBOL_RENAME_OVERRIDES)."""
    ticker = SYMBOL_RENAME_OVERRIDES.get(ticker, ticker)
    return ticker.replace(".", "-")


def get_all_tickers(conn, ticker_filter: list = None) -> list:
    """Every ticker that has ever appeared in the membership table, or just
    `ticker_filter` if provided (for targeted re-downloads of specific tickers
    without re-running the full universe)."""
    if ticker_filter:
        return sorted(set(ticker_filter))
    df = pd.read_sql("SELECT DISTINCT ticker FROM membership ORDER BY ticker", conn)
    return df["ticker"].tolist()


def get_already_done(conn) -> set:
    df = pd.read_sql("SELECT ticker FROM download_log WHERE status = 'ok'", conn)
    return set(df["ticker"].tolist())


def normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """Strip timezone and normalize to plain dates -- same fix as alpha3_backtest's
    download_data, needed so prices from different tickers/batches always align."""
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    return df


def save_ticker_data(conn, ticker: str, df: pd.DataFrame):
    """Write one ticker's OHLCV rows to the prices table (upsert via replace)."""
    if df is None or df.empty:
        return 0
    df = df.copy()
    df = normalize_index(df)

    required = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        # Fail loudly rather than silently writing NULLs for every row --
        # this exact failure mode (unexpected column shape, e.g. leftover
        # MultiIndex columns) previously caused every field to write NULL
        # with no error at all.
        raise ValueError(
            f"{ticker}: DataFrame missing expected columns {missing}. "
            f"Actual columns: {list(df.columns)}. Refusing to write NULL rows."
        )

    records = []
    for date, row in df.iterrows():
        records.append((
            ticker,
            date.strftime("%Y-%m-%d"),
            float(row["Open"]) if pd.notna(row["Open"]) else None,
            float(row["High"]) if pd.notna(row["High"]) else None,
            float(row["Low"]) if pd.notna(row["Low"]) else None,
            float(row["Close"]) if pd.notna(row["Close"]) else None,
            float(row["Adj Close"]) if pd.notna(row["Adj Close"]) else None,
            float(row["Volume"]) if pd.notna(row["Volume"]) else None,
        ))
    conn.executemany(
        """INSERT OR REPLACE INTO prices
           (ticker, date, open, high, low, close, adj_close, volume)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        records,
    )
    return len(records)


def log_result(conn, ticker: str, status: str, n_rows: int = 0,
                start: str = None, end: str = None, error: str = None):
    conn.execute(
        """INSERT OR REPLACE INTO download_log
           (ticker, status, rows_downloaded, date_range_start, date_range_end,
            error_message, last_attempted)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (ticker, status, n_rows, start, end, error,
         datetime.now(timezone.utc).isoformat()),
    )


def download_single(ticker: str, start: str, end: str):
    """Fallback path: download one ticker at a time. Returns (df_or_None, error_or_None)."""
    import yfinance as yf
    try:
        df = yf.download(yf_ticker_symbol(ticker), start=start, end=end,
                          auto_adjust=False, progress=False)
        if df.empty:
            return None, "empty (no data returned -- likely delisted or invalid symbol)"
        return df, None
    except Exception as e:
        return None, str(e)


def download_batch(tickers: list, start: str, end: str):
    """Try a batched multi-ticker download. Returns dict {ticker: df} for tickers
    that came back cleanly; tickers missing from the dict need the single-ticker
    fallback."""
    import yfinance as yf
    symbols = [yf_ticker_symbol(t) for t in tickers]
    symbol_to_ticker = dict(zip(symbols, tickers))

    try:
        raw = yf.download(symbols, start=start, end=end, auto_adjust=False,
                           progress=False, group_by="ticker", threads=True)
    except Exception:
        return {}  # whole batch failed -- caller falls back to per-ticker

    results = {}
    if len(symbols) == 1:
        # yfinance's behavior here is version-dependent: sometimes it returns
        # flat columns for a single ticker, sometimes it still returns
        # MultiIndex (ticker, field) columns because group_by='ticker' was
        # passed explicitly. Normalize to flat columns either way -- this was
        # a real bug: leaving MultiIndex columns in place meant every field
        # lookup (row.get('Open')) silently returned None, writing NULL to
        # every row instead of raising an error.
        if not raw.empty:
            if isinstance(raw.columns, pd.MultiIndex):
                raw = raw.droplevel(0, axis=1) if raw.columns.get_level_values(0)[0] == symbols[0] \
                    else raw.droplevel(1, axis=1)
            results[tickers[0]] = raw
        return results

    for sym in symbols:
        try:
            sub = raw[sym]
        except (KeyError, Exception):
            continue
        sub = sub.dropna(how="all")
        if not sub.empty:
            results[symbol_to_ticker[sym]] = sub
    return results


def run(start: str, end: str, batch_size: int = 50, force: bool = False,
        sleep_between_batches: float = 1.0, ticker_filter: list = None,
        db_path: str = None):
    conn = sqlite3.connect(db_path or DB_PATH)

    all_tickers = get_all_tickers(conn, ticker_filter)

    if ticker_filter:
        # explicit targeting always re-attempts the listed tickers,
        # regardless of prior download_log status
        todo = all_tickers
        already_done = set()
    else:
        already_done = set() if force else get_already_done(conn)
        todo = [t for t in all_tickers if t not in already_done]

    print(f"Total tickers: {len(all_tickers)}")
    print(f"Already downloaded: {len(already_done)}")
    print(f"To download: {len(todo)}")

    n_ok, n_empty, n_error = 0, 0, 0

    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        print(f"\nBatch {i // batch_size + 1}/{(len(todo) - 1) // batch_size + 1}: "
              f"{len(batch)} tickers ({batch[0]}...{batch[-1]})")

        batch_results = download_batch(batch, start, end)

        # anything not in batch_results needs the single-ticker fallback
        missing = [t for t in batch if t not in batch_results]

        for ticker, df in batch_results.items():
            try:
                n_rows = save_ticker_data(conn, ticker, df)
                log_result(conn, ticker, "ok", n_rows, start, end)
                n_ok += 1
            except ValueError as e:
                log_result(conn, ticker, "error", 0, start, end, str(e))
                n_error += 1

        for ticker in missing:
            df, error = download_single(ticker, start, end)
            if df is not None:
                try:
                    n_rows = save_ticker_data(conn, ticker, df)
                    log_result(conn, ticker, "ok", n_rows, start, end)
                    n_ok += 1
                except ValueError as e:
                    log_result(conn, ticker, "error", 0, start, end, str(e))
                    n_error += 1
            elif error and "empty" in error:
                log_result(conn, ticker, "empty", 0, start, end, error)
                n_empty += 1
            else:
                log_result(conn, ticker, "error", 0, start, end, error)
                n_error += 1

        conn.commit()
        time.sleep(sleep_between_batches)

    print(f"\nDone. ok={n_ok} empty={n_empty} error={n_error}")

    if n_error > 0 or n_empty > 0:
        print("\nTickers with issues (see download_log table for details):")
        problems = pd.read_sql(
            "SELECT ticker, status, error_message FROM download_log "
            "WHERE status != 'ok' ORDER BY ticker", conn)
        print(problems.to_string(index=False))

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="sp500_pit.db",
                         help="Path to the point-in-time database to download prices into. "
                              "E.g. --db sp400_pit.db for the mid-cap universe.")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--tickers", default=None,
                         help="Comma-separated list of specific tickers to "
                              "(re-)download, bypassing the full-universe "
                              "skip-if-done logic. E.g. --tickers FL,HBI,SEE,CMA")
    args = parser.parse_args()

    ticker_filter = None
    if args.tickers:
        ticker_filter = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    run(args.start, args.end, batch_size=args.batch_size, force=args.force,
        sleep_between_batches=args.sleep, ticker_filter=ticker_filter, db_path=args.db)
