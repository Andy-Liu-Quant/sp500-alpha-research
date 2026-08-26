"""
Adds price-storage tables to sp500_pit.db.

  prices        : daily OHLCV, keyed on (ticker, date)
  download_log  : per-ticker download status, for resuming interrupted runs
                   and diagnosing which tickers failed and why

Run this once (or whenever you want to reset the schema) before running
download_prices.py.
"""

import sqlite3

DB_PATH = "sp500_pit.db"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.executescript("""
CREATE TABLE IF NOT EXISTS prices (
    ticker      TEXT NOT NULL,
    date        TEXT NOT NULL,   -- ISO 'YYYY-MM-DD'
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL,
    adj_close   REAL,
    volume      REAL,
    PRIMARY KEY (ticker, date)
);

CREATE INDEX IF NOT EXISTS idx_prices_ticker ON prices(ticker);
CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date);

CREATE TABLE IF NOT EXISTS download_log (
    ticker          TEXT PRIMARY KEY,
    status          TEXT,     -- 'ok', 'empty', 'error'
    rows_downloaded INTEGER,
    date_range_start TEXT,
    date_range_end   TEXT,
    error_message   TEXT,
    last_attempted  TEXT      -- ISO timestamp
);
""")

conn.commit()

for table in ("prices", "download_log"):
    n = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"{table}: {n} rows (table ready)")

conn.close()
