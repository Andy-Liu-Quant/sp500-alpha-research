"""
Builds sp400_pit.db -- IDENTICAL schema to sp500_pit.db, so every existing
downstream script (download_prices.py, db_data_loader.py, alphas.py,
run_sp500_backtest.py, screen_alphas.py, combine_alphas.py,
ml_combine_alphas.py) works against it unchanged via --db sp400_pit.db.
None of those scripts hardcode any S&P-500-specific assumption; they only
take a db_path parameter.

KNOWN LIMITATIONS vs. the S&P 500 build (see build_sp400_membership.py
and project README for more):
  - Wikipedia's S&P 400 current-constituents table has no "Date added"
    column, so 139 of the 400 currently-active tickers have an unresolved
    (unknown) start date, vs. 0 for S&P 500.
  - Changes log coverage starts September 2014 (vs. 2009 for S&P 500) --
    reconstruction before that date is less reliable.
  - Ticker renames not explicitly logged as an add/remove pair (e.g.
    GPS->GAP) are not automatically merged into one continuous stint.
"""
import sqlite3
import pandas as pd

DB_PATH = "sp400_pit.db"

current = pd.read_csv("parsed_sp400_current.csv")
changes = pd.read_csv("parsed_sp400_changes.csv")
membership = pd.read_csv("parsed_sp400_membership.csv")

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.executescript("""
DROP TABLE IF EXISTS constituents_current;
DROP TABLE IF EXISTS index_changes;
DROP TABLE IF EXISTS membership;
DROP TABLE IF EXISTS prices;
DROP TABLE IF EXISTS download_log;

CREATE TABLE constituents_current (
    ticker              TEXT PRIMARY KEY,
    security            TEXT NOT NULL,
    gics_sector         TEXT,
    gics_sub_industry   TEXT,
    hq_location         TEXT
);

CREATE TABLE index_changes (
    effective_date      TEXT NOT NULL,
    added_ticker        TEXT,
    added_security       TEXT,
    removed_ticker       TEXT,
    removed_security     TEXT,
    reason               TEXT
);

CREATE TABLE membership (
    ticker      TEXT NOT NULL,
    security    TEXT,
    start_date  TEXT,
    end_date    TEXT,
    UNIQUE(ticker, start_date, end_date)
);

CREATE TABLE prices (
    ticker      TEXT NOT NULL,
    date        TEXT NOT NULL,
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL,
    adj_close   REAL,
    volume      REAL,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE download_log (
    ticker          TEXT PRIMARY KEY,
    status          TEXT,
    rows_downloaded INTEGER,
    date_range_start TEXT,
    date_range_end   TEXT,
    error_message   TEXT,
    last_attempted  TEXT
);

CREATE INDEX idx_membership_dates ON membership(start_date, end_date);
CREATE INDEX idx_membership_ticker ON membership(ticker);
CREATE INDEX idx_changes_date ON index_changes(effective_date);
CREATE INDEX idx_prices_ticker ON prices(ticker);
CREATE INDEX idx_prices_date ON prices(date);
""")

current.to_sql("constituents_current", conn, if_exists="append", index=False)
changes.to_sql("index_changes", conn, if_exists="append", index=False)
membership.to_sql("membership", conn, if_exists="append", index=False)
conn.commit()

for table in ("constituents_current", "index_changes", "membership", "prices", "download_log"):
    n = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"{table}: {n} rows")

conn.close()
print(f"\nDatabase written to {DB_PATH}")
