"""
Builds a SQLite database of S&P 500 point-in-time constituent data.

Tables:
  constituents_current : current snapshot metadata (sector, sub-industry, HQ, CIK, founded)
  index_changes         : raw historical additions/removals log (source of truth)
  membership            : derived point-in-time membership intervals (ticker, start, end)

The membership table is what you actually query for "which stocks were in
the S&P 500 on date D" -- it's derived by build_membership.py, which walks
index_changes in reverse chronological order from current constituents.

Re-run build_membership.py first if index_changes has been updated, then
re-run this script to refresh the database.
"""
import sqlite3
import pandas as pd

DB_PATH = "sp500_pit.db"

current = pd.read_csv("parsed_current.csv")
changes = pd.read_csv("parsed_changes.csv")
membership = pd.read_csv("parsed_membership.csv")

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.executescript("""
DROP TABLE IF EXISTS constituents_current;
DROP TABLE IF EXISTS index_changes;
DROP TABLE IF EXISTS membership;

CREATE TABLE constituents_current (
    ticker              TEXT PRIMARY KEY,
    security            TEXT NOT NULL,
    gics_sector         TEXT,
    gics_sub_industry   TEXT,
    hq_location         TEXT,
    date_added          TEXT,   -- ISO date string, most recent addition date
    cik                 TEXT,
    founded             TEXT
);

CREATE TABLE index_changes (
    effective_date      TEXT NOT NULL,
    added_ticker        TEXT,
    added_security      TEXT,
    removed_ticker       TEXT,
    removed_security    TEXT,
    reason               TEXT
);

CREATE TABLE membership (
    ticker      TEXT NOT NULL,
    security    TEXT,
    start_date  TEXT,   -- NULL = unknown, member since before log coverage (2019-01-02)
    end_date    TEXT,   -- NULL = still active as of last database refresh
    UNIQUE(ticker, start_date, end_date)
);

CREATE INDEX idx_membership_dates ON membership(start_date, end_date);
CREATE INDEX idx_membership_ticker ON membership(ticker);
CREATE INDEX idx_changes_date ON index_changes(effective_date);
""")

current.to_sql("constituents_current", conn, if_exists="append", index=False)
changes.to_sql("index_changes", conn, if_exists="append", index=False)
membership.to_sql("membership", conn, if_exists="append", index=False)

conn.commit()

# sanity check counts
for table in ("constituents_current", "index_changes", "membership"):
    n = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"{table}: {n} rows")

conn.close()
print(f"\nDatabase written to {DB_PATH}")
