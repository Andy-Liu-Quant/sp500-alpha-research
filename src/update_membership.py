"""
Updates constituents_current, index_changes, and membership tables in your
EXISTING sp500_pit.db with the extended 2009-2026 changes log -- WITHOUT
touching your prices or download_log tables (your already-downloaded price
history is left completely alone).

Run this from the same directory as your existing sp500_pit.db, after
copying parsed_changes.csv, parsed_current.csv, and parsed_membership.csv
(provided alongside this script) into that same directory.

Usage:
    python update_membership.py
"""
import sqlite3
import pandas as pd

DB_PATH = "sp500_pit.db"

current = pd.read_csv("parsed_current.csv")
changes = pd.read_csv("parsed_changes.csv")
membership = pd.read_csv("parsed_membership.csv")

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# only touch the 3 membership-related tables -- prices and download_log
# are never referenced here, so your downloaded price history is untouched
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
    date_added          TEXT,
    cik                 TEXT,
    founded             TEXT
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

CREATE INDEX idx_membership_dates ON membership(start_date, end_date);
CREATE INDEX idx_membership_ticker ON membership(ticker);
CREATE INDEX idx_changes_date ON index_changes(effective_date);
""")

current.to_sql("constituents_current", conn, if_exists="append", index=False)
changes.to_sql("index_changes", conn, if_exists="append", index=False)
membership.to_sql("membership", conn, if_exists="append", index=False)
conn.commit()

for table in ("constituents_current", "index_changes", "membership", "prices", "download_log"):
    try:
        n = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"{table}: {n} rows")
    except sqlite3.OperationalError:
        print(f"{table}: table does not exist (skipped)")

conn.close()
print("\nMembership tables updated. prices and download_log were not touched.")
