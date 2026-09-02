"""
Safely updates the membership-related tables (constituents_current,
index_changes, membership) in an EXISTING point-in-time database --
WITHOUT touching the prices or download_log tables, so your already-
downloaded price history is left completely alone.

WHY THIS SCRIPT EXISTS: build_sp500_db.py and build_sp400_db.py both
DROP and recreate ALL FIVE tables, including prices and download_log.
That's correct for a from-scratch build, but destructive if you've
already downloaded price data (which can take hours). Use this script
instead whenever you want to refresh membership data on a database
that already has prices in it.

Supports both universes -- the two indexes have different
constituents_current schemas (the S&P 500 Wikipedia table has
date_added/cik/founded columns; the S&P 400 table does not), so pass
--universe to pick the right one.

Usage:
    # S&P 500 (expects parsed_current.csv, parsed_changes.csv, parsed_membership.csv)
    python update_membership.py --universe sp500 --db sp500_pit.db

    # S&P 400 (expects parsed_sp400_*.csv files)
    python update_membership.py --universe sp400 --db sp400_pit.db
"""
import argparse
import sqlite3

import pandas as pd

SCHEMAS = {
    "sp500": {
        "csv_current": "parsed_current.csv",
        "csv_changes": "parsed_changes.csv",
        "csv_membership": "parsed_membership.csv",
        "default_db": "sp500_pit.db",
        "constituents_ddl": """
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
        """,
    },
    "sp400": {
        "csv_current": "parsed_sp400_current.csv",
        "csv_changes": "parsed_sp400_changes.csv",
        "csv_membership": "parsed_sp400_membership.csv",
        "default_db": "sp400_pit.db",
        # S&P 400's Wikipedia table has no date_added/cik/founded columns
        "constituents_ddl": """
            CREATE TABLE constituents_current (
                ticker              TEXT PRIMARY KEY,
                security            TEXT NOT NULL,
                gics_sector         TEXT,
                gics_sub_industry   TEXT,
                hq_location         TEXT
            );
        """,
    },
}

SHARED_DDL = """
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
"""


def run(universe: str, db_path: str = None):
    spec = SCHEMAS[universe]
    db_path = db_path or spec["default_db"]

    current = pd.read_csv(spec["csv_current"])
    changes = pd.read_csv(spec["csv_changes"])
    membership = pd.read_csv(spec["csv_membership"])

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # record pre-existing price/log row counts so we can PROVE they survived
    def safe_count(table):
        try:
            return cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        except sqlite3.OperationalError:
            return None

    prices_before = safe_count("prices")
    log_before = safe_count("download_log")

    # only the 3 membership-related tables are dropped -- prices and
    # download_log are never referenced in this DDL at all
    cur.executescript(
        "DROP TABLE IF EXISTS constituents_current;\n"
        "DROP TABLE IF EXISTS index_changes;\n"
        "DROP TABLE IF EXISTS membership;\n"
        + spec["constituents_ddl"]
        + SHARED_DDL
    )

    current.to_sql("constituents_current", conn, if_exists="append", index=False)
    changes.to_sql("index_changes", conn, if_exists="append", index=False)
    membership.to_sql("membership", conn, if_exists="append", index=False)
    conn.commit()

    print(f"Updated {db_path} (universe: {universe})\n")
    for table in ("constituents_current", "index_changes", "membership"):
        n = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {n} rows (rebuilt)")

    prices_after = safe_count("prices")
    log_after = safe_count("download_log")

    print()
    for name, before, after in (("prices", prices_before, prices_after),
                                 ("download_log", log_before, log_after)):
        if before is None:
            print(f"  {name}: table does not exist (nothing to preserve)")
        elif before == after:
            print(f"  {name}: {after} rows -- UNCHANGED (preserved correctly)")
        else:
            print(f"  {name}: WARNING -- was {before}, now {after}. This should "
                  f"not happen; investigate before trusting this database.")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", choices=["sp500", "sp400"], required=True)
    parser.add_argument("--db", default=None,
                         help="Database path. Defaults to sp500_pit.db / sp400_pit.db "
                              "depending on --universe.")
    args = parser.parse_args()
    run(args.universe, args.db)
