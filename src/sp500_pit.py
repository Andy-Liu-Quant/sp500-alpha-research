"""
Query interface for the point-in-time S&P 500 database (sp500_pit.db).

Usage:
    from sp500_pit import get_membership_on_date, get_membership_range, get_sector_map

    tickers = get_membership_on_date("2022-06-15")
    universe_by_day = get_membership_range("2020-01-01", "2024-12-31")
    sectors = get_sector_map()  # {ticker: gics_sector}, current classification
"""
import sqlite3
import pandas as pd

DB_PATH = "sp500_pit.db"

# Earliest date our index_changes log covers -- membership rows with
# start_date IS NULL are only reliably "known member" from this date forward.
EARLIEST_RELIABLE_DATE = "2019-01-02"


def _connect():
    return sqlite3.connect(DB_PATH)


def get_membership_on_date(date: str) -> list:
    """
    Return the list of tickers that were S&P 500 constituents on `date`
    (a 'YYYY-MM-DD' string). A ticker counts as a member if:
        (start_date IS NULL OR start_date <= date) AND
        (end_date IS NULL OR end_date >= date)
    NULL start_date is treated as "member since before our log coverage
    began" (2019-01-02) -- so queries before that date may be incomplete.
    """
    conn = _connect()
    query = """
        SELECT DISTINCT ticker FROM membership
        WHERE (start_date IS NULL OR start_date <= ?)
          AND (end_date IS NULL OR end_date >= ?)
    """
    result = pd.read_sql(query, conn, params=(date, date))
    conn.close()
    return sorted(result["ticker"].tolist())


def get_membership_range(start_date: str, end_date: str, freq: str = "W") -> dict:
    """
    Return {date_str: [tickers]} for each period boundary in the range,
    at the given pandas frequency (default weekly -- matches typical
    rebalance cadence, avoids computing membership every single day).
    """
    dates = pd.date_range(start_date, end_date, freq=freq)
    conn = _connect()
    membership = pd.read_sql("SELECT * FROM membership", conn)
    conn.close()

    result = {}
    for d in dates:
        d_str = d.strftime("%Y-%m-%d")
        mask = (
            (membership["start_date"].isna() | (membership["start_date"] <= d_str))
            & (membership["end_date"].isna() | (membership["end_date"] >= d_str))
        )
        result[d_str] = sorted(membership.loc[mask, "ticker"].unique().tolist())
    return result


def get_sector_map() -> dict:
    """Return {ticker: gics_sector} using the current classification.
    Note: this is NOT point-in-time -- a stock's sector classification can
    change over time (e.g. GICS 2018 reclassification), but Wikipedia only
    gives us the current mapping. Fine for most backtests; flag if you need
    point-in-time sector history too.
    """
    conn = _connect()
    df = pd.read_sql("SELECT ticker, gics_sector FROM constituents_current", conn)
    conn.close()
    return dict(zip(df["ticker"], df["gics_sector"]))


def get_sub_industry_map() -> dict:
    conn = _connect()
    df = pd.read_sql("SELECT ticker, gics_sub_industry FROM constituents_current", conn)
    conn.close()
    return dict(zip(df["ticker"], df["gics_sub_industry"]))


if __name__ == "__main__":
    # quick smoke test
    print("Members on 2020-01-15:", len(get_membership_on_date("2020-01-15")))
    print("Members on 2024-01-15:", len(get_membership_on_date("2024-01-15")))
    print("Members today (2026-08-01):", len(get_membership_on_date("2026-08-01")))

    # HES should be in the 2024 list but not a later one; XYZ vice versa
    m2024 = get_membership_on_date("2024-06-01")
    m2026 = get_membership_on_date("2026-06-01")
    print("\nHES in 2024-06-01 list:", "HES" in m2024)
    print("HES in 2026-06-01 list:", "HES" in m2026)
    print("XYZ in 2024-06-01 list:", "XYZ" in m2024)
    print("XYZ in 2026-06-01 list:", "XYZ" in m2026)
