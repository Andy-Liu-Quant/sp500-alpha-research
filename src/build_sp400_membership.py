"""
Reconstructs point-in-time S&P 400 membership intervals -- same algorithm
as build_membership.py (S&P 500), with one adaptation: the S&P 400
Wikipedia page's current-constituents table has no "Date added" column
(unlike S&P 500's), so every currently-active ticker's stint opens with
an UNRESOLVED start date by default. The backward walk through the
changes log then resolves as many of these as it can (any ticker whose
most recent entry happened within our log's coverage, Sept 2014 onward);
any ticker that has been a constituent since before that window remains
"start unknown, member since before log coverage" -- the same fallback
convention used throughout this project for the S&P 500 build.
"""
import pandas as pd

current = pd.read_csv("parsed_sp400_current.csv")

changes = pd.read_csv("parsed_sp400_changes.csv")
changes["effective_date"] = pd.to_datetime(changes["effective_date"])
changes = changes.sort_values("effective_date", ascending=False).reset_index(drop=True)

EARLIEST_LOG_DATE = changes["effective_date"].min()
print(f"Changes log coverage starts: {EARLIEST_LOG_DATE.date()}")

open_stints = {}
closed_intervals = []

# Step 1: open every currently-active ticker's stint -- but WITHOUT a known
# start date this time (no date_added column available for S&P 400)
for _, row in current.iterrows():
    open_stints[row["ticker"]] = {
        "start": None,  # unresolved -- will try to resolve via changes log below
        "end": None,    # still active today
        "security": row["security"],
    }

# Step 2: walk changes in reverse chronological order (identical algorithm
# to build_membership.py)
for _, chg in changes.iterrows():
    d = chg["effective_date"]

    added_t = chg["added_ticker"] if pd.notna(chg["added_ticker"]) and chg["added_ticker"] else None
    removed_t = chg["removed_ticker"] if pd.notna(chg["removed_ticker"]) and chg["removed_ticker"] else None

    if added_t:
        if added_t in open_stints and open_stints[added_t]["start"] is None:
            open_stints[added_t]["start"] = d
            closed_intervals.append({
                "ticker": added_t,
                "security": open_stints[added_t]["security"],
                "start_date": d,
                "end_date": open_stints[added_t]["end"],
            })
            del open_stints[added_t]

    if removed_t:
        if removed_t in open_stints:
            if open_stints[removed_t]["start"] is not None:
                closed_intervals.append({
                    "ticker": removed_t,
                    "security": chg["removed_security"],
                    "start_date": open_stints[removed_t]["start"],
                    "end_date": open_stints[removed_t]["end"],
                })
                open_stints[removed_t] = {"start": None, "end": d, "security": chg["removed_security"]}
        else:
            open_stints[removed_t] = {"start": None, "end": d, "security": chg["removed_security"]}

# Step 3: flush all remaining open stints
for ticker, stint in open_stints.items():
    closed_intervals.append({
        "ticker": ticker,
        "security": stint["security"],
        "start_date": stint["start"],
        "end_date": stint["end"],
    })

membership = pd.DataFrame(closed_intervals)
membership = membership.sort_values(["ticker", "start_date"], na_position="first").reset_index(drop=True)

print(f"Built {len(membership)} membership intervals for {membership['ticker'].nunique()} tickers")
print(f"Still-active (end_date is NULL): {membership['end_date'].isna().sum()}")
print(f"Historical/removed (end_date set): {membership['end_date'].notna().sum()}")
print(f"Unknown start (start_date is NULL): {membership['start_date'].isna().sum()}")
print(f"  -- of which currently-active with unknown start: "
      f"{membership[membership['end_date'].isna() & membership['start_date'].isna()].shape[0]} "
      f"(no 'date added' column was available for these, unlike the S&P 500 build)")

membership.to_csv("parsed_sp400_membership.csv", index=False)
