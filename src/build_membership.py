"""
Reconstructs point-in-time S&P 500 membership intervals from:
  1. The current constituent list (each active ticker's most recent "date added")
  2. The historical additions/removals log (effective_date, added, removed)

Algorithm: walk the changes log in reverse-chronological order (most recent
first), maintaining a dict of "open" stints per ticker. Each change event
(date D: X added, Y removed) means: immediately before D, X was NOT a member
(its stint starts at D, already captured from the current-constituents table)
and Y WAS a member (Y's stint must extend backward from D; if we don't later
find an earlier "Y added" event in the log, its start is left unknown/None,
meaning "at least since before our log coverage begins").

Produces a `membership` table of (ticker, security, start_date, end_date)
intervals. end_date = NULL means still active today. start_date = NULL
means "unknown, but was already a member as of the earliest date our
changes log covers."
"""
import pandas as pd

current = pd.read_csv("parsed_current.csv")
current["date_added"] = pd.to_datetime(current["date_added"], errors="coerce")

changes = pd.read_csv("parsed_changes.csv")
changes["effective_date"] = pd.to_datetime(changes["effective_date"])
changes = changes.sort_values("effective_date", ascending=False).reset_index(drop=True)

EARLIEST_LOG_DATE = changes["effective_date"].min()

# ticker -> {"start": Timestamp or None, "end": Timestamp or None, "security": str}
# start=None means "still needs to be resolved by an earlier 'added' event"
open_stints = {}
closed_intervals = []  # list of dicts: ticker, security, start_date, end_date

# Step 1: open a stint for every currently-active ticker, from its date_added
for _, row in current.iterrows():
    open_stints[row["ticker"]] = {
        "start": row["date_added"] if pd.notna(row["date_added"]) else None,
        "end": None,  # still active today
        "security": row["security"],
    }

# Step 2: walk changes in reverse chronological order
for _, chg in changes.iterrows():
    d = chg["effective_date"]

    added_t = chg["added_ticker"] if pd.notna(chg["added_ticker"]) and chg["added_ticker"] else None
    removed_t = chg["removed_ticker"] if pd.notna(chg["removed_ticker"]) and chg["removed_ticker"] else None

    # --- resolve the ADDED ticker's stint start (confirms/handles edge cases) ---
    if added_t:
        if added_t in open_stints and open_stints[added_t]["start"] is None:
            # this stint's start was unresolved (ticker was removed at some
            # later date we already processed, and this is where it re-entered)
            open_stints[added_t]["start"] = d
            closed_intervals.append({
                "ticker": added_t,
                "security": open_stints[added_t]["security"],
                "start_date": d,
                "end_date": open_stints[added_t]["end"],
            })
            del open_stints[added_t]
        # else: matches the current table's date_added (already correctly
        # open with start=d), or is an older/duplicate record -- nothing to do

    # --- open (or extend) a stint for the REMOVED ticker ---
    if removed_t:
        if removed_t in open_stints:
            # ticker re-entered later (already tracked open with a *later*
            # end date) -- this earlier removal marks the end of an *older*
            # stint, distinct from the currently tracked one. Close the
            # currently tracked one only if this removal date precedes its end.
            # In practice this means: start a new not-yet-resolved stint only
            # if the currently open one has already been resolved (has a start).
            if open_stints[removed_t]["start"] is not None:
                # current open stint is fully resolved already; this removal
                # event refers to an even earlier, separate stint
                closed_intervals.append({
                    "ticker": removed_t,
                    "security": chg["removed_security"],
                    "start_date": open_stints[removed_t]["start"],
                    "end_date": open_stints[removed_t]["end"],
                })
                open_stints[removed_t] = {"start": None, "end": d, "security": chg["removed_security"]}
            # else: already an unresolved open stint ending at a later date
            # than this -- keep the later (more recent) end date, ignore
        else:
            open_stints[removed_t] = {"start": None, "end": d, "security": chg["removed_security"]}

# Step 3: flush all remaining open stints (currently active tickers, and any
# removed tickers whose start was never found within our log window)
for ticker, stint in open_stints.items():
    closed_intervals.append({
        "ticker": ticker,
        "security": stint["security"],
        "start_date": stint["start"],  # None = unknown, active since before log coverage
        "end_date": stint["end"],
    })

membership = pd.DataFrame(closed_intervals)
membership = membership.sort_values(["ticker", "start_date"], na_position="first").reset_index(drop=True)

print(f"Built {len(membership)} membership intervals for {membership['ticker'].nunique()} tickers")
print(f"Still-active (end_date is NULL): {membership['end_date'].isna().sum()}")
print(f"Historical/removed (end_date set): {membership['end_date'].notna().sum()}")
print(f"Unknown start (start_date is NULL): {membership['start_date'].isna().sum()}")

membership.to_csv("parsed_membership.csv", index=False)
print("\nSample removed-ticker intervals:")
print(membership[membership["end_date"].notna()].head(10).to_string(index=False))
