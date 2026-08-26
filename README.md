# S&P 500 Formulaic Alpha Research

An end-to-end quantitative research pipeline for testing systematic equity
alpha signals on the S&P 500: a survivorship-bias-free, point-in-time
database built from scratch; a library of formulaic alphas (WorldQuant
101-style) with a unified, alpha-agnostic point-in-time masking
architecture; a constrained portfolio optimizer; and a multi-stage
validation process (individual screening → correlation-aware ensembling →
walk-forward machine learning) used to reach an honest conclusion about
whether these signals have tradeable edge.

**This is independent research, not a live trading system.** No real
capital was ever deployed. The value of the project is in the
infrastructure and the validation discipline, not in a backtest number.

## Headline result

None of the 13 formulaic alphas implemented here — individually,
naively ensembled, or combined via Lasso/Random Forest/Gradient
Boosting with proper walk-forward validation — showed a robust,
out-of-sample Information Coefficient above the noise floor (roughly
`|IC| > 0.02`, `IC IR > 0.3`) on the S&P 500 large-cap universe. The
most promising result (gradient-boosted combination of all 13 signals)
reached an out-of-sample IC of ~0.012 with an IC IR of ~0.10 — a real,
measurable improvement over naive combination, but still short of a
genuinely tradeable signal, and before transaction costs are even
considered.

This is treated as a real finding, not a failure: these are published,
decade-old formulas being tested on the most heavily-arbitraged,
best-covered segment of the US equity market. The more interesting
result is *how* that conclusion was reached — see
[Key findings](#key-findings-and-what-they-mean) below.

## Repository structure

```
sp500-alpha-research/
├── src/
│   ├── download_prices.py       # batched, resumable OHLCV downloader (yfinance)
│   ├── build_membership.py      # reconstructs point-in-time index membership
│   ├── build_sp500_db.py        # builds the SQLite schema + loads membership data
│   ├── update_membership.py     # updates membership tables without touching price data
│   ├── add_price_tables.py      # adds the prices/download_log schema
│   ├── sp500_pit.py             # lightweight point-in-time query interface
│   ├── db_data_loader.py        # loads prices + point-in-time mask for the backtest
│   ├── alphas.py                # alpha signal library + unified masking helpers
│   ├── alpha3_backtest.py       # core backtest engine (optimizer, cost model, diagnostics)
│   ├── run_sp500_backtest.py    # full DB-driven backtest entry point
│   ├── screen_alphas.py         # fast IC-only screening (no optimizer) across alphas
│   ├── combine_alphas.py        # pairwise alpha correlation + naive ensembling
│   ├── ml_combine_alphas.py     # Lasso / Random Forest / Gradient Boosting combiner
│   └── sp500_pit.db             # starter DB: point-in-time membership only, no price data
├── data/raw_reference/          # parsed Wikipedia source data behind the membership DB
├── requirements.txt
└── README.md
```

## Architecture

**1. Point-in-time database** (`build_sp500_db.py`, `build_membership.py`,
`download_prices.py`)

Backtesting on "the current S&P 500 list" introduces survivorship bias —
it silently excludes every company that was ever acquired, went private,
or went bankrupt. This project instead reconstructs genuine index
membership history: which tickers were *actually* constituents on any
given historical date, built by walking a hand-researched log of
addition/removal events (sourced from Wikipedia, spanning 2009–2026)
backward from the current constituent list. Stored in SQLite with three
core tables: `constituents_current`, `index_changes`, and the derived
`membership` intervals table.

Price history (OHLCV) is downloaded separately via a resumable, batched
`yfinance` pipeline with automatic single-ticker fallback and per-ticker
failure logging — necessary at this scale, since a meaningful fraction
of tickers that have ever been in the index are now delisted (acquired,
taken private, or, in a handful of confirmed cases, genuinely bankrupt —
`SIVB`, `FRC`, `CHK`, `WIN`, `FTR`, `DF`, `MNK`, `DNR`, `BIG`, and `WFR`
are excluded from the backtest by default for this reason, since a
missing final price for a bankruptcy silently truncates a real loss to
0% rather than reflecting it).

**2. Alpha library with unified point-in-time masking** (`alphas.py`)

13 formulaic alphas are implemented, covering several structurally
different patterns: pure cross-sectional (`correlation(rank(x), rank(y))`),
pure time-series (`ts_rank`, `ts_argmax` applied per-stock with no
cross-stock comparison), and hybrids of both in the same formula.

The key design decision: rather than classifying each alpha's masking
strategy individually, every alpha function takes the point-in-time
eligibility mask directly and applies it via one of three small shared
helpers (`cross_sectional_rank`, `cross_sectional_corr`,
`mask_final_output`) at exactly the point in its formula where stocks are
actually compared to each other. This makes adding a new alpha in the
future a matter of translating its formula and calling the right helper
at its `rank(...)`/`correlation(...)` step — never re-deriving a
masking strategy from scratch.

**3. Constrained portfolio optimizer** (`alpha3_backtest.py`)

Daily/weekly rebalanced long-short portfolio construction via convex
optimization (`cvxpy`), enforcing dollar-neutrality and beta-neutrality
as hard constraints and a soft per-sector gross-exposure cap, with an
auto-scaled, smoothed turnover penalty to control transaction cost drag
without needing to re-tune the penalty magnitude for every new alpha's
different natural signal dispersion.

**4. Multi-stage signal validation** (`screen_alphas.py`,
`combine_alphas.py`, `ml_combine_alphas.py`)

A deliberately cheap-to-expensive escalation: fast IC-only screening
(no optimizer) across all candidate alphas and multiple forward-return
horizons → pairwise correlation analysis to check whether combining
alphas would add genuine diversification → naive rank-averaged ensembles
→ Lasso/Random Forest/Gradient Boosting combination with strict
walk-forward (expanding-window) validation and an embargo period to
prevent target-window leakage across fold boundaries.

## Key findings (and what they mean)

- **A published, well-known signal isn't automatically a good one.**
  These formulas are ~10 years old and public; the S&P 500 is the most
  arbitraged segment of the US market. Both individual-alpha and
  ensemble testing were consistent with this signal family having been
  competed away.
- **Combining weak signals has a real, quantifiable mathematical
  ceiling.** Naive rank-averaging of correlated alphas added almost
  nothing; a gradient-boosted nonlinear combination extracted ~2.7x more
  out-of-sample signal than the naive ensemble — real evidence of
  nonlinear structure existing, even though the absolute magnitude
  stayed below a tradeable threshold.
- **Several real methodology bugs were caught and fixed during
  development**, not glossed over — each is worth knowing about since
  they're common failure modes in quant ML generally, not specific to
  this project:
  - An early turnover-penalty implementation solved the optimizer daily
    and resampled to a weekly holding period *afterward* — meaning the
    penalty was silently anchored to a throwaway intermediate day's
    portfolio, not the actual previously-held one. Turnover measured at
    real rebalance dates was consequently near its theoretical ceiling
    despite the "penalty" being nonzero.
  - A naive ML target used same-day (`pct_change()`-only) returns
    labeled "forward return" — actually a backward-looking quantity,
    since the day's alpha was itself computed using that same day's
    close.
  - Once fixed to a genuine 5-day forward return (matching the
    strategy's weekly rebalance), a new leakage risk appeared: a
    training row near a walk-forward fold boundary would have its
    *label* computed from prices inside the test period. Fixed with an
    explicit embargo period.
  - A Lasso regularization strength that was well-calibrated for one
    fold silently collapsed to all-zero coefficients (and therefore an
    undefined correlation) in later folds as the training sample grew —
    caught by adding an explicit degenerate-prediction check rather than
    let a `NaN` pass silently.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

The included `src/sp500_pit.db` has point-in-time index membership
already built (2009–2026) but **no price data** — that has to be
downloaded locally since it's environment-specific and too large to
commit.

```bash
cd src

# 1. Add the price schema, then download OHLCV for every ticker that has
#    ever been an S&P 500 constituent (resumable -- safe to re-run)
python add_price_tables.py
python download_prices.py --start 2009-01-01 --end 2026-08-01

# 2. Also grab the benchmark used for beta-neutralization
python download_prices.py --tickers SPY --start 2009-01-01 --end 2026-08-01

# 3. Fast screen: rank all registered alphas by gross IC (no optimizer, seconds not minutes)
python screen_alphas.py --start 2009-01-01 --end 2026-08-01 --horizons 1,5,10,20

# 4. Check pairwise alpha correlation + naive ensemble tests
python combine_alphas.py --start 2009-01-01 --end 2026-08-01

# 5. ML-based combination with walk-forward validation
python ml_combine_alphas.py --start 2009-01-01 --end 2026-08-01

# 6. Full backtest (optimizer, turnover control, cost model, CAPM-adjusted IR) for one alpha
python run_sp500_backtest.py --start 2009-01-01 --end 2026-08-01 --alpha alpha3 --turnover-penalty 1.0
```

To rebuild the membership database from scratch (e.g. to extend the
history further, or refresh recent index changes):

```bash
python build_membership.py    # rebuilds membership.csv from the raw reference data
python build_sp500_db.py      # writes it into sp500_pit.db
```

## Limitations and honest caveats

- **Sector classification is not point-in-time** — it reflects each
  company's *current* GICS sector, not its historical one. A company
  that changed sectors would be misclassified for the sector-cap
  constraint in earlier periods.
- **~40 delisted tickers have no available price history** even after
  a manual investigation into the cause of each (documented during
  development) — mostly older, obscure pre-2015 acquisitions where
  Yahoo Finance's data retention is genuinely spotty. This thins the
  point-in-time universe somewhat in the earliest years of the backtest.
- **This is a large-cap-only result.** The natural next step (not yet
  built) is a similarly-constructed point-in-time database for a
  mid-cap index (e.g. S&P 400), where these same signals may have more
  room to work given lower analyst coverage and less competing
  quantitative capital.
- **All results are gross of realistic market impact modeling** — the
  cost model used is a simple turnover-proportional charge, not a
  market-impact or liquidity-aware model.
