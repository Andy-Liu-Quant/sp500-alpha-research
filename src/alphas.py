"""
Alpha signal implementations from the WorldQuant 101 Formulaic Alphas.

UNIFIED MASKING DESIGN -- one mechanism, decoupled from any specific alpha's
structure, correct for any future alpha without needing to classify it.

The key insight: masking correctness was never really about "alpha type" --
it's about WHERE in a formula stocks get compared to each other. Every one
of these formulas follows the same underlying pattern:
  - TIME-SERIES operations (delta, stddev, ts_rank, ts_argmax, decay_linear,
    ...) transform each stock's OWN history. They should always run on the
    full, unmasked price/volume history -- masking here would insert
    artificial gaps into a stock's genuine trading history for no benefit,
    since there's no cross-stock comparison happening yet.
  - CROSS-SECTIONAL operations (rank(...), correlation(rank(x), rank(y), d))
    are the ONLY places stocks actually get compared to each other. THIS is
    where masking matters: an ineligible stock's data must not contaminate
    the comparison for stocks that genuinely were eligible that day.

So instead of classifying each alpha as "pre/post/internal" masked (which
requires reasoning about its specific structure every time), every alpha
function here just takes `mask` as a parameter and calls one of the shared
helpers below -- cross_sectional_rank() or cross_sectional_corr() --
at exactly the point(s) in the formula where the WorldQuant notation says
`rank(...)` or `correlation(...)`. If an alpha has no cross-sectional
step at all, it calls mask_final_output() once at the very end.

This makes the masking mechanism genuinely alpha-agnostic: adding any
future alpha only requires knowing where ITS formula says `rank(...)`,
not reasoning about pre/post/internal masking strategy from scratch.
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Shared cross-sectional helpers -- the ONE place masking logic lives.
# Every alpha's rank(...) / correlation(...) step should go through one
# of these, and nowhere else in an alpha function should `mask` be
# applied to input data.
# ---------------------------------------------------------------------

def cross_sectional_rank(df: pd.DataFrame, mask: pd.DataFrame, pct: bool = True) -> pd.DataFrame:
    """
    The masked equivalent of WorldQuant notation `rank(x)`: masks to NaN
    on non-eligible (date, ticker) cells FIRST, then ranks cross-sectionally
    (axis=1, i.e. across tickers, for each date) so the ranking on any given
    day only ever reflects genuinely eligible peers that day.
    """
    return df.where(mask).rank(axis=1, pct=pct)


def cross_sectional_corr(x: pd.DataFrame, y: pd.DataFrame, mask: pd.DataFrame,
                          window: int) -> pd.DataFrame:
    """
    The masked equivalent of WorldQuant notation `correlation(x, y, d)`
    where x and y are themselves cross-sectional ranks (e.g. Alpha#3's
    correlation(rank(open), rank(volume), d)): masks x and y first, so the
    ranks feeding the rolling correlation only reflect genuinely eligible
    peers, then computes the rolling correlation per ticker.
    """
    x_masked = x.where(mask)
    y_masked = y.where(mask)
    corr = x_masked.rolling(window).corr(y_masked)
    # pandas rolling .corr() can occasionally emit +-inf from floating-point
    # division by near-zero variance; correlation is bounded [-1,1]
    return corr.replace([np.inf, -np.inf], np.nan)


def cross_sectional_covariance(x: pd.DataFrame, y: pd.DataFrame, mask: pd.DataFrame,
                                window: int) -> pd.DataFrame:
    """
    The masked equivalent of WorldQuant notation `covariance(x, y, d)` where
    x and y are themselves cross-sectional ranks (e.g. Alpha#13's
    covariance(rank(close), rank(volume), d)): masks x and y first, then
    computes the rolling covariance per ticker. Mirrors cross_sectional_corr.
    """
    x_masked = x.where(mask)
    y_masked = y.where(mask)
    return x_masked.rolling(window).cov(y_masked)


def mask_final_output(df: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """
    For alphas with NO cross-sectional step at all (e.g. Alpha#7): the
    "cross-sectional point" is trivially the end of the formula, so this
    just masks the final output. Kept as its own named function (rather
    than a bare .where(mask) inline) so every alpha's masking call is
    visually consistent and self-documenting.
    """
    return df.where(mask)


# ---------------------------------------------------------------------
# Time-series primitives (no masking logic -- these always operate on
# whatever DataFrame they're given, full or masked, and don't care)
# ---------------------------------------------------------------------

def ts_rank(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Ts_Rank(x, d): the time-series percentile rank of today's value of x
    within its own trailing d-day window (per stock -- no cross-sectional
    comparison). Thin wrapper around pandas' built-in rolling rank, kept as
    a named function so alpha implementations read closer to the WorldQuant
    notation and so this logic lives in one place.
    """
    return df.rolling(window).rank(pct=True)


def ts_argmax(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Ts_ArgMax(x, d): how many days ago (within the trailing d-day window,
    including today) the maximum value of x occurred. Convention used here
    (matching the most common public implementations of the WorldQuant 101
    formulas): 0 = the max occurred today (most recent day in the window),
    d-1 = the max occurred d-1 days ago (the oldest day in the window).
    """
    def _argmax_days_ago(x):
        return window - 1 - np.argmax(x)
    return df.rolling(window).apply(_argmax_days_ago, raw=True)


# ---------------------------------------------------------------------
# Alpha implementations. Every function takes `mask` as its last
# parameter and calls a cross_sectional_*/mask_final_output helper at
# exactly the point(s) corresponding to `rank(...)`/`correlation(...)`
# in the WorldQuant formula notation. Nothing else needs masking.
# ---------------------------------------------------------------------

def compute_alpha1(closes: pd.DataFrame, mask: pd.DataFrame,
                    stddev_window: int = 20, argmax_window: int = 5):
    """
    Alpha#1: rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2), 5)) - 0.5
    """
    returns = closes.pct_change()
    stddev20 = returns.rolling(stddev_window).std()

    x = stddev20.where(returns < 0, closes)               # (returns < 0) ? stddev : close
    signed_power = np.sign(x) * (x ** 2)                   # SignedPower(x, 2)
    ts_argmax_result = ts_argmax(signed_power, argmax_window)  # Ts_ArgMax(..., 5)

    # formula's rank(...) step -- the ONE place masking applies
    alpha = cross_sectional_rank(ts_argmax_result, mask) - 0.5
    return alpha


def compute_alpha3(opens: pd.DataFrame, volumes: pd.DataFrame, mask: pd.DataFrame,
                    window: int = 10):
    """
    Alpha#3: -1 * correlation(rank(open), rank(volume), window)
    """
    # formula's rank(open), rank(volume) steps -- masked cross-sectionally
    rank_open = cross_sectional_rank(opens, mask)
    rank_volume = cross_sectional_rank(volumes, mask)

    # correlation(...) of two already-masked-and-ranked series -- no
    # further masking needed here, mask was already applied above
    rolling_corr = rank_open.rolling(window).corr(rank_volume)
    rolling_corr = rolling_corr.replace([np.inf, -np.inf], np.nan)
    return -1 * rolling_corr


def compute_alpha7(closes: pd.DataFrame, volumes: pd.DataFrame, mask: pd.DataFrame,
                    delta_window: int = 7, rank_window: int = 60, adv_window: int = 20,
                    use_dollar_volume: bool = True):
    """
    Alpha#7: (adv20 < volume)
                ? (-1 * ts_rank(abs(delta(close, 7)), 60) * sign(delta(close, 7)))
                : (-1 * 1)

    No cross-sectional step anywhere in this formula (ts_rank compares each
    stock only to its own trailing history) -- masking applies once, to the
    final output, via mask_final_output().

    On volume-spike days (today's volume > its own 20-day average): bets
    against the stock's recent 7-day price move, sized by how extreme that
    move was relative to the stock's own trailing 60 days (contrarian/
    reversal, volume-confirmed). On quiet-volume days: flat -1 for every
    stock (no signal content, constant mild short bias).

    NOTE on adv20 vs volume: the WorldQuant 101 paper's own data glossary
    defines adv{d} as average daily DOLLAR volume, while the bare `volume`
    field is SHARE volume -- so the literal formula compares dollar-volume-
    average against share-volume, which is dimensionally inconsistent.
    use_dollar_volume=True (default) resolves this by using dollar volume
    (close * volume) consistently on both sides of the comparison. Set to
    False for the literal-but-dimensionally-odd share-volume reading.
    """
    if use_dollar_volume:
        activity = closes * volumes
    else:
        activity = volumes

    adv20 = activity.rolling(adv_window).mean()
    delta7 = closes.diff(delta_window)
    abs_delta7 = delta7.abs()
    sign7 = np.sign(delta7)

    ts_rank_result = ts_rank(abs_delta7, rank_window)

    condition = activity > adv20
    alpha_spike = -1 * ts_rank_result * sign7
    alpha_quiet = pd.DataFrame(-1.0, index=closes.index, columns=closes.columns)

    alpha = alpha_spike.where(condition, alpha_quiet)

    insufficient_history = adv20.isna() | ts_rank_result.isna()
    alpha = alpha.where(~insufficient_history, np.nan)

    # no cross-sectional step in this formula -- mask the final output
    alpha = mask_final_output(alpha, mask)
    return alpha


def compute_alpha2(closes: pd.DataFrame, opens: pd.DataFrame, volumes: pd.DataFrame,
                    mask: pd.DataFrame, window: int = 6):
    """
    Alpha#2: -1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6)

    Same structural pattern as Alpha#3: two cross-sectional ranks (masked
    via cross_sectional_rank), then a per-stock rolling correlation between
    them (already-masked inputs, no further masking needed).
    """
    log_volume = np.log(volumes)
    delta_log_vol = log_volume.diff(2)
    intraday_return = (closes - opens) / opens

    rank_delta_vol = cross_sectional_rank(delta_log_vol, mask)
    rank_intraday = cross_sectional_rank(intraday_return, mask)

    corr = rank_delta_vol.rolling(window).corr(rank_intraday)
    corr = corr.replace([np.inf, -np.inf], np.nan)
    return -1 * corr


def compute_alpha4(lows: pd.DataFrame, mask: pd.DataFrame, window: int = 9):
    """
    Alpha#4: -1 * Ts_Rank(rank(low), 9)

    Hybrid, but a simple one: rank(low) is cross-sectional (masked via
    cross_sectional_rank), and Ts_Rank is then applied to that ALREADY-
    masked-and-ranked series -- no separate masking step needed afterward,
    since the cross-sectional rank already produced NaN on ineligible days
    and the time-series Ts_Rank naturally propagates that.
    """
    rank_low = cross_sectional_rank(lows, mask)
    return -1 * ts_rank(rank_low, window)


def compute_alpha6(opens: pd.DataFrame, volumes: pd.DataFrame, mask: pd.DataFrame,
                    window: int = 10):
    """
    Alpha#6: -1 * correlation(open, volume, 10)

    Pure time-series -- correlates RAW open and volume per stock (unlike
    Alpha#2/#3, there's no rank(...) wrapping either side, so this is not
    a cross-sectional comparison at all). Compute on full unmasked history,
    mask only the final output.
    """
    corr = opens.rolling(window).corr(volumes)
    corr = corr.replace([np.inf, -np.inf], np.nan)
    alpha = -1 * corr
    return mask_final_output(alpha, mask)


def compute_alpha9(closes: pd.DataFrame, mask: pd.DataFrame, window: int = 5):
    """
    Alpha#9: (0 < ts_min(delta(close, 1), 5))
                ? delta(close, 1)
                : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))

    Pure time-series (a persistence/reversal switch: rides a clean sustained
    move, fades an ambiguous one) -- no cross-sectional step anywhere, mask
    the final output only.
    """
    delta1 = closes.diff(1)
    ts_min5 = delta1.rolling(window).min()
    ts_max5 = delta1.rolling(window).max()

    alpha = (-1 * delta1).where((ts_min5 <= 0) & (ts_max5 >= 0), delta1)
    # equivalent to the nested ternary: if ts_min>0 use delta1; elif ts_max<0
    # use delta1; else use -delta1 -- both "clean move" branches return
    # delta1, so they collapse into one condition check against the "mixed"
    # (fade) case
    return mask_final_output(alpha, mask)


def compute_alpha12(closes: pd.DataFrame, volumes: pd.DataFrame, mask: pd.DataFrame):
    """
    Alpha#12: sign(delta(volume, 1)) * (-1 * delta(close, 1))

    Pure time-series, no rolling window even -- pointwise per stock, per
    day. No cross-sectional step, mask the final output only.
    """
    alpha = np.sign(volumes.diff(1)) * (-1 * closes.diff(1))
    return mask_final_output(alpha, mask)


def compute_alpha13(closes: pd.DataFrame, volumes: pd.DataFrame, mask: pd.DataFrame,
                     window: int = 5):
    """
    Alpha#13: -1 * rank(covariance(rank(close), rank(volume), 5))

    Doubly cross-sectional: rank(close)/rank(volume) are masked via
    cross_sectional_rank, their rolling covariance is a per-stock time
    series of an already-masked quantity (cross_sectional_covariance), and
    then the OUTER rank(...) is a SECOND cross-sectional step needing its
    own masking pass -- demonstrates the helpers compose cleanly for
    formulas with more than one rank(...) in sequence.
    """
    rank_close = cross_sectional_rank(closes, mask)
    rank_volume = cross_sectional_rank(volumes, mask)
    cov = rank_close.rolling(window).cov(rank_volume)
    return -1 * cross_sectional_rank(cov, mask)


def compute_alpha33(closes: pd.DataFrame, opens: pd.DataFrame, mask: pd.DataFrame):
    """
    Alpha#33: rank(-1 * (1 - (open / close)))

    Single cross-sectional step at the very end -- the open/close ratio
    itself is a pointwise (non-rolling) per-stock quantity that doesn't
    need masking on its own, since nothing is compared across stocks until
    the final rank(...).
    """
    x = -1 * (1 - (opens / closes))
    return cross_sectional_rank(x, mask)


def compute_alpha38(closes: pd.DataFrame, opens: pd.DataFrame, mask: pd.DataFrame,
                     window: int = 10):
    """
    Alpha#38: -1 * rank(Ts_Rank(close, 10)) * rank(close / open)

    Two independent cross-sectional rank(...) calls multiplied together:
    Ts_Rank(close, 10) is computed on full unmasked history first (it's
    time-series, per stock), then wrapped in rank(...) (masked); close/open
    is a pointwise ratio, also wrapped in its own rank(...) (masked).
    """
    ts_rank_close = ts_rank(closes, window)
    rank_ts_rank_close = cross_sectional_rank(ts_rank_close, mask)
    rank_ratio = cross_sectional_rank(closes / opens, mask)
    return -1 * rank_ts_rank_close * rank_ratio


def compute_alpha54(closes: pd.DataFrame, opens: pd.DataFrame, highs: pd.DataFrame,
                     lows: pd.DataFrame, mask: pd.DataFrame):
    """
    Alpha#54: (-1 * ((low - close) * (open^5))) / ((low - high) * (close^5))

    Pure pointwise formula -- no rolling window, no cross-sectional
    comparison anywhere. Mask the final output only.
    """
    numerator = -1 * ((lows - closes) * (opens ** 5))
    denominator = (lows - highs) * (closes ** 5)
    alpha = numerator / denominator
    alpha = alpha.replace([np.inf, -np.inf], np.nan)
    return mask_final_output(alpha, mask)


def compute_alpha101(closes: pd.DataFrame, opens: pd.DataFrame, highs: pd.DataFrame,
                      lows: pd.DataFrame, mask: pd.DataFrame):
    """
    Alpha#101: (close - open) / ((high - low) + 0.001)

    The simplest alpha in the whole set -- pure pointwise, no rolling
    window, no cross-sectional comparison. Mask the final output only.
    """
    alpha = (closes - opens) / ((highs - lows) + 0.001)
    return mask_final_output(alpha, mask)


# ---------------------------------------------------------------------
# Registry. No masking-strategy field needed anymore -- every alpha
# function has the same signature shape (ends with `mask`), so the
# dispatcher just always passes it through.
# ---------------------------------------------------------------------
ALPHA_REGISTRY = {
    "alpha1": {"func": compute_alpha1, "inputs": ("closes",)},
    "alpha2": {"func": compute_alpha2, "inputs": ("closes", "opens", "volumes")},
    "alpha3": {"func": compute_alpha3, "inputs": ("opens", "volumes")},
    "alpha4": {"func": compute_alpha4, "inputs": ("lows",)},
    "alpha6": {"func": compute_alpha6, "inputs": ("opens", "volumes")},
    "alpha7": {"func": compute_alpha7, "inputs": ("closes", "volumes")},
    "alpha9": {"func": compute_alpha9, "inputs": ("closes",)},
    "alpha12": {"func": compute_alpha12, "inputs": ("closes", "volumes")},
    "alpha13": {"func": compute_alpha13, "inputs": ("closes", "volumes")},
    "alpha33": {"func": compute_alpha33, "inputs": ("closes", "opens")},
    "alpha38": {"func": compute_alpha38, "inputs": ("closes", "opens")},
    "alpha54": {"func": compute_alpha54, "inputs": ("closes", "opens", "highs", "lows")},
    "alpha101": {"func": compute_alpha101, "inputs": ("closes", "opens", "highs", "lows")},
}
