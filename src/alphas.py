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


def cross_sectional_scale(df: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """
    The masked equivalent of WorldQuant notation `scale(x)`: masks first,
    then normalizes each day's cross-section so the absolute values sum to
    1 (x / sum(|x|) across eligible tickers that day). Like rank(...), this
    is inherently a cross-sectional operation -- it needs to know the full
    eligible universe's magnitudes that day to normalize correctly, so
    masking happens here, at the point of normalization, same principle
    as cross_sectional_rank.
    """
    masked = df.where(mask)
    daily_abs_sum = masked.abs().sum(axis=1)
    return masked.div(daily_abs_sum, axis=0)


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


def compute_alpha10(closes: pd.DataFrame, mask: pd.DataFrame, window: int = 4):
    """
    Alpha#10: rank(
        (0 < ts_min(delta(close, 1), 4))
            ? delta(close, 1)
            : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))
    )

    Same persistence/reversal switch as Alpha#9, but wrapped in an outer
    rank(...) -- a hybrid where the inner conditional is time-series (full
    unmasked history), and only the final rank(...) is cross-sectional
    (masked here).
    """
    delta1 = closes.diff(1)
    ts_min_w = delta1.rolling(window).min()
    ts_max_w = delta1.rolling(window).max()

    inner = (-1 * delta1).where((ts_min_w <= 0) & (ts_max_w >= 0), delta1)
    return cross_sectional_rank(inner, mask)


def compute_alpha14(closes: pd.DataFrame, opens: pd.DataFrame, volumes: pd.DataFrame,
                     mask: pd.DataFrame, rank_window: int = 3, corr_window: int = 10):
    """
    Alpha#14: (-1 * rank(delta(returns, 3))) * correlation(open, volume, 10)

    A product of a cross-sectional term (rank(delta(returns,3)), masked via
    cross_sectional_rank) and a pure time-series term (correlation(open,
    volume, 10), no rank -- computed on full history, unmasked). Since
    pandas propagates NaN through elementwise multiplication, the masked
    first factor already ensures the product is correctly NaN wherever
    the stock wasn't eligible -- no additional masking step needed on the
    second factor or the product itself.
    """
    returns = closes.pct_change()
    delta_returns = returns.diff(rank_window)
    rank_term = -1 * cross_sectional_rank(delta_returns, mask)

    corr_term = opens.rolling(corr_window).corr(volumes)
    corr_term = corr_term.replace([np.inf, -np.inf], np.nan)

    return rank_term * corr_term


def compute_alpha15(highs: pd.DataFrame, volumes: pd.DataFrame, mask: pd.DataFrame,
                     corr_window: int = 3, sum_window: int = 3):
    """
    Alpha#15: -1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3)

    Doubly cross-sectional, like Alpha#13: rank(high)/rank(volume) masked
    via cross_sectional_rank, their rolling correlation is a per-stock
    time series of an already-masked quantity, the OUTER rank(...) is a
    second cross-sectional step (masked again), and the final sum(...,3)
    is a plain time-series rolling sum needing no further masking (NaN
    propagates through it automatically).
    """
    rank_high = cross_sectional_rank(highs, mask)
    rank_volume = cross_sectional_rank(volumes, mask)
    corr = rank_high.rolling(corr_window).corr(rank_volume)
    corr = corr.replace([np.inf, -np.inf], np.nan)

    rank_corr = cross_sectional_rank(corr, mask)
    return -1 * rank_corr.rolling(sum_window).sum()


def compute_alpha19(closes: pd.DataFrame, mask: pd.DataFrame,
                     delta_window: int = 7, sum_window: int = 250):
    """
    Alpha#19: (-1 * sign(((close - delay(close, 7)) + delta(close, 7))))
                 * (1 + rank((1 + sum(returns, 250))))

    Hybrid: the sign(...) term is pure time-series (full unmasked
    history); rank((1 + sum(returns, 250))) is cross-sectional (masked via
    cross_sectional_rank). Same product-propagates-NaN logic as Alpha#14.
    Note the ~1-year (250 trading day) lookback for the sum(returns, ...)
    term -- a much longer window than most of the other alphas here.
    """
    returns = closes.pct_change()
    delay_close_7 = closes.shift(delta_window)
    delta_close_7 = closes.diff(delta_window)

    sign_term = -1 * np.sign((closes - delay_close_7) + delta_close_7)

    sum_returns = returns.rolling(sum_window).sum()
    rank_term = 1 + cross_sectional_rank(1 + sum_returns, mask)

    return sign_term * rank_term


def compute_alpha21(closes: pd.DataFrame, volumes: pd.DataFrame, mask: pd.DataFrame,
                     short_window: int = 2, long_window: int = 8, adv_window: int = 20):
    """
    Alpha#21: nested conditional comparing short vs. long moving averages
    (with a volatility band) and current volume vs. its own 20-day average
    -- purely time-series/pointwise, no rank anywhere in the formula.
    Mask the final output only.

        ((sum(close,8)/8 + stddev(close,8)) < sum(close,2)/2)
            ? -1
            : ((sum(close,2)/2 < (sum(close,8)/8 - stddev(close,8)))
                ? 1
                : ((volume/adv20 >= 1) ? 1 : -1))
    """
    ma_long = closes.rolling(long_window).mean()
    std_long = closes.rolling(long_window).std()
    ma_short = closes.rolling(short_window).mean()
    adv20 = volumes.rolling(adv_window).mean()
    vol_ratio = volumes / adv20

    cond1 = (ma_long + std_long) < ma_short
    cond2 = ma_short < (ma_long - std_long)
    cond3 = vol_ratio >= 1

    alpha = pd.DataFrame(1.0, index=closes.index, columns=closes.columns)
    alpha = alpha.where(cond3, -1.0)          # innermost branch
    alpha = alpha.where(~cond2, 1.0)          # cond2 overrides innermost
    alpha = alpha.where(~cond1, -1.0)         # cond1 overrides everything

    # preserve NaN wherever the underlying rolling calcs don't have enough history
    insufficient = ma_long.isna() | std_long.isna() | adv20.isna()
    alpha = alpha.where(~insufficient, np.nan)

    return mask_final_output(alpha, mask)


def compute_alpha24(closes: pd.DataFrame, mask: pd.DataFrame,
                     ma_window: int = 100, delta_window: int = 100, fallback_window: int = 3):
    """
    Alpha#24: if the 100-day moving average has grown less than 5% over
    the last 100 days (i.e. a long-term uptrend has stalled/flattened),
    bet on reversion from the local 100-day low; otherwise, fade the
    recent 3-day move. Purely time-series/pointwise, no rank anywhere.
    Mask the final output only.

        ((delta(sum(close,100)/100, 100) / delay(close,100)) <= 0.05)
            ? (-1 * (close - ts_min(close, 100)))
            : (-1 * delta(close, 3))
    """
    ma100 = closes.rolling(ma_window).mean()
    ma100_delta = ma100.diff(delta_window)
    delay_close_100 = closes.shift(delta_window)
    ratio = ma100_delta / delay_close_100

    ts_min_100 = closes.rolling(ma_window).min()
    branch_a = -1 * (closes - ts_min_100)
    branch_b = -1 * closes.diff(fallback_window)

    alpha = branch_a.where(ratio <= 0.05, branch_b)

    insufficient = ma100.isna() | ratio.isna()
    alpha = alpha.where(~insufficient, np.nan)

    return mask_final_output(alpha, mask)


def compute_alpha26(highs: pd.DataFrame, volumes: pd.DataFrame, mask: pd.DataFrame,
                     ts_rank_window: int = 5, corr_window: int = 5, ts_max_window: int = 3):
    """
    Alpha#26: -1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)

    Purely time-series composition -- ts_rank compares each stock only to
    its own history, the correlation of two such per-stock time series is
    still per-stock (no cross-sectional comparison anywhere), and ts_max
    is a rolling max. No cross-sectional step at all; mask the final
    output only.
    """
    ts_rank_volume = ts_rank(volumes, ts_rank_window)
    ts_rank_high = ts_rank(highs, ts_rank_window)

    corr = ts_rank_volume.rolling(corr_window).corr(ts_rank_high)
    corr = corr.replace([np.inf, -np.inf], np.nan)

    alpha = -1 * corr.rolling(ts_max_window).max()
    return mask_final_output(alpha, mask)


def compute_alpha17(closes: pd.DataFrame, volumes: pd.DataFrame, mask: pd.DataFrame,
                     ts_rank_close_window: int = 10, adv_window: int = 20, ts_rank_vol_window: int = 5):
    """
    Alpha#17: (-1 * rank(ts_rank(close, 10)))
                 * rank(delta(delta(close, 1), 1))
                 * rank(ts_rank(volume / adv20, 5))

    Triple product of three cross-sectional ranks, each built on a
    different time-series intermediate (ts_rank of close, second
    difference of close, ts_rank of relative volume). All three ranked
    terms masked independently via cross_sectional_rank; the product
    naturally stays NaN wherever any factor is.
    """
    ts_rank_close = ts_rank(closes, ts_rank_close_window)
    term1 = -1 * cross_sectional_rank(ts_rank_close, mask)

    second_diff = closes.diff(1).diff(1)
    term2 = cross_sectional_rank(second_diff, mask)

    adv20 = volumes.rolling(adv_window).mean()
    vol_ratio = volumes / adv20
    ts_rank_vol_ratio = ts_rank(vol_ratio, ts_rank_vol_window)
    term3 = cross_sectional_rank(ts_rank_vol_ratio, mask)

    return term1 * term2 * term3


def compute_alpha18(closes: pd.DataFrame, opens: pd.DataFrame, mask: pd.DataFrame,
                     std_window: int = 5, corr_window: int = 10):
    """
    Alpha#18: -1 * rank(stddev(abs(close - open), 5) + (close - open)
                         + correlation(close, open, 10))

    Time-series composite (stddev of the intraday range's magnitude, the
    signed intraday move itself, and a raw unranked close-open
    correlation) computed on full history, wrapped in a single outer
    rank(...) -- the one cross-sectional step, masked there.
    """
    intraday = closes - opens
    std_term = intraday.abs().rolling(std_window).std()
    corr_term = closes.rolling(corr_window).corr(opens)
    corr_term = corr_term.replace([np.inf, -np.inf], np.nan)

    inner = std_term + intraday + corr_term
    return -1 * cross_sectional_rank(inner, mask)


def compute_alpha20(closes: pd.DataFrame, opens: pd.DataFrame, highs: pd.DataFrame,
                     lows: pd.DataFrame, mask: pd.DataFrame):
    """
    Alpha#20: (-1 * rank(open - delay(high, 1)))
                 * rank(open - delay(close, 1))
                 * rank(open - delay(low, 1))

    Triple product of three cross-sectional ranks, each on a simple
    pointwise gap (today's open vs. yesterday's high/close/low) -- no
    rolling windows at all in the inputs, only in the sense that
    delay(x,1) is a 1-day shift.
    """
    term1 = -1 * cross_sectional_rank(opens - highs.shift(1), mask)
    term2 = cross_sectional_rank(opens - closes.shift(1), mask)
    term3 = cross_sectional_rank(opens - lows.shift(1), mask)
    return term1 * term2 * term3


def compute_alpha22(highs: pd.DataFrame, closes: pd.DataFrame, volumes: pd.DataFrame,
                     mask: pd.DataFrame, corr_window: int = 5, delta_window: int = 5,
                     std_window: int = 20):
    """
    Alpha#22: -1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20)))

    Mixed product: delta(correlation(high, volume, 5), 5) is pure
    time-series (the correlation itself is raw/unranked, and delta of a
    time series is still time-series -- no masking needed there);
    rank(stddev(close, 20)) is the one cross-sectional term, masked via
    cross_sectional_rank. Product propagates NaN correctly from the
    masked side.
    """
    corr = highs.rolling(corr_window).corr(volumes)
    corr = corr.replace([np.inf, -np.inf], np.nan)
    delta_corr = corr.diff(delta_window)

    std_close = closes.rolling(std_window).std()
    rank_term = cross_sectional_rank(std_close, mask)

    return -1 * (delta_corr * rank_term)


def compute_alpha28(lows: pd.DataFrame, highs: pd.DataFrame, closes: pd.DataFrame,
                     volumes: pd.DataFrame, mask: pd.DataFrame, adv_window: int = 20,
                     corr_window: int = 5):
    """
    Alpha#28: scale(correlation(adv20, low, 5) + (high + low)/2 - close)

    correlation(adv20, low, 5) is pure time-series (both sides are
    per-stock series, no cross-sectional comparison), combined with the
    pointwise (high+low)/2 - close term, then the WHOLE thing goes
    through scale(...) -- the one cross-sectional step (needs to know the
    full eligible universe's magnitudes that day to normalize), masked
    via cross_sectional_scale.
    """
    adv20 = volumes.rolling(adv_window).mean()
    corr = adv20.rolling(corr_window).corr(lows)
    corr = corr.replace([np.inf, -np.inf], np.nan)

    inner = corr + (highs + lows) / 2 - closes
    return cross_sectional_scale(inner, mask)


def compute_alpha34(closes: pd.DataFrame, mask: pd.DataFrame,
                     short_std_window: int = 2, long_std_window: int = 5):
    """
    Alpha#34: rank((1 - rank(stddev(returns, 2) / stddev(returns, 5)))
                     + (1 - rank(delta(close, 1))))

    Two inner cross-sectional rank terms (each masked independently, then
    each subtracted from 1) summed together, then wrapped in an OUTER
    rank(...) -- a third, final cross-sectional step, also masked. Good
    example of the helpers composing through a doubly-nested rank chain.
    """
    returns = closes.pct_change()
    std_ratio = returns.rolling(short_std_window).std() / returns.rolling(long_std_window).std()
    term1 = 1 - cross_sectional_rank(std_ratio, mask)

    delta_close = closes.diff(1)
    term2 = 1 - cross_sectional_rank(delta_close, mask)

    inner = term1 + term2
    return cross_sectional_rank(inner, mask)


def compute_alpha37(closes: pd.DataFrame, opens: pd.DataFrame, mask: pd.DataFrame,
                     corr_window: int = 200):
    """
    Alpha#37: rank(correlation(delay(open - close, 1), close, 200)) + rank(open - close)

    correlation(delay(open-close,1), close, 200) is pure time-series (a
    long 200-day lookback, raw/unranked inputs), wrapped in an outer
    rank(...) (masked); rank(open - close) is a separate cross-sectional
    rank on a pointwise term (also masked). The two ranked terms are summed.
    """
    lagged_intraday = (opens - closes).shift(1)
    corr = lagged_intraday.rolling(corr_window).corr(closes)
    corr = corr.replace([np.inf, -np.inf], np.nan)
    term1 = cross_sectional_rank(corr, mask)

    term2 = cross_sectional_rank(opens - closes, mask)
    return term1 + term2


def compute_alpha40(highs: pd.DataFrame, volumes: pd.DataFrame, mask: pd.DataFrame,
                     std_window: int = 10, corr_window: int = 10):
    """
    Alpha#40: -1 * rank(stddev(high, 10)) * correlation(high, volume, 10)

    Mixed product, same pattern as Alpha#22: rank(stddev(high,10)) is
    cross-sectional (masked); correlation(high, volume, 10) is pure
    time-series (raw, unranked, no masking needed there).
    """
    std_high = highs.rolling(std_window).std()
    rank_term = -1 * cross_sectional_rank(std_high, mask)

    corr = highs.rolling(corr_window).corr(volumes)
    corr = corr.replace([np.inf, -np.inf], np.nan)

    return rank_term * corr


def compute_alpha43(closes: pd.DataFrame, volumes: pd.DataFrame, mask: pd.DataFrame,
                     adv_window: int = 20, ts_rank_vol_window: int = 20,
                     delta_window: int = 7, ts_rank_delta_window: int = 8):
    """
    Alpha#43: ts_rank(volume / adv20, 20) * ts_rank(-1 * delta(close, 7), 8)

    Purely time-series: both factors are ts_rank of a per-stock quantity
    (no cross-sectional comparison anywhere in the formula). Mask the
    final product's output only.
    """
    adv20 = volumes.rolling(adv_window).mean()
    vol_ratio = volumes / adv20
    term1 = ts_rank(vol_ratio, ts_rank_vol_window)

    neg_delta7 = -1 * closes.diff(delta_window)
    term2 = ts_rank(neg_delta7, ts_rank_delta_window)

    alpha = term1 * term2
    return mask_final_output(alpha, mask)


def compute_alpha44(highs: pd.DataFrame, volumes: pd.DataFrame, mask: pd.DataFrame,
                     corr_window: int = 5):
    """
    Alpha#44: -1 * correlation(high, rank(volume), 5)

    Mixed-input correlation: high is raw/unranked (time-series), but
    rank(volume) IS cross-sectional -- masked via cross_sectional_rank.
    Only one side of the correlation needs masking; since rank(volume) is
    NaN on ineligible days, the rolling correlation naturally propagates
    that NaN into the result without needing to separately mask `high`.
    """
    rank_volume = cross_sectional_rank(volumes, mask)
    corr = highs.rolling(corr_window).corr(rank_volume)
    corr = corr.replace([np.inf, -np.inf], np.nan)
    return -1 * corr


def compute_alpha45(closes: pd.DataFrame, volumes: pd.DataFrame, mask: pd.DataFrame,
                     delay_window: int = 5, sum_window_a: int = 20, corr_window_a: int = 2,
                     sum_window_b: int = 5, sum_window_c: int = 20, corr_window_b: int = 2):
    """
    Alpha#45: -1 * ( rank(sum(delay(close, 5), 20) / 20)
                      * correlation(close, volume, 2)
                      * rank(correlation(sum(close, 5), sum(close, 20), 2)) )

    Triple product: a cross-sectional rank of a time-series-derived
    quantity (masked), a raw/unranked time-series correlation (no
    masking needed), and a cross-sectional rank of ANOTHER time-series
    correlation (masked). Demonstrates masking only ever applied at the
    two explicit rank(...) steps, nowhere else.
    """
    delayed_close_sum = closes.shift(delay_window).rolling(sum_window_a).sum() / sum_window_a
    term1 = cross_sectional_rank(delayed_close_sum, mask)

    term2 = closes.rolling(corr_window_a).corr(volumes)
    term2 = term2.replace([np.inf, -np.inf], np.nan)

    sum_close_5 = closes.rolling(sum_window_b).sum()
    sum_close_20 = closes.rolling(sum_window_c).sum()
    corr_of_sums = sum_close_5.rolling(corr_window_b).corr(sum_close_20)
    corr_of_sums = corr_of_sums.replace([np.inf, -np.inf], np.nan)
    term3 = cross_sectional_rank(corr_of_sums, mask)

    return -1 * (term1 * term2 * term3)


def compute_alpha46(closes: pd.DataFrame, mask: pd.DataFrame):
    """
    Alpha#46: three-branch momentum/reversal switch based on the
    acceleration of the close's 10-day drift:

        accel = (delay(close,20) - delay(close,10))/10 - (delay(close,10) - close)/10
        (0.25 < accel)  ? -1
        (accel < 0)     ? 1
        else            -1 * (close - delay(close, 1))

    Purely time-series/pointwise, no rank anywhere. Mask the final
    output only.
    """
    accel = (closes.shift(20) - closes.shift(10)) / 10 - (closes.shift(10) - closes) / 10

    alpha = -1 * closes.diff(1)               # innermost (else) branch
    alpha = alpha.where(~(accel < 0), 1.0)     # middle branch overrides
    alpha = alpha.where(~(accel > 0.25), -1.0)  # outer branch overrides all

    alpha = alpha.where(~accel.isna(), np.nan)
    return mask_final_output(alpha, mask)


def compute_alpha49(closes: pd.DataFrame, mask: pd.DataFrame):
    """
    Alpha#49: two-branch variant of Alpha#46's acceleration switch, with
    a -0.1 threshold instead of a three-way 0.25/0 split:

        accel = (delay(close,20) - delay(close,10))/10 - (delay(close,10) - close)/10
        (accel < -0.1) ? 1 : -1 * (close - delay(close, 1))

    Purely time-series/pointwise, no rank anywhere.
    """
    accel = (closes.shift(20) - closes.shift(10)) / 10 - (closes.shift(10) - closes) / 10
    alpha = (-1 * closes.diff(1)).where(~(accel < -0.1), 1.0)
    alpha = alpha.where(~accel.isna(), np.nan)
    return mask_final_output(alpha, mask)


def compute_alpha51(closes: pd.DataFrame, mask: pd.DataFrame):
    """
    Alpha#51: same family as Alpha#49, threshold -0.05 instead of -0.1.
    """
    accel = (closes.shift(20) - closes.shift(10)) / 10 - (closes.shift(10) - closes) / 10
    alpha = (-1 * closes.diff(1)).where(~(accel < -0.05), 1.0)
    alpha = alpha.where(~accel.isna(), np.nan)
    return mask_final_output(alpha, mask)


def compute_alpha53(closes: pd.DataFrame, highs: pd.DataFrame, lows: pd.DataFrame,
                     mask: pd.DataFrame, delta_window: int = 9):
    """
    Alpha#53: -1 * delta( ((close - low) - (high - close)) / (close - low), 9)

    Pointwise ratio, then a 9-day time-series delta -- no cross-sectional
    step anywhere. Mask the final output only.
    """
    ratio = ((closes - lows) - (highs - closes)) / (closes - lows)
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    alpha = -1 * ratio.diff(delta_window)
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
    "alpha10": {"func": compute_alpha10, "inputs": ("closes",)},
    "alpha12": {"func": compute_alpha12, "inputs": ("closes", "volumes")},
    "alpha13": {"func": compute_alpha13, "inputs": ("closes", "volumes")},
    "alpha14": {"func": compute_alpha14, "inputs": ("closes", "opens", "volumes")},
    "alpha15": {"func": compute_alpha15, "inputs": ("highs", "volumes")},
    "alpha19": {"func": compute_alpha19, "inputs": ("closes",)},
    "alpha21": {"func": compute_alpha21, "inputs": ("closes", "volumes")},
    "alpha24": {"func": compute_alpha24, "inputs": ("closes",)},
    "alpha26": {"func": compute_alpha26, "inputs": ("highs", "volumes")},
    "alpha33": {"func": compute_alpha33, "inputs": ("closes", "opens")},
    "alpha38": {"func": compute_alpha38, "inputs": ("closes", "opens")},
    "alpha54": {"func": compute_alpha54, "inputs": ("closes", "opens", "highs", "lows")},
    "alpha17": {"func": compute_alpha17, "inputs": ("closes", "volumes")},
    "alpha18": {"func": compute_alpha18, "inputs": ("closes", "opens")},
    "alpha20": {"func": compute_alpha20, "inputs": ("closes", "opens", "highs", "lows")},
    "alpha22": {"func": compute_alpha22, "inputs": ("highs", "closes", "volumes")},
    "alpha28": {"func": compute_alpha28, "inputs": ("lows", "highs", "closes", "volumes")},
    "alpha34": {"func": compute_alpha34, "inputs": ("closes",)},
    "alpha37": {"func": compute_alpha37, "inputs": ("closes", "opens")},
    "alpha40": {"func": compute_alpha40, "inputs": ("highs", "volumes")},
    "alpha43": {"func": compute_alpha43, "inputs": ("closes", "volumes")},
    "alpha44": {"func": compute_alpha44, "inputs": ("highs", "volumes")},
    "alpha45": {"func": compute_alpha45, "inputs": ("closes", "volumes")},
    "alpha46": {"func": compute_alpha46, "inputs": ("closes",)},
    "alpha49": {"func": compute_alpha49, "inputs": ("closes",)},
    "alpha51": {"func": compute_alpha51, "inputs": ("closes",)},
    "alpha53": {"func": compute_alpha53, "inputs": ("closes", "highs", "lows")},
    "alpha101": {"func": compute_alpha101, "inputs": ("closes", "opens", "highs", "lows")},
}
