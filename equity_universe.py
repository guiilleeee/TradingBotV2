"""Equity universe (Financial Modeling Prep) and screening signal (yfinance).

Two things this module is careful about:

1. **Index membership vs. liquidity are different questions.** FMP's Nasdaq
   constituent endpoint has, at different times, meant either "Nasdaq-100" or
   "every stock listed on the Nasdaq exchange" (several thousand names) -- and
   which one it currently means cannot be verified without a live API key (every
   FMP endpoint, including ones that would just tell you the route exists,
   requires a valid key; an invalid key returns the same 401 for a real path and
   a made-up one). Rather than silently trust a guess, `fetch_nasdaq_constituents`
   sanity-checks the returned count against `NASDAQ_100_MAX_PLAUSIBLE_SIZE` and
   falls back to treating it as unavailable if it looks like the whole exchange.
   This is intentionally conservative: better to fall back to S&P 500 alone for a
   week than to silently widen the tradeable universe to thousands of illiquid
   tickers because a schema assumption was wrong.
2. **Every network call degrades, nothing raises past this module.** A stale
   universe (S&P 500 only, or last week's cache) is always a safe outcome for a
   weekly screen; a crashed screening job is not allowed to be one bad HTTP call
   away, per the brief's low-blast-radius requirement.

Why the Layer 1/2 signal comes from yfinance, not FMP's market movers
-----------------------------------------------------------------------
An earlier version scored liquidity and momentum from FMP's most-actives /
biggest-gainers / biggest-losers endpoints, intersected with the S&P 500 +
Nasdaq universe. Diagnosed against a real run: those endpoints work (50 real
rows each), but they're whole-market scans dominated by small, volatile,
non-index names (BTAI, CHPT, ADBT-shaped tickers) -- large, already-liquid
index constituents rarely show up on a "biggest mover in the entire market"
list. Intersecting that against 503 index names left only ~2 with any signal,
and the other 3 of 5 selected symbols were meaningless alphabetical backfill
presented as if they'd been chosen for a reason. That's a structural mismatch
between the data source and the question being asked ("what's moving in OUR
universe"), not a bug in the intersection logic -- so `fetch_universe_price_data`
measures the universe directly instead of hoping it overlaps someone else's list.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Sequence, Set

import pandas as pd
import requests
import yfinance as yf

from secrets_redaction import sanitize

FMP_BASE_URL = "https://financialmodelingprep.com"
HTTP_TIMEOUT = 30.0
# Free tier is 250 requests/day (confirmed against FMP's published pricing page).
# This module makes at most ~4 FMP calls per run (2 constituent endpoints x up
# to 2 path attempts each -- volume/momentum no longer come from FMP at all,
# see the module docstring) and this job runs weekly, so the daily quota is
# never remotely at risk -- but a small pause between calls is cheap insurance
# against an undocumented per-minute limit.
INTER_CALL_DELAY_SECONDS = 0.4

# A real Nasdaq-100 has ~100-105 members (multiple share classes for a couple of
# constituents). The full Nasdaq exchange listing has thousands. If FMP's
# "constituent" endpoint ever returns something in between, this threshold
# decides whether to trust it as index membership.
NASDAQ_100_MAX_PLAUSIBLE_SIZE = 160

WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Layer-1 volume floor for equities. S&P 500 / Nasdaq-100 membership already
# excludes almost all illiquid names by construction, so this is defense in
# depth rather than the primary filter -- unlike the crypto side, where
# Hyperliquid's permissionless listing makes volume the *main* line of defense.
MIN_EQUITY_VOLUME = 100_000.0

# Layer-1/2 blend: liquidity is weighted higher than momentum because Layer 1's
# job is safety (don't hand the model something illiquid), and momentum is the
# "trending" tiebreaker on top of that -- same split used on the crypto side in
# screening.py, kept consistent across asset classes on purpose.
VOLUME_WEIGHT = 0.6
MOMENTUM_WEIGHT = 0.4


class FMPError(RuntimeError):
    """Raised for a hard FMP failure the caller should know about (missing key)."""


def _api_key() -> str:
    key = os.environ.get("FMP_API_KEY")
    if not key:
        raise FMPError(
            "FMP_API_KEY is not set. Sign up for a free key at "
            "financialmodelingprep.com and set it as the FMP_API_KEY environment "
            "variable / GitHub secret."
        )
    return key


def _get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """One FMP GET call. Raises on any failure -- callers decide how to degrade.

    Whatever this raises is guaranteed not to carry the literal API key in its
    message, however far it propagates. Unlike every other credential this
    project holds (sent via header or signature), FMP's key travels as a query-
    string parameter -- so a raw HTTPError's default string form embeds it
    directly, in the URL, in plain text. Caught and re-raised as a sanitized
    FMPError here, once, rather than depending on every future caller (or a
    debug print added to one later) to remember to scrub it at whatever print
    or log site the exception eventually reaches.
    """
    query = dict(params or {})
    query["apikey"] = _api_key()
    try:
        resp = requests.get(f"{FMP_BASE_URL}{path}", params=query, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        # FMP returns a 200 with an {"Error Message": ...} body for some failure
        # modes (bad params, plan-gated endpoints) rather than a 4xx status.
        if isinstance(data, dict) and "Error Message" in data:
            raise FMPError(f"FMP {path}: {data['Error Message']}")
    except Exception as exc:
        raise FMPError(sanitize(f"{type(exc).__name__}: {exc}")) from None
    time.sleep(INTER_CALL_DELAY_SECONDS)
    return data


def _extract_symbols(rows: Any) -> List[str]:
    """Pull `symbol` out of a list of FMP row dicts, skipping anything malformed."""
    if not isinstance(rows, list):
        return []
    out = []
    for row in rows:
        if isinstance(row, dict) and row.get("symbol"):
            out.append(str(row["symbol"]).strip().upper())
    return out


# --------------------------------------------------------------------- S&P 500


def _scrape_sp500_from_wikipedia() -> List[str]:
    """Fallback S&P 500 membership, scraped from Wikipedia's constituents table.

    No API key needed, so this is also what a cycle falls back to when
    FMP_API_KEY is entirely unset. The table's id ("constituents") and its
    Symbol-first column layout are stable, well-known structural facts about
    this specific page, not a schema this project controls.
    """
    import io

    resp = requests.get(
        WIKIPEDIA_SP500_URL, timeout=HTTP_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"}
    )
    resp.raise_for_status()
    # StringIO, not the bare response text: passing a raw HTML string directly to
    # read_html makes some pandas/lxml combinations treat it as a filename to open
    # rather than markup to parse, which fails on real page content every time.
    tables = pd.read_html(io.StringIO(resp.text), attrs={"id": "constituents"})
    if not tables:
        raise FMPError("Wikipedia S&P 500 page: no table with id='constituents' found")
    symbols = tables[0]["Symbol"].astype(str).str.strip().str.upper().tolist()
    # Wikipedia uses a dot for share classes (BRK.B); yfinance and every other
    # symbol we handle uses a dash. Normalise so this list is usable downstream.
    return [s.replace(".", "-") for s in symbols if s]


def fetch_sp500_constituents() -> List[str]:
    """S&P 500 membership: FMP stable, then FMP legacy, then Wikipedia.

    Never raises -- a totally failed fetch returns [], and the caller (
    `build_equity_universe`) treats an empty S&P 500 the same as an empty
    Nasdaq list: the universe just ends up smaller, never absent.
    """
    for path in ("/stable/sp-500", "/api/v3/sp500_constituent"):
        try:
            symbols = _extract_symbols(_get(path))
            if symbols:
                return symbols
        except Exception:
            continue

    try:
        return _scrape_sp500_from_wikipedia()
    except Exception:
        return []


# ---------------------------------------------------------------------- Nasdaq


def fetch_nasdaq_constituents() -> List[str]:
    """Nasdaq-100 membership if FMP's data plausibly is that; else [].

    See the module docstring: this endpoint's exact scope cannot be confirmed
    without a live key, and a size that looks like "the whole Nasdaq exchange"
    is treated as untrustworthy for this purpose and dropped rather than
    silently widening the universe to thousands of names.
    """
    for path in ("/stable/nasdaq-constituent", "/api/v3/nasdaq_constituent"):
        try:
            symbols = _extract_symbols(_get(path))
        except Exception:
            continue

        if not symbols:
            continue
        if len(symbols) > NASDAQ_100_MAX_PLAUSIBLE_SIZE:
            # This is almost certainly the full exchange listing, not the -100
            # index. Don't use it, but don't treat the call as a failure either
            # -- fall through to returning [] rather than trying the legacy path
            # too, since it would very likely have the same scope problem.
            return []
        return symbols

    return []


# ------------------------------------------------------------- price data


# yfinance's own return-value contract when handed an empty ticker list is to
# raise inside pandas.concat ("No objects to concatenate") rather than return
# an empty frame -- confirmed live. Guarded for explicitly rather than letting
# an empty universe (already handled upstream, but cheap to guard here too)
# turn into a confusing pandas traceback.
_MIN_TRADING_DAYS_FOR_MOMENTUM = 2
PRICE_DATA_FETCH_PERIOD = "5d"


def fetch_universe_price_data(symbols: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Real momentum and volume for the whole equity universe, one batch call.

    Replaces the earlier FMP-movers-based signal (see the module docstring for
    why): `yf.download()` accepts the full symbol list in a single request --
    verified live against all 503 real S&P 500 tickers, ~15s, 501/503
    returned usable data (the other 2 failed with "possibly delisted" on that
    run despite being real, currently-listed constituents -- a real, expected
    per-symbol failure mode at this scale, not a reason to fail the batch).
    This is the same "one bulk call, not one per symbol" shape as
    `screening.fetch_crypto_volumes`'s Hyperliquid call.

    Returns `{symbol: {"price_change_pct": ..., "volume": ...}}` only for
    symbols yfinance actually returned usable data for. A symbol missing from
    the result (delisted, renamed, or a transient fetch failure inside the
    batch) is simply absent -- callers treat that identically to "no signal
    this week", never a crash.
    """
    symbols = list(symbols)
    if not symbols:
        return {}

    try:
        df = yf.download(
            symbols,
            period=PRICE_DATA_FETCH_PERIOD,
            interval="1d",
            group_by="ticker",
            progress=False,
            auto_adjust=True,
            threads=True,
        )
    except Exception:
        return {}

    if df is None or df.empty:
        return {}

    present = {c[0] for c in df.columns} if isinstance(df.columns, pd.MultiIndex) else set()

    out: Dict[str, Dict[str, float]] = {}
    for symbol in symbols:
        if symbol not in present:
            continue
        try:
            close = df[symbol]["Close"].dropna()
            volume = df[symbol]["Volume"].dropna()
        except KeyError:
            continue
        if len(close) < _MIN_TRADING_DAYS_FOR_MOMENTUM or volume.empty:
            continue

        prev_close = float(close.iloc[-2])
        if prev_close == 0:
            continue

        out[symbol] = {
            "price_change_pct": (float(close.iloc[-1]) / prev_close - 1.0) * 100.0,
            "volume": float(volume.iloc[-1]),
        }

    return out


# ---------------------------------------------------------------- universe


def build_equity_universe() -> Set[str]:
    """S&P 500 union Nasdaq(-100), deduplicated.

    Never empty by construction unless BOTH constituent fetches fail (including
    the key-free Wikipedia fallback for S&P 500) -- in which case the caller
    falls back to whatever symbols.yaml or config.yaml already has, per the
    brief's "never trade nothing" requirement, which lives one layer up in
    screening.py / main.py rather than being re-implemented here.
    """
    sp500 = set(fetch_sp500_constituents())
    nasdaq = set(fetch_nasdaq_constituents())
    return sp500 | nasdaq


# ------------------------------------------------------------------ scoring


def percentile_ranks(values: Dict[str, float]) -> Dict[str, float]:
    """0..1 rank of each value within `values`.

    Equal values get an equal rank (dense rank over the *unique* sorted values,
    not position in an arbitrarily-tiebroken full ordering) -- two candidates
    with identical volume or momentum must score identically on that signal.
    Symbol only ever breaks a tie in the final sort the caller does over the
    combined score, never inside this function.
    """
    if not values:
        return {}
    unique_sorted = sorted(set(values.values()))
    if len(unique_sorted) == 1:
        return {symbol: 1.0 for symbol in values}
    rank_of_value = {v: i / (len(unique_sorted) - 1) for i, v in enumerate(unique_sorted)}
    return {symbol: rank_of_value[v] for symbol, v in values.items()}


def score_equities(
    universe: Set[str], price_data: Dict[str, Dict[str, float]]
) -> List[Dict[str, Any]]:
    """Rank `universe` by liquidity (Layer 1) and momentum (Layer 2).

    `price_data` comes from `fetch_universe_price_data`, measured directly
    against `universe` rather than intersected with someone else's list of
    market-wide movers -- see the module docstring for why that distinction
    is the whole fix here. Returns every scored candidate, highest score
    first -- `screening.py` takes the top 5.

    Momentum is included for *every* symbol with real price data, not just
    ones clearing some "notable move" bar: a real, even-if-small, price change
    is still a real relative-momentum data point once percentile-ranked
    against the rest of the universe, and gating it would recreate the same
    "only the loudest movers count" mismatch this replacement exists to fix.
    Volume keeps its liquidity floor (MIN_EQUITY_VOLUME) -- that one really is
    a "not thin enough to trust" gate, not a relative ranking.

    A symbol missing from `price_data` entirely (a delisted ticker, a
    transient fetch failure inside the batch -- both observed live, see
    `fetch_universe_price_data`) still gets a row here (score 0.0, no
    signal), so `select_top_equities` can backfill toward 5 deterministically
    without ever being left short. With real universe-wide data this should
    now be the rare exception, not most of the universe.
    """
    volume_by_symbol: Dict[str, float] = {}
    momentum_by_symbol: Dict[str, float] = {}
    for symbol, data in price_data.items():
        if symbol not in universe:
            continue
        volume = data.get("volume")
        if isinstance(volume, (int, float)) and volume >= MIN_EQUITY_VOLUME:
            volume_by_symbol[symbol] = float(volume)
        pct = data.get("price_change_pct")
        if isinstance(pct, (int, float)):
            momentum_by_symbol[symbol] = abs(float(pct))

    volume_pct = percentile_ranks(volume_by_symbol)
    momentum_pct = percentile_ranks(momentum_by_symbol)

    results = []
    for symbol in sorted(universe):
        v = volume_pct.get(symbol, 0.0)
        m = momentum_pct.get(symbol, 0.0)
        results.append(
            {
                "symbol": symbol,
                "score": VOLUME_WEIGHT * v + MOMENTUM_WEIGHT * m,
                "volume": volume_by_symbol.get(symbol),
                "momentum_pct": momentum_by_symbol.get(symbol),
                "has_signal": symbol in volume_by_symbol or symbol in momentum_by_symbol,
            }
        )

    results.sort(key=lambda r: (r["score"], r["symbol"]), reverse=True)
    return results


def select_top_equities(scored: Sequence[Dict[str, Any]], count: int) -> List[str]:
    """Top `count` symbols: signal-bearing candidates first, then a deterministic
    backfill from the rest of the universe if fewer than `count` have any signal
    at all this week (e.g. a handful of delisted/failed tickers in the batch
    fetch, not a reason to hand the model fewer than the requested slate --
    see fetch_universe_price_data for why this should now be a rare exception
    rather than most of the universe).
    """
    with_signal = [r for r in scored if r["has_signal"]]
    without_signal = [r for r in scored if not r["has_signal"]]
    ordered = with_signal + without_signal
    return [r["symbol"] for r in ordered[:count]]
