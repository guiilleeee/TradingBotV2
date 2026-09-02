"""Market data and derived indicators.

Everything here is read-only and failure-tolerant in one specific direction:
a headline fetch may fail silently, price data may not.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import List, Optional

import pandas as pd
import requests
import yfinance as yf

from models import TechnicalIndicators

# 120, not 60. yfinance has interpreted the "Nd" period both ways across versions
# -- as N calendar days (where equities trade only ~5/7 of them, so 60d yields
# roughly 42 bars, fewer than the 50 SMA-50 needs, which is what crashed the
# indicator step) and, as of 1.x, as N returned bars. 120 is safe under either
# reading: ~85 bars on the calendar interpretation, 120 on the bar interpretation,
# both comfortably clear of the 51-bar floor. The compute_indicators guard below
# is what actually enforces it; this is just a sensible default.
#
# The lookback is deliberately identical for equities and crypto. Crypto trades
# 7 days a week and was never affected, but a longer window costs nothing and one
# code path beats a branch that can only ever be wrong for one side.
DEFAULT_PERIOD = "120d"

RSI_PERIOD = 14
SMA_SHORT = 20
SMA_LONG = 50

_YAHOO_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={q}&region=US&lang=en-US"
# Yahoo answers 429 Too Many Requests to a bare requests/urllib User-Agent, on the
# very first call. Without this header the feed never returns anything and the
# try/except below quietly hands the model an empty headline list forever.
_HEADLINE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    )
}
_CRYPTO_SUFFIX_RE = re.compile(r"-(USDT|USD)$", re.IGNORECASE)


def fetch_ohlcv(
    symbol: str,
    period: str = DEFAULT_PERIOD,
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch OHLCV bars for `symbol`.

    The dropna on Close is not optional: yfinance routinely returns a trailing row
    for the current, still-incomplete session whose Close is NaN. Left in place it
    poisons the last-bar reads below.
    """
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df is None or df.empty:
        raise ValueError(f"{symbol}: yfinance returned no bars for period={period} interval={interval}")

    if "Close" not in df.columns:
        raise ValueError(f"{symbol}: yfinance response has no Close column (got {list(df.columns)})")

    df = df.dropna(subset=["Close"])

    if df.empty:
        raise ValueError(f"{symbol}: every returned bar had a NaN Close")

    return df


def _wilder_rsi(close: pd.Series, period: int = RSI_PERIOD) -> float:
    """RSI with Wilder's smoothing, seeded from the first `period` changes.

    Written out rather than pulled from TA-Lib -- one function is not worth a
    native dependency that has to be built on every CI runner.
    """
    delta = close.diff().dropna()
    if len(delta) < period:
        raise ValueError(
            f"RSI-{period} needs at least {period + 1} bars, got {len(delta) + 1}"
        )

    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)

    # Seed: simple mean of the first `period` gains/losses.
    avg_gain = float(gains.iloc[:period].mean())
    avg_loss = float(losses.iloc[:period].mean())

    # Then smooth forward across the remainder of the series.
    for i in range(period, len(delta)):
        avg_gain = (avg_gain * (period - 1) + float(gains.iloc[i])) / period
        avg_loss = (avg_loss * (period - 1) + float(losses.iloc[i])) / period

    if avg_loss == 0.0:
        return 100.0 if avg_gain > 0.0 else 50.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_indicators(
    df: pd.DataFrame,
    rsi_period: int = RSI_PERIOD,
    sma_short: int = SMA_SHORT,
    sma_long: int = SMA_LONG,
) -> TechnicalIndicators:
    """Derive the indicator bundle from OHLCV bars.

    The length guard is explicit and up front. Without it a short frame produces
    NaN SMAs that surface much later as an opaque Pydantic validation error on
    `sma_50` -- which tells you nothing about the real cause.
    """
    required = sma_long + 1
    if len(df) < required:
        raise ValueError(
            f"Not enough bars to compute indicators: have {len(df)}, need {required} "
            f"(SMA-{sma_long} plus one prior bar for the change columns); "
            f"short by {required - len(df)}. Widen the fetch period."
        )

    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series([0.0] * len(df))

    sma_s = float(close.rolling(sma_short).mean().iloc[-1])
    sma_l = float(close.rolling(sma_long).mean().iloc[-1])

    last_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])
    price_change_pct = ((last_close / prev_close) - 1.0) * 100.0 if prev_close else 0.0

    last_vol = float(volume.iloc[-1])
    prev_vol = float(volume.iloc[-2])
    volume_change_pct = ((last_vol / prev_vol) - 1.0) * 100.0 if prev_vol else 0.0

    return TechnicalIndicators(
        rsi_14=_wilder_rsi(close, rsi_period),
        sma_20=sma_s,
        sma_50=sma_l,
        price_change_pct=price_change_pct,
        volume_change_pct=volume_change_pct,
    )


def latest_price(df: pd.DataFrame) -> float:
    """Close of the last complete bar."""
    return float(df["Close"].astype(float).iloc[-1])


def fetch_headlines(symbol: str, limit: int = 5, timeout: float = 10.0) -> List[str]:
    """Recent Yahoo Finance headlines for `symbol`.

    Never raises. A news outage is not a reason to skip a trading decision, so the
    caller gets an empty list and the model is told there are no headlines.
    """
    try:
        query = _CRYPTO_SUFFIX_RE.sub("", symbol)
        resp = requests.get(
            _YAHOO_RSS.format(q=query), timeout=timeout, headers=_HEADLINE_HEADERS
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        titles: List[str] = []
        for item in root.iter("item"):
            title: Optional[ET.Element] = item.find("title")
            if title is not None and title.text:
                titles.append(title.text.strip())
            if len(titles) >= limit:
                break
        return titles
    except Exception:
        return []
