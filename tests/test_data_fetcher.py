import numpy as np
import pandas as pd
import pytest

import data_fetcher


def frame(n, start=100.0, step=0.5, volume=1_000_000.0):
    closes = [start + step * i for i in range(n)]
    return pd.DataFrame({"Close": closes, "Volume": [volume] * n})


def test_default_period_covers_sma_50_for_equities():
    # yfinance has read "Nd" as both N calendar days and N bars across versions.
    # The calendar reading is the pessimistic one -- equities trade ~5/7 of
    # calendar days -- so check the default clears the SMA-50 floor even there.
    # At 60d that yields ~42 bars, which is the regression that used to crash the
    # indicator step outright.
    days = int(data_fetcher.DEFAULT_PERIOD.rstrip("d"))
    assert days >= 120
    worst_case_bars = days * 5 / 7
    assert worst_case_bars > data_fetcher.SMA_LONG + 1
    assert 60 * 5 / 7 < data_fetcher.SMA_LONG + 1  # why 60d was not enough


def test_short_frame_raises_a_message_that_names_the_shortfall():
    with pytest.raises(ValueError) as exc:
        data_fetcher.compute_indicators(frame(40))
    message = str(exc.value)
    assert "have 40" in message
    assert "need 51" in message
    assert "short by 11" in message


def test_exactly_enough_bars_is_accepted():
    result = data_fetcher.compute_indicators(frame(51))
    assert result.sma_50 > 0
    assert result.sma_20 > 0


def test_price_and_volume_change_are_measured_against_the_previous_bar():
    df = pd.DataFrame({"Close": [100.0] * 60 + [110.0], "Volume": [100.0] * 60 + [150.0]})
    result = data_fetcher.compute_indicators(df)
    assert result.price_change_pct == pytest.approx(10.0)
    assert result.volume_change_pct == pytest.approx(50.0)


def test_rsi_is_100_when_every_change_is_a_gain():
    assert data_fetcher._wilder_rsi(pd.Series(frame(60)["Close"])) == pytest.approx(100.0)


def test_rsi_is_0_when_every_change_is_a_loss():
    closes = pd.Series([200.0 - i for i in range(60)])
    assert data_fetcher._wilder_rsi(closes) == pytest.approx(0.0)


def test_rsi_of_a_flat_series_is_neutral():
    assert data_fetcher._wilder_rsi(pd.Series([100.0] * 60)) == pytest.approx(50.0)


def test_rsi_uses_wilder_smoothing_not_a_simple_mean():
    # Wilder seeds on the first `period` changes then smooths forward, so a late
    # shock is damped rather than averaged in equally. Hand-computed reference.
    closes = pd.Series([100.0 + i for i in range(15)] + [110.0])
    expected = _reference_wilder(closes, 14)
    assert data_fetcher._wilder_rsi(closes, 14) == pytest.approx(expected)


def _reference_wilder(closes, period):
    delta = closes.diff().dropna().tolist()
    gains = [max(d, 0.0) for d in delta]
    losses = [max(-d, 0.0) for d in delta]
    avg_g = sum(gains[:period]) / period
    avg_l = sum(losses[:period]) / period
    for i in range(period, len(delta)):
        avg_g = (avg_g * (period - 1) + gains[i]) / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
    if avg_l == 0:
        return 100.0 if avg_g > 0 else 50.0
    return 100.0 - 100.0 / (1.0 + avg_g / avg_l)


def test_rsi_needs_enough_changes():
    with pytest.raises(ValueError, match="needs at least 15 bars"):
        data_fetcher._wilder_rsi(pd.Series([100.0] * 10))


def test_fetch_ohlcv_drops_the_trailing_nan_close(monkeypatch):
    # yfinance routinely returns a trailing row for the current, incomplete
    # session whose Close is NaN. Left in, it poisons every last-bar read.
    raw = pd.DataFrame({"Close": [100.0, 101.0, np.nan], "Volume": [1.0, 2.0, 3.0]})

    class FakeTicker:
        def __init__(self, symbol):
            pass

        def history(self, period, interval):
            return raw.copy()

    monkeypatch.setattr(data_fetcher.yf, "Ticker", FakeTicker)
    out = data_fetcher.fetch_ohlcv("AAPL")
    assert len(out) == 2
    assert data_fetcher.latest_price(out) == 101.0


def test_fetch_ohlcv_raises_when_everything_is_nan(monkeypatch):
    class FakeTicker:
        def __init__(self, symbol):
            pass

        def history(self, period, interval):
            return pd.DataFrame({"Close": [np.nan, np.nan], "Volume": [1.0, 2.0]})

    monkeypatch.setattr(data_fetcher.yf, "Ticker", FakeTicker)
    with pytest.raises(ValueError, match="NaN Close"):
        data_fetcher.fetch_ohlcv("AAPL")


def test_fetch_ohlcv_raises_on_an_empty_response(monkeypatch):
    class FakeTicker:
        def __init__(self, symbol):
            pass

        def history(self, period, interval):
            return pd.DataFrame()

    monkeypatch.setattr(data_fetcher.yf, "Ticker", FakeTicker)
    with pytest.raises(ValueError, match="no bars"):
        data_fetcher.fetch_ohlcv("NOPE")


@pytest.mark.parametrize(
    "symbol,expected",
    [("BTC-USD", "BTC"), ("ETH-USDT", "ETH"), ("btc-usd", "btc"), ("AAPL", "AAPL")],
)
def test_headline_query_strips_the_crypto_suffix(symbol, expected):
    assert data_fetcher._CRYPTO_SUFFIX_RE.sub("", symbol) == expected


def test_headlines_return_empty_instead_of_raising(monkeypatch):
    # A news outage is not a reason to skip a trading decision.
    def boom(*a, **kw):
        raise RuntimeError("network down")

    monkeypatch.setattr(data_fetcher.requests, "get", boom)
    assert data_fetcher.fetch_headlines("AAPL") == []


def test_headlines_request_sends_a_browser_user_agent(monkeypatch):
    # Yahoo answers 429 to a bare requests User-Agent, so without this the feed
    # returns nothing and the model silently gets no headlines, forever.
    seen = {}

    class FakeResponse:
        content = b"<rss><channel></channel></rss>"

        def raise_for_status(self):
            pass

    def fake_get(url, timeout=None, headers=None):
        seen["headers"] = headers or {}
        return FakeResponse()

    monkeypatch.setattr(data_fetcher.requests, "get", fake_get)
    data_fetcher.fetch_headlines("AAPL")
    assert "Mozilla" in seen["headers"].get("User-Agent", "")


def test_headlines_are_parsed_and_limited(monkeypatch):
    xml = b"""<?xml version="1.0"?><rss><channel>
      <item><title>One</title></item>
      <item><title>Two</title></item>
      <item><title>Three</title></item>
    </channel></rss>"""

    class FakeResponse:
        content = xml

        def raise_for_status(self):
            pass

    monkeypatch.setattr(data_fetcher.requests, "get", lambda *a, **kw: FakeResponse())
    assert data_fetcher.fetch_headlines("AAPL", limit=2) == ["One", "Two"]
