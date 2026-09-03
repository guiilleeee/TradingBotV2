"""FMP integration, Wikipedia fallback, yfinance batch pricing, and equity
scoring -- all offline.

No test here touches the network. FMP calls are mocked with response shapes
documented on FMP's own doc pages (verified via search, since every FMP
endpoint -- including a request that would just prove a path exists -- requires
a real key, and none is available in this environment); the Wikipedia fallback
is exercised against a small local HTML fixture shaped like the real page's
constituents table, not the live site. `fetch_universe_price_data`'s shape
(MultiIndex `(symbol, field)` columns from `group_by="ticker"`, an all-NaN
column for a delisted-shaped symbol, an empty-list call raising inside
pandas) was verified live against yfinance 1.7.0 and the real S&P 500 list
before writing these fakes -- see equity_universe.py's docstring.
"""

import pandas as pd
import pytest

import equity_universe


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("unexpected network call")

    monkeypatch.setattr(equity_universe.requests, "get", boom)


@pytest.fixture(autouse=True)
def fmp_key(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "test-key")


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    # The real code paces calls with a small delay; tests should not pay for it.
    monkeypatch.setattr(equity_universe.time, "sleep", lambda *_: None)


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise equity_universe.requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload

    @property
    def text(self):
        return self._payload


def mock_get(monkeypatch, by_path):
    """route requests.get(url, ...) to a canned response keyed by URL suffix."""

    def fake_get(url, params=None, timeout=None):
        for suffix, payload in by_path.items():
            if url.endswith(suffix):
                return FakeResponse(payload)
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(equity_universe.requests, "get", fake_get)


# ------------------------------------------------------------------- api key


def test_missing_api_key_raises_a_named_error(monkeypatch):
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    with pytest.raises(equity_universe.FMPError, match="FMP_API_KEY"):
        equity_universe._get("/stable/sp-500")


# ---------------------------------------------------------------- S&P 500


def test_sp500_from_stable_endpoint(monkeypatch):
    mock_get(monkeypatch, {
        "/stable/sp-500": [
            {"symbol": "AAPL", "name": "Apple Inc"},
            {"symbol": "MSFT", "name": "Microsoft Corp"},
        ],
    })
    assert equity_universe.fetch_sp500_constituents() == ["AAPL", "MSFT"]


def test_sp500_falls_back_to_legacy_path_when_stable_is_empty(monkeypatch):
    mock_get(monkeypatch, {
        "/stable/sp-500": [],
        "/api/v3/sp500_constituent": [{"symbol": "AAPL"}],
    })
    assert equity_universe.fetch_sp500_constituents() == ["AAPL"]


def test_sp500_falls_back_to_wikipedia_when_fmp_entirely_fails(monkeypatch):
    def fake_get(url, params=None, timeout=None, headers=None):
        if "wikipedia" in url:
            return FakeResponse(_SP500_FIXTURE_HTML)
        raise equity_universe.requests.HTTPError("FMP down")

    monkeypatch.setattr(equity_universe.requests, "get", fake_get)

    symbols = equity_universe.fetch_sp500_constituents()
    assert symbols == ["AAPL", "MSFT", "BRK-B"]  # dot normalised to dash


def test_sp500_returns_empty_when_everything_fails(monkeypatch):
    def boom(*a, **kw):
        raise equity_universe.requests.ConnectionError("down")

    monkeypatch.setattr(equity_universe.requests, "get", boom)
    assert equity_universe.fetch_sp500_constituents() == []


_SP500_FIXTURE_HTML = """
<html><body>
<table id="constituents" class="wikitable sortable">
<tr><th>Symbol</th><th>Security</th><th>GICS Sector</th></tr>
<tr><td>AAPL</td><td>Apple Inc.</td><td>Information Technology</td></tr>
<tr><td>MSFT</td><td>Microsoft</td><td>Information Technology</td></tr>
<tr><td>BRK.B</td><td>Berkshire Hathaway</td><td>Financials</td></tr>
</table>
</body></html>
"""


# ------------------------------------------------------------------ Nasdaq


def test_nasdaq_accepts_a_plausibly_sized_result(monkeypatch):
    rows = [{"symbol": f"SYM{i}"} for i in range(100)]
    mock_get(monkeypatch, {"/stable/nasdaq-constituent": rows})
    result = equity_universe.fetch_nasdaq_constituents()
    assert len(result) == 100


def test_nasdaq_rejects_a_result_that_looks_like_the_whole_exchange(monkeypatch):
    # If FMP's "constituent" endpoint turns out to mean "every Nasdaq-listed
    # stock" (thousands), this must not silently become the trading universe.
    rows = [{"symbol": f"SYM{i}"} for i in range(3000)]
    mock_get(monkeypatch, {"/stable/nasdaq-constituent": rows})
    assert equity_universe.fetch_nasdaq_constituents() == []


def test_nasdaq_at_the_plausibility_boundary():
    small = [{"symbol": f"S{i}"} for i in range(equity_universe.NASDAQ_100_MAX_PLAUSIBLE_SIZE)]
    large = [{"symbol": f"S{i}"} for i in range(equity_universe.NASDAQ_100_MAX_PLAUSIBLE_SIZE + 1)]
    assert len(equity_universe._extract_symbols(small)) == equity_universe.NASDAQ_100_MAX_PLAUSIBLE_SIZE
    assert len(equity_universe._extract_symbols(large)) == equity_universe.NASDAQ_100_MAX_PLAUSIBLE_SIZE + 1


def test_nasdaq_has_no_key_free_fallback(monkeypatch):
    # Unlike S&P 500, there is no reliably-scrapable Wikipedia table for
    # Nasdaq-100 (its member list has no ticker column) -- a total FMP failure
    # here means an empty Nasdaq contribution, not a crash.
    def boom(*a, **kw):
        raise equity_universe.requests.ConnectionError("down")

    monkeypatch.setattr(equity_universe.requests, "get", boom)
    assert equity_universe.fetch_nasdaq_constituents() == []


# --------------------------------------------------------------- universe


def test_universe_is_the_union_of_both_indices(monkeypatch):
    mock_get(monkeypatch, {
        "/stable/sp-500": [{"symbol": "AAPL"}, {"symbol": "MSFT"}],
        "/stable/nasdaq-constituent": [{"symbol": "MSFT"}, {"symbol": "NVDA"}],
    })
    assert equity_universe.build_equity_universe() == {"AAPL", "MSFT", "NVDA"}


# ----------------------------------------------------------- universe pricing


def fake_price_frame(data):
    """{symbol: {"Close": [...], "Volume": [...]}} -> a MultiIndex DataFrame
    shaped exactly like `yf.download(..., group_by="ticker")`'s real return
    value (verified live -- see equity_universe.py's docstring): top-level
    columns are symbols, second level is OHLCV field.
    """
    frames = {symbol: pd.DataFrame(cols) for symbol, cols in data.items()}
    return pd.concat(frames, axis=1)


def test_price_data_computes_change_from_the_last_two_closes_and_latest_volume(monkeypatch):
    frame = fake_price_frame({
        "AAPL": {"Close": [100.0, 110.0], "Volume": [1_000_000.0, 2_000_000.0]},
    })
    monkeypatch.setattr(equity_universe.yf, "download", lambda *a, **kw: frame)

    data = equity_universe.fetch_universe_price_data(["AAPL"])

    assert data["AAPL"]["price_change_pct"] == pytest.approx(10.0)
    assert data["AAPL"]["volume"] == pytest.approx(2_000_000.0)


def test_price_data_never_calls_download_for_an_empty_symbol_list(monkeypatch):
    # yf.download([]) raises inside pandas.concat rather than returning an
    # empty frame -- confirmed live -- so an empty list must short-circuit
    # before ever reaching yf.download at all.
    def explode(*a, **kw):
        raise AssertionError("must not call yf.download with no symbols")

    monkeypatch.setattr(equity_universe.yf, "download", explode)
    assert equity_universe.fetch_universe_price_data([]) == {}


def test_price_data_drops_a_symbol_with_all_nan_data(monkeypatch):
    # The real shape of a delisted/bad ticker mixed into a batch download:
    # its columns exist but are entirely NaN (confirmed live).
    import numpy as np

    frame = fake_price_frame({
        "AAPL": {"Close": [100.0, 110.0], "Volume": [1_000_000.0, 2_000_000.0]},
        "DEADTICKER": {"Close": [np.nan, np.nan], "Volume": [np.nan, np.nan]},
    })
    monkeypatch.setattr(equity_universe.yf, "download", lambda *a, **kw: frame)

    data = equity_universe.fetch_universe_price_data(["AAPL", "DEADTICKER"])

    assert "AAPL" in data
    assert "DEADTICKER" not in data


def test_price_data_drops_a_symbol_missing_from_the_result_entirely(monkeypatch):
    frame = fake_price_frame({"AAPL": {"Close": [100.0, 110.0], "Volume": [1e6, 2e6]}})
    monkeypatch.setattr(equity_universe.yf, "download", lambda *a, **kw: frame)

    data = equity_universe.fetch_universe_price_data(["AAPL", "NOTINRESULT"])

    assert "NOTINRESULT" not in data


def test_price_data_needs_at_least_two_trading_days(monkeypatch):
    frame = fake_price_frame({"AAPL": {"Close": [100.0], "Volume": [1e6]}})
    monkeypatch.setattr(equity_universe.yf, "download", lambda *a, **kw: frame)
    assert equity_universe.fetch_universe_price_data(["AAPL"]) == {}


def test_price_data_degrades_to_empty_on_a_download_exception(monkeypatch):
    def boom(*a, **kw):
        raise ConnectionError("yahoo is down")

    monkeypatch.setattr(equity_universe.yf, "download", boom)
    assert equity_universe.fetch_universe_price_data(["AAPL"]) == {}


def test_price_data_returns_empty_on_a_completely_empty_frame(monkeypatch):
    monkeypatch.setattr(equity_universe.yf, "download", lambda *a, **kw: pd.DataFrame())
    assert equity_universe.fetch_universe_price_data(["AAPL"]) == {}


def test_price_data_is_a_single_batched_call_not_one_per_symbol(monkeypatch):
    calls = []

    def fake_download(symbols, **kw):
        calls.append(list(symbols))
        return fake_price_frame({s: {"Close": [100.0, 101.0], "Volume": [1e6, 1e6]} for s in symbols})

    monkeypatch.setattr(equity_universe.yf, "download", fake_download)
    symbols = [f"SYM{i}" for i in range(50)]
    equity_universe.fetch_universe_price_data(symbols)

    assert len(calls) == 1
    assert calls[0] == symbols


def test_fmp_error_message_body_is_treated_as_a_failure(monkeypatch):
    # FMP returns HTTP 200 with an {"Error Message": ...} body for some failure
    # modes (bad params, plan-gated endpoints) instead of a 4xx status.
    mock_get(monkeypatch, {"/stable/sp-500": {"Error Message": "Limit Reach"}})
    assert equity_universe.fetch_sp500_constituents() == []


# ----------------------------------------------------------- percentile ranks


def test_percentile_ranks_of_an_empty_input():
    assert equity_universe.percentile_ranks({}) == {}


def test_percentile_ranks_span_zero_to_one():
    ranks = equity_universe.percentile_ranks({"A": 10.0, "B": 30.0, "C": 20.0})
    assert ranks["A"] == 0.0
    assert ranks["C"] == 0.5
    assert ranks["B"] == 1.0


def test_percentile_rank_of_a_single_value_is_one():
    assert equity_universe.percentile_ranks({"A": 5.0}) == {"A": 1.0}


# --------------------------------------------------------------------- scoring


def price_data(**per_symbol):
    """{"AAPL": (change_pct, volume), ...} -> the price_data shape score_equities takes."""
    return {
        symbol: {"price_change_pct": change, "volume": volume}
        for symbol, (change, volume) in per_symbol.items()
    }


def test_volume_below_the_floor_is_not_counted_towards_volume_signal():
    universe = {"AAPL"}
    data = price_data(AAPL=(0.0, 1.0))  # real momentum (flat), but a tiny volume
    scored = equity_universe.score_equities(universe, data)
    assert scored[0]["volume"] is None  # excluded by the floor
    # Still has_signal, though: a real (if zero) price change is itself signal
    # now -- this is the core behavioural change over the old FMP-list gate.
    assert scored[0]["has_signal"] is True


def test_a_symbol_outside_the_universe_is_ignored():
    universe = {"AAPL"}
    data = price_data(TSLA=(5.0, 1e8))
    scored = equity_universe.score_equities(universe, data)
    assert len(scored) == 1
    assert scored[0]["symbol"] == "AAPL"
    assert scored[0]["has_signal"] is False


def test_a_symbol_with_no_price_data_at_all_has_no_signal():
    # The one case that should now be rare (a delisted/failed ticker in the
    # batch, not "wasn't on someone else's movers list").
    universe = {"AAPL", "MSFT"}
    data = price_data(AAPL=(1.2, 5e7))
    scored = equity_universe.score_equities(universe, data)
    by_symbol = {r["symbol"]: r for r in scored}
    assert by_symbol["AAPL"]["has_signal"] is True
    assert by_symbol["MSFT"]["has_signal"] is False


def test_volume_and_momentum_both_contribute():
    universe = {"AAPL", "MSFT", "GOOG"}
    data = price_data(AAPL=(0.5, 5e7), MSFT=(0.5, 1e7), GOOG=(9.0, 5e7))
    scored = equity_universe.score_equities(universe, data)
    by_symbol = {r["symbol"]: r for r in scored}
    assert by_symbol["AAPL"]["has_signal"] is True
    assert by_symbol["MSFT"]["has_signal"] is True
    assert by_symbol["GOOG"]["has_signal"] is True
    # AAPL and MSFT have identical (small) momentum; AAPL's higher volume rank
    # must be what puts it ahead, proving the volume half of the blend counts.
    assert by_symbol["AAPL"]["score"] > by_symbol["MSFT"]["score"]


def test_a_loser_contributes_momentum_by_magnitude_not_sign():
    universe = {"AAPL", "MSFT"}
    data = price_data(AAPL=(5.0, 1e7), MSFT=(-5.0, 1e7))
    scored = equity_universe.score_equities(universe, data)
    by_symbol = {r["symbol"]: r for r in scored}
    assert by_symbol["AAPL"]["score"] == by_symbol["MSFT"]["score"]


def test_every_symbol_with_real_data_carries_momentum_even_without_a_big_move():
    # The core fix: momentum is no longer gated on "was this on a whole-market
    # gainers/losers list" -- any real price change, however small, is a real
    # relative-momentum data point once percentile-ranked against the universe.
    universe = {"AAPL", "MSFT", "GOOG"}
    data = price_data(AAPL=(0.01, 1e7), MSFT=(0.02, 1e7), GOOG=(0.03, 1e7))
    scored = equity_universe.score_equities(universe, data)
    assert all(r["has_signal"] for r in scored)
    assert all(r["momentum_pct"] is not None for r in scored)


def test_scoring_is_deterministic_across_repeated_calls():
    universe = {"AAPL", "MSFT", "GOOG", "TSLA"}
    data = price_data(AAPL=(1.0, 5e7), MSFT=(-2.0, 3e7), GOOG=(0.5, 2e7), TSLA=(4.0, 9e7))
    first = equity_universe.score_equities(universe, data)
    second = equity_universe.score_equities(universe, data)
    assert [r["symbol"] for r in first] == [r["symbol"] for r in second]


# --------------------------------------------------------------- selection


def test_select_top_prefers_signal_bearing_candidates():
    scored = [
        {"symbol": "NOSIGNAL", "score": 0.0, "has_signal": False},
        {"symbol": "SIGNAL", "score": 0.3, "has_signal": True},
    ]
    assert equity_universe.select_top_equities(scored, 1) == ["SIGNAL"]


def test_select_top_backfills_deterministically_when_short_on_signal():
    # Only one symbol has real signal this week; the rest must still be filled,
    # not left short of the requested count.
    scored = [
        {"symbol": "A", "score": 0.0, "has_signal": False},
        {"symbol": "B", "score": 0.9, "has_signal": True},
        {"symbol": "C", "score": 0.0, "has_signal": False},
    ]
    result = equity_universe.select_top_equities(scored, 3)
    assert len(result) == 3
    assert result[0] == "B"
    assert set(result) == {"A", "B", "C"}


def test_select_top_never_exceeds_the_pool_size():
    scored = [{"symbol": "A", "score": 0.0, "has_signal": False}]
    assert equity_universe.select_top_equities(scored, 5) == ["A"]
