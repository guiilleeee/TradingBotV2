"""FMP integration, Wikipedia fallback, and equity scoring -- all offline.

No test here touches the network. FMP calls are mocked with response shapes
documented on FMP's own doc pages (verified via search, since every FMP
endpoint -- including a request that would just prove a path exists -- requires
a real key, and none is available in this environment); the Wikipedia fallback
is exercised against a small local HTML fixture shaped like the real page's
constituents table, not the live site.
"""

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


# ----------------------------------------------------------------- movers


def test_market_movers_fetches_all_three_lists(monkeypatch):
    mock_get(monkeypatch, {
        "/stable/most-actives": [{"symbol": "AAPL", "volume": 5e7}],
        "/stable/biggest-gainers": [{"symbol": "NVDA", "changesPercentage": 8.2}],
        "/stable/biggest-losers": [{"symbol": "INTC", "changesPercentage": -6.1}],
    })
    movers = equity_universe.fetch_market_movers()
    assert movers["most_actives"][0]["symbol"] == "AAPL"
    assert movers["gainers"][0]["symbol"] == "NVDA"
    assert movers["losers"][0]["symbol"] == "INTC"


def test_one_mover_endpoint_failing_does_not_cost_the_others(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        if url.endswith("/stable/biggest-gainers"):
            raise equity_universe.requests.HTTPError("gainers down")
        if url.endswith("/stable/most-actives"):
            return FakeResponse([{"symbol": "AAPL", "volume": 1e7}])
        return FakeResponse([])

    monkeypatch.setattr(equity_universe.requests, "get", fake_get)
    movers = equity_universe.fetch_market_movers()
    assert movers["gainers"] == []
    assert movers["most_actives"][0]["symbol"] == "AAPL"


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


def test_volume_below_the_floor_is_not_counted_as_signal():
    universe = {"AAPL"}
    movers = {"most_actives": [{"symbol": "AAPL", "volume": 1.0}], "gainers": [], "losers": []}
    scored = equity_universe.score_equities(universe, movers)
    assert scored[0]["has_signal"] is False


def test_a_symbol_outside_the_universe_is_ignored():
    universe = {"AAPL"}
    movers = {"most_actives": [{"symbol": "TSLA", "volume": 1e8}], "gainers": [], "losers": []}
    scored = equity_universe.score_equities(universe, movers)
    assert len(scored) == 1
    assert scored[0]["symbol"] == "AAPL"
    assert scored[0]["has_signal"] is False


def test_volume_and_momentum_both_contribute():
    universe = {"AAPL", "MSFT", "GOOG"}
    movers = {
        "most_actives": [
            {"symbol": "AAPL", "volume": 5e7},
            {"symbol": "MSFT", "volume": 1e7},
        ],
        "gainers": [{"symbol": "GOOG", "changesPercentage": 9.0}],
        "losers": [],
    }
    scored = equity_universe.score_equities(universe, movers)
    by_symbol = {r["symbol"]: r for r in scored}
    assert by_symbol["AAPL"]["has_signal"] is True
    assert by_symbol["MSFT"]["has_signal"] is True
    assert by_symbol["GOOG"]["has_signal"] is True
    # AAPL has the top volume rank and no momentum; still scores above MSFT
    # (lower volume, no momentum) because of the 0.6 volume weight.
    assert by_symbol["AAPL"]["score"] > by_symbol["MSFT"]["score"]


def test_a_loser_contributes_momentum_by_magnitude_not_sign():
    universe = {"AAPL", "MSFT"}
    movers = {
        "most_actives": [],
        "gainers": [{"symbol": "AAPL", "changesPercentage": 5.0}],
        "losers": [{"symbol": "MSFT", "changesPercentage": -5.0}],
    }
    scored = equity_universe.score_equities(universe, movers)
    by_symbol = {r["symbol"]: r for r in scored}
    assert by_symbol["AAPL"]["score"] == by_symbol["MSFT"]["score"]


def test_scoring_is_deterministic_across_repeated_calls():
    universe = {"AAPL", "MSFT", "GOOG", "TSLA"}
    movers = {"most_actives": [], "gainers": [], "losers": []}
    first = equity_universe.score_equities(universe, movers)
    second = equity_universe.score_equities(universe, movers)
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
