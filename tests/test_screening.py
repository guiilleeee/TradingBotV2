"""Crypto screening, symbol-file writing, and orchestration -- all offline.

The load-bearing property here is reuse: `build_crypto_universe` must call
execution.resolve_spot_market (the same function execution.py uses before ever
placing a real order) rather than re-deriving "is this spot" independently --
tested by observation (a fake exchange whose perp markets are refused) rather
than by reading the source.
"""

import yaml
import pytest

import equity_universe
import execution
import market_intel
import screening


# ----------------------------------------------------------- fake exchange


def spot_market(hl_symbol, base, coin_id, quote="USDC"):
    return {
        "id": coin_id, "symbol": hl_symbol, "base": base, "quote": quote,
        "type": "spot", "spot": True, "swap": False, "contract": False, "active": True,
    }


def perp_market(hl_symbol, base, coin_id="BTC"):
    return {
        "id": coin_id, "symbol": hl_symbol, "base": base, "quote": "USDC", "settle": "USDC",
        "type": "swap", "spot": False, "swap": True, "contract": True, "active": True,
    }


class FakeMultiMarketExchange:
    """A ccxt-shaped fake carrying several markets, for universe-building tests."""

    def __init__(self, markets):
        self.markets = markets  # hl_symbol -> market dict, pre-"loaded"

    def load_markets(self):
        return self.markets

    def market(self, hl_symbol):
        return self.markets[hl_symbol]


def default_markets():
    return {
        "BTC/USDC": spot_market("BTC/USDC", "BTC", "@142"),
        "ETH/USDC": spot_market("ETH/USDC", "ETH", "@151"),
        "SOL/USDC": spot_market("SOL/USDC", "SOL", "@156"),
        "HYPE/USDC": spot_market("HYPE/USDC", "HYPE", "@107"),
        "PURR/USDC": spot_market("PURR/USDC", "PURR", "PURR/USDC"),
        # A perpetual sitting right alongside the spot markets, exactly as the
        # real venue does -- this must never survive build_crypto_universe.
        "BTC/USDC:USDC": perp_market("BTC/USDC:USDC", "BTC"),
    }


# ---------------------------------------------------------- universe reuse


def test_universe_only_contains_confirmed_spot_markets():
    exchange = FakeMultiMarketExchange(default_markets())
    universe = screening.build_crypto_universe(exchange)
    assert "BTC/USDC:USDC" not in universe
    assert set(universe) == {"BTC/USDC", "ETH/USDC", "SOL/USDC", "HYPE/USDC", "PURR/USDC"}


def test_universe_building_calls_executions_own_spot_check(monkeypatch):
    """Not just 'the perp is excluded' -- that the SAME function did the excluding."""
    calls = []
    real = execution.resolve_spot_market

    def spy(exchange, hl_symbol):
        calls.append(hl_symbol)
        return real(exchange, hl_symbol)

    monkeypatch.setattr(execution, "resolve_spot_market", spy)
    exchange = FakeMultiMarketExchange(default_markets())
    screening.build_crypto_universe(exchange)
    assert set(calls) == set(default_markets().keys())


def test_a_market_missing_every_spot_flag_is_excluded():
    markets = {"WEIRD/USDC": {"id": "@1", "symbol": "WEIRD/USDC", "base": "WEIRD", "quote": "USDC"}}
    exchange = FakeMultiMarketExchange(markets)
    assert screening.build_crypto_universe(exchange) == {}


# --------------------------------------------------------------- volume


def ctx(coin, day_ntl_vlm):
    return {"coin": coin, "dayNtlVlm": str(day_ntl_vlm)}


def test_volume_is_looked_up_by_market_id_not_by_symbol_name(monkeypatch):
    # The raw Hyperliquid universe mostly uses opaque "@N" ids as its `coin`
    # field, not human-readable pairs -- volume lookup must go through
    # market['id'], never string-match against the ccxt "BTC/USDC" symbol.
    exchange = FakeMultiMarketExchange(default_markets())
    exchange.load_markets()
    payload = [
        {"universe": []},
        [ctx("@142", 46_000_000), ctx("@151", 12_000_000), ctx("PURR/USDC", 1_600_000)],
    ]
    monkeypatch.setattr(market_intel, "post_info", lambda body: payload)

    volumes = screening.fetch_crypto_volumes(exchange, ["BTC/USDC", "ETH/USDC", "PURR/USDC", "SOL/USDC"])
    assert volumes["BTC/USDC"] == 46_000_000
    assert volumes["ETH/USDC"] == 12_000_000
    assert volumes["PURR/USDC"] == 1_600_000
    assert "SOL/USDC" not in volumes  # no matching ctx -- degrades, does not crash


def test_volume_fetch_requests_the_bulk_endpoint_once(monkeypatch):
    calls = []

    def fake_post_info(body):
        calls.append(body)
        return [{"universe": []}, []]

    monkeypatch.setattr(market_intel, "post_info", fake_post_info)
    exchange = FakeMultiMarketExchange(default_markets())
    exchange.load_markets()
    screening.fetch_crypto_volumes(exchange, list(exchange.markets.keys()))
    assert len(calls) == 1
    assert calls[0] == {"type": "spotMetaAndAssetCtxs"}


# --------------------------------------------------------------- positioning


def test_positioning_reuses_market_intels_own_pipeline(monkeypatch):
    calls = {"leaderboard": 0, "top_wallets": 0, "aggregate": 0}
    monkeypatch.setattr(market_intel, "fetch_leaderboard",
                        lambda: (calls.__setitem__("leaderboard", calls["leaderboard"] + 1) or [{"x": 1}]))
    monkeypatch.setattr(market_intel, "top_wallets",
                        lambda rows, limit: (calls.__setitem__("top_wallets", calls["top_wallets"] + 1) or [("0xa", 1.0)]))
    monkeypatch.setattr(market_intel, "aggregate_positioning",
                        lambda wallets: (calls.__setitem__("aggregate", calls["aggregate"] + 1) or {"BTC": {"long": 1.0, "short": 0.0}}))

    result = screening.fetch_crypto_positioning(limit=50)
    assert result == {"BTC": {"long": 1.0, "short": 0.0}}
    assert calls == {"leaderboard": 1, "top_wallets": 1, "aggregate": 1}


def test_positioning_degrades_to_empty_on_failure(monkeypatch):
    def boom():
        raise RuntimeError("leaderboard down")

    monkeypatch.setattr(market_intel, "fetch_leaderboard", boom)
    assert screening.fetch_crypto_positioning() == {}


def test_positioning_skips_the_wallet_pass_when_the_leaderboard_is_empty(monkeypatch):
    monkeypatch.setattr(market_intel, "fetch_leaderboard", lambda: [])

    def explode(*a, **kw):
        raise AssertionError("must not aggregate with no wallets")

    monkeypatch.setattr(market_intel, "top_wallets", lambda rows, limit: [])
    monkeypatch.setattr(market_intel, "aggregate_positioning", explode)
    assert screening.fetch_crypto_positioning() == {}


# ------------------------------------------------------------------ scoring


def test_crypto_scoring_excludes_markets_below_the_volume_floor():
    universe = {"BTC/USDC": spot_market("BTC/USDC", "BTC", "@1")}
    volumes = {"BTC/USDC": screening.MIN_CRYPTO_VOLUME_USD - 1}
    scored = screening.score_crypto(universe, volumes, {})
    assert scored == []


# -------------------------------------------------------- stablecoin gate


def test_stablecoin_base_never_appears_in_scored_output_regardless_of_volume():
    """A USDT/USD-shaped market with very high volume must be excluded before scoring.

    This is the canonical acceptance test for the stablecoin gate: high volume
    would otherwise rank the pair at the top of the list, but the gate must
    drop it before any ranking occurs, just like the volume floor drops thin
    markets before ranking.
    """
    universe = {
        # High-volume stablecoin-vs-stablecoin -- should be gated out.
        "USDT/USDC": spot_market("USDT/USDC", "USDT", "@200"),
        # Normal crypto markets to confirm they still pass through.
        "BTC/USDC": spot_market("BTC/USDC", "BTC", "@1"),
        "ETH/USDC": spot_market("ETH/USDC", "ETH", "@2"),
    }
    # USDT gets the highest volume -- without the gate it would rank first.
    volumes = {
        "USDT/USDC": 999_000_000.0,
        "BTC/USDC": 50_000_000.0,
        "ETH/USDC": 20_000_000.0,
    }
    scored = screening.score_crypto(universe, volumes, {})
    symbols = [r["symbol"] for r in scored]
    assert "USDT-USD" not in symbols, "USDT-base market must never appear in scored output"
    assert "BTC-USD" in symbols
    assert "ETH-USD" in symbols


def test_all_stablecoin_bases_are_excluded_by_the_gate():
    """Every entry in STABLECOIN_BASES is independently gated out."""
    stablecoin_markets = {
        f"{coin}/USDC": spot_market(f"{coin}/USDC", coin, f"@{i}")
        for i, coin in enumerate(screening.STABLECOIN_BASES)
    }
    volumes = {sym: 1_000_000.0 for sym in stablecoin_markets}
    scored = screening.score_crypto(stablecoin_markets, volumes, {})
    assert scored == [], (
        f"Expected no scored output; got: {[r['symbol'] for r in scored]}"
    )


def test_stablecoin_gate_is_case_insensitive():
    """Lowercase stablecoin base symbols are caught by the gate."""
    universe = {"usdt/USDC": spot_market("usdt/USDC", "usdt", "@300")}
    volumes = {"usdt/USDC": 1_000_000.0}
    scored = screening.score_crypto(universe, volumes, {})
    assert scored == []


def test_crypto_scoring_converts_to_the_internal_dash_usd_symbol():
    universe = {"BTC/USDC": spot_market("BTC/USDC", "BTC", "@1")}
    volumes = {"BTC/USDC": 1_000_000.0}
    scored = screening.score_crypto(universe, volumes, {})
    assert scored[0]["symbol"] == "BTC-USD"


def test_crypto_scoring_blends_volume_and_positioning():
    universe = {
        "BTC/USDC": spot_market("BTC/USDC", "BTC", "@1"),
        "ETH/USDC": spot_market("ETH/USDC", "ETH", "@2"),
    }
    volumes = {"BTC/USDC": 1_000_000.0, "ETH/USDC": 1_000_000.0}  # tied volume
    positioning = {"BTC": {"long": 900_000.0, "short": 100_000.0}}  # only BTC has smart-money interest
    scored = screening.score_crypto(universe, volumes, positioning)
    by_symbol = {r["symbol"]: r for r in scored}
    assert by_symbol["BTC-USD"]["score"] > by_symbol["ETH-USD"]["score"]


def test_crypto_scoring_handles_no_positioning_data_at_all():
    universe = {"BTC/USDC": spot_market("BTC/USDC", "BTC", "@1")}
    volumes = {"BTC/USDC": 1_000_000.0}
    scored = screening.score_crypto(universe, volumes, {})
    assert scored[0]["positioning_usd"] is None
    assert scored[0]["score"] >= 0.0


# ---------------------------------------------------------------- file writer


def test_writer_produces_exactly_ten_symbols_with_the_right_split(tmp_path):
    out = tmp_path / "symbols.yaml"
    screening._write_symbols_file(
        str(out),
        equity_symbols=["A", "B", "C", "D", "E"],
        crypto_symbols=["BTC-USD", "ETH-USD", "SOL-USD", "HYPE-USD", "PURR-USD"],
        equity_scored=[{"symbol": s, "score": 0.5} for s in "ABCDE"],
        crypto_scored=[
            {"symbol": s, "score": 0.5}
            for s in ["BTC-USD", "ETH-USD", "SOL-USD", "HYPE-USD", "PURR-USD"]
        ],
    )
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    symbols = doc["symbols"]
    assert len(symbols) == 10
    equities = [s for s in symbols if s["asset_class"] == "equity"]
    cryptos = [s for s in symbols if s["asset_class"] == "crypto"]
    assert len(equities) == 5
    assert len(cryptos) == 5


def test_writer_output_is_shaped_for_mains_loader(tmp_path):
    # main.load_config reads entry["symbol"] / entry["asset_class"] dicts --
    # the exact shape config.yaml's own `symbols:` list already uses.
    out = tmp_path / "symbols.yaml"
    screening._write_symbols_file(
        str(out), ["AAPL"], ["BTC-USD"],
        [{"symbol": "AAPL", "score": 0.1}], [{"symbol": "BTC-USD", "score": 0.1}],
    )
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    for entry in doc["symbols"]:
        assert set(entry.keys()) == {"symbol", "asset_class"}


def test_writer_only_ever_contributes_the_symbols_key(tmp_path):
    # scores/generated_at are informational; main.load_config must never read
    # anything from this file except "symbols".
    out = tmp_path / "symbols.yaml"
    screening._write_symbols_file(
        str(out), ["AAPL"], ["BTC-USD"],
        [{"symbol": "AAPL", "score": 0.1}], [{"symbol": "BTC-USD", "score": 0.1}],
    )
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert set(doc.keys()) >= {"symbols"}
    assert "risk" not in doc and "live_execution" not in doc and "max_risk_pct" not in doc


def test_writer_is_atomic_no_temp_file_left_behind(tmp_path):
    out = tmp_path / "symbols.yaml"
    screening._write_symbols_file(
        str(out), ["AAPL"], ["BTC-USD"],
        [{"symbol": "AAPL", "score": 0.1}], [{"symbol": "BTC-USD", "score": 0.1}],
    )
    assert out.exists()
    assert not (tmp_path / "symbols.yaml.tmp").exists()


def test_writer_overwrites_a_stale_file_cleanly(tmp_path):
    out = tmp_path / "symbols.yaml"
    out.write_text("symbols:\n  - symbol: OLD\n    asset_class: equity\n", encoding="utf-8")
    screening._write_symbols_file(
        str(out), ["NEW"], ["BTC-USD"],
        [{"symbol": "NEW", "score": 0.1}], [{"symbol": "BTC-USD", "score": 0.1}],
    )
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert "OLD" not in [s["symbol"] for s in doc["symbols"]]
    assert "NEW" in [s["symbol"] for s in doc["symbols"]]


# ------------------------------------------------------------- orchestration


@pytest.fixture
def stub_equity_side(monkeypatch):
    """A healthy, signal-bearing equity universe of exactly 5+ candidates."""
    universe = {"AAPL", "MSFT", "GOOG", "AMZN", "META", "NFLX"}
    monkeypatch.setattr(equity_universe, "build_equity_universe", lambda: universe)
    monkeypatch.setattr(
        equity_universe, "fetch_market_movers",
        lambda: {
            "most_actives": [{"symbol": s, "volume": 1e7} for s in universe],
            "gainers": [], "losers": [],
        },
    )


@pytest.fixture
def stub_crypto_side(monkeypatch):
    """A healthy crypto universe of exactly 5+ liquid spot markets."""
    markets = default_markets()
    monkeypatch.setattr(execution, "_hyperliquid_exchange", lambda is_live: FakeMultiMarketExchange(markets))
    monkeypatch.setattr(
        screening, "fetch_crypto_volumes",
        lambda exchange, symbols: {s: 1_000_000.0 for s in symbols if s != "BTC/USDC:USDC"},
    )
    monkeypatch.setattr(screening, "fetch_crypto_positioning", lambda limit=50: {})


def test_run_screening_writes_ten_symbols_on_a_healthy_run(
    tmp_path, stub_equity_side, stub_crypto_side
):
    out = tmp_path / "symbols.yaml"
    rc = screening.run_screening(str(out))
    assert rc == 0
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert len(doc["symbols"]) == 10


def test_run_screening_fails_without_writing_when_equity_universe_is_too_small(
    tmp_path, stub_crypto_side, monkeypatch
):
    monkeypatch.setattr(equity_universe, "build_equity_universe", lambda: {"AAPL"})
    out = tmp_path / "symbols.yaml"
    out.write_text("symbols: [{symbol: OLD, asset_class: equity}]\n", encoding="utf-8")

    rc = screening.run_screening(str(out))

    assert rc == 1
    # last week's file is untouched, not overwritten with a short list
    assert "OLD" in out.read_text(encoding="utf-8")


def test_run_screening_fails_without_writing_when_crypto_universe_is_too_small(
    tmp_path, stub_equity_side, monkeypatch
):
    monkeypatch.setattr(
        execution, "_hyperliquid_exchange",
        lambda is_live: FakeMultiMarketExchange({"BTC/USDC": spot_market("BTC/USDC", "BTC", "@1")}),
    )
    out = tmp_path / "symbols.yaml"
    out.write_text("symbols: [{symbol: OLD, asset_class: crypto}]\n", encoding="utf-8")

    rc = screening.run_screening(str(out))

    assert rc == 1
    assert "OLD" in out.read_text(encoding="utf-8")


def test_run_screening_fails_without_writing_when_too_few_crypto_clear_the_volume_floor(
    tmp_path, stub_equity_side, monkeypatch
):
    markets = default_markets()
    monkeypatch.setattr(execution, "_hyperliquid_exchange", lambda is_live: FakeMultiMarketExchange(markets))
    # Only one market clears the liquidity floor -- not enough for 5.
    monkeypatch.setattr(
        screening, "fetch_crypto_volumes",
        lambda exchange, symbols: {"BTC/USDC": 1_000_000.0},
    )
    monkeypatch.setattr(screening, "fetch_crypto_positioning", lambda limit=50: {})

    out = tmp_path / "symbols.yaml"
    out.write_text("symbols: [{symbol: OLD, asset_class: crypto}]\n", encoding="utf-8")

    rc = screening.run_screening(str(out))

    assert rc == 1
    assert "OLD" in out.read_text(encoding="utf-8")


def test_run_screening_leaves_the_file_untouched_on_an_unexpected_exception(
    tmp_path, stub_crypto_side, monkeypatch
):
    def boom():
        raise RuntimeError("something in FMP parsing broke")

    monkeypatch.setattr(equity_universe, "build_equity_universe", boom)
    out = tmp_path / "symbols.yaml"
    out.write_text("symbols: [{symbol: OLD, asset_class: equity}]\n", encoding="utf-8")

    rc = screening.run_screening(str(out))

    assert rc == 1
    assert "OLD" in out.read_text(encoding="utf-8")


def test_run_screening_never_touches_the_file_on_first_failed_run(tmp_path, monkeypatch):
    # No pre-existing file at all -- a failure must not create a bad one either.
    monkeypatch.setattr(equity_universe, "build_equity_universe", lambda: set())
    out = tmp_path / "symbols.yaml"

    rc = screening.run_screening(str(out))

    assert rc == 1
    assert not out.exists()
