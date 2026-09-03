"""Trader-positioning intelligence: shape, hedging, and failing soft.

The load-bearing property is negative: this data is an input to the model's
reasoning and can never become a shortcut around the risk layer.
"""

import json

import pytest

import market_intel


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """No test in this file may touch the network."""

    def boom(*a, **kw):
        raise AssertionError("unexpected network call")

    monkeypatch.setattr(market_intel.requests, "get", boom)
    monkeypatch.setattr(market_intel.requests, "post", boom)


def leaderboard_row(address, value, roi):
    return {
        "ethAddress": address,
        "accountValue": str(value),
        "windowPerformances": [
            ["day", {"pnl": "1", "roi": "0.01", "vlm": "1"}],
            ["month", {"pnl": "1", "roi": str(roi), "vlm": "1"}],
        ],
    }


# ------------------------------------------------------------------- symbols


@pytest.mark.parametrize(
    "symbol,coin",
    [("BTC-USD", "BTC"), ("ETH-USDT", "ETH"), ("sol-usd", "SOL"), ("HYPE", "HYPE")],
)
def test_base_symbol(symbol, coin):
    assert market_intel.base_symbol(symbol) == coin


# --------------------------------------------------------------- ranking


def test_top_wallets_ignores_small_accounts():
    # A 900% month on $200 is noise, not information.
    rows = [
        leaderboard_row("0xsmall", 200, 9.0),
        leaderboard_row("0xbig", 500_000, 0.4),
    ]
    assert [a for a, _ in market_intel.top_wallets(rows)] == ["0xbig"]


def test_top_wallets_ranks_by_return_not_size():
    rows = [
        leaderboard_row("0xhuge", 90_000_000, 0.01),
        leaderboard_row("0xskilled", 250_000, 1.5),
    ]
    assert [a for a, _ in market_intel.top_wallets(rows)][0] == "0xskilled"


def test_top_wallets_respects_the_limit():
    rows = [leaderboard_row(f"0x{i}", 1_000_000, i / 10) for i in range(50)]
    assert len(market_intel.top_wallets(rows, limit=5)) == 5


def test_top_wallets_survives_malformed_rows():
    rows = [
        {"ethAddress": "0xa"},  # no accountValue
        {"accountValue": "999999"},  # no address
        leaderboard_row("0xgood", 1_000_000, 0.5),
    ]
    assert [a for a, _ in market_intel.top_wallets(rows)] == ["0xgood"]


# ------------------------------------------------------------ aggregation


def state(*positions):
    return {
        "assetPositions": [
            {"position": {"coin": c, "szi": str(s), "positionValue": str(v)}}
            for c, s, v in positions
        ]
    }


def test_aggregate_splits_long_and_short_notional(monkeypatch):
    responses = {
        "0xa": state(("BTC", 1.0, 100_000), ("ETH", -2.0, 50_000)),
        "0xb": state(("BTC", 0.5, 40_000)),
    }
    monkeypatch.setattr(market_intel, "post_info", lambda body: responses[body["user"]])

    totals = market_intel.aggregate_positioning([("0xa", 1.0), ("0xb", 0.5)])
    assert totals["BTC"]["long"] == pytest.approx(140_000)
    assert totals["BTC"]["short"] == pytest.approx(0)
    assert totals["ETH"]["short"] == pytest.approx(50_000)
    assert totals["BTC"]["sampled_wallets"] == 2


def test_a_failing_wallet_is_skipped_not_fatal(monkeypatch):
    def flaky(body):
        if body["user"] == "0xbad":
            raise RuntimeError("timeout")
        return state(("BTC", 1.0, 100_000))

    monkeypatch.setattr(market_intel, "post_info", flaky)
    totals = market_intel.aggregate_positioning([("0xbad", 1.0), ("0xok", 0.5)])
    assert totals["BTC"]["long"] == pytest.approx(100_000)
    assert totals["BTC"]["sampled_wallets"] == 1  # only one actually answered


def test_zero_size_positions_are_ignored(monkeypatch):
    monkeypatch.setattr(market_intel, "post_info", lambda body: state(("BTC", 0.0, 100_000)))
    assert market_intel.aggregate_positioning([("0xa", 1.0)]) == {}


# --------------------------------------------------------------- summary


def totals_for(long_usd, short_usd, wallets=5, sampled=10):
    return {
        "BTC": {
            "long": long_usd, "short": short_usd,
            "wallets": float(wallets), "sampled_wallets": float(sampled),
        }
    }


def test_summary_reports_a_net_long_lean():
    text = market_intel.summarise("BTC", totals_for(900_000, 100_000), None)
    assert "net long by roughly 80%" in text
    assert "$900,000 long" in text


def test_summary_reports_a_net_short_lean():
    text = market_intel.summarise("BTC", totals_for(100_000, 900_000), None)
    assert "net short by roughly 80%" in text


def test_summary_always_labels_the_data_as_perps_not_spot():
    # These are leveraged perpetual positions; the bot trades spot unleveraged.
    # Implying otherwise would misrepresent the signal to the model.
    text = market_intel.summarise("BTC", totals_for(900_000, 100_000), 0.0000125)
    assert "PERPETUAL" in text
    assert "not spot holdings" in text
    assert "trades spot without leverage" in text


def test_summary_always_hedges_it_as_bias_not_certainty():
    text = market_intel.summarise("BTC", totals_for(900_000, 100_000), None)
    assert "directional bias only" in text
    assert "never as confirmation" in text
    assert "own technical justification" in text


def test_summary_includes_funding_direction():
    positive = market_intel.summarise("BTC", {}, 0.0000125)
    negative = market_intel.summarise("BTC", {}, -0.0000125)
    assert "longs are paying shorts" in positive
    assert "shorts are paying longs" in negative


def test_summary_is_none_when_there_is_nothing_to_say():
    # Silence beats a padded or invented summary.
    assert market_intel.summarise("BTC", {}, None) is None


def test_summary_ignores_a_sample_too_thin_to_mean_anything():
    assert market_intel.summarise("BTC", totals_for(500.0, 100.0), None) is None


def test_summary_skips_a_coin_nobody_holds():
    assert market_intel.summarise("DOGE", totals_for(900_000, 100_000), None) is None


# ------------------------------------------------------- fetch_positioning


def test_equities_get_no_positioning_and_make_no_network_call():
    # The autouse fixture turns any network call into an assertion failure, so
    # this also proves the equity path short-circuits before any request.
    assert market_intel.fetch_positioning("AAPL", "equity") is None


def test_positioning_returns_none_when_the_leaderboard_is_unavailable(monkeypatch):
    monkeypatch.setattr(market_intel, "fetch_leaderboard", lambda cache_path: [])
    monkeypatch.setattr(market_intel, "fetch_funding_rate", lambda coin: None)
    assert market_intel.fetch_positioning("BTC-USD", "crypto") is None


def test_positioning_never_raises(monkeypatch):
    # A cycle must never stop because a leaderboard was slow.
    def boom(*a, **kw):
        raise RuntimeError("everything is on fire")

    monkeypatch.setattr(market_intel, "fetch_leaderboard", boom)
    assert market_intel.fetch_positioning("BTC-USD", "crypto") is None


def test_positioning_end_to_end_with_stubs(monkeypatch):
    monkeypatch.setattr(
        market_intel, "fetch_leaderboard",
        lambda cache_path: [leaderboard_row("0xa", 1_000_000, 0.9)],
    )
    monkeypatch.setattr(
        market_intel, "post_info", lambda body: state(("BTC", 1.0, 900_000))
    )
    monkeypatch.setattr(market_intel, "fetch_funding_rate", lambda coin: 0.0000125)

    text = market_intel.fetch_positioning("BTC-USD", "crypto")
    assert "net long by roughly 100%" in text
    assert "PERPETUAL" in text


# ----------------------------------------------------------------- cache


def test_a_fresh_cache_is_reused_without_a_download(tmp_path):
    # The leaderboard is ~37MB; re-downloading it per symbol per cycle is waste.
    # The autouse no-network fixture makes a download fail loudly here.
    cache = tmp_path / "lb.json"
    cache.write_text(json.dumps({"leaderboardRows": [leaderboard_row("0xa", 1_000_000, 0.5)]}),
                     encoding="utf-8")
    rows = market_intel.fetch_leaderboard(str(cache))
    assert [r["ethAddress"] for r in rows] == ["0xa"]


def test_an_unreadable_cache_degrades_to_empty(tmp_path):
    cache = tmp_path / "lb.json"
    cache.write_text("not json at all", encoding="utf-8")
    assert market_intel.fetch_leaderboard(str(cache)) == []


# ------------------------------------------- funding vs wallets must stay honest


def test_summary_flags_a_divergence_instead_of_claiming_agreement():
    # Wallets net LONG but funding negative (crowd leaning short). Asserting the
    # two agree would feed the model a falsehood dressed up as data.
    text = market_intel.summarise("BTC", totals_for(8_000_000, 0), -0.0000006)
    assert "net long" in text
    assert "shorts are paying longs" in text
    assert "OPPOSITE direction" in text
    assert "same direction as the wallet sample" not in text


def test_summary_confirms_agreement_only_when_it_is_real():
    text = market_intel.summarise("BTC", totals_for(9_000_000, 100_000), 0.0000125)
    assert "same direction as the wallet sample" in text
    assert "OPPOSITE" not in text


def test_summary_makes_no_agreement_claim_without_a_wallet_sample():
    text = market_intel.summarise("BTC", {}, 0.0000125)
    assert "broader market on the long side" in text
    assert "wallet sample" not in text


@pytest.mark.parametrize(
    "wallet_long,funding,expected",
    [
        (True, 0.00001, "same direction"),
        (True, -0.00001, "OPPOSITE"),
        (False, -0.00001, "same direction"),
        (False, 0.00001, "OPPOSITE"),
    ],
)
def test_every_lean_combination_is_described_correctly(wallet_long, funding, expected):
    totals = totals_for(9_000_000, 0) if wallet_long else totals_for(0, 9_000_000)
    assert expected in market_intel.summarise("BTC", totals, funding)
