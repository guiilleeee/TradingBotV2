"""Trader-positioning intelligence: shape, hedging, and failing soft.

The load-bearing property is negative: this data is an input to the model's
reasoning and can never become a shortcut around the risk layer.
"""

import json
import time

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


# ------------------------------------------------------------ recent trades


def fill(coin, dir_, sz, px, time_ms, oid=1):
    return {"coin": coin, "sz": str(sz), "px": str(px), "dir": dir_, "time": time_ms, "oid": oid}


def test_recent_trades_groups_partial_fills_sharing_an_order_id():
    # Verified live against Hyperliquid: one order routinely fills across several
    # price levels as separate rows sharing one oid -- these must collapse to one
    # trade, not read as several independent trades.
    fills = [
        fill("BTC", "Open Long", 0.3, 50000, 1_000_000, oid=1),
        fill("BTC", "Open Long", 0.2, 50010, 1_000_050, oid=1),
    ]
    trades = market_intel.recent_trades_for_coin(fills, "BTC", since_ms=0)
    assert len(trades) == 1
    assert trades[0]["notional"] == pytest.approx(0.3 * 50000 + 0.2 * 50010)
    assert trades[0]["time"] == 1_000_050
    assert trades[0]["dir"] == "Open Long"


def test_recent_trades_ignores_other_coins():
    fills = [fill("ETH", "Open Long", 10, 3000, 1_000_000, oid=1)]
    assert market_intel.recent_trades_for_coin(fills, "BTC", since_ms=0) == []


def test_recent_trades_ignores_spot_fills_by_construction():
    # Spot fills are named "@<index>", never a bare coin symbol -- they can never
    # match `coin`, so a spot dust conversion or spot sell never leaks in here.
    fills = [fill("@142", "Sell", 1, 100, 1_000_000, oid=2)]
    assert market_intel.recent_trades_for_coin(fills, "BTC", since_ms=0) == []


def test_recent_trades_ignores_non_significant_fill_directions():
    fills = [fill("BTC", "Buy", 1.0, 50000, 1_000_000, oid=1)]
    assert market_intel.recent_trades_for_coin(fills, "BTC", since_ms=0) == []


def test_recent_trades_excludes_fills_before_the_cutoff():
    fills = [fill("BTC", "Open Long", 1.0, 50000, 500, oid=1)]
    assert market_intel.recent_trades_for_coin(fills, "BTC", since_ms=1000) == []


def test_recent_trades_drops_dust_below_the_notional_floor():
    fills = [fill("BTC", "Open Long", 0.001, 50000, 1_000_000, oid=1)]  # $50
    assert market_intel.recent_trades_for_coin(fills, "BTC", since_ms=0) == []


def test_recent_trades_are_sorted_newest_first():
    fills = [
        fill("BTC", "Open Long", 1.0, 50000, 1_000_000, oid=1),
        fill("BTC", "Close Long", 1.0, 51000, 2_000_000, oid=2),
    ]
    trades = market_intel.recent_trades_for_coin(fills, "BTC", since_ms=0)
    assert [t["time"] for t in trades] == [2_000_000, 1_000_000]


# --------------------------------------------------------- describing trades


def test_describe_recent_trades_formats_amount_direction_and_recency():
    now_ms = 10_000_000_000
    entries = [("0xabcdef1234", {"dir": "Open Long", "time": now_ms - 3 * 3_600_000, "notional": 52000.0})]
    text = market_intel.describe_recent_trades(entries, now_ms)
    assert "wallet ending ...1234" in text
    assert "opened a $52,000 long" in text
    assert "3 hours ago" in text


def test_describe_recent_trades_pluralises_a_single_hour_and_minute_correctly():
    now_ms = 10_000_000_000
    one_hour = [("0x0000000001", {"dir": "Close Short", "time": now_ms - 3_600_000, "notional": 1000.0})]
    assert "1 hour ago" in market_intel.describe_recent_trades(one_hour, now_ms)

    one_min = [("0x0000000002", {"dir": "Close Short", "time": now_ms - 60_000, "notional": 1000.0})]
    assert "1 minute ago" in market_intel.describe_recent_trades(one_min, now_ms)


def test_describe_recent_trades_returns_none_when_empty():
    assert market_intel.describe_recent_trades([], 10_000_000) is None


def test_describe_recent_trades_caps_the_line_count():
    now_ms = 10_000_000_000
    entries = [
        (f"0x{i:040d}", {"dir": "Open Long", "time": now_ms - i * 60_000, "notional": 10_000.0})
        for i in range(10)
    ]
    text = market_intel.describe_recent_trades(entries, now_ms)
    assert text.count("wallet ending") == market_intel.MAX_RECENT_TRADES_IN_SUMMARY


# ------------------------------------------------------- fetch_recent_trade_notes


def test_fetch_recent_trade_notes_skips_a_failing_wallet(monkeypatch):
    now_ms = time.time() * 1000.0

    def flaky(body):
        if body["user"] == "0xbad":
            raise RuntimeError("timeout")
        return [fill("BTC", "Open Long", 1.0, 50000, now_ms - 3 * 3_600_000, oid=1)]

    monkeypatch.setattr(market_intel, "post_info", flaky)
    text = market_intel.fetch_recent_trade_notes("BTC", [("0xbad", 1.0), ("0xok", 0.5)], now_ms)
    assert text is not None
    assert "opened a $50,000 long" in text


def test_fetch_recent_trade_notes_returns_none_when_nothing_recent(monkeypatch):
    monkeypatch.setattr(market_intel, "post_info", lambda body: [])
    assert market_intel.fetch_recent_trade_notes("BTC", [("0xa", 1.0)]) is None


def test_fetch_recent_trade_notes_degrades_on_an_unexpected_response_shape(monkeypatch):
    monkeypatch.setattr(market_intel, "post_info", lambda body: {"unexpected": "shape"})
    assert market_intel.fetch_recent_trade_notes("BTC", [("0xa", 1.0)]) is None


# ---------------------------------------------- summarise with recent trade notes


def test_summary_includes_recent_trade_notes_alongside_aggregate():
    notes = "Specific recent activity from the same sampled wallets in the last ~24h: wallet ending ...1234 opened a $52,000 long 3 hours ago."
    text = market_intel.summarise("BTC", totals_for(900_000, 100_000), None, notes)
    assert "net long by roughly 80%" in text
    assert "wallet ending ...1234 opened a $52,000 long 3 hours ago" in text
    assert "ago.." not in text  # no double-period join artifact


def test_summary_is_not_none_when_only_recent_trade_notes_exist():
    notes = "Specific recent activity from the same sampled wallets in the last ~24h: wallet ending ...1234 opened a $52,000 long 3 hours ago."
    text = market_intel.summarise("BTC", {}, None, notes)
    assert text is not None
    assert "wallet ending ...1234" in text
    assert "not spot holdings" in text  # the hedge still applies with no aggregate line


def test_summary_still_none_when_nothing_at_all():
    assert market_intel.summarise("BTC", {}, None, None) is None


def test_summary_hedge_still_mentions_trades_not_just_positions():
    notes = "Specific recent activity from the same sampled wallets in the last ~24h: wallet ending ...1234 opened a $52,000 long 3 hours ago."
    text = market_intel.summarise("BTC", totals_for(900_000, 100_000), None, notes)
    assert "trades spot without leverage" in text
    assert "directional bias only" in text


def test_positioning_end_to_end_includes_recent_trades(monkeypatch):
    monkeypatch.setattr(
        market_intel, "fetch_leaderboard",
        lambda cache_path: [leaderboard_row("0xa", 1_000_000, 0.9)],
    )
    now_ms = time.time() * 1000.0

    def fake_post_info(body):
        if body["type"] == "clearinghouseState":
            return state(("BTC", 1.0, 900_000))
        if body["type"] == "userFills":
            return [fill("BTC", "Open Long", 1.04, 50000, now_ms - 3 * 3_600_000, oid=1)]
        raise AssertionError(f"unexpected info type: {body['type']}")

    monkeypatch.setattr(market_intel, "post_info", fake_post_info)
    monkeypatch.setattr(market_intel, "fetch_funding_rate", lambda coin: None)

    text = market_intel.fetch_positioning("BTC-USD", "crypto")
    assert "net long by roughly 100%" in text
    assert "wallet ending" in text
    assert "opened a $52,000 long" in text
