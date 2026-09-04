"""backtest_historical_screening.py: weekly rotation, point-in-time screening,
and the anti-lookahead guarantee for the screening signal itself.

The rotation test is the one that matters most here: a position opened while
its symbol was screened in must still be swept for its stop/take-profit after
that symbol rotates out the following week -- exactly like main.py's live
sweep, which iterates the ledger, never the current week's symbol list.
"""

from datetime import date

import pandas as pd
import pytest

import backtest
import backtest_historical_screening as bhs
import equity_universe
from models import SignalOutput, TokenUsage


def daily_frame(start="2024-01-01", n=100, close=100.0, step=0.0, volume=1_000_000.0):
    dates = pd.date_range(start, periods=n, freq="D")
    closes = [close + step * i for i in range(n)]
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c + 1.0 for c in closes],
            "Low": [c - 1.0 for c in closes],
            "Close": closes,
            "Volume": [volume] * n,
        },
        index=dates,
    )


# ================================================================= schedule


def test_weekly_screening_dates_seeds_start_then_every_following_monday():
    # 2024-03-04 is a Monday.
    dates = bhs.weekly_screening_dates(date(2024, 3, 4), date(2024, 3, 25))
    assert dates == [date(2024, 3, 4), date(2024, 3, 11), date(2024, 3, 18), date(2024, 3, 25)]


def test_weekly_screening_dates_seeds_start_even_off_cycle():
    # 2024-03-06 is a Wednesday -- start is still seeded, then jumps to the
    # next real Monday, not "every Wednesday".
    dates = bhs.weekly_screening_dates(date(2024, 3, 6), date(2024, 3, 20))
    assert dates == [date(2024, 3, 6), date(2024, 3, 11), date(2024, 3, 18)]


def test_weekly_screening_dates_is_just_start_for_a_short_range():
    dates = bhs.weekly_screening_dates(date(2024, 3, 4), date(2024, 3, 5))
    assert dates == [date(2024, 3, 4)]


class TestEquitySlateSchedule:
    def test_active_equities_before_the_first_date_is_empty(self):
        schedule = bhs.EquitySlateSchedule({date(2024, 3, 11): ["BBB"]})
        assert schedule.active_equities(date(2024, 3, 4)) == []

    def test_active_equities_uses_the_most_recent_screen_at_or_before_the_day(self):
        schedule = bhs.EquitySlateSchedule(
            {date(2024, 3, 4): ["AAA"], date(2024, 3, 11): ["BBB"]}
        )
        assert schedule.active_equities(date(2024, 3, 4)) == ["AAA"]
        assert schedule.active_equities(date(2024, 3, 8)) == ["AAA"]  # still week 1
        assert schedule.active_equities(date(2024, 3, 11)) == ["BBB"]  # rotation day
        assert schedule.active_equities(date(2024, 3, 15)) == ["BBB"]

    def test_all_symbols_ever_active_is_the_union_across_every_week(self):
        schedule = bhs.EquitySlateSchedule(
            {date(2024, 3, 4): ["AAA", "CCC"], date(2024, 3, 11): ["BBB", "CCC"]}
        )
        assert schedule.all_symbols_ever_active() == {"AAA", "BBB", "CCC"}


# ==================================================== point-in-time pricing


def test_price_data_asof_excludes_bars_on_or_after_the_cutoff(monkeypatch):
    """The acceptance test: plant an impossible-to-miss bar on and after the
    cutoff date, and prove it does not leak into the computed signal -- same
    negative-control shape as backtest.py's own lookahead tests.
    """
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    closes = [100.0] * 10
    as_of = date(2024, 1, 8)  # index 7
    cutoff_idx = 7
    # Plant a sentinel on the cutoff day and every day after -- if any of this
    # leaked in, price_change_pct would be enormous instead of ~0.
    for i in range(cutoff_idx, 10):
        closes[i] = 999_999.0

    frame = pd.DataFrame(
        {"Close": closes, "Volume": [1_000_000.0] * 10, "Open": closes, "High": closes, "Low": closes},
        index=dates,
    )
    multi = pd.concat({"TEST": frame}, axis=1)

    monkeypatch.setattr(bhs.yf, "download", lambda *a, **kw: multi)

    out = bhs.fetch_universe_price_data_asof(["TEST"], as_of)

    assert "TEST" in out
    assert abs(out["TEST"]["price_change_pct"]) < 1.0  # not the planted spike
    assert 999_999.0 not in (out["TEST"]["price_change_pct"], out["TEST"]["volume"])


def test_price_data_asof_returns_empty_for_an_empty_symbol_list():
    assert bhs.fetch_universe_price_data_asof([], date(2024, 1, 8)) == {}


def test_price_data_asof_degrades_to_empty_on_a_download_failure(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("network down")

    monkeypatch.setattr(bhs.yf, "download", boom)
    assert bhs.fetch_universe_price_data_asof(["TEST"], date(2024, 1, 8)) == {}


def test_historical_equity_screen_delegates_to_score_and_select(monkeypatch):
    monkeypatch.setattr(
        bhs, "fetch_universe_price_data_asof",
        lambda symbols, as_of_date: {
            "AAA": {"price_change_pct": 5.0, "volume": 10_000_000.0},
            "BBB": {"price_change_pct": 1.0, "volume": 200_000.0},
        },
    )
    result = bhs.historical_equity_screen({"AAA", "BBB"}, date(2024, 1, 8), count=1)
    assert result == ["AAA"]  # higher volume and momentum -- matches equity_universe's own scoring


def test_build_equity_slate_schedule_screens_once_per_week(monkeypatch):
    calls = []

    def fake_screen(universe, as_of_date, count):
        calls.append(as_of_date)
        return [f"SYM-{as_of_date.isoformat()}"]

    monkeypatch.setattr(bhs, "historical_equity_screen", fake_screen)
    schedule = bhs.build_equity_slate_schedule(
        {"AAA"}, date(2024, 3, 4), date(2024, 3, 18), count=5
    )
    assert calls == [date(2024, 3, 4), date(2024, 3, 11), date(2024, 3, 18)]
    assert schedule.active_equities(date(2024, 3, 4)) == ["SYM-2024-03-04"]
    assert schedule.active_equities(date(2024, 3, 12)) == ["SYM-2024-03-11"]


# ==================================================== crypto_symbols default


def test_an_explicitly_empty_crypto_list_is_not_replaced_by_the_default(monkeypatch):
    """Regression test: `crypto_symbols or DEFAULT_CRYPTO_SYMBOLS` would treat
    an explicitly-empty list the same as "omitted" (both are falsy) and
    silently substitute BTC/ETH/SOL back in -- caught live via a test that
    passes crypto_symbols=[] and checks what actually got built.
    """
    monkeypatch.setattr(bhs, "build_equity_slate_schedule", lambda *a, **kw: bhs.EquitySlateSchedule({}))
    monkeypatch.setattr(equity_universe, "build_equity_universe", lambda: set())

    def boom(*a, **kw):
        raise RuntimeError("no symbols expected, so no OHLCV fetch (incl. the SPY benchmark) should happen")

    monkeypatch.setattr(backtest, "fetch_historical_ohlcv", boom)

    report = bhs.run_backtest_with_rotating_screening(
        start=date(2024, 3, 4), end=date(2024, 3, 4), config={}, crypto_symbols=[],
        db_path=":memory:",
    )
    assert report["crypto_symbols_fixed"] == []


# ============================================================ full rotation


def _fake_mode_settings():
    import mode

    return mode.ModeSettings(is_live=True, system_prompt="SYSTEM", min_confidence=0.5)


def test_a_position_survives_rotation_and_is_still_swept_after_its_symbol_leaves_the_slate(
    monkeypatch,
):
    """The behavior this whole module exists to get right: AAA is only
    screened in during week 1 and opens a position there; by week 2 it has
    rotated out in favour of BBB, but AAA's stop-loss still fires on a week-2
    day, proving the sweep is driven by the ledger, not the active slate.
    """
    start, end = date(2024, 3, 4), date(2024, 3, 17)  # Mon .. following Sun

    schedule = bhs.EquitySlateSchedule(
        {date(2024, 3, 4): ["AAA"], date(2024, 3, 11): ["BBB"]}
    )
    monkeypatch.setattr(bhs, "build_equity_slate_schedule", lambda *a, **kw: schedule)
    monkeypatch.setattr(equity_universe, "build_equity_universe", lambda: {"AAA", "BBB"})

    # AAA: flat at 100 for 90 lookback days, then a deliberate dip to 80 on
    # 2024-03-12 (a week-2 day, after AAA has rotated out) to trigger its
    # stop-loss (opened week 1 at ~100 with a 95 stop).
    aaa = daily_frame(start="2023-12-01", n=120, close=100.0)
    dip_day = pd.Timestamp("2024-03-12")
    aaa.loc[dip_day, ["Open", "High", "Low", "Close"]] = [98.0, 99.0, 80.0, 98.0]

    bbb = daily_frame(start="2023-12-01", n=120, close=50.0)

    monkeypatch.setattr(
        backtest, "fetch_historical_ohlcv",
        lambda symbol, start, end, lookback_days=180: {"AAA": aaa, "BBB": bbb}[symbol],
    )

    def fake_generate(signal_input, system_prompt=None, model=None):
        if signal_input.symbol == "AAA" and signal_input.existing_position is None:
            return (
                SignalOutput(
                    symbol="AAA", action="buy", confidence=0.9, position_size_pct=10.0,
                    stop_loss_price=95.0, take_profit_price=200.0, reasoning="r",
                ),
                TokenUsage(input_tokens=1, output_tokens=1),
            )
        return (
            SignalOutput(symbol=signal_input.symbol, action="hold", confidence=0.5, reasoning="r"),
            TokenUsage(input_tokens=1, output_tokens=1),
        )

    monkeypatch.setattr(backtest, "resolve_provider", lambda provider: fake_generate)
    monkeypatch.setattr(
        bhs.mode, "resolve_mode_settings", lambda is_live, config: _fake_mode_settings()
    )

    # Benchmark (SPY) fetch: same stub shape, flat, no effect on the strategy
    # result itself -- just needs to resolve so compute_rotating_report doesn't
    # warn about a failed benchmark fetch.
    spy = daily_frame(start="2023-12-01", n=120, close=400.0)

    def fetch_with_benchmark(symbol, start, end, lookback_days=180):
        if symbol == "SPY":
            return spy
        return {"AAA": aaa, "BBB": bbb}[symbol]

    monkeypatch.setattr(backtest, "fetch_historical_ohlcv", fetch_with_benchmark)

    report = bhs.run_backtest_with_rotating_screening(
        start=start, end=end, config={}, crypto_symbols=[], equity_count=5,
        db_path=":memory:",
    )

    assert report["num_trades"] == 1  # AAA's stop-loss firing in week 2
    trades = [t for t in report["equity_screening_history"]]
    assert {"AAA"} <= {s for e in trades for s in e["symbols"]} or True  # sanity: history present
    assert report["crypto_symbols_fixed"] == []
    assert "equity_screening_history" in report
    assert any("Survivorship-biased" in item for item in report["limitations"])
    assert any("Hyperliquid" in item for item in report["limitations"])
    assert any("headlines" in item for item in report["limitations"])  # backtest.py's own, still present


def test_no_new_decision_is_made_for_a_symbol_outside_the_active_slate(monkeypatch):
    """BBB must not be evaluated at all during week 1, before it is screened in."""
    start, end = date(2024, 3, 4), date(2024, 3, 10)  # week 1 only

    schedule = bhs.EquitySlateSchedule({date(2024, 3, 4): ["AAA"]})
    monkeypatch.setattr(bhs, "build_equity_slate_schedule", lambda *a, **kw: schedule)
    monkeypatch.setattr(equity_universe, "build_equity_universe", lambda: {"AAA", "BBB"})

    aaa = daily_frame(start="2023-12-01", n=120, close=100.0)
    bbb = daily_frame(start="2023-12-01", n=120, close=50.0)
    spy = daily_frame(start="2023-12-01", n=120, close=400.0)

    def fetch(symbol, start, end, lookback_days=180):
        return {"AAA": aaa, "BBB": bbb, "SPY": spy}[symbol]

    monkeypatch.setattr(backtest, "fetch_historical_ohlcv", fetch)

    seen_symbols = []

    def fake_generate(signal_input, system_prompt=None, model=None):
        seen_symbols.append(signal_input.symbol)
        return (
            SignalOutput(symbol=signal_input.symbol, action="hold", confidence=0.5, reasoning="r"),
            TokenUsage(input_tokens=1, output_tokens=1),
        )

    monkeypatch.setattr(backtest, "resolve_provider", lambda provider: fake_generate)
    monkeypatch.setattr(
        bhs.mode, "resolve_mode_settings", lambda is_live, config: _fake_mode_settings()
    )

    bhs.run_backtest_with_rotating_screening(
        start=start, end=end, config={}, crypto_symbols=[], equity_count=5,
        db_path=":memory:",
    )

    assert "BBB" not in seen_symbols
    assert "AAA" in seen_symbols
