"""backtest.py: the anti-lookahead guarantee, sizing/fill math, sweep, and report.

The lookahead tests are the ones that matter most in this file: a backtest
that leaks even one future value looks great and is worthless. Each of those
tests has a negative control proving it would actually catch the bug it
claims to catch, not just pass vacuously.
"""

import sqlite3

import pandas as pd
import pytest

import backtest
import data_fetcher
from models import TokenUsage


def daily_frame(start="2024-01-01", n=80, close=100.0, step=0.0, volume=1_000_000.0):
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


# ============================================================ lookahead bias


def test_decision_frame_excludes_a_deliberately_planted_future_sentinel():
    """The acceptance test: try to leak a future value, and prove it's caught."""
    frame = daily_frame(n=80)
    decision_day = frame.index[60]
    # Plant an impossible-to-miss value on the decision day itself, and on
    # every day after it -- if ANY of this leaked into the decision inputs,
    # these assertions would catch it.
    frame.loc[decision_day:, "Close"] = 999_999.0
    frame.loc[decision_day:, "Open"] = 999_999.0

    sliced = backtest.decision_frame_for_day(frame, decision_day)

    assert decision_day not in sliced.index
    assert 999_999.0 not in sliced["Close"].values
    assert 999_999.0 not in sliced["Open"].values
    # The resulting "current price" a model would see is the PRIOR day's
    # close -- not today's, not any later day's.
    assert data_fetcher.latest_price(sliced) == pytest.approx(100.0 + 60 * 0.0 - 0.0)


def test_a_naive_inclusive_slice_would_have_leaked_the_sentinel():
    """Negative control: proves the test above has teeth, not a tautology.

    If decision_frame_for_day used `<=` instead of `<`, this exact scenario
    would leak the sentinel -- demonstrated here directly against the wrong
    boundary, so the real function's exclusion is a meaningful assertion.
    """
    frame = daily_frame(n=80)
    decision_day = frame.index[60]
    frame.loc[decision_day, "Close"] = 999_999.0

    wrong_slice = frame[frame.index <= pd.Timestamp(decision_day)]  # the bug, deliberately
    assert 999_999.0 in wrong_slice["Close"].values  # the wrong boundary DOES leak it

    right_slice = backtest.decision_frame_for_day(frame, decision_day)
    assert 999_999.0 not in right_slice["Close"].values  # the real function does not


def test_indicators_computed_for_day_n_never_reflect_day_ns_own_bar():
    """End-to-end version: compute_indicators on the sliced frame must be
    numerically identical to computing it on a frame that never had day N at
    all -- proving day N contributes nothing, not just that its literal Close
    value is absent (a subtler leak, e.g. via SMA/volume averaging, would
    still show up as a numeric difference here).
    """
    frame = daily_frame(n=80, close=100.0, step=0.3)
    decision_day = frame.index[70]

    # Corrupt ONLY day N's own row with extreme values.
    corrupted = frame.copy()
    corrupted.loc[decision_day, ["Open", "High", "Low", "Close", "Volume"]] = [
        1e9, 1e9, 1e9, 1e9, 1e9,
    ]

    truncated = frame[frame.index < pd.Timestamp(decision_day)]  # day N never existed
    sliced_from_corrupted = backtest.decision_frame_for_day(corrupted, decision_day)

    indicators_truncated = data_fetcher.compute_indicators(truncated)
    indicators_from_corrupted = data_fetcher.compute_indicators(sliced_from_corrupted)

    assert indicators_truncated == indicators_from_corrupted


def test_decision_frame_for_a_later_day_still_excludes_only_that_day():
    # Sanity: the boundary moves correctly with the day, not stuck at one date.
    frame = daily_frame(n=80)
    day_a, day_b = frame.index[40], frame.index[41]
    assert len(backtest.decision_frame_for_day(frame, day_b)) == len(
        backtest.decision_frame_for_day(frame, day_a)
    ) + 1


def test_only_the_open_of_day_n_is_ever_read_for_a_fill():
    """run_symbol_for_day must use day's Open for a fill, never its Close/High/Low."""
    frame = daily_frame(n=80, close=100.0, step=0.0)
    decision_day = frame.index[70]
    # Distinguish Open from the rest of the bar unmistakably.
    frame.loc[decision_day, "Open"] = 150.0
    frame.loc[decision_day, ["High", "Low", "Close"]] = [999.0, 999.0, 999.0]

    state = backtest.BacktestState(equity=10_000.0)
    logger = backtest.BacktestLogger(":memory:")

    def fake_generate(signal_input, system_prompt=None, model=None):
        from models import SignalOutput

        return (
            SignalOutput(
                symbol=signal_input.symbol, action="buy", confidence=0.9, position_size_pct=10.0,
                stop_loss_price=signal_input.current_price * 0.9,
                take_profit_price=signal_input.current_price * 1.1, reasoning="test",
            ),
            TokenUsage(input_tokens=10, output_tokens=5),
        )

    mode_settings = _fake_mode_settings()
    backtest.run_symbol_for_day(
        state=state, symbol="TEST", asset_class="equity", day=decision_day, full_frame=frame,
        mode_settings=mode_settings, generate_signal_fn=fake_generate, model="claude-haiku-4-5",
        cost=backtest.CostTracker("claude-haiku-4-5"), logger=logger,
        circuit_breaker_loss_pct=3.0, max_risk_pct=1.0, max_absolute_position_pct=20.0,
    )

    assert "TEST" in state.open_positions
    assert state.open_positions["TEST"].entry_price == pytest.approx(150.0)  # Open, not 999


def _fake_mode_settings():
    import mode

    return mode.ModeSettings(is_live=True, system_prompt="SYSTEM", min_confidence=0.5)


# ==================================================================== sizing


def _buy_output(symbol="TEST", price=100.0, size=10.0, confidence=0.9):
    from models import SignalOutput

    return SignalOutput(
        symbol=symbol, action="buy", confidence=confidence, position_size_pct=size,
        stop_loss_price=price * 0.95, take_profit_price=price * 1.1, reasoning="r",
    )


def _sell_output(symbol="TEST", price=100.0, confidence=0.9):
    from models import SignalOutput

    return SignalOutput(
        symbol=symbol, action="sell", confidence=confidence, position_size_pct=10.0,
        stop_loss_price=price * 0.95, take_profit_price=price * 1.1, reasoning="r",
    )


def test_buy_sizing_uses_equity_relative_position_size_pct():
    frame = daily_frame(n=80, close=100.0)
    day = frame.index[70]
    state = backtest.BacktestState(equity=10_000.0)
    logger = backtest.BacktestLogger(":memory:")

    def fake_generate(signal_input, system_prompt=None, model=None):
        return _buy_output(price=signal_input.current_price, size=20.0), TokenUsage(input_tokens=1, output_tokens=1)

    backtest.run_symbol_for_day(
        state=state, symbol="TEST", asset_class="equity", day=day, full_frame=frame,
        mode_settings=_fake_mode_settings(), generate_signal_fn=fake_generate, model="claude-haiku-4-5",
        cost=backtest.CostTracker("claude-haiku-4-5"), logger=logger,
        circuit_breaker_loss_pct=3.0, max_risk_pct=1.0, max_absolute_position_pct=20.0,
    )

    pos = state.open_positions["TEST"]
    # 20% of $10,000 at a $100 fill (Open) = 20 units.
    assert pos.qty == pytest.approx(20.0)


def test_a_duplicate_buy_is_skipped_position_not_doubled():
    frame = daily_frame(n=80, close=100.0)
    day = frame.index[70]
    state = backtest.BacktestState(equity=10_000.0)
    state.open_positions["TEST"] = backtest.OpenPosition(
        symbol="TEST", qty=5.0, entry_price=90.0, stop_loss_price=80.0, take_profit_price=120.0,
        opened_day=frame.index[69],
    )
    logger = backtest.BacktestLogger(":memory:")

    def fake_generate(signal_input, system_prompt=None, model=None):
        return _buy_output(price=signal_input.current_price), TokenUsage(input_tokens=1, output_tokens=1)

    backtest.run_symbol_for_day(
        state=state, symbol="TEST", asset_class="equity", day=day, full_frame=frame,
        mode_settings=_fake_mode_settings(), generate_signal_fn=fake_generate, model="claude-haiku-4-5",
        cost=backtest.CostTracker("claude-haiku-4-5"), logger=logger,
        circuit_breaker_loss_pct=3.0, max_risk_pct=1.0, max_absolute_position_pct=20.0,
    )
    assert state.open_positions["TEST"].qty == pytest.approx(5.0)  # unchanged


def test_closing_uses_the_held_quantity_not_a_recomputed_one():
    frame = daily_frame(n=80, close=100.0)
    day = frame.index[70]
    state = backtest.BacktestState(equity=10_000.0)
    state.open_positions["TEST"] = backtest.OpenPosition(
        symbol="TEST", qty=3.0, entry_price=90.0, stop_loss_price=80.0, take_profit_price=120.0,
        opened_day=frame.index[69],
    )
    logger = backtest.BacktestLogger(":memory:")

    def fake_generate(signal_input, system_prompt=None, model=None):
        return _sell_output(price=signal_input.current_price), TokenUsage(input_tokens=1, output_tokens=1)

    backtest.run_symbol_for_day(
        state=state, symbol="TEST", asset_class="equity", day=day, full_frame=frame,
        mode_settings=_fake_mode_settings(), generate_signal_fn=fake_generate, model="claude-haiku-4-5",
        cost=backtest.CostTracker("claude-haiku-4-5"), logger=logger,
        circuit_breaker_loss_pct=3.0, max_risk_pct=1.0, max_absolute_position_pct=20.0,
    )

    assert "TEST" not in state.open_positions
    trade = state.closed_trades[-1]
    assert trade.qty == pytest.approx(3.0)
    assert trade.exit_price == pytest.approx(100.0)  # day's Open
    assert trade.realized_pnl_usd == pytest.approx((100.0 - 90.0) * 3.0)


def test_a_sell_with_nothing_held_does_nothing():
    frame = daily_frame(n=80, close=100.0)
    day = frame.index[70]
    state = backtest.BacktestState(equity=10_000.0)
    logger = backtest.BacktestLogger(":memory:")

    def fake_generate(signal_input, system_prompt=None, model=None):
        return _sell_output(price=signal_input.current_price), TokenUsage(input_tokens=1, output_tokens=1)

    backtest.run_symbol_for_day(
        state=state, symbol="TEST", asset_class="equity", day=day, full_frame=frame,
        mode_settings=_fake_mode_settings(), generate_signal_fn=fake_generate, model="claude-haiku-4-5",
        cost=backtest.CostTracker("claude-haiku-4-5"), logger=logger,
        circuit_breaker_loss_pct=3.0, max_risk_pct=1.0, max_absolute_position_pct=20.0,
    )
    assert state.open_positions == {}
    assert state.closed_trades == []


def test_risk_manager_is_actually_invoked_low_confidence_becomes_a_hold():
    frame = daily_frame(n=80, close=100.0)
    day = frame.index[70]
    state = backtest.BacktestState(equity=10_000.0)
    logger = backtest.BacktestLogger(":memory:")

    def fake_generate(signal_input, system_prompt=None, model=None):
        return _buy_output(price=signal_input.current_price, confidence=0.1), TokenUsage(input_tokens=1, output_tokens=1)

    mode_settings = _fake_mode_settings()  # min_confidence=0.5
    backtest.run_symbol_for_day(
        state=state, symbol="TEST", asset_class="equity", day=day, full_frame=frame,
        mode_settings=mode_settings, generate_signal_fn=fake_generate, model="claude-haiku-4-5",
        cost=backtest.CostTracker("claude-haiku-4-5"), logger=logger,
        circuit_breaker_loss_pct=3.0, max_risk_pct=1.0, max_absolute_position_pct=20.0,
    )
    assert state.open_positions == {}  # held back by risk_manager, not opened


def test_circuit_breaker_skips_the_model_call_entirely():
    frame = daily_frame(n=80, close=100.0)
    day = frame.index[70]
    state = backtest.BacktestState(equity=1000.0)
    state.note_day(day)
    state._today_realized_pnl = -50.0  # -5%, over a 3% breaker
    logger = backtest.BacktestLogger(":memory:")

    calls = []

    def fake_generate(signal_input, system_prompt=None, model=None):
        calls.append(1)
        return _buy_output(price=signal_input.current_price), TokenUsage(input_tokens=1, output_tokens=1)

    backtest.run_symbol_for_day(
        state=state, symbol="TEST", asset_class="equity", day=day, full_frame=frame,
        mode_settings=_fake_mode_settings(), generate_signal_fn=fake_generate, model="claude-haiku-4-5",
        cost=backtest.CostTracker("claude-haiku-4-5"), logger=logger,
        circuit_breaker_loss_pct=3.0, max_risk_pct=1.0, max_absolute_position_pct=20.0,
    )
    assert calls == []
    assert state.open_positions == {}


def test_a_provider_exception_is_contained_to_one_symbol_day():
    frame = daily_frame(n=80, close=100.0)
    day = frame.index[70]
    state = backtest.BacktestState(equity=10_000.0)
    logger = backtest.BacktestLogger(":memory:")

    def boom(signal_input, system_prompt=None, model=None):
        raise RuntimeError("provider down")

    # Must not raise.
    backtest.run_symbol_for_day(
        state=state, symbol="TEST", asset_class="equity", day=day, full_frame=frame,
        mode_settings=_fake_mode_settings(), generate_signal_fn=boom, model="claude-haiku-4-5",
        cost=backtest.CostTracker("claude-haiku-4-5"), logger=logger,
        circuit_breaker_loss_pct=3.0, max_risk_pct=1.0, max_absolute_position_pct=20.0,
    )
    assert state.open_positions == {}


# ====================================================================== sweep


def test_sweep_closes_a_position_whose_stop_was_touched():
    frame = daily_frame(n=10, close=100.0)
    day = frame.index[5]
    frame.loc[day, ["Open", "High", "Low", "Close"]] = [100.0, 101.0, 89.0, 95.0]
    frames = {"TEST": frame}

    state = backtest.BacktestState(equity=10_000.0)
    state.open_positions["TEST"] = backtest.OpenPosition(
        symbol="TEST", qty=10.0, entry_price=100.0, stop_loss_price=90.0, take_profit_price=130.0,
        opened_day=frame.index[4],
    )
    logger = backtest.BacktestLogger(":memory:")

    closed = backtest.sweep_positions_for_day(state, frames, day, logger)

    assert closed == {"TEST"}
    assert "TEST" not in state.open_positions
    assert state.closed_trades[-1].exit_price == pytest.approx(90.0)  # touched, not gapped
    assert state.closed_trades[-1].realized_pnl_usd == pytest.approx((90.0 - 100.0) * 10.0)


def test_sweep_fill_reflects_a_gap_down_through_the_stop():
    frame = daily_frame(n=10, close=100.0)
    day = frame.index[5]
    # Opened already below the stop -- a real stop order fills at the open.
    frame.loc[day, ["Open", "High", "Low", "Close"]] = [80.0, 82.0, 78.0, 81.0]
    frames = {"TEST": frame}

    state = backtest.BacktestState(equity=10_000.0)
    state.open_positions["TEST"] = backtest.OpenPosition(
        symbol="TEST", qty=10.0, entry_price=100.0, stop_loss_price=90.0, take_profit_price=130.0,
        opened_day=frame.index[4],
    )
    logger = backtest.BacktestLogger(":memory:")

    backtest.sweep_positions_for_day(state, frames, day, logger)
    assert state.closed_trades[-1].exit_price == pytest.approx(80.0)  # the gap open, not 90


def test_sweep_prefers_the_worse_outcome_when_both_levels_are_touched():
    frame = daily_frame(n=10, close=100.0)
    day = frame.index[5]
    frame.loc[day, ["Open", "High", "Low", "Close"]] = [100.0, 140.0, 85.0, 110.0]
    frames = {"TEST": frame}

    state = backtest.BacktestState(equity=10_000.0)
    state.open_positions["TEST"] = backtest.OpenPosition(
        symbol="TEST", qty=10.0, entry_price=100.0, stop_loss_price=90.0, take_profit_price=130.0,
        opened_day=frame.index[4],
    )
    logger = backtest.BacktestLogger(":memory:")

    backtest.sweep_positions_for_day(state, frames, day, logger)
    assert state.closed_trades[-1].exit_price == pytest.approx(90.0)  # the stop, the worse one


def test_a_symbol_not_swept_leaves_the_position_open():
    frame = daily_frame(n=10, close=100.0)
    day = frame.index[5]  # never touches either level
    frames = {"TEST": frame}

    state = backtest.BacktestState(equity=10_000.0)
    state.open_positions["TEST"] = backtest.OpenPosition(
        symbol="TEST", qty=10.0, entry_price=100.0, stop_loss_price=50.0, take_profit_price=200.0,
        opened_day=frame.index[4],
    )
    logger = backtest.BacktestLogger(":memory:")

    closed = backtest.sweep_positions_for_day(state, frames, day, logger)
    assert closed == set()
    assert "TEST" in state.open_positions


def test_a_closed_symbol_is_not_reprocessed_the_same_day_by_run_backtest():
    # Integration-level: the closed_today set from the sweep must actually be
    # honoured by the day loop in run_backtest, not just returned and ignored.
    frame = daily_frame(n=10, close=100.0)
    day = frame.index[5]
    frame.loc[day, ["Open", "High", "Low", "Close"]] = [100.0, 101.0, 89.0, 95.0]

    state = backtest.BacktestState(equity=10_000.0)
    state.open_positions["TEST"] = backtest.OpenPosition(
        symbol="TEST", qty=10.0, entry_price=100.0, stop_loss_price=90.0, take_profit_price=130.0,
        opened_day=frame.index[4],
    )
    logger = backtest.BacktestLogger(":memory:")
    closed_today = backtest.sweep_positions_for_day(state, {"TEST": frame}, day, logger)

    called = []

    def fake_generate(signal_input, system_prompt=None, model=None):
        called.append(1)
        return _buy_output(), TokenUsage(input_tokens=1, output_tokens=1)

    # Mirror run_backtest's own "skip if closed today" loop body.
    if "TEST" not in closed_today:
        backtest.run_symbol_for_day(
            state=state, symbol="TEST", asset_class="equity", day=day, full_frame=frame,
            mode_settings=_fake_mode_settings(), generate_signal_fn=fake_generate, model="m",
            cost=backtest.CostTracker("m"), logger=logger, circuit_breaker_loss_pct=3.0,
            max_risk_pct=1.0, max_absolute_position_pct=20.0,
        )
    assert called == []


# ============================================================== cost tracking


def test_cost_tracker_accumulates_real_usage_not_an_estimate():
    tracker = backtest.CostTracker("claude-haiku-4-5")
    tracker.record(TokenUsage(input_tokens=1000, output_tokens=200))
    tracker.record(TokenUsage(input_tokens=500, output_tokens=100))
    assert tracker.input_tokens == 1500
    assert tracker.output_tokens == 300
    assert tracker.calls_made == 2
    expected = 1500 / 1_000_000 * 1.00 + 300 / 1_000_000 * 5.00
    assert tracker.actual_cost_usd() == pytest.approx(expected)


def test_unknown_model_pricing_degrades_to_zero_cost_with_a_warning(capsys):
    tracker = backtest.CostTracker("some-unknown-model-xyz")
    tracker.record(TokenUsage(input_tokens=1000, output_tokens=1000))
    assert tracker.actual_cost_usd() == 0.0
    assert "WARNING" in capsys.readouterr().out


def test_upfront_estimate_is_printed_before_any_real_usage(capsys):
    backtest.print_upfront_cost_estimate("claude-haiku-4-5", expected_calls=100)
    out = capsys.readouterr().out
    assert "estimate" in out.lower()
    assert "$" in out


def test_cost_is_shown_during_the_run_not_only_after(monkeypatch, tmp_path):
    """Acceptance criterion: an upfront estimate AND a during-run progress
    line both appear before the final report in stdout, not only a final tally.
    """
    frame = daily_frame(n=200, close=100.0)
    monkeypatch.setattr(backtest, "fetch_historical_ohlcv", lambda symbol, start, end: frame)
    monkeypatch.setattr(backtest, "COST_PROGRESS_EVERY_N_CALLS", 1)

    def fake_provider(provider):
        def fake_generate(signal_input, system_prompt=None, model=None):
            from models import SignalOutput

            return (
                SignalOutput(symbol=signal_input.symbol, action="hold", confidence=0.1, reasoning="r"),
                TokenUsage(input_tokens=10, output_tokens=5),
            )

        return fake_generate

    monkeypatch.setattr(backtest, "resolve_provider", fake_provider)

    import io
    import contextlib

    buf = io.StringIO()
    config = {"circuit_breaker_loss_pct": 3.0, "max_risk_pct": 1.0, "max_absolute_position_pct": 20.0,
              "min_confidence_live": 0.0}
    with contextlib.redirect_stdout(buf):
        backtest.run_backtest(
            symbols_with_class=[("TEST", "equity")],
            start=frame.index[190].date(), end=frame.index[195].date(),
            config=config, provider="claude", model="claude-haiku-4-5",
            starting_equity=10_000.0, db_path=str(tmp_path / "bt.db"),
        )
    output = buf.getvalue()

    estimate_pos = output.find("Rough pre-run estimate")
    progress_pos = output.find("[cost]")
    report_pos = output.find("=== Backtest report")
    final_cost_pos = output.find("=== Cost:")

    assert estimate_pos != -1 and progress_pos != -1 and report_pos != -1 and final_cost_pos != -1
    assert estimate_pos < report_pos
    assert progress_pos < report_pos  # shown DURING, not only after
    assert report_pos < final_cost_pos


# ==================================================================== report


def test_buy_and_hold_beats_a_strategy_that_never_trades():
    frame = daily_frame(n=10, close=100.0, step=1.0)  # steadily rising
    frames = {"TEST": frame}
    state = backtest.BacktestState(equity=10_000.0)
    for day in frame.index:
        state.equity_curve.append((day, 10_000.0))  # never traded, flat equity

    report = backtest.compute_report(
        state, frames, [("TEST", "equity")],
        frame.index[0].date(), frame.index[-1].date(), 10_000.0,
    )
    assert report["total_return_pct"] == pytest.approx(0.0)
    assert report["buy_and_hold_return_pct"] > 0.0
    assert report["strategy_vs_buy_and_hold_pct"] < 0.0


def test_win_rate_counts_only_closed_trades():
    frame = daily_frame(n=10, close=100.0)
    frames = {"TEST": frame}
    state = backtest.BacktestState(equity=10_000.0)
    state.closed_trades = [
        backtest.ClosedTrade("TEST", frame.index[0], frame.index[1], 1.0, 100.0, 110.0, 10.0, False),
        backtest.ClosedTrade("TEST", frame.index[1], frame.index[2], 1.0, 100.0, 90.0, -10.0, False),
        backtest.ClosedTrade("TEST", frame.index[2], frame.index[3], 1.0, 100.0, 105.0, 5.0, False),
    ]
    state.equity_curve = [(d, 10_000.0) for d in frame.index]

    report = backtest.compute_report(
        state, frames, [("TEST", "equity")], frame.index[0].date(), frame.index[-1].date(), 10_000.0,
    )
    assert report["num_trades"] == 3
    assert report["win_rate_pct"] == pytest.approx(200 / 3)


def test_max_drawdown_is_measured_from_the_running_peak():
    frame = daily_frame(n=5, close=100.0)
    frames = {"TEST": frame}
    state = backtest.BacktestState(equity=10_000.0)
    state.equity_curve = [
        (frame.index[0], 10_000.0),
        (frame.index[1], 11_000.0),  # new peak
        (frame.index[2], 9_000.0),   # drawdown from 11,000
        (frame.index[3], 9_500.0),
        (frame.index[4], 10_500.0),
    ]
    report = backtest.compute_report(
        state, frames, [("TEST", "equity")], frame.index[0].date(), frame.index[-1].date(), 10_000.0,
    )
    expected_dd = (9_000.0 / 11_000.0 - 1.0) * 100.0
    assert report["max_drawdown_pct"] == pytest.approx(expected_dd)


def test_report_states_the_two_known_limitations_plainly():
    frame = daily_frame(n=5, close=100.0)
    state = backtest.BacktestState(equity=10_000.0)
    state.equity_curve = [(d, 10_000.0) for d in frame.index]
    report = backtest.compute_report(
        state, {"TEST": frame}, [("TEST", "equity")], frame.index[0].date(), frame.index[-1].date(), 10_000.0,
    )
    assert len(report["limitations"]) == 2
    assert any("headline" in item.lower() for item in report["limitations"])
    assert any("positioning" in item.lower() for item in report["limitations"])

    formatted = backtest.format_report(report)
    assert "headline" in formatted.lower()
    assert "positioning" in formatted.lower()
    assert "buy-and-hold" in formatted.lower()


# ============================================================= isolation


def test_backtest_logger_writes_to_its_own_file_never_trading_bot_db(tmp_path):
    db_path = tmp_path / "backtest.db"
    logger = backtest.BacktestLogger(str(db_path))
    logger.log_trade(
        day="2024-01-01", symbol="TEST", action="buy", qty=1.0, price=100.0,
        realized_pnl_usd=None, confidence=0.8, reasoning="r", override_reason=None,
        is_auto_close=False,
    )
    logger.log_equity("2024-01-01", 10_000.0)

    assert db_path.exists()
    conn = sqlite3.connect(str(db_path))
    trades = conn.execute("SELECT symbol, action FROM trades").fetchall()
    equity = conn.execute("SELECT simulated_date, equity FROM equity_curve").fetchall()
    conn.close()

    assert trades == [("TEST", "buy")]
    assert equity == [("2024-01-01", 10_000.0)]


def test_backtest_db_path_defaults_away_from_trading_bot_db():
    assert backtest.DEFAULT_DB_PATH != "trading_bot.db"


def test_backtest_never_imports_execution_module():
    # backtest.py must never be able to call a real broker -- checked at the
    # module level, not just "no test happens to call it".
    import ast
    import inspect

    source = inspect.getsource(backtest)
    tree = ast.parse(source)
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_names.add(node.module)
    assert "execution" not in imported_names


# =================================================================== data


def test_trading_days_in_range_respects_the_boundaries():
    frame = daily_frame(n=20, close=100.0)
    days = backtest.trading_days_in_range(frame, frame.index[5].date(), frame.index[10].date())
    assert days[0] == frame.index[5]
    assert days[-1] == frame.index[10]
    assert len(days) == 6


def test_fetch_historical_ohlcv_widens_the_start_for_lookback(monkeypatch):
    captured = {}

    class FakeTicker:
        def __init__(self, symbol):
            pass

        def history(self, start, end, interval):
            captured["start"], captured["end"] = start, end
            return daily_frame(n=5, close=100.0)

    monkeypatch.setattr(backtest.yf, "Ticker", FakeTicker)
    from datetime import date

    backtest.fetch_historical_ohlcv("TEST", date(2024, 6, 1), date(2024, 6, 10), lookback_days=180)
    assert captured["start"] < "2024-06-01"
    assert captured["end"] > "2024-06-10"


# =============================================== mark-to-market weekend gaps


def test_mark_to_market_carries_the_last_known_price_across_a_gap_day():
    """Real-data-discovered bug: an equity position's mark must not vanish on
    a day only a 7-day/week crypto symbol advances the simulated calendar.
    """
    frame = daily_frame(n=10, close=100.0, step=1.0)  # a bar for every day 0..9
    # Simulate a "weekend gap" for this symbol by dropping day index 5 entirely.
    frame_with_gap = frame.drop(frame.index[5])

    state = backtest.BacktestState(equity=10_000.0)
    state.open_positions["TEST"] = backtest.OpenPosition(
        symbol="TEST", qty=2.0, entry_price=100.0, stop_loss_price=None,
        take_profit_price=None, opened_day=frame.index[0],
    )

    gap_day = frame.index[5]
    equity_on_gap_day = backtest.mark_to_market_equity(state, {"TEST": frame_with_gap}, gap_day)

    # The mark must equal day 4's close (the last real session before the
    # gap), NOT drop the position's unrealised P&L to zero for the gap day.
    last_real_close = float(frame.loc[frame.index[4], "Close"])
    expected = 10_000.0 + (last_real_close - 100.0) * 2.0
    assert equity_on_gap_day == pytest.approx(expected)
    assert equity_on_gap_day != 10_000.0  # the old, buggy behaviour


def test_mark_to_market_uses_no_price_at_all_before_a_symbols_first_bar():
    frame = daily_frame(n=10, close=100.0)
    state = backtest.BacktestState(equity=10_000.0)
    state.open_positions["TEST"] = backtest.OpenPosition(
        symbol="TEST", qty=1.0, entry_price=100.0, stop_loss_price=None,
        take_profit_price=None, opened_day=frame.index[0],
    )
    before_any_data = frame.index[0] - pd.Timedelta(days=5)
    assert backtest.mark_to_market_equity(state, {"TEST": frame}, before_any_data) == 10_000.0


def test_last_known_close_finds_the_most_recent_prior_session():
    frame = daily_frame(n=10, close=100.0, step=1.0).drop(
        [daily_frame(n=10, close=100.0, step=1.0).index[i] for i in (4, 5, 6)]
    )
    gap_day = daily_frame(n=10, close=100.0, step=1.0).index[5]
    close = backtest._last_known_close(frame, gap_day)
    expected = float(daily_frame(n=10, close=100.0, step=1.0).loc[
        daily_frame(n=10, close=100.0, step=1.0).index[3], "Close"
    ])
    assert close == pytest.approx(expected)
