import pytest
from pydantic import ValidationError

from models import (
    ExecutionResult,
    ExistingPosition,
    SignalInput,
    SignalOutput,
    TechnicalIndicators,
    TradeSignal,
    parse_signal_output,
)


def indicators(**overrides):
    base = dict(rsi_14=55.0, sma_20=101.0, sma_50=99.0, price_change_pct=1.2, volume_change_pct=-3.4)
    base.update(overrides)
    return TechnicalIndicators(**base)


# ------------------------------------------------------------- ExecutionResult


def test_filled_order_must_report_qty():
    # The simulated ledger rebuilds positions from qty. A fill without one is the
    # bug that broke position tracking in the previous build.
    with pytest.raises(ValidationError, match="positive qty"):
        ExecutionResult(status="success", fill_price=100.0)
    with pytest.raises(ValidationError, match="positive qty"):
        ExecutionResult(status="dry_run", fill_price=100.0, qty=0.0)


def test_skipped_and_error_results_need_no_qty():
    assert ExecutionResult(status="skipped", message="nothing to sell").qty is None
    assert ExecutionResult(status="error", message="broker down").qty is None


def test_filled_order_with_qty_is_accepted():
    result = ExecutionResult(status="success", qty=1.5, fill_price=100.0, realized_pnl_usd=-4.0)
    assert result.qty == 1.5


# ---------------------------------------------------------------- indicators


@pytest.mark.parametrize("field,value", [("rsi_14", -1), ("rsi_14", 101), ("sma_20", 0), ("sma_50", -5)])
def test_indicator_bounds_are_enforced(field, value):
    with pytest.raises(ValidationError):
        indicators(**{field: value})


def test_existing_position_requires_a_real_entry_price():
    with pytest.raises(ValidationError):
        ExistingPosition(qty=1.0, avg_entry_price=0.0)


# --------------------------------------------------------------- SignalOutput


def test_action_is_lowercased_and_stripped():
    assert SignalOutput(symbol="AAPL", action="  BUY ", confidence=0.5,
                        position_size_pct=1.0, stop_loss_price=1.0,
                        take_profit_price=2.0).action == "buy"


def test_hold_forces_levels_and_size_to_nothing():
    out = SignalOutput(
        symbol="AAPL", action="hold", confidence=0.9, position_size_pct=42.0,
        stop_loss_price=90.0, take_profit_price=120.0,
    )
    assert out.stop_loss_price is None
    assert out.take_profit_price is None
    assert out.position_size_pct == 0.0


def test_unknown_action_is_rejected():
    with pytest.raises(ValidationError):
        SignalOutput(symbol="AAPL", action="short", confidence=0.5)


def test_confidence_must_be_a_probability():
    with pytest.raises(ValidationError):
        SignalOutput(symbol="AAPL", action="hold", confidence=1.4)


# ----------------------------------------------------------------- TradeSignal


def test_trade_signal_keeps_the_models_original_action():
    signal = TradeSignal(
        symbol="AAPL", action="hold", confidence=0.3, position_size_pct=0.0,
        stop_loss_price=None, take_profit_price=None, reasoning="r",
        override_reason="confidence below minimum", raw_action="buy",
    )
    assert signal.action == "hold"
    assert signal.raw_action == "buy"


def test_trade_signal_inherits_the_hold_zeroing():
    signal = TradeSignal(
        symbol="AAPL", action="hold", confidence=0.3, position_size_pct=15.0,
        stop_loss_price=90.0, take_profit_price=110.0, reasoning="r", raw_action="buy",
    )
    assert (signal.position_size_pct, signal.stop_loss_price, signal.take_profit_price) == (0.0, None, None)


# ------------------------------------------------------------------- SignalInput


def test_signal_input_rejects_impossible_numbers():
    with pytest.raises(ValidationError):
        SignalInput(symbol="AAPL", asset_class="equity", current_price=0.0,
                    account_equity_usd=1000.0, technical_indicators=indicators())
    with pytest.raises(ValidationError):
        SignalInput(symbol="AAPL", asset_class="equity", current_price=10.0,
                    account_equity_usd=0.0, technical_indicators=indicators())


def test_signal_input_defaults_to_no_headlines_and_no_position():
    si = SignalInput(symbol="AAPL", asset_class="equity", current_price=10.0,
                     account_equity_usd=1000.0, technical_indicators=indicators())
    assert si.recent_headlines == []
    assert si.existing_position is None


# --------------------------------------------------------- parse_signal_output

GOOD = ('{"symbol":"XXX","action":"BUY","confidence":0.7,"position_size_pct":5,'
        '"stop_loss_price":90,"take_profit_price":120,"reasoning":"analisi"}')


def test_parser_overwrites_the_symbol_with_the_one_we_asked_about():
    # A model echoing the wrong ticker must never route an order elsewhere.
    assert parse_signal_output(GOOD, "AAPL").symbol == "AAPL"


def test_parser_strips_a_markdown_fence():
    assert parse_signal_output(f"```json\n{GOOD}\n```", "AAPL").action == "buy"


def test_parser_recovers_json_wrapped_in_prose():
    assert parse_signal_output(f"Here you go:\n{GOOD}\nHope that helps.", "AAPL").action == "buy"


@pytest.mark.parametrize("text", ["", "   ", "no json at all", "{not json}"])
def test_parser_raises_clearly_on_junk(text):
    with pytest.raises(ValueError, match="AAPL"):
        parse_signal_output(text, "AAPL")
