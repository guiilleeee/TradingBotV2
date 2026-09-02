import csv
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from models import ExecutionResult, SignalInput, SignalOutput, TechnicalIndicators, TradeSignal


def signal_input(symbol="AAPL", price=100.0, equity=1000.0):
    return SignalInput(
        symbol=symbol,
        asset_class="equity",
        current_price=price,
        account_equity_usd=equity,
        technical_indicators=TechnicalIndicators(
            rsi_14=55.0, sma_20=101.0, sma_50=99.0, price_change_pct=1.0, volume_change_pct=2.0
        ),
        recent_headlines=["headline"],
    )


def trade_signal(symbol="AAPL", action="buy", raw_action="buy", override=None):
    return TradeSignal(
        symbol=symbol,
        action=action,
        confidence=0.8,
        position_size_pct=20.0 if action != "hold" else 0.0,
        stop_loss_price=95.0 if action != "hold" else None,
        take_profit_price=115.0 if action != "hold" else None,
        reasoning="analisi en catala",
        override_reason=override,
        raw_action=raw_action,
    )


# ------------------------------------------------------------------- signals


def test_log_signal_round_trips_every_blob(tmp_logger, read_signals):
    raw = SignalOutput(
        symbol="AAPL", action="buy", confidence=0.8, position_size_pct=5.0,
        stop_loss_price=95.0, take_profit_price=115.0, reasoning="analisi",
    )
    tmp_logger.log_signal(
        "AAPL",
        signal_input(),
        raw,
        trade_signal(override="clamped"),
        ExecutionResult(status="dry_run", qty=2.0, fill_price=100.0),
    )

    row = read_signals()[0]
    assert row["symbol"] == "AAPL"
    assert row["override_reason"] == "clamped"
    assert row["signal_input"]["account_equity_usd"] == 1000.0
    assert row["raw_output"]["action"] == "buy"
    assert row["final_signal"]["raw_action"] == "buy"
    assert row["execution_result"]["qty"] == 2.0
    # UTC, and parseable.
    assert datetime.fromisoformat(row["timestamp"]).tzinfo is not None


def test_timestamps_are_utc(tmp_logger, read_signals):
    tmp_logger.log_signal("AAPL", signal_input(), None, trade_signal())
    stamp = datetime.fromisoformat(read_signals()[0]["timestamp"])
    assert stamp.utcoffset() == timedelta(0)


def test_auto_close_row_carries_equity_and_reasoning(tmp_logger, read_signals):
    tmp_logger.log_auto_close_signal(
        symbol="AAPL", reason="Take-profit assolit", price=110.0, qty=2.0, pnl=20.0, equity=1020.0
    )
    row = read_signals()[0]

    # Both of these were missing in an earlier build: the dashboard's
    # "Patrimoni total" went blank and reasoning rendered as "undefined".
    assert row["signal_input"]["account_equity_usd"] == 1020.0
    assert row["final_signal"]["reasoning"] == "Take-profit assolit"
    assert row["raw_output"]["reasoning"] == "Take-profit assolit"
    assert row["final_signal"]["action"] == "sell"
    assert row["final_signal"]["raw_action"] == "sell"
    assert row["execution_result"]["realized_pnl_usd"] == 20.0
    assert row["execution_result"]["qty"] == 2.0
    assert row["override_reason"] == "automatic exit"


def test_auto_close_row_has_the_same_keys_a_real_row_does(tmp_logger, read_signals):
    tmp_logger.log_signal("AAPL", signal_input(), None, trade_signal())
    tmp_logger.log_auto_close_signal("AAPL", "Stop-loss activat", 92.0, 2.0, -16.0, 984.0)
    real, synthetic = read_signals()

    for key in ("symbol", "action", "confidence", "position_size_pct", "reasoning", "raw_action"):
        assert key in synthetic["final_signal"], key
        assert key in real["final_signal"], key
    for key in ("symbol", "current_price", "account_equity_usd", "recent_headlines"):
        assert key in synthetic["signal_input"], key
        assert key in real["signal_input"], key


# ----------------------------------------------------------------------- pnl


def test_realized_pnl_accumulates(tmp_logger):
    assert tmp_logger.get_all_time_realized_pnl() == 0.0
    tmp_logger.record_pnl("AAPL", 25.0)
    tmp_logger.record_pnl("MSFT", -10.0)
    assert tmp_logger.get_all_time_realized_pnl() == pytest.approx(15.0)


def test_today_loss_pct_is_relative_to_equity(tmp_logger):
    tmp_logger.record_pnl("AAPL", -30.0)
    assert tmp_logger.get_today_realized_loss_pct(1000.0) == pytest.approx(-3.0)


def test_today_loss_pct_uses_the_utc_day_not_the_local_one(tmp_logger):
    """Yesterday's loss must not count against today's circuit breaker."""
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    conn = sqlite3.connect(tmp_logger.db_path)
    conn.execute(
        "INSERT INTO pnl (timestamp, symbol, realized_pnl_usd) VALUES (?, ?, ?)",
        (yesterday, "AAPL", -500.0),
    )
    conn.commit()
    conn.close()

    assert tmp_logger.get_today_realized_loss_pct(1000.0) == pytest.approx(0.0)
    # ...but it still counts toward all-time equity.
    assert tmp_logger.get_all_time_realized_pnl() == pytest.approx(-500.0)


def test_today_loss_pct_is_zero_with_no_equity(tmp_logger):
    tmp_logger.record_pnl("AAPL", -30.0)
    assert tmp_logger.get_today_realized_loss_pct(0.0) == 0.0


# -------------------------------------------------------------- last buy price


def test_last_buy_price_prefers_the_fill(tmp_logger):
    tmp_logger.log_signal(
        "BTC-USD", signal_input("BTC-USD", price=50000.0), None,
        trade_signal("BTC-USD", "buy"),
        ExecutionResult(status="success", qty=0.01, fill_price=50100.0),
    )
    assert tmp_logger.get_last_buy_price("BTC-USD") == pytest.approx(50100.0)


def test_last_buy_price_falls_back_to_the_signal_price(tmp_logger):
    tmp_logger.log_signal(
        "BTC-USD", signal_input("BTC-USD", price=50000.0), None,
        trade_signal("BTC-USD", "buy"),
        ExecutionResult(status="dry_run", qty=0.01),
    )
    assert tmp_logger.get_last_buy_price("BTC-USD") == pytest.approx(50000.0)


def test_last_buy_price_ignores_holds_sells_and_failures(tmp_logger):
    tmp_logger.log_signal("BTC-USD", signal_input("BTC-USD", 40000.0), None,
                          trade_signal("BTC-USD", "hold", "hold"))
    tmp_logger.log_signal("BTC-USD", signal_input("BTC-USD", 41000.0), None,
                          trade_signal("BTC-USD", "buy"),
                          ExecutionResult(status="error", message="rejected"))
    tmp_logger.log_signal("BTC-USD", signal_input("BTC-USD", 42000.0), None,
                          trade_signal("BTC-USD", "buy"),
                          ExecutionResult(status="success", qty=0.01, fill_price=42500.0))
    tmp_logger.log_signal("BTC-USD", signal_input("BTC-USD", 43000.0), None,
                          trade_signal("BTC-USD", "sell", "sell"),
                          ExecutionResult(status="success", qty=0.01, fill_price=43500.0))

    # Newest successful BUY wins; the later sell and the rejected buy are ignored.
    assert tmp_logger.get_last_buy_price("BTC-USD") == pytest.approx(42500.0)


def test_last_buy_price_is_none_when_there_is_no_buy(tmp_logger):
    assert tmp_logger.get_last_buy_price("BTC-USD") is None


# ------------------------------------------------------------- simulated ledger


def test_open_position_is_an_upsert(tmp_logger):
    tmp_logger.open_simulated_position("AAPL", 2.0, 100.0, 95.0, 110.0)
    tmp_logger.open_simulated_position("AAPL", 5.0, 102.0, 97.0, 120.0)

    rows = tmp_logger.get_all_simulated_positions()
    assert len(rows) == 1
    assert rows[0]["qty"] == 5.0
    assert rows[0]["avg_entry_price"] == 102.0
    assert rows[0]["stop_loss_price"] == 97.0


def test_get_simulated_position_returns_a_validated_model(tmp_logger):
    tmp_logger.open_simulated_position("AAPL", 2.0, 100.0)
    position = tmp_logger.get_simulated_position("AAPL")
    assert position.qty == 2.0
    assert position.avg_entry_price == 100.0


def test_close_removes_the_position(tmp_logger):
    tmp_logger.open_simulated_position("AAPL", 2.0, 100.0)
    tmp_logger.close_simulated_position("AAPL")
    assert tmp_logger.get_simulated_position("AAPL") is None
    assert tmp_logger.get_all_simulated_positions() == []


def test_closing_a_position_that_is_not_there_is_harmless(tmp_logger):
    tmp_logger.close_simulated_position("NOPE")


def test_unknown_symbol_has_no_position(tmp_logger):
    assert tmp_logger.get_simulated_position("NOPE") is None


# ------------------------------------------------------------------- csv export


def test_csv_export_flattens_the_blobs(tmp_logger, tmp_path):
    tmp_logger.log_signal(
        "AAPL", signal_input(), None, trade_signal(override="clamped"),
        ExecutionResult(status="dry_run", qty=2.0, fill_price=100.0, realized_pnl_usd=None),
    )
    tmp_logger.log_auto_close_signal("AAPL", "Take-profit assolit", 110.0, 2.0, 20.0, 1020.0)

    path = tmp_path / "signals.csv"
    count = tmp_logger.export_signals_csv(str(path))
    assert count == 2

    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    # Oldest first, so the dashboard's "newest row" is genuinely the last one.
    assert rows[0]["action"] == "buy"
    assert rows[1]["action"] == "sell"
    assert rows[1]["account_equity_usd"] == "1020.0"
    assert rows[1]["realized_pnl_usd"] == "20.0"
    assert rows[0]["reasoning"] == "analisi en catala"
    assert rows[0]["raw_action"] == "buy"


def test_csv_export_handles_an_empty_database(tmp_logger, tmp_path):
    path = tmp_path / "signals.csv"
    assert tmp_logger.export_signals_csv(str(path)) == 0
    with open(path, newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == []


def test_schema_has_exactly_the_three_tables(tmp_logger):
    conn = sqlite3.connect(tmp_logger.db_path)
    names = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }
    conn.close()
    assert names == {"signals", "pnl", "simulated_positions"}
