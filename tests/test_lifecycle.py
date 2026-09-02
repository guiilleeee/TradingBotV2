"""End-to-end simulated position lifecycle against a throwaway database.

Open a position, move the price across an exit level, run the sweep, and check
every downstream consequence: the ledger row is gone, the P&L is booked, the
all-time total reflects it, and the synthetic signal row carries a real equity
value and real reasoning text.
"""

import pandas as pd
import pytest

import data_fetcher
import main

CONFIG = {
    "symbols": [
        {"symbol": "AAPL", "asset_class": "equity"},
        {"symbol": "MSFT", "asset_class": "equity"},
    ],
    "fallback_equity_usd": 1000.0,
}
START_EQUITY = 1000.0


def patch_prices(monkeypatch, prices):
    """Feed the sweep fixed prices instead of hitting the network."""

    def fake_fetch(symbol, period=data_fetcher.DEFAULT_PERIOD, interval="1d"):
        if symbol not in prices:
            raise ValueError(f"no fixture price for {symbol}")
        return pd.DataFrame({"Close": [prices[symbol]], "Volume": [1.0]})

    monkeypatch.setattr(data_fetcher, "fetch_ohlcv", fake_fetch)
    monkeypatch.setattr(data_fetcher, "latest_price", lambda df: float(df["Close"].iloc[-1]))


def run_sweep_and_log(bot_logger, monkeypatch, prices):
    """Mirror main.run_cycle's simulation ordering: sweep, then equity, then log."""
    patch_prices(monkeypatch, prices)
    sweep = main.sweep_open_positions(bot_logger, CONFIG, is_live=False, equity_hint=START_EQUITY)
    equity = START_EQUITY + bot_logger.get_all_time_realized_pnl() + sweep.unrealized_pnl
    for closure in sweep.closures:
        bot_logger.log_auto_close_signal(
            symbol=closure.symbol,
            reason=closure.reason,
            price=closure.price,
            qty=closure.qty,
            pnl=closure.pnl,
            equity=equity,
        )
    return sweep, equity


def test_take_profit_closes_and_books_pnl(tmp_logger, read_signals, monkeypatch):
    tmp_logger.open_simulated_position(
        "AAPL", qty=2.0, avg_entry_price=100.0, stop_loss_price=95.0, take_profit_price=110.0
    )

    sweep, equity = run_sweep_and_log(tmp_logger, monkeypatch, {"AAPL": 112.0})

    assert sweep.closed_symbols == {"AAPL"}
    assert tmp_logger.get_simulated_position("AAPL") is None
    assert tmp_logger.get_all_simulated_positions() == []

    expected_pnl = (112.0 - 100.0) * 2.0
    assert sweep.closures[0].pnl == pytest.approx(expected_pnl)
    assert tmp_logger.get_all_time_realized_pnl() == pytest.approx(expected_pnl)

    # A position closed in this sweep must not also count as still-open unrealised.
    assert sweep.unrealized_pnl == pytest.approx(0.0)
    assert equity == pytest.approx(START_EQUITY + expected_pnl)

    rows = read_signals()
    assert len(rows) == 1
    row = rows[0]
    assert row["symbol"] == "AAPL"
    assert row["signal_input"]["account_equity_usd"] == pytest.approx(equity)
    assert row["signal_input"]["account_equity_usd"] > 0
    assert row["final_signal"]["reasoning"]
    assert "Take-profit" in row["final_signal"]["reasoning"]
    assert row["final_signal"]["action"] == "sell"
    assert row["execution_result"]["realized_pnl_usd"] == pytest.approx(expected_pnl)
    assert row["execution_result"]["qty"] == pytest.approx(2.0)


def test_stop_loss_closes_and_books_loss(tmp_logger, read_signals, monkeypatch):
    tmp_logger.open_simulated_position(
        "AAPL", qty=2.0, avg_entry_price=100.0, stop_loss_price=95.0, take_profit_price=110.0
    )

    sweep, equity = run_sweep_and_log(tmp_logger, monkeypatch, {"AAPL": 92.0})

    assert sweep.closed_symbols == {"AAPL"}
    assert tmp_logger.get_simulated_position("AAPL") is None

    expected_pnl = (92.0 - 100.0) * 2.0
    assert expected_pnl < 0
    assert tmp_logger.get_all_time_realized_pnl() == pytest.approx(expected_pnl)
    assert sweep.unrealized_pnl == pytest.approx(0.0)
    assert equity == pytest.approx(START_EQUITY + expected_pnl)

    row = read_signals()[0]
    assert row["signal_input"]["account_equity_usd"] == pytest.approx(equity)
    assert "Stop-loss" in row["final_signal"]["reasoning"]
    assert row["execution_result"]["realized_pnl_usd"] == pytest.approx(expected_pnl)


def test_booked_loss_feeds_the_circuit_breaker(tmp_logger, monkeypatch):
    """The whole reason record_pnl has to actually be called."""
    tmp_logger.open_simulated_position(
        "AAPL", qty=2.0, avg_entry_price=100.0, stop_loss_price=95.0, take_profit_price=110.0
    )
    run_sweep_and_log(tmp_logger, monkeypatch, {"AAPL": 92.0})

    loss_pct = tmp_logger.get_today_realized_loss_pct(START_EQUITY)
    assert loss_pct == pytest.approx(-1.6)  # -16 USD on 1000 USD equity
    assert loss_pct < 0


def test_untouched_position_stays_open_as_unrealized(tmp_logger, read_signals, monkeypatch):
    tmp_logger.open_simulated_position(
        "MSFT", qty=3.0, avg_entry_price=200.0, stop_loss_price=180.0, take_profit_price=240.0
    )

    sweep, equity = run_sweep_and_log(tmp_logger, monkeypatch, {"MSFT": 210.0})

    assert sweep.closed_symbols == set()
    assert sweep.closures == []
    assert tmp_logger.get_simulated_position("MSFT") is not None
    assert sweep.unrealized_pnl == pytest.approx((210.0 - 200.0) * 3.0)
    assert tmp_logger.get_all_time_realized_pnl() == pytest.approx(0.0)
    assert equity == pytest.approx(START_EQUITY + 30.0)
    assert read_signals() == []


def test_no_double_counting_when_one_closes_and_one_stays_open(tmp_logger, monkeypatch):
    tmp_logger.open_simulated_position(
        "AAPL", qty=2.0, avg_entry_price=100.0, stop_loss_price=95.0, take_profit_price=110.0
    )
    tmp_logger.open_simulated_position(
        "MSFT", qty=3.0, avg_entry_price=200.0, stop_loss_price=180.0, take_profit_price=240.0
    )

    sweep, equity = run_sweep_and_log(tmp_logger, monkeypatch, {"AAPL": 115.0, "MSFT": 210.0})

    realized = (115.0 - 100.0) * 2.0  # 30, booked
    unrealized = (210.0 - 200.0) * 3.0  # 30, still open

    assert sweep.closed_symbols == {"AAPL"}
    assert tmp_logger.get_all_time_realized_pnl() == pytest.approx(realized)
    assert sweep.unrealized_pnl == pytest.approx(unrealized)
    # AAPL contributes exactly once, through the realised side.
    assert equity == pytest.approx(START_EQUITY + realized + unrealized)


def test_gap_through_both_levels_prefers_the_stop(tmp_logger, monkeypatch):
    # A single bar that crosses both levels is ambiguous; assume the worse fill.
    tmp_logger.open_simulated_position(
        "AAPL", qty=1.0, avg_entry_price=100.0, stop_loss_price=120.0, take_profit_price=110.0
    )
    sweep, _ = run_sweep_and_log(tmp_logger, monkeypatch, {"AAPL": 115.0})
    assert "Stop-loss" in sweep.closures[0].reason


def test_unavailable_price_leaves_the_position_alone(tmp_logger, monkeypatch):
    tmp_logger.open_simulated_position(
        "AAPL", qty=2.0, avg_entry_price=100.0, stop_loss_price=95.0, take_profit_price=110.0
    )
    sweep, _ = run_sweep_and_log(tmp_logger, monkeypatch, {})  # no fixture price at all

    assert sweep.closed_symbols == set()
    assert tmp_logger.get_simulated_position("AAPL") is not None
    assert tmp_logger.get_all_time_realized_pnl() == pytest.approx(0.0)
