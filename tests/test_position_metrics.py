"""position_metrics.py: holdings-list data for the dashboard.

Network (yfinance) is always stubbed -- these test the shape and the graceful
degradation, not real market data.
"""

import json

import pandas as pd
import pytest

import data_fetcher
import position_metrics
from logger import BotLogger
from models import ExistingPosition


def infer_asset_class(symbol, config):
    return "crypto" if symbol.upper().endswith("-USD") else "equity"


def fake_history(last_close, days=365 * 7, start_close=None):
    """A daily-bar frame running back `days` days, linearly interpolated so every
    return window in RETURN_WINDOWS resolves to a distinct, checkable number.
    """
    start_close = start_close if start_close is not None else last_close
    idx = pd.date_range(end=pd.Timestamp.now("UTC").normalize(), periods=days, freq="D")
    closes = pd.Series(
        [start_close + (last_close - start_close) * i / (days - 1) for i in range(days)],
        index=idx,
    )
    return pd.DataFrame({"Close": closes, "Volume": [1_000_000.0] * days})


# ------------------------------------------------------------- _return_since


def test_return_since_computes_pct_change_to_the_nearest_prior_bar():
    df = fake_history(last_close=110.0, days=30, start_close=100.0)
    pct = position_metrics._return_since(df, 7)
    assert pct is not None
    # Not exact (linear interpolation isn't the real market), just sane in sign/scale.
    assert 0 < pct < 15


def test_return_since_is_none_when_history_does_not_reach_back_that_far():
    df = fake_history(last_close=110.0, days=10, start_close=100.0)
    assert position_metrics._return_since(df, 365 * 6) is None


# --------------------------------------------------------- _position_row


def test_position_row_fills_every_field_when_history_fetch_succeeds(monkeypatch):
    monkeypatch.setattr(
        data_fetcher, "fetch_ohlcv", lambda symbol, period="10y": fake_history(120.0)
    )

    row = position_metrics._position_row(
        symbol="AAPL", asset_class="equity", qty=2.0, avg_entry_price=100.0,
        stop_loss_price=90.0, take_profit_price=140.0, opened_at="2026-01-01T00:00:00+00:00",
    )

    assert row["symbol"] == "AAPL"
    assert row["current_price"] == pytest.approx(120.0)
    assert row["unrealized_pnl_usd"] == pytest.approx((120.0 - 100.0) * 2.0)
    assert row["unrealized_pnl_pct"] == pytest.approx(20.0)
    assert row["stop_loss_price"] == 90.0
    assert row["take_profit_price"] == 140.0
    for key in position_metrics.RETURN_WINDOWS:
        assert key in row


def test_position_row_degrades_to_nulls_when_history_fetch_fails(monkeypatch):
    def boom(symbol, period="10y"):
        raise ValueError("no data")

    monkeypatch.setattr(data_fetcher, "fetch_ohlcv", boom)

    row = position_metrics._position_row(
        symbol="NEWCOIN-USD", asset_class="crypto", qty=5.0, avg_entry_price=1.0,
        stop_loss_price=None, take_profit_price=None, opened_at=None,
    )

    assert row["symbol"] == "NEWCOIN-USD"
    assert row["qty"] == 5.0
    assert row["avg_entry_price"] == 1.0
    assert row["current_price"] is None
    assert row["unrealized_pnl_usd"] is None
    assert all(row[key] is None for key in position_metrics.RETURN_WINDOWS)


def test_position_row_still_prices_a_recently_listed_symbol_with_short_history(monkeypatch):
    # Real crypto scenario: enough history for 1w/1m, nowhere near enough for 6y.
    monkeypatch.setattr(
        data_fetcher, "fetch_ohlcv", lambda symbol, period="10y": fake_history(50.0, days=20, start_close=40.0)
    )

    row = position_metrics._position_row(
        symbol="NEW-USD", asset_class="crypto", qty=10.0, avg_entry_price=45.0,
        stop_loss_price=None, take_profit_price=None, opened_at=None,
    )

    assert row["current_price"] == pytest.approx(50.0)
    assert row["return_1w_pct"] is not None
    assert row["return_6y_pct"] is None


# ------------------------------------------------------- compute_position_metrics


def test_simulation_reads_positions_from_the_ledger(tmp_path, monkeypatch):
    bot_logger = BotLogger(str(tmp_path / "t.db"))
    bot_logger.open_simulated_position("AAPL", qty=2.0, avg_entry_price=100.0, stop_loss_price=90.0, take_profit_price=140.0)
    bot_logger.open_simulated_position("BTC-USD", qty=0.1, avg_entry_price=50000.0)

    monkeypatch.setattr(
        data_fetcher, "fetch_ohlcv", lambda symbol, period="10y": fake_history(120.0)
    )

    rows = position_metrics.compute_position_metrics(
        bot_logger, config={"symbols": []}, is_live=False, infer_asset_class=infer_asset_class
    )

    symbols = {r["symbol"] for r in rows}
    assert symbols == {"AAPL", "BTC-USD"}
    aapl = next(r for r in rows if r["symbol"] == "AAPL")
    assert aapl["asset_class"] == "equity"
    assert aapl["stop_loss_price"] == 90.0
    btc = next(r for r in rows if r["symbol"] == "BTC-USD")
    assert btc["asset_class"] == "crypto"


def test_simulation_skips_a_closed_zero_qty_row(tmp_path, monkeypatch):
    bot_logger = BotLogger(str(tmp_path / "t.db"))
    bot_logger.open_simulated_position("AAPL", qty=2.0, avg_entry_price=100.0)
    bot_logger.close_simulated_position("AAPL")

    rows = position_metrics.compute_position_metrics(
        bot_logger, config={"symbols": []}, is_live=False, infer_asset_class=infer_asset_class
    )
    assert rows == []


def test_live_mode_reads_positions_from_the_broker(tmp_path, monkeypatch):
    bot_logger = BotLogger(str(tmp_path / "t.db"))
    config = {
        "symbols": [
            {"symbol": "AAPL", "asset_class": "equity"},
            {"symbol": "MSFT", "asset_class": "equity"},
        ]
    }

    import execution

    def fake_fetch_existing_position(symbol, asset_class, is_live, bot_logger):
        if symbol == "AAPL":
            return ExistingPosition(qty=3.0, avg_entry_price=150.0)
        return None  # MSFT: flat

    monkeypatch.setattr(execution, "fetch_existing_position", fake_fetch_existing_position)
    monkeypatch.setattr(
        data_fetcher, "fetch_ohlcv", lambda symbol, period="10y": fake_history(160.0)
    )

    rows = position_metrics.compute_position_metrics(
        bot_logger, config=config, is_live=True, infer_asset_class=infer_asset_class
    )

    assert [r["symbol"] for r in rows] == ["AAPL"]
    assert rows[0]["qty"] == 3.0
    assert rows[0]["avg_entry_price"] == 150.0
    # No bot-managed exit for AAPL in this test -- a broker-side bracket is assumed,
    # so stop/take must be an honest null, never guessed.
    assert rows[0]["stop_loss_price"] is None


def test_live_mode_one_symbols_broker_failure_does_not_block_the_rest(tmp_path, monkeypatch):
    bot_logger = BotLogger(str(tmp_path / "t.db"))
    config = {
        "symbols": [
            {"symbol": "AAPL", "asset_class": "equity"},
            {"symbol": "MSFT", "asset_class": "equity"},
        ]
    }

    import execution

    def fake_fetch_existing_position(symbol, asset_class, is_live, bot_logger):
        if symbol == "AAPL":
            raise RuntimeError("Alpaca outage")
        return ExistingPosition(qty=1.0, avg_entry_price=300.0)

    monkeypatch.setattr(execution, "fetch_existing_position", fake_fetch_existing_position)
    monkeypatch.setattr(
        data_fetcher, "fetch_ohlcv", lambda symbol, period="10y": fake_history(310.0)
    )

    rows = position_metrics.compute_position_metrics(
        bot_logger, config=config, is_live=True, infer_asset_class=infer_asset_class
    )

    assert [r["symbol"] for r in rows] == ["MSFT"]


# --------------------------------------------------------------- export_positions_json


def test_export_positions_json_writes_generated_at_and_is_live(tmp_path):
    path = tmp_path / "positions.json"
    position_metrics.export_positions_json(
        [{"symbol": "AAPL", "qty": 1.0}], str(path), is_live=True
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["is_live"] is True
    assert payload["positions"] == [{"symbol": "AAPL", "qty": 1.0}]
    assert "generated_at" in payload
