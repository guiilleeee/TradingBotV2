"""diagnose_backtest_loss.py: trade pairing, the four report sections, and the
confidence-bucket float-truncation guard.

Every test builds its database through backtest.BacktestLogger.log_trade --
the exact same writer both backtest.py and backtest_historical_screening.py
use -- so a passing test here is evidence the tool actually works against
either engine's real output, not just a hand-rolled schema.
"""

from datetime import date

import pytest

import backtest
import diagnose_backtest_loss as diag


def make_db(tmp_path, name="test.db"):
    return backtest.BacktestLogger(str(tmp_path / name))


def log_buy(logger, day, symbol, qty, price, confidence):
    logger.log_trade(
        day=day, symbol=symbol, action="buy", qty=qty, price=price,
        realized_pnl_usd=None, confidence=confidence, reasoning="entry", override_reason=None,
        is_auto_close=False,
    )


def log_stop_loss_sell(logger, day, symbol, qty, price, pnl, level=95.0):
    logger.log_trade(
        day=day, symbol=symbol, action="sell", qty=qty, price=price,
        realized_pnl_usd=pnl, confidence=None,
        reasoning=f"Stop-loss automatic (nivell {level:g}, preu {price:g}).",
        override_reason="automatic exit", is_auto_close=True,
    )


def log_take_profit_sell(logger, day, symbol, qty, price, pnl, level=120.0):
    logger.log_trade(
        day=day, symbol=symbol, action="sell", qty=qty, price=price,
        realized_pnl_usd=pnl, confidence=None,
        reasoning=f"Take-profit automatic (nivell {level:g}, preu {price:g}).",
        override_reason="automatic exit", is_auto_close=True,
    )


def log_model_sell(logger, day, symbol, qty, price, pnl, confidence=0.55):
    logger.log_trade(
        day=day, symbol=symbol, action="sell", qty=qty, price=price,
        realized_pnl_usd=pnl, confidence=confidence, reasoning="model decided to exit",
        override_reason=None, is_auto_close=False,
    )


@pytest.fixture
def populated_db(tmp_path):
    logger = make_db(tmp_path)

    # AAPL: opened, stopped out at a loss two days later.
    log_buy(logger, date(2024, 1, 1), "AAPL", qty=10.0, price=100.0, confidence=0.70)
    log_stop_loss_sell(logger, date(2024, 1, 3), "AAPL", qty=10.0, price=95.0, pnl=-50.0)

    # AAPL again: opened, take-profit ten days later.
    log_buy(logger, date(2024, 1, 5), "AAPL", qty=5.0, price=100.0, confidence=0.90)
    log_take_profit_sell(logger, date(2024, 1, 15), "AAPL", qty=5.0, price=120.0, pnl=100.0)

    # BTC-USD: opened, the model itself decided to sell two days later.
    log_buy(logger, date(2024, 1, 2), "BTC-USD", qty=0.01, price=40_000.0, confidence=0.60)
    log_model_sell(logger, date(2024, 1, 4), "BTC-USD", qty=0.01, price=41_000.0, pnl=10.0)

    # MSFT: opened, never closed -- must be counted separately, not as a trade.
    log_buy(logger, date(2024, 1, 10), "MSFT", qty=2.0, price=200.0, confidence=0.95)

    # A sell for a symbol never bought in this log -- must be skipped, not crash.
    log_model_sell(logger, date(2024, 1, 20), "GOOG", qty=1.0, price=150.0, pnl=5.0)

    logger.close()
    return str(tmp_path / "test.db")


# ============================================================== reconstruction


def test_closed_trades_are_paired_correctly_and_open_position_is_separate(populated_db):
    report, closed = diag.build_report(populated_db)
    assert report["total_trade_rows"] == 8  # 4 buys logged + 4 sells, MSFT buy has no sell... wait
    assert report["closed_trades_analyzed"] == 3
    assert report["positions_still_open_at_end_of_log"] == 1  # MSFT

    symbols = sorted(t.symbol for t in closed)
    assert symbols == ["AAPL", "AAPL", "BTC-USD"]


def test_an_orphan_sell_with_no_matching_buy_is_skipped_not_crashed(populated_db):
    # GOOG's sell has no buy anywhere in the log -- proven by it not appearing
    # in the closed set and nothing raising while building the report.
    report, closed = diag.build_report(populated_db)
    assert "GOOG" not in {t.symbol for t in closed}


def test_holding_days_is_the_calendar_gap_between_buy_and_sell(populated_db):
    _, closed = diag.build_report(populated_db)
    by_pnl = {round(t.realized_pnl_usd): t for t in closed}
    assert by_pnl[-50].holding_days == 2   # Jan 1 -> Jan 3
    assert by_pnl[100].holding_days == 10  # Jan 5 -> Jan 15
    assert by_pnl[10].holding_days == 2    # Jan 2 -> Jan 4


# ==================================================================== sections


def test_realized_risk_reward(populated_db):
    report, _ = diag.build_report(populated_db)
    rr = report["risk_reward"]
    assert rr["wins"] == 2
    assert rr["losses"] == 1
    assert rr["win_rate_pct"] == pytest.approx(200 / 3)
    assert rr["avg_win_usd"] == pytest.approx((100.0 + 10.0) / 2)
    assert rr["avg_loss_usd"] == pytest.approx(-50.0)
    assert rr["risk_reward_ratio"] == pytest.approx(55.0 / 50.0)
    assert rr["total_realized_pnl_usd"] == pytest.approx(60.0)


def test_breakdown_by_asset_class(populated_db):
    report, _ = diag.build_report(populated_db)
    by_class = {row["name"]: row for row in report["by_asset_class"]}
    assert by_class["equity"]["count"] == 2
    assert by_class["equity"]["total_pnl_usd"] == pytest.approx(50.0)  # -50 + 100
    assert by_class["crypto"]["count"] == 1
    assert by_class["crypto"]["total_pnl_usd"] == pytest.approx(10.0)


def test_breakdown_by_symbol(populated_db):
    report, _ = diag.build_report(populated_db)
    by_symbol = {row["name"]: row for row in report["by_symbol"]}
    assert by_symbol["AAPL"]["count"] == 2
    assert by_symbol["AAPL"]["total_pnl_usd"] == pytest.approx(50.0)
    assert by_symbol["BTC-USD"]["count"] == 1
    assert by_symbol["BTC-USD"]["total_pnl_usd"] == pytest.approx(10.0)


def test_breakdown_by_close_reason(populated_db):
    report, _ = diag.build_report(populated_db)
    by_reason = {row["close_reason"]: row for row in report["by_close_reason"]}
    assert by_reason["stop_loss"]["count"] == 1
    assert by_reason["stop_loss"]["avg_holding_days"] == pytest.approx(2.0)
    assert by_reason["stop_loss"]["avg_pnl_usd"] == pytest.approx(-50.0)
    assert by_reason["take_profit"]["count"] == 1
    assert by_reason["take_profit"]["avg_holding_days"] == pytest.approx(10.0)
    assert by_reason["model_sell"]["count"] == 1
    assert by_reason["model_sell"]["avg_holding_days"] == pytest.approx(2.0)


def test_a_close_reason_never_seen_by_the_engine_has_no_row(populated_db):
    report, _ = diag.build_report(populated_db)
    reasons = {row["close_reason"] for row in report["by_close_reason"]}
    assert diag.CLOSE_REASON_UNKNOWN_AUTO not in reasons


def test_confidence_vs_outcome_buckets_and_correlation(populated_db):
    report, _ = diag.build_report(populated_db)
    cvo = report["confidence_vs_outcome"]
    assert cvo["trades_with_confidence"] == 3
    assert cvo["trades_missing_confidence"] == 0

    buckets = {row["confidence_bucket"]: row for row in cvo["buckets"]}
    assert buckets["0.60-0.65"]["count"] == 1
    assert buckets["0.60-0.65"]["avg_pnl_usd"] == pytest.approx(10.0)
    assert buckets["0.70-0.75"]["count"] == 1
    assert buckets["0.70-0.75"]["avg_pnl_usd"] == pytest.approx(-50.0)
    assert buckets["0.90-0.95"]["count"] == 1
    assert buckets["0.90-0.95"]["avg_pnl_usd"] == pytest.approx(100.0)

    # Higher confidence (0.9 -> +100) beat lower confidence (0.7 -> -50) here,
    # so the correlation must come out positive -- not asserting an exact
    # value, just the sign, since this is a real Pearson computation.
    assert cvo["confidence_vs_pnl_correlation"] > 0


# ---------------------------------------------- confidence bucket float guard


def test_confidence_bucketing_survives_the_float_division_edge_case():
    """0.7 / 0.05 == 13.999999999999998 in real float64 arithmetic (verified
    live) -- a bare int() truncation would silently sort a confidence of
    exactly 0.70 into the 0.65-0.70 bucket instead of 0.70-0.75. This is the
    negative-control-shaped test proving the round() guard actually matters.
    """
    trade = diag.ClosedTrade(
        symbol="X", asset_class="equity", opened_date="2024-01-01", closed_date="2024-01-02",
        holding_days=1, qty=1.0, entry_price=100.0, exit_price=101.0,
        realized_pnl_usd=1.0, entry_confidence=0.70, close_reason="model_sell",
    )
    result = diag.confidence_vs_outcome([trade])
    assert result["buckets"][0]["confidence_bucket"] == "0.70-0.75"


@pytest.mark.parametrize("confidence,expected_bucket", [
    (0.70, "0.70-0.75"),
    (0.60, "0.60-0.65"),
    (0.90, "0.90-0.95"),
    (0.99, "0.95-1.00"),
    (0.50, "0.50-0.55"),
])
def test_confidence_bucketing_is_correct_at_every_boundary(confidence, expected_bucket):
    trade = diag.ClosedTrade(
        symbol="X", asset_class="equity", opened_date="2024-01-01", closed_date="2024-01-02",
        holding_days=1, qty=1.0, entry_price=100.0, exit_price=101.0,
        realized_pnl_usd=1.0, entry_confidence=confidence, close_reason="model_sell",
    )
    result = diag.confidence_vs_outcome([trade])
    assert result["buckets"][0]["confidence_bucket"] == expected_bucket


# ============================================================ asset class + CSV


@pytest.mark.parametrize("symbol,expected", [
    ("AAPL", "equity"), ("BTC-USD", "crypto"), ("ETH-USDT", "crypto"), ("btc-usd", "crypto"),
])
def test_infer_asset_class(symbol, expected):
    assert diag.infer_asset_class(symbol) == expected


def test_write_trades_csv_round_trips_every_closed_trade(populated_db, tmp_path):
    _, closed = diag.build_report(populated_db)
    csv_path = tmp_path / "out.csv"
    diag.write_trades_csv(closed, str(csv_path))

    import csv as csv_module

    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv_module.DictReader(handle))
    assert len(rows) == 3
    assert {row["symbol"] for row in rows} == {"AAPL", "BTC-USD"}


# ==================================================================== format


def test_format_report_text_includes_all_four_sections(populated_db):
    report, _ = diag.build_report(populated_db)
    text = diag.format_report_text(report)
    assert "1. Realized risk:reward" in text
    assert "2. Breakdown by asset class" in text
    assert "2b. Breakdown by symbol" in text
    assert "3. Breakdown by how the trade closed" in text
    assert "4. Confidence vs. outcome" in text


def test_format_report_text_never_crashes_on_an_empty_database(tmp_path):
    logger = make_db(tmp_path)
    logger.close()
    report, _ = diag.build_report(str(tmp_path / "test.db"))
    text = diag.format_report_text(report)
    assert "Closed trades analyzed: 0" in text
