"""Trade-level diagnostic report for a backtest's trade log.

Numbers only, no interpretation or fix proposed -- that discussion happens
once the numbers are actually looked at together. This module computes and
prints; it does not conclude anything about whether the strategy is good.

Works against EITHER backtest database unmodified: `backtest.py` and
`backtest_historical_screening.py` both write through the exact same
`backtest.BacktestLogger` schema (the rotating-screening module reuses it
as-is -- see its own module docstring, "Reused, not reimplemented"), so this
tool takes a bare `--db` path and never assumes which one produced it.

Trade reconstruction
---------------------
The `trades` table logs individual buy/sell rows, not pre-paired closed
trades -- `realized_pnl_usd` lives on the sell row only. This module pairs
each sell with the most recently opened, still-open buy for the same symbol
(FIFO per symbol), which is exactly correct given the engine's own invariant:
at most one open position per symbol at a time (`run_symbol_for_day` only
opens a new position when `existing is None`), so a symbol's buy always
precedes its closing sell in insertion order with nothing in between. Rows
read in `id ASC` order, which is both insertion order and simulated-date
order for everything this schema can produce.

A buy left open at the end of the log (no matching sell) is not a closed
trade -- it is counted and reported separately, never silently dropped and
never treated as a loss or a win it hasn't actually realized.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

# Categorises how a position closed. "stop_loss"/"take_profit" come from the
# fixed reasoning text sweep_positions_for_day always writes for an automatic
# exit ("Stop-loss automatic (...)" / "Take-profit automatic (...)"; see
# backtest.py). "model_sell" is any is_auto_close=0 sell -- the model itself
# decided to close, not a level being touched. "auto_close_unknown" is a
# defensive fallback for an is_auto_close=1 row whose reasoning doesn't match
# either known prefix -- should not occur given the current engine, kept
# rather than silently mis-bucketing a row into the wrong category.
CLOSE_REASON_STOP_LOSS = "stop_loss"
CLOSE_REASON_TAKE_PROFIT = "take_profit"
CLOSE_REASON_MODEL_SELL = "model_sell"
CLOSE_REASON_UNKNOWN_AUTO = "auto_close_unknown"

# 0.05-wide confidence buckets from 0.50 up -- both live and simulation
# min-confidence thresholds (0.65 / 0.40, config.yaml) fall on a 0.05
# boundary, and backtest.py always evaluates under the live prompt
# (mode.resolve_mode_settings(True, ...)), so every buy's confidence should
# sit at or above the live threshold outside of a clamp/override path.
CONFIDENCE_BUCKET_WIDTH = 0.05


def infer_asset_class(symbol: str) -> str:
    """Same fallback heuristic main.infer_asset_class uses when a symbol has
    no config entry to consult -- a trade log has no config context at all,
    only the bare symbol string, so this is the only signal available.
    """
    return "crypto" if symbol.upper().endswith(("-USD", "-USDT")) else "equity"


def _parse_date(value: str) -> datetime:
    # simulated_date is written as str(day) in BacktestLogger.log_trade, where
    # `day` is a pandas Timestamp -- "YYYY-MM-DD HH:MM:SS" for every row this
    # schema has ever produced.
    return datetime.fromisoformat(value)


def categorize_close(row: sqlite3.Row) -> str:
    if not row["is_auto_close"]:
        return CLOSE_REASON_MODEL_SELL
    reasoning = row["reasoning"] or ""
    if reasoning.startswith("Stop-loss"):
        return CLOSE_REASON_STOP_LOSS
    if reasoning.startswith("Take-profit"):
        return CLOSE_REASON_TAKE_PROFIT
    return CLOSE_REASON_UNKNOWN_AUTO


@dataclass
class ClosedTrade:
    symbol: str
    asset_class: str
    opened_date: str
    closed_date: str
    holding_days: int
    qty: float
    entry_price: float
    exit_price: float
    realized_pnl_usd: float
    entry_confidence: Optional[float]
    close_reason: str


def load_trade_rows(db_path: str) -> List[sqlite3.Row]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute("SELECT * FROM trades ORDER BY id ASC").fetchall()
    finally:
        conn.close()


def reconstruct_closed_trades(rows: List[sqlite3.Row]) -> "tuple[List[ClosedTrade], int]":
    """Pair buys with sells per symbol (FIFO). Returns (closed_trades, still_open_count).

    A malformed log -- a sell with no open buy for that symbol -- is skipped
    rather than raising: this is a read-only diagnostic over data that has
    already been written, not a place to enforce the writer's own invariants.
    """
    open_buy_by_symbol: Dict[str, sqlite3.Row] = {}
    closed: List[ClosedTrade] = []

    for row in rows:
        symbol = row["symbol"]
        if row["action"] == "buy":
            open_buy_by_symbol[symbol] = row
            continue
        if row["action"] != "sell":
            continue

        buy_row = open_buy_by_symbol.pop(symbol, None)
        if buy_row is None:
            continue  # sell with no matching open buy in this log -- skip, don't guess

        opened = _parse_date(buy_row["simulated_date"])
        closed_at = _parse_date(row["simulated_date"])
        pnl = row["realized_pnl_usd"]
        if pnl is None:
            continue  # a sell row must carry its realised P&L to be usable here

        closed.append(
            ClosedTrade(
                symbol=symbol,
                asset_class=infer_asset_class(symbol),
                opened_date=buy_row["simulated_date"],
                closed_date=row["simulated_date"],
                holding_days=(closed_at - opened).days,
                qty=float(buy_row["qty"]),
                entry_price=float(buy_row["price"]),
                exit_price=float(row["price"]),
                realized_pnl_usd=float(pnl),
                entry_confidence=buy_row["confidence"],
                close_reason=categorize_close(row),
            )
        )

    return closed, len(open_buy_by_symbol)


# ------------------------------------------------------------- report sections


def realized_risk_reward(trades: List[ClosedTrade]) -> Dict[str, Any]:
    wins = [t.realized_pnl_usd for t in trades if t.realized_pnl_usd > 0]
    losses = [t.realized_pnl_usd for t in trades if t.realized_pnl_usd < 0]
    breakeven = [t.realized_pnl_usd for t in trades if t.realized_pnl_usd == 0]

    avg_win = sum(wins) / len(wins) if wins else None
    avg_loss = sum(losses) / len(losses) if losses else None  # negative
    ratio = (avg_win / abs(avg_loss)) if (avg_win is not None and avg_loss) else None

    return {
        "total_closed_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "breakeven": len(breakeven),
        "win_rate_pct": (len(wins) / len(trades) * 100.0) if trades else None,
        "avg_win_usd": avg_win,
        "avg_loss_usd": avg_loss,
        "risk_reward_ratio": ratio,
        "total_realized_pnl_usd": sum(t.realized_pnl_usd for t in trades),
    }


def _group_stats(trades: List[ClosedTrade], key: str) -> List[Dict[str, Any]]:
    groups: Dict[str, List[ClosedTrade]] = {}
    for t in trades:
        groups.setdefault(getattr(t, key), []).append(t)

    out = []
    for name, group in groups.items():
        wins = [t for t in group if t.realized_pnl_usd > 0]
        out.append(
            {
                "name": name,
                "count": len(group),
                "total_pnl_usd": sum(t.realized_pnl_usd for t in group),
                "avg_pnl_usd": sum(t.realized_pnl_usd for t in group) / len(group),
                "win_rate_pct": len(wins) / len(group) * 100.0,
            }
        )
    out.sort(key=lambda r: r["total_pnl_usd"])
    return out


def breakdown_by_symbol(trades: List[ClosedTrade]) -> List[Dict[str, Any]]:
    return _group_stats(trades, "symbol")


def breakdown_by_asset_class(trades: List[ClosedTrade]) -> List[Dict[str, Any]]:
    return _group_stats(trades, "asset_class")


def breakdown_by_close_reason(trades: List[ClosedTrade]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[ClosedTrade]] = {}
    for t in trades:
        groups.setdefault(t.close_reason, []).append(t)

    out = []
    for reason, group in groups.items():
        wins = [t for t in group if t.realized_pnl_usd > 0]
        out.append(
            {
                "close_reason": reason,
                "count": len(group),
                "avg_holding_days": sum(t.holding_days for t in group) / len(group),
                "avg_pnl_usd": sum(t.realized_pnl_usd for t in group) / len(group),
                "win_rate_pct": len(wins) / len(group) * 100.0,
            }
        )
    out.sort(key=lambda r: r["close_reason"])
    return out


def confidence_vs_outcome(trades: List[ClosedTrade]) -> Dict[str, Any]:
    """Bucketed confidence -> outcome table, plus a single Pearson correlation
    coefficient between entry confidence and realised P&L across every closed
    trade that actually has a recorded entry confidence (always true for a
    buy in this engine -- only sells can be confidence=None, and this table
    is keyed on the BUY's confidence, not the exit's).
    """
    with_confidence = [t for t in trades if t.entry_confidence is not None]

    buckets: Dict[float, List[ClosedTrade]] = {}
    for t in with_confidence:
        # round() before int() truncation guards against float division landing
        # just under a bucket boundary (0.7 / 0.05 == 13.999999999999998 in
        # real float64 arithmetic, verified live -- a bare int() would silently
        # put a confidence of exactly 0.70 into the 0.65-0.70 bucket instead of
        # 0.70-0.75). Confidence values landing exactly on a 0.05 boundary are
        # not an edge case here: both min-confidence thresholds (0.65 live,
        # 0.40 simulation) and typical model output are round numbers.
        floor = round(int(round(t.entry_confidence / CONFIDENCE_BUCKET_WIDTH, 6)) * CONFIDENCE_BUCKET_WIDTH, 2)
        buckets.setdefault(floor, []).append(t)

    bucket_rows = []
    for floor in sorted(buckets.keys()):
        group = buckets[floor]
        wins = [t for t in group if t.realized_pnl_usd > 0]
        bucket_rows.append(
            {
                "confidence_bucket": f"{floor:.2f}-{floor + CONFIDENCE_BUCKET_WIDTH:.2f}",
                "count": len(group),
                "avg_pnl_usd": sum(t.realized_pnl_usd for t in group) / len(group),
                "win_rate_pct": len(wins) / len(group) * 100.0,
            }
        )

    correlation = _pearson_correlation(
        [t.entry_confidence for t in with_confidence],
        [t.realized_pnl_usd for t in with_confidence],
    )

    return {
        "trades_with_confidence": len(with_confidence),
        "trades_missing_confidence": len(trades) - len(with_confidence),
        "buckets": bucket_rows,
        "confidence_vs_pnl_correlation": correlation,
    }


def _pearson_correlation(xs: List[float], ys: List[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    denom = (var_x * var_y) ** 0.5
    return (cov / denom) if denom else None


def build_report(db_path: str) -> Dict[str, Any]:
    rows = load_trade_rows(db_path)
    closed_trades, still_open_count = reconstruct_closed_trades(rows)

    return {
        "db_path": db_path,
        "total_trade_rows": len(rows),
        "closed_trades_analyzed": len(closed_trades),
        "positions_still_open_at_end_of_log": still_open_count,
        "risk_reward": realized_risk_reward(closed_trades),
        "by_symbol": breakdown_by_symbol(closed_trades),
        "by_asset_class": breakdown_by_asset_class(closed_trades),
        "by_close_reason": breakdown_by_close_reason(closed_trades),
        "confidence_vs_outcome": confidence_vs_outcome(closed_trades),
    }, closed_trades


# --------------------------------------------------------------------- format


def _fmt(value: Optional[float], spec: str = "+.2f", none_text: str = "n/a") -> str:
    return none_text if value is None else format(value, spec)


def format_report_text(report: Dict[str, Any]) -> str:
    lines = [
        f"=== Trade diagnostic: {report['db_path']} ===",
        f"Trade rows: {report['total_trade_rows']}   "
        f"Closed trades analyzed: {report['closed_trades_analyzed']}   "
        f"Still open at end of log: {report['positions_still_open_at_end_of_log']}",
        "",
        "-- 1. Realized risk:reward --",
    ]
    rr = report["risk_reward"]
    lines += [
        f"  Wins: {rr['wins']}   Losses: {rr['losses']}   Breakeven: {rr['breakeven']}   "
        f"Win rate: {_fmt(rr['win_rate_pct'], '.1f')}%",
        f"  Avg win: ${_fmt(rr['avg_win_usd'])}   Avg loss: ${_fmt(rr['avg_loss_usd'])}   "
        f"Risk:reward ratio: {_fmt(rr['risk_reward_ratio'], '.2f')}",
        f"  Total realized P&L: ${rr['total_realized_pnl_usd']:+.2f}",
        "",
        "-- 2. Breakdown by asset class --",
    ]
    for row in report["by_asset_class"]:
        lines.append(
            f"  {row['name']:<8} count={row['count']:<4} total_pnl=${row['total_pnl_usd']:+.2f} "
            f"avg_pnl=${row['avg_pnl_usd']:+.2f} win_rate={row['win_rate_pct']:.1f}%"
        )
    lines.append("")
    lines.append("-- 2b. Breakdown by symbol --")
    for row in report["by_symbol"]:
        lines.append(
            f"  {row['name']:<10} count={row['count']:<4} total_pnl=${row['total_pnl_usd']:+.2f} "
            f"avg_pnl=${row['avg_pnl_usd']:+.2f} win_rate={row['win_rate_pct']:.1f}%"
        )
    lines.append("")
    lines.append("-- 3. Breakdown by how the trade closed --")
    for row in report["by_close_reason"]:
        lines.append(
            f"  {row['close_reason']:<18} count={row['count']:<4} "
            f"avg_holding_days={row['avg_holding_days']:.1f} avg_pnl=${row['avg_pnl_usd']:+.2f} "
            f"win_rate={row['win_rate_pct']:.1f}%"
        )
    lines.append("")
    lines.append("-- 4. Confidence vs. outcome --")
    cvo = report["confidence_vs_outcome"]
    lines.append(
        f"  Trades with recorded confidence: {cvo['trades_with_confidence']}   "
        f"Missing: {cvo['trades_missing_confidence']}"
    )
    lines.append(
        f"  Pearson correlation (entry confidence vs realized P&L): "
        f"{_fmt(cvo['confidence_vs_pnl_correlation'], '+.3f')}"
    )
    for row in cvo["buckets"]:
        lines.append(
            f"  {row['confidence_bucket']:<12} count={row['count']:<4} "
            f"avg_pnl=${row['avg_pnl_usd']:+.2f} win_rate={row['win_rate_pct']:.1f}%"
        )
    return "\n".join(lines)


def write_trades_csv(trades: List[ClosedTrade], path: str) -> None:
    fields = [
        "symbol", "asset_class", "opened_date", "closed_date", "holding_days",
        "qty", "entry_price", "exit_price", "realized_pnl_usd", "entry_confidence",
        "close_reason",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for t in trades:
            writer.writerow({f: getattr(t, f) for f in fields})


# ------------------------------------------------------------------------ CLI


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Trade-level diagnostic report for a backtest.BacktestLogger database "
        "(works against backtest.db or backtest_historical_screening.db unmodified)."
    )
    parser.add_argument("--db", required=True, help="Path to the backtest sqlite database")
    parser.add_argument("--csv", default=None, help="Also write the per-trade table to this CSV path")
    args = parser.parse_args()

    report, closed_trades = build_report(args.db)
    print(format_report_text(report))

    if args.csv:
        write_trades_csv(closed_trades, args.csv)
        print(f"\nWrote {len(closed_trades)} closed trades to {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
