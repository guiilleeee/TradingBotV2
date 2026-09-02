"""SQLite persistence: signal audit trail, realised P&L, and the bot-managed ledger.

Three tables and no ORM. Every timestamp written here is UTC, and every read that
filters by day must use UTC too -- mixing the two silently misaligns the day
boundary depending on which machine the cycle happens to run on.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

from models import ExecutionResult, ExistingPosition, SignalInput, SignalOutput, TradeSignal

DEFAULT_DB_PATH = "trading_bot.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT NOT NULL,
    symbol           TEXT NOT NULL,
    signal_input     TEXT,
    raw_output       TEXT,
    final_signal     TEXT,
    override_reason  TEXT,
    execution_result TEXT
);

CREATE TABLE IF NOT EXISTS pnl (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT NOT NULL,
    symbol           TEXT NOT NULL,
    realized_pnl_usd REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS simulated_positions (
    symbol            TEXT PRIMARY KEY,
    qty               REAL NOT NULL,
    avg_entry_price   REAL NOT NULL,
    stop_loss_price   REAL,
    take_profit_price REAL,
    opened_at         TEXT NOT NULL
);
"""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_day_start_iso() -> str:
    now = datetime.now(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()


def _dump(value: Any) -> Optional[str]:
    """JSON-serialise a Pydantic model, a dict, or None for a blob column."""
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return json.dumps(value.model_dump(), default=str)
    return json.dumps(value, default=str)


def _load(blob: Any) -> Dict[str, Any]:
    try:
        return json.loads(blob) if blob else {}
    except (TypeError, ValueError):
        return {}


class BotLogger:
    """Owns the SQLite file. One connection per call, so nothing stays locked.

    Note on `simulated_positions`: in simulation it holds every open position. In
    live it holds only positions the bot has to exit itself -- fractional/notional
    entries the broker will not accept bracket legs for. Same table, same sweep.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ signals

    def log_signal(
        self,
        symbol: str,
        signal_input: Optional[SignalInput],
        raw_output: Optional[SignalOutput],
        final_signal: Optional[TradeSignal],
        execution_result: Optional[ExecutionResult] = None,
    ) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO signals (timestamp, symbol, signal_input, raw_output, "
                "final_signal, override_reason, execution_result) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    utc_now_iso(),
                    symbol,
                    _dump(signal_input),
                    _dump(raw_output),
                    _dump(final_signal),
                    final_signal.override_reason if final_signal else None,
                    _dump(execution_result),
                ),
            )
            return int(cur.lastrowid or 0)

    def log_auto_close_signal(
        self,
        symbol: str,
        reason: str,
        price: float,
        qty: float,
        pnl: float,
        equity: float,
    ) -> int:
        """Write a synthetic signal row for a stop-loss / take-profit auto-close.

        The dashboard renders every row through one code path, so these blobs must
        carry the same keys a model-driven row does. Two in particular:
        account_equity_usd, which the dashboard reads off the newest row for
        "Patrimoni total", and reasoning, which rendered as "undefined" without one.
        Both were omitted in an earlier build; hence the asserts below.
        """
        signal_input: Dict[str, Any] = {
            "symbol": symbol,
            "current_price": price,
            "account_equity_usd": equity,
            "existing_position": {"qty": qty, "avg_entry_price": None},
            "technical_indicators": None,
            "recent_headlines": [],
            "synthetic": True,
        }
        decision: Dict[str, Any] = {
            "symbol": symbol,
            "action": "sell",
            "confidence": 1.0,
            "position_size_pct": 0.0,
            "stop_loss_price": None,
            "take_profit_price": None,
            "reasoning": reason,
        }
        final_signal = dict(decision, override_reason="automatic exit", raw_action="sell")
        execution_result = {
            "status": "success",
            "order_id": None,
            "fill_price": price,
            "message": reason,
            "realized_pnl_usd": pnl,
            "qty": qty,
        }

        assert signal_input["account_equity_usd"] is not None, "synthetic row needs equity"
        assert final_signal["reasoning"], "synthetic row needs reasoning"

        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO signals (timestamp, symbol, signal_input, raw_output, "
                "final_signal, override_reason, execution_result) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    utc_now_iso(),
                    symbol,
                    json.dumps(signal_input),
                    json.dumps(decision),
                    json.dumps(final_signal),
                    "automatic exit",
                    json.dumps(execution_result),
                ),
            )
            return int(cur.lastrowid or 0)

    def get_last_buy_price(self, symbol: str) -> Optional[float]:
        """Cost basis for `symbol` taken from the most recent logged buy.

        Needed for OKX, whose spot balance carries no entry price. This is only
        correct because the duplicate-buy guard in execution.py keeps at most one
        open buy per symbol at a time, so "the last buy" is the open position.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT final_signal, signal_input, execution_result FROM signals "
                "WHERE symbol = ? ORDER BY id DESC LIMIT 200",
                (symbol,),
            ).fetchall()

        for row in rows:
            final = _load(row["final_signal"])
            if final.get("action") != "buy":
                continue

            execution = _load(row["execution_result"])
            if execution.get("status") not in ("success", "dry_run"):
                continue

            fill = execution.get("fill_price")
            if fill:
                return float(fill)

            entered = _load(row["signal_input"]).get("current_price")
            if entered:
                return float(entered)

        return None

    # ---------------------------------------------------------------------- pnl

    def record_pnl(self, symbol: str, amount: float) -> None:
        """Book a realised P&L amount.

        Must be called on every close, live and simulated. An earlier build defined
        this method and never invoked it anywhere, which left the circuit breaker
        permanently inert: get_today_realized_loss_pct always summed an empty table.
        """
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO pnl (timestamp, symbol, realized_pnl_usd) VALUES (?, ?, ?)",
                (utc_now_iso(), symbol, float(amount)),
            )

    def get_today_realized_loss_pct(self, equity: float) -> float:
        """Today's realised P&L as a percent of equity. Negative means a loss.

        UTC day boundary, matching how the timestamps are written. Local time here
        would misalign the day depending on where the runner happens to live.
        """
        if equity <= 0:
            return 0.0
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(realized_pnl_usd), 0.0) AS total FROM pnl WHERE timestamp >= ?",
                (_utc_day_start_iso(),),
            ).fetchone()
        return (float(row["total"]) / equity) * 100.0

    def get_all_time_realized_pnl(self) -> float:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(realized_pnl_usd), 0.0) AS total FROM pnl"
            ).fetchone()
        return float(row["total"])

    # --------------------------------------------------------- simulated ledger

    def get_simulated_position(self, symbol: str) -> Optional[ExistingPosition]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT qty, avg_entry_price FROM simulated_positions WHERE symbol = ?",
                (symbol,),
            ).fetchone()
        if row is None or float(row["qty"]) <= 0:
            return None
        return ExistingPosition(
            qty=float(row["qty"]), avg_entry_price=float(row["avg_entry_price"])
        )

    def get_all_simulated_positions(self) -> List[Dict[str, Any]]:
        """Full rows, including the exit levels the sweep needs."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT symbol, qty, avg_entry_price, stop_loss_price, take_profit_price, "
                "opened_at FROM simulated_positions ORDER BY symbol"
            ).fetchall()
        return [dict(row) for row in rows]

    def open_simulated_position(
        self,
        symbol: str,
        qty: float,
        avg_entry_price: float,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO simulated_positions (symbol, qty, avg_entry_price, "
                "stop_loss_price, take_profit_price, opened_at) VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(symbol) DO UPDATE SET qty=excluded.qty, "
                "avg_entry_price=excluded.avg_entry_price, "
                "stop_loss_price=excluded.stop_loss_price, "
                "take_profit_price=excluded.take_profit_price, opened_at=excluded.opened_at",
                (
                    symbol,
                    float(qty),
                    float(avg_entry_price),
                    stop_loss_price,
                    take_profit_price,
                    utc_now_iso(),
                ),
            )

    def close_simulated_position(self, symbol: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM simulated_positions WHERE symbol = ?", (symbol,))

    # ------------------------------------------------------------------- export

    def export_signals_csv(self, path: str, limit: int = 500) -> int:
        """Flatten the newest `limit` signal rows into the dashboard's CSV."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM signals ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()

        fields = [
            "id",
            "timestamp",
            "symbol",
            "action",
            "raw_action",
            "confidence",
            "position_size_pct",
            "stop_loss_price",
            "take_profit_price",
            "current_price",
            "account_equity_usd",
            "override_reason",
            "execution_status",
            "fill_price",
            "realized_pnl_usd",
            "reasoning",
        ]

        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in reversed(rows):
                final = _load(row["final_signal"])
                inp = _load(row["signal_input"])
                execution = _load(row["execution_result"])
                writer.writerow(
                    {
                        "id": row["id"],
                        "timestamp": row["timestamp"],
                        "symbol": row["symbol"],
                        "action": final.get("action"),
                        "raw_action": final.get("raw_action"),
                        "confidence": final.get("confidence"),
                        "position_size_pct": final.get("position_size_pct"),
                        "stop_loss_price": final.get("stop_loss_price"),
                        "take_profit_price": final.get("take_profit_price"),
                        "current_price": inp.get("current_price"),
                        "account_equity_usd": inp.get("account_equity_usd"),
                        "override_reason": row["override_reason"],
                        "execution_status": execution.get("status"),
                        "fill_price": execution.get("fill_price"),
                        "realized_pnl_usd": execution.get("realized_pnl_usd"),
                        "reasoning": final.get("reasoning"),
                    }
                )
        return len(rows)
