"""Per-position metrics for the dashboard's holdings list: returns and unrealised P&L.

Scope is deliberately narrow: only the positions actually open right now (a handful
of symbols, cheap to compute in full), never the 503-symbol screening universe --
that's what equity_universe.fetch_universe_price_data is for, and this module does
not touch it.

Two callers:
  - main.py, once per 4h trading cycle, after every symbol has been processed.
  - This module's own __main__ entry point, run standalone by a separate, much more
    frequent GitHub Actions job (see .github/workflows/refresh_positions.yml) that
    re-prices open positions every 15 minutes. That job makes zero AI calls and runs
    no trading logic at all -- it is a plain price refresh, unrelated in cost and
    kind to the 4h cycle, and must never be confused with one.

Degrades gracefully at three independent levels, matching this project's house
style (data_fetcher.fetch_headlines, market_intel.fetch_positioning): a symbol whose
price history can't be fetched at all still gets a row (every field null except the
static ones already known from the ledger/broker); a symbol whose history fetch
succeeds but is too short for a given return window gets null for that window only,
never a fabricated number; and one symbol's failure never blocks the rest of the
export.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

import data_fetcher
from logger import utc_now_iso
from models import AssetClass

# 10y comfortably covers the longest window below (6y) for any symbol old enough to
# have that much history, while yfinance simply returns whatever shorter range
# actually exists for a recently-listed symbol -- it does not error on a partial
# range, only on a genuinely empty one (see data_fetcher.fetch_ohlcv).
HISTORY_PERIOD = "10y"

# Calendar days, not trading/bar counts -- these are "1 week ago" in the ordinary
# sense, matched against the nearest bar on or before that date.
RETURN_WINDOWS: Dict[str, int] = {
    "return_1w_pct": 7,
    "return_1m_pct": 30,
    "return_6m_pct": 182,
    "return_1y_pct": 365,
    "return_3y_pct": 365 * 3,
    "return_6y_pct": 365 * 6,
}

DEFAULT_POSITIONS_PATH = "positions.json"


def _return_since(df: pd.DataFrame, days_ago: int) -> Optional[float]:
    """% change from the closest bar on/before `days_ago` calendar days back to now.

    None (never a fabricated number) whenever no bar that old exists in `df` -- the
    caller's honest signal that this symbol doesn't have enough history for this
    window (a recently listed stock, or most crypto, for the longer windows).
    """
    close = df["Close"].astype(float)
    last_close = float(close.iloc[-1])
    target = df.index[-1] - pd.Timedelta(days=days_ago)

    eligible = df.index[df.index <= target]
    if len(eligible) == 0:
        return None

    past_close = float(close.loc[eligible[-1]])
    if past_close == 0:
        return None
    return (last_close / past_close - 1.0) * 100.0


def _empty_row(
    symbol: str,
    asset_class: AssetClass,
    qty: float,
    avg_entry_price: float,
    stop_loss_price: Optional[float],
    take_profit_price: Optional[float],
    opened_at: Optional[str],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "symbol": symbol,
        "asset_class": asset_class,
        "qty": qty,
        "avg_entry_price": avg_entry_price,
        "current_price": None,
        "unrealized_pnl_usd": None,
        "unrealized_pnl_pct": None,
        "stop_loss_price": stop_loss_price,
        "take_profit_price": take_profit_price,
        "opened_at": opened_at,
    }
    for key in RETURN_WINDOWS:
        row[key] = None
    return row


def _position_row(
    symbol: str,
    asset_class: AssetClass,
    qty: float,
    avg_entry_price: float,
    stop_loss_price: Optional[float],
    take_profit_price: Optional[float],
    opened_at: Optional[str],
) -> Dict[str, Any]:
    row = _empty_row(
        symbol, asset_class, qty, avg_entry_price, stop_loss_price, take_profit_price, opened_at
    )

    try:
        df = data_fetcher.fetch_ohlcv(symbol, period=HISTORY_PERIOD)
    except Exception:
        # No price history at all -- the row still carries qty/avg_entry_price/stop/
        # take/opened_at, just no mark-to-market. Never blocks the rest of the export.
        return row

    try:
        current_price = data_fetcher.latest_price(df)
        row["current_price"] = current_price
        row["unrealized_pnl_usd"] = (current_price - avg_entry_price) * qty
        row["unrealized_pnl_pct"] = (current_price / avg_entry_price - 1.0) * 100.0
    except Exception:
        pass

    for key, days in RETURN_WINDOWS.items():
        try:
            row[key] = _return_since(df, days)
        except Exception:
            row[key] = None

    return row


def compute_position_metrics(
    bot_logger: Any,
    config: Dict[str, Any],
    is_live: bool,
    infer_asset_class: Callable[[str, Dict[str, Any]], AssetClass],
) -> List[Dict[str, Any]]:
    """Metrics for every currently-open position.

    Simulation reads the ledger directly -- it is the whole truth there. Live reads
    the real broker position for each configured symbol (small list, one call each);
    stop/take/opened_at for a live position come from the same `simulated_positions`
    table when the bot itself is managing that exit (see logger.py's note on that
    table's dual role), and are null when a broker-side bracket owns the exit instead
    -- an honest "not tracked here", never a guessed value.
    """
    rows: List[Dict[str, Any]] = []

    if not is_live:
        for sim in bot_logger.get_all_simulated_positions():
            qty = float(sim["qty"])
            if qty <= 0:
                continue
            symbol = str(sim["symbol"])
            asset_class = infer_asset_class(symbol, config)
            rows.append(
                _position_row(
                    symbol=symbol,
                    asset_class=asset_class,
                    qty=qty,
                    avg_entry_price=float(sim["avg_entry_price"]),
                    stop_loss_price=sim["stop_loss_price"],
                    take_profit_price=sim["take_profit_price"],
                    opened_at=sim["opened_at"],
                )
            )
        return rows

    import execution  # deferred: only live mode needs broker calls at all

    managed = {
        str(r["symbol"]): r for r in bot_logger.get_all_simulated_positions()
    }

    for entry in config.get("symbols", []) or []:
        symbol = entry["symbol"] if isinstance(entry, dict) else str(entry)
        asset_class = infer_asset_class(symbol, config)

        try:
            position = execution.fetch_existing_position(
                symbol=symbol, asset_class=asset_class, is_live=True, bot_logger=bot_logger
            )
        except Exception as exc:  # noqa: BLE001 - one symbol's broker lookup never blocks the rest
            print(f"  [positions] {symbol}: live position lookup failed ({exc}); skipped")
            continue

        if position is None or position.qty <= 0:
            continue

        own = managed.get(symbol)
        rows.append(
            _position_row(
                symbol=symbol,
                asset_class=asset_class,
                qty=position.qty,
                avg_entry_price=position.avg_entry_price,
                stop_loss_price=own["stop_loss_price"] if own else None,
                take_profit_price=own["take_profit_price"] if own else None,
                opened_at=own["opened_at"] if own else None,
            )
        )

    return rows


def export_positions_json(
    rows: List[Dict[str, Any]], path: str = DEFAULT_POSITIONS_PATH, is_live: bool = False
) -> int:
    """Write positions.json. `generated_at` is what the dashboard's dynamic clocks
    and 30-60s polling loop key off of to know whether they have a fresh file.
    """
    payload = {
        "generated_at": utc_now_iso(),
        "is_live": is_live,
        "positions": rows,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return len(rows)


def main() -> int:
    """Standalone entry point: refresh positions.json only. No AI calls, no trading
    logic, no signal generation -- just re-pricing whatever is already open. This is
    the job `refresh_positions.yml` runs every 15 minutes, deliberately separate from
    and far cheaper than main.py's 4h cycle.
    """
    import argparse

    import main as main_module  # deferred to dodge the main.py <-> position_metrics cycle
    from logger import BotLogger
    from mode import resolve_is_live

    parser = argparse.ArgumentParser(description="Refresh positions.json (price-only, no trading).")
    parser.add_argument("--config", default=main_module.DEFAULT_CONFIG_PATH)
    parser.add_argument("--symbols", default=None)
    args = parser.parse_args()

    config = main_module.load_config(args.config, args.symbols)
    is_live = resolve_is_live(config)
    bot_logger = BotLogger(config.get("db_path", "trading_bot.db"))

    rows = compute_position_metrics(bot_logger, config, is_live, main_module.infer_asset_class)
    positions_path = config.get("positions_path", DEFAULT_POSITIONS_PATH)
    export_positions_json(rows, positions_path, is_live=is_live)
    print(f"Refreshed {len(rows)} open positions -> {positions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
