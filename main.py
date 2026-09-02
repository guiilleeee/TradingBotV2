"""One trading cycle, end to end.

Run order matters and is deliberate:

  1. resolve the mode (live vs simulation) -- structurally, not by config accident
  2. sweep open bot-managed positions for stop/take crossings, booking realised P&L
  3. compute equity from what is left open, so a swept position is never counted
     as both realised and unrealised
  4. per symbol: circuit breaker, data, model, risk manager, execution, log
  5. export the dashboard CSV
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import data_fetcher
import execution
import market_intel
import risk_manager
from logger import BotLogger
from mode import ModeSettings, resolve_is_live, resolve_mode_settings
from models import AssetClass, ExistingPosition, SignalInput, TradeSignal

DEFAULT_CONFIG_PATH = "config.yaml"


# --------------------------------------------------------------------- config


def load_config(path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    import yaml  # imported here so importing this module needs no config deps

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def infer_asset_class(symbol: str, config: Dict[str, Any]) -> AssetClass:
    """Asset class for a symbol, preferring the config and falling back on shape.

    The fallback matters for the sweep: a position can outlive its config entry,
    and it still has to be closed on the right venue.
    """
    for entry in config.get("symbols", []) or []:
        if isinstance(entry, dict) and entry.get("symbol") == symbol:
            return "crypto" if entry.get("asset_class") == "crypto" else "equity"
    return "crypto" if symbol.upper().endswith(("-USD", "-USDT")) else "equity"


def get_provider(config: Dict[str, Any]) -> Tuple[str, Any]:
    """Return (name, generate_signal) for the configured provider."""
    name = str(config.get("signal_provider", "claude")).strip().lower()
    if name == "gemini":
        import signal_generator_gemini

        return name, signal_generator_gemini.generate_signal
    if name == "claude":
        import signal_generator

        return name, signal_generator.generate_signal
    raise ValueError(f"unknown signal_provider {name!r}; expected 'claude' or 'gemini'")


# ---------------------------------------------------------------------- sweep


@dataclass
class Closure:
    symbol: str
    reason: str
    price: float
    qty: float
    pnl: float


@dataclass
class SweepResult:
    closed_symbols: Set[str] = field(default_factory=set)
    closures: List[Closure] = field(default_factory=list)
    unrealized_pnl: float = 0.0


def sweep_open_positions(
    bot_logger: BotLogger,
    config: Dict[str, Any],
    is_live: bool,
    equity_hint: float,
) -> SweepResult:
    """Close any bot-managed position whose price has crossed its stop or target.

    One pass, before any new symbol is considered. It does two jobs at once on
    purpose: a position it closes books its P&L as realised via `record_pnl`, and
    only the positions it leaves open contribute to `unrealized_pnl`. Splitting
    these into two passes is how the same dollar ends up counted twice.

    Synthetic log rows are not written here -- the caller writes them once equity
    is known, so every row carries a real "Patrimoni total" value.
    """
    result = SweepResult()

    for row in bot_logger.get_all_simulated_positions():
        symbol = str(row["symbol"])
        qty = float(row["qty"])
        entry = float(row["avg_entry_price"])
        stop = row["stop_loss_price"]
        take = row["take_profit_price"]

        if qty <= 0:
            bot_logger.close_simulated_position(symbol)
            continue

        try:
            price = data_fetcher.latest_price(data_fetcher.fetch_ohlcv(symbol))
        except Exception as exc:  # noqa: BLE001
            print(f"  [sweep] {symbol}: price unavailable ({exc}); leaving position open")
            result.unrealized_pnl += 0.0
            continue

        # Long-only, so a stop is crossed from above and a target from below.
        # If a gap crosses both in one bar, assume the worse of the two.
        hit_stop = stop is not None and price <= float(stop)
        hit_take = take is not None and price >= float(take)

        if not (hit_stop or hit_take):
            result.unrealized_pnl += (price - entry) * qty
            continue

        if hit_stop:
            reason = (
                f"Stop-loss activat: el preu {price:.6g} ha creuat el nivell "
                f"{float(stop):.6g}. Posicio tancada automaticament."
            )
        else:
            reason = (
                f"Take-profit assolit: el preu {price:.6g} ha superat l'objectiu "
                f"{float(take):.6g}. Posicio tancada automaticament."
            )

        fill = price
        if is_live:
            # Live managed exits place a real closing order. If it does not fill,
            # the position stays on the books and stays unrealised.
            exit_signal = TradeSignal(
                symbol=symbol,
                action="sell",
                confidence=1.0,
                position_size_pct=0.0,
                stop_loss_price=None,
                take_profit_price=None,
                reasoning=reason,
                override_reason="automatic exit",
                raw_action="sell",
            )
            exec_result = execution.execute_trade(
                signal=exit_signal,
                asset_class=infer_asset_class(symbol, config),
                current_price=price,
                live_equity=equity_hint,
                is_live=True,
                existing_position=ExistingPosition(qty=qty, avg_entry_price=entry),
            )
            if exec_result.status != "success":
                print(f"  [sweep] {symbol}: managed exit did not fill ({exec_result.message})")
                result.unrealized_pnl += (price - entry) * qty
                continue
            fill = float(exec_result.fill_price or price)
            qty = float(exec_result.qty or qty)

        pnl = (fill - entry) * qty
        bot_logger.record_pnl(symbol, pnl)
        bot_logger.close_simulated_position(symbol)
        result.closed_symbols.add(symbol)
        result.closures.append(
            Closure(symbol=symbol, reason=reason, price=fill, qty=qty, pnl=pnl)
        )
        print(f"  [sweep] {symbol}: closed at {fill:.6g}, P&L {pnl:+.2f} USD")

    return result


# ----------------------------------------------------------------------- cycle


def run_cycle(config_path: str = DEFAULT_CONFIG_PATH) -> int:
    config = load_config(config_path)
    is_live = resolve_is_live(config)
    settings: ModeSettings = resolve_mode_settings(is_live, config)

    bot_logger = BotLogger(config.get("db_path", "trading_bot.db"))
    provider_name, generate_signal = get_provider(config)

    fallback_equity = float(config.get("fallback_equity_usd", 1000.0))
    circuit_breaker_loss_pct = float(config.get("circuit_breaker_loss_pct", 3.0))
    max_risk_pct = float(config.get("max_risk_pct", 1.0))
    max_absolute_position_pct = float(config.get("max_absolute_position_pct", 20.0))

    print(f"=== TradingBot cycle | mode={settings.label} | provider={provider_name} ===")
    print(f"min_confidence={settings.min_confidence:.2f}  max_risk={max_risk_pct:.2f}%  "
          f"cap={max_absolute_position_pct:.2f}%  breaker=-{circuit_breaker_loss_pct:.2f}%")

    # --- equity + sweep ---------------------------------------------------
    if is_live:
        equity = execution.fetch_live_equity(fallback_equity)
        sweep = sweep_open_positions(bot_logger, config, is_live, equity)
        if sweep.closures:
            equity = execution.fetch_live_equity(fallback_equity)
    else:
        sweep = sweep_open_positions(bot_logger, config, is_live, fallback_equity)
        # Realised P&L already includes anything the sweep just booked, and only
        # still-open positions contributed unrealised. No dollar is counted twice.
        equity = (
            fallback_equity
            + bot_logger.get_all_time_realized_pnl()
            + sweep.unrealized_pnl
        )

    equity = max(equity, 0.01)  # keep downstream gt=0 validators satisfiable

    for closure in sweep.closures:
        bot_logger.log_auto_close_signal(
            symbol=closure.symbol,
            reason=closure.reason,
            price=closure.price,
            qty=closure.qty,
            pnl=closure.pnl,
            equity=equity,
        )

    print(f"Equity: ${equity:,.2f}  (auto-closed this cycle: "
          f"{sorted(sweep.closed_symbols) or 'none'})")

    # --- per symbol -------------------------------------------------------
    for entry in config.get("symbols", []) or []:
        symbol = entry["symbol"] if isinstance(entry, dict) else str(entry)
        asset_class = infer_asset_class(symbol, config)

        if symbol in sweep.closed_symbols:
            print(f"- {symbol}: auto-closed this cycle, not re-evaluating")
            continue

        try:
            _process_symbol(
                symbol=symbol,
                asset_class=asset_class,
                config=config,
                settings=settings,
                bot_logger=bot_logger,
                generate_signal=generate_signal,
                equity=equity,
                circuit_breaker_loss_pct=circuit_breaker_loss_pct,
                max_risk_pct=max_risk_pct,
                max_absolute_position_pct=max_absolute_position_pct,
            )
        except Exception as exc:  # noqa: BLE001 - one symbol never kills the cycle
            print(f"- {symbol}: FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc(file=sys.stdout)

    # --- export -----------------------------------------------------------
    csv_path = config.get("csv_path", "signals.csv")
    rows = bot_logger.export_signals_csv(csv_path)
    print(f"Exported {rows} signal rows to {csv_path}")
    return 0


def _process_symbol(
    symbol: str,
    asset_class: AssetClass,
    config: Dict[str, Any],
    settings: ModeSettings,
    bot_logger: BotLogger,
    generate_signal: Any,
    equity: float,
    circuit_breaker_loss_pct: float,
    max_risk_pct: float,
    max_absolute_position_pct: float,
) -> None:
    is_live = settings.is_live

    df = data_fetcher.fetch_ohlcv(symbol)
    indicators = data_fetcher.compute_indicators(df)
    current_price = data_fetcher.latest_price(df)

    existing_position: Optional[ExistingPosition] = execution.fetch_existing_position(
        symbol=symbol, asset_class=asset_class, is_live=is_live, bot_logger=bot_logger
    )

    # Positioning context is an input to the model's reasoning and nothing more.
    # It returns None for equities and on any failure, and the signal it produces
    # still has to clear risk_manager.validate() in full like any other.
    positioning = None
    if config.get("use_market_positioning", True):
        positioning = market_intel.fetch_positioning(symbol, asset_class)

    signal_input = SignalInput(
        symbol=symbol,
        asset_class=asset_class,
        current_price=current_price,
        account_equity_usd=equity,
        existing_position=existing_position,
        technical_indicators=indicators,
        recent_headlines=data_fetcher.fetch_headlines(symbol),
        market_positioning=positioning,
    )

    # Recomputed per symbol so a loss taken earlier in this cycle can still trip
    # the breaker for the symbols that follow.
    today_loss_pct = bot_logger.get_today_realized_loss_pct(equity)

    if today_loss_pct <= -abs(circuit_breaker_loss_pct):
        # Breaker is already tripped, so skip the model call entirely -- there is
        # no decision it could return that we would act on, and it costs money.
        blocked = TradeSignal(
            symbol=symbol,
            action="hold",
            confidence=0.0,
            position_size_pct=0.0,
            stop_loss_price=None,
            take_profit_price=None,
            reasoning=(
                f"Circuit breaker actiu: la perdua realitzada d'avui ({today_loss_pct:.2f}%) "
                f"supera el limit de -{abs(circuit_breaker_loss_pct):.2f}%. "
                "No s'ha consultat el model."
            ),
            override_reason=(
                f"circuit breaker tripped at {today_loss_pct:.2f}%; model call skipped"
            ),
            raw_action="hold",
        )
        bot_logger.log_signal(symbol, signal_input, None, blocked, None)
        print(f"- {symbol}: circuit breaker tripped ({today_loss_pct:.2f}%); skipped")
        return

    raw = generate_signal(signal_input, system_prompt=settings.system_prompt)

    final = risk_manager.validate(
        raw=raw,
        current_price=current_price,
        today_realized_loss_pct=today_loss_pct,
        circuit_breaker_loss_pct=circuit_breaker_loss_pct,
        max_risk_pct=max_risk_pct,
        max_absolute_position_pct=max_absolute_position_pct,
        min_confidence=settings.min_confidence,
    )

    exec_result = None
    if final.action != "hold":
        exec_result = execution.execute_trade(
            signal=final,
            asset_class=asset_class,
            current_price=current_price,
            live_equity=equity,
            is_live=is_live,
            existing_position=existing_position,
        )

    bot_logger.log_signal(symbol, signal_input, raw, final, exec_result)

    if exec_result and exec_result.status in ("success", "dry_run"):
        if exec_result.realized_pnl_usd is not None:
            # Without this call the circuit breaker never sees a loss and is inert.
            bot_logger.record_pnl(symbol, float(exec_result.realized_pnl_usd))

        _update_ledger(bot_logger, final, exec_result, current_price, is_live)

    print(
        f"- {symbol}: {final.raw_action} -> {final.action} "
        f"(conf {final.confidence:.2f}, size {final.position_size_pct:.2f}%)"
        + (f" | override: {final.override_reason}" if final.override_reason else "")
        + (f" | exec: {exec_result.status} - {exec_result.message}" if exec_result else "")
    )


def _update_ledger(
    bot_logger: BotLogger,
    final: TradeSignal,
    exec_result: Any,
    current_price: float,
    is_live: bool,
) -> None:
    """Keep the bot-managed ledger in step with what just happened.

    In simulation it tracks every position. In live it tracks only fills with no
    broker-side bracket, which are the ones the sweep has to exit.
    """
    if final.action == "sell":
        bot_logger.close_simulated_position(final.symbol)
        return

    if final.action != "buy":
        return

    if is_live and not execution.needs_managed_exit(exec_result):
        # A live bracket order already carries its own stop and target at the
        # broker; tracking it here would double up the exit.
        return

    bot_logger.open_simulated_position(
        symbol=final.symbol,
        qty=float(exec_result.qty or 0.0),
        avg_entry_price=float(exec_result.fill_price or current_price),
        stop_loss_price=final.stop_loss_price,
        take_profit_price=final.take_profit_price,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one TradingBot cycle.")
    parser.add_argument("--config", default=os.environ.get("BOT_CONFIG", DEFAULT_CONFIG_PATH))
    args = parser.parse_args()
    return run_cycle(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
