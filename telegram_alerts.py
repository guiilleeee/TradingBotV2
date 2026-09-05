"""Telegram push notifications: a convenience channel, never load-bearing.

Same philosophy as headlines (data_fetcher.py) and positioning data
(market_intel.py): this fails soft, always. A missing credential, a network
error, a bad response from Telegram -- every one of these is logged locally and
swallowed, never raised. Nothing in this module can affect a cycle's return
value, a screening run's outcome, or any trading decision. If
TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is unset, every send is a silent no-op,
the same pattern as Alpaca/Hyperliquid credentials being optional in
simulation -- Telegram is entirely opt-in.

Every message this module sends goes through `format_alert`, which is the only
place the mode label is ever written. No call site builds its own header: it
must be structurally impossible for an alert to reach Telegram without an
unambiguous SIMULACIO/REAL prefix, the same discipline mode.py applies to the
live/simulation split itself.
"""

from __future__ import annotations

import os
from typing import Iterable

import requests

from secrets_redaction import SECRET_ENV_VARS as _SECRET_ENV_VARS
from secrets_redaction import sanitize as _sanitize

TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"
HTTP_TIMEOUT = 15.0

# Deliberately plain ASCII, no accents -- matches this project's established
# house style for user-facing Catalan text (see risk_manager.py / main.py's
# reasoning strings: "perdua", "limit", not "pèrdua", "límit").
SIMULATION_LABEL = "\U0001f7e2 SIMULACIO"  # green circle
LIVE_LABEL = "\U0001f534 REAL"  # red circle

# Telegram's own hard cap is 4096 chars; this is a much tighter proactive
# limit so a runaway error message still reads as a notification, not a wall
# of text, on a phone lock screen.
MAX_MESSAGE_LENGTH = 800

# _sanitize / _SECRET_ENV_VARS re-exported (not redefined) from
# secrets_redaction.py -- that module is now the single source of truth, used
# by execution.py and equity_universe.py too. Kept under these names here so
# every existing caller and test in this file is unaffected by the move.


def _truncate(text: str, limit: int = MAX_MESSAGE_LENGTH) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"  # ellipsis


# ------------------------------------------------------------------- core


def format_alert(is_live: bool, body: str) -> str:
    """Prefix `body` with an unmistakable mode label. The only place this happens.

    The label is always the first thing in the message, before any other
    content -- a phone notification preview, which often truncates after a
    handful of characters, must show the mode even if nothing else survives.
    """
    label = LIVE_LABEL if is_live else SIMULATION_LABEL
    return f"{label}\n{body}"


def send_alert(text: str) -> None:
    """POST `text` to Telegram. Never raises, never returns anything meaningful.

    A missing credential is the expected state for anyone who hasn't set up
    Telegram, not an error -- exactly like an unset ALPACA_API_KEY in
    simulation. A network failure or a bad response is logged and swallowed:
    a notification channel being down must never be why a trading cycle or a
    screening run fails.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return

    safe_text = _truncate(_sanitize(text))

    try:
        resp = requests.post(
            TELEGRAM_API_URL.format(token=token),
            json={"chat_id": chat_id, "text": safe_text},
            timeout=HTTP_TIMEOUT,
        )
        if resp.status_code >= 400:
            print(
                "[telegram_alerts] Telegram rejected the message "
                f"({resp.status_code}): {_sanitize(resp.text)[:200]}"
            )
    except Exception as exc:  # noqa: BLE001 - a notification must never break a cycle
        print(
            f"[telegram_alerts] failed to send alert: "
            f"{type(exc).__name__}: {_sanitize(str(exc))[:200]}"
        )


# ------------------------------------------------------------- event alerts
#
# Every function below builds one Catalan message body and sends it through
# format_alert + send_alert. Callers in main.py / screening.py never build a
# message or a mode label themselves -- this is the one and only place either
# happens, so there is no ad-hoc call site to drift out of sync.


def send_trade_alert(
    is_live: bool,
    symbol: str,
    action: str,
    size_pct: float,
    price: float,
    confidence: float,
    reasoning: str,
) -> None:
    """A buy or sell that actually executed (status success or dry_run)."""
    emoji = "\U0001f4c8" if action == "buy" else "\U0001f4c9"  # chart up / down
    verb = "COMPRA" if action == "buy" else "VENDA"
    one_line = (reasoning or "").splitlines()[0] if reasoning else ""
    body = (
        f"{emoji} {verb} {symbol}\n"
        f"Mida: {size_pct:.2f}% | Preu: {price:.6g} | Confianca: {confidence:.2f}\n"
        f"{one_line}"
    )
    send_alert(format_alert(is_live, body))


def send_auto_close_alert(
    is_live: bool, symbol: str, reason: str, realized_pnl_usd: float
) -> None:
    """A stop-loss or take-profit auto-close from the per-cycle sweep."""
    body = (
        f"\U0001f514 Tancament automatic: {symbol}\n"
        f"{reason}\n"
        f"P&L realitzat: {realized_pnl_usd:+.2f} USD"
    )
    send_alert(format_alert(is_live, body))


def send_circuit_breaker_alert(
    is_live: bool, today_loss_pct: float, threshold_pct: float
) -> None:
    """The circuit breaker transitioning from not-tripped to tripped, once."""
    body = (
        "⛔ Circuit breaker activat\n"
        f"Perdua realitzada avui: {today_loss_pct:.2f}% "
        f"(limit: -{abs(threshold_pct):.2f}%)\n"
        "No es realitzaran mes operacions fins que reiniciï el dia (UTC)."
    )
    send_alert(format_alert(is_live, body))


def send_cycle_failure_alert(is_live: bool, error_summary: str) -> None:
    """The whole run_cycle raising -- not a single symbol failing."""
    body = f"\U0001f198 El cicle de trading ha fallat\n{error_summary}"
    send_alert(format_alert(is_live, body))


def send_screening_complete_alert(
    is_live: bool, equity_symbols: Iterable[str], crypto_symbols: Iterable[str]
) -> None:
    """The weekly screen produced a new 10-symbol list."""
    body = (
        "\U0001f4cb Cribratge setmanal completat\n"
        f"Accions: {', '.join(equity_symbols)}\n"
        f"Cripto: {', '.join(crypto_symbols)}"
    )
    send_alert(format_alert(is_live, body))


def send_screening_failure_alert(is_live: bool, reason: str) -> None:
    """The weekly screen failed and left last week's symbols.yaml untouched.

    This never blocks trading -- the 4h cycle keeps working on the old list --
    but it is worth a human noticing, since otherwise it fails silently.
    """
    body = (
        "⚠️ El cribratge setmanal ha fallat\n"
        f"{reason}\n"
        "El cicle de 4h seguira utilitzant la llista de simbols de la setmana anterior."
    )
    send_alert(format_alert(is_live, body))
