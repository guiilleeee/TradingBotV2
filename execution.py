"""Order routing for Alpaca (US equities) and Hyperliquid (crypto spot).

Three rules run through everything in this module:

  * Real broker APIs are only ever touched when `is_live` is True. Simulation never
    authenticates and never places an order -- it runs the same sizing and the same
    guards, then returns a dry_run result.
  * The system is spot-only. It opens longs and closes them. It never shorts, so a
    sell with nothing held is a skip, not an order.
  * No leverage, ever. Hyperliquid lists 503 perpetual markets alongside its 303
    spot markets on the same exchange id, so every crypto order is checked against
    a resolved spot market before submission. See `resolve_spot_market`.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional, Tuple

import requests

from models import AssetClass, ExecutionResult, ExistingPosition, TradeSignal

# Live endpoint by default. This whole path is already gated behind an explicit
# `live_execution: true`, so silently routing a "live" run to paper would be its
# own kind of wrong. Point ALPACA_BASE_URL at https://paper-api.alpaca.markets to
# rehearse against paper with live_execution on.
ALPACA_BASE_URL = os.environ.get("ALPACA_BASE_URL", "https://api.alpaca.markets")

# Alpaca rejects notional orders below $1.
ALPACA_MIN_NOTIONAL_USD = 1.0
# Limit offset for a bracket's stop_loss leg, as a fraction of the stop price.
STOP_LIMIT_BUFFER = 0.01
HTTP_TIMEOUT = 20.0

# Marks a fill whose exit the bot has to manage itself, because no bracket could be
# attached (a notional/fractional equity entry, or any Hyperliquid spot entry). main.py
# registers these in the ledger so the per-cycle sweep can close them.
MANAGED_EXIT_MARKER = "[managed-exit]"


def needs_managed_exit(result: ExecutionResult) -> bool:
    """True when this fill has no broker-side stop and the sweep must cover it."""
    return MANAGED_EXIT_MARKER in (result.message or "")


# --------------------------------------------------------------------- symbols


class NotSpotMarketError(RuntimeError):
    """Raised when a crypto order would touch anything but a confirmed spot market.

    This is the leverage guard. It is an exception rather than a skip because a
    perpetual market resolving here means the symbol mapping is wrong, and a wrong
    mapping must stop the symbol dead rather than quietly do nothing.
    """


# Hyperliquid spot quotes against USDC. Confirmed against load_markets(): 286 of
# the 303 live spot markets are USDC-quoted, and every symbol we care about is.
# Pinned explicitly because several bases list against more than one quote (BTC has
# both BTC/USDC and BTC/USDH) and picking the wrong one is a silent mis-trade.
HYPERLIQUID_QUOTE = "USDC"

# Internal symbols follow the yfinance convention. Only these suffixes may reach a
# crypto venue -- see to_hyperliquid_symbol for why that matters here specifically.
_CRYPTO_SUFFIXES = ("-USDT", "-USD")


def to_hyperliquid_symbol(symbol: str) -> str:
    """Map an internal yfinance-style symbol to the Hyperliquid/ccxt convention.

    Internally everything is `BTC-USD` because that is what yfinance wants;
    Hyperliquid spot wants `BTC/USDC`. The two do not coincide -- map explicitly,
    and do not assume OKX's old `BASE/USDT` shape carries over.

    The suffix requirement is not cosmetic. Hyperliquid lists tokenized equities as
    genuine spot markets -- `AAPL/USDC` and `MSFT/USDC` both exist and are live --
    so a bare "AAPL" reaching this function would resolve to a real, tradeable spot
    market and pass every spot check downstream. Requiring the crypto suffix is what
    keeps an equity symbol from ever being routed to the crypto venue.
    """
    base = symbol.upper()
    for suffix in _CRYPTO_SUFFIXES:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    else:
        raise NotSpotMarketError(
            f"{symbol!r} does not carry a crypto suffix {_CRYPTO_SUFFIXES}; refusing to map it "
            "to a Hyperliquid market. Hyperliquid lists tokenized equities as real spot "
            "markets (AAPL/USDC, MSFT/USDC), so an equity symbol reaching here would "
            "otherwise resolve to a valid market and trade."
        )
    return f"{base}/{HYPERLIQUID_QUOTE}"


def hyperliquid_base_currency(symbol: str) -> str:
    return to_hyperliquid_symbol(symbol).split("/")[0]


def resolve_spot_market(exchange: Any, hl_symbol: str) -> Dict[str, Any]:
    """Resolve `hl_symbol` and prove it is spot before any order can reference it.

    This is the one guarantee in the crypto path that must never regress silently,
    in the same spirit as the live/simulation split in mode.py. Hyperliquid serves
    spot and perpetual markets from one exchange id -- 303 spot against 503 swap --
    so "it loaded, therefore it is fine" is not good enough. Every condition below
    is checked independently: a market has to fail all of them to be a perp, and
    passing by accident would require the exchange to lie four different ways.
    """
    market = exchange.market(hl_symbol)  # raises BadSymbol if it does not exist

    problems = []
    if market.get("type") != "spot":
        problems.append(f"type={market.get('type')!r}, expected 'spot'")
    if market.get("spot") is not True:
        problems.append(f"spot={market.get('spot')!r}, expected True")
    if market.get("swap") is not False:
        problems.append(f"swap={market.get('swap')!r}, expected False")
    if market.get("contract"):
        problems.append(f"contract={market.get('contract')!r}, expected falsy")
    # Perps carry a settle suffix (BTC/USDC:USDC); spot symbols never do.
    if ":" in str(market.get("symbol", "")):
        problems.append(f"symbol={market.get('symbol')!r} carries a settle suffix")
    if market.get("quote") != HYPERLIQUID_QUOTE:
        problems.append(f"quote={market.get('quote')!r}, expected {HYPERLIQUID_QUOTE!r}")

    if problems:
        raise NotSpotMarketError(
            f"refusing to trade {hl_symbol}: not a confirmed unleveraged spot market "
            f"({'; '.join(problems)})"
        )

    return market


# ----------------------------------------------------------------- credentials


def _alpaca_credentials() -> Tuple[Optional[str], Optional[str]]:
    return os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_API_SECRET")


def _alpaca_headers() -> Dict[str, str]:
    key, secret = _alpaca_credentials()
    return {
        "APCA-API-KEY-ID": key or "",
        "APCA-API-SECRET-KEY": secret or "",
        "Content-Type": "application/json",
    }


def _hyperliquid_credentials() -> Tuple[Optional[str], Optional[str]]:
    """Hyperliquid authenticates with a wallet address and its private key.

    Deliberately not the API-key/secret/passphrase triple OKX used --
    ccxt reports requiredCredentials = {walletAddress, privateKey} for this venue.
    """
    return (
        os.environ.get("HYPERLIQUID_WALLET_ADDRESS"),
        os.environ.get("HYPERLIQUID_PRIVATE_KEY"),
    )


def _hyperliquid_exchange(is_live: bool) -> Any:
    """Build a ccxt Hyperliquid client.

    In simulation this is unauthenticated on purpose: `load_markets`,
    `amount_to_precision` and the info endpoints are public, so sim applies the
    real precision, the real spot check and the real $10 minimum without ever
    holding a key.
    """
    import ccxt

    if not is_live:
        return ccxt.hyperliquid({"enableRateLimit": True})

    wallet, private_key = _hyperliquid_credentials()
    return ccxt.hyperliquid(
        {
            "walletAddress": wallet,
            "privateKey": private_key,
            "enableRateLimit": True,
        }
    )


# ------------------------------------------------------------------- positions


def fetch_existing_position(
    symbol: str,
    asset_class: AssetClass,
    is_live: bool,
    bot_logger: Any,
) -> Optional[ExistingPosition]:
    """Current holding for `symbol`, from the broker in live and the ledger in sim."""
    if not is_live:
        # No broker call at all. The simulated ledger is the whole truth here.
        return bot_logger.get_simulated_position(symbol)

    if asset_class == "equity":
        return _fetch_alpaca_position(symbol)
    return _fetch_hyperliquid_position(symbol, bot_logger)


def _fetch_alpaca_position(symbol: str) -> Optional[ExistingPosition]:
    """Live equity position, or None if Alpaca says there genuinely isn't one.

    A failed lookup deliberately raises rather than returning None. "None" here
    means "no position held", which the duplicate-buy guard reads as permission
    to open one -- so swallowing a broker outage would let the bot double up on a
    position it already has. The caller skips the symbol instead.
    """
    resp = requests.get(
        f"{ALPACA_BASE_URL}/v2/positions/{symbol}",
        headers=_alpaca_headers(),
        timeout=HTTP_TIMEOUT,
    )
    if resp.status_code == 404:
        return None  # Alpaca's explicit "flat in this symbol"
    resp.raise_for_status()

    data = resp.json()
    qty = float(data.get("qty", 0.0))
    # Alpaca returns the cost basis natively, so no reconstruction is needed.
    avg = float(data.get("avg_entry_price", 0.0))
    if qty <= 0 or avg <= 0:
        return None
    return ExistingPosition(qty=qty, avg_entry_price=avg)


def _hyperliquid_spot_balances(exchange: Any, wallet: Optional[str] = None) -> Dict[str, Any]:
    """Spot balances only.

    The `type: spot` parameter is load-bearing. ccxt's fetch_balance for this venue
    defaults to the PERPETUALS account ("wallet type, ['spot','swap'], defaults to
    swap" per its own docstring), so omitting it reads the wrong account entirely.
    """
    params: Dict[str, Any] = {"type": "spot"}
    if wallet:
        params["user"] = wallet
    return exchange.fetch_balance(params)


def _fetch_hyperliquid_position(symbol: str, bot_logger: Any) -> Optional[ExistingPosition]:
    """Live crypto spot position. Raises on lookup failure, for the reason above."""
    exchange = _hyperliquid_exchange(is_live=True)
    wallet, _ = _hyperliquid_credentials()
    balance = _hyperliquid_spot_balances(exchange, wallet)

    base = hyperliquid_base_currency(symbol)
    qty = float((balance.get("total") or {}).get(base, 0.0) or 0.0)
    if qty <= 0:
        return None

    # A spot balance carries no cost basis, so it comes from our own log. This is
    # only sound because the duplicate-buy guard below allows at most one open buy
    # per symbol at a time: "the last buy" is this position.
    entry = bot_logger.get_last_buy_price(symbol)
    if not entry or entry <= 0:
        raise RuntimeError(
            f"{symbol}: Hyperliquid reports {qty:g} {base} held but no logged buy price is "
            "available, so the cost basis is unknown; refusing to trade this symbol blind"
        )
    return ExistingPosition(qty=qty, avg_entry_price=float(entry))


def fetch_live_equity(fallback: float) -> float:
    """Total account equity across both venues, falling back on any failure."""
    total = 0.0
    found = False

    key, secret = _alpaca_credentials()
    if key and secret:
        try:
            resp = requests.get(
                f"{ALPACA_BASE_URL}/v2/account", headers=_alpaca_headers(), timeout=HTTP_TIMEOUT
            )
            resp.raise_for_status()
            total += float(resp.json().get("equity", 0.0))
            found = True
        except Exception:
            pass

    wallet, private_key = _hyperliquid_credentials()
    if wallet and private_key:
        try:
            exchange = _hyperliquid_exchange(is_live=True)
            balance = _hyperliquid_spot_balances(exchange, wallet)
            total += float((balance.get("total") or {}).get(HYPERLIQUID_QUOTE, 0.0) or 0.0)
            found = True
        except Exception:
            pass

    return total if found and total > 0 else fallback


# ------------------------------------------------------------------- guardrails


def _guard(
    signal: TradeSignal, existing_position: Optional[ExistingPosition]
) -> Optional[ExecutionResult]:
    """Duplicate-buy and naked-sell checks. Returns a skip result, or None to proceed."""
    held = existing_position.qty if existing_position else 0.0

    if signal.action == "buy" and held > 0:
        return ExecutionResult(
            status="skipped",
            message=f"position in {signal.symbol} already exists ({held:g} held); not adding",
        )

    if signal.action == "sell" and held <= 0:
        return ExecutionResult(
            status="skipped",
            message=f"nothing to sell in {signal.symbol}; spot-only, so not opening a short",
        )

    return None


def _stop_limit_price(stop_price: float, exit_side: str) -> float:
    """Limit price for a bracket's stop_loss leg, offset in the direction that works.

    The leg that closes a long is a SELL stop, so its limit must sit BELOW the stop
    (x0.99) or it can never fill. The mirror case -- a BUY stop closing a short --
    needs the limit ABOVE (x1.01). We only ever open longs, but hardcoding one
    multiplier for both sides is precisely how that ends up silently wrong.
    """
    if exit_side == "sell":
        return round(stop_price * (1.0 - STOP_LIMIT_BUFFER), 2)
    if exit_side == "buy":
        return round(stop_price * (1.0 + STOP_LIMIT_BUFFER), 2)
    raise ValueError(f"unknown exit side {exit_side!r}")


# --------------------------------------------------------------------- equities


def _execute_equity(
    signal: TradeSignal,
    current_price: float,
    live_equity: float,
    is_live: bool,
    existing_position: Optional[ExistingPosition],
) -> ExecutionResult:
    key, secret = _alpaca_credentials()
    if is_live and not (key and secret):
        return ExecutionResult(
            status="error", message="ALPACA_API_KEY / ALPACA_API_SECRET missing; cannot trade live"
        )

    blocked = _guard(signal, existing_position)
    if blocked:
        return blocked

    if signal.action == "sell":
        # Closing uses what is actually held. Recomputing a size here would try to
        # sell an amount unrelated to the position.
        assert existing_position is not None  # guaranteed by _guard
        qty = existing_position.qty
        body: Dict[str, Any] = {
            "symbol": signal.symbol,
            "side": "sell",
            "type": "market",
            "time_in_force": "day",
            "qty": _format_qty(qty),
        }
        # No bracket on a closing order: it is liquidating, not opening exposure.
        realized = (current_price - existing_position.avg_entry_price) * qty
        return _submit_alpaca(
            body,
            is_live=is_live,
            qty=qty,
            fill_price=current_price,
            realized_pnl_usd=realized,
            entry_price=existing_position.avg_entry_price,
            note=f"close {qty:g} {signal.symbol}",
        )

    # --- buy ---------------------------------------------------------------
    budget_usd = live_equity * (signal.position_size_pct / 100.0)
    whole_shares = math.floor(budget_usd / current_price) if current_price > 0 else 0

    if whole_shares >= 1:
        # Whole shares support bracket legs, so the stop lives at the broker and
        # survives the bot being offline.
        body = {
            "symbol": signal.symbol,
            "side": "buy",
            "type": "market",
            "time_in_force": "gtc",
            "qty": str(whole_shares),
            "order_class": "bracket",
            "take_profit": {"limit_price": round(float(signal.take_profit_price or 0.0), 2)},
            "stop_loss": {
                "stop_price": round(float(signal.stop_loss_price or 0.0), 2),
                # Exit side is a sell, because we are opening a long.
                "limit_price": _stop_limit_price(float(signal.stop_loss_price or 0.0), "sell"),
            },
        }
        return _submit_alpaca(
            body,
            is_live=is_live,
            qty=float(whole_shares),
            fill_price=current_price,
            note=f"open {whole_shares} {signal.symbol} with bracket",
        )

    # Sub-one-share budget. Whole-share-only buying computes to 0 shares here and
    # silently skips every equity trade on a small account -- the exact bug this
    # build exists to not repeat. Use a notional order instead.
    if budget_usd < ALPACA_MIN_NOTIONAL_USD:
        return ExecutionResult(
            status="skipped",
            message=(
                f"{signal.symbol}: budget ${budget_usd:.2f} is below Alpaca's "
                f"${ALPACA_MIN_NOTIONAL_USD:.2f} minimum notional"
            ),
        )

    # Verified against Alpaca's current docs: fractional/notional orders support
    # market, limit, stop and stop_limit with time_in_force=day only, and cannot
    # carry bracket legs. So the entry goes in bare and the exit is bot-managed
    # via the ledger sweep -- better than skipping the trade outright.
    body = {
        "symbol": signal.symbol,
        "side": "buy",
        "type": "market",
        "time_in_force": "day",
        "notional": f"{budget_usd:.2f}",
    }
    qty_est = budget_usd / current_price
    return _submit_alpaca(
        body,
        is_live=is_live,
        qty=qty_est,
        fill_price=current_price,
        note=(
            f"open ${budget_usd:.2f} notional of {signal.symbol} "
            f"(~{qty_est:.6f} sh); no bracket possible on a notional order {MANAGED_EXIT_MARKER}"
        ),
    )


def _format_qty(qty: float) -> str:
    """Alpaca accepts up to 9 decimals; trim trailing zeros so whole lots stay clean."""
    return f"{qty:.9f}".rstrip("0").rstrip(".")


def _submit_alpaca(
    body: Dict[str, Any],
    is_live: bool,
    qty: float,
    fill_price: float,
    note: str,
    realized_pnl_usd: Optional[float] = None,
    entry_price: Optional[float] = None,
) -> ExecutionResult:
    if not is_live:
        return ExecutionResult(
            status="dry_run",
            message=f"[sim] {note}",
            qty=qty,
            fill_price=fill_price,
            realized_pnl_usd=realized_pnl_usd,
        )

    resp = requests.post(
        f"{ALPACA_BASE_URL}/v2/orders",
        headers=_alpaca_headers(),
        json=body,
        timeout=HTTP_TIMEOUT,
    )
    if resp.status_code >= 400:
        return ExecutionResult(
            status="error", message=f"Alpaca rejected the order ({resp.status_code}): {resp.text[:300]}"
        )

    data = resp.json()
    filled_price = data.get("filled_avg_price")
    filled_qty = data.get("filled_qty")

    actual_price = float(filled_price) if filled_price else fill_price
    actual_qty = float(filled_qty) if filled_qty and float(filled_qty) > 0 else qty

    # Recompute realised P&L from what actually filled, not from the pre-trade
    # price. This number feeds the circuit breaker, so an estimate is not enough.
    if entry_price is not None:
        realized_pnl_usd = (actual_price - entry_price) * actual_qty

    return ExecutionResult(
        status="success",
        order_id=str(data.get("id")) if data.get("id") else None,
        fill_price=actual_price,
        qty=actual_qty,
        realized_pnl_usd=realized_pnl_usd,
        message=note,
    )


# ----------------------------------------------------------------------- crypto


def _execute_crypto(
    signal: TradeSignal,
    current_price: float,
    live_equity: float,
    is_live: bool,
    existing_position: Optional[ExistingPosition],
) -> ExecutionResult:
    wallet, private_key = _hyperliquid_credentials()
    if is_live and not (wallet and private_key):
        return ExecutionResult(
            status="error",
            message=(
                "HYPERLIQUID_WALLET_ADDRESS / HYPERLIQUID_PRIVATE_KEY missing; cannot trade live"
            ),
        )

    blocked = _guard(signal, existing_position)
    if blocked:
        return blocked

    # Raises NotSpotMarketError for anything without a crypto suffix, which is what
    # stops an equity symbol reaching a venue that really does list AAPL/USDC.
    hl_symbol = to_hyperliquid_symbol(signal.symbol)

    if signal.action == "sell":
        assert existing_position is not None  # guaranteed by _guard
        qty = existing_position.qty
        realized: Optional[float] = (
            current_price - existing_position.avg_entry_price
        ) * qty
    else:
        qty = (live_equity * (signal.position_size_pct / 100.0)) / current_price
        realized = None

    # Precision, the spot check and the minimum all come from Hyperliquid itself.
    # These endpoints are public, so simulation gets identical rounding, an
    # identical spot guarantee and the identical $10 floor without any credential.
    #
    # Unlike the equity path there is no sim-mode leniency here: if the market
    # cannot be resolved and proven spot, no order is described, not even a
    # simulated one. A dry_run that skipped the leverage check would be a dry_run
    # of the wrong system.
    try:
        exchange = _hyperliquid_exchange(is_live)
        exchange.load_markets()
        market = resolve_spot_market(exchange, hl_symbol)
        qty = float(exchange.amount_to_precision(hl_symbol, qty))
    except NotSpotMarketError:
        raise
    except Exception as exc:
        # A symbol with no spot market at all is a clean skip with a legible
        # reason, not an error. Several assets are perpetual-only on Hyperliquid
        # (BNB and XRP among them) and being spot-only, we simply cannot trade
        # them here -- that should read as a venue limitation in the log, not as
        # a broker malfunction to go debugging.
        if type(exc).__name__ == "BadSymbol" or "does not have market symbol" in str(exc):
            return ExecutionResult(
                status="skipped",
                message=(
                    f"{signal.symbol}: no {hl_symbol} spot market on Hyperliquid "
                    "(likely perpetual-only there); this bot is spot-only, so it cannot "
                    "trade this symbol on this venue"
                ),
            )
        return ExecutionResult(
            status="error",
            message=f"Hyperliquid market lookup failed for {hl_symbol}: {type(exc).__name__}: {exc}",
        )

    if qty <= 0:
        return ExecutionResult(
            status="skipped", message=f"{hl_symbol}: computed size rounded to zero"
        )

    # Hyperliquid's constraint is a $10 minimum NOTIONAL, exposed as limits.cost.min
    # and uniformly 10.0 across all 303 spot markets. limits.amount.min is None on
    # every one of them, so the size-based check used for OKX would be a silent
    # no-op here -- check the notional instead.
    min_cost = (((market.get("limits") or {}).get("cost") or {}).get("min")) or 0.0
    notional = qty * current_price
    if min_cost and notional < float(min_cost):
        return ExecutionResult(
            status="skipped",
            message=(
                f"{hl_symbol}: order notional ${notional:.2f} is under the exchange "
                f"${float(min_cost):.2f} minimum"
            ),
        )

    # Hyperliquid spot has no reliable ccxt-level bracket, so every crypto entry is
    # bot-managed and closed by the ledger sweep.
    note = (
        f"{'close' if signal.action == 'sell' else 'open'} {qty:g} {hl_symbol} spot"
        + (f" {MANAGED_EXIT_MARKER}" if signal.action == "buy" else "")
    )

    if not is_live:
        return ExecutionResult(
            status="dry_run",
            message=f"[sim] {note}",
            qty=qty,
            fill_price=current_price,
            realized_pnl_usd=realized,
        )

    # `amount` is in base currency and the market is the spot one proven above.
    # No reduceOnly, no margin params, nothing that could imply a leveraged leg.
    order = exchange.create_order(hl_symbol, "market", signal.action, qty)
    fill = float(order.get("average") or order.get("price") or current_price)
    filled_qty = float(order.get("filled") or qty)

    # Same reasoning as the equity path: realised P&L comes from the real fill,
    # because the circuit breaker acts on it.
    if signal.action == "sell" and existing_position is not None:
        realized = (fill - existing_position.avg_entry_price) * filled_qty

    return ExecutionResult(
        status="success",
        order_id=str(order.get("id")) if order.get("id") else None,
        fill_price=fill,
        qty=filled_qty,
        realized_pnl_usd=realized,
        message=note,
    )


# ------------------------------------------------------------------ entry point


def execute_trade(
    signal: TradeSignal,
    asset_class: AssetClass,
    current_price: float,
    live_equity: float,
    is_live: bool,
    existing_position: Optional[ExistingPosition] = None,
) -> ExecutionResult:
    """Route one signal to its venue.

    Wrapped end to end: a catastrophic failure on one symbol returns an error
    result for that symbol and never takes the rest of the cycle down with it.
    """
    try:
        if signal.action == "hold":
            return ExecutionResult(status="skipped", message="hold; nothing to execute")

        if asset_class == "equity":
            return _execute_equity(
                signal, current_price, live_equity, is_live, existing_position
            )
        return _execute_crypto(signal, current_price, live_equity, is_live, existing_position)
    except NotSpotMarketError as exc:
        # Surfaced separately from a generic failure so it reads as what it is in
        # the log and the dashboard: the leverage guard refusing a trade, not a
        # broker hiccup. No order was placed either way.
        return ExecutionResult(
            status="error", message=f"{signal.symbol}: SPOT GUARD blocked this order: {exc}"
        )
    except Exception as exc:  # noqa: BLE001 - one symbol must not kill the cycle
        return ExecutionResult(
            status="error", message=f"{signal.symbol}: execution failed: {type(exc).__name__}: {exc}"
        )
