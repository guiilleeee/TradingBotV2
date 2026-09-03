"""Weekly symbol screening: pick 10 candidates for the 4h cycle to reason about.

This job never trades and never touches the model. It only decides what the
4h cycle gets to *analyse* -- the AI's own judgment (system prompt rules 4/5:
no volume confirmation, conflicting signals -> hold) and risk_manager.py remain
the only things that can turn a candidate into an actual order. Nothing written
here can widen or bypass either.

Two-layer selection, mirrored across asset classes:

  Layer 1 (objective, no AI): liquidity. Equities are filtered by real trading
    volume measured directly against the S&P 500 + Nasdaq universe (yfinance,
    one batched download -- see equity_universe.fetch_universe_price_data);
    crypto is filtered by real 24h notional volume pulled directly from
    Hyperliquid. Hyperliquid's spot listing is permissionless, so for crypto
    this is the actual first line of defense against thin or scam tokens --
    for equities, S&P 500 / Nasdaq membership already excludes almost all of
    that, so this layer is closer to a formality.
  Layer 2 (trending, still no AI): momentum for equities, from the same
    yfinance batch as Layer 1's volume (an earlier version scored this from
    FMP's whole-market gainers/losers lists, which almost never overlap a
    curated 503-name index universe -- see equity_universe.py's module
    docstring for the diagnosis); aggregate top-wallet positioning for crypto
    (market_intel.py, reused as-is).

Blast radius: this entire module can fail in any way and the 4h cycle is
unaffected -- `main.load_config` falls back to whatever `symbols.yaml` (or
`config.yaml`) already has. `run_screening` enforces that on the writing side:
it only ever replaces `symbols.yaml` after producing a complete, valid 10-symbol
result, and never partially or emptily overwrites it.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List

import equity_universe
import execution
import market_intel
import telegram_alerts

DEFAULT_OUTPUT_PATH = "symbols.yaml"
DEFAULT_CONFIG_PATH_FOR_ALERTS = "config.yaml"

EQUITY_COUNT = 5
CRYPTO_COUNT = 5

# Below this, a Hyperliquid spot market is thin enough that it does not belong
# in front of the model at all. This is the primary anti-scam-token defense for
# crypto, unlike the equity side where index membership does most of that work.
MIN_CRYPTO_VOLUME_USD = 50_000.0

# A market whose base currency is a stablecoin has no directional trading value
# for this strategy. This gate is applied before scoring, so a stablecoin-base
# pair can never appear in crypto_scored regardless of its volume or positioning.
# Checked case-insensitively so "usdt" and "USDT" both match.
# Sources: USDT/USDC are Hyperliquid's primary quote assets and do trade as spot
# base currencies; BUSD/TUSD/FDUSD/PYUSD are legacy or issuer stablecoins;
# DAI/FRAX/LUSD/USDD/USDE/USDS/USDP are decentralised/yield stablecoins that
# have appeared or may appear as base currencies on permissionless spot venues.
STABLECOIN_BASES: frozenset[str] = frozenset({
    "USDT", "USDC", "DAI", "BUSD", "TUSD", "FDUSD",
    "PYUSD", "USDP", "FRAX", "LUSD", "USDD", "USDE", "USDS",
})

# Same 0.6/0.4 liquidity-over-momentum split used for equities in
# equity_universe.py, kept consistent across asset classes on purpose.
VOLUME_WEIGHT = 0.6
POSITIONING_WEIGHT = 0.4

# How many top wallets to sample for the aggregate positioning signal. Larger
# than market_intel's per-symbol default (15) because this is a one-shot,
# once-a-week, whole-universe view rather than a per-cycle per-symbol lookup --
# a bigger sample is worth the extra ~30s here.
CRYPTO_LEADERBOARD_SAMPLE = 50


# ------------------------------------------------------------------ crypto


def build_crypto_universe(exchange: Any) -> Dict[str, Dict[str, Any]]:
    """Every Hyperliquid market that passes execution.py's own spot check.

    Deliberately calls execution.resolve_spot_market for every candidate rather
    than re-deriving "is this spot" from the market dict here -- the guarantee
    that no crypto symbol reaches this system without being provably spot lives
    in exactly one place (execution.py), and this function borrows it rather
    than risking a second implementation drifting from the first.
    """
    exchange.load_markets()
    universe: Dict[str, Dict[str, Any]] = {}
    for hl_symbol in list(exchange.markets.keys()):
        try:
            market = execution.resolve_spot_market(exchange, hl_symbol)
        except execution.NotSpotMarketError:
            continue
        universe[hl_symbol] = market
    return universe


def fetch_crypto_volumes(exchange: Any, hl_symbols: List[str]) -> Dict[str, float]:
    """24h notional (USD) volume per Hyperliquid symbol, one bulk call.

    Verified live: `spotMetaAndAssetCtxs` returns a `ctxs` array whose entries
    carry their own `coin` id, and every ccxt spot market's `market['id']`
    matches one of those coin ids exactly (confirmed 303/303 on the live venue)
    -- so this is a single request for the whole universe's volume, not one
    call per symbol.
    """
    payload = market_intel.post_info({"type": "spotMetaAndAssetCtxs"})
    _, ctxs = payload[0], payload[1]
    ctx_by_coin = {c["coin"]: c for c in ctxs if "coin" in c}

    volumes: Dict[str, float] = {}
    for hl_symbol in hl_symbols:
        market = exchange.markets.get(hl_symbol) or {}
        ctx = ctx_by_coin.get(market.get("id"))
        if ctx is None:
            continue
        try:
            volumes[hl_symbol] = float(ctx.get("dayNtlVlm", 0.0))
        except (TypeError, ValueError):
            continue
    return volumes


def fetch_crypto_positioning(limit: int = CRYPTO_LEADERBOARD_SAMPLE) -> Dict[str, Dict[str, float]]:
    """Aggregate top-wallet positioning per coin, reusing market_intel.py as-is.

    Same leaderboard, same cache file, same aggregation function the 4h cycle
    already uses for its per-symbol positioning note -- not a second fetcher.
    Degrades to {} on any failure; a missing Layer-2 signal just means every
    crypto candidate scores 0 on that half of the blend, never a crash.
    """
    try:
        rows = market_intel.fetch_leaderboard()
        wallets = market_intel.top_wallets(rows, limit=limit)
        return market_intel.aggregate_positioning(wallets) if wallets else {}
    except Exception:
        return {}


def _internal_crypto_symbol(market: Dict[str, Any]) -> str:
    """Hyperliquid/ccxt market -> this project's internal symbol convention.

    Internally everything is `BTC-USD` (yfinance's convention, used for price
    and indicator data) even though Hyperliquid spot actually quotes against
    USDC -- execution.to_hyperliquid_symbol already performs exactly this
    USD-in / USDC-out translation for orders, so this is its mirror image for
    screening output.
    """
    return f"{market['base']}-USD"


def score_crypto(
    universe: Dict[str, Dict[str, Any]],
    volumes: Dict[str, float],
    positioning: Dict[str, Dict[str, float]],
) -> List[Dict[str, Any]]:
    """Rank the spot universe by liquidity (Layer 1) and wallet positioning (Layer 2).

    Mirrors equity_universe.score_equities in shape and weighting on purpose --
    same volume-primary blend, just fed from Hyperliquid instead of FMP.

    Two objective, pre-scoring gates are applied before any ranking:
      1. Volume floor  -- markets below MIN_CRYPTO_VOLUME_USD are dropped.
      2. Stablecoin gate -- markets whose base currency is in STABLECOIN_BASES
         are dropped. A stablecoin-vs-stablecoin pair has no directional trading
         value for this strategy and must never reach the model, regardless of
         its volume or positioning numbers.
    """
    survivors = {
        hl_symbol: vol
        for hl_symbol, vol in volumes.items()
        if hl_symbol in universe
        and vol >= MIN_CRYPTO_VOLUME_USD
        and universe[hl_symbol]["base"].upper() not in STABLECOIN_BASES
    }

    positioning_notional: Dict[str, float] = {}
    for hl_symbol, market in universe.items():
        coin = market["base"]
        bucket = positioning.get(coin)
        if bucket:
            positioning_notional[hl_symbol] = bucket.get("long", 0.0) + bucket.get("short", 0.0)

    volume_pct = equity_universe.percentile_ranks(survivors)
    positioning_pct = equity_universe.percentile_ranks(
        {k: v for k, v in positioning_notional.items() if k in survivors}
    )

    results = []
    for hl_symbol, market in universe.items():
        if hl_symbol not in survivors:
            continue  # below the liquidity floor -- the crypto anti-scam gate
        v = volume_pct.get(hl_symbol, 0.0)
        p = positioning_pct.get(hl_symbol, 0.0)
        results.append(
            {
                "symbol": _internal_crypto_symbol(market),
                "hl_symbol": hl_symbol,
                "score": VOLUME_WEIGHT * v + POSITIONING_WEIGHT * p,
                "volume_usd": survivors[hl_symbol],
                "positioning_usd": positioning_notional.get(hl_symbol),
            }
        )

    results.sort(key=lambda r: (r["score"], r["symbol"]), reverse=True)
    return results


# ------------------------------------------------------------------ output


def _entries(symbols: List[str], asset_class: str) -> List[Dict[str, str]]:
    return [{"symbol": s, "asset_class": asset_class} for s in symbols]


def _current_is_live_for_alerts(config_path: str = DEFAULT_CONFIG_PATH_FOR_ALERTS) -> bool:
    """Best-effort is_live, read only to label this run's Telegram alerts.

    Screening itself has no live/sim behavior of its own -- it does the same
    universe-building and scoring regardless of live_execution. This exists
    purely so its alerts carry the same unmistakable mode label every other
    alert in this project carries (the brief's "every single alert" is read
    literally here). Defaults to False (simulation) on any failure to read or
    parse config.yaml -- the same fail-safe default mode.py itself uses, kept
    consistent rather than inventing a different default for a notification.
    """
    try:
        import yaml

        import mode

        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        return mode.resolve_is_live(config)
    except Exception:
        return False


def run_screening(
    output_path: str = DEFAULT_OUTPUT_PATH,
    config_path: str = DEFAULT_CONFIG_PATH_FOR_ALERTS,
) -> int:
    """Build the 10-symbol slate and write it, or leave the existing file alone.

    Returns 0 on a complete, valid write; 1 on anything short of that. A
    non-zero return must never come with a partial or empty write -- the whole
    point of writing to a temp path first is that a crash midway through
    leaves last week's symbols.yaml exactly as it was.
    """
    is_live = _current_is_live_for_alerts(config_path)

    def fail(reason: str) -> int:
        print(f"FAILED: {reason} Leaving {output_path} untouched.")
        telegram_alerts.send_screening_failure_alert(is_live, reason)
        return 1

    try:
        print("=== Weekly symbol screening ===")

        print("Building equity universe (S&P 500 + Nasdaq)...")
        equity_pool = equity_universe.build_equity_universe()
        print(f"  universe: {len(equity_pool)} symbols")
        if len(equity_pool) < EQUITY_COUNT:
            return fail(
                f"equity universe has only {len(equity_pool)} symbols, need at "
                f"least {EQUITY_COUNT}."
            )

        print("  fetching universe-wide volume/momentum (one batched yfinance call)...")
        price_data = equity_universe.fetch_universe_price_data(sorted(equity_pool))
        print(f"  usable price data for {len(price_data)}/{len(equity_pool)} universe symbols")
        equity_scored = equity_universe.score_equities(equity_pool, price_data)
        equity_symbols = equity_universe.select_top_equities(equity_scored, EQUITY_COUNT)
        signal_count = sum(1 for r in equity_scored if r["has_signal"])
        print(f"  {signal_count} symbols carried real volume/momentum signal this week")
        print(f"  selected: {equity_symbols}")

        print("Building crypto universe (Hyperliquid spot)...")
        exchange = execution._hyperliquid_exchange(is_live=False)
        crypto_universe = build_crypto_universe(exchange)
        print(f"  universe: {len(crypto_universe)} confirmed spot markets")
        if len(crypto_universe) < CRYPTO_COUNT:
            return fail(
                f"crypto universe has only {len(crypto_universe)} spot markets, "
                f"need at least {CRYPTO_COUNT}."
            )

        volumes = fetch_crypto_volumes(exchange, list(crypto_universe.keys()))
        positioning = fetch_crypto_positioning()
        crypto_scored = score_crypto(crypto_universe, volumes, positioning)
        print(f"  {len(crypto_scored)} markets cleared the ${MIN_CRYPTO_VOLUME_USD:,.0f} "
              f"liquidity floor and stablecoin gate")
        if len(crypto_scored) < CRYPTO_COUNT:
            return fail(
                f"only {len(crypto_scored)} crypto markets cleared the liquidity floor, "
                f"need at least {CRYPTO_COUNT}."
            )
        crypto_symbols = [r["symbol"] for r in crypto_scored[:CRYPTO_COUNT]]
        print(f"  selected: {crypto_symbols}")

    except Exception as exc:  # noqa: BLE001 - any failure here must not touch the file
        print(f"FAILED: screening raised {type(exc).__name__}: {exc}")
        traceback.print_exc(file=sys.stdout)
        print(f"Leaving {output_path} untouched -- the 4h cycle keeps last week's list.")
        telegram_alerts.send_screening_failure_alert(is_live, f"{type(exc).__name__}: {exc}")
        return 1

    _write_symbols_file(output_path, equity_symbols, crypto_symbols, equity_scored, crypto_scored)
    print(f"Wrote {output_path}: {len(equity_symbols)} equity + {len(crypto_symbols)} crypto")
    telegram_alerts.send_screening_complete_alert(is_live, equity_symbols, crypto_symbols)
    return 0


def _write_symbols_file(
    output_path: str,
    equity_symbols: List[str],
    crypto_symbols: List[str],
    equity_scored: List[Dict[str, Any]],
    crypto_scored: List[Dict[str, Any]],
) -> None:
    """Atomic write: a temp file plus a rename, so a crash mid-write can never
    leave symbols.yaml half-written or truncated.
    """
    import os
    import yaml

    equity_by_symbol = {r["symbol"]: r for r in equity_scored}
    crypto_by_symbol = {r["symbol"]: r for r in crypto_scored}

    document = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbols": _entries(equity_symbols, "equity") + _entries(crypto_symbols, "crypto"),
        # Informational only -- main.py's loader reads nothing but "symbols"
        # above, so nothing here can ever affect what the 4h cycle trades.
        "scores": {
            "equity": {
                s: {"score": round(equity_by_symbol[s]["score"], 4)}
                for s in equity_symbols
            },
            "crypto": {
                s: {"score": round(crypto_by_symbol[s]["score"], 4)}
                for s in crypto_symbols
            },
        },
    }

    tmp_path = f"{output_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        handle.write(
            "# Generated weekly by screening.py -- do not hand-edit, it is "
            "overwritten every run.\n"
            "# Contributes ONLY the `symbols` list to the 4h cycle; every risk "
            "parameter, threshold, and provider setting stays in config.yaml.\n"
        )
        yaml.safe_dump(document, handle, sort_keys=False)
    os.replace(tmp_path, output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the weekly symbol screen.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH_FOR_ALERTS)
    args = parser.parse_args()
    return run_screening(args.output, args.config)


if __name__ == "__main__":
    raise SystemExit(main())
