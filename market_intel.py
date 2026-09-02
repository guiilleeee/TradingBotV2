"""Public Hyperliquid trader-positioning intelligence.

What this is: a small, factual, human-readable sentence about how large successful
Hyperliquid wallets are currently positioned in an asset, handed to the model as
one more input alongside price, indicators and headlines.

What this is emphatically NOT: a copy-trading mechanism. Nothing here decides
anything. The output is a string on SignalInput, and the only thing that ever reads
it is the model's own reasoning. There is no path from this module to an order that
does not pass through risk_manager.validate() exactly like every other signal --
same confidence threshold, same stop-loss requirement, same sizing, same caps.

One honest caveat is baked into the wording it produces: the positions being read
are Hyperliquid PERPETUAL positions, because that is the only positioning data the
venue exposes. We trade spot without leverage. The summary says "perps" out loud
rather than implying these wallets hold spot.

Every function here fails soft. Positioning data is a nice-to-have; a cycle must
never stop because a leaderboard was slow.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

INFO_URL = "https://api.hyperliquid.xyz/info"
# The leaderboard is NOT on the /info endpoint -- {"type": "leaderboard"} there
# returns 422. It lives on a separate stats host, verified against the live API.
LEADERBOARD_URL = "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard"

_HEADERS = {"Content-Type": "application/json"}
HTTP_TIMEOUT = 30.0

# The leaderboard is ~37 MB and ~45,000 rows. It is cheap in wall time (~2s) but
# not something to re-download for every symbol in every cycle, so it is cached on
# disk and refreshed at most this often.
LEADERBOARD_CACHE_HOURS = 12.0
DEFAULT_CACHE_PATH = ".hl_leaderboard_cache.json"

# How many top wallets to inspect. Each costs one clearinghouseState request, so
# this trades breadth against cycle time (~0.3s per wallet).
TOP_WALLETS = 15
# Ignore small accounts: a 900% monthly ROI on $200 is noise, not information.
MIN_ACCOUNT_VALUE_USD = 100_000.0
# Below this many dollars of aggregate exposure, the sample is too thin to report.
MIN_TOTAL_NOTIONAL_USD = 50_000.0


def _post_info(body: Dict[str, Any]) -> Any:
    resp = requests.post(INFO_URL, json=body, timeout=HTTP_TIMEOUT, headers=_HEADERS)
    resp.raise_for_status()
    return resp.json()


def base_symbol(symbol: str) -> str:
    """`BTC-USD` -> `BTC`. Hyperliquid names its assets by bare base currency."""
    out = symbol.upper()
    for suffix in ("-USDT", "-USD"):
        if out.endswith(suffix):
            return out[: -len(suffix)]
    return out


# ------------------------------------------------------------------ leaderboard


def _window_metric(row: Dict[str, Any], window: str, field: str) -> float:
    for entry in row.get("windowPerformances") or []:
        if len(entry) == 2 and entry[0] == window:
            try:
                return float(entry[1].get(field, 0.0))
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def fetch_leaderboard(cache_path: str = DEFAULT_CACHE_PATH) -> List[Dict[str, Any]]:
    """Public leaderboard rows, cached on disk.

    Returns [] on any failure -- the caller degrades to no positioning data.
    """
    try:
        if os.path.exists(cache_path):
            age_hours = (time.time() - os.path.getmtime(cache_path)) / 3600.0
            if age_hours < LEADERBOARD_CACHE_HOURS:
                with open(cache_path, "r", encoding="utf-8") as handle:
                    return json.load(handle).get("leaderboardRows", [])

        resp = requests.get(LEADERBOARD_URL, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()
        try:
            with open(cache_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
        except OSError:
            pass  # an unwritable cache is not worth failing over
        return payload.get("leaderboardRows", [])
    except Exception:
        return []


def top_wallets(
    rows: List[Dict[str, Any]],
    limit: int = TOP_WALLETS,
    window: str = "month",
) -> List[Tuple[str, float]]:
    """The best-performing sizeable accounts, as (address, roi) pairs.

    Ranked by return over `window` rather than absolute PnL, so a merely enormous
    account does not outrank a genuinely skilful one. Filtered by account value so
    the ranking is not dominated by tiny accounts posting huge percentages.
    """
    candidates = []
    for row in rows:
        try:
            if float(row.get("accountValue", 0.0)) < MIN_ACCOUNT_VALUE_USD:
                continue
        except (TypeError, ValueError):
            continue
        address = row.get("ethAddress")
        if address:
            candidates.append((address, _window_metric(row, window, "roi")))

    candidates.sort(key=lambda pair: pair[1], reverse=True)
    return candidates[:limit]


# ------------------------------------------------------------------- positions


def aggregate_positioning(wallets: List[Tuple[str, float]]) -> Dict[str, Dict[str, float]]:
    """Net long/short notional per asset across `wallets`.

    A wallet that fails to load is skipped, not fatal -- a partial sample is still
    informative, and the caller reports how many wallets actually answered.
    """
    totals: Dict[str, Dict[str, float]] = {}
    sampled = 0

    for address, _roi in wallets:
        try:
            state = _post_info({"type": "clearinghouseState", "user": address})
        except Exception:
            continue

        sampled += 1
        for asset_position in state.get("assetPositions") or []:
            position = asset_position.get("position") or {}
            coin = position.get("coin")
            if not coin:
                continue
            try:
                size = float(position.get("szi", 0.0))
                notional = abs(float(position.get("positionValue", 0.0)))
            except (TypeError, ValueError):
                continue
            # Decide the side before touching `totals`, so a flat or zero-value
            # position does not leave an empty bucket behind that later reads as
            # "this coin was seen" when nobody actually holds it.
            if notional <= 0 or size == 0:
                continue

            bucket = totals.setdefault(coin, {"long": 0.0, "short": 0.0, "wallets": 0.0})
            if size > 0:
                bucket["long"] += notional
            else:
                bucket["short"] += notional
            bucket["wallets"] += 1

    for bucket in totals.values():
        bucket["sampled_wallets"] = float(sampled)
    return totals


def fetch_funding_rate(coin: str) -> Optional[float]:
    """Current funding rate for `coin`'s perp, a cheap crowd-positioning proxy.

    Positive funding means longs are paying shorts, i.e. the crowd leans long.
    """
    try:
        meta, contexts = _post_info({"type": "metaAndAssetCtxs"})
        names = [entry.get("name") for entry in meta.get("universe", [])]
        index = names.index(coin)
        return float(contexts[index].get("funding"))
    except Exception:
        return None


# --------------------------------------------------------------------- summary


def summarise(coin: str, totals: Dict[str, Dict[str, float]], funding: Optional[float]) -> Optional[str]:
    """Turn the aggregates into one short, factual, clearly-hedged sentence.

    Returns None when there is nothing worth saying. Silence is correct here --
    an invented or padded summary is worse than no summary.
    """
    parts: List[str] = []
    bucket = totals.get(coin)
    wallet_lean: Optional[str] = None

    if bucket:
        long_usd = bucket.get("long", 0.0)
        short_usd = bucket.get("short", 0.0)
        total = long_usd + short_usd
        if total >= MIN_TOTAL_NOTIONAL_USD:
            net_pct = (long_usd - short_usd) / total * 100.0
            wallet_lean = "long" if net_pct >= 0 else "short"
            lean = (
                f"net long by roughly {net_pct:.0f}%"
                if net_pct >= 0
                else f"net short by roughly {abs(net_pct):.0f}%"
            )
            parts.append(
                f"Across the {int(bucket.get('sampled_wallets', 0))} best-performing large "
                f"Hyperliquid wallets sampled, {int(bucket.get('wallets', 0))} hold a "
                f"{coin} PERPETUAL position, and in aggregate they are {lean} "
                f"(${long_usd:,.0f} long vs ${short_usd:,.0f} short)"
            )

    if funding is not None:
        # Positive funding means longs pay shorts, i.e. the broad market leans long.
        crowd_lean = "long" if funding > 0 else "short"
        direction = "longs are paying shorts" if funding > 0 else "shorts are paying longs"
        note = (
            f"the {coin} perpetual funding rate is {funding * 100:.4f}% ({direction}), "
            f"which puts the broader market on the {crowd_lean} side"
        )
        # Never assert that the two agree. They often do not, and a divergence
        # between a skilled-wallet sample and the wider crowd is real information
        # -- claiming agreement that is not there would be feeding the model a
        # falsehood dressed as data.
        if wallet_lean is not None:
            note += (
                ", the same direction as the wallet sample above"
                if wallet_lean == crowd_lean
                else ", which is the OPPOSITE direction to the wallet sample above"
            )
        parts.append(note)

    if not parts:
        return None

    return (
        "; ".join(parts)
        + ". These are leveraged perpetual positions taken by other traders, not spot "
        "holdings, and this bot trades spot without leverage. Treat this as directional "
        "bias only, never as confirmation, and never as a reason to act without your own "
        "technical justification."
    )


def fetch_positioning(
    symbol: str,
    asset_class: str,
    cache_path: str = DEFAULT_CACHE_PATH,
    limit: int = TOP_WALLETS,
) -> Optional[str]:
    """Positioning context for `symbol`, or None when there is none to give.

    Never raises. Equities have no Hyperliquid positioning at all and return None
    immediately, without a single network call.
    """
    if asset_class != "crypto":
        return None

    try:
        coin = base_symbol(symbol)
        wallets = top_wallets(fetch_leaderboard(cache_path), limit=limit)
        totals = aggregate_positioning(wallets) if wallets else {}
        return summarise(coin, totals, fetch_funding_rate(coin))
    except Exception:
        # Positioning is a nice-to-have. A cycle never stops for it.
        return None
