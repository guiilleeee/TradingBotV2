"""Public Hyperliquid trader-positioning intelligence.

What this is: a small, factual, human-readable summary of how large successful
Hyperliquid wallets are positioned in an asset AND what they have actually been
doing in it recently, handed to the model as one more input alongside price,
indicators and headlines. The aggregate percentage (existing) and the specific
recent trades (this module's newer half) are two views of the same underlying
sample -- the same top wallets, richer detail, nothing new in kind.

What this is emphatically NOT: a copy-trading mechanism. Nothing here decides
anything. The output is a string on SignalInput, and the only thing that ever reads
it is the model's own reasoning. There is no path from this module to an order that
does not pass through risk_manager.validate() exactly like every other signal --
same confidence threshold, same stop-loss requirement, same sizing, same caps. A
specific, individually-attributed trade ("wallet ending ...4f2a opened a $52,000
long 3 hours ago") is still just one more sentence of directional bias, covered by
the exact same system-prompt rule as the aggregate percentage it sits next to --
see prompts.py rule 5.

One honest caveat is baked into the wording it produces: the positions and trades
being read are Hyperliquid PERPETUAL activity, because that is the only positioning
data the venue exposes. We trade spot without leverage. The summary says "perps" out
loud rather than implying these wallets hold spot.

Every function here fails soft. Positioning data is a nice-to-have; a cycle must
never stop because a leaderboard was slow, or because one wallet's fill history
timed out.
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

# How many top wallets to inspect. Each costs one clearinghouseState request (for
# the aggregate position) plus one userFills request (for recent trades) -- twice
# the calls of the aggregate-only version, but the same wallet COUNT, which is the
# thing that actually scales cost. ~0.3s per wallet per call.
TOP_WALLETS = 15
# Ignore small accounts: a 900% monthly ROI on $200 is noise, not information.
MIN_ACCOUNT_VALUE_USD = 100_000.0
# Below this many dollars of aggregate exposure, the sample is too thin to report.
MIN_TOTAL_NOTIONAL_USD = 50_000.0

# How far back to look for individual recent trades, and how small a single trade
# can be before it's noise rather than information -- same philosophy as
# MIN_ACCOUNT_VALUE_USD/MIN_TOTAL_NOTIONAL_USD above, applied at the trade level.
RECENT_TRADES_WINDOW_HOURS = 24.0
MIN_TRADE_NOTIONAL_USD = 5_000.0
# A few lines at most, not a wall of raw data -- across ALL sampled wallets combined,
# not per wallet, so one hyperactive wallet cannot crowd out everyone else's trades.
MAX_RECENT_TRADES_IN_SUMMARY = 5


def post_info(body: Dict[str, Any]) -> Any:
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
            state = post_info({"type": "clearinghouseState", "user": address})
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


# --------------------------------------------------------------- recent trades


# Hyperliquid's own vocabulary for a perp fill's effect on the position. A fill on
# a SPOT market (Hyperliquid names those "@<index>", not a plain coin symbol) uses
# "Buy"/"Sell" instead -- excluded by construction here, not filtered explicitly:
# it never matches `coin`, which is always a bare perp name like "BTC" (see
# base_symbol), so a spot fill simply never passes the coin-equality check below.
_SIGNIFICANT_FILL_DIRS = {
    "Open Long", "Open Short", "Close Long", "Close Short", "Long > Short", "Short > Long",
}


def recent_trades_for_coin(
    fills: List[Dict[str, Any]], coin: str, since_ms: float
) -> List[Dict[str, Any]]:
    """One wallet's `coin` perp fills since `since_ms`, grouped into whole trades.

    A single order routinely fills across several price levels as separate rows
    sharing one `oid` (order id) -- verified live against Hyperliquid's real
    userFills response. Grouped here so one trade decision produces one entry, not
    one per partial fill; `time` becomes the latest fill in the group (when the
    order finished, not when it started) and `notional` their summed dollar size.
    """
    grouped: Dict[Any, Dict[str, Any]] = {}
    for fill in fills:
        if fill.get("coin") != coin or fill.get("dir") not in _SIGNIFICANT_FILL_DIRS:
            continue
        try:
            fill_time = float(fill.get("time", 0.0))
            notional = abs(float(fill.get("sz", 0.0))) * float(fill.get("px", 0.0))
        except (TypeError, ValueError):
            continue
        if fill_time < since_ms:
            continue

        bucket = grouped.setdefault(
            fill.get("oid"), {"dir": fill.get("dir"), "time": fill_time, "notional": 0.0}
        )
        bucket["notional"] += notional
        bucket["time"] = max(bucket["time"], fill_time)

    trades = [t for t in grouped.values() if t["notional"] >= MIN_TRADE_NOTIONAL_USD]
    trades.sort(key=lambda t: t["time"], reverse=True)
    return trades


def _redact_address(address: str) -> str:
    """`0xABCDEF...1234` -> `...1234`. Enough to distinguish wallets in one summary
    without printing a full address the model has no legitimate use for.
    """
    return "..." + address[-4:] if len(address) >= 4 else address


_TRADE_VERBS = {
    "Open Long": "opened a ${notional:,.0f} long",
    "Open Short": "opened a ${notional:,.0f} short",
    "Close Long": "closed a ${notional:,.0f} long",
    "Close Short": "closed a ${notional:,.0f} short",
    "Long > Short": "flipped a ${notional:,.0f} position from long to short",
    "Short > Long": "flipped a ${notional:,.0f} position from short to long",
}


def _describe_time_ago(trade_time_ms: float, now_ms: float) -> str:
    hours = max(0.0, (now_ms - trade_time_ms) / 3_600_000.0)
    if hours < 1.0:
        minutes = max(1, round(hours * 60))
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    rounded = round(hours)
    return f"{rounded} hour{'s' if rounded != 1 else ''} ago"


def describe_recent_trades(
    entries: List[Tuple[str, Dict[str, Any]]], now_ms: float
) -> Optional[str]:
    """Turn (address, trade) pairs into the factual, specific sentence the model
    sees -- e.g. "wallet ending ...4f2a opened a $52,000 long 3 hours ago". None
    when there is nothing recent enough or large enough to be worth reporting.
    """
    if not entries:
        return None

    lines = []
    for address, trade in entries[:MAX_RECENT_TRADES_IN_SUMMARY]:
        verb = _TRADE_VERBS.get(trade["dir"])
        if verb is None:
            continue
        lines.append(
            f"wallet ending {_redact_address(address)} "
            f"{verb.format(notional=trade['notional'])} "
            f"{_describe_time_ago(trade['time'], now_ms)}"
        )

    if not lines:
        return None

    return (
        f"Specific recent activity from the same sampled wallets in the last "
        f"~{int(RECENT_TRADES_WINDOW_HOURS)}h: " + "; ".join(lines) + "."
    )


def fetch_recent_trade_notes(
    coin: str, wallets: List[Tuple[str, float]], now_ms: Optional[float] = None
) -> Optional[str]:
    """The specific-trades sentence for `coin`, sampled from the same `wallets`
    already used for the aggregate position summary.

    One wallet's fill history failing (timeout, malformed address, anything) is
    skipped, never fatal -- identical philosophy to aggregate_positioning's
    per-wallet try/except.
    """
    now_ms = now_ms if now_ms is not None else time.time() * 1000.0
    since_ms = now_ms - RECENT_TRADES_WINDOW_HOURS * 3_600_000.0

    entries: List[Tuple[str, Dict[str, Any]]] = []
    for address, _roi in wallets:
        try:
            fills = post_info({"type": "userFills", "user": address})
        except Exception:
            continue
        if not isinstance(fills, list):
            continue
        for trade in recent_trades_for_coin(fills, coin, since_ms):
            entries.append((address, trade))

    entries.sort(key=lambda pair: pair[1]["time"], reverse=True)
    return describe_recent_trades(entries, now_ms)


def fetch_funding_rate(coin: str) -> Optional[float]:
    """Current funding rate for `coin`'s perp, a cheap crowd-positioning proxy.

    Positive funding means longs are paying shorts, i.e. the crowd leans long.
    """
    try:
        meta, contexts = post_info({"type": "metaAndAssetCtxs"})
        names = [entry.get("name") for entry in meta.get("universe", [])]
        index = names.index(coin)
        return float(contexts[index].get("funding"))
    except Exception:
        return None


# --------------------------------------------------------------------- summary


def summarise(
    coin: str,
    totals: Dict[str, Dict[str, float]],
    funding: Optional[float],
    recent_trade_notes: Optional[str] = None,
) -> Optional[str]:
    """Turn the aggregates -- and, when available, specific recent trades -- into
    one short, factual, clearly-hedged passage.

    `recent_trade_notes` (from fetch_recent_trade_notes) is optional and additive:
    it can make an otherwise-empty summary non-empty (recent trades exist even when
    no aggregate position clears MIN_TOTAL_NOTIONAL_USD), but never replaces the
    aggregate view or the hedging paragraph below -- both apply to it equally.

    Returns None when there is nothing worth saying at all. Silence is correct here
    -- an invented or padded summary is worse than no summary.
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

    if not parts and not recent_trade_notes:
        return None

    # The aggregate/funding sentence and the specific-trades sentence are each
    # optional on their own (recent trades can exist for a coin with no aggregate
    # position large enough to clear MIN_TOTAL_NOTIONAL_USD, and vice versa). Each
    # is stripped of its own trailing period before joining so exactly one period
    # ever separates sentences -- never a bare concatenation artifact like "..".
    lead_sentences = []
    if parts:
        lead_sentences.append("; ".join(parts))
    if recent_trade_notes:
        lead_sentences.append(recent_trade_notes.rstrip("."))

    return (
        ". ".join(lead_sentences)
        + ". These are leveraged perpetual positions (and, where noted above, trades) "
        "taken by other traders, not spot holdings, and this bot trades spot without "
        "leverage. Treat this as directional bias only, never as confirmation, and "
        "never as a reason to act without your own technical justification."
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
        recent_trade_notes = fetch_recent_trade_notes(coin, wallets) if wallets else None
        return summarise(coin, totals, fetch_funding_rate(coin), recent_trade_notes)
    except Exception:
        # Positioning is a nice-to-have. A cycle never stops for it.
        return None
