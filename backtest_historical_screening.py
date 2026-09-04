"""Walk-forward backtest with the EQUITY universe re-screened at each historical
week boundary, instead of backtest.py's single fixed symbol list for the whole
window.

Why this is a different (stronger) question than backtest.py's plain run
--------------------------------------------------------------------------
In production the equity slate is not static: weekly_screening.yml re-runs
screening.py every Monday 06:00 UTC and rewrites symbols.yaml, so the 4h cycle
trades a rotating set of names, not whatever five happened to be picked once.
backtest.py's `run_backtest` answers "were the model's decisions good, given a
symbol list you chose"; this module answers "would the SYSTEM AS IT ACTUALLY
OPERATES -- weekly rescreen + 4h decisions -- have been good", which is the
question that actually matters for a real go/no-go call, since a static-list
backtest can look good or bad for reasons that have nothing to do with
whether weekly screening itself finds anything worth trading.

Reused, not reimplemented: every per-day primitive (decision_frame_for_day,
sweep_positions_for_day, run_symbol_for_day, mark_to_market_equity,
BacktestState, BacktestLogger, CostTracker, resolve_provider) comes straight
from backtest.py. The one structural change is that the day loop's symbol
list is no longer fixed -- it is resolved per day from whichever week's
screening result was most recently in effect, exactly mirroring how
main.load_config re-reads symbols.yaml fresh each cycle in production.

An open position never disappears just because its symbol rotates out of the
following week's slate: `sweep_positions_for_day` already iterates
`state.open_positions` (the ledger), never the active symbol list -- the same
way main.py's live sweep iterates every bot-managed position regardless of
what is in the current week's config.yaml/symbols.yaml. Only the decision to
open a NEW position is gated by "is this symbol in this week's slate".

Two honest, ADDITIONAL data gaps on top of backtest.py's existing two (no
historical headlines, no historical Hyperliquid positioning) -- read this
before trusting a result
--------------------------------------------------------------------------
3. **Survivorship-biased equity universe.** There is no free archive of
   historical S&P 500 / Nasdaq-100 constituent lists. Every historical week's
   candidate pool is drawn from TODAY's membership
   (equity_universe.build_equity_universe(), fetched once at the start of the
   run), not the membership that was actually in effect on that historical
   date. This cuts both ways and does not net out to "safe": a stock later
   removed from the index for poor performance is still in the pool and can
   still be screened OUT on its own bad numbers (fine), but a stock that
   hadn't been ADDED to the index yet is available for selection in weeks
   before it should have been eligible, and a stock that would have failed to
   qualify for membership back then is included anyway. There is no way to
   correct this without a paid point-in-time constituents dataset.
4. **No historical Hyperliquid spot volume, universe, or positioning.**
   Hyperliquid's spotMetaAndAssetCtxs only exposes today's rolling 24h volume
   and there is no free archive -- so the crypto HALF of screening.py (which
   spot markets exist, their liquidity ranking, their positioning) cannot be
   reproduced historically at all. Crypto symbols are therefore held FIXED
   for the whole run (default: BTC-USD, ETH-USD, SOL-USD, matching
   config.yaml) -- only the equity side is actually re-screened week to week.

Cost model is identical to backtest.py: same provider, same per-symbol
per-day model call, same CostTracker. Re-screening itself costs nothing (no
model call, pure yfinance) -- it only changes WHICH equities get evaluated
each week, not how many decisions get made per week.
"""

from __future__ import annotations

import argparse
import bisect
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

import backtest
import equity_universe
import main as main_module
import mode
from models import AssetClass

DEFAULT_DB_PATH = "backtest_historical_screening.db"
DEFAULT_EQUITY_COUNT = 5
DEFAULT_CRYPTO_SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD"]
DEFAULT_BENCHMARK_SYMBOL = "SPY"

# Calendar days of history fetched before each screening date to compute that
# week's volume/momentum signal -- comfortably covers a long weekend or a
# holiday while still only ever reading bars strictly before the screening
# date itself (see fetch_universe_price_data_asof's `end` handling).
SCREENING_LOOKBACK_CALENDAR_DAYS = 10

ADDITIONAL_LIMITATIONS = [
    "Survivorship-biased equity universe: every historical week's candidate "
    "pool is TODAY's S&P 500 + Nasdaq-100 membership (no free point-in-time "
    "constituents archive exists), not the membership actually in effect on "
    "that historical date.",
    "No historical Hyperliquid spot volume, universe, or positioning: the "
    "crypto side of screening cannot be reproduced historically at all "
    "(no free archive of past 24h volume). Crypto symbols are held fixed "
    "for the whole run instead of being re-screened weekly.",
]


# ------------------------------------------------------- point-in-time screen


def fetch_universe_price_data_asof(
    symbols: List[str], as_of_date: date, lookback_days: int = SCREENING_LOOKBACK_CALENDAR_DAYS
) -> Dict[str, Dict[str, float]]:
    """Historical analogue of equity_universe.fetch_universe_price_data.

    Same output shape and same volume-floor-free/momentum-for-everyone
    scoring input, just anchored to `as_of_date` instead of "now". `end` is
    passed to yfinance as `as_of_date` itself, which yfinance treats as
    exclusive -- so the request can only ever return bars strictly before
    `as_of_date`. Belt-and-suspenders: the returned frame is filtered again
    here rather than trusting that boundary alone, so a symbol whose bars ever
    include `as_of_date` or later (an off-by-one in yfinance's own handling,
    a returned bar for a still-in-progress session) can never leak into the
    signal a historical screening run at 06:00 UTC that day could not
    actually have seen yet.
    """
    symbols = list(symbols)
    if not symbols:
        return {}

    cutoff = pd.Timestamp(as_of_date)
    start = as_of_date - timedelta(days=lookback_days)

    try:
        df = yf.download(
            symbols,
            start=start.isoformat(),
            end=as_of_date.isoformat(),
            interval="1d",
            group_by="ticker",
            progress=False,
            auto_adjust=True,
            threads=True,
        )
    except Exception:
        return {}

    if df is None or df.empty:
        return {}

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[df.index < cutoff]  # explicit anti-lookahead guard -- see docstring
    if df.empty:
        return {}

    present = {c[0] for c in df.columns} if isinstance(df.columns, pd.MultiIndex) else set()

    out: Dict[str, Dict[str, float]] = {}
    for symbol in symbols:
        if symbol not in present:
            continue
        try:
            close = df[symbol]["Close"].dropna()
            volume = df[symbol]["Volume"].dropna()
        except KeyError:
            continue
        if len(close) < 2 or volume.empty:
            continue

        prev_close = float(close.iloc[-2])
        if prev_close == 0:
            continue

        out[symbol] = {
            "price_change_pct": (float(close.iloc[-1]) / prev_close - 1.0) * 100.0,
            "volume": float(volume.iloc[-1]),
        }

    return out


def historical_equity_screen(
    universe: set, as_of_date: date, count: int = DEFAULT_EQUITY_COUNT
) -> List[str]:
    """The equity slate screening.py would have picked as of `as_of_date`,
    using only price data strictly before that date -- reuses
    equity_universe.score_equities/select_top_equities exactly as-is, only
    the price_data input is historical instead of live.
    """
    price_data = fetch_universe_price_data_asof(sorted(universe), as_of_date)
    scored = equity_universe.score_equities(universe, price_data)
    return equity_universe.select_top_equities(scored, count)


def weekly_screening_dates(start: date, end: date) -> List[date]:
    """`start` itself, then every Monday after it up to `end`.

    Mirrors weekly_screening.yml's Monday 06:00 UTC cadence. Seeded with an
    initial screen on `start` (whatever weekday that is) so the very first
    simulated week also trades a real, freshly-screened slate rather than an
    empty one -- in production this is equivalent to "symbols.yaml already
    exists from the most recent real Monday before this backtest's start".
    """
    days_to_next_monday = (7 - start.weekday()) % 7 or 7
    dates = [start]
    d = start + timedelta(days=days_to_next_monday)
    while d <= end:
        dates.append(d)
        d += timedelta(days=7)
    return dates


class EquitySlateSchedule:
    """Which equities are active on any given simulated day.

    Built once from `weekly_screening_dates`, then queried per day via binary
    search -- O(log weeks) per day, not a re-scan of the whole schedule.
    """

    def __init__(self, slate_by_date: Dict[date, List[str]]):
        self.dates = sorted(slate_by_date.keys())
        self.slate_by_date = slate_by_date

    def active_equities(self, day: Any) -> List[str]:
        # `day` is a pandas Timestamp everywhere it's actually called from (the
        # backtest day loop iterates frame.index entries), while `self.dates`
        # are plain `date` objects (dict keys from weekly_screening_dates) --
        # comparing the two types directly raises in pandas, so normalise here
        # rather than at every call site.
        day_as_date = day.date() if hasattr(day, "date") else day
        idx = bisect.bisect_right(self.dates, day_as_date) - 1
        if idx < 0:
            return []
        return self.slate_by_date[self.dates[idx]]

    def all_symbols_ever_active(self) -> set:
        out: set = set()
        for symbols in self.slate_by_date.values():
            out.update(symbols)
        return out


def build_equity_slate_schedule(
    universe: set, start: date, end: date, count: int = DEFAULT_EQUITY_COUNT
) -> EquitySlateSchedule:
    slate_by_date: Dict[date, List[str]] = {}
    for screen_date in weekly_screening_dates(start, end):
        slate_by_date[screen_date] = historical_equity_screen(universe, screen_date, count)
        print(f"  [screen] {screen_date}: {slate_by_date[screen_date]}")
    return EquitySlateSchedule(slate_by_date)


# ------------------------------------------------------------------- report


def compute_rotating_report(
    state: "backtest.BacktestState",
    start: date,
    end: date,
    starting_equity: float,
    crypto_symbols: List[str],
    schedule: EquitySlateSchedule,
) -> Dict[str, Any]:
    """Same core metrics as backtest.compute_report, plus a benchmark that
    actually fits a rotating universe: buy-and-hold the broad market
    (DEFAULT_BENCHMARK_SYMBOL), not "the same symbols" -- there is no single
    fixed set of symbols to hold here.
    """
    final_equity = state.equity_curve[-1][1] if state.equity_curve else starting_equity
    total_return_pct = (final_equity / starting_equity - 1.0) * 100.0 if starting_equity > 0 else 0.0

    closed = state.closed_trades
    wins = [t for t in closed if t.realized_pnl_usd > 0]
    win_rate_pct = (len(wins) / len(closed) * 100.0) if closed else 0.0

    peak = starting_equity
    max_drawdown_pct = 0.0
    for _day, eq in state.equity_curve:
        peak = max(peak, eq)
        if peak > 0:
            max_drawdown_pct = min(max_drawdown_pct, (eq / peak - 1.0) * 100.0)

    try:
        benchmark_frame = backtest.fetch_historical_ohlcv(DEFAULT_BENCHMARK_SYMBOL, start, end)
        days_in_range = backtest.trading_days_in_range(benchmark_frame, start, end)
        entry_open = float(benchmark_frame.loc[days_in_range[0], "Open"])
        exit_close = float(benchmark_frame.loc[days_in_range[-1], "Close"])
        buy_and_hold_return_pct = (exit_close / entry_open - 1.0) * 100.0
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: benchmark ({DEFAULT_BENCHMARK_SYMBOL}) fetch failed: {exc}")
        buy_and_hold_return_pct = None

    return {
        "start": str(start),
        "end": str(end),
        "crypto_symbols_fixed": crypto_symbols,
        "equity_screening_history": [
            {"effective_from": str(d), "symbols": schedule.slate_by_date[d]}
            for d in schedule.dates
        ],
        "starting_equity": starting_equity,
        "final_equity": final_equity,
        "total_return_pct": total_return_pct,
        "num_trades": len(closed),
        "win_rate_pct": win_rate_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "benchmark_symbol": DEFAULT_BENCHMARK_SYMBOL,
        "buy_and_hold_return_pct": buy_and_hold_return_pct,
        "strategy_vs_buy_and_hold_pct": (
            total_return_pct - buy_and_hold_return_pct
            if buy_and_hold_return_pct is not None else None
        ),
        "limitations": list(backtest.LIMITATIONS) + list(ADDITIONAL_LIMITATIONS),
    }


def format_rotating_report(report: Dict[str, Any]) -> str:
    lines = [
        f"=== Historical-screening backtest: {report['start']} .. {report['end']} ===",
        f"Crypto (fixed, not re-screened): {', '.join(report['crypto_symbols_fixed'])}",
        "Equity screening history (rotated weekly):",
    ]
    for entry in report["equity_screening_history"]:
        lines.append(f"  {entry['effective_from']}: {', '.join(entry['symbols']) or '(none)'}")
    bh = report["buy_and_hold_return_pct"]
    vs = report["strategy_vs_buy_and_hold_pct"]
    lines += [
        f"Starting equity: ${report['starting_equity']:,.2f}   "
        f"Final equity: ${report['final_equity']:,.2f}",
        f"Total return: {report['total_return_pct']:+.2f}%",
        f"Buy-and-hold ({report['benchmark_symbol']}, same period): "
        + (f"{bh:+.2f}%" if bh is not None else "unavailable"),
        "Strategy vs. buy-and-hold: "
        + (f"{vs:+.2f} pp" if vs is not None else "unavailable"),
        f"Trades: {report['num_trades']}   Win rate: {report['win_rate_pct']:.1f}%   "
        f"Max drawdown: {report['max_drawdown_pct']:.2f}%",
        "",
        "Limitations (read before trusting this result):",
    ]
    lines += [f"  - {item}" for item in report["limitations"]]
    return "\n".join(lines)


# --------------------------------------------------------------------- run


def run_backtest_with_rotating_screening(
    start: date,
    end: date,
    config: Dict[str, Any],
    crypto_symbols: Optional[List[str]] = None,
    equity_count: int = DEFAULT_EQUITY_COUNT,
    provider: str = backtest.DEFAULT_PROVIDER,
    model: Optional[str] = None,
    starting_equity: float = backtest.DEFAULT_STARTING_EQUITY,
    db_path: str = DEFAULT_DB_PATH,
) -> Dict[str, Any]:
    # Deliberately `is None`, not a bare truthiness check: an explicitly empty
    # list ("run this with no crypto at all") must not be silently replaced by
    # the default -- only an actually-omitted argument should be.
    crypto_symbols = list(DEFAULT_CRYPTO_SYMBOLS if crypto_symbols is None else crypto_symbols)
    model = model or backtest.DEFAULT_MODEL_BY_PROVIDER[provider]
    generate_signal_fn = backtest.resolve_provider(provider)
    mode_settings = mode.resolve_mode_settings(True, config)

    circuit_breaker_loss_pct = float(config.get("circuit_breaker_loss_pct", 3.0))
    max_risk_pct = float(config.get("max_risk_pct", 1.0))
    max_absolute_position_pct = float(config.get("max_absolute_position_pct", 20.0))

    print(f"=== Historical-screening backtest | {start} .. {end} | provider={provider} model={model} ===")
    for item in backtest.LIMITATIONS + ADDITIONAL_LIMITATIONS:
        print(f"NOTE: {item}")

    print("Building today's equity universe (S&P 500 + Nasdaq) as the survivorship-biased proxy pool...")
    universe = equity_universe.build_equity_universe()
    print(f"  universe: {len(universe)} symbols")

    print("Re-screening equities at each historical week boundary (no model calls, pure yfinance)...")
    schedule = build_equity_slate_schedule(universe, start, end, equity_count)

    all_equity_symbols = schedule.all_symbols_ever_active()
    all_symbols_with_class: List[Tuple[str, AssetClass]] = (
        [(s, "equity") for s in sorted(all_equity_symbols)]
        + [(s, "crypto") for s in crypto_symbols]
    )
    print(
        f"{len(all_equity_symbols)} distinct equities selected across the whole run "
        f"(union, since a rotated-out symbol's open position is still tracked); "
        f"{len(crypto_symbols)} fixed crypto symbols"
    )

    full_frames: Dict[str, pd.DataFrame] = {}
    for symbol, _asset_class in all_symbols_with_class:
        try:
            full_frames[symbol] = backtest.fetch_historical_ohlcv(symbol, start, end)
        except Exception as exc:  # noqa: BLE001 - one symbol's data outage never kills the run
            print(f"  WARNING: {symbol}: historical OHLCV fetch failed ({exc}); excluded from the run")

    all_days: List[Any] = sorted(
        {d for frame in full_frames.values() for d in backtest.trading_days_in_range(frame, start, end)}
    )
    print(f"{len(all_days)} simulated trading days")

    # Cost estimate: crypto trades every simulated day, equities only on the
    # (up to DEFAULT_EQUITY_COUNT) days each is actually active -- roughly
    # len(all_days) * (len(crypto_symbols) + equity_count) model calls, a
    # reasonable upper-bound approximation for the upfront estimate.
    backtest.print_upfront_cost_estimate(
        model, expected_calls=len(all_days) * (len(crypto_symbols) + equity_count)
    )

    state = backtest.BacktestState(equity=starting_equity)
    logger = backtest.BacktestLogger(db_path)
    cost = backtest.CostTracker(model)

    for day in all_days:
        state.note_day(day)
        closed_today = backtest.sweep_positions_for_day(state, full_frames, day, logger)

        active_today: List[Tuple[str, AssetClass]] = (
            [(s, "equity") for s in schedule.active_equities(day)]
            + [(s, "crypto") for s in crypto_symbols]
        )

        for symbol, asset_class in active_today:
            if symbol in closed_today:
                continue
            frame = full_frames.get(symbol)
            if frame is None or day not in frame.index:
                continue
            backtest.run_symbol_for_day(
                state=state, symbol=symbol, asset_class=asset_class, day=day, full_frame=frame,
                mode_settings=mode_settings, generate_signal_fn=generate_signal_fn, model=model,
                cost=cost, logger=logger, circuit_breaker_loss_pct=circuit_breaker_loss_pct,
                max_risk_pct=max_risk_pct, max_absolute_position_pct=max_absolute_position_pct,
            )

        equity_today = backtest.mark_to_market_equity(state, full_frames, day)
        state.equity_curve.append((day, equity_today))
        logger.log_equity(day, equity_today)

        if cost.calls_made and cost.calls_made % backtest.COST_PROGRESS_EVERY_N_CALLS == 0:
            print(cost.progress_line())

    report = compute_rotating_report(state, start, end, starting_equity, crypto_symbols, schedule)
    print()
    print(format_rotating_report(report))
    print()
    print(cost.final_line())

    report["cost"] = {
        "model": model,
        "calls_made": cost.calls_made,
        "input_tokens": cost.input_tokens,
        "output_tokens": cost.output_tokens,
        "actual_cost_usd": cost.actual_cost_usd(),
    }
    logger.close()
    return report


# ---------------------------------------------------------------------- CLI


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Walk-forward backtest with the equity universe re-screened weekly."
    )
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--crypto-symbols", default=",".join(DEFAULT_CRYPTO_SYMBOLS),
        help="Comma-separated, held fixed for the whole run (not re-screened -- see module docstring)",
    )
    parser.add_argument("--equity-count", type=int, default=DEFAULT_EQUITY_COUNT)
    parser.add_argument("--provider", default=backtest.DEFAULT_PROVIDER, choices=["claude", "gemini"])
    parser.add_argument("--model", default=None, help=f"Default per provider: {backtest.DEFAULT_MODEL_BY_PROVIDER}")
    parser.add_argument("--starting-equity", type=float, default=backtest.DEFAULT_STARTING_EQUITY)
    parser.add_argument("--config", default=main_module.DEFAULT_CONFIG_PATH)
    parser.add_argument("--db", default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--report-json", default=None,
        help="Also write the full report dict (incl. equity_screening_history and cost) as JSON to this path",
    )
    args = parser.parse_args()

    config = main_module.load_config(args.config, None)
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    crypto_symbols = [s.strip() for s in args.crypto_symbols.split(",") if s.strip()]

    report = run_backtest_with_rotating_screening(
        start=start, end=end, config=config, crypto_symbols=crypto_symbols,
        equity_count=args.equity_count, provider=args.provider, model=args.model,
        starting_equity=args.starting_equity, db_path=args.db,
    )

    if args.report_json:
        import json

        with open(args.report_json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
        print(f"Wrote full report JSON to {args.report_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
