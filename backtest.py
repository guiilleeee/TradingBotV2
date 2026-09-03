"""Walk-forward backtest: does the model's reasoning actually have edge?

Everything built in earlier phases proves the system behaves safely and
consistently. This module is the first thing that asks whether its decisions
are any *good* -- the one piece standing between this project and a real
go/no-go call on live money.

Two honest, unavoidable data gaps -- read this before trusting a result
--------------------------------------------------------------------------
1. **No historical headlines.** `data_fetcher.fetch_headlines` only returns
   *current* Yahoo RSS items; there is no free archive of "what the headlines
   were" on a past date. Every backtest day passes `recent_headlines=[]`.
   This degrades exactly the way a live headline-fetch failure already does
   (system prompt rule 5 already treats headlines as optional bias) -- no new
   code path, just less context than a live run would have.
2. **No historical Hyperliquid positioning.** The leaderboard is a live
   snapshot with no archive. `market_positioning` is `None` for every day.
Both gaps are printed in the run banner and carried in the returned report
dict (`report["limitations"]`) -- not just a code comment -- because a result
read later without this context is a result that looks better than it is.

Anti-lookahead guarantee
-------------------------
For simulated day N, `decision_frame_for_day` returns only bars *strictly
before* day N -- day N's own Close, High, Low, Volume are never in the frame
handed to `data_fetcher.compute_indicators` or used as `current_price`. Day
N's own bar is read exactly once, for its Open (the fill price) and its
High/Low (for stop/take-profit touches on a position opened on an earlier
day). See `decision_frame_for_day` and `tests/test_backtest.py`'s lookahead
tests -- one proves the real function excludes a deliberately-planted future
value, the other proves the test has teeth by showing a naive `<=` slice
would have leaked it.

Isolation
---------
A dedicated `backtest.db` (via `BacktestLogger`), never `trading_bot.db`.
Nothing here calls `execution.execute_trade`, `BotLogger()`, or any live
broker function -- every fill is computed locally from historical OHLCV.
`risk_manager.validate` and the equity-relative sizing formula from
execution.py are reused exactly as live would use them; only the venue-
specific order mechanics (whole-share-vs-notional, exchange minimums) are
out of scope here, since those affect execution plumbing, not whether the
model's calls have edge.
"""

from __future__ import annotations

import argparse
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

import data_fetcher
import main as main_module
import mode
import risk_manager
from models import AssetClass, ExistingPosition, SignalInput, TokenUsage, TradeSignal

DEFAULT_DB_PATH = "backtest.db"
DEFAULT_STARTING_EQUITY = 10_000.0

# Calendar days of history fetched *before* the requested start date, so the
# very first simulated day already has enough bars for SMA-50. 180 calendar
# days is comfortably over the ~51 trading days needed even for a 7-day/week
# crypto symbol, and far more so for a 5-day/week equity.
LOOKBACK_CALENDAR_DAYS = 180

# USD per 1,000,000 tokens: (input, output). Verified at build time against
# Anthropic's published pricing and a live pricing search for Gemini -- see
# the note on gemini-3.7-flash below, since the brief's premise ("free") does
# not hold at the paid API tier and that is worth being explicit about rather
# than silently repeating.
MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-5": (5.00, 25.00),
    # Introductory rate through 2026-12-31 (doubles to $1.50/$7.50 after).
    # Gemini does have a separate free-tier API key with rate limits, but the
    # paid per-token rate used for a real cost estimate is not zero -- stated
    # honestly here rather than reporting a "free" backtest that understates
    # what a production run on this model would actually cost.
    "gemini-3.7-flash": (0.75, 3.75),
}

DEFAULT_PROVIDER = "claude"
# Cheap-but-capable is the point of a backtest -- Haiku for evaluating
# whether the *prompt and strategy* have edge, not the live provider (Opus)
# a full multi-month, multi-symbol run on Opus would run into real money for
# what is fundamentally an evaluation pass, not a production decision.
DEFAULT_MODEL_BY_PROVIDER: Dict[str, str] = {
    "claude": "claude-haiku-4-5",
    "gemini": "gemini-3.7-flash",
}

# Rough, pre-run-only assumption for the upfront cost estimate -- refined
# continuously afterward by real per-call usage, never relied on alone.
ROUGH_INPUT_TOKENS_PER_CALL = 900
ROUGH_OUTPUT_TOKENS_PER_CALL = 250

COST_PROGRESS_EVERY_N_CALLS = 20

LIMITATIONS = [
    "No historical headlines: recent_headlines=[] for every simulated day "
    "(Yahoo RSS only ever returns current items; there is no free archive).",
    "No historical Hyperliquid positioning: market_positioning=None for every "
    "simulated day (the leaderboard is a live snapshot with no archive).",
]


# ------------------------------------------------------------------ data


def fetch_historical_ohlcv(
    symbol: str,
    start: date,
    end: date,
    lookback_days: int = LOOKBACK_CALENDAR_DAYS,
) -> pd.DataFrame:
    """Daily OHLCV for `symbol` covering [start, end] plus lookback history.

    Deliberately not data_fetcher.fetch_ohlcv -- that function fetches a
    relative window anchored to "now" (`period="120d"`), which cannot express
    a fixed historical range. This reuses the same yfinance ticker convention
    and the same NaN-Close cleanup, just via `start`/`end` instead of `period`.
    """
    fetch_start = start - timedelta(days=lookback_days)
    fetch_end = end + timedelta(days=1)  # yfinance's `end` is exclusive

    ticker = yf.Ticker(symbol)
    df = ticker.history(start=fetch_start.isoformat(), end=fetch_end.isoformat(), interval="1d")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df is None or df.empty:
        raise ValueError(f"{symbol}: no historical bars for {fetch_start}..{end}")

    df = df.dropna(subset=["Close"])
    if df.empty:
        raise ValueError(f"{symbol}: every bar in {fetch_start}..{end} had a NaN Close")

    # Normalise to timezone-naive timestamps -- yfinance returns exchange-local
    # tz-aware timestamps, and comparing those against plain `date` objects
    # elsewhere in this module would silently misalign by the tz offset.
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    return df


def decision_frame_for_day(full_frame: pd.DataFrame, day: Any) -> pd.DataFrame:
    """Bars strictly *before* `day`. The entire anti-lookahead guarantee lives here.

    Every indicator, every `current_price` handed to the model for day N comes
    from this function's output -- never from `full_frame` directly. Day N's
    own row (Open/High/Low/Close/Volume) is not present in the result at all,
    so there is no slice, index, or column access anywhere else in this module
    that could accidentally read it.
    """
    day_ts = pd.Timestamp(day)
    return full_frame[full_frame.index < day_ts]


def trading_days_in_range(full_frame: pd.DataFrame, start: date, end: date) -> List[pd.Timestamp]:
    """Real session dates within [start, end] -- whatever the venue actually traded.

    No calendar logic of our own: a symbol's own index already reflects
    whether it trades 5 or 7 days a week, weekends and holidays included or
    excluded correctly by construction.
    """
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    return sorted(d for d in full_frame.index if start_ts <= d <= end_ts)


# --------------------------------------------------------------- state


@dataclass
class OpenPosition:
    symbol: str
    qty: float
    entry_price: float
    stop_loss_price: Optional[float]
    take_profit_price: Optional[float]
    opened_day: Any


@dataclass
class ClosedTrade:
    symbol: str
    opened_day: Any
    closed_day: Any
    qty: float
    entry_price: float
    exit_price: float
    realized_pnl_usd: float
    is_auto_close: bool


@dataclass
class BacktestState:
    equity: float
    open_positions: Dict[str, OpenPosition] = field(default_factory=dict)
    closed_trades: List[ClosedTrade] = field(default_factory=list)
    equity_curve: List[Tuple[Any, float]] = field(default_factory=list)
    _today: Optional[Any] = None
    _today_realized_pnl: float = 0.0

    def note_day(self, day: Any) -> None:
        """Reset the daily-loss accumulator on a new simulated day.

        This is the whole reason the circuit breaker works correctly here:
        `today_realized_loss_pct` below is scoped to *this backtest's own
        simulated calendar*, never to the real wall-clock date the script
        happens to run on -- unlike logger.BotLogger.get_today_realized_loss_pct,
        which is intentionally not reused for this (see BacktestLogger's
        docstring for why).
        """
        if self._today != day:
            self._today = day
            self._today_realized_pnl = 0.0

    def today_realized_loss_pct(self) -> float:
        if self.equity <= 0:
            return 0.0
        return (self._today_realized_pnl / self.equity) * 100.0

    def record_realized(self, amount: float) -> None:
        self.equity += amount
        self._today_realized_pnl += amount


def _last_known_close(frame: pd.DataFrame, day: Any) -> Optional[float]:
    """Most recent Close at or before `day` -- the mark a real broker would show.

    Not `frame.loc[day, "Close"]`: `all_days` is the union of every symbol's
    own session dates, so on a day only a 7-day/week crypto symbol trades, an
    equity's frame simply has no row for it. Silently treating that as "no
    unrealised P&L today" would make an open equity position's contribution
    to the equity curve blink on and off across weekends -- a real, found-via-
    real-data bug (confirmed against real AAPL/BTC-USD dry-run output: an
    open AAPL position's mark vanished on Saturday/Sunday even though BTC-USD
    kept advancing the simulated calendar). Carrying the last real session's
    close forward is what "marked to market over a weekend" actually means.
    """
    eligible = frame[frame.index <= pd.Timestamp(day)]
    if eligible.empty:
        return None
    return float(eligible["Close"].iloc[-1])


def mark_to_market_equity(state: BacktestState, full_frames: Dict[str, pd.DataFrame], day: Any) -> float:
    """Equity including unrealised P&L of positions still open at day's close.

    Mirrors main.py's live equity computation (realised + unrealised, counted
    exactly once each) -- a position swept closed earlier the same day already
    contributed its P&L via `record_realized` and is no longer in
    `open_positions`, so it is never double-counted here.
    """
    unrealized = 0.0
    for symbol, pos in state.open_positions.items():
        frame = full_frames.get(symbol)
        if frame is None:
            continue
        close = _last_known_close(frame, day)
        if close is not None:
            unrealized += (close - pos.entry_price) * pos.qty
    return state.equity + unrealized


# --------------------------------------------------------------- sweep


def _stop_take_fill(day_open: float, level: float, is_stop: bool) -> float:
    """Realistic fill for a stop/take-profit touched on `day`.

    A gap through the level is real: a stop triggered by a gap-down open
    fills at the (worse) open, not at the stop price nobody could have
    traded at; a take-profit gapped through by a gap-up open fills at the
    (better) open, same asymmetric-gap-risk reasoning as everywhere else
    "worse of the two" language appears in this project (main.py's sweep).
    """
    if is_stop:
        return day_open if day_open < level else level
    return day_open if day_open > level else level


def sweep_positions_for_day(
    state: BacktestState,
    full_frames: Dict[str, pd.DataFrame],
    day: Any,
    logger: "BacktestLogger",
) -> set:
    """Close any open position whose stop or take-profit was touched today.

    Daily-bar analogue of main.sweep_open_positions: runs once per day, before
    any fresh decision, and a symbol closed here is skipped for a new decision
    the same day (mirrored in run_backtest's main loop) -- exactly the
    `auto_closed_symbols` pattern the live cycle already uses.
    """
    closed_today: set = set()

    for symbol, pos in list(state.open_positions.items()):
        frame = full_frames.get(symbol)
        if frame is None or day not in frame.index:
            continue  # no bar today for this symbol -- leave the position open

        bar = frame.loc[day]
        low, high, day_open = float(bar["Low"]), float(bar["High"]), float(bar["Open"])

        hit_stop = pos.stop_loss_price is not None and low <= pos.stop_loss_price
        hit_take = pos.take_profit_price is not None and high >= pos.take_profit_price
        if not (hit_stop or hit_take):
            continue

        # If a single bar's range crosses both levels, assume the worse
        # outcome -- same rule, same rationale, as main.py's live sweep.
        if hit_stop:
            level, is_stop, reason = pos.stop_loss_price, True, "Stop-loss"
        else:
            level, is_stop, reason = pos.take_profit_price, False, "Take-profit"

        fill = _stop_take_fill(day_open, level, is_stop)
        pnl = (fill - pos.entry_price) * pos.qty

        state.record_realized(pnl)
        state.closed_trades.append(
            ClosedTrade(
                symbol=symbol, opened_day=pos.opened_day, closed_day=day, qty=pos.qty,
                entry_price=pos.entry_price, exit_price=fill, realized_pnl_usd=pnl,
                is_auto_close=True,
            )
        )
        logger.log_trade(
            day=day, symbol=symbol, action="sell", qty=pos.qty, price=fill,
            realized_pnl_usd=pnl, confidence=None,
            reasoning=f"{reason} automatic (nivell {level:.6g}, preu {fill:.6g}).",
            override_reason="automatic exit", is_auto_close=True,
        )
        del state.open_positions[symbol]
        closed_today.add(symbol)

    return closed_today


# ------------------------------------------------------------------ cost


class CostTracker:
    """Accumulates real per-call token usage and reports actual, not estimated, cost."""

    def __init__(self, model: str):
        self.model = model
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls_made = 0
        price = MODEL_PRICING.get(model)
        if price is None:
            print(f"WARNING: no known pricing for model {model!r}; cost will report as $0.00")
        self.input_price, self.output_price = price or (0.0, 0.0)

    def record(self, usage: TokenUsage) -> None:
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.calls_made += 1

    def actual_cost_usd(self) -> float:
        return (
            self.input_tokens / 1_000_000 * self.input_price
            + self.output_tokens / 1_000_000 * self.output_price
        )

    def progress_line(self) -> str:
        return (
            f"  [cost] {self.calls_made} calls so far -- {self.input_tokens:,} in / "
            f"{self.output_tokens:,} out tokens -- ${self.actual_cost_usd():.4f} actual so far"
        )

    def final_line(self) -> str:
        return (
            f"=== Cost: {self.calls_made} calls, {self.input_tokens:,} input + "
            f"{self.output_tokens:,} output tokens, ${self.actual_cost_usd():.4f} actual "
            f"({self.model}) ==="
        )


def print_upfront_cost_estimate(model: str, expected_calls: int) -> None:
    """A rough estimate BEFORE the run starts -- refined live as real calls land.

    Deliberately not a token-counting API round-trip: the estimate only needs
    to be in the right ballpark before the loop begins, and the CostTracker's
    running total (real usage, printed periodically during the run) is the
    number actually worth trusting. Showing both satisfies "before/during, not
    only after" without pretending a pre-run guess is precise.
    """
    price = MODEL_PRICING.get(model, (0.0, 0.0))
    est_input = expected_calls * ROUGH_INPUT_TOKENS_PER_CALL
    est_output = expected_calls * ROUGH_OUTPUT_TOKENS_PER_CALL
    est_cost = est_input / 1_000_000 * price[0] + est_output / 1_000_000 * price[1]
    print(
        f"Rough pre-run estimate: ~{expected_calls} model calls -> ~${est_cost:.2f} "
        f"({model}). Refined live below as real usage comes in."
    )


# --------------------------------------------------------------- provider


def resolve_provider(provider: str) -> Callable[..., Tuple[Any, TokenUsage]]:
    """Return the `generate_signal_with_usage` function for `provider`.

    Same two providers main.get_provider knows about, same error message
    shape for an unknown one -- this is the one new thing backtest.py needs
    from either provider module (real per-call token usage), which is why it
    calls `_with_usage` directly rather than going through main.get_provider.
    """
    if provider == "claude":
        import signal_generator

        return signal_generator.generate_signal_with_usage
    if provider == "gemini":
        import signal_generator_gemini

        return signal_generator_gemini.generate_signal_with_usage
    raise ValueError(f"unknown provider {provider!r}; expected 'claude' or 'gemini'")


# --------------------------------------------------------------- persistence


_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    simulated_date   TEXT NOT NULL,
    symbol           TEXT NOT NULL,
    action           TEXT NOT NULL,
    qty              REAL NOT NULL,
    price            REAL NOT NULL,
    realized_pnl_usd REAL,
    confidence       REAL,
    reasoning        TEXT,
    override_reason  TEXT,
    is_auto_close    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS equity_curve (
    simulated_date TEXT PRIMARY KEY,
    equity         REAL NOT NULL
);
"""


class BacktestLogger:
    """SQLite persistence for one backtest run, isolated from trading_bot.db.

    Deliberately NOT logger.BotLogger pointed at a different path. BotLogger's
    schema stamps every row with the real wall-clock UTC time
    (`logger.utc_now_iso()`), which is correct for a live cycle and wrong for
    a backtest -- every row from a multi-month simulated run would carry
    today's real timestamp instead of the simulated date it actually happened
    on, and BotLogger.get_today_realized_loss_pct filters by that same real
    date, which would silently break the circuit breaker (see
    BacktestState.note_day). A purpose-built schema with a `simulated_date`
    column is more honest than forcing a live-shaped table to mean something
    it doesn't.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        # One connection held for the object's lifetime, not reopened per
        # call -- besides being the obviously cheaper design for a run that
        # may write thousands of rows, a fresh sqlite3.connect(":memory:")
        # per call would silently discard everything written so far (each
        # connection to ":memory:" is its own separate, empty database),
        # which a reopen-per-call design would only ever surface as a
        # confusing "no such table" from a test, not from the real file path
        # this always runs against in production.
        self.db_path = db_path
        self._db = sqlite3.connect(db_path)
        self._db.executescript(_SCHEMA)
        self._db.commit()

    @contextmanager
    def _conn(self):
        yield self._db
        self._db.commit()

    def log_trade(
        self,
        day: Any,
        symbol: str,
        action: str,
        qty: float,
        price: float,
        realized_pnl_usd: Optional[float],
        confidence: Optional[float],
        reasoning: str,
        override_reason: Optional[str],
        is_auto_close: bool,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO trades (simulated_date, symbol, action, qty, price, "
                "realized_pnl_usd, confidence, reasoning, override_reason, is_auto_close) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(day), symbol, action, float(qty), float(price),
                    realized_pnl_usd, confidence, reasoning, override_reason,
                    1 if is_auto_close else 0,
                ),
            )

    def log_equity(self, day: Any, equity: float) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO equity_curve (simulated_date, equity) VALUES (?, ?) "
                "ON CONFLICT(simulated_date) DO UPDATE SET equity=excluded.equity",
                (str(day), float(equity)),
            )

    def close(self) -> None:
        self._db.close()


# ------------------------------------------------------------- per-symbol day


def run_symbol_for_day(
    state: BacktestState,
    symbol: str,
    asset_class: AssetClass,
    day: Any,
    full_frame: pd.DataFrame,
    mode_settings: "mode.ModeSettings",
    generate_signal_fn: Callable[..., Tuple[Any, TokenUsage]],
    model: str,
    cost: CostTracker,
    logger: BacktestLogger,
    circuit_breaker_loss_pct: float,
    max_risk_pct: float,
    max_absolute_position_pct: float,
) -> None:
    """One symbol, one simulated day: decide, size, and (maybe) fill.

    The anti-lookahead boundary is enforced entirely by what this function
    reads from `full_frame`: `decision_frame_for_day` for everything that
    goes into the model's input, and exactly one read of day's own Open for
    the fill -- never Close, High, or Low of day itself for the decision.
    """
    decision_frame = decision_frame_for_day(full_frame, day)
    if len(decision_frame) < data_fetcher.SMA_LONG + 1:
        return  # not enough lookback yet for this symbol on this day

    try:
        indicators = data_fetcher.compute_indicators(decision_frame)
    except ValueError:
        return

    decision_price = data_fetcher.latest_price(decision_frame)  # day-1's close, never day's own

    existing = state.open_positions.get(symbol)
    existing_position = (
        ExistingPosition(qty=existing.qty, avg_entry_price=existing.entry_price)
        if existing
        else None
    )

    signal_input = SignalInput(
        symbol=symbol,
        asset_class=asset_class,
        current_price=decision_price,
        account_equity_usd=max(state.equity, 0.01),
        existing_position=existing_position,
        technical_indicators=indicators,
        recent_headlines=[],  # limitation 1 -- see module docstring
        market_positioning=None,  # limitation 2 -- see module docstring
    )

    today_loss_pct = state.today_realized_loss_pct()

    if today_loss_pct <= -abs(circuit_breaker_loss_pct):
        final = TradeSignal(
            symbol=symbol, action="hold", confidence=0.0, position_size_pct=0.0,
            stop_loss_price=None, take_profit_price=None,
            reasoning=(
                f"Circuit breaker actiu: perdua realitzada avui ({today_loss_pct:.2f}%) "
                f"supera el limit de -{abs(circuit_breaker_loss_pct):.2f}%. No consultat el model."
            ),
            override_reason=f"circuit breaker tripped at {today_loss_pct:.2f}%; model call skipped",
            raw_action="hold",
        )
    else:
        try:
            raw, usage = generate_signal_fn(
                signal_input, system_prompt=mode_settings.system_prompt, model=model
            )
        except Exception as exc:  # noqa: BLE001 - one symbol/day never kills the run
            print(f"  [{day}] {symbol}: model call failed: {type(exc).__name__}: {exc}")
            return
        cost.record(usage)

        final = risk_manager.validate(
            raw=raw,
            current_price=decision_price,
            today_realized_loss_pct=today_loss_pct,
            circuit_breaker_loss_pct=circuit_breaker_loss_pct,
            max_risk_pct=max_risk_pct,
            max_absolute_position_pct=max_absolute_position_pct,
            min_confidence=mode_settings.min_confidence,
        )

    if day not in full_frame.index:
        return  # symbol has no bar today (e.g. equity on a crypto-only weekend day)
    fill_price = float(full_frame.loc[day, "Open"])

    if final.action == "buy" and existing is None and fill_price > 0:
        # Same equity-relative sizing formula execution.py uses for a buy --
        # only the venue-specific whole-share/notional split is out of scope.
        budget = state.equity * (final.position_size_pct / 100.0)
        qty = budget / fill_price
        if qty > 0:
            state.open_positions[symbol] = OpenPosition(
                symbol=symbol, qty=qty, entry_price=fill_price,
                stop_loss_price=final.stop_loss_price, take_profit_price=final.take_profit_price,
                opened_day=day,
            )
            logger.log_trade(
                day=day, symbol=symbol, action="buy", qty=qty, price=fill_price,
                realized_pnl_usd=None, confidence=final.confidence, reasoning=final.reasoning,
                override_reason=final.override_reason, is_auto_close=False,
            )
    elif final.action == "sell" and existing is not None:
        # Closing uses what is actually held, never a freshly computed size --
        # the same invariant execution.py enforces for a live close.
        pnl = (fill_price - existing.entry_price) * existing.qty
        state.record_realized(pnl)
        state.closed_trades.append(
            ClosedTrade(
                symbol=symbol, opened_day=existing.opened_day, closed_day=day,
                qty=existing.qty, entry_price=existing.entry_price, exit_price=fill_price,
                realized_pnl_usd=pnl, is_auto_close=False,
            )
        )
        logger.log_trade(
            day=day, symbol=symbol, action="sell", qty=existing.qty, price=fill_price,
            realized_pnl_usd=pnl, confidence=final.confidence, reasoning=final.reasoning,
            override_reason=final.override_reason, is_auto_close=False,
        )
        del state.open_positions[symbol]


# ---------------------------------------------------------------- report


def compute_report(
    state: BacktestState,
    full_frames: Dict[str, pd.DataFrame],
    symbols_with_class: List[Tuple[str, AssetClass]],
    start: date,
    end: date,
    starting_equity: float,
) -> Dict[str, Any]:
    """Total return, win rate, drawdown, and -- front and center -- buy-and-hold.

    A strategy that cannot beat holding the same symbols over the same period
    is not demonstrating any edge, so buy_and_hold_return_pct and
    strategy_vs_buy_and_hold_pct are top-level fields here, not an appendix.
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

    # Buy-and-hold: split starting equity evenly across the same symbols,
    # bought at the first simulated day's open, held to the last day's close.
    per_symbol_budget = starting_equity / len(symbols_with_class) if symbols_with_class else 0.0
    buy_and_hold_final = 0.0
    for symbol, _asset_class in symbols_with_class:
        frame = full_frames.get(symbol)
        days_in_range = trading_days_in_range(frame, start, end) if frame is not None else []
        if not days_in_range:
            buy_and_hold_final += per_symbol_budget  # no data at all -- treat as flat
            continue
        entry_open = float(frame.loc[days_in_range[0], "Open"])
        exit_close = float(frame.loc[days_in_range[-1], "Close"])
        shares = per_symbol_budget / entry_open if entry_open > 0 else 0.0
        buy_and_hold_final += shares * exit_close
    buy_and_hold_return_pct = (
        (buy_and_hold_final / starting_equity - 1.0) * 100.0 if starting_equity > 0 else 0.0
    )

    return {
        "start": str(start),
        "end": str(end),
        "symbols": [s for s, _ in symbols_with_class],
        "starting_equity": starting_equity,
        "final_equity": final_equity,
        "total_return_pct": total_return_pct,
        "num_trades": len(closed),
        "win_rate_pct": win_rate_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "buy_and_hold_return_pct": buy_and_hold_return_pct,
        "strategy_vs_buy_and_hold_pct": total_return_pct - buy_and_hold_return_pct,
        "limitations": list(LIMITATIONS),
    }


def format_report(report: Dict[str, Any]) -> str:
    lines = [
        f"=== Backtest report: {report['start']} .. {report['end']} ===",
        f"Symbols: {', '.join(report['symbols'])}",
        f"Starting equity: ${report['starting_equity']:,.2f}   "
        f"Final equity: ${report['final_equity']:,.2f}",
        f"Total return: {report['total_return_pct']:+.2f}%",
        f"Buy-and-hold return (same symbols, same period): "
        f"{report['buy_and_hold_return_pct']:+.2f}%",
        f"Strategy vs. buy-and-hold: {report['strategy_vs_buy_and_hold_pct']:+.2f} pp",
        f"Trades: {report['num_trades']}   Win rate: {report['win_rate_pct']:.1f}%   "
        f"Max drawdown: {report['max_drawdown_pct']:.2f}%",
        "",
        "Limitations (read before trusting this result):",
    ]
    lines += [f"  - {item}" for item in report["limitations"]]
    return "\n".join(lines)


# ------------------------------------------------------------------ run


def run_backtest(
    symbols_with_class: List[Tuple[str, AssetClass]],
    start: date,
    end: date,
    config: Dict[str, Any],
    provider: str = DEFAULT_PROVIDER,
    model: Optional[str] = None,
    starting_equity: float = DEFAULT_STARTING_EQUITY,
    db_path: str = DEFAULT_DB_PATH,
) -> Dict[str, Any]:
    model = model or DEFAULT_MODEL_BY_PROVIDER[provider]
    generate_signal_fn = resolve_provider(provider)

    # The LIVE prompt and confidence threshold -- this is meant to answer
    # "would this strategy be worth running for real", so it evaluates the
    # version of the strategy that would actually run live, not simulation
    # mode's looser threshold (reused via mode.py exactly as main.py does).
    mode_settings = mode.resolve_mode_settings(True, config)

    circuit_breaker_loss_pct = float(config.get("circuit_breaker_loss_pct", 3.0))
    max_risk_pct = float(config.get("max_risk_pct", 1.0))
    max_absolute_position_pct = float(config.get("max_absolute_position_pct", 20.0))

    print(f"=== Backtest | {start} .. {end} | provider={provider} model={model} ===")
    for item in LIMITATIONS:
        print(f"NOTE: {item}")

    full_frames: Dict[str, pd.DataFrame] = {}
    for symbol, _asset_class in symbols_with_class:
        full_frames[symbol] = fetch_historical_ohlcv(symbol, start, end)

    all_days: List[Any] = sorted(
        {d for frame in full_frames.values() for d in trading_days_in_range(frame, start, end)}
    )
    print(f"{len(all_days)} simulated trading days across {len(symbols_with_class)} symbols")

    print_upfront_cost_estimate(model, expected_calls=len(all_days) * len(symbols_with_class))

    state = BacktestState(equity=starting_equity)
    logger = BacktestLogger(db_path)
    cost = CostTracker(model)

    for day in all_days:
        state.note_day(day)
        closed_today = sweep_positions_for_day(state, full_frames, day, logger)

        for symbol, asset_class in symbols_with_class:
            if symbol in closed_today:
                continue
            frame = full_frames.get(symbol)
            if frame is None or day not in frame.index:
                continue
            run_symbol_for_day(
                state=state, symbol=symbol, asset_class=asset_class, day=day, full_frame=frame,
                mode_settings=mode_settings, generate_signal_fn=generate_signal_fn, model=model,
                cost=cost, logger=logger, circuit_breaker_loss_pct=circuit_breaker_loss_pct,
                max_risk_pct=max_risk_pct, max_absolute_position_pct=max_absolute_position_pct,
            )

        equity_today = mark_to_market_equity(state, full_frames, day)
        state.equity_curve.append((day, equity_today))
        logger.log_equity(day, equity_today)

        if cost.calls_made and cost.calls_made % COST_PROGRESS_EVERY_N_CALLS == 0:
            print(cost.progress_line())

    report = compute_report(state, full_frames, symbols_with_class, start, end, starting_equity)
    print()
    print(format_report(report))
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


def _default_symbols(config: Dict[str, Any]) -> List[Tuple[str, AssetClass]]:
    return [
        (entry["symbol"], main_module.infer_asset_class(entry["symbol"], config))
        for entry in (config.get("symbols") or [])
        if isinstance(entry, dict)
    ]


def _parse_symbols_arg(raw: str, config: Dict[str, Any]) -> List[Tuple[str, AssetClass]]:
    symbols = [s.strip() for s in raw.split(",") if s.strip()]
    return [(s, main_module.infer_asset_class(s, config)) for s in symbols]


def main() -> int:
    parser = argparse.ArgumentParser(description="Walk-forward backtest of the trading strategy.")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--symbols", default=None, help="Comma-separated; default: config.yaml/symbols.yaml")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER, choices=["claude", "gemini"])
    parser.add_argument("--model", default=None, help=f"Default per provider: {DEFAULT_MODEL_BY_PROVIDER}")
    parser.add_argument("--starting-equity", type=float, default=DEFAULT_STARTING_EQUITY)
    parser.add_argument("--config", default=main_module.DEFAULT_CONFIG_PATH)
    parser.add_argument("--symbols-file", default=main_module.DEFAULT_SYMBOLS_PATH)
    parser.add_argument("--db", default=DEFAULT_DB_PATH)
    args = parser.parse_args()

    config = main_module.load_config(args.config, args.symbols_file)
    symbols_with_class = (
        _parse_symbols_arg(args.symbols, config) if args.symbols else _default_symbols(config)
    )
    if not symbols_with_class:
        print("No symbols to backtest (empty --symbols and no symbols in config.yaml/symbols.yaml).")
        return 1

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    run_backtest(
        symbols_with_class=symbols_with_class,
        start=start,
        end=end,
        config=config,
        provider=args.provider,
        model=args.model,
        starting_equity=args.starting_equity,
        db_path=args.db,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
