"""Pydantic v2 models shared across the trading pipeline.

Every value that crosses a module boundary is one of these types, so a bad
number is rejected at the boundary rather than three modules later.
"""

from __future__ import annotations

import json
import re
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Action = Literal["buy", "sell", "hold"]
AssetClass = Literal["equity", "crypto"]
ExecutionStatus = Literal["success", "skipped", "error", "dry_run"]


class ExecutionResult(BaseModel):
    """Outcome of one attempted trade on one venue."""

    model_config = ConfigDict(extra="forbid")

    status: ExecutionStatus
    order_id: Optional[str] = None
    fill_price: Optional[float] = None
    message: str = ""
    realized_pnl_usd: Optional[float] = None
    # qty is load-bearing: the simulated ledger reconstructs positions from it.
    # An earlier build omitted it and position tracking silently broke.
    qty: Optional[float] = None

    @model_validator(mode="after")
    def _filled_orders_must_report_qty(self) -> "ExecutionResult":
        if self.status in ("success", "dry_run"):
            if self.qty is None or self.qty <= 0:
                raise ValueError(
                    f"ExecutionResult(status={self.status!r}) must carry a positive qty "
                    "so the simulated ledger can book the position"
                )
        return self


class TechnicalIndicators(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rsi_14: float = Field(ge=0, le=100)
    sma_20: float = Field(gt=0)
    sma_50: float = Field(gt=0)
    # Change since the PREVIOUS DAILY BAR -- deliberately not called "24h".
    # For equities the previous bar can be ~3 calendar days away over a weekend
    # or holiday, so this is not a true rolling 24-hour window.
    price_change_pct: float
    volume_change_pct: float


class ExistingPosition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    qty: float
    avg_entry_price: float = Field(gt=0)


class SignalInput(BaseModel):
    """Everything the model is allowed to see about one symbol."""

    model_config = ConfigDict(extra="forbid")

    symbol: str
    asset_class: AssetClass
    current_price: float = Field(gt=0)
    account_equity_usd: float = Field(gt=0)
    existing_position: Optional[ExistingPosition] = None
    technical_indicators: TechnicalIndicators
    recent_headlines: List[str] = Field(default_factory=list)
    # How large, successful Hyperliquid wallets are currently positioned in this
    # asset, as one pre-summarised sentence. Directional bias only, exactly the
    # same standing as recent_headlines -- see hard rule 5 in the system prompt.
    # None whenever unavailable, which is always the case for equities. Never
    # fabricated, and never a substitute for the technical case.
    market_positioning: Optional[str] = None


class SignalOutput(BaseModel):
    """Raw model output, before the risk manager touches it."""

    model_config = ConfigDict(extra="forbid")

    symbol: str
    action: Action
    confidence: float = Field(ge=0, le=1)
    # Advisory only. Real sizing is computed downstream from the stop distance.
    position_size_pct: float = Field(default=0.0, ge=0, le=100)
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    reasoning: str = ""

    @field_validator("action", mode="before")
    @classmethod
    def _normalise_action(cls, v: object) -> object:
        if isinstance(v, str):
            return v.strip().lower()
        return v

    @model_validator(mode="after")
    def _hold_carries_no_trade_levels(self) -> "SignalOutput":
        if self.action == "hold":
            object.__setattr__(self, "stop_loss_price", None)
            object.__setattr__(self, "take_profit_price", None)
            object.__setattr__(self, "position_size_pct", 0.0)
        return self


class TokenUsage(BaseModel):
    """Token counts from one provider call.

    Not needed by the live 4h cycle (main.py never reads this), but backtest.py
    needs real per-call usage to report an accurate running/actual cost rather
    than an estimate that never gets corrected. Kept as a plain counts holder --
    pricing tables and cost math belong to whoever is paying, not to the
    provider modules.
    """

    model_config = ConfigDict(extra="forbid")

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class TradeSignal(SignalOutput):
    """Post-risk-manager signal. This is what execution acts on."""

    model_config = ConfigDict(extra="forbid")

    # Joined list of every risk rule that fired, or None if the model's call stood.
    override_reason: Optional[str] = None
    # The model's original action before any override. Kept for auditability --
    # without it there is no way to tell a real "hold" from an overridden "buy".
    raw_action: Action


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def parse_signal_output(text: str, expected_symbol: str) -> SignalOutput:
    """Turn a provider's raw text response into a validated SignalOutput.

    Structured-output modes should already hand back bare JSON, but a fenced block
    still shows up often enough that stripping it is cheaper than a failed cycle.

    The symbol is overwritten with the one we asked about. A model that echoes the
    wrong ticker would otherwise route a real order to the wrong instrument.
    """
    candidate = (text or "").strip()
    if not candidate:
        raise ValueError(f"{expected_symbol}: provider returned an empty response")

    fenced = _JSON_FENCE_RE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    elif not candidate.startswith("{"):
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(
                f"{expected_symbol}: no JSON object in provider response: {candidate[:200]!r}"
            )
        candidate = candidate[start : end + 1]

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{expected_symbol}: provider response was not valid JSON ({exc}): {candidate[:200]!r}"
        ) from exc

    payload["symbol"] = expected_symbol
    return SignalOutput.model_validate(payload)
