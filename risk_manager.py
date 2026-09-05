"""Code-enforced risk layer.

The model proposes; this module disposes. Every rule here can only ever make a
trade smaller or turn it into a hold -- there is no path through this function
that makes a position larger than the caller's caps allow.

Deliberately knows nothing about live vs simulation: `min_confidence` arrives
already resolved by the caller. One responsibility, one signature.
"""

from __future__ import annotations

from typing import List, Optional

from models import SignalOutput, TradeSignal

# A stop this close to the entry is not a risk boundary, it is noise. Sizing off it
# would divide by a near-zero number and demand an enormous position.
MIN_STOP_DISTANCE_PCT = 0.003  # 0.3%

# Below this reward:risk, a setup cannot be profitable at any win rate this
# strategy has actually shown -- the diagnostic on the historical-screening
# backtest found a realised 1.18 against a 37.5% win rate that needed ~1.67 to
# break even. 1.5 is deliberately a bit under that break-even figure rather
# than tuned to it exactly: the win rate itself may improve once the worst
# setups stop qualifying at all, and this is a hypothesis to re-test, not a
# number picked to match one past sample. Config-driven (min_reward_risk_ratio
# in config.yaml); this default only protects callers that predate the
# parameter, e.g. existing tests -- every real caller reads it from config.
DEFAULT_MIN_REWARD_RISK_RATIO = 1.5


def validate(
    raw: SignalOutput,
    current_price: float,
    today_realized_loss_pct: float,
    circuit_breaker_loss_pct: float,
    max_risk_pct: float,
    max_absolute_position_pct: float,
    min_confidence: float,
    min_reward_risk_ratio: float = DEFAULT_MIN_REWARD_RISK_RATIO,
) -> TradeSignal:
    """Apply the risk rules in order and return the signal execution may act on.

    Args:
        raw: the model's unmodified output.
        current_price: last close, used for the stop-distance calculation.
        today_realized_loss_pct: today's realised P&L as a percentage of equity.
            Negative means a loss.
        circuit_breaker_loss_pct: positive magnitude at which trading halts for the day.
        max_risk_pct: percent of equity to risk on one trade (e.g. 1.0 for 1%).
        max_absolute_position_pct: hard cap on position size as a percent of equity.
        min_confidence: already resolved for the current mode by the caller.
        min_reward_risk_ratio: minimum acceptable reward:risk (take-profit distance
            over stop-loss distance, both from current_price). Structural, not part
            of the live/simulation confidence-threshold split -- applies identically
            in both modes.
    """
    reasons: List[str] = []

    raw_action = raw.action
    action = raw.action
    size = raw.position_size_pct
    stop: Optional[float] = raw.stop_loss_price
    take: Optional[float] = raw.take_profit_price

    # 1. Circuit breaker. Checked first so nothing else can talk us past it.
    breaker = abs(circuit_breaker_loss_pct)
    if action in ("buy", "sell") and today_realized_loss_pct <= -breaker:
        reasons.append(
            f"circuit breaker: today's realised P&L {today_realized_loss_pct:.2f}% is at or "
            f"beyond the -{breaker:.2f}% daily limit"
        )
        action = "hold"

    # 2. Confidence threshold.
    if action != "hold" and raw.confidence < min_confidence:
        reasons.append(
            f"confidence {raw.confidence:.2f} is below the {min_confidence:.2f} minimum for this mode"
        )
        action = "hold"

    # 3. Clearly broken model sizing. This rejects nonsense output only -- the real
    #    number comes from rule 4 below and overwrites whatever the model suggested.
    if action in ("buy", "sell") and size <= 0:
        reasons.append(
            f"model returned a non-positive position_size_pct ({size}) for a {action}"
        )
        action = "hold"

    # 4. Risk-based sizing from the stop distance.
    if action in ("buy", "sell") and stop is not None:
        if current_price <= 0:
            reasons.append(f"current_price {current_price} is not positive; cannot size the trade")
            action = "hold"
        else:
            stop_distance_pct = abs(current_price - stop) / current_price
            if stop_distance_pct < MIN_STOP_DISTANCE_PCT:
                reasons.append(
                    f"stop-loss {stop:.6g} sits {stop_distance_pct * 100:.3f}% from price "
                    f"{current_price:.6g}, under the {MIN_STOP_DISTANCE_PCT * 100:.1f}% minimum "
                    "to be a credible risk boundary"
                )
                action = "hold"
            else:
                # 4b. Reward:risk floor. Buy only, deliberately -- a sell's
                # stop_loss_price/take_profit_price are schema-required (every
                # buy/sell must carry both, per prompts.py's hard rule 2) but
                # never actually used to manage anything once a sell executes:
                # _update_ledger closes the position from the ledger's own
                # qty/entry price the instant action == "sell" and never reads
                # either field again. Applying this floor to a sell would risk
                # trapping the bot in a position the model has already decided
                # to exit, based on numbers that don't describe anything real.
                # Reward:risk is an entry question; only checkable once a valid
                # stop distance exists and a take-profit is actually present
                # (a missing take_profit is rejected on its own by rule 5
                # below, never treated as a reward:risk failure here). Placed
                # before sizing is computed: a setup this rule rejects should
                # never have a size computed for it in the first place, even
                # though rule 6's catch-all zeroing would discard it either way.
                if action == "buy" and take is not None:
                    risk_distance = abs(current_price - stop)
                    reward_distance = abs(take - current_price)
                    reward_risk_ratio = reward_distance / risk_distance
                    if reward_risk_ratio < min_reward_risk_ratio:
                        reasons.append(
                            f"reward:risk {reward_risk_ratio:.2f} is below the "
                            f"{min_reward_risk_ratio:.2f} minimum (take-profit {take:.6g}, "
                            f"stop-loss {stop:.6g}, price {current_price:.6g})"
                        )
                        action = "hold"

                if action in ("buy", "sell"):
                    computed = max_risk_pct / stop_distance_pct
                    if computed > max_absolute_position_pct:
                        reasons.append(
                            f"risk-based size {computed:.2f}% clamped to the "
                            f"{max_absolute_position_pct:.2f}% absolute position cap"
                        )
                        computed = max_absolute_position_pct
                    size = computed

    # 5. Missing exit levels.
    if action in ("buy", "sell"):
        missing = [
            name
            for name, value in (("stop_loss_price", stop), ("take_profit_price", take))
            if value is None
        ]
        if missing:
            reasons.append(f"{action} is missing required {' and '.join(missing)}")
            action = "hold"

    # 6. Catch-all zeroing. Any route to hold lands here, including a hold the model
    #    produced on its own -- this is not per-rule cleanup.
    if action == "hold":
        size = 0.0
        stop = None
        take = None

    return TradeSignal(
        symbol=raw.symbol,
        action=action,
        confidence=raw.confidence,
        position_size_pct=size,
        stop_loss_price=stop,
        take_profit_price=take,
        reasoning=raw.reasoning,
        override_reason="; ".join(reasons) if reasons else None,
        raw_action=raw_action,
    )
