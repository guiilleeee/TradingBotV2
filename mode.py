"""Live vs simulation resolution.

This is the one guarantee in the project that must never regress: the live path
is structurally incapable of picking up the simulation addendum or the simulation
confidence threshold. That is enforced by construction -- `_live_settings` does not
reference SIMULATION_ADDENDUM or the simulation config key at all, so no value of
any config option can route them into a live run. It does not depend on a comment
telling someone to revert something before going live.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from prompts import SIMULATION_ADDENDUM, SYSTEM_PROMPT

DEFAULT_MIN_CONFIDENCE_LIVE = 0.65
DEFAULT_MIN_CONFIDENCE_SIMULATION = 0.40


@dataclass(frozen=True)
class ModeSettings:
    is_live: bool
    system_prompt: str
    min_confidence: float

    @property
    def label(self) -> str:
        return "LIVE" if self.is_live else "SIMULATION"


def _live_settings(config: Mapping[str, Any]) -> ModeSettings:
    """Live mode. Base prompt verbatim, live threshold, nothing else in scope."""
    return ModeSettings(
        is_live=True,
        system_prompt=SYSTEM_PROMPT,
        min_confidence=float(config.get("min_confidence_live", DEFAULT_MIN_CONFIDENCE_LIVE)),
    )


def _simulation_settings(config: Mapping[str, Any]) -> ModeSettings:
    """Simulation mode. Base prompt plus the addendum, simulation threshold."""
    return ModeSettings(
        is_live=False,
        system_prompt=SYSTEM_PROMPT + SIMULATION_ADDENDUM,
        min_confidence=float(
            config.get("min_confidence_simulation", DEFAULT_MIN_CONFIDENCE_SIMULATION)
        ),
    )


def resolve_is_live(config: Mapping[str, Any]) -> bool:
    """Read `live_execution` strictly.

    Only a real boolean True enables live trading. A string, a 1, or a typo resolves
    to simulation -- the failure mode of a malformed config must be "paper trade",
    never "spend real money".
    """
    return config.get("live_execution") is True


def resolve_mode_settings(is_live: bool, config: Mapping[str, Any]) -> ModeSettings:
    """Pick the prompt and confidence threshold for this run."""
    settings = _live_settings(config) if is_live is True else _simulation_settings(config)

    # Belt and braces: assert the invariant the whole design exists to protect.
    if settings.is_live and SIMULATION_ADDENDUM in settings.system_prompt:
        raise AssertionError("live prompt contains the simulation addendum")

    return settings
