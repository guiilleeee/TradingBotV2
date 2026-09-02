"""Claude signal provider.

Shares SYSTEM_PROMPT and the response schema with the Gemini provider so the two
cannot drift. The system prompt arrives as a parameter -- main.py composes the
effective prompt for the current mode and passes it in. The constant is never
mutated.
"""

from __future__ import annotations

from typing import Any, Optional

from models import SignalInput, SignalOutput, parse_signal_output
from prompts import SIGNAL_JSON_SCHEMA, SYSTEM_PROMPT, build_user_prompt

# Verified against current Anthropic model documentation at build time rather than
# recalled: claude-opus-5 is the current default model ID.
DEFAULT_MODEL = "claude-opus-5"
MAX_TOKENS = 16000

# No temperature is sent. `temperature` was removed from the current Claude model
# family (Opus 4.7 onward) and now returns a 400. Determinism comes from the strict
# JSON schema below instead. The Gemini provider still honours temperature=0.2.


def _client() -> Any:
    # Imported lazily so the test suite and the risk/logging modules do not need
    # the SDK installed just to be imported.
    import anthropic

    return anthropic.Anthropic()


def generate_signal(
    signal_input: SignalInput,
    system_prompt: str = SYSTEM_PROMPT,
    model: str = DEFAULT_MODEL,
    client: Optional[Any] = None,
) -> SignalOutput:
    """Ask Claude for one decision on one symbol."""
    client = client or _client()

    response = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": build_user_prompt(signal_input.model_dump_json(indent=2)),
            }
        ],
        output_config={"format": {"type": "json_schema", "schema": SIGNAL_JSON_SCHEMA}},
    )

    # A refusal comes back as HTTP 200 with no usable content, so check before
    # reading blocks. The caller treats this as a per-symbol failure and moves on;
    # a symbol we cannot analyse is a symbol we do not trade.
    if getattr(response, "stop_reason", None) == "refusal":
        details = getattr(response, "stop_details", None)
        raise RuntimeError(
            f"{signal_input.symbol}: Claude declined the request (stop_reason=refusal, "
            f"details={details})"
        )

    text = next((b.text for b in response.content if getattr(b, "type", None) == "text"), "")
    return parse_signal_output(text, signal_input.symbol)
