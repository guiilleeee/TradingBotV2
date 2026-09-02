"""Claude signal provider.

Shares SYSTEM_PROMPT and the response schema with the Gemini provider so the two
cannot drift. The system prompt arrives as a parameter -- main.py composes the
effective prompt for the current mode and passes it in. The constant is never
mutated.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from models import SignalInput, SignalOutput, parse_signal_output
from prompts import SIGNAL_JSON_SCHEMA, SYSTEM_PROMPT, build_user_prompt

# Verified against current Anthropic model documentation at build time rather than
# recalled: claude-opus-5 is the current default model ID.
DEFAULT_MODEL = "claude-opus-5"
MAX_TOKENS = 16000

# Required on every request made with an identity-linked API key. See _client().
WORKSPACE_HEADER = "anthropic-workspace-id"

# No temperature is sent. `temperature` was removed from the current Claude model
# family (Opus 4.7 onward) and now returns a 400. Determinism comes from the strict
# JSON schema below instead. The Gemini provider still honours temperature=0.2.


def _client() -> Any:
    # Imported lazily so the test suite and the risk/logging modules do not need
    # the SDK installed just to be imported.
    import anthropic

    # API keys created in the Console under a personal account are
    # "identity-linked" and the API rejects every request from one without an
    # anthropic-workspace-id header:
    #
    #   anthropic-workspace-id is required when authenticating with an
    #   identity-linked API key; send the id of the workspace this request acts in.
    #
    # Checked here rather than left to the API so a misconfigured run fails once,
    # immediately, naming the variable -- instead of a generic 400 on every symbol
    # of every cycle, which is what actually happened.
    workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    if not workspace_id:
        raise RuntimeError(
            "ANTHROPIC_WORKSPACE_ID is not set. Anthropic requires the "
            "anthropic-workspace-id header on every request made with an "
            "identity-linked API key (one created in the Console under a personal "
            "account). Find the workspace id in the Console and set it as the "
            "ANTHROPIC_WORKSPACE_ID environment variable / GitHub secret."
        )

    # Verified against the installed SDK (anthropic 1.3.0): `default_headers` on
    # the constructor is merged into every outgoing request, confirmed by
    # inspecting a built request rather than by reading the parameter name.
    return anthropic.Anthropic(
        default_headers={WORKSPACE_HEADER: workspace_id},
    )


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
