"""Gemini signal provider.

Same interface, same system prompt, same output schema as the Claude provider --
`generate_signal(signal_input, system_prompt=..., ...) -> SignalOutput`. Only the
transport differs.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Optional, Tuple, TypeVar

from models import SignalInput, SignalOutput, TokenUsage, parse_signal_output
from prompts import SIGNAL_JSON_SCHEMA, SYSTEM_PROMPT, build_user_prompt

# Verified against the live Gemini model list at build time rather than recalled --
# a hardcoded-from-memory model ID was deprecated mid-project once already.
# gemini-3.7-flash is the current latest *stable* Flash model. Deliberately not a
# `-preview` ID: those are exactly what disappears underneath a scheduled job.
DEFAULT_MODEL = "gemini-3.7-flash"

TEMPERATURE = 0.2

# Same reasoning as signal_generator.py's retry: a transient, server-side
# condition should never cost a symbol its whole cycle. Gemini's equivalent of
# Anthropic's 529 is a 5xx (ServerError -- overloaded/unavailable) or a 429
# (ClientError, rate limit exceeded); every other ClientError code (400 bad
# request, 401/403 auth, 404, ...) is a real rejection and fails immediately,
# exactly as before this existed.
MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = (1, 3)  # wait after attempt 1 fails, then after attempt 2

T = TypeVar("T")


def _client() -> Any:
    # Lazy import: the SDK is only needed when this provider is actually selected.
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    return genai.Client(api_key=api_key)


def _is_transient(exc: BaseException) -> bool:
    from google.genai import errors

    if isinstance(exc, errors.ServerError):
        return True  # any 5xx -- overloaded/unavailable, always transient
    if isinstance(exc, errors.ClientError):
        return getattr(exc, "code", None) == 429  # rate limit only; a real 4xx is not
    return False


def _call_with_retries(make_call: Callable[[], T], symbol: str) -> T:
    """Retry `make_call` only on a transient error (see module docstring above
    this constant block). Every other error -- including a genuine
    schema-dialect rejection, which the caller's own except-and-fall-back
    logic still needs to see -- propagates immediately, unretried. Logged
    plainly on every attempt so a cycle's output still shows what happened.
    """
    last_exc: Optional[BaseException] = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return make_call()
        except Exception as exc:
            if not _is_transient(exc):
                raise
            last_exc = exc
            print(f"{symbol}: Gemini transient error (attempt {attempt}/{MAX_ATTEMPTS}): {exc}")
            if attempt == MAX_ATTEMPTS:
                raise
            time.sleep(RETRY_BACKOFF_SECONDS[attempt - 1])
    raise last_exc  # unreachable -- the loop above always returns or raises


def generate_signal(
    signal_input: SignalInput,
    system_prompt: str = SYSTEM_PROMPT,
    model: str = DEFAULT_MODEL,
    client: Optional[Any] = None,
) -> SignalOutput:
    """Ask Gemini for one decision on one symbol.

    Thin wrapper over generate_signal_with_usage that discards token usage --
    kept as the stable public entry point so nothing about its behaviour or
    signature changes for any existing caller.
    """
    output, _usage = generate_signal_with_usage(
        signal_input, system_prompt=system_prompt, model=model, client=client
    )
    return output


def generate_signal_with_usage(
    signal_input: SignalInput,
    system_prompt: str = SYSTEM_PROMPT,
    model: str = DEFAULT_MODEL,
    client: Optional[Any] = None,
) -> Tuple[SignalOutput, TokenUsage]:
    """Same call as generate_signal, but also returns token usage.

    Exists for backtest.py's cost tracking. Verified against the installed SDK
    (google-genai): usage lives on `response.usage_metadata`, with
    `prompt_token_count` / `candidates_token_count` -- not `.usage.input_tokens`
    the way Claude's response shapes it. Extraction is defensive so a fake
    client in a test that never sets `usage_metadata` degrades to zero.
    """
    from google.genai import types

    client = client or _client()
    user_prompt = build_user_prompt(signal_input.model_dump_json(indent=2))

    try:
        response = _call_with_retries(
            lambda: client.models.generate_content(
                model=model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=TEMPERATURE,
                    response_mime_type="application/json",
                    response_json_schema=SIGNAL_JSON_SCHEMA,
                ),
            ),
            signal_input.symbol,
        )
    except Exception:
        # Gemini's schema dialect is stricter than Claude's about some JSON Schema
        # constructs. Rather than let a schema-dialect quarrel silently kill every
        # symbol for days, fall back to JSON mode with the schema inlined in the
        # prompt. parse_signal_output validates the result either way. Also
        # reached if the schema-mode call above exhausted its own retries on a
        # persistent transient error -- falling back to plain JSON mode (itself
        # retried the same way) is a reasonable degrade, not a misfire.
        response = _call_with_retries(
            lambda: client.models.generate_content(
                model=model,
                contents=f"{user_prompt}\n\nReturn JSON matching this schema exactly:\n{SIGNAL_JSON_SCHEMA}",
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=TEMPERATURE,
                    response_mime_type="application/json",
                ),
            ),
            signal_input.symbol,
        )

    output = parse_signal_output(getattr(response, "text", "") or "", signal_input.symbol)

    usage_obj = getattr(response, "usage_metadata", None)
    usage = TokenUsage(
        input_tokens=getattr(usage_obj, "prompt_token_count", None) or 0,
        output_tokens=getattr(usage_obj, "candidates_token_count", None) or 0,
    )
    return output, usage
