"""Provider call shapes, pinned with fake clients -- no network, no keys.

These catch the drift that hurts most: a provider whose request silently stops
matching the SDK, which shows up as every symbol failing for days.
"""

import json

import pytest

import signal_generator
import signal_generator_gemini
from models import SignalInput, TechnicalIndicators
from prompts import SIGNAL_JSON_SCHEMA, SYSTEM_PROMPT

PAYLOAD = {
    "symbol": "AAPL",
    "action": "buy",
    "confidence": 0.72,
    "position_size_pct": 8.0,
    "stop_loss_price": 305.62,
    "take_profit_price": 364.15,
    "reasoning": "Preu per sobre de la SMA-50.",
}


def make_input(symbol="AAPL"):
    return SignalInput(
        symbol=symbol,
        asset_class="equity",
        current_price=325.13,
        account_equity_usd=1000.0,
        technical_indicators=TechnicalIndicators(
            rsi_14=61.2, sma_20=311.28, sma_50=312.76,
            price_change_pct=2.61, volume_change_pct=26.85,
        ),
        recent_headlines=["Apple news"],
    )


# ------------------------------------------------------------------- Claude


class FakeBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class FakeMessage:
    def __init__(self, text, stop_reason="end_turn"):
        self.content = [FakeBlock(text)]
        self.stop_reason = stop_reason
        self.stop_details = None


class FakeAnthropic:
    def __init__(self, text=json.dumps(PAYLOAD), stop_reason="end_turn", raise_first=None):
        self._text = text
        self._stop_reason = stop_reason
        # raise_first: an exception (or list of exceptions, one per early call)
        # to raise before eventually returning a real response.
        self._raise_first = raise_first
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._raise_first:
            queue = self._raise_first if isinstance(self._raise_first, list) else [self._raise_first]
            if len(self.calls) <= len(queue):
                raise queue[len(self.calls) - 1]
        return FakeMessage(self._text, self._stop_reason)


def test_claude_request_shape():
    client = FakeAnthropic()
    out = signal_generator.generate_signal(make_input(), client=client)

    call = client.calls[0]
    assert call["model"] == signal_generator.DEFAULT_MODEL == "claude-opus-5"
    assert call["system"] == SYSTEM_PROMPT
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["output_config"]["format"]["schema"] == SIGNAL_JSON_SCHEMA
    assert call["messages"][0]["role"] == "user"
    assert "AAPL" in call["messages"][0]["content"]
    assert out.action == "buy"
    assert out.confidence == 0.72


def test_claude_never_sends_temperature():
    # `temperature` was removed from the current model family and is not even in
    # the SDK signature any more; sending it is a 400.
    client = FakeAnthropic()
    signal_generator.generate_signal(make_input(), client=client)
    assert "temperature" not in client.calls[0]


def test_claude_uses_the_prompt_it_is_given():
    client = FakeAnthropic()
    signal_generator.generate_signal(make_input(), system_prompt="CUSTOM", client=client)
    assert client.calls[0]["system"] == "CUSTOM"
    # The shared constant is passed around, never mutated.
    assert SYSTEM_PROMPT != "CUSTOM"


def test_claude_refusal_raises_rather_than_returning_junk():
    client = FakeAnthropic(text="", stop_reason="refusal")
    with pytest.raises(RuntimeError, match="declined"):
        signal_generator.generate_signal(make_input(), client=client)


def test_claude_parses_a_fenced_response():
    client = FakeAnthropic(text=f"```json\n{json.dumps(PAYLOAD)}\n```")
    assert signal_generator.generate_signal(make_input(), client=client).action == "buy"


# --------------------------------------------------------- retry on overload


def _overloaded_error(message="Overloaded"):
    """A real anthropic.OverloadedError -- constructed the same way the SDK
    itself would build one from a 529 response, not a stand-in exception, so
    a test here proves the actual exception type is what gets caught.
    """
    import anthropic
    import httpx2

    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(
        529, request=request, json={"error": {"type": "overloaded_error", "message": message}}
    )
    return anthropic.OverloadedError(
        message, response=response, body={"error": {"type": "overloaded_error", "message": message}}
    )


def _bad_request_error(message="model: field required"):
    import anthropic
    import httpx2

    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(
        400, request=request, json={"error": {"type": "invalid_request_error", "message": message}}
    )
    return anthropic.BadRequestError(
        message, response=response,
        body={"error": {"type": "invalid_request_error", "message": message}},
    )


def test_claude_retries_a_transient_overload_and_uses_the_eventual_success(monkeypatch, capsys):
    monkeypatch.setattr(signal_generator.time, "sleep", lambda seconds: None)
    client = FakeAnthropic(raise_first=_overloaded_error())

    out = signal_generator.generate_signal(make_input(), client=client)

    assert out.action == "buy"  # the eventual (second-attempt) success is what's used
    assert len(client.calls) == 2
    assert "attempt 1/3" in capsys.readouterr().out


def test_claude_gives_up_after_max_attempts_of_persistent_overload(monkeypatch):
    monkeypatch.setattr(signal_generator.time, "sleep", lambda seconds: None)
    client = FakeAnthropic(raise_first=[_overloaded_error(), _overloaded_error(), _overloaded_error()])

    import anthropic

    with pytest.raises(anthropic.OverloadedError):
        signal_generator.generate_signal(make_input(), client=client)
    assert len(client.calls) == signal_generator.MAX_ATTEMPTS == 3


def test_claude_does_not_retry_a_real_rejection(monkeypatch):
    """A non-transient error (e.g. a real 400) fails immediately, same as
    today -- no retry, no backoff sleep at all.
    """
    sleep_calls = []
    monkeypatch.setattr(signal_generator.time, "sleep", lambda seconds: sleep_calls.append(seconds))
    client = FakeAnthropic(raise_first=_bad_request_error())

    import anthropic

    with pytest.raises(anthropic.BadRequestError):
        signal_generator.generate_signal(make_input(), client=client)
    assert len(client.calls) == 1  # no retry at all
    assert sleep_calls == []


# ------------------------------------------------------------------- Gemini


class FakeGeminiResponse:
    def __init__(self, text):
        self.text = text


class FakeGemini:
    def __init__(self, text=json.dumps(PAYLOAD), fail_first=False, raise_first=None):
        self._text = text
        self._fail_first = fail_first
        # raise_first: an exception (or list of exceptions, one per early call)
        # to raise before eventually returning a real response.
        self._raise_first = raise_first
        self.calls = []
        self.models = self

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self._fail_first and len(self.calls) == 1:
            raise ValueError("Unsupported schema construct")
        if self._raise_first:
            queue = self._raise_first if isinstance(self._raise_first, list) else [self._raise_first]
            if len(self.calls) <= len(queue):
                raise queue[len(self.calls) - 1]
        return FakeGeminiResponse(self._text)


def test_gemini_request_shape():
    client = FakeGemini()
    out = signal_generator_gemini.generate_signal(make_input(), client=client)

    call = client.calls[0]
    config = call["config"]
    assert call["model"] == signal_generator_gemini.DEFAULT_MODEL == "gemini-3.7-flash"
    assert "preview" not in call["model"]  # a preview ID would vanish under a cron job
    assert config.system_instruction == SYSTEM_PROMPT
    assert config.temperature == 0.2
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema == SIGNAL_JSON_SCHEMA
    assert out.action == "buy"


def test_gemini_falls_back_when_the_schema_dialect_is_rejected():
    # A schema quarrel must not silently kill every symbol for days.
    client = FakeGemini(fail_first=True)
    out = signal_generator_gemini.generate_signal(make_input(), client=client)

    assert len(client.calls) == 2
    assert client.calls[1]["config"].response_json_schema is None
    assert client.calls[1]["config"].response_mime_type == "application/json"
    assert "schema" in client.calls[1]["contents"].lower()
    assert out.action == "buy"


# --------------------------------------------------------- retry on overload


def _gemini_server_error(code=503, message="overloaded"):
    from google.genai import errors

    return errors.ServerError(code, {"error": {"message": message}}, None)


def _gemini_rate_limit_error(message="rate limit exceeded"):
    from google.genai import errors

    return errors.ClientError(429, {"error": {"message": message}}, None)


def _gemini_bad_request_error(message="invalid argument"):
    from google.genai import errors

    return errors.ClientError(400, {"error": {"message": message}}, None)


def test_gemini_retries_a_transient_server_error_and_uses_the_eventual_success(monkeypatch, capsys):
    monkeypatch.setattr(signal_generator_gemini.time, "sleep", lambda seconds: None)
    client = FakeGemini(raise_first=_gemini_server_error())

    out = signal_generator_gemini.generate_signal(make_input(), client=client)

    assert out.action == "buy"
    assert len(client.calls) == 2
    assert "attempt 1/3" in capsys.readouterr().out


def test_gemini_retries_a_429_rate_limit_the_same_way(monkeypatch):
    monkeypatch.setattr(signal_generator_gemini.time, "sleep", lambda seconds: None)
    client = FakeGemini(raise_first=_gemini_rate_limit_error())

    out = signal_generator_gemini.generate_signal(make_input(), client=client)
    assert out.action == "buy"
    assert len(client.calls) == 2


def test_gemini_gives_up_after_max_attempts_of_persistent_overload_then_falls_back(monkeypatch):
    """After 3 failed attempts on the schema-mode call, the existing fallback-
    to-plain-JSON-mode path takes over (itself retried the same way) -- a
    reasonable degrade, not a misfire, since the outer except is unchanged.
    """
    monkeypatch.setattr(signal_generator_gemini.time, "sleep", lambda seconds: None)
    client = FakeGemini(
        raise_first=[_gemini_server_error(), _gemini_server_error(), _gemini_server_error()]
    )

    out = signal_generator_gemini.generate_signal(make_input(), client=client)
    assert out.action == "buy"
    assert len(client.calls) == 4  # 3 exhausted attempts + 1 fallback success
    assert client.calls[3]["config"].response_json_schema is None  # the fallback call


def test_gemini_does_not_retry_a_real_client_error_within_the_retry_loop(monkeypatch):
    """A real 400 is not transient -- _call_with_retries must not retry it,
    though the existing outer except still triggers the schema-dialect
    fallback (unrelated to retry, unchanged behaviour).
    """
    sleep_calls = []
    monkeypatch.setattr(signal_generator_gemini.time, "sleep", lambda seconds: sleep_calls.append(seconds))
    client = FakeGemini(raise_first=_gemini_bad_request_error())

    out = signal_generator_gemini.generate_signal(make_input(), client=client)
    assert out.action == "buy"  # the fallback call succeeds
    assert len(client.calls) == 2  # one failed attempt (not retried) + one fallback
    assert sleep_calls == []  # never slept -- the 400 was never treated as transient


def test_gemini_uses_the_prompt_it_is_given():
    client = FakeGemini()
    signal_generator_gemini.generate_signal(make_input(), system_prompt="CUSTOM", client=client)
    assert client.calls[0]["config"].system_instruction == "CUSTOM"


# ------------------------------------------------------- both stay in sync


def test_both_providers_share_one_prompt_and_one_schema():
    assert signal_generator.SYSTEM_PROMPT is signal_generator_gemini.SYSTEM_PROMPT
    assert signal_generator.SIGNAL_JSON_SCHEMA is signal_generator_gemini.SIGNAL_JSON_SCHEMA


def test_both_providers_have_the_same_signature():
    import inspect

    claude = inspect.signature(signal_generator.generate_signal).parameters
    gemini = inspect.signature(signal_generator_gemini.generate_signal).parameters
    assert list(claude) == list(gemini)
    assert claude["system_prompt"].default == gemini["system_prompt"].default == SYSTEM_PROMPT


def test_both_providers_reject_a_mismatched_symbol_echo():
    payload = dict(PAYLOAD, symbol="TSLA")
    claude_out = signal_generator.generate_signal(
        make_input("AAPL"), client=FakeAnthropic(text=json.dumps(payload))
    )
    gemini_out = signal_generator_gemini.generate_signal(
        make_input("AAPL"), client=FakeGemini(text=json.dumps(payload))
    )
    assert claude_out.symbol == "AAPL"
    assert gemini_out.symbol == "AAPL"


def test_the_schema_matches_what_signal_output_requires():
    from models import SignalOutput

    assert set(SIGNAL_JSON_SCHEMA["required"]) == set(SignalOutput.model_fields)
    assert SIGNAL_JSON_SCHEMA["additionalProperties"] is False
    assert SIGNAL_JSON_SCHEMA["properties"]["action"]["enum"] == ["buy", "sell", "hold"]


# ------------------------------------------------ identity-linked workspace id


def test_client_attaches_the_workspace_header(monkeypatch):
    """The header must be on the constructed client, set from the env var alone."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_TEST123")

    captured = {}

    class FakeAnthropicModule:
        @staticmethod
        def Anthropic(**kwargs):
            captured.update(kwargs)
            return "client"

    monkeypatch.setitem(__import__("sys").modules, "anthropic", FakeAnthropicModule)

    assert signal_generator._client() == "client"
    assert captured["default_headers"] == {"anthropic-workspace-id": "wrkspc_TEST123"}


def test_workspace_header_actually_reaches_an_outgoing_request(monkeypatch):
    """End-to-end through the real SDK: the header is on the wire.

    Asserting on the constructor argument alone would only prove we passed a
    keyword the SDK happens to accept. This builds an actual request and reads
    its headers back, so a future SDK change that stops honouring
    `default_headers` fails here rather than in production.
    """
    import anthropic
    from anthropic._models import FinalRequestOptions

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_TEST123")

    client = signal_generator._client()
    assert isinstance(client, anthropic.Anthropic)

    request = client._build_request(
        FinalRequestOptions.construct(
            method="post",
            url="/v1/messages",
            json_data={"model": "m", "max_tokens": 1, "messages": []},
        )
    )
    headers = {k.lower(): v for k, v in request.headers.items()}
    assert headers.get("anthropic-workspace-id") == "wrkspc_TEST123"


def test_missing_workspace_id_fails_immediately_and_names_the_variable(monkeypatch):
    # Better one clear error than Anthropic's generic 400 repeated on every
    # symbol of every cycle, which is what this replaces.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)

    with pytest.raises(RuntimeError, match="ANTHROPIC_WORKSPACE_ID"):
        signal_generator._client()


def test_empty_workspace_id_is_treated_as_missing(monkeypatch):
    # An empty GitHub secret expands to "", which would otherwise sail through
    # and produce the exact 400 this check exists to prevent.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "")

    with pytest.raises(RuntimeError, match="ANTHROPIC_WORKSPACE_ID"):
        signal_generator._client()


def test_an_injected_client_does_not_require_the_workspace_id(monkeypatch):
    # Callers passing their own client (and the tests above) must not be forced
    # to set an env var for a client they already built.
    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
    out = signal_generator.generate_signal(make_input(), client=FakeAnthropic())
    assert out.action == "buy"


# ---------------------------------------------------------- token usage (backtest.py)


class FakeUsage:
    def __init__(self, input_tokens, output_tokens):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class FakeAnthropicWithUsage(FakeAnthropic):
    def __init__(self, *a, input_tokens=120, output_tokens=45, **kw):
        super().__init__(*a, **kw)
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens

    def create(self, **kwargs):
        message = super().create(**kwargs)
        message.usage = FakeUsage(self._input_tokens, self._output_tokens)
        return message


def test_claude_generate_signal_discards_usage_but_still_works():
    client = FakeAnthropicWithUsage()
    out = signal_generator.generate_signal(make_input(), client=client)
    assert out.action == "buy"


def test_claude_with_usage_returns_real_token_counts():
    client = FakeAnthropicWithUsage(input_tokens=500, output_tokens=80)
    out, usage = signal_generator.generate_signal_with_usage(make_input(), client=client)
    assert out.action == "buy"
    assert usage.input_tokens == 500
    assert usage.output_tokens == 80


def test_claude_with_usage_degrades_to_zero_when_the_fake_has_no_usage_attr():
    # A fake client (or a real response shape change) with no .usage attribute
    # at all must not raise -- it degrades to zero, same fail-soft posture as
    # everything else optional in this project.
    client = FakeAnthropic()  # no .usage set anywhere
    out, usage = signal_generator.generate_signal_with_usage(make_input(), client=client)
    assert out.action == "buy"
    assert usage.input_tokens == 0
    assert usage.output_tokens == 0


class FakeGeminiUsage:
    def __init__(self, prompt_token_count, candidates_token_count):
        self.prompt_token_count = prompt_token_count
        self.candidates_token_count = candidates_token_count


class FakeGeminiWithUsage(FakeGemini):
    def __init__(self, *a, prompt_tokens=90, candidates_tokens=30, **kw):
        super().__init__(*a, **kw)
        self._prompt_tokens = prompt_tokens
        self._candidates_tokens = candidates_tokens

    def generate_content(self, **kwargs):
        response = super().generate_content(**kwargs)
        response.usage_metadata = FakeGeminiUsage(self._prompt_tokens, self._candidates_tokens)
        return response


def test_gemini_with_usage_returns_real_token_counts():
    client = FakeGeminiWithUsage(prompt_tokens=400, candidates_tokens=60)
    out, usage = signal_generator_gemini.generate_signal_with_usage(make_input(), client=client)
    assert out.action == "buy"
    assert usage.input_tokens == 400
    assert usage.output_tokens == 60


def test_gemini_with_usage_degrades_to_zero_when_the_fake_has_no_usage_metadata():
    client = FakeGemini()  # no .usage_metadata set anywhere
    out, usage = signal_generator_gemini.generate_signal_with_usage(make_input(), client=client)
    assert out.action == "buy"
    assert usage.input_tokens == 0
    assert usage.output_tokens == 0


def test_both_providers_expose_the_same_with_usage_signature():
    import inspect

    claude_params = list(inspect.signature(signal_generator.generate_signal_with_usage).parameters)
    gemini_params = list(inspect.signature(signal_generator_gemini.generate_signal_with_usage).parameters)
    assert claude_params == gemini_params
