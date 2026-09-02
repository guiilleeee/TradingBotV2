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
    def __init__(self, text=json.dumps(PAYLOAD), stop_reason="end_turn"):
        self._text = text
        self._stop_reason = stop_reason
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
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


# ------------------------------------------------------------------- Gemini


class FakeGeminiResponse:
    def __init__(self, text):
        self.text = text


class FakeGemini:
    def __init__(self, text=json.dumps(PAYLOAD), fail_first=False):
        self._text = text
        self._fail_first = fail_first
        self.calls = []
        self.models = self

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self._fail_first and len(self.calls) == 1:
            raise ValueError("Unsupported schema construct")
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
