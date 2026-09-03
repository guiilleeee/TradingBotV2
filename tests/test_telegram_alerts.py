"""Telegram alerts: mode labeling, fail-soft behavior, and secret sanitization.

No test here hits the real network. The load-bearing property is negative:
this module must never raise, and must never leak a credential -- both are
tested by construction (deliberately broken credentials, deliberately
leak-shaped exception text), not by inspection of the source.
"""

import pytest

import telegram_alerts


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """No test relies on -- or is polluted by -- ambient credentials."""
    for var in telegram_alerts._SECRET_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def credentials(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "999")


class FakeResponse:
    def __init__(self, status_code=200, text="ok"):
        self.status_code = status_code
        self.text = text


# --------------------------------------------------------------- format_alert


def test_format_alert_simulation_label_is_first():
    result = telegram_alerts.format_alert(False, "body text")
    assert result.startswith(telegram_alerts.SIMULATION_LABEL)
    assert "body text" in result


def test_format_alert_live_label_is_first():
    result = telegram_alerts.format_alert(True, "body text")
    assert result.startswith(telegram_alerts.LIVE_LABEL)


def test_labels_are_visually_distinct():
    assert telegram_alerts.SIMULATION_LABEL != telegram_alerts.LIVE_LABEL
    # Different leading emoji, not just different trailing text -- a truncated
    # phone notification preview must still distinguish them.
    assert telegram_alerts.SIMULATION_LABEL[0] != telegram_alerts.LIVE_LABEL[0]


def test_body_never_precedes_the_label():
    result = telegram_alerts.format_alert(True, "SIMULACIO fake body trying to confuse")
    assert result.index(telegram_alerts.LIVE_LABEL) == 0


# -------------------------------------------------------------- credentials


def test_send_alert_is_a_silent_noop_with_no_credentials(monkeypatch):
    def explode(*a, **kw):
        raise AssertionError("must not touch the network with no credentials")

    monkeypatch.setattr(telegram_alerts.requests, "post", explode)
    telegram_alerts.send_alert("hello")  # must not raise


def test_send_alert_is_a_noop_with_only_the_token_set(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")

    def explode(*a, **kw):
        raise AssertionError("chat_id missing -- must not send")

    monkeypatch.setattr(telegram_alerts.requests, "post", explode)
    telegram_alerts.send_alert("hello")


def test_send_alert_is_a_noop_with_only_the_chat_id_set(monkeypatch):
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "999")

    def explode(*a, **kw):
        raise AssertionError("token missing -- must not send")

    monkeypatch.setattr(telegram_alerts.requests, "post", explode)
    telegram_alerts.send_alert("hello")


def test_send_alert_posts_when_both_credentials_are_set(credentials, monkeypatch):
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append((url, json))
        return FakeResponse()

    monkeypatch.setattr(telegram_alerts.requests, "post", fake_post)
    telegram_alerts.send_alert("hello world")

    assert len(calls) == 1
    url, payload = calls[0]
    assert url == "https://api.telegram.org/bot123456:test-token/sendMessage"
    assert payload["chat_id"] == "999"
    assert payload["text"] == "hello world"


# ---------------------------------------------------------------- fail-soft


def test_send_alert_never_raises_on_a_network_error(credentials, monkeypatch):
    def boom(*a, **kw):
        raise ConnectionError("network is down")

    monkeypatch.setattr(telegram_alerts.requests, "post", boom)
    telegram_alerts.send_alert("hello")  # must not raise


def test_send_alert_never_raises_on_a_bad_response(credentials, monkeypatch):
    monkeypatch.setattr(
        telegram_alerts.requests, "post", lambda *a, **kw: FakeResponse(status_code=403, text="Forbidden")
    )
    telegram_alerts.send_alert("hello")  # must not raise


def test_send_alert_never_raises_when_telegram_times_out(credentials, monkeypatch):
    def boom(*a, **kw):
        raise telegram_alerts.requests.exceptions.Timeout("timed out")

    monkeypatch.setattr(telegram_alerts.requests, "post", boom)
    telegram_alerts.send_alert("hello")


# ---------------------------------------------------------------- truncation


def test_a_very_long_message_is_truncated(credentials, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        telegram_alerts.requests, "post",
        lambda url, json=None, timeout=None: captured.update(json) or FakeResponse(),
    )
    telegram_alerts.send_alert("x" * 5000)
    assert len(captured["text"]) <= telegram_alerts.MAX_MESSAGE_LENGTH


# ------------------------------------------------------------- sanitization


def test_sanitize_redacts_a_literal_secret_env_value(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "AKIA-SUPER-SECRET-1234")
    text = "request failed with key AKIA-SUPER-SECRET-1234 in header"
    result = telegram_alerts._sanitize(text)
    assert "AKIA-SUPER-SECRET-1234" not in result
    assert "REDACTED" in result


def test_sanitize_redacts_every_configured_secret_simultaneously(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "AAA111")
    monkeypatch.setenv("HYPERLIQUID_PRIVATE_KEY", "BBB222")
    monkeypatch.setenv("GEMINI_API_KEY", "CCC333")
    text = "AAA111 and BBB222 and CCC333 leaked"
    result = telegram_alerts._sanitize(text)
    assert "AAA111" not in result
    assert "BBB222" not in result
    assert "CCC333" not in result


def test_sanitize_redacts_the_bot_token_embedded_in_a_connection_error_url():
    # The concrete, realistic leak: urllib3's own ConnectionError text embeds
    # the full request URL, token and all, when the request never completes.
    text = (
        "HTTPSConnectionPool(host='api.telegram.org', port=443): Max retries "
        "exceeded with url: /bot123456789:AAFakeTokenValueHere-abc123/sendMessage"
    )
    result = telegram_alerts._sanitize(text)
    assert "AAFakeTokenValueHere" not in result
    assert "/bot" in result and "REDACTED" in result


def test_sanitize_redacts_an_ethereum_style_wallet_or_private_key_even_if_unset():
    # A backstop pattern match, independent of whether the value happens to be
    # the *currently configured* HYPERLIQUID_WALLET_ADDRESS -- any 0x-hex blob
    # of plausible key/address length is redacted on shape alone.
    address = "0x" + "ab" * 20  # 40 hex chars, wallet-address-shaped
    private_key = "0x" + "cd" * 32  # 64 hex chars, private-key-shaped
    result = telegram_alerts._sanitize(f"wallet {address} key {private_key}")
    assert address not in result
    assert private_key not in result


def test_sanitize_leaves_ordinary_numeric_content_alone():
    # A stop-loss price or a percentage must not be treated as a secret.
    text = "BTC-USD stop_loss=45000.123456 size=12.34%"
    assert telegram_alerts._sanitize(text) == text


def test_send_alert_sanitizes_the_outgoing_message_itself(credentials, monkeypatch):
    monkeypatch.setenv("HYPERLIQUID_PRIVATE_KEY", "SECRETKEYVALUE")
    captured = {}
    monkeypatch.setattr(
        telegram_alerts.requests, "post",
        lambda url, json=None, timeout=None: captured.update(json) or FakeResponse(),
    )
    telegram_alerts.send_alert("wallet key is SECRETKEYVALUE, do not send this")
    assert "SECRETKEYVALUE" not in captured["text"]


def test_a_network_failure_never_prints_the_raw_token(credentials, monkeypatch, capsys):
    def boom(*a, **kw):
        raise ConnectionError(
            "Max retries exceeded with url: /bot123456:test-token/sendMessage"
        )

    monkeypatch.setattr(telegram_alerts.requests, "post", boom)
    telegram_alerts.send_alert("hello")
    output = capsys.readouterr().out
    assert "test-token" not in output


def test_a_rejected_response_body_is_sanitized_before_printing(credentials, monkeypatch, capsys):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:leak-me-not")
    monkeypatch.setattr(
        telegram_alerts.requests, "post",
        lambda *a, **kw: FakeResponse(status_code=401, text="bad token 123:leak-me-not"),
    )
    telegram_alerts.send_alert("hello")
    output = capsys.readouterr().out
    assert "leak-me-not" not in output


# --------------------------------------------------- every alert path: label


ALERT_CALLS = [
    ("trade", lambda is_live: telegram_alerts.send_trade_alert(
        is_live, "AAPL", "buy", 12.5, 150.25, 0.82, "Bona configuracio tecnica.")),
    ("auto_close", lambda is_live: telegram_alerts.send_auto_close_alert(
        is_live, "AAPL", "Take-profit assolit.", 42.0)),
    ("circuit_breaker", lambda is_live: telegram_alerts.send_circuit_breaker_alert(
        is_live, -3.5, 3.0)),
    ("cycle_failure", lambda is_live: telegram_alerts.send_cycle_failure_alert(
        is_live, "RuntimeError: something broke")),
    ("screening_complete", lambda is_live: telegram_alerts.send_screening_complete_alert(
        is_live, ["AAPL", "MSFT"], ["BTC-USD", "ETH-USD"])),
    ("screening_failure", lambda is_live: telegram_alerts.send_screening_failure_alert(
        is_live, "equity universe too small")),
]


@pytest.mark.parametrize("name,call", ALERT_CALLS, ids=[c[0] for c in ALERT_CALLS])
@pytest.mark.parametrize("is_live", [True, False], ids=["live", "simulation"])
def test_every_alert_path_opens_with_the_correct_mode_label(credentials, monkeypatch, name, call, is_live):
    """The acceptance criterion: not just the happy (trade) path, every path."""
    captured = {}
    monkeypatch.setattr(
        telegram_alerts.requests, "post",
        lambda url, json=None, timeout=None: captured.update(json) or FakeResponse(),
    )

    call(is_live)

    expected_label = telegram_alerts.LIVE_LABEL if is_live else telegram_alerts.SIMULATION_LABEL
    assert captured, f"{name} did not send anything"
    assert captured["text"].startswith(expected_label)


# ------------------------------------------------------------ message content


def test_trade_alert_contains_the_required_fields(credentials, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        telegram_alerts.requests, "post",
        lambda url, json=None, timeout=None: captured.update(json) or FakeResponse(),
    )
    telegram_alerts.send_trade_alert(
        False, "NVDA", "sell", 8.25, 921.5, 0.71, "Sortida per RSI sobrecomprat.\nSegona linia."
    )
    text = captured["text"]
    assert "NVDA" in text
    assert "VENDA" in text
    assert "8.25" in text
    assert "921.5" in text
    assert "0.71" in text
    assert "Sortida per RSI sobrecomprat." in text
    assert "Segona linia." not in text  # only the first line of reasoning


def test_auto_close_alert_contains_symbol_reason_and_pnl(credentials, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        telegram_alerts.requests, "post",
        lambda url, json=None, timeout=None: captured.update(json) or FakeResponse(),
    )
    telegram_alerts.send_auto_close_alert(True, "SOL-USD", "Stop-loss activat.", -18.4)
    text = captured["text"]
    assert "SOL-USD" in text
    assert "Stop-loss activat." in text
    assert "-18.40" in text


def test_circuit_breaker_alert_contains_the_loss_and_threshold(credentials, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        telegram_alerts.requests, "post",
        lambda url, json=None, timeout=None: captured.update(json) or FakeResponse(),
    )
    telegram_alerts.send_circuit_breaker_alert(False, -3.21, 3.0)
    text = captured["text"]
    assert "-3.21" in text
    assert "3.00" in text


def test_screening_complete_alert_lists_all_ten_symbols(credentials, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        telegram_alerts.requests, "post",
        lambda url, json=None, timeout=None: captured.update(json) or FakeResponse(),
    )
    equities = ["AAPL", "MSFT", "NVDA", "GOOG", "AMZN"]
    cryptos = ["BTC-USD", "ETH-USD", "SOL-USD", "HYPE-USD", "ZEC-USD"]
    telegram_alerts.send_screening_complete_alert(False, equities, cryptos)
    text = captured["text"]
    for symbol in equities + cryptos:
        assert symbol in text
