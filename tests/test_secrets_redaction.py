"""secrets_redaction.py: the single source of truth telegram_alerts.py,
execution.py, and equity_universe.py all import `sanitize` from.

telegram_alerts.py's own tests (test_telegram_alerts.py) already cover the
literal-value scrub and the bot-token/hex backstops end to end through
telegram_alerts._sanitize (a re-export of this module's `sanitize`) -- this
file focuses on what's new here: the FMP query-string pattern, and confirming
the module's public names are what execution.py/equity_universe.py actually
import.
"""

import secrets_redaction


def test_sanitize_redacts_a_literal_secret_env_value(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "sk-fmp-real-secret-value")
    text = "request failed: sk-fmp-real-secret-value was rejected"
    result = secrets_redaction.sanitize(text)
    assert "sk-fmp-real-secret-value" not in result
    assert "REDACTED" in result


def test_sanitize_redacts_fmp_apikey_in_a_url_even_when_unset_in_this_process():
    # Pattern-based backstop: catches the shape even when the value was never
    # in *this* process's environment (a value already rotated, or -- as in
    # equity_universe.py's own use -- an exception constructed directly in a
    # test rather than from a real live request).
    text = (
        "500 Server Error for url: "
        "https://financialmodelingprep.com/stable/sp-500?apikey=totally-unset-value-xyz"
    )
    result = secrets_redaction.sanitize(text)
    assert "totally-unset-value-xyz" not in result
    assert "apikey=***REDACTED***" in result


def test_sanitize_redacts_fmp_apikey_when_not_the_first_query_param():
    text = "https://financialmodelingprep.com/stable/sp-500?limit=10&apikey=secretvalue123&extra=1"
    result = secrets_redaction.sanitize(text)
    assert "secretvalue123" not in result
    assert "limit=10" in result  # only the key itself is scrubbed, not the rest of the query
    assert "extra=1" in result


def test_sanitize_leaves_ordinary_text_alone():
    text = "insufficient buying power for AAPL at 100.00 (attempt 2/3)"
    assert secrets_redaction.sanitize(text) == text


def test_sanitize_handles_empty_and_none_like_input():
    assert secrets_redaction.sanitize("") == ""


def test_execution_and_equity_universe_import_the_same_sanitize_function():
    import equity_universe
    import execution

    assert execution.sanitize is secrets_redaction.sanitize
    assert equity_universe.sanitize is secrets_redaction.sanitize


def test_telegram_alerts_reexports_the_same_module_as_before():
    import telegram_alerts

    assert telegram_alerts._sanitize is secrets_redaction.sanitize
    assert telegram_alerts._SECRET_ENV_VARS is secrets_redaction.SECRET_ENV_VARS
