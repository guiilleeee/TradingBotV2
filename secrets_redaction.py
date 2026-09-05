"""Single source of truth for stripping secrets out of text before it reaches
anywhere it could be seen or persisted -- a Telegram message, a print that lands
in GitHub Actions logs, or a field written into `trading_bot.db` (a file this
project commits to a now-public repository).

Originally lived only in telegram_alerts.py, scoped to the Telegram send path.
Pulled out here because that scope was the gap: nothing sanitized what
execution.py commits into the database, and nothing sanitized what any module
prints to stdout. Both now import `sanitize` from here instead of each keeping
(or lacking) their own copy -- a project-wide secret catalogue only needs
updating in one place.
"""

from __future__ import annotations

import os
import re

# Every secret this project's modules read from the environment, redacted by
# literal value wherever it appears. Deliberately an exact-value match against
# the live environment rather than a guessed pattern -- precise (no false
# positives on ordinary numbers) and exhaustive for what this project actually
# holds.
SECRET_ENV_VARS = (
    "TELEGRAM_BOT_TOKEN",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_WORKSPACE_ID",
    "GEMINI_API_KEY",
    "ALPACA_API_KEY",
    "ALPACA_API_SECRET",
    "HYPERLIQUID_WALLET_ADDRESS",
    "HYPERLIQUID_PRIVATE_KEY",
    "FMP_API_KEY",
)

# Backstop patterns for shapes a secret is likely to leak in even when the
# literal-value scrub above doesn't catch it (the value was never in *this*
# process's environment at all -- a test constructing the exception directly,
# or a value already rotated):
#   - a Telegram bot token embedded in a request URL inside a connection-error
#     message (urllib3's own exception text routinely includes the full
#     request URL, token and all -- the concrete case that motivated this
#     module in the first place);
#   - an Ethereum-style 0x hex string (Hyperliquid wallet addresses are 40 hex
#     chars, private keys are 64 -- one length-agnostic pattern catches both);
#   - FMP's API key, which travels as a query-string parameter
#     (equity_universe.py's `_get`) unlike every other credential here, which
#     travels via header or signature -- a raw HTTPError's default string form
#     embeds it directly in the URL.
_BOT_TOKEN_IN_URL_RE = re.compile(r"/bot\d+:[A-Za-z0-9_-]+")
_HEX_SECRET_RE = re.compile(r"0x[0-9a-fA-F]{40,}")
_FMP_APIKEY_IN_URL_RE = re.compile(r"([?&]apikey=)[^&\s]+", re.IGNORECASE)

REDACTED = "***REDACTED***"


def sanitize(text: str) -> str:
    """Strip anything that looks like a secret out of `text`.

    Two layers: an exact-value scrub against every secret this project's
    modules actually read from the environment, and pattern-based backstops
    for the shapes above. Safe to call on anything, unconditionally -- a
    misconfigured request could echo a credential back in its own error text,
    so this runs on every candidate string rather than only when something
    looks suspicious.
    """
    if not text:
        return text
    for var in SECRET_ENV_VARS:
        value = os.environ.get(var)
        if value:
            text = text.replace(value, REDACTED)
    text = _BOT_TOKEN_IN_URL_RE.sub(f"/bot{REDACTED}", text)
    text = _HEX_SECRET_RE.sub(f"0x{REDACTED}", text)
    text = _FMP_APIKEY_IN_URL_RE.sub(rf"\1{REDACTED}", text)
    return text
