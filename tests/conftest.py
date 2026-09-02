import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from logger import BotLogger  # noqa: E402  (needs the sys.path line above)

REAL_DB = ROOT / "trading_bot.db"


@pytest.fixture
def tmp_logger(tmp_path):
    """A BotLogger on a throwaway SQLite file.

    The assertion is not decoration: a test that reached the real trading_bot.db
    would rewrite live position state.
    """
    db_path = tmp_path / "test_trading_bot.db"
    assert db_path.resolve() != REAL_DB.resolve()
    assert str(tmp_path) in str(db_path)
    return BotLogger(str(db_path))


@pytest.fixture
def read_signals(tmp_logger):
    """Return every signals row as a dict with the JSON blobs already decoded."""

    def _read():
        conn = sqlite3.connect(tmp_logger.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute("SELECT * FROM signals ORDER BY id").fetchall()
        finally:
            conn.close()

        out = []
        for row in rows:
            item = dict(row)
            for key in ("signal_input", "raw_output", "final_signal", "execution_result"):
                item[key] = json.loads(item[key]) if item[key] else None
            out.append(item)
        return out

    return _read
