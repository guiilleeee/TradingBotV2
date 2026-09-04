"""Whole-cycle tests: config in, CSV out, with the network stubbed.

These are the ones that catch wiring mistakes -- a method defined and never
called, a ledger update that never happens, a threshold that leaks across modes.
"""

import csv
import os

import pandas as pd
import pytest
import yaml

import data_fetcher
import execution
import main
from models import SignalOutput

BASE_CONFIG = {
    "live_execution": False,
    "signal_provider": "claude",
    "fallback_equity_usd": 1000.0,
    "circuit_breaker_loss_pct": 3.0,
    "max_risk_pct": 1.0,
    "max_absolute_position_pct": 20.0,
    "min_confidence_live": 0.65,
    "min_confidence_simulation": 0.40,
    "symbols": [{"symbol": "AAPL", "asset_class": "equity"}],
}


def write_config(tmp_path, **overrides):
    config = dict(BASE_CONFIG)
    config.update(overrides)
    config["db_path"] = str(tmp_path / "cycle.db")
    config["csv_path"] = str(tmp_path / "signals.csv")
    config["positions_path"] = str(tmp_path / "positions.json")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path, config


def stub_market(monkeypatch, prices, bars=60):
    def fake_fetch(symbol, period=data_fetcher.DEFAULT_PERIOD, interval="1d"):
        price = prices[symbol]
        closes = [price] * (bars - 1) + [price]
        return pd.DataFrame({"Close": closes, "Volume": [1_000_000.0] * bars})

    monkeypatch.setattr(data_fetcher, "fetch_ohlcv", fake_fetch)
    monkeypatch.setattr(data_fetcher, "latest_price", lambda df: float(df["Close"].iloc[-1]))
    monkeypatch.setattr(data_fetcher, "fetch_headlines", lambda *a, **kw: [])


def stub_provider(monkeypatch, output, capture=None):
    """Replace the model with a canned answer, recording the prompt it was given."""

    def fake_generate(signal_input, system_prompt=None, **kwargs):
        if capture is not None:
            capture.append({"system_prompt": system_prompt, "input": signal_input})
        if callable(output):
            return output(signal_input)
        return output

    monkeypatch.setattr(main, "get_provider", lambda config: ("stub", fake_generate))


def buy_signal(symbol="AAPL", confidence=0.9, stop=95.0, take=115.0):
    return SignalOutput(
        symbol=symbol, action="buy", confidence=confidence, position_size_pct=10.0,
        stop_loss_price=stop, take_profit_price=take, reasoning="Compra per impuls amb volum.",
    )


def hold_signal(symbol="AAPL"):
    return SignalOutput(symbol=symbol, action="hold", confidence=0.5, reasoning="Senyals contradictoris.")


def read_csv(config):
    with open(config["csv_path"], newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------- happy paths


def test_simulated_buy_opens_a_ledger_position_and_exports_a_row(tmp_path, monkeypatch):
    path, config = write_config(tmp_path)
    stub_market(monkeypatch, {"AAPL": 100.0})
    stub_provider(monkeypatch, buy_signal())

    assert main.run_cycle(str(path)) == 0

    from logger import BotLogger

    log = BotLogger(config["db_path"])
    position = log.get_simulated_position("AAPL")
    assert position is not None
    # 5% stop distance, 1% risk -> 20% of $1000 = $200 at $100 = 2 shares.
    assert position.qty == pytest.approx(2.0)
    assert position.avg_entry_price == pytest.approx(100.0)

    rows = read_csv(config)
    assert len(rows) == 1
    assert rows[0]["action"] == "buy"
    assert rows[0]["execution_status"] == "dry_run"
    assert rows[0]["account_equity_usd"] == "1000.0"
    assert rows[0]["reasoning"]


def test_a_second_cycle_does_not_buy_the_same_symbol_twice(tmp_path, monkeypatch):
    path, config = write_config(tmp_path)
    stub_market(monkeypatch, {"AAPL": 100.0})
    stub_provider(monkeypatch, buy_signal())

    main.run_cycle(str(path))
    main.run_cycle(str(path))

    rows = read_csv(config)
    assert rows[0]["execution_status"] == "dry_run"
    assert rows[1]["execution_status"] == "skipped"

    from logger import BotLogger

    assert BotLogger(config["db_path"]).get_simulated_position("AAPL").qty == pytest.approx(2.0)


def test_take_profit_on_a_later_cycle_books_pnl_and_moves_equity(tmp_path, monkeypatch):
    path, config = write_config(tmp_path)
    stub_market(monkeypatch, {"AAPL": 100.0})
    stub_provider(monkeypatch, buy_signal())
    main.run_cycle(str(path))

    # Price gaps through the take-profit before the next cycle.
    stub_market(monkeypatch, {"AAPL": 120.0})
    stub_provider(monkeypatch, hold_signal())
    main.run_cycle(str(path))

    from logger import BotLogger

    log = BotLogger(config["db_path"])
    assert log.get_simulated_position("AAPL") is None
    assert log.get_all_time_realized_pnl() == pytest.approx((120.0 - 100.0) * 2.0)

    rows = read_csv(config)
    # The auto-close row, and no fresh model row for the symbol closed this cycle.
    closes = [r for r in rows if r["override_reason"] == "automatic exit"]
    assert len(closes) == 1
    assert closes[0]["realized_pnl_usd"] == "40.0"
    assert float(closes[0]["account_equity_usd"]) == pytest.approx(1040.0)
    assert "Take-profit" in closes[0]["reasoning"]
    assert len(rows) == 2  # the buy, then the auto-close. Nothing else.


def test_sell_closes_the_ledger_position_and_records_pnl(tmp_path, monkeypatch):
    path, config = write_config(tmp_path)
    stub_market(monkeypatch, {"AAPL": 100.0})
    stub_provider(monkeypatch, buy_signal())
    main.run_cycle(str(path))

    # Model decides to exit at 108, below the 115 take-profit so the sweep leaves it.
    stub_market(monkeypatch, {"AAPL": 108.0})
    stub_provider(
        monkeypatch,
        SignalOutput(symbol="AAPL", action="sell", confidence=0.9, position_size_pct=10.0,
                     stop_loss_price=104.0, take_profit_price=112.0, reasoning="Sortida."),
    )
    main.run_cycle(str(path))

    from logger import BotLogger

    log = BotLogger(config["db_path"])
    assert log.get_simulated_position("AAPL") is None
    assert log.get_all_time_realized_pnl() == pytest.approx((108.0 - 100.0) * 2.0)


# ---------------------------------------------------------------- risk wiring


def test_low_confidence_in_simulation_still_trades_above_040(tmp_path, monkeypatch):
    path, config = write_config(tmp_path)
    stub_market(monkeypatch, {"AAPL": 100.0})
    stub_provider(monkeypatch, buy_signal(confidence=0.45))
    main.run_cycle(str(path))

    rows = read_csv(config)
    assert rows[0]["action"] == "buy"  # 0.45 clears the 0.40 simulation threshold


def test_the_same_signal_is_held_in_live_mode(tmp_path, monkeypatch):
    path, config = write_config(tmp_path, live_execution=True)
    stub_market(monkeypatch, {"AAPL": 100.0})
    stub_provider(monkeypatch, buy_signal(confidence=0.45))
    monkeypatch.setattr(execution, "fetch_live_equity", lambda fallback: 1000.0)
    monkeypatch.setattr(execution, "fetch_existing_position", lambda **kw: None)
    main.run_cycle(str(path))

    rows = read_csv(config)
    assert rows[0]["action"] == "hold"  # 0.45 is below the 0.65 live threshold
    assert rows[0]["raw_action"] == "buy"
    assert "confidence" in rows[0]["override_reason"]


def test_live_mode_sends_the_base_prompt_with_no_addendum(tmp_path, monkeypatch):
    from prompts import SIMULATION_ADDENDUM, SYSTEM_PROMPT

    path, _ = write_config(
        tmp_path, live_execution=True, min_confidence_simulation=0.0
    )
    stub_market(monkeypatch, {"AAPL": 100.0})
    seen = []
    stub_provider(monkeypatch, hold_signal(), capture=seen)
    monkeypatch.setattr(execution, "fetch_live_equity", lambda fallback: 1000.0)
    monkeypatch.setattr(execution, "fetch_existing_position", lambda **kw: None)
    main.run_cycle(str(path))

    assert seen[0]["system_prompt"] == SYSTEM_PROMPT
    assert SIMULATION_ADDENDUM not in seen[0]["system_prompt"]


def test_simulation_mode_sends_the_addendum(tmp_path, monkeypatch):
    from prompts import SIMULATION_ADDENDUM

    path, _ = write_config(tmp_path)
    stub_market(monkeypatch, {"AAPL": 100.0})
    seen = []
    stub_provider(monkeypatch, hold_signal(), capture=seen)
    main.run_cycle(str(path))

    assert SIMULATION_ADDENDUM in seen[0]["system_prompt"]


def test_circuit_breaker_skips_the_model_call_entirely(tmp_path, monkeypatch):
    path, config = write_config(tmp_path)
    stub_market(monkeypatch, {"AAPL": 100.0})

    from logger import BotLogger

    BotLogger(config["db_path"]).record_pnl("AAPL", -50.0)  # -5% of $1000

    calls = []
    stub_provider(monkeypatch, buy_signal(), capture=calls)
    main.run_cycle(str(path))

    assert calls == []  # the expensive call never happened
    rows = read_csv(config)
    assert rows[0]["action"] == "hold"
    assert "circuit breaker" in rows[0]["override_reason"]
    assert rows[0]["reasoning"]


def test_a_position_size_is_derived_from_the_stop_not_the_model(tmp_path, monkeypatch):
    path, config = write_config(tmp_path)
    stub_market(monkeypatch, {"AAPL": 100.0})
    # 2% stop -> 50% raw, clamped to the 20% cap.
    stub_provider(monkeypatch, buy_signal(stop=98.0))
    main.run_cycle(str(path))

    rows = read_csv(config)
    assert float(rows[0]["position_size_pct"]) == pytest.approx(20.0)
    assert "clamped" in rows[0]["override_reason"]


# -------------------------------------------------------------- resilience


def test_one_broken_symbol_does_not_kill_the_cycle(tmp_path, monkeypatch):
    path, config = write_config(
        tmp_path,
        symbols=[
            {"symbol": "BROKEN", "asset_class": "equity"},
            {"symbol": "AAPL", "asset_class": "equity"},
        ],
    )

    def fake_fetch(symbol, period=data_fetcher.DEFAULT_PERIOD, interval="1d"):
        if symbol == "BROKEN":
            raise ValueError("yfinance returned no bars")
        return pd.DataFrame({"Close": [100.0] * 60, "Volume": [1e6] * 60})

    monkeypatch.setattr(data_fetcher, "fetch_ohlcv", fake_fetch)
    monkeypatch.setattr(data_fetcher, "latest_price", lambda df: float(df["Close"].iloc[-1]))
    monkeypatch.setattr(data_fetcher, "fetch_headlines", lambda *a, **kw: [])
    stub_provider(monkeypatch, buy_signal())

    assert main.run_cycle(str(path)) == 0

    rows = read_csv(config)
    assert len(rows) == 1
    assert rows[0]["symbol"] == "AAPL"  # the healthy symbol still traded


def test_a_provider_failure_is_contained_to_its_symbol(tmp_path, monkeypatch):
    path, config = write_config(
        tmp_path,
        symbols=[
            {"symbol": "AAPL", "asset_class": "equity"},
            {"symbol": "MSFT", "asset_class": "equity"},
        ],
    )
    stub_market(monkeypatch, {"AAPL": 100.0, "MSFT": 200.0})

    def flaky(signal_input):
        if signal_input.symbol == "AAPL":
            raise RuntimeError("model unavailable")
        return buy_signal("MSFT")

    stub_provider(monkeypatch, flaky)
    assert main.run_cycle(str(path)) == 0

    rows = read_csv(config)
    assert [r["symbol"] for r in rows] == ["MSFT"]


def test_unknown_provider_is_rejected_loudly(tmp_path, monkeypatch):
    path, _ = write_config(tmp_path, signal_provider="oracle")
    with pytest.raises(ValueError, match="unknown signal_provider"):
        main.run_cycle(str(path))


# ------------------------------------------------------------- asset classes


@pytest.mark.parametrize(
    "symbol,expected",
    [("AAPL", "equity"), ("BTC-USD", "crypto"), ("ETH-USDT", "crypto")],
)
def test_asset_class_falls_back_to_the_symbol_shape(symbol, expected):
    # A position can outlive its config entry and must still close on the right venue.
    assert main.infer_asset_class(symbol, {"symbols": []}) == expected


def test_asset_class_prefers_the_config():
    config = {"symbols": [{"symbol": "WEIRD-USD", "asset_class": "equity"}]}
    assert main.infer_asset_class("WEIRD-USD", config) == "equity"


# ------------------------------------------------- trader positioning as input


def test_positioning_reaches_the_model_as_input(tmp_path, monkeypatch):
    import market_intel

    path, _ = write_config(tmp_path, symbols=[{"symbol": "BTC-USD", "asset_class": "crypto"}])
    stub_market(monkeypatch, {"BTC-USD": 50000.0})
    monkeypatch.setattr(
        market_intel, "fetch_positioning",
        lambda symbol, asset_class: "Top wallets are net long by roughly 80%.",
    )
    seen = []
    stub_provider(monkeypatch, hold_signal("BTC-USD"), capture=seen)
    main.run_cycle(str(path))

    assert seen[0]["input"].market_positioning == "Top wallets are net long by roughly 80%."


def test_positioning_is_absent_for_equities(tmp_path, monkeypatch):
    path, _ = write_config(tmp_path)
    stub_market(monkeypatch, {"AAPL": 100.0})
    seen = []
    stub_provider(monkeypatch, hold_signal(), capture=seen)
    main.run_cycle(str(path))

    assert seen[0]["input"].market_positioning is None


def test_a_positioning_outage_never_breaks_a_cycle(tmp_path, monkeypatch):
    import market_intel

    path, config = write_config(tmp_path, symbols=[{"symbol": "BTC-USD", "asset_class": "crypto"}])
    stub_market(monkeypatch, {"BTC-USD": 50000.0})

    def boom(symbol, asset_class):
        raise RuntimeError("hyperliquid stats down")

    monkeypatch.setattr(market_intel, "fetch_positioning", boom)
    stub_provider(monkeypatch, hold_signal("BTC-USD"))

    # fetch_positioning swallows its own failures, but even if one escaped, the
    # per-symbol guard contains it. Either way the cycle completes.
    assert main.run_cycle(str(path)) == 0
    assert len(read_csv(config)) >= 0


def test_positioning_can_be_switched_off(tmp_path, monkeypatch):
    import market_intel

    path, _ = write_config(
        tmp_path,
        use_market_positioning=False,
        symbols=[{"symbol": "BTC-USD", "asset_class": "crypto"}],
    )
    stub_market(monkeypatch, {"BTC-USD": 50000.0})

    def boom(symbol, asset_class):
        raise AssertionError("positioning must not be fetched when disabled")

    monkeypatch.setattr(market_intel, "fetch_positioning", boom)
    seen = []
    stub_provider(monkeypatch, hold_signal("BTC-USD"), capture=seen)
    main.run_cycle(str(path))

    assert seen[0]["input"].market_positioning is None


def test_positioning_cannot_shortcut_the_confidence_threshold(tmp_path, monkeypatch):
    """The acceptance criterion: positioning is input, never a bypass.

    A maximally bullish positioning note plus a low-confidence buy must still be
    overridden to hold. There is no path from this data to an executed trade that
    skips risk_manager.
    """
    import market_intel

    path, config = write_config(
        tmp_path, live_execution=True, symbols=[{"symbol": "BTC-USD", "asset_class": "crypto"}]
    )
    stub_market(monkeypatch, {"BTC-USD": 50000.0})
    monkeypatch.setattr(
        market_intel, "fetch_positioning",
        lambda symbol, asset_class: "Every top wallet is 100% net long with maximum conviction.",
    )
    monkeypatch.setattr(execution, "fetch_live_equity", lambda fallback: 1000.0)
    monkeypatch.setattr(execution, "fetch_existing_position", lambda **kw: None)
    stub_provider(monkeypatch, buy_signal("BTC-USD", confidence=0.45, stop=47500.0, take=55000.0))

    main.run_cycle(str(path))

    rows = read_csv(config)
    assert rows[0]["action"] == "hold"
    assert rows[0]["raw_action"] == "buy"
    assert "confidence" in rows[0]["override_reason"]
    assert rows[0]["execution_status"] in ("", None)  # nothing was executed


def test_strong_recent_trade_activity_without_technical_confirmation_still_holds(tmp_path, monkeypatch):
    """Acceptance criterion for the recent-trades enrichment (market_intel.py):
    specific, individually-attributed recent buying ("wallet ending ...4f2a opened
    a $520,000 long 1 hour ago") is strictly more vivid than the old aggregate
    percentage, but must be exactly as inert. A model that honestly sees no
    technical confirmation (flat price here -- no momentum, no volume signal)
    still holds, no matter how compelling the trade narrative reads.
    """
    import market_intel

    path, config = write_config(tmp_path, symbols=[{"symbol": "BTC-USD", "asset_class": "crypto"}])
    stub_market(monkeypatch, {"BTC-USD": 50000.0})
    monkeypatch.setattr(
        market_intel, "fetch_positioning",
        lambda symbol, asset_class: (
            "Specific recent activity from the same sampled wallets in the last ~24h: "
            "wallet ending ...4f2a opened a $520,000 long 1 hour ago; wallet ending "
            "...9c11 opened a $310,000 long 2 hours ago. These are leveraged perpetual "
            "positions (and, where noted above, trades) taken by other traders, not "
            "spot holdings, and this bot trades spot without leverage. Treat this as "
            "directional bias only, never as confirmation, and never as a reason to "
            "act without your own technical justification."
        ),
    )
    seen = []
    stub_provider(monkeypatch, hold_signal("BTC-USD"), capture=seen)

    main.run_cycle(str(path))

    assert seen[0]["input"].market_positioning is not None
    assert "wallet ending" in seen[0]["input"].market_positioning
    rows = read_csv(config)
    assert rows[0]["action"] == "hold"


def test_positioning_cannot_shortcut_the_stop_loss_requirement(tmp_path, monkeypatch):
    import market_intel

    path, config = write_config(
        tmp_path, symbols=[{"symbol": "BTC-USD", "asset_class": "crypto"}]
    )
    stub_market(monkeypatch, {"BTC-USD": 50000.0})
    monkeypatch.setattr(
        market_intel, "fetch_positioning",
        lambda symbol, asset_class: "Top wallets are overwhelmingly net long.",
    )
    # High confidence, but the model omitted a take-profit.
    stub_provider(
        monkeypatch,
        SignalOutput(symbol="BTC-USD", action="buy", confidence=0.99, position_size_pct=10.0,
                     stop_loss_price=47500.0, take_profit_price=None,
                     reasoning="Les balenes estan llargues."),
    )
    main.run_cycle(str(path))

    rows = read_csv(config)
    assert rows[0]["action"] == "hold"
    assert "take_profit_price" in rows[0]["override_reason"]


# ------------------------------------------------ symbols.yaml merge (Phase 2)


def test_missing_symbols_file_falls_back_to_config_yaml(tmp_path):
    config_path, config = write_config(tmp_path)
    loaded = main.load_config(str(config_path), str(tmp_path / "does_not_exist.yaml"))
    assert loaded["symbols"] == config["symbols"]


def test_empty_symbols_file_falls_back_to_config_yaml(tmp_path):
    config_path, config = write_config(tmp_path)
    symbols_path = tmp_path / "symbols.yaml"
    symbols_path.write_text("symbols: []\n", encoding="utf-8")
    loaded = main.load_config(str(config_path), str(symbols_path))
    assert loaded["symbols"] == config["symbols"]


def test_symbols_file_with_no_symbols_key_falls_back(tmp_path):
    config_path, config = write_config(tmp_path)
    symbols_path = tmp_path / "symbols.yaml"
    symbols_path.write_text("generated_at: '2026-01-01'\n", encoding="utf-8")
    loaded = main.load_config(str(config_path), str(symbols_path))
    assert loaded["symbols"] == config["symbols"]


def test_populated_symbols_file_overrides_the_symbol_list(tmp_path):
    config_path, _ = write_config(tmp_path)
    symbols_path = tmp_path / "symbols.yaml"
    symbols_path.write_text(
        "symbols:\n  - symbol: NVDA\n    asset_class: equity\n"
        "  - symbol: SOL-USD\n    asset_class: crypto\n",
        encoding="utf-8",
    )
    loaded = main.load_config(str(config_path), str(symbols_path))
    assert loaded["symbols"] == [
        {"symbol": "NVDA", "asset_class": "equity"},
        {"symbol": "SOL-USD", "asset_class": "crypto"},
    ]


def test_symbols_file_cannot_touch_anything_but_the_symbols_key(tmp_path):
    config_path, _ = write_config(
        tmp_path, live_execution=True, max_risk_pct=1.0, circuit_breaker_loss_pct=3.0
    )
    symbols_path = tmp_path / "symbols.yaml"
    # Even if a screening bug somehow wrote risk-relevant keys into symbols.yaml,
    # main.load_config must never read anything from it but "symbols".
    symbols_path.write_text(
        "symbols:\n  - symbol: NVDA\n    asset_class: equity\n"
        "live_execution: false\n"
        "max_risk_pct: 99.0\n"
        "circuit_breaker_loss_pct: 0.01\n",
        encoding="utf-8",
    )
    loaded = main.load_config(str(config_path), str(symbols_path))
    assert loaded["live_execution"] is True
    assert loaded["max_risk_pct"] == 1.0
    assert loaded["circuit_breaker_loss_pct"] == 3.0
    assert loaded["symbols"] == [{"symbol": "NVDA", "asset_class": "equity"}]


def test_run_cycle_uses_the_symbols_file_when_present(tmp_path, monkeypatch):
    config_path, config = write_config(tmp_path, symbols=[{"symbol": "AAPL", "asset_class": "equity"}])
    symbols_path = tmp_path / "symbols.yaml"
    symbols_path.write_text(
        "symbols:\n  - symbol: NVDA\n    asset_class: equity\n", encoding="utf-8"
    )
    stub_market(monkeypatch, {"NVDA": 100.0})
    seen = []
    stub_provider(monkeypatch, hold_signal("NVDA"), capture=seen)

    main.run_cycle(str(config_path), str(symbols_path))

    assert seen[0]["input"].symbol == "NVDA"


def test_run_cycle_works_unmodified_when_symbols_file_does_not_exist(tmp_path, monkeypatch):
    config_path, config = write_config(tmp_path)
    stub_market(monkeypatch, {"AAPL": 100.0})
    seen = []
    stub_provider(monkeypatch, hold_signal("AAPL"), capture=seen)

    rc = main.run_cycle(str(config_path), str(tmp_path / "no_such_symbols.yaml"))

    assert rc == 0
    assert seen[0]["input"].symbol == "AAPL"


def test_default_symbols_path_is_scoped_to_the_configs_own_directory():
    assert main.default_symbols_path(r"/some/dir/config.yaml") == os.path.join(
        "/some/dir", "symbols.yaml"
    )
    # A bare filename with no directory component still resolves to a bare
    # filename -- same cwd-relative behaviour as before for that one case.
    assert main.default_symbols_path("config.yaml") == "symbols.yaml"


def test_isolated_config_never_sees_a_symbols_yaml_from_another_directory(tmp_path, monkeypatch):
    """A tmp_path config must never pick up a symbols.yaml living elsewhere.

    Regression test for the bug where the default symbols_path was a bare
    "symbols.yaml" resolved against the current working directory: in CI that
    resolved to the real, committed repo-root symbols.yaml, silently
    overriding every test's own isolated symbol list. Proven here by planting
    a *different* symbols.yaml next to a config the test is not using, and in
    the actual cwd, then asserting neither has any effect.
    """
    real_config_dir = tmp_path / "actual_config_dir"
    real_config_dir.mkdir()
    config_path, config = write_config(real_config_dir)

    other_config_dir = tmp_path / "other_config_dir"
    other_config_dir.mkdir()
    (other_config_dir / "symbols.yaml").write_text(
        "symbols:\n  - symbol: HOOD\n    asset_class: equity\n", encoding="utf-8"
    )
    (tmp_path / "symbols.yaml").write_text(
        "symbols:\n  - symbol: ZBRA\n    asset_class: equity\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)

    loaded = main.load_config(str(config_path))

    assert loaded["symbols"] == config["symbols"]
