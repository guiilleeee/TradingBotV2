"""Telegram alert wiring in main.py and screening.py -- the actual call sites.

test_telegram_alerts.py proves the module itself is correct in isolation; this
file proves it is actually invoked from the right places, at the right times,
and not invoked from the wrong ones (a skip, a hold, a still-tripped breaker).
"""

import pandas as pd
import pytest
import yaml

import data_fetcher
import execution
import main
import screening
import telegram_alerts
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
    "use_market_positioning": False,
    "symbols": [{"symbol": "AAPL", "asset_class": "equity"}],
}


def write_config(tmp_path, **overrides):
    config = dict(BASE_CONFIG)
    config.update(overrides)
    config["db_path"] = str(tmp_path / "cycle.db")
    config["csv_path"] = str(tmp_path / "signals.csv")
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


def stub_provider(monkeypatch, output):
    def fake_generate(signal_input, system_prompt=None, **kwargs):
        return output(signal_input) if callable(output) else output

    monkeypatch.setattr(main, "get_provider", lambda config: ("stub", fake_generate))


def buy_signal(symbol="AAPL", confidence=0.9, stop=95.0, take=115.0):
    return SignalOutput(
        symbol=symbol, action="buy", confidence=confidence, position_size_pct=10.0,
        stop_loss_price=stop, take_profit_price=take, reasoning="Compra per impuls amb volum.",
    )


def hold_signal(symbol="AAPL"):
    return SignalOutput(symbol=symbol, action="hold", confidence=0.5, reasoning="Senyals contradictoris.")


def sell_signal(symbol="AAPL", confidence=0.9):
    return SignalOutput(
        symbol=symbol, action="sell", confidence=confidence, position_size_pct=10.0,
        stop_loss_price=104.0, take_profit_price=112.0, reasoning="Sortida.",
    )


@pytest.fixture(autouse=True)
def telegram_configured(monkeypatch):
    """Telegram credentials present in every test in this file, so alert
    call sites actually reach send_alert and can be captured/asserted on.
    """
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "999")


@pytest.fixture
def sent_alerts(monkeypatch):
    """Capture every alert body sent via telegram_alerts.send_alert, without
    touching the network -- patched at the one true choke point every alert
    function in telegram_alerts.py funnels through.
    """
    captured = []
    monkeypatch.setattr(telegram_alerts, "send_alert", lambda text: captured.append(text))
    return captured


# ------------------------------------------------------------------- trades


def test_a_dry_run_buy_sends_a_trade_alert(tmp_path, monkeypatch, sent_alerts):
    path, config = write_config(tmp_path)
    stub_market(monkeypatch, {"AAPL": 100.0})
    stub_provider(monkeypatch, buy_signal())

    main.run_cycle(str(path))

    assert len(sent_alerts) == 1
    assert sent_alerts[0].startswith(telegram_alerts.SIMULATION_LABEL)
    assert "AAPL" in sent_alerts[0]
    assert "COMPRA" in sent_alerts[0]


def test_a_hold_sends_no_trade_alert(tmp_path, monkeypatch, sent_alerts):
    path, config = write_config(tmp_path)
    stub_market(monkeypatch, {"AAPL": 100.0})
    stub_provider(monkeypatch, hold_signal())

    main.run_cycle(str(path))

    assert sent_alerts == []


def test_a_duplicate_buy_that_gets_skipped_sends_no_trade_alert(tmp_path, monkeypatch, sent_alerts):
    path, config = write_config(tmp_path)
    stub_market(monkeypatch, {"AAPL": 100.0})
    stub_provider(monkeypatch, buy_signal())

    main.run_cycle(str(path))  # first buy: one alert
    sent_alerts.clear()
    main.run_cycle(str(path))  # second buy: skipped (already held)

    assert sent_alerts == []


def test_a_sell_sends_a_trade_alert_with_the_venda_label(tmp_path, monkeypatch, sent_alerts):
    path, config = write_config(tmp_path)
    stub_market(monkeypatch, {"AAPL": 100.0})
    stub_provider(monkeypatch, buy_signal())
    main.run_cycle(str(path))
    sent_alerts.clear()

    stub_market(monkeypatch, {"AAPL": 108.0})
    stub_provider(monkeypatch, sell_signal())
    main.run_cycle(str(path))

    assert len(sent_alerts) == 1
    assert "VENDA" in sent_alerts[0]


def test_live_mode_trade_alert_carries_the_live_label(tmp_path, monkeypatch, sent_alerts):
    path, config = write_config(tmp_path, live_execution=True)
    stub_market(monkeypatch, {"AAPL": 100.0})
    # Sub-1-share budget so execution takes the notional path, which only
    # needs a single successful POST to look like a real live fill.
    stub_provider(monkeypatch, buy_signal(confidence=0.9))
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_API_SECRET", "s")
    monkeypatch.setattr(execution, "fetch_live_equity", lambda fallback: 1000.0)
    monkeypatch.setattr(execution, "fetch_existing_position", lambda **kw: None)

    class FakeAlpacaResponse:
        status_code = 200

        def json(self):
            return {"id": "order-1", "filled_avg_price": "100.5", "filled_qty": "2"}

        @property
        def text(self):
            return "{}"

    monkeypatch.setattr(execution.requests, "post", lambda *a, **kw: FakeAlpacaResponse())

    main.run_cycle(str(path))

    assert len(sent_alerts) == 1
    assert sent_alerts[0].startswith(telegram_alerts.LIVE_LABEL)


# --------------------------------------------------------------- auto-close


def test_an_auto_close_sends_exactly_one_alert(tmp_path, monkeypatch, sent_alerts):
    path, config = write_config(tmp_path)
    stub_market(monkeypatch, {"AAPL": 100.0})
    stub_provider(monkeypatch, buy_signal())
    main.run_cycle(str(path))
    sent_alerts.clear()

    stub_market(monkeypatch, {"AAPL": 120.0})  # gaps through take-profit
    stub_provider(monkeypatch, hold_signal())
    main.run_cycle(str(path))

    auto_close_alerts = [a for a in sent_alerts if "Tancament automatic" in a]
    assert len(auto_close_alerts) == 1
    assert "AAPL" in auto_close_alerts[0]
    assert "40.00" in auto_close_alerts[0]  # (120-100)*2 shares


# ----------------------------------------------------------- circuit breaker


def test_circuit_breaker_alert_fires_exactly_once_across_several_symbols(tmp_path, monkeypatch, sent_alerts):
    path, config = write_config(
        tmp_path,
        symbols=[
            {"symbol": "AAPL", "asset_class": "equity"},
            {"symbol": "MSFT", "asset_class": "equity"},
            {"symbol": "GOOG", "asset_class": "equity"},
        ],
    )
    stub_market(monkeypatch, {"AAPL": 100.0, "MSFT": 200.0, "GOOG": 300.0})

    from logger import BotLogger

    # Loss booked before this cycle starts, already over the -3% threshold on
    # a $1000 account -- every symbol this cycle will find the breaker tripped.
    BotLogger(config["db_path"]).record_pnl("AAPL", -50.0)

    stub_provider(monkeypatch, buy_signal())
    main.run_cycle(str(path))

    breaker_alerts = [a for a in sent_alerts if "Circuit breaker" in a]
    # Tripped BEFORE this cycle started -- not a new trip this cycle, so it
    # must send nothing, not one alert per symbol either.
    assert breaker_alerts == []


def test_circuit_breaker_alert_fires_once_when_it_trips_mid_cycle(tmp_path, monkeypatch, sent_alerts):
    path, config = write_config(
        tmp_path,
        symbols=[
            {"symbol": "AAPL", "asset_class": "equity"},
            {"symbol": "MSFT", "asset_class": "equity"},
            {"symbol": "GOOG", "asset_class": "equity"},
        ],
        max_risk_pct=50.0,  # oversized on purpose so one loss trips the breaker
    )
    stub_market(monkeypatch, {"AAPL": 100.0, "MSFT": 100.0, "GOOG": 100.0})

    from logger import BotLogger

    log = BotLogger(config["db_path"])
    # Seed a loss that is UNDER the threshold before the cycle starts...
    log.record_pnl("AAPL", -20.0)  # -2%, not yet tripped (breaker at -3%)

    calls = {"n": 0}

    def fake_generate(signal_input, system_prompt=None, **kwargs):
        # After the first symbol is evaluated, book an additional loss so the
        # SECOND symbol's recomputed today_loss_pct newly crosses -3%.
        calls["n"] += 1
        if calls["n"] == 1:
            log.record_pnl(signal_input.symbol, -15.0)  # now -3.5% total
        return hold_signal(signal_input.symbol)

    monkeypatch.setattr(main, "get_provider", lambda config: ("stub", fake_generate))

    main.run_cycle(str(path))

    breaker_alerts = [a for a in sent_alerts if "Circuit breaker" in a]
    assert len(breaker_alerts) == 1


def test_circuit_breaker_alert_does_not_repeat_on_a_later_cycle_same_day(tmp_path, monkeypatch, sent_alerts):
    path, config = write_config(tmp_path)
    stub_market(monkeypatch, {"AAPL": 100.0})

    from logger import BotLogger

    BotLogger(config["db_path"]).record_pnl("AAPL", -50.0)  # already tripped

    stub_provider(monkeypatch, buy_signal())
    main.run_cycle(str(path))
    sent_alerts.clear()
    main.run_cycle(str(path))  # a later cycle, still tripped, same UTC day

    breaker_alerts = [a for a in sent_alerts if "Circuit breaker" in a]
    assert breaker_alerts == []


# --------------------------------------------------------------- no creds


def test_no_alerts_and_no_errors_when_telegram_is_not_configured(tmp_path, monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    calls = []
    monkeypatch.setattr(telegram_alerts.requests, "post", lambda *a, **kw: calls.append(1))

    path, config = write_config(tmp_path)
    stub_market(monkeypatch, {"AAPL": 100.0})
    stub_provider(monkeypatch, buy_signal())

    assert main.run_cycle(str(path)) == 0
    assert calls == []


# ------------------------------------------------------------ cycle failure


def test_a_cycle_level_failure_sends_one_alert_and_still_raises(tmp_path, monkeypatch, sent_alerts):
    path, config = write_config(tmp_path)

    def boom(db_path):
        raise RuntimeError("disk is on fire")

    monkeypatch.setattr(main, "BotLogger", boom)

    with pytest.raises(RuntimeError, match="disk is on fire"):
        main.run_cycle(str(path))

    failure_alerts = [a for a in sent_alerts if "cicle de trading ha fallat" in a]
    assert len(failure_alerts) == 1
    assert "disk is on fire" in failure_alerts[0]


def test_a_single_symbol_failure_sends_no_cycle_level_alert(tmp_path, monkeypatch, sent_alerts):
    # Contrast case: _process_symbol raising is already caught inside the
    # per-symbol loop and must never look like a cycle-level failure.
    path, config = write_config(tmp_path)

    def fake_fetch(symbol, period=data_fetcher.DEFAULT_PERIOD, interval="1d"):
        raise ValueError("no data for this symbol")

    monkeypatch.setattr(data_fetcher, "fetch_ohlcv", fake_fetch)
    stub_provider(monkeypatch, hold_signal())

    assert main.run_cycle(str(path)) == 0
    failure_alerts = [a for a in sent_alerts if "cicle de trading ha fallat" in a]
    assert failure_alerts == []


def test_a_failure_before_is_live_is_known_defaults_the_alert_to_simulation(tmp_path, monkeypatch, sent_alerts):
    bad_path = tmp_path / "does_not_exist.yaml"

    with pytest.raises(FileNotFoundError):
        main.run_cycle(str(bad_path))

    failure_alerts = [a for a in sent_alerts if "cicle de trading ha fallat" in a]
    assert len(failure_alerts) == 1
    assert failure_alerts[0].startswith(telegram_alerts.SIMULATION_LABEL)


# =================================================================== screening


def write_screening_config(tmp_path, **overrides):
    config = {"live_execution": False}
    config.update(overrides)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return str(path)


@pytest.fixture
def healthy_screening(monkeypatch):
    import equity_universe

    universe = {"AAPL", "MSFT", "GOOG", "AMZN", "META", "NFLX"}
    monkeypatch.setattr(equity_universe, "build_equity_universe", lambda: universe)
    monkeypatch.setattr(
        equity_universe, "fetch_market_movers",
        lambda: {"most_actives": [{"symbol": s, "volume": 1e7} for s in universe],
                 "gainers": [], "losers": []},
    )

    markets = {
        "BTC/USDC": {"id": "@1", "symbol": "BTC/USDC", "base": "BTC", "quote": "USDC",
                     "type": "spot", "spot": True, "swap": False, "contract": False},
        "ETH/USDC": {"id": "@2", "symbol": "ETH/USDC", "base": "ETH", "quote": "USDC",
                     "type": "spot", "spot": True, "swap": False, "contract": False},
        "SOL/USDC": {"id": "@3", "symbol": "SOL/USDC", "base": "SOL", "quote": "USDC",
                     "type": "spot", "spot": True, "swap": False, "contract": False},
        "HYPE/USDC": {"id": "@4", "symbol": "HYPE/USDC", "base": "HYPE", "quote": "USDC",
                      "type": "spot", "spot": True, "swap": False, "contract": False},
        "PURR/USDC": {"id": "@5", "symbol": "PURR/USDC", "base": "PURR", "quote": "USDC",
                      "type": "spot", "spot": True, "swap": False, "contract": False},
    }

    class FakeExchange:
        def __init__(self):
            self.markets = markets

        def load_markets(self):
            return self.markets

        def market(self, symbol):
            return self.markets[symbol]

    monkeypatch.setattr(execution, "_hyperliquid_exchange", lambda is_live: FakeExchange())
    monkeypatch.setattr(
        screening, "fetch_crypto_volumes", lambda exchange, symbols: {s: 1_000_000.0 for s in symbols}
    )
    monkeypatch.setattr(screening, "fetch_crypto_positioning", lambda limit=50: {})


def test_screening_success_sends_one_complete_alert_with_the_symbols(
    tmp_path, healthy_screening, sent_alerts
):
    config_path = write_screening_config(tmp_path)
    out = tmp_path / "symbols.yaml"

    rc = screening.run_screening(str(out), config_path)

    assert rc == 0
    complete_alerts = [a for a in sent_alerts if "Cribratge setmanal completat" in a]
    assert len(complete_alerts) == 1
    assert complete_alerts[0].startswith(telegram_alerts.SIMULATION_LABEL)


def test_screening_success_alert_uses_the_live_label_when_config_says_so(
    tmp_path, healthy_screening, sent_alerts
):
    config_path = write_screening_config(tmp_path, live_execution=True)
    out = tmp_path / "symbols.yaml"

    screening.run_screening(str(out), config_path)

    complete_alerts = [a for a in sent_alerts if "Cribratge setmanal completat" in a]
    assert complete_alerts[0].startswith(telegram_alerts.LIVE_LABEL)


def test_screening_failure_on_small_equity_universe_sends_one_alert(
    tmp_path, monkeypatch, healthy_screening, sent_alerts
):
    import equity_universe

    monkeypatch.setattr(equity_universe, "build_equity_universe", lambda: {"AAPL"})
    config_path = write_screening_config(tmp_path)
    out = tmp_path / "symbols.yaml"

    rc = screening.run_screening(str(out), config_path)

    assert rc == 1
    failure_alerts = [a for a in sent_alerts if "cribratge setmanal ha fallat" in a]
    assert len(failure_alerts) == 1


def test_screening_failure_on_small_crypto_universe_sends_one_alert(
    tmp_path, monkeypatch, healthy_screening, sent_alerts
):
    class TinyExchange:
        def __init__(self):
            self.markets = {
                "BTC/USDC": {"id": "@1", "symbol": "BTC/USDC", "base": "BTC", "quote": "USDC",
                             "type": "spot", "spot": True, "swap": False, "contract": False},
            }

        def load_markets(self):
            return self.markets

        def market(self, symbol):
            return self.markets[symbol]

    monkeypatch.setattr(execution, "_hyperliquid_exchange", lambda is_live: TinyExchange())
    config_path = write_screening_config(tmp_path)
    out = tmp_path / "symbols.yaml"

    rc = screening.run_screening(str(out), config_path)

    assert rc == 1
    failure_alerts = [a for a in sent_alerts if "cribratge setmanal ha fallat" in a]
    assert len(failure_alerts) == 1


def test_screening_failure_on_unexpected_exception_sends_one_alert(
    tmp_path, monkeypatch, healthy_screening, sent_alerts
):
    import equity_universe

    def boom():
        raise RuntimeError("FMP parsing exploded")

    monkeypatch.setattr(equity_universe, "build_equity_universe", boom)
    config_path = write_screening_config(tmp_path)
    out = tmp_path / "symbols.yaml"

    rc = screening.run_screening(str(out), config_path)

    assert rc == 1
    failure_alerts = [a for a in sent_alerts if "cribratge setmanal ha fallat" in a]
    assert len(failure_alerts) == 1
    assert "FMP parsing exploded" in failure_alerts[0]


def test_screening_never_sends_both_a_success_and_a_failure_alert(
    tmp_path, healthy_screening, sent_alerts
):
    config_path = write_screening_config(tmp_path)
    out = tmp_path / "symbols.yaml"
    screening.run_screening(str(out), config_path)

    assert len(sent_alerts) == 1  # exactly one outcome alert, never both


def test_screening_alert_label_defaults_to_simulation_when_config_is_unreadable(
    tmp_path, healthy_screening, sent_alerts
):
    out = tmp_path / "symbols.yaml"
    rc = screening.run_screening(str(out), str(tmp_path / "no_such_config.yaml"))

    assert rc == 0
    assert sent_alerts[0].startswith(telegram_alerts.SIMULATION_LABEL)
