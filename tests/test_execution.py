import json

import pytest

import execution
from models import ExistingPosition, TradeSignal


def signal(action="buy", size=10.0, stop=95.0, take=115.0, symbol="AAPL"):
    return TradeSignal(
        symbol=symbol,
        action=action,
        confidence=0.9,
        position_size_pct=size,
        stop_loss_price=stop,
        take_profit_price=take,
        reasoning="prova",
        raw_action=action,
    )


@pytest.fixture(autouse=True)
def no_credentials(monkeypatch):
    """Simulation must work with no keys present at all."""
    for var in (
        "ALPACA_API_KEY",
        "ALPACA_API_SECRET",
        "HYPERLIQUID_WALLET_ADDRESS",
        "HYPERLIQUID_PRIVATE_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


# ------------------------------------------------------------------- symbols


@pytest.mark.parametrize(
    "internal,hl",
    [("BTC-USD", "BTC/USDC"), ("ETH-USD", "ETH/USDC"),
     ("SOL-USDT", "SOL/USDC"), ("btc-usd", "BTC/USDC")],
)
def test_hyperliquid_symbol_mapping(internal, hl):
    # Hyperliquid spot quotes against USDC, not OKX's old USDT.
    assert execution.to_hyperliquid_symbol(internal) == hl


def test_hyperliquid_base_currency():
    assert execution.hyperliquid_base_currency("BTC-USD") == "BTC"


@pytest.mark.parametrize("equity_symbol", ["AAPL", "MSFT", "NVDA", "AMZN"])
def test_an_equity_symbol_can_never_be_mapped_to_hyperliquid(equity_symbol):
    # AAPL/USDC and MSFT/USDC are REAL, live Hyperliquid spot markets (tokenized
    # equities). A bare equity ticker reaching the mapper would resolve to a valid
    # spot market and pass every spot check, so the suffix requirement is the only
    # thing standing between a config slip and an equity order on a crypto venue.
    with pytest.raises(execution.NotSpotMarketError, match="crypto suffix"):
        execution.to_hyperliquid_symbol(equity_symbol)


# ------------------------------------------------------- stop-limit direction


def test_stop_limit_sits_below_the_stop_when_closing_a_long():
    # The bracket leg that closes a long is a SELL stop, so the limit must be
    # below the stop or it can never fill.
    assert execution._stop_limit_price(100.0, "sell") == pytest.approx(99.0)


def test_stop_limit_sits_above_the_stop_for_a_buy_stop():
    # The mirror case. Hardcoding x0.99 for both sides is how this goes wrong.
    assert execution._stop_limit_price(100.0, "buy") == pytest.approx(101.0)


def test_unknown_exit_side_is_rejected():
    with pytest.raises(ValueError):
        execution._stop_limit_price(100.0, "sideways")


# ------------------------------------------------------------------- guards


def test_buy_is_skipped_when_a_position_already_exists():
    result = execution.execute_trade(
        signal("buy"), "equity", 100.0, 1000.0, is_live=False,
        existing_position=ExistingPosition(qty=3.0, avg_entry_price=90.0),
    )
    assert result.status == "skipped"
    assert "already exists" in result.message


def test_sell_with_nothing_held_is_skipped_not_shorted():
    result = execution.execute_trade(
        signal("sell"), "equity", 100.0, 1000.0, is_live=False, existing_position=None
    )
    assert result.status == "skipped"
    assert "nothing to sell" in result.message
    assert "short" in result.message


def test_hold_never_reaches_a_venue():
    result = execution.execute_trade(signal("hold"), "equity", 100.0, 1000.0, is_live=False)
    assert result.status == "skipped"


# ------------------------------------------------------- credentials gating


def test_simulation_needs_no_credentials():
    result = execution.execute_trade(signal("buy"), "equity", 100.0, 10000.0, is_live=False)
    assert result.status == "dry_run"
    assert result.qty is not None


def test_live_without_alpaca_credentials_errors():
    result = execution.execute_trade(signal("buy"), "equity", 100.0, 10000.0, is_live=True)
    assert result.status == "error"
    assert "ALPACA_API_KEY" in result.message


def test_live_without_hyperliquid_credentials_errors():
    result = execution.execute_trade(
        signal("buy", symbol="BTC-USD"), "crypto", 50000.0, 10000.0, is_live=True
    )
    assert result.status == "error"
    assert "HYPERLIQUID_WALLET_ADDRESS" in result.message
    assert "HYPERLIQUID_PRIVATE_KEY" in result.message


# ------------------------------------------------------------------- sizing


def test_opening_size_comes_from_equity_and_percent():
    # 10% of $10,000 at $100 = 10 whole shares.
    result = execution.execute_trade(signal("buy", size=10.0), "equity", 100.0, 10000.0, is_live=False)
    assert result.status == "dry_run"
    assert result.qty == pytest.approx(10.0)


def test_closing_uses_the_held_quantity_not_a_recomputed_one():
    # A freshly computed size here would be 10 shares; the position is 3.
    held = ExistingPosition(qty=3.0, avg_entry_price=90.0)
    result = execution.execute_trade(
        signal("sell", size=10.0), "equity", 100.0, 10000.0, is_live=False, existing_position=held
    )
    assert result.qty == pytest.approx(3.0)
    assert result.realized_pnl_usd == pytest.approx((100.0 - 90.0) * 3.0)


def test_closing_a_fractional_position_sells_the_exact_fraction():
    held = ExistingPosition(qty=0.4237, avg_entry_price=200.0)
    result = execution.execute_trade(
        signal("sell"), "equity", 210.0, 10000.0, is_live=False, existing_position=held
    )
    assert result.qty == pytest.approx(0.4237)


def test_small_account_buys_notional_instead_of_skipping():
    # $50 budget on a $300 share is 0 whole shares. Whole-share-only sizing
    # silently skipped every equity trade on a small account in the old build.
    result = execution.execute_trade(signal("buy", size=5.0), "equity", 300.0, 1000.0, is_live=False)
    assert result.status == "dry_run"
    assert result.qty == pytest.approx(50.0 / 300.0)
    assert "notional" in result.message


def test_notional_entry_is_flagged_for_a_bot_managed_exit():
    result = execution.execute_trade(signal("buy", size=5.0), "equity", 300.0, 1000.0, is_live=False)
    assert execution.needs_managed_exit(result) is True


def test_whole_share_entry_is_not_flagged_for_a_managed_exit():
    result = execution.execute_trade(signal("buy", size=10.0), "equity", 100.0, 10000.0, is_live=False)
    assert execution.needs_managed_exit(result) is False
    assert "bracket" in result.message


def test_budget_below_the_alpaca_minimum_is_a_clean_skip():
    result = execution.execute_trade(signal("buy", size=0.05), "equity", 300.0, 1000.0, is_live=False)
    assert result.status == "skipped"
    assert "minimum notional" in result.message


# ------------------------------------------------- live Alpaca request shape


class FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None

    @property
    def text(self):
        return json.dumps(self._payload)


@pytest.fixture
def capture_alpaca(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_API_SECRET", "s")
    sent = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        sent["url"] = url
        sent["body"] = json
        return FakeResponse({"id": "order-1", "filled_avg_price": "100.5", "filled_qty": "10"})

    monkeypatch.setattr(execution.requests, "post", fake_post)
    return sent


def test_live_whole_share_buy_attaches_a_correctly_directed_bracket(capture_alpaca):
    result = execution.execute_trade(
        signal("buy", size=10.0, stop=95.0, take=115.0), "equity", 100.0, 10000.0, is_live=True
    )
    body = capture_alpaca["body"]

    assert result.status == "success"
    assert result.order_id == "order-1"
    assert body["order_class"] == "bracket"
    assert body["qty"] == "10"
    assert body["stop_loss"]["stop_price"] == pytest.approx(95.0)
    # Sell stop -> limit below the stop.
    assert body["stop_loss"]["limit_price"] == pytest.approx(94.05)
    assert body["stop_loss"]["limit_price"] < body["stop_loss"]["stop_price"]
    assert body["take_profit"]["limit_price"] == pytest.approx(115.0)


def test_live_closing_sell_attaches_no_bracket(capture_alpaca):
    held = ExistingPosition(qty=4.0, avg_entry_price=90.0)
    execution.execute_trade(
        signal("sell"), "equity", 100.0, 10000.0, is_live=True, existing_position=held
    )
    body = capture_alpaca["body"]

    assert body["side"] == "sell"
    assert body["qty"] == "4"
    assert "order_class" not in body
    assert "stop_loss" not in body
    assert "take_profit" not in body


def test_live_notional_buy_sends_notional_and_day_tif(capture_alpaca):
    execution.execute_trade(signal("buy", size=5.0), "equity", 300.0, 1000.0, is_live=True)
    body = capture_alpaca["body"]

    # Alpaca supports notional only on market/limit/stop/stop_limit with TIF=day,
    # and will not attach bracket legs to one.
    assert body["notional"] == "50.00"
    assert "qty" not in body
    assert body["time_in_force"] == "day"
    assert "order_class" not in body


def test_alpaca_rejection_becomes_an_error_result(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_API_SECRET", "s")

    class Rejected:
        status_code = 422
        text = "insufficient buying power"

        def json(self):
            return {}

    monkeypatch.setattr(execution.requests, "post", lambda *a, **kw: Rejected())
    result = execution.execute_trade(signal("buy", size=10.0), "equity", 100.0, 10000.0, is_live=True)
    assert result.status == "error"
    assert "insufficient buying power" in result.message


def test_an_exploding_venue_returns_an_error_not_a_crash(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_API_SECRET", "s")

    def boom(*a, **kw):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(execution.requests, "post", boom)
    result = execution.execute_trade(signal("buy", size=10.0), "equity", 100.0, 10000.0, is_live=True)
    assert result.status == "error"
    assert "connection reset" in result.message


# ------------------------------------------------------------------- crypto


def spot_market(symbol="BTC/USDC", **overrides):
    """A market dict shaped like a real ccxt Hyperliquid SPOT market."""
    market = {
        "id": "@142", "symbol": symbol, "base": symbol.split("/")[0], "quote": "USDC",
        "type": "spot", "spot": True, "swap": False, "contract": False, "active": True,
        "precision": {"amount": 1e-05, "price": 0.001},
        "limits": {"amount": {"min": None}, "cost": {"min": 10.0}},
    }
    market.update(overrides)
    return market


def perp_market(symbol="BTC/USDC:USDC"):
    """A market dict shaped like a real ccxt Hyperliquid PERPETUAL market."""
    return {
        "id": "BTC", "symbol": symbol, "base": "BTC", "quote": "USDC", "settle": "USDC",
        "type": "swap", "spot": False, "swap": True, "contract": True, "active": True,
        "precision": {"amount": 1e-05, "price": 0.1},
        "limits": {"amount": {"min": None}, "cost": {"min": 10.0}},
    }


class FakeHyperliquid:
    def __init__(self, market=None, order=None, balances=None):
        self._market = market if market is not None else spot_market()
        self._order = order or {"id": "hl-1", "average": 50000.0, "filled": 0.002}
        self._balances = balances or {"total": {}}
        self.orders = []
        self.balance_params = []

    def load_markets(self):
        return {self._market["symbol"]: self._market}

    def market(self, symbol):
        return self._market

    def amount_to_precision(self, symbol, amount):
        return "%.8f" % float(amount)

    def create_order(self, symbol, type_, side, amount, price=None, params=None):
        self.orders.append(
            {"symbol": symbol, "type": type_, "side": side,
             "amount": amount, "price": price, "params": params}
        )
        return self._order

    def fetch_balance(self, params=None):
        self.balance_params.append(params or {})
        return self._balances


# ------------------------------------------------- THE SPOT / NO-LEVERAGE GUARD


def test_resolve_spot_market_accepts_a_real_spot_market():
    assert execution.resolve_spot_market(FakeHyperliquid(), "BTC/USDC")["spot"] is True


def test_resolve_spot_market_rejects_a_perpetual():
    with pytest.raises(execution.NotSpotMarketError) as exc:
        execution.resolve_spot_market(FakeHyperliquid(market=perp_market()), "BTC/USDC:USDC")
    message = str(exc.value)
    assert "not a confirmed unleveraged spot market" in message
    assert "type='swap'" in message


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ({"type": "swap"}, "type='swap'"),
        ({"spot": False}, "spot=False"),
        ({"spot": None}, "spot=None"),
        ({"swap": True}, "swap=True"),
        ({"contract": True}, "contract=True"),
        ({"symbol": "BTC/USDC:USDC"}, "settle suffix"),
        ({"quote": "USDT"}, "quote='USDT'"),
        ({"quote": "USDH"}, "quote='USDH'"),
    ],
)
def test_every_spot_condition_is_checked_independently(mutation, expected):
    # Each flag is load-bearing on its own: flipping any single one must reject the
    # market. Checking only `type` would let a mislabelled perp through.
    exchange = FakeHyperliquid(market=spot_market(**mutation))
    with pytest.raises(execution.NotSpotMarketError) as exc:
        execution.resolve_spot_market(exchange, "BTC/USDC")
    assert expected in str(exc.value)


def test_no_order_is_ever_placed_on_a_perpetual_market(monkeypatch):
    """The acceptance criterion, tested by observation rather than by omission."""
    monkeypatch.setenv("HYPERLIQUID_WALLET_ADDRESS", "0xabc")
    monkeypatch.setenv("HYPERLIQUID_PRIVATE_KEY", "0xdef")
    exchange = FakeHyperliquid(market=perp_market())
    monkeypatch.setattr(execution, "_hyperliquid_exchange", lambda is_live: exchange)

    result = execution.execute_trade(
        signal("buy", size=10.0, symbol="BTC-USD", stop=45000.0, take=60000.0),
        "crypto", 50000.0, 10000.0, is_live=True,
    )

    assert result.status == "error"
    assert "SPOT GUARD" in result.message
    assert exchange.orders == []  # nothing was submitted


def test_the_spot_guard_also_applies_in_simulation(monkeypatch):
    # A dry_run that skipped the leverage check would be a dry_run of a system we
    # are not running.
    exchange = FakeHyperliquid(market=perp_market())
    monkeypatch.setattr(execution, "_hyperliquid_exchange", lambda is_live: exchange)

    result = execution.execute_trade(
        signal("buy", size=10.0, symbol="BTC-USD"), "crypto", 50000.0, 10000.0, is_live=False
    )
    assert result.status == "error"
    assert "SPOT GUARD" in result.message
    assert exchange.orders == []


def test_a_live_spot_order_carries_no_leverage_or_margin_params(monkeypatch):
    monkeypatch.setenv("HYPERLIQUID_WALLET_ADDRESS", "0xabc")
    monkeypatch.setenv("HYPERLIQUID_PRIVATE_KEY", "0xdef")
    exchange = FakeHyperliquid()
    monkeypatch.setattr(execution, "_hyperliquid_exchange", lambda is_live: exchange)

    execution.execute_trade(
        signal("buy", size=10.0, symbol="BTC-USD"), "crypto", 50000.0, 10000.0, is_live=True
    )

    assert len(exchange.orders) == 1
    order = exchange.orders[0]
    assert order["symbol"] == "BTC/USDC"
    assert ":" not in order["symbol"]
    assert order["type"] == "market"
    # No reduceOnly, no marginMode, no leverage -- nothing implying a leveraged leg.
    assert not order["params"]


# --------------------------------------------------------------- crypto sizing


def test_simulated_crypto_buy_is_a_dry_run_flagged_for_managed_exit(monkeypatch):
    monkeypatch.setattr(execution, "_hyperliquid_exchange", lambda is_live: FakeHyperliquid())
    result = execution.execute_trade(
        signal("buy", size=10.0, symbol="BTC-USD", stop=45000.0, take=60000.0),
        "crypto", 50000.0, 10000.0, is_live=False,
    )
    assert result.status == "dry_run"
    assert result.qty == pytest.approx(1000.0 / 50000.0)
    # Hyperliquid spot has no reliable bracket, so every crypto entry is bot-managed.
    assert execution.needs_managed_exit(result) is True
    assert "BTC/USDC spot" in result.message


def test_simulated_crypto_close_uses_the_held_quantity(monkeypatch):
    monkeypatch.setattr(execution, "_hyperliquid_exchange", lambda is_live: FakeHyperliquid())
    held = ExistingPosition(qty=0.02, avg_entry_price=40000.0)
    result = execution.execute_trade(
        signal("sell", symbol="BTC-USD"), "crypto", 50000.0, 10000.0,
        is_live=False, existing_position=held,
    )
    assert result.qty == pytest.approx(0.02)
    assert result.realized_pnl_usd == pytest.approx((50000.0 - 40000.0) * 0.02)
    assert execution.needs_managed_exit(result) is False


def test_crypto_below_the_ten_dollar_minimum_notional_is_a_clean_skip(monkeypatch):
    # Hyperliquid's real constraint is a $10 minimum NOTIONAL (limits.cost.min),
    # uniform across all 303 spot markets. limits.amount.min is None on every one
    # of them, so a size-based check would be a silent no-op.
    monkeypatch.setattr(execution, "_hyperliquid_exchange", lambda is_live: FakeHyperliquid())
    result = execution.execute_trade(
        signal("buy", size=5.0, symbol="BTC-USD"), "crypto", 50000.0, 100.0, is_live=False
    )
    assert result.status == "skipped"
    assert "under the exchange $10.00 minimum" in result.message
    assert "$5.00" in result.message


def test_crypto_just_above_the_minimum_notional_is_allowed(monkeypatch):
    monkeypatch.setattr(execution, "_hyperliquid_exchange", lambda is_live: FakeHyperliquid())
    result = execution.execute_trade(
        signal("buy", size=11.0, symbol="BTC-USD"), "crypto", 50000.0, 100.0, is_live=False
    )
    assert result.status == "dry_run"


def test_crypto_precision_comes_from_the_exchange(monkeypatch):
    class Rounding(FakeHyperliquid):
        def amount_to_precision(self, symbol, amount):
            return "%.4f" % round(float(amount), 4)

    monkeypatch.setattr(execution, "_hyperliquid_exchange", lambda is_live: Rounding())
    result = execution.execute_trade(
        signal("buy", size=10.0, symbol="BTC-USD"), "crypto", 50000.0, 10000.0, is_live=False
    )
    assert result.qty == pytest.approx(0.02)


def test_a_market_lookup_failure_is_an_error_not_a_blind_order(monkeypatch):
    class Broken(FakeHyperliquid):
        def load_markets(self):
            raise RuntimeError("hyperliquid unreachable")

    monkeypatch.setattr(execution, "_hyperliquid_exchange", lambda is_live: Broken())
    result = execution.execute_trade(
        signal("buy", size=10.0, symbol="BTC-USD"), "crypto", 50000.0, 10000.0, is_live=False
    )
    assert result.status == "error"
    assert "market lookup failed" in result.message


# ---------------------------------------------------------------- positions


class FakeLogger:
    def __init__(self, position=None, last_buy=None):
        self._position = position
        self._last_buy = last_buy
        self.touched = False

    def get_simulated_position(self, symbol):
        self.touched = True
        return self._position

    def get_last_buy_price(self, symbol):
        return self._last_buy


def test_simulation_reads_the_ledger_and_never_a_broker(monkeypatch):
    def explode(*a, **kw):
        raise AssertionError("simulation must not call a broker")

    monkeypatch.setattr(execution.requests, "get", explode)
    monkeypatch.setattr(execution, "_hyperliquid_exchange", explode)

    held = ExistingPosition(qty=2.0, avg_entry_price=100.0)
    fake = FakeLogger(position=held)
    assert execution.fetch_existing_position("AAPL", "equity", False, fake) is held
    assert execution.fetch_existing_position("BTC-USD", "crypto", False, fake) is held
    assert fake.touched is True


def test_live_equity_position_comes_from_alpaca(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_API_SECRET", "s")
    monkeypatch.setattr(
        execution.requests,
        "get",
        lambda *a, **kw: FakeResponse({"qty": "3", "avg_entry_price": "97.25"}),
    )
    position = execution.fetch_existing_position("AAPL", "equity", True, FakeLogger())
    assert position.qty == 3.0
    # Alpaca returns the cost basis natively, so nothing is reconstructed.
    assert position.avg_entry_price == 97.25


def test_live_missing_alpaca_position_is_none(monkeypatch):
    class NotFound:
        status_code = 404

        def raise_for_status(self):
            raise AssertionError("404 must be read as flat, not raised on")

    monkeypatch.setattr(execution.requests, "get", lambda *a, **kw: NotFound())
    assert execution.fetch_existing_position("AAPL", "equity", True, FakeLogger()) is None


def test_a_broker_outage_raises_instead_of_reporting_flat(monkeypatch):
    # Returning None on a failed lookup would read as "no position held", which
    # the duplicate-buy guard treats as permission to open one. The symbol has to
    # fail loudly so the cycle skips it instead of doubling up.
    def boom(*a, **kw):
        raise RuntimeError("alpaca unreachable")

    monkeypatch.setattr(execution.requests, "get", boom)
    with pytest.raises(RuntimeError, match="alpaca unreachable"):
        execution.fetch_existing_position("AAPL", "equity", True, FakeLogger())


def test_live_crypto_cost_basis_comes_from_the_last_logged_buy(monkeypatch):
    exchange = FakeHyperliquid(balances={"total": {"BTC": 0.05}})
    monkeypatch.setattr(execution, "_hyperliquid_exchange", lambda is_live: exchange)
    position = execution.fetch_existing_position(
        "BTC-USD", "crypto", True, FakeLogger(last_buy=41000.0)
    )
    assert position.qty == 0.05
    # A spot balance carries no entry price, so it comes from our own log.
    assert position.avg_entry_price == 41000.0


def test_balance_lookup_explicitly_asks_for_the_spot_wallet(monkeypatch):
    # ccxt's fetch_balance for Hyperliquid defaults to the PERPETUALS account
    # ("defaults to swap" per its own docstring). Omitting type=spot would read
    # the wrong account entirely and report positions we do not hold on spot.
    monkeypatch.setenv("HYPERLIQUID_WALLET_ADDRESS", "0xabc")
    monkeypatch.setenv("HYPERLIQUID_PRIVATE_KEY", "0xdef")
    exchange = FakeHyperliquid(balances={"total": {"BTC": 0.05}})
    monkeypatch.setattr(execution, "_hyperliquid_exchange", lambda is_live: exchange)

    execution.fetch_existing_position("BTC-USD", "crypto", True, FakeLogger(last_buy=41000.0))

    assert exchange.balance_params[0]["type"] == "spot"
    assert exchange.balance_params[0]["user"] == "0xabc"


def test_live_crypto_without_a_known_cost_basis_refuses_to_guess(monkeypatch):
    # Holding coins with no recorded entry price means P&L cannot be computed.
    # Trading the symbol blind is worse than skipping it.
    monkeypatch.setattr(
        execution, "_hyperliquid_exchange",
        lambda is_live: FakeHyperliquid(balances={"total": {"BTC": 0.05}}),
    )
    with pytest.raises(RuntimeError, match="cost basis is unknown"):
        execution.fetch_existing_position("BTC-USD", "crypto", True, FakeLogger())


def test_live_crypto_with_no_balance_is_flat(monkeypatch):
    monkeypatch.setattr(
        execution, "_hyperliquid_exchange",
        lambda is_live: FakeHyperliquid(balances={"total": {"BTC": 0.0}}),
    )
    assert execution.fetch_existing_position("BTC-USD", "crypto", True, FakeLogger()) is None


def test_live_equity_falls_back_when_no_venue_answers(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_API_SECRET", "s")

    def boom(*a, **kw):
        raise RuntimeError("down")

    monkeypatch.setattr(execution.requests, "get", boom)
    assert execution.fetch_live_equity(1234.0) == 1234.0


def test_live_close_books_pnl_from_the_actual_fill(monkeypatch):
    # The circuit breaker acts on this number, so it must reflect what filled,
    # not the pre-trade price the decision was made at.
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_API_SECRET", "s")
    monkeypatch.setattr(
        execution.requests,
        "post",
        lambda *a, **kw: FakeResponse({"id": "o", "filled_avg_price": "97.0", "filled_qty": "4"}),
    )

    held = ExistingPosition(qty=4.0, avg_entry_price=100.0)
    result = execution.execute_trade(
        signal("sell"), "equity", 105.0, 10000.0, is_live=True, existing_position=held
    )

    assert result.fill_price == pytest.approx(97.0)
    # (97 - 100) * 4 = -12, not the (105 - 100) * 4 = +20 the pre-trade price implies.
    assert result.realized_pnl_usd == pytest.approx(-12.0)


def test_live_crypto_close_books_pnl_from_the_actual_fill(monkeypatch):
    monkeypatch.setenv("HYPERLIQUID_WALLET_ADDRESS", "0xabc")
    monkeypatch.setenv("HYPERLIQUID_PRIVATE_KEY", "0xdef")

    monkeypatch.setattr(
        execution, "_hyperliquid_exchange",
        lambda is_live: FakeHyperliquid(order={"id": "hl-1", "average": 48000.0, "filled": 0.02}),
    )
    held = ExistingPosition(qty=0.02, avg_entry_price=50000.0)
    result = execution.execute_trade(
        signal("sell", symbol="BTC-USD"), "crypto", 52000.0, 10000.0,
        is_live=True, existing_position=held,
    )

    assert result.fill_price == pytest.approx(48000.0)
    assert result.realized_pnl_usd == pytest.approx((48000.0 - 50000.0) * 0.02)


def test_a_perpetual_only_symbol_is_a_clean_skip_with_a_legible_reason(monkeypatch):
    # BNB and XRP are perpetual-only on Hyperliquid. Being spot-only, we cannot
    # trade them there -- that is a venue limitation, not a malfunction.
    import ccxt

    class NoSpot(FakeHyperliquid):
        def market(self, symbol):
            raise ccxt.BadSymbol(f"hyperliquid does not have market symbol {symbol}")

    monkeypatch.setattr(execution, "_hyperliquid_exchange", lambda is_live: NoSpot())
    result = execution.execute_trade(
        signal("buy", size=10.0, symbol="BNB-USD"), "crypto", 600.0, 10000.0, is_live=False
    )
    assert result.status == "skipped"
    assert "no BNB/USDC spot market on Hyperliquid" in result.message
    assert "spot-only" in result.message
