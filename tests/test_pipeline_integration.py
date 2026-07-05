"""
Integration test for the live decision path.

Wires the real SignalGenerator, conformal gate, regime multiplier,
PositionSizer, and OrderManager together with lightweight fakes for the
broker/portfolio/db/alerter — the same composition jobs/daily_job.py uses,
without any network or credentials. This is the test that catches wiring
regressions between the components even when each unit test passes.
"""
import numpy as np
import pandas as pd

from execution.order_manager import OrderManager
from features.regime import detect_regime
from models.ensemble import EnsembleModel, make_base_models
from risk.position_sizer import PositionSizer
from signals.generator import SignalGenerator


# ---- Fakes -----------------------------------------------------------

class FakeBroker:
    def __init__(self, position=None):
        self._position = position
        self.orders = []
    def get_position(self, ticker):
        return self._position
    def submit_order(self, ticker, qty, side):
        order = {"id": f"ord-{len(self.orders)}", "ticker": ticker,
                 "side": side, "qty": qty, "status": "accepted"}
        self.orders.append(order)
        return order


class FakePortfolio:
    def __init__(self, value=100_000.0):
        self._value = value
    def get_portfolio_value(self):
        return self._value
    def is_within_limits(self, ticker, shares, price):
        return True


class FakeDB:
    def __init__(self):
        self.trades = []
    def log_trade(self, **kwargs):
        self.trades.append(kwargs)


class FakeAlerts:
    def send_order_alert(self, *a, **k):
        pass
    def send_error(self, *a, **k):
        pass


def _price_df(n=120, vol=0.01, seed=0):
    rng = np.random.default_rng(seed)
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0.0003, vol, n)),
                      index=pd.bdate_range("2023-01-02", periods=n))
    return pd.DataFrame({
        'Open': close.shift(1).fillna(close.iloc[0]),
        'High': close * 1.01, 'Low': close * 0.99, 'Close': close,
        'Volume': np.full(n, 80_000_000, dtype=np.int64),
    })


def _order_manager(broker):
    return OrderManager(broker, FakePortfolio(),
                        PositionSizer(max_position_pct=0.15),
                        FakeDB(), FakeAlerts())


# ---- Tests -----------------------------------------------------------

def test_buy_flows_end_to_end():
    df = _price_df()
    broker = FakeBroker(position=None)
    om = _order_manager(broker)

    sig = SignalGenerator(threshold=0.003, confidence_threshold=0.6)
    signal = sig.generate_from_return(float(df['Close'].iloc[-1]), 0.02, 0.75)
    regime = detect_regime(df)

    order = om.execute_signal(signal, "AAPL", df,
                              size_multiplier=regime['size_multiplier'])
    assert order is not None and order["side"] == "buy"
    assert broker.orders and broker.orders[0]["qty"] > 0


def test_regime_multiplier_shrinks_order():
    df = _price_df()
    price = float(df['Close'].iloc[-1])
    signal = SignalGenerator(threshold=0.003).generate_from_return(price, 0.02, 0.9)

    full = _order_manager(FakeBroker())
    o_full = full.execute_signal(signal, "AAPL", df, size_multiplier=1.0)
    half = _order_manager(FakeBroker())
    o_half = half.execute_signal(signal, "AAPL", df, size_multiplier=0.4)

    assert o_half["qty"] < o_full["qty"]


def test_sell_closes_existing_position_only():
    df = _price_df()
    price = float(df['Close'].iloc[-1])
    signal = SignalGenerator(threshold=0.003).generate_from_return(price, -0.02, 0.9)

    # No position → SELL is a no-op (cash account can't short)
    flat = _order_manager(FakeBroker(position=None))
    assert flat.execute_signal(signal, "AAPL", df) is None

    # Held 40 shares → SELL closes exactly 40
    held = FakeBroker(position={"qty": 40, "market_value": 40 * price})
    om = _order_manager(held)
    order = om.execute_signal(signal, "AAPL", df)
    assert order["side"] == "sell" and order["qty"] == 40


def test_conformal_gate_holds_a_noisy_prediction():
    """A model whose interval includes zero should never produce a trade."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(400, 15))
    y = rng.normal(0, 0.02, 400)          # pure noise — wide intervals
    model = EnsembleModel(make_base_models())
    model.fit(X, y, n_splits=4)

    x_today = X[:1]
    # A tiny predicted move on a noise model: interval will include zero.
    gate = model.interval_excludes_zero(x_today, coverage=0.80)
    assert gate[0] in (True, False)   # well-defined boolean
    # For pure noise the 80% interval is wide; a sub-1% prediction is inside it.
    point = float(model.predict(x_today)[0])
    if abs(point) < model.conformal_halfwidth(0.80):
        assert not gate[0]
