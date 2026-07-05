"""
Backtester behavior with a deterministic dummy model — verifies fills,
costs, stop-loss, and that live and backtest share the same prediction
contract (models output next-day returns).
"""
import numpy as np
import pandas as pd

from backtest.engine import BacktestEngine
from risk.position_sizer import PositionSizer
from signals.generator import SignalGenerator


class AlwaysBullish:
    """Predicts a constant +1% next-day return."""
    def predict(self, X):
        return np.full(len(X), 0.01)


class AlwaysNeutral:
    """Predicts exactly 0% — should never trade."""
    def predict(self, X):
        return np.zeros(len(X))


def run_engine(ohlcv, model, **engine_kwargs):
    engine = BacktestEngine(commission_per_share=0.01,
                            initial_capital=100_000.0,
                            **engine_kwargs)
    signal_gen = SignalGenerator(threshold=0.003, confidence_threshold=0.60)
    sizer = PositionSizer(risk_per_trade=0.01, atr_multiplier=2.0,
                          max_position_pct=0.15)
    sentiment = pd.DataFrame(columns=['score'])
    return engine.run(ohlcv, sentiment, model, signal_gen, sizer)


def test_bullish_model_buys(ohlcv):
    curve = run_engine(ohlcv, AlwaysBullish())
    assert not curve.empty
    buys = curve[curve['trade_side'] == 'BUY']
    assert len(buys) >= 1
    # First buy consumes cash
    first_buy_date = buys.index[0]
    assert curve.loc[first_buy_date, 'cash'] < 100_000.0


def test_neutral_model_never_trades(ohlcv):
    curve = run_engine(ohlcv, AlwaysNeutral())
    assert not curve.empty
    assert (curve['trade_side'] == '').all()
    assert (curve['portfolio_value'] == 100_000.0).all()


def test_slippage_worsens_fills(ohlcv):
    """Buys with slippage must fill above the raw open price."""
    curve_slip = run_engine(ohlcv, AlwaysBullish(), slippage_bps=50.0)
    buys = curve_slip[curve_slip['trade_side'] == 'BUY']
    assert len(buys) >= 1
    first = buys.iloc[0]
    raw_open = float(ohlcv.loc[buys.index[0], 'Open'])
    assert first['trade_price'] > raw_open


def test_stop_loss_exits_position():
    """A steady downtrend after entry must trigger the stop-loss SELL."""
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2022-01-03", periods=150)
    # Flat for 80 days, then a steady decline
    close = np.concatenate([
        np.full(80, 100.0),
        100.0 * np.cumprod(np.full(70, 0.99)),
    ])
    close = pd.Series(close, index=dates) + rng.normal(0, 0.05, 150)
    df = pd.DataFrame({
        'Open':   close.shift(1).fillna(close.iloc[0]),
        'High':   close * 1.004,
        'Low':    close * 0.996,
        'Close':  close,
        'Volume': np.full(150, 80_000_000, dtype=np.int64),
    })

    curve = run_engine(df, AlwaysBullish(), stop_loss_pct=0.02)
    sells = curve[curve['trade_side'] == 'SELL']
    assert len(sells) >= 1, "stop-loss never fired on a 50% decline"
