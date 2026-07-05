"""Signal generation from predicted returns and prices."""
from signals.generator import SignalGenerator


def make_gen():
    return SignalGenerator(threshold=0.01, confidence_threshold=0.60)


def test_buy_on_confident_upmove():
    sig = make_gen().generate_from_return(150.0, 0.02, 0.75)
    assert sig['signal'] == 'BUY'
    assert abs(sig['predicted'] - 153.0) < 1e-6


def test_sell_on_confident_downmove():
    sig = make_gen().generate_from_return(150.0, -0.02, 0.75)
    assert sig['signal'] == 'SELL'


def test_hold_below_threshold():
    sig = make_gen().generate_from_return(150.0, 0.005, 0.90)
    assert sig['signal'] == 'HOLD'


def test_hold_on_low_confidence():
    sig = make_gen().generate_from_return(150.0, 0.02, 0.50)
    assert sig['signal'] == 'HOLD'


def test_hold_on_invalid_price():
    sig = make_gen().generate(0.0, 10.0, 0.9)
    assert sig['signal'] == 'HOLD'


def test_delta_pct_reported_as_percent():
    sig = make_gen().generate_from_return(200.0, 0.015, 0.9)
    assert abs(sig['delta_pct'] - 1.5) < 1e-6
