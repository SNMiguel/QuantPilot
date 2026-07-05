"""Trade narrator — templated fallback (no LLM) is correct and never raises."""
from monitoring.narrator import TradeNarrator


def _narrator():
    n = TradeNarrator()
    n._client = None      # force templated fallback deterministically
    n.enabled = False
    return n


def test_empty_context():
    assert _narrator().narrate(100_000.0, {}) == "No tickers were evaluated today."


def test_all_hold_message():
    ctx = {"AAPL": {"signal": "HOLD", "predicted_return": 0.0,
                    "confidence": 0.3, "sentiment": {}, "order": False}}
    out = _narrator().narrate(100_000.0, ctx)
    assert "held" in out.lower()


def test_buy_mentioned_in_template():
    ctx = {
        "AAPL": {"signal": "BUY", "predicted_return": 0.01, "confidence": 0.7,
                 "sentiment": {"direction": "bullish"}, "order": True,
                 "blocked": False},
        "MSFT": {"signal": "HOLD", "predicted_return": 0.0, "confidence": 0.2,
                 "sentiment": {}, "order": False},
    }
    out = _narrator().narrate(100_000.0, ctx)
    assert "AAPL" in out and "BUY" in out
    assert "MSFT" not in out   # holds are omitted from the template


def test_blocked_trade_is_explained():
    ctx = {"GOOGL": {"signal": "SELL", "predicted_return": -0.01,
                     "confidence": 0.9, "sentiment": {"direction": "bearish"},
                     "order": False, "blocked": True}}
    out = _narrator().narrate(100_000.0, ctx)
    assert "block" in out.lower()
