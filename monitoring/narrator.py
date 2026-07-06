"""
Trade narrator - turns the day's raw decision inputs into a short,
plain-English explanation of what the system did and why.

A cron job that posts "AAPL BUY, MSFT HOLD, GOOGL SELL" tells you nothing
about the reasoning. This asks Claude to narrate the actual numbers the
pipeline produced - predicted return, confidence, the sentiment verdict
and its key events, the risk gate outcome - into a few sentences a human
can read over morning coffee. It is strictly grounded: the model is given
only the computed values and told not to invent market commentary, so it
explains the decisions rather than editorializing about the market.

Fallback: with no ANTHROPIC_API_KEY (or no `anthropic` package), it emits
a compact templated summary instead of calling the API - the daily job
never depends on the narrator succeeding.
"""
import os


_SYSTEM = (
    "You explain an automated trading system's daily decisions to its "
    "operator. You are given, per ticker, the model's predicted next-day "
    "return, a confidence score, the generated signal, the news-sentiment "
    "verdict with its key events, and whether any risk gate blocked the "
    "trade. Write 2-4 sentences total across all tickers. Explain WHY each "
    "non-HOLD decision was made and note anything unusual (a signal fired "
    "against negative sentiment, a trade blocked by the exposure cap, low "
    "confidence). Use ONLY the numbers provided; do not invent prices, "
    "news, or market context. Be concise and factual. No hype, no emojis, "
    "no em-dashes; write plain sentences."
)


class TradeNarrator:
    """Generates a natural-language daily rationale, Claude-backed."""

    def __init__(self, model: str = None, anthropic_api_key: str = None):
        if model is None:
            try:
                import config
                model = getattr(config, "LLM_MODEL", "claude-opus-4-8")
            except Exception:
                model = "claude-opus-4-8"
        self.model = model
        self._client = self._build_client(anthropic_api_key)
        self.enabled = self._client is not None

    @staticmethod
    def _build_client(anthropic_api_key: str):
        key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            return None
        try:
            import anthropic
            return anthropic.Anthropic(api_key=key)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Narrate
    # ------------------------------------------------------------------

    def narrate(self, portfolio_value: float, context: dict) -> str:
        """
        Args:
            portfolio_value: Current equity.
            context: {ticker: {
                        'signal': str, 'predicted_return': float,
                        'confidence': float, 'sentiment': dict,
                        'blocked': bool|None, 'order': bool}}

        Returns:
            A short plain-English paragraph. Never raises.
        """
        if not context:
            return "No tickers were evaluated today."

        if not self.enabled:
            return self._template(portfolio_value, context)

        try:
            return self._llm_narrate(portfolio_value, context)
        except Exception as exc:
            print(f"  Note: narrator failed ({exc}) - templated summary.")
            return self._template(portfolio_value, context)

    def _llm_narrate(self, portfolio_value: float, context: dict) -> str:
        lines = [f"Portfolio equity: ${portfolio_value:,.2f}", ""]
        for ticker, c in context.items():
            s = c.get("sentiment", {}) or {}
            events = ", ".join(s.get("key_events", [])[:3]) or "none"
            lines.append(
                f"{ticker}: signal={c.get('signal')}, "
                f"predicted_return={c.get('predicted_return', 0.0):+.3%}, "
                f"confidence={c.get('confidence', 0.0):.2f}, "
                f"sentiment={s.get('score', 0.0):+.2f} ({s.get('direction', 'n/a')}), "
                f"key_events=[{events}], "
                f"volatility_regime={c.get('regime', 'normal')}, "
                f"risk_blocked={bool(c.get('blocked'))}, "
                f"order_placed={bool(c.get('order'))}"
            )
        user = "\n".join(lines)

        response = self._client.messages.create(
            model=self.model,
            max_tokens=512,
            system=_SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        return next(b.text for b in response.content if b.type == "text").strip()

    @staticmethod
    def _template(portfolio_value: float, context: dict) -> str:
        parts = []
        for ticker, c in context.items():
            sig = c.get("signal", "HOLD")
            if sig == "HOLD":
                continue
            s = c.get("sentiment", {}) or {}
            reason = (f"predicted {c.get('predicted_return', 0.0):+.2%} "
                      f"at {c.get('confidence', 0.0):.0%} confidence, "
                      f"news {s.get('direction', 'neutral')}")
            if c.get("blocked"):
                parts.append(f"{ticker} {sig} signal was blocked by the risk limit ({reason}).")
            elif c.get("order"):
                parts.append(f"{ticker} {sig} order placed: {reason}.")
            else:
                parts.append(f"{ticker} {sig} ({reason}) but no order placed (flat or zero-size).")
        if not parts:
            return "All tickers held today; no trades met the signal and confidence gates."
        return " ".join(parts)


if __name__ == "__main__":
    n = TradeNarrator()
    print(f"Narrator LLM enabled: {n.enabled}")
    demo = {
        "AAPL": {"signal": "BUY", "predicted_return": 0.008, "confidence": 0.67,
                 "sentiment": {"score": 0.4, "direction": "bullish",
                               "key_events": ["Q3 earnings beat"]},
                 "blocked": False, "order": True},
        "MSFT": {"signal": "HOLD", "predicted_return": 0.001, "confidence": 0.33,
                 "sentiment": {"score": 0.0, "direction": "neutral", "key_events": []},
                 "blocked": False, "order": False},
        "GOOGL": {"signal": "SELL", "predicted_return": -0.009, "confidence": 1.0,
                  "sentiment": {"score": -0.3, "direction": "bearish",
                                "key_events": ["antitrust ruling"]},
                  "blocked": False, "order": True},
    }
    print(n.narrate(101_234.56, demo))
    print("monitoring/narrator.py: OK")
