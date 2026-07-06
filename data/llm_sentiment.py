"""
LLM-powered news sentiment via the Anthropic Messages API.

Why this exists: VADER (data/news_sentiment.py) scores words in isolation.
"Apple crushes earnings estimates" reads as negative to VADER because
"crushes" is a violent word, and "Microsoft cuts guidance" barely
registers. A language model reads the headline the way a trader does -
it knows an earnings beat is bullish and a guidance cut is bearish.

This module asks Claude to read the day's headlines for one ticker and
return a single structured verdict: overall direction, a score in
[-1, 1], a confidence, and the concrete events it keyed on. The score
slots into the exact same `sentiment` DB column and feature the VADER
path already produces, so nothing downstream changes.

Graceful degradation is deliberate: if the `anthropic` package isn't
installed, no API key is set, or any call fails, this falls back to the
VADER scorer so a daily job never crashes over sentiment. That means the
system runs today with zero extra setup and gets smarter the moment an
ANTHROPIC_API_KEY is present.
"""
import json
import os

from data.news_sentiment import NewsSentiment

# The structured verdict we ask Claude to return. Numeric bounds aren't
# expressible in the structured-output schema, so we state the ranges in
# the prompt and clamp on our side.
_SCHEMA = {
    "type": "object",
    "properties": {
        "direction": {
            "type": "string",
            "enum": ["bullish", "bearish", "neutral"],
        },
        "score": {
            "type": "number",
            "description": "Net next-day directional read in [-1.0, 1.0]; "
                           "negative = bearish, positive = bullish.",
        },
        "confidence": {
            "type": "number",
            "description": "How decisive the news is, in [0.0, 1.0]. "
                           "Routine or mixed coverage is low.",
        },
        "key_events": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Short phrases naming the price-relevant events "
                           "(e.g. 'Q3 earnings beat', 'guidance cut', "
                           "'antitrust suit'). Empty if none.",
        },
    },
    "required": ["direction", "score", "confidence", "key_events"],
    "additionalProperties": False,
}

_SYSTEM = (
    "You are an equity news analyst. Read the day's headlines for one "
    "company and judge their net effect on the stock's NEXT trading day. "
    "Weigh materiality: earnings, guidance, M&A, regulation, and product "
    "launches move stocks; routine coverage and opinion pieces do not. "
    "Ignore sensational wording and judge the actual event. Return only "
    "the structured verdict."
)


class LLMSentiment:
    """
    Claude-based sentiment scorer with a VADER fallback.

    Public surface mirrors NewsSentiment so the two are drop-in
    interchangeable in the daily job:
        get_daily_score(ticker, date, db=None) -> float
    plus get_daily_analysis() when the caller wants the full verdict
    (used by the trade narrator and the dashboard).
    """

    def __init__(self, news_api_key: str, model: str = None,
                 anthropic_api_key: str = None):
        """
        Args:
            news_api_key:      NewsAPI key (headline source; still needed).
            model:            Claude model id. Defaults to config.LLM_MODEL.
            anthropic_api_key: Overrides ANTHROPIC_API_KEY from the env.
        """
        self._vader = NewsSentiment(news_api_key)

        if model is None:
            try:
                import config
                model = getattr(config, "LLM_MODEL", "claude-opus-4-8")
            except Exception:
                model = "claude-opus-4-8"
        self.model = model

        self._client = self._build_client(anthropic_api_key)
        self.enabled = self._client is not None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    @staticmethod
    def _build_client(anthropic_api_key: str):
        """Return an Anthropic client, or None if unavailable (-> fallback)."""
        key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            return None
        try:
            import anthropic
        except ImportError:
            print("  Note: 'anthropic' not installed - using VADER sentiment.")
            return None
        try:
            return anthropic.Anthropic(api_key=key)
        except Exception as exc:
            print(f"  Note: Anthropic client init failed ({exc}) - VADER fallback.")
            return None

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def get_daily_analysis(self, ticker: str, date: str) -> dict:
        """
        Full structured verdict for a ticker on a date.

        Returns a dict:
            {'direction', 'score', 'confidence', 'key_events',
             'source': 'llm' | 'vader', 'n_articles'}
        Never raises - falls back to VADER on any problem.
        """
        articles = self._vader.fetch_articles(ticker, date)

        if not self.enabled or not articles:
            score = self._vader.score_articles(articles)
            return {
                "direction": self._label(score),
                "score": score,
                "confidence": 0.3 if articles else 0.0,
                "key_events": [],
                "source": "vader",
                "n_articles": len(articles),
            }

        try:
            verdict = self._analyze(ticker, articles)
            verdict["source"] = "llm"
            verdict["n_articles"] = len(articles)
            return verdict
        except Exception as exc:
            print(f"  Note: LLM sentiment failed for {ticker} ({exc}) - VADER fallback.")
            score = self._vader.score_articles(articles)
            return {
                "direction": self._label(score),
                "score": score,
                "confidence": 0.3,
                "key_events": [],
                "source": "vader",
                "n_articles": len(articles),
            }

    def get_daily_score(self, ticker: str, date: str, db=None) -> float:
        """
        Compound sentiment score in [-1, 1], optionally cached in the DB.
        Signature-compatible with NewsSentiment.get_daily_score.
        """
        analysis = self.get_daily_analysis(ticker, date)
        score = analysis["score"]

        if db is not None:
            db.upsert_sentiment(date, ticker, score)

        events = ", ".join(analysis["key_events"][:3]) or "no material events"
        print(f"  Sentiment {ticker} {date}: {score:+.3f} "
              f"[{analysis['source']}] ({events})")
        return score

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _analyze(self, ticker: str, articles: list) -> dict:
        """Call Claude with structured output and clamp the result."""
        headlines = "\n".join(
            f"- {a['title']}. {a['description']}".strip()
            for a in articles
        )
        user = (
            f"Ticker: {ticker}\n"
            f"Today's headlines:\n{headlines}\n\n"
            "Return the net next-day directional read."
        )

        response = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=_SYSTEM,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
        )

        text = next(b.text for b in response.content if b.type == "text")
        data = json.loads(text)

        score = max(-1.0, min(1.0, float(data.get("score", 0.0))))
        conf  = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
        return {
            "direction":   data.get("direction", self._label(score)),
            "score":       score,
            "confidence":  conf,
            "key_events":  list(data.get("key_events", []))[:5],
        }

    @staticmethod
    def _label(score: float) -> str:
        return ("bullish" if score > 0.05
                else "bearish" if score < -0.05 else "neutral")


if __name__ == "__main__":
    import config

    scorer = LLMSentiment(config.NEWS_API_KEY)
    print(f"LLM sentiment enabled: {scorer.enabled}  (model={scorer.model})")

    from datetime import date, timedelta
    test_date = (date.today() - timedelta(days=2)).isoformat()
    for ticker in config.WATCHLIST:
        analysis = scorer.get_daily_analysis(ticker, test_date)
        print(f"  {ticker}: {analysis}")

    print("data/llm_sentiment.py: OK")
