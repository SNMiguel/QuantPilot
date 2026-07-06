"""
Local demonstration of the Claude-powered intelligence layer.

Runs the two LLM features end to end using ONLY an ANTHROPIC_API_KEY:
no Alpaca, NewsAPI, database, or trained models are touched. Sample
headlines stand in for the live NewsAPI feed so the pipeline can be shown
on any machine.

    1. LLMSentiment reads each ticker's headlines and returns a structured
       verdict (direction, score, confidence, key events).
    2. Those verdicts, plus mock signal/return numbers, are handed to the
       TradeNarrator, which writes the plain-English daily rationale that
       would normally post to Discord.

Usage:
    ANTHROPIC_API_KEY=sk-ant-...  python demo_local.py
    (or put ANTHROPIC_API_KEY in .env)
"""
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from data.llm_sentiment import LLMSentiment
from monitoring.narrator import TradeNarrator


# Sample headlines standing in for a live NewsAPI fetch. Each list is what
# fetch_articles() would return: dicts with 'title' and 'description'.
SAMPLE_HEADLINES = {
    "AAPL": [
        {"title": "Apple crushes Q3 earnings estimates on record iPhone demand",
         "description": "Revenue and EPS both beat consensus; services hit an all-time high."},
        {"title": "Apple raises full-year guidance, announces expanded buyback",
         "description": "Management cited strong momentum heading into the holiday quarter."},
    ],
    "MSFT": [
        {"title": "Microsoft reports in-line quarter, cloud growth steady",
         "description": "Azure grew as expected; no change to the outlook."},
        {"title": "Analysts hold Microsoft rating unchanged after routine update",
         "description": "Commentary described the print as uneventful."},
    ],
    "GOOGL": [
        {"title": "Google hit with antitrust ruling, judge orders remedies",
         "description": "Court found the company maintained an illegal search monopoly."},
        {"title": "Alphabet shares slip as regulators weigh structural penalties",
         "description": "Uncertainty over potential breakup weighed on the stock."},
    ],
}

# Mock signal numbers that a real daily job would compute from the models.
# Direction here is chosen to line up with the sample news for a clean demo.
MOCK_SIGNALS = {
    "AAPL":  {"signal": "BUY",  "predicted_return": 0.0091, "confidence": 0.67, "blocked": False, "order": True},
    "MSFT":  {"signal": "HOLD", "predicted_return": 0.0004, "confidence": 0.33, "blocked": False, "order": False},
    "GOOGL": {"signal": "SELL", "predicted_return": -0.0117, "confidence": 1.00, "blocked": False, "order": True},
}


def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set. Set it in your environment or .env "
              "to run the live demo. Without it the modules fall back to VADER "
              "and a templated summary.\n")

    # news_api_key is unused here because we feed headlines directly, so an
    # empty string is fine - nothing calls NewsAPI in this demo.
    scorer   = LLMSentiment(news_api_key="")
    narrator = TradeNarrator()

    print("=" * 70)
    print(f"Sentiment engine : {'Claude (' + scorer.model + ')' if scorer.enabled else 'VADER fallback'}")
    print(f"Narrator engine  : {'Claude (' + narrator.model + ')' if narrator.enabled else 'templated fallback'}")
    print("=" * 70)

    context = {}
    for ticker, articles in SAMPLE_HEADLINES.items():
        print(f"\n{ticker} headlines:")
        for a in articles:
            print(f"  - {a['title']}")

        # Bypass NewsAPI: analyze the sample headlines directly. Falls back
        # to a neutral VADER-style verdict if Claude is unavailable.
        if scorer.enabled:
            verdict = scorer._analyze(ticker, articles)
            verdict["source"] = "llm"
        else:
            score = scorer._vader.score_articles(articles)
            verdict = {"direction": scorer._label(score), "score": score,
                       "confidence": 0.3, "key_events": [], "source": "vader"}

        print(f"  -> {verdict['direction'].upper()}  "
              f"score={verdict['score']:+.2f}  conf={verdict['confidence']:.2f}  "
              f"events={verdict['key_events'] or 'none'}")

        context[ticker] = {**MOCK_SIGNALS[ticker], "sentiment": verdict,
                           "regime": "normal"}

    print("\n" + "=" * 70)
    print("Daily rationale (this is what posts to Discord):\n")
    print(narrator.narrate(portfolio_value=101_234.56, context=context))
    print("=" * 70)


if __name__ == "__main__":
    main()
