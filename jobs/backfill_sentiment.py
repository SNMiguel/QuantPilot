"""
Backfill historical daily sentiment into the DB.

The live daily job pulls headlines from NewsAPI, whose free tier only
reaches back about 30 days, so a multi-year backtest runs with no
sentiment. Alpaca's news endpoint (data.alpaca.markets/v1beta1/news)
serves historical headlines for years back at no extra cost, using the
same Alpaca credentials already in .env.

This job pages through that endpoint for a ticker and window, buckets the
headlines by calendar day, scores each day with the same LLMSentiment
scorer the live path uses (Claude when ANTHROPIC_API_KEY is set, VADER
otherwise), and upserts one score per day into the sentiment table. After
running it, backtest_job sees a populated sentiment feature instead of
zeros.

Usage:
    python -m jobs.backfill_sentiment --ticker AAPL --start 2024-01-01
    python -m jobs.backfill_sentiment --ticker AAPL --vader   # skip LLM
"""
import argparse
from collections import defaultdict
from datetime import date, timedelta

import requests

import config
from data.database import Database
from data.llm_sentiment import LLMSentiment

_NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
_MAX_HEADLINES_PER_DAY = 8  # keep the per-day LLM prompt small


def fetch_news(ticker: str, start: str, end: str) -> dict:
    """
    Page through Alpaca news and bucket articles by calendar day.

    Returns {date_str: [ {title, description}, ... ]}.
    """
    headers = {
        "APCA-API-KEY-ID": config.ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": config.ALPACA_SECRET_KEY,
    }
    by_day = defaultdict(list)
    page_token = None
    total = 0

    while True:
        params = {
            "symbols": ticker,
            "start": start,
            "end": end,
            "limit": 50,
            "sort": "asc",
        }
        if page_token:
            params["page_token"] = page_token

        resp = requests.get(_NEWS_URL, headers=headers, params=params, timeout=20)
        resp.raise_for_status()
        payload = resp.json()

        for a in payload.get("news", []):
            day = a.get("created_at", "")[:10]
            if not day:
                continue
            by_day[day].append({
                "title": a.get("headline", "") or "",
                "description": a.get("summary", "") or "",
            })
            total += 1

        page_token = payload.get("next_page_token")
        if not page_token:
            break

    print(f"  Fetched {total} articles across {len(by_day)} days")
    return by_day


def backfill(ticker: str, start: str, end: str, use_llm: bool = True) -> int:
    print(f"\n{'='*55}")
    print(f"  Sentiment backfill: {ticker}  {start} to {end}")
    print(f"{'='*55}")

    db = Database(config.DB_URL)
    scorer = LLMSentiment(news_api_key=config.NEWS_API_KEY)
    engine = "Claude" if (use_llm and scorer.enabled) else "VADER"
    print(f"  Scoring engine: {engine}")

    print("Fetching historical news from Alpaca...")
    by_day = fetch_news(ticker, start, end)
    if not by_day:
        print("  No news returned for this window. Nothing to backfill.")
        return 0

    written = 0
    for day in sorted(by_day):
        articles = by_day[day][:_MAX_HEADLINES_PER_DAY]

        if use_llm and scorer.enabled:
            try:
                verdict = scorer._analyze(ticker, articles)
                score = verdict["score"]
            except Exception as exc:
                print(f"  {day}: LLM failed ({exc}) - VADER fallback")
                score = scorer._vader.score_articles(articles)
        else:
            score = scorer._vader.score_articles(articles)

        db.upsert_sentiment(day, ticker, score)
        written += 1
        if written % 25 == 0:
            print(f"  ...{written} days scored (latest {day}: {score:+.3f})")

    print(f"  Wrote {written} daily sentiment scores to the DB.")
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill historical sentiment")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--start", default=None,
                        help="YYYY-MM-DD (default: 2 years ago)")
    parser.add_argument("--end", default=None,
                        help="YYYY-MM-DD (default: today)")
    parser.add_argument("--vader", action="store_true",
                        help="Force VADER scoring even if Claude is available")
    args = parser.parse_args()

    end = args.end or date.today().isoformat()
    start = args.start or (date.today() - timedelta(days=730)).isoformat()

    backfill(args.ticker, start, end, use_llm=not args.vader)
