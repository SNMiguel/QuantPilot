"""
Historical and recent news headlines from Alpaca's news API.

One source for headlines everywhere: the live daily job and the historical
backfill both read from here, so live and backtested sentiment come from
the same place. NewsAPI's free tier only reaches back about 30 days and
returns nothing on some days; Alpaca's archive covers years with the
Alpaca credentials already in .env.

Every function returns article dicts shaped like {'title', 'description'}
so they drop straight into LLMSentiment and the VADER scorer. All failures
degrade to an empty list rather than raising - sentiment is never allowed
to crash a job.
"""
from collections import defaultdict
from datetime import datetime, timedelta

import requests

_NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
_MAX_HEADLINES_PER_DAY = 8   # keep per-day LLM prompts small


def _headers(api_key: str, secret_key: str) -> dict:
    return {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }


def fetch_news_range(ticker: str, start: str, end: str,
                     api_key: str, secret_key: str,
                     max_per_day: int = _MAX_HEADLINES_PER_DAY) -> dict:
    """
    Page through Alpaca news and bucket articles by calendar day.

    Args:
        ticker:     Symbol, e.g. 'AAPL'.
        start, end: ISO dates 'YYYY-MM-DD' (inclusive range).
        api_key, secret_key: Alpaca credentials.
        max_per_day: Cap on articles kept per day.

    Returns:
        {date_str: [{'title', 'description'}, ...]}. Empty on any error.
    """
    by_day = defaultdict(list)
    page_token = None
    try:
        while True:
            params = {
                "symbols": ticker,
                "start":   start,
                "end":     end,
                "limit":   50,
                "sort":    "asc",
            }
            if page_token:
                params["page_token"] = page_token

            resp = requests.get(_NEWS_URL, headers=_headers(api_key, secret_key),
                                params=params, timeout=20)
            resp.raise_for_status()
            payload = resp.json()

            for a in payload.get("news", []):
                day = a.get("created_at", "")[:10]
                if not day or len(by_day[day]) >= max_per_day:
                    continue
                by_day[day].append({
                    "title":       a.get("headline", "") or "",
                    "description": a.get("summary", "") or "",
                })

            page_token = payload.get("next_page_token")
            if not page_token:
                break
    except Exception as exc:
        print(f"WARN: Alpaca news fetch failed for {ticker} "
              f"{start}..{end}: {exc}")
        return {}

    return dict(by_day)


def fetch_daily_articles(ticker: str, date: str,
                         api_key: str, secret_key: str,
                         max_articles: int = _MAX_HEADLINES_PER_DAY) -> list:
    """
    Fetch one calendar day's articles for a ticker.

    Alpaca treats a bare date as midnight, so a same-day start/end is a
    zero-width range; we query [date, date+1) and keep that day's news.
    Returns an empty list on any error (never raises).
    """
    try:
        nxt = (datetime.strptime(date, "%Y-%m-%d")
               + timedelta(days=1)).strftime("%Y-%m-%d")
    except ValueError:
        return []
    by_day = fetch_news_range(ticker, date, nxt, api_key, secret_key,
                              max_per_day=max_articles)
    return by_day.get(date, [])[:max_articles]


if __name__ == "__main__":
    import config
    arts = fetch_daily_articles("AAPL", "2024-01-04",
                                config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
    print(f"AAPL 2024-01-04: {len(arts)} articles")
    for a in arts[:3]:
        print(f"  - {a['title']}")
    print("data/alpaca_news.py: OK")
