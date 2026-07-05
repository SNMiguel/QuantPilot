"""
News sentiment fetcher and scorer.
Fetches headlines from NewsAPI and scores them with VADER (local, no extra API cost).
"""
import nltk
import requests
from datetime import datetime, timedelta

# Download VADER lexicon on first run (one-time, ~1MB)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

from nltk.sentiment.vader import SentimentIntensityAnalyzer


class NewsSentiment:
    """Fetches news articles and returns a daily sentiment score per ticker."""

    def __init__(self, api_key: str):
        self.api_key  = api_key
        self.analyzer = SentimentIntensityAnalyzer()
        self.base_url = "https://newsapi.org/v2/everything"

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------

    def fetch_articles(self, ticker: str, date: str) -> list:
        """
        Fetch up to 5 news articles mentioning ticker on a given date.

        Args:
            ticker: Stock ticker e.g. 'AAPL'
            date:   ISO date string 'YYYY-MM-DD'

        Returns:
            List of article dicts with 'title' and 'description' keys.
            Returns empty list on any error (never raises).
        """
        # Map ticker to a more searchable company name for better results
        name_map = {
            'AAPL':  'Apple',
            'MSFT':  'Microsoft',
            'GOOGL': 'Google',
        }
        query = name_map.get(ticker, ticker)

        # NewsAPI free tier: articles from last 30 days only
        params = {
            'q':        query,
            'from':     date,
            'to':       date,
            'language': 'en',
            'sortBy':   'relevancy',
            'pageSize': 5,
            'apiKey':   self.api_key,
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            articles = response.json().get('articles', [])
            return [
                {
                    'title':       a.get('title', '') or '',
                    'description': a.get('description', '') or '',
                }
                for a in articles
            ]
        except Exception as e:
            print(f"WARN: NewsAPI fetch failed for {ticker} on {date}: {e}")
            return []

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------

    def score_articles(self, articles: list) -> float:
        """
        Score a list of articles using VADER sentiment analysis.

        Scores the concatenation of title + description for each article.
        Returns the mean compound score across all articles.
        Compound score is in [-1.0, 1.0]: negative = bearish, positive = bullish.

        Returns 0.0 (neutral) if articles list is empty.
        """
        if not articles:
            return 0.0

        scores = []
        for article in articles:
            text = f"{article['title']}. {article['description']}"
            score = self.analyzer.polarity_scores(text)['compound']
            scores.append(score)

        return sum(scores) / len(scores)

    # ------------------------------------------------------------------
    # Orchestrate
    # ------------------------------------------------------------------

    def get_daily_score(self, ticker: str, date: str, db=None) -> float:
        """
        Fetch articles, score them, optionally store in DB, return score.

        Args:
            ticker: Stock ticker e.g. 'AAPL'
            date:   ISO date string 'YYYY-MM-DD'
            db:     Optional Database instance to cache the score.

        Returns:
            Compound sentiment score in [-1.0, 1.0].
        """
        articles = self.fetch_articles(ticker, date)
        score    = self.score_articles(articles)

        if db is not None:
            db.upsert_sentiment(date, ticker, score)

        print(f"  Sentiment {ticker} {date}: {score:+.3f}  ({len(articles)} articles)")
        return score


if __name__ == "__main__":
    import config

    ns = NewsSentiment(config.NEWS_API_KEY)

    # Test with a recent date (NewsAPI free tier: last 30 days only)
    from datetime import date, timedelta
    test_date = (date.today() - timedelta(days=2)).isoformat()

    for ticker in config.WATCHLIST:
        score = ns.get_daily_score(ticker, test_date)

    print("data/news_sentiment.py: OK")
