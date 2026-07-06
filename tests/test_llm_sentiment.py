"""
LLM sentiment scorer - fallback correctness and stubbed-client parsing.
No network and no `anthropic` package are required.
"""
import json

import pytest

from data.llm_sentiment import LLMSentiment


class _StubTextBlock:
    type = "text"
    def __init__(self, text):
        self.text = text


class _StubResponse:
    def __init__(self, text):
        self.content = [_StubTextBlock(text)]


class _StubMessages:
    def __init__(self, payload):
        self._payload = payload
    def create(self, **kwargs):
        return _StubResponse(json.dumps(self._payload))


class _StubClient:
    def __init__(self, payload):
        self.messages = _StubMessages(payload)


@pytest.fixture
def scorer():
    s = LLMSentiment("dummy_news_key")
    # Force the fallback path deterministically regardless of local env.
    s._client = None
    s.enabled = False
    return s


def test_falls_back_to_vader_without_client(scorer, monkeypatch):
    monkeypatch.setattr(scorer, "fetch_articles",
                        lambda t, d: [{"title": "Apple beats earnings",
                                       "description": "strong quarter"}])
    out = scorer.get_daily_analysis("AAPL", "2026-01-02")
    assert out["source"] == "vader"
    assert -1.0 <= out["score"] <= 1.0
    assert set(out) >= {"direction", "score", "confidence", "key_events"}


def test_empty_news_is_neutral(scorer, monkeypatch):
    monkeypatch.setattr(scorer, "fetch_articles", lambda t, d: [])
    out = scorer.get_daily_analysis("AAPL", "2026-01-02")
    assert out["score"] == 0.0
    assert out["confidence"] == 0.0
    assert out["n_articles"] == 0


def test_llm_path_parses_and_clamps(scorer, monkeypatch):
    monkeypatch.setattr(scorer, "fetch_articles",
                        lambda t, d: [{"title": "x", "description": "y"}])
    # Out-of-range score must clamp to [-1, 1]; confidence to [0, 1].
    scorer._client = _StubClient({
        "direction": "bullish", "score": 1.9, "confidence": 1.4,
        "key_events": ["Q3 earnings beat", "raised guidance"],
    })
    scorer.enabled = True
    out = scorer.get_daily_analysis("AAPL", "2026-01-02")
    assert out["source"] == "llm"
    assert out["score"] == 1.0
    assert out["confidence"] == 1.0
    assert "Q3 earnings beat" in out["key_events"]


def test_get_daily_score_upserts(scorer, monkeypatch):
    monkeypatch.setattr(scorer, "fetch_articles", lambda t, d: [])
    calls = {}
    class _DB:
        def upsert_sentiment(self, date, ticker, score):
            calls["args"] = (date, ticker, score)
    score = scorer.get_daily_score("MSFT", "2026-01-02", db=_DB())
    assert calls["args"][1] == "MSFT"
    assert calls["args"][2] == score
