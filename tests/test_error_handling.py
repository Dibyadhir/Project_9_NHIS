"""Tests for error handling and article processing.

These tests use fake responses, so they need no internet connection and
use none of your NewsAPI or Gemini quota.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

import requests
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from newsapi.newsapi_exception import NewsAPIException

import langchain_config as cfg
from utils.news_processor import Article, build_articles_digest, process_articles


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
def newsapi_error(code: str, message: str = "test error") -> NewsAPIException:
    return NewsAPIException({"status": "error", "code": code, "message": message})


def fake_newsapi_client(error: Exception) -> MagicMock:
    client = MagicMock()
    client.get_everything.side_effect = error
    return client


def make_article(title="Headline", description="Some description", content="") -> Article:
    return Article(
        title=title,
        source="Test Source",
        published_at="01 Jan 2026, 10:00 UTC",
        url=f"https://example.com/{title.replace(' ', '-')}",
        description=description,
        content=content,
    )


def llm_that_returns(content) -> RunnableLambda:
    return RunnableLambda(lambda _: AIMessage(content=content))


def llm_that_fails(message: str) -> RunnableLambda:
    def fail(_):
        raise RuntimeError(message)

    return RunnableLambda(fail)


MODEL_ENV = {"GEMINI_MODEL": "model-a", "GEMINI_FALLBACK_MODEL": "model-b"}


# ---------------------------------------------------------------------------
# NewsAPI errors
# ---------------------------------------------------------------------------
class TestNewsFetching(unittest.TestCase):
    def test_empty_query_is_rejected(self):
        with self.assertRaises(cfg.NewsFetchError):
            cfg.get_news_articles("   ")

    def test_missing_newsapi_key(self):
        with patch.dict(os.environ, {"NEWSAPI_KEY": ""}):
            with self.assertRaises(cfg.ConfigurationError):
                cfg.get_news_articles("Tesla")

    def test_invalid_newsapi_key(self):
        client = fake_newsapi_client(newsapi_error("apiKeyInvalid"))
        with patch.object(cfg, "get_newsapi_client", return_value=client):
            with self.assertRaises(cfg.NewsFetchError) as context:
                cfg.get_news_articles("Tesla")
        self.assertIn("invalid", str(context.exception).lower())

    def test_newsapi_rate_limit(self):
        client = fake_newsapi_client(newsapi_error("rateLimited"))
        with patch.object(cfg, "get_newsapi_client", return_value=client):
            with self.assertRaises(cfg.NewsFetchError) as context:
                cfg.get_news_articles("Tesla")
        self.assertIn("limit", str(context.exception).lower())

    def test_network_failure(self):
        client = fake_newsapi_client(requests.exceptions.ConnectionError("offline"))
        with patch.object(cfg, "get_newsapi_client", return_value=client):
            with self.assertRaises(cfg.NewsFetchError) as context:
                cfg.get_news_articles("Tesla")
        self.assertIn("internet", str(context.exception).lower())

    def test_other_newsapi_error_shows_message(self):
        client = fake_newsapi_client(newsapi_error("unexpectedError", "Something broke"))
        with patch.object(cfg, "get_newsapi_client", return_value=client):
            with self.assertRaises(cfg.NewsFetchError) as context:
                cfg.get_news_articles("Tesla")
        self.assertIn("Something broke", str(context.exception))

    def test_no_articles_found(self):
        with patch.object(cfg, "get_news_articles", return_value=[]):
            with self.assertRaises(cfg.NoArticlesError):
                cfg.fetch_articles("Tesla")

    def test_only_removed_articles_counts_as_no_articles(self):
        removed = [{"title": "[Removed]", "url": "https://removed.com"}]
        with patch.object(cfg, "get_news_articles", return_value=removed):
            with self.assertRaises(cfg.NoArticlesError):
                cfg.fetch_articles("Tesla")


# ---------------------------------------------------------------------------
# Article cleaning
# ---------------------------------------------------------------------------
class TestArticleProcessing(unittest.TestCase):
    def test_missing_description_falls_back_to_content(self):
        raw = [
            {
                "title": "Headline",
                "url": "https://a.com",
                "description": None,
                "content": "Fallback text [+1200 chars]",
                "source": {"name": "Source A"},
                "publishedAt": "2026-11-10T14:30:00Z",
            }
        ]
        article = process_articles(raw)[0]
        self.assertEqual(article.text, "Fallback text")
        self.assertEqual(article.published_at, "10 Nov 2026, 14:30 UTC")

    def test_article_without_text_is_kept_but_not_summarized(self):
        raw = [{"title": "Headline", "url": "https://a.com", "description": None, "content": None}]
        articles = process_articles(raw)
        self.assertEqual(len(articles), 1)
        self.assertFalse(articles[0].has_text)
        self.assertEqual(build_articles_digest(articles), "")

    def test_digest_numbers_match_article_positions(self):
        articles = [make_article("No text", description=""), make_article("Has text")]
        digest = build_articles_digest(articles)
        self.assertTrue(digest.startswith("[2]"))
        self.assertNotIn("[1]", digest)

    def test_duplicate_articles_are_removed(self):
        raw = [
            {"title": "Same story", "url": "https://a.com/1", "description": "x"},
            {"title": "Same story", "url": "https://a.com/1", "description": "x"},
        ]
        self.assertEqual(len(process_articles(raw)), 1)

    def test_html_is_removed_from_text(self):
        raw = [{"title": "H", "url": "https://a.com", "description": "<p>Hello <b>world</b></p>"}]
        self.assertEqual(process_articles(raw)[0].description, "Hello world")


# ---------------------------------------------------------------------------
# Gemini errors and behavior
# ---------------------------------------------------------------------------
@patch.dict(os.environ, MODEL_ENV)
class TestSummarization(unittest.TestCase):
    def test_model_receives_query_and_article_text(self):
        captured = []

        def capture(prompt_value):
            captured.append(prompt_value.to_string())
            return AIMessage(content="Summary [1]")

        digest = "[1] Tesla headline\nText: Deliveries rose"
        with patch.object(cfg, "build_llm", return_value=RunnableLambda(capture)):
            summary = cfg.generate_final_summary("Tesla stock news", digest)

        self.assertEqual(summary, "Summary [1]")
        self.assertIn("Tesla stock news", captured[0])
        self.assertIn("Deliveries rose", captured[0])

    def test_content_blocks_are_converted_to_text(self):
        blocks = [{"type": "text", "text": "Hello "}, {"type": "text", "text": "there!"}]
        with patch.object(cfg, "build_llm", return_value=llm_that_returns(blocks)):
            self.assertEqual(cfg.generate_final_summary("q", "[1] x"), "Hello there!")

    def test_gemini_busy_503(self):
        failing = llm_that_fails("503 UNAVAILABLE. This model is currently experiencing high demand.")
        with patch.object(cfg, "build_llm", return_value=failing):
            with self.assertRaises(cfg.SummarizationError) as context:
                cfg.generate_final_summary("q", "[1] x")
        self.assertIn("busy", str(context.exception).lower())

    def test_gemini_rate_limit_429(self):
        failing = llm_that_fails("429 RESOURCE_EXHAUSTED. You exceeded your current quota.")
        with patch.object(cfg, "build_llm", return_value=failing):
            with self.assertRaises(cfg.SummarizationError) as context:
                cfg.generate_final_summary("q", "[1] x")
        self.assertIn("rate limit", str(context.exception).lower())

    def test_invalid_gemini_key_does_not_try_fallback(self):
        calls = []

        def factory(model_name):
            calls.append(model_name)
            return llm_that_fails("API key not valid. Please pass a valid API key.")

        with patch.object(cfg, "build_llm", side_effect=factory):
            with self.assertRaises(cfg.ConfigurationError):
                cfg.generate_final_summary("q", "[1] x")
        self.assertEqual(calls, ["model-a"])

    def test_fallback_model_is_used_when_primary_is_busy(self):
        def factory(model_name):
            if model_name == "model-a":
                return llm_that_fails("503 UNAVAILABLE high demand")
            return llm_that_returns("Fallback summary")

        with patch.object(cfg, "build_llm", side_effect=factory):
            self.assertEqual(cfg.generate_final_summary("q", "[1] x"), "Fallback summary")

    def test_empty_response_is_an_error(self):
        with patch.object(cfg, "build_llm", return_value=llm_that_returns("")):
            with self.assertRaises(cfg.SummarizationError):
                cfg.generate_final_summary("q", "[1] x")


# ---------------------------------------------------------------------------
# Whole workflow
# ---------------------------------------------------------------------------
class TestWorkflow(unittest.TestCase):
    def test_no_summary_is_generated_when_articles_have_no_text(self):
        articles = [make_article(description="", content="")]
        with patch.object(cfg, "fetch_articles", return_value=articles), patch.object(
            cfg, "generate_final_summary"
        ) as mock_generate:
            result = cfg.get_summary("Tesla")

        self.assertIsNone(result.summary)
        self.assertIsNotNone(result.warning)
        mock_generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()