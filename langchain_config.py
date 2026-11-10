"""NewsAPI + LangChain + Google Gemini setup for the Equity Research News Tool."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, timedelta

import requests
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from newsapi import NewsApiClient
from newsapi.newsapi_exception import NewsAPIException

from utils.news_processor import Article, build_articles_digest, process_articles

load_dotenv()

DEFAULT_MODEL = "gemini-3.5-flash-lite"
MAX_ARTICLES = 10
DAYS_BACK = 14
MAX_QUERY_LENGTH = 500


# ---------------------------------------------------------------------------
# Errors: every message is safe to show directly to the user.
# ---------------------------------------------------------------------------
class NewsToolError(Exception):
    """Base class for all expected errors."""


class ConfigurationError(NewsToolError):
    """A key or setting is missing."""


class NewsFetchError(NewsToolError):
    """NewsAPI could not return results."""


class NoArticlesError(NewsToolError):
    """The search worked but found no usable articles."""


class SummarizationError(NewsToolError):
    """Gemini could not produce a summary."""


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an AI assistant supporting an equity research analyst.
You will receive a search query and a numbered list of real news articles.

Rules:
- Use ONLY the information in the articles provided. Do not add facts from memory.
- Cite articles with their numbers, for example [1] or [2][4].
- If the articles are only loosely related to the query, say so clearly.
- If the articles conflict or lack detail, point that out instead of guessing.
- Do not give investment advice or price predictions.
- Copy numbers, currency symbols (such as ₹, $ or €), names and dates exactly as written in the articles.

Write the summary in Markdown with these sections:
**Overview** (2-3 sentences)
**Key developments** (3-5 bullet points)
**Potential market relevance** (only what the articles support)
**Risks and uncertainties** (only what the articles support)
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Query: {query}\n\nNews articles:\n{articles}"),
    ]
)


# ---------------------------------------------------------------------------
# NewsAPI
# ---------------------------------------------------------------------------
def _get_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigurationError(
            f"{name} is missing. Add it to your .env file and restart the app."
        )
    return value


def get_newsapi_client() -> NewsApiClient:
    """Create the NewsAPI client using the key from .env."""
    return NewsApiClient(api_key=_get_env("NEWSAPI_KEY"))


def _error_field(error: Exception, method_name: str) -> str:
    try:
        return str(getattr(error, method_name)())
    except Exception:
        return ""


def _translate_newsapi_error(error: NewsAPIException) -> NewsFetchError:
    code = _error_field(error, "get_code")
    message = _error_field(error, "get_message") or str(error)

    if code in {"apiKeyInvalid", "apiKeyMissing", "apiKeyDisabled"}:
        return NewsFetchError(
            "Your NewsAPI key is invalid, missing or disabled. "
            "Check NEWSAPI_KEY in your .env file."
        )
    if code in {"rateLimited", "apiKeyExhausted"}:
        return NewsFetchError(
            "NewsAPI request limit reached (the free plan allows about 100 "
            "requests per day). Please try again later."
        )
    if code == "maximumResultsReached":
        return NewsFetchError("NewsAPI's result limit was reached for this search.")
    return NewsFetchError(f"NewsAPI returned an error: {message}")


def get_news_articles(
    query: str, page_size: int = MAX_ARTICLES, days_back: int = DAYS_BACK
) -> list[dict]:
    """Retrieve raw articles from NewsAPI for the given query."""
    query = query.strip()
    if not query:
        raise NewsFetchError("Please enter a company name, ticker or topic.")
    if len(query) > MAX_QUERY_LENGTH:
        raise NewsFetchError("That search is too long. Please shorten it.")

    client = get_newsapi_client()
    from_date = (date.today() - timedelta(days=days_back)).isoformat()

    try:
        response = client.get_everything(
            q=query,
            language="en",
            sort_by="relevancy",
            from_param=from_date,
            page_size=page_size,
        )
    except NewsAPIException as error:
        raise _translate_newsapi_error(error) from error
    except requests.exceptions.RequestException as error:
        raise NewsFetchError(
            "Could not reach NewsAPI. Check your internet connection and try again."
        ) from error

    return response.get("articles", [])


def fetch_articles(query: str) -> list[Article]:
    """Retrieve articles and extract clean article information."""
    raw_articles = get_news_articles(query)
    articles = process_articles(raw_articles, max_articles=MAX_ARTICLES)
    if not articles:
        raise NoArticlesError(
            "No articles were found for this search. Try a different company "
            "name, a ticker, or broader keywords."
        )
    return articles


def summarize_articles(articles: list[Article]) -> str:
    """Combine article details into one numbered digest for the AI model."""
    return build_articles_digest(articles)


# ---------------------------------------------------------------------------
# Gemini via LangChain
# ---------------------------------------------------------------------------
def build_llm(model_name: str) -> ChatGoogleGenerativeAI:
    """Create the Gemini chat model. GOOGLE_API_KEY is read from the environment."""
    return ChatGoogleGenerativeAI(
        model=model_name,
        timeout=60,
        max_retries=2,
    )


def _message_to_text(message) -> str:
    """Extract plain text. Gemini may return a string or a list of content blocks."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content.strip()

    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts).strip()


def _translate_llm_error(error: Exception) -> NewsToolError:
    if isinstance(error, NewsToolError):
        return error

    text = str(error).lower()
    if any(s in text for s in ("api key required", "api_key_invalid", "api key not valid",
                               "permission_denied", "unauthenticated")):
        return ConfigurationError(
            "Your Google Gemini API key is missing or invalid. "
            "Check GOOGLE_API_KEY in your .env file."
        )
    if "429" in text or "resource_exhausted" in text or "quota" in text:
        return SummarizationError(
            "Gemini rate limit or quota reached. Please wait a minute and try again."
        )
    if "503" in text or "unavailable" in text or "high demand" in text:
        return SummarizationError(
            "Gemini is currently busy. Please try again in a few minutes."
        )
    if "404" in text or "not_found" in text:
        return SummarizationError(
            "The Gemini model name was not found. Change GEMINI_MODEL in your .env file."
        )
    if "timed out" in text or "timeout" in text or "connection" in text:
        return SummarizationError(
            "Could not reach Gemini. Check your internet connection and try again."
        )
    return SummarizationError(f"The AI model request failed: {str(error)[:150]}")


def generate_final_summary(query: str, digest: str) -> str:
    """Send the query and the real article text to Gemini and return the summary."""
    primary = os.getenv("GEMINI_MODEL", "").strip() or DEFAULT_MODEL
    fallback = os.getenv("GEMINI_FALLBACK_MODEL", "").strip()
    model_names = [primary] + ([fallback] if fallback and fallback != primary else [])

    last_error: Exception | None = None
    for model_name in model_names:
        try:
            chain = prompt | build_llm(model_name)
            response = chain.invoke({"query": query, "articles": digest})
            summary = _message_to_text(response)
            if summary:
                return summary
            last_error = SummarizationError("The AI model returned an empty response.")
        except Exception as error:  # translated below into a friendly message
            last_error = error
            translated = _translate_llm_error(error)
            if isinstance(translated, ConfigurationError):
                break  # a bad key will fail on the fallback model too

    raise _translate_llm_error(last_error)


# ---------------------------------------------------------------------------
# Full workflow
# ---------------------------------------------------------------------------
@dataclass
class ResearchResult:
    query: str
    articles: list[Article]
    summary: str | None
    warning: str | None = None


def get_summary(query: str) -> ResearchResult:
    """Query -> NewsAPI -> clean articles -> Gemini summary."""
    articles = fetch_articles(query)
    digest = summarize_articles(articles)

    if not digest:
        return ResearchResult(
            query=query,
            articles=articles,
            summary=None,
            warning=(
                "Articles were found, but none included a description or text, "
                "so no AI summary was generated."
            ),
        )

    summary = generate_final_summary(query, digest)
    return ResearchResult(query=query, articles=articles, summary=summary)