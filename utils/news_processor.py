"""Helpers that clean raw NewsAPI articles and prepare them for the AI model."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone

HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
TRUNCATION_PATTERN = re.compile(r"\[\+\d+\s+chars\]")
REMOVED_MARKER = "[Removed]"


@dataclass(frozen=True)
class Article:
    """One cleaned news article."""

    title: str
    source: str
    published_at: str
    url: str
    description: str
    content: str

    @property
    def text(self) -> str:
        """Best available text: the description, or the content snippet if missing."""
        return self.description or self.content

    @property
    def has_text(self) -> bool:
        return bool(self.text)


def clean_text(value) -> str:
    """Remove HTML tags, NewsAPI truncation markers and extra whitespace."""
    if not value:
        return ""
    text = HTML_TAG_PATTERN.sub(" ", str(value))
    text = TRUNCATION_PATTERN.sub("", text)
    text = " ".join(text.split())
    return "" if text == REMOVED_MARKER else text


def format_published_date(raw_date) -> str:
    """Convert '2026-11-10T14:30:00Z' into '10 Nov 2026, 14:30 UTC'."""
    if not raw_date:
        return "Date unavailable"
    try:
        parsed = datetime.fromisoformat(str(raw_date).replace("Z", "+00:00"))
    except ValueError:
        return "Date unavailable"
    return parsed.astimezone(timezone.utc).strftime("%d %b %Y, %H:%M UTC")


def process_articles(raw_articles: list[dict], max_articles: int = 10) -> list[Article]:
    """Turn raw NewsAPI dictionaries into clean, de-duplicated Article objects."""
    articles: list[Article] = []
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()

    for raw in raw_articles:
        title = clean_text(raw.get("title"))
        url = (raw.get("url") or "").strip()

        # Skip removed or unusable entries.
        if not title or not url:
            continue

        # Skip duplicates (same link or same headline).
        title_key = title.lower()
        if url in seen_urls or title_key in seen_titles:
            continue
        seen_urls.add(url)
        seen_titles.add(title_key)

        source_info = raw.get("source") or {}
        articles.append(
            Article(
                title=title,
                source=clean_text(source_info.get("name")) or "Unknown source",
                published_at=format_published_date(raw.get("publishedAt")),
                url=url,
                description=clean_text(raw.get("description")),
                content=clean_text(raw.get("content")),
            )
        )
        if len(articles) >= max_articles:
            break

    return articles


def build_articles_digest(articles: list[Article], max_chars_per_article: int = 700) -> str:
    """Build the numbered text block that is sent to the AI model.

    Numbers match the article's position in the full list, so a citation
    like [3] in the summary points to article 3 in the app. Articles with
    no text are skipped here but still appear in the on-screen list.
    """
    blocks = []
    for number, article in enumerate(articles, start=1):
        if not article.has_text:
            continue
        text = article.text
        if len(text) > max_chars_per_article:
            text = text[:max_chars_per_article].rstrip() + "..."
        blocks.append(
            f"[{number}] {article.title}\n"
            f"Source: {article.source} | Published: {article.published_at}\n"
            f"Text: {text}"
        )
    return "\n\n".join(blocks)