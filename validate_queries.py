"""Live validation: runs the four example queries against the real APIs.

Run it with:  python validate_queries.py
It uses about 4 NewsAPI requests and 4 Gemini calls.
"""

import re
import time

import requests

from langchain_config import (
    NewsToolError,
    fetch_articles,
    generate_final_summary,
    summarize_articles,
)

QUERIES = [
    "Tesla stock news",
    "Apple financial news",
    "NVIDIA AI market news",
    "Reliance Industries stock news",
]

REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; EquityResearchNewsTool/1.0)"}


def check_link(url: str) -> tuple[str, str]:
    """Return (status, detail). Many news sites block scripts, so 403 is not a failure."""
    try:
        response = requests.get(
            url, headers=REQUEST_HEADERS, timeout=10, stream=True, allow_redirects=True
        )
        response.close()
    except requests.exceptions.RequestException as error:
        return "FAIL", type(error).__name__

    code = response.status_code
    if code < 400:
        return "OK", str(code)
    if code in (401, 403, 429):
        return "BLOCKED", f"{code} (site blocks scripts, open it manually)"
    return "FAIL", str(code)


def check_citations(summary: str, articles) -> list[str]:
    """Every [n] in the summary must point to a real article that had text."""
    problems = []
    cited = {int(number) for number in re.findall(r"\[(\d+)\]", summary)}
    if not cited:
        problems.append("The summary cites no articles.")
    for number in sorted(cited):
        if number < 1 or number > len(articles):
            problems.append(f"Citation [{number}] does not match any article.")
        elif not articles[number - 1].has_text:
            problems.append(f"Citation [{number}] points to an article with no text.")
    return problems


def validate_query(query: str) -> bool:
    print("=" * 78)
    print(f"QUERY: {query}")
    print("=" * 78)

    try:
        articles = fetch_articles(query)
    except NewsToolError as error:
        print(f"FAIL: could not retrieve articles. {error}\n")
        return False

    print(f"Articles retrieved: {len(articles)}")
    link_failures = 0
    for number, article in enumerate(articles, start=1):
        status, detail = check_link(article.url)
        if status == "FAIL":
            link_failures += 1
        print(f"  [{number}] {article.title[:70]}")
        print(f"       {article.source} | {article.published_at} | link {status} ({detail})")

    digest = summarize_articles(articles)
    if not digest:
        print("FAIL: no article had any text to summarize.\n")
        return False

    try:
        summary = generate_final_summary(query, digest)
    except NewsToolError as error:
        print(f"FAIL: summary was not generated. {error}\n")
        return False

    print("\n--- AI SUMMARY ---\n")
    print(summary)

    problems = check_citations(summary, articles)
    print("\n--- CHECKS ---")
    print(f"Articles found ........ PASS ({len(articles)})")
    print(f"Links reachable ....... {'PASS' if link_failures == 0 else f'CHECK ({link_failures} failed)'}")
    print(f"Citations valid ....... {'PASS' if not problems else 'FAIL'}")
    for problem in problems:
        print(f"    - {problem}")
    print()
    return not problems and link_failures == 0


def main() -> None:
    results = {}
    for index, query in enumerate(QUERIES):
        results[query] = validate_query(query)
        if index < len(QUERIES) - 1:
            time.sleep(2)  # be gentle with the free API limits

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for query, passed in results.items():
        print(f"{'PASS' if passed else 'CHECK'}  {query}")
    print("\nA CHECK result is not always a bug. Read the details above.")


if __name__ == "__main__":
    main()