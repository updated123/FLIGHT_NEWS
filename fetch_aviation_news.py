#!/usr/bin/env python3
"""Fetch aviation news from aviationa2z.com via WordPress REST API."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from html import unescape
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

BASE_URL = "https://aviationa2z.com"
API_POSTS = f"{BASE_URL}/wp-json/wp/v2/posts"
USER_AGENT = "Mozilla/5.0 (compatible; AviationNewsFetcher/1.0)"
MAX_PER_PAGE = 100
SITE_TZ = ZoneInfo("Asia/Kolkata")


def http_get(url: str, timeout: int = 30) -> tuple[bytes, dict[str, str]]:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        headers = {key.lower(): value for key, value in response.headers.items()}
        return response.read(), headers


def strip_html(html: str) -> str:
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
    text = re.sub(r"(?is)<br\s*/?>", "\n", text)
    text = re.sub(r"(?is)</p>", "\n\n", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_terms(post: dict[str, Any]) -> tuple[list[str], list[str]]:
    categories: list[str] = []
    tags: list[str] = []
    for group in post.get("_embedded", {}).get("wp:term", []) or []:
        for term in group:
            taxonomy = term.get("taxonomy")
            name = term.get("name")
            if not name:
                continue
            if taxonomy == "category":
                categories.append(name)
            elif taxonomy == "post_tag":
                tags.append(name)
    return categories, tags


def parse_featured_image(post: dict[str, Any]) -> str | None:
    media_items = post.get("_embedded", {}).get("wp:featuredmedia") or []
    if not media_items:
        return None
    return media_items[0].get("source_url")


def parse_author(post: dict[str, Any]) -> str | None:
    authors = post.get("_embedded", {}).get("author") or []
    if not authors:
        return None
    return authors[0].get("name")


def normalize_post(post: dict[str, Any], include_content: bool) -> dict[str, Any]:
    title = unescape(re.sub(r"<[^>]+>", "", post.get("title", {}).get("rendered", "")))
    excerpt_html = post.get("excerpt", {}).get("rendered", "")
    content_html = post.get("content", {}).get("rendered", "")
    categories, tags = parse_terms(post)

    item: dict[str, Any] = {
        "id": post.get("id"),
        "title": title,
        "url": post.get("link"),
        "slug": post.get("slug"),
        "published_at": post.get("date"),
        "published_at_gmt": post.get("date_gmt"),
        "modified_at": post.get("modified"),
        "author": parse_author(post),
        "categories": categories,
        "tags": tags,
        "featured_image": parse_featured_image(post),
        "excerpt": strip_html(excerpt_html),
    }
    if include_content:
        item["content_text"] = strip_html(content_html)
        item["word_count"] = len(item["content_text"].split())
    return item


def fetch_posts(
    *,
    limit: int,
    category: str | None,
    include_content: bool,
    delay_seconds: float,
    on_date: date | None = None,
) -> list[dict[str, Any]]:
    articles: list[dict[str, Any]] = []
    page = 1

    while len(articles) < limit:
        per_page = min(MAX_PER_PAGE, limit - len(articles))
        params: dict[str, Any] = {
            "per_page": per_page,
            "page": page,
            "_embed": 1,
        }
        if category:
            params["categories"] = category
        if on_date:
            start = datetime.combine(on_date, datetime.min.time(), tzinfo=SITE_TZ)
            end = start + timedelta(days=1)
            params["after"] = start.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            params["before"] = end.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        url = f"{API_POSTS}?{urlencode(params)}"
        try:
            payload, headers = http_get(url)
        except HTTPError as exc:
            if exc.code == 400 and page > 1:
                break
            raise
        except URLError as exc:
            raise SystemExit(f"Network error: {exc}") from exc

        posts = json.loads(payload.decode("utf-8"))
        if not posts:
            break

        for post in posts:
            articles.append(normalize_post(post, include_content))
            if len(articles) >= limit:
                break

        total_pages = int(headers.get("x-wp-totalpages", "1"))
        if page >= total_pages:
            break
        page += 1
        if delay_seconds:
            time.sleep(delay_seconds)

    return articles


def fetch_categories() -> list[dict[str, Any]]:
    url = f"{BASE_URL}/wp-json/wp/v2/categories?per_page=100"
    payload, _ = http_get(url)
    categories = json.loads(payload.decode("utf-8"))
    return [
        {
            "id": cat["id"],
            "name": cat["name"],
            "slug": cat["slug"],
            "count": cat.get("count", 0),
        }
        for cat in categories
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch news from aviationa2z.com")
    parser.add_argument("--limit", type=int, default=50, help="Max articles to fetch (default: 50)")
    parser.add_argument(
        "--category",
        help="Category slug filter (e.g. airline-news, airport-news). Use --list-categories.",
    )
    parser.add_argument(
        "--full-content",
        action="store_true",
        help="Include full article text (slower, larger JSON)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="aviation_news.json",
        help="Output JSON file (default: aviation_news.json)",
    )
    parser.add_argument(
        "--today",
        action="store_true",
        help="Fetch only articles published today (Asia/Kolkata site timezone)",
    )
    parser.add_argument(
        "--date",
        help="Fetch articles for a specific date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="Print available categories and exit",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="Delay between paginated API requests in seconds (default: 0.2)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.list_categories:
        categories = fetch_categories()
        for cat in sorted(categories, key=lambda item: item["name"].lower()):
            print(f"{cat['slug']:30} {cat['name']} ({cat['count']})")
        return

    if args.today and args.date:
        raise SystemExit("Use either --today or --date, not both.")

    on_date: date | None = None
    if args.today:
        on_date = datetime.now(SITE_TZ).date()
    elif args.date:
        on_date = date.fromisoformat(args.date)

    category_id: str | None = None
    if args.category:
        categories = fetch_categories()
        match = next((c for c in categories if c["slug"] == args.category), None)
        if not match:
            slugs = ", ".join(c["slug"] for c in categories[:12])
            raise SystemExit(f"Unknown category '{args.category}'. Examples: {slugs}")
        category_id = str(match["id"])

    label = on_date.isoformat() if on_date else "latest"
    print(f"Fetching up to {args.limit} {label} articles from {BASE_URL} ...", file=sys.stderr)
    articles = fetch_posts(
        limit=args.limit,
        category=category_id,
        include_content=args.full_content,
        delay_seconds=args.delay,
        on_date=on_date,
    )

    output = {
        "source": BASE_URL,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "date_filter": on_date.isoformat() if on_date else None,
        "timezone": "Asia/Kolkata",
        "count": len(articles),
        "articles": articles,
    }

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"Saved {len(articles)} articles to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
