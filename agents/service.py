"""Booking analysis service — fetch news, process, return compact impact response."""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

from groq import Groq

from agents.groq_utils import groq_chat_with_retry
from agents.flight_booking_agent import (
    DEFAULT_MODEL,
    SITE_TZ,
    get_groq_client,
    parse_json_response,
)
from fetch_aviation_news import fetch_all_posts

SITE_TZ_NAME = "Asia/Kolkata"
NEWS_BATCH_SIZE = int(os.getenv("NEWS_BATCH_SIZE", "3"))
NO_IMPACT_MSG = "No significant price impact from today's news for this booking."


def chunk_list(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def booking_label(booking: dict[str, Any]) -> str:
    return (
        f"{booking.get('airline')} | {booking.get('origin')} → {booking.get('destination')} "
        f"| departure {booking.get('departure_date')}"
    )


def normalize_analysis(raw: dict[str, Any]) -> dict[str, Any]:
    analysis = {
        "has_price_impact": bool(raw.get("has_price_impact")),
        "impact": raw.get("impact") or raw.get("impact_summary") or raw.get("suggestion") or "",
    }
    if not analysis["has_price_impact"]:
        analysis["impact"] = ""
    return analysis


def failed_analysis(error: str) -> dict[str, Any]:
    return {"has_price_impact": False, "impact": ""}


def analyze_news_batch(
    client: Groq,
    articles: list[dict[str, Any]],
    start_index: int,
    booking: dict[str, Any],
) -> list[dict[str, Any]]:
    """Analyze up to 3 news articles for impact on a specific booking."""
    news_items = []
    for offset, article in enumerate(articles):
        news_items.append(
            {
                "index": start_index + offset,
                "title": article.get("title", ""),
                "excerpt": article.get("excerpt", ""),
            }
        )

    prompt = f"""You are an aviation pricing analyst. For EACH news article, decide if it has SIGNIFICANT price impact on THIS specific booking only.

Booking (use only these details):
- Airline: {booking.get("airline")}
- Origin: {booking.get("origin")}
- Destination: {booking.get("destination")}
- Departure date: {booking.get("departure_date")}

Rules per article:
- has_price_impact=true ONLY if this news clearly affects ticket prices for THIS airline and/or THIS route and/or departure timing.
- Military, celebrity, salary, rankings, unrelated regions/airlines → has_price_impact=false.
- Do NOT invent impacts.
- impact: one short sentence explaining price effect on THIS booking. Empty string if no impact.

News batch:
{json.dumps(news_items, indent=2)}

Return ONLY valid JSON:
{{
  "articles": [
    {{
      "index": number,
      "has_price_impact": boolean,
      "impact": "short impact on this booking, or empty string"
    }}
  ]
}}
Must return exactly {len(articles)} items.
"""

    response = groq_chat_with_retry(
        client,
        model=DEFAULT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Output only JSON. No impact unless clearly significant for the given booking.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=1200,
        response_format={"type": "json_object"},
    )
    parsed = parse_json_response(response.choices[0].message.content or "{}")
    batch_results = parsed.get("articles") or []

    analyses: list[dict[str, Any]] = []
    for offset in range(len(articles)):
        match = next(
            (item for item in batch_results if item.get("index") == start_index + offset),
            None,
        )
        if match is None and offset < len(batch_results):
            match = batch_results[offset]
        if match is None:
            analyses.append(failed_analysis("Missing batch result"))
        else:
            analyses.append(normalize_analysis(match))
    return analyses


def build_final_suggestion(
    client: Groq,
    booking: dict[str, Any],
    impacted_news: dict[str, str],
) -> str:
    if not impacted_news:
        return NO_IMPACT_MSG

    prompt = f"""Based on impacted news below, write ONE final booking suggestion (max 2 sentences) for this ticket.

Booking: {booking_label(booking)}

Impacted news:
{json.dumps(impacted_news, indent=2)}

Return ONLY valid JSON:
{{"final_suggestion": "your advice"}}
"""

    response = groq_chat_with_retry(
        client,
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=300,
        response_format={"type": "json_object"},
    )
    parsed = parse_json_response(response.choices[0].message.content or "{}")
    return parsed.get("final_suggestion") or list(impacted_news.values())[0]


def build_compact_response(
    impacted_news: dict[str, str],
    final_suggestion: str,
) -> dict[str, Any]:
    """Return final_suggestion plus news→impact map for significant items only."""
    response: dict[str, Any] = {"final_suggestion": final_suggestion}
    response.update(impacted_news)
    return response


def analyze_booking_with_news(
    booking: dict[str, Any],
    *,
    on_date: Optional[date] = None,
) -> dict[str, Any]:
    """
    Fetch all today's news → process in batches of 3 → return compact JSON.
    Only news with significant impact on the booking is included.
    """
    if on_date is None:
        on_date = datetime.now(SITE_TZ).date()

    client = get_groq_client()

    articles = fetch_all_posts(
        category=None,
        include_content=False,
        delay_seconds=0.05,
        on_date=on_date,
    )

    impacted_news: dict[str, str] = {}
    batches = chunk_list(articles, NEWS_BATCH_SIZE)

    for batch_number, batch in enumerate(batches, start=1):
        start_index = (batch_number - 1) * NEWS_BATCH_SIZE + 1

        try:
            analyses = analyze_news_batch(client, batch, start_index, booking)
        except Exception:
            analyses = [failed_analysis("batch failed") for _ in batch]

        for article, analysis in zip(batch, analyses):
            if not analysis.get("has_price_impact"):
                continue
            impact = (analysis.get("impact") or "").strip()
            if not impact or "no significant" in impact.lower():
                continue
            title = article.get("title") or "Untitled"
            impacted_news[title] = impact

    final_suggestion = build_final_suggestion(client, booking, impacted_news)
    return build_compact_response(impacted_news, final_suggestion)
