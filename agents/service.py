"""Booking analysis service — fetch news, process, return final suggestion."""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

from groq import Groq

from agents.groq_utils import groq_chat_with_retry, throttle_between_calls
from agents.flight_booking_agent import (
    DEFAULT_MODEL,
    SITE_TZ,
    analyze_news_impact,
    build_analysis_summary,
    create_dummy_booking,
    get_groq_client,
    parse_json_response,
)
from fetch_aviation_news import fetch_all_posts

SITE_TZ_NAME = "Asia/Kolkata"
NEWS_BATCH_SIZE = int(os.getenv("NEWS_BATCH_SIZE", "3"))


def chunk_list(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def normalize_analysis(raw: dict[str, Any]) -> dict[str, Any]:
    analysis = {
        "has_price_impact": bool(raw.get("has_price_impact")),
        "impact_summary": raw.get("impact_summary", ""),
        "affected_airlines": raw.get("affected_airlines") or [],
        "affected_routes": raw.get("affected_routes") or [],
        "price_direction": raw.get("price_direction", "none"),
        "estimated_price_change_pct": float(raw.get("estimated_price_change_pct") or 0),
        "confidence": raw.get("confidence", "low"),
        "suggestion": raw.get("suggestion", ""),
    }
    if not analysis["has_price_impact"]:
        analysis["suggestion"] = "No significant price impact from this news."
        analysis["price_direction"] = "none"
        analysis["estimated_price_change_pct"] = 0
    return analysis


def failed_analysis(error: str) -> dict[str, Any]:
    return normalize_analysis(
        {
            "has_price_impact": False,
            "impact_summary": f"Analysis failed: {error}",
            "suggestion": "No significant price impact from this news.",
            "confidence": "low",
        }
    )


def analyze_news_batch(
    client: Groq,
    articles: list[dict[str, Any]],
    start_index: int,
) -> list[dict[str, Any]]:
    """Analyze up to 3 news articles in a single Groq call."""
    if len(articles) == 1:
        return [analyze_news_impact(client, articles[0])]

    news_items = []
    for offset, article in enumerate(articles):
        news_items.append(
            {
                "index": start_index + offset,
                "title": article.get("title", ""),
                "excerpt": article.get("excerpt", ""),
                "categories": article.get("categories", []),
                "tags": (article.get("tags") or [])[:8],
            }
        )

    prompt = f"""You are an aviation pricing analyst. Analyze EACH news article below for flight ticket price impact.

Rules per article:
- has_price_impact=true only for routes, airlines, airports, fuel, strikes, new/cancelled flights, regulations, supply/demand changes.
- Salary, celebrity, rankings, guides, military-only news → has_price_impact=false.
- Do NOT invent random impacts.
- affected_routes: IATA pairs like "DXB-FRA" only when inferable, else [].
- price_direction: increase|decrease|stable|none

News batch:
{json.dumps(news_items, indent=2)}

Return ONLY valid JSON:
{{
  "articles": [
    {{
      "index": number,
      "has_price_impact": boolean,
      "impact_summary": "one or two sentences",
      "affected_airlines": [],
      "affected_routes": [],
      "price_direction": "increase|decrease|stable|none",
      "estimated_price_change_pct": number,
      "confidence": "low|medium|high",
      "suggestion": "actionable advice or 'No significant price impact from this news.'"
    }}
  ]
}}
Must return exactly {len(articles)} items in the same order as input indices.
"""

    response = groq_chat_with_retry(
        client,
        model=DEFAULT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Analyze each article separately. Output only the final JSON object.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=1500,
        response_format={"type": "json_object"},
    )
    parsed = parse_json_response(response.choices[0].message.content or "{}")
    batch_results = parsed.get("articles") or []

    analyses: list[dict[str, Any]] = []
    for offset, article in enumerate(articles):
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


def generate_final_suggestion(
    client: Groq,
    booking: dict[str, Any],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    impacted = [r for r in results if r["analysis"].get("has_price_impact")]
    airline = (booking.get("airline") or "").lower()
    origin = (booking.get("origin") or "").upper()
    destination = (booking.get("destination") or "").upper()
    route_key = f"{origin}-{destination}"

    news_digest = []
    for item in results:
        news = item["news"]
        analysis = item["analysis"]
        title = (news.get("title") or "").lower()
        excerpt = (news.get("excerpt") or "").lower()
        routes = analysis.get("affected_routes") or []
        airlines = [a.lower() for a in analysis.get("affected_airlines") or []]

        relevant = (
            analysis.get("has_price_impact")
            or (airline and airline in title)
            or (airline and airline in excerpt)
            or (airline and airline in airlines)
            or route_key in [r.upper() for r in routes]
            or origin.lower() in title
            or destination.lower() in title
        )
        if not relevant:
            continue

        news_digest.append(
            {
                "title": news.get("title"),
                "has_price_impact": analysis.get("has_price_impact"),
                "impact_summary": analysis.get("impact_summary"),
                "price_direction": analysis.get("price_direction"),
                "suggestion": analysis.get("suggestion"),
            }
        )

    if not news_digest:
        news_digest = [
            {
                "title": item["news"].get("title"),
                "has_price_impact": item["analysis"].get("has_price_impact"),
                "impact_summary": item["analysis"].get("impact_summary"),
                "price_direction": item["analysis"].get("price_direction"),
                "suggestion": item["analysis"].get("suggestion"),
            }
            for item in results[:8]
        ]

    prompt = f"""You are an aviation pricing advisor. Based on today's news and this specific ticket booking, give a final recommendation.

Rules:
- Only reference news that realistically affects THIS booking (airline, route, dates, region).
- If no news affects this booking, say clearly: "No significant price impact from today's news for this booking."
- Do NOT invent impacts or give random advice.
- Be specific to the booking details provided.

Ticket booking:
- Airline: {booking.get("airline")}
- Flight: {booking.get("flight_number", "N/A")}
- Route: {booking.get("origin")} → {booking.get("destination")}
- Departure: {booking.get("departure_date")}
- Cabin: {booking.get("cabin_class", "Economy")}
- Current fare: {booking.get("base_fare")} {booking.get("currency", "USD")}

Today's news analyses ({len(results)} articles, {len(impacted)} with potential impact):
{json.dumps(news_digest, indent=2)}

Return ONLY valid JSON:
{{
  "has_price_impact": boolean,
  "final_suggestion": "clear actionable advice for this specific booking",
  "recommended_action": "book_now|wait|monitor|no_action",
  "price_outlook": "increase|decrease|stable|none",
  "estimated_price_change_pct": number,
  "confidence": "low|medium|high",
  "relevant_headlines": ["only headlines that matter for this booking"],
  "reasoning": "brief explanation tied to news and booking"
}}
"""

    response = groq_chat_with_retry(
        client,
        model=DEFAULT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Reason carefully, then output only the final JSON object.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=1000,
        response_format={"type": "json_object"},
    )
    parsed = parse_json_response(response.choices[0].message.content or "{}")

    if not parsed.get("has_price_impact"):
        parsed["final_suggestion"] = (
            parsed.get("final_suggestion")
            or "No significant price impact from today's news for this booking."
        )
        parsed["recommended_action"] = parsed.get("recommended_action") or "no_action"
        parsed["price_outlook"] = "none"
        parsed["estimated_price_change_pct"] = 0

    return parsed


def analyze_booking_with_news(
    booking: dict[str, Any],
    *,
    on_date: Optional[date] = None,
) -> dict[str, Any]:
    """
    Full flow: fetch all today's news → process in batches of 3 → final suggestion.
    Returns API-ready response (no file writes).
    """
    if on_date is None:
        on_date = datetime.now(SITE_TZ).date()

    client = get_groq_client()
    fetched_at = datetime.now(SITE_TZ).isoformat()

    articles = fetch_all_posts(
        category=None,
        include_content=False,
        delay_seconds=0.05,
        on_date=on_date,
    )

    batches = chunk_list(articles, NEWS_BATCH_SIZE)
    results: list[dict[str, Any]] = []

    for batch_number, batch in enumerate(batches, start=1):
        if batch_number > 1:
            throttle_between_calls()

        start_index = (batch_number - 1) * NEWS_BATCH_SIZE + 1
        end_index = start_index + len(batch) - 1

        try:
            analyses = analyze_news_batch(client, batch, start_index)
        except Exception as exc:
            analyses = []
            for article in batch:
                try:
                    analyses.append(analyze_news_impact(client, article))
                except Exception as inner_exc:
                    analyses.append(failed_analysis(str(inner_exc)))

        for article, analysis in zip(batch, analyses):
            dummy = create_dummy_booking(article, analysis)
            results.append(
                {
                    "batch": batch_number,
                    "batch_range": f"{start_index}-{end_index}",
                    "news": {
                        "id": article.get("id"),
                        "title": article.get("title"),
                        "url": article.get("url"),
                        "published_at": article.get("published_at"),
                        "excerpt": article.get("excerpt"),
                        "categories": article.get("categories"),
                    },
                    "analysis": analysis,
                    "dummy_booking": dummy,
                }
            )

    summary = build_analysis_summary(results)
    final = generate_final_suggestion(client, booking, results)
    processed_at = datetime.now(SITE_TZ).isoformat()

    suggested_fare = booking.get("base_fare", 0)
    pct = float(final.get("estimated_price_change_pct") or 0)
    outlook = final.get("price_outlook", "none")
    if outlook == "increase":
        suggested_fare = round(float(booking.get("base_fare", 0)) * (1 + abs(pct) / 100), 2)
    elif outlook == "decrease":
        suggested_fare = round(float(booking.get("base_fare", 0)) * (1 - abs(pct) / 100), 2)

    return {
        "status": "ok",
        "model": DEFAULT_MODEL,
        "date_filter": on_date.isoformat(),
        "timezone": SITE_TZ_NAME,
        "fetched_at": fetched_at,
        "processed_at": processed_at,
        "booking": booking,
        "news_count": len(articles),
        "batch_size": NEWS_BATCH_SIZE,
        "batches_processed": len(batches),
        "summary": summary,
        "final_suggestion": final,
        "pricing": {
            "current_fare": booking.get("base_fare"),
            "suggested_fare": suggested_fare,
            "currency": booking.get("currency", "USD"),
            "price_outlook": outlook,
            "estimated_price_change_pct": pct,
        },
        "article_analyses": summary.get("article_analyses", []),
    }
