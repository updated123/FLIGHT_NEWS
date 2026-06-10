"""Booking analysis service — fetch news, process, return final suggestion."""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

from groq import Groq

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


def generate_final_suggestion(
    client: Groq,
    booking: dict[str, Any],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    impacted = [r for r in results if r["analysis"].get("has_price_impact")]
    news_digest = []
    for item in results:
        news = item["news"]
        analysis = item["analysis"]
        news_digest.append(
            {
                "title": news.get("title"),
                "has_price_impact": analysis.get("has_price_impact"),
                "impact_summary": analysis.get("impact_summary"),
                "price_direction": analysis.get("price_direction"),
                "suggestion": analysis.get("suggestion"),
            }
        )

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

    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Reason carefully, then output only the final JSON object.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=2500,
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
    Full flow: fetch all today's news → process one-by-one → final suggestion.
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

    results: list[dict[str, Any]] = []
    for article in articles:
        try:
            analysis = analyze_news_impact(client, article)
            dummy = create_dummy_booking(article, analysis)
        except Exception as exc:
            analysis = {
                "has_price_impact": False,
                "impact_summary": f"Analysis failed: {exc}",
                "affected_airlines": [],
                "affected_routes": [],
                "price_direction": "none",
                "estimated_price_change_pct": 0,
                "confidence": "low",
                "suggestion": "No significant price impact from this news.",
            }
            dummy = None

        results.append(
            {
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
