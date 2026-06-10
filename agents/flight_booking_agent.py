#!/usr/bin/env python3
"""
LangGraph agent: fetch aviationa2z news → analyze price impact (Groq) → dummy bookings.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from datetime import date, datetime, timedelta
from typing import Any, Literal, Optional, TypedDict
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import END, START, StateGraph

from agents.groq_utils import groq_chat_with_retry

# Reuse existing news fetcher
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fetch_aviation_news import fetch_posts  # noqa: E402

load_dotenv()

SITE_TZ = ZoneInfo("Asia/Kolkata")
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")
DUMMY_PASSENGER = "Demo Passenger"


class AgentState(TypedDict):
    articles: list[dict[str, Any]]
    current_index: int
    results: list[dict[str, Any]]
    date_filter: Optional[str]
    fetched_at: Optional[str]
    article_limit: int
    news_output_path: str
    bookings_output_path: str
    summary_output_path: str


def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to .env")
    return Groq(api_key=api_key)


def parse_json_response(text: str) -> dict[str, Any]:
    text = text.strip()
    think_close = "</" + "think>"
    if think_close in text:
        text = text.split(think_close, 1)[-1].strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object in model response: {text[:200]}")
    return json.loads(match.group())


def analyze_news_impact(client: Groq, article: dict[str, Any]) -> dict[str, Any]:
    prompt = f"""You are an aviation pricing analyst. Think carefully about whether this news can realistically affect future flight ticket prices.

Rules:
- Only say has_price_impact=true if the news clearly relates to routes, airlines, airports, fuel, strikes, new flights, cancellations, regulations, or events that would change supply/demand or costs.
- Salary articles, celebrity stories, rankings/listicles, general guides, or unrelated news → has_price_impact=false.
- Do NOT invent random impacts. If unclear or negligible, set has_price_impact=false.
- affected_routes must be real IATA airport pairs like "JFK-LHR" only when inferable from the news; otherwise use an empty list.
- price_direction must be one of: increase, decrease, stable, none

Return ONLY valid JSON with this schema:
{{
  "has_price_impact": boolean,
  "impact_summary": "one or two sentences",
  "affected_airlines": ["airline names if any"],
  "affected_routes": ["ORIGIN-DEST"],
  "price_direction": "increase|decrease|stable|none",
  "estimated_price_change_pct": number or 0,
  "confidence": "low|medium|high",
  "suggestion": "actionable booking advice tied to the news, or exactly 'No significant price impact from this news.'"
}}

News title: {article.get("title", "")}
Excerpt: {article.get("excerpt", "")}
Categories: {", ".join(article.get("categories", []))}
Tags: {", ".join(article.get("tags", [])[:8])}
"""

    response = groq_chat_with_retry(
        client,
        model=DEFAULT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You reason step-by-step internally, then output only the final JSON object.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=800,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    parsed = parse_json_response(content)

    if not parsed.get("has_price_impact"):
        parsed["suggestion"] = "No significant price impact from this news."
        parsed["price_direction"] = "none"
        parsed["estimated_price_change_pct"] = 0

    return parsed


def infer_route(analysis: dict[str, Any], article: dict[str, Any]) -> tuple[str, str]:
    routes = analysis.get("affected_routes") or []
    if routes and "-" in routes[0]:
        origin, dest = routes[0].split("-", 1)
        return origin.strip().upper(), dest.strip().upper()

    tags = " ".join(article.get("tags", [])).lower()
    title = (article.get("title") or "").lower()
    blob = f"{title} {tags}"

    route_map = {
        "toronto": ("YYZ", "DEL"),
        "delta": ("ATL", "LAX"),
        "southwest": ("DAL", "LAS"),
        "air canada": ("YYZ", "YVR"),
        "indigo": ("DEL", "BOM"),
        "cathay": ("HKG", "SIN"),
        "air france": ("CDG", "JFK"),
        "china": ("PEK", "PVG"),
    }
    for key, pair in route_map.items():
        if key in blob:
            return pair
    return ("DEL", "BOM")


def infer_airline(analysis: dict[str, Any], article: dict[str, Any]) -> str:
    airlines = analysis.get("affected_airlines") or []
    if airlines:
        return str(airlines[0])
    title = article.get("title") or "Generic Airline"
    for name in (
        "Delta",
        "Southwest",
        "Air Canada",
        "IndiGo",
        "Cathay Pacific",
        "Air France",
        "American Airlines",
        "United Airlines",
    ):
        if name.lower() in title.lower():
            return name
    return "Aviation Carrier"


def create_dummy_booking(
    article: dict[str, Any],
    analysis: dict[str, Any],
) -> Optional[dict[str, Any]]:
    if not analysis.get("has_price_impact"):
        return None

    origin, destination = infer_route(analysis, article)
    airline = infer_airline(analysis, article)
    departure = (datetime.now(SITE_TZ) + timedelta(days=30)).date().isoformat()
    base_fare = 420.0
    pct = float(analysis.get("estimated_price_change_pct") or 0)
    direction = analysis.get("price_direction", "stable")

    if direction == "increase":
        suggested = round(base_fare * (1 + abs(pct) / 100), 2)
    elif direction == "decrease":
        suggested = round(base_fare * (1 - abs(pct) / 100), 2)
    else:
        suggested = base_fare

    return {
        "booking_id": f"DUMMY-{uuid.uuid4().hex[:8].upper()}",
        "status": "dummy_hold",
        "passenger_name": DUMMY_PASSENGER,
        "airline": airline,
        "flight_number": f"{airline[:2].upper()}{100 + hash(article.get('id', 0)) % 900}",
        "origin": origin,
        "destination": destination,
        "departure_date": departure,
        "cabin_class": "Economy",
        "currency": "USD",
        "base_fare": base_fare,
        "suggested_fare": suggested,
        "price_change_pct": pct,
        "price_direction": direction,
        "created_at": datetime.now(SITE_TZ).isoformat(),
        "linked_news_id": article.get("id"),
        "linked_news_title": article.get("title"),
        "linked_news_url": article.get("url"),
        "booking_reason": analysis.get("suggestion"),
    }


def fetch_news_node(state: AgentState) -> dict[str, Any]:
    on_date = None
    if state.get("date_filter"):
        on_date = date.fromisoformat(state["date_filter"])
    else:
        on_date = datetime.now(SITE_TZ).date()

    article_limit = state.get("article_limit") or 100
    print(f"Fetching news for {on_date.isoformat()} (limit {article_limit}) ...", file=sys.stderr)
    articles = fetch_posts(
        limit=article_limit,
        category=None,
        include_content=False,
        delay_seconds=0.1,
        on_date=on_date,
    )
    print(f"Fetched {len(articles)} article(s) to process.", file=sys.stderr)

    news_path = state.get("news_output_path") or "aviation_news_today.json"
    news_payload = {
        "source": "aviationa2z.com",
        "date_filter": on_date.isoformat(),
        "timezone": "Asia/Kolkata",
        "fetched_at": datetime.now(SITE_TZ).isoformat(),
        "count": len(articles),
        "articles": articles,
    }
    with open(news_path, "w", encoding="utf-8") as handle:
        json.dump(news_payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"Saved {len(articles)} news articles → {news_path}", file=sys.stderr)

    return {
        "articles": articles,
        "current_index": 0,
        "results": [],
        "date_filter": on_date.isoformat(),
        "fetched_at": datetime.now(SITE_TZ).isoformat(),
    }


def process_article_node(state: AgentState) -> dict[str, Any]:
    client = get_groq_client()
    idx = state["current_index"]
    article = state["articles"][idx]
    title = article.get("title", "Untitled")

    print(f"[{idx + 1}/{len(state['articles'])}] Analyzing: {title[:70]}...", file=sys.stderr)

    try:
        analysis = analyze_news_impact(client, article)
        booking = create_dummy_booking(article, analysis)
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
        booking = None

    result = {
        "news": {
            "id": article.get("id"),
            "title": article.get("title"),
            "url": article.get("url"),
            "published_at": article.get("published_at"),
            "excerpt": article.get("excerpt"),
            "categories": article.get("categories"),
        },
        "analysis": analysis,
        "dummy_booking": booking,
    }

    time.sleep(0.3)
    return {
        "results": state["results"] + [result],
        "current_index": idx + 1,
    }


def build_analysis_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    with_impact = [r for r in results if r["analysis"].get("has_price_impact")]
    no_impact = [r for r in results if not r["analysis"].get("has_price_impact")]

    article_summaries = []
    for index, item in enumerate(results, start=1):
        analysis = item["analysis"]
        news = item["news"]
        article_summaries.append(
            {
                "index": index,
                "news_id": news.get("id"),
                "title": news.get("title"),
                "url": news.get("url"),
                "published_at": news.get("published_at"),
                "has_price_impact": analysis.get("has_price_impact", False),
                "impact_summary": analysis.get("impact_summary"),
                "price_direction": analysis.get("price_direction"),
                "estimated_price_change_pct": analysis.get("estimated_price_change_pct"),
                "confidence": analysis.get("confidence"),
                "affected_airlines": analysis.get("affected_airlines", []),
                "affected_routes": analysis.get("affected_routes", []),
                "suggestion": analysis.get("suggestion"),
                "dummy_booking_created": item.get("dummy_booking") is not None,
            }
        )

    return {
        "articles_processed": len(results),
        "with_price_impact": len(with_impact),
        "no_price_impact": len(no_impact),
        "dummy_bookings_created": sum(1 for r in results if r.get("dummy_booking")),
        "headlines_with_impact": [r["news"]["title"] for r in with_impact],
        "headlines_no_impact": [r["news"]["title"] for r in no_impact],
        "article_analyses": article_summaries,
    }


def save_results_node(state: AgentState) -> dict[str, Any]:
    bookings_path = state.get("bookings_output_path") or "dummy_bookings_today.json"
    summary_path = state.get("summary_output_path") or "analysis_summary_today.json"

    bookings = [r["dummy_booking"] for r in state["results"] if r.get("dummy_booking")]
    summary_body = build_analysis_summary(state["results"])
    processed_at = datetime.now(SITE_TZ).isoformat()

    bookings_payload = {
        "source": "aviationa2z.com",
        "agent": "langgraph-flight-booking",
        "type": "dummy_bookings",
        "date_filter": state.get("date_filter"),
        "fetched_at": state.get("fetched_at"),
        "processed_at": processed_at,
        "count": len(bookings),
        "bookings": bookings,
    }

    summary_payload = {
        "source": "aviationa2z.com",
        "agent": "langgraph-flight-booking",
        "type": "analysis_summary",
        "model": DEFAULT_MODEL,
        "date_filter": state.get("date_filter"),
        "fetched_at": state.get("fetched_at"),
        "processed_at": processed_at,
        **summary_body,
    }

    with open(bookings_path, "w", encoding="utf-8") as handle:
        json.dump(bookings_payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"Saved {len(bookings)} dummy bookings → {bookings_path}", file=sys.stderr)
    print(
        f"Saved analysis summary ({summary_body['articles_processed']} articles) → {summary_path}",
        file=sys.stderr,
    )
    return {}


def route_after_process(state: AgentState) -> Literal["process_article", "save_results"]:
    if state["current_index"] < len(state["articles"]):
        return "process_article"
    return "save_results"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("fetch_news", fetch_news_node)
    graph.add_node("process_article", process_article_node)
    graph.add_node("save_results", save_results_node)

    graph.add_edge(START, "fetch_news")
    graph.add_edge("fetch_news", "process_article")
    graph.add_conditional_edges("process_article", route_after_process)
    graph.add_edge("save_results", END)
    return graph.compile()


def main() -> int:
    parser = argparse.ArgumentParser(description="LangGraph agent: news → price impact → dummy bookings")
    parser.add_argument("--today", action="store_true", help="Process today's news (IST)")
    parser.add_argument("--date", help="Process news for YYYY-MM-DD (IST)")
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Number of news articles to fetch and process (default: 1)",
    )
    parser.add_argument(
        "--news-output",
        default="aviation_news_today.json",
        help="Raw fetched news JSON (default: aviation_news_today.json)",
    )
    parser.add_argument(
        "--bookings-output",
        default="dummy_bookings_today.json",
        help="Dummy bookings JSON (default: dummy_bookings_today.json)",
    )
    parser.add_argument(
        "--summary-output",
        default="analysis_summary_today.json",
        help="Post-analysis summary JSON (default: analysis_summary_today.json)",
    )
    args = parser.parse_args()

    date_filter = None
    if args.date:
        date_filter = args.date
    elif args.today or True:
        date_filter = datetime.now(SITE_TZ).date().isoformat()

    app = build_graph()
    app.invoke(
        {
            "articles": [],
            "current_index": 0,
            "results": [],
            "date_filter": date_filter,
            "fetched_at": None,
            "article_limit": args.limit,
            "news_output_path": args.news_output,
            "bookings_output_path": args.bookings_output,
            "summary_output_path": args.summary_output,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
