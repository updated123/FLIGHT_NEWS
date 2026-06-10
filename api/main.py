"""FastAPI server for flight booking news analysis."""

from __future__ import annotations

import os
import sys
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

load_dotenv(os.path.join(ROOT, ".env"))

from agents.service import analyze_booking_with_news  # noqa: E402

app = FastAPI(
    title="FLIGHT_NEWS API",
    description="Analyze ticket bookings against today's aviation news",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TicketBookingRequest(BaseModel):
    airline: str = Field(..., example="Delta")
    origin: str = Field(..., min_length=3, max_length=3, example="JFK")
    destination: str = Field(..., min_length=3, max_length=3, example="LAX")
    departure_date: str = Field(..., example="2026-07-15", description="YYYY-MM-DD")
    base_fare: float = Field(..., gt=0, example=420.0)
    flight_number: Optional[str] = Field(None, example="DL123")
    cabin_class: str = Field("Economy", example="Economy")
    currency: str = Field("USD", example="USD")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "flight-news-analyzer"}


@app.post("/api/analyze-booking")
def analyze_booking(request: TicketBookingRequest) -> dict:
    """
    Run complete flow for a ticket booking:
    1. Fetch today's aviation news
    2. Process each article for price impact
    3. Return final suggestion for this specific booking
    """
    try:
        booking = request.model_dump()
        return analyze_booking_with_news(booking)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
