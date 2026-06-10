# FLIGHT_NEWS

LangGraph agent that fetches aviation news from [aviationa2z.com](https://aviationa2z.com), analyzes price impact with Groq, and creates dummy flight bookings when relevant.

## Features

- Fetch today's aviation news (IST timezone)
- Process articles one-by-one with Groq `qwen/qwen3-32b`
- Separate output files:
  - `aviation_news_today.json` — raw fetched news
  - `dummy_bookings_today.json` — dummy bookings only
  - `analysis_summary_today.json` — post-analysis summary

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your GROQ_API_KEY to .env
```

## Usage

```bash
# Test with 1 article (default)
python agents/flight_booking_agent.py --today --limit 1

# Process all today's news
python agents/flight_booking_agent.py --today --limit 100

# Custom output paths
python agents/flight_booking_agent.py --today \
  --news-output aviation_news_today.json \
  --bookings-output dummy_bookings_today.json \
  --summary-output analysis_summary_today.json
```

## LangGraph flow

```
START → fetch_news → process_article (loop) → save_results → END
```

## Environment

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Groq API key |
| `GROQ_MODEL` | Default: `qwen/qwen3-32b` |
