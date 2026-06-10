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

## API endpoint

Start the server:

```bash
uvicorn api.main:app --reload --port 8000
```

**POST** `/api/analyze-booking` — pass ticket details, get full analysis + final suggestion:

```bash
curl -X POST http://localhost:8000/api/analyze-booking \
  -H "Content-Type: application/json" \
  -d '{
    "airline": "Emirates",
    "origin": "DXB",
    "destination": "FRA",
    "departure_date": "2026-07-20"
  }'
```

Fetches **all** of today's news, processes in batches of 3, returns only **significant impacts** for your booking.

Example response:
```json
{
  "final_suggestion": "Book DXB-FRA soon; reduced Emirates A380 capacity may push fares up.",
  "Emirates Will No Longer Fly A380 to Germany": "Reduced capacity on Germany routes may increase Emirates fares on DXB-FRA."
}
```

If no significant impact:
```json
{
  "final_suggestion": "No significant price impact from today's news for this booking."
}
```

## LangGraph CLI flow

```
START → fetch_news → process_article (loop) → save_results → END
```

## Environment

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Groq API key |
| `GROQ_MODEL` | Default: `qwen/qwen3-32b` |

## Deploy on Render

### 1. Push code to GitHub
Repo: https://github.com/updated123/FLIGHT_NEWS

### 2. Create Render Web Service
1. Go to [render.com](https://render.com) → **New** → **Web Service**
2. Connect GitHub account → select **updated123/FLIGHT_NEWS**
3. Render auto-detects `render.yaml` (or configure manually below)

### 3. Manual settings (if not using Blueprint)
| Setting | Value |
|---------|-------|
| **Runtime** | Python 3 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn api.main:app --host 0.0.0.0 --port $PORT` |
| **Health Check Path** | `/health` |

### 4. Environment variables (Render Dashboard → Environment)
| Key | Value |
|-----|-------|
| `GROQ_API_KEY` | your Groq API key |
| `GROQ_MODEL` | `qwen/qwen3-32b` |

### 5. Deploy
Click **Create Web Service** → Render builds and deploys automatically.

### 6. Test live API
```bash
curl -X POST https://YOUR-APP.onrender.com/api/analyze-booking \
  -H "Content-Type: application/json" \
  -d '{
    "airline": "Emirates",
    "origin": "DXB",
    "destination": "FRA",
    "departure_date": "2026-07-20"
  }'
```

Docs UI: `https://YOUR-APP.onrender.com/docs`
