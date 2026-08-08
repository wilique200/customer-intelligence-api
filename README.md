# Pulse — Customer Intelligence Backend

FastAPI backend serving the churn model and the Data Analyst module,
with Supabase auth and per-organization data persistence.

## Structure

```
app/
  main.py              - app setup, CORS, router registration
  routers/
    churn.py            - POST /api/churn/predict
    data_analyst.py      - POST /api/data-analyst/analyze, /chat
    history.py            - GET /api/history/churn, /data-analyst
    status.py              - health check + coming-soon stubs
  services/
    ml_models.py          - churn model loading, preprocessing, prediction
    data_analyst.py       - auto-EDA, forecasting, Gemini chat
    auth.py                - Supabase token verification
    supabase_client.py     - shared Supabase client (service role)
supabase/
  schema.sql               - full database schema, run this first
```

## Setup

1. **Supabase project**: create one at supabase.com (free tier). In the
   SQL Editor, run `supabase/schema.sql` in full.
2. **Test the signup trigger immediately** — sign up one test user
   (Authentication tab -> Add user, or via the frontend once it's
   running) and confirm a row appears in both `organizations` and
   `organization_members`. This exact trigger is the single most common
   thing to need a fix on a fresh setup — check it before testing
   anything else.
3. **Environment variables** — copy `.env.example` to `.env` for local
   dev, or set these on Render:
   - `HF_REPO_ID` — already set to the churn model repo
   - `GEMINI_API_KEY` — free key from aistudio.google.com
   - `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` — Project Settings -> API
     in Supabase (service role key, not anon key)
   - `ALLOWED_ORIGIN` — your Vercel frontend URL, once it exists

## Deploying to Render

Use the Blueprint flow with `render.yaml` (already configured), or
connect the repo manually with:
- Build: `pip install -r requirements.txt`
- Start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Health check path: `/api/health`

## Local development

```
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Test at `http://localhost:8000/docs` — every endpoint except `/api/health`
and the `*/status` stubs now requires a Supabase access token
(`Authorization: Bearer <token>`), so you'll need a logged-in session
from the frontend to get a real token to test with.
