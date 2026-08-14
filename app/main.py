"""
Customer Intelligence API — app entry point.

Run locally: uvicorn app.main:app --reload
Render start command: uvicorn app.main:app --host 0.0.0.0 --port $PORT
(note the app.main:app path — this changed from the old flat main.py:app
now that the code is split into app/routers and app/services)
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.services import ml_models
from app.routers import churn, data_analyst, history, status, expenses


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models.load_churn_models()
    yield


app = FastAPI(title="Customer Intelligence API", lifespan=lifespan)

# Comma-separated if you ever need more than one (e.g. a Vercel preview
# URL alongside production). Defaults to "*" until locked to a real
# frontend URL.
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGIN", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(status.router)
app.include_router(churn.router)
app.include_router(data_analyst.router)
app.include_router(history.router)
app.include_router(expenses.router)
