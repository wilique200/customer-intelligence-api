"""Data Analyst endpoints — auth-protected, persisted to Supabase.
The chat endpoint now fetches its context from a saved analysis by ID
rather than requiring the frontend to resend the full summary each
time, since persistence makes that possible."""

import pandas as pd
from io import StringIO
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from pydantic import BaseModel

from app.services import data_analyst as da
from app.services.auth import get_current_user, get_user_organization_id
from app.services.supabase_client import get_supabase_admin_client

router = APIRouter(prefix="/api/data-analyst", tags=["data-analyst"])


@router.post("/analyze")
async def analyze_data(file: UploadFile = File(...), user=Depends(get_current_user)):
    contents = await file.read()
    try:
        df = pd.read_csv(StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")
    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded file has no data.")

    summary, forecast = da.analyze_dataset(df)
    org_id = await get_user_organization_id(user.id, user.email)

    supabase = get_supabase_admin_client()
    saved = supabase.table("data_analyst_analyses").insert({
        "organization_id": org_id,
        "uploaded_by": user.id,
        "filename": file.filename,
        "summary": summary,
        "forecast": forecast,
    }).execute()

    analysis_id = saved.data[0]["id"] if saved.data else None
    return {"analysis_id": analysis_id, "summary": summary, "forecast": forecast}


class ChatRequest(BaseModel):
    analysis_id: str
    question: str


@router.post("/chat")
async def chat_with_data(payload: ChatRequest, user=Depends(get_current_user)):
    supabase = get_supabase_admin_client()
    org_id = await get_user_organization_id(user.id, user.email)

    analysis = (
        supabase.table("data_analyst_analyses")
        .select("*")
        .eq("id", payload.analysis_id)
        .eq("organization_id", org_id)  # belt-and-suspenders — service key bypasses RLS, so filter explicitly
        .single()
        .execute()
    )
    if not analysis.data:
        raise HTTPException(status_code=404, detail="Analysis not found.")

    try:
        answer = da.ask_gemini_about_data(payload.question, analysis.data["summary"], analysis.data.get("forecast"))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Chat request failed: {e}")

    supabase.table("data_analyst_chat_messages").insert([
        {"analysis_id": payload.analysis_id, "organization_id": org_id, "role": "user", "content": payload.question},
        {"analysis_id": payload.analysis_id, "organization_id": org_id, "role": "assistant", "content": answer},
    ]).execute()

    return {"answer": answer}


@router.post("/predict")
async def predict_column(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    user=Depends(get_current_user),
):
    """Trains a quick model on a user-chosen column. Re-accepts the file
    rather than reloading from Supabase, since only the EDA summary is
    persisted, not the raw data — the frontend resends the same File
    object it already has in memory from the initial analyze call, so
    this is invisible to the user (no second upload prompt)."""
    contents = await file.read()
    try:
        df = pd.read_csv(StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    try:
        result = da.train_quick_model(df, target_column)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model training failed: {e}")

    return result
