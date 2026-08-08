"""Churn Risk endpoint — now auth-protected and persisted to Supabase."""

import pandas as pd
from io import StringIO
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends

from app.services import ml_models
from app.services.auth import get_current_user, get_user_organization_id
from app.services.supabase_client import get_supabase_admin_client

router = APIRouter(prefix="/api/churn", tags=["churn"])


@router.post("/predict")
async def predict_churn(file: UploadFile = File(...), user=Depends(get_current_user)):
    if not ml_models.models_loaded():
        raise HTTPException(status_code=503, detail="Churn model not loaded — check HF_REPO_ID on the server.")

    contents = await file.read()
    try:
        raw_df = pd.read_csv(StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    results, threshold, warning = ml_models.run_churn_prediction(raw_df)
    org_id = await get_user_organization_id(user.id)

    supabase = get_supabase_admin_client()
    saved = supabase.table("churn_analyses").insert({
        "organization_id": org_id,
        "uploaded_by": user.id,
        "filename": file.filename,
        "total_customers": len(results),
        "flagged_count": sum(r["flagged"] for r in results),
        "threshold": threshold,
        "data_completeness_warning": warning,
        "results": results,
    }).execute()

    analysis_id = saved.data[0]["id"] if saved.data else None

    return {
        "analysis_id": analysis_id,
        "threshold": threshold,
        "total_customers": len(results),
        "flagged_count": sum(r["flagged"] for r in results),
        "data_completeness_warning": warning,
        "results": results,
    }
