"""Fetch past analyses for the logged-in user's organization — powers
a "History" view in the frontend so results persist across sessions."""

from fastapi import APIRouter, Depends
from app.services.auth import get_current_user, get_user_organization_id
from app.services.supabase_client import get_supabase_admin_client

router = APIRouter(prefix="/api/history", tags=["history"])


@router.get("/churn")
async def churn_history(user=Depends(get_current_user)):
    org_id = await get_user_organization_id(user.id, user.email)
    supabase = get_supabase_admin_client()
    result = (
        supabase.table("churn_analyses")
        .select("id, filename, total_customers, flagged_count, threshold, created_at")
        .eq("organization_id", org_id)
        .order("created_at", desc=True)
        .limit(50)
        .execute()
    )
    return {"analyses": result.data}


@router.get("/data-analyst")
async def data_analyst_history(user=Depends(get_current_user)):
    org_id = await get_user_organization_id(user.id, user.email)
    supabase = get_supabase_admin_client()
    result = (
        supabase.table("data_analyst_analyses")
        .select("id, filename, created_at")
        .eq("organization_id", org_id)
        .order("created_at", desc=True)
        .limit(50)
        .execute()
    )
    return {"analyses": result.data}
