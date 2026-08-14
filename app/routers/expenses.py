"""Expense Extractor endpoints — upload a receipt/invoice image, Gemini
reads it directly (no separate OCR), user reviews before it's saved."""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional

from app.services import gemini_client
from app.services.auth import get_current_user, get_user_organization_id
from app.services.supabase_client import get_supabase_admin_client

router = APIRouter(prefix="/api/expenses", tags=["expenses"])

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.post("/parse")
async def parse_receipt(file: UploadFile = File(...), user=Depends(get_current_user)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type {file.content_type}. Use JPG, PNG, or WEBP.")

    image_bytes = await file.read()
    if len(image_bytes) > 15 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 15MB)")

    try:
        parsed = gemini_client.parse_receipt_image(image_bytes, file.content_type)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not read the receipt: {e}")

    return parsed


class ExpenseRecord(BaseModel):
    vendor: Optional[str] = None
    date: Optional[str] = None
    amount: Optional[float] = None
    category: Optional[str] = None
    currency: Optional[str] = "USD"


@router.post("")
async def save_expense(payload: ExpenseRecord, user=Depends(get_current_user)):
    """Saves after the user has reviewed the parsed fields on the
    frontend — same pattern as FinGuard's statement import, where
    Gemini's read is a draft, not an automatic commit."""
    org_id = await get_user_organization_id(user.id, user.email)
    supabase = get_supabase_admin_client()

    saved = supabase.table("expense_records").insert({
        "organization_id": org_id,
        "uploaded_by": user.id,
        "vendor": payload.vendor,
        "expense_date": payload.date,
        "amount": payload.amount,
        "category": payload.category,
        "currency": payload.currency or "USD",
    }).execute()

    return {"record": saved.data[0] if saved.data else None}


@router.get("")
async def list_expenses(user=Depends(get_current_user)):
    org_id = await get_user_organization_id(user.id, user.email)
    supabase = get_supabase_admin_client()
    result = (
        supabase.table("expense_records")
        .select("*")
        .eq("organization_id", org_id)
        .order("created_at", desc=True)
        .limit(100)
        .execute()
    )
    return {"records": result.data}
