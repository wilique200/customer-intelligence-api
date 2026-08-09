"""
Auth dependency for protected routes.

Verifies the Supabase access token by asking Supabase's own auth server
to validate it (via supabase.auth.get_user), rather than manually
checking the JWT signature ourselves. Slightly more network overhead
per request, but avoids this code breaking if Supabase ever changes
their token signing scheme -- correctness over micro-optimization here.
"""

from fastapi import Header, HTTPException
from app.services.supabase_client import get_supabase_admin_client


async def get_current_user(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = authorization.removeprefix("Bearer ").strip()
    supabase = get_supabase_admin_client()

    try:
        user_response = supabase.auth.get_user(token)
        user = user_response.user
        if not user:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        return user
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


async def get_user_organization_id(user_id: str, user_email: str = "") -> str:
    """Every authenticated user belongs to exactly one organization for
    now (the one auto-created at signup). Multi-org membership would
    extend this, not replace it."""
    supabase = get_supabase_admin_client()
    result = (
        supabase.table("organization_members")
        .select("organization_id")
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    print(f"[org lookup] user_id={user_id} email={user_email} rows_found={len(result.data)}")
    if not result.data:
        raise HTTPException(
            status_code=404,
            detail=f"No organization found for user_id={user_id} (email={user_email}). "
                   f"Check this exact user_id exists in organization_members in Supabase.",
        )
    return result.data[0]["organization_id"]
