"""
Supabase client — single shared instance, using the SERVICE ROLE key
(not the anon key) because the backend needs to read/write across
whatever organization a verified user belongs to, after auth.py has
already confirmed who they are. The service key bypasses RLS, which is
why every query in this codebase that touches user data must explicitly
filter by organization_id -- RLS is not doing that filtering for us on
this key, only on the frontend's anon-key client.
"""

import os
from supabase import create_client, Client

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")

print(f"[startup diagnostic] SUPABASE_URL: set={bool(SUPABASE_URL)}, length={len(SUPABASE_URL)}, starts_with_https={SUPABASE_URL.startswith('https://')}")
print(f"[startup diagnostic] SUPABASE_SERVICE_KEY: set={bool(SUPABASE_SERVICE_KEY)}, length={len(SUPABASE_SERVICE_KEY)}, starts_with_eyJ={SUPABASE_SERVICE_KEY.startswith('eyJ')}")

_client: Client = None


def get_supabase_admin_client() -> Client:
    global _client
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set as environment variables"
            )
        _client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return _client
