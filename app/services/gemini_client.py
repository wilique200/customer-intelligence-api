"""
Gemini client — using the official google-generativeai SDK, matching
FinGuard's proven working setup, rather than raw REST calls. The REST
endpoint doesn't accept the newer "AQ." key format Google AI Studio now
issues on some accounts; the SDK handles it correctly.
"""

import json
import logging
import os

import google.generativeai as genai

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")  # matches FinGuard's current default

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def _clean_json_response(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return cleaned.strip()


def map_columns(unmatched_uploaded: list, still_missing_expected: list) -> dict:
    """Semantic column-name matching for whatever the cheap deterministic
    pass in ml_models.py couldn't resolve. Only ever sees column names,
    never actual customer data."""
    if not GEMINI_API_KEY or not unmatched_uploaded or not still_missing_expected:
        return {}

    prompt = f"""You are matching CSV column names to a fixed schema for a churn prediction model.
Uploaded columns with no obvious match: {unmatched_uploaded}
Schema columns still needing a match: {still_missing_expected}

Return ONLY a JSON object mapping uploaded column name -> schema column name,
including only genuinely confident semantic matches (e.g. "Bill Amount" ->
"monthlycharges" is confident; do not guess when unsure). Omit any uploaded
column with no confident match. Return just the JSON object, nothing else."""

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            prompt, generation_config={"response_mime_type": "application/json"}
        )
        mapping = json.loads(_clean_json_response(response.text))
        return {k: v for k, v in mapping.items() if k in unmatched_uploaded and v in still_missing_expected}
    except Exception:
        logger.exception("Gemini column mapping failed, continuing without it")
        return {}


def answer_data_question(question: str, summary: dict, forecast: dict) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY isn't set on the server.")

    context = json.dumps({"data_summary": summary, "forecast": forecast}, default=str)
    prompt = f"""You are a data analyst assistant. Answer the user's question using ONLY the
dataset summary below. If the summary doesn't contain enough information to answer
confidently, say so honestly rather than guessing.

Dataset summary:
{context}

Question: {question}

Give a concise, direct answer (2-4 sentences), referencing specific numbers from the
summary where relevant."""

    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)
    return response.text.strip()
