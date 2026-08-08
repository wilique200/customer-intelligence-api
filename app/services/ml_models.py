"""
Churn model loading, preprocessing, and prediction logic. Ported as-is
from the earlier single-file main.py -- every fix from live testing
(column normalization, numeric coercion, the Gemini fallback matcher,
the context-aware explain_risk fallback) is preserved exactly, just
relocated so main.py doesn't have to hold all of this directly.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import requests
from huggingface_hub import hf_hub_download

HF_REPO_ID = os.environ.get("HF_REPO_ID", "your-hf-username/customer-intelligence-churn-model")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
# Model naming on the free tier shifts fairly often — if this 404s, check
# the current model list at aistudio.google.com and swap the name below.
GEMINI_MODEL = "gemini-2.0-flash-lite"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

MODEL_FILES = [
    "final_catboost_model.pkl", "final_lightgbm_model.pkl", "final_lr_model.pkl",
    "final_scaler.pkl", "final_meta_model.pkl", "final_feature_columns.pkl",
    "final_decision_threshold.pkl",
]

_models = {}


def load_churn_models():
    try:
        local_paths = {f: hf_hub_download(repo_id=HF_REPO_ID, filename=f) for f in MODEL_FILES}
        _models["catboost"] = joblib.load(local_paths["final_catboost_model.pkl"])
        _models["lightgbm"] = joblib.load(local_paths["final_lightgbm_model.pkl"])
        _models["lr"] = joblib.load(local_paths["final_lr_model.pkl"])
        _models["scaler"] = joblib.load(local_paths["final_scaler.pkl"])
        _models["meta"] = joblib.load(local_paths["final_meta_model.pkl"])
        _models["feature_columns"] = joblib.load(local_paths["final_feature_columns.pkl"])
        _models["threshold"] = joblib.load(local_paths["final_decision_threshold.pkl"])
        print(f"Loaded churn model bundle from HF Hub: {HF_REPO_ID}")
    except Exception as e:
        print(f"WARNING: failed to load models from HF Hub ({HF_REPO_ID}) — churn predictions will fail until this is fixed. {e}")


def models_loaded() -> bool:
    return "catboost" in _models


RAW_COLUMN_DEFAULTS = {
    "age": 45, "gender": "Male", "annual_income": 48954.6, "education": "bachelor",
    "marital_status": "married", "dependents": 1, "tenure": 16, "contract": "one_year",
    "payment_method": "credit_card", "paperless_billing": "Yes", "senior_citizen": 0,
    "monthlycharges": 85.0, "totalcharges": 1250.0, "num_services": 2,
    "has_phone_service": 1, "has_internet_service": 1, "has_online_security": 0,
    "has_online_backup": 0, "has_device_protection": 0, "has_tech_support": 0,
    "has_streaming_tv": 1, "has_streaming_movies": 1, "customer_satisfaction": 7.0,
    "num_complaints": 1.0, "num_service_calls": 2, "late_payments": 0,
    "avg_monthly_gb": 27.77, "days_since_last_interaction": 31, "credit_score": 680.0,
}

NUMERIC_COLUMNS = {
    "age", "annual_income", "dependents", "tenure", "senior_citizen",
    "monthlycharges", "totalcharges", "num_services", "has_phone_service",
    "has_internet_service", "has_online_security", "has_online_backup",
    "has_device_protection", "has_tech_support", "has_streaming_tv",
    "has_streaming_movies", "customer_satisfaction", "num_complaints",
    "num_service_calls", "late_payments", "avg_monthly_gb",
    "days_since_last_interaction", "credit_score",
}


def normalize_columns(df: pd.DataFrame):
    normalized_expected = {c.lower().replace(" ", "").replace("_", ""): c for c in RAW_COLUMN_DEFAULTS}
    rename_map = {}
    for col in df.columns:
        key = col.lower().replace(" ", "").replace("_", "")
        if key in normalized_expected:
            rename_map[col] = normalized_expected[key]
    df = df.rename(columns=rename_map)
    matched_expected = set(rename_map.values())
    unmatched_uploaded = [c for c in df.columns if c not in RAW_COLUMN_DEFAULTS]
    still_missing_expected = [c for c in RAW_COLUMN_DEFAULTS if c not in matched_expected and c not in df.columns]
    return df, unmatched_uploaded, still_missing_expected


def llm_map_columns(unmatched_uploaded: list, still_missing_expected: list) -> dict:
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
        resp = requests.post(
            GEMINI_URL,
            params={"key": GEMINI_API_KEY},
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=10,
        )
        resp.raise_for_status()
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        for fence in ("```json", "```"):
            if text.startswith(fence):
                text = text[len(fence):]
            if text.endswith("```"):
                text = text[:-3]
        mapping = json.loads(text.strip())
        return {k: v for k, v in mapping.items() if k in unmatched_uploaded and v in still_missing_expected}
    except Exception as e:
        print(f"WARNING: Gemini column mapping failed, continuing without it: {e}")
        return {}


def fill_missing_columns(df: pd.DataFrame):
    df = df.copy()
    defaulted = []
    for col, default_val in RAW_COLUMN_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default_val
            defaulted.append(col)
    return df, defaulted


def coerce_numeric_columns(df: pd.DataFrame):
    df = df.copy()
    coerced = []
    for col in NUMERIC_COLUMNS:
        if col in df.columns and df[col].dtype == object:
            cleaned = df[col].astype(str).str.replace(r"[$,]", "", regex=True).str.strip()
            numeric = pd.to_numeric(cleaned, errors="coerce")
            if numeric.isna().any():
                numeric = numeric.fillna(RAW_COLUMN_DEFAULTS.get(col, 0))
                coerced.append(col)
            df[col] = numeric
    return df, coerced


def preprocess_customer_data(df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    df = df.copy()

    df["monthlycharges"] = df["monthlycharges"].clip(upper=195.48)

    expected_total = df["monthlycharges"] * df["tenure"]
    df["totalcharges"] = np.clip(df["totalcharges"], expected_total - 1000, expected_total + 1000)

    df["avg_monthly_gb"] = df["avg_monthly_gb"].fillna(27.77)
    df["credit_score"] = df["credit_score"].fillna(680.0)
    df["annual_income"] = df["annual_income"].fillna(48954.6)
    df["num_complaints"] = df["num_complaints"].fillna(1.0)
    df["customer_satisfaction"] = df["customer_satisfaction"].fillna(7.0)

    df["is_month_to_month"] = (df["contract"] == "month_to_month").astype(int)
    df["short_tenure_mtm"] = ((df["tenure"] < 6) & (df["is_month_to_month"] == 1)).astype(int)
    df["long_dormant"] = (df["days_since_last_interaction"] == 365).astype(int)
    df["num_complaints"] = df["num_complaints"].clip(upper=5)
    df["num_service_calls"] = df["num_service_calls"].clip(upper=8)

    df_encoded = pd.get_dummies(
        df, columns=["contract", "payment_method", "gender", "education", "marital_status"], drop_first=True
    )
    df_encoded["paperless_billing"] = (df_encoded["paperless_billing"] == "Yes").astype(int)

    for col in feature_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    return df_encoded[feature_columns]


def explain_risk(row: pd.Series, risk_pct: float) -> str:
    """Rule-based explanation from the EDA findings, standing in for SHAP
    until that's added alongside the sentiment integration."""
    if row.get("short_tenure_mtm", 0) == 1:
        return "Short tenure + month-to-month contract"
    if row.get("is_month_to_month", 0) == 1:
        return "Month-to-month contract"
    if row.get("num_complaints", 0) >= 3:
        return "High complaint volume"
    if row.get("customer_satisfaction", 10) <= 3:
        return "Low customer satisfaction"
    if row.get("num_service_calls", 0) >= 5:
        return "High service call volume"
    if risk_pct < 30:
        return "No major risk factors identified"
    return "Elevated risk on multiple factors"


def run_churn_prediction(raw_df: pd.DataFrame):
    """Full pipeline: normalize -> LLM-assist -> default -> coerce ->
    preprocess -> predict. Returns (results list, threshold, warning)."""
    raw_df, unmatched_uploaded, still_missing_expected = normalize_columns(raw_df)
    llm_mapping = llm_map_columns(unmatched_uploaded, still_missing_expected)
    if llm_mapping:
        raw_df = raw_df.rename(columns=llm_mapping)
    raw_df, defaulted_columns = fill_missing_columns(raw_df)
    raw_df, coerced_columns = coerce_numeric_columns(raw_df)

    customer_ids = raw_df["customer_id"] if "customer_id" in raw_df.columns else pd.Series(range(len(raw_df)))

    X = preprocess_customer_data(raw_df, _models["feature_columns"])
    X_scaled = _models["scaler"].transform(X)

    stack_input = pd.DataFrame({
        "catboost": _models["catboost"].predict_proba(X)[:, 1],
        "lightgbm": _models["lightgbm"].predict_proba(X)[:, 1],
        "logistic_regression": _models["lr"].predict_proba(X_scaled)[:, 1],
    })
    risk_scores = _models["meta"].predict_proba(stack_input)[:, 1]
    threshold = _models["threshold"]

    results = []
    for i in range(len(raw_df)):
        risk_pct = round(float(risk_scores[i]) * 100, 1)
        results.append({
            "customer_id": str(customer_ids.iloc[i]),
            "risk_score": risk_pct,
            "flagged": bool(risk_scores[i] >= threshold),
            "reason": explain_risk(X.iloc[i], risk_pct),
        })

    warning = None
    flagged_cols = defaulted_columns + coerced_columns
    if flagged_cols or llm_mapping:
        pct_missing = len(flagged_cols) / len(RAW_COLUMN_DEFAULTS)
        severity = "Predictions may be unreliable" if pct_missing > 0.3 else ("Minor impact on accuracy" if flagged_cols else "Data looks complete")
        parts = []
        if defaulted_columns:
            parts.append(f"not found (estimated): {', '.join(defaulted_columns)}")
        if coerced_columns:
            parts.append(f"had unusable values, cleaned (some estimated): {', '.join(coerced_columns)}")
        if llm_mapping:
            matches = ", ".join(f"'{k}' -> {v}" for k, v in llm_mapping.items())
            parts.append(f"AI-matched column names: {matches}")
        warning = f"{severity} — " + "; ".join(parts)

    return results, round(threshold * 100, 1), warning
