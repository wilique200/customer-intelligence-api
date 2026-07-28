"""
Customer Intelligence Module — API Backend

Serves:
- POST /api/churn/predict — upload a CSV of customer data, get risk scores +
  a rule-based "why flagged" reason per customer. Uses the finalized stack
  (CatBoost + LightGBM + LR + meta-learner) from churn_model_finalize.py.
- GET  /api/sentiment/status — returns "coming_soon" until the DistilBERT
  model is deployed (on hold pending a paid Render tier)
- GET  /api/health — basic liveness check

Model artifacts are downloaded from Hugging Face Hub at startup (public
repo, so no token needed to read — only the one-time upload from Kaggle
needed a token). Set HF_REPO_ID below (or via an environment variable on
Render) to your actual HF username/repo.

Preprocessing note: the cap/impute constants below are hardcoded from the
actual training run's printed output (monthlycharges cap 195.48, and the
five imputation medians). The totalcharges gap-outlier bound is
approximated here rather than using the exact training-time quantiles,
since those specific numbers weren't saved as an artifact — worth saving
them explicitly next time you retrain, for exact train/serve consistency.
"""

import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from io import StringIO
from huggingface_hub import hf_hub_download

HF_REPO_ID = os.environ.get("HF_REPO_ID", "your-hf-username/customer-intelligence-churn-model")

MODEL_FILES = [
    "final_catboost_model.pkl",
    "final_lightgbm_model.pkl",
    "final_lr_model.pkl",
    "final_scaler.pkl",
    "final_meta_model.pkl",
    "final_feature_columns.pkl",
    "final_decision_threshold.pkl",
]

app = FastAPI(title="Customer Intelligence API")

# Comma-separated list if you ever need more than one (e.g. a Vercel
# preview URL alongside the production one). Defaults to "*" until you
# have a real frontend URL to lock it to.
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGIN", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

_models = {}


@app.on_event("startup")
def load_models():
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
        print(f"WARNING: failed to load models from HF Hub ({HF_REPO_ID}) — /api/churn/predict will fail until this is fixed. {e}")


def preprocess_customer_data(df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    df = df.copy()

    # Cap monthlycharges at the training-time 99.9th percentile
    df["monthlycharges"] = df["monthlycharges"].clip(upper=195.48)

    # Fix totalcharges gap outliers (approximated — clip relative to the
    # expected value rather than the exact training-time quantile bounds)
    expected_total = df["monthlycharges"] * df["tenure"]
    df["totalcharges"] = np.clip(df["totalcharges"], expected_total - 1000, expected_total + 1000)

    # Impute missing values with training-time medians
    df["avg_monthly_gb"] = df["avg_monthly_gb"].fillna(27.77)
    df["credit_score"] = df["credit_score"].fillna(680.0)
    df["annual_income"] = df["annual_income"].fillna(48954.6)
    df["num_complaints"] = df["num_complaints"].fillna(1.0)
    df["customer_satisfaction"] = df["customer_satisfaction"].fillna(7.0)

    # Derived features (same as training)
    df["is_month_to_month"] = (df["contract"] == "month_to_month").astype(int)
    df["short_tenure_mtm"] = ((df["tenure"] < 6) & (df["is_month_to_month"] == 1)).astype(int)
    df["long_dormant"] = (df["days_since_last_interaction"] == 365).astype(int)
    df["num_complaints"] = df["num_complaints"].clip(upper=5)
    df["num_service_calls"] = df["num_service_calls"].clip(upper=8)

    # Encode categoricals
    df = pd.get_dummies(df, columns=["contract", "payment_method", "gender", "education", "marital_status"], drop_first=True)
    df["paperless_billing"] = (df["paperless_billing"] == "Yes").astype(int)

    # Align to the exact training feature set — add any missing dummy
    # columns as 0, drop anything extra, and enforce column order
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    return df[feature_columns]


def explain_risk(row: pd.Series) -> str:
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
    return "Elevated risk on multiple factors"


@app.get("/api/health")
def health():
    return {"status": "ok", "churn_model_loaded": "catboost" in _models}


@app.get("/api/sentiment/status")
def sentiment_status():
    return {
        "status": "coming_soon",
        "message": "Sentiment analysis is trained and validated (86% accuracy) but not yet deployed.",
    }


@app.get("/api/data-analyst/status")
def data_analyst_status():
    return {"status": "coming_soon", "message": "Data Analyst module is planned but not yet built."}


@app.get("/api/documents/status")
def documents_status():
    return {"status": "coming_soon", "message": "Document Q&A module is planned but not yet built."}


@app.get("/api/expenses/status")
def expenses_status():
    return {"status": "coming_soon", "message": "Expense Extractor module is planned but not yet built."}


@app.post("/api/churn/predict")
async def predict_churn(file: UploadFile = File(...)):
    if "catboost" not in _models:
        raise HTTPException(status_code=503, detail="Churn model not loaded — check MODEL_DIR on the server.")

    contents = await file.read()
    try:
        raw_df = pd.read_csv(StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    customer_ids = raw_df["customer_id"] if "customer_id" in raw_df.columns else pd.Series(range(len(raw_df)))

    try:
        X = preprocess_customer_data(raw_df, _models["feature_columns"])
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing expected column in upload: {e}")

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
            "reason": explain_risk(X.iloc[i]),
        })

    return {
        "threshold": round(threshold * 100, 1),
        "total_customers": len(results),
        "flagged_count": sum(r["flagged"] for r in results),
        "results": results,
    }
