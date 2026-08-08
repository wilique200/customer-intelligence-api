"""Health check + "coming soon" stubs for modules not built yet."""

from fastapi import APIRouter
from app.services import ml_models

router = APIRouter(prefix="/api", tags=["status"])


@router.get("/health")
def health():
    return {"status": "ok", "churn_model_loaded": ml_models.models_loaded()}


@router.get("/sentiment/status")
def sentiment_status():
    return {
        "status": "coming_soon",
        "message": "Sentiment analysis is trained and validated (86% accuracy) but not yet deployed.",
    }


@router.get("/documents/status")
def documents_status():
    return {"status": "coming_soon", "message": "Document Q&A module is planned but not yet built."}


@router.get("/expenses/status")
def expenses_status():
    return {"status": "coming_soon", "message": "Expense Extractor module is planned but not yet built."}
