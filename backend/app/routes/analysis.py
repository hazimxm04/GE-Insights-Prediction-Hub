import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fastapi import APIRouter, HTTPException
from backend.app.config import settings

router = APIRouter(prefix="/analysis", tags=["analysis"])

@router.get("/metadata/{state}")
async def get_state_metadata(state: str):
    """Get model metadata: accuracy, features, OOD stats"""
    if state not in settings.VALID_STATES:
        raise HTTPException(status_code=400, detail=f"Invalid state: {state}")

    metadata_path = settings.MODELS_DIR / state / "metadata.json"

    if not metadata_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No metadata found for {state}"
        )

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return metadata

@router.get("/feature-importance/{state}")
async def get_feature_importance(state: str):
    """Get feature importance from trained RF model"""
    if state not in settings.VALID_STATES:
        raise HTTPException(status_code=400, detail=f"Invalid state: {state}")

    metadata_path = settings.MODELS_DIR / state / "metadata.json"
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Metadata not found")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    importance = metadata.get("feature_importance", {})

    # Sort by importance
    sorted_importance = sorted(
        importance.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return {
        "state": state,
        "feature_importance": [
            {"feature": k, "importance": v}
            for k, v in sorted_importance
        ]
    }

@router.get("/ood/{state}")
async def get_ood_analysis(state: str):
    """Get OOD flagged seats from 2026 validation"""
    if state not in settings.VALID_STATES:
        raise HTTPException(status_code=400, detail=f"Invalid state: {state}")

    import pandas as pd
    val_path = settings.MODELS_DIR / state / "validation_2026.csv"

    if not val_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No 2026 validation data for {state}"
        )

    df = pd.read_csv(val_path)

    ood_seats = df[df["is_ood"] == True]
    in_dist_seats = df[df["is_ood"] == False]

    # Accuracy breakdown
    def accuracy(subset):
        if len(subset) == 0:
            return None
        correct = (
            (subset["probability"] >= 0.5).astype(int) ==
            subset["actual_non_bn"]
        ).mean()
        return round(float(correct), 4)

    return {
        "state": state,
        "total_seats": len(df),
        "ood_seats": {
            "count": len(ood_seats),
            "percentage": round(len(ood_seats)/len(df)*100, 1),
            "accuracy": accuracy(ood_seats),
            "seats": ood_seats["seat_name"].tolist()
                     if "seat_name" in ood_seats.columns else []
        },
        "in_distribution_seats": {
            "count": len(in_dist_seats),
            "percentage": round(len(in_dist_seats)/len(df)*100, 1),
            "accuracy": accuracy(in_dist_seats),
        },
        "interpretation": (
            "100% OOD indicates full regime shift — "
            "political dynamics in 2026 are completely "
            "different from training data (2018-2023)"
            if len(ood_seats) == len(df)
            else f"{len(ood_seats)} seats show regime-shift patterns"
        )
    }

@router.get("/validation-summary")
async def get_validation_summary():
    """Get 2026 validation results across all states"""
    import pandas as pd

    results = {}

    for state in settings.VALID_STATES:
        val_path = settings.MODELS_DIR / state / "validation_2026.csv"
        meta_path = settings.MODELS_DIR / state / "metadata.json"

        if not val_path.exists():
            results[state] = {"status": "no_validation_data"}
            continue

        df = pd.read_csv(val_path)

        with open(meta_path, "r") as f:
            metadata = json.load(f)

        correct = (
            (df["probability"] >= 0.5).astype(int) ==
            df["actual_non_bn"]
        ).mean()

        results[state] = {
            "seats":    len(df),
            "accuracy": round(float(correct), 4),
            "ood_pct":  round(df["is_ood"].mean() * 100, 1),
            "training": {
                "rf_accuracy":  metadata.get("models", {})
                                .get("random_forest", {})
                                .get("accuracy"),
                "ensemble_accuracy": metadata.get("models", {})
                                    .get("ensemble", {})
                                    .get("accuracy"),
            }
        }

    return {
        "validation_year": 2026,
        "note": "Training accuracy inflated (train=test). "
                "Validation accuracy is honest performance.",
        "results": results
    }