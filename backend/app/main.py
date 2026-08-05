import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.config import settings
from backend.app.routes import predictions, analysis

# ── App setup ─────────────────────────────────────────────────────

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="""
    Malaysian State Election Predictor API.
    
    Predicts DUN seat outcomes for Johor, Negeri Sembilan, and Melaka.
    Uses RF + XGB ensemble with OOD detection for regime-shift seats.
    Validated on actual 2026 election results.
    
    Data source: electiondata.my (CC0 license)
    """
)

# ── CORS ──────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────

app.include_router(predictions.router)
app.include_router(analysis.router)

# ── Root endpoints ────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name":    settings.API_TITLE,
        "version": settings.API_VERSION,
        "states":  settings.VALID_STATES,
        "docs":    "/docs",
        "endpoints": {
            "predict_all":     "GET  /predict/all/{state}",
            "predict_seat":    "POST /predict/seat/{state}",
            "metadata":        "GET  /analysis/metadata/{state}",
            "feature_imp":     "GET  /analysis/feature-importance/{state}",
            "ood_analysis":    "GET  /analysis/ood/{state}",
            "val_summary":     "GET  /analysis/validation-summary",
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

# ── Run ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )