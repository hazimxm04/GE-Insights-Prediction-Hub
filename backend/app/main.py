from fastapi import FastAPI
from app.config import settings
from app.routes import predictions, analysis, health
from utils.logging import setup_logging

setup_logging()

app = FastAPI(
    title="Multi-State Election Predictor",
    version="1.0.0"
)

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(predictions.router, tags=["predictions"])
app.include_router(analysis.router, tags=["analysis"])

@app.get("/")
async def root():
    return {"status": "alive"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=settings.DEBUG)