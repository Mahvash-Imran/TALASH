"""
main.py  –  FastAPI Application Entry Point
==========================================
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from .routes import router

app = FastAPI(
    title="TALASH Candidate Evaluation API",
    description="Automated Faculty Candidate Evaluation & Research Analysis System API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
from .routes import router
from .jd_routes import router as jd_router

app.include_router(router)
app.include_router(jd_router)

# Mount Data Analysis Directory for direct file access if needed
data_dir = Path("data/analysis")
if data_dir.exists():
    app.mount("/static/analysis", StaticFiles(directory=str(data_dir)), name="analysis_static")

# Mount Frontend Dashboard
frontend_dir = Path("frontend")
if frontend_dir.exists():
    app.mount("/dashboard", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


@app.get("/")
def root():
    """Serve the frontend dashboard if it exists, otherwise return API info."""
    frontend_index = Path("frontend/index.html")
    if frontend_index.exists():
        return FileResponse(str(frontend_index))
    return {
        "message": "Welcome to TALASH Candidate Evaluation API",
        "dashboard": "/dashboard",
        "documentation": "/docs",
        "health_check": "/api/v1/health",
    }
