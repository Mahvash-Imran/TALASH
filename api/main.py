"""
main.py  –  FastAPI Application Entry Point
==========================================
"""

# Load .env first — must be before any import that reads os.environ
try:
    from pathlib import Path as _Path
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(_Path(__file__).resolve().parent.parent / ".env", override=True)
except ImportError:
    pass

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

def ensure_seed_data():
    """Ensure data/ has evaluation CSVs and rankings even if a volume is mounted."""
    import shutil
    seed_dir = Path("seed_data")
    if not seed_dir.exists():
        return
    for folder in ["rankings", "extracted", "analysis"]:
        src = seed_dir / folder
        dst = Path("data") / folder
        if src.exists():
            dst.mkdir(parents=True, exist_ok=True)
            for item in src.glob("*"):
                if item.is_file() and not (dst / item.name).exists():
                    try:
                        shutil.copy2(item, dst / item.name)
                    except Exception:
                        pass

ensure_seed_data()


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
from .auth import router as auth_router

app.include_router(auth_router)
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
