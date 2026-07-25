"""
schemas.py  –  Pydantic Schemas for FastAPI Endpoints
==================================================
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    service: str = "TALASH Backend API"


class UploadResponse(BaseModel):
    filename: str
    candidate_id: str
    status: str
    message: str


class CandidateSummary(BaseModel):
    candidate_id: str
    candidate_name: str
    overall_composite_score: float
    candidate_tier: str
    total_experience_years: Optional[float] = 0.0
    scholarly_label: Optional[str] = "N/A"
    educational_strength_label: Optional[str] = "N/A"
    # Dashboard integration fields
    status: Optional[str] = "done"          # done | processing | pending | error
    missing_info_count: Optional[int] = 0   # flagged missing fields count
    uploaded_at: Optional[str] = None       # ISO date string


class CompareRequest(BaseModel):
    candidate_ids: List[str] = Field(..., min_items=2, description="List of candidate IDs to compare side-by-side")


class CompareResponse(BaseModel):
    comparison_count: int
    candidates: List[Dict[str, Any]]
    ranking: List[Dict[str, Any]]


class EmailDraftResponse(BaseModel):
    candidate_id: str
    subject: str
    body: str
    has_missing_info: bool
