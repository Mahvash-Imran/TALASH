"""
auth.py — User Authentication & Multi-Tenant Directory Resolution
===================================================================
Provides user registration, password verification, session token management,
and tenant-isolated data directory resolution for TALASH.
"""

import json
import hashlib
import os
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Header, Depends
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["TALASH Authentication"])

USERS_FILE = Path("data/users.json")
TENANTS_DIR = Path("data/tenants")

# Ensure base directories exist
USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
TENANTS_DIR.mkdir(parents=True, exist_ok=True)

# Default admin account
DEFAULT_ADMIN_EMAIL = "admin@talash.pk"
DEFAULT_ADMIN_PASS = "talash2026"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class UserRegisterRequest(BaseModel):
    name: str
    email: str
    password: str

class UserLoginRequest(BaseModel):
    email: str
    password: str

class AuthResponse(BaseModel):
    status: str
    message: str
    token: Optional[str] = None
    user: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _hash_password(password: str) -> str:
    """Hash password using SHA-256 with salt."""
    salt = "TALASH_SECURE_SALT_2026"
    return hashlib.sha256((password + salt).encode('utf-8')).hexdigest()

def _load_users() -> Dict[str, Dict[str, Any]]:
    """Load users from JSON database."""
    if not USERS_FILE.exists():
        # Initialize default admin user
        default_users = {
            DEFAULT_ADMIN_EMAIL: {
                "name": "TALASH Administrator",
                "email": DEFAULT_ADMIN_EMAIL,
                "password_hash": _hash_password(DEFAULT_ADMIN_PASS),
                "role": "admin",
                "created_at": "2026-08-21T10:00:00"
            }
        }
        _save_users(default_users)
        return default_users
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_users(users: Dict[str, Dict[str, Any]]):
    """Save users dictionary to JSON file."""
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)

def get_tenant_dir(user_email: Optional[str] = None) -> Path:
    """
    Resolves the data directory for the active user tenant.
    - Default admin or unauthenticated fallback -> data/analysis/
    - Regular user tenant -> data/tenants/{clean_email}/analysis/
    """
    if not user_email or user_email.lower() == DEFAULT_ADMIN_EMAIL.lower():
        target = Path("data/analysis")
        target.mkdir(parents=True, exist_ok=True)
        return target

    clean_email = user_email.lower().replace("@", "_at_").replace(".", "_")
    tenant_base = TENANTS_DIR / clean_email / "analysis"
    tenant_base.mkdir(parents=True, exist_ok=True)

    # Seed new tenant with baseline dataset structure if missing
    comp_file = tenant_base / "composite_evaluations.csv"
    if not comp_file.exists():
        seed_dir = Path("seed_data/analysis")
        if seed_dir.exists():
            for f in seed_dir.glob("*.csv"):
                try:
                    shutil.copy2(f, tenant_base / f.name)
                except Exception:
                    pass

    return tenant_base


def get_current_user_email(x_user_email: Optional[str] = Header(None)) -> str:
    """FastAPI dependency to extract current logged-in user email."""
    if not x_user_email or x_user_email.strip() == "":
        return DEFAULT_ADMIN_EMAIL
    return x_user_email.strip().lower()


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@router.post("/register", response_model=AuthResponse)
def register(req: UserRegisterRequest):
    """Register a new user account with private tenant storage."""
    email = req.email.strip().lower()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Invalid email address.")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")

    users = _load_users()
    if email in users:
        raise HTTPException(status_code=400, detail="An account with this email already exists.")

    user_data = {
        "name": req.name.strip() or email.split("@")[0].title(),
        "email": email,
        "password_hash": _hash_password(req.password),
        "role": "user",
        "created_at": "2026-08-21T10:00:00"
    }
    users[email] = user_data
    _save_users(users)

    # Initialize tenant directory
    get_tenant_dir(email)

    return AuthResponse(
        status="ok",
        message="Account created successfully. You can now log in.",
        token=f"token_{email}",
        user={"name": user_data["name"], "email": email, "role": "user"}
    )


@router.post("/login", response_model=AuthResponse)
def login(req: UserLoginRequest):
    """Authenticate user with email and password."""
    email = req.email.strip().lower()
    users = _load_users()

    if email not in users:
        # Check if login matches default admin
        if email == DEFAULT_ADMIN_EMAIL and req.password == DEFAULT_ADMIN_PASS:
            users[email] = {
                "name": "TALASH Administrator",
                "email": DEFAULT_ADMIN_EMAIL,
                "password_hash": _hash_password(DEFAULT_ADMIN_PASS),
                "role": "admin"
            }
            _save_users(users)
        else:
            raise HTTPException(status_code=401, detail="Invalid email or password.")

    user = users[email]
    if user.get("password_hash") != _hash_password(req.password):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    # Ensure tenant directory exists
    get_tenant_dir(email)

    return AuthResponse(
        status="ok",
        message="Login successful.",
        token=f"token_{email}",
        user={"name": user["name"], "email": email, "role": user.get("role", "user")}
    )


@router.get("/me")
def get_current_user_profile(email: str = Depends(get_current_user_email)):
    """Returns profile info for active logged-in user."""
    users = _load_users()
    user = users.get(email, {
        "name": email.split("@")[0].title(),
        "email": email,
        "role": "user"
    })
    return {
        "name": user.get("name"),
        "email": email,
        "role": user.get("role", "user"),
        "tenant_dir": str(get_tenant_dir(email))
    }
