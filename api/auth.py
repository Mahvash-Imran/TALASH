"""
auth.py — User Authentication & Multi-Tenant Directory Resolution
===================================================================
Provides user registration, password verification, session token management,
and tenant-isolated data directory resolution for TALASH.

Security model
--------------
* Passwords are hashed with PBKDF2-HMAC-SHA256 and a random per-user salt.
* Login issues a random opaque bearer token (only its SHA-256 is stored server-side).
* Every protected endpoint depends on ``get_current_user_email`` which validates
  the ``Authorization: Bearer <token>`` header and returns 401 otherwise.
* The default admin password is never hardcoded: set ``TALASH_ADMIN_PASSWORD`` or a
  random one is generated on first start into ``data/ADMIN_PASSWORD.txt`` (git-ignored).
"""

import datetime
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["TALASH Authentication"])

USERS_FILE = Path("data/users.json")
SESSIONS_FILE = Path("data/sessions.json")
ADMIN_PASSWORD_FILE = Path("data/ADMIN_PASSWORD.txt")
TENANTS_DIR = Path("data/tenants")

# Ensure base directories exist
USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
TENANTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_ADMIN_EMAIL = os.environ.get("TALASH_ADMIN_EMAIL", "admin@talash.pk").strip().lower()
SESSION_TTL_SECONDS = int(os.environ.get("TALASH_SESSION_HOURS", "12")) * 3600
MIN_PASSWORD_LENGTH = 8
PBKDF2_ITERATIONS = 200_000
_LEGACY_SALT = "TALASH_SECURE_SALT_2026"
_LEGACY_DEFAULT_PASS = "talash2026"  # only used to detect + rotate the old public default
EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")

_file_lock = threading.RLock()
_failed_logins: Dict[str, list] = {}
MAX_FAILED_LOGINS = 5
LOCKOUT_SECONDS = 60


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class UserRegisterRequest(BaseModel):
    name: str = Field(..., max_length=100)
    email: str = Field(..., max_length=254)
    password: str = Field(..., max_length=256)


class UserLoginRequest(BaseModel):
    email: str = Field(..., max_length=254)
    password: str = Field(..., max_length=256)


class AuthResponse(BaseModel):
    status: str
    message: str
    token: Optional[str] = None
    user: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

def _hash_password(password: str, salt: Optional[str] = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), PBKDF2_ITERATIONS)
    return f"pbkdf2${PBKDF2_ITERATIONS}${salt}${digest.hex()}"


def _legacy_hash(password: str) -> str:
    return hashlib.sha256((password + _LEGACY_SALT).encode("utf-8")).hexdigest()


def _verify_password(password: str, stored: str) -> bool:
    if not stored:
        return False
    if stored.startswith("pbkdf2$"):
        try:
            _, iters, salt, digest = stored.split("$")
            calc = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), int(iters)).hex()
            return hmac.compare_digest(calc, digest)
        except Exception:
            return False
    # Legacy hash — transparently upgraded to PBKDF2 on next successful login
    return hmac.compare_digest(_legacy_hash(password), stored)


# ---------------------------------------------------------------------------
# Storage (atomic writes, lock-protected)
# ---------------------------------------------------------------------------

def _atomic_write_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def _admin_password() -> str:
    env = os.environ.get("TALASH_ADMIN_PASSWORD", "").strip()
    if env:
        return env
    if ADMIN_PASSWORD_FILE.exists():
        stored = ADMIN_PASSWORD_FILE.read_text(encoding="utf-8").strip()
        if stored:
            return stored
    pw = secrets.token_urlsafe(12)
    ADMIN_PASSWORD_FILE.parent.mkdir(parents=True, exist_ok=True)
    ADMIN_PASSWORD_FILE.write_text(pw + "\n", encoding="utf-8")
    logger.warning("Generated admin password and saved it to %s (admin: %s)", ADMIN_PASSWORD_FILE, DEFAULT_ADMIN_EMAIL)
    return pw


def _admin_record(password: str) -> Dict[str, Any]:
    return {
        "name": "TALASH Administrator",
        "email": DEFAULT_ADMIN_EMAIL,
        "password_hash": _hash_password(password),
        "role": "admin",
        "created_at": _now(),
    }


def _load_users() -> Dict[str, Dict[str, Any]]:
    """Load users from the JSON database, creating / repairing the admin account."""
    with _file_lock:
        if not USERS_FILE.exists():
            users = {DEFAULT_ADMIN_EMAIL: _admin_record(_admin_password())}
            _atomic_write_json(USERS_FILE, users)
            return users
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                users = json.load(f)
        except Exception as exc:
            # Never return {} here: the next save would wipe every account.
            logger.error("users.json is unreadable: %s", exc)
            raise HTTPException(status_code=500, detail="User database is unavailable.")
        admin = users.get(DEFAULT_ADMIN_EMAIL)
        # Rotate the old publicly-known default admin password
        if admin is None or _verify_password(_LEGACY_DEFAULT_PASS, admin.get("password_hash", "")):
            users[DEFAULT_ADMIN_EMAIL] = _admin_record(_admin_password())
            _atomic_write_json(USERS_FILE, users)
        return users


def _save_users(users: Dict[str, Dict[str, Any]]):
    with _file_lock:
        _atomic_write_json(USERS_FILE, users)


# ---------------------------------------------------------------------------
# Sessions (random opaque tokens; only a SHA-256 of the token is stored)
# ---------------------------------------------------------------------------

def _token_key(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _load_sessions() -> Dict[str, Dict[str, Any]]:
    try:
        with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _create_session(email: str) -> str:
    token = secrets.token_urlsafe(32)
    with _file_lock:
        now = time.time()
        sessions = {k: v for k, v in _load_sessions().items() if v.get("expires", 0) > now}
        sessions[_token_key(token)] = {"email": email, "expires": now + SESSION_TTL_SECONDS}
        _atomic_write_json(SESSIONS_FILE, sessions)
    return token


def _revoke_session(token: str):
    with _file_lock:
        sessions = _load_sessions()
        if sessions.pop(_token_key(token), None) is not None:
            _atomic_write_json(SESSIONS_FILE, sessions)


def _extract_bearer(authorization: Optional[str]) -> Optional[str]:
    if not authorization or not isinstance(authorization, str):
        return None
    parts = authorization.strip().split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer" and parts[1].strip():
        return parts[1].strip()
    return None


def get_current_user_email(authorization: Optional[str] = Header(None)) -> str:
    """
    FastAPI dependency: validates the Bearer token and returns the user's email.
    Raises 401 when the token is missing, invalid or expired — there is no anonymous fallback.
    """
    token = _extract_bearer(authorization)
    if token:
        with _file_lock:
            sess = _load_sessions().get(_token_key(token))
        if sess and sess.get("expires", 0) > time.time():
            return sess["email"]
    raise HTTPException(
        status_code=401,
        detail="Authentication required. Please sign in.",
        headers={"WWW-Authenticate": "Bearer"},
    )


# ---------------------------------------------------------------------------
# Tenant directories
# ---------------------------------------------------------------------------

def tenant_slug(email: str) -> str:
    """Filesystem-safe identifier for a user (only [a-z0-9_])."""
    return re.sub(r"[^a-z0-9_]", "_", email.lower().replace("@", "_at_").replace(".", "_"))


def get_tenant_dir(user_email: Optional[str] = None) -> Path:
    """
    Resolves the data directory for the active user tenant.
    - Admin -> data/analysis/
    - Regular user tenant -> data/tenants/{slug}/analysis/
    """
    if not user_email or user_email.lower() == DEFAULT_ADMIN_EMAIL:
        target = Path("data/analysis")
        target.mkdir(parents=True, exist_ok=True)
        return target

    tenant_base = TENANTS_DIR / tenant_slug(user_email) / "analysis"
    tenant_base.mkdir(parents=True, exist_ok=True)

    # New user tenant — always start with an empty evaluations file (no pre-loaded candidates)
    comp_file = tenant_base / "composite_evaluations.csv"
    if not comp_file.exists():
        headers = "candidate_id,overall_composite_score,candidate_tier,education_score,research_score,supervision_score,innovation_score,breadth_score,collaboration_score,experience_score,candidate_name\n"
        try:
            comp_file.write_text(headers, encoding="utf-8")
        except Exception:
            logger.exception("Could not initialise tenant evaluations file")

    return tenant_base


def _is_admin_dir(tenant_analysis_dir: Path) -> bool:
    return Path(tenant_analysis_dir) == Path("data/analysis")


def record_tenant_candidate(tenant_analysis_dir: Path, candidate_id: str):
    """Record an uploaded candidate ID under this tenant."""
    tenant_analysis_dir = Path(tenant_analysis_dir)
    if _is_admin_dir(tenant_analysis_dir):
        return
    c_file = tenant_analysis_dir / "candidate_ids.json"
    ids = []
    if c_file.exists():
        try:
            with open(c_file, "r", encoding="utf-8") as f:
                ids = json.load(f)
        except Exception:
            logger.exception("candidate_ids.json unreadable")
            ids = []
    if candidate_id not in ids:
        ids.append(candidate_id)
        _atomic_write_json(c_file, ids)


def filter_tenant_analysis_files(tenant_analysis_dir: Path):
    """
    Ensures all CSVs in a non-admin tenant directory only contain rows
    for candidates that this specific tenant has uploaded.
    """
    tenant_analysis_dir = Path(tenant_analysis_dir)
    if _is_admin_dir(tenant_analysis_dir):
        return
    c_file = tenant_analysis_dir / "candidate_ids.json"
    allowed_ids = set()
    if c_file.exists():
        try:
            with open(c_file, "r", encoding="utf-8") as f:
                allowed_ids = set(json.load(f))
        except Exception:
            logger.exception("candidate_ids.json unreadable")
            allowed_ids = set()

    import pandas as pd
    for csv_file in tenant_analysis_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file, dtype=str)
            if "candidate_id" in df.columns:
                df[df["candidate_id"].isin(allowed_ids)].to_csv(csv_file, index=False)
        except Exception:
            logger.exception("Could not filter %s", csv_file)


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

def _check_lockout(key: str):
    now = time.time()
    attempts = [t for t in _failed_logins.get(key, []) if now - t < LOCKOUT_SECONDS]
    _failed_logins[key] = attempts
    if len(attempts) >= MAX_FAILED_LOGINS:
        raise HTTPException(status_code=429, detail=f"Too many failed attempts. Try again in {LOCKOUT_SECONDS} seconds.")


@router.post("/register", response_model=AuthResponse)
def register(req: UserRegisterRequest):
    """Register a new user account with private tenant storage."""
    email = req.email.strip().lower()
    if not EMAIL_RE.match(email):
        raise HTTPException(status_code=400, detail="Invalid email address.")
    if len(req.password) < MIN_PASSWORD_LENGTH:
        raise HTTPException(status_code=400, detail=f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")

    with _file_lock:
        users = _load_users()
        if email in users:
            raise HTTPException(status_code=400, detail="An account with this email already exists.")
        slug = tenant_slug(email)
        if any(tenant_slug(e) == slug for e in users):
            raise HTTPException(status_code=400, detail="An account with a conflicting email already exists.")
        user_data = {
            "name": req.name.strip() or email.split("@")[0].title(),
            "email": email,
            "password_hash": _hash_password(req.password),
            "role": "user",
            "created_at": _now(),
        }
        users[email] = user_data
        _save_users(users)

    get_tenant_dir(email)
    token = _create_session(email)
    return AuthResponse(
        status="ok",
        message="Account created successfully.",
        token=token,
        user={"name": user_data["name"], "email": email, "role": "user"},
    )


@router.post("/login", response_model=AuthResponse)
def login(req: UserLoginRequest):
    """Authenticate user with email and password; returns an opaque session token."""
    email = req.email.strip().lower()
    _check_lockout(email)
    users = _load_users()
    user = users.get(email)

    if not user or not _verify_password(req.password, user.get("password_hash", "")):
        _failed_logins.setdefault(email, []).append(time.time())
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    _failed_logins.pop(email, None)
    if not user["password_hash"].startswith("pbkdf2$"):
        with _file_lock:
            users = _load_users()
            users[email]["password_hash"] = _hash_password(req.password)
            _save_users(users)

    get_tenant_dir(email)
    token = _create_session(email)
    return AuthResponse(
        status="ok",
        message="Login successful.",
        token=token,
        user={"name": user["name"], "email": email, "role": user.get("role", "user")},
    )


@router.post("/logout")
def logout(authorization: Optional[str] = Header(None)):
    """Invalidate the current session token."""
    token = _extract_bearer(authorization)
    if token:
        _revoke_session(token)
    return {"status": "ok"}


@router.get("/me")
def get_current_user_profile(email: str = Depends(get_current_user_email)):
    """Returns profile info for the active logged-in user."""
    users = _load_users()
    user = users.get(email, {"name": email.split("@")[0].title(), "email": email, "role": "user"})
    return {
        "name": user.get("name"),
        "email": email,
        "role": user.get("role", "user"),
    }
