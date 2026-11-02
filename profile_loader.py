"""
profile_loader.py  –  Local File Caching & Department Profile Manager
====================================================================

WHY THIS MODULE EXISTS
----------------------
Provides high-performance, thread-safe, local JSON caching and management for
TALASH Department Profiles (`/profiles/<slug>.json`).
  - Adapts the Universal Scoring Engine to any discipline without code changes.
  - Caches validated `DepartmentProfile` instances in-memory with file timestamp tracking.
  - Supports slug normalization, fuzzy matching across department titles, and listing.
  - Integrates seamlessly with pandas and FastAPI routes for transparent reporting.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from rapidfuzz import fuzz, process

from models import DepartmentProfile, UniversalDimension, RolePreset

logger = logging.getLogger(__name__)

# Default profiles directory relative to talash root
DEFAULT_PROFILES_DIR = Path(__file__).resolve().parent / "profiles"

# Thread-safe in-memory cache: slug -> (DepartmentProfile, last_modified_timestamp)
_PROFILE_CACHE: Dict[str, Tuple[DepartmentProfile, float]] = {}
_CACHE_LOCK = threading.Lock()


def _normalize_slug(text: str) -> str:
    """Converts department name or input string to a URL/file-safe slug."""
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_").lower()
    return clean or "general_department"


def get_profiles_dir(custom_dir: Optional[Path] = None) -> Path:
    """Resolves and ensures the profiles directory exists."""
    p_dir = custom_dir if custom_dir is not None else DEFAULT_PROFILES_DIR
    p_dir.mkdir(parents=True, exist_ok=True)
    return p_dir


def load_profile_from_file(file_path: Path) -> Optional[DepartmentProfile]:
    """Loads, parses, and validates a DepartmentProfile from a JSON file."""
    if not file_path.exists() or not file_path.is_file():
        return None
    try:
        content = file_path.read_text(encoding="utf-8")
        raw_data = json.loads(content)
        profile = DepartmentProfile.model_validate(raw_data)
        return profile
    except Exception as exc:
        logger.error("Failed to load department profile from %s: %s", file_path, exc)
        return None


def get_profile(
    slug_or_name: str,
    profiles_dir: Optional[Path] = None,
    allow_fuzzy: bool = True,
) -> Optional[DepartmentProfile]:
    """
    Retrieves a DepartmentProfile by slug or human-readable name.
    1. Checks in-memory cache (with mtime invalidation).
    2. Searches profiles_dir/<slug>.json on disk.
    3. If allow_fuzzy is True, fuzzy-matches against all existing profile names.
    """
    if not slug_or_name or not slug_or_name.strip():
        return None

    target_dir = get_profiles_dir(profiles_dir)
    target_slug = _normalize_slug(slug_or_name)
    target_file = target_dir / f"{target_slug}.json"

    with _CACHE_LOCK:
        # Check cache if file exists and has not been modified
        if target_slug in _PROFILE_CACHE and target_file.exists():
            cached_profile, cached_mtime = _PROFILE_CACHE[target_slug]
            try:
                curr_mtime = target_file.stat().st_mtime
                if curr_mtime <= cached_mtime:
                    return cached_profile
            except OSError:
                pass

        # Try direct file load
        if target_file.exists():
            profile = load_profile_from_file(target_file)
            if profile:
                try:
                    mtime = target_file.stat().st_mtime
                except OSError:
                    mtime = 0.0
                _PROFILE_CACHE[target_slug] = (profile, mtime)
                return profile

    # Multi-stage fuzzy / keyword resolution fallback across available profiles
    if allow_fuzzy:
        available = list_profiles(profiles_dir=target_dir)
        if available:
            q_clean = slug_or_name.strip().lower()
            stop_words = {"department", "school", "faculty", "college", "studies", "sciences", "program", "division", "of", "and", "the", "&"}
            q_words = set(w for w in re.findall(r"\w+", q_clean) if len(w) > 2 and w not in stop_words)

            # 1. Exact name match
            for p in available:
                if p["department_name"].lower() == q_clean:
                    return get_profile(p["slug"], profiles_dir=target_dir, allow_fuzzy=False)

            # 2. Score all available profiles by keyword overlap and fuzzy metrics
            best_slug = None
            best_score = 0.0

            for p in available:
                p_slug_words = set(p["slug"].split("_"))
                p_name_words = set(w.lower() for w in re.findall(r"\w+", p["department_name"]) if len(w) > 2 and w not in stop_words)
                combined_words = p_slug_words | p_name_words

                overlap = len(q_words & combined_words)
                target = f"{p['slug']} {p['department_name']}".lower()

                # Weighted composite similarity
                fuzzy_wratio = fuzz.WRatio(q_clean, target)
                fuzzy_token_set = fuzz.token_set_ratio(q_clean, target)
                fuzzy_score = max(fuzzy_wratio, fuzzy_token_set)

                # Overlap boost
                total_score = fuzzy_score + (overlap * 20.0)

                if total_score > best_score and (fuzzy_score >= 60.0 or overlap >= 1):
                    best_score = total_score
                    best_slug = p["slug"]

            if best_slug:
                logger.info(
                    "Fuzzy-matched query '%s' to profile '%s' (score: %.1f)",
                    slug_or_name,
                    best_slug,
                    best_score,
                )
                return get_profile(best_slug, profiles_dir=target_dir, allow_fuzzy=False)

    return None


def save_profile(
    profile: DepartmentProfile,
    profiles_dir: Optional[Path] = None,
    overwrite: bool = True,
) -> Path:
    """
    Persists a DepartmentProfile to disk as formatted JSON and updates the memory cache.
    """
    target_dir = get_profiles_dir(profiles_dir)
    file_path = target_dir / f"{profile.slug}.json"

    if file_path.exists() and not overwrite:
        raise FileExistsError(f"Profile file already exists at {file_path}")

    # Ensure profile passes schema validation
    profile_dict = profile.model_dump()
    json_str = json.dumps(profile_dict, indent=2, ensure_ascii=False)
    file_path.write_text(json_str, encoding="utf-8")

    # Update memory cache
    with _CACHE_LOCK:
        try:
            mtime = file_path.stat().st_mtime
        except OSError:
            mtime = 0.0
        _PROFILE_CACHE[profile.slug] = (profile, mtime)

    logger.info("Saved department profile '%s' to %s", profile.slug, file_path)
    return file_path


def list_profiles(profiles_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Lists all department profiles available in the profiles directory.
    Returns metadata summaries suitable for dropdowns, admin panels, and API responses.
    """
    target_dir = get_profiles_dir(profiles_dir)
    results: List[Dict[str, Any]] = []

    for file_path in sorted(target_dir.glob("*.json")):
        profile = load_profile_from_file(file_path)
        if profile:
            results.append(
                {
                    "slug": profile.slug,
                    "department_name": profile.department_name,
                    "version": profile.version,
                    "status": profile.status,
                    "description": profile.description,
                    "hard_gates_count": len(profile.hard_gates),
                    "artifact_types": list(profile.artifact_taxonomy.keys()),
                    "core_skills_count": len(profile.skill_taxonomy.core_skills),
                    "file_path": str(file_path),
                }
            )

    return results


def delete_profile(slug: str, profiles_dir: Optional[Path] = None) -> bool:
    """Removes a profile from disk and evicts it from the in-memory cache."""
    target_dir = get_profiles_dir(profiles_dir)
    target_slug = _normalize_slug(slug)
    file_path = target_dir / f"{target_slug}.json"

    with _CACHE_LOCK:
        _PROFILE_CACHE.pop(target_slug, None)

    if file_path.exists():
        try:
            file_path.unlink()
            logger.info("Deleted profile file %s", file_path)
            return True
        except OSError as exc:
            logger.error("Failed to delete profile file %s: %s", file_path, exc)
            return False
    return False


def reload_cache(profiles_dir: Optional[Path] = None) -> int:
    """Clears and re-reads all available profiles from disk into memory."""
    target_dir = get_profiles_dir(profiles_dir)
    count = 0
    with _CACHE_LOCK:
        _PROFILE_CACHE.clear()
        for file_path in target_dir.glob("*.json"):
            profile = load_profile_from_file(file_path)
            if profile:
                try:
                    mtime = file_path.stat().st_mtime
                except OSError:
                    mtime = 0.0
                _PROFILE_CACHE[profile.slug] = (profile, mtime)
                count += 1
    logger.info("Reloaded %d department profiles into cache.", count)
    return count


def export_profiles_catalog_df(profiles_dir: Optional[Path] = None) -> pd.DataFrame:
    """Exports available profiles metadata as a pandas DataFrame."""
    profiles = list_profiles(profiles_dir)
    if not profiles:
        return pd.DataFrame(
            columns=["slug", "department_name", "status", "version", "description", "file_path"]
        )
    return pd.DataFrame(profiles)
