"""
profile_generator.py  –  Automated Department Profile Generator via Claude (Anthropic SDK)
========================================================================================

WHY THIS MODULE EXISTS
----------------------
When a candidate or job description arrives for a department that lacks a profile in
`/profiles/<slug>.json`, this module automatically drafts a domain-tailored, transparent
DepartmentProfile using Claude (Anthropic SDK structured JSON output).

KEY SPECIFICATIONS:
  1. Universal 8-Dimension Alignment:
     - Base weights strictly normalized to sum to 1.0 across the 8 dimensions.
     - Role modifiers for faculty, researcher, industry_professional, fresh_graduate, executive.
  2. Domain-Specific Taxonomies:
     - Hard gate rules (pass/fail thresholds, accreditation requirements).
     - Field statistical baselines for percentile normalizations.
     - Artifact taxonomy (discipline-specific evidence: papers, patents, briefs, cases, etc.).
     - Professional credentials and skill hierarchies.
  3. Strict Pydantic Validation:
     - Every generated profile is validated against `DepartmentProfile` Pydantic model.
     - Always stamped with `status="draft"` pending human administrator verification.
  4. Resilient Fallbacks:
     - Primary: Anthropic SDK (`claude-3-5-sonnet-20241022` / `claude-3-haiku-20240307`).
     - Secondary: OpenAI/Groq compatible provider (if configured in .env).
     - Heuristic: Deterministic domain synthesizer (ensures zero downtime if offline).
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from models import (
    DepartmentProfile,
    UniversalDimension,
    RolePreset,
    GateStatus,
    HardGateRule,
    BaselineMetrics,
    ArtifactTypeConfig,
    CredentialRequirement,
    SkillTaxonomyConfig,
)
from profile_loader import get_profiles_dir, save_profile

logger = logging.getLogger(__name__)

# Load .env for LLM credentials
load_dotenv(Path(__file__).resolve().parent / ".env", override=False)


SYSTEM_PROMPT = """You are a Principal Academic Administrator and AI Recruitment Architect.
Your task is to draft an authoritative, domain-specific DepartmentProfile configuration for a university recruitment system.

The profile must conform strictly to the following 8 Universal Evaluation Dimensions (base_weights must sum to 1.0):
1. "academic_foundation": Degree level, relevance, GPA normalized 0-1, institution tier.
2. "domain_competence": Core and specialized technical/domain skills.
3. "professional_experience": Relevant years of experience, seniority progression, scale.
4. "evidence_of_output": Tangible discipline-defined outputs (papers, patents, code, designs, clinical cases, legal cases, exhibitions).
5. "credentials_licensure": Professional licenses, board memberships, certifications.
6. "impact_recognition": Awards, grants, citations, measurable outcomes, press.
7. "leadership_teaching_service": Supervision, mentoring, teaching, committees, society roles.
8. "trajectory_growth": Promotion pace, recent upskilling, career coherence.

CRITICAL CONSTRAINTS:
1. base_weights must sum to 1.0.
2. role_modifiers must provide relative multipliers for: "faculty", "researcher", "industry_professional", "fresh_graduate", "executive".
3. hard_gates must define mandatory baseline criteria (e.g. minimum degree or license) with failure_status ("not_eligible" or "needs_verification").
4. field_baselines must define statistical norms (median, p75, p90, max_cap) for relevant output counts.
5. artifact_taxonomy must define 3-5 discipline-specific artifact types and quality indicators.
6. credentials_taxonomy must list key professional licenses/memberships.
7. skill_taxonomy must provide core_skills, secondary_skills, and sub_branches.
8. status must strictly be "draft".
9. Output MUST be valid JSON only, without any markdown formatting or commentary.
"""


def _clean_json_text(text: str) -> str:
    """Strips markdown code fences and cleans raw JSON text."""
    t = text.strip()
    if t.startswith("```json"):
        t = t[7:]
    elif t.startswith("```"):
        t = t[3:]
    if t.endswith("```"):
        t = t[:-3]
    return t.strip()


def _slugify(name: str) -> str:
    """Creates a clean filesystem slug from department name."""
    s = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip()).strip("_").lower()
    return s or "general_department"


def generate_with_anthropic(
    department_name: str,
    sample_jd_text: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = "claude-3-5-sonnet-20241022",
) -> Dict[str, Any]:
    """Calls Claude via Anthropic SDK using structured output."""
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError("anthropic package is not installed. Install via 'pip install anthropic'.") from e

    anthropic_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set.")

    client = anthropic.Anthropic(api_key=anthropic_key)

    user_prompt = f"Draft a comprehensive recruitment DepartmentProfile for the department: '{department_name}'."
    if sample_jd_text:
        user_prompt += f"\n\nHere is a sample Job Description for reference:\n```\n{sample_jd_text[:3000]}\n```"
    user_prompt += "\nOutput the complete profile strictly as a valid JSON object matching the DepartmentProfile schema. Status must be 'draft'."

    logger.info("Calling Anthropic Claude (%s) to generate profile for '%s'...", model, department_name)

    # Use Claude with system prompt
    response = client.messages.create(
        model=model,
        max_tokens=4000,
        temperature=0.2,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    content_text = ""
    for block in response.content:
        if hasattr(block, "text"):
            content_text += block.text

    cleaned = _clean_json_text(content_text)
    return json.loads(cleaned)


def generate_with_openai_fallback(
    department_name: str,
    sample_jd_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Fallback generator using OpenAI/Groq configured in .env."""
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is not configured for fallback.")

    client = OpenAI(api_key=api_key, base_url=base_url)

    user_prompt = f"Generate a DepartmentProfile JSON configuration for '{department_name}'."
    if sample_jd_text:
        user_prompt += f"\nSample JD text:\n{sample_jd_text[:2000]}"
    user_prompt += "\nReturn strictly JSON with status='draft'."

    logger.info("Calling OpenAI/Groq provider (%s) for department profile...", model)

    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    raw_text = response.choices[0].message.content or "{}"
    return json.loads(_clean_json_text(raw_text))


def generate_heuristic_draft(
    department_name: str,
    sample_jd_text: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Deterministic domain synthesizer fallback if no LLM APIs are accessible.
    Produces a valid, realistic starter draft profile ready for human customization.
    """
    slug = _slugify(department_name)
    logger.warning("Generating heuristic draft profile for '%s' (no LLM available).", department_name)

    # Infer domain keywords
    words = [w.lower() for w in re.findall(r"\w+", department_name) if len(w) > 3]

    return {
        "slug": slug,
        "department_name": department_name,
        "version": "1.0.0",
        "status": "draft",
        "description": f"Automated starter draft profile for {department_name}. Pending administrative review.",
        "base_weights": {
            "academic_foundation": 0.18,
            "domain_competence": 0.20,
            "professional_experience": 0.18,
            "evidence_of_output": 0.16,
            "credentials_licensure": 0.08,
            "impact_recognition": 0.08,
            "leadership_teaching_service": 0.07,
            "trajectory_growth": 0.05,
        },
        "role_modifiers": {
            "faculty": {
                "evidence_of_output": 1.4,
                "academic_foundation": 1.3,
                "leadership_teaching_service": 1.4,
            },
            "researcher": {
                "evidence_of_output": 1.6,
                "impact_recognition": 1.3,
            },
            "industry_professional": {
                "professional_experience": 1.6,
                "domain_competence": 1.3,
            },
            "fresh_graduate": {
                "academic_foundation": 1.8,
                "domain_competence": 1.5,
            },
            "executive": {
                "leadership_teaching_service": 1.7,
                "professional_experience": 1.6,
            },
        },
        "hard_gates": [
            {
                "gate_id": f"gate_{slug}_degree",
                "name": f"Relevant Degree in {department_name}",
                "dimension": "academic_foundation",
                "field_path": "education.highest_degree",
                "operator": "in",
                "expected_value": ["phd", "ms", "mphil", "bs", "master", "bachelor"],
                "failure_status": "not_eligible",
                "failure_message": f"Candidate requires an accredited degree in {department_name} or related discipline.",
            }
        ],
        "field_baselines": {
            "scholarly_publications": {"median": 4, "p75": 10, "p90": 20, "max_cap": 60, "unit": "count"},
            "experience_years": {"median": 5, "p75": 10, "p90": 16, "max_cap": 30, "unit": "years"},
        },
        "artifact_taxonomy": {
            "journal": {
                "weight": 1.0,
                "quality_indicators": ["Peer reviewed", "HEC Recognized", "Impact Factor", "Indexed"],
                "max_score_contribution": 100.0,
            },
            "conference": {
                "weight": 0.85,
                "quality_indicators": ["International Proceedings", "Peer reviewed"],
                "max_score_contribution": 80.0,
            },
            "book": {
                "weight": 0.90,
                "quality_indicators": ["Academic Publisher", "Edited Volume"],
                "max_score_contribution": 90.0,
            },
        },
        "credentials_taxonomy": [
            {
                "code": "PROFESSIONAL_REGISTRATION",
                "name": f"Professional Council / Society Membership for {department_name}",
                "is_mandatory": False,
                "acceptable_equivalents": [],
            }
        ],
        "skill_taxonomy": {
            "core_skills": words + ["research methodology", "data analysis", "domain practice", "communication"],
            "secondary_skills": ["project management", "technical reporting", "curriculum planning"],
            "sub_branches": {"primary": words},
        },
        "institution_tiers": {
            "Tier 1": ["HEC Recognized Top Institutions", "International Accredited Universities"],
        },
        "notes": "Generated by TALASH automated profile synthesizer. Please review and update status to 'active'.",
    }


def generate_department_profile(
    department_name: str,
    sample_jd_text: Optional[str] = None,
    slug: Optional[str] = None,
    save: bool = True,
    profiles_dir: Optional[Path] = None,
    anthropic_api_key: Optional[str] = None,
) -> DepartmentProfile:
    """
    Main entry point for automated department profile generation.
    1. Tries Anthropic Claude SDK.
    2. Falls back to OpenAI/Groq if Anthropic is unavailable.
    3. Falls back to deterministic heuristic generator if no LLMs are reachable.
    4. Validates the output through Pydantic DepartmentProfile schema.
    5. Flags status as 'draft'.
    6. Persists to disk if save=True and caches locally.
    """
    raw_data: Optional[Dict[str, Any]] = None
    target_slug = slug or _slugify(department_name)

    # 1. Primary: Try Anthropic Claude
    try:
        raw_data = generate_with_anthropic(
            department_name=department_name,
            sample_jd_text=sample_jd_text,
            api_key=anthropic_api_key,
        )
        logger.info("Successfully generated profile for '%s' using Anthropic Claude.", department_name)
    except Exception as anthropic_err:
        logger.warning("Anthropic profile generation failed/skipped: %s. Trying fallback...", anthropic_err)

    # 2. Secondary: Try OpenAI / Groq
    if raw_data is None:
        try:
            raw_data = generate_with_openai_fallback(
                department_name=department_name,
                sample_jd_text=sample_jd_text,
            )
            logger.info("Successfully generated profile for '%s' using OpenAI/Groq fallback.", department_name)
        except Exception as openai_err:
            logger.warning("OpenAI profile generation failed/skipped: %s. Using heuristic synthesizer...", openai_err)

    # 3. Tertiary: Heuristic synthesizer
    if raw_data is None:
        raw_data = generate_heuristic_draft(
            department_name=department_name,
            sample_jd_text=sample_jd_text,
        )

    # Ensure required identity fields
    raw_data["slug"] = target_slug
    raw_data["department_name"] = raw_data.get("department_name") or department_name
    raw_data["status"] = "draft"  # Strictly marked draft pending human approval

    # Validate against Pydantic model
    profile = DepartmentProfile.model_validate(raw_data)

    # Persist and cache
    if save:
        save_profile(profile, profiles_dir=profiles_dir)

    return profile
