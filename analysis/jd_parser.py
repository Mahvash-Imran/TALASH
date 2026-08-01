"""
jd_parser.py  –  Module 11: Job Description (JD) Requirements Extractor
========================================================================

WHY THIS FILE EXISTS
--------------------
Extracts structured requirements from raw Job Description text (or file content)
using the Groq/OpenAI LLM client, with a deterministic rule-based fallback mode.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_JD_SYSTEM_PROMPT = """You are an expert HR and Academic Job Description analyzer for the TALASH faculty recruitment system.
Your task is to parse a Job Description and extract structured requirements into a valid JSON object.

STRICT RULES:
1. Return ONLY a valid JSON object. No markdown fences, no explanation.
2. If a requirement is not specified, set it to null or an empty array [].
3. For min_experience_years, extract a number (float or int). If not stated, set to 0.

JSON SCHEMA:
{
  "title": "Job Title or Position Name",
  "required_degree_level": "PhD / MS / BS / Bachelors / Master / etc.",
  "required_discipline": ["Computer Science", "AI", "Software Engineering"],
  "min_experience_years": 3,
  "required_skills": ["Python", "Machine Learning", "PyTorch"],
  "preferred_skills": ["Docker", "Kubernetes"],
  "research_areas": ["Computer Vision", "NLP"],
  "publication_requirement": "At least 3 Scopus indexed papers",
  "other_requirements": ["HEC recognized PhD", "Postdoc experience"]
}
"""

_JD_USER_PROMPT_TEMPLATE = """Parse the following Job Description text into structured requirements JSON:

JOB DESCRIPTION:
{jd_text}
"""


def parse_job_description(
    jd_text: str,
    api_key: Optional[str] = None,
    model: str = "llama-3.3-70b-versatile",
    base_url: Optional[str] = None,
    skip_llm: bool = False,
) -> Dict[str, Any]:
    """
    Parses raw Job Description text into a structured JSON dict of requirements.
    Uses Groq/OpenAI LLM if available; falls back to heuristic rule extraction if skip_llm=True or LLM fails.
    """
    if not jd_text or not jd_text.strip():
        return _empty_jd_dict("Untitled Position")

    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    is_valid_key = bool(key and not str(key).startswith("your_") and len(str(key).strip()) > 20)

    if skip_llm or not is_valid_key:
        return _heuristic_parse_jd(jd_text)

    try:
        from openai import OpenAI
        b_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
        client = OpenAI(api_key=key, base_url=b_url)

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _JD_SYSTEM_PROMPT},
                {"role": "user", "content": _JD_USER_PROMPT_TEMPLATE.format(jd_text=jd_text)},
            ],
            temperature=0.1,
            max_tokens=1000,
        )
        content = resp.choices[0].message.content or ""
        cleaned = _clean_json_text(content)
        parsed = json.loads(cleaned)
        return _normalize_jd_dict(parsed, jd_text)
    except Exception as e:
        logger.warning(f"LLM JD parsing failed: {e}. Falling back to heuristic extraction.")
        return _heuristic_parse_jd(jd_text)


def _clean_json_text(text: str) -> str:
    """Strip markdown backticks and clean JSON string."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _empty_jd_dict(title: str = "Faculty Position") -> Dict[str, Any]:
    return {
        "title": title,
        "required_degree_level": "BS",
        "required_discipline": [],
        "min_experience_years": 0,
        "required_skills": [],
        "preferred_skills": [],
        "research_areas": [],
        "publication_requirement": None,
        "other_requirements": [],
    }


def _normalize_jd_dict(parsed: Dict[str, Any], raw_text: str) -> Dict[str, Any]:
    """Ensures all required fields exist and have correct types."""
    res = _empty_jd_dict()
    if isinstance(parsed, dict):
        res["title"] = str(parsed.get("title") or "Faculty Position").strip()
        res["required_degree_level"] = str(parsed.get("required_degree_level") or "").strip()
        
        req_disc = parsed.get("required_discipline")
        if isinstance(req_disc, list):
            res["required_discipline"] = [str(x).strip() for x in req_disc if str(x).strip()]
        elif isinstance(req_disc, str) and req_disc:
            res["required_discipline"] = [req_disc.strip()]

        try:
            res["min_experience_years"] = float(parsed.get("min_experience_years") or 0)
        except (ValueError, TypeError):
            res["min_experience_years"] = 0

        req_sk = parsed.get("required_skills")
        if isinstance(req_sk, list):
            res["required_skills"] = [str(x).strip() for x in req_sk if str(x).strip()]

        pref_sk = parsed.get("preferred_skills")
        if isinstance(pref_sk, list):
            res["preferred_skills"] = [str(x).strip() for x in pref_sk if str(x).strip()]

        r_areas = parsed.get("research_areas")
        if isinstance(r_areas, list):
            res["research_areas"] = [str(x).strip() for x in r_areas if str(x).strip()]

        res["publication_requirement"] = parsed.get("publication_requirement")
        
        other_req = parsed.get("other_requirements")
        if isinstance(other_req, list):
            res["other_requirements"] = [str(x).strip() for x in other_req if str(x).strip()]

    return res


def _heuristic_parse_jd(jd_text: str) -> Dict[str, Any]:
    """Rule-based heuristic extractor when LLM is unavailable."""
    text_lower = jd_text.lower()
    
    # Title extraction
    title = "Faculty Position"
    lines = [l.strip() for l in jd_text.splitlines() if l.strip()]
    if lines:
        title = lines[0][:80]
        for line in lines[:5]:
            if any(kw in line.lower() for kw in ["assistant professor", "associate professor", "professor", "lecturer", "engineer", "lead"]):
                title = line
                break

    # Degree level
    degree = "BS"
    if "phd" in text_lower or "ph.d" in text_lower or "doctorate" in text_lower:
        degree = "PhD"
    elif "ms" in text_lower or "m.s" in text_lower or "mphil" in text_lower or "master" in text_lower:
        degree = "MS"

    # Min experience years
    min_exp = 0
    exp_match = re.search(r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:experience|exp)', text_lower)
    if exp_match:
        min_exp = int(exp_match.group(1))

    # Disciplines
    disciplines = []
    common_disciplines = [
        "Computer Science", "Software Engineering", "Artificial Intelligence",
        "Data Science", "Electrical Engineering", "Cybersecurity", "Information Technology",
        "Machine Learning", "Computer Engineering", "Robotics"
    ]
    for disc in common_disciplines:
        if disc.lower() in text_lower:
            disciplines.append(disc)

    # Hard skills heuristics
    skills = []
    common_skills = [
        "Python", "C++", "Java", "PyTorch", "TensorFlow", "Machine Learning",
        "Deep Learning", "Computer Vision", "NLP", "SQL", "Docker", "Kubernetes",
        "Cloud", "AWS", "Linux", "Git", "React", "Node.js", "FastAPI", "Data Structures"
    ]
    for skill in common_skills:
        if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text_lower):
            skills.append(skill)

    return {
        "title": title,
        "required_degree_level": degree,
        "required_discipline": disciplines,
        "min_experience_years": min_exp,
        "required_skills": skills[:8],
        "preferred_skills": skills[8:12],
        "research_areas": disciplines,
        "publication_requirement": "Peer-reviewed journal publications" if "publication" in text_lower or "research" in text_lower else None,
        "other_requirements": [],
    }
