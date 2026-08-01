"""
jd_parser.py  –  Module 11: Job Description (JD) Requirements Extractor
========================================================================
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
2. Extract all explicit technical skills, domain expertise, tools, and platforms mentioned in the job text into "required_skills".
3. Extract degree level (PhD, MS, BS) and required academic discipline (e.g. Computer Science, Electrical Engineering, Microelectronics, Cybersecurity).

JSON SCHEMA:
{
  "title": "Job Title or Position Name",
  "required_degree_level": "PhD / MS / BS",
  "required_discipline": ["Computer Science", "Software Engineering"],
  "min_experience_years": 3,
  "required_skills": ["Python", "Machine Learning", "PyTorch"],
  "preferred_skills": ["Docker", "Kubernetes"],
  "research_areas": ["Computer Vision", "NLP"],
  "publication_requirement": "At least 3 Scopus indexed papers",
  "other_requirements": []
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
    res = _empty_jd_dict()
    if isinstance(parsed, dict):
        res["title"] = str(parsed.get("title") or "Faculty Position").strip()
        res["required_degree_level"] = str(parsed.get("required_degree_level") or "BS").strip()

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

    # If required_skills came out empty from LLM, run heuristic skill extraction as supplement
    if not res["required_skills"]:
        res["required_skills"] = _extract_skills_from_text(raw_text)

    return res


def _extract_skills_from_text(text: str) -> List[str]:
    """Dynamic skill and domain phrase extractor from JD text."""
    skills = []
    text_clean = text.replace("\n", " ").strip()

    # Pattern 1: Explicit mentions after "skills in", "experience in", "knowledge of", "proficiency in", "expertise in", "requirements:"
    phrases = re.findall(
        r'(?:skills?|experience|knowledge|expertise|proficiency|requirements?|seeking|need)\s*(?:in|with|of|for|:)?\s*([A-Za-z0-9\+\#\s,\/\-\.\(\)]+?)(?:\.|\n|;|$|and\s+years|with\s+PhD|degree)',
        text_clean, re.IGNORECASE
    )
    for p in phrases:
        for chunk in re.split(r'[,;/]', p):
            chunk_clean = chunk.strip()
            # Exclude noise words
            if len(chunk_clean) > 2 and not any(nw in chunk_clean.lower() for nw in ["years", "experience", "degree", "position", "seeking", "looking", "candidate", "applicant"]):
                skills.append(chunk_clean)

    # Pattern 2: Known technical & academic domain vocabulary
    domain_keywords = [
        "Computer Vision", "Machine Learning", "Deep Learning", "Artificial Intelligence", "AI", "NLP",
        "Natural Language Processing", "Cybersecurity", "Information Security", "Network Security",
        "IoT", "Internet of Things", "Microelectronics", "Analog IC Design", "Digital Signal Processing",
        "Signal Processing", "Embedded Systems", "Robotics", "Data Science", "Software Engineering",
        "Cloud Computing", "Distributed Systems", "Wireless Communication", "Power Systems",
        "Python", "C++", "Java", "PyTorch", "TensorFlow", "FastAPI", "Docker", "Kubernetes", "SQL", "AutoCAD", "PLC"
    ]
    for kw in domain_keywords:
        if re.search(r'\b' + re.escape(kw.lower()) + r'\b', text.lower()):
            if kw not in skills:
                skills.append(kw)

    # Deduplicate preserving order
    seen = set()
    deduped = []
    for s in skills:
        s_norm = s.lower()
        if s_norm not in seen:
            seen.add(s_norm)
            deduped.append(s)

    return deduped[:10]


def _heuristic_parse_jd(jd_text: str) -> Dict[str, Any]:
    """Rule-based heuristic extractor when LLM is unavailable."""
    text_lower = jd_text.lower()

    # Title extraction
    title = "Faculty Position"
    lines = [l.strip() for l in jd_text.splitlines() if l.strip()]
    if lines:
        title = lines[0][:80]
        for line in lines[:5]:
            if any(kw in line.lower() for kw in ["assistant professor", "associate professor", "professor", "lecturer", "engineer", "lead", "specialist"]):
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
        min_exp = float(exp_match.group(1))

    # Disciplines
    disciplines = []
    common_disciplines = [
        "Computer Science", "Software Engineering", "Artificial Intelligence",
        "Data Science", "Electrical Engineering", "Cybersecurity", "Information Technology",
        "Machine Learning", "Computer Engineering", "Robotics", "Microelectronics", "Telecom"
    ]
    for disc in common_disciplines:
        if disc.lower() in text_lower:
            disciplines.append(disc)

    skills = _extract_skills_from_text(jd_text)

    return {
        "title": title,
        "required_degree_level": degree,
        "required_discipline": disciplines,
        "min_experience_years": min_exp,
        "required_skills": skills,
        "preferred_skills": [],
        "research_areas": disciplines,
        "publication_requirement": "Peer-reviewed journal publications" if "publication" in text_lower or "research" in text_lower else None,
        "other_requirements": [],
    }
