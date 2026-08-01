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


def _extract_min_experience(text: str) -> float:
    """
    Robustly extracts minimum experience requirement from JD text.
    Handles patterns like:
      - 'Minimum 8+ years of post-qualification teaching or research experience'
      - 'at least 5 years of relevant experience'
      - '8+ years experience in Computer Vision'
      - 'experience of 7 years'
      - 'minimum of 10 years post-PhD'
    """
    text_lower = text.lower()

    # Priority 1: 'experience of N years'
    m = re.search(r'experience\s+of\s+(\d+)\s*\+?\s*(?:years?|yrs?)', text_lower)
    if m:
        return float(m.group(1))

    # Priority 2: Explicit qualifier words right before N years
    m = re.search(
        r'(?:minimum|at least|minimum of|at least|over|more than|at-least)\s+(\d+)\s*\+?\s*(?:years?|yrs?)',
        text_lower
    )
    if m:
        return float(m.group(1))

    # Priority 3: N years near experience/post/qualification keywords
    m = re.search(
        r'(\d+)\s*\+?\s*(?:years?|yrs?)\s+'
        r'(?:of\s+)?(?:post.qualification|post-phd|relevant|teaching|research|industry|academic|'
        r'post\s+qualification|professional|work)\s*(?:experience|exp)',
        text_lower
    )
    if m:
        return float(m.group(1))

    # Priority 4: Broadest – any 'N+ years' or 'N years' close to the word 'experience'
    # Only take the FIRST / MINIMUM number to avoid grabbing CGPAs or years like "2020"
    candidates = []
    for m in re.finditer(r'(\d{1,2})\s*\+?\s*(?:years?|yrs?)\b', text_lower):
        num = int(m.group(1))
        if 1 <= num <= 40:  # Plausible experience range, avoids years like 2014
            # Check that 'experience' or 'exp' appears within 80 chars
            window = text_lower[max(0, m.start() - 80): m.end() + 80]
            if re.search(r'\bexperience\b|\bexp\b|\bpost', window):
                candidates.append(num)

    if candidates:
        return float(min(candidates))  # Return lowest = minimum requirement

    return 0.0


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

        # Always re-extract experience from raw text as a sanity-check / fallback
        if res["min_experience_years"] == 0:
            res["min_experience_years"] = _extract_min_experience(raw_text)

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
    """
    Extracts only specific technical skills and domain keywords from JD text.
    Avoids including long sentence fragments.
    """
    # First pass: known technical & academic domain vocabulary
    domain_keywords = [
        "Computer Vision", "Machine Learning", "Deep Learning", "Artificial Intelligence",
        "Natural Language Processing", "NLP", "Cybersecurity", "Information Security",
        "Network Security", "IoT", "Internet of Things", "Microelectronics", "Analog IC Design",
        "Digital Signal Processing", "Signal Processing", "Embedded Systems", "Robotics",
        "Data Science", "Software Engineering", "Cloud Computing", "Distributed Systems",
        "Wireless Communication", "Power Systems", "VLSI", "Verilog", "FPGA",
        "Python", "C++", "Java", "PyTorch", "TensorFlow", "OpenCV", "FastAPI",
        "Docker", "Kubernetes", "SQL", "AutoCAD", "PLC", "MATLAB", "R",
        "Image Processing", "Object Detection", "Semantic Segmentation", "Reinforcement Learning",
        "Penetration Testing", "Cryptography", "Blockchain", "Big Data", "Hadoop", "Spark",
        "5G", "LTE", "OFDM", "Antenna Design", "RF Engineering",
    ]

    found = []
    for kw in domain_keywords:
        if re.search(r'\b' + re.escape(kw) + r'\b', text, re.IGNORECASE):
            found.append(kw)

    # Second pass: explicit skill list items (short, specific phrases after skill-indicator words)
    # Only take phrases that are short (< 5 words) and look like skill names
    skill_indicators = re.findall(
        r'(?:skills?|tools?|technologies?|proficiency in|expertise in|knowledge of)\s*[:\-]?\s*([^\n]{3,120})',
        text, re.IGNORECASE
    )
    for phrase in skill_indicators:
        for chunk in re.split(r'[,;/]', phrase):
            chunk_clean = chunk.strip().rstrip('.')
            words = chunk_clean.split()
            # Only take short, clean skill names (1-4 words, no long sentences, at least 2 chars)
            if 2 <= len(chunk_clean) <= 40 and 1 <= len(words) <= 4 and not any(
                nw in chunk_clean.lower() for nw in [
                    "years", "experience", "degree", "position", "seeking",
                    "candidate", "applicant", "university", "teaching",
                ]
            ):
                if chunk_clean and chunk_clean not in found:
                    found.append(chunk_clean)

    # Deduplicate preserving order
    seen = set()
    deduped = []
    for s in found:
        s_norm = s.lower()
        if s_norm not in seen:
            seen.add(s_norm)
            deduped.append(s)

    return deduped[:15]


def _extract_title_from_jd(text: str) -> str:
    """Extracts a clean job title from JD text."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    title = "Faculty Position"

    # Skip common header/label lines and find first meaningful title line
    for line in lines[:8]:
        lower = line.lower()
        # Skip lines that are just labels
        if lower.startswith("job description:") or lower.startswith("job title:") or lower.startswith("position:"):
            # Strip the label and use the remainder if it exists
            remainder = re.sub(r'^(job description|job title|position)\s*:\s*', '', line, flags=re.IGNORECASE).strip()
            if remainder:
                title = remainder
                break
        elif any(kw in lower for kw in [
            "assistant professor", "associate professor", "professor", "lecturer",
            "engineer", "specialist", "researcher", "faculty", "lead", "instructor"
        ]):
            title = line[:100]
            break

    # Clean up any remaining "JOB DESCRIPTION:" prefix
    title = re.sub(r'^(?:job description|job title|position)\s*:\s*', '', title, flags=re.IGNORECASE).strip()
    return title or "Faculty Position"


def _heuristic_parse_jd(jd_text: str) -> Dict[str, Any]:
    """Rule-based heuristic extractor when LLM is unavailable."""
    text_lower = jd_text.lower()

    title = _extract_title_from_jd(jd_text)

    # Degree level
    degree = "BS"
    if "phd" in text_lower or "ph.d" in text_lower or "doctorate" in text_lower:
        degree = "PhD"
    elif re.search(r'\bms\b|\bm\.s\b|mphil|master', text_lower):
        degree = "MS"

    # Min experience years — use robust extractor
    min_exp = _extract_min_experience(jd_text)

    # Disciplines
    disciplines = []
    common_disciplines = [
        "Computer Science", "Software Engineering", "Artificial Intelligence",
        "Data Science", "Electrical Engineering", "Cybersecurity", "Information Technology",
        "Machine Learning", "Computer Engineering", "Robotics", "Microelectronics", "Telecom",
        "Electronics", "Information Security",
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
        "publication_requirement": (
            "Peer-reviewed journal publications"
            if "publication" in text_lower or "research" in text_lower
            else None
        ),
        "other_requirements": [],
    }
