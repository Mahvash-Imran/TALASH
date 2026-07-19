"""
topic_verifier.py  –  Module 7 Helper: Research Taxonomy, Entropy & Trend Analysis
===================================================================================

WHY THIS FILE EXISTS
--------------------
Provides deterministic logic for Module 7 (Topic Variability & Research Breadth):
  1. Taxonomy Reference Dictionary & Keyword Matcher (TF-IDF / Rule-based).
  2. Shannon Entropy & Diversity Score Calculation.
  3. Temporal Trend Analysis (year-by-year theme evolution).
  4. Research Profile Classification: Specialist (>70%), Focused (50-70%), Interdisciplinary (<=50%).
"""

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from rapidfuzz import fuzz

# ---------------------------------------------------------------------------
# Predefined Computer Science & Engineering Academic Taxonomy
# ---------------------------------------------------------------------------
RESEARCH_TAXONOMY = {
    "Natural Language Processing": [
        "nlp", "natural language", "text classification", "sentiment", "sentiment analysis",
        "named entity", "translation", "transformer", "bert", "gpt", "large language model",
        "llm", "question answering", "summarization", "text mining", "linguistic", "word embedding",
        "speech recognition", "language model", "parsing", "tokenization", "text processing"
    ],
    "Computer Vision & Image Processing": [
        "computer vision", "image processing", "object detection", "segmentation", "facial recognition",
        "action recognition", "convolutional", "cnn", "image classification", "pattern recognition",
        "visual", "biometric", "video analysis", "medical imaging", "motion estimation", "mhi", "camera",
        "feature extraction", "edge detection", "deep vision"
    ],
    "Machine Learning & Artificial Intelligence": [
        "machine learning", "deep learning", "neural network", "artificial intelligence", "supervised",
        "unsupervised", "reinforcement learning", "classification", "clustering", "regression", "svm",
        "random forest", "feature selection", "optimization", "predictive model", "autoencoder", "gan",
        "hyperparameter", "ensemble learning", "transfer learning"
    ],
    "Wireless Networks & IoT": [
        "wireless", "sensor network", "wsn", "iot", "internet of things", "5g", "6g", "mimo",
        "cf-mmimo", "xl-mimo", "beamforming", "cellular", "telecommunication", "routing protocol",
        "radio", "channel estimation", "antenna", "signal transmission", "network protocol", "ad-hoc",
        "vehicular", "vanet", "satellite", "non-terrestrial"
    ],
    "Cybersecurity & Privacy": [
        "security", "cybersecurity", "privacy", "encryption", "cryptography", "malware", "intrusion detection",
        "authentication", "blockchain", "firewall", "cyber attack", "vulnerability", "steganography",
        "information hiding", "access control", "threat", "phishing", "forensics"
    ],
    "Biomedical & Health Informatics": [
        "biomedical", "healthcare", "medical", "eeg", "ecg", "emg", "heart rate", "biosignal",
        "cancer", "disease", "patient", "clinical", "health monitoring", "telemedicine", "bio-inspired",
        "epilepsy", "seizure", "retinal", "edema"
    ],
    "Software Engineering & Web Technologies": [
        "software engineering", "software testing", "code quality", "refactoring", "agile", "devops",
        "web development", "api", "microservices", "software architecture", "bug detection",
        "source code", "version control", "software maintenance"
    ],
    "Robotics & Automation": [
        "robotics", "robot", "autonomous", "drone", "uav", "control system", "navigation",
        "actuator", "kinematics", "slam", "industrial automation", "mechatronics", "path planning"
    ],
    "Energy & Renewable Systems": [
        "energy", "solar", "photovoltaic", "wind turbine", "power system", "battery", "grid",
        "smart grid", "power electronics", "energy storage", "pyrolysis", "bio-oil", "fuel cell"
    ],
    "Data Science & Big Data": [
        "big data", "data analytics", "data mining", "recommendation system", "recommender",
        "database", "data warehouse", "spark", "hadoop", "time series", "social network analysis"
    ],
    "General Computer Science / Other": []
}


# ---------------------------------------------------------------------------
# Fallback Keyword & Taxonomy Matcher (TF-IDF / Rule-Based)
# ---------------------------------------------------------------------------

def classify_paper_theme_rule_based(
    title: str,
    venue: str = "",
) -> Tuple[str, Optional[str], List[str]]:
    """
    Classifies a paper into (primary_theme, secondary_theme, keywords)
    using taxonomy rule matching.
    """
    combined_text = f"{title} {venue}".lower()
    scores: Dict[str, float] = {}
    found_keywords: Dict[str, List[str]] = {}

    for theme, keywords in RESEARCH_TAXONOMY.items():
        if not keywords:
            continue
        score = 0.0
        kw_matched = []
        for kw in keywords:
            # Check exact phrase or word boundary match
            if re.search(r"\b" + re.escape(kw) + r"\b", combined_text):
                score += 2.0 if len(kw) > 3 else 1.0
                kw_matched.append(kw)

        if score > 0:
            scores[theme] = score
            found_keywords[theme] = kw_matched

    if not scores:
        return "General Computer Science / Other", None, []

    sorted_themes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    primary = sorted_themes[0][0]
    secondary = sorted_themes[1][0] if len(sorted_themes) > 1 and sorted_themes[1][1] >= 1.0 else None
    matched_kws = found_keywords.get(primary, [])

    return primary, secondary, matched_kws


# ---------------------------------------------------------------------------
# Shannon Entropy & Diversity Score Calculation
# ---------------------------------------------------------------------------

def calculate_shannon_entropy(theme_counts: Dict[str, int]) -> Tuple[float, float]:
    """
    Computes Shannon Entropy H and Normalized Shannon Entropy H_norm:
      H = - sum(p_i * ln(p_i))
      H_norm = H / ln(total_distinct_themes)  (scaled to 0.0 - 1.0)
    Returns (raw_entropy, normalized_entropy).
    """
    total = sum(theme_counts.values())
    if total <= 1 or len(theme_counts) <= 1:
        return 0.0, 0.0

    entropy = 0.0
    for count in theme_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log(p)

    k = len(theme_counts)
    max_entropy = math.log(k) if k > 1 else 1.0
    normalized_entropy = round(entropy / max_entropy, 3)

    return round(entropy, 3), normalized_entropy


# ---------------------------------------------------------------------------
# Temporal Trend Analysis
# ---------------------------------------------------------------------------

def analyze_temporal_trend(
    publications: List[Dict[str, Any]],
    paper_themes: Dict[str, str]  # title -> theme lookup
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Groups publications by year and calculates the dominant theme per year.
    Returns (temporal_trend_list, trend_pattern_summary).
    """
    year_map: Dict[int, List[str]] = {}

    for pub in publications:
        title = str(pub.get("title") or "").strip()
        year_str = str(pub.get("year") or "").strip()

        # Parse year
        m = re.search(r"\b(19\d\d|20\d\d)\b", year_str)
        if not m:
            continue
        year = int(m.group(1))

        theme = paper_themes.get(title, "General Computer Science / Other")
        year_map.setdefault(year, []).append(theme)

    if not year_map:
        return [], "Insufficient Temporal Data"

    sorted_years = sorted(year_map.keys())
    temporal_trend = []
    dominant_per_year = []

    for yr in sorted_years:
        themes = year_map[yr]
        counts = Counter(themes)
        top_theme = counts.most_common(1)[0][0]
        temporal_trend.append({
            "year": yr,
            "publication_count": len(themes),
            "dominant_theme": top_theme,
            "theme_counts": dict(counts)
        })
        dominant_per_year.append(top_theme)

    unique_dominant = set(dominant_per_year)

    if len(unique_dominant) == 1:
        trend_pattern = "Stable Focus"
    elif len(unique_dominant) <= 2:
        trend_pattern = "Evolving Focus"
    else:
        trend_pattern = "Shifting / Interdisciplinary Focus"

    return temporal_trend, trend_pattern


# ---------------------------------------------------------------------------
# Research Profile Classification
# ---------------------------------------------------------------------------

def classify_research_profile_type(
    total_pubs: int,
    dominant_percentage: float
) -> str:
    """
    Classifies candidate research profile type:
      - Specialist: > 70% of publications in dominant theme
      - Focused Researcher: 50% - 70% of publications in dominant theme
      - Interdisciplinary: <= 50% of publications in dominant theme
      - No Publications: 0 publications
    """
    if total_pubs == 0:
        return "No Publications"
    if dominant_percentage > 70.0:
        return "Specialist"
    if dominant_percentage >= 50.0:
        return "Focused Researcher"
    return "Interdisciplinary"
