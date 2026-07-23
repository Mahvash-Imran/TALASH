"""
timeline_builder.py  –  Module 9 Helper: Unified Date Normalization, Overlaps, Gaps & Progression
====================================================================================================

WHY THIS FILE EXISTS
--------------------
Provides deterministic logic for Module 9 (Professional Experience & Skill Alignment Analysis):
  1. Date Normalizer: Converts mixed date strings ('17-Sep', '23-Aug', '2024', '2011-09', 'Present') to standard YYYY-MM format.
  2. Unified Timeline Generator: Combines education and employment events chronologically.
  3. Overlap Detector & Classifier: Detects concurrent periods, classifying as 'Acceptable' (e.g. TA during MS) vs 'Suspicious'.
  4. Gap Detector & Justifier: Flags gaps > 3 months between professional activities and cross-references active degree/publication activity.
  5. Career Progression Profiler: Tracks career rank evolution (Lecturer -> Assistant Professor -> Associate Professor).
"""

import datetime
import re
from typing import Any, Dict, List, Optional, Tuple


def standardize_date(raw_date: Any, default_year: Optional[int] = None, is_end: bool = False) -> Tuple[Optional[str], Optional[datetime.date]]:
    """
    Standardizes mixed date inputs into (YYYY-MM string, datetime.date object).
    Handles:
      - 'Present', 'Currently', 'Till Date', 'Ongoing' -> current date
      - '17-Sep', '23-Aug' (yy-Mon) -> '2017-09', '2023-08'
      - '2024', '2011' -> '2024-01' (or '2024-12' if is_end)
      - '2011-09', '09/2011' -> '2011-09'
      - 'Sept 2017', 'August 2023' -> '2017-09', '2023-08'
    """
    if raw_date is None or str(raw_date).strip().lower() in ("nan", "none", "", "null"):
        return None, None

    s = str(raw_date).strip()

    # Handle 'Present' / 'Current'
    if re.search(r"\b(present|currently|till\s+date|ongoing|continue|now)\b", s, re.IGNORECASE):
        now = datetime.date.today()
        return now.strftime("%Y-%m"), now

    # Month mapping dictionary
    month_map = {
        "jan": 1, "january": 1,
        "feb": 2, "february": 2,
        "mar": 3, "march": 3,
        "apr": 4, "april": 4,
        "may": 5,
        "jun": 6, "june": 6,
        "jul": 7, "july": 7,
        "aug": 8, "august": 8,
        "sep": 9, "sept": 9, "september": 9,
        "oct": 10, "october": 10,
        "nov": 11, "november": 11,
        "dec": 12, "december": 12,
    }

    # Format 1: '17-Sep' or '23-Aug' or 'Sep-17' (YY-Mon or Mon-YY)
    m1 = re.match(r"^(\d{2})[\/\-\s]+([a-zA-Z]{3,9})$", s)
    if m1:
        part1, part2 = m1.group(1), m1.group(2).lower()
        if part2 in month_map:
            yy = int(part1)
            year = 2000 + yy if yy <= 50 else 1900 + yy
            month = month_map[part2]
            dt = datetime.date(year, month, 1)
            return dt.strftime("%Y-%m"), dt

    m1_rev = re.match(r"^([a-zA-Z]{3,9})[\/\-\s]+(\d{2})$", s)
    if m1_rev:
        part1, part2 = m1_rev.group(1).lower(), m1_rev.group(2)
        if part1 in month_map:
            yy = int(part2)
            year = 2000 + yy if yy <= 50 else 1900 + yy
            month = month_map[part1]
            dt = datetime.date(year, month, 1)
            return dt.strftime("%Y-%m"), dt

    # Format 2: 'Sep 2017' or 'September 2017'
    m2 = re.match(r"^([a-zA-Z]{3,9})[\/\-\s]+(\d{4})$", s)
    if m2:
        mon_str, yr_str = m2.group(1).lower(), int(m2.group(2))
        if mon_str in month_map:
            month = month_map[mon_str]
            dt = datetime.date(yr_str, month, 1)
            return dt.strftime("%Y-%m"), dt

    # Format 3: '2017 Sep' or '2017-09' or '09/2017'
    m3 = re.match(r"^(\d{4})[\/\-\s]+(\d{1,2})$", s)
    if m3:
        yr_str, month = int(m3.group(1)), int(m3.group(2))
        if 1 <= month <= 12:
            dt = datetime.date(yr_str, month, 1)
            return dt.strftime("%Y-%m"), dt

    m3_rev = re.match(r"^(\d{1,2})[\/\-\s]+(\d{4})$", s)
    if m3_rev:
        month, yr_str = int(m3_rev.group(1)), int(m3_rev.group(2))
        if 1 <= month <= 12:
            dt = datetime.date(yr_str, month, 1)
            return dt.strftime("%Y-%m"), dt

    # Format 4: '2024' (Year only)
    m4 = re.match(r"^(\d{4})$", s)
    if m4:
        yr_str = int(m4.group(1))
        month = 12 if is_end else 1
        dt = datetime.date(yr_str, month, 1)
        return dt.strftime("%Y-%m"), dt

    # Format 5: '2024-09-15' (Full ISO date)
    m5 = re.match(r"^(\d{4})[\/\-](\d{1,2})[\/\-](\d{1,2})$", s)
    if m5:
        yr, mo, dy = int(m5.group(1)), int(m5.group(2)), int(m5.group(3))
        if 1 <= mo <= 12:
            dt = datetime.date(yr, mo, min(dy, 28))
            return dt.strftime("%Y-%m"), dt

    # Fallback if default_year provided
    if default_year:
        month = 12 if is_end else 1
        dt = datetime.date(default_year, month, 1)
        return dt.strftime("%Y-%m"), dt

    return None, None


def build_candidate_timeline(
    candidate_id: str,
    education_records: List[Dict[str, Any]],
    experience_records: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Combines education and experience records into a single chronologically sorted timeline.
    Each event dictionary contains:
      event_type ('education' or 'experience')
      title (degree name or job title)
      organization (institution or company name)
      start_str, start_date (YYYY-MM and datetime.date)
      end_str, end_date (YYYY-MM and datetime.date)
      is_current (bool)
      raw_record (original dict)
    """
    timeline = []

    # Add education events
    for edu in education_records:
        deg = str(edu.get("degree") or edu.get("level") or "Degree").strip()
        inst = str(edu.get("institution") or "University").strip()
        sy_raw = edu.get("start_year")
        ey_raw = edu.get("end_year")

        s_str, s_dt = standardize_date(sy_raw, is_end=False)
        e_str, e_dt = standardize_date(ey_raw, is_end=True)

        if not s_dt and e_dt:
            # If start date missing, estimate start as 2 years before end date
            s_dt = datetime.date(max(e_dt.year - 2, 1990), 1, 1)
            s_str = s_dt.strftime("%Y-%m")

        if s_dt or e_dt:
            timeline.append({
                "candidate_id": candidate_id,
                "event_type": "education",
                "title": deg,
                "organization": inst,
                "start_str": s_str or "Unknown",
                "start_date": s_dt,
                "end_str": e_str or "Present",
                "end_date": e_dt or datetime.date.today(),
                "is_current": (ey_raw is None or str(ey_raw).strip().lower() in ("present", "ongoing", "")),
                "raw_record": edu,
            })

    # Add experience events
    for exp in experience_records:
        jtitle = str(exp.get("job_title") or "Position").strip()
        org = str(exp.get("organization") or "Organization").strip()
        s_raw = exp.get("start_date")
        e_raw = exp.get("end_date")

        s_str, s_dt = standardize_date(s_raw, is_end=False)
        e_str, e_dt = standardize_date(e_raw, is_end=True)

        if s_dt or e_dt:
            timeline.append({
                "candidate_id": candidate_id,
                "event_type": "experience",
                "title": jtitle,
                "organization": org,
                "start_str": s_str or "Unknown",
                "start_date": s_dt,
                "end_str": e_str or "Present",
                "end_date": e_dt or datetime.date.today(),
                "is_current": (e_raw is None or str(e_raw).strip().lower() in ("present", "currently", "till date", "ongoing", "")),
                "raw_record": exp,
            })

    # Sort events chronologically by start_date (fallback to end_date)
    timeline.sort(key=lambda x: (x["start_date"] or datetime.date(1970, 1, 1), x["end_date"] or datetime.date(1970, 1, 1)))

    return timeline


def detect_overlaps(timeline_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Detects overlapping time periods between timeline events.
    Classifies overlap as:
      - 'Acceptable': TA/RA/Lecturer concurrent with MS/PhD studies, or part-time roles.
      - 'Suspicious': Concurrent full-time dual jobs or multi-institution conflicts.
    """
    overlaps = []
    n = len(timeline_events)

    for i in range(n):
        e1 = timeline_events[i]
        s1, e1_end = e1["start_date"], e1["end_date"]
        if not s1 or not e1_end:
            continue

        for j in range(i + 1, n):
            e2 = timeline_events[j]
            s2, e2_end = e2["start_date"], e2["end_date"]
            if not s2 or not e2_end:
                continue

            # Check interval overlap: max(s1, s2) < min(e1_end, e2_end)
            overlap_start = max(s1, s2)
            overlap_end = min(e1_end, e2_end)

            if overlap_start < overlap_end:
                # Determine overlap duration in months
                months = (overlap_end.year - overlap_start.year) * 12 + (overlap_end.month - overlap_start.month)
                if months <= 1:
                    continue  # Ignore minor 1-month boundary transition overlaps

                # Determine overlap nature
                t1, t2 = e1["event_type"], e2["event_type"]
                title1_low = e1["title"].lower()
                title2_low = e2["title"].lower()

                # Rule 1: Education + Student Academic Job (TA/RA/Lecturer/Research Assistant) -> Acceptable
                academic_roles = ("teaching assistant", "ta", "research assistant", "ra", "lecturer", "tutor", "lab engineer", "student worker")
                is_ta_ra = any(r in title1_low for r in academic_roles) or any(r in title2_low for r in academic_roles)
                is_edu_job = (t1 == "education" and t2 == "experience") or (t1 == "experience" and t2 == "education")

                if is_edu_job and is_ta_ra:
                    classification = "Acceptable"
                    reason = "Academic employment (TA/RA/Lecturer) concurrent with degree studies."
                elif t1 == "education" and t2 == "education":
                    classification = "Suspicious"
                    reason = "Concurrent enrolment in multiple degree programs."
                elif t1 == "experience" and t2 == "experience":
                    if is_ta_ra:
                        classification = "Acceptable"
                        reason = "Concurrent academic teaching or assistant roles."
                    else:
                        classification = "Suspicious"
                        reason = "Concurrent multi-job employment."
                else:
                    classification = "Suspicious"
                    reason = "Full-time employment concurrent with degree studies."

                overlaps.append({
                    "candidate_id": e1["candidate_id"],
                    "event1_title": f"[{e1['event_type'].upper()}] {e1['title']} @ {e1['organization']}",
                    "event2_title": f"[{e2['event_type'].upper()}] {e2['title']} @ {e2['organization']}",
                    "overlap_start": overlap_start.strftime("%Y-%m"),
                    "overlap_end": overlap_end.strftime("%Y-%m"),
                    "duration_months": months,
                    "classification": classification,
                    "reason": reason,
                })

    return overlaps


def detect_experience_gaps(
    timeline_events: List[Dict[str, Any]],
    candidate_pubs: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Detects timeline gaps > 3 months between professional activities.
    Checks whether the gap is justified by:
      - Active degree enrolment
      - Research publication outputs during the gap period
    """
    gaps = []
    # Filter to experience events only (or experience + education)
    prof_events = [e for e in timeline_events if e["start_date"] and e["end_date"]]
    if len(prof_events) < 2:
        return gaps

    # Collect publication years for research justification
    pub_years = set()
    if candidate_pubs:
        for p in candidate_pubs:
            y = p.get("year")
            try:
                if y and not str(y).startswith("nan"):
                    pub_years.add(int(float(y)))
            except (ValueError, TypeError):
                pass

    for i in range(len(prof_events) - 1):
        prev_end = prof_events[i]["end_date"]
        next_start = prof_events[i + 1]["start_date"]

        if next_start > prev_end:
            months_gap = (next_start.year - prev_end.year) * 12 + (next_start.month - prev_end.month)
            if months_gap > 3:
                # Check for justification
                # 1. Degree enrolment during gap?
                degree_during_gap = [
                    e["title"] for e in prof_events
                    if e["event_type"] == "education"
                    and e["start_date"] <= prev_end and e["end_date"] >= next_start
                ]
                # 2. Publication activity during gap?
                gap_years = set(range(prev_end.year, next_start.year + 1))
                active_pubs = gap_years.intersection(pub_years)

                if degree_during_gap:
                    status = "Justified"
                    reason = f"Enrolled in {', '.join(degree_during_gap)} during gap."
                elif active_pubs:
                    status = "Justified"
                    reason = f"Active research publication output ({len(active_pubs)} year(s) active) during gap."
                else:
                    status = "Unjustified Gap"
                    reason = f"No recorded degree enrolment or employment for {months_gap} months."

                gaps.append({
                    "candidate_id": prof_events[i]["candidate_id"],
                    "gap_start": prev_end.strftime("%Y-%m"),
                    "gap_end": next_start.strftime("%Y-%m"),
                    "duration_months": months_gap,
                    "prev_activity": f"[{prof_events[i]['event_type'].upper()}] {prof_events[i]['title']} @ {prof_events[i]['organization']}",
                    "next_activity": f"[{prof_events[i+1]['event_type'].upper()}] {prof_events[i+1]['title']} @ {prof_events[i+1]['organization']}",
                    "justification_status": status,
                    "reason": reason,
                })

    return gaps


def assess_career_progression(experience_events: List[Dict[str, Any]]) -> Tuple[str, float]:
    """
    Evaluates career progression trajectory and total experience duration in years.
    Returns: (progression_label, total_experience_years)
    """
    if not experience_events:
        return "No Professional Experience Listed", 0.0

    # Calculate total non-overlapping experience months
    months_sum = 0
    academic_ranks = []

    rank_keywords = {
        "lecturer": 1,
        "lab engineer": 1,
        "assistant professor": 2,
        "associate professor": 3,
        "professor": 4,
        "chair": 4,
        "dean": 5,
        "director": 4,
        "junior": 1,
        "engineer": 2,
        "senior": 3,
        "lead": 4,
        "manager": 4,
    }

    for exp in experience_events:
        s, e = exp["start_date"], exp["end_date"]
        if s and e and e >= s:
            m = (e.year - s.year) * 12 + (e.month - s.month)
            months_sum += max(m, 1)

        title_low = exp["title"].lower()
        for kw, rank in rank_keywords.items():
            if kw in title_low:
                academic_ranks.append((s or datetime.date(1970, 1, 1), rank, exp["title"]))
                break

    total_years = round(months_sum / 12.0, 1)

    if not academic_ranks:
        if total_years >= 5:
            return "Steady Career Path", total_years
        return "Early Career", total_years

    # Check rank trend
    academic_ranks.sort(key=lambda x: x[0])
    ranks_only = [r[1] for r in academic_ranks]

    is_upward = any(ranks_only[i] < ranks_only[i + 1] for i in range(len(ranks_only) - 1))
    has_high_rank = any(r >= 3 for r in ranks_only)

    if is_upward or has_high_rank:
        trajectory = "Upward Progression"
    elif len(set(ranks_only)) == 1:
        trajectory = "Stable Rank"
    else:
        trajectory = "Mixed Progression"

    return trajectory, total_years
