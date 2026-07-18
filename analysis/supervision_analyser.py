"""
supervision_analyser.py  –  Module 4: Student Supervision Analysis
===================================================================

WHY THIS FILE EXISTS
--------------------
This module evaluates each candidate's academic mentoring contribution by:
  - Reading supervision.csv (produced by Module 1)
  - Counting main vs. co-supervisor roles for MS and PhD students
  - Cross-referencing supervised student names with co-author lists in
    publications.csv using fuzzy matching (rapidfuzz)
  - Generating a structured JSON record and LLM-written supervisory
    assessment paragraph per candidate

DESIGN DECISIONS
----------------
- Real-world situation: supervision data is almost never present in CVs
  (candidates either don't include it or supervisory roles are inferred
  from context). The module handles this gracefully by flagging
  "data_missing = True" and surfacing the candidate for follow-up email.

- Fuzzy matching threshold of 80 (out of 100) is used for student name
  matching against co-author strings. This catches partial names (e.g.
  "Ali Hassan" matching "M. Ali Hassan" or "Hassan, Ali").

- The supervision.csv may have placeholder rows (candidate_id present but
  all other fields NaN). These are treated the same as missing data.

- LLM assessment is generated even when data is missing — it produces a
  "no supervision data available" note flagging the candidate for follow-up.

- Checkpointing: results are saved after each candidate so a re-run after
  cancellation resumes from where it left off.

OUTPUT FILES
------------
  data/analysis/supervision_profiles.csv   – one row per candidate
  data/analysis/supervision_joint_pubs.csv – one row per matched joint paper
  data/analysis/supervision_profiles.xlsx  – two-sheet workbook
  data/analysis/supervision_report.txt     – human-readable plain-text report
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from rapidfuzz import fuzz, process as rfprocess

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fuzzy match threshold (0-100).  80 balances recall vs. false positives.
# ---------------------------------------------------------------------------
_FUZZY_THRESHOLD = 80

# ---------------------------------------------------------------------------
# Known co/main role synonyms that appear in CVs
# ---------------------------------------------------------------------------
_MAIN_ROLE_SYNONYMS = frozenset({
    "main", "primary", "principal", "supervisor", "main supervisor",
    "primary supervisor", "principal supervisor",
})
_CO_ROLE_SYNONYMS = frozenset({
    "co", "co-supervisor", "cosupervisor", "co supervisor",
    "external", "external supervisor", "joint", "joint supervisor",
    "co-guide", "associate supervisor",
})

# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

_SUPERVISION_SYSTEM = (
    "You are an academic evaluation expert for a university recruitment system. "
    "Given structured supervision statistics for a faculty candidate, write a "
    "concise, factual supervisory assessment. "
    "Return ONLY a valid JSON object with exactly two keys: "
    "\"supervisory_strength\" and \"supervisory_assessment\". "
    "supervisory_strength must be one of: "
    "\"Strong\", \"Moderate\", \"Limited\", \"Needs Clarification\". "
    "supervisory_assessment must be 2-4 sentences in third person, factual, no embellishment. "
    "If data is missing, set supervisory_strength to \"Needs Clarification\" and explain "
    "that supervision data was not found in the CV and a follow-up is required."
)

_SUPERVISION_USER_TEMPLATE = (
    "Assess this candidate's supervision record and return JSON.\n\n"
    "CANDIDATE ID: {candidate_id}\n\n"
    "SUPERVISION SUMMARY:\n"
    "  MS students (main supervisor)  : {ms_main}\n"
    "  MS students (co-supervisor)    : {ms_co}\n"
    "  PhD students (main supervisor) : {phd_main}\n"
    "  PhD students (co-supervisor)   : {phd_co}\n"
    "  Total joint papers with students: {total_joint}\n"
    "  Typical authorship in joint papers: {typical_role}\n"
    "  Data missing from CV            : {data_missing}\n\n"
    "SUGGESTED STRENGTH (rule-based): {rule_label}\n\n"
    "Return JSON:"
)


# ---------------------------------------------------------------------------
# Rule-based strength label (seeds LLM)
# ---------------------------------------------------------------------------

def _rule_based_supervision_label(
    ms_main: int, ms_co: int, phd_main: int, phd_co: int,
    total_joint: int, data_missing: bool,
) -> str:
    if data_missing:
        return "Needs Clarification"
    total_main = ms_main + phd_main
    total_all  = total_main + ms_co + phd_co
    if total_all == 0:
        return "Needs Clarification"
    if phd_main >= 2 or total_main >= 5:
        return "Strong"
    if phd_main >= 1 or total_main >= 2 or total_all >= 4:
        return "Moderate"
    return "Limited"


# ---------------------------------------------------------------------------
# Student name → co-author cross-reference (Task 4.4)
# ---------------------------------------------------------------------------

def _normalize_name(name: str) -> str:
    """Strip punctuation, lower-case, collapse whitespace."""
    name = re.sub(r"[,.\-–]", " ", name)
    return " ".join(name.lower().split())


def find_joint_papers(
    student_name: str,
    candidate_publications: List[Dict[str, Any]],
    candidate_name: str,
    threshold: int = _FUZZY_THRESHOLD,
) -> List[Dict[str, Any]]:
    """
    Task 4.4: Check if `student_name` appears in the authors list of any
    publication belonging to this candidate. Uses rapidfuzz token_sort_ratio
    to handle partial / reordered name forms.

    Returns a list of matched paper records with authorship context.
    """
    if not student_name or str(student_name).lower() in ("nan", "none", ""):
        return []

    norm_student = _normalize_name(student_name)
    norm_candidate = _normalize_name(candidate_name)
    matches = []

    for pub in candidate_publications:
        authors_raw = str(pub.get("authors") or "")
        if not authors_raw or authors_raw.lower() in ("nan", "none", ""):
            continue

        # Split authors by common delimiters
        author_list = [a.strip() for a in re.split(r"[,;&]|and\b", authors_raw) if a.strip()]
        norm_author_list = [_normalize_name(a) for a in author_list]

        # Check if student name fuzzy-matches any individual author entry
        result = rfprocess.extractOne(
            norm_student,
            norm_author_list,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold,
        )
        if result is None:
            continue

        matched_author_str, score, matched_idx = result

        # Determine candidate's authorship position
        cand_pos = None
        student_pos = matched_idx + 1  # 1-indexed
        for idx, norm_auth in enumerate(norm_author_list):
            r = fuzz.token_sort_ratio(norm_candidate, norm_auth)
            if r >= threshold:
                cand_pos = idx + 1
                break

        # Determine candidate's role relative to student
        if cand_pos is None:
            candidate_position = "Unknown"
        elif cand_pos == 1:
            candidate_position = "First Author"
        elif cand_pos < student_pos:
            candidate_position = f"Author {cand_pos} (before student)"
        elif cand_pos > student_pos:
            candidate_position = f"Author {cand_pos} (after student)"
        else:
            candidate_position = f"Author {cand_pos}"

        # Check if candidate is last author (often = corresponding)
        if cand_pos == len(norm_author_list):
            candidate_position = "Corresponding/Last Author"

        matches.append({
            "paper_title":          pub.get("title"),
            "year":                 pub.get("year"),
            "venue":                pub.get("venue"),
            "student_name":         student_name,
            "student_matched_as":   author_list[matched_idx],
            "match_score":          round(score, 1),
            "candidate_position":   candidate_position,
            "student_position":     f"Author {student_pos}",
            "total_authors":        len(author_list),
        })

    return matches


# ---------------------------------------------------------------------------
# Main analyser class
# ---------------------------------------------------------------------------

class SupervisionAnalyser:
    """
    Orchestrates Module 4 tasks 4.1–4.6.

    Usage
    -----
    analyser = SupervisionAnalyser(
        supervision_csv  = "data/extracted/supervision.csv",
        publications_csv = "data/extracted/publications.csv",
        candidates_csv   = "data/extracted/candidates.csv",
        output_dir       = "data/analysis",
    )
    analyser.run()
    """

    def __init__(
        self,
        supervision_csv:  str = "data/extracted/supervision.csv",
        publications_csv: str = "data/extracted/publications.csv",
        candidates_csv:   str = "data/extracted/candidates.csv",
        output_dir:       str = "data/analysis",
        api_key:          Optional[str] = None,
        model:            str = "meta-llama/llama-4-scout-17b-16e-instruct",
        base_url:         Optional[str] = None,
        skip_llm:         bool = False,
    ):
        self.supervision_csv  = Path(supervision_csv)
        self.publications_csv = Path(publications_csv)
        self.candidates_csv   = Path(candidates_csv)
        self.output_dir       = Path(output_dir)
        self.skip_llm         = skip_llm
        self.api_key          = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model            = model
        self.base_url         = base_url or os.environ.get("OPENAI_BASE_URL")

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public: run
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Path]:
        logger.info("=" * 60)
        logger.info("  TALASH Module 4 – Student Supervision Analysis")
        logger.info("  Supervision CSV  : %s", self.supervision_csv)
        logger.info("  Publications CSV : %s", self.publications_csv)
        logger.info("  Candidates CSV   : %s", self.candidates_csv)
        logger.info("  Output dir       : %s", self.output_dir)
        logger.info("=" * 60)

        sup_df  = self._load_csv(self.supervision_csv,  "supervision")
        pub_df  = self._load_csv(self.publications_csv, "publications")
        cand_df = self._load_csv(self.candidates_csv,   "candidates")

        if cand_df is None or cand_df.empty:
            logger.error("candidates.csv missing or empty. Run Module 1 first.")
            return {}

        if pub_df is None:
            pub_df = pd.DataFrame()

        # Build lookups
        name_map: Dict[str, str] = dict(
            zip(cand_df["candidate_id"].astype(str), cand_df["name"].astype(str))
        )
        all_cids = cand_df["candidate_id"].dropna().unique().tolist()

        # Group supervision rows by candidate_id
        sup_by_cid: Dict[str, List[Dict]] = {}
        if sup_df is not None and not sup_df.empty:
            for _, row in sup_df.iterrows():
                cid  = str(row.get("candidate_id", "")).strip()
                data = row.to_dict()
                # Only keep rows that have at least student_name
                if cid:
                    sup_by_cid.setdefault(cid, []).append(data)

        # Group publications by candidate_id
        pub_by_cid: Dict[str, List[Dict]] = {}
        if not pub_df.empty:
            for _, row in pub_df.iterrows():
                cid = str(row.get("candidate_id", "")).strip()
                if cid:
                    pub_by_cid.setdefault(cid, []).append(row.to_dict())

        # ------------------------------------------------------------------
        # Checkpointing
        # ------------------------------------------------------------------
        checkpoint_path = self.output_dir / "_supervision_checkpoint.json"
        checkpoint      = self._load_checkpoint(checkpoint_path)
        completed_cids  = set(checkpoint.get("completed", []))
        all_profiles:   List[Dict] = list(checkpoint.get("profiles",   []))
        all_joint_pubs: List[Dict] = list(checkpoint.get("joint_pubs", []))

        if completed_cids:
            logger.info(
                "Checkpoint found: %d candidate(s) already done, resuming.",
                len(completed_cids)
            )

        for i, cid in enumerate(all_cids, 1):
            if cid in completed_cids:
                logger.info("[%d/%d] SKIP (checkpoint): %s", i, len(all_cids), cid)
                continue

            logger.info("[%d/%d] Analysing: %s", i, len(all_cids), cid)
            cand_name   = name_map.get(str(cid), str(cid))
            sup_records = sup_by_cid.get(cid, [])
            cand_pubs   = pub_by_cid.get(cid, [])

            profile, joint_papers = self._analyse_candidate(
                cid, cand_name, sup_records, cand_pubs
            )
            all_profiles.append(profile)
            all_joint_pubs.extend(joint_papers)

            completed_cids.add(cid)
            self._save_checkpoint(checkpoint_path, {
                "completed":  list(completed_cids),
                "profiles":   all_profiles,
                "joint_pubs": all_joint_pubs,
            })

        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info("Checkpoint cleared (run complete).")

        return self._export(all_profiles, all_joint_pubs)

    # ------------------------------------------------------------------
    # Per-candidate analysis
    # ------------------------------------------------------------------

    def _analyse_candidate(
        self,
        candidate_id:   str,
        candidate_name: str,
        sup_records:    List[Dict],
        cand_pubs:      List[Dict],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Run tasks 4.1–4.6 for one candidate."""

        # ------------------------------------------------------------------
        # Task 4.1: Determine data availability
        # A record with all-NaN fields is treated as "not provided".
        # ------------------------------------------------------------------
        valid_records = [
            r for r in sup_records
            if str(r.get("student_name", "")).strip().lower()
               not in ("nan", "none", "")
        ]
        data_missing = len(valid_records) == 0

        # ------------------------------------------------------------------
        # Tasks 4.2 & 4.3: Count supervisor roles
        # ------------------------------------------------------------------
        ms_main = ms_co = phd_main = phd_co = 0

        for r in valid_records:
            level = str(r.get("level") or "").strip().lower()
            role  = str(r.get("role")  or "").strip().lower()

            is_ms  = any(x in level for x in ("ms", "m.s", "msc", "master"))
            is_phd = any(x in level for x in ("phd", "ph.d", "doctorate"))
            is_mphil = any(x in level for x in ("mphil", "m.phil"))

            # MPhil treated as MS-equivalent
            if is_mphil:
                is_ms = True

            is_main = role in _MAIN_ROLE_SYNONYMS or "main" in role or "primary" in role
            is_co   = role in _CO_ROLE_SYNONYMS   or "co" in role   or "external" in role

            if is_ms:
                if is_main:  ms_main += 1
                elif is_co:  ms_co   += 1
                else:        ms_main += 1   # default to main if role is ambiguous
            elif is_phd:
                if is_main:  phd_main += 1
                elif is_co:  phd_co   += 1
                else:        phd_main += 1
            else:
                # Unknown level — count under ms_main as generic
                if not (is_ms or is_phd):
                    ms_main += 1

        # ------------------------------------------------------------------
        # Task 4.4: Joint publications with supervised students
        # ------------------------------------------------------------------
        all_joint_papers: List[Dict] = []
        for r in valid_records:
            sname = str(r.get("student_name") or "").strip()
            if not sname or sname.lower() in ("nan", "none"):
                continue
            matched = find_joint_papers(sname, cand_pubs, candidate_name)
            for m in matched:
                m["candidate_id"] = candidate_id
            all_joint_papers.extend(matched)

        # De-duplicate: same paper + same student should not appear twice
        seen = set()
        deduped_joint: List[Dict] = []
        for jp in all_joint_papers:
            key = (jp.get("paper_title"), jp.get("student_name"))
            if key not in seen:
                seen.add(key)
                deduped_joint.append(jp)

        total_joint = len(deduped_joint)

        # Typical authorship role in joint papers
        if deduped_joint:
            roles = [jp.get("candidate_position", "Unknown") for jp in deduped_joint]
            from collections import Counter
            typical_role = Counter(roles).most_common(1)[0][0]
        else:
            typical_role = "N/A"

        # ------------------------------------------------------------------
        # Task 4.5: Supervision quality interpretation
        # ------------------------------------------------------------------
        rule_label = _rule_based_supervision_label(
            ms_main, ms_co, phd_main, phd_co, total_joint, data_missing
        )

        if self.skip_llm:
            strength   = rule_label
            assessment = self._rule_based_assessment(
                candidate_id, ms_main, ms_co, phd_main, phd_co,
                total_joint, data_missing, rule_label
            )
        else:
            interp     = self._interpret_supervision(
                candidate_id, ms_main, ms_co, phd_main, phd_co,
                total_joint, typical_role, data_missing, rule_label
            )
            strength   = interp.get("supervisory_strength",  rule_label)
            assessment = interp.get("supervisory_assessment", "")
            time.sleep(0.5)   # gentle rate-limit buffer

        # ------------------------------------------------------------------
        # Task 4.6: Flag missing data
        # ------------------------------------------------------------------
        email_flag = (
            "REQUEST_SUPERVISION_DATA"
            if data_missing
            else "OK"
        )

        profile: Dict[str, Any] = {
            "candidate_id":               candidate_id,
            "candidate_name":             candidate_name,
            "ms_main_supervisor":         ms_main,
            "ms_co_supervisor":           ms_co,
            "phd_main_supervisor":        phd_main,
            "phd_co_supervisor":          phd_co,
            "total_ms_supervised":        ms_main + ms_co,
            "total_phd_supervised":       phd_main + phd_co,
            "total_students_supervised":  ms_main + ms_co + phd_main + phd_co,
            "total_joint_papers":         total_joint,
            "typical_joint_paper_role":   typical_role,
            "data_missing":               data_missing,
            "supervisory_strength":       strength,
            "supervisory_assessment":     assessment,
            "rule_based_label":           rule_label,
            "email_flag":                 email_flag,
        }

        return profile, deduped_joint

    # ------------------------------------------------------------------
    # LLM call for supervisory assessment
    # ------------------------------------------------------------------

    def _interpret_supervision(
        self,
        candidate_id: str,
        ms_main: int, ms_co: int,
        phd_main: int, phd_co: int,
        total_joint: int,
        typical_role: str,
        data_missing: bool,
        rule_label: str,
    ) -> Dict[str, Any]:
        user_prompt = _SUPERVISION_USER_TEMPLATE.format(
            candidate_id = candidate_id,
            ms_main      = ms_main,
            ms_co        = ms_co,
            phd_main     = phd_main,
            phd_co       = phd_co,
            total_joint  = total_joint,
            typical_role = typical_role,
            data_missing = data_missing,
            rule_label   = rule_label,
        )
        try:
            raw    = self._call_llm(_SUPERVISION_SYSTEM, user_prompt)
            parsed = self._parse_json(raw, candidate_id)
            if parsed and "supervisory_strength" in parsed:
                return parsed
        except Exception as e:
            logger.warning("LLM supervision assessment failed for '%s': %s", candidate_id, e)

        return {
            "supervisory_strength":  rule_label,
            "supervisory_assessment": self._rule_based_assessment(
                candidate_id, ms_main, ms_co, phd_main, phd_co,
                total_joint, data_missing, rule_label
            ),
        }

    @staticmethod
    def _rule_based_assessment(
        candidate_id: str,
        ms_main: int, ms_co: int,
        phd_main: int, phd_co: int,
        total_joint: int,
        data_missing: bool,
        label: str,
    ) -> str:
        if data_missing:
            return (
                f"No supervision information was found in the CV of {candidate_id}. "
                "This candidate has been flagged for follow-up to provide details of "
                "supervised MS/PhD students including names, degree levels, and graduation years."
            )
        total = ms_main + ms_co + phd_main + phd_co
        parts = [
            f"The candidate has supervised {total} postgraduate student(s) in total "
            f"({ms_main} MS main, {ms_co} MS co, {phd_main} PhD main, {phd_co} PhD co)."
        ]
        if total_joint:
            parts.append(f"{total_joint} joint publication(s) were identified with supervised students.")
        parts.append(f"Overall supervisory profile is assessed as {label}.")
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export(
        self,
        profiles:   List[Dict],
        joint_pubs: List[Dict],
    ) -> Dict[str, Path]:
        paths: Dict[str, Path] = {}

        # Sort: Strong → Moderate → Limited → Needs Clarification
        strength_rank = {"Strong": 1, "Moderate": 2, "Limited": 3, "Needs Clarification": 4}
        profiles = sorted(
            profiles,
            key=lambda x: (
                strength_rank.get(x.get("supervisory_strength"), 99),
                x.get("candidate_id", ""),
            )
        )

        # 1. Profiles CSV
        p = self.output_dir / "supervision_profiles.csv"
        pd.DataFrame(profiles).to_csv(p, index=False)
        logger.info("Wrote %d rows -> %s", len(profiles), p.name)
        paths["profiles"] = p

        # 2. Joint publications CSV
        if joint_pubs:
            p = self.output_dir / "supervision_joint_pubs.csv"
            pd.DataFrame(joint_pubs).to_csv(p, index=False)
            logger.info("Wrote %d rows -> %s", len(joint_pubs), p.name)
            paths["joint_pubs"] = p

        # 3. Excel workbook
        p = self.output_dir / "supervision_profiles.xlsx"
        with pd.ExcelWriter(p, engine="openpyxl") as writer:
            pd.DataFrame(profiles).to_excel(
                writer, sheet_name="Supervision Profiles", index=False
            )
            if joint_pubs:
                pd.DataFrame(joint_pubs).to_excel(
                    writer, sheet_name="Joint Publications", index=False
                )
        logger.info("Excel workbook written -> %s", p.name)
        paths["excel"] = p

        # 4. Plain-text report
        self._write_text_report(profiles, paths)

        return paths

    def _write_text_report(self, profiles: List[Dict], paths: Dict):
        p = self.output_dir / "supervision_report.txt"

        strength_counts: Dict[str, int] = {}
        for pr in profiles:
            s = pr.get("supervisory_strength") or "Unknown"
            strength_counts[s] = strength_counts.get(s, 0) + 1

        missing_count = sum(1 for pr in profiles if pr.get("data_missing"))

        lines = [
            "=" * 70,
            "  TALASH Module 4 – Student Supervision Analysis Report",
            "=" * 70,
            f"  Candidates analysed           : {len(profiles)}",
            f"  Candidates with NO data (flagged for email) : {missing_count}",
            "",
            "  Supervisory Strength Distribution:",
        ]
        for lbl, cnt in sorted(strength_counts.items()):
            lines.append(f"    {lbl:<25}: {cnt}")

        lines += [
            "",
            "-" * 70,
            "  Per-Candidate Supervision Summaries",
            "-" * 70,
        ]

        for pr in profiles:
            cid      = pr["candidate_id"]
            strength = pr.get("supervisory_strength", "Unknown")
            missing  = pr.get("data_missing", True)
            lines.append(f"\n  [{strength}] {cid}")
            if missing:
                lines.append("  ** Supervision data NOT provided in CV — flagged for follow-up email **")
            else:
                lines.append(
                    f"  MS  : main={pr.get('ms_main_supervisor',0)}  "
                    f"co={pr.get('ms_co_supervisor',0)}"
                )
                lines.append(
                    f"  PhD : main={pr.get('phd_main_supervisor',0)}  "
                    f"co={pr.get('phd_co_supervisor',0)}"
                )
                lines.append(
                    f"  Joint papers with students: "
                    f"{pr.get('total_joint_papers',0)}"
                )
            if pr.get("supervisory_assessment"):
                lines.append(f"  Summary: {pr['supervisory_assessment']}")

        lines.append("\n" + "=" * 70)
        p.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Text report written -> %s", p.name)
        paths["report"] = p

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_checkpoint(path: Path) -> Dict:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    @staticmethod
    def _save_checkpoint(path: Path, data: Dict):
        try:
            path.write_text(
                json.dumps(data, ensure_ascii=False, default=str),
                encoding="utf-8"
            )
        except Exception as e:
            logger.warning("Could not save checkpoint: %s", e)

    # ------------------------------------------------------------------
    # LLM + JSON helpers
    # ------------------------------------------------------------------

    def _call_llm(self, system: str, user: str) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed.")

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model    = self.model,
            messages = [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature = 0.1,
            max_tokens  = 512,
        )
        return response.choices[0].message.content or ""

    @staticmethod
    def _parse_json(raw: str, name: str) -> Optional[Dict]:
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        for pat in [
            r"```(?:json)?\s*(\{.*?\})\s*```",
            r"(\{.*\})",
        ]:
            m = re.search(pat, raw, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass
        logger.error("Could not parse LLM JSON for '%s'. Raw:\n%s", name, raw[:400])
        return None

    @staticmethod
    def _load_csv(path: Path, label: str) -> Optional[pd.DataFrame]:
        if not path.exists():
            logger.warning("%s CSV not found: %s", label, path)
            return None
        try:
            df = pd.read_csv(path, dtype=str)
            logger.info("Loaded %d rows from %s", len(df), path.name)
            return df
        except Exception as e:
            logger.error("Failed to read %s: %s", path.name, e)
            return None
