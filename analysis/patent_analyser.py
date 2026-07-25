"""
patent_analyser.py  –  Module 6: Patents Analysis Orchestrator
============================================================

WHY THIS FILE EXISTS
--------------------
Orchestrates Part 6 (Patents Analysis) for all candidates:
  - Reads patents.csv and candidates.csv (from Module 1)
  - Evaluates inventor role, filing jurisdiction (National vs International), and verification status
  - Computes per-candidate aggregate innovation metrics
  - Calls LLM to generate innovation assessment paragraphs (with rule-based fallback)
  - Exports patent_profiles.csv, patent_aggregates.csv, patent_aggregates.xlsx, and patent_report.txt
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .patent_verifier import (
    classify_inventor_role,
    classify_jurisdiction,
    build_patent_verification_link,
    check_patent_quality_flags,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM Prompts
# ---------------------------------------------------------------------------

_PATENT_SYSTEM_PROMPT = (
    "You are an expert technology transfer and intellectual property evaluator for a university recruitment system. "
    "You will receive structured patent and innovation statistics for a faculty candidate. "
    "Your task is to assess their applied research and patent output. "
    "Return ONLY a valid JSON object with exactly two keys: "
    "\"innovation_label\" and \"innovation_assessment\". "
    "innovation_label must be one of: \"Strong\", \"Moderate\", \"Limited\", \"No Patents Listed\". "
    "innovation_assessment must be 2-4 sentences in third person, factual, with no embellishment. "
    "If no patents are listed, set innovation_label to \"No Patents Listed\" and state that no patents were listed in the CV."
)

_PATENT_USER_TEMPLATE = (
    "Assess this candidate's patent and innovation record and return JSON.\n\n"
    "CANDIDATE ID: {candidate_id}\n\n"
    "PATENT STATISTICS:\n"
    "  Total patents                : {total_patents}\n"
    "  Sole inventor count          : {sole_inventor_count}\n"
    "  Lead inventor count          : {lead_inventor_count}\n"
    "  Co-inventor count            : {co_inventor_count}\n"
    "  Contributing innovator count : {contributing_innovator_count}\n"
    "  Unknown inventor role count  : {unknown_role_count}\n"
    "  International patents        : {international_patents}\n"
    "  National (Pakistan) patents  : {national_patents}\n"
    "  Verifiable patents           : {verifiable_patents}\n"
    "  Unverifiable patents         : {unverifiable_patents}\n\n"
    "SUGGESTED LABEL (rule-based): {rule_label}\n\n"
    "Return JSON:"
)


def _rule_based_patent_label(
    total_patents: int,
    lead_inventor_count: int,
    international_patents: int,
    verifiable_patents: int,
) -> str:
    """Deterministic label logic for patent profile."""
    if total_patents == 0:
        return "No Patents Listed"
    if (international_patents >= 1 or lead_inventor_count >= 2) and verifiable_patents >= 1:
        return "Strong"
    if total_patents >= 1:
        return "Moderate"
    return "Limited"


class PatentProfileAnalyser:
    """
    Orchestrates tasks 6.1–6.5 for every candidate in patents.csv and candidates.csv.

    Usage
    -----
    analyser = PatentProfileAnalyser(
        patents_csv    = "data/extracted/patents.csv",
        candidates_csv = "data/extracted/candidates.csv",
        output_dir     = "data/analysis",
    )
    analyser.run()
    """

    def __init__(
        self,
        patents_csv:    str = "data/extracted/patents.csv",
        candidates_csv: str = "data/extracted/candidates.csv",
        output_dir:     str = "data/analysis",
        api_key:        Optional[str] = None,
        model:          str = "meta-llama/llama-4-scout-17b-16e-instruct",
        base_url:       Optional[str] = None,
        skip_llm:       bool = False,
    ):
        self.patents_csv    = Path(patents_csv)
        self.candidates_csv = Path(candidates_csv)
        self.output_dir     = Path(output_dir)
        self.api_key        = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model          = model
        self.base_url       = base_url or os.environ.get("OPENAI_BASE_URL")
        self.skip_llm       = skip_llm or not bool(self.api_key and not str(self.api_key).startswith("your_") and len(str(self.api_key).strip()) > 20)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public: run
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Path]:
        logger.info("=" * 60)
        logger.info("  TALASH Module 6 – Patents Analysis")
        logger.info("  Patents CSV      : %s", self.patents_csv)
        logger.info("  Candidates CSV   : %s", self.candidates_csv)
        logger.info("  Output dir       : %s", self.output_dir)
        logger.info("=" * 60)

        pat_df  = self._load_csv(self.patents_csv, "patents")
        cand_df = self._load_csv(self.candidates_csv, "candidates")

        if cand_df is None or cand_df.empty:
            logger.error("candidates.csv missing or empty. Run Module 1 first.")
            return {}

        name_map: Dict[str, str] = dict(
            zip(cand_df["candidate_id"].astype(str), cand_df["name"].astype(str))
        )
        all_cids = cand_df["candidate_id"].dropna().unique().tolist()

        # Group patents by candidate_id
        pat_by_cid: Dict[str, List[Dict]] = {}
        if pat_df is not None and not pat_df.empty:
            for _, row in pat_df.iterrows():
                cid = str(row.get("candidate_id", "")).strip()
                if cid:
                    pat_by_cid.setdefault(cid, []).append(row.to_dict())

        # ------------------------------------------------------------------
        # Checkpointing
        # ------------------------------------------------------------------
        checkpoint_path = self.output_dir / "_patent_checkpoint.json"
        checkpoint      = self._load_checkpoint(checkpoint_path)
        completed_cids  = set(checkpoint.get("completed", []))

        all_patent_profiles: List[Dict] = list(checkpoint.get("patent_profiles", []))
        all_aggregates:      List[Dict] = list(checkpoint.get("aggregates",      []))

        if completed_cids:
            logger.info(
                "Checkpoint found: %d candidate(s) already done, resuming.",
                len(completed_cids)
            )

        for i, cid in enumerate(all_cids, 1):
            if cid in completed_cids:
                logger.info("[%d/%d] SKIP (checkpoint): %s", i, len(all_cids), cid)
                continue

            logger.info("[%d/%d] Analysing patents for: %s", i, len(all_cids), cid)
            cand_name = name_map.get(str(cid), str(cid))
            raw_patents = pat_by_cid.get(str(cid), [])

            patent_rows, aggregate = self._analyse_candidate(cid, cand_name, raw_patents)

            all_patent_profiles.extend(patent_rows)
            all_aggregates.append(aggregate)

            completed_cids.add(str(cid))
            self._save_checkpoint(checkpoint_path, {
                "completed":       list(completed_cids),
                "patent_profiles": all_patent_profiles,
                "aggregates":      all_aggregates,
            })

        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info("Checkpoint cleared (run complete).")

        return self._export(all_patent_profiles, all_aggregates)

    # ------------------------------------------------------------------
    # Per-Candidate Analysis
    # ------------------------------------------------------------------

    def _analyse_candidate(
        self,
        candidate_id: str,
        candidate_name: str,
        raw_patents: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:

        # Filter valid patent entries (must have title or patent_number or inventors)
        valid_patents = [
            p for p in raw_patents
            if (str(p.get("title") or "").strip().lower() not in ("nan", "none", "")
                or str(p.get("patent_number") or "").strip().lower() not in ("nan", "none", "")
                or str(p.get("inventors") or "").strip().lower() not in ("nan", "none", ""))
        ]

        patent_records = []
        sole_inventor = 0
        lead_inventor = 0
        co_inventor = 0
        contributing_innovator = 0
        unknown_role = 0
        international_count = 0
        national_count = 0
        unknown_jurisdiction_count = 0
        verifiable_count = 0
        unverifiable_count = 0

        for p in valid_patents:
            title       = str(p.get("title") or "").strip()
            pat_num     = str(p.get("patent_number") or "").strip()
            date_str    = str(p.get("date") or "").strip()
            inventors   = str(p.get("inventors") or "").strip()
            country_raw = str(p.get("country") or "").strip()
            raw_link    = str(p.get("link") or p.get("online_link") or "").strip()

            # Tasks 6.2 - 6.4
            role = classify_inventor_role(inventors, candidate_name)
            country_norm, jurisdiction = classify_jurisdiction(country_raw, pat_num)
            v_link, verifiable = build_patent_verification_link(pat_num, title, raw_link)

            row_data = {
                "candidate_id":     candidate_id,
                "patent_number":    pat_num,
                "title":            title,
                "date":             date_str,
                "inventors":        inventors,
                "country":          country_norm,
                "jurisdiction":     jurisdiction,
                "online_link":      v_link or raw_link,
                "inventor_role":    role,
                "verifiable":       verifiable,
            }

            flags = check_patent_quality_flags(row_data, verifiable)
            row_data["data_quality_flags"] = flags

            patent_records.append(row_data)

            # Accumulate stats
            if role == "Sole Inventor":
                sole_inventor += 1
            elif role == "Lead Inventor":
                lead_inventor += 1
            elif role == "Co-Inventor":
                co_inventor += 1
            elif role == "Contributing Innovator":
                contributing_innovator += 1
            else:
                unknown_role += 1

            if jurisdiction == "National (Pakistan)":
                national_count += 1
            elif jurisdiction == "International":
                international_count += 1
            else:
                unknown_jurisdiction_count += 1

            if verifiable:
                verifiable_count += 1
            else:
                unverifiable_count += 1

        total_patents = len(patent_records)
        rule_label = _rule_based_patent_label(
            total_patents, lead_inventor + sole_inventor, international_count, verifiable_count
        )

        # Task 6.5: LLM Innovation Assessment
        if self.skip_llm:
            innovation_label = rule_label
            assessment = self._rule_based_assessment(
                candidate_name, total_patents, sole_inventor, lead_inventor,
                co_inventor, international_count, national_count, rule_label
            )
        else:
            interp = self._interpret_patent_profile(
                candidate_id, total_patents, sole_inventor, lead_inventor,
                co_inventor, contributing_innovator, unknown_role,
                international_count, national_count, verifiable_count,
                unverifiable_count, rule_label
            )
            innovation_label = interp.get("innovation_label", rule_label)
            assessment = interp.get("innovation_assessment", "")
            time.sleep(0.5)

        aggregate = {
            "candidate_id":                 candidate_id,
            "candidate_name":               candidate_name,
            "total_patents":                total_patents,
            "sole_inventor_count":          sole_inventor,
            "lead_inventor_count":          lead_inventor,
            "co_inventor_count":            co_inventor,
            "contributing_innovator_count": contributing_innovator,
            "unknown_role_count":           unknown_role,
            "international_patents":        international_count,
            "national_patents":             national_count,
            "unknown_jurisdiction_patents": unknown_jurisdiction_count,
            "verifiable_patents":           verifiable_count,
            "unverifiable_patents":         unverifiable_count,
            "innovation_label":             innovation_label,
            "innovation_assessment":        assessment,
            "rule_based_label":             rule_label,
        }

        return patent_records, aggregate

    # ------------------------------------------------------------------
    # LLM Interpretation
    # ------------------------------------------------------------------

    def _interpret_patent_profile(
        self,
        candidate_id: str,
        total_patents: int,
        sole_inventor: int,
        lead_inventor: int,
        co_inventor: int,
        contributing_innovator: int,
        unknown_role: int,
        international_count: int,
        national_count: int,
        verifiable_count: int,
        unverifiable_count: int,
        rule_label: str,
    ) -> Dict[str, Any]:
        if total_patents == 0:
            return {
                "innovation_label": "No Patents Listed",
                "innovation_assessment": f"Candidate {candidate_id} has no patents listed in their CV."
            }

        user_prompt = _PATENT_USER_TEMPLATE.format(
            candidate_id                = candidate_id,
            total_patents               = total_patents,
            sole_inventor_count         = sole_inventor,
            lead_inventor_count         = lead_inventor,
            co_inventor_count           = co_inventor,
            contributing_innovator_count= contributing_innovator,
            unknown_role_count          = unknown_role,
            international_patents       = international_count,
            national_patents            = national_count,
            verifiable_patents          = verifiable_count,
            unverifiable_patents        = unverifiable_count,
            rule_label                  = rule_label,
        )

        try:
            raw = self._call_llm(_PATENT_SYSTEM_PROMPT, user_prompt)
            parsed = self._parse_json(raw, candidate_id)
            if parsed and "innovation_label" in parsed:
                return parsed
        except Exception as e:
            logger.warning("LLM patent assessment failed for '%s': %s", candidate_id, e)

        return {
            "innovation_label": rule_label,
            "innovation_assessment": self._rule_based_assessment(
                candidate_id, total_patents, sole_inventor, lead_inventor,
                co_inventor, international_count, national_count, rule_label
            ),
        }

    @staticmethod
    def _rule_based_assessment(
        name: str,
        total: int,
        sole: int,
        lead: int,
        co: int,
        intl: int,
        nat: int,
        label: str,
    ) -> str:
        if total == 0:
            return f"Candidate {name} has no patents listed in their CV."
        parts = [f"Candidate {name} holds {total} patent(s)."]
        if sole or lead:
            parts.append(f"{sole + lead} patent(s) as sole or lead inventor.")
        if intl:
            parts.append(f"{intl} international patent(s).")
        if nat:
            parts.append(f"{nat} national (Pakistani) patent(s).")
        parts.append(f"Overall applied innovation and IP profile is assessed as {label}.")
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export(
        self,
        patent_profiles: List[Dict],
        aggregates:      List[Dict],
    ) -> Dict[str, Path]:
        paths: Dict[str, Path] = {}

        label_rank = {"Strong": 1, "Moderate": 2, "Limited": 3, "No Patents Listed": 4}
        aggregates = sorted(
            aggregates,
            key=lambda x: (
                label_rank.get(x.get("innovation_label"), 99),
                x.get("candidate_id", ""),
            )
        )

        # 1. Patent Profiles CSV
        p = self.output_dir / "patent_profiles.csv"
        pd.DataFrame(patent_profiles).to_csv(p, index=False)
        logger.info("Wrote %d rows -> %s", len(patent_profiles), p.name)
        paths["patent_profiles"] = p

        # 2. Patent Aggregates CSV
        p = self.output_dir / "patent_aggregates.csv"
        pd.DataFrame(aggregates).to_csv(p, index=False)
        logger.info("Wrote %d rows -> %s", len(aggregates), p.name)
        paths["patent_aggregates"] = p

        # 3. Excel Workbook
        p = self.output_dir / "patent_aggregates.xlsx"
        with pd.ExcelWriter(p, engine="openpyxl") as writer:
            pd.DataFrame(aggregates).to_excel(
                writer, sheet_name="Patent Aggregates", index=False
            )
            pd.DataFrame(patent_profiles).to_excel(
                writer, sheet_name="Patent Profiles", index=False
            )
        logger.info("Excel workbook written -> %s", p.name)
        paths["excel"] = p

        # 4. Text Report
        self._write_text_report(aggregates, patent_profiles, paths)

        return paths

    def _write_text_report(self, aggregates: List[Dict], patent_profiles: List[Dict], paths: Dict):
        p = self.output_dir / "patent_report.txt"

        label_counts: Dict[str, int] = {}
        for agg in aggregates:
            s = agg.get("innovation_label") or "Unknown"
            label_counts[s] = label_counts.get(s, 0) + 1

        total_patents_found = len(patent_profiles)

        lines = [
            "=" * 70,
            "  TALASH Module 6 – Patents Analysis Report",
            "=" * 70,
            f"  Candidates analysed           : {len(aggregates)}",
            f"  Total patents extracted       : {total_patents_found}",
            "",
            "  Innovation & IP Output Label Distribution:",
        ]
        for lbl, cnt in sorted(label_counts.items()):
            lines.append(f"    {lbl:<25}: {cnt}")

        lines += [
            "",
            "-" * 70,
            "  Per-Candidate Patent Summaries",
            "-" * 70,
        ]

        for agg in aggregates:
            cid   = agg["candidate_id"]
            lbl   = agg.get("innovation_label", "Unknown")
            tot   = agg.get("total_patents", 0)
            lines.append(f"\n  [{lbl}] {cid}")
            lines.append(f"  Total Patents: {tot} (Sole={agg.get('sole_inventor_count',0)}, Lead={agg.get('lead_inventor_count',0)}, Co={agg.get('co_inventor_count',0)})")
            lines.append(f"  Jurisdiction: International={agg.get('international_patents',0)}, National={agg.get('national_patents',0)}")
            if agg.get("innovation_assessment"):
                lines.append(f"  Summary: {agg['innovation_assessment']}")

        lines.append("\n" + "=" * 70)
        p.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Text report written -> %s", p.name)
        paths["report"] = p

    # ------------------------------------------------------------------
    # Checkpoint Helpers
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
    # LLM + JSON Helpers
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
