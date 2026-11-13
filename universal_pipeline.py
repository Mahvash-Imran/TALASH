"""
universal_pipeline.py  –  Pipeline Integration Glue for Universal Scoring
========================================================================

WHY THIS MODULE EXISTS
----------------------
Integrates the department-agnostic Universal Scoring Core with TALASH's
existing data extraction layer, pipeline orchestrator, and FastAPI API routes:
  1. Automated Data Assembly:
     - Assembles NormalizedCV instances from TALASH's tabular CSV files
       (candidates, education, experience, skills, publications, books, patents, supervision).
     - Fully respects multi-tenancy directories when X-User-Email / tenant_dir is active.
  2. Department Auto-Inference:
     - Infers candidate's academic discipline from degree and experience records
       if no specific department profile is selected.
  3. Non-Regression Dual-Export:
     - Emits rich explainable universal evaluation outputs (universal_evaluations.json & csv).
     - Seamlessly updates composite_evaluations.csv and composite_evaluations.xlsx
       via .to_legacy_dict(), guaranteeing 100% backward compatibility for all existing
       dashboard UI views, compare tools, and JD matching routes.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

import models
from models import (
    CandidateEvaluationResult,
    CandidateTier,
    NormalizedCV,
    RolePreset,
    UniversalDimension,
    DepartmentProfile,
)
import profile_loader
import universal_scoring

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_EXTRACTED_DIR = BASE_DIR / "data" / "extracted"
DEFAULT_ANALYSIS_DIR = BASE_DIR / "data" / "analysis"


class UniversalPipeline:
    """
    High-level orchestrator connecting TALASH extracted data with the Universal Scoring Core.
    """

    def __init__(
        self,
        extracted_dir: Union[str, Path] = DEFAULT_EXTRACTED_DIR,
        analysis_dir: Union[str, Path] = DEFAULT_ANALYSIS_DIR,
        profiles_dir: Optional[Union[str, Path]] = None,
    ):
        self.extracted_dir = Path(extracted_dir)
        self.analysis_dir = Path(analysis_dir)
        self.profiles_dir = Path(profiles_dir) if profiles_dir else profile_loader.DEFAULT_PROFILES_DIR
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

    def load_normalized_cv(self, candidate_id: str) -> Optional[NormalizedCV]:
        """Loads and constructs a single NormalizedCV from extracted tables."""
        cvs = self.load_normalized_cvs(candidate_ids=[candidate_id])
        return cvs[0] if cvs else None

    def load_normalized_cvs(
        self,
        candidate_ids: Optional[List[str]] = None,
    ) -> List[NormalizedCV]:
        """
        Loads all or specified candidates from extracted CSV tables into NormalizedCV models.
        """
        cands_csv = self.extracted_dir / "candidates.csv"
        if not cands_csv.exists():
            logger.warning("candidates.csv not found at %s", cands_csv)
            return []

        try:
            cand_df = pd.read_csv(cands_csv, dtype=str).fillna("")
        except Exception as exc:
            logger.error("Failed to read candidates.csv: %s", exc)
            return []

        if candidate_ids:
            target_ids = {str(cid).strip() for cid in candidate_ids}
            cand_df = cand_df[cand_df["candidate_id"].isin(target_ids)]

        # Load supplementary tables
        edu_df = self._safe_read_csv("education.csv")
        exp_df = self._safe_read_csv("experience.csv")
        skills_df = self._safe_read_csv("skills.csv")
        pubs_df = self._safe_read_csv("publications.csv")
        books_df = self._safe_read_csv("books.csv")
        pats_df = self._safe_read_csv("patents.csv")
        sup_df = self._safe_read_csv("supervision.csv")

        normalized_cvs: List[NormalizedCV] = []

        for _, r in cand_df.iterrows():
            cid = str(r.get("candidate_id", "")).strip()
            if not cid:
                continue

            c_edus = self._filter_rows(edu_df, cid)
            c_exps = self._filter_rows(exp_df, cid)
            c_sks = self._filter_rows(skills_df, cid)
            c_pubs = self._filter_rows(pubs_df, cid)
            c_books = self._filter_rows(books_df, cid)
            c_pats = self._filter_rows(pats_df, cid)
            c_sups = self._filter_rows(sup_df, cid)

            cv = NormalizedCV.from_extracted_tables(
                candidate_id=cid,
                cand_row=r.to_dict(),
                education_rows=c_edus,
                experience_rows=c_exps,
                skills_rows=c_sks,
                publication_rows=c_pubs,
                book_rows=c_books,
                patent_rows=c_pats,
                supervision_rows=c_sups,
            )
            normalized_cvs.append(cv)

        return normalized_cvs

    def infer_department_slug(self, cv: NormalizedCV) -> str:
        """
        Infers the most appropriate department profile slug for a candidate based on
        their degree titles, specializations, skills, and experience keywords.
        """
        combined_text = " ".join(
            [e.degree for e in cv.education]
            + [e.specialization or "" for e in cv.education]
            + [s.skill_name for s in cv.skills]
            + [e.job_title for e in cv.experience]
        ).lower()

        # Keyword heuristics aligned with the 6 seed disciplines
        if re.search(r"\b(electrical|electronics|telecom|power systems|rf|vlsi|signal processing|embedded)\b", combined_text):
            return "electrical_engineering"
        elif re.search(r"\b(medicine|surgery|mbbs|clinical|physician|pathology|pediatrics|hospital|dr\.?)\b", combined_text):
            return "medicine"
        elif re.search(r"\b(law|llb|llm|advocate|bar council|legal|court|judicial|jurisprudence)\b", combined_text):
            return "law"
        elif re.search(r"\b(bba|mba|finance|marketing|management|business|accounting|supply chain)\b", combined_text):
            return "business_administration"
        elif re.search(r"\b(literature|linguistics|english|poetry|humanities|cultural studies)\b", combined_text):
            return "english_literature"
        else:
            # Default to computer_science as primary computational baseline
            return "computer_science"

    def evaluate_single(
        self,
        candidate_id: str,
        department_slug: Optional[str] = None,
        role_preset: RolePreset = RolePreset.FACULTY,
        jd_requirements: Optional[Dict[str, Any]] = None,
    ) -> Optional[CandidateEvaluationResult]:
        """Evaluates a single candidate and returns the comprehensive evaluation result."""
        cv = self.load_normalized_cv(candidate_id)
        if not cv:
            return None

        slug = department_slug or self.infer_department_slug(cv)
        profile = profile_loader.get_profile(slug, profiles_dir=self.profiles_dir)
        if not profile:
            # Fall back to computer_science if requested slug not found
            profile = profile_loader.get_profile("computer_science", profiles_dir=self.profiles_dir)

        assert profile is not None, "At least one department profile must be loaded."
        return universal_scoring.evaluate_candidate(
            cv=cv,
            profile=profile,
            role_preset=role_preset,
            jd_requirements=jd_requirements,
        )

    def evaluate_all(
        self,
        candidate_ids: Optional[List[str]] = None,
        department_slug: Optional[str] = None,
        role_preset: RolePreset = RolePreset.FACULTY,
        jd_requirements: Optional[Dict[str, Any]] = None,
        save: bool = True,
    ) -> Dict[str, Any]:
        """
        Runs Universal Evaluation on all (or targeted) candidates.
        Produces:
          - universal_evaluations.json (full explainability data)
          - universal_evaluations.csv (all 8 dimensions + gate status + scores)
          - composite_evaluations.csv & .xlsx (100% legacy-compatible schema)
        """
        cvs = self.load_normalized_cvs(candidate_ids=candidate_ids)
        if not cvs:
            logger.warning("No candidate records found to evaluate.")
            return {"count": 0, "results": []}

        results: List[CandidateEvaluationResult] = []
        legacy_rows: List[Dict[str, Any]] = []
        universal_rows: List[Dict[str, Any]] = []

        # Cache profiles by slug during batch run
        profile_cache: Dict[str, DepartmentProfile] = {}

        for cv in cvs:
            slug = department_slug or self.infer_department_slug(cv)
            if slug not in profile_cache:
                prof = profile_loader.get_profile(slug, profiles_dir=self.profiles_dir)
                if not prof:
                    prof = profile_loader.get_profile("computer_science", profiles_dir=self.profiles_dir)
                profile_cache[slug] = prof

            profile = profile_cache[slug]
            eval_res = universal_scoring.evaluate_candidate(
                cv=cv,
                profile=profile,
                role_preset=role_preset,
                jd_requirements=jd_requirements,
            )
            results.append(eval_res)

            # Legacy row for zero regression
            legacy_dict = eval_res.to_legacy_dict()
            legacy_rows.append(legacy_dict)

            # Universal tabular row
            u_row = {
                "candidate_id": eval_res.candidate_id,
                "candidate_name": eval_res.candidate_name,
                "department": eval_res.department_slug,
                "role_preset": eval_res.role_preset.value,
                "gate_status": eval_res.gate_status.value,
                "final_score": eval_res.final_score,
                "tier": eval_res.tier.value,
                "evidence_coverage": eval_res.evidence_coverage,
                "confidence_multiplier": eval_res.confidence_multiplier,
            }
            for dim, d_res in eval_res.dimension_scores.items():
                u_row[f"score_{dim.value}"] = d_res.score
                u_row[f"weight_{dim.value}"] = round(d_res.weight, 4)
            universal_rows.append(u_row)

        # Sort by final score descending
        results.sort(key=lambda x: x.final_score, reverse=True)
        legacy_rows.sort(key=lambda x: x["overall_composite_score"], reverse=True)
        universal_rows.sort(key=lambda x: x["final_score"], reverse=True)

        outputs: Dict[str, Any] = {
            "count": len(results),
            "results": results,
        }

        if save:
            # 1. Save universal_evaluations.json
            json_file = self.analysis_dir / "universal_evaluations.json"
            json_data = [r.model_dump() for r in results]
            json_file.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")
            outputs["universal_json"] = str(json_file)

            # 2. Save universal_evaluations.csv
            u_csv_file = self.analysis_dir / "universal_evaluations.csv"
            pd.DataFrame(universal_rows).to_csv(u_csv_file, index=False)
            outputs["universal_csv"] = str(u_csv_file)

            # 3. Save composite_evaluations.csv (LEGACY ZERO REGRESSION)
            comp_csv_file = self.analysis_dir / "composite_evaluations.csv"
            pd.DataFrame(legacy_rows).to_csv(comp_csv_file, index=False)
            outputs["composite_csv"] = str(comp_csv_file)

            # 4. Save composite_evaluations.xlsx
            comp_xlsx_file = self.analysis_dir / "composite_evaluations.xlsx"
            with pd.ExcelWriter(comp_xlsx_file, engine="openpyxl") as writer:
                pd.DataFrame(legacy_rows).to_excel(writer, sheet_name="Composite Evaluations", index=False)
            outputs["composite_xlsx"] = str(comp_xlsx_file)

            logger.info("Universal evaluation complete. Saved %d evaluations to %s", len(results), self.analysis_dir)

        return outputs

    def _safe_read_csv(self, filename: str) -> Optional[pd.DataFrame]:
        """Safely loads a CSV file from extracted_dir or analysis_dir."""
        p1 = self.extracted_dir / filename
        if p1.exists():
            try:
                return pd.read_csv(p1, dtype=str).fillna("")
            except Exception:
                pass
        p2 = self.analysis_dir / filename
        if p2.exists():
            try:
                return pd.read_csv(p2, dtype=str).fillna("")
            except Exception:
                pass
        return None

    @staticmethod
    def _filter_rows(df: Optional[pd.DataFrame], candidate_id: str) -> List[Dict[str, Any]]:
        if df is None or df.empty:
            return []
        try:
            matched = df[df["candidate_id"] == candidate_id]
            return [row.to_dict() for _, row in matched.iterrows()]
        except Exception:
            return []


def run_universal_pipeline(
    extracted_dir: Union[str, Path] = DEFAULT_EXTRACTED_DIR,
    analysis_dir: Union[str, Path] = DEFAULT_ANALYSIS_DIR,
    department_slug: Optional[str] = None,
    role_preset: RolePreset = RolePreset.FACULTY,
) -> Dict[str, Any]:
    """Convenience functional interface for executing the universal pipeline."""
    pipe = UniversalPipeline(extracted_dir=extracted_dir, analysis_dir=analysis_dir)
    return pipe.evaluate_all(department_slug=department_slug, role_preset=role_preset, save=True)
