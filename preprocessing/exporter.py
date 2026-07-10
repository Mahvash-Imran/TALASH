"""
exporter.py  –  Tasks 1.4 & 1.5: Relational Output & Parsing Report
=====================================================================

WHY THIS FILE EXISTS
--------------------
The plan (Sections 1.4 & 1.5) requires:
  • Produce separate CSV files — one per entity type (8 tables)
  • Write all tables to a single multi-sheet Excel workbook
  • Generate a parsing report: how many CVs processed, fields missing, errors

RELATIONAL OUTPUT STRUCTURE (from plan Section 1.4)
-----------------------------------------------------
  candidates.csv   – one row per candidate
  education.csv    – one row per degree
  experience.csv   – one row per job role
  skills.csv       – one row per skill
  publications.csv – one row per publication
  supervision.csv  – one row per supervised student
  books.csv        – one row per book
  patents.csv      – one row per patent

DESIGN DECISIONS
----------------
- Every table uses candidate_id as the foreign key so downstream modules
  can JOIN on it. candidate_id is derived from the PDF filename (stem),
  which is unique in the input folder.
- pandas DataFrames are built per-entity so the same data is used for
  both CSV and Excel exports — no duplication of logic.
- The parsing_report sheet (and report CSV) provides a human-readable
  summary that satisfies "generate a parsing report" from the plan.
- We add an auto_increment integer key (`row_id`) to each table for
  unambiguous identification of individual rows (useful in downstream SQL).
- Columns are ordered consistently with the plan's table definition,
  then extended with any extra fields added during normalization.
"""

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column definitions (plan Section 1.4 + normalization extras)
# ---------------------------------------------------------------------------

CANDIDATES_COLS = [
    "candidate_id", "name", "email", "phone", "address", "cnic",
    "source_filename", "processed_at"
]

EDUCATION_COLS = [
    "row_id", "candidate_id",
    "level", "degree", "specialization", "institution", "country", "board",
    "start_year", "end_year",
    "marks_percentage", "marks_percentage_original",
    "cgpa", "cgpa_scale", "cgpa_normalized_4",
    "the_rank", "the_rank_range", "qs_rank", "qs_rank_range", "ranking_notes"
]

EXPERIENCE_COLS = [
    "row_id", "candidate_id",
    "job_title", "organization",
    "start_date", "end_date", "employment_type", "description"
]

SKILLS_COLS = [
    "row_id", "candidate_id", "skill_name", "category"
]

PUBLICATIONS_COLS = [
    "row_id", "candidate_id",
    "type", "title", "venue", "year", "authors", "doi", "url", "issn"
]

SUPERVISION_COLS = [
    "row_id", "candidate_id",
    "student_name", "level", "role", "year", "thesis_title"
]

BOOKS_COLS = [
    "row_id", "candidate_id",
    "title", "authors", "isbn", "publisher", "year", "link"
]

PATENTS_COLS = [
    "row_id", "candidate_id",
    "patent_number", "title", "date", "inventors", "country", "link"
]

REPORT_COLS = [
    "candidate_id", "source_filename", "status",
    "pages", "education_count", "experience_count", "skills_count",
    "publications_count", "supervision_count", "books_count", "patents_count",
    "missing_fields", "warnings", "error_message"
]


# ---------------------------------------------------------------------------
# Exporter
# ---------------------------------------------------------------------------

class Exporter:
    """
    Builds relational DataFrames from a list of normalized extraction results
    and writes them to CSV files and a multi-sheet Excel workbook.

    Usage
    -----
    exporter = Exporter(output_dir="data/extracted")
    exporter.add_candidate(
        candidate_filename="john_doe",
        normalized_data=clean_dict,
        validation=validation_report,
        pdf_result=pdf_read_result
    )
    exporter.add_failure("broken_cv", pdf_result=pdf_result, error="...")
    exporter.export()
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Accumulator lists (each item becomes one row)
        self._candidates: List[Dict] = []
        self._education: List[Dict] = []
        self._experience: List[Dict] = []
        self._skills: List[Dict] = []
        self._publications: List[Dict] = []
        self._supervision: List[Dict] = []
        self._books: List[Dict] = []
        self._patents: List[Dict] = []
        self._report: List[Dict] = []

        self._row_counters: Dict[str, int] = {}  # tracks auto-increment IDs per table

    # ------------------------------------------------------------------
    # Public: add data
    # ------------------------------------------------------------------

    def add_candidate(
        self,
        candidate_filename: str,
        normalized_data: Dict[str, Any],
        validation=None,      # ValidationReport from llm_extractor
        pdf_result=None,      # PDFReadResult from pdf_reader
    ):
        """
        Add one successfully processed candidate to all tables.

        Parameters
        ----------
        candidate_filename : str
            PDF filename stem – used as candidate_id (unique key).
        normalized_data : dict
            Output of Normalizer.normalize()
        validation : ValidationReport, optional
        pdf_result : PDFReadResult, optional
        """
        cid = self._make_candidate_id(candidate_filename)
        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"

        pinfo = normalized_data.get("personal_info") or {}

        # ── candidates table ──────────────────────────────────────────────
        self._candidates.append({
            "candidate_id": cid,
            "name":            pinfo.get("name"),
            "email":           pinfo.get("email"),
            "phone":           pinfo.get("phone"),
            "address":         pinfo.get("address"),
            "cnic":            pinfo.get("cnic"),
            "source_filename": candidate_filename + ".pdf",
            "processed_at":    now,
        })

        # ── education table ───────────────────────────────────────────────
        for edu in (normalized_data.get("education") or []):
            row = {"row_id": self._next_id("education"), "candidate_id": cid}
            row.update(edu)
            self._education.append(row)

        # ── experience table ──────────────────────────────────────────────
        for exp in (normalized_data.get("experience") or []):
            row = {"row_id": self._next_id("experience"), "candidate_id": cid}
            row.update(exp)
            self._experience.append(row)

        # ── skills table ──────────────────────────────────────────────────
        for sk in (normalized_data.get("skills") or []):
            row = {"row_id": self._next_id("skills"), "candidate_id": cid}
            row.update(sk)
            self._skills.append(row)

        # ── publications table ────────────────────────────────────────────
        for pub in (normalized_data.get("publications") or []):
            row = {"row_id": self._next_id("publications"), "candidate_id": cid}
            row.update(pub)
            self._publications.append(row)

        # ── supervision table ─────────────────────────────────────────────
        for sup in (normalized_data.get("supervision") or []):
            row = {"row_id": self._next_id("supervision"), "candidate_id": cid}
            row.update(sup)
            self._supervision.append(row)

        # ── books table ───────────────────────────────────────────────────
        for book in (normalized_data.get("books") or []):
            row = {"row_id": self._next_id("books"), "candidate_id": cid}
            row.update(book)
            self._books.append(row)

        # ── patents table ─────────────────────────────────────────────────
        for pat in (normalized_data.get("patents") or []):
            row = {"row_id": self._next_id("patents"), "candidate_id": cid}
            row.update(pat)
            self._patents.append(row)

        # ── parsing report ────────────────────────────────────────────────
        missing = validation.summary() if validation else ""
        warnings = "; ".join(getattr(pdf_result, "warnings", [])) if pdf_result else ""
        self._report.append({
            "candidate_id":       cid,
            "source_filename":    candidate_filename + ".pdf",
            "status":             "success",
            "pages":              getattr(pdf_result, "page_count", None),
            "education_count":    len(normalized_data.get("education") or []),
            "experience_count":   len(normalized_data.get("experience") or []),
            "skills_count":       len(normalized_data.get("skills") or []),
            "publications_count": len(normalized_data.get("publications") or []),
            "supervision_count":  len(normalized_data.get("supervision") or []),
            "books_count":        len(normalized_data.get("books") or []),
            "patents_count":      len(normalized_data.get("patents") or []),
            "missing_fields":     missing,
            "warnings":           warnings,
            "error_message":      None,
        })

    def add_failure(
        self,
        candidate_filename: str,
        pdf_result=None,
        error: str = "Unknown error",
    ):
        """
        Record a CV that could not be processed (PDF read failure or LLM failure).
        No data rows are added to other tables, only a failure row in the report.
        """
        cid = self._make_candidate_id(candidate_filename)
        warnings = "; ".join(getattr(pdf_result, "warnings", [])) if pdf_result else ""
        self._report.append({
            "candidate_id":       cid,
            "source_filename":    candidate_filename + ".pdf",
            "status":             "failed",
            "pages":              getattr(pdf_result, "page_count", 0),
            "education_count":    0,
            "experience_count":   0,
            "skills_count":       0,
            "publications_count": 0,
            "supervision_count":  0,
            "books_count":        0,
            "patents_count":      0,
            "missing_fields":     "",
            "warnings":           warnings,
            "error_message":      error,
        })

    # ------------------------------------------------------------------
    # Public: write to disk
    # ------------------------------------------------------------------

    def export(self) -> str:
        """
        Write all tables to:
          1. Individual CSV files in output_dir
          2. A single multi-sheet Excel workbook: talash_extracted.xlsx
          3. A human-readable parsing report: parsing_report.csv

        Returns
        -------
        str
            Path to the Excel workbook.
        """
        # Build DataFrames
        dfs = self._build_dataframes()

        # Write individual CSVs
        for name, df in dfs.items():
            csv_path = self.output_dir / f"{name}.csv"
            df.to_csv(csv_path, index=False)
            logger.info("Wrote %d rows -> %s", len(df), csv_path.name)

        # Write multi-sheet Excel workbook
        workbook_path = self.output_dir / "talash_extracted.xlsx"
        with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
            for name, df in dfs.items():
                sheet_name = name[:31]  # Excel sheet names max 31 chars
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                # Auto-fit column widths
                ws = writer.sheets[sheet_name]
                for col_cells in ws.columns:
                    max_len = max(
                        (len(str(c.value)) if c.value is not None else 0)
                        for c in col_cells
                    )
                    ws.column_dimensions[col_cells[0].column_letter].width = (
                        min(max_len + 2, 60)
                    )

        logger.info("Excel workbook written -> %s", workbook_path)

        # Print parsing report to console
        self._print_report(dfs["parsing_report"])

        return str(workbook_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_dataframes(self) -> Dict[str, "pd.DataFrame"]:
        """Convert accumulator lists to DataFrames, enforcing column order."""
        return {
            "candidates":    self._to_df(self._candidates,    CANDIDATES_COLS),
            "education":     self._to_df(self._education,     EDUCATION_COLS),
            "experience":    self._to_df(self._experience,    EXPERIENCE_COLS),
            "skills":        self._to_df(self._skills,        SKILLS_COLS),
            "publications":  self._to_df(self._publications,  PUBLICATIONS_COLS),
            "supervision":   self._to_df(self._supervision,   SUPERVISION_COLS),
            "books":         self._to_df(self._books,         BOOKS_COLS),
            "patents":       self._to_df(self._patents,       PATENTS_COLS),
            "parsing_report": self._to_df(self._report,       REPORT_COLS),
        }

    @staticmethod
    def _to_df(rows: List[Dict], preferred_cols: List[str]) -> "pd.DataFrame":
        """
        Build a DataFrame, placing preferred_cols first (in order),
        then any extra columns added during normalization.
        """
        if not rows:
            return pd.DataFrame(columns=preferred_cols)

        df = pd.DataFrame(rows)
        # Add missing preferred columns as NaN
        for col in preferred_cols:
            if col not in df.columns:
                df[col] = None

        # Reorder: preferred first, then extra columns alphabetically
        extra = sorted(c for c in df.columns if c not in preferred_cols)
        df = df[preferred_cols + extra]
        return df

    @staticmethod
    def _make_candidate_id(filename: str) -> str:
        """Derive a clean candidate_id from the filename stem."""
        # Remove extension if accidentally included
        stem = Path(filename).stem
        # Replace spaces/special chars with underscores
        import re
        clean = re.sub(r"[^\w\-]", "_", stem).strip("_")
        return clean or "unknown"

    def _next_id(self, table: str) -> int:
        """Auto-increment row ID per table."""
        self._row_counters[table] = self._row_counters.get(table, 0) + 1
        return self._row_counters[table]

    @staticmethod
    def _print_report(report_df: "pd.DataFrame"):
        """Print a formatted parsing report to stdout."""
        total     = len(report_df)
        succeeded = len(report_df[report_df["status"] == "success"])
        failed    = len(report_df[report_df["status"] == "failed"])

        print("\n" + "=" * 60)
        print("  TALASH PRE-PROCESSING REPORT")
        print("=" * 60)
        print(f"  Total CVs processed : {total}")
        print(f"  Successful          : {succeeded}")
        print(f"  Failed              : {failed}")
        print("-" * 60)

        for _, row in report_df.iterrows():
            status_icon = "OK" if row["status"] == "success" else "FAIL"
            print(f"\n  {status_icon} {row['source_filename']} ({row['candidate_id']})")
            if row["status"] == "success":
                print(
                    f"     Pages: {row['pages']} | "
                    f"Education: {row['education_count']} | "
                    f"Experience: {row['experience_count']} | "
                    f"Skills: {row['skills_count']} | "
                    f"Publications: {row['publications_count']}"
                )
                print(
                    f"     Supervision: {row['supervision_count']} | "
                    f"Books: {row['books_count']} | "
                    f"Patents: {row['patents_count']}"
                )
                if row.get("missing_fields") and row["missing_fields"] != "OK":
                    print(f"     [!] Missing: {row['missing_fields']}")
                if row.get("warnings"):
                    print(f"     [!] Warnings: {row['warnings']}")
            else:
                print(f"     [FAIL] Error: {row['error_message']}")

        print("\n" + "=" * 60 + "\n")
