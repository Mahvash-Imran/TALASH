"""
book_analyser.py  –  Module 5: Books Authored / Co-Authored Analysis
==================================================================

WHY THIS FILE EXISTS
--------------------
Orchestrates Part 5 (Books Authored / Co-Authored) for all candidates:
  - Reads books.csv and candidates.csv (from Module 1)
  - Evaluates authorship role, publisher credibility, ISBN validity, and verifiability
  - Computes per-candidate book aggregate metrics
  - Calls LLM to generate scholarly assessment paragraphs (with rule-based fallback)
  - Exports book_profiles.csv, book_aggregates.csv, book_aggregates.xlsx, and book_report.txt
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .book_verifier import (
    classify_authorship_role,
    evaluate_publisher_credibility,
    validate_isbn,
    check_book_quality_and_verifiability,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM Prompts
# ---------------------------------------------------------------------------

_BOOK_SYSTEM_PROMPT = (
    "You are an expert academic evaluator for a university recruitment system. "
    "You will receive structured statistics regarding a faculty candidate's authored or co-authored books. "
    "Your task is to assess their book publishing record. "
    "Return ONLY a valid JSON object with exactly two keys: "
    "\"scholarly_label\" and \"book_assessment\". "
    "scholarly_label must be one of: \"Strong\", \"Moderate\", \"Limited\", \"No Books Listed\". "
    "book_assessment must be 2-4 sentences in third person, factual, with no embellishment. "
    "If no books are listed, set scholarly_label to \"No Books Listed\" and state that no books were listed in the CV."
)

_BOOK_USER_TEMPLATE = (
    "Assess this candidate's book publication record and return JSON.\n\n"
    "CANDIDATE ID: {candidate_id}\n\n"
    "BOOK STATISTICS:\n"
    "  Total books                  : {total_books}\n"
    "  Sole authored books          : {sole_authored}\n"
    "  Lead authored books          : {lead_authored}\n"
    "  Co-authored books            : {co_authored}\n"
    "  Contributing/Chapter books   : {contributing_authored}\n"
    "  Recognized Academic publisher: {recognized_pub_count}\n"
    "  Self-published / vanity      : {self_published_count}\n"
    "  Unknown publisher            : {unknown_pub_count}\n"
    "  Verifiable books             : {verifiable_count}\n\n"
    "SUGGESTED LABEL (rule-based): {rule_label}\n\n"
    "Return JSON:"
)


def _rule_based_book_label(
    total_books: int,
    sole_authored: int,
    lead_authored: int,
    recognized_pub_count: int,
    verifiable_count: int,
) -> str:
    """Deterministic label logic for book record."""
    if total_books == 0:
        return "No Books Listed"
    if (sole_authored >= 1 or lead_authored >= 1 or recognized_pub_count >= 1) and verifiable_count >= 1:
        return "Strong"
    if total_books >= 1 and verifiable_count >= 1:
        return "Moderate"
    return "Limited"


class BookProfileAnalyser:
    """
    Orchestrates tasks 5.1–5.5 for every candidate in books.csv and candidates.csv.

    Usage
    -----
    analyser = BookProfileAnalyser(
        books_csv      = "data/extracted/books.csv",
        candidates_csv = "data/extracted/candidates.csv",
        output_dir     = "data/analysis",
    )
    analyser.run()
    """

    def __init__(
        self,
        books_csv:      str = "data/extracted/books.csv",
        candidates_csv: str = "data/extracted/candidates.csv",
        output_dir:     str = "data/analysis",
        api_key:        Optional[str] = None,
        model:          str = "meta-llama/llama-4-scout-17b-16e-instruct",
        base_url:       Optional[str] = None,
        skip_llm:       bool = False,
    ):
        self.books_csv      = Path(books_csv)
        self.candidates_csv = Path(candidates_csv)
        self.output_dir     = Path(output_dir)
        self.skip_llm       = skip_llm
        self.api_key        = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model          = model
        self.base_url       = base_url or os.environ.get("OPENAI_BASE_URL")

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public: run
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Path]:
        logger.info("=" * 60)
        logger.info("  TALASH Module 5 – Books Authored / Co-Authored Analysis")
        logger.info("  Books CSV        : %s", self.books_csv)
        logger.info("  Candidates CSV   : %s", self.candidates_csv)
        logger.info("  Output dir       : %s", self.output_dir)
        logger.info("=" * 60)

        books_df = self._load_csv(self.books_csv, "books")
        cand_df  = self._load_csv(self.candidates_csv, "candidates")

        if cand_df is None or cand_df.empty:
            logger.error("candidates.csv is missing or empty. Run Module 1 first.")
            return {}

        name_map: Dict[str, str] = dict(
            zip(cand_df["candidate_id"].astype(str), cand_df["name"].astype(str))
        )
        all_cids = cand_df["candidate_id"].dropna().unique().tolist()

        # Group books by candidate_id
        books_by_cid: Dict[str, List[Dict]] = {}
        if books_df is not None and not books_df.empty:
            for _, row in books_df.iterrows():
                cid = str(row.get("candidate_id", "")).strip()
                if cid:
                    books_by_cid.setdefault(cid, []).append(row.to_dict())

        # ------------------------------------------------------------------
        # Checkpointing
        # ------------------------------------------------------------------
        checkpoint_path = self.output_dir / "_book_checkpoint.json"
        checkpoint      = self._load_checkpoint(checkpoint_path)
        completed_cids  = set(checkpoint.get("completed", []))

        all_book_profiles: List[Dict] = list(checkpoint.get("book_profiles", []))
        all_aggregates:    List[Dict] = list(checkpoint.get("aggregates",    []))

        if completed_cids:
            logger.info(
                "Checkpoint found: %d candidate(s) already done, resuming.",
                len(completed_cids)
            )

        for i, cid in enumerate(all_cids, 1):
            if cid in completed_cids:
                logger.info("[%d/%d] SKIP (checkpoint): %s", i, len(all_cids), cid)
                continue

            logger.info("[%d/%d] Analysing books for: %s", i, len(all_cids), cid)
            cand_name = name_map.get(str(cid), str(cid))
            raw_books = books_by_cid.get(str(cid), [])

            book_rows, aggregate = self._analyse_candidate(cid, cand_name, raw_books)

            all_book_profiles.extend(book_rows)
            all_aggregates.append(aggregate)

            completed_cids.add(str(cid))
            self._save_checkpoint(checkpoint_path, {
                "completed":     list(completed_cids),
                "book_profiles": all_book_profiles,
                "aggregates":    all_aggregates,
            })

        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info("Checkpoint cleared (run complete).")

        return self._export(all_book_profiles, all_aggregates)

    # ------------------------------------------------------------------
    # Per-Candidate Analysis
    # ------------------------------------------------------------------

    def _analyse_candidate(
        self,
        candidate_id: str,
        candidate_name: str,
        raw_books: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:

        # Filter valid book entries (must have title or authors or ISBN)
        valid_books = [
            b for b in raw_books
            if (str(b.get("title") or "").strip().lower() not in ("nan", "none", "")
                or str(b.get("authors") or "").strip().lower() not in ("nan", "none", "")
                or str(b.get("isbn") or "").strip().lower() not in ("nan", "none", ""))
        ]

        book_records = []
        sole_authored = 0
        lead_authored = 0
        co_authored = 0
        contributing_authored = 0
        recognized_pub_count = 0
        self_published_count = 0
        unknown_pub_count = 0
        verifiable_count = 0

        for b in valid_books:
            title = str(b.get("title") or "").strip()
            authors = str(b.get("authors") or "").strip()
            isbn = str(b.get("isbn") or "").strip()
            publisher = str(b.get("publisher") or "").strip()
            year = str(b.get("year") or "").strip()
            link = str(b.get("link") or b.get("online_link") or "").strip()

            # Tasks 5.2 - 5.4
            role = classify_authorship_role(authors, candidate_name)
            credibility = evaluate_publisher_credibility(publisher)
            isbn_valid = validate_isbn(isbn)

            row_data = {
                "candidate_id": candidate_id,
                "title": title,
                "authors": authors,
                "isbn": isbn,
                "publisher": publisher,
                "year": year,
                "link": link,
                "authorship_role": role,
                "publisher_credibility": credibility,
                "isbn_valid": isbn_valid,
            }

            verifiable, flags = check_book_quality_and_verifiability(row_data, isbn_valid)
            row_data["verifiable"] = verifiable
            row_data["data_quality_flags"] = flags

            book_records.append(row_data)

            # Accumulate stats
            if role == "Sole Author":
                sole_authored += 1
            elif role == "Lead Author":
                lead_authored += 1
            elif role == "Co-Author":
                co_authored += 1
            elif role == "Contributing Author":
                contributing_authored += 1

            if credibility == "Recognized Academic":
                recognized_pub_count += 1
            elif credibility == "Self-Published":
                self_published_count += 1
            else:
                unknown_pub_count += 1

            if verifiable:
                verifiable_count += 1

        total_books = len(book_records)
        rule_label = _rule_based_book_label(
            total_books, sole_authored, lead_authored, recognized_pub_count, verifiable_count
        )

        # Task 5.5: LLM Scholarly Assessment
        if self.skip_llm:
            scholarly_label = rule_label
            assessment = self._rule_based_assessment(
                candidate_name, total_books, sole_authored, lead_authored,
                co_authored, recognized_pub_count, rule_label
            )
        else:
            interp = self._interpret_book_profile(
                candidate_id, total_books, sole_authored, lead_authored,
                co_authored, contributing_authored, recognized_pub_count,
                self_published_count, unknown_pub_count, verifiable_count, rule_label
            )
            scholarly_label = interp.get("scholarly_label", rule_label)
            assessment = interp.get("book_assessment", "")
            time.sleep(0.5)

        aggregate = {
            "candidate_id": candidate_id,
            "candidate_name": candidate_name,
            "total_books": total_books,
            "sole_authored": sole_authored,
            "lead_authored": lead_authored,
            "co_authored": co_authored,
            "contributing_authored": contributing_authored,
            "recognized_publisher_count": recognized_pub_count,
            "self_published_count": self_published_count,
            "unknown_publisher_count": unknown_pub_count,
            "verifiable_books_count": verifiable_count,
            "scholarly_label": scholarly_label,
            "book_assessment": assessment,
            "rule_based_label": rule_label,
        }

        return book_records, aggregate

    # ------------------------------------------------------------------
    # LLM Interpretation
    # ------------------------------------------------------------------

    def _interpret_book_profile(
        self,
        candidate_id: str,
        total_books: int,
        sole_authored: int,
        lead_authored: int,
        co_authored: int,
        contributing_authored: int,
        recognized_pub_count: int,
        self_published_count: int,
        unknown_pub_count: int,
        verifiable_count: int,
        rule_label: str,
    ) -> Dict[str, Any]:
        if total_books == 0:
            return {
                "scholarly_label": "No Books Listed",
                "book_assessment": f"Candidate {candidate_id} has no books listed in their CV."
            }

        user_prompt = _BOOK_USER_TEMPLATE.format(
            candidate_id         = candidate_id,
            total_books          = total_books,
            sole_authored        = sole_authored,
            lead_authored        = lead_authored,
            co_authored          = co_authored,
            contributing_authored= contributing_authored,
            recognized_pub_count = recognized_pub_count,
            self_published_count = self_published_count,
            unknown_pub_count    = unknown_pub_count,
            verifiable_count     = verifiable_count,
            rule_label           = rule_label,
        )

        try:
            raw = self._call_llm(_BOOK_SYSTEM_PROMPT, user_prompt)
            parsed = self._parse_json(raw, candidate_id)
            if parsed and "scholarly_label" in parsed:
                return parsed
        except Exception as e:
            logger.warning("LLM book assessment failed for '%s': %s", candidate_id, e)

        return {
            "scholarly_label": rule_label,
            "book_assessment": self._rule_based_assessment(
                candidate_id, total_books, sole_authored, lead_authored,
                co_authored, recognized_pub_count, rule_label
            ),
        }

    @staticmethod
    def _rule_based_assessment(
        name: str,
        total: int,
        sole: int,
        lead: int,
        co: int,
        recognized: int,
        label: str,
    ) -> str:
        if total == 0:
            return f"Candidate {name} has no books authored or listed in their CV."
        parts = [f"Candidate {name} has authored or co-authored {total} book(s)."]
        if sole:
            parts.append(f"{sole} book(s) as sole author.")
        if lead:
            parts.append(f"{lead} book(s) as lead author.")
        if co:
            parts.append(f"{co} book(s) as co-author.")
        if recognized:
            parts.append(f"{recognized} published with recognized academic presses.")
        parts.append(f"Overall book dissemination contribution is assessed as {label}.")
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export(
        self,
        book_profiles: List[Dict],
        aggregates:    List[Dict],
    ) -> Dict[str, Path]:
        paths: Dict[str, Path] = {}

        label_rank = {"Strong": 1, "Moderate": 2, "Limited": 3, "No Books Listed": 4}
        aggregates = sorted(
            aggregates,
            key=lambda x: (
                label_rank.get(x.get("scholarly_label"), 99),
                x.get("candidate_id", ""),
            )
        )

        # 1. Book Profiles CSV
        p = self.output_dir / "book_profiles.csv"
        pd.DataFrame(book_profiles).to_csv(p, index=False)
        logger.info("Wrote %d rows -> %s", len(book_profiles), p.name)
        paths["book_profiles"] = p

        # 2. Book Aggregates CSV
        p = self.output_dir / "book_aggregates.csv"
        pd.DataFrame(aggregates).to_csv(p, index=False)
        logger.info("Wrote %d rows -> %s", len(aggregates), p.name)
        paths["book_aggregates"] = p

        # 3. Excel Workbook
        p = self.output_dir / "book_aggregates.xlsx"
        with pd.ExcelWriter(p, engine="openpyxl") as writer:
            pd.DataFrame(aggregates).to_excel(
                writer, sheet_name="Book Aggregates", index=False
            )
            pd.DataFrame(book_profiles).to_excel(
                writer, sheet_name="Book Profiles", index=False
            )
        logger.info("Excel workbook written -> %s", p.name)
        paths["excel"] = p

        # 4. Text Report
        self._write_text_report(aggregates, book_profiles, paths)

        return paths

    def _write_text_report(self, aggregates: List[Dict], book_profiles: List[Dict], paths: Dict):
        p = self.output_dir / "book_report.txt"

        label_counts: Dict[str, int] = {}
        for agg in aggregates:
            s = agg.get("scholarly_label") or "Unknown"
            label_counts[s] = label_counts.get(s, 0) + 1

        total_books_found = len(book_profiles)

        lines = [
            "=" * 70,
            "  TALASH Module 5 – Books Authored / Co-Authored Report",
            "=" * 70,
            f"  Candidates analysed           : {len(aggregates)}",
            f"  Total books extracted         : {total_books_found}",
            "",
            "  Scholarly Dissemination Label Distribution:",
        ]
        for lbl, cnt in sorted(label_counts.items()):
            lines.append(f"    {lbl:<25}: {cnt}")

        lines += [
            "",
            "-" * 70,
            "  Per-Candidate Book Summaries",
            "-" * 70,
        ]

        for agg in aggregates:
            cid   = agg["candidate_id"]
            lbl   = agg.get("scholarly_label", "Unknown")
            tot   = agg.get("total_books", 0)
            lines.append(f"\n  [{lbl}] {cid}")
            lines.append(f"  Total Books: {tot} (Sole={agg.get('sole_authored',0)}, Lead={agg.get('lead_authored',0)}, Co={agg.get('co_authored',0)})")
            lines.append(f"  Publisher Credibility: Recognized={agg.get('recognized_publisher_count',0)}, Self-Published={agg.get('self_published_count',0)}, Unknown={agg.get('unknown_publisher_count',0)}")
            if agg.get("book_assessment"):
                lines.append(f"  Summary: {agg['book_assessment']}")

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
