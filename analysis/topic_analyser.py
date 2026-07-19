"""
topic_analyser.py  –  Module 7: Topic Variability & Research Breadth Orchestrator
==============================================================================

WHY THIS FILE EXISTS
--------------------
Orchestrates Part 7 (Topic Variability & Research Breadth Analysis):
  - Reads publications.csv and candidates.csv (from Module 1)
  - Classifies each publication into research themes (LLM with deterministic fallback)
  - Computes candidate-level metrics: theme counts, percentages, dominant theme, Shannon entropy diversity score
  - Analyzes temporal research trends across publication years
  - Classifies profile type: Specialist (>70%), Focused Researcher (50-70%), Interdisciplinary (<=50%)
  - Generates LLM narrative summaries
  - Exports research_breadth_profiles.csv, publication_themes.csv, research_breadth_profiles.xlsx, and research_breadth_report.txt
"""

import json
import logging
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .topic_verifier import (
    RESEARCH_TAXONOMY,
    classify_paper_theme_rule_based,
    calculate_shannon_entropy,
    analyze_temporal_trend,
    classify_research_profile_type,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM Prompts for Batch Paper Classification & Profile Interpretation
# ---------------------------------------------------------------------------

_BATCH_CLASSIFY_SYSTEM = (
    "You are an expert academic taxonomy classifier for computer science, engineering, and technology publications. "
    "Classify each publication title into ONE primary research theme from the provided taxonomy: "
    "1. Natural Language Processing\n"
    "2. Computer Vision & Image Processing\n"
    "3. Machine Learning & Artificial Intelligence\n"
    "4. Wireless Networks & IoT\n"
    "5. Cybersecurity & Privacy\n"
    "6. Biomedical & Health Informatics\n"
    "7. Software Engineering & Web Technologies\n"
    "8. Robotics & Automation\n"
    "9. Energy & Renewable Systems\n"
    "10. Data Science & Big Data\n"
    "11. General Computer Science / Other\n\n"
    "Return ONLY a valid JSON object mapping each 1-indexed paper ID (e.g. \"1\", \"2\") to a JSON dict: "
    "{\"primary_theme\": \"...\", \"secondary_theme\": \"... or null\", \"keywords\": [\"...\"]}"
)

_TOPIC_INTERPRET_SYSTEM = (
    "You are an expert academic research evaluator for a university recruitment system. "
    "You will receive structured research breadth statistics for a candidate. "
    "Your task is to write a concise, factual assessment of their research scope and focus. "
    "Return ONLY a valid JSON object with exactly two keys: "
    "\"research_profile_type\" and \"summary\". "
    "research_profile_type must be one of: \"Specialist\", \"Focused Researcher\", \"Interdisciplinary\", \"No Publications\". "
    "summary must be 2-4 sentences in third person, factual, with no embellishment."
)

_TOPIC_INTERPRET_USER_TEMPLATE = (
    "Assess this candidate's research profile breadth and return JSON.\n\n"
    "CANDIDATE ID: {candidate_id}\n\n"
    "RESEARCH BREADTH STATISTICS:\n"
    "  Total publications           : {total_pubs}\n"
    "  Dominant research theme      : {dominant_theme} ({dominant_pct}%)\n"
    "  Theme breakdown              : {themes_json}\n"
    "  Shannon Entropy Diversity    : {diversity_score} (0.0=focused, 1.0=diverse)\n"
    "  Temporal trend pattern       : {trend_pattern}\n\n"
    "SUGGESTED PROFILE TYPE (rule-based): {rule_type}\n\n"
    "Return JSON:"
)


class TopicBreadthAnalyser:
    """
    Orchestrates tasks 7.1–7.6 for every candidate in publications.csv and candidates.csv.

    Usage
    -----
    analyser = TopicBreadthAnalyser(
        publications_csv = "data/extracted/publications.csv",
        candidates_csv   = "data/extracted/candidates.csv",
        output_dir       = "data/analysis",
    )
    analyser.run()
    """

    def __init__(
        self,
        publications_csv: str = "data/extracted/publications.csv",
        candidates_csv:   str = "data/extracted/candidates.csv",
        output_dir:       str = "data/analysis",
        api_key:          Optional[str] = None,
        model:            str = "meta-llama/llama-4-scout-17b-16e-instruct",
        base_url:         Optional[str] = None,
        skip_llm:         bool = False,
    ):
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
        logger.info("  TALASH Module 7 – Topic Variability & Research Breadth Analysis")
        logger.info("  Publications CSV : %s", self.publications_csv)
        logger.info("  Candidates CSV   : %s", self.candidates_csv)
        logger.info("  Output dir       : %s", self.output_dir)
        logger.info("=" * 60)

        pub_df  = self._load_csv(self.publications_csv, "publications")
        cand_df = self._load_csv(self.candidates_csv,   "candidates")

        if cand_df is None or cand_df.empty:
            logger.error("candidates.csv missing or empty. Run Module 1 first.")
            return {}

        name_map: Dict[str, str] = dict(
            zip(cand_df["candidate_id"].astype(str), cand_df["name"].astype(str))
        )
        all_cids = cand_df["candidate_id"].dropna().unique().tolist()

        # Group publications by candidate_id
        pub_by_cid: Dict[str, List[Dict]] = {}
        if pub_df is not None and not pub_df.empty:
            for _, row in pub_df.iterrows():
                cid = str(row.get("candidate_id", "")).strip()
                if cid:
                    pub_by_cid.setdefault(cid, []).append(row.to_dict())

        # ------------------------------------------------------------------
        # Checkpointing
        # ------------------------------------------------------------------
        checkpoint_path = self.output_dir / "_topic_checkpoint.json"
        checkpoint      = self._load_checkpoint(checkpoint_path)
        completed_cids  = set(checkpoint.get("completed", []))

        all_breadth_profiles: List[Dict] = list(checkpoint.get("breadth_profiles", []))
        all_pub_themes:       List[Dict] = list(checkpoint.get("pub_themes",       []))

        if completed_cids:
            logger.info(
                "Checkpoint found: %d candidate(s) already done, resuming.",
                len(completed_cids)
            )

        for i, cid in enumerate(all_cids, 1):
            if cid in completed_cids:
                logger.info("[%d/%d] SKIP (checkpoint): %s", i, len(all_cids), cid)
                continue

            logger.info("[%d/%d] Analysing research breadth for: %s", i, len(all_cids), cid)
            cand_name = name_map.get(str(cid), str(cid))
            cand_pubs = pub_by_cid.get(str(cid), [])

            breadth_profile, pub_theme_rows = self._analyse_candidate(
                cid, cand_name, cand_pubs
            )

            all_breadth_profiles.append(breadth_profile)
            all_pub_themes.extend(pub_theme_rows)

            completed_cids.add(str(cid))
            self._save_checkpoint(checkpoint_path, {
                "completed":        list(completed_cids),
                "breadth_profiles": all_breadth_profiles,
                "pub_themes":       all_pub_themes,
            })

        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info("Checkpoint cleared (run complete).")

        return self._export(all_breadth_profiles, all_pub_themes)

    # ------------------------------------------------------------------
    # Per-Candidate Analysis
    # ------------------------------------------------------------------

    def _analyse_candidate(
        self,
        candidate_id: str,
        candidate_name: str,
        cand_pubs: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:

        # Filter valid publication entries (must have title)
        valid_pubs = [
            p for p in cand_pubs
            if str(p.get("title") or "").strip().lower() not in ("nan", "none", "")
        ]

        total_pubs = len(valid_pubs)

        if total_pubs == 0:
            empty_profile = {
                "candidate_id":               candidate_id,
                "candidate_name":             candidate_name,
                "total_publications":         0,
                "dominant_theme":             "N/A",
                "dominant_theme_percentage":  0.0,
                "diversity_score":            0.0,
                "raw_shannon_entropy":        0.0,
                "distinct_themes_count":      0,
                "research_profile_type":      "No Publications",
                "trend_pattern":              "N/A",
                "themes_breakdown_json":      "{}",
                "temporal_trend_json":        "[]",
                "summary":                    f"Candidate {candidate_id} has no publications listed in their CV.",
                "rule_based_label":           "No Publications",
            }
            return empty_profile, []

        # Tasks 7.1 - 7.3: Classify paper themes (LLM batch or rule fallback)
        classified_papers = self._classify_papers_batch(candidate_id, valid_pubs)

        pub_theme_rows = []
        theme_counts: Dict[str, int] = {}
        paper_theme_lookup: Dict[str, str] = {}

        for pub, class_info in zip(valid_pubs, classified_papers):
            title     = str(pub.get("title") or "").strip()
            year      = str(pub.get("year") or "").strip()
            venue     = str(pub.get("venue") or "").strip()
            primary   = class_info.get("primary_theme", "General Computer Science / Other")
            secondary = class_info.get("secondary_theme")
            keywords  = class_info.get("keywords", [])

            theme_counts[primary] = theme_counts.get(primary, 0) + 1
            paper_theme_lookup[title] = primary

            pub_theme_rows.append({
                "candidate_id":    candidate_id,
                "title":           title,
                "year":            year,
                "venue":           venue,
                "primary_theme":   primary,
                "secondary_theme": secondary or "",
                "keywords":        ", ".join(keywords) if isinstance(keywords, list) else str(keywords or ""),
            })

        # Task 7.4: Variability & Diversity Measurement
        theme_percentages = {
            thm: round((cnt / total_pubs) * 100.0, 1)
            for thm, cnt in sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
        }

        dominant_theme = max(theme_counts, key=theme_counts.get)
        dominant_pct = theme_percentages[dominant_theme]

        raw_entropy, norm_diversity_score = calculate_shannon_entropy(theme_counts)

        # Task 7.5: Temporal Trend Analysis
        temporal_trend, trend_pattern = analyze_temporal_trend(valid_pubs, paper_theme_lookup)

        # Task 7.6: Research Profile Classification
        rule_type = classify_research_profile_type(total_pubs, dominant_pct)

        if self.skip_llm:
            profile_type = rule_type
            summary = self._rule_based_summary(
                candidate_name, total_pubs, dominant_theme, dominant_pct,
                len(theme_counts), norm_diversity_score, rule_type
            )
        else:
            interp = self._interpret_topic_profile(
                candidate_id, total_pubs, dominant_theme, dominant_pct,
                theme_counts, norm_diversity_score, trend_pattern, rule_type
            )
            profile_type = interp.get("research_profile_type", rule_type)
            summary = interp.get("summary", "")
            time.sleep(0.5)

        breadth_profile = {
            "candidate_id":               candidate_id,
            "candidate_name":             candidate_name,
            "total_publications":         total_pubs,
            "dominant_theme":             dominant_theme,
            "dominant_theme_percentage":  dominant_pct,
            "diversity_score":            norm_diversity_score,
            "raw_shannon_entropy":        raw_entropy,
            "distinct_themes_count":      len(theme_counts),
            "research_profile_type":      profile_type,
            "trend_pattern":              trend_pattern,
            "themes_breakdown_json":      json.dumps(theme_percentages, ensure_ascii=False),
            "temporal_trend_json":        json.dumps(temporal_trend, ensure_ascii=False),
            "summary":                    summary,
            "rule_based_label":           rule_type,
        }

        return breadth_profile, pub_theme_rows

    # ------------------------------------------------------------------
    # Batch Paper Classification (LLM or Rule Fallback)
    # ------------------------------------------------------------------

    def _classify_papers_batch(
        self,
        candidate_id: str,
        valid_pubs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Classify candidate papers into themes using LLM or rule-based fallback."""
        if self.skip_llm:
            return [
                self._rule_classify_single(p)
                for p in valid_pubs
            ]

        # Build prompt listing candidate paper titles
        paper_lines = []
        for idx, p in enumerate(valid_pubs, 1):
            t = str(p.get("title") or "").strip()
            v = str(p.get("venue") or "").strip()
            paper_lines.append(f"Paper {idx}: \"{t}\" (Venue: {v})")

        user_prompt = "Classify these publication titles into primary_theme, secondary_theme, and keywords:\n\n" + "\n".join(paper_lines)

        try:
            raw = self._call_llm(_BATCH_CLASSIFY_SYSTEM, user_prompt)
            parsed = self._parse_json(raw, candidate_id)
            if parsed and isinstance(parsed, dict):
                results = []
                for idx in range(1, len(valid_pubs) + 1):
                    item = parsed.get(str(idx)) or parsed.get(idx) or {}
                    if not item.get("primary_theme"):
                        item = self._rule_classify_single(valid_pubs[idx - 1])
                    results.append(item)
                if len(results) == len(valid_pubs):
                    return results
        except Exception as e:
            logger.warning("LLM batch theme classification failed for '%s': %s", candidate_id, e)

        # Fallback to rule-based classification
        return [self._rule_classify_single(p) for p in valid_pubs]

    @staticmethod
    def _rule_classify_single(pub: Dict[str, Any]) -> Dict[str, Any]:
        t = str(pub.get("title") or "").strip()
        v = str(pub.get("venue") or "").strip()
        pri, sec, kws = classify_paper_theme_rule_based(t, v)
        return {
            "primary_theme": pri,
            "secondary_theme": sec,
            "keywords": kws,
        }

    # ------------------------------------------------------------------
    # LLM Interpretation
    # ------------------------------------------------------------------

    def _interpret_topic_profile(
        self,
        candidate_id: str,
        total_pubs: int,
        dominant_theme: str,
        dominant_pct: float,
        theme_counts: Dict[str, int],
        diversity_score: float,
        trend_pattern: str,
        rule_type: str,
    ) -> Dict[str, Any]:
        user_prompt = _TOPIC_INTERPRET_USER_TEMPLATE.format(
            candidate_id   = candidate_id,
            total_pubs     = total_pubs,
            dominant_theme = dominant_theme,
            dominant_pct   = dominant_pct,
            themes_json    = json.dumps(theme_counts, ensure_ascii=False),
            diversity_score= diversity_score,
            trend_pattern  = trend_pattern,
            rule_type      = rule_type,
        )

        try:
            raw = self._call_llm(_TOPIC_INTERPRET_SYSTEM, user_prompt)
            parsed = self._parse_json(raw, candidate_id)
            if parsed and "research_profile_type" in parsed:
                return parsed
        except Exception as e:
            logger.warning("LLM topic breadth assessment failed for '%s': %s", candidate_id, e)

        return {
            "research_profile_type": rule_type,
            "summary": self._rule_based_summary(
                candidate_id, total_pubs, dominant_theme, dominant_pct,
                len(theme_counts), diversity_score, rule_type
            ),
        }

    @staticmethod
    def _rule_based_summary(
        name: str,
        total: int,
        dominant_theme: str,
        dominant_pct: float,
        num_themes: int,
        diversity_score: float,
        label: str,
    ) -> str:
        if total == 0:
            return f"Candidate {name} has no publications listed in their CV."
        parts = [
            f"Candidate {name} has authored {total} publication(s) across {num_themes} research theme(s).",
            f"The primary research concentration is in {dominant_theme}, accounting for {dominant_pct:.1f}% of all publications.",
            f"With a normalized diversity score of {diversity_score:.2f}, the overall research profile is classified as {label}."
        ]
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export(
        self,
        breadth_profiles: List[Dict],
        pub_themes:       List[Dict],
    ) -> Dict[str, Path]:
        paths: Dict[str, Path] = {}

        label_rank = {"Specialist": 1, "Focused Researcher": 2, "Interdisciplinary": 3, "No Publications": 4}
        breadth_profiles = sorted(
            breadth_profiles,
            key=lambda x: (
                label_rank.get(x.get("research_profile_type"), 99),
                x.get("candidate_id", ""),
            )
        )

        # 1. Publication Themes CSV
        p = self.output_dir / "publication_themes.csv"
        pd.DataFrame(pub_themes).to_csv(p, index=False)
        logger.info("Wrote %d rows -> %s", len(pub_themes), p.name)
        paths["publication_themes"] = p

        # 2. Research Breadth Profiles CSV
        p = self.output_dir / "research_breadth_profiles.csv"
        pd.DataFrame(breadth_profiles).to_csv(p, index=False)
        logger.info("Wrote %d rows -> %s", len(breadth_profiles), p.name)
        paths["breadth_profiles"] = p

        # 3. Excel Workbook
        p = self.output_dir / "research_breadth_profiles.xlsx"
        with pd.ExcelWriter(p, engine="openpyxl") as writer:
            pd.DataFrame(breadth_profiles).to_excel(
                writer, sheet_name="Research Breadth Profiles", index=False
            )
            pd.DataFrame(pub_themes).to_excel(
                writer, sheet_name="Publication Themes", index=False
            )
        logger.info("Excel workbook written -> %s", p.name)
        paths["excel"] = p

        # 4. Text Report
        self._write_text_report(breadth_profiles, pub_themes, paths)

        return paths

    def _write_text_report(self, breadth_profiles: List[Dict], pub_themes: List[Dict], paths: Dict):
        p = self.output_dir / "research_breadth_report.txt"

        type_counts: Dict[str, int] = {}
        for bp in breadth_profiles:
            t = bp.get("research_profile_type") or "Unknown"
            type_counts[t] = type_counts.get(t, 0) + 1

        lines = [
            "=" * 70,
            "  TALASH Module 7 – Topic Variability & Research Breadth Report",
            "=" * 70,
            f"  Candidates analysed           : {len(breadth_profiles)}",
            f"  Total publications classified : {len(pub_themes)}",
            "",
            "  Research Profile Type Distribution:",
        ]
        for lbl, cnt in sorted(type_counts.items()):
            lines.append(f"    {lbl:<25}: {cnt}")

        lines += [
            "",
            "-" * 70,
            "  Per-Candidate Research Breadth Summaries",
            "-" * 70,
        ]

        for bp in breadth_profiles:
            cid  = bp["candidate_id"]
            ptype = bp.get("research_profile_type", "Unknown")
            tot  = bp.get("total_publications", 0)
            dom  = bp.get("dominant_theme", "N/A")
            pct  = bp.get("dominant_theme_percentage", 0.0)
            div  = bp.get("diversity_score", 0.0)

            lines.append(f"\n  [{ptype}] {cid}")
            lines.append(f"  Total Publications: {tot} | Dominant Theme: {dom} ({pct:.1f}%)")
            lines.append(f"  Diversity Score (Shannon Entropy): {div:.2f} | Trend Pattern: {bp.get('trend_pattern','N/A')}")
            if bp.get("themes_breakdown_json"):
                lines.append(f"  Theme Breakdown: {bp['themes_breakdown_json']}")
            if bp.get("summary"):
                lines.append(f"  Summary: {bp['summary']}")

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
            max_tokens  = 2048,
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
