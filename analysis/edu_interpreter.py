"""
edu_interpreter.py  –  Task 2.9: Educational Strength Interpretation
======================================================================

WHY THIS FILE EXISTS
--------------------
Task 2.9 is the LAST step and the only one that uses the LLM. It takes
the fully-computed structured output of steps 2.1-2.8 and asks the LLM
to write a human-readable summary and assign one of four strength labels.

DESIGN DECISIONS
----------------
- The LLM receives structured data (not raw CV text) so the prompt is small
  (~1500 tokens per candidate) and very targeted.
- The strength label is assigned first by deterministic rules (see _rule_based_label)
  and the LLM is instructed to use that as a starting point and adjust only
  if it finds strong evidence to the contrary.  This prevents hallucination
  of an unwarranted "Strong" label.
- We do NOT ask the LLM to re-extract data; we ask it only to synthesise
  and narrate what the structured fields already show.
- One Groq API call per candidate (very token-efficient: ~500 in, ~300 out).
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an expert academic evaluator for a university recruitment system.
You will receive structured information about a candidate's educational background.
Your task is to write a concise educational assessment.

OUTPUT RULES:
1. Return ONLY a valid JSON object. No markdown, no code fences, no explanation.
2. The JSON must contain exactly two keys: "educational_strength" and "summary".
3. "educational_strength" must be one of: "Strong", "Moderate", "Weak", "Needs Clarification"
4. "summary" must be 2-4 sentences, written in third person, stating the facts without embellishment.
5. Base your assessment ONLY on the structured data provided. Do not invent or infer facts.

STRENGTH LABEL GUIDE:
- Strong: PhD from ranked university, consistent specialization, strong marks (>75% normalized), gaps justified.
- Moderate: Masters or PhD from unranked university, or minor drift, or some unexplained gaps.
- Weak: Only UG/HSSC, or declining performance, or significant unexplained gaps.
- Needs Clarification: Key fields missing (no marks, no institution, no years).
"""

_USER_PROMPT_TEMPLATE = """Assess this candidate's educational profile and return the JSON.

CANDIDATE ID: {candidate_id}

DEGREES (chronological):
{degrees_text}

HIGHEST DEGREE: {highest_degree}
PROGRESSION CONSISTENT: {progression_consistent}
PERFORMANCE TREND: {performance_trend}
SPECIALIZATION DRIFT: {drift_text}

EDUCATIONAL GAPS:
{gaps_text}

SUGGESTED STRENGTH LABEL (rule-based, adjust only with strong justification): {rule_label}

Return JSON:
"""


class EduInterpreter:
    """
    Task 2.9: LLM-based educational strength interpretation.

    Usage
    -----
    interp = EduInterpreter()
    result = interp.interpret(candidate_id="C001", analysis_result={...})
    # result["educational_strength"] = "Strong" | "Moderate" | "Weak" | "Needs Clarification"
    # result["summary"] = "Dr. X has a PhD from ..."
    """

    def __init__(
        self,
        api_key: Optional[str]  = None,
        model:   str            = "llama-3.3-70b-versatile",
        base_url: Optional[str] = None,
        temperature: float      = 0.1,
    ):
        self.api_key     = api_key    or os.environ.get("OPENAI_API_KEY", "")
        self.model       = model
        self.base_url    = base_url   or os.environ.get("OPENAI_BASE_URL")
        self.temperature = temperature

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def interpret(
        self,
        candidate_id:  str,
        analysis:      Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate educational strength label + summary paragraph for one candidate.

        Parameters
        ----------
        candidate_id : str
            Identifier for the candidate (used in prompt and output).
        analysis : dict
            Output of EducationalProfileAnalyser.analyse_candidate() — the
            combined result of tasks 2.1-2.8.

        Returns
        -------
        dict with keys:
            educational_strength : str
            summary              : str
            rule_based_label     : str   (for transparency / debugging)
        """
        rule_label = self._rule_based_label(analysis)

        user_prompt = _USER_PROMPT_TEMPLATE.format(
            candidate_id        = candidate_id,
            degrees_text        = self._format_degrees(analysis.get("degrees_sorted", [])),
            highest_degree      = analysis.get("highest_degree", "Not provided"),
            progression_consistent = str(analysis.get("progression_consistent", "Unknown")),
            performance_trend   = analysis.get("performance_trend", "insufficient data"),
            drift_text          = self._format_drift(analysis.get("specialization_drift", [])),
            gaps_text           = self._format_gaps(analysis.get("educational_gaps", [])),
            rule_label          = rule_label,
        )

        try:
            raw = self._call_llm(_SYSTEM_PROMPT, user_prompt)
            parsed = self._parse_json(raw, candidate_id)
            if parsed and "educational_strength" in parsed and "summary" in parsed:
                parsed["rule_based_label"] = rule_label
                return parsed
        except Exception as e:
            logger.warning("LLM interpretation failed for '%s': %s", candidate_id, e)

        # Fallback: use rule-based label with generic summary
        return {
            "educational_strength": rule_label,
            "summary": self._rule_based_summary(candidate_id, analysis, rule_label),
            "rule_based_label": rule_label,
        }

    # ------------------------------------------------------------------
    # Rule-based label (deterministic fallback + LLM seed)
    # ------------------------------------------------------------------

    @staticmethod
    def _rule_based_label(analysis: Dict) -> str:
        """
        Assign a strength label using purely deterministic rules.
        This is used to seed the LLM and as a fallback.
        """
        highest   = analysis.get("highest_degree", "Not provided")
        trend     = analysis.get("performance_trend", "insufficient data")
        gaps      = analysis.get("educational_gaps", [])
        degrees   = analysis.get("degrees_sorted", [])

        # Check for missing critical data
        if highest in ("Not provided", "Other"):
            return "Needs Clarification"

        has_ranked = any(
            d.get("the_rank") not in (None, "Not Ranked", "")
            or d.get("qs_rank") not in (None, "Not Ranked", "")
            for d in degrees
        )

        unexplained_gaps = [
            g for g in gaps
            if g.get("significant") and g.get("justified_by") is None
            and g.get("justification_type") == "Unexplained"
        ]

        avg_marks = None
        scores = [
            d.get("marks_normalized")
            for d in degrees
            if d.get("marks_normalized") is not None
        ]
        if scores:
            avg_marks = sum(scores) / len(scores)

        # Strong: PhD/PG from ranked uni, good marks, no unexplained gaps
        if highest == "PhD" and has_ranked and not unexplained_gaps:
            if avg_marks is None or avg_marks >= 65:
                return "Strong"

        if highest in ("PhD", "PG") and not unexplained_gaps:
            if avg_marks and avg_marks >= 75:
                return "Strong"
            return "Moderate"

        if highest == "PG":
            return "Moderate"

        if highest == "UG":
            return "Weak"

        if highest in ("HSSC", "SSC"):
            return "Weak"

        return "Needs Clarification"

    @staticmethod
    def _rule_based_summary(candidate_id: str, analysis: Dict, label: str) -> str:
        """Generate a plain-English summary without LLM."""
        highest = analysis.get("highest_degree", "Not provided")
        trend   = analysis.get("performance_trend", "insufficient data")
        gaps    = [g for g in analysis.get("educational_gaps", []) if g.get("significant")]
        n_unex  = sum(1 for g in gaps if not g.get("justified_by"))

        parts = [f"The candidate's highest qualification is {highest}."]
        if trend != "insufficient data":
            parts.append(f"Academic performance trend is {trend}.")
        if n_unex:
            parts.append(f"There are {n_unex} unexplained gap(s) in the academic timeline.")
        elif gaps:
            parts.append("All identified educational gaps appear to be justified by professional activity.")
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Formatting helpers for the prompt
    # ------------------------------------------------------------------

    @staticmethod
    def _format_degrees(degrees: List[Dict]) -> str:
        if not degrees:
            return "No degree records found."
        lines = []
        for d in degrees:
            lvl   = d.get("standard_level", "?")
            deg   = d.get("degree", "?")
            inst  = d.get("institution", "?")
            yr    = f"{d.get('start_year', '?')}-{d.get('end_year', '?')}"
            norm  = d.get("marks_normalized")
            marks = f"{norm:.1f}%" if norm is not None else "marks not provided"
            the_r = d.get("the_rank") or "Not Ranked"
            qs_r  = d.get("qs_rank")  or "Not Ranked"
            lines.append(
                f"  [{lvl}] {deg} | {inst} ({yr}) | {marks} | "
                f"THE: {the_r} | QS: {qs_r}"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_drift(drift: List[Dict]) -> str:
        if not drift:
            return "None detected"
        return "; ".join(
            f"{d['from_degree'][:40]} -> {d['to_degree'][:40]} (overlap: {d['overlap_score']})"
            for d in drift
        )

    @staticmethod
    def _format_gaps(gaps: List[Dict]) -> str:
        significant = [g for g in gaps if g.get("significant")]
        if not significant:
            return "No significant gaps detected."
        lines = []
        for g in significant:
            justified = g.get("justified_by") or "Unexplained"
            lines.append(
                f"  {g.get('between')}: {g.get('gap_years', '?')} year(s) -> {justified}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    def _call_llm(self, system: str, user: str) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""

    @staticmethod
    def _parse_json(raw: str, name: str) -> Optional[Dict]:
        """Parse JSON from LLM response with fallback extraction."""
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        # Strip markdown fences
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        # Find first {...}
        m = re.search(r"\{.*?\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        logger.error("Could not parse LLM JSON for '%s'. Raw:\n%s", name, raw[:400])
        return None
