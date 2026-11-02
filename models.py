"""
models.py  –  Universal Core Pydantic Models & Data Schemas for TALASH
======================================================================

WHY THIS MODULE EXISTS
----------------------
Defines the single source of truth for TALASH's department-agnostic, transparent,
and explainable CV ranking and evaluation architecture:
  1. Universal Dimensions & Role Presets:
     - 8 Universal evaluation dimensions (0-100 scale).
     - 5 Role presets modifying baseline dimensional weights.
     - Hard gate eligibility statuses & candidate quality tiers.
  2. Normalized Candidate Schema (NormalizedCV):
     - Comprehensive, typed CV representation with scrubbable PII for blind fairness.
     - Direct factory method `from_extracted_tables()` for 100% backward compatibility
       with TALASH's legacy CSV tables (data/extracted/*.csv).
  3. Department Profile Schema (DepartmentProfile):
     - Declarative configuration defining baseline weights, role modifiers, hard gates,
       field baselines, artifact taxonomies, credential rules, and skill graphs.
  4. Scoring Results & Audit Output:
     - Detailed dimensional breakdowns, hard gate audits, and `CandidateEvaluationResult`.
     - Backward compatibility adapter `to_legacy_dict()` to seamlessly feed legacy
       dashboards, reports, and API responses without regressions.
"""

from __future__ import annotations

import datetime
import hashlib
import math
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ============================================================================
# 1. ENUMS & CONSTANTS
# ============================================================================

class UniversalDimension(str, Enum):
    """
    The 8 canonical evaluation dimensions scored on a 0-100 scale.
    Department-agnostic: every discipline evaluates across these same 8 pillars.
    """
    ACADEMIC_FOUNDATION = "academic_foundation"
    DOMAIN_COMPETENCE = "domain_competence"
    PROFESSIONAL_EXPERIENCE = "professional_experience"
    EVIDENCE_OF_OUTPUT = "evidence_of_output"
    CREDENTIALS_LICENSURE = "credentials_licensure"
    IMPACT_RECOGNITION = "impact_recognition"
    LEADERSHIP_TEACHING_SERVICE = "leadership_teaching_service"
    TRAJECTORY_GROWTH = "trajectory_growth"


class RolePreset(str, Enum):
    """
    Hiring context role presets. Modifies dimensional weights based on the target position.
    """
    FACULTY = "faculty"
    RESEARCHER = "researcher"
    INDUSTRY_PROFESSIONAL = "industry_professional"
    FRESH_GRADUATE = "fresh_graduate"
    EXECUTIVE = "executive"


class GateStatus(str, Enum):
    """
    Hard pass/fail gate eligibility classification.
    Hard gates never silently alter numeric scores; they gate eligibility explicitly.
    """
    ELIGIBLE = "eligible"
    NOT_ELIGIBLE = "not_eligible"
    NEEDS_VERIFICATION = "needs_verification"


class CandidateTier(str, Enum):
    """
    Standardized tier classifications aligned with TALASH composite evaluation criteria.
    """
    TIER_1 = "Tier 1: Exceptional Fit"          # Score >= 80.0
    TIER_2 = "Tier 2: Strong Candidate"         # Score 65.0 - 79.99
    TIER_3 = "Tier 3: Moderate Candidate"       # Score 50.0 - 64.99
    TIER_4 = "Tier 4: Needs Clarification/Weak" # Score < 50.0


class ArtifactCategory(str, Enum):
    """
    Department-defined artifact classification categories.
    """
    SCHOLARLY_PUBLICATION = "scholarly_publication"  # Journals, conferences, preprints
    PATENT_INVENTION = "patent_invention"            # Issued/pending patents, PCT filings
    BOOK_MONOGRAPH = "book_monograph"                # Authored books, edited volumes, book chapters
    SOFTWARE_CODE = "software_code"                  # Repositories, software packages, open source
    CREATIVE_DESIGN = "creative_design"              # Design portfolios, architectural plans, exhibitions
    CLINICAL_CASE = "clinical_case"                  # Medical cases, surgical logs, clinical trials
    LEGAL_PROCEEDING = "legal_proceeding"            # Briefs, argued cases, judicial opinions
    POLICY_BRIEF = "policy_brief"                    # Whitepapers, policy advisories, government reports
    COMMERCIAL_VENTURE = "commercial_venture"        # Startups founded, products launched, spin-outs
    OTHER = "other"


class FlagSeverity(str, Enum):
    """Severity ratings for fairness audit flags."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================================
# 2. NORMALIZED CV SUB-MODELS (PERSONAL DATA & EVIDENCE)
# ============================================================================

class PersonalIdentifiers(BaseModel):
    """
    Personally Identifiable Information (PII) extracted from CV.
    Stripped or masked during blind evaluation to prevent cognitive bias.
    """
    model_config = ConfigDict(extra="ignore")

    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    cnic: Optional[str] = None
    gender: Optional[str] = None
    photo_url: Optional[str] = None
    age: Optional[int] = None
    date_of_birth: Optional[str] = None
    marital_status: Optional[str] = None
    nationality: Optional[str] = None

    def anonymized_copy(self) -> PersonalIdentifiers:
        """Returns a scrubbed copy where all identifying fields are cleared."""
        return PersonalIdentifiers(
            name="ANONYMIZED_CANDIDATE",
            email=None,
            phone=None,
            address=None,
            cnic=None,
            gender=None,
            photo_url=None,
            age=None,
            date_of_birth=None,
            marital_status=None,
            nationality=None,
        )


class EducationRecord(BaseModel):
    """
    Structured academic qualification record with GPA normalization capabilities.
    """
    model_config = ConfigDict(extra="ignore")

    row_id: Optional[int] = None
    level: str = Field(..., description="e.g. PhD, MS, MPhil, BS, Bachelor, MBBS, JD, Postdoc")
    degree: str = Field(..., description="Formal degree title (e.g. Electrical Engineering, Law, Surgery)")
    specialization: Optional[str] = None
    institution: Optional[str] = None
    country: Optional[str] = None
    start_year: Optional[float] = None
    end_year: Optional[float] = None
    marks_percentage: Optional[float] = None
    cgpa: Optional[float] = None
    cgpa_scale: Optional[float] = Field(default=4.0, description="Scale base (4.0, 5.0, 10.0, 100.0)")
    letter_grade: Optional[str] = None
    cgpa_normalized_1: Optional[float] = Field(
        default=None,
        description="GPA normalized strictly to a 0.0 - 1.0 floating point scale"
    )
    cgpa_normalized_4: Optional[float] = Field(
        default=None,
        description="GPA normalized to 4.0 scale (preserved for TALASH v1 backward compatibility)"
    )
    institution_tier: Optional[str] = Field(
        default=None,
        description="Categorical tier (Tier 1, Tier 2, Top 100, etc.)"
    )
    institution_rank: Optional[str] = None
    qs_rank: Optional[str] = None
    the_rank: Optional[str] = None
    is_verified: bool = True

    @staticmethod
    def normalize_gpa_to_unit(
        gpa: Optional[float] = None,
        scale: Optional[float] = None,
        percentage: Optional[float] = None,
        letter: Optional[str] = None,
    ) -> Optional[float]:
        """
        Robustly normalizes any academic grading format into a standard 0.0 - 1.0 unit scale:
          - Scale-based (4.0, 5.0, 10.0, 20.0)
          - Percentage marks (0-100%)
          - Common Pakistani & Commonwealth divisional honors / letter grades
        """
        # 1. Percentage marks
        if percentage is not None and not math.isnan(percentage):
            if percentage > 0.0:
                p = min(max(percentage, 0.0), 100.0)
                return round(p / 100.0, 4)

        # 2. Explicit scale conversion
        if gpa is not None and not math.isnan(gpa):
            g = float(gpa)
            if scale is not None and not math.isnan(scale) and scale > 0.0:
                s = float(scale)
                if s == 4.0:
                    return round(min(max(g / 4.0, 0.0), 1.0), 4)
                elif s == 5.0:
                    return round(min(max(g / 5.0, 0.0), 1.0), 4)
                elif s == 10.0:
                    return round(min(max(g / 10.0, 0.0), 1.0), 4)
                elif s >= 20.0:
                    return round(min(max(g / s, 0.0), 1.0), 4)
                else:
                    return round(min(max(g / s, 0.0), 1.0), 4)

            # Heuristic scale inference if scale not explicitly provided
            if g <= 4.0:
                return round(min(max(g / 4.0, 0.0), 1.0), 4)
            elif g <= 5.0:
                return round(min(max(g / 5.0, 0.0), 1.0), 4)
            elif g <= 10.0:
                return round(min(max(g / 10.0, 0.0), 1.0), 4)
            elif g <= 100.0:
                return round(min(max(g / 100.0, 0.0), 1.0), 4)

        # 3. Letter grade / division conversion
        if letter:
            clean = letter.strip().upper()
            # Check phrases first
            division_map = {
                "FIRST DIVISION": 0.85,
                "1ST DIVISION": 0.85,
                "SECOND DIVISION": 0.65,
                "2ND DIVISION": 0.65,
                "THIRD DIVISION": 0.50,
                "3RD DIVISION": 0.50,
                "DISTINCTION": 0.95,
                "HIGH HONORS": 0.95,
                "HONORS": 0.90,
                "PASS": 0.50,
            }
            for phrase, val in division_map.items():
                if phrase in clean:
                    return val

            # Check exact letter grades or bounded letter grade patterns
            letter_map = {
                "A+": 1.0, "A": 0.95, "A-": 0.90,
                "B+": 0.85, "B": 0.78, "B-": 0.70,
                "C+": 0.65, "C": 0.58, "C-": 0.50,
                "D+": 0.45, "D": 0.40, "D-": 0.35,
                "F": 0.0,
            }
            # Direct match
            if clean in letter_map:
                return letter_map[clean]

            # Bounded regex match for letter grades within notes like 'Grade: A'
            m = re.search(r'\b([A-DF][+-]?)\b', clean)
            if m and m.group(1) in letter_map:
                return letter_map[m.group(1)]

        return None

    @model_validator(mode="after")
    def populate_normalized_gpas(self) -> EducationRecord:
        """Automatically populates normalized GPA unit and 4.0 scales."""
        if self.cgpa_normalized_1 is None:
            norm = self.normalize_gpa_to_unit(
                gpa=self.cgpa,
                scale=self.cgpa_scale,
                percentage=self.marks_percentage,
                letter=self.letter_grade,
            )
            self.cgpa_normalized_1 = norm

        if self.cgpa_normalized_4 is None and self.cgpa_normalized_1 is not None:
            self.cgpa_normalized_4 = round(self.cgpa_normalized_1 * 4.0, 2)

        return self


class ExperienceRecord(BaseModel):
    """
    Work experience, employment tenure, and organizational responsibility record.
    """
    model_config = ConfigDict(extra="ignore")

    row_id: Optional[int] = None
    job_title: str
    organization: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    duration_years: float = Field(default=0.0, ge=0.0)
    is_current: bool = False
    seniority_level: Optional[str] = Field(
        default=None,
        description="entry, junior, mid, senior, lead, director, executive, professor, etc."
    )
    employment_type: Optional[str] = Field(
        default="full_time",
        description="full_time, part_time, visiting, contract, internship"
    )
    description: Optional[str] = None
    responsibilities: List[str] = Field(default_factory=list)
    scope_scale: Optional[str] = Field(
        default=None,
        description="Team size, budget managed, lab facilities or organizational scope"
    )
    is_relevant: Optional[bool] = Field(
        default=None,
        description="Whether this role matches the target department / job scope"
    )


class SkillRecord(BaseModel):
    """
    Extracted candidate skill or competency with evidence tracking.
    """
    model_config = ConfigDict(extra="ignore")

    row_id: Optional[int] = None
    skill_name: str
    category: Optional[str] = Field(
        default="domain",
        description="core, domain, technical, methodological, tool, soft, language"
    )
    proficiency: Optional[str] = Field(
        default=None,
        description="beginner, intermediate, advanced, expert"
    )
    years_of_experience: Optional[float] = None
    evidence_count: int = Field(default=1, ge=0)
    verified_sources: List[str] = Field(
        default_factory=list,
        description="Locations where skill was corroborated (e.g. 'experience:2', 'pubs:5')"
    )


class EvidenceArtifactRecord(BaseModel):
    """
    Department-defined output artifact (papers, patents, code, designs, cases, etc.).
    """
    model_config = ConfigDict(extra="ignore")

    row_id: Optional[int] = None
    artifact_type: str = Field(
        ...,
        description="paper, patent, book, code_repo, clinical_case, legal_case, exhibition, design, etc."
    )
    category: ArtifactCategory = ArtifactCategory.OTHER
    title: str
    venue_or_platform: Optional[str] = Field(
        default=None,
        description="Journal, conference, publisher, patent office, GitHub, courtroom, gallery, etc."
    )
    year: Optional[float] = None
    authors_or_collaborators: List[str] = Field(default_factory=list)
    role: Optional[str] = Field(
        default=None,
        description="first_author, corresponding, lead_inventor, co_author, sole_counsel, etc."
    )
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Citations, impact factor, HEC category (W/X/Y), GitHub stars, downloads, etc."
    )
    url_or_doi: Optional[str] = None
    is_verified: bool = True


class CredentialLicensureRecord(BaseModel):
    """
    Professional certifications, board memberships, and mandatory legal/technical licenses.
    """
    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., description="e.g. PEC, PMDC/PMC, Bar Council License, PMP, AWS Architect")
    issuing_body: str = Field(..., description="Issuing entity, board, or certification agency")
    license_number: Optional[str] = None
    issue_date: Optional[str] = None
    expiry_date: Optional[str] = None
    status: str = Field(default="active", description="active, expired, pending, in_progress")
    verification_url: Optional[str] = None
    is_verified: bool = False


class ImpactRecognitionRecord(BaseModel):
    """
    Awards, competitive grants, citations, measurable business/scientific impact, or press.
    """
    model_config = ConfigDict(extra="ignore")

    title: str
    category: str = Field(
        default="award",
        description="award, grant, honor, keynote, citation_milestone, press, fellowship"
    )
    granting_body: Optional[str] = None
    amount: Optional[float] = Field(default=None, ge=0.0)
    currency: Optional[str] = "PKR"
    year: Optional[float] = None
    description: Optional[str] = None
    measurable_outcome: Optional[str] = None


class LeadershipTeachingRecord(BaseModel):
    """
    Student supervision, pedagogical roles, committee service, and institutional leadership.
    """
    model_config = ConfigDict(extra="ignore")

    category: str = Field(
        default="supervision",
        description="supervision, teaching, mentoring, committee, administration, editorial"
    )
    title_or_role: str
    organization_or_program: Optional[str] = None
    level: Optional[str] = Field(
        default=None,
        description="PhD, MS, Undergraduate, Departmental, National, International"
    )
    student_name: Optional[str] = None
    thesis_title: Optional[str] = None
    mentees_count: int = Field(default=0, ge=0)
    joint_publications_count: int = Field(default=0, ge=0)
    year: Optional[float] = None


class TrajectoryGrowthRecord(BaseModel):
    """
    Signals of career velocity, progression pacing, coherence, and recent upskilling.
    """
    model_config = ConfigDict(extra="ignore")

    promotion_pace: Optional[str] = Field(
        default=None,
        description="accelerated, standard, steady, slow, unclear"
    )
    career_coherence: Optional[str] = Field(
        default=None,
        description="coherent, multidisciplinary, pivoting, fragmented"
    )
    recent_upskilling: List[str] = Field(default_factory=list)
    velocity_score: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    notable_inflections: List[str] = Field(default_factory=list)


class FairnessAuditFlag(BaseModel):
    """
    Non-penalizing fairness audit item generated when anomalies/gaps are identified.
    Surfaced to hiring managers for human inquiry instead of automatic silent score cuts.
    """
    model_config = ConfigDict(extra="ignore")

    code: str = Field(..., description="Unique machine code (e.g. CAREER_GAP, MISSING_DEGREE_DATES)")
    severity: FlagSeverity = FlagSeverity.MEDIUM
    section: str = Field(..., description="CV section where anomaly originated")
    description: str = Field(..., description="Explanation of what was detected")
    suggested_inquiry: Optional[str] = Field(
        default=None,
        description="Constructive interview question to clarify this flag with the candidate"
    )


# ============================================================================
# 3. ROOT NORMALIZED CV SCHEMA
# ============================================================================

class NormalizedCV(BaseModel):
    """
    Canonical, normalized representation of an applicant CV across all disciplines.
    Provides complete backward compatibility with TALASH extracted tabular data.
    """
    model_config = ConfigDict(extra="ignore")

    candidate_id: str
    raw_text: Optional[str] = None
    source_filename: Optional[str] = None
    processed_at: Optional[str] = None

    personal_identifiers: PersonalIdentifiers = Field(default_factory=PersonalIdentifiers)
    education: List[EducationRecord] = Field(default_factory=list)
    experience: List[ExperienceRecord] = Field(default_factory=list)
    skills: List[SkillRecord] = Field(default_factory=list)
    artifacts: List[EvidenceArtifactRecord] = Field(default_factory=list)
    credentials: List[CredentialLicensureRecord] = Field(default_factory=list)
    impact_recognition: List[ImpactRecognitionRecord] = Field(default_factory=list)
    leadership_teaching: List[LeadershipTeachingRecord] = Field(default_factory=list)
    trajectory_growth: Optional[TrajectoryGrowthRecord] = None
    flags: List[FairnessAuditFlag] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def anonymized_copy(self) -> NormalizedCV:
        """
        Creates a copy of the candidate CV with all personal identifiers scrubbed,
        ensuring fair, blind numeric evaluation free of demographic bias.
        """
        anonymized_id = hashlib.sha256(self.candidate_id.encode("utf-8")).hexdigest()[:12]
        cv_copy = self.model_copy(deep=True)
        cv_copy.candidate_id = f"cand_{anonymized_id}"
        cv_copy.personal_identifiers = self.personal_identifiers.anonymized_copy()
        if cv_copy.raw_text:
            # Mask out raw text to prevent accidental leakage in downstream LLM calls
            cv_copy.raw_text = "[SCRUBBED FOR BLIND FAIR EVALUATION]"
        return cv_copy

    @classmethod
    def from_extracted_tables(
        cls,
        candidate_id: str,
        cand_row: Optional[Dict[str, Any]] = None,
        education_rows: Optional[List[Dict[str, Any]]] = None,
        experience_rows: Optional[List[Dict[str, Any]]] = None,
        skills_rows: Optional[List[Dict[str, Any]]] = None,
        publication_rows: Optional[List[Dict[str, Any]]] = None,
        book_rows: Optional[List[Dict[str, Any]]] = None,
        patent_rows: Optional[List[Dict[str, Any]]] = None,
        supervision_rows: Optional[List[Dict[str, Any]]] = None,
    ) -> NormalizedCV:
        """
        Factory method constructing a NormalizedCV directly from TALASH's extracted CSV rows.
        Guarantees 100% backward compatibility with legacy pipelines.
        """
        cand_row = cand_row or {}
        education_rows = education_rows or []
        experience_rows = experience_rows or []
        skills_rows = skills_rows or []
        publication_rows = publication_rows or []
        book_rows = book_rows or []
        patent_rows = patent_rows or []
        supervision_rows = supervision_rows or []

        # 1. Personal Identifiers
        pii = PersonalIdentifiers(
            name=_safe_str(cand_row.get("name") or cand_row.get("candidate_name")),
            email=_safe_str(cand_row.get("email")),
            phone=_safe_str(cand_row.get("phone")),
            address=_safe_str(cand_row.get("address")),
            cnic=_safe_str(cand_row.get("cnic")),
            source_filename=_safe_str(cand_row.get("source_filename")),
        )

        # 2. Education Records
        edus: List[EducationRecord] = []
        for r in education_rows:
            level = _safe_str(r.get("level")) or "BS"
            deg = _safe_str(r.get("degree")) or level
            edus.append(
                EducationRecord(
                    row_id=_safe_int(r.get("row_id")),
                    level=level,
                    degree=deg,
                    specialization=_safe_str(r.get("specialization")),
                    institution=_safe_str(r.get("institution")),
                    country=_safe_str(r.get("country")),
                    start_year=_safe_float(r.get("start_year")),
                    end_year=_safe_float(r.get("end_year")),
                    marks_percentage=_safe_float(r.get("marks_percentage")),
                    cgpa=_safe_float(r.get("cgpa")),
                    cgpa_scale=_safe_float(r.get("cgpa_scale")) or 4.0,
                    cgpa_normalized_4=_safe_float(r.get("cgpa_normalized_4")),
                    institution_tier=_safe_str(r.get("the_rank_range") or r.get("qs_rank_range")),
                    the_rank=_safe_str(r.get("the_rank")),
                    qs_rank=_safe_str(r.get("qs_rank")),
                )
            )

        # 3. Experience Records
        exps: List[ExperienceRecord] = []
        for r in experience_rows:
            title = _safe_str(r.get("job_title")) or "Professional"
            org = _safe_str(r.get("organization")) or "Organization"
            exps.append(
                ExperienceRecord(
                    row_id=_safe_int(r.get("row_id")),
                    job_title=title,
                    organization=org,
                    start_date=_safe_str(r.get("start_date")),
                    end_date=_safe_str(r.get("end_date")),
                    employment_type=_safe_str(r.get("employment_type")) or "full_time",
                    description=_safe_str(r.get("description")),
                )
            )

        # 4. Skills Records
        sks: List[SkillRecord] = []
        for r in skills_rows:
            sk_name = _safe_str(r.get("skill_name"))
            if sk_name:
                sks.append(
                    SkillRecord(
                        row_id=_safe_int(r.get("row_id")),
                        skill_name=sk_name,
                        category=_safe_str(r.get("category")) or "domain",
                    )
                )

        # 5. Output Artifacts (Publications + Books + Patents)
        artifacts: List[EvidenceArtifactRecord] = []

        # Publications
        for r in publication_rows:
            title = _safe_str(r.get("title"))
            if not title:
                continue
            p_type = (_safe_str(r.get("type")) or "journal").lower()
            authors_str = _safe_str(r.get("authors")) or ""
            authors_list = [a.strip() for a in authors_str.split(",") if a.strip()]
            artifacts.append(
                EvidenceArtifactRecord(
                    row_id=_safe_int(r.get("row_id")),
                    artifact_type=p_type,
                    category=ArtifactCategory.SCHOLARLY_PUBLICATION,
                    title=title,
                    venue_or_platform=_safe_str(r.get("venue")),
                    year=_safe_float(r.get("year")),
                    authors_or_collaborators=authors_list,
                    url_or_doi=_safe_str(r.get("doi") or r.get("url") or r.get("link")),
                )
            )

        # Books
        for r in book_rows:
            title = _safe_str(r.get("title"))
            if not title:
                continue
            authors_str = _safe_str(r.get("authors")) or ""
            authors_list = [a.strip() for a in authors_str.split(",") if a.strip()]
            artifacts.append(
                EvidenceArtifactRecord(
                    row_id=_safe_int(r.get("row_id")),
                    artifact_type="book",
                    category=ArtifactCategory.BOOK_MONOGRAPH,
                    title=title,
                    venue_or_platform=_safe_str(r.get("publisher")),
                    year=_safe_float(r.get("year")),
                    authors_or_collaborators=authors_list,
                    metrics={"isbn": _safe_str(r.get("isbn"))},
                    url_or_doi=_safe_str(r.get("link")),
                )
            )

        # Patents
        for r in patent_rows:
            title = _safe_str(r.get("title"))
            if not title:
                continue
            inv_str = _safe_str(r.get("inventors")) or ""
            inv_list = [i.strip() for i in inv_str.split(",") if i.strip()]
            artifacts.append(
                EvidenceArtifactRecord(
                    row_id=_safe_int(r.get("row_id")),
                    artifact_type="patent",
                    category=ArtifactCategory.PATENT_INVENTION,
                    title=title,
                    venue_or_platform=_safe_str(r.get("country")) or "Patent Office",
                    metrics={"patent_number": _safe_str(r.get("patent_number"))},
                    authors_or_collaborators=inv_list,
                    url_or_doi=_safe_str(r.get("link")),
                )
            )

        # 6. Leadership & Teaching (Supervision)
        leadership: List[LeadershipTeachingRecord] = []
        for r in supervision_rows:
            s_name = _safe_str(r.get("student_name"))
            t_title = _safe_str(r.get("thesis_title"))
            if not s_name and not t_title:
                continue
            leadership.append(
                LeadershipTeachingRecord(
                    category="supervision",
                    title_or_role=_safe_str(r.get("role")) or "Supervisor",
                    level=_safe_str(r.get("level")) or "MS",
                    student_name=s_name,
                    thesis_title=t_title,
                    year=_safe_float(r.get("year")),
                )
            )

        return cls(
            candidate_id=candidate_id,
            source_filename=_safe_str(cand_row.get("source_filename")),
            processed_at=_safe_str(cand_row.get("processed_at")),
            personal_identifiers=pii,
            education=edus,
            experience=exps,
            skills=sks,
            artifacts=artifacts,
            leadership_teaching=leadership,
        )


# ============================================================================
# 4. DEPARTMENT PROFILE SCHEMA & RULES
# ============================================================================

class HardGateRule(BaseModel):
    """
    Mandatory hard pass/fail qualification check executed before scoring.
    """
    model_config = ConfigDict(extra="ignore")

    gate_id: str
    name: str
    dimension: UniversalDimension
    field_path: str = Field(
        ...,
        description="Dot-notated path in NormalizedCV (e.g. 'education.highest_degree', 'credentials.has_pec')"
    )
    operator: str = Field(
        ...,
        description="gte, gt, lte, lt, eq, neq, in, contains, exists, custom"
    )
    expected_value: Any
    failure_status: GateStatus = GateStatus.NOT_ELIGIBLE
    failure_message: str


class BaselineMetrics(BaseModel):
    """
    Empirical field baselines used for percentile normalization across departments.
    """
    model_config = ConfigDict(extra="ignore")

    median: float = Field(..., ge=0.0)
    p75: float = Field(..., ge=0.0)
    p90: float = Field(..., ge=0.0)
    max_cap: float = Field(..., ge=0.0)
    unit: str = "count"


class ArtifactTypeConfig(BaseModel):
    """
    Rules and relative weights for discipline-specific artifact types.
    """
    model_config = ConfigDict(extra="ignore")

    weight: float = Field(default=1.0, ge=0.0)
    quality_indicators: List[str] = Field(default_factory=list)
    max_score_contribution: float = Field(default=100.0, ge=0.0, le=100.0)
    baseline: Optional[BaselineMetrics] = None


class CredentialRequirement(BaseModel):
    """
    Required or preferred professional registrations / licenses in this discipline.
    """
    model_config = ConfigDict(extra="ignore")

    code: str
    name: str
    is_mandatory: bool = False
    verification_authority: Optional[str] = None
    acceptable_equivalents: List[str] = Field(default_factory=list)


class SkillTaxonomyConfig(BaseModel):
    """
    Taxonomy of core vs. secondary skills and hierarchical branch synonyms for semantic matching.
    """
    model_config = ConfigDict(extra="ignore")

    core_skills: List[str] = Field(default_factory=list)
    secondary_skills: List[str] = Field(default_factory=list)
    sub_branches: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Sub-branch map (e.g. 'ai': ['machine learning', 'deep learning'])"
    )


class DepartmentProfile(BaseModel):
    """
    Declarative configuration profile adapting the Universal Scoring Engine
    to any academic department or professional domain without code changes.
    Stored at /profiles/<slug>.json.
    """
    model_config = ConfigDict(extra="ignore")

    slug: str = Field(..., description="Unique URL/file-safe identifier (e.g. 'computer_science', 'law')")
    department_name: str = Field(..., description="Human-readable department name")
    version: str = "1.0.0"
    status: str = Field(default="active", description="'active' | 'draft' | 'archived'")
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    description: str = ""

    # Universal dimension weights (sum to 1.0)
    base_weights: Dict[UniversalDimension, float]

    # Role modifiers modifying base_weights per hiring preset
    role_modifiers: Dict[RolePreset, Dict[UniversalDimension, float]] = Field(default_factory=dict)

    # Mandatory pass/fail gates
    hard_gates: List[HardGateRule] = Field(default_factory=list)

    # Statistical baselines for percentile normalizations
    field_baselines: Dict[str, BaselineMetrics] = Field(default_factory=dict)

    # Department-defined output artifact configuration
    artifact_taxonomy: Dict[str, ArtifactTypeConfig] = Field(default_factory=dict)

    # Licensures & Credentials
    credentials_taxonomy: List[CredentialRequirement] = Field(default_factory=list)

    # Skill graphs
    skill_taxonomy: SkillTaxonomyConfig = Field(default_factory=SkillTaxonomyConfig)

    # Recognized elite institution tiers / accreditation bodies
    institution_tiers: Dict[str, List[str]] = Field(default_factory=dict)

    notes: Optional[str] = None

    @field_validator("base_weights")
    @classmethod
    def validate_and_normalize_base_weights(
        cls, weights: Dict[UniversalDimension, float]
    ) -> Dict[UniversalDimension, float]:
        """
        Validates that base_weights cover universal dimensions and normalizes them
        so their sum equals exactly 1.0.
        """
        if not weights:
            raise ValueError("base_weights must not be empty.")

        total = sum(weights.values())
        if total <= 0:
            raise ValueError("Sum of base_weights must be greater than zero.")

        # Normalize weights so they sum strictly to 1.0
        normalized = {dim: round(w / total, 6) for dim, w in weights.items()}
        # Correct floating point rounding discrepancy on first element
        diff = 1.0 - sum(normalized.values())
        if normalized:
            first_key = next(iter(normalized))
            normalized[first_key] = round(normalized[first_key] + diff, 6)

        return normalized

    def get_effective_weights(self, role_preset: RolePreset) -> Dict[UniversalDimension, float]:
        """
        Calculates normalized dimensional weights for a specific role preset:
        effective_weight_d = base_weight_d * role_modifier_d -> renormalized to sum to 1.0.
        """
        modifiers = self.role_modifiers.get(role_preset, {})
        raw_weights: Dict[UniversalDimension, float] = {}

        for dim, base_w in self.base_weights.items():
            mod = modifiers.get(dim, 1.0)
            raw_weights[dim] = max(base_w * mod, 0.0)

        total = sum(raw_weights.values())
        if total <= 0:
            return self.base_weights.copy()

        normalized = {dim: round(w / total, 6) for dim, w in raw_weights.items()}
        # Reconcile rounding to sum to 1.0
        diff = 1.0 - sum(normalized.values())
        if normalized:
            first_key = next(iter(normalized))
            normalized[first_key] = round(normalized[first_key] + diff, 6)

        return normalized


# ============================================================================
# 5. SCORING RESULTS & EVALUATION OUTPUT
# ============================================================================

class DimensionResult(BaseModel):
    """
    Score, evidence coverage, and granular explanation for one universal dimension.
    """
    model_config = ConfigDict(extra="ignore")

    dimension: UniversalDimension
    score: float = Field(..., ge=0.0, le=100.0, description="Raw dimensional score (0-100)")
    weight: float = Field(..., ge=0.0, le=1.0, description="Normalized weight applied")
    weighted_score: float = Field(..., ge=0.0, le=100.0, description="score * weight")
    has_sufficient_evidence: bool = Field(
        ...,
        description="Whether candidate had sufficient data extracted for this dimension"
    )
    evidence_coverage_ratio: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Degree of evidence completeness for this dimension"
    )
    evidence_summary: str = Field(
        ...,
        description="Human-readable transparent rationale explaining the dimensional score"
    )
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Granular component breakdown (e.g. gpa_norm, tier_pts, years_exp)"
    )
    flags: List[str] = Field(default_factory=list)


class GateCheckResult(BaseModel):
    """Result of an individual hard gate rule evaluation."""
    model_config = ConfigDict(extra="ignore")

    gate_id: str
    gate_name: str
    passed: bool
    status: GateStatus
    reason: str
    actual_value: Any = None
    expected_value: Any = None


class GateEvaluation(BaseModel):
    """Consolidated report across all profile-defined hard gate rules."""
    model_config = ConfigDict(extra="ignore")

    overall_status: GateStatus = GateStatus.ELIGIBLE
    passed: bool = True
    summary: str = "Candidate satisfies all mandatory hard gates."
    checks: List[GateCheckResult] = Field(default_factory=list)
    failure_reasons: List[str] = Field(default_factory=list)
    verification_reasons: List[str] = Field(default_factory=list)


class CandidateEvaluationResult(BaseModel):
    """
    Complete, explainable evaluation output for an individual candidate.
    Includes full dimensional scoring breakdown, gate eligibility, fairness audit flags,
    confidence metrics, and backward-compatibility methods for TALASH v1 consumers.
    """
    model_config = ConfigDict(extra="ignore")

    candidate_id: str
    candidate_name: Optional[str] = None
    anonymized_id: str
    department_slug: str
    department_name: str
    role_preset: RolePreset

    # Pre-scoring hard gate eligibility
    gate_evaluation: GateEvaluation
    gate_status: GateStatus

    # Dimensional scores (0-100)
    dimension_scores: Dict[UniversalDimension, DimensionResult]

    # Universal formula components
    base_composite_score: float = Field(
        ..., ge=0.0, le=100.0,
        description="Sum of (weight_d * score_d) across all dimensions"
    )
    evidence_coverage: float = Field(
        ..., ge=0.0, le=1.0,
        description="Fraction of dimensions with sufficient extracted evidence"
    )
    confidence_multiplier: float = Field(
        ..., ge=0.85, le=1.0,
        description="0.85 + (0.15 * evidence_coverage)"
    )
    final_score: float = Field(
        ..., ge=0.0, le=100.0,
        description="base_composite_score * confidence_multiplier"
    )
    tier: CandidateTier

    # Fairness & Explainability
    flags: List[FairnessAuditFlag] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)
    human_review_recommended: bool = False
    evaluated_at: str = Field(
        default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    engine_version: str = "2.0.0"

    def to_legacy_dict(self) -> Dict[str, Any]:
        """
        Produces a dictionary matching TALASH's legacy composite_evaluations.csv
        and candidate profile API schemas. Guarantees 0% regression across all
        existing dashboard components, CSV exporters, and comparison tools.
        """
        # Map universal dimensions to legacy 7-field equivalents
        edu = self.dimension_scores.get(UniversalDimension.ACADEMIC_FOUNDATION)
        res = self.dimension_scores.get(UniversalDimension.EVIDENCE_OF_OUTPUT)
        sup = self.dimension_scores.get(UniversalDimension.LEADERSHIP_TEACHING_SERVICE)
        exp = self.dimension_scores.get(UniversalDimension.PROFESSIONAL_EXPERIENCE)
        imp = self.dimension_scores.get(UniversalDimension.IMPACT_RECOGNITION)
        dom = self.dimension_scores.get(UniversalDimension.DOMAIN_COMPETENCE)
        tra = self.dimension_scores.get(UniversalDimension.TRAJECTORY_GROWTH)

        # Scale scores to legacy max point allocations (20, 25, 10, 10, 10, 10, 15)
        # to match existing visualization thresholds
        edu_score = round(((edu.score if edu else 50.0) / 100.0) * 20.0, 1)
        res_score = round(((res.score if res else 50.0) / 100.0) * 25.0, 1)
        sup_score = round(((sup.score if sup else 50.0) / 100.0) * 10.0, 1)
        inn_score = round(((imp.score if imp else 50.0) / 100.0) * 10.0, 1)
        bre_score = round(((dom.score if dom else 50.0) / 100.0) * 10.0, 1)
        col_score = round(((tra.score if tra else 50.0) / 100.0) * 10.0, 1)
        exp_score = round(((exp.score if exp else 50.0) / 100.0) * 15.0, 1)

        return {
            "candidate_id": self.candidate_id,
            "candidate_name": self.candidate_name or self.candidate_id,
            "overall_composite_score": round(self.final_score, 1),
            "candidate_tier": self.tier.value,
            "education_score": edu_score,
            "research_score": res_score,
            "supervision_score": sup_score,
            "innovation_score": inn_score,
            "breadth_score": bre_score,
            "collaboration_score": col_score,
            "experience_score": exp_score,
            # Supplementary universal metrics
            "gate_status": self.gate_status.value,
            "confidence_multiplier": round(self.confidence_multiplier, 3),
            "evidence_coverage": round(self.evidence_coverage, 2),
            "evaluated_at": self.evaluated_at,
        }


# ============================================================================
# 6. INTERNAL UTILITY HELPERS
# ============================================================================

def _safe_str(val: Any) -> Optional[str]:
    """Helper converting pandas/nan values to sanitized string or None."""
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() == "nan" or s.lower() == "none":
        return None
    return s


def _safe_float(val: Any) -> Optional[float]:
    """Helper converting pandas/nan values to float or None."""
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


def _safe_int(val: Any) -> Optional[int]:
    """Helper converting pandas/nan values to int or None."""
    f = _safe_float(val)
    if f is None:
        return None
    return int(f)
