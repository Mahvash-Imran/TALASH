"""
pipeline_orchestrator.py  –  Master Integration Pipeline for Parts 1–10
========================================================================

WHY THIS FILE EXISTS
--------------------
Provides the single master orchestrator class (`MasterPipeline`) that integrates Parts 1 through 10
into a unified pipeline execution per candidate or across the full batch.
"""

import io
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow importing from root directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis.educational_profile import EducationalProfileAnalyser
from analysis.research_profile import ResearchProfileAnalyser
from analysis.supervision_analyser import SupervisionAnalyser
from analysis.book_analyser import BookProfileAnalyser
from analysis.patent_analyser import PatentProfileAnalyser
from analysis.topic_analyser import TopicBreadthAnalyser
from analysis.collaboration_analyser import CollaborationAnalyser
from analysis.experience_analyser import ExperienceProfileAnalyser
from analysis.composite_evaluator import CompositeEvaluator, compute_candidate_composite_score
from analysis.email_drafter import extract_missing_candidate_info, draft_followup_email

logger = logging.getLogger(__name__)


class MasterPipeline:
    """
    Integrates Modules 1 through 10 into a single master evaluation pipeline.
    """

    def __init__(
        self,
        extracted_dir: str = "data/extracted",
        analysis_dir:  str = "data/analysis",
        api_key:       Optional[str] = None,
        model:         str = "meta-llama/llama-4-scout-17b-16e-instruct",
        base_url:      Optional[str] = None,
        skip_llm:      bool = True,
    ):
        self.extracted_dir = Path(extracted_dir)
        self.analysis_dir  = Path(analysis_dir)
        self.api_key       = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model         = model
        self.base_url      = base_url or os.environ.get("OPENAI_BASE_URL")
        is_valid_key       = bool(self.api_key and not str(self.api_key).startswith("your_") and len(str(self.api_key).strip()) > 20)
        self.skip_llm      = skip_llm or not is_valid_key

        self.analysis_dir.mkdir(parents=True, exist_ok=True)

    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Runs Modules 2 through 10 sequentially across all candidates.
        """
        logger.info("============================================================")
        logger.info("  TALASH Master Pipeline – Executing Modules 2 to 10")
        logger.info("============================================================")

        # 1. Module 2: Educational Profile
        m2 = EducationalProfileAnalyser(output_dir=str(self.analysis_dir))
        m2.run()

        # 2. Module 3: Research Profile
        m3 = ResearchProfileAnalyser(
            output_dir=str(self.analysis_dir),
            api_key=self.api_key, model=self.model, base_url=self.base_url, skip_llm=self.skip_llm
        )
        m3.run()

        # 3. Module 4: Supervision Analysis
        m4 = SupervisionAnalyser(
            output_dir=str(self.analysis_dir),
            api_key=self.api_key, model=self.model, base_url=self.base_url, skip_llm=self.skip_llm
        )
        m4.run()

        # 4. Module 5: Books Analysis
        m5 = BookProfileAnalyser(
            output_dir=str(self.analysis_dir),
            api_key=self.api_key, model=self.model, base_url=self.base_url, skip_llm=self.skip_llm
        )
        m5.run()

        # 5. Module 6: Patents Analysis
        m6 = PatentProfileAnalyser(
            output_dir=str(self.analysis_dir),
            api_key=self.api_key, model=self.model, base_url=self.base_url, skip_llm=self.skip_llm
        )
        m6.run()

        # 6. Module 7: Topic Breadth Analysis
        m7 = TopicBreadthAnalyser(
            output_dir=str(self.analysis_dir),
            api_key=self.api_key, model=self.model, base_url=self.base_url, skip_llm=self.skip_llm
        )
        m7.run()

        # 7. Module 8: Collaboration Analysis
        m8 = CollaborationAnalyser(
            output_dir=str(self.analysis_dir),
            api_key=self.api_key, model=self.model, base_url=self.base_url, skip_llm=self.skip_llm
        )
        m8.run()

        # 8. Module 9: Experience & Skill Alignment
        m9 = ExperienceProfileAnalyser(
            output_dir=str(self.analysis_dir),
            api_key=self.api_key, model=self.model, base_url=self.base_url, skip_llm=self.skip_llm
        )
        m9.run()

        # 9. Module 10: Composite Candidate Evaluation
        m10 = CompositeEvaluator(output_dir=str(self.analysis_dir))
        comp_paths = m10.run()

        logger.info("Master Pipeline Execution Complete.")
        return comp_paths


if __name__ == "__main__":
    pipeline = MasterPipeline(skip_llm=True)
    res = pipeline.run_full_pipeline()
    print("Master pipeline complete:", res)
