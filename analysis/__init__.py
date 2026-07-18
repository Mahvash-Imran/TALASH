"""
__init__.py  –  Analysis Package
=================================
Exports the main orchestrators for:
  Module 2: Educational Profile Analysis
  Module 3: Research Profile Analysis
  Module 4: Student Supervision Analysis
"""

from .educational_profile  import EducationalProfileAnalyser
from .research_profile     import ResearchProfileAnalyser
from .supervision_analyser import SupervisionAnalyser

__all__ = [
    "EducationalProfileAnalyser",
    "ResearchProfileAnalyser",
    "SupervisionAnalyser",
]
