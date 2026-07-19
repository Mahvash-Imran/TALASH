"""
__init__.py  –  Analysis Package
=================================
Exports the main orchestrators for:
  Module 2: Educational Profile Analysis
  Module 3: Research Profile Analysis
  Module 4: Student Supervision Analysis
  Module 5: Books Authored / Co-Authored Analysis
  Module 6: Patents Analysis
"""

from .educational_profile  import EducationalProfileAnalyser
from .research_profile     import ResearchProfileAnalyser
from .supervision_analyser import SupervisionAnalyser
from .book_analyser        import BookProfileAnalyser
from .patent_analyser      import PatentProfileAnalyser

__all__ = [
    "EducationalProfileAnalyser",
    "ResearchProfileAnalyser",
    "SupervisionAnalyser",
    "BookProfileAnalyser",
    "PatentProfileAnalyser",
]
