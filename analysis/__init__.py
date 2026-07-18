"""
__init__.py  –  Analysis Package
=================================
Exports the main orchestrators for:
  Module 2: Educational Profile Analysis
  Module 3: Research Profile Analysis
"""

from .educational_profile import EducationalProfileAnalyser
from .research_profile     import ResearchProfileAnalyser

__all__ = [
    "EducationalProfileAnalyser",
    "ResearchProfileAnalyser",
]
