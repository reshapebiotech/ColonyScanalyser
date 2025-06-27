"""
Pipeline processing package for ColonyScanalyser.

This package contains the main pipeline orchestration classes
for running the complete colony analysis workflow.
"""

from .processor import ColonyPipelineProcessor

__all__ = [
    "ColonyPipelineProcessor",
]
