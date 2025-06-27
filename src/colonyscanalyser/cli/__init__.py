"""
Command Line Interface (CLI) package for ColonyScanalyser.

This package provides the command line interface functionality including
argument parsing and command handling for the ColonyScanalyser tool.
"""

from .args import create_parser
from .main import main

__all__ = [
    "create_parser",
    "main",
]
