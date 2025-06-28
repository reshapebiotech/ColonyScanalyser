"""
Main entry point for ColonyScanalyser.

This module provides the main function that serves as the entry point for the
ColonyScanalyser command line tool. It delegates to the CLI module for actual
command processing.
"""

from colonyscanalyser.cli.main import main

if __name__ == "__main__":
    main()
