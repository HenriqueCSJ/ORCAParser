"""
Base class for all ORCA output parsing modules.
Each module is responsible for extracting one logical section of the output.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseModule(ABC):
    """
    Abstract base class for all parser modules.

    To add a new property/section, subclass BaseModule, implement `parse()`,
    and register the subclass in parser.py's MODULE_REGISTRY.
    """

    # Override in subclasses to give a human-readable name
    name: str = "base"

    def __init__(self, context: Dict[str, Any]):
        """
        Parameters
        ----------
        context : dict
            Shared context dict populated by the parser, carrying flags such as
            'is_uhf', 'has_symmetry', 'scf_type', etc.
        """
        self.context = context

    @abstractmethod
    def parse(self, lines: list[str]) -> Optional[Dict[str, Any]]:
        """
        Parse the relevant section(s) from the full list of output lines.

        Parameters
        ----------
        lines : list[str]
            All lines of the ORCA output file (already stripped of newlines).

        Returns
        -------
        dict or None
            Parsed data, or None if this section is not present.
        """

    # ------------------------------------------------------------------ #
    # Utility helpers available to all subclasses                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def find_line(lines: list[str], pattern: str, start: int = 0) -> int:
        """Return index of first line containing *pattern* (case-insensitive) at or after *start*, else -1."""
        pat = pattern.lower()
        for i in range(start, len(lines)):
            if pat in lines[i].lower():
                return i
        return -1

    @staticmethod
    def find_all_lines(lines: list[str], pattern: str) -> list[int]:
        """Return list of all line indices containing *pattern* (case-insensitive)."""
        pat = pattern.lower()
        return [i for i, ln in enumerate(lines) if pat in ln.lower()]

    @staticmethod
    def find_last_line(lines: list[str], pattern: str) -> int:
        """Return index of last line containing *pattern* (case-insensitive), else -1."""
        pat = pattern.lower()
        for i in range(len(lines) - 1, -1, -1):
            if pat in lines[i].lower():
                return i
        return -1

    @staticmethod
    def find_last_line_exact(
        lines: list[str],
        pattern: str,
        start: int = 0,
    ) -> int:
        """Return index of last line whose stripped text equals *pattern* (case-insensitive), else -1."""
        pat = pattern.strip().lower()
        for i in range(len(lines) - 1, start - 1, -1):
            if lines[i].strip().lower() == pat:
                return i
        return -1

    @staticmethod
    def safe_float(value: str) -> Optional[float]:
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def safe_int(value: str) -> Optional[int]:
        try:
            return int(value)
        except (ValueError, TypeError):
            return None
