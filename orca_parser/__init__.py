"""
orca_parser – modular parser for ORCA quantum chemistry output files.

Quick start::

    from orca_parser import ORCAParser

    p = ORCAParser("my_calc.out")
    data = p.parse()
    p.to_json("my_calc.json")
    p.to_csv("csv_output/")

The package still exposes ``MODULE_REGISTRY`` for compatibility, but new
parser-section extensions should register through
``orca_parser.parser_section_registry`` instead of editing parser globals.
"""

from .parser import ORCAParser, MODULE_REGISTRY

__version__ = "1.0.0"
__all__ = ["ORCAParser", "MODULE_REGISTRY"]
