"""
orca_parser – modular parser for ORCA quantum chemistry output files.

Quick start::

    from orca_parser import ORCAParser

    p = ORCAParser("my_calc.out")
    data = p.parse()
    p.to_json("my_calc.json")
    p.to_csv("csv_output/")
"""

from .parser import ORCAParser, MODULE_REGISTRY

__version__ = "1.0.0"
__all__ = ["ORCAParser", "MODULE_REGISTRY"]
