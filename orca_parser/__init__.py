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
``orca_parser.parser_section_registry`` or, preferably, export an explicit
``PLUGIN_BUNDLE`` from a module discovered under ``orca_parser.modules``.
"""

from .plugin_bundle import PluginBundle, PluginMetadata, PluginOption
from .plugin_discovery import bootstrap_plugin_bundles

bootstrap_plugin_bundles()

from .parser import ORCAParser, MODULE_REGISTRY

__version__ = "1.0.0"
__all__ = [
    "ORCAParser",
    "MODULE_REGISTRY",
    "PluginBundle",
    "PluginMetadata",
    "PluginOption",
]
