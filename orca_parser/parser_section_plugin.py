"""Shared parser-section plugin contracts.

These small dataclasses used to live inside ``parser_section_registry.py``.
That worked while only the central registry created built-in sections, but it
created an awkward circular-import trap once built-in parser modules started
exporting their own ``PLUGIN_BUNDLE`` declarations.

Keeping the contracts in a neutral module lets:

* parser modules declare their own parser-section contributions
* the parser registry stay focused on registration / resolution behavior
* plugin bundles reuse the same types without importing registry internals
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from .modules.base import BaseModule


@dataclass(frozen=True)
class ParserSectionPlugin:
    """Plugin-like definition of one parser section.

    A section plugin answers one parser-core question:

        "Which module owns this logical output section?"

    The parser only needs the section key, the module class, and whether the
    section should always be included as part of the parser's core view.
    Registration order is parse order.
    """

    key: str
    module_class: type["BaseModule"]
    always_include: bool = False


@dataclass(frozen=True)
class ParserSectionAlias:
    """Named expansion for one or more parser sections."""

    name: str
    section_keys: Sequence[str]
