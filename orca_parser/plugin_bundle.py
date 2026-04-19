"""Explicit plugin-bundle contracts for autodiscovered ORCA parser modules.

The goal of this module is to keep autodiscovery *inspectable*.

Instead of relying on import-time side effects, a module can export one
``PLUGIN_BUNDLE`` object that declares what it contributes to the software:

* parser sections / aliases
* calculation-family behavior
* markdown / CSV section hooks
* CLI-visible options
* short documentation metadata

The discovery layer can then load those declarations and feed them into the
existing registries without guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from .job_family_registry import CalculationFamilyPlugin
from .output.csv_section_registry import CSVSectionPlugin
from .output.markdown_section_registry import MarkdownSectionPlugin
from .parser_section_plugin import ParserSectionAlias, ParserSectionPlugin


@dataclass(frozen=True)
class PluginMetadata:
    """Human-facing metadata for a plugin bundle."""

    key: str
    name: str
    version: str = "1.0"
    short_help: str = ""
    description: str = ""
    docs_path: str = ""
    examples: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class PluginOption:
    """Declarative CLI option owned by a plugin.

    The field set intentionally mirrors the small subset of ``argparse`` we
    already use in this project.  Keeping the option declarative means the CLI
    can discover plugin parameters and expose them in ``--help`` without the
    plugin touching the central parser.
    """

    dest: str
    flags: Sequence[str]
    help: str
    default: Any = None
    action: Optional[str] = None
    choices: Sequence[Any] = field(default_factory=tuple)
    metavar: Optional[str] = None
    nargs: Any = None
    type: Optional[Callable[[str], Any]] = None
    scope: str = "general"

    def argparse_kwargs(self) -> dict[str, Any]:
        """Return ``argparse`` keyword arguments for this option."""

        kwargs: dict[str, Any] = {
            "dest": self.dest,
            "help": self.help,
            "default": self.default,
        }
        if self.action:
            kwargs["action"] = self.action
        if self.choices:
            kwargs["choices"] = list(self.choices)
        if self.metavar is not None:
            kwargs["metavar"] = self.metavar
        if self.nargs is not None:
            kwargs["nargs"] = self.nargs
        if self.type is not None:
            kwargs["type"] = self.type
        return kwargs


@dataclass(frozen=True)
class PluginBundle:
    """One explicit declaration of everything a module contributes."""

    metadata: PluginMetadata
    parser_sections: Sequence[ParserSectionPlugin] = field(default_factory=tuple)
    parser_aliases: Sequence[ParserSectionAlias] = field(default_factory=tuple)
    calculation_families: Sequence[CalculationFamilyPlugin] = field(default_factory=tuple)
    markdown_sections: Sequence[MarkdownSectionPlugin] = field(default_factory=tuple)
    csv_sections: Sequence[CSVSectionPlugin] = field(default_factory=tuple)
    options: Sequence[PluginOption] = field(default_factory=tuple)
