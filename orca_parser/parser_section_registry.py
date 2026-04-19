"""Registry for parser-section plugins and aliases.

This module is the parser-side counterpart to the newer normalization and
family-registry seams. Historically, ``parser.py`` owned three separate
hard-coded concerns:

* which parser modules exist
* which sections are always included
* which aliases expand to which section groups

That made new parser sections expensive to add because every new section had
to edit central parser globals before it could participate in normal CLI/API
flows. The registry below moves those choices behind a small plugin-like
contract so the parser can discover sections instead of naming them by hand.
"""

from __future__ import annotations

from typing import Optional

from .modules import (
    CHELPGModule,
    DipoleMomentModule,
    EPRModule,
    HirshfeldModule,
    LoewdinModule,
    MBISModule,
    MayerModule,
    MullikenModule,
    NBOModule,
    OrbitalEnergiesModule,
    QROModule,
    SCFModule,
    SolvationModule,
)

from .parser_section_plugin import ParserSectionAlias, ParserSectionPlugin


_PARSER_SECTION_PLUGINS: list[ParserSectionPlugin] = []
_PARSER_SECTION_ALIASES: list[ParserSectionAlias] = []


def register_parser_section_plugin(
    plugin: ParserSectionPlugin,
    *,
    replace: bool = False,
) -> None:
    """Register a parser section plugin.

    ``replace=True`` exists for the same reason as the family registry's
    replacement mode: it lets tests or downstream extensions swap behavior
    without editing parser internals.
    """

    global _PARSER_SECTION_PLUGINS

    if replace:
        _PARSER_SECTION_PLUGINS = [
            existing
            for existing in _PARSER_SECTION_PLUGINS
            if existing.key != plugin.key
        ]
    elif any(existing.key == plugin.key for existing in _PARSER_SECTION_PLUGINS):
        raise ValueError(f"Parser section already registered: {plugin.key}")

    _PARSER_SECTION_PLUGINS.append(plugin)


def register_parser_section_alias(
    alias: ParserSectionAlias,
    *,
    replace: bool = False,
) -> None:
    """Register a parser-section alias."""

    global _PARSER_SECTION_ALIASES

    if replace:
        _PARSER_SECTION_ALIASES = [
            existing
            for existing in _PARSER_SECTION_ALIASES
            if existing.name != alias.name
        ]
    elif any(existing.name == alias.name for existing in _PARSER_SECTION_ALIASES):
        raise ValueError(f"Parser section alias already registered: {alias.name}")

    _PARSER_SECTION_ALIASES.append(alias)


def get_registered_parser_section_plugins() -> tuple[ParserSectionPlugin, ...]:
    """Return parser section plugins in parse order."""

    return tuple(_PARSER_SECTION_PLUGINS)


def get_registered_parser_section_aliases() -> tuple[ParserSectionAlias, ...]:
    """Return registered parser-section aliases."""

    return tuple(_PARSER_SECTION_ALIASES)


def get_core_parser_section_keys() -> set[str]:
    """Return section keys that are always included."""

    return {
        plugin.key
        for plugin in get_registered_parser_section_plugins()
        if plugin.always_include
    }


def get_parser_section_alias_map() -> dict[str, list[str]]:
    """Return the current alias expansion map.

    ``all`` is generated dynamically so it always reflects the current
    registry contents.
    """

    alias_map = {
        alias.name: list(alias.section_keys)
        for alias in get_registered_parser_section_aliases()
    }
    alias_map["all"] = [plugin.key for plugin in get_registered_parser_section_plugins()]
    return alias_map


def resolve_requested_parser_sections(sections) -> Optional[set[str]]:
    """Expand aliases and return the set of section keys to run.

    ``None`` means "run everything", matching the parser's long-standing API.
    Core sections are always added for explicit section requests.
    """

    if sections is None:
        return None
    if isinstance(sections, str):
        sections = [sections]

    alias_map = get_parser_section_alias_map()
    requested: set[str] = set()
    for token in sections:
        normalized = token.lower().strip()
        if normalized == "all":
            return None
        if normalized in alias_map:
            requested.update(alias_map[normalized])
        else:
            requested.add(normalized)

    requested.update(get_core_parser_section_keys())
    return requested


def iter_active_parser_section_plugins(
    active_sections: Optional[set[str]],
) -> tuple[ParserSectionPlugin, ...]:
    """Return parser section plugins active for this parse call."""

    plugins = get_registered_parser_section_plugins()
    if active_sections is None:
        return plugins
    return tuple(
        plugin
        for plugin in plugins
        if plugin.key in active_sections
    )


register_parser_section_plugin(
    ParserSectionPlugin("scf", SCFModule, always_include=True)
)
register_parser_section_plugin(
    ParserSectionPlugin("orbital_energies", OrbitalEnergiesModule)
)
register_parser_section_plugin(ParserSectionPlugin("qro", QROModule))
register_parser_section_plugin(ParserSectionPlugin("mulliken", MullikenModule))
register_parser_section_plugin(ParserSectionPlugin("loewdin", LoewdinModule))
register_parser_section_plugin(ParserSectionPlugin("mayer", MayerModule))
register_parser_section_plugin(ParserSectionPlugin("hirshfeld", HirshfeldModule))
register_parser_section_plugin(ParserSectionPlugin("mbis", MBISModule))
register_parser_section_plugin(ParserSectionPlugin("chelpg", CHELPGModule))
register_parser_section_plugin(ParserSectionPlugin("dipole", DipoleMomentModule))
register_parser_section_plugin(ParserSectionPlugin("solvation", SolvationModule))
register_parser_section_plugin(ParserSectionPlugin("nbo", NBOModule))
register_parser_section_plugin(ParserSectionPlugin("epr", EPRModule))

register_parser_section_alias(
    ParserSectionAlias(
        name="charges",
        section_keys=("mulliken", "loewdin", "hirshfeld", "mbis", "chelpg"),
    )
)
register_parser_section_alias(
    ParserSectionAlias(
        name="population",
        section_keys=("mulliken", "loewdin", "mayer"),
    )
)
register_parser_section_alias(
    ParserSectionAlias(
        name="mos",
        section_keys=("orbital_energies", "qro"),
    )
)
register_parser_section_alias(
    ParserSectionAlias(
        name="bonds",
        section_keys=("mayer", "loewdin"),
    )
)
register_parser_section_alias(ParserSectionAlias(name="nbo", section_keys=("nbo",)))
register_parser_section_alias(
    ParserSectionAlias(name="dipole", section_keys=("dipole",))
)
register_parser_section_alias(
    ParserSectionAlias(name="solvation", section_keys=("solvation",))
)
register_parser_section_alias(ParserSectionAlias(name="epr", section_keys=("epr",)))
