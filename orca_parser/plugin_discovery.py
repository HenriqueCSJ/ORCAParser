"""Autodiscovery and registration for explicit plugin bundles.

This layer sits *on top of* the existing registries.

The registries remain the stable machine. Discovery simply finds modules that
export ``PLUGIN_BUNDLE`` or ``PLUGIN_BUNDLES`` and feeds those declarations
into the registries. That keeps the "drop in a module file" developer
experience while preserving explicit validation and debuggable state.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Iterable, Sequence

from .job_family_registry import register_calculation_family_plugin
from .output.csv_section_registry import register_csv_section_plugin
from .output.markdown_section_registry import register_markdown_section_plugin
from .parser_section_registry import (
    register_parser_section_alias,
    register_parser_section_plugin,
)
from .plugin_bundle import PluginBundle, PluginMetadata, PluginOption


_DEFAULT_PLUGIN_PACKAGES = ("orca_parser.modules",)
_BOOTSTRAPPED_PACKAGES: set[str] = set()
_REGISTERED_PLUGIN_BUNDLES: dict[str, PluginBundle] = {}
_REGISTERED_PLUGIN_SOURCES: dict[str, str] = {}
_REGISTERED_PLUGIN_OPTIONS: dict[str, PluginOption] = {}


def _iter_plugin_module_names(package_name: str) -> tuple[str, ...]:
    """Return importable submodule names for a plugin package."""

    package = importlib.import_module(package_name)
    package_path = getattr(package, "__path__", None)
    if not package_path:
        return ()

    module_names: list[str] = []
    for module_info in pkgutil.iter_modules(package_path):
        if module_info.name.startswith("_"):
            continue
        module_names.append(f"{package_name}.{module_info.name}")
    return tuple(sorted(module_names))


def _coerce_bundle_sequence(exported: object, module_name: str) -> tuple[PluginBundle, ...]:
    """Normalize plugin bundle exports to a tuple of ``PluginBundle`` objects."""

    if exported is None:
        return ()
    if isinstance(exported, PluginBundle):
        return (exported,)
    if isinstance(exported, (list, tuple)):
        bundles = tuple(exported)
        if not all(isinstance(bundle, PluginBundle) for bundle in bundles):
            raise TypeError(
                f"{module_name} exports PLUGIN_BUNDLES with non-PluginBundle values."
            )
        return bundles
    raise TypeError(
        f"{module_name} exports an unsupported plugin bundle object: {type(exported)!r}"
    )


def _iter_exported_bundles(module_name: str) -> tuple[PluginBundle, ...]:
    """Import a module and return any explicit plugin bundles it exports."""

    module = importlib.import_module(module_name)
    bundles: list[PluginBundle] = []
    if hasattr(module, "PLUGIN_BUNDLE"):
        bundles.extend(_coerce_bundle_sequence(getattr(module, "PLUGIN_BUNDLE"), module_name))
    if hasattr(module, "PLUGIN_BUNDLES"):
        bundles.extend(_coerce_bundle_sequence(getattr(module, "PLUGIN_BUNDLES"), module_name))
    return tuple(bundles)


def _validate_plugin_metadata(metadata: PluginMetadata, module_name: str) -> None:
    """Reject malformed or conflicting plugin metadata early."""

    if not metadata.key.strip():
        raise ValueError(f"{module_name} exports a plugin bundle with an empty metadata.key.")
    if not metadata.name.strip():
        raise ValueError(f"{module_name} exports a plugin bundle with an empty metadata.name.")


def _validate_plugin_option(option: PluginOption, module_name: str) -> None:
    """Validate declarative plugin CLI options before argparse sees them."""

    if not option.dest.strip():
        raise ValueError(f"{module_name} exports a plugin option with an empty dest.")
    if not option.flags:
        raise ValueError(
            f"{module_name} exports plugin option '{option.dest}' without CLI flags."
        )
    if any(not str(flag).startswith("-") for flag in option.flags):
        raise ValueError(
            f"{module_name} exports plugin option '{option.dest}' with invalid flags."
        )


def register_plugin_bundle(bundle: PluginBundle, *, source_module: str) -> None:
    """Register one bundle into the existing parser/output registries."""

    _validate_plugin_metadata(bundle.metadata, source_module)

    key = bundle.metadata.key
    existing_source = _REGISTERED_PLUGIN_SOURCES.get(key)
    if existing_source is not None:
        if existing_source == source_module:
            return
        raise ValueError(
            f"Plugin key '{key}' is already registered by {existing_source}; "
            f"{source_module} cannot reuse it."
        )

    for option in bundle.options:
        _validate_plugin_option(option, source_module)
        if option.dest in _REGISTERED_PLUGIN_OPTIONS:
            raise ValueError(
                f"Plugin option dest '{option.dest}' is already registered."
            )
        for existing in _REGISTERED_PLUGIN_OPTIONS.values():
            if set(existing.flags) & set(option.flags):
                overlap = sorted(set(existing.flags) & set(option.flags))
                raise ValueError(
                    f"Plugin option flags {overlap} from {source_module} overlap with "
                    f"already registered plugin option '{existing.dest}'."
                )

    for plugin in bundle.parser_sections:
        register_parser_section_plugin(plugin)
    for alias in bundle.parser_aliases:
        register_parser_section_alias(alias)
    for family in bundle.calculation_families:
        register_calculation_family_plugin(family)
    for section in bundle.markdown_sections:
        register_markdown_section_plugin(section)
    for section in bundle.csv_sections:
        register_csv_section_plugin(section)

    _REGISTERED_PLUGIN_BUNDLES[key] = bundle
    _REGISTERED_PLUGIN_SOURCES[key] = source_module
    for option in bundle.options:
        _REGISTERED_PLUGIN_OPTIONS[option.dest] = option


def bootstrap_plugin_bundles(
    package_names: Sequence[str] = _DEFAULT_PLUGIN_PACKAGES,
) -> tuple[PluginBundle, ...]:
    """Discover and register plugin bundles from one or more packages.

    The bootstrap is intentionally idempotent per package so the parser, CLI,
    and library entry points can all call it safely.
    """

    discovered: list[PluginBundle] = []
    for package_name in package_names:
        if package_name in _BOOTSTRAPPED_PACKAGES:
            continue
        for module_name in _iter_plugin_module_names(package_name):
            for bundle in _iter_exported_bundles(module_name):
                register_plugin_bundle(bundle, source_module=module_name)
                discovered.append(bundle)
        _BOOTSTRAPPED_PACKAGES.add(package_name)
    return tuple(discovered)


def get_registered_plugin_bundles() -> tuple[PluginBundle, ...]:
    """Return plugin bundles that were discovered and registered."""

    return tuple(_REGISTERED_PLUGIN_BUNDLES.values())


def get_registered_plugin_options() -> tuple[PluginOption, ...]:
    """Return all discovered plugin CLI options."""

    return tuple(_REGISTERED_PLUGIN_OPTIONS.values())


def get_plugin_option_values(namespace: object) -> dict[str, object]:
    """Extract discovered plugin option values from an argparse namespace."""

    values: dict[str, object] = {}
    for option in get_registered_plugin_options():
        if hasattr(namespace, option.dest):
            values[option.dest] = getattr(namespace, option.dest)
    return values


def build_plugin_help_section() -> str:
    """Return a compact documentation appendix for discovered plugins."""

    bundles = get_registered_plugin_bundles()
    if not bundles:
        return ""

    lines = ["", "Discovered Plugin Bundles", "-------------------------"]
    for bundle in sorted(bundles, key=lambda item: item.metadata.key):
        meta = bundle.metadata
        lines.append(f"  {meta.key:<12} {meta.name}")
        if meta.short_help:
            lines.append(f"      {meta.short_help}")
        if meta.docs_path:
            lines.append(f"      docs: {meta.docs_path}")
        if meta.examples:
            for example in meta.examples:
                lines.append(f"      e.g. {example}")
        if bundle.options:
            option_flags = ", ".join(
                "/".join(option.flags)
                for option in bundle.options
            )
            lines.append(f"      options: {option_flags}")
    return "\n".join(lines)
