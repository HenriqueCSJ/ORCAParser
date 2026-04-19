"""Registry for calculation-family plugins.

This module is the next architectural seam after ``job_snapshot`` /
``job_series`` / ``final_snapshot``. Those normalization layers answer
"what did ORCA run?" and "which final/stepwise payload is authoritative?".
The family registry answers the next question:

    "How should a normalized calculation family behave downstream?"

Historically the answer lived in several places:
* ``job_snapshot.py`` classified the family and picked labels
* ``markdown_writer.py`` decided which family-specific sections to render
* ``csv_writer.py`` decided which family-specific CSV files to write

That made new calculation families expensive to add because each family had to
teach multiple modules about its existence. The registry below moves the
family-specific hooks behind one plugin-like object. Adding a new family
should now mean:

1. register one ``CalculationFamilyPlugin``
2. provide its matcher, labels, and optional output hooks
3. let the existing parser / markdown / CSV orchestration discover it

The built-in families are intentionally migrating out of this file and into the
modules that already understand their domain semantics. The central registry is
shrinking toward "fallbacks and registration behavior" rather than becoming a
second home for chemistry knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

from .render_options import RenderOptions


MarkdownSection = tuple[str, str]
FormatNumber = Callable[[Any, str], str]
MakeTable = Callable[[List[tuple]], str]
WriteCSV = Callable[[Path, str, List[Dict[str, Any]], List[str]], Path]

FamilyMatcher = Callable[
    [Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]],
    bool,
]
FamilyStateLabelBuilder = Callable[
    [Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]],
    str,
]
FamilyMarkdownSectionBuilder = Callable[
    [Dict[str, Any], FormatNumber, MakeTable, RenderOptions],
    List[MarkdownSection],
]
FamilyComparisonSectionBuilder = Callable[
    [List[Dict[str, Any]], List[str], FormatNumber, MakeTable, RenderOptions],
    List[MarkdownSection],
]
FamilyCSVWriter = Callable[
    [Dict[str, Any], Path, str, WriteCSV],
    List[Path],
]


@dataclass(frozen=True)
class CalculationFamilyPlugin:
    """Plugin-like description of one normalized calculation family.

    The core idea is that the family owns both identity and behavior:
    * ``matcher`` decides whether a parsed job belongs to this family
    * labels define how the family should be described in normalized metadata
    * markdown / comparison / CSV hooks teach downstream exporters how to
      surface the family without adding more branching to top-level writers

    Most families will only need a subset of the hooks. ``single_point`` is a
    good example: it needs a matcher and a default label, but no extra
    markdown/CSV sections.
    """

    family: str
    default_calculation_label: str
    matcher: FamilyMatcher
    electronic_state_kind: str = "ground_state"
    build_special_electronic_state_label: FamilyStateLabelBuilder | None = None
    build_state_target_label: FamilyStateLabelBuilder | None = None
    render_markdown_sections: FamilyMarkdownSectionBuilder | None = None
    render_comparison_sections: FamilyComparisonSectionBuilder | None = None
    csv_writers: Sequence[FamilyCSVWriter] = field(default_factory=tuple)
    comparison_order: int = 50

    def special_electronic_state_label(
        self,
        meta: Dict[str, Any],
        data: Dict[str, Any],
        deltascf: Dict[str, Any],
        excited_state_optimization: Dict[str, Any],
    ) -> str:
        """Return the family-specific electronic-state label, if any."""
        if self.build_special_electronic_state_label is None:
            return ""
        return self.build_special_electronic_state_label(
            meta,
            data,
            deltascf,
            excited_state_optimization,
        )

    def state_target_label(
        self,
        meta: Dict[str, Any],
        data: Dict[str, Any],
        deltascf: Dict[str, Any],
        excited_state_optimization: Dict[str, Any],
    ) -> str:
        """Return the family-specific state target label, if any."""
        if self.build_state_target_label is None:
            return ""
        return self.build_state_target_label(
            meta,
            data,
            deltascf,
            excited_state_optimization,
        )


_CALCULATION_FAMILY_PLUGINS: List[CalculationFamilyPlugin] = []


def register_calculation_family_plugin(
    plugin: CalculationFamilyPlugin,
    *,
    replace: bool = False,
) -> None:
    """Register a calculation-family plugin.

    ``replace=True`` is intentionally supported because it makes the extension
    seam testable. A downstream developer can temporarily swap the built-in
    behavior for one family without patching the writers themselves.
    """

    global _CALCULATION_FAMILY_PLUGINS

    if replace:
        _CALCULATION_FAMILY_PLUGINS = [
            existing
            for existing in _CALCULATION_FAMILY_PLUGINS
            if existing.family != plugin.family
        ]
    elif any(existing.family == plugin.family for existing in _CALCULATION_FAMILY_PLUGINS):
        raise ValueError(f"Calculation family already registered: {plugin.family}")

    _CALCULATION_FAMILY_PLUGINS.append(plugin)


def get_registered_calculation_family_plugins() -> tuple[CalculationFamilyPlugin, ...]:
    """Return registered family plugins in match / render order."""
    return tuple(_CALCULATION_FAMILY_PLUGINS)


def _fallback_calculation_label(calculation_type: str) -> str:
    """Title-case fallback when no family plugin supplies a label."""
    if calculation_type:
        return calculation_type.replace("_", " ").replace("-", " ").title()
    return "Unknown"


def match_calculation_family_plugin(
    meta: Dict[str, Any],
    data: Dict[str, Any],
    context: Dict[str, Any],
    deltascf: Dict[str, Any],
    excited_state_optimization: Dict[str, Any],
) -> CalculationFamilyPlugin:
    """Return the first registered plugin whose matcher claims the job."""
    for plugin in get_registered_calculation_family_plugins():
        if plugin.matcher(meta, data, context, deltascf, excited_state_optimization):
            return plugin

    calculation_type = str(meta.get("calculation_type", "")).strip()
    return CalculationFamilyPlugin(
        family=(
            calculation_type.lower().replace(" ", "_").replace("-", "_")
            if calculation_type
            else "unknown"
        ),
        default_calculation_label=_fallback_calculation_label(calculation_type),
        matcher=lambda *_args: False,
    )


def get_calculation_family_plugin(data: Dict[str, Any]) -> CalculationFamilyPlugin:
    """Return the plugin for a parsed dataset.

    Output code should normally resolve the family from ``job_snapshot``.
    Falling back to the matcher keeps the seam usable during tests or for
    partially normalized data structures.
    """
    snapshot = data.get("job_snapshot")
    if isinstance(snapshot, dict):
        family = str(snapshot.get("calculation_family", "")).strip()
        if family:
            for plugin in get_registered_calculation_family_plugins():
                if plugin.family == family:
                    return plugin

    meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    context = data.get("context") if isinstance(data.get("context"), dict) else {}
    deltascf = meta.get("deltascf") if isinstance(meta.get("deltascf"), dict) else {}
    excopt = meta.get("excited_state_optimization")
    if not isinstance(excopt, dict):
        tddft = data.get("tddft")
        if isinstance(tddft, dict):
            excopt = tddft.get("excited_state_optimization")
    if not isinstance(excopt, dict):
        excopt = {}
    return match_calculation_family_plugin(meta, data, context, deltascf, excopt)


def iter_active_comparison_family_plugins(
    datasets: Iterable[Dict[str, Any]],
) -> List[CalculationFamilyPlugin]:
    """Return active family plugins for a comparison document.

    The list is deduplicated and sorted using each plugin's explicit
    ``comparison_order``. That keeps comparison output stable while still
    making the registry the single source of family-specific comparison hooks.
    """
    active_families = {
        get_calculation_family_plugin(dataset).family
        for dataset in datasets
    }
    plugins = [
        plugin
        for plugin in get_registered_calculation_family_plugins()
        if plugin.family in active_families and plugin.render_comparison_sections is not None
    ]
    return sorted(plugins, key=lambda plugin: (plugin.comparison_order, plugin.family))
