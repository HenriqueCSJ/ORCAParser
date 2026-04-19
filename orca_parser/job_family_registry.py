"""Registry for calculation-family plugins.

This module is the next architectural seam after ``job_snapshot`` /
``job_series`` / ``final_snapshot``.  Those normalization layers answer
"what did ORCA run?" and "which final/stepwise payload is authoritative?".
The family registry answers the next question:

    "How should a normalized calculation family behave downstream?"

Historically the answer lived in several places:
* ``job_snapshot.py`` classified the family and picked labels
* ``markdown_writer.py`` decided which family-specific sections to render
* ``csv_writer.py`` decided which family-specific CSV files to write

That made new calculation families expensive to add because each family had to
teach multiple modules about its existence.  The registry below moves the
family-specific hooks behind one plugin-like object.  Adding a new family
should now mean:

1. register one ``CalculationFamilyPlugin``
2. provide its matcher, labels, and optional output hooks
3. let the existing parser / markdown / CSV orchestration discover it
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

    Most families will only need a subset of the hooks.  ``single_point`` is a
    good example: it needs a matcher and a default label, but no extra
    markdown/CSV sections.
    """

    family: str
    default_calculation_label: str
    matcher: FamilyMatcher
    electronic_state_kind: str = "ground_state"
    build_special_electronic_state_label: Optional[FamilyStateLabelBuilder] = None
    build_state_target_label: Optional[FamilyStateLabelBuilder] = None
    render_markdown_sections: Optional[FamilyMarkdownSectionBuilder] = None
    render_comparison_sections: Optional[FamilyComparisonSectionBuilder] = None
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
    seam testable.  A downstream developer can temporarily swap the built-in
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
    ``comparison_order``.  That keeps comparison output stable while still
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


def _matches_deltascf(
    meta: Dict[str, Any],
    data: Dict[str, Any],
    context: Dict[str, Any],
    deltascf: Dict[str, Any],
    excited_state_optimization: Dict[str, Any],
) -> bool:
    del data, context, excited_state_optimization
    calc_type = str(meta.get("calculation_type", "")).strip().lower()
    return bool(deltascf or calc_type == "deltascf")


def _matches_excited_state_optimization(
    meta: Dict[str, Any],
    data: Dict[str, Any],
    context: Dict[str, Any],
    deltascf: Dict[str, Any],
    excited_state_optimization: Dict[str, Any],
) -> bool:
    del data, context, deltascf
    return bool(excited_state_optimization)


def _matches_single_point(
    meta: Dict[str, Any],
    data: Dict[str, Any],
    context: Dict[str, Any],
    deltascf: Dict[str, Any],
    excited_state_optimization: Dict[str, Any],
) -> bool:
    del data, context, deltascf, excited_state_optimization
    calc_type = str(meta.get("calculation_type", "")).strip().lower()
    return "single point" in calc_type or calc_type == ""


def _excited_state_target_label_from_payload(excopt: Dict[str, Any]) -> str:
    """Short target label such as S1/T2/root 3 for excited-state families."""
    label = str(excopt.get("target_state_label") or "").strip()
    if label:
        return label
    root = excopt.get("target_root")
    if root is None:
        root = excopt.get("final_root")
    if root is None:
        return ""
    return f"root {root}"


def _build_deltascf_state_label(
    meta: Dict[str, Any],
    data: Dict[str, Any],
    deltascf: Dict[str, Any],
    excited_state_optimization: Dict[str, Any],
) -> str:
    del meta, data, deltascf, excited_state_optimization
    return "DeltaSCF excited-state"


def _build_excited_state_optimization_label(
    meta: Dict[str, Any],
    data: Dict[str, Any],
    deltascf: Dict[str, Any],
    excited_state_optimization: Dict[str, Any],
) -> str:
    del meta, data, deltascf
    target = _excited_state_target_label_from_payload(excited_state_optimization)
    if target:
        return f"Excited-state optimization ({target})"
    return "Excited-state optimization"


def _build_excited_state_target_label(
    meta: Dict[str, Any],
    data: Dict[str, Any],
    deltascf: Dict[str, Any],
    excited_state_optimization: Dict[str, Any],
) -> str:
    del meta, data, deltascf
    return _excited_state_target_label_from_payload(excited_state_optimization)


def _render_deltascf_markdown_sections(
    data: Dict[str, Any],
    format_number: FormatNumber,
    make_table: MakeTable,
    render_options: RenderOptions,
) -> List[MarkdownSection]:
    """Render DeltaSCF-specific markdown sections."""
    del render_options

    from .output import job_state as _job_state
    from .output.markdown_sections_state import render_deltascf_section

    formatter = lambda value: format_number(value)
    body = render_deltascf_section(
        data,
        get_deltascf_data=_job_state.get_deltascf_data,
        deltascf_target_summary=lambda deltascf: _job_state.deltascf_target_summary(
            deltascf,
            formatter=formatter,
        ),
        format_deltascf_vector=lambda values: _job_state.format_deltascf_vector(
            values,
            formatter=formatter,
        ),
        yes_no_unknown=_job_state.yes_no_unknown,
        make_table=make_table,
    )
    return [("DeltaSCF / Excited-State Target", body)] if body else []


def _render_excited_state_optimization_markdown_sections(
    data: Dict[str, Any],
    format_number: FormatNumber,
    make_table: MakeTable,
    render_options: RenderOptions,
) -> List[MarkdownSection]:
    """Render excited-state optimization sections owned by that family."""

    from .job_series import get_geom_opt_series
    from .output import job_state as _job_state
    from .output.markdown_sections_basic import render_geom_opt_section
    from .output.markdown_sections_state import render_excited_state_opt_section

    sections: List[MarkdownSection] = []

    excited_body = render_excited_state_opt_section(
        data,
        get_excited_state_opt_data=_job_state.get_excited_state_opt_data,
        excited_state_target_label=_job_state.excited_state_target_label,
        yes_no_unknown=_job_state.yes_no_unknown,
        format_number=format_number,
        make_table=make_table,
    )
    if excited_body:
        sections.append(("Excited-State Geometry Optimization", excited_body))

    geom_opt = get_geom_opt_series(data)
    geom_opt_body = render_geom_opt_section(
        geom_opt,
        format_number=format_number,
        make_table=make_table,
        cycle_preview_count=render_options.geom_opt_cycle_preview_count,
    ) if geom_opt else ""
    if geom_opt_body:
        sections.append(("Geometry Optimization", geom_opt_body))

    return sections


def _render_deltascf_comparison_sections(
    datasets: List[Dict[str, Any]],
    labels: List[str],
    format_number: FormatNumber,
    make_table: MakeTable,
    render_options: RenderOptions,
) -> List[MarkdownSection]:
    """Render the comparison table for DeltaSCF families."""
    del format_number, render_options

    from .output import job_state as _job_state

    if not any(get_calculation_family_plugin(dataset).family == "deltascf" for dataset in datasets):
        return []

    rows = [("", "electronic state", "target", "metric", "keep ref")]
    for label, dataset in zip(labels, datasets):
        if get_calculation_family_plugin(dataset).family != "deltascf":
            rows.append((label, "—", "—", "—", "—"))
            continue
        deltascf = _job_state.get_deltascf_data(dataset)
        rows.append((
            label,
            _job_state.electronic_state_label(dataset, ground_state_label="ground-state") or "ground-state",
            _job_state.deltascf_target_summary(deltascf) or "—",
            deltascf.get("aufbau_metric") or "—",
            _job_state.yes_no_unknown(deltascf.get("keep_initial_reference")),
        ))
    return [("DeltaSCF", make_table(rows))]


def _render_excited_state_optimization_comparison_sections(
    datasets: List[Dict[str, Any]],
    labels: List[str],
    format_number: FormatNumber,
    make_table: MakeTable,
    render_options: RenderOptions,
) -> List[MarkdownSection]:
    """Render the comparison table for excited-state optimization families."""
    del format_number, render_options

    from .output import job_state as _job_state

    if not any(
        get_calculation_family_plugin(dataset).family == "excited_state_optimization"
        for dataset in datasets
    ):
        return []

    rows = [("", "electronic state", "target", "input", "followiroot", "SOC grad", "final root")]
    for label, dataset in zip(labels, datasets):
        if get_calculation_family_plugin(dataset).family != "excited_state_optimization":
            rows.append((label, "—", "—", "—", "—", "—", "—"))
            continue
        excited_state_optimization = _job_state.get_excited_state_opt_data(dataset)
        rows.append((
            label,
            _job_state.electronic_state_label(dataset, ground_state_label="ground-state") or "ground-state",
            _job_state.excited_state_target_label(excited_state_optimization) or "—",
            (
                f"%{excited_state_optimization.get('input_block')}"
                if excited_state_optimization.get("input_block")
                else "—"
            ),
            (
                _job_state.yes_no_unknown(excited_state_optimization.get("followiroot"))
                if "followiroot" in excited_state_optimization
                else "—"
            ),
            (
                _job_state.yes_no_unknown(excited_state_optimization.get("socgrad"))
                if "socgrad" in excited_state_optimization
                else "—"
            ),
            (
                str(excited_state_optimization.get("final_root"))
                if excited_state_optimization.get("final_root") is not None
                else "—"
            ),
        ))
    return [("Excited-State Optimization", make_table(rows))]


def _write_deltascf_csv_sections(
    data: Dict[str, Any],
    directory: Path,
    stem: str,
    write_csv: WriteCSV,
) -> List[Path]:
    """Write DeltaSCF CSV sections through the registered family hook."""
    from .output import job_state as _job_state
    from .output.csv_sections_state import write_deltascf_section

    return write_deltascf_section(
        data,
        directory,
        stem,
        write_csv=write_csv,
        electronic_state_label=lambda dataset: _job_state.electronic_state_label(
            dataset,
            ground_state_label="Ground-state",
        ),
        get_deltascf_data=_job_state.get_deltascf_data,
        is_deltascf=_job_state.is_deltascf,
        format_deltascf_target=_job_state.format_deltascf_target,
        format_simple_vector=_job_state.format_simple_vector,
        bool_to_label=_job_state.bool_to_label,
    )


def _write_excited_state_optimization_csv_sections(
    data: Dict[str, Any],
    directory: Path,
    stem: str,
    write_csv: WriteCSV,
) -> List[Path]:
    """Write excited-state optimization CSV sections via the family hook."""
    from .output import job_state as _job_state
    from .output.csv_sections_basic import write_geom_opt_section
    from .output.csv_sections_state import write_excited_state_optimization_section

    files = write_excited_state_optimization_section(
        data,
        directory,
        stem,
        write_csv=write_csv,
        electronic_state_label=lambda dataset: _job_state.electronic_state_label(
            dataset,
            ground_state_label="Ground-state",
        ),
        get_excited_state_opt_data=_job_state.get_excited_state_opt_data,
        excited_state_target_label=_job_state.excited_state_target_label,
        bool_to_label=_job_state.bool_to_label,
        format_simple_vector=_job_state.format_simple_vector,
    )
    files.extend(
        write_geom_opt_section(
            data,
            directory,
            stem,
            write_csv=write_csv,
        )
    )
    return files


register_calculation_family_plugin(
    CalculationFamilyPlugin(
        family="deltascf",
        default_calculation_label="DeltaSCF",
        matcher=_matches_deltascf,
        electronic_state_kind="deltascf_excited_state",
        build_special_electronic_state_label=_build_deltascf_state_label,
        render_markdown_sections=_render_deltascf_markdown_sections,
        render_comparison_sections=_render_deltascf_comparison_sections,
        csv_writers=(_write_deltascf_csv_sections,),
        comparison_order=10,
    )
)
register_calculation_family_plugin(
    CalculationFamilyPlugin(
        family="excited_state_optimization",
        default_calculation_label="Excited-State Geometry Optimization",
        matcher=_matches_excited_state_optimization,
        electronic_state_kind="excited_state_optimization",
        build_special_electronic_state_label=_build_excited_state_optimization_label,
        build_state_target_label=_build_excited_state_target_label,
        render_markdown_sections=_render_excited_state_optimization_markdown_sections,
        render_comparison_sections=_render_excited_state_optimization_comparison_sections,
        csv_writers=(_write_excited_state_optimization_csv_sections,),
        comparison_order=20,
    )
)
register_calculation_family_plugin(
    CalculationFamilyPlugin(
        family="single_point",
        default_calculation_label="Single Point",
        matcher=_matches_single_point,
    )
)
