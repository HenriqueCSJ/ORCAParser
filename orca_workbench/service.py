"""Parser-backed service layer for ORCA Workbench.

This module owns the non-UI behavior used by the FastAPI backend and fallback
desktop launcher:

* discover ORCA output files
* expose parser sections and plugin options as declarative UI controls
* run the existing parser and output writers
* build compact scientific summaries, provenance notes, and preview payloads

It deliberately does not parse ORCA text itself.  All scientific extraction,
including spectra, populations, NBO/NPA, TDDFT, CASSCF, NEVPT2, GOAT, and
coupled-cluster sections, is delegated to :class:`orca_parser.ORCAParser` and
the existing output writers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from orca_parser import ORCAParser
from orca_parser.final_snapshot import (
    get_final_dipole,
    get_final_orbital_energies,
)
from orca_parser.output.job_state import (
    calculation_type_label,
    electronic_state_label,
    get_basis_set,
    get_charge,
    get_excited_state_opt_data,
    get_job_name,
    get_method_header_label,
    get_multiplicity,
    symmetry_inline_label,
)
from orca_parser.parser import is_auxiliary_orca_file
from orca_parser.parser_section_registry import (
    get_parser_section_alias_map,
    get_registered_parser_section_plugins,
)
from orca_parser.plugin_discovery import (
    bootstrap_plugin_bundles,
    get_registered_plugin_options,
)
from orca_parser.render_options import RENDER_OPTION_UNSET


ORCA_OUTPUT_EXTENSIONS = {".out", ".log"}
SKIPPED_SCAN_DIRS = {
    ".git",
    ".pytest_cache",
    ".pytest_tmp",
    "__pycache__",
    "build",
    "dist",
    "orca_parser.egg-info",
}


@dataclass(frozen=True)
class SectionChoice:
    """One selectable parser section or alias for the GUI."""

    key: str
    label: str
    kind: str
    expands_to: tuple[str, ...] = ()
    always_include: bool = False


@dataclass
class ExportOptions:
    """Per-file export settings used by the workbench."""

    output_dir: Path | None = None
    write_json: bool = True
    write_csv: bool = True
    write_markdown: bool = False
    write_hdf5: bool = False
    compare_markdown: bool = False
    json_indent: int = 2
    json_strip_none: bool = False
    json_compress: bool = False
    h5_compression: str | None = "gzip"
    h5_level: int = 4
    detail_scope: str = "auto"
    goat_max_relative_energy_kcal_mol: Any = RENDER_OPTION_UNSET


@dataclass
class WorkbenchSummary:
    """Small normalized summary used for cards, lists, and quick previews."""

    source_file: str
    file_name: str
    job_name: str = ""
    calculation_type: str = ""
    electronic_state: str = ""
    method: str = ""
    basis_set: str = ""
    charge: Any = ""
    multiplicity: Any = ""
    symmetry: str = ""
    reference: str = ""
    energy_Eh: Any = None
    homo_lumo_gap_eV: Any = None
    dipole_D: Any = None
    opt_cycles: Any = None
    opt_converged: Any = None
    tddft_trajectory_class: str = ""
    final_s1_wavelength_nm: Any = None
    final_s1_oscillator_strength: Any = None
    parsed_sections: tuple[str, ...] = ()
    warning_count: int = 0

    def as_text(self) -> str:
        """Render the summary as readable plain text for the preview pane."""

        lines = [
            f"File: {self.file_name}",
            f"Job: {self.job_name or 'N/A'}",
            f"Calculation: {self.calculation_type or 'N/A'}",
            f"State: {self.electronic_state or 'Ground-state'}",
            f"Method: {self.method or 'N/A'}",
            f"Basis: {self.basis_set or 'N/A'}",
            f"Charge / multiplicity: {self.charge} / {self.multiplicity}",
            f"Reference: {self.reference or 'N/A'}",
            f"Symmetry: {self.symmetry or 'none'}",
        ]
        if self.energy_Eh is not None:
            lines.append(f"Energy: {format_number(self.energy_Eh, 10)} Eh")
        if self.homo_lumo_gap_eV is not None:
            lines.append(f"HOMO-LUMO gap: {format_number(self.homo_lumo_gap_eV, 4)} eV")
        if self.dipole_D is not None:
            lines.append(f"Dipole: {format_number(self.dipole_D, 4)} D")
        if self.opt_cycles is not None:
            converged = yes_no(self.opt_converged)
            lines.append(f"Optimization cycles: {self.opt_cycles} (converged: {converged})")
        if self.tddft_trajectory_class:
            lines.append(f"TDDFT trajectory: {self.tddft_trajectory_class}")
        if self.final_s1_wavelength_nm is not None:
            lines.append(
                "Final S1: "
                f"{format_number(self.final_s1_wavelength_nm, 1)} nm, "
                f"f={format_number(self.final_s1_oscillator_strength, 5)}"
            )
        lines.append(f"Parsed sections: {', '.join(self.parsed_sections) or 'N/A'}")
        lines.append(f"Warnings / parser notes: {self.warning_count}")
        return "\n".join(lines)


@dataclass
class WorkbenchResult:
    """Result of one parser run."""

    path: Path
    status: str
    summary: WorkbenchSummary | None = None
    parser: ORCAParser | None = None
    data: dict[str, Any] = field(default_factory=dict)
    output_paths: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str = ""


def discover_orca_outputs(paths: Iterable[str | Path]) -> list[Path]:
    """Discover ORCA ``.out`` / ``.log`` files from files or directories."""

    discovered: list[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if path.is_dir():
            for candidate in _iter_orca_outputs(path):
                resolved = candidate.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    discovered.append(candidate)
        elif _is_orca_output_candidate(path):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                discovered.append(path)
    return sorted(discovered, key=lambda item: str(item).lower())


def _iter_orca_outputs(root: Path) -> Iterable[Path]:
    """Yield output candidates while skipping common cache/build folders."""

    for child in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        if child.is_dir():
            if child.name in SKIPPED_SCAN_DIRS:
                continue
            yield from _iter_orca_outputs(child)
        elif _is_orca_output_candidate(child):
            yield child


def _is_orca_output_candidate(path: Path) -> bool:
    """Return True when *path* looks like a parseable ORCA output candidate."""

    return (
        path.suffix.lower() in ORCA_OUTPUT_EXTENSIONS
        and not is_auxiliary_orca_file(path)
    )


def available_section_choices() -> list[SectionChoice]:
    """Return parser aliases and section keys for GUI selection."""

    bootstrap_plugin_bundles()
    alias_map = get_parser_section_alias_map()
    plugins = get_registered_parser_section_plugins()

    choices = [
        SectionChoice(
            key="all",
            label="all - every registered parser section",
            kind="alias",
            expands_to=tuple(alias_map.get("all", ())),
        )
    ]

    for alias, section_keys in sorted(alias_map.items()):
        if alias == "all":
            continue
        expanded = tuple(section_keys)
        label = f"{alias} - {', '.join(expanded)}"
        choices.append(
            SectionChoice(
                key=alias,
                label=label,
                kind="alias",
                expands_to=expanded,
            )
        )

    for plugin in plugins:
        label = plugin.key
        if plugin.always_include:
            label = f"{plugin.key} - core"
        choices.append(
            SectionChoice(
                key=plugin.key,
                label=label,
                kind="section",
                expands_to=(plugin.key,),
                always_include=plugin.always_include,
            )
        )

    deduped: dict[tuple[str, str], SectionChoice] = {}
    for choice in choices:
        deduped[(choice.kind, choice.key)] = choice
    return list(deduped.values())


def default_plugin_options() -> dict[str, Any]:
    """Return plugin option defaults discovered from parser modules."""

    bootstrap_plugin_bundles()
    return {
        option.dest: option.default
        for option in get_registered_plugin_options()
    }


def available_plugin_options():
    """Return declarative plugin options for dynamic GUI controls."""

    bootstrap_plugin_bundles()
    return get_registered_plugin_options()


def parse_orca_file(
    path: str | Path,
    *,
    sections: list[str] | None = None,
    plugin_options: Mapping[str, Any] | None = None,
    export_options: ExportOptions | None = None,
) -> WorkbenchResult:
    """Parse one ORCA file through :class:`ORCAParser`.

    The workbench passes selected sections and plugin options through unchanged,
    then wraps the parsed payload with warnings, exports, a compact summary, and
    provenance-friendly state for the GUI.
    """

    file_path = Path(path)
    try:
        parser = ORCAParser(file_path, plugin_options=dict(plugin_options or {}))
        data = parser.parse(sections=sections)
    except Exception as exc:  # noqa: BLE001
        return WorkbenchResult(
            path=file_path,
            status="failed",
            warnings=[],
            error=str(exc),
        )

    warnings = collect_warnings(data)
    output_paths: list[Path] = []
    if export_options is not None:
        output_paths.extend(_write_outputs(parser, file_path, export_options, warnings))

    summary = build_workbench_summary(data, file_path, warning_count=len(warnings))
    return WorkbenchResult(
        path=file_path,
        status="parsed",
        summary=summary,
        parser=parser,
        data=data,
        output_paths=output_paths,
        warnings=warnings,
    )


def _write_outputs(
    parser: ORCAParser,
    path: Path,
    options: ExportOptions,
    warnings: list[str],
) -> list[Path]:
    """Write selected parser outputs, returning the paths that succeeded."""

    outdir = options.output_dir or path.parent
    outdir.mkdir(parents=True, exist_ok=True)
    stem = path.stem
    written: list[Path] = []

    def record_export(label: str, callback) -> None:
        try:
            value = callback()
            if isinstance(value, list):
                written.extend(Path(item) for item in value)
            elif value is not None:
                written.append(Path(value))
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{label} export failed: {exc}")

    if options.write_json:
        record_export(
            "JSON",
            lambda: parser.to_json(
                outdir / f"{stem}.json",
                indent=options.json_indent,
                strip_none=options.json_strip_none,
                compress=options.json_compress,
            ),
        )
    if options.write_csv:
        record_export("CSV", lambda: parser.to_csv(outdir / f"{stem}_csv"))
    if options.write_markdown:
        markdown_kwargs = {"detail_scope": options.detail_scope}
        if options.goat_max_relative_energy_kcal_mol is not RENDER_OPTION_UNSET:
            markdown_kwargs["goat_max_relative_energy_kcal_mol"] = (
                options.goat_max_relative_energy_kcal_mol
            )
        record_export(
            "Markdown",
            lambda: parser.to_markdown(outdir / f"{stem}.md", **markdown_kwargs),
        )
    if options.write_hdf5:
        record_export(
            "HDF5",
            lambda: parser.to_hdf5(
                outdir / f"{stem}.h5",
                compression=options.h5_compression,
                compression_opts=options.h5_level,
            ),
        )
    return written


def write_comparison_report(
    results: Iterable[WorkbenchResult],
    output_dir: str | Path,
    *,
    detail_scope: str = "auto",
    goat_max_relative_energy_kcal_mol: Any = RENDER_OPTION_UNSET,
) -> Path | None:
    """Write a comparison report for successfully parsed jobs."""

    parsers = [
        result.parser
        for result in results
        if result.parser is not None and result.status == "parsed"
    ]
    if len(parsers) < 2:
        return None
    kwargs = {"detail_scope": detail_scope}
    if goat_max_relative_energy_kcal_mol is not RENDER_OPTION_UNSET:
        kwargs["goat_max_relative_energy_kcal_mol"] = goat_max_relative_energy_kcal_mol
    return ORCAParser.compare(parsers, Path(output_dir) / "comparison.md", **kwargs)


def build_workbench_summary(
    data: Mapping[str, Any],
    path: str | Path,
    *,
    warning_count: int = 0,
) -> WorkbenchSummary:
    """Build the cross-module summary shown in Workbench job lists.

    Values are read from normalized parser views where possible, so repeated
    optimization-step data, final snapshots, and excited-state trajectory
    summaries remain separated from one another.
    """

    file_path = Path(path)
    context = data.get("context") if isinstance(data.get("context"), dict) else {}
    scf = data.get("scf") if isinstance(data.get("scf"), dict) else {}
    geom_opt = data.get("geom_opt") if isinstance(data.get("geom_opt"), dict) else {}
    orbital_energies = get_final_orbital_energies(dict(data))
    dipole = get_final_dipole(dict(data))
    tddft = data.get("tddft") if isinstance(data.get("tddft"), dict) else {}
    trajectory = {}
    if isinstance(tddft, dict):
        trajectory = (tddft.get("trajectory") or {}).get("final_summary") or {}

    parsed_sections = tuple(
        sorted(
            key
            for key, value in data.items()
            if key not in {"source_file", "context"}
            and not key.endswith("_parse_error")
            and value not in (None, {}, [])
        )
    )

    state_label = electronic_state_label(dict(data), ground_state_label="Ground-state")
    energy = scf.get("final_single_point_energy_Eh") or scf.get("total_energy_Eh")
    gap = orbital_energies.get("HOMO_LUMO_gap_eV")
    dipole_magnitude = dipole.get("magnitude_Debye")

    return WorkbenchSummary(
        source_file=str(file_path),
        file_name=file_path.name,
        job_name=get_job_name(dict(data)) or file_path.stem,
        calculation_type=calculation_type_label(dict(data)) or "",
        electronic_state=state_label or "Ground-state",
        method=get_method_header_label(dict(data)) or "",
        basis_set=get_basis_set(dict(data)) or "",
        charge=get_charge(dict(data)),
        multiplicity=get_multiplicity(dict(data)),
        symmetry=symmetry_inline_label(dict(data)) or "",
        reference=str(context.get("reference_type") or context.get("hf_type") or ""),
        energy_Eh=energy,
        homo_lumo_gap_eV=gap,
        dipole_D=dipole_magnitude,
        opt_cycles=geom_opt.get("n_cycles"),
        opt_converged=geom_opt.get("converged"),
        tddft_trajectory_class=str(trajectory.get("trajectory_class") or ""),
        final_s1_wavelength_nm=trajectory.get("final_S1_wavelength_nm"),
        final_s1_oscillator_strength=trajectory.get("final_S1_oscillator_strength"),
        parsed_sections=parsed_sections,
        warning_count=warning_count,
    )


def build_provenance_text(data: Mapping[str, Any]) -> str:
    """Return a provenance-focused plain text view for the GUI.

    The view highlights source file, job/final snapshot selection,
    excited-state root-following metadata, density-analysis context, and parser
    notes without relabeling reference-density data as excited-state data.
    """

    lines: list[str] = []
    source_file = data.get("source_file")
    if source_file:
        lines.append(f"Source file: {source_file}")

    job_snapshot = data.get("job_snapshot")
    if isinstance(job_snapshot, dict) and job_snapshot:
        lines.append("\nJob snapshot")
        for key in (
            "job_id",
            "source_relpath",
            "calculation_family",
            "electronic_state_kind",
            "state_target_label",
            "is_excited_state_optimization",
            "is_deltascf",
            "is_surface_scan",
            "is_goat",
        ):
            if key in job_snapshot:
                lines.append(f"  {key}: {job_snapshot[key]}")

    final_snapshot = data.get("final_snapshot")
    if isinstance(final_snapshot, dict) and final_snapshot:
        lines.append("\nFinal snapshot")
        lines.append(f"  selection: {final_snapshot.get('selection', '')}")
        charge_schemes = sorted((final_snapshot.get("charges") or {}).keys())
        if charge_schemes:
            lines.append(f"  charge schemes: {', '.join(charge_schemes)}")

    excopt = get_excited_state_opt_data(dict(data))
    if excopt:
        lines.append("\nExcited-state optimization")
        for key in (
            "input_block",
            "input_nroots",
            "target_root",
            "target_multiplicity",
            "target_state_label",
            "followiroot",
            "final_root",
            "input_electron_density",
        ):
            if key in excopt:
                lines.append(f"  {key}: {excopt[key]}")

    density = data.get("density_analysis")
    if isinstance(density, dict):
        summary = density.get("summary")
        if isinstance(summary, dict) and summary:
            lines.append("\nDensity analysis")
            for key, value in summary.items():
                lines.append(f"  {key}: {value}")

    warnings = collect_warnings(data)
    if warnings:
        lines.append("\nWarnings and parser notes")
        for warning in warnings:
            lines.append(f"  - {warning}")
    return "\n".join(lines) if lines else "No provenance data available."


def preview_json_views(data: Mapping[str, Any]) -> str:
    """Return a compact JSON preview of the normalized parser views."""

    preview = {
        key: data.get(key)
        for key in ("job_snapshot", "final_snapshot", "job_series")
        if isinstance(data.get(key), dict)
    }
    if not preview:
        preview = {"available_top_level_keys": sorted(data.keys())}
    return json.dumps(preview, indent=2, default=str)


def collect_warnings(data: Mapping[str, Any], *, limit: int = 80) -> list[str]:
    """Collect parse errors and warning-like fields from a parsed payload."""

    warnings: list[str] = []
    _collect_warning_values(data, warnings, path=(), limit=limit)
    return warnings[:limit]


def _collect_warning_values(
    value: Any,
    warnings: list[str],
    *,
    path: tuple[str, ...],
    limit: int,
) -> None:
    """Recursive warning collector used by :func:`collect_warnings`."""

    if len(warnings) >= limit:
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            lowered = key_text.lower()
            child_path = (*path, key_text)
            if key_text.endswith("_parse_error"):
                warnings.append(f"{'.'.join(child_path)}: {child}")
            elif lowered in {"warning", "warnings", "parse_warnings"}:
                _append_warning_payload(warnings, child_path, child, limit)
            elif "warning" in lowered and not isinstance(child, (dict, list, tuple)):
                warnings.append(f"{'.'.join(child_path)}: {child}")
            if len(warnings) >= limit:
                return
            if isinstance(child, (Mapping, list, tuple)):
                _collect_warning_values(child, warnings, path=child_path, limit=limit)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            if len(warnings) >= limit:
                return
            if isinstance(child, (Mapping, list, tuple)):
                _collect_warning_values(child, warnings, path=(*path, str(index)), limit=limit)


def _append_warning_payload(
    warnings: list[str],
    path: tuple[str, ...],
    payload: Any,
    limit: int,
) -> None:
    """Append a warning payload in a compact way."""

    label = ".".join(path)
    if isinstance(payload, str):
        warnings.append(f"{label}: {payload}")
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            if len(warnings) >= limit:
                return
            warnings.append(f"{label}: {item}")
    elif isinstance(payload, Mapping):
        for key, item in payload.items():
            if len(warnings) >= limit:
                return
            warnings.append(f"{label}.{key}: {item}")
    elif payload:
        warnings.append(f"{label}: {payload}")


def yes_no(value: Any) -> str:
    """Format booleans for human-facing text."""

    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "?"


def format_number(value: Any, digits: int) -> str:
    """Format a number defensively for workbench previews."""

    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    if value is None:
        return "N/A"
    return str(value)
