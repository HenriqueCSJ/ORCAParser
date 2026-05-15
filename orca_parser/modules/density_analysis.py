"""Density-specific population analysis for double-hybrid and MP2 runs.

ORCA double-hybrid optimizations can print population analyses for several
density objects at each geometry: the SCF density, the unrelaxed MP2 density,
and the relaxed MP2 density. This module preserves those contexts separately
and delegates the actual population/NBO grammars to their owning modules.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..output.csv_section_registry import CSVSectionPlugin
from ..output.markdown_section_registry import MarkdownSectionPlugin
from ..parser_section_plugin import ParserSectionAlias, ParserSectionPlugin
from ..plugin_bundle import PluginBundle, PluginMetadata
from .base import BaseModule
from .dipole import parse_dipole_moment_blocks
from .nbo import NBOModule
from .population import (
    CHELPGModule,
    HirshfeldModule,
    LoewdinModule,
    MBISModule,
    MayerModule,
    MullikenModule,
)


_FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
_CYCLE_RE = re.compile(r"GEOMETRY OPTIMIZATION CYCLE\s+(\d+)", re.I)
_FINAL_STATIONARY_RE = re.compile(r"FINAL ENERGY EVALUATION AT THE STATIONARY POINT", re.I)
_SCF_START_RE = re.compile(r"^\s*\*\s*MULLIKEN POPULATION ANALYSIS\s*\*\s*$", re.I)
_UNRELAXED_START_RE = re.compile(r"UNRELAXED MP2 DENSITY POPULATION ANALYSIS", re.I)
_RELAXED_START_RE = re.compile(r"RELAXED MP2 DENSITY POPULATION ANALYSIS", re.I)
_INPUT_DENSITY_RE = re.compile(r"Input electron density\s+\.+\s+(\S+)", re.I)
_BASE_NAME_RE = re.compile(r"BaseName \([^)]*\)\s+\.+\s+(\S+)", re.I)
_SCF_DENSITY_RE = re.compile(r"Input SCF Electron Density\s+\.+\s+(\S+)", re.I)
_CORRELATED_DENSITY_RE = re.compile(r"Input Correlated Electron Density\s+\.+\s+(\S+)", re.I)
_WEIGHTED_DENSITY_RE = re.compile(r"Input Energy Weighted Density\s+\.+\s+(\S+)", re.I)
_MP2_CORRELATION_RE = re.compile(r"RI-MP2 CORRELATION ENERGY:\s+(?P<value>%s)\s+Eh" % _FLOAT_RE, re.I)
_MP2_TOTAL_RE = re.compile(r"MP2 TOTAL ENERGY:\s+(?P<value>%s)\s+Eh" % _FLOAT_RE, re.I)
_GRADIENT_NORM_RE = re.compile(r"NORM OF THE MP2 GRADIENT:\s+(?P<value>%s)" % _FLOAT_RE, re.I)
_MAX_MEMORY_RE = re.compile(r"Maximum memory used.*:\s+(?P<value>%s)\s+MB" % _FLOAT_RE, re.I)
_TRACE_RE = re.compile(r"Trace of the density to be diagonalized\s*=\s*(?P<value>%s)" % _FLOAT_RE, re.I)
_NAT_OCC_RE = re.compile(r"N\[\s*(?P<index>\d+)\]\s*=\s*(?P<value>%s)" % _FLOAT_RE)
_KEY_VALUE_RE = re.compile(r"^\s*(?P<key>[^.]+?)\s+\.\.\.\s+(?P<value>\S.*)$")


_POPULATION_MODULES: tuple[tuple[str, type[BaseModule]], ...] = (
    ("mulliken", MullikenModule),
    ("loewdin", LoewdinModule),
    ("mayer", MayerModule),
    ("hirshfeld", HirshfeldModule),
    ("mbis", MBISModule),
    ("chelpg", CHELPGModule),
    ("nbo", NBOModule),
)


class DensityAnalysisModule(BaseModule):
    """Parse density-specific population analyses without reimplementing them."""

    name = "density_analysis"

    def parse(self, lines: List[str]) -> Optional[Dict[str, Any]]:
        starts = _find_density_population_starts(lines)
        formations = _parse_mp2_density_formations(lines)
        dipoles = _parse_density_dipoles(lines)
        sidecars = _detect_density_sidecars(self.context)

        if not starts and not formations and not dipoles and not sidecars:
            return None

        analyses = _parse_density_population_blocks(lines, starts, formations, self.context)
        by_stage = _group_by_stage(analyses)

        data: Dict[str, Any] = {
            "summary": _build_summary(analyses, formations, sidecars, dipoles),
        }
        if analyses:
            data["analyses"] = analyses
            data["by_stage"] = by_stage
        if formations:
            data["mp2_density_formations"] = formations
        if dipoles:
            data["dipoles"] = dipoles
        if sidecars:
            data["sidecar_files"] = sidecars

        return data


def _find_density_population_starts(lines: Sequence[str]) -> List[Dict[str, Any]]:
    starts: List[Dict[str, Any]] = []
    inside_mp2_population = False
    for idx, line in enumerate(lines):
        upper = line.upper()
        if _UNRELAXED_START_RE.search(upper):
            inside_mp2_population = True
            starts.append({
                "line_index": idx,
                "density_kind": "mp2_unrelaxed",
                "method": "MP2",
                "level": "unrelaxed",
            })
            continue
        if _RELAXED_START_RE.search(upper):
            inside_mp2_population = True
            starts.append({
                "line_index": idx,
                "density_kind": "mp2_relaxed",
                "method": "MP2",
                "level": "relaxed",
            })
            continue
        if inside_mp2_population and _ends_mp2_density_population_context(upper):
            inside_mp2_population = False
        if _SCF_START_RE.match(line) and not inside_mp2_population:
            starts.append({
                "line_index": idx,
                "density_kind": "scf",
                "method": "SCF",
                "level": "scf",
            })
    return sorted(starts, key=lambda item: int(item["line_index"]))


def _ends_mp2_density_population_context(upper_line: str) -> bool:
    return any(
        pattern in upper_line
        for pattern in (
            "RI-MP2 ENERGY+GRADIENT",
            "MP2 TOTAL ENERGY",
            "THE FINAL MP2 GRADIENT",
            "FINAL SINGLE POINT ENERGY",
            "FINAL ENERGY EVALUATION AT THE STATIONARY POINT",
            "OPTIMIZATION RUN DONE",
            "ORCA TERMINATED NORMALLY",
        )
    )


def _parse_density_population_blocks(
    lines: Sequence[str],
    starts: Sequence[Dict[str, Any]],
    formations: Sequence[Dict[str, Any]],
    context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    analyses: List[Dict[str, Any]] = []
    for pos, start_info in enumerate(starts):
        start = int(start_info["line_index"])
        next_start = int(starts[pos + 1]["line_index"]) if pos + 1 < len(starts) else len(lines)
        end = _density_block_end(lines, start, next_start, str(start_info["density_kind"]))
        block_lines = list(lines[start:end])
        stage_info = _stage_for_line(lines, start)
        density_file = _density_file_for_block(lines, block_lines, start, formations, start_info, context)

        record: Dict[str, Any] = {
            "analysis_index": len(analyses) + 1,
            "stage": stage_info["stage"],
            "geometry_stage": stage_info["geometry_stage"],
            "optimization_cycle": stage_info.get("optimization_cycle"),
            "density_kind": start_info["density_kind"],
            "method": start_info["method"],
            "level": start_info["level"],
            "line_start": start + 1,
            "line_end": end,
            "input_electron_density_file": density_file,
        }
        base_name = _first_match(block_lines, _BASE_NAME_RE)
        if base_name:
            record["base_name"] = base_name

        parsed_sections = _parse_shared_population_sections(block_lines, context)
        if parsed_sections:
            record["population"] = parsed_sections
            record["population_summary"] = _summarize_population_sections(parsed_sections)

        analyses.append(record)
    return analyses


def _density_block_end(
    lines: Sequence[str],
    start: int,
    next_start: int,
    density_kind: str,
) -> int:
    boundary_patterns = []
    if density_kind == "scf":
        boundary_patterns = [
            "RI-MP2 ENERGY+GRADIENT",
            "ORCA  MP2",
            "MP2 DENSITY FORMATION",
        ]
    else:
        boundary_patterns = [
            "MP2 TOTAL ENERGY",
            "THE FINAL MP2 GRADIENT",
            "FINAL SINGLE POINT ENERGY",
            "FINAL ENERGY EVALUATION AT THE STATIONARY POINT",
            "OPTIMIZATION RUN DONE",
        ]

    end = next_start
    for idx in range(start + 1, min(next_start, len(lines))):
        upper = lines[idx].upper()
        if any(pattern in upper for pattern in boundary_patterns):
            end = idx
            break
    return end


def _parse_shared_population_sections(
    block_lines: List[str],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    sections: Dict[str, Any] = {}
    module_context = dict(context)
    for key, module_class in _POPULATION_MODULES:
        try:
            parsed = module_class(module_context).parse(block_lines)
        except Exception as exc:  # noqa: BLE001
            sections[f"{key}_parse_error"] = str(exc)
            continue
        if parsed:
            sections[key] = parsed
    return sections


def _summarize_population_sections(sections: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    mulliken = sections.get("mulliken") or {}
    if mulliken.get("sum_of_charges") is not None:
        summary["mulliken_sum_of_charges"] = mulliken["sum_of_charges"]
    if mulliken.get("atomic_charges"):
        summary["mulliken_atom_count"] = len(mulliken["atomic_charges"])
    loewdin = sections.get("loewdin") or {}
    if loewdin.get("sum_of_charges") is not None:
        summary["loewdin_sum_of_charges"] = loewdin["sum_of_charges"]
    if loewdin.get("atomic_charges"):
        summary["loewdin_atom_count"] = len(loewdin["atomic_charges"])
    hirshfeld = sections.get("hirshfeld") or {}
    for key in ("total_alpha_density", "total_beta_density"):
        if hirshfeld.get(key) is not None:
            summary[f"hirshfeld_{key}"] = hirshfeld[key]
    mayer = sections.get("mayer") or {}
    if mayer.get("atomic_data"):
        summary["mayer_atom_count"] = len(mayer["atomic_data"])
    nbo = sections.get("nbo") or {}
    npa = nbo.get("npa_summary") or nbo.get("npa") or {}
    if isinstance(npa, dict) and npa.get("atoms"):
        summary["nbo_npa_atom_count"] = len(npa["atoms"])
    elif isinstance(npa, list):
        summary["nbo_npa_atom_count"] = len(npa)
    return summary


def _parse_mp2_density_formations(lines: Sequence[str]) -> List[Dict[str, Any]]:
    starts = [
        idx for idx, line in enumerate(lines)
        if "RI-MP2 ENERGY+GRADIENT" in line.upper()
    ]
    formations: List[Dict[str, Any]] = []
    for pos, start in enumerate(starts):
        end = starts[pos + 1] if pos + 1 < len(starts) else len(lines)
        block = lines[start:end]
        stage_info = _stage_for_line(lines, start)
        formation: Dict[str, Any] = {
            "formation_index": len(formations) + 1,
            "stage": stage_info["stage"],
            "geometry_stage": stage_info["geometry_stage"],
            "optimization_cycle": stage_info.get("optimization_cycle"),
            "line_start": start + 1,
            "line_end": end,
        }

        settings = _parse_mp2_settings(block)
        if settings:
            formation["settings"] = settings

        natural_occupations: List[Dict[str, Any]] = []
        for line in block:
            if _MP2_CORRELATION_RE.search(line):
                formation["ri_mp2_correlation_energy_Eh"] = float(_MP2_CORRELATION_RE.search(line).group("value"))
            if _MP2_TOTAL_RE.search(line):
                formation["mp2_total_energy_Eh"] = float(_MP2_TOTAL_RE.search(line).group("value"))
            if _GRADIENT_NORM_RE.search(line):
                formation["mp2_gradient_norm"] = float(_GRADIENT_NORM_RE.search(line).group("value"))
            if _MAX_MEMORY_RE.search(line):
                formation["maximum_memory_MB"] = float(_MAX_MEMORY_RE.search(line).group("value"))
            if "Storing the unrelaxed density" in line:
                formation["stored_unrelaxed_density"] = "done" in line.lower()
            if "Storing the relaxed density" in line:
                formation["stored_relaxed_density"] = "done" in line.lower()
            trace = _TRACE_RE.search(line)
            if trace:
                formation["density_trace"] = float(trace.group("value"))
            if "Sum of eigenvalues" in line:
                value = re.search(_FLOAT_RE, line)
                if value:
                    formation["natural_occupation_sum"] = float(value.group(0))
            occ = _NAT_OCC_RE.search(line)
            if occ:
                natural_occupations.append({
                    "index": int(occ.group("index")),
                    "occupation": float(occ.group("value")),
                })
            for regex, key in (
                (_SCF_DENSITY_RE, "scf_density_file"),
                (_CORRELATED_DENSITY_RE, "correlated_density_file"),
                (_WEIGHTED_DENSITY_RE, "energy_weighted_density_file"),
            ):
                match = regex.search(line)
                if match:
                    formation[key] = match.group(1)

        if natural_occupations:
            formation["natural_occupation_count"] = len(natural_occupations)
            formation["natural_occupations"] = natural_occupations
        if _formation_has_density_signal(formation):
            formations.append(formation)
    return formations


def _parse_mp2_settings(block: Sequence[str]) -> Dict[str, Any]:
    settings: Dict[str, Any] = {}
    key_map = {
        "Dimension of the orbital basis": "orbital_basis_dimension",
        "Dimension of the AuxC basis": "auxc_basis_dimension",
        "Memory devoted to MP2": "memory_MB",
        "Data format for matrix containers": "matrix_container_format",
        "Compression type for matrix containers": "matrix_container_compression",
        "Scaling for aa/bb pairs": "aa_bb_pair_scaling",
        "Scaling for ab pairs": "ab_pair_scaling",
        "Overall scaling of the MP2 energy": "overall_mp2_energy_scaling",
        "Max. number of iterations": "cpscf_max_iterations",
        "Convergence Tolerance": "cpscf_convergence_tolerance",
        "Number of perturbations": "cpscf_perturbations",
        "Perturbation type": "cpscf_perturbation_type",
    }
    for line in block:
        match = _KEY_VALUE_RE.match(line)
        if not match:
            continue
        printed_key = match.group("key").strip()
        target_key = key_map.get(printed_key)
        if not target_key:
            continue
        settings[target_key] = _coerce_scalar(match.group("value").strip())
    return settings


def _formation_has_density_signal(formation: Dict[str, Any]) -> bool:
    signal_keys = {
        "stored_unrelaxed_density",
        "stored_relaxed_density",
        "density_trace",
        "scf_density_file",
        "correlated_density_file",
        "natural_occupation_count",
    }
    return any(key in formation for key in signal_keys)


def _parse_density_dipoles(lines: Sequence[str]) -> List[Dict[str, Any]]:
    dipoles = []
    for block in parse_dipole_moment_blocks(list(lines)):
        method = str(block.get("method", "")).upper()
        level = str(block.get("level", "")).lower()
        if method not in {"SCF", "MP2"}:
            continue
        stage_info = _stage_for_line(lines, int(block.get("line", 1)) - 1)
        density_kind = "scf"
        if method == "MP2" and "unrelaxed" in level:
            density_kind = "mp2_unrelaxed"
        elif method == "MP2" and "relaxed" in level:
            density_kind = "mp2_relaxed"
        block.update({
            "stage": stage_info["stage"],
            "geometry_stage": stage_info["geometry_stage"],
            "optimization_cycle": stage_info.get("optimization_cycle"),
            "density_kind": density_kind,
        })
        dipoles.append(block)
    return dipoles


def _stage_for_line(lines: Sequence[str], idx: int) -> Dict[str, Any]:
    final_idx = _find_first(lines, _FINAL_STATIONARY_RE)
    cycle_markers = _cycle_markers(lines)
    cycle = None
    for line_idx, cycle_number in cycle_markers:
        if line_idx <= idx:
            cycle = cycle_number
        else:
            break

    if final_idx != -1 and idx > final_idx:
        return {"stage": "final", "geometry_stage": "final", "optimization_cycle": cycle}
    if cycle_markers and cycle == cycle_markers[0][1]:
        return {"stage": "initial", "geometry_stage": "initial", "optimization_cycle": cycle}
    if cycle is not None:
        return {"stage": f"cycle_{cycle}", "geometry_stage": "optimization_cycle", "optimization_cycle": cycle}
    return {"stage": "single_point", "geometry_stage": "single_point", "optimization_cycle": None}


def _cycle_markers(lines: Sequence[str]) -> List[tuple[int, int]]:
    markers: List[tuple[int, int]] = []
    for idx, line in enumerate(lines):
        match = _CYCLE_RE.search(line)
        if match:
            markers.append((idx, int(match.group(1))))
    return markers


def _find_first(lines: Sequence[str], regex: re.Pattern[str]) -> int:
    for idx, line in enumerate(lines):
        if regex.search(line):
            return idx
    return -1


def _density_file_for_block(
    lines: Sequence[str],
    block_lines: Sequence[str],
    start: int,
    formations: Sequence[Dict[str, Any]],
    start_info: Dict[str, Any],
    context: Dict[str, Any],
) -> str:
    direct = _first_match(block_lines, _INPUT_DENSITY_RE)
    if direct:
        return direct
    density_kind = str(start_info["density_kind"])
    if density_kind == "scf":
        formation = _nearest_following_formation(start, formations)
        if formation and formation.get("scf_density_file"):
            return str(formation["scf_density_file"])
        stem = context.get("source_stem")
        if stem:
            return f"{stem}.scfp"
    return ""


def _nearest_following_formation(
    line_index: int,
    formations: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    for formation in formations:
        if int(formation.get("line_start", 0)) >= line_index + 1:
            return formation
    return None


def _first_match(lines: Sequence[str], regex: re.Pattern[str]) -> str:
    for line in lines:
        match = regex.search(line)
        if match:
            return match.group(1)
    return ""


def _coerce_scalar(value: str) -> Any:
    cleaned = value.strip()
    if cleaned.upper() in {"YES", "NO"}:
        return cleaned.upper() == "YES"
    number = re.match(rf"^({_FLOAT_RE})\s*(?:MB)?\s*$", cleaned, re.I)
    if number:
        token = number.group(1)
        try:
            if any(ch in token for ch in ".eE"):
                return float(token)
            return int(token)
        except ValueError:
            return token
    return cleaned


def _detect_density_sidecars(context: Dict[str, Any]) -> List[Dict[str, Any]]:
    source_dir = context.get("source_dir")
    source_stem = context.get("source_stem")
    if not source_dir or not source_stem:
        return []
    directory = Path(str(source_dir))
    sidecars: List[Dict[str, Any]] = []
    for suffix, role in (
        (".densities", "density_container"),
        (".densitiesinfo", "density_container_index"),
        (".mp2nat", "mp2_natural_orbitals"),
    ):
        path = directory / f"{source_stem}{suffix}"
        if path.exists():
            sidecars.append({"role": role, "file": path.name, "size_bytes": path.stat().st_size})
    return sidecars


def _group_by_stage(analyses: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for record in analyses:
        stage = str(record.get("stage", "unknown"))
        density_kind = str(record.get("density_kind", "unknown"))
        grouped.setdefault(stage, {})[density_kind] = record
    return grouped


def _build_summary(
    analyses: Sequence[Dict[str, Any]],
    formations: Sequence[Dict[str, Any]],
    sidecars: Sequence[Dict[str, Any]],
    dipoles: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    stages = []
    density_kinds = []
    stage_to_kinds: Dict[str, set[str]] = {}
    for record in analyses:
        stage = record.get("stage")
        kind = record.get("density_kind")
        if stage and stage not in stages:
            stages.append(stage)
        if kind and kind not in density_kinds:
            density_kinds.append(kind)
        if stage and kind:
            stage_to_kinds.setdefault(str(stage), set()).add(str(kind))
    required_triple = {"scf", "mp2_unrelaxed", "mp2_relaxed"}
    return {
        "analysis_count": len(analyses),
        "mp2_density_formation_count": len(formations),
        "density_dipole_count": len(dipoles),
        "stages": stages,
        "density_kinds": density_kinds,
        "sidecar_file_count": len(sidecars),
        "has_initial_and_final_triples": (
            required_triple.issubset(stage_to_kinds.get("initial", set()))
            and required_triple.issubset(stage_to_kinds.get("final", set()))
        ),
    }


def _format_number(value: Any, spec: str = ".6g") -> str:
    if isinstance(value, (int, float)):
        return format(value, spec)
    if value is None or value == "":
        return "-"
    return str(value)


def _render_density_analysis_blocks(data, helpers, render_options):  # noqa: ANN001
    del render_options
    density = data.get("density_analysis") or {}
    analyses = density.get("analyses") or []
    formations = density.get("mp2_density_formations") or []
    dipoles = density.get("dipoles") or []
    sidecars = density.get("sidecar_files") or []
    if not analyses and not formations and not dipoles:
        return []

    lines: List[str] = []
    summary = density.get("summary") or {}
    bits = [
        f"**Density analyses:** {summary.get('analysis_count', len(analyses))}",
        f"**MP2 density formations:** {summary.get('mp2_density_formation_count', len(formations))}",
    ]
    if sidecars:
        bits.append("**Sidecars:** " + ", ".join(f"`{item.get('file')}`" for item in sidecars))
    lines.append("  ".join(bits))

    if analyses:
        rows = [("stage", "cycle", "density", "file", "Mulliken sum", "Loewdin sum", "Hirshfeld alpha/beta")]
        for record in analyses:
            pop = record.get("population_summary") or {}
            hirshfeld = "-"
            if pop.get("hirshfeld_total_alpha_density") is not None or pop.get("hirshfeld_total_beta_density") is not None:
                hirshfeld = (
                    f"{_format_number(pop.get('hirshfeld_total_alpha_density'))}/"
                    f"{_format_number(pop.get('hirshfeld_total_beta_density'))}"
                )
            rows.append((
                str(record.get("stage", "")),
                str(record.get("optimization_cycle") or "-"),
                str(record.get("density_kind", "")),
                f"`{record.get('input_electron_density_file')}`" if record.get("input_electron_density_file") else "-",
                _format_number(pop.get("mulliken_sum_of_charges")),
                _format_number(pop.get("loewdin_sum_of_charges")),
                hirshfeld,
            ))
        lines.append("**Population Contexts**\n" + helpers.make_table(rows))

    if formations:
        rows = [("stage", "cycle", "corr E Eh", "MP2 E Eh", "trace", "N occ", "SCF density", "corr density")]
        for formation in formations:
            rows.append((
                str(formation.get("stage", "")),
                str(formation.get("optimization_cycle") or "-"),
                _format_number(formation.get("ri_mp2_correlation_energy_Eh"), ".10f"),
                _format_number(formation.get("mp2_total_energy_Eh"), ".10f"),
                _format_number(formation.get("density_trace"), ".6f"),
                str(formation.get("natural_occupation_count", "-")),
                f"`{formation.get('scf_density_file')}`" if formation.get("scf_density_file") else "-",
                f"`{formation.get('correlated_density_file')}`" if formation.get("correlated_density_file") else "-",
            ))
        lines.append("**MP2 Density Formation**\n" + helpers.make_table(rows))

    if dipoles:
        rows = [("stage", "density", "method", "level", "|mu| D", "mu vector au")]
        for dipole in dipoles:
            vec = dipole.get("total_dipole_au") or {}
            vector = "-"
            if vec:
                vector = f"({_format_number(vec.get('x'))}, {_format_number(vec.get('y'))}, {_format_number(vec.get('z'))})"
            rows.append((
                str(dipole.get("stage", "")),
                str(dipole.get("density_kind", "")),
                str(dipole.get("method", "")),
                str(dipole.get("level", "-")),
                _format_number(dipole.get("magnitude_Debye"), ".6f"),
                vector,
            ))
        lines.append("**Density Dipoles**\n" + helpers.make_table(rows))

    return [f"{'#' * (helpers.heading_level + 1)} Density-Specific Analyses\n" + "\n\n".join(lines)]


def _write_density_analysis_csv_files(
    data: Dict[str, Any],
    directory: Path,
    stem: str,
    write_csv: Callable[[Path, str, List[Dict[str, Any]], List[str]], Path],
) -> List[Path]:
    density = data.get("density_analysis") or {}
    files: List[Path] = []
    analyses = density.get("analyses") or []
    if analyses:
        rows = []
        for record in analyses:
            pop = record.get("population_summary") or {}
            rows.append({
                "analysis_index": record.get("analysis_index"),
                "stage": record.get("stage"),
                "geometry_stage": record.get("geometry_stage"),
                "optimization_cycle": record.get("optimization_cycle"),
                "density_kind": record.get("density_kind"),
                "method": record.get("method"),
                "level": record.get("level"),
                "input_electron_density_file": record.get("input_electron_density_file"),
                "mulliken_sum_of_charges": pop.get("mulliken_sum_of_charges"),
                "loewdin_sum_of_charges": pop.get("loewdin_sum_of_charges"),
                "hirshfeld_total_alpha_density": pop.get("hirshfeld_total_alpha_density"),
                "hirshfeld_total_beta_density": pop.get("hirshfeld_total_beta_density"),
                "line_start": record.get("line_start"),
                "line_end": record.get("line_end"),
            })
        files.append(write_csv(
            directory,
            f"{stem}_density_analyses.csv",
            rows,
            [
                "analysis_index", "stage", "geometry_stage", "optimization_cycle",
                "density_kind", "method", "level", "input_electron_density_file",
                "mulliken_sum_of_charges", "loewdin_sum_of_charges",
                "hirshfeld_total_alpha_density", "hirshfeld_total_beta_density",
                "line_start", "line_end",
            ],
        ))
        files.extend(_write_density_charge_csv(data, directory, stem, write_csv, "mulliken"))
        files.extend(_write_density_charge_csv(data, directory, stem, write_csv, "loewdin"))

    formations = density.get("mp2_density_formations") or []
    if formations:
        fields = [
            "formation_index", "stage", "geometry_stage", "optimization_cycle",
            "ri_mp2_correlation_energy_Eh", "mp2_total_energy_Eh", "mp2_gradient_norm",
            "density_trace", "natural_occupation_sum", "natural_occupation_count",
            "stored_unrelaxed_density", "stored_relaxed_density",
            "scf_density_file", "correlated_density_file", "energy_weighted_density_file",
            "maximum_memory_MB",
        ]
        files.append(write_csv(
            directory,
            f"{stem}_mp2_density_formations.csv",
            [{field: row.get(field) for field in fields} for row in formations],
            fields,
        ))

    dipoles = density.get("dipoles") or []
    if dipoles:
        fields = [
            "stage", "geometry_stage", "optimization_cycle", "density_kind",
            "method", "level", "magnitude_Debye", "magnitude_au",
            "total_dipole_x_au", "total_dipole_y_au", "total_dipole_z_au",
        ]
        rows = []
        for dipole in dipoles:
            vec = dipole.get("total_dipole_au") or {}
            rows.append({
                "stage": dipole.get("stage"),
                "geometry_stage": dipole.get("geometry_stage"),
                "optimization_cycle": dipole.get("optimization_cycle"),
                "density_kind": dipole.get("density_kind"),
                "method": dipole.get("method"),
                "level": dipole.get("level"),
                "magnitude_Debye": dipole.get("magnitude_Debye"),
                "magnitude_au": dipole.get("magnitude_au"),
                "total_dipole_x_au": vec.get("x"),
                "total_dipole_y_au": vec.get("y"),
                "total_dipole_z_au": vec.get("z"),
            })
        files.append(write_csv(directory, f"{stem}_density_dipoles.csv", rows, fields))
    return files


def _write_density_charge_csv(
    data: Dict[str, Any],
    directory: Path,
    stem: str,
    write_csv: Callable[[Path, str, List[Dict[str, Any]], List[str]], Path],
    section_key: str,
) -> List[Path]:
    rows: List[Dict[str, Any]] = []
    density = data.get("density_analysis") or {}
    for record in density.get("analyses") or []:
        section = ((record.get("population") or {}).get(section_key) or {})
        atoms = section.get("atomic_charges") or []
        for atom in atoms:
            rows.append({
                "analysis_index": record.get("analysis_index"),
                "stage": record.get("stage"),
                "density_kind": record.get("density_kind"),
                "input_electron_density_file": record.get("input_electron_density_file"),
                "atom_index": atom.get("index"),
                "symbol": atom.get("symbol"),
                "charge": atom.get("charge"),
                "spin_population": atom.get("spin_population"),
            })
    if not rows:
        return []
    fields = [
        "analysis_index", "stage", "density_kind", "input_electron_density_file",
        "atom_index", "symbol", "charge", "spin_population",
    ]
    return [write_csv(directory, f"{stem}_density_{section_key}_charges.csv", rows, fields)]


DENSITY_ANALYSIS_MARKDOWN_SECTION_PLUGIN = MarkdownSectionPlugin(
    key="density_analysis",
    order=34,
    render_molecule_blocks=_render_density_analysis_blocks,
)

DENSITY_ANALYSIS_CSV_SECTION_PLUGIN = CSVSectionPlugin(
    key="density_analysis",
    order=34,
    render_files=_write_density_analysis_csv_files,
)

PLUGIN_BUNDLE = PluginBundle(
    metadata=PluginMetadata(
        key="density_analysis",
        name="Density-Specific Analyses",
        short_help=(
            "Built-in parser for SCF/MP2 relaxed and unrelaxed density "
            "population analyses in double-hybrid and MP2 jobs."
        ),
        description=(
            "Preserves repeated ORCA population/NBO analyses by density "
            "context while delegating Mulliken, Loewdin, Mayer, Hirshfeld, "
            "MBIS, CHELPG, and NBO grammars to their owner modules."
        ),
        docs_path="README.md",
        examples=(
            "orca_parser double_hybrid_opt.out --sections density_analysis --markdown --csv",
            "orca_parser double_hybrid_opt.out --sections opt --markdown --csv",
        ),
    ),
    parser_sections=(
        ParserSectionPlugin("density_analysis", DensityAnalysisModule),
    ),
    parser_aliases=(
        ParserSectionAlias(name="density_analysis", section_keys=("density_analysis",)),
        ParserSectionAlias(name="densities", section_keys=("density_analysis",)),
        ParserSectionAlias(name="double_hybrid", section_keys=("geom_opt", "density_analysis")),
    ),
    markdown_sections=(
        DENSITY_ANALYSIS_MARKDOWN_SECTION_PLUGIN,
    ),
    csv_sections=(
        DENSITY_ANALYSIS_CSV_SECTION_PLUGIN,
    ),
)
