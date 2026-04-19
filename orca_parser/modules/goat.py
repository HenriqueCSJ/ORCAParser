"""
GOAT conformer-search parser module.

Extracts the GOAT final ensemble table, global-minimum markers, ensemble
thermochemistry, and the xyz filenames printed near the end of the run.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..job_family_registry import CalculationFamilyPlugin
from ..job_series import get_goat_series
from ..parser_section_plugin import ParserSectionAlias, ParserSectionPlugin
from ..plugin_bundle import PluginBundle, PluginMetadata
from ..render_options import RenderOptions
from .base import BaseModule


_GLOBAL_MINIMUM_RE = re.compile(r"Global minimum found!", re.I)
_WRITE_STRUCTURE_RE = re.compile(r"Writing structure to\s+(\S+)", re.I)
_ENSEMBLE_HEADER_RE = re.compile(r"#\s*Final ensemble info\s*#", re.I)
_ENSEMBLE_ROW_RE = re.compile(
    r"^\s*(\d+)\s+(-?\d+(?:\.\d+)?)\s+(\d+)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*$"
)
_BELOW_THRESHOLD_RE = re.compile(
    r"Conformers below\s+(-?\d+(?:\.\d+)?)\s+kcal/mol:\s+(\d+)",
    re.I,
)
_LOWEST_ENERGY_RE = re.compile(
    r"Lowest energy conformer\s*:\s*(-?\d+(?:\.\d+)?)\s+Eh",
    re.I,
)
_SCONF_RE = re.compile(
    r"Sconf at\s+(-?\d+(?:\.\d+)?)\s+K\s*:\s*(-?\d+(?:\.\d+)?)\s+cal/\(molK\)",
    re.I,
)
_GCONF_RE = re.compile(
    r"Gconf at\s+(-?\d+(?:\.\d+)?)\s+K\s*:\s*(-?\d+(?:\.\d+)?)\s+kcal/mol",
    re.I,
)
_FINAL_ENSEMBLE_RE = re.compile(r"Writing final ensemble to\s+(\S+)", re.I)


def _parse_ensemble_rows(lines: List[str], start: int) -> List[Dict[str, Any]]:
    """Parse GOAT final-ensemble rows starting just below the header block."""
    rows: List[Dict[str, Any]] = []
    started = False

    for line in lines[start:]:
        match = _ENSEMBLE_ROW_RE.match(line)
        if match:
            started = True
            rows.append({
                "conformer": int(match.group(1)),
                "relative_energy_kcal_mol": float(match.group(2)),
                "degeneracy": int(match.group(3)),
                "percent_total": float(match.group(4)),
                "percent_cumulative": float(match.group(5)),
            })
            continue

        if started:
            if not line.strip():
                break
            if _BELOW_THRESHOLD_RE.search(line):
                break

    return rows


def _matches_goat(
    meta: Dict[str, Any],
    data: Dict[str, Any],
    context: Dict[str, Any],
    deltascf: Dict[str, Any],
    excited_state_optimization: Dict[str, Any],
) -> bool:
    """Match GOAT jobs for normalized family behavior.

    Keeping this matcher in the module means the parser logic and the family
    classification evolve together instead of teaching a central registry about
    GOAT in a second place.
    """

    del deltascf, excited_state_optimization
    calc_type = str(meta.get("calculation_type", "")).strip().lower()
    return bool(data.get("goat") or context.get("is_goat") or "goat" in calc_type)


def _render_goat_markdown_sections(
    data: Dict[str, Any],
    format_number: Callable[[Any, str], str],
    make_table: Callable[[List[tuple]], str],
    render_options: RenderOptions,
) -> List[tuple[str, str]]:
    """Render GOAT markdown output via the family plugin hook."""

    from ..output.markdown_sections_basic import render_goat_section

    goat = get_goat_series(data)
    if not goat:
        return []

    body = render_goat_section(
        goat,
        format_number=format_number,
        make_table=make_table,
        max_relative_energy_kcal_mol=render_options.goat_max_relative_energy_kcal_mol,
    )
    return [("GOAT Conformer Search", body)] if body else []


def _write_goat_csv_sections(
    data: Dict[str, Any],
    directory: Path,
    stem: str,
    write_csv: Callable[[Path, str, List[Dict[str, Any]], List[str]], Path],
) -> List[Path]:
    """Write GOAT CSV output via the family plugin hook."""

    from ..output.csv_sections_basic import write_goat_section

    return write_goat_section(
        data,
        directory,
        stem,
        write_csv=write_csv,
    )


class GOATModule(BaseModule):
    """Parse GOAT conformer-search summary data."""

    name = "goat"

    def parse(self, lines: List[str]) -> Optional[Dict[str, Any]]:
        if not self.context.get("is_goat") and not any(_ENSEMBLE_HEADER_RE.search(line) for line in lines):
            return None

        data: Dict[str, Any] = {
            "global_minimum_found": False,
            "ensemble": [],
        }

        global_minimum_idx = self.find_last_line(lines, "Global minimum found!")
        if global_minimum_idx != -1:
            data["global_minimum_found"] = True
            for line in lines[global_minimum_idx + 1: min(global_minimum_idx + 8, len(lines))]:
                match = _WRITE_STRUCTURE_RE.search(line)
                if match:
                    data["global_minimum_xyz_file"] = match.group(1)
                    break

        ensemble_idx = self.find_last_line(lines, "# Final ensemble info #")
        if ensemble_idx != -1:
            ensemble = _parse_ensemble_rows(lines, ensemble_idx + 1)
            if ensemble:
                data["ensemble"] = ensemble
                data["n_conformers"] = len(ensemble)
                data["global_minimum_conformer"] = ensemble[0]["conformer"]
                data["top_population_percent"] = ensemble[0]["percent_total"]
                data["max_relative_energy_kcal_mol"] = ensemble[-1]["relative_energy_kcal_mol"]
                data["final_cumulative_percent"] = ensemble[-1]["percent_cumulative"]

        for line in lines[ensemble_idx + 1 if ensemble_idx != -1 else 0:]:
            match = _BELOW_THRESHOLD_RE.search(line)
            if match:
                data["conformer_energy_window_kcal_mol"] = float(match.group(1))
                data["conformers_below_energy_window"] = int(match.group(2))
                break

        lowest_energy_idx = self.find_last_line(lines, "Lowest energy conformer")
        if lowest_energy_idx != -1:
            match = _LOWEST_ENERGY_RE.search(lines[lowest_energy_idx])
            if match:
                data["lowest_energy_conformer_Eh"] = float(match.group(1))

        sconf_idx = self.find_last_line(lines, "Sconf at")
        if sconf_idx != -1:
            match = _SCONF_RE.search(lines[sconf_idx])
            if match:
                data["temperature_K"] = float(match.group(1))
                data["sconf_cal_molK"] = float(match.group(2))

        gconf_idx = self.find_last_line(lines, "Gconf at")
        if gconf_idx != -1:
            match = _GCONF_RE.search(lines[gconf_idx])
            if match:
                if "temperature_K" not in data:
                    data["temperature_K"] = float(match.group(1))
                data["gconf_kcal_mol"] = float(match.group(2))

        final_ensemble_idx = self.find_last_line(lines, "Writing final ensemble to")
        if final_ensemble_idx != -1:
            match = _FINAL_ENSEMBLE_RE.search(lines[final_ensemble_idx])
            if match:
                data["final_ensemble_xyz_file"] = match.group(1)

        if not data.get("ensemble") and not data.get("global_minimum_found"):
            return None

        return data


PLUGIN_BUNDLE = PluginBundle(
    metadata=PluginMetadata(
        key="goat",
        name="GOAT Conformer Search",
        short_help="Built-in GOAT parser family with ensemble markdown/CSV hooks.",
        description=(
            "Self-registering built-in GOAT parser module. Owns the parser "
            "section, alias, normalized calculation-family behavior, and GOAT "
            "family-specific markdown/CSV output hooks."
        ),
        docs_path="README.md",
        examples=(
            "orca_parser conformers.out --sections goat --markdown --csv",
            "orca_parser conformers.out --markdown --goat-max-relative-energy-kcal 5",
        ),
    ),
    parser_sections=(
        ParserSectionPlugin("goat", GOATModule),
    ),
    parser_aliases=(
        ParserSectionAlias(name="goat", section_keys=("goat",)),
    ),
    calculation_families=(
        CalculationFamilyPlugin(
            family="goat",
            default_calculation_label="GOAT Conformer Search",
            matcher=_matches_goat,
            render_markdown_sections=_render_goat_markdown_sections,
            csv_writers=(_write_goat_csv_sections,),
        ),
    ),
)
