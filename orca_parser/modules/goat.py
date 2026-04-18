"""
GOAT conformer-search parser module.

Extracts the GOAT final ensemble table, global-minimum markers, ensemble
thermochemistry, and the xyz filenames printed near the end of the run.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

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
