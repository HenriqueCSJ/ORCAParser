"""CASSCF and CASSCF-owned NEVPT2 parser module.

The CASSCF output is more history-like than most ORCA sections: the macro
iterations, orbital rotations, active-space occupations, and state-averaged
roots are often more important than a single final number.  This module keeps
those histories in JSON-safe structures so JSON, CSV, HDF5, and markdown can
all expose the same parse-time authority.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from ..job_family_registry import CalculationFamilyPlugin
from ..output.csv_section_registry import CSVSectionPlugin
from ..output.markdown_section_registry import (
    MarkdownRenderHelpers,
    MarkdownSectionPlugin,
)
from ..parser_section_plugin import ParserSectionAlias, ParserSectionPlugin
from ..plugin_bundle import PluginBundle, PluginMetadata, PluginOption
from ..render_options import RenderOptions
from .base import BaseModule


_FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
_SETTING_RE = re.compile(r"^\s*(?P<label>.+?)\s+\.{3,}\s*(?P<value>.+?)\s*$")
_ENERGY_RE = re.compile(
    rf"E\(CAS\)=\s*(?P<energy>{_FLOAT_RE})\s+Eh\s+DE=\s*(?P<delta>{_FLOAT_RE})",
    re.I,
)
_GAP_RE = re.compile(
    rf"Energy gap subspaces:\s*Ext-Act\s*=\s*(?P<ext>{_FLOAT_RE})"
    rf"\s+Act-Int\s*=\s*(?P<act>{_FLOAT_RE})",
    re.I,
)
_N_OCC_RE = re.compile(rf"N\(occ\)=\s*(?P<values>(?:{_FLOAT_RE}\s*)+)", re.I)
_GRAD_RE = re.compile(
    rf"\|\|g\|\|\s*=\s*(?P<norm>{_FLOAT_RE})\s+Max\(G\)=\s*(?P<max>{_FLOAT_RE})"
    r"\s+Rot=(?P<rot>\d+\s*,\s*\d+)",
    re.I,
)
_OPTION_GRAD_RE = re.compile(
    rf"Option=(?P<option>[A-Za-z0-9_-]+):\s*\|\|g\|\|\s*=\s*(?P<norm>{_FLOAT_RE})",
    re.I,
)
_OPTION_PERCENT_RE = re.compile(rf"=\s*(?P<percent>{_FLOAT_RE})\s*%")
_ORBITAL_UPDATE_RE = re.compile(r"Orbital Update\s*\[\s*(?P<method>[^\]]+?)\s*\]", re.I)
_SUPERCI_RE = re.compile(
    rf"SUPERCI-ITER\s+(?P<iteration>\d+):\s*DE=\s*(?P<delta>{_FLOAT_RE})"
    rf"\s+<r\|r>=\s*(?P<residual>{_FLOAT_RE})",
    re.I,
)
_SX_RE = re.compile(
    rf"<Psi\(SX\)\|Psi\(SX\)>=\s*(?P<overlap>{_FLOAT_RE})\s+DE\(SX\)="
    rf"(?P<delta>{_FLOAT_RE})",
    re.I,
)
_LARGEST_COEFF_RE = re.compile(
    rf"Largest coefficient .*=\s*(?P<value>{_FLOAT_RE})",
    re.I,
)
_DENSITY_BUILD_RE = re.compile(
    rf"Density\s+(?P<status>close to|off from)\s+convergence\s+\((?P<value>{_FLOAT_RE})\)\."
    r"\s+(?P<build>Exact|Approximate)\s+active Fock build",
    re.I,
)
_INACTIVE_RE = re.compile(rf"Inactive Energy E0\s*=\s*(?P<energy>{_FLOAT_RE})\s+Eh", re.I)
_FINAL_ENERGY_RE = re.compile(
    rf"Final CASSCF energy\s*:\s*(?P<energy>{_FLOAT_RE})\s+Eh\s+(?P<ev>{_FLOAT_RE})\s+eV",
    re.I,
)
_ORBITAL_RE = re.compile(
    rf"^\s*(?P<index>\d+)\s+(?P<occ>{_FLOAT_RE})\s+(?P<eh>{_FLOAT_RE})\s+(?P<ev>{_FLOAT_RE})(?:\s+(?P<irrep>\S+))?\s*$"
)
_STATE_BLOCK_RE = re.compile(
    r"^\s*CAS-SCF STATES FOR BLOCK\s+(?P<block>\d+)\s+MULT=\s*(?P<mult>\d+)"
    r"\s+NROOTS=\s*(?P<nroots>\d+)",
    re.I,
)
_ROOT_RE = re.compile(
    rf"^\s*ROOT\s+(?P<root>\d+):\s+E=\s*(?P<energy>{_FLOAT_RE})\s+Eh"
    rf"(?:\s+(?P<ev>{_FLOAT_RE})\s+eV\s+(?P<cm>{_FLOAT_RE})\s+cm\*\*-1)?",
    re.I,
)
_CONFIG_RE = re.compile(
    rf"^\s*(?P<weight>{_FLOAT_RE})\s+\[\s*(?P<index>\d+)\s*\]:\s*(?P<occupation>[0-9A-Za-z]+)\s*$"
)
_LOWEST_ROOT_RE = re.compile(
    rf"LOWEST ROOT\s*\(ROOT\s*(?P<root>\d+)\s*,?\s*MULT\s*(?P<mult>\d+)\)\s*=\s*"
    rf"(?P<energy>{_FLOAT_RE})\s+Eh\s+(?P<ev>{_FLOAT_RE})\s+eV",
    re.I,
)
_TRANSITION_ROW_RE = re.compile(
    rf"^\s*(?P<state>\d+):\s+(?P<root>\d+)\s+(?P<mult>\d+)\s+"
    rf"(?P<de_au>{_FLOAT_RE})\s+(?P<de_ev>{_FLOAT_RE})\s+(?P<de_cm>{_FLOAT_RE})\s*$"
)
_MATRIX_HEADER_RE = re.compile(r"^\s*\d+(?:\s+\d+)*\s*$")
_MATRIX_ROW_RE = re.compile(rf"^\s*(?P<row>\d+)\s+(?P<values>(?:{_FLOAT_RE}\s*)+)$")
_TRACE_RE = re.compile(rf"Trace of the (?P<kind>electron|spin) density:\s*(?P<trace>{_FLOAT_RE})", re.I)
_ENERGY_COMPONENT_RE = re.compile(
    rf"^\s*(?P<label>[A-Za-z][A-Za-z0-9 /().-]*?)\s*:\s*(?P<eh>{_FLOAT_RE})"
    rf"(?:\s+Eh)?(?:\s+(?P<ev>{_FLOAT_RE})\s+eV)?\s*$"
)
_ENERGY_TOTAL_RE = re.compile(rf"^\s*(?P<energy>{_FLOAT_RE})\s*$")
_ORBITAL_RANGE_RE = re.compile(
    r"^\s*(?P<space>Internal|Active|External)\s+(?P<first>\d+)\s*-\s*(?P<last>\d+)"
    r"\s+\(\s*(?P<count>\d+)\s+orbitals?\)",
    re.I,
)
_CI_BLOCK_RE = re.compile(r"^\s*BLOCK\s+(?P<block>\d+)\s+WEIGHT=\s*(?P<weight>[-+]?\d*\.?\d+)", re.I)
_CI_ROOT_WEIGHT_RE = re.compile(r"^\s*ROOT=(?P<root>\d+)\s+WEIGHT=\s*(?P<weight>[-+]?\d*\.?\d+)", re.I)
_COMPOSITION_HEADER_RE = re.compile(r"^\s*(?P<indices>\d+(?:\s+\d+)*)\s*$")
_COMPOSITION_ROW_RE = re.compile(
    rf"^\s*(?P<atom>\d+)\s+(?P<element>[A-Z][a-z]?)\s+(?P<orbital>\S+)\s+"
    rf"(?P<values>(?:{_FLOAT_RE}\s*)+)$"
)
_NEVPT_ROOT_RE = re.compile(r"^\s*MULT\s+(?P<mult>\d+),\s*ROOT\s+(?P<root>\d+)", re.I)
_NEVPT_CLASS_RE = re.compile(rf"^\s*Class\s+(?P<class>\S+)\s*:\s*dE\s*=\s*(?P<energy>{_FLOAT_RE})", re.I)
_NEVPT_TOTAL_CORR_RE = re.compile(rf"Total Energy Correction\s*:\s*dE\s*=\s*(?P<energy>{_FLOAT_RE})", re.I)
_NEVPT_REF_RE = re.compile(rf"(?:Reference|Zero Order)\s+Energy\s*:\s*E0\s*=\s*(?P<energy>{_FLOAT_RE})", re.I)
_NEVPT_TOTAL_RE = re.compile(rf"Total Energy \(E0\+dE\)\s*:\s*E\s*=\s*(?P<energy>{_FLOAT_RE})", re.I)
_TOTAL_ENERGY_ROW_RE = re.compile(
    rf"^\s*(?P<state>\d+):\s+(?P<root>\d+)\s+(?P<mult>\d+)\s+(?P<energy>{_FLOAT_RE})"
    rf"\s+EDIAG\[(?P<ediag>\d+)\]\s+(?P<ediag_value>{_FLOAT_RE})"
)
_TIMING_RE = re.compile(
    rf"^\s*(?P<label>[A-Za-z][A-Za-z0-9 ,()/-]+?)\s+\.{{3,}}\s+"
    rf"(?P<seconds>{_FLOAT_RE})\s+sec(?:\s+\(\s*(?P<percent>{_FLOAT_RE})%\))?",
    re.I,
)
_SPECTRUM_ROW_RE = re.compile(
    rf"^\s*(?P<from>\S+)\s*->\s*(?P<to>\S+)\s+"
    rf"(?P<energy_ev>{_FLOAT_RE})\s+(?P<energy_cm>{_FLOAT_RE})\s+"
    rf"(?P<wavelength_nm>{_FLOAT_RE})\s+(?P<a>{_FLOAT_RE})\s+"
    rf"(?P<b>{_FLOAT_RE})\s+(?P<c>{_FLOAT_RE})\s+(?P<d>{_FLOAT_RE})"
    rf"(?:\s+(?P<e>{_FLOAT_RE}))?\s*$"
)
_TRANSITION_LABEL_RE = re.compile(
    r"^(?P<root>\d+)-(?P<mult>\d+(?:\.\d+)?)(?P<irrep>[A-Za-z]*)$"
)
_NATORB_BLOCK_RE = re.compile(r"^\s*BLOCK\s+(?P<block>\d+)\s+\(Multiplicity\s+(?P<mult>\d+)\):", re.I)
_NATORB_ROOT_RE = re.compile(r"^\s*Root\s+(?P<root>\d+)\s*:", re.I)
_NATORB_STORED_RE = re.compile(r"-+>\s*stored as\s*(?P<path>\S+)", re.I)
_QDPT_HEADING_RE = re.compile(r"^\s*QDPT WITH (?P<label>.+?)\s*$", re.I)
_QDPT_LEVEL_RE = re.compile(
    rf"^\s*(?P<state>\d+):\s+(?P<energy_cm>{_FLOAT_RE})\s+"
    rf"(?P<energy_ev>{_FLOAT_RE})\s+(?P<population>{_FLOAT_RE})\s*$"
)
_QDPT_EIGENVECTOR_STATE_RE = re.compile(rf"^\s*STATE\s+(?P<state>\d+):\s+(?P<energy_cm>{_FLOAT_RE})\s*$")
_QDPT_EIGENVECTOR_COMPONENT_RE = re.compile(
    rf"^\s*(?P<weight>{_FLOAT_RE})\s+(?P<real>{_FLOAT_RE})\s+(?P<imag>{_FLOAT_RE})"
    r"\s*:\s*(?P<block>\d+)\s+(?P<root>\d+)\s+(?P<spin>\S+)\s+(?P<ms>\S+)\s*$"
)
_GFACTORS_RE = re.compile(
    rf"^\s*(?P<x>{_FLOAT_RE})\s+(?P<y>{_FLOAT_RE})\s+(?P<z>{_FLOAT_RE})\s+iso\s*=\s*(?P<iso>{_FLOAT_RE})"
)
_D_VALUE_RE = re.compile(rf"^\s*D\s*=\s*(?P<d>{_FLOAT_RE})\s*cm-1", re.I)
_E_OVER_D_RE = re.compile(rf"^\s*E/D\s*=\s*(?P<e_over_d>{_FLOAT_RE})", re.I)


def _to_float(value: str | None) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: str | None) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _numbers(text: str) -> List[float]:
    return [float(value) for value in re.findall(_FLOAT_RE, text)]


def _normalize_key(label: str) -> str:
    key = label.strip().lower()
    key = key.replace("#", "number")
    key = key.replace("**", "2")
    key = re.sub(r"[^a-z0-9]+", "_", key)
    return key.strip("_")


def _coerce_value(raw: str) -> Any:
    value = raw.strip()
    if not value:
        return ""

    percent_match = re.match(rf"^(?P<number>{_FLOAT_RE})\s*%\s+of\s+\|\|g\|\|$", value, re.I)
    if percent_match:
        return {
            "value": float(percent_match.group("number")),
            "unit": "percent_of_gradient_norm",
            "raw": value,
        }

    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    if re.fullmatch(_FLOAT_RE, value):
        return float(value)
    return value


def _parse_setting_line(line: str) -> Optional[tuple[str, Any, str]]:
    match = _SETTING_RE.match(line)
    if not match:
        return None
    label = match.group("label").strip()
    raw_value = match.group("value").strip()
    return _normalize_key(label), _coerce_value(raw_value), raw_value


def _find_exact(lines: Sequence[str], heading: str, start: int = 0) -> int:
    target = heading.strip().lower()
    for index in range(start, len(lines)):
        if lines[index].strip().lower() == target:
            return index
    return -1


def _find_last_exact(lines: Sequence[str], heading: str, start: int = 0) -> int:
    target = heading.strip().lower()
    for index in range(len(lines) - 1, start - 1, -1):
        if lines[index].strip().lower() == target:
            return index
    return -1


def _find_next_exact(lines: Sequence[str], headings: Iterable[str], start: int) -> int:
    targets = {heading.strip().lower() for heading in headings}
    for index in range(start, len(lines)):
        if lines[index].strip().lower() in targets:
            return index
    return len(lines)


def _find_line_containing(lines: Sequence[str], text: str, start: int = 0) -> int:
    target = text.lower()
    for index in range(start, len(lines)):
        if target in lines[index].lower():
            return index
    return -1


def _is_dashed(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and set(stripped) <= {"-"}


def _matrix_to_rows(matrix: Dict[str, Any]) -> List[Dict[str, Any]]:
    columns = matrix.get("columns") or []
    rows = []
    for row_label, values in zip(matrix.get("rows") or [], matrix.get("matrix") or []):
        row: Dict[str, Any] = {"row": row_label}
        for column, value in zip(columns, values):
            row[f"col_{column}"] = value
        rows.append(row)
    return rows


class CASSCFModule(BaseModule):
    """Parse CASSCF active-space, convergence, state, and NEVPT2 output."""

    name = "casscf"

    def parse(self, lines: List[str]) -> Optional[Dict[str, Any]]:
        """Return the full CASSCF/NEVPT2 payload when this output contains CASSCF data."""
        if not self._looks_like_casscf(lines):
            return None

        data: Dict[str, Any] = {}

        input_block = self._parse_input_block()
        if input_block:
            data["input"] = input_block

        setup = self._parse_setup(lines)
        if setup:
            data.update(setup)

        convergence = self._parse_convergence(lines)
        if convergence:
            data["convergence"] = convergence

        results = self._parse_results(lines)
        if results:
            data["results"] = results

        orbital_energies = self._parse_orbital_energies(lines)
        if orbital_energies:
            data["orbital_energies"] = orbital_energies

        state_blocks = self._parse_state_blocks(lines)
        if state_blocks:
            data["state_blocks"] = state_blocks
            data["states"] = [
                {
                    **state,
                    "block_index": block.get("block_index"),
                    "multiplicity": block.get("multiplicity"),
                }
                for block in state_blocks
                for state in block.get("roots", [])
            ]

        transitions = self._parse_transition_energies(lines, "SA-CASSCF TRANSITION ENERGIES")
        if transitions:
            data["transition_energies"] = transitions

        density = self._parse_matrix(lines, "DENSITY MATRIX", "electron")
        if density:
            data["density_matrix"] = density

        spin_density = self._parse_matrix(lines, "SPIN-DENSITY MATRIX", "spin")
        if spin_density:
            data["spin_density_matrix"] = spin_density

        components = self._parse_energy_components(lines)
        if components:
            data["energy_components"] = components

        active_indices = self._active_orbital_indices(data)
        orbital_energy_window = self._orbital_energy_window(data, active_indices)
        if orbital_energy_window:
            data["orbital_energy_window"] = orbital_energy_window

        loewdin_active = self._parse_loewdin_compositions(
            lines,
            "LOEWDIN REDUCED ACTIVE MOs",
            selected_indices=None,
            contribution_threshold=0.0,
            stop_at=("ORCA POPULATION ANALYSIS",),
        )
        if loewdin_active:
            data["loewdin_reduced_active_mos"] = loewdin_active

        selected_window = self._orbital_composition_window(data, active_indices)
        if selected_window:
            loewdin_window = self._parse_loewdin_compositions(
                lines,
                "LOEWDIN ORBITAL-COMPOSITIONS",
                selected_indices=selected_window,
                contribution_threshold=0.0,
                stop_at=("LOEWDIN REDUCED ACTIVE MOs",),
            )
            if loewdin_window:
                loewdin_window["selected_min_index"] = min(selected_window)
                loewdin_window["selected_max_index"] = max(selected_window)
                data["loewdin_orbital_compositions"] = loewdin_window

        nevpt2 = self._parse_nevpt2(lines)
        if nevpt2:
            data["nevpt2"] = nevpt2

        spectra = self._parse_spectra(lines)
        if spectra:
            data["spectra"] = spectra

        relativistic = self._parse_relativistic_properties(lines)
        if relativistic:
            data["relativistic"] = relativistic

        raw_report_sections = self._parse_raw_report_sections(lines)
        if raw_report_sections:
            data["raw_report_sections"] = raw_report_sections

        self._attach_state_assignments(data)

        summary = self._build_summary(data)
        if summary:
            data["summary"] = summary
            self.context["casscf_summary"] = summary
            if summary.get("method_label"):
                self.context["level_of_theory"] = summary["method_label"]

        return data or None

    def _looks_like_casscf(self, lines: Sequence[str]) -> bool:
        """Detect CASSCF from normalized context, echoed input, or printed CASSCF headers."""
        if str(self.context.get("reference_type", "")).upper() == "CASSCF":
            return True
        input_blocks = {
            str(name).lower()
            for name in (self.context.get("input_echo") or {}).get("block_names", [])
        }
        if "casscf" in input_blocks:
            return True
        for line in lines[:5000]:
            if "ORCA-CASSCF" in line or "CAS-SCF ITERATIONS" in line:
                return True
        return False

    def _parse_input_block(self) -> Dict[str, Any]:
        """Return the echoed ``%casscf`` input block settings when available."""
        block = ((self.context.get("input_echo") or {}).get("blocks") or {}).get("casscf")
        if not isinstance(block, dict):
            return {}
        settings = dict(block.get("settings") or {})
        parsed: Dict[str, Any] = {
            "raw_lines": list(block.get("raw_lines") or []),
            "settings": settings,
        }
        for key in ("nel", "norb", "mult", "nroots", "ptmethod"):
            if key in settings:
                parsed[key] = settings[key]
        return parsed

    def _parse_setup(self, lines: Sequence[str]) -> Dict[str, Any]:
        """Parse pre-iteration active-space, CI, SCF, PT2, and orbital setup sections."""
        setup: Dict[str, Any] = {}

        memory: Dict[str, float] = {}
        for line in lines:
            if "CAS-SCF ITERATIONS" in line:
                break
            match = re.search(rf"CASSCF \(estimated\) memory needed\s+\.{{3,}}\s*({_FLOAT_RE})\s+MB", line, re.I)
            if match:
                memory["casscf_estimated_MB"] = float(match.group(1))
                continue
            match = re.search(rf"NEVPT2 \(estimated\) memory needed\s+\.{{3,}}\s*({_FLOAT_RE})\s+MB", line, re.I)
            if match:
                memory["nevpt2_estimated_MB"] = float(match.group(1))
        if memory:
            setup["memory"] = memory

        system = self._parse_system_settings(lines)
        if system:
            setup["system"] = system

        ci_step = self._parse_ci_step(lines)
        if ci_step:
            setup["ci_step"] = ci_step

        section_specs = (
            ("orbital_improvement", "ORBITAL-IMPROVEMENT-STEP:", ("SCF-SETTINGS:",)),
            ("scf_settings", "SCF-SETTINGS:", ("PT2-SETTINGS:", "FINAL ORBITALS:", "CAS-SCF ITERATIONS")),
            ("pt2_settings", "PT2-SETTINGS:", ("FINAL ORBITALS:", "CAS-SCF ITERATIONS")),
            ("final_orbitals", "FINAL ORBITALS:", ("CAS-SCF ITERATIONS",)),
        )
        for key, heading, end_headings in section_specs:
            settings = self._parse_settings_section(lines, heading, end_headings)
            if key == "pt2_settings":
                start = _find_exact(lines, heading)
                end = _find_next_exact(lines, end_headings, start + 1) if start != -1 else start
                if start != -1:
                    for line in lines[start + 1:end]:
                        method_match = re.search(r"PT2\s*=\s*([A-Za-z0-9_-]+)", line, re.I)
                        if method_match:
                            settings["pt2_method"] = method_match.group(1)
                            break
            if settings:
                setup[key] = settings

        return setup

    def _parse_system_settings(self, lines: Sequence[str]) -> Dict[str, Any]:
        """Parse active electron/orbital counts and internal/active/external ranges."""
        start = _find_exact(lines, "SYSTEM-SPECIFIC SETTINGS:")
        if start == -1:
            return {}
        end = _find_next_exact(lines, ("CI-STEP:",), start + 1)
        system: Dict[str, Any] = {}
        ranges: List[Dict[str, Any]] = []

        for line in lines[start + 1:end]:
            range_match = _ORBITAL_RANGE_RE.match(line)
            if range_match:
                ranges.append(
                    {
                        "space": range_match.group("space").lower(),
                        "first_index": int(range_match.group("first")),
                        "last_index": int(range_match.group("last")),
                        "n_orbitals": int(range_match.group("count")),
                    }
                )
                continue

            parsed = _parse_setting_line(line)
            if parsed:
                key, value, raw = parsed
                system[key] = value
                system[f"{key}_raw"] = raw

        if ranges:
            system["orbital_ranges"] = ranges
            active = next((item for item in ranges if item["space"] == "active"), None)
            if active:
                system["active_orbital_range"] = active
        return system

    def _parse_ci_step(self, lines: Sequence[str]) -> Dict[str, Any]:
        """Parse multiplicity blocks, roots, weights, and CI convergence settings."""
        start = _find_exact(lines, "CI-STEP:")
        if start == -1:
            return {}
        end = _find_next_exact(lines, ("INTEGRAL-TRANSFORMATION-STEP:",), start + 1)
        ci_step: Dict[str, Any] = {"blocks": []}
        current_block: Optional[Dict[str, Any]] = None

        for line in lines[start + 1:end]:
            block_match = _CI_BLOCK_RE.match(line)
            if block_match:
                current_block = {
                    "block_index": int(block_match.group("block")),
                    "weight": float(block_match.group("weight")),
                    "root_weights": [],
                }
                ci_step["blocks"].append(current_block)
                continue

            root_match = _CI_ROOT_WEIGHT_RE.match(line)
            if root_match and current_block is not None:
                current_block["root_weights"].append(
                    {
                        "root": int(root_match.group("root")),
                        "weight": float(root_match.group("weight")),
                    }
                )
                continue

            parsed = _parse_setting_line(line)
            if not parsed:
                continue
            key, value, raw = parsed
            if current_block is not None and key in {
                "multiplicity",
                "number_configurations",
                "number_csfs",
                "number_roots",
            }:
                current_block[key] = value
                current_block[f"{key}_raw"] = raw
            else:
                ci_step[key] = value
                ci_step[f"{key}_raw"] = raw

        ci_step["n_blocks"] = len(ci_step["blocks"])
        return ci_step if ci_step.get("blocks") or len(ci_step) > 2 else {}

    def _parse_settings_section(
        self,
        lines: Sequence[str],
        heading: str,
        end_headings: Sequence[str],
    ) -> Dict[str, Any]:
        """Parse simple ORCA ``label ... value`` setting blocks into normalized keys."""
        start = _find_exact(lines, heading)
        if start == -1:
            return {}
        end = _find_next_exact(lines, end_headings, start + 1)
        settings: Dict[str, Any] = {}
        for line in lines[start + 1:end]:
            parsed = _parse_setting_line(line)
            if not parsed:
                continue
            key, value, raw = parsed
            settings[key] = value
            settings[f"{key}_raw"] = raw
        return settings

    def _parse_convergence(self, lines: Sequence[str]) -> Dict[str, Any]:
        """Parse every CAS-SCF macro-iteration so convergence evolution is preserved."""
        start = _find_exact(lines, "CAS-SCF ITERATIONS")
        if start == -1:
            return {}
        end = _find_exact(lines, "CASSCF RESULTS", start + 1)
        if end == -1:
            end = len(lines)

        iterations: List[Dict[str, Any]] = []
        current: Optional[Dict[str, Any]] = None
        pending_option: Optional[Dict[str, Any]] = None
        converged_iteration: Optional[int] = None
        pending_final_printing_iteration = False

        for line in lines[start + 1:end]:
            macro_match = re.match(r"^\s*MACRO-ITERATION\s+(\d+):", line, re.I)
            if macro_match:
                if current is not None:
                    iterations.append(current)
                current = {
                    "macro_iteration": int(macro_match.group(1)),
                    "ci_iterations": [],
                    "orbital_options": [],
                    "events": [],
                }
                if pending_final_printing_iteration:
                    current["phase"] = "final_printing"
                    pending_final_printing_iteration = False
                pending_option = None
                continue

            if current is None:
                continue

            ci_match = re.match(r"^\s*CI-ITERATION\s+(\d+):", line, re.I)
            if ci_match:
                current["ci_iterations"].append({"iteration": int(ci_match.group(1))})
                continue

            inactive_match = _INACTIVE_RE.search(line)
            if inactive_match:
                current["inactive_energy_Eh"] = float(inactive_match.group("energy"))
                continue

            if "switching to Step=" in line:
                current["events"].append(line.strip())
                match = re.search(r"Convergence to\s+(\S+)\s+achieved\s+-\s+switching to Step=(\S+)", line, re.I)
                if match:
                    current["switch_gradient_threshold"] = _to_float(match.group(1))
                    current["switch_step_to"] = match.group(2)
                continue

            energy_match = _ENERGY_RE.search(line)
            if energy_match:
                current["energy_Eh"] = float(energy_match.group("energy"))
                current["delta_energy_Eh"] = float(energy_match.group("delta"))
                continue

            gap_match = _GAP_RE.search(line)
            if gap_match:
                current["energy_gap_ext_act_Eh"] = float(gap_match.group("ext"))
                current["energy_gap_act_int_Eh"] = float(gap_match.group("act"))
                continue

            occ_match = _N_OCC_RE.search(line)
            if occ_match:
                current["active_occupations"] = _numbers(occ_match.group("values"))
                continue

            grad_match = _GRAD_RE.search(line)
            if grad_match:
                current["gradient_norm"] = float(grad_match.group("norm"))
                current["max_gradient"] = float(grad_match.group("max"))
                rotations = [
                    int(part.strip())
                    for part in grad_match.group("rot").split(",")
                    if part.strip()
                ]
                current["rotation_indices"] = rotations
                current["rotation_label"] = ",".join(str(part) for part in rotations)
                continue

            option_match = _OPTION_GRAD_RE.search(line)
            if option_match:
                pending_option = {
                    "name": option_match.group("option"),
                    "gradient_norm": float(option_match.group("norm")),
                }
                current["orbital_options"].append(pending_option)
                continue

            if pending_option is not None:
                percent_match = _OPTION_PERCENT_RE.search(line)
                if percent_match:
                    pending_option["percent"] = float(percent_match.group("percent"))
                    pending_option = None
                    continue

            update_match = _ORBITAL_UPDATE_RE.search(line)
            if update_match:
                current["orbital_update"] = update_match.group("method").strip()
                continue

            superci_match = _SUPERCI_RE.search(line)
            if superci_match:
                current.setdefault("superci_iterations", []).append(
                    {
                        "iteration": int(superci_match.group("iteration")),
                        "delta_energy_Eh": float(superci_match.group("delta")),
                        "residual_norm": float(superci_match.group("residual")),
                    }
                )
                continue

            sx_match = _SX_RE.search(line)
            if sx_match:
                current["superci_overlap"] = float(sx_match.group("overlap"))
                current["superci_delta_energy_Eh"] = float(sx_match.group("delta"))
                continue

            coeff_match = _LARGEST_COEFF_RE.search(line)
            if coeff_match:
                current["superci_largest_coefficient"] = float(coeff_match.group("value"))
                continue

            density_match = _DENSITY_BUILD_RE.search(line)
            if density_match:
                current["density_convergence_status"] = density_match.group("status").lower().replace(" ", "_")
                current["density_convergence"] = float(density_match.group("value"))
                current["active_fock_build"] = density_match.group("build").lower()
                continue

            stripped = line.strip()
            if "THE CAS-SCF GRADIENT HAS CONVERGED" in stripped:
                current["gradient_converged"] = True
                converged_iteration = current.get("macro_iteration")
                continue
            if "FINALIZING ORBITALS" in stripped:
                current["finalizing_orbitals"] = True
                continue
            if "DOING ONE FINAL ITERATION FOR PRINTING" in stripped:
                current["final_printing_iteration_requested"] = True
                current["events"].append("DOING ONE FINAL ITERATION FOR PRINTING")
                pending_final_printing_iteration = True
                continue
            if "All densities will be recomputed" in stripped:
                current["all_densities_recomputed"] = True
                continue
            if stripped.startswith("--- Storing electron density"):
                current["stores_root_densities"] = True

        if current is not None:
            iterations.append(current)

        if not iterations:
            return {}

        final_metrics = next(
            (
                iteration
                for iteration in reversed(iterations)
                if "energy_Eh" in iteration
            ),
            iterations[-1],
        )
        return {
            "macro_iterations": iterations,
            "n_macro_iterations": len(iterations),
            "converged": converged_iteration is not None,
            "converged_iteration": converged_iteration,
            "final": dict(final_metrics),
        }

    def _parse_results(self, lines: Sequence[str]) -> Dict[str, Any]:
        """Parse the final CASSCF result banner."""
        start = _find_exact(lines, "CASSCF RESULTS")
        if start == -1:
            return {}
        results: Dict[str, Any] = {}
        for line in lines[start + 1:min(start + 20, len(lines))]:
            match = _FINAL_ENERGY_RE.search(line)
            if match:
                results["final_energy_Eh"] = float(match.group("energy"))
                results["final_energy_eV"] = float(match.group("ev"))
                break
        return results

    def _parse_orbital_energies(self, lines: Sequence[str]) -> List[Dict[str, Any]]:
        """Parse the CASSCF-printed orbital energies immediately after final results."""
        start = _find_exact(lines, "CASSCF RESULTS")
        orbital_header = _find_exact(lines, "ORBITAL ENERGIES", start if start != -1 else 0)
        if orbital_header == -1:
            return []

        orbitals: List[Dict[str, Any]] = []
        for line in lines[orbital_header + 1:]:
            match = _ORBITAL_RE.match(line)
            if match:
                item: Dict[str, Any] = {
                    "index": int(match.group("index")),
                    "occupation": float(match.group("occ")),
                    "energy_Eh": float(match.group("eh")),
                    "energy_eV": float(match.group("ev")),
                }
                if match.group("irrep"):
                    item["irrep"] = match.group("irrep")
                orbitals.append(item)
                continue
            if orbitals and not line.strip():
                break
        return orbitals

    def _parse_state_blocks(self, lines: Sequence[str]) -> List[Dict[str, Any]]:
        """Parse CAS-SCF state blocks, roots, and CI configuration weights."""
        blocks: List[Dict[str, Any]] = []
        current_block: Optional[Dict[str, Any]] = None
        current_root: Optional[Dict[str, Any]] = None

        for line in lines:
            block_match = _STATE_BLOCK_RE.match(line)
            if block_match:
                if current_block is not None:
                    blocks.append(current_block)
                current_block = {
                    "block_index": int(block_match.group("block")),
                    "multiplicity": int(block_match.group("mult")),
                    "nroots": int(block_match.group("nroots")),
                    "roots": [],
                }
                current_root = None
                continue

            if current_block is None:
                continue

            if line.strip() in {
                "SA-CASSCF TRANSITION ENERGIES",
                "DENSITY MATRIX",
                "SPIN-DENSITY MATRIX",
                "ENERGY COMPONENTS",
            }:
                blocks.append(current_block)
                current_block = None
                current_root = None
                continue

            root_match = _ROOT_RE.match(line)
            if root_match:
                current_root = {
                    "root": int(root_match.group("root")),
                    "energy_Eh": float(root_match.group("energy")),
                    "configurations": [],
                }
                if root_match.group("ev") is not None:
                    current_root["relative_energy_eV"] = float(root_match.group("ev"))
                if root_match.group("cm") is not None:
                    current_root["relative_energy_cm-1"] = float(root_match.group("cm"))
                current_block["roots"].append(current_root)
                continue

            config_match = _CONFIG_RE.match(line)
            if config_match and current_root is not None:
                current_root["configurations"].append(
                    {
                        "weight": float(config_match.group("weight")),
                        "configuration_index": int(config_match.group("index")),
                        "occupation_string": config_match.group("occupation"),
                    }
                )

        if current_block is not None:
            blocks.append(current_block)

        for block in blocks:
            for root in block.get("roots", []):
                configs = root.get("configurations", [])
                if configs:
                    dominant = max(configs, key=lambda item: item.get("weight", 0.0))
                    root["dominant_configuration"] = dict(dominant)
                    root["configuration_count"] = len(configs)
        return blocks

    def _parse_transition_energies(self, lines: Sequence[str], heading: str) -> Dict[str, Any]:
        """Parse CASSCF/NEVPT2-style transition tables with state/root/mult labels."""
        start = _find_exact(lines, heading)
        if start == -1:
            return {}
        transitions: Dict[str, Any] = {"states": []}
        for line in lines[start + 1:]:
            lowest_match = _LOWEST_ROOT_RE.search(line)
            if lowest_match:
                transitions["lowest_root"] = {
                    "root": int(lowest_match.group("root")),
                    "multiplicity": int(lowest_match.group("mult")),
                    "energy_Eh": float(lowest_match.group("energy")),
                    "energy_eV": float(lowest_match.group("ev")),
                }
                continue

            row_match = _TRANSITION_ROW_RE.match(line)
            if row_match:
                transitions["states"].append(
                    {
                        "state": int(row_match.group("state")),
                        "root": int(row_match.group("root")),
                        "multiplicity": int(row_match.group("mult")),
                        "delta_energy_Eh": float(row_match.group("de_au")),
                        "delta_energy_eV": float(row_match.group("de_ev")),
                        "delta_energy_cm-1": float(row_match.group("de_cm")),
                    }
                )
                continue

            if transitions["states"] and not line.strip():
                break
            if transitions["states"] and _is_dashed(line):
                break

        return transitions if transitions.get("states") or transitions.get("lowest_root") else {}

    def _parse_matrix(self, lines: Sequence[str], heading: str, trace_kind: str) -> Dict[str, Any]:
        """Find and parse a named density-like matrix section."""
        start = _find_exact(lines, heading)
        if start == -1:
            return {}
        return self._parse_matrix_at(lines, start, trace_kind=trace_kind)

    def _parse_matrix_at(
        self,
        lines: Sequence[str],
        start: int,
        *,
        trace_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Parse ORCA block-printed matrices into row/column/matrix lists."""
        current_columns: List[int] = []
        row_maps: Dict[int, Dict[int, float]] = {}
        all_columns: set[int] = set()
        trace: Optional[float] = None
        started = False

        for line in lines[start + 1:]:
            trace_match = _TRACE_RE.search(line)
            if trace_match and (trace_kind is None or trace_match.group("kind").lower() == trace_kind):
                trace = float(trace_match.group("trace"))
                break

            header_match = _MATRIX_HEADER_RE.match(line)
            if header_match:
                current_columns = [int(item) for item in line.split()]
                all_columns.update(current_columns)
                started = True
                continue

            row_match = _MATRIX_ROW_RE.match(line)
            if row_match and current_columns:
                row_index = int(row_match.group("row"))
                values = _numbers(row_match.group("values"))
                row = row_maps.setdefault(row_index, {})
                for column, value in zip(current_columns, values):
                    row[column] = value
                started = True
                continue

            stripped = line.strip()
            if started and stripped and not _is_dashed(stripped):
                if not header_match and not row_match:
                    if stripped.isupper() or stripped.startswith("***"):
                        break

        if not row_maps:
            return {}

        rows = sorted(row_maps)
        columns = sorted(all_columns)
        matrix = [
            [row_maps[row].get(column) for column in columns]
            for row in rows
        ]
        result: Dict[str, Any] = {
            "rows": rows,
            "columns": columns,
            "matrix": matrix,
            "n_rows": len(rows),
            "n_columns": len(columns),
        }
        if trace is not None:
            result["trace"] = trace
        return result

    def _parse_energy_components(self, lines: Sequence[str]) -> Dict[str, Any]:
        """Parse one-electron, two-electron, nuclear, virial, and core components."""
        start = _find_exact(lines, "ENERGY COMPONENTS")
        if start == -1:
            return {}
        components: Dict[str, Any] = {"components": {}, "total_checks_Eh": []}
        expect_total = False

        for line in lines[start + 1:]:
            if line.strip() == "LOEWDIN ORBITAL-COMPOSITIONS":
                break
            if _is_dashed(line):
                expect_total = True
                continue
            if expect_total:
                total_match = _ENERGY_TOTAL_RE.match(line)
                if total_match:
                    components["total_checks_Eh"].append(float(total_match.group("energy")))
                    expect_total = False
                    continue

            match = _ENERGY_COMPONENT_RE.match(line)
            if match:
                key = _normalize_key(match.group("label"))
                entry: Dict[str, Any] = {"value": float(match.group("eh"))}
                label_lower = match.group("label").lower()
                if "virial" in label_lower:
                    entry["unit"] = "ratio"
                else:
                    entry["unit"] = "Eh"
                    if match.group("ev") is not None:
                        entry["value_eV"] = float(match.group("ev"))
                components["components"][key] = entry

            if components["components"] and not line.strip():
                continue

        return components if components["components"] else {}

    def _active_orbital_indices(self, data: Dict[str, Any]) -> List[int]:
        """Return active orbital indices from the setup range or fractional occupations."""
        active = (data.get("system") or {}).get("active_orbital_range") or {}
        first = active.get("first_index")
        last = active.get("last_index")
        if isinstance(first, int) and isinstance(last, int):
            return list(range(first, last + 1))
        occupations = (data.get("convergence") or {}).get("final", {}).get("active_occupations") or []
        orbital_energies = data.get("orbital_energies") or []
        if occupations and orbital_energies:
            fractional = [
                orbital["index"]
                for orbital in orbital_energies
                if 0.05 < float(orbital.get("occupation", 0.0)) < 1.95
            ]
            return fractional
        return []

    def _configured_orbital_window_size(self) -> int:
        """Return the requested orbital window around the active/frontier range."""
        option_value = (self.context.get("plugin_options") or {}).get("casscf_orbital_window", 30)
        try:
            return max(0, int(option_value))
        except (TypeError, ValueError):
            return 30

    def _orbital_composition_window(self, data: Dict[str, Any], active_indices: List[int]) -> List[int]:
        """Return the bounded orbital index window used for bulky Loewdin compositions."""
        window = self._configured_orbital_window_size()

        if active_indices:
            first = max(0, min(active_indices) - window)
            last = max(active_indices) + window
            return list(range(first, last + 1))

        orbitals = data.get("orbital_energies") or []
        if not orbitals:
            return []

        homo = None
        lumo = None
        for orbital in orbitals:
            if float(orbital.get("occupation", 0.0)) > 0.5:
                homo = int(orbital["index"])
            elif homo is not None and lumo is None:
                lumo = int(orbital["index"])
                break
        if homo is None:
            return []
        first = max(0, homo - window)
        last = (lumo if lumo is not None else homo) + window
        return list(range(first, last + 1))

    def _orbital_energy_window(self, data: Dict[str, Any], active_indices: List[int]) -> Dict[str, Any]:
        """Return the requested active/frontier window for the orbital-energy report."""
        orbitals = data.get("orbital_energies") or []
        if not orbitals:
            return {}

        window = self._configured_orbital_window_size()
        available = [int(orbital["index"]) for orbital in orbitals if orbital.get("index") is not None]
        if not available:
            return {}

        if active_indices:
            anchor_first = min(active_indices)
            anchor_last = max(active_indices)
            anchor = "active_space"
        else:
            homo = None
            lumo = None
            for orbital in orbitals:
                index = int(orbital["index"])
                if float(orbital.get("occupation", 0.0)) > 0.5:
                    homo = index
                elif homo is not None and lumo is None:
                    lumo = index
                    break
            if homo is None:
                return {}
            anchor_first = homo
            anchor_last = lumo if lumo is not None else homo
            anchor = "frontier"

        requested_first = max(0, anchor_first - window)
        requested_last = anchor_last + window
        selected = [index for index in available if requested_first <= index <= requested_last]
        if not selected:
            return {}
        return {
            "window": window,
            "anchor": anchor,
            "anchor_first_index": anchor_first,
            "anchor_last_index": anchor_last,
            "selected_min_index": min(selected),
            "selected_max_index": max(selected),
            "selected_indices": selected,
        }

    def _parse_loewdin_compositions(
        self,
        lines: Sequence[str],
        heading: str,
        *,
        selected_indices: Optional[Sequence[int]],
        contribution_threshold: float,
        stop_at: Sequence[str],
        start_at: int = 0,
    ) -> Dict[str, Any]:
        """Parse ORCA Loewdin MO composition tables for selected orbital columns."""
        start = _find_exact(lines, heading, start_at)
        if start == -1:
            return {}
        selected = set(selected_indices) if selected_indices is not None else None
        stop_targets = {item.strip().lower() for item in stop_at}
        orbitals: Dict[int, Dict[str, Any]] = {}
        contribution_rows = 0
        index = start + 1

        while index < len(lines):
            stripped = lines[index].strip()
            if stripped.lower() in stop_targets:
                break
            if stripped == "ORCA POPULATION ANALYSIS":
                break

            header_match = _COMPOSITION_HEADER_RE.match(lines[index])
            if not header_match:
                index += 1
                continue

            indices = [int(item) for item in header_match.group("indices").split()]
            energies = _numbers(lines[index + 1]) if index + 1 < len(lines) else []
            occupations = _numbers(lines[index + 2]) if index + 2 < len(lines) else []
            keep_positions = [
                position
                for position, orbital_index in enumerate(indices)
                if selected is None or orbital_index in selected
            ]
            for position in keep_positions:
                orbital_index = indices[position]
                orbitals.setdefault(
                    orbital_index,
                    {
                        "index": orbital_index,
                        "energy_Eh": energies[position] if position < len(energies) else None,
                        "occupation": occupations[position] if position < len(occupations) else None,
                        "contributions": [],
                    },
                )

            index += 4
            while index < len(lines):
                row_line = lines[index]
                if not row_line.strip():
                    index += 1
                    break
                if row_line.strip().lower() in stop_targets:
                    break
                row_match = _COMPOSITION_ROW_RE.match(row_line)
                if not row_match:
                    if _COMPOSITION_HEADER_RE.match(row_line):
                        break
                    index += 1
                    continue

                values = _numbers(row_match.group("values"))
                for position in keep_positions:
                    if position >= len(values):
                        continue
                    value = values[position]
                    if abs(value) <= contribution_threshold:
                        continue
                    orbital_index = indices[position]
                    orbitals[orbital_index]["contributions"].append(
                        {
                            "atom_index": int(row_match.group("atom")),
                            "element": row_match.group("element"),
                            "ao_label": row_match.group("orbital"),
                            "percent": value,
                        }
                    )
                    contribution_rows += 1
                index += 1

        parsed_orbitals = [orbitals[key] for key in sorted(orbitals)]
        for orbital in parsed_orbitals:
            orbital["contributions"].sort(
                key=lambda item: abs(float(item.get("percent", 0.0))),
                reverse=True,
            )
            orbital["n_contributions"] = len(orbital["contributions"])

        return {
            "orbitals": parsed_orbitals,
            "n_orbitals": len(parsed_orbitals),
            "n_contributions": contribution_rows,
            "contribution_threshold_percent": contribution_threshold,
        } if parsed_orbitals else {}

    def _parse_nevpt2(self, lines: Sequence[str]) -> Dict[str, Any]:
        """Parse SC-NEVPT2 and QD-NEVPT2 summaries printed inside CASSCF jobs."""
        if _find_exact(lines, "NEVPT2 Results") == -1 and _find_exact(lines, "QD-NEVPT2 Results") == -1:
            return {}

        nevpt2: Dict[str, Any] = {}
        state_results = self._parse_nevpt2_state_results(lines, "NEVPT2 Results", "NEVPT2 TOTAL ENERGIES")
        if state_results:
            nevpt2["state_results"] = state_results

        total_energies = self._parse_total_energy_table(lines, "NEVPT2 TOTAL ENERGIES")
        if total_energies:
            nevpt2["total_energies"] = total_energies

        transitions = self._parse_transition_energies(lines, "NEVPT2 TRANSITION ENERGIES")
        if transitions:
            nevpt2["transition_energies"] = transitions

        corrections = self._parse_transition_energy_corrections(lines)
        if corrections:
            nevpt2["transition_energy_corrections"] = corrections

        qd: Dict[str, Any] = {}
        qd_state_results = self._parse_qd_nevpt2_state_results(lines)
        if qd_state_results:
            qd["state_results"] = qd_state_results
        qd_total = self._parse_total_energy_table(lines, "QD-NEVPT2 TOTAL ENERGIES")
        if qd_total:
            qd["total_energies"] = qd_total
        qd_transitions = self._parse_transition_energies(lines, "QD-NEVPT2 TRANSITION ENERGIES")
        if qd_transitions:
            qd["transition_energies"] = qd_transitions
        qd_density_idx = _find_exact(lines, "DENSITY MATRIX (QD-NEVPT2 CORRECTED)")
        if qd_density_idx != -1:
            qd_density = self._parse_matrix_at(lines, qd_density_idx, trace_kind="electron")
            if qd_density:
                qd["density_matrix"] = qd_density
        qd_spin_idx = _find_exact(lines, "SPIN-DENSITY MATRIX (QD-NEVPT2 CORRECTED)")
        if qd_spin_idx != -1:
            qd_spin = self._parse_matrix_at(lines, qd_spin_idx, trace_kind="spin")
            if qd_spin:
                qd["spin_density_matrix"] = qd_spin
        corrected_population_idx = _find_line_containing(
            lines,
            "QD-NEVPT2: Repeating the population analysis with the corrected densities",
        )
        if corrected_population_idx != -1:
            corrected_active_mos = self._parse_loewdin_compositions(
                lines,
                "LOEWDIN REDUCED ACTIVE MOs",
                selected_indices=None,
                contribution_threshold=0.0,
                stop_at=("ORCA POPULATION ANALYSIS", "CASSCF UV, CD spectra and dipole moments"),
                start_at=corrected_population_idx,
            )
            if corrected_active_mos:
                qd["corrected_loewdin_reduced_active_mos"] = corrected_active_mos
        natural_orbitals = self._parse_qd_nevpt2_natural_orbitals(lines)
        if natural_orbitals:
            qd["state_specific_natural_orbitals"] = natural_orbitals
        if qd:
            nevpt2["qd_nevpt2"] = qd

        timings = self._parse_nevpt2_timings(lines)
        if timings:
            nevpt2["timings"] = timings

        return nevpt2

    def _parse_qd_nevpt2_state_results(self, lines: Sequence[str]) -> List[Dict[str, Any]]:
        """Parse Van Vleck QD-NEVPT2 multiplicity/root blocks and configurations."""
        start = _find_exact(lines, "QD-NEVPT2 Results")
        if start == -1:
            return []
        end = _find_exact(lines, "QD-NEVPT2 TOTAL ENERGIES", start + 1)
        if end == -1:
            end = len(lines)

        states: List[Dict[str, Any]] = []
        current_mult: Optional[int] = None
        current: Optional[Dict[str, Any]] = None
        current_hamiltonian: Optional[Dict[str, Any]] = None
        index = start + 1
        while index < end:
            line = lines[index]
            mult_match = re.match(r"^\s*MULT\s+(?P<mult>\d+)\s*$", line, re.I)
            if mult_match:
                if current is not None:
                    states.append(current)
                    current = None
                current_mult = int(mult_match.group("mult"))
                current_hamiltonian = None
                index += 1
                continue

            if "Total Hamiltonian to be processed" in line:
                current_hamiltonian = self._parse_matrix_at(lines, index)
                index += 1
                continue

            root_match = re.match(r"^\s*ROOT\s*=\s*(?P<root>\d+)\s*$", line, re.I)
            if root_match:
                if current is not None:
                    states.append(current)
                current = {
                    "multiplicity": current_mult,
                    "root": int(root_match.group("root")),
                    "configurations": [],
                }
                if current_hamiltonian:
                    current["hamiltonian"] = current_hamiltonian
                index += 1
                continue

            if current is None:
                index += 1
                continue

            config_match = _CONFIG_RE.match(line)
            if config_match:
                current["configurations"].append(
                    {
                        "weight": float(config_match.group("weight")),
                        "configuration_index": int(config_match.group("index")),
                        "occupation_string": config_match.group("occupation"),
                    }
                )
                index += 1
                continue

            total_corr = _NEVPT_TOTAL_CORR_RE.search(line)
            if total_corr:
                current["total_energy_correction_Eh"] = float(total_corr.group("energy"))
                index += 1
                continue

            ref = _NEVPT_REF_RE.search(line)
            if ref:
                current["reference_energy_Eh"] = float(ref.group("energy"))
                index += 1
                continue

            total = _NEVPT_TOTAL_RE.search(line)
            if total:
                current["total_energy_Eh"] = float(total.group("energy"))
                index += 1
                continue

            index += 1

        if current is not None:
            states.append(current)

        for state in states:
            configs = state.get("configurations", [])
            if configs:
                state["dominant_configuration"] = dict(
                    max(configs, key=lambda item: item.get("weight", 0.0))
                )
                state["configuration_count"] = len(configs)
        return states

    def _parse_nevpt2_state_results(
        self,
        lines: Sequence[str],
        start_heading: str,
        end_heading: str,
    ) -> List[Dict[str, Any]]:
        """Parse per-root NEVPT2 corrections, reference energies, and totals."""
        start = _find_exact(lines, start_heading)
        if start == -1:
            return []
        end = _find_exact(lines, end_heading, start + 1)
        if end == -1:
            end = len(lines)

        states: List[Dict[str, Any]] = []
        current: Optional[Dict[str, Any]] = None
        for line in lines[start + 1:end]:
            root_match = _NEVPT_ROOT_RE.match(line)
            if root_match:
                if current is not None:
                    states.append(current)
                current = {
                    "multiplicity": int(root_match.group("mult")),
                    "root": int(root_match.group("root")),
                    "class_corrections": {},
                    "configurations": [],
                }
                continue

            if current is None:
                continue

            config_match = _CONFIG_RE.match(line)
            if config_match:
                current["configurations"].append(
                    {
                        "weight": float(config_match.group("weight")),
                        "configuration_index": int(config_match.group("index")),
                        "occupation_string": config_match.group("occupation"),
                    }
                )
                continue

            class_match = _NEVPT_CLASS_RE.match(line)
            if class_match:
                current["class_corrections"][class_match.group("class")] = float(class_match.group("energy"))
                continue

            total_corr = _NEVPT_TOTAL_CORR_RE.search(line)
            if total_corr:
                current["total_energy_correction_Eh"] = float(total_corr.group("energy"))
                continue

            ref = _NEVPT_REF_RE.search(line)
            if ref:
                current["reference_energy_Eh"] = float(ref.group("energy"))
                continue

            total = _NEVPT_TOTAL_RE.search(line)
            if total:
                current["total_energy_Eh"] = float(total.group("energy"))

        if current is not None:
            states.append(current)
        return states

    def _parse_total_energy_table(self, lines: Sequence[str], heading: str) -> List[Dict[str, Any]]:
        """Parse NEVPT2/QD-NEVPT2 total-energy tables including EDIAG metadata."""
        start = _find_exact(lines, heading)
        if start == -1:
            return []
        ediag_unit = "Eh"
        for header_line in lines[start + 1:start + 10]:
            if _TOTAL_ENERGY_ROW_RE.match(header_line):
                break
            if "MRCI SOC BLOCK INPUT" in header_line and "cm**-1" in header_line:
                ediag_unit = "cm-1"
                break
        rows: List[Dict[str, Any]] = []
        for line in lines[start + 1:]:
            match = _TOTAL_ENERGY_ROW_RE.match(line)
            if match:
                rows.append(
                    {
                        "state": int(match.group("state")),
                        "root": int(match.group("root")),
                        "multiplicity": int(match.group("mult")),
                        "energy_Eh": float(match.group("energy")),
                        "ediag_index": int(match.group("ediag")),
                        "ediag_value": float(match.group("ediag_value")),
                        "ediag_unit": ediag_unit,
                    }
                )
                continue
            if rows and not line.strip():
                break
            if rows and _is_dashed(line):
                break
        return rows

    def _parse_transition_energy_corrections(self, lines: Sequence[str]) -> List[Dict[str, Any]]:
        """Parse NEVPT2 correction-to-transition-energy rows."""
        start = _find_exact(lines, "NEVPT2 CORRECTION TO THE TRANSITION ENERGY")
        if start == -1:
            return []
        end = _find_exact(lines, "QD-NEVPT2 Results", start + 1)
        if end == -1:
            end = len(lines)
        corrections: List[Dict[str, Any]] = []
        for line in lines[start + 1:end]:
            match = _TRANSITION_ROW_RE.match(line)
            if match:
                corrections.append(
                    {
                        "state": int(match.group("state")),
                        "root": int(match.group("root")),
                        "multiplicity": int(match.group("mult")),
                        "delta_energy_Eh": float(match.group("de_au")),
                        "delta_energy_eV": float(match.group("de_ev")),
                        "delta_energy_cm-1": float(match.group("de_cm")),
                    }
                )
        return corrections

    def _parse_nevpt2_timings(self, lines: Sequence[str]) -> List[Dict[str, Any]]:
        """Parse the NEVPT2 timing table when ORCA prints it."""
        start = _find_exact(lines, "TIMINGS NEVPT2")
        if start == -1:
            return []
        rows: List[Dict[str, Any]] = []
        for line in lines[start + 1:start + 80]:
            match = _TIMING_RE.match(line)
            if not match:
                continue
            item: Dict[str, Any] = {
                "label": match.group("label").strip(),
                "seconds": float(match.group("seconds")),
            }
            if match.group("percent") is not None:
                item["percent"] = float(match.group("percent"))
            rows.append(item)
        return rows

    def _parse_qd_nevpt2_natural_orbitals(self, lines: Sequence[str]) -> List[Dict[str, Any]]:
        """Parse state-specific QD-NEVPT2 natural orbital occupations and files."""
        start = _find_line_containing(lines, "State-specific QD-NEVPT2 natural orbitals")
        if start == -1:
            return []
        end = _find_exact(lines, "CASSCF UV, CD spectra and dipole moments", start + 1)
        if end == -1:
            end = len(lines)

        rows: List[Dict[str, Any]] = []
        current_block: Optional[int] = None
        current_mult: Optional[int] = None
        current: Optional[Dict[str, Any]] = None
        for line in lines[start + 1:end]:
            block_match = _NATORB_BLOCK_RE.match(line)
            if block_match:
                if current is not None:
                    rows.append(current)
                    current = None
                current_block = int(block_match.group("block"))
                current_mult = int(block_match.group("mult"))
                continue

            root_match = _NATORB_ROOT_RE.match(line)
            if root_match:
                if current is not None:
                    rows.append(current)
                current = {
                    "block_index": current_block,
                    "multiplicity": current_mult,
                    "root": int(root_match.group("root")),
                }
                continue

            if current is None:
                continue

            occ_match = _N_OCC_RE.search(line)
            if occ_match:
                current["natural_occupations"] = _numbers(occ_match.group("values"))
                continue

            stored_match = _NATORB_STORED_RE.search(line)
            if stored_match:
                current["file"] = stored_match.group("path")

        if current is not None:
            rows.append(current)
        return rows

    def _parse_spectra(self, lines: Sequence[str]) -> List[Dict[str, Any]]:
        """Parse CASSCF, NEVPT2-diagonal, and QD-NEVPT2 UV/CD spectra tables."""
        headings = [
            ("casscf", "CASSCF UV, CD spectra and dipole moments"),
            ("casscf_nevpt2_diagonal", "CASSCF (NEVPT2 diagonal energies) UV, CD spectra and dipole moments"),
            ("qd_nevpt2", "QD-NEVPT2 UV, CD spectra and dipole moments"),
        ]
        sections: List[Dict[str, Any]] = []
        for key, heading in headings:
            start = _find_exact(lines, heading)
            if start == -1:
                continue
            possible_ends = [
                _find_exact(lines, other_heading, start + 1)
                for _, other_heading in headings
                if other_heading != heading
            ]
            rel_start = _find_exact(lines, "CASSCF RELATIVISTIC PROPERTIES", start + 1)
            if rel_start != -1:
                possible_ends.append(rel_start)
            end_candidates = [item for item in possible_ends if item != -1]
            end = min(end_candidates) if end_candidates else len(lines)
            absorption: List[Dict[str, Any]] = []
            cd: List[Dict[str, Any]] = []
            mode = ""
            for line in lines[start + 1:end]:
                if "ABSORPTION SPECTRUM" in line:
                    mode = "absorption"
                    continue
                if "CD SPECTRUM" in line:
                    mode = "cd"
                    continue
                row = self._parse_spectrum_row(line, mode)
                if not row:
                    continue
                if mode == "absorption":
                    absorption.append(row)
                elif mode == "cd":
                    cd.append(row)
            if absorption or cd:
                sections.append(
                    {
                        "key": key,
                        "title": heading,
                        "absorption": absorption,
                        "cd": cd,
                    }
                )
        return sections

    def _parse_spectrum_row(self, line: str, mode: str) -> Optional[Dict[str, Any]]:
        """Parse one absorption or CD spectrum row."""
        if mode not in {"absorption", "cd"}:
            return None
        match = _SPECTRUM_ROW_RE.match(line)
        if not match:
            return None
        row: Dict[str, Any] = {
            "from": match.group("from"),
            "to": match.group("to"),
            "energy_eV": float(match.group("energy_ev")),
            "energy_cm-1": float(match.group("energy_cm")),
            "wavelength_nm": float(match.group("wavelength_nm")),
        }
        row.update(self._parse_transition_label("from", match.group("from")))
        row.update(self._parse_transition_label("to", match.group("to")))
        if mode == "absorption":
            row.update(
                {
                    "oscillator_strength": float(match.group("a")),
                    "dipole_strength_au2": float(match.group("b")),
                    "transition_dipole_x_au": float(match.group("c")),
                    "transition_dipole_y_au": float(match.group("d")),
                    "transition_dipole_z_au": float(match.group("e")) if match.group("e") is not None else None,
                }
            )
        else:
            row.update(
                {
                    "rotatory_strength_1e40_cgs": float(match.group("a")),
                    "magnetic_dipole_x_au": float(match.group("b")),
                    "magnetic_dipole_y_au": float(match.group("c")),
                    "magnetic_dipole_z_au": float(match.group("d")),
                }
            )
        return row

    def _parse_transition_label(self, prefix: str, label: str) -> Dict[str, Any]:
        """Break ORCA labels like ``0-4A`` into root, multiplicity, and irrep."""
        match = _TRANSITION_LABEL_RE.match(label)
        if not match:
            return {f"{prefix}_label": label}
        mult_text = match.group("mult")
        mult: Any = int(float(mult_text)) if float(mult_text).is_integer() else float(mult_text)
        return {
            f"{prefix}_label": label,
            f"{prefix}_root": int(match.group("root")),
            f"{prefix}_multiplicity": mult,
            f"{prefix}_irrep": match.group("irrep") or "",
        }

    def _parse_relativistic_properties(self, lines: Sequence[str]) -> Dict[str, Any]:
        """Parse QDPT levels/eigenvectors plus g- and D-tensor summaries."""
        start = _find_exact(lines, "CASSCF RELATIVISTIC PROPERTIES")
        if start == -1:
            return {}
        end = _find_line_containing(lines, "EPR properties:", start + 1)
        if end == -1:
            end = _find_line_containing(lines, "ORCA TERMINATED NORMALLY", start + 1)
        if end == -1:
            end = len(lines)

        result: Dict[str, Any] = {}
        basis = self._parse_relativistic_basis(lines[start:end])
        if basis:
            result["basis"] = basis

        qdpt_blocks: List[Dict[str, Any]] = []
        index = start + 1
        while index < end:
            heading_match = _QDPT_HEADING_RE.match(lines[index])
            if not heading_match:
                index += 1
                continue
            next_index = index + 1
            while next_index < end and not _QDPT_HEADING_RE.match(lines[next_index]):
                next_index += 1
            qdpt_blocks.append(self._parse_qdpt_block(lines[index:next_index], heading_match.group("label").strip()))
            index = next_index
        if qdpt_blocks:
            result["qdpt_blocks"] = qdpt_blocks
        return result

    def _parse_relativistic_basis(self, block_lines: Sequence[str]) -> Dict[str, Any]:
        """Parse the Order/MS/Mult/irrep state basis printed before QDPT blocks."""
        basis: Dict[str, Any] = {}
        for line in block_lines[:30]:
            stripped = line.strip()
            if stripped.startswith("Order"):
                basis["order"] = [int(value) for value in re.findall(r"[-+]?\d+", stripped)]
            elif stripped.startswith("MS *2"):
                basis["ms_times_2"] = [int(value) for value in re.findall(r"[-+]?\d+", stripped)]
            elif stripped.startswith("Mult"):
                basis["multiplicities"] = [int(value) for value in re.findall(r"[-+]?\d+", stripped)]
            elif stripped.startswith("irrep"):
                basis["irreps"] = [int(value) for value in re.findall(r"[-+]?\d+", stripped)]
        return basis

    def _parse_qdpt_block(self, block_lines: Sequence[str], label: str) -> Dict[str, Any]:
        """Parse one QDPT correction variant."""
        block: Dict[str, Any] = {
            "label": label,
            "levels": [],
            "eigenvectors": [],
            "g_matrices": [],
            "d_tensors": [],
        }
        index = 0
        while index < len(block_lines):
            line = block_lines[index]
            if "Doing QDPT" in line:
                block["mode"] = line.strip()
            lowest_match = re.search(rf"Lowest eigenvalue .*:\s*(?P<value>{_FLOAT_RE})\s+Eh", line, re.I)
            if lowest_match:
                block["lowest_eigenvalue_Eh"] = float(lowest_match.group("value"))
            stabilization_match = re.search(rf"Energy stabilization:\s*(?P<value>{_FLOAT_RE})\s*cm-1", line, re.I)
            if stabilization_match:
                block["energy_stabilization_cm-1"] = float(stabilization_match.group("value"))
            if "Eigenvalues:" in line:
                levels, index = self._parse_qdpt_levels(block_lines, index + 1)
                block["levels"] = levels
                continue
            if line.strip() == "Eigenvectors:":
                vectors, index = self._parse_qdpt_eigenvectors(block_lines, index + 1)
                block["eigenvectors"].extend(vectors)
                continue
            if line.strip() in {
                "ELECTRONIC G-MATRIX FROM EFFECTIVE HAMILTONIAN",
                "ELECTRONIC G-MATRIX",
                "ELECTRONIC G-MATRIX: S contribution",
                "ELECTRONIC G-MATRIX: L contribution",
            }:
                g_matrix, index = self._parse_g_matrix_block(block_lines, index)
                if g_matrix:
                    block["g_matrices"].append(g_matrix)
                continue
            if line.strip() == "ZERO-FIELD SPLITTING":
                d_tensor, index = self._parse_d_tensor_block(block_lines, index)
                if d_tensor:
                    block["d_tensors"].append(d_tensor)
                continue
            index += 1
        return block

    def _parse_qdpt_levels(self, block_lines: Sequence[str], start: int) -> tuple[List[Dict[str, Any]], int]:
        levels: List[Dict[str, Any]] = []
        index = start
        while index < len(block_lines):
            match = _QDPT_LEVEL_RE.match(block_lines[index])
            if match:
                levels.append(
                    {
                        "state": int(match.group("state")),
                        "energy_cm-1": float(match.group("energy_cm")),
                        "energy_eV": float(match.group("energy_ev")),
                        "boltzmann_population": float(match.group("population")),
                    }
                )
                index += 1
                continue
            if levels and not block_lines[index].strip():
                break
            if levels and block_lines[index].strip().startswith("The threshold"):
                break
            index += 1
        return levels, index

    def _parse_qdpt_eigenvectors(self, block_lines: Sequence[str], start: int) -> tuple[List[Dict[str, Any]], int]:
        vectors: List[Dict[str, Any]] = []
        current: Optional[Dict[str, Any]] = None
        index = start
        while index < len(block_lines):
            line = block_lines[index]
            state_match = _QDPT_EIGENVECTOR_STATE_RE.match(line)
            if state_match:
                if current is not None:
                    vectors.append(current)
                current = {
                    "state": int(state_match.group("state")),
                    "energy_cm-1": float(state_match.group("energy_cm")),
                    "components": [],
                }
                index += 1
                continue
            component_match = _QDPT_EIGENVECTOR_COMPONENT_RE.match(line)
            if component_match and current is not None:
                current["components"].append(
                    {
                        "weight": float(component_match.group("weight")),
                        "real": float(component_match.group("real")),
                        "imaginary": float(component_match.group("imag")),
                        "block": int(component_match.group("block")),
                        "root": int(component_match.group("root")),
                        "spin": component_match.group("spin"),
                        "ms": component_match.group("ms"),
                    }
                )
                index += 1
                continue
            stripped = line.strip()
            if current is not None and stripped and (
                stripped.startswith("-")
                or stripped.startswith("*")
                or stripped.startswith("Computing")
                or stripped in {
                    "ELECTRONIC G-MATRIX FROM EFFECTIVE HAMILTONIAN",
                    "ELECTRONIC G-MATRIX",
                    "ELECTRONIC G-MATRIX: S contribution",
                    "ELECTRONIC G-MATRIX: L contribution",
                    "ZERO-FIELD SPLITTING",
                }
            ):
                break
            index += 1
        if current is not None:
            vectors.append(current)
        return vectors, index

    def _parse_g_matrix_block(self, block_lines: Sequence[str], start: int) -> tuple[Dict[str, Any], int]:
        title = block_lines[start].strip()
        end = start + 1
        while end < len(block_lines):
            stripped = block_lines[end].strip()
            if end > start + 1 and stripped in {
                "ELECTRONIC G-MATRIX FROM EFFECTIVE HAMILTONIAN",
                "ELECTRONIC G-MATRIX",
                "ELECTRONIC G-MATRIX: S contribution",
                "ELECTRONIC G-MATRIX: L contribution",
                "ZERO-FIELD SPLITTING",
            }:
                break
            if end > start + 1 and stripped.startswith("KRAMERS PAIR"):
                break
            end += 1
        section = block_lines[start:end]
        result: Dict[str, Any] = {"title": title}
        for offset, line in enumerate(section):
            if line.strip() == "total g-matrix:":
                matrix = self._parse_matrix_at(section, offset)
                if matrix:
                    result["matrix"] = matrix
            if line.strip().startswith("g-factors:") and offset + 1 < len(section):
                match = _GFACTORS_RE.match(section[offset + 1])
                if match:
                    result["g_factors"] = {
                        "x": float(match.group("x")),
                        "y": float(match.group("y")),
                        "z": float(match.group("z")),
                        "iso": float(match.group("iso")),
                    }
            if line.strip().startswith("g-shifts:") and offset + 1 < len(section):
                match = _GFACTORS_RE.match(section[offset + 1])
                if match:
                    result["g_shifts"] = {
                        "x": float(match.group("x")),
                        "y": float(match.group("y")),
                        "z": float(match.group("z")),
                        "iso": float(match.group("iso")),
                    }
        return result, end

    def _parse_d_tensor_block(self, block_lines: Sequence[str], start: int) -> tuple[Dict[str, Any], int]:
        title_lines = [block_lines[start].strip()]
        if start + 1 < len(block_lines) and block_lines[start + 1].strip():
            title_lines.append(block_lines[start + 1].strip())
        end = start + 1
        while end < len(block_lines):
            stripped = block_lines[end].strip()
            if end > start + 1 and stripped in {
                "ELECTRONIC G-MATRIX FROM EFFECTIVE HAMILTONIAN",
                "ELECTRONIC G-MATRIX",
                "ELECTRONIC G-MATRIX: S contribution",
                "ELECTRONIC G-MATRIX: L contribution",
                "ZERO-FIELD SPLITTING",
            }:
                break
            if end > start + 1 and stripped.startswith("Computing the QDPT Transition"):
                break
            end += 1
        section = block_lines[start:end]
        result: Dict[str, Any] = {"title": " ".join(title_lines)}
        for line in section:
            d_match = _D_VALUE_RE.match(line)
            if d_match:
                result["D_cm-1"] = float(d_match.group("d"))
            e_match = _E_OVER_D_RE.match(line)
            if e_match:
                result["E_over_D"] = float(e_match.group("e_over_d"))
        return result, end

    def _attach_state_assignments(self, data: Dict[str, Any]) -> None:
        """Join transition-energy state labels with root configs and PT2 corrections."""
        root_lookup = self._root_lookup(data.get("state_blocks") or [])
        transitions = data.get("transition_energies") or {}
        assignments = self._build_assignments(transitions, root_lookup=root_lookup)
        if assignments:
            data["state_assignments"] = assignments

        nevpt2 = data.get("nevpt2") or {}
        if nevpt2:
            nevpt2_assignments = self._build_assignments(
                nevpt2.get("transition_energies") or {},
                root_lookup=root_lookup,
                total_energies=nevpt2.get("total_energies") or [],
                state_results=nevpt2.get("state_results") or [],
                transition_corrections=nevpt2.get("transition_energy_corrections") or [],
            )
            if nevpt2_assignments:
                nevpt2["state_assignments"] = nevpt2_assignments

            qd = nevpt2.get("qd_nevpt2") or {}
            if qd:
                qd_assignments = self._build_assignments(
                    qd.get("transition_energies") or {},
                    root_lookup=root_lookup,
                    total_energies=qd.get("total_energies") or [],
                    state_results=qd.get("state_results") or [],
                )
                if qd_assignments:
                    qd["state_assignments"] = qd_assignments

    def _root_lookup(self, state_blocks: Sequence[Dict[str, Any]]) -> Dict[tuple[int, int], Dict[str, Any]]:
        lookup: Dict[tuple[int, int], Dict[str, Any]] = {}
        for block in state_blocks:
            mult = block.get("multiplicity")
            if not isinstance(mult, int):
                continue
            for root in block.get("roots", []):
                if isinstance(root.get("root"), int):
                    lookup[(mult, root["root"])] = root
        return lookup

    def _build_assignments(
        self,
        transitions: Dict[str, Any],
        *,
        root_lookup: Dict[tuple[int, int], Dict[str, Any]],
        total_energies: Sequence[Dict[str, Any]] = (),
        state_results: Sequence[Dict[str, Any]] = (),
        transition_corrections: Sequence[Dict[str, Any]] = (),
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        total_by_state = {row.get("state"): row for row in total_energies}
        total_by_root = {
            (row.get("multiplicity"), row.get("root")): row
            for row in total_energies
        }
        result_by_root = {
            (row.get("multiplicity"), row.get("root")): row
            for row in state_results
        }
        correction_by_state = {row.get("state"): row for row in transition_corrections}

        lowest = transitions.get("lowest_root") or {}
        if lowest:
            base = {
                "state": 0,
                "root": lowest.get("root"),
                "multiplicity": lowest.get("multiplicity"),
                "delta_energy_Eh": 0.0,
                "delta_energy_eV": 0.0,
                "delta_energy_cm-1": 0.0,
                "reference": "lowest_root",
            }
            if lowest.get("energy_Eh") is not None:
                base["absolute_energy_Eh"] = lowest.get("energy_Eh")
            rows.append(self._decorate_assignment(base, root_lookup, total_by_state, total_by_root, result_by_root, correction_by_state))

        for state in transitions.get("states") or []:
            rows.append(
                self._decorate_assignment(
                    dict(state),
                    root_lookup,
                    total_by_state,
                    total_by_root,
                    result_by_root,
                    correction_by_state,
                )
            )
        return rows

    def _decorate_assignment(
        self,
        row: Dict[str, Any],
        root_lookup: Dict[tuple[int, int], Dict[str, Any]],
        total_by_state: Dict[Any, Dict[str, Any]],
        total_by_root: Dict[tuple[Any, Any], Dict[str, Any]],
        result_by_root: Dict[tuple[Any, Any], Dict[str, Any]],
        correction_by_state: Dict[Any, Dict[str, Any]],
    ) -> Dict[str, Any]:
        key = (row.get("multiplicity"), row.get("root"))
        root_data = root_lookup.get(key) or {}
        result = result_by_root.get(key) or {}
        total = total_by_state.get(row.get("state")) or total_by_root.get(key) or {}
        correction = correction_by_state.get(row.get("state")) or {}

        if root_data.get("energy_Eh") is not None:
            row["casscf_root_energy_Eh"] = root_data.get("energy_Eh")
        configurations = result.get("configurations") or root_data.get("configurations") or []
        if configurations:
            row["configurations"] = [dict(config) for config in configurations]
            row["dominant_configuration"] = dict(max(configurations, key=lambda item: item.get("weight", 0.0)))
        for source_key, target_key in (
            ("total_energy_correction_Eh", "root_total_energy_correction_Eh"),
            ("reference_energy_Eh", "root_reference_energy_Eh"),
            ("total_energy_Eh", "root_total_energy_Eh"),
        ):
            if result.get(source_key) is not None:
                row[target_key] = result[source_key]
        if total:
            row["pt2_total_energy_Eh"] = total.get("energy_Eh")
            row["ediag_index"] = total.get("ediag_index")
            row["ediag_value"] = total.get("ediag_value")
            row["ediag_unit"] = total.get("ediag_unit")
        if correction:
            row["transition_correction_Eh"] = correction.get("delta_energy_Eh")
            row["transition_correction_eV"] = correction.get("delta_energy_eV")
            row["transition_correction_cm-1"] = correction.get("delta_energy_cm-1")
        return row

    def _parse_raw_report_sections(self, lines: Sequence[str]) -> List[Dict[str, Any]]:
        """Preserve long ORCA report blocks whose layout carries scientific context."""
        sections: List[Dict[str, Any]] = []

        def add_block(key: str, title: str, start: int, end: int) -> None:
            if start == -1 or end == -1 or end <= start:
                return
            block_lines = [line.rstrip() for line in lines[start:end]]
            if not any(line.strip() for line in block_lines):
                return
            sections.append(
                {
                    "key": key,
                    "title": title,
                    "start_line": start + 1,
                    "end_line": end,
                    "n_lines": len(block_lines),
                    "lines": block_lines,
                }
            )

        qd_start = _find_exact(lines, "QD-NEVPT2 Results")
        qd_end = _find_exact(lines, "QD-NEVPT2 TOTAL ENERGIES", qd_start + 1) if qd_start != -1 else -1
        add_block("qd_nevpt2_van_vleck", "QD-NEVPT2 Van Vleck Results", qd_start, qd_end)

        spectra_start = _find_exact(lines, "CASSCF UV, CD spectra and dipole moments")
        spectra_end = _find_exact(lines, "CASSCF RELATIVISTIC PROPERTIES", spectra_start + 1) if spectra_start != -1 else -1
        add_block("casscf_uv_cd_spectra", "CASSCF / NEVPT2 UV, CD Spectra and Dipole Moments", spectra_start, spectra_end)

        rel_start = _find_exact(lines, "CASSCF RELATIVISTIC PROPERTIES")
        rel_end = _find_line_containing(lines, "EPR properties:", rel_start + 1) if rel_start != -1 else -1
        if rel_end == -1 and rel_start != -1:
            rel_end = _find_line_containing(lines, "ORCA TERMINATED NORMALLY", rel_start + 1)
        add_block("casscf_relativistic_properties", "CASSCF Relativistic Properties", rel_start, rel_end)

        return sections

    def _build_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build compact CASSCF metadata for job labels, reports, and comparisons."""
        summary: Dict[str, Any] = {
            "method": "CASSCF",
            "method_label": "CASSCF",
        }
        pt2_method = (data.get("pt2_settings") or {}).get("pt2_method")
        if isinstance(pt2_method, str) and pt2_method:
            summary["pt2_method"] = pt2_method
            summary["method_label"] = f"CASSCF/{pt2_method}"
        elif data.get("nevpt2"):
            summary["pt2_method"] = "NEVPT2"
            summary["method_label"] = "CASSCF/NEVPT2"

        system = data.get("system") or {}
        if system.get("number_of_active_electrons") is not None:
            summary["active_electrons"] = system.get("number_of_active_electrons")
        if system.get("number_of_active_orbitals") is not None:
            summary["active_orbitals"] = system.get("number_of_active_orbitals")
        active_range = system.get("active_orbital_range")
        if active_range:
            summary["active_orbital_range"] = active_range

        results = data.get("results") or {}
        if results.get("final_energy_Eh") is not None:
            summary["final_energy_Eh"] = results["final_energy_Eh"]
            summary["final_energy_eV"] = results.get("final_energy_eV")

        convergence = data.get("convergence") or {}
        if convergence:
            summary["converged"] = convergence.get("converged", False)
            summary["n_macro_iterations"] = convergence.get("n_macro_iterations")
            summary["converged_iteration"] = convergence.get("converged_iteration")
            final = convergence.get("final") or {}
            for key in (
                "gradient_norm",
                "max_gradient",
                "energy_gap_ext_act_Eh",
                "energy_gap_act_int_Eh",
                "rotation_label",
            ):
                if final.get(key) is not None:
                    summary[f"final_{key}"] = final[key]

        state_blocks = data.get("state_blocks") or []
        if state_blocks:
            summary["state_block_count"] = len(state_blocks)
            summary["state_count"] = sum(len(block.get("roots", [])) for block in state_blocks)
            summary["multiplicities"] = [block.get("multiplicity") for block in state_blocks]

        transitions = data.get("transition_energies") or {}
        if transitions.get("states"):
            summary["transition_count"] = len(transitions["states"])

        nevpt2 = data.get("nevpt2") or {}
        if nevpt2:
            summary["has_nevpt2"] = True
            if nevpt2.get("qd_nevpt2"):
                summary["has_qd_nevpt2"] = True
        return summary


def _matches_casscf(
    meta: Dict[str, Any],
    data: Dict[str, Any],
    context: Dict[str, Any],
    deltascf: Dict[str, Any],
    excited_state_optimization: Dict[str, Any],
) -> bool:
    """Return true when parsed data/context belongs to the CASSCF family."""
    del meta, deltascf, excited_state_optimization
    return bool(
        data.get("casscf")
        or str(context.get("reference_type", "")).upper() == "CASSCF"
        or str(context.get("hf_type", "")).upper() == "CASSCF"
    )


def _text_block(lines: Sequence[str] | str) -> str:
    """Return a fenced markdown text block for ORCA-formatted sections."""
    text = "\n".join(lines) if not isinstance(lines, str) else lines
    text = text.rstrip()
    if "```" in text:
        text = text.replace("```", "` ` `")
    return f"```text\n{text}\n```"


def _format_occupations(values: Sequence[Any], helpers: MarkdownRenderHelpers) -> str:
    """Format active-space occupation vectors for compact markdown tables."""
    if not values:
        return ""
    return " ".join(helpers.format_number(value, ".5f") for value in values)


def _format_configuration(config: Dict[str, Any], helpers: MarkdownRenderHelpers) -> str:
    """Format one CI configuration with coefficient/weight and occupation string."""
    config_index = config.get("configuration_index")
    if isinstance(config_index, int):
        index_text = f"{config_index:>6}"
    else:
        index_text = str(config_index)
    return (
        f"{helpers.format_number(config.get('weight'), '.5f')} "
        f"[{index_text}]: {config.get('occupation_string')}"
    )


def _format_configuration_digest(config: Dict[str, Any], helpers: MarkdownRenderHelpers) -> str:
    """Format one CI configuration as an unambiguous weight-to-occupation pair."""
    if not config:
        return "n/a"
    return (
        f"{helpers.format_number(config.get('weight'), '.5f')} -> "
        f"{config.get('occupation_string', '')}"
    ).strip()


def _configuration_digest(
    configurations: Sequence[Dict[str, Any]],
    helpers: MarkdownRenderHelpers,
    *,
    limit: Optional[int] = None,
) -> str:
    """Format a semicolon-separated list of CI weights and occupation strings."""
    if not configurations:
        return "n/a"
    shown = list(configurations if limit is None else configurations[:limit])
    text = "; ".join(_format_configuration_digest(config, helpers) for config in shown)
    if limit is not None and len(configurations) > limit:
        text += f"; +{len(configurations) - limit} more"
    return text


def _format_matrix_rows(matrix: Dict[str, Any], helpers: MarkdownRenderHelpers) -> List[tuple]:
    """Return markdown table rows for a parsed numeric matrix."""
    header = ("",) + tuple(str(column) for column in matrix.get("columns", []))
    rows: List[tuple] = [header]
    for row_label, values in zip(matrix.get("rows", []), matrix.get("matrix", [])):
        rows.append((row_label,) + tuple(helpers.format_number(value, ".6f") for value in values))
    return rows


def _matrix_markdown_block(title: str, matrix: Dict[str, Any], helpers: MarkdownRenderHelpers) -> str:
    """Render a compact parsed matrix block."""
    trace = matrix.get("trace")
    trace_text = ""
    if trace is not None:
        trace_text = f"trace={helpers.format_number(trace, '.6f')}\n\n"
    return f"{title}\n{trace_text}{helpers.make_table(_format_matrix_rows(matrix, helpers))}"


def _state_assignment_rows(
    assignments: Sequence[Dict[str, Any]],
    helpers: MarkdownRenderHelpers,
    *,
    energy_label: str,
    absolute_key: str,
    include_corrections: bool = False,
    include_root_corrections: bool = False,
    config_limit: Optional[int] = None,
) -> List[tuple]:
    """Build joined state/root/configuration rows for CASSCF and NEVPT2 reports."""
    header = ["state", "root", "mult", "DE Eh", "DE eV", "DE cm^-1", energy_label]
    if include_corrections:
        header.extend(["corr Eh", "corr eV", "corr cm^-1"])
    if include_root_corrections:
        header.extend(["root dE Eh", "E0 Eh"])
    header.extend(["dominant weight -> config", "weights -> configurations"])
    rows: List[tuple] = [tuple(header)]

    for assignment in assignments:
        row: List[Any] = [
            assignment.get("state"),
            assignment.get("root"),
            assignment.get("multiplicity"),
            helpers.format_number(assignment.get("delta_energy_Eh"), ".6f"),
            helpers.format_number(assignment.get("delta_energy_eV"), ".3f"),
            helpers.format_number(assignment.get("delta_energy_cm-1"), ".1f"),
            helpers.format_number(assignment.get(absolute_key), ".9f"),
        ]
        if include_corrections:
            row.extend(
                [
                    helpers.format_number(assignment.get("transition_correction_Eh"), ".6f"),
                    helpers.format_number(assignment.get("transition_correction_eV"), ".3f"),
                    helpers.format_number(assignment.get("transition_correction_cm-1"), ".1f"),
                ]
            )
        if include_root_corrections:
            row.extend(
                [
                    helpers.format_number(assignment.get("root_total_energy_correction_Eh"), ".9f"),
                    helpers.format_number(assignment.get("root_reference_energy_Eh"), ".9f"),
                ]
            )
        configurations = assignment.get("configurations") or []
        row.extend(
            [
                _format_configuration_digest(assignment.get("dominant_configuration") or {}, helpers),
                _configuration_digest(configurations, helpers, limit=config_limit),
            ]
        )
        rows.append(tuple(row))
    return rows


def _spectra_markdown_blocks(
    spectra: Sequence[Dict[str, Any]],
    helpers: MarkdownRenderHelpers,
    heading: str,
) -> List[str]:
    """Render parsed CASSCF/NEVPT2/QD-NEVPT2 absorption and CD spectra."""
    blocks: List[str] = []
    for section in spectra:
        title = section.get("title") or "CASSCF Spectrum"
        absorption = section.get("absorption") or []
        if absorption:
            rows = [("transition", "E eV", "E cm^-1", "nm", "fosc", "D2", "DX", "DY", "DZ")]
            for row in absorption:
                rows.append(
                    (
                        f"{row.get('from_label')} -> {row.get('to_label')}",
                        helpers.format_number(row.get("energy_eV"), ".6f"),
                        helpers.format_number(row.get("energy_cm-1"), ".1f"),
                        helpers.format_number(row.get("wavelength_nm"), ".1f"),
                        helpers.format_number(row.get("oscillator_strength"), ".9f"),
                        helpers.format_number(row.get("dipole_strength_au2"), ".5f"),
                        helpers.format_number(row.get("transition_dipole_x_au"), ".5f"),
                        helpers.format_number(row.get("transition_dipole_y_au"), ".5f"),
                        helpers.format_number(row.get("transition_dipole_z_au"), ".5f"),
                    )
                )
            blocks.append(f"{heading} {title} - Absorption\n{helpers.make_table(rows)}")

        cd_rows = section.get("cd") or []
        if cd_rows:
            rows = [("transition", "E eV", "E cm^-1", "nm", "R", "MX", "MY", "MZ")]
            for row in cd_rows:
                rows.append(
                    (
                        f"{row.get('from_label')} -> {row.get('to_label')}",
                        helpers.format_number(row.get("energy_eV"), ".6f"),
                        helpers.format_number(row.get("energy_cm-1"), ".1f"),
                        helpers.format_number(row.get("wavelength_nm"), ".1f"),
                        helpers.format_number(row.get("rotatory_strength_1e40_cgs"), ".5f"),
                        helpers.format_number(row.get("magnetic_dipole_x_au"), ".5f"),
                        helpers.format_number(row.get("magnetic_dipole_y_au"), ".5f"),
                        helpers.format_number(row.get("magnetic_dipole_z_au"), ".5f"),
                    )
                )
            blocks.append(f"{heading} {title} - CD\n{helpers.make_table(rows)}")
    return blocks


def _relativistic_markdown_blocks(
    relativistic: Dict[str, Any],
    helpers: MarkdownRenderHelpers,
    heading: str,
    render_options: RenderOptions,
) -> List[str]:
    """Render parsed QDPT levels, eigenvectors, g tensors, and D tensors."""
    blocks: List[str] = []
    basis = relativistic.get("basis") or {}
    if basis:
        rows = [("field", "values")]
        for key, label in (
            ("order", "Order"),
            ("ms_times_2", "MS *2"),
            ("multiplicities", "Mult"),
            ("irreps", "irrep"),
        ):
            if basis.get(key):
                rows.append((label, " ".join(str(value) for value in basis[key])))
        blocks.append(f"{heading} CASSCF Relativistic State Basis\n{helpers.make_table(rows)}")

    for block in relativistic.get("qdpt_blocks") or []:
        label = block.get("label", "QDPT")
        summary_rows = [
            ("label", label),
            ("mode", block.get("mode", "n/a")),
            ("lowest eigenvalue Eh", helpers.format_number(block.get("lowest_eigenvalue_Eh"), ".9f")),
            ("energy stabilization cm^-1", helpers.format_number(block.get("energy_stabilization_cm-1"), ".5f")),
            ("levels", len(block.get("levels") or [])),
            ("eigenvectors", len(block.get("eigenvectors") or [])),
            ("g matrices", len(block.get("g_matrices") or [])),
            ("D tensors", len(block.get("d_tensors") or [])),
        ]
        blocks.append(f"{heading} QDPT {label} Summary\n{helpers.make_table(summary_rows)}")

        levels = block.get("levels") or []
        if levels:
            shown_levels = levels if render_options.is_full else levels[:20]
            rows = [("state", "cm^-1", "eV", "Boltzmann population")]
            for level in shown_levels:
                rows.append(
                    (
                        level.get("state"),
                        helpers.format_number(level.get("energy_cm-1"), ".2f"),
                        helpers.format_number(level.get("energy_eV"), ".4f"),
                        helpers.format_number(level.get("boltzmann_population"), ".3e"),
                    )
                )
            blocks.append(f"{heading} QDPT {label} Relativistic Levels\n{helpers.make_table(rows)}")

        eigenvectors = block.get("eigenvectors") or []
        if eigenvectors:
            rows = [("state", "state cm^-1", "weight", "real", "imag", "block", "root", "spin", "Ms")]
            vector_limit = None if render_options.is_full else 20
            emitted = 0
            for vector in eigenvectors:
                for component in vector.get("components") or []:
                    if vector_limit is not None and emitted >= vector_limit:
                        break
                    rows.append(
                        (
                            vector.get("state"),
                            helpers.format_number(vector.get("energy_cm-1"), ".4f"),
                            helpers.format_number(component.get("weight"), ".6f"),
                            helpers.format_number(component.get("real"), ".6f"),
                            helpers.format_number(component.get("imaginary"), ".6f"),
                            component.get("block"),
                            component.get("root"),
                            component.get("spin"),
                            component.get("ms"),
                        )
                    )
                    emitted += 1
                if vector_limit is not None and emitted >= vector_limit:
                    break
            blocks.append(f"{heading} QDPT {label} Eigenvector Components\n{helpers.make_table(rows)}")

        g_matrices = block.get("g_matrices") or []
        if g_matrices:
            rows = [("matrix", "gx", "gy", "gz", "giso", "shift x", "shift y", "shift z", "shift iso")]
            for matrix in g_matrices:
                factors = matrix.get("g_factors") or {}
                shifts = matrix.get("g_shifts") or {}
                rows.append(
                    (
                        matrix.get("title"),
                        helpers.format_number(factors.get("x"), ".7f"),
                        helpers.format_number(factors.get("y"), ".7f"),
                        helpers.format_number(factors.get("z"), ".7f"),
                        helpers.format_number(factors.get("iso"), ".7f"),
                        helpers.format_number(shifts.get("x"), ".7f"),
                        helpers.format_number(shifts.get("y"), ".7f"),
                        helpers.format_number(shifts.get("z"), ".7f"),
                        helpers.format_number(shifts.get("iso"), ".7f"),
                    )
                )
            blocks.append(f"{heading} QDPT {label} g-Matrix Summary\n{helpers.make_table(rows)}")

        d_tensors = block.get("d_tensors") or []
        if d_tensors:
            rows = [("tensor", "D cm^-1", "E/D")]
            for tensor in d_tensors:
                rows.append(
                    (
                        tensor.get("title"),
                        helpers.format_number(tensor.get("D_cm-1"), ".6f"),
                        helpers.format_number(tensor.get("E_over_D"), ".6f"),
                    )
                )
            blocks.append(f"{heading} QDPT {label} Zero-Field Splitting Summary\n{helpers.make_table(rows)}")
    return blocks


def _state_configurations_text(state_blocks: Sequence[Dict[str, Any]], helpers: MarkdownRenderHelpers) -> str:
    """Render all CAS-SCF roots and configuration strings in ORCA-like text."""
    lines: List[str] = []
    for block in state_blocks:
        lines.append(
            f"CAS-SCF STATES FOR BLOCK {block.get('block_index')} "
            f"MULT={block.get('multiplicity')} NROOTS={block.get('nroots')}"
        )
        for root in block.get("roots", []):
            root_index = root.get("root")
            root_label = f"{root_index:>3}" if isinstance(root_index, int) else str(root_index)
            root_line = (
                f"ROOT {root_label}: E={helpers.format_number(root.get('energy_Eh'), '.10f')} Eh"
            )
            if root.get("relative_energy_eV") is not None:
                root_line += (
                    f"  {helpers.format_number(root.get('relative_energy_eV'), '.3f')} eV"
                    f"  {helpers.format_number(root.get('relative_energy_cm-1'), '.1f')} cm**-1"
                )
            lines.append(root_line)
            for config in root.get("configurations", []):
                lines.append(f"  {_format_configuration(config, helpers)}")
        lines.append("")
    return "\n".join(lines).strip()


def _nevpt2_state_results_text(
    states: Sequence[Dict[str, Any]],
    title: str,
    helpers: MarkdownRenderHelpers,
) -> str:
    """Render NEVPT2/QD-NEVPT2 per-root configurations and energy corrections."""
    lines: List[str] = [title]
    for state in states:
        lines.append("")
        lines.append(f"MULT {state.get('multiplicity')}, ROOT {state.get('root')}")
        for config in state.get("configurations", []):
            lines.append(f"  {_format_configuration(config, helpers)}")
        if state.get("total_energy_correction_Eh") is not None:
            lines.append(
                "  Total Energy Correction dE = "
                f"{helpers.format_number(state.get('total_energy_correction_Eh'), '.14f')} Eh"
            )
        if state.get("reference_energy_Eh") is not None:
            lines.append(
                "  Zero/Reference Energy E0 = "
                f"{helpers.format_number(state.get('reference_energy_Eh'), '.14f')} Eh"
            )
        if state.get("total_energy_Eh") is not None:
            lines.append(
                "  Total Energy E0+dE = "
                f"{helpers.format_number(state.get('total_energy_Eh'), '.14f')} Eh"
            )
        corrections = state.get("class_corrections") or {}
        if corrections:
            lines.append("  Class corrections:")
            for label, value in sorted(corrections.items()):
                lines.append(f"    {label}: {helpers.format_number(value, '.14f')} Eh")
    return "\n".join(lines).strip()


def _loewdin_mo_rows(
    active_mos: Dict[str, Any],
    helpers: MarkdownRenderHelpers,
    *,
    max_contributions: Optional[int],
) -> List[tuple]:
    rows = [("MO", "occ", "E Eh", "contributions")]
    for orbital in active_mos.get("orbitals") or []:
        contributions = orbital.get("contributions") or []
        shown = contributions if max_contributions is None else contributions[:max_contributions]
        suffix = ""
        if max_contributions is not None and len(contributions) > max_contributions:
            suffix = f"; +{len(contributions) - max_contributions} more"
        top = "; ".join(
            f"{item.get('atom_index')} {item.get('element')} {item.get('ao_label')} "
            f"{helpers.format_number(item.get('percent'), '.1f')}%"
            for item in shown
        ) + suffix
        rows.append(
            (
                orbital.get("index"),
                helpers.format_number(orbital.get("occupation"), ".5f"),
                helpers.format_number(orbital.get("energy_Eh"), ".5f"),
                top or "n/a",
            )
        )
    return rows


def _casscf_markdown_blocks(
    data: Dict[str, Any],
    helpers: MarkdownRenderHelpers,
    render_options: RenderOptions,
) -> List[str]:
    """Render the standalone CASSCF markdown sections."""
    casscf = data.get("casscf")
    if not casscf:
        return []

    h2 = "#" * (helpers.heading_level + 1)
    h3 = "#" * (helpers.heading_level + 2)
    blocks: List[str] = []

    summary = casscf.get("summary") or {}
    summary_rows = [
        ("method", summary.get("method_label", "CASSCF")),
        ("active electrons", summary.get("active_electrons", "—")),
        ("active orbitals", summary.get("active_orbitals", "—")),
        ("final E(CASSCF) Eh", helpers.format_number(summary.get("final_energy_Eh"), ".10f")),
        ("converged", "yes" if summary.get("converged") else "no"),
        ("macro iterations", summary.get("n_macro_iterations", "—")),
        ("final ||g||", helpers.format_number(summary.get("final_gradient_norm"), ".6g")),
        ("final Max(G)", helpers.format_number(summary.get("final_max_gradient"), ".6g")),
    ]
    blocks.append(f"{h2} CASSCF Active-Space Summary\n{helpers.make_table(summary_rows)}")

    convergence = casscf.get("convergence") or {}
    iterations = convergence.get("macro_iterations") or []
    if iterations:
        rows = [
            (
                "iter",
                "E(CAS) Eh",
                "DE Eh",
                "Ext-Act",
                "Act-Int",
                "N(occ)",
                "||g||",
                "Max(G)",
                "Rot",
                "update",
            )
        ]
        for iteration in iterations:
            update = iteration.get("orbital_update") or iteration.get("switch_step_to") or "n/a"
            row = [
                iteration.get("macro_iteration", ""),
                helpers.format_number(iteration.get("energy_Eh"), ".10f"),
                helpers.format_number(iteration.get("delta_energy_Eh"), ".3e"),
                helpers.format_number(iteration.get("energy_gap_ext_act_Eh"), ".3f"),
                helpers.format_number(iteration.get("energy_gap_act_int_Eh"), ".3f"),
                _format_occupations(iteration.get("active_occupations") or [], helpers),
                helpers.format_number(iteration.get("gradient_norm"), ".3e"),
                helpers.format_number(iteration.get("max_gradient"), ".3e"),
                iteration.get("rotation_label", "n/a"),
                update,
            ]
            if iteration.get("phase") == "final_printing":
                row = [f"**{value}**" if value not in ("", None) else value for value in row]
            rows.append(tuple(row))
        blocks.append(
            f"{h3} CASSCF Convergence History\n"
            "Final-printing iterations are bold.\n\n"
            f"{helpers.make_table(rows)}"
        )

    orbital_energies = casscf.get("orbital_energies") or []
    if orbital_energies:
        rows = [("MO", "occ", "E Eh", "E eV", "irrep")]
        window = casscf.get("orbital_energy_window") or {}
        selected = set(window.get("selected_indices") or [])
        orbitals_to_show = [
            orbital for orbital in orbital_energies if not selected or orbital.get("index") in selected
        ]
        for orbital in orbitals_to_show:
            rows.append(
                (
                    orbital.get("index"),
                    helpers.format_number(orbital.get("occupation"), ".5f"),
                    helpers.format_number(orbital.get("energy_Eh"), ".6f"),
                    helpers.format_number(orbital.get("energy_eV"), ".3f"),
                    orbital.get("irrep", ""),
                )
            )
        title = "CASSCF Orbital Energies"
        if window.get("selected_min_index") is not None:
            title += (
                f" ({window.get('selected_min_index')}-"
                f"{window.get('selected_max_index')}; +/-{window.get('window')} "
                f"around {window.get('anchor', 'frontier').replace('_', ' ')})"
            )
        blocks.append(f"{h3} {title}\n{helpers.make_table(rows)}")

    assignments = casscf.get("state_assignments") or []
    if assignments:
        rows = _state_assignment_rows(
            assignments,
            helpers,
            energy_label="E(CAS) Eh",
            absolute_key="casscf_root_energy_Eh",
            config_limit=None,
        )
        blocks.append(f"{h3} SA-CASSCF State Energies and Configurations\n{helpers.make_table(rows)}")

    state_blocks = casscf.get("state_blocks") or []
    if False and state_blocks and not assignments:
        rows = [("block", "mult", "root", "E Eh", "rel eV", "dominant configuration")]
        for block in state_blocks:
            for root in block.get("roots", []):
                dominant = root.get("dominant_configuration") or {}
                label = "—"
                if dominant:
                    label = (
                        f"{helpers.format_number(dominant.get('weight'), '.5f')} "
                        f"[{dominant.get('configuration_index')}] "
                        f"{dominant.get('occupation_string')}"
                    )
                rows.append(
                    (
                        block.get("block_index"),
                        block.get("multiplicity"),
                        root.get("root"),
                        helpers.format_number(root.get("energy_Eh"), ".10f"),
                        helpers.format_number(root.get("relative_energy_eV"), ".3f"),
                        label,
                    )
                )
        blocks.append(f"{h3} CAS-SCF States\n{helpers.make_table(rows)}")
        if render_options.is_full:
            blocks.append(
                f"{h3} CAS-SCF Root Configurations\n"
                + _text_block(_state_configurations_text(state_blocks, helpers))
            )

    transitions = casscf.get("transition_energies") or {}
    if transitions.get("states") and not assignments:
        rows = [("state", "root", "mult", "DE Eh", "DE eV", "DE cm^-1")]
        for state in transitions["states"]:
            rows.append(
                (
                    state.get("state"),
                    state.get("root"),
                    state.get("multiplicity"),
                    helpers.format_number(state.get("delta_energy_Eh"), ".6f"),
                    helpers.format_number(state.get("delta_energy_eV"), ".3f"),
                    helpers.format_number(state.get("delta_energy_cm-1"), ".1f"),
                )
            )
        blocks.append(f"{h3} SA-CASSCF Transition Energies\n{helpers.make_table(rows)}")

    matrix_sections = []
    for key, title in (("density_matrix", "Density Matrix"), ("spin_density_matrix", "Spin-Density Matrix")):
        matrix = casscf.get(key) or {}
        if matrix and matrix.get("n_rows", 0) <= 12 and matrix.get("n_columns", 0) <= 12:
            header = ("",) + tuple(str(column) for column in matrix.get("columns", []))
            rows = [header]
            for row_label, values in zip(matrix.get("rows", []), matrix.get("matrix", [])):
                rows.append((row_label,) + tuple(helpers.format_number(value, ".6f") for value in values))
            trace = matrix.get("trace")
            matrix_sections.append(
                f"{h3} {title}\ntrace={helpers.format_number(trace, '.6f')}\n\n{helpers.make_table(rows)}"
            )
    blocks.extend(matrix_sections)

    active_mos = casscf.get("loewdin_reduced_active_mos") or {}
    if active_mos.get("orbitals"):
        rows = _loewdin_mo_rows(
            active_mos,
            helpers,
            max_contributions=None if render_options.is_full else 5,
        )
        blocks.append(f"{h3} Loewdin Reduced Active MOs\n{helpers.make_table(rows)}")

    nevpt2 = casscf.get("nevpt2") or {}
    if nevpt2:
        blocks.extend(_nevpt2_markdown_blocks(nevpt2, helpers, h3, render_options))

    spectra = casscf.get("spectra") or []
    if spectra:
        blocks.extend(_spectra_markdown_blocks(spectra, helpers, h3))

    relativistic = casscf.get("relativistic") or {}
    if relativistic:
        blocks.extend(_relativistic_markdown_blocks(relativistic, helpers, h3, render_options))

    return blocks


def _nevpt2_markdown_blocks(
    nevpt2: Dict[str, Any],
    helpers: MarkdownRenderHelpers,
    heading: str,
    render_options: RenderOptions,
) -> List[str]:
    """Render NEVPT2 and QD-NEVPT2 markdown tables without hiding roots/configurations."""
    blocks: List[str] = []

    assignments = nevpt2.get("state_assignments") or []
    if assignments:
        rows = _state_assignment_rows(
            assignments,
            helpers,
            energy_label="E(NEVPT2) Eh",
            absolute_key="pt2_total_energy_Eh",
            include_corrections=True,
            config_limit=None,
        )
        blocks.append(f"{heading} NEVPT2 State Energies, Corrections, and Configurations\n{helpers.make_table(rows)}")

    qd = nevpt2.get("qd_nevpt2") or {}
    qd_assignments = qd.get("state_assignments") or []
    if qd_assignments:
        rows = _state_assignment_rows(
            qd_assignments,
            helpers,
            energy_label="E(QD-NEVPT2) Eh",
            absolute_key="pt2_total_energy_Eh",
            include_root_corrections=True,
            config_limit=None,
        )
        blocks.append(f"{heading} QD-NEVPT2 State Energies and Corrected Configurations\n{helpers.make_table(rows)}")

    for source, title in (
        (nevpt2.get("total_energies") or [], "NEVPT2 Total Energies"),
        (qd.get("total_energies") or [], "QD-NEVPT2 Total Energies"),
    ):
        if not source:
            continue
        rows = [("state", "root", "mult", "Energy Eh", "EDIAG", "EDIAG unit")]
        for row in source:
            rows.append(
                (
                    row.get("state"),
                    row.get("root"),
                    row.get("multiplicity"),
                    helpers.format_number(row.get("energy_Eh"), ".9f"),
                    helpers.format_number(row.get("ediag_value"), ".6f"),
                    row.get("ediag_unit", ""),
                )
            )
        blocks.append(f"{heading} {title}\n{helpers.make_table(rows)}")

    qd_state_results = qd.get("state_results") or []
    hamiltonians: Dict[Any, Dict[str, Any]] = {}
    for state in qd_state_results:
        mult = state.get("multiplicity")
        if mult not in hamiltonians and state.get("hamiltonian"):
            hamiltonians[mult] = state["hamiltonian"]
    for mult, matrix in hamiltonians.items():
        if matrix.get("n_rows", 0) <= 12 and matrix.get("n_columns", 0) <= 12:
            blocks.append(
                _matrix_markdown_block(
                    f"{heading} QD-NEVPT2 Total Hamiltonian (MULT {mult})",
                    matrix,
                    helpers,
                )
            )

    for key, title in (
        ("corrected_density_matrix", "QD-NEVPT2 Corrected Density Matrix"),
        ("corrected_spin_density_matrix", "QD-NEVPT2 Corrected Spin-Density Matrix"),
    ):
        matrix = qd.get(key) or {}
        if matrix and matrix.get("n_rows", 0) <= 12 and matrix.get("n_columns", 0) <= 12:
            blocks.append(_matrix_markdown_block(f"{heading} {title}", matrix, helpers))

    natural_orbitals = qd.get("state_specific_natural_orbitals") or []
    if natural_orbitals:
        rows = [("block", "mult", "root", "N(occ)", "file")]
        for row in natural_orbitals:
            rows.append(
                (
                    row.get("block_index"),
                    row.get("multiplicity"),
                    row.get("root"),
                    _format_occupations(row.get("natural_occupations") or [], helpers),
                    row.get("file", ""),
                )
            )
        blocks.append(f"{heading} QD-NEVPT2 State-Specific Natural Orbitals\n{helpers.make_table(rows)}")

    corrected_active_mos = qd.get("corrected_loewdin_reduced_active_mos") or {}
    if corrected_active_mos.get("orbitals"):
        rows = _loewdin_mo_rows(
            corrected_active_mos,
            helpers,
            max_contributions=None if render_options.is_full else 5,
        )
        blocks.append(f"{heading} QD-NEVPT2 Corrected Loewdin Reduced Active MOs\n{helpers.make_table(rows)}")

    return blocks


def _casscf_comparison_blocks(
    datasets: List[Dict[str, Any]],
    labels: List[str],
    helpers: MarkdownRenderHelpers,
    render_options: RenderOptions,
) -> List[str]:
    """Render one comparison-table block for CASSCF jobs."""
    del render_options
    if not any(dataset.get("casscf") for dataset in datasets):
        return []
    rows = [("", "active", "E(CASSCF) Eh", "conv", "iters", "||g||", "states", "NEVPT2")]
    for label, dataset in zip(labels, datasets):
        casscf = dataset.get("casscf") or {}
        summary = casscf.get("summary") or {}
        active = "—"
        if summary.get("active_electrons") is not None and summary.get("active_orbitals") is not None:
            active = f"({summary['active_electrons']},{summary['active_orbitals']})"
        rows.append(
            (
                label,
                active,
                helpers.format_number(summary.get("final_energy_Eh"), ".10f"),
                "yes" if summary.get("converged") else "no",
                summary.get("n_macro_iterations", "—"),
                helpers.format_number(summary.get("final_gradient_norm"), ".3e"),
                summary.get("state_count", "—"),
                "yes" if summary.get("has_nevpt2") else "no",
            )
        )
    return ["## CASSCF\n" + helpers.make_table(rows)]


def _configuration_summary_plain(configurations: Sequence[Dict[str, Any]]) -> str:
    """Return CI configurations as plain CSV-friendly weight->occupation pairs."""
    return ";".join(
        f"{config.get('weight')}->{config.get('occupation_string')}"
        for config in configurations
    )


def _assignment_csv_rows(assignments: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten joined state assignments for CSV export."""
    rows: List[Dict[str, Any]] = []
    for assignment in assignments:
        dominant = assignment.get("dominant_configuration") or {}
        rows.append(
            {
                "state": assignment.get("state"),
                "root": assignment.get("root"),
                "multiplicity": assignment.get("multiplicity"),
                "delta_energy_Eh": assignment.get("delta_energy_Eh"),
                "delta_energy_eV": assignment.get("delta_energy_eV"),
                "delta_energy_cm-1": assignment.get("delta_energy_cm-1"),
                "casscf_root_energy_Eh": assignment.get("casscf_root_energy_Eh"),
                "pt2_total_energy_Eh": assignment.get("pt2_total_energy_Eh"),
                "transition_correction_Eh": assignment.get("transition_correction_Eh"),
                "transition_correction_eV": assignment.get("transition_correction_eV"),
                "transition_correction_cm-1": assignment.get("transition_correction_cm-1"),
                "root_total_energy_correction_Eh": assignment.get("root_total_energy_correction_Eh"),
                "root_reference_energy_Eh": assignment.get("root_reference_energy_Eh"),
                "root_total_energy_Eh": assignment.get("root_total_energy_Eh"),
                "dominant_weight": dominant.get("weight"),
                "dominant_occupation_string": dominant.get("occupation_string"),
                "configurations": _configuration_summary_plain(assignment.get("configurations") or []),
            }
        )
    return rows


def _spectrum_csv_rows(section: Dict[str, Any], mode: str) -> List[Dict[str, Any]]:
    """Flatten one parsed spectrum section for CSV export."""
    rows: List[Dict[str, Any]] = []
    for row in section.get(mode) or []:
        base = {
            "spectrum": section.get("key"),
            "title": section.get("title"),
            "transition_from": row.get("from_label"),
            "transition_to": row.get("to_label"),
            "from_root": row.get("from_root"),
            "from_multiplicity": row.get("from_multiplicity"),
            "to_root": row.get("to_root"),
            "to_multiplicity": row.get("to_multiplicity"),
            "energy_eV": row.get("energy_eV"),
            "energy_cm-1": row.get("energy_cm-1"),
            "wavelength_nm": row.get("wavelength_nm"),
        }
        if mode == "absorption":
            base.update(
                {
                    "oscillator_strength": row.get("oscillator_strength"),
                    "dipole_strength_au2": row.get("dipole_strength_au2"),
                    "transition_dipole_x_au": row.get("transition_dipole_x_au"),
                    "transition_dipole_y_au": row.get("transition_dipole_y_au"),
                    "transition_dipole_z_au": row.get("transition_dipole_z_au"),
                }
            )
        else:
            base.update(
                {
                    "rotatory_strength_1e40_cgs": row.get("rotatory_strength_1e40_cgs"),
                    "magnetic_dipole_x_au": row.get("magnetic_dipole_x_au"),
                    "magnetic_dipole_y_au": row.get("magnetic_dipole_y_au"),
                    "magnetic_dipole_z_au": row.get("magnetic_dipole_z_au"),
                }
            )
        rows.append(base)
    return rows


def _write_casscf_csv_files(
    data: Dict[str, Any],
    directory: Path,
    stem: str,
    write_csv: Callable[[Path, str, List[Dict[str, Any]], List[str]], Path],
) -> List[Path]:
    """Write CASSCF CSV exports for summaries, histories, states, matrices, and NEVPT2."""
    casscf = data.get("casscf")
    if not casscf:
        return []

    files: List[Path] = []
    summary = casscf.get("summary") or {}
    if summary:
        files.append(
            write_csv(
                directory,
                f"{stem}_casscf_summary.csv",
                [summary],
                [
                    "method_label",
                    "active_electrons",
                    "active_orbitals",
                    "final_energy_Eh",
                    "converged",
                    "n_macro_iterations",
                    "converged_iteration",
                    "state_count",
                    "transition_count",
                    "has_nevpt2",
                    "has_qd_nevpt2",
                ],
            )
        )

    convergence_rows = []
    for iteration in (casscf.get("convergence") or {}).get("macro_iterations") or []:
        options = "; ".join(
            f"{option.get('name')}:{option.get('percent', '')}%"
            for option in iteration.get("orbital_options", [])
        )
        convergence_rows.append(
            {
                "macro_iteration": iteration.get("macro_iteration"),
                "energy_Eh": iteration.get("energy_Eh"),
                "delta_energy_Eh": iteration.get("delta_energy_Eh"),
                "energy_gap_ext_act_Eh": iteration.get("energy_gap_ext_act_Eh"),
                "energy_gap_act_int_Eh": iteration.get("energy_gap_act_int_Eh"),
                "active_occupations": ";".join(str(value) for value in iteration.get("active_occupations", [])),
                "gradient_norm": iteration.get("gradient_norm"),
                "max_gradient": iteration.get("max_gradient"),
                "rotation_label": iteration.get("rotation_label"),
                "orbital_update": iteration.get("orbital_update", ""),
                "density_convergence": iteration.get("density_convergence", ""),
                "active_fock_build": iteration.get("active_fock_build", ""),
                "orbital_options": options,
            }
        )
    if convergence_rows:
        files.append(
            write_csv(
                directory,
                f"{stem}_casscf_convergence.csv",
                convergence_rows,
                [
                    "macro_iteration",
                    "energy_Eh",
                    "delta_energy_Eh",
                    "energy_gap_ext_act_Eh",
                    "energy_gap_act_int_Eh",
                    "active_occupations",
                    "gradient_norm",
                    "max_gradient",
                    "rotation_label",
                    "orbital_update",
                    "density_convergence",
                    "active_fock_build",
                    "orbital_options",
                ],
            )
        )

    state_rows: List[Dict[str, Any]] = []
    config_rows: List[Dict[str, Any]] = []
    for block in casscf.get("state_blocks") or []:
        for root in block.get("roots", []):
            dominant = root.get("dominant_configuration") or {}
            state_rows.append(
                {
                    "block_index": block.get("block_index"),
                    "multiplicity": block.get("multiplicity"),
                    "root": root.get("root"),
                    "energy_Eh": root.get("energy_Eh"),
                    "relative_energy_eV": root.get("relative_energy_eV", ""),
                    "relative_energy_cm-1": root.get("relative_energy_cm-1", ""),
                    "dominant_weight": dominant.get("weight", ""),
                    "dominant_configuration_index": dominant.get("configuration_index", ""),
                    "dominant_occupation_string": dominant.get("occupation_string", ""),
                    "configuration_count": root.get("configuration_count", ""),
                }
            )
            for config in root.get("configurations", []):
                config_rows.append(
                    {
                        "block_index": block.get("block_index"),
                        "multiplicity": block.get("multiplicity"),
                        "root": root.get("root"),
                        "weight": config.get("weight"),
                        "configuration_index": config.get("configuration_index"),
                        "occupation_string": config.get("occupation_string"),
                    }
                )
    if state_rows:
        files.append(
            write_csv(
                directory,
                f"{stem}_casscf_states.csv",
                state_rows,
                [
                    "block_index",
                    "multiplicity",
                    "root",
                    "energy_Eh",
                    "relative_energy_eV",
                    "relative_energy_cm-1",
                    "dominant_weight",
                    "dominant_configuration_index",
                    "dominant_occupation_string",
                    "configuration_count",
                ],
            )
        )
    if config_rows:
        files.append(
            write_csv(
                directory,
                f"{stem}_casscf_configurations.csv",
                config_rows,
                ["block_index", "multiplicity", "root", "weight", "configuration_index", "occupation_string"],
            )
        )

    transitions = (casscf.get("transition_energies") or {}).get("states") or []
    if transitions:
        files.append(
            write_csv(
                directory,
                f"{stem}_casscf_transition_energies.csv",
                transitions,
                ["state", "root", "multiplicity", "delta_energy_Eh", "delta_energy_eV", "delta_energy_cm-1"],
            )
        )

    assignment_columns = [
        "state",
        "root",
        "multiplicity",
        "delta_energy_Eh",
        "delta_energy_eV",
        "delta_energy_cm-1",
        "casscf_root_energy_Eh",
        "pt2_total_energy_Eh",
        "transition_correction_Eh",
        "transition_correction_eV",
        "transition_correction_cm-1",
        "root_total_energy_correction_Eh",
        "root_reference_energy_Eh",
        "root_total_energy_Eh",
        "dominant_weight",
        "dominant_occupation_string",
        "configurations",
    ]
    casscf_assignments = _assignment_csv_rows(casscf.get("state_assignments") or [])
    if casscf_assignments:
        files.append(
            write_csv(
                directory,
                f"{stem}_casscf_state_assignments.csv",
                casscf_assignments,
                assignment_columns,
            )
        )

    for key, filename in (
        ("density_matrix", f"{stem}_casscf_density_matrix.csv"),
        ("spin_density_matrix", f"{stem}_casscf_spin_density_matrix.csv"),
    ):
        matrix = casscf.get(key) or {}
        rows = _matrix_to_rows(matrix)
        if rows:
            columns = ["row"] + [f"col_{column}" for column in matrix.get("columns", [])]
            files.append(write_csv(directory, filename, rows, columns))

    active_rows = []
    for orbital in (casscf.get("loewdin_reduced_active_mos") or {}).get("orbitals") or []:
        for contribution in orbital.get("contributions", []):
            active_rows.append(
                {
                    "orbital_index": orbital.get("index"),
                    "orbital_energy_Eh": orbital.get("energy_Eh"),
                    "orbital_occupation": orbital.get("occupation"),
                    **contribution,
                }
            )
    if active_rows:
        files.append(
            write_csv(
                directory,
                f"{stem}_casscf_loewdin_reduced_active_mos.csv",
                active_rows,
                ["orbital_index", "orbital_energy_Eh", "orbital_occupation", "atom_index", "element", "ao_label", "percent"],
            )
        )

    absorption_rows: List[Dict[str, Any]] = []
    cd_rows: List[Dict[str, Any]] = []
    for section in casscf.get("spectra") or []:
        absorption_rows.extend(_spectrum_csv_rows(section, "absorption"))
        cd_rows.extend(_spectrum_csv_rows(section, "cd"))
    if absorption_rows:
        files.append(
            write_csv(
                directory,
                f"{stem}_casscf_spectra_absorption.csv",
                absorption_rows,
                [
                    "spectrum",
                    "title",
                    "transition_from",
                    "transition_to",
                    "from_root",
                    "from_multiplicity",
                    "to_root",
                    "to_multiplicity",
                    "energy_eV",
                    "energy_cm-1",
                    "wavelength_nm",
                    "oscillator_strength",
                    "dipole_strength_au2",
                    "transition_dipole_x_au",
                    "transition_dipole_y_au",
                    "transition_dipole_z_au",
                ],
            )
        )
    if cd_rows:
        files.append(
            write_csv(
                directory,
                f"{stem}_casscf_spectra_cd.csv",
                cd_rows,
                [
                    "spectrum",
                    "title",
                    "transition_from",
                    "transition_to",
                    "from_root",
                    "from_multiplicity",
                    "to_root",
                    "to_multiplicity",
                    "energy_eV",
                    "energy_cm-1",
                    "wavelength_nm",
                    "rotatory_strength_1e40_cgs",
                    "magnetic_dipole_x_au",
                    "magnetic_dipole_y_au",
                    "magnetic_dipole_z_au",
                ],
            )
        )

    nevpt2 = casscf.get("nevpt2") or {}
    for rows, filename in (
        (nevpt2.get("total_energies") or [], f"{stem}_nevpt2_total_energies.csv"),
        ((nevpt2.get("qd_nevpt2") or {}).get("total_energies") or [], f"{stem}_qd_nevpt2_total_energies.csv"),
    ):
        if rows:
            files.append(
                write_csv(
                    directory,
                    filename,
                    rows,
                    ["state", "root", "multiplicity", "energy_Eh", "ediag_index", "ediag_value", "ediag_unit"],
                )
            )

    nevpt2_assignments = _assignment_csv_rows(nevpt2.get("state_assignments") or [])
    if nevpt2_assignments:
        files.append(
            write_csv(
                directory,
                f"{stem}_nevpt2_state_assignments.csv",
                nevpt2_assignments,
                assignment_columns,
            )
        )

    qd = nevpt2.get("qd_nevpt2") or {}
    qd_assignments = _assignment_csv_rows(qd.get("state_assignments") or [])
    if qd_assignments:
        files.append(
            write_csv(
                directory,
                f"{stem}_qd_nevpt2_state_assignments.csv",
                qd_assignments,
                assignment_columns,
            )
        )

    natural_rows = []
    for row in qd.get("state_specific_natural_orbitals") or []:
        natural_rows.append(
            {
                "block_index": row.get("block_index"),
                "multiplicity": row.get("multiplicity"),
                "root": row.get("root"),
                "natural_occupations": ";".join(str(value) for value in row.get("natural_occupations") or []),
                "file": row.get("file"),
            }
        )
    if natural_rows:
        files.append(
            write_csv(
                directory,
                f"{stem}_qd_nevpt2_natural_orbitals.csv",
                natural_rows,
                ["block_index", "multiplicity", "root", "natural_occupations", "file"],
            )
        )

    rel = casscf.get("relativistic") or {}
    level_rows = []
    vector_rows = []
    g_rows = []
    d_rows = []
    for block in rel.get("qdpt_blocks") or []:
        label = block.get("label")
        for level in block.get("levels") or []:
            level_rows.append({"qdpt_label": label, **level})
        for vector in block.get("eigenvectors") or []:
            for component in vector.get("components") or []:
                vector_rows.append(
                    {
                        "qdpt_label": label,
                        "state": vector.get("state"),
                        "state_energy_cm-1": vector.get("energy_cm-1"),
                        **component,
                    }
                )
        for matrix in block.get("g_matrices") or []:
            factors = matrix.get("g_factors") or {}
            shifts = matrix.get("g_shifts") or {}
            g_rows.append(
                {
                    "qdpt_label": label,
                    "title": matrix.get("title"),
                    "g_x": factors.get("x"),
                    "g_y": factors.get("y"),
                    "g_z": factors.get("z"),
                    "g_iso": factors.get("iso"),
                    "shift_x": shifts.get("x"),
                    "shift_y": shifts.get("y"),
                    "shift_z": shifts.get("z"),
                    "shift_iso": shifts.get("iso"),
                }
            )
        for tensor in block.get("d_tensors") or []:
            d_rows.append({"qdpt_label": label, **tensor})
    if level_rows:
        files.append(
            write_csv(
                directory,
                f"{stem}_casscf_qdpt_levels.csv",
                level_rows,
                ["qdpt_label", "state", "energy_cm-1", "energy_eV", "boltzmann_population"],
            )
        )
    if vector_rows:
        files.append(
            write_csv(
                directory,
                f"{stem}_casscf_qdpt_eigenvectors.csv",
                vector_rows,
                ["qdpt_label", "state", "state_energy_cm-1", "weight", "real", "imaginary", "block", "root", "spin", "ms"],
            )
        )
    if g_rows:
        files.append(
            write_csv(
                directory,
                f"{stem}_casscf_qdpt_g_matrices.csv",
                g_rows,
                ["qdpt_label", "title", "g_x", "g_y", "g_z", "g_iso", "shift_x", "shift_y", "shift_z", "shift_iso"],
            )
        )
    if d_rows:
        files.append(
            write_csv(
                directory,
                f"{stem}_casscf_qdpt_d_tensors.csv",
                d_rows,
                ["qdpt_label", "title", "D_cm-1", "E_over_D"],
            )
        )

    return files


CASSCF_MARKDOWN_SECTION_PLUGIN = MarkdownSectionPlugin(
    key="casscf",
    order=75,
    render_molecule_blocks=_casscf_markdown_blocks,
    render_comparison_blocks=_casscf_comparison_blocks,
)

CASSCF_CSV_SECTION_PLUGIN = CSVSectionPlugin(
    key="casscf",
    order=75,
    render_files=_write_casscf_csv_files,
)


PLUGIN_BUNDLE = PluginBundle(
    metadata=PluginMetadata(
        key="casscf",
        name="CASSCF / NEVPT2",
        short_help="Parse CASSCF convergence, active-space states, matrices, Loewdin active MOs, and NEVPT2 summaries.",
        description=(
            "Self-registering CASSCF parser section with bounded active/frontier "
            "Loewdin orbital-composition parsing and CASSCF-owned NEVPT2/QD-NEVPT2 summaries."
        ),
        docs_path="README.md",
        examples=(
            "orca_parser job.out --sections casscf --markdown --csv",
            "orca_parser job.out --sections casscf --casscf-orbital-window 30",
        ),
    ),
    parser_sections=(
        ParserSectionPlugin("casscf", CASSCFModule),
    ),
    parser_aliases=(
        ParserSectionAlias(
            name="casscf",
            section_keys=("casscf", "mulliken", "loewdin", "mayer", "hirshfeld", "mbis", "chelpg"),
        ),
        ParserSectionAlias(
            name="nevpt2",
            section_keys=("casscf", "mulliken", "loewdin", "mayer", "hirshfeld", "mbis", "chelpg"),
        ),
    ),
    calculation_families=(
        CalculationFamilyPlugin(
            family="casscf",
            default_calculation_label="CASSCF",
            matcher=_matches_casscf,
            comparison_order=45,
        ),
    ),
    markdown_sections=(
        CASSCF_MARKDOWN_SECTION_PLUGIN,
    ),
    csv_sections=(
        CASSCF_CSV_SECTION_PLUGIN,
    ),
    options=(
        PluginOption(
            dest="casscf_orbital_window",
            flags=("--casscf-orbital-window", "--casscf-orbital-energy-window"),
            help=(
                "Number of orbitals below and above the CASSCF active/frontier range "
                "to show in orbital-energy and Loewdin active-window tables (default: 30)."
            ),
            default=30,
            type=int,
            metavar="N",
            scope="casscf",
        ),
    ),
)
