"""Coupled-cluster, EOM-CCSD, and STEOM-CCSD parser modules.

The ground-state coupled-cluster section owns ORCA-MDCI wavefunction settings,
CC convergence, diagnostics, F12 corrections, perturbative triples, largest
amplitudes, and CC natural orbital occupations.  Population analysis and NBO
remain delegated to their shared modules through parser aliases.

The EOM/STEOM section owns the method-specific active-space/root context while
reusing the shared spectrum-table parser and shared NTO parser.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..job_family_registry import CalculationFamilyPlugin
from ..output.csv_section_registry import CSVSectionPlugin
from ..output.markdown_section_registry import MarkdownRenderHelpers, MarkdownSectionPlugin
from ..parser_section_plugin import ParserSectionAlias, ParserSectionPlugin
from ..plugin_bundle import PluginBundle, PluginMetadata
from ..render_options import RenderOptions
from .base import BaseModule
from .spectrum_parser import parse_spectrum_table
from .transition_orbitals import parse_natural_transition_orbitals


_FLOAT = r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?"
_KV_RE = re.compile(r"^\s*(?P<label>.+?)\s*(?:\.{2,}|:)\s*(?P<value>.+?)\s*$")
_ITER_ROW_RE = re.compile(
    rf"^\s*(?P<iteration>\d+)\s+"
    rf"(?P<total>{_FLOAT})\s+"
    rf"(?P<corr>{_FLOAT})\s+"
    rf"(?P<delta>{_FLOAT})\s+"
    rf"(?P<residual>{_FLOAT})\s+"
    rf"(?P<time>{_FLOAT})\s*$"
)
_ITER_SPLIT_ROW_RE = re.compile(
    rf"^\s*(?P<total>{_FLOAT})\s+"
    rf"(?P<corr>{_FLOAT})\s+"
    rf"(?P<delta>{_FLOAT})\s+"
    rf"(?P<residual>{_FLOAT})\s+"
    rf"(?P<time>{_FLOAT})\s*$"
)
_IROOT_RE = re.compile(
    rf"^\s*IROOT=\s*(?P<root>\d+)\s*:\s*"
    rf"(?P<energy_au>{_FLOAT})\s+au\s+"
    rf"(?P<energy_ev>{_FLOAT})\s+eV\s+"
    rf"(?P<energy_cm1>{_FLOAT})\s+cm\*\*-1",
    re.I,
)
_CIS_CONTRIBUTION_RE = re.compile(
    rf"^\s*(?P<from_orbital>\S+)\s*->\s*(?P<to_orbital>\S+)\s+"
    rf"(?P<weight>{_FLOAT})\s+\(\s*(?P<coefficient>{_FLOAT})\s*\)\s*$"
)
_AMPLITUDE_RE = re.compile(
    rf"^\s*(?P<amplitude>{_FLOAT})\s+(?P<from_orbital>\S+)\s*->\s*(?P<to_orbital>\S+)\s*$"
)
_DIPOLE_ROW_RE = re.compile(
    rf"^\s*IROOT=\s*(?P<root>\d+)\s*:\s*"
    rf"(?P<energy_ev>{_FLOAT})\s+"
    rf"(?P<dx>{_FLOAT})\s+"
    rf"(?P<dy>{_FLOAT})\s+"
    rf"(?P<dz>{_FLOAT})\s+"
    rf"(?P<magnitude>{_FLOAT})\s*$",
    re.I,
)
_OCCUPATION_RE = re.compile(rf"N\[\s*(?P<index>\d+)\s*\]\s*=\s*(?P<occupation>{_FLOAT})")


class CoupledClusterModule(BaseModule):
    """Parse ground-state CCSD/CCSD(T)/F12 data printed by ORCA-MDCI."""

    name = "coupled_cluster"

    def parse(self, lines: List[str]) -> Optional[Dict[str, Any]]:
        if not _looks_like_coupled_cluster_output(lines, self.context):
            return None

        data: Dict[str, Any] = {}

        mdci_input = _parse_mdci_input(self.context)
        if mdci_input:
            data["input"] = mdci_input

        wavefunction = _parse_wavefunction_type(lines)
        if wavefunction:
            data["wavefunction"] = wavefunction

        algorithm = _parse_named_settings(lines, "Algorithmic settings:")
        if algorithm:
            data["algorithmic_settings"] = algorithm

        dlpno = _parse_named_settings(lines, "DLPNO SETTINGS")
        if dlpno:
            data["dlpno_settings"] = dlpno

        iteration_blocks = _parse_cc_iteration_blocks(lines)
        if iteration_blocks:
            data["iteration_blocks"] = iteration_blocks
            data["iterations"] = iteration_blocks[-1]

        energy_blocks = _parse_coupled_cluster_energy_blocks(lines)
        if energy_blocks:
            data["energy_blocks"] = energy_blocks
            data["energy"] = energy_blocks[-1]
            data["final_energy"] = energy_blocks[-1]

        f12 = _parse_f12_correction(lines)
        if f12:
            data["f12_correction"] = f12

        triples = _parse_triples_correction(lines)
        if triples:
            data["triples_correction"] = triples

        amplitudes = _parse_largest_amplitudes(lines)
        if amplitudes:
            data["largest_amplitudes"] = amplitudes

        occupations = _parse_natural_orbital_occupation_blocks(lines)
        if occupations:
            data["natural_orbital_occupations"] = occupations[-1]
            if len(occupations) > 1:
                data["natural_orbital_occupation_blocks"] = occupations

        summary = _build_cc_summary(data)
        if summary:
            data["summary"] = summary

        return data or None


class EOMSTEOMModule(BaseModule):
    """Parse EOM-DLPNO-CCSD and STEOM-CCSD root, spectra, and NTO data."""

    name = "eom_steom"

    def parse(self, lines: List[str]) -> Optional[Dict[str, Any]]:
        if not _looks_like_eom_steom_output(lines, self.context):
            return None

        data: Dict[str, Any] = {}

        mdci_input = _parse_mdci_input(self.context)
        if mdci_input:
            data["input"] = mdci_input

        cis = _parse_cis_seed_results(lines)
        if cis:
            data["cis_seed_results"] = cis

        active_selection = _parse_active_space_selection(lines)
        if active_selection:
            data["active_space_selection"] = active_selection

        calculation_blocks = _parse_eom_steom_calculation_blocks(lines)
        if calculation_blocks:
            data["calculation_blocks"] = calculation_blocks
            eom_blocks = [
                block
                for block in calculation_blocks
                if block.get("calculation_kind") == "eom_ccsd"
            ]
            steom_blocks = [
                block
                for block in calculation_blocks
                if block.get("calculation_kind") == "steom_ccsd"
            ]
            if eom_blocks:
                data["eom_blocks"] = eom_blocks
            if steom_blocks:
                data["steom"] = steom_blocks[-1]
                data["steom_blocks"] = steom_blocks

        dipoles = _parse_unrelaxed_excited_state_dipoles(lines)
        if dipoles:
            data["unrelaxed_excited_state_dipoles"] = dipoles

        spectra, spectra_history = _parse_eom_steom_spectra(lines)
        if spectra:
            data["spectra"] = spectra
        if spectra_history:
            data["spectra_history"] = spectra_history

        nto_states = parse_natural_transition_orbitals(lines)
        if nto_states:
            _annotate_significant_nto_pairs(nto_states)
            data["nto_states"] = nto_states

        summary = _build_eom_steom_summary(data)
        if summary:
            data["summary"] = summary

        return data or None


def _looks_like_coupled_cluster_output(lines: Sequence[str], context: Dict[str, Any]) -> bool:
    tokens = {
        str(token).upper()
        for token in ((context.get("input_echo") or {}).get("bang_tokens") or [])
    }
    if any("CCSD" in token or "STEOM" in token or "EOM" in token for token in tokens):
        return True
    return any("ORCA-MATRIX DRIVEN CI" in line.upper() for line in lines)


def _looks_like_eom_steom_output(lines: Sequence[str], context: Dict[str, Any]) -> bool:
    tokens = {
        str(token).upper()
        for token in ((context.get("input_echo") or {}).get("bang_tokens") or [])
    }
    if any("STEOM" in token or "EOM" in token for token in tokens):
        return True
    return any(
        re.search(r"\b[RU]?HF\s+(?:EOM-DLPNO-CCSD|STEOM-CCSD)\s+CALCULATION\b", line, re.I)
        for line in lines
    )


def _parse_mdci_input(context: Dict[str, Any]) -> Dict[str, Any]:
    input_echo = context.get("input_echo") or {}
    mdci = (input_echo.get("blocks") or {}).get("mdci") or {}
    if not mdci:
        return {}
    result: Dict[str, Any] = {}
    settings = mdci.get("settings")
    if isinstance(settings, dict) and settings:
        result["settings"] = dict(settings)
    raw_lines = mdci.get("raw_lines")
    if isinstance(raw_lines, list) and raw_lines:
        result["line_count"] = len(raw_lines)
    return result


def _parse_wavefunction_type(lines: Sequence[str]) -> Dict[str, Any]:
    start = _find_header_line(lines, "Wavefunction type")
    if start == -1:
        return {}

    data: Dict[str, Any] = {}
    for line in lines[start + 1 : min(start + 80, len(lines))]:
        if "Algorithmic settings" in line:
            break
        stripped = line.strip()
        if not stripped:
            continue

        orbital_range = re.match(
            r"^(?P<label>Internal|Virtual)\s+Orbitals:\s+"
            r"(?P<start>\d+)\s+\.\.\.\s+(?P<end>\d+)"
            r"\s+\(\s*(?P<count>\d+)\s+MO'?s"
            r"(?:/\s*(?P<electrons>\d+)\s+electrons)?\s*\)",
            stripped,
            re.I,
        )
        if orbital_range:
            key = f"{orbital_range.group('label').lower()}_orbitals"
            entry = {
                "start": int(orbital_range.group("start")),
                "end": int(orbital_range.group("end")),
                "count": int(orbital_range.group("count")),
            }
            if orbital_range.group("electrons") is not None:
                entry["electrons"] = int(orbital_range.group("electrons"))
            data[key] = entry
            continue

        parsed = _parse_key_value_line(line)
        if parsed is None:
            continue
        label, value = parsed
        data[_normalize_label(label)] = _parse_scalar(value)

    return data


def _parse_named_settings(lines: Sequence[str], heading: str) -> Dict[str, Any]:
    start = _find_line(lines, heading)
    if start == -1:
        return {}

    data: Dict[str, Any] = {}
    started = False
    for line in lines[start + 1 : min(start + 260, len(lines))]:
        upper = line.upper()
        if started and (
            "COUPLED CLUSTER ITERATIONS" in upper
            or "COUPLED CLUSTER ENERGY" in upper
            or "F12 CORRECTION" in upper
            or "TRIPLES CORRECTION" in upper
            or "AUXILIARY BASIS" in upper
        ):
            break
        parsed = _parse_key_value_line(line)
        if parsed is None:
            continue
        label, value = parsed
        started = True
        data[_normalize_label(label)] = _parse_scalar(value)
    return data


def _parse_cc_iteration_blocks(lines: Sequence[str]) -> List[Dict[str, Any]]:
    starts = [
        idx
        for idx, line in enumerate(lines)
        if "COUPLED CLUSTER ITERATIONS" in line.upper()
    ]
    blocks: List[Dict[str, Any]] = []

    for block_index, start in enumerate(starts):
        rows: List[Dict[str, Any]] = []
        pending_iteration: Optional[int] = None
        converged = False
        j = start + 1
        while j < len(lines):
            line = lines[j]
            upper = line.upper()
            if j > start + 5 and "COUPLED CLUSTER ENERGY" in upper:
                break
            if j > start + 5 and "F12 CORRECTION" in upper:
                break
            if "COUPLED-CLUSTER ITERATIONS HAVE CONVERGED" in upper:
                converged = True
            if "COUPLED-CLUSTER ITERATIONS DID NOT CONVERGE" in upper:
                converged = False

            marker = re.match(r"^\s*(\d+)\s*=>", line)
            if marker:
                pending_iteration = int(marker.group(1))
                j += 1
                continue

            row = _match_iteration_row(line)
            if row is not None:
                rows.append(row)
                pending_iteration = None
                j += 1
                continue

            if pending_iteration is not None:
                split_match = _ITER_SPLIT_ROW_RE.match(line)
                if split_match:
                    rows.append(_iteration_row_from_match(split_match, pending_iteration))
                    pending_iteration = None

            j += 1

        if rows:
            blocks.append(
                {
                    "block_index": block_index,
                    "rows": rows,
                    "n_iterations": len(rows),
                    "converged": converged,
                    "final": rows[-1],
                }
            )

    return blocks


def _match_iteration_row(line: str) -> Optional[Dict[str, Any]]:
    match = _ITER_ROW_RE.match(line)
    if match is None:
        return None
    return _iteration_row_from_match(match, int(match.group("iteration")))


def _iteration_row_from_match(match: re.Match[str], iteration: int) -> Dict[str, Any]:
    return {
        "iteration": iteration,
        "total_energy_Eh": float(match.group("total")),
        "correlation_energy_Eh": float(match.group("corr")),
        "delta_energy_Eh": float(match.group("delta")),
        "residual": float(match.group("residual")),
        "time_s": float(match.group("time")),
    }


def _parse_coupled_cluster_energy_blocks(lines: Sequence[str]) -> List[Dict[str, Any]]:
    starts = [
        idx
        for idx, line in enumerate(lines)
        if line.strip().upper() == "COUPLED CLUSTER ENERGY"
    ]
    blocks: List[Dict[str, Any]] = []
    for block_index, start in enumerate(starts):
        block: Dict[str, Any] = {"block_index": block_index, "components": {}}
        for line in lines[start + 1 : min(start + 80, len(lines))]:
            upper = line.upper()
            if "LARGEST AMPLITUDES" in upper or "TRIPLES CORRECTION" in upper:
                break
            parsed = _parse_key_value_line(line)
            if parsed is None:
                continue
            label, value = parsed
            key = _cc_energy_key(label)
            parsed_value = _parse_first_float(value)
            if parsed_value is None:
                parsed_value = _parse_scalar(value)
            block["components"][key] = parsed_value
            if key:
                block[key] = parsed_value
        if len(block) > 2:
            blocks.append(block)
    return blocks


def _parse_f12_correction(lines: Sequence[str]) -> Dict[str, Any]:
    start = _find_header_line(lines, "RHF F12 CORRECTION")
    if start == -1:
        return {}
    data: Dict[str, Any] = {}
    for line in lines[start + 1 : min(start + 100, len(lines))]:
        upper = line.upper()
        if "COUPLED CLUSTER ENERGY" in upper or "TRIPLES CORRECTION" in upper:
            break
        parsed = _parse_key_value_line(line)
        if parsed is None:
            continue
        label, value = parsed
        key = _f12_key(label)
        if key is None:
            continue
        parsed_value = _parse_first_float(value)
        if parsed_value is not None:
            data[key] = parsed_value
    return data


def _parse_triples_correction(lines: Sequence[str]) -> Dict[str, Any]:
    start = _find_header_line(lines, "RHF TRIPLES CORRECTION")
    if start == -1:
        return {}
    data: Dict[str, Any] = {}
    for line in lines[start + 1 : min(start + 160, len(lines))]:
        upper = line.upper()
        if "NATURAL ORBITAL" in upper or "ORCA POPULATION ANALYSIS" in upper:
            break
        parsed = _parse_key_value_line(line)
        if parsed is None:
            continue
        label, value = parsed
        key = _triples_key(label)
        if key is None:
            continue
        parsed_value = _parse_first_float(value)
        if parsed_value is None:
            parsed_value = _parse_scalar(value)
        data[key] = parsed_value
    return data


def _parse_largest_amplitudes(lines: Sequence[str]) -> List[Dict[str, Any]]:
    start = _find_line(lines, "Largest amplitudes")
    if start == -1:
        return []
    rows: List[Dict[str, Any]] = []
    for line in lines[start + 1 : min(start + 120, len(lines))]:
        if not line.strip() and rows:
            break
        parts = line.split()
        if len(parts) < 5:
            continue
        numbers = [_safe_float(part) for part in parts]
        if all(value is None for value in numbers):
            continue
        if len(parts) >= 5 and _safe_float(parts[-1]) is not None:
            rows.append(
                {
                    "tokens": parts[:-1],
                    "amplitude": float(parts[-1]),
                }
            )
    return rows


def _parse_natural_orbital_occupation_blocks(lines: Sequence[str]) -> List[Dict[str, Any]]:
    starts = [
        idx
        for idx, line in enumerate(lines)
        if "NATURAL ORBITAL OCCUPATION NUMBERS" in line.upper()
    ]
    blocks: List[Dict[str, Any]] = []
    for block_index, start in enumerate(starts):
        rows: List[Dict[str, Any]] = []
        for line in lines[start + 1 : min(start + 180, len(lines))]:
            if rows and ("ORCA POPULATION ANALYSIS" in line.upper() or "TIMINGS" in line.upper()):
                break
            for match in _OCCUPATION_RE.finditer(line):
                rows.append(
                    {
                        "index": int(match.group("index")),
                        "occupation": float(match.group("occupation")),
                    }
                )
        if rows:
            blocks.append(
                {
                    "block_index": block_index,
                    "orbitals": rows,
                    "n_orbitals": len(rows),
                    "frontier_window": [
                        row
                        for row in rows
                        if 0.02 <= row.get("occupation", 0.0) <= 1.98
                    ],
                }
            )
    return blocks


def _parse_cis_seed_results(lines: Sequence[str]) -> Dict[str, Any]:
    start = _find_line(lines, "CIS RESULTS")
    if start == -1:
        return {}
    roots: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for line in lines[start + 1 : min(start + 220, len(lines))]:
        upper = line.upper()
        if current is not None and (
            "UNRELAXED EXCITED STATE DIPOLE MOMENTS" in upper
            or "STATE AVERAGED NATURAL ORBITALS" in upper
            or "RHF EOM-DLPNO-CCSD" in upper
        ):
            break
        root_match = _IROOT_RE.match(line)
        if root_match:
            current = _root_from_match(root_match)
            current["contributions"] = []
            roots.append(current)
            continue
        contribution_match = _CIS_CONTRIBUTION_RE.match(line)
        if contribution_match and current is not None:
            current["contributions"].append(
                {
                    "from_orbital": contribution_match.group("from_orbital"),
                    "to_orbital": contribution_match.group("to_orbital"),
                    "weight": float(contribution_match.group("weight")),
                    "coefficient": float(contribution_match.group("coefficient")),
                }
            )
    for root in roots:
        _annotate_dominant_contribution(root, "contributions", "weight")
    return {"roots": roots, "n_roots": len(roots)} if roots else {}


def _parse_active_space_selection(lines: Sequence[str]) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for line in lines:
        match = re.search(r"No\.?\s+of roots active in\s+(IP|EA)\s+calculation:\s*(\d+)", line, re.I)
        if match:
            data[f"{match.group(1).lower()}_active_roots"] = int(match.group(2))
    return data


def _parse_eom_steom_calculation_blocks(lines: Sequence[str]) -> List[Dict[str, Any]]:
    starts = [
        idx
        for idx, line in enumerate(lines)
        if "RHF EOM-DLPNO-CCSD CALCULATION" in line.upper()
        or "RHF STEOM-CCSD CALCULATION" in line.upper()
    ]
    blocks: List[Dict[str, Any]] = []
    for block_index, start in enumerate(starts):
        end = starts[block_index + 1] if block_index + 1 < len(starts) else len(lines)
        block_lines = lines[start:end]
        title = lines[start].strip()
        is_steom = "STEOM" in title.upper()
        block: Dict[str, Any] = {
            "block_index": block_index,
            "title": title,
            "calculation_kind": "steom_ccsd" if is_steom else "eom_ccsd",
            "settings": _parse_key_values_in_lines(block_lines[:120]),
        }
        eom_type = block["settings"].get("eom_type")
        if eom_type:
            block["eom_type"] = str(eom_type)
        elif is_steom:
            block["eom_type"] = "STEOM"

        histories = _parse_root_histories(block_lines)
        if histories:
            block["root_histories"] = histories

        roots = _parse_eom_steom_result_roots(block_lines)
        if roots:
            block["roots"] = roots
            block["n_roots"] = len(roots)

        final_active = _parse_final_active_roots(block_lines)
        if final_active:
            block.update(final_active)

        done = _parse_done_seconds(block_lines)
        if done is not None:
            block["wall_time_s"] = done

        blocks.append(block)
    return blocks


def _parse_root_histories(lines: Sequence[str]) -> List[Dict[str, Any]]:
    histories: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for line in lines:
        batch_match = re.search(r"BATCH\s+(\d+)\s+OF\s+(\d+)", line, re.I)
        if batch_match:
            current = {
                "kind": "batch",
                "batch": int(batch_match.group(1)),
                "n_batches": int(batch_match.group(2)),
                "rows": [],
            }
            histories.append(current)
            continue
        root_match = re.search(r"Solving Root No\s+(\d+)", line, re.I)
        if root_match:
            current = {"kind": "root", "root": int(root_match.group(1)), "rows": []}
            histories.append(current)
            continue
        if current is None:
            continue
        row = _parse_eom_iteration_line(line)
        if row:
            current["rows"].append(row)
        if "CONVERGED" in line.upper():
            current["converged"] = True
    for history in histories:
        rows = history.get("rows") or []
        if rows:
            history["n_iterations"] = len(rows)
            history["final"] = rows[-1]
    return histories


def _parse_eom_iteration_line(line: str) -> Optional[Dict[str, Any]]:
    parts = line.split()
    if len(parts) < 4 or not parts[0].isdigit():
        return None
    numeric = [_safe_float(part) for part in parts[1:4]]
    if any(value is None for value in numeric):
        return None
    return {
        "iteration": int(parts[0]),
        "delta_energy_Eh": float(numeric[0]),
        "residual": float(numeric[1]),
        "time_s": float(numeric[2]),
    }


def _parse_eom_steom_result_roots(lines: Sequence[str]) -> List[Dict[str, Any]]:
    roots: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    amplitude_target = "amplitudes"
    in_results = False
    for line in lines:
        upper = line.upper()
        if "EOM-CCSD RESULTS" in upper or "STEOM-CCSD RESULTS" in upper:
            in_results = True
            continue
        if not in_results:
            continue
        if "UNRELAXED EXCITED STATE DIPOLE MOMENTS" in upper or "SPECTRUM FOR" in upper:
            break

        root_match = _IROOT_RE.match(line)
        if root_match:
            current = _root_from_match(root_match)
            current["amplitudes"] = []
            roots.append(current)
            amplitude_target = "amplitudes"
            continue
        if current is None:
            continue
        if "AMPLITUDE" in upper and "CANONICAL" in upper:
            amplitude_target = "canonical_amplitudes"
            current.setdefault("canonical_amplitudes", [])
            continue
        if "AMPLITUDE" in upper and "EXCITATION" in upper:
            amplitude_target = "amplitudes"
            continue

        amplitude_match = _AMPLITUDE_RE.match(line)
        if amplitude_match:
            current.setdefault(amplitude_target, []).append(
                {
                    "amplitude": float(amplitude_match.group("amplitude")),
                    "from_orbital": amplitude_match.group("from_orbital"),
                    "to_orbital": amplitude_match.group("to_orbital"),
                    "excitation": (
                        f"{amplitude_match.group('from_orbital')} -> "
                        f"{amplitude_match.group('to_orbital')}"
                    ),
                }
            )
            continue

        gs_match = re.search(r"Ground state amplitude:\s*(%s)" % _FLOAT, line, re.I)
        if gs_match:
            current["ground_state_amplitude"] = float(gs_match.group(1))
            continue
        singles_match = re.search(r"Percentage singles character\s*=\s*(%s)" % _FLOAT, line, re.I)
        if singles_match:
            current["singles_character_percent"] = float(singles_match.group(1))
            continue
        active_match = re.search(r"Percentage Active Character\s*(%s)" % _FLOAT, line, re.I)
        if active_match:
            current["active_character_percent"] = float(active_match.group(1))
            continue
        if "WARNING" in upper or "HANDLE WITH CARE" in upper:
            current.setdefault("warnings", []).append(line.strip())

    for root in roots:
        _annotate_dominant_contribution(root, "amplitudes", "amplitude")
        _annotate_dominant_contribution(root, "canonical_amplitudes", "amplitude")
    return roots


def _parse_final_active_roots(lines: Sequence[str]) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for line in lines:
        match = re.search(r"Final no active\s+(IP|EA)\s+roots:\s*(\d+)", line, re.I)
        if match:
            data[f"final_active_{match.group(1).lower()}_roots"] = int(match.group(2))
    return data


def _parse_done_seconds(lines: Sequence[str]) -> Optional[float]:
    for line in lines:
        match = re.search(r"done in\s*\(\s*(%s)\s*\)" % _FLOAT, line, re.I)
        if match:
            return float(match.group(1))
    return None


def _parse_unrelaxed_excited_state_dipoles(lines: Sequence[str]) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []
    context_label = "unknown"
    for idx, line in enumerate(lines):
        upper = line.upper()
        if "CIS RESULTS" in upper:
            context_label = "cis_seed"
        elif "STEOM-CCSD RESULTS" in upper:
            context_label = "steom"
        elif "EOM-CCSD RESULTS" in upper:
            context_label = "eom"
        if "UNRELAXED EXCITED STATE DIPOLE MOMENTS" not in upper:
            continue

        rows: List[Dict[str, Any]] = []
        for table_line in lines[idx + 1 : min(idx + 80, len(lines))]:
            if rows and "----" in table_line:
                break
            row_match = _DIPOLE_ROW_RE.match(table_line)
            if row_match:
                rows.append(
                    {
                        "root": int(row_match.group("root")),
                        "energy_eV": float(row_match.group("energy_ev")),
                        "dx_au": float(row_match.group("dx")),
                        "dy_au": float(row_match.group("dy")),
                        "dz_au": float(row_match.group("dz")),
                        "magnitude_D": float(row_match.group("magnitude")),
                    }
                )
        if rows:
            sections.append({"context": context_label, "rows": rows, "n_rows": len(rows)})
    return sections


def _parse_eom_steom_spectra(
    lines: Sequence[str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    tables: List[Dict[str, Any]] = []
    contexts: Dict[str, Dict[str, Any]] = {}
    current_context_key = "unlabeled"
    current_context_label = "Unlabeled"

    i = 0
    while i < len(lines):
        context_match = re.search(r"SPECTRUM FOR\s+(.+?)\s*$", lines[i], re.I)
        if context_match:
            current_context_label = context_match.group(1).strip()
            current_context_key = _normalize_label(current_context_label)
            contexts.setdefault(
                current_context_key,
                {"label": current_context_label, "tables": {}},
            )
            i += 1
            continue

        table, next_index = parse_spectrum_table(
            lines,
            i,
            len(lines),
            stop_phrases=("NATURAL TRANSITION ORBITALS", "TIMINGS"),
            allow_soc_prefix=True,
        )
        if table is None:
            i = max(next_index, i + 1)
            continue
        if current_context_key == "unlabeled":
            contexts.setdefault(current_context_key, {"label": current_context_label, "tables": {}})
        table = dict(table)
        table["moment_context"] = current_context_key
        table["moment_context_label"] = current_context_label
        tables.append(table)
        contexts[current_context_key]["tables"][table["kind"]] = table
        i = max(next_index, i + 1)

    if not tables:
        return {}, []
    return {"tables": tables, "contexts": contexts, "table_count": len(tables)}, tables


def _annotate_significant_nto_pairs(states: List[Dict[str, Any]], threshold: float = 0.10) -> None:
    for state in states:
        pairs = list(state.get("pairs") or [])
        significant = []
        for pair in pairs:
            meets_threshold = (pair.get("occupation") or 0.0) >= threshold
            pair["meets_reporting_threshold"] = meets_threshold
            if meets_threshold:
                significant.append(pair)
        significant.sort(key=lambda item: item.get("occupation", 0.0), reverse=True)
        state["significant_occupation_threshold"] = threshold
        state["significant_pairs"] = significant
        state["significant_pair_count"] = len(significant)


def _root_from_match(match: re.Match[str]) -> Dict[str, Any]:
    return {
        "root": int(match.group("root")),
        "energy_Eh": float(match.group("energy_au")),
        "energy_eV": float(match.group("energy_ev")),
        "energy_cm-1": float(match.group("energy_cm1")),
    }


def _annotate_dominant_contribution(
    root: Dict[str, Any],
    collection_key: str,
    value_key: str,
) -> None:
    rows = root.get(collection_key) or []
    if not rows:
        return
    dominant = max(rows, key=lambda row: abs(row.get(value_key, 0.0)))
    if collection_key == "canonical_amplitudes":
        key = "dominant_canonical_amplitude"
    elif collection_key == "amplitudes":
        key = "dominant_amplitude"
    else:
        key = "dominant_contribution"
    root[key] = dict(dominant)


def _parse_key_values_in_lines(lines: Sequence[str]) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for line in lines:
        parsed = _parse_key_value_line(line)
        if parsed is None:
            continue
        label, value = parsed
        data[_normalize_label(label)] = _parse_scalar(value)
    return data


def _parse_key_value_line(line: str) -> Optional[Tuple[str, str]]:
    match = _KV_RE.match(line)
    if match is None:
        return None
    label = match.group("label").strip()
    value = match.group("value").strip()
    if not label or not value:
        return None
    if set(label) <= {"-", "=", "*"}:
        return None
    return label, value


def _parse_scalar(value: str) -> Any:
    cleaned = value.strip().strip('"').strip("'")
    upper = cleaned.upper()
    if upper in {"ON", "TRUE", "YES"}:
        return True
    if upper in {"OFF", "FALSE", "NO"}:
        return False
    if re.fullmatch(r"[-+]?\d+", cleaned):
        return int(cleaned)
    if re.fullmatch(_FLOAT, cleaned):
        return float(cleaned)
    first_float = _parse_first_float(cleaned)
    if first_float is not None and re.search(r"\b(EH|EV|SEC|%)\b", upper):
        return first_float
    return cleaned


def _parse_first_float(value: str) -> Optional[float]:
    match = re.search(_FLOAT, value)
    if not match:
        return None
    return float(match.group(0))


def _safe_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _find_line(lines: Sequence[str], pattern: str, start: int = 0) -> int:
    needle = pattern.upper()
    for idx in range(start, len(lines)):
        if needle in lines[idx].upper():
            return idx
    return -1


def _find_header_line(lines: Sequence[str], heading: str, start: int = 0) -> int:
    needle = heading.upper()
    for idx in range(start, len(lines)):
        stripped = lines[idx].strip().upper()
        if stripped.startswith(needle):
            return idx
    return -1


def _normalize_label(label: str) -> str:
    cleaned = label.strip().lower()
    cleaned = cleaned.replace("<s|s>**1/2", "singles_norm")
    cleaned = cleaned.replace("(t)", "triples")
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    return cleaned.strip("_")


def _cc_energy_key(label: str) -> str:
    compact = re.sub(r"\s+", " ", label.strip()).upper()
    mapping = {
        "E(0)": "reference_energy_Eh",
        "E(CORR)(STRONG-PAIRS)": "strong_pairs_correlation_energy_Eh",
        "E(CORR)(WEAK-PAIRS)": "weak_pairs_correlation_energy_Eh",
        "E(CORR)(CORRECTED)": "corrected_correlation_energy_Eh",
        "E(TOT)": "total_energy_Eh",
        "E(TOT)-BEFORE F12": "total_energy_before_f12_Eh",
        "E(TOT)-BEFORE F12 CORRECTIONS": "total_energy_before_f12_Eh",
        "E(TOT)-AFTER HF CORRECTION": "total_energy_after_hf_correction_Eh",
        "E(TOT)-AFTER ALL F12 CORRECTIONS": "total_energy_after_f12_Eh",
        "SINGLES NORM <S|S>**1/2": "singles_norm",
        "T1 DIAGNOSTIC": "t1_diagnostic",
    }
    return mapping.get(compact, _normalize_label(label))


def _f12_key(label: str) -> Optional[str]:
    compact = label.strip().upper()
    if compact.startswith("F12 CORRECTION"):
        return "f12_correlation_correction_Eh"
    if compact.startswith("HF CORRECTION") or compact.startswith("HARTREE-FOCK CORRECTION"):
        return "hf_correction_Eh"
    if compact.startswith("SUM OF F12 AND HF CORRECTION") or compact.startswith("SUM OF THE TWO"):
        return "sum_correction_Eh"
    if compact.startswith("TOTAL TIME"):
        return "time_s"
    return None


def _triples_key(label: str) -> Optional[str]:
    compact = re.sub(r"\s+", " ", label.strip()).upper()
    mapping = {
        "TRIPLES CORRECTION (T)": "triples_correction_Eh",
        "F = CCSD(WITH F12)/CCSD(WITHOUT F12)": "f12_ccsd_ratio",
        "F = CCSD (WITH F12)/ CCSD (WITHOUT F12)": "f12_ccsd_ratio",
        "SCALED TRIPLES CORRECTION (T)": "scaled_triples_correction_Eh",
        "FINAL CORRELATION ENERGY (WITHOUT F12)": "final_correlation_without_f12_Eh",
        "FINAL CORRELATION ENERGY (WITH F12)": "final_correlation_with_f12_Eh",
        "FINAL CORRELATION ENERGY (CCSD SCALED (T))": "final_correlation_scaled_triples_Eh",
        "F12-E(CCSD)": "f12_ccsd_energy_Eh",
        "F12-E(CCSD(T))": "f12_ccsdt_energy_Eh",
        "F12-ECCSD(T) WITH (T) SCALED THROUGH CCSD": "f12_ccsdt_scaled_triples_energy_Eh",
    }
    return mapping.get(compact)


def _build_cc_summary(data: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    wavefunction = data.get("wavefunction") or {}
    if wavefunction.get("correlation_treatment"):
        summary["correlation_treatment"] = wavefunction.get("correlation_treatment")
    if "perturbative_triple_excitations" in wavefunction:
        summary["has_perturbative_triples"] = bool(wavefunction.get("perturbative_triple_excitations"))
    if "calculation_of_f12_correction" in wavefunction:
        summary["has_f12_correction"] = bool(wavefunction.get("calculation_of_f12_correction"))

    iterations = data.get("iterations") or {}
    if iterations:
        summary["cc_iterations"] = iterations.get("n_iterations")
        summary["cc_converged"] = iterations.get("converged")

    energy = data.get("final_energy") or {}
    for key in ("total_energy_Eh", "total_energy_after_f12_Eh", "t1_diagnostic", "singles_norm"):
        if key in energy:
            summary[key] = energy[key]

    triples = data.get("triples_correction") or {}
    for key in ("f12_ccsdt_energy_Eh", "f12_ccsdt_scaled_triples_energy_Eh", "triples_correction_Eh"):
        if key in triples:
            summary[key] = triples[key]

    return summary


def _build_eom_steom_summary(data: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    active = data.get("active_space_selection") or {}
    summary.update(active)
    steom = data.get("steom") or {}
    if steom.get("roots"):
        roots = steom["roots"]
        summary["steom_root_count"] = len(roots)
        summary["lowest_steom_root_eV"] = min(root.get("energy_eV", float("inf")) for root in roots)
    if data.get("eom_blocks"):
        summary["eom_block_count"] = len(data["eom_blocks"])
    spectra = data.get("spectra") or {}
    if spectra.get("table_count"):
        summary["spectrum_table_count"] = spectra["table_count"]
    if data.get("nto_states"):
        summary["nto_state_count"] = len(data["nto_states"])
    return summary


def _format_number(value: Any, fmt: str = ".6g") -> str:
    if value is None or value == "":
        return "-"
    try:
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return str(value)


def _format_amplitudes(rows: Sequence[Dict[str, Any]], limit: int = 4) -> str:
    if not rows:
        return "-"
    ordered = sorted(rows, key=lambda row: abs(row.get("amplitude", 0.0)), reverse=True)
    pieces = [
        f"{_format_number(row.get('amplitude'), '.6f')} -> {row.get('excitation')}"
        for row in ordered[:limit]
    ]
    if len(rows) > limit:
        pieces.append(f"+{len(rows) - limit} more")
    return "; ".join(pieces)


def _format_nto_pairs(rows: Sequence[Dict[str, Any]], limit: int = 4) -> str:
    if not rows:
        return "-"
    ordered = sorted(rows, key=lambda row: row.get("occupation", 0.0), reverse=True)
    pieces = [
        f"{_format_number(row.get('occupation'), '.6f')} -> {row.get('from_orbital')} -> {row.get('to_orbital')}"
        for row in ordered[:limit]
    ]
    if len(rows) > limit:
        pieces.append(f"+{len(rows) - limit} more")
    return "; ".join(pieces)


def _coupled_cluster_markdown_blocks(
    data: Dict[str, Any],
    helpers: MarkdownRenderHelpers,
    render_options: RenderOptions,
) -> List[str]:
    del render_options
    cc = data.get("coupled_cluster")
    if not isinstance(cc, dict):
        return []

    h2 = "#" * (helpers.heading_level + 1)
    h3 = "#" * (helpers.heading_level + 2)
    blocks = [f"{h2} Coupled-Cluster"]

    summary = cc.get("summary") or {}
    if summary:
        rows = [("quantity", "value")]
        for key in (
            "correlation_treatment",
            "cc_converged",
            "cc_iterations",
            "total_energy_Eh",
            "total_energy_after_f12_Eh",
            "f12_ccsdt_energy_Eh",
            "f12_ccsdt_scaled_triples_energy_Eh",
            "t1_diagnostic",
            "singles_norm",
        ):
            if key in summary:
                rows.append((key.replace("_", " "), _format_number(summary[key])))
        blocks.append(helpers.make_table(rows))

    iterations = cc.get("iterations") or {}
    final = iterations.get("final") or {}
    if final:
        rows = [
            ("iter", "E(tot) Eh", "E(corr) Eh", "Delta Eh", "residual", "time s"),
            (
                str(final.get("iteration")),
                _format_number(final.get("total_energy_Eh"), ".10f"),
                _format_number(final.get("correlation_energy_Eh"), ".10f"),
                _format_number(final.get("delta_energy_Eh"), ".3e"),
                _format_number(final.get("residual"), ".3e"),
                _format_number(final.get("time_s"), ".2f"),
            ),
        ]
        blocks.append(f"{h3} CC Convergence\n{helpers.make_table(rows)}")

    triples = cc.get("triples_correction") or {}
    if triples:
        rows = [("quantity", "Eh")]
        for key in (
            "triples_correction_Eh",
            "scaled_triples_correction_Eh",
            "f12_ccsd_energy_Eh",
            "f12_ccsdt_energy_Eh",
            "f12_ccsdt_scaled_triples_energy_Eh",
        ):
            if key in triples:
                rows.append((key.replace("_", " "), _format_number(triples[key], ".10f")))
        blocks.append(f"{h3} Perturbative Triples / F12\n{helpers.make_table(rows)}")

    occupations = (cc.get("natural_orbital_occupations") or {}).get("frontier_window") or []
    if occupations:
        rows = [("MO", "occupation")]
        for row in occupations[:40]:
            rows.append((str(row.get("index")), _format_number(row.get("occupation"), ".6f")))
        blocks.append(f"{h3} CC Natural Occupation Window\n{helpers.make_table(rows)}")

    return ["\n\n".join(blocks)]


def _eom_steom_markdown_blocks(
    data: Dict[str, Any],
    helpers: MarkdownRenderHelpers,
    render_options: RenderOptions,
) -> List[str]:
    del render_options
    eom = data.get("eom_steom")
    if not isinstance(eom, dict):
        return []

    h2 = "#" * (helpers.heading_level + 1)
    h3 = "#" * (helpers.heading_level + 2)
    blocks = [f"{h2} EOM / STEOM-CCSD"]

    active = eom.get("active_space_selection") or {}
    if active:
        rows = [("quantity", "value")]
        for key, value in active.items():
            rows.append((key.replace("_", " "), str(value)))
        blocks.append(f"{h3} Active-Space Root Selection\n{helpers.make_table(rows)}")

    for block in eom.get("eom_blocks") or []:
        roots = block.get("roots") or []
        if not roots:
            continue
        label = block.get("eom_type") or block.get("title") or "EOM"
        rows = [("root", "Eh", "eV", "cm^-1", "% singles", "dominant amplitude", "amplitudes")]
        for root in roots:
            dominant = root.get("dominant_amplitude") or {}
            rows.append(
                (
                    str(root.get("root")),
                    _format_number(root.get("energy_Eh"), ".9f"),
                    _format_number(root.get("energy_eV"), ".3f"),
                    _format_number(root.get("energy_cm-1"), ".1f"),
                    _format_number(root.get("singles_character_percent"), ".2f"),
                    (
                        f"{_format_number(dominant.get('amplitude'), '.6f')} -> "
                        f"{dominant.get('excitation')}"
                        if dominant
                        else "-"
                    ),
                    _format_amplitudes(root.get("amplitudes") or []),
                )
            )
        blocks.append(f"{h3} {label} Roots\n{helpers.make_table(rows)}")

    steom = eom.get("steom") or {}
    if steom.get("roots"):
        rows = [("root", "Eh", "eV", "cm^-1", "% active", "ground amp", "amplitudes", "canonical amplitudes")]
        for root in steom["roots"]:
            rows.append(
                (
                    str(root.get("root")),
                    _format_number(root.get("energy_Eh"), ".9f"),
                    _format_number(root.get("energy_eV"), ".3f"),
                    _format_number(root.get("energy_cm-1"), ".1f"),
                    _format_number(root.get("active_character_percent"), ".2f"),
                    _format_number(root.get("ground_state_amplitude"), ".6f"),
                    _format_amplitudes(root.get("amplitudes") or []),
                    _format_amplitudes(root.get("canonical_amplitudes") or []),
                )
            )
        blocks.append(f"{h3} STEOM Roots\n{helpers.make_table(rows)}")

    spectra = (eom.get("spectra") or {}).get("tables") or []
    if spectra:
        rows = [("context", "kind", "transitions")]
        for table in spectra:
            rows.append(
                (
                    table.get("moment_context_label", ""),
                    table.get("kind", ""),
                    str(table.get("transition_count", len(table.get("transitions") or []))),
                )
            )
        blocks.append(f"{h3} STEOM Spectra\n{helpers.make_table(rows)}")

    ntos = eom.get("nto_states") or []
    if ntos:
        rows = [("state", "eV", "cm^-1", "file", "significant NTO pairs")]
        for state in ntos:
            rows.append(
                (
                    str(state.get("state")),
                    _format_number(state.get("energy_eV"), ".3f"),
                    _format_number(state.get("energy_cm1"), ".1f"),
                    state.get("output_file", "-"),
                    _format_nto_pairs(state.get("significant_pairs") or state.get("pairs") or []),
                )
            )
        blocks.append(f"{h3} Natural Transition Orbitals\n{helpers.make_table(rows)}")

    return ["\n\n".join(blocks)]


def _write_coupled_cluster_csv_files(
    data: Dict[str, Any],
    directory: Path,
    stem: str,
    write_csv: Callable[[Path, str, List[Dict[str, Any]], List[str]], Path],
) -> List[Path]:
    cc = data.get("coupled_cluster")
    if not isinstance(cc, dict):
        return []
    files: List[Path] = []

    rows = (cc.get("iterations") or {}).get("rows") or []
    if rows:
        files.append(
            write_csv(
                directory,
                f"{stem}_coupled_cluster_iterations.csv",
                rows,
                ["iteration", "total_energy_Eh", "correlation_energy_Eh", "delta_energy_Eh", "residual", "time_s"],
            )
        )

    summary = cc.get("summary") or {}
    if summary:
        files.append(
            write_csv(
                directory,
                f"{stem}_coupled_cluster_summary.csv",
                [{"quantity": key, "value": value} for key, value in summary.items()],
                ["quantity", "value"],
            )
        )

    occ_rows = (cc.get("natural_orbital_occupations") or {}).get("orbitals") or []
    if occ_rows:
        files.append(
            write_csv(
                directory,
                f"{stem}_coupled_cluster_natural_occupations.csv",
                occ_rows,
                ["index", "occupation"],
            )
        )

    return files


def _write_eom_steom_csv_files(
    data: Dict[str, Any],
    directory: Path,
    stem: str,
    write_csv: Callable[[Path, str, List[Dict[str, Any]], List[str]], Path],
) -> List[Path]:
    eom = data.get("eom_steom")
    if not isinstance(eom, dict):
        return []
    files: List[Path] = []

    root_rows: List[Dict[str, Any]] = []
    for block in eom.get("calculation_blocks") or []:
        for root in block.get("roots") or []:
            root_rows.append(
                {
                    "block_index": block.get("block_index"),
                    "calculation_kind": block.get("calculation_kind"),
                    "eom_type": block.get("eom_type"),
                    "root": root.get("root"),
                    "energy_Eh": root.get("energy_Eh"),
                    "energy_eV": root.get("energy_eV"),
                    "energy_cm-1": root.get("energy_cm-1"),
                    "singles_character_percent": root.get("singles_character_percent"),
                    "active_character_percent": root.get("active_character_percent"),
                    "ground_state_amplitude": root.get("ground_state_amplitude"),
                    "dominant_amplitude": (root.get("dominant_amplitude") or {}).get("amplitude"),
                    "dominant_excitation": (root.get("dominant_amplitude") or {}).get("excitation"),
                }
            )
    if root_rows:
        files.append(
            write_csv(
                directory,
                f"{stem}_eom_steom_roots.csv",
                root_rows,
                [
                    "block_index",
                    "calculation_kind",
                    "eom_type",
                    "root",
                    "energy_Eh",
                    "energy_eV",
                    "energy_cm-1",
                    "singles_character_percent",
                    "active_character_percent",
                    "ground_state_amplitude",
                    "dominant_amplitude",
                    "dominant_excitation",
                ],
            )
        )

    amp_rows: List[Dict[str, Any]] = []
    for block in eom.get("calculation_blocks") or []:
        for root in block.get("roots") or []:
            for amplitude_kind in ("amplitudes", "canonical_amplitudes"):
                for amp in root.get(amplitude_kind) or []:
                    amp_rows.append(
                        {
                            "block_index": block.get("block_index"),
                            "calculation_kind": block.get("calculation_kind"),
                            "eom_type": block.get("eom_type"),
                            "root": root.get("root"),
                            "amplitude_kind": amplitude_kind,
                            **amp,
                        }
                    )
    if amp_rows:
        files.append(
            write_csv(
                directory,
                f"{stem}_eom_steom_amplitudes.csv",
                amp_rows,
                ["block_index", "calculation_kind", "eom_type", "root", "amplitude_kind", "amplitude", "from_orbital", "to_orbital", "excitation"],
            )
        )

    spectrum_rows: List[Dict[str, Any]] = []
    for table in (eom.get("spectra") or {}).get("tables") or []:
        for transition in table.get("transitions") or []:
            spectrum_rows.append(
                {
                    "moment_context": table.get("moment_context"),
                    "moment_context_label": table.get("moment_context_label"),
                    "kind": table.get("kind"),
                    **transition,
                }
            )
    if spectrum_rows:
        files.append(
            write_csv(
                directory,
                f"{stem}_eom_steom_spectra.csv",
                spectrum_rows,
                [
                    "moment_context",
                    "moment_context_label",
                    "kind",
                    "from_state_label",
                    "to_state_label",
                    "energy_eV",
                    "energy_cm1",
                    "wavelength_nm",
                    "oscillator_strength",
                    "dipole_strength_au2",
                    "velocity_strength_au2",
                    "rotatory_strength_cgs",
                    "dx_au",
                    "dy_au",
                    "dz_au",
                    "px_au",
                    "py_au",
                    "pz_au",
                    "mx_au",
                    "my_au",
                    "mz_au",
                ],
            )
        )

    nto_rows: List[Dict[str, Any]] = []
    for state in eom.get("nto_states") or []:
        for pair in state.get("pairs") or []:
            nto_rows.append(
                {
                    "state": state.get("state"),
                    "energy_eV": state.get("energy_eV"),
                    "energy_cm1": state.get("energy_cm1"),
                    "output_file": state.get("output_file"),
                    **pair,
                }
            )
    if nto_rows:
        files.append(
            write_csv(
                directory,
                f"{stem}_eom_steom_nto.csv",
                nto_rows,
                ["state", "energy_eV", "energy_cm1", "output_file", "from_orbital", "to_orbital", "from_index", "from_spin", "to_index", "to_spin", "occupation"],
            )
        )

    return files


def _matches_coupled_cluster(
    meta: Dict[str, Any],
    data: Dict[str, Any],
    context: Dict[str, Any],
    deltascf: Dict[str, Any],
    excited_state_optimization: Dict[str, Any],
) -> bool:
    del meta, context, deltascf, excited_state_optimization
    return isinstance(data.get("coupled_cluster"), dict) and not isinstance(data.get("eom_steom"), dict)


def _matches_eom_steom(
    meta: Dict[str, Any],
    data: Dict[str, Any],
    context: Dict[str, Any],
    deltascf: Dict[str, Any],
    excited_state_optimization: Dict[str, Any],
) -> bool:
    del meta, context, deltascf, excited_state_optimization
    return isinstance(data.get("eom_steom"), dict)


COUPLED_CLUSTER_MARKDOWN_SECTION_PLUGIN = MarkdownSectionPlugin(
    key="coupled_cluster",
    order=72,
    render_molecule_blocks=_coupled_cluster_markdown_blocks,
)

EOM_STEOM_MARKDOWN_SECTION_PLUGIN = MarkdownSectionPlugin(
    key="eom_steom",
    order=73,
    render_molecule_blocks=_eom_steom_markdown_blocks,
)

COUPLED_CLUSTER_CSV_SECTION_PLUGIN = CSVSectionPlugin(
    key="coupled_cluster",
    order=72,
    render_files=_write_coupled_cluster_csv_files,
)

EOM_STEOM_CSV_SECTION_PLUGIN = CSVSectionPlugin(
    key="eom_steom",
    order=73,
    render_files=_write_eom_steom_csv_files,
)


PLUGIN_BUNDLE = PluginBundle(
    metadata=PluginMetadata(
        key="coupled_cluster",
        name="CCSD / CCSD(T) / EOM / STEOM",
        short_help=(
            "Parse ORCA-MDCI coupled-cluster convergence, CC diagnostics, F12/triples corrections, "
            "EOM/STEOM roots, shared-format spectra, and NTOs."
        ),
        description=(
            "Self-registering coupled-cluster parser sections. Population analysis and NBO are "
            "delegated through aliases, spectra are parsed by the shared spectrum parser, and NTOs "
            "are parsed by the shared transition-orbital helper."
        ),
        docs_path="README.md",
        examples=(
            "orca_parser ccsdt.out --sections ccsdt --markdown --csv",
            "orca_parser steom.out --sections steom --markdown --csv",
        ),
    ),
    parser_sections=(
        ParserSectionPlugin("coupled_cluster", CoupledClusterModule),
        ParserSectionPlugin("eom_steom", EOMSTEOMModule),
    ),
    parser_aliases=(
        ParserSectionAlias(
            name="cc",
            section_keys=("coupled_cluster", "mulliken", "loewdin", "mayer", "hirshfeld", "mbis", "chelpg", "nbo", "dipole"),
        ),
        ParserSectionAlias(
            name="ccsd",
            section_keys=("coupled_cluster", "mulliken", "loewdin", "mayer", "hirshfeld", "mbis", "chelpg", "nbo", "dipole"),
        ),
        ParserSectionAlias(
            name="ccsdt",
            section_keys=("coupled_cluster", "mulliken", "loewdin", "mayer", "hirshfeld", "mbis", "chelpg", "nbo", "dipole"),
        ),
        ParserSectionAlias(
            name="coupled_cluster",
            section_keys=("coupled_cluster", "mulliken", "loewdin", "mayer", "hirshfeld", "mbis", "chelpg", "nbo", "dipole"),
        ),
        ParserSectionAlias(
            name="eom",
            section_keys=("coupled_cluster", "eom_steom", "mulliken", "loewdin", "mayer", "hirshfeld", "mbis", "chelpg", "nbo", "dipole"),
        ),
        ParserSectionAlias(
            name="steom",
            section_keys=("coupled_cluster", "eom_steom", "mulliken", "loewdin", "mayer", "hirshfeld", "mbis", "chelpg", "nbo", "dipole"),
        ),
        ParserSectionAlias(
            name="eom_steom",
            section_keys=("coupled_cluster", "eom_steom", "mulliken", "loewdin", "mayer", "hirshfeld", "mbis", "chelpg", "nbo", "dipole"),
        ),
    ),
    calculation_families=(
        CalculationFamilyPlugin(
            family="eom_steom",
            default_calculation_label="EOM / STEOM-CCSD",
            matcher=_matches_eom_steom,
            electronic_state_kind="excited_state",
            comparison_order=42,
        ),
        CalculationFamilyPlugin(
            family="coupled_cluster",
            default_calculation_label="Coupled Cluster",
            matcher=_matches_coupled_cluster,
            comparison_order=43,
        ),
    ),
    markdown_sections=(
        COUPLED_CLUSTER_MARKDOWN_SECTION_PLUGIN,
        EOM_STEOM_MARKDOWN_SECTION_PLUGIN,
    ),
    csv_sections=(
        COUPLED_CLUSTER_CSV_SECTION_PLUGIN,
        EOM_STEOM_CSV_SECTION_PLUGIN,
    ),
)
