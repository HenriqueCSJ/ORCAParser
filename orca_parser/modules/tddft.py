"""
Module for TDDFT/CIS excited-state parsing.

Extracts:
  - %tddft / %cis input settings from the echoed input block
  - Excited-state manifolds and per-state excitation amplitudes
  - Natural transition orbital (NTO) blocks
  - Absorption and CD spectra tables
  - CIS/TDDFT total-energy summary
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseModule


class TDDFTModule(BaseModule):
    """Parses TDDFT/TDA/CIS excited-state output."""

    name = "tddft"

    _INPUT_BLOCK_START_RE = re.compile(r"^\|\s*\d+>\s*%(tddft|cis)\b", re.I)
    _INPUT_LINE_RE = re.compile(
        r"^\|\s*\d+>\s*([A-Za-z][\w-]*)\s*(?:=)?\s*(.+?)\s*$"
    )
    _INPUT_END_RE = re.compile(r"^\|\s*\d+>\s*end\b", re.I)

    _EXCITED_STATES_HEADER_RE = re.compile(
        r"^\s*(?P<method>(?:TD-DFT|CIS)(?:/[A-Za-z0-9-]+)?)\s+EXCITED STATES"
        r"(?:\s+\((?P<manifold>[^)]+)\))?\s*$",
        re.I,
    )
    _PRINT_THRESHOLD_RE = re.compile(
        r"printed if larger than\s+([-\d.eE+]+)", re.I
    )
    _STATE_RE = re.compile(
        r"^\s*STATE\s+(?P<state>\d+)\s*:\s*E=\s*(?P<energy_au>[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s+au"
        r"\s+(?P<energy_ev>[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s+eV"
        r"\s+(?P<energy_cm1>[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s+cm\*\*-1"
        r"(?:\s+<S\*\*2>\s*=\s*(?P<s_squared>[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?))?"
        r"(?:\s+Sym:\s*(?P<symmetry>\S+))?"
        r"(?:\s+Mult\s+(?P<multiplicity>\d+))?",
        re.I,
    )
    _EXCITATION_RE = re.compile(
        r"^\s*(?P<from_orbital>\d+[A-Za-z]+)\s*->\s*(?P<to_orbital>\d+[A-Za-z]+)\s*:\s*"
        r"(?P<weight>[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)"
        r"(?:\s*\(c=\s*(?P<coefficient>[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\))?",
        re.I,
    )

    _NTO_HEADER_RE = re.compile(
        r"^\s*NATURAL TRANSITION ORBITALS FOR STATE\s+(?P<state>\d+)\s*$",
        re.I,
    )
    _NTO_FILE_RE = re.compile(
        r"Natural Transition Orbitals were saved in\s+(\S+)", re.I
    )
    _NTO_THRESHOLD_RE = re.compile(
        r"Threshold for printing occupation numbers\s+([-\d.eE+]+)", re.I
    )
    _NTO_ENERGY_RE = re.compile(
        r"^\s*E=\s*(?P<energy_au>[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s+au"
        r"\s+(?P<energy_ev>[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s+eV"
        r"\s+(?P<energy_cm1>[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s+cm\*\*-1",
        re.I,
    )
    _NTO_PAIR_RE = re.compile(
        r"^\s*(?P<from_orbital>\d+[A-Za-z]+)\s*->\s*(?P<to_orbital>\d+[A-Za-z]+)\s*:\s*n=\s*"
        r"(?P<occupation>[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)",
        re.I,
    )

    _CENTER_OF_MASS_RE = re.compile(
        r"Center of mass\s*=\s*\(\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\s*\)",
        re.I,
    )
    _TOTAL_ENERGY_HEADER = "CIS/TD-DFT TOTAL ENERGY"
    _TOTAL_SCF_RE = re.compile(r"E\(SCF\)\s*=\s*([-\d.]+)\s+Eh", re.I)
    _TOTAL_DELTA_RE = re.compile(
        r"DE\(([^)]+)\)\s*=\s*([-\d.]+)\s+Eh(?:\s+\(Root\s+(\d+)\))?",
        re.I,
    )
    _TOTAL_ENERGY_RE = re.compile(r"E\(tot\)\s*=\s*([-\d.]+)\s+Eh", re.I)
    _MAX_MEMORY_RE = re.compile(
        r"Maximum memory used throughout the entire .*?:\s*([-\d.]+)\s*MB",
        re.I,
    )

    _SPECTRUM_TABLES: Dict[str, Tuple[str, List[str]]] = {
        "ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS": (
            "absorption_electric_dipole",
            [
                "oscillator_strength",
                "dipole_strength_au2",
                "dx_au",
                "dy_au",
                "dz_au",
            ],
        ),
        "ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS": (
            "absorption_velocity_dipole",
            [
                "oscillator_strength",
                "velocity_strength_au2",
                "px_au",
                "py_au",
                "pz_au",
            ],
        ),
        "CD SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS": (
            "cd_electric_dipole",
            [
                "rotatory_strength_cgs",
                "mx_au",
                "my_au",
                "mz_au",
            ],
        ),
        "CD SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS": (
            "cd_velocity_dipole",
            [
                "rotatory_strength_cgs",
                "mx_au",
                "my_au",
                "mz_au",
            ],
        ),
    }

    def parse(self, lines):
        data: Dict[str, Any] = {}

        input_blocks = self._parse_input_blocks(lines)
        if input_blocks:
            data["input_blocks"] = input_blocks
            if len(input_blocks) == 1:
                data["input"] = input_blocks[0]

        excited_state_blocks = self._parse_excited_state_blocks(lines)
        if excited_state_blocks:
            data["excited_state_blocks"] = excited_state_blocks
            data["excited_states"] = [
                state
                for block in excited_state_blocks
                for state in block.get("states", [])
            ]
            data["final_excited_state_block"] = excited_state_blocks[-1]

        nto_states = self._parse_nto_states(lines)
        if nto_states:
            data["nto_states"] = nto_states

        spectra, spectra_history = self._parse_spectra(lines)
        if spectra:
            data["spectra"] = spectra
        if spectra_history:
            data["spectra_history"] = spectra_history

        total_energy_blocks = self._parse_total_energy_blocks(lines)
        if total_energy_blocks:
            data["total_energy_blocks"] = total_energy_blocks
            data["total_energy"] = total_energy_blocks[-1]

        if not data:
            return None

        summary = self._build_summary(
            input_blocks=input_blocks,
            excited_state_blocks=excited_state_blocks,
            nto_states=nto_states,
            spectra=spectra,
            total_energy_blocks=total_energy_blocks,
        )
        if summary:
            data["summary"] = summary

        return data

    def _parse_input_blocks(self, lines: List[str]) -> List[Dict[str, Any]]:
        input_end = self.find_line(lines, "****END OF INPUT****")
        if input_end == -1:
            input_end = len(lines)

        blocks: List[Dict[str, Any]] = []
        i = 0
        while i < input_end:
            start_match = self._INPUT_BLOCK_START_RE.match(lines[i])
            if not start_match:
                i += 1
                continue

            block_name = start_match.group(1).lower()
            block: Dict[str, Any] = {
                "block": block_name,
                "settings": {},
                "raw_lines": [],
            }

            j = i + 1
            while j < input_end:
                line = lines[j]
                if self._INPUT_END_RE.match(line):
                    break
                block["raw_lines"].append(line.strip())
                line_match = self._INPUT_LINE_RE.match(line)
                if line_match:
                    key = line_match.group(1).lower()
                    value = self._parse_scalar(line_match.group(2))
                    block["settings"][key] = value
                j += 1

            blocks.append(block)
            i = j + 1

        return blocks

    def _parse_excited_state_blocks(self, lines: List[str]) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []
        block_index = 0
        i = 0

        while i < len(lines):
            header_match = self._EXCITED_STATES_HEADER_RE.match(lines[i].strip())
            if not header_match:
                i += 1
                continue

            block: Dict[str, Any] = {
                "block_index": block_index,
                "method": header_match.group("method").upper(),
                "manifold": header_match.group("manifold"),
                "header": lines[i].strip(),
                "states": [],
            }

            print_threshold = self._find_print_threshold(lines, i + 1, i + 8)
            if print_threshold is not None:
                block["print_threshold"] = print_threshold

            current_state: Optional[Dict[str, Any]] = None
            j = i + 1
            while j < len(lines):
                line = lines[j]
                stripped = line.strip()

                if j != i and self._EXCITED_STATES_HEADER_RE.match(stripped):
                    break
                if self._is_excited_state_boundary(stripped):
                    break

                state_match = self._STATE_RE.match(line)
                if state_match:
                    if current_state is not None:
                        block["states"].append(current_state)
                    current_state = self._build_state_dict(
                        state_match=state_match,
                        block=block,
                        order_in_block=len(block["states"]) + 1,
                    )
                    j += 1
                    continue

                transition_match = self._EXCITATION_RE.match(line)
                if transition_match and current_state is not None:
                    current_state["transitions"].append(
                        self._build_excitation_dict(transition_match)
                    )

                j += 1

            if current_state is not None:
                block["states"].append(current_state)

            if block["states"]:
                block["state_count"] = len(block["states"])
                blocks.append(block)
                block_index += 1

            i = j

        return blocks

    def _parse_nto_states(self, lines: List[str]) -> List[Dict[str, Any]]:
        nto_states: List[Dict[str, Any]] = []
        i = 0

        while i < len(lines):
            header_match = self._NTO_HEADER_RE.match(lines[i].strip())
            if not header_match:
                i += 1
                continue

            state_data: Dict[str, Any] = {
                "state": int(header_match.group("state")),
                "pairs": [],
            }

            j = i + 1
            while j < len(lines):
                line = lines[j]
                stripped = line.strip()

                if j != i and self._NTO_HEADER_RE.match(stripped):
                    break
                if self._is_nto_boundary(stripped):
                    break

                file_match = self._NTO_FILE_RE.search(line)
                if file_match:
                    state_data["output_file"] = file_match.group(1)

                threshold_match = self._NTO_THRESHOLD_RE.search(line)
                if threshold_match:
                    state_data["print_threshold"] = self.safe_float(
                        threshold_match.group(1)
                    )

                energy_match = self._NTO_ENERGY_RE.match(line)
                if energy_match:
                    state_data["energy_au"] = float(energy_match.group("energy_au"))
                    state_data["energy_eV"] = float(energy_match.group("energy_ev"))
                    state_data["energy_cm1"] = float(
                        energy_match.group("energy_cm1")
                    )
                    state_data["wavelength_nm"] = self._wavelength_from_cm1(
                        state_data["energy_cm1"]
                    )

                pair_match = self._NTO_PAIR_RE.match(line)
                if pair_match:
                    pair = self._build_orbital_pair_dict(
                        from_orbital=pair_match.group("from_orbital"),
                        to_orbital=pair_match.group("to_orbital"),
                    )
                    pair["occupation"] = float(pair_match.group("occupation"))
                    state_data["pairs"].append(pair)

                j += 1

            state_data["pair_count"] = len(state_data["pairs"])
            nto_states.append(state_data)
            i = j

        return nto_states

    def _parse_spectra(
        self, lines: List[str]
    ) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
        spectra: Dict[str, Dict[str, Any]] = {}
        spectra_history: List[Dict[str, Any]] = []
        last_center_of_mass: Optional[Dict[str, float]] = None

        i = 0
        while i < len(lines):
            com_match = self._CENTER_OF_MASS_RE.search(lines[i])
            if com_match:
                last_center_of_mass = {
                    "x": float(com_match.group(1)),
                    "y": float(com_match.group(2)),
                    "z": float(com_match.group(3)),
                }

            title = lines[i].strip().upper()
            if title not in self._SPECTRUM_TABLES:
                i += 1
                continue

            kind, value_fields = self._SPECTRUM_TABLES[title]
            table: Dict[str, Any] = {
                "kind": kind,
                "title": lines[i].strip(),
                "transitions": [],
            }
            if last_center_of_mass is not None:
                table["center_of_mass"] = dict(last_center_of_mass)

            started = False
            j = i + 1
            while j < len(lines):
                stripped = lines[j].strip()
                upper = stripped.upper()

                if not stripped:
                    if started:
                        break
                    j += 1
                    continue

                if upper in self._SPECTRUM_TABLES and j != i:
                    break
                if self._TOTAL_ENERGY_HEADER in upper:
                    break

                row = self._parse_spectrum_row(lines[j], value_fields)
                if row is not None:
                    started = True
                    table["transitions"].append(row)

                j += 1

            if table["transitions"]:
                table["transition_count"] = len(table["transitions"])
                spectra[kind] = table
                spectra_history.append(dict(table))

            i = j

        return spectra, spectra_history

    def _parse_total_energy_blocks(self, lines: List[str]) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []
        header_indices = self.find_all_lines(lines, self._TOTAL_ENERGY_HEADER)

        for block_index, idx in enumerate(header_indices):
            block: Dict[str, Any] = {"block_index": block_index}

            for line in lines[idx : idx + 20]:
                scf_match = self._TOTAL_SCF_RE.search(line)
                if scf_match:
                    block["scf_energy_Eh"] = float(scf_match.group(1))

                delta_match = self._TOTAL_DELTA_RE.search(line)
                if delta_match:
                    block["excitation_method"] = delta_match.group(1)
                    block["delta_energy_Eh"] = float(delta_match.group(2))
                    if delta_match.group(3) is not None:
                        block["root"] = int(delta_match.group(3))

                total_match = self._TOTAL_ENERGY_RE.search(line)
                if total_match:
                    block["total_energy_Eh"] = float(total_match.group(1))

            for line in lines[idx : idx + 40]:
                memory_match = self._MAX_MEMORY_RE.search(line)
                if memory_match:
                    block["maximum_memory_MB"] = float(memory_match.group(1))
                    break

            if len(block) > 1:
                blocks.append(block)

        return blocks

    def _build_summary(
        self,
        input_blocks: List[Dict[str, Any]],
        excited_state_blocks: List[Dict[str, Any]],
        nto_states: List[Dict[str, Any]],
        spectra: Dict[str, Dict[str, Any]],
        total_energy_blocks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}

        if input_blocks:
            first_block = input_blocks[0]
            summary["input_block"] = first_block.get("block")
            for key in ("nroots", "tda", "donto"):
                if key in first_block.get("settings", {}):
                    summary[key] = first_block["settings"][key]

        if excited_state_blocks:
            final_block = excited_state_blocks[-1]
            summary["excited_state_block_count"] = len(excited_state_blocks)
            summary["final_method"] = final_block.get("method")
            if final_block.get("manifold") is not None:
                summary["final_manifold"] = final_block.get("manifold")
            summary["final_state_count"] = len(final_block.get("states", []))

        if nto_states:
            summary["nto_state_count"] = len(nto_states)

        if spectra:
            summary["spectrum_tables"] = sorted(spectra)

        if total_energy_blocks:
            final_energy = total_energy_blocks[-1]
            if "root" in final_energy:
                summary["final_root"] = final_energy["root"]
            if "delta_energy_Eh" in final_energy:
                summary["delta_energy_Eh"] = final_energy["delta_energy_Eh"]

        return summary

    def _find_print_threshold(
        self, lines: List[str], start: int, stop: int
    ) -> Optional[float]:
        for j in range(start, min(stop, len(lines))):
            match = self._PRINT_THRESHOLD_RE.search(lines[j])
            if match:
                return self.safe_float(match.group(1))
        return None

    def _build_state_dict(
        self,
        state_match: re.Match,
        block: Dict[str, Any],
        order_in_block: int,
    ) -> Dict[str, Any]:
        energy_cm1 = float(state_match.group("energy_cm1"))
        state_data: Dict[str, Any] = {
            "block_index": block["block_index"],
            "method": block.get("method"),
            "manifold": block.get("manifold"),
            "order_in_block": order_in_block,
            "state": int(state_match.group("state")),
            "energy_au": float(state_match.group("energy_au")),
            "energy_eV": float(state_match.group("energy_ev")),
            "energy_cm1": energy_cm1,
            "wavelength_nm": self._wavelength_from_cm1(energy_cm1),
            "transitions": [],
        }

        if state_match.group("s_squared") is not None:
            state_data["s_squared"] = float(state_match.group("s_squared"))
        if state_match.group("symmetry") is not None:
            state_data["symmetry"] = state_match.group("symmetry")
        if state_match.group("multiplicity") is not None:
            state_data["multiplicity"] = int(state_match.group("multiplicity"))

        return state_data

    def _build_excitation_dict(self, match: re.Match) -> Dict[str, Any]:
        transition = self._build_orbital_pair_dict(
            from_orbital=match.group("from_orbital"),
            to_orbital=match.group("to_orbital"),
        )
        transition["weight"] = float(match.group("weight"))
        if match.group("coefficient") is not None:
            transition["coefficient"] = float(match.group("coefficient"))
        return transition

    def _build_orbital_pair_dict(
        self, from_orbital: str, to_orbital: str
    ) -> Dict[str, Any]:
        from_index, from_spin = self._split_orbital_label(from_orbital)
        to_index, to_spin = self._split_orbital_label(to_orbital)
        return {
            "from_orbital": from_orbital,
            "to_orbital": to_orbital,
            "from_index": from_index,
            "from_spin": from_spin,
            "to_index": to_index,
            "to_spin": to_spin,
        }

    def _parse_spectrum_row(
        self, line: str, value_fields: List[str]
    ) -> Optional[Dict[str, Any]]:
        match = re.match(
            r"^\s*(?P<from_state>\S+)\s*->\s*(?P<to_state>\S+)\s+(?P<rest>.+)$",
            line,
        )
        if not match:
            return None

        parts = match.group("rest").split()
        expected = 3 + len(value_fields)
        if len(parts) != expected:
            return None

        values = [self.safe_float(part) for part in parts]
        if any(value is None for value in values):
            return None

        row: Dict[str, Any] = {
            "from_state_label": match.group("from_state"),
            "to_state_label": match.group("to_state"),
            "energy_eV": float(values[0]),
            "energy_cm1": float(values[1]),
            "wavelength_nm": float(values[2]),
        }
        for field, value in zip(value_fields, values[3:]):
            row[field] = float(value)

        from_root, from_label = self._split_state_label(row["from_state_label"])
        to_root, to_label = self._split_state_label(row["to_state_label"])
        if from_root is not None:
            row["from_root"] = from_root
        if from_label is not None:
            row["from_state_suffix"] = from_label
        if to_root is not None:
            row["to_root"] = to_root
        if to_label is not None:
            row["to_state_suffix"] = to_label

        return row

    def _is_excited_state_boundary(self, stripped: str) -> bool:
        upper = stripped.upper()
        return (
            upper.startswith("NATURAL TRANSITION ORBITALS FOR STATE")
            or upper.startswith("STORING AMPLITUDES IN GBW FILE")
            or "ABSORPTION SPECTRUM VIA TRANSITION" in upper
            or "CD SPECTRUM VIA TRANSITION" in upper
            or self._TOTAL_ENERGY_HEADER in upper
        )

    def _is_nto_boundary(self, stripped: str) -> bool:
        upper = stripped.upper()
        return (
            "ABSORPTION SPECTRUM VIA TRANSITION" in upper
            or "CD SPECTRUM VIA TRANSITION" in upper
            or self._TOTAL_ENERGY_HEADER in upper
            or upper.startswith("TD-DFT/TDA-EXCITATION SPECTRA")
        )

    def _split_orbital_label(self, label: str) -> Tuple[Optional[int], Optional[str]]:
        match = re.match(r"(?P<index>\d+)(?P<spin>[A-Za-z]+)$", label.strip())
        if not match:
            return None, None
        return int(match.group("index")), match.group("spin").lower()

    def _split_state_label(self, label: str) -> Tuple[Optional[int], Optional[str]]:
        match = re.match(r"(?P<root>\d+)-(?P<label>.+)$", label.strip())
        if not match:
            return None, None
        return int(match.group("root")), match.group("label")

    def _parse_scalar(self, value: str) -> Any:
        cleaned = value.strip().strip('"').strip("'")
        lowered = cleaned.lower()

        if lowered in {"true", "yes", "on"}:
            return True
        if lowered in {"false", "no", "off"}:
            return False
        if re.fullmatch(r"[-+]?\d+", cleaned):
            return int(cleaned)
        if re.fullmatch(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?", cleaned):
            return float(cleaned)
        return cleaned

    def _wavelength_from_cm1(self, energy_cm1: float) -> Optional[float]:
        if not energy_cm1:
            return None
        return 1.0e7 / energy_cm1
