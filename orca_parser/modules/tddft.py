"""
Module for TDDFT/CIS excited-state parsing.

Extracts:
  - %tddft / %cis input settings from the echoed input block
  - Excited-state manifolds and per-state excitation amplitudes
  - Natural transition orbital (NTO) blocks
  - Absorption and CD spectra tables
  - CIS/TDDFT total-energy summary
  - Excited-state geometry-optimization targets and root-follow metadata
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseModule


_HARTREE_TO_EV = 27.211386245988
# Keep this tolerance intentionally small: it is meant to link the NTO block
# back to the corresponding TDDFT root for the same printed state, not to
# "fix" ORCA numbering by aggressively rematching nearby roots.
_NTO_ENERGY_MATCH_TOLERANCE_EV = 0.005


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
    _ANALYTIC_GRADIENT_RE = re.compile(
        r"WARNING:\s*\((?:CIS/TDDFT|TDDFT/CIS)\)\s*:\s*Analytic excited state gradients requested",
        re.I,
    )
    _GEOM_OPT_CYCLE_RE = re.compile(r"GEOMETRY OPTIMIZATION CYCLE\s+(\d+)", re.I)
    _FOLLOW_IROOT_RE = re.compile(r"Follow IRoot\s+\.+\s+(\S+)", re.I)
    _STATE_OF_INTEREST_RE = re.compile(r"State of interest\s+\.+\s+(\d+)", re.I)
    _BLOCK_IROOT_RE = re.compile(r"^\s*IROOT\s+(\d+)\s*$", re.I)
    _INPUT_ELECTRON_DENSITY_RE = re.compile(
        r"Input electron density\s+\.+\s+(\S+)",
        re.I,
    )
    _CISPRE_JOB_TITLE_RE = re.compile(
        r"Job title:\s+ORCA Job:\s+(.+)$",
        re.I,
    )
    _ROOT_UPDATE_RE = re.compile(
        r"The IROOT now is:\s*(?:\.{3}\s*)?(\d+)",
        re.I,
    )
    _EXCITED_STATE_GRADIENT_DONE_RE = re.compile(
        r"EXCITED STATE GRADIENT DONE",
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
            # Preserve ORCA root numbering and add a separate energy-based view.
            self._annotate_excited_state_energy_ranks(excited_state_blocks)
            data["excited_state_blocks"] = excited_state_blocks
            data["excited_states"] = [
                state
                for block in excited_state_blocks
                for state in block.get("states", [])
            ]
            data["final_excited_state_block"] = excited_state_blocks[-1]

        nto_states = self._parse_nto_states(lines)
        if nto_states:
            # NTO sections also carry ORCA root labels. Annotate them against the
            # final excited-state block so downstream code can see both:
            #   1. ORCA root identity (`state`)
            #   2. energy ordering (`energy_rank`)
            # plus a consistency diagnostic for suspicious files.
            self._annotate_nto_state_matches(
                nto_states,
                excited_state_blocks[-1].get("states", []) if excited_state_blocks else [],
            )
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

        excited_state_optimization = self._parse_excited_state_optimization(
            lines=lines,
            input_blocks=input_blocks,
            excited_state_blocks=excited_state_blocks,
            total_energy_blocks=total_energy_blocks,
        )
        if excited_state_optimization:
            data["excited_state_optimization"] = excited_state_optimization

        if not data:
            return None

        summary = self._build_summary(
            input_blocks=input_blocks,
            excited_state_blocks=excited_state_blocks,
            nto_states=nto_states,
            spectra=spectra,
            total_energy_blocks=total_energy_blocks,
        )
        if excited_state_optimization:
            summary.update(
                {
                    "iroot": excited_state_optimization.get("target_root"),
                    "irootmult": excited_state_optimization.get("target_multiplicity"),
                    "followiroot": excited_state_optimization.get("followiroot"),
                    "target_state_label": excited_state_optimization.get("target_state_label"),
                }
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

    def _annotate_excited_state_energy_ranks(
        self,
        excited_state_blocks: List[Dict[str, Any]],
    ) -> None:
        """Annotate each parsed state with its energy rank inside the block.

        ORCA root numbers are authoritative and must remain unchanged. The extra
        energy-rank field makes it explicit when `STATE 1` is not the lowest
        excitation, which happens in real TDDFT outputs.
        """
        for block in excited_state_blocks:
            states = block.get("states", [])
            ranked_states = sorted(
                states,
                key=lambda state: (
                    state.get("energy_eV", float("inf")),
                    state.get("state", float("inf")),
                    state.get("order_in_block", float("inf")),
                ),
            )
            for energy_rank, state in enumerate(ranked_states, start=1):
                state["energy_rank"] = energy_rank

            if states:
                block["energy_order_matches_root_order"] = all(
                    state.get("state") == state.get("energy_rank")
                    for state in states
                )

    def _annotate_nto_state_matches(
        self,
        nto_states: List[Dict[str, Any]],
        reference_states: List[Dict[str, Any]],
    ) -> None:
        """Attach energy-rank and energy-consistency metadata to NTO states.

        We keep the ORCA root index from the NTO header as-is, then add:
          - `energy_rank` inherited from the matching TDDFT root
          - root-to-root energy delta
          - nearest-by-energy TDDFT root within a small tolerance

        This preserves ORCA identity while making downstream validation easy.
        The goal is to make energy ordering explicit, not to silently replace
        the printed ORCA root numbering with a new numbering scheme.
        """
        if not reference_states:
            return

        states_by_root = {
            state.get("state"): state
            for state in reference_states
            if state.get("state") is not None
        }

        for nto_state in nto_states:
            root = nto_state.get("state")
            root_state = states_by_root.get(root)
            matching_states = self._states_within_energy_tolerance(
                nto_state.get("energy_eV"),
                reference_states,
                _NTO_ENERGY_MATCH_TOLERANCE_EV,
            )
            if root_state is not None:
                if root_state.get("energy_rank") is not None:
                    nto_state["energy_rank"] = root_state.get("energy_rank")

                root_delta = self._energy_delta_ev(
                    nto_state.get("energy_eV"),
                    root_state.get("energy_eV"),
                )
                if root_delta is not None:
                    nto_state["root_energy_delta_eV"] = root_delta
                    nto_state["root_energy_match_within_tolerance"] = (
                        root_delta <= _NTO_ENERGY_MATCH_TOLERANCE_EV
                    )

            matched_state, matched_delta = self._nearest_state_by_energy(
                nto_state.get("energy_eV"),
                reference_states,
            )
            if matched_state is None or matched_delta is None:
                continue

            nto_state["energy_matched_state"] = matched_state.get("state")
            nto_state["energy_matched_delta_eV"] = matched_delta
            if matched_state.get("energy_rank") is not None:
                nto_state["energy_matched_rank"] = matched_state.get("energy_rank")
            nto_state["energy_match_within_tolerance"] = (
                matched_delta <= _NTO_ENERGY_MATCH_TOLERANCE_EV
            )
            if root_state is not None:
                # Near-degenerate roots can share the same energy window. Treat
                # the NTO as internally consistent when its same-root TDDFT
                # state is among the roots that match within tolerance.
                nto_state["energy_match_consistent"] = (
                    any(
                        candidate.get("state") == root_state.get("state")
                        for candidate in matching_states
                    )
                )

    def _nearest_state_by_energy(
        self,
        energy_eV: Any,
        reference_states: List[Dict[str, Any]],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
        """Return the TDDFT state closest in energy to the supplied value."""
        if energy_eV is None:
            return None, None

        best_state: Optional[Dict[str, Any]] = None
        best_delta: Optional[float] = None
        target = float(energy_eV)
        for state in reference_states:
            delta = self._energy_delta_ev(target, state.get("energy_eV"))
            if delta is None:
                continue
            if best_delta is None or delta < best_delta:
                best_state = state
                best_delta = delta
        return best_state, best_delta

    def _states_within_energy_tolerance(
        self,
        energy_eV: Any,
        reference_states: List[Dict[str, Any]],
        tolerance_eV: float,
    ) -> List[Dict[str, Any]]:
        """Return all reference states with energies within the requested tolerance."""
        matches: List[Dict[str, Any]] = []
        for state in reference_states:
            delta = self._energy_delta_ev(energy_eV, state.get("energy_eV"))
            if delta is not None and delta <= tolerance_eV:
                matches.append(state)
        return matches

    def _energy_delta_ev(self, left: Any, right: Any) -> Optional[float]:
        """Safely compute an absolute energy difference in eV."""
        if left is None or right is None:
            return None
        return abs(float(left) - float(right))

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

            cycle_number = self._nearest_geometry_optimization_cycle(lines, idx)
            if cycle_number is not None:
                block["optimization_cycle"] = cycle_number

            followiroot_runtime = self._nearest_followiroot_setting(lines, idx)
            if followiroot_runtime is not None:
                block["followiroot_runtime"] = followiroot_runtime

            for line in lines[idx : idx + 320]:
                state_match = self._STATE_OF_INTEREST_RE.search(line)
                if state_match:
                    block["state_of_interest"] = int(state_match.group(1))

                iroot_match = self._BLOCK_IROOT_RE.match(line.strip())
                if iroot_match:
                    block["current_iroot"] = int(iroot_match.group(1))

                density_match = self._INPUT_ELECTRON_DENSITY_RE.search(line)
                if density_match:
                    block["input_electron_density"] = density_match.group(1)

            if len(block) > 1:
                blocks.append(block)

        return blocks

    def _parse_excited_state_optimization(
        self,
        lines: List[str],
        input_blocks: List[Dict[str, Any]],
        excited_state_blocks: List[Dict[str, Any]],
        total_energy_blocks: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        has_geom_opt = (
            self.find_line(lines, "Geometry Optimization Run") != -1
            or any(self._GEOM_OPT_CYCLE_RE.search(line) for line in lines)
        )
        if not has_geom_opt:
            return None

        target_block = next(
            (
                block
                for block in input_blocks
                if block.get("block") in {"tddft", "cis"}
            ),
            None,
        )
        if not target_block:
            return None

        settings = target_block.get("settings", {})
        has_explicit_target = (
            "iroot" in settings
            or "irootmult" in settings
            or "followiroot" in settings
            or "socgrad" in settings
        )
        analytic_gradients_requested = any(
            self._ANALYTIC_GRADIENT_RE.search(line)
            for line in lines[:5000]
        )
        gradient_count = sum(
            1 for line in lines if self._EXCITED_STATE_GRADIENT_DONE_RE.search(line)
        )
        if not has_explicit_target and not analytic_gradients_requested and gradient_count == 0:
            return None

        first_block = excited_state_blocks[0] if excited_state_blocks else {}
        final_block = excited_state_blocks[-1] if excited_state_blocks else {}
        manifold = self._normalize_manifold(
            settings.get("irootmult")
            or final_block.get("manifold")
            or first_block.get("manifold")
        )
        target_root = self.safe_int(settings.get("iroot"))
        root_updates = [
            int(match.group(1))
            for line in lines
            if (match := self._ROOT_UPDATE_RE.search(line))
        ]
        cycle_records = [
            {
                "block_index": block.get("block_index"),
                "optimization_cycle": block.get("optimization_cycle"),
                "root": block.get("root"),
                "state_of_interest": block.get("state_of_interest"),
                "current_iroot": block.get("current_iroot"),
                "delta_energy_Eh": block.get("delta_energy_Eh"),
                "delta_energy_eV": (
                    block["delta_energy_Eh"] * _HARTREE_TO_EV
                    if block.get("delta_energy_Eh") is not None
                    else None
                ),
                "total_energy_Eh": block.get("total_energy_Eh"),
                "followiroot_runtime": block.get("followiroot_runtime"),
                "input_electron_density": block.get("input_electron_density"),
            }
            for block in total_energy_blocks
        ]
        job_titles = [
            match.group(1).strip()
            for line in lines
            if (match := self._CISPRE_JOB_TITLE_RE.search(line))
        ]

        final_root = None
        final_state_of_interest = None
        final_density = None
        if total_energy_blocks:
            final_block_energy = total_energy_blocks[-1]
            if final_block_energy.get("current_iroot") is not None:
                final_root = final_block_energy.get("current_iroot")
            elif final_block_energy.get("root") is not None:
                final_root = final_block_energy.get("root")
            final_state_of_interest = final_block_energy.get("state_of_interest")
            final_density = final_block_energy.get("input_electron_density")

        summary: Dict[str, Any] = {
            "input_block": target_block.get("block"),
            "input_nroots": settings.get("nroots"),
            "target_root": target_root,
            "target_multiplicity": manifold,
            "target_state_label": self._format_state_label(manifold, target_root),
            "followiroot": self._parse_bool_like(settings.get("followiroot")),
            "firkeepfirstref": self._parse_bool_like(settings.get("firkeepfirstref")),
            "firen_thresh_eV": settings.get("firenthresh"),
            "firs2_thresh": settings.get("firs2thresh"),
            "firsthresh": settings.get("firsthresh"),
            "firminoverlap": settings.get("firminoverlap"),
            "firdynoverlap": self._parse_bool_like(settings.get("firdynoverlap")),
            "firdynoverratio": settings.get("firdynoverratio"),
            "socgrad": self._parse_bool_like(settings.get("socgrad")),
            "analytic_excited_state_gradients": analytic_gradients_requested,
            "gradient_block_count": gradient_count,
            "root_follow_updates": root_updates,
            "final_root": final_root,
            "final_state_of_interest": final_state_of_interest,
            "input_electron_density": final_density,
            "cispre_job_title": job_titles[-1] if job_titles else None,
            "cycle_records": cycle_records,
        }

        return {
            key: value
            for key, value in summary.items()
            if value not in (None, [], {})
        }

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
            # Surface the ordering diagnostic in the summary so markdown, CSV,
            # and CLI code do not need to recompute it.
            summary["final_state_order_matches_root_order"] = final_block.get(
                "energy_order_matches_root_order"
            )

        if nto_states:
            summary["nto_state_count"] = len(nto_states)
            summary["nto_energy_match_tolerance_eV"] = _NTO_ENERGY_MATCH_TOLERANCE_EV
            # All NTOs should link back to the same-root TDDFT state within the
            # accepted tolerance; if not, the file is worth a closer look.
            summary["nto_root_alignment_consistent"] = all(
                nto_state.get("energy_match_consistent", True)
                for nto_state in nto_states
            )

        if spectra:
            summary["spectrum_tables"] = sorted(spectra)

        if total_energy_blocks:
            final_energy = total_energy_blocks[-1]
            if "root" in final_energy:
                summary["final_root"] = final_energy["root"]
            if "delta_energy_Eh" in final_energy:
                summary["delta_energy_Eh"] = final_energy["delta_energy_Eh"]

        return summary

    def _nearest_geometry_optimization_cycle(
        self, lines: List[str], index: int
    ) -> Optional[int]:
        for j in range(index, -1, -1):
            match = self._GEOM_OPT_CYCLE_RE.search(lines[j])
            if match:
                return int(match.group(1))
        return None

    def _nearest_followiroot_setting(
        self, lines: List[str], index: int
    ) -> Optional[bool]:
        for j in range(index, max(-1, index - 1200), -1):
            match = self._FOLLOW_IROOT_RE.search(lines[j])
            if match:
                return self._parse_bool_like(match.group(1))
        return None

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
        if "," in cleaned:
            parts = [part.strip() for part in cleaned.split(",")]
            if parts and all(
                re.fullmatch(r"[-+]?\d+", part)
                or re.fullmatch(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?", part)
                for part in parts
            ):
                parsed: List[Any] = []
                for part in parts:
                    if re.fullmatch(r"[-+]?\d+", part):
                        parsed.append(int(part))
                    else:
                        parsed.append(float(part))
                return parsed
        if re.fullmatch(r"[-+]?\d+", cleaned):
            return int(cleaned)
        if re.fullmatch(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?", cleaned):
            return float(cleaned)
        return cleaned

    def _parse_bool_like(self, value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if value is None:
            return None
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "on"}:
                return True
            if lowered in {"false", "no", "off"}:
                return False
        return None

    def _normalize_manifold(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        cleaned = str(value).strip().lower()
        if not cleaned:
            return None
        if cleaned.endswith("s"):
            cleaned = cleaned[:-1]
        aliases = {
            "singlet": "singlet",
            "triplet": "triplet",
            "sf": "spin-flip",
            "spin-flip": "spin-flip",
            "soc": "soc",
        }
        return aliases.get(cleaned, cleaned)

    def _format_state_label(
        self, manifold: Optional[str], root: Optional[int]
    ) -> Optional[str]:
        if root is None:
            return None
        prefix_map = {
            "singlet": "S",
            "triplet": "T",
            "soc": "SOC",
            "spin-flip": "SF",
        }
        prefix = prefix_map.get((manifold or "").lower())
        if prefix == "SOC":
            return f"{prefix}{root}"
        if prefix:
            return f"{prefix}{root}"
        return f"root {root}"

    def _wavelength_from_cm1(self, energy_cm1: float) -> Optional[float]:
        if not energy_cm1:
            return None
        return 1.0e7 / energy_cm1
