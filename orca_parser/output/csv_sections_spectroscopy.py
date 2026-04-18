"""CSV writers for spectroscopy-oriented exports.

These sections cover EPR and TDDFT/CIS output, which are bulky enough to live
outside the main CSV dispatcher while still sharing the same row-oriented
serialization style.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List


WriteCSV = Callable[[Path, str, List[Dict], List[str]], Path]
BoolToLabel = Callable[[Any], str]


def write_epr_section(
    data: Dict[str, Any],
    directory: Path,
    stem: str,
    *,
    write_csv: WriteCSV,
) -> List[Path]:
    """Write compact EPR tables for g-tensor, hyperfine, and quadrupole data."""
    epr = data.get("epr")
    if not epr:
        return []

    files = []

    g_tensor = epr.get("g_tensor", {})
    atoms = g_tensor.get("atom_analysis", {}).get("atom_contributions", [])
    if atoms:
        rows = [{
            "element": atom.get("element"),
            "atom_index": atom.get("atom_index"),
            "g_1": atom.get("values", [None, None, None])[0],
            "g_2": atom.get("values", [None, None, None])[1],
            "g_3": atom.get("values", [None, None, None])[2],
            "iso": atom.get("iso"),
        } for atom in atoms]
        files.append(write_csv(
            directory, f"{stem}_epr_g_atoms.csv", rows,
            ["element", "atom_index", "g_1", "g_2", "g_3", "iso"],
        ))

    hyperfine = epr.get("hyperfine", {})
    nuclei = hyperfine.get("nuclei", [])
    if nuclei:
        rows = []
        for nucleus in nuclei:
            base = {
                "nucleus_index": nucleus.get("nucleus_index"),
                "element": nucleus.get("element"),
                "isotope": nucleus.get("isotope"),
            }
            for key, principal_components in nucleus.get("principal_components", {}).items():
                values = principal_components.get("values_MHz", [None, None, None])
                rows.append({
                    **base,
                    "component": key,
                    "A_1_MHz": values[0],
                    "A_2_MHz": values[1],
                    "A_3_MHz": values[2],
                    "A_iso_MHz": principal_components.get("A_iso_MHz"),
                    "A_PC_MHz": principal_components.get("A_PC_MHz"),
                })
        if rows:
            files.append(write_csv(
                directory, f"{stem}_epr_hyperfine.csv", rows,
                [
                    "nucleus_index", "element", "isotope", "component",
                    "A_1_MHz", "A_2_MHz", "A_3_MHz", "A_iso_MHz", "A_PC_MHz",
                ],
            ))

    quadrupole_rows = []
    for nucleus in nuclei:
        quadrupole = nucleus.get("quadrupole_coupling")
        if quadrupole:
            quadrupole_rows.append({
                "nucleus_index": nucleus.get("nucleus_index"),
                "element": nucleus.get("element"),
                "isotope": nucleus.get("isotope"),
                "e2qQ_MHz": quadrupole.get("e2qQ_MHz"),
                "eta": quadrupole.get("eta"),
            })
    if quadrupole_rows:
        files.append(write_csv(
            directory, f"{stem}_epr_quadrupole.csv", quadrupole_rows,
            ["nucleus_index", "element", "isotope", "e2qQ_MHz", "eta"],
        ))

    return files


def write_tddft_section(
    data: Dict[str, Any],
    directory: Path,
    stem: str,
    *,
    write_csv: WriteCSV,
    bool_to_label: BoolToLabel,
) -> List[Path]:
    """Write TDDFT/CIS states, transitions, spectra, NTOs, and total energies."""
    tddft = data.get("tddft")
    if not tddft:
        return []

    files: List[Path] = []

    states = tddft.get("excited_states", [])
    if states:
        state_rows = []
        transition_rows = []

        for state in states:
            transitions = state.get("transitions", [])
            # Persist the parser-normalized significant view so CSV consumers do
            # not need to reimplement the TDDFT reporting thresholds themselves.
            significant_transitions = state.get("significant_transitions", [])
            dominant = max(
                transitions,
                key=lambda item: item.get("weight", 0.0),
                default={},
            )
            state_rows.append({
                "block_index": state.get("block_index"),
                "method": state.get("method"),
                "manifold": state.get("manifold", ""),
                "order_in_block": state.get("order_in_block"),
                "state": state.get("state"),
                "energy_rank": state.get("energy_rank", ""),
                "state_label": state.get("state_label", ""),
                "symmetry": state.get("symmetry", ""),
                "energy_au": state.get("energy_au"),
                "energy_eV": state.get("energy_eV"),
                "energy_cm1": state.get("energy_cm1"),
                "wavelength_nm": state.get("wavelength_nm", ""),
                "s_squared": state.get("s_squared", ""),
                "multiplicity": state.get("multiplicity", ""),
                "significant_transition_weight_threshold": state.get(
                    "significant_transition_weight_threshold",
                    "",
                ),
                "significant_transition_count": state.get(
                    "significant_transition_count",
                    "",
                ),
                "significant_transitions": "; ".join(
                    f"{transition.get('from_orbital', '')} -> {transition.get('to_orbital', '')}"
                    for transition in significant_transitions
                ),
                "significant_weights_percent": "; ".join(
                    f"{100.0 * float(transition.get('weight', 0.0)):.1f}"
                    for transition in significant_transitions
                    if transition.get("weight") is not None
                ),
                "dominant_from_orbital": dominant.get("from_orbital", ""),
                "dominant_to_orbital": dominant.get("to_orbital", ""),
                "dominant_weight": dominant.get("weight", ""),
                "dominant_coefficient": dominant.get("coefficient", ""),
            })

            for transition in transitions:
                transition_rows.append({
                    "block_index": state.get("block_index"),
                    "method": state.get("method"),
                    "manifold": state.get("manifold", ""),
                    "state": state.get("state"),
                    "energy_rank": state.get("energy_rank", ""),
                    "state_label": state.get("state_label", ""),
                    "from_orbital": transition.get("from_orbital"),
                    "from_index": transition.get("from_index", ""),
                    "from_spin": transition.get("from_spin", ""),
                    "to_orbital": transition.get("to_orbital"),
                    "to_index": transition.get("to_index", ""),
                    "to_spin": transition.get("to_spin", ""),
                    "weight": transition.get("weight"),
                    "coefficient": transition.get("coefficient"),
                    "meets_reporting_threshold": transition.get(
                        "meets_reporting_threshold",
                        "",
                    ),
                    "significant_transition_weight_threshold": state.get(
                        "significant_transition_weight_threshold",
                        "",
                    ),
                })

        files.append(write_csv(
            directory, f"{stem}_tddft_states.csv", state_rows,
            [
                "block_index", "method", "manifold", "order_in_block",
                "state", "energy_rank", "state_label", "symmetry", "energy_au", "energy_eV", "energy_cm1",
                "wavelength_nm", "s_squared", "multiplicity",
                "significant_transition_weight_threshold", "significant_transition_count",
                "significant_transitions", "significant_weights_percent",
                "dominant_from_orbital", "dominant_to_orbital",
                "dominant_weight", "dominant_coefficient",
            ],
        ))

        if transition_rows:
            files.append(write_csv(
                directory, f"{stem}_tddft_transitions.csv", transition_rows,
                [
                    "block_index", "method", "manifold", "state", "energy_rank", "state_label",
                    "from_orbital", "from_index", "from_spin",
                    "to_orbital", "to_index", "to_spin",
                    "weight", "coefficient",
                    "meets_reporting_threshold",
                    "significant_transition_weight_threshold",
                ],
            ))

    nto_states = tddft.get("nto_states", [])
    if nto_states:
        nto_rows = []
        for state in nto_states:
            for pair in state.get("pairs", []):
                nto_rows.append({
                    "state": state.get("state"),
                    "energy_rank": state.get("energy_rank", ""),
                    "output_file": state.get("output_file", ""),
                    "print_threshold": state.get("print_threshold", ""),
                    "energy_au": state.get("energy_au", ""),
                    "energy_eV": state.get("energy_eV", ""),
                    "energy_cm1": state.get("energy_cm1", ""),
                    "wavelength_nm": state.get("wavelength_nm", ""),
                    "energy_match_consistent": state.get("energy_match_consistent", ""),
                    "energy_matched_state": state.get("energy_matched_state", ""),
                    "energy_matched_rank": state.get("energy_matched_rank", ""),
                    "energy_matched_delta_eV": state.get("energy_matched_delta_eV", ""),
                    "significant_occupation_threshold": state.get(
                        "significant_occupation_threshold",
                        "",
                    ),
                    "significant_pair_count": state.get("significant_pair_count", ""),
                    "from_orbital": pair.get("from_orbital"),
                    "from_index": pair.get("from_index", ""),
                    "from_spin": pair.get("from_spin", ""),
                    "to_orbital": pair.get("to_orbital"),
                    "to_index": pair.get("to_index", ""),
                    "to_spin": pair.get("to_spin", ""),
                    "occupation": pair.get("occupation"),
                    "meets_reporting_threshold": pair.get(
                        "meets_reporting_threshold",
                        "",
                    ),
                })
        if nto_rows:
            files.append(write_csv(
                directory, f"{stem}_tddft_nto.csv", nto_rows,
                [
                    "state", "energy_rank", "output_file", "print_threshold",
                    "energy_au", "energy_eV", "energy_cm1", "wavelength_nm",
                    "energy_match_consistent", "energy_matched_state",
                    "energy_matched_rank", "energy_matched_delta_eV",
                    "significant_occupation_threshold", "significant_pair_count",
                    "from_orbital", "from_index", "from_spin",
                    "to_orbital", "to_index", "to_spin", "occupation",
                    "meets_reporting_threshold",
                ],
            ))

    spectra = tddft.get("spectra", {})
    spectrum_suffixes = {
        "absorption_electric_dipole": "tddft_absorption_electric.csv",
        "absorption_velocity_dipole": "tddft_absorption_velocity.csv",
        "cd_electric_dipole": "tddft_cd_electric.csv",
        "cd_velocity_dipole": "tddft_cd_velocity.csv",
    }
    base_fields = [
        "from_state_label", "from_root", "from_state_suffix",
        "to_state_label", "to_root", "to_state_suffix",
        "energy_eV", "energy_cm1", "wavelength_nm",
    ]
    for kind, filename in spectrum_suffixes.items():
        table = spectra.get(kind)
        if not table:
            continue
        transitions = table.get("transitions", [])
        if not transitions:
            continue

        center = table.get("center_of_mass", {})
        rows = []
        for transition in transitions:
            row = dict(transition)
            row["center_of_mass_x"] = center.get("x", "")
            row["center_of_mass_y"] = center.get("y", "")
            row["center_of_mass_z"] = center.get("z", "")
            rows.append(row)

        extra_fields = [
            field
            for field in rows[0].keys()
            if field not in base_fields
            and field not in {"center_of_mass_x", "center_of_mass_y", "center_of_mass_z"}
        ]
        files.append(write_csv(
            directory, f"{stem}_{filename}", rows,
            base_fields + extra_fields + [
                "center_of_mass_x", "center_of_mass_y", "center_of_mass_z",
            ],
        ))

    total_energy_blocks = tddft.get("total_energy_blocks", [])
    if total_energy_blocks:
        rows = []
        for block in total_energy_blocks:
            rows.append({
                "block_index": block.get("block_index", ""),
                "excitation_method": block.get("excitation_method", ""),
                "root": block.get("root", ""),
                "optimization_cycle": block.get("optimization_cycle", ""),
                "state_of_interest": block.get("state_of_interest", ""),
                "current_iroot": block.get("current_iroot", ""),
                "scf_energy_Eh": block.get("scf_energy_Eh", ""),
                "delta_energy_Eh": block.get("delta_energy_Eh", ""),
                "total_energy_Eh": block.get("total_energy_Eh", ""),
                "maximum_memory_MB": block.get("maximum_memory_MB", ""),
                "followiroot_runtime": bool_to_label(block.get("followiroot_runtime")),
                "input_electron_density": block.get("input_electron_density", ""),
            })
        files.append(write_csv(
            directory, f"{stem}_tddft_total_energy.csv", rows,
            [
                "block_index", "excitation_method", "root",
                "optimization_cycle", "state_of_interest", "current_iroot",
                "scf_energy_Eh", "delta_energy_Eh", "total_energy_Eh",
                "maximum_memory_MB", "followiroot_runtime", "input_electron_density",
            ],
        ))

    return files
