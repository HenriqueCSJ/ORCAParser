"""CSV writers for job-state oriented exports.

These exports change whenever ORCA job modes expand, so keeping them separate
from the main CSV writer helps the top-level module stay focused on dispatch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List

from ..job_series import get_surface_scan_series


WriteCSV = Callable[[Path, str, List[Dict], List[str]], Path]
BoolToLabel = Callable[[Any], str]
ElectronicStateLabel = Callable[[Dict[str, Any]], str]
GetSymmetryData = Callable[[Dict[str, Any]], Dict[str, Any]]
GetDeltaSCFData = Callable[[Dict[str, Any]], Dict[str, Any] | None]
GetExcitedStateOptData = Callable[[Dict[str, Any]], Dict[str, Any] | None]
IsDeltaSCF = Callable[[Dict[str, Any]], bool]
IsSurfaceScan = Callable[[Dict[str, Any]], bool]
FormatDeltaSCFTarget = Callable[[Dict[str, Any] | None], str]
FormatSimpleVector = Callable[[Any], str]
ExcitedStateTargetLabel = Callable[[Dict[str, Any] | None], str]
GetJobSnapshot = Callable[[Dict[str, Any]], Dict[str, Any]]


def write_metadata_section(
    data: Dict[str, Any],
    directory: Path,
    stem: str,
    *,
    write_csv: WriteCSV,
    electronic_state_label: ElectronicStateLabel,
    get_symmetry_data: GetSymmetryData,
    get_deltascf_data: GetDeltaSCFData,
    get_excited_state_opt_data: GetExcitedStateOptData,
    bool_to_label: BoolToLabel,
    is_surface_scan: IsSurfaceScan,
    format_deltascf_target: FormatDeltaSCFTarget,
    excited_state_target_label: ExcitedStateTargetLabel,
    get_job_snapshot: GetJobSnapshot,
) -> List[Path]:
    """Write a one-row metadata summary for downstream filtering/grouping."""
    meta = data.get("metadata", {})
    if not meta:
        return []

    snapshot = get_job_snapshot(data)
    sym = get_symmetry_data(data)
    deltascf = get_deltascf_data(data) or {}
    excopt = get_excited_state_opt_data(data) or {}
    geom = data.get("geometry", {})
    surface_scan = get_surface_scan_series(data)
    row = {
        "job_id": snapshot.get("job_id", meta.get("job_id", "")),
        "job_name": snapshot.get("job_name", meta.get("job_name", "")),
        "source_file": data.get("source_file", ""),
        "source_relpath": snapshot.get("source_relpath", meta.get("source_relpath", "")),
        "program_version": meta.get("program_version", ""),
        "run_date": meta.get("run_date", ""),
        "host": meta.get("host", ""),
        "calculation_type": snapshot.get("calculation_type", meta.get("calculation_type", "")),
        "calculation_family": snapshot.get("calculation_family", ""),
        "electronic_state": electronic_state_label(data),
        "hf_type": snapshot.get("hf_type", meta.get("hf_type", "")),
        "method": snapshot.get("method", meta.get("method", "")),
        "functional": snapshot.get("functional", meta.get("functional", "")),
        "level_of_theory": snapshot.get("level_of_theory", meta.get("level_of_theory", "")),
        "method_header_label": snapshot.get("method_header_label", ""),
        "method_table_label": snapshot.get("method_table_label", ""),
        "basis_set": snapshot.get("basis_set", meta.get("basis_set", "")),
        "aux_basis_set": snapshot.get("aux_basis_set", meta.get("aux_basis_set", "")),
        "charge": snapshot.get("charge", meta.get("charge", "")),
        "multiplicity": snapshot.get("multiplicity", meta.get("multiplicity", "")),
        "point_group": sym.get("point_group", ""),
        "reduced_point_group": sym.get("reduced_point_group", ""),
        "orbital_irrep_group": sym.get("orbital_irrep_group", ""),
        "symmetry_label": sym.get("symmetry_label", ""),
        "has_symmetry": bool_to_label(sym.get("has_symmetry")),
        "use_sym": bool_to_label(sym.get("use_sym")),
        "input_use_sym": bool_to_label(sym.get("input_use_sym", meta.get("input_use_sym"))),
        "n_irreps": sym.get("n_irreps", ""),
        "initial_guess_irrep": sym.get("initial_guess_irrep", ""),
        "symmetry_perfected_point_group": geom.get("symmetry_perfected_point_group", ""),
        "symmetry_perfected_atoms": len(geom.get("symmetry_cartesian_angstrom") or []),
        "is_surface_scan": bool_to_label(is_surface_scan(data)),
        "scan_mode": surface_scan.get("mode", ""),
        "scan_parameters": surface_scan.get("n_parameters", ""),
        "scan_steps": surface_scan.get("n_constrained_optimizations", ""),
        "deltascf_target": format_deltascf_target(deltascf),
        "deltascf_metric": deltascf.get("aufbau_metric", ""),
        "keep_initial_reference": bool_to_label(deltascf.get("keep_initial_reference")),
        "excited_state_target": excited_state_target_label(excopt),
        "excited_state_input_block": excopt.get("input_block", ""),
        "excited_state_followiroot": bool_to_label(excopt.get("followiroot")),
        "excited_state_socgrad": bool_to_label(excopt.get("socgrad")),
        "excited_state_final_root": excopt.get("final_root", ""),
        "input_keywords": " ".join(meta.get("input_keywords") or []),
    }
    return [write_csv(directory, f"{stem}_metadata.csv", [row], list(row.keys()))]


def write_symmetry_section(
    data: Dict[str, Any],
    directory: Path,
    stem: str,
    *,
    write_csv: WriteCSV,
    get_symmetry_data: GetSymmetryData,
    bool_to_label: BoolToLabel,
) -> List[Path]:
    """Write symmetry summary plus irrep-resolved details when available."""
    sym = get_symmetry_data(data)
    if not sym:
        return []

    files = []
    geom = data.get("geometry", {})
    orbital_energies = data.get("orbital_energies", {})

    summary_row = {
        "use_sym": bool_to_label(sym.get("use_sym")),
        "auto_detected_point_group": sym.get("auto_detected_point_group", ""),
        "point_group": sym.get("point_group", ""),
        "reduced_point_group": sym.get("reduced_point_group", ""),
        "orbital_irrep_group": sym.get("orbital_irrep_group", ""),
        "petite_list_algorithm": bool_to_label(sym.get("petite_list_algorithm")),
        "n_irreps": sym.get("n_irreps", ""),
        "initial_guess_irrep": sym.get("initial_guess_irrep", ""),
        "setup_rms_distance_au": sym.get("setup_rms_distance_au", ""),
        "setup_max_distance_au": sym.get("setup_max_distance_au", ""),
        "setup_threshold_au": sym.get("setup_threshold_au", ""),
        "setup_time_s": sym.get("setup_time_s", ""),
        "symmetry_perfected_point_group": geom.get("symmetry_perfected_point_group", ""),
        "symmetry_perfected_atoms": len(geom.get("symmetry_cartesian_angstrom") or []),
    }
    files.append(write_csv(
        directory, f"{stem}_symmetry.csv", [summary_row], list(summary_row.keys())
    ))

    irreps = sym.get("irreps") or []
    alpha_occ = orbital_energies.get("alpha_occupied_per_irrep") or {}
    beta_occ = orbital_energies.get("beta_occupied_per_irrep") or {}
    total_occ = orbital_energies.get("occupied_per_irrep") or {}
    irrep_order = [entry.get("label", "") for entry in irreps if entry.get("label")]
    for mapping in (alpha_occ, beta_occ, total_occ):
        for label in mapping:
            if label not in irrep_order:
                irrep_order.append(label)

    if irrep_order:
        irreps_by_label = {entry.get("label"): entry for entry in irreps}
        rows = []
        for label in irrep_order:
            entry = irreps_by_label.get(label, {})
            rows.append({
                "irrep": label,
                "n_basis_functions": entry.get("n_basis_functions", ""),
                "offset": entry.get("offset", ""),
                "occupied_alpha": alpha_occ.get(label, ""),
                "occupied_beta": beta_occ.get(label, ""),
                "occupied_total": total_occ.get(label, ""),
            })
        files.append(write_csv(
            directory, f"{stem}_symmetry_irreps.csv", rows,
            ["irrep", "n_basis_functions", "offset", "occupied_alpha", "occupied_beta", "occupied_total"],
        ))

    return files


def write_deltascf_section(
    data: Dict[str, Any],
    directory: Path,
    stem: str,
    *,
    write_csv: WriteCSV,
    electronic_state_label: ElectronicStateLabel,
    get_deltascf_data: GetDeltaSCFData,
    is_deltascf: IsDeltaSCF,
    format_deltascf_target: FormatDeltaSCFTarget,
    format_simple_vector: FormatSimpleVector,
    bool_to_label: BoolToLabel,
) -> List[Path]:
    """Write DeltaSCF excited-state target metadata when present."""
    deltascf = get_deltascf_data(data) or {}
    if not is_deltascf(data):
        return []

    files = []
    summary_row = {
        "electronic_state": electronic_state_label(data),
        "target_configuration": format_deltascf_target(deltascf),
        "alphaconf": format_simple_vector(deltascf.get("alphaconf")),
        "betaconf": format_simple_vector(deltascf.get("betaconf")),
        "ionizealpha": deltascf.get("ionizealpha", ""),
        "ionizebeta": deltascf.get("ionizebeta", ""),
        "aufbau_metric": deltascf.get("aufbau_metric", ""),
        "keep_initial_reference": bool_to_label(deltascf.get("keep_initial_reference")),
    }
    files.append(write_csv(
        directory, f"{stem}_deltascf.csv", [summary_row], list(summary_row.keys())
    ))

    target_rows = []
    for spin_key, label in (("alpha_occupation", "alpha"), ("beta_occupation", "beta")):
        values = deltascf.get(spin_key) or []
        for idx, occupation in enumerate(values, start=1):
            target_rows.append({
                "spin": label,
                "slot": idx,
                "occupation": occupation,
            })
    if target_rows:
        files.append(write_csv(
            directory, f"{stem}_deltascf_occupations.csv", target_rows,
            ["spin", "slot", "occupation"],
        ))

    return files


def write_excited_state_optimization_section(
    data: Dict[str, Any],
    directory: Path,
    stem: str,
    *,
    write_csv: WriteCSV,
    electronic_state_label: ElectronicStateLabel,
    get_excited_state_opt_data: GetExcitedStateOptData,
    excited_state_target_label: ExcitedStateTargetLabel,
    bool_to_label: BoolToLabel,
    format_simple_vector: FormatSimpleVector,
) -> List[Path]:
    """Write excited-state geometry-optimization metadata and cycle history."""
    excopt = get_excited_state_opt_data(data) or {}
    if not excopt:
        return []

    files: List[Path] = []
    summary_row = {
        "electronic_state": electronic_state_label(data),
        "target_state": excited_state_target_label(excopt),
        "target_root": excopt.get("target_root", ""),
        "target_multiplicity": excopt.get("target_multiplicity", ""),
        "input_block": excopt.get("input_block", ""),
        "input_nroots": excopt.get("input_nroots", ""),
        "followiroot": bool_to_label(excopt.get("followiroot")),
        "firkeepfirstref": bool_to_label(excopt.get("firkeepfirstref")),
        "analytic_excited_state_gradients": bool_to_label(
            excopt.get("analytic_excited_state_gradients")
        ),
        "socgrad": bool_to_label(excopt.get("socgrad")),
        "final_root": excopt.get("final_root", ""),
        "final_state_of_interest": excopt.get("final_state_of_interest", ""),
        "gradient_block_count": excopt.get("gradient_block_count", ""),
        "input_electron_density": excopt.get("input_electron_density", ""),
        "cispre_job_title": excopt.get("cispre_job_title", ""),
        "firen_thresh_eV": excopt.get("firen_thresh_eV", ""),
        "firs2_thresh": excopt.get("firs2_thresh", ""),
        "firsthresh": excopt.get("firsthresh", ""),
        "firminoverlap": excopt.get("firminoverlap", ""),
        "firdynoverlap": bool_to_label(excopt.get("firdynoverlap")),
        "firdynoverratio": format_simple_vector(excopt.get("firdynoverratio")),
        "root_follow_updates": format_simple_vector(excopt.get("root_follow_updates")),
    }
    files.append(write_csv(
        directory,
        f"{stem}_excited_state_optimization.csv",
        [summary_row],
        list(summary_row.keys()),
    ))

    cycle_records = excopt.get("cycle_records") or []
    if cycle_records:
        cycle_rows = []
        for record in cycle_records:
            cycle_rows.append({
                "block_index": record.get("block_index", ""),
                "optimization_cycle": record.get("optimization_cycle", ""),
                "root": record.get("root", ""),
                "current_iroot": record.get("current_iroot", ""),
                "state_of_interest": record.get("state_of_interest", ""),
                "delta_energy_Eh": record.get("delta_energy_Eh", ""),
                "delta_energy_eV": record.get("delta_energy_eV", ""),
                "total_energy_Eh": record.get("total_energy_Eh", ""),
                "followiroot_runtime": bool_to_label(record.get("followiroot_runtime")),
                "input_electron_density": record.get("input_electron_density", ""),
            })
        files.append(write_csv(
            directory,
            f"{stem}_excited_state_optimization_cycles.csv",
            cycle_rows,
            [
                "block_index",
                "optimization_cycle",
                "root",
                "current_iroot",
                "state_of_interest",
                "delta_energy_Eh",
                "delta_energy_eV",
                "total_energy_Eh",
                "followiroot_runtime",
                "input_electron_density",
            ],
        ))

    return files
