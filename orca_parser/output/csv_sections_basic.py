"""Self-contained CSV writers for general job-summary sections.

These sections are small, broadly applicable exports that do not belong to a
specialized domain like NBO, TDDFT, or EPR. Keeping them here helps the main
CSV writer stay focused on dispatch instead of inline data shaping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List

from ..final_snapshot import (
    get_final_dipole,
    get_final_geometry,
)


WriteCSV = Callable[[Path, str, List[Dict], List[str]], Path]
FormatSimpleVector = Callable[[Any], str]


def write_geometry_section(
    data: Dict[str, Any],
    directory: Path,
    stem: str,
    *,
    write_csv: WriteCSV,
) -> List[Path]:
    """Write standard and symmetry-perfected Cartesian coordinates."""
    geometry = get_final_geometry(data)
    files: List[Path] = []

    cartesian = geometry.get("cartesian_angstrom")
    if cartesian:
        rows = []
        for atom in cartesian:
            rows.append({
                "index": atom.get("index"),
                "symbol": atom.get("symbol"),
                "x_ang": atom.get("x_ang"),
                "y_ang": atom.get("y_ang"),
                "z_ang": atom.get("z_ang"),
            })
        files.append(write_csv(
            directory, f"{stem}_geometry.csv", rows,
            ["index", "symbol", "x_ang", "y_ang", "z_ang"],
        ))

    symmetry_cartesian = geometry.get("symmetry_cartesian_angstrom")
    if symmetry_cartesian:
        point_group = geometry.get("symmetry_perfected_point_group", "")
        rows = []
        for atom in symmetry_cartesian:
            rows.append({
                "point_group": point_group,
                "index": atom.get("index"),
                "symbol": atom.get("symbol"),
                "x_ang": atom.get("x_ang"),
                "y_ang": atom.get("y_ang"),
                "z_ang": atom.get("z_ang"),
            })
        files.append(write_csv(
            directory, f"{stem}_geometry_symmetry.csv", rows,
            ["point_group", "index", "symbol", "x_ang", "y_ang", "z_ang"],
        ))

    return files


def write_dipole_section(
    data: Dict[str, Any],
    directory: Path,
    stem: str,
    *,
    write_csv: WriteCSV,
) -> List[Path]:
    """Write permanent dipole components and magnitudes."""
    dipole = get_final_dipole(data)
    if not dipole:
        return []

    row = {
        "ex": dipole.get("electronic_contribution_au", {}).get("x", ""),
        "ey": dipole.get("electronic_contribution_au", {}).get("y", ""),
        "ez": dipole.get("electronic_contribution_au", {}).get("z", ""),
        "nx": dipole.get("nuclear_contribution_au", {}).get("x", ""),
        "ny": dipole.get("nuclear_contribution_au", {}).get("y", ""),
        "nz": dipole.get("nuclear_contribution_au", {}).get("z", ""),
        "total_x_au": dipole.get("total_dipole_au", {}).get("x", ""),
        "total_y_au": dipole.get("total_dipole_au", {}).get("y", ""),
        "total_z_au": dipole.get("total_dipole_au", {}).get("z", ""),
        "magnitude_au": dipole.get("magnitude_au", ""),
        "magnitude_Debye": dipole.get("magnitude_Debye", ""),
    }
    return [write_csv(directory, f"{stem}_dipole.csv", [row], list(row.keys()))]


def write_solvation_section(
    data: Dict[str, Any],
    directory: Path,
    stem: str,
    *,
    write_csv: WriteCSV,
) -> List[Path]:
    """Write solvation summary plus the parsed model history."""
    solvation = data.get("solvation")
    if not solvation:
        return []

    files: List[Path] = []
    summary = solvation.get("summary", {})
    latest_cpcm = solvation.get("cpcm", {})
    latest_alpb = solvation.get("alpb", {})
    latest_cosmors = solvation.get("cosmors", {})

    summary_row = {
        "is_solvated": solvation.get("is_solvated"),
        "primary_model": solvation.get("primary_model", ""),
        "solvent": solvation.get("solvent", ""),
        "models": ";".join(solvation.get("models", [])),
        "input_controlled": summary.get("input_controlled", ""),
        "input_model": summary.get("input_model", ""),
        "input_solvent": summary.get("input_solvent", ""),
        "output_model": summary.get("output_model", ""),
        "output_solvent": summary.get("output_solvent", ""),
        "draco": solvation.get("input_flags", {}).get("draco", False),
        "smd18": solvation.get("input_flags", {}).get("smd18", False),
        "epsilon": summary.get(
            "epsilon",
            latest_cpcm.get("epsilon", latest_alpb.get("epsilon", "")),
        ),
        "surface_type": summary.get("surface_type", latest_cpcm.get("surface_type", "")),
        "epsilon_function_type": latest_cpcm.get("epsilon_function_type", ""),
        "rsolv_ang": latest_cpcm.get("rsolv_ang", ""),
        "cpcm_block_count": summary.get("cpcm_block_count", 0),
        "alpb_block_count": summary.get("alpb_block_count", 0),
        "cosmors_block_count": summary.get("cosmors_block_count", 0),
        "dGsolv_Eh": latest_cosmors.get("dGsolv_Eh", ""),
        "free_energy_shift_Eh": latest_alpb.get("free_energy_shift_Eh", ""),
    }
    files.append(write_csv(
        directory, f"{stem}_solvation.csv", [summary_row],
        [
            "is_solvated", "primary_model", "solvent", "models",
            "input_controlled", "input_model", "input_solvent",
            "output_model", "output_solvent",
            "draco", "smd18",
            "epsilon", "surface_type", "epsilon_function_type", "rsolv_ang",
            "cpcm_block_count", "alpb_block_count", "cosmors_block_count",
            "dGsolv_Eh", "free_energy_shift_Eh",
        ],
    ))

    history_rows: List[Dict[str, Any]] = []
    for directive in solvation.get("input_directives", []):
        history_rows.append({
            "source": directive.get("source"),
            "line_index": directive.get("line_index"),
            "model": directive.get("model"),
            "solvent": directive.get("solvent", ""),
            "epsilon": "",
            "surface_type": "",
            "reference_state": "",
        })

    for block in solvation.get("cpcm_blocks", []):
        history_rows.append({
            "source": "output_cpcm",
            "line_index": block.get("line_index"),
            "model": block.get("model"),
            "solvent": block.get("solvent", ""),
            "epsilon": block.get("epsilon", ""),
            "surface_type": block.get("surface_type", ""),
            "reference_state": "",
        })

    for block in solvation.get("alpb_blocks", []):
        history_rows.append({
            "source": "output_alpb",
            "line_index": block.get("line_index"),
            "model": block.get("model"),
            "solvent": block.get("solvent", ""),
            "epsilon": block.get("epsilon", ""),
            "surface_type": "",
            "reference_state": block.get("reference_state", ""),
        })

    for block in solvation.get("cosmors_blocks", []):
        history_rows.append({
            "source": "output_cosmors",
            "line_index": block.get("line_index"),
            "model": block.get("model"),
            "solvent": block.get("solvent", ""),
            "epsilon": "",
            "surface_type": "",
            "reference_state": "",
        })

    if history_rows:
        history_rows.sort(
            key=lambda item: (item.get("line_index", -1), str(item.get("source", "")))
        )
        files.append(write_csv(
            directory, f"{stem}_solvation_history.csv", history_rows,
            [
                "source", "line_index", "model", "solvent",
                "epsilon", "surface_type", "reference_state",
            ],
        ))

    return files


def write_geom_opt_section(
    data: Dict[str, Any],
    directory: Path,
    stem: str,
    *,
    write_csv: WriteCSV,
) -> List[Path]:
    """Write per-cycle geometry optimization convergence history."""
    geometry_optimization = data.get("geom_opt")
    if not geometry_optimization:
        return []

    cycles = geometry_optimization.get("cycles", [])
    if not cycles:
        return []

    fields = [
        "cycle", "energy_Eh", "energy_change_Eh", "trust_radius_bohr",
        "energy_change_val", "energy_change_tol", "energy_change_conv",
        "rms_gradient_val", "rms_gradient_tol", "rms_gradient_conv",
        "max_gradient_val", "max_gradient_tol", "max_gradient_conv",
        "rms_step_val", "rms_step_tol", "rms_step_conv",
        "max_step_val", "max_step_tol", "max_step_conv",
        "rmsd_to_initial_ang", "rmsd_to_previous_ang",
        "orca_converged",
    ]
    rows: List[Dict[str, Any]] = []
    for cycle in cycles:
        row: Dict[str, Any] = {
            "cycle": cycle.get("cycle"),
            "energy_Eh": cycle.get("energy_Eh"),
            "energy_change_Eh": cycle.get("energy_change_Eh", ""),
            "trust_radius_bohr": cycle.get("trust_radius_bohr", ""),
            "rmsd_to_initial_ang": cycle.get("rmsd_to_initial_ang", ""),
            "rmsd_to_previous_ang": cycle.get("rmsd_to_previous_ang", ""),
            "orca_converged": cycle.get("orca_converged", False),
        }
        convergence = cycle.get("convergence", {})
        for key in ("energy_change", "rms_gradient", "max_gradient", "rms_step", "max_step"):
            entry = convergence.get(key, {})
            row[f"{key}_val"] = entry.get("value", "")
            row[f"{key}_tol"] = entry.get("tolerance", "")
            row[f"{key}_conv"] = entry.get("converged", "")
        rows.append(row)

    return [write_csv(directory, f"{stem}_geom_opt.csv", rows, fields)]


def write_surface_scan_section(
    data: Dict[str, Any],
    directory: Path,
    stem: str,
    *,
    write_csv: WriteCSV,
    format_simple_vector: FormatSimpleVector,
) -> List[Path]:
    """Write relaxed-surface-scan definition, steps, and sidecar files."""
    surface_scan = data.get("surface_scan")
    if not surface_scan:
        return []

    files: List[Path] = []
    parameters = surface_scan.get("parameters") or []
    steps = surface_scan.get("steps") or []

    if parameters:
        rows: List[Dict[str, Any]] = []
        for idx, parameter in enumerate(parameters, start=1):
            rows.append({
                "parameter_index": idx,
                "label": parameter.get("label", ""),
                "kind": parameter.get("kind", ""),
                "coordinate_type": parameter.get("coordinate_type", ""),
                "atoms": ",".join(str(atom) for atom in parameter.get("atoms") or []),
                "unit": parameter.get("unit", ""),
                "mode": parameter.get("mode", ""),
                "start": parameter.get("start", ""),
                "end": parameter.get("end", ""),
                "steps": parameter.get("steps", ""),
                "values": format_simple_vector(parameter.get("values")),
            })
        files.append(write_csv(
            directory, f"{stem}_surface_scan_parameters.csv", rows,
            [
                "parameter_index", "label", "kind", "coordinate_type", "atoms",
                "unit", "mode", "start", "end", "steps", "values",
            ],
        ))

    if steps:
        rows = []
        max_coords = max(len(step.get("coordinate_values") or []) for step in steps)
        for step in steps:
            row: Dict[str, Any] = {
                "step": step.get("step", ""),
                "actual_energy_Eh": step.get("actual_energy_Eh", ""),
                "relative_actual_energy_kcal_mol": step.get("relative_actual_energy_kcal_mol", ""),
                "scf_energy_Eh": step.get("scf_energy_Eh", ""),
                "relative_scf_energy_kcal_mol": step.get("relative_scf_energy_kcal_mol", ""),
                "optimized_xyz_file": step.get("optimized_xyz_file", ""),
            }
            values = step.get("coordinate_values") or []
            labels = step.get("coordinate_labels") or []
            for idx in range(max_coords):
                row[f"coord_{idx + 1}_label"] = labels[idx] if idx < len(labels) else ""
                row[f"coord_{idx + 1}_value"] = values[idx] if idx < len(values) else ""
            rows.append(row)
        fields = [
            "step",
            *sum(
                (
                    [f"coord_{idx + 1}_label", f"coord_{idx + 1}_value"]
                    for idx in range(max_coords)
                ),
                [],
            ),
            "actual_energy_Eh",
            "relative_actual_energy_kcal_mol",
            "scf_energy_Eh",
            "relative_scf_energy_kcal_mol",
            "optimized_xyz_file",
        ]
        files.append(write_csv(
            directory, f"{stem}_surface_scan.csv", rows, fields,
        ))

    sidecars = surface_scan.get("sidecar_files") or {}
    if sidecars:
        row = {
            "mode": surface_scan.get("mode", ""),
            "n_parameters": surface_scan.get("n_parameters", ""),
            "n_constrained_optimizations": surface_scan.get("n_constrained_optimizations", ""),
            "actual_surface_dat": sidecars.get("actual_surface_dat", ""),
            "scf_surface_dat": sidecars.get("scf_surface_dat", ""),
            "allxyz": sidecars.get("allxyz", ""),
            "xyzall": sidecars.get("xyzall", ""),
            "trajectory_xyz": sidecars.get("trajectory_xyz", ""),
            "allxyz_frame_count": sidecars.get("allxyz_frame_count", ""),
        }
        files.append(write_csv(
            directory, f"{stem}_surface_scan_summary.csv", [row], list(row.keys()),
        ))

    return files


def write_goat_section(
    data: Dict[str, Any],
    directory: Path,
    stem: str,
    *,
    write_csv: WriteCSV,
) -> List[Path]:
    """Write GOAT conformer-search summary and full final ensemble."""
    goat = data.get("goat")
    if not goat:
        return []

    files: List[Path] = []
    ensemble = goat.get("ensemble") or []

    if ensemble:
        rows: List[Dict[str, Any]] = []
        for row in ensemble:
            rows.append({
                "conformer": row.get("conformer"),
                "relative_energy_kcal_mol": row.get("relative_energy_kcal_mol"),
                "degeneracy": row.get("degeneracy"),
                "percent_total": row.get("percent_total"),
                "percent_cumulative": row.get("percent_cumulative"),
            })
        files.append(write_csv(
            directory, f"{stem}_goat_ensemble.csv", rows,
            [
                "conformer",
                "relative_energy_kcal_mol",
                "degeneracy",
                "percent_total",
                "percent_cumulative",
            ],
        ))

    summary_row = {
        "global_minimum_found": goat.get("global_minimum_found", ""),
        "global_minimum_conformer": goat.get("global_minimum_conformer", ""),
        "lowest_energy_conformer_Eh": goat.get("lowest_energy_conformer_Eh", ""),
        "n_conformers": goat.get("n_conformers", ""),
        "conformer_energy_window_kcal_mol": goat.get("conformer_energy_window_kcal_mol", ""),
        "conformers_below_energy_window": goat.get("conformers_below_energy_window", ""),
        "temperature_K": goat.get("temperature_K", ""),
        "sconf_cal_molK": goat.get("sconf_cal_molK", ""),
        "gconf_kcal_mol": goat.get("gconf_kcal_mol", ""),
        "top_population_percent": goat.get("top_population_percent", ""),
        "max_relative_energy_kcal_mol": goat.get("max_relative_energy_kcal_mol", ""),
        "final_cumulative_percent": goat.get("final_cumulative_percent", ""),
        "global_minimum_xyz_file": goat.get("global_minimum_xyz_file", ""),
        "final_ensemble_xyz_file": goat.get("final_ensemble_xyz_file", ""),
    }
    files.append(write_csv(
        directory, f"{stem}_goat_summary.csv", [summary_row], list(summary_row.keys()),
    ))

    return files
