"""Canonical normalization for optimization-like job histories.

This module complements ``final_snapshot`` and ``job_snapshot`` by capturing
stepwise or ensemble-style data once during parsing. The goal is to give
writers a stable, parse-time authority for GOAT ensembles, geometry
optimization cycles, relaxed surface scans, and excited-state optimization
cycle records.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _copy_mapping(value: Any) -> Dict[str, Any]:
    """Return a deep-copied dict or an empty mapping."""
    if isinstance(value, dict):
        return deepcopy(value)
    return {}


def _copy_sequence(value: Any) -> List[Any]:
    """Return a deep-copied list or an empty sequence."""
    if isinstance(value, list):
        return deepcopy(value)
    return []


def _normalize_geom_opt(section: Any) -> Dict[str, Any]:
    """Normalize geometry-optimization history to stable keys."""
    mapping = _copy_mapping(section)
    if not mapping:
        return {}

    normalized = {
        "converged": mapping.get("converged"),
        "n_cycles": mapping.get("n_cycles"),
        "opt_keyword": mapping.get("opt_keyword", ""),
        "settings": _copy_mapping(mapping.get("settings")),
        "tolerances": _copy_mapping(mapping.get("tolerances")),
        "initial_geometry_angstrom": _copy_sequence(mapping.get("initial_geometry_angstrom")),
        "final_geometry_angstrom": _copy_sequence(mapping.get("final_geometry_angstrom")),
        "final_energy_Eh": mapping.get("final_energy_Eh"),
        "rmsd_initial_to_final_ang": mapping.get("rmsd_initial_to_final_ang"),
        "mass_weighted_rmsd_initial_to_final_ang": mapping.get(
            "mass_weighted_rmsd_initial_to_final_ang"
        ),
        "cycles": _copy_sequence(mapping.get("cycles")),
    }

    for key, value in mapping.items():
        if key in normalized:
            continue
        normalized[key] = deepcopy(value)
    return normalized


def _normalize_surface_scan(section: Any) -> Dict[str, Any]:
    """Normalize relaxed-surface-scan history to stable keys."""
    mapping = _copy_mapping(section)
    if not mapping:
        return {}

    normalized = {
        "mode": mapping.get("mode", ""),
        "simultaneous_scan": mapping.get("simultaneous_scan"),
        "n_parameters": mapping.get("n_parameters"),
        "n_constrained_optimizations": mapping.get("n_constrained_optimizations"),
        "parameters": _copy_sequence(mapping.get("parameters")),
        "steps": _copy_sequence(mapping.get("steps")),
        "surface_actual_energy": _copy_sequence(mapping.get("surface_actual_energy")),
        "surface_scf_energy": _copy_sequence(mapping.get("surface_scf_energy")),
        "sidecar_files": _copy_mapping(mapping.get("sidecar_files")),
        "actual_energy_min_Eh": mapping.get("actual_energy_min_Eh"),
        "actual_energy_max_Eh": mapping.get("actual_energy_max_Eh"),
        "actual_energy_span_kcal_mol": mapping.get("actual_energy_span_kcal_mol"),
        "scf_energy_min_Eh": mapping.get("scf_energy_min_Eh"),
        "scf_energy_max_Eh": mapping.get("scf_energy_max_Eh"),
        "scf_energy_span_kcal_mol": mapping.get("scf_energy_span_kcal_mol"),
    }

    for key, value in mapping.items():
        if key in normalized:
            continue
        normalized[key] = deepcopy(value)
    return normalized


def _normalize_goat(section: Any) -> Dict[str, Any]:
    """Normalize GOAT ensemble data to stable keys."""
    mapping = _copy_mapping(section)
    if not mapping:
        return {}

    normalized = {
        "global_minimum_found": mapping.get("global_minimum_found"),
        "global_minimum_xyz_file": mapping.get("global_minimum_xyz_file", ""),
        "final_ensemble_xyz_file": mapping.get("final_ensemble_xyz_file", ""),
        "n_conformers": mapping.get("n_conformers"),
        "global_minimum_conformer": mapping.get("global_minimum_conformer"),
        "top_population_percent": mapping.get("top_population_percent"),
        "max_relative_energy_kcal_mol": mapping.get("max_relative_energy_kcal_mol"),
        "final_cumulative_percent": mapping.get("final_cumulative_percent"),
        "conformer_energy_window_kcal_mol": mapping.get("conformer_energy_window_kcal_mol"),
        "conformers_below_energy_window": mapping.get("conformers_below_energy_window"),
        "lowest_energy_conformer_Eh": mapping.get("lowest_energy_conformer_Eh"),
        "temperature_K": mapping.get("temperature_K"),
        "sconf_cal_molK": mapping.get("sconf_cal_molK"),
        "gconf_kcal_mol": mapping.get("gconf_kcal_mol"),
        "ensemble": _copy_sequence(mapping.get("ensemble")),
    }

    for key, value in mapping.items():
        if key in normalized:
            continue
        normalized[key] = deepcopy(value)
    return normalized


def _normalize_excited_state_optimization(section: Any) -> Dict[str, Any]:
    """Normalize excited-state optimization history to stable keys."""
    mapping = _copy_mapping(section)
    if not mapping:
        return {}

    normalized = {
        "input_block": mapping.get("input_block", ""),
        "input_nroots": mapping.get("input_nroots"),
        "target_root": mapping.get("target_root"),
        "target_multiplicity": mapping.get("target_multiplicity", ""),
        "target_state_label": mapping.get("target_state_label", ""),
        "followiroot": mapping.get("followiroot"),
        "firkeepfirstref": mapping.get("firkeepfirstref"),
        "firen_thresh_eV": mapping.get("firen_thresh_eV"),
        "firs2_thresh": mapping.get("firs2_thresh"),
        "firsthresh": mapping.get("firsthresh"),
        "firminoverlap": mapping.get("firminoverlap"),
        "firdynoverlap": mapping.get("firdynoverlap"),
        "firdynoverratio": deepcopy(mapping.get("firdynoverratio")),
        "socgrad": mapping.get("socgrad"),
        "analytic_excited_state_gradients": mapping.get("analytic_excited_state_gradients"),
        "gradient_block_count": mapping.get("gradient_block_count"),
        "root_follow_updates": _copy_sequence(mapping.get("root_follow_updates")),
        "final_root": mapping.get("final_root"),
        "final_state_of_interest": mapping.get("final_state_of_interest"),
        "input_electron_density": mapping.get("input_electron_density", ""),
        "cispre_job_title": mapping.get("cispre_job_title", ""),
        "cycle_records": _copy_sequence(mapping.get("cycle_records")),
    }

    for key, value in mapping.items():
        if key in normalized:
            continue
        normalized[key] = deepcopy(value)
    return normalized


@dataclass
class JobSeries:
    """Canonical optimization/ensemble histories for downstream rendering."""

    geom_opt: Dict[str, Any] = field(default_factory=dict)
    surface_scan: Dict[str, Any] = field(default_factory=dict)
    goat: Dict[str, Any] = field(default_factory=dict)
    excited_state_optimization: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the normalized series to a JSON-safe mapping."""
        return {
            "geom_opt": deepcopy(self.geom_opt),
            "surface_scan": deepcopy(self.surface_scan),
            "goat": deepcopy(self.goat),
            "excited_state_optimization": deepcopy(self.excited_state_optimization),
        }


def build_job_series(data: Dict[str, Any]) -> Optional[JobSeries]:
    """Build canonical optimization-like histories from parsed module payloads."""
    excited_state_optimization: Dict[str, Any] = {}
    tddft = data.get("tddft")
    if isinstance(tddft, dict):
        excited_state_optimization = _normalize_excited_state_optimization(
            tddft.get("excited_state_optimization")
        )

    series = JobSeries(
        geom_opt=_normalize_geom_opt(data.get("geom_opt")),
        surface_scan=_normalize_surface_scan(data.get("surface_scan")),
        goat=_normalize_goat(data.get("goat")),
        excited_state_optimization=excited_state_optimization,
    )

    if not any(
        (
            series.geom_opt,
            series.surface_scan,
            series.goat,
            series.excited_state_optimization,
        )
    ):
        return None
    return series


def get_job_series(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return the normalized job-series mapping, if present."""
    series = data.get("job_series")
    return series if isinstance(series, dict) else {}


def get_geom_opt_series(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return the canonical geometry-optimization history with fallback."""
    series = get_job_series(data).get("geom_opt")
    if isinstance(series, dict) and series:
        return series
    return _normalize_geom_opt(data.get("geom_opt"))


def get_surface_scan_series(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return the canonical relaxed-scan history with fallback."""
    series = get_job_series(data).get("surface_scan")
    if isinstance(series, dict) and series:
        return series
    return _normalize_surface_scan(data.get("surface_scan"))


def get_goat_series(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return the canonical GOAT ensemble history with fallback."""
    series = get_job_series(data).get("goat")
    if isinstance(series, dict) and series:
        return series
    return _normalize_goat(data.get("goat"))


def get_excited_state_optimization_series(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return the canonical excited-state optimization history with fallback."""
    series = get_job_series(data).get("excited_state_optimization")
    if isinstance(series, dict) and series:
        return series

    tddft = data.get("tddft")
    if isinstance(tddft, dict):
        return _normalize_excited_state_optimization(
            tddft.get("excited_state_optimization")
        )
    return {}
