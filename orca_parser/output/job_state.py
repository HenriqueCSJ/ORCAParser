"""Shared job-state helpers for output writers.

This module centralizes the normalization logic that both markdown and CSV
writers need for symmetry, DeltaSCF, excited-state optimizations, and scan
metadata. Keeping it in one place avoids the two output paths drifting apart
as new job types are added.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

from ..job_series import get_excited_state_optimization_series
from ..job_snapshot import get_job_snapshot


def get_deltascf_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return DeltaSCF metadata if this is an excited-state DeltaSCF job."""
    snapshot = get_job_snapshot(data)
    deltascf = snapshot.get("deltascf")
    if isinstance(deltascf, dict) and deltascf:
        return dict(deltascf)

    meta = data.get("metadata", {})
    if str(meta.get("calculation_type", "")).lower() != "deltascf":
        return {}
    return dict(meta.get("deltascf") or {})


def is_deltascf(data: Dict[str, Any]) -> bool:
    """Whether the parsed job is a DeltaSCF excited-state calculation."""
    snapshot = get_job_snapshot(data)
    if "is_deltascf" in snapshot:
        return bool(snapshot.get("is_deltascf"))

    calculation_type = str(data.get("metadata", {}).get("calculation_type", "")).lower()
    return bool(get_deltascf_data(data) or calculation_type == "deltascf")


def get_excited_state_opt_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return excited-state geometry-optimization metadata when available."""
    combined: Dict[str, Any] = {}
    series = get_excited_state_optimization_series(data)
    if isinstance(series, dict) and series:
        combined.update(deepcopy(series))

    snapshot = get_job_snapshot(data)
    excopt = snapshot.get("excited_state_optimization")
    if isinstance(excopt, dict) and excopt:
        # The series carries cycle history; the job snapshot carries the
        # normalized display metadata. Merge them so renderers read one shape.
        for key, value in excopt.items():
            if key not in combined or combined[key] in ("", None, [], {}):
                combined[key] = deepcopy(value)
    if combined:
        return combined

    meta = data.get("metadata", {})
    excopt = meta.get("excited_state_optimization")
    if isinstance(excopt, dict) and excopt:
        return dict(excopt)

    tddft = data.get("tddft", {})
    if isinstance(tddft, dict):
        excopt = tddft.get("excited_state_optimization")
        if isinstance(excopt, dict) and excopt:
            return dict(excopt)
    return {}


def is_excited_state_opt(data: Dict[str, Any]) -> bool:
    """Whether the parsed job is an excited-state geometry optimization."""
    snapshot = get_job_snapshot(data)
    if "is_excited_state_optimization" in snapshot:
        return bool(snapshot.get("is_excited_state_optimization"))
    return bool(get_excited_state_opt_data(data))


def excited_state_target_label(excopt: Dict[str, Any]) -> str:
    """Short state label such as S1/T2/root 3 for excited-state optimizations."""
    if not excopt:
        return ""
    label = str(excopt.get("target_state_label") or "").strip()
    if label:
        return label
    root = excopt.get("target_root")
    if root is None:
        root = excopt.get("final_root")
    if root is None:
        return ""
    return f"root {root}"


def electronic_state_label(data: Dict[str, Any], ground_state_label: str = "") -> str:
    """Return a short label describing the electronic state treatment."""
    snapshot = get_job_snapshot(data)
    special = snapshot.get("special_electronic_state_label")
    if special:
        return str(special)
    return ground_state_label


def yes_no_unknown(value: Any) -> str:
    """Render boolean-like values as yes/no/?."""
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "?"


def bool_to_label(value: Any) -> str:
    """Render bool-like values consistently for CSV exports."""
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return ""


def format_simple_vector(values: Optional[List[Any]]) -> str:
    """Compact comma-separated rendering for vectors used in metadata tables."""
    if not values:
        return ""
    rendered = []
    for value in values:
        if isinstance(value, float) and value.is_integer():
            rendered.append(str(int(value)))
        else:
            rendered.append(str(value))
    return ",".join(rendered)


def format_deltascf_vector(values: Any, formatter=None) -> str:
    """Compact formatter for occupation / configuration vectors."""
    if not isinstance(values, list) or not values:
        return ""

    parts: List[str] = []
    for value in values:
        if isinstance(value, int) and not isinstance(value, bool):
            parts.append(str(value))
            continue
        if isinstance(value, float) and value.is_integer():
            parts.append(str(int(value)))
            continue
        if formatter is not None:
            parts.append(formatter(value))
        else:
            parts.append(str(value))
    return " ".join(parts)


def format_deltascf_target(deltascf: Dict[str, Any], joiner: str = " | ") -> str:
    """Build a compact DeltaSCF target summary."""
    if not deltascf:
        return ""

    parts: List[str] = []
    for key in ("alphaconf", "betaconf"):
        values = deltascf.get(key)
        if values:
            parts.append(f"{key.upper()} {format_simple_vector(values)}")
    for key in ("ionizealpha", "ionizebeta"):
        value = deltascf.get(key)
        if value is not None:
            parts.append(f"{key.upper()} {value}")
    return joiner.join(parts)


def deltascf_target_summary(deltascf: Dict[str, Any], formatter=None) -> str:
    """One-line DeltaSCF target summary for markdown or comparison tables."""
    if not deltascf:
        return ""

    parts: List[str] = []
    for key in ("alphaconf", "betaconf"):
        values = deltascf.get(key)
        if isinstance(values, list) and values:
            vector = format_deltascf_vector(values, formatter=formatter).replace(" ", ",")
            parts.append(f"{key.upper()} {vector}")
    for key in ("ionizealpha", "ionizebeta"):
        value = deltascf.get(key)
        if value is not None:
            parts.append(f"{key.upper()} {value}")
    return "; ".join(parts)


def get_symmetry_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return normalized symmetry metadata assembled from metadata/geometry."""
    snapshot = get_job_snapshot(data)
    symmetry = snapshot.get("symmetry")
    if isinstance(symmetry, dict) and symmetry:
        return dict(symmetry)

    meta = data.get("metadata", {})
    geom = data.get("geometry", {})
    sym = dict(meta.get("symmetry") or {})

    if meta.get("point_group") and "point_group" not in sym:
        sym["point_group"] = meta["point_group"]
    if meta.get("reduced_point_group") and "reduced_point_group" not in sym:
        sym["reduced_point_group"] = meta["reduced_point_group"]
    if meta.get("orbital_irrep_group") and "orbital_irrep_group" not in sym:
        sym["orbital_irrep_group"] = meta["orbital_irrep_group"]
    if geom.get("symmetry_perfected_point_group") and "geometry_point_group" not in sym:
        sym["geometry_point_group"] = geom["symmetry_perfected_point_group"]
    if "point_group" not in sym and sym.get("geometry_point_group"):
        sym["point_group"] = sym["geometry_point_group"]

    return sym


def has_symmetry(data: Dict[str, Any]) -> bool:
    """Whether a parsed dataset carries symmetry information."""
    snapshot = get_job_snapshot(data)
    symmetry = snapshot.get("symmetry")
    if isinstance(symmetry, dict) and "has_symmetry" in symmetry:
        return bool(symmetry.get("has_symmetry"))

    sym = get_symmetry_data(data)
    return bool(
        sym.get("use_sym") is True
        or sym.get("point_group")
        or sym.get("auto_detected_point_group")
        or sym.get("reduced_point_group")
        or sym.get("orbital_irrep_group")
        or data.get("geometry", {}).get("symmetry_cartesian_au")
    )


def symmetry_on_off(sym: Dict[str, Any]) -> str:
    """Format UseSym as ON/OFF/? for tables."""
    if sym.get("use_sym_label"):
        return str(sym["use_sym_label"])
    if sym.get("use_sym") is True:
        return "ON"
    if sym.get("use_sym") is False:
        return "OFF"
    return "?"


def symmetry_inline_label(data: Dict[str, Any]) -> str:
    """Compact symmetry label, keeping full and reduced/orbital groups distinct."""
    sym = get_symmetry_data(data)
    if sym.get("symmetry_label"):
        return str(sym["symmetry_label"])
    if not has_symmetry(data):
        return ""

    point_group = (
        sym.get("point_group")
        or sym.get("auto_detected_point_group")
        or sym.get("geometry_point_group")
    )
    orbital_group = sym.get("orbital_irrep_group") or sym.get("reduced_point_group")

    if point_group and orbital_group and orbital_group != point_group:
        return f"{point_group} -> {orbital_group}"
    return orbital_group or point_group or ""


def has_symmetry_setup(sym: Dict[str, Any]) -> bool:
    """Whether a symmetry payload includes initial-guess cleanup metadata."""
    return any(
        key in sym
        for key in (
            "initial_guess_method",
            "initial_guess_source_file",
            "initial_guess_geometry_matches",
            "initial_guess_basis_matches",
            "initial_guess_irreps_reassigned",
            "initial_guess_mos_renormalized",
            "initial_guess_mos_reorthogonalized",
            "initial_guess_reorthogonalization_method",
        )
    )


def is_surface_scan(data: Dict[str, Any]) -> bool:
    """Whether this parsed job is a relaxed surface scan."""
    snapshot = get_job_snapshot(data)
    if "is_surface_scan" in snapshot:
        return bool(snapshot.get("is_surface_scan"))
    return bool(data.get("surface_scan"))


def calculation_type_label(data: Dict[str, Any]) -> str:
    """Return the normalized display label for the job's calculation type."""
    snapshot = get_job_snapshot(data)
    if snapshot.get("calculation_type"):
        return str(snapshot["calculation_type"])
    return str(data.get("metadata", {}).get("calculation_type", ""))


def get_job_name(data: Dict[str, Any]) -> str:
    """Return the normalized job name for display and exports."""
    snapshot = get_job_snapshot(data)
    if snapshot.get("job_name"):
        return str(snapshot["job_name"])
    return str(data.get("metadata", {}).get("job_name", ""))


def get_job_id(data: Dict[str, Any]) -> str:
    """Return the normalized unique job identifier."""
    snapshot = get_job_snapshot(data)
    if snapshot.get("job_id"):
        return str(snapshot["job_id"])
    return str(data.get("metadata", {}).get("job_id", "") or data.get("source_file", ""))


def get_basis_set(data: Dict[str, Any]) -> str:
    """Return the normalized primary basis set label."""
    snapshot = get_job_snapshot(data)
    if snapshot.get("basis_set"):
        return str(snapshot["basis_set"])
    return str(data.get("metadata", {}).get("basis_set", ""))


def get_aux_basis_set(data: Dict[str, Any]) -> str:
    """Return the normalized auxiliary basis set label."""
    snapshot = get_job_snapshot(data)
    if snapshot.get("aux_basis_set"):
        return str(snapshot["aux_basis_set"])
    return str(data.get("metadata", {}).get("aux_basis_set", ""))


def get_charge(data: Dict[str, Any]) -> Any:
    """Return the normalized total charge."""
    snapshot = get_job_snapshot(data)
    if "charge" in snapshot:
        return snapshot.get("charge")
    return data.get("metadata", {}).get("charge", "")


def get_multiplicity(data: Dict[str, Any]) -> Any:
    """Return the normalized spin multiplicity."""
    snapshot = get_job_snapshot(data)
    if "multiplicity" in snapshot:
        return snapshot.get("multiplicity")
    return data.get("metadata", {}).get("multiplicity", "")


def get_method_header_label(data: Dict[str, Any]) -> str:
    """Preferred method label for per-molecule markdown headers."""
    snapshot = get_job_snapshot(data)
    if snapshot.get("method_header_label"):
        return str(snapshot["method_header_label"])

    meta = data.get("metadata", {})
    context = data.get("context", {})
    if meta.get("level_of_theory"):
        return str(meta["level_of_theory"])
    functional = meta.get("functional", "?")
    basis = meta.get("basis_set", "?")
    return f"{context.get('hf_type', '?')} {functional}/{basis}"


def get_method_table_label(data: Dict[str, Any]) -> str:
    """Preferred method label for comparison tables."""
    snapshot = get_job_snapshot(data)
    if snapshot.get("method_table_label"):
        return str(snapshot["method_table_label"])

    meta = data.get("metadata", {})
    if meta.get("method"):
        return str(meta["method"])
    if meta.get("functional"):
        return str(meta["functional"])
    return str(data.get("context", {}).get("hf_type", "?"))
