"""Shared job-state helpers for output writers.

This module centralizes the normalization logic that both markdown and CSV
writers need for symmetry, DeltaSCF, excited-state optimizations, and scan
metadata. Keeping it in one place avoids the two output paths drifting apart
as new job types are added.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def get_deltascf_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return DeltaSCF metadata if this is an excited-state DeltaSCF job."""
    meta = data.get("metadata", {})
    if str(meta.get("calculation_type", "")).lower() != "deltascf":
        return {}
    return dict(meta.get("deltascf") or {})


def is_deltascf(data: Dict[str, Any]) -> bool:
    """Whether the parsed job is a DeltaSCF excited-state calculation."""
    calculation_type = str(data.get("metadata", {}).get("calculation_type", "")).lower()
    return bool(get_deltascf_data(data) or calculation_type == "deltascf")


def get_excited_state_opt_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return excited-state geometry-optimization metadata when available."""
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
    if is_deltascf(data):
        return "DeltaSCF excited-state"
    excopt = get_excited_state_opt_data(data)
    if excopt:
        target = excited_state_target_label(excopt)
        if target:
            return f"Excited-state optimization ({target})"
        return "Excited-state optimization"
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
    if data.get("context", {}).get("has_symmetry"):
        return True

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
    if not has_symmetry(data):
        return ""

    sym = get_symmetry_data(data)
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
    return bool(data.get("surface_scan"))
