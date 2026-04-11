"""Canonical normalization for job identity, state, and method labels.

This module builds a single authoritative "job snapshot" from the parser's
metadata and context. The goal mirrors ``final_snapshot``: normalize the
meaning of job type, electronic state, symmetry, and display labels once
during parsing so downstream code does not keep re-deriving those semantics
from raw metadata strings.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


def _copy_mapping(value: Any) -> Dict[str, Any]:
    """Return a deep-copied dict or an empty mapping."""
    if isinstance(value, dict):
        return deepcopy(value)
    return {}


def _get_symmetry_payload(
    meta: Dict[str, Any],
    data: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Normalize symmetry metadata to a stable shape."""
    # ORCA spreads symmetry hints across multiple places. Merge them once here
    # so every downstream renderer sees the same payload and label policy.
    context = context or {}
    geom = data.get("geometry") if isinstance(data.get("geometry"), dict) else {}
    sym = _copy_mapping(meta.get("symmetry"))

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
    if "input_use_sym" not in sym and "input_use_sym" in meta:
        sym["input_use_sym"] = meta.get("input_use_sym")

    point_group = (
        sym.get("point_group")
        or sym.get("auto_detected_point_group")
        or sym.get("geometry_point_group")
        or ""
    )
    orbital_group = sym.get("orbital_irrep_group") or sym.get("reduced_point_group") or ""
    if point_group and orbital_group and orbital_group != point_group:
        symmetry_label = f"{point_group} -> {orbital_group}"
    else:
        symmetry_label = orbital_group or point_group or ""

    has_symmetry = bool(
        context.get("has_symmetry")
        or sym.get("use_sym") is True
        or point_group
        or orbital_group
        or geom.get("symmetry_cartesian_au")
    )

    sym["symmetry_label"] = symmetry_label
    sym["has_symmetry"] = has_symmetry
    return sym


def _get_deltascf_payload(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Return normalized DeltaSCF metadata."""
    return _copy_mapping(meta.get("deltascf"))


def _get_excited_state_opt_payload(meta: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Return normalized excited-state optimization metadata."""
    excopt = _copy_mapping(meta.get("excited_state_optimization"))
    if excopt:
        return excopt

    tddft = data.get("tddft")
    if isinstance(tddft, dict):
        excopt = _copy_mapping(tddft.get("excited_state_optimization"))
        if excopt:
            return excopt
    return {}


def _excited_state_target_label(excopt: Dict[str, Any]) -> str:
    """Short target label such as S1/T2/root 3."""
    label = str(excopt.get("target_state_label") or "").strip()
    if label:
        return label
    root = excopt.get("target_root")
    if root is None:
        root = excopt.get("final_root")
    if root is None:
        return ""
    return f"root {root}"


def _method_header_label(meta: Dict[str, Any], context: Dict[str, Any]) -> str:
    """Preferred single-line method label for molecule headers."""
    if meta.get("level_of_theory"):
        return str(meta["level_of_theory"])

    functional = meta.get("functional")
    basis = meta.get("basis_set")
    if functional and basis:
        return f"{functional}/{basis}"

    method = meta.get("method")
    if method and basis:
        return f"{method}/{basis}"
    if method:
        return str(method)

    hf_type = context.get("hf_type", "?")
    functional = meta.get("functional", "?")
    basis = meta.get("basis_set", "?")
    return f"{hf_type} {functional}/{basis}"


def _method_table_label(meta: Dict[str, Any], context: Dict[str, Any]) -> str:
    """Preferred method label for comparison tables."""
    if meta.get("method"):
        return str(meta["method"])
    if meta.get("functional"):
        return str(meta["functional"])
    return str(context.get("hf_type", "?"))


def _calculation_family(
    meta: Dict[str, Any],
    data: Dict[str, Any],
    context: Dict[str, Any],
    *,
    is_deltascf: bool,
    is_excited_state_optimization: bool,
) -> str:
    """Return a stable internal job-family identifier."""
    calc_type = str(meta.get("calculation_type", "")).strip().lower()

    if data.get("goat") or context.get("is_goat") or "goat" in calc_type:
        return "goat"
    if data.get("surface_scan") or context.get("is_surface_scan"):
        return "surface_scan"
    if is_deltascf:
        return "deltascf"
    if is_excited_state_optimization:
        return "excited_state_optimization"
    if "geometry optimization" in calc_type:
        return "geometry_optimization"
    if "single point" in calc_type:
        return "single_point"
    if calc_type:
        return calc_type.replace(" ", "_").replace("-", "_")
    return "unknown"


def _default_calculation_label(family: str) -> str:
    """Fallback display label for a normalized job family."""
    labels = {
        "goat": "GOAT Conformer Search",
        "surface_scan": "Relaxed Surface Scan",
        "deltascf": "DeltaSCF",
        "excited_state_optimization": "Excited-State Geometry Optimization",
        "geometry_optimization": "Geometry Optimization",
        "single_point": "Single Point",
    }
    return labels.get(family, family.replace("_", " ").title())


@dataclass
class JobSnapshot:
    """Canonical job/state metadata for downstream rendering."""

    job_id: str
    job_name: str
    source_file: str
    source_relpath: str
    calculation_type: str
    calculation_family: str
    method: str
    functional: str
    level_of_theory: str
    method_header_label: str
    method_table_label: str
    basis_set: str
    aux_basis_set: str
    hf_type: str
    charge: Any
    multiplicity: Any
    is_deltascf: bool = False
    is_excited_state_optimization: bool = False
    is_surface_scan: bool = False
    is_goat: bool = False
    electronic_state_kind: str = "ground_state"
    special_electronic_state_label: str = ""
    state_target_label: str = ""
    symmetry: Dict[str, Any] = field(default_factory=dict)
    deltascf: Dict[str, Any] = field(default_factory=dict)
    excited_state_optimization: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the snapshot to a JSON-safe mapping."""
        return {
            "job_id": self.job_id,
            "job_name": self.job_name,
            "source_file": self.source_file,
            "source_relpath": self.source_relpath,
            "calculation_type": self.calculation_type,
            "calculation_family": self.calculation_family,
            "method": self.method,
            "functional": self.functional,
            "level_of_theory": self.level_of_theory,
            "method_header_label": self.method_header_label,
            "method_table_label": self.method_table_label,
            "basis_set": self.basis_set,
            "aux_basis_set": self.aux_basis_set,
            "hf_type": self.hf_type,
            "charge": self.charge,
            "multiplicity": self.multiplicity,
            "is_deltascf": self.is_deltascf,
            "is_excited_state_optimization": self.is_excited_state_optimization,
            "is_surface_scan": self.is_surface_scan,
            "is_goat": self.is_goat,
            "electronic_state_kind": self.electronic_state_kind,
            "special_electronic_state_label": self.special_electronic_state_label,
            "state_target_label": self.state_target_label,
            "symmetry": deepcopy(self.symmetry),
            "deltascf": deepcopy(self.deltascf),
            "excited_state_optimization": deepcopy(self.excited_state_optimization),
        }


def build_job_snapshot(
    data: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> Optional[JobSnapshot]:
    """Build a canonical job snapshot from parsed metadata and context."""
    meta = data.get("metadata")
    if not isinstance(meta, dict) or not meta:
        return None

    context = context or {}
    deltascf = _get_deltascf_payload(meta)
    excopt = _get_excited_state_opt_payload(meta, data)
    symmetry = _get_symmetry_payload(meta, data, context=context)

    is_deltascf = bool(deltascf or str(meta.get("calculation_type", "")).strip().lower() == "deltascf")
    is_excited_state_optimization = bool(excopt)
    # Keep the family small and stable. It is meant for routing output logic
    # and regressions, not for preserving every free-form ORCA phrase.
    family = _calculation_family(
        meta,
        data,
        context,
        is_deltascf=is_deltascf,
        is_excited_state_optimization=is_excited_state_optimization,
    )

    if is_deltascf:
        electronic_state_kind = "deltascf_excited_state"
        special_electronic_state_label = "DeltaSCF excited-state"
        state_target_label = ""
    elif is_excited_state_optimization:
        electronic_state_kind = "excited_state_optimization"
        state_target_label = _excited_state_target_label(excopt)
        if state_target_label:
            special_electronic_state_label = f"Excited-state optimization ({state_target_label})"
        else:
            special_electronic_state_label = "Excited-state optimization"
    else:
        electronic_state_kind = "ground_state"
        special_electronic_state_label = ""
        state_target_label = ""

    calculation_type = str(meta.get("calculation_type") or _default_calculation_label(family))

    # Snapshot-level labels deliberately live here so markdown, CSV, and CLI
    # summaries do not re-derive presentation rules independently.
    return JobSnapshot(
        job_id=str(meta.get("job_id", context.get("job_id", ""))),
        job_name=str(meta.get("job_name", "")),
        source_file=str(data.get("source_file", "")),
        source_relpath=str(meta.get("source_relpath", context.get("source_relpath", ""))),
        calculation_type=calculation_type,
        calculation_family=family,
        method=str(meta.get("method", "")),
        functional=str(meta.get("functional", "")),
        level_of_theory=str(meta.get("level_of_theory", "")),
        method_header_label=_method_header_label(meta, context),
        method_table_label=_method_table_label(meta, context),
        basis_set=str(meta.get("basis_set", "")),
        aux_basis_set=str(meta.get("aux_basis_set", "")),
        hf_type=str(meta.get("hf_type", context.get("hf_type", ""))),
        charge=meta.get("charge", context.get("charge", "")),
        multiplicity=meta.get("multiplicity", context.get("multiplicity", "")),
        is_deltascf=is_deltascf,
        is_excited_state_optimization=is_excited_state_optimization,
        is_surface_scan=family == "surface_scan",
        is_goat=family == "goat",
        electronic_state_kind=electronic_state_kind,
        special_electronic_state_label=special_electronic_state_label,
        state_target_label=state_target_label,
        symmetry=symmetry,
        deltascf=deltascf,
        excited_state_optimization=excopt,
    )


def get_job_snapshot(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return the normalized job snapshot mapping, if present."""
    snapshot = data.get("job_snapshot")
    return snapshot if isinstance(snapshot, dict) else {}
