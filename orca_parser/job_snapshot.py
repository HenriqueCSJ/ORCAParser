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

from .job_family_registry import match_calculation_family_plugin


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

    reference_type = context.get("reference_type") or context.get("hf_type", "?")
    functional = meta.get("functional", "?")
    basis = meta.get("basis_set", "?")
    return f"{reference_type} {functional}/{basis}"


def _method_table_label(meta: Dict[str, Any], context: Dict[str, Any]) -> str:
    """Preferred method label for comparison tables."""
    if meta.get("method"):
        return str(meta["method"])
    if meta.get("functional"):
        return str(meta["functional"])
    return str(context.get("reference_type") or context.get("hf_type", "?"))


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
    reference_type: str
    is_unrestricted: bool
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
            "reference_type": self.reference_type,
            "is_unrestricted": self.is_unrestricted,
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
    # The registry owns family matching and labels.  Adding a new calculation
    # family should now mean registering one plugin instead of expanding
    # snapshot / markdown / CSV branching separately.
    family_plugin = match_calculation_family_plugin(
        meta,
        data,
        context,
        deltascf,
        excopt,
    )
    family = family_plugin.family

    electronic_state_kind = family_plugin.electronic_state_kind
    special_electronic_state_label = family_plugin.special_electronic_state_label(
        meta,
        data,
        deltascf,
        excopt,
    )
    state_target_label = family_plugin.state_target_label(
        meta,
        data,
        deltascf,
        excopt,
    )

    # The family registry owns the normalized label that downstream renderers
    # should use.  Raw metadata may still carry the original parser payload,
    # but adding a new family should not require updating label logic in
    # several modules.
    calculation_type = str(family_plugin.default_calculation_label)

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
        reference_type=str(meta.get("reference_type", context.get("reference_type", ""))),
        is_unrestricted=bool(context.get("is_unrestricted", context.get("is_uhf", False))),
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
