"""Canonical final-state normalization for geometry-dependent properties.

This module builds a single authoritative "final snapshot" from the parser's
module outputs. The goal is to normalize which geometry-dependent values are
considered final once, during parsing, so downstream writers no longer need to
guess across multiple raw module payloads.

The snapshot is stored as ``data["final_snapshot"]`` for compatibility with
existing JSON outputs, while the helper accessors below let markdown/CSV code
prefer the normalized view with fallback to legacy raw sections during the
migration.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


_CHARGE_SCHEMES = ("mulliken", "loewdin", "hirshfeld", "mbis", "chelpg")


def _copy_mapping(value: Any) -> Dict[str, Any]:
    """Return a deep-copied dict or an empty mapping."""
    if isinstance(value, dict):
        return deepcopy(value)
    return {}


def _normalize_population_section(section: Any) -> Dict[str, Any]:
    """Normalize a population-analysis section to use a stable ``atoms`` key."""
    mapping = _copy_mapping(section)
    atoms = (
        mapping.get("atoms")
        or mapping.get("atomic_charges")
        or mapping.get("atomic_data")
        or []
    )
    if not isinstance(atoms, list):
        atoms = []
    normalized = {"atoms": deepcopy(atoms)}
    for key, value in mapping.items():
        if key in {"atoms", "atomic_charges", "atomic_data"}:
            continue
        normalized[key] = deepcopy(value)
    return normalized if normalized["atoms"] or len(normalized) > 1 else {}


def _normalize_mayer_section(section: Any) -> Dict[str, Any]:
    """Normalize Mayer data to stable ``atoms`` and ``bond_orders`` keys."""
    mapping = _copy_mapping(section)
    atoms = mapping.get("atoms") or mapping.get("atomic_data") or []
    bond_orders = mapping.get("bond_orders") or []
    normalized = {
        "atoms": deepcopy(atoms if isinstance(atoms, list) else []),
        "bond_orders": deepcopy(bond_orders if isinstance(bond_orders, list) else []),
    }
    for key, value in mapping.items():
        if key in {"atoms", "atomic_data", "bond_orders"}:
            continue
        normalized[key] = deepcopy(value)
    return normalized if normalized["atoms"] or normalized["bond_orders"] or len(normalized) > 2 else {}


@dataclass
class FinalSnapshot:
    """Canonical final-state view for geometry-dependent sections."""

    selection: str
    geometry: Dict[str, Any] = field(default_factory=dict)
    orbital_energies: Dict[str, Any] = field(default_factory=dict)
    dipole: Dict[str, Any] = field(default_factory=dict)
    charges: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    mayer: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the snapshot to a JSON-safe mapping."""
        return {
            "selection": self.selection,
            "geometry": deepcopy(self.geometry),
            "orbital_energies": deepcopy(self.orbital_energies),
            "dipole": deepcopy(self.dipole),
            "charges": deepcopy(self.charges),
            "mayer": deepcopy(self.mayer),
        }


def _selection_label(data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
    """Describe how the final snapshot was selected."""
    context = context or {}
    meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    calc_type = str(meta.get("calculation_type", "")).lower()

    if data.get("geom_opt") or "optimization" in calc_type:
        return "last_reported_optimization_step"
    if context.get("is_surface_scan") or data.get("surface_scan"):
        return "last_reported_scan_step"
    return "single_reported_state"


def build_final_snapshot(
    data: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> Optional[FinalSnapshot]:
    """Build a canonical final-state snapshot from parsed module payloads."""
    geometry = _copy_mapping(data.get("geometry"))
    orbital_energies = _copy_mapping(data.get("orbital_energies"))
    dipole = _copy_mapping(data.get("dipole"))
    charges = {
        scheme: normalized
        for scheme in _CHARGE_SCHEMES
        if (normalized := _normalize_population_section(data.get(scheme)))
    }
    mayer = _normalize_mayer_section(data.get("mayer"))

    if not any((geometry, orbital_energies, dipole, charges, mayer)):
        return None

    return FinalSnapshot(
        selection=_selection_label(data, context=context),
        geometry=geometry,
        orbital_energies=orbital_energies,
        dipole=dipole,
        charges=charges,
        mayer=mayer,
    )


def get_final_snapshot(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return the normalized final snapshot mapping, if present."""
    snapshot = data.get("final_snapshot")
    return snapshot if isinstance(snapshot, dict) else {}


def get_final_geometry(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return the canonical final geometry block with legacy fallback."""
    snapshot = get_final_snapshot(data).get("geometry")
    if isinstance(snapshot, dict) and snapshot:
        return snapshot
    return _copy_mapping(data.get("geometry"))


def get_final_orbital_energies(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return the canonical final orbital-energy block with legacy fallback."""
    snapshot = get_final_snapshot(data).get("orbital_energies")
    if isinstance(snapshot, dict) and snapshot:
        return snapshot
    return _copy_mapping(data.get("orbital_energies"))


def get_final_dipole(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return the canonical final dipole block with legacy fallback."""
    snapshot = get_final_snapshot(data).get("dipole")
    if isinstance(snapshot, dict) and snapshot:
        return snapshot
    return _copy_mapping(data.get("dipole"))


def get_final_population_section(data: Dict[str, Any], scheme: str) -> Dict[str, Any]:
    """Return a normalized final population section for one charge scheme."""
    charges = get_final_snapshot(data).get("charges")
    if isinstance(charges, dict):
        section = charges.get(scheme)
        if isinstance(section, dict) and section:
            return section
    return _normalize_population_section(data.get(scheme))


def get_final_mayer_section(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return a normalized final Mayer section with stable keys."""
    snapshot = get_final_snapshot(data).get("mayer")
    if isinstance(snapshot, dict) and snapshot:
        return snapshot
    return _normalize_mayer_section(data.get("mayer"))
