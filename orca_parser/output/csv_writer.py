"""
CSV output writers for ORCA parser results.

Each logical section that contains tabular data is written to its own CSV file.
The filename follows the pattern:  {job_name}_{section}.csv

Sections exported:
  - geometry          : Cartesian coordinates (Å)
  - orbital_energies  : per spin (alpha/beta for UHF)
  - qro               : QRO energies and occupation
  - mulliken          : atomic charges (and spin for UHF)
  - mulliken_orb      : reduced orbital charges
  - loewdin           : atomic charges
  - loewdin_orb       : reduced orbital charges
  - mayer             : Mayer atomic populations + bond orders
  - hirshfeld         : Hirshfeld charges (+ spin)
  - mbis              : MBIS charges (+ spin + valence)
  - chelpg            : CHELPG electrostatic charges
  - nbo_nao           : NAO occupancies (overall; spin columns for UHF)
  - nbo_npa           : NPA summary (overall + spin density for UHF)
  - nbo_wiberg        : Wiberg bond index matrix
  - nbo_nbi           : NBI matrix
  - nbo_lewis         : NBO Lewis-structure orbitals
  - nbo_e2            : E2 perturbation theory
  - nbo_nlmo_hyb      : NLMO hybridization/polarization
  - nbo_nlmo_bo       : NLMO bond orders
  - nbo_nlmo_steric   : NLMO steric exchange energies
  - alpha/beta variants where applicable (UHF)
"""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..final_snapshot import (
    get_final_mayer_section as _get_final_mayer_section,
    get_final_orbital_energies as _get_final_orbital_energies,
    get_final_population_section as _get_final_population_section,
)
from ..job_snapshot import get_job_snapshot as _get_job_snapshot
from .job_state import (
    bool_to_label as _bool_to_label,
    electronic_state_label as _shared_electronic_state_label,
    excited_state_target_label as _excited_state_target_label,
    format_deltascf_target as _format_deltascf_target,
    format_simple_vector as _format_simple_vector,
    get_deltascf_data as _get_deltascf_data,
    get_excited_state_opt_data as _get_excited_state_opt_data,
    get_symmetry_data as _get_symmetry_data,
    is_deltascf as _is_deltascf,
    is_surface_scan as _is_surface_scan,
)
from .csv_sections_basic import (
    write_dipole_section as _write_basic_dipole_section,
    write_geom_opt_section as _write_basic_geom_opt_section,
    write_geometry_section as _write_basic_geometry_section,
    write_goat_section as _write_basic_goat_section,
    write_solvation_section as _write_basic_solvation_section,
    write_surface_scan_section as _write_basic_surface_scan_section,
)
from .csv_sections_nbo import (
    write_nbo_e2_section as _write_nbo_e2_section,
    write_nbo_lewis_section as _write_nbo_lewis_section,
    write_nbo_matrix_section as _write_nbo_matrix_section,
    write_nbo_nao_section as _write_nbo_nao_section,
    write_nbo_nlmo_bond_order_section as _write_nbo_nlmo_bond_order_section,
    write_nbo_nlmo_hybridization_section as _write_nbo_nlmo_hybridization_section,
    write_nbo_nlmo_steric_section as _write_nbo_nlmo_steric_section,
    write_nbo_npa_section as _write_nbo_npa_section,
)
from .csv_sections_spectroscopy import (
    write_epr_section as _write_spectroscopy_epr_section,
    write_tddft_section as _write_spectroscopy_tddft_section,
)
from .csv_sections_state import (
    write_deltascf_section as _write_state_deltascf_section,
    write_excited_state_optimization_section as _write_state_excited_state_optimization_section,
    write_metadata_section as _write_state_metadata_section,
    write_symmetry_section as _write_state_symmetry_section,
)


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _stem(data: Dict[str, Any]) -> str:
    """Job-name stem derived from the source file or metadata."""
    snapshot = _get_job_snapshot(data)
    jn = snapshot.get("job_name")
    if jn:
        return str(jn)

    meta = data.get("metadata", {})
    jn = meta.get("job_name")
    if jn:
        return jn
    src = data.get("source_file", "orca")
    return Path(src).stem


def _write_csv(directory: Path, filename: str, rows: List[Dict], fieldnames: List[str]) -> Path:
    """Write a list of row dicts to *directory/filename*."""
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / filename
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return out


def _flatten_vector(d: Optional[Dict]) -> Dict[str, float]:
    """Convert {'x': 1, 'y': 2, 'z': 3} → {'x': 1, 'y': 2, 'z': 3}."""
    if d is None:
        return {}
    return {k: v for k, v in d.items() if k in ("x", "y", "z")}


def _electronic_state_label(data: Dict[str, Any]) -> str:
    """Short electronic-state label for metadata exports."""
    return _shared_electronic_state_label(data, ground_state_label="Ground-state")


# ─────────────────────────────────────────────────────────────────
# Section writers
# ─────────────────────────────────────────────────────────────────

def _write_geometry(data, directory, stem) -> List[Path]:
    return _write_basic_geometry_section(
        data,
        directory,
        stem,
        write_csv=_write_csv,
    )


def _write_metadata(data: Dict[str, Any], directory: Path, stem: str) -> List[Path]:
    """Write a one-row metadata summary for downstream filtering/grouping."""
    return _write_state_metadata_section(
        data,
        directory,
        stem,
        write_csv=_write_csv,
        electronic_state_label=_electronic_state_label,
        get_symmetry_data=_get_symmetry_data,
        get_deltascf_data=_get_deltascf_data,
        get_excited_state_opt_data=_get_excited_state_opt_data,
        bool_to_label=_bool_to_label,
        is_surface_scan=_is_surface_scan,
        format_deltascf_target=_format_deltascf_target,
        excited_state_target_label=_excited_state_target_label,
        get_job_snapshot=_get_job_snapshot,
    )


def _write_symmetry(data: Dict[str, Any], directory: Path, stem: str) -> List[Path]:
    """Write symmetry summary plus irrep-resolved details when available."""
    return _write_state_symmetry_section(
        data,
        directory,
        stem,
        write_csv=_write_csv,
        get_symmetry_data=_get_symmetry_data,
        bool_to_label=_bool_to_label,
    )


def _write_deltascf(data: Dict[str, Any], directory: Path, stem: str) -> List[Path]:
    """Write DeltaSCF excited-state target metadata when present."""
    return _write_state_deltascf_section(
        data,
        directory,
        stem,
        write_csv=_write_csv,
        electronic_state_label=_electronic_state_label,
        get_deltascf_data=_get_deltascf_data,
        is_deltascf=_is_deltascf,
        format_deltascf_target=_format_deltascf_target,
        format_simple_vector=_format_simple_vector,
        bool_to_label=_bool_to_label,
    )


def _write_excited_state_optimization(
    data: Dict[str, Any], directory: Path, stem: str
) -> List[Path]:
    """Write excited-state geometry-optimization metadata and cycle history."""
    return _write_state_excited_state_optimization_section(
        data,
        directory,
        stem,
        write_csv=_write_csv,
        electronic_state_label=_electronic_state_label,
        get_excited_state_opt_data=_get_excited_state_opt_data,
        excited_state_target_label=_excited_state_target_label,
        bool_to_label=_bool_to_label,
        format_simple_vector=_format_simple_vector,
    )


def _write_orbital_energies(data, directory, stem) -> List[Path]:
    oe = _get_final_orbital_energies(data)
    files = []

    # RHF single set
    if "orbitals" in oe:
        rows = []
        for orb in oe["orbitals"]:
            rows.append({
                "index": orb.get("index"),
                "occupation": orb.get("occupation"),
                "energy_Eh": orb.get("energy_Eh"),
                "energy_eV": orb.get("energy_eV"),
                "irrep": orb.get("irrep", ""),
            })
        files.append(_write_csv(directory, f"{stem}_orbital_energies.csv", rows,
                                ["index", "occupation", "energy_Eh", "energy_eV", "irrep"]))

    # UHF: alpha and beta
    for spin in ("alpha", "beta"):
        key = f"{spin}_orbitals"
        if key in oe:
            rows = []
            for orb in oe[key]:
                rows.append({
                    "index": orb.get("index"),
                    "occupation": orb.get("occupation"),
                    "energy_Eh": orb.get("energy_Eh"),
                    "energy_eV": orb.get("energy_eV"),
                    "irrep": orb.get("irrep", ""),
                })
            files.append(_write_csv(
                directory, f"{stem}_orbital_energies_{spin}.csv", rows,
                ["index", "occupation", "energy_Eh", "energy_eV", "irrep"]
            ))
    return files


def _write_qro(data, directory, stem) -> List[Path]:
    qro = data.get("qro")
    if not qro:
        return []
    rows = []
    for orb in qro.get("orbitals", []):
        rows.append({
            "index": orb.get("index"),
            "occupation": orb.get("occupation"),
            "type": orb.get("type", ""),
            "energy_Eh": orb.get("energy_Eh"),
            "energy_eV": orb.get("energy_eV"),
            "alpha_energy_eV": orb.get("alpha_energy_eV", ""),
            "beta_energy_eV": orb.get("beta_energy_eV", ""),
        })
    fn = _write_csv(directory, f"{stem}_qro.csv", rows,
                    ["index", "occupation", "type", "energy_Eh", "energy_eV",
                     "alpha_energy_eV", "beta_energy_eV"])
    return [fn]


def _write_mulliken(data, directory, stem) -> List[Path]:
    ml = _get_final_population_section(data, "mulliken")
    files = []
    atoms = _get_pop_atoms(ml)
    if atoms:
        rows = [{
            "index": a.get("index"), "symbol": a.get("symbol"),
            "charge": a.get("charge"), "spin_population": a.get("spin_population", ""),
        } for a in atoms]
        files.append(_write_csv(directory, f"{stem}_mulliken_charges.csv", rows,
                                ["index", "symbol", "charge", "spin_population"]))

    orb_data = ml.get("reduced_orbital_charges", [])
    if orb_data:
        rows2 = []
        for atom in orb_data:
            for cont in atom.get("contributions", []):
                rows2.append({
                    "atom_index": atom.get("index"),
                    "atom_symbol": atom.get("symbol"),
                    "angular": cont.get("angular"),
                    "charge": cont.get("charge"),
                    "spin": cont.get("spin", ""),
                })
        if rows2:
            files.append(_write_csv(directory, f"{stem}_mulliken_orb.csv", rows2,
                                    ["atom_index", "atom_symbol", "angular", "charge", "spin"]))
    return files


def _write_loewdin(data, directory, stem) -> List[Path]:
    lo = _get_final_population_section(data, "loewdin")
    files = []
    atoms = _get_pop_atoms(lo)
    if atoms:
        rows = [{
            "index": a.get("index"), "symbol": a.get("symbol"),
            "charge": a.get("charge"), "spin_population": a.get("spin_population", ""),
        } for a in atoms]
        files.append(_write_csv(directory, f"{stem}_loewdin_charges.csv", rows,
                                ["index", "symbol", "charge", "spin_population"]))
    orb_data = lo.get("reduced_orbital_charges", [])
    if orb_data:
        rows2 = []
        for atom in orb_data:
            for cont in atom.get("contributions", []):
                rows2.append({
                    "atom_index": atom.get("index"),
                    "atom_symbol": atom.get("symbol"),
                    "angular": cont.get("angular"),
                    "charge": cont.get("charge"),
                })
        if rows2:
            files.append(_write_csv(directory, f"{stem}_loewdin_orb.csv", rows2,
                                    ["atom_index", "atom_symbol", "angular", "charge"]))
    return files


def _write_mayer(data, directory, stem) -> List[Path]:
    ma = _get_final_mayer_section(data)
    files = []
    atoms = ma.get("atoms", [])
    if atoms:
        rows = [{
            "index": a.get("index"), "symbol": a.get("symbol"),
            "NA": a.get("NA"), "ZA": a.get("ZA"), "QA": a.get("QA"),
            "VA": a.get("VA"), "BVA": a.get("BVA"), "FA": a.get("FA"),
        } for a in atoms]
        files.append(_write_csv(directory, f"{stem}_mayer_atoms.csv", rows,
                                ["index", "symbol", "NA", "ZA", "QA", "VA", "BVA", "FA"]))
    bonds = ma.get("bond_orders", [])
    if bonds:
        files.append(_write_csv(directory, f"{stem}_mayer_bonds.csv", bonds,
                                ["atom_i", "symbol_i", "atom_j", "symbol_j", "bond_order"]))
    return files


def _get_pop_atoms(sec: dict) -> list:
    """Return atom list from a population section regardless of internal key name.

    Mulliken/Loewdin/CHELPG use 'atomic_charges';
    Hirshfeld/MBIS use 'atomic_data';
    fallback: 'atoms'.
    """
    return sec.get("atoms") or sec.get("atomic_charges") or sec.get("atomic_data") or []


def _write_hirshfeld(data, directory, stem) -> List[Path]:
    hi    = _get_final_population_section(data, "hirshfeld")
    atoms = _get_pop_atoms(hi)
    if not atoms:
        return []
    has_spin = any("spin_population" in a for a in atoms)
    rows = [{
        "index":          a.get("index"),
        "symbol":         a.get("symbol"),
        "charge":         a.get("charge"),
        "spin_population": a.get("spin_population", "") if has_spin else "",
    } for a in atoms]
    cols = ["index", "symbol", "charge"] + (["spin_population"] if has_spin else [])
    fn = _write_csv(directory, f"{stem}_hirshfeld.csv", rows, cols)
    return [fn]


def _write_mbis(data, directory, stem) -> List[Path]:
    mb    = _get_final_population_section(data, "mbis")
    atoms = _get_pop_atoms(mb)
    files = []
    if atoms:
        has_spin = any("spin_population" in a for a in atoms)
        rows = [{
            "index":           a.get("index"),
            "symbol":          a.get("symbol"),
            "charge":          a.get("charge"),
            "population":      a.get("population", ""),
            "spin_population": a.get("spin_population", "") if has_spin else "",
        } for a in atoms]
        cols = ["index", "symbol", "charge", "population"] + (["spin_population"] if has_spin else [])
        files.append(_write_csv(directory, f"{stem}_mbis.csv", rows, cols))
    valence = mb.get("valence_shell", [])
    if valence:
        rows2 = [{
            "index": v.get("index"), "symbol": v.get("symbol"),
            "population": v.get("population"), "width_au": v.get("width_au"),
        } for v in valence]
        files.append(_write_csv(directory, f"{stem}_mbis_valence.csv", rows2,
                                ["index", "symbol", "population", "width_au"]))
    return files


def _write_chelpg(data, directory, stem) -> List[Path]:
    ch    = _get_final_population_section(data, "chelpg")
    atoms = _get_pop_atoms(ch)
    if not atoms:
        return []
    rows = [{
        "index": a.get("index"), "symbol": a.get("symbol"),
        "charge": a.get("charge"),
    } for a in atoms]
    fn = _write_csv(directory, f"{stem}_chelpg.csv", rows,
                    ["index", "symbol", "charge"])
    return [fn]


def _write_nbo_nao(data, directory, stem, spin_label="") -> List[Path]:
    del spin_label
    return _write_nbo_nao_section(
        data,
        directory,
        stem,
        write_csv=_write_csv,
    )


def _write_nbo_npa(data, directory, stem) -> List[Path]:
    return _write_nbo_npa_section(
        data,
        directory,
        stem,
        write_csv=_write_csv,
    )


def _write_nbo_matrix(data, directory, stem, key, filename_suffix) -> List[Path]:
    return _write_nbo_matrix_section(
        data,
        directory,
        stem,
        write_csv=_write_csv,
        key=key,
        filename_suffix=filename_suffix,
    )


def _write_nbo_lewis(data, directory, stem) -> List[Path]:
    return _write_nbo_lewis_section(
        data,
        directory,
        stem,
        write_csv=_write_csv,
    )


def _write_nbo_e2(data, directory, stem) -> List[Path]:
    return _write_nbo_e2_section(
        data,
        directory,
        stem,
        write_csv=_write_csv,
    )


def _write_nbo_nlmo_hyb(data, directory, stem) -> List[Path]:
    return _write_nbo_nlmo_hybridization_section(
        data,
        directory,
        stem,
        write_csv=_write_csv,
    )


def _write_nbo_nlmo_bo(data, directory, stem) -> List[Path]:
    return _write_nbo_nlmo_bond_order_section(
        data,
        directory,
        stem,
        write_csv=_write_csv,
    )


def _write_nbo_nlmo_steric(data, directory, stem) -> List[Path]:
    return _write_nbo_nlmo_steric_section(
        data,
        directory,
        stem,
        write_csv=_write_csv,
    )


def _write_epr(data, directory, stem) -> List[Path]:
    return _write_spectroscopy_epr_section(
        data,
        directory,
        stem,
        write_csv=_write_csv,
    )


def _write_dipole(data, directory, stem) -> List[Path]:
    return _write_basic_dipole_section(
        data,
        directory,
        stem,
        write_csv=_write_csv,
    )


def _write_tddft(data, directory, stem) -> List[Path]:
    return _write_spectroscopy_tddft_section(
        data,
        directory,
        stem,
        write_csv=_write_csv,
        bool_to_label=_bool_to_label,
    )


def _write_solvation(data, directory, stem) -> List[Path]:
    return _write_basic_solvation_section(
        data,
        directory,
        stem,
        write_csv=_write_csv,
    )


def _write_geom_opt(data: Dict[str, Any], directory: Path, stem: str) -> List[Path]:
    return _write_basic_geom_opt_section(
        data,
        directory,
        stem,
        write_csv=_write_csv,
    )


def _write_surface_scan(data: Dict[str, Any], directory: Path, stem: str) -> List[Path]:
    return _write_basic_surface_scan_section(
        data,
        directory,
        stem,
        write_csv=_write_csv,
        format_simple_vector=_format_simple_vector,
    )


def _write_goat(data: Dict[str, Any], directory: Path, stem: str) -> List[Path]:
    return _write_basic_goat_section(
        data,
        directory,
        stem,
        write_csv=_write_csv,
    )


# ─────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────

def write_csvs(data: Dict[str, Any], directory: Path) -> List[Path]:
    """
    Write all tabular sections to individual CSV files in *directory*.

    Parameters
    ----------
    data : dict
        Output from :class:`orca_parser.ORCAParser.parse`.
    directory : Path
        Output directory (created if it does not exist).

    Returns
    -------
    list of Path
        All files written.
    """
    directory = Path(directory)
    stem = _stem(data)
    written: List[Path] = []

    writers = [
        _write_metadata,
        _write_geometry,
        _write_symmetry,
        _write_deltascf,
        _write_excited_state_optimization,
        _write_orbital_energies,
        _write_qro,
        _write_mulliken,
        _write_loewdin,
        _write_mayer,
        _write_hirshfeld,
        _write_mbis,
        _write_chelpg,
        _write_nbo_nao,
        _write_nbo_npa,
        _write_dipole,
        _write_solvation,
        _write_tddft,
        _write_nbo_lewis,
        _write_nbo_e2,
        _write_nbo_nlmo_hyb,
        _write_nbo_nlmo_bo,
        _write_nbo_nlmo_steric,
        _write_epr,
        _write_goat,
        _write_surface_scan,
        _write_geom_opt,
    ]

    for writer in writers:
        try:
            files = writer(data, directory, stem)
            written.extend(files)
        except Exception:  # noqa: BLE001
            pass  # Section absent or parse issue; continue silently

    # Wiberg and NBI matrices
    try:
        written.extend(_write_nbo_matrix(data, directory, stem, "wiberg_matrix", "wiberg"))
    except Exception:
        pass
    try:
        written.extend(_write_nbo_matrix(data, directory, stem, "nbi_matrix", "nbi"))
    except Exception:
        pass

    return written
