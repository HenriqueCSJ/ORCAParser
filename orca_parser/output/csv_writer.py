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


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _stem(data: Dict[str, Any]) -> str:
    """Job-name stem derived from the source file or metadata."""
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


# ─────────────────────────────────────────────────────────────────
# Section writers
# ─────────────────────────────────────────────────────────────────

def _write_geometry(data, directory, stem) -> List[Path]:
    geo = data.get("geometry", {})
    cart = geo.get("cartesian_angstrom")
    if not cart:
        return []
    rows = []
    for atom in cart:
        rows.append({
            "index": atom.get("index"),
            "symbol": atom.get("symbol"),
            "x_ang": atom.get("x_ang"),
            "y_ang": atom.get("y_ang"),
            "z_ang": atom.get("z_ang"),
        })
    fn = _write_csv(directory, f"{stem}_geometry.csv", rows,
                    ["index", "symbol", "x_ang", "y_ang", "z_ang"])
    return [fn]


def _write_orbital_energies(data, directory, stem) -> List[Path]:
    oe = data.get("orbital_energies", {})
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
    ml = data.get("mulliken", {})
    files = []
    atoms = ml.get("atomic_charges", [])
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
    lo = data.get("loewdin", {})
    files = []
    atoms = lo.get("atomic_charges", [])
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
    ma = data.get("mayer", {})
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
    hi    = data.get("hirshfeld", {})
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
    mb    = data.get("mbis", {})
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
    ch    = data.get("chelpg", {})
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
    """NAO occupancies - handles RHF, UHF overall, UHF alpha/beta."""
    nbo = data.get("nbo", {})
    files = []

    def _export(naos, suffix):
        if not naos:
            return
        # Detect if spin column present
        has_spin = any("spin" in n for n in naos[:5])
        fieldnames = ["index", "symbol", "atom_no", "angular", "type", "occupancy", "energy_Eh"]
        if has_spin:
            fieldnames.append("spin")
        rows = [{
            "index": n.get("index"),
            "symbol": n.get("symbol"),
            "atom_no": n.get("atom_no"),
            "angular": n.get("angular"),
            "type": n.get("type"),
            "occupancy": n.get("occupancy"),
            "energy_Eh": n.get("energy_Eh", ""),
            "spin": n.get("spin", ""),
        } for n in naos]
        files.append(_write_csv(directory, f"{stem}_nbo_nao{suffix}.csv", rows, fieldnames))

    # Overall (RHF or UHF total density)
    _export(nbo.get("nao_occupancies"), "")
    # UHF spin-specific
    for spin in ("alpha", "beta"):
        spin_data = nbo.get(spin, {})
        _export(spin_data.get("nao_occupancies"), f"_{spin}")
    return files


def _write_nbo_npa(data, directory, stem) -> List[Path]:
    nbo = data.get("nbo", {})
    files = []

    def _export(atoms, suffix):
        if not atoms:
            return
        has_spin = any("spin_density" in a for a in atoms[:5])
        fieldnames = ["index", "symbol", "natural_charge", "core_pop",
                      "valence_pop", "rydberg_pop", "total_pop"]
        if has_spin:
            fieldnames.append("spin_density")
        rows = [{
            "index": a.get("index"), "symbol": a.get("symbol"),
            "natural_charge": a.get("natural_charge"),
            "core_pop": a.get("core_pop"),
            "valence_pop": a.get("valence_pop"),
            "rydberg_pop": a.get("rydberg_pop"),
            "total_pop": a.get("total_pop"),
            "spin_density": a.get("spin_density", ""),
        } for a in atoms]
        files.append(_write_csv(directory, f"{stem}_nbo_npa{suffix}.csv", rows, fieldnames))

    _export(nbo.get("npa_summary"), "")
    for spin in ("alpha", "beta"):
        _export(nbo.get(spin, {}).get("npa_summary"), f"_{spin}")
    return files


def _write_nbo_matrix(data, directory, stem, key, filename_suffix) -> List[Path]:
    """Generic matrix writer for Wiberg/NBI (list of {atom, symbol, values})."""
    nbo = data.get("nbo", {})
    files = []

    def _export(matrix, suffix):
        if not matrix:
            return
        # Flatten matrix rows
        rows = []
        for i, row in enumerate(matrix):
            atom_i = row.get("atom", i + 1)
            sym_i = row.get("symbol", "")
            for j, val in enumerate(row.get("values", [])):
                rows.append({
                    "atom_i": atom_i, "symbol_i": sym_i,
                    "atom_j": j + 1, "bond_index": val,
                })
        if rows:
            files.append(_write_csv(
                directory, f"{stem}_nbo_{filename_suffix}{suffix}.csv", rows,
                ["atom_i", "symbol_i", "atom_j", "bond_index"]
            ))

    _export(nbo.get(key), "")
    for spin in ("alpha", "beta"):
        _export(nbo.get(spin, {}).get(key), f"_{spin}")
    return files


def _write_nbo_lewis(data, directory, stem) -> List[Path]:
    nbo = data.get("nbo", {})
    files = []

    def _export(nbos, suffix):
        if not nbos:
            return
        rows = []
        for n in nbos:
            # Flatten hybrid contributions
            hybs = n.get("hybrids", [])
            hyb_str = "; ".join(
                f"{h.get('atom','')}({h.get('symbol','')}): "
                f"s={h.get('s_pct',0):.1f}% p={h.get('p_pct',0):.1f}% "
                f"d={h.get('d_pct',0):.1f}%"
                for h in hybs
            ) if hybs else ""
            rows.append({
                "nbo_index": n.get("index"),
                "type": n.get("type"),
                "occupancy": n.get("occupancy"),
                "energy_Eh": n.get("energy_Eh", ""),
                "label": n.get("label", ""),
                "s_pct": hybs[0].get("s_pct", "") if hybs else "",
                "p_pct": hybs[0].get("p_pct", "") if hybs else "",
                "d_pct": hybs[0].get("d_pct", "") if hybs else "",
                "hybridization_detail": hyb_str,
            })
        files.append(_write_csv(
            directory, f"{stem}_nbo_lewis{suffix}.csv", rows,
            ["nbo_index", "type", "occupancy", "energy_Eh", "label",
             "s_pct", "p_pct", "d_pct", "hybridization_detail"]
        ))

    _export(nbo.get("nbo_lewis"), "")
    for spin in ("alpha", "beta"):
        _export(nbo.get(spin, {}).get("nbo_lewis"), f"_{spin}")
    return files


def _write_nbo_e2(data, directory, stem) -> List[Path]:
    nbo = data.get("nbo", {})
    files = []

    def _export(entries, suffix):
        if not entries:
            return
        files.append(_write_csv(
            directory, f"{stem}_nbo_e2{suffix}.csv", entries,
            ["donor", "acceptor", "E2_kcal_mol", "E_gap_au", "Fock_au"]
        ))

    _export(nbo.get("e2_perturbation"), "")
    for spin in ("alpha", "beta"):
        _export(nbo.get(spin, {}).get("e2_perturbation"), f"_{spin}")
    return files


def _write_nbo_nlmo_hyb(data, directory, stem) -> List[Path]:
    nbo = data.get("nbo", {})
    files = []

    def _export(nlmos, suffix):
        if not nlmos:
            return
        rows = []
        for nl in nlmos:
            # Flatten per-atom contributions
            for contrib in nl.get("contributions", []):
                rows.append({
                    "nlmo_index": nl.get("index"),
                    "occupancy": nl.get("occupancy"),
                    "parent_pct": nl.get("parent_pct", ""),
                    "parent_nbo": nl.get("parent_nbo", ""),
                    "atom": contrib.get("atom", ""),
                    "symbol": contrib.get("symbol", ""),
                    "pct": contrib.get("pct", ""),
                    "s_pct": contrib.get("s_pct", ""),
                    "p_pct": contrib.get("p_pct", ""),
                    "d_pct": contrib.get("d_pct", ""),
                })
        if rows:
            files.append(_write_csv(
                directory, f"{stem}_nbo_nlmo_hyb{suffix}.csv", rows,
                ["nlmo_index", "occupancy", "parent_pct", "parent_nbo",
                 "atom", "symbol", "pct", "s_pct", "p_pct", "d_pct"]
            ))

    _export(nbo.get("nlmo_hybridization"), "")
    for spin in ("alpha", "beta"):
        _export(nbo.get(spin, {}).get("nlmo_hybridization"), f"_{spin}")
    return files


def _write_nbo_nlmo_bo(data, directory, stem) -> List[Path]:
    nbo = data.get("nbo", {})
    files = []

    def _export(bonds, suffix):
        if not bonds:
            return
        files.append(_write_csv(
            directory, f"{stem}_nbo_nlmo_bo{suffix}.csv", bonds,
            ["atom_i", "atom_j", "nlmo_index", "bond_order", "hybrid_overlap"]
        ))

    _export(nbo.get("nlmo_bond_orders"), "")
    for spin in ("alpha", "beta"):
        _export(nbo.get(spin, {}).get("nlmo_bond_orders"), f"_{spin}")
    return files


def _write_nbo_nlmo_steric(data, directory, stem) -> List[Path]:
    nbo = data.get("nbo", {})
    files = []

    def _export(steric, suffix):
        if not steric:
            return
        # Unit contributions
        unit_rows = steric.get("unit_contributions", [])
        if unit_rows:
            files.append(_write_csv(
                directory, f"{stem}_nbo_nlmo_steric{suffix}.csv", unit_rows,
                ["nlmo_index", "label", "dE_kcal_mol"]
            ))
        # Pairwise
        pair_rows = steric.get("pairwise", [])
        if pair_rows:
            files.append(_write_csv(
                directory, f"{stem}_nbo_nlmo_steric_pairwise{suffix}.csv", pair_rows,
                ["nlmo_i", "nlmo_j", "S_ij", "dE_ij_kcal_mol"]
            ))

    _export(nbo.get("nlmo_steric"), "")
    for spin in ("alpha", "beta"):
        _export(nbo.get(spin, {}).get("nlmo_steric"), f"_{spin}")
    return files


def _write_dipole(data, directory, stem) -> List[Path]:
    dip = data.get("dipole")
    if not dip:
        return []
    row = {
        "ex": dip.get("electronic_contribution_au", {}).get("x", ""),
        "ey": dip.get("electronic_contribution_au", {}).get("y", ""),
        "ez": dip.get("electronic_contribution_au", {}).get("z", ""),
        "nx": dip.get("nuclear_contribution_au", {}).get("x", ""),
        "ny": dip.get("nuclear_contribution_au", {}).get("y", ""),
        "nz": dip.get("nuclear_contribution_au", {}).get("z", ""),
        "total_x_au": dip.get("total_dipole_au", {}).get("x", ""),
        "total_y_au": dip.get("total_dipole_au", {}).get("y", ""),
        "total_z_au": dip.get("total_dipole_au", {}).get("z", ""),
        "magnitude_au": dip.get("magnitude_au", ""),
        "magnitude_Debye": dip.get("magnitude_Debye", ""),
    }
    fn = _write_csv(directory, f"{stem}_dipole.csv", [row],
                    list(row.keys()))
    return [fn]


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
        _write_geometry,
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
        _write_nbo_lewis,
        _write_nbo_e2,
        _write_nbo_nlmo_hyb,
        _write_nbo_nlmo_bo,
        _write_nbo_nlmo_steric,
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
