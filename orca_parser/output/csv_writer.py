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


def _get_symmetry_data(data: Dict[str, Any]) -> Dict[str, Any]:
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

    return sym


def _is_deltascf(data: Dict[str, Any]) -> bool:
    """Whether this parsed job is a DeltaSCF excited-state calculation."""
    return str(data.get("metadata", {}).get("calculation_type", "")).lower() == "deltascf"


def _electronic_state_label(data: Dict[str, Any]) -> str:
    """Short electronic-state label for metadata exports."""
    return "DeltaSCF excited-state" if _is_deltascf(data) else "Ground-state"


def _is_surface_scan(data: Dict[str, Any]) -> bool:
    """Whether this parsed job is a relaxed surface scan."""
    return bool(data.get("surface_scan"))


def _bool_to_label(value: Any) -> str:
    """Render bool-like values consistently for CSV exports."""
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return ""


def _format_simple_vector(values: Optional[List[Any]]) -> str:
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


def _format_deltascf_target(deltascf: Dict[str, Any]) -> str:
    """Build a compact target summary such as 'ALPHACONF 0,1'."""
    parts: List[str] = []
    for key in ("alphaconf", "betaconf"):
        values = deltascf.get(key)
        if values:
            parts.append(f"{key.upper()} {_format_simple_vector(values)}")
    for key in ("ionizealpha", "ionizebeta"):
        value = deltascf.get(key)
        if value is not None:
            parts.append(f"{key.upper()} {value}")
    return " | ".join(parts)


# ─────────────────────────────────────────────────────────────────
# Section writers
# ─────────────────────────────────────────────────────────────────

def _write_geometry(data, directory, stem) -> List[Path]:
    geo = data.get("geometry", {})
    files = []

    cart = geo.get("cartesian_angstrom")
    if cart:
        rows = []
        for atom in cart:
            rows.append({
                "index": atom.get("index"),
                "symbol": atom.get("symbol"),
                "x_ang": atom.get("x_ang"),
                "y_ang": atom.get("y_ang"),
                "z_ang": atom.get("z_ang"),
            })
        files.append(_write_csv(
            directory, f"{stem}_geometry.csv", rows,
            ["index", "symbol", "x_ang", "y_ang", "z_ang"],
        ))

    sym_cart = geo.get("symmetry_cartesian_angstrom")
    if sym_cart:
        point_group = geo.get("symmetry_perfected_point_group", "")
        sym_rows = []
        for atom in sym_cart:
            sym_rows.append({
                "point_group": point_group,
                "index": atom.get("index"),
                "symbol": atom.get("symbol"),
                "x_ang": atom.get("x_ang"),
                "y_ang": atom.get("y_ang"),
                "z_ang": atom.get("z_ang"),
            })
        files.append(_write_csv(
            directory, f"{stem}_geometry_symmetry.csv", sym_rows,
            ["point_group", "index", "symbol", "x_ang", "y_ang", "z_ang"],
        ))

    return files


def _write_metadata(data: Dict[str, Any], directory: Path, stem: str) -> List[Path]:
    """Write a one-row metadata summary for downstream filtering/grouping."""
    meta = data.get("metadata", {})
    if not meta:
        return []

    sym = _get_symmetry_data(data)
    deltascf = meta.get("deltascf") or {}
    geom = data.get("geometry", {})
    surface_scan = data.get("surface_scan") or {}
    row = {
        "job_name": meta.get("job_name", ""),
        "source_file": data.get("source_file", ""),
        "program_version": meta.get("program_version", ""),
        "run_date": meta.get("run_date", ""),
        "host": meta.get("host", ""),
        "calculation_type": meta.get("calculation_type", ""),
        "electronic_state": _electronic_state_label(data),
        "hf_type": meta.get("hf_type", ""),
        "functional": meta.get("functional", ""),
        "basis_set": meta.get("basis_set", ""),
        "charge": meta.get("charge", ""),
        "multiplicity": meta.get("multiplicity", ""),
        "point_group": sym.get("point_group", ""),
        "reduced_point_group": sym.get("reduced_point_group", ""),
        "orbital_irrep_group": sym.get("orbital_irrep_group", ""),
        "use_sym": _bool_to_label(sym.get("use_sym")),
        "n_irreps": sym.get("n_irreps", ""),
        "initial_guess_irrep": sym.get("initial_guess_irrep", ""),
        "symmetry_perfected_point_group": geom.get("symmetry_perfected_point_group", ""),
        "symmetry_perfected_atoms": len(geom.get("symmetry_cartesian_angstrom") or []),
        "is_surface_scan": _bool_to_label(_is_surface_scan(data)),
        "scan_mode": surface_scan.get("mode", ""),
        "scan_parameters": surface_scan.get("n_parameters", ""),
        "scan_steps": surface_scan.get("n_constrained_optimizations", ""),
        "deltascf_target": _format_deltascf_target(deltascf),
        "deltascf_metric": deltascf.get("aufbau_metric", ""),
        "keep_initial_reference": _bool_to_label(deltascf.get("keep_initial_reference")),
        "input_keywords": " ".join(meta.get("input_keywords") or []),
    }
    fn = _write_csv(directory, f"{stem}_metadata.csv", [row], list(row.keys()))
    return [fn]


def _write_symmetry(data: Dict[str, Any], directory: Path, stem: str) -> List[Path]:
    """Write symmetry summary plus irrep-resolved details when available."""
    sym = _get_symmetry_data(data)
    if not sym:
        return []

    files = []
    geom = data.get("geometry", {})
    oe = data.get("orbital_energies", {})

    summary_row = {
        "use_sym": _bool_to_label(sym.get("use_sym")),
        "auto_detected_point_group": sym.get("auto_detected_point_group", ""),
        "point_group": sym.get("point_group", ""),
        "reduced_point_group": sym.get("reduced_point_group", ""),
        "orbital_irrep_group": sym.get("orbital_irrep_group", ""),
        "petite_list_algorithm": _bool_to_label(sym.get("petite_list_algorithm")),
        "n_irreps": sym.get("n_irreps", ""),
        "initial_guess_irrep": sym.get("initial_guess_irrep", ""),
        "setup_rms_distance_au": sym.get("setup_rms_distance_au", ""),
        "setup_max_distance_au": sym.get("setup_max_distance_au", ""),
        "setup_threshold_au": sym.get("setup_threshold_au", ""),
        "setup_time_s": sym.get("setup_time_s", ""),
        "symmetry_perfected_point_group": geom.get("symmetry_perfected_point_group", ""),
        "symmetry_perfected_atoms": len(geom.get("symmetry_cartesian_angstrom") or []),
    }
    files.append(_write_csv(
        directory, f"{stem}_symmetry.csv", [summary_row], list(summary_row.keys())
    ))

    irreps = sym.get("irreps") or []
    alpha_occ = oe.get("alpha_occupied_per_irrep") or {}
    beta_occ = oe.get("beta_occupied_per_irrep") or {}
    total_occ = oe.get("occupied_per_irrep") or {}
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
        files.append(_write_csv(
            directory, f"{stem}_symmetry_irreps.csv", rows,
            ["irrep", "n_basis_functions", "offset", "occupied_alpha", "occupied_beta", "occupied_total"],
        ))

    return files


def _write_deltascf(data: Dict[str, Any], directory: Path, stem: str) -> List[Path]:
    """Write DeltaSCF excited-state target metadata when present."""
    meta = data.get("metadata", {})
    deltascf = meta.get("deltascf") or {}
    if not _is_deltascf(data):
        return []

    files = []
    summary_row = {
        "electronic_state": _electronic_state_label(data),
        "target_configuration": _format_deltascf_target(deltascf),
        "alphaconf": _format_simple_vector(deltascf.get("alphaconf")),
        "betaconf": _format_simple_vector(deltascf.get("betaconf")),
        "ionizealpha": deltascf.get("ionizealpha", ""),
        "ionizebeta": deltascf.get("ionizebeta", ""),
        "aufbau_metric": deltascf.get("aufbau_metric", ""),
        "keep_initial_reference": _bool_to_label(deltascf.get("keep_initial_reference")),
    }
    files.append(_write_csv(
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
        files.append(_write_csv(
            directory, f"{stem}_deltascf_occupations.csv", target_rows,
            ["spin", "slot", "occupation"],
        ))

    return files


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


def _write_epr(data, directory, stem) -> List[Path]:
    epr = data.get("epr")
    if not epr:
        return []
    files = []

    # ── g-tensor atom contributions ─────────────────────────────────
    g = epr.get("g_tensor", {})
    atoms = g.get("atom_analysis", {}).get("atom_contributions", [])
    if atoms:
        rows = [{
            "element": a.get("element"),
            "atom_index": a.get("atom_index"),
            "g_1": a.get("values", [None, None, None])[0],
            "g_2": a.get("values", [None, None, None])[1],
            "g_3": a.get("values", [None, None, None])[2],
            "iso": a.get("iso"),
        } for a in atoms]
        files.append(_write_csv(
            directory, f"{stem}_epr_g_atoms.csv", rows,
            ["element", "atom_index", "g_1", "g_2", "g_3", "iso"],
        ))

    # ── Hyperfine: per-nucleus principal components ─────────────────
    hf = epr.get("hyperfine", {})
    nuclei = hf.get("nuclei", [])
    if nuclei:
        rows = []
        for nuc in nuclei:
            base = {
                "nucleus_index": nuc.get("nucleus_index"),
                "element": nuc.get("element"),
                "isotope": nuc.get("isotope"),
            }
            for key, pc in nuc.get("principal_components", {}).items():
                vals = pc.get("values_MHz", [None, None, None])
                rows.append({
                    **base,
                    "component": key,
                    "A_1_MHz": vals[0],
                    "A_2_MHz": vals[1],
                    "A_3_MHz": vals[2],
                    "A_iso_MHz": pc.get("A_iso_MHz"),
                    "A_PC_MHz": pc.get("A_PC_MHz"),
                })
        if rows:
            files.append(_write_csv(
                directory, f"{stem}_epr_hyperfine.csv", rows,
                ["nucleus_index", "element", "isotope", "component",
                 "A_1_MHz", "A_2_MHz", "A_3_MHz", "A_iso_MHz", "A_PC_MHz"],
            ))

    # ── Quadrupole coupling ─────────────────────────────────────────
    quad_rows = []
    for nuc in nuclei:
        qc = nuc.get("quadrupole_coupling")
        if qc:
            quad_rows.append({
                "nucleus_index": nuc.get("nucleus_index"),
                "element": nuc.get("element"),
                "isotope": nuc.get("isotope"),
                "e2qQ_MHz": qc.get("e2qQ_MHz"),
                "eta": qc.get("eta"),
            })
    if quad_rows:
        files.append(_write_csv(
            directory, f"{stem}_epr_quadrupole.csv", quad_rows,
            ["nucleus_index", "element", "isotope", "e2qQ_MHz", "eta"],
        ))

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


def _write_tddft(data, directory, stem) -> List[Path]:
    tddft = data.get("tddft")
    if not tddft:
        return []

    files: List[Path] = []

    states = tddft.get("excited_states", [])
    if states:
        state_rows = []
        transition_rows = []

        for state in states:
            transitions = state.get("transitions", [])
            dominant = max(
                transitions,
                key=lambda item: item.get("weight", 0.0),
                default={},
            )
            state_rows.append({
                "block_index": state.get("block_index"),
                "method": state.get("method"),
                "manifold": state.get("manifold", ""),
                "order_in_block": state.get("order_in_block"),
                "state": state.get("state"),
                "energy_au": state.get("energy_au"),
                "energy_eV": state.get("energy_eV"),
                "energy_cm1": state.get("energy_cm1"),
                "wavelength_nm": state.get("wavelength_nm", ""),
                "s_squared": state.get("s_squared", ""),
                "multiplicity": state.get("multiplicity", ""),
                "dominant_from_orbital": dominant.get("from_orbital", ""),
                "dominant_to_orbital": dominant.get("to_orbital", ""),
                "dominant_weight": dominant.get("weight", ""),
                "dominant_coefficient": dominant.get("coefficient", ""),
            })

            for transition in transitions:
                transition_rows.append({
                    "block_index": state.get("block_index"),
                    "method": state.get("method"),
                    "manifold": state.get("manifold", ""),
                    "state": state.get("state"),
                    "from_orbital": transition.get("from_orbital"),
                    "from_index": transition.get("from_index", ""),
                    "from_spin": transition.get("from_spin", ""),
                    "to_orbital": transition.get("to_orbital"),
                    "to_index": transition.get("to_index", ""),
                    "to_spin": transition.get("to_spin", ""),
                    "weight": transition.get("weight"),
                    "coefficient": transition.get("coefficient"),
                })

        files.append(_write_csv(
            directory, f"{stem}_tddft_states.csv", state_rows,
            [
                "block_index", "method", "manifold", "order_in_block",
                "state", "energy_au", "energy_eV", "energy_cm1",
                "wavelength_nm", "s_squared", "multiplicity",
                "dominant_from_orbital", "dominant_to_orbital",
                "dominant_weight", "dominant_coefficient",
            ],
        ))

        if transition_rows:
            files.append(_write_csv(
                directory, f"{stem}_tddft_transitions.csv", transition_rows,
                [
                    "block_index", "method", "manifold", "state",
                    "from_orbital", "from_index", "from_spin",
                    "to_orbital", "to_index", "to_spin",
                    "weight", "coefficient",
                ],
            ))

    nto_states = tddft.get("nto_states", [])
    if nto_states:
        nto_rows = []
        for state in nto_states:
            for pair in state.get("pairs", []):
                nto_rows.append({
                    "state": state.get("state"),
                    "output_file": state.get("output_file", ""),
                    "print_threshold": state.get("print_threshold", ""),
                    "energy_au": state.get("energy_au", ""),
                    "energy_eV": state.get("energy_eV", ""),
                    "energy_cm1": state.get("energy_cm1", ""),
                    "wavelength_nm": state.get("wavelength_nm", ""),
                    "from_orbital": pair.get("from_orbital"),
                    "from_index": pair.get("from_index", ""),
                    "from_spin": pair.get("from_spin", ""),
                    "to_orbital": pair.get("to_orbital"),
                    "to_index": pair.get("to_index", ""),
                    "to_spin": pair.get("to_spin", ""),
                    "occupation": pair.get("occupation"),
                })
        if nto_rows:
            files.append(_write_csv(
                directory, f"{stem}_tddft_nto.csv", nto_rows,
                [
                    "state", "output_file", "print_threshold",
                    "energy_au", "energy_eV", "energy_cm1", "wavelength_nm",
                    "from_orbital", "from_index", "from_spin",
                    "to_orbital", "to_index", "to_spin", "occupation",
                ],
            ))

    spectra = tddft.get("spectra", {})
    spectrum_suffixes = {
        "absorption_electric_dipole": "tddft_absorption_electric.csv",
        "absorption_velocity_dipole": "tddft_absorption_velocity.csv",
        "cd_electric_dipole": "tddft_cd_electric.csv",
        "cd_velocity_dipole": "tddft_cd_velocity.csv",
    }
    base_fields = [
        "from_state_label", "from_root", "from_state_suffix",
        "to_state_label", "to_root", "to_state_suffix",
        "energy_eV", "energy_cm1", "wavelength_nm",
    ]
    for kind, filename in spectrum_suffixes.items():
        table = spectra.get(kind)
        if not table:
            continue
        transitions = table.get("transitions", [])
        if not transitions:
            continue

        center = table.get("center_of_mass", {})
        rows = []
        for transition in transitions:
            row = dict(transition)
            row["center_of_mass_x"] = center.get("x", "")
            row["center_of_mass_y"] = center.get("y", "")
            row["center_of_mass_z"] = center.get("z", "")
            rows.append(row)

        extra_fields = [
            field
            for field in rows[0].keys()
            if field not in base_fields
            and field not in {"center_of_mass_x", "center_of_mass_y", "center_of_mass_z"}
        ]
        files.append(_write_csv(
            directory, f"{stem}_{filename}", rows,
            base_fields + extra_fields + [
                "center_of_mass_x", "center_of_mass_y", "center_of_mass_z",
            ],
        ))

    total_energy_blocks = tddft.get("total_energy_blocks", [])
    if total_energy_blocks:
        files.append(_write_csv(
            directory, f"{stem}_tddft_total_energy.csv", total_energy_blocks,
            [
                "block_index", "excitation_method", "root",
                "scf_energy_Eh", "delta_energy_Eh", "total_energy_Eh",
                "maximum_memory_MB",
            ],
        ))

    return files


def _write_solvation(data, directory, stem) -> List[Path]:
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
    files.append(_write_csv(
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
        files.append(_write_csv(
            directory, f"{stem}_solvation_history.csv", history_rows,
            [
                "source", "line_index", "model", "solvent",
                "epsilon", "surface_type", "reference_state",
            ],
        ))

    return files


def _write_geom_opt(data: Dict[str, Any], directory: Path, stem: str) -> List[Path]:
    gopt = data.get("geom_opt")
    if not gopt:
        return []
    cycles = gopt.get("cycles", [])
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
    rows: List[Dict] = []
    for c in cycles:
        row: Dict[str, Any] = {
            "cycle": c.get("cycle"),
            "energy_Eh": c.get("energy_Eh"),
            "energy_change_Eh": c.get("energy_change_Eh", ""),
            "trust_radius_bohr": c.get("trust_radius_bohr", ""),
            "rmsd_to_initial_ang": c.get("rmsd_to_initial_ang", ""),
            "rmsd_to_previous_ang": c.get("rmsd_to_previous_ang", ""),
            "orca_converged": c.get("orca_converged", False),
        }
        conv = c.get("convergence", {})
        for key in ("energy_change", "rms_gradient", "max_gradient", "rms_step", "max_step"):
            entry = conv.get(key, {})
            row[f"{key}_val"] = entry.get("value", "")
            row[f"{key}_tol"] = entry.get("tolerance", "")
            row[f"{key}_conv"] = entry.get("converged", "")
        rows.append(row)

    fn = _write_csv(directory, f"{stem}_geom_opt.csv", rows, fields)
    return [fn]


def _write_surface_scan(data: Dict[str, Any], directory: Path, stem: str) -> List[Path]:
    scan = data.get("surface_scan")
    if not scan:
        return []

    files: List[Path] = []
    parameters = scan.get("parameters") or []
    steps = scan.get("steps") or []

    if parameters:
        rows: List[Dict[str, Any]] = []
        for idx, parameter in enumerate(parameters, start=1):
            row: Dict[str, Any] = {
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
                "values": _format_simple_vector(parameter.get("values")),
            }
            rows.append(row)
        files.append(_write_csv(
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
        files.append(_write_csv(
            directory, f"{stem}_surface_scan.csv", rows, fields,
        ))

    sidecars = scan.get("sidecar_files") or {}
    if sidecars:
        row = {
            "mode": scan.get("mode", ""),
            "n_parameters": scan.get("n_parameters", ""),
            "n_constrained_optimizations": scan.get("n_constrained_optimizations", ""),
            "actual_surface_dat": sidecars.get("actual_surface_dat", ""),
            "scf_surface_dat": sidecars.get("scf_surface_dat", ""),
            "allxyz": sidecars.get("allxyz", ""),
            "xyzall": sidecars.get("xyzall", ""),
            "trajectory_xyz": sidecars.get("trajectory_xyz", ""),
            "allxyz_frame_count": sidecars.get("allxyz_frame_count", ""),
        }
        files.append(_write_csv(
            directory, f"{stem}_surface_scan_summary.csv", [row], list(row.keys()),
        ))

    return files


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
