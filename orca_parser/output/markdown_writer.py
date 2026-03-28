"""
Markdown output writer for ORCA parser results.

Designed for AI-assisted paper writing: maximum information density,
minimum structural noise, publication-ready table formatting.

Single molecule  → to_markdown()
Multi-molecule   → compare()
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Public entry points
# ─────────────────────────────────────────────────────────────────────────────

def write_markdown(data: Dict[str, Any], path: Path) -> Path:
    """Write a single-molecule markdown report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render_molecule(data), encoding="utf-8")
    return path


def write_comparison(datasets: List[Dict[str, Any]], path: Path) -> Path:
    """Write a multi-molecule comparison markdown document."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render_comparison(datasets), encoding="utf-8")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Single-molecule renderer
# ─────────────────────────────────────────────────────────────────────────────

def _render_molecule(
    data: Dict[str, Any],
    heading_level: int = 1,
    display_label: Optional[str] = None,
    source_display: Optional[str] = None,
) -> str:
    """Full markdown report for one molecule."""
    H = "#" * heading_level
    H2 = "#" * (heading_level + 1)
    H3 = "#" * (heading_level + 2)

    blocks: List[str] = []

    meta = data.get("metadata", {})
    ctx  = data.get("context",  {})
    scf  = data.get("scf",      {})
    src_path = Path(data.get("source_file", "unknown"))
    src  = source_display or src_path.name

    # ── Title ──────────────────────────────────────────────────────────────
    job   = display_label or meta.get("job_name", src_path.stem)
    func  = meta.get("functional", "?")
    basis = meta.get("basis_set",  "?")
    hftyp = ctx.get("hf_type", "RHF")
    charge = meta.get("charge", 0)
    mult   = meta.get("multiplicity", 1)
    sym    = f"  symmetry={_irrep_group(data)}" if ctx.get("has_symmetry") else ""
    blocks.append(
        f"{H} {job}\n"
        f"`{hftyp} {func}/{basis}` | charge={charge} mult={mult}{sym}  \n"
        f"source: `{src}`"
    )

    # ── SCF summary ────────────────────────────────────────────────────────
    scf_lines = []
    E = scf.get("final_single_point_energy_Eh")
    if E is not None:
        scf_lines.append(f"**Energy:** {E:.10f} Eh")
    if "dispersion_correction_Eh" in scf:
        scf_lines.append(f"**Dispersion (D):** {scf['dispersion_correction_Eh']:.8f} Eh")
    if "s_squared" in scf:
        s2     = scf["s_squared"]
        s2_id  = scf.get("s_squared_ideal", "?")
        contam = abs(s2 - float(s2_id)) if isinstance(s2_id, (int, float)) else None
        warn   = "  ⚠️ *contamination > 0.01*" if contam and contam > 0.01 else ""
        scf_lines.append(f"**⟨S²⟩:** {s2:.6f} (ideal {s2_id}){warn}")

    dip = data.get("dipole", {})
    if dip.get("magnitude_Debye") is not None:
        vec = dip.get("total_debye", {})
        xyz = ""
        if vec:
            xyz = f"  ({vec.get('x',0):.4f}, {vec.get('y',0):.4f}, {vec.get('z',0):.4f}) D"
        scf_lines.append(f"**Dipole:** {dip['magnitude_Debye']:.4f} D{xyz}")

    if scf_lines:
        blocks.append("\n".join(scf_lines))

    # -- Basis-set summary -----------------------------------------------
    basis_set = data.get("basis_set")
    if basis_set:
        basis_section = _render_basis_set_section(basis_set)
        if basis_section:
            blocks.append(f"{H2} Basis Set\n{basis_section}")

    # -- Geometry optimization summary -----------------------------------
    geom_opt = data.get("geom_opt")
    if geom_opt:
        geom_opt_section = _render_geom_opt_section(geom_opt)
        if geom_opt_section:
            blocks.append(f"{H2} Geometry Optimization\n{geom_opt_section}")

    # -- Solvation --------------------------------------------------------
    solvation = data.get("solvation")
    if solvation:
        solvation_section = _render_solvation_section(solvation)
        if solvation_section:
            blocks.append(f"{H2} Solvation\n{solvation_section}")

    # -- TDDFT / CIS excited states ---------------------------------------
    tddft = data.get("tddft")
    if tddft:
        tddft_section = _render_tddft_section(tddft)
        if tddft_section:
            blocks.append(f"{H2} TDDFT Excited States\n{tddft_section}")

    # ── Orbital energies ───────────────────────────────────────────────────
    oe = data.get("orbital_energies")
    if oe:
        blocks.append(f"{H2} Frontier Orbital Energies")
        if not ctx.get("is_uhf"):
            homo  = oe.get("HOMO_energy_eV")
            lumo  = oe.get("LUMO_energy_eV")
            gap   = oe.get("HOMO_LUMO_gap_eV")
            hirr  = oe.get("HOMO_irrep", "")
            lirr  = oe.get("LUMO_irrep", "")
            rows = [("", "eV", "irrep"),
                    ("HOMO", f"{homo:.4f}" if homo else "—", hirr),
                    ("LUMO", f"{lumo:.4f}" if lumo else "—", lirr),
                    ("Gap",  f"{gap:.4f}"  if gap  else "—", "")]
            if not hirr:  # drop irrep column if no symmetry
                rows = [(r[0], r[1]) for r in rows]
            blocks.append(_table(rows))
        else:
            ah = oe.get("alpha_HOMO_energy_eV")
            al = oe.get("alpha_LUMO_energy_eV")
            bh = oe.get("beta_HOMO_energy_eV")
            bl = oe.get("beta_LUMO_energy_eV")
            ag = oe.get("alpha_HOMO_LUMO_gap_eV")
            bg = oe.get("beta_HOMO_LUMO_gap_eV")
            ahi = oe.get("alpha_HOMO_irrep", "")
            ali = oe.get("alpha_LUMO_irrep", "")
            bhi = oe.get("beta_HOMO_irrep",  "")
            bli = oe.get("beta_LUMO_irrep",  "")
            has_irr = bool(ahi or ali)
            if has_irr:
                rows = [("", "α eV", "α irrep", "β eV", "β irrep"),
                        ("HOMO", _f(ah), ahi, _f(bh), bhi),
                        ("LUMO", _f(al), ali, _f(bl), bli),
                        ("Gap",  _f(ag), "",  _f(bg), "")]
            else:
                rows = [("",     "α eV", "β eV"),
                        ("HOMO", _f(ah), _f(bh)),
                        ("LUMO", _f(al), _f(bl)),
                        ("Gap",  _f(ag), _f(bg))]
            blocks.append(_table(rows))

        # Occupied per irrep
        occ = oe.get("occupied_per_irrep") or oe.get("alpha_occupied_per_irrep")
        if occ:
            b_occ = oe.get("beta_occupied_per_irrep")
            if b_occ:
                irrs = sorted(set(list(occ) + list(b_occ)))
                rows = [("Irrep",) + tuple(irrs),
                        ("α occ",) + tuple(occ.get(i, 0) for i in irrs),
                        ("β occ",) + tuple(b_occ.get(i, 0) for i in irrs)]
            else:
                irrs = sorted(occ)
                rows = [("Irrep",) + tuple(irrs),
                        ("occ",)   + tuple(occ.get(i, 0) for i in irrs)]
            blocks.append(_table(rows))

    # ── QRO ───────────────────────────────────────────────────────────────
    qro = data.get("qro")
    if qro:
        blocks.append(f"{H2} Quasi-Restricted Orbitals")
        blocks.append(
            f"DOMO = {qro.get('n_domo')}  |  "
            f"SOMO = {qro.get('n_somo')}  |  "
            f"VMO = {qro.get('n_vmo')}"
        )
        somos = qro.get("somo_details", [])
        if somos:
            has_irr = any("irrep" in s for s in somos)
            if has_irr:
                header = ("MO", "irrep", "ε (Eh)", "ε (eV)", "α (eV)", "β (eV)")
                rows   = [header] + [
                    (str(s["index"]),
                     s.get("irrep", ""),
                     f"{s['energy_Eh']:.6f}",
                     f"{s['energy_eV']:.4f}",
                     f"{s['alpha_energy_eV']:.4f}",
                     f"{s['beta_energy_eV']:.4f}")
                    for s in somos
                ]
            else:
                header = ("MO", "ε (Eh)", "ε (eV)", "α (eV)", "β (eV)")
                rows   = [header] + [
                    (str(s["index"]),
                     f"{s['energy_Eh']:.6f}",
                     f"{s['energy_eV']:.4f}",
                     f"{s['alpha_energy_eV']:.4f}",
                     f"{s['beta_energy_eV']:.4f}")
                    for s in somos
                ]
            blocks.append(_table(rows))


    # ── Atomic charges — all schemes, aligned by atom index ──────────────
    epr = data.get("epr")
    if epr:
        epr_section = _render_epr_section(epr, heading_level=heading_level + 2)
        if epr_section:
            blocks.append(f"{H2} EPR / Magnetic Properties\n{epr_section}")

    charge_data = _collect_charges(data)
    if charge_data:
        blocks.append(f"{H2} Atomic Charges")
        atom_syms = ctx.get("atom_symbols", [])  # 0-based list: atom_syms[0] = first atom
        schemes   = list(charge_data.keys())
        # Collect all unique atom indices across all schemes, sorted
        all_indices = sorted(set(idx for d in charge_data.values() for idx in d))
        rows = [("Atom",) + tuple(schemes)]
        for idx in all_indices:
            # atom_syms is 0-based; ORCA atom indices may be 0- or 1-based depending on section
            # Use modular mapping: if index > len(atom_syms)-1, cap at last
            sym_i = min(idx, len(atom_syms) - 1) if atom_syms else idx
            sym   = atom_syms[sym_i] if atom_syms else "?"
            lbl   = f"{sym}{idx + 1}" if idx < len(atom_syms) else f"?{idx}"
            # Special case: ORCA Mulliken/Loewdin use 1-based index so idx=1 → atom_syms[0]
            # The label should match: sym at position (idx-1) for 1-based, idx for 0-based
            # Resolve by checking what symbol each scheme says this index is
            # Use the first scheme that has this index to determine symbol
            for s in schemes:
                if idx in charge_data[s]:
                    break
            # For label: use ORCA index to position in atom_syms list
            # 0-based schemes (hirshfeld/mbis): idx 0 = atom_syms[0]
            # 1-based schemes (mulliken/loewdin): idx 1 = atom_syms[0]
            # Determine offset by checking if index 0 exists in any scheme
            rows.append((lbl,) + tuple(
                f"{charge_data[s][idx]:.4f}" if idx in charge_data[s] else "—"
                for s in schemes
            ))
        blocks.append(_table(rows))

    # ── Spin populations — all schemes, aligned by atom index ─────────────
    if ctx.get("is_uhf"):
        spin_data = _collect_spin(data)
        if spin_data:
            blocks.append(f"{H2} Atomic Spin Populations")
            atom_syms = ctx.get("atom_symbols", [])
            schemes   = list(spin_data.keys())
            all_indices = sorted(set(idx for d in spin_data.values() for idx in d))
            rows = [("Atom",) + tuple(schemes)]
            for idx in all_indices:
                sym_i = min(idx, len(atom_syms) - 1) if atom_syms else idx
                sym   = atom_syms[sym_i] if atom_syms else "?"
                lbl   = f"{sym}{idx + 1}" if idx < len(atom_syms) else f"?{idx}"
                rows.append((lbl,) + tuple(
                    f"{spin_data[s][idx]:.4f}" if idx in spin_data[s] else "—"
                    for s in schemes
                ))
            blocks.append(_table(rows))
            # Spin totals
            totals = {s: sum(spin_data[s].values()) for s in schemes}
            blocks.append("**Spin totals:** " +
                          "  ".join(f"{s} = {totals[s]:.4f}" for s in schemes))

    # ── Mayer bond orders — full list ─────────────────────────────────────
    mayer   = data.get("mayer", {})
    bo_list = mayer.get("bond_orders", [])
    if bo_list:
        blocks.append(f"{H2} Mayer Bond Orders ({len(bo_list)} bonds)")
        rows = [("Bond", "Order")]
        for b in bo_list:
            si = b.get("symbol_i", "?") + str(b.get("atom_i", 0) + 1)
            sj = b.get("symbol_j", "?") + str(b.get("atom_j", 0) + 1)
            rows.append((f"{si}\u2013{sj}", f"{b.get('bond_order', 0):.4f}"))
        blocks.append(_table(rows))

    # ── NBO ── full population analysis ───────────────────────────────────
    nbo    = data.get("nbo")
    is_uhf = ctx.get("is_uhf", False)
    if nbo:
        blocks.append(f"{H2} NBO Analysis")

        # ── NPA summary (overall for UHF, main for RHF) ──────────────────
        # UHF: overall_npa_summary has both charge and spin_density
        # RHF: npa_summary
        npa = nbo.get("overall_npa_summary") if is_uhf else nbo.get("npa_summary", [])
        if npa:
            has_spin = any("spin_density" in a for a in npa)
            if has_spin:
                rows = [("Atom", "NPA charge", "Core", "Valence", "Rydberg", "Total", "Spin ρ")]
            else:
                rows = [("Atom", "NPA charge", "Core", "Valence", "Rydberg", "Total")]
            for a in npa:
                lbl = f"{a.get('symbol','?')}{a.get('index','')}"
                row = [lbl,
                       f"{a.get('natural_charge', 0):.5f}",
                       f"{a.get('core_pop', 0):.5f}",
                       f"{a.get('valence_pop', 0):.5f}",
                       f"{a.get('rydberg_pop', 0):.5f}",
                       f"{a.get('total_pop', 0):.5f}"]
                if has_spin:
                    row.append(f"{a.get('spin_density', 0):.5f}")
                rows.append(tuple(row))
            blocks.append(f"{H3} NPA Charges\n" + _table(rows))

        # ── Alpha and Beta NPA (UHF) ─────────────────────────────────────
        if is_uhf:
            for spin_key, label in [("alpha_spin", "α"), ("beta_spin", "β")]:
                spin_sec = nbo.get(spin_key, {})
                snpa = spin_sec.get("npa_summary", [])
                if snpa:
                    rows = [("Atom", "NPA charge", "Core", "Valence", "Rydberg", "Total")]
                    for a in snpa:
                        lbl = f"{a.get('symbol','?')}{a.get('index','')}"
                        rows.append((lbl,
                                     f"{a.get('natural_charge',0):.5f}",
                                     f"{a.get('core_pop',0):.5f}",
                                     f"{a.get('valence_pop',0):.5f}",
                                     f"{a.get('rydberg_pop',0):.5f}",
                                     f"{a.get('total_pop',0):.5f}"))
                    blocks.append(f"{H3} NPA — {label} spin\n" + _table(rows))

        # ── Wiberg bond indices ───────────────────────────────────────────
        wbi_key = "overall_wiberg_bond_indices" if is_uhf else "wiberg_bond_indices"
        wbi     = nbo.get(wbi_key, {})
        matrix  = wbi.get("matrix", []) if isinstance(wbi, dict) else []
        if matrix and isinstance(matrix, list) and isinstance(matrix[0], list):
            atom_syms = ctx.get("atom_symbols", [])
            n = len(matrix)
            bonds_wbi = [
                (f"{atom_syms[i]}{i+1}" if i < len(atom_syms) else str(i+1),
                 f"{atom_syms[j]}{j+1}" if j < len(atom_syms) else str(j+1),
                 matrix[i][j])
                for i in range(n) for j in range(i+1, n)
                if isinstance(matrix[i][j], (int, float)) and matrix[i][j] > 0.05
            ]
            if bonds_wbi:
                rows = [("Bond", "WBI")]
                for si, sj, val in bonds_wbi:
                    rows.append((f"{si}\u2013{sj}", f"{val:.4f}"))
                blocks.append(f"{H3} Wiberg Bond Indices ({len(bonds_wbi)} bonds)\n" + _table(rows))

        # ── Alpha/Beta Wiberg (UHF) ───────────────────────────────────────
        if is_uhf:
            for spin_key, label in [("alpha_spin", "α"), ("beta_spin", "β")]:
                spin_sec = nbo.get(spin_key, {})
                swbi = spin_sec.get("wiberg_bond_indices", {})
                smat = swbi.get("matrix", []) if isinstance(swbi, dict) else []
                if smat and isinstance(smat, list) and isinstance(smat[0], list):
                    atom_syms = ctx.get("atom_symbols", [])
                    n = len(smat)
                    sbonds = [
                        (f"{atom_syms[i]}{i+1}" if i < len(atom_syms) else str(i+1),
                         f"{atom_syms[j]}{j+1}" if j < len(atom_syms) else str(j+1),
                         smat[i][j])
                        for i in range(n) for j in range(i+1, n)
                        if isinstance(smat[i][j], (int, float)) and smat[i][j] > 0.05
                    ]
                    if sbonds:
                        rows = [("Bond", "WBI")]
                        for si, sj, val in sbonds:
                            rows.append((f"{si}\u2013{sj}", f"{val:.4f}"))
                        blocks.append(f"{H3} Wiberg — {label} spin ({len(sbonds)} bonds)\n" + _table(rows))

        # ── E(2) perturbation ─────────────────────────────────────────────
        if is_uhf:
            for spin_key, label in [("alpha_spin", "α"), ("beta_spin", "β")]:
                e2 = nbo.get(spin_key, {}).get("e2_perturbation", [])
                if e2:
                    blocks.append(
                        f"{H3} E(2) Perturbation — {label} spin ({len(e2)} entries)\n"
                        + _render_e2_table(e2)
                    )
        else:
            e2 = nbo.get("e2_perturbation", [])
            if e2:
                blocks.append(f"{H3} Second-Order Perturbation E(2)\n" + _render_e2_table(e2))

    # ── Geometry ──────────────────────────────────────────────────────────
    geom = data.get("geometry", {})
    atoms = geom.get("cartesian_angstrom", [])
    if atoms:
        blocks.append(f"{H2} Geometry (Å)")
        rows = [("Atom", "x", "y", "z")]
        for i, a in enumerate(atoms):
            rows.append((
                f"{a.get('symbol','?')}{i+1}",
                f"{a['x_ang']:.6f}", f"{a['y_ang']:.6f}", f"{a['z_ang']:.6f}",
            ))
        blocks.append(_table(rows))

    return "\n\n".join(blocks) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Multi-molecule comparison renderer
# ─────────────────────────────────────────────────────────────────────────────

def _render_comparison(datasets: List[Dict[str, Any]]) -> str:
    """Comparison document: overview table + individual molecule sections."""
    if not datasets:
        return "# ORCA Comparison\n\n*No data provided.*\n"

    blocks: List[str] = ["# ORCA Calculation Comparison"]

    labels = _comparison_labels(datasets)

    # ── Method table ───────────────────────────────────────────────────────
    rows = [("", "method", "basis", "charge", "mult", "symmetry")]
    for lbl, d in zip(labels, datasets):
        meta = d.get("metadata", {})
        ctx  = d.get("context",  {})
        rows.append((
            lbl,
            f"{ctx.get('hf_type','?')} {meta.get('functional','?')}",
            meta.get("basis_set", "?"),
            str(meta.get("charge", "?")),
            str(meta.get("multiplicity", "?")),
            _irrep_group(d) if ctx.get("has_symmetry") else "C1",
        ))
    blocks.append("## Methods\n" + _table(rows))

    # ── Energy table ───────────────────────────────────────────────────────
    rows = [("", "E (Eh)", "⟨S²⟩", "ideal", "dipole (D)")]
    has_s2 = any(d.get("scf", {}).get("s_squared") is not None for d in datasets)
    if not has_s2:
        rows = [("", "E (Eh)", "dipole (D)")]
    for lbl, d in zip(labels, datasets):
        scf = d.get("scf", {})
        dip = d.get("dipole", {})
        E   = scf.get("final_single_point_energy_Eh")
        s2  = scf.get("s_squared")
        s2i = scf.get("s_squared_ideal", "")
        mu  = dip.get("magnitude_Debye")
        if has_s2:
            rows.append((lbl,
                         f"{E:.10f}" if E else "—",
                         f"{s2:.6f}"  if s2 is not None else "—",
                         str(s2i),
                         f"{mu:.4f}" if mu else "—"))
        else:
            rows.append((lbl,
                         f"{E:.10f}" if E else "—",
                         f"{mu:.4f}" if mu else "—"))
    blocks.append("## Energies\n" + _table(rows))

    # ── Frontier orbitals comparison ───────────────────────────────────────
    is_uhf_any = any(d.get("context", {}).get("is_uhf") for d in datasets)
    oe_blocks = []
    for lbl, d in zip(labels, datasets):
        oe  = d.get("orbital_energies", {})
        ctx = d.get("context", {})
        if not oe:
            continue
        if not ctx.get("is_uhf"):
            gap = oe.get("HOMO_LUMO_gap_eV")
            oe_blocks.append(f"{lbl}: HOMO {_f(oe.get('HOMO_energy_eV'))} / "
                             f"LUMO {_f(oe.get('LUMO_energy_eV'))} / "
                             f"gap {_f(gap)} eV")
        else:
            oe_blocks.append(f"{lbl}: α-HOMO {_f(oe.get('alpha_HOMO_energy_eV'))} / "
                             f"β-HOMO {_f(oe.get('beta_HOMO_energy_eV'))} eV")
        qro = d.get("qro")
        if qro:
            oe_blocks.append(f"  QRO: DOMO={qro.get('n_domo')} "
                             f"SOMO={qro.get('n_somo')} VMO={qro.get('n_vmo')}")
            for s in qro.get("somo_details", []):
                irr = f" ({s['irrep']})" if "irrep" in s else ""
                oe_blocks.append(f"  SOMO {s['index']}{irr}: "
                                 f"α={s['alpha_energy_eV']:.4f} "
                                 f"β={s['beta_energy_eV']:.4f} eV")
    if oe_blocks:
        blocks.append("## Frontier Orbitals\n" + "\n".join(oe_blocks))

    # ── Charge comparison ──────────────────────────────────────────────────
    # Only show schemes present in ALL molecules
    if any(d.get("epr") for d in datasets):
        rows = [[
            "",
            "D (cm^-1)",
            "E/D",
            "g_iso",
            "g_x",
            "g_y",
            "g_z",
            "top A_iso (MHz)",
            "top nucleus",
        ]]
        for lbl, d in zip(labels, datasets):
            epr = d.get("epr") or {}
            zfs = epr.get("zero_field_splitting") or {}
            g_vals = _epr_g_principal_values(epr)
            g_iso = _epr_g_iso(epr)
            top_nucleus, top_aiso = _epr_top_hyperfine(epr)
            rows.append((
                lbl,
                _f(zfs.get("D_cm-1"), ".6g"),
                _f(zfs.get("E_over_D")),
                _f(g_iso, ".7f"),
                _f(g_vals[0], ".7f") if len(g_vals) > 0 else "—",
                _f(g_vals[1], ".7f") if len(g_vals) > 1 else "—",
                _f(g_vals[2], ".7f") if len(g_vals) > 2 else "—",
                _f(top_aiso),
                top_nucleus or "—",
            ))
        blocks.append("## EPR / Magnetic\n" + _table(rows))

    all_schemes: List[set] = []
    for d in datasets:
        schemes = set()
        for s in ("mulliken","loewdin","hirshfeld","mbis","chelpg"):
            if _get_charges(d, s):
                schemes.add(s)
        all_schemes.append(schemes)
    common_schemes = sorted(set.intersection(*all_schemes)) if all_schemes else []

    if common_schemes:
        # one table per scheme, rows = molecules, cols = atoms
        for scheme in common_schemes:
            atom_syms = datasets[0].get("context", {}).get("atom_symbols", [])
            n_atoms   = len(_get_charges(datasets[0], scheme) or [])
            header    = ("",) + tuple(
                f"{atom_syms[i]}{i+1}" if i < len(atom_syms) else str(i+1)
                for i in range(n_atoms)
            )
            rows = [header]
            for lbl, d in zip(labels, datasets):
                charges = _get_charges(d, scheme) or []
                rows.append((lbl,) + tuple(f"{c:.4f}" for c in charges))
            blocks.append(f"## {scheme.capitalize()} Charges\n" + _table(rows))

    # ── Per-molecule full reports ──────────────────────────────────────────
    blocks.append("---\n# Individual Reports")
    for lbl, d in zip(labels, datasets):
        blocks.append(_render_molecule(
            d,
            heading_level=2,
            display_label=lbl,
            source_display=lbl,
        ))

    return "\n\n".join(blocks) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _table(rows) -> str:
    """Build a GitHub-flavoured markdown table from a list of tuples."""
    if not rows:
        return ""
    rows = [tuple(str(c) for c in r) for r in rows]
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    sep    = "| " + " | ".join("-" * w for w in widths) + " |"
    lines  = []
    for idx, row in enumerate(rows):
        line = "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(row)) + " |"
        lines.append(line)
        if idx == 0:
            lines.append(sep)
    return "\n".join(lines)


def _f(val, fmt=".4f") -> str:
    """Format a float or return '—'."""
    if val is None:
        return "—"
    try:
        return format(float(val), fmt)
    except (TypeError, ValueError):
        return str(val)


def _render_tddft_section(tddft: Dict[str, Any]) -> str:
    """Compact TDDFT/CIS summary for markdown reports."""
    final_block = tddft.get("final_excited_state_block")
    if not final_block:
        blocks = tddft.get("excited_state_blocks", [])
        final_block = blocks[-1] if blocks else None
    if not final_block:
        return ""

    lines = []
    summary = tddft.get("summary", {})
    meta_bits = []
    if summary.get("input_block"):
        meta_bits.append(f"input `%{summary['input_block']}`")
    if summary.get("nroots") is not None:
        meta_bits.append(f"nroots={summary['nroots']}")
    if summary.get("tda") is not None:
        meta_bits.append(f"tda={summary['tda']}")
    if summary.get("donto") is not None:
        meta_bits.append(f"donto={summary['donto']}")
    if final_block.get("method"):
        meta_bits.append(final_block["method"])
    if final_block.get("manifold"):
        meta_bits.append(str(final_block["manifold"]).lower())
    if meta_bits:
        lines.append(" | ".join(str(bit) for bit in meta_bits))

    states = final_block.get("states", [])
    if states:
        fosc_map = {}
        spectrum = tddft.get("spectra", {}).get("absorption_electric_dipole", {})
        for transition in spectrum.get("transitions", []):
            root = transition.get("to_root")
            if root is not None:
                fosc_map[root] = transition.get("oscillator_strength")

        rows = [("State", "E (eV)", "lambda (nm)", "fosc", "dominant excitation", "weight")]
        for state in states[:10]:
            dominant = max(
                state.get("transitions", []),
                key=lambda item: item.get("weight", 0.0),
                default={},
            )
            excitation = "—"
            if dominant:
                excitation = (
                    f"{dominant.get('from_orbital', '')} -> "
                    f"{dominant.get('to_orbital', '')}"
                )
            rows.append((
                str(state.get("state", "")),
                _f(state.get("energy_eV")),
                _f(state.get("wavelength_nm")),
                _f(fosc_map.get(state.get("state"))),
                excitation,
                _f(dominant.get("weight"), ".3f") if dominant else "—",
            ))
        lines.append(_table(rows))

    nto_states = tddft.get("nto_states", [])
    if nto_states:
        rows = [("State", "leading NTO pair", "occ.")]
        for nto_state in nto_states[:5]:
            lead = max(
                nto_state.get("pairs", []),
                key=lambda item: item.get("occupation", 0.0),
                default={},
            )
            if not lead:
                continue
            rows.append((
                str(nto_state.get("state", "")),
                f"{lead.get('from_orbital', '')} -> {lead.get('to_orbital', '')}",
                _f(lead.get("occupation")),
            ))
        if len(rows) > 1:
            lines.append("**Leading NTO Pairs**\n" + _table(rows))

    total_energy = tddft.get("total_energy", {})
    if total_energy:
        energy_bits = []
        if total_energy.get("root") is not None:
            energy_bits.append(f"root {total_energy['root']}")
        if total_energy.get("delta_energy_Eh") is not None:
            energy_bits.append(f"DE = {_f(total_energy['delta_energy_Eh'], '.6f')} Eh")
        if total_energy.get("total_energy_Eh") is not None:
            energy_bits.append(f"E(tot) = {_f(total_energy['total_energy_Eh'], '.6f')} Eh")
        if energy_bits:
            lines.append("**Total Energy:** " + " | ".join(energy_bits))

    return "\n\n".join(lines)


def _render_basis_set_section(basis_set: Dict[str, Any]) -> str:
    """Compact basis-set summary for markdown reports."""
    lines: List[str] = []

    bits = []
    if basis_set.get("n_basis_functions") is not None:
        bits.append(f"**Basis functions:** {basis_set['n_basis_functions']}")
    if basis_set.get("n_shells") is not None:
        bits.append(f"**Shells:** {basis_set['n_shells']}")
    if basis_set.get("max_angular_momentum") is not None:
        bits.append(f"**Max l:** {basis_set['max_angular_momentum']}")
    if basis_set.get("n_groups") is not None:
        bits.append(f"**Distinct groups:** {basis_set['n_groups']}")
    if bits:
        lines.append(" | ".join(bits))

    groups = basis_set.get("groups") or {}
    if groups:
        rows = [("Group", "Element", "Description")]
        for idx in sorted(groups):
            group = groups[idx]
            rows.append((
                str(idx),
                str(group.get("element", "")),
                str(group.get("description", "")),
            ))
        lines.append(_table(rows))

    mapping = basis_set.get("atom_group_mapping") or {}
    if mapping:
        counts: Dict[int, int] = {}
        for group_idx in mapping.values():
            counts[group_idx] = counts.get(group_idx, 0) + 1
        rows = [("Group", "Atoms mapped")]
        for idx in sorted(counts):
            rows.append((str(idx), str(counts[idx])))
        lines.append("**Atom-to-group counts**\n" + _table(rows))

    return "\n\n".join(lines)


def _render_geom_opt_section(geom_opt: Dict[str, Any]) -> str:
    """Compact geometry-optimization summary for markdown reports."""
    lines: List[str] = []

    bits = []
    converged = geom_opt.get("converged")
    if converged is not None:
        bits.append(f"**Converged:** {'yes' if converged else 'no'}")
    if geom_opt.get("n_cycles") is not None:
        bits.append(f"**Cycles:** {geom_opt['n_cycles']}")
    if geom_opt.get("opt_keyword"):
        bits.append(f"**Keyword:** {geom_opt['opt_keyword']}")
    if geom_opt.get("final_energy_Eh") is not None:
        bits.append(f"**Final energy:** {_f(geom_opt['final_energy_Eh'], '.10f')} Eh")
    if bits:
        lines.append(" | ".join(bits))

    rmsd_bits = []
    if geom_opt.get("rmsd_initial_to_final_ang") is not None:
        rmsd_bits.append(
            f"**RMSD initial->final:** {_f(geom_opt['rmsd_initial_to_final_ang'])} A"
        )
    if geom_opt.get("mass_weighted_rmsd_initial_to_final_ang") is not None:
        rmsd_bits.append(
            "**MW RMSD initial->final:** "
            f"{_f(geom_opt['mass_weighted_rmsd_initial_to_final_ang'])} A"
        )
    if rmsd_bits:
        lines.append(" | ".join(rmsd_bits))

    settings = geom_opt.get("settings") or {}
    if settings:
        rows = [("Setting", "Value")]
        for key, value in settings.items():
            rows.append((str(key), str(value)))
        lines.append("**Settings**\n" + _table(rows))

    tolerances = geom_opt.get("tolerances") or {}
    if tolerances:
        rows = [("Tolerance", "Value")]
        for key, value in tolerances.items():
            rows.append((str(key), str(value)))
        lines.append("**Convergence Tolerances**\n" + _table(rows))

    cycles = geom_opt.get("cycles") or []
    if cycles:
        rows = [[
            "Cycle",
            "E (Eh)",
            "dE (Eh)",
            "RMS grad",
            "Max grad",
            "RMS step",
            "Max step",
            "Trust (bohr)",
        ]]
        cycle_entries: List[Optional[Dict[str, Any]]] = list(cycles)
        note = ""
        if len(cycle_entries) > 12:
            cycle_entries = cycle_entries[:6] + [None] + cycle_entries[-6:]
            note = f"*Showing first 6 and last 6 of {len(cycles)} cycles.*"

        for cyc in cycle_entries:
            if cyc is None:
                rows.append(("...", "...", "...", "...", "...", "...", "...", "..."))
                continue
            conv = cyc.get("convergence") or {}
            rows.append((
                str(cyc.get("cycle", "")),
                _f(cyc.get("energy_Eh"), ".10f"),
                _f(cyc.get("energy_change_Eh"), ".6f"),
                _f((conv.get("rms_gradient") or {}).get("value"), ".2e"),
                _f((conv.get("max_gradient") or {}).get("value"), ".2e"),
                _f((conv.get("rms_step") or {}).get("value"), ".2e"),
                _f((conv.get("max_step") or {}).get("value"), ".2e"),
                _f(cyc.get("trust_radius_bohr"), ".3f"),
            ))

        cycle_table = _table(rows)
        if note:
            cycle_table += f"\n\n{note}"
        lines.append("**Cycle Summary**\n" + cycle_table)

    return "\n\n".join(lines)


def _render_epr_section(epr: Dict[str, Any], heading_level: int = 3) -> str:
    """Compact EPR summary for markdown reports."""
    H = "#" * heading_level
    lines: List[str] = []

    zfs = epr.get("zero_field_splitting") or {}
    if zfs:
        lines.append(f"{H} Zero-Field Splitting")
        bits = []
        if zfs.get("D_cm-1") is not None:
            bits.append(f"**D:** {_f(zfs['D_cm-1'], '.6g')} cm^-1")
        if zfs.get("E_over_D") is not None:
            bits.append(f"**E/D:** {_f(zfs['E_over_D'])}")
        if "right_handed" in zfs:
            bits.append(f"**Right-handed:** {'yes' if zfs['right_handed'] else 'no'}")
        if bits:
            lines.append(" | ".join(bits))

        eigs = zfs.get("eigenvalues") or []
        if eigs:
            lines.append(
                "**Principal values:** "
                + ", ".join(_f(value, ".6g") for value in eigs)
                + " cm^-1"
            )

        raw_matrix = zfs.get("raw_matrix")
        if raw_matrix:
            lines.append("**Raw D Tensor**\n" + _render_matrix(raw_matrix, ".6g"))

        contributions = zfs.get("individual_contributions") or {}
        if contributions:
            rows = [("Contribution", "D (cm^-1)", "E (cm^-1)")]
            for label, values in contributions.items():
                rows.append((
                    str(label),
                    _f(values.get("D_cm-1"), ".6g"),
                    _f(values.get("E_cm-1"), ".6g"),
                ))
            lines.append("**Contribution Breakdown**\n" + _table(rows))

    g_tensor = epr.get("g_tensor") or {}
    if g_tensor:
        lines.append(f"{H} g-Tensor")
        g_values = _epr_g_principal_values(epr)
        g_iso = _epr_g_iso(epr)
        bits = []
        if g_values:
            bits.append(
                "**Principal values:** "
                + ", ".join(_f(value, ".7f") for value in g_values)
            )
        if g_iso is not None:
            bits.append(f"**g_iso:** {_f(g_iso, '.7f')}")
        if bits:
            lines.append(" | ".join(bits))

        g_matrix = g_tensor.get("g_matrix")
        if g_matrix:
            lines.append("**g Matrix**\n" + _render_matrix(g_matrix, ".7f"))

        breakdown = g_tensor.get("breakdown") or {}
        if breakdown:
            rows = [("Contribution", "g_x", "g_y", "g_z", "g_iso")]
            for label, entry in breakdown.items():
                values = entry.get("values") or []
                rows.append((
                    str(label),
                    _f(values[0], ".7f") if len(values) > 0 else "—",
                    _f(values[1], ".7f") if len(values) > 1 else "—",
                    _f(values[2], ".7f") if len(values) > 2 else "—",
                    _f(entry.get("iso"), ".7f"),
                ))
            lines.append("**Breakdown**\n" + _table(rows))

        atom_analysis = (g_tensor.get("atom_analysis") or {}).get("atom_contributions") or []
        if atom_analysis:
            top_atoms = sorted(
                atom_analysis,
                key=lambda item: abs(item.get("iso", 0.0)),
                reverse=True,
            )
            shown = top_atoms[:8]
            rows = [("Atom", "g_x", "g_y", "g_z", "g_iso")]
            for atom in shown:
                values = atom.get("values") or []
                rows.append((
                    f"{atom.get('element', '?')}{atom.get('atom_index', '')}",
                    _f(values[0], ".7g") if len(values) > 0 else "—",
                    _f(values[1], ".7g") if len(values) > 1 else "—",
                    _f(values[2], ".7g") if len(values) > 2 else "—",
                    _f(atom.get("iso"), ".7g"),
                ))
            atom_text = "**Top Atom Contributions**\n" + _table(rows)
            if len(top_atoms) > len(shown):
                atom_text += (
                    f"\n\n*Showing top {len(shown)} of {len(top_atoms)} atoms "
                    "by |g_iso|.*"
                )
            lines.append(atom_text)

    hyperfine = epr.get("hyperfine") or {}
    if hyperfine:
        lines.append(f"{H} Hyperfine / EFG")
        bits = []
        if hyperfine.get("nucleus_count") is not None:
            bits.append(f"**Nuclei parsed:** {hyperfine['nucleus_count']}")
        requested = hyperfine.get("requested_counts") or {}
        if requested:
            bits.append(
                "**Requested:** "
                + ", ".join(f"{key}={value}" for key, value in requested.items())
            )
        if bits:
            lines.append(" | ".join(bits))

        hyperfine_table = _render_hyperfine_summary_table(hyperfine.get("nuclei") or [])
        if hyperfine_table:
            lines.append(hyperfine_table)

    return "\n\n".join(lines)


def _render_solvation_section(solvation: Dict[str, Any]) -> str:
    """Compact implicit-solvation summary for markdown reports."""
    summary = solvation.get("summary", {})
    lines: List[str] = []

    if solvation.get("is_solvated"):
        bits = []
        if solvation.get("primary_model"):
            bits.append(f"**Model:** {solvation['primary_model']}")
        if solvation.get("solvent"):
            bits.append(f"**Solvent:** {solvation['solvent']}")
        if summary.get("models"):
            bits.append(f"**Seen:** {', '.join(summary['models'])}")
        if bits:
            lines.append(" | ".join(bits))
    else:
        message = "**Final Input State:** no implicit solvation detected"
        if summary.get("output_model"):
            message += f" (earlier output blocks seen: {summary['output_model']}"
            if summary.get("output_solvent"):
                message += f" in {summary['output_solvent']}"
            message += ")"
        lines.append(message)

    flags = []
    if solvation.get("input_flags", {}).get("draco"):
        flags.append("DRACO")
    if solvation.get("input_flags", {}).get("smd18"):
        flags.append("SMD18")
    if flags:
        lines.append("**Flags:** " + ", ".join(flags))

    model = solvation.get("primary_model")
    if model in {"CPCM", "CPCMC", "SMD"} and solvation.get("cpcm"):
        cpcm = solvation["cpcm"]
        rows = [("parameter", "value")]
        if cpcm.get("epsilon") is not None:
            rows.append(("epsilon", _f(cpcm.get("epsilon"))))
        if cpcm.get("refractive_index") is not None:
            rows.append(("refrac", _f(cpcm.get("refractive_index"))))
        if cpcm.get("rsolv_ang") is not None:
            rows.append(("rsolv (A)", _f(cpcm.get("rsolv_ang"))))
        if cpcm.get("surface_type"):
            rows.append(("surface", str(cpcm["surface_type"])))
        if cpcm.get("epsilon_function_type"):
            rows.append(("eps. function", str(cpcm["epsilon_function_type"])))
        if model == "SMD" and cpcm.get("smd_descriptors"):
            for key in ("soln", "soln25", "sola", "solb", "solg", "solc", "solh"):
                if key in cpcm["smd_descriptors"]:
                    rows.append((key, _f(cpcm["smd_descriptors"][key])))
        if len(rows) > 1:
            lines.append(_table(rows))
    elif model == "ALPB" and solvation.get("alpb"):
        alpb = solvation["alpb"]
        rows = [("parameter", "value")]
        if alpb.get("epsilon") is not None:
            rows.append(("epsilon", _f(alpb.get("epsilon"))))
        if alpb.get("reference_state"):
            rows.append(("reference state", str(alpb["reference_state"])))
        if alpb.get("free_energy_shift_kcal_mol") is not None:
            rows.append(("free energy shift (kcal/mol)", _f(alpb.get("free_energy_shift_kcal_mol"))))
        if alpb.get("interaction_kernel"):
            rows.append(("kernel", str(alpb["interaction_kernel"])))
        if len(rows) > 1:
            lines.append(_table(rows))
    elif model == "COSMO-RS" and solvation.get("cosmors"):
        cosmors = solvation["cosmors"]
        rows = [("parameter", "value")]
        if cosmors.get("functional"):
            rows.append(("functional", str(cosmors["functional"])))
        if cosmors.get("basis_set"):
            rows.append(("basis", str(cosmors["basis_set"])))
        if cosmors.get("dGsolv_kcal_mol") is not None:
            rows.append(("dGsolv (kcal/mol)", _f(cosmors.get("dGsolv_kcal_mol"))))
        if len(rows) > 1:
            lines.append(_table(rows))

    return "\n\n".join(lines)


def _mol_label(data: Dict[str, Any]) -> str:
    """Short identifier for a molecule/calculation."""
    meta = data.get("metadata", {})
    name = meta.get("job_name") or Path(data.get("source_file", "mol")).stem
    return name


def _comparison_labels(datasets: List[Dict[str, Any]]) -> List[str]:
    """Return stable labels for comparison reports.

    Prefer source paths relative to the common parent directory of all parsed
    files. If that does not work, fall back to the shortest unique path suffix.
    """
    source_paths = [Path(d.get("source_file", "mol")) for d in datasets]
    if len(source_paths) <= 1:
        return [source_paths[0].name] if source_paths else []
    source_strs = [str(p) for p in source_paths]

    try:
        common = Path(os.path.commonpath(source_strs))
        relative = [Path(os.path.relpath(p, common)).as_posix() for p in source_paths]
        if len(set(relative)) == len(relative):
            return relative
    except ValueError:
        pass

    labels = [p.name for p in source_paths]
    depth = 2
    max_depth = max((len(p.parts) for p in source_paths), default=1) + 1
    while len(set(labels)) != len(labels) and depth <= max_depth:
        labels = []
        for path in source_paths:
            suffix = path.parts[-depth:] if len(path.parts) >= depth else path.parts
            labels.append(Path(*suffix).as_posix())
        depth += 1
    return labels


def _irrep_group(data: Dict[str, Any]) -> str:
    """Try to extract the point group label from the metadata or orbital data."""
    meta = data.get("metadata", {})
    pg   = meta.get("point_group", "")
    if pg:
        return pg
    # Fall back: look at irrep labels in orbital_energies
    oe = data.get("orbital_energies", {})
    orbs = oe.get("orbitals") or oe.get("alpha_orbitals") or []
    for o in orbs:
        irr = o.get("irrep", "")
        if irr:
            # irrep looks like "1-A1" → point group C2v etc.; just return the label
            label = irr.split("-", 1)[-1].strip() if "-" in irr else irr
            return label  # rough: just the first irrep label
    return "?"


def _get_atom_list(sec: dict) -> list:
    """Return per-atom list from a population section regardless of internal key name.

    Mulliken, Loewdin, CHELPG  → 'atomic_charges'
    Hirshfeld, MBIS             → 'atomic_data'
    fallback                    → 'atoms'
    """
    return sec.get("atoms") or sec.get("atomic_charges") or sec.get("atomic_data") or []


def _collect_charges(data, schemes=("mulliken", "loewdin", "hirshfeld", "mbis", "chelpg")):
    """Return dict of scheme → {atom_index: charge} for all present schemes."""
    result = {}
    for s in schemes:
        sec = data.get(s, {})
        if not sec:
            continue
        atoms = _get_atom_list(sec)
        d = {a["index"]: a["charge"] for a in atoms if "index" in a and "charge" in a}
        if d:
            result[s.capitalize()] = d
    return result


def _get_charges(data, scheme: str) -> Optional[List[float]]:
    """Return flat list of charges (positional) — used only for multi-mol comparison."""
    sec = data.get(scheme, {})
    if not sec:
        return None
    atoms   = _get_atom_list(sec)
    charges = [a.get("charge") for a in atoms if "charge" in a]
    return charges if charges else None


def _collect_spin(data, schemes=("mulliken", "loewdin", "hirshfeld", "mbis")):
    """Return dict of scheme → {atom_index: spin_population} for UHF calculations."""
    result = {}
    for s in schemes:
        sec = data.get(s, {})
        if not sec:
            continue
        atoms = _get_atom_list(sec)
        d = {a["index"]: a["spin_population"]
             for a in atoms if "index" in a and "spin_population" in a}
        if d:
            result[s.capitalize()] = d
    return result

# ─────────────────────────────────────────────────────────────────────────────
# Large-system summarization helpers
# ─────────────────────────────────────────────────────────────────────────────

def _group_by_symbol(atom_syms, values):
    """Group per-atom values by element symbol. Returns {symbol: [values]}."""
    groups = {}
    for i, sym in enumerate(atom_syms):
        if i < len(values) and values[i] is not None:
            groups.setdefault(sym, []).append(values[i])
    return groups


def _stats(vals):
    """Return (min, max, mean) for a list of floats."""
    if not vals:
        return None, None, None
    return min(vals), max(vals), sum(vals) / len(vals)


def _summarize_charges(charge_data, atom_syms, n_atoms, schemes) -> str:
    """Compact charge summary for large systems: per-element stats."""
    lines = [f"*{n_atoms} atoms — summarized by element*\n"]
    for scheme in schemes:
        vals = charge_data[scheme]
        groups = _group_by_symbol(atom_syms, vals)
        rows = [("Element", "n", "min", "max", "mean")]
        for sym in sorted(groups):
            g = groups[sym]
            mn, mx, av = _stats(g)
            rows.append((sym, str(len(g)), f"{mn:.4f}", f"{mx:.4f}", f"{av:.4f}"))
        lines.append(f"**{scheme}:**\n" + _table(rows))
    return "\n\n".join(lines)


def _summarize_spin(spin_data, atom_syms, n_atoms, schemes) -> str:
    """Compact spin population summary for large systems."""
    lines = [f"*{n_atoms} atoms — summarized by element*\n"]
    for scheme in schemes:
        vals = spin_data[scheme]
        groups = _group_by_symbol(atom_syms, vals)
        total  = sum(sum(g) for g in groups.values())
        rows   = [("Element", "n", "min", "max", "mean", "total")]
        for sym in sorted(groups):
            g = groups[sym]
            mn, mx, av = _stats(g)
            rows.append((sym, str(len(g)), f"{mn:.4f}", f"{mx:.4f}", f"{av:.4f}", f"{sum(g):.4f}"))
        rows.append(("**Total**", "", "", "", "", f"**{total:.4f}**"))
        lines.append(f"**{scheme}:**\n" + _table(rows))
    return "\n\n".join(lines)


def _summarize_bond_orders(bo_list) -> str:
    """Compact bond order summary: distribution by bond type."""
    import math
    # Bin bonds by type pair and rounded order
    type_bins = {}
    for b in bo_list:
        si, sj = b.get("symbol_i", "?"), b.get("symbol_j", "?")
        key = f"{min(si,sj)}–{max(si,sj)}"
        type_bins.setdefault(key, []).append(b.get("bond_order", 0.0))

    lines = [f"*{len(bo_list)} bonds total — distribution by bond type*\n"]
    rows  = [("Type", "n", "min", "max", "mean")]
    for btype in sorted(type_bins):
        vals = type_bins[btype]
        mn, mx, av = _stats(vals)
        rows.append((btype, str(len(vals)), f"{mn:.4f}", f"{mx:.4f}", f"{av:.4f}"))
    lines.append(_table(rows))

    # Also show distinct bond-order clusters (rounded to 3 dp)
    order_counts = {}
    for b in bo_list:
        key = round(b.get("bond_order", 0.0), 3)
        order_counts[key] = order_counts.get(key, 0) + 1
    if len(order_counts) <= 10:
        lines.append("\n*Bond-order clusters:*  " +
                     "  ".join(f"{k:.3f}×{v}" for k, v in sorted(order_counts.items())))
    return "\n".join(lines)


def _summarize_npa(npa, has_spin) -> str:
    """Compact NPA summary for large systems."""
    groups_q  = {}
    groups_sp = {}
    for a in npa:
        sym = a.get("symbol", "?")
        q   = a.get("natural_charge")
        sp  = a.get("spin_density")
        if q is not None:
            groups_q.setdefault(sym, []).append(q)
        if sp is not None:
            groups_sp.setdefault(sym, []).append(sp)

    n_total = len(npa)
    lines   = [f"*{n_total} atoms — summarized by element*\n"]

    if has_spin:
        rows = [("Element", "n", "q min", "q max", "q mean", "spin total")]
        for sym in sorted(groups_q):
            qg  = groups_q.get(sym, [])
            spg = groups_sp.get(sym, [])
            mn, mx, av = _stats(qg)
            rows.append((sym, str(len(qg)), f"{mn:.4f}", f"{mx:.4f}", f"{av:.4f}",
                         f"{sum(spg):.4f}" if spg else "—"))
    else:
        rows = [("Element", "n", "q min", "q max", "q mean")]
        for sym in sorted(groups_q):
            qg = groups_q.get(sym, [])
            mn, mx, av = _stats(qg)
            rows.append((sym, str(len(qg)), f"{mn:.4f}", f"{mx:.4f}", f"{av:.4f}"))

    lines.append(_table(rows))
    return "\n".join(lines)


def _summarize_wbi(bonds_wbi) -> str:
    """Compact WBI summary for large systems."""
    type_bins = {}
    for si, sj, val in bonds_wbi:
        sym_i = ''.join(c for c in si if c.isalpha())
        sym_j = ''.join(c for c in sj if c.isalpha())
        key   = f"{min(sym_i,sym_j)}–{max(sym_i,sym_j)}"
        type_bins.setdefault(key, []).append(val)

    lines = [f"*{len(bonds_wbi)} bonds (WBI > 0.05) — distribution by type*\n"]
    rows  = [("Type", "n", "min", "max", "mean")]
    for btype in sorted(type_bins):
        vals = type_bins[btype]
        mn, mx, av = _stats(vals)
        rows.append((btype, str(len(vals)), f"{mn:.4f}", f"{mx:.4f}", f"{av:.4f}"))
    lines.append(_table(rows))
    return "\n".join(lines)


def _render_e2_table(e2) -> str:
    """Render E(2) table: top 20 by energy, then summary of rest."""
    top   = sorted(e2, key=lambda x: -x.get("E2_kcal_mol", 0))
    shown = top[:20]
    rows  = [("Donor", "Acceptor", "E(2) kcal/mol", "ΔE a.u.", "F a.u.")]
    for entry in shown:
        rows.append((
            entry.get("donor",    ""),
            entry.get("acceptor", ""),
            f"{entry.get('E2_kcal_mol', 0):.2f}",
            f"{entry.get('E_gap_au',    0):.4f}",
            f"{entry.get('Fock_au',     0):.4f}",
        ))
    result = _table(rows)
    if len(e2) > 20:
        total = sum(e.get("E2_kcal_mol", 0) for e in e2)
        result += f"\n\n*Showing top 20 of {len(e2)} interactions. Total E(2) = {total:.2f} kcal/mol.*"
    return result


def _render_matrix(matrix, value_fmt=".6f") -> str:
    """Render a 3x3 matrix as a markdown table."""
    rows = [("", "x", "y", "z")]
    labels = ("x", "y", "z")
    for label, values in zip(labels, matrix):
        rows.append((label,) + tuple(_f(value, value_fmt) for value in values[:3]))
    return _table(rows)


def _epr_g_total_entry(epr: Dict[str, Any]) -> Dict[str, Any]:
    """Return the total g-tensor entry when available."""
    breakdown = (epr.get("g_tensor") or {}).get("breakdown") or {}
    if "g(tot)" in breakdown:
        return breakdown["g(tot)"]
    for key, entry in breakdown.items():
        norm = key.lower().replace(" ", "")
        if "g" in norm and "tot" in norm:
            return entry
    return {}


def _epr_g_principal_values(epr: Dict[str, Any]) -> List[float]:
    """Return principal g values, falling back to the g-matrix diagonal."""
    entry = _epr_g_total_entry(epr)
    values = entry.get("values") or []
    if values:
        return list(values)

    g_matrix = (epr.get("g_tensor") or {}).get("g_matrix") or []
    if len(g_matrix) == 3 and all(len(row) >= 3 for row in g_matrix):
        return [g_matrix[0][0], g_matrix[1][1], g_matrix[2][2]]
    return []


def _epr_g_iso(epr: Dict[str, Any]) -> Optional[float]:
    """Return isotropic g when available."""
    entry = _epr_g_total_entry(epr)
    iso = entry.get("iso")
    if iso is not None:
        return iso

    values = _epr_g_principal_values(epr)
    if len(values) == 3:
        return sum(values) / 3.0
    return None


def _epr_top_hyperfine(epr: Dict[str, Any]) -> tuple[Optional[str], Optional[float]]:
    """Return the label and A_iso of the strongest hyperfine nucleus."""
    nuclei = (epr.get("hyperfine") or {}).get("nuclei") or []
    ranked = _rank_hyperfine_nuclei(nuclei)
    if not ranked:
        return None, None
    top = ranked[0]
    return _format_nucleus_label(top["nucleus"]), top["A_iso_MHz"]


def _rank_hyperfine_nuclei(nuclei: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return nuclei ranked by absolute A_iso."""
    ranked = []
    for nucleus in nuclei:
        principal = (nucleus.get("principal_components") or {}).get("A(Tot)") or {}
        values = principal.get("values_MHz") or []
        a_iso = principal.get("A_iso_MHz")
        if a_iso is None:
            matrix = nucleus.get("total_HFC_matrix_MHz") or []
            if len(matrix) == 3 and all(len(row) >= 3 for row in matrix):
                a_iso = (matrix[0][0] + matrix[1][1] + matrix[2][2]) / 3.0
        if a_iso is None and not values:
            continue
        ranked.append({
            "nucleus": nucleus,
            "A_iso_MHz": a_iso,
            "values_MHz": list(values),
            "quadrupole": nucleus.get("quadrupole_coupling") or {},
        })
    ranked.sort(
        key=lambda item: abs(item.get("A_iso_MHz") or 0.0),
        reverse=True,
    )
    return ranked


def _format_nucleus_label(nucleus: Dict[str, Any]) -> str:
    """Format a compact nucleus label for markdown tables."""
    element = str(nucleus.get("element", "?"))
    index = nucleus.get("nucleus_index")
    label = str(nucleus.get("label", "") or "")
    isotope = nucleus.get("isotope")

    if isinstance(index, int):
        text = f"{element}{index + 1}"
    else:
        text = element
    if label:
        text += label
    if isotope is not None:
        text += f" ({isotope})"
    return text


def _render_hyperfine_summary_table(nuclei: List[Dict[str, Any]]) -> str:
    """Render the strongest hyperfine nuclei as a compact markdown table."""
    ranked = _rank_hyperfine_nuclei(nuclei)
    if not ranked:
        return ""

    shown = ranked[:15]
    rows = [("Nucleus", "A_iso (MHz)", "A_x", "A_y", "A_z", "e2qQ (MHz)", "eta")]
    for item in shown:
        values = item.get("values_MHz") or []
        quadrupole = item.get("quadrupole") or {}
        rows.append((
            _format_nucleus_label(item["nucleus"]),
            _f(item.get("A_iso_MHz")),
            _f(values[0]) if len(values) > 0 else "—",
            _f(values[1]) if len(values) > 1 else "—",
            _f(values[2]) if len(values) > 2 else "—",
            _f(quadrupole.get("e2qQ_MHz")),
            _f(quadrupole.get("eta")),
        ))

    table = "**Top Hyperfine Couplings**\n" + _table(rows)
    if len(ranked) > len(shown):
        table += (
            f"\n\n*Showing top {len(shown)} of {len(ranked)} nuclei "
            "by |A_iso|.*"
        )
    return table
