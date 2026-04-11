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

from ..final_snapshot import (
    get_final_dipole as _get_final_dipole,
    get_final_geometry as _get_final_geometry,
    get_final_orbital_energies as _get_final_orbital_energies,
)
from ..job_series import (
    get_geom_opt_series as _get_geom_opt_series,
    get_goat_series as _get_goat_series,
    get_surface_scan_series as _get_surface_scan_series,
)
from .job_state import (
    deltascf_target_summary as _shared_deltascf_target_summary,
    electronic_state_label as _electronic_state_label,
    excited_state_target_label as _excited_state_target_label,
    get_basis_set as _get_basis_set,
    get_charge as _get_charge,
    format_deltascf_vector as _shared_format_deltascf_vector,
    get_deltascf_data as _get_deltascf_data,
    get_excited_state_opt_data as _get_excited_state_opt_data,
    get_method_header_label as _get_method_header_label,
    get_method_table_label as _get_method_table_label,
    get_job_id as _get_job_id,
    get_job_name as _get_job_name,
    get_multiplicity as _get_multiplicity,
    get_symmetry_data as _get_symmetry_data,
    has_symmetry as _has_symmetry,
    has_symmetry_setup as _has_symmetry_setup,
    is_deltascf as _is_deltascf,
    is_excited_state_opt as _is_excited_state_opt,
    symmetry_inline_label as _symmetry_inline_label,
    symmetry_on_off as _symmetry_on_off,
    yes_no_unknown as _yes_no_unknown,
)
from .markdown_sections_analysis import (
    build_cmo_lookup as _build_cmo_lookup,
    build_orbital_irrep_lookup as _build_orbital_irrep_lookup,
    format_transition_cmo_character as _format_transition_cmo_character,
    format_transition_with_irreps as _format_transition_with_irreps,
    get_atom_list as _get_atom_list,
    get_charges as _get_charges,
    render_analysis_sections as _render_analysis_sections,
)
from .markdown_sections_basic import (
    render_basis_set_section as _render_basic_basis_set_section,
    render_dipole_section as _render_basic_dipole_section,
    render_geom_opt_section as _render_basic_geom_opt_section,
    render_goat_section as _render_basic_goat_section,
    render_solvation_section as _render_basic_solvation_section,
    render_surface_scan_section as _render_basic_surface_scan_section,
)
from .markdown_sections_spectroscopy import (
    epr_g_iso as _epr_g_iso,
    epr_g_principal_values as _epr_g_principal_values,
    epr_top_hyperfine as _epr_top_hyperfine,
    render_epr_section as _render_spectroscopy_epr_section,
    render_tddft_section as _render_spectroscopy_tddft_section,
)
from .markdown_sections_state import (
    render_deltascf_section as _render_state_deltascf_section,
    render_excited_state_opt_section as _render_state_excited_state_opt_section,
    render_symmetry_section as _render_state_symmetry_section,
)


AU_TO_DEBYE = 2.541746473


def _format_deltascf_vector(values: Any) -> str:
    """Preserve markdown-friendly numeric formatting for DeltaSCF vectors."""
    return _shared_format_deltascf_vector(values, formatter=_f)


def _deltascf_target_summary(deltascf: Dict[str, Any]) -> str:
    """Preserve markdown-friendly numeric formatting for DeltaSCF targets."""
    return _shared_deltascf_target_summary(deltascf, formatter=_f)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry points
# ─────────────────────────────────────────────────────────────────────────────

def write_markdown(
    data: Dict[str, Any],
    path: Path,
    *,
    goat_max_relative_energy_kcal_mol: Optional[float] = None,
) -> Path:
    """Write a single-molecule markdown report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _render_molecule(
            data,
            goat_max_relative_energy_kcal_mol=goat_max_relative_energy_kcal_mol,
        ),
        encoding="utf-8",
    )
    return path


def write_comparison(
    datasets: List[Dict[str, Any]],
    path: Path,
    *,
    goat_max_relative_energy_kcal_mol: Optional[float] = 10.0,
) -> Path:
    """Write a multi-molecule comparison markdown document."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _render_comparison(
            datasets,
            goat_max_relative_energy_kcal_mol=goat_max_relative_energy_kcal_mol,
        ),
        encoding="utf-8",
    )
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Single-molecule renderer
# ─────────────────────────────────────────────────────────────────────────────

def _render_molecule(
    data: Dict[str, Any],
    heading_level: int = 1,
    display_label: Optional[str] = None,
    source_display: Optional[str] = None,
    goat_max_relative_energy_kcal_mol: Optional[float] = None,
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
    job = display_label or _get_job_name(data) or src_path.stem
    charge = _get_charge(data)
    mult = _get_multiplicity(data)
    state_label = _electronic_state_label(data)
    sym_label = _symmetry_inline_label(data)
    state  = f"  state={state_label}" if state_label else ""
    sym    = f"  symmetry={sym_label}" if sym_label else ""
    job_id = _get_job_id(data)
    blocks.append(
        f"{H} {job}\n"
        f"`{_get_method_header_label(data)}` | charge={charge} mult={mult}{state}{sym}  \n"
        f"source: `{src}` | id: `{job_id}`"
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

    dip = _get_final_dipole(data)
    if dip.get("magnitude_Debye") is not None:
        vec = _get_dipole_vector(dip, "total_dipole", "Debye")
        xyz = ""
        if vec:
            xyz = (
                f"  ({vec.get('x', 0.0):.4f}, {vec.get('y', 0.0):.4f}, "
                f"{vec.get('z', 0.0):.4f}) D"
            )
        scf_lines.append(f"**Dipole:** {dip['magnitude_Debye']:.4f} D{xyz}")

    if scf_lines:
        blocks.append("\n".join(scf_lines))

    dipole_section = _render_dipole_section(dip)
    if dipole_section:
        blocks.append(f"{H2} Dipole Moment\n{dipole_section}")

    # -- Basis-set summary -----------------------------------------------
    basis_set = data.get("basis_set")
    if basis_set:
        basis_section = _render_basis_set_section(basis_set)
        if basis_section:
            blocks.append(f"{H2} Basis Set\n{basis_section}")

    symmetry_section = _render_symmetry_section(data)
    if symmetry_section:
        blocks.append(f"{H2} Symmetry\n{symmetry_section}")

    deltascf_section = _render_deltascf_section(data)
    if deltascf_section:
        blocks.append(f"{H2} DeltaSCF / Excited-State Target\n{deltascf_section}")

    excited_state_opt_section = _render_excited_state_opt_section(data)
    if excited_state_opt_section:
        blocks.append(f"{H2} Excited-State Geometry Optimization\n{excited_state_opt_section}")

    surface_scan = _get_surface_scan_series(data)
    if surface_scan:
        surface_scan_section = _render_surface_scan_section(surface_scan)
        if surface_scan_section:
            blocks.append(f"{H2} Relaxed Surface Scan\n{surface_scan_section}")

    goat = _get_goat_series(data)
    if goat:
        goat_section = _render_goat_section(
            goat,
            max_relative_energy_kcal_mol=goat_max_relative_energy_kcal_mol,
        )
        if goat_section:
            blocks.append(f"{H2} GOAT Conformer Search\n{goat_section}")

    # -- Geometry optimization summary -----------------------------------
    geom_opt = _get_geom_opt_series(data)
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
        tddft_section = _render_tddft_section(tddft, data)
        if tddft_section:
            blocks.append(f"{H2} TDDFT Excited States\n{tddft_section}")

    # ── Orbital energies ───────────────────────────────────────────────────
    oe = _get_final_orbital_energies(data)
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

        orbital_window = _render_irrep_orbital_window(data)
        if orbital_window:
            blocks.append(f"{H2} Irrep-Resolved Orbital Window\n{orbital_window}")

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

    blocks.extend(
        _render_analysis_sections(
            data,
            context=ctx,
            heading_level=heading_level,
            format_number=_f,
            make_table=_table,
        )
    )

    # ── Geometry ──────────────────────────────────────────────────────────
    geom = _get_final_geometry(data)
    atoms = geom.get("symmetry_cartesian_angstrom") or geom.get("cartesian_angstrom", [])
    if atoms:
        heading = "Geometry (symmetry-perfected, Å)" if geom.get("symmetry_cartesian_angstrom") else "Geometry (Å)"
        blocks.append(f"{H2} {heading}")
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

def _render_comparison(
    datasets: List[Dict[str, Any]],
    *,
    goat_max_relative_energy_kcal_mol: Optional[float] = 10.0,
) -> str:
    """Comparison document: overview table + individual molecule sections."""
    if not datasets:
        return "# ORCA Comparison\n\n*No data provided.*\n"

    blocks: List[str] = ["# ORCA Calculation Comparison"]

    labels = _comparison_labels(datasets)

    # ── Method table ───────────────────────────────────────────────────────
    rows = [("", "method", "basis", "charge", "mult", "electronic state", "symmetry")]
    for lbl, d in zip(labels, datasets):
        rows.append((
            lbl,
            _get_method_table_label(d),
            _get_basis_set(d) or "—",
            str(_get_charge(d) if _get_charge(d) != "" else "?"),
            str(_get_multiplicity(d) if _get_multiplicity(d) != "" else "?"),
            _electronic_state_label(d) or "ground-state",
            _symmetry_inline_label(d) or "C1",
        ))
    blocks.append("## Methods\n" + _table(rows))

    if any(_is_deltascf(d) for d in datasets):
        rows = [("", "electronic state", "target", "metric", "keep ref")]
        for lbl, d in zip(labels, datasets):
            deltascf = _get_deltascf_data(d)
            rows.append((
                lbl,
                _electronic_state_label(d) or "ground-state",
                _deltascf_target_summary(deltascf) or "—",
                deltascf.get("aufbau_metric") or "—",
                _yes_no_unknown(deltascf.get("keep_initial_reference")),
            ))
        blocks.append("## DeltaSCF\n" + _table(rows))

    if any(_is_excited_state_opt(d) for d in datasets):
        rows = [("", "electronic state", "target", "input", "followiroot", "SOC grad", "final root")]
        for lbl, d in zip(labels, datasets):
            excopt = _get_excited_state_opt_data(d)
            rows.append((
                lbl,
                _electronic_state_label(d) or "ground-state",
                _excited_state_target_label(excopt) or "—",
                f"%{excopt.get('input_block')}" if excopt.get("input_block") else "—",
                _yes_no_unknown(excopt.get("followiroot")) if "followiroot" in excopt else "—",
                _yes_no_unknown(excopt.get("socgrad")) if "socgrad" in excopt else "—",
                str(excopt.get("final_root")) if excopt.get("final_root") is not None else "—",
            ))
        blocks.append("## Excited-State Optimization\n" + _table(rows))

    if any(_has_symmetry(d) for d in datasets):
        rows = [("", "UseSym", "point group", "reduced", "orbital irreps", "n_irreps", "initial guess")]
        for lbl, d in zip(labels, datasets):
            sym = _get_symmetry_data(d)
            rows.append((
                lbl,
                _symmetry_on_off(sym),
                sym.get("point_group") or sym.get("auto_detected_point_group") or "?",
                sym.get("reduced_point_group") or "?",
                sym.get("orbital_irrep_group") or "?",
                str(sym.get("n_irreps", "?")),
                sym.get("initial_guess_irrep") or "?",
            ))
        blocks.append("## Symmetry\n" + _table(rows))

        if any(_has_symmetry_setup(_get_symmetry_data(d)) for d in datasets):
            rows = [("", "guess", "MO file", "geom match", "basis match", "reassign irreps", "renorm", "reorth")]
            for lbl, d in zip(labels, datasets):
                sym = _get_symmetry_data(d)
                reorth = _yes_no_unknown(sym.get("initial_guess_mos_reorthogonalized"))
                if reorth == "yes" and sym.get("initial_guess_reorthogonalization_method"):
                    reorth = f"{reorth} ({sym['initial_guess_reorthogonalization_method']})"
                rows.append((
                    lbl,
                    sym.get("initial_guess_method") or "—",
                    sym.get("initial_guess_source_file") or "—",
                    _yes_no_unknown(sym.get("initial_guess_geometry_matches")) if "initial_guess_geometry_matches" in sym else "—",
                    _yes_no_unknown(sym.get("initial_guess_basis_matches")) if "initial_guess_basis_matches" in sym else "—",
                    _yes_no_unknown(sym.get("initial_guess_irreps_reassigned")) if "initial_guess_irreps_reassigned" in sym else "—",
                    _yes_no_unknown(sym.get("initial_guess_mos_renormalized")) if "initial_guess_mos_renormalized" in sym else "—",
                    reorth if "initial_guess_mos_reorthogonalized" in sym else "—",
                ))
            blocks.append("## Symmetry Setup\n" + _table(rows))

        if any(
            _get_final_orbital_energies(d).get("alpha_occupied_per_irrep")
            or _get_final_orbital_energies(d).get("beta_occupied_per_irrep")
            or _get_final_orbital_energies(d).get("occupied_per_irrep")
            for d in datasets
        ):
            any_uhf = any(d.get("context", {}).get("is_uhf") for d in datasets)
            if any_uhf:
                rows = [("", "alpha occupied per irrep", "beta occupied per irrep")]
                for lbl, d in zip(labels, datasets):
                    rows.append((
                        lbl,
                        _compact_irrep_counts(d, "a"),
                        _compact_irrep_counts(d, "b"),
                    ))
            else:
                rows = [("", "occupied per irrep")]
                for lbl, d in zip(labels, datasets):
                    rows.append((lbl, _compact_irrep_counts(d)))
            blocks.append("## Symmetry Occupations\n" + _table(rows))

    if any(d.get("surface_scan") for d in datasets):
        rows = [("", "mode", "parameters", "steps", "coordinates", "span (kcal/mol)")]
        for lbl, d in zip(labels, datasets):
            scan = d.get("surface_scan") or {}
            parameters = scan.get("parameters") or []
            coord_summary = "; ".join(
                f"{parameter.get('label', '?')} "
                f"{_f(parameter.get('start'))}->{_f(parameter.get('end'))} "
                f"({parameter.get('steps', '?')})"
                if parameter.get("mode") != "values"
                else f"{parameter.get('label', '?')} [{len(parameter.get('values') or [])} values]"
                for parameter in parameters
            ) or "—"
            rows.append((
                lbl,
                scan.get("mode", "—"),
                str(scan.get("n_parameters", "—")),
                str(scan.get("n_constrained_optimizations", "—")),
                coord_summary,
                _f(scan.get("actual_energy_span_kcal_mol")),
            ))
        blocks.append("## Surface Scans\n" + _table(rows))

    # ── Energy table ───────────────────────────────────────────────────────
    rows = [("", "E (Eh)", "⟨S²⟩", "ideal", "dipole (D)")]
    has_s2 = any(d.get("scf", {}).get("s_squared") is not None for d in datasets)
    if not has_s2:
        rows = [("", "E (Eh)", "dipole (D)")]
    for lbl, d in zip(labels, datasets):
        scf = d.get("scf", {})
        dip = _get_final_dipole(d)
        E   = scf.get("final_single_point_energy_Eh")
        s2  = scf.get("s_squared")
        s2i = scf.get("s_squared_ideal", "")
        mu  = dip.get("magnitude_Debye")
        if has_s2:
            rows.append((lbl,
                         f"{E:.10f}" if E else "—",
                         f"{s2:.6f}"  if s2 is not None else "—",
                         str(s2i),
                         f"{mu:.4f}" if mu is not None else "—"))
        else:
            rows.append((lbl,
                         f"{E:.10f}" if E else "—",
                         f"{mu:.4f}" if mu is not None else "—"))
    blocks.append("## Energies\n" + _table(rows))

    # ── Frontier orbitals comparison ───────────────────────────────────────
    is_uhf_any = any(d.get("context", {}).get("is_uhf") for d in datasets)
    oe_blocks = []
    for lbl, d in zip(labels, datasets):
        oe  = _get_final_orbital_energies(d)
        ctx = d.get("context", {})
        if not oe:
            continue
        if not ctx.get("is_uhf"):
            gap = oe.get("HOMO_LUMO_gap_eV")
            oe_blocks.append(f"{lbl}: HOMO {_format_orbital_energy_with_irrep(oe.get('HOMO_energy_eV'), oe.get('HOMO_irrep'))} / "
                             f"LUMO {_format_orbital_energy_with_irrep(oe.get('LUMO_energy_eV'), oe.get('LUMO_irrep'))} / "
                             f"gap {_f(gap)} eV")
        else:
            oe_blocks.append(
                f"{lbl}: α-HOMO {_format_orbital_energy_with_irrep(oe.get('alpha_HOMO_energy_eV'), oe.get('alpha_HOMO_irrep'))} / "
                f"β-HOMO {_format_orbital_energy_with_irrep(oe.get('beta_HOMO_energy_eV'), oe.get('beta_HOMO_irrep'))} / "
                f"α-LUMO {_format_orbital_energy_with_irrep(oe.get('alpha_LUMO_energy_eV'), oe.get('alpha_LUMO_irrep'))} / "
                f"β-LUMO {_format_orbital_energy_with_irrep(oe.get('beta_LUMO_energy_eV'), oe.get('beta_LUMO_irrep'))}"
            )
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
            goat_max_relative_energy_kcal_mol=goat_max_relative_energy_kcal_mol,
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


def _get_dipole_vector(
    dipole: Dict[str, Any],
    stem: str,
    unit: str,
) -> Optional[Dict[str, float]]:
    """Return a dipole vector in the requested unit, converting from a.u. if needed."""
    vector = dipole.get(f"{stem}_{unit}")
    if isinstance(vector, dict):
        return vector

    if unit != "Debye":
        return None

    vector_au = dipole.get(f"{stem}_au")
    if not isinstance(vector_au, dict):
        return None
    return {
        axis: float(value) * AU_TO_DEBYE
        for axis, value in vector_au.items()
    }


def _format_vector(vector: Optional[Dict[str, Any]], fmt: str = ".6f") -> str:
    """Format an xyz vector for compact markdown display."""
    if not isinstance(vector, dict):
        return ""
    try:
        return ", ".join(
            format(float(vector.get(axis, 0.0)), fmt)
            for axis in ("x", "y", "z")
        )
    except (TypeError, ValueError):
        return ""


def _format_orbital_energy_with_irrep(value: Any, irrep: Any) -> str:
    """Format an orbital energy and append its irrep when available."""
    text = _f(value)
    if irrep:
        return f"{text} ({irrep})"
    return text


def _frontier_region_label(offset: int, occupied: bool) -> str:
    """Return HOMO/LUMO-style labels for frontier windows."""
    if occupied:
        return "HOMO" if offset == 0 else f"HOMO-{offset}"
    return "LUMO" if offset == 0 else f"LUMO+{offset}"


def _render_orbital_window_table(orbitals: List[Dict[str, Any]], window: int = 12) -> str:
    """Render a compact frontier orbital window with irrep labels."""
    if not orbitals or not any(orbital.get("irrep") for orbital in orbitals):
        return ""

    occupied = [orbital for orbital in orbitals if float(orbital.get("occupation", 0.0) or 0.0) > 1e-8]
    virtual = [orbital for orbital in orbitals if float(orbital.get("occupation", 0.0) or 0.0) <= 1e-8]
    if not occupied and not virtual:
        return ""

    rows = [("Region", "NO", "occ", "E (Eh)", "E (eV)", "irrep")]

    start_occ = max(0, len(occupied) - window)
    for idx, orbital in enumerate(occupied[start_occ:], start=start_occ):
        offset = len(occupied) - 1 - idx
        rows.append((
            _frontier_region_label(offset, occupied=True),
            str(orbital.get("index", "")),
            _f(orbital.get("occupation")),
            _f(orbital.get("energy_Eh"), ".6f"),
            _f(orbital.get("energy_eV")),
            str(orbital.get("irrep", "") or "—"),
        ))

    for offset, orbital in enumerate(virtual[:window]):
        rows.append((
            _frontier_region_label(offset, occupied=False),
            str(orbital.get("index", "")),
            _f(orbital.get("occupation")),
            _f(orbital.get("energy_Eh"), ".6f"),
            _f(orbital.get("energy_eV")),
            str(orbital.get("irrep", "") or "—"),
        ))

    return _table(rows)


def _render_irrep_orbital_window(data: Dict[str, Any], window: int = 12) -> str:
    """Render irrep-resolved frontier windows for RHF/UHF orbital lists."""
    oe = _get_final_orbital_energies(data)
    ctx = data.get("context", {})

    if ctx.get("is_uhf"):
        blocks: List[str] = []
        alpha_table = _render_orbital_window_table(oe.get("alpha_orbitals") or [], window=window)
        beta_table = _render_orbital_window_table(oe.get("beta_orbitals") or [], window=window)
        if alpha_table:
            blocks.append("**Alpha orbitals**\n" + alpha_table)
        if beta_table:
            blocks.append("**Beta orbitals**\n" + beta_table)
        return "\n\n".join(blocks)

    return _render_orbital_window_table(oe.get("orbitals") or [], window=window)


def _render_dipole_section(dipole: Dict[str, Any]) -> str:
    """Render permanent-dipole and rotational-axis dipole information."""
    return _render_basic_dipole_section(
        dipole,
        format_number=_f,
        make_table=_table,
        get_dipole_vector=_get_dipole_vector,
        format_vector=_format_vector,
    )


def _render_tddft_section(tddft: Dict[str, Any], data: Optional[Dict[str, Any]] = None) -> str:
    """Compact TDDFT/CIS summary for markdown reports."""
    return _render_spectroscopy_tddft_section(
        tddft,
        data,
        format_number=_f,
        make_table=_table,
        build_orbital_irrep_lookup=_build_orbital_irrep_lookup,
        build_cmo_lookup=_build_cmo_lookup,
        format_transition_with_irreps=_format_transition_with_irreps,
        format_transition_cmo_character=_format_transition_cmo_character,
        yes_no_unknown=_yes_no_unknown,
        get_excited_state_opt_data=_get_excited_state_opt_data,
    )


def _render_basis_set_section(basis_set: Dict[str, Any]) -> str:
    """Compact basis-set summary for markdown reports."""
    return _render_basic_basis_set_section(
        basis_set,
        make_table=_table,
    )


def _render_surface_scan_section(surface_scan: Dict[str, Any]) -> str:
    """Compact relaxed surface scan summary for markdown reports."""
    return _render_basic_surface_scan_section(
        surface_scan,
        format_number=_f,
        make_table=_table,
    )


def _render_goat_section(
    goat: Dict[str, Any],
    *,
    max_relative_energy_kcal_mol: Optional[float] = None,
) -> str:
    """Compact GOAT conformer-search summary for markdown reports."""
    return _render_basic_goat_section(
        goat,
        format_number=_f,
        make_table=_table,
        max_relative_energy_kcal_mol=max_relative_energy_kcal_mol,
    )


def _render_geom_opt_section(geom_opt: Dict[str, Any]) -> str:
    """Compact geometry-optimization summary for markdown reports."""
    return _render_basic_geom_opt_section(
        geom_opt,
        format_number=_f,
        make_table=_table,
    )


def _render_epr_section(epr: Dict[str, Any], heading_level: int = 3) -> str:
    """Compact EPR summary for markdown reports."""
    return _render_spectroscopy_epr_section(
        epr,
        heading_level,
        format_number=_f,
        make_table=_table,
        render_matrix=_render_matrix,
    )


def _render_solvation_section(solvation: Dict[str, Any]) -> str:
    """Compact implicit-solvation summary for markdown reports."""
    return _render_basic_solvation_section(
        solvation,
        format_number=_f,
        make_table=_table,
    )


def _mol_label(data: Dict[str, Any]) -> str:
    """Short identifier for a molecule/calculation."""
    name = _get_job_name(data) or Path(data.get("source_file", "mol")).stem
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


def _compact_irrep_counts(data: Dict[str, Any], spin: str = "") -> str:
    """Compact formatter for occupied orbital counts per irrep."""
    oe = _get_final_orbital_energies(data)
    sym = _get_symmetry_data(data)
    spin_key = _normalize_spin_key(spin)
    if spin_key == "a":
        mapping = oe.get("alpha_occupied_per_irrep") or {}
    elif spin_key == "b":
        mapping = oe.get("beta_occupied_per_irrep") or {}
    else:
        mapping = oe.get("occupied_per_irrep") or {}

    if not mapping:
        return "—"

    order: List[str] = [
        entry.get("label", "")
        for entry in sym.get("irreps", [])
        if entry.get("label")
    ]
    for label in mapping:
        if label not in order:
            order.append(label)
    return ", ".join(f"{label}={mapping[label]}" for label in order if label in mapping) or "—"


def _irrep_group(data: Dict[str, Any]) -> str:
    """Try to extract the point group label from the metadata or orbital data."""
    sym_label = _symmetry_inline_label(data)
    if sym_label:
        return sym_label
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


def _render_symmetry_section(data: Dict[str, Any]) -> str:
    """Render a dedicated symmetry summary for UseSym / irrep-aware jobs."""
    return _render_state_symmetry_section(
        data,
        has_symmetry=_has_symmetry,
        get_symmetry_data=_get_symmetry_data,
        symmetry_on_off=_symmetry_on_off,
        has_symmetry_setup=_has_symmetry_setup,
        yes_no_unknown=_yes_no_unknown,
        format_number=_f,
        make_table=_table,
    )


def _render_deltascf_section(data: Dict[str, Any]) -> str:
    """Render a dedicated DeltaSCF section for excited-state SCF jobs."""
    return _render_state_deltascf_section(
        data,
        get_deltascf_data=_get_deltascf_data,
        deltascf_target_summary=_deltascf_target_summary,
        format_deltascf_vector=_format_deltascf_vector,
        yes_no_unknown=_yes_no_unknown,
        make_table=_table,
    )


def _render_excited_state_opt_section(data: Dict[str, Any]) -> str:
    """Render CIS/TDDFT excited-state geometry-optimization metadata."""
    return _render_state_excited_state_opt_section(
        data,
        get_excited_state_opt_data=_get_excited_state_opt_data,
        excited_state_target_label=_excited_state_target_label,
        yes_no_unknown=_yes_no_unknown,
        format_number=_f,
        make_table=_table,
    )


def _normalize_spin_key(spin: Any) -> str:
    """Normalize ORCA spin labels to a/b/'' keys."""
    if not spin:
        return ""
    text = str(spin).strip().lower()
    if text.startswith("a"):
        return "a"
    if text.startswith("b"):
        return "b"
    return ""


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


def _render_matrix(matrix, value_fmt=".6f") -> str:
    """Render a 3x3 matrix as a markdown table."""
    rows = [("", "x", "y", "z")]
    labels = ("x", "y", "z")
    for label, values in zip(labels, matrix):
        rows.append((label,) + tuple(_f(value, value_fmt) for value in values[:3]))
    return _table(rows)


