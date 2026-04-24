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
from ..job_family_registry import (
    get_calculation_family_plugin as _get_calculation_family_plugin,
    iter_active_comparison_family_plugins as _iter_active_comparison_family_plugins,
)
from .job_state import (
    electronic_state_label as _electronic_state_label,
    get_basis_set as _get_basis_set,
    get_charge as _get_charge,
    get_excited_state_opt_data as _get_excited_state_opt_data,
    get_method_header_label as _get_method_header_label,
    get_method_table_label as _get_method_table_label,
    get_job_id as _get_job_id,
    get_job_name as _get_job_name,
    get_multiplicity as _get_multiplicity,
    get_symmetry_data as _get_symmetry_data,
    has_symmetry as _has_symmetry,
    has_symmetry_setup as _has_symmetry_setup,
    symmetry_inline_label as _symmetry_inline_label,
    symmetry_on_off as _symmetry_on_off,
    yes_no_unknown as _yes_no_unknown,
)
from ..render_options import (
    RENDER_OPTION_UNSET,
    RenderOptions,
    build_render_options as _build_render_options,
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
    render_solvation_section as _render_basic_solvation_section,
)
from .markdown_sections_spectroscopy import (
    epr_g_iso as _epr_g_iso,
    epr_g_principal_values as _epr_g_principal_values,
    epr_top_hyperfine as _epr_top_hyperfine,
    render_epr_section as _render_spectroscopy_epr_section,
    render_tddft_section as _render_spectroscopy_tddft_section,
)
from .markdown_sections_state import render_symmetry_section as _render_state_symmetry_section
from .markdown_section_registry import (
    MarkdownRenderHelpers as _MarkdownRenderHelpers,
    iter_comparison_markdown_section_plugins as _iter_comparison_markdown_section_plugins,
    iter_molecule_markdown_section_plugins as _iter_molecule_markdown_section_plugins,
)


AU_TO_DEBYE = 2.541746473


# ─────────────────────────────────────────────────────────────────────────────
# Public entry points
# ─────────────────────────────────────────────────────────────────────────────

def write_markdown(
    data: Dict[str, Any],
    path: Path,
    *,
    goat_max_relative_energy_kcal_mol=RENDER_OPTION_UNSET,
    detail_scope: str = "auto",
) -> Path:
    """Write a single-molecule markdown report."""
    render_options = _build_render_options(
        comparison=False,
        detail_scope=detail_scope,
        goat_max_relative_energy_kcal_mol=goat_max_relative_energy_kcal_mol,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _render_molecule_registry(
            data,
            render_options=render_options,
        ),
        encoding="utf-8",
    )
    return path


def write_comparison(
    datasets: List[Dict[str, Any]],
    path: Path,
    *,
    goat_max_relative_energy_kcal_mol=RENDER_OPTION_UNSET,
    detail_scope: str = "auto",
) -> Path:
    """Write a multi-molecule comparison markdown document."""
    render_options = _build_render_options(
        comparison=True,
        detail_scope=detail_scope,
        goat_max_relative_energy_kcal_mol=goat_max_relative_energy_kcal_mol,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _render_comparison_registry(
            datasets,
            render_options=render_options,
        ),
        encoding="utf-8",
    )
    return path


def _markdown_render_helpers(heading_level: int) -> _MarkdownRenderHelpers:
    """Bundle shared formatting helpers for registry-driven markdown sections."""
    return _MarkdownRenderHelpers(
        heading_level=heading_level,
        format_number=_f,
        make_table=_table,
        render_matrix=_render_matrix,
        get_dipole_vector=_get_dipole_vector,
        format_vector=_format_vector,
        format_orbital_energy_with_irrep=_format_orbital_energy_with_irrep,
        compact_irrep_counts=_compact_irrep_counts,
        render_irrep_orbital_window=_render_irrep_orbital_window,
    )


def _render_molecule_registry(
    data: Dict[str, Any],
    heading_level: int = 1,
    display_label: Optional[str] = None,
    source_display: Optional[str] = None,
    render_options: Optional[RenderOptions] = None,
) -> str:
    """Registry-driven single-molecule markdown report."""
    render_options = render_options or _build_render_options(comparison=False)
    H = "#" * heading_level
    H2 = "#" * (heading_level + 1)

    blocks: List[str] = []

    scf = data.get("scf", {})
    src_path = Path(data.get("source_file", "unknown"))
    src = source_display or src_path.name

    job = display_label or _get_job_name(data) or src_path.stem
    charge = _get_charge(data)
    mult = _get_multiplicity(data)
    state_label = _electronic_state_label(data)
    sym_label = _symmetry_inline_label(data)
    state = f"  state={state_label}" if state_label else ""
    sym = f"  symmetry={sym_label}" if sym_label else ""
    job_id = _get_job_id(data)
    blocks.append(
        f"{H} {job}\n"
        f"`{_get_method_header_label(data)}` | charge={charge} mult={mult}{state}{sym}  \n"
        f"source: `{src}` | id: `{job_id}`"
    )

    scf_lines = []
    energy = scf.get("final_single_point_energy_Eh")
    if energy is not None:
        scf_lines.append(f"**Energy:** {energy:.10f} Eh")
    if "dispersion_correction_Eh" in scf:
        scf_lines.append(f"**Dispersion (D):** {scf['dispersion_correction_Eh']:.8f} Eh")
    if "s_squared" in scf:
        s_squared = scf["s_squared"]
        ideal_s_squared = scf.get("s_squared_ideal", "?")
        contamination = (
            abs(s_squared - float(ideal_s_squared))
            if isinstance(ideal_s_squared, (int, float))
            else None
        )
        warning = "  âš ï¸ *contamination > 0.01*" if contamination and contamination > 0.01 else ""
        scf_lines.append(f"**âŸ¨SÂ²âŸ©:** {s_squared:.6f} (ideal {ideal_s_squared}){warning}")

    dipole = _get_final_dipole(data)
    if dipole.get("magnitude_Debye") is not None:
        vector = _get_dipole_vector(dipole, "total_dipole", "Debye")
        xyz = ""
        if vector:
            xyz = (
                f"  ({vector.get('x', 0.0):.4f}, {vector.get('y', 0.0):.4f}, "
                f"{vector.get('z', 0.0):.4f}) D"
            )
        scf_lines.append(f"**Dipole:** {dipole['magnitude_Debye']:.4f} D{xyz}")

    if scf_lines:
        blocks.append("\n".join(scf_lines))

    helpers = _markdown_render_helpers(heading_level)
    # Common standalone sections now come from the registry rather than a
    # giant writer-local branch chain. The writer keeps only the report shell.
    for plugin in _iter_molecule_markdown_section_plugins():
        if plugin.order >= 50:
            continue
        blocks.extend(plugin.render_molecule_blocks(data, helpers, render_options))

    family_plugin = _get_calculation_family_plugin(data)
    if family_plugin.render_markdown_sections is not None:
        for heading, body in family_plugin.render_markdown_sections(
            data,
            _f,
            _table,
            render_options,
        ):
            if body:
                blocks.append(f"{H2} {heading}\n{body}")

    for plugin in _iter_molecule_markdown_section_plugins():
        if plugin.order < 50:
            continue
        blocks.extend(plugin.render_molecule_blocks(data, helpers, render_options))

    return "\n\n".join(blocks) + "\n"


def _render_comparison_registry(
    datasets: List[Dict[str, Any]],
    *,
    render_options: Optional[RenderOptions] = None,
) -> str:
    """Registry-driven comparison markdown report."""
    render_options = render_options or _build_render_options(comparison=True)
    if not datasets:
        return "# ORCA Comparison\n\n*No data provided.*\n"

    blocks: List[str] = ["# ORCA Calculation Comparison"]
    labels = _comparison_labels(datasets)
    helpers = _markdown_render_helpers(1)

    comparison_renderers = []
    # Comparison sections share the same registry idea. Family-specific hooks
    # and common sections now meet in one ordered render queue.
    for plugin in _iter_comparison_markdown_section_plugins():
        comparison_renderers.append(
            (
                plugin.order,
                plugin.key,
                lambda section_plugin=plugin: section_plugin.render_comparison_blocks(
                    datasets,
                    labels,
                    helpers,
                    render_options,
                ),
            )
        )

    for plugin in _iter_active_comparison_family_plugins(datasets):
        comparison_renderers.append(
            (
                plugin.comparison_order,
                plugin.family,
                lambda family_plugin=plugin: [
                    (heading, body)
                    for heading, body in family_plugin.render_comparison_sections(
                        datasets,
                        labels,
                        _f,
                        _table,
                        render_options,
                    )
                    if body
                ],
            )
        )

    for _order, _key, render_blocks in sorted(
        comparison_renderers,
        key=lambda item: (item[0], item[1]),
    ):
        for block in render_blocks():
            if isinstance(block, tuple):
                heading, body = block
                blocks.append(f"## {heading}\n{body}")
            else:
                blocks.append(block)

    blocks.append("---\n# Individual Reports")
    for label, dataset in zip(labels, datasets):
        blocks.append(
            _render_molecule_registry(
                dataset,
                heading_level=2,
                display_label=label,
                source_display=label,
                render_options=render_options,
            )
        )

    return "\n\n".join(blocks) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Single-molecule renderer
# ─────────────────────────────────────────────────────────────────────────────

def _render_molecule(
    data: Dict[str, Any],
    heading_level: int = 1,
    display_label: Optional[str] = None,
    source_display: Optional[str] = None,
    render_options: Optional[RenderOptions] = None,
) -> str:
    """Compatibility shim for the registry-driven standalone renderer."""
    # Keep this name for older internal callers, but route immediately to the
    # registry-backed implementation so there is only one live render path.
    return _render_molecule_registry(
        data,
        heading_level=heading_level,
        display_label=display_label,
        source_display=source_display,
        render_options=render_options,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Multi-molecule comparison renderer
# ─────────────────────────────────────────────────────────────────────────────

def _render_comparison(
    datasets: List[Dict[str, Any]],
    *,
    render_options: Optional[RenderOptions] = None,
) -> str:
    """Compatibility shim for the registry-driven comparison renderer."""
    return _render_comparison_registry(
        datasets,
        render_options=render_options,
    )


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


def _render_orbital_window_table(
    orbitals: List[Dict[str, Any]],
    window: Optional[int] = 12,
) -> str:
    """Render a frontier window or the full irrep-resolved orbital list."""
    if not orbitals or not any(orbital.get("irrep") for orbital in orbitals):
        return ""

    occupied = [orbital for orbital in orbitals if float(orbital.get("occupation", 0.0) or 0.0) > 1e-8]
    virtual = [orbital for orbital in orbitals if float(orbital.get("occupation", 0.0) or 0.0) <= 1e-8]
    if not occupied and not virtual:
        return ""

    rows = [("Region", "NO", "occ", "E (Eh)", "E (eV)", "irrep")]

    occupied_window = occupied if window is None else occupied[max(0, len(occupied) - window):]
    start_occ = 0 if window is None else max(0, len(occupied) - window)
    for idx, orbital in enumerate(occupied_window, start=start_occ):
        offset = len(occupied) - 1 - idx
        rows.append((
            _frontier_region_label(offset, occupied=True),
            str(orbital.get("index", "")),
            _f(orbital.get("occupation")),
            _f(orbital.get("energy_Eh"), ".6f"),
            _f(orbital.get("energy_eV")),
            str(orbital.get("irrep", "") or "—"),
        ))

    virtual_window = virtual if window is None else virtual[:window]
    for offset, orbital in enumerate(virtual_window):
        rows.append((
            _frontier_region_label(offset, occupied=False),
            str(orbital.get("index", "")),
            _f(orbital.get("occupation")),
            _f(orbital.get("energy_Eh"), ".6f"),
            _f(orbital.get("energy_eV")),
            str(orbital.get("irrep", "") or "—"),
        ))

    return _table(rows)


def _render_irrep_orbital_window(
    data: Dict[str, Any],
    window: Optional[int] = 12,
) -> str:
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


def _render_epr_section(
    epr: Dict[str, Any],
    heading_level: int = 3,
    *,
    render_options: Optional[RenderOptions] = None,
) -> str:
    """Compact EPR summary for markdown reports."""
    render_options = render_options or _build_render_options(comparison=False)
    return _render_spectroscopy_epr_section(
        epr,
        heading_level,
        format_number=_f,
        make_table=_table,
        render_matrix=_render_matrix,
        top_hyperfine_nuclei=render_options.epr_top_hyperfine_nuclei,
        top_atom_contributions=render_options.epr_top_atom_contributions,
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


