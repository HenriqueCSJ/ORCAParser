"""Registry for common markdown output sections.

This is the markdown-side equivalent of the parser-section registry. The goal
is not to replace every renderer with a brand-new abstraction, but to stop the
top-level writer from owning a giant sequence of hard-coded section branches.

Each plugin returns ready-to-append markdown blocks. That keeps nested section
structures such as NBO analysis intact while letting the writer discover the
available sections by registration order instead of by a hand-written chain of
``if`` statements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from ..final_snapshot import (
    get_final_dipole as _get_final_dipole,
    get_final_geometry as _get_final_geometry,
    get_final_orbital_energies as _get_final_orbital_energies,
)
from ..render_options import RenderOptions
from .job_state import (
    electronic_state_label as _electronic_state_label,
    get_basis_set as _get_basis_set,
    get_charge as _get_charge,
    get_excited_state_opt_data as _get_excited_state_opt_data,
    get_multiplicity as _get_multiplicity,
    get_symmetry_data as _get_symmetry_data,
    has_symmetry as _has_symmetry,
    has_symmetry_setup as _has_symmetry_setup,
    symmetry_inline_label as _symmetry_inline_label,
    symmetry_on_off as _symmetry_on_off,
    yes_no_unknown as _yes_no_unknown,
)
from .markdown_sections_analysis import (
    build_cmo_lookup as _build_cmo_lookup,
    build_orbital_irrep_lookup as _build_orbital_irrep_lookup,
    format_transition_cmo_character as _format_transition_cmo_character,
    format_transition_with_irreps as _format_transition_with_irreps,
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


FormatNumber = Callable[[Any, str], str]
MakeTable = Callable[[List[tuple]], str]
RenderMatrix = Callable[[Sequence[Sequence[Any]], str], str]
GetDipoleVector = Callable[[Dict[str, Any], str, str], Optional[Dict[str, float]]]
FormatVector = Callable[[Optional[Dict[str, Any]], str], str]
FormatOrbitalEnergyWithIrrep = Callable[[Any, Any], str]
CompactIrrepCounts = Callable[[Dict[str, Any], str], str]
RenderIrrepOrbitalWindow = Callable[[Dict[str, Any], Optional[int]], str]


@dataclass(frozen=True)
class MarkdownRenderHelpers:
    """Helper callbacks shared by common markdown section plugins."""

    heading_level: int
    format_number: FormatNumber
    make_table: MakeTable
    render_matrix: RenderMatrix
    get_dipole_vector: GetDipoleVector
    format_vector: FormatVector
    format_orbital_energy_with_irrep: FormatOrbitalEnergyWithIrrep
    compact_irrep_counts: CompactIrrepCounts
    render_irrep_orbital_window: RenderIrrepOrbitalWindow


MarkdownMoleculeRenderer = Callable[
    [Dict[str, Any], MarkdownRenderHelpers, RenderOptions],
    List[str],
]
MarkdownComparisonRenderer = Callable[
    [List[Dict[str, Any]], List[str], MarkdownRenderHelpers, RenderOptions],
    List[str],
]


@dataclass(frozen=True)
class MarkdownSectionPlugin:
    """Plugin-like markdown section definition."""

    key: str
    order: int = 50
    render_molecule_blocks: Optional[MarkdownMoleculeRenderer] = None
    render_comparison_blocks: Optional[MarkdownComparisonRenderer] = None


_MARKDOWN_SECTION_PLUGINS: List[MarkdownSectionPlugin] = []


def register_markdown_section_plugin(
    plugin: MarkdownSectionPlugin,
    *,
    replace: bool = False,
) -> None:
    """Register a markdown section plugin."""

    global _MARKDOWN_SECTION_PLUGINS

    if replace:
        _MARKDOWN_SECTION_PLUGINS = [
            existing
            for existing in _MARKDOWN_SECTION_PLUGINS
            if existing.key != plugin.key
        ]
    elif any(existing.key == plugin.key for existing in _MARKDOWN_SECTION_PLUGINS):
        raise ValueError(f"Markdown section already registered: {plugin.key}")

    _MARKDOWN_SECTION_PLUGINS.append(plugin)


def get_registered_markdown_section_plugins() -> tuple[MarkdownSectionPlugin, ...]:
    """Return registered markdown section plugins in render order."""

    return tuple(sorted(_MARKDOWN_SECTION_PLUGINS, key=lambda plugin: (plugin.order, plugin.key)))


def iter_molecule_markdown_section_plugins() -> tuple[MarkdownSectionPlugin, ...]:
    """Return plugins that render standalone report sections."""

    return tuple(
        plugin
        for plugin in get_registered_markdown_section_plugins()
        if plugin.render_molecule_blocks is not None
    )


def iter_comparison_markdown_section_plugins() -> tuple[MarkdownSectionPlugin, ...]:
    """Return plugins that render comparison report sections."""

    return tuple(
        plugin
        for plugin in get_registered_markdown_section_plugins()
        if plugin.render_comparison_blocks is not None
    )


def _molecule_dipole_blocks(
    data: Dict[str, Any],
    helpers: MarkdownRenderHelpers,
    render_options: RenderOptions,
) -> List[str]:
    del render_options
    dipole = _get_final_dipole(data)
    section = _render_basic_dipole_section(
        dipole,
        format_number=helpers.format_number,
        make_table=helpers.make_table,
        get_dipole_vector=helpers.get_dipole_vector,
        format_vector=helpers.format_vector,
    )
    if not section:
        return []
    h2 = "#" * (helpers.heading_level + 1)
    return [f"{h2} Dipole Moment\n{section}"]


def _molecule_basis_set_blocks(
    data: Dict[str, Any],
    helpers: MarkdownRenderHelpers,
    render_options: RenderOptions,
) -> List[str]:
    del render_options
    basis_set = data.get("basis_set")
    if not basis_set:
        return []
    section = _render_basic_basis_set_section(
        basis_set,
        make_table=helpers.make_table,
    )
    if not section:
        return []
    h2 = "#" * (helpers.heading_level + 1)
    return [f"{h2} Basis Set\n{section}"]


def _molecule_symmetry_blocks(
    data: Dict[str, Any],
    helpers: MarkdownRenderHelpers,
    render_options: RenderOptions,
) -> List[str]:
    del render_options
    section = _render_state_symmetry_section(
        data,
        has_symmetry=_has_symmetry,
        get_symmetry_data=_get_symmetry_data,
        symmetry_on_off=_symmetry_on_off,
        has_symmetry_setup=_has_symmetry_setup,
        yes_no_unknown=_yes_no_unknown,
        format_number=helpers.format_number,
        make_table=helpers.make_table,
    )
    if not section:
        return []
    h2 = "#" * (helpers.heading_level + 1)
    return [f"{h2} Symmetry\n{section}"]


def _molecule_solvation_blocks(
    data: Dict[str, Any],
    helpers: MarkdownRenderHelpers,
    render_options: RenderOptions,
) -> List[str]:
    del render_options
    solvation = data.get("solvation")
    if not solvation:
        return []
    section = _render_basic_solvation_section(
        solvation,
        format_number=helpers.format_number,
        make_table=helpers.make_table,
    )
    if not section:
        return []
    h2 = "#" * (helpers.heading_level + 1)
    return [f"{h2} Solvation\n{section}"]


def _molecule_tddft_blocks(
    data: Dict[str, Any],
    helpers: MarkdownRenderHelpers,
    render_options: RenderOptions,
) -> List[str]:
    del render_options
    tddft = data.get("tddft")
    if not tddft:
        return []
    section = _render_spectroscopy_tddft_section(
        tddft,
        data,
        format_number=helpers.format_number,
        make_table=helpers.make_table,
        build_orbital_irrep_lookup=_build_orbital_irrep_lookup,
        build_cmo_lookup=_build_cmo_lookup,
        format_transition_with_irreps=_format_transition_with_irreps,
        format_transition_cmo_character=_format_transition_cmo_character,
        yes_no_unknown=_yes_no_unknown,
        get_excited_state_opt_data=_get_excited_state_opt_data,
    )
    if not section:
        return []
    h2 = "#" * (helpers.heading_level + 1)
    return [f"{h2} TDDFT Excited States\n{section}"]


def _molecule_frontier_orbital_blocks(
    data: Dict[str, Any],
    helpers: MarkdownRenderHelpers,
    render_options: RenderOptions,
) -> List[str]:
    oe = _get_final_orbital_energies(data)
    if not oe:
        return []
    ctx = data.get("context", {})
    h2 = "#" * (helpers.heading_level + 1)
    blocks = [f"{h2} Frontier Orbital Energies"]
    if not ctx.get("is_uhf"):
        homo = oe.get("HOMO_energy_eV")
        lumo = oe.get("LUMO_energy_eV")
        gap = oe.get("HOMO_LUMO_gap_eV")
        hirr = oe.get("HOMO_irrep", "")
        lirr = oe.get("LUMO_irrep", "")
        rows = [
            ("", "eV", "irrep"),
            ("HOMO", f"{homo:.4f}" if homo else "—", hirr),
            ("LUMO", f"{lumo:.4f}" if lumo else "—", lirr),
            ("Gap", f"{gap:.4f}" if gap else "—", ""),
        ]
        if not hirr:
            rows = [(row[0], row[1]) for row in rows]
        blocks.append(helpers.make_table(rows))
    else:
        ah = oe.get("alpha_HOMO_energy_eV")
        al = oe.get("alpha_LUMO_energy_eV")
        bh = oe.get("beta_HOMO_energy_eV")
        bl = oe.get("beta_LUMO_energy_eV")
        ag = oe.get("alpha_HOMO_LUMO_gap_eV")
        bg = oe.get("beta_HOMO_LUMO_gap_eV")
        ahi = oe.get("alpha_HOMO_irrep", "")
        ali = oe.get("alpha_LUMO_irrep", "")
        bhi = oe.get("beta_HOMO_irrep", "")
        bli = oe.get("beta_LUMO_irrep", "")
        has_irr = bool(ahi or ali)
        if has_irr:
            rows = [
                ("", "α eV", "α irrep", "β eV", "β irrep"),
                ("HOMO", helpers.format_number(ah), ahi, helpers.format_number(bh), bhi),
                ("LUMO", helpers.format_number(al), ali, helpers.format_number(bl), bli),
                ("Gap", helpers.format_number(ag), "", helpers.format_number(bg), ""),
            ]
        else:
            rows = [
                ("", "α eV", "β eV"),
                ("HOMO", helpers.format_number(ah), helpers.format_number(bh)),
                ("LUMO", helpers.format_number(al), helpers.format_number(bl)),
                ("Gap", helpers.format_number(ag), helpers.format_number(bg)),
            ]
        blocks.append(helpers.make_table(rows))

    occ = oe.get("occupied_per_irrep") or oe.get("alpha_occupied_per_irrep")
    if occ:
        beta_occ = oe.get("beta_occupied_per_irrep")
        if beta_occ:
            irreps = sorted(set(list(occ) + list(beta_occ)))
            rows = [
                ("Irrep",) + tuple(irreps),
                ("α occ",) + tuple(occ.get(irrep, 0) for irrep in irreps),
                ("β occ",) + tuple(beta_occ.get(irrep, 0) for irrep in irreps),
            ]
        else:
            irreps = sorted(occ)
            rows = [
                ("Irrep",) + tuple(irreps),
                ("occ",) + tuple(occ.get(irrep, 0) for irrep in irreps),
            ]
        blocks.append(helpers.make_table(rows))

    orbital_window = helpers.render_irrep_orbital_window(
        data,
        render_options.orbital_window,
    )
    if orbital_window:
        blocks.append(f"{h2} Irrep-Resolved Orbital Window\n{orbital_window}")
    return blocks


def _molecule_qro_blocks(
    data: Dict[str, Any],
    helpers: MarkdownRenderHelpers,
    render_options: RenderOptions,
) -> List[str]:
    del render_options
    qro = data.get("qro")
    if not qro:
        return []
    h2 = "#" * (helpers.heading_level + 1)
    blocks = [
        f"{h2} Quasi-Restricted Orbitals",
        (
            f"DOMO = {qro.get('n_domo')}  |  "
            f"SOMO = {qro.get('n_somo')}  |  "
            f"VMO = {qro.get('n_vmo')}"
        ),
    ]
    somos = qro.get("somo_details", [])
    if somos:
        has_irr = any("irrep" in somo for somo in somos)
        if has_irr:
            rows = [("MO", "irrep", "ε (Eh)", "ε (eV)", "α (eV)", "β (eV)")]
            rows.extend(
                (
                    str(somo["index"]),
                    somo.get("irrep", ""),
                    f"{somo['energy_Eh']:.6f}",
                    f"{somo['energy_eV']:.4f}",
                    f"{somo['alpha_energy_eV']:.4f}",
                    f"{somo['beta_energy_eV']:.4f}",
                )
                for somo in somos
            )
        else:
            rows = [("MO", "ε (Eh)", "ε (eV)", "α (eV)", "β (eV)")]
            rows.extend(
                (
                    str(somo["index"]),
                    f"{somo['energy_Eh']:.6f}",
                    f"{somo['energy_eV']:.4f}",
                    f"{somo['alpha_energy_eV']:.4f}",
                    f"{somo['beta_energy_eV']:.4f}",
                )
                for somo in somos
            )
        blocks.append(helpers.make_table(rows))
    return blocks


def _molecule_epr_blocks(
    data: Dict[str, Any],
    helpers: MarkdownRenderHelpers,
    render_options: RenderOptions,
) -> List[str]:
    epr = data.get("epr")
    if not epr:
        return []
    section = _render_spectroscopy_epr_section(
        epr,
        helpers.heading_level + 2,
        format_number=helpers.format_number,
        make_table=helpers.make_table,
        render_matrix=helpers.render_matrix,
        top_hyperfine_nuclei=render_options.epr_top_hyperfine_nuclei,
        top_atom_contributions=render_options.epr_top_atom_contributions,
    )
    if not section:
        return []
    h2 = "#" * (helpers.heading_level + 1)
    return [f"{h2} EPR / Magnetic Properties\n{section}"]


def _molecule_analysis_blocks(
    data: Dict[str, Any],
    helpers: MarkdownRenderHelpers,
    render_options: RenderOptions,
) -> List[str]:
    return _render_analysis_sections(
        data,
        context=data.get("context", {}),
        heading_level=helpers.heading_level,
        format_number=helpers.format_number,
        make_table=helpers.make_table,
        render_options=render_options,
    )


def _molecule_geometry_blocks(
    data: Dict[str, Any],
    helpers: MarkdownRenderHelpers,
    render_options: RenderOptions,
) -> List[str]:
    del render_options
    geom = _get_final_geometry(data)
    atoms = geom.get("symmetry_cartesian_angstrom") or geom.get("cartesian_angstrom", [])
    if not atoms:
        return []
    h2 = "#" * (helpers.heading_level + 1)
    heading = (
        "Geometry (symmetry-perfected, Å)"
        if geom.get("symmetry_cartesian_angstrom")
        else "Geometry (Å)"
    )
    rows = [("Atom", "x", "y", "z")]
    for index, atom in enumerate(atoms):
        rows.append(
            (
                f"{atom.get('symbol', '?')}{index + 1}",
                f"{atom['x_ang']:.6f}",
                f"{atom['y_ang']:.6f}",
                f"{atom['z_ang']:.6f}",
            )
        )
    return [f"{h2} {heading}\n{helpers.make_table(rows)}"]


def _comparison_methods_blocks(
    datasets: List[Dict[str, Any]],
    labels: List[str],
    helpers: MarkdownRenderHelpers,
    render_options: RenderOptions,
) -> List[str]:
    del render_options
    rows = [("", "method", "basis", "charge", "mult", "electronic state", "symmetry")]
    for label, dataset in zip(labels, datasets):
        rows.append(
            (
                label,
                dataset.get("job_snapshot", {}).get("method_table_label")
                or dataset.get("metadata", {}).get("method")
                or "—",
                _get_basis_set(dataset) or "—",
                str(_get_charge(dataset) if _get_charge(dataset) != "" else "?"),
                str(_get_multiplicity(dataset) if _get_multiplicity(dataset) != "" else "?"),
                _electronic_state_label(dataset) or "ground-state",
                _symmetry_inline_label(dataset) or "C1",
            )
        )
    return [f"## Methods\n{helpers.make_table(rows)}"]


def _comparison_symmetry_blocks(
    datasets: List[Dict[str, Any]],
    labels: List[str],
    helpers: MarkdownRenderHelpers,
    render_options: RenderOptions,
) -> List[str]:
    del render_options
    blocks: List[str] = []
    if not any(_has_symmetry(dataset) for dataset in datasets):
        return blocks

    rows = [("", "UseSym", "point group", "reduced", "orbital irreps", "n_irreps", "initial guess")]
    for label, dataset in zip(labels, datasets):
        symmetry = _get_symmetry_data(dataset)
        rows.append(
            (
                label,
                _symmetry_on_off(symmetry),
                symmetry.get("point_group") or symmetry.get("auto_detected_point_group") or "?",
                symmetry.get("reduced_point_group") or "?",
                symmetry.get("orbital_irrep_group") or "?",
                str(symmetry.get("n_irreps", "?")),
                symmetry.get("initial_guess_irrep") or "?",
            )
        )
    blocks.append(f"## Symmetry\n{helpers.make_table(rows)}")

    if any(_has_symmetry_setup(_get_symmetry_data(dataset)) for dataset in datasets):
        rows = [("", "guess", "MO file", "geom match", "basis match", "reassign irreps", "renorm", "reorth")]
        for label, dataset in zip(labels, datasets):
            symmetry = _get_symmetry_data(dataset)
            reorth = _yes_no_unknown(symmetry.get("initial_guess_mos_reorthogonalized"))
            if reorth == "yes" and symmetry.get("initial_guess_reorthogonalization_method"):
                reorth = f"{reorth} ({symmetry['initial_guess_reorthogonalization_method']})"
            rows.append(
                (
                    label,
                    symmetry.get("initial_guess_method") or "—",
                    symmetry.get("initial_guess_source_file") or "—",
                    _yes_no_unknown(symmetry.get("initial_guess_geometry_matches")) if "initial_guess_geometry_matches" in symmetry else "—",
                    _yes_no_unknown(symmetry.get("initial_guess_basis_matches")) if "initial_guess_basis_matches" in symmetry else "—",
                    _yes_no_unknown(symmetry.get("initial_guess_irreps_reassigned")) if "initial_guess_irreps_reassigned" in symmetry else "—",
                    _yes_no_unknown(symmetry.get("initial_guess_mos_renormalized")) if "initial_guess_mos_renormalized" in symmetry else "—",
                    reorth if "initial_guess_mos_reorthogonalized" in symmetry else "—",
                )
            )
        blocks.append(f"## Symmetry Setup\n{helpers.make_table(rows)}")

    if any(
        _get_final_orbital_energies(dataset).get("alpha_occupied_per_irrep")
        or _get_final_orbital_energies(dataset).get("beta_occupied_per_irrep")
        or _get_final_orbital_energies(dataset).get("occupied_per_irrep")
        for dataset in datasets
    ):
        any_uhf = any(dataset.get("context", {}).get("is_uhf") for dataset in datasets)
        if any_uhf:
            rows = [("", "alpha occupied per irrep", "beta occupied per irrep")]
            for label, dataset in zip(labels, datasets):
                rows.append(
                    (
                        label,
                        helpers.compact_irrep_counts(dataset, "a"),
                        helpers.compact_irrep_counts(dataset, "b"),
                    )
                )
        else:
            rows = [("", "occupied per irrep")]
            for label, dataset in zip(labels, datasets):
                rows.append((label, helpers.compact_irrep_counts(dataset, "")))
        blocks.append(f"## Symmetry Occupations\n{helpers.make_table(rows)}")

    return blocks


def _comparison_energy_blocks(
    datasets: List[Dict[str, Any]],
    labels: List[str],
    helpers: MarkdownRenderHelpers,
    render_options: RenderOptions,
) -> List[str]:
    del render_options
    rows = [("", "E (Eh)", "⟨S²⟩", "ideal", "dipole (D)")]
    has_s2 = any(dataset.get("scf", {}).get("s_squared") is not None for dataset in datasets)
    if not has_s2:
        rows = [("", "E (Eh)", "dipole (D)")]

    for label, dataset in zip(labels, datasets):
        scf = dataset.get("scf", {})
        dipole = _get_final_dipole(dataset)
        energy = scf.get("final_single_point_energy_Eh")
        s_squared = scf.get("s_squared")
        ideal = scf.get("s_squared_ideal", "")
        magnitude = dipole.get("magnitude_Debye")
        if has_s2:
            rows.append(
                (
                    label,
                    f"{energy:.10f}" if energy else "—",
                    f"{s_squared:.6f}" if s_squared is not None else "—",
                    str(ideal),
                    f"{magnitude:.4f}" if magnitude is not None else "—",
                )
            )
        else:
            rows.append(
                (
                    label,
                    f"{energy:.10f}" if energy else "—",
                    f"{magnitude:.4f}" if magnitude is not None else "—",
                )
            )

    return [f"## Energies\n{helpers.make_table(rows)}"]


def _comparison_frontier_orbital_blocks(
    datasets: List[Dict[str, Any]],
    labels: List[str],
    helpers: MarkdownRenderHelpers,
    render_options: RenderOptions,
) -> List[str]:
    del render_options
    orbital_blocks: List[str] = []
    for label, dataset in zip(labels, datasets):
        orbitals = _get_final_orbital_energies(dataset)
        ctx = dataset.get("context", {})
        if not orbitals:
            continue
        if not ctx.get("is_uhf"):
            gap = orbitals.get("HOMO_LUMO_gap_eV")
            orbital_blocks.append(
                f"{label}: HOMO "
                f"{helpers.format_orbital_energy_with_irrep(orbitals.get('HOMO_energy_eV'), orbitals.get('HOMO_irrep'))} / "
                f"LUMO "
                f"{helpers.format_orbital_energy_with_irrep(orbitals.get('LUMO_energy_eV'), orbitals.get('LUMO_irrep'))} / "
                f"gap {helpers.format_number(gap)} eV"
            )
        else:
            orbital_blocks.append(
                f"{label}: α-HOMO "
                f"{helpers.format_orbital_energy_with_irrep(orbitals.get('alpha_HOMO_energy_eV'), orbitals.get('alpha_HOMO_irrep'))} / "
                f"β-HOMO "
                f"{helpers.format_orbital_energy_with_irrep(orbitals.get('beta_HOMO_energy_eV'), orbitals.get('beta_HOMO_irrep'))} / "
                f"α-LUMO "
                f"{helpers.format_orbital_energy_with_irrep(orbitals.get('alpha_LUMO_energy_eV'), orbitals.get('alpha_LUMO_irrep'))} / "
                f"β-LUMO "
                f"{helpers.format_orbital_energy_with_irrep(orbitals.get('beta_LUMO_energy_eV'), orbitals.get('beta_LUMO_irrep'))}"
            )
        qro = dataset.get("qro")
        if qro:
            orbital_blocks.append(
                f"  QRO: DOMO={qro.get('n_domo')} SOMO={qro.get('n_somo')} VMO={qro.get('n_vmo')}"
            )
            for somo in qro.get("somo_details", []):
                irrep = f" ({somo['irrep']})" if "irrep" in somo else ""
                orbital_blocks.append(
                    f"  SOMO {somo['index']}{irrep}: "
                    f"α={somo['alpha_energy_eV']:.4f} "
                    f"β={somo['beta_energy_eV']:.4f} eV"
                )
    if not orbital_blocks:
        return []
    return ["## Frontier Orbitals\n" + "\n".join(orbital_blocks)]


def _comparison_epr_blocks(
    datasets: List[Dict[str, Any]],
    labels: List[str],
    helpers: MarkdownRenderHelpers,
    render_options: RenderOptions,
) -> List[str]:
    del render_options
    if not any(dataset.get("epr") for dataset in datasets):
        return []
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
    for label, dataset in zip(labels, datasets):
        epr = dataset.get("epr") or {}
        zfs = epr.get("zero_field_splitting") or {}
        g_vals = _epr_g_principal_values(epr)
        g_iso = _epr_g_iso(epr)
        top_nucleus, top_aiso = _epr_top_hyperfine(epr)
        rows.append(
            (
                label,
                helpers.format_number(zfs.get("D_cm-1"), ".6g"),
                helpers.format_number(zfs.get("E_over_D")),
                helpers.format_number(g_iso, ".7f"),
                helpers.format_number(g_vals[0], ".7f") if len(g_vals) > 0 else "—",
                helpers.format_number(g_vals[1], ".7f") if len(g_vals) > 1 else "—",
                helpers.format_number(g_vals[2], ".7f") if len(g_vals) > 2 else "—",
                helpers.format_number(top_aiso),
                top_nucleus or "—",
            )
        )
    return [f"## EPR / Magnetic\n{helpers.make_table(rows)}"]


def _comparison_charge_blocks(
    datasets: List[Dict[str, Any]],
    labels: List[str],
    helpers: MarkdownRenderHelpers,
    render_options: RenderOptions,
) -> List[str]:
    del render_options
    all_schemes: List[set[str]] = []
    for dataset in datasets:
        schemes: set[str] = set()
        for scheme in ("mulliken", "loewdin", "hirshfeld", "mbis", "chelpg"):
            if _get_charges(dataset, scheme):
                schemes.add(scheme)
        all_schemes.append(schemes)
    common_schemes = sorted(set.intersection(*all_schemes)) if all_schemes else []
    blocks: List[str] = []
    if not common_schemes:
        return blocks

    atom_symbols = datasets[0].get("context", {}).get("atom_symbols", [])
    for scheme in common_schemes:
        atom_count = len(_get_charges(datasets[0], scheme) or [])
        header = ("",) + tuple(
            f"{atom_symbols[index]}{index + 1}" if index < len(atom_symbols) else str(index + 1)
            for index in range(atom_count)
        )
        rows = [header]
        for label, dataset in zip(labels, datasets):
            charges = _get_charges(dataset, scheme) or []
            rows.append((label,) + tuple(f"{charge:.4f}" for charge in charges))
        blocks.append(f"## {scheme.capitalize()} Charges\n{helpers.make_table(rows)}")
    return blocks


register_markdown_section_plugin(
    MarkdownSectionPlugin(
        key="dipole",
        order=10,
        render_molecule_blocks=_molecule_dipole_blocks,
    )
)
register_markdown_section_plugin(
    MarkdownSectionPlugin(
        key="basis_set",
        order=20,
        render_molecule_blocks=_molecule_basis_set_blocks,
    )
)
register_markdown_section_plugin(
    MarkdownSectionPlugin(
        key="symmetry",
        order=30,
        render_molecule_blocks=_molecule_symmetry_blocks,
        render_comparison_blocks=_comparison_symmetry_blocks,
    )
)
register_markdown_section_plugin(
    MarkdownSectionPlugin(
        key="methods",
        order=10,
        render_comparison_blocks=_comparison_methods_blocks,
    )
)
register_markdown_section_plugin(
    MarkdownSectionPlugin(
        key="solvation",
        order=60,
        render_molecule_blocks=_molecule_solvation_blocks,
    )
)
register_markdown_section_plugin(
    MarkdownSectionPlugin(
        key="tddft",
        order=70,
        render_molecule_blocks=_molecule_tddft_blocks,
    )
)
register_markdown_section_plugin(
    MarkdownSectionPlugin(
        key="energies",
        order=60,
        render_comparison_blocks=_comparison_energy_blocks,
    )
)
register_markdown_section_plugin(
    MarkdownSectionPlugin(
        key="frontier_orbitals",
        order=80,
        render_molecule_blocks=_molecule_frontier_orbital_blocks,
        render_comparison_blocks=_comparison_frontier_orbital_blocks,
    )
)
register_markdown_section_plugin(
    MarkdownSectionPlugin(
        key="qro",
        order=90,
        render_molecule_blocks=_molecule_qro_blocks,
    )
)
register_markdown_section_plugin(
    MarkdownSectionPlugin(
        key="epr",
        order=100,
        render_molecule_blocks=_molecule_epr_blocks,
        render_comparison_blocks=_comparison_epr_blocks,
    )
)
register_markdown_section_plugin(
    MarkdownSectionPlugin(
        key="charges_analysis",
        order=90,
        render_comparison_blocks=_comparison_charge_blocks,
    )
)
register_markdown_section_plugin(
    MarkdownSectionPlugin(
        key="analysis",
        order=110,
        render_molecule_blocks=_molecule_analysis_blocks,
    )
)
register_markdown_section_plugin(
    MarkdownSectionPlugin(
        key="geometry",
        order=120,
        render_molecule_blocks=_molecule_geometry_blocks,
    )
)
