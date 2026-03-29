"""Self-contained markdown renderers for spectroscopy-heavy sections."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple


FormatNumber = Callable[..., str]
MakeTable = Callable[[List[tuple]], str]
RenderMatrix = Callable[[Any, str], str]
BuildOrbitalIrrepLookup = Callable[[Dict[str, Any]], Dict[str, Dict[int, str]]]
BuildCMOLookup = Callable[[Dict[str, Any]], Dict[int, Dict[str, Any]]]
FormatTransitionWithIrreps = Callable[[Dict[str, Any], Dict[str, Dict[int, str]]], str]
FormatTransitionCMOCharacter = Callable[[Dict[str, Any], Dict[int, Dict[str, Any]]], str]
YesNoUnknown = Callable[[Any], str]
GetExcitedStateOptData = Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]


def _format_spectrum_transition(transition: Dict[str, Any]) -> str:
    """Format a TDDFT spectrum transition label."""
    from_label = str(transition.get("from_state_label", "") or "").strip()
    to_label = str(transition.get("to_state_label", "") or "").strip()
    if from_label and to_label:
        return f"{from_label} -> {to_label}"
    return from_label or to_label or "—"


def _render_tddft_spectrum_table(
    heading: str,
    table: Dict[str, Any],
    value_label: str,
    value_key: str,
    component_headers: List[Tuple[str, str]],
    *,
    format_number: FormatNumber,
    make_table: MakeTable,
) -> str:
    """Render one parsed TDDFT absorption/CD spectrum table."""
    transitions = table.get("transitions") or []
    if not transitions:
        return ""

    lines = [f"### {heading}"]
    meta_bits = [f"transitions={len(transitions)}"]
    center_of_mass = table.get("center_of_mass")
    if isinstance(center_of_mass, dict):
        coords = ", ".join(
            format_number(center_of_mass.get(axis))
            for axis in ("x", "y", "z")
        )
        meta_bits.append(f"center of mass = ({coords})")
    lines.append(" | ".join(meta_bits))

    rows = [("Transition", "E (eV)", "E (cm^-1)", "lambda (nm)", value_label)]
    rows[0] = rows[0] + tuple(header for header, _ in component_headers)

    for transition in transitions:
        row = [
            _format_spectrum_transition(transition),
            format_number(transition.get("energy_eV"), ".6f"),
            format_number(transition.get("energy_cm1"), ".1f"),
            format_number(transition.get("wavelength_nm"), ".1f"),
            format_number(transition.get(value_key), ".9f"),
        ]
        for _, key in component_headers:
            row.append(format_number(transition.get(key), ".5f"))
        rows.append(tuple(row))

    lines.append(make_table(rows))
    return "\n\n".join(lines)


def _render_tddft_spectra_section(
    tddft: Dict[str, Any],
    *,
    format_number: FormatNumber,
    make_table: MakeTable,
) -> str:
    """Render all available TDDFT/CIS absorption and CD spectra tables."""
    spectra = tddft.get("spectra") or {}
    if not spectra:
        return ""

    sections = []
    layouts = (
        (
            "absorption_electric_dipole",
            "Absorption Spectrum (Electric Dipole)",
            "fosc (D2)",
            "oscillator_strength",
            [("D2 (au^2)", "dipole_strength_au2"), ("DX (au)", "dx_au"), ("DY (au)", "dy_au"), ("DZ (au)", "dz_au")],
        ),
        (
            "absorption_velocity_dipole",
            "Absorption Spectrum (Velocity Dipole)",
            "fosc (P2)",
            "oscillator_strength",
            [("P2 (au^2)", "velocity_strength_au2"), ("PX (au)", "px_au"), ("PY (au)", "py_au"), ("PZ (au)", "pz_au")],
        ),
        (
            "cd_electric_dipole",
            "CD Spectrum (Electric Dipole)",
            "R (1e40*cgs)",
            "rotatory_strength_cgs",
            [("MX (au)", "mx_au"), ("MY (au)", "my_au"), ("MZ (au)", "mz_au")],
        ),
        (
            "cd_velocity_dipole",
            "CD Spectrum (Velocity Dipole)",
            "R (1e40*cgs)",
            "rotatory_strength_cgs",
            [("MX (au)", "mx_au"), ("MY (au)", "my_au"), ("MZ (au)", "mz_au")],
        ),
    )

    for kind, heading, value_label, value_key, component_headers in layouts:
        section = _render_tddft_spectrum_table(
            heading,
            spectra.get(kind, {}),
            value_label,
            value_key,
            component_headers,
            format_number=format_number,
            make_table=make_table,
        )
        if section:
            sections.append(section)

    return "\n\n".join(sections)


def render_tddft_section(
    tddft: Dict[str, Any],
    data: Optional[Dict[str, Any]] = None,
    *,
    format_number: FormatNumber,
    make_table: MakeTable,
    build_orbital_irrep_lookup: BuildOrbitalIrrepLookup,
    build_cmo_lookup: BuildCMOLookup,
    format_transition_with_irreps: FormatTransitionWithIrreps,
    format_transition_cmo_character: FormatTransitionCMOCharacter,
    yes_no_unknown: YesNoUnknown,
    get_excited_state_opt_data: GetExcitedStateOptData,
) -> str:
    """Compact TDDFT/CIS summary for markdown reports."""
    final_block = tddft.get("final_excited_state_block")
    if not final_block:
        blocks = tddft.get("excited_state_blocks", [])
        final_block = blocks[-1] if blocks else None
    if not final_block:
        return ""

    lines = []
    render_data = data or {}
    irrep_lookup = build_orbital_irrep_lookup(render_data)
    cmo_lookup = build_cmo_lookup(render_data)
    summary = tddft.get("summary", {})
    excopt = get_excited_state_opt_data(render_data) or {}
    meta_bits = []
    if summary.get("input_block"):
        meta_bits.append(f"input `%{summary['input_block']}`")
    if summary.get("nroots") is not None:
        meta_bits.append(f"nroots={summary['nroots']}")
    if excopt.get("target_root") is not None:
        meta_bits.append(f"iroot={excopt['target_root']}")
    if excopt.get("target_multiplicity"):
        meta_bits.append(f"irootmult={excopt['target_multiplicity']}")
    if "followiroot" in excopt:
        meta_bits.append(f"followiroot={yes_no_unknown(excopt.get('followiroot'))}")
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
        oscillator_strengths = {}
        spectrum = tddft.get("spectra", {}).get("absorption_electric_dipole", {})
        for transition in spectrum.get("transitions", []):
            root = transition.get("to_root")
            if root is not None:
                oscillator_strengths[root] = transition.get("oscillator_strength")

        state_has_symmetry = any(
            state.get("state_label") or state.get("symmetry") or state.get("irrep")
            for state in states
        )
        state_has_cmo = bool(cmo_lookup)
        if state_has_symmetry:
            header = ["State", "Symmetry", "E (eV)", "lambda (nm)", "fosc", "dominant excitation"]
        else:
            header = ["State", "E (eV)", "lambda (nm)", "fosc", "dominant excitation"]
        if state_has_cmo:
            header.append("CMO/NBO character")
        header.append("weight")
        rows = [tuple(header)]
        for state in states[:10]:
            dominant = max(
                state.get("transitions", []),
                key=lambda item: item.get("weight", 0.0),
                default={},
            )
            excitation = format_transition_with_irreps(dominant, irrep_lookup) if dominant else "—"
            cmo_character = format_transition_cmo_character(dominant, cmo_lookup) if dominant else "—"
            state_row = [str(state.get("state", ""))]
            if state_has_symmetry:
                state_row.append(
                    str(state.get("state_label") or state.get("symmetry") or state.get("irrep") or "—")
                )
            state_row.extend([
                format_number(state.get("energy_eV")),
                format_number(state.get("wavelength_nm")),
                format_number(oscillator_strengths.get(state.get("state"))),
                excitation,
            ])
            if state_has_cmo:
                state_row.append(cmo_character)
            state_row.append(format_number(dominant.get("weight"), ".3f") if dominant else "—")
            rows.append(tuple(state_row))
        lines.append(make_table(rows))

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
                format_transition_with_irreps(lead, irrep_lookup),
                format_number(lead.get("occupation")),
            ))
        if len(rows) > 1:
            lines.append("**Leading NTO Pairs**\n" + make_table(rows))

    spectra_section = _render_tddft_spectra_section(
        tddft,
        format_number=format_number,
        make_table=make_table,
    )
    if spectra_section:
        lines.append(spectra_section)

    total_energy = tddft.get("total_energy", {})
    if total_energy:
        energy_bits = []
        if total_energy.get("root") is not None:
            energy_bits.append(f"root {total_energy['root']}")
        if total_energy.get("delta_energy_Eh") is not None:
            energy_bits.append(f"DE = {format_number(total_energy['delta_energy_Eh'], '.6f')} Eh")
        if total_energy.get("total_energy_Eh") is not None:
            energy_bits.append(f"E(tot) = {format_number(total_energy['total_energy_Eh'], '.6f')} Eh")
        if energy_bits:
            lines.append("**Total Energy:** " + " | ".join(energy_bits))

    return "\n\n".join(lines)


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


def epr_g_principal_values(epr: Dict[str, Any]) -> List[float]:
    """Return principal g values, falling back to the g-matrix diagonal."""
    entry = _epr_g_total_entry(epr)
    values = entry.get("values") or []
    if values:
        return list(values)

    g_matrix = (epr.get("g_tensor") or {}).get("g_matrix") or []
    if len(g_matrix) == 3 and all(len(row) >= 3 for row in g_matrix):
        return [g_matrix[0][0], g_matrix[1][1], g_matrix[2][2]]
    return []


def epr_g_iso(epr: Dict[str, Any]) -> Optional[float]:
    """Return isotropic g when available."""
    entry = _epr_g_total_entry(epr)
    iso = entry.get("iso")
    if iso is not None:
        return iso

    values = epr_g_principal_values(epr)
    if len(values) == 3:
        return sum(values) / 3.0
    return None


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


def epr_top_hyperfine(epr: Dict[str, Any]) -> Tuple[Optional[str], Optional[float]]:
    """Return the label and A_iso of the strongest hyperfine nucleus."""
    nuclei = (epr.get("hyperfine") or {}).get("nuclei") or []
    ranked = _rank_hyperfine_nuclei(nuclei)
    if not ranked:
        return None, None
    top = ranked[0]
    return _format_nucleus_label(top["nucleus"]), top["A_iso_MHz"]


def _render_hyperfine_summary_table(
    nuclei: List[Dict[str, Any]],
    *,
    format_number: FormatNumber,
    make_table: MakeTable,
) -> str:
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
            format_number(item.get("A_iso_MHz")),
            format_number(values[0]) if len(values) > 0 else "—",
            format_number(values[1]) if len(values) > 1 else "—",
            format_number(values[2]) if len(values) > 2 else "—",
            format_number(quadrupole.get("e2qQ_MHz")),
            format_number(quadrupole.get("eta")),
        ))

    table = "**Top Hyperfine Couplings**\n" + make_table(rows)
    if len(ranked) > len(shown):
        table += (
            f"\n\n*Showing top {len(shown)} of {len(ranked)} nuclei "
            "by |A_iso|.*"
        )
    return table


def render_epr_section(
    epr: Dict[str, Any],
    heading_level: int = 3,
    *,
    format_number: FormatNumber,
    make_table: MakeTable,
    render_matrix: RenderMatrix,
) -> str:
    """Compact EPR summary for markdown reports."""
    heading = "#" * heading_level
    lines: List[str] = []

    zfs = epr.get("zero_field_splitting") or {}
    if zfs:
        lines.append(f"{heading} Zero-Field Splitting")
        bits = []
        if zfs.get("D_cm-1") is not None:
            bits.append(f"**D:** {format_number(zfs['D_cm-1'], '.6g')} cm^-1")
        if zfs.get("E_over_D") is not None:
            bits.append(f"**E/D:** {format_number(zfs['E_over_D'])}")
        if "right_handed" in zfs:
            bits.append(f"**Right-handed:** {'yes' if zfs['right_handed'] else 'no'}")
        if bits:
            lines.append(" | ".join(bits))

        eigenvalues = zfs.get("eigenvalues") or []
        if eigenvalues:
            lines.append(
                "**Principal values:** "
                + ", ".join(format_number(value, ".6g") for value in eigenvalues)
                + " cm^-1"
            )

        raw_matrix = zfs.get("raw_matrix")
        if raw_matrix:
            lines.append("**Raw D Tensor**\n" + render_matrix(raw_matrix, ".6g"))

        contributions = zfs.get("individual_contributions") or {}
        if contributions:
            rows = [("Contribution", "D (cm^-1)", "E (cm^-1)")]
            for label, values in contributions.items():
                rows.append((
                    str(label),
                    format_number(values.get("D_cm-1"), ".6g"),
                    format_number(values.get("E_cm-1"), ".6g"),
                ))
            lines.append("**Contribution Breakdown**\n" + make_table(rows))

    g_tensor = epr.get("g_tensor") or {}
    if g_tensor:
        lines.append(f"{heading} g-Tensor")
        g_values = epr_g_principal_values(epr)
        g_iso = epr_g_iso(epr)
        bits = []
        if g_values:
            bits.append(
                "**Principal values:** "
                + ", ".join(format_number(value, ".7f") for value in g_values)
            )
        if g_iso is not None:
            bits.append(f"**g_iso:** {format_number(g_iso, '.7f')}")
        if bits:
            lines.append(" | ".join(bits))

        g_matrix = g_tensor.get("g_matrix")
        if g_matrix:
            lines.append("**g Matrix**\n" + render_matrix(g_matrix, ".7f"))

        breakdown = g_tensor.get("breakdown") or {}
        if breakdown:
            rows = [("Contribution", "g_x", "g_y", "g_z", "g_iso")]
            for label, entry in breakdown.items():
                values = entry.get("values") or []
                rows.append((
                    str(label),
                    format_number(values[0], ".7f") if len(values) > 0 else "—",
                    format_number(values[1], ".7f") if len(values) > 1 else "—",
                    format_number(values[2], ".7f") if len(values) > 2 else "—",
                    format_number(entry.get("iso"), ".7f"),
                ))
            lines.append("**Breakdown**\n" + make_table(rows))

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
                    format_number(values[0], ".7g") if len(values) > 0 else "—",
                    format_number(values[1], ".7g") if len(values) > 1 else "—",
                    format_number(values[2], ".7g") if len(values) > 2 else "—",
                    format_number(atom.get("iso"), ".7g"),
                ))
            atom_text = "**Top Atom Contributions**\n" + make_table(rows)
            if len(top_atoms) > len(shown):
                atom_text += (
                    f"\n\n*Showing top {len(shown)} of {len(top_atoms)} atoms "
                    "by |g_iso|.*"
                )
            lines.append(atom_text)

    hyperfine = epr.get("hyperfine") or {}
    if hyperfine:
        lines.append(f"{heading} Hyperfine / EFG")
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

        hyperfine_table = _render_hyperfine_summary_table(
            hyperfine.get("nuclei") or [],
            format_number=format_number,
            make_table=make_table,
        )
        if hyperfine_table:
            lines.append(hyperfine_table)

    return "\n\n".join(lines)
