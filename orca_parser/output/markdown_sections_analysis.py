"""Markdown renderers for chemistry-heavy analysis sections.

This module holds the dense charge/spin/Mayer/NBO reporting logic that would
otherwise bloat ``markdown_writer.py``. It also exposes the orbital-irrep and
NBO-CMO helper functions used by the TDDFT renderer so those chemistry-aware
lookups stay in one place.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


FormatNumber = Callable[[Any, str], str]
MakeTable = Callable[[List[tuple]], str]


def build_orbital_irrep_lookup(data: Dict[str, Any]) -> Dict[str, Dict[int, str]]:
    """Build a spin-aware lookup of orbital index -> irrep."""
    orbitals = data.get("orbital_energies", {})
    lookup: Dict[str, Dict[int, str]] = {"": {}, "a": {}, "b": {}}

    for spin_key, values in (
        ("", orbitals.get("orbitals") or []),
        ("a", orbitals.get("alpha_orbitals") or []),
        ("b", orbitals.get("beta_orbitals") or []),
    ):
        for orbital in values:
            index = orbital.get("index")
            irrep = orbital.get("irrep")
            if index is None or not irrep:
                continue
            lookup[spin_key][int(index)] = str(irrep)

    return lookup


def format_transition_with_irreps(
    transition: Dict[str, Any],
    irrep_lookup: Dict[str, Dict[int, str]],
) -> str:
    """Format an excitation/NTO pair with orbital irreps when available."""
    from_label = str(transition.get("from_orbital", "") or "")
    to_label = str(transition.get("to_orbital", "") or "")
    from_irrep = _lookup_transition_irrep(
        irrep_lookup,
        transition.get("from_index"),
        transition.get("from_spin"),
    )
    to_irrep = _lookup_transition_irrep(
        irrep_lookup,
        transition.get("to_index"),
        transition.get("to_spin"),
    )
    if from_irrep:
        from_label = f"{from_label} ({from_irrep})"
    if to_irrep:
        to_label = f"{to_label} ({to_irrep})"
    if not from_label and not to_label:
        return "—"
    return f"{from_label} -> {to_label}".strip()


def build_cmo_lookup(data: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """Build a lookup from ORCA orbital index to NBO CMO entry."""
    lookup: Dict[int, Dict[str, Any]] = {}
    nbo = data.get("nbo", {})
    for cmo in nbo.get("cmo_analysis") or []:
        index = cmo.get("orca_orbital_index")
        if index is None and cmo.get("mo_index") is not None:
            try:
                index = int(cmo["mo_index"]) - 1
            except (TypeError, ValueError):
                index = None
        if index is None:
            continue
        lookup[int(index)] = cmo
    return lookup


def format_transition_cmo_character(
    transition: Dict[str, Any],
    cmo_lookup: Dict[int, Dict[str, Any]],
) -> str:
    """Format a TDDFT transition in terms of dominant NBO CMO character."""
    try:
        from_index = int(transition.get("from_index"))
        to_index = int(transition.get("to_index"))
    except (TypeError, ValueError):
        return "—"

    donor = _summarize_cmo_character(cmo_lookup.get(from_index), limit=2, with_percent=False)
    acceptor = _summarize_cmo_character(cmo_lookup.get(to_index), limit=2, with_percent=False)
    if donor == "—" and acceptor == "—":
        return "—"
    return f"{donor} -> {acceptor}"


def get_atom_list(section: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return per-atom rows regardless of the population section's key name."""
    return section.get("atoms") or section.get("atomic_charges") or section.get("atomic_data") or []


def get_charges(data: Dict[str, Any], scheme: str) -> Optional[List[float]]:
    """Return a flat list of charges for multi-molecule comparison rendering."""
    section = data.get(scheme, {})
    if not section:
        return None
    atoms = get_atom_list(section)
    charges = [atom.get("charge") for atom in atoms if "charge" in atom]
    return charges if charges else None


def render_analysis_sections(
    data: Dict[str, Any],
    *,
    context: Dict[str, Any],
    heading_level: int,
    format_number: FormatNumber,
    make_table: MakeTable,
) -> List[str]:
    """Render the dense analysis-heavy sections for a single report."""
    h2 = "#" * (heading_level + 1)
    h3 = "#" * (heading_level + 2)
    blocks: List[str] = []
    atom_symbols = context.get("atom_symbols", [])

    charge_data = _collect_charges(data)
    if charge_data:
        rows = [("Atom",) + tuple(charge_data)]
        all_indices = sorted({index for values in charge_data.values() for index in values})
        for index in all_indices:
            rows.append((_atom_label(atom_symbols, index),) + tuple(
                f"{charge_data[scheme][index]:.4f}" if index in charge_data[scheme] else "—"
                for scheme in charge_data
            ))
        blocks.append(f"{h2} Atomic Charges\n{make_table(rows)}")

    if context.get("is_uhf"):
        spin_data = _collect_spin(data)
        if spin_data:
            rows = [("Atom",) + tuple(spin_data)]
            all_indices = sorted({index for values in spin_data.values() for index in values})
            for index in all_indices:
                rows.append((_atom_label(atom_symbols, index),) + tuple(
                    f"{spin_data[scheme][index]:.4f}" if index in spin_data[scheme] else "—"
                    for scheme in spin_data
                ))
            totals = {scheme: sum(values.values()) for scheme, values in spin_data.items()}
            blocks.append(
                f"{h2} Atomic Spin Populations\n"
                f"{make_table(rows)}\n\n"
                + "**Spin totals:** "
                + "  ".join(f"{scheme} = {total:.4f}" for scheme, total in totals.items())
            )

    reduced_orbital_section = _render_reduced_orbital_populations(data, format_number, make_table)
    if reduced_orbital_section:
        blocks.append(f"{h2} Reduced Orbital Populations\n{reduced_orbital_section}")

    mayer = data.get("mayer", {}) or {}
    bond_orders = mayer.get("bond_orders", [])
    if bond_orders:
        rows = [("Bond", "Order")]
        for bond in bond_orders:
            atom_i = f"{bond.get('symbol_i', '?')}{bond.get('atom_i', 0) + 1}"
            atom_j = f"{bond.get('symbol_j', '?')}{bond.get('atom_j', 0) + 1}"
            rows.append((f"{atom_i}–{atom_j}", f"{bond.get('bond_order', 0):.4f}"))
        blocks.append(f"{h2} Mayer Bond Orders ({len(bond_orders)} bonds)\n{make_table(rows)}")

    nbo = data.get("nbo")
    is_uhf = bool(context.get("is_uhf", False))
    if nbo:
        blocks.append(f"{h2} NBO Analysis")

        npa = nbo.get("overall_npa_summary") if is_uhf else nbo.get("npa_summary", [])
        if npa:
            has_spin = any("spin_density" in atom for atom in npa)
            if has_spin:
                rows = [("Atom", "NPA charge", "Core", "Valence", "Rydberg", "Total", "Spin ρ")]
            else:
                rows = [("Atom", "NPA charge", "Core", "Valence", "Rydberg", "Total")]
            for atom in npa:
                row = [
                    f"{atom.get('symbol', '?')}{atom.get('index', '')}",
                    f"{atom.get('natural_charge', 0):.5f}",
                    f"{atom.get('core_pop', 0):.5f}",
                    f"{atom.get('valence_pop', 0):.5f}",
                    f"{atom.get('rydberg_pop', 0):.5f}",
                    f"{atom.get('total_pop', 0):.5f}",
                ]
                if has_spin:
                    row.append(f"{atom.get('spin_density', 0):.5f}")
                rows.append(tuple(row))
            blocks.append(f"{h3} NPA Charges\n{make_table(rows)}")

        electron_config = _render_natural_electron_configuration(nbo, is_uhf, make_table)
        if electron_config:
            blocks.append(f"{h3} Natural Electron Configuration\n{electron_config}")

        cmo_section = _render_cmo_analysis_section(nbo, data, make_table)
        if cmo_section:
            blocks.append(f"{h3} Canonical MO Character (NBO CMO)\n{cmo_section}")

        if is_uhf:
            for spin_key, label in (("alpha_spin", "α"), ("beta_spin", "β")):
                spin_section = nbo.get(spin_key, {}) or {}
                spin_npa = spin_section.get("npa_summary", [])
                if not spin_npa:
                    continue
                rows = [("Atom", "NPA charge", "Core", "Valence", "Rydberg", "Total")]
                for atom in spin_npa:
                    rows.append((
                        f"{atom.get('symbol', '?')}{atom.get('index', '')}",
                        f"{atom.get('natural_charge', 0):.5f}",
                        f"{atom.get('core_pop', 0):.5f}",
                        f"{atom.get('valence_pop', 0):.5f}",
                        f"{atom.get('rydberg_pop', 0):.5f}",
                        f"{atom.get('total_pop', 0):.5f}",
                    ))
                blocks.append(f"{h3} NPA — {label} spin\n{make_table(rows)}")

        wbi_key = "overall_wiberg_bond_indices" if is_uhf else "wiberg_bond_indices"
        wbi = nbo.get(wbi_key, {})
        matrix = wbi.get("matrix", []) if isinstance(wbi, dict) else []
        if matrix and isinstance(matrix, list) and isinstance(matrix[0], list):
            wbi_bonds = _collect_wiberg_bonds(matrix, atom_symbols)
            if wbi_bonds:
                rows = [("Bond", "WBI")]
                for atom_i, atom_j, value in wbi_bonds:
                    rows.append((f"{atom_i}–{atom_j}", f"{value:.4f}"))
                blocks.append(f"{h3} Wiberg Bond Indices ({len(wbi_bonds)} bonds)\n{make_table(rows)}")

        if is_uhf:
            for spin_key, label in (("alpha_spin", "α"), ("beta_spin", "β")):
                spin_section = nbo.get(spin_key, {}) or {}
                spin_wbi = spin_section.get("wiberg_bond_indices", {})
                matrix = spin_wbi.get("matrix", []) if isinstance(spin_wbi, dict) else []
                if matrix and isinstance(matrix, list) and isinstance(matrix[0], list):
                    spin_bonds = _collect_wiberg_bonds(matrix, atom_symbols)
                    if spin_bonds:
                        rows = [("Bond", "WBI")]
                        for atom_i, atom_j, value in spin_bonds:
                            rows.append((f"{atom_i}–{atom_j}", f"{value:.4f}"))
                        blocks.append(f"{h3} Wiberg — {label} spin ({len(spin_bonds)} bonds)\n{make_table(rows)}")

        if is_uhf:
            for spin_key, label in (("alpha_spin", "α"), ("beta_spin", "β")):
                entries = (nbo.get(spin_key, {}) or {}).get("e2_perturbation", [])
                if entries:
                    blocks.append(
                        f"{h3} E(2) Perturbation — {label} spin ({len(entries)} entries)\n"
                        + _render_e2_table(entries, make_table)
                    )
        else:
            entries = nbo.get("e2_perturbation", [])
            if entries:
                blocks.append(f"{h3} Second-Order Perturbation E(2)\n" + _render_e2_table(entries, make_table))

    return blocks


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


def _lookup_transition_irrep(
    irrep_lookup: Dict[str, Dict[int, str]],
    index: Any,
    spin: Any,
) -> str:
    """Look up the irrep for an orbital index/spin pair."""
    if index is None:
        return ""
    try:
        index_int = int(index)
    except (TypeError, ValueError):
        return ""

    spin_key = _normalize_spin_key(spin)
    if spin_key and index_int in irrep_lookup.get(spin_key, {}):
        return irrep_lookup[spin_key][index_int]
    return irrep_lookup.get("", {}).get(index_int, "")


def _aggregate_cmo_character(entry: Dict[str, Any]) -> List[tuple[str, float]]:
    """Aggregate NBO CMO contributions by compact orbital-character label."""
    totals: Dict[str, float] = {}
    for contribution in entry.get("nbo_contributions") or []:
        label = contribution.get("character_label") or contribution.get("nbo_desc")
        if not label:
            continue
        percent = contribution.get("approx_percent")
        if percent is None:
            coefficient = contribution.get("coefficient")
            try:
                percent = 100.0 * float(coefficient) * float(coefficient)
            except (TypeError, ValueError):
                percent = 0.0
        totals[str(label)] = totals.get(str(label), 0.0) + float(percent)
    return sorted(totals.items(), key=lambda item: item[1], reverse=True)


def _summarize_cmo_character(
    entry: Optional[Dict[str, Any]],
    *,
    limit: int = 2,
    with_percent: bool = False,
) -> str:
    """Summarize the dominant NBO character of a canonical MO."""
    if not entry:
        return "—"
    ordered = _aggregate_cmo_character(entry)
    if not ordered:
        return "—"
    pieces = []
    for label, percent in ordered[:limit]:
        if with_percent:
            pieces.append(f"{label} {percent:.1f}%")
        else:
            pieces.append(label)
    return "; ".join(pieces) if with_percent else " + ".join(pieces)


def _collect_relevant_cmo_indices(
    data: Dict[str, Any],
    cmo_lookup: Dict[int, Dict[str, Any]],
    *,
    state_limit: int = 10,
    frontier_window: int = 8,
) -> List[int]:
    """Choose the most relevant canonical orbitals to display."""
    indices: List[int] = []
    seen: set[int] = set()

    tddft = data.get("tddft", {})
    final_block = tddft.get("final_excited_state_block")
    if not final_block:
        blocks = tddft.get("excited_state_blocks", [])
        final_block = blocks[-1] if blocks else None

    if final_block:
        for state in (final_block.get("states") or [])[:state_limit]:
            dominant = max(
                state.get("transitions", []),
                key=lambda item: item.get("weight", 0.0),
                default={},
            )
            for key in ("from_index", "to_index"):
                value = dominant.get(key)
                if value is None:
                    continue
                try:
                    index = int(value)
                except (TypeError, ValueError):
                    continue
                if index in cmo_lookup and index not in seen:
                    indices.append(index)
                    seen.add(index)
        if indices:
            return sorted(indices)

    occupied = sorted(index for index, cmo in cmo_lookup.items() if cmo.get("type") == "occ")
    virtual = sorted(index for index, cmo in cmo_lookup.items() if cmo.get("type") == "vir")
    fallback = occupied[-frontier_window:] + virtual[:frontier_window]
    return [index for index in fallback if index in cmo_lookup]


def _render_natural_electron_configuration(
    nbo: Dict[str, Any],
    is_uhf: bool,
    make_table: MakeTable,
) -> str:
    """Render the Natural Electron Configuration table."""
    configs = nbo.get("overall_electron_configurations") if is_uhf else nbo.get("electron_configurations")
    if not configs:
        configs = nbo.get("electron_configurations") or nbo.get("overall_electron_configurations") or []
    if not configs:
        return ""

    rows = [("Atom", "NBO atom #", "ORCA atom idx", "Configuration")]
    for entry in configs:
        rows.append((
            f"{entry.get('symbol', '?')}{entry.get('index', '')}",
            str(entry.get("atom_nbo_index", entry.get("index", ""))),
            str(entry.get("atom_orca_index", "—")),
            str(entry.get("configuration", "")),
        ))

    note = "NBO prints atom numbers as 1-based; ORCA's internal atom indexing used in some sections is 0-based."
    return f"{note}\n\n{make_table(rows)}"


def _render_cmo_analysis_section(
    nbo: Dict[str, Any],
    data: Dict[str, Any],
    make_table: MakeTable,
) -> str:
    """Render a compact CMO/NBO character table for TDDFT/frontier orbitals."""
    del nbo
    cmo_lookup = build_cmo_lookup(data)
    if not cmo_lookup:
        return ""

    indices = _collect_relevant_cmo_indices(data, cmo_lookup)
    if not indices:
        return ""

    rows = [("ORCA MO", "CMO MO", "Type", "E (Eh)", "Leading character", "Top contributions")]
    for index in indices:
        entry = cmo_lookup.get(index)
        if not entry:
            continue
        rows.append((
            str(index),
            str(entry.get("nbo_mo_index", entry.get("mo_index", ""))),
            str(entry.get("type", "")),
            _format_optional_float(entry.get("energy_au"), ".5f"),
            _summarize_cmo_character(entry, limit=1, with_percent=False),
            _summarize_cmo_character(entry, limit=3, with_percent=True),
        ))

    if len(rows) == 1:
        return ""

    tddft = data.get("tddft", {})
    has_tddft = bool(tddft.get("final_excited_state_block") or tddft.get("excited_state_blocks"))
    scope = (
        "Shown for canonical orbitals referenced by the dominant TDDFT/CIS excitations."
        if has_tddft
        else "Shown for frontier-adjacent canonical orbitals."
    )
    note = "NBO CMO numbering is 1-based; ORCA orbital numbers are 0-based, so CMO MO N corresponds to ORCA MO N-1."
    return f"{scope}\n{note}\n\n{make_table(rows)}"


def _collect_charges(
    data: Dict[str, Any],
    schemes: tuple[str, ...] = ("mulliken", "loewdin", "hirshfeld", "mbis", "chelpg"),
) -> Dict[str, Dict[int, Any]]:
    """Return dict of scheme -> {atom_index: charge} for all present schemes."""
    result: Dict[str, Dict[int, Any]] = {}
    for scheme in schemes:
        section = data.get(scheme, {})
        if not section:
            continue
        atoms = get_atom_list(section)
        values = {
            atom["index"]: atom["charge"]
            for atom in atoms
            if "index" in atom and "charge" in atom
        }
        if values:
            result[scheme.capitalize()] = values
    return result


def _collect_spin(
    data: Dict[str, Any],
    schemes: tuple[str, ...] = ("mulliken", "loewdin", "hirshfeld", "mbis"),
) -> Dict[str, Dict[int, Any]]:
    """Return dict of scheme -> {atom_index: spin_population} for UHF calculations."""
    result: Dict[str, Dict[int, Any]] = {}
    for scheme in schemes:
        section = data.get(scheme, {})
        if not section:
            continue
        atoms = get_atom_list(section)
        values = {
            atom["index"]: atom["spin_population"]
            for atom in atoms
            if "index" in atom and "spin_population" in atom
        }
        if values:
            result[scheme.capitalize()] = values
    return result


def _render_shell_totals_table(
    atom_blocks: List[Dict[str, Any]],
    format_number: FormatNumber,
    make_table: MakeTable,
) -> str:
    """Render compact per-atom shell totals for reduced orbital populations."""
    if not atom_blocks:
        return ""

    preferred_shells = ["s", "p", "d", "f", "g", "h"]
    seen_shells: List[str] = []
    for atom in atom_blocks:
        for shell in (atom.get("shell_totals") or {}):
            if shell not in seen_shells:
                seen_shells.append(shell)

    shell_columns = [shell for shell in preferred_shells if shell in seen_shells]
    shell_columns.extend(sorted(shell for shell in seen_shells if shell not in shell_columns))
    if not shell_columns:
        return ""

    rows = [("Atom",) + tuple(shell_columns) + ("Total",)]
    for atom in atom_blocks:
        index = atom.get("index")
        symbol = str(atom.get("symbol", "?"))
        atom_label = f"{symbol}{int(index) + 1}" if isinstance(index, int) else symbol
        shell_totals = atom.get("shell_totals") or {}
        total = sum(
            float(value)
            for value in shell_totals.values()
            if isinstance(value, (int, float))
        )
        row = [atom_label]
        for shell in shell_columns:
            row.append(
                format_number(shell_totals.get(shell), ".6f")
                if shell in shell_totals
                else "—"
            )
        row.append(format_number(total, ".6f"))
        rows.append(tuple(row))

    return make_table(rows)


def _render_reduced_orbital_populations(
    data: Dict[str, Any],
    format_number: FormatNumber,
    make_table: MakeTable,
) -> str:
    """Render Mulliken/Loewdin reduced orbital shell populations."""
    sections: List[str] = []
    for scheme_key, scheme_label in (("mulliken", "Mulliken"), ("loewdin", "Loewdin")):
        scheme = data.get(scheme_key, {}) or {}
        charges = scheme.get("reduced_orbital_charges") or []
        spin = scheme.get("reduced_orbital_spin_populations") or []

        if charges:
            sections.append(
                f"**{scheme_label} shell totals — charge**\n"
                + _render_shell_totals_table(charges, format_number, make_table)
            )
        if spin:
            sections.append(
                f"**{scheme_label} shell totals — spin**\n"
                + _render_shell_totals_table(spin, format_number, make_table)
            )

    return "\n\n".join(section for section in sections if section)


def _collect_wiberg_bonds(matrix: List[List[Any]], atom_symbols: List[str]) -> List[tuple[str, str, float]]:
    """Collect WBI entries above the report threshold with atom labels."""
    bonds: List[tuple[str, str, float]] = []
    n_atoms = len(matrix)
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            value = matrix[i][j]
            if not isinstance(value, (int, float)) or value <= 0.05:
                continue
            atom_i = f"{atom_symbols[i]}{i+1}" if i < len(atom_symbols) else str(i + 1)
            atom_j = f"{atom_symbols[j]}{j+1}" if j < len(atom_symbols) else str(j + 1)
            bonds.append((atom_i, atom_j, value))
    return bonds


def _render_e2_table(entries: List[Dict[str, Any]], make_table: MakeTable) -> str:
    """Render E(2) table: top 20 by energy, then a compact summary of the rest."""
    top = sorted(entries, key=lambda item: -item.get("E2_kcal_mol", 0))
    shown = top[:20]
    rows = [("Donor", "Acceptor", "E(2) kcal/mol", "ΔE a.u.", "F a.u.")]
    for entry in shown:
        rows.append((
            entry.get("donor", ""),
            entry.get("acceptor", ""),
            f"{entry.get('E2_kcal_mol', 0):.2f}",
            f"{entry.get('E_gap_au', 0):.4f}",
            f"{entry.get('Fock_au', 0):.4f}",
        ))
    result = make_table(rows)
    if len(entries) > 20:
        total = sum(entry.get("E2_kcal_mol", 0) for entry in entries)
        result += f"\n\n*Showing top 20 of {len(entries)} interactions. Total E(2) = {total:.2f} kcal/mol.*"
    return result


def _atom_label(atom_symbols: List[str], index: int) -> str:
    """Preserve the current report atom-label behaviour for merged charge tables."""
    if not atom_symbols:
        return f"?{index}"
    symbol_index = min(index, len(atom_symbols) - 1)
    symbol = atom_symbols[symbol_index]
    if index < len(atom_symbols):
        return f"{symbol}{index + 1}"
    return f"?{index}"


def _format_optional_float(value: Any, fmt: str) -> str:
    """Format a float when present, otherwise use a markdown dash."""
    if value is None:
        return "—"
    try:
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return str(value)
