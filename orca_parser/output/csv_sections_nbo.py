"""CSV writers for NBO-oriented exports.

These sections cover the denser NBO/NAO/NLMO tabular outputs so the main CSV
writer can stay focused on orchestration rather than chemistry-specific row
flattening logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List


WriteCSV = Callable[[Path, str, List[Dict], List[str]], Path]


def write_nbo_nao_section(
    data: Dict[str, Any],
    directory: Path,
    stem: str,
    *,
    write_csv: WriteCSV,
) -> List[Path]:
    """Write NAO occupancy tables for RHF and spin-resolved UHF cases."""
    nbo = data.get("nbo", {})
    files: List[Path] = []

    def _export(naos: List[Dict[str, Any]], suffix: str) -> None:
        if not naos:
            return
        has_spin = any("spin" in nao for nao in naos[:5])
        fieldnames = ["index", "symbol", "atom_no", "angular", "type", "occupancy", "energy_Eh"]
        if has_spin:
            fieldnames.append("spin")
        rows = [{
            "index": nao.get("index"),
            "symbol": nao.get("symbol"),
            "atom_no": nao.get("atom_no"),
            "angular": nao.get("angular"),
            "type": nao.get("type"),
            "occupancy": nao.get("occupancy"),
            "energy_Eh": nao.get("energy_Eh", ""),
            "spin": nao.get("spin", ""),
        } for nao in naos]
        files.append(write_csv(directory, f"{stem}_nbo_nao{suffix}.csv", rows, fieldnames))

    _export(nbo.get("nao_occupancies"), "")
    for spin in ("alpha", "beta"):
        _export(nbo.get(spin, {}).get("nao_occupancies"), f"_{spin}")
    return files


def write_nbo_npa_section(
    data: Dict[str, Any],
    directory: Path,
    stem: str,
    *,
    write_csv: WriteCSV,
) -> List[Path]:
    """Write NPA summary tables for total and spin-resolved densities."""
    nbo = data.get("nbo", {})
    files: List[Path] = []

    def _export(atoms: List[Dict[str, Any]], suffix: str) -> None:
        if not atoms:
            return
        has_spin = any("spin_density" in atom for atom in atoms[:5])
        fieldnames = [
            "index", "symbol", "natural_charge", "core_pop",
            "valence_pop", "rydberg_pop", "total_pop",
        ]
        if has_spin:
            fieldnames.append("spin_density")
        rows = [{
            "index": atom.get("index"),
            "symbol": atom.get("symbol"),
            "natural_charge": atom.get("natural_charge"),
            "core_pop": atom.get("core_pop"),
            "valence_pop": atom.get("valence_pop"),
            "rydberg_pop": atom.get("rydberg_pop"),
            "total_pop": atom.get("total_pop"),
            "spin_density": atom.get("spin_density", ""),
        } for atom in atoms]
        files.append(write_csv(directory, f"{stem}_nbo_npa{suffix}.csv", rows, fieldnames))

    _export(nbo.get("npa_summary"), "")
    for spin in ("alpha", "beta"):
        _export(nbo.get(spin, {}).get("npa_summary"), f"_{spin}")
    return files


def write_nbo_matrix_section(
    data: Dict[str, Any],
    directory: Path,
    stem: str,
    *,
    write_csv: WriteCSV,
    key: str,
    filename_suffix: str,
) -> List[Path]:
    """Write flattened NBO bond-index matrices such as Wiberg and NBI."""
    nbo = data.get("nbo", {})
    files: List[Path] = []

    def _export(matrix: List[Dict[str, Any]], suffix: str) -> None:
        if not matrix:
            return
        rows = []
        for i, row in enumerate(matrix):
            atom_i = row.get("atom", i + 1)
            symbol_i = row.get("symbol", "")
            for j, value in enumerate(row.get("values", [])):
                rows.append({
                    "atom_i": atom_i,
                    "symbol_i": symbol_i,
                    "atom_j": j + 1,
                    "bond_index": value,
                })
        if rows:
            files.append(write_csv(
                directory,
                f"{stem}_nbo_{filename_suffix}{suffix}.csv",
                rows,
                ["atom_i", "symbol_i", "atom_j", "bond_index"],
            ))

    _export(nbo.get(key), "")
    for spin in ("alpha", "beta"):
        _export(nbo.get(spin, {}).get(key), f"_{spin}")
    return files


def write_nbo_lewis_section(
    data: Dict[str, Any],
    directory: Path,
    stem: str,
    *,
    write_csv: WriteCSV,
) -> List[Path]:
    """Write flattened Lewis/NBO tables with hybridization detail strings."""
    nbo = data.get("nbo", {})
    files: List[Path] = []

    def _export(nbos: List[Dict[str, Any]], suffix: str) -> None:
        if not nbos:
            return
        rows = []
        for orbital in nbos:
            hybrids = orbital.get("hybrids", [])
            hybridization_detail = "; ".join(
                f"{hybrid.get('atom', '')}({hybrid.get('symbol', '')}): "
                f"s={hybrid.get('s_pct', 0):.1f}% p={hybrid.get('p_pct', 0):.1f}% "
                f"d={hybrid.get('d_pct', 0):.1f}%"
                for hybrid in hybrids
            ) if hybrids else ""
            rows.append({
                "nbo_index": orbital.get("index"),
                "type": orbital.get("type"),
                "occupancy": orbital.get("occupancy"),
                "energy_Eh": orbital.get("energy_Eh", ""),
                "label": orbital.get("label", ""),
                "s_pct": hybrids[0].get("s_pct", "") if hybrids else "",
                "p_pct": hybrids[0].get("p_pct", "") if hybrids else "",
                "d_pct": hybrids[0].get("d_pct", "") if hybrids else "",
                "hybridization_detail": hybridization_detail,
            })
        files.append(write_csv(
            directory,
            f"{stem}_nbo_lewis{suffix}.csv",
            rows,
            [
                "nbo_index", "type", "occupancy", "energy_Eh", "label",
                "s_pct", "p_pct", "d_pct", "hybridization_detail",
            ],
        ))

    _export(nbo.get("nbo_lewis"), "")
    for spin in ("alpha", "beta"):
        _export(nbo.get(spin, {}).get("nbo_lewis"), f"_{spin}")
    return files


def write_nbo_e2_section(
    data: Dict[str, Any],
    directory: Path,
    stem: str,
    *,
    write_csv: WriteCSV,
) -> List[Path]:
    """Write second-order perturbation tables for total and spin-resolved data."""
    nbo = data.get("nbo", {})
    files: List[Path] = []

    def _export(entries: List[Dict[str, Any]], suffix: str) -> None:
        if not entries:
            return
        files.append(write_csv(
            directory,
            f"{stem}_nbo_e2{suffix}.csv",
            entries,
            ["donor", "acceptor", "E2_kcal_mol", "E_gap_au", "Fock_au"],
        ))

    _export(nbo.get("e2_perturbation"), "")
    for spin in ("alpha", "beta"):
        _export(nbo.get(spin, {}).get("e2_perturbation"), f"_{spin}")
    return files


def write_nbo_nlmo_hybridization_section(
    data: Dict[str, Any],
    directory: Path,
    stem: str,
    *,
    write_csv: WriteCSV,
) -> List[Path]:
    """Write per-atom NLMO hybridization contributions."""
    nbo = data.get("nbo", {})
    files: List[Path] = []

    def _export(nlmos: List[Dict[str, Any]], suffix: str) -> None:
        if not nlmos:
            return
        rows = []
        for nlmo in nlmos:
            for contribution in nlmo.get("contributions", []):
                rows.append({
                    "nlmo_index": nlmo.get("index"),
                    "occupancy": nlmo.get("occupancy"),
                    "parent_pct": nlmo.get("parent_pct", ""),
                    "parent_nbo": nlmo.get("parent_nbo", ""),
                    "atom": contribution.get("atom", ""),
                    "symbol": contribution.get("symbol", ""),
                    "pct": contribution.get("pct", ""),
                    "s_pct": contribution.get("s_pct", ""),
                    "p_pct": contribution.get("p_pct", ""),
                    "d_pct": contribution.get("d_pct", ""),
                })
        if rows:
            files.append(write_csv(
                directory,
                f"{stem}_nbo_nlmo_hyb{suffix}.csv",
                rows,
                [
                    "nlmo_index", "occupancy", "parent_pct", "parent_nbo",
                    "atom", "symbol", "pct", "s_pct", "p_pct", "d_pct",
                ],
            ))

    _export(nbo.get("nlmo_hybridization"), "")
    for spin in ("alpha", "beta"):
        _export(nbo.get(spin, {}).get("nlmo_hybridization"), f"_{spin}")
    return files


def write_nbo_nlmo_bond_order_section(
    data: Dict[str, Any],
    directory: Path,
    stem: str,
    *,
    write_csv: WriteCSV,
) -> List[Path]:
    """Write NLMO bond-order tables."""
    nbo = data.get("nbo", {})
    files: List[Path] = []

    def _export(bonds: List[Dict[str, Any]], suffix: str) -> None:
        if not bonds:
            return
        files.append(write_csv(
            directory,
            f"{stem}_nbo_nlmo_bo{suffix}.csv",
            bonds,
            ["atom_i", "atom_j", "nlmo_index", "bond_order", "hybrid_overlap"],
        ))

    _export(nbo.get("nlmo_bond_orders"), "")
    for spin in ("alpha", "beta"):
        _export(nbo.get(spin, {}).get("nlmo_bond_orders"), f"_{spin}")
    return files


def write_nbo_nlmo_steric_section(
    data: Dict[str, Any],
    directory: Path,
    stem: str,
    *,
    write_csv: WriteCSV,
) -> List[Path]:
    """Write NLMO steric unit and pairwise tables."""
    nbo = data.get("nbo", {})
    files: List[Path] = []

    def _export(steric: Dict[str, Any], suffix: str) -> None:
        if not steric:
            return
        unit_rows = steric.get("unit_contributions", [])
        if unit_rows:
            files.append(write_csv(
                directory,
                f"{stem}_nbo_nlmo_steric{suffix}.csv",
                unit_rows,
                ["nlmo_index", "label", "dE_kcal_mol"],
            ))
        pair_rows = steric.get("pairwise", [])
        if pair_rows:
            files.append(write_csv(
                directory,
                f"{stem}_nbo_nlmo_steric_pairwise{suffix}.csv",
                pair_rows,
                ["nlmo_i", "nlmo_j", "S_ij", "dE_ij_kcal_mol"],
            ))

    _export(nbo.get("nlmo_steric"), "")
    for spin in ("alpha", "beta"):
        _export(nbo.get(spin, {}).get("nlmo_steric"), f"_{spin}")
    return files
