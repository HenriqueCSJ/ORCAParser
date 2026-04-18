"""Registry for common CSV output sections.

The parser and markdown paths already moved toward registry-driven extension
points. This module gives CSV exports the same seam so adding a new common
section no longer requires editing ``csv_writer.py``'s central writer list.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

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
    get_deltascf_data as _get_deltascf_data,
    get_excited_state_opt_data as _get_excited_state_opt_data,
    get_symmetry_data as _get_symmetry_data,
    is_surface_scan as _is_surface_scan,
)
from .csv_sections_basic import (
    write_dipole_section as _write_basic_dipole_section,
    write_geometry_section as _write_basic_geometry_section,
    write_solvation_section as _write_basic_solvation_section,
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
    write_metadata_section as _write_state_metadata_section,
    write_symmetry_section as _write_state_symmetry_section,
)


WriteCSV = Callable[[Path, str, List[Dict[str, Any]], List[str]], Path]
CSVSectionRenderer = Callable[[Dict[str, Any], Path, str, WriteCSV], List[Path]]


@dataclass(frozen=True)
class CSVSectionPlugin:
    """Plugin-like description of one common CSV export section."""

    key: str
    order: int = 50
    render_files: CSVSectionRenderer | None = None


_CSV_SECTION_PLUGINS: List[CSVSectionPlugin] = []


def register_csv_section_plugin(
    plugin: CSVSectionPlugin,
    *,
    replace: bool = False,
) -> None:
    """Register a CSV section plugin."""
    global _CSV_SECTION_PLUGINS

    if replace:
        _CSV_SECTION_PLUGINS = [
            existing
            for existing in _CSV_SECTION_PLUGINS
            if existing.key != plugin.key
        ]
    elif any(existing.key == plugin.key for existing in _CSV_SECTION_PLUGINS):
        raise ValueError(f"CSV section already registered: {plugin.key}")

    _CSV_SECTION_PLUGINS.append(plugin)


def get_registered_csv_section_plugins() -> tuple[CSVSectionPlugin, ...]:
    """Return CSV section plugins in stable render order."""
    return tuple(sorted(_CSV_SECTION_PLUGINS, key=lambda plugin: (plugin.order, plugin.key)))


def iter_csv_section_plugins() -> tuple[CSVSectionPlugin, ...]:
    """Return registered CSV section plugins that emit files."""
    return tuple(
        plugin
        for plugin in get_registered_csv_section_plugins()
        if plugin.render_files is not None
    )


def _electronic_state_label(data: Dict[str, Any]) -> str:
    """Short electronic-state label for metadata exports."""
    return _shared_electronic_state_label(data, ground_state_label="Ground-state")


def _get_pop_atoms(section: dict) -> list:
    """Return atom rows regardless of the population section's internal key."""
    return section.get("atoms") or section.get("atomic_charges") or section.get("atomic_data") or []


def _metadata_files(data: Dict[str, Any], directory: Path, stem: str, write_csv: WriteCSV) -> List[Path]:
    return _write_state_metadata_section(
        data,
        directory,
        stem,
        write_csv=write_csv,
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


def _geometry_files(data: Dict[str, Any], directory: Path, stem: str, write_csv: WriteCSV) -> List[Path]:
    return _write_basic_geometry_section(
        data,
        directory,
        stem,
        write_csv=write_csv,
    )


def _symmetry_files(data: Dict[str, Any], directory: Path, stem: str, write_csv: WriteCSV) -> List[Path]:
    return _write_state_symmetry_section(
        data,
        directory,
        stem,
        write_csv=write_csv,
        get_symmetry_data=_get_symmetry_data,
        bool_to_label=_bool_to_label,
    )


def _orbital_energy_files(data: Dict[str, Any], directory: Path, stem: str, write_csv: WriteCSV) -> List[Path]:
    orbitals = _get_final_orbital_energies(data)
    files: List[Path] = []

    if "orbitals" in orbitals:
        rows = [{
            "index": orbital.get("index"),
            "occupation": orbital.get("occupation"),
            "energy_Eh": orbital.get("energy_Eh"),
            "energy_eV": orbital.get("energy_eV"),
            "irrep": orbital.get("irrep", ""),
        } for orbital in orbitals["orbitals"]]
        files.append(
            write_csv(
                directory,
                f"{stem}_orbital_energies.csv",
                rows,
                ["index", "occupation", "energy_Eh", "energy_eV", "irrep"],
            )
        )

    for spin in ("alpha", "beta"):
        key = f"{spin}_orbitals"
        if key not in orbitals:
            continue
        rows = [{
            "index": orbital.get("index"),
            "occupation": orbital.get("occupation"),
            "energy_Eh": orbital.get("energy_Eh"),
            "energy_eV": orbital.get("energy_eV"),
            "irrep": orbital.get("irrep", ""),
        } for orbital in orbitals[key]]
        files.append(
            write_csv(
                directory,
                f"{stem}_orbital_energies_{spin}.csv",
                rows,
                ["index", "occupation", "energy_Eh", "energy_eV", "irrep"],
            )
        )

    return files


def _qro_files(data: Dict[str, Any], directory: Path, stem: str, write_csv: WriteCSV) -> List[Path]:
    qro = data.get("qro")
    if not qro:
        return []
    rows = [{
        "index": orbital.get("index"),
        "occupation": orbital.get("occupation"),
        "type": orbital.get("type", ""),
        "energy_Eh": orbital.get("energy_Eh"),
        "energy_eV": orbital.get("energy_eV"),
        "alpha_energy_eV": orbital.get("alpha_energy_eV", ""),
        "beta_energy_eV": orbital.get("beta_energy_eV", ""),
    } for orbital in qro.get("orbitals", [])]
    return [
        write_csv(
            directory,
            f"{stem}_qro.csv",
            rows,
            ["index", "occupation", "type", "energy_Eh", "energy_eV", "alpha_energy_eV", "beta_energy_eV"],
        )
    ]


def _mulliken_files(data: Dict[str, Any], directory: Path, stem: str, write_csv: WriteCSV) -> List[Path]:
    section = _get_final_population_section(data, "mulliken")
    files: List[Path] = []
    atoms = _get_pop_atoms(section)
    if atoms:
        rows = [{
            "index": atom.get("index"),
            "symbol": atom.get("symbol"),
            "charge": atom.get("charge"),
            "spin_population": atom.get("spin_population", ""),
        } for atom in atoms]
        files.append(
            write_csv(
                directory,
                f"{stem}_mulliken_charges.csv",
                rows,
                ["index", "symbol", "charge", "spin_population"],
            )
        )

    orbital_data = section.get("reduced_orbital_charges", [])
    if orbital_data:
        rows = []
        for atom in orbital_data:
            for contribution in atom.get("contributions", []):
                rows.append({
                    "atom_index": atom.get("index"),
                    "atom_symbol": atom.get("symbol"),
                    "angular": contribution.get("angular"),
                    "charge": contribution.get("charge"),
                    "spin": contribution.get("spin", ""),
                })
        if rows:
            files.append(
                write_csv(
                    directory,
                    f"{stem}_mulliken_orb.csv",
                    rows,
                    ["atom_index", "atom_symbol", "angular", "charge", "spin"],
                )
            )
    return files


def _loewdin_files(data: Dict[str, Any], directory: Path, stem: str, write_csv: WriteCSV) -> List[Path]:
    section = _get_final_population_section(data, "loewdin")
    files: List[Path] = []
    atoms = _get_pop_atoms(section)
    if atoms:
        rows = [{
            "index": atom.get("index"),
            "symbol": atom.get("symbol"),
            "charge": atom.get("charge"),
            "spin_population": atom.get("spin_population", ""),
        } for atom in atoms]
        files.append(
            write_csv(
                directory,
                f"{stem}_loewdin_charges.csv",
                rows,
                ["index", "symbol", "charge", "spin_population"],
            )
        )

    orbital_data = section.get("reduced_orbital_charges", [])
    if orbital_data:
        rows = []
        for atom in orbital_data:
            for contribution in atom.get("contributions", []):
                rows.append({
                    "atom_index": atom.get("index"),
                    "atom_symbol": atom.get("symbol"),
                    "angular": contribution.get("angular"),
                    "charge": contribution.get("charge"),
                })
        if rows:
            files.append(
                write_csv(
                    directory,
                    f"{stem}_loewdin_orb.csv",
                    rows,
                    ["atom_index", "atom_symbol", "angular", "charge"],
                )
            )
    return files


def _mayer_files(data: Dict[str, Any], directory: Path, stem: str, write_csv: WriteCSV) -> List[Path]:
    section = _get_final_mayer_section(data)
    files: List[Path] = []
    atoms = section.get("atoms", [])
    if atoms:
        rows = [{
            "index": atom.get("index"),
            "symbol": atom.get("symbol"),
            "NA": atom.get("NA"),
            "ZA": atom.get("ZA"),
            "QA": atom.get("QA"),
            "VA": atom.get("VA"),
            "BVA": atom.get("BVA"),
            "FA": atom.get("FA"),
        } for atom in atoms]
        files.append(
            write_csv(
                directory,
                f"{stem}_mayer_atoms.csv",
                rows,
                ["index", "symbol", "NA", "ZA", "QA", "VA", "BVA", "FA"],
            )
        )
    bonds = section.get("bond_orders", [])
    if bonds:
        files.append(
            write_csv(
                directory,
                f"{stem}_mayer_bonds.csv",
                bonds,
                ["atom_i", "symbol_i", "atom_j", "symbol_j", "bond_order"],
            )
        )
    return files


def _hirshfeld_files(data: Dict[str, Any], directory: Path, stem: str, write_csv: WriteCSV) -> List[Path]:
    section = _get_final_population_section(data, "hirshfeld")
    atoms = _get_pop_atoms(section)
    if not atoms:
        return []
    has_spin = any("spin_population" in atom for atom in atoms)
    rows = [{
        "index": atom.get("index"),
        "symbol": atom.get("symbol"),
        "charge": atom.get("charge"),
        "spin_population": atom.get("spin_population", "") if has_spin else "",
    } for atom in atoms]
    columns = ["index", "symbol", "charge"] + (["spin_population"] if has_spin else [])
    return [write_csv(directory, f"{stem}_hirshfeld.csv", rows, columns)]


def _mbis_files(data: Dict[str, Any], directory: Path, stem: str, write_csv: WriteCSV) -> List[Path]:
    section = _get_final_population_section(data, "mbis")
    atoms = _get_pop_atoms(section)
    files: List[Path] = []
    if atoms:
        has_spin = any("spin_population" in atom for atom in atoms)
        rows = [{
            "index": atom.get("index"),
            "symbol": atom.get("symbol"),
            "charge": atom.get("charge"),
            "population": atom.get("population", ""),
            "spin_population": atom.get("spin_population", "") if has_spin else "",
        } for atom in atoms]
        columns = ["index", "symbol", "charge", "population"] + (["spin_population"] if has_spin else [])
        files.append(write_csv(directory, f"{stem}_mbis.csv", rows, columns))

    valence_shell = section.get("valence_shell", [])
    if valence_shell:
        rows = [{
            "index": valence.get("index"),
            "symbol": valence.get("symbol"),
            "population": valence.get("population"),
            "width_au": valence.get("width_au"),
        } for valence in valence_shell]
        files.append(
            write_csv(
                directory,
                f"{stem}_mbis_valence.csv",
                rows,
                ["index", "symbol", "population", "width_au"],
            )
        )
    return files


def _chelpg_files(data: Dict[str, Any], directory: Path, stem: str, write_csv: WriteCSV) -> List[Path]:
    section = _get_final_population_section(data, "chelpg")
    atoms = _get_pop_atoms(section)
    if not atoms:
        return []
    rows = [{
        "index": atom.get("index"),
        "symbol": atom.get("symbol"),
        "charge": atom.get("charge"),
    } for atom in atoms]
    return [write_csv(directory, f"{stem}_chelpg.csv", rows, ["index", "symbol", "charge"])]


def _nbo_nao_files(data: Dict[str, Any], directory: Path, stem: str, write_csv: WriteCSV) -> List[Path]:
    return _write_nbo_nao_section(data, directory, stem, write_csv=write_csv)


def _nbo_npa_files(data: Dict[str, Any], directory: Path, stem: str, write_csv: WriteCSV) -> List[Path]:
    return _write_nbo_npa_section(data, directory, stem, write_csv=write_csv)


def _dipole_files(data: Dict[str, Any], directory: Path, stem: str, write_csv: WriteCSV) -> List[Path]:
    return _write_basic_dipole_section(data, directory, stem, write_csv=write_csv)


def _solvation_files(data: Dict[str, Any], directory: Path, stem: str, write_csv: WriteCSV) -> List[Path]:
    return _write_basic_solvation_section(data, directory, stem, write_csv=write_csv)


def _tddft_files(data: Dict[str, Any], directory: Path, stem: str, write_csv: WriteCSV) -> List[Path]:
    return _write_spectroscopy_tddft_section(
        data,
        directory,
        stem,
        write_csv=write_csv,
        bool_to_label=_bool_to_label,
    )


def _nbo_lewis_files(data: Dict[str, Any], directory: Path, stem: str, write_csv: WriteCSV) -> List[Path]:
    return _write_nbo_lewis_section(data, directory, stem, write_csv=write_csv)


def _nbo_e2_files(data: Dict[str, Any], directory: Path, stem: str, write_csv: WriteCSV) -> List[Path]:
    return _write_nbo_e2_section(data, directory, stem, write_csv=write_csv)


def _nbo_nlmo_hybridization_files(data: Dict[str, Any], directory: Path, stem: str, write_csv: WriteCSV) -> List[Path]:
    return _write_nbo_nlmo_hybridization_section(data, directory, stem, write_csv=write_csv)


def _nbo_nlmo_bond_order_files(data: Dict[str, Any], directory: Path, stem: str, write_csv: WriteCSV) -> List[Path]:
    return _write_nbo_nlmo_bond_order_section(data, directory, stem, write_csv=write_csv)


def _nbo_nlmo_steric_files(data: Dict[str, Any], directory: Path, stem: str, write_csv: WriteCSV) -> List[Path]:
    return _write_nbo_nlmo_steric_section(data, directory, stem, write_csv=write_csv)


def _epr_files(data: Dict[str, Any], directory: Path, stem: str, write_csv: WriteCSV) -> List[Path]:
    return _write_spectroscopy_epr_section(data, directory, stem, write_csv=write_csv)


def _wiberg_matrix_files(data: Dict[str, Any], directory: Path, stem: str, write_csv: WriteCSV) -> List[Path]:
    return _write_nbo_matrix_section(
        data,
        directory,
        stem,
        write_csv=write_csv,
        key="wiberg_matrix",
        filename_suffix="wiberg",
    )


def _nbi_matrix_files(data: Dict[str, Any], directory: Path, stem: str, write_csv: WriteCSV) -> List[Path]:
    return _write_nbo_matrix_section(
        data,
        directory,
        stem,
        write_csv=write_csv,
        key="nbi_matrix",
        filename_suffix="nbi",
    )


for order, key, renderer in (
    (10, "metadata", _metadata_files),
    (20, "geometry", _geometry_files),
    (30, "symmetry", _symmetry_files),
    (40, "orbital_energies", _orbital_energy_files),
    (50, "qro", _qro_files),
    (60, "mulliken", _mulliken_files),
    (70, "loewdin", _loewdin_files),
    (80, "mayer", _mayer_files),
    (90, "hirshfeld", _hirshfeld_files),
    (100, "mbis", _mbis_files),
    (110, "chelpg", _chelpg_files),
    (120, "nbo_nao", _nbo_nao_files),
    (130, "nbo_npa", _nbo_npa_files),
    (140, "dipole", _dipole_files),
    (150, "solvation", _solvation_files),
    (160, "tddft", _tddft_files),
    (170, "nbo_lewis", _nbo_lewis_files),
    (180, "nbo_e2", _nbo_e2_files),
    (190, "nbo_nlmo_hybridization", _nbo_nlmo_hybridization_files),
    (200, "nbo_nlmo_bond_order", _nbo_nlmo_bond_order_files),
    (210, "nbo_nlmo_steric", _nbo_nlmo_steric_files),
    (220, "epr", _epr_files),
    (230, "wiberg_matrix", _wiberg_matrix_files),
    (240, "nbi_matrix", _nbi_matrix_files),
):
    register_csv_section_plugin(
        CSVSectionPlugin(
            key=key,
            order=order,
            render_files=renderer,
        )
    )
