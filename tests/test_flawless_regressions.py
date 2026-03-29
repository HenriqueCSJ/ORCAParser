from __future__ import annotations

import csv
from pathlib import Path

import pytest

from orca_parser import ORCAParser
from orca_parser.parser import is_auxiliary_orca_file


REPO_ROOT = Path(__file__).resolve().parents[1]
SP_SYM_OUT = REPO_ROOT / "sample_outs" / "N" / "SP_Sym" / "NC.out"
SP_SYM_DELTA_OUT = REPO_ROOT / "sample_outs" / "N" / "SP_Sym_Delta" / "NC.out"
TDDFT_SYM_OUT = REPO_ROOT / "sample_outs" / "R2SCAN-QIDH" / "F3CNO.out"
TDDFT_NBO_OUT = REPO_ROOT / "sample_outs" / "Diox" / "TDDFT" / "RDB_vinyl_a_TDDFT_Diox.out"
SCAN_OUT = REPO_ROOT / "sample_outs" / "Scan" / "F3CNO.out"


@pytest.fixture(scope="session")
def sp_sym_parser() -> ORCAParser:
    parser = ORCAParser(SP_SYM_OUT)
    parser.parse()
    return parser


@pytest.fixture(scope="session")
def sp_sym_delta_parser() -> ORCAParser:
    parser = ORCAParser(SP_SYM_DELTA_OUT)
    parser.parse()
    return parser


@pytest.fixture(scope="session")
def tddft_sym_parser() -> ORCAParser:
    parser = ORCAParser(TDDFT_SYM_OUT)
    parser.parse()
    return parser


@pytest.fixture(scope="session")
def tddft_nbo_parser() -> ORCAParser:
    parser = ORCAParser(TDDFT_NBO_OUT)
    parser.parse()
    return parser


@pytest.fixture(scope="session")
def surface_scan_parser() -> ORCAParser:
    parser = ORCAParser(SCAN_OUT)
    parser.parse()
    return parser


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_auxiliary_atom_outputs_are_rejected(tmp_path: Path) -> None:
    aux_path = tmp_path / "BiC_atom83.out"
    aux_path.write_text("stub helper file\n", encoding="utf-8")

    assert is_auxiliary_orca_file(aux_path) is True
    with pytest.raises(ValueError, match="Auxiliary ORCA atom/ECP files"):
        ORCAParser(aux_path).parse()


def test_use_sym_metadata_and_irreps_are_preserved(sp_sym_parser: ORCAParser) -> None:
    data = sp_sym_parser.data
    meta = data["metadata"]
    sym = meta["symmetry"]
    geom = data["geometry"]
    orbitals = data["orbital_energies"]

    assert meta["point_group"] == "Ih"
    assert meta["reduced_point_group"] == "D2h"
    assert meta["orbital_irrep_group"] == "D2h"
    assert sym["use_sym"] is True
    assert sym["n_irreps"] == 8
    assert sym["initial_guess_irrep"] == "4-B3u"
    assert geom["symmetry_perfected_point_group"] == "Ih"
    assert len(geom["symmetry_cartesian_angstrom"]) == 61
    assert orbitals["alpha_occupied_per_irrep"]["B3u"] == 25
    assert orbitals["beta_occupied_per_irrep"]["B3u"] == 24


def test_markdown_calls_out_symmetry_and_deltascf(
    tmp_path: Path,
    sp_sym_parser: ORCAParser,
    sp_sym_delta_parser: ORCAParser,
) -> None:
    sym_path = tmp_path / "sp_sym.md"
    delta_path = tmp_path / "sp_sym_delta.md"

    sp_sym_parser.to_markdown(sym_path)
    sp_sym_delta_parser.to_markdown(delta_path)

    sym_text = sym_path.read_text(encoding="utf-8")
    delta_text = delta_path.read_text(encoding="utf-8")

    assert "## Symmetry" in sym_text
    assert "Ih -> D2h" in sym_text
    assert "UseSym" in sym_text
    assert "Symmetry-perfected geometry" in sym_text
    assert "Initial Guess / Symmetry Cleanup" in sym_text
    assert "## Irrep-Resolved Orbital Window" in sym_text
    assert "## Reduced Orbital Populations" in sym_text
    assert "Mulliken shell totals — charge" in sym_text

    assert "state=DeltaSCF excited-state" in delta_text
    assert "## DeltaSCF / Excited-State Target" in delta_text
    assert "ALPHACONF 0,1" in delta_text
    assert "not a ground-state single-point" in delta_text


def test_comparison_markdown_distinguishes_ground_and_excited_states(
    tmp_path: Path,
    sp_sym_parser: ORCAParser,
    sp_sym_delta_parser: ORCAParser,
) -> None:
    comparison_path = tmp_path / "comparison.md"
    ORCAParser.compare([sp_sym_parser, sp_sym_delta_parser], comparison_path)
    text = comparison_path.read_text(encoding="utf-8")

    assert "## Methods" in text
    assert "electronic state" in text
    assert "DeltaSCF excited-state" in text
    assert "## DeltaSCF" in text
    assert "## Symmetry Setup" in text
    assert "## Symmetry Occupations" in text


def test_csv_exports_include_metadata_symmetry_and_deltascf(
    tmp_path: Path,
    sp_sym_parser: ORCAParser,
    sp_sym_delta_parser: ORCAParser,
) -> None:
    sym_dir = tmp_path / "sym"
    delta_dir = tmp_path / "delta"
    sp_sym_parser.to_csv(sym_dir)
    sp_sym_delta_parser.to_csv(delta_dir)

    sym_metadata = _read_csv_rows(sym_dir / "NC_metadata.csv")
    sym_summary = _read_csv_rows(sym_dir / "NC_symmetry.csv")
    sym_irreps = _read_csv_rows(sym_dir / "NC_symmetry_irreps.csv")
    sym_geometry = _read_csv_rows(sym_dir / "NC_geometry_symmetry.csv")

    assert sym_metadata[0]["point_group"] == "Ih"
    assert sym_metadata[0]["reduced_point_group"] == "D2h"
    assert sym_metadata[0]["use_sym"] == "yes"
    assert sym_summary[0]["initial_guess_irrep"] == "4-B3u"
    assert any(row["irrep"] == "B3u" and row["occupied_alpha"] == "25" for row in sym_irreps)
    assert len(sym_geometry) == 61
    assert all(row["point_group"] == "Ih" for row in sym_geometry)

    delta_metadata = _read_csv_rows(delta_dir / "NC_metadata.csv")
    delta_summary = _read_csv_rows(delta_dir / "NC_deltascf.csv")
    delta_targets = _read_csv_rows(delta_dir / "NC_deltascf_occupations.csv")

    assert delta_metadata[0]["calculation_type"] == "DeltaSCF"
    assert delta_metadata[0]["electronic_state"] == "DeltaSCF excited-state"
    assert delta_summary[0]["target_configuration"] == "ALPHACONF 0,1"
    assert delta_summary[0]["aufbau_metric"] == "MOM"
    assert delta_summary[0]["keep_initial_reference"] == "yes"
    assert any(row["spin"] == "alpha" and row["slot"] == "6" and row["occupation"] == "0.0" for row in delta_targets)


def test_reduced_orbital_populations_are_parsed_for_charge_and_spin(
    sp_sym_parser: ORCAParser,
) -> None:
    data = sp_sym_parser.data

    mulliken_charge = data["mulliken"]["reduced_orbital_charges"]
    mulliken_spin = data["mulliken"]["reduced_orbital_spin_populations"]
    loewdin_charge = data["loewdin"]["reduced_orbital_charges"]
    loewdin_spin = data["loewdin"]["reduced_orbital_spin_populations"]

    assert len(mulliken_charge) == 61
    assert len(mulliken_spin) == 61
    assert len(loewdin_charge) == 61
    assert len(loewdin_spin) == 61

    assert mulliken_charge[0]["symbol"] == "N"
    assert "p" in mulliken_charge[0]["shell_totals"]
    assert "p" in mulliken_spin[0]["shell_totals"]


def test_tddft_symmetry_and_dominant_excitations_are_preserved(
    tmp_path: Path,
    tddft_sym_parser: ORCAParser,
) -> None:
    data = tddft_sym_parser.data
    excited_states = data["tddft"]["final_excited_state_block"]["states"]
    dipole = data["dipole"]
    spectra = data["tddft"]["spectra"]

    assert data["metadata"]["point_group"] == "Cs"
    assert data["metadata"]["orbital_irrep_group"] == "Cs"
    assert excited_states[0]["symmetry"] == 'A"'
    assert excited_states[0]["transitions"]
    assert excited_states[0]["transitions"][0]["from_orbital"] == "23a"
    assert excited_states[0]["transitions"][0]["to_orbital"] == "24a"
    assert dipole["magnitude_Debye"] == pytest.approx(0.276865075)
    assert dipole["total_dipole_au"]["x"] == pytest.approx(0.099935375)
    assert dipole["rotational_constants_cm1"][0] == pytest.approx(0.185875)
    assert spectra["absorption_electric_dipole"]["transition_count"] == 80
    assert spectra["cd_electric_dipole"]["transition_count"] == 80
    assert spectra["cd_velocity_dipole"]["transition_count"] == 80

    markdown_path = tmp_path / "tddft_sym.md"
    tddft_sym_parser.to_markdown(markdown_path)
    text = markdown_path.read_text(encoding="utf-8")

    assert "## Dipole Moment" in text
    assert "**Total magnitude:** 0.2769 D | 0.108925 a.u." in text
    assert "## TDDFT Excited States" in text
    assert "| State | Symmetry | E (eV)" in text
    assert '| 1     | A"' in text
    assert "23a (17-A') -> 24a (8-A\")" in text
    assert "### Absorption Spectrum (Electric Dipole)" in text
    assert "### Absorption Spectrum (Velocity Dipole)" in text
    assert "### CD Spectrum (Electric Dipole)" in text
    assert "### CD Spectrum (Velocity Dipole)" in text
    assert "0-1A' -> 1-1A\"" in text


def test_nbo_electron_configurations_and_cmo_indices_are_aligned(
    tddft_nbo_parser: ORCAParser,
) -> None:
    data = tddft_nbo_parser.data
    nbo = data["nbo"]

    first_config = nbo["electron_configurations"][0]
    assert first_config["symbol"] == "C"
    assert first_config["index"] == 1
    assert first_config["atom_nbo_index"] == 1
    assert first_config["atom_orca_index"] == 0

    cmo_lookup = {
        entry["nbo_mo_index"]: entry
        for entry in nbo["cmo_analysis"]
    }
    homo_like = cmo_lookup[110]
    assert homo_like["orca_orbital_index"] == 109

    lead_labels = [contribution.get("character_label") for contribution in homo_like["nbo_contributions"][:3]]
    assert lead_labels[0] == "n(S17)"
    assert lead_labels[1] == "pi(C7-C8)"
    assert homo_like["nbo_contributions"][0]["approx_percent"] == pytest.approx(100.0 * 0.534 * 0.534)

    lumo_like = cmo_lookup[111]
    assert lumo_like["orca_orbital_index"] == 110
    assert lumo_like["nbo_contributions"][0]["character_label"] == "pi*(C7-C8)"


def test_markdown_renders_natural_electron_configurations_and_cmo_character(
    tmp_path: Path,
    tddft_nbo_parser: ORCAParser,
) -> None:
    markdown_path = tmp_path / "tddft_nbo.md"
    tddft_nbo_parser.to_markdown(markdown_path)
    text = markdown_path.read_text(encoding="utf-8")

    assert "### Natural Electron Configuration" in text
    assert "NBO prints atom numbers as 1-based" in text
    assert "NBO atom #" in text
    assert "ORCA atom idx" in text
    assert "Configuration" in text
    assert "### Canonical MO Character (NBO CMO)" in text
    assert "CMO MO N corresponds to ORCA MO N-1" in text
    assert "ORCA MO" in text
    assert "CMO MO" in text
    assert "Leading character" in text
    assert "Top contributions" in text
    assert "n(S17)" in text
    assert "pi*(C7-C8)" in text
    assert "CMO/NBO character" in text


def test_surface_scan_is_parsed_without_fake_geom_opt(
    surface_scan_parser: ORCAParser,
) -> None:
    data = surface_scan_parser.data
    scan = data["surface_scan"]
    parameter = scan["parameters"][0]
    first_step = scan["steps"][0]
    last_step = scan["steps"][-1]

    assert data["metadata"]["calculation_type"] == "Relaxed Surface Scan"
    assert data["context"]["is_surface_scan"] is True
    assert "geom_opt" not in data

    assert scan["mode"] == "single"
    assert scan["n_parameters"] == 1
    assert scan["n_constrained_optimizations"] == 37
    assert parameter["kind"] == "D"
    assert parameter["atoms"] == [5, 2, 0, 4]
    assert parameter["start"] == pytest.approx(-180.0)
    assert parameter["end"] == pytest.approx(360.0)
    assert parameter["steps"] == 37

    assert len(scan["steps"]) == 37
    assert first_step["coordinate_values"][0] == pytest.approx(-180.0)
    assert first_step["actual_energy_Eh"] == pytest.approx(-467.64138313)
    assert first_step["optimized_xyz_file"] == "F3CNO.001.xyz"
    assert last_step["coordinate_values"][0] == pytest.approx(360.0)
    assert last_step["actual_energy_Eh"] == pytest.approx(-467.64286504)

    sidecars = scan["sidecar_files"]
    assert sidecars["actual_surface_dat"].endswith("F3CNO.relaxscanact.dat")
    assert sidecars["scf_surface_dat"].endswith("F3CNO.relaxscanscf.dat")
    assert sidecars["allxyz"].endswith("F3CNO.allxyz")
    assert sidecars["allxyz_frame_count"] == 37


def test_surface_scan_markdown_and_comparison_render(
    tmp_path: Path,
    surface_scan_parser: ORCAParser,
) -> None:
    markdown_path = tmp_path / "scan.md"
    comparison_path = tmp_path / "scan_comparison.md"

    surface_scan_parser.to_markdown(markdown_path)
    ORCAParser.compare([surface_scan_parser], comparison_path)

    text = markdown_path.read_text(encoding="utf-8")
    comparison = comparison_path.read_text(encoding="utf-8")

    assert "## Relaxed Surface Scan" in text
    assert "**Constrained optimizations:** 37" in text
    assert "D(5,2,0,4)" in text
    assert "F3CNO.relaxscanact.dat" in text
    assert "**Surface Profile**" in text
    assert "-180.0000" in text
    assert "360.0000" in text
    assert "F3CNO.037.xyz" in text

    assert "## Surface Scans" in comparison
    assert "single" in comparison
    assert "37" in comparison
    assert "D(5,2,0,4) -180.0000->360.0000 (37)" in comparison


def test_surface_scan_csv_exports(
    tmp_path: Path,
    surface_scan_parser: ORCAParser,
) -> None:
    out_dir = tmp_path / "scan_csv"
    surface_scan_parser.to_csv(out_dir)

    metadata_rows = _read_csv_rows(out_dir / "F3CNO_metadata.csv")
    parameter_rows = _read_csv_rows(out_dir / "F3CNO_surface_scan_parameters.csv")
    scan_rows = _read_csv_rows(out_dir / "F3CNO_surface_scan.csv")
    summary_rows = _read_csv_rows(out_dir / "F3CNO_surface_scan_summary.csv")

    assert metadata_rows[0]["is_surface_scan"] == "yes"
    assert metadata_rows[0]["scan_mode"] == "single"
    assert metadata_rows[0]["scan_parameters"] == "1"
    assert metadata_rows[0]["scan_steps"] == "37"

    assert parameter_rows[0]["label"] == "D(5,2,0,4)"
    assert parameter_rows[0]["kind"] == "D"
    assert parameter_rows[0]["atoms"] == "5,2,0,4"
    assert parameter_rows[0]["steps"] == "37"

    assert len(scan_rows) == 37
    assert scan_rows[0]["coord_1_value"] == "-180.0"
    assert scan_rows[0]["optimized_xyz_file"] == "F3CNO.001.xyz"
    assert scan_rows[-1]["coord_1_value"] == "360.0"
    assert scan_rows[-1]["optimized_xyz_file"] == "F3CNO.037.xyz"

    assert summary_rows[0]["allxyz_frame_count"] == "37"
