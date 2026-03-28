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

    assert data["metadata"]["point_group"] == "Cs"
    assert data["metadata"]["orbital_irrep_group"] == "Cs"
    assert excited_states[0]["symmetry"] == 'A"'
    assert excited_states[0]["transitions"]
    assert excited_states[0]["transitions"][0]["from_orbital"] == "23a"
    assert excited_states[0]["transitions"][0]["to_orbital"] == "24a"

    markdown_path = tmp_path / "tddft_sym.md"
    tddft_sym_parser.to_markdown(markdown_path)
    text = markdown_path.read_text(encoding="utf-8")

    assert "## TDDFT Excited States" in text
    assert "| State | Symmetry | E (eV)" in text
    assert '| 1     | A"' in text
    assert "23a (17-A') -> 24a (8-A\")" in text
