from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from orca_parser import ORCAParser


REPO_ROOT = Path(__file__).resolve().parents[1]
RDB_LMMP_ROOT = REPO_ROOT / "sample_outs" / "RDB_LMMP" / "a" / "CH2Cl2"
GOAT_OUT = RDB_LMMP_ROOT / "GOAT" / "RDB_LMMPa.out"
S0_OUT = RDB_LMMP_ROOT / "S0" / "RDB_LMMP_S0_CH2Cl2.out"
S1_OUT = RDB_LMMP_ROOT / "S1" / "RDB_LMMP_S1_CH2Cl2.out"
TDDFT_OUT = RDB_LMMP_ROOT / "TDDFT" / "RDB_LMMP_TDDFT_CH2Cl2.out"


def _parse_quick(path: Path) -> ORCAParser:
    parser = ORCAParser(path)
    parser.parse(sections=["scf"])
    return parser


def _parse_with_sections(path: Path, sections: list[str]) -> ORCAParser:
    parser = ORCAParser(path)
    parser.parse(sections=sections)
    return parser


@pytest.fixture(scope="module")
def goat_parser() -> ORCAParser:
    return _parse_quick(GOAT_OUT)


@pytest.fixture(scope="module")
def goat_full_parser() -> ORCAParser:
    return _parse_with_sections(GOAT_OUT, ["goat"])


@pytest.fixture(scope="module")
def s0_parser() -> ORCAParser:
    return _parse_quick(S0_OUT)


@pytest.fixture(scope="module")
def s1_parser() -> ORCAParser:
    return _parse_quick(S1_OUT)


@pytest.fixture(scope="module")
def tddft_parser() -> ORCAParser:
    return _parse_quick(TDDFT_OUT)


def test_rdb_lmmp_goat_uses_input_echo_for_method_energy_and_id(
    goat_parser: ORCAParser,
) -> None:
    meta = goat_parser.data["metadata"]
    scf = goat_parser.data["scf"]

    assert meta["calculation_type"] == "GOAT Conformer Search"
    assert meta["method"] == "XTB2"
    assert meta["level_of_theory"] == "XTB2"
    assert meta["charge"] == 0
    assert meta["multiplicity"] == 1
    assert meta["job_id"] == "sample_outs/RDB_LMMP/a/CH2Cl2/GOAT/RDB_LMMPa.out"
    assert scf["final_single_point_energy_Eh"] == pytest.approx(-134.763101523590)


def test_rdb_lmmp_goat_parses_final_ensemble_and_thermochemistry(
    goat_full_parser: ORCAParser,
) -> None:
    goat = goat_full_parser.data["goat"]

    assert goat["global_minimum_found"] is True
    assert goat["global_minimum_xyz_file"] == "RDB_LMMPa.globalminimum.xyz"
    assert goat["final_ensemble_xyz_file"] == "RDB_LMMPa.finalensemble.xyz"
    assert goat["n_conformers"] == 1147
    assert goat["global_minimum_conformer"] == 0
    assert goat["conformers_below_energy_window"] == 229
    assert goat["conformer_energy_window_kcal_mol"] == pytest.approx(3.0)
    assert goat["lowest_energy_conformer_Eh"] == pytest.approx(-134.778230)
    assert goat["sconf_cal_molK"] == pytest.approx(9.95)
    assert goat["gconf_kcal_mol"] == pytest.approx(-1.50)
    assert goat["top_population_percent"] == pytest.approx(7.97)
    assert goat["final_cumulative_percent"] == pytest.approx(100.0)
    assert goat["ensemble"][0]["conformer"] == 0
    assert goat["ensemble"][0]["relative_energy_kcal_mol"] == pytest.approx(0.0)
    assert goat["ensemble"][0]["percent_total"] == pytest.approx(7.97)
    assert goat["ensemble"][-1]["conformer"] == 1146
    assert goat["ensemble"][-1]["relative_energy_kcal_mol"] == pytest.approx(5.997)


def test_rdb_lmmp_tddft_prefers_input_model_chemistry_and_last_final_energy(
    tddft_parser: ORCAParser,
) -> None:
    meta = tddft_parser.data["metadata"]
    scf = tddft_parser.data["scf"]

    assert meta["calculation_type"] == "Single Point"
    assert meta["method"] == "SOS-wPBEPP86"
    assert meta["basis_set"] == "Def2-TZVP"
    assert meta["aux_basis_set"] == "Def2-TZVP/C"
    assert meta["level_of_theory"] == "SOS-wPBEPP86/Def2-TZVP"
    assert meta["job_id"] == (
        "sample_outs/RDB_LMMP/a/CH2Cl2/TDDFT/RDB_LMMP_TDDFT_CH2Cl2.out"
    )
    assert scf["final_single_point_energy_Eh"] == pytest.approx(-2082.974142271619)


def test_rdb_lmmp_excited_state_opt_is_identified_from_input(
    s1_parser: ORCAParser,
) -> None:
    meta = s1_parser.data["metadata"]
    scf = s1_parser.data["scf"]
    excited = meta["excited_state_optimization"]

    assert meta["calculation_type"] == "Excited-State Geometry Optimization"
    assert meta["method"] == "LibXC(wB97X-D4)"
    assert meta["basis_set"] == "Def2-TZVP"
    assert excited["input_block"] == "tddft"
    assert excited["target_root"] == 1
    assert excited["target_state_label"] == "S1"
    assert excited["followiroot"] is False
    assert scf["final_single_point_energy_Eh"] == pytest.approx(-2089.737911658443)


def test_rdb_lmmp_markdown_and_comparison_use_input_driven_labels(
    goat_parser: ORCAParser,
    goat_full_parser: ORCAParser,
    s0_parser: ORCAParser,
    s1_parser: ORCAParser,
    tddft_parser: ORCAParser,
) -> None:
    tmp_dir = REPO_ROOT / ".pytest_tmp" / f"rdb_lmmp_regressions_{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = tmp_dir / "rdb_lmmp_tddft.md"
    comparison_path = tmp_dir / "comparison.md"

    tddft_parser.to_markdown(markdown_path)
    ORCAParser.compare(
        [goat_parser, s0_parser, s1_parser, tddft_parser],
        comparison_path,
    )

    markdown_text = markdown_path.read_text(encoding="utf-8")
    comparison_text = comparison_path.read_text(encoding="utf-8")
    goat_markdown_path = tmp_dir / "rdb_lmmp_goat.md"
    goat_full_parser.to_markdown(goat_markdown_path)
    goat_markdown_text = goat_markdown_path.read_text(encoding="utf-8")
    goat_csv_dir = tmp_dir / "goat_csv"
    goat_full_parser.to_csv(goat_csv_dir)
    goat_summary_text = (goat_csv_dir / "RDB_LMMPa_goat_summary.csv").read_text(encoding="utf-8")
    goat_ensemble_text = (goat_csv_dir / "RDB_LMMPa_goat_ensemble.csv").read_text(encoding="utf-8")

    assert "`SOS-wPBEPP86/Def2-TZVP`" in markdown_text
    assert "id: `sample_outs/RDB_LMMP/a/CH2Cl2/TDDFT/RDB_LMMP_TDDFT_CH2Cl2.out`" in markdown_text
    assert "**Energy:** -2082.9741422716 Eh" in markdown_text
    assert "## GOAT Conformer Search" in goat_markdown_text
    assert "**Conformers:** 1147" in goat_markdown_text
    assert "**Sconf:** 9.95 cal/(molK)" in goat_markdown_text
    assert "RDB_LMMPa.globalminimum.xyz" in goat_markdown_text
    assert "Showing first 15 and last 5 of 1147 conformers" in goat_markdown_text
    assert "lowest_energy_conformer_Eh" in goat_summary_text
    assert "-134.77823" in goat_summary_text
    assert "conformer,relative_energy_kcal_mol,degeneracy,percent_total,percent_cumulative" in goat_ensemble_text
    assert "1146,5.997,1,0.0,100.0" in goat_ensemble_text

    assert "| GOAT/RDB_LMMPa.out              | XTB2" in comparison_text
    assert "| S0/RDB_LMMP_S0_CH2Cl2.out       | LibXC(wB97X-D4)" in comparison_text
    assert "| S1/RDB_LMMP_S1_CH2Cl2.out       | LibXC(wB97X-D4)" in comparison_text
    assert "| TDDFT/RDB_LMMP_TDDFT_CH2Cl2.out | SOS-wPBEPP86" in comparison_text
    assert "-134.7631015236" in comparison_text
    assert "-2089.8906388712" in comparison_text
    assert "-2089.7379116584" in comparison_text
    assert "-2082.9741422716" in comparison_text
