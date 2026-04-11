from __future__ import annotations

import re
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


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def _find_last_exact(lines: list[str], label: str) -> int:
    target = label.strip().upper()
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip().upper() == target:
            return i
    raise AssertionError(f"Could not find exact label: {label}")


def _extract_last_cartesian_angstrom(lines: list[str]) -> list[dict[str, float | str]]:
    idx = _find_last_exact(lines, "CARTESIAN COORDINATES (ANGSTROEM)")
    atoms: list[dict[str, float | str]] = []
    for line in lines[idx + 2:]:
        match = re.match(r"\s+(\w+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", line)
        if match:
            atoms.append(
                {
                    "symbol": match.group(1),
                    "x_ang": float(match.group(2)),
                    "y_ang": float(match.group(3)),
                    "z_ang": float(match.group(4)),
                }
            )
        elif atoms and not line.strip():
            break
    return atoms


def _extract_last_orbital_summary(lines: list[str]) -> dict[str, float | int]:
    idx = _find_last_exact(lines, "ORBITAL ENERGIES")
    orbitals: list[dict[str, float | int]] = []
    for line in lines[idx + 1:]:
        match = re.match(
            r"\s+(\d+)\s+([\d.]+)\s+([-\d.]+)\s+([-\d.]+)(?:\s+\S+)?",
            line,
        )
        if match:
            orbitals.append(
                {
                    "index": int(match.group(1)),
                    "occupation": float(match.group(2)),
                    "energy_Eh": float(match.group(3)),
                    "energy_eV": float(match.group(4)),
                }
            )
        elif orbitals and (not line.strip() or "MOLECULAR ORBITALS" in line or "---" in line):
            break

    homo = None
    lumo = None
    for orbital in orbitals:
        if orbital["occupation"] > 0.5:
            homo = orbital
        elif homo is not None and lumo is None:
            lumo = orbital
            break

    assert homo is not None
    assert lumo is not None

    return {
        "HOMO_index": int(homo["index"]),
        "HOMO_energy_Eh": float(homo["energy_Eh"]),
        "HOMO_energy_eV": float(homo["energy_eV"]),
        "LUMO_index": int(lumo["index"]),
        "LUMO_energy_Eh": float(lumo["energy_Eh"]),
        "LUMO_energy_eV": float(lumo["energy_eV"]),
        "HOMO_LUMO_gap_eV": float(lumo["energy_eV"]) - float(homo["energy_eV"]),
    }


def _extract_last_mulliken_charge(lines: list[str], atom_index: int) -> float:
    idx = _find_last_exact(lines, "MULLIKEN ATOMIC CHARGES")
    charges: dict[int, float] = {}
    for line in lines[idx + 2:]:
        match = re.match(r"\s+(\d+)\s+(\w+)\s*:\s+([-\d.]+)", line)
        if match:
            charges[int(match.group(1))] = float(match.group(3))
        elif charges and ("Sum of atomic charges" in line or not line.strip()):
            break
    return charges[atom_index]


def _extract_last_mayer_bond_order(
    lines: list[str],
    atom_i: int,
    atom_j: int,
) -> float:
    idx = max(i for i, line in enumerate(lines) if "MAYER POPULATION ANALYSIS" in line.upper())
    idx_bo = next(
        i for i in range(idx, len(lines))
        if "Mayer bond orders larger than" in lines[i]
    )
    bond_orders: dict[tuple[int, int], float] = {}
    for line in lines[idx_bo + 1:]:
        found = False
        for match in re.finditer(
            r"B\(\s*(\d+)-(\w+)\s*,\s*(\d+)-(\w+)\s*\)\s*:\s*([-\d.]+)",
            line,
        ):
            key = tuple(sorted((int(match.group(1)), int(match.group(3)))))
            bond_orders[key] = float(match.group(5))
            found = True
        if bond_orders and not found and not line.strip():
            break
    return bond_orders[tuple(sorted((atom_i, atom_j)))]


def _extract_last_total_dipole(lines: list[str]) -> dict[str, float]:
    idx = _find_last_exact(lines, "DIPOLE MOMENT")
    for line in lines[idx: idx + 40]:
        match = re.search(
            r"Total Dipole Moment\s+:\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)",
            line,
        )
        if match:
            return {
                "x": float(match.group(1)),
                "y": float(match.group(2)),
                "z": float(match.group(3)),
            }
    raise AssertionError("Could not extract final dipole moment")


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


@pytest.fixture(scope="module")
def s0_final_step_parser() -> ORCAParser:
    return _parse_with_sections(S0_OUT, ["charges", "population", "mos", "dipole"])


@pytest.fixture(scope="module")
def s1_final_step_parser() -> ORCAParser:
    return _parse_with_sections(S1_OUT, ["charges", "population", "mos", "dipole"])


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
        [goat_full_parser, s0_parser, s1_parser, tddft_parser],
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
    assert "Showing all 1147 conformers." in goat_markdown_text
    assert "| 100       | 2.1700" in goat_markdown_text
    assert "lowest_energy_conformer_Eh" in goat_summary_text
    assert "-134.77823" in goat_summary_text
    assert "conformer,relative_energy_kcal_mol,degeneracy,percent_total,percent_cumulative" in goat_ensemble_text
    assert "1146,5.997,1,0.0,100.0" in goat_ensemble_text

    assert "| GOAT/RDB_LMMPa.out              | XTB2" in comparison_text
    assert "| S0/RDB_LMMP_S0_CH2Cl2.out       | LibXC(wB97X-D4)" in comparison_text
    assert "| S1/RDB_LMMP_S1_CH2Cl2.out       | LibXC(wB97X-D4)" in comparison_text
    assert "| TDDFT/RDB_LMMP_TDDFT_CH2Cl2.out | SOS-wPBEPP86" in comparison_text
    assert "Showing all 1147 conformers (all within dE <= 10.0000 kcal/mol)." in comparison_text
    assert "-134.7631015236" in comparison_text
    assert "-2089.8906388712" in comparison_text
    assert "-2089.7379116584" in comparison_text
    assert "-2082.9741422716" in comparison_text


def test_goat_markdown_cutoff_can_be_overridden(goat_full_parser: ORCAParser) -> None:
    tmp_dir = REPO_ROOT / ".pytest_tmp" / f"rdb_lmmp_goat_cutoff_{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    limited_markdown_path = tmp_dir / "rdb_lmmp_goat_limited.md"
    goat_full_parser.to_markdown(
        limited_markdown_path,
        goat_max_relative_energy_kcal_mol=3.0,
    )
    limited_markdown_text = limited_markdown_path.read_text(encoding="utf-8")

    compare_full_path = tmp_dir / "comparison_full_goat.md"
    ORCAParser.compare(
        [goat_full_parser],
        compare_full_path,
        goat_max_relative_energy_kcal_mol=None,
    )
    compare_full_text = compare_full_path.read_text(encoding="utf-8")

    assert "Showing 229 of 1147 conformers with dE <= 3.0000 kcal/mol." in limited_markdown_text
    assert "| 228       | 2.9970" in limited_markdown_text
    assert "| 229       | 3.0260" not in limited_markdown_text
    assert "Showing all 1147 conformers." in compare_full_text


def test_rdb_lmmp_s1_uses_final_geometry_step_for_geometry_dependent_properties(
    s0_final_step_parser: ORCAParser,
    s1_final_step_parser: ORCAParser,
) -> None:
    s1_lines = _read_lines(S1_OUT)
    raw_s1_geometry = _extract_last_cartesian_angstrom(s1_lines)
    raw_s1_orbitals = _extract_last_orbital_summary(s1_lines)
    raw_s1_mulliken_c0 = _extract_last_mulliken_charge(s1_lines, 0)
    raw_s1_mayer_01 = _extract_last_mayer_bond_order(s1_lines, 0, 1)
    raw_s1_dipole = _extract_last_total_dipole(s1_lines)

    s0_geometry = s0_final_step_parser.data["geometry"]["cartesian_angstrom"]
    s1_geometry = s1_final_step_parser.data["geometry"]["cartesian_angstrom"]
    s0_orbitals = s0_final_step_parser.data["orbital_energies"]
    s1_orbitals = s1_final_step_parser.data["orbital_energies"]
    s1_mulliken = s1_final_step_parser.data["mulliken"]["atomic_charges"]
    s1_mayer = s1_final_step_parser.data["mayer"]["bond_orders"]
    s1_dipole = s1_final_step_parser.data["dipole"]["total_dipole_au"]

    assert len(raw_s1_geometry) == len(s1_geometry)
    assert s1_geometry[0]["symbol"] == raw_s1_geometry[0]["symbol"] == "C"
    assert s1_geometry[0]["x_ang"] == pytest.approx(raw_s1_geometry[0]["x_ang"])
    assert s1_geometry[0]["y_ang"] == pytest.approx(raw_s1_geometry[0]["y_ang"])
    assert s1_geometry[0]["z_ang"] == pytest.approx(raw_s1_geometry[0]["z_ang"])
    assert s1_geometry[1]["x_ang"] == pytest.approx(raw_s1_geometry[1]["x_ang"])
    assert s1_geometry[2]["z_ang"] == pytest.approx(raw_s1_geometry[2]["z_ang"])

    assert s1_geometry[0]["x_ang"] != pytest.approx(s0_geometry[0]["x_ang"])
    assert s1_geometry[0]["y_ang"] != pytest.approx(s0_geometry[0]["y_ang"])
    assert s1_geometry[0]["z_ang"] != pytest.approx(s0_geometry[0]["z_ang"])

    assert s1_orbitals["HOMO_index"] == raw_s1_orbitals["HOMO_index"]
    assert s1_orbitals["LUMO_index"] == raw_s1_orbitals["LUMO_index"]
    assert s1_orbitals["HOMO_energy_eV"] == pytest.approx(raw_s1_orbitals["HOMO_energy_eV"])
    assert s1_orbitals["LUMO_energy_eV"] == pytest.approx(raw_s1_orbitals["LUMO_energy_eV"])
    assert s1_orbitals["HOMO_LUMO_gap_eV"] == pytest.approx(raw_s1_orbitals["HOMO_LUMO_gap_eV"])

    assert s1_orbitals["HOMO_energy_eV"] != pytest.approx(s0_orbitals["HOMO_energy_eV"])
    assert s1_orbitals["LUMO_energy_eV"] != pytest.approx(s0_orbitals["LUMO_energy_eV"])

    assert s1_mulliken[0]["charge"] == pytest.approx(raw_s1_mulliken_c0)
    assert any(
        tuple(sorted((bond["atom_i"], bond["atom_j"]))) == (0, 1)
        and bond["bond_order"] == pytest.approx(raw_s1_mayer_01)
        for bond in s1_mayer
    )
    assert s1_dipole["x"] == pytest.approx(raw_s1_dipole["x"])
    assert s1_dipole["y"] == pytest.approx(raw_s1_dipole["y"])
    assert s1_dipole["z"] == pytest.approx(raw_s1_dipole["z"])
