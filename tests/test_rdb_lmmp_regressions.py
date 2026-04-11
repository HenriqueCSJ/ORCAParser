from __future__ import annotations

import csv
import re
import uuid
from pathlib import Path

import pytest

from orca_parser import ORCAParser


REPO_ROOT = Path(__file__).resolve().parents[1]
RDB_LMMP_ROOT = REPO_ROOT / "sample_outs" / "RDB_LMMP" / "a" / "CH2Cl2"
SCAN_OUT = REPO_ROOT / "sample_outs" / "Scan" / "F3CNO.out"
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


@pytest.fixture(scope="module")
def scan_parser() -> ORCAParser:
    return _parse_with_sections(SCAN_OUT, ["scan"])


@pytest.fixture(scope="module")
def s0_opt_parser() -> ORCAParser:
    return _parse_with_sections(S0_OUT, ["opt"])


@pytest.fixture(scope="module")
def s1_excited_opt_parser() -> ORCAParser:
    return _parse_with_sections(S1_OUT, ["tddft", "opt"])


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


def test_job_snapshot_normalizes_job_identity_and_state_labels(
    goat_parser: ORCAParser,
    s1_parser: ORCAParser,
    tddft_parser: ORCAParser,
) -> None:
    goat_snapshot = goat_parser.data["job_snapshot"]
    s1_snapshot = s1_parser.data["job_snapshot"]
    tddft_snapshot = tddft_parser.data["job_snapshot"]

    assert goat_snapshot["calculation_family"] == "goat"
    assert goat_snapshot["method_header_label"] == "XTB2"
    assert goat_snapshot["method_table_label"] == "XTB2"
    assert goat_snapshot["is_goat"] is True
    assert goat_snapshot["electronic_state_kind"] == "ground_state"
    assert goat_snapshot["job_id"] == "sample_outs/RDB_LMMP/a/CH2Cl2/GOAT/RDB_LMMPa.out"

    assert s1_snapshot["calculation_family"] == "excited_state_optimization"
    assert s1_snapshot["method_header_label"] == "LibXC(wB97X-D4)/Def2-TZVP"
    assert s1_snapshot["method_table_label"] == "LibXC(wB97X-D4)"
    assert s1_snapshot["special_electronic_state_label"] == "Excited-state optimization (S1)"
    assert s1_snapshot["state_target_label"] == "S1"
    assert s1_snapshot["charge"] == 0
    assert s1_snapshot["multiplicity"] == 1
    assert s1_snapshot["is_excited_state_optimization"] is True

    assert tddft_snapshot["calculation_family"] == "single_point"
    assert tddft_snapshot["method_header_label"] == "SOS-wPBEPP86/Def2-TZVP"
    assert tddft_snapshot["method_table_label"] == "SOS-wPBEPP86"
    assert tddft_snapshot["special_electronic_state_label"] == ""
    assert tddft_snapshot["basis_set"] == "Def2-TZVP"
    assert tddft_snapshot["aux_basis_set"] == "Def2-TZVP/C"


def test_job_snapshot_remains_authoritative_for_markdown_metadata_and_comparison() -> None:
    s0_parser = _parse_quick(S0_OUT)
    s1_parser = _parse_quick(S1_OUT)
    snapshot = s1_parser.data["job_snapshot"]

    s1_parser.data["metadata"]["job_name"] = "BROKEN_JOB"
    s1_parser.data["metadata"]["job_id"] = "BROKEN/ID"
    s1_parser.data["metadata"]["calculation_type"] = "BROKEN_TYPE"
    s1_parser.data["metadata"]["method"] = "BROKEN_METHOD"
    s1_parser.data["metadata"]["functional"] = "BROKEN_FUNCTIONAL"
    s1_parser.data["metadata"]["level_of_theory"] = "BROKEN_LOT"
    s1_parser.data["metadata"]["basis_set"] = "BROKEN_BASIS"
    s1_parser.data["metadata"]["aux_basis_set"] = "BROKEN_AUX"
    s1_parser.data["metadata"]["charge"] = 999
    s1_parser.data["metadata"]["multiplicity"] = 999
    s1_parser.data["metadata"]["input_use_sym"] = True
    s1_parser.data["metadata"]["excited_state_optimization"] = {
        "target_state_label": "BROKEN_STATE",
        "target_root": 999,
    }
    s1_parser.data["metadata"]["symmetry"] = {
        "point_group": "BROKEN_PG",
        "orbital_irrep_group": "BROKEN_IRREP",
        "input_use_sym": True,
    }

    tmp_dir = REPO_ROOT / ".pytest_tmp" / f"rdb_lmmp_job_snapshot_{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = tmp_dir / "s1_job_snapshot.md"
    comparison_path = tmp_dir / "comparison.md"
    csv_dir = tmp_dir / "csv"

    s1_parser.to_markdown(markdown_path)
    s1_parser.to_csv(csv_dir)
    ORCAParser.compare([s0_parser, s1_parser], comparison_path)

    markdown_text = markdown_path.read_text(encoding="utf-8")
    comparison_text = comparison_path.read_text(encoding="utf-8")

    metadata_csv_path = csv_dir / "RDB_LMMP_S1_CH2Cl2_metadata.csv"
    with metadata_csv_path.open(encoding="utf-8", newline="") as handle:
        metadata_row = next(csv.DictReader(handle))

    assert "BROKEN" not in markdown_text
    assert "BROKEN" not in comparison_text
    assert "`LibXC(wB97X-D4)/Def2-TZVP`" in markdown_text
    assert "charge=0 mult=1" in markdown_text
    assert "state=Excited-state optimization (S1)" in markdown_text
    assert (
        "id: `sample_outs/RDB_LMMP/a/CH2Cl2/S1/RDB_LMMP_S1_CH2Cl2.out`"
        in markdown_text
    )

    assert "BROKEN" not in ",".join(metadata_row.values())
    assert metadata_row["job_name"] == snapshot["job_name"]
    assert metadata_row["job_id"] == snapshot["job_id"]
    assert metadata_row["calculation_type"] == snapshot["calculation_type"]
    assert metadata_row["calculation_family"] == snapshot["calculation_family"]
    assert metadata_row["method"] == snapshot["method"]
    assert metadata_row["functional"] == snapshot["functional"]
    assert metadata_row["level_of_theory"] == snapshot["level_of_theory"]
    assert metadata_row["method_header_label"] == snapshot["method_header_label"]
    assert metadata_row["method_table_label"] == snapshot["method_table_label"]
    assert metadata_row["basis_set"] == snapshot["basis_set"]
    assert metadata_row["aux_basis_set"] == snapshot["aux_basis_set"]
    assert metadata_row["charge"] == str(snapshot["charge"])
    assert metadata_row["multiplicity"] == str(snapshot["multiplicity"])
    assert metadata_row["electronic_state"] == snapshot["special_electronic_state_label"]

    assert "S1/RDB_LMMP_S1_CH2Cl2.out" in comparison_text
    assert "LibXC(wB97X-D4)" in comparison_text
    assert "Def2-TZVP" in comparison_text
    assert "Excited-state optimization (S1)" in comparison_text


def test_job_series_normalizes_stepwise_job_histories(
    goat_full_parser: ORCAParser,
    scan_parser: ORCAParser,
    s0_opt_parser: ORCAParser,
    s1_excited_opt_parser: ORCAParser,
) -> None:
    goat_series = goat_full_parser.data["job_series"]["goat"]
    scan_series = scan_parser.data["job_series"]["surface_scan"]
    s0_series = s0_opt_parser.data["job_series"]["geom_opt"]
    s1_series = s1_excited_opt_parser.data["job_series"]["excited_state_optimization"]

    assert goat_series["n_conformers"] == 1147
    assert len(goat_series["ensemble"]) == 1147
    assert goat_series["lowest_energy_conformer_Eh"] == pytest.approx(-134.778230)

    assert scan_series["n_parameters"] == 1
    assert len(scan_series["steps"]) == 37
    assert scan_series["mode"] == "single"

    assert s0_series["converged"] is True
    assert s0_series["n_cycles"] == 31
    assert len(s0_series["cycles"]) == 31

    assert s1_series["target_state_label"] == "S1"
    assert s1_series["gradient_block_count"] == 44
    assert len(s1_series["cycle_records"]) == 44


def test_job_series_remains_authoritative_for_stepwise_markdown_and_csv_exports() -> None:
    goat_parser = _parse_with_sections(GOAT_OUT, ["goat"])
    scan_parser = _parse_with_sections(SCAN_OUT, ["scan"])
    s0_parser = _parse_with_sections(S0_OUT, ["opt"])
    s1_parser = _parse_with_sections(S1_OUT, ["tddft", "opt"])

    goat_series = goat_parser.data["job_series"]["goat"]
    scan_series = scan_parser.data["job_series"]["surface_scan"]
    s0_series = s0_parser.data["job_series"]["geom_opt"]
    s1_series = s1_parser.data["job_series"]["excited_state_optimization"]

    goat_parser.data["goat"] = {
        "n_conformers": 1,
        "ensemble": [{"conformer": 9999, "relative_energy_kcal_mol": 999.0}],
        "conformers_below_energy_window": 1,
    }
    scan_parser.data["surface_scan"] = {
        "mode": "BROKEN_MODE",
        "n_parameters": 999,
        "n_constrained_optimizations": 1,
        "parameters": [],
        "steps": [{"step": 999, "actual_energy_Eh": 999.0}],
    }
    s0_parser.data["geom_opt"] = {
        "converged": False,
        "n_cycles": 1,
        "cycles": [{"cycle": 999, "energy_Eh": 999.0}],
    }
    s1_parser.data.setdefault("tddft", {})["excited_state_optimization"] = {
        "target_state_label": "BROKEN_STATE",
        "gradient_block_count": 1,
        "cycle_records": [{"optimization_cycle": 999, "total_energy_Eh": 999.0}],
    }
    s1_parser.data.setdefault("metadata", {})["excited_state_optimization"] = {
        "target_state_label": "BROKEN_STATE",
    }

    tmp_dir = REPO_ROOT / ".pytest_tmp" / f"rdb_lmmp_job_series_{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    goat_md = tmp_dir / "goat.md"
    goat_csv = tmp_dir / "goat_csv"
    goat_parser.to_markdown(goat_md)
    goat_parser.to_csv(goat_csv)

    scan_md = tmp_dir / "scan.md"
    scan_csv = tmp_dir / "scan_csv"
    scan_parser.to_markdown(scan_md)
    scan_parser.to_csv(scan_csv)

    s0_md = tmp_dir / "s0_opt.md"
    s0_csv = tmp_dir / "s0_csv"
    s0_parser.to_markdown(s0_md)
    s0_parser.to_csv(s0_csv)

    s1_csv = tmp_dir / "s1_csv"
    s1_parser.to_csv(s1_csv)

    goat_md_text = goat_md.read_text(encoding="utf-8")
    scan_md_text = scan_md.read_text(encoding="utf-8")
    s0_md_text = s0_md.read_text(encoding="utf-8")
    with (goat_csv / "RDB_LMMPa_goat_summary.csv").open(encoding="utf-8", newline="") as handle:
        goat_summary_row = next(csv.DictReader(handle))
    with (goat_csv / "RDB_LMMPa_goat_ensemble.csv").open(encoding="utf-8", newline="") as handle:
        goat_ensemble_rows = list(csv.DictReader(handle))
    with (scan_csv / "F3CNO_surface_scan.csv").open(encoding="utf-8", newline="") as handle:
        scan_rows = list(csv.DictReader(handle))
    with (s0_csv / "RDB_LMMP_S0_CH2Cl2_geom_opt.csv").open(encoding="utf-8", newline="") as handle:
        s0_opt_rows = list(csv.DictReader(handle))
    with (
        s1_csv / "RDB_LMMP_S1_CH2Cl2_excited_state_optimization_cycles.csv"
    ).open(encoding="utf-8", newline="") as handle:
        s1_excited_rows = list(csv.DictReader(handle))

    assert "9999" not in goat_md_text
    assert goat_summary_row["n_conformers"] == str(goat_series["n_conformers"])
    assert goat_summary_row["conformers_below_energy_window"] == str(
        goat_series["conformers_below_energy_window"]
    )
    assert len(goat_ensemble_rows) == len(goat_series["ensemble"])
    assert goat_ensemble_rows[-1]["conformer"] == str(goat_series["ensemble"][-1]["conformer"])
    assert float(goat_ensemble_rows[-1]["relative_energy_kcal_mol"]) == pytest.approx(
        goat_series["ensemble"][-1]["relative_energy_kcal_mol"]
    )
    assert str(goat_series["n_conformers"]) in goat_md_text

    assert "BROKEN_MODE" not in scan_md_text
    assert scan_series["mode"] in scan_md_text
    assert len(scan_rows) == len(scan_series["steps"])
    assert scan_rows[0]["step"] == str(scan_series["steps"][0]["step"])
    assert float(scan_rows[0]["actual_energy_Eh"]) == pytest.approx(
        scan_series["steps"][0]["actual_energy_Eh"]
    )
    assert float(scan_rows[-1]["actual_energy_Eh"]) == pytest.approx(
        scan_series["steps"][-1]["actual_energy_Eh"]
    )

    assert "999.0" not in s0_md_text
    assert f"**Cycles:** {s0_series['n_cycles']}" in s0_md_text
    assert len(s0_opt_rows) == len(s0_series["cycles"])
    assert s0_opt_rows[0]["cycle"] == str(s0_series["cycles"][0]["cycle"])
    assert float(s0_opt_rows[-1]["energy_Eh"]) == pytest.approx(
        s0_series["cycles"][-1]["energy_Eh"]
    )

    assert len(s1_excited_rows) == len(s1_series["cycle_records"])
    assert s1_excited_rows[0]["optimization_cycle"] == str(
        s1_series["cycle_records"][0]["optimization_cycle"]
    )
    assert float(s1_excited_rows[-1]["total_energy_Eh"]) == pytest.approx(
        s1_series["cycle_records"][-1]["total_energy_Eh"]
    )


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

    snapshot = s1_final_step_parser.data["final_snapshot"]
    snapshot_geometry = snapshot["geometry"]["cartesian_angstrom"]
    snapshot_orbitals = snapshot["orbital_energies"]
    snapshot_mulliken = snapshot["charges"]["mulliken"]["atoms"]
    snapshot_mayer = snapshot["mayer"]["bond_orders"]
    snapshot_dipole = snapshot["dipole"]["total_dipole_au"]

    assert snapshot["selection"] == "last_reported_optimization_step"
    assert snapshot_geometry[0]["x_ang"] == pytest.approx(raw_s1_geometry[0]["x_ang"])
    assert snapshot_geometry[0]["y_ang"] == pytest.approx(raw_s1_geometry[0]["y_ang"])
    assert snapshot_geometry[0]["z_ang"] == pytest.approx(raw_s1_geometry[0]["z_ang"])
    assert snapshot_orbitals["HOMO_energy_eV"] == pytest.approx(raw_s1_orbitals["HOMO_energy_eV"])
    assert snapshot_orbitals["LUMO_energy_eV"] == pytest.approx(raw_s1_orbitals["LUMO_energy_eV"])
    assert snapshot_mulliken[0]["charge"] == pytest.approx(raw_s1_mulliken_c0)
    assert any(
        tuple(sorted((bond["atom_i"], bond["atom_j"]))) == (0, 1)
        and bond["bond_order"] == pytest.approx(raw_s1_mayer_01)
        for bond in snapshot_mayer
    )
    assert snapshot_dipole["x"] == pytest.approx(raw_s1_dipole["x"])
    assert snapshot_dipole["y"] == pytest.approx(raw_s1_dipole["y"])
    assert snapshot_dipole["z"] == pytest.approx(raw_s1_dipole["z"])


def test_final_snapshot_remains_authoritative_for_markdown_and_csv() -> None:
    parser = _parse_with_sections(S1_OUT, ["charges", "population", "mos", "dipole"])
    snapshot = parser.data["final_snapshot"]

    real_geom_x = snapshot["geometry"]["cartesian_angstrom"][0]["x_ang"]
    real_homo = snapshot["orbital_energies"]["HOMO_energy_eV"]
    real_dipole = snapshot["dipole"]["magnitude_Debye"]
    real_charge = snapshot["charges"]["mulliken"]["atoms"][0]["charge"]
    real_bond = next(
        bond["bond_order"]
        for bond in snapshot["mayer"]["bond_orders"]
        if tuple(sorted((bond["atom_i"], bond["atom_j"]))) == (0, 1)
    )

    parser.data["geometry"]["cartesian_angstrom"][0]["x_ang"] = 999.0
    parser.data["orbital_energies"]["HOMO_energy_eV"] = 999.0
    parser.data["dipole"]["magnitude_Debye"] = 999.0
    parser.data["mulliken"]["atomic_charges"][0]["charge"] = 999.0
    parser.data["mayer"]["bond_orders"][0]["bond_order"] = 999.0

    tmp_dir = REPO_ROOT / ".pytest_tmp" / f"rdb_lmmp_snapshot_{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = tmp_dir / "s1_snapshot.md"
    csv_dir = tmp_dir / "csv"
    parser.to_markdown(markdown_path)
    parser.to_csv(csv_dir)

    markdown_text = markdown_path.read_text(encoding="utf-8")
    geometry_csv = (csv_dir / "RDB_LMMP_S1_CH2Cl2_geometry.csv").read_text(encoding="utf-8")
    dipole_csv = (csv_dir / "RDB_LMMP_S1_CH2Cl2_dipole.csv").read_text(encoding="utf-8")
    mulliken_csv = (csv_dir / "RDB_LMMP_S1_CH2Cl2_mulliken_charges.csv").read_text(encoding="utf-8")
    mayer_csv = (csv_dir / "RDB_LMMP_S1_CH2Cl2_mayer_bonds.csv").read_text(encoding="utf-8")

    assert "999.0" not in markdown_text
    assert "999.0" not in geometry_csv
    assert "999.0" not in dipole_csv
    assert "999.0" not in mulliken_csv
    assert "999.0" not in mayer_csv

    assert f"{real_geom_x:.6f}" in markdown_text
    assert f"{real_homo:.4f}" in markdown_text
    assert f"{real_dipole}" in dipole_csv
    assert f"{real_charge}" in mulliken_csv
    assert f"{real_bond}" in mayer_csv
