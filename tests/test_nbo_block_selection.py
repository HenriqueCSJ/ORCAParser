from __future__ import annotations

import csv
from pathlib import Path

from orca_parser import ORCAParser


def _nbo_summary(charge_c1: float, charge_c2: float) -> str:
    return f"""
Now starting NBO....

 *********************************** NBO 7.0 ***********************************
 Summary of Natural Population Analysis:

                                     Natural Population
             Natural    ---------------------------------------------
  Atom No    Charge        Core      Valence    Rydberg      Total
 --------------------------------------------------------------------
    C  1    {charge_c1: .5f}      1.99999     3.80000    0.02000     5.90000
    C  2    {charge_c2: .5f}      1.99999     3.90000    0.02000     5.90000
 ====================================================================
"""


def _write_repeated_nbo_fixture(path: Path) -> None:
    path.write_text(
        """
                                 *****************
                                 * O   R   C   A *
                                 *****************
Program Version 6.0.0
|  1> %tddft
|  2>   nroots 5
|  3>   iroot 1
|  4> end
****END OF INPUT****
*                GEOMETRY OPTIMIZATION CYCLE   1            *
FINAL SINGLE POINT ENERGY     -1.0
"""
        + _nbo_summary(0.10000, 0.20000)
        + """
*                GEOMETRY OPTIMIZATION CYCLE   2            *
FINAL ENERGY EVALUATION AT THE STATIONARY POINT
------------------------------------------------------------------------------
                  RELAXED CIS/TDA DENSITY POPULATION ANALYSIS
                                   IROOT 1
------------------------------------------------------------------------------
------------------------------------------------------------------------------
                           ORCA POPULATION ANALYSIS
------------------------------------------------------------------------------
Input electron density              ... demo.cispre.singlet.iroot1
BaseName (.gbw .S,...)              ... demo
"""
        + _nbo_summary(0.30000, 0.40000)
        + """
ORCA TERMINATED NORMALLY
""",
        encoding="utf-8",
    )


def test_nbo_parser_selects_final_excited_state_npa_block(tmp_path: Path) -> None:
    output = tmp_path / "S1b" / "demo.out"
    output.parent.mkdir()
    _write_repeated_nbo_fixture(output)

    parser = ORCAParser(output)
    data = parser.parse(sections=["nbo"])
    nbo = data["nbo"]

    assert nbo["nbo_block_index"] == 2
    assert nbo["nbo_block_count"] == 2
    assert nbo["optimization_cycle"] == 2
    assert nbo["is_final_cycle"] is True
    assert nbo["density_context"] == "excited_state"
    assert nbo["density_kind"] == "relaxed_cis_tda"
    assert nbo["excited_state_specific"] is True
    assert nbo["root"] == 1
    assert nbo["stage"] == "S1b"
    assert nbo["input_electron_density_file"] == "demo.cispre.singlet.iroot1"
    assert nbo["npa_summary"][0]["natural_charge"] == 0.30000
    assert nbo["npa_summary"][1]["natural_charge"] == 0.40000
    assert "Multiple NBO blocks found" in nbo["warnings"][0]

    markdown_path = tmp_path / "nbo.md"
    parser.to_markdown(markdown_path)
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "NBO/NPA Provenance" in markdown
    assert "block 2/2" in markdown
    assert "density=excited_state" in markdown

    csv_dir = tmp_path / "csv"
    parser.to_csv(csv_dir)
    csv_path = csv_dir / "demo_nbo_npa.csv"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert rows[0]["natural_charge"] == "0.3"
    assert rows[0]["nbo_block_index"] == "2"
    assert rows[0]["density_context"] == "excited_state"
    assert rows[0]["input_electron_density_file"] == "demo.cispre.singlet.iroot1"
