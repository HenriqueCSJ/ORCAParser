from __future__ import annotations

from pathlib import Path

from orca_parser import ORCAParser


def _write_followiroot_fixture(path: Path) -> None:
    path.write_text(
        """
                                 *****************
                                 * O   R   C   A *
                                 *****************
Program Version 6.0.0
|  1> %TdDfT
|  2>   nRoOtS        5
|  3>   iRoOt         3
|  4>   fOlLoWiRoOt   TrUe
|  5>   fIrKeEpFiRsTrEf TrUe
|  6>   fIrEnThReSh   1.0
|  7>   fIrS2ThReSh   0.5
|  8>   fIrStHrEsH    0.05
|  9>   fIrMiNoVeRlAp 0.5
| 10>   fIrDyNoVeRlAp TrUe
| 11>   fIrDyNoVeRrAtIo 0.3,0.6
| 12> EnD
****END OF INPUT****
*                GEOMETRY OPTIMIZATION CYCLE   1            *
WARNING: (TDDFT/CIS) : Analytic excited state gradients requested
CIS/TD-DFT TOTAL ENERGY
E(SCF)       =    -1.000000 Eh
DE(CIS)      =     0.100000 Eh (Root 3)
E(tot)       =    -0.900000 Eh
Follow IRoot                                ... true
State of interest                           ... 3
IROOT 3
Input electron density                      ... demo.cispre.singlet.iroot3
The IROOT now is:                                ... 2
EXCITED STATE GRADIENT DONE
*                GEOMETRY OPTIMIZATION CYCLE   2            *
CIS/TD-DFT TOTAL ENERGY
E(SCF)       =    -1.010000 Eh
DE(CIS)      =     0.090000 Eh (Root 2)
E(tot)       =    -0.920000 Eh
Follow IRoot                                ... true
State of interest                           ... 2
IROOT 2
Input electron density                      ... demo.cispre.singlet.iroot2
EXCITED STATE GRADIENT DONE
ORCA TERMINATED NORMALLY
""",
        encoding="utf-8",
    )


def test_tddft_followiroot_tracks_active_flag_and_root_changes(tmp_path: Path) -> None:
    output = tmp_path / "followiroot.out"
    _write_followiroot_fixture(output)

    tddft = ORCAParser(output).parse(sections=["TdDfT"])["tddft"]
    excopt = tddft["excited_state_optimization"]

    assert tddft["input"]["block"] == "tddft"
    assert tddft["input"]["settings"]["followiroot"] is True
    assert excopt["target_root"] == 3
    assert excopt["target_state_label"] == "root 3"
    assert excopt["followiroot"] is True
    assert excopt["firkeepfirstref"] is True
    assert excopt["firen_thresh_eV"] == 1.0
    assert excopt["firs2_thresh"] == 0.5
    assert excopt["firsthresh"] == 0.05
    assert excopt["firminoverlap"] == 0.5
    assert excopt["firdynoverlap"] is True
    assert excopt["firdynoverratio"] == [0.3, 0.6]
    assert excopt["root_follow_updates"] == [2]
    assert excopt["final_root"] == 2

    cycle_records = excopt["cycle_records"]
    assert [record["optimization_cycle"] for record in cycle_records] == [1, 2]
    assert [record["current_iroot"] for record in cycle_records] == [3, 2]
    assert [record["state_of_interest"] for record in cycle_records] == [3, 2]
    assert [record["followiroot_runtime"] for record in cycle_records] == [True, True]
