from __future__ import annotations

from pathlib import Path

import pytest

from orca_parser import ORCAParser


def _write_tddft_trajectory_fixture(path: Path, *, followiroot: bool = True) -> None:
    follow_text = "true" if followiroot else "false"
    root_following = ""
    if followiroot:
        root_following = """
Largest overlap:                                ... 0.921063
Ratio second largest/largest:                  ... 0.507532
The IROOT now is:                                ... 2
"""

    path.write_text(
        f"""
                                 *****************
                                 * O   R   C   A *
                                 *****************
Program Version 6.0.0
|  1> %tddft
|  2>   nroots      3
|  3>   iroot       1
|  4>   followiroot {follow_text}
|  5> end
****END OF INPUT****
Geometry Optimization Run
*                GEOMETRY OPTIMIZATION CYCLE   1            *
CIS/TD-DFT TOTAL ENERGY
E(SCF)       =    -100.000000 Eh
DE(CIS)      =       0.160000 Eh (Root 1)
E(tot)       =     -99.840000 Eh
Follow IRoot                                ... {follow_text}
State of interest                           ... 1
IROOT 1
Input electron density                      ... demo.cispre.singlet.iroot1
TD-DFT/TDA EXCITED STATES (SINGLETS)
the weight of the individual excitations are printed if larger than 1.0e-02
STATE  1:  E=   0.160000 au      4.354 eV    35116.0 cm**-1 <S**2> =   0.000000
    10a ->  11a  :     0.650000 (c=  0.80622577)
STATE  2:  E=   0.200000 au      5.442 eV    43899.0 cm**-1 <S**2> =   0.000000
    10a ->  12a  :     0.580000 (c=  0.76157731)

----------------------------------------------------------------------------------------------------
                     ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS
----------------------------------------------------------------------------------------------------
     Transition      Energy     Energy  Wavelength fosc(D2)      D2        DX        DY        DZ
                      (eV)      (cm-1)    (nm)                 (au**2)    (au)      (au)      (au)
----------------------------------------------------------------------------------------------------
  0-1A  ->  1-1A    4.354000   35116.0   284.8   0.020000000   0.10000   0.10000   0.00000   0.00000
  0-1A  ->  2-1A    5.442000   43899.0   227.8   0.500000000   2.50000   1.50000   0.50000   0.00000

*                GEOMETRY OPTIMIZATION CYCLE   2            *
{root_following}
CIS/TD-DFT TOTAL ENERGY
E(SCF)       =    -100.020000 Eh
DE(CIS)      =       0.150000 Eh (Root 2)
E(tot)       =     -99.870000 Eh
Follow IRoot                                ... {follow_text}
State of interest                           ... 2
IROOT 2
Input electron density                      ... demo.cispre.singlet.iroot2
TD-DFT/TDA EXCITED STATES (SINGLETS)
the weight of the individual excitations are printed if larger than 1.0e-02
STATE  1:  E=   0.180000 au      4.898 eV    39506.0 cm**-1 <S**2> =   0.000000
    10a ->  12a  :     0.550000 (c=  0.74161985)
STATE  2:  E=   0.150000 au      4.082 eV    32923.0 cm**-1 <S**2> =   0.000000
    10a ->  11a  :     0.700000 (c=  0.83666003)

----------------------------------------------------------------------------------------------------
                     ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS
----------------------------------------------------------------------------------------------------
     Transition      Energy     Energy  Wavelength fosc(D2)      D2        DX        DY        DZ
                      (eV)      (cm-1)    (nm)                 (au**2)    (au)      (au)      (au)
----------------------------------------------------------------------------------------------------
  0-1A  ->  1-1A    4.898000   39506.0   253.1   0.030000000   0.12000   0.10000   0.10000   0.00000
  0-1A  ->  2-1A    4.082000   32923.0   303.7   0.400000000   2.00000   1.30000   0.40000   0.10000

ORCA TERMINATED NORMALLY
""",
        encoding="utf-8",
    )


def test_tddft_trajectory_keeps_orca_roots_separate_from_ranked_states(
    tmp_path: Path,
) -> None:
    output = tmp_path / "s1_followiroot.out"
    _write_tddft_trajectory_fixture(output, followiroot=True)

    tddft = ORCAParser(output).parse(sections=["tddft"])["tddft"]
    trajectory = tddft["trajectory"]

    assert trajectory["kind"] == "tddft_cis_optimization_trajectory"
    assert len(trajectory["state_rows"]) == 4

    step2_s1 = next(
        row
        for row in trajectory["state_rows"]
        if row["step_index"] == 2 and row["energy_rank"] == 1
    )
    assert step2_s1["source_context"] == "optimization_step_spectrum"
    assert step2_s1["orca_root_printed"] == 2
    assert step2_s1["state_label"] == "S1"
    assert step2_s1["oscillator_strength"] == pytest.approx(0.4)
    assert step2_s1["transition_dipole_norm_or_T2"] == pytest.approx(2.0)
    assert step2_s1["root_following_overlap"] == pytest.approx(0.921063)
    assert step2_s1["root_following_second_largest_ratio"] == pytest.approx(0.507532)
    assert step2_s1["is_followed_or_target_root"] is True

    final_step = trajectory["step_summaries"][-1]
    assert final_step["S1_orca_root"] == 2
    assert final_step["followed_or_target_root"] == 2
    assert final_step["followed_or_target_state_label"] == "S1"
    assert final_step["brightest_all_orca_root"] == 2
    assert final_step["possible_root_flip_flag"] is True

    final_summary = trajectory["final_summary"]
    assert final_summary["final_S1_orca_root"] == 2
    assert final_summary["final_S1_wavelength_nm"] == pytest.approx(303.7)
    assert final_summary["possible_root_flip_count"] == 1
    assert final_summary["trajectory_class"] == "mixed_or_state_instability"


def test_tddft_trajectory_is_built_without_followiroot(tmp_path: Path) -> None:
    output = tmp_path / "s1_no_followiroot.out"
    _write_tddft_trajectory_fixture(output, followiroot=False)

    tddft = ORCAParser(output).parse(sections=["tddft"])["tddft"]
    excopt = tddft["excited_state_optimization"]
    trajectory = tddft["trajectory"]

    assert excopt["followiroot"] is False
    assert len(trajectory["step_summaries"]) == 2
    assert trajectory["step_summaries"][0]["root_following_overlap"] == ""
    assert trajectory["step_summaries"][1]["S1_orca_root"] == 2


def test_tddft_trajectory_does_not_cross_wire_singlet_and_triplet_roots(
    tmp_path: Path,
) -> None:
    output = tmp_path / "singlet_triplet.out"
    output.write_text(
        """
                                 *****************
                                 * O   R   C   A *
                                 *****************
Program Version 6.0.0
|  1> %tddft
|  2>   nroots 2
|  3>   triplets true
|  4> end
****END OF INPUT****
TD-DFT/TDA EXCITED STATES (SINGLETS)
STATE  1:  E=   0.100000 au      2.721 eV    21946.0 cm**-1 <S**2> =   0.000000
    10a ->  11a  :     0.700000 (c=  0.83666003)
STATE  2:  E=   0.150000 au      4.082 eV    32923.0 cm**-1 <S**2> =   0.000000
    10a ->  12a  :     0.500000 (c=  0.70710678)

TD-DFT/TDA EXCITED STATES (TRIPLETS)
STATE  1:  E=   0.080000 au      2.177 eV    17557.0 cm**-1 <S**2> =   2.000000
    10a ->  11a  :     0.800000 (c=  0.89442719)
STATE  2:  E=   0.120000 au      3.265 eV    26336.0 cm**-1 <S**2> =   2.000000
    10a ->  12a  :     0.600000 (c=  0.77459667)

----------------------------------------------------------------------------------------------------
                     ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS
----------------------------------------------------------------------------------------------------
     Transition      Energy     Energy  Wavelength fosc(D2)      D2        DX        DY        DZ
                      (eV)      (cm-1)    (nm)                 (au**2)    (au)      (au)      (au)
----------------------------------------------------------------------------------------------------
  0-1A  ->  1-1A    2.721000   21946.0   455.7   0.123000000   0.50000   0.30000   0.10000   0.00000
  0-1A  ->  2-1A    4.082000   32923.0   303.7   0.456000000   1.50000   1.10000   0.20000   0.10000

ORCA TERMINATED NORMALLY
""",
        encoding="utf-8",
    )

    tddft = ORCAParser(output).parse(sections=["tddft"])["tddft"]
    assert tddft["final_excited_state_block"]["manifold"] == "SINGLETS"
    assert {
        block["manifold"]
        for block in tddft["final_excited_state_blocks"]
    } == {"SINGLETS", "TRIPLETS"}

    rows = tddft["trajectory"]["state_rows"]
    singlet_root_1 = next(
        row
        for row in rows
        if row["normalized_manifold"] == "singlet" and row["orca_root_printed"] == 1
    )
    triplet_root_1 = next(
        row
        for row in rows
        if row["normalized_manifold"] == "triplet" and row["orca_root_printed"] == 1
    )

    assert singlet_root_1["oscillator_strength"] == pytest.approx(0.123)
    assert triplet_root_1["oscillator_strength"] is None
    assert triplet_root_1["parse_quality_flag"] == "missing_electric_dipole_spectrum"

    step_summaries = tddft["trajectory"]["step_summaries"]
    assert {summary["manifold"] for summary in step_summaries} == {"singlet", "triplet"}
    singlet_summary = next(
        summary for summary in step_summaries if summary["manifold"] == "singlet"
    )
    triplet_summary = next(
        summary for summary in step_summaries if summary["manifold"] == "triplet"
    )
    assert singlet_summary["S1_orca_root"] == 1
    assert triplet_summary["S1_orca_root"] == 1
    assert triplet_summary["S1_oscillator_strength"] is None

    final_summary = tddft["trajectory"]["final_summary"]
    assert final_summary["manifold"] == "singlet"
    assert final_summary["final_S1_orca_root"] == 1


def test_tddft_trajectory_keeps_spectrum_only_rows_per_manifold(
    tmp_path: Path,
) -> None:
    output = tmp_path / "spectrum_only_triplet.out"
    output.write_text(
        """
                                 *****************
                                 * O   R   C   A *
                                 *****************
Program Version 6.0.0
|  1> %tddft
|  2>   nroots 1
|  3>   triplets true
|  4> end
****END OF INPUT****
TD-DFT/TDA EXCITED STATES (SINGLETS)
STATE  1:  E=   0.100000 au      2.721 eV    21946.0 cm**-1 <S**2> =   0.000000
    10a ->  11a  :     0.700000 (c=  0.83666003)

----------------------------------------------------------------------------------------------------
                     ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS
----------------------------------------------------------------------------------------------------
     Transition      Energy     Energy  Wavelength fosc(D2)      D2        DX        DY        DZ
                      (eV)      (cm-1)    (nm)                 (au**2)    (au)      (au)      (au)
----------------------------------------------------------------------------------------------------
  0-1A  ->  1-1A    2.721000   21946.0   455.7   0.123000000   0.50000   0.30000   0.10000   0.00000
  0-1A  ->  1-3A    1.900000   15325.0   652.5   0.010000000   0.05000   0.03000   0.01000   0.00000

ORCA TERMINATED NORMALLY
""",
        encoding="utf-8",
    )

    tddft = ORCAParser(output).parse(sections=["tddft"])["tddft"]
    rows = tddft["trajectory"]["state_rows"]

    singlet_rows = [row for row in rows if row["normalized_manifold"] == "singlet"]
    triplet_rows = [row for row in rows if row["normalized_manifold"] == "triplet"]

    assert len(singlet_rows) == 1
    assert len(triplet_rows) == 1
    assert singlet_rows[0]["parse_quality_flag"] == ""
    assert triplet_rows[0]["parse_quality_flag"] == "missing_excited_state_block"
    assert triplet_rows[0]["state_label"] == "T1"
    assert triplet_rows[0]["oscillator_strength"] == pytest.approx(0.01)
