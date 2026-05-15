from __future__ import annotations

import textwrap
from pathlib import Path

from orca_parser import ORCAParser


def _write_double_hybrid_fixture(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """
            O   R   C   A
            Program Version 6.0.0
            Number of atoms                         ... 2
            Total Charge           Charge           .... 0
            Multiplicity           Mult             .... 1
            *                GEOMETRY OPTIMIZATION CYCLE   1            *

                                ********************************
                                * MULLIKEN POPULATION ANALYSIS *
                                ********************************

            -----------------------
            MULLIKEN ATOMIC CHARGES
            -----------------------
               0 C :    0.100000
               1 O :   -0.100000
            Sum of atomic charges:    0.0000000

                               *******************************
                               * LOEWDIN POPULATION ANALYSIS *
                               *******************************

            ----------------------
            LOEWDIN ATOMIC CHARGES
            ----------------------
               0 C :    0.050000
               1 O :   -0.050000
            Sum of atomic charges:    0.0000000

            ----------------------
            RI-MP2 ENERGY+GRADIENT
            ----------------------
            Dimension of the orbital basis            ...  12
            Dimension of the AuxC basis               ...  24
            Memory devoted to MP2                     ... 256 MB
            Overall scaling of the MP2 energy         ...   1.000e+00
            -----------------------------------------------
             RI-MP2 CORRELATION ENERGY:     -0.120000000 Eh
            -----------------------------------------------

            ---------------------
            MP2 DENSITY FORMATION
            ---------------------
            Storing the unrelaxed density                    ... done
            Finalizing the relaxed density                   ... done
            Storing the relaxed density                      ... done
            Trace of the density to be diagonalized = 10.000000
            Sum of eigenvalues = 10.000000
            Natural Orbital Occupation Numbers:
            N[  0] =   1.99900000
            N[  1] =   0.00100000
            Input SCF Electron Density              ... calc.scfp
            Input Correlated Electron Density       ... calc.pmp2re
            Input Energy Weighted Density           ... calc.wmp2.tmp

                            *********************************************
                            * UNRELAXED MP2 DENSITY POPULATION ANALYSIS *
                            *********************************************

            ------------------------------------------------------------------------------
                                       ORCA POPULATION ANALYSIS
            ------------------------------------------------------------------------------
            Input electron density              ... calc.pmp2ur
            BaseName (.gbw .S,...)              ... calc

                                ********************************
                                * MULLIKEN POPULATION ANALYSIS *
                                ********************************

            -----------------------
            MULLIKEN ATOMIC CHARGES
            -----------------------
               0 C :    0.200000
               1 O :   -0.200000
            Sum of atomic charges:    0.0000000

                               *******************************
                               * LOEWDIN POPULATION ANALYSIS *
                               *******************************

            ----------------------
            LOEWDIN ATOMIC CHARGES
            ----------------------
               0 C :    0.150000
               1 O :   -0.150000
            Sum of atomic charges:    0.0000000

                            *******************************************
                            * RELAXED MP2 DENSITY POPULATION ANALYSIS *
                            *******************************************

            ------------------------------------------------------------------------------
                                       ORCA POPULATION ANALYSIS
            ------------------------------------------------------------------------------
            Input electron density              ... calc.pmp2re
            BaseName (.gbw .S,...)              ... calc

                                ********************************
                                * MULLIKEN POPULATION ANALYSIS *
                                ********************************

            -----------------------
            MULLIKEN ATOMIC CHARGES
            -----------------------
               0 C :    0.300000
               1 O :   -0.300000
            Sum of atomic charges:    0.0000000

            MP2 TOTAL ENERGY:     -100.500000000 Eh
            NORM OF THE MP2 GRADIENT:  0.000123
            FINAL SINGLE POINT ENERGY      -100.600000000

                             *** FINAL ENERGY EVALUATION AT THE STATIONARY POINT ***

                                ********************************
                                * MULLIKEN POPULATION ANALYSIS *
                                ********************************

            -----------------------
            MULLIKEN ATOMIC CHARGES
            -----------------------
               0 C :    0.110000
               1 O :   -0.110000
            Sum of atomic charges:    0.0000000

                               *******************************
                               * LOEWDIN POPULATION ANALYSIS *
                               *******************************

            ----------------------
            LOEWDIN ATOMIC CHARGES
            ----------------------
               0 C :    0.055000
               1 O :   -0.055000
            Sum of atomic charges:    0.0000000

            ----------------------
            RI-MP2 ENERGY+GRADIENT
            ----------------------
            -----------------------------------------------
             RI-MP2 CORRELATION ENERGY:     -0.130000000 Eh
            -----------------------------------------------
            Storing the unrelaxed density                    ... done
            Finalizing the relaxed density                   ... done
            Storing the relaxed density                      ... done
            Trace of the density to be diagonalized = 10.000000
            Sum of eigenvalues = 10.000000
            Natural Orbital Occupation Numbers:
            N[  0] =   1.99800000
            N[  1] =   0.00200000
            Input SCF Electron Density              ... calc.scfp
            Input Correlated Electron Density       ... calc.pmp2re
            Input Energy Weighted Density           ... calc.wmp2.tmp

                            *********************************************
                            * UNRELAXED MP2 DENSITY POPULATION ANALYSIS *
                            *********************************************

            ------------------------------------------------------------------------------
                                       ORCA POPULATION ANALYSIS
            ------------------------------------------------------------------------------
            Input electron density              ... calc.pmp2ur
            BaseName (.gbw .S,...)              ... calc

                                ********************************
                                * MULLIKEN POPULATION ANALYSIS *
                                ********************************

            -----------------------
            MULLIKEN ATOMIC CHARGES
            -----------------------
               0 C :    0.210000
               1 O :   -0.210000
            Sum of atomic charges:    0.0000000

                            *******************************************
                            * RELAXED MP2 DENSITY POPULATION ANALYSIS *
                            *******************************************

            ------------------------------------------------------------------------------
                                       ORCA POPULATION ANALYSIS
            ------------------------------------------------------------------------------
            Input electron density              ... calc.pmp2re
            BaseName (.gbw .S,...)              ... calc

                                ********************************
                                * MULLIKEN POPULATION ANALYSIS *
                                ********************************

            -----------------------
            MULLIKEN ATOMIC CHARGES
            -----------------------
               0 C :    0.310000
               1 O :   -0.310000
            Sum of atomic charges:    0.0000000

            MP2 TOTAL ENERGY:     -100.700000000 Eh
            NORM OF THE MP2 GRADIENT:  0.000045

            -------------
            DIPOLE MOMENT
            -------------

            Method             : SCF
            Type of density    : Electron Density
            Multiplicity       :   1
            Energy             :  -100.0000000000000000 Eh
            Basis              : AO
                                    X                 Y                 Z
            Electronic contribution:     -1.000000000       0.000000000       0.500000000
            Nuclear contribution   :      1.100000000       0.000000000      -0.400000000
                            -----------------------------------------
            Total Dipole Moment    :      0.100000000       0.000000000       0.100000000
                            -----------------------------------------
            Magnitude (a.u.)       :      0.141421356
            Magnitude (Debye)      :      0.359429043

            -------------
            DIPOLE MOMENT
            -------------

            Method             : MP2
            Type of density    : Electron Density
            Level              : Relaxed density
            Multiplicity       :   1
            Energy             :     0.0000000000000000 Eh
            Basis              : AO
                                    X                 Y                 Z
            Electronic contribution:     -1.000000000       0.000000000       0.500000000
            Nuclear contribution   :      1.200000000       0.000000000      -0.400000000
                            -----------------------------------------
            Total Dipole Moment    :      0.200000000       0.000000000       0.100000000
                            -----------------------------------------
            Magnitude (a.u.)       :      0.223606798
            Magnitude (Debye)      :      0.568355734

            ORCA TERMINATED NORMALLY
            """
        ),
        encoding="utf-8",
    )


def test_double_hybrid_density_analysis_keeps_initial_and_final_density_contexts(tmp_path: Path) -> None:
    output = tmp_path / "calc.out"
    _write_double_hybrid_fixture(output)

    data = ORCAParser(output).parse(sections=["density_analysis"])
    density = data["density_analysis"]

    assert density["summary"]["analysis_count"] == 6
    assert density["summary"]["has_initial_and_final_triples"] is True
    assert set(density["by_stage"]) == {"initial", "final"}

    initial = density["by_stage"]["initial"]
    final = density["by_stage"]["final"]
    assert set(initial) == {"scf", "mp2_unrelaxed", "mp2_relaxed"}
    assert set(final) == {"scf", "mp2_unrelaxed", "mp2_relaxed"}
    assert initial["scf"]["input_electron_density_file"] == "calc.scfp"
    assert initial["mp2_unrelaxed"]["input_electron_density_file"] == "calc.pmp2ur"
    assert final["mp2_relaxed"]["input_electron_density_file"] == "calc.pmp2re"

    assert initial["mp2_unrelaxed"]["population"]["mulliken"]["atomic_charges"][0]["charge"] == 0.2
    assert final["mp2_relaxed"]["population"]["mulliken"]["atomic_charges"][0]["charge"] == 0.31

    formations = density["mp2_density_formations"]
    assert len(formations) == 2
    assert formations[0]["stage"] == "initial"
    assert formations[0]["density_trace"] == 10.0
    assert formations[0]["natural_occupations"][0]["occupation"] == 1.999
    assert formations[1]["stage"] == "final"
    assert formations[1]["mp2_total_energy_Eh"] == -100.7

    assert density["dipoles"][-1]["density_kind"] == "mp2_relaxed"
    assert density["dipoles"][-1]["magnitude_Debye"] == 0.568355734


def test_opt_alias_includes_density_analysis_for_double_hybrid_outputs(tmp_path: Path) -> None:
    output = tmp_path / "calc.out"
    _write_double_hybrid_fixture(output)

    data = ORCAParser(output).parse(sections=["opt"])

    assert "geom_opt" in data
    assert "density_analysis" in data
    assert data["density_analysis"]["summary"]["analysis_count"] == 6
