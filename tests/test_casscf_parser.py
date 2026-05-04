from __future__ import annotations

from pathlib import Path

from orca_parser import ORCAParser
from orca_parser.modules.casscf import CASSCFModule
from orca_parser.parser_section_registry import get_parser_section_alias_map


REPO_ROOT = Path(__file__).resolve().parents[1]
CASSCF_SAMPLE = REPO_ROOT / "sample_outs" / "CASSCFb2" / "NC.out"


def _casscf_context() -> dict:
    return {
        "reference_type": "CASSCF",
        "plugin_options": {"casscf_orbital_window": 1},
        "input_echo": {
            "block_names": ["casscf"],
            "blocks": {
                "casscf": {
                    "raw_lines": ["nel 3", "norb 6", "mult 4, 2", "nroots 2, 2"],
                    "settings": {
                        "nel": 3,
                        "norb": 6,
                        "mult": "4, 2",
                        "nroots": "2, 2",
                    },
                }
            },
        },
    }


def test_casscf_module_parses_convergence_states_matrices_and_active_mos() -> None:
    lines = [
        "ORCA-CASSCF",
        "SYSTEM-SPECIFIC SETTINGS:",
        "Number of active electrons          ...    3",
        "Number of active orbitals           ...    6",
        "Determined orbital ranges:",
        "   Active       182 -  187 (   6 orbitals)",
        "CI-STEP:",
        "Number of multiplicity blocks       ...    1",
        "BLOCK  0 WEIGHT=   1.0000",
        "  Multiplicity                      ...    4",
        "  #(Roots)                          ...    2",
        "    ROOT=0 WEIGHT=    0.500000",
        "    ROOT=1 WEIGHT=    0.500000",
        "INTEGRAL-TRANSFORMATION-STEP:",
        "CAS-SCF ITERATIONS",
        "MACRO-ITERATION   1:",
        "   --- Inactive Energy E0 = -100.00000000 Eh",
        "   E(CAS)= -101.000000000 Eh DE=    0.000000e+00",
        "   --- Energy gap subspaces: Ext-Act = 0.399   Act-Int = 0.011",
        "   N(occ)=  0.84 0.75 0.20",
        "   ||g|| =     2.670751e-01 Max(G)=   -1.121894e-01 Rot=188,61",
        "   --- Option=FreezeAct: ||g|| =      0.267075083",
        "                               = 100.00%",
        "   --- Orbital Update [   SuperCI]",
        "SUPERCI-ITER   0: DE=    -0.002243793 <r|r>=      0.001018866",
        "   --- Density off from convergence (3.492e-01). Exact active Fock build",
        "MACRO-ITERATION   2:",
        "   E(CAS)= -101.100000000 Eh DE=   -1.000000e-01",
        "   --- Energy gap subspaces: Ext-Act = 0.071   Act-Int = -0.051",
        "   N(occ)=  0.85 0.75 0.20",
        "   ||g|| =     7.006994e-04 Max(G)=   -2.633866e-04 Rot=1916,73",
        "                     ---- THE CAS-SCF GRADIENT HAS CONVERGED ----",
        "                    ---- DOING ONE FINAL ITERATION FOR PRINTING ----",
        "MACRO-ITERATION   3:",
        "   E(CAS)= -101.100001000 Eh DE=   -1.000000e-06",
        "   --- Energy gap subspaces: Ext-Act = 0.071   Act-Int = -0.051",
        "   N(occ)=  0.86 0.74 0.20",
        "   ||g|| =     1.006994e-03 Max(G)=   -1.633866e-04 Rot=1916,73",
        "CASSCF RESULTS",
        "Final CASSCF energy       : -101.100000000 Eh  -2751.0000 eV",
        "ORBITAL ENERGIES",
        "  NO   OCC          E(Eh)            E(eV)",
        " 182   0.8500      -0.300000        -8.1634",
        " 183   0.7500      -0.200000        -5.4423",
        "",
        "CAS-SCF STATES FOR BLOCK  0 MULT= 4 NROOTS=2",
        "ROOT   0:  E=   -101.5000000000 Eh",
        "      1.00000 [     0]: 111000",
        "ROOT   1:  E=   -101.1000000000 Eh  10.884 eV  87770.0 cm**-1",
        "      0.70000 [     1]: 110100",
        "      0.30000 [     2]: 110010",
        "SA-CASSCF TRANSITION ENERGIES",
        "LOWEST ROOT (ROOT 0 ,MULT 4) =  -101.500000000 Eh -2761.000 eV",
        "STATE   ROOT MULT  DE/a.u.    DE/eV    DE/cm**-1",
        "   1:    1    4   0.400000    10.884  87770.0",
        "",
        "DENSITY MATRIX",
        "                  0          1",
        "      0       0.800000   0.100000",
        "      1       0.100000   0.200000",
        "Trace of the electron density:  1.000000",
        "SPIN-DENSITY MATRIX",
        "                  0          1",
        "      0       0.600000   0.000000",
        "      1       0.000000   0.400000",
        "Trace of the spin density:  1.000000",
        "ENERGY COMPONENTS",
        "One electron energy          :   -10.000000000 Eh        -272.1139 eV",
        "Two electron energy          :     1.000000000 Eh          27.2114 eV",
        "Virial ratio                 :     -2.000000000",
        "LOEWDIN ORBITAL-COMPOSITIONS",
        "                    181       182       183       184",
        "                  -0.4      -0.3      -0.2       0.1",
        "                   2.0       0.8       0.7       0.0",
        "                  --------  --------  --------  --------",
        " 0 N  px              0.0      95.0       1.0       0.0",
        "",
        "LOEWDIN REDUCED ACTIVE MOs",
        "                    182       183",
        "                  -0.3      -0.2",
        "                   0.8       0.7",
        "                  --------  --------",
        " 0 N  px             95.0       1.0",
        "",
        "ORCA POPULATION ANALYSIS",
    ]

    data = CASSCFModule(_casscf_context()).parse(lines)

    assert data is not None
    assert data["convergence"]["converged"] is True
    assert data["convergence"]["n_macro_iterations"] == 3
    assert data["convergence"]["macro_iterations"][0]["orbital_options"][0]["percent"] == 100.0
    assert data["convergence"]["final"]["phase"] == "final_printing"
    assert data["convergence"]["final"]["active_occupations"] == [0.86, 0.74, 0.2]
    assert data["state_blocks"][0]["roots"][1]["dominant_configuration"]["occupation_string"] == "110100"
    assert data["transition_energies"]["states"][0]["delta_energy_eV"] == 10.884
    assert data["density_matrix"]["matrix"] == [[0.8, 0.1], [0.1, 0.2]]
    assert data["spin_density_matrix"]["trace"] == 1.0
    assert data["energy_components"]["components"]["virial_ratio"]["unit"] == "ratio"
    assert data["loewdin_reduced_active_mos"]["orbitals"][0]["contributions"][0]["percent"] == 95.0
    assert data["loewdin_orbital_compositions"]["n_orbitals"] == 4


def test_casscf_module_parses_nevpt2_and_qd_nevpt2_tables() -> None:
    lines = [
        "ORCA-CASSCF",
        "NEVPT2 Results",
        "    MULT 4, ROOT 0",
        "  Class V0_ijab :\t dE = -1.000000000000",
        " \t Total Energy Correction : dE = -1.25000000000000",
        " \t Reference  Energy       : E0 = -100.00000000000000",
        " \t Total Energy (E0+dE)    : E  = -101.25000000000000",
        "NEVPT2 TOTAL ENERGIES",
        "STATE   ROOT MULT  Energy/a.u.   MRCI SOC BLOCK INPUT (Eh)",
        "   0:    0    4  -101.250000    EDIAG[0]  -101.250000000",
        "",
        "NEVPT2 TRANSITION ENERGIES",
        "LOWEST ROOT (ROOT 0, MULT 4) =  -101.250000000 Eh -2755.000 eV",
        "STATE ROOT MULT  DE/a.u.     DE/eV    DE/cm**-1",
        "   1:   0    2   0.085604     2.329  18788.0",
        "",
        "NEVPT2 CORRECTION TO THE TRANSITION ENERGY",
        "STATE  ROOT MULT DE/a.u. DE/eV \tDE/cm**-1",
        "   1:    0    2  -0.025088     -0.683  -5506.3",
        "QD-NEVPT2 Results",
        "   *********************",
        "    MULT 4",
        "   *********************",
        "   Total Hamiltonian to be processed",
        "                  0",
        "      0     -101.260000",
        " --------------------------",
        " \t ROOT = 0",
        " --------------------------",
        "      0.99999 [     0]: 111000",
        " \t Total Energy Correction : dE = -1.26000000000000",
        " \t Zero Order Energy       : E0 = -100.00000000000000",
        " \t Total Energy (E0+dE)    : E  = -101.26000000000000",
        "QD-NEVPT2 TOTAL ENERGIES",
        "STATE   ROOT MULT  Energy/a.u.   MRCI SOC BLOCK INPUT (cm**-1)",
        "   0:    0    4  -101.260000    EDIAG[0] -222222.100000",
        "",
        "QD-NEVPT2 TRANSITION ENERGIES",
        "LOWEST ROOT (ROOT 0, MULT 4) =  -101.260000000 Eh -2756.000 eV",
        "STATE ROOT MULT  DE/a.u.     DE/eV    DE/cm**-1",
        "   1:   0    2   0.085605     2.330  18789.5",
        "",
        "TIMINGS NEVPT2",
        "Sum of individual times        ...    10.0 sec (100.0%)",
    ]

    data = CASSCFModule(_casscf_context()).parse(lines)

    assert data is not None
    assert data["nevpt2"]["state_results"][0]["class_corrections"]["V0_ijab"] == -1.0
    assert data["nevpt2"]["total_energies"][0]["ediag_unit"] == "Eh"
    assert data["nevpt2"]["transition_energies"]["states"][0]["delta_energy_eV"] == 2.329
    assert data["nevpt2"]["transition_energy_corrections"][0]["delta_energy_cm-1"] == -5506.3
    assert data["nevpt2"]["qd_nevpt2"]["state_results"][0]["configurations"][0]["occupation_string"] == "111000"
    assert data["nevpt2"]["qd_nevpt2"]["state_results"][0]["hamiltonian"]["matrix"] == [[-101.26]]
    assert data["nevpt2"]["qd_nevpt2"]["total_energies"][0]["ediag_unit"] == "cm-1"
    assert data["nevpt2"]["qd_nevpt2"]["transition_energies"]["states"][0]["delta_energy_eV"] == 2.33
    assert data["nevpt2"]["timings"][0]["percent"] == 100.0


def test_orca_parser_normalizes_casscf_job_snapshot_labels(tmp_path: Path) -> None:
    out_path = tmp_path / "minimal_casscf.out"
    out_path.write_text(
        "\n".join(
            [
                "Program Version 6.1.0",
                "An Ab Initio, DFT and Semiempirical electronic structure package",
                "================================================================================",
                "                                       INPUT FILE",
                "================================================================================",
                "NAME = minimal_casscf.inp",
                "|  1> ! MOREAD",
                "|  2> %casscf",
                "|  3>   nel 3",
                "|  4>   norb 6",
                "|  5> end",
                "                         ****END OF INPUT****",
                "Hartree-Fock type      HFTyp           .... CASSCF",
                "Multiplicity           Mult            ....    4",
                "ORCA-CASSCF",
                "CASSCF RESULTS",
                "Final CASSCF energy       : -101.100000000 Eh  -2751.0000 eV",
                "ORCA TERMINATED NORMALLY",
                "TOTAL RUN TIME: 0 days 0 hours 0 minutes 1 seconds",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    data = ORCAParser(out_path).parse(sections=["casscf"])

    assert data["job_snapshot"]["calculation_family"] == "casscf"
    assert data["job_snapshot"]["method_header_label"] == "CASSCF"


def test_casscf_markdown_renders_full_configs_nevpt2_and_raw_blocks(tmp_path: Path) -> None:
    from orca_parser.output.markdown_writer import write_markdown

    lines = [
        "ORCA-CASSCF",
        "CAS-SCF ITERATIONS",
        "MACRO-ITERATION   1:",
        "   E(CAS)= -101.000000000 Eh DE=    0.000000e+00",
        "   --- Energy gap subspaces: Ext-Act = 0.399   Act-Int = 0.011",
        "   N(occ)=  0.84 0.75 0.20",
        "   ||g|| =     2.670751e-01 Max(G)=   -1.121894e-01 Rot=188,61",
        "                    ---- DOING ONE FINAL ITERATION FOR PRINTING ----",
        "MACRO-ITERATION   2:",
        "   E(CAS)= -101.000001000 Eh DE=   -1.000000e-06",
        "   --- Energy gap subspaces: Ext-Act = 0.399   Act-Int = 0.011",
        "   N(occ)=  0.85 0.74 0.20",
        "   ||g|| =     1.670751e-03 Max(G)=   -1.121894e-04 Rot=188,61",
        "CASSCF RESULTS",
        "CAS-SCF STATES FOR BLOCK  1 MULT= 2 NROOTS=1",
        "ROOT   1:  E=   -101.1000000000 Eh  0.003 eV     27.7 cm**-1",
        "      0.32772 [    24]: 012000",
        "      0.28237 [     0]: 210000",
        "SA-CASSCF TRANSITION ENERGIES",
        "LOWEST ROOT (ROOT 1 ,MULT 2) =  -101.100000000 Eh -2751.000 eV",
        "STATE   ROOT MULT  DE/a.u.    DE/eV    DE/cm**-1",
        "",
        "NEVPT2 TOTAL ENERGIES",
        "STATE   ROOT MULT  Energy/a.u.   MRCI SOC BLOCK INPUT (Eh)",
        "   0:    0    4  -101.250000    EDIAG[0]  -101.250000000",
        "",
        "NEVPT2 TRANSITION ENERGIES",
        "STATE ROOT MULT  DE/a.u.     DE/eV    DE/cm**-1",
        "   1:   0    2   0.085604     2.329  18788.0",
        "",
        "NEVPT2 CORRECTION TO THE TRANSITION ENERGY",
        "STATE  ROOT MULT DE/a.u. DE/eV \tDE/cm**-1",
        "   1:    0    2  -0.025088     -0.683  -5506.3",
        "QD-NEVPT2 Results",
        "    MULT 4",
        "   Total Hamiltonian to be processed",
        "                  0",
        "      0     -101.260000",
        " \t ROOT = 0",
        "      0.99999 [     0]: 111000",
        " \t Total Energy Correction : dE = -1.26000000000000",
        " \t Zero Order Energy       : E0 = -100.00000000000000",
        " \t Total Energy (E0+dE)    : E  = -101.26000000000000",
        "QD-NEVPT2 TOTAL ENERGIES",
        "STATE   ROOT MULT  Energy/a.u.   MRCI SOC BLOCK INPUT (cm**-1)",
        "   0:    0    4  -101.260000    EDIAG[0] -222222.100000",
        "",
        "QD-NEVPT2 TRANSITION ENERGIES",
        "STATE ROOT MULT  DE/a.u.     DE/eV    DE/cm**-1",
        "   1:   0    2   0.085605     2.330  18789.5",
        "",
        "DENSITY MATRIX (QD-NEVPT2 CORRECTED)",
        "                  0          1",
        "      0       0.800000   0.100000",
        "      1       0.100000   0.200000",
        "Trace of the electron density:  1.000000",
        "SPIN-DENSITY MATRIX (QD-NEVPT2 CORRECTED)",
        "                  0          1",
        "      0       0.600000   0.000000",
        "      1       0.000000   0.400000",
        "Trace of the spin density:  1.000000",
        "",
        "State-specific QD-NEVPT2 natural orbitals",
        "BLOCK 0 (Multiplicity 4):",
        "Root 0:",
        " N(occ) =  1.00000  1.00000  1.00000  0.00000  0.00000  0.00000",
        "         ---> stored as NC.mult.4.iroot.0.QD-NEVPT2.natorbs",
        " *** QD-NEVPT2: Repeating the population analysis with the corrected densities ***",
        "LOEWDIN REDUCED ACTIVE MOs",
        "                    182",
        "                  -0.3",
        "                   0.8",
        "                  --------",
        " 0 N  px             95.0",
        "",
        "ORCA POPULATION ANALYSIS",
        "CASSCF UV, CD spectra and dipole moments",
        "ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS",
        "  0-4A  ->  1-4A    9.385168   75696.5   132.1   0.000000000   0.00000  -0.00000  -0.00000  -0.00000",
        "CD SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS",
        "  0-4A  ->  1-4A    9.385168   75696.5   132.1   -0.00000  -0.00170   0.00000   0.00689",
        "CASSCF RELATIVISTIC PROPERTIES",
        "QDPT WITH CASSCF DIAGONAL ENERGIES",
        "Doing QDPT with ONLY SOC!",
        "Lowest eigenvalue of the SOC matrix:   -101.26000000 Eh",
        "Energy stabilization:    -0.16760 cm-1",
        "Eigenvalues:     cm-1         eV      Boltzmann populations at T =  300.000 K",
        "   0:            0.00        0.0000       2.50e-01",
        "The threshold for printing is 0.0100",
        "Eigenvectors:",
        "                         Weight      Real          Image    : Block Root    Spin   Ms",
        " STATE   0:       0.0000",
        "                         0.732753     0.145725     0.843515 :     0    0  3/2  3/2",
        "SOC CORRECTED ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS",
        "  0-4.0A  ->  1-4.0A    3.012090   24294.2   411.6   0.000000000   0.00000   0.00000   0.00000   0.00000",
        "SOC CORRECTED CD SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS",
        "  0-4.0A  ->  1-4.0A    3.012090   24294.2   411.6   -0.00000   0.00003   0.00000   0.00000",
        "ELECTRONIC G-MATRIX",
        "g-factors:",
        "   2.0021842    2.0021842    2.0021843 iso =    2.0021842",
        "ZERO-FIELD SPLITTING",
        "D   =    0.001914  cm-1",
        "E/D =    0.000173",
        "EPR properties:",
    ]
    casscf = CASSCFModule(_casscf_context()).parse(lines)
    assert casscf is not None

    out_path = tmp_path / "casscf.md"
    write_markdown({"source_file": "synthetic.out", "casscf": casscf}, out_path)
    text = out_path.read_text(encoding="utf-8")

    assert "N(occ)" in text
    assert "Final-printing iterations are bold" in text
    assert "DOING ONE FINAL ITERATION FOR PRINTING" not in text
    assert "0.32772 -> 012000" in text
    assert "0.28237 -> 210000" in text
    assert "[    24]" not in text
    assert "210000" in text
    assert "SA-CASSCF State Energies and Configurations" in text
    assert "NEVPT2 State Energies, Corrections, and Configurations" in text
    assert "QD-NEVPT2 State Energies and Corrected Configurations" in text
    assert "QD-NEVPT2 Total Hamiltonian" in text
    assert "QD-NEVPT2 Van Vleck Results" not in text
    assert "```text" not in text
    assert "QD-NEVPT2 Corrected Density Matrix" in text
    assert "QD-NEVPT2 Corrected Spin-Density Matrix" in text
    assert "QD-NEVPT2 Corrected Loewdin Reduced Active MOs" in text
    assert "QD-NEVPT2 State-Specific Natural Orbitals" in text
    assert "CASSCF UV, CD spectra and dipole moments - Absorption" in text
    assert "SOC-corrected spectra - Absorption" in text
    assert "SOC-corrected spectra - CD" in text
    assert "QDPT CASSCF DIAGONAL ENERGIES Relativistic Levels" in text
    assert "g-Matrix Summary" in text
    assert "Zero-Field Splitting Summary" in text


def test_casscf_alias_reuses_existing_population_sections() -> None:
    alias = get_parser_section_alias_map()["casscf"]

    assert alias[0] == "casscf"
    assert {"mulliken", "loewdin", "mayer", "hirshfeld", "mbis", "chelpg"}.issubset(alias)


def test_casscf_alias_parses_spin_density_population_headings(tmp_path: Path) -> None:
    out_path = tmp_path / "casscf_population.out"
    out_path.write_text(
        "\n".join(
            [
                "Program Version 6.1.0",
                "An Ab Initio, DFT and Semiempirical electronic structure package",
                "ORCA-CASSCF",
                "CASSCF RESULTS",
                "Final CASSCF energy       : -101.100000000 Eh  -2751.0000 eV",
                "ORCA POPULATION ANALYSIS",
                "MULLIKEN ATOMIC CHARGES AND SPIN DENSITIES",
                "------------------------------------------",
                "   0 N :    0.357772    1.570244",
                "Sum of atomic charges       :   -0.0000000",
                "Sum of atomic spin densities:    2.0000000",
                "MULLIKEN REDUCED ORBITAL CHARGES AND SPIN DENSITIES",
                "CHARGE",
                "  0 N s       :     4.008921  s :     4.008921",
                "SPIN",
                "  0 N s       :     1.000000  s :     1.000000",
                "LOEWDIN ATOMIC CHARGES AND SPIN DENSITIES",
                "-----------------------------------------",
                "   0 N :    0.100000    1.900000",
                "Sum of atomic charges       :    0.0000000",
                "Sum of atomic spin densities:    2.0000000",
                "LOEWDIN REDUCED ORBITAL CHARGES AND SPIN DENSITIES",
                "CHARGE",
                "  0 N s       :     3.900000  s :     3.900000",
                "SPIN",
                "  0 N s       :     0.900000  s :     0.900000",
                "ORCA TERMINATED NORMALLY",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    data = ORCAParser(out_path).parse(sections=["casscf"])

    assert data["mulliken"]["atomic_charges"][0]["spin_population"] == 1.570244
    assert data["mulliken"]["sum_of_spin_populations"] == 2.0
    assert data["mulliken"]["reduced_orbital_spin_populations"][0]["shell_totals"]["s"] == 1.0
    assert data["loewdin"]["atomic_charges"][0]["spin_population"] == 1.9
    assert data["casscf"]["population_analyses"][0]["section_keys"] == ["mulliken", "loewdin"]


def test_real_casscf_sample_keeps_corrected_and_reused_outputs() -> None:
    data = ORCAParser(CASSCF_SAMPLE).parse(sections=["casscf", "mulliken", "loewdin"])
    casscf = data["casscf"]
    qd = casscf["nevpt2"]["qd_nevpt2"]

    assert data["mulliken"]["atomic_charges"][0]["spin_population"] is not None
    assert data["loewdin"]["atomic_charges"][0]["spin_population"] is not None
    assert len(casscf["population_analyses"]) >= 20
    assert casscf["population_analyses"][0]["section_keys"] == ["mulliken", "loewdin"]
    assert qd["corrected_density_matrix"]["trace"] == 3.0
    assert qd["corrected_spin_density_matrix"]["trace"] == 2.0
    assert "raw_report_sections" not in casscf
    soc_sections = [section for section in casscf["spectra"] if section.get("qdpt_label")]
    assert len(soc_sections) == 3
    assert all(len(section["absorption"]) > 100 for section in soc_sections)
    assert all(len(section["cd"]) > 100 for section in soc_sections)
