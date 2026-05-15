from __future__ import annotations

from pathlib import Path

import pytest

from orca_parser import ORCAParser
from orca_parser.modules.transition_orbitals import parse_natural_transition_orbitals


REPO_ROOT = Path(__file__).resolve().parents[1]
STEOM_SAMPLE = REPO_ROOT / "sample_outs" / "STEOM2" / "F3CNO.out"
CCSDT_SAMPLE = REPO_ROOT / "sample_outs" / "CCSDT" / "F3CNO.out"


def _require_sample(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"local private sample is not present: {path.name}")
    return path


def test_ccsdt_sample_parses_f12_triples_convergence_and_shared_sections() -> None:
    data = ORCAParser(_require_sample(CCSDT_SAMPLE)).parse(sections=["ccsdt"])
    cc = data["coupled_cluster"]

    assert data["job_snapshot"]["calculation_family"] == "coupled_cluster"
    assert cc["wavefunction"]["correlation_treatment"] == "CCSD"
    assert cc["wavefunction"]["perturbative_triple_excitations"] is True
    assert cc["wavefunction"]["calculation_of_f12_correction"] is True
    assert cc["wavefunction"]["internal_orbitals"]["electrons"] == 36

    assert cc["iterations"]["converged"] is True
    assert cc["iterations"]["final"]["iteration"] == 22
    assert cc["iterations"]["final"]["residual"] == pytest.approx(5e-9)

    assert cc["energy"]["total_energy_after_f12_Eh"] == pytest.approx(-467.099708267)
    assert cc["f12_correction"]["sum_correction_Eh"] == pytest.approx(-0.090707721353)
    assert cc["triples_correction"]["triples_correction_Eh"] == pytest.approx(-0.056498278)
    assert cc["triples_correction"]["scaled_triples_correction_Eh"] == pytest.approx(-0.059997843)
    assert cc["triples_correction"]["f12_ccsdt_energy_Eh"] == pytest.approx(-467.156206545)
    assert cc["natural_orbital_occupations"]["n_orbitals"] > 100

    # The CC alias delegates ordinary population/NBO/dipole parsing to the
    # shared owner modules instead of re-implementing them locally.
    assert data.get("mulliken")
    assert data.get("loewdin")
    assert data.get("nbo")
    assert data.get("dipole")
    assert "raw_report_sections" not in cc


def test_steom_sample_parses_eom_steom_spectra_ntos_and_avoids_tddft_duplication() -> None:
    data = ORCAParser(_require_sample(STEOM_SAMPLE)).parse(sections=["steom"])
    eom = data["eom_steom"]
    cc = data["coupled_cluster"]

    assert data["job_snapshot"]["calculation_family"] == "eom_steom"
    assert cc["summary"]["cc_converged"] is True
    assert eom["active_space_selection"] == {"ip_active_roots": 3, "ea_active_roots": 2}
    assert len(eom["eom_blocks"]) == 2

    steom_roots = eom["steom"]["roots"]
    assert len(steom_roots) == 2
    assert steom_roots[0]["energy_eV"] == pytest.approx(1.807)
    assert steom_roots[0]["dominant_amplitude"]["amplitude"] == pytest.approx(-0.987862)
    assert steom_roots[0]["dominant_amplitude"]["excitation"] == "23 -> 24"
    assert steom_roots[1]["active_character_percent"] == pytest.approx(92.78)
    assert steom_roots[1]["warnings"]

    spectra = eom["spectra"]
    assert spectra["table_count"] == 12
    assert set(spectra["contexts"]) == {
        "right_transition_moments",
        "left_transition_moments",
        "left_right_transition_moments",
    }
    left_right = spectra["contexts"]["left_right_transition_moments"]["tables"]
    assert "absorption_electric_dipole" in left_right
    assert "cd_electric_dipole" in left_right

    ntos = eom["nto_states"]
    assert ntos[0]["pairs"][0]["occupation"] == pytest.approx(0.99855874)
    assert ntos[1]["pairs"][0]["from_orbital"] == "23a"
    assert ntos[1]["pairs"][0]["to_orbital"] == "24a"
    assert ntos[1]["pairs"][0]["occupation"] == pytest.approx(0.99359505)

    all_sections = ORCAParser(_require_sample(STEOM_SAMPLE)).parse()
    assert "tddft" not in all_sections
    assert all_sections["eom_steom"]["summary"]["spectrum_table_count"] == 12


def test_shared_nto_parser_handles_orca_steom_rows_without_raw_text() -> None:
    lines = [
        "NATURAL TRANSITION ORBITALS FOR STATE    2",
        "Natural Transition Orbitals were saved in calc.s2.nto",
        "Threshold for printing occupation numbers 0.001000",
        " E=   0.291229 au      7.925 eV    63917.5 cm**-1",
        "    23a ->  24a  : n=  0.99359505",
        "    22a ->  25a  : n=  0.00397138",
    ]

    states = parse_natural_transition_orbitals(lines)

    assert states == [
        {
            "state": 2,
            "pairs": [
                {
                    "from_orbital": "23a",
                    "to_orbital": "24a",
                    "from_index": 23,
                    "from_spin": "a",
                    "to_index": 24,
                    "to_spin": "a",
                    "occupation": 0.99359505,
                },
                {
                    "from_orbital": "22a",
                    "to_orbital": "25a",
                    "from_index": 22,
                    "from_spin": "a",
                    "to_index": 25,
                    "to_spin": "a",
                    "occupation": 0.00397138,
                },
            ],
            "output_file": "calc.s2.nto",
            "print_threshold": 0.001,
            "energy_au": 0.291229,
            "energy_eV": 7.925,
            "energy_cm1": 63917.5,
            "wavelength_nm": pytest.approx(156.45167598857904),
            "pair_count": 2,
        }
    ]
