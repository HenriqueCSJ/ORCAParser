from pathlib import Path

from orca_workbench.service import (
    available_section_choices,
    build_provenance_text,
    build_workbench_summary,
    collect_warnings,
    discover_orca_outputs,
)


def test_discover_orca_outputs_skips_auxiliary_and_cache_dirs(tmp_path):
    keep = tmp_path / "calc.out"
    keep.write_text("not parsed in this test", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    keep_log = nested / "other.log"
    keep_log.write_text("not parsed in this test", encoding="utf-8")
    auxiliary = nested / "calc_atom83.out"
    auxiliary.write_text("helper", encoding="utf-8")
    diagnostic = nested / "calc_diag.log"
    diagnostic.write_text("helper", encoding="utf-8")
    cache = tmp_path / ".pytest_tmp"
    cache.mkdir()
    cached = cache / "stale.out"
    cached.write_text("cache", encoding="utf-8")

    discovered = discover_orca_outputs([tmp_path])

    assert {path.name for path in discovered} == {"calc.out", "other.log"}
    assert all(path != auxiliary for path in discovered)
    assert all(path != diagnostic for path in discovered)
    assert all(path != cached for path in discovered)


def test_available_section_choices_include_dynamic_aliases():
    choices = available_section_choices()
    keys = {(choice.kind, choice.key) for choice in choices}

    assert ("alias", "all") in keys
    assert ("alias", "casscf") in keys
    assert ("section", "tddft") in keys


def test_workbench_summary_prefers_normalized_views(tmp_path):
    source = tmp_path / "sample.out"
    data = {
        "source_file": str(source),
        "context": {"reference_type": "RKS"},
        "metadata": {},
        "job_snapshot": {
            "job_name": "sample",
            "calculation_type": "Excited-State Geometry Optimization",
            "basis_set": "def2-SVP",
            "method_header_label": "wB97X-D4/def2-SVP",
            "charge": 0,
            "multiplicity": 1,
            "reference_type": "RKS",
        },
        "final_snapshot": {
            "selection": "last_reported_optimization_step",
            "orbital_energies": {"HOMO_LUMO_gap_eV": 4.2},
            "dipole": {"magnitude_Debye": 7.5},
            "charges": {"mulliken": {"atoms": []}},
        },
        "scf": {"final_single_point_energy_Eh": -123.456789},
        "geom_opt": {"n_cycles": 5, "converged": True},
    }

    summary = build_workbench_summary(data, source, warning_count=2)
    text = summary.as_text()
    provenance = build_provenance_text(data)

    assert summary.job_name == "sample"
    assert summary.homo_lumo_gap_eV == 4.2
    assert summary.dipole_D == 7.5
    assert "Optimization cycles: 5" in text
    assert "selection: last_reported_optimization_step" in provenance
    assert "charge schemes: mulliken" in provenance


def test_collect_warnings_finds_parse_errors_and_nested_warnings():
    data = {
        "nbo_parse_error": "bad NBO block",
        "tddft": {
            "trajectory": {
                "warnings": ["root following ambiguity"],
            }
        },
    }

    warnings = collect_warnings(data)

    assert "nbo_parse_error: bad NBO block" in warnings
    assert "tddft.trajectory.warnings: root following ambiguity" in warnings
