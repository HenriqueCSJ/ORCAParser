"""Markdown renderers for job-state specific sections.

These sections cover ORCA job modes that tend to grow over time:
symmetry-aware jobs, DeltaSCF excited states, and excited-state
optimizations. Keeping them separate from ``markdown_writer.py`` helps the
main writer stay focused on document assembly.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List


FormatNumber = Callable[[Any, str], str]
MakeTable = Callable[[List[tuple]], str]
YesNoUnknown = Callable[[Any], str]
HasSymmetry = Callable[[Dict[str, Any]], bool]
GetSymmetryData = Callable[[Dict[str, Any]], Dict[str, Any]]
SymmetryOnOff = Callable[[Dict[str, Any]], str]
HasSymmetrySetup = Callable[[Dict[str, Any]], bool]
GetDeltaSCFData = Callable[[Dict[str, Any]], Dict[str, Any] | None]
DeltaSCFTargetSummary = Callable[[Dict[str, Any]], str]
FormatDeltaSCFVector = Callable[[Any], str]
GetExcitedStateOptData = Callable[[Dict[str, Any]], Dict[str, Any] | None]
ExcitedStateTargetLabel = Callable[[Dict[str, Any]], str]


def render_symmetry_section(
    data: Dict[str, Any],
    *,
    has_symmetry: HasSymmetry,
    get_symmetry_data: GetSymmetryData,
    symmetry_on_off: SymmetryOnOff,
    has_symmetry_setup: HasSymmetrySetup,
    yes_no_unknown: YesNoUnknown,
    format_number: FormatNumber,
    make_table: MakeTable,
) -> str:
    """Render a dedicated symmetry summary for UseSym / irrep-aware jobs."""
    if not has_symmetry(data):
        return ""

    sym = get_symmetry_data(data)
    geom = data.get("geometry", {})
    orbital_energies = data.get("orbital_energies", {})

    lines: List[str] = []

    summary_rows = [("field", "value")]
    if "use_sym" in sym or sym.get("use_sym_label"):
        summary_rows.append(("UseSym", symmetry_on_off(sym)))
    if sym.get("auto_detected_point_group"):
        summary_rows.append(("Auto-detected point group", str(sym["auto_detected_point_group"])))
    if sym.get("point_group"):
        summary_rows.append(("Point group", str(sym["point_group"])))
    if sym.get("reduced_point_group"):
        summary_rows.append(("Reduced point group", str(sym["reduced_point_group"])))
    if sym.get("orbital_irrep_group"):
        summary_rows.append(("Orbital irrep group", str(sym["orbital_irrep_group"])))
    if sym.get("petite_list_algorithm_label"):
        summary_rows.append(("Petite-list algorithm", str(sym["petite_list_algorithm_label"])))
    if sym.get("n_irreps") is not None:
        summary_rows.append(("Number of irreps", str(sym["n_irreps"])))
    if sym.get("initial_guess_irrep"):
        summary_rows.append(("Initial guess irrep", str(sym["initial_guess_irrep"])))
    if geom.get("symmetry_perfected_point_group"):
        summary_rows.append((
            "Symmetry-perfected geometry",
            f"{geom['symmetry_perfected_point_group']} ({len(geom.get('symmetry_cartesian_au', []))} atoms)",
        ))
    if len(summary_rows) > 1:
        lines.append(make_table(summary_rows))

    if has_symmetry_setup(sym):
        guess_rows = [("field", "value")]
        if sym.get("initial_guess_method"):
            guess_rows.append(("Initial guess method", str(sym["initial_guess_method"])))
        if sym.get("initial_guess_source_file"):
            guess_rows.append(("Guess MO file", str(sym["initial_guess_source_file"])))
        if "initial_guess_geometry_matches" in sym:
            guess_rows.append(("Input geometry matches", yes_no_unknown(sym.get("initial_guess_geometry_matches"))))
        if "initial_guess_basis_matches" in sym:
            guess_rows.append(("Input basis matches", yes_no_unknown(sym.get("initial_guess_basis_matches"))))
        if "initial_guess_irreps_reassigned" in sym:
            guess_rows.append((
                "Irreps reassigned after cleanup",
                yes_no_unknown(sym.get("initial_guess_irreps_reassigned")),
            ))
        if "initial_guess_mos_renormalized" in sym:
            guess_rows.append(("MOs renormalized", yes_no_unknown(sym.get("initial_guess_mos_renormalized"))))
        if "initial_guess_mos_reorthogonalized" in sym:
            reorth_value = yes_no_unknown(sym.get("initial_guess_mos_reorthogonalized"))
            method = sym.get("initial_guess_reorthogonalization_method")
            if method and reorth_value == "yes":
                reorth_value = f"{reorth_value} ({method})"
            guess_rows.append(("MOs reorthogonalized", reorth_value))
        lines.append("**Initial Guess / Symmetry Cleanup**\n" + make_table(guess_rows))

    setup_bits = []
    if sym.get("setup_rms_distance_au") is not None:
        setup_bits.append(f"RMS correction={format_number(sym.get('setup_rms_distance_au'), '.6g')} au")
    if sym.get("setup_max_distance_au") is not None:
        setup_bits.append(f"max correction={format_number(sym.get('setup_max_distance_au'), '.6g')} au")
    if sym.get("setup_threshold_au") is not None:
        setup_bits.append(f"threshold={format_number(sym.get('setup_threshold_au'), '.6g')} au")
    if sym.get("setup_time_s") is not None:
        setup_bits.append(f"setup time={format_number(sym.get('setup_time_s'), '.3f')} s")
    if setup_bits:
        lines.append(" | ".join(setup_bits))

    irreps = sym.get("irreps", [])
    alpha_occ = orbital_energies.get("alpha_occupied_per_irrep") or {}
    beta_occ = orbital_energies.get("beta_occupied_per_irrep") or {}
    occupied = orbital_energies.get("occupied_per_irrep") or {}

    irrep_order: List[str] = [entry.get("label", "") for entry in irreps if entry.get("label")]
    for mapping in (alpha_occ, beta_occ, occupied):
        for label in mapping:
            if label not in irrep_order:
                irrep_order.append(label)

    if irrep_order:
        irreps_by_label = {entry.get("label"): entry for entry in irreps}
        header: List[str] = ["Irrep"]
        include_basis = bool(irreps)
        if include_basis:
            header.extend(["SA basis fns", "Offset"])
        if alpha_occ and beta_occ:
            header.extend(["α occ", "β occ"])
        elif occupied:
            header.append("occ")

        rows = [tuple(header)]
        for label in irrep_order:
            row: List[str] = [label]
            if include_basis:
                entry = irreps_by_label.get(label, {})
                row.extend([
                    str(entry.get("n_basis_functions", "?")),
                    str(entry.get("offset", "?")),
                ])
            if alpha_occ and beta_occ:
                row.extend([
                    str(alpha_occ.get(label, "?")),
                    str(beta_occ.get(label, "?")),
                ])
            elif occupied:
                row.append(str(occupied.get(label, "?")))
            rows.append(tuple(row))
        lines.append("**Irreps**\n" + make_table(rows))

    return "\n\n".join(lines)


def render_deltascf_section(
    data: Dict[str, Any],
    *,
    get_deltascf_data: GetDeltaSCFData,
    deltascf_target_summary: DeltaSCFTargetSummary,
    format_deltascf_vector: FormatDeltaSCFVector,
    yes_no_unknown: YesNoUnknown,
    make_table: MakeTable,
) -> str:
    """Render a dedicated DeltaSCF section for excited-state SCF jobs."""
    deltascf = get_deltascf_data(data)
    if not deltascf:
        return ""

    lines = [
        "**This is a DeltaSCF excited-state SCF calculation, not a ground-state single-point.**",
    ]

    rows = [("field", "value"), ("Electronic state", "DeltaSCF excited-state")]
    target = deltascf_target_summary(deltascf)
    if target:
        rows.append(("Target configuration", target))
    if deltascf.get("aufbau_metric"):
        rows.append(("Aufbau metric", str(deltascf["aufbau_metric"])))
    if "keep_initial_reference" in deltascf:
        rows.append(("Keep initial reference", yes_no_unknown(deltascf.get("keep_initial_reference"))))
    lines.append(make_table(rows))

    alpha_occ = format_deltascf_vector(deltascf.get("alpha_occupation"))
    beta_occ = format_deltascf_vector(deltascf.get("beta_occupation"))
    if alpha_occ:
        lines.append(f"`Alpha target occupation:` {alpha_occ}")
    if beta_occ:
        lines.append(f"`Beta target occupation:` {beta_occ}")

    return "\n\n".join(lines)


def render_excited_state_opt_section(
    data: Dict[str, Any],
    *,
    get_excited_state_opt_data: GetExcitedStateOptData,
    excited_state_target_label: ExcitedStateTargetLabel,
    yes_no_unknown: YesNoUnknown,
    format_number: FormatNumber,
    make_table: MakeTable,
) -> str:
    """Render CIS/TDDFT excited-state geometry-optimization metadata."""
    excopt = get_excited_state_opt_data(data)
    if not excopt:
        return ""

    lines = [
        "**This geometry optimization targets an excited state, not the ground-state surface.**",
    ]

    rows = [("field", "value")]
    target = excited_state_target_label(excopt)
    if target:
        rows.append(("Target state", target))
    if excopt.get("target_root") is not None:
        rows.append(("Target root", str(excopt["target_root"])))
    if excopt.get("target_multiplicity"):
        rows.append(("Target manifold", str(excopt["target_multiplicity"])))
    if excopt.get("input_block"):
        rows.append(("Input block", f"%{excopt['input_block']}"))
    if excopt.get("input_nroots") is not None:
        rows.append(("NRoots", str(excopt["input_nroots"])))
    if "followiroot" in excopt:
        rows.append(("Follow IRoot", yes_no_unknown(excopt.get("followiroot"))))
    if "firkeepfirstref" in excopt:
        rows.append(("FIRKeepFirstRef", yes_no_unknown(excopt.get("firkeepfirstref"))))
    if "analytic_excited_state_gradients" in excopt:
        rows.append((
            "Analytic excited-state gradients",
            yes_no_unknown(excopt.get("analytic_excited_state_gradients")),
        ))
    if "socgrad" in excopt:
        rows.append(("SOC gradient", yes_no_unknown(excopt.get("socgrad"))))
    if excopt.get("final_root") is not None:
        rows.append(("Final root", str(excopt["final_root"])))
    if excopt.get("final_state_of_interest") is not None:
        rows.append(("Final state of interest", str(excopt["final_state_of_interest"])))
    if excopt.get("input_electron_density"):
        rows.append(("Input electron density", str(excopt["input_electron_density"])))
    if excopt.get("cispre_job_title"):
        rows.append(("CIS/TDDFT subjob", str(excopt["cispre_job_title"])))
    if excopt.get("gradient_block_count") is not None:
        rows.append(("Gradient evaluations", str(excopt["gradient_block_count"])))
    lines.append(make_table(rows))

    threshold_rows = [("keyword", "value")]
    for label, key in (
        ("FIRENTHRESH (eV)", "firen_thresh_eV"),
        ("FIRS2THRESH", "firs2_thresh"),
        ("FIRSTHRESH", "firsthresh"),
        ("FIRMINOVERLAP", "firminoverlap"),
        ("FIRDYNOVERLAP", "firdynoverlap"),
        ("FIRDYNOVERRATIO", "firdynoverratio"),
    ):
        value = excopt.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            rendered = ", ".join(format_number(item) for item in value)
        elif isinstance(value, bool):
            rendered = yes_no_unknown(value)
        else:
            rendered = str(value)
        threshold_rows.append((label, rendered))
    if len(threshold_rows) > 1:
        lines.append("**Root-Following Controls**\n" + make_table(threshold_rows))

    cycle_rows = [("cycle", "root", "state of interest", "DE (eV)", "E(tot) (Eh)", "followiroot", "density file")]
    for record in (excopt.get("cycle_records") or [])[:12]:
        cycle_rows.append((
            str(record.get("optimization_cycle", "—")),
            str(record.get("current_iroot") if record.get("current_iroot") is not None else record.get("root", "—")),
            str(record.get("state_of_interest", "—")),
            format_number(record.get("delta_energy_eV")),
            format_number(record.get("total_energy_Eh"), ".6f"),
            yes_no_unknown(record.get("followiroot_runtime")) if "followiroot_runtime" in record else "—",
            str(record.get("input_electron_density", "—")),
        ))
    if len(cycle_rows) > 1:
        lines.append("**Optimization Root History**\n" + make_table(cycle_rows))

    return "\n\n".join(lines)
