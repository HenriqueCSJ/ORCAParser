"""Self-contained markdown section renderers.

These renderers are separated from ``markdown_writer.py`` so the main writer
can focus on document assembly instead of holding every section implementation
inline. The functions accept formatting helpers as callables, which keeps them
decoupled from the monolithic writer module and makes later extraction easier.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


FormatNumber = Callable[[Any, str], str]
MakeTable = Callable[[List[tuple]], str]
GetDipoleVector = Callable[[Dict[str, Any], str, str], Optional[Dict[str, float]]]
FormatVector = Callable[[Optional[Dict[str, Any]], str], str]


def render_dipole_section(
    dipole: Dict[str, Any],
    *,
    format_number: FormatNumber,
    make_table: MakeTable,
    get_dipole_vector: GetDipoleVector,
    format_vector: FormatVector,
) -> str:
    """Render permanent-dipole and rotational-axis dipole information."""
    if not dipole:
        return ""

    lines: List[str] = []
    magnitude_bits = []
    if dipole.get("magnitude_Debye") is not None:
        magnitude_bits.append(f"{format_number(dipole.get('magnitude_Debye'))} D")
    if dipole.get("magnitude_au") is not None:
        magnitude_bits.append(f"{format_number(dipole.get('magnitude_au'), '.6f')} a.u.")
    if magnitude_bits:
        lines.append("**Total magnitude:** " + " | ".join(magnitude_bits))

    total_au = get_dipole_vector(dipole, "total_dipole", "au")
    if total_au:
        lines.append(f"**Total vector (a.u.):** ({format_vector(total_au)})")
    total_debye = get_dipole_vector(dipole, "total_dipole", "Debye")
    if total_debye:
        lines.append(f"**Total vector (D):** ({format_vector(total_debye)})")

    component_rows = [("Contribution", "X (a.u.)", "Y (a.u.)", "Z (a.u.)")]
    for label, key in (
        ("Electronic", "electronic_contribution"),
        ("Nuclear", "nuclear_contribution"),
        ("Total", "total_dipole"),
    ):
        vector = get_dipole_vector(dipole, key, "au")
        if not vector:
            continue
        component_rows.append((
            label,
            format_number(vector.get("x"), ".6f"),
            format_number(vector.get("y"), ".6f"),
            format_number(vector.get("z"), ".6f"),
        ))
    if len(component_rows) > 1:
        lines.append("**Cartesian components (a.u.)**\n" + make_table(component_rows))

    rotational_bits = []
    rot_cm1 = dipole.get("rotational_constants_cm1")
    if rot_cm1:
        rotational_bits.append(
            "cm^-1 = (" + ", ".join(format_number(value, ".6f") for value in rot_cm1) + ")"
        )
    rot_mhz = dipole.get("rotational_constants_MHz")
    if rot_mhz:
        rotational_bits.append(
            "MHz = (" + ", ".join(format_number(value, ".3f") for value in rot_mhz) + ")"
        )
    if rotational_bits:
        lines.append("**Rotational constants:** " + " | ".join(rotational_bits))

    rot_axes_au = dipole.get("dipole_rot_axes_au")
    if rot_axes_au:
        lines.append(
            f"**Dipole on rotational axes (a.u.):** ({format_vector(rot_axes_au)})"
        )
    rot_axes_debye = dipole.get("dipole_rot_axes_Debye")
    if rot_axes_debye:
        lines.append(
            f"**Dipole on rotational axes (D):** ({format_vector(rot_axes_debye)})"
        )

    return "\n\n".join(lines)


def render_basis_set_section(
    basis_set: Dict[str, Any],
    *,
    make_table: MakeTable,
) -> str:
    """Compact basis-set summary for markdown reports."""
    lines: List[str] = []

    bits = []
    if basis_set.get("n_basis_functions") is not None:
        bits.append(f"**Basis functions:** {basis_set['n_basis_functions']}")
    if basis_set.get("n_shells") is not None:
        bits.append(f"**Shells:** {basis_set['n_shells']}")
    if basis_set.get("max_angular_momentum") is not None:
        bits.append(f"**Max l:** {basis_set['max_angular_momentum']}")
    if basis_set.get("n_groups") is not None:
        bits.append(f"**Distinct groups:** {basis_set['n_groups']}")
    if bits:
        lines.append(" | ".join(bits))

    groups = basis_set.get("groups") or {}
    if groups:
        rows = [("Group", "Element", "Description")]
        for idx in sorted(groups):
            group = groups[idx]
            rows.append((
                str(idx),
                str(group.get("element", "")),
                str(group.get("description", "")),
            ))
        lines.append(make_table(rows))

    mapping = basis_set.get("atom_group_mapping") or {}
    if mapping:
        counts: Dict[int, int] = {}
        for group_idx in mapping.values():
            counts[group_idx] = counts.get(group_idx, 0) + 1
        rows = [("Group", "Atoms mapped")]
        for idx in sorted(counts):
            rows.append((str(idx), str(counts[idx])))
        lines.append("**Atom-to-group counts**\n" + make_table(rows))

    return "\n\n".join(lines)


def _scan_surfaces_differ(surface_scan: Dict[str, Any]) -> bool:
    """Whether the actual and SCF scan surfaces differ meaningfully."""
    actual = surface_scan.get("surface_actual_energy") or []
    scf = surface_scan.get("surface_scf_energy") or []
    if not actual or not scf or len(actual) != len(scf):
        return bool(scf)
    for a_row, s_row in zip(actual, scf):
        a_energy = a_row.get("energy_Eh")
        s_energy = s_row.get("energy_Eh")
        if a_energy is None or s_energy is None:
            continue
        if abs(a_energy - s_energy) > 1.0e-9:
            return True
    return False


def render_surface_scan_section(
    surface_scan: Dict[str, Any],
    *,
    format_number: FormatNumber,
    make_table: MakeTable,
) -> str:
    """Compact relaxed surface scan summary for markdown reports."""
    lines: List[str] = []

    bits: List[str] = []
    if surface_scan.get("mode"):
        bits.append(f"**Mode:** {surface_scan['mode']}")
    if surface_scan.get("n_parameters") is not None:
        bits.append(f"**Parameters:** {surface_scan['n_parameters']}")
    if surface_scan.get("n_constrained_optimizations") is not None:
        bits.append(
            f"**Constrained optimizations:** "
            f"{surface_scan['n_constrained_optimizations']}"
        )
    if surface_scan.get("actual_energy_span_kcal_mol") is not None:
        bits.append(
            f"**Actual energy span:** "
            f"{format_number(surface_scan['actual_energy_span_kcal_mol'])} kcal/mol"
        )
    if surface_scan.get("scf_energy_span_kcal_mol") is not None and _scan_surfaces_differ(surface_scan):
        bits.append(
            f"**SCF energy span:** "
            f"{format_number(surface_scan['scf_energy_span_kcal_mol'])} kcal/mol"
        )
    if bits:
        lines.append(" | ".join(bits))

    parameters = surface_scan.get("parameters") or []
    if parameters:
        rows = [("label", "type", "atoms", "definition", "unit")]
        for parameter in parameters:
            atoms = ",".join(str(atom) for atom in parameter.get("atoms") or [])
            if parameter.get("mode") == "values":
                values = parameter.get("values") or []
                definition = ", ".join(format_number(value) for value in values)
            else:
                definition = (
                    f"{format_number(parameter.get('start'))} -> {format_number(parameter.get('end'))} "
                    f"({parameter.get('steps', '?')} steps)"
                )
            rows.append((
                parameter.get("label", "?"),
                parameter.get("coordinate_type", "?"),
                atoms or "?",
                definition,
                parameter.get("unit", ""),
            ))
        lines.append("**Scan Coordinates**\n" + make_table(rows))

    sidecars = surface_scan.get("sidecar_files") or {}
    sidecar_bits: List[str] = []
    for key in ("actual_surface_dat", "scf_surface_dat", "allxyz", "xyzall", "trajectory_xyz"):
        if sidecars.get(key):
            sidecar_bits.append(f"{key}=`{Path(sidecars[key]).name}`")
    if sidecars.get("allxyz_frame_count") is not None:
        sidecar_bits.append(f"frames={sidecars['allxyz_frame_count']}")
    if sidecar_bits:
        lines.append("**Sidecar files:** " + "  ".join(sidecar_bits))

    steps = surface_scan.get("steps") or []
    if steps:
        parameter_labels = steps[0].get("coordinate_labels") or [
            parameter.get("label", f"coord {idx + 1}")
            for idx, parameter in enumerate(parameters)
        ]
        header = ("step",) + tuple(parameter_labels) + ("E_actual (Eh)", "dE_actual (kcal/mol)")
        include_scf = _scan_surfaces_differ(surface_scan) or any(
            row.get("scf_energy_Eh") is not None for row in steps
        )
        if include_scf:
            header += ("E_SCF (Eh)", "dE_SCF (kcal/mol)")
        include_xyz = any(row.get("optimized_xyz_file") for row in steps)
        if include_xyz:
            header += ("xyz",)

        rows: List[tuple] = [header]
        for step in steps:
            row = (
                str(step.get("step", "")),
                *tuple(format_number(value) for value in (step.get("coordinate_values") or [])),
                format_number(step.get("actual_energy_Eh"), ".10f"),
                format_number(step.get("relative_actual_energy_kcal_mol")),
            )
            if include_scf:
                row += (
                    format_number(step.get("scf_energy_Eh"), ".10f"),
                    format_number(step.get("relative_scf_energy_kcal_mol")),
                )
            if include_xyz:
                row += (step.get("optimized_xyz_file", "—"),)
            rows.append(row)
        lines.append("**Surface Profile**\n" + make_table(rows))

    return "\n\n".join(lines)


def render_geom_opt_section(
    geom_opt: Dict[str, Any],
    *,
    format_number: FormatNumber,
    make_table: MakeTable,
) -> str:
    """Compact geometry-optimization summary for markdown reports."""
    lines: List[str] = []

    bits = []
    converged = geom_opt.get("converged")
    if converged is not None:
        bits.append(f"**Converged:** {'yes' if converged else 'no'}")
    if geom_opt.get("n_cycles") is not None:
        bits.append(f"**Cycles:** {geom_opt['n_cycles']}")
    if geom_opt.get("opt_keyword"):
        bits.append(f"**Keyword:** {geom_opt['opt_keyword']}")
    if geom_opt.get("final_energy_Eh") is not None:
        bits.append(f"**Final energy:** {format_number(geom_opt['final_energy_Eh'], '.10f')} Eh")
    if bits:
        lines.append(" | ".join(bits))

    rmsd_bits = []
    if geom_opt.get("rmsd_initial_to_final_ang") is not None:
        rmsd_bits.append(
            f"**RMSD initial->final:** {format_number(geom_opt['rmsd_initial_to_final_ang'])} A"
        )
    if geom_opt.get("mass_weighted_rmsd_initial_to_final_ang") is not None:
        rmsd_bits.append(
            "**MW RMSD initial->final:** "
            f"{format_number(geom_opt['mass_weighted_rmsd_initial_to_final_ang'])} A"
        )
    if rmsd_bits:
        lines.append(" | ".join(rmsd_bits))

    settings = geom_opt.get("settings") or {}
    if settings:
        rows = [("Setting", "Value")]
        for key, value in settings.items():
            rows.append((str(key), str(value)))
        lines.append("**Settings**\n" + make_table(rows))

    tolerances = geom_opt.get("tolerances") or {}
    if tolerances:
        rows = [("Tolerance", "Value")]
        for key, value in tolerances.items():
            rows.append((str(key), str(value)))
        lines.append("**Convergence Tolerances**\n" + make_table(rows))

    cycles = geom_opt.get("cycles") or []
    if cycles:
        rows = [[
            "Cycle",
            "E (Eh)",
            "dE (Eh)",
            "RMS grad",
            "Max grad",
            "RMS step",
            "Max step",
            "Trust (bohr)",
        ]]
        cycle_entries: List[Optional[Dict[str, Any]]] = list(cycles)
        note = ""
        if len(cycle_entries) > 12:
            cycle_entries = cycle_entries[:6] + [None] + cycle_entries[-6:]
            note = f"*Showing first 6 and last 6 of {len(cycles)} cycles.*"

        for cyc in cycle_entries:
            if cyc is None:
                rows.append(("...", "...", "...", "...", "...", "...", "...", "..."))
                continue
            conv = cyc.get("convergence") or {}
            rows.append((
                str(cyc.get("cycle", "")),
                format_number(cyc.get("energy_Eh"), ".10f"),
                format_number(cyc.get("energy_change_Eh"), ".6f"),
                format_number((conv.get("rms_gradient") or {}).get("value"), ".2e"),
                format_number((conv.get("max_gradient") or {}).get("value"), ".2e"),
                format_number((conv.get("rms_step") or {}).get("value"), ".2e"),
                format_number((conv.get("max_step") or {}).get("value"), ".2e"),
                format_number(cyc.get("trust_radius_bohr"), ".3f"),
            ))

        cycle_table = make_table(rows)
        if note:
            cycle_table += f"\n\n{note}"
        lines.append("**Cycle Summary**\n" + cycle_table)

    return "\n\n".join(lines)


def render_solvation_section(
    solvation: Dict[str, Any],
    *,
    format_number: FormatNumber,
    make_table: MakeTable,
) -> str:
    """Compact implicit-solvation summary for markdown reports."""
    summary = solvation.get("summary", {})
    lines: List[str] = []

    if solvation.get("is_solvated"):
        bits = []
        if solvation.get("primary_model"):
            bits.append(f"**Model:** {solvation['primary_model']}")
        if solvation.get("solvent"):
            bits.append(f"**Solvent:** {solvation['solvent']}")
        if summary.get("models"):
            bits.append(f"**Seen:** {', '.join(summary['models'])}")
        if bits:
            lines.append(" | ".join(bits))
    else:
        message = "**Final Input State:** no implicit solvation detected"
        if summary.get("output_model"):
            message += f" (earlier output blocks seen: {summary['output_model']}"
            if summary.get("output_solvent"):
                message += f" in {summary['output_solvent']}"
            message += ")"
        lines.append(message)

    flags = []
    if solvation.get("input_flags", {}).get("draco"):
        flags.append("DRACO")
    if solvation.get("input_flags", {}).get("smd18"):
        flags.append("SMD18")
    if flags:
        lines.append("**Flags:** " + ", ".join(flags))

    model = solvation.get("primary_model")
    if model in {"CPCM", "CPCMC", "SMD"} and solvation.get("cpcm"):
        cpcm = solvation["cpcm"]
        rows = [("parameter", "value")]
        if cpcm.get("epsilon") is not None:
            rows.append(("epsilon", format_number(cpcm.get("epsilon"))))
        if cpcm.get("refractive_index") is not None:
            rows.append(("refrac", format_number(cpcm.get("refractive_index"))))
        if cpcm.get("rsolv_ang") is not None:
            rows.append(("rsolv (A)", format_number(cpcm.get("rsolv_ang"))))
        if cpcm.get("surface_type"):
            rows.append(("surface", str(cpcm["surface_type"])))
        if cpcm.get("epsilon_function_type"):
            rows.append(("eps. function", str(cpcm["epsilon_function_type"])))
        if model == "SMD" and cpcm.get("smd_descriptors"):
            for key in ("soln", "soln25", "sola", "solb", "solg", "solc", "solh"):
                if key in cpcm["smd_descriptors"]:
                    rows.append((key, format_number(cpcm["smd_descriptors"][key])))
        if len(rows) > 1:
            lines.append(make_table(rows))
    elif model == "ALPB" and solvation.get("alpb"):
        alpb = solvation["alpb"]
        rows = [("parameter", "value")]
        if alpb.get("epsilon") is not None:
            rows.append(("epsilon", format_number(alpb.get("epsilon"))))
        if alpb.get("reference_state"):
            rows.append(("reference state", str(alpb["reference_state"])))
        if alpb.get("free_energy_shift_kcal_mol") is not None:
            rows.append(("free energy shift (kcal/mol)", format_number(alpb.get("free_energy_shift_kcal_mol"))))
        if alpb.get("interaction_kernel"):
            rows.append(("kernel", str(alpb["interaction_kernel"])))
        if len(rows) > 1:
            lines.append(make_table(rows))
    elif model == "COSMO-RS" and solvation.get("cosmors"):
        cosmors = solvation["cosmors"]
        rows = [("parameter", "value")]
        if cosmors.get("functional"):
            rows.append(("functional", str(cosmors["functional"])))
        if cosmors.get("basis_set"):
            rows.append(("basis", str(cosmors["basis_set"])))
        if cosmors.get("dGsolv_kcal_mol") is not None:
            rows.append(("dGsolv (kcal/mol)", format_number(cosmors.get("dGsolv_kcal_mol"))))
        if len(rows) > 1:
            lines.append(make_table(rows))

    return "\n\n".join(lines)
