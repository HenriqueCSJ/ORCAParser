#!/usr/bin/env python3
"""
orca_parser CLI
===============

Parse one or more ORCA output files and write results to JSON, HDF5, CSV,
and/or Markdown.

Notes
-----
  * Directories are searched recursively for ``*.out`` and ``*.log`` files.
  * Auxiliary ORCA helper/ECP files such as ``*_atom83.out`` are skipped
    during recursive discovery and rejected if passed directly.
  * Geometry-dependent final-state exports prefer the final converged block,
    not the first block, for multi-step jobs such as geometry optimizations.
  * Render detail defaults are mode-aware: stand-alone markdown reports show
    full bulky sections by default, while comparison reports keep compact
    ranges by default. Use ``--detail-scope`` to force either behavior.
  * Markdown, CSV, and CLI summaries prefer normalized parse-time views
    (``job_snapshot``, ``job_series``, ``final_snapshot``) so job labels,
    stepwise histories, and final-state values stay aligned.
  * Markdown and CSV exports explicitly distinguish DeltaSCF excited-state
    jobs from ordinary ground-state single-points.
  * `%tddft` / `%cis` excited-state geometry optimizations are reported as
    excited-state optimization jobs, including target-state and root-follow
    metadata (`IRoot`, `IRootMult`, `FOLLOWIROOT`, FIR controls).
  * TDDFT/NTO output preserves ORCA root numbering exactly. When ORCA prints
    roots out of energy order, markdown and CSV add an explicit ``E-rank``
    field instead of silently renumbering the states. TDDFT summaries report
    all CI contributions with weight >= 10% and all NTO pairs with occupancy
    ``n >= 0.10``.
  * GOAT conformer searches are parsed as dedicated GOAT jobs, including the
    final ensemble table, global-minimum xyz, conformer window counts, and
    ensemble thermochemistry.
  * Relaxed surface scans are parsed as scan jobs, not collapsed into a
    single geometry optimization. Scan coordinates, per-step energies, and
    discovered ``relaxscan*.dat`` / ``allxyz`` sidecars are exported.
  * For UseSym jobs, symmetry metadata includes the detected point group,
    reduced/orbital irrep group, irrep occupations, and symmetry-perfected
    geometry when ORCA prints it.

Section aliases
---------------
  all        Everything (default)
  charges    mulliken, loewdin, hirshfeld, mbis, chelpg
  population mulliken, loewdin, mayer
  mos        orbital_energies, qro
  bonds      mayer, loewdin
  nbo        nbo
  dipole     dipole
  solvation  solvation
  tddft      tddft
  geometry   geometry, basis_set
  epr        epr (ZFS, g-tensor, hyperfine/EFG)
  goat       goat (GOAT final ensemble, minimum, Sconf/Gconf)
  opt        geom_opt (optimization cycles, convergence, RMSD)
  scan       surface_scan (relaxed scan coordinates, per-step energies)

  Plus any individual section name: scf, mulliken, mayer, chelpg, ...
  Core sections (metadata, geometry, basis_set, scf) are always included.

Examples
--------
  # Parse a single file (default: JSON + CSV, all sections)
  orca_parser water.out

  # Extract only charge data and molecular orbitals
  orca_parser water.out --sections charges mos

  # Compact gzipped JSON, skip CSV output
  orca_parser water.out --compact --gzip --no-csv

  # Export to HDF5 format only (no JSON, no CSV)
  orca_parser water.out --hdf5 --no-json --no-csv

  # HDF5 with LZF compression instead of default gzip
  orca_parser water.out --hdf5 --h5-compression lzf --no-json --no-csv

  # Generate an AI-readable markdown report alongside JSON
  orca_parser water.out --markdown

  # Force compact/ranged markdown even for a stand-alone file
  orca_parser water.out --markdown --detail-scope compact

  # Parse NBO analysis only
  orca_parser water.out --sections nbo

  # Multiple files with a human-readable summary printed to stdout
  orca_parser water.out ethanol.out benzene.out --summary

  # Multiple files with multi-molecule comparison document
  orca_parser *.out --compare --outdir results/

  # Compare all *.out and *.log files found recursively in a directory
  orca_parser calculations/ --compare --outdir results/

  # Force full per-molecule detail in comparison mode
  orca_parser calculations/ --compare --detail-scope full

  # Mix files and directories
  orca_parser extra.out calculations/ --compare

  # Quiet mode (no progress messages), all output to a custom directory
  orca_parser water.out --outdir /tmp/parsed --quiet

  # Strip null values from JSON for cleaner output
  orca_parser water.out --strip-none

  # Only dipole and population data, with 4-space JSON indentation
  orca_parser water.out --sections dipole population --indent 4

  # Extract EPR data (ZFS, g-tensor, hyperfine coupling)
  orca_parser radical.out --sections epr

  # Parse geometry optimization data (per-cycle energies, convergence, RMSD)
  orca_parser geom_opt.out --sections opt --summary

  # Parse a relaxed surface scan and export its scan table
  orca_parser relaxscan.out --sections scan --markdown --csv

  # Parse a GOAT conformer search and export the final ensemble
  orca_parser conformers.out --sections goat --markdown --csv

  # Limit GOAT markdown tables to conformers within 3 kcal/mol
  orca_parser conformers.out --sections goat --markdown --goat-max-relative-energy-kcal 3

  # Force the full GOAT ensemble even in comparison mode
  orca_parser conformers.out other.out --compare --goat-max-relative-energy-kcal all

  # DeltaSCF excited-state jobs are called out explicitly in markdown/CSV
  orca_parser excited_state.out --markdown --csv

  # TDDFT/CIS excited-state optimization (for example S1)
  orca_parser excited_opt.out --sections tddft opt --markdown --csv

  # Show TDDFT roots with explicit energy ranking in markdown / CSV
  orca_parser vertical_tddft.out --sections tddft --markdown --csv
"""

import argparse
import sys
from pathlib import Path
from typing import Any, List

from orca_parser.final_snapshot import (
    get_final_dipole as _get_final_dipole,
    get_final_orbital_energies as _get_final_orbital_energies,
)
from orca_parser.output.job_state import (
    calculation_type_label as _calculation_type_label,
    electronic_state_label as _electronic_state_label,
    get_basis_set as _get_basis_set,
    get_charge as _get_charge,
    get_excited_state_opt_data as _get_excited_state_opt_data,
    get_job_name as _get_job_name,
    get_method_header_label as _get_method_header_label,
    get_multiplicity as _get_multiplicity,
    symmetry_inline_label as _symmetry_inline_label,
)
from orca_parser.plugin_discovery import (
    bootstrap_plugin_bundles as _bootstrap_plugin_bundles,
    build_plugin_help_section as _build_plugin_help_section,
    get_plugin_option_values as _get_plugin_option_values,
    get_registered_plugin_options as _get_registered_plugin_options,
)
from orca_parser.parser import is_auxiliary_orca_file


_GOAT_CUTOFF_UNSET = object()


def _parse_goat_markdown_cutoff(value: str):
    """Parse a GOAT markdown cutoff argument."""
    text = value.strip().lower()
    if text in {"all", "full", "none"}:
        return None
    try:
        cutoff = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "GOAT cutoff must be a non-negative number or 'all'."
        ) from exc
    if cutoff < 0:
        raise argparse.ArgumentTypeError(
            "GOAT cutoff must be a non-negative number or 'all'."
        )
    return cutoff


def parse_args() -> argparse.Namespace:
    """Build and parse CLI arguments for ``orca_parser``."""
    _bootstrap_plugin_bundles()
    plugin_help = _build_plugin_help_section()
    p = argparse.ArgumentParser(
        prog="orca_parser",
        description=(
            "Parse ORCA quantum chemistry output files into JSON, CSV, "
            "HDF5, and Markdown. Handles ground-state, DeltaSCF excited-state, "
            "TDDFT/CIS excited-state optimization, GOAT conformer-search, "
            "UseSym/symmetry-aware, EPR, relaxed surface-scan, and "
            "geometry-optimization jobs, using normalized final-state and "
            "job-metadata summaries for downstream output. TDDFT output keeps "
            "ORCA root numbering intact and adds explicit energy-rank "
            "annotations when roots are not printed in ascending energy order."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__ + plugin_help,
    )
    p.add_argument("files", nargs="+", metavar="FILE",
                   help="ORCA output file(s) or directory(ies) to parse. "
                        "Directories are searched recursively for *.out and "
                        "*.log files; helper/ECP files such as *_atom83.out "
                        "are skipped.")

    # ── Section selection ────────────────────────────────────────────
    p.add_argument("--sections", nargs="+", metavar="SEC", default=None,
                   help="Sections / aliases to parse (default: all). "
                        "Examples: --sections charges mos dipole, "
                        "--sections epr, --sections goat, --sections opt, --sections scan")

    # ── Output format flags ─────────────────────────────────────────
    p.add_argument("--json",    dest="write_json", action="store_true",  default=True,
                   help="Write JSON output (.json or .json.gz). Default: on")
    p.add_argument("--no-json", dest="write_json", action="store_false",
                   help="Disable JSON output")
    p.add_argument("--csv",     dest="write_csv",  action="store_true",  default=True,
                   help="Write CSV tables. Default: on")
    p.add_argument("--no-csv",  dest="write_csv",  action="store_false",
                   help="Disable CSV output")
    p.add_argument("--hdf5",    dest="write_hdf5",     action="store_true",  default=False,
                   help="Write HDF5 output (.h5)")
    p.add_argument("--no-hdf5", dest="write_hdf5",     action="store_false",
                   help="Disable HDF5 output")
    p.add_argument("--markdown", dest="write_markdown", action="store_true",  default=False,
                   help="Write compact AI-readable markdown report (.md) "
                        "with symmetry, DeltaSCF / TDDFT excited-state "
                        "labeling, TDDFT root/energy-rank summaries, "
                        "significant CI (>=10%%) / NTO (n>=0.10) tables, "
                        "root-follow summaries, GOAT ensemble summaries, "
                        "and surface-scan summaries")
    p.add_argument("--no-markdown", dest="write_markdown", action="store_false",
                   help="Disable markdown output")
    p.add_argument("--compare",  dest="write_compare",  action="store_true",  default=False,
                   help="Write multi-molecule comparison document (comparison.md). "
                        "Implies --markdown. Accepts directories - searches "
                        "recursively for *.out and *.log files, skipping "
                        "auxiliary *_atomNN helper outputs.")
    p.add_argument(
        "--detail-scope",
        choices=("auto", "full", "compact"),
        default="auto",
        help=(
            "Control how much bulky markdown detail to show. "
            "'auto' keeps the mode-aware defaults: standalone markdown is full, "
            "comparison markdown is compact. Use 'full' to force everything, "
            "or 'compact' to force ranged summaries."
        ),
    )
    p.add_argument(
        "--goat-max-relative-energy-kcal",
        type=_parse_goat_markdown_cutoff,
        default=_GOAT_CUTOFF_UNSET,
        metavar="N|all",
        help=(
            "Limit GOAT ensemble tables in markdown to conformers with "
            "dE <= N kcal/mol, or use 'all' to print the full ensemble. "
            "By default, standalone markdown shows all conformers while "
            "comparison markdown shows dE <= 10 kcal/mol."
        ),
    )

    # ── JSON options ─────────────────────────────────────────────────
    p.add_argument("--indent", type=int, default=2, metavar="N",
                   help="JSON indentation (default: 2; 0 = compact)")
    p.add_argument("--compact", action="store_true",
                   help="Compact JSON: no indentation + strip None values")
    p.add_argument("--gzip", dest="compress", action="store_true",
                   help="Gzip-compress JSON output (.json.gz)")
    p.add_argument("--strip-none", action="store_true",
                   help="Remove null-valued keys from JSON")

    # ── HDF5 options ─────────────────────────────────────────────────
    p.add_argument("--h5-compression", default="gzip", metavar="FILTER",
                   help="HDF5 compression filter: gzip, lzf, or none (default: gzip)")
    p.add_argument("--h5-level", type=int, default=4, metavar="N",
                   help="gzip compression level 1-9 (default: 4)")

    # ── Misc ─────────────────────────────────────────────────────────
    p.add_argument("--outdir", default=None, metavar="DIR",
                   help="Output directory (default: same dir as input file; "
                        "for directory/compare mode, the chosen output root)")
    p.add_argument("--summary", action="store_true",
                   help="Print a human-readable summary to stdout, including "
                        "normalized job labels, basic symmetry/spin "
                        "diagnostics, final frontier orbitals/dipole when "
                        "available, TDDFT root-order diagnostics when "
                        "available, and scan info when applicable")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress progress messages")

    # Plugin bundles can advertise their own options declaratively. The CLI
    # discovers them here so adding a module can extend help/arguments without
    # editing this central parser.
    for option in _get_registered_plugin_options():
        p.add_argument(*option.flags, **option.argparse_kwargs())

    return p.parse_args()


_ORCA_EXTENSIONS = {".out", ".log"}


def _resolve_files(paths: List[str]) -> List[Path]:
    """Expand directories recursively into *.out and *.log files.

    Regular files are passed through as-is. Directories are searched
    recursively for files matching ORCA output extensions, excluding
    auxiliary ORCA helper files such as ``*_atom83.out``.
    Results are sorted by path for deterministic ordering.
    """
    resolved: List[Path] = []
    for p in paths:
        fp = Path(p)
        if fp.is_dir():
            for ext in sorted(_ORCA_EXTENSIONS):
                for candidate in sorted(fp.rglob(f"*{ext}")):
                    if is_auxiliary_orca_file(candidate):
                        continue
                    resolved.append(candidate)
        else:
            resolved.append(fp)
    return resolved


def _print_summary(data: dict, path: Path) -> None:
    print(f"\n{'='*60}")
    print(f"  File : {path.name}")
    print(f"{'='*60}")

    meta = data.get("metadata", {})
    ctx  = data.get("context",  {})
    state_label = _electronic_state_label(data, ground_state_label="Ground-state")
    symmetry_label = _symmetry_inline_label(data)
    print(f"  ORCA version : {meta.get('orca_version', 'N/A')}")
    print(f"  Job          : {_get_job_name(data) or 'N/A'}")
    print(f"  Calc type    : {_calculation_type_label(data) or 'N/A'}")
    print(f"  State        : {state_label}")
    print(f"  Reference    : {ctx.get('reference_type', ctx.get('hf_type', 'N/A'))}")
    print(f"  HF type      : {ctx.get('hf_type', 'N/A')}")
    print(f"  Method       : {_get_method_header_label(data)}")
    print(f"  Basis        : {_get_basis_set(data) or 'N/A'}")
    print(f"  Charge/Mult  : {_get_charge(data)} / {_get_multiplicity(data)}")
    print(f"  Symmetry     : {symmetry_label or 'none'}")
    excopt = _get_excited_state_opt_data(data)
    if excopt:
        label = excopt.get("target_state_label") or excopt.get("target_root")
        if label:
            print(f"  Excited root : {label}")
        if excopt.get("input_block"):
            print(f"  ES method    : %{excopt['input_block']}")
        if "followiroot" in excopt:
            follow = "yes" if excopt.get("followiroot") else "no"
            print(f"  Follow IRoot : {follow}")

    scf = data.get("scf", {})
    if scf:
        E = scf.get("final_single_point_energy_Eh") or scf.get("total_energy_Eh")
        print(f"\n  Energy (Eh)  : {E}")
        if "s_squared" in scf:
            print(f"  <S^2>        : {scf['s_squared']:.6f}  "
                  f"(ideal {scf.get('s_squared_ideal', 'N/A')})")

    oe = _get_final_orbital_energies(data)
    if "HOMO_LUMO_gap_eV" in oe:
        print(f"  HOMO-LUMO (eV): {oe['HOMO_LUMO_gap_eV']:.4f}")
    for spin in ("alpha", "beta"):
        key = f"{spin}_HOMO_LUMO_gap_eV"
        if key in oe:
            print(f"  {spin.capitalize()} gap (eV): {oe[key]:.4f}")

    dip = _get_final_dipole(data)
    if dip:
        mag = dip.get('magnitude_Debye')
        print(f"  Dipole (D)   : {mag:.4f}" if mag is not None else "  Dipole (D)   : N/A")

    qro = data.get("qro", {})
    if qro:
        print(f"  QRO          : DOMO={qro.get('n_domo')}  "
              f"SOMO={qro.get('n_somo')}  VMO={qro.get('n_vmo')}")

    gopt = data.get("geom_opt", {})
    if gopt:
        n = gopt.get("n_cycles", 0)
        conv = "YES" if gopt.get("converged") else "NO"
        print(f"  Opt cycles   : {n}  (converged: {conv})")
        if gopt.get("final_energy_Eh") is not None:
            print(f"  Opt energy   : {gopt['final_energy_Eh']:.10f} Eh")
        if gopt.get("rmsd_initial_to_final_ang") is not None:
            print(f"  RMSD i->f    : {gopt['rmsd_initial_to_final_ang']:.6f} A")

    scan = data.get("surface_scan", {})
    if scan:
        mode = scan.get("mode", "N/A")
        n_params = scan.get("n_parameters", 0)
        n_steps = len(scan.get("steps", []))
        print(f"  Scan mode    : {mode}  ({n_params} coordinate(s), {n_steps} step(s))")
        if scan.get("energy_span_kcal_mol") is not None:
            print(f"  Scan span    : {scan['energy_span_kcal_mol']:.4f} kcal/mol")

    goat = data.get("goat", {})
    if goat:
        n_confs = goat.get("n_conformers", 0)
        print(f"  GOAT confs   : {n_confs}")
        if goat.get("conformers_below_energy_window") is not None:
            window = goat.get("conformer_energy_window_kcal_mol")
            print(
                f"  Below {window:.2f} kcal/mol: "
                f"{goat['conformers_below_energy_window']}"
            )
        if goat.get("lowest_energy_conformer_Eh") is not None:
            print(
                "  GOAT minimum : "
                f"{goat['lowest_energy_conformer_Eh']:.6f} Eh"
            )
        if goat.get("sconf_cal_molK") is not None:
            print(f"  Sconf        : {goat['sconf_cal_molK']:.2f} cal/(molK)")
        if goat.get("gconf_kcal_mol") is not None:
            print(f"  Gconf        : {goat['gconf_kcal_mol']:.2f} kcal/mol")

    for name, key in [("Mulliken", "mulliken"), ("Hirshfeld", "hirshfeld"),
                      ("MBIS", "mbis"), ("CHELPG", "chelpg")]:
        section = data.get(key, {})
        atoms = section.get("atoms") or section.get("atomic_charges")
        if atoms:
            charges = [a.get("charge") for a in atoms if "charge" in a]
            if charges:
                print(f"  {name:10}: {', '.join(f'{c:.3f}' for c in charges)}")
    print()


def _resolve_h5_compression(filter_name: str) -> str | None:
    """Normalize the HDF5 compression argument for downstream writers."""
    return None if filter_name.lower() == "none" else filter_name


def _emit_write_status(path: Path, quiet: bool, *, label: str | None = None, end: str = "  ") -> None:
    """Print a compact one-line status update for a written artifact."""
    if quiet:
        return
    size_kb = path.stat().st_size / 1024
    display_name = label or path.name
    print(f"-> {display_name} ({size_kb:.1f} KB)", end=end, flush=True)


def _parse_single_file(
    fp: Path,
    *,
    sections: list[str] | None,
    plugin_options: dict[str, Any],
):
    """Parse one ORCA output file and return the parser plus parsed data."""
    from orca_parser import ORCAParser

    parser = ORCAParser(fp, plugin_options=plugin_options)
    data = parser.parse(sections=sections)
    return parser, data


def _write_requested_outputs(
    parser,
    data: dict[str, Any],
    fp: Path,
    args: argparse.Namespace,
    *,
    goat_cutoff,
    h5_compression: str | None,
) -> None:
    """Write all requested per-file outputs for one parsed job."""
    outdir = Path(args.outdir) if args.outdir else fp.parent
    stem = fp.stem

    if args.write_json:
        written = parser.to_json(
            outdir / f"{stem}.json",
            indent=0 if args.compact else args.indent,
            strip_none=args.strip_none or args.compact,
            compress=args.compress,
        )
        _emit_write_status(written, args.quiet)

    if args.write_hdf5:
        written = parser.to_hdf5(
            outdir / f"{stem}.h5",
            compression=h5_compression,
            compression_opts=args.h5_level,
        )
        _emit_write_status(written, args.quiet)

    if args.write_csv:
        csv_dir = outdir / f"{stem}_csv"
        parser.to_csv(csv_dir)
        if not args.quiet:
            table_count = sum(1 for _ in csv_dir.glob("*.csv"))
            print(f"-> {csv_dir.name}/ ({table_count} tables)", end="  ", flush=True)

    if args.write_markdown:
        markdown_kwargs = {"detail_scope": args.detail_scope}
        if goat_cutoff is not _GOAT_CUTOFF_UNSET:
            markdown_kwargs["goat_max_relative_energy_kcal_mol"] = goat_cutoff
        written = parser.to_markdown(
            outdir / f"{stem}.md",
            **markdown_kwargs,
        )
        _emit_write_status(written, args.quiet, end="")

    if not args.quiet:
        print()

    if args.summary:
        _print_summary(data, fp)


def _write_comparison_output(
    parsers: list[Any],
    resolved_files: list[Path],
    args: argparse.Namespace,
    *,
    goat_cutoff,
) -> None:
    """Write the multi-job comparison document when requested."""
    if not args.write_compare or len(parsers) <= 1:
        return

    from orca_parser import ORCAParser

    comp_outdir = Path(args.outdir) if args.outdir else resolved_files[0].parent
    compare_kwargs = {"detail_scope": args.detail_scope}
    if goat_cutoff is not _GOAT_CUTOFF_UNSET:
        compare_kwargs["goat_max_relative_energy_kcal_mol"] = goat_cutoff
    written = ORCAParser.compare(
        parsers,
        comp_outdir / "comparison.md",
        **compare_kwargs,
    )
    _emit_write_status(written, args.quiet, label=f"comparison: {written.name}", end="\n")


def main() -> None:
    """Run the ``orca_parser`` command-line workflow."""
    args = parse_args()
    plugin_options = _get_plugin_option_values(args)

    h5_compression = _resolve_h5_compression(args.h5_compression)
    goat_cutoff = args.goat_max_relative_energy_kcal
    if args.write_compare:
        args.write_markdown = True

    all_parsers = []
    resolved_files = _resolve_files(args.files)

    if not resolved_files:
        print("[ERROR] No ORCA output files found.", file=sys.stderr)
        sys.exit(1)

    for fp in resolved_files:
        if not fp.exists():
            print(f"[ERROR] File not found: {fp}", file=sys.stderr)
            continue

        if not args.quiet:
            sec_label = " ".join(args.sections) if args.sections else "all"
            print(f"Parsing {fp.name} [{sec_label}] ...", end=" ", flush=True)

        try:
            parser, data = _parse_single_file(
                fp,
                sections=args.sections,
                plugin_options=plugin_options,
            )
        except Exception as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            continue

        all_parsers.append(parser)
        _write_requested_outputs(
            parser,
            data,
            fp,
            args,
            goat_cutoff=goat_cutoff,
            h5_compression=h5_compression,
        )

    _write_comparison_output(
        all_parsers,
        resolved_files,
        args,
        goat_cutoff=goat_cutoff,
    )


if __name__ == "__main__":
    main()
