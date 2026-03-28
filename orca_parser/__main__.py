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
  * Markdown and CSV exports explicitly distinguish DeltaSCF excited-state
    jobs from ordinary ground-state single-points.
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
  geometry   geometry, basis_set
  epr        epr (ZFS, g-tensor, hyperfine/EFG)
  opt        geom_opt (optimization cycles, convergence, RMSD)

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

  # Parse NBO analysis only
  orca_parser water.out --sections nbo

  # Multiple files with a human-readable summary printed to stdout
  orca_parser water.out ethanol.out benzene.out --summary

  # Multiple files with multi-molecule comparison document
  orca_parser *.out --compare --outdir results/

  # Compare all *.out and *.log files found recursively in a directory
  orca_parser calculations/ --compare --outdir results/

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

  # DeltaSCF excited-state jobs are called out explicitly in markdown/CSV
  orca_parser excited_state.out --markdown --csv
"""

import argparse
import sys
from pathlib import Path
from typing import List

from orca_parser.parser import is_auxiliary_orca_file


def parse_args():
    p = argparse.ArgumentParser(
        prog="orca_parser",
        description=(
            "Parse ORCA quantum chemistry output files into JSON, CSV, "
            "HDF5, and Markdown. Handles ground-state, DeltaSCF excited-state, "
            "UseSym/symmetry-aware, EPR, TDDFT, and geometry-optimization jobs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
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
                        "--sections epr, --sections opt")

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
                        "with symmetry and DeltaSCF state labeling")
    p.add_argument("--no-markdown", dest="write_markdown", action="store_false",
                   help="Disable markdown output")
    p.add_argument("--compare",  dest="write_compare",  action="store_true",  default=False,
                   help="Write multi-molecule comparison document (comparison.md). "
                        "Implies --markdown. Accepts directories — searches "
                        "recursively for *.out and *.log files, skipping "
                        "auxiliary *_atomNN helper outputs.")

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
                        "calculation type and basic symmetry/spin diagnostics")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress progress messages")

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
    print(f"  ORCA version : {meta.get('orca_version', 'N/A')}")
    print(f"  Job          : {meta.get('job_name', 'N/A')}")
    print(f"  Calc type    : {meta.get('calculation_type', 'N/A')}")
    print(f"  HF type      : {ctx.get('hf_type', 'N/A')}")
    print(f"  Functional   : {meta.get('functional', 'N/A')}")
    print(f"  Basis        : {meta.get('basis_set', 'N/A')}")
    print(f"  Charge/Mult  : {meta.get('charge', 'N/A')} / {meta.get('multiplicity', 'N/A')}")
    print(f"  Symmetry     : {'yes' if ctx.get('has_symmetry') else 'no'}")

    scf = data.get("scf", {})
    if scf:
        E = scf.get("final_single_point_energy_Eh") or scf.get("total_energy_Eh")
        print(f"\n  Energy (Eh)  : {E}")
        if "s_squared" in scf:
            print(f"  <S²>         : {scf['s_squared']:.6f}  "
                  f"(ideal {scf.get('s_squared_ideal', 'N/A')})")

    oe = data.get("orbital_energies", {})
    if "HOMO_LUMO_gap_eV" in oe:
        print(f"  HOMO-LUMO (eV): {oe['HOMO_LUMO_gap_eV']:.4f}")
    for spin in ("alpha", "beta"):
        key = f"{spin}_HOMO_LUMO_gap_eV"
        if key in oe:
            print(f"  {spin.capitalize()} gap (eV): {oe[key]:.4f}")

    dip = data.get("dipole", {})
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
            print(f"  RMSD i→f     : {gopt['rmsd_initial_to_final_ang']:.6f} Å")

    for name, key in [("Mulliken", "mulliken"), ("Hirshfeld", "hirshfeld"),
                      ("MBIS", "mbis"), ("CHELPG", "chelpg")]:
        section = data.get(key, {})
        atoms = section.get("atoms") or section.get("atomic_charges")
        if atoms:
            charges = [a.get("charge") for a in atoms if "charge" in a]
            if charges:
                print(f"  {name:10}: {', '.join(f'{c:.3f}' for c in charges)}")
    print()


def main():
    args = parse_args()
    from orca_parser import ORCAParser

    h5_compression = None if args.h5_compression.lower() == "none" else args.h5_compression
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
            parser = ORCAParser(fp)
            data = parser.parse(sections=args.sections)
            all_parsers.append(parser)
        except Exception as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            continue

        outdir = Path(args.outdir) if args.outdir else fp.parent

        # ── JSON ────────────────────────────────────────────────────
        if args.write_json:
            indent  = 0 if args.compact else args.indent
            strip   = args.strip_none or args.compact
            gzip_on = args.compress
            json_path = outdir / (fp.stem + ".json")
            written = parser.to_json(json_path, indent=indent,
                                     strip_none=strip, compress=gzip_on)
            if not args.quiet:
                size = written.stat().st_size
                print(f"→ {written.name} ({size/1024:.1f} KB)", end="  ", flush=True)

        # ── HDF5 ────────────────────────────────────────────────────
        if args.write_hdf5:
            h5_path = outdir / (fp.stem + ".h5")
            written = parser.to_hdf5(h5_path,
                                     compression=h5_compression,
                                     compression_opts=args.h5_level)
            if not args.quiet:
                size = written.stat().st_size
                print(f"→ {written.name} ({size/1024:.1f} KB)", end="  ", flush=True)

        # ── CSV ─────────────────────────────────────────────────────
        if args.write_csv:
            csv_dir = outdir / (fp.stem + "_csv")
            files = parser.to_csv(csv_dir)
            if not args.quiet:
                print(f"→ {csv_dir.name}/ ({len(files)} tables)", end="  ", flush=True)

        # ── Markdown (per file) ──────────────────────────────────────
        if args.write_markdown:
            md_path = outdir / (fp.stem + ".md")
            written = parser.to_markdown(md_path)
            if not args.quiet:
                size = written.stat().st_size
                print(f"→ {written.name} ({size/1024:.1f} KB)", end="", flush=True)

        if not args.quiet:
            print()

        if args.summary:
            _print_summary(data, fp)

    # ── Comparison document (all files) ─────────────────────────────
    if args.write_compare and len(all_parsers) > 1:
        comp_outdir = Path(args.outdir) if args.outdir else resolved_files[0].parent
        comp_path   = comp_outdir / "comparison.md"
        written     = ORCAParser.compare(all_parsers, comp_path)
        if not args.quiet:
            size = written.stat().st_size
            print(f"\n→ comparison: {written.name} ({size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
