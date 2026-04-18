# ORCA Parser

A modular Python parser for [ORCA](https://orcaforum.kofo.mpg.de/) quantum chemistry output files. Extracts computed properties into structured formats (JSON, CSV, HDF5, Markdown) for downstream analysis, machine-learning pipelines, and automated report generation.

## Features

- **20 parser modules** covering SCF energies, orbital data, implicit-solvation settings (CPCM/SMD/ALPB/COSMO-RS), TDDFT/CIS excited states and spectra, excited-state geometry optimizations, GOAT conformer searches, 6 population analysis schemes (Mulliken, Loewdin, Mayer, Hirshfeld, MBIS, CHELPG), dipole moments, NBO analysis, EPR properties, geometry optimizations, and relaxed surface scans
- **4 output formats**: JSON (with optional gzip), CSV (one table per section), HDF5, and AI-readable Markdown
- Handles both **RHF/RKS** and **UHF/UKS** calculations with spin-resolved data
- **Geometry optimization** tracking: per-cycle energies, convergence criteria, trust radii, Kabsch-aligned RMSD (initial/final, step-to-step, mass-weighted)
- **GOAT conformer-search** support: final ensemble tables, relative populations, energy windows, global-minimum/final-ensemble xyz files, and ensemble thermochemistry (`Sconf`, `Gconf`)
- **Relaxed surface scan** support: scan coordinate definitions (`B`, `A`, `D`), simultaneous vs. nested scans, per-step energies, optimized `*.xyz` files, and detected `relaxscan*.dat` / `allxyz` sidecars
- **DeltaSCF** metadata extraction: ALPHACONF/BETACONF, IONIZEALPHA/IONIZEBETA, occupation vectors, MOM/IMOM settings, with explicit excited-state labeling in Markdown and CSV outputs
- **TDDFT/CIS excited-state optimization** support: `%tddft` / `%cis` targets (`IRoot`, `IRootMult`), `FOLLOWIROOT` / FIR controls, analytic excited-state gradient detection, per-cycle target-state history, and explicit `Sx` / `Tx` labeling in Markdown and CSV outputs
- **TDDFT/NTO root-order diagnostics**: preserves ORCA's printed root numbers, adds explicit energy ranks when roots are not printed in ascending energy order, records NTO-to-root energy consistency checks, and keeps parser-level reporting thresholds for significant CI contributions (`weight >= 10%`) and NTO pairs (`n >= 0.10`)
- **EPR/EPR properties**: Zero-Field Splitting, g-tensor (with atomic breakdown), hyperfine coupling, EFG, quadrupole
- **UseSym / point-group symmetry** support: detected point group, reduced/orbital irrep group, irrep occupations, and symmetry-perfected geometries when ORCA prints them
- **Normalized parse-time views** for downstream development: `job_snapshot` (job identity/state/method labels), `job_series` (GOAT/optimization/scan histories), and `final_snapshot` (authoritative final-step geometry-dependent properties)
- **Case-insensitive** keyword and section matching (ORCA is case-insensitive, so is this parser)
- Selective section extraction via aliases (`charges`, `mos`, `bonds`, `opt`, `epr`, etc.)
- Batch processing of multiple files with **multi-molecule comparison** reports
- **Recursive directory search** for `*.out` and `*.log` files, excluding auxiliary ORCA helper/ECP files such as `*_atom83.out`
- **ORCA output validation** — rejects non-ORCA files (requires program banner + normal termination)
- **Quasi-Restricted Orbital (QRO)** parsing for UHF calculations
- Comprehensive **Natural Bond Orbital (NBO)** analysis extraction

## Requirements

- Python >= 3.10
- Optional: `numpy` (for Kabsch-aligned RMSD in geometry optimizations; plain RMSD fallback without it)
- Optional: `h5py` (for HDF5 output; requires `numpy`)

## Installation

```bash
pip install .
```

Or for development:

```bash
pip install -e .
```

## Quick Start

### Command Line

```bash
# Parse a file — produces JSON + CSV by default
orca_parser water.out

# Extract only charges and molecular orbitals
orca_parser water.out --sections charges mos

# Compact gzipped JSON, no CSV
orca_parser water.out --compact --gzip --no-csv

# HDF5 output only
orca_parser water.out --hdf5 --no-json --no-csv

# Markdown report for AI-assisted paper writing
orca_parser water.out --markdown

# DeltaSCF excited-state job: markdown/CSV call this out explicitly
orca_parser excited_state.out --markdown --csv

# TDDFT/CIS excited-state optimization to S1 with root-follow metadata
orca_parser excited_opt.out --sections tddft opt --markdown --csv

# Vertical TDDFT with explicit ORCA root numbers and energy ranks
orca_parser vertical_tddft.out --sections tddft --markdown --csv

# Compare multiple molecules
orca_parser water.out ethanol.out benzene.out --compare --outdir results/

# Recursively find all *.out and *.log files in a directory and compare
orca_parser calculations/ --compare --outdir results/

# Human-readable summary to stdout
orca_parser water.out --summary

# Parse geometry optimization (per-cycle energies, convergence, RMSD)
orca_parser geom_opt.out --sections opt --summary

# Parse a relaxed surface scan and export its scan table
orca_parser relaxscan.out --sections scan --markdown --csv

# Parse a GOAT conformer search and export the final ensemble
orca_parser conformers.out --sections goat --markdown --csv

# Limit GOAT markdown to conformers within 3 kcal/mol
orca_parser conformers.out --sections goat --markdown --goat-max-relative-energy-kcal 3

# Extract EPR data (ZFS, g-tensor, hyperfine coupling)
orca_parser radical.out --sections epr
```

### Python API

```python
from orca_parser import ORCAParser

parser = ORCAParser("water.out")
data = parser.parse()

# Access parsed data
energy = data["scf"]["final_single_point_energy_Eh"]
charges = data["mulliken"]["atomic_charges"]

# Export
parser.to_json("water.json")
parser.to_csv("water_csv/")
parser.to_markdown("water.md")

# Selective parsing
data = parser.parse(sections=["charges", "dipole"])

# Geometry optimization data
opt = data.get("geom_opt", {})
if opt:
    print(f"Converged: {opt['converged']} in {opt['n_cycles']} cycles")
    print(f"Final energy: {opt['final_energy_Eh']} Eh")
    print(f"RMSD initial→final: {opt['rmsd_initial_to_final_ang']} Å")

# Relaxed surface scan data
scan = data.get("surface_scan", {})
if scan:
    print(f"Scan mode: {scan['mode']}")
    print(f"Scan coordinates: {len(scan['parameters'])}")
    print(f"Scan steps: {len(scan['steps'])}")

# Normalized parse-time views for downstream tooling
job = data.get("job_snapshot", {})
final = data.get("final_snapshot", {})
series = data.get("job_series", {})
print(job.get("calculation_family"))
print(final.get("orbital_energies", {}).get("HOMO_energy_eV"))
print(len(series.get("goat", {}).get("ensemble", [])))

# Multi-molecule comparison
p1, p2 = ORCAParser("mol1.out"), ORCAParser("mol2.out")
p1.parse(); p2.parse()
ORCAParser.compare([p1, p2], "comparison.md")
```

## Extracted Data

### Core (always included)

| Module | Data |
|--------|------|
| **metadata** | ORCA version, job name, functional, basis set, charge, multiplicity, calculation type, DeltaSCF detection (ALPHACONF/BETACONF, IONIZEALPHA/IONIZEBETA, occupation vectors, MOM/IMOM), excited-state optimization metadata (`IRoot`, `IRootMult`, `FOLLOWIROOT`, FIR controls), symmetry metadata |
| **geometry** | Cartesian coordinates (Å and a.u.), internal coordinates, symmetry-perfected geometry when printed by ORCA |
| **basis_set** | Basis set groups, atom mappings, shell count |
| **scf** | Total energy, energy components, DFT data, convergence, spin expectation |

### Optional Sections

| Module | Data |
|--------|------|
| **orbital_energies** | HOMO/LUMO, orbital energies (RHF/UHF), irrep labels, occupied orbitals per irrep |
| **qro** | Quasi-Restricted Orbitals — DOMO/SOMO/VMO classification (UHF only) |
| **solvation** | Implicit-solvation detection, model/solvent summary, `%cpcm`/`%cosmors` inputs, CPCM/SMD, ALPB, and COSMO-RS output blocks |
| **tddft** | TDDFT/CIS input settings, excited-state configurations, NTOs, absorption/CD spectra, CIS/TDDFT total-energy summary, excited-state optimization target/root-follow metadata, and `energy_rank` annotations that keep ORCA root numbering separate from energy ordering |
| **goat** | GOAT global-minimum status, final ensemble, conformer populations, energy windows, and ensemble thermochemistry |
| **surface_scan** | Relaxed scan definitions, scan mode, per-step coordinates, actual/SCF surface energies, optimized XYZ files, detected sidecar trajectory files |
| **mulliken** | Mulliken charges, spin populations, reduced orbital charges |
| **loewdin** | Loewdin charges, spin populations, reduced orbital charges |
| **mayer** | Mayer bond orders, atomic valence data (NA, ZA, QA, VA, BVA, FA) |
| **hirshfeld** | Hirshfeld charges and spin populations |
| **mbis** | MBIS charges, populations, valence shell data |
| **chelpg** | CHELPG electrostatic potential charges |
| **dipole** | Dipole moment (electronic, nuclear, total), rotational constants |
| **nbo** | NAO, NPA, Wiberg indices, Lewis structure, E(2) perturbation, NLMO |
| **epr** | Zero-Field Splitting, g-tensor (with atom analysis), hyperfine coupling, EFG, quadrupole |
| **geom_opt** | Per-cycle energies, convergence criteria (5 items), trust radii, geometries, internal coord extrema, RMSD (to initial, to previous, mass-weighted), optimization settings/tolerances, OPT/LooseOPT/TightOPT detection |

### Section Aliases

Use aliases on the CLI to select groups of related sections:

| Alias | Expands to |
|-------|------------|
| `all` | All 20 modules (default) |
| `charges` | mulliken, loewdin, hirshfeld, mbis, chelpg |
| `population` | mulliken, loewdin, mayer |
| `mos` | orbital_energies, qro |
| `bonds` | mayer, loewdin |
| `nbo` | nbo |
| `dipole` | dipole |
| `solvation` | solvation |
| `tddft` | tddft |
| `geometry` | geometry, basis_set |
| `epr` | epr (ZFS, g-tensor, hyperfine/EFG) |
| `goat` | goat (GOAT final ensemble, minimum, Sconf/Gconf) |
| `opt` | geom_opt (optimization cycles, convergence, RMSD) |
| `scan` | surface_scan (relaxed scan coordinates, per-step energies, XYZ/sidecar files) |

## Output Formats

### JSON

Default format. Supports indentation control, gzip compression, and null-value stripping.

```bash
orca_parser water.out                          # Pretty JSON (indent=2)
orca_parser water.out --compact                # Compact, no nulls
orca_parser water.out --gzip                   # Compressed .json.gz
orca_parser water.out --indent 4 --strip-none  # Custom indent, no nulls
```

### CSV

One CSV file per data section, written to a `{name}_csv/` directory. Enabled by default.

New metadata-oriented CSV tables include:

- `*_metadata.csv`
- `*_symmetry.csv`
- `*_symmetry_irreps.csv`
- `*_geometry_symmetry.csv` when a symmetry-perfected geometry is available
- `*_deltascf.csv`
- `*_deltascf_occupations.csv`
- `*_excited_state_optimization.csv`
- `*_excited_state_optimization_cycles.csv`
- `*_goat_summary.csv`
- `*_goat_ensemble.csv`
- `*_surface_scan_summary.csv`
- `*_surface_scan_parameters.csv`
- `*_surface_scan.csv`

```bash
orca_parser water.out             # Creates water_csv/ with tables
orca_parser water.out --no-csv    # Disable CSV output
```

### HDF5

Hierarchical binary format with optional compression. Requires `h5py`.

```bash
orca_parser water.out --hdf5 --no-json --no-csv
orca_parser water.out --hdf5 --h5-compression lzf
orca_parser water.out --hdf5 --h5-level 9          # Max gzip compression
```

### Markdown

AI-readable reports optimized for LLM-assisted paper writing. Includes publication-ready tables, spin diagnostics, frontier orbital analysis, explicit DeltaSCF excited-state labeling, TDDFT/CIS excited-state optimization sections with root-history tables, dedicated symmetry sections for UseSym jobs, GOAT ensemble tables, and relaxed surface-scan summaries with per-step tables.

For TDDFT/NTO jobs, markdown keeps ORCA's printed `STATE n` root labels and adds an `E-rank` column when the roots are not ordered by excitation energy. This avoids silently renumbering roots while still making the lowest-energy states obvious in the report.

TDDFT markdown and CSV reporting use fixed parser-level thresholds so the same chemistry is shown consistently across outputs:
- CI / MO-basis excitation contributions: report all contributions with `weight >= 10%`
- NTO pairs: report all pairs with `n >= 0.10`
- Oscillator strengths: always keep the printed value, including dark states with `f ~= 0`
- Recommended lowest-root view: keep at least the lowest 5 roots available via `summary["recommended_lowest_roots"]`

```bash
orca_parser water.out --markdown                   # Per-file report
orca_parser *.out --compare --outdir results/      # Multi-molecule comparison
orca_parser water.out --markdown --detail-scope compact
orca_parser *.out --compare --detail-scope full
orca_parser conformers.out --markdown --goat-max-relative-energy-kcal 3
orca_parser conformers.out other.out --compare --goat-max-relative-energy-kcal all
```

Markdown now uses a shared render-policy layer:

- standalone markdown defaults to **full** detail for bulky sections such as GOAT ensembles, orbital windows, CMO/NBO character tables, and optimization histories
- comparison markdown defaults to **compact** ranges for those same sections
- `--detail-scope compact` forces compact rendering even for a single file
- `--detail-scope full` forces full rendering even in compare mode

Today this policy mainly affects markdown/report truncation. CSV and JSON remain full-fidelity exports. The same shared policy object is intended to become the universal range-control seam for future bulky sections.

Standalone markdown prints the full GOAT ensemble by default. Comparison markdown truncates GOAT tables at `dE <= 10 kcal/mol` unless you override it with `--goat-max-relative-energy-kcal`.

TDDFT CSV exports follow the same policy: `state` remains the ORCA root number, while `energy_rank` records the sorted-by-energy position. NTO CSV tables also expose energy-consistency fields (`energy_match_consistent`, `energy_matched_state`, `energy_matched_rank`, `energy_matched_delta_eV`) so downstream workflows can validate root assignments without inventing a new numbering scheme.

## CLI Reference

```
orca_parser [options] FILE [FILE ...]

Positional:
  FILE                    One or more ORCA output files or directories.
                          Directories are searched recursively for *.out and
                          *.log files. Auxiliary helper/ECP files such as
                          *_atom83.out are skipped.

Section Selection:
  --sections SEC [SEC ...]  Sections/aliases to parse (default: all)
                           Common aliases: charges, mos, dipole, solvation,
                           tddft, goat, opt, scan

Output Formats:
  --json / --no-json      JSON output (default: on)
  --csv / --no-csv        CSV output (default: on)
  --hdf5 / --no-hdf5      HDF5 output (default: off)
  --markdown / --no-markdown  Markdown report (default: off)
  --detail-scope auto|full|compact
                          Control bulky markdown sections. 'auto' keeps
                          stand-alone reports full and comparison reports
                          compact.
  --goat-max-relative-energy-kcal N|all
                          Limit GOAT markdown tables by relative energy
  --compare               Multi-molecule comparison (implies --markdown)
                          and supports directory inputs; defaults to
                          dE <= 10 kcal/mol for GOAT tables

JSON Options:
  --indent N              Indentation level (default: 2; 0 = compact)
  --compact               No indent + strip nulls
  --gzip                  Gzip-compress JSON output
  --strip-none            Remove null-valued keys

HDF5 Options:
  --h5-compression FILTER  gzip, lzf, or none (default: gzip)
  --h5-level N            Compression level 1-9 (default: 4)

Misc:
  --outdir DIR            Output directory (default: same as input)
  --summary               Print normalized job labels plus final-step summary
  --quiet                 Suppress progress messages
```

## Development Notes

For new output work, prefer the normalized parse-time views over the raw module payloads:

- `data["job_snapshot"]`: canonical job ID, job family, display labels, symmetry summary, and excited-state/DeltaSCF metadata
- `data["final_snapshot"]`: authoritative final-step geometry, frontier orbitals, dipole, charges, and bond orders
- `data["job_series"]`: authoritative GOAT ensembles, geometry-optimization cycles, relaxed scan steps, and excited-state optimization cycle histories

These layers exist specifically to prevent repeated "first block vs last block" bugs and to keep markdown, CSV, and CLI summaries aligned.

## Important ORCA-specific behavior

- Files such as `BiC_atom83.out` or `SbC_atom51.out` are treated as auxiliary ORCA atom/ECP helper outputs, not real calculation outputs.
- Recursive directory parsing skips those helper files automatically.
- Passing one directly to the parser raises a validation error instead of silently treating it as a calculation.
- DeltaSCF jobs are reported as excited-state SCF calculations, not ordinary single-point calculations.
- `%tddft` / `%cis` geometry optimizations with analytic excited-state gradients are reported as excited-state optimizations, not ordinary ground-state geometry optimizations.
- For excited-state optimizations the parser exports the target state (`S1`, `T2`, etc.), `IRoot`, `IRootMult`, `FOLLOWIROOT`, FIR controls, and the per-step root/state-of-interest history when ORCA prints it.
- For multistep optimizations the final geometry-dependent properties come from the final converged block, not the first SCF/geometry block in the file.
- UseSym jobs report the detected point group and the reduced/orbital irrep group separately when ORCA prints both.
- GOAT jobs are reported separately from ordinary single points and geometry optimizations, with the final ensemble and thermochemistry exported explicitly.
- Relaxed surface scans are reported as scan jobs, not ordinary geometry optimizations; scan coordinates and per-step energy profiles are exported separately.

## License

GNU General Public License v3 — see [LICENSE](LICENSE).
