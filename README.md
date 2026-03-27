# ORCA Parser

A modular Python parser for [ORCA](https://orcaforum.kofo.mpg.de/) quantum chemistry output files. Extracts computed properties into structured formats (JSON, CSV, HDF5, Markdown) for downstream analysis, machine-learning pipelines, and automated report generation.

## Features

- **14 parser modules** covering SCF energies, orbital data, 6 population analysis schemes (Mulliken, Loewdin, Mayer, Hirshfeld, MBIS, CHELPG), dipole moments, NBO analysis, and more
- **4 output formats**: JSON (with optional gzip), CSV (one table per section), HDF5, and AI-readable Markdown
- Handles both **RHF/RKS** and **UHF/UKS** calculations with spin-resolved data
- **Point group symmetry** support (irrep labels on orbitals)
- Selective section extraction via aliases (`charges`, `mos`, `bonds`, etc.)
- Batch processing of multiple files with **multi-molecule comparison** reports
- **Quasi-Restricted Orbital (QRO)** parsing for UHF calculations
- Comprehensive **Natural Bond Orbital (NBO)** analysis extraction

## Requirements

- Python >= 3.10
- Optional: `h5py` and `numpy` (for HDF5 output)

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

# Compare multiple molecules
orca_parser water.out ethanol.out benzene.out --compare --outdir results/

# Human-readable summary to stdout
orca_parser water.out --summary
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

# Multi-molecule comparison
parsers = [ORCAParser(f).parse() or p for f, p in
           [("mol1.out", ORCAParser("mol1.out")),
            ("mol2.out", ORCAParser("mol2.out"))]]
p1, p2 = ORCAParser("mol1.out"), ORCAParser("mol2.out")
p1.parse(); p2.parse()
ORCAParser.compare([p1, p2], "comparison.md")
```

## Extracted Data

### Core (always included)

| Module | Data |
|--------|------|
| **metadata** | ORCA version, job name, functional, basis set, charge, multiplicity |
| **geometry** | Cartesian coordinates (Å and a.u.), internal coordinates, symmetry |
| **basis_set** | Basis set groups, atom mappings, shell count |
| **scf** | Total energy, energy components, DFT data, convergence, spin expectation |

### Optional Sections

| Module | Data |
|--------|------|
| **orbital_energies** | HOMO/LUMO, orbital energies (RHF/UHF), irrep labels |
| **qro** | Quasi-Restricted Orbitals — DOMO/SOMO/VMO classification (UHF only) |
| **mulliken** | Mulliken charges, spin populations, reduced orbital charges |
| **loewdin** | Loewdin charges, spin populations, reduced orbital charges |
| **mayer** | Mayer bond orders, atomic valence data (NA, ZA, QA, VA, BVA, FA) |
| **hirshfeld** | Hirshfeld charges and spin populations |
| **mbis** | MBIS charges, populations, valence shell data |
| **chelpg** | CHELPG electrostatic potential charges |
| **dipole** | Dipole moment (electronic, nuclear, total), rotational constants |
| **nbo** | NAO, NPA, Wiberg indices, Lewis structure, E(2) perturbation, NLMO |

### Section Aliases

Use aliases on the CLI to select groups of related sections:

| Alias | Expands to |
|-------|------------|
| `all` | All 14 modules (default) |
| `charges` | mulliken, loewdin, hirshfeld, mbis, chelpg |
| `population` | mulliken, loewdin, mayer |
| `mos` | orbital_energies, qro |
| `bonds` | mayer, loewdin |
| `nbo` | nbo |
| `dipole` | dipole |
| `geometry` | geometry, basis_set |

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

AI-readable reports optimized for LLM-assisted paper writing. Includes publication-ready tables, spin diagnostics, and frontier orbital analysis.

```bash
orca_parser water.out --markdown                   # Per-file report
orca_parser *.out --compare --outdir results/      # Multi-molecule comparison
```

## CLI Reference

```
orca_parser [options] FILE [FILE ...]

Positional:
  FILE                    One or more ORCA output files

Section Selection:
  --sections SEC [SEC ...]  Sections/aliases to parse (default: all)

Output Formats:
  --json / --no-json      JSON output (default: on)
  --csv / --no-csv        CSV output (default: on)
  --hdf5 / --no-hdf5      HDF5 output (default: off)
  --markdown / --no-markdown  Markdown report (default: off)
  --compare               Multi-molecule comparison (implies --markdown)

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
  --summary               Print human-readable summary to stdout
  --quiet                 Suppress progress messages
```

## License

GNU General Public License v3 — see [LICENSE](LICENSE).
