# ORCA Parser

Structured parsing for [ORCA](https://orcaforum.kofo.mpg.de/) quantum-chemistry output files.

`orca_parser` turns ORCA jobs into clean JSON, CSV, HDF5, and Markdown so the results are easier to inspect, compare, automate, and feed into downstream analysis pipelines.

It is designed for real multistep ORCA outputs, not just ideal single-point jobs. That includes excited-state optimizations, GOAT conformer searches, relaxed surface scans, DeltaSCF jobs, symmetry-aware calculations, and spin-resolved wavefunctions.

## Why this project exists

- Parse ORCA outputs once and reuse the data everywhere.
- Keep machine-readable exports and human-readable reports aligned.
- Prefer final converged data for multistep jobs instead of accidentally using the first printed block.
- Make it easier to add new calculation families without rewriting every output writer.

## Highlights

- 20 parser modules covering SCF, orbitals, solvation, TDDFT/CIS, GOAT, population analyses, dipole moments, NBO, EPR, geometry optimization, and relaxed scans
- 4 output formats: JSON, CSV, HDF5, and Markdown
- RHF/RKS and UHF/UKS support, including spin-resolved data
- GOAT ensemble parsing with populations, energy windows, global minimum, and ensemble thermochemistry
- Excited-state geometry optimization support with target-root tracking and root-follow metadata
- TDDFT/NTO handling that preserves ORCA root numbering while also exposing energy rank
- Symmetry-aware parsing for UseSym jobs, including point-group and irrep metadata
- Recursive directory parsing for `*.out` and `*.log`, while skipping ORCA helper/ECP files such as `*_atom83.out`
- Normalized parse-time views for downstream development: `job_snapshot`, `job_series`, and `final_snapshot`

## Supported workflows

| Workflow | What is extracted |
| --- | --- |
| Single point | Final energy, orbitals, dipole, population analyses, spectra, NBO, EPR, symmetry, solvation |
| DeltaSCF | Occupation controls, MOM/IMOM metadata, explicit excited-state labeling |
| TDDFT / CIS vertical excitations | Excited states, CI contributions, NTOs, absorption/CD spectra, root-energy ranking |
| Excited-state geometry optimization | Final converged geometry-dependent properties, target-state metadata, per-cycle history |
| Ground-state geometry optimization | Per-cycle energies, convergence criteria, trust radii, RMSD, final geometry |
| Relaxed surface scan | Scan coordinates, mode, per-step energies, sidecar trajectories |
| GOAT conformer search | Final ensemble table, populations, relative energies, thermochemistry, xyz references |
| UseSym / symmetry-cleanup jobs | Point group, reduced/orbital irrep groups, symmetry-perfected geometries when printed |

## Requirements

- Python 3.10+
- Optional: `numpy` for Kabsch-aligned RMSD during geometry optimization analysis
- Optional: `h5py` for HDF5 export

## Installation

Standard install:

```bash
pip install .
```

Editable install for development:

```bash
pip install -e .
```

## Quick start

### Command line

```bash
# Parse a single file (JSON + CSV by default)
orca_parser water.out

# Extract only selected sections
orca_parser water.out --sections charges mos

# Produce a Markdown report
orca_parser water.out --markdown

# Compare several jobs in one report
orca_parser water.out ethanol.out benzene.out --compare --outdir results/

# Recursively compare all ORCA outputs in a directory tree
orca_parser calculations/ --compare --outdir results/

# Human-readable summary to stdout
orca_parser water.out --summary

# GOAT final ensemble
orca_parser conformers.out --sections goat --markdown --csv

# Relaxed scan
orca_parser relaxscan.out --sections scan --markdown --csv

# Excited-state optimization
orca_parser excited_opt.out --sections tddft opt --markdown --csv
```

### Python API

```python
from orca_parser import ORCAParser

parser = ORCAParser("water.out")
data = parser.parse()

energy = data["scf"]["final_single_point_energy_Eh"]
charges = data["mulliken"]["atomic_charges"]

parser.to_json("water.json")
parser.to_csv("water_csv/")
parser.to_markdown("water.md")

# Normalized parse-time views
job = data["job_snapshot"]
final = data["final_snapshot"]
series = data["job_series"]

print(job["calculation_family"])
print(final["orbital_energies"]["HOMO_energy_eV"])
print(len(series.get("goat", {}).get("ensemble", [])))
```

## Output formats

| Format | Purpose | Default |
| --- | --- | --- |
| JSON | Full structured export for downstream tooling | On |
| CSV | One table per section for spreadsheets and scripts | On |
| HDF5 | Hierarchical binary export for large workflows | Off |
| Markdown | Human-readable / AI-readable report | Off |

### JSON

```bash
orca_parser water.out
orca_parser water.out --compact
orca_parser water.out --gzip
orca_parser water.out --indent 4 --strip-none
```

### CSV

```bash
orca_parser water.out
orca_parser water.out --no-csv
```

### HDF5

```bash
orca_parser water.out --hdf5 --no-json --no-csv
orca_parser water.out --hdf5 --h5-compression lzf
orca_parser water.out --hdf5 --h5-level 9
```

### Markdown

```bash
orca_parser water.out --markdown
orca_parser *.out --compare --outdir results/
orca_parser water.out --markdown --detail-scope compact
orca_parser *.out --compare --detail-scope full
orca_parser conformers.out --markdown --goat-max-relative-energy-kcal 3
orca_parser conformers.out other.out --compare --goat-max-relative-energy-kcal all
```

## Markdown detail policy

Markdown uses a shared render-policy layer so bulky sections behave consistently.

- Stand-alone reports default to `full`
- Compare mode defaults to `compact`
- `--detail-scope compact` forces compact rendering
- `--detail-scope full` forces full rendering

This currently affects Markdown/report truncation behavior. JSON and CSV remain full-fidelity exports.

GOAT-specific behavior:

- Stand-alone GOAT Markdown prints the full ensemble by default
- Compare mode truncates GOAT tables at `dE <= 10 kcal/mol` by default
- `--goat-max-relative-energy-kcal` overrides that behavior

## TDDFT / NTO reporting policy

The TDDFT module keeps ORCA's printed root numbering and adds extra metadata instead of silently renumbering states by energy.

- `state` = ORCA printed root number
- `energy_rank` = where that root falls after sorting by excitation energy
- NTO mapping uses an energy-matching tolerance of `+-0.005 eV`

Fixed parser-level thresholds:

- CI / MO-basis contributions: keep all contributions with `weight >= 10%`
- NTO pairs: keep all pairs with `n >= 0.10`
- Oscillator strengths: always keep the printed value, including dark states
- Lowest-root safety margin: expose at least the lowest 5 roots in the TDDFT summary helpers

## Common recipes

### Single point, full standalone report

```bash
orca_parser sp.out --markdown
```

### Compact standalone report

```bash
orca_parser sp.out --markdown --detail-scope compact
```

### Full compare report

```bash
orca_parser calc_a.out calc_b.out --compare --detail-scope full
```

### GOAT ensemble within a chosen energy window

```bash
orca_parser goat.out --sections goat --markdown --goat-max-relative-energy-kcal 5
```

### TDDFT spectroscopy export

```bash
orca_parser vertical_tddft.out --sections tddft --markdown --csv
```

## Extracted data

### Core sections

These are always included.

| Module | Data |
| --- | --- |
| `metadata` | ORCA version, job name, method, basis, charge, multiplicity, job metadata, symmetry metadata |
| `geometry` | Cartesian coordinates, internal coordinates, symmetry-perfected geometry when printed |
| `basis_set` | Basis groups, atom mappings, shell count |
| `scf` | Final single-point energy, components, convergence, DFT data, spin expectation |

### Optional sections

| Module | Data |
| --- | --- |
| `orbital_energies` | HOMO/LUMO, orbital energies, irrep labels, occupied orbitals per irrep |
| `qro` | Quasi-restricted orbitals for UHF calculations |
| `solvation` | CPCM/SMD, ALPB, COSMO-RS, and `%cpcm` / `%cosmors` metadata |
| `tddft` | Excited states, spectra, CI contributions, NTOs, energy-rank metadata, excited-state optimization metadata |
| `goat` | Final ensemble, populations, relative energies, thermochemistry, minimum/ensemble xyz references |
| `surface_scan` | Scan definitions, mode, coordinates, energies, optimized xyz files, sidecar files |
| `mulliken` | Charges, spin populations, reduced orbital charges |
| `loewdin` | Charges, spin populations, reduced orbital charges |
| `mayer` | Bond orders and atomic valence data |
| `hirshfeld` | Charges and spin populations |
| `mbis` | Charges, populations, valence-shell data |
| `chelpg` | Electrostatic potential charges |
| `dipole` | Cartesian dipole components, total magnitude, rotational constants |
| `nbo` | NAO, NPA, Wiberg indices, Lewis structure, E(2), NLMO |
| `epr` | ZFS, g-tensor, hyperfine coupling, EFG, quadrupole |
| `geom_opt` | Per-cycle energies, convergence criteria, trust radii, RMSD, optimization settings |

### Section aliases

| Alias | Expands to |
| --- | --- |
| `all` | Everything |
| `charges` | `mulliken`, `loewdin`, `hirshfeld`, `mbis`, `chelpg` |
| `population` | `mulliken`, `loewdin`, `mayer` |
| `mos` | `orbital_energies`, `qro` |
| `bonds` | `mayer`, `loewdin` |
| `nbo` | `nbo` |
| `dipole` | `dipole` |
| `solvation` | `solvation` |
| `tddft` | `tddft` |
| `geometry` | `geometry`, `basis_set` |
| `epr` | `epr` |
| `goat` | `goat` |
| `opt` | `geom_opt` |
| `scan` | `surface_scan` |

## CLI reference

```text
orca_parser [options] FILE [FILE ...]

Positional:
  FILE
      One or more ORCA output files or directories.
      Directories are searched recursively for *.out and *.log files.
      Auxiliary helper/ECP files such as *_atom83.out are skipped.

Section selection:
  --sections SEC [SEC ...]
      Sections or aliases to parse (default: all)

Output formats:
  --json / --no-json
  --csv / --no-csv
  --hdf5 / --no-hdf5
  --markdown / --no-markdown
  --compare

Markdown/report controls:
  --detail-scope auto|full|compact
  --goat-max-relative-energy-kcal N|all

JSON options:
  --indent N
  --compact
  --gzip
  --strip-none

HDF5 options:
  --h5-compression FILTER
  --h5-level N

Misc:
  --outdir DIR
  --summary
  --quiet
```

For the full up-to-date help text, run:

```bash
orca_parser --help
```

## Development notes

The codebase now has normalized parse-time views that downstream tooling should prefer over raw module payloads.

- `data["job_snapshot"]`
  Canonical job identity, family classification, display labels, symmetry summary, DeltaSCF metadata, excited-state metadata
- `data["final_snapshot"]`
  Authoritative final-step geometry-dependent properties such as final geometry, frontier orbitals, dipole, charges, and bond orders
- `data["job_series"]`
  Authoritative stepwise histories for GOAT, geometry optimization, relaxed scans, and excited-state optimization cycles

These layers exist to prevent repeated "first block vs last block" bugs and to keep Markdown, CSV, and CLI summaries aligned.

Calculation-family output behavior is also moving toward registry-driven dispatch, so new families can be added with less scattered wiring than before.
Parser-section registration now follows the same direction through `parser_section_registry.py`, so new sections and aliases can be added without editing `parser.py`.
Common standalone/comparison markdown sections now register through `output/markdown_section_registry.py`, and common CSV exports register through `output/csv_section_registry.py`, so new output sections can be added without extending the top-level writer lists.

## Important ORCA-specific behavior

- Helper/ECP files such as `*_atom83.out` are treated as auxiliary ORCA outputs and skipped
- Recursive directory parsing skips helper files automatically
- Passing a helper file directly raises a validation error
- DeltaSCF jobs are labeled explicitly instead of being treated as ordinary single points
- `%tddft` / `%cis` geometry optimizations are labeled as excited-state optimizations
- Excited-state optimizations export target-state metadata and root-follow history when available
- Geometry-dependent properties for multistep jobs come from the final converged block, not the first printed block
- UseSym jobs report point-group and irrep-group information separately when ORCA prints both
- GOAT jobs are exported as GOAT calculations, not ordinary single points or optimizations
- Relaxed surface scans are treated as scan jobs, not ordinary geometry optimizations

## License

GNU General Public License v3. See [LICENSE](LICENSE).
