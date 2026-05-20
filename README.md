# ORCA Parser

Structured parsing for [ORCA](https://orcaforum.kofo.mpg.de/) quantum-chemistry output files.

`orca_parser` turns ORCA jobs into clean JSON, CSV, HDF5, and Markdown so the results are easier to inspect, compare, automate, and feed into downstream analysis pipelines.

It is designed for real multistep ORCA outputs, not just ideal single-point jobs. That includes excited-state optimizations, GOAT conformer searches, relaxed surface scans, DeltaSCF jobs, CASSCF/NEVPT2 active-space calculations, coupled-cluster CCSD/CCSD(T)/F12 jobs, EOM/STEOM-CCSD excited-state jobs, symmetry-aware calculations, and spin-resolved wavefunctions.

## Why this project exists

- Parse ORCA outputs once and reuse the data everywhere.
- Keep machine-readable exports and human-readable reports aligned.
- Prefer final converged data for multistep jobs instead of accidentally using the first printed block.
- Make it easier to add new calculation families without rewriting every output writer.

## Highlights

- Built-in parser modules covering SCF, orbitals, solvation, TDDFT/CIS, EOM/STEOM, coupled cluster, GOAT, density-specific analyses, population analyses, dipole moments, NBO/NPA, EPR, geometry optimization, and relaxed scans
- 4 output formats: JSON, CSV, HDF5, and Markdown
- RHF/RKS and UHF/UKS support, including spin-resolved data
- GOAT ensemble parsing with populations, energy windows, global minimum, and ensemble thermochemistry
- CASSCF / SC-NEVPT2 / QD-NEVPT2 parsing with active-space convergence histories, CAS roots, transition-energy assignments, density matrices, spectra, QDPT relativistic properties, and active-MO compositions
- CCSD / CCSD(T) / F12 parsing with MDCI wavefunction settings, CC convergence, T1/singles diagnostics, F12 corrections, perturbative triples, and natural occupations
- EOM/STEOM-CCSD parsing with IP/EA active-root selection, EOM and STEOM root amplitudes, active/singles character, unrelaxed excited-state dipoles, spectra through the shared spectrum parser, and NTOs through the shared transition-orbital parser
- Double-hybrid / MP2 optimization density tracking that keeps SCF, unrelaxed MP2, and relaxed MP2 population/NBO analyses separate for initial and final geometries
- Excited-state geometry optimization support with target-root tracking, root-follow metadata, and cycle-by-cycle TDDFT/CIS state/spectrum trajectory diagnostics
- Repeated NBO/NPA blocks are provenance-aware: optimization jobs select the final valid NBO block, record block/cycle/density/root metadata, and keep supplemental non-density NBO subsections when ORCA prints them only in another block
- TDDFT/NTO handling that preserves ORCA root numbering while also exposing energy rank
- Symmetry-aware parsing for UseSym jobs, with explicit no-symmetry normalization when ORCA defaults symmetry off
- Recursive directory parsing for `*.out` and `*.log`, while skipping ORCA helper/ECP files such as `*_atom83.out`
- Normalized parse-time views for downstream development: `job_snapshot`, `job_series`, and `final_snapshot`

## Supported workflows

| Workflow | What is extracted |
| --- | --- |
| Single point | Final energy, orbitals, dipole, population analyses, spectra, NBO, EPR, symmetry, solvation |
| DeltaSCF | Occupation controls, MOM/IMOM metadata, explicit excited-state labeling |
| TDDFT / CIS vertical excitations | Excited states, CI contributions, NTOs, absorption/CD spectra, root-energy ranking |
| EOM / STEOM-CCSD | CIS seed roots, IP/EA active-space selection, EOM roots, STEOM roots/amplitudes, shared-format spectra, NTOs, and unrelaxed excited-state dipoles |
| CCSD / CCSD(T) / F12 | MDCI wavefunction setup, CC convergence history, F12 corrections, perturbative triples, T1/singles diagnostics, largest amplitudes, and natural orbital occupations |
| Excited-state geometry optimization | Final converged geometry-dependent properties, target-state metadata, per-cycle history, and TDDFT/CIS optimization-step state tracking |
| Ground-state geometry optimization | Per-cycle energies, convergence criteria, trust radii, RMSD, final geometry |
| Double-hybrid / MP2 geometry optimization | Initial/final SCF density, unrelaxed MP2 density, relaxed MP2 density, MP2 density-formation metadata, density-specific population/NBO analyses, and density dipoles when printed |
| Relaxed surface scan | Scan coordinates, mode, per-step energies, sidecar trajectories |
| GOAT conformer search | Final ensemble table, populations, relative energies, thermochemistry, xyz references |
| CASSCF / NEVPT2 | Macro-iteration convergence history, active occupations, CAS-SCF roots/configurations joined to SA-CASSCF/NEVPT2/QD-NEVPT2 state assignments, density/spin-density matrices, Loewdin active MOs, UV/CD spectra, QDPT eigenvectors, g tensors, and D tensors |
| UseSym / symmetry-cleanup jobs | Point group, reduced/orbital irrep groups, symmetry-perfected geometries when printed |

## Requirements

- Python 3.10+
- Base JSON, CSV, and Markdown parsing uses the Python standard library
- Optional: `numpy` for Kabsch-aligned RMSD during geometry optimization analysis
- Optional: `numpy` and `h5py` for HDF5 export
- Optional: `fastapi` and `uvicorn` for the ORCA Workbench local web app

## Installation

Standard install:

```bash
python -m pip install .
```

Editable install for development:

```bash
python -m pip install -e .
```

Install optional HDF5 support:

```bash
python -m pip install ".[hdf5]"
```

Editable development install with test dependencies:

```bash
python -m pip install -e ".[test]"
```

Editable install with the ORCA Workbench backend dependencies:

```bash
python -m pip install -e ".[workbench]"
```

## Privacy guardrails

This repo now includes an explicit tracked-file safety check so local scratch
artifacts and private reference outputs do not quietly leak into git history.

- CI runs [`tools/check_tracked_private_artifacts.py`](tools/check_tracked_private_artifacts.py) on pushes and pull requests
- blocked patterns live in [`.privacy_guardrails.json`](.privacy_guardrails.json)
- the current guardrail blocks tracked `.codex*`, `.pytest_tmp`, `.pytest_cache`, `sample_md`, `sample_outs`, and `codex_eval_tmp_privacy` paths

To enable the local pre-commit hook:

```bash
git config core.hooksPath .githooks
```

To run the check manually:

```bash
python tools/check_tracked_private_artifacts.py --repo-root .
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

# Double-hybrid optimization with SCF/MP2 relaxed and unrelaxed density contexts
orca_parser double_hybrid_opt.out --sections opt --markdown --csv

# CASSCF / NEVPT2 active-space report
orca_parser casscf.out --sections casscf --markdown --csv

# Keep a wider CASSCF orbital-energy / Loewdin active-window range
orca_parser casscf.out --sections casscf --casscf-orbital-window 50 --markdown

# CCSD(T)/F12 coupled-cluster report
orca_parser ccsdt.out --sections ccsdt --markdown --csv

# EOM/STEOM-CCSD report
orca_parser steom.out --sections steom --markdown --csv
```

### ORCA Workbench local web UI

The project also includes a modern local web workbench:

```bash
orca_workbench
```

This starts the FastAPI backend on `http://127.0.0.1:8765`. For frontend
development, run the React/Vite interface in another terminal:

```bash
cd orca_workbench/web
npm install
npm run dev
```

Then open `http://127.0.0.1:5173`.

To build the frontend so the Python server can serve it directly:

```bash
cd orca_workbench/web
npm install
npm run build
cd ../..
orca_workbench
```

After a build, open `http://127.0.0.1:8765`.

ORCA Workbench is a parser-backed scientific workbench and data visualizer. It
lets you open local ORCA `.out` / `.log` files or folders with native file
dialogs, automatically parses the selected outputs, discovers the property
blocks actually present, selects those parsed properties for viewing by default,
and then builds the visible analysis panels from what was actually parsed.

The interface is organized around detected scientific domains rather than a
fixed list of generic parser tabs. For example, a TDDFT output exposes spectra,
orbitals, geometry, populations, excited-state, table, provenance, raw-data, and
export workspaces only when those surfaces are present. A CASSCF/NEVPT2 output
adds the multireference workspace; a plain optimization does not pretend to have
spectra. Utility views such as `Overview`, `Tables`, `Provenance`, `Raw data`,
and `Exports` remain available when the parsed payload supports them.

Plots are chosen by data type, not forced through one generic chart. Spectra are
shown as oscillator-strength stick plots, orbital energies as level ladders,
geometries as coordinate projections, populations as signed charge/spin/bond
bars, excited and multireference states as energy ladders, and optimization
lines are drawn only for ordered cycle/iteration series.

Global JSON/CSV/Markdown/HDF5 exports remain available through the parser
backend. The workbench also provides immediate client-side exports for the
currently selected structured JSON, all visible tables, and filtered table rows.
The default workflow is data-first: open outputs -> parse everything
discoverable -> use the detected scientific panels -> hide irrelevant property
blocks -> visualize and export the curated analysis surface.

The workbench is intentionally a wrapper around the existing parser. It does
not reimplement population, NBO, spectra, TDDFT, CASSCF, NEVPT2, or any other
scientific parser logic. Parser options exposed by modules, such as the CASSCF
orbital-window control, are discovered from the same plugin metadata used by
the CLI.

The browser-level smoke test exercises the real GUI, including automatic
parsing, detected-domain navigation, spectra/orbital/table visualization,
property filtering, screenshots, and JSON/CSV downloads:

```bash
cd orca_workbench/web
npm run test:ui
```

The older standard-library Tk prototype remains available as a fallback:

```bash
orca_workbench_tk
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

Requires the optional HDF5 dependencies (`python -m pip install ".[hdf5]"`).

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
- NTO mapping uses an energy-matching tolerance of `+/-0.005 eV`
- Optimization-step spectra are kept separate from final single-point TDDFT spectra with `source_context = optimization_step_spectrum` or `final_single_point_tddft`
- Trajectory CSVs join each cycle's `STATE` block to the same-cycle electric-dipole spectrum, preserving ORCA roots, ranked `S1`/`S2` labels, oscillator strengths, transition dipoles, root-following overlaps, near-degeneracy flags, and possible root/state tracking changes
- Singlet, triplet, and other manifolds are tracked separately; oscillator strengths from a singlet spectrum are not assigned to triplet roots that happen to share the same printed root number

Fixed parser-level thresholds:

- CI / MO-basis contributions: keep all contributions with `weight >= 10%`
- NTO pairs: keep all pairs with `n >= 0.10`
- Oscillator strengths: always keep the printed value, including dark states
- Lowest-root safety margin: expose at least the lowest 5 roots in the TDDFT summary helpers

## CASSCF / NEVPT2 reporting policy

The CASSCF module treats active-space convergence as a history, not just a final number.

- Every macro-iteration keeps `E(CAS)`, `DE`, Ext-Act/Act-Int gaps, active occupations, `||g||`, `Max(G)`, rotation labels, FreezeAct/related option metrics, orbital-update mode, and SuperCI details when printed.
- CAS-SCF state blocks preserve ORCA block, multiplicity, root, absolute energy, relative energy, and configuration weights/occupation strings.
- Markdown reports join SA-CASSCF transition rows to the matching CAS-SCF root/configuration data, so rows expose state, root, multiplicity, transition energy, absolute energy, dominant configuration, and all printed configuration weights as explicit `weight -> configuration` pairs such as `0.28237 -> 210000`.
- NEVPT2 and QD-NEVPT2 reports join transition energies, transition-energy corrections, total energies, root energy corrections, and corrected CI configurations into digestible state tables instead of raw Van Vleck text dumps.
- Density and spin-density matrices are exported as JSON-safe row/column/matrix structures rather than NumPy objects.
- QD-NEVPT2 corrected density/spin-density matrices, corrected reduced active MOs, and state-specific natural-orbital occupations/files are parsed separately from the uncorrected CASSCF values.
- CASSCF, CASSCF with NEVPT2 diagonal energies, QD-NEVPT2, TDDFT/CIS, and SOC-corrected QDPT UV/CD spectra are parsed through the shared ORCA spectrum table parser.
- QDPT relativistic sections are parsed into level, eigenvector-component, g-matrix, zero-field-splitting, and SOC-corrected spectrum summary tables for each printed energy model.
- Ordinary Mulliken/Loewdin/Mayer/Hirshfeld/MBIS/CHELPG population analysis is handled by the existing population modules, including repeated CASSCF/QD-NEVPT2 population-analysis passes. NBO output is handled by the existing NBO module. The `casscf` and `nevpt2` aliases expand to those shared modules automatically rather than reimplementing their grammars inside the CASSCF parser.
- The CASSCF module only owns CASSCF-specific population-like tables: Loewdin reduced active MOs, QD-NEVPT2 corrected reduced active MOs, state-specific QD-NEVPT2 natural-orbital occupations, and a bounded Loewdin orbital-composition window around the active space.
- Raw ORCA report blocks are not embedded in normal JSON/HDF5/Markdown output; long Van Vleck, spectra, and relativistic sections are represented by parsed tables.
- Markdown keeps CASSCF orbital-energy tables centered on the active/frontier window; it does not print the full orbital list by default.
- `--casscf-orbital-window N` (alias: `--casscf-orbital-energy-window N`) controls how many orbitals below and above the active/frontier range are retained from CASSCF orbital-energy and Loewdin active-window tables. The default is `30`.

## Coupled-cluster / EOM / STEOM reporting policy

The coupled-cluster module owns MDCI/CC-specific data, and only that data.

- `coupled_cluster` parses wavefunction settings, algorithmic/DLPNO settings, CC iteration history, coupled-cluster energy components, F12 corrections, perturbative triples, largest amplitudes, and CC natural orbital occupations.
- `eom_steom` parses CIS seed roots, IP/EA active-root selection, EOM roots, STEOM roots, root amplitudes, active/singles character, unrelaxed excited-state dipoles, STEOM spectra, and STEOM NTOs.
- Ordinary Mulliken/Loewdin/Mayer/Hirshfeld/MBIS/CHELPG population analysis is still handled by the existing population modules. NBO output is still handled by the existing NBO module. The `cc`, `ccsd`, `ccsdt`, `coupled_cluster`, `eom`, `steom`, and `eom_steom` aliases expand to those shared modules instead of copying their grammars.
- STEOM and EOM spectra use `spectrum_parser.py`, the same shared table parser used by TDDFT/CIS and CASSCF-derived spectra.
- STEOM NTOs use `transition_orbitals.py`, the same shared NTO parser used by TDDFT/CIS.
- TDDFT no longer consumes STEOM/CASSCF spectra or NTOs merely because the table format is shared; method-specific context decides which module owns the output.
- Raw ORCA report blocks are not embedded in normal CC/EOM/STEOM JSON, HDF5, CSV, or Markdown output.

## Double-hybrid / MP2 density reporting policy

The `density_analysis` module exists for ORCA outputs where one calculation prints population analyses for more than one density object.

- Double-hybrid and MP2 optimizations can print the SCF density, unrelaxed MP2 density, and relaxed MP2 density at the optimization geometry and again at the final stationary geometry.
- These are preserved as separate structured records with `stage` (`initial`, `final`, or cycle-specific), `density_kind` (`scf`, `mp2_unrelaxed`, `mp2_relaxed`), and the printed density filename such as `F3CNO.scfp`, `F3CNO.pmp2ur`, or `F3CNO.pmp2re`.
- The module records MP2 density-formation metadata, including stored relaxed/unrelaxed-density flags, SCF/correlated/energy-weighted density names, density trace, natural occupations, MP2 total energy, and gradient norm when printed.
- Mulliken, Loewdin, Mayer, Hirshfeld, MBIS, CHELPG, and NBO analyses inside each density block are parsed by their existing owner modules. `density_analysis` only slices the ORCA output into density contexts and calls those modules.
- Density-specific dipole blocks are parsed through the shared dipole parser and kept separate for SCF, unrelaxed MP2, and relaxed MP2 densities when ORCA prints them.
- The `opt` alias includes `density_analysis`, so double-hybrid geometry optimizations do not lose the MP2 density contexts when parsed as ordinary optimizations.

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
| `tddft` | Excited states, spectra, CI contributions, NTOs, energy-rank metadata, excited-state optimization metadata, and TDDFT/CIS optimization-step trajectory diagnostics |
| `coupled_cluster` | CCSD/CCSD(T)/F12 wavefunction settings, CC convergence, diagnostics, F12/triples corrections, largest amplitudes, and natural occupations |
| `eom_steom` | EOM/STEOM active-root selection, EOM roots, STEOM amplitudes, spectra, NTOs, and unrelaxed excited-state dipoles |
| `density_analysis` | Density-specific SCF, unrelaxed MP2, and relaxed MP2 population/NBO analyses, MP2 density formation metadata, sidecar density containers, and density dipoles |
| `casscf` | CASSCF convergence history, active-space setup, joined CAS-SCF/NEVPT2/QD-NEVPT2 state assignments, matrices, energy components, active-MO compositions, repeated population-analysis passes, spectra including SOC-corrected QDPT tables, QDPT eigenvectors, g tensors, and D tensors |
| `goat` | Final ensemble, populations, relative energies, thermochemistry, minimum/ensemble xyz references |
| `surface_scan` | Scan definitions, mode, coordinates, energies, optimized xyz files, sidecar files |
| `mulliken` | Charges, spin populations, reduced orbital charges |
| `loewdin` | Charges, spin populations, reduced orbital charges |
| `mayer` | Bond orders and atomic valence data |
| `hirshfeld` | Charges and spin populations |
| `mbis` | Charges, populations, valence-shell data |
| `chelpg` | Electrostatic potential charges |
| `dipole` | Cartesian dipole components, total magnitude, rotational constants |
| `nbo` | NAO, NPA, Wiberg indices, Lewis structure, E(2), NLMO; repeated optimization blocks are selected by final-block provenance and NPA CSV/Markdown includes block/density/root metadata |
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
| `cc`, `ccsd`, `ccsdt`, `coupled_cluster` | `coupled_cluster`, `mulliken`, `loewdin`, `mayer`, `hirshfeld`, `mbis`, `chelpg`, `nbo`, `dipole` |
| `eom`, `steom`, `eom_steom` | `coupled_cluster`, `eom_steom`, `mulliken`, `loewdin`, `mayer`, `hirshfeld`, `mbis`, `chelpg`, `nbo`, `dipole` |
| `density_analysis`, `densities` | `density_analysis` |
| `double_hybrid` | `geom_opt`, `density_analysis` |
| `casscf` | `casscf`, `mulliken`, `loewdin`, `mayer`, `hirshfeld`, `mbis`, `chelpg`, `nbo` |
| `nevpt2` | `casscf`, `mulliken`, `loewdin`, `mayer`, `hirshfeld`, `mbis`, `chelpg`, `nbo` |
| `geometry` | `geometry`, `basis_set` |
| `epr` | `epr` |
| `goat` | `goat` |
| `opt` | `geom_opt`, `density_analysis` |
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
  --casscf-orbital-window N / --casscf-orbital-energy-window N

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

### How to incorporate a new module now

The recommended path is now an explicit plugin bundle with automatic
discovery.

1. Add a new module file under `orca_parser/modules/`.
2. Subclass `BaseModule` and implement `parse()`.
3. Export a `PLUGIN_BUNDLE` that declares what the module contributes:
   - parser sections
   - section aliases
   - optional calculation-family behavior
   - optional Markdown / CSV section hooks
   - optional CLI-visible plugin options
   - short documentation metadata
4. Let autodiscovery register the bundle into the existing registries.

That means a new feature can usually arrive by *adding one file* instead of
editing the top-level parser, Markdown writer, CSV writer, and help text in
parallel.

Example shape:

```python
from orca_parser.modules.base import BaseModule
from orca_parser.plugin_bundle import PluginBundle, PluginMetadata, PluginOption
from orca_parser.parser_section_plugin import ParserSectionAlias, ParserSectionPlugin


class MyNewModule(BaseModule):
    name = "my_new_section"

    def parse(self, lines):
        return {"example": True}


PLUGIN_BUNDLE = PluginBundle(
    metadata=PluginMetadata(
        key="my_new_plugin",
        name="My New Plugin",
        short_help="Parse my new ORCA section.",
        docs_path="docs/my_new_plugin.md",
        examples=("orca_parser calc.out --sections my_new_section",),
    ),
    parser_sections=(
        ParserSectionPlugin(key="my_new_section", module_class=MyNewModule),
    ),
    parser_aliases=(
        ParserSectionAlias(name="my_new_alias", section_keys=("my_new_section",)),
    ),
    options=(
        PluginOption(
            dest="my_new_threshold",
            flags=("--my-new-threshold",),
            help="Example plugin-owned CLI option.",
            default=0.1,
            type=float,
        ),
    ),
)
```

The parser/CLI bootstrap layer discovers `PLUGIN_BUNDLE` objects under
`orca_parser.modules` automatically. Plugin-owned option values are exposed to
modules through `self.context["plugin_options"]` and carried into parsed data
as `data["plugin_options"]`, so future family/render hooks can use the same
declarative option channel.

If the new data is authoritative final-step or stepwise history data, wire it
into the normalized parse-time layers instead of teaching each writer how to
guess it from raw payloads:
- `final_snapshot` for final converged properties
- `job_series` for stepwise histories
- `job_snapshot` for normalized job identity / classification

## Important ORCA-specific behavior

- Helper/ECP files such as `*_atom83.out` are treated as auxiliary ORCA outputs and skipped
- Recursive directory parsing skips helper files automatically
- Passing a helper file directly raises a validation error
- DeltaSCF jobs are labeled explicitly instead of being treated as ordinary single points
- `%tddft` / `%cis` geometry optimizations are labeled as excited-state optimizations
- Excited-state optimizations export target-state metadata, root-follow history, and cycle-by-cycle state/spectrum trajectory tables when available
- Geometry-dependent properties for multistep jobs come from the final converged block, not the first printed block
- UseSym jobs report point-group and irrep-group information separately when ORCA prints both
- If neither `UseSym` nor an explicit `%sym` request enables symmetry, the normalized input intent is treated as symmetry-off to match ORCA's default behavior
- GOAT jobs are exported as GOAT calculations, not ordinary single points or optimizations
- Relaxed surface scans are treated as scan jobs, not ordinary geometry optimizations

## License

GNU General Public License v3. See [LICENSE](LICENSE).
