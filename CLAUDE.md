# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable for development)
pip install -e .

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_rdb_lmmp_regressions.py

# Run a single test by name
pytest tests/test_rdb_lmmp_regressions.py -k "test_name"

# CLI usage
python -m orca_parser <file.out> [--sections <sec1,sec2>] [--format json|csv|md|hdf5]
```

No linter/formatter is configured in the project.

## Architecture

### Data Flow

```
ORCA .out file
  → ORCAParser (parser.py)          # orchestrates module dispatch
  → BaseModule subclasses           # each owns one section of the output
  → job_snapshot / final_snapshot / job_series   # normalization layers
  → Output Writers (JSON/CSV/MD/HDF5)
```

### Three Normalization Layers

After raw parsing, three dedicated modules normalize the output before any writer sees it:

- **`job_snapshot.py`** — Canonical job identity: family, method, basis, symmetry, DeltaSCF/excited-state classification. Eliminates scattered "what kind of job is this?" logic from writers.
- **`final_snapshot.py`** — Authoritative final-state properties for multistep jobs (last converged cycle, not first printed block). Normalizes across population analysis schemes.
- **`job_series.py`** — Stepwise histories: geometry optimization cycles, GOAT ensemble members, scan steps, excited-state optimization root-follow metadata.

### Plugin System

Every capability is declared through a `PluginBundle` (see `plugin_bundle.py`). Each module under `orca_parser/modules/` exports a `PLUGIN_BUNDLE` constant that registers:

- Which parser section(s) the module owns (via `parser_section_registry.py`)
- Section aliases (e.g. `charges` → [mulliken, loewdin, hirshfeld, mbis, chelpg])
- Which calculation family the bundle describes, with its `matcher` callable, display labels, and optional CSV/Markdown rendering hooks (via `job_family_registry.py`)
- CLI options the module contributes

`plugin_discovery.py` autodiscovers all `PLUGIN_BUNDLE` instances under `orca_parser.modules` at import time. Bootstrap is idempotent — safe to call from parser, CLI, or library entry points.

### Calculation Families

Defined in `job_family_registry.py`. Each family is a `CalculationFamilyPlugin` with:
- `matcher`: callable that inspects parsed data to claim a job
- `electronic_state_kind`: `"ground_state"` or `"excited_state"`
- Optional hooks: `markdown_comparison_section`, `csv_exports`

Current families: `single_point`, `geometry_optimization`, `tddft`, `excited_state_optimization`, `goat`, `surface_scan`, `deltascf`.

### Parser Modules (`orca_parser/modules/`)

All inherit from `BaseModule` (`modules/base.py`). Key shared context flags passed to every module: `is_uhf`, `has_symmetry`, `hf_type`, `scf_type`, `multiplicity`, `charge`, `plugin_options`. Utility methods: `find_line()`, `find_all_lines()`, `find_last_line()`, `safe_float()`, `safe_int()`.

Core sections always parsed (never skipped): `metadata`, `geometry`, `basis_set`, `scf`.

### Output Writers (`orca_parser/output/`)

- **JSON** (`json_writer.py`): Full-fidelity; numpy-safe encoding; optional gzip.
- **CSV** (`csv_writer.py`): Registry-driven via `CSVSectionPlugin`; each family plugin can inject additional exports.
- **Markdown** (`markdown_writer.py`): Standalone (full detail) and comparison (compact) modes; section truncation policies per section.
- **HDF5** (`hdf5_writer.py`): Binary hierarchical; requires `h5py`.

### ORCA-Specific Behaviors to Preserve

- Auxiliary/ECP sidecar files (`*_atom83.out`, etc.) are rejected at the file-acceptance stage.
- DeltaSCF jobs are explicitly classified — never treated as ordinary single points.
- TDDFT root numbering follows ORCA's printed order; energy-rank metadata is added separately.
- For multistep jobs, the **last** converged block is authoritative, not the first printed one.
- Symmetry-perfected geometries and irrep labels are tracked separately from input geometry.

## Dependencies

Python 3.10+. Optional: `numpy` (Kabsch RMSD), `h5py` (HDF5 export). No external parsing libraries — all parsing is pure regex against ORCA text output.
