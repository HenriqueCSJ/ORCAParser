"""HDF5 output writer for ORCA parser results.

Layout
------
/                           root
├── source_file             str attribute on root
├── context/                group — scalar flags and metadata
├── metadata/               group — ORCA version, job info
├── geometry/               group
│   ├── symbols             string dataset  (n_atoms,)
│   ├── coordinates_Ang     float64 dataset (n_atoms, 3)
│   └── ...
├── scf/                    group — scalar energy etc.
├── orbital_energies/       group
│   ├── alpha_orbitals/     group — index, energy_Eh, energy_eV, ...
│   ├── beta_orbitals/      group (UHF only)
│   └── (scalar attrs)
├── qro/                    group (UHF only)
│   └── orbitals/           group
├── mulliken/               group
│   └── atoms/              group — charge, spin_population per atom
├── nbo/                    group
│   └── nao/, npa/, ...
└── ...

Design rules
------------
* Scalars (str, int, float, bool) → HDF5 attributes on the parent group.
* Lists of dicts (orbital tables, atom tables) → columnar datasets inside a
  sub-group: one 1-D dataset per field name, with a shared length.
* Lists of scalars → 1-D dataset.
* Nested dicts → sub-groups, recursively.
* None values are silently skipped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────

def write_hdf5(
    data: Dict[str, Any],
    path: Path,
    compression: Optional[str] = "gzip",
    compression_opts: int = 4,
) -> Path:
    """Write parsed ORCA data to an HDF5 file.

    Parameters
    ----------
    data : dict
        Output from ``ORCAParser.parse()``.
    path : Path
        Destination ``.h5`` file (created/overwritten).
    compression : str or None
        HDF5 compression filter: ``"gzip"``, ``"lzf"``, or ``None``.
    compression_opts : int
        Compression level for gzip (1–9).

    Returns
    -------
    Path
        The path that was written.
    """
    import h5py  # lazy import — not in stdlib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    _cmp = _compression_kwargs(compression, compression_opts)

    with h5py.File(path, "w") as hf:
        hf.attrs["source_file"] = str(data.get("source_file", ""))
        hf.attrs["orca_parser_version"] = _package_version()

        for section, value in data.items():
            if section in ("source_file",) or value is None:
                continue
            _write_section(hf, section, value, _cmp)

    return path


# ─────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────

def _compression_kwargs(compression: Optional[str], compression_opts: int) -> dict[str, Any]:
    """Return h5py dataset compression kwargs for supported CLI filters."""
    if compression is None:
        return {}

    filter_name = compression.lower()
    if filter_name in ("", "none"):
        return {}
    if filter_name == "gzip":
        return {"compression": "gzip", "compression_opts": compression_opts}
    if filter_name == "lzf":
        return {"compression": "lzf"}

    raise ValueError(f"Unsupported HDF5 compression filter: {compression}")


def _package_version() -> str:
    """Read the package version without letting HDF5 metadata drift stale."""
    try:
        from .. import __version__
    except Exception:
        return "unknown"
    return __version__


def _write_section(parent, name: str, value: Any, cmp: dict) -> None:
    """Dispatch a top-level section value into the HDF5 hierarchy."""
    if value is None:
        return

    if isinstance(value, dict):
        grp = parent.require_group(name)
        _write_dict(grp, value, cmp)

    elif isinstance(value, list):
        grp = parent.require_group(name)
        _write_list(grp, value, cmp)

    elif isinstance(value, (str, int, float, bool, np.integer, np.floating)):
        parent.attrs[name] = _scalar(value)

    # anything else: skip silently


def _write_dict(grp, d: dict, cmp: dict) -> None:
    """Write a dict into an HDF5 group: scalars → attrs, complex → sub-groups."""
    for key, val in d.items():
        key = str(key)  # HDF5 names must be strings
        if val is None:
            continue

        if isinstance(val, dict):
            sub = grp.require_group(key)
            _write_dict(sub, val, cmp)

        elif isinstance(val, list):
            if not val:
                continue
            if isinstance(val[0], dict):
                # List of records → columnar datasets in a sub-group
                sub = grp.require_group(key)
                _write_records(sub, val, cmp)
            else:
                # Homogeneous list → 1-D dataset
                _write_array(grp, key, val, cmp)

        elif isinstance(val, np.ndarray):
            _write_dataset(grp, key, val, cmp)

        elif isinstance(val, (str, int, float, bool, np.integer, np.floating)):
            grp.attrs[key] = _scalar(val)

        # else: skip


def _write_list(grp, lst: list, cmp: dict) -> None:
    """Write a top-level list.  If list-of-dicts → columnar; else 1-D array."""
    if not lst:
        return
    if isinstance(lst[0], dict):
        _write_records(grp, lst, cmp)
    else:
        _write_array(grp, "values", lst, cmp)


def _write_records(grp, records: list, cmp: dict) -> None:
    """Columnar storage: one dataset per field, all with the same length."""
    if not records:
        return

    # Collect all field names in insertion order
    fields: dict = {}
    for rec in records:
        for k in rec:
            if k not in fields:
                fields[k] = []

    for rec in records:
        for k in fields:
            fields[k].append(rec.get(k))

    for k, col in fields.items():
        _write_array(grp, k, col, cmp)


def _write_array(grp, name: str, lst: list, cmp: dict) -> None:
    """Convert a list to an appropriate numpy array and write as a dataset."""
    # Filter None → masked or string-safe
    clean = [v for v in lst if v is not None]
    if not clean:
        return

    # Detect type
    if all(isinstance(v, bool) for v in clean):
        arr = np.array([str(v) for v in lst], dtype=h5py_string_dtype())
        grp.create_dataset(name, data=arr)
        return

    if all(isinstance(v, (int, np.integer)) for v in clean):
        # Fill None with a sentinel
        arr = np.array([v if v is not None else -9999 for v in lst], dtype=np.int64)
        ds = grp.create_dataset(name, data=arr, **cmp)
        ds.attrs["none_sentinel"] = -9999
        return

    if all(isinstance(v, (int, float, np.integer, np.floating)) for v in clean):
        arr = np.array([v if v is not None else np.nan for v in lst], dtype=np.float64)
        grp.create_dataset(name, data=arr, **cmp)
        return

    # Mixed or string → variable-length UTF-8 strings
    str_arr = np.array([str(v) if v is not None else "" for v in lst],
                       dtype=h5py_string_dtype())
    grp.create_dataset(name, data=str_arr)


def _write_dataset(grp, name: str, arr: np.ndarray, cmp: dict) -> None:
    grp.create_dataset(name, data=arr, **cmp)


def _scalar(val):
    """Convert a Python scalar to an HDF5-safe type."""
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        return float(val)
    return val


_h5py_str_dtype = None

def h5py_string_dtype():
    """Return h5py variable-length UTF-8 string dtype (cached)."""
    global _h5py_str_dtype
    if _h5py_str_dtype is None:
        import h5py
        _h5py_str_dtype = h5py.string_dtype(encoding="utf-8")
    return _h5py_str_dtype
