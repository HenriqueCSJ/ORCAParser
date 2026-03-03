"""JSON output writer for ORCA parser results."""

import gzip
import json
from pathlib import Path
from typing import Any, Dict, Optional


class _NumpySafeEncoder(json.JSONEncoder):
    """Handles types that default JSON encoder can't serialize."""

    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        return super().default(obj)


def _strip_none(obj: Any) -> Any:
    """Recursively remove keys with None values from dicts."""
    if isinstance(obj, dict):
        return {k: _strip_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_none(v) for v in obj]
    return obj


def write_json(
    data: Dict[str, Any],
    path: Path,
    indent: Optional[int] = 2,
    strip_none: bool = False,
    compress: bool = False,
) -> Path:
    """Write *data* as a JSON file to *path*.

    Parameters
    ----------
    data : dict
        Parsed ORCA output dictionary.
    path : Path
        Destination file path. If *compress* is True and the path doesn't
        end with ``.gz``, the suffix is appended automatically.
    indent : int or None
        Indentation level. ``None`` or ``0`` → compact (no newlines).
    strip_none : bool
        Remove keys with ``None`` values recursively before serialising.
    compress : bool
        Write gzip-compressed JSON (``.json.gz``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if compress and not path.name.endswith(".gz"):
        path = path.with_suffix(path.suffix + ".gz")

    payload = _strip_none(data) if strip_none else data
    _indent = indent if indent else None  # treat 0 same as None

    serialised = json.dumps(
        payload,
        indent=_indent,
        cls=_NumpySafeEncoder,
        ensure_ascii=False,
    ).encode("utf-8")

    if compress:
        with gzip.open(path, "wb", compresslevel=6) as fh:
            fh.write(serialised)
    else:
        with open(path, "wb") as fh:
            fh.write(serialised)

    return path
