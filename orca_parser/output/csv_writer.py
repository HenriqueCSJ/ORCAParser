"""CSV output writer for ORCA parser results.

The top-level writer is intentionally thin. Common CSV exports are discovered
through ``csv_section_registry`` and family-specific exports come from the
calculation-family registry. That keeps this module focused on orchestration
instead of owning a growing list of section-specific branches.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

from ..job_family_registry import get_calculation_family_plugin as _get_calculation_family_plugin
from ..job_snapshot import get_job_snapshot as _get_job_snapshot
from .csv_section_registry import iter_csv_section_plugins as _iter_csv_section_plugins


def _stem(data: Dict[str, Any]) -> str:
    """Return the stable output stem for one parsed job."""
    snapshot = _get_job_snapshot(data)
    job_name = snapshot.get("job_name")
    if job_name:
        return str(job_name)

    meta = data.get("metadata", {})
    job_name = meta.get("job_name")
    if job_name:
        return job_name

    source = data.get("source_file", "orca")
    return Path(source).stem


def _write_csv(
    directory: Path,
    filename: str,
    rows: List[Dict[str, Any]],
    fieldnames: List[str],
) -> Path:
    """Write a list of dictionaries to ``directory / filename``."""
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / filename
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return output


def write_csvs(data: Dict[str, Any], directory: Path) -> List[Path]:
    """Write all CSV exports for one parsed ORCA job."""
    directory = Path(directory)
    stem = _stem(data)
    written: List[Path] = []

    # Common exports are fully registry-driven now, so new sections can be
    # registered without extending this orchestrator.
    for plugin in _iter_csv_section_plugins():
        try:
            written.extend(plugin.render_files(data, directory, stem, _write_csv))
        except Exception:  # noqa: BLE001
            pass

    # Family-specific exports remain a separate registry seam because they are
    # conditional on the normalized calculation family, not on generic sections.
    family_plugin = _get_calculation_family_plugin(data)
    for writer in family_plugin.csv_writers:
        try:
            written.extend(writer(data, directory, stem, _write_csv))
        except Exception:  # noqa: BLE001
            pass

    return written
