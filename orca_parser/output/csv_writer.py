"""CSV output writer for ORCA parser results.

The top-level writer is intentionally thin. Common CSV exports are discovered
through ``csv_section_registry`` and family-specific exports come from the
calculation-family registry. That keeps this module focused on orchestration
instead of owning a growing list of section-specific branches.
"""

from __future__ import annotations

import csv
import warnings
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


def _warn_csv_export_failure(origin: str, exc: Exception) -> None:
    """Surface a non-fatal CSV export failure to the caller.

    CSV export is intentionally best-effort so one broken section does not block
    the rest of the job. We still warn loudly because silently swallowing those
    errors makes it very hard to notice that a partial export occurred.
    """

    warnings.warn(
        f"Skipping CSV export from {origin}: {exc}",
        RuntimeWarning,
        stacklevel=2,
    )


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
        except Exception as exc:  # noqa: BLE001
            _warn_csv_export_failure(f"CSV section plugin '{plugin.key}'", exc)

    # Family-specific exports remain a separate registry seam because they are
    # conditional on the normalized calculation family, not on generic sections.
    family_plugin = _get_calculation_family_plugin(data)
    for writer in family_plugin.csv_writers:
        try:
            written.extend(writer(data, directory, stem, _write_csv))
        except Exception as exc:  # noqa: BLE001
            writer_name = getattr(writer, "__name__", repr(writer))
            _warn_csv_export_failure(
                f"calculation family '{family_plugin.family}' writer '{writer_name}'",
                exc,
            )

    return written
