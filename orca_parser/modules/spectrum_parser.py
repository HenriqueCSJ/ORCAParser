"""Shared ORCA UV/CD spectrum table parser.

ORCA prints the same absorption and CD spectrum tables from several drivers:
TDDFT/CIS, CASSCF, NEVPT2-corrected CASSCF, and SOC-corrected QDPT blocks.
This module owns the common table grammar so method-specific modules can add
their own context without reimplementing row parsing.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


SPECTRUM_TABLES: Dict[str, Tuple[str, List[str]]] = {
    "ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS": (
        "absorption_electric_dipole",
        [
            "oscillator_strength",
            "dipole_strength_au2",
            "dx_au",
            "dy_au",
            "dz_au",
        ],
    ),
    "ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS": (
        "absorption_velocity_dipole",
        [
            "oscillator_strength",
            "velocity_strength_au2",
            "px_au",
            "py_au",
            "pz_au",
        ],
    ),
    "CD SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS": (
        "cd_electric_dipole",
        [
            "rotatory_strength_cgs",
            "mx_au",
            "my_au",
            "mz_au",
        ],
    ),
    "CD SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS": (
        "cd_velocity_dipole",
        [
            "rotatory_strength_cgs",
            "mx_au",
            "my_au",
            "mz_au",
        ],
    ),
}


@dataclass(frozen=True)
class SpectrumTableInfo:
    """Metadata for a recognized ORCA spectrum table heading."""

    mode: str
    kind: str
    value_fields: List[str]
    title: str
    normalized_title: str
    soc_corrected: bool = False


def spectrum_table_info(line: str, *, allow_soc_prefix: bool = True) -> Optional[SpectrumTableInfo]:
    """Return metadata when *line* is a supported ORCA spectrum table heading."""
    title = line.strip()
    if not title:
        return None

    normalized = title.upper()
    soc_corrected = False
    if allow_soc_prefix and normalized.startswith("SOC CORRECTED "):
        normalized = normalized[len("SOC CORRECTED ") :]
        soc_corrected = True

    if normalized not in SPECTRUM_TABLES:
        return None

    kind, value_fields = SPECTRUM_TABLES[normalized]
    if kind.startswith("absorption"):
        mode = "absorption"
    elif kind.startswith("cd"):
        mode = "cd"
    else:
        return None

    return SpectrumTableInfo(
        mode=mode,
        kind=kind,
        value_fields=list(value_fields),
        title=title,
        normalized_title=normalized,
        soc_corrected=soc_corrected,
    )


def parse_spectrum_table(
    lines: Sequence[str],
    start: int,
    end: Optional[int] = None,
    *,
    stop_phrases: Sequence[str] = (),
    allow_soc_prefix: bool = True,
) -> Tuple[Optional[Dict[str, Any]], int]:
    """Parse one ORCA spectrum table beginning at *start*.

    Returns ``(table, next_index)``. ``table`` is ``None`` when *start* is not a
    recognized spectrum heading or when no transition rows were found.
    """
    info = spectrum_table_info(lines[start], allow_soc_prefix=allow_soc_prefix)
    if info is None:
        return None, start + 1

    upper_stop_phrases = tuple(phrase.upper() for phrase in stop_phrases)
    limit = len(lines) if end is None else min(end, len(lines))
    table: Dict[str, Any] = {
        "kind": info.kind,
        "mode": info.mode,
        "title": info.title,
        "transitions": [],
    }
    if info.soc_corrected:
        table["soc_corrected"] = True

    index = start + 1
    started = False
    while index < limit:
        stripped = lines[index].strip()
        upper = stripped.upper()

        if not stripped:
            if started:
                break
            index += 1
            continue

        if spectrum_table_info(stripped, allow_soc_prefix=allow_soc_prefix) is not None:
            break
        if upper_stop_phrases and any(phrase in upper for phrase in upper_stop_phrases):
            break

        row = parse_spectrum_row(lines[index], info.value_fields)
        if row is not None:
            table["transitions"].append(row)
            started = True

        index += 1

    if not table["transitions"]:
        return None, index

    table["transition_count"] = len(table["transitions"])
    return table, index


def parse_spectrum_row(line: str, value_fields: Sequence[str]) -> Optional[Dict[str, Any]]:
    """Parse one transition row from a supported ORCA spectrum table."""
    match = re.match(
        r"^\s*(?P<from_state>\S+)\s*->\s*(?P<to_state>\S+)\s+(?P<rest>.+)$",
        line,
    )
    if not match:
        return None

    parts = match.group("rest").split()
    expected = 3 + len(value_fields)
    if len(parts) != expected:
        return None

    values = [_safe_float(part) for part in parts]
    if any(value is None for value in values):
        return None

    row: Dict[str, Any] = {
        "from_state_label": match.group("from_state"),
        "to_state_label": match.group("to_state"),
        "energy_eV": float(values[0]),
        "energy_cm1": float(values[1]),
        "wavelength_nm": float(values[2]),
    }
    for field, value in zip(value_fields, values[3:]):
        row[field] = float(value)

    from_root, from_label = split_state_label(row["from_state_label"])
    to_root, to_label = split_state_label(row["to_state_label"])
    if from_root is not None:
        row["from_root"] = from_root
    if from_label is not None:
        row["from_state_suffix"] = from_label
    if to_root is not None:
        row["to_root"] = to_root
    if to_label is not None:
        row["to_state_suffix"] = to_label

    return row


def split_state_label(label: str) -> Tuple[Optional[int], Optional[str]]:
    """Split ORCA transition labels such as ``0-4A`` or ``0-4.0A``."""
    match = re.match(r"(?P<root>\d+)-(?P<label>.+)$", label.strip())
    if not match:
        return None, None
    return int(match.group("root")), match.group("label")


def _safe_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
