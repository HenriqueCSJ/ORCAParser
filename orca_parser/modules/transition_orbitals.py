"""Shared ORCA natural-transition-orbital parsing helpers.

Natural transition orbital (NTO) tables are printed by more than one ORCA
driver.  TDDFT/CIS and STEOM-CCSD use the same row grammar, so this module owns
that small shared language and lets method-specific modules add their own
context around the parsed states.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


_FLOAT = r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?"
_NTO_HEADER_RE = re.compile(
    r"^\s*NATURAL TRANSITION ORBITALS FOR STATE\s+(?P<state>\d+)\s*$",
    re.I,
)
_NTO_FILE_RE = re.compile(
    r"Natural Transition Orbitals were saved in\s+(\S+)",
    re.I,
)
_NTO_THRESHOLD_RE = re.compile(
    rf"Threshold for printing occupation numbers\s+({_FLOAT})",
    re.I,
)
_NTO_ENERGY_RE = re.compile(
    rf"^\s*E=\s*(?P<energy_au>{_FLOAT})\s+au"
    rf"\s+(?P<energy_ev>{_FLOAT})\s+eV"
    rf"\s+(?P<energy_cm1>{_FLOAT})\s+cm\*\*-1",
    re.I,
)
_NTO_PAIR_RE = re.compile(
    rf"^\s*(?P<from_orbital>\d+[A-Za-z]+)\s*->\s*"
    rf"(?P<to_orbital>\d+[A-Za-z]+)\s*:\s*n=\s*"
    rf"(?P<occupation>{_FLOAT})",
    re.I,
)


def parse_natural_transition_orbitals(
    lines: Sequence[str],
    start: int = 0,
    end: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Parse ORCA ``NATURAL TRANSITION ORBITALS FOR STATE`` blocks.

    The returned dictionaries contain only structured NTO metadata and pair
    rows; they deliberately do not retain raw ORCA text.
    """

    limit = len(lines) if end is None else min(end, len(lines))
    states: List[Dict[str, Any]] = []

    i = max(start, 0)
    while i < limit:
        header_match = _NTO_HEADER_RE.match(lines[i].strip())
        if not header_match:
            i += 1
            continue

        state_data: Dict[str, Any] = {
            "state": int(header_match.group("state")),
            "pairs": [],
        }

        j = i + 1
        while j < limit:
            line = lines[j]
            stripped = line.strip()

            if j != i and _NTO_HEADER_RE.match(stripped):
                break
            if _is_nto_boundary(stripped):
                break

            file_match = _NTO_FILE_RE.search(line)
            if file_match:
                state_data["output_file"] = file_match.group(1)

            threshold_match = _NTO_THRESHOLD_RE.search(line)
            if threshold_match:
                state_data["print_threshold"] = float(threshold_match.group(1))

            energy_match = _NTO_ENERGY_RE.match(line)
            if energy_match:
                energy_cm1 = float(energy_match.group("energy_cm1"))
                state_data["energy_au"] = float(energy_match.group("energy_au"))
                state_data["energy_eV"] = float(energy_match.group("energy_ev"))
                state_data["energy_cm1"] = energy_cm1
                if energy_cm1:
                    state_data["wavelength_nm"] = 1.0e7 / energy_cm1

            pair = parse_nto_pair(line)
            if pair is not None:
                state_data["pairs"].append(pair)

            j += 1

        state_data["pair_count"] = len(state_data["pairs"])
        states.append(state_data)
        i = j

    return states


def parse_nto_pair(line: str) -> Optional[Dict[str, Any]]:
    """Parse one NTO pair row such as ``23a -> 24a : n=0.998``."""

    pair_match = _NTO_PAIR_RE.match(line)
    if pair_match is None:
        return None

    pair = build_orbital_pair_dict(
        pair_match.group("from_orbital"),
        pair_match.group("to_orbital"),
    )
    pair["occupation"] = float(pair_match.group("occupation"))
    return pair


def build_orbital_pair_dict(from_orbital: str, to_orbital: str) -> Dict[str, Any]:
    """Return a normalized donor/acceptor orbital-pair dictionary."""

    from_index, from_spin = split_orbital_label(from_orbital)
    to_index, to_spin = split_orbital_label(to_orbital)
    return {
        "from_orbital": from_orbital,
        "to_orbital": to_orbital,
        "from_index": from_index,
        "from_spin": from_spin,
        "to_index": to_index,
        "to_spin": to_spin,
    }


def split_orbital_label(label: str) -> Tuple[Optional[int], Optional[str]]:
    """Split labels such as ``23a`` into ``(23, "a")``."""

    match = re.match(r"(?P<index>\d+)(?P<spin>[A-Za-z]+)$", label.strip())
    if not match:
        return None, None
    return int(match.group("index")), match.group("spin").lower()


def _is_nto_boundary(stripped: str) -> bool:
    upper = stripped.upper()
    return (
        "ABSORPTION SPECTRUM VIA TRANSITION" in upper
        or "CD SPECTRUM VIA TRANSITION" in upper
        or "CIS/TD-DFT TOTAL ENERGY" in upper
        or upper.startswith("TD-DFT/TDA-EXCITATION SPECTRA")
        or upper.startswith("FINAL SINGLE POINT ENERGY")
        or upper.startswith("ORCA POPULATION ANALYSIS")
        or upper.startswith("TIMINGS")
    )
