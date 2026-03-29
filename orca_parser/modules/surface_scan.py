"""
Relaxed surface scan parser module.

Extracts scan coordinate definitions, per-step values, surface energies, and
related sidecar files written by ORCA (``relaxscan*.dat`` / ``allxyz``).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseModule


_SCAN_HEADER_RE = re.compile(r"Relaxed Surface Scan", re.I)
_PARAM_COUNT_RE = re.compile(
    r"There is\s+(\d+)\s+parameter(?:s)?\s+to be scanned\.",
    re.I,
)
_OPT_COUNT_RE = re.compile(
    r"There will be\s+(\d+)\s+constrained geometry optimizations\.",
    re.I,
)
_HEADER_PARAM_RE = re.compile(
    r"^\s*(Bond|Angle|Dihedral)\s*\(([^)]+)\):\s*range=\s*"
    r"(-?\d+(?:\.\d+)?)\s*\.\.\s*(-?\d+(?:\.\d+)?)\s*steps\s*=\s*(\d+)",
    re.I,
)
_STEP_RE = re.compile(r"RELAXED SURFACE SCAN STEP\s+(\d+)", re.I)
_STEP_VALUE_RE = re.compile(
    r"^\s*\*\s*(Bond|Angle|Dihedral)\s*\(([^)]+)\)\s*:\s*"
    r"(-?\d+(?:\.\d+)?)",
    re.I,
)
_SURFACE_BLOCK_RE = re.compile(
    r"^The Calculated Surface using the '([^']+)'",
    re.I,
)
_COLUMN_RE = re.compile(r"Column\s+(\d+):\s*(.+)", re.I)
_STORED_XYZ_RE = re.compile(
    r"Storing optimized geometry in\s+(\S+)\s+\.\.\.\s+done",
    re.I,
)


def _coord_kind_from_name(name: str) -> str:
    low = name.lower()
    if low.startswith("bond"):
        return "B"
    if low.startswith("angle"):
        return "A"
    if low.startswith("dihedral"):
        return "D"
    return name[:1].upper()


def _coord_name_from_kind(kind: str) -> str:
    return {
        "B": "Bond",
        "A": "Angle",
        "D": "Dihedral",
    }.get(kind.upper(), kind.upper())


def _coord_unit(kind: str) -> str:
    return "A" if kind.upper() == "B" else "deg"


def _coord_label(kind: str, atoms: List[int]) -> str:
    joined = ",".join(str(atom) for atom in atoms)
    return f"{kind.upper()}({joined})"


def _parse_atoms(raw_atoms: str) -> List[int]:
    atoms: List[int] = []
    for token in raw_atoms.split(","):
        token = token.strip()
        if token:
            atoms.append(int(token))
    return atoms


def _safe_float(token: str) -> Optional[float]:
    try:
        return float(token)
    except ValueError:
        return None


def _parse_bool_value(value: str) -> Optional[bool]:
    low = value.strip().lower()
    if low in {"true", "yes", "on"}:
        return True
    if low in {"false", "no", "off"}:
        return False
    return None


def _parse_scan_input_block(lines: List[str]) -> Dict[str, Any]:
    in_geom = False
    in_scan = False
    simultaneous_scan: Optional[bool] = None
    parameters: List[Dict[str, Any]] = []

    for line in lines:
        if "****END OF INPUT****" in line:
            break
        m = re.match(r"^\|\s*\d+>\s*(.*)$", line)
        if not m:
            continue
        content = m.group(1).strip()
        if not content:
            continue

        low = content.lower()
        if low.startswith("%geom"):
            in_geom = True
            continue
        if not in_geom:
            continue
        if low == "end":
            if in_scan:
                in_scan = False
                continue
            in_geom = False
            continue
        if low == "scan":
            in_scan = True
            continue
        if low.startswith("simul_scan"):
            parts = content.split(None, 1)
            if len(parts) == 2:
                simultaneous_scan = _parse_bool_value(parts[1])
            continue
        if not in_scan:
            continue

        parsed = _parse_scan_definition_line(content)
        if parsed:
            parsed["source"] = "input"
            parameters.append(parsed)

    return {
        "simultaneous_scan": simultaneous_scan,
        "parameters": parameters,
    }


def _parse_scan_definition_line(content: str) -> Optional[Dict[str, Any]]:
    # Range syntax: D 5 2 0 4 = -180, 360, 37
    m = re.match(
        r"^\s*([BAD])\s+([\d\s]+?)\s*=\s*"
        r"(-?\d+(?:\.\d+)?)\s*,\s*"
        r"(-?\d+(?:\.\d+)?)\s*,\s*(\d+)\s*$",
        content,
        re.I,
    )
    if m:
        kind = m.group(1).upper()
        atoms = [int(tok) for tok in m.group(2).split() if tok.strip()]
        return {
            "kind": kind,
            "coordinate_type": _coord_name_from_kind(kind),
            "atoms": atoms,
            "label": _coord_label(kind, atoms),
            "unit": _coord_unit(kind),
            "mode": "range",
            "start": float(m.group(3)),
            "end": float(m.group(4)),
            "steps": int(m.group(5)),
        }

    # Explicit values syntax: B 1 2 [1.2 1.4 1.6]
    m = re.match(
        r"^\s*([BAD])\s+([\d\s]+?)\s*\[\s*(.+?)\s*\]\s*$",
        content,
        re.I,
    )
    if m:
        kind = m.group(1).upper()
        atoms = [int(tok) for tok in m.group(2).split() if tok.strip()]
        values = [
            float(tok)
            for tok in re.split(r"[\s,]+", m.group(3).strip())
            if tok.strip()
        ]
        return {
            "kind": kind,
            "coordinate_type": _coord_name_from_kind(kind),
            "atoms": atoms,
            "label": _coord_label(kind, atoms),
            "unit": _coord_unit(kind),
            "mode": "values",
            "values": values,
            "steps": len(values),
        }

    return None


def _parse_header_parameters(lines: List[str], header_idx: int) -> List[Dict[str, Any]]:
    parameters: List[Dict[str, Any]] = []
    for line in lines[header_idx: min(header_idx + 40, len(lines))]:
        m = _HEADER_PARAM_RE.match(line)
        if not m:
            continue
        coord_type = m.group(1).capitalize()
        kind = _coord_kind_from_name(coord_type)
        atoms = _parse_atoms(m.group(2))
        parameters.append({
            "kind": kind,
            "coordinate_type": coord_type,
            "atoms": atoms,
            "label": _coord_label(kind, atoms),
            "unit": _coord_unit(kind),
            "mode": "range",
            "start": float(m.group(3)),
            "end": float(m.group(4)),
            "steps": int(m.group(5)),
            "source": "output_header",
        })
    return parameters


def _parse_step_targets(lines: List[str]) -> List[Dict[str, Any]]:
    steps: List[Dict[str, Any]] = []
    i = 0
    while i < len(lines):
        m = _STEP_RE.search(lines[i])
        if not m:
            i += 1
            continue
        step_number = int(m.group(1))
        coordinate_values: List[float] = []
        coordinate_labels: List[str] = []
        j = i + 1
        while j < min(i + 20, len(lines)):
            mv = _STEP_VALUE_RE.match(lines[j])
            if not mv:
                j += 1
                continue
            kind = _coord_kind_from_name(mv.group(1))
            atoms = _parse_atoms(mv.group(2))
            coordinate_labels.append(_coord_label(kind, atoms))
            coordinate_values.append(float(mv.group(3)))
            j += 1
        steps.append({
            "step": step_number,
            "coordinate_labels": coordinate_labels,
            "coordinate_values": coordinate_values,
        })
        i = j
    return steps


def _parse_surface_blocks(lines: List[str], n_parameters: int) -> Dict[str, List[Dict[str, Any]]]:
    surfaces: Dict[str, List[Dict[str, Any]]] = {}
    i = 0
    while i < len(lines):
        m = _SURFACE_BLOCK_RE.match(lines[i].strip())
        if not m:
            i += 1
            continue
        key = m.group(1).strip().lower().replace(" ", "_")
        rows: List[Dict[str, Any]] = []
        j = i + 1
        while j < len(lines):
            stripped = lines[j].strip()
            if not stripped:
                break
            if _SURFACE_BLOCK_RE.match(stripped):
                break
            if stripped.startswith("-") and set(stripped) == {"-"}:
                break
            parts = stripped.split()
            if len(parts) < n_parameters + 1:
                j += 1
                continue
            values = [_safe_float(tok) for tok in parts[: n_parameters + 1]]
            if any(value is None for value in values):
                j += 1
                continue
            rows.append({
                "coordinate_values": [float(v) for v in values[:-1]],
                "energy_Eh": float(values[-1]),
            })
            j += 1
        surfaces[key] = rows
        i = j
    return surfaces


def _parse_relaxscan_dat(path: Path, n_parameters: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return rows

    for line in content:
        parts = line.split()
        if len(parts) < n_parameters + 1:
            continue
        values = [_safe_float(tok) for tok in parts[: n_parameters + 1]]
        if any(value is None for value in values):
            continue
        rows.append({
            "coordinate_values": [float(v) for v in values[:-1]],
            "energy_Eh": float(values[-1]),
        })
    return rows


def _count_xyz_frames(path: Path) -> Optional[int]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None

    i = 0
    count = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped or stripped == ">":
            i += 1
            continue
        try:
            n_atoms = int(stripped)
        except ValueError:
            break
        if i + n_atoms + 1 >= len(lines):
            break
        count += 1
        i += n_atoms + 2
    return count or None


def _detect_sidecar_files(source_path: Path) -> Dict[str, Any]:
    base = source_path.with_suffix("")
    candidates = {
        "actual_surface_dat": base.with_suffix(".relaxscanact.dat"),
        "scf_surface_dat": base.with_suffix(".relaxscanscf.dat"),
        "allxyz": base.with_suffix(".allxyz"),
        "xyzall": base.with_suffix(".xyzall"),
        "trajectory_xyz": source_path.with_name(f"{base.name}_trj.xyz"),
    }

    sidecars: Dict[str, Any] = {}
    for key, candidate in candidates.items():
        if candidate.exists():
            sidecars[key] = str(candidate)

    allxyz_path = None
    if "allxyz" in sidecars:
        allxyz_path = Path(sidecars["allxyz"])
    elif "xyzall" in sidecars:
        allxyz_path = Path(sidecars["xyzall"])
    if allxyz_path is not None:
        frame_count = _count_xyz_frames(allxyz_path)
        if frame_count is not None:
            sidecars["allxyz_frame_count"] = frame_count

    return sidecars


def _merge_step_data(
    parameter_labels: List[str],
    step_targets: List[Dict[str, Any]],
    actual_surface: List[Dict[str, Any]],
    scf_surface: List[Dict[str, Any]],
    optimized_xyz_files: List[str],
) -> List[Dict[str, Any]]:
    count = max(
        len(step_targets),
        len(actual_surface),
        len(scf_surface),
        len(optimized_xyz_files),
    )
    steps: List[Dict[str, Any]] = []
    for idx in range(count):
        row: Dict[str, Any] = {
            "step": idx + 1,
            "coordinate_labels": parameter_labels,
            "coordinate_values": [],
        }
        if idx < len(step_targets):
            row["step"] = step_targets[idx].get("step", idx + 1)
            row["coordinate_values"] = step_targets[idx].get("coordinate_values", [])
        if idx < len(actual_surface):
            row["coordinate_values"] = actual_surface[idx].get(
                "coordinate_values",
                row["coordinate_values"],
            )
            row["actual_energy_Eh"] = actual_surface[idx].get("energy_Eh")
        if idx < len(scf_surface):
            row["coordinate_values"] = scf_surface[idx].get(
                "coordinate_values",
                row["coordinate_values"],
            )
            row["scf_energy_Eh"] = scf_surface[idx].get("energy_Eh")
        if idx < len(optimized_xyz_files):
            row["optimized_xyz_file"] = optimized_xyz_files[idx]
        steps.append(row)

    actual_energies = [
        row["actual_energy_Eh"]
        for row in steps
        if row.get("actual_energy_Eh") is not None
    ]
    if actual_energies:
        minimum = min(actual_energies)
        for row in steps:
            energy = row.get("actual_energy_Eh")
            if energy is not None:
                row["relative_actual_energy_kcal_mol"] = (energy - minimum) * 627.509474

    scf_energies = [
        row["scf_energy_Eh"]
        for row in steps
        if row.get("scf_energy_Eh") is not None
    ]
    if scf_energies:
        minimum = min(scf_energies)
        for row in steps:
            energy = row.get("scf_energy_Eh")
            if energy is not None:
                row["relative_scf_energy_kcal_mol"] = (energy - minimum) * 627.509474

    return steps


class SurfaceScanModule(BaseModule):
    """Parse relaxed surface scans and their summary tables."""

    name = "surface_scan"

    def parse(self, lines: List[str]) -> Optional[Dict[str, Any]]:
        if not any(_SCAN_HEADER_RE.search(line) for line in lines[:5000]):
            return None

        header_idx = self.find_line(lines, "Relaxed Surface Scan")
        input_scan = _parse_scan_input_block(lines)
        header_parameters = _parse_header_parameters(lines, header_idx) if header_idx != -1 else []
        parameters = input_scan.get("parameters") or header_parameters

        n_parameters = len(parameters)
        if not n_parameters:
            m = next((match for line in lines for match in [_PARAM_COUNT_RE.search(line)] if match), None)
            if m:
                n_parameters = int(m.group(1))

        n_constrained_optimizations = None
        for line in lines:
            m = _OPT_COUNT_RE.search(line)
            if m:
                n_constrained_optimizations = int(m.group(1))
                break

        step_targets = _parse_step_targets(lines)
        optimized_xyz_files = [
            m.group(1)
            for line in lines
            for m in [_STORED_XYZ_RE.search(line)]
            if m
        ]

        actual_surface: List[Dict[str, Any]] = []
        scf_surface: List[Dict[str, Any]] = []

        if n_parameters:
            surfaces = _parse_surface_blocks(lines, n_parameters)
            actual_surface = surfaces.get("actual_energy", [])
            scf_surface = surfaces.get("scf_energy", [])

        source_path = Path(self.context.get("source_path", ""))
        sidecars = _detect_sidecar_files(source_path) if source_path else {}

        if n_parameters and not actual_surface and sidecars.get("actual_surface_dat"):
            actual_surface = _parse_relaxscan_dat(Path(sidecars["actual_surface_dat"]), n_parameters)
        if n_parameters and not scf_surface and sidecars.get("scf_surface_dat"):
            scf_surface = _parse_relaxscan_dat(Path(sidecars["scf_surface_dat"]), n_parameters)

        parameter_labels = [parameter["label"] for parameter in parameters]
        steps = _merge_step_data(
            parameter_labels,
            step_targets,
            actual_surface,
            scf_surface,
            optimized_xyz_files,
        )

        if n_constrained_optimizations is None and steps:
            n_constrained_optimizations = len(steps)
        if not n_parameters and steps and steps[0].get("coordinate_values"):
            n_parameters = len(steps[0]["coordinate_values"])

        mode = "single"
        if n_parameters > 1:
            mode = "simultaneous" if input_scan.get("simultaneous_scan") else "nested"

        data: Dict[str, Any] = {
            "n_parameters": n_parameters,
            "n_constrained_optimizations": n_constrained_optimizations,
            "mode": mode,
            "simultaneous_scan": input_scan.get("simultaneous_scan"),
            "parameters": parameters,
            "steps": steps,
            "surface_actual_energy": actual_surface,
            "surface_scf_energy": scf_surface,
            "sidecar_files": sidecars,
        }

        if steps:
            actual_energies = [row.get("actual_energy_Eh") for row in steps if row.get("actual_energy_Eh") is not None]
            if actual_energies:
                data["actual_energy_min_Eh"] = min(actual_energies)
                data["actual_energy_max_Eh"] = max(actual_energies)
                data["actual_energy_span_kcal_mol"] = (
                    (max(actual_energies) - min(actual_energies)) * 627.509474
                )
            scf_energies = [row.get("scf_energy_Eh") for row in steps if row.get("scf_energy_Eh") is not None]
            if scf_energies:
                data["scf_energy_min_Eh"] = min(scf_energies)
                data["scf_energy_max_Eh"] = max(scf_energies)
                data["scf_energy_span_kcal_mol"] = (
                    (max(scf_energies) - min(scf_energies)) * 627.509474
                )

        return data
