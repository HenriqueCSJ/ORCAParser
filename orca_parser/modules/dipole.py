"""
Dipole moment and rotational spectrum module.

Extracts:
  - Dipole moment vector (electronic + nuclear + total) in a.u. and Debye
  - Magnitude (a.u. and Debye)
  - Rotational constants (cm⁻¹ and MHz)
  - Dipole components along rotational axes
"""

import re
from typing import Any, Dict, Optional

from .base import BaseModule


AU_TO_DEBYE = 2.541746473


class DipoleMomentModule(BaseModule):
    """Parses the DIPOLE MOMENT and rotational spectrum sections."""

    name = "dipole"
    _SECTION_HEADER_RE = re.compile(r"^\s*DIPOLE MOMENT\s*$", re.I)

    @staticmethod
    def _to_debye(vector: Dict[str, float]) -> Dict[str, float]:
        return {
            axis: float(value) * AU_TO_DEBYE
            for axis, value in vector.items()
        }

    def parse(self, lines):
        header_indices = [
            i for i, line in enumerate(lines)
            if self._SECTION_HEADER_RE.match(line)
        ]
        if not header_indices:
            return None
        idx = header_indices[-1]

        data: Dict[str, Any] = {}

        # Scan the property block itself.
        for ln in lines[idx: idx + 40]:
            m = re.search(
                r"Electronic contribution:\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", ln
            )
            if m:
                data["electronic_contribution_au"] = {
                    "x": float(m.group(1)),
                    "y": float(m.group(2)),
                    "z": float(m.group(3)),
                }
            m = re.search(
                r"Nuclear contribution\s+:\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", ln
            )
            if m:
                data["nuclear_contribution_au"] = {
                    "x": float(m.group(1)),
                    "y": float(m.group(2)),
                    "z": float(m.group(3)),
                }
            m = re.search(
                r"Total Dipole Moment\s+:\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", ln
            )
            if m:
                data["total_dipole_au"] = {
                    "x": float(m.group(1)),
                    "y": float(m.group(2)),
                    "z": float(m.group(3)),
                }
            m = re.search(r"Magnitude \(a\.u\.\)\s+:\s+([-\d.]+)", ln)
            if m:
                data["magnitude_au"] = float(m.group(1))
            m = re.search(r"Magnitude \(Debye\)\s+:\s+([-\d.]+)", ln)
            if m:
                data["magnitude_Debye"] = float(m.group(1))

        for key in (
            "electronic_contribution_au",
            "nuclear_contribution_au",
            "total_dipole_au",
        ):
            vector = data.get(key)
            if vector:
                data[key.replace("_au", "_Debye")] = self._to_debye(vector)

        # Rotational spectrum (follows the dipole block)
        idx_rot = self.find_line(lines, "Rotational spectrum", idx)
        if idx_rot != -1:
            for ln in lines[idx_rot: idx_rot + 10]:
                m = re.search(
                    r"Rotational constants in cm-1:\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)",
                    ln,
                )
                if m:
                    data["rotational_constants_cm1"] = [
                        float(m.group(1)),
                        float(m.group(2)),
                        float(m.group(3)),
                    ]
                m = re.search(
                    r"Rotational constants in MHz\s+:\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)",
                    ln,
                )
                if m:
                    data["rotational_constants_MHz"] = [
                        float(m.group(1)),
                        float(m.group(2)),
                        float(m.group(3)),
                    ]
                m = re.search(
                    r"x,y,z \[a\.u\.\]\s+:\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", ln
                )
                if m:
                    data["dipole_rot_axes_au"] = {
                        "x": float(m.group(1)),
                        "y": float(m.group(2)),
                        "z": float(m.group(3)),
                    }
                m = re.search(
                    r"x,y,z \[Debye\]\s+:\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", ln
                )
                if m:
                    data["dipole_rot_axes_Debye"] = {
                        "x": float(m.group(1)),
                        "y": float(m.group(2)),
                        "z": float(m.group(3)),
                    }

        return data if data else None
