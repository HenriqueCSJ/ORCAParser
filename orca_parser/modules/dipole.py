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


class DipoleMomentModule(BaseModule):
    """Parses the DIPOLE MOMENT and rotational spectrum sections."""

    name = "dipole"

    def parse(self, lines):
        idx = self.find_line(lines, "DIPOLE MOMENT")
        if idx == -1:
            return None

        data: Dict[str, Any] = {}

        # Scan the block (~25 lines)
        for ln in lines[idx: idx + 30]:
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
