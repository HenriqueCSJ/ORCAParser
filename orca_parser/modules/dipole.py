"""
Dipole moment and rotational spectrum module.

Extracts:
  - Dipole moment vector (electronic + nuclear + total) in a.u. and Debye
  - Magnitude (a.u. and Debye)
  - Rotational constants (cm⁻¹ and MHz)
  - Dipole components along rotational axes
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..output.csv_section_registry import DIPOLE_CSV_SECTION_PLUGIN
from ..output.markdown_section_registry import DIPOLE_MARKDOWN_SECTION_PLUGIN
from ..parser_section_plugin import ParserSectionAlias, ParserSectionPlugin
from ..plugin_bundle import PluginBundle, PluginMetadata
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
        blocks = parse_dipole_moment_blocks(lines)
        if not blocks:
            return None
        data = dict(blocks[-1])
        if len(blocks) > 1:
            data["all_blocks"] = blocks
        return data if data else None


def _to_debye(vector: Dict[str, float]) -> Dict[str, float]:
    return {
        axis: float(value) * AU_TO_DEBYE
        for axis, value in vector.items()
    }


def parse_dipole_moment_blocks(lines: List[str]) -> List[Dict[str, Any]]:
    """Parse every ORCA ``DIPOLE MOMENT`` block with method/density context."""

    header_re = DipoleMomentModule._SECTION_HEADER_RE
    header_indices = [
        i for i, line in enumerate(lines)
        if header_re.match(line)
    ]
    blocks: List[Dict[str, Any]] = []
    if not header_indices:
        return blocks

    for block_number, idx in enumerate(header_indices, start=1):
        end = header_indices[block_number] if block_number < len(header_indices) else len(lines)
        data: Dict[str, Any] = {"block_number": block_number, "line": idx + 1}

        for ln in lines[idx:end]:
            m = re.search(r"Method\s+:\s+(.+)", ln)
            if m:
                data["method"] = m.group(1).strip()
            m = re.search(r"Type of density\s+:\s+(.+)", ln)
            if m:
                data["density_type"] = m.group(1).strip()
            m = re.search(r"Level\s+:\s+(.+)", ln)
            if m:
                data["level"] = m.group(1).strip()
            m = re.search(r"Multiplicity\s+:\s+(\d+)", ln)
            if m:
                data["multiplicity"] = int(m.group(1))
            m = re.search(r"Irrep\s+:\s+(\d+)", ln)
            if m:
                data["irrep"] = int(m.group(1))
            m = re.search(r"Energy\s+:\s+([-\d.]+)", ln)
            if m:
                data["energy_Eh"] = float(m.group(1))
            m = re.search(r"Basis\s+:\s+(.+)", ln)
            if m:
                data["basis"] = m.group(1).strip()
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
                data[key.replace("_au", "_Debye")] = _to_debye(vector)

        idx_rot = -1
        for j in range(idx, end):
            if "Rotational spectrum" in lines[j]:
                idx_rot = j
                break
        if idx_rot != -1:
            for ln in lines[idx_rot:min(idx_rot + 10, end)]:
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
        blocks.append(data)

    return blocks


PLUGIN_BUNDLE = PluginBundle(
    metadata=PluginMetadata(
        key="dipole_section",
        name="Dipole Section",
        short_help="Built-in dipole parser section owned by dipole.py.",
        description=(
            "Self-registering built-in parser section for dipole moments and "
            "rotational-spectrum dipole metadata."
        ),
        docs_path="README.md",
        examples=(
            "orca_parser job.out --sections dipole",
        ),
    ),
    parser_sections=(
        ParserSectionPlugin("dipole", DipoleMomentModule),
    ),
    parser_aliases=(
        ParserSectionAlias(name="dipole", section_keys=("dipole",)),
    ),
    markdown_sections=(
        DIPOLE_MARKDOWN_SECTION_PLUGIN,
    ),
    csv_sections=(
        DIPOLE_CSV_SECTION_PLUGIN,
    ),
)
