"""
Module for DFT EPR calculations.

Extracts:
- Zero-Field Splitting (ZFS): D-tensor, eigenvalues, eigenvectors,
  transition part matrices, individual contributions (D/E in cm^-1)
- Electronic g-tensor: g-matrix, DSO/PSO contributions, breakdown by
  mechanism, orientation, atom/bond analysis
- Hyperfine coupling (HFC): per-nucleus total HFC matrix, SD and
  NOC/SOC contributions, principal components (A_iso, A_dip, A_orb),
  local/nonlocal decomposition, natural spin-orbital contributions
- Electric Field Gradient (EFG): raw EFG matrix, eigenvalues,
  quadrupole coupling parameters (e²qQ, eta)
"""

import re
from typing import Any, Dict, List, Optional

from .base import BaseModule

# ─────────────────────────────────────────────────────────────────────
# Regex helpers
# ─────────────────────────────────────────────────────────────────────

_FLOAT = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?"


def _to_float(s: str) -> float:
    return float(s.replace("D", "E").replace("d", "e"))


def _floats(line: str) -> List[float]:
    """Extract all float tokens from a line."""
    return [_to_float(x) for x in re.findall(_FLOAT, line)]


def _last3(line: str) -> Optional[List[float]]:
    """Return the last three floats on a line, or None."""
    vals = _floats(line)
    return vals[-3:] if len(vals) >= 3 else None


def _parse_matrix3(lines: List[str], start: int) -> Optional[List[List[float]]]:
    """Parse a 3x3 matrix from lines following *start*.

    Skips separator lines (all dashes) and column-index headers (``0 1 2``).
    """
    mat: List[List[float]] = []
    for j in range(start + 1, min(start + 10, len(lines))):
        stripped = lines[j].strip()
        if not stripped or set(stripped) == {"-"}:
            continue
        row = _last3(lines[j])
        if row is None:
            continue
        # Skip column-index header "0  1  2"
        if row == [0.0, 1.0, 2.0]:
            continue
        mat.append(row)
        if len(mat) == 3:
            return mat
    return mat if len(mat) == 3 else None


def _parse_orientation(lines: List[str], start: int) -> Optional[Dict[str, List[float]]]:
    """Parse an orientation block (X/Y/Z rows with 3 direction cosines)."""
    orient: Dict[str, List[float]] = {}
    for j in range(start + 1, min(start + 6, len(lines))):
        m = re.match(
            r"^\s*([XYZxyz])\s+(" + _FLOAT + r")\s+(" + _FLOAT + r")\s+(" + _FLOAT + r")\s*$",
            lines[j].strip(),
        )
        if m:
            orient[m.group(1).upper()] = [
                _to_float(m.group(2)),
                _to_float(m.group(3)),
                _to_float(m.group(4)),
            ]
        if len(orient) == 3:
            return orient
    return orient or None


def _parse_contribution_with_eigs(
    block: List[str], heading_pattern: str,
) -> Optional[Dict[str, Any]]:
    """Parse a contribution sub-block (matrix + eigenvalue contributions)."""
    for i, line in enumerate(block):
        if not re.search(heading_pattern, line, re.I):
            continue
        matrix: List[List[float]] = []
        for j in range(i + 2, min(i + 5, len(block))):
            row = _last3(block[j])
            if row:
                matrix.append(row)
        eigs = None
        for j in range(i + 5, min(i + 10, len(block))):
            if re.search(
                r"Contributions to the eigenvalues|Eigenvalue contributions",
                block[j], re.I,
            ):
                if j + 1 < len(block):
                    eigs = _last3(block[j + 1])
                break
        return {
            "matrix": matrix if len(matrix) == 3 else None,
            "eigenvalue_contributions": eigs,
        }
    return None


# ─────────────────────────────────────────────────────────────────────
# ZFS sub-parser
# ─────────────────────────────────────────────────────────────────────

def _parse_zfs(lines: List[str], start: int, end: int) -> Dict[str, Any]:
    """Parse the ZERO-FIELD SPLITTING section."""
    section = lines[start:end]
    data: Dict[str, Any] = {}

    for idx, line in enumerate(section):
        if re.search(r"raw-matrix\s*:", line, re.I):
            data["raw_matrix"] = _parse_matrix3(section, idx)

        elif re.search(r"diagonalized D matrix\s*:", line, re.I):
            vals = _floats(section[idx + 1]) if idx + 1 < len(section) else []
            if len(vals) >= 3:
                data["eigenvalues"] = vals[:3]
                eigvecs: List[List[float]] = []
                for j in range(idx + 2, min(idx + 5, len(section))):
                    row = _last3(section[j])
                    if row:
                        eigvecs.append(row)
                if len(eigvecs) == 3:
                    data["eigenvectors"] = eigvecs

        elif re.search(r"Tensor is right-handed", line, re.I):
            data["right_handed"] = True

        elif m := re.search(
            r"Direction\s+X=(\d+)\s+Y=(\d+)\s+Z=(\d+)", line
        ):
            data["principal_axis_order"] = {
                "X": int(m.group(1)),
                "Y": int(m.group(2)),
                "Z": int(m.group(3)),
            }

        elif m := re.search(r"^\s*D\s*=\s*(" + _FLOAT + r")\s+cm\*\*-1", line):
            data["D_cm-1"] = _to_float(m.group(1))

        elif m := re.search(r"^\s*E/D\s*=\s*(" + _FLOAT + r")", line):
            data["E_over_D"] = _to_float(m.group(1))

        elif re.match(
            r"^\s*(ALPHA-PART|BETA-PART|ALPHA->BETA-PART|BETA->ALPHA-PART)\s*$",
            line,
        ):
            label = line.strip()
            data.setdefault("transition_part_matrices", {})[label] = (
                _parse_matrix3(section, idx)
            )

        elif re.search(
            r"^\s*Individual contributions \(in cm\*\*-1\)\s*$", line
        ):
            data["individual_contributions"] = _parse_individual_contributions(
                section, idx
            )

    return data


def _parse_individual_contributions(
    lines: List[str], start: int,
) -> Dict[str, Dict[str, float]]:
    """Parse D/E individual contribution table."""
    out: Dict[str, Dict[str, float]] = {}
    for j in range(start + 1, len(lines)):
        line = lines[j]
        if re.search(r"EPR g-tensor done|ELECTRONIC G-MATRIX", line):
            break
        m = re.match(
            r"^\s*(.+?)\s*:\s*(" + _FLOAT + r")\s+(" + _FLOAT + r")\s*$", line
        )
        if m:
            out[" ".join(m.group(1).split())] = {
                "D_cm-1": _to_float(m.group(2)),
                "E_cm-1": _to_float(m.group(3)),
            }
    return out


# ─────────────────────────────────────────────────────────────────────
# g-tensor sub-parser
# ─────────────────────────────────────────────────────────────────────

def _parse_g_tensor(lines: List[str], start: int, end: int) -> Dict[str, Any]:
    """Parse the ELECTRONIC G-MATRIX section."""
    section = lines[start:end]
    data: Dict[str, Any] = {}

    for idx, line in enumerate(section):
        if re.search(r"^\s*The g-matrix\s*:", line):
            data["g_matrix"] = _parse_matrix3(section, idx)

        elif re.search(r"^\s*DSO contribution to g-matrix\s*:", line):
            data["DSO_contribution"] = _parse_matrix3(section, idx)

        elif re.search(r"^\s*PSO contribution to g-matrix\s*:", line):
            data["PSO_contribution"] = _parse_matrix3(section, idx)

        elif re.search(r"^\s*Breakdown of the contributions\s*$", line):
            data["breakdown"] = _parse_g_breakdown(section, idx)

        elif re.search(r"^\s*Orientation\s*:", line):
            data["orientation"] = _parse_orientation(section, idx)

    return data


def _parse_g_breakdown(
    lines: List[str], start: int,
) -> Dict[str, Dict[str, Any]]:
    """Parse the g-tensor breakdown table."""
    out: Dict[str, Dict[str, Any]] = {}
    for j in range(start + 1, len(lines)):
        line = lines[j]
        if re.search(r"^\s*Orientation\s*:|^\s*Notes:", line):
            break
        stripped = line.strip()
        if not stripped or set(stripped) == {"-"}:
            continue
        m = re.match(
            r"^\s*([A-Za-z0-9()\/+\-.\s]+?)\s+("
            + _FLOAT + r")\s+(" + _FLOAT + r")\s+(" + _FLOAT
            + r")(?:\s+iso=\s*(" + _FLOAT + r"))?\s*$",
            line,
        )
        if m:
            label = " ".join(m.group(1).split())
            entry: Dict[str, Any] = {
                "values": [
                    _to_float(m.group(2)),
                    _to_float(m.group(3)),
                    _to_float(m.group(4)),
                ]
            }
            if m.group(5) is not None:
                entry["iso"] = _to_float(m.group(5))
            out[label] = entry
    return out


def _parse_g_atom_analysis(
    lines: List[str], start: int, end: int,
) -> Dict[str, Any]:
    """Parse G-TENSOR ATOM AND BOND ANALYSIS."""
    out: Dict[str, Any] = {"atom_contributions": []}
    totals: Dict[str, Any] = {}

    rx_atom = re.compile(
        r"^\s*([A-Za-z]+)\s*-\s*(\d+)\s*:\s*("
        + _FLOAT + r")\s+(" + _FLOAT + r")\s+(" + _FLOAT
        + r")\s+iso=\s*(" + _FLOAT + r")\s*(\*?)\s*$"
    )
    rx_total = re.compile(
        r"^\s*(Total(?:-1c|-2c)?|Total)\s+("
        + _FLOAT + r")\s+(" + _FLOAT + r")\s+(" + _FLOAT
        + r")\s+(" + _FLOAT + r")\s*$",
        re.I,
    )

    for j in range(start, end):
        line = lines[j]
        m_atom = rx_atom.match(line)
        if m_atom:
            out["atom_contributions"].append({
                "element": m_atom.group(1),
                "atom_index": int(m_atom.group(2)),
                "values": [
                    _to_float(m_atom.group(3)),
                    _to_float(m_atom.group(4)),
                    _to_float(m_atom.group(5)),
                ],
                "iso": _to_float(m_atom.group(6)),
                "flagged": bool(m_atom.group(7)),
            })
            continue
        m_total = rx_total.match(line)
        if m_total:
            totals[m_total.group(1)] = {
                "values": [
                    _to_float(m_total.group(2)),
                    _to_float(m_total.group(3)),
                    _to_float(m_total.group(4)),
                ],
                "iso": _to_float(m_total.group(5)),
            }

    if totals:
        out["totals"] = totals
    return out


# ─────────────────────────────────────────────────────────────────────
# Hyperfine / EFG sub-parser
# ─────────────────────────────────────────────────────────────────────

def _parse_hyperfine(
    lines: List[str], start: int, end: int,
) -> Dict[str, Any]:
    """Parse the ELECTRIC AND MAGNETIC HYPERFINE STRUCTURE section."""
    data: Dict[str, Any] = {}

    # Summary info before first nucleus
    nucleus_indices = [
        i for i in range(start, end)
        if re.match(r"^\s*Nucleus\s+\d+[A-Za-z]+\s*:", lines[i])
    ]
    summary_end = nucleus_indices[0] if nucleus_indices else end

    # Extract counts and metadata
    for j in range(start, summary_end):
        line = lines[j]
        m = re.match(
            r"^\s*Number of nuclei to compute\s+(.+?)\s*:\s*(\d+)\s*$", line
        )
        if m:
            data.setdefault("requested_counts", {})[m.group(1).strip()] = (
                int(m.group(2))
            )

    # Parse each nucleus block
    nuclei: List[Dict[str, Any]] = []
    for k, i0 in enumerate(nucleus_indices):
        i1 = nucleus_indices[k + 1] if k + 1 < len(nucleus_indices) else end
        nuclei.append(_parse_nucleus_block(lines[i0:i1]))

    data["nuclei"] = nuclei
    data["nucleus_count"] = len(nuclei)
    return data


def _parse_nucleus_block(block: List[str]) -> Dict[str, Any]:
    """Parse one nucleus sub-block (HFC + EFG + quadrupole)."""
    out: Dict[str, Any] = {}
    joined = "\n".join(block)

    # ── Header: nucleus identity ──────────────────────────────────
    m = re.search(
        r"Nucleus\s+(\d+)([A-Za-z]+)\s*:\s*([A-Za-z])\s*:"
        r"\s*Isotope=\s*(\d+)\s+I=\s*(" + _FLOAT + r")\s+P=\s*("
        + _FLOAT + r")\s+(.+)",
        joined,
    )
    if m:
        out["nucleus_index"] = int(m.group(1))
        out["element"] = m.group(2)
        out["label"] = m.group(3)
        out["isotope"] = int(m.group(4))
        out["spin_I"] = _to_float(m.group(5))
        out["P_MHz_per_au3"] = _to_float(m.group(6))

    # ── Quadrupole header ─────────────────────────────────────────
    m = re.search(
        r"Q\s*:\s*Isotope=\s*(\d+)\s+I=\s*(" + _FLOAT
        + r")\s+Q=\s*(" + _FLOAT + r")\s+(\S+)",
        joined,
    )
    if m:
        out["quadrupole"] = {
            "isotope": int(m.group(1)),
            "spin_I": _to_float(m.group(2)),
            "Q": _to_float(m.group(3)),
            "Q_units": m.group(4),
        }

    # ── HFC/EFG calculation flags ─────────────────────────────────
    m = re.search(
        r"HFC:\s*iso\s*=\s*(YES|NO)\s+dip\s*=\s*(YES|NO)"
        r"\s+orb\s*=\s*(YES|NO)\s+gauge\s*=\s*(YES|NO)",
        joined, re.I,
    )
    if m:
        out["HFC_flags"] = {
            "iso": m.group(1).upper() == "YES",
            "dip": m.group(2).upper() == "YES",
            "orb": m.group(3).upper() == "YES",
            "gauge": m.group(4).upper() == "YES",
        }

    m = re.search(
        r"EFG:\s*fgrad\s*=\s*(YES|NO)\s+rho\s*=\s*(YES|NO)",
        joined, re.I,
    )
    if m:
        out["EFG_flags"] = {
            "fgrad": m.group(1).upper() == "YES",
            "rho": m.group(2).upper() == "YES",
        }

    # ── Matrices and scalar values ────────────────────────────────
    for idx, line in enumerate(block):
        if re.search(r"Total HFC matrix \(all values in MHz\)\s*:", line, re.I):
            out["total_HFC_matrix_MHz"] = _parse_matrix3(block, idx + 1)

        elif re.search(r"SD contribution to HFC matrix\s*:", line, re.I):
            out["SD_contribution_matrix"] = _parse_matrix3(block, idx)

        elif re.search(r"NOC/SOC contribution to HFC matrix\s*:", line, re.I):
            out["NOC_SOC_contribution_matrix"] = _parse_matrix3(block, idx)

        elif re.search(
            r"Gauge correction \(diamagnetic\) contribution to HFC matrix\s*:",
            line, re.I,
        ):
            out["gauge_diamagnetic_contribution_matrix"] = _parse_matrix3(
                block, idx
            )

        elif re.search(r"^\s*Orientation\s*:\s*$", line):
            if "orientation" not in out:
                out["orientation"] = _parse_orientation(block, idx)

        elif re.search(
            r"Raw EFG matrix \(all values in a\.u\.\*\*-3\)\s*:", line, re.I
        ):
            out["raw_EFG_matrix_au-3"] = _parse_matrix3(block, idx + 1)

        elif m := re.match(
            r"^\s*V\(Tot\)\s+(" + _FLOAT + r")\s+(" + _FLOAT + r")\s+("
            + _FLOAT + r")\s*$",
            line,
        ):
            out["V_tot_eigenvalues_au-3"] = [
                _to_float(m.group(1)),
                _to_float(m.group(2)),
                _to_float(m.group(3)),
            ]

        elif m := re.match(
            r"^\s*V\(El\s*\)\s+(" + _FLOAT + r")\s+(" + _FLOAT + r")\s+("
            + _FLOAT + r")\s*$",
            line,
        ):
            out.setdefault("EFG_contributions", {})["V_El"] = [
                _to_float(m.group(1)),
                _to_float(m.group(2)),
                _to_float(m.group(3)),
            ]

        elif m := re.match(
            r"^\s*V\(Nuc\)\s+(" + _FLOAT + r")\s+(" + _FLOAT + r")\s+("
            + _FLOAT + r")\s*$",
            line,
        ):
            out.setdefault("EFG_contributions", {})["V_Nuc"] = [
                _to_float(m.group(1)),
                _to_float(m.group(2)),
                _to_float(m.group(3)),
            ]

        elif m := re.match(
            r"^\s*e\*\*2qQ\s*=\s*(" + _FLOAT + r")\s+MHz\s*$", line
        ):
            out.setdefault("quadrupole_coupling", {})["e2qQ_MHz"] = (
                _to_float(m.group(1))
            )

        elif m := re.match(
            r"^\s*e\*\*2qQ/\(4I\*\(2I-1\)\)\s*=\s*(" + _FLOAT + r")\s+MHz",
            line,
        ):
            out.setdefault("quadrupole_coupling", {})[
                "e2qQ_over_4I_2I-1_MHz"
            ] = _to_float(m.group(1))

        elif m := re.match(r"^\s*eta\s*=\s*(" + _FLOAT + r")\s*$", line):
            out.setdefault("quadrupole_coupling", {})["eta"] = _to_float(
                m.group(1)
            )

    # ── Principal components (A(Tot), A(FC), A(SD), etc.) ─────────
    principal_rx = re.compile(
        r"^\s*(A\([^)]+\)|A\(ORB\+DIA\))\s+("
        + _FLOAT + r")\s+(" + _FLOAT + r")\s+(" + _FLOAT
        + r")(?:\s+A\(PC\)\s*=\s*(" + _FLOAT
        + r"))?(?:\s+A\(iso\)\s*=\s*(" + _FLOAT + r"))?\s*$"
    )
    principal: Dict[str, Any] = {}
    for line in block:
        m = principal_rx.match(line)
        if m:
            entry: Dict[str, Any] = {
                "values_MHz": [
                    _to_float(m.group(2)),
                    _to_float(m.group(3)),
                    _to_float(m.group(4)),
                ]
            }
            if m.group(5) is not None:
                entry["A_PC_MHz"] = _to_float(m.group(5))
            if m.group(6) is not None:
                entry["A_iso_MHz"] = _to_float(m.group(6))
            principal[m.group(1)] = entry
    if principal:
        out["principal_components"] = principal

    # ── A(FC) local/nonlocal summary ──────────────────────────────
    local_fc: Dict[str, float] = {}
    for line in block:
        for label, pattern in [
            ("local_MHz", r"Total local contribution\s*:\s*(" + _FLOAT + r")"),
            ("bond_MHz", r"Total bond contribution\s*:\s*(" + _FLOAT + r")"),
            ("distant_MHz", r"Total distant contribution\s*:\s*(" + _FLOAT + r")"),
        ]:
            m = re.match(r"^\s*" + pattern + r"\s*$", line)
            if m:
                local_fc[label] = _to_float(m.group(1))
    if local_fc:
        out["A_FC_local_nonlocal"] = local_fc

    # ── A(SD) local/nonlocal decomposition ────────────────────────
    sd_decomp: Dict[str, Any] = {}
    for label, pat in [
        ("one_center", r"One center contribution to the hyperfine coupling"),
        ("two_center_point_charge", r"Two center 'point charge' contribution to the hyperfine coupling"),
        ("two_center_bond", r"Two center 'bond' contribution to the hyperfine coupling"),
        ("three_center", r"Three center contribution to the hyperfine coupling"),
    ]:
        parsed = _parse_contribution_with_eigs(block, pat)
        if parsed:
            sd_decomp[label] = parsed
    if sd_decomp:
        out["A_SD_decomposition"] = sd_decomp

    # ── EFG local/nonlocal decomposition ──────────────────────────
    efg_decomp: Dict[str, Any] = {}
    for label, pat in [
        ("one_center", r"One center contribution to the field gradient"),
        ("two_center_point_charge", r"Two center 'point charge' contribution to the field gradient"),
        ("two_center_bond", r"Two center 'bond' contribution to the field gradient"),
        ("three_center", r"Three center contribution to the field gradient"),
    ]:
        parsed = _parse_contribution_with_eigs(block, pat)
        if parsed:
            efg_decomp[label] = parsed
    if efg_decomp:
        out["EFG_decomposition"] = efg_decomp

    # ── Natural spin-orbital contributions to A(FC) ───────────────
    nso = _parse_nso_table(block)
    if nso:
        out["A_FC_natural_spin_orbital_contributions"] = nso

    return out


def _parse_nso_table(block: List[str]) -> Optional[List[Dict[str, float]]]:
    """Parse the 'Natural spin-orbital contributions' table."""
    start = None
    for i, line in enumerate(block):
        if re.search(r"Natural spin-orbital contributions", line, re.I):
            start = i
            break
    if start is None:
        return None

    data: List[Dict[str, float]] = []
    rx = re.compile(
        r"^\s*(\d+)\s+(" + _FLOAT + r"):\s+(" + _FLOAT + r")\s*$"
    )
    started = False
    for j in range(start + 1, len(block)):
        line = block[j]
        if re.search(r"^\s*\*+\s*$", line):
            break
        if re.search(r"LOCAL VS\. NONLOCAL CONTRIBUTIONS", line, re.I):
            break
        m = rx.match(line)
        if m:
            started = True
            data.append({
                "mo_index": int(m.group(1)),
                "occupation": _to_float(m.group(2)),
                "contribution_MHz": _to_float(m.group(3)),
            })
        elif started:
            break
    return data or None


# ─────────────────────────────────────────────────────────────────────
# Main EPR module
# ─────────────────────────────────────────────────────────────────────

class EPRModule(BaseModule):
    """
    Extracts EPR (Electron Paramagnetic Resonance) properties:
    Zero-Field Splitting, g-tensor, and hyperfine/EFG data.

    Returns None if no EPR sections are found in the output.
    """

    name = "epr"

    def parse(self, lines: list[str]) -> Optional[Dict[str, Any]]:
        data: Dict[str, Any] = {}

        # ── Zero-Field Splitting ──────────────────────────────────
        zfs_start = self.find_line(lines, "ZERO-FIELD SPLITTING")
        if zfs_start != -1:
            # Find end: either g-matrix heading or end of file
            zfs_end = self.find_line(lines, "ELECTRONIC G-MATRIX", zfs_start + 1)
            if zfs_end == -1:
                zfs_end = len(lines)
            zfs = _parse_zfs(lines, zfs_start, zfs_end)
            if zfs:
                data["zero_field_splitting"] = zfs

        # ── Electronic g-tensor ───────────────────────────────────
        g_start = self.find_line(lines, "ELECTRONIC G-MATRIX")
        if g_start != -1:
            # Atom analysis may follow
            g_atom_start = self.find_line(
                lines, "G-TENSOR ATOM AND BOND ANALYSIS", g_start + 1
            )
            g_section_end = g_atom_start if g_atom_start != -1 else self.find_line(
                lines, "ELECTRIC AND MAGNETIC HYPERFINE STRUCTURE", g_start + 1
            )
            if g_section_end == -1:
                g_section_end = len(lines)

            g = _parse_g_tensor(lines, g_start, g_section_end)
            if g:
                data["g_tensor"] = g

            # Atom/bond analysis
            if g_atom_start != -1:
                atom_end = self.find_line(
                    lines, "ELECTRIC AND MAGNETIC HYPERFINE STRUCTURE",
                    g_atom_start + 1,
                )
                if atom_end == -1:
                    atom_end = len(lines)
                atom_analysis = _parse_g_atom_analysis(
                    lines, g_atom_start + 1, atom_end
                )
                if atom_analysis.get("atom_contributions"):
                    data.setdefault("g_tensor", {})["atom_analysis"] = (
                        atom_analysis
                    )

        # ── Hyperfine / EFG ───────────────────────────────────────
        hf_start = self.find_line(
            lines, "ELECTRIC AND MAGNETIC HYPERFINE STRUCTURE"
        )
        if hf_start != -1:
            hf = _parse_hyperfine(lines, hf_start, len(lines))
            if hf:
                data["hyperfine"] = hf

        return data if data else None
