"""
NBO (Natural Bond Orbital) analysis module.

Extracts all NBO sections:
  - Mulliken populations by AO (NBO section)
  - Mayer-Mulliken bond orders (NBO section)
  - Natural populations (NAO occupancies + energies)
  - NPA summary (charge, core, valence, rydberg, total, electron config)
  - Wiberg bond indices (NAO basis)
  - NBI (Natural Binding Index)
  - NBO Lewis structure (CR, LP, BD, BD* occupancies, energies, hybridization)
  - NHO directionality and bond bending
  - CMO analysis (NBO contributions to canonical MOs)
  - Second-order perturbation theory (E2, E(NL)-E(L), F(L,NL))
  - NLMO analysis (hybridization/polarization, bond orders, steric)
  - NPEPA
  - NBO summary table
  - Repeated-block provenance for optimization outputs

For UHF/UKS: all NBO sections are parsed separately for alpha and beta spins.
The overall NPA (with spin density column) is also captured.
For optimization outputs with repeated NBO runs, the final valid block is used
for density-sensitive NPA data while non-density subsections printed only in an
earlier block are preserved with supplemental block provenance.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..output.csv_section_registry import (
    NBI_MATRIX_CSV_SECTION_PLUGIN,
    NBO_E2_CSV_SECTION_PLUGIN,
    NBO_LEWIS_CSV_SECTION_PLUGIN,
    NBO_NAO_CSV_SECTION_PLUGIN,
    NBO_NLMO_BOND_ORDER_CSV_SECTION_PLUGIN,
    NBO_NLMO_HYBRIDIZATION_CSV_SECTION_PLUGIN,
    NBO_NLMO_STERIC_CSV_SECTION_PLUGIN,
    NBO_NPA_CSV_SECTION_PLUGIN,
    WIBERG_MATRIX_CSV_SECTION_PLUGIN,
)
from ..parser_section_plugin import ParserSectionAlias, ParserSectionPlugin
from ..plugin_bundle import PluginBundle, PluginMetadata
from .base import BaseModule


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _find_nbo_start(lines: List[str]) -> int:
    blocks = _find_nbo_blocks(lines)
    if not blocks:
        return -1
    return int(blocks[0]["line_start_index"])


def _find_nbo_blocks(lines: List[str]) -> List[Dict[str, Any]]:
    """Return all NBO 7 blocks with enough provenance to choose safely."""

    blocks: List[Dict[str, Any]] = []
    current_cycle: Optional[int] = None
    last_final_stationary: Optional[int] = None

    for idx, line in enumerate(lines):
        cycle_match = re.search(r"GEOMETRY OPTIMIZATION CYCLE\s+(\d+)", line, re.I)
        if cycle_match:
            current_cycle = int(cycle_match.group(1))
        if "FINAL ENERGY EVALUATION AT THE STATIONARY POINT" in line.upper():
            last_final_stationary = idx
        if "NBO 7." in line and "***" in line:
            blocks.append({
                "line_start_index": idx,
                "line_start": idx + 1,
                "optimization_cycle": current_cycle,
                "is_after_final_stationary": last_final_stationary is not None,
                "final_stationary_line": (
                    last_final_stationary + 1
                    if last_final_stationary is not None
                    else None
                ),
            })

    for block_index, block in enumerate(blocks):
        start = int(block["line_start_index"])
        end = (
            int(blocks[block_index + 1]["line_start_index"])
            if block_index + 1 < len(blocks)
            else len(lines)
        )
        block["nbo_block_index"] = block_index + 1
        block["nbo_block_count"] = len(blocks)
        block["line_end_index"] = end
        block["line_end"] = end
        block["has_npa_summary"] = any(
            "Summary of Natural Population Analysis:" in line
            for line in lines[start:end]
        )
        block.update(_infer_nbo_density_context(lines, start))

    return blocks


def _infer_nbo_density_context(lines: List[str], nbo_start: int) -> Dict[str, Any]:
    """Infer which density a printed NBO block follows."""

    meta: Dict[str, Any] = {
        "density_context": "reference_scf",
        "density_kind": "scf",
        "excited_state_specific": False,
    }
    scan_start = max(0, nbo_start - 6000)
    context_line: Optional[int] = None
    for idx in range(nbo_start - 1, scan_start - 1, -1):
        upper = lines[idx].upper()
        if "NBO 7." in upper and "***" in upper:
            break
        if "RELAXED CIS/TDA DENSITY POPULATION ANALYSIS" in upper:
            context_line = idx
            meta.update({
                "density_context": "excited_state",
                "density_kind": "relaxed_cis_tda",
                "excited_state_specific": True,
                "density_context_line": idx + 1,
            })
            break
        if "UNRELAXED CIS/TDA DENSITY POPULATION ANALYSIS" in upper:
            context_line = idx
            meta.update({
                "density_context": "excited_state",
                "density_kind": "unrelaxed_cis_tda",
                "excited_state_specific": True,
                "density_context_line": idx + 1,
            })
            break

    if context_line is not None:
        for idx in range(context_line, nbo_start):
            root_match = re.search(r"\bIROOT\s+(\d+)\b", lines[idx], re.I)
            if root_match:
                meta["root"] = int(root_match.group(1))
            density_match = re.search(
                r"Input electron density\s+\.+\s+(\S+)",
                lines[idx],
                re.I,
            )
            if density_match:
                meta["input_electron_density_file"] = density_match.group(1)
            base_match = re.search(
                r"BaseName \([^)]*\)\s+\.+\s+(\S+)",
                lines[idx],
                re.I,
            )
            if base_match:
                meta["base_name"] = base_match.group(1)

    return meta


def _select_nbo_block(blocks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Choose the final trustworthy NBO block instead of the first occurrence."""

    valid = [block for block in blocks if block.get("has_npa_summary")]
    if not valid:
        valid = list(blocks)
    if not valid:
        return None

    final_excited = [
        block
        for block in valid
        if block.get("is_after_final_stationary")
        and block.get("density_context") == "excited_state"
    ]
    if final_excited:
        return final_excited[-1]

    final_blocks = [
        block
        for block in valid
        if block.get("is_after_final_stationary")
    ]
    if final_blocks:
        return final_blocks[-1]

    return valid[-1]


def _stage_from_source_path(context: Dict[str, Any]) -> Optional[str]:
    """Infer a dossier-like stage label from the source path when obvious."""

    source = str(context.get("source_path") or "")
    if not source:
        return None
    parts = re.split(r"[\\/]+", source)
    known = {"S0", "S1", "S1B", "TDDFT", "TDDFT_S1"}
    for part in reversed(parts):
        normalized = part.upper()
        if normalized in known:
            return "S1b" if normalized == "S1B" else normalized
    return None


def _nbo_block_public_metadata(block: Dict[str, Any]) -> Dict[str, Any]:
    """Return JSON-safe NBO provenance fields without raw ORCA text."""

    keys = (
        "nbo_block_index",
        "nbo_block_count",
        "line_start",
        "line_end",
        "optimization_cycle",
        "is_after_final_stationary",
        "final_stationary_line",
        "density_context",
        "density_kind",
        "density_context_line",
        "excited_state_specific",
        "input_electron_density_file",
        "base_name",
        "root",
        "has_npa_summary",
    )
    return {
        key: block.get(key)
        for key in keys
        if block.get(key) is not None
    }


def _nbo_to_orca_index(index: Optional[int]) -> Optional[int]:
    """Convert NBO's 1-based printed indices to ORCA's 0-based convention."""
    if index is None:
        return None
    return int(index) - 1


def _annotate_atom_index(entry: Dict[str, Any], source_key: str, prefix: str) -> None:
    """Preserve the printed NBO atom index and add the ORCA-aligned zero-based index."""
    index = entry.get(source_key)
    if not isinstance(index, int):
        return
    entry[f"{prefix}_nbo_index"] = index
    entry[f"{prefix}_orca_index"] = _nbo_to_orca_index(index)


def _annotate_nbo_orbital_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Attach atom-index metadata and a compact orbital-character label."""
    _annotate_atom_index(entry, "atom1_index", "atom1")
    _annotate_atom_index(entry, "atom2_index", "atom2")

    type_number = entry.get("type_number")
    if isinstance(type_number, str) and type_number.isdigit():
        type_number = int(type_number)

    entry["character_class"] = _classify_nbo_character(entry.get("type"), type_number)
    entry["character_label"] = _format_nbo_character_label(entry)
    return entry


def _classify_nbo_character(nbo_type: Optional[str], type_number: Optional[int]) -> Optional[str]:
    """Map NBO orbital kinds to a compact chemical character label."""
    if not nbo_type:
        return None
    clean = nbo_type.strip().upper()
    if clean == "LP":
        return "n"
    if clean == "LP*":
        return "n*"
    if clean == "BD":
        return "sigma" if type_number == 1 else "pi"
    if clean == "BD*":
        return "sigma*" if type_number == 1 else "pi*"
    if clean == "CR":
        return "core"
    if clean == "RY":
        return "rydberg"
    return clean.lower()


def _format_nbo_character_label(parsed: Dict[str, Any]) -> Optional[str]:
    """Build a compact human-readable NBO character label."""
    character = parsed.get("character_class")
    atom1_symbol = parsed.get("atom1_symbol")
    atom1_nbo_index = parsed.get("atom1_nbo_index")
    atom2_symbol = parsed.get("atom2_symbol")
    atom2_nbo_index = parsed.get("atom2_nbo_index")

    if not character:
        return None
    if atom1_symbol and atom1_nbo_index is not None and atom2_symbol and atom2_nbo_index is not None:
        return f"{character}({atom1_symbol}{atom1_nbo_index}-{atom2_symbol}{atom2_nbo_index})"
    if atom1_symbol and atom1_nbo_index is not None:
        return f"{character}({atom1_symbol}{atom1_nbo_index})"
    return character


def _parse_nbo_descriptor(desc: str) -> Dict[str, Any]:
    """Parse a compact NBO descriptor like 'BD*( 2) C 7- C 8*'."""
    cleaned = re.sub(r"\s+", " ", desc.strip())
    parsed: Dict[str, Any] = {"raw": desc.strip(), "normalized": cleaned}

    header_match = re.match(
        r"(?P<nbo_type>[A-Za-z]{1,3}\*?)\s*\(\s*(?P<type_number>\d+)\)\s*(?P<tail>.+)",
        cleaned,
    )
    if not header_match:
        return parsed

    nbo_type = header_match.group("nbo_type")
    type_number = int(header_match.group("type_number"))
    parsed["nbo_type"] = nbo_type
    parsed["type_number"] = type_number

    tail = header_match.group("tail").strip()
    atom_match = re.match(
        r"(?P<atom1_symbol>[A-Za-z]{1,3})\s*(?P<atom1_index>\d+)"
        r"(?:\s*-\s*(?P<atom2_symbol>[A-Za-z]{1,3})\s*(?P<atom2_index>\d+))?"
        r"(?P<tag>\*|\([^)]+\))?\s*$",
        tail,
    )
    if atom_match:
        atom1_index = int(atom_match.group("atom1_index"))
        parsed["atom1_symbol"] = atom_match.group("atom1_symbol")
        parsed["atom1_nbo_index"] = atom1_index
        parsed["atom1_orca_index"] = _nbo_to_orca_index(atom1_index)
        if atom_match.group("atom2_symbol") and atom_match.group("atom2_index"):
            atom2_index = int(atom_match.group("atom2_index"))
            parsed["atom2_symbol"] = atom_match.group("atom2_symbol")
            parsed["atom2_nbo_index"] = atom2_index
            parsed["atom2_orca_index"] = _nbo_to_orca_index(atom2_index)
        if atom_match.group("tag"):
            parsed["tag"] = atom_match.group("tag")

    parsed["character_class"] = _classify_nbo_character(nbo_type, type_number)
    parsed["character_label"] = _format_nbo_character_label(parsed)
    return parsed


def _parse_matrix(lines: List[str], start: int, n_atoms: int) -> Optional[List[List[float]]]:
    """Parse a symmetric n_atoms × n_atoms matrix (e.g. Wiberg, Mayer)."""
    matrix = []
    for ln in lines[start:]:
        m = re.match(r"\s+\d+\.\s+\w+", ln)
        if m:
            vals = re.findall(r"[-\d.]+", ln)
            # First two tokens are index and label (skip), rest are values
            row = [float(v) for v in vals[1:]]  # skip the index
            matrix.append(row)
        elif matrix and ln.strip() == "":
            break
    return matrix if matrix else None


def _parse_npa_summary(lines: List[str], start: int, is_uhf: bool) -> Optional[Dict]:
    """Parse the 'Summary of Natural Population Analysis' block."""
    atoms = []
    for ln in lines[start:]:
        if "=====" in ln or "Total" in ln.split()[0:1]:
            break
        m = re.match(
            r"\s+(\w+)\s+(\d+)\s+([-\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
            + (r"\s+([-\d.]+)" if is_uhf else r""),
            ln,
        )
        if m:
            entry = {
                "symbol": m.group(1),
                "index": int(m.group(2)),
                "natural_charge": float(m.group(3)),
                "core_pop": float(m.group(4)),
                "valence_pop": float(m.group(5)),
                "rydberg_pop": float(m.group(6)),
                "total_pop": float(m.group(7)),
            }
            if is_uhf and m.group(8):
                entry["spin_density"] = float(m.group(8))
            _annotate_atom_index(entry, "index", "atom")
            atoms.append(entry)
    return atoms if atoms else None


def _parse_nao_occupancies(lines: List[str], start: int, has_spin: bool) -> List[Dict]:
    """Parse the NATURAL POPULATIONS (NAO occupancies) block."""
    naos = []
    for ln in lines[start:]:
        if has_spin:
            m = re.match(
                r"\s+(\d+)\s+(\w+)\s+(\d+)\s+(\w+)\s+(\w+\([^)]+\))\s+([\d.]+)\s+([-\d.]+)",
                ln,
            )
            if m:
                entry = {
                    "index": int(m.group(1)),
                    "symbol": m.group(2),
                    "atom_no": int(m.group(3)),
                    "angular": m.group(4),
                    "type": m.group(5),
                    "occupancy": float(m.group(6)),
                    "spin": float(m.group(7)),
                }
                _annotate_atom_index(entry, "atom_no", "atom")
                naos.append(entry)
                continue
        else:
            m = re.match(
                r"\s+(\d+)\s+(\w+)\s+(\d+)\s+(\w+)\s+(\w+\([^)]+\))\s+([\d.]+)\s+([-\d.]+)",
                ln,
            )
            if m:
                entry = {
                    "index": int(m.group(1)),
                    "symbol": m.group(2),
                    "atom_no": int(m.group(3)),
                    "angular": m.group(4),
                    "type": m.group(5),
                    "occupancy": float(m.group(6)),
                    "energy_Eh": float(m.group(7)),
                }
                _annotate_atom_index(entry, "atom_no", "atom")
                naos.append(entry)
                continue

        if "Summary" in ln or "Wiberg" in ln:
            break
    return naos


def _parse_nao_occupancies_v2(lines: List[str], start: int) -> List[Dict]:
    """Alternate NAO parser for lines like:
       1  O  1  s  Cor( 1s)   2.00000   -19.34307
       or with spin column:
       1  O  1  s  Cor( 1s)   2.00000    0.00000
    """
    naos = []
    for ln in lines[start:]:
        m = re.match(
            r"\s+(\d+)\s+(\w+)\s+(\d+)\s+(\S+)\s+(\S+\([^)]+\))\s+([-\d.]+)\s+([-\d.]+)",
            ln,
        )
        if m:
            entry = {
                "index": int(m.group(1)),
                "symbol": m.group(2),
                "atom_no": int(m.group(3)),
                "angular": m.group(4),
                "type": m.group(5),
                "occupancy": float(m.group(6)),
                "energy_or_spin": float(m.group(7)),
            }
            _annotate_atom_index(entry, "atom_no", "atom")
            naos.append(entry)
        elif naos and ("Summary" in ln or "Wiberg" in ln or "NBI" in ln):
            break
    return naos


def _parse_wiberg(lines: List[str], start: int) -> Optional[Dict]:
    """Parse Wiberg bond index matrix."""
    data = {}
    # Header row with atom labels
    header = None
    for i, ln in enumerate(lines[start: start + 5]):
        m = re.match(r"\s+Atom\s+([\d\s]+)", ln)
        if m:
            header_idx = start + i
            break
    else:
        header_idx = start

    matrix = []
    totals = []
    in_matrix = False
    in_totals = False

    for ln in lines[header_idx:]:
        if "Atom" in ln and "------" not in ln and not in_matrix:
            in_matrix = True
            continue
        if in_matrix:
            m = re.match(r"\s+(\d+)\.\s+(\w+)((?:\s+[\d.]+)+)", ln)
            if m:
                vals = [float(v) for v in m.group(3).split()]
                matrix.append({"atom": int(m.group(1)), "symbol": m.group(2), "values": vals})
            elif "Totals" in ln:
                in_matrix = False
                in_totals = True
            elif ln.strip() == "" and matrix:
                break
        if in_totals:
            m = re.match(r"\s+(\d+)\.\s+(\w+)((?:\s+[\d.]+)+)", ln)
            if m:
                totals.append({"atom": int(m.group(1)), "symbol": m.group(2), "total": float(m.group(3).split()[0])})
            elif ln.strip() == "" and totals:
                break

    if matrix:
        data["matrix"] = matrix
    if totals:
        data["totals_by_atom"] = totals
    return data if data else None


def _parse_nbo_lewis(lines: List[str], start: int) -> List[Dict]:
    """
    Parse NBO Lewis structure (occupancy, type, atoms, hybridization).
    Entries look like:
       1. (2.00000) CR ( 1) O  1  s(100.00%)
       2. (1.99840) LP ( 1) O  1  s(  0.00%)p 1.00( 99.83%)d 0.00(  0.16%)
       4. (1.99952) BD ( 1) O  1- H  2
                   ( 73.42%)   0.8569* O  1 s( 23.87%)...
    """
    nbos = []
    current = None

    for ln in lines[start:]:
        # Check for end of NBO section
        if "non-Lewis" in ln or "NHO DIRECTIONALITY" in ln or "CMO:" in ln:
            if current:
                nbos.append(current)
            break

        # New NBO entry: number. (occ) TYPE (n) Atom  idx ...
        m = re.match(
            r"\s+(\d+)\.\s+\(([\d.]+)\)\s+(\w+)\s*\(\s*(\d+)\)\s+(\w+)\s+(\d+)(-\s+(\w+)\s+(\d+))?",
            ln,
        )
        if m:
            if current:
                nbos.append(current)
            current = {
                "nbo_index": int(m.group(1)),
                "occupancy": float(m.group(2)),
                "type": m.group(3),  # CR, LP, BD, BD*, RY
                "type_number": int(m.group(4)),
                "atom1_symbol": m.group(5),
                "atom1_index": int(m.group(6)),
                "hybridization": [],
            }
            if m.group(7):  # bond: has second atom
                current["atom2_symbol"] = m.group(8)
                current["atom2_index"] = int(m.group(9))
            _annotate_nbo_orbital_entry(current)

            # Hybridization on same line
            hyb = re.findall(r"([spdf])\s*\(([\d.]+)%\)", ln)
            if hyb:
                current["hybridization"] = [{"orbital": h[0], "percent": float(h[1])} for h in hyb]
            continue

        # Hybridization continuation
        if current:
            hyb = re.findall(r"([spdf])\s*\(([\d.]+)%\)", ln)
            if hyb and not current.get("hybridization"):
                current["hybridization"] = [{"orbital": h[0], "percent": float(h[1])} for h in hyb]

            # Polarization coefficient for BD orbitals
            m_pol = re.match(r"\s+\(\s*([\d.]+)%\)\s+([-\d.]+)\*\s+(\w+)\s+(\d+)", ln)
            if m_pol:
                pol_key = f"polarization_atom{len(current.get('polarization', [])) + 1}"
                if "polarization" not in current:
                    current["polarization"] = []
                pol_entry = {
                    "percent": float(m_pol.group(1)),
                    "coefficient": float(m_pol.group(2)),
                    "symbol": m_pol.group(3),
                    "atom_index": int(m_pol.group(4)),
                }
                _annotate_atom_index(pol_entry, "atom_index", "atom")
                current["polarization"].append(pol_entry)

    if current:
        nbos.append(current)

    return nbos


def _parse_nbo_summary(lines: List[str], start: int) -> List[Dict]:
    """Parse the NATURAL BOND ORBITALS summary table."""
    entries = []
    for ln in lines[start:]:
        # "  1. CR ( 1) O  1             2.00000   -19.34307"
        m = re.match(
            r"\s+(\d+)\.\s+(\w+)\s*\(\s*(\d+)\)\s+(\w+)\s+(\d+)(?:-\s+(\w+)\s+(\d+))?\s+([\d.]+)\s+([-\d.]+)",
            ln,
        )
        if m:
            entry = {
                "index": int(m.group(1)),
                "type": m.group(2),
                "type_number": int(m.group(3)),
                "atom1_symbol": m.group(4),
                "atom1_index": int(m.group(5)),
                "occupancy": float(m.group(8)),
                "energy_Eh": float(m.group(9)),
            }
            if m.group(6):
                entry["atom2_symbol"] = m.group(6)
                entry["atom2_index"] = int(m.group(7))
            _annotate_nbo_orbital_entry(entry)
            entries.append(entry)
        elif "----" in ln and entries:
            break
        elif "Total Lewis" in ln or "Total unit" in ln:
            break
    return entries


def _parse_e2_perturbation(lines: List[str], start: int) -> List[Dict]:
    """Parse second-order perturbation theory analysis."""
    interactions = []
    for ln in lines[start:]:
        m = re.match(
            r"\s+(\d+)\.\s+(\w+.*?)\s{2,}(\d+)\.\s+(\w+.*?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
            ln,
        )
        if m:
            interactions.append({
                "donor_nbo": m.group(1).strip(),
                "donor_desc": m.group(2).strip(),
                "acceptor_nbo": m.group(3).strip(),
                "acceptor_desc": m.group(4).strip(),
                "E2_kcal_mol": float(m.group(5)),
                "E_NL_minus_E_L_au": float(m.group(6)),
                "F_LNL_au": float(m.group(7)),
            })
        elif "NATURAL BOND ORBITALS" in ln or "NATURAL LOCALIZED" in ln:
            break
        elif ln.strip() == "" and len(interactions) > 3:
            break

    # Alternative parser for the actual format
    if not interactions:
        for ln in lines[start:]:
            # "    2. LP ( 1) O  1            35. RY ( 2) H  2            1.25    2.40   0.049"
            m = re.match(
                r"\s+(\d+\.\s+\w+\s*\(\s*\d+\)\s+\w+\s+\d+(?:-\s+\w+\s+\d+)?)\s+"
                r"(\d+\.\s+\w+\s*\(\s*\d+\)\s+\w+\s+\d+(?:-\s+\w+\s+\d+)?)\s+"
                r"([\d.]+)\s+([\d.]+)\s+([\d.]+)",
                ln,
            )
            if m:
                interactions.append({
                    "donor": m.group(1).strip(),
                    "acceptor": m.group(2).strip(),
                    "E2_kcal_mol": float(m.group(3)),
                    "E_NL_minus_E_L_au": float(m.group(4)),
                    "F_LNL_au": float(m.group(5)),
                })
            elif interactions and ("NATURAL BOND ORBITALS" in ln or "NATURAL LOCALIZED" in ln):
                break

    return interactions


def _parse_nlmo_steric(lines: List[str], start: int) -> Dict:
    """Parse NLMO steric analysis."""
    data = {}
    individual = []
    pairwise = []

    in_individual = False
    in_pairwise = False

    for ln in lines[start:]:
        if "NLMOs (i) in unit" in ln:
            in_individual = True
            continue
        if "Pairwise steric" in ln:
            in_individual = False
            in_pairwise = True
            continue
        if "Total steric exchange energy" in ln:
            m = re.search(r"Total steric exchange energy[,:]?\s+([-\d.]+)\s+kcal", ln)
            if m:
                data["total_steric_energy_kcal_mol"] = float(m.group(1))
        if "Total disjoint NLMO" in ln:
            m = re.search(r":\s+([-\d.]+)", ln)
            if m:
                data["total_disjoint_steric_kcal_mol"] = float(m.group(1))
        if "Overall disjoint NLMO" in ln:
            m = re.search(r":\s+([-\d.]+)", ln)
            if m:
                data["overall_disjoint_steric_kcal_mol"] = float(m.group(1))

        if in_individual:
            m = re.match(r"\s+(\d+)\.\s+(\w+.*?)\s+([-\d.]+)$", ln)
            if m:
                individual.append({"nbo": m.group(2).strip(), "dE_kcal_mol": float(m.group(3))})
            elif ln.strip() == "" and individual:
                in_individual = False

        if in_pairwise:
            m = re.match(
                r"\s+(\w+.*?)\s{3,}(\w+.*?)\s+([\d.]+)\s+([-\d.]+)\s*$", ln
            )
            if m:
                pairwise.append({
                    "nlmo_i": m.group(1).strip(),
                    "nlmo_j": m.group(2).strip(),
                    "overlap": float(m.group(3)),
                    "dE_kcal_mol": float(m.group(4)),
                })

        if "NBO analysis completed" in ln:
            break

    if individual:
        data["individual_contributions"] = individual
    if pairwise:
        data["pairwise_contributions"] = pairwise

    return data


def _parse_nlmo_hybridization(lines: List[str], start: int) -> List[Dict]:
    """Parse NLMO hybridization/polarization analysis block."""
    nlmos = []
    current = None

    for ln in lines[start:]:
        # New NLMO entry: "  1. (2.00000)  99.9998% CR ( 1) O  1"
        m = re.match(r"\s+(\d+)\.\s+\(([\d.]+)\)\s+([\d.]+)%\s+(.+)", ln)
        if m:
            if current:
                nlmos.append(current)
            current = {
                "index": int(m.group(1)),
                "occupancy": float(m.group(2)),
                "parent_pct": float(m.group(3)),
                "parent_desc": m.group(4).strip(),
                "contributions": [],
            }
            continue

        # Contribution: "   73.408%  O  1 s( 24.09%)p 3.14( 75.57%)..."
        if current:
            m2 = re.match(r"\s+([\d.]+)%\s+(\w+)\s+(\d+)\s+(.+)", ln)
            if m2:
                hyb = re.findall(r"([spdf])\s*\(([\d.]+)%\)", m2.group(4))
                contribution = {
                    "percent": float(m2.group(1)),
                    "symbol": m2.group(2),
                    "atom_index": int(m2.group(3)),
                    "hybridization": [{"orbital": h[0], "percent": float(h[1])} for h in hyb],
                }
                _annotate_atom_index(contribution, "atom_index", "atom")
                current["contributions"].append(contribution)

        if "Atom I" in ln or "Linear NLMO" in ln or "NBO/NLMO STERIC" in ln:
            if current:
                nlmos.append(current)
            break

    if current and current not in nlmos:
        nlmos.append(current)

    return nlmos


def _parse_nlmo_bond_orders(lines: List[str], start: int) -> Dict:
    """Parse NLMO bond order data."""
    data = {}
    matrix = []
    totals = []
    in_matrix = False
    in_totals = False

    for ln in lines[start:]:
        if "Atom-Atom Net Linear" in ln:
            in_matrix = True
            continue
        if "Linear NLMO/NPA Bond Orders, Totals" in ln:
            in_matrix = False
            in_totals = True
            continue
        if in_matrix:
            m = re.match(r"\s+(\d+)\.\s+(\w+)((?:\s+[\d.]+)+)", ln)
            if m:
                matrix.append({
                    "atom": int(m.group(1)),
                    "symbol": m.group(2),
                    "values": [float(v) for v in m.group(3).split()],
                })
            elif ln.strip() == "" and matrix:
                in_matrix = False
        if in_totals:
            m = re.match(r"\s+(\d+)\.\s+(\w+)\s+([\d.]+)", ln)
            if m:
                totals.append({
                    "atom": int(m.group(1)),
                    "symbol": m.group(2),
                    "total_bond_order": float(m.group(3)),
                })
            elif ln.strip() == "" and totals:
                break

        if "NBO/NLMO STERIC" in ln or "Individual LMO" in ln:
            break

    if matrix:
        data["nlmo_bond_order_matrix"] = matrix
    if totals:
        data["nlmo_bond_order_totals"] = totals
    return data


def _parse_cmo_analysis(lines: List[str], start: int) -> List[Dict]:
    """Parse CMO NBO contributions to canonical MOs."""
    cmos = []
    current = None

    for ln in lines[start:]:
        # New MO: "  MO   1 (occ): orbital energy =  -19.34311 a.u."
        m = re.match(r"\s+MO\s+(\d+)\s+\((occ|vir)\):\s+orbital energy\s+=\s+([-\d.]+)\s+a\.u\.", ln)
        if m:
            if current:
                cmos.append(current)
            mo_index = int(m.group(1))
            current = {
                "mo_index": mo_index,
                "nbo_mo_index": mo_index,
                "orca_orbital_index": mo_index - 1,
                "type": m.group(2),
                "energy_au": float(m.group(3)),
                "nbo_contributions": [],
            }
            continue

        if current:
            # Contribution: "               -1.000*[  1]: CR ( 1) O 1(cr)"
            m2 = re.match(r"\s+([-\d.]+)\*\[\s*(\d+)\]:\s+(.+)", ln)
            if m2:
                coefficient = float(m2.group(1))
                contribution = {
                    "coefficient": coefficient,
                    "weight": coefficient * coefficient,
                    "approx_percent": 100.0 * coefficient * coefficient,
                    "nbo_index": int(m2.group(2)),
                    "nbo_desc": m2.group(3).strip(),
                }
                contribution.update(_parse_nbo_descriptor(contribution["nbo_desc"]))
                current["nbo_contributions"].append(contribution)

        if "Molecular Orbital Atom-Atom Bonding" in ln or "SECOND ORDER PERTURBATION" in ln:
            if current:
                cmos.append(current)
            break

    return cmos


def _parse_cmo_bonding_character(lines: List[str], start: int) -> List[Dict]:
    """Parse MO bonding/non-bonding/anti-bonding character table."""
    entries = []
    for ln in lines[start:]:
        m = re.match(r"\s+(\d+)\((o|v)\)\s+.*?(\d+\.\d+)\(b\)\s+(\d+\.\d+)\(n\)\s+(\d+\.\d+)\(a\)", ln)
        if m:
            mo_index = int(m.group(1))
            entries.append({
                "mo_index": mo_index,
                "nbo_mo_index": mo_index,
                "orca_orbital_index": mo_index - 1,
                "type": "occ" if m.group(2).lower() == "o" else "vir",
                "bonding_frac": float(m.group(3)),
                "nonbonding_frac": float(m.group(4)),
                "antibonding_frac": float(m.group(5)),
            })
        if "SECOND ORDER PERTURBATION" in ln:
            break
    return entries


def _parse_nho_directionality(lines: List[str], start: int) -> List[Dict]:
    """Parse NHO directionality table."""
    entries = []
    for ln in lines[start:]:
        # "   2. LP ( 1) O  1          --     --     24.7   89.1   --      --     --    --"
        m = re.match(
            r"\s+(\d+)\.\s+(\w+)\s*\(\s*(\d+)\)\s+(\w+)\s+(\d+)(?:-\s+(\w+)\s+(\d+))?\s+"
            r"([\d.\-]+)\s+([\d.\-]+)\s+([\d.\-]+)\s+([\d.\-]+)\s+([\d.\-]+)",
            ln,
        )
        if m:
            entry = {
                "nbo_index": int(m.group(1)),
                "type": m.group(2),
                "type_number": int(m.group(3)),
                "atom1_symbol": m.group(4),
                "atom1_index": int(m.group(5)),
            }
            if m.group(6):
                entry["atom2_symbol"] = m.group(6)
                entry["atom2_index"] = int(m.group(7))
            _annotate_nbo_orbital_entry(entry)
            entries.append(entry)
        elif "NHO interhybrid" in ln or "CMO:" in ln:
            break
    return entries


# ---------------------------------------------------------------------------
# Main NBO module
# ---------------------------------------------------------------------------

class NBOModule(BaseModule):
    """
    Parses the complete NBO 7.x output section.

    For RHF/RKS: parses a single NBO run.
    For UHF/UKS: parses the overall NPA (with spin), then the alpha-spin
                 and beta-spin NBO sections separately.
    """

    name = "nbo"

    def parse(self, lines):
        all_lines = list(lines)
        is_uhf = self.context.get("is_uhf", False)

        nbo_blocks = _find_nbo_blocks(all_lines)
        selected_block = _select_nbo_block(nbo_blocks)
        if not selected_block:
            return None
        nbo_start = int(selected_block["line_start_index"])
        # Bound the parse at the end of the selected NBO block.  ORCA can print
        # NBO repeatedly during optimizations; unbounded first-hit searches must
        # not drift into a later density context.
        lines = all_lines[: int(selected_block["line_end_index"])]
        block_metadata = _nbo_block_public_metadata(selected_block)
        block_metadata["is_final_cycle"] = bool(
            selected_block.get("is_after_final_stationary")
        )
        stage = _stage_from_source_path(self.context)
        if stage:
            block_metadata["stage"] = stage

        data = self._parse_nbo_payload(lines, nbo_start, is_uhf)
        supplemental = self._supplement_missing_subsections(
            data=data,
            all_lines=all_lines,
            blocks=nbo_blocks,
            selected_block=selected_block,
            is_uhf=is_uhf,
        )

        if data:
            data.update(block_metadata)
            data["selected_nbo_block"] = dict(block_metadata)
            data["nbo_blocks"] = [
                _nbo_block_public_metadata(block)
                for block in nbo_blocks
            ]
            if supplemental:
                data["supplemental_nbo_blocks"] = supplemental
            warnings = []
            if len(nbo_blocks) > 1:
                warnings.append(
                    "Multiple NBO blocks found; selected the final valid block "
                    f"{block_metadata.get('nbo_block_index')}/"
                    f"{block_metadata.get('nbo_block_count')}."
                )
            if supplemental:
                warnings.append(
                    "Some NBO subsections were only printed in other NBO blocks "
                    "and were preserved with supplemental block provenance."
                )
            input_echo = self.context.get("input_echo") or {}
            block_names = {
                str(name).lower()
                for name in input_echo.get("block_names", [])
            }
            settings = (
                ((input_echo.get("blocks") or {}).get("tddft") or {}).get("settings")
                or ((input_echo.get("blocks") or {}).get("cis") or {}).get("settings")
                or {}
            )
            if (
                block_names & {"tddft", "cis"}
                and settings.get("iroot") is not None
                and block_metadata.get("density_context") != "excited_state"
            ):
                warnings.append(
                    "NBO/NPA was not marked as an excited-state density block; "
                    "do not interpret these NPA charges as excited-state charges."
                )
            if warnings:
                data["warnings"] = warnings

        return data if data else None

    def _parse_nbo_payload(self, lines, nbo_start: int, is_uhf: bool) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        # NBO Mulliken population by AO (overall, always present)
        idx = self.find_line(lines, "Mulliken Population Analysis (by orbital):", nbo_start)
        if idx != -1:
            ao_pops = self._parse_ao_populations(lines, idx + 3)
            data["nbo_ao_mulliken_populations"] = ao_pops

        # Mayer-Mulliken bond orders from NBO section (overall density matrix)
        idx_mm = self.find_line(lines, "Mayer-Mulliken bond order matrix:", nbo_start)
        if idx_mm != -1:
            data["nbo_mayer_mulliken_bond_orders"] = self._parse_bond_matrix(lines, idx_mm + 3)

        idx_mv = self.find_line(lines, "Mayer-Mulliken atomic valencies:", nbo_start)
        if idx_mv != -1:
            data["nbo_mayer_mulliken_valencies"] = self._parse_atom_valencies(lines, idx_mv + 3)

        if not is_uhf:
            # ---- RHF/RKS: single NBO analysis ----
            data.update(self._parse_rhf_nbo(lines, nbo_start))
        else:
            # ---- UHF/UKS: overall NPA (with spin), then alpha and beta ----
            # Overall NPA (occupancy + spin column)
            idx_npa = self.find_line(lines, "NATURAL POPULATIONS:  Natural atomic orbital occupancies", nbo_start)
            if idx_npa != -1:
                # This block has: Occupancy  Spin columns
                naos = self._parse_nao_with_spin(lines, idx_npa + 3)
                data["overall_nao_occupancies"] = naos

                idx_sum = self.find_line(lines, "Summary of Natural Population Analysis:", idx_npa)
                if idx_sum != -1:
                    npa_sum = self._parse_npa_uhf(lines, idx_sum + 4)
                    if npa_sum:
                        data["overall_npa_summary"] = npa_sum

                    idx_elcfg = self.find_line(lines, "Natural Electron Configuration", idx_sum)
                    if idx_elcfg != -1:
                        data["overall_electron_configurations"] = self._parse_electron_configs(lines, idx_elcfg + 2)

            # Overall Wiberg (based on overall density)
            idx_wib = self.find_line(lines, "Wiberg bond index matrix in the NAO basis:", nbo_start)
            if idx_wib != -1:
                data["overall_wiberg_bond_indices"] = _parse_wiberg(lines, idx_wib + 3)

            idx_nbi = self.find_line(lines, "NBI: Natural Binding Index", nbo_start)
            if idx_nbi != -1:
                data["overall_nbi"] = self._parse_bond_matrix(lines, idx_nbi + 4)

            # Alpha spin NBO analysis
            idx_alpha = self.find_line(lines, "Alpha spin orbitals", nbo_start)
            if idx_alpha != -1:
                data["alpha_spin"] = self._parse_spin_nbo(lines, idx_alpha, "alpha")

            # Beta spin NBO analysis
            idx_beta = self.find_line(lines, "NATURAL BOND ORBITAL ANALYSIS, beta spin orbitals:", nbo_start)
            if idx_beta != -1:
                data["beta_spin"] = self._parse_spin_nbo(lines, idx_beta, "beta")

        # NPEPA
        idx_npepa = self.find_line(lines, "NATURAL POLY-ELECTRON POPULATION ANALYSIS", nbo_start)
        if idx_npepa != -1:
            data["npepa"] = self._parse_npepa(lines, idx_npepa)

        return data

    def _supplement_missing_subsections(
        self,
        *,
        data: Dict[str, Any],
        all_lines: List[str],
        blocks: List[Dict[str, Any]],
        selected_block: Dict[str, Any],
        is_uhf: bool,
    ) -> List[Dict[str, Any]]:
        """Preserve non-density NBO subsections absent from the selected block."""

        supplemental_keys = (
            "cmo_analysis",
            "cmo_bonding_character",
            "npepa",
            "nlmo_hybridization",
            "nlmo_bond_orders",
            "nlmo_steric",
            "lmo_bond_orders",
            "nho_interhybrid_angles_raw",
        )
        missing = {
            key
            for key in supplemental_keys
            if not data.get(key)
        }
        if not missing:
            return []

        supplemental: List[Dict[str, Any]] = []
        selected_start = int(selected_block["line_start_index"])
        for block in reversed(blocks):
            if int(block["line_start_index"]) == selected_start:
                continue
            block_lines = all_lines[: int(block["line_end_index"])]
            parsed = self._parse_nbo_payload(
                block_lines,
                int(block["line_start_index"]),
                is_uhf,
            )
            added = []
            for key in sorted(missing):
                if parsed.get(key):
                    data[key] = parsed[key]
                    added.append(key)
            if added:
                meta = _nbo_block_public_metadata(block)
                meta["provided_sections"] = added
                supplemental.append(meta)
                missing.difference_update(added)
            if not missing:
                break
        return list(reversed(supplemental))

    # ------------------------------------------------------------------
    # RHF/RKS sub-parsers
    # ------------------------------------------------------------------

    def _parse_rhf_nbo(self, lines, nbo_start):
        data = {}

        # NAO occupancies + energies
        idx_nao = self.find_line(lines, "NATURAL POPULATIONS:  Natural atomic orbital occupancies", nbo_start)
        if idx_nao != -1:
            data["nao_occupancies"] = _parse_nao_occupancies_v2(lines, idx_nao + 3)

        # NPA summary
        idx_sum = self.find_line(lines, "Summary of Natural Population Analysis:", nbo_start)
        if idx_sum != -1:
            atoms = []
            for ln in lines[idx_sum + 4:]:
                m = re.match(
                    r"\s+(\w+)\s+(\d+)\s+([-\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
                    ln,
                )
                if m:
                    atoms.append({
                        "symbol": m.group(1), "index": int(m.group(2)),
                        "natural_charge": float(m.group(3)),
                        "core_pop": float(m.group(4)),
                        "valence_pop": float(m.group(5)),
                        "rydberg_pop": float(m.group(6)),
                        "total_pop": float(m.group(7)),
                    })
                elif "====" in ln or "Total" in ln.split()[:1]:
                    break
            if atoms:
                data["npa_summary"] = atoms

            # Totals
            for ln in lines[idx_sum: idx_sum + 60]:
                if "Core" in ln and "99." in ln:
                    m = re.match(r"\s+Core\s+([\d.]+)", ln)
                    if m:
                        data["core_population"] = float(m.group(1))
                if "Valence" in ln and "Natural Minimal" not in ln and "non-Lewis" not in ln:
                    m = re.match(r"\s+Valence\s+([\d.]+)", ln)
                    if m:
                        data["valence_population"] = float(m.group(1))
                if "Natural Rydberg" in ln:
                    m = re.search(r"Natural Rydberg Basis\s+([\d.]+)", ln)
                    if m:
                        data["rydberg_population"] = float(m.group(1))

            # Electron configurations
            idx_ecfg = self.find_line(lines, "Natural Electron Configuration", idx_sum)
            if idx_ecfg != -1:
                data["electron_configurations"] = self._parse_electron_configs(lines, idx_ecfg + 2)

        # Wiberg
        idx_wib = self.find_line(lines, "Wiberg bond index matrix in the NAO basis:", nbo_start)
        if idx_wib != -1:
            data["wiberg_bond_indices"] = _parse_wiberg(lines, idx_wib + 3)

        # NBI
        idx_nbi = self.find_line(lines, "NBI: Natural Binding Index", nbo_start)
        if idx_nbi != -1:
            data["natural_binding_index"] = self._parse_bond_matrix(lines, idx_nbi + 4)

        # NBO Lewis structure
        idx_nbo = self.find_line(lines, "NATURAL BOND ORBITAL ANALYSIS:", nbo_start)
        if idx_nbo != -1:
            data["nbo_lewis_structure"] = self._parse_nbo_lewis_summary(lines, idx_nbo)
            # Detailed orbital list
            idx_orb = self.find_line(lines, "Bond orbital / Coefficients / Hybrids", idx_nbo)
            if idx_orb != -1:
                data["nbo_orbitals"] = _parse_nbo_lewis(lines, idx_orb + 2)

        # NBO summary table
        idx_nbosumm = self.find_line(lines, "NATURAL BOND ORBITALS (Summary):", nbo_start)
        if idx_nbosumm != -1:
            data["nbo_summary"] = _parse_nbo_summary(lines, idx_nbosumm + 5)

        # NHO directionality
        idx_nho = self.find_line(lines, "NHO DIRECTIONALITY AND BOND BENDING", nbo_start)
        if idx_nho != -1:
            data["nho_directionality"] = _parse_nho_directionality(lines, idx_nho + 6)

        # NHO interhybrid angles
        idx_iha = self.find_line(lines, "NHO interhybrid", nbo_start)
        if idx_iha != -1:
            data["nho_interhybrid_angles_raw"] = []
            for ln in lines[idx_iha: idx_iha + 15]:
                if ln.strip() and "---" not in ln:
                    data["nho_interhybrid_angles_raw"].append(ln.strip())

        # CMO analysis
        idx_cmo = self.find_line(lines, "CMO: NBO Analysis of Canonical Molecular Orbitals", nbo_start)
        if idx_cmo != -1:
            data["cmo_analysis"] = _parse_cmo_analysis(lines, idx_cmo + 2)
            idx_char = self.find_line(lines, "Molecular Orbital Atom-Atom Bonding Character", idx_cmo)
            if idx_char != -1:
                data["cmo_bonding_character"] = _parse_cmo_bonding_character(lines, idx_char + 3)

        # Second-order perturbation
        idx_e2 = self.find_line(lines, "SECOND ORDER PERTURBATION THEORY ANALYSIS", nbo_start)
        if idx_e2 != -1:
            data["e2_perturbation"] = self._parse_e2_table(lines, idx_e2 + 5)

        # NLMO analysis
        idx_nlmo = self.find_line(lines, "NATURAL LOCALIZED MOLECULAR ORBITAL (NLMO) ANALYSIS", nbo_start)
        if idx_nlmo != -1:
            idx_hyb = self.find_line(lines, "Hybridization/Polarization Analysis of NLMOs", idx_nlmo)
            if idx_hyb != -1:
                data["nlmo_hybridization"] = _parse_nlmo_hybridization(lines, idx_hyb + 2)
            idx_lmob = self.find_line(lines, "Individual LMO bond orders", idx_nlmo)
            if idx_lmob != -1:
                lmo_bonds = []
                for ln in lines[idx_lmob + 2:]:
                    m = re.match(r"\s+(\d+)\s+(\d+)\s+(\d+)\s+([-\d.]+)\s+([-\d.]+)", ln)
                    if m:
                        lmo_bonds.append({
                            "atom_i": int(m.group(1)), "atom_j": int(m.group(2)),
                            "nlmo": int(m.group(3)), "bond_order": float(m.group(4)),
                            "hybrid_overlap": float(m.group(5)),
                        })
                    elif lmo_bonds and ln.strip() == "":
                        break
                if lmo_bonds:
                    data["lmo_bond_orders"] = lmo_bonds
            data["nlmo_bond_orders"] = _parse_nlmo_bond_orders(lines, idx_nlmo)

        # NLMO steric
        idx_ster = self.find_line(lines, "NBO/NLMO STERIC ANALYSIS", nbo_start)
        if idx_ster != -1:
            data["nlmo_steric"] = _parse_nlmo_steric(lines, idx_ster)

        return data

    # ------------------------------------------------------------------
    # UHF spin-resolved NBO sub-parsers
    # ------------------------------------------------------------------

    def _parse_spin_nbo(self, lines, start, spin):
        """Parse one spin NBO section (alpha or beta).

        In ORCA output, the NAO occupancies and NPA summary for each spin appear
        *before* the "NATURAL BOND ORBITAL ANALYSIS, {spin} spin" header line.
        We therefore search backward from `start` for these blocks (up to 5000 lines)
        while searching forward for everything else (NBO orbitals, E(2), NLMO...).
        """
        data = {}

        # Strategy: try forward search first (correct for alpha, where spin-specific blocks
        # appear AFTER the section banner). If nothing found, fall back to backward search
        # (needed for beta, where NAO/NPA/Wiberg blocks appear BEFORE the beta NBO header).
        pre_start = max(0, start - 5000)

        # ── NAO occupancies ────────────────────────────────────────────────────
        idx_nao = self.find_line(lines, "NATURAL POPULATIONS:  Natural atomic orbital occupancies", start)
        if idx_nao == -1:
            all_nao = [i for i in range(pre_start, start)
                       if "NATURAL POPULATIONS:  Natural atomic orbital occupancies" in lines[i]]
            if all_nao:
                idx_nao = all_nao[-1]
        if idx_nao != -1:
            data["nao_occupancies"] = _parse_nao_occupancies_v2(lines, idx_nao + 3)

        # ── NPA summary ────────────────────────────────────────────────────────
        idx_sum = self.find_line(lines, "Summary of Natural Population Analysis:", start)
        if idx_sum == -1:
            all_sum = [i for i in range(pre_start, start)
                       if "Summary of Natural Population Analysis:" in lines[i]]
            if all_sum:
                idx_sum = all_sum[-1]
        if idx_sum != -1:
            atoms = []
            for ln in lines[idx_sum + 4:]:
                m = re.match(
                    r"\s+(\w+)\s+(\d+)\s+([-\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
                    ln,
                )
                if m:
                    atoms.append({
                        "symbol": m.group(1), "index": int(m.group(2)),
                        "natural_charge": float(m.group(3)),
                        "core_pop": float(m.group(4)),
                        "valence_pop": float(m.group(5)),
                        "rydberg_pop": float(m.group(6)),
                        "total_pop": float(m.group(7)),
                    })
                elif "====" in ln:
                    break
            if atoms:
                data["npa_summary"] = atoms

            idx_ecfg = self.find_line(lines, "Natural Electron Configuration", idx_sum)
            if idx_ecfg != -1:
                data["electron_configurations"] = self._parse_electron_configs(lines, idx_ecfg + 2)

        # ── Forward search: everything else (NBO orbitals, E(2), NLMO...) ─────

        # Wiberg: try forward first (alpha), then backward (beta lays it before its NBO header)
        idx_wib = self.find_line(lines, "Wiberg bond index matrix in the NAO basis:", start)
        if idx_wib == -1:
            all_wib = [i for i in range(pre_start, start)
                       if "Wiberg bond index matrix in the NAO basis:" in lines[i]]
            if all_wib:
                idx_wib = all_wib[-1]
        if idx_wib != -1:
            data["wiberg_bond_indices"] = _parse_wiberg(lines, idx_wib + 3)

        # NBI for this spin
        idx_nbi = self.find_line(lines, "NBI: Natural Binding Index", start)
        if idx_nbi != -1:
            data["natural_binding_index"] = self._parse_bond_matrix(lines, idx_nbi + 4)

        # NBO Lewis for this spin
        nbo_key = f"NATURAL BOND ORBITAL ANALYSIS, {spin} spin orbitals:"
        idx_nbo = self.find_line(lines, nbo_key, start)
        if idx_nbo != -1:
            data["nbo_lewis_structure"] = self._parse_nbo_lewis_summary(lines, idx_nbo)
            idx_orb = self.find_line(lines, "Bond orbital / Coefficients / Hybrids", idx_nbo)
            if idx_orb != -1:
                data["nbo_orbitals"] = _parse_nbo_lewis(lines, idx_orb + 2)

        # NBO summary
        idx_nbosumm = self.find_line(lines, "NATURAL BOND ORBITALS (Summary):", idx_nbo if idx_nbo != -1 else start)
        if idx_nbosumm != -1:
            data["nbo_summary"] = _parse_nbo_summary(lines, idx_nbosumm + 5)

        # NHO
        idx_nho = self.find_line(lines, "NHO DIRECTIONALITY AND BOND BENDING", start)
        if idx_nho != -1:
            data["nho_directionality"] = _parse_nho_directionality(lines, idx_nho + 6)

        # CMO
        idx_cmo = self.find_line(lines, "CMO: NBO Analysis of Canonical Molecular Orbitals", start)
        if idx_cmo != -1:
            data["cmo_analysis"] = _parse_cmo_analysis(lines, idx_cmo + 2)
            idx_char = self.find_line(lines, "Molecular Orbital Atom-Atom Bonding Character", idx_cmo)
            if idx_char != -1:
                data["cmo_bonding_character"] = _parse_cmo_bonding_character(lines, idx_char + 3)

        # E2 perturbation
        idx_e2 = self.find_line(lines, "SECOND ORDER PERTURBATION THEORY ANALYSIS", start)
        if idx_e2 != -1:
            data["e2_perturbation"] = self._parse_e2_table(lines, idx_e2 + 5)

        # NLMO
        idx_nlmo_hyb = self.find_line(
            lines,
            f"Hybridization/Polarization Analysis of NLMOs in NAO Basis, {'Alpha' if spin == 'alpha' else 'Beta'} Spin:",
            start,
        )
        if idx_nlmo_hyb != -1:
            data["nlmo_hybridization"] = _parse_nlmo_hybridization(lines, idx_nlmo_hyb + 2)
            data["nlmo_bond_orders"] = _parse_nlmo_bond_orders(lines, idx_nlmo_hyb)
            # LMO bond orders appear after the NLMO hybridization block
            idx_lmob = self.find_line(lines, "Individual LMO bond orders", idx_nlmo_hyb)
            if idx_lmob != -1:
                lmo_bonds = []
                for ln in lines[idx_lmob + 2:]:
                    m = re.match(r"\s+(\d+)\s+(\d+)\s+(\d+)\s+([-\d.]+)\s+([-\d.]+)", ln)
                    if m:
                        lmo_bonds.append({
                            "atom_i": int(m.group(1)), "atom_j": int(m.group(2)),
                            "nlmo": int(m.group(3)), "bond_order": float(m.group(4)),
                            "hybrid_overlap": float(m.group(5)),
                        })
                    elif lmo_bonds and ln.strip() == "":
                        break
                if lmo_bonds:
                    data["lmo_bond_orders"] = lmo_bonds

        # NLMO steric (per spin unit)
        idx_ster = self.find_line(lines, "NBO/NLMO STERIC ANALYSIS", start)
        if idx_ster != -1:
            data["nlmo_steric"] = _parse_nlmo_steric(lines, idx_ster)

        return data

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _parse_nbo_lewis_summary(self, lines, start):
        """Parse the NBO Lewis structure occupancy/% summary block."""
        result = {}
        for ln in lines[start: start + 20]:
            m = re.search(r"Lewis\s+([\d.]+)\s+\(\s*([\d.]+)%", ln)
            if m:
                result["total_lewis_occupancy"] = float(m.group(1))
                result["total_lewis_pct"] = float(m.group(2))
            m2 = re.search(r"non-Lewis\s+([\d.]+)\s+\(\s*([\d.]+)%", ln)
            if m2:
                result["total_non_lewis_occupancy"] = float(m2.group(1))
                result["total_non_lewis_pct"] = float(m2.group(2))
            m3 = re.search(r"Core\s+([\d.]+)\s+\(", ln)
            if m3:
                result["core_occupancy"] = float(m3.group(1))
            m4 = re.search(r"Valence Lewis\s+([\d.]+)\s+\(", ln)
            if m4:
                result["valence_lewis_occupancy"] = float(m4.group(1))
            m5 = re.search(r"Valence non-Lewis\s+([\d.]+)\s+\(", ln)
            if m5:
                result["valence_non_lewis_occupancy"] = float(m5.group(1))
            m6 = re.search(r"Rydberg non-Lewis\s+([\d.]+)\s+\(", ln)
            if m6:
                result["rydberg_non_lewis_occupancy"] = float(m6.group(1))
        return result

    def _parse_ao_populations(self, lines, start):
        """Parse NBO Mulliken AO populations block."""
        pops = []
        for ln in lines[start:]:
            m = re.match(r"\s+(\d+)\s+(\w+)\s+(\d+)\s+(\w+)\s+([-\d.]+)", ln)
            if m:
                entry = {
                    "ao_index": int(m.group(1)),
                    "symbol": m.group(2),
                    "atom_no": int(m.group(3)),
                    "angular": m.group(4),
                    "population": float(m.group(5)),
                }
                _annotate_atom_index(entry, "atom_no", "atom")
                pops.append(entry)
            elif "* Total *" in ln or "Mayer" in ln:
                break
        return pops

    def _parse_bond_matrix(self, lines, start):
        """Generic bond order/NBI matrix parser."""
        matrix = []
        for ln in lines[start:]:
            m = re.match(r"\s+(\d+)\.\s+(\w+)((?:\s+[\d.]+)+)", ln)
            if m:
                entry = {
                    "atom": int(m.group(1)),
                    "symbol": m.group(2),
                    "values": [float(v) for v in m.group(3).split()],
                }
                _annotate_atom_index(entry, "atom", "atom")
                matrix.append(entry)
            elif matrix and ln.strip() == "":
                break
        return matrix

    def _parse_atom_valencies(self, lines, start):
        """Parse Mayer-Mulliken atomic valencies."""
        vals = []
        for ln in lines[start:]:
            m = re.match(r"\s+(\d+)\.\s+(\w+)\s+([\d.]+)", ln)
            if m:
                entry = {
                    "atom": int(m.group(1)),
                    "symbol": m.group(2),
                    "valency": float(m.group(3)),
                }
                _annotate_atom_index(entry, "atom", "atom")
                vals.append(entry)
            elif vals and ln.strip() == "":
                break
        return vals

    def _parse_electron_configs(self, lines, start):
        """Parse natural electron configuration strings."""
        configs = []
        for ln in lines[start:]:
            m = re.match(r"\s+(\w+)\s+(\d+)\s+(.+)", ln)
            if m:
                entry = {
                    "symbol": m.group(1),
                    "index": int(m.group(2)),
                    "configuration": m.group(3).strip(),
                }
                _annotate_atom_index(entry, "index", "atom")
                configs.append(entry)
            elif ln.strip() == "" and configs:
                break
        return configs

    def _parse_npa_uhf(self, lines, start):
        """Parse NPA summary for UHF (includes spin density column)."""
        atoms = []
        for ln in lines[start:]:
            m = re.match(
                r"\s+(\w+)\s+(\d+)\s+([-\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([-\d.]+)",
                ln,
            )
            if m:
                entry = {
                    "symbol": m.group(1), "index": int(m.group(2)),
                    "natural_charge": float(m.group(3)),
                    "core_pop": float(m.group(4)),
                    "valence_pop": float(m.group(5)),
                    "rydberg_pop": float(m.group(6)),
                    "total_pop": float(m.group(7)),
                    "spin_density": float(m.group(8)),
                }
                _annotate_atom_index(entry, "index", "atom")
                atoms.append(entry)
            elif "====" in ln:
                break
        return atoms

    def _parse_nao_with_spin(self, lines, start):
        """Parse NAO block that has Occupancy + Spin columns (UHF overall)."""
        naos = []
        for ln in lines[start:]:
            m = re.match(
                r"\s+(\d+)\s+(\w+)\s+(\d+)\s+(\S+)\s+(\S+\([^)]+\))\s+([-\d.]+)\s+([-\d.]+)",
                ln,
            )
            if m:
                entry = {
                    "index": int(m.group(1)), "symbol": m.group(2),
                    "atom_no": int(m.group(3)), "angular": m.group(4),
                    "type": m.group(5),
                    "occupancy": float(m.group(6)),
                    "spin": float(m.group(7)),
                }
                _annotate_atom_index(entry, "atom_no", "atom")
                naos.append(entry)
            elif naos and ("Summary" in ln or "Wiberg" in ln or "Alpha spin" in ln):
                break
        return naos

    def _parse_e2_table(self, lines, start):
        """Parse the E2 perturbation table.

        Handles multiple NBO formats:
          - Simple LP/CR/RY:  '101. LP ( 2)Bi  1'
          - Bond BD/BD*:      '224. BD*( 1) C  2- C  3'
          - Water-style:      '2. LP ( 1) O  1'
        """
        _NBO = r"\d+\.\s+\w+\*?\s*\(\s*\d+\)\s*\w+\s+\d+(?:\s*-\s*\w+\s+\d+)?"
        _pat = re.compile(
            r"\s+(" + _NBO + r")\s+(" + _NBO + r")\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
        )
        entries = []
        for ln in lines[start:]:
            m = _pat.match(ln)
            if m:
                entries.append({
                    "donor":       m.group(1).strip(),
                    "acceptor":    m.group(2).strip(),
                    "E2_kcal_mol": float(m.group(3)),
                    "E_gap_au":    float(m.group(4)),
                    "Fock_au":     float(m.group(5)),
                })
            elif "NATURAL BOND ORBITALS" in ln or "NATURAL LOCALIZED" in ln:
                break
        return entries

    def _parse_npepa(self, lines, start):
        """Parse NPEPA (Natural Poly-Electron Population Analysis) block."""
        data = {}
        entries = []
        for ln in lines[start: start + 20]:
            m = re.match(r"\s+([\d.]+)\s+(<.+>)", ln)
            if m:
                entries.append({"probability": float(m.group(1)), "determinant": m.group(2)})
        if entries:
            data["determinants"] = entries
        return data


PLUGIN_BUNDLE = PluginBundle(
    metadata=PluginMetadata(
        key="nbo_section",
        name="NBO Section",
        short_help="Built-in NBO parser section owned by nbo.py.",
        description=(
            "Self-registering built-in parser section for NBO analysis, "
            "including NPA, NBO/NLMO character, perturbation analysis, and "
            "related orbital-composition tables."
        ),
        docs_path="README.md",
        examples=(
            "orca_parser job.out --sections nbo",
        ),
    ),
    parser_sections=(
        ParserSectionPlugin("nbo", NBOModule),
    ),
    parser_aliases=(
        ParserSectionAlias(name="nbo", section_keys=("nbo",)),
    ),
    csv_sections=(
        NBO_NAO_CSV_SECTION_PLUGIN,
        NBO_NPA_CSV_SECTION_PLUGIN,
        NBO_LEWIS_CSV_SECTION_PLUGIN,
        NBO_E2_CSV_SECTION_PLUGIN,
        NBO_NLMO_HYBRIDIZATION_CSV_SECTION_PLUGIN,
        NBO_NLMO_BOND_ORDER_CSV_SECTION_PLUGIN,
        NBO_NLMO_STERIC_CSV_SECTION_PLUGIN,
        WIBERG_MATRIX_CSV_SECTION_PLUGIN,
        NBI_MATRIX_CSV_SECTION_PLUGIN,
    ),
)
