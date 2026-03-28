"""
Module for orbital energies.

Handles:
- RHF/RKS: single set of orbitals (with optional irrep labels under symmetry)
- UHF/UKS: alpha and beta orbital sets (with optional irrep labels)
- QRO (Quasi-Restricted Orbitals) for UHF/UKS
- UNO natural orbital occupancies
"""

import re
from typing import Any, Dict, List, Optional

from .base import BaseModule


def _parse_orbital_block(lines: List[str], start: int) -> List[Dict]:
    """Parse a block of orbital energies starting after the header line."""
    orbitals = []
    for ln in lines[start:]:
        # Match: index  occ  E(Eh)  E(eV)  [Irrep]
        m = re.match(
            r"\s+(\d+)\s+([\d.]+)\s+([-\d.]+)\s+([-\d.]+)(?:\s+(\S+))?",
            ln,
        )
        if m:
            entry = {
                "index": int(m.group(1)),
                "occupation": float(m.group(2)),
                "energy_Eh": float(m.group(3)),
                "energy_eV": float(m.group(4)),
            }
            if m.group(5):
                entry["irrep"] = m.group(5)
            orbitals.append(entry)
        elif orbitals and (ln.strip() == "" or "MOLECULAR ORBITALS" in ln or "---" in ln):
            break
    return orbitals


def _find_last_line(lines: List[str], needle: str) -> int:
    """Return the last line index containing *needle*, or -1 if absent."""
    for idx in range(len(lines) - 1, -1, -1):
        if needle in lines[idx]:
            return idx
    return -1


def _parse_irrep_occupations(lines: List[str], start: int) -> Dict[str, int]:
    """Parse occupied orbital counts per irrep from a header line index."""
    occupations: Dict[str, int] = {}
    for ln in lines[start + 1:]:
        m = re.match(r"^\s*(\w+)\s+-\s+(\d+)", ln)
        if m:
            occupations[m.group(1)] = int(m.group(2))
        elif occupations:
            break
    return occupations


class OrbitalEnergiesModule(BaseModule):
    """
    Extracts orbital energies.

    For RHF/RKS: data["orbitals"] = list of {index, occupation, energy_Eh, energy_eV[, irrep]}
    For UHF/UKS: data["alpha_orbitals"] and data["beta_orbitals"]

    Also records HOMO/LUMO indices and energies.
    """

    name = "orbital_energies"

    def parse(self, lines):
        data = {}
        is_uhf = self.context.get("is_uhf", False)

        idx = self.find_line(lines, "ORBITAL ENERGIES")
        if idx == -1:
            return None

        if not is_uhf:
            # RHF: single block, possibly with Irrep column
            # Header: NO  OCC  E(Eh)  E(eV)  [Irrep]
            block_start = idx + 3  # skip 'ORBITAL ENERGIES', '----------------', blank
            # Find actual data start
            for i in range(idx + 1, min(idx + 6, len(lines))):
                if re.match(r"\s+\d+\s+[\d.]+\s+[-\d.]+", lines[i]):
                    block_start = i
                    break
            orbitals = _parse_orbital_block(lines, block_start)
            data["orbitals"] = orbitals

            # Determine HOMO/LUMO
            homo_idx = None
            lumo_idx = None
            for orb in orbitals:
                if orb["occupation"] > 0.5:
                    homo_idx = orb["index"]
                else:
                    if homo_idx is not None and lumo_idx is None:
                        lumo_idx = orb["index"]

            if homo_idx is not None:
                homo = next(o for o in orbitals if o["index"] == homo_idx)
                data["HOMO_index"] = homo_idx
                data["HOMO_energy_Eh"] = homo["energy_Eh"]
                data["HOMO_energy_eV"] = homo["energy_eV"]
                if "irrep" in homo:
                    data["HOMO_irrep"] = homo["irrep"]

            if lumo_idx is not None:
                lumo = next(o for o in orbitals if o["index"] == lumo_idx)
                data["LUMO_index"] = lumo_idx
                data["LUMO_energy_Eh"] = lumo["energy_Eh"]
                data["LUMO_energy_eV"] = lumo["energy_eV"]
                if "irrep" in lumo:
                    data["LUMO_irrep"] = lumo["irrep"]

            if homo_idx is not None and lumo_idx is not None:
                data["HOMO_LUMO_gap_eV"] = data["LUMO_energy_eV"] - data["HOMO_energy_eV"]
                data["HOMO_LUMO_gap_Eh"] = data["LUMO_energy_Eh"] - data["HOMO_energy_Eh"]

        else:
            # UHF: SPIN UP block, then SPIN DOWN block
            idx_up = self.find_line(lines, "SPIN UP ORBITALS", idx)
            idx_dn = self.find_line(lines, "SPIN DOWN ORBITALS", idx)

            if idx_up != -1:
                # Start parsing after the header line
                for i in range(idx_up + 1, min(idx_up + 6, len(lines))):
                    if re.match(r"\s+\d+\s+[\d.]+\s+[-\d.]+", lines[i]):
                        alpha_start = i
                        break
                else:
                    alpha_start = idx_up + 3
                alpha_orbs = _parse_orbital_block(lines, alpha_start)
                data["alpha_orbitals"] = alpha_orbs

                # Alpha HOMO/LUMO
                alpha_homo = alpha_lumo = None
                for orb in alpha_orbs:
                    if orb["occupation"] > 0.5:
                        alpha_homo = orb
                    elif alpha_homo is not None and alpha_lumo is None:
                        alpha_lumo = orb
                if alpha_homo:
                    data["alpha_HOMO_index"] = alpha_homo["index"]
                    data["alpha_HOMO_energy_Eh"] = alpha_homo["energy_Eh"]
                    data["alpha_HOMO_energy_eV"] = alpha_homo["energy_eV"]
                    if "irrep" in alpha_homo:
                        data["alpha_HOMO_irrep"] = alpha_homo["irrep"]
                if alpha_lumo:
                    data["alpha_LUMO_index"] = alpha_lumo["index"]
                    data["alpha_LUMO_energy_Eh"] = alpha_lumo["energy_Eh"]
                    data["alpha_LUMO_energy_eV"] = alpha_lumo["energy_eV"]
                    if "irrep" in alpha_lumo:
                        data["alpha_LUMO_irrep"] = alpha_lumo["irrep"]

            if idx_dn != -1:
                for i in range(idx_dn + 1, min(idx_dn + 6, len(lines))):
                    if re.match(r"\s+\d+\s+[\d.]+\s+[-\d.]+", lines[i]):
                        beta_start = i
                        break
                else:
                    beta_start = idx_dn + 3
                beta_orbs = _parse_orbital_block(lines, beta_start)
                data["beta_orbitals"] = beta_orbs

                # Beta HOMO/LUMO
                beta_homo = beta_lumo = None
                for orb in beta_orbs:
                    if orb["occupation"] > 0.5:
                        beta_homo = orb
                    elif beta_homo is not None and beta_lumo is None:
                        beta_lumo = orb
                if beta_homo:
                    data["beta_HOMO_index"] = beta_homo["index"]
                    data["beta_HOMO_energy_Eh"] = beta_homo["energy_Eh"]
                    data["beta_HOMO_energy_eV"] = beta_homo["energy_eV"]
                    if "irrep" in beta_homo:
                        data["beta_HOMO_irrep"] = beta_homo["irrep"]
                if beta_lumo:
                    data["beta_LUMO_index"] = beta_lumo["index"]
                    data["beta_LUMO_energy_Eh"] = beta_lumo["energy_Eh"]
                    data["beta_LUMO_energy_eV"] = beta_lumo["energy_eV"]
                    if "irrep" in beta_lumo:
                        data["beta_LUMO_irrep"] = beta_lumo["irrep"]

        # Occupied orbitals per irrep (symmetry only)
        idx_occ = _find_last_line(lines, "Number of occupied orbitals per irrep")
        if idx_occ == -1:
            idx_occ = _find_last_line(lines, "Number of occupied orbitals per irrep of operator 0")
        if idx_occ != -1:
            irrep_occ = _parse_irrep_occupations(lines, idx_occ)
            if irrep_occ:
                data["occupied_per_irrep"] = irrep_occ

        # Also grab alpha/beta per irrep for UHF
        if is_uhf:
            idx_op0 = _find_last_line(lines, "Number of occupied orbitals per irrep of operator 0")
            idx_op1 = _find_last_line(lines, "Number of occupied orbitals per irrep of operator 1")
            if idx_op0 != -1:
                occ_alpha = _parse_irrep_occupations(lines, idx_op0)
                if occ_alpha:
                    data["alpha_occupied_per_irrep"] = occ_alpha
                    data["occupied_per_irrep"] = occ_alpha
            if idx_op1 != -1:
                occ_beta = _parse_irrep_occupations(lines, idx_op1)
                if occ_beta:
                    data["beta_occupied_per_irrep"] = occ_beta

        return data if data else None


class QROModule(BaseModule):
    """
    Extracts Quasi-Restricted Orbital (QRO) data for UHF/UKS calculations.

    The QRO scheme (Neese, 2005) generates a set of spatial orbitals from the
    alpha and beta density matrices:
      - DOMO (doubly-occupied): occupation 2
      - SOMO (singly-occupied): occupation 1, with individual alpha/beta energies
      - VMO (virtual): occupation 0

    Output format per orbital:
        index  (occ)  : energy_Eh  energy_eV  [alpha_eV=  beta_eV=]
    """

    name = "qro"

    def parse(self, lines):
        if not self.context.get("is_uhf", False):
            return None

        data = {}

        idx = self.find_line(lines, "Orbital Energies of Quasi-Restricted MO")
        if idx == -1:
            return None

        # Optional irrep field between occupation and colon:
        #   no-sym: "4( 1) :  ..."
        #   sym:    "4( 1)   1- B1:  ..."
        _irrep = r"(?:\s+(?P<irrep>\d+-\s*\S+))?"

        orbitals = []
        for ln in lines[idx + 1:]:
            # SOMO first (alpha=/beta= suffix present)
            m_somo = re.match(
                r"\s+(?P<idx>\d+)\(\s*(?P<occ>\d+)\)" + _irrep
                + r"\s*:\s+(?P<Eh>[-\d.]+)\s+a\.u\.\s+(?P<eV>[-\d.]+)\s+eV"
                r"\s+alpha=\s*(?P<aeV>[-\d.]+)\s+beta=\s*(?P<beV>[-\d.]+)",
                ln,
            )
            if m_somo:
                entry = {
                    "index": int(m_somo.group("idx")),
                    "occupation": int(m_somo.group("occ")),
                    "energy_Eh": float(m_somo.group("Eh")),
                    "energy_eV": float(m_somo.group("eV")),
                    "alpha_energy_eV": float(m_somo.group("aeV")),
                    "beta_energy_eV": float(m_somo.group("beV")),
                    "type": "SOMO",
                }
                if m_somo.group("irrep"):
                    entry["irrep"] = m_somo.group("irrep").strip()
                orbitals.append(entry)
                continue

            # DOMO / VMO
            m_plain = re.match(
                r"\s+(?P<idx>\d+)\(\s*(?P<occ>\d+)\)" + _irrep
                + r"\s*:\s+(?P<Eh>[-\d.]+)\s+a\.u\.\s+(?P<eV>[-\d.]+)\s+eV\s*$",
                ln,
            )
            if m_plain:
                occ = int(m_plain.group("occ"))
                entry = {
                    "index": int(m_plain.group("idx")),
                    "occupation": occ,
                    "energy_Eh": float(m_plain.group("Eh")),
                    "energy_eV": float(m_plain.group("eV")),
                    "type": "DOMO" if occ == 2 else "VMO",
                }
                if m_plain.group("irrep"):
                    entry["irrep"] = m_plain.group("irrep").strip()
                orbitals.append(entry)
                continue

            # Blank line ends the block
            if ln.strip() == "" and orbitals:
                break

        if orbitals:
            data["orbitals"] = orbitals
            data["n_domo"] = sum(1 for o in orbitals if o["type"] == "DOMO")
            data["n_somo"] = sum(1 for o in orbitals if o["type"] == "SOMO")
            data["n_vmo"] = sum(1 for o in orbitals if o["type"] == "VMO")

            # Collect SOMO details
            somos = [o for o in orbitals if o["type"] == "SOMO"]
            if somos:
                data["somo_details"] = somos

        # UNO natural orbital check (just flag that file was written)
        if self.find_line(lines, "UHF Natural Orbitals were saved") != -1:
            data["uno_file_written"] = True
        if self.find_line(lines, "UHF Natural Spin-Orbitals were saved") != -1:
            data["unso_file_written"] = True

        return data if data else None
