"""
Population analysis modules: Mulliken, Loewdin, Mayer, Hirshfeld, MBIS, CHELPG.
"""

import re
from typing import Any, Dict, Iterable, List, Optional

from .base import BaseModule


# ---------------------------------------------------------------------------
# Helper: parse a block of "atom charge [spin]" rows
# ---------------------------------------------------------------------------

def _parse_atomic_values(lines: List[str], start: int, n_cols: int = 1):
    """
    Parse rows like:
      " 0 O :  -0.663829"                      (n_cols=1 → charge)
      " 0 O :  -0.663829   0.000000"            (n_cols=2 → charge, spin)
    Returns list of dicts.
    """
    atoms = []
    for ln in lines[start:]:
        # Match: index  Symbol  :  val1  [val2]
        m = re.match(r"\s+(\d+)\s+(\w+)\s*:\s+([-\d.]+)(?:\s+([-\d.]+))?", ln)
        if m:
            entry = {"index": int(m.group(1)), "symbol": m.group(2), "charge": float(m.group(3))}
            if n_cols == 2 and m.group(4):
                entry["spin_population"] = float(m.group(4))
            atoms.append(entry)
        elif ln.strip().startswith("Sum") or ln.strip() == "":
            if atoms:
                break
    return atoms


def _find_exact_label(lines: List[str], label: str, start: int = 0) -> int:
    """Return the index of the first line whose stripped uppercase text equals *label*."""
    target = label.strip().upper()
    for i in range(start, len(lines)):
        if lines[i].strip().upper() == target:
            return i
    return -1


def _parse_reduced_orbital_charges(
    lines: List[str],
    start: int,
    stop_exact: Optional[Iterable[str]] = None,
) -> List[Dict]:
    """Parse reduced orbital charge blocks per atom (Mulliken or Loewdin)."""
    atom_blocks = []
    current_atom = None
    exact_stops = {marker.strip().upper() for marker in (stop_exact or [])}

    for ln in lines[start:]:
        stripped = ln.strip()
        upper = stripped.upper()

        if upper in exact_stops:
            if current_atom is not None:
                atom_blocks.append(current_atom)
            break

        # New atom header, with or without a space between a two-letter symbol and the shell label:
        # "  0 O s       :     3.806774  s :     3.806774"
        # "  0 Bis       :    12.117041  s :    12.117041"
        m = re.match(
            r"\s+(?P<index>\d+)\s+"
            r"(?:(?P<symbol>[A-Z][a-z]?)\s+(?P<orbital>\w+)|(?P<compact>[A-Z][a-z]?)(?P<compact_orbital>[spdfgh]))"
            r"\s+:\s+(?P<population>[-\d.]+)\s+(?P<shell>\w+)\s+:\s+(?P<shell_total>[-\d.]+)",
            ln,
        )
        if m:
            if current_atom is not None:
                atom_blocks.append(current_atom)
            symbol = m.group("symbol") or m.group("compact")
            orbital = m.group("orbital") or m.group("compact_orbital")
            current_atom = {
                "index": int(m.group("index")),
                "symbol": symbol,
                "orbitals": [],
                "shell_totals": {},
            }
            current_atom["orbitals"].append({
                "orbital": orbital,
                "population": float(m.group("population")),
            })
            current_atom["shell_totals"][m.group("shell")] = float(m.group("shell_total"))
            continue

        # Continuation: "      pz      :     1.879214  p :     4.834638"
        m2 = re.match(r"\s+(\w+\d*\*?)\s+:\s+([-\d.]+)\s+(\w+)\s+:\s+([-\d.]+)", ln)
        if m2 and current_atom is not None:
            current_atom["orbitals"].append({"orbital": m2.group(1), "population": float(m2.group(2))})
            current_atom["shell_totals"][m2.group(3)] = float(m2.group(4))
            continue

        # Orbital-only line (within shell): "      pz      :     1.879214"
        m3 = re.match(r"\s+(\w+\d*[\+\-]?\d*)\s+:\s+([-\d.]+)\s*$", ln)
        if m3 and current_atom is not None:
            current_atom["orbitals"].append({"orbital": m3.group(1), "population": float(m3.group(2))})
            continue

        # Blank line or heading ends block
        if ln.strip() == "" and current_atom is not None:
            atom_blocks.append(current_atom)
            current_atom = None
        if (
            "LOEWDIN" in ln
            or "MAYER" in ln
            or "HIRSHFELD" in ln
            or "CHELPG" in ln
            or "MBIS" in ln
            or "ELECTROSTATIC POTENTIAL" in ln
        ):
            if current_atom is not None:
                atom_blocks.append(current_atom)
            break

    if current_atom is not None:
        atom_blocks.append(current_atom)

    return atom_blocks


# ---------------------------------------------------------------------------
# Mulliken
# ---------------------------------------------------------------------------

class MullikenModule(BaseModule):
    """
    Mulliken population analysis.
    Handles both RHF (charges only) and UHF (charges + spin populations).
    """

    name = "mulliken"

    def parse(self, lines):
        data = {}
        is_uhf = self.context.get("is_uhf", False)

        # --- Atomic charges (and spin for UHF) ---
        if is_uhf:
            header = "MULLIKEN ATOMIC CHARGES AND SPIN POPULATIONS"
        else:
            header = "MULLIKEN ATOMIC CHARGES"

        idx = self.find_line(lines, header)
        if idx == -1:
            return None

        # Skip separator line
        atoms = _parse_atomic_values(lines, idx + 2, n_cols=2 if is_uhf else 1)
        data["atomic_charges"] = atoms

        # Sum check
        for ln in lines[idx: idx + len(atoms) + 10]:
            m = re.search(r"Sum of atomic charges\s*:\s*([-\d.e+]+)", ln)
            if m:
                data["sum_of_charges"] = float(m.group(1))

        # --- Reduced orbital charges ---
        idx_red = self.find_line(lines, "MULLIKEN REDUCED ORBITAL CHARGES")
        if idx_red != -1:
            idx_charge = _find_exact_label(lines, "CHARGE", idx_red)
            charge_start = idx_charge + 1 if idx_charge != -1 else idx_red + 2
            blocks = _parse_reduced_orbital_charges(
                lines,
                charge_start,
                stop_exact={"SPIN"},
            )
            if is_uhf:
                if blocks:
                    data["reduced_orbital_charges"] = blocks
                idx_spin = _find_exact_label(lines, "SPIN", idx_red)
                if idx_spin != -1:
                    spin_blocks = _parse_reduced_orbital_charges(lines, idx_spin + 1)
                    if spin_blocks:
                        data["reduced_orbital_spin_populations"] = spin_blocks
            else:
                if blocks:
                    data["reduced_orbital_charges"] = blocks

        # --- Frontier MO population analysis ---
        idx_fmo = self.find_line(lines, "FRONTIER MOLECULAR ORBITAL POPULATION ANALYSIS")
        if idx_fmo != -1:
            fmo_data = []
            for ln in lines[idx_fmo + 5:]:
                # Match atom line: "  0-O  0.983671  0.954178  -0.352265  0.445586"
                m = re.match(
                    r"\s+(\d+)-(\w+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)",
                    ln
                )
                if m:
                    fmo_data.append({
                        "index": int(m.group(1)),
                        "symbol": m.group(2),
                        "HOMO_Mulliken": float(m.group(3)),
                        "HOMO_Loewdin": float(m.group(4)),
                        "LUMO_Mulliken": float(m.group(5)),
                        "LUMO_Loewdin": float(m.group(6)),
                    })
                elif "---" in ln and fmo_data:
                    break
            if fmo_data:
                data["frontier_mo_population"] = fmo_data

            # For UHF, there may be HOMO/LUMO per operator
            if is_uhf:
                for idx_op in self.find_all_lines(lines, "OPERATOR OP ="):
                    m = re.search(r"OPERATOR OP = (\d+): HOMO=\s*(\d+) LUMO=\s*(\d+)", lines[idx_op])
                    if m:
                        op = int(m.group(1))
                        spin = "alpha" if op == 0 else "beta"
                        key = f"{spin}_frontier_mo"
                        data[key] = {"HOMO": int(m.group(2)), "LUMO": int(m.group(3))}
                        # Parse atom contributions for this operator
                        fmo_op = []
                        for ln in lines[idx_op + 4:]:
                            m2 = re.match(
                                r"\s+(\d+)-(\w+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)",
                                ln
                            )
                            if m2:
                                fmo_op.append({
                                    "index": int(m2.group(1)),
                                    "symbol": m2.group(2),
                                    "HOMO_Mulliken": float(m2.group(3)),
                                    "HOMO_Loewdin": float(m2.group(4)),
                                    "LUMO_Mulliken": float(m2.group(5)),
                                    "LUMO_Loewdin": float(m2.group(6)),
                                })
                            elif "---" in ln and fmo_op:
                                break
                        if fmo_op:
                            data[f"{spin}_frontier_mo_population"] = fmo_op

        return data if data else None


# ---------------------------------------------------------------------------
# Loewdin
# ---------------------------------------------------------------------------

class LoewdinModule(BaseModule):
    """Loewdin population analysis (charges, spin populations, reduced orbital charges)."""

    name = "loewdin"

    def parse(self, lines):
        data = {}
        is_uhf = self.context.get("is_uhf", False)

        header = "LOEWDIN ATOMIC CHARGES AND SPIN POPULATIONS" if is_uhf else "LOEWDIN ATOMIC CHARGES"
        idx = self.find_line(lines, header)
        if idx == -1:
            return None

        atoms = _parse_atomic_values(lines, idx + 2, n_cols=2 if is_uhf else 1)
        data["atomic_charges"] = atoms

        idx_red = self.find_line(lines, "LOEWDIN REDUCED ORBITAL CHARGES")
        if idx_red != -1:
            idx_charge = _find_exact_label(lines, "CHARGE", idx_red)
            charge_start = idx_charge + 1 if idx_charge != -1 else idx_red + 2
            blocks = _parse_reduced_orbital_charges(
                lines,
                charge_start,
                stop_exact={"SPIN"},
            )
            if is_uhf:
                if blocks:
                    data["reduced_orbital_charges"] = blocks
                idx_spin = _find_exact_label(lines, "SPIN", idx_red)
                if idx_spin != -1:
                    spin_blocks = _parse_reduced_orbital_charges(lines, idx_spin + 1)
                    if spin_blocks:
                        data["reduced_orbital_spin_populations"] = spin_blocks
            else:
                if blocks:
                    data["reduced_orbital_charges"] = blocks

        return data if data else None


# ---------------------------------------------------------------------------
# Mayer
# ---------------------------------------------------------------------------

class MayerModule(BaseModule):
    """
    Mayer population analysis: atomic data (NA, ZA, QA, VA, BVA, FA) and bond orders.
    """

    name = "mayer"

    def parse(self, lines):
        data = {}

        idx = self.find_line(lines, "MAYER POPULATION ANALYSIS")
        if idx == -1:
            return None

        # Atomic data table
        atoms = []
        for ln in lines[idx:]:
            m = re.match(
                r"\s+(\d+)\s+(\w+)\s+([\d.]+)\s+([\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)",
                ln,
            )
            if m:
                atoms.append({
                    "index": int(m.group(1)),
                    "symbol": m.group(2),
                    "NA": float(m.group(3)),
                    "ZA": float(m.group(4)),
                    "QA": float(m.group(5)),
                    "VA_total_valence": float(m.group(6)),
                    "BVA_bonded_valence": float(m.group(7)),
                    "FA_free_valence": float(m.group(8)),
                })
            elif atoms and ln.strip() == "":
                break
        data["atomic_data"] = atoms

        # Bond orders
        idx_bo = self.find_line(lines, "Mayer bond orders larger than", idx)
        if idx_bo != -1:
            bond_orders = []
            for ln in lines[idx_bo + 1:]:
                # "B(  0-O ,  1-H ) :   0.8983 B(  0-O ,  2-H ) :   0.8983"
                # Three bonds per line in large systems
                found = False
                for m in re.finditer(
                    r"B\(\s*(\d+)-(\w+)\s*,\s*(\d+)-(\w+)\s*\)\s*:\s*([-\d.]+)",
                    ln,
                ):
                    bond_orders.append({
                        "atom_i": int(m.group(1)),
                        "symbol_i": m.group(2),
                        "atom_j": int(m.group(3)),
                        "symbol_j": m.group(4),
                        "bond_order": float(m.group(5)),
                    })
                    found = True
                if not found and ln.strip() == "":
                    break
            data["bond_orders"] = bond_orders

        return data if data else None


# ---------------------------------------------------------------------------
# Hirshfeld
# ---------------------------------------------------------------------------

class HirshfeldModule(BaseModule):
    """Hirshfeld charge and spin population analysis."""

    name = "hirshfeld"

    def parse(self, lines):
        data = {}
        idx = self.find_line(lines, "HIRSHFELD ANALYSIS")
        if idx == -1:
            return None

        # Integrated densities
        for ln in lines[idx: idx + 8]:
            m = re.search(r"Total integrated alpha density\s+=\s+([\d.]+)", ln)
            if m:
                data["total_alpha_density"] = float(m.group(1))
            m = re.search(r"Total integrated beta density\s+=\s+([\d.]+)", ln)
            if m:
                data["total_beta_density"] = float(m.group(1))

        atoms = []
        # Start search from idx+5 to skip header lines; read until TOTAL or blank
        for ln in lines[idx + 5:]:
            m = re.match(r"\s+(\d+)\s+(\w+)\s+([-\d.]+)\s+([-\d.]+)", ln)
            if m:
                atoms.append({
                    "index": int(m.group(1)),
                    "symbol": m.group(2),
                    "charge": float(m.group(3)),
                    "spin_population": float(m.group(4)),
                })
            elif "TOTAL" in ln:
                m2 = re.search(r"TOTAL\s+([-\d.]+)\s+([-\d.]+)", ln)
                if m2:
                    data["total_charge"] = float(m2.group(1))
                    data["total_spin"] = float(m2.group(2))
                break
            elif atoms and ln.strip() == "":
                break

        data["atomic_data"] = atoms
        return data if data else None


# ---------------------------------------------------------------------------
# MBIS
# ---------------------------------------------------------------------------

class MBISModule(BaseModule):
    """MBIS (Minimal Basis Iterative Stockholder) charge analysis."""

    name = "mbis"

    def parse(self, lines):
        data = {}
        idx = self.find_line(lines, "MBIS ANALYSIS")
        if idx == -1:
            return None

        for ln in lines[idx: idx + 10]:
            m = re.search(r"Number of iterations\s+\.\.\.\s+(\d+)", ln)
            if m:
                data["n_iterations"] = int(m.group(1))
            m = re.search(r"Total integrated alpha density\s+\.\.\.\s+([\d.]+)", ln)
            if m:
                data["total_alpha_density"] = float(m.group(1))
            m = re.search(r"Total integrated beta density\s+\.\.\.\s+([\d.]+)", ln)
            if m:
                data["total_beta_density"] = float(m.group(1))

        # Atomic data: charge, population, spin — read until TOTAL line
        atoms = []
        for ln in lines[idx + 8:]:
            m = re.match(r"\s+(\d+)\s+(\w+)\s+([-\d.]+)\s+([\d.]+)\s+([-\d.]+)", ln)
            if m:
                atoms.append({
                    "index": int(m.group(1)),
                    "symbol": m.group(2),
                    "charge": float(m.group(3)),
                    "population": float(m.group(4)),
                    "spin_population": float(m.group(5)),
                })
            elif "TOTAL" in ln:
                m2 = re.match(r"\s+TOTAL\s+([-\d.]+)\s+([\d.]+)\s+([-\d.]+)", ln)
                if m2:
                    data["total_charge"] = float(m2.group(1))
                    data["total_population"] = float(m2.group(2))
                    data["total_spin"] = float(m2.group(3))
                break
        data["atomic_data"] = atoms

        # Valence shell data
        idx_vs = self.find_line(lines, "MBIS VALENCE-SHELL DATA", idx)
        if idx_vs != -1:
            valence = []
            for ln in lines[idx_vs + 2:]:
                m = re.match(r"\s+(\d+)\s+(\w+)\s+([\d.]+)\s+([\d.]+)", ln)
                if m:
                    valence.append({
                        "index": int(m.group(1)),
                        "symbol": m.group(2),
                        "population": float(m.group(3)),
                        "width_au": float(m.group(4)),
                    })
                elif ln.strip() == "" and valence:
                    break
            data["valence_shell"] = valence

        return data if data else None


# ---------------------------------------------------------------------------
# CHELPG
# ---------------------------------------------------------------------------

class CHELPGModule(BaseModule):
    """CHELPG electrostatic potential charges."""

    name = "chelpg"

    def parse(self, lines):
        data = {}
        idx = self.find_line(lines, "CHELPG CHARGES GENERATION")
        if idx == -1:
            return None

        atoms = []
        in_charges = False
        for ln in lines[idx:]:
            if "CHELPG Charges" in ln:
                in_charges = True
                continue
            if in_charges:
                m = re.match(r"\s+(\d+)\s+(\w+)\s+:\s+([-\d.]+)", ln)
                if m:
                    atoms.append({
                        "index": int(m.group(1)),
                        "symbol": m.group(2),
                        "charge": float(m.group(3)),
                    })
                m2 = re.search(r"Total charge:\s+([-\d.e+]+)", ln)
                if m2:
                    data["total_charge"] = float(m2.group(1))
                    break

        data["atomic_charges"] = atoms
        return data if data else None
