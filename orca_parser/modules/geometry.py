"""
Modules for calculation metadata, geometry, and basis set information.
"""

import re
from typing import Any, Dict, Optional

from .base import BaseModule


class MetadataModule(BaseModule):
    """Extracts program version, job name, host, date, input keywords, etc."""

    name = "metadata"

    def parse(self, lines):
        data = {}

        # ORCA version
        for ln in lines:
            m = re.search(r"Program Version\s+([\d.]+)", ln)
            if m:
                data["orca_version"] = m.group(1)
                break

        # Job name from input filename
        for ln in lines:
            m = re.search(r"NAME\s*=\s*(\S+)", ln)
            if m:
                data["job_name"] = m.group(1).replace(".inp", "")
                break

        # Host, date, working directory
        for ln in lines:
            m = re.search(r"Host name:\s+(\S+)", ln)
            if m:
                data["host"] = m.group(1)
            m = re.search(r"Starting time:\s+(.+)", ln)
            if m:
                data["start_time"] = m.group(1).strip()
            m = re.search(r"Working dir\.:\s+(.+)", ln)
            if m:
                data["working_dir"] = m.group(1).strip()

        # Calculation type
        for ln in lines:
            if "Single Point Calculation" in ln:
                data["calculation_type"] = "Single Point"
                break
            elif "Geometry Optimization" in ln:
                data["calculation_type"] = "Geometry Optimization"
                break
            elif "Frequency Calculation" in ln:
                data["calculation_type"] = "Frequency"
                break

        # SCF type (RHF/UHF)
        for ln in lines:
            m = re.search(r"Hartree-Fock type\s+HFTyp\s+\.\.\.\.\s+(\w+)", ln)
            if m:
                data["hf_type"] = m.group(1)
                break

        # Functional
        for ln in lines:
            m = re.search(r"Functional name\s+\.\.\.\.\s+(.+)", ln)
            if m:
                data["functional"] = m.group(1).strip()
                break

        # Charge and multiplicity
        for ln in lines:
            m = re.search(r"Total Charge\s+Charge\s+\.\.\.\.\s+(-?\d+)", ln)
            if m:
                data["charge"] = int(m.group(1))
            m = re.search(r"Multiplicity\s+Mult\s+\.\.\.\.\s+(\d+)", ln)
            if m:
                data["multiplicity"] = int(m.group(1))
            m = re.search(r"Number of Electrons\s+NEL\s+\.\.\.\.\s+(\d+)", ln)
            if m:
                data["n_electrons"] = int(m.group(1))

        # Basis set
        for ln in lines:
            m = re.search(r"Your calculation utilizes the basis:\s+(.+)", ln)
            if m:
                data["basis_set"] = m.group(1).strip()
                break

        # Basis dimension
        for ln in lines:
            m = re.search(r"Basis Dimension\s+Dim\s+\.\.\.\.\s+(\d+)", ln)
            if m:
                data["n_basis_functions"] = int(m.group(1))
                break

        # Nuclear repulsion
        for ln in lines:
            m = re.search(r"Nuclear Repulsion\s+ENuc\s+\.\.\.\.\s+([\d.]+)", ln)
            if m:
                data["nuclear_repulsion_Eh"] = float(m.group(1))
                break

        # Symmetry
        for ln in lines:
            m = re.search(r"Symmetry-adapted orbitals\s+\.\.\.\.\s+(\w+)", ln)
            if m:
                data["point_group"] = m.group(1)
                self.context["has_symmetry"] = True
                break
        else:
            if "has_symmetry" not in self.context:
                self.context["has_symmetry"] = False

        # Relativistic method
        for ln in lines:
            m = re.search(r"Relativistic Method\s+\.\.\.\s+(\S+)", ln)
            if m:
                data["relativistic_method"] = m.group(1)
                break

        # Set context flags based on parsed data
        hf_type = data.get("hf_type", "RHF")
        self.context["is_uhf"] = hf_type == "UHF"
        self.context["hf_type"] = hf_type

        return data if data else None


class GeometryModule(BaseModule):
    """Extracts Cartesian coordinates (Å and a.u.) and internal coordinates."""

    name = "geometry"

    def parse(self, lines):
        data = {}

        # Cartesian in Angstrom
        idx = self.find_line(lines, "CARTESIAN COORDINATES (ANGSTROEM)")
        if idx != -1:
            atoms_ang = []
            for ln in lines[idx + 2:]:
                m = re.match(r"\s+(\w+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", ln)
                if m:
                    atoms_ang.append({
                        "symbol": m.group(1),
                        "x_ang": float(m.group(2)),
                        "y_ang": float(m.group(3)),
                        "z_ang": float(m.group(4)),
                    })
                elif ln.strip() == "" and atoms_ang:
                    break
            data["cartesian_angstrom"] = atoms_ang

        # Cartesian in a.u. (includes Z, mass)
        idx = self.find_line(lines, "CARTESIAN COORDINATES (A.U.)")
        if idx != -1:
            atoms_au = []
            for ln in lines[idx + 3:]:
                m = re.match(r"\s+(\d+)\s+(\w+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", ln)
                if m:
                    atoms_au.append({
                        "index": int(m.group(1)),
                        "symbol": m.group(2),
                        "nuclear_charge": float(m.group(3)),
                        "fragment": int(m.group(4)),
                        "mass_amu": float(m.group(5)),
                        "x_au": float(m.group(6)),
                        "y_au": float(m.group(7)),
                        "z_au": float(m.group(8)),
                    })
                elif ln.strip() == "" and atoms_au:
                    break
            data["cartesian_au"] = atoms_au

        # Internal coordinates (Angstrom)
        idx = self.find_line(lines, "INTERNAL COORDINATES (ANGSTROEM)")
        if idx != -1:
            internals = []
            for ln in lines[idx + 2:]:
                parts = ln.split()
                if len(parts) >= 7 and parts[0][0].isupper():
                    try:
                        internals.append({
                            "symbol": parts[0],
                            "ref1": int(parts[1]),
                            "ref2": int(parts[2]),
                            "ref3": int(parts[3]),
                            "r_ang": float(parts[4]),
                            "angle_deg": float(parts[5]),
                            "dihedral_deg": float(parts[6]),
                        })
                    except (ValueError, IndexError):
                        pass
                elif ln.strip() == "" and internals:
                    break
            data["internal_angstrom"] = internals

        # Number of atoms
        if "cartesian_angstrom" in data:
            data["n_atoms"] = len(data["cartesian_angstrom"])
            self.context["n_atoms"] = data["n_atoms"]
            self.context["atom_symbols"] = [a["symbol"] for a in data["cartesian_angstrom"]]

        # Symmetry coordinates
        idx = self.find_line(lines, "Symmetry-perfected Cartesians for point group")
        if idx != -1:
            m = re.search(r"point group\s+(\w+)", lines[idx])
            if m:
                data["symmetry_point_group"] = m.group(1)
                sym_atoms = []
                for ln in lines[idx + 5:]:
                    m2 = re.match(r"\s+(\d+)\s+(\w+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", ln)
                    if m2:
                        sym_atoms.append({
                            "index": int(m2.group(1)),
                            "symbol": m2.group(2),
                            "x_au": float(m2.group(3)),
                            "y_au": float(m2.group(4)),
                            "z_au": float(m2.group(5)),
                        })
                    elif ln.strip() == "" and sym_atoms:
                        break
                if sym_atoms:
                    data["symmetry_cartesian_au"] = sym_atoms

        return data if data else None


class BasisSetModule(BaseModule):
    """Extracts basis set composition per atom."""

    name = "basis_set"

    def parse(self, lines):
        data = {}

        # Groups
        idx = self.find_line(lines, "BASIS SET INFORMATION")
        if idx == -1:
            return None

        # Number of distinct groups
        m = re.search(r"There are (\d+) groups of distinct atoms", lines[idx + 2] if idx + 2 < len(lines) else "")
        if m:
            data["n_groups"] = int(m.group(1))

        # Parse group descriptions
        groups = {}
        for ln in lines[idx:idx + 30]:
            m = re.match(r"\s+Group\s+(\d+)\s+Type\s+(\w+)\s+:\s+(.+)", ln)
            if m:
                groups[int(m.group(1))] = {"element": m.group(2), "description": m.group(3).strip()}

        # Atom -> group mapping (read until blank line after data)
        atom_groups = {}
        for ln in lines[idx:]:
            m = re.match(r"Atom\s+(\d+)(\w+)\s+basis set group =>\s+(\d+)", ln)
            if m:
                atom_groups[f"{m.group(1)}{m.group(2)}"] = int(m.group(3))
            elif atom_groups and ln.strip() == "":
                break

        data["groups"] = groups
        data["atom_group_mapping"] = atom_groups

        # Number of basis functions, shells, etc.
        for ln in lines:
            m = re.search(r"Number of basis functions\s+\.\.\.\s+(\d+)", ln)
            if m:
                data["n_basis_functions"] = int(m.group(1))
            m = re.search(r"Number of shells\s+\.\.\.\s+(\d+)", ln)
            if m:
                data["n_shells"] = int(m.group(1))
            m = re.search(r"Maximum angular momentum\s+\.\.\.\s+(\d+)", ln)
            if m:
                data["max_angular_momentum"] = int(m.group(1))

        return data if data else None
