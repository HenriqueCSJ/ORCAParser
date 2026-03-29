"""
Modules for calculation metadata, geometry, and basis set information.
"""

import re
from typing import Any, Dict, List, Optional

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

        symmetry = self._parse_symmetry(lines)
        if symmetry:
            data["symmetry"] = symmetry
            if symmetry.get("point_group"):
                data["point_group"] = symmetry["point_group"]
            elif symmetry.get("auto_detected_point_group"):
                data["point_group"] = symmetry["auto_detected_point_group"]
            elif symmetry.get("reduced_point_group"):
                data["point_group"] = symmetry["reduced_point_group"]

            if symmetry.get("reduced_point_group"):
                data["reduced_point_group"] = symmetry["reduced_point_group"]
            if symmetry.get("orbital_irrep_group"):
                data["orbital_irrep_group"] = symmetry["orbital_irrep_group"]

            self.context["has_symmetry"] = True
            if symmetry.get("point_group"):
                self.context["point_group"] = symmetry["point_group"]
            if symmetry.get("orbital_irrep_group"):
                self.context["orbital_irrep_group"] = symmetry["orbital_irrep_group"]
        elif "has_symmetry" not in self.context:
            self.context["has_symmetry"] = False

        # Relativistic method
        for ln in lines:
            m = re.search(r"Relativistic Method\s+\.\.\.\s+(\S+)", ln)
            if m:
                data["relativistic_method"] = m.group(1)
                break

        # ── Input keywords (from echoed input "! ..." lines) ─────────
        input_keywords = []
        for ln in lines:
            m = re.match(r"^\|\s*\d+>\s*!\s*(.+)", ln)
            if m:
                input_keywords.extend(m.group(1).split())
            if "****END OF INPUT****" in ln:
                break
        if input_keywords:
            data["input_keywords"] = input_keywords
        is_surface_scan = (
            any("Relaxed Surface Scan" in ln for ln in lines[:5000])
            or self._input_contains_geom_scan(lines)
        )
        if is_surface_scan:
            data["calculation_type"] = "Relaxed Surface Scan"
            self.context["is_surface_scan"] = True

        # ── DeltaSCF detection and metadata ──────────────────────────
        # DeltaSCF converges SCF to an excited state (saddle point on the
        # energy surface) rather than the ground state.  Used for excited
        # states, core-ionized/core-excited states, and charge-transfer
        # states.  Excitation energy = E(DeltaSCF) − E(ground state).
        is_deltascf = any(
            kw.upper() == "DELTASCF" for kw in input_keywords
        )
        if is_deltascf:
            data["calculation_type"] = "DeltaSCF"
            deltascf: Dict[str, Any] = {}

            # Parse the %SCF block for DeltaSCF-specific options
            in_scf_block = False
            for ln in lines:
                if re.match(r"^\|\s*\d+>\s*%SCF", ln, re.I):
                    in_scf_block = True
                    continue
                if in_scf_block:
                    if re.match(r"^\|\s*\d+>\s*END", ln, re.I):
                        break

                    # ALPHACONF / BETACONF — frontier orbital occupation
                    # defining the target excited-state configuration.
                    # e.g. "0,1" = HOMO→LUMO; "0,1,1" = HOMO-1→LUMO
                    m = re.match(
                        r"^\|\s*\d+>\s*(ALPHACONF|BETACONF)\s+(.+)",
                        ln, re.I,
                    )
                    if m:
                        vals = [
                            int(v.strip())
                            for v in m.group(2).split(",")
                            if v.strip().lstrip("-").isdigit()
                        ]
                        deltascf[m.group(1).lower()] = vals
                        continue

                    # IONIZEALPHA / IONIZEBETA — remove an electron from
                    # a specific MO (used for core-hole / ionized states).
                    m = re.match(
                        r"^\|\s*\d+>\s*(IONIZEALPHA|IONIZEBETA)\s+(\d+)",
                        ln, re.I,
                    )
                    if m:
                        deltascf[m.group(1).lower()] = int(m.group(2))
                        continue

            # Parse the DELTA-SCF INITIAL CONFIGURATION block from output
            idx_ds = self.find_line(
                lines, "DELTA-SCF INITIAL CONFIGURATION"
            )
            if idx_ds != -1:
                for j in range(idx_ds + 1, min(idx_ds + 20, len(lines))):
                    ln = lines[j]
                    # Alpha/Beta occupation vectors — the actual electron
                    # arrangement being converged to
                    m = re.match(
                        r"^\s*(Alpha|Beta)\s*:\s*([\d.\s]+)", ln
                    )
                    if m:
                        occ = [
                            float(v)
                            for v in m.group(2).split()
                        ]
                        deltascf[f"{m.group(1).lower()}_occupation"] = occ

                    # MOM vs PMOM — overlap metric used to keep SCF
                    # converging toward the target excited state
                    m = re.match(
                        r"^\s*Aufbau metric\s+\.+\s+(\S+)", ln
                    )
                    if m:
                        deltascf["aufbau_metric"] = m.group(1)

                    # IMOM — whether the initial (non-aufbau) reference
                    # is kept throughout, or updated each iteration
                    m = re.match(
                        r"^\s*Keep initial reference\s+\.+\s+(\S+)", ln
                    )
                    if m:
                        deltascf["keep_initial_reference"] = (
                            m.group(1).lower() == "true"
                        )

                    # Stop at the SCF iterations header
                    if "---D-I-I-S---" in ln or "---S-O-S-C-F---" in ln:
                        break

            if deltascf:
                data["deltascf"] = deltascf

        # Set context flags based on parsed data
        hf_type = data.get("hf_type", "RHF")
        self.context["is_uhf"] = hf_type.upper() == "UHF"
        self.context["hf_type"] = hf_type

        return data if data else None

    def _input_contains_geom_scan(self, lines: List[str]) -> bool:
        """Detect echoed ``%geom ... scan`` blocks in the ORCA input."""
        in_geom = False
        for ln in lines:
            if "****END OF INPUT****" in ln:
                break
            m = re.match(r"^\|\s*\d+>\s*(.*)$", ln)
            if not m:
                continue
            content = m.group(1).strip().lower()
            if content.startswith("%geom"):
                in_geom = True
                continue
            if in_geom and content == "scan":
                return True
            if in_geom and content == "end":
                in_geom = False
        return False

    def _parse_symmetry(self, lines: List[str]) -> Dict[str, Any]:
        """Extract ORCA symmetry metadata, keeping geometry and MO irreps distinct."""
        sym: Dict[str, Any] = {}

        for i, ln in enumerate(lines):
            m = re.search(r"INITIAL GUESS:\s+(.+)", ln)
            if m:
                sym["initial_guess_method"] = m.group(1).strip()
                continue

            m = re.search(r"Guess MOs are being read from file:\s+(\S+)", ln)
            if m:
                sym["initial_guess_source_file"] = m.group(1)
                continue

            if "Input Geometry matches current geometry" in ln:
                sym["initial_guess_geometry_matches"] = "(good)" in ln.lower()
                continue

            if "Input basis set matches current basis set" in ln:
                sym["initial_guess_basis_matches"] = "(good)" in ln.lower()
                continue

            if "We clean up the input orbitals and determine their irreps" in ln:
                sym["initial_guess_irreps_reassigned"] = True
                continue

            if "MOs were renormalized" in ln:
                sym["initial_guess_mos_renormalized"] = True
                continue

            m = re.search(r"MOs were reorthogonalized(?:\s+\(([^)]+)\))?", ln)
            if m:
                sym["initial_guess_mos_reorthogonalized"] = True
                if m.group(1):
                    sym["initial_guess_reorthogonalization_method"] = m.group(1).strip()
                continue

            m = re.search(r"Auto-detected point group\s+\.+\s+(\S+)", ln)
            if m:
                sym["auto_detected_point_group"] = m.group(1)
                continue

            m = re.search(r"Reduced point group\s+\.+\s+(\S+)", ln)
            if m:
                sym["reduced_point_group"] = m.group(1)
                continue

            m = re.search(r"Root mean square distance\s+\.+\s+([-\d.Ee+]+)\s+au", ln)
            if m:
                sym["setup_rms_distance_au"] = float(m.group(1))
                continue

            m = re.search(r"Maximum distance\s+\.+\s+([-\d.Ee+]+)\s+au", ln)
            if m:
                sym["setup_max_distance_au"] = float(m.group(1))
                continue

            m = re.search(r"Threshold in use\s+\.+\s+([-\d.Ee+]+)\s+au", ln)
            if m:
                sym["setup_threshold_au"] = float(m.group(1))
                continue

            m = re.search(r"Time for symmetry setup\s+\.+\s+([-\d.Ee+]+)\s+s", ln)
            if m:
                sym["setup_time_s"] = float(m.group(1))
                continue

            m = re.search(r"Symmetry handling\s+UseSym\s+\.+\s+(\w+)", ln)
            if m:
                value = m.group(1).upper()
                sym["use_sym"] = value == "ON"
                sym["use_sym_label"] = value
                continue

            m = re.match(r"^\s*Point group\s+\.+\s+(\S+)", ln)
            if m:
                sym["point_group"] = m.group(1)
                continue

            m = re.search(r"Symmetry-adapted orbitals\s+\.+\s+(\S+)", ln)
            if m:
                sym["orbital_irrep_group"] = m.group(1)
                continue

            m = re.search(r"Petite-list algorithm\s+\.+\s+(\S+)", ln)
            if m:
                value = m.group(1).upper()
                sym["petite_list_algorithm"] = value == "ON"
                sym["petite_list_algorithm_label"] = value
                continue

            m = re.search(r"Number of irreps\s+\.+\s+(\d+)", ln)
            if m:
                sym["n_irreps"] = int(m.group(1))
                irreps: List[Dict[str, Any]] = []
                for sub_ln in lines[i + 1:]:
                    m_ir = re.match(
                        r"^\s*Irrep\s+(\S+)\s+has\s+(\d+)\s+symmetry adapted basis functions"
                        r"\s+\(ofs=\s*(\d+)\)",
                        sub_ln,
                    )
                    if not m_ir:
                        if irreps:
                            break
                        continue
                    irreps.append({
                        "label": m_ir.group(1),
                        "n_basis_functions": int(m_ir.group(2)),
                        "offset": int(m_ir.group(3)),
                    })
                if irreps:
                    sym["irreps"] = irreps
                continue

            m = re.search(r"The symmetry of the initial guess is\s+(\S+)", ln)
            if m:
                sym["initial_guess_irrep"] = m.group(1)

        return sym


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
                data["symmetry_perfected_point_group"] = m.group(1)
                sym_atoms = []
                for ln in lines[idx + 1:]:
                    if "Symmetry-perfected Cartesians" in ln and "; Ang" in ln:
                        break
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

                sym_atoms_ang = []
                ang_header_seen = False
                for ln in lines[idx + 1:]:
                    if "Symmetry-perfected Cartesians" in ln and "; Ang" in ln:
                        ang_header_seen = True
                        continue
                    if not ang_header_seen:
                        continue
                    m3 = re.match(r"\s+(\d+)\s+(\w+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", ln)
                    if m3:
                        sym_atoms_ang.append({
                            "index": int(m3.group(1)),
                            "symbol": m3.group(2),
                            "x_ang": float(m3.group(3)),
                            "y_ang": float(m3.group(4)),
                            "z_ang": float(m3.group(5)),
                        })
                    elif ln.strip() == "" and sym_atoms_ang:
                        break
                if sym_atoms_ang:
                    data["symmetry_cartesian_angstrom"] = sym_atoms_ang

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
