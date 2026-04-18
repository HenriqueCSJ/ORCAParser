"""
Module for SCF total energy, convergence data, and DFT components.
"""

import re
from typing import Any, Dict, Optional

from .base import BaseModule


class SCFModule(BaseModule):
    """
    Extracts:
    - Total energy and energy components (nuclear repulsion, electronic, 1e, 2e)
    - Virial components (kinetic, potential)
    - DFT components (N_alpha, N_beta, XC, NL)
    - SCF convergence metrics
    - <S**2> for UHF
    """

    name = "scf"

    def parse(self, lines):
        data = {}

        # --- Total energy (from FINAL SINGLE POINT ENERGY) ---
        idx = self.find_last_line(lines, "FINAL SINGLE POINT ENERGY")
        if idx != -1:
            m = re.search(r"FINAL SINGLE POINT ENERGY\s+([-\d.]+)", lines[idx])
            if m:
                data["final_single_point_energy_Eh"] = float(m.group(1))

        # --- Energy components from the last TOTAL SCF ENERGY block ---
        idx = self.find_last_line(lines, "TOTAL SCF ENERGY")
        if idx != -1:
            for ln in lines[idx: idx + 25]:
                m = re.search(r"Total Energy\s+:\s+([-\d.]+)\s+Eh\s+([-\d.]+)\s+eV", ln)
                if m:
                    data["total_energy_Eh"] = float(m.group(1))
                    data["total_energy_eV"] = float(m.group(2))
                m = re.search(r"Nuclear Repulsion\s+:\s+([-\d.]+)\s+Eh", ln)
                if m:
                    data["nuclear_repulsion_Eh"] = float(m.group(1))
                m = re.search(r"Electronic Energy\s+:\s+([-\d.]+)\s+Eh", ln)
                if m:
                    data["electronic_energy_Eh"] = float(m.group(1))
                m = re.search(r"One Electron Energy:\s+([-\d.]+)\s+Eh", ln)
                if m:
                    data["one_electron_energy_Eh"] = float(m.group(1))
                m = re.search(r"Two Electron Energy:\s+([-\d.]+)\s+Eh", ln)
                if m:
                    data["two_electron_energy_Eh"] = float(m.group(1))
                m = re.search(r"Potential Energy\s+:\s+([-\d.]+)\s+Eh", ln)
                if m:
                    data["potential_energy_Eh"] = float(m.group(1))
                m = re.search(r"Kinetic Energy\s+:\s+([-\d.]+)\s+Eh", ln)
                if m:
                    data["kinetic_energy_Eh"] = float(m.group(1))
                m = re.search(r"Virial Ratio\s+:\s+([-\d.]+)", ln)
                if m:
                    data["virial_ratio"] = float(m.group(1))

        # --- DFT components ---
        dft = {}
        idx_dft = self.find_last_line(lines, "DFT components:")
        if idx_dft != -1:
            for ln in lines[idx_dft: idx_dft + 15]:
                m = re.search(r"N\(Alpha\)\s+:\s+([\d.]+)\s+electrons", ln)
                if m:
                    dft["N_alpha"] = float(m.group(1))
                m = re.search(r"N\(Beta\)\s+:\s+([\d.]+)\s+electrons", ln)
                if m:
                    dft["N_beta"] = float(m.group(1))
                m = re.search(r"N\(Total\)\s+:\s+([\d.]+)\s+electrons", ln)
                if m:
                    dft["N_total"] = float(m.group(1))
                m = re.search(r"E\(XC\)\s+:\s+([-\d.]+)\s+Eh", ln)
                if m:
                    dft["E_xc_Eh"] = float(m.group(1))
                m = re.search(r"NL Energy, E\(C,NL\)\s+:\s+([-\d.]+)\s+Eh", ln)
                if m:
                    dft["E_nl_Eh"] = float(m.group(1))
            if dft:
                data["dft_components"] = dft

        # --- NL dispersion energy ---
        idx_nl = self.find_last_line(lines, "NL    Energy:")
        if idx_nl != -1:
            m = re.search(r"NL\s+Energy:\s+([-\d.]+)", lines[idx_nl])
            if m:
                data["nl_dispersion_energy_Eh"] = float(m.group(1))

        # --- SCF convergence ---
        conv = {}
        idx_c = self.find_last_line(lines, "SCF CONVERGENCE")
        if idx_c != -1:
            for ln in lines[idx_c: idx_c + 12]:
                m = re.search(r"Last Energy change\s+\.\.\.\s+([-\d.e+]+)\s+Tolerance\s+:\s+([-\d.e+]+)", ln)
                if m:
                    conv["last_energy_change"] = float(m.group(1))
                    conv["energy_tolerance"] = float(m.group(2))
                m = re.search(r"Last MAX-Density change\s+\.\.\.\s+([-\d.e+]+)\s+Tolerance\s+:\s+([-\d.e+]+)", ln)
                if m:
                    conv["last_max_density_change"] = float(m.group(1))
                    conv["density_tolerance"] = float(m.group(2))
                m = re.search(r"Last Orbital Gradient\s+\.\.\.\s+([-\d.e+]+)\s+Tolerance\s+:\s+([-\d.e+]+)", ln)
                if m:
                    conv["last_orbital_gradient"] = float(m.group(1))
        if conv:
            data["scf_convergence"] = conv

        # Count SCF cycles
        idx_success = self.find_last_line(lines, "SCF CONVERGED AFTER")
        if idx_success != -1:
            m = re.search(r"SCF CONVERGED AFTER\s+(\d+)\s+CYCLES", lines[idx_success])
            if m:
                data["scf_cycles"] = int(m.group(1))

        # --- S**2 for UHF ---
        idx_s2 = self.find_last_line(lines, "Expectation value of <S**2>")
        if idx_s2 != -1:
            m = re.search(r"Expectation value of <S\*\*2>\s+:\s+([\d.]+)", lines[idx_s2])
            if m:
                data["s_squared"] = float(m.group(1))
                # Ideal S(S+1)
                mult = self.context.get("multiplicity")
                if mult:
                    s = (mult - 1) / 2
                    data["s_squared_ideal"] = s * (s + 1)
                    data["spin_contamination"] = data["s_squared"] - data["s_squared_ideal"]

        return data if data else None
