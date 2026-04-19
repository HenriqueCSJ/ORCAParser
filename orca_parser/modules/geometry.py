"""
Modules for calculation metadata, geometry, and basis set information.
"""

from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional

from ..input_semantics import (
    detect_input_symmetry_request,
    infer_reference_type,
    is_unrestricted_reference,
)
from ..job_family_registry import CalculationFamilyPlugin
from ..output.csv_section_registry import GEOMETRY_CSV_SECTION_PLUGIN
from ..output.markdown_section_registry import (
    BASIS_SET_MARKDOWN_SECTION_PLUGIN,
    GEOMETRY_MARKDOWN_SECTION_PLUGIN,
)
from ..parser_section_plugin import ParserSectionAlias, ParserSectionPlugin
from ..plugin_bundle import PluginBundle, PluginMetadata
from ..render_options import RenderOptions
from .base import BaseModule


class MetadataModule(BaseModule):
    """Extracts program version, job name, host, date, input keywords, etc."""

    name = "metadata"

    def parse(self, lines):
        data: Dict[str, Any] = {}
        self._apply_input_echo_metadata(data)

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
                data.setdefault("input_name", m.group(1))
                data.setdefault("job_name", m.group(1).replace(".inp", ""))
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
        if "calculation_type" not in data:
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

        # SCF spin-treatment label from ORCA output (e.g. RHF/UHF/ROHF)
        for ln in lines:
            m = re.search(r"Hartree-Fock type\s+HFTyp\s+\.\.\.\.\s+(\w+)", ln)
            if m:
                data["hf_type"] = m.group(1)
                break

        # Functional
        for ln in lines:
            m = re.search(r"Functional name\s+\.\.\.\.\s+(.+)", ln)
            if m:
                reported_functional = m.group(1).strip()
                data["reported_functional"] = reported_functional
                data.setdefault("functional", reported_functional)
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
                reported_basis = m.group(1).strip()
                data["reported_basis_set"] = reported_basis
                data.setdefault("basis_set", reported_basis)
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
        input_use_sym = data.get("input_use_sym")
        if symmetry:
            if input_use_sym is not None and "use_sym" not in symmetry:
                symmetry["use_sym"] = input_use_sym
                symmetry["use_sym_label"] = "ON" if input_use_sym else "OFF"
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
        elif input_use_sym is not None:
            data["symmetry"] = {
                "use_sym": input_use_sym,
                "use_sym_label": "ON" if input_use_sym else "OFF",
            }
            self.context["has_symmetry"] = bool(input_use_sym)
        elif "has_symmetry" not in self.context:
            self.context["has_symmetry"] = False

        # Relativistic method
        for ln in lines:
            m = re.search(r"Relativistic Method\s+\.\.\.\s+(\S+)", ln)
            if m:
                data["relativistic_method"] = m.group(1)
                break

        # ── Input keywords (from echoed input "! ..." lines) ─────────
        input_keywords = data.get("input_keywords") or []
        is_surface_scan = (
            any("Relaxed Surface Scan" in ln for ln in lines[:5000])
            or self._input_contains_geom_scan(lines)
            or self._input_echo_contains_geom_scan()
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

        # Normalize reference semantics once here so every downstream module
        # gets the same answer for RHF/RKS/UHF/UKS-style decisions.
        reference_type = infer_reference_type(
            bang_tokens=(self.context.get("input_echo") or {}).get("bang_tokens", []),
            hf_type=data.get("hf_type", self.context.get("hf_type", "RHF")),
            method=data.get("method"),
            functional=data.get("functional"),
            reported_functional=data.get("reported_functional"),
        ) or "RHF"
        is_unrestricted = is_unrestricted_reference(reference_type)

        data["reference_type"] = reference_type
        self.context["reference_type"] = reference_type
        self.context["is_unrestricted"] = is_unrestricted
        # ``is_uhf`` remains as a compatibility alias for existing parser and
        # rendering code that still uses the older name.
        self.context["is_uhf"] = is_unrestricted
        self.context["hf_type"] = data.get("hf_type", self.context.get("hf_type", "RHF"))

        return data if data else None

    _METHOD_SKIP_TOKENS = {
        "RHF",
        "UHF",
        "ROHF",
        "RKS",
        "UKS",
        "ROKS",
        "SP",
        "OPT",
        "OPTTS",
        "FREQ",
        "NUMFREQ",
        "ANFREQ",
        "GOAT",
        "CHELPG",
        "AIM",
        "ALLPOP",
        "KEEPDENS",
        "FMOPOP",
        "MULLIKEN",
        "LOEWDIN",
        "MAYER",
        "REDUCEDPOP",
        "PRINTBASIS",
        "PRINTMOS",
        "NBO",
    }
    _CALCULATION_KEYWORDS = {
        "SP",
        "OPT",
        "OPTTS",
        "GOAT",
        "FREQ",
        "NUMFREQ",
        "ANFREQ",
        "IRC",
        "SCAN",
    }
    _BASIS_TOKEN_RE = re.compile(
        r"(?i)^(?:"
        r"(?:ma-)?def2-"
        r"|x2c-"
        r"|jora-"
        r"|sarc"
        r"|dhf-"
        r"|cc-p"
        r"|aug-cc"
        r"|jun-cc"
        r"|may-cc"
        r"|pcseg"
        r"|pc-"
        r"|svp$"
        r"|sv\(p\)$"
        r"|tzvp"
        r"|tzvpp"
        r"|qzvp"
        r"|qzvpp"
        r"|6-31"
        r"|6-311"
        r"|sto-"
        r"|minis"
        r"|minix"
        r"|lanl"
        r"|sdd"
        r"|ano"
        r").*"
    )

    def _apply_input_echo_metadata(self, data: Dict[str, Any]) -> None:
        """Promote reliable metadata from the echoed ORCA input block."""
        input_echo = self.context.get("input_echo") or {}
        data["job_id"] = self.context.get("job_id", "")

        source_relpath = self.context.get("source_relpath")
        if source_relpath:
            data["source_relpath"] = source_relpath

        if not input_echo:
            return

        input_name = input_echo.get("input_name")
        if input_name:
            data["input_name"] = input_name
            data.setdefault("job_name", input_name.replace(".inp", ""))

        bang_lines = input_echo.get("bang_lines") or []
        if bang_lines:
            data["input_lines"] = list(bang_lines)
            data["input_line"] = " ".join(bang_lines)

        bang_tokens = input_echo.get("bang_tokens") or []
        if bang_tokens:
            data["input_keywords"] = list(bang_tokens)

        block_names = input_echo.get("block_names") or []
        if block_names:
            data["input_blocks"] = list(block_names)

        structure_input = input_echo.get("structure_input") or {}
        if structure_input:
            if structure_input.get("charge") is not None:
                data.setdefault("charge", structure_input["charge"])
            if structure_input.get("multiplicity") is not None:
                data.setdefault("multiplicity", structure_input["multiplicity"])
            if structure_input.get("kind"):
                data["input_structure_type"] = structure_input["kind"]
            if structure_input.get("source"):
                data["input_structure_source"] = structure_input["source"]

        derived = self._derive_model_chemistry(input_echo)
        for key, value in derived.items():
            if value not in (None, "", []):
                data[key] = value

        input_use_sym = self._derive_input_symmetry_requested(input_echo)
        if input_use_sym is not None:
            data["input_use_sym"] = input_use_sym

        calculation_type = self._derive_calculation_type_from_input(input_echo)
        if calculation_type:
            data["calculation_type"] = calculation_type

        excited_state_input = self._derive_excited_state_input_metadata(input_echo)
        if excited_state_input:
            data["excited_state_optimization"] = excited_state_input

    def _derive_model_chemistry(self, input_echo: Dict[str, Any]) -> Dict[str, Any]:
        """Infer method, basis, and level labels from echoed ``!`` tokens."""
        tokens = input_echo.get("bang_tokens") or []
        if not tokens:
            return {}

        method = None
        basis_set = None
        aux_basis_set = None
        solvent_model = None

        for token in tokens:
            if solvent_model is None and self._looks_like_solvent_model(token):
                solvent_model = token
                continue
            if aux_basis_set is None and self._looks_like_aux_basis(token):
                aux_basis_set = token
                continue
            if basis_set is None and self._looks_like_basis(token):
                basis_set = token
                continue
            if method is None and not self._is_non_method_token(token):
                method = token

        level_of_theory = method or ""
        if method and basis_set:
            level_of_theory = f"{method}/{basis_set}"
        elif basis_set:
            level_of_theory = basis_set

        derived: Dict[str, Any] = {}
        if method:
            derived["method"] = method
            derived.setdefault("functional", method)
        if basis_set:
            derived["basis_set"] = basis_set
        if aux_basis_set:
            derived["aux_basis_set"] = aux_basis_set
        if solvent_model:
            derived["solvent_model"] = solvent_model
        if level_of_theory:
            derived["level_of_theory"] = level_of_theory
        return derived

    def _derive_calculation_type_from_input(self, input_echo: Dict[str, Any]) -> Optional[str]:
        """Classify the run from the echoed input before output parsing drifts."""
        tokens_upper = {token.upper() for token in input_echo.get("bang_tokens") or []}
        blocks = {name.lower() for name in input_echo.get("block_names") or []}

        if "GOAT" in tokens_upper or "goat" in blocks:
            return "GOAT Conformer Search"
        if self._input_echo_contains_geom_scan_dict(input_echo):
            return "Relaxed Surface Scan"
        if any(token in tokens_upper for token in {"OPT", "OPTTS"}):
            if any(block in blocks for block in {"tddft", "cis"}):
                return "Excited-State Geometry Optimization"
            return "Geometry Optimization"
        if any(token in tokens_upper for token in {"FREQ", "NUMFREQ", "ANFREQ"}):
            return "Frequency"
        return "Single Point"

    def _derive_excited_state_input_metadata(
        self,
        input_echo: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Extract lightweight excited-state optimization metadata from input."""
        tokens_upper = {token.upper() for token in input_echo.get("bang_tokens") or []}
        if not any(token in tokens_upper for token in {"OPT", "OPTTS"}):
            return None

        blocks = input_echo.get("blocks") or {}
        block_name = next((name for name in ("tddft", "cis") if name in blocks), None)
        if block_name is None:
            return None

        settings = dict(blocks.get(block_name, {}).get("settings") or {})
        excited_state: Dict[str, Any] = {"input_block": block_name}

        target_root = settings.get("iroot")
        if isinstance(target_root, int):
            excited_state["target_root"] = target_root

        irootmult = settings.get("irootmult")
        if isinstance(irootmult, str) and irootmult:
            excited_state["irootmult"] = irootmult.upper()

        if "followiroot" in settings:
            excited_state["followiroot"] = bool(settings["followiroot"])

        if "firkeepfirstref" in settings:
            excited_state["firkeepfirstref"] = bool(settings["firkeepfirstref"])

        if "nroots" in settings:
            excited_state["nroots"] = settings["nroots"]

        if isinstance(target_root, int) and target_root >= 0:
            mult_label = str(irootmult).strip().upper() if irootmult else ""
            state_prefix = "T" if mult_label == "TRIPLET" else "S"
            excited_state["target_state_label"] = f"{state_prefix}{target_root}"

        return excited_state or None

    def _derive_input_symmetry_requested(self, input_echo: Dict[str, Any]) -> Optional[bool]:
        """Detect normalized symmetry intent from the input.

        ORCA defaults to not using symmetry unless ``UseSym`` (or the
        equivalent ``%sym`` setting) turns it on.  We therefore normalize the
        absence of a symmetry request to ``False`` instead of leaving it
        unknown.
        """
        token_request = detect_input_symmetry_request(input_echo.get("bang_tokens") or [])
        if token_request is not None:
            return token_request

        sym_block = (input_echo.get("blocks") or {}).get("sym") or {}
        for raw_line in sym_block.get("raw_lines") or []:
            lowered = raw_line.lower()
            if "usesym" not in lowered:
                continue
            if "false" in lowered or "off" in lowered:
                return False
            if "true" in lowered or "on" in lowered:
                return True
        return False

    def _input_echo_contains_geom_scan(self) -> bool:
        """Detect ``%geom ... scan`` in the shared echoed-input payload."""
        return self._input_echo_contains_geom_scan_dict(
            self.context.get("input_echo") or {}
        )

    @classmethod
    def _input_echo_contains_geom_scan_dict(cls, input_echo: Dict[str, Any]) -> bool:
        geom_block = (input_echo.get("blocks") or {}).get("geom") or {}
        for raw_line in geom_block.get("raw_lines") or []:
            if raw_line.strip().lower() == "scan":
                return True
        return False

    def _is_non_method_token(self, token: str) -> bool:
        token_upper = token.upper()
        if token_upper in self._METHOD_SKIP_TOKENS:
            return True
        if token_upper.startswith("PAL") and token_upper[3:].isdigit():
            return True
        if token_upper in self._CALCULATION_KEYWORDS:
            return True
        if self._looks_like_basis(token) or self._looks_like_aux_basis(token):
            return True
        if self._looks_like_solvent_model(token):
            return True
        return False

    def _looks_like_basis(self, token: str) -> bool:
        if self._looks_like_aux_basis(token):
            return False
        return bool(self._BASIS_TOKEN_RE.match(token))

    @staticmethod
    def _looks_like_aux_basis(token: str) -> bool:
        token_upper = token.upper()
        return (
            token_upper.endswith("/C")
            or token_upper.endswith("/J")
            or token_upper.endswith("/JK")
            or token_upper.endswith("JKFIT")
        )

    @staticmethod
    def _looks_like_solvent_model(token: str) -> bool:
        token_upper = token.upper()
        return token_upper.startswith(("CPCM(", "SMD(", "ALPB(", "COSMO(", "DDCOSMO("))

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


def _matches_deltascf(
    meta: Dict[str, Any],
    data: Dict[str, Any],
    context: Dict[str, Any],
    deltascf: Dict[str, Any],
    excited_state_optimization: Dict[str, Any],
) -> bool:
    """Match DeltaSCF jobs from the same module that parsed their metadata."""

    del data, context, excited_state_optimization
    calc_type = str(meta.get("calculation_type", "")).strip().lower()
    return bool(deltascf or calc_type == "deltascf")


def _build_deltascf_state_label(
    meta: Dict[str, Any],
    data: Dict[str, Any],
    deltascf: Dict[str, Any],
    excited_state_optimization: Dict[str, Any],
) -> str:
    del meta, data, deltascf, excited_state_optimization
    return "DeltaSCF excited-state"


def _render_deltascf_markdown_sections(
    data: Dict[str, Any],
    format_number: Callable[[Any, str], str],
    make_table: Callable[[List[tuple]], str],
    render_options: RenderOptions,
) -> List[tuple[str, str]]:
    """Render DeltaSCF markdown from the owning metadata module."""

    del render_options

    from ..output import job_state as _job_state
    from ..output.markdown_sections_state import render_deltascf_section

    formatter = lambda value: format_number(value)
    body = render_deltascf_section(
        data,
        get_deltascf_data=_job_state.get_deltascf_data,
        deltascf_target_summary=lambda deltascf: _job_state.deltascf_target_summary(
            deltascf,
            formatter=formatter,
        ),
        format_deltascf_vector=lambda values: _job_state.format_deltascf_vector(
            values,
            formatter=formatter,
        ),
        yes_no_unknown=_job_state.yes_no_unknown,
        make_table=make_table,
    )
    return [("DeltaSCF / Excited-State Target", body)] if body else []


def _render_deltascf_comparison_sections(
    datasets: List[Dict[str, Any]],
    labels: List[str],
    format_number: Callable[[Any, str], str],
    make_table: Callable[[List[tuple]], str],
    render_options: RenderOptions,
) -> List[tuple[str, str]]:
    """Render DeltaSCF comparison sections from the owning module."""

    del format_number, render_options

    from ..job_family_registry import get_calculation_family_plugin
    from ..output import job_state as _job_state

    if not any(get_calculation_family_plugin(dataset).family == "deltascf" for dataset in datasets):
        return []

    rows = [("", "electronic state", "target", "metric", "keep ref")]
    for label, dataset in zip(labels, datasets):
        if get_calculation_family_plugin(dataset).family != "deltascf":
            rows.append((label, "—", "—", "—", "—"))
            continue
        deltascf = _job_state.get_deltascf_data(dataset)
        rows.append((
            label,
            _job_state.electronic_state_label(dataset, ground_state_label="ground-state") or "ground-state",
            _job_state.deltascf_target_summary(deltascf) or "—",
            deltascf.get("aufbau_metric") or "—",
            _job_state.yes_no_unknown(deltascf.get("keep_initial_reference")),
        ))
    return [("DeltaSCF", make_table(rows))]


def _write_deltascf_csv_sections(
    data: Dict[str, Any],
    directory: Path,
    stem: str,
    write_csv: Callable[[Path, str, List[Dict[str, Any]], List[str]], Path],
) -> List[Path]:
    """Write DeltaSCF CSV output from the owning metadata module."""

    from ..output import job_state as _job_state
    from ..output.csv_sections_state import write_deltascf_section

    return write_deltascf_section(
        data,
        directory,
        stem,
        write_csv=write_csv,
        electronic_state_label=lambda dataset: _job_state.electronic_state_label(
            dataset,
            ground_state_label="Ground-state",
        ),
        get_deltascf_data=_job_state.get_deltascf_data,
        is_deltascf=_job_state.is_deltascf,
        format_deltascf_target=_job_state.format_deltascf_target,
        format_simple_vector=_job_state.format_simple_vector,
        bool_to_label=_job_state.bool_to_label,
    )


class GeometryModule(BaseModule):
    """Extracts Cartesian coordinates (Å and a.u.) and internal coordinates."""

    name = "geometry"

    def parse(self, lines):
        data = {}

        # Geometry optimizations print this block for every step; prefer the
        # converged geometry instead of the input/start structure.
        idx = self.find_last_line_exact(lines, "CARTESIAN COORDINATES (ANGSTROEM)")
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
        idx = self.find_last_line_exact(lines, "CARTESIAN COORDINATES (A.U.)")
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
        idx = self.find_last_line_exact(lines, "INTERNAL COORDINATES (ANGSTROEM)")
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


PLUGIN_BUNDLES = (
    PluginBundle(
        metadata=PluginMetadata(
            key="geometry_sections",
            name="Geometry Core Sections",
            short_help="Built-in metadata/geometry/basis parser sections owned by geometry.py.",
            description=(
                "Self-registering built-in parser sections for metadata, "
                "final geometry, and basis-set composition, plus the "
                "geometry section alias."
            ),
            docs_path="README.md",
            examples=(
                "orca_parser job.out --sections geometry",
                "orca_parser job.out --sections metadata basis_set",
            ),
        ),
        parser_sections=(
            ParserSectionPlugin("metadata", MetadataModule, always_include=True),
            ParserSectionPlugin("geometry", GeometryModule, always_include=True),
            ParserSectionPlugin("basis_set", BasisSetModule, always_include=True),
        ),
        parser_aliases=(
            ParserSectionAlias(name="geometry", section_keys=("geometry", "basis_set")),
        ),
        markdown_sections=(
            BASIS_SET_MARKDOWN_SECTION_PLUGIN,
            GEOMETRY_MARKDOWN_SECTION_PLUGIN,
        ),
        csv_sections=(
            GEOMETRY_CSV_SECTION_PLUGIN,
        ),
    ),
    PluginBundle(
        metadata=PluginMetadata(
            key="deltascf",
            name="DeltaSCF Family",
            short_help="Built-in DeltaSCF family behavior owned by the metadata module.",
            description=(
                "Owns normalized DeltaSCF family classification plus the "
                "family-specific markdown and CSV hooks used by downstream "
                "renderers."
            ),
            docs_path="README.md",
            examples=(
                "orca_parser excited_state.out --markdown",
                "orca_parser *.out --compare",
            ),
        ),
        calculation_families=(
            CalculationFamilyPlugin(
                family="deltascf",
                default_calculation_label="DeltaSCF",
                matcher=_matches_deltascf,
                electronic_state_kind="deltascf_excited_state",
                build_special_electronic_state_label=_build_deltascf_state_label,
                render_markdown_sections=_render_deltascf_markdown_sections,
                render_comparison_sections=_render_deltascf_comparison_sections,
                csv_writers=(_write_deltascf_csv_sections,),
                comparison_order=10,
            ),
        ),
    ),
)
