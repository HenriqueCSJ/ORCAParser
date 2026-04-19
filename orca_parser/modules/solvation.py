"""
Module for implicit-solvation parsing.

Extracts:
  - Simple-input solvation directives such as CPCM(...), SMD(...), ALPB(...)
  - %cpcm and %cosmors echoed input blocks
  - CPCM/SMD output initialization blocks
  - ALPB output blocks (xTB/GOAT workflows)
  - OpenCOSMO-RS output blocks

The main goal is to answer, robustly:
  1. Was implicit solvation requested for the final input state?
  2. Which model and solvent were used?
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from ..parser_section_plugin import ParserSectionAlias, ParserSectionPlugin
from ..plugin_bundle import PluginBundle, PluginMetadata
from .base import BaseModule


class SolvationModule(BaseModule):
    """Parses implicit-solvation settings and output blocks."""

    name = "solvation"

    _SIMPLE_INPUT_RE = re.compile(r"^\|\s*(\d+)>\s*!\s*(.+)$")
    _INPUT_BLOCK_START_RE = re.compile(r"^\|\s*(\d+)>\s*%(cpcm|cosmors)\b", re.I)
    _INPUT_BLOCK_LINE_RE = re.compile(r"^\|\s*\d+>\s*([A-Za-z][\w-]*)\s*(.*)$")
    _INPUT_BLOCK_END_RE = re.compile(r"^\|\s*\d+>\s*end\b", re.I)
    _MODEL_TOKEN_RE = re.compile(r"^(CPCM|CPCMC|SMD|ALPB|COSMORS)\((.+)\)$", re.I)

    _CPCM_HEADER_RE = re.compile(r"^\s*CPCM SOLVATION MODEL\s*$", re.I)
    _KV_DOTS_RE = re.compile(r"^\s*([^.:][^.]*?)\s+\.\.\.\s+(.+?)\s*$")
    _CPCM_DIELECTRIC_RE = re.compile(r"CPCM Dielectric\s*:\s*([-\d.]+)\s+Eh", re.I)
    _SMD_CDS_COMPONENT_RE = re.compile(r"SMD CDS \(Gcds\)\s*:\s*([-\d.]+)\s+Eh", re.I)
    _SMD_CDS_CORR_RE = re.compile(
        r"SMD CDS free energy correction energy\s*:\s*([-\d.]+)\s+Kcal/mol",
        re.I,
    )
    _SURFACE_CHARGE_RE = re.compile(r"Surface-charge\s*:\s*([-\d.]+)", re.I)
    _CORRECTED_CHARGE_RE = re.compile(r"Corrected charge\s*:\s*([-\d.]+)", re.I)
    _OUTLYING_CHARGE_RE = re.compile(
        r"Outlying charge corr\.\s*:\s*([-\d.]+)\s+Eh", re.I
    )
    _FREE_ENERGY_CAV_DISP_RE = re.compile(
        r"Free-energy \(cav\+disp\)\s*:\s*([-\d.]+)\s+Eh", re.I
    )

    _ALPB_HEADER_RE = re.compile(r"^\s*\*\s*Solvation model:\s*(\S+)", re.I)
    _ALPB_SPACED_RE = re.compile(
        r"^\s*(?:\*\s*)?([A-Za-z][A-Za-z0-9 \-()/]+?)\s{2,}(.+?)\s*$"
    )
    _ALPB_PROGRAM_CALL_RE = re.compile(r"--alpb\s+([A-Za-z0-9_,.+\-]+)", re.I)
    _FREE_ENERGY_SHIFT_RE = re.compile(
        r"Free energy shift\s+([-\d.E+]+)\s+Eh\s+([-\d.E+]+)\s+kcal/mol",
        re.I,
    )
    _SURFACE_TENSION_RE = re.compile(
        r"Surface tension\s+([-\d.E+]+)\s+Eh\s+([-\d.E+]+)\s+dyn/cm",
        re.I,
    )
    _GRID_POINTS_RE = re.compile(r"Grid points\s+(\d+)\s+per atom", re.I)

    _COSMORS_HEADER_RE = re.compile(r"^\s*OPENCOSMO-RS CALCULATION\s*$", re.I)
    _DGSOLV_RE = re.compile(
        r"Free energy of solvation \(dGsolv\)\s*:\s*([-\d.]+)\s+Eh\s+([-\d.]+)\s+kcal/mol",
        re.I,
    )

    def parse(self, lines):
        input_end = self.find_line(lines, "****END OF INPUT****")
        if input_end == -1:
            input_end = len(lines)

        input_directives, input_flags = self._parse_simple_input(lines, input_end)
        cpcm_input_blocks = self._parse_input_blocks(lines, input_end, "cpcm")
        cosmors_input_blocks = self._parse_input_blocks(lines, input_end, "cosmors")

        input_directives.extend(self._infer_directives_from_cpcm_blocks(cpcm_input_blocks))
        input_directives.extend(self._infer_directives_from_cosmors_blocks(cosmors_input_blocks))
        input_directives.sort(
            key=lambda item: (item.get("line_index", -1), item.get("source", ""))
        )

        cpcm_blocks = self._parse_cpcm_output_blocks(lines)
        alpb_blocks = self._parse_alpb_output_blocks(lines)
        cosmors_blocks = self._parse_cosmors_output_blocks(lines)

        if not any([
            input_directives,
            cpcm_input_blocks,
            cosmors_input_blocks,
            cpcm_blocks,
            alpb_blocks,
            cosmors_blocks,
            input_flags.get("draco"),
            input_flags.get("smd18"),
        ]):
            return None

        summary = self._build_summary(
            input_directives=input_directives,
            input_flags=input_flags,
            cpcm_blocks=cpcm_blocks,
            alpb_blocks=alpb_blocks,
            cosmors_blocks=cosmors_blocks,
        )

        data: Dict[str, Any] = {
            "is_solvated": summary.get("is_solvated", False),
            "primary_model": summary.get("primary_model"),
            "solvent": summary.get("solvent"),
            "models": summary.get("models", []),
            "input_directives": input_directives,
            "input_flags": input_flags,
            "summary": summary,
        }

        if cpcm_input_blocks:
            data["cpcm_input_blocks"] = cpcm_input_blocks
            data["cpcm_input"] = cpcm_input_blocks[-1]
        if cosmors_input_blocks:
            data["cosmors_input_blocks"] = cosmors_input_blocks
            data["cosmors_input"] = cosmors_input_blocks[-1]
        if cpcm_blocks:
            data["cpcm_blocks"] = cpcm_blocks
            data["cpcm"] = cpcm_blocks[-1]
        if alpb_blocks:
            data["alpb_blocks"] = alpb_blocks
            data["alpb"] = alpb_blocks[-1]
        if cosmors_blocks:
            data["cosmors_blocks"] = cosmors_blocks
            data["cosmors"] = cosmors_blocks[-1]

        return data

    def _parse_simple_input(
        self,
        lines: List[str],
        input_end: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, bool]]:
        directives: List[Dict[str, Any]] = []
        flags = {"draco": False, "smd18": False}

        for i, line in enumerate(lines[:input_end]):
            match = self._SIMPLE_INPUT_RE.match(line)
            if not match:
                continue

            input_line = int(match.group(1))
            raw_text = match.group(2).strip()
            for token in raw_text.split():
                upper = token.upper()
                if upper == "NOCPCM":
                    directives.append({
                        "source": "input_simple",
                        "line_index": i,
                        "input_line": input_line,
                        "model": "NOCPCM",
                        "solvent": None,
                        "raw": token,
                    })
                    continue

                model_match = self._MODEL_TOKEN_RE.match(token)
                if model_match:
                    directives.append({
                        "source": "input_simple",
                        "line_index": i,
                        "input_line": input_line,
                        "model": self._canonical_model(model_match.group(1)),
                        "solvent": self._clean_string(model_match.group(2)),
                        "raw": token,
                    })
                    continue

                if upper == "DRACO":
                    flags["draco"] = True
                elif upper == "SMD18":
                    flags["smd18"] = True

        return directives, flags

    def _parse_input_blocks(
        self,
        lines: List[str],
        input_end: int,
        block_name: str,
    ) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []
        i = 0

        while i < input_end:
            start_match = self._INPUT_BLOCK_START_RE.match(lines[i])
            if not start_match or start_match.group(2).lower() != block_name:
                i += 1
                continue

            block: Dict[str, Any] = {
                "block": block_name,
                "line_index": i,
                "input_line": int(start_match.group(1)),
                "settings": {},
                "raw_lines": [],
            }

            j = i + 1
            while j < input_end:
                current = lines[j]
                if self._INPUT_BLOCK_END_RE.match(current):
                    break
                block["raw_lines"].append(current.strip())
                line_match = self._INPUT_BLOCK_LINE_RE.match(current)
                if line_match:
                    key = line_match.group(1).lower()
                    value = line_match.group(2).strip()
                    if value:
                        block["settings"][key] = self._parse_scalar(value)
                j += 1

            blocks.append(block)
            i = j + 1

        return blocks

    def _infer_directives_from_cpcm_blocks(
        self,
        blocks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        directives: List[Dict[str, Any]] = []
        for block in blocks:
            settings = block.get("settings", {})
            smd_enabled = bool(settings.get("smd"))
            model = "SMD" if smd_enabled else None
            solvent = settings.get("smdsolvent") if smd_enabled else None

            if model is None and settings.get("solvent") is not None:
                model = "CPCM"
                solvent = settings.get("solvent")

            if model is not None:
                directives.append({
                    "source": "input_block",
                    "line_index": block.get("line_index", -1),
                    "input_line": block.get("input_line"),
                    "model": model,
                    "solvent": self._clean_string(solvent),
                    "block": "cpcm",
                })
        return directives

    def _infer_directives_from_cosmors_blocks(
        self,
        blocks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        directives: List[Dict[str, Any]] = []
        for block in blocks:
            settings = block.get("settings", {})
            solvent = settings.get("solvent") or settings.get("solventfilename")
            if solvent is None:
                continue
            directives.append({
                "source": "input_block",
                "line_index": block.get("line_index", -1),
                "input_line": block.get("input_line"),
                "model": "COSMO-RS",
                "solvent": self._clean_string(solvent),
                "block": "cosmors",
            })
        return directives

    def _parse_cpcm_output_blocks(self, lines: List[str]) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []
        header_indices = [
            i for i, line in enumerate(lines)
            if self._CPCM_HEADER_RE.match(line.strip())
        ]

        for block_index, idx in enumerate(header_indices):
            block: Dict[str, Any] = {
                "block_index": block_index,
                "line_index": idx,
                "model": "CPCM",
            }
            section: Optional[str] = None
            window = lines[idx: min(idx + 160, len(lines))]

            for line in window:
                stripped = line.strip()
                upper = stripped.upper()

                if "SMD-CDS SOLVENT DESCRIPTORS" in upper:
                    block["model"] = "SMD"
                    section = "smd_descriptors"
                    continue
                if stripped == "Radii:":
                    section = "radii"
                    continue
                if stripped == "CPCM parameters:":
                    section = "cpcm_parameters"
                    continue

                key_match = self._KV_DOTS_RE.match(line)
                if key_match:
                    key = key_match.group(1).strip().rstrip(":").lower()
                    raw_value = key_match.group(2).strip()

                    if key == "epsilon":
                        block["epsilon"] = self.safe_float(raw_value.split()[0])
                    elif key == "refrac":
                        block["refractive_index"] = self.safe_float(
                            raw_value.split()[0]
                        )
                    elif key == "rsolv":
                        block["rsolv_ang"] = self.safe_float(raw_value.split()[0])
                    elif key == "surface type":
                        block["surface_type"] = raw_value
                    elif key == "discretization scheme":
                        block["discretization_scheme"] = raw_value
                    elif key == "epsilon function type":
                        block["epsilon_function_type"] = raw_value
                    elif key == "solvent":
                        block["solvent"] = self._clean_string(raw_value)
                    elif key == "cavity surface points":
                        block["cavity_surface_points"] = self.safe_int(
                            raw_value.split()[0]
                        )
                    elif key == "cavity volume":
                        block["cavity_volume"] = self.safe_float(
                            raw_value.split()[0]
                        )
                    elif key == "cavity surface-area":
                        block["cavity_surface_area"] = self.safe_float(
                            raw_value.split()[0]
                        )
                    elif key == "scheme" and section == "radii":
                        block["radii_scheme"] = raw_value
                    elif section == "smd_descriptors" and key in {
                        "soln", "soln25", "sola", "solb", "solg", "solc", "solh"
                    }:
                        block.setdefault("smd_descriptors", {})[key] = (
                            self.safe_float(raw_value.split()[0])
                        )

                dielectric_match = self._CPCM_DIELECTRIC_RE.search(line)
                if dielectric_match:
                    block["cpcm_dielectric_Eh"] = float(dielectric_match.group(1))

                smd_component_match = self._SMD_CDS_COMPONENT_RE.search(line)
                if smd_component_match:
                    block["model"] = "SMD"
                    block["smd_cds_Eh"] = float(smd_component_match.group(1))

                smd_corr_match = self._SMD_CDS_CORR_RE.search(line)
                if smd_corr_match:
                    block["model"] = "SMD"
                    block["smd_cds_kcal_mol"] = float(smd_corr_match.group(1))

                surface_charge_match = self._SURFACE_CHARGE_RE.search(line)
                if surface_charge_match:
                    block["surface_charge"] = float(surface_charge_match.group(1))

                corrected_charge_match = self._CORRECTED_CHARGE_RE.search(line)
                if corrected_charge_match:
                    block["corrected_charge"] = float(
                        corrected_charge_match.group(1)
                    )

                outlying_match = self._OUTLYING_CHARGE_RE.search(line)
                if outlying_match:
                    block["outlying_charge_correction_Eh"] = float(
                        outlying_match.group(1)
                    )

                free_energy_match = self._FREE_ENERGY_CAV_DISP_RE.search(line)
                if free_energy_match:
                    block["free_energy_cav_disp_Eh"] = float(
                        free_energy_match.group(1)
                    )

            if len(block) > 3:
                blocks.append(block)

        return blocks

    def _parse_alpb_output_blocks(self, lines: List[str]) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []

        for idx, line in enumerate(lines):
            header_match = self._ALPB_HEADER_RE.match(line)
            if not header_match:
                continue

            block: Dict[str, Any] = {
                "block_index": len(blocks),
                "line_index": idx,
                "model": header_match.group(1).upper(),
            }

            for window_line in lines[max(0, idx - 10): min(idx + 40, len(lines))]:
                program_call_match = self._ALPB_PROGRAM_CALL_RE.search(window_line)
                if program_call_match:
                    block["solvent_from_program_call"] = self._clean_string(
                        program_call_match.group(1)
                    )

            j = idx + 1
            while j < len(lines):
                current = lines[j]
                stripped = current.strip()
                if not stripped:
                    break
                if j != idx and self._ALPB_HEADER_RE.match(current):
                    break

                spaced_match = self._ALPB_SPACED_RE.match(current)
                if spaced_match:
                    key = spaced_match.group(1).strip().lower()
                    raw_value = spaced_match.group(2).strip()
                    if key == "solvent":
                        block["solvent"] = self._clean_string(raw_value)
                    elif key == "parameter file":
                        block["parameter_file"] = raw_value
                    elif key == "dielectric constant":
                        block["epsilon"] = self.safe_float(raw_value.split()[0])
                    elif key == "reference state":
                        block["reference_state"] = raw_value
                    elif key == "temperature":
                        block["temperature_k"] = self.safe_float(
                            raw_value.split()[0]
                        )
                    elif key == "density":
                        block["density_kg_l"] = self.safe_float(
                            raw_value.split()[0]
                        )
                    elif key == "solvent mass":
                        block["solvent_mass_g_mol"] = self.safe_float(
                            raw_value.split()[0]
                        )
                    elif key == "interaction kernel":
                        block["interaction_kernel"] = raw_value
                    elif key == "h-bond correction":
                        block["h_bond_correction"] = self._parse_scalar(raw_value)
                    elif key == "ion screening":
                        block["ion_screening"] = self._parse_scalar(raw_value)

                shift_match = self._FREE_ENERGY_SHIFT_RE.search(current)
                if shift_match:
                    block["free_energy_shift_Eh"] = float(shift_match.group(1))
                    block["free_energy_shift_kcal_mol"] = float(shift_match.group(2))

                tension_match = self._SURFACE_TENSION_RE.search(current)
                if tension_match:
                    block["surface_tension_Eh"] = float(tension_match.group(1))
                    block["surface_tension_dyn_cm"] = float(tension_match.group(2))

                grid_match = self._GRID_POINTS_RE.search(current)
                if grid_match:
                    block["grid_points_per_atom"] = int(grid_match.group(1))

                j += 1

            if "solvent" not in block and "solvent_from_program_call" in block:
                block["solvent"] = block["solvent_from_program_call"]

            if len(block) > 3:
                blocks.append(block)

        return blocks

    def _parse_cosmors_output_blocks(self, lines: List[str]) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []
        header_indices = [
            i for i, line in enumerate(lines)
            if self._COSMORS_HEADER_RE.match(line.strip())
        ]

        for block_index, idx in enumerate(header_indices):
            block: Dict[str, Any] = {
                "block_index": block_index,
                "line_index": idx,
                "model": "COSMO-RS",
            }
            section: Optional[str] = None

            for line in lines[idx: min(idx + 220, len(lines))]:
                stripped = line.strip()
                if stripped == "GENERAL INFORMATION":
                    section = "general"
                    continue
                if stripped == "SOLVENT INFORMATION":
                    section = "solvent"
                    continue
                if stripped == "SOLVATION DATA":
                    section = "solvation"
                    continue

                key_match = self._KV_DOTS_RE.match(line)
                if key_match:
                    key = key_match.group(1).strip().rstrip(":").lower()
                    raw_value = key_match.group(2).strip()

                    if section == "general":
                        if key == "calculation method":
                            block["calculation_method"] = raw_value
                        elif key == "functional":
                            block["functional"] = raw_value
                        elif key == "basis set":
                            block["basis_set"] = raw_value
                    elif section == "solvent":
                        if key == "solvent name":
                            block["solvent"] = self._clean_string(raw_value)
                        elif key == "number of atoms":
                            block["solvent_n_atoms"] = self.safe_int(
                                raw_value.split()[0]
                            )
                        elif key == "total charge":
                            block["solvent_charge"] = self.safe_int(
                                raw_value.split()[0]
                            )
                        elif key == "multiplicity":
                            block["solvent_multiplicity"] = self.safe_int(
                                raw_value.split()[0]
                            )
                    elif section == "solvation" and key == "reference temperature":
                        block["reference_temperature_k"] = self.safe_float(
                            raw_value.split()[0]
                        )

                dgsolv_match = self._DGSOLV_RE.search(line)
                if dgsolv_match:
                    block["dGsolv_Eh"] = float(dgsolv_match.group(1))
                    block["dGsolv_kcal_mol"] = float(dgsolv_match.group(2))

            if len(block) > 3:
                blocks.append(block)

        return blocks

    def _build_summary(
        self,
        input_directives: List[Dict[str, Any]],
        input_flags: Dict[str, bool],
        cpcm_blocks: List[Dict[str, Any]],
        alpb_blocks: List[Dict[str, Any]],
        cosmors_blocks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}

        input_state: Optional[Dict[str, Any]] = None
        for directive in input_directives:
            if directive.get("model") == "NOCPCM":
                input_state = None
            else:
                input_state = directive

        output_blocks: List[Dict[str, Any]] = []
        output_blocks.extend(cpcm_blocks)
        output_blocks.extend(alpb_blocks)
        output_blocks.extend(cosmors_blocks)
        latest_output = max(
            output_blocks,
            key=lambda item: item.get("line_index", -1),
            default=None,
        )

        explicit_input_control = bool(input_directives)
        summary["input_controlled"] = explicit_input_control
        summary["input_directive_count"] = len(input_directives)
        summary["draco"] = bool(input_flags.get("draco"))
        summary["smd18"] = bool(input_flags.get("smd18"))
        summary["cpcm_block_count"] = len(cpcm_blocks)
        summary["alpb_block_count"] = len(alpb_blocks)
        summary["cosmors_block_count"] = len(cosmors_blocks)

        if explicit_input_control:
            summary["is_solvated"] = input_state is not None
            summary["primary_model"] = input_state.get("model") if input_state else None
            summary["solvent"] = input_state.get("solvent") if input_state else None
            summary["input_model"] = input_state.get("model") if input_state else None
            summary["input_solvent"] = input_state.get("solvent") if input_state else None
        else:
            summary["is_solvated"] = latest_output is not None
            summary["primary_model"] = latest_output.get("model") if latest_output else None
            summary["solvent"] = latest_output.get("solvent") if latest_output else None
            summary["input_model"] = None
            summary["input_solvent"] = None

        if latest_output is not None:
            summary["output_model"] = latest_output.get("model")
            summary["output_solvent"] = latest_output.get("solvent")
            if summary.get("is_solvated") and latest_output.get("solvent"):
                summary["solvent"] = latest_output.get("solvent")

        models = {
            directive.get("model")
            for directive in input_directives
            if directive.get("model") and directive.get("model") != "NOCPCM"
        }
        models.update(block.get("model") for block in output_blocks if block.get("model"))
        summary["models"] = sorted(models)

        if cpcm_blocks:
            latest_cpcm = cpcm_blocks[-1]
            if latest_cpcm.get("epsilon") is not None:
                summary["epsilon"] = latest_cpcm.get("epsilon")
            if latest_cpcm.get("surface_type") is not None:
                summary["surface_type"] = latest_cpcm.get("surface_type")
        elif alpb_blocks:
            latest_alpb = alpb_blocks[-1]
            if latest_alpb.get("epsilon") is not None:
                summary["epsilon"] = latest_alpb.get("epsilon")

        return summary

    @staticmethod
    def _canonical_model(raw_model: str) -> str:
        raw = raw_model.strip().upper()
        if raw == "COSMORS":
            return "COSMO-RS"
        return raw

    @staticmethod
    def _clean_string(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip().strip('"').strip("'")
        return text or None

    def _parse_scalar(self, value: str) -> Any:
        stripped = value.strip()
        if not stripped:
            return ""

        cleaned = self._clean_string(stripped)
        if cleaned is None:
            return ""

        upper = cleaned.upper()
        if upper == "TRUE":
            return True
        if upper == "FALSE":
            return False

        integer = self.safe_int(cleaned)
        if integer is not None and re.fullmatch(r"[-+]?\d+", cleaned):
            return integer

        floating = self.safe_float(cleaned)
        if floating is not None and re.fullmatch(
            r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?",
            cleaned,
        ):
            return floating

        return cleaned


PLUGIN_BUNDLE = PluginBundle(
    metadata=PluginMetadata(
        key="solvation_section",
        name="Solvation Section",
        short_help="Built-in implicit-solvation parser section owned by solvation.py.",
        description=(
            "Self-registering built-in parser section for CPCM, SMD, ALPB, "
            "and COSMO-RS input/output metadata."
        ),
        docs_path="README.md",
        examples=(
            "orca_parser solvated.out --sections solvation",
        ),
    ),
    parser_sections=(
        ParserSectionPlugin("solvation", SolvationModule),
    ),
    parser_aliases=(
        ParserSectionAlias(name="solvation", section_keys=("solvation",)),
    ),
)
