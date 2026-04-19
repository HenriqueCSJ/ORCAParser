"""
Main ORCA output file parser.

Usage
-----
from orca_parser import ORCAParser

p = ORCAParser("water_nosym_rks.out")
data = p.parse()
p.to_json("result.json")
p.to_csv("result_dir/")
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .final_snapshot import build_final_snapshot
from .input_semantics import (
    detect_input_symmetry_request,
    infer_reference_type,
    is_unrestricted_reference,
)
from .job_series import build_job_series
from .job_snapshot import build_job_snapshot
from .plugin_discovery import bootstrap_plugin_bundles
from .parser_section_registry import (
    get_core_parser_section_keys,
    get_parser_section_alias_map,
    get_registered_parser_section_plugins,
    iter_active_parser_section_plugins,
    resolve_requested_parser_sections,
)
from .render_options import RENDER_OPTION_UNSET

# Discover explicit plugin bundles before freezing compatibility snapshots so
# newly dropped-in modules participate in normal parser flows automatically.
bootstrap_plugin_bundles()

# Legacy compatibility snapshots for external code that still imports these
# names from ``orca_parser.parser``. The parser itself now consumes the
# registry helpers directly.
MODULE_REGISTRY: List[tuple] = [
    (plugin.key, plugin.module_class)
    for plugin in get_registered_parser_section_plugins()
]
_CORE_SECTIONS: set = get_core_parser_section_keys()
SECTION_ALIASES: Dict[str, List[str]] = get_parser_section_alias_map()

_AUXILIARY_ATOM_FILE_RE = re.compile(
    r"_atom\d+\.(?:out|log)$",
    re.IGNORECASE,
)
_INPUT_NAME_RE = re.compile(r"^NAME\s*=\s*(\S+)")
_INPUT_END_RE = re.compile(r"\*{4}END OF INPUT\*{4}", re.IGNORECASE)
_INPUT_ECHO_LINE_RE = re.compile(r"^\|\s*(\d+)>\s*(.*)$")
_INPUT_BANG_RE = re.compile(r"^\|\s*\d+>\s*!\s*(.+)$")
_INPUT_BLOCK_START_RE = re.compile(r"^\|\s*\d+>\s*%([A-Za-z][\w-]*)\b\s*(.*)$")
_INPUT_BLOCK_END_RE = re.compile(r"^\|\s*\d+>\s*end\b", re.IGNORECASE)
_INPUT_STRUCTURE_RE = re.compile(
    r"^\|\s*\d+>\s*\*\s*(\w+)\s+(-?\d+)\s+(\d+)(?:\s+(\S+))?",
    re.IGNORECASE,
)


def _normalize_job_path(path: Path) -> str:
    """Return a stable path string suitable for per-job IDs."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _coerce_input_value(value: str) -> Any:
    """Best-effort scalar coercion for echoed input settings."""
    cleaned = value.split("#", 1)[0].strip()
    if not cleaned:
        return ""

    lowered = cleaned.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    try:
        if any(ch in cleaned for ch in (".", "e", "E")):
            return float(cleaned)
        return int(cleaned)
    except ValueError:
        return cleaned


def _parse_input_block_setting(content: str) -> Optional[tuple[str, Any]]:
    """Parse a simple ORCA echoed-input setting line into key/value."""
    stripped = content.strip()
    if not stripped or stripped.startswith("!"):
        return None

    if "=" in stripped:
        left, right = stripped.split("=", 1)
        key = left.strip().split()[0].lower()
        return key, _coerce_input_value(right)

    parts = stripped.split(None, 1)
    if len(parts) == 2:
        return parts[0].lower(), _coerce_input_value(parts[1])
    return parts[0].lower(), True


def _parse_input_echo(lines: List[str]) -> Dict[str, Any]:
    """Parse the echoed ORCA input block once for run-scoped metadata."""
    input_end = next(
        (idx for idx, line in enumerate(lines) if _INPUT_END_RE.search(line)),
        len(lines),
    )

    data: Dict[str, Any] = {
        "bang_lines": [],
        "bang_tokens": [],
        "block_names": [],
        "blocks": {},
    }

    i = 0
    while i < input_end:
        line = lines[i]

        name_match = _INPUT_NAME_RE.match(line)
        if name_match:
            data["input_name"] = name_match.group(1)
            i += 1
            continue

        bang_match = _INPUT_BANG_RE.match(line)
        if bang_match:
            bang_line = bang_match.group(1).strip()
            if bang_line:
                data["bang_lines"].append(bang_line)
                data["bang_tokens"].extend(bang_line.split())
            i += 1
            continue

        structure_match = _INPUT_STRUCTURE_RE.match(line)
        if structure_match:
            data["structure_input"] = {
                "kind": structure_match.group(1).lower(),
                "charge": int(structure_match.group(2)),
                "multiplicity": int(structure_match.group(3)),
                "source": structure_match.group(4) or "",
            }
            i += 1
            continue

        block_match = _INPUT_BLOCK_START_RE.match(line)
        if not block_match:
            i += 1
            continue

        block_name = block_match.group(1).lower()
        header_remainder = block_match.group(2).strip()
        block: Dict[str, Any] = {
            "name": block_name,
            "raw_lines": [],
            "settings": {},
        }
        if header_remainder:
            block["inline"] = True
            block["raw_lines"].append(header_remainder)
            parsed = _parse_input_block_setting(header_remainder)
            if parsed:
                key, value = parsed
                block["settings"][key] = value
            data["block_names"].append(block_name)
            data["blocks"][block_name] = block
            i += 1
            continue

        j = i + 1
        while j < input_end:
            inner_line = lines[j]
            if _INPUT_BLOCK_END_RE.match(inner_line):
                break
            inner_match = _INPUT_ECHO_LINE_RE.match(inner_line)
            if inner_match:
                content = inner_match.group(2).strip()
                if content:
                    block["raw_lines"].append(content)
                    parsed = _parse_input_block_setting(content)
                    if parsed:
                        key, value = parsed
                        block["settings"][key] = value
            j += 1

        data["block_names"].append(block_name)
        data["blocks"][block_name] = block
        i = j + 1 if j < input_end else j

    if data["bang_lines"]:
        data["bang_text"] = " ".join(data["bang_lines"])
    return data


def is_auxiliary_orca_file(path: str | Path) -> bool:
    """Return True for ORCA helper atom/ECP files such as ``*_atom83.out``."""
    return bool(_AUXILIARY_ATOM_FILE_RE.search(Path(path).name))


def _resolve_sections(sections) -> Optional[set]:
    """Compatibility wrapper around the parser-section registry."""
    return resolve_requested_parser_sections(sections)


class ORCAParser:
    """
    Parses a single ORCA output file and extracts all available properties
    through a modular, extensible architecture.

    Parameters
    ----------
    filepath : str or Path
        Path to the ORCA .out file.

    Attributes
    ----------
    data : dict
        Parsed results, populated after calling :meth:`parse`.
    context : dict
        Shared flags set from the metadata pass:
        ``is_unrestricted`` / compatibility alias ``is_uhf``,
        ``reference_type``, ``hf_type``, ``has_symmetry``,
        ``multiplicity``, ``n_atoms``, ``atom_symbols``.
    """

    def __init__(
        self,
        filepath: str | Path,
        *,
        plugin_options: Optional[Dict[str, Any]] = None,
    ):
        self.filepath = Path(filepath)
        self.data: Dict[str, Any] = {}
        self.context: Dict[str, Any] = {}
        self._lines: List[str] = []
        self.plugin_options: Dict[str, Any] = dict(plugin_options or {})

    # ---------------------------------------------------------------- #
    # Public API                                                        #
    # ---------------------------------------------------------------- #

    def parse(self, sections=None) -> Dict[str, Any]:
        """Read the file and run selected modules. Returns the data dict.

        Parameters
        ----------
        sections : str, list of str, or None
            Which sections to parse. Accepts individual section keys
            (e.g. ``"mulliken"``) or group aliases:

            * ``"all"``        — everything (default)
            * ``"charges"``    — mulliken, loewdin, hirshfeld, mbis, chelpg
            * ``"population"`` — mulliken, loewdin, mayer
            * ``"mos"``        — orbital_energies, qro
            * ``"bonds"``      — mayer, loewdin
            * ``"nbo"``        — nbo
            * ``"dipole"``     — dipole
            * ``"solvation"``  — solvation
            * ``"tddft"``      — tddft
            * ``"geometry"``   — geometry, basis_set

            Core sections (metadata, geometry, basis_set, scf) are always
            included. Combine freely: ``["charges", "mos", "dipole"]``.
        """
        with open(self.filepath, "r", encoding="utf-8", errors="replace") as fh:
            self._lines = [ln.rstrip("\n") for ln in fh.readlines()]

        if is_auxiliary_orca_file(self.filepath):
            raise ValueError(
                "Auxiliary ORCA atom/ECP files are not included in parsing: "
                f"{self.filepath.name}"
            )

        if not self._is_orca_output():
            raise ValueError(
                f"Not a valid ORCA output file: {self.filepath.name}"
            )

        self._build_context()

        active = _resolve_sections(sections)  # None → run all

        results: Dict[str, Any] = {
            "source_file": str(self.filepath),
        }
        if self.plugin_options:
            results["plugin_options"] = dict(self.plugin_options)
        results["context"] = dict(self.context)

        for plugin in iter_active_parser_section_plugins(active):
            try:
                mod = plugin.module_class(self.context)
                result = mod.parse(self._lines)
                if result is not None:
                    results[plugin.key] = result
            except Exception as exc:  # noqa: BLE001
                results[f"{plugin.key}_parse_error"] = str(exc)

        self._postprocess_results(results)
        results["context"] = dict(self.context)
        self.data = results
        return results

    def to_json(
        self,
        path: str | Path,
        indent: int = 2,
        strip_none: bool = False,
        compress: bool = False,
    ) -> Path:
        """Serialize data to a JSON file.

        Parameters
        ----------
        path : str or Path
            Destination path. If *compress* is True and the path does not
            already end with ``.gz``, the suffix is appended automatically.
        indent : int
            Indentation level. Use ``0`` or ``None`` for compact one-liners.
        strip_none : bool
            Remove keys whose value is ``None`` recursively. Reduces file
            size noticeably for sparse outputs.
        compress : bool
            Write gzip-compressed JSON (``.json.gz``).
        """
        from .output.json_writer import write_json
        return write_json(
            self.data, Path(path),
            indent=indent,
            strip_none=strip_none,
            compress=compress,
        )

    def to_csv(self, directory: str | Path) -> List[Path]:
        """Write flat CSV tables to *directory*. Returns list of created files."""
        from .output.csv_writer import write_csvs
        return write_csvs(self.data, Path(directory))

    def to_hdf5(
        self,
        path: str | Path,
        compression: str = "gzip",
        compression_opts: int = 4,
    ) -> Path:
        """Serialize data to an HDF5 file.

        Parameters
        ----------
        path : str or Path
            Destination ``.h5`` file.
        compression : str
            HDF5 compression filter (``"gzip"``, ``"lzf"``, or ``None``).
        compression_opts : int
            Compression level for gzip (1–9).
        """
        from .output.hdf5_writer import write_hdf5
        return write_hdf5(
            self.data, Path(path),
            compression=compression,
            compression_opts=compression_opts,
        )

    def to_markdown(
        self,
        path: str | Path,
        *,
        goat_max_relative_energy_kcal_mol=RENDER_OPTION_UNSET,
        detail_scope: str = "auto",
    ) -> Path:
        """Write a compact, AI-readable markdown report.

        Designed for feeding into LLMs for paper writing: maximum information
        density, publication-ready tables, spin diagnostics up front.
        """
        from .output.markdown_writer import write_markdown
        return write_markdown(
            self.data,
            Path(path),
            goat_max_relative_energy_kcal_mol=goat_max_relative_energy_kcal_mol,
            detail_scope=detail_scope,
        )

    # ---------------------------------------------------------------- #
    # Class-level utilities                                             #
    # ---------------------------------------------------------------- #

    @classmethod
    def compare(
        cls,
        parsers: "list[ORCAParser]",
        path: str | Path,
        *,
        goat_max_relative_energy_kcal_mol=RENDER_OPTION_UNSET,
        detail_scope: str = "auto",
    ) -> Path:
        """Write a multi-molecule comparison markdown document.

        Parameters
        ----------
        parsers : list of ORCAParser
            Already-parsed parser instances (``parse()`` must have been called).
        path : str or Path
            Destination ``.md`` file.
        """
        from .output.markdown_writer import write_comparison
        datasets = [p.data for p in parsers]
        return write_comparison(
            datasets,
            Path(path),
            goat_max_relative_energy_kcal_mol=goat_max_relative_energy_kcal_mol,
            detail_scope=detail_scope,
        )



    def _is_orca_output(self) -> bool:
        """Check that the file looks like a valid ORCA output.

        Requires *both*:
        - An ORCA program banner somewhere near the top (first 200 lines)
        - ``ORCA TERMINATED NORMALLY`` or ``TOTAL RUN TIME`` near the end
          (last 50 lines), indicating the calculation completed.
        """
        head = self._lines[:200]
        tail = self._lines[-50:] if len(self._lines) > 50 else self._lines

        has_banner = any(
            "An Ab Initio, DFT and Semiempirical electronic structure package"
            in ln or "Program Version" in ln
            for ln in head
        )
        has_termination = any(
            "ORCA TERMINATED NORMALLY" in ln or "TOTAL RUN TIME" in ln
            for ln in tail
        )
        return has_banner and has_termination

    def _build_context(self) -> None:
        """
        First-pass scan to populate context flags that modules depend on.
        This runs *before* the module loop so every module gets accurate flags.
        """
        ctx: Dict[str, Any] = {
            "is_uhf": False,
            "is_unrestricted": False,
            "has_symmetry": False,
            "is_surface_scan": False,
            "hf_type": "RHF",
            "reference_type": "RHF",
            "charge": 0,
            "multiplicity": 1,
            "n_atoms": 0,
            "atom_symbols": [],
            "source_path": str(self.filepath),
            "source_dir": str(self.filepath.parent),
            "source_stem": self.filepath.stem,
            "source_relpath": _normalize_job_path(self.filepath),
            "job_id": _normalize_job_path(self.filepath),
            "plugin_options": dict(self.plugin_options),
        }

        for ln in self._lines:
            # HF type → UHF detection
            m = re.search(r"Hartree-Fock type\s+HFTyp\s+\.\.\.\.\s+(\w+)", ln)
            if m:
                ctx["hf_type"] = m.group(1)

            # Multiplicity
            m = re.search(r"Total Charge\s+Charge\s+\.\.\.\.\s+(-?\d+)", ln)
            if m:
                ctx["charge"] = int(m.group(1))

            m = re.search(r"Multiplicity\s+Mult\s+\.\.\.\.\s+(\d+)", ln)
            if m:
                ctx["multiplicity"] = int(m.group(1))

            # Symmetry detection: presence of irrep labels
            if "Number of irreps" in ln:
                ctx["has_symmetry"] = True

            if "Relaxed Surface Scan" in ln:
                ctx["is_surface_scan"] = True

            # Number of atoms
            m = re.search(r"Number of atoms\s+\.\.\.\s+(\d+)", ln)
            if m:
                ctx["n_atoms"] = int(m.group(1))

        input_echo = _parse_input_echo(self._lines)
        if input_echo:
            ctx["input_echo"] = input_echo
            structure_input = input_echo.get("structure_input") or {}
            if "charge" in structure_input:
                ctx["charge"] = structure_input["charge"]
            if "multiplicity" in structure_input:
                ctx["multiplicity"] = structure_input["multiplicity"]
            if "bang_tokens" in input_echo:
                bang_upper = {token.upper() for token in input_echo.get("bang_tokens", [])}
                if "GOAT" in bang_upper:
                    ctx["is_goat"] = True
                input_use_sym = detect_input_symmetry_request(input_echo.get("bang_tokens", []))
                if input_use_sym is not None:
                    ctx["input_use_sym"] = input_use_sym

        # Normalize the user-facing reference semantics once here so every
        # downstream parser module sees the same restricted/unrestricted view.
        ctx["reference_type"] = infer_reference_type(
            bang_tokens=(ctx.get("input_echo") or {}).get("bang_tokens", []),
            hf_type=ctx.get("hf_type"),
        ) or "RHF"
        ctx["is_unrestricted"] = is_unrestricted_reference(ctx["reference_type"])
        # ``is_uhf`` survives as a compatibility alias while the rest of the
        # codebase migrates toward the clearer ``is_unrestricted`` name.
        ctx["is_uhf"] = ctx["is_unrestricted"]

        # Collect atom symbols from Cartesian coordinate block
        idx = -1
        for i, ln in enumerate(self._lines):
            if "CARTESIAN COORDINATES (ANGSTROEM)" in ln:
                idx = i
                break
        if idx != -1:
            symbols = []
            for ln in self._lines[idx + 2:]:
                m = re.match(r"\s+([A-Z][a-z]?)\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+", ln)
                if m:
                    symbols.append(m.group(1))
                elif symbols and ln.strip() == "":
                    break
            ctx["atom_symbols"] = symbols
            if symbols and ctx["n_atoms"] == 0:
                ctx["n_atoms"] = len(symbols)

        self.context = ctx

    def _postprocess_results(self, results: Dict[str, Any]) -> None:
        """Promote cross-module derived metadata after all modules finish."""
        meta = results.get("metadata")
        if not isinstance(meta, dict):
            return

        tddft = results.get("tddft")
        excited_state_opt: Dict[str, Any] = {}
        if isinstance(tddft, dict):
            excited_state_opt = dict(tddft.get("excited_state_optimization") or {})

        if excited_state_opt:
            meta["excited_state_optimization"] = excited_state_opt
            calc_type = str(meta.get("calculation_type", "")).strip().lower()
            if calc_type == "geometry optimization":
                meta["calculation_type"] = "Excited-State Geometry Optimization"

        # Freeze normalized job identity/state metadata once so downstream
        # writers do not have to rediscover it from loosely shaped payloads.
        job_snapshot = build_job_snapshot(results, context=self.context)
        if job_snapshot is not None:
            results["job_snapshot"] = job_snapshot.to_dict()

        # Stepwise histories get the same treatment: GOAT/opt/scan/ES-opt
        # should have one parse-time authority before any output code runs.
        job_series = build_job_series(results)
        if job_series is not None:
            results["job_series"] = job_series.to_dict()

        final_snapshot = build_final_snapshot(results, context=self.context)
        if final_snapshot is not None:
            results["final_snapshot"] = final_snapshot.to_dict()
