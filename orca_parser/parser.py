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

from .modules import (
    MetadataModule,
    GeometryModule,
    BasisSetModule,
    SCFModule,
    OrbitalEnergiesModule,
    QROModule,
    MullikenModule,
    LoewdinModule,
    MayerModule,
    HirshfeldModule,
    MBISModule,
    CHELPGModule,
    DipoleMomentModule,
    SolvationModule,
    TDDFTModule,
    NBOModule,
    EPRModule,
    GeomOptModule,
)

# ─────────────────────────────────────────────────────────────────
# Registry: ordered list of (key, ModuleClass) pairs.
# The key is used in the output dict.
# ─────────────────────────────────────────────────────────────────
MODULE_REGISTRY: List[tuple] = [
    ("metadata",         MetadataModule),
    ("geometry",         GeometryModule),
    ("basis_set",        BasisSetModule),
    ("scf",              SCFModule),
    ("orbital_energies", OrbitalEnergiesModule),
    ("qro",              QROModule),
    ("mulliken",         MullikenModule),
    ("loewdin",          LoewdinModule),
    ("mayer",            MayerModule),
    ("hirshfeld",        HirshfeldModule),
    ("mbis",             MBISModule),
    ("chelpg",           CHELPGModule),
    ("dipole",           DipoleMomentModule),
    ("solvation",        SolvationModule),
    ("tddft",            TDDFTModule),
    ("nbo",              NBOModule),
    ("epr",              EPRModule),
    ("geom_opt",         GeomOptModule),
]

# Sections always included regardless of the `sections` argument.
_CORE_SECTIONS: set = {"metadata", "geometry", "basis_set", "scf"}

# Convenience aliases that expand to lists of section keys.
SECTION_ALIASES: Dict[str, List[str]] = {
    "all":        [key for key, _ in MODULE_REGISTRY],
    "charges":    ["mulliken", "loewdin", "hirshfeld", "mbis", "chelpg"],
    "population": ["mulliken", "loewdin", "mayer"],
    "mos":        ["orbital_energies", "qro"],
    "bonds":      ["mayer", "loewdin"],
    "nbo":        ["nbo"],
    "dipole":     ["dipole"],
    "solvation":  ["solvation"],
    "tddft":      ["tddft"],
    "geometry":   ["geometry", "basis_set"],
    "epr":        ["epr"],
    "opt":        ["geom_opt"],
}

_AUXILIARY_ATOM_FILE_RE = re.compile(
    r"_atom\d+\.(?:out|log)$",
    re.IGNORECASE,
)


def is_auxiliary_orca_file(path: str | Path) -> bool:
    """Return True for ORCA helper atom/ECP files such as ``*_atom83.out``."""
    return bool(_AUXILIARY_ATOM_FILE_RE.search(Path(path).name))


def _resolve_sections(sections) -> Optional[set]:
    """Expand aliases and return the full set of section keys to run.

    Returns None to indicate 'run everything'.
    """
    if sections is None:
        return None
    if isinstance(sections, str):
        sections = [sections]
    requested: set = set()
    for token in sections:
        token = token.lower().strip()
        if token == "all":
            return None  # short-circuit
        if token in SECTION_ALIASES:
            requested.update(SECTION_ALIASES[token])
        else:
            requested.add(token)
    # Always add core sections
    requested.update(_CORE_SECTIONS)
    return requested


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
        ``is_uhf``, ``has_symmetry``, ``hf_type``, ``multiplicity``,
        ``n_atoms``, ``atom_symbols``.
    """

    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)
        self.data: Dict[str, Any] = {}
        self.context: Dict[str, Any] = {}
        self._lines: List[str] = []

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
        results["context"] = dict(self.context)

        for key, ModClass in MODULE_REGISTRY:
            if active is not None and key not in active:
                continue
            try:
                mod = ModClass(self.context)
                result = mod.parse(self._lines)
                if result is not None:
                    results[key] = result
            except Exception as exc:  # noqa: BLE001
                results[f"{key}_parse_error"] = str(exc)

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

    def to_markdown(self, path: str | Path) -> Path:
        """Write a compact, AI-readable markdown report.

        Designed for feeding into LLMs for paper writing: maximum information
        density, publication-ready tables, spin diagnostics up front.
        """
        from .output.markdown_writer import write_markdown
        return write_markdown(self.data, Path(path))

    # ---------------------------------------------------------------- #
    # Class-level utilities                                             #
    # ---------------------------------------------------------------- #

    @classmethod
    def compare(
        cls,
        parsers: "list[ORCAParser]",
        path: str | Path,
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
        return write_comparison(datasets, Path(path))



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
            "has_symmetry": False,
            "hf_type": "RHF",
            "multiplicity": 1,
            "n_atoms": 0,
            "atom_symbols": [],
        }

        for ln in self._lines:
            # HF type → UHF detection
            m = re.search(r"Hartree-Fock type\s+HFTyp\s+\.\.\.\.\s+(\w+)", ln)
            if m:
                ctx["hf_type"] = m.group(1)
                ctx["is_uhf"] = m.group(1).upper() == "UHF"

            # Multiplicity
            m = re.search(r"Multiplicity\s+Mult\s+\.\.\.\.\s+(\d+)", ln)
            if m:
                ctx["multiplicity"] = int(m.group(1))

            # Symmetry detection: presence of irrep labels
            if "Number of irreps" in ln:
                ctx["has_symmetry"] = True

            # Number of atoms
            m = re.search(r"Number of atoms\s+\.\.\.\s+(\d+)", ln)
            if m:
                ctx["n_atoms"] = int(m.group(1))

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
