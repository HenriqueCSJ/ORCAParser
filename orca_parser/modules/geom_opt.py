"""
Geometry optimization parser module.

Extracts per-cycle energies, geometries, convergence criteria, trust radii,
internal coordinate extrema, and the final stationary-point structure.
Computes RMSD between initial/final and consecutive geometries when numpy
is available.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..job_family_registry import CalculationFamilyPlugin
from ..job_series import get_geom_opt_series
from ..parser_section_plugin import ParserSectionAlias, ParserSectionPlugin
from ..plugin_bundle import PluginBundle, PluginMetadata
from ..render_options import RenderOptions
from .base import BaseModule

# ── Regex patterns ────────────────────────────────────────────────────

_CYCLE_RE = re.compile(r"GEOMETRY OPTIMIZATION CYCLE\s+(\d+)", re.I)
_ENERGY_RE = re.compile(
    r"FINAL SINGLE POINT ENERGY\s+(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)"
)
_TRUST_RE = re.compile(
    r"New trust radius\s+\.+\s+(-?\d+(?:\.\d+)?)"
)
_COORD_RE = re.compile(
    r"^\s*([A-Z][a-z]?)\s+"
    r"(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+"
    r"(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+"
    r"(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s*$"
)
_CONV_RE = re.compile(
    r"^\s*(Energy change|RMS gradient|MAX gradient|RMS step|MAX step)\s+"
    r"(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+"
    r"(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+"
    r"(YES|NO)\s*$",
    re.I,
)
_MAX_INT_1_RE = re.compile(
    r"Max\(Bonds\)\s+(-?\d+(?:\.\d+)?)\s+Max\(Angles\)\s+(-?\d+(?:\.\d+)?)",
    re.I,
)
_MAX_INT_2_RE = re.compile(
    r"Max\(Dihed\)\s+(-?\d+(?:\.\d+)?)\s+Max\(Improp\)\s+(-?\d+(?:\.\d+)?)",
    re.I,
)


# ── Numpy-optional RMSD ──────────────────────────────────────────────

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


def _kabsch_rmsd(
    mobile: list[list[float]],
    reference: list[list[float]],
    weights: list[float],
) -> float:
    """Kabsch-aligned weighted RMSD.  Requires numpy."""
    mob = np.array(mobile, dtype=float)
    ref = np.array(reference, dtype=float)
    w = np.array(weights, dtype=float)
    w_sum = w.sum()

    mob_c = np.sum(mob * w[:, None], axis=0) / w_sum
    ref_c = np.sum(ref * w[:, None], axis=0) / w_sum
    mob_cen = mob - mob_c
    ref_cen = ref - ref_c

    H = mob_cen.T @ (w[:, None] * ref_cen)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T

    aligned = mob_cen @ R + ref_c
    diff = aligned - ref
    sq = np.sum(diff * diff, axis=1)
    return float(np.sqrt(np.sum(w * sq) / w_sum))


def _simple_rmsd(
    coords_a: list[list[float]],
    coords_b: list[list[float]],
) -> float:
    """Plain (no alignment) RMSD — pure Python fallback."""
    n = len(coords_a)
    s = 0.0
    for a, b in zip(coords_a, coords_b):
        s += (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
    return math.sqrt(s / n)


def _compute_rmsd(
    geom_a: list[dict],
    geom_b: list[dict],
    mass_weighted: bool = False,
) -> Optional[float]:
    """Compute RMSD between two geometry dicts.

    Uses Kabsch alignment when numpy is available, plain RMSD otherwise.
    """
    if not geom_a or not geom_b or len(geom_a) != len(geom_b):
        return None

    coords_a = [[a["x"], a["y"], a["z"]] for a in geom_a]
    coords_b = [[b["x"], b["y"], b["z"]] for b in geom_b]

    if _HAS_NUMPY:
        if mass_weighted:
            from ..constants import ATOMIC_MASSES
            weights = [
                ATOMIC_MASSES.get(a["element"], 1.0) for a in geom_a
            ]
        else:
            weights = [1.0] * len(geom_a)
        return _kabsch_rmsd(coords_a, coords_b, weights)

    return _simple_rmsd(coords_a, coords_b)


# ── Cartesian coordinate block parser ────────────────────────────────

def _parse_cartesian_block(lines: list[str], header_idx: int) -> list[dict]:
    """Parse a CARTESIAN COORDINATES (ANGSTROEM) block into atom dicts."""
    atoms: list[dict] = []
    i = header_idx + 1
    # skip dashes / blank lines
    while i < len(lines) and (not lines[i].strip() or set(lines[i].strip()) <= {"-"}):
        i += 1
    while i < len(lines):
        m = _COORD_RE.match(lines[i])
        if not m:
            break
        sym = m.group(1)
        atoms.append({
            "element": sym[0].upper() + sym[1:].lower() if len(sym) > 1 else sym.upper(),
            "x": float(m.group(2)),
            "y": float(m.group(3)),
            "z": float(m.group(4)),
        })
        i += 1
    return atoms


# ── Per-cycle parser ─────────────────────────────────────────────────

def _parse_cycle(cycle_number: int, block: list[str]) -> dict:
    """Extract all data from one optimization cycle block."""
    cycle: Dict[str, Any] = {"cycle": cycle_number}

    # Geometry
    for idx, ln in enumerate(block):
        if "CARTESIAN COORDINATES (ANGSTROEM)" in ln.upper():
            geom = _parse_cartesian_block(block, idx)
            if geom:
                cycle["geometry_angstrom"] = geom
            break

    # Energy — take the last occurrence in the block
    energies = []
    for ln in block:
        m = _ENERGY_RE.search(ln)
        if m:
            energies.append(float(m.group(1)))
    if energies:
        cycle["energy_Eh"] = energies[-1]

    # Trust radius
    for ln in block:
        m = _TRUST_RE.search(ln)
        if m:
            cycle["trust_radius_bohr"] = float(m.group(1))

    # Convergence criteria
    convergence: Dict[str, Dict[str, Any]] = {}
    for ln in block:
        m = _CONV_RE.match(ln)
        if m:
            key = m.group(1).lower().replace(" ", "_")
            convergence[key] = {
                "value": float(m.group(2)),
                "tolerance": float(m.group(3)),
                "converged": m.group(4).upper() == "YES",
            }
    if convergence:
        cycle["convergence"] = convergence

    # Internal coordinate maxima
    internals: Dict[str, float] = {}
    for ln in block:
        m1 = _MAX_INT_1_RE.search(ln)
        if m1:
            internals["max_bond_ang"] = float(m1.group(1))
            internals["max_angle_deg"] = float(m1.group(2))
        m2 = _MAX_INT_2_RE.search(ln)
        if m2:
            internals["max_dihedral_deg"] = float(m2.group(1))
            internals["max_improper_deg"] = float(m2.group(2))
    if internals:
        cycle["internal_coord_maxima"] = internals

    # Convergence signal from ORCA
    cycle["orca_converged"] = any(
        "THE OPTIMIZATION HAS CONVERGED" in ln.upper()
        or "CONVERGENCE WILL THEREFORE BE SIGNALED NOW" in ln.upper()
        for ln in block
    )

    return cycle


# ── Optimization settings parser ─────────────────────────────────────

def _parse_opt_setup(lines: list[str]) -> Dict[str, Dict[str, str]]:
    """Parse the 'Geometry optimization settings' and 'Convergence Tolerances' blocks."""
    settings: Dict[str, str] = {}
    tolerances: Dict[str, str] = {}
    mode: Optional[str] = None

    for ln in lines:
        stripped = ln.strip()
        low = stripped.lower()
        if low == "geometry optimization settings:":
            mode = "settings"
            continue
        if low == "convergence tolerances:":
            mode = "tolerances"
            continue
        if mode is None:
            continue
        if not stripped:
            mode = None
            continue
        if "...." not in ln:
            continue

        left, right = ln.split("....", 1)
        key = left.strip()
        value = right.strip()
        if mode == "settings":
            settings[key] = value
        else:
            tolerances[key] = value

    result: Dict[str, Dict[str, str]] = {}
    if settings:
        result["settings"] = settings
    if tolerances:
        result["tolerances"] = tolerances
    return result


def _matches_geometry_optimization(
    meta: Dict[str, Any],
    data: Dict[str, Any],
    context: Dict[str, Any],
    deltascf: Dict[str, Any],
    excited_state_optimization: Dict[str, Any],
) -> bool:
    """Match plain geometry optimizations from normalized metadata or parsed data."""

    del context, deltascf, excited_state_optimization
    calc_type = str(meta.get("calculation_type", "")).strip().lower()
    return bool(data.get("geom_opt") or "geometry optimization" in calc_type)


def _render_geometry_optimization_markdown_sections(
    data: Dict[str, Any],
    format_number: Callable[[Any, str], str],
    make_table: Callable[[List[tuple]], str],
    render_options: RenderOptions,
) -> List[tuple[str, str]]:
    """Render plain optimization history from the normalized geometry series."""

    geom_opt = get_geom_opt_series(data)
    if not geom_opt:
        return []

    from ..output.markdown_sections_basic import render_geom_opt_section

    body = render_geom_opt_section(
        geom_opt,
        format_number=format_number,
        make_table=make_table,
        cycle_preview_count=render_options.geom_opt_cycle_preview_count,
    )
    return [("Geometry Optimization", body)] if body else []


def _write_geometry_optimization_csv_sections(
    data: Dict[str, Any],
    directory: Path,
    stem: str,
    write_csv: Callable[[Path, str, List[Dict[str, Any]], List[str]], Path],
) -> List[Path]:
    """Write optimization-cycle CSV exports from the family plugin."""

    from ..output.csv_sections_basic import write_geom_opt_section

    return write_geom_opt_section(
        data,
        directory,
        stem,
        write_csv=write_csv,
    )


# ── Module class ─────────────────────────────────────────────────────

class GeomOptModule(BaseModule):
    """Parse geometry optimization runs: per-cycle data, convergence, RMSD."""

    name = "geom_opt"

    def parse(self, lines: list[str]) -> Optional[Dict[str, Any]]:
        if self.context.get("is_surface_scan"):
            return None

        # Detect whether this is a geometry optimization at all
        has_opt_cycle = any(_CYCLE_RE.search(ln) for ln in lines[:5000])
        if not has_opt_cycle:
            return None

        data: Dict[str, Any] = {}

        # Detect OPT keyword variant from input keywords (already parsed
        # by MetadataModule and stored in context or we scan ourselves)
        opt_keyword = self._detect_opt_keyword(lines)
        if opt_keyword:
            data["opt_keyword"] = opt_keyword

        # Optimization settings / tolerances
        setup = _parse_opt_setup(lines)
        if setup:
            data.update(setup)

        # Split into cycle blocks and parse each
        cycle_starts: list[tuple[int, int]] = []
        for i, ln in enumerate(lines):
            m = _CYCLE_RE.search(ln)
            if m:
                cycle_starts.append((i, int(m.group(1))))

        cycles: list[dict] = []
        for idx, (start, cycle_num) in enumerate(cycle_starts):
            # End of this cycle = start of next cycle or stationary-point block
            end = len(lines)
            for j in range(start + 1, len(lines)):
                if _CYCLE_RE.search(lines[j]) or \
                   "FINAL ENERGY EVALUATION AT THE STATIONARY POINT" in lines[j].upper():
                    end = j
                    break
            block = lines[start:end]
            cycles.append(_parse_cycle(cycle_num, block))

        # Final stationary point
        final_geometry: Optional[list[dict]] = None
        final_energy: Optional[float] = None
        stat_idx = self.find_line(lines, "FINAL ENERGY EVALUATION AT THE STATIONARY POINT")
        if stat_idx != -1:
            for j in range(stat_idx + 1, min(stat_idx + 200, len(lines))):
                if "CARTESIAN COORDINATES (ANGSTROEM)" in lines[j].upper():
                    final_geometry = _parse_cartesian_block(lines, j)
                    break
            for j in range(stat_idx + 1, len(lines)):
                m = _ENERGY_RE.search(lines[j])
                if m:
                    final_energy = float(m.group(1))
                    break

        # Fallback to last cycle
        if final_energy is None and cycles:
            final_energy = cycles[-1].get("energy_Eh")
        if final_geometry is None and cycles:
            final_geometry = cycles[-1].get("geometry_angstrom")

        # Initial geometry
        initial_geometry: Optional[list[dict]] = None
        if cycles and "geometry_angstrom" in cycles[0]:
            initial_geometry = cycles[0]["geometry_angstrom"]

        # Overall convergence
        converged = any(c.get("orca_converged", False) for c in cycles)

        # Compute per-cycle energy changes and RMSD
        for i, cyc in enumerate(cycles):
            energy = cyc.get("energy_Eh")
            prev_energy = cycles[i - 1].get("energy_Eh") if i > 0 else None
            if i > 0 and energy is not None and prev_energy is not None:
                cyc["energy_change_Eh"] = energy - prev_energy

            geom = cyc.get("geometry_angstrom")
            if geom and initial_geometry and len(geom) == len(initial_geometry):
                rmsd = _compute_rmsd(geom, initial_geometry)
                if rmsd is not None:
                    cyc["rmsd_to_initial_ang"] = round(rmsd, 8)
                mw_rmsd = _compute_rmsd(geom, initial_geometry, mass_weighted=True)
                if mw_rmsd is not None:
                    cyc["mass_weighted_rmsd_to_initial_ang"] = round(mw_rmsd, 8)

            if i > 0:
                prev_geom = cycles[i - 1].get("geometry_angstrom")
                if geom and prev_geom and len(geom) == len(prev_geom):
                    rmsd = _compute_rmsd(geom, prev_geom)
                    if rmsd is not None:
                        cyc["rmsd_to_previous_ang"] = round(rmsd, 8)

        # Assemble output
        data["converged"] = converged
        data["n_cycles"] = len(cycles)

        if initial_geometry:
            data["initial_geometry_angstrom"] = initial_geometry
        if final_geometry:
            data["final_geometry_angstrom"] = final_geometry
        if final_energy is not None:
            data["final_energy_Eh"] = final_energy

        # Initial → final RMSD
        if initial_geometry and final_geometry and len(initial_geometry) == len(final_geometry):
            rmsd = _compute_rmsd(initial_geometry, final_geometry)
            if rmsd is not None:
                data["rmsd_initial_to_final_ang"] = round(rmsd, 8)
            mw = _compute_rmsd(initial_geometry, final_geometry, mass_weighted=True)
            if mw is not None:
                data["mass_weighted_rmsd_initial_to_final_ang"] = round(mw, 8)

        data["cycles"] = cycles

        return data

    def _detect_opt_keyword(self, lines: list[str]) -> Optional[str]:
        """Find OPT / LooseOPT / TightOPT from the echoed input block."""
        in_input = False
        bang_lines: list[str] = []
        input_re = re.compile(r"^\|\s*\d+>\s?(.*)$")

        for ln in lines:
            if "INPUT FILE" in ln.upper():
                in_input = True
                continue
            if not in_input:
                continue
            m = input_re.match(ln)
            if m:
                content = m.group(1)
                if content.strip().startswith("!"):
                    bang_lines.append(content)
            if "****END OF INPUT****" in ln:
                break

        # Check for optimization keywords (most specific first)
        for keyword in ("TIGHTOPT", "LOOSEOPT", "OPT"):
            pat = re.compile(rf"(?<![A-Za-z]){keyword}(?![A-Za-z])", re.I)
            for bl in bang_lines:
                if pat.search(bl):
                    return keyword
        return None


PLUGIN_BUNDLE = PluginBundle(
    metadata=PluginMetadata(
        key="geom_opt",
        name="Geometry Optimization",
        short_help="Built-in geometry optimization parser family with cycle-history output hooks.",
        description=(
            "Self-registering built-in geometry optimization module. Owns the "
            "parser section, the opt alias, the normalized geometry "
            "optimization family matcher, and the family-specific markdown/CSV hooks."
        ),
        docs_path="README.md",
        examples=(
            "orca_parser optimization.out --sections opt --markdown --csv",
            "orca_parser optimization.out --detail-scope full",
        ),
    ),
    parser_sections=(
        ParserSectionPlugin("geom_opt", GeomOptModule),
    ),
    parser_aliases=(
        ParserSectionAlias(name="opt", section_keys=("geom_opt",)),
    ),
    calculation_families=(
        CalculationFamilyPlugin(
            family="geometry_optimization",
            default_calculation_label="Geometry Optimization",
            matcher=_matches_geometry_optimization,
            render_markdown_sections=_render_geometry_optimization_markdown_sections,
            csv_writers=(_write_geometry_optimization_csv_sections,),
        ),
    ),
)
