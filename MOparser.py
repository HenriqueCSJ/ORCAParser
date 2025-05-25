# ================================================================
# ORCA Frontier Orbital Analyzer – Enhanced Edition (v5.1 - Gemini Mod)
# 2025-05-25
# ================================================================
"""
Enhanced ORCA orbital analyzer with improved performance, error handling,
and quality of life features including export capabilities, caching,
and interactive options. Plotting for combined spins adjusted for closeness.
"""

import math
import re
import json
import warnings
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union, Any, NamedTuple
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from datetime import datetime
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from IPython.display import display, Markdown, HTML
from tqdm.auto import tqdm

# ================================================================
# CONFIGURATION AND CONSTANTS
# ================================================================

@dataclass
class PlotStyle:
    """Centralized plot styling configuration."""
    occupied_color: str = "royalblue"
    virtual_color: str = "crimson"
    alpha_occupied_color: str = "#4169E1"  # Royal blue variant
    alpha_virtual_color: str = "#DC143C"   # Crimson variant
    beta_occupied_color: str = "#6495ED"   # Cornflower blue
    beta_virtual_color: str = "#FF6347"    # Tomato
    gap_arrow_color: str = "black"
    gap_text_bgcolor: str = "white"
    mu_line_color: str = "slategray"
    mu_text_bgcolor: str = "white"
    spin_gap_arrow_color: str = "dimgrey"
    line_linewidth: float = 2.2
    arrow_linewidth_main: float = 1.6
    arrow_linewidth_spin: float = 1.3
    title_fontsize: int = 14
    axis_label_fontsize: int = 12
    mo_label_fontsize_single: int = 9
    mo_label_fontsize_combined: int = 8
    annotation_fontsize_main: int = 9
    annotation_fontsize_small: int = 8
    figure_dpi: int = 150
    style_name: str = "seaborn-v0_8-whitegrid"

@dataclass
class AnalyzerConfig:
    """Main configuration for the analyzer."""
    hartree_to_ev: float = 27.211_386_245_988
    occupancy_threshold: float = 0.1  # General threshold for considering an orbital "occupied"
    convergence_threshold: float = 1e-9
    max_orbital_display: int = 100
    enable_caching: bool = True
    cache_dir: Path = Path(".orca_cache")
    export_dir: Path = Path("orca_results")
    verbose: bool = True
    show_progress: bool = True
    auto_save_plots: bool = False
    plot_format: str = "png"  # png, pdf, svg
    energy_units: str = "eV"  # eV, hartree, kcal/mol
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        if self.enable_caching:
            self.cache_dir.mkdir(exist_ok=True)
        # Always create export dir to avoid errors when saving manually
        self.export_dir.mkdir(exist_ok=True, parents=True)

# Energy conversion factors relative to eV
ENERGY_CONVERSIONS = {
    "eV": 1.0,
    "hartree": 1/27.211_386_245_988,
    "kcal/mol": 1/23.06054, # 1 eV = 23.06054 kcal/mol
    "kJ/mol": 1/96.4853 # 1 eV = 96.4853 kJ/mol
}

# ================================================================
# DATA STRUCTURES
# ================================================================

@dataclass
class Orbital:
    """Represents a single molecular orbital."""
    number: int
    occupancy: float
    energy_hartree: float
    energy_ev: float
    irrep: str = "N/A"
    spin: Optional[str] = None  # 'alpha', 'beta', or None for restricted
    
    def energy_in_unit(self, unit: str = "eV", config: Optional[AnalyzerConfig] = None) -> float:
        """Get energy in specified unit."""
        base_ev_val = self.energy_ev
        if unit.lower() == "hartree":
            return self.energy_hartree # Use the direct hartree value for precision
        
        conversion_factor = ENERGY_CONVERSIONS.get(unit.lower(), 1.0)
        # For conversion from eV to another unit, multiply by factor
        # For conversion TO eV, the factor is 1.0. Our storage is already eV.
        # The stored ENERGY_CONVERSIONS are factor such that X_eV * factor = Y_unit
        # So, energy_in_eV / conversion_factor_to_eV = energy_in_other_unit
        # This is a bit confusing. Let's define conversion factors FROM eV
        # eV_to_kcal_mol = 23.06054
        # So, E_kcal_mol = E_eV * eV_to_kcal_mol
        
        # Let's redefine ENERGY_CONVERSIONS to be factors to multiply E_eV by.
        # ENERGY_CONVERSIONS_FROM_EV = { "eV": 1.0, "hartree": 1/config.hartree_to_ev if config else 1/27.211386, ...}
        # This is simpler. Using the provided structure:
        # energy_ev is the base. To convert energy_ev to 'unit', we need factor_ev_to_unit
        # The current ENERGY_CONVERSIONS are factor_unit_to_ev (1 unit = X eV)
        # So, energy_in_unit = energy_ev / ENERGY_CONVERSIONS[unit] (if it was 1 unit = X eV)
        # Ah, the dict is `eV_value * dict_value = new_unit_value` -> this is incorrect based on values.
        # `hartree`: 1/27.211.. means `E_eV * (1/27.211) = E_Hartree`. Correct.
        # `kcal/mol`: 23.061 means `E_eV * 23.061 = E_kcal/mol`. Correct.
        return self.energy_ev * ENERGY_CONVERSIONS.get(unit.lower(), 1.0)

    
    @property
    def is_occupied(self) -> bool: # Using a fixed threshold for now as AnalyzerConfig is not easily passed here
        """Check if orbital is occupied based on a general threshold."""
        return self.occupancy > 0.1 # Default general threshold

@dataclass
class OrbitalSet:
    """Container for a set of orbitals with analysis methods."""
    orbitals: List[Orbital]
    spin_type: str  # 'restricted', 'alpha', 'beta'
    config: AnalyzerConfig = field(default_factory=AnalyzerConfig) # Allow passing config
    
    def __post_init__(self):
        """Sort orbitals by energy."""
        self.orbitals.sort(key=lambda x: x.energy_hartree)
    
    @property
    def homo(self) -> Optional[Orbital]:
        """Get the highest occupied molecular orbital."""
        occupied = [orb for orb in self.orbitals if orb.occupancy > self.config.occupancy_threshold]
        return max(occupied, key=lambda x: x.energy_hartree) if occupied else None
    
    @property
    def lumo(self) -> Optional[Orbital]:
        """Get the lowest unoccupied molecular orbital."""
        # Virtual orbitals typically have occupancy < threshold (e.g. < 0.001 or just check against general occ threshold)
        virtual = [orb for orb in self.orbitals if orb.occupancy <= self.config.occupancy_threshold]
        return min(virtual, key=lambda x: x.energy_hartree) if virtual else None
    
    @property
    def gap(self) -> Optional[float]:
        """Calculate HOMO-LUMO gap in eV."""
        if self.homo and self.lumo:
            return self.lumo.energy_ev - self.homo.energy_ev
        return None
    
    def get_frontier_orbitals(self, n: int = 5) -> List[Orbital]:
        """Get n orbitals around HOMO and LUMO."""
        if not self.orbitals: return []
        current_homo = self.homo
        current_lumo = self.lumo

        if not current_homo or not current_lumo:
            # Return middle orbitals if no clear frontier
            mid = len(self.orbitals) // 2
            start = max(0, mid - n)
            end = min(len(self.orbitals), mid + n + 1) # n above, n below, plus center
            return self.orbitals[start:end]
        
        try:
            homo_idx = self.orbitals.index(current_homo)
            lumo_idx = self.orbitals.index(current_lumo)
        except ValueError: # Should not happen if HOMO/LUMO are from self.orbitals
            mid = len(self.orbitals) // 2
            start = max(0, mid - n)
            end = min(len(self.orbitals), mid + n + 1)
            return self.orbitals[start:end]

        start_idx = max(0, homo_idx - n)
        end_idx = min(len(self.orbitals), lumo_idx + n + 1) # n above LUMO, +1 for inclusive slice end
        
        return self.orbitals[start_idx:end_idx]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        data = []
        for orb in self.orbitals:
            data.append({
                'NO': orb.number,
                'Occ': orb.occupancy,
                'E_h': orb.energy_hartree,
                'E_eV': orb.energy_ev,
                'Irrep': orb.irrep,
                'Spin': orb.spin if orb.spin else self.spin_type
            })
        return pd.DataFrame(data)

@dataclass
class ElectronicDescriptors:
    """Container for conceptual DFT descriptors (all in eV)."""
    homo_energy_ev: Optional[float] = None
    lumo_energy_ev: Optional[float] = None
    gap_ev: Optional[float] = None
    ionization_potential_ev: Optional[float] = None # Koopmans: -HOMO
    electron_affinity_ev: Optional[float] = None    # Koopmans: -LUMO
    chemical_potential_ev: Optional[float] = None   # (HOMO+LUMO)/2
    electronegativity_ev: Optional[float] = None    # -chemical_potential
    hardness_ev: Optional[float] = None             # LUMO-HOMO (or (IP-EA)/2)
    softness_ev_inv: Optional[float] = None         # 1/hardness
    
    @classmethod
    def from_frontier_orbitals(cls, homo: Optional[Orbital], lumo: Optional[Orbital], config: AnalyzerConfig) -> 'ElectronicDescriptors':
        """Calculate descriptors from HOMO and LUMO (energies in eV)."""
        if not homo or not lumo:
            return cls()
        
        homo_ev = homo.energy_ev
        lumo_ev = lumo.energy_ev
        
        gap = lumo_ev - homo_ev
        mu = 0.5 * (lumo_ev + homo_ev)
        softness = 1.0 / gap if abs(gap) > config.convergence_threshold else float('inf')
        
        return cls(
            homo_energy_ev=homo_ev,
            lumo_energy_ev=lumo_ev,
            gap_ev=gap,
            ionization_potential_ev=-homo_ev,
            electron_affinity_ev=-lumo_ev,
            chemical_potential_ev=mu,
            electronegativity_ev=-mu,
            hardness_ev=gap, # Pearson hardness often approximated by H-L gap
            softness_ev_inv=softness
        )
    
    def to_dataframe(self, target_unit: str = "eV", config: Optional[AnalyzerConfig] = None) -> pd.DataFrame:
        """Convert to formatted DataFrame, energies in target_unit."""
        conv_factor = ENERGY_CONVERSIONS.get(target_unit.lower(), 1.0)
        
        data = {
            f"ε HOMO ({target_unit})": self.homo_energy_ev * conv_factor if self.homo_energy_ev is not None else None,
            f"ε LUMO ({target_unit})": self.lumo_energy_ev * conv_factor if self.lumo_energy_ev is not None else None,
            f"Gap ({target_unit})": self.gap_ev * conv_factor if self.gap_ev is not None else None,
            f"IP ≈ −εH ({target_unit})": self.ionization_potential_ev * conv_factor if self.ionization_potential_ev is not None else None,
            f"EA ≈ −εL ({target_unit})": self.electron_affinity_ev * conv_factor if self.electron_affinity_ev is not None else None,
            f"μ ({target_unit})": self.chemical_potential_ev * conv_factor if self.chemical_potential_ev is not None else None,
            f"χ = −μ ({target_unit})": self.electronegativity_ev * conv_factor if self.electronegativity_ev is not None else None,
            f"η ({target_unit})": self.hardness_ev * conv_factor if self.hardness_ev is not None else None,
            f"S (1/{target_unit})": self.softness_ev_inv / conv_factor if self.softness_ev_inv is not None and conv_factor != 0 else None # Softness unit is inverse energy
        }
        return pd.DataFrame([data]).T.rename(columns={0: "Value"})

# ================================================================
# PARSER WITH IMPROVED ERROR HANDLING
# ================================================================

class OrcaParser:
    """Robust ORCA output parser with caching and error recovery."""
    
    _PATTERNS = {
        'orbital_line': re.compile(
            r"""^\s*
                (?P<no>\d+)\s+
                (?P<occ>[0-9]*\.[0-9]+(?:[Ee][+\-]?\d+)?)\s+
                (?P<Eh>[+\-]?[0-9]*\.[0-9]+(?:[Ee][+\-]?\d+)?)\s+
                (?P<EeV>[+\-]?[0-9]*\.[0-9]+(?:[Ee][+\-]?\d+)?)
                \s*
                (?P<Irrep>\S+)?
                """,
            re.VERBOSE | re.MULTILINE
        ),
        'orbital_banner': re.compile(
            r"^-+\s*ORBITAL\s+ENERGIES\s*-+", 
            re.MULTILINE | re.IGNORECASE
        ),
        'spin_up': re.compile(
            r"^\s*SPIN\s+UP\s+ORBITALS\s*$",
            re.MULTILINE | re.IGNORECASE
        ),
        'spin_down': re.compile(
            r"^\s*SPIN\s+DOWN\s+ORBITALS\s*$",
            re.MULTILINE | re.IGNORECASE
        ),
        'column_header': re.compile(
            r"^\s*NO\s+OCC\s+E\(Eh\)\s+E\(eV\)(?:\s+Irrep)?\s*$",
            re.MULTILINE
        ),
        'section_end': re.compile(
            r"^-{10,}\s*\n(?:MOLECULAR ORBITALS|TIMINGS FOR THIS JOB|[A-Z\s]+CALCULATION TIME|Total SCF time)", # Added "Total SCF time"
            re.MULTILINE | re.IGNORECASE
        )
    }
    
    def __init__(self, config: AnalyzerConfig):
        self.config = config
    
    @lru_cache(maxsize=32)
    def _get_file_hash(self, file_path: Path) -> str:
        import hashlib
        return hashlib.md5(file_path.read_bytes()).hexdigest()
    
    def parse(self, source: Union[str, Path], force_reparse: bool = False) -> Dict[str, Optional[OrbitalSet]]:
        file_path_obj: Optional[Path] = None
        content: str
        source_name: str

        if isinstance(source, Path) or (isinstance(source, str) and Path(source).is_file()):
            file_path_obj = Path(source)
            source_name = file_path_obj.name
            if self.config.enable_caching and not force_reparse:
                cache_file = self.config.cache_dir / f"{file_path_obj.stem}_{self._get_file_hash(file_path_obj)}.pkl"
                if cache_file.exists():
                    try:
                        with open(cache_file, 'rb') as f:
                            cached_data = pickle.load(f)
                        # Could add a version check for the cached data structure itself
                        if self.config.verbose: print(f"📂 Loading '{source_name}' from cache: {cache_file}")
                        return cached_data
                    except Exception as e:
                        warnings.warn(f"Cache read failed for {cache_file}: {e}. Reparsing.")
            
            content = file_path_obj.read_text(encoding='utf-8', errors='ignore')
        elif isinstance(source, str):
            content = source
            source_name = "input_text_string"
        else:
            raise TypeError("Source must be a file path (str or Path) or a raw text string.")

        if self.config.verbose: print(f"🔍 Parsing '{source_name}'...")
        
        try:
            result = self._parse_content(content)
            
            if file_path_obj and self.config.enable_caching:
                # Cache uses file hash in name to avoid stale data if content changes but stem is same
                cache_file = self.config.cache_dir / f"{file_path_obj.stem}_{self._get_file_hash(file_path_obj)}.pkl"
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(result, f)
                    if self.config.verbose: print(f"💾 Cached '{source_name}' results to: {cache_file}")
                except Exception as e:
                    warnings.warn(f"Cache write failed for {cache_file}: {e}")
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to parse ORCA output from '{source_name}': {str(e)}") from e

    def _parse_content(self, content: str) -> Dict[str, Optional[OrbitalSet]]:
        results = {"restricted": None, "alpha": None, "beta": None}
        banner_matches = list(self._PATTERNS['orbital_banner'].finditer(content))
        if not banner_matches: raise ValueError("No 'ORBITAL ENERGIES' section found.")
        
        last_banner = banner_matches[-1]
        section_text = content[last_banner.end():]
        
        end_match = self._PATTERNS['section_end'].search(section_text)
        if end_match: section_text = section_text[:end_match.start()]
        
        spin_up_match = self._PATTERNS['spin_up'].search(section_text)
        spin_down_match = self._PATTERNS['spin_down'].search(section_text)
        
        if spin_up_match and spin_down_match:
            alpha_text = section_text[spin_up_match.end():spin_down_match.start()]
            beta_text = section_text[spin_down_match.end():]
            results['alpha'] = self._parse_orbital_block(alpha_text, 'alpha')
            results['beta'] = self._parse_orbital_block(beta_text, 'beta')
        elif spin_up_match: # ROHF typically
            alpha_text = section_text[spin_up_match.end():]
            results['alpha'] = self._parse_orbital_block(alpha_text, 'alpha')
            if self.config.verbose: warnings.warn("Found only SPIN UP orbitals. Treating as ROHF-like alpha channel.")
        else:
            results['restricted'] = self._parse_orbital_block(section_text, 'restricted')
        
        if not any(v for v in results.values() if v and v.orbitals):
            raise ValueError("No orbital data lines could be parsed from any identified section.")
        return results
    
    def _parse_orbital_block(self, text: str, spin_type: str) -> Optional[OrbitalSet]:
        header_match = self._PATTERNS['column_header'].search(text)
        if not header_match:
            warnings.warn(f"No column header found in {spin_type} orbital block. Skipping.")
            return None
        
        data_text = text[header_match.end():]
        orbitals = []
        for line_content in data_text.splitlines():
            line = line_content.strip()
            if not line:
                if orbitals: break 
                continue
            
            match = self._PATTERNS['orbital_line'].match(line)
            if match:
                orbitals.append(Orbital(
                    number=int(match['no']),
                    occupancy=float(match['occ']),
                    energy_hartree=float(match['Eh']),
                    energy_ev=float(match['EeV']),
                    irrep=match['Irrep'] or "N/A",
                    spin=spin_type if spin_type != 'restricted' else None
                ))
            elif orbitals: break 
        
        return OrbitalSet(orbitals, spin_type, config=self.config) if orbitals else None

# ================================================================
# ENHANCED PLOTTING WITH EXPORT CAPABILITIES
# ================================================================

class OrbitalPlotter:
    def __init__(self, config: AnalyzerConfig, style: PlotStyle):
        self.config = config
        self.style = style
        self._setup_plotting()
    
    def _setup_plotting(self):
        try:
            plt.style.use(self.style.style_name)
        except OSError:
            warnings.warn(f"Matplotlib style '{self.style.style_name}' not found. Using default.")
            plt.style.use('default') # Fallback
        plt.rcParams['figure.dpi'] = self.style.figure_dpi
        plt.rcParams['savefig.dpi'] = self.style.figure_dpi
        plt.rcParams['font.size'] = 10 # A general base, specific ones set later

    def plot_single_spin(
        self, 
        orbital_set: OrbitalSet,
        n_around: int = 7,
        degeneracy_threshold_ev: float = 0.1, # Renamed for clarity
        save_path: Optional[Path] = None,
        interactive: bool = False # Placeholder for future
    ) -> plt.Figure:
        
        frontier_orbs = orbital_set.get_frontier_orbitals(n_around)
        if not frontier_orbs:
            fig, ax = plt.subplots(figsize=(8,3)); ax.text(0.5,0.5,"No orbitals to display.",ha='center',va='center'); ax.axis('off'); return fig
        
        groups = self._group_degenerate_orbitals(frontier_orbs, degeneracy_threshold_ev)
        fig, ax = plt.subplots(figsize=(10, 12)) # Keep it fairly large for clarity
        
        x_center, level_width, text_offset = 0.5, 0.6, 0.08
        
        energies_for_plot = [orb.energy_in_unit(self.config.energy_units) for orb in frontier_orbs]
        y_min, y_max = min(energies_for_plot) - 1, max(energies_for_plot) + 1
        
        for group in groups:
            self._plot_orbital_group(ax, group, x_center, level_width, text_offset, label_side='both') # 'both' for single plots
        
        descriptors = ElectronicDescriptors.from_frontier_orbitals(orbital_set.homo, orbital_set.lumo, self.config)
        if orbital_set.homo and orbital_set.lumo and descriptors.gap_ev is not None:
            self._add_gap_annotation(ax, orbital_set.homo.energy_in_unit(self.config.energy_units), 
                                     orbital_set.lumo.energy_in_unit(self.config.energy_units), 
                                     descriptors.gap_ev * ENERGY_CONVERSIONS.get(self.config.energy_units.lower(), 1.0), 
                                     x_center)
        
        if descriptors.chemical_potential_ev is not None:
            mu_plot_unit = descriptors.chemical_potential_ev * ENERGY_CONVERSIONS.get(self.config.energy_units.lower(), 1.0)
            self._add_chemical_potential_line(ax, mu_plot_unit, x_center, level_width + 0.1) # slightly wider line

        ax.set_xlim(0, 1) # Simpler xlim for single plot centered at 0.5
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel(f"Energy ({self.config.energy_units})", fontsize=self.style.axis_label_fontsize, weight='bold')
        ax.set_title(
            f"{orbital_set.spin_type.capitalize()} MO Energies (Degeneracy ≤ {degeneracy_threshold_ev:.2f} eV)",
            fontsize=self.style.title_fontsize, weight='bold', pad=15
        )
        ax.set_xticks([])
        for spine in ['top', 'right', 'bottom']: ax.spines[spine].set_visible(False)
        ax.spines['left'].set_linewidth(1.3)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=15, prune='both'))
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        
        plt.tight_layout()
        if save_path or (self.config.auto_save_plots and not interactive): # Avoid saving if just for interactive display
            self._save_figure(fig, save_path or self._generate_filename(f"{orbital_set.spin_type}_orbitals"))
        return fig

    def plot_combined_spins(
        self,
        alpha_set: Optional[OrbitalSet],
        beta_set: Optional[OrbitalSet],
        n_around: int = 7,
        degeneracy_threshold_ev: float = 0.1,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 12)) # Adjusted from 12,14 to 10,12 like single plots
        
        # MODIFIED: Parameters for closer plots
        x_alpha, x_beta = 0.30, 0.70  # Centers for alpha and beta plots (axes fraction)
        level_width = 0.35             # Width of each spin's orbital line block (axes fraction)
        text_offset_combined = 0.05    # Text offset for combined plot labels

        all_plot_energies = []
        
        alpha_groups, beta_groups = None, None
        if alpha_set and alpha_set.orbitals:
            alpha_frontier = alpha_set.get_frontier_orbitals(n_around)
            alpha_groups = self._group_degenerate_orbitals(alpha_frontier, degeneracy_threshold_ev)
            all_plot_energies.extend([orb.energy_in_unit(self.config.energy_units) for orb in alpha_frontier])
        if beta_set and beta_set.orbitals:
            beta_frontier = beta_set.get_frontier_orbitals(n_around)
            beta_groups = self._group_degenerate_orbitals(beta_frontier, degeneracy_threshold_ev)
            all_plot_energies.extend([orb.energy_in_unit(self.config.energy_units) for orb in beta_frontier])

        if alpha_groups:
            for group in alpha_groups:
                self._plot_orbital_group(
                    ax, group, x_alpha, level_width, text_offset_combined,
                    color_occupied=self.style.alpha_occupied_color,
                    color_virtual=self.style.alpha_virtual_color,
                    label_side='left' # MODIFIED: MO labels on left for alpha
                )
            if alpha_set and alpha_set.homo and alpha_set.lumo and alpha_set.gap is not None:
                 gap_alpha_plot_unit = alpha_set.gap * ENERGY_CONVERSIONS.get(self.config.energy_units.lower(), 1.0)
                 self._add_gap_annotation(ax, alpha_set.homo.energy_in_unit(self.config.energy_units),
                                          alpha_set.lumo.energy_in_unit(self.config.energy_units),
                                          gap_alpha_plot_unit, x_alpha, label_prefix='α-')
        
        if beta_groups:
            for group in beta_groups:
                self._plot_orbital_group(
                    ax, group, x_beta, level_width, text_offset_combined,
                    color_occupied=self.style.beta_occupied_color,
                    color_virtual=self.style.beta_virtual_color,
                    label_side='right' # MODIFIED: MO labels on right for beta
                )
            if beta_set and beta_set.homo and beta_set.lumo and beta_set.gap is not None:
                gap_beta_plot_unit = beta_set.gap * ENERGY_CONVERSIONS.get(self.config.energy_units.lower(), 1.0)
                self._add_gap_annotation(ax, beta_set.homo.energy_in_unit(self.config.energy_units),
                                         beta_set.lumo.energy_in_unit(self.config.energy_units),
                                         gap_beta_plot_unit, x_beta, label_prefix='β-')

        if all_plot_energies:
            y_min, y_max = min(all_plot_energies) - 1, max(all_plot_energies) + 1
            ax.set_ylim(y_min, y_max)
        else:
            ax.set_ylim(-10,10); ax.text(0.5,0.5,"No alpha/beta orbitals found to plot.",ha='center',va='center',transform=ax.transAxes,color='red')

        ax.text(x_alpha, 1.015, "Alpha Spin (α)", ha='center', va='bottom', fontsize=self.style.axis_label_fontsize-1, weight='bold', transform=ax.transAxes)
        ax.text(x_beta, 1.015, "Beta Spin (β)", ha='center', va='bottom', fontsize=self.style.axis_label_fontsize-1, weight='bold', transform=ax.transAxes)
        
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_ylabel(f"Energy ({self.config.energy_units})", fontsize=self.style.axis_label_fontsize, weight='bold')
        ax.set_title(
            f"Combined Alpha & Beta MO Energies (Degeneracy ≤ {degeneracy_threshold_ev:.2f} eV)",
            fontsize=self.style.title_fontsize, weight='bold', pad=30 # Increased pad for spin labels
        )
        for spine in ['top', 'right', 'bottom']: ax.spines[spine].set_visible(False)
        ax.spines['left'].set_linewidth(1.3)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=18, prune='both'))
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust for title/spin labels
        if save_path or self.config.auto_save_plots:
            self._save_figure(fig, save_path or self._generate_filename("combined_orbitals"))
        return fig

    def _group_degenerate_orbitals(
        self, 
        orbitals: List[Orbital], 
        threshold_ev: float
    ) -> List[Dict[str, Any]]:
        if not orbitals: return []
        # Ensure orbitals are sorted by energy for grouping
        sorted_orbitals = sorted(orbitals, key=lambda x: x.energy_ev)

        if threshold_ev <= 0: # No grouping
            return [{"orbitals": [orb], 
                     "mean_energy_ev": orb.energy_ev,
                     "is_occupied": orb.occupancy > self.config.occupancy_threshold, # Use config threshold
                     "irreps": [orb.irrep] if orb.irrep != "N/A" else []
                    } for orb in sorted_orbitals]

        groups, current_group_orbs = [], []
        for orb in sorted_orbitals:
            is_orb_occupied = orb.occupancy > self.config.occupancy_threshold
            if not current_group_orbs or \
               (abs(orb.energy_ev - np.mean([o.energy_ev for o in current_group_orbs])) <= threshold_ev and
                is_orb_occupied == (current_group_orbs[0].occupancy > self.config.occupancy_threshold)):
                current_group_orbs.append(orb)
            else:
                groups.append({
                    "orbitals": current_group_orbs,
                    "mean_energy_ev": np.mean([o.energy_ev for o in current_group_orbs]),
                    "is_occupied": current_group_orbs[0].occupancy > self.config.occupancy_threshold,
                    "irreps": sorted(list(set(o.irrep for o in current_group_orbs if o.irrep != "N/A")))
                })
                current_group_orbs = [orb]
        
        if current_group_orbs: # Add last group
             groups.append({
                "orbitals": current_group_orbs,
                "mean_energy_ev": np.mean([o.energy_ev for o in current_group_orbs]),
                "is_occupied": current_group_orbs[0].occupancy > self.config.occupancy_threshold,
                "irreps": sorted(list(set(o.irrep for o in current_group_orbs if o.irrep != "N/A")))
            })
        return groups

    def _plot_orbital_group(
        self, 
        ax: plt.Axes, 
        group: Dict[str, Any],
        x_center: float,
        width: float,      # Width of the lines area for this group
        text_offset: float, # Offset for text from the edge of lines area
        color_occupied: Optional[str] = None,
        color_virtual: Optional[str] = None,
        label_side: str = 'both' # MODIFIED: 'left', 'right', or 'both'
    ):
        if not group.get("orbitals"):
            if self.config.verbose: print("⚠️ Warning: Attempting to plot empty orbital group.")
            return
            
        orbitals_in_group = group["orbitals"]
        # Energy for plotting should be in target units
        y_pos_plot_unit = group["mean_energy_ev"] * ENERGY_CONVERSIONS.get(self.config.energy_units.lower(), 1.0)
        is_group_occupied = group["is_occupied"]
        
        color = (color_occupied or self.style.occupied_color) if is_group_occupied else \
                (color_virtual or self.style.virtual_color)
        
        n_lines = min(len(orbitals_in_group), 4) 
        line_area_start_x = x_center - width / 2
        line_area_end_x = x_center + width / 2

        if n_lines == 1:
            ax.hlines(y_pos_plot_unit, line_area_start_x, line_area_end_x, 
                      color=color, lw=self.style.line_linewidth)
        else:
            # Draw multiple short segments within the allocated width
            segment_total_width = width * 0.8 # Use 80% of width for segments to leave small gaps
            gap_between_segments = segment_total_width * 0.1 / (n_lines -1 ) if n_lines > 1 else 0
            single_segment_width = (segment_total_width - gap_between_segments*(n_lines-1)) / n_lines
            
            current_x = x_center - segment_total_width/2
            for i in range(n_lines):
                ax.hlines(y_pos_plot_unit, current_x, current_x + single_segment_width,
                          color=color, lw=self.style.line_linewidth)
                current_x += single_segment_width + gap_between_segments
        
        # --- MODIFIED TEXT LABELING LOGIC ---
        # MO info string
        numbers = sorted([orb.number for orb in orbitals_in_group])
        label_mo_info = f"MO {numbers[0]}" if len(numbers) == 1 else f"MO {numbers[0]}-{numbers[-1]}"
        if group.get("irreps"): label_mo_info += f" ({'/'.join(group['irreps'])})"
        if len(orbitals_in_group) > 1: label_mo_info += f" (×{len(orbitals_in_group)})"
        
        # Energy value string (always prepared, used by 'both')
        energy_val_str = f"{group['mean_energy_ev']:.2f}" # Use 2 decimal places for eV in plot
        if self.config.energy_units.lower() != "ev":
             energy_val_str = f"{group['mean_energy_ev'] * ENERGY_CONVERSIONS.get(self.config.energy_units.lower(), 1.0):.2f}"

        energy_val_str_display = f"{'≈ ' if len(orbitals_in_group) > 1 else ''}{energy_val_str} {self.config.energy_units}"

        # Determine fontsize based on plot type context (single vs combined)
        current_fontsize = self.style.mo_label_fontsize_single if label_side == 'both' \
                           else self.style.mo_label_fontsize_combined

        if label_side == 'left': # Typically for Alpha in combined plot
            ax.text(line_area_start_x - text_offset, y_pos_plot_unit, label_mo_info,
                    ha='right', va='center', color=color, fontsize=current_fontsize)
        elif label_side == 'right': # Typically for Beta in combined plot
            ax.text(line_area_end_x + text_offset, y_pos_plot_unit, label_mo_info,
                    ha='left', va='center', color=color, fontsize=current_fontsize)
        elif label_side == 'both': # Typically for single spin plots
            ax.text(line_area_start_x - text_offset, y_pos_plot_unit, label_mo_info,
                    ha='right', va='center', color=color, fontsize=current_fontsize)
            ax.text(line_area_end_x + text_offset, y_pos_plot_unit, energy_val_str_display,
                    ha='left', va='center', color=color, fontsize=current_fontsize)

    def _add_gap_annotation(
        self, ax: plt.Axes, homo_energy_plot_unit: float, lumo_energy_plot_unit: float, 
        gap_plot_unit: float, x_center: float, label_prefix: str = ""
    ):
        ax.annotate("", xy=(x_center, homo_energy_plot_unit), xycoords='data',
                    xytext=(x_center, lumo_energy_plot_unit), textcoords='data',
                    arrowprops=dict(arrowstyle="<->", lw=self.style.arrow_linewidth_main,
                                    color=self.style.gap_arrow_color, shrinkA=0, shrinkB=0))
        ax.text(x_center, (homo_energy_plot_unit + lumo_energy_plot_unit) / 2,
                f" {label_prefix}Gap = {gap_plot_unit:.2f} {self.config.energy_units} ", # Use 2 decimal places
                ha='center', va='center', weight='bold', fontsize=self.style.annotation_fontsize_main,
                bbox=dict(boxstyle="round,pad=0.25", fc=self.style.gap_text_bgcolor, alpha=0.9))
    
    def _add_chemical_potential_line(self, ax: plt.Axes, mu_plot_unit: float, x_center: float, line_total_width: float):
        # line_total_width is the span the dashed line should cover
        xmin_frac = (x_center - line_total_width / 2) / (ax.get_xlim()[1] - ax.get_xlim()[0]) # Normalize based on current xlim
        xmax_frac = (x_center + line_total_width / 2) / (ax.get_xlim()[1] - ax.get_xlim()[0])
        
        # Ensure fractions are within [0,1] if plot limits are tight
        plot_xmin, plot_xmax = ax.get_xlim()
        line_start_abs = x_center - line_total_width / 2
        line_end_abs = x_center + line_total_width / 2

        ax.axhline(mu_plot_unit, color=self.style.mu_line_color, ls=':', 
                   lw=self.style.arrow_linewidth_main - 0.3,
                   xmin= (line_start_abs - plot_xmin) / (plot_xmax - plot_xmin) if (plot_xmax - plot_xmin) !=0 else 0.05,
                   xmax= (line_end_abs - plot_xmin) / (plot_xmax - plot_xmin) if (plot_xmax - plot_xmin) !=0 else 0.95
                  )
        
        ax.text(x_center + line_total_width / 2 + 0.02, mu_plot_unit, # Small offset for text from line end
                f"μ = {mu_plot_unit:.2f} {self.config.energy_units}", # Use 2 decimal places
                ha='left', va='center', fontsize=self.style.annotation_fontsize_small,
                bbox=dict(boxstyle='round,pad=0.15', fc=self.style.mu_text_bgcolor, alpha=0.8))
    
    def _save_figure(self, fig: plt.Figure, path: Union[str,Path]):
        path_obj = Path(path)
        path_obj.parent.mkdir(exist_ok=True, parents=True)
        
        final_path_str = str(path_obj)
        if not path_obj.suffix or path_obj.suffix.lower() not in [".png", ".pdf", ".svg", ".jpg", ".jpeg"]:
             final_path_str = str(path_obj.with_suffix(f".{self.config.plot_format.lower()}"))

        fig.savefig(final_path_str, dpi=self.style.figure_dpi, bbox_inches='tight')
        if self.config.verbose: print(f"💾 Plot saved to: {final_path_str}")
    
    def _generate_filename(self, plot_type: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.config.export_dir / f"orbitals_{plot_type}_{timestamp}" # Suffix added by _save_figure

    def _add_interactivity(self, fig: plt.Figure, ax: plt.Axes, orbitals: List[Orbital]):
        # Placeholder for future matplotlib interactive features
        if self.config.verbose: print("ℹ️ Interactive mode placeholder.")
        pass

# ================================================================
# ANALYSIS ENGINE WITH EXPORT CAPABILITIES
# ================================================================

class OrcaAnalyzer:
    def __init__(self, config: Optional[AnalyzerConfig] = None, style: Optional[PlotStyle] = None):
        self.config = config or AnalyzerConfig()
        self.style = style or PlotStyle()
        self.parser = OrcaParser(self.config)
        self.plotter = OrbitalPlotter(self.config, self.style)
        self.results: Dict[str, Any] = {} # To store results of the last analysis
    
    def analyze(
        self,
        source: Union[str, Path],
        orbitals_around: int = 7,
        degeneracy_threshold_ev: float = 0.1,
        export_data_formats: Optional[List[str]] = None, # e.g. ["json", "excel"]
        export_plot_formats: Optional[List[str]] = None, # e.g. ["png", "pdf"]
        interactive: bool = False # For plots
    ) -> Dict[str, Any]:
        
        # Determine effective export settings
        _export_data = bool(export_data_formats)
        _export_plots = bool(export_plot_formats) or self.config.auto_save_plots

        try:
            orbital_sets = self.parser.parse(source)
            results: Dict[str, Any] = {
                "source_name": Path(source).name if isinstance(source, Path) or (isinstance(source, str) and Path(source).is_file()) else "text_input",
                "timestamp": datetime.now().isoformat(),
                "orbital_data_raw": {spin: (oset.orbitals if oset else []) for spin, oset in orbital_sets.items()}, # For inspection
                "descriptors": {}, "plots": {}, "summary": {}
            }
            
            # Process restricted or alpha/beta individually
            for spin_type in ["restricted", "alpha", "beta"]:
                orbital_set = orbital_sets.get(spin_type)
                if orbital_set and orbital_set.orbitals: # Ensure there are orbitals
                    self._analyze_spin_channel(
                        results, spin_type, orbital_set,
                        orbitals_around, degeneracy_threshold_ev,
                        _export_plots, interactive # Pass effective export plot flag
                    )
            
            # Combined analysis for unrestricted calculations
            alpha_set, beta_set = orbital_sets.get("alpha"), orbital_sets.get("beta")
            if alpha_set and alpha_set.orbitals and beta_set and beta_set.orbitals:
                self._analyze_unrestricted(
                    results, alpha_set, beta_set,
                    orbitals_around, degeneracy_threshold_ev, _export_plots # Pass effective export plot flag
                )
            
            self._generate_summary(results, orbital_sets)
            
            if _export_data and export_data_formats:
                self._export_results(results, orbital_sets, export_data_formats)
            
            self.results = results
            return results
            
        except Exception as e:
            error_msg = f"Analysis failed for '{str(source)}': {type(e).__name__} - {str(e)}"
            if self.config.verbose:
                import traceback
                print(f"\n❌ {error_msg}")
                print("\nTraceback (most recent call last):"); traceback.print_exc()
            return {"error": error_msg, "source": str(source), "timestamp": datetime.now().isoformat()}

    def _analyze_spin_channel(
        self, results_dict: Dict, spin_type: str, orbital_set: OrbitalSet,
        n_around: int, degen_thresh: float, export_plt: bool, interactive_plt: bool
    ):
        if self.config.verbose: print(f"\n📊 Analyzing {spin_type} orbitals...")
        
        desc = ElectronicDescriptors.from_frontier_orbitals(orbital_set.homo, orbital_set.lumo, self.config)
        results_dict["descriptors"][spin_type] = desc
        
        if self.config.verbose: self._display_spin_results(spin_type, orbital_set, desc)
        
        save_path_obj = None
        if export_plt:
             save_path_obj = self.plotter._generate_filename(f"{results_dict['source_name']}_{spin_type}")

        fig = self.plotter.plot_single_spin(
            orbital_set, n_around, degen_thresh,
            save_path=save_path_obj, interactive=interactive_plt
        )
        results_dict["plots"][spin_type] = fig if not interactive_plt else "Interactive plot shown"
        if not interactive_plt: plt.show() # Show non-interactive plots

    def _analyze_unrestricted(
        self, results_dict: Dict, alpha_s: OrbitalSet, beta_s: OrbitalSet,
        n_around: int, degen_thresh: float, export_plt: bool
    ):
        if self.config.verbose: print("\n🔄 Analyzing combined alpha/beta system...")
        
        # Determine overall HOMO and LUMO from combined set
        possible_homos = [orb for orb_set in (alpha_s, beta_s) if orb_set.homo for orb in [orb_set.homo]]
        possible_lumos = [orb for orb_set in (alpha_s, beta_s) if orb_set.lumo for orb in [orb_set.lumo]]
        
        overall_homo = max(possible_homos, key=lambda o: o.energy_ev) if possible_homos else None
        overall_lumo = min(possible_lumos, key=lambda o: o.energy_ev) if possible_lumos else None
        
        overall_desc = ElectronicDescriptors.from_frontier_orbitals(overall_homo, overall_lumo, self.config)
        results_dict["descriptors"]["overall"] = overall_desc
        
        if self.config.verbose and overall_homo and overall_lumo and overall_desc.gap_ev is not None:
            print(f"  Overall HOMO ({overall_homo.spin if overall_homo else 'N/A'}): {overall_homo.energy_ev:.3f} eV")
            print(f"  Overall LUMO ({overall_lumo.spin if overall_lumo else 'N/A'}): {overall_lumo.energy_ev:.3f} eV")
            print(f"  Overall Gap: {overall_desc.gap_ev:.3f} eV")

        save_path_obj = None
        if export_plt:
            save_path_obj = self.plotter._generate_filename(f"{results_dict['source_name']}_combined")

        fig = self.plotter.plot_combined_spins(
            alpha_s, beta_s, n_around, degen_thresh, save_path=save_path_obj
        )
        results_dict["plots"]["combined"] = fig
        plt.show()

    def _display_spin_results(self, spin_t: str, orb_set: OrbitalSet, desc: ElectronicDescriptors):
        print(f"\n--- {spin_t.capitalize()} Frontier Orbitals (Energies in {self.config.energy_units}) ---")
        unit = self.config.energy_units
        if orb_set.homo: h = orb_set.homo; print(f"HOMO: MO {h.number} ({h.irrep}) {h.energy_in_unit(unit):.3f} {unit} (Occ: {h.occupancy:.3f})")
        else: print("HOMO: Not found")
        if orb_set.lumo: l = orb_set.lumo; print(f"LUMO: MO {l.number} ({l.irrep}) {l.energy_in_unit(unit):.3f} {unit} (Occ: {l.occupancy:.3f})")
        else: print("LUMO: Not found")
        
        if desc.homo_energy_ev is not None:
            display(HTML(desc.to_dataframe(target_unit=unit)._repr_html_())) # Use HTML for better display

    def _generate_summary(self, results_dict: Dict, orbital_sets: Dict[str, Optional[OrbitalSet]]):
        summary = {
            "calculation_type": "restricted" if orbital_sets.get("restricted") and orbital_sets["restricted"].orbitals else "unrestricted",
            "spin_channels_found": [k for k, v in orbital_sets.items() if v and v.orbitals],
            "gaps_ev": {}
        }
        for spin, desc in results_dict["descriptors"].items():
            if hasattr(desc, 'gap_ev') and desc.gap_ev is not None: summary["gaps_ev"][spin] = desc.gap_ev
        
        if summary["gaps_ev"]:
            min_gap_spin = min(summary["gaps_ev"], key=summary["gaps_ev"].get)
            summary["smallest_gap_ev"] = {"value": summary["gaps_ev"][min_gap_spin], "spin": min_gap_spin}
        results_dict["summary"] = summary

    def _export_results(self, results: Dict, orbital_sets: Dict[str, Optional[OrbitalSet]], formats: List[str]):
        base_name = self.plotter._generate_filename(results['source_name']).stem # Get timestamped base
        
        if "json" in formats:
            json_path = self.config.export_dir / f"{base_name}.json"
            serializable_results = {
                "summary": results["summary"],
                "descriptors": {k: asdict(v) if v else None for k,v in results["descriptors"].items()},
                "orbital_dataframes": {
                    k: oset.to_dataframe().to_dict(orient='records') if oset and oset.orbitals else None 
                    for k,oset in orbital_sets.items()
                },
                "source_name": results["source_name"], "timestamp": results["timestamp"]
            }
            with open(json_path, 'w') as f: json.dump(serializable_results, f, indent=2, default=str)
            if self.config.verbose: print(f"💾 JSON data exported to: {json_path}")

        if "excel" in formats:
            excel_path = self.config.export_dir / f"{base_name}.xlsx"
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                pd.DataFrame([results["summary"]]).to_excel(writer, sheet_name='Summary', index=False)
                for spin, oset in orbital_sets.items():
                    if oset and oset.orbitals: 
                        oset.to_dataframe().to_excel(writer, sheet_name=f'{spin.capitalize()} Orbitals', index=False)
                
                desc_list = []
                for spin, desc_obj in results["descriptors"].items():
                    if desc_obj:
                        df_desc = desc_obj.to_dataframe(target_unit="eV") # Export canonical eV
                        df_desc.index.name = "Descriptor"
                        df_desc.reset_index(inplace=True)
                        df_desc["Spin_Channel"] = spin
                        desc_list.append(df_desc)
                if desc_list:
                     pd.concat(desc_list).set_index(["Spin_Channel", "Descriptor"]).to_excel(writer, sheet_name='Descriptors')
            if self.config.verbose: print(f"💾 Excel data exported to: {excel_path}")
    
    def compare_files(self, file_paths: List[Union[str,Path]], metric: str = "gap_ev") -> pd.DataFrame:
        # Simplified comparison focusing on a few key metrics
        comp_data = []
        prog_bar = tqdm(file_paths, desc="Comparing files", disable=not self.config.show_progress)
        for f_path in prog_bar:
            prog_bar.set_postfix_str(Path(f_path).name)
            res = self.analyze(f_path, export_data_formats=None, export_plot_formats=None) # No exports during compare
            if "error" in res:
                comp_data.append({"file":Path(f_path).name, "error": res["error"]}); continue
            
            row = {"file": Path(f_path).name, "type": res["summary"]["calculation_type"]}
            for spin, desc in res["descriptors"].items():
                if hasattr(desc, metric): row[f"{spin}_{metric}"] = getattr(desc, metric)
            comp_data.append(row)
        return pd.DataFrame(comp_data)

# ================================================================
# CONVENIENCE FUNCTIONS & EXAMPLE USAGE
# ================================================================

def quick_analyze(
    file_or_text: Union[str, Path],
    orbitals_around_frontier: int = 7,
    degeneracy_eV: float = 0.1,
    save_plots_as: Optional[List[str]] = None, # e.g., ["png", "pdf"]
    save_data_as: Optional[List[str]] = None, # e.g., ["json"]
    verbose_output: bool = True
) -> Dict[str, Any]:
    
    cfg = AnalyzerConfig(verbose=verbose_output, auto_save_plots=bool(save_plots_as))
    analyzer_instance = OrcaAnalyzer(config=cfg)
    return analyzer_instance.analyze(
        source=file_or_text,
        orbitals_around=orbitals_around_frontier,
        degeneracy_threshold_ev=degeneracy_eV,
        export_plot_formats=save_plots_as,
        export_data_formats=save_data_as
    )

def test_plot_generation(analyzer: OrcaAnalyzer, spin_type: str):
    """Helper to generate a plot for testing specific spin type."""
    if spin_type == "combined":
        alpha_orbs = [Orbital(n,1.0 if n < 3 else 0.0, -0.5+0.05*n, (-0.5+0.05*n)*27.2, "A", "alpha") for n in range(1,6)]
        beta_orbs = [Orbital(n,1.0 if n < 2 else 0.0, -0.55+0.06*n, (-0.55+0.06*n)*27.2, "A", "beta") for n in range(1,6)]
        alpha_set = OrbitalSet(alpha_orbs, "alpha", config=analyzer.config)
        beta_set = OrbitalSet(beta_orbs, "beta", config=analyzer.config)
        analyzer.plotter.plot_combined_spins(alpha_set, beta_set, n_around=2, degeneracy_threshold_ev=0.05)
    else: # restricted or single alpha/beta
        orbs = [Orbital(n,2.0 if n < 3 and spin_type=="restricted" else (1.0 if n<3 else 0.0), 
                        -0.5+0.05*n, (-0.5+0.05*n)*27.2, "A1g", spin_type) for n in range(1,6)]
        orb_set = OrbitalSet(orbs, spin_type, config=analyzer.config)
        analyzer.plotter.plot_single_spin(orb_set, n_around=2, degeneracy_threshold_ev=0.05)
    plt.suptitle(f"Test Plot: {spin_type.capitalize()}", y=1.03, weight='bold')
    plt.show()

if __name__ == "__main__":
    orca_file_input = "CoC60.out"  # Replace with your .out file name or ""
    orca_text_input = None      # Or paste ORCA output text here if not using a file

    # --- Basic User Configuration ---
    cfg_user = AnalyzerConfig(
        verbose=True,
        enable_caching=True,
        auto_save_plots=False, # Set True to save plots automatically (uses config.plot_format)
        plot_format="png",     # Format for auto-saved plots
        energy_units="eV"      # Units for display in plots and tables
    )
    style_user = PlotStyle(figure_dpi=120) # Slightly lower DPI for faster display if needed

    analyzer = OrcaAnalyzer(config=cfg_user, style=style_user)

    # --- Choose Analysis Type ---
    # Option 1: Analyze a single file
    if orca_file_input or orca_text_input:
        print("🚀 Running single file analysis...")
        analysis_results = analyzer.analyze(
            source=orca_file_input if orca_file_input else orca_text_input,
            orbitals_around=7,
            degeneracy_threshold_ev=0.1,
            export_data_formats=["json", "excel"], # Specify desired data formats for export
            export_plot_formats=["png"]            # Specify desired plot formats for export (if auto_save_plots is False, these still trigger named save)
        )
        if "error" not in analysis_results:
            print("\n" + "="*60 + "\n✅ ANALYSIS COMPLETE\n" + "="*60)
            print(f"Summary for: {analysis_results['source_name']}")
            for k,v in analysis_results["summary"].items(): print(f"  {k}: {v}")
        else:
            print(f"\n❌ ANALYSIS FAILED: {analysis_results['error']}")

    # Option 2: Compare multiple files (example)
    # files_to_compare = ["path/to/file1.out", "path/to/file2.out"]
    # if all(Path(f).exists() for f in files_to_compare):
    #     print("\n🚀 Running file comparison...")
    #     comparison_df = analyzer.compare_files(files_to_compare, metric="gap_ev")
    #     print("\n--- Comparison Results (gap_ev) ---")
    #     display(HTML(comparison_df.to_html(na_rep='N/A')))
    # else:
    #     print("\nℹ️ Skipping file comparison example (files not found).")

    # Option 3: Test plot generation (for developers/testing)
    # print("\n🚀 Generating test plots...")
    # test_plot_generation(analyzer, "restricted")
    # test_plot_generation(analyzer, "alpha") # Example for alpha only (ROHF-like)
    # test_plot_generation(analyzer, "combined")
    # print("✅ Test plots generated.")

    print("\nScript finished.")