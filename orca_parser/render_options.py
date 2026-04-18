"""Shared markdown render-policy helpers.

The codebase now has several section-specific display windows:
* GOAT ensemble energy cutoffs
* frontier orbital windows
* compact CMO/NBO character tables
* shortened optimization-cycle histories
* compact E(2) / EPR summaries

Historically each section owned its own default, which made stand-alone and
comparison rendering diverge through ad hoc arguments.  This module provides
one normalized render-policy object so output code can ask:

    "Are we rendering a full stand-alone report or a compact comparison view?"

The intent is not to freeze the list of knobs forever.  It gives us one place
to add future range controls without threading new one-off flags through the
entire output stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


RenderDetailScope = Literal["auto", "full", "compact"]
ResolvedRenderDetailScope = Literal["full", "compact"]

RENDER_OPTION_UNSET = object()


@dataclass(frozen=True)
class RenderOptions:
    """Normalized markdown/report rendering policy.

    ``resolved_detail_scope`` is the only semantic switch most callers should
    care about.  The remaining fields capture the concrete windows that compact
    mode uses today.  ``None`` means "show everything" for the given section.
    """

    requested_detail_scope: RenderDetailScope
    resolved_detail_scope: ResolvedRenderDetailScope
    goat_max_relative_energy_kcal_mol: Optional[float]
    orbital_window: Optional[int]
    cmo_state_limit: Optional[int]
    cmo_frontier_window: Optional[int]
    geom_opt_cycle_preview_count: Optional[int]
    e2_top_interactions: Optional[int]
    epr_top_hyperfine_nuclei: Optional[int]
    epr_top_atom_contributions: Optional[int]

    @property
    def is_full(self) -> bool:
        """Whether the policy requests full, untruncated section rendering."""
        return self.resolved_detail_scope == "full"

    @property
    def is_compact(self) -> bool:
        """Whether the policy requests compact/ranged section rendering."""
        return self.resolved_detail_scope == "compact"


def normalize_render_detail_scope(scope: str) -> RenderDetailScope:
    """Validate and normalize the user-facing render-detail scope."""
    normalized = str(scope or "auto").strip().lower()
    if normalized not in {"auto", "full", "compact"}:
        raise ValueError(
            "Render detail scope must be one of: auto, full, compact."
        )
    return normalized  # type: ignore[return-value]


def build_render_options(
    *,
    comparison: bool,
    detail_scope: str = "auto",
    goat_max_relative_energy_kcal_mol=RENDER_OPTION_UNSET,
) -> RenderOptions:
    """Build the normalized render policy for stand-alone or compare mode.

    ``auto`` is the mode split the user asked for:
    * stand-alone reports default to full
    * comparison documents default to compact
    """

    requested_scope = normalize_render_detail_scope(detail_scope)
    if requested_scope == "auto":
        resolved_scope: ResolvedRenderDetailScope = "compact" if comparison else "full"
    else:
        resolved_scope = requested_scope

    if goat_max_relative_energy_kcal_mol is RENDER_OPTION_UNSET:
        goat_cutoff = 10.0 if resolved_scope == "compact" else None
    else:
        goat_cutoff = goat_max_relative_energy_kcal_mol

    if resolved_scope == "compact":
        return RenderOptions(
            requested_detail_scope=requested_scope,
            resolved_detail_scope=resolved_scope,
            goat_max_relative_energy_kcal_mol=goat_cutoff,
            orbital_window=12,
            cmo_state_limit=10,
            cmo_frontier_window=8,
            geom_opt_cycle_preview_count=6,
            e2_top_interactions=20,
            epr_top_hyperfine_nuclei=15,
            epr_top_atom_contributions=8,
        )

    return RenderOptions(
        requested_detail_scope=requested_scope,
        resolved_detail_scope=resolved_scope,
        goat_max_relative_energy_kcal_mol=goat_cutoff,
        orbital_window=None,
        cmo_state_limit=None,
        cmo_frontier_window=None,
        geom_opt_cycle_preview_count=None,
        e2_top_interactions=None,
        epr_top_hyperfine_nuclei=None,
        epr_top_atom_contributions=None,
    )
