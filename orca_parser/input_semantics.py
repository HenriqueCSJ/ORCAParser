"""Shared normalization for input-level reference and symmetry semantics.

This module exists for one specific maintenance goal: keep "what reference
type is this?" and "did the input explicitly request symmetry?" decisions in
one place instead of letting metadata parsing, parser context building, and
downstream renderers each grow their own slightly different heuristics.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional


_REFERENCE_TOKENS = {"RHF", "UHF", "ROHF", "RKS", "UKS", "ROKS"}
_UNRESTRICTED_REFERENCES = {"UHF", "UKS"}
_KS_REFERENCE_BY_HF_TYPE = {"RHF": "RKS", "UHF": "UKS", "ROHF": "ROKS"}

# Positive markers are safer than treating every unknown method name as a DFT
# functional.  This lets us classify common ORCA DFA names as KS-like while
# avoiding obvious wavefunction and semiempirical families such as MP2 or xTB.
_DFT_MARKERS = (
    "B3LYP",
    "B97",
    "B97M",
    "B97X",
    "BLYP",
    "BP86",
    "CAMB3LYP",
    "DSD",
    "HSE",
    "LC",
    "M05",
    "M06",
    "M11",
    "MN15",
    "OLYP",
    "PBE",
    "PBE0",
    "PBEPP86",
    "PWP",
    "QIDH",
    "R2SCAN",
    "REVPBE",
    "SCAN",
    "TPSS",
    "TPSSh",
    "TPSSH",
    "WB97",
)


def normalize_keyword_token(token: Any) -> str:
    """Return a punctuation-free uppercase token for robust input matching."""

    if token is None:
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(token).upper())


def extract_explicit_reference_type(bang_tokens: Iterable[Any]) -> Optional[str]:
    """Return the last explicit RHF/RKS/UHF/UKS-style token from ``!`` input."""

    explicit: Optional[str] = None
    for token in bang_tokens:
        normalized = normalize_keyword_token(token)
        if normalized in _REFERENCE_TOKENS:
            explicit = normalized
    return explicit


def detect_input_symmetry_request(bang_tokens: Iterable[Any]) -> Optional[bool]:
    """Return explicit input symmetry intent when ``!`` tokens request it."""

    normalized = {
        normalize_keyword_token(token)
        for token in bang_tokens
        if str(token).strip()
    }
    if "USESYM" in normalized:
        return True
    if normalized & {"NOUSESYM", "NOSYM", "NOSYMMETRY"}:
        return False
    return None


def infer_reference_type(
    *,
    bang_tokens: Iterable[Any] = (),
    hf_type: Any = None,
    method: Any = None,
    functional: Any = None,
    reported_functional: Any = None,
) -> str:
    """Infer a normalized reference label such as ``RKS`` or ``UKS``.

    Preference order:
    1. explicit input token (``! UKS`` / ``! RKS`` / etc.)
    2. output ``HFTyp`` if ORCA already printed the final reference label
    3. KS-style upgrade of RHF/UHF/ROHF when the parsed method clearly looks
       like a density functional
    4. raw ``HFTyp`` as the final fallback
    """

    explicit = extract_explicit_reference_type(bang_tokens)
    if explicit:
        return explicit

    normalized_hf_type = normalize_keyword_token(hf_type)
    if normalized_hf_type in {"RKS", "UKS", "ROKS"}:
        return normalized_hf_type

    if normalized_hf_type in _KS_REFERENCE_BY_HF_TYPE and _looks_like_density_functional(
        method=method,
        functional=functional,
        reported_functional=reported_functional,
    ):
        return _KS_REFERENCE_BY_HF_TYPE[normalized_hf_type]

    if normalized_hf_type in _REFERENCE_TOKENS:
        return normalized_hf_type

    return normalized_hf_type


def is_unrestricted_reference(reference_type: Any) -> bool:
    """Return True for unrestricted references while keeping ROHF/ROKS distinct."""

    return normalize_keyword_token(reference_type) in _UNRESTRICTED_REFERENCES


def _looks_like_density_functional(
    *,
    method: Any,
    functional: Any,
    reported_functional: Any,
) -> bool:
    """Heuristic for KS-like methods used only when ORCA prints RHF/UHF.

    ORCA's ``HFTyp`` line carries spin treatment but not always whether the
    reference is HF-like or KS-like.  We therefore look for strong, positive
    DFT markers in the parsed functional/method labels instead of guessing from
    every unknown token.
    """

    candidates = [reported_functional, functional, method]
    for candidate in candidates:
        normalized = normalize_keyword_token(candidate)
        if not normalized:
            continue
        if normalized in _REFERENCE_TOKENS or normalized == "HF":
            continue
        if any(marker in normalized for marker in _DFT_MARKERS):
            return True
    return False
