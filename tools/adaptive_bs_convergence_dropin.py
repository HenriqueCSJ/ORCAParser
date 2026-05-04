"""Drop-in adaptive SCF protocol for the pasted BSCalculator script.

Usage in the calculation script, after BSCalculator is defined:

    from tools.adaptive_bs_convergence_dropin import install_adaptive_protocol
    install_adaptive_protocol(BSCalculator, C)

Optional, inside the scan loop before running HS/BS:

    calc.set_prior_densities(previous_hs_dm, previous_bs_dm)

Then save new prior densities after each successful structure:

    previous_hs_dm = calc.get_last_density("HIGH SPIN")
    previous_bs_dm = calc.get_last_density("BROKEN SYMMETRY")

This module deliberately monkey-patches only BSCalculator convergence behavior.
It expects the original class to keep its existing helper methods such as
_make_uks, _print_geometry, _analyze_population, _print_energy_components,
generate_bs_guess, _run_stability_refinement, and _persist_results_json.
"""

from __future__ import annotations

import gc
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pyscf import scf


class _PlainC:
    BLUE = ""
    CYAN = ""
    GREEN = ""
    YELLOW = ""
    RED = ""
    WHITE = ""
    BOLD = ""
    HEADER = ""
    UNDERLINE = ""
    END = ""


def _to_cpu_array(x: Any) -> np.ndarray:
    if x is None:
        raise ValueError("Cannot convert None to an array.")
    if hasattr(x, "get"):
        x = x.get()
    elif hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x, dtype=np.float64, order="C")


def _copy_dm(dm: Any) -> Optional[np.ndarray]:
    if dm is None:
        return None
    try:
        return _to_cpu_array(dm).copy()
    except Exception:
        return None


def _safe_dm_from_mf(mf: Any) -> Optional[np.ndarray]:
    try:
        return _copy_dm(mf.make_rdm1())
    except Exception:
        return None


def _sanitize_atom_spin_map(mol: Any, atom_spin_map: Optional[Dict[int, int]]) -> Dict[int, int]:
    clean: Dict[int, int] = {}
    if not atom_spin_map:
        return clean
    for key, value in atom_spin_map.items():
        try:
            idx = int(key)
        except Exception:
            continue
        if 0 <= idx < mol.natm:
            clean[idx] = int(np.sign(value)) if value != 0 else 0
    return clean


def _atom_spin_populations(mol: Any, mf: Any) -> Optional[np.ndarray]:
    try:
        s = _to_cpu_array(mf.get_ovlp())
        dm = _to_cpu_array(mf.make_rdm1())
        if dm.ndim == 3:
            dm_a, dm_b = dm[0], dm[1]
        else:
            dm_a = dm_b = dm / 2.0

        pop_a = np.diag(dm_a @ s)
        pop_b = np.diag(dm_b @ s)
        aoslices = mol.aoslice_by_atom()
        spins = []
        for ia in range(mol.natm):
            start, end = aoslices[ia][2], aoslices[ia][3]
            spins.append(float(np.sum(pop_a[start:end]) - np.sum(pop_b[start:end])))
        return np.asarray(spins)
    except Exception:
        return None


def _local_spin_pattern_ok(
    mol: Any,
    mf: Any,
    atom_spin_map: Dict[int, int],
    threshold: float = 0.02,
) -> bool:
    if not atom_spin_map:
        return True
    spins = _atom_spin_populations(mol, mf)
    if spins is None:
        return False
    for idx, sign in atom_spin_map.items():
        if sign > 0 and spins[idx] < threshold:
            return False
        if sign < 0 and spins[idx] > -threshold:
            return False
    return True


def _spin_square(mf: Any) -> float:
    try:
        return float(mf.spin_square()[0])
    except Exception:
        return float("nan")


def _stage_plan(label: str) -> List[Dict[str, Any]]:
    is_bs = label.upper().startswith("BROKEN")
    return [
        {
            "name": "coarse damping",
            "max_cycle": 60,
            "conv_tol": 1e-5,
            "level_shift": 1.5 if is_bs else 0.8,
            "damp": 0.70 if is_bs else 0.40,
            "diis_space": 8,
            "diis_start_cycle": 8,
            "grid_level": 3,
            "smear_sigma": 0.020,
            "diis_class": "EDIIS",
        },
        {
            "name": "spin-lock rescue",
            "max_cycle": 80,
            "conv_tol": 5e-6,
            "level_shift": 2.5 if is_bs else 1.2,
            "damp": 0.85 if is_bs else 0.55,
            "diis_space": 10,
            "diis_start_cycle": 12,
            "grid_level": 3,
            "smear_sigma": 0.035,
            "diis_class": "ADIIS",
        },
        {
            "name": "balanced DIIS",
            "max_cycle": 120,
            "conv_tol": 1e-6,
            "level_shift": 0.60 if is_bs else 0.35,
            "damp": 0.20 if is_bs else 0.10,
            "diis_space": 12,
            "diis_start_cycle": 3,
            "grid_level": 4,
            "smear_sigma": 0.010,
            "diis_class": "CDIIS",
        },
        {
            "name": "fine polish",
            "max_cycle": 180,
            "conv_tol": 1e-8,
            "level_shift": 0.10,
            "damp": 0.0,
            "diis_space": 14,
            "diis_start_cycle": 1,
            "grid_level": 4,
            "smear_sigma": None,
            "diis_class": "CDIIS",
        },
        {
            "name": "zero-shift finish",
            "max_cycle": 220,
            "conv_tol": 1e-8,
            "level_shift": 0.0,
            "damp": 0.0,
            "diis_space": 16,
            "diis_start_cycle": 1,
            "grid_level": 5,
            "smear_sigma": None,
            "diis_class": "CDIIS",
        },
        {
            "name": "Newton/AH rescue",
            "max_cycle": 80,
            "conv_tol": 1e-8,
            "level_shift": 0.0,
            "damp": 0.0,
            "diis_space": 12,
            "diis_start_cycle": 1,
            "grid_level": 5,
            "smear_sigma": None,
            "diis_class": "CDIIS",
            "newton": True,
        },
    ]


def install_adaptive_protocol(calc_cls: Any, colors: Optional[Any] = None) -> None:
    """Install adaptive convergence methods onto the existing BSCalculator."""

    C = colors or _PlainC

    def set_prior_densities(self: Any, hs_dm: Any = None, bs_dm: Any = None) -> None:
        self._prior_hs_dm = _copy_dm(hs_dm)
        self._prior_bs_dm = _copy_dm(bs_dm)

    def get_last_density(self: Any, label: str) -> Optional[np.ndarray]:
        if label.upper().startswith("HIGH"):
            return _safe_dm_from_mf(getattr(self, "hs_mf", None))
        return _safe_dm_from_mf(getattr(self, "bs_mf", None))

    def _configure_attempt_mf(self: Any, stage: Dict[str, Any]) -> Any:
        mf = self._make_uks(self.mol)
        mf.max_cycle = int(stage["max_cycle"])
        mf.conv_tol = float(stage["conv_tol"])
        mf.level_shift = float(stage["level_shift"])
        mf.damp = float(stage["damp"])
        mf.diis_space = int(stage["diis_space"])
        if hasattr(mf, "diis_start_cycle"):
            mf.diis_start_cycle = int(stage["diis_start_cycle"])
        if getattr(mf, "grids", None) is not None:
            mf.grids.level = int(stage["grid_level"])

        diis_name = stage.get("diis_class")
        if diis_name and hasattr(scf.diis, diis_name):
            mf.DIIS = getattr(scf.diis, diis_name)

        sigma = stage.get("smear_sigma")
        if sigma is not None:
            try:
                mf = scf.addons.smearing_(mf, sigma=float(sigma), method="fermi")
            except Exception:
                pass

        if stage.get("newton"):
            try:
                mf = mf.newton()
                mf.max_cycle = int(stage["max_cycle"])
                mf.conv_tol = float(stage["conv_tol"])
            except Exception:
                pass

        return mf

    def _initial_guess_candidates(
        self: Any,
        label: str,
        atom_spin_map: Dict[int, int],
    ) -> List[Tuple[str, Optional[np.ndarray]]]:
        candidates: List[Tuple[str, Optional[np.ndarray]]] = []
        is_bs = label.upper().startswith("BROKEN")

        if is_bs:
            candidates.append(("prior BS density", getattr(self, "_prior_bs_dm", None)))
            if atom_spin_map:
                try:
                    dm_atom = scf.uhf.init_guess_by_atom(self.mol, atom_spin_map)
                    candidates.append(("atomic spin-map guess", dm_atom))
                except Exception:
                    pass
            try:
                candidates.append(("localized SOMO flip", self.generate_bs_guess(self.mol, atom_spin_map)))
            except Exception:
                pass
            candidates.append(("prior HS density", getattr(self, "_prior_hs_dm", None)))
        else:
            candidates.append(("prior HS density", getattr(self, "_prior_hs_dm", None)))

        for key in ("minao", "atom", "huckel", "1e"):
            try:
                mf0 = self._make_uks(self.mol)
                candidates.append((f"{key} initial guess", mf0.get_init_guess(key=key)))
            except Exception:
                pass

        seen: set = set()
        clean: List[Tuple[str, Optional[np.ndarray]]] = []
        for name, dm in candidates:
            dm_copy = _copy_dm(dm)
            if dm_copy is None:
                continue
            signature = (name, dm_copy.shape)
            if signature in seen:
                continue
            seen.add(signature)
            clean.append((name, dm_copy))
        return clean

    def _solution_score(
        self: Any,
        mf: Any,
        label: str,
        atom_spin_map: Dict[int, int],
        min_bs_s2: float,
    ) -> float:
        if mf is None:
            return -1e12
        score = 0.0
        if getattr(mf, "converged", False):
            score += 1000.0
        s2 = _spin_square(mf)
        if np.isfinite(s2):
            if label.upper().startswith("BROKEN"):
                score += 150.0 if s2 >= min_bs_s2 else -300.0
                score += 75.0 if _local_spin_pattern_ok(self.mol, mf, atom_spin_map) else -75.0
            else:
                score -= abs(s2 - getattr(self, "_target_s2_ideal", s2)) * 10.0
        try:
            if np.isfinite(float(mf.e_tot)):
                score += 1.0
        except Exception:
            pass
        return score

    def _solution_acceptable(
        self: Any,
        mf: Any,
        label: str,
        atom_spin_map: Dict[int, int],
        min_bs_s2: float,
    ) -> bool:
        if mf is None or not getattr(mf, "converged", False):
            return False
        if not label.upper().startswith("BROKEN"):
            return True
        s2 = _spin_square(mf)
        if not np.isfinite(s2) or s2 < min_bs_s2:
            return False
        return _local_spin_pattern_ok(self.mol, mf, atom_spin_map)

    def _run_adaptive_scf(
        self: Any,
        label: str,
        atom_spin_map: Optional[Dict[int, int]] = None,
        min_bs_s2: float = 0.10,
    ) -> Tuple[Optional[Any], float]:
        atom_spin_map = _sanitize_atom_spin_map(self.mol, atom_spin_map)
        guesses = self._initial_guess_candidates(label, atom_spin_map)
        stages = _stage_plan(label)

        t0 = time.time()
        best_mf = None
        best_score = -1e12

        print(f"  > {C.BOLD}Adaptive SCF protocol enabled{C.END}")
        print(f"  > Guess candidates: {len(guesses)} | stages per candidate: {len(stages)}")

        for guess_name, dm0 in guesses:
            current_dm = _copy_dm(dm0)
            print(f"\n  > {C.CYAN}Trying guess: {guess_name}{C.END}")

            for stage in stages:
                mf = self._configure_attempt_mf(stage)
                dm_backend = current_dm
                try:
                    dm_backend = self._as_backend_array(current_dm)
                except Exception:
                    pass

                print(
                    f"    - {stage['name']}: shift={stage['level_shift']} "
                    f"damp={stage['damp']} maxcyc={stage['max_cycle']} "
                    f"diis={stage.get('diis_class', 'default')}"
                )

                try:
                    mf.kernel(dm0=dm_backend)
                except Exception as exc:
                    print(f"      {C.YELLOW}[stage failed] {repr(exc)}{C.END}")

                try:
                    mf = self._run_stability_refinement(mf)
                except Exception:
                    pass

                s2 = _spin_square(mf)
                conv = bool(getattr(mf, "converged", False))
                try:
                    energy_txt = f"{float(mf.e_tot):.10f}"
                except Exception:
                    energy_txt = "N/A"
                print(f"      result: converged={conv} E={energy_txt} S2={s2:.5f}")

                score = self._solution_score(mf, label, atom_spin_map, min_bs_s2)
                if score > best_score:
                    best_mf, best_score = mf, score

                if self._solution_acceptable(mf, label, atom_spin_map, min_bs_s2):
                    elapsed = time.time() - t0
                    print(f"  > {C.GREEN}Accepted adaptive SCF solution from {stage['name']}.{C.END}")
                    return mf, elapsed

                next_dm = _safe_dm_from_mf(mf)
                if next_dm is not None:
                    current_dm = next_dm

        elapsed = time.time() - t0
        if best_mf is not None:
            print(f"  > {C.YELLOW}No fully acceptable SCF solution found; using best available attempt.{C.END}")
        return best_mf, elapsed

    def run_calculation(self: Any, label: str, spin: int, atom_spin_map: Optional[Dict[int, int]] = None) -> Optional[Dict[str, Any]]:
        print(f"\n{C.BOLD}{C.BLUE}{'#' * 80}")
        print(f"{(label + f'  (2S = {spin})').center(80)}")
        print(f"{'#' * 80}{C.END}\n")

        gc.collect()
        self.mol.spin = spin
        self.mol.build()
        self._print_geometry(self.mol)

        mf_probe = self._make_uks(self.mol)
        if mf_probe.__class__.__module__.lower().startswith("gpu4pyscf"):
            print(f"  > {C.GREEN}Backend: gpu4pyscf{C.END}")
        else:
            print(f"  > {C.YELLOW}Backend: pyscf-compatible backend{C.END}")
        print(f"  > {C.BOLD}Density Fitting (RI) Enabled{C.END}")

        self._target_s2_ideal = 0.25 * spin * (spin + 2)

        mf, elapsed = self._run_adaptive_scf(label, atom_spin_map=atom_spin_map)
        if mf is None:
            print(f"\n{C.RED}[FATAL ERROR] All adaptive SCF attempts failed before producing a usable object.{C.END}")
            return None

        converged = bool(getattr(mf, "converged", False))
        status_str = f"{C.GREEN}CONVERGED{C.END}" if converged else f"{C.RED}NOT CONVERGED{C.END}"
        print(f"\n    Status  : {status_str}   ({elapsed:.2f} s)")
        print(f"    Energy  : {float(mf.e_tot):.10f} Eh")

        self._analyze_population(self.mol, mf)
        self._print_energy_components(self.mol, mf)

        mo_occ = _to_cpu_array(mf.mo_occ)
        if mo_occ.ndim == 1:
            n_alpha = n_beta = int(round(np.sum(mo_occ) / 2.0))
        else:
            n_alpha = int(round(np.sum(mo_occ[0])))
            n_beta = int(round(np.sum(mo_occ[1])))

        s2_actual = _spin_square(mf)
        s_ideal = (n_alpha - n_beta) / 2.0
        s2_ideal = s_ideal * (s_ideal + 1.0)

        result = {
            "label": label,
            "E": float(mf.e_tot),
            "S2": float(s2_actual),
            "S2_ideal": float(s2_ideal),
            "Spin": float((np.sqrt(max(0.0, 1.0 + 4.0 * s2_actual)) - 1.0) / 2.0),
            "n_alpha": int(n_alpha),
            "n_beta": int(n_beta),
            "converged": bool(converged),
            "wall_time": float(elapsed),
            "adaptive_score": float(self._solution_score(mf, label, _sanitize_atom_spin_map(self.mol, atom_spin_map), 0.10)),
        }

        if label == "HIGH SPIN":
            self.hs_result = result
            self.hs_mf = mf
        else:
            self.bs_result = result
            self.bs_mf = mf

        self._persist_results_json()
        return result

    calc_cls.set_prior_densities = set_prior_densities
    calc_cls.get_last_density = get_last_density
    calc_cls._configure_attempt_mf = _configure_attempt_mf
    calc_cls._initial_guess_candidates = _initial_guess_candidates
    calc_cls._solution_score = _solution_score
    calc_cls._solution_acceptable = _solution_acceptable
    calc_cls._run_adaptive_scf = _run_adaptive_scf
    calc_cls.run_calculation = run_calculation
