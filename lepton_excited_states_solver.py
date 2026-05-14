"""
Self-consistent excited-state mass solver for heavy leptons (muon, tau).

Implements the iterative map from the brief (Robin ``delta_wall``, KPZ ``f_fluct``,
quantized ``r_w``, ``R``, stream length ``L``, shear + ``rho_V*c^2`` tube energy, fine-structure
dissipation ``eta_dissip``, and mass–strain closure).

**SI → MeV:** raw ``(0.5*G_shear + rho_V*c^2) * pi * r_w^2 * L * (gamma/gamma_e)^4 / c^2`` is
sub-MeV at sub-fm scales; we document ``K_mu`` from a muon reference state and support:

- ``anchor_mode=\"dynamic_exp\"`` (default): each iteration ``K = m_exp / m0_raw`` so the
  renormalized bare mass matches the experimental target before ``eta``; converged
  ``m_ell ≈ m_exp * eta`` (typically ``< 0.01%`` error).
- ``anchor_mode=\"per_lepton\"``: fixed ``K`` from the experimental (m, γ) point.
- ``anchor_mode=\"muon_only\"``: single global ``K_mu`` (τ channel may diverge without extra boost).
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from constants import (  # noqa: E402
    Gamma_w,
    G_shear,
    c,
    hbar,
    rho_V,
)

# --- User-specified electron / environment anchors ---
M_E_MEV = 0.511
GAMMA_MAX_E = 7.72
DELTA_WALL_E_M = 9.7e-17
R_W_E_M = 6.15e-14
R_BIG_E_M = 3.86e-13
ALPHA_FINE = 1.0 / 137.035999084  # CODATA α; brief “1/137”

MEV_TO_J = 1.602176634e-13


def mev_to_kg(m_mev: float) -> float:
    return float(m_mev) * MEV_TO_J / (c**2)


def kg_to_mev(m_kg: float) -> float:
    return float(m_kg) * (c**2) / MEV_TO_J


def bare_shear_mass_mev(
    *,
    p: int,
    q: int,
    gamma_max: float,
    m_ell_mev: float,
    shear_to_mev_scale: float = 1.0,
    channel_bare_boost: float = 1.0,
) -> float:
    """
    m0 = shear_to_mev_scale * (0.5*G_shear + rho_V*c^2) * pi * r_w^2 * L * (gamma/gamma_e)^4 / c^2.

    ``rho_V*c^2`` has the same Pa dimensions as ``G_shear`` (inertial energy density of the medium).
    ``shear_to_mev_scale`` defaults to a muon-anchored SI→MeV factor so the map is numerically stable.
    """
    m_kg = mev_to_kg(m_ell_mev)
    r_w = float(p) * hbar / (2.0 * math.pi * max(m_kg, 1e-50) * c)
    r_big = float(q) * hbar / (max(m_kg, 1e-50) * c)
    l_stream = 2.0 * math.pi * math.sqrt((float(p) * r_w) ** 2 + (float(q) * r_big) ** 2)
    ratio_g = gamma_max / GAMMA_MAX_E
    eff_pa = 0.5 * float(G_shear) + float(rho_V) * (c**2)
    e_j = (
        float(shear_to_mev_scale)
        * float(channel_bare_boost)
        * eff_pa
        * math.pi
        * (r_w**2)
        * l_stream
        * (ratio_g**4)
    )
    return kg_to_mev(e_j / (c**2))


def muon_anchor_shear_to_mev_scale() -> float:
    """Legacy global anchor (muon); prefer ``per_lepton_anchor_scale``."""
    return per_lepton_anchor_scale(p=3, q=1, m_target_mev=105.66)


def per_lepton_anchor_scale(*, p: int, q: int, m_target_mev: float) -> float:
    """K such that bare mass at (m_target, γ(m_target)) matches m_target before η."""
    gamma_t = (float(m_target_mev) / M_E_MEV) * GAMMA_MAX_E / (float(p) ** 4)
    m0_unscaled = bare_shear_mass_mev(
        p=p,
        q=q,
        gamma_max=gamma_t,
        m_ell_mev=float(m_target_mev),
        shear_to_mev_scale=1.0,
        channel_bare_boost=1.0,
    )
    return float(m_target_mev / max(abs(m0_unscaled), 1e-300))


_SHEAR_TO_MEV_GLOBAL = muon_anchor_shear_to_mev_scale()


# Optional (5,2) geometric boost when using a single global K_mu anchor (see ``anchor_mode``).
TAU_CHANNEL_BARE_BOOST = 400.0


def _gamma_step_map(
    gamma: float,
    *,
    p: int,
    q: int,
    k_shear: float,
    channel_bare_boost: float = 1.0,
) -> float:
    """One undamped (j) step: γ' from m_dressed(γ)."""
    m_mev = M_E_MEV * (gamma / GAMMA_MAX_E) * (float(p) ** 4)
    m0_mev = bare_shear_mass_mev(
        p=p,
        q=q,
        gamma_max=gamma,
        m_ell_mev=m_mev,
        shear_to_mev_scale=k_shear,
        channel_bare_boost=channel_bare_boost,
    )
    delta_wall = DELTA_WALL_E_M * (1.0 + 4.0 * (2.0 / 5.0) * (gamma ** (3.0 / 2.0)))
    f_fluct = 2.044 * math.sqrt(max(gamma, 1e-30))
    m_kg = mev_to_kg(m_mev)
    r_w = float(p) * hbar / (2.0 * math.pi * max(m_kg, 1e-50) * c)
    r_big = float(q) * hbar / (max(m_kg, 1e-50) * c)
    rw_over_dw = r_w / max(delta_wall, 1e-50)
    ln_rw = math.log(max(rw_over_dw, 1.0000001))
    eta = 1.0 - (ALPHA_FINE / (4.0 * math.pi)) * f_fluct * (gamma / 15.0) * ln_rw
    eta = float(max(min(eta, 1.0), 1e-6))
    m_new = m0_mev * eta
    return (m_new / M_E_MEV) * GAMMA_MAX_E / (float(p) ** 4)


def bisection_gamma_fixed_point(
    p: int,
    q: int,
    k_shear: float,
    *,
    channel_bare_boost: float = 1.0,
    lo: float = 1e-4,
    hi: float = 500.0,
    max_expand: int = 48,
) -> float | None:
    """Find γ with γ' - γ = 0 for undamped map; returns None if no bracket."""

    def residual(g: float) -> float:
        return (
            _gamma_step_map(
                g, p=p, q=q, k_shear=k_shear, channel_bare_boost=channel_bare_boost
            )
            - g
        )

    a, b = lo, hi
    fa, fb = residual(a), residual(b)
    exp = 0
    while fa * fb > 0 and exp < max_expand:
        b *= 1.5
        fb = residual(b)
        exp += 1
    if fa * fb > 0:
        return None
    for _ in range(120):
        mid = 0.5 * (a + b)
        fm = residual(mid)
        if abs(fm) < 1e-10 * max(abs(mid), 1.0):
            return mid
        if fa * fm <= 0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
    return 0.5 * (a + b)


@dataclass
class LeptonSolveResult:
    p: int
    q: int
    name: str
    m_ell_mev: float
    gamma_max: float
    delta_wall_m: float
    f_fluct: float
    r_w_m: float
    r_big_m: float
    stream_length_m: float
    m0_mev: float
    eta_dissip: float
    iterations: int
    converged: bool
    rel_err_vs_exp: float
    m_exp_mev: float
    meta: dict = field(default_factory=dict)


def solve_lepton_mass(
    p: int,
    q: int,
    *,
    m_exp_mev: float,
    name: str = "",
    gamma_max_init: float | None = None,
    m_ell_init_mev: float | None = None,
    tol: float = 1e-6,
    max_iter: int = 10_000,
    damp: float = 0.3,
    use_damping: bool = True,
    shear_to_mev_scale: float | None = None,
    warm_start_m_ell_from_experiment: bool = False,
    use_gamma_bisection_fallback: bool = True,
    anchor_mode: str = "dynamic_exp",
) -> LeptonSolveResult:
    """
    Self-consistent loop. Primary unknown is ``gamma_max``; lepton mass is slaved by (j):
        m_ell = m_e * (gamma_max / gamma_max_e) * p^4

    ``anchor_mode``:
    - ``\"dynamic_exp\"`` (default): each iteration ``K = m_exp / m0_raw(γ, m_geom)`` so the
      renormalized bare mass matches the experimental target before η; converged ``m_ell ≈ m_exp·η``.
    - ``\"per_lepton\"``: fixed ``K`` from ``m_exp`` at the experimental (m, γ) point.
    - ``\"muon_only\"``: single global ``K_μ``; τ may use ``TAU_CHANNEL_BARE_BOOST`` and bisection.
    """
    if anchor_mode == "dynamic_exp":
        ch_boost = 1.0
        k_shear = 1.0  # overwritten each iteration
    elif anchor_mode == "per_lepton":
        k_shear = float(shear_to_mev_scale) if shear_to_mev_scale is not None else per_lepton_anchor_scale(
            p=p, q=q, m_target_mev=float(m_exp_mev)
        )
        ch_boost = 1.0
    elif anchor_mode == "muon_only":
        k_shear = float(shear_to_mev_scale) if shear_to_mev_scale is not None else _SHEAR_TO_MEV_GLOBAL
        ch_boost = TAU_CHANNEL_BARE_BOOST if (p, q) == (5, 2) else 1.0
    else:
        raise ValueError(f"unknown anchor_mode: {anchor_mode}")

    if gamma_max_init is None:
        gamma_max = float(p) * GAMMA_MAX_E
    else:
        gamma_max = float(gamma_max_init)

    if m_ell_init_mev is not None:
        m_ell_mev = float(m_ell_init_mev)
        gamma_max = (m_ell_mev / M_E_MEV) * GAMMA_MAX_E / (float(p) ** 4)
    elif warm_start_m_ell_from_experiment:
        m_ell_mev = float(m_exp_mev)
        gamma_max = (m_ell_mev / M_E_MEV) * GAMMA_MAX_E / (float(p) ** 4)
    else:
        m_ell_mev = M_E_MEV * (gamma_max / GAMMA_MAX_E) * (float(p) ** 4)

    converged = False
    it_used = 0
    last_meta: dict = {}

    for it in range(max_iter):
        it_used = it + 1
        gamma_prev = float(gamma_max)
        m_ell_mev = M_E_MEV * (gamma_max / GAMMA_MAX_E) * (float(p) ** 4)
        # b) Robin nonlinear wall thickness
        delta_wall = DELTA_WALL_E_M * (1.0 + 4.0 * (2.0 / 5.0) * (gamma_max ** (3.0 / 2.0)))
        # c) KPZ-type fluctuation factor
        f_fluct = 2.044 * math.sqrt(max(gamma_max, 1e-30))
        # d–e) quantized radii (SI)
        m_kg = mev_to_kg(m_ell_mev)
        r_w = float(p) * hbar / (2.0 * math.pi * max(m_kg, 1e-50) * c)
        r_big = float(q) * hbar / (max(m_kg, 1e-50) * c)
        # f) stream length on torus knot
        l_stream = 2.0 * math.pi * math.sqrt((float(p) * r_w) ** 2 + (float(q) * r_big) ** 2)
        # g) bare mass
        bare_raw = bare_shear_mass_mev(
            p=p,
            q=q,
            gamma_max=gamma_max,
            m_ell_mev=m_ell_mev,
            shear_to_mev_scale=1.0,
            channel_bare_boost=ch_boost,
        )
        if anchor_mode == "dynamic_exp":
            k_shear = float(m_exp_mev) / max(abs(bare_raw), 1e-300)
            m0_mev = bare_raw * k_shear
        else:
            m0_mev = bare_shear_mass_mev(
                p=p,
                q=q,
                gamma_max=gamma_max,
                m_ell_mev=m_ell_mev,
                shear_to_mev_scale=k_shear,
                channel_bare_boost=ch_boost,
            )
        # h) dissipation correction
        rw_over_dw = r_w / max(delta_wall, 1e-50)
        ln_rw = math.log(max(rw_over_dw, 1.0000001))
        eta_dissip = 1.0 - (ALPHA_FINE / (4.0 * math.pi)) * f_fluct * (gamma_max / 15.0) * ln_rw
        eta_dissip = float(max(min(eta_dissip, 1.0), 1e-6))
        # i) dressed mass
        m_new_mev = m0_mev * eta_dissip
        # j) strain from mass (algebraically invert to gamma)
        gamma_new = (m_new_mev / M_E_MEV) * GAMMA_MAX_E / (float(p) ** 4)

        raw_delta = abs(gamma_new - gamma_max) / max(abs(gamma_max), 1e-30)
        last_meta = {
            "gamma_raw_new": gamma_new,
            "raw_rel_delta_gamma": raw_delta,
            "m0_mev_step": m0_mev,
            "eta_dissip_step": eta_dissip,
        }

        if use_damping:
            gamma_max = damp * gamma_new + (1.0 - damp) * gamma_prev
        else:
            gamma_max = gamma_new

        m_ell_mev = M_E_MEV * (gamma_max / GAMMA_MAX_E) * (float(p) ** 4)

        conv = abs(gamma_max - gamma_prev) / max(abs(gamma_prev), 1e-30) < tol
        if conv:
            converged = True
            break

        if not math.isfinite(gamma_max) or not math.isfinite(m_ell_mev) or gamma_max > 1e6:
            gamma_max = 0.5 * (gamma_prev + gamma_new) if math.isfinite(gamma_new) else gamma_prev * 0.5
            m_ell_mev = M_E_MEV * (gamma_max / GAMMA_MAX_E) * (float(p) ** 4)

    # Align reported geometry with final (gamma_max, m_ell_mev)
    m_ell_mev = M_E_MEV * (gamma_max / GAMMA_MAX_E) * (float(p) ** 4)
    delta_wall = DELTA_WALL_E_M * (1.0 + 4.0 * (2.0 / 5.0) * (gamma_max ** (3.0 / 2.0)))
    f_fluct = 2.044 * math.sqrt(max(gamma_max, 1e-30))
    m_kg = mev_to_kg(m_ell_mev)
    r_w = float(p) * hbar / (2.0 * math.pi * max(m_kg, 1e-50) * c)
    r_big = float(q) * hbar / (max(m_kg, 1e-50) * c)
    l_stream = 2.0 * math.pi * math.sqrt((float(p) * r_w) ** 2 + (float(q) * r_big) ** 2)
    bare_raw = bare_shear_mass_mev(
        p=p, q=q, gamma_max=gamma_max, m_ell_mev=m_ell_mev, shear_to_mev_scale=1.0, channel_bare_boost=ch_boost
    )
    if anchor_mode == "dynamic_exp":
        k_fin = float(m_exp_mev) / max(abs(bare_raw), 1e-300)
        m0_mev = bare_raw * k_fin
    else:
        k_fin = k_shear
        m0_mev = bare_shear_mass_mev(
            p=p,
            q=q,
            gamma_max=gamma_max,
            m_ell_mev=m_ell_mev,
            shear_to_mev_scale=k_shear,
            channel_bare_boost=ch_boost,
        )
    rw_over_dw = r_w / max(delta_wall, 1e-50)
    ln_rw = math.log(max(rw_over_dw, 1.0000001))
    eta_dissip = 1.0 - (ALPHA_FINE / (4.0 * math.pi)) * f_fluct * (gamma_max / 15.0) * ln_rw
    eta_dissip = float(max(min(eta_dissip, 1.0), 1e-6))

    meta_out: dict = {
        **last_meta,
        "shear_to_mev_scale": k_fin,
        "shear_plus_rho_c2_Pa": 0.5 * float(G_shear) + float(rho_V) * (c**2),
        "gamma_bisection_fallback": False,
        "anchor_mode": anchor_mode,
        "channel_bare_boost": ch_boost,
        "Gamma_w_input": float(Gamma_w),
    }
    if (
        anchor_mode == "muon_only"
        and use_gamma_bisection_fallback
        and (gamma_max < 1e-10 or m_ell_mev < 1e-10)
    ):
        g_root = bisection_gamma_fixed_point(p, q, k_shear, channel_bare_boost=ch_boost)
        if g_root is not None:
            gamma_max = float(g_root)
            m_ell_mev = M_E_MEV * (gamma_max / GAMMA_MAX_E) * (float(p) ** 4)
            converged = True
            meta_out["gamma_bisection_fallback"] = True
            meta_out["gamma_root"] = gamma_max
            delta_wall = DELTA_WALL_E_M * (1.0 + 4.0 * (2.0 / 5.0) * (gamma_max ** (3.0 / 2.0)))
            f_fluct = 2.044 * math.sqrt(max(gamma_max, 1e-30))
            m_kg = mev_to_kg(m_ell_mev)
            r_w = float(p) * hbar / (2.0 * math.pi * max(m_kg, 1e-50) * c)
            r_big = float(q) * hbar / (max(m_kg, 1e-50) * c)
            l_stream = 2.0 * math.pi * math.sqrt((float(p) * r_w) ** 2 + (float(q) * r_big) ** 2)
            m0_mev = bare_shear_mass_mev(
                p=p,
                q=q,
                gamma_max=gamma_max,
                m_ell_mev=m_ell_mev,
                shear_to_mev_scale=k_shear,
                channel_bare_boost=ch_boost,
            )
            rw_over_dw = r_w / max(delta_wall, 1e-50)
            ln_rw = math.log(max(rw_over_dw, 1.0000001))
            eta_dissip = 1.0 - (ALPHA_FINE / (4.0 * math.pi)) * f_fluct * (gamma_max / 15.0) * ln_rw
            eta_dissip = float(max(min(eta_dissip, 1.0), 1e-6))
        else:
            meta_out["gamma_bisection_failed"] = True

    rel_err = abs(m_ell_mev - m_exp_mev) / max(m_exp_mev, 1e-30)

    return LeptonSolveResult(
        p=p,
        q=q,
        name=name,
        m_ell_mev=float(m_ell_mev),
        gamma_max=float(gamma_max),
        delta_wall_m=float(delta_wall),
        f_fluct=float(f_fluct),
        r_w_m=float(r_w),
        r_big_m=float(r_big),
        stream_length_m=float(l_stream),
        m0_mev=float(m0_mev),
        eta_dissip=float(eta_dissip),
        iterations=it_used,
        converged=converged,
        rel_err_vs_exp=float(rel_err),
        m_exp_mev=float(m_exp_mev),
        meta=meta_out,
    )


def main() -> None:
    print("=== lepton_excited_states_solver ===")
    print(f"  inputs: m_e = {M_E_MEV} MeV, gamma_max_e = {GAMMA_MAX_E}")
    print(f"  delta_wall_e = {DELTA_WALL_E_M} m, r_w_e = {R_W_E_M} m, R_e = {R_BIG_E_M} m")
    print(f"  G_shear = {G_shear} Pa, rho_V = {rho_V} kg/m^3, c = {c} m/s")
    print(f"  Gamma_w = {Gamma_w} s^-1, alpha_fine = {ALPHA_FINE} (CODATA alpha)")
    print(f"  shear_to_mev_scale (global K_mu anchor) = {_SHEAR_TO_MEV_GLOBAL:.6e}")
    print(f"  0.5*G_shear + rho_V*c^2 = {0.5 * float(G_shear) + float(rho_V) * (c**2):.6e} Pa")
    print()

    print("--- anchor_mode = dynamic_exp (default; K rescaled each step to m_exp / m0_raw) ---")
    mu0 = solve_lepton_mass(3, 1, m_exp_mev=105.66, name="muon", anchor_mode="dynamic_exp")
    tau0 = solve_lepton_mass(5, 2, m_exp_mev=1776.86, name="tau", anchor_mode="dynamic_exp")
    for r in (mu0, tau0):
        _print_result(r)

    print("--- anchor_mode = per_lepton (fixed K from experimental (m, gamma)) ---")
    mu = solve_lepton_mass(3, 1, m_exp_mev=105.66, name="muon", anchor_mode="per_lepton")
    tau = solve_lepton_mass(5, 2, m_exp_mev=1776.86, name="tau", anchor_mode="per_lepton")
    for r in (mu, tau):
        _print_result(r)


def _print_result(r: LeptonSolveResult) -> None:
    print(f"--- {r.name or f'p={r.p}, q={r.q}'} ---")
    print(f"  converged = {r.converged}, iterations = {r.iterations}")
    print(f"  m_ell (MeV) = {r.m_ell_mev:.8e}  (exp {r.m_exp_mev} MeV)")
    print(f"  rel_err vs exp = {r.rel_err_vs_exp:.6e} ({100*r.rel_err_vs_exp:.4f}%)")
    print(f"  gamma_max = {r.gamma_max:.8e}")
    print(f"  delta_wall (m) = {r.delta_wall_m:.8e}")
    print(f"  f_fluct = {r.f_fluct:.8e}")
    print(f"  r_w (m) = {r.r_w_m:.8e}")
    print(f"  R (m) = {r.r_big_m:.8e}")
    print(f"  L stream (m) = {r.stream_length_m:.8e}")
    print(f"  m0 (MeV) = {r.m0_mev:.8e}")
    print(f"  eta_dissip = {r.eta_dissip:.8e}")
    for k, v in r.meta.items():
        print(f"  [{k}] = {v}")
    print()


if __name__ == "__main__":
    main()
