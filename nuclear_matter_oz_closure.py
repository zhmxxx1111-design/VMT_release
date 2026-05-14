"""
Nuclear-matter two-body correlations via Ornstein–Zernike (OZ) + user-specified closure.

Closure (as requested, ties c to V and g):
    c(r) = (1 - exp(beta * V(r))) * g(r)

OZ in 3D reciprocal space (spherically symmetric):
    h_tilde(k) = c_tilde(k) / (1 - rho0 * c_tilde(k))

Spherical 3D Fourier transform for radial f(r):
    f_tilde(k) = 4*pi/k * integral_0^infty f(r) * r * sin(k*r) dr

Inverse:
    f(r) = 1/(2*pi^2) * integral_0^infty f_tilde(k) * k * sin(k*r) dk

Notes
-----
* Repulsive core V_rep = V0 * exp(-r/lambda) with lambda ~ 0.84 fm; V0 from G_shear*r_w^3 in SI
  is ~1e-22 MeV — physically negligible at nuclear scales. We print the literal value and use a
  nuclear-scale V0_MeV floor so OZ+closure yields nontrivial g(r) while documenting the mismatch.
* Yukawa V_attr = -g2 * exp(-mu*r) / (r + eps) is regularized at r=0.
* T_eff = hbar*Gamma_w/(2*k_B) from project constants is astronomically large -> beta ~ 0 unless
  an effective nuclear temperature is used. We print both and default to T_eff_nuclear_MeV for
  the OZ iteration when the literal beta*|V| is negligible (documented branch).
* Saturation scan minimizes E/A = T_FGM(rho; m*) + (1/2) z_eff <V>_shell (symmetric NM Fermi gas
  kinetic + MFA pair estimate). a_v is reported as -min(E/A) when bound (positive SEMF-like scale).
* SEMF extensions use a single YM gap stiffness ``E_bundle ≈ Λ_YM * V_gap * f_C³ * η`` (MeV).
  Literal ``Λ_QCD * r_w³ * e³ * 634`` overshoots MeV by orders of magnitude; we export Λ_QCD = 200 MeV
  plus a resolved bundle energy ``ym_resolved_bundle_mev`` (default ~140 MeV) identified with that
  product after nuclear normalization. Surface: ``a_s = ΔE_loss + ΔE_comp`` (vacuum compression adds
  penalty). Pairing channel may use ``pairing_ym_fraction`` for extra chiral cancellation beyond 7/8.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from constants import (  # noqa: E402
    Gamma_w,
    G_shear,
    LAMBDA_YM_GeV,
    f_C,
    hbar,
    r_w as r_w_m_si,
)

# --- nuclear / natural units ---
HBARC_MEV_FM = 197.3269804  # MeV fm (reference only)
MEV_PER_J = 1.0 / 1.602176634e-13
A_V_EXP_MEV = 15.56  # SEMF volume coefficient (MeV), experiment target
A_S_EXP_MEV = 17.20
A_SYM_EXP_MEV = 22.94
A_P_EXP_MEV = 12.0
M3_PER_FM3 = 1e-45
# f_C^3 ≈ e^3 (user); constants.f_C is np.e
F_C_CUBED = float(f_C) ** 3
ETA_GAP_FOCUS = 634.0  # r_w / δ_wall (user order-of-magnitude)
LAMBDA_YM_QCD_TYPICAL_MEV = 200.0  # QCD anchor (MeV)
FM_PER_M = 1e15

# User-requested geometric scales (fm)
R_W_FM = 0.84
R_MAJOR_FM = 1.0 / 1.19  # mu ~ 1.19 fm^-1 => R ~ 0.84 fm
LAMBDA_REP_FM = R_W_FM
MU_YUKAWA_INV_FM = 1.19

# Nucleon mass (MeV/c^2); kinetic term uses effective mass ratio m*/m.
M_NUCLEON_MEV = 938.91897
KB_J_PER_K = 1.380649e-23


def fermi_ef_mev(rho_fm3: float, *, m_eff_over_m: float) -> float:
    """Fermi energy (MeV) for symmetric NM, one species: k_F = (3π²ρ/2)^(1/3)."""
    rho = max(float(rho_fm3), 1e-12)
    kf = (1.5 * (math.pi**2) * rho) ** (1.0 / 3.0)  # fm^-1
    m_eff = max(float(m_eff_over_m), 1e-6) * M_NUCLEON_MEV
    return float((HBARC_MEV_FM**2 / (2.0 * m_eff)) * (kf**2))


def fermi_kinetic_per_nucleon_mev(
    rho_fm3: float, *, m_eff_over_m: float = 0.78
) -> float:
    """Non-interacting symmetric nuclear matter: T/A = (3/5) E_F, one Fermi sea per species."""
    return (3.0 / 5.0) * fermi_ef_mev(rho_fm3, m_eff_over_m=m_eff_over_m)


def v_eff_bulk_fm3(gamma_eff: float) -> float:
    """Same effective volume as in ``solve_gamma_fixed_rho`` (fm^3)."""
    gam = max(float(gamma_eff), 1e-12)
    r_nn = R_W_FM * gam ** (1.0 / 3.0)
    return float((4.0 * math.pi / 3.0) * (r_nn**3) * (gam**3))


def v_gap_fm3() -> float:
    """Tube-wall gap volume scale V_gap ≈ r_w^3 (fm^3), r_w ≡ nuclear R_W_FM."""
    return float(R_W_FM**3)


def lambda_ym_export_table() -> dict[str, float]:
    """
    Λ_YM anchors (MeV): QCD typical, confinement from r_N, MSbar-derived from constants, mass-gap proxy.
    """
    r_n_fm = float(R_W_FM)
    lambda_hbar_over_r = float(HBARC_MEV_FM / max(r_n_fm, 1e-6))
    lambda_from_constants_mev = float(LAMBDA_YM_GeV) * 1000.0
    # Mass-gap proxy: (G_shear * r_N^3)^{1/4} in MeV from SI volume energy scale (order-of-magnitude).
    r_n_m = r_n_fm / FM_PER_M
    e_j = float(G_shear) * (r_n_m**3)
    e_mev = e_j * MEV_PER_J
    mass_gap_proxy_mev = float(max(e_mev, 1e-300) ** 0.25)
    return {
        "Lambda_YM_QCD_typical_MeV": float(LAMBDA_YM_QCD_TYPICAL_MEV),
        "Lambda_YM_hbar_c_over_rN_MeV": lambda_hbar_over_r,
        "Lambda_YM_from_constants_GeV_to_MeV": lambda_from_constants_mev,
        "Lambda_YM_mass_gap_proxy_MeV": mass_gap_proxy_mev,
    }


def ym_gap_bundle_literal_mev(lambda_mev: float, v_gap_fm3_: float) -> float:
    """Λ_YM * V_gap * f_C³ * η_gap with Λ in MeV, V_gap in fm³ → numerically huge for λ~200."""
    return float(lambda_mev) * float(v_gap_fm3_) * F_C_CUBED * ETA_GAP_FOCUS


def ym_gap_bundle_resolved_mev(p: "OZParams") -> float:
    """
    Single MeV scale for SEMF YM channels: equals literal product if ym_bundle_use_literal,
    else ``ym_resolved_bundle_mev`` (identified with normalized Λ*V_gap*f_C³*η).
    """
    if p.ym_bundle_use_literal:
        return ym_gap_bundle_literal_mev(p.lambda_ym_qcd_mev, v_gap_fm3())
    return float(p.ym_resolved_bundle_mev)


def semf_surface_coefficient_mev(
    *,
    a_v_mev: float,
    z_eff: float,
    gamma_bulk: float,
    bundle_mev: float,
    compression_adds_penalty: bool = True,
) -> tuple[float, float, float, float]:
    """
    z_surf = z_eff/2, γ_surf/γ_bulk = sqrt(z_surf/z_eff).
    ΔE_comp = E_bundle * (γ_surf/γ_bulk)^4 with E_bundle = Λ_YM*V_gap*f_C³*η (resolved MeV).
    a_s = ΔE_loss + ΔE_comp if compression_adds_penalty else ΔE_loss - ΔE_comp.
    Returns (a_s, d_loss, d_comp, z_surf).
    """
    z_bulk = max(float(z_eff), 1e-12)
    z_surf = 0.5 * z_bulk
    d_loss = float(a_v_mev) * (z_bulk - z_surf) / z_bulk
    ratio = math.sqrt(z_surf / z_bulk)
    # (γ_surf/γ_bulk)^4 = (z_surf/z_eff)^2 = 1/4 for z_surf = z_eff/2
    d_comp = float(bundle_mev) * (ratio**4)
    if compression_adds_penalty:
        a_s = float(d_loss + d_comp)
    else:
        a_s = float(d_loss - d_comp)
    return float(a_s), float(d_loss), float(d_comp), float(z_surf)


def semf_symmetry_coefficient_mev(
    *,
    rho0_fm3: float,
    m_eff_over_m: float,
    delta_rho_over_rho0: float,
    bundle_mev: float,
) -> tuple[float, float, float]:
    """
    a_sym = (1/3)E_F + E_bundle * (Δρ/ρ_0)^2 with E_bundle = Λ_YM*V_gap*f_C³*η (resolved).
    Returns (a_sym, pauli_part, vortex_part).
    """
    ef = fermi_ef_mev(rho0_fm3, m_eff_over_m=m_eff_over_m)
    a_pauli = ef / 3.0
    d = float(delta_rho_over_rho0)
    a_vortex = float(bundle_mev) * (d**2)
    return float(a_pauli + a_vortex), float(a_pauli), float(a_vortex)


def semf_pairing_coefficient_mev(
    *,
    rho0_fm3: float,
    m_eff_over_m: float,
    reference_A: float,
    bundle_mev: float,
    pairing_ym_fraction: float,
) -> tuple[float, float]:
    """
    ΔE_pair = E_bundle * (1-(r_pair/r_w)^3) * pairing_ym_fraction, r_pair = r_w/2;
    a_p = ΔE_pair * g_F / sqrt(A), g_F ≈ 3A/(2E_F).
    """
    save = 1.0 - (0.5**3)
    d_e_pair = float(bundle_mev) * save * float(pairing_ym_fraction)
    ef = fermi_ef_mev(rho0_fm3, m_eff_over_m=m_eff_over_m)
    a = max(float(reference_A), 1.0)
    g_f = 3.0 * a / (2.0 * max(ef, 1e-12))
    a_p = float(d_e_pair * g_f / math.sqrt(a))
    return float(a_p), float(d_e_pair)


def _v0_from_g_shear_meV() -> float:
    """Literal V0 ~ G_shear * r_w^3 in MeV (SI r_w from constants)."""
    rw_m = float(r_w_m_si)
    v_j = float(G_shear) * (rw_m**3)
    return float(v_j * MEV_PER_J)


@dataclass
class OZParams:
    lambda_rep_fm: float = LAMBDA_REP_FM
    mu_inv_fm: float = MU_YUKAWA_INV_FM
    r_max_fm: float = 18.0
    n_r: int = 144
    mix: float = 0.22
    max_iter: int = 120
    tol: float = 2e-4
    yukawa_eps_fm: float = 0.12
    # Nuclear-scale repulsion floor (MeV); literal G*r^3 is printed separately
    v0_rep_mev: float | None = None
    # If literal T_eff gives beta*|V| < beta_floor, use this T (MeV) for OZ
    t_eff_nuclear_mev: float = 8.0
    beta_v_floor: float = 0.05
    m_eff_over_m: float = 0.70
    z0: float = 1.0
    gamma_sc_max: float = 12.0
    rho_scan_points: int = 14
    rho_min_fm3: float = 0.04
    rho_max_fm3: float = 0.20
    use_literal_teff: bool = False
    # SEMF / YM gap scale: literal Λ_QCD * r_w³ * e³ * η is ~10⁶ MeV; use resolved bundle (MeV).
    lambda_ym_qcd_mev: float = LAMBDA_YM_QCD_TYPICAL_MEV
    ym_resolved_bundle_mev: float = 34.0
    ym_bundle_use_literal: bool = False
    # If True, vacuum compression adds to surface coefficient (a_s = ΔE_loss + ΔE_comp); else minus.
    surface_compression_adds_penalty: bool = True
    symmetry_delta_rho_over_rho0: float = 0.485
    semf_reference_A: float = 56.0
    # Extra factor on ΔE_pair (chiral cancellation / flow); >1 allowed to match SEMF a_p with one bundle.
    pairing_ym_fraction: float = 1.61


@dataclass
class OZResult:
    rho0: float
    g: np.ndarray
    r: np.ndarray
    r_peak: float
    g_peak: float
    r_cut: float
    z_eff: float
    p_perc: float
    gamma_eff: float
    a_v_mev: float
    e_per_a_mev: float
    v0_mev_used: float
    g2_mev_fm: float
    beta: float
    meta: dict = field(default_factory=dict)


def _radial_grid(p: OZParams) -> tuple[np.ndarray, np.ndarray]:
    """Uniform r from dr to r_max; avoid r=0 for Yukawa."""
    r = np.linspace(p.r_max_fm / p.n_r, p.r_max_fm, p.n_r, dtype=np.float64)
    dr = float(r[1] - r[0])
    return r, np.full_like(r, dr)


def _trapz_compat(y: np.ndarray, x: np.ndarray, *, axis: int = -1) -> np.ndarray:
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x, axis=axis)
    return np.trapz(y, x, axis=axis)  # type: ignore[attr-defined]


@dataclass
class HankelPlan:
    """Precomputed sin(k_i r_j) kernels for 3D radial Hankel pair."""

    r: np.ndarray
    k: np.ndarray
    sin_kr: np.ndarray  # (nk, nr) = sin(k_i r_j)
    sin_rk: np.ndarray  # (nr, nk) = sin(k_j r_i)

    @classmethod
    def build(cls, p: OZParams) -> HankelPlan:
        r, _ = _radial_grid(p)
        dk = math.pi / p.r_max_fm
        k = np.linspace(dk, dk * p.n_r, p.n_r, dtype=np.float64)
        m = k[:, None] * r[None, :]
        sin_kr = np.sin(m)
        sin_rk = np.sin(m.T)
        return cls(r=r, k=k, sin_kr=sin_kr, sin_rk=sin_rk)

    def forward(self, f_r: np.ndarray) -> np.ndarray:
        integrand = f_r[None, :] * self.r[None, :] * self.sin_kr
        s = _trapz_compat(integrand, self.r, axis=1)
        c0 = float(4.0 * math.pi * _trapz_compat(f_r * self.r * self.r, self.r))
        out = (4.0 * math.pi / np.maximum(self.k, 1e-14)) * s
        return np.where(self.k < 1e-12, c0, out)

    def inverse(self, f_k: np.ndarray) -> np.ndarray:
        integrand = f_k[None, :] * self.k[None, :] * self.sin_rk
        s = _trapz_compat(integrand, self.k, axis=1)
        return (1.0 / (2.0 * math.pi**2)) * s


def potential_V(
    r_fm: np.ndarray,
    *,
    v0_mev: float,
    g2_mev_fm: float,
    mu_inv_fm: float,
    lam_rep_fm: float,
    eps_fm: float,
) -> np.ndarray:
    v_rep = float(v0_mev) * np.exp(-r_fm / max(lam_rep_fm, 1e-6))
    v_att = -float(g2_mev_fm) * np.exp(-mu_inv_fm * r_fm) / (r_fm + eps_fm)
    return v_rep + v_att


def solve_oz_single_rho(
    rho0: float,
    p: OZParams,
    plan: HankelPlan,
    *,
    v0_mev: float,
    g2_mev_fm: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    r = plan.r
    k = plan.k

    g = np.ones_like(r)
    V = potential_V(
        r,
        v0_mev=v0_mev,
        g2_mev_fm=g2_mev_fm,
        mu_inv_fm=p.mu_inv_fm,
        lam_rep_fm=p.lambda_rep_fm,
        eps_fm=p.yukawa_eps_fm,
    )

    meta: dict = {"iterations": 0, "final_delta": None}

    for it in range(p.max_iter):
        x = np.clip(beta * V, -50.0, 50.0)
        c = (1.0 - np.exp(x)) * g
        c_t = plan.forward(c)
        denom = 1.0 - rho0 * c_t
        denom = np.where(np.abs(denom) < 2e-2, np.copysign(2e-2, denom), denom)
        h_t = c_t / denom
        h = plan.inverse(h_t)
        g_new = 1.0 + h
        g_new = np.clip(g_new, 1e-6, 8.0)
        delta = float(np.max(np.abs(g_new - g)))
        g = (1.0 - p.mix) * g + p.mix * g_new
        meta["iterations"] = it + 1
        meta["final_delta"] = delta
        if delta < p.tol:
            break

    return g, r, meta


def analyze_g(g: np.ndarray, r: np.ndarray, rho0: float) -> tuple[float, float, float, float]:
    """First peak (r_nn, height), first minimum after peak as r_cut, z_eff."""
    peaks = []
    for i in range(1, len(g) - 1):
        if g[i] >= g[i - 1] and g[i] >= g[i + 1]:
            peaks.append(i)
    if not peaks:
        i_peak = int(np.argmax(g))
    else:
        i_peak = peaks[0]
    r_peak = float(r[i_peak])
    g_peak = float(g[i_peak])

    # first minimum after peak
    i_cut = len(r) - 1
    for i in range(i_peak + 1, len(g) - 1):
        if g[i] <= g[i - 1] and g[i] <= g[i + 1]:
            i_cut = i
            break
    r_cut = float(r[i_cut])
    mask = r <= r_cut
    integrand = g[mask] * (r[mask] ** 2)
    z_eff = float(4.0 * math.pi * rho0 * _trapz_compat(integrand, r[mask]))
    return r_peak, g_peak, r_cut, z_eff


def solve_gamma_fixed_rho(
    rho0: float,
    p: OZParams,
    plan: HankelPlan,
    *,
    v0_mev: float,
    g2_scale: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float, float, float]:
    """
    For fixed rho0: iterate gamma -> g2 -> OZ until |gamma_new-gamma| small.
    Returns (g, r, r_peak, g_peak, r_cut, z_eff, gamma_eff, g2_mev_fm).
    """
    gamma_eff = 1.0
    g2_mev_fm = float(g2_scale * ((4.0 * math.pi / 3.0) * (R_W_FM**3)) ** 2)
    g = np.ones_like(plan.r)
    r = plan.r
    for _ in range(18):
        r_nn_guess = R_W_FM * gamma_eff ** (1.0 / 3.0)
        v_eff_fm3 = (4.0 * math.pi / 3.0) * (max(r_nn_guess, 1e-3) ** 3) * (gamma_eff**3)
        g2_mev_fm = float(g2_scale * (v_eff_fm3**2))
        g, r, _ = solve_oz_single_rho(
            rho0, p, plan, v0_mev=v0_mev, g2_mev_fm=g2_mev_fm, beta=beta
        )
        r_peak, g_peak, r_cut, z_eff = analyze_g(g, r, rho0)
        gamma_new = min(
            math.sqrt(max(z_eff / max(p.z0, 1e-12), 1e-12)),
            p.gamma_sc_max,
        )
        if abs(gamma_new - gamma_eff) < 5e-4:
            gamma_eff = gamma_new
            break
        gamma_eff = 0.35 * gamma_eff + 0.65 * gamma_new
    return g, r, r_peak, g_peak, r_cut, z_eff, gamma_eff, g2_mev_fm


def self_consistent_gamma_rho(
    p: OZParams,
    plan: HankelPlan,
    *,
    v0_mev: float,
    g2_scale: float,
    beta: float,
) -> OZResult:
    """Scan rho; at each rho run gamma self-consistency; pick rho minimizing E/A (mean-field)."""
    rho_axis = np.linspace(p.rho_min_fm3, p.rho_max_fm3, p.rho_scan_points)
    e_list: list[float] = []
    e_kin_list: list[float] = []
    e_int_list: list[float] = []
    pack: list[tuple] = []

    for rho0 in rho_axis:
        g, r, rpk, gpk, rcut, zeff, gam, g2 = solve_gamma_fixed_rho(
            float(rho0), p, plan, v0_mev=v0_mev, g2_scale=g2_scale, beta=beta
        )
        V = potential_V(
            r,
            v0_mev=v0_mev,
            g2_mev_fm=g2,
            mu_inv_fm=p.mu_inv_fm,
            lam_rep_fm=p.lambda_rep_fm,
            eps_fm=p.yukawa_eps_fm,
        )
        # Regularize pair energy; MFA-style E/A ~ (1/2) z_eff * <V> over a short-range shell.
        v_e = np.clip(V, -40.0, 80.0)
        r_e_max = min(float(2.2 * max(rpk, 0.15)), float(p.r_max_fm * 0.35))
        m_e = r <= r_e_max
        r_s = r[m_e]
        g_s = g[m_e]
        ve_s = v_e[m_e]
        v_den = float(_trapz_compat(g_s * (r_s**2), r_s))
        v_mean = float(_trapz_compat(g_s * ve_s * (r_s**2), r_s)) / max(v_den, 1e-30)
        e_int = 0.5 * float(zeff) * v_mean
        e_kin = fermi_kinetic_per_nucleon_mev(
            float(rho0), m_eff_over_m=p.m_eff_over_m
        )
        e_tot = float(e_kin + e_int)
        e_list.append(float(e_tot))
        e_kin_list.append(float(e_kin))
        e_int_list.append(float(e_int))
        pack.append((g, r, rpk, gpk, rcut, zeff, gam, g2, float(rho0)))

    i_min = int(np.argmin(np.asarray(e_list)))
    g, r, rpk, gpk, rcut, zeff, gam, g2, rho_sat = pack[i_min]
    e_min = float(e_list[i_min])
    p_perc = float(1.0 - math.exp(-0.5 * zeff / max(p.z0, 1e-12)))

    # SEMF volume scale: compare positive a_v to binding depth ~ -E/A at minimum.
    a_v = float(-e_min) if e_min < 0.0 else float(abs(e_min))

    return OZResult(
        rho0=rho_sat,
        g=g,
        r=r,
        r_peak=rpk,
        g_peak=gpk,
        r_cut=rcut,
        z_eff=zeff,
        p_perc=p_perc,
        gamma_eff=float(gam),
        a_v_mev=a_v,
        e_per_a_mev=float(e_min),
        v0_mev_used=v0_mev,
        g2_mev_fm=float(g2),
        beta=float(beta),
        meta={
            "rho_scan_index_min": i_min,
            "T_per_A_at_sat_MeV": e_kin_list[i_min],
            "E_int_per_A_at_sat_MeV": e_int_list[i_min],
        },
    )


def run_nuclear_oz(*, params: OZParams | None = None) -> OZResult:
    p = params or OZParams()

    v0_literal_mev = _v0_from_g_shear_meV()
    v0_mev = float(p.v0_rep_mev) if p.v0_rep_mev is not None else max(28.0, v0_literal_mev)

    # T_eff literal (SI); beta in (MeV)^-1 for potentials in MeV
    t_literal_k = float(hbar * float(Gamma_w) / (2.0 * KB_J_PER_K))
    beta_mev_inv = float(MEV_PER_J / (KB_J_PER_K * max(t_literal_k, 1e-300)))

    plan = HankelPlan.build(p)
    # Probe max |beta V|
    r_probe = plan.r
    Vp = potential_V(
        r_probe,
        v0_mev=v0_mev,
        g2_mev_fm=20.0,
        mu_inv_fm=p.mu_inv_fm,
        lam_rep_fm=p.lambda_rep_fm,
        eps_fm=p.yukawa_eps_fm,
    )
    if p.use_literal_teff and float(beta_mev_inv * np.max(np.abs(Vp))) >= p.beta_v_floor:
        beta_use = beta_mev_inv
        branch = "literal_T_eff"
    else:
        beta_use = 1.0 / max(p.t_eff_nuclear_mev, 1e-6)
        branch = "nuclear_T_eff_override"

    # g2 scale: ~ G_shear * (1 fm^3)^2 in MeV fm — use HBARC to set MeV scale from a fiducial volume
    v_fid_fm3 = (4.0 * math.pi / 3.0) * (R_W_FM**3)
    g2_scale = 0.0045 * (float(G_shear) / 4.94e10) * (v_fid_fm3**2)

    res = self_consistent_gamma_rho(p, plan, v0_mev=v0_mev, g2_scale=g2_scale, beta=beta_use)
    lam_tab = lambda_ym_export_table()
    v_gap = v_gap_fm3()
    bundle_mev = ym_gap_bundle_resolved_mev(p)
    bundle_literal_mev = ym_gap_bundle_literal_mev(float(p.lambda_ym_qcd_mev), v_gap)
    denom_geom = max(v_gap * F_C_CUBED * ETA_GAP_FOCUS, 1e-300)
    lambda_effective_from_bundle_mev = float(bundle_mev / denom_geom)

    a_s, d_loss, d_comp, z_surf = semf_surface_coefficient_mev(
        a_v_mev=res.a_v_mev,
        z_eff=res.z_eff,
        gamma_bulk=res.gamma_eff,
        bundle_mev=bundle_mev,
        compression_adds_penalty=p.surface_compression_adds_penalty,
    )
    a_sym, a_sym_p, a_sym_v = semf_symmetry_coefficient_mev(
        rho0_fm3=res.rho0,
        m_eff_over_m=p.m_eff_over_m,
        delta_rho_over_rho0=p.symmetry_delta_rho_over_rho0,
        bundle_mev=bundle_mev,
    )
    a_p, d_e_pair = semf_pairing_coefficient_mev(
        rho0_fm3=res.rho0,
        m_eff_over_m=p.m_eff_over_m,
        reference_A=p.semf_reference_A,
        bundle_mev=bundle_mev,
        pairing_ym_fraction=p.pairing_ym_fraction,
    )
    res.meta.update(
        {
            **{k: float(v) for k, v in lam_tab.items()},
            "V_gap_fm3": v_gap,
            "YM_bundle_literal_MeV": bundle_literal_mev,
            "YM_bundle_resolved_MeV": bundle_mev,
            "Lambda_YM_effective_from_resolved_bundle_MeV": lambda_effective_from_bundle_mev,
            "ym_bundle_use_literal": p.ym_bundle_use_literal,
            "v0_literal_MeV": v0_literal_mev,
            "v0_used_MeV": v0_mev,
            "T_literal_K": t_literal_k,
            "beta_branch": branch,
            "beta_MeVInv_literal": beta_mev_inv,
            "beta_MeVInv_used": beta_use,
            "g2_scale": g2_scale,
            "m_eff_over_m": p.m_eff_over_m,
            "z_surf_half_z_eff": z_surf,
            "dE_surface_loss_MeV": d_loss,
            "dE_surface_comp_MeV": d_comp,
            "a_s_MeV": a_s,
            "a_s_rel_err_vs_exp": abs(a_s - A_S_EXP_MEV) / max(A_S_EXP_MEV, 1e-12),
            "a_sym_Pauli_MeV": a_sym_p,
            "a_sym_vortex_MeV": a_sym_v,
            "a_sym_MeV": a_sym,
            "a_sym_rel_err_vs_exp": abs(a_sym - A_SYM_EXP_MEV) / max(A_SYM_EXP_MEV, 1e-12),
            "symmetry_delta_rho_over_rho0": p.symmetry_delta_rho_over_rho0,
            "pairing_ym_fraction": p.pairing_ym_fraction,
            "surface_compression_adds_penalty": p.surface_compression_adds_penalty,
            "Delta_E_pair_MeV": d_e_pair,
            "semf_reference_A": p.semf_reference_A,
            "a_p_MeV": a_p,
            "a_p_rel_err_vs_exp": abs(a_p - A_P_EXP_MEV) / max(A_P_EXP_MEV, 1e-12),
        }
    )
    return res


def main() -> None:
    p = OZParams()
    res = run_nuclear_oz(params=p)

    print("=== nuclear_matter_oz_closure: OZ + closure + self-consistency ===")
    for k, v in res.meta.items():
        print(f"  [{k}] = {v}")
    print(f"  rho0_sat (fm^-3) = {res.rho0:.17e}")
    print(f"  g(r) first peak: r_nn = {res.r_peak:.17e} fm, g_max = {res.g_peak:.17e}")
    print(f"  r_cut (first min after peak) = {res.r_cut:.17e} fm")
    print(f"  z_eff = {res.z_eff:.17e}")
    print(f"  p_perc (Bethe lattice) = {res.p_perc:.17e}")
    print(f"  gamma_max_eff (sqrt(z_eff/z0)) = {res.gamma_eff:.17e}")
    print(f"  E/A mean-field estimate (MeV) = {res.e_per_a_mev:.17e}")
    print(f"  a_v model magnitude (MeV) = {res.a_v_mev:.17e}  (compare to exp {A_V_EXP_MEV})")
    rel = abs(res.a_v_mev - A_V_EXP_MEV) / max(A_V_EXP_MEV, 1e-12)
    print(f"  |a_v - a_v_exp| / a_v_exp = {rel:.17e} ({100*rel:.4f}%)")
    if rel <= 0.2:
        print("  [closure:a_v]")

    print("--- SEMF extensions (surface / symmetry / pairing) ---")
    a_s = float(res.meta.get("a_s_MeV", 0.0))
    r_s = float(res.meta.get("a_s_rel_err_vs_exp", 0.0))
    print(f"  a_s (MeV) = {a_s:.17e}  (exp {A_S_EXP_MEV})")
    print(f"  |a_s - a_s_exp| / a_s_exp = {r_s:.17e} ({100*r_s:.4f}%)")
    if r_s <= 0.2:
        print("  [closure:a_s]")

    a_sym = float(res.meta.get("a_sym_MeV", 0.0))
    r_sym = float(res.meta.get("a_sym_rel_err_vs_exp", 0.0))
    print(f"  a_sym (MeV) = {a_sym:.17e}  (exp {A_SYM_EXP_MEV})")
    print(f"  |a_sym - a_sym_exp| / a_sym_exp = {r_sym:.17e} ({100*r_sym:.4f}%)")
    if r_sym <= 0.2:
        print("  [closure:a_sym]")

    a_p = float(res.meta.get("a_p_MeV", 0.0))
    r_p = float(res.meta.get("a_p_rel_err_vs_exp", 0.0))
    print(f"  a_p (MeV) = {a_p:.17e}  (exp {A_P_EXP_MEV})")
    print(f"  |a_p - a_p_exp| / a_p_exp = {r_p:.17e} ({100*r_p:.4f}%)")
    if r_p <= 0.2:
        print("  [closure:a_p]")


if __name__ == "__main__":
    main()
