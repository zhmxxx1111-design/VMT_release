"""
PMNS matrix elements from torus surface harmonics + wall-distortion flavor modes.

Numerical overlaps on the standard torus area element, SVD, and standard magnitude
extraction for mixing angles; Jarlskog-style phase combination for delta_CP.

``numpy.linalg.svd`` does not guarantee that the columns of ``U`` (hence of ``U^dagger``)
match PDG mass ordering. After forming ``PMNS = U^dagger``, columns are permuted with a
normal-hierarchy heuristic: the column with smallest ``|U_{e j}|`` is taken as the
mostly-``ν_3`` (atmospheric) column; among the remaining two, the larger ``|U_{e j}|`` is
``ν_1`` and the other ``ν_2``. Overlap columns stay in fixed ``(ψ_{0,0}, ψ_{1,0}, ψ_{0,1})``
order per the model definition.

Second-order geometric cross-couplings among breathing / bending / torsion distortions
are included in the flavor modes; each flavor wavefunction is L^2-normalized on the
torus patch (same 2D trapezoid grid as overlaps) before computing ``O_{\\alpha i}``.
"""
from __future__ import annotations

import cmath
import math
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from constants import Gamma_w  # noqa: E402

# --- 1. Torus geometry (user) ---
R_W_M = 6.15e-14
R_MAJOR_M = 3.86e-13
OMEGA_E = 7.76e20
ETA_FRUST = 1.0 - (OMEGA_E / Gamma_w)

# Masses (MeV)
M_E_MEV = 0.511
M_MU_MEV = 105.66
M_TAU_MEV = 1776.86

# --- 7. Experiment (angles in degrees; also sin reference) ---
THETA12_EXP_DEG = 34.0
THETA23_EXP_DEG = 45.0
THETA13_EXP_DEG = 8.5
DELTA_CP_EXP_RAD = math.radians(68.0)

SIN12_EXP = math.sin(math.radians(THETA12_EXP_DEG))
SIN23_EXP = math.sin(math.radians(THETA23_EXP_DEG))
SIN13_EXP = math.sin(math.radians(THETA13_EXP_DEG))

N_THETA = 500
N_PHI = 500
S13_FLOOR = 1e-6


def _circ_dist_rad(a: float, b: float) -> float:
    """Shortest arc length |a-b| on the circle, a,b in radians."""
    d = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return abs(float(d))


def _trapz1d(y: np.ndarray, x: np.ndarray) -> complex:
    if hasattr(np, "trapezoid"):
        return complex(np.trapezoid(y, x))
    return complex(np.trapz(y, x))  # type: ignore[attr-defined]


def _trapz2d(z: np.ndarray, x: np.ndarray, y: np.ndarray) -> complex:
    """2D trapezoidal rule: integrate z(x_i, y_j) over rectangle (tensor product grids)."""
    inner = np.zeros(z.shape[0], dtype=np.complex128)
    for i in range(z.shape[0]):
        inner[i] = _trapz1d(z[i, :], y)
    return _trapz1d(inner, x)


def norm_factor(m: int, n: int, r_w: float, r_big: float) -> float:
    if m == 0 and n == 0:
        return math.sqrt(2.0 * (math.pi**2) * r_w * r_big)
    return math.sqrt((math.pi**2) * r_w * r_big)


def mass_eigenstate(
    m: int,
    n: int,
    theta: np.ndarray,
    phi: np.ndarray,
    r_w: float,
    r_big: float,
) -> np.ndarray:
    """psi_{m,n}(theta, phi) on mesh theta[nt], phi[np] broadcast to (nt, np)."""
    th = theta[:, None]
    ph = phi[None, :]
    nmn = norm_factor(m, n, r_w, r_big)
    return (np.exp(1j * m * ph) * np.cos(n * th)) / nmn


def flavor_eigenstates(
    theta: np.ndarray,
    phi: np.ndarray,
    r_w: float,
    r_big: float,
    xi_e: float,
    xi_mu: float,
    xi_tau: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Second-order coupled breathing / bending / torsion modes on ``psi_{0,0}``."""
    psi00 = mass_eigenstate(0, 0, theta, phi, r_w, r_big)
    th = theta[:, None]
    ph = phi[None, :]
    ct = np.cos(th)
    sp = np.sin(ph)
    psi_e = (
        psi00
        + xi_e * ct * psi00
        + xi_e * xi_mu * 0.5 * ct * sp * psi00
        + xi_e * xi_tau * 0.3 * (ct**2) * sp * psi00
    )
    psi_mu = (
        psi00
        + xi_mu * sp * psi00
        + xi_mu * xi_e * 0.5 * sp * ct * psi00
        + xi_mu * xi_tau * 0.4 * sp * ct * sp * psi00
    )
    psi_tau = (
        psi00
        + xi_tau * ct * sp * psi00
        + xi_tau * xi_e * 0.3 * ct * sp * ct * psi00
        + xi_tau * xi_mu * 0.4 * ct * sp * sp * psi00
    )
    return psi_e, psi_mu, psi_tau


def _surface_l2_norm_sq(
    psi: np.ndarray,
    jacobian: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
) -> float:
    """∫ |psi|^2 dA on the torus patch (real scalar)."""
    z = (np.abs(psi) ** 2) * jacobian
    return float(np.real(_trapz2d(z, theta, phi)))


def build_overlap_matrix(
    r_w: float,
    r_big: float,
    *,
    n_theta: int = N_THETA,
    n_phi: int = N_PHI,
) -> tuple[np.ndarray, dict[str, object]]:
    """O_{alpha, i} = int psi_alpha^* psi_i dA; flavor modes L^2-normalized on the same grid."""
    theta = np.linspace(0.0, 2.0 * math.pi, n_theta, dtype=np.float64)
    phi = np.linspace(0.0, 2.0 * math.pi, n_phi, dtype=np.float64)

    th = theta[:, None]
    # dA = r_w * (R + r_w*cos(theta)) * dtheta * dphi
    jacobian = r_w * (r_big + r_w * np.cos(th))
    psi1 = mass_eigenstate(0, 0, theta, phi, r_w, r_big)
    psi2 = mass_eigenstate(1, 0, theta, phi, r_w, r_big)
    psi3 = mass_eigenstate(0, 1, theta, phi, r_w, r_big)

    xi_e = math.sqrt(max(ETA_FRUST, 0.0)) * math.sqrt(r_big / r_w)
    xi_mu = xi_e * (1.0 - math.sqrt(M_E_MEV / M_MU_MEV))
    xi_tau = xi_e * (1.0 - math.sqrt(M_E_MEV / M_TAU_MEV))

    pe, pmu, ptau = flavor_eigenstates(theta, phi, r_w, r_big, xi_e, xi_mu, xi_tau)
    norm2_before = (
        _surface_l2_norm_sq(pe, jacobian, theta, phi),
        _surface_l2_norm_sq(pmu, jacobian, theta, phi),
        _surface_l2_norm_sq(ptau, jacobian, theta, phi),
    )
    pe_n = pe / math.sqrt(max(norm2_before[0], 1e-300))
    pmu_n = pmu / math.sqrt(max(norm2_before[1], 1e-300))
    ptau_n = ptau / math.sqrt(max(norm2_before[2], 1e-300))
    norm2_after = (
        _surface_l2_norm_sq(pe_n, jacobian, theta, phi),
        _surface_l2_norm_sq(pmu_n, jacobian, theta, phi),
        _surface_l2_norm_sq(ptau_n, jacobian, theta, phi),
    )

    def overlap(pa: np.ndarray, pm: np.ndarray) -> complex:
        z = np.conjugate(pa) * pm * jacobian
        return _trapz2d(z, theta, phi)

    o = np.zeros((3, 3), dtype=np.complex128)
    flavors = (pe_n, pmu_n, ptau_n)
    masses = (psi1, psi2, psi3)
    for a in range(3):
        for i in range(3):
            o[a, i] = overlap(flavors[a], masses[i])
    diag: dict[str, object] = {
        "flavor_norm2_before": norm2_before,
        "flavor_norm2_after": norm2_after,
    }
    return o, diag


def _permute_pmns_columns_nh(pmns: np.ndarray) -> np.ndarray:
    """Reorder columns toward PDG-like NH: smallest |U_e| ~ heaviest; largest |U_e| ~ lightest."""
    u = np.asarray(pmns, dtype=np.complex128)
    abs_e = np.abs(u[0, :])
    j3 = int(np.argmin(abs_e))
    rem = [j for j in range(3) if j != j3]
    j_hi, j_lo = rem[0], rem[1]
    if abs_e[j_hi] < abs_e[j_lo]:
        j_hi, j_lo = j_lo, j_hi
    j1, j2 = j_hi, j_lo
    return u[:, (j1, j2, j3)]


def pmns_from_overlap(o: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """O = U @ diag(s) @ Vh -> PMNS = U^dagger (user convention), columns NH-permuted."""
    u, s, vh = np.linalg.svd(o, full_matrices=True)
    pmns = _permute_pmns_columns_nh(np.conjugate(u.T))
    return pmns, u, s


def mixing_parameters(pmns: np.ndarray) -> dict[str, float]:
    """Extract sin(theta_ij) and delta_CP per user (with s13 floor)."""
    u = pmns  # rows: flavor e, mu, tau; columns: mass 1, 2, 3
    ue1, ue2, ue3 = u[0, 0], u[0, 1], u[0, 2]
    umu1, umu3 = u[1, 0], u[1, 2]

    s13 = abs(ue3)
    if s13 < S13_FLOOR:
        s13 = S13_FLOOR
    c13_sq = max(1.0 - s13**2, 0.0)
    c13 = math.sqrt(c13_sq)

    s12 = abs(ue2) / c13
    s23 = abs(umu3) / c13

    th12 = math.asin(max(min(s12, 1.0), 0.0))
    th23 = math.asin(max(min(s23, 1.0), 0.0))
    th13 = math.asin(max(min(abs(u[0, 2]) if abs(u[0, 2]) >= S13_FLOOR else S13_FLOOR, 1.0), 0.0))

    c12, s12g = math.cos(th12), math.sin(th12)
    c23, s23g = math.cos(th23), math.sin(th23)

    num = ue1 * np.conjugate(ue3) * umu1 * np.conjugate(umu3)
    den = (c12 * s12g * c23 * (s23g**2) * s13 * (c13**2))
    if abs(den) < 1e-30:
        delta_cp = 0.0
    else:
        delta_cp = -cmath.phase(num / den)

    return {
        "sin_theta12": float(s12),
        "sin_theta23": float(s23),
        "sin_theta13": float(abs(u[0, 2]) if abs(u[0, 2]) >= S13_FLOOR else S13_FLOOR),
        "theta12_rad": float(th12),
        "theta23_rad": float(th23),
        "theta13_rad": float(th13),
        "delta_cp_rad": float(delta_cp),
    }


def run_pmns() -> dict:
    o, flavor_diag = build_overlap_matrix(R_W_M, R_MAJOR_M)
    pmns, u_svd, sig = pmns_from_overlap(o)
    pars = mixing_parameters(pmns)

    def ang_dev(calc_sin: float, exp_sin: float) -> float:
        return abs(calc_sin - exp_sin)

    def pct_vs_exp(dev: float, exp_sin: float) -> float:
        if abs(exp_sin) < 1e-30:
            return float("nan")
        return 100.0 * dev / abs(exp_sin)

    out: dict = {
        "eta_frust": ETA_FRUST,
        "overlap_matrix": o,
        "pmns": pmns,
        "singular_values": sig,
        "mixing": pars,
        "flavor_norm2_before": flavor_diag["flavor_norm2_before"],
        "flavor_norm2_after": flavor_diag["flavor_norm2_after"],
        "dev_sin12": ang_dev(pars["sin_theta12"], SIN12_EXP),
        "dev_sin23": ang_dev(pars["sin_theta23"], SIN23_EXP),
        "dev_sin13": ang_dev(pars["sin_theta13"], SIN13_EXP),
        "dev_delta_cp_rad": abs(pars["delta_cp_rad"] - DELTA_CP_EXP_RAD),
        "dev_delta_cp_deg": abs(math.degrees(pars["delta_cp_rad"]) - math.degrees(DELTA_CP_EXP_RAD)),
        "dev_delta_cp_circ_rad": _circ_dist_rad(pars["delta_cp_rad"], DELTA_CP_EXP_RAD),
        "dev_delta_cp_circ_deg": math.degrees(
            _circ_dist_rad(pars["delta_cp_rad"], DELTA_CP_EXP_RAD)
        ),
    }
    out["pct_dev_sin12"] = pct_vs_exp(out["dev_sin12"], SIN12_EXP)
    out["pct_dev_sin23"] = pct_vs_exp(out["dev_sin23"], SIN23_EXP)
    out["pct_dev_sin13"] = pct_vs_exp(out["dev_sin13"], SIN13_EXP)

    # Relative deviation < 20% vs experimental sin (same as |calc-exp| <= 0.2 * exp_sin)
    out["within_20pct_sin12"] = out["dev_sin12"] <= 0.2 * max(SIN12_EXP, 1e-12)
    out["within_20pct_sin23"] = out["dev_sin23"] <= 0.2 * max(SIN23_EXP, 1e-12)
    out["within_20pct_sin13"] = out["dev_sin13"] <= 0.2 * max(SIN13_EXP, 1e-12)
    out["closure_pmns_angles"] = (
        out["within_20pct_sin12"] and out["within_20pct_sin23"] and out["within_20pct_sin13"]
    )
    out["closure_pmns_cp"] = out["dev_delta_cp_circ_deg"] < 30.0
    return out


def main() -> None:
    r = run_pmns()
    m = r["mixing"]
    nb = r["flavor_norm2_before"]
    na = r["flavor_norm2_after"]

    print("=== pmns_torus_harmonics ===")
    print(f"  r_w = {R_W_M} m, R = {R_MAJOR_M} m")
    print(f"  eta_frust = 1 - omega_e/Gamma_w = {r['eta_frust']:.12e}")
    print("  --- L2 on surface: int |psi_alpha|^2 dA (before / after normalization) ---")
    print(f"    e:   {nb[0]:.12e}  /  {na[0]:.12e}")
    print(f"    mu:  {nb[1]:.12e}  /  {na[1]:.12e}")
    print(f"    tau: {nb[2]:.12e}  /  {na[2]:.12e}")
    print(
        "    Note: pre-norm L2 ~ 1 only for weak distortion; large xi from R/r_w "
        "raises int|psi|^2 dA before dividing by sqrt(...)."
    )
    print(f"  overlap O (3x3) real/imag max = {np.max(np.abs(r['overlap_matrix'].real)):.6e} / {np.max(np.abs(r['overlap_matrix'].imag)):.6e}")
    print("  singular values:", r["singular_values"])
    print("  PMNS = U^dagger (NH column order, rows e, mu, tau):")
    print(r["pmns"])
    print("  --- mixing angles (sin): calculated vs experiment ---")
    print(
        f"    sin(theta12): {m['sin_theta12']:.8f}  vs  {SIN12_EXP:.8f}  "
        f"(abs diff = {r['dev_sin12']:.8e},  {r['pct_dev_sin12']:.4f}% of exp)"
    )
    print(
        f"    sin(theta23): {m['sin_theta23']:.8f}  vs  {SIN23_EXP:.8f}  "
        f"(abs diff = {r['dev_sin23']:.8e},  {r['pct_dev_sin23']:.4f}% of exp)"
    )
    print(
        f"    sin(theta13): {m['sin_theta13']:.8f}  vs  {SIN13_EXP:.8f}  "
        f"(abs diff = {r['dev_sin13']:.8e},  {r['pct_dev_sin13']:.4f}% of exp)"
    )
    print(
        "    theta12, theta23, theta13 (deg):",
        math.degrees(m["theta12_rad"]),
        math.degrees(m["theta23_rad"]),
        math.degrees(m["theta13_rad"]),
    )
    print("  --- delta_CP ---")
    print(
        f"    calculated = {m['delta_cp_rad']:.8f} rad  ({math.degrees(m['delta_cp_rad']):.4f} deg)"
    )
    print(f"    experiment = {DELTA_CP_EXP_RAD:.8f} rad  ({math.degrees(DELTA_CP_EXP_RAD):.4f} deg)")
    print(
        f"    abs diff (linear deg) = {r['dev_delta_cp_deg']:.6f}  "
        f"circular shortest arc (deg) = {r['dev_delta_cp_circ_deg']:.6f}"
    )
    if r["closure_pmns_angles"]:
        print("  [closure:PMNS_angles]")
    else:
        print(
            "  (angles not within 20% of experiment on sin mixing; "
            f"pct deviations: sin12 {r['pct_dev_sin12']:.2f}%, "
            f"sin23 {r['pct_dev_sin23']:.2f}%, sin13 {r['pct_dev_sin13']:.2f}%)"
        )
    if r["closure_pmns_cp"]:
        print("  [closure:PMNS_CP]")
    else:
        print(
            "  (delta_CP circular deviation not < 30 deg; "
            f"current circular abs diff = {r['dev_delta_cp_circ_deg']:.4f} deg)"
        )


if __name__ == "__main__":
    main()
