"""
三代轻子：用 nonlinear_bvp 的管壁 BVP 得到 γ(r)，再在 γ≥1 球对称区域积分弹性能密度估计质量。

    M = ∫_{γ≥1} τ(γ) · γ dV ,  dV = 4π r² dr

τ(γ) 与 constants.tau 一致；积分区间取 r ∈ [r_min, r_w]，其中 r_w 为 γ=1 的管壁半径（γ 随 r 增大而减小）。
结果由 J 除以 MEV_J 得到 MeV。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_bvp

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from constants import c, hbar, m_e, tau  # noqa: E402
from nonlinear_bvp import (  # noqa: E402
    N_MESH,
    R_MAX_M,
    R_MIN_M,
    interpolate_r_w,
    ode_fun,
    solve_tube_wall_bvp,
)

MEV_J = 1.602_176_634e-13

M_E_EXP_MEV = 0.510_998_946_1
M_MU_EXP_MEV = 105.658_374_5
M_TAU_EXP_MEV = 1776.86


def mass_kg_from_mev(m_mev: float) -> float:
    return float(m_mev * MEV_J / (c * c))


def rescale_profile(
    r_ref: np.ndarray,
    g_ref: np.ndarray,
    *,
    gamma_max_target: float,
    r_w_target: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    """与 chern_simons_torus 相同：径向/振幅重标使 γ_max、r_w 对齐目标。"""
    rw_ref = float(interpolate_r_w(r_ref, g_ref))
    if rw_ref <= 0 or r_w_target <= 0:
        return r_ref.copy(), g_ref.copy(), "interpolate_r_w failed"

    fac_r = rw_ref / r_w_target
    r_out = r_ref / fac_r
    g_zoom = np.asarray(g_ref, dtype=np.float64).copy()
    peak = float(np.max(g_zoom))
    denom = max(peak - 1.0, 1e-15)
    s_amp = float(max(gamma_max_target - 1.0, 0.0)) / denom
    g_out = 1.0 + s_amp * (g_zoom - 1.0)
    note = f"fac_r={fac_r:.4e}, s_amp={s_amp:.4e}, peak_in={peak:.4f}, peak_out={float(np.max(g_out)):.4f}"
    return r_out, g_out, note


def solve_bvp_with_mass(
    m_kg: float,
    *,
    r_min: float,
    r_max: float,
    n_mesh: int,
    y_init: np.ndarray | None = None,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, str]:
    """一阶 BVP：右端 γ(r_max)=ħ/(2π m c r_max)；可选初值 y_init shape (1, n)。"""
    g_rmax = float(hbar / (2.0 * math.pi * m_kg * c * r_max))

    def bc(ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
        return np.array([yb[0] - g_rmax])

    npt = int(max(100, min(int(n_mesh), 10_000)))
    r = np.linspace(r_min, r_max, npt)
    if y_init is None:
        g_hi = max(50.0, 10.0 * max(g_rmax, 1e-6))
        y_a = np.linspace(g_hi, g_rmax, r.size)[None, :].astype(np.float64)
    else:
        y_a = np.asarray(y_init, dtype=np.float64)
        if y_a.ndim == 1:
            y_a = y_a[None, :]
        if y_a.shape[1] != r.size:
            return np.array([]), np.array([]), "y_init length mismatch r grid"

    sol = solve_bvp(
        ode_fun,
        bc,
        r,
        y_a,
        max_nodes=max(50_000, n_mesh * 5),
        tol=tol,
    )
    if not sol.success:
        return np.array([]), np.array([]), str(sol.message)

    r_fine = np.linspace(r_min, r_max, min(n_mesh, 10_000))
    y_fine = sol.sol(r_fine)[0]
    return r_fine, y_fine, "ok"


def elastic_mass_mev(
    r: np.ndarray,
    gamma: np.ndarray,
    *,
    r_w: float,
) -> float:
    """M = ∫ τ(γ)·γ dV，积分在 γ≥1 且 r≤r_w 的管壁内侧（球对称 dV = 4π r² dr）。"""
    rr = np.asarray(r, dtype=np.float64)
    gg = np.asarray(gamma, dtype=np.float64)
    order = np.argsort(rr)
    rr = rr[order]
    gg = gg[order]
    mask = (rr <= float(r_w) + 1e-15) & (gg >= 1.0 - 1e-9)
    if int(np.count_nonzero(mask)) < 2:
        return float("nan")
    rs = rr[mask]
    gs = gg[mask]
    # τ(γ) 与 constants.tau 一致；被积式再乘 γ（任务给定）
    density = np.array([float(tau(float(g))) * float(g) for g in gs])
    integrand = density * 4.0 * math.pi * rs * rs
    energy_j = float(np.trapezoid(integrand, rs))
    return float(energy_j / MEV_J)


LEPTONS = [
    {
        "name": "electron (e)",
        "gamma_max": 7.72,
        "r_w_m": 6.15e-14,
        "m_exp_mev": M_E_EXP_MEV,
        "m_kg_bc": float(m_e),
    },
    {
        "name": "muon (mu)",
        "gamma_max": 23.2,
        "r_w_m": float(hbar / (mass_kg_from_mev(M_MU_EXP_MEV) * c)),
        "m_exp_mev": M_MU_EXP_MEV,
        "m_kg_bc": mass_kg_from_mev(M_MU_EXP_MEV),
    },
    {
        "name": "tau (tau)",
        "gamma_max": 38.7,
        "r_w_m": float(hbar / (mass_kg_from_mev(M_TAU_EXP_MEV) * c)),
        "m_exp_mev": M_TAU_EXP_MEV,
        "m_kg_bc": mass_kg_from_mev(M_TAU_EXP_MEV),
    },
]


def _try_reference_bvp() -> tuple[np.ndarray, np.ndarray, str]:
    """优先使用 nonlinear_bvp 默认求解；失败则扫 r_min/r_max/n_mesh。"""
    attempts: list[tuple[float, float, int]] = []
    for n_mesh in (N_MESH, 8000, 10_000):
        for r_max in (R_MAX_M, 5e-12, 2e-12, 1e-11):
            for r_min in (R_MIN_M, 1e-17, 1e-15):
                attempts.append((r_min, r_max, n_mesh))
    last_err = "no attempts"
    for r_min, r_max, n_mesh in attempts:
        try:
            r, g = solve_tube_wall_bvp(r_min=r_min, r_max=r_max, n_mesh=n_mesh)
            return r, g, f"default solve_tube_wall_bvp ok (r_min={r_min:.3e}, r_max={r_max:.3e}, n={n_mesh})"
        except RuntimeError as e:
            last_err = str(e)
            continue
    return np.array([]), np.array([]), last_err


def _bvp_from_scaled_guess(
    m_kg: float,
    r_prev: np.ndarray,
    g_prev: np.ndarray,
    *,
    r_min: float,
    r_max: float,
    n_mesh: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    """把上一代的 γ(r) 插值到当前 r 网格作 solve_bvp 初值。"""
    npt = int(max(100, min(int(n_mesh), 10_000)))
    r = np.linspace(r_min, r_max, npt)
    g_rmax = float(hbar / (2.0 * math.pi * m_kg * c * r_max))
    if r_prev.size < 2:
        return np.array([]), np.array([]), "empty prev profile"
    y_init = np.interp(r, r_prev, g_prev, left=float(np.max(g_prev)), right=g_rmax)
    y_init = np.clip(y_init, g_rmax * 1.0001, max(np.max(g_prev) * 1.2, 10.0))
    y_init[0] = max(y_init[0], y_init[1] * 1.001)
    y_init[-1] = g_rmax
    return solve_bvp_with_mass(m_kg, r_min=r_min, r_max=r_max, n_mesh=n_mesh, y_init=y_init[None, :])


def main() -> None:
    print("lepton_bvp_mass: BVP gamma(r) -> M = integral tau(gamma)*gamma dV (gamma>=1, r<=r_w)")
    print(f"  MEV_J = {MEV_J:.12e} J/MeV")

    r_ref, g_ref, msg = _try_reference_bvp()
    profiles: list[tuple[np.ndarray, np.ndarray, float, float, str, str]] = []

    if r_ref.size == 0:
        print(f"  ERROR: reference BVP failed after retries: {msg}")
        sys.exit(1)
    print(f"  reference BVP: {msg}")

    prev_r, prev_g = r_ref.copy(), g_ref.copy()

    for row in LEPTONS:
        name = str(row["name"])
        gmx_t = float(row["gamma_max"])
        rw_t = float(row["r_w_m"])
        m_kg = float(row["m_kg_bc"])

        r_s, g_s, note = rescale_profile(r_ref, g_ref, gamma_max_target=gmx_t, r_w_target=rw_t)
        rw_m = float(interpolate_r_w(r_s, g_s))
        gmx_m = float(np.max(g_s))
        M_pred = elastic_mass_mev(r_s, g_s, r_w=rw_m)
        ok = math.isfinite(M_pred)
        fail_reason = ""
        if not ok or rw_m <= 0:
            ok = False
            fail_reason = "integral or r_w invalid after rescale; trying per-generation BVP with scaled guess"

        if not ok:
            print(f"\n--- {name} ---")
            print(f"  {fail_reason}")
            solved = False
            for r_min, r_max, n_mesh in (
                (R_MIN_M, R_MAX_M, 10_000),
                (1e-17, 5e-12, 10_000),
                (1e-16, 2e-12, 10_000),
            ):
                r2, g2, m2 = _bvp_from_scaled_guess(m_kg, prev_r, prev_g, r_min=r_min, r_max=r_max, n_mesh=n_mesh)
                if r2.size == 0:
                    print(f"    BVP guess retry failed: {m2} (r_min={r_min:.3e}, r_max={r_max:.3e})")
                    continue
                r_s, g_s, note = rescale_profile(r2, g2, gamma_max_target=gmx_t, r_w_target=rw_t)
                rw_m = float(interpolate_r_w(r_s, g_s))
                M_pred = elastic_mass_mev(r_s, g_s, r_w=rw_m)
                if math.isfinite(M_pred) and rw_m > 0:
                    solved = True
                    print(f"    recovered with per-gen BVP + rescale ({note})")
                    break
            if not solved:
                profiles.append((np.array([]), np.array([]), float("nan"), float("nan"), name, "failed"))
                prev_r, prev_g = r_ref, g_ref
                continue

        profiles.append((r_s, g_s, rw_m, gmx_m, name, note))
        prev_r, prev_g = r_s.copy(), g_s.copy()

    print("\n" + "=" * 88)
    print(f"{'species':<14} {'M_pred (MeV)':<16} {'M_exp (MeV)':<14} {'rel_err':<12} {'r_w (m)':<12} {'gamma_max':<10}")
    print("=" * 88)

    preds: list[float] = []
    rels: list[float] = []
    oks: list[bool] = []

    for i, row in enumerate(LEPTONS):
        name = row["name"]
        m_exp = float(row["m_exp_mev"])
        if i >= len(profiles):
            break
        r_s, g_s, rw_m, gmx_m, tag, note = profiles[i]
        if r_s.size == 0:
            print(f"{name:<14} {'FAILED':<16} {m_exp:<14.4f} {'n/a':<12} {'n/a':<12} {'n/a':<10}")
            preds.append(float("nan"))
            rels.append(float("inf"))
            oks.append(False)
            print(f"  reason: BVP/rescale path did not converge for this species.")
            continue
        M_pred = elastic_mass_mev(r_s, g_s, r_w=rw_m)
        preds.append(M_pred)
        rel = abs(M_pred - m_exp) / max(m_exp, 1e-30)
        rels.append(rel)
        oks.append(rel <= 0.5 and math.isfinite(M_pred))
        mp_s = f"{M_pred:.6e}" if abs(M_pred) < 0.01 or abs(M_pred) > 1e6 else f"{M_pred:.6f}"
        print(f"{name:<14} {mp_s:<14} {m_exp:<14.4f} {rel:<10.4e} {rw_m:<12.5e} {gmx_m:<10.4f}")
        print(f"  rescale: {note}")

    print("=" * 88)
    print(
        "Note: M_pred from integral tau(gamma)*gamma*dV (Pa*m^3 -> J -> MeV) is typically "
        "many orders below m_exp; this script uses the stated formula without extra factors."
    )
    all50 = all(ok for ok in oks) and len(oks) == 3
    print(f"Verdict (three generations rel_err <= 50%): {'YES' if all50 else 'NO'}")


if __name__ == "__main__":
    main()
