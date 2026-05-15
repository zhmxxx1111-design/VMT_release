"""
管壁过渡区非线性边值问题求解器（虚体介质论第18章 + 附录L）。

一阶 ODE：dγ/dr = -2τ(γ) / (r·τ'(γ))，τ(γ) 由 constants.tau 给出。
右边界 r_max 处 Dirichlet 固定 γ；左端 r_min 为区间端点（一阶方程仅需一端代数约束）。

说明：一阶标量 BVP 在数学上由单点边界条件定解；左端「对称性」在完整球几何中
常对应 Neumann 型约束，此处不强行施加第二 Dirichlet，以免过约束。
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

from constants import G_shear, c, hbar, m_e, tau  # noqa: E402

R_MIN_M = 1e-16
R_MAX_M = 1e-11
N_MESH = 5000
TOL_BVP = 1e-8


def tau_prime(gamma: float) -> float:
    """dτ/dγ，与 constants.tau 分段一致。"""
    g = float(gamma)
    if g < 1.0:
        return float(G_shear)
    return float(G_shear * ((1.0 + 4.0 * (g - 1.0) ** 2) + g * 8.0 * (g - 1.0)))


def ode_fun(r: np.ndarray, y: np.ndarray) -> np.ndarray:
    g = np.clip(y[0], 1e-14, 1e8)
    out = np.zeros_like(y)
    for i in range(r.size):
        gi = float(g[i])
        ri = float(max(r[i], 1e-30))
        ta = float(tau(gi))
        tp = tau_prime(gi)
        tp_safe = float(math.copysign(max(abs(tp), 1e-30), tp if tp != 0 else 1.0))
        out[0, i] = -2.0 * ta / (ri * tp_safe)
    return out


def bc_fun(ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
    """右端 Dirichlet：γ(r_max) = ħ/(2π m_e c r_max)。"""
    g_rmax = float(hbar / (2.0 * math.pi * m_e * c * R_MAX_M))
    return np.array([yb[0] - g_rmax])


def solve_tube_wall_bvp(
    *,
    r_min: float = R_MIN_M,
    r_max: float = R_MAX_M,
    n_mesh: int = N_MESH,
    tol: float = TOL_BVP,
) -> tuple[np.ndarray, np.ndarray]:
    """
    返回 (r, gamma) 数值解（r 升序）。
    """
    g_rmax = float(hbar / (2.0 * math.pi * m_e * c * r_max))
    r = np.linspace(r_min, r_max, max(100, min(n_mesh, 10_000)))
    # γ(r) 单调降（dγ/dr<0）：r_min 端取较大初值，r_max 端贴近 Dirichlet 值 g_rmax
    g_hi = max(50.0, 10.0 * max(g_rmax, 1e-6))
    y_init = np.linspace(g_hi, g_rmax, r.size)[None, :]

    sol = solve_bvp(
        ode_fun,
        bc_fun,
        r,
        y_init,
        max_nodes=max(50_000, n_mesh * 5),
        tol=tol,
    )
    if not sol.success:
        raise RuntimeError(f"solve_bvp failed: {sol.message}")

    r_fine = np.linspace(r_min, r_max, min(n_mesh, 5000))
    y_fine = sol.sol(r_fine)[0]
    return r_fine, y_fine


def interpolate_r_w(r: np.ndarray, gamma: np.ndarray) -> float:
    """γ=1 的径向位置（γ 沿 r 增大而减小时，找首次由 ≥1 跨到 <1）。"""
    g = np.asarray(gamma, dtype=float)
    rr = np.asarray(r, dtype=float)
    for i in range(1, rr.size):
        if g[i - 1] >= 1.0 and g[i] < 1.0:
            g0, g1 = g[i - 1], g[i]
            r0, r1 = rr[i - 1], rr[i]
            t = (1.0 - g0) / (g1 - g0)
            return float(r0 + t * (r1 - r0))
    return float(rr[np.argmin(np.abs(g - 1.0))])


def main() -> None:
    r, gamma = solve_tube_wall_bvp()
    r_w = interpolate_r_w(r, gamma)
    gamma_max = float(np.max(gamma))
    wkb_amp = float(gamma_max ** 0.75)

    print("nonlinear_bvp: tube-wall transition (solve_bvp)")
    print(f"  r_min, r_max [m] = {R_MIN_M:.3e}, {R_MAX_M:.3e}")
    print(f"  mesh points = {r.size}, tol = {TOL_BVP:.1e}")
    print(f"  r_w (gamma=1) [m]     = {r_w:.6e}")
    print(f"  gamma_max (core)    = {gamma_max:.6f}")
    print(f"  WKB amplitude ~ gamma^(3/4) = {wkb_amp:.6f}")


if __name__ == "__main__":
    main()
