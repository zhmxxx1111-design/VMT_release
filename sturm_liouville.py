"""
声子 Sturm–Liouville 特征值问题数值求解器（虚体介质论第18章 + 附录L）。

在 r∈[r_min, r_w] 上用有限差分近似 -u'' ≈ λ u（均匀介质、无量纲刚度），
比较 Dirichlet 与 Neumann 外边界，取最小正特征值平方根作为 ω₀ 的代理。

局限（诚实标注）：未包含变密度 M(r)、非均匀 K(r) 或完整极坐标算符；
与 Γ_w^(eff) 的比值仅为数量级对照，不可替代严格谱理论。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from constants import G_shear, rho_V  # noqa: E402
from nonlinear_bvp import interpolate_r_w, solve_tube_wall_bvp  # noqa: E402

GAMMA_W_EFF = 4.17e20  # Hz，题设对照频率


def _laplacian_1d_dirichlet(n: int, dr: float) -> np.ndarray:
    """-d²/dr² 在 n 个内点上的 Dirichlet-Dirichlet 差分矩阵 (n×n)。"""
    a = np.zeros((n, n))
    c0 = 1.0 / (dr**2)
    for i in range(n):
        a[i, i] = 2.0 * c0
        if i > 0:
            a[i, i - 1] = -c0
        if i < n - 1:
            a[i, i + 1] = -c0
    return a


def _laplacian_1d_neumann_last(n: int, dr: float) -> np.ndarray:
    """内点 Dirichlet 在 r_min 端等效为 u[0]=0；外端 Neumann u'(r_w)=0 修改最后一行。"""
    a = _laplacian_1d_dirichlet(n, dr)
    c0 = 1.0 / (dr**2)
    a[-1, -1] = c0  # ghost: u_{N+1}=u_N => second diff coeff change
    a[-1, -2] = -c0
    return a


def smallest_eig_omega(a: np.ndarray, c_sound: float) -> float:
    w, v = np.linalg.eigh(a)
    lam = float(np.min(w[w > 1e-30]))
    return float(c_sound * math.sqrt(lam))


def main() -> None:
    r, gamma = solve_tube_wall_bvp()
    r_w = interpolate_r_w(r, gamma)
    r_min = float(r[0])
    n = 400
    dr = (r_w - r_min) / (n + 1)
    r_int = r_min + dr * np.arange(1, n + 1)
    _ = r_int
    c_sound = math.sqrt(max(G_shear / max(rho_V, 1e-300), 0.0))

    a_d = _laplacian_1d_dirichlet(n, dr)
    a_n = _laplacian_1d_neumann_last(n, dr)
    om_d = smallest_eig_omega(a_d, c_sound)
    om_n = smallest_eig_omega(a_n, c_sound)
    omega0 = min(om_d, om_n)
    ratio = omega0 / GAMMA_W_EFF

    print("sturm_liouville: 1D Laplacian phonon mode proxy")
    print(f"  r_min, r_w [m]      = {r_min:.3e}, {r_w:.3e}")
    print(f"  c_sound [m/s]     = {c_sound:.3e}")
    print(f"  omega0 (Dirichlet outer guess) ~ {om_d:.6e} Hz")
    print(f"  omega0 (Neumann outer guess)     ~ {om_n:.6e} Hz")
    print(f"  omega0 (min of above)            = {omega0:.6e} Hz")
    print(f"  Gamma_w^(eff)                    = {GAMMA_W_EFF:.6e} Hz")
    print(f"  omega0 / Gamma_w^(eff)           = {ratio:.6e}")


if __name__ == "__main__":
    main()
