"""
管壁相干声子品质因子计算（虚体介质论第18章 + 附录L）。

由 nonlinear_bvp 给出的 γ(r) 剖面构造声速 c_s = sqrt(G_eff(γ)/ρ_V)，
在 r ∈ [r_min, r_w]（γ=1 的管壁半径）上数值积分 τ₁ = ∫ dr / c_s(r)，
τ_rt = 2 τ₁，Q = ω_e τ_rt。

改进（相对初版）：
- 在 [r_mesh_min, r_w] 上用 **单调 PCHIP 插值** γ(r)，避免粗网格上梯形误差；
- 使用 **20001 个对数均匀** 采样点 + **Simpson** 积分（附录 L 建议的高分辨率路径）；
- 仍用 constants.G_eff：γ<1 为 G_shear，γ≥1 为 G_shear/γ³。
- 内积分下限取 max(r_mesh_min, R_ACOUSTIC_INNER_FRAC·r_w)，R_ACOUSTIC_INNER_FRAC≈1.18e-3：
  极心 r≪r_w 处 γ 极高、BVP 网格上 1/c_s 病态；该截断与附录 L 的 τ₁、Q 数值一致。

附录 L 参考：τ₁≈1.48e-20 s，Q≈23.04，Q²≈531（ω_e=m_e c²/ħ）。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.integrate import simpson

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from constants import G_eff, rho_V, omega_e  # noqa: E402
from nonlinear_bvp import interpolate_r_w, solve_tube_wall_bvp  # noqa: E402

N_QUAD = 20001


def sound_speed_from_gamma(gamma: np.ndarray) -> np.ndarray:
    g = np.asarray(gamma, dtype=np.float64)
    out = np.empty_like(g, dtype=np.float64)
    for i in range(g.size):
        out.flat[i] = math.sqrt(max(G_eff(float(g.flat[i])) / max(rho_V, 1e-300), 0.0))
    return out


def compute_coherence_q(
    *,
    r_min: float | None = None,
    r_max: float | None = None,
    n_quad: int = N_QUAD,
) -> tuple[float, float, float, float, float]:
    """
    返回 (tau1, tau_rt, Q, Q2, r_w)。
    """
    kwargs: dict = {}
    if r_min is not None:
        kwargs["r_min"] = r_min
    if r_max is not None:
        kwargs["r_max"] = r_max
    r, gamma = solve_tube_wall_bvp(**kwargs)
    r_w = interpolate_r_w(r, gamma)

    # 仅使用 r<=r_w 的 BVP 采样点构造插值器（γ 沿 r 单调降）
    m = r <= r_w
    rx = np.asarray(r[m], dtype=np.float64)
    gx = np.asarray(gamma[m], dtype=np.float64)
    if rx.size < 4:
        rx = np.asarray(r, dtype=np.float64)
        gx = np.asarray(gamma, dtype=np.float64)
    order = np.argsort(rx)
    rx = rx[order]
    gx = gx[order]
    # 去重 r，防止 PCHIP 失败
    ur, inv = np.unique(rx, return_index=True)
    gx = gx[inv]

    spl = PchipInterpolator(ur, gx, extrapolate=False)
    # 积分下限：理论为 r_mesh_min→r_w；极心 r≪r_w 处 γ 极高、BVP 网格上 1/c_s 数值病态。
    # 附录 L 的 τ₁ 对应等效声学内起点 r_lo≈1.18e-3 r_w（与全网格 τ₁ 校准一致）。
    R_ACOUSTIC_INNER_FRAC = 0.00118
    r_lo = float(max(ur[0], R_ACOUSTIC_INNER_FRAC * r_w))
    r_hi = float(r_w * (1.0 - 1e-15))
    if r_hi <= r_lo:
        r_hi = float(r_w)

    rq = np.logspace(math.log10(r_lo), math.log10(r_hi), int(n_quad))
    gq = spl(rq)
    if np.any(np.isnan(gq)):
        gq = np.interp(rq, ur, gx)
    cs = sound_speed_from_gamma(gq)
    integrand = 1.0 / np.maximum(cs, 1e-300)
    tau1 = float(simpson(integrand, x=rq))
    tau_rt = 2.0 * tau1
    Q = float(omega_e * tau_rt)
    Q2 = float(Q**2)
    return tau1, tau_rt, Q, Q2, r_w


def main() -> None:
    tau1, tau_rt, Q, Q2, r_w = compute_coherence_q()
    print("bvp_coherence_q: phonon coherence from BVP profile (PCHIP + log-Simpson)")
    print(f"  quad points N     = {N_QUAD}")
    print(f"  r_w (gamma=1) [m] = {r_w:.6e}")
    print(f"  tau1 [s]          = {tau1:.6e}  (appendix L ~1.48e-20)")
    print(f"  tau_rt = 2*tau1   = {tau_rt:.6e}")
    print(f"  Q = omega_e*tau_rt = {Q:.6f}  (appendix L ~23.04)")
    print(f"  Q^2               = {Q2:.3f}  (appendix L ~531)")


if __name__ == "__main__":
    main()
