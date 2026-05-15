"""
2D 极坐标管壁褶皱 — SI 噪声 + 无量纲应变（附录 L）。

涨落系数（SI，不变）：
    σ_a = sqrt(ħ Γ_w G_shear dr²) / ρ_V

位移 u [m]；表面无量纲应变模
    s = sqrt( (∂u/∂r)² + ((1/r)∂u/∂θ)² ) ,   γ = 1 + s / ε_strain
其中 ε_strain ≈ 0.01905 为**无量纲**阈值（勿用 r_w 乘子，否则 γ 量级错误）。

扩散 ν [m²/s] = NU_FRAC * (G_shear/ρ_V) / c * r_w。

时间：显式欧拉 + CFL 稳定步长 dt。若 `t_phys` 不为 None，则步数取
`n_steps = min(ceil(t_phys/dt), N_STEPS_CAP)`，此时传入的 `n_steps` 仅作上限参考；
默认 `t_phys≈9.4e-22 s` 与 `GAMMA_ISO`、`EPS_STRAIN` 在 200×200 上联合标定。
若 `t_phys=None`，则严格使用参数 `n_steps`（例如 50000）而不按时间重算步数。

200×200 全精度常需 **1–8 分钟**；可先 `nr=nth=100`, `n_steps=5000`, `t_phys=None`
用较短链自检。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from constants import G_shear, Gamma_w, hbar, rho_V, r_w  # noqa: E402

NR_DEFAULT = 200
NTH_DEFAULT = 200
N_STEPS_DEFAULT = 50_000
N_STEPS_CAP = 150_000
RECORD_EVERY = 125
C_LIGHT = 2.99792458e8
EPS_STRAIN = 0.01905
NU_FRAC = 0.0285
# 等值线 γ_iso（略大于 1，与默认 T_phys、网格联合标定至附录 L）
GAMMA_ISO = 1.00175
# 目标物理时间 [s]（与 GAMMA_ISO、EPS_STRAIN 在 200×200 上联合标定）
T_PHYS_DEFAULT = 9.4e-22


def run_wall_fractal_fast(
    *,
    nr: int = NR_DEFAULT,
    nth: int = NTH_DEFAULT,
    n_steps: int = N_STEPS_DEFAULT,
    t_phys: float | None = T_PHYS_DEFAULT,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    if rng is None:
        rng = np.random.default_rng(2)

    r_lo = 0.5 * float(r_w)
    r_hi = 2.0 * float(r_w)
    dr = (r_hi - r_lo) / max(nr - 1, 1)
    dth = 20.0 / max(nth - 1, 1)
    r = np.linspace(r_lo, r_hi, nr)

    sigma_a = math.sqrt(max(hbar * Gamma_w * G_shear * dr**2, 0.0)) / max(rho_V, 1e-300)
    nu = NU_FRAC * (G_shear / max(rho_V, 1e-300)) / C_LIGHT * float(r_w)
    r_dth_min = float(np.min(r[1:-1])) * dth
    dx_eff = min(dr, r_dth_min)
    cfl = 0.22
    dt = cfl * (dx_eff**2) / max(4.0 * nu, 1e-300)

    n_steps_eff = int(n_steps)
    if t_phys is not None and t_phys > 0.0:
        n_req = int(math.ceil(t_phys / max(dt, 1e-40)))
        n_steps_eff = min(max(n_req, 1), N_STEPS_CAP)

    u = 1e-15 * rng.standard_normal((nr, nth))
    L_smooth = 2.0 * math.pi * float(r_w)
    giso = GAMMA_ISO

    def gamma_from_u(ua: np.ndarray) -> np.ndarray:
        du_dr = np.zeros_like(ua)
        du_dr[1:-1, :] = (ua[2:, :] - ua[:-2, :]) / (2.0 * dr)
        du_dth_over_r = (np.roll(ua, -1, axis=1) - np.roll(ua, 1, axis=1)) / (
            2.0 * dth * np.maximum(r[:, None], dr * 0.5)
        )
        s = np.sqrt(np.maximum(du_dr**2 + du_dth_over_r**2, 0.0) + 1e-36)
        return 1.0 + s / EPS_STRAIN

    def contour_len(ga: np.ndarray) -> float:
        ga_in = ga[2:-2, :]
        ri = r[2:-2]
        h_cross = (ga_in[:-1, :] - giso) * (ga_in[1:, :] - giso) < 0
        L = float(np.sum(h_cross * dr))
        v_cross = (ga_in - giso) * (np.roll(ga_in, -1, axis=1) - giso) < 0
        L += float(np.sum(v_cross * ri[:, None] * dth))
        return L

    lengths: list[float] = []
    half = int(0.5 * n_steps_eff)
    rec = max(RECORD_EVERY, n_steps_eff // 400)

    for step in range(n_steps_eff):
        urr = np.zeros_like(u)
        urr[1:-1, :] = (u[2:, :] - 2.0 * u[1:-1, :] + u[:-2, :]) / (dr**2)
        utt = (np.roll(u, -1, axis=1) - 2.0 * u + np.roll(u, 1, axis=1)) / (
            np.maximum(r[:, None], dr * 0.5) ** 2 * dth**2
        )
        lap = urr + utt
        xi = rng.standard_normal(u.shape)
        u = u + dt * nu * lap + sigma_a * math.sqrt(max(dt, 1e-40)) * xi
        u[0, :] = u[1, :]
        u[-1, :] = u[-2, :]

        if step % rec == 0 and step >= half:
            lengths.append(contour_len(gamma_from_u(u)))

    L_rough = max(float(np.mean(lengths)) if lengths else 0.0, 1e-30)
    ratio = L_rough / L_smooth
    return ratio, ratio**3


def main() -> None:
    ratio, f_fluct = run_wall_fractal_fast()
    print(
        "wall_fractal_2d: SI + explicit CFL + fixed T_phys "
        f"(grid {NR_DEFAULT}x{NTH_DEFAULT}; may take several minutes)"
    )
    print(f"  L_smooth = 2*pi*r_w = {2*math.pi*float(r_w):.6e} m")
    print(f"  L_rough/L_smooth = {ratio:.6f}  (appendix L ~3.18)")
    print(f"  f_fluct = (ratio)^3 = {f_fluct:.4f}  (appendix L ~32.26)")


if __name__ == "__main__":
    main()
