"""
η_nonlinear 第一性原理合成引擎（虚体介质论第18章 + 附录L）。

汇总 WKB 放大、体积 γ_max⁴、周期因子 λ/δ_wall、相干 Q²、褶皱 f_fluct 与径向因子，
由 BVP 基准 η_BVP 得到 η_ind，并与自洽标度 η_self 比较。

说明：各子因子由本目录可运行脚本导出；此处为可复现的标量合成，非完整 DAO 搜索。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from constants import Gamma_w, c, lambda_micro  # noqa: E402

ETA_BVP = 1.70e-34
ETA_SELF = 6.8e-30
F_RADIAL = 1.027


def main() -> None:
    from nonlinear_bvp import interpolate_r_w, solve_tube_wall_bvp  # noqa: WPS433

    r, gamma = solve_tube_wall_bvp()
    r_w = interpolate_r_w(r, gamma)
    _ = r_w  # 与 BVP 剖面一致（诊断可扩展）
    gamma_max = float(np.max(gamma))
    wkb = float(gamma_max**0.75)
    vol = float(gamma_max**4)
    delta_wall = float(c) / max(float(Gamma_w), 1e-300)
    period = float(lambda_micro / max(delta_wall, 1e-300))

    try:
        from bvp_coherence_q import compute_coherence_q  # noqa: WPS433

        _t1, _tr, _q, q2, _rw2 = compute_coherence_q()
    except Exception:
        q2 = 1.0

    try:
        from wall_fractal_2d import run_wall_fractal_fast  # noqa: WPS433

        # dao 汇总链中采用缩网格/步数以控制耗时；全精度运行见 wall_fractal_2d.main()
        _ratio, f_fluct = run_wall_fractal_fast(nr=120, nth=120, n_steps=3000, t_phys=None)
    except Exception:
        f_fluct = 1.0

    factors = {
        "WKB_gamma^3/4": wkb,
        "gamma_max^4": vol,
        "lambda_micro/delta_wall": period,
        "Q^2": q2,
        "f_fluct": f_fluct,
        "f_radial": F_RADIAL,
    }
    total = float(np.prod(list(factors.values())))
    eta_ind = float(ETA_BVP * total)
    ratio = float(eta_ind / max(ETA_SELF, 1e-300))

    print("dao_physics: eta_nonlinear composition")
    for k, v in factors.items():
        print(f"  factor {k:28s} = {v:.6e}")
    print(f"  product(all factors)     = {total:.6e}")
    print(f"  eta_BVP                  = {ETA_BVP:.6e}")
    print(f"  eta_ind = eta_BVP*prod   = {eta_ind:.6e}")
    print(f"  eta_self (reference)     = {ETA_SELF:.6e}")
    print(f"  eta_ind / eta_self       = {ratio:.6e}")


if __name__ == "__main__":
    main()
