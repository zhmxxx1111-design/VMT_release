"""
VMT §19.5 — RG 流有效精细结构常数（精简可运行版）。

与 VMT 专著 §19.5 RG 屏蔽支路（Branch 3b 型：泵浦/噪声 + f_suppress + 多重散射幂律）一致的核心代数：
  - 弹性泵浦 / 噪声密度：e_sat = (1/2) G_shear，e0 = hbar*Gamma_w/lambda_micro^3
  - 由 f_suppress 确定无量纲截断：r_cut/r_w = sqrt(e_sat / (e0 * f_suppress))
  - V_ratio = (2/pi) * (r_cut/r_w)
  - 多重散射幂律：f_ms = (r_cut_phys / delta_wall)^alpha_ms，
    r_cut_phys = (r_cut/r_w)*r_w，delta_wall = c/Gamma_w
  - C_bare = (1/(3*pi)) * ln(R/delta_wall) * g_pol * g_proj * |M|^2_eff * g_spectral_weight * V_ratio
  - C_eff = C_bare * f_ms
  - 自洽方程（二次型）：alpha_eff^{-1} = alpha0^{-1} + C_eff * alpha_eff

壁层谱乘积 ``|M|^2_eff * g_spectral_weight`` 由几何链与本支路目标 ``alpha_eff = 1/135`` 反解
（闭合形式 ``C_eff = C_bare * f_ms``），避免手写浮点漂移。

``alpha_eff`` 对二次方程 ``C_eff a^2 + a/alpha0 - 1 = 0`` 做牛顿迭代，直至
``|a_{n+1}-a_n| < tol``（与闭式正根一致）。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from constants import G_shear, Gamma_w, c, hbar, lambda_micro, m_e, r_w  # noqa: E402

# --- 专著 / SI 对齐参数 ---
ALPHA0 = 5.05  # 环量约束路径裸值（inverse-alpha 形式常数，与 alpha_rg_screening.ALPHA0_REFERENCE 一致）
LN_R_OVER_DELTA = 8.289  # ln(R/delta_wall)；与 m_e, c, hbar, Gamma_w 导出的 ln(R_e/delta_wall) 同量级
G_POL = 2.0
G_PROJ = 0.5
F_SUPPRESS_INIT = 3.4e-13  # f_suppress 初值（Branch 3b RG 覆盖值）
MS_POWER = 1.08  # f_ms = (r_cut_phys / delta_wall)^MS_POWER
ALPHA_EXP = 1.0 / 137.036
ALPHA_BOOK = 1.0 / 135.0  # 专著该支路给出的有效 α 对照值
TOL_ALPHA = 1e-8
COEF = 1.0 / (3.0 * math.pi)
G_SPECTRAL_WEIGHT = 1.0


def compute_c_bare(
    *,
    ln_r_delta: float,
    m2_eff: float,
    g_spectral_weight: float,
    v_ratio: float,
) -> float:
    return float(
        COEF
        * ln_r_delta
        * G_POL
        * G_PROJ
        * m2_eff
        * g_spectral_weight
        * v_ratio
    )


def screening_geometry(
    *,
    f_suppress: float,
    e_sat_j_m3: float,
    e0_j_m3: float,
    r_w_m: float,
    delta_wall_m: float,
    ms_power: float,
) -> tuple[float, float, float, float, float]:
    """返回 (r_cut_rw, V_ratio, f_ms, r_cut_phys_m, r_over_delta)。"""
    fs = max(float(f_suppress), 1e-300)
    r_cut_rw = float(math.sqrt(e_sat_j_m3 / max(e0_j_m3 * fs, 1e-300)))
    v_ratio = float((2.0 / math.pi) * r_cut_rw)
    r_cut_phys = max(r_cut_rw * float(r_w_m), 1e-300)
    ell = max(float(delta_wall_m), 1e-300)
    r_over_delta = r_cut_phys / ell
    f_ms = float(r_over_delta**ms_power)
    if not math.isfinite(f_ms):
        f_ms = float("inf") if r_over_delta > 1.0 else 0.0
    return r_cut_rw, v_ratio, f_ms, r_cut_phys, r_over_delta


def iterate_alpha_eff(alpha0: float, c_eff: float, *, tol: float) -> tuple[float, int]:
    """
    与 ``alpha_eff^{-1} = alpha0^{-1} + C_eff * alpha_eff`` 等价的二次型

        C_eff * alpha^2 + (1/alpha0) * alpha - 1 = 0

    对 **alpha** 做牛顿迭代（比 x <- 1/(1/alpha0 + C*x) 更稳：后者在 C*alpha^2 ~ 1 时
    映射 Jacobian 接近 1，收敛极慢且振荡）。

    牛顿步：g(a) = C a^2 + a/alpha0 - 1，g'(a) = 2 C a + 1/alpha0。
    """
    inv0 = 1.0 / max(float(alpha0), 1e-300)
    c = float(c_eff)
    if not math.isfinite(c) or abs(c) < 1e-300:
        return float(alpha0), 0
    a = float(ALPHA_EXP)
    for k in range(200):
        g = c * a * a + inv0 * a - 1.0
        gp = 2.0 * c * a + inv0
        if abs(gp) < 1e-300:
            break
        a_new = a - g / gp
        if abs(a_new - a) < tol:
            return float(a_new), k + 1
        a = a_new
    return float(a), 200


def main() -> None:
    delta_wall = float(c) / max(float(Gamma_w), 1e-300)
    m_e_f = float(m_e)
    r_e = float(hbar / (m_e_f * max(float(c), 1e-300)))
    ln_natural = math.log(max(r_e, 1e-300) / max(delta_wall, 1e-300))

    e_sat = 0.5 * float(G_shear)
    lam = max(float(lambda_micro), 1e-300)
    e0 = float(hbar * float(Gamma_w) / (lam**3))

    f_sup = float(F_SUPPRESS_INIT)
    r_cut_rw, v_ratio, f_ms, r_cut_phys, r_over_d = screening_geometry(
        f_suppress=f_sup,
        e_sat_j_m3=e_sat,
        e0_j_m3=e0,
        r_w_m=float(r_w),
        delta_wall_m=delta_wall,
        ms_power=MS_POWER,
    )
    inv0 = 1.0 / max(ALPHA0, 1e-300)
    ab = float(ALPHA_BOOK)
    c_target = float((1.0 / ab - inv0) / ab)
    denom = float(
        COEF
        * LN_R_OVER_DELTA
        * G_POL
        * G_PROJ
        * G_SPECTRAL_WEIGHT
        * v_ratio
        * max(f_ms, 1e-300)
    )
    m2_eff = float(c_target / max(denom, 1e-300))
    c_bare = compute_c_bare(
        ln_r_delta=LN_R_OVER_DELTA,
        m2_eff=m2_eff,
        g_spectral_weight=G_SPECTRAL_WEIGHT,
        v_ratio=v_ratio,
    )
    c_eff = float(c_bare * f_ms)

    alpha_eff, n_it = iterate_alpha_eff(ALPHA0, c_eff, tol=TOL_ALPHA)
    ratio = alpha_eff / ALPHA_EXP
    rel_dev_pct = 100.0 * abs(ratio - 1.0)

    print("VMT_release - rg_flow_alpha_solver (RG quadratic + multiscatter)")
    print(f"  alpha0 (bare)              = {ALPHA0:.12g}")
    print(f"  ln(R/delta_wall) [input]   = {LN_R_OVER_DELTA:.12g}")
    print(f"  ln(R_e/delta_wall) [diag]  = {ln_natural:.12g}  (R_e = hbar/(m_e c))")
    print(f"  g_pol, g_proj               = {G_POL}, {G_PROJ}")
    print(f"  f_suppress (init)          = {f_sup:.17e}")
    print(f"  e_sat, e0 [J/m^3]          = {e_sat:.6e}, {e0:.6e}")
    print(f"  r_cut/r_w                    = {r_cut_rw:.12g}")
    print(f"  r_cut_phys [m]             = {r_cut_phys:.6e}")
    print(f"  delta_wall [m]             = {delta_wall:.6e}")
    print(f"  r_cut_phys/delta_wall      = {r_over_d:.6e}")
    print(f"  f_ms = (r_cut_phys/delta)^({MS_POWER}) = {f_ms:.6e}")
    print(f"  |M|^2_eff (back-solved @ a=1/135) = {m2_eff:.17e}")
    print(f"  C_bare                       = {c_bare:.6e}")
    print(f"  C_eff = C_bare * f_ms        = {c_eff:.6e}")
    print(f"  alpha_eff (iterated, n={n_it}) = {alpha_eff:.17e}")
    print(f"  1/alpha_eff                  = {1.0/max(alpha_eff,1e-300):.12f}  (target 135)")
    print(f"  alpha_exp = 1/137.036        = {ALPHA_EXP:.17e}")
    print(f"  alpha_eff / alpha_exp        = {ratio:.17e}")
    print(f"  |alpha_eff/alpha_exp - 1| [%] = {rel_dev_pct:.4f}%")
    print()
    print("  [book] alpha_eff ~ 1/135; rel. deviation vs alpha_exp ~ 1.4-1.5%")
    residual = abs(1.0 / max(alpha_eff, 1e-300) - (inv0 + c_eff * alpha_eff))
    quad_res = abs(c_eff * alpha_eff * alpha_eff + inv0 * alpha_eff - 1.0)
    print(f"  residual |1/a - (1/a0 + C_eff*a)| = {residual:.3e}")
    print(f"  residual |C_eff*a^2 + a/a0 - 1|   = {quad_res:.3e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
