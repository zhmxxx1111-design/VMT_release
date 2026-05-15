"""
环面 Chern–Simons 介质耦合算符数值验证（虚体介质论管线）。

修正后的介质极化（塑性耗散饱和，--logspace 时使用）：
    ln f(γ) = α ∫₁^γ [τ′(γ′)/G_shear] · exp(-(γ′/γ_sat)²) dγ′
    f(γ) = exp(ln f(γ))

τ′ 与 nonlinear_bvp / constants.tau 分段一致。γ_sat 为极化饱和阈值。

剖面：nonlinear_bvp 参考解 + 径向/振幅重标（同前）。

--logspace：对 ln f 在过渡区内 τ 加权平均得 mean_ln_f，再 f_bar = min(exp(mean_ln_f), 1e300)。

--gamma_sat：阻尼尺度；``auto`` 时在 [5, 50] 上搜索（三代默认 n_r=0,1,2 下 C_μ/C_e）。

--gamma_core：管芯极化阈值（默认 16）；``auto`` 时固定 γ_sat=6.5 并在 [10,25] 上反推 γ_core（τ 质量 30% 判据）。

径向呼吸子模式 ``--n_r``：加权 ``w = τ(γ) · φ_{n_r}(γ)``，默认三代 ``(0,1,2)``，
``φ_{n_r}(γ) = 1 + n_r · (γ/γ_max)²``（γ_max 为该代剖面峰值）。

质量（MeV）：无管芯拆分时 λ = m_μ^exp / C_μ，M_i = λ·C_i。
有管芯拆分时 C_trans = 2π·f_trans，C_core = 2π·(V_ratio·f_core_raw)，C_total = C_trans + C_core；
λ = m_μ^exp / C_trans^μ（仅用 μ 过渡区定标），M_i = λ·C_trans,i + C_core,i（管芯项不进入 λ 分母）。

管芯体积极化（--gamma_core）：过渡区 γ∈[1,γ_core] 用原耗散截断；管芯 γ>γ_core 用常数截断
exp(-(γ_core/γ_sat)²) 并乘以体积因子 (γ_max/γ_core)³ 叠加到 f̄_total。
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from constants import G_shear, alpha, c, hbar, tau  # noqa: E402
from nonlinear_bvp import interpolate_r_w, solve_tube_wall_bvp  # noqa: E402

MEV_J = 1.602_176_634e-13
HBAR_C = float(hbar * c)

F_BAR_LOG_CAP = 1e300
LN_F_BAR_CAP = math.log(F_BAR_LOG_CAP)

# 电子 / μ 子质量比（反推 γ_sat 目标）
M_E_EXP_MEV = 0.510_998_946_1
M_MU_EXP_MEV = 105.658_374_5
M_TAU_EXP_MEV = 1776.86
TARGET_C_RATIO = M_MU_EXP_MEV / M_E_EXP_MEV  # C_mu/C_e

_LN_F_CACHE: dict[tuple[float, float], float] = {}
_LN_F_CORE_CACHE: dict[tuple[float, float, float], float] = {}


def tau_prime_gamma(gp: float) -> float:
    g = float(gp)
    if g < 1.0:
        return float(G_shear)
    return float(G_shear * ((1.0 + 4.0 * (g - 1.0) ** 2) + g * 8.0 * (g - 1.0)))


def exponent_integral_one_to_gamma(gamma: float, *, gamma_integrate_to: float | None = None) -> float:
    if gamma <= 1.0:
        return 0.0
    g_hi = float(min(float(gamma), float(gamma_integrate_to))) if gamma_integrate_to is not None else float(gamma)
    if g_hi <= 1.0:
        return 0.0

    def integrand(gp: float) -> float:
        return tau_prime_gamma(gp) / max(G_shear, 1e-300)

    val, _ = quad(integrand, 1.0, g_hi, limit=200)
    return float(val)


def f_medium(gamma: float, *, gamma_integrate_to: float | None = None) -> float:
    x = float(alpha) * exponent_integral_one_to_gamma(gamma, gamma_integrate_to=gamma_integrate_to)
    x = min(x, 700.0)
    return math.exp(x)


def exponent_full_for_log(gamma: float) -> float:
    """未阻尼的 α∫τ′/G（诊断）。"""
    return float(alpha) * exponent_integral_one_to_gamma(gamma, gamma_integrate_to=None)


def ln_f_medium_damped(gamma: float, gamma_sat: float) -> float:
    """
    ln f(γ) = α ∫₁^γ [τ′(γ′)/G_shear] exp(-(γ′/γ_sat)²) dγ′ ；γ≤1 为 0。
    """
    if gamma <= 1.0:
        return 0.0
    gs = max(float(gamma_sat), 1e-9)
    key = (round(float(gamma), 10), round(gs, 8))
    if key in _LN_F_CACHE:
        return _LN_F_CACHE[key]

    def integrand(gp: float) -> float:
        base = tau_prime_gamma(gp) / max(G_shear, 1e-300)
        damp = math.exp(-((float(gp) / gs) ** 2))
        return base * damp

    val, _ = quad(integrand, 1.0, float(gamma), limit=300)
    out = float(alpha) * val
    _LN_F_CACHE[key] = out
    return out


def ln_f_medium_damped_core(gamma: float, gamma_sat: float, gamma_core: float) -> float:
    """
    管芯分支：ln f(γ) = α · exp(-(γ_core/γ_sat)²) · ∫₁^γ τ′/G dγ′
    （耗散截断在积分内为常数，不随 γ′ 增大而增强。）
    """
    if gamma <= 1.0:
        return 0.0
    gs = max(float(gamma_sat), 1e-9)
    gc = max(float(gamma_core), 1e-9)
    key = (round(float(gamma), 10), round(gs, 8), round(gc, 8))
    if key in _LN_F_CORE_CACHE:
        return _LN_F_CORE_CACHE[key]
    k = math.exp(-((gc / gs) ** 2))
    undamped = exponent_integral_one_to_gamma(float(gamma))
    out = float(alpha) * k * float(undamped)
    _LN_F_CORE_CACHE[key] = out
    return out


def phi_nr(gamma: float, n_r: int, gamma_max: float) -> float:
    """径向呼吸子权重因子 φ_{n_r}(γ) = 1 + n_r (γ/γ_max)²。"""
    gm = max(float(gamma_max), 1e-30)
    return 1.0 + float(n_r) * (float(gamma) / gm) ** 2


def ln_f_bar_weighted_tau(
    r: np.ndarray,
    gamma: np.ndarray,
    *,
    r_w: float,
    gamma_sat: float,
    n_r: int = 0,
    gamma_clip_max: float | None = None,
) -> tuple[float, float, int]:
    gmax = float(np.max(gamma)) if gamma_clip_max is None else float(gamma_clip_max)
    mask = (r <= r_w) & (gamma >= 1.0 - 1e-9) & (gamma <= gmax + 1e-6)
    if int(np.count_nonzero(mask)) < 3:
        return float("nan"), float("nan"), int(np.count_nonzero(mask))

    rr = r[mask]
    order = np.argsort(rr)
    rr = rr[order]
    gg = gamma[mask][order]
    ln_ff = np.array([ln_f_medium_damped(float(g), gamma_sat) for g in gg])
    ww = np.array(
        [
            max(float(tau(float(g))), 0.0) * phi_nr(float(g), n_r, gmax)
            for g in gg
        ]
    )
    num = float(np.trapezoid(ln_ff * ww, rr))
    den = float(np.trapezoid(ww, rr))
    if den <= 0.0:
        return float("nan"), float("nan"), int(rr.size)
    mean_ln = num / den
    mean_ln_clamped = min(mean_ln, LN_F_BAR_CAP)
    f_bar_log = float(math.exp(mean_ln_clamped))
    return mean_ln, f_bar_log, int(rr.size)


def ln_f_bar_weighted_tau_core_split(
    r: np.ndarray,
    gamma: np.ndarray,
    *,
    r_w: float,
    gamma_sat: float,
    gamma_core: float,
    n_r: int = 0,
    gamma_clip_max: float | None = None,
) -> tuple[float, float, float, float, float, float, float, int, int]:
    """
    过渡区 γ∈[1,γ_core]：ln f 用耗散截断 exp(-(γ′/γ_sat)²)；
    管芯 γ>γ_core：ln f 用常数截断 exp(-(γ_core/γ_sat)²) 的 ln_f_medium_damped_core；
    f̄_total = f̄_trans + (γ_max/γ_core)³ · f̄_core_raw；f_core_contrib = V_ratio · f_core_raw。

    Returns:
        f_bar, mean_ln_trans, mean_ln_core, f_trans, f_core_raw, f_core_contrib, v_ratio, npts_trans, npts_core
    """
    gmx = float(np.max(gamma)) if gamma_clip_max is None else float(gamma_clip_max)
    gc = float(gamma_core)

    base_mask = (r <= r_w) & (gamma >= 1.0 - 1e-9) & (gamma <= gmx + 1e-6)
    n_base = int(np.count_nonzero(base_mask))
    if n_base < 3:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            0.0,
            0,
            0,
        )

    rr = r[base_mask]
    gg = gamma[base_mask]
    order = np.argsort(rr)
    rr = rr[order]
    gg = gg[order]
    ln_ff_tr = np.array([ln_f_medium_damped(float(g), gamma_sat) for g in gg])
    ln_ff_co = np.array([ln_f_medium_damped_core(float(g), gamma_sat, gc) for g in gg])
    ww = np.array(
        [max(float(tau(float(g))), 0.0) * phi_nr(float(g), n_r, gmx) for g in gg]
    )

    m_tr = gg <= gc + 1e-9
    m_co = gg > gc + 1e-9
    npts_trans = int(np.count_nonzero(m_tr))
    npts_core = int(np.count_nonzero(m_co))

    if npts_trans < 3:
        mean_ln, fbar, _ = ln_f_bar_weighted_tau(
            r,
            gamma,
            r_w=r_w,
            gamma_sat=gamma_sat,
            n_r=n_r,
            gamma_clip_max=gamma_clip_max,
        )
        return fbar, mean_ln, float("nan"), fbar, float("nan"), 0.0, 0.0, npts_trans, npts_core

    def _weighted_ln_mean(rr_a: np.ndarray, ln_a: np.ndarray, w_a: np.ndarray) -> float:
        if int(rr_a.size) < 1:
            return float("nan")
        if int(rr_a.size) < 3:
            den = float(np.sum(w_a))
            if den <= 0.0:
                return float("nan")
            return float(np.sum(ln_a * w_a) / den)
        ordx = np.argsort(rr_a)
        rr_s = rr_a[ordx]
        ln_s = ln_a[ordx]
        w_s = w_a[ordx]
        num = float(np.trapezoid(ln_s * w_s, rr_s))
        den = float(np.trapezoid(w_s, rr_s))
        if den <= 0.0:
            return float("nan")
        return num / den

    def _trap_mean_ln(ln_a: np.ndarray, mask: np.ndarray) -> float:
        return _weighted_ln_mean(rr[mask], ln_a[mask], ww[mask])

    mean_ln_trans = _trap_mean_ln(ln_ff_tr, m_tr)
    mean_ln_clamped_tr = min(mean_ln_trans, LN_F_BAR_CAP)
    f_trans = float(math.exp(mean_ln_clamped_tr)) if math.isfinite(mean_ln_trans) else float("nan")

    v_ratio = (gmx / max(gc, 1e-30)) ** 3 if gmx > gc + 1e-9 else 0.0
    mean_ln_core = float("nan")
    f_core_raw = 0.0
    if gmx > gc + 1e-9 and npts_core >= 1:
        mean_ln_core = _trap_mean_ln(ln_ff_co, m_co)
        if math.isfinite(mean_ln_core):
            mean_ln_clamped_co = min(mean_ln_core, LN_F_BAR_CAP)
            f_core_raw = float(math.exp(mean_ln_clamped_co))

    f_core_contrib = v_ratio * f_core_raw
    f_bar = f_trans + f_core_contrib
    if not math.isfinite(f_bar):
        f_bar = float("nan")

    return (
        f_bar,
        mean_ln_trans,
        mean_ln_core,
        f_trans,
        f_core_raw,
        f_core_contrib,
        v_ratio,
        npts_trans,
        npts_core,
    )


def rescale_profile(
    r_ref: np.ndarray,
    g_ref: np.ndarray,
    *,
    gamma_max_target: float,
    r_w_target: float,
) -> tuple[np.ndarray, np.ndarray, str]:
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


def f_bar_weighted_tau(
    r: np.ndarray,
    gamma: np.ndarray,
    *,
    r_w: float,
    gamma_clip_max: float | None = None,
    gamma_integrate_to: float | None = None,
) -> tuple[float, int]:
    gmax = float(np.max(gamma)) if gamma_clip_max is None else float(gamma_clip_max)
    mask = (r <= r_w) & (gamma >= 1.0 - 1e-9) & (gamma <= gmax + 1e-6)
    if int(np.count_nonzero(mask)) < 3:
        return float("nan"), int(np.count_nonzero(mask))

    rr = r[mask]
    order = np.argsort(rr)
    rr = rr[order]
    gg = gamma[mask][order]
    ff = np.array([f_medium(float(g), gamma_integrate_to=gamma_integrate_to) for g in gg])
    ww = np.array([max(float(tau(float(g))), 0.0) for g in gg])
    num = float(np.trapezoid(ff * ww, rr))
    den = float(np.trapezoid(ww, rr))
    if den <= 0.0:
        return float("nan"), int(rr.size)
    return num / den, int(rr.size)


def mev_from_j(E_j: float) -> float:
    return float(E_j / MEV_J)


def parse_n_r_list(s: str) -> list[int]:
    """``0,1,2`` 三代分别；单个整数则三代同值。"""
    t = str(s).strip()
    if "," in t:
        parts = [int(x.strip()) for x in t.split(",")]
        if len(parts) != 3:
            sys.exit("error: --n_r expects exactly three comma-separated ints, e.g. 0,1,2")
        return parts
    v = int(t)
    return [v, v, v]


LEPTONS = [
    {
        "name": "electron (e)",
        "gamma_max": 7.72,
        "r_w": 6.15e-14,
        "m_exp_mev": M_E_EXP_MEV,
    },
    {
        "name": "muon (mu)",
        "gamma_max": 23.2,
        "r_w": 6.15e-14 / 3.0,
        "m_exp_mev": M_MU_EXP_MEV,
    },
    {
        "name": "tau (tau)",
        "gamma_max": 38.7,
        "r_w": 6.15e-14 / 5.0,
        "m_exp_mev": M_TAU_EXP_MEV,
    },
]


def _profiles_for_leptons(
    r_ref: np.ndarray,
    g_ref: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, float, float, str]]:
    out: list[tuple[np.ndarray, np.ndarray, float, float, str]] = []
    for row in LEPTONS:
        r_m, g_m, note = rescale_profile(
            r_ref,
            g_ref,
            gamma_max_target=float(row["gamma_max"]),
            r_w_target=float(row["r_w"]),
        )
        rw_m = float(interpolate_r_w(r_m, g_m))
        gmx_m = float(np.max(g_m))
        out.append((r_m, g_m, rw_m, gmx_m, note))
    return out


def _compute_C_dims_damped(
    profiles: list[tuple[np.ndarray, np.ndarray, float, float, str]],
    gamma_sat: float,
    n_r_list: list[int],
    *,
    gamma_core: float | None = None,
) -> tuple[list[float], list[float], list[int], list[float], list[dict[str, float | int]]]:
    """
    返回 (f_bars, C_dims, npts_list, mean_ln_list, split_info)。
    gamma_core is None：整段过渡区阻尼（旧行为）；否则过渡/管芯拆分。
    """
    global _LN_F_CACHE
    _LN_F_CACHE.clear()
    _LN_F_CORE_CACHE.clear()
    pq = 1
    f_bars: list[float] = []
    C_dims: list[float] = []
    npts_l: list[int] = []
    mean_ln_l: list[float] = []
    split_l: list[dict[str, float | int]] = []
    for idx, (r_m, g_m, rw_m, gmx_m, _note) in enumerate(profiles):
        nr = int(n_r_list[idx]) if idx < len(n_r_list) else 0
        if gamma_core is None:
            mean_ln, fbar, npts = ln_f_bar_weighted_tau(
                r_m,
                g_m,
                r_w=rw_m,
                gamma_sat=gamma_sat,
                n_r=nr,
                gamma_clip_max=gmx_m,
            )
            c0 = float(2.0 * math.pi * pq * fbar)
            split_l.append(
                {
                    "mode": 0,
                    "npts_trans": npts,
                    "npts_core": 0,
                    "v_ratio": 0.0,
                    "f_trans": fbar,
                    "f_core_raw": float("nan"),
                    "f_core_contrib": 0.0,
                    "mean_ln_core": float("nan"),
                    "C_trans": c0,
                    "C_core": 0.0,
                    "C_total": c0,
                }
            )
            f_bars.append(fbar)
            C_dims.append(c0)
        else:
            (
                fbar,
                mean_ln_tr,
                mean_ln_co,
                f_tr,
                f_co_raw,
                f_core_contrib,
                v_ratio,
                nt,
                nc,
            ) = ln_f_bar_weighted_tau_core_split(
                r_m,
                g_m,
                r_w=rw_m,
                gamma_sat=gamma_sat,
                gamma_core=float(gamma_core),
                n_r=nr,
                gamma_clip_max=gmx_m,
            )
            mean_ln = mean_ln_tr
            npts = int(nt + nc)
            c_tr = 2.0 * math.pi * pq * float(f_tr)
            c_co = 2.0 * math.pi * pq * float(f_core_contrib)
            c_tot = c_tr + c_co
            split_l.append(
                {
                    "mode": 1,
                    "npts_trans": nt,
                    "npts_core": nc,
                    "v_ratio": float(v_ratio),
                    "f_trans": float(f_tr),
                    "f_core_raw": float(f_co_raw),
                    "f_core_contrib": float(f_core_contrib),
                    "mean_ln_core": float(mean_ln_co),
                    "C_trans": float(c_tr),
                    "C_core": float(c_co),
                    "C_total": float(c_tot),
                }
            )
            f_bars.append(fbar)
            C_dims.append(c_tot)
        npts_l.append(npts)
        mean_ln_l.append(mean_ln)
    return f_bars, C_dims, npts_l, mean_ln_l, split_l


def _mass_preds_trans_plus_core(
    split_infos: list[dict[str, float | int]],
) -> tuple[float, list[float], list[float], list[float]]:
    """
    λ = m_μ^exp / C_trans^μ（仅 μ 子过渡区）；M_i = λ·C_trans,i + C_core,i（管芯不入 λ 分母）。
    返回 (lambda, M_preds, C_trans_list, C_core_list)。
    """
    c_tr = [float(split_infos[i]["C_trans"]) for i in range(3)]
    c_co = [float(split_infos[i]["C_core"]) for i in range(3)]
    if c_tr[1] <= 1e-300:
        return float("nan"), [float("nan")] * 3, c_tr, c_co
    lam = float(M_MU_EXP_MEV / c_tr[1])
    preds = [lam * c_tr[i] + c_co[i] for i in range(3)]
    return lam, preds, c_tr, c_co


def _ratio_mu_over_e(C_dims: list[float]) -> float:
    if len(C_dims) < 2 or C_dims[0] <= 1e-300:
        return float("inf")
    return float(C_dims[1] / C_dims[0])


def find_gamma_sat_auto(
    profiles: list[tuple[np.ndarray, np.ndarray, float, float, str]],
    n_r_list: list[int],
    *,
    lo: float = 5.0,
    hi: float = 50.0,
    n_scan: int = 120,
) -> tuple[float, float]:
    """Scan gamma_sat in [lo, hi] so C_mu/C_e ~ m_mu/m_e; return (best_gs, min |log R - log target|)."""
    best_gs = 15.0
    best_err = float("inf")
    for k in range(n_scan + 1):
        gs = lo + (hi - lo) * (k / max(n_scan, 1))
        C_dims = _compute_C_dims_damped(profiles, gs, n_r_list, gamma_core=None)[1]
        R = _ratio_mu_over_e(C_dims)
        if not math.isfinite(R):
            continue
        err = abs(math.log(R) - math.log(TARGET_C_RATIO))
        if err < best_err:
            best_err = err
            best_gs = gs
    # 局部细化：在最优点附近再扫一圈
    span = (hi - lo) / max(n_scan, 1) * 3.0
    lo2 = max(lo, best_gs - span)
    hi2 = min(hi, best_gs + span)
    for k in range(61):
        gs = lo2 + (hi2 - lo2) * (k / 60.0)
        C_dims = _compute_C_dims_damped(profiles, gs, n_r_list, gamma_core=None)[1]
        R = _ratio_mu_over_e(C_dims)
        if not math.isfinite(R):
            continue
        err = abs(math.log(R) - math.log(TARGET_C_RATIO))
        if err < best_err:
            best_err = err
            best_gs = gs
    return best_gs, best_err


def find_gamma_core_auto(
    profiles: list[tuple[np.ndarray, np.ndarray, float, float, str]],
    n_r_list: list[int],
    *,
    gamma_sat: float = 6.5,
) -> tuple[float | None, float, list[float], list[float], bool, list[float], list[float]]:
    """
    扫描 γ_core ∈ [10, 25] 步长 0.5；固定 γ_sat；λ = m_μ/C_trans^μ；M_i = λ C_trans,i + C_core,i。
    若有候选使三代相对实验均 ≤30%，在其中取 |M_τ - m_τ^exp| 最小；
    否则若有 M_τ 在 30% 内，在该子集中取 |M_τ - m_τ^exp| 最小；
    否则在全网格上取 |M_τ - m_τ^exp| 最小。
    返回 (best_gamma_core, best_abs_err_tau_mev, f_bars, C_dims, all_three_ok, C_trans, C_core)。
    """
    grid = [float(x) for x in np.arange(10.0, 25.5, 0.5)]
    cand_three: list[tuple[float, float, list[float], list[float], list[float], list[float]]] = []
    cand_tau: list[tuple[float, float, list[float], list[float], list[float], list[float]]] = []
    cand_global: list[tuple[float, float, list[float], list[float], list[float], list[float]]] = []

    for gc in grid:
        f_bars, C_dims, _, _, split_infos = _compute_C_dims_damped(
            profiles, float(gamma_sat), n_r_list, gamma_core=gc
        )
        lam, preds, c_tr, c_co = _mass_preds_trans_plus_core(split_infos)
        if not all(math.isfinite(x) for x in preds) or not math.isfinite(lam):
            continue
        rels = [
            abs(preds[i] - float(LEPTONS[i]["m_exp_mev"])) / max(float(LEPTONS[i]["m_exp_mev"]), 1e-30)
            for i in range(3)
        ]
        abs_tau = abs(preds[2] - M_TAU_EXP_MEV)
        row = (gc, abs_tau, f_bars, C_dims, c_tr, c_co)
        cand_global.append(row)
        if rels[2] <= 0.3:
            cand_tau.append(row)
        if rels[0] <= 0.3 and rels[1] <= 0.3 and rels[2] <= 0.3:
            cand_three.append(row)

    if cand_three:
        pool = cand_three
        all_ok = True
    elif cand_tau:
        pool = cand_tau
        all_ok = False
    else:
        pool = cand_global
        all_ok = False
    if not pool:
        return None, float("inf"), [], [], False, [], []
    best_gc, best_abs, fb, cd, c_tr_b, c_co_b = min(pool, key=lambda t: t[1])[0:6]
    return best_gc, float(best_abs), fb, cd, all_ok, c_tr_b, c_co_b


def main(
    *,
    logspace: bool = False,
    gamma_sat: float = 15.0,
    gamma_sat_auto: bool = False,
    gamma_core: float = 16.0,
    gamma_core_auto: bool = False,
    n_r_list: list[int] | None = None,
) -> None:
    pq = 1
    GAMMA_INT_CAP = 5.2
    if n_r_list is None:
        n_r_list = [0, 1, 2]
    if len(n_r_list) != 3:
        raise ValueError("n_r_list must have length 3 (e, mu, tau)")

    r_ref, g_ref = solve_tube_wall_bvp()
    rw_ref = float(interpolate_r_w(r_ref, g_ref))
    gm_ref = float(np.max(g_ref))
    profiles = _profiles_for_leptons(r_ref, g_ref)

    run_gamma_sat_auto = bool(gamma_sat_auto)
    gs_use = float(gamma_sat)
    gc_use = float(gamma_core)

    if gamma_core_auto:
        gs_use = 6.5
        run_gamma_sat_auto = False
        print("gamma_core=auto: fixing gamma_sat = 6.5 (muon-scale lock); scanning gamma_core in [10, 25] step 0.5 ...")
        if gamma_sat_auto:
            print("  (note) --gamma_sat auto is disabled while --gamma_core auto is active.")
        best_gc, abs_tau, fb_scan, cd_scan, all_three_pool, ctr_scan, cco_scan = find_gamma_core_auto(
            profiles, n_r_list, gamma_sat=gs_use
        )
        if best_gc is not None:
            gc_use = float(best_gc)
            if len(ctr_scan) >= 3 and ctr_scan[1] > 1e-300:
                lam_s = float(M_MU_EXP_MEV / ctr_scan[1])
                M_tau_s = float(lam_s * ctr_scan[2] + cco_scan[2])
                rel_tau_s = abs(M_tau_s - M_TAU_EXP_MEV) / M_TAU_EXP_MEV
            else:
                lam_s = float("nan")
                M_tau_s = float("nan")
                rel_tau_s = float("nan")
            print(
                f"  best gamma_core = {gc_use:.2f}  (|M_tau - m_tau^exp| = {abs_tau:.2f} MeV; "
                f"rel_err(tau)={rel_tau_s:.4f} at scan optimum; "
                f"pool had all-three<=30%: {all_three_pool})"
            )
            if not all_three_pool:
                print("  (note) optimal pool was 'tau within 30% only' (no scan point had e,mu,tau all <=30%).")
        else:
            print("  warning: no gamma_core with M_tau within 30% of experiment; using gamma_core=16.0.")
            gc_use = 16.0

    if run_gamma_sat_auto:
        print("scanning gamma_sat in [5, 50] to match C_mu/C_e ~ m_mu/m_e ...")
        gs_use, err_log = find_gamma_sat_auto(profiles, n_r_list)
        print(f"  best gamma_sat = {gs_use:.6f}  (|log(C_mu/C_e) - log(target)| = {err_log:.6e})")
        print(f"  target C_mu/C_e = {TARGET_C_RATIO:.4f}")

    use_core = (logspace and not run_gamma_sat_auto) or gamma_core_auto

    mode_parts = []
    if logspace or run_gamma_sat_auto:
        extra_gc = f", gamma_core={gc_use:.4g} (core polarisation)" if use_core else ""
        mode_parts.append(
            f"logspace damped ln f, gamma_sat={gs_use:.6g}, n_r={tuple(n_r_list)}{extra_gc}"
        )
    else:
        mode_parts.append(f"linear f, gamma_int_cap={GAMMA_INT_CAP}")
    print(f"chern_simons_torus: reference BVP + rescale -> f_bar -> C  [{'; '.join(mode_parts)}]")
    print(f"  reference: r_w_ref = {rw_ref:.6e} m, gamma_max_ref = {gm_ref:.4f}")

    f_bars: list[float] = []
    C_dims: list[float] = []
    E_geos: list[float] = []
    mean_lns: list[float] = []
    npts_list: list[int] = []
    split_infos: list[dict[str, float | int]] | None = None

    if logspace or run_gamma_sat_auto:
        f_bars, C_dims, npts_list, mean_lns, split_infos = _compute_C_dims_damped(
            profiles,
            gs_use,
            n_r_list,
            gamma_core=(gc_use if use_core else None),
        )
    else:
        f_bars = []
        C_dims = []
        npts_list = []
        mean_lns = []
        split_infos = None

    for i, row in enumerate(LEPTONS):
        r_m, g_m, rw_m, gmx_m, note = profiles[i]

        if logspace or run_gamma_sat_auto:
            fbar = f_bars[i]
            mean_ln_f = mean_lns[i]
            npts = npts_list[i]
            C_dim = C_dims[i]
        else:
            mean_ln_f = float("nan")
            fbar, npts = f_bar_weighted_tau(
                r_m, g_m, r_w=rw_m, gamma_clip_max=gmx_m, gamma_integrate_to=GAMMA_INT_CAP
            )
            C_dim = 2.0 * math.pi * pq * fbar
            f_bars.append(fbar)
            C_dims.append(C_dim)
            mean_lns.append(mean_ln_f)
            npts_list.append(npts)

        E_geo_mev = mev_from_j(HBAR_C / max(float(row["r_w"]), 1e-30))
        E_geos.append(E_geo_mev)

        print(f"\n--- {row['name']} ---")
        print(f"  targets    : gamma_max~{row['gamma_max']}, r_w~{row['r_w']:.6e} m")
        print(f"  after map  : gamma_max={gmx_m:.4f}, r_w={rw_m:.6e} m  ({note})")
        xp_full = exponent_full_for_log(gmx_m)
        try:
            log10_fmax = float(xp_full * math.log10(math.e))
        except (OverflowError, ValueError):
            log10_fmax = float("inf")
        if not math.isfinite(log10_fmax) or abs(log10_fmax) > 200.0:
            log10_fmax = float("inf")
        print(f"  (diag) undamped log10 f(gamma_max) ~ {log10_fmax:.3e}")
        if logspace or run_gamma_sat_auto:
            print(f"  damped ln f: gamma_sat={gs_use:.6g}, n_r={int(n_r_list[i])}, phi=1+n_r*(g/gmax)^2")
            if use_core and split_infos is not None:
                sp = split_infos[i]
                mlc = sp["mean_ln_core"]
                mlc_s = f"{float(mlc):.6f}" if isinstance(mlc, (int, float)) and math.isfinite(float(mlc)) else "nan"
                fcr = sp["f_core_raw"]
                fcr_s = f"{float(fcr):.6e}" if isinstance(fcr, (int, float)) and math.isfinite(float(fcr)) else "nan"
                print(
                    f"  core split: gamma_core={gc_use:.4g}, n_pts trans/core = {sp['npts_trans']}/{sp['npts_core']}, "
                    f"V_ratio=(gmax/gamma_core)^3={sp['v_ratio']:.4g}"
                )
                print(f"    f_trans={sp['f_trans']:.6e}, f_core_raw={fcr_s}, f_core_contrib={float(sp.get('f_core_contrib', 0.0)):.6e}, mean_ln_core={mlc_s}")
            print(f"  mean_ln_f (transition zone) = {mean_ln_f:.6f}")
        else:
            print(f"  linear f integral cap gamma<={GAMMA_INT_CAP}")
        print(f"  mask points: {npts}")
        print(f"  f_bar     = {fbar:.6e}")
        print(f"  C_dimless = 2*pi*p*q*f_bar = {C_dim:.6e}")
        if (logspace or run_gamma_sat_auto) and fbar >= F_BAR_LOG_CAP * 0.999:
            print("  (note) f_bar hit cap 1e300")
        print(f"  E_geo=hbar*c/r_w = {E_geo_mev:.3f} MeV")

    # 质量：管芯拆分时 λ = m_mu/C_trans^μ，M_i = λ C_trans,i + C_core,i；否则 λ = m_mu/C_μ，M_i = λ C_i
    print("\n--- masses ---")
    if not all(math.isfinite(x) for x in f_bars):
        print("  Fit failed (non-finite f_bar).")
    elif use_core and split_infos is not None:
        lam, preds, c_trs, c_cos = _mass_preds_trans_plus_core(split_infos)
        if not math.isfinite(lam) or c_trs[1] <= 1e-300:
            print("  Fit failed (C_trans_mu invalid or lambda non-finite).")
        else:
            print(f"  lambda = m_mu_exp / C_trans^mu = {lam:.6e} MeV / C_trans_unit")
            print(
                f"  M_i = lambda * C_trans,i + C_core,i  (C_total = C_trans + C_core = C_dim in print above)"
            )
            print(
                f"  achieved C_mu/C_e (using C_total) = {C_dims[1]/max(C_dims[0],1e-300):.4f}  (target {TARGET_C_RATIO:.4f})"
            )
            ok_all = True
            ok_30_all = True
            for i, row in enumerate(LEPTONS):
                M_pred = preds[i]
                m_exp = float(row["m_exp_mev"])
                rel = abs(M_pred - m_exp) / max(m_exp, 1e-30)
                ok_30_i = rel <= 0.3
                if not ok_30_i:
                    ok_30_all = False
                if i == 0:
                    ok_i = M_pred < 0.52
                    flag = "OK" if ok_i else "e: M_pred >= 0.52 MeV"
                else:
                    ok_i = rel <= 0.5
                    flag = "OK" if ok_i else f"rel_err={rel:.3f}"
                if not ok_i:
                    ok_all = False
                lam_ct = lam * c_trs[i]
                print(
                    f"  {row['name']}: f_bar={f_bars[i]:.6e}  C_trans={c_trs[i]:.6e}  C_core={c_cos[i]:.6e}  "
                    f"C_total={C_dims[i]:.6e}  lambda*C_trans={lam_ct:.4f} MeV  M_pred={M_pred:.4f} MeV  "
                    f"exp={m_exp:.4f} MeV  rel_err={rel:.4f}  (legacy {flag}; within_30%={'YES' if ok_30_i else 'NO'})"
                )
            print(
                "\n  Verdict (e M_pred<0.52 MeV, mu/tau rel_err<=50%): "
                + ("YES" if ok_all else "NO")
            )
            print(
                "  Verdict (three generations rel_err<=30%): " + ("YES" if ok_30_all else "NO")
            )
    elif C_dims[1] > 1e-300:
        lam = float(M_MU_EXP_MEV / C_dims[1])
        print(f"  lambda = m_mu_exp / C_mu = {lam:.6e} MeV / C_unit")
        if logspace or run_gamma_sat_auto:
            print(f"  achieved C_mu/C_e = {C_dims[1]/max(C_dims[0],1e-300):.4f}  (target {TARGET_C_RATIO:.4f})")

        ok_all = True
        ok_30_all = True
        for i, row in enumerate(LEPTONS):
            M_pred = lam * C_dims[i]
            m_exp = float(row["m_exp_mev"])
            rel = abs(M_pred - m_exp) / max(m_exp, 1e-30)
            ok_30_i = rel <= 0.3
            if not ok_30_i:
                ok_30_all = False
            if i == 0:
                ok_i = M_pred < 0.52
                flag = "OK" if ok_i else "e: M_pred >= 0.52 MeV"
            else:
                ok_i = rel <= 0.5
                flag = "OK" if ok_i else f"rel_err={rel:.3f}"
            if not ok_i:
                ok_all = False
            print(
                f"  {row['name']}: f_bar={f_bars[i]:.6e}  C_dim={C_dims[i]:.6e}  "
                f"M_pred={M_pred:.4f} MeV  exp={m_exp:.4f} MeV  rel_err={rel:.4f}  "
                f"(legacy {flag}; within_30%={'YES' if ok_30_i else 'NO'})"
            )
        print(
            "\n  Verdict (e M_pred<0.52 MeV, mu/tau rel_err<=50%): "
            + ("YES" if ok_all else "NO")
        )
        if logspace or run_gamma_sat_auto:
            print(
                "  Verdict (three generations rel_err<=30%): " + ("YES" if ok_30_all else "NO")
            )
    else:
        print("  Fit failed (C_mu~0).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chern-Simons torus operator (VMT_release)")
    parser.add_argument(
        "--logspace",
        action="store_true",
        help="use damped ln f weighted average then exp (needs gamma_sat for damping)",
    )
    parser.add_argument(
        "--gamma_sat",
        type=str,
        default="15.0",
        help="polarization saturation scale (default 15), or 'auto' to scan [5,50]",
    )
    parser.add_argument(
        "--n_r",
        type=str,
        default="0,1,2",
        help="radial breather indices for e,mu,tau (e.g. 0,1,2) or one int for all three",
    )
    parser.add_argument(
        "--gamma_core",
        type=str,
        default="16.0",
        help="core polarisation threshold gamma_core (default 16), or 'auto' to scan [10,25] at gamma_sat=6.5",
    )
    args = parser.parse_args()
    gs_raw = str(args.gamma_sat).strip().lower()
    auto = gs_raw == "auto"
    if auto:
        gsv = 15.0
    else:
        try:
            gsv = float(args.gamma_sat)
        except ValueError:
            sys.exit("error: --gamma_sat must be a float or 'auto'")

    gc_raw = str(args.gamma_core).strip().lower()
    gc_auto = gc_raw == "auto"
    if gc_auto:
        gcv = 16.0
    else:
        try:
            gcv = float(args.gamma_core)
        except ValueError:
            sys.exit("error: --gamma_core must be a float or 'auto'")

    if not args.logspace and not auto and gsv != 15.0:
        print("note: --gamma_sat only affects --logspace (damped ln f); linear mode unchanged.")

    if not args.logspace and not auto and not gc_auto and gcv != 16.0:
        print("note: --gamma_core only affects --logspace (damped ln f); linear mode unchanged.")

    nr_list = parse_n_r_list(args.n_r)

    if not args.logspace and not auto and not gc_auto and any(nr != 0 for nr in nr_list):
        print("note: breather weights (--n_r) apply only with --logspace or --gamma_sat auto or --gamma_core auto.")

    main(
        logspace=args.logspace or auto or gc_auto,
        gamma_sat=gsv,
        gamma_sat_auto=auto,
        gamma_core=gcv,
        gamma_core_auto=gc_auto,
        n_r_list=nr_list,
    )
