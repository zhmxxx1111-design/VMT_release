"""
虚体介质论 - 公设参数与导出常量
所有数值来自论文各章推导
"""
import numpy as np

# === 三条公设 ===
c = 2.99792458e8  # 真空基准波速 (m/s)
hbar = 1.054571817e-34  # 最小拓扑作用量 (J·s)
E_universe = 8.16e89  # 宇宙总能量 (J)

# === 第二章导出 ===
# 量子相干尺度 (m)：与论文/SI 表格对齐（~5.03e-10）；用于 screening 体积与噪声元胞 ``lambda_micro^3``。
lambda_micro = 5.03e-10
G_shear = 4.94e10  # 真空剪切模量 (Pa)
rho_V = 5.50e-7  # 真空惯性浓度 (kg/m^3)
H_initial = c / lambda_micro  # 初始膨胀率 (s^-1)
H_bare = 1.90e-18  # 基态膨胀率 (s^-1)

# === 第三章导出 ===
f_C = np.e  # 色荷增益因子，三维拓扑必然

# === 第四章导出 ===
omega_W = 1.22e26  # W玻色子频率 (Hz)
Gamma_w = 3.09e24  # 背景涨落频率 (s^-1)
alpha = 1.0 / 137.035999084  # 精细结构常数

# === 第五章参数 ===
m_e = 9.10938356e-31  # 电子质量 (kg)
# 管截面半径 / 大圆半径：与论文 ``r_w = hbar/(2*pi*m_e*c)``, ``R = hbar/(m_e*c)`` 一致
r_w = float(hbar / (2.0 * np.pi * m_e * c))
R_e = float(hbar / (m_e * c))

# === Yang–Mills 参考尺度（微扰匹配；非从公设唯一确定）===
# 诚实标注：Λ_YM 依赖 α_s 的实验锚点与 MS/MSbar 方案；此处用单圈跑动从 1 GeV 演化到 μ=1/λ_micro。
ALPHA_S_1GEV = 0.33  # 强耦合 α_s(1 GeV)，文献常用量级
HBAR_C_J_m = float(hbar * c)  # ħc（J·m）
E_1GEV_J = 1.602176634e-10  # 1 GeV → J
# 1 GeV 对应的波数 k_ref = E/(ħc) [m^-1]
K_REF_PER_M = float(E_1GEV_J / max(HBAR_C_J_m, 1e-300))
HBAR_C_GeV_m = 197.3269804e-15  # ħc 常用值 GeV·m（与 K_REF_PER_M 自检一致）
B0_SU3_PURE_YM = float(11.0 * 3.0 / (48.0 * np.pi**2))  # b_0 = 11N/(48π²), N=3, 无 fermion


def g_squared_yang_mills_at_mu_per_m(mu_per_m: float) -> float:
    """
    单圈：1/g²(μ) = 1/g²(μ₀) + 2 b₀ ln(μ/μ₀)，μ₀ 取 1 GeV 对应波数，g²(μ₀)=4π α_s(1 GeV)。
    当 μ ≪ μ₀ 且 ln 过大导致 1/g²≤0 时，截断到微扰域下限（诚实：非微扰区需另模型）。
    """
    mu = float(max(mu_per_m, 1e-300))
    mu0 = float(max(K_REF_PER_M, 1e-300))
    g2_ref = float(4.0 * np.pi * ALPHA_S_1GEV)
    inv = 1.0 / g2_ref + 2.0 * B0_SU3_PURE_YM * float(np.log(mu / mu0))
    inv = float(max(inv, 1e-6 / max(g2_ref, 1e-30)))
    return float(1.0 / inv)


# μ = 1/λ_micro [m^-1]；Λ_YM = μ exp(-1/(2 b₀ g²(μ)))（与 estimates.shell_bound 中 UV 公式同一 b₀）
_MU_YM_PER_M = float(1.0 / max(lambda_micro, 1e-300))
_G2_AT_MICRO = g_squared_yang_mills_at_mu_per_m(_MU_YM_PER_M)
LAMBDA_YM = float(_MU_YM_PER_M * np.exp(-1.0 / (2.0 * B0_SU3_PURE_YM * max(_G2_AT_MICRO, 1e-30))))
# 便于与高能文献对比的 GeV 标度（Λ 本为质量纲；此处 Λ·m 为无量纲乘 ħc 得 GeV）
LAMBDA_YM_GeV = float(LAMBDA_YM * HBAR_C_GeV_m)

# === 第十七章参数 ===
# 螺旋环面结参数
p_electron = 1  # 极向缠绕数
q_electron = 1  # 角向缠绕数
Omega_0 = 3.9e28  # 背景涡度 (s^-1)

# === 非线性本构 ===


def tau(gamma):
    """真空非线性本构关系 (第5.1.3.1节)"""
    if gamma < 1:
        return G_shear * gamma
    return G_shear * gamma * (1 + 4 * (gamma - 1) ** 2)


def chi2_vacuum():
    """真空二阶非线性张量 (第18.3.1节)"""
    return 8 * G_shear


# === 有效剪切模量 (深度塑性区) ===


def G_eff(gamma):
    """γ ≫ 1 时的有效剪切模量"""
    if gamma < 1:
        return G_shear
    return G_shear / gamma**3


if __name__ == "__main__":
    print("OK constants module loaded")
    print(f"  G_shear = {G_shear:.2e} Pa")
    print(f"  rho_V = {rho_V:.2e} kg/m^3")
    print(f"  Gamma_w = {Gamma_w:.2e} Hz")
    print(f"  lambda_micro = {lambda_micro:.2e} m")
    print(f"  LAMBDA_YM = {LAMBDA_YM:.6e} m^-1  (~ {LAMBDA_YM_GeV:.6e} GeV)")
