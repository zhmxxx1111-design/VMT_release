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
lambda_micro = 5.03e-10
G_shear = 4.94e10  # 真空剪切模量 (Pa)
rho_V = 5.50e-7  # 真空惯性浓度 (kg/m^3)
H_initial = c / lambda_micro
H_bare = 1.90e-18

# === 第三章导出 ===
f_C = np.e

# === 第四章导出 ===
omega_W = 1.22e26
Gamma_w = 3.09e24
alpha = 1.0 / 137.035999084

# === 第五章参数 ===
m_e = 9.10938356e-31
r_w = float(hbar / (2.0 * np.pi * m_e * c))
R_e = float(hbar / (m_e * c))

# 电子 Compton 角频率 ω_e = m_e c² / ħ（管壁相干 / 声子尺度常用）
omega_e = float(m_e * c**2 / hbar)

# === Yang–Mills 参考尺度 ===
ALPHA_S_1GEV = 0.33
HBAR_C_J_m = float(hbar * c)
E_1GEV_J = 1.602176634e-10
K_REF_PER_M = float(E_1GEV_J / max(HBAR_C_J_m, 1e-300))
HBAR_C_GeV_m = 197.3269804e-15
B0_SU3_PURE_YM = float(11.0 * 3.0 / (48.0 * np.pi**2))


def g_squared_yang_mills_at_mu_per_m(mu_per_m: float) -> float:
    mu = float(max(mu_per_m, 1e-300))
    mu0 = float(max(K_REF_PER_M, 1e-300))
    g2_ref = float(4.0 * np.pi * ALPHA_S_1GEV)
    inv = 1.0 / g2_ref + 2.0 * B0_SU3_PURE_YM * float(np.log(mu / mu0))
    inv = float(max(inv, 1e-6 / max(g2_ref, 1e-30)))
    return float(1.0 / inv)


_MU_YM_PER_M = float(1.0 / max(lambda_micro, 1e-300))
_G2_AT_MICRO = g_squared_yang_mills_at_mu_per_m(_MU_YM_PER_M)
LAMBDA_YM = float(_MU_YM_PER_M * np.exp(-1.0 / (2.0 * B0_SU3_PURE_YM * max(_G2_AT_MICRO, 1e-300))))
LAMBDA_YM_GeV = float(LAMBDA_YM * HBAR_C_GeV_m)

p_electron = 1
q_electron = 1
Omega_0 = 3.9e28


def tau(gamma: float) -> float:
    if gamma < 1:
        return G_shear * gamma
    return G_shear * gamma * (1 + 4 * (gamma - 1) ** 2)


def chi2_vacuum():
    return 8 * G_shear


def G_eff(gamma: float) -> float:
    if gamma < 1:
        return G_shear
    return G_shear / gamma**3


if __name__ == "__main__":
    print("OK constants module loaded")
    print(f"  omega_e = {omega_e:.6e} Hz")
    print(f"  r_w = {r_w:.6e} m")
