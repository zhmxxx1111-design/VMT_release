# VMT_release - 虚体介质论核心求解器

本仓库收录《虚体介质论》相关数值求解脚本，与 `constants.py` 同目录平铺；所有求解器通过将脚本所在目录加入 `sys.path` 后 **`from constants import ...`**，仅依赖当前目录下的 `constants.py`，不依赖其他仓库目录结构。

## 求解器一览（各一行）

| 文件 | 功能 |
|------|------|
| `nuclear_matter_oz_closure.py` | OZ 积分方程 + 指定闭包下的核物质两体关联，液滴型 SEMF 系数（体积/表面/对称/对能）与饱和扫描导出。 |
| `lepton_excited_states_solver.py` | 重轻子（μ、τ）激发态质量自洽迭代（Robin / KPZ / 管几何 + 耗散闭合）。 |
| `pmns_torus_harmonics.py` | 环面谐波与壁畸变味模重叠积分、SVD 提取 PMNS 元与混合角、Jarlskog 型 δ_CP。 |
| `triangle_impedance_lens.py` | 等边三角形阻抗透镜：径向 Helmholtz + 几何光学/Fresnel 链，输出温伯格角链 `sin²θ_W` 等。 |
| `rg_flow_alpha_solver.py` | α 的 RG 二次自洽方程 + 多重散射幂律 `f_ms`，由 `f_suppress` 与泵浦/噪声确定 `r_cut`，输出 `α_eff` 及相对 `1/137.036` 的偏差。 |

## 运行方法

在仓库根目录（与脚本同级）执行：

```bash
python nuclear_matter_oz_closure.py
python lepton_excited_states_solver.py
python pmns_torus_harmonics.py
python triangle_impedance_lens.py
python rg_flow_alpha_solver.py
```

## 预期输出摘要

- **nuclear_matter_oz_closure**：打印饱和密度、OZ+闭包 `g(r)` 峰、液滴系数 `a_v, a_s, a_sym, a_p` 及与实验 SEMF 参考值的相对误差；含分支说明（如 `a_v` 与 15.56 MeV 偏差较大等）。
- **lepton_excited_states_solver**：默认 `dynamic_exp` 下 μ/τ 质量收敛、与实验质量相对误差约 `10⁻⁶` 量级；另打印 `per_lepton` 对照模式。
- **pmns_torus_harmonics**：PMNS 矩阵、奇异值、`sinθ_ij` 与 PDG 参考角对比及 δ_CP；脚本会注明与实验带是否一致。
- **triangle_impedance_lens**：Fresnel 透射、阻抗比、多源/多方法下 `sin²θ_W` 与 0.231 对比；Helmholtz 径向通量诊断。
- **rg_flow_alpha_solver**：`r_cut/r_w`、`f_ms`、`C_eff`、`α_eff`（目标支路 `1/135`）、`α_eff/α_exp` 及相对偏差百分比、方程残差。

## 环境要求

- **Python**：3.9 及以上  
- **依赖**：`numpy`、`scipy`（`triangle_impedance_lens.py` 使用 `scipy.special`；其余以 `numpy` 为主）

安装依赖：

```bash
pip install -r requirements.txt
```
