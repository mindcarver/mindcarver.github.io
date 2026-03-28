# 04-Barra 风险模型

> **预计学习时间**：2-3 小时
>
> **难度**：⭐⭐⭐⭐
>
> **核心问题**：机构怎么用结构化的方法管理几百只股票的组合风险？

---

## 从一个直觉出发

假设你管理一个包含 500 只股票的组合。你想知道：

1. **我的组合有多大风险？** 总波动率是多少？
2. **风险从哪里来？** 是大盘风险，还是行业集中，还是某些因子暴露太多？
3. **我的 Alpha 来自哪里？** 是选股能力强，还是碰巧踩中了某个风格？

要回答这些问题，你需要的不是一个简单的"波动率"数字，而是一个**结构化的风险分解框架**。

**Barra 风险模型就是这样的框架**——它把每只股票的风险分解为"因子风险"和"特异性风险"两部分，让组合经理清楚地看到风险从哪里来。

---

## 一、结构化风险模型框架

### 1.1 核心方程

$$r_i = \sum_{k=1}^{K} X_{i,k} \cdot f_k + \varepsilon_i$$

其中：
- $r_i$：股票 $i$ 的收益
- $X_{i,k}$：股票 $i$ 对因子 $k$ 的暴露（因子载荷）
- $f_k$：因子 $k$ 的收益
- $\varepsilon_i$：股票 $i$ 的特异性收益（无法被因子解释的部分）

白话版本：**股票收益 = 它在各个因子上的暴露 x 各个因子的表现 + 个股特有的涨跌**。

### 1.2 为什么需要风险模型？

| 方法 | 优点 | 缺点 |
|------|------|------|
| 直接计算组合波动率 | 简单 | 500 只股票需要 500x500 协方差矩阵（125,000 个参数），估计极其不稳定 |
| 单因子模型（CAPM） | 只需估计 N 个 Beta | 解释力太弱，忽略了行业和风格差异 |
| 结构化风险模型（Barra） | 只需估计 K 个因子收益和 N 个特异性风险 | 因子选择和暴露计算需要额外工作 |

**结构化风险模型的核心优势**：把高维问题（N 只股票的协方差）降维为低维问题（K 个因子的协方差），大幅提升估计稳定性。

### 1.3 Barra 的整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                   Barra 风险模型架构                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  股票收益                                                    │
│    │                                                        │
│    ├── 因子收益（系统性）                                     │
│    │     ├── 行业因子（如金融、科技）                        │
│    │     ├── 风格因子（如价值、规模、动量）                   │
│    │     └── 国家/地区因子（全球模型）                       │
│    │                                                        │
│    └── 特异性收益（个股层面）                                 │
│          └── 无法被因子解释的收益                             │
│                                                             │
│  ─────────────────────────────────────                       │
│                                                             │
│  模型输入：因子暴露矩阵 X                                    │
│  模型输出：因子收益 f、特异性收益 ε、协方差矩阵 Σ            │
│                                                             │
│  ─────────────────────────────────────                       │
│                                                             │
│  应用：                                                      │
│  ├── 组合风险预测                                            │
│  ├── 风险归因（因子 vs 特异）                                │
│  ├── 组合优化（Barra 约束）                                  │
│  └── 绩效归因（收益来源分解）                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 二、因子收益率估计

### 2.1 方法：WLS 横截面回归

Barra 使用**加权最小二乘法（WLS）**估计因子收益，而不是普通 OLS。

**为什么用加权？** 因为大公司对市场的影响更大。市值 1000 亿的公司和市值 10 亿的公司，在截面回归中应该有不同的重要性。

权重通常是市值的平方根：

$$w_i = \sqrt{cap_i}$$

回归方程：

$$\frac{r_i}{w_i} = \sum_{k=1}^{K} \frac{X_{i,k}}{w_i} \cdot f_k + \frac{\varepsilon_i}{w_i}$$

等价于最小化：

$$\min_{f} \sum_{i=1}^{N} w_i^2 (r_i - X_i f)^2$$

### 2.2 Python 代码

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

np.random.seed(42)

# ============================================================
# 模拟数据：200 只股票，60 个月
# ============================================================
n_stocks = 200
n_months = 60
n_factors = 5  # 5 个风格因子

factor_names = ['价值', '规模', '动量', '质量', '低波动']

# 因子暴露（每个股票对每个因子的暴露）
# 模拟时让暴露有一定自相关
factor_exposures = np.zeros((n_stocks, n_factors))
for t_idx in range(n_factors):
    base = np.random.normal(0, 1, n_stocks)
    factor_exposures[:, t_idx] = base

# 添加行业因子（10 个行业）
n_industries = 10
industry_labels = np.random.randint(0, n_industries, n_stocks)
industry_exposures = np.zeros((n_stocks, n_industries))
for i in range(n_stocks):
    industry_exposures[i, industry_labels[i]] = 1

# 合并所有因子
all_exposures = np.column_stack([factor_exposures, industry_exposures])
total_factors = n_factors + n_industries

# 市值（权重）
market_cap = np.random.lognormal(mean=5, sigma=1.5, n_stocks)
weights = np.sqrt(market_cap)  # 市值平方根权重

# ============================================================
# 真实的因子收益
# ============================================================
true_factor_returns = np.array([
    0.005,   # 价值
    0.003,   # 规模
    0.004,   # 动量
    0.002,   # 质量
    0.001,   # 低波动
] + [0.0] * n_industries)  # 行业因子收益设为 0（简化）

# ============================================================
# WLS 横截面回归估计因子收益
# ============================================================
estimated_factor_returns = np.zeros((n_months, total_factors))

for t in range(n_months):
    # 因子收益每月变化（围绕真实值波动）
    monthly_factor_returns = true_factor_returns + np.random.normal(0, 0.005, total_factors)

    # 股票收益 = 因子暴露 x 因子收益 + 特异性收益
    stock_returns = all_exposures @ monthly_factor_returns + np.random.normal(0, 0.03, n_stocks)

    # WLS 回归
    X = all_exposures
    W = np.diag(weights)

    # WLS 矩阵解：(X'WX)^{-1} X'Wy
    XtW = X.T @ W
    XtWX = XtW @ X
    XtWy = XtW @ stock_returns

    # 加入正则化（防止矩阵不可逆）
    reg_lambda = 0.01
    XtWX_reg = XtWX + reg_lambda * np.eye(total_factors)

    f_hat = np.linalg.solve(XtWX_reg, XtWy)
    estimated_factor_returns[t] = f_hat

# ============================================================
# 输出结果
# ============================================================
print("=" * 60)
print("WLS 因子收益估计（月度平均）")
print("=" * 60)
print(f"{'因子':>10} {'真实收益':>10} {'估计收益':>10} {'差异':>10}")
print("-" * 45)
for i, name in enumerate(factor_names):
    true_r = true_factor_returns[i]
    est_r = np.mean(estimated_factor_returns[:, i])
    diff = est_r - true_r
    print(f"{name:>10} {true_r:10.4f} {est_r:10.4f} {diff:10.4f}")
```

---

## 三、特异性风险估计

### 3.1 EWMA 指数加权

股票的特异性风险不是恒定的——市场平静时小，危机时大。

Barra 使用**指数加权移动平均（EWMA）**来估计特异性风险：

$$\sigma_{\varepsilon,i,t}^2 = \lambda \cdot \sigma_{\varepsilon,i,t-1}^2 + (1 - \lambda) \cdot \varepsilon_{i,t}^2$$

其中 $\lambda$ 是衰减因子（通常取 0.9-0.97）。

白话版本：**最近的数据权重大，远期的数据权重小**。这样能更快地捕捉风险水平的变化。

### 3.2 Python 代码

```python
import numpy as np
import pandas as pd

np.random.seed(42)

# ============================================================
# 模拟特异性收益序列
# ============================================================
n_days = 500
n_stocks = 50

# 特异性收益（波动率随时间变化）
true_vol = np.zeros(n_days)
for t in range(n_days):
    # 真实波动率随机变化
    if t == 0:
        true_vol[t] = 0.02
    else:
        true_vol[t] = (0.95 * true_vol[t-1]
                       + 0.05 * 0.02 + np.random.normal(0, 0.003))

# 确保波动率为正
true_vol = np.maximum(true_vol, 0.005)

# 生成特异性收益
idio_returns = np.zeros((n_days, n_stocks))
for s in range(n_stocks):
    idio_returns[:, s] = np.random.normal(0, 1, n_days) * true_vol

# ============================================================
# EWMA 估计特异性风险
# ============================================================
lambda_ewma = 0.94  # 衰减因子（Barra 常用 0.94）

# 用第一只股票演示
stock_idio = idio_returns[:, 0]
ewma_var = np.zeros(n_days)
ewma_vol = np.zeros(n_days)

# 初始化
ewma_var[0] = stock_idio[0] ** 2

for t in range(1, n_days):
    ewma_var[t] = lambda_ewma * ewma_var[t-1] + (1 - lambda_ewma) * stock_idio[t]**2
    ewma_vol[t] = np.sqrt(ewma_var[t])

# ============================================================
# 对比：简单移动平均 vs EWMA
# ============================================================
window = 20  # 20 天移动平均
sma_vol = pd.Series(stock_idio).rolling(window).std().values

# 计算真实波动率的 60 日移动平均（作为基准）
true_vol_smooth = pd.Series(true_vol).rolling(60).mean().values

# 输出（取后 100 天）
print("=" * 55)
print("特异性风险估计：EWMA vs 简单移动平均")
print("=" * 55)
print(f"\n{'方法':>15} {'平均估计值':>12} {'真实值':>10} {'偏差':>10}")
print("-" * 50)

# 对比最后 100 天
idx_start = n_days - 100
ewma_avg = np.mean(ewma_vol[idx_start:])
sma_avg = np.nanmean(sma_vol[idx_start:])
true_avg = np.mean(true_vol_smooth[idx_start:])

print(f"{'EWMA':>15} {ewma_avg:12.4f} {true_avg:10.4f} {ewma_avg-true_avg:10.4f}")
print(f"{'SMA(20日)':>15} {sma_avg:12.4f} {true_avg:10.4f} {sma_avg-true_avg:10.4f}")
print(f"\nEWMA 对波动率变化的反应更快")
```

---

## 四、协方差矩阵

### 4.1 为什么需要协方差矩阵？

组合的风险不仅取决于个股的波动率，还取决于股票之间的**相关性**。

$$\sigma_p^2 = w^T \Sigma w$$

其中 $w$ 是组合权重向量，$\Sigma$ 是 N x N 的协方差矩阵。

对于 500 只股票，$\Sigma$ 有 $500 \times 499 / 2 = 124,750$ 个独立参数需要估计。在 T（比如 60 个月）远小于 N 的情况下，直接估计样本协方差矩阵是非常不稳定的。

### 4.2 因子模型协方差

Barra 的核心思路：通过因子模型把高维协方差分解为低维。

$$\Sigma = X \Sigma_f X^T + D$$

其中：
- $X$：因子暴露矩阵（N x K）
- $\Sigma_f$：因子收益的协方差矩阵（K x K）
- $D$：特异性风险的对角矩阵（N x N）

白话版本：**股票之间的协方差由因子之间的协方差决定。两只股票暴露于相同因子 → 它们的收益相关。**

**参数数量对比**：
- 直接估计：$\frac{N(N+1)}{2} = \frac{500 \times 501}{2} = 125,250$ 个参数
- 因子模型：$\frac{K(K+1)}{2} + N = \frac{20 \times 21}{2} + 500 = 710$ 个参数

参数数量减少了 99.4%！

### 4.3 样本协方差的问题

直接用历史收益计算样本协方差矩阵有以下问题：

1. **维度灾难**：N > T 时矩阵不可逆
2. **估计误差**：即使 N < T，样本协方差矩阵的秩最大为 T-1
3. **特征值分散**：最大的几个特征值"吸收"了太多方差，小的特征值不稳定

### 4.4 收缩估计

一个折中方案：**在样本协方差和结构化协方差之间取加权平均**。

$$\hat{\Sigma} = \delta \cdot \Sigma_{sample} + (1 - \delta) \cdot \Sigma_{target}$$

其中 $\Sigma_{target}$ 可以是单位矩阵、单因子模型或 Barra 因子模型，$\delta$ 是收缩强度（0 到 1 之间）。

### 4.5 Python 代码

```python
import numpy as np
import pandas as pd

np.random.seed(42)

# ============================================================
# 模拟数据：100 只股票，60 个月
# ============================================================
n_stocks = 100
n_months = 60
n_factors = 8

factor_names = ['价值', '规模', '动量', '质量', '低波动', '流动性', '行业1', '行业2']

# 因子暴露
factor_exposures = np.random.normal(0, 0.5, (n_stocks, n_factors))

# 因子收益协方差矩阵
# 让因子之间有相关性
factor_corr = np.array([
    [1.00, -0.30, -0.20, 0.25, -0.15, 0.05, 0.00, 0.00],
    [-0.30, 1.00,  0.10, -0.10, -0.35, -0.05, 0.00, 0.00],
    [-0.20, 0.10,  1.00, -0.05, 0.05, 0.10, 0.00, 0.00],
    [0.25, -0.10, -0.05, 1.00, 0.10, 0.05, 0.00, 0.00],
    [-0.15, -0.35, 0.05, 0.10, 1.00, -0.10, 0.00, 0.00],
    [0.05, -0.05, 0.10, 0.05, -0.10, 1.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.20],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.20, 1.00],
])

# 因子波动率
factor_vols = np.array([0.03, 0.025, 0.035, 0.02, 0.015, 0.025, 0.02, 0.02])
factor_cov = np.outer(factor_vols, factor_vols) * factor_corr

# 确保正定
eigvals = np.linalg.eigvalsh(factor_cov)
if np.min(eigvals) < 0:
    factor_cov += (abs(np.min(eigvals)) + 0.001) * np.eye(n_factors)

# 特异性风险
idio_risk = np.random.uniform(0.015, 0.04, n_stocks)

# ============================================================
# 股票收益协方差矩阵（因子模型）
# ============================================================
# Sigma = X * Sigma_f * X^T + D
X = factor_exposures
Sigma_f = factor_cov
D = np.diag(idio_risk ** 2)

Sigma_barra = X @ Sigma_f @ X.T + D

# ============================================================
# 对比：样本协方差矩阵
# ============================================================
# 模拟月度收益数据
factor_returns_monthly = np.random.multivariate_normal(
    np.zeros(n_factors), Sigma_f, n_months
)
stock_returns_monthly = factor_returns_monthly @ X.T + np.random.normal(
    0, 1, (n_months, n_stocks)
) * idio_risk

Sigma_sample = np.cov(stock_returns_monthly, rowvar=False)

# ============================================================
# 评估协方差矩阵质量
# ============================================================
# 方法：比较两个矩阵的条件数和特征值分布
eig_barra = np.linalg.eigvalsh(Sigma_barra)
eig_sample = np.linalg.eigvalsh(Sigma_sample)

cond_barra = np.max(eig_barra) / np.min(eig_barra)
cond_sample = np.max(eig_sample) / np.min(np.abs(eig_sample[eig_sample > 0]))

print("=" * 60)
print("协方差矩阵对比")
print("=" * 60)
print(f"\n{'方法':>15} {'条件数':>12} {'最大特征值':>12} {'最小特征值':>12}")
print("-" * 55)
print(f"{'Barra 因子模型':>15} {cond_barra:12.2f} {np.max(eig_barra):12.6f} "
      f"{np.min(eig_barra):12.6f}")
print(f"{'样本协方差':>15} {cond_sample:12.2f} {np.max(eig_sample):12.6f} "
      f"{np.min(np.abs(eig_sample)):12.6f}")

# 前五大特征值解释的方差比例
total_var_barra = np.sum(eig_barra)
top5_ratio_barra = np.sum(sorted(eig_barra, reverse=True)[:5]) / total_var_barra
top5_ratio_sample = np.sum(sorted(eig_sample, reverse=True)[:5]) / np.sum(eig_sample)

print(f"\n前 5 大特征值解释方差比例:")
print(f"  Barra 因子模型: {top5_ratio_barra:.1%}")
print(f"  样本协方差:     {top5_ratio_sample:.1%}")
print(f"\n说明: Barra 模型的特征值分布更均匀，估计更稳定")
```

---

## 五、组合风险归因

### 5.1 核心思想

你的组合收益可以分解为：

$$r_p = \sum_{k} h_{p,k} \cdot f_k + \varepsilon_p$$

其中 $h_{p,k} = \sum_i w_i \cdot X_{i,k}$ 是组合对因子 $k$ 的总暴露（组合权重 x 因子暴露）。

**风险归因**就是回答：**组合的总风险中，有多少来自因子，多少来自特异性？每个因子贡献了多少？**

### 5.2 风险分解

组合方差可以分解为：

$$\sigma_p^2 = \underbrace{h_p^T \Sigma_f h_p}_{\text{因子风险}} + \underbrace{\sum_i w_i^2 \sigma_{\varepsilon,i}^2}_{\text{特异性风险}}$$

因子风险占比 = 因子风险 / 总风险

### 5.3 Python 代码

```python
import numpy as np
import pandas as pd

np.random.seed(42)

# ============================================================
# 模拟组合数据
# ============================================================
n_stocks = 100
n_factors = 6
factor_names = ['市场', '价值', '规模', '动量', '质量', '低波动']

# 因子暴露
X = np.random.normal(0, 0.5, (n_stocks, n_factors))

# 组合权重（等权，简化）
w = np.ones(n_stocks) / n_stocks

# 因子收益协方差
factor_vols = np.array([0.04, 0.03, 0.025, 0.035, 0.02, 0.018])
factor_corr = np.array([
    [1.00, 0.10, 0.30, 0.05, 0.15, -0.30],
    [0.10, 1.00, -0.30, -0.20, 0.25, -0.15],
    [0.30, -0.30, 1.00, 0.10, -0.10, -0.35],
    [0.05, -0.20, 0.10, 1.00, 0.05, 0.05],
    [0.15, 0.25, -0.10, 0.05, 1.00, 0.10],
    [-0.30, -0.15, -0.35, 0.05, 0.10, 1.00],
])
Sigma_f = np.outer(factor_vols, factor_vols) * factor_corr

# 确保正定
eigvals = np.linalg.eigvalsh(Sigma_f)
if np.min(eigvals) < 0:
    Sigma_f += (abs(np.min(eigvals)) + 0.001) * np.eye(n_factors)

# 特异性风险
idio_risk = np.random.uniform(0.015, 0.04, n_stocks)

# ============================================================
# 组合因子暴露
# ============================================================
# h_k = sum_i w_i * X_{i,k}
h_p = w @ X  # 组合对各因子的暴露

print("=" * 55)
print("组合因子暴露分析")
print("=" * 55)
for k, name in enumerate(factor_names):
    print(f"  {name:>8}: {h_p[k]:.4f}")

# ============================================================
# 风险归因
# ============================================================
# 因子风险
factor_risk = h_p @ Sigma_f @ h_p

# 特异性风险
spec_risk = np.sum(w**2 * idio_risk**2)

# 总风险
total_risk = factor_risk + spec_risk

# 各因子贡献的风险
marginal_risk = Sigma_f @ h_p  # 边际风险贡献
factor_contrib = h_p * marginal_risk  # 各因子风险贡献

# 转为年化波动率
total_vol = np.sqrt(total_risk) * np.sqrt(12)
factor_vol = np.sqrt(factor_risk) * np.sqrt(12)
spec_vol = np.sqrt(spec_risk) * np.sqrt(12)

print(f"\n{'='*55}")
print("组合风险分解")
print(f"{'='*55}")
print(f"\n  总风险（年化波动率）: {total_vol:.2%}")
print(f"  因子风险:             {factor_vol:.2%} ({factor_risk/total_risk:.1%})")
print(f"  特异性风险:           {spec_vol:.2%} ({spec_risk/total_risk:.1%})")

print(f"\n各因子风险贡献:")
print(f"{'因子':>10} {'暴露':>10} {'风险贡献':>12} {'占比':>10}")
print("-" * 45)
for k, name in enumerate(factor_names):
    contrib = factor_contrib[k]
    pct = contrib / total_risk * 100
    print(f"{name:>10} {h_p[k]:10.4f} {contrib:12.6f} {pct:9.1f}%")
```

### 5.4 风险归因的实战价值

| 应用 | 说明 |
|------|------|
| 识别集中风险 | 如果"规模因子"贡献了 60% 的风险，说明组合过度暴露于小盘股 |
| 控制风格偏离 | 如果你想做行业中性，需要确保行业因子的暴露接近零 |
| 理解收益来源 | "我的组合涨了 3%，其中 2% 来自市场，1% 来自价值因子" |
| 设定风险预算 | 给每个因子分配最大风险预算，防止过度集中 |

---

## 六、Alpha 对齐（Alpha Alignment）

### 6.1 核心思想

在实际管理组合时，你的 Alpha 信号（预测收益）和风险模型给出的因子暴露之间可能不一致。

**Alpha 对齐**就是把 Alpha 信号中"和风险模型因子重合的部分"分离出来，只保留纯 Alpha（纯特异性信号）。

### 6.2 方法

$$\alpha_i^{pure} = \alpha_i - \sum_k \hat{\lambda}_k \cdot X_{i,k}$$

其中 $\hat{\lambda}_k$ 是用所有 Alpha 值对因子暴露做回归得到的系数。

白话版本：**把 Alpha 中可以用因子解释的部分去掉，剩下的才是"真正的 Alpha"。**

### 6.3 Python 代码

```python
import numpy as np
import statsmodels.api as sm

np.random.seed(42)

# ============================================================
# 模拟数据
# ============================================================
n_stocks = 300
n_factors = 5
factor_names = ['价值', '规模', '动量', '质量', '低波动']

# 原始 Alpha 信号（研究员给出的预测收益）
raw_alpha = np.random.normal(0.01, 0.005, n_stocks)

# 因子暴露
X = np.random.normal(0, 0.5, (n_stocks, n_factors))

# Alpha 中混入了因子暴露
# 比如 Alpha 信号和规模因子相关（倾向于给小公司更高的 Alpha）
raw_alpha = raw_alpha + 0.003 * X[:, 1]  # 规模因子的系统性偏差

# ============================================================
# Alpha 对齐：回归剔除因子成分
# ============================================================
X_reg = sm.add_constant(X)
model = sm.OLS(raw_alpha, X_reg).fit()

# 纯 Alpha = 原始 Alpha - 因子成分
factor_component = X_reg @ model.params
pure_alpha = raw_alpha - factor_component + model.params[0]  # 保留截距

# ============================================================
# 输出结果
# ============================================================
print("=" * 60)
print("Alpha 对齐")
print("=" * 60)

print(f"\n原始 Alpha 对因子的回归系数:")
for k, name in enumerate(factor_names):
    print(f"  {name:>8}: {model.params[k+1]:.6f} "
          f"(p={model.pvalues[k+1]:.4f})")

print(f"\n对比:")
print(f"  原始 Alpha 均值:   {np.mean(raw_alpha):.6f}")
print(f"  原始 Alpha 标准差: {np.std(raw_alpha):.6f}")
print(f"  纯 Alpha 均值:     {np.mean(pure_alpha):.6f}")
print(f"  纯 Alpha 标准差:   {np.std(pure_alpha):.6f}")
print(f"\n说明: 纯 Alpha 的标准差更小")
print(f"因为去掉了因子成分，Alpha 更'纯粹'")
```

---

## 七、完整案例：构建简化版 Barra 风险模型

下面用一个完整的例子，把前面的所有环节串起来。

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

np.random.seed(42)

# ============================================================
# 参数设置
# ============================================================
n_stocks = 200
n_months = 120  # 10 年
n_style_factors = 5
n_industries = 8

# ============================================================
# 第 1 步：定义因子
# ============================================================
style_names = ['价值', '规模', '动量', '质量', '低波动']
industry_names = [f'行业{i}' for i in range(n_industries)]

# 因子暴露
style_exposures = np.zeros((n_stocks, n_style_factors))
for t in range(n_months):
    if t == 0:
        for k in range(n_style_factors):
            style_exposures[:, k] = np.random.normal(0, 1, n_stocks)
    else:
        for k in range(n_style_factors):
            style_exposures[:, k] = (
                0.95 * style_exposures[:, k]
                + np.random.normal(0, 0.2, n_stocks)
            )

industry_labels = np.random.randint(0, n_industries, n_stocks)
industry_exposures = np.zeros((n_stocks, n_industries))
for i in range(n_stocks):
    industry_exposures[i, industry_labels[i]] = 1

# 合并所有因子暴露
all_exposures = np.column_stack([style_exposures, industry_exposures])
total_factors = n_style_factors + n_industries
all_factor_names = style_names + industry_names

# 市值
market_cap = np.random.lognormal(mean=5, sigma=1.5, n_stocks)
weights = np.sqrt(market_cap)

# ============================================================
# 第 2 步：每月估计因子收益（WLS）
# ============================================================
all_factor_returns = np.zeros((n_months, total_factors))
all_residuals = np.zeros((n_months, n_stocks))
all_stock_returns = np.zeros((n_months, n_stocks))

# 真实的风格因子溢价
true_style_premiums = np.array([0.004, 0.002, 0.005, 0.003, 0.001])

for t in range(n_months):
    # 因子收益（围绕真实溢价波动）
    style_returns = true_style_premiums + np.random.normal(0, 0.003, n_style_factors)
    industry_returns = np.random.normal(0, 0.002, n_industries)
    monthly_factor_returns = np.concatenate([style_returns, industry_returns])

    # 股票收益
    stock_returns = all_exposures @ monthly_factor_returns + np.random.normal(0, 0.025, n_stocks)
    all_stock_returns[t] = stock_returns

    # WLS 回归
    W = np.diag(weights)
    XtW = all_exposures.T @ W
    XtWX = XtW @ all_exposures + 0.01 * np.eye(total_factors)
    XtWy = XtW @ stock_returns
    f_hat = np.linalg.solve(XtWX, XtWy)

    all_factor_returns[t] = f_hat
    all_residuals[t] = stock_returns - all_exposures @ f_hat

# ============================================================
# 第 3 步：估计因子收益协方差矩阵
# ============================================================
Sigma_f = np.cov(all_factor_returns.T)

# Newey-West 调整（简化版：不实际计算，用样本协方差近似）
print("=" * 55)
print("因子收益协方差矩阵（年化）")
print("=" * 55)
Sigma_f_annual = Sigma_f * 12
cov_df = pd.DataFrame(
    Sigma_f_annual[:n_style_factors, :n_style_factors],
    index=style_names,
    columns=style_names
)
print(cov_df.round(6))

# ============================================================
# 第 4 步：估计特异性风险（EWMA）
# ============================================================
lambda_ewma = 0.94
ewma_var = np.zeros((n_months, n_stocks))
ewma_var[0] = all_residuals[0] ** 2

for t in range(1, n_months):
    ewma_var[t] = lambda_ewma * ewma_var[t-1] + (1 - lambda_ewma) * all_residuals[t]**2

# 最终特异性风险（最后一个月的 EWMA 波动率）
idio_vol = np.sqrt(ewma_var[-1])

# ============================================================
# 第 5 步：构建协方差矩阵
# ============================================================
# 获取最后一个月的因子暴露
X_current = all_exposures
D = np.diag(idio_vol ** 2)
Sigma_stock = X_current @ Sigma_f @ X_current.T + D

# ============================================================
# 第 6 步：组合风险归因
# ============================================================
# 等权组合
w = np.ones(n_stocks) / n_stocks

# 组合因子暴露
h_p = w @ all_exposures

# 因子风险
factor_risk = h_p @ Sigma_f @ h_p
spec_risk = np.sum(w**2 * idio_vol**2)
total_risk = factor_risk + spec_risk

total_vol = np.sqrt(total_risk) * np.sqrt(12)
factor_vol = np.sqrt(factor_risk) * np.sqrt(12)
spec_vol = np.sqrt(spec_risk) * np.sqrt(12)

print(f"\n{'='*55}")
print("简化版 Barra 风险模型 — 组合风险归因")
print(f"{'='*55}")
print(f"\n组合: 等权 {n_stocks} 只股票")
print(f"\n总风险（年化波动率）: {total_vol:.2%}")
print(f"  因子风险:           {factor_vol:.2%} ({factor_risk/total_risk:.1%})")
print(f"  特异性风险:         {spec_vol:.2%} ({spec_risk/total_risk:.1%})")

print(f"\n风格因子风险贡献:")
marginal = Sigma_f @ h_p
factor_contrib = h_p * marginal
for k, name in enumerate(style_names):
    pct = factor_contrib[k] / total_risk * 100
    print(f"  {name:>8}: 暴露={h_p[k]:.4f}, "
          f"风险贡献={factor_contrib[k]:.6f}, 占比={pct:.1f}%")

# ============================================================
# 第 7 步：预测个股风险
# ============================================================
# 预测每只股票的年化波动率
pred_vols = np.sqrt(np.diag(Sigma_stock)) * np.sqrt(12)

print(f"\n个股风险预测（年化波动率）:")
print(f"  平均: {np.mean(pred_vols):.2%}")
print(f"  中位数: {np.median(pred_vols):.2%}")
print(f"  范围: [{np.min(pred_vols):.2%}, {np.max(pred_vols):.2%}]")
print(f"\n低风险股票数（< 15%）: {np.sum(pred_vols < 0.15)}")
print(f"高风险股票数（> 30%）: {np.sum(pred_vols > 0.30)}")
```

---

## 八、Barra 模型的实际应用要点

| 应用 | 关键点 |
|------|--------|
| 组合优化 | 用 Barra 协方差矩阵替代样本协方差，优化更稳定 |
| 风险限制 | 限制组合对特定因子的暴露（如"价值因子暴露不超过 +1 标准差"） |
| 绩效归因 | 分解组合收益为"因子贡献"和"选股贡献" |
| 压力测试 | 通过因子场景分析（如"如果价值因子跌 5%，组合会怎样？"） |

### Barra 模型的局限性

1. **因子暴露的计算依赖数据质量**：财务数据的滞后、异常值、会计准则变化都会影响
2. **历史协方差可能不反映未来**：市场 Regime 变化时，因子之间的相关性会变
3. **模型风险**：选择了错误的因子或遗漏了重要因子，风险预测就不准
4. **更新频率**：月度更新可能不够快，日频更新则计算量大幅增加

---

## 小结

| 概念 | 要点 |
|------|------|
| 结构化风险模型 | 收益 = 因子收益 + 特异性收益 |
| WLS 估计 | 用市值加权估计因子收益 |
| EWMA 特异性风险 | 指数加权，对变化反应更快 |
| 因子模型协方差 | Sigma = X * Sigma_f * X' + D，把高维问题降维 |
| 风险归因 | 分解组合风险为因子风险和特异性风险 |
| Alpha 对齐 | 去掉 Alpha 中的因子成分，保留纯 Alpha |

**Barra 风险模型是机构量化投资的"基础设施"**。不管你用什么策略赚钱，你最终都需要用它来管理风险、解释收益、优化组合。

→ 本模块完结。建议继续学习：[组合管理](../组合管理/01-组合构建理论.md) —— 如何用风险模型做组合优化
