# IC/Rank IC 评估指标

## 1. IC（Information Coefficient）基础

### 1.1 IC的定义

**数学定义**

IC（Information Coefficient，信息系数）是预测值与实际值的Pearson相关系数：

$$ IC = \rho(\hat{y}, y) = \frac{Cov(\hat{y}, y)}{\sigma_{\hat{y}} \sigma_y} $$

其中：
- $\hat{y}$ 是预测值（预测收益率）
- $y$ 是实际值（实际收益率）
- $\rho$ 是Pearson相关系数
- $Cov$ 是协方差
- $\sigma$ 是标准差

**取值范围**

- IC ∈ [-1, 1]
- IC = 1：完全正相关，预测完全准确
- IC = 0：无相关性，预测无效
- IC = -1：完全负相关，预测完全相反

**计算示例**

```python
import numpy as np
from scipy.stats import pearsonr

# 假设我们有预测值和实际值
y_pred = np.array([0.01, 0.02, -0.01, 0.03, 0.005])
y_true = np.array([0.015, 0.025, -0.005, 0.035, 0.01])

# 计算IC
ic = pearsonr(y_pred, y_true)[0]
print(f"IC = {ic:.4f}")

# 手动计算
def calculate_ic(y_pred, y_true):
    mean_pred = np.mean(y_pred)
    mean_true = np.mean(y_true)

    std_pred = np.std(y_pred)
    std_true = np.std(y_true)

    covariance = np.mean((y_pred - mean_pred) * (y_true - mean_true))
    ic = covariance / (std_pred * std_true)

    return ic

ic_manual = calculate_ic(y_pred, y_true)
print(f"手动计算 IC = {ic_manual:.4f}")
```

### 1.2 IC在量化中的意义

**量化投资的核心问题**

量化投资的核心是预测股票的相对强弱，而非精确的收益率预测。

**IC的优势**

1. **非线性不敏感**：只关心排序，不关心绝对值
2. **稳健性**：对异常值不敏感
3. **可解释性**：直接衡量预测能力

**IC与收益的关系**

假设我们根据预测值构建多空组合：

$$ \text{Return}_{t} = \frac{1}{N_{\text{long}}} \sum_{i \in \text{long}} r_{i,t} - \frac{1}{N_{\text{short}}} \sum_{i \in \text{short}} r_{i,t} $$

其中多空组合根据预测值排序选择。

理论上，IC越高，组合收益越高：

$$ E[\text{Return}] \propto IC \times \sigma_r $$

其中 $\sigma_r$ 是收益率的标准差。

## 2. Rank IC

### 2.1 Rank IC的定义

**数学定义**

Rank IC是预测值排序与实际值排序的Spearman秩相关系数：

$$ \text{Rank IC} = \rho(\text{rank}(\hat{y}), \text{rank}(y)) $$

其中 $\text{rank}(\cdot)$ 是排名函数。

**计算示例**

```python
from scipy.stats import spearmanr

# 假设我们有预测值和实际值
y_pred = np.array([0.01, 0.02, -0.01, 0.03, 0.005])
y_true = np.array([0.015, 0.025, -0.005, 0.035, 0.01])

# 计算Rank IC
rank_ic = spearmanr(y_pred, y_true)[0]
print(f"Rank IC = {rank_ic:.4f}")

# 手动计算
def calculate_rank_ic(y_pred, y_true):
    rank_pred = pd.Series(y_pred).rank()
    rank_true = pd.Series(y_true).rank()

    # 使用Pearson相关系数计算排名的相关性
    ic = pearsonr(rank_pred, rank_true)[0]

    return ic

rank_ic_manual = calculate_rank_ic(y_pred, y_true)
print(f"手动计算 Rank IC = {rank_ic_manual:.4f}")
```

### 2.2 IC与Rank IC的区别

**区别对比**

| 特性 | IC | Rank IC |
|------|-----|---------|
| 相关系数类型 | Pearson | Spearman |
| 敏感性 | 对数值敏感 | 只对排序敏感 |
| 异常值 | 敏感 | 不敏感 |
| 适用场景 | 线性关系 | 单调关系 |
| 计算复杂度 | O(N) | O(N log N) |

**数值示例**

```python
# 示例：异常值的影响
y_pred = np.array([0.01, 0.02, -0.01, 0.03, 0.005])
y_true = np.array([0.015, 0.025, -0.005, 0.035, 0.01])

# 加入异常值
y_pred_outlier = np.array([0.01, 0.02, -0.01, 0.03, 10.0])  # 第5个值异常大
y_true_outlier = np.array([0.015, 0.025, -0.005, 0.035, 0.01])

# 计算IC
ic_normal = pearsonr(y_pred, y_true)[0]
ic_outlier = pearsonr(y_pred_outlier, y_true_outlier)[0]

# 计算Rank IC
rank_ic_normal = spearmanr(y_pred, y_true)[0]
rank_ic_outlier = spearmanr(y_pred_outlier, y_true_outlier)[0]

print(f"正常数据: IC = {ic_normal:.4f}, Rank IC = {rank_ic_normal:.4f}")
print(f"异常数据: IC = {ic_outlier:.4f}, Rank IC = {rank_ic_outlier:.4f}")

# 结果：IC受异常值影响大，Rank IC几乎不变
```

**量化场景的选择**

在量化投资中，通常更关注Rank IC，因为：

1. 我们关注的是股票的相对排名
2. 收益率存在异常值（涨跌停、停牌等）
3. 排序比精确值更稳定

## 3. IC的统计显著性检验

### 3.1 t检验

**假设检验**

- H0（原假设）：IC = 0（预测无能力）
- H1（备择假设）：IC ≠ 0（预测有能力）

**检验统计量**

$$ t = \frac{IC \times \sqrt{N - 2}}{\sqrt{1 - IC^2}} $$

其中 $N$ 是样本数量。

**代码实现**

```python
from scipy.stats import t

def ic_t_test(ic, n_samples, alpha=0.05):
    """
    IC的t检验

    参数:
        ic: IC值
        n_samples: 样本数量
        alpha: 显著性水平

    返回:
        t_statistic: t统计量
        p_value: p值
        is_significant: 是否显著
    """
    # 计算t统计量
    t_statistic = ic * np.sqrt(n_samples - 2) / np.sqrt(1 - ic ** 2)

    # 计算p值（双尾检验）
    p_value = 2 * (1 - t.cdf(abs(t_statistic), df=n_samples - 2))

    # 判断显著性
    is_significant = p_value < alpha

    return t_statistic, p_value, is_significant

# 示例
ic = 0.05
n_samples = 252  # 一年的交易日

t_stat, p_val, sig = ic_t_test(ic, n_samples)

print(f"IC = {ic:.4f}")
print(f"t统计量 = {t_stat:.4f}")
print(f"p值 = {p_val:.4f}")
print(f"是否显著: {sig}")
```

### 3.2 IC的置信区间

**置信区间计算**

IC的置信区间可以通过Fisher变换计算：

$$ z = \frac{1}{2} \ln\left(\frac{1 + IC}{1 - IC}\right) $$

z的标准误差：

$$ SE_z = \frac{1}{\sqrt{N - 3}} $$

置信区间：

$$ CI_{IC} = \tanh\left(z \pm z_{1-\alpha/2} \times SE_z\right) $$

**代码实现**

```python
from scipy.stats import norm

def ic_confidence_interval(ic, n_samples, alpha=0.05):
    """
    IC的置信区间

    参数:
        ic: IC值
        n_samples: 样本数量
        alpha: 显著性水平

    返回:
        (lower, upper): 置信区间
    """
    # Fisher变换
    z = 0.5 * np.log((1 + ic) / (1 - ic))

    # 计算标准误差
    se_z = 1 / np.sqrt(n_samples - 3)

    # 计算置信区间
    z_critical = norm.ppf(1 - alpha / 2)
    z_lower = z - z_critical * se_z
    z_upper = z + z_critical * se_z

    # 反Fisher变换
    ic_lower = np.tanh(z_lower)
    ic_upper = np.tanh(z_upper)

    return ic_lower, ic_upper

# 示例
ic = 0.05
n_samples = 252

lower, upper = ic_confidence_interval(ic, n_samples)
print(f"IC = {ic:.4f}")
print(f"95%置信区间: [{lower:.4f}, {upper:.4f}]")
```

## 4. IC的时序分析

### 4.1 滚动IC

**滚动IC的定义**

滚动IC是在固定时间窗口内计算的IC序列，用于分析IC的稳定性。

**代码实现**

```python
def rolling_ic(y_pred, y_true, window=20):
    """
    滚动IC计算

    参数:
        y_pred: 预测值序列，shape=[n_samples]
        y_true: 实际值序列，shape=[n_samples]
        window: 滚动窗口大小

    返回:
        ic_series: IC序列
    """
    n_samples = len(y_pred)
    ic_series = []

    for i in range(window, n_samples + 1):
        window_pred = y_pred[i-window:i]
        window_true = y_true[i-window:i]

        ic = pearsonr(window_pred, window_true)[0]
        ic_series.append(ic)

    return np.array(ic_series)

# 示例
y_pred = np.random.randn(500)
y_true = y_pred * 0.5 + np.random.randn(500) * 0.5

ic_series = rolling_ic(y_pred, y_true, window=20)

print(f"平均IC: {np.mean(ic_series):.4f}")
print(f"IC标准差: {np.std(ic_series):.4f}")
print(f"IC最大值: {np.max(ic_series):.4f}")
print(f"IC最小值: {np.min(ic_series):.4f}")
```

**滚动IC可视化**

```python
import matplotlib.pyplot as plt

def plot_rolling_ic(y_pred, y_true, window=20):
    """
    绘制滚动IC
    """
    ic_series = rolling_ic(y_pred, y_true, window)

    plt.figure(figsize=(12, 6))
    plt.plot(ic_series, label='Rolling IC')
    plt.axhline(y=0, color='r', linestyle='--', label='Zero Line')
    plt.axhline(y=np.mean(ic_series), color='g', linestyle='--', label='Mean IC')
    plt.xlabel('Time')
    plt.ylabel('IC')
    plt.title(f'Rolling IC (Window={window})')
    plt.legend()
    plt.grid(True)
    plt.show()

# 使用示例
plot_rolling_ic(y_pred, y_true, window=20)
```

### 4.2 IC衰减分析

**IC衰减的定义**

IC衰减是指预测值与未来不同周期实际值的IC，用于分析预测的时效性。

**代码实现**

```python
def ic_decay(y_pred, y_true, max_lag=10):
    """
    IC衰减分析

    参数:
        y_pred: 预测值序列
        y_true: 实际值序列
        max_lag: 最大滞后阶数

    返回:
        ic_decay_series: IC衰减序列
    """
    ic_decay_series = []

    for lag in range(max_lag + 1):
        # 对齐数据
        pred = y_pred[:len(y_pred) - lag]
        true = y_true[lag:len(y_true)]

        # 计算IC
        ic = pearsonr(pred, true)[0]
        ic_decay_series.append(ic)

    return np.array(ic_decay_series)

# 示例
y_pred = np.random.randn(500)
y_true = np.random.randn(500)

ic_decay_series = ic_decay(y_pred, y_true, max_lag=10)

for lag, ic in enumerate(ic_decay_series):
    print(f"Lag {lag}: IC = {ic:.4f}")
```

**IC衰减可视化**

```python
def plot_ic_decay(y_pred, y_true, max_lag=10):
    """
    绘制IC衰减曲线
    """
    ic_decay_series = ic_decay(y_pred, y_true, max_lag)

    plt.figure(figsize=(10, 6))
    plt.bar(range(max_lag + 1), ic_decay_series)
    plt.xlabel('Lag')
    plt.ylabel('IC')
    plt.title('IC Decay Analysis')
    plt.grid(True, axis='y')
    plt.show()

# 使用示例
plot_ic_decay(y_pred, y_true, max_lag=10)
```

## 5. IR（Information Ratio）

### 5.1 IR的定义

**数学定义**

IR（Information Ratio，信息比率）是IC的均值除以IC的标准差：

$$ IR = \frac{E[IC]}{\sigma_{IC}} $$

其中：
- $E[IC]$ 是IC的期望（均值）
- $\sigma_{IC}$ 是IC的标准差

**意义**

IR衡量预测能力的稳定性：
- IR高：IC均值高且稳定
- IR低：IC均值低或不稳定

**代码实现**

```python
def calculate_ir(ic_series):
    """
    计算IR

    参数:
        ic_series: IC序列

    返回:
        ir: 信息比率
        ic_mean: IC均值
        ic_std: IC标准差
    """
    ic_mean = np.mean(ic_series)
    ic_std = np.std(ic_series, ddof=1)  # 使用样本标准差

    if ic_std == 0:
        ir = 0
    else:
        ir = ic_mean / ic_std

    return ir, ic_mean, ic_std

# 示例
ic_series = np.array([0.05, 0.03, 0.07, 0.04, 0.06])

ir, ic_mean, ic_std = calculate_ir(ic_series)

print(f"IC均值: {ic_mean:.4f}")
print(f"IC标准差: {ic_std:.4f}")
print(f"IR: {ir:.4f}")
```

### 5.2 IR与IC的关系

**关系分析**

$$ IR = \frac{E[IC]}{\sigma_{IC}} $$

- **IC高，IR高**：预测能力强且稳定
- **IC高，IR低**：预测能力强但不稳定
- **IC低，IR高**：预测能力弱但稳定
- **IC低，IR低**：预测能力弱且不稳定

**示例对比**

```python
# 场景1：IC高且稳定
ic_series_1 = np.array([0.05, 0.05, 0.05, 0.05, 0.05])
ir_1, mean_1, std_1 = calculate_ir(ic_series_1)
print(f"场景1 - IC={mean_1:.4f}, IR={ir_1:.4f} (高且稳定)")

# 场景2：IC高但不稳定
ic_series_2 = np.array([0.10, 0.00, 0.10, 0.00, 0.10])
ir_2, mean_2, std_2 = calculate_ir(ic_series_2)
print(f"场景2 - IC={mean_2:.4f}, IR={ir_2:.4f} (高但不稳定)")

# 场景3：IC低但稳定
ic_series_3 = np.array([0.02, 0.02, 0.02, 0.02, 0.02])
ir_3, mean_3, std_3 = calculate_ir(ic_series_3)
print(f"场景3 - IC={mean_3:.4f}, IR={ir_3:.4f} (低但稳定)")
```

### 5.3 年化IR

**年化IR的计算**

如果IC按日计算，年化IR为：

$$ IR_{annual} = IR_{daily} \times \sqrt{252} $$

其中252是每年的交易日数量。

**代码实现**

```python
def annualized_ir(ic_series, periods_per_year=252):
    """
    年化IR

    参数:
        ic_series: IC序列
        periods_per_year: 每年周期数

    返回:
        ir_annual: 年化IR
    """
    ir, ic_mean, ic_std = calculate_ir(ic_series)
    ir_annual = ir * np.sqrt(periods_per_year)

    return ir_annual

# 示例：每日IC
daily_ic_series = np.random.randn(252) * 0.01 + 0.03

ir_annual = annualized_ir(daily_ic_series)
print(f"年化IR: {ir_annual:.4f}")
```

## 6. IC在不同子集上的表现

### 6.1 按市场状态分析IC

**市场状态分类**

```python
def analyze_ic_by_market_regime(y_pred, y_true, regimes):
    """
    按市场状态分析IC

    参数:
        y_pred: 预测值
        y_true: 实际值
        regimes: 市场状态（-1=熊市, 0=震荡, 1=牛市）

    返回:
        dict: {regime: {'ic': ic, 'n_samples': n_samples}}
    """
    results = {}

    for regime in [-1, 0, 1]:
        mask = regimes == regime
        y_pred_regime = y_pred[mask]
        y_true_regime = y_true[mask]

        if len(y_pred_regime) > 0:
            ic = pearsonr(y_pred_regime, y_true_regime)[0]
            results[regime] = {
                'ic': ic,
                'n_samples': len(y_pred_regime)
            }

    return results

# 示例
regimes = np.random.choice([-1, 0, 1], size=len(y_pred), p=[0.2, 0.6, 0.2])

results = analyze_ic_by_market_regime(y_pred, y_true, regimes)

for regime in [-1, 0, 1]:
    regime_name = {-1: '熊市', 0: '震荡市', 1: '牛市'}[regime]
    if regime in results:
        print(f"{regime_name}: IC={results[regime]['ic']:.4f}, 样本数={results[regime]['n_samples']}")
```

### 6.2 按行业分析IC

```python
def analyze_ic_by_industry(y_pred, y_true, industry_codes):
    """
    按行业分析IC

    参数:
        y_pred: 预测值
        y_true: 实际值
        industry_codes: 行业代码

    返回:
        dict: {industry: {'ic': ic, 'n_samples': n_samples}}
    """
    results = {}

    for code in np.unique(industry_codes):
        mask = industry_codes == code
        y_pred_industry = y_pred[mask]
        y_true_industry = y_true[mask]

        if len(y_pred_industry) > 0:
            ic = pearsonr(y_pred_industry, y_true_industry)[0]
            results[code] = {
                'ic': ic,
                'n_samples': len(y_pred_industry)
            }

    return results

# 示例
industry_codes = np.random.choice([1, 2, 3, 4, 5], size=len(y_pred))

results = analyze_ic_by_industry(y_pred, y_true, industry_codes)

for code, result in results.items():
    print(f"行业{code}: IC={result['ic']:.4f}, 样本数={result['n_samples']}")
```

## 7. IC在模型评估中的应用

### 7.1 交叉验证中的IC评估

```python
from sklearn.model_selection import TimeSeriesSplit

def cross_validate_ic(X, y, params, model, n_splits=5):
    """
    时间序列交叉验证，评估IC

    参数:
        X: 特征矩阵
        y: 目标变量
        params: 模型参数
        model: 模型对象
        n_splits: 折数

    返回:
        ic_scores: 每折的IC得分
        models: 训练的模型列表
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    ic_scores = []
    models = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"Fold {fold + 1}/{n_splits}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 训练模型
        model.fit(X_train, y_train)

        # 预测
        y_pred = model.predict(X_val)

        # 计算IC
        ic = pearsonr(y_pred, y_val)[0]
        ic_scores.append(ic)
        models.append(model)

        print(f"  Val IC: {ic:.4f}")

    return ic_scores, models

# 示例
from lightgbm import LGBMRegressor

params = {
    'objective': 'regression',
    'num_leaves': 31,
    'learning_rate': 0.05,
}

model = LGBMRegressor(**params)
ic_scores, models = cross_validate_ic(X, y, params, model, n_splits=5)

print(f"\n平均IC: {np.mean(ic_scores):.4f}")
print(f"IC标准差: {np.std(ic_scores):.4f}")
```

### 7.2 IC与模型选择

```python
def select_model_by_ic(X_train, y_train, X_val, y_val, param_grid, model_class):
    """
    基于IC选择最佳模型

    参数:
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        param_grid: 参数网格
        model_class: 模型类

    返回:
        best_model: 最佳模型
        best_params: 最佳参数
        best_ic: 最佳IC
    """
    best_model = None
    best_params = None
    best_ic = -np.inf

    from itertools import product

    keys = param_grid.keys()
    values = param_grid.values()

    for combination in product(*values):
        params = dict(zip(keys, combination))

        # 训练模型
        model = model_class(**params)
        model.fit(X_train, y_train)

        # 预测
        y_pred = model.predict(X_val)

        # 计算IC
        ic = pearsonr(y_pred, y_val)[0]

        if ic > best_ic:
            best_ic = ic
            best_model = model
            best_params = params

        print(f"Params: {params}, IC: {ic:.4f}")

    return best_model, best_params, best_ic

# 示例
param_grid = {
    'num_leaves': [31, 63],
    'learning_rate': [0.01, 0.05, 0.1],
    'min_data_in_leaf': [10, 20],
}

best_model, best_params, best_ic = select_model_by_ic(
    X_train, y_train, X_val, y_val, param_grid, LGBMRegressor
)

print(f"\n最佳参数: {best_params}")
print(f"最佳IC: {best_ic:.4f}")
```

## 8. 总结

IC和Rank IC是量化投资中最重要的评估指标：

1. **IC定义**：预测值与实际值的Pearson相关系数
2. **Rank IC**：预测值排序与实际值排序的Spearman秩相关系数
3. **统计显著性**：通过t检验和置信区间验证IC的显著性
4. **时序分析**：滚动IC和IC衰减分析预测的稳定性和时效性
5. **IR指标**：衡量IC的稳定性，IC均值除以标准差
6. **多维度分析**：按市场状态、行业等子集分析IC表现

IC是量化模型评估的核心，正确的IC分析是构建有效量化策略的基础。
