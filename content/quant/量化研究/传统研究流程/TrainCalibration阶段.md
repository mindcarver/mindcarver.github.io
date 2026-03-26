# Train Calibration 阶段详细文档

**文档ID**: QSP-TC-v1.0
**阶段编号**: 03
**阶段名称**: Train Calibration (训练校准)
**日期**: 2026-03-26
**状态**: v1.0
**负责角色**: Quant Researcher

---

## 目录

1. [阶段定义与核心目的](#1-阶段定义与核心目的)
2. [关键约束与纪律](#2-关键约束与纪律)
3. [校准内容清单](#3-校准内容清单)
4. [阈值校准](#4-阈值校准)
5. [分位数切点校准](#5-分位数切点校准)
6. [质量过滤标准](#6-质量过滤标准)
7. [Regime 切分](#7-regime-切分)
8. [Coarse-to-fine 搜索策略](#8-coarse-to-fine-搜索策略)
9. [参数空间搜索规范](#9-参数空间搜索规范)
10. [参数搜索轨迹](#10-参数搜索轨迹)
11. [Formal Gate 要求](#11-formal-gate-要求)
12. [常见错误与防范](#12-常见错误与防范)
13. [输出 Artifact](#13-输出-artifact)
14. [与 Test Evidence 的交接](#14-与-test-evidence-的交接)
15. [校准报告模板](#15-校准报告模板)

---

## 1. 阶段定义与核心目的

### 1.1 阶段定义

**Train Calibration (训练校准)** 是在不接触未来数据的前提下，"定尺子"的阶段。

**核心职责**：
- 在 In-Sample (IS) 数据上确定信号的所有阈值、切点和过滤标准
- 冻结信号参数、质量标准、Regime 定义
- 为 Test Evidence 阶段提供"尺子"，而非"最优结果"

### 1.2 "定尺子"的含义

**什么是"尺子"**：
```yaml
"尺子" (calibration_tools):
  thresholds:
    description: "信号判断的临界值"
    example: "RSI > 70 为超买"
    frozen_at: "Train Calibration"

  quantile_cuts:
    description: "分位数的切点位置"
    example: "Q5 = 80th percentile, Q1 = 20th percentile"
    frozen_at: "Train Calibration"

  quality_filters:
    description: "数据质量的可接受标准"
    example: "覆盖率 ≥ 80%, outlier率 ≤ 5%"
    frozen_at: "Train Calibration"

  regime_thresholds:
    description: "市场状态分界的临界值"
    example: "波动率 > 2% 为高波动状态"
    frozen_at: "Train Calibration"
```

**与"优化"的区别**：
| 维度 | Calibration (校准) | Optimization (优化) |
|------|-------------------|-------------------|
| **目标** | 确定稳定的判断标准 | 最大化某个指标 |
| **数据** | 只用 IS 数据 | 容易无意中使用 OOS |
| **验证** | 在 Test 验证 | 容易过拟合 |
| **冻结** | 一旦冻结不再改变 | 容易反复调整 |
| **心态** | "定规则" | "找最好的" |

### 1.3 核心目的

**校准什么**：
1. **信号阈值**: 确定多头/空头/中性的判断边界
2. **分位切点**: 确定分层分析的分界线
3. **质量标准**: 确定数据可接受性的最低要求
4. **Regime 定义**: 确定不同市场环境的划分标准

**不校准什么**：
- 不追求"最优"参数组合（避免过拟合）
- 不根据 Test 结果回头调整（防止样本污染）
- 不进行参数微调以改善 OOS 表现（防止隐性的未来信息泄漏）

---

## 2. 关键约束与纪律

### 2.1 铁律：只能用 IS 数据

**数据范围约束**：
```yaml
data_constraint:
  allowed_data:
    period: "[T_start, T_train_end]"
    name: "In-Sample (IS)"
    purpose: "参数校准、阈值确定、质量标准设定"

  forbidden_data:
    period: "(T_train_end, T_holdout_end]"
    name: "Out-of-Sample (OOS)"
    forbidden_actions:
      - "不能用于确定阈值"
      - "不能用于选择参数"
      - "不能用于设定质量标准"
      - "不能用于调整 Regime 定义"
```

**检测机制**：
```python
def validate_is_only_calibration(timestamps, train_end_date):
    """
    验证校准只使用 IS 数据

    Args:
        timestamps: 数据时间戳
        train_end_date: Train 结束日期

    Raises:
        ValueError: 如果发现 OOS 数据
    """
    max_timestamp = timestamps.max()

    if max_timestamp > train_end_date:
        raise ValueError(
            f"检测到 OOS 数据: max_timestamp={max_timestamp}, "
            f"train_end={train_end_date}. "
            f"校准只能使用 IS 数据!"
        )
```

### 2.2 禁忌：不能根据 Test 结果回头重算

**禁止的行为模式**：
```yaml
forbidden_workflow:
  scenario_1: "在 Test 上验证 → 发现表现差 → 回头改 Train 参数"
  scenario_2: "在 Test 上验证 → 发现方向反 → 回头调阈值"
  scenario_3: "在 Test 上验证 → 发现某些标的差 → 回头改质量标准"

  correct_workflow:
    step_1: "在 IS 上校准参数"
    step_2: "冻结所有校准结果"
    step_3: "在 Test 上验证（不改参数）"
    step_4: "如果 Test 失败 → 要么接受要么回到 Signal Ready"
```

**防止回退机制**：
```python
# 生成不可变的校准规范
@dataclass(frozen=True)
class TrainCalibrationSpec:
    """Train Calibration 冻结规范"""
    thresholds: Dict[str, float]
    quantile_cuts: Dict[str, List[float]]
    quality_filters: Dict[str, Any]
    regime_thresholds: Dict[str, float]
    param_id: str
    frozen_at: str

    def __post_init__(self):
        # 防止修改
        object.__setattr__(self, '_immutable', True)
```

### 2.3 校准纪律

**先校准，再验证**：
1. 在 IS 上完成所有校准
2. 生成 Frozen Spec
3. 在 Test 上验证（不能改）
4. 如果验证失败，分析原因但不回头调整

**负结果保留**：
- 记录所有尝试过的参数组合
- 记录被拒绝的阈值设定
- 保留失败的分位切点尝试
- 避免"只保留成功结果"的偏差

---

## 3. 校准内容清单

### 3.1 完整校准清单

```yaml
calibration_checklist:

  signal_thresholds:
    description: "信号值转化为交易决策的临界值"
    examples:
      - "RSI > 70: 超买信号"
      - "Momentum > 0: 正动量"
      - "Z-score > 2: 异常高位"
    frozen_as: "thresholds.yaml"

  quantile_cuts:
    description: "将信号分为多层（如Q1-Q5）的切点"
    examples:
      - "Q5: 80th percentile"
      - "Q4: 60th percentile"
      - "Q3: 40th percentile"
      - "Q2: 20th percentile"
      - "Q1: 0th percentile"
    frozen_as: "quantiles.yaml"

  quality_filters:
    description: "数据质量的可接受标准"
    metrics:
      - "min_coverage: 0.8"
      - "max_staleness: 0.1"
      - "max_outlier_rate: 0.05"
      - "min_signal_strength: 0.3"
    frozen_as: "quality_filters.yaml"

  regime_thresholds:
    description: "市场状态分界的临界值"
    examples:
      - "volatility > 2%: 高波动状态"
      - "trend_strength > 0.5: 趋势状态"
      - "volume_ratio > 1.5: 高成交量状态"
    frozen_as: "regime_definition.yaml"

  signal_parameters:
    description: "信号计算的所有参数"
    examples:
      - "lookback_window"
      - "smoothing_period"
      - "normalization_method"
      - "decay_factor"
    frozen_as: "signal_params.yaml"
```

### 3.2 校准优先级

**必须校准（Formal Gate）**：
1. 信号阈值（决定交易方向）
2. 分位切点（用于分层分析）
3. 质量过滤标准（用于 Whitelist 确定）

**建议校准（Audit Gate）**：
1. Regime 切分（用于细分分析）
2. 标准化参数（用于信号计算）
3. 异常值边界（用于数据清洗）

---

## 4. 阈值校准

### 4.1 定义

**信号阈值** 是将连续信号值转换为离散交易决策（多头/空头/中性）的临界值。

### 4.2 校准方法

**基于分布的阈值**：
```python
def calibrate_distribution_threshold(signal, method='percentile'):
    """
    基于信号分布校准阈值

    Args:
        signal: IS 信号值
        method: 校准方法

    Returns:
        threshold: 校准后的阈值
    """
    if method == 'percentile':
        # 使用分位数作为阈值
        upper_threshold = np.percentile(signal, 75)
        lower_threshold = np.percentile(signal, 25)

    elif method == 'std':
        # 使用标准差作为阈值
        mean = signal.mean()
        std = signal.std()
        upper_threshold = mean + 2 * std
        lower_threshold = mean - 2 * std

    elif method == 'fixed':
        # 使用固定阈值（需基于领域知识）
        upper_threshold = 70  # 如 RSI
        lower_threshold = 30

    return {
        'upper_threshold': upper_threshold,
        'lower_threshold': lower_threshold,
        'method': method
    }
```

**基于收益的阈值**：
```python
def calibrate_return_threshold(signal, returns, n_bins=100):
    """
    基于收益表现校准最优阈值

    Args:
        signal: IS 信号值
        returns: IS 收益率
        n_bins: 搜索粒度

    Returns:
        optimal_threshold: 最优阈值
    """
    # 生成候选阈值
    thresholds = np.linspace(signal.min(), signal.max(), n_bins)

    best_sharpe = -np.inf
    best_threshold = None

    for threshold in thresholds:
        # 基于阈值生成信号
        signal_binary = (signal > threshold).astype(int)

        # 计算收益
        portfolio_return = (signal_binary * returns).mean()
        portfolio_vol = (signal_binary * returns).std()

        if portfolio_vol > 0:
            sharpe = portfolio_return / portfolio_vol

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_threshold = threshold

    return {
        'threshold': best_threshold,
        'sharpe': best_sharpe,
        'method': 'return_optimization'
    }
```

### 4.3 稳健性校准

**避免过拟合的校准**：
```python
def robust_threshold_calibration(signal, returns, n_folds=5):
    """
    稳健的阈值校准（使用交叉验证）

    Args:
        signal: IS 信号值
        returns: IS 收益率
        n_folds: 交叉验证折数

    Returns:
        robust_threshold: 稳健阈值
    """
    from sklearn.model_selection import TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=n_folds)
    threshold_results = []

    for train_idx, val_idx in tscv.split(signal):
        signal_train = signal.iloc[train_idx]
        returns_train = returns.iloc[train_idx]

        # 在训练折上校准
        threshold = calibrate_return_threshold(
            signal_train, returns_train
        )['threshold']

        # 在验证折上评估
        signal_val = signal.iloc[val_idx]
        returns_val = returns.iloc[val_idx]

        signal_binary = (signal_val > threshold).astype(int)
        sharpe = signal_binary.mean() / (signal_binary.std() + 1e-8)

        threshold_results.append({
            'threshold': threshold,
            'val_sharpe': sharpe
        })

    # 选择在验证折上表现最稳定的阈值
    #（而非最优的阈值）
    results_df = pd.DataFrame(threshold_results)
    robust_threshold = results_df.groupby(
        'threshold'
    )['val_sharpe'].mean().idxmax()

    return robust_threshold
```

### 4.4 阈值冻结

**冻结规范**：
```yaml
threshold_spec:
  version: "v1.0"
  frozen_at: "Train Calibration"
  immutable: true

  upper_threshold:
    value: 0.8
    method: "percentile_75"
    calibrated_on: "IS data"
    calibration_date: "2026-03-26"

  lower_threshold:
    value: -0.6
    method: "percentile_25"
    calibrated_on: "IS data"
    calibration_date: "2026-03-26"

  neutral_zone:
    description: "上下阈值之间的区域为中性区"
    range: "[-0.6, 0.8]"

  next_stage_constraints:
    forbidden:
      - "不能在 Test 上重新估计阈值"
      - "不能根据 Test 结果调整阈值"
      - "不能基于 OOS 数据修改阈值"
```

---

## 5. 分位数切点校准

### 5.1 定义

**分位数切点** 是将连续信号值分为多个层级（如 Q1-Q5）的临界值。

### 5.2 校准方法

**等频分位**：
```python
def calibrate_quantile_cuts(signal, n_quantiles=5):
    """
    校准等频分位切点

    Args:
        signal: IS 信号值
        n_quantiles: 分位数数量

    Returns:
        quantile_cuts: 分位切点
    """
    # 计算分位切点
    quantiles = np.linspace(0, 100, n_quantiles + 1)
    cuts = [np.percentile(signal, q) for q in quantiles]

    # 确保切点唯一
    unique_cuts = sorted(set(cuts))

    # 如果切点数量不足，使用等距切点
    if len(unique_cuts) < n_quantiles + 1:
        unique_cuts = np.linspace(
            signal.min(),
            signal.max(),
            n_quantiles + 1
        ).tolist()

    return {
        'cuts': unique_cuts,
        'n_quantiles': n_quantiles,
        'method': 'equal_frequency',
        'calibrated_on': 'IS data'
    }
```

**等距分位**：
```python
def calibrate_equal_width_cuts(signal, n_quantiles=5):
    """
    校准等距分位切点

    Args:
        signal: IS 信号值
        n_quantiles: 分位数数量

    Returns:
        quantile_cuts: 分位切点
    """
    min_val = signal.min()
    max_val = signal.max()
    step = (max_val - min_val) / n_quantiles

    cuts = [min_val + i * step for i in range(n_quantiles + 1)]

    return {
        'cuts': cuts,
        'n_quantiles': n_quantiles,
        'method': 'equal_width',
        'calibrated_on': 'IS data'
    }
```

**自定义分位**：
```python
def calibrate_custom_cuts(signal, percentiles=[0, 20, 40, 60, 80, 100]):
    """
    校准自定义分位切点

    Args:
        signal: IS 信号值
        percentiles: 自定义分位数

    Returns:
        quantile_cuts: 分位切点
    """
    cuts = [np.percentile(signal, p) for p in percentiles]

    return {
        'cuts': cuts,
        'percentiles': percentiles,
        'method': 'custom_percentiles',
        'calibrated_on': 'IS data'
    }
```

### 5.3 分位切点验证

**验证原则**：
1. **分布均衡**: 每个分位应有足够的样本量
2. **切点稳定**: 切点值应该稳定，不应随微小数据变化而剧烈变化
3. **收益单调**: 分位收益应该有单调性（对于有效信号）

**验证代码**：
```python
def validate_quantile_cuts(signal, returns, quantile_cuts):
    """
    验证分位切点的合理性

    Args:
        signal: IS 信号值
        returns: IS 收益率
        quantile_cuts: 分位切点

    Returns:
        validation_result: 验证结果
    """
    # 分层
    layers = pd.cut(signal, bins=quantile_cuts, labels=False)

    # 验证1: 分布均衡
    layer_counts = layers.value_counts().sort_index()
    min_count = layer_counts.min()
    max_count = layer_counts.max()
    balance_ratio = min_count / max_count

    # 验证2: 收益单调性
    layer_returns = returns.groupby(layers).mean()
    monotonicity = (layer_returns.diff().dropna() > 0).all()

    # 验证3: 切点稳定性（使用 Bootstrap）
    stable_cuts = []
    for _ in range(100):
        sample_signal = signal.sample(frac=0.8, replace=True)
        sample_cuts = calibrate_quantile_cuts(sample_signal)['cuts']
        stable_cuts.append(sample_cuts)

    cut_std = np.std(stable_cuts, axis=0)

    return {
        'balance_ratio': balance_ratio,
        'balance_acceptable': balance_ratio >= 0.5,
        'monotonicity': monotonicity,
        'cut_stability': cut_std.tolist(),
        'overall_valid': (balance_ratio >= 0.5) and monotonicity
    }
```

### 5.4 分位切点冻结

**冻结规范**：
```yaml
quantile_spec:
  version: "v1.0"
  frozen_at: "Train Calibration"
  immutable: true

  quantile_cuts:
    Q5: 0.85
    Q4: 0.45
    Q3: 0.15
    Q2: -0.25
    Q1: -0.75
    method: "equal_frequency"

  validation:
    balance_ratio: 0.82
    monotonicity: true
    cut_stability_max_std: 0.03

  next_stage_constraints:
    forbidden:
      - "不能在 Test 上重新计算切点"
      - "不能根据 Test 表现调整切点"
      - "不能改变分位数量"
```

---

## 6. 质量过滤标准

### 6.1 定义

**质量过滤标准** 是确定数据可接受性的最低要求，用于后续的 Whitelist 确定。

### 6.2 质量指标

**关键质量指标**：
```python
def calculate_quality_metrics(data):
    """
    计算数据质量指标

    Args:
        data: 原始数据

    Returns:
        quality_metrics: 质量指标字典
    """
    metrics = {}

    # 1. 覆盖率 (Coverage)
    metrics['coverage'] = data.notna().mean()

    # 2. 停滞率 (Staleness)
    if 'price' in data.columns:
        price_diff = data['price'].diff()
        metrics['staleness'] = (price_diff == 0).mean()

    # 3. 异常值率 (Outlier Rate)
    if 'value' in data.columns:
        q1 = data['value'].quantile(0.25)
        q3 = data['value'].quantile(0.75)
        iqr = q3 - q1
        outliers = ((data['value'] < q1 - 3*iqr) |
                    (data['value'] > q3 + 3*iqr))
        metrics['outlier_rate'] = outliers.mean()

    # 4. 零值率 (Zero Rate)
    if 'value' in data.columns:
        metrics['zero_rate'] = (data['value'] == 0).mean()

    # 5. 信号强度 (Signal Strength)
    if 'signal' in data.columns:
        metrics['signal_strength'] = data['signal'].abs().mean()

    return metrics
```

### 6.3 质量阈值校准

**基于分布的阈值**：
```python
def calibrate_quality_thresholds(is_data, quality_metrics_list):
    """
    校准质量过滤阈值

    Args:
        is_data: IS 数据
        quality_metrics_list: 质量指标列表

    Returns:
        quality_thresholds: 质量阈值
    """
    # 对每个标的计算质量指标
    all_metrics = {}
    for symbol in is_data['symbol'].unique():
        symbol_data = is_data[is_data['symbol'] == symbol]
        all_metrics[symbol] = calculate_quality_metrics(symbol_data)

    metrics_df = pd.DataFrame(all_metrics).T

    # 基于分布确定阈值
    thresholds = {}

    for metric in quality_metrics_list:
        if metric in ['coverage', 'signal_strength']:
            # 越高越好，使用下分位数
            thresholds[f'min_{metric}'] = metrics_df[metric].quantile(0.2)
        else:
            # 越低越好，使用上分位数
            thresholds[f'max_{metric}'] = metrics_df[metric].quantile(0.8)

    return thresholds
```

**领域知识阈值**：
```python
# 基于领域知识的质量阈值
DOMAIN_KNOWLEDGE_THRESHOLDS = {
    'coverage': 0.8,         # 至少80%的数据有效
    'staleness': 0.1,        # 最多10%的数据停滞
    'outlier_rate': 0.05,    # 最多5%的异常值
    'signal_strength': 0.3,  # 最低信号强度
}
```

### 6.4 质量标准冻结

**冻结规范**：
```yaml
quality_filters_spec:
  version: "v1.0"
  frozen_at: "Train Calibration"
  immutable: true

  thresholds:
    min_coverage: 0.8
    max_staleness: 0.1
    max_outlier_rate: 0.05
    min_signal_strength: 0.3

  calibration_method:
    primary: "domain_knowledge"
    secondary: "distribution_based"

  rationale:
    coverage: "需要足够的数据来计算可靠的信号"
    staleness: "过多的停滞数据表明流动性问题"
    outlier_rate: "过多异常值表明数据质量差"
    signal_strength: "太弱的信号无交易价值"

  next_stage_constraints:
    forbidden:
      - "不能在 Test 上放宽质量标准"
      - "不能根据 Test 表现调整阈值"
      - "不能对个别标的例外处理"
```

---

## 7. Regime 切分

### 7.1 定义

**Regime (状态/体制)** 是不同市场环境的分类，如高/低波动状态、趋势/震荡状态等。

### 7.2 Regime 类型

**常见 Regime 类型**：
```yaml
regime_types:

  volatility_regime:
    name: "波动率状态"
    states:
      - "low_volatility"
      - "medium_volatility"
      - "high_volatility"
    indicator: "realized_volatility"

  trend_regime:
    name: "趋势状态"
    states:
      - "uptrend"
      - "sideways"
      - "downtrend"
    indicator: "trend_strength"

  volume_regime:
    name: "成交量状态"
    states:
      - "low_volume"
      - "normal_volume"
      - "high_volume"
    indicator: "volume_ratio"

  market_cycle_regime:
    name: "市场周期"
    states:
      - "bull_market"
      - "bear_market"
      - "transition"
    indicator: "market_index_return"
```

### 7.3 Regime 切分校准

**基于分位数的切分**：
```python
def calibrate_regime_thresholds(regime_indicator, n_regimes=3):
    """
    校准 Regime 切分阈值

    Args:
        regime_indicator: Regime 指标（如波动率）
        n_regimes: Regime 数量

    Returns:
        regime_thresholds: Regime 切分阈值
    """
    # 使用分位数作为切点
    percentiles = np.linspace(0, 100, n_regimes + 1)
    thresholds = [np.percentile(regime_indicator, p) for p in percentiles]

    return {
        'thresholds': thresholds,
        'n_regimes': n_regimes,
        'method': 'quantile_based'
    }
```

**基于统计检验的切分**：
```python
def calibrate_regime_change_points(regime_indicator, min_segment_length=30):
    """
    基于变点检测校准 Regime 切分

    Args:
        regime_indicator: Regime 指标
        min_segment_length: 最小区段长度

    Returns:
        change_points: 变点位置
    """
    from ruptures import Pelt

    # 使用 Pelt 算法检测变点
    algo = Pelt(model="rbf").fit(regime_indicator.values)
    change_points = algo.predict(pen=10)

    # 过滤掉太短的区段
    filtered_points = [
        p for p in change_points
        if all((np.diff([0] + change_points + [len(regime_indicator)]) >= min_segment_length))
    ]

    return {
        'change_points': filtered_points,
        'method': 'change_point_detection'
    }
```

### 7.4 Regime 定义冻结

**冻结规范**：
```yaml
regime_spec:
  version: "v1.0"
  frozen_at: "Train Calibration"
  immutable: true

  volatility_regime:
    indicator: "realized_volatility_20d"
    thresholds:
      low: "0% - 1%"
      medium: "1% - 2.5%"
      high: "> 2.5%"
    cut_points: [0.01, 0.025]

  trend_regime:
    indicator: "adx_14d"
    thresholds:
      weak_trend: "< 20"
      strong_trend: "> 25"
    cut_points: [20, 25]

  next_stage_constraints:
    forbidden:
      - "不能在 Test 上重新估计切点"
      - "不能根据 Test 表现调整 Regime 定义"
      - "不能新增 Regime 类型"
```

---

## 8. Coarse-to-fine 搜索策略

### 8.1 定义

**Coarse-to-fine Search (粗到精搜索)** 是先粗网格搜索，再在局部精细搜索的参数优化策略。

### 8.2 搜索流程

**三阶段搜索**：
```python
def coarse_to_fine_search(param_grid, evaluation_func, max_iterations=100):
    """
    粗到精搜索

    Args:
        param_grid: 参数网格
        evaluation_func: 评估函数
        max_iterations: 最大迭代次数

    Returns:
        best_params: 最优参数
    """
    # 阶段1: 粗网格搜索
    print("Phase 1: Coarse grid search")
    coarse_results = grid_search(
        param_grid['coarse'],
        evaluation_func
    )
    best_coarse = max(coarse_results, key=lambda x: x['score'])

    # 阶段2: 局部精细搜索
    print("Phase 2: Fine local search")
    fine_grid = generate_fine_grid(
        best_coarse['params'],
        param_grid['fine_step']
    )
    fine_results = grid_search(
        fine_grid,
        evaluation_func
    )
    best_fine = max(fine_results, key=lambda x: x['score'])

    # 阶段3: 局部优化（可选）
    print("Phase 3: Local optimization")
    final_params = local_optimization(
        best_fine['params'],
        evaluation_func
    )

    return {
        'best_params': final_params,
        'coarse_best': best_coarse,
        'fine_best': best_fine,
        'search_trajectory': {
            'coarse': coarse_results,
            'fine': fine_results
        }
    }
```

### 8.3 参数网格定义

**粗网格**：
```python
COARSE_PARAM_GRID = {
    'lookback': [5, 10, 20, 40, 60],
    'smoothing': [1, 3, 5, 10],
    'threshold': [0.5, 1.0, 1.5, 2.0],
    'decay': [0.9, 0.95, 0.99]
}
```

**精细网格**：
```python
FINE_PARAM_GRID = {
    'lookback_step': 2,
    'smoothing_step': 1,
    'threshold_step': 0.1,
    'decay_step': 0.01
}
```

### 8.4 搜索策略可视化

```
Coarse-to-fine Search 示意图：

阶段1: 粗网格 (5×4×4×3 = 240 点)
├─ lookback: [5, 10, 20, 40, 60]
├─ smoothing: [1, 3, 5, 10]
├─ threshold: [0.5, 1.0, 1.5, 2.0]
└─ decay: [0.9, 0.95, 0.99]

      最优点: (lookback=20, smoothing=5, threshold=1.0, decay=0.95)
                ↓

阶段2: 精细搜索 (5×5×5×5 = 125 点)
├─ lookback: [16, 18, 20, 22, 24]
├─ smoothing: [3, 4, 5, 6, 7]
├─ threshold: [0.8, 0.9, 1.0, 1.1, 1.2]
└─ decay: [0.93, 0.94, 0.95, 0.96, 0.97]

      最优点: (lookback=22, smoothing=6, threshold=0.9, decay=0.94)
                ↓

阶段3: 局部优化
└─ 使用梯度下降等算法在局部优化
```

---

## 9. 参数空间搜索规范

### 9.1 参数空间定义

**完整参数空间**：
```yaml
parameter_space:

  signal_parameters:
    lookback_window:
      type: "integer"
      range: "[5, 100]"
      constraint: "必须为奇数"

    smoothing_period:
      type: "integer"
      range: "[1, 20]"
      constraint: "必须小于 lookback_window/3"

    normalization_method:
      type: "categorical"
      options: ["zscore", "minmax", "rank"]

  threshold_parameters:
    upper_threshold:
      type: "float"
      range: "[0.5, 3.0]"

    lower_threshold:
      type: "float"
      range: "[-3.0, -0.5]"

  quality_parameters:
    min_coverage:
      type: "float"
      range: "[0.5, 0.99]"

    max_outlier_rate:
      type: "float"
      range: "[0.01, 0.2]"
```

### 9.2 搜索方法

**网格搜索**：
```python
def grid_search(param_grid, evaluation_func, n_jobs=-1):
    """
    网格搜索

    Args:
        param_grid: 参数网格
        evaluation_func: 评估函数
        n_jobs: 并行任务数

    Returns:
        results: 所有参数组合的结果
    """
    from itertools import product
    from joblib import Parallel, delayed

    # 生成所有参数组合
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(product(*param_values))

    # 并行评估
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluation_func)(
            **dict(zip(param_names, combination))
        )
        for combination in all_combinations
    )

    # 组合结果
    search_results = []
    for combination, result in zip(all_combinations, results):
        search_results.append({
            'params': dict(zip(param_names, combination)),
            'score': result
        })

    return search_results
```

**随机搜索**：
```python
def random_search(param_bounds, evaluation_func, n_iter=100, random_state=None):
    """
    随机搜索

    Args:
        param_bounds: 参数边界
        evaluation_func: 评估函数
        n_iter: 迭代次数
        random_state: 随机种子

    Returns:
        results: 搜索结果
    """
    if random_state is not None:
        np.random.seed(random_state)

    results = []

    for _ in range(n_iter):
        # 随机采样参数
        params = {}
        for param_name, bounds in param_bounds.items():
            if bounds['type'] == 'integer':
                params[param_name] = np.random.randint(
                    bounds['range'][0],
                    bounds['range'][1] + 1
                )
            elif bounds['type'] == 'float':
                params[param_name] = np.random.uniform(
                    bounds['range'][0],
                    bounds['range'][1]
                )
            elif bounds['type'] == 'categorical':
                params[param_name] = np.random.choice(bounds['options'])

        # 评估
        score = evaluation_func(**params)

        results.append({
            'params': params,
            'score': score
        })

    return results
```

**贝叶斯优化**：
```python
def bayesian_optimization(param_bounds, evaluation_func, n_iter=50):
    """
    贝叶斯优化

    Args:
        param_bounds: 参数边界
        evaluation_func: 评估函数
        n_iter: 迭代次数

    Returns:
        results: 优化结果
    """
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical

    # 定义搜索空间
    space = []
    param_names = []

    for param_name, bounds in param_bounds.items():
        param_names.append(param_name)

        if bounds['type'] == 'integer':
            space.append(Integer(
                bounds['range'][0],
                bounds['range'][1],
                name=param_name
            ))
        elif bounds['type'] == 'float':
            space.append(Real(
                bounds['range'][0],
                bounds['range'][1],
                name=param_name
            ))
        elif bounds['type'] == 'categorical':
            space.append(Categorical(
                bounds['options'],
                name=param_name
            ))

    # 定义目标函数（最小化负分数）
    def objective(params):
        param_dict = dict(zip(param_names, params))
        return -evaluation_func(**param_dict)

    # 运行优化
    result = gp_minimize(
        objective,
        space,
        n_calls=n_iter,
        random_state=42
    )

    # 返回最优参数
    best_params = dict(zip(param_names, result.x))

    return {
        'best_params': best_params,
        'best_score': -result.fun,
        'optimization_trajectory': result.func_vals
    }
```

### 9.3 搜索预算管理

**计算预算分配**：
```yaml
search_budget:

  total_budget: "1000 次评估"

  allocation:
    coarse_search: 300  # 30%
    fine_search: 500    # 50%
    local_optimization: 200  # 20%

  early_stopping:
    enabled: true
    patience: 50
    min_improvement: 0.01

  parallel_execution:
    enabled: true
    max_workers: 8
```

---

## 10. 参数搜索轨迹

### 10.1 定义

**参数搜索轨迹** 是记录所有尝试过的参数组合及其结果的完整记录。

### 10.2 轨迹记录

**轨迹数据结构**：
```python
@dataclass
class ParameterSearchRecord:
    """参数搜索记录"""
    run_id: str
    timestamp: str
    stage: str  # "coarse", "fine", "optimization"

    # 参数
    params: Dict[str, Any]

    # 结果
    score: float
    metrics: Dict[str, float]

    # 元数据
    evaluation_time: float
    is_best: bool
    notes: str = ""
```

**轨迹记录器**：
```python
class ParameterSearchTracker:
    """参数搜索轨迹记录器"""

    def __init__(self, output_path):
        self.output_path = output_path
        self.records = []

    def record(self, record: ParameterSearchRecord):
        """记录一次参数搜索"""
        self.records.append(record)

    def save(self):
        """保存轨迹到文件"""
        records_df = pd.DataFrame([
            {
                'run_id': r.run_id,
                'timestamp': r.timestamp,
                'stage': r.stage,
                'params': json.dumps(r.params),
                'score': r.score,
                'metrics': json.dumps(r.metrics),
                'evaluation_time': r.evaluation_time,
                'is_best': r.is_best,
                'notes': r.notes
            }
            for r in self.records
        ])

        records_df.to_csv(
            self.output_path / 'parameter_search_trajectory.csv',
            index=False
        )

    def get_best_params(self, stage=None):
        """获取最优参数"""
        filtered = self.records
        if stage is not None:
            filtered = [r for r in filtered if r.stage == stage]

        return max(filtered, key=lambda x: x.score)
```

### 10.3 轨迹分析

**参数敏感性分析**：
```python
def analyze_parameter_sensitivity(search_trajectory):
    """
    分析参数敏感性

    Args:
        search_trajectory: 搜索轨迹

    Returns:
        sensitivity: 参数敏感性分析结果
    """
    df = pd.DataFrame(search_trajectory)

    # 对每个参数计算敏感性
    sensitivity = {}

    for param in df['params'].iloc[0].keys():
        # 提取该参数的所有值和对应分数
        param_scores = []
        for _, row in df.iterrows():
            param_value = row['params'][param]
            param_scores.append({
                'value': param_value,
                'score': row['score']
            })

        param_df = pd.DataFrame(param_scores)

        # 计算分数随参数值的变化
        param_grouped = param_df.groupby('value')['score'].agg(['mean', 'std'])

        sensitivity[param] = {
            'range': param_df['value'].max() - param_df['value'].min(),
            'score_range': param_grouped['mean'].max() - param_grouped['mean'].min(),
            'stability': param_grouped['std'].mean()
        }

    return sensitivity
```

### 10.4 轨迹可视化

**搜索轨迹图**：
```python
def visualize_search_trajectory(search_trajectory, output_path):
    """
    可视化搜索轨迹

    Args:
        search_trajectory: 搜索轨迹
        output_path: 输出路径
    """
    import matplotlib.pyplot as plt

    df = pd.DataFrame(search_trajectory)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. 分数随迭代的变化
    ax1 = axes[0, 0]
    ax1.plot(df.index, df['score'], marker='o', alpha=0.5)
    ax1.plot(df['score'].cummax(), color='red', label='Best so far')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Score')
    ax1.set_title('Search Progress')
    ax1.legend()

    # 2. 分数分布
    ax2 = axes[0, 1]
    ax2.hist(df['score'], bins=50, alpha=0.7)
    ax2.axvline(df['score'].max(), color='red', linestyle='--', label='Best')
    ax2.set_xlabel('Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Score Distribution')
    ax2.legend()

    # 3. 参数 vs 分数 (选择主要参数)
    ax3 = axes[1, 0]
    param_names = list(df['params'].iloc[0].keys())
    if param_names:
        main_param = param_names[0]
        param_values = [r['params'][main_param] for r in search_trajectory]
        ax3.scatter(param_values, df['score'], alpha=0.5)
        ax3.set_xlabel(main_param)
        ax3.set_ylabel('Score')
        ax3.set_title(f'{main_param} vs Score')

    # 4. 阶段对比
    ax4 = axes[1, 1]
    stage_scores = df.groupby('stage')['score'].apply(list)
    ax4.boxplot([scores for scores in stage_scores.values],
                labels=stage_scores.index)
    ax4.set_xlabel('Stage')
    ax4.set_ylabel('Score')
    ax4.set_title('Score by Stage')

    plt.tight_layout()
    plt.savefig(output_path / 'search_trajectory.png', dpi=150)
    plt.close()
```

---

## 11. Formal Gate 要求

### 11.1 门禁检查清单

**PASS 条件**：
```yaml
formal_gate_requirements:

  thresholds_calibrated:
    requirement: "信号阈值已校准"
    evidence: "thresholds.yaml"
    validation: "阈值在 IS 上合理且稳定"
    status: "PASS / FAIL"

  quantiles_calibrated:
    requirement: "分位切点已校准"
    evidence: "quantiles.yaml"
    validation: "分层收益具有单调性"
    status: "PASS / FAIL"

  quality_filters_set:
    requirement: "质量过滤标准已设定"
    evidence: "quality_filters.yaml"
    validation: "基于领域知识或分布分析"
    status: "PASS / FAIL"

  regime_defined:
    requirement: "Regime 切分已定义（如适用）"
    evidence: "regime_definition.yaml"
    validation: "Regime 切分稳定且有意义"
    status: "PASS / FAIL"

  parameters_searched:
    requirement: "参数搜索已完成"
    evidence: "parameter_search_trajectory.csv"
    validation: "搜索覆盖足够的空间"
    status: "PASS / FAIL"

  is_only:
    requirement: "只使用 IS 数据进行校准"
    evidence: "data_audit_log + timestamp_validation"
    validation: "未使用任何 OOS 数据"
    status: "PASS / FAIL"

  frozen_spec_generated:
    requirement: "Frozen Spec 已生成"
    evidence: "train_calibration_spec.yaml"
    validation: "所有校准结果已冻结"
    status: "PASS / FAIL"

  negative_results_retained:
    requirement: "负结果已保留"
    evidence: "search_trajectory包含所有尝试"
    validation: "未只保留成功结果"
    status: "PASS / FAIL"
```

### 11.2 决策状态

**可能的状态**：
```yaml
verdict_states:

  PASS:
    description: "所有校准完成，可进入 Test Evidence"
    frozen_items:
      - "信号阈值 (thresholds)"
      - "分位切点 (quantile_cuts)"
      - "质量过滤标准 (quality_filters)"
      - "Regime 定义 (regime_definition)"
      - "信号参数 (signal_params)"

  CONDITIONAL_PASS:
    description: "核心校准完成，但有需注意的限制"
    conditions:
      - "某些 Regime 样本量较少"
      - "质量标准较宽松"
      - "参数搜索未完全收敛"

  RETRY:
    description: "实现问题修复后可重试"
    scope:
      - "代码实现 bug 修复"
      - "数据处理错误修复"
      - "计算逻辑修正"
    forbidden:
      - "不能改变研究主问题"
      - "不能使用 OOS 数据"

  RESEARCH_AGAIN:
    description: "校准失败，需回到 Signal Ready 或更早阶段"
    rollback_stage: "02_signal_ready 或更早"
    reasons:
      - "信号在 IS 上就无效"
      - "无法找到合理的校准结果"
      - "质量标准无法满足"

  NO_GO:
    description: "信号假设不成立，终止研究线"
    reasons:
      - "IS 上完全没有信号"
      - "质量标准无法满足且无调整空间"
      - "信号机制不成立"
```

---

## 12. 常见错误与防范

### 12.1 错误清单

| 错误类型 | 描述 | 后果 | 防范措施 |
|---------|------|------|---------|
| **使用 OOS 数据** | 在校准时无意中使用了 Test 数据 | 样本污染 | 时间戳验证 |
| **过拟合阈值** | 在 IS 上过度优化阈值 | Test 失败 | 稳健性校准 |
| **忽略负结果** | 只保留表现好的参数组合 | 幸存者偏差 | 完整轨迹记录 |
| **调整质量标准** | 根据 Test 结果调整质量标准 | 过拟合 | 质量标准提前冻结 |
| **参数搜索不足** | 搜索空间太小或搜索不充分 | 错过更优参数 | 搜索预算管理 |
| **忽略 Regime** | 不考虑不同市场状态 | 信号不稳定 | Regime 分析 |
| **混淆单位** | 混淆价差单位和资金单位 | 虚假高收益 | 明确标注单位 |

### 12.2 防范机制

**代码层面**：
```python
# 时间范围检查
def validate_is_timestamps(timestamps, train_end_date):
    """验证时间戳在 IS 范围内"""
    if timestamps.max() > train_end_date:
        raise ValueError(
            f"检测到 OOS 时间戳: {timestamps.max()}"
        )

# 负结果保留检查
def validate_negative_results_retained(search_trajectory, all_attempts):
    """验证所有尝试都被记录"""
    if len(search_trajectory) < len(all_attempts):
        raise ValueError(
            f"轨迹记录不完整: {len(search_trajectory)} < {len(all_attempts)}"
        )

# 参数冻结检查
def validate_parameters_frozen(spec, modification_attempt):
    """防止修改已冻结的参数"""
    if modification_attempt in spec.frozen_items:
        raise ValueError(
            f"参数 {modification_attempt} 已冻结，不能修改"
        )
```

**流程层面**：
1. **时间戳验证**: 自动检查所有数据时间戳
2. **负结果保留**: 强制记录所有参数尝试
3. **代码审查**: 检查是否有隐性 OOS 使用
4. **自动化检查**: CI/CD 中集成校准检查

---

## 13. 输出 Artifact

### 13.1 机器可读产物

**必需文件**：
```yaml
machine_readable_artifacts:

  train_calibration_spec.yaml:
    description: "Train Calibration 冻结规范"
    content:
      - "信号阈值"
      - "分位切点"
      - "质量过滤标准"
      - "Regime 定义"
      - "信号参数"

  thresholds.yaml:
    description: "信号阈值定义"
    content:
      upper_threshold: float
      lower_threshold: float
      method: str

  quantiles.yaml:
    description: "分位切点定义"
    content:
      cuts: list[float]
      n_quantiles: int
      method: str

  quality_filters.yaml:
    description: "质量过滤标准"
    content:
      min_coverage: float
      max_staleness: float
      max_outlier_rate: float
      min_signal_strength: float

  regime_definition.yaml:
    description: "Regime 定义"
    content:
      indicator: str
      thresholds: dict
      cut_points: list[float]

  parameter_search_trajectory.csv:
    description: "参数搜索轨迹"
    schema:
      - run_id
      - timestamp
      - stage
      - params
      - score
      - metrics
      - evaluation_time
      - is_best

  calibration_results.parquet:
    description: "校准结果（原始数据）"
    schema:
      - timestamp
      - symbol
      - signal
      - thresholded_signal
      - quantile
      - regime
```

### 13.2 人类可读产物

**必需文档**：
```yaml
human_readable_artifacts:

  train_calibration_report.md:
    description: "Train Calibration 阶段总结报告"
    sections:
      - "执行摘要"
      - "校准方法"
      - "核心发现"
      - "冻结内容"
      - "下一步计划"

  field_dictionary.md:
    description: "字段字典"
    content:
      - "所有机器可读产物的字段说明"
      - "字段类型、含义、单位"

  artifact_catalog.md:
    description: "产物目录"
    content:
      - "产物列表"
      - "用途说明"
      - "消费者说明"

  gate_decision.md:
    description: "门禁决策文档"
    content:
      - "stage: 03_train_calibration"
      - "status: PASS / CONDITIONAL_PASS / ..."
      - "decision_basis: 决策依据"
      - "frozen_scope: 冻结范围"
      - "next_steps: 下一步行动"
```

---

## 14. 与 Test Evidence 的交接

### 14.1 交接内容

**Frozen Spec (冻结规范)**：
```yaml
frozen_spec_handover:
  metadata:
    from_stage: "03_train_calibration"
    to_stage: "04_test_evidence"
    frozen_at: "2026-03-26"
    lineage_id: "<lineage_id>"
    run_id: "<run_id>"

  signal_thresholds:
    file: "thresholds.yaml"
    key: "thresholds"
    frozen: true
    description: "信号判断阈值"

  quantile_cuts:
    file: "quantiles.yaml"
    key: "cuts"
    frozen: true
    description: "分位数切点"

  quality_filters:
    file: "quality_filters.yaml"
    key: "filters"
    frozen: true
    description: "质量过滤标准"

  regime_definition:
    file: "regime_definition.yaml"
    key: "regime"
    frozen: true
    description: "Regime 切分规则"

  signal_parameters:
    file: "train_calibration_spec.yaml"
    key: "param_id"
    frozen: true
    description: "信号参数组合"
```

### 14.2 交接验证

**Test Evidence 阶段职责**：
1. **验证冻结内容**: 确认接收的 Frozen Spec 完整
2. **不重估参数**: 严格按照 Train 冻结的参数验证
3. **不修改阈值**: 使用 Train 冻结的阈值
4. **不调整标准**: 使用 Train 冻结的质量标准

**交接清单**：
```yaml
handover_checklist:

  frozen_spec_complete:
    items:
      - "阈值已提供"
      - "分位切点已提供"
      - "质量标准已提供"
      - "Regime 定义已提供"
    status: "✓ / ✗"

  calibration_results_available:
    items:
      - "校准结果文件存在"
      - "参数搜索轨迹存在"
      - "校准报告存在"
    status: "✓ / ✗"

  documentation_complete:
    items:
      - "Calibration 报告完整"
      - "字段字典完整"
      - "产物目录完整"
      - "门禁决策清晰"
    status: "✓ / ✗"

  handover_approved:
    approvers:
      - "Quant Researcher (提供方)"
      - "Quant Researcher (接收方)"
    status: "✓ / ✗"
```

### 14.3 禁止事项

**Test Evidence 不能做的事**：
```yaml
forbidden_in_test:
  parameter_reestimation: "禁止重新估计参数"
  threshold_adjustment: "禁止调整 Train 冻结的阈值"
  quantile_recalculation: "禁止重新计算分位切点"
  quality_standard_change: "禁止修改质量标准"
  regime_redefinition: "禁止重新定义 Regime"
```

---

## 15. 校准报告模板

### 15.1 Train Calibration 报告模板

```markdown
---
doc_id: TC-{lineage_id}-{run_id}
title: Train Calibration Report — {策略名称}
date: YYYY-MM-DD
status: PASS | CONDITIONAL_PASS | RETRY | RESEARCH_AGAIN | NO_GO
owner: Quant Researcher
lineage_id: {lineage_id}
run_id: {run_id}
---

## 1. 执行摘要

**校准结论**: {一句话总结}

**核心校准结果**:
- 信号阈值: 上限 {X.XX}, 下限 {X.XX}
- 分位切点: {切点列表}
- 质量标准: 覆盖率 ≥ {XX}%, outlier率 ≤ {XX}%
- Regime 定义: {Regime 描述}

**决策**: {PASS/CONDITIONAL_PASS/RETRY/RESEARCH_AGAIN/NO_GO}

## 2. 校准方法

### 2.1 数据范围
- **IS 期**: {YYYY-MMDD} 至 {YYYY-MMDD}
- **样本量**: {N} 条记录
- **标的数**: {N} 个

### 2.2 校准步骤
1. 信号阈值校准
2. 分位切点校准
3. 质量过滤标准设定
4. Regime 切分定义
5. 参数空间搜索

### 2.3 搜索策略
- **粗网格**: {描述}
- **精细搜索**: {描述}
- **总评估次数**: {N}

## 3. 核心发现

### 3.1 信号阈值
- **上限阈值**: {X.XX} ({方法})
- **下限阈值**: {X.XX} ({方法})
- **中性区**: [{X.XX}, {X.XX}]
- **验证**: {通过/未通过}

### 3.2 分位切点
| 分位 | 切点 | 样本量 | 平均收益 |
|-----|------|--------|---------|
| Q5 | {X.XX} | {N} | {X.XX}% |
| Q4 | {X.XX} | {N} | {X.XX}% |
| Q3 | {X.XX} | {N} | {X.XX}% |
| Q2 | {X.XX} | {N} | {X.XX}% |
| Q1 | {X.XX} | {N} | {X.XX}% |

**单调性检验**: {通过/未通过}

### 3.3 质量标准
- **最小覆盖率**: {XX.XX}%
- **最大停滞率**: {XX.XX}%
- **最大异常值率**: {XX.XX}%
- **最小信号强度**: {X.XX}

**设定依据**: {领域知识/分布分析/混合}

### 3.4 Regime 定义
- **指标**: {指标名称}
- **状态数**: {N}
- **切点**: {切点列表}

### 3.5 参数搜索结果
- **最优参数**: {参数字典}
- **最优分数**: {X.XX}
- **搜索范围**: {描述}

## 4. 参数搜索轨迹

### 4.1 搜索概览
- **总评估次数**: {N}
- **粗网格**: {N} 次
- **精细搜索**: {N} 次
- **局部优化**: {N} 次

### 4.2 参数敏感性
| 参数 | 敏感性 | 稳定性 | 最优值 |
|-----|--------|--------|--------|
| {param1} | {高/中/低} | {X.XX} | {value} |
| {param2} | {高/中/低} | {X.XX} | {value} |

### 4.3 搜索可视化
{附搜索轨迹图}

## 5. 稳健性分析

### 5.1 阈值稳健性
- **Bootstrap 验证**: {通过/未通过}
- **交叉验证**: {通过/未通过}
- **时间子样本**: {通过/未通过}

### 5.2 分位稳健性
- **分布均衡**: {ratio} (≥ 0.5 为合格)
- **切点稳定性**: {最大 std} (≤ 0.05 为合格)

### 5.3 Regime 稳定性
- **状态持久性**: {平均持续时间}
- **转换频率**: {描述}

## 6. 冻结内容

### 6.1 Frozen Spec
```yaml
thresholds:
  upper: {value}
  lower: {value}

quantiles:
  cuts: [{values}]

quality_filters:
  min_coverage: {value}
  max_outlier_rate: {value}

regime_definition:
  indicator: {name}
  cut_points: [{values}]

signal_parameters:
  param_id: {id}
  params: {dict}
```

### 6.2 下一步不能改的内容
- [ ] 不能重新估计阈值
- [ ] 不能重新计算分位切点
- [ ] 不能修改质量标准
- [ ] 不能重新定义 Regime
- [ ] 不能使用 OOS 数据调整参数

## 7. 局限性与风险

### 7.1 校准局限
{描述校准的局限性}

### 7.2 潜在风险
{描述潜在风险}

### 7.3 需要在 Test Evidence 验证的
{需要验证的内容}

## 8. 下一步计划

### 8.1 进入 Test Evidence
**前提条件**:
- [ ] 所有 Formal Gate 通过
- [ ] Frozen Spec 已生成
- [ ] 交接验证完成

### 8.2 Test Evidence 重点
{需要在 Test Evidence 重点关注的内容}

## 9. 附录

### 9.1 详细数据
{附详细数据表}

### 9.2 图表
{附可视化图表}

### 9.3 负结果记录
{附被拒绝的参数组合}
```

### 15.2 Gate Decision 模板

```markdown
---
doc_id: GD-{lineage_id}-{run_id}
title: Gate Decision — Train Calibration
date: YYYY-MM-DD
stage: 03_train_calibration
lineage_id: {lineage_id}
run_id: {run_id}
---

## 决策状态

**状态**: {PASS / CONDITIONAL_PASS / RETRY / RESEARCH_AGAIN / NO_GO}

## 决策依据

### Formal Gate 检查结果

| 检查项 | 要求 | 结果 | 状态 |
|--------|------|------|------|
| 阈值已校准 | 必需 | {是/否} | {PASS/FAIL} |
| 分位切点已校准 | 必需 | {是/否} | {PASS/FAIL} |
| 质量标准已设定 | 必需 | {是/否} | {PASS/FAIL} |
| Regime 已定义 | 如适用 | {是/否} | {PASS/FAIL} |
| 参数搜索已完成 | 必需 | {是/否} | {PASS/FAIL} |
| 只用 IS 数据 | 必需 | {是/否} | {PASS/FAIL} |
| Frozen Spec 已生成 | 必需 | {是/否} | {PASS/FAIL} |
| 负结果已保留 | 必需 | {是/否} | {PASS/FAIL} |

### Audit Gate 检查结果
{补充性检查结果，不直接阻断}

## 冻结范围

**已冻结内容**:
- 信号阈值: {上限/下限}
- 分位切点: {切点}
- 质量标准: {标准}
- Regime 定义: {定义}
- 信号参数: {param_id}

**验证依据**:
- 文件: train_calibration_spec.yaml
- 路径: {path}
- 校验和: {checksum}

## 下一步不能改的内容

**禁止事项**:
- [ ] 不能在 Test 上重新估计阈值
- [ ] 不能在 Test 上重新计算分位切点
- [ ] 不能在 Test 上修改质量标准
- [ ] 不能在 Test 上重新定义 Regime
- [ ] 不能使用 OOS 数据调整任何参数

## 决策理由

{详细说明为什么做出这个决策}

## 后续行动

### 如果 PASS / CONDITIONAL_PASS:
- 生成 Frozen Spec
- 与 Test Evidence 交接
- 提供完整文档

### 如果 RETRY:
- 明确 retry 范围
- 修改内容清单
- 预期改善

### 如果 RESEARCH_AGAIN:
- 回滚阶段: {stage}
- 允许修改: {...}
- 禁止修改: {...}

### 如果 NO_GO:
- 终止原因
- 经验教训
- 归档建议

## 审批

**Quant Researcher**: {签名} {日期}
**Reviewer**: {签名} {日期}
**PM / Risk**: {签名} {日期}
```

---

## 附录：关键检查清单

### A. Train Calibration 阶段启动检查

- [ ] Signal Ready 阶段已完成并 PASS
- [ ] Signal Ready Frozen Spec 已接收并验证
- [ ] IS 数据已准备并验证
- [ ] 时间切分已确认
- [ ] 参数清单已确认

### B. Train Calibration 阶段执行检查

- [ ] 只使用 IS 数据进行校准
- [ ] 完成阈值校准
- [ ] 完成分位切点校准
- [ ] 完成质量标准设定
- [ ] 完成 Regime 定义（如适用）
- [ ] 完成参数空间搜索
- [ ] 记录完整搜索轨迹
- [ ] 保留所有负结果

### C. Train Calibration 阶段完成检查

- [ ] 所有 Formal Gate 通过
- [ ] 校准报告已完成
- [ ] Frozen Spec 已生成
- [ ] 产物目录已完成
- [ ] 字段字典已完成
- [ ] Gate Decision 已记录
- [ ] 与 Test Evidence 交接完成

---

**文档版本**: v1.0
**最后更新**: 2026-03-26
**下次评审**: 2026-06-26
