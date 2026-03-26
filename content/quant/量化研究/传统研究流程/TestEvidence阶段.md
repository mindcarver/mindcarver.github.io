# Test Evidence 阶段详细文档

**文档ID**: QSP-TE-v1.0
**阶段编号**: 04
**阶段名称**: Test Evidence (测试证据)
**日期**: 2026-03-26
**状态**: v1.0
**负责角色**: Quant Researcher

---

## 目录

1. [阶段定义与核心目的](#1-阶段定义与核心目的)
2. [关键约束与纪律](#2-关键约束与纪律)
3. [验证内容清单](#3-验证内容清单)
4. [Whitelist 确定](#4-whitelist-确定)
5. [Best Horizon 确定](#5-best-horizon-确定)
6. [Spread-unit 分析](#6-spread-unit-分析)
7. [拥挤度分析](#7-拥挤度分析)
8. [Alpha 来源分析](#8-alpha-来源分析)
9. [Formal Gate 要求](#9-formal-gate-要求)
10. [常见错误与防范](#10-常见错误与防范)
11. [输出 Artifact](#11-输出-artifact)
12. [与 Backtest Ready 的交接](#12-与-backtest-ready-的交接)
13. [验证报告模板](#13-验证报告模板)

---

## 1. 阶段定义与核心目的

### 1.1 阶段定义

**Test Evidence (测试证据)** 是在独立样本上验证冻结后的信号结构是否成立的阶段。

**核心职责**：
- 在 Out-of-Sample (OOS) 数据上验证 Train Calibration 阶段冻结的信号结构
- 确认信号方向、分层收益、风险指标在独立样本上仍然成立
- 回答"Alpha 是否真实存在"的问题，而非"Alpha 能赚多少钱"

### 1.2 与 Train Calibration 的关系

| 维度 | Train Calibration | Test Evidence |
|------|-------------------|---------------|
| **数据范围** | In-Sample (IS) | Out-of-Sample (OOS) |
| **职责** | "定尺子" | "验证尺子" |
| **冻结内容** | 阈值、分位切点、质量过滤标准 | 复用 Train 的所有冻结内容 |
| **禁忌** | 不能使用 Test 结果 | 不能重估 Train 的尺子 |
| **验证目标** | 确定信号参数 | 验证信号泛化能力 |

### 1.3 核心目的

**验证什么**：
1. **方向正确性**：信号方向在 OOS 上是否与 IS 一致
2. **结构稳定性**：分层收益结构是否保持
3. **风险可控性**：Sharpe、Drawdown 等指标是否可接受
4. **拥挤度风险**：Alpha 是否来自拥挤策略
5. **Alpha 来源**：明确收益的经济学来源

**不验证什么**：
- 绝对收益最大化（这是 Backtest 的职责）
- 交易规则可行性（这是 Backtest 的职责）
- 执行成本精确性（这是 Backtest 的职责）

---

## 2. 关键约束与纪律

### 2.1 铁律：只能复用 Train 冻结的尺子

**冻结内容清单**：
```yaml
train_frozen_items:
  thresholds:
    - 信号阈值
    - 分位切点 (quantile cuts)
    - 质量过滤标准
    - 异常值边界

  parameters:
    - 参数组合 (param_id)
    - 时间窗口参数
    - 标准化方法
    - 转换函数参数

  universe:
    - 标的集合定义
    - 准入条件
    - 退出条件

  time_split:
    - Train/Test 时间切分点
    - Regime 切分规则
```

**禁忌清单**：
```yaml
forbidden_actions:
  parameter_reestimation: "禁止在 OOS 上重新估计参数"
  threshold_adjustment: "禁止调整 Train 确定的阈值"
  universe_modification: "禁止在 Test 阶段修改 Universe"
  time_split_change: "禁止改变 Train/Test 时间切分"
  cherry_picking: "禁止挑选表现好的参数组合"
```

### 2.2 OOS 数据独立性原则

**时间切分规范**：
```
Train: [T_start, T_train_end]
Test:  (T_train_end, T_test_end]
Backtest: (T_test_end, T_backtest_end]
Holdout: (T_backtest_end, T_holdout_end]
```

**关键要求**：
1. Test 数据完全未参与 Train 阶段
2. 不能用 Test 结果指导 Train 参数调整
3. Test 窗口一旦确定，不能移动

### 2.3 验证纪律

**先验证，再解释**：
1. 先用冻结的尺子在 OOS 上验证
2. 如果验证失败，分析原因
3. 不能先看结果，再决定用什么尺子

**负结果保留**：
- 失败的验证结果必须记录
- 不能只保留成功的参数组合
- 避免幸存者偏差

---

## 3. 验证内容清单

### 3.1 信号方向验证

**验证目标**：确认信号在 OOS 上的预测方向与 IS 一致

**验证方法**：
```python
# 方向一致性检验
def validate_direction(signal, returns, train_threshold):
    """
    验证 OOS 上的信号方向

    Args:
        signal: OOS 信号值
        returns: OOS 收益率
        train_threshold: Train 冻结的阈值

    Returns:
        direction_consistency: 方向一致性指标
    """
    # 使用 Train 冻结的阈值
    oos_signal_binary = (signal > train_threshold).astype(int)

    # 计算方向一致性
    direction_correct = (oos_signal_binary * np.sign(returns) > 0).mean()

    return direction_correct
```

**Formal Gate**：
- OOS 方向一致性 ≥ 55% (可调整)
- 不能出现方向反转（< 50%）

### 3.2 分层收益验证

**验证目标**：确认信号分层的收益结构在 OOS 上保持稳定

**分层方法**：
```python
def validate_layering(signal, returns, train_quantiles):
    """
    验证分层收益结构

    Args:
        signal: OOS 信号值
        returns: OOS 收益率
        train_quantiles: Train 冻结的分位切点

    Returns:
        layer_returns: 各层收益
        monotonicity: 单调性检验
    """
    # 使用 Train 冻结的分位切点
    layers = pd.cut(signal, bins=train_quantiles, labels=False)

    # 计算各层收益
    layer_returns = returns.groupby(layers).mean()

    # 检验单调性
    monotonicity = (layer_returns.diff().dropna() > 0).all()

    return layer_returns, monotonicity
```

**Formal Gate**：
- 分层收益保持单调性（多头信号）
- Q5-Q1 收益差显著为正
- 不能出现分层倒置

### 3.3 Sharpe/Drawdown 验证

**验证目标**：确认风险调整后收益在 OOS 上可接受

**计算方法**：
```python
def validate_risk_metrics(returns, annualization_factor=252):
    """
    计算风险指标

    Args:
        returns: OOS 收益序列
        annualization_factor: 年化因子

    Returns:
        sharpe: 夏普比率
        max_drawdown: 最大回撤
        win_rate: 胜率
    """
    # 年化收益
    annual_return = returns.mean() * annualization_factor

    # 年化波动率
    annual_vol = returns.std() * np.sqrt(annualization_factor)

    # Sharpe Ratio (假设无风险利率为0)
    sharpe = annual_return / annual_vol

    # 最大回撤
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # 胜率
    win_rate = (returns > 0).mean()

    return {
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'annual_return': annual_return,
        'annual_vol': annual_vol
    }
```

**Formal Gate**：
- Sharpe Ratio ≥ 1.0 (最低要求)
- Max Drawdown ≤ -20% (可调整)
- 胜率 ≥ 52%

---

## 4. Whitelist 确定

### 4.1 定义

**Whitelist (白名单)** 是通过验证、允许进入 Backtest 的标的集合。

**关键原则**：
- 在 Test Evidence 阶段冻结
- Backtest 阶段不能重选
- 防止在回测上重新选币

### 4.2 确定方法

**基于信号质量的过滤**：
```python
def determine_whitelist(signal_quality_metrics, quality_thresholds):
    """
    基于 Train 冻结的质量标准确定 Whitelist

    Args:
        signal_quality_metrics: 各标的信号质量指标
        quality_thresholds: Train 冻结的质量阈值

    Returns:
        whitelist: 通过验证的标的列表
    """
    # 应用 Train 冻结的质量标准
    pass_coverage = (signal_quality_metrics['coverage'] >=
                     quality_thresholds['min_coverage'])
    pass_staleness = (signal_quality_metrics['staleness'] <=
                      quality_thresholds['max_staleness'])
    pass_outlier = (signal_quality_metrics['outlier_rate'] <=
                    quality_thresholds['max_outlier_rate'])

    # 综合判断
    whitelist = signal_quality_metrics.index[
        pass_coverage & pass_staleness & pass_outlier
    ]

    return whitelist.tolist()
```

**质量指标**：
- **Coverage**: 数据覆盖率 ≥ 80%
- **Staleness**: 停滞数据比例 ≤ 10%
- **Outlier Rate**: 异常值比例 ≤ 5%
- **Signal Strength**: 信号强度符合要求

### 4.3 Whitelist 冻结

**冻结内容**：
```yaml
whitelist_spec:
  version: "v1.0"
  frozen_at: "Test Evidence 阶段"
  symbols: ["BTC_USDT", "ETH_USDT", ...]
  total_count: 42

  quality_criteria:
    min_coverage: 0.8
    max_staleness: 0.1
    max_outlier_rate: 0.05

  rejected_symbols:
    - symbol: "XYZ_USDT"
      reason: "覆盖率不足 (65%)"
    - symbol: "ABC_USDT"
      reason: "信号强度不足"
```

**Formal Gate**：
- Whitelist 必须在 Test Evidence 结束时冻结
- 必须记录被拒绝标的及其原因
- Backtest 阶段只能使用 Whitelist 中的标的

---

## 5. Best Horizon 确定

### 5.1 定义

**Best Horizon (最佳预测期)** 是信号预测效果最好的时间周期。

**关键原则**：
- 在 Test Evidence 阶段确定
- 决定持仓周期和交易频率
- Backtest 阶段不能重估

### 5.2 确定方法

**IC 分析**：
```python
def determine_best_horizon(signal, returns, horizons, train_best_horizon=None):
    """
    确定 Best Horizon

    Args:
        signal: 信号值
        returns: 收益率序列
        horizons: 候选预测期列表
        train_best_horizon: Train 确定的最佳预测期（可选）

    Returns:
        best_horizon: 最佳预测期
        ic_analysis: IC 分析结果
    """
    ic_results = {}

    for h in horizons:
        # 计算 h 期后的收益
        forward_returns = returns.shift(-h)

        # 计算 IC (Information Coefficient)
        ic = signal.corr(forward_returns)

        # 计算 IC 的统计显著性
        n = signal.notna().sum()
        t_stat = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))

        ic_results[h] = {
            'ic': ic,
            't_stat': t_stat,
            'p_value': p_value,
            'ir': ic / forward_returns.std()  # Information Ratio
        }

    # 选择最佳 Horizon (IC 绝对值最大且显著)
    significant_horizons = {
        h: metrics for h, metrics in ic_results.items()
        if metrics['p_value'] < 0.05
    }

    if significant_horizons:
        best_horizon = max(significant_horizons.items(),
                          key=lambda x: abs(x[1]['ic']))[0]
    else:
        best_horizon = train_best_horizon  # 退回到 Train 的选择

    return best_horizon, ic_results
```

**Formal Gate**：
- Best Horizon 必须基于 OOS 上的 IC 分析
- IC 必须统计显著 (p < 0.05)
- 不能在 Backtest 阶段重新估计

### 5.3 Horizon 稳定性检验

**稳健性检查**：
- 相邻 Horizon 的 IC 应该平滑变化
- 不能出现单一 Horizon 突然最优的情况
- 检查不同 Regime 下的 Horizon 稳定性

---

## 6. Spread-unit 分析

### 6.1 定义

**Spread-unit (价差单位)** 是用价格变动而非资金计算收益的单位。

**关键限制**：
- 不能用于冒充正式回测收益
- 回测收益必须基于正式资金记账口径
- 仅用于信号验证和结构分析

### 6.2 计算方法

**标准化价差单位**：
```python
def calculate_spread_unit_returns(signals, prices, capital_base=10000):
    """
    计算价差单位收益

    Args:
        signals: 信号值 (已标准化)
        prices: 价格序列
        capital_base: 名义本金 (仅用于标准化)

    Returns:
        spread_returns: 价差单位收益
    """
    # 计算价差 (价格变化)
    price_diff = prices.diff()

    # 信号标准化 (假设已标准化为均值0、标准差1)
    # 标准化信号 × 价格变化 = 价差单位收益
    position_units = signals  # 单位: 标准化信号
    spread_returns = (position_units.shift(1) * price_diff) / capital_base

    return spread_returns
```

**与资金记账的区别**：
```python
# 价差单位收益 (仅用于验证)
spread_unit_return = signal * price_change

# 资金记账收益 (用于正式回测)
capital_return = (capital * signal * price_change -
                  transaction_cost - slippage) / capital
```

### 6.3 使用场景

**适用场景**：
- 信号方向验证
- 分层收益分析
- IC/IR 计算
- Alpha 来源分析

**不适用场景**：
- 正式回测收益报告
- 策略容量评估
- 风险指标计算 (Drawdown 等)

**Formal Gate**：
- 必须明确标注使用的是价差单位
- 不能用价差单位收益冒充资金收益
- 正式回测必须使用资金记账口径

---

## 7. 拥挤度分析

### 7.1 定义

**Crowding (拥挤度)** 是与已知拥挤策略或风格暴露的重叠程度。

**风险**：
- 可能导致同时平仓
- 收益归零
- 流动性风险

### 7.2 分析方法

**与已知策略的相关性**：
```python
def analyze_crowding(signal, known_strategy_signals):
    """
    分析信号拥挤度

    Args:
        signal: 当前信号
        known_strategy_signals: 已知策略信号字典

    Returns:
        crowding_metrics: 拥挤度指标
    """
    crowding_metrics = {}

    for strategy_name, known_signal in known_strategy_signals.items():
        # 计算信号相关性
        correlation = signal.corr(known_signal)

        # 计算信号重叠度 (同向比例)
        overlap = ((signal > 0) & (known_signal > 0)).mean() + \
                  ((signal < 0) & (known_signal < 0)).mean()

        crowding_metrics[strategy_name] = {
            'correlation': correlation,
            'overlap': overlap,
            'risk_level': assess_crowding_risk(correlation, overlap)
        }

    return crowding_metrics

def assess_crowding_risk(correlation, overlap):
    """
    评估拥挤度风险等级
    """
    if abs(correlation) > 0.8 and overlap > 0.7:
        return 'HIGH'
    elif abs(correlation) > 0.6 and overlap > 0.6:
        return 'MEDIUM'
    else:
        return 'LOW'
```

**风格暴露分析**：
```python
def analyze_style_exposure(signal, style_factors):
    """
    分析风格因子暴露

    Args:
        signal: 当前信号
        style_factors: 风格因子 (价值、动量、质量等)

    Returns:
        style_exposure: 风格暴露度
    """
    exposure = {}

    for factor_name, factor in style_factors.items():
        # 回归分析
        model = sm.OLS(signal, sm.add_constant(factor)).fit()
        exposure[factor_name] = {
            'beta': model.params[1],
            't_stat': model.tvalues[1],
            'p_value': model.pvalues[1],
            'r_squared': model.rsquared
        }

    return exposure
```

### 7.3 拥挤度报告

**报告模板**：
```yaml
crowding_report:
  overall_risk: "MEDIUM"

  strategy_overlap:
    - strategy: "Momentum_1M"
      correlation: 0.65
      overlap: 0.72
      risk_level: "MEDIUM"
    - strategy: "Mean_Reversion_Short"
      correlation: -0.25
      overlap: 0.45
      risk_level: "LOW"

  style_exposure:
    - factor: "Momentum"
      beta: 0.58
      t_stat: 3.21
      p_value: 0.001
      significant: true
    - factor: "Value"
      beta: 0.12
      t_stat: 0.85
      p_value: 0.395
      significant: false

  recommendation: >
    信号与短期动量策略存在中等相关性，
    建议在 Backtest 中评估流动性风险。
```

**Formal Gate**：
- 必须完成拥挤度分析
- 高拥挤度必须记录风险提示
- 不能忽略拥挤度风险

---

## 8. Alpha 来源分析

### 8.1 目的

**Alpha 来源分析** 旨在明确收益的经济学来源，回答"为什么这个策略能赚钱"。

### 8.2 分析方法

**收益归因**：
```python
def attribute_alpha(signal, returns, factors):
    """
    Alpha 收益归因

    Args:
        signal: 信号
        returns: 收益率
        factors: 风险因子 (市场、行业、风格等)

    Returns:
        attribution: 收益归因结果
    """
    # 构建回归模型
    X = pd.DataFrame(factors)
    X['signal'] = signal
    X = sm.add_constant(X)

    model = sm.OLS(returns, X).fit()

    # 分解收益来源
    attribution = {
        'alpha': model.params['signal'],
        'alpha_t_stat': model.tvalues['signal'],
        'alpha_p_value': model.pvalues['signal'],
        'factor_exposures': {
            factor: {
                'beta': model.params[factor],
                'contribution': model.params[factor] * X[factor].mean()
            }
            for factor in factors.keys()
        },
        'r_squared': model.rsquared
    }

    return attribution
```

**机制分析**：
1. **风险补偿**: 承担某种系统性风险的补偿
2. **行为偏差**: 利用市场参与者行为偏差
3. **信息优势**: 拥有独特信息或处理能力
4. **流动性提供**: 为市场提供流动性的补偿

### 8.3 Alpha 真实性检验

**稳健性检验清单**：
```yaml
robustness_checks:
  subsample_stability:
    - 将 OOS 期分为多个子样本
    - 检查各子样本上 Alpha 稳定性
    - 不能只在单一子样本上成立

  regime_stability:
    - 不同市场状态下 Alpha 表现
    - 牛市/熊市/震荡市
    - 高波动/低波动环境

  cross_universe:
    - 不同子集标的上的 Alpha 表现
    - 不能只在少数标的上成立

  alternative_spec:
    - 不同参数设定下的稳健性
    - 稍微改变参数，Alpha 不应消失
```

**Formal Gate**：
- 必须明确 Alpha 的经济学来源
- 必须通过稳健性检验
- 不能接受"数据挖掘"式的 Alpha

---

## 9. Formal Gate 要求

### 9.1 门禁检查清单

**PASS 条件**：
```yaml
formal_gate_requirements:

  signal_direction:
    requirement: "OOS 方向一致性 ≥ 55%"
    evidence: "direction_consistency_metric"
    status: "PASS / FAIL"

  layering_monotonicity:
    requirement: "分层收益保持单调性"
    evidence: "layer_returns_analysis"
    status: "PASS / FAIL"

  risk_metrics:
    sharpe_requirement: "Sharpe ≥ 1.0"
    drawdown_requirement: "Max Drawdown ≤ -20%"
    win_rate_requirement: "Win Rate ≥ 52%"
    evidence: "risk_metrics_report"
    status: "PASS / FAIL"

  whitelist_frozen:
    requirement: "Whitelist 已确定并冻结"
    evidence: "whitelist_spec.yaml"
    status: "PASS / FAIL"

  best_horizon_determined:
    requirement: "Best Horizon 已确定且显著"
    evidence: "ic_analysis_report"
    status: "PASS / FAIL"

  crowding_analyzed:
    requirement: "拥挤度分析已完成"
    evidence: "crowding_report"
    status: "PASS / FAIL"

  alpha_source_identified:
    requirement: "Alpha 来源已明确"
    evidence: "alpha_attribution_report"
    status: "PASS / FAIL"

  no_reestimation:
    requirement: "未重估 Train 冻结的参数"
    evidence: "code_review + parameter_manifest"
    status: "PASS / FAIL"
```

### 9.2 决策状态

**可能的状态**：
```yaml
verdict_states:

  PASS:
    description: "所有 Formal Gate 通过，可进入 Backtest Ready"
    frozen_items:
      - "Whitelist (标的列表)"
      - "Best Horizon (预测期)"
      - "信号参数 (train_param_id)"
      - "质量阈值 (quality_thresholds)"

  CONDITIONAL_PASS:
    description: "核心验证通过，但有需记录的风险"
    conditions:
      - "拥挤度风险需在 Backtest 中评估"
      - "某些 Regime 下表现不佳需记录"
      - "Alpha 来源需进一步验证"

  RETRY:
    description: "实现问题修复后可重试"
    scope:
      - "代码实现 bug 修复"
      - "数据处理错误修复"
      - "计算逻辑修正"
    forbidden:
      - "不能改变研究主问题"
      - "不能重估 Train 参数"

  RESEARCH_AGAIN:
    description: "证据不足，需回到 Train 或更早阶段"
    rollback_stage: "03_train_freeze 或更早"
    reasons:
      - "OOS 方向不一致"
      - "分层结构倒置"
      - "Alpha 不稳健"

  NO_GO:
    description: "策略假设不成立，终止研究线"
    reasons:
      - "OOS 上完全无 Alpha"
      - "拥挤度过高且无法规避"
      - "Alpha 来源可疑或无法解释"
```

---

## 10. 常见错误与防范

### 10.1 错误清单

| 错误类型 | 描述 | 后果 | 防范措施 |
|---------|------|------|---------|
| **重估参数** | 在 OOS 上重新估计参数 | 样本污染 | 使用参数清单，锁定 Train 结果 |
| **挑选参数** | 挑选 OOS 表现好的参数组合 | 幸存者偏差 | 保留所有参数组合结果 |
| **移动切分** | 调整 Train/Test 时间切分 | 样本泄漏 | 时间切分在 Mandate 冻结 |
| **重新选币** | 在 Test 阶段重新筛选标的 | 过拟合 | Whitelist 冻结机制 |
| **忽略拥挤度** | 不分析拥挤度风险 | 潜在归零 | 强制拥挤度分析 |
| **混淆单位** | 用价差单位冒充资金收益 | 虚假高收益 | 明确标注单位类型 |
| **选择性报告** | 只报告好的结果 | 幸存者偏差 | 负结果保留纪律 |

### 10.2 防范机制

**代码层面**：
```python
# 参数冻结检查
def validate_train_frozen(param_id, train_params):
    """
    验证参数是否来自 Train 冻结
    """
    if param_id not in train_params:
        raise ValueError(f"参数 {param_id} 不在 Train 冻结清单中")

    return train_params[param_id]

# 时间切分检查
def validate_time_split(test_start_date, mandate_config):
    """
    验证时间切分是否符合 Mandate
    """
    if test_start_date != mandate_config['test_start']:
        raise ValueError(
            f"Test 开始时间 {test_start_date} 与 Mandate "
            f"冻结时间 {mandate_config['test_start']} 不一致"
        )

# Whitelist 冻结检查
def validate_whitelist(symbols, test_whitelist):
    """
    验证是否使用 Test 冻结的 Whitelist
    """
    extra_symbols = set(symbols) - set(test_whitelist)
    if extra_symbols:
        raise ValueError(
            f"发现 Whitelist 外的标的: {extra_symbols}"
        )
```

**流程层面**：
1. **参数清单**: 所有参数必须有 Param ID
2. **代码审查**: 检查是否有重估逻辑
3. **自动化检查**: CI/CD 中集成冻结检查
4. **负结果保留**: 强制记录失败结果

---

## 11. 输出 Artifact

### 11.1 机器可读产物

**必需文件**：
```yaml
machine_readable_artifacts:

  test_evidence_spec.yaml:
    description: "Test Evidence 阶段冻结规范"
    content:
      - "Whitelist (标的列表)"
      - "Best Horizon (预测期)"
      - "质量阈值"
      - "Regime 切分规则"

  test_results.parquet:
    description: "OOS 验证结果 (原始数据)"
    schema:
      - "timestamp"
      - "symbol"
      - "signal"
      - "forward_return"
      - "layer"
      - "regime"

  ic_analysis.csv:
    description: "IC 分析结果"
    schema:
      - "horizon"
      - "ic"
      - "t_stat"
      - "p_value"
      - "ir"

  risk_metrics.json:
    description: "风险指标汇总"
    content:
      - "sharpe"
      - "max_drawdown"
      - "win_rate"
      - "annual_return"
      - "annual_vol"

  crowding_report.yaml:
    description: "拥挤度分析报告"
    content:
      - "strategy_overlap"
      - "style_exposure"
      - "risk_assessment"

  alpha_attribution.json:
    description: "Alpha 收益归因"
    content:
      - "alpha"
      - "factor_exposures"
      - "attribution_breakdown"
```

### 11.2 人类可读产物

**必需文档**：
```yaml
human_readable_artifacts:

  test_evidence_report.md:
    description: "Test Evidence 阶段总结报告"
    sections:
      - "执行摘要"
      - "验证方法"
      - "核心发现"
      - "风险评估"
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
      - "stage: 04_test_evidence"
      - "status: PASS / CONDITIONAL_PASS / RETRY / ..."
      - "decision_basis: 决策依据"
      - "frozen_scope: 冻结范围"
      - "next_steps: 下一步行动"
```

---

## 12. 与 Backtest Ready 的交接

### 12.1 交接内容

**Frozen Spec (冻结规范)**：
```yaml
frozen_spec_handover:
  metadata:
    from_stage: "04_test_evidence"
    to_stage: "05_backtest_ready"
    frozen_at: "2026-03-26"
    lineage_id: "<lineage_id>"
    run_id: "<run_id>"

  whitelist:
    file: "test_evidence_spec.yaml"
    key: "whitelist"
    frozen: true
    description: "允许进入回测的标的列表"

  best_horizon:
    file: "test_evidence_spec.yaml"
    key: "best_horizon"
    frozen: true
    description: "信号预测期"
    value: 5  # 例如: 5天

  signal_parameters:
    file: "train_calibration_spec.yaml"
    key: "param_id"
    frozen: true
    description: "信号参数组合"

  quality_thresholds:
    file: "train_calibration_spec.yaml"
    key: "quality_thresholds"
    frozen: true
    description: "质量过滤标准"

  regime_definition:
    file: "train_calibration_spec.yaml"
    key: "regime_thresholds"
    frozen: true
    description: "Regime 切分规则"

  validation_results:
    file: "test_results.parquet"
    description: "OOS 验证结果，供 Backtest 参考"
```

### 12.2 交接验证

**Backtest Ready 阶段职责**：
1. **验证冻结内容**: 确认接收的 Frozen Spec 完整
2. **不重估参数**: 严格按照 Test 冻结的参数执行
3. **不修改 Whitelist**: 只使用 Whitelist 中的标的
4. **不改变 Horizon**: 使用 Test 确定的 Best Horizon

**交接清单**：
```yaml
handover_checklist:

  frozen_spec_complete:
    items:
      - "Whitelist 已提供"
      - "Best Horizon 已提供"
      - "信号参数已提供"
      - "质量阈值已提供"
    status: "✓ / ✗"

  validation_results_available:
    items:
      - "Test 结果文件存在"
      - "IC 分析报告存在"
      - "风险指标存在"
      - "拥挤度报告存在"
    status: "✓ / ✗"

  documentation_complete:
    items:
      - "Test Evidence 报告完整"
      - "字段字典完整"
      - "产物目录完整"
      - "门禁决策清晰"
    status: "✓ / ✗"

  handover_approved:
    approvers:
      - "Quant Researcher (提供方)"
      - "Quant Dev (接收方)"
    status: "✓ / ✗"
```

### 12.3 禁止事项

**Backtest Ready 不能做的事**：
```yaml
forbidden_in_backtest:
  parameter_reestimation: "禁止重新估计参数"
  whitelist_modification: "禁止修改 Whitelist"
  horizon_change: "禁止改变 Best Horizon"
  threshold_adjustment: "禁止调整 Train/Test 冻结的阈值"
  universe_expansion: "禁止扩大标的范围"
  time_split_change: "禁止改变时间切分"
```

---

## 13. 验证报告模板

### 13.1 Test Evidence 报告模板

```markdown
---
doc_id: TE-{lineage_id}-{run_id}
title: Test Evidence Report — {策略名称}
date: YYYY-MM-DD
status: PASS | CONDITIONAL_PASS | RETRY | RESEARCH_AGAIN | NO_GO
owner: Quant Researcher
lineage_id: {lineage_id}
run_id: {run_id}
---

## 1. 执行摘要

**验证结论**: {一句话总结}

**核心发现**:
- OOS 方向一致性: {XX}% ({PASS/FAIL})
- Sharpe Ratio: {X.XX} ({PASS/FAIL})
- 最大回撤: {XX}% ({PASS/FAIL})
- 拥挤度风险: {LOW/MEDIUM/HIGH}

**冻结内容**:
- Whitelist: {N} 个标的
- Best Horizon: {N} 天
- 参数 ID: {param_id}

**决策**: {PASS/CONDITIONAL_PASS/RETRY/RESEARCH_AGAIN/NO_GO}

## 2. 验证方法

### 2.1 数据范围
- **Train 期**: {YYYY-MMDD} 至 {YYYY-MMDD}
- **Test 期**: {YYYY-MMDD} 至 {YYYY-MMDD}
- **标的数**: Train {N} 个, Test {N} 个

### 2.2 信号定义
- **信号公式**: {公式描述}
- **参数 ID**: {param_id}
- **标准化方法**: {方法描述}

### 2.3 验证步骤
1. 信号方向验证
2. 分层收益验证
3. 风险指标计算
4. Whitelist 确定
5. Best Horizon 确定
6. 拥挤度分析
7. Alpha 来源分析

## 3. 核心发现

### 3.1 信号方向
- **方向一致性**: {XX.XX}%
- **统计显著性**: {t-stat} (p = {p_value})
- **与 IS 对比**: {IS: XX%, OOS: XX%}

### 3.2 分层收益
| 层 | 平均收益 | t-stat | 胜率 |
|---|---------|--------|------|
| Q1 | {X.XX}% | {X.XX} | {XX}% |
| Q2 | {X.XX}% | {X.XX} | {XX}% |
| Q3 | {X.XX}% | {X.XX} | {XX}% |
| Q4 | {X.XX}% | {X.XX} | {XX}% |
| Q5 | {X.XX}% | {X.XX} | {XX}% |
| Q5-Q1 | {X.XX}% | {X.XX} | - |

**单调性检验**: {PASS/FAIL}

### 3.3 风险指标
- **Sharpe Ratio**: {X.XX}
- **年化收益**: {XX.XX}%
- **年化波动**: {XX.XX}%
- **最大回撤**: {XX.XX}%
- **胜率**: {XX.XX}%

### 3.4 IC 分析
| Horizon | IC | t-stat | p-value | IR |
|---------|-----|--------|---------|-----|
| 1d | {X.XX} | {X.XX} | {p} | {X.XX} |
| 3d | {X.XX} | {X.XX} | {p} | {X.XX} |
| 5d | {X.XX} | {X.XX} | {p} | {X.XX} |
| 10d | {X.XX} | {X.XX} | {p} | {X.XX} |

**Best Horizon**: {N} 天

## 4. Whitelist 确定

### 4.1 最终 Whitelist
- **标的数**: {N} 个
- **覆盖率**: {XX.XX}%
- **主要拒绝原因**:
  - 覆盖率不足: {N} 个
  - 信号强度不足: {N} 个
  - 数据质量问题: {N} 个

### 4.2 Whitelist 列表
{附详细标的列表}

## 5. 拥挤度分析

### 5.1 策略重叠
| 策略 | 相关性 | 重叠度 | 风险等级 |
|------|--------|--------|---------|
| {策略1} | {X.XX} | {XX%} | {LOW/MEDIUM/HIGH} |
| {策略2} | {X.XX} | {XX%} | {LOW/MEDIUM/HIGH} |

### 5.2 风格暴露
| 风格因子 | Beta | t-stat | 显著性 |
|---------|------|--------|--------|
| {因子1} | {X.XX} | {X.XX} | {是/否} |
| {因子2} | {X.XX} | {X.XX} | {是/否} |

**拥挤度总体评估**: {LOW/MEDIUM/HIGH}

## 6. Alpha 来源分析

### 6.1 收益归因
- **纯 Alpha**: {XX.XX}%
- **风格暴露**: {XX.XX}%
- **其他因子**: {XX.XX}%

### 6.2 机制解释
{经济学解释：为什么这个策略能赚钱}

### 6.3 稳健性检验
- **子样本稳定性**: {PASS/FAIL}
- **Regime 稳定性**: {PASS/FAIL}
- **跨标的稳健性**: {PASS/FAIL}

## 7. 风险评估

### 7.1 主要风险
1. **拥挤度风险**: {描述}
2. **Regime 风险**: {描述}
3. **流动性风险**: {描述}

### 7.2 限制条件
- **只能使用 Whitelist 标的**
- **必须使用 Best Horizon**
- **必须使用 Train 冻结的参数**

## 8. 冻结内容

### 8.1 Frozen Spec
```yaml
whitelist: [...]
best_horizon: {N}
param_id: {param_id}
quality_thresholds: {...}
regime_definition: {...}
```

### 8.2 下一步不能改的内容
- [ ] 不能重新估计参数
- [ ] 不能修改 Whitelist
- [ ] 不能改变 Best Horizon
- [ ] 不能调整质量阈值

## 9. 下一步计划

### 9.1 进入 Backtest Ready
**前提条件**:
- [ ] 所有 Formal Gate 通过
- [ ] Frozen Spec 已生成
- [ ] 交接验证完成

### 9.2 风险提示
{对 Backtest 阶段的风险提示}

### 9.3 需要特别关注
{需要 Backtest 阶段特别关注的事项}

## 10. 附录

### 10.1 详细数据
{附详细数据表}

### 10.2 图表
{附可视化图表}

### 10.3 负结果记录
{附被拒绝的参数组合、标的等}
```

### 13.2 Gate Decision 模板

```markdown
---
doc_id: GD-{lineage_id}-{run_id}
title: Gate Decision — Test Evidence
date: YYYY-MM-DD
stage: 04_test_evidence
lineage_id: {lineage_id}
run_id: {run_id}
---

## 决策状态

**状态**: {PASS / CONDITIONAL_PASS / RETRY / RESEARCH_AGAIN / NO_GO}

## 决策依据

### Formal Gate 检查结果

| 检查项 | 要求 | 结果 | 状态 |
|--------|------|------|------|
| 信号方向 | ≥ 55% | {XX%} | {PASS/FAIL} |
| 分层单调性 | 保持单调 | {是/否} | {PASS/FAIL} |
| Sharpe Ratio | ≥ 1.0 | {X.XX} | {PASS/FAIL} |
| 最大回撤 | ≤ -20% | {XX%} | {PASS/FAIL} |
| Whitelist 冻结 | 必需 | {是/否} | {PASS/FAIL} |
| Best Horizon | 必需且显著 | {是/否} | {PASS/FAIL} |
| 拥挤度分析 | 必需 | {是/否} | {PASS/FAIL} |
| Alpha 来源 | 必需明确 | {是/否} | {PASS/FAIL} |
| 无重估参数 | 必需 | {是/否} | {PASS/FAIL} |

### Audit Gate 检查结果
{补充性检查结果，不直接阻断}

## 冻结范围

**已冻结内容**:
- Whitelist: {N} 个标的
- Best Horizon: {N} 天
- 参数 ID: {param_id}
- 质量阈值: {...}
- Regime 定义: {...}

**验证依据**:
- 文件: test_evidence_spec.yaml
- 路径: {path}
- 校验和: {checksum}

## 下一步不能改的内容

**禁止事项**:
- [ ] 不能重新估计参数
- [ ] 不能修改 Whitelist
- [ ] 不能改变 Best Horizon
- [ ] 不能调整质量阈值
- [ ] 不能修改 Regime 定义

## 决策理由

{详细说明为什么做出这个决策}

## 后续行动

### 如果 PASS / CONDITIONAL_PASS:
- 生成 Frozen Spec
- 与 Backtest Ready 交接
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

### A. Test Evidence 阶段启动检查

- [ ] Train Calibration 阶段已完成并 PASS
- [ ] Train Frozen Spec 已接收并验证
- [ ] OOS 数据已准备并验证
- [ ] 时间切分已确认
- [ ] 参数清单已确认

### B. Test Evidence 阶段执行检查

- [ ] 使用 Train 冻结的参数
- [ ] 在 OOS 数据上验证
- [ ] 完成所有验证内容
- [ ] 完成 Whitelist 确定
- [ ] 完成 Best Horizon 确定
- [ ] 完成拥挤度分析
- [ ] 完成 Alpha 来源分析

### C. Test Evidence 阶段完成检查

- [ ] 所有 Formal Gate 通过
- [ ] 验证报告已完成
- [ ] Frozen Spec 已生成
- [ ] 产物目录已完成
- [ ] 字段字典已完成
- [ ] Gate Decision 已记录
- [ ] 与 Backtest Ready 交接完成

---

**文档版本**: v1.0
**最后更新**: 2026-03-26
**下次评审**: 2026-06-26
