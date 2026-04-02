# Test 阶段 -- 横截面因子研究的样本外验证

## 1. 阶段定义与核心目的

### 1.1 阶段定义

**Test（样本外测试）** 是横截面因子研究中，将 Train 阶段冻结的因子结构与分组规则，在完全独立的 OOS 数据上进行统计验证的阶段。

本阶段回答的核心问题不是"因子能赚多少钱"，而是：

> **因子预测能力在样本外是否真实存在？分组收益结构是否稳健？**

### 1.2 在横截面因子研究中的定位

```
Train（因子构建 + 参数冻结）
  ↓ 冻结：因子公式、分位切点、分组方法
Test（样本外统计验证） ← 你在这里
  ↓ 冻结：OOS 验证结论、Regime 表现
Backtest（回测仿真）
  ↓ 冻结：交易规则、成本模型
Holdout（留存验证）
```

### 1.3 核心目的

| 验证维度 | 具体内容 |
|----------|----------|
| IC 稳定性 | OOS 上 IC 是否显著、是否衰减 |
| 分层收益 | 按因子分组的截面收益是否保持单调性 |
| 组合表现 | 多空组合的风险调整收益是否可接受 |
| Regime 鲁棒性 | 因子在不同市场状态下的表现是否一致 |

---

## 2. OOS IC 分析

### 2.1 IC（Information Coefficient）基础

IC 是因子值与未来收益率的截面相关系数，是衡量因子预测能力的核心指标。

```python
import numpy as np
import pandas as pd
from scipy import stats

def compute_cross_sectional_ic(factor_values, forward_returns):
    """
    计算横截面 IC（Spearman 秩相关）

    参数:
        factor_values: DataFrame, index=date, columns=asset, 因子值
        forward_returns: DataFrame, index=date, columns=asset, 未来收益率

    返回:
        ic_series: Series, 每期的 IC 值
    """
    ic_list = []
    common_dates = factor_values.index.intersection(forward_returns.index)

    for date in common_dates:
        f = factor_values.loc[date].dropna()
        r = forward_returns.loc[date].dropna()
        common_assets = f.index.intersection(r.index)

        if len(common_assets) < 10:  # 截面标的数不足，跳过
            continue

        ic, _ = stats.spearmanr(f[common_assets], r[common_assets])
        ic_list.append({'date': date, 'IC': ic})

    return pd.DataFrame(ic_list).set_index('date')['IC']
```

### 2.2 IC 稳定性检验

#### 滚动 IC 均值

```python
def rolling_ic_analysis(ic_series, window=12):
    """
    滚动 IC 均值分析，检测因子是否随时间退化

    参数:
        ic_series: 每期 IC 序列
        window: 滚动窗口（期数）

    返回:
        滚动 IC 均值与 IR（Information Ratio）
    """
    rolling_mean = ic_series.rolling(window).mean()
    rolling_std = ic_series.rolling(window).std()
    rolling_ir = rolling_mean / rolling_std

    return pd.DataFrame({
        'IC_mean': rolling_mean,
        'IC_std': rolling_std,
        'IR': rolling_ir
    })
```

#### IC 衰减分析

IC 衰减检验因子预测能力随时间推移的衰退速度，帮助判断因子的半衰期。

```python
def ic_decay_analysis(factor_values, returns, max_lag=20):
    """
    IC 衰减分析：因子对不同前瞻期收益的预测能力

    参数:
        factor_values: 因子值矩阵
        returns: 收益率矩阵
        max_lag: 最大前瞻期数

    返回:
        decay_df: 各前瞻期的 IC 与 IR
    """
    results = []
    for lag in range(1, max_lag + 1):
        fwd_ret = returns.shift(-lag)
        ic = compute_cross_sectional_ic(factor_values, fwd_ret)
        if len(ic) > 0:
            results.append({
                'lag': lag,
                'IC_mean': ic.mean(),
                'IC_std': ic.std(),
                'IR': ic.mean() / ic.std() if ic.std() > 0 else 0,
                'IC_positive_ratio': (ic > 0).mean()
            })

    return pd.DataFrame(results)
```

**Formal Gate 要求**：

| 指标 | 最低标准 | 说明 |
|------|----------|------|
| OOS IC 均值 | > 0.03 | 月度 IC，显著为正 |
| OOS IC t-stat | > 2.0 | 统计显著性 |
| 正 IC 比例 | > 55% | IC 为正的期数占比 |
| IC 衰减率 | 20 期后 IC > 0 | 因子不应在第 2-3 周就完全失效 |

### 2.3 正负 IC 比例

```python
def ic_ratio_analysis(ic_series):
    """
    分析正负 IC 的统计特征

    返回:
        report: 正负 IC 比例报告
    """
    positive_ic = (ic_series > 0).sum()
    negative_ic = (ic_series < 0).sum()
    total = len(ic_series)

    return {
        'positive_count': int(positive_ic),
        'negative_count': int(negative_ic),
        'positive_ratio': positive_ic / total,
        'negative_ratio': negative_ic / total,
        'ic_skewness': ic_series.skew(),
        'ic_kurtosis': ic_series.kurtosis(),
        # 连续正/负 IC 的最长持续时间
        'max_consecutive_positive': _max_consecutive(ic_series > 0),
        'max_consecutive_negative': _max_consecutive(ic_series < 0),
    }
```

---

## 3. OOS 分组收益分析

### 3.1 分组方法

横截面因子研究的核心方法论：按因子值将截面标的分为多组，检验各组收益是否存在单调递增/递减关系。

```python
def form_factor_portfolios(factor_values, forward_returns, n_groups=5):
    """
    按因子值构建分组组合

    参数:
        factor_values: DataFrame, 因子值
        forward_returns: DataFrame, 未来收益率
        n_groups: 分组数（5=五档, 10=十分位）

    返回:
        portfolio_returns: DataFrame, 各组收益序列
        group_labels: 各期各标的的分组标签
    """
    portfolio_returns = {}
    group_labels = pd.DataFrame(index=factor_values.index,
                                columns=factor_values.columns)

    for date in factor_values.index:
        if date not in forward_returns.index:
            continue

        f = factor_values.loc[date].dropna()
        r = forward_returns.loc[date].dropna()
        common = f.index.intersection(r.index)

        if len(common) < n_groups * 2:
            continue

        # 使用 Train 冻结的分位切点（不能在 OOS 重新计算）
        try:
            labels = pd.qcut(f[common], q=n_groups, labels=False,
                             duplicates='drop')
        except ValueError:
            # 当截面分布有大量重复值时退化为等分
            labels = pd.cut(f[common], bins=n_groups, labels=False)

        group_labels.loc[date, common] = labels

        for g in range(n_groups):
            mask = labels == g
            if mask.sum() > 0:
                ret = r[common][mask].mean()
                portfolio_returns.setdefault(g, []).append(ret)
            else:
                portfolio_returns.setdefault(g, []).append(np.nan)

    # 转换为 DataFrame
    dates = factor_values.index.intersection(forward_returns.index)
    result = pd.DataFrame(portfolio_returns, index=dates)
    result.columns = [f'Q{i+1}' for i in range(result.shape[1])]

    return result, group_labels
```

### 3.2 五档 / 十分位分组

| 分组方式 | 适用场景 | 优势 | 劣势 |
|----------|----------|------|------|
| 五档分组 | 标的数 < 100 | 每组标的充足，收益更稳定 | 分辨率较低 |
| 十分位分组 | 标的数 >= 100 | 边际收益更精细 | 端组可能标的稀少 |
| 三档分组 | 标的数 < 30 | 极端情况下的最低要求 | 信息损失大 |

### 3.3 多空收益与单调性检验

```python
def evaluate_portfolio_performance(portfolio_returns):
    """
    评估分组组合表现

    返回:
        metrics: 分组表现指标
    """
    n_groups = portfolio_returns.shape[1]
    long_group = portfolio_returns.iloc[:, -1]   # 最高组（多头）
    short_group = portfolio_returns.iloc[:, 0]    # 最低组（空头）

    # 多空收益
    long_short = long_group - short_group

    # 单调性检验：相邻组的收益差是否显著为正
    monotonicity_tests = []
    for i in range(n_groups - 1):
        diff = portfolio_returns.iloc[:, i+1] - portfolio_returns.iloc[:, i]
        t_stat, p_val = stats.ttest_1samp(diff.dropna(), 0)
        monotonicity_tests.append({
            'from_group': f'Q{i+1}',
            'to_group': f'Q{i+2}',
            'spread': diff.mean(),
            't_stat': t_stat,
            'p_value': p_val,
            'significant': p_val < 0.05
        })

    # 整体单调性：Spearman 秩相关
    group_means = portfolio_returns.mean()
    group_ranks = range(1, n_groups + 1)
    mono_corr, mono_p = stats.spearmanr(group_ranks, group_means)

    return {
        'long_short_return': long_short.mean(),
        'long_short_sharpe': (long_short.mean() / long_short.std()
                              * np.sqrt(252)),
        'long_short_t_stat': stats.ttest_1samp(long_short.dropna(), 0)[0],
        'monotonicity_tests': monotonicity_tests,
        'monotonicity_corr': mono_corr,
        'monotonicity_p_value': mono_p,
        'group_annualized_returns': portfolio_returns.mean() * 252,
    }
```

**Formal Gate 要求**：

| 指标 | 最低标准 | 说明 |
|------|----------|------|
| 多空年化收益 | > 0 | Q5 - Q1 收益为正 |
| 多空 t-stat | > 2.0 | 多空收益统计显著 |
| 单调性相关系数 | > 0.8 | 组收益与组序号的 Spearman 相关 |
| 分层倒置 | 不允许 | 相邻组收益差不出现系统性反转 |

---

## 4. OOS 组合表现

### 4.1 核心风险指标

```python
def portfolio_risk_metrics(portfolio_returns, long_short=None):
    """
    计算多空组合的风险指标

    参数:
        portfolio_returns: 分组收益矩阵
        long_short: 多空收益序列（若为 None 则自动计算）
    """
    if long_short is None:
        long_short = portfolio_returns.iloc[:, -1] - portfolio_returns.iloc[:, 0]

    annual_factor = 252
    ann_ret = long_short.mean() * annual_factor
    ann_vol = long_short.std() * np.sqrt(annual_factor)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    # 最大回撤
    cumulative = (1 + long_short).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()

    # Calmar 比率
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    # 换手率（基于分组稳定性估算）
    # 如果标的频繁在组间切换，说明因子排序不稳定
    turnover = estimate_turnover(portfolio_returns)

    return {
        'annualized_return': ann_ret,
        'annualized_volatility': ann_vol,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'calmar_ratio': calmar,
        'win_rate': (long_short > 0).mean(),
        'turnover_estimate': turnover,
    }

def estimate_turnover(group_labels):
    """
    估算分组换手率：相邻期标的在不同组之间切换的比例
    """
    changes = group_labels.diff().dropna()
    changed = (changes != 0).sum().sum()
    total = changes.count().sum()
    return changed / total if total > 0 else 0
```

### 4.2 指标解读标准

| 指标 | 优秀 | 合格 | 警告 |
|------|------|------|------|
| Sharpe Ratio | > 1.5 | 0.8 - 1.5 | < 0.8 |
| Max Drawdown | > -10% | -10% ~ -20% | < -20% |
| Calmar Ratio | > 1.0 | 0.5 - 1.0 | < 0.5 |
| 年化换手率 | < 300% | 300% - 600% | > 600% |
| 月度胜率 | > 60% | 52% - 60% | < 52% |

**Formal Gate 要求**：

```yaml
formal_gate:
  sharpe_ratio: ">= 0.8"
  max_drawdown: ">= -25%"
  calmar_ratio: ">= 0.4"
  win_rate: ">= 52%"
```

---

## 5. Regime Robustness（市场状态鲁棒性）

### 5.1 为什么必须做 Regime 检验

一个因子可能在整体 OOS 上表现良好，但实际上只在特定市场环境下有效（例如只在牛市赚钱）。如果不做 Regime 分析，因子在环境切换时可能突然失效。

### 5.2 牛熊市分别检验

```python
def regime_performance(factor_values, forward_returns, market_index,
                       bull_threshold=0.02, bear_threshold=-0.02):
    """
    按牛熊市分别检验因子表现

    参数:
        market_index: Series, 市场指数收益率（用于划分 Regime）
        bull_threshold: 牛市判定阈值（月度收益 > 此值）
        bear_threshold: 熊市判定阈值
    """
    # 按 Regime 划分日期
    market_monthly = market_index.resample('M').apply(lambda x: (1+x).prod()-1)

    regimes = pd.Series('sideways', index=market_monthly.index)
    regimes[market_monthly > bull_threshold] = 'bull'
    regimes[market_monthly < bear_threshold] = 'bear'

    # 将日度日期映射到 Regime
    daily_regime = regimes.reindex(market_index.index, method='ffill')

    results = {}
    for regime_name in ['bull', 'bear', 'sideways']:
        mask = daily_regime == regime_name
        if mask.sum() == 0:
            continue

        fv = factor_values[mask]
        fr = forward_returns[mask]
        ic = compute_cross_sectional_ic(fv, fr)

        if len(ic) > 0:
            results[regime_name] = {
                'periods': len(ic),
                'IC_mean': ic.mean(),
                'IC_t_stat': ic.mean() / ic.std() * np.sqrt(len(ic))
                               if ic.std() > 0 else 0,
                'IC_positive_ratio': (ic > 0).mean(),
                'sharpe_estimate': ic.mean() / ic.std() if ic.std() > 0 else 0,
            }

    return results
```

### 5.3 波动率 Regime

```python
def volatility_regime_analysis(factor_values, forward_returns,
                                market_index, vol_window=60,
                                high_vol_quantile=0.75,
                                low_vol_quantile=0.25):
    """
    按波动率 Regime 检验因子表现
    """
    realized_vol = market_index.rolling(vol_window).std() * np.sqrt(252)

    high_vol = realized_vol > realized_vol.quantile(high_vol_quantile)
    low_vol = realized_vol < realized_vol.quantile(low_vol_quantile)

    results = {}
    for regime_name, mask in [('high_vol', high_vol), ('low_vol', low_vol),
                               ('normal_vol', ~(high_vol | low_vol))]:
        fv = factor_values[mask]
        fr = forward_returns[mask]
        ic = compute_cross_sectional_ic(fv, fr)

        if len(ic) > 5:
            results[regime_name] = {
                'periods': len(ic),
                'IC_mean': ic.mean(),
                'IR': ic.mean() / ic.std() if ic.std() > 0 else 0,
                'IC_positive_ratio': (ic > 0).mean(),
            }

    return results
```

### 5.4 流动性 Regime

```python
def liquidity_regime_analysis(factor_values, forward_returns,
                               volume_data, liq_window=20):
    """
    按市场流动性 Regime 检验因子表现
    """
    # 用市场总成交量作为流动性代理
    total_volume = volume_data.sum(axis=1)
    avg_volume = total_volume.rolling(liq_window).mean()
    volume_ratio = total_volume / avg_volume

    high_liq = volume_ratio > 1.3
    low_liq = volume_ratio < 0.7

    results = {}
    for regime_name, mask in [('high_liquidity', high_liq),
                               ('low_liquidity', low_liq),
                               ('normal_liquidity', ~(high_liq | low_liq))]:
        fv = factor_values[mask]
        fr = forward_returns[mask]
        ic = compute_cross_sectional_ic(fv, fr)

        if len(ic) > 5:
            results[regime_name] = {
                'periods': len(ic),
                'IC_mean': ic.mean(),
                'IR': ic.mean() / ic.std() if ic.std() > 0 else 0,
            }

    return results
```

### 5.5 Regime Robustness 判定标准

**Formal Gate 要求**：

```yaml
formal_gate:
  # 因子必须在至少 2 种 Regime 下 IC 显著为正
  significant_regimes: ">= 2"

  # 不允许在任何 Regime 下 IC 显著为负
  no_negative_regime: true

  # Regime 间 IC 方差不能过大（因子表现不能极端不稳定）
  ic_cross_regime_cv: "< 1.0"  # 变异系数

  # 最差 Regime 的 IC 不能低于 -0.01
  worst_regime_ic: ">= -0.01"
```

| Regime 类型 | 判定场景 | 处理方式 |
|-------------|----------|----------|
| 因子只在牛市有效 | IC 在 bear regime 显著为负 | 记录风险，Backtest 中加入 Regime 条件判断 |
| 因子在所有 Regime 都有效 | IC 在所有 regime 显著为正 | 最理想情况，因子鲁棒性最强 |
| 因子只在低波动有效 | 高波动 IC 为零或微负 | 评估因子是否可与其他因子互补 |
| 因子在流动性差时失效 | 低流动性 IC 显著下降 | Backtest 阶段需加入流动性过滤条件 |

---

## 6. Formal Gate 总汇

### 6.1 门禁检查清单

```yaml
formal_gate_test_stage:

  FG-IC_OOS:
    requirement: "OOS IC 均值 > 0.03 且 t-stat > 2.0"
    evidence: "ic_analysis.csv"
    status: "PASS / FAIL"

  FG-IC_RATIO:
    requirement: "正 IC 比例 > 55%"
    evidence: "ic_ratio_report.json"
    status: "PASS / FAIL"

  FG-IC_DECAY:
    requirement: "20 期后 IC 仍 > 0"
    evidence: "ic_decay_analysis.csv"
    status: "PASS / FAIL"

  FG-LONG_SHORT:
    requirement: "多空收益 t-stat > 2.0"
    evidence: "portfolio_analysis.json"
    status: "PASS / FAIL"

  FG-MONOTONICITY:
    requirement: "分组收益单调性相关系数 > 0.8"
    evidence: "monotonicity_test.json"
    status: "PASS / FAIL"

  FG-SHARPE:
    requirement: "多空组合 Sharpe >= 0.8"
    evidence: "risk_metrics.json"
    status: "PASS / FAIL"

  FG-DRAWDOWN:
    requirement: "最大回撤 >= -25%"
    evidence: "risk_metrics.json"
    status: "PASS / FAIL"

  FG-REGIME:
    requirement: ">= 2 种 Regime 下 IC 显著为正"
    evidence: "regime_analysis.json"
    status: "PASS / FAIL"

  FG-NO_NEGATIVE_REGIME:
    requirement: "无 Regime 下 IC 显著为负"
    evidence: "regime_analysis.json"
    status: "PASS / FAIL"
```

### 6.2 决策状态

| 状态 | 条件 | 后续动作 |
|------|------|----------|
| **PASS** | 所有 FG 通过 | 进入 Backtest 阶段 |
| **CONDITIONAL PASS** | 核心 FG 通过，Regime 有瑕疵 | 记录风险条件，进入 Backtest |
| **RETRY** | 代码 bug 或数据处理错误 | 修复后重新运行 Test，不修改 Train 参数 |
| **RESEARCH_AGAIN** | IC 不显著或 Regime 鲁棒性不足 | 回退到 Train 阶段或重新构建因子 |
| **NO_GO** | 因子完全无效 | 终止该因子研究线，归档负结果 |

---

## 7. 常见错误与防范

### 7.1 错误清单

| 错误类型 | 描述 | 后果 | 防范措施 |
|----------|------|------|----------|
| **OOS 上重新估计分位切点** | 在 OOS 数据上重新计算 qcut 的边界 | 样本泄漏，IC 虚高 | 分位切点在 Train 冻结，OOS 直接 apply |
| **忽略因子值的分布漂移** | 因子分布在 OOS 发生系统性偏移 | 分组不再是等分，组间不均衡 | 检查 OOS 因子分布与 IS 的 KS 检验 |
| **IC 用 Pearson 而非 Spearman** | 使用线性相关而非秩相关 | 受极端值影响，IC 不可靠 | 横截面 IC 统一使用 Spearman |
| **忽略截面样本量** | 某些期截面标的过少仍计算 IC | IC 方差极大，无统计意义 | 设定最低截面标的数（如 >= 10） |
| **不检验 Regime** | 只看整体 OOS IC | 因子可能只在特定市场环境下有效 | 强制做牛熊/波动率/流动性 Regime 分析 |
| **多重比较未校正** | 同时检验多个前瞻期 IC | 偶然显著的假阳性 | 使用 Bonferroni 或 BH 校正 |
| **幸存者偏差** | 只报告通过的 Regime | 高估因子鲁棒性 | 报告所有 Regime 结果，包括失败的 |

### 7.2 实际案例

**案例：流动性因子在牛市中 IC 反转**

某研究团队构建了一个 Amihud 非流动性因子，整体 OOS IC = 0.04（通过门禁），但 Regime 分析显示：

```yaml
regime_results:
  bull_market:
    IC_mean: -0.02    # 反转！
    IC_positive_ratio: 0.38
  bear_market:
    IC_mean: 0.08     # 熊市极强
    IC_positive_ratio: 0.72
  sideways:
    IC_mean: 0.03
    IC_positive_ratio: 0.58
```

**处理方式**：
- 不直接淘汰（因子在熊市有强预测力）
- 标记为 CONDITIONAL PASS
- 在 Backtest 阶段加入 Regime 条件：牛市降低仓位或暂停
- 评估与其他因子的互补性（作为防御性因子使用）

---

## 8. 输出 Artifact

### 8.1 机器可读产物

```yaml
ic_analysis.csv:
  用途: 每期 IC 序列
  字段: date, IC
  消费者: 滚动分析、IC 衰减

ic_decay_analysis.csv:
  用途: 各前瞻期 IC
  字段: lag, IC_mean, IC_std, IR, IC_positive_ratio
  消费者: 最佳持仓期决策

portfolio_returns.parquet:
  用途: 各分组收益序列
  粒度: 日频
  消费者: 风险指标计算、可视化

risk_metrics.json:
  用途: 组合风险指标汇总
  内容: sharpe, max_drawdown, calmar, turnover
  消费者: Formal Gate 判定

regime_analysis.json:
  用途: 各 Regime 下的因子表现
  内容: bull/bear/vol/liquidity regime 结果
  消费者: 鲁棒性评估
```

### 8.2 人类可读产物

```yaml
test_report.md:
  用途: Test 阶段完整报告
  消费者: 团队评审、Backtest 参考

gate_decision.md:
  用途: 门禁决策文档
  必需字段: stage, status, decision_basis, frozen_scope, next_steps
  消费者: 流程管理

negative_results/:
  用途: 保存失败的 Regime、不达标的分组结果
  重要性: 避免幸存者偏差
```

---

## 9. 与 Backtest 阶段的交接

### 9.1 冻结传递

```yaml
frozen_spec_handover:
  from_stage: "test"
  to_stage: "backtest"

  frozen_items:
    factor_formula: "Train 冻结的因子计算公式"
    quantile_cuts: "Train 冻结的分位切点"
    best_horizon: "IC 衰减分析确定的最佳持仓期"
    regime_risks: "Test 阶段识别的 Regime 风险点"
    quality_filters: "Train 冻结的数据质量过滤条件"

  backtest_must_use:
    - "只能使用 Test 冻结的分组方法"
    - "不能重新估计分位切点"
    - "必须考虑 Regime 风险条件"
```

### 9.2 Backtest 阶段需要特别关注的事项

1. **换手率**：Test 阶段估算的分组换手率直接影响 Backtest 的交易成本
2. **Regime 条件**：如果因子在某些 Regime 下失效，Backtest 需要设计 Regime 切换逻辑
3. **容量预估**：Test 的分组结果决定了多空组合的标的构成，Backtest 需基于此评估容量

---

**文档版本**: v1.0
**最后更新**: 2026-04-02
**适用领域**: 横截面因子研究
