# Holdout 阶段 -- 留存验证

## 1. 阶段定义

**Holdout（留存验证）** 是在完全未参与任何设计、调参或优化环节的最终时间窗口上，对策略进行最后一次独立验证。

本阶段的定位：

> 这不是又一次回测，而是策略投入实盘前的盲测——只有一次机会，结果即为终审。

### 1.1 在流程中的定位

```
Train（因子构建 + 参数冻结）
  ↓
Test（样本外统计验证）
  ↓ 冻结：因子结构、分组规则、Regime 结论
Backtest（回测仿真）
  ↓ 冻结：交易规则、成本模型、执行方案
Holdout（留存验证） ← 你在这里
  ↓ 冻结：全部内容，不可修改
Shadow Admission / 实盘
```

### 1.2 目的

| 目的 | 具体内容 |
|------|----------|
| **最终独立验证** | 在完全未见过的数据窗口上确认策略有效性 |
| 过拟合终审 | 检测因多轮优化导致的隐性过拟合 |
| 泛化能力检验 | 验证策略在新市场环境下的适应能力 |
| 晋级或淘汰决策 | 作为策略能否进入实盘的唯一终审依据 |

### 1.3 关键约束

```yaml
holdout_iron_rules:
  完全隔离: "Holdout 窗口从未参与任何阶段的设计或调参"
  单向验证: "只能验证，不能根据结果回头调整任何参数"
  一次机会: "只有一次验证机会，不能多次尝试后选择最优"
  冻结即冻结: "Backtest 冻结的所有内容在 Holdout 中完全不可动"
  结果即结果: "接受结果，不通过就走 Rollback 或 Child Lineage"
```

---

## 2. 最终独立评估

### 2.1 Holdout 窗口要求

Holdout 窗口必须在 Mandate 阶段就划定，与研究全流程的时间切分一致：

```
Train:     [T_start,            T_train_end]
Test:      (T_train_end,        T_test_end]
Backtest:  (T_test_end,         T_backtest_end]
Holdout:   (T_backtest_end,     T_holdout_end]  ← 完全独立
```

窗口长度建议：

| 市场类型 | 建议 Holdout 长度 | 最低要求 |
|----------|------------------|----------|
| 加密货币（高波动） | 6-12 个月 | 3 个月 |
| 股票市场 | 12-24 个月 | 6 个月 |
| 期货市场 | 12-18 个月 | 6 个月 |

### 2.2 所有指标终审

本阶段需要对前序所有阶段的指标进行终审，确认没有显著退化。

```python
def holdout_full_audit(backtest_metrics, holdout_metrics, degradation_thresholds):
    """
    Holdout 终审：对比 Backtest 和 Holdout 的全部指标

    参数:
        backtest_metrics: dict, Backtest 阶段指标
        holdout_metrics: dict, Holdout 阶段指标
        degradation_thresholds: dict, 各指标允许的最大退化幅度

    返回:
        audit_report: 终审报告
    """
    audit_results = {}

    # 核心指标对比
    comparison_items = {
        'IC_mean': {'label': 'IC 均值', 'threshold': 0.50},
        'IC_IR': {'label': 'IC IR', 'threshold': 0.40},
        'long_short_return': {'label': '多空收益', 'threshold': 0.50},
        'sharpe_ratio': {'label': 'Sharpe', 'threshold': 0.40},
        'calmar_ratio': {'label': 'Calmar', 'threshold': 0.50},
        'win_rate': {'label': '月度胜率', 'threshold': 0.10},
        'max_drawdown': {'label': '最大回撤', 'threshold': 0.50},
        'turnover': {'label': '换手率', 'threshold': 0.30},
    }

    for key, config in comparison_items.items():
        bt_val = backtest_metrics.get(key, 0)
        ho_val = holdout_metrics.get(key, 0)
        threshold = config['threshold']

        # 计算退化幅度（注意回撤和换手率是越小越好）
        if key in ['max_drawdown', 'turnover']:
            degradation = abs(ho_val / bt_val - 1) if bt_val != 0 else float('inf')
            degraded = degradation > threshold
        elif bt_val != 0:
            degradation = max(0, 1 - ho_val / bt_val)
            degraded = degradation > threshold
        else:
            degradation = float('inf') if ho_val == 0 else 0
            degraded = ho_val <= 0

        audit_results[key] = {
            'label': config['label'],
            'backtest_value': bt_val,
            'holdout_value': ho_val,
            'degradation': degradation,
            'threshold': threshold,
            'degraded': degraded,
            'verdict': 'FAIL' if degraded else 'PASS',
        }

    return audit_results
```

### 2.3 终审报告模板

```markdown
## Holdout 终审报告

### 核心指标对比

| 指标 | Backtest | Holdout | 退化幅度 | 阈值 | 判定 |
|------|----------|---------|----------|------|------|
| IC 均值 | 0.045 | 0.038 | 15.6% | 50% | PASS |
| IC IR | 0.82 | 0.65 | 20.7% | 40% | PASS |
| 多空收益（年化） | 22.5% | 17.8% | 20.9% | 50% | PASS |
| Sharpe | 1.45 | 1.12 | 22.8% | 40% | PASS |
| Calmar | 1.85 | 1.20 | 35.1% | 50% | PASS |
| 月度胜率 | 65% | 58% | 10.8% | 10% | FAIL |
| 最大回撤 | -12% | -18% | 50.0% | 50% | PASS |
| 换手率 | 380% | 420% | 10.5% | 30% | PASS |

### 总结
- PASS: 7 项
- FAIL: 1 项（月度胜率退化超阈值）
- 总判定: CONDITIONAL PASS
```

---

## 3. 过拟合检测

### 3.1 过拟合信号分级

过拟合是本阶段最需要警惕的问题。以下是按指标退化程度的过拟合信号分级：

强过拟合信号（直接 NO_GO）：

```yaml
strong_overfitting_signals:
  - condition: "Holdout 多空收益为负"
    severity: "致命"
    implication: "因子在独立样本上完全失效"

  - condition: "Holdout IC 均值为负"
    severity: "致命"
    implication: "因子预测方向反转，属于过拟合产物"

  - condition: "收益退化 > 70%"
    severity: "致命"
    implication: "因子严重过拟合历史数据"
```

中等过拟合信号（需评估风险）：

```yaml
moderate_overfitting_signals:
  - condition: "收益退化 40%-70%"
    severity: "警告"
    implication: "因子可能存在中度过拟合"

  - condition: "Sharpe 退化 > 50%"
    severity: "警告"
    implication: "风险调整后收益不可接受"

  - condition: "回撤扩大 > 2 倍"
    severity: "警告"
    implication: "风险管理在极端情况下失效"

  - condition: "Regime 表现与 Backtest 严重不一致"
    severity: "警告"
    implication: "因子依赖特定市场环境"
```

弱过拟合信号（可接受）：

```yaml
weak_overfitting_signals:
  - condition: "收益退化 10%-30%"
    severity: "正常"
    implication: "正常的样本外衰减"

  - condition: "个别月份表现异常"
    severity: "正常"
    implication: "市场噪声导致的随机波动"

  - condition: "IC 轻微下降"
    severity: "正常"
    implication: "因子在更长周期上自然衰减"
```

### 3.2 过拟合检测方法

```python
def overfitting_diagnosis(backtest_results, holdout_results):
    """
    过拟合诊断：多维度检测过拟合信号

    参数:
        backtest_results: Backtest 阶段全部结果
        holdout_results: Holdout 阶段全部结果

    返回:
        diagnosis: 过拟合诊断报告
    """
    diagnosis = {}

    # 维度 1：收益水平对比
    bt_return = backtest_results['annual_return']
    ho_return = holdout_results['annual_return']
    return_decay = 1 - ho_return / bt_return if bt_return > 0 else 0
    diagnosis['return_decay'] = {
        'value': return_decay,
        'severity': _classify_severity(return_decay,
            thresholds=[0.3, 0.5, 0.7])
    }

    # 维度 2：收益分布对比（Kolmogorov-Smirnov 检验）
    bt_daily = backtest_results['daily_returns']
    ho_daily = holdout_results['daily_returns']
    ks_stat, ks_p = stats.ks_2samp(bt_daily, ho_daily)
    diagnosis['distribution_change'] = {
        'ks_statistic': ks_stat,
        'ks_p_value': ks_p,
        'distribution_shifted': ks_p < 0.05,
        'severity': '中等' if ks_p < 0.05 else '正常',
    }

    # 维度 3：回撤模式对比
    bt_dd = backtest_results['drawdown_series']
    ho_dd = holdout_results['drawdown_series']
    diagnosis['drawdown_comparison'] = {
        'bt_max_dd': bt_dd.min(),
        'ho_max_dd': ho_dd.min(),
        'dd_ratio': abs(ho_dd.min() / bt_dd.min())
                        if bt_dd.min() != 0 else float('inf'),
        'severity': _classify_severity(
            abs(ho_dd.min() / bt_dd.min()) - 1,
            thresholds=[0.5, 1.0, 2.0])
    }

    # 维度 4：月度收益相关性
    bt_monthly = bt_daily.resample('M').apply(lambda x: (1+x).prod()-1)
    ho_monthly = ho_daily.resample('M').apply(lambda x: (1+x).prod()-1)
    month_corr, _ = stats.spearmanr(
        bt_monthly.rank(), ho_monthly.rank()
    )
    diagnosis['monthly_pattern'] = {
        'rank_correlation': month_corr,
        'pattern_consistent': month_corr > 0.3,
        'severity': '正常' if month_corr > 0.3 else '中等',
    }

    # 综合判定
    severities = [v['severity'] for v in diagnosis.values()]
    if '致命' in severities:
        overall = '强过拟合'
    elif '中等' in severities:
        overall = '中等过拟合风险'
    else:
        overall = '过拟合风险可控'

    diagnosis['overall_verdict'] = overall

    return diagnosis


def _classify_severity(value, thresholds):
    """根据阈值分级"""
    if value < thresholds[0]:
        return '正常'
    elif value < thresholds[1]:
        return '弱'
    elif value < thresholds[2]:
        return '中等'
    else:
        return '致命'
```

### 3.3 Holdout 期间市场环境分析

```python
def holdout_environment_analysis(holdout_window, market_data):
    """
    分析 Holdout 期间的市场环境特征

    用于区分"策略过拟合"和"市场环境变化导致的表现差异"
    """
    analysis = {}

    # 整体市场表现
    market_return = (1 + market_data['close'].pct_change()).prod() - 1
    analysis['market_return'] = market_return

    # 波动率环境
    realized_vol = market_data['close'].pct_change().std() * np.sqrt(252)
    analysis['realized_volatility'] = realized_vol

    # 牛熊市划分
    monthly_returns = market_data['close'].resample('M').last().pct_change()
    bull_months = (monthly_returns > 0.02).sum()
    bear_months = (monthly_returns < -0.02).sum()
    sideways_months = len(monthly_returns) - bull_months - bear_months
    analysis['regime_distribution'] = {
        'bull': int(bull_months),
        'bear': int(bear_months),
        'sideways': int(sideways_months),
    }

    # 特殊事件标记
    extreme_days = market_data[market_data['close'].pct_change().abs() > 0.10]
    analysis['extreme_events'] = len(extreme_days)

    return analysis
```

关键原则：如果 Holdout 期间市场环境与 Train/Test 期间存在显著差异（如从牛市转为熊市），需要在决策中区分"过拟合"和"环境不匹配"。

---

## 4. 晋级或淘汰决策

### 4.1 决策矩阵

最终决策基于以下矩阵：

```yaml
decision_matrix:

  PASS:
    conditions:
      - "Holdout 多空收益 > 0"
      - "Holdout Sharpe >= 0.6"
      - "收益退化 < 40%"
      - "无强过拟合信号"
      - "过拟合诊断为'过拟合风险可控'"
    action: "进入 Shadow Admission / 实盘准备"
    authority: "自动晋级，记录决策依据"

  CONDITIONAL_PASS:
    conditions:
      - "Holdout 多空收益 > 0"
      - "Holdout Sharpe >= 0.4"
      - "收益退化 40%-60%"
      - "中等过拟合信号 <= 2 个"
    action: "延期观察 + 有条件晋级"
    required_actions:
      - "明确记录降级条件"
      - "设定实盘监控指标和止损线"
      - "限制初始资金规模"
      - "缩短实盘观察周期"
    authority: "需团队讨论确认"

  EXTEND_OBSERVATION:
    conditions:
      - "Holdout 多空收益 > 0 但接近 0"
      - "收益退化 50%-70%"
      - "个别指标显著退化但非全部"
      - "可能存在市场环境不匹配"
    action: "延期观察 3-6 个月"
    required_actions:
      - "扩展 Holdout 窗口"
      - "分析市场环境变化的影响"
      - "评估因子是否需要结构性调整"
    authority: "需团队讨论 + PM 审批"

  NO_GO:
    conditions:
      - "Holdout 多空收益 <= 0"
      - "Holdout IC 均值为负"
      - "收益退化 > 70%"
      - "出现强过拟合信号"
    action: "终止研究线"
    required_actions:
      - "完整记录失败原因"
      - "归档负结果"
      - "提取经验教训"
    authority: "自动淘汰，归档记录"
```

### 4.2 晋级条件清单

以下是策略从 Holdout 晋级的完整条件清单：

```yaml
promotion_checklist:

  指标门槛:
    - [ ] Holdout IC 均值 > 0.02
    - [ ] Holdout IC t-stat > 1.5
    - [ ] Holdout 正 IC 比例 > 52%
    - [ ] Holdout 多空收益 > 0
    - [ ] Holdout 多空 Sharpe >= 0.6
    - [ ] Holdout 最大回撤 >= -30%
    - [ ] Holdout Calmar >= 0.3

  稳定性要求:
    - [ ] 收益退化 < 40%（相对 Backtest）
    - [ ] Sharpe 退化 < 50%
    - [ ] 月度收益分布未发生显著变化（KS p > 0.05）
    - [ ] 无连续 3 个月以上亏损

  过拟合控制:
    - [ ] 过拟合诊断为"过拟合风险可控"
    - [ ] 无强过拟合信号
    - [ ] 中等过拟合信号不超过 2 个

  合规性:
    - [ ] 未修改 Backtest 冻结的任何参数
    - [ ] 只运行了一次 Holdout
    - [ ] 所有 artifact 完整
    - [ ] 负结果已归档
```

### 4.3 淘汰标准

```yaml
elimination_criteria:

  直接淘汰（自动 NO_GO）:
    - "Holdout 多空收益 <= 0"
    - "Holdout IC 均值 <= 0"
    - "收益退化 > 70%"

  条件淘汰（需评估后决定）:
    - "Holdout Sharpe < 0.4 且收益退化 > 50%"
    - "最大回撤 > -40%"
    - "过拟合诊断为'强过拟合'"
    - "市场环境匹配但策略仍然失效"

  淘汰后处理:
    - "完整填写失败分析报告"
    - "归档至 negative_results/ 目录"
    - "提取可复用的经验教训"
    - "评估是否启动 Child Lineage"
```

### 4.4 延期观察条件

延期观察适用于那些"不确定是否过拟合"的边界情况：

```yaml
extend_observation_protocol:

  触发条件:
    - "Holdout 收益 > 0 但 Sharpe < 0.6"
    - "收益退化 50%-70% 但可能由市场环境变化导致"
    - "部分 Regime 表现退化但非全部"

  观察期要求:
    duration: "3-6 个月"
    monitoring:
      - "月度 IC 和 IR"
      - "月度多空收益和 Sharpe"
      - "最大回撤"
      - "换手率和交易成本"

  退出条件:
    pass: "观察期内 Sharpe >= 0.6 且收益稳定"
    fail: "观察期内出现连续 2 个月亏损或 Sharpe < 0.3"

  注意事项:
    - "延期观察期间不能修改任何冻结参数"
    - "延期观察不算作新的 Holdout 运行"
    - "延期观察的结果需要重新评估决策矩阵"
```

---

## 5. Holdout vs Backtest 区别

| 维度 | Backtest | Holdout |
|------|----------|---------|
| 数据用途 | 可能经过多轮优化 | 完全未参与设计 |
| 修改自由度 | 可调整成本模型和执行方案 | 完全冻结 |
| 尝试次数 | 可多次迭代 | 只有一次机会 |
| 目的 | 验证可交易性和容量 | 验证泛化能力 |
| 结果性质 | 可能存在过拟合 | 真正的盲测结果 |
| 成本模型 | 可微调参数 | 必须 100% 复用 Backtest |
| 决策后果 | 不通过可重试 | 不通过只能归档或回退 |

---

## 6. Formal Gate 要求

### 6.1 门禁检查清单

```yaml
formal_gate_holdout_stage:

  FG-HO_IC:
    requirement: "Holdout IC 均值 > 0.02 且 t-stat > 1.5"
    evidence: "holdout_ic_analysis.csv"
    status: "PASS / FAIL"

  FG-HO_RETURN:
    requirement: "Holdout 多空收益 > 0"
    evidence: "holdout_results.parquet"
    status: "PASS / FAIL"

  FG-HO_SHARPE:
    requirement: "Holdout 多空 Sharpe >= 0.6"
    evidence: "holdout_risk_metrics.json"
    status: "PASS / FAIL"

  FG-DEGRADATION:
    requirement: "收益退化 < 40%（相对 Backtest）"
    evidence: "degradation_analysis.json"
    status: "PASS / FAIL"

  FG-OVERFITTING:
    requirement: "无强过拟合信号"
    evidence: "overfitting_diagnosis.json"
    status: "PASS / FAIL"

  FG-NO_MODIFICATION:
    requirement: "未修改 Backtest 冻结的任何内容"
    evidence: "code_review + parameter_manifest"
    status: "PASS / FAIL"

  FG-SINGLE_RUN:
    requirement: "只运行了一次 Holdout"
    evidence: "execution_log.json"
    status: "PASS / FAIL"

  FG-ARTIFACT_COMPLETE:
    requirement: "所有产物完整"
    evidence: "artifact_catalog.md"
    status: "PASS / FAIL"
```

### 6.2 决策状态总览

| 状态 | 适用条件 | 后续动作 | 审批要求 |
|------|----------|----------|----------|
| **PASS** | 所有 FG 通过 | 进入 Shadow Admission | 自动 |
| **CONDITIONAL PASS** | 核心 FG 通过，有瑕疵 | 限制性晋级 | 团队讨论 |
| **EXTEND** | 边界情况，需更多数据 | 延期观察 3-6 月 | PM 审批 |
| **NO_GO** | 核心指标不达标 | 终止并归档 | 自动 |

---

## 7. 常见错误与防范

### 7.1 错误清单

| 错误类型 | 描述 | 后果 | 防范措施 |
|----------|------|------|----------|
| **根据 Holdout 结果回头调参** | 看到结果不理想就回到 Backtest 调整 | Holdout 窗口参与设计，验证失效 | 铁律：结果即结果 |
| **多次 Holdout 选最优** | 尝试多个参数组合选 Holdout 表现最好的 | 本质是在 Holdout 上优化 | 铁律：只运行一次 |
| **把 Holdout 当额外优化窗口** | 在 Holdout 上测试不同规则变体 | 完全破坏验证意义 | Holdout 只做验证 |
| **忽略市场环境变化** | 不分析 Holdout 期间的 regime 变化 | 错判过拟合 | 必须做环境分析 |
| **只报告好的指标** | 选择性展示通过的指标 | 高估策略质量 | 报告全部指标，含失败的 |
| **Holdout 窗口过短** | Holdout 不足 3 个月 | 统计不显著，结论不可靠 | 窗口至少 3-6 个月 |

### 7.2 实际案例

案例 1：价值因子 Holdout 失败——过拟合典型

某团队构建了一个多因子价值模型，Backtest 表现优异：

```
Backtest 阶段:
  多空年化收益: 28%
  Sharpe: 1.85
  最大回撤: -11%

Holdout 阶段（6 个月）:
  多空年化收益: -5.2%
  Sharpe: -0.32
  IC 均值: -0.008（方向反转！）
```

诊断结论：强过拟合信号。因子在 Train/Test 期间的价值溢价效应在 Holdout 期间完全消失。可能原因：市场风格切换，价值因子在成长风格主导的市场中失效。

处理方式：NO_GO，终止该研究线。归档负结果，记录"因子对市场风格高度敏感"。

---

案例 2：流动性因子 CONDITIONAL PASS

```
Backtest 阶段:
  多空年化收益: 18%
  Sharpe: 1.35
  最大回撤: -14%

Holdout 阶段（6 个月）:
  多空年化收益: 11.2%（退化 38%）
  Sharpe: 0.82（退化 39%）
  最大回撤: -19%（退化 36%）
  IC 均值: 0.032（Backtest 0.045，退化 29%）
```

诊断结论：过拟合风险可控。所有指标退化在 40% 以内，无强过拟合信号。收益下降主要因为 Holdout 期间市场波动率低于历史平均，流动性溢价收窄。

处理方式：CONDITIONAL PASS。限制初始资金为策略容量的 50%，设定 3 个月实盘观察期，Sharpe 低于 0.4 则自动止损退出。

---

案例 3：动量因子 EXTEND——环境不匹配

```
Backtest 阶段:
  多空年化收益: 22%
  Sharpe: 1.55
  最大回撤: -16%

Holdout 阶段（6 个月，处于震荡市）:
  多空年化收益: 3.5%（退化 84%）
  Sharpe: 0.25
  最大回撤: -22%
  IC 均值: 0.012（Backtest 0.05）
```

环境分析：Holdout 期间市场处于窄幅震荡（月度波动 < 5%），动量因子需要趋势性行情才能发挥作用。这不是因子过拟合，而是 Regime 不匹配。

处理方式：EXTEND_OBSERVATION。延期 3 个月观察，如果在趋势性行情回归后 Sharpe 恢复到 0.6 以上，则 CONDITIONAL PASS。同时建议将动量因子与 Regime 判断条件结合使用。

---

## 8. 失败处理

### 8.1 失败原因分析框架

```yaml
failure_analysis_framework:

  维度 1: 过拟合
    indicators:
      - "收益大幅退化"
      - "IC 方向反转"
      - "回撤模式完全改变"
    actions:
      - "归档负结果"
      - "分析过拟合来源（参数过多？样本太少？）"
      - "提取经验教训"

  维度 2: 市场环境变化
    indicators:
      - "特定 Regime 下失效"
      - "Holdout 环境与历史显著不同"
      - "因子经济学逻辑仍然成立"
    actions:
      - "考虑 EXTEND_OBSERVATION"
      - "评估加入 Regime 条件判断"
      - "考虑与其他因子组合"

  维度 3: 执行层面问题
    indicators:
      - "毛收益正常但净收益为负"
      - "滑点或成本远超 Backtest 预期"
      - "流动性严重恶化"
    actions:
      - "评估是否降低再平衡频率"
      - "剔除流动性瓶颈标的"
      - "缩小策略规模"

  维度 4: 因子本身失效
    indicators:
      - "IC 持续为负或趋近于零"
      - "因子经济学逻辑不再成立"
      - "市场结构发生根本性变化"
    actions:
      - "NO_GO，终止研究线"
      - "归档经验教训"
      - "探索新的因子方向"
```

### 8.2 Rollback 与 Child Lineage

```yaml
rollback_rules:

  允许 Rollback 的情况:
    - "代码实现 bug"
    - "数据处理错误"
    - "成本模型参数错误"
    - "时间戳对齐问题"

  Rollback 禁止事项:
    - "不能修改因子公式"
    - "不能修改分位切点"
    - "不能重新划定时间窗口"
    - "不能选择性报告结果"

  Child Lineage 适用情况:
    - "因子需要结构性调整"
    - "Universe 需要重新定义"
    - "信号机制需要改进"
    - "需要全新的研究方向"
```

---

## 9. 输出 Artifact

### 9.1 机器可读产物

```yaml
holdout_results.parquet:
  用途: Holdout 验证核心结果
  粒度: 日频
  字段: date, portfolio_return, gross_return, cost, drawdown
  消费者: Shadow Admission

holdout_metrics.json:
  用途: 关键指标汇总
  内容: IC, return, sharpe, calmar, drawdown, turnover
  消费者: 决策支持、报告生成

degradation_analysis.json:
  用途: 指标退化分析
  内容: 各指标 Backtest vs Holdout 对比
  消费者: 过拟合诊断

overfitting_diagnosis.json:
  用途: 过拟合诊断报告
  内容: 各维度过拟合信号、综合判定
  消费者: 晋级/淘汰决策

holdout_environment_analysis.json:
  用途: Holdout 期间市场环境分析
  内容: 市场收益、波动率、Regime 分布、极端事件
  消费者: 环境归因分析
```

### 9.2 人类可读产物

```yaml
holdout_report.md:
  用途: Holdout 阶段完整报告
  内容: 按"终审报告模板"撰写
  消费者: 团队评审、决策参考

gate_decision.md:
  用途: 门禁决策文档
  必需字段:
    stage: "Holdout"
    status: "PASS / CONDITIONAL_PASS / EXTEND / NO_GO"
    decision_basis: [决策依据]
    frozen_scope: [确认冻结的内容]
    next_steps: [下一步行动]
  消费者: 流程管理

negative_results/:
  用途: 失败记录归档
  内容: 失败的 Holdout 运行、诊断报告
  重要性: 避免幸存者偏差

failure_analysis.md:
  用途: 失败原因分析（NO_GO 时必需）
  内容: 失败原因、经验教训、改进建议
  消费者: 团队学习、未来研究参考
```

---

## 10. 与 Shadow Admission / 实盘的交接

### 10.1 晋级后的交接内容

```yaml
handover_on_pass:

  完整传递的冻结内容:
    - 因子公式和参数（来自 Train）
    - 分组规则和分位切点（来自 Test）
    - 交易规则和成本模型（来自 Backtest）
    - 执行方案（来自 Backtest）
    - Holdout 验证结论

  Shadow Admission 需要额外关注:
    - Holdout 识别的 Regime 风险
    - 策略容量上限和 AUM 限制
    - 成本模型在当前市场环境下的有效性
    - 实盘监控指标和止损条件

  CONDITIONAL_PASS 的额外约束:
    - 初始资金限制（通常为容量的 30-50%）
    - 实盘观察期（通常 1-3 个月）
    - 止损线（Sharpe < 阈值自动退出）
    - 定期复审（每周/每月）
```

### 10.2 实盘监控指标

```yaml
live_monitoring_metrics:

  日频监控:
    - "日度组合收益"
    - "日度回撤"
    - "因子 IC"

  周频监控:
    - "周度 Sharpe"
    - "换手率和交易成本"
    - "Regime 判断"

  月频监控:
    - "月度多空收益"
    - "月度 IC 和 IR"
    - "与 Backtest/Holdout 的对比"
    - "过拟合预警指标"

  止损条件:
    hard_stop: "累计亏损 > 15% 自动退出"
    soft_stop: "连续 3 个月 Sharpe < 0.3 触发复审"
    regime_stop: "Regime 切换到已知不利环境时降仓"
```

---

**文档版本**: v1.0
**最后更新**: 2026-04-02
**适用领域**: 横截面因子研究
