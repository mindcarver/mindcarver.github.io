# Shadow Admission 阶段详细文档

**文档ID**: QSP-SA-v1.0
**阶段编号**: 06
**阶段名称**: Shadow Admission (影子准入)
**日期**: 2026-03-26
**状态**: v1.0
**负责角色**: PM + Risk + Quant Researcher + Trader

---

## 目录

1. [阶段定义与核心目的](#1-阶段定义与核心目的)
2. [为什么需要 Shadow Admission](#2-为什么需要-shadow-admission)
3. [Shadow Admission 与实盘的关系](#3-shadow-admission-与实盘的关系)
4. [执行语义补充](#4-执行语义补充)
5. [撮合假设明确](#5-撮合假设明确)
6. [容量评估](#6-容量评估)
7. [成本模型精化](#7-成本模型精化)
8. [监控方案设计](#8-监控方案设计)
9. [风险控制机制](#9-风险控制机制)
10. [Formal Gate 要求](#10-formal-gate-要求)
11. [常见错误与防范](#11-常见错误与防范)
12. [输出 Artifact](#12-输出-artifact)
13. [投产交接](#13-投产交接)
14. [影子准入文档模板](#14-影子准入文档模板)

---

## 1. 阶段定义与核心目的

### 1.1 阶段定义

**Shadow Admission (影子准入)** 是策略进入模拟实盘环境前的治理准入阶段。

**核心特征**：
- **模拟实盘**: 在接近真实的环境中运行策略
- **治理准入**: 不等于最终投产，只是更高级别的准入
- **风险隔离**: 资金隔离，不会影响实盘
- **全面验证**: 验证策略在更真实条件下的表现

### 1.2 核心目的

**为什么需要 Shadow Admission**：

| 目的 | 说明 | 价值 |
|------|------|------|
| **执行验证** | 验证策略在真实执行环境下是否可行 | 发现回测未暴露的问题 |
| **容量验证** | 验证策略在实际资金规模下是否可行 | 避免容量不足导致收益衰减 |
| **成本验证** | 验证成本模型是否准确 | 避免成本吞噬利润 |
| **风险验证** | 验证风险控制机制是否有效 | 防止极端损失 |
| **监控验证** | 验证监控方案是否完善 | 确保投产后的可观测性 |

### 1.3 与前序阶段的关系

```
Holdout Validation (最终验证)
    ↓ PASS
Shadow Admission (影子准入)
    ↓ PASS
Production (投产)
```

**关键差异**：
```yaml
stage_differences:

  backtest_ready:
    environment: "历史数据回测"
    execution: "模拟撮合"
    cost: "简化成本模型"
    focus: "策略逻辑验证"

  shadow_admission:
    environment: "模拟实盘 (实时数据)"
    execution: "模拟撮合或小规模实盘"
    cost: "精确成本模型"
    focus: "执行可行性验证"

  production:
    environment: "实盘"
    execution: "真实撮合"
    cost: "真实成本"
    focus: "策略运行"
```

---

## 2. 为什么需要 Shadow Admission

### 2.1 回测与实盘的差距

**常见差距**：
```yaml
gaps_between_backtest_and_production:

  execution_gaps:
    issue: "撮合假设过于理想"
    examples:
      - "假设立即成交"
      - "假设无滑点"
      - "假设无流动性约束"
    consequence: "实盘收益大幅低于回测"

  cost_gaps:
    issue: "成本模型不准确"
    examples:
      - "低估手续费"
      - "忽略融资成本"
      - "忽略税务成本"
    consequence: "净利润被成本吞噬"

  capacity_gaps:
    issue: "容量估计不足"
    examples:
      - "忽略自冲击"
      - "忽略流动性约束"
      - "忽略市场影响"
    consequence: "规模扩大后收益衰减"

  operational_gaps:
    issue: "运营问题未考虑"
    examples:
      - "系统故障"
      - "网络延迟"
      - "人为错误"
    consequence: "策略无法正常运行"
```

### 2.2 Shadow Admission 的价值

**价值示例**：
```yaml
example_value:

  case_1_execution_issue:
    backtest_result: "Sharpe 2.5, 年化 30%"
    shadow_discovery: "订单无法及时成交，实际滑点 0.5%"
    shadow_result: "Sharpe 1.2, 年化 12%"
    action: "优化订单逻辑或放弃策略"

  case_2_capacity_issue:
    backtest_result: "$1M 资金下 Sharpe 2.0"
    shadow_discovery: "$500K 后自冲击显著"
    shadow_result: "$1M 资金下 Sharpe 0.8"
    action: "限制规模或修改策略"

  case_3_cost_issue:
    backtest_result: "毛利 25%, 净利 20%"
    shadow_discovery: "实际成本 8%, 净利 17%"
    shadow_result: "仍可接受，但收益低于预期"
    action: "优化成本或调整预期"
```

---

## 3. Shadow Admission 与实盘的关系

### 3.1 模拟实盘 vs 真实实盘

**对比表**：
```yaml
shadow_vs_production:

  capital:
    shadow: "隔离资金，不影响实盘"
    production: "真实资金"

  execution:
    shadow: "模拟撮合 (或小规模真实撮合)"
    production: "真实撮合"

  risk:
    shadow: "策略风险不影响实际资产"
    production: "策略风险影响实际资产"

  duration:
    shadow: "通常 1-3 个月"
    production: "长期运行"

  purpose:
    shadow: "验证策略可行性"
    production: "实现策略收益"
```

### 3.2 Shadow Admission 的两种模式

**模式 A：纯模拟模式**：
```yaml
pure_simulation_mode:

  execution: "模拟撮合引擎"
  data_feed: "实时市场数据"
  capital: "虚拟资金"
  orders: "生成订单但不实际执行"
  fills: "基于市场数据模拟成交"
  advantages:
    - "零资金风险"
    - "快速迭代"
    - "可测试极端情况"
  disadvantages:
    - "撮合假设可能仍不真实"
    - "无法验证真实执行"
```

**模式 B：小规模实盘模式**：
```yaml
small_scale_production_mode:

  execution: "真实撮合"
  data_feed: "实时市场数据"
  capital: "小规模真实资金 (如 $10K)"
  orders: "真实订单执行"
  fills: "真实成交"
  advantages:
    - "完全真实的执行环境"
    - "验证真实成本"
  disadvantages:
    - "有资金风险"
    - "可能影响市场 (如资金过小)"
```

### 3.3 模式选择

**选择指南**：
```yaml
mode_selection_criteria:

  use_pure_simulation:
    - "新策略首次验证"
    - "执行逻辑复杂，需要充分测试"
    - "资金预算有限"
    - "市场流动性充足"

  use_small_scale_production:
    - "策略已经过充分验证"
    - "需要验证真实撮合"
    - "需要验证真实成本"
    - "有足够的风险承受能力"
```

---

## 4. 执行语义补充

### 4.1 执行语义定义

**执行语义** 是策略如何生成和执行订单的详细规范。

**必需内容**：
```yaml
execution_semantics:

  order_generation:
    - "订单生成时机"
    - "订单类型 (市价/限价)"
    - "订单大小计算"
    - "订单方向 (多头/空头)"

  order_routing:
    - "交易所选择"
    - "路由策略"
    - "拆单规则"

  execution_logic:
    - "成交逻辑"
    - "部分成交处理"
    - "成交失败处理"

  position_management:
    - "持仓管理"
    - "仓位调整"
    - "平仓逻辑"
```

### 4.2 执行语义文档模板

**详细模板**：
```markdown
## 执行语义规范

### 信号到订单的转换

**信号读取**:
- 信号源: {信号来源}
- 读取频率: {频率}
- 信号阈值: {阈值}

**订单生成**:
```python
def generate_order(signal, current_position):
    """
    根据信号生成订单
    """
    if signal > entry_threshold and current_position == 0:
        return Order(type='MARKET', side='BUY', size=calculate_size())
    elif signal < exit_threshold and current_position > 0:
        return Order(type='MARKET', side='SELL', size=current_position)
    else:
        return None
```

### 订单类型选择

| 场景 | 订单类型 | 理由 |
|------|---------|------|
| 入场 | {类型} | {理由} |
| 出场 | {类型} | {理由} |
| 止损 | {类型} | {理由} |

### 订单大小计算

**基础订单大小**:
```
size = capital * leverage * target_exposure / current_price
```

**约束条件**:
- 最小订单: {约束}
- 最大订单: {约束}
- 流动性约束: {约束}

### 部分成交处理

**策略**:
- 订单拆分: {是否}
- 拆分规则: {规则}
- 超时处理: {处理方式}

### 成交失败处理

**重试逻辑**:
- 重试次数: {N}
- 重试间隔: {时间}
- 失败处理: {处理方式}
```

### 4.3 执行语义验证

**验证检查**：
```python
def validate_execution_semantics(execution_spec):
    """
    验证执行语义的完整性
    """
    checks = {
        'order_generation_defined': 'order_generation' in execution_spec,
        'order_routing_defined': 'order_routing' in execution_spec,
        'execution_logic_defined': 'execution_logic' in execution_spec,
        'position_management_defined': 'position_management' in execution_spec,
        'error_handling_defined': 'error_handling' in execution_spec,
    }

    all_passed = all(checks.values())

    return {
        'passed': all_passed,
        'checks': checks,
        'missing': [k for k, v in checks.items() if not v]
    }
```

---

## 5. 撮合假设明确

### 5.1 撮合假设的重要性

**撮合假设** 是关于订单如何成交的假设。

**关键假设**：
```yaml
filling_assumptions:

  market_orders:
    assumption: "市价单立即成交"
    reality: "可能有滑点和部分成交"
    mitigation: "记录滑点，调整预期"

  limit_orders:
    assumption: "限价单在指定价格成交"
    reality: "可能不成交"
    mitigation: "设置超时，转市价"

  slippage:
    assumption: "滑点为固定值或百分比"
    reality: "滑点随市场条件变化"
    mitigation: "使用动态滑点模型"

  partial_fills:
    assumption: "订单完全成交"
    reality: "可能部分成交"
    mitigation: "处理部分成交逻辑"
```

### 5.2 撮合模型

**撮合模型示例**：
```python
class FillingModel:
    """
    撮合模型
    """
    def __init__(self, config):
        self.slippage_model = config['slippage_model']
        self.partial_fill_model = config['partial_fill_model']
        self.fill_probability = config.get('fill_probability', 1.0)

    def simulate_fill(self, order, market_data):
        """
        模拟订单成交
        """
        # 1. 检查是否成交
        if random.random() > self.fill_probability:
            return None  # 未成交

        # 2. 计算滑点
        slippage = self.calculate_slippage(order, market_data)

        # 3. 计算成交价格
        if order.side == 'BUY':
            fill_price = market_data['ask'] + slippage
        else:
            fill_price = market_data['bid'] - slippage

        # 4. 计算成交量
        fill_size = self.calculate_fill_size(order, market_data)

        return {
            'price': fill_price,
            'size': fill_size,
            'timestamp': market_data['timestamp']
        }

    def calculate_slippage(self, order, market_data):
        """
        计算滑点
        """
        if self.slippage_model['type'] == 'fixed':
            return self.slippage_model['value']
        elif self.slippage_model['type'] == 'percentage':
            return order.price * self.slippage_model['value']
        elif self.slippage_model['type'] == 'dynamic':
            # 基于市场波动率的动态滑点
            volatility = market_data['volatility']
            return order.price * volatility * self.slippage_model['multiplier']

    def calculate_fill_size(self, order, market_data):
        """
        计算成交量
        """
        if self.partial_fill_model['type'] == 'full':
            return order.size
        elif self.partial_fill_model['type'] == 'fraction':
            # 基于流动性的部分成交
            liquidity_ratio = min(
                order.size / market_data['volume'],
                self.partial_fill_model['max_fill_ratio']
            )
            return order.size * liquidity_ratio
```

### 5.3 撮合假设文档

**文档模板**：
```yaml
filling_assumptions_spec:

  market_orders:
    fill_probability: 1.0  # 100% 成交
    slippage_model:
      type: "dynamic"
      base_bps: 5  # 基础 5bp
      volatility_multiplier: 0.1  # 波动率系数

  limit_orders:
    fill_probability: 0.8  # 80% 成交
    time_in_force: "GTC"  # Good Till Cancelled
    expiry_minutes: 60

  partial_fills:
    enabled: true
    max_fill_ratio: 0.2  # 最多成交 20% 的量
    min_fill_size: 0.001  # 最小成交量

  market_impact:
    enabled: true
    model: "square_root"
    impact_coefficient: 0.01

  documentation: >
    撮合假设基于历史订单统计分析。
    在 Shadow Admission 阶段将验证这些假设的准确性。
```

---

## 6. 容量评估

### 6.1 容量的定义

**策略容量** 是策略能够容纳的资金规模上限。

**容量限制因素**：
```yaml
capacity_limiting_factors:

  liquidity:
    factor: "市场流动性"
    description: "日交易量限制可交易规模"
    measure: "策略交易量 / 市场交易量"

  market_impact:
    factor: "自冲击"
    description: "策略自身交易对价格的影响"
    measure: "冲击成本"

  concentration:
    factor: "持仓集中度"
    description: "单一标的上过大仓位"
    measure: "单一持仓 / 总资金"

  execution:
    factor: "执行能力"
    description: "订单执行速度和能力"
    measure: "订单成交率和延迟"
```

### 6.2 容量评估方法

**评估方法**：
```python
def estimate_strategy_capacity(backtest_results, market_data):
    """
    估计策略容量
    """
    capacity_analysis = {}

    # 1. 流动性约束
    # 策略日均交易量不应超过市场日均交易量的 1%
    avg_strategy_volume = backtest_results['daily_volume'].mean()
    avg_market_volume = market_data['daily_volume'].mean()

    liquidity_capacity = avg_market_volume * 0.01 / backtest_results['turnover'].mean()

    # 2. 自冲击约束
    # 自冲击成本 = coefficient * sqrt(策略交易量 / 市场交易量)
    # 设定最大可接受冲击成本为 5bp
    max_impact_cost = 0.0005  # 5bp
    impact_coefficient = 0.01

    # 反推最大交易量
    max_volume_ratio = (max_impact_cost / impact_coefficient) ** 2
    impact_capacity = avg_market_volume * max_volume_ratio

    # 3. 集中度约束
    # 单一持仓不超过总资金的 10%
    max_single_position = backtest_results['max_position_size'].max()
    concentration_capacity = max_single_position / 0.10

    # 4. 综合容量
    capacity_analysis['liquidity_capacity'] = liquidity_capacity
    capacity_analysis['impact_capacity'] = impact_capacity
    capacity_analysis['concentration_capacity'] = concentration_capacity

    # 取最小值作为策略容量
    capacity_analysis['estimated_capacity'] = min(
        liquidity_capacity,
        impact_capacity,
        concentration_capacity
    )

    return capacity_analysis
```

### 6.3 容量衰减曲线

**容量与收益的关系**：
```python
def simulate_capacity_decay(backtest_results, capital_range):
    """
    模拟不同资金规模下的收益衰减
    """
    decay_curve = []

    for capital in capital_range:
        # 按比例放大交易量
        scaled_volume = backtest_results['volume'] * (capital / backtest_results['base_capital'])

        # 计算冲击成本
        impact_cost = calculate_impact_cost(scaled_volume, backtest_results['market_volume'])

        # 计算净收益 (扣除冲击成本)
        gross_return = backtest_results['return']
        net_return = gross_return - impact_cost

        # 计算 Sharpe
        sharpe = net_return.mean() / net_return.std() * np.sqrt(252)

        decay_curve.append({
            'capital': capital,
            'gross_return': gross_return.mean() * 252,
            'impact_cost': impact_cost.mean() * 252,
            'net_return': net_return.mean() * 252,
            'sharpe': sharpe
        })

    return pd.DataFrame(decay_curve)
```

### 6.4 容量评估报告

**报告模板**：
```markdown
## 容量评估报告

### 估计容量
| 约束类型 | 容量上限 | 限制因素 |
|---------|---------|---------|
| 流动性约束 | ${X}M | 日均交易量 |
| 自冲击约束 | ${Y}M | 冲击成本 |
| 集中度约束 | ${Z}M | 单一持仓限制 |

**综合容量估计**: ${min(X,Y,Z)}M

### 容量衰减曲线
| 资金规模 | 毛收益 | 冲击成本 | 净收益 | Sharpe |
|---------|--------|---------|--------|--------|
| $100K | XX% | X% | XX% | X.XX |
| $500K | XX% | X% | XX% | X.XX |
| $1M | XX% | X% | XX% | X.XX |
| $5M | XX% | X% | XX% | X.XX |

### 推荐规模
**保守规模**: ${容量 * 0.5}M
**正常规模**: ${容量 * 0.7}M
**最大规模**: ${容量}M

### 风险提示
{容量相关的风险提示}
```

---

## 7. 成本模型精化

### 7.1 成本组成

**完整成本模型**：
```yaml
cost_components:

  trading_costs:
    commission:
      description: "交易手续费"
      model: "按交易量百分比"
      typical_value: "0.1% (双向)"

    slippage:
      description: "滑点成本"
      model: "动态模型"
      typical_value: "2-10bp"

    market_impact:
      description: "自冲击成本"
      model: "平方根模型"
      typical_value: "与交易量相关"

  holding_costs:
    funding_cost:
      description: "资金成本"
      model: "利率 × 持仓价值"

    borrowing_cost:
      description: "借券/做空成本"
      model: "费率 × 借券价值"

  operational_costs:
    infrastructure:
      description: "基础设施成本"
      model: "固定成本"

    monitoring:
      description: "监控和维护成本"
      model: "固定成本"

  other_costs:
    taxes:
      description: "税务成本"
      model: "根据税率计算"

    regulatory:
      description: "监管成本"
      model: "固定成本"
```

### 7.2 成本模型实现

**成本计算函数**：
```python
def calculate_total_costs(trades, positions, cost_config):
    """
    计算总成本
    """
    total_costs = 0

    # 1. 交易成本
    for trade in trades:
        # 手续费
        commission = trade['value'] * cost_config['commission_rate']
        total_costs += commission

        # 滑点
        slippage = trade['value'] * cost_config['slippage_rate']
        total_costs += slippage

        # 市场冲击
        market_impact = cost_config['impact_coefficient'] * np.sqrt(
            trade['value'] / trade['market_volume']
        ) * trade['value']
        total_costs += market_impact

    # 2. 持仓成本
    for position in positions:
        # 资金成本
        funding_cost = position['value'] * cost_config['funding_rate'] * position['days_held']
        total_costs += funding_cost

        # 借券成本 (如果有空头)
        if position['size'] < 0:
            borrowing_cost = abs(position['value']) * cost_config['borrowing_rate'] * position['days_held']
            total_costs += borrowing_cost

    return total_costs
```

### 7.3 成本模型验证

**验证方法**：
```yaml
cost_model_validation:

  historical_comparison:
    method: "与历史实际成本对比"
    requirement: "模型成本 vs 实际成本差异 < 20%"

  shadow_admission_testing:
    method: "在影子准入中验证"
    requirement: "记录实际成本，调整模型"

  sensitivity_analysis:
    method: "参数敏感性分析"
    requirement: "关键参数变动 ±50% 评估影响"
```

---

## 8. 监控方案设计

### 8.1 监控指标体系

**核心监控指标**：
```yaml
monitoring_metrics:

  performance_metrics:
    - name: "daily_pnl"
      description: "日盈亏"
      frequency: "daily"

    - name: "cumulative_return"
      description: "累计收益"
      frequency: "daily"

    - name: "sharpe_ratio"
      description: "夏普比率"
      frequency: "weekly"

    - name: "max_drawdown"
      description: "最大回撤"
      frequency: "daily"

  execution_metrics:
    - name: "fill_rate"
      description: "成交率"
      frequency: "daily"

    - name: "slippage"
      description: "平均滑点"
      frequency: "daily"

    - name: "order_latency"
      description: "订单延迟"
      frequency: "hourly"

  risk_metrics:
    - name: "position_concentration"
      description: "持仓集中度"
      frequency: "daily"

    - name: "var_95"
      description: "95% VaR"
      frequency: "daily"

    - name: "beta"
      description: "市场 Beta"
      frequency: "weekly"

  operational_metrics:
    - name: "system_uptime"
      description: "系统可用性"
      frequency: "hourly"

    - name: "data_latency"
      description: "数据延迟"
      frequency: "hourly"

    - name: "error_rate"
      description: "错误率"
      frequency: "daily"
```

### 8.2 告警规则

**告警配置**：
```yaml
alert_rules:

  performance_alerts:
    - metric: "daily_pnl"
      condition: "< -$10,000"
      severity: "HIGH"
      action: "立即通知 PM 和 Risk"

    - metric: "max_drawdown"
      condition: "< -15%"
      severity: "MEDIUM"
      action: "通知团队，评估是否暂停"

  execution_alerts:
    - metric: "fill_rate"
      condition: "< 80%"
      severity: "HIGH"
      action: "检查撮合问题"

    - metric: "slippage"
      condition: "> 20bp"
      severity: "MEDIUM"
      action: "评估订单策略"

  risk_alerts:
    - metric: "position_concentration"
      condition: "> 20%"
      severity: "MEDIUM"
      action: "评估集中度风险"

    - metric: "var_95"
      condition: ">$50,000"
      severity: "HIGH"
      action: "降低仓位"

  operational_alerts:
    - metric: "system_uptime"
      condition: "< 99%"
      severity: "HIGH"
      action: "立即修复"

    - metric: "error_rate"
      condition: "> 5%"
      severity: "MEDIUM"
      action: "调查错误原因"
```

### 8.3 监控仪表盘

**仪表盘设计**：
```yaml
dashboard_layout:

  overview_section:
    - "累计收益曲线"
    - "当前持仓"
    - "今日盈亏"
    - "关键指标"

  performance_section:
    - "收益分布"
    - "回撤分析"
    - "Sharpe 趋势"

  execution_section:
    - "成交率"
    - "滑点分析"
    - "订单延迟"

  risk_section:
    - "持仓分析"
    - "VaR 分析"
    - "Beta 分析"

  operational_section:
    - "系统状态"
    - "数据质量"
    - "错误日志"
```

---

## 9. 风险控制机制

### 9.1 风险限制

**风险限制设置**：
```yaml
risk_limits:

  position_limits:
    max_total_exposure: "1.0"  # 100% 总资金
    max_single_position: "0.10"  # 10% 单一持仓
    max_sector_exposure: "0.30"  # 30% 行业暴露

  leverage_limits:
    max_gross_exposure: "1.5"  # 150% 总敞口
    max_net_exposure: "0.5"  # 50% 净敞口

  drawdown_limits:
    max_daily_drawdown: "-0.05"  # -5% 日最大回撤
    max_total_drawdown: "-0.15"  # -15% 总最大回撤"

  loss_limits:
    max_daily_loss: "-$20,000"  # 日最大亏损
    max_weekly_loss: "-$50,000"  # 周最大亏损"

  concentration_limits:
    max_correlation_positions: "5"  # 最大相关持仓数
    min_diversification_ratio: "0.5"  # 最小分散度
```

### 9.2 风险控制流程

**风险控制机制**：
```python
class RiskController:
    """
    风险控制器
    """
    def __init__(self, risk_limits):
        self.limits = risk_limits
        self.current_state = {}

    def check_order(self, order, current_positions):
        """
        检查订单是否符合风险限制
        """
        checks = {}

        # 1. 检查单一持仓限制
        if order.symbol in current_positions:
            new_size = current_positions[order.symbol] + order.size
            checks['single_position'] = abs(new_size) <= self.limits['max_single_position']
        else:
            checks['single_position'] = abs(order.size) <= self.limits['max_single_position']

        # 2. 检查总敞口限制
        total_exposure = sum(abs(p) for p in current_positions.values())
        new_exposure = total_exposure + abs(order.size)
        checks['total_exposure'] = new_exposure <= self.limits['max_total_exposure']

        # 3. 检查净敞口限制
        net_exposure = sum(current_positions.values())
        new_net = net_exposure + order.size
        checks['net_exposure'] = abs(new_net) <= self.limits['max_net_exposure']

        # 4. 检查回撤限制
        checks['drawdown'] = self.check_drawdown_limit()

        return all(checks.values()), checks

    def check_drawdown_limit(self):
        """
        检查回撤限制
        """
        current_drawdown = self.calculate_current_drawdown()
        return current_drawdown >= self.limits['max_total_drawdown']

    def check_loss_limit(self, daily_pnl):
        """
        检查亏损限制
        """
        return daily_pnl >= self.limits['max_daily_loss']
```

### 9.3 紧急停止机制

**紧急停止触发条件**：
```yaml
emergency_stop_triggers:

  immediate_stop:
    - "系统故障或数据异常"
    - "超出最大回撤限制"
    - "超出最大亏损限制"
    - "风险控制失效"

  gradual_stop:
    - "持续亏损 (如连续 5 天)"
    - "市场环境变化"
    - "策略性能下降"

  stop_procedure:
    1. "停止新订单"
    2. "平仓现有持仓"
    3. "记录停止原因"
    4. "通知相关人员"
    5. "分析停止原因"
```

---

## 10. Formal Gate 要求

### 10.1 Shadow Admission 阶段 Formal Gate

**必需检查项**：
```yaml
shadow_admission_formal_gates:

  execution_semantics_complete:
    requirement: "执行语义已完整定义"
    criteria:
      - "订单生成逻辑清晰"
      - "撮合假设明确"
      - "异常处理完善"
    evidence: "execution_semantics.md"
    status: "PASS | FAIL"

  capacity_assessed:
    requirement: "策略容量已评估"
    criteria:
      - "容量估计完成"
      - "衰减曲线分析完成"
      - "推荐规模明确"
    evidence: "capacity_assessment_report.md"
    status: "PASS | FAIL"

  cost_model_refined:
    requirement: "成本模型已精化"
    criteria:
      - "所有成本项已考虑"
      - "成本模型已验证"
      - "成本在可接受范围"
    evidence: "cost_model_report.md"
    status: "PASS | FAIL"

  monitoring_designed:
    requirement: "监控方案已设计"
    criteria:
      - "监控指标完整"
      - "告警规则明确"
      - "仪表盘设计完成"
    evidence: "monitoring_plan.md"
    status: "PASS | FAIL"

  risk_controls_defined:
    requirement: "风险控制机制已定义"
    criteria:
      - "风险限制明确"
      - "风险控制流程清晰"
      - "紧急停止机制完善"
    evidence: "risk_control_plan.md"
    status: "PASS | FAIL"

  shadow_period_completed:
    requirement: "影子准入期已完成"
    criteria:
      - "运行时间 ≥ 1 个月"
      - "市场环境代表性"
      - "问题已记录和解决"
    evidence: "shadow_period_summary.md"
    status: "PASS | FAIL"
```

### 10.2 决策状态

**可能的状态**：
```yaml
verdict_states:

  PASS:
    description: "所有 Formal Gate 通过，可进入投产"
    conditions:
      - "Shadow 期表现符合预期"
      - "执行可行性确认"
      - "风险可控"
      - "监控方案完善"

  CONDITIONAL_PASS:
    description: "核心要求通过，但有条件限制"
    conditions:
      - "限制规模"
      - "加强监控"
      - "定期审查"

  RETRY:
    description: "执行或成本问题，修复后重试"
    scope:
      - "优化执行逻辑"
      - "调整撮合假设"
      - "修改成本模型"

  EXTEND_SHADOW:
    description: "需要延长影子准入期"
    reasons:
      - "市场环境不典型"
      - "需要更多数据验证"
      - "发现新问题需要解决"

  NO_GO:
    description: "策略不适合实盘，终止"
    reasons:
      - "执行不可行"
      - "容量不足"
      - "成本过高"
      - "风险不可控"
```

---

## 11. 常见错误与防范

### 11.1 错误清单

| 错误类型 | 描述 | 后果 | 防范措施 |
|---------|------|------|---------|
| **过度乐观** | 假设撮合过于理想 | 实盘收益远低于预期 | 使用保守撮合假设 |
| **忽略容量** | 不评估策略容量 | 规模扩大后收益衰减 | 提前评估容量 |
| **低估成本** | 成本模型不完整 | 净利润被成本吞噬 | 考虑所有成本项 |
| **缺少监控** | 不设计监控方案 | 问题发现不及时 | 完善监控体系 |
| **风险不足** | 风险控制不严格 | 极端情况下大额亏损 | 严格风险限制 |
| **急于投产** | Shadow 期不够充分 | 实盘暴露未发现问题 | 充分运行 Shadow |

### 11.2 防范机制

**流程层面**：
```yaml
safety_mechanisms:

  conservative_assumptions:
    principle: "使用保守假设，给实盘留余量"
    application:
      - "撮合假设保守"
      - "成本估计高估"
      - "容量估计低估"

  staged_rollout:
    principle: "分阶段投产，逐步扩大规模"
    stages:
      - "纯模拟"
      - "小规模实盘"
      - "中等规模"
      - "全规模"

  continuous_validation:
    principle: "持续验证假设"
    actions:
      - "对比实际与假设"
      - "调整模型"
      - "更新预期"

  independent_review:
    principle: "独立审查关键决策"
    reviewers:
      - "Risk 独立审查风险限制"
      - "Trader 独立审查执行方案"
      - "PM 独立审查整体可行性"
```

---

## 12. 输出 Artifact

### 12.1 机器可读产物

**必需文件**：
```yaml
machine_readable_artifacts:

  execution_semantics.yaml:
    description: "执行语义规范"
    content:
      - "订单生成逻辑"
      - "撮合假设"
      - "异常处理"

  capacity_assessment.yaml:
    description: "容量评估结果"
    content:
      - "容量估计"
      - "衰减曲线"
      - "推荐规模"

  cost_model.yaml:
    description: "成本模型配置"
    content:
      - "成本项定义"
      - "成本参数"
      - "验证结果"

  monitoring_config.yaml:
    description: "监控配置"
    content:
      - "监控指标"
      - "告警规则"
      - "仪表盘配置"

  risk_limits.yaml:
    description: "风险限制配置"
    content:
      - "风险限制"
      - "控制流程"
      - "停止机制"

  shadow_performance.parquet:
    description: "影子期表现数据"
    content:
      - "日收益数据"
      - "执行数据"
      - "成本数据"
```

### 12.2 人类可读产物

**必需文档**：
```yaml
human_readable_artifacts:

  shadow_admission_report.md:
    description: "影子准入总结报告"
    sections:
      - "执行摘要"
      - "执行语义验证"
      - "容量评估"
      - "成本模型验证"
      - "监控方案验证"
      - "风险控制验证"
      - "Shadow 期表现"
      - "问题与解决方案"
      - "投产建议"

  execution_semantics_documentation.md:
    description: "执行语义详细文档"
    content:
      - "订单生成详细逻辑"
      - "撮合假设详细说明"
      - "异常处理详细流程"

  monitoring_plan.md:
    description: "监控方案文档"
    content:
      - "监控指标定义"
      - "告警规则说明"
      - "仪表盘使用指南"

  risk_control_plan.md:
    description: "风险控制方案文档"
    content:
      - "风险限制说明"
      - "控制流程详细说明"
      - "紧急停止流程"

  shadow_period_summary.md:
    description: "影子期总结"
    content:
      - "运行期间关键事件"
      - "性能表现总结"
      - "发现的问题"
      - "实施的调整"
```

---

## 13. 投产交接

### 13.1 交接内容

**Shadow Admission → Production 交接清单**：
```yaml
production_handover:

  execution_ready:
    execution_semantics: "已验证的执行语义"
    filling_model: "已验证的撮合模型"
    infrastructure: "就绪的基础设施"

  capacity_confirmed:
    recommended_scale: "推荐规模"
    max_safe_scale: "最大安全规模"
    scale_up_plan: "扩容计划"

  cost_confirmed:
    cost_model: "验证后的成本模型"
    expected_net_return: "预期净收益"
    cost_monitoring: "成本监控方案"

  monitoring_ready:
    monitoring_system: "已部署的监控系统"
    alert_system: "已配置的告警系统"
    dashboard: "已配置的仪表盘"

  risk_controls_ready:
    risk_limits: "已配置的风险限制"
    risk_controller: "已部署的风险控制器"
    emergency_stop: "已测试的紧急停止"

  documentation_complete:
    all_plans: "所有方案文档完整"
    runbooks: "操作手册完整"
    contact_list: "联系人清单"
```

### 13.2 投产前最终检查

**最终检查清单**：
```yaml
pre_production_checks:

  technical_readiness:
    - [ ] 系统已部署并通过测试
    - [ ] 监控已配置并测试
    - [ ] 告警已配置并测试
    - [ ] 风险控制已部署并测试
    - [ ] 紧急停止已测试

  operational_readiness:
    - [ ] 操作员已培训
    - [ ] 值班表已安排
    - [ ] 联系人已确认
    - [ ] 应急流程已明确

  governance_readiness:
    - [ ] PM 批准投产
    - [ ] Risk 批准风险限制
    - [ ] Trader 确认执行方案
    - [ ] 合规审查通过

  documentation_readiness:
    - [ ] 所有文档完整
    - [ ] 操作手册更新
    - [ ] 应急预案准备
    - [ ] 交接记录完整
```

### 13.3 投产决策

**决策框架**：
```yaml
production_decision:

  go_live:
    conditions:
      - "所有 Formal Gate 通过"
      - "Shadow 期表现良好"
      - "无未解决的重大问题"
      - "所有准备工作完成"

  conditional_go_live:
    conditions:
      - "核心 Formal Gate 通过"
      - "Shadow 期表现可接受"
      - "有已知但有条件的问题"
      - "准备工作基本完成"

    conditions_list:
      - "限制规模在推荐范围内"
      - "加强监控频率"
      - "定期审查 (如每周)"
      - "准备降级方案"

  delay_production:
    conditions:
      - "有未解决的重大问题"
      - "Shadow 期表现不理想"
      - "准备工作未完成"
      - "需要更多验证时间"

  cancel_production:
    conditions:
      - "策略不适合实盘"
      - "风险不可控"
      - "成本过高"
      - "执行不可行"
```

---

## 14. 影子准入文档模板

### 14.1 Shadow Admission 报告模板

```markdown
---
doc_id: SA-{lineage_id}-{run_id}
title: Shadow Admission Report — {策略名称}
date: YYYY-MM-DD
status: PASS | CONDITIONAL_PASS | RETRY | EXTEND_SHADOW | NO_GO
version: 1.0
owner: PM + Risk + Quant Researcher
lineage_id: {lineage_id}
run_id: {run_id}
---

## 执行摘要

**策略名称**: {策略名称}
**影子期**: {开始日期} 至 {结束日期}
**运行模式**: {纯模拟 / 小规模实盘}

**总体评估**: {PASS / CONDITIONAL_PASS / RETRY / EXTEND_SHADOW / NO_GO}

**核心发现**:
- 执行可行性: {评估结果}
- 容量评估: {容量估计}
- 成本验证: {成本与预期对比}
- 风险控制: {风险事件和应对}

**投产建议**: {建议}

## 执行语义验证

### 订单生成
**验证结果**: {PASS / FAIL}
**发现的问题**:
- {问题描述}
- {解决方案}

### 撮合假设
**假设 vs 实际**:
| 假设项 | 假设值 | 实际值 | 差异 | 状态 |
|--------|--------|--------|------|------|
| 成交率 | {假设} | {实际} | {差异} | {状态} |
| 滑点 | {假设} | {实际} | {差异} | {状态} |

**假设调整**:
- {需要调整的假设}
- {调整原因}
- {新假设值}

### 异常处理
**测试的异常情况**:
- {异常情况1}: {处理结果}
- {异常情况2}: {处理结果}

## 容量评估

### 估计容量
| 约束类型 | 容量上限 | 限制因素 |
|---------|---------|---------|
| 流动性约束 | {容量} | {因素} |
| 自冲击约束 | {容量} | {因素} |
| 集中度约束 | {容量} | {因素} |

**综合容量**: ${容量}M

### 容量衰减
| 资金规模 | 毛收益 | 冲击成本 | 净收益 | Sharpe |
|---------|--------|---------|--------|--------|
| {规模1} | {收益} | {成本} | {净收益} | {Sharpe} |
| {规模2} | {收益} | {成本} | {净收益} | {Sharpe} |

### 推荐规模
- **保守**: ${规模}M
- **正常**: ${规模}M
- **最大**: ${规模}M

## 成本模型验证

### 成本对比
| 成本项 | 回测假设 | 影子期实际 | 差异 | 状态 |
|--------|---------|-----------|------|------|
| 手续费 | {假设} | {实际} | {差异} | {状态} |
| 滑点 | {假设} | {实际} | {差异} | {状态} |
| 冲击成本 | {假设} | {实际} | {差异} | {状态} |

### 收益影响
| 指标 | 回测 | 影子期 | 差异 |
|------|------|--------|------|
| 毛收益 | {值} | {值} | {差异} |
| 总成本 | {值} | {值} | {差异} |
| 净收益 | {值} | {值} | {差异} |
| Sharpe | {值} | {值} | {差异} |

## 监控方案验证

### 监控指标
**指标表现**:
- {指标1}: {描述}
- {指标2}: {描述}

### 告警触发
**触发的告警**:
- {告警1}: {次数} 次，原因: {原因}
- {告警2}: {次数} 次，原因: {原因}

### 仪表盘
**仪表盘使用情况**:
- {使用情况}
- {改进建议}

## 风险控制验证

### 风险事件
**记录的风险事件**:
| 日期 | 事件类型 | 描述 | 应对 | 结果 |
|------|---------|------|------|------|
| {日期} | {类型} | {描述} | {应对} | {结果} |

### 风险限制
**限制测试**:
| 限制类型 | 限制值 | 是否触发 | 应对 |
|---------|--------|---------|------|
| {限制1} | {值} | {是/否} | {应对} |
| {限制2} | {值} | {是/否} | {应对} |

## Shadow 期表现

### 收益表现
| 指标 | 回测 | 影子期 | 差异 |
|------|------|--------|------|
| 累计收益 | {值} | {值} | {差异} |
| 年化收益 | {值} | {值} | {差异} |
| Sharpe | {值} | {值} | {差异} |
| 最大回撤 | {值} | {值} | {差异} |

### 执行表现
| 指标 | 目标 | 实际 | 状态 |
|------|------|------|------|
| 成交率 | {目标} | {实际} | {状态} |
| 平均滑点 | {目标} | {实际} | {状态} |
| 订单延迟 | {目标} | {实际} | {状态} |

### 市场环境
**影子期市场环境**:
- {环境描述}
- {对策略的影响}

## 发现的问题与解决方案

### 严重问题
{如果有严重问题}

### 轻微问题
{如果有轻微问题}

### 实施的调整
{实施的调整和效果}

## 投产建议

### 决策
**建议**: {GO / CONDITIONAL_GO / DELAY / CANCEL}

### 理由
{详细理由}

### 投产条件
{如果是条件投产，列出条件}

### 风险提示
{投产风险提示}

### 后续计划
{投产后的监控和调整计划}

## 附录

### A. 详细执行数据
{详细执行数据}

### B. 成本明细
{成本明细}

### C. 监控图表
{监控图表}

### D. 风险事件日志
{风险事件日志}
```

---

**文档版本**: v1.0
**最后更新**: 2026-03-26
**下次评审**: 2026-06-26
