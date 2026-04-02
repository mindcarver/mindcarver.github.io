# Backtest 阶段 -- 回测仿真

## 1. 阶段定义

**Backtest（回测仿真）** 是将 Test 阶段验证通过的因子分组策略，在考虑真实交易成本、滑点、风险约束和容量限制的条件下，模拟完整交易执行过程的阶段。

本阶段回答的问题：

> 因子策略在真实交易约束下是否仍然盈利？策略容量和可实现性如何？

### 1.1 在流程中的定位

```
Test（样本外统计验证）
  ↓ 冻结：因子公式、分位切点、分组结果
Backtest（回测仿真） ← 你在这里
  ↓ 冻结：交易规则、成本模型、执行方案
Holdout（留存验证）
```

### 1.2 与 Test 阶段的区别

| 维度 | Test 阶段 | Backtest 阶段 |
|------|-----------|---------------|
| 成本 | 不考虑 | 必须考虑全部成本 |
| 滑点 | 不考虑 | 必须建模 |
| 风险约束 | 不施加 | 必须施加 |
| 容量 | 不评估 | 必须评估 |
| 执行 | 理想化 | 尽量贴近真实 |

---

## 2. 交易成本建模

### 2.1 成本构成全景

横截面策略涉及大量标的的多空组合，成本建模直接决定回测可信度。完整成本清单如下：

```
直接成本:
  ├── 交易手续费（Maker / Taker）
  ├── 平台使用费
  └── 监管成本（印花税、过户费）

资金成本:
  ├── 资金费率（永续合约，每 8h 结算一次）
  ├── 借贷成本（做空需要借入资产）
  └── 保证金机会成本

隐性成本:
  ├── 滑点（执行价格偏离中间价）
  ├── 市场冲击（大额订单推动价格）
  └── 延迟成本（信号到执行的时滞）

操作成本:
  ├── 撤单费用
  └── 维持保证金追加
```

### 2.2 手续费模型

主流加密货币交易所的费率结构（2026 年参考值）：

| 交易所 | Maker | Taker | 备注 |
|--------|-------|-------|------|
| Binance Futures | 0.02% | 0.04% | VIP 费率更低 |
| OKX Futures | 0.02% | 0.05% | 挂单返佣 |
| Bybit Futures | 0.01% | 0.06% | Maker 极低 |
| dYdX | 0.02% | 0.05% | 去中心化 |

```python
def calculate_trading_fee(trade_value_usd, maker_rate=0.0002, taker_rate=0.0004,
                          order_type='taker'):
    """
    交易手续费计算

    参数:
        trade_value_usd: 交易金额（USD）
        maker_rate: 挂单费率
        taker_rate: 吃单费率
        order_type: 订单类型 'maker' 或 'taker'

    返回:
        fee_usd: 手续费金额
    """
    if order_type == 'maker':
        return trade_value_usd * maker_rate
    else:
        return trade_value_usd * taker_rate
```

注意：
- 再平衡涉及数十个标的同时调仓，优先使用限价单降低成本
- 但限价单有成交概率问题，部分订单可能无法成交导致跟踪误差
- 回测中应假设保守的成交率（限价单约 70%-80% 成交）

### 2.3 资金费率成本

永续合约的资金费率是加密市场特有的成本项，对多空策略影响很大。

```python
def calculate_funding_cost(position_value_usd, funding_rates_series, position_side='long'):
    """
    资金费率成本计算

    参数:
        position_value_usd: 持仓金额
        funding_rates_series: 资金费率序列（每 8h 一次）
        position_side: 'long' 多头 / 'short' 空头

    返回:
        total_cost: 总资金费率成本
    """
    if position_side == 'long':
        # 多头：正费率时支付，负费率时收取
        cost = position_value_usd * funding_rates_series.clip(lower=0).sum()
        rebate = position_value_usd * funding_rates_series.clip(upper=0).sum()
        return cost + rebate
    else:
        # 空头：正费率时收取，负费率时支付
        cost = position_value_usd * funding_rates_series.clip(upper=0).sum()
        rebate = position_value_usd * funding_rates_series.clip(lower=0).sum()
        return cost + rebate
```

注意：
- 年化资金费率通常在 -10% ~ +30%，极端行情下可达 +100%+
- 做多热门标的（如 memecoin）时资金费率成本极高
- 回测必须使用实际的历史资金费率数据，不能用常数假设

### 2.4 借贷成本（做空）

```python
def calculate_borrow_cost(short_value_usd, borrow_rate_series):
    """
    做空借贷成本

    注意: 做空需要借入资产，需支付借贷利率
    热门做空标的借贷利率可能极高
    """
    daily_borrow_rate = borrow_rate_series / 365
    total_borrow_cost = short_value_usd * daily_borrow_rate.sum()
    return total_borrow_cost
```

注意：
- 借贷利率通常年化 5%-30%，热门做空标的可能超过 50%
- 交易所的借贷池深度有限，大规模做空时利率飙升
- 回测中应考虑借贷利率与做空规模的非线性关系

---

## 3. 滑点建模

### 3.1 滑点模型分层

横截面策略通常涉及大量标的，滑点建模直接影响回测可信度。

```python
def slippage_model(order_size_usd, avg_daily_volume_usd, mid_price, bid_ask_spread,
                   model_type='square_root'):
    """
    滑点模型

    参数:
        order_size_usd: 订单金额（USD）
        avg_daily_volume_usd: 日均成交量（USD）
        mid_price: 中间价
        bid_ask_spread: 买卖价差
        model_type: 滑点模型类型
    """
    participation_rate = order_size_usd / avg_daily_volume_usd

    if model_type == 'linear':
        # 线性模型：最简单，偏保守
        slippage_bps = participation_rate * 100  # 基点

    elif model_type == 'square_root':
        # 平方根模型：学术界和业界广泛使用
        # Kyle (1985) 和 Torre & Ferrari (1999) 的经典框架
        slippage_bps = 0.1 * np.sqrt(participation_rate) * 100

    elif model_type == 'nonlinear':
        # 非线性冲击模型：考虑流动性深度
        # 大额订单的边际冲击递增
        slippage_bps = 0.05 * (participation_rate ** 0.6) * 100
    else:
        raise ValueError(f"未知滑点模型: {model_type}")

    # 加上买卖价差的一半作为最低滑点
    half_spread_bps = (bid_ask_spread / mid_price) * 10000 / 2
    total_slippage_bps = max(slippage_bps, half_spread_bps)

    return total_slippage_bps / 10000  # 转换为小数
```

### 3.2 四种滑点模型对比

| 滑点模型 | 核心假设 | 适用场景 | 保守程度 |
|----------|----------|----------|----------|
| 固定滑点 | slippage = k（如 5bps） | 快速估算、早期验证 | 低 |
| 线性模型 | slippage = k * participation_rate | 小额订单（参与率 < 1%） | 中 |
| 平方根模型 | slippage = k * sqrt(participation_rate) | 中等规模（1%-5%） | 中高 |
| 非线性模型 | slippage = k * participation_rate^0.6 | 大额订单（> 5%） | 高 |

注意：
- 回测默认使用平方根模型，这是业界标准
- 对比不同模型的结果差异：如果差异巨大，说明策略对滑点敏感
- 对容量瓶颈标的使用非线性模型做压力测试

### 3.3 订单簿深度影响

```python
def orderbook_impact_estimate(order_size_usd, depth_data):
    """
    基于订单簿深度估算冲击成本

    参数:
        order_size_usd: 订单金额
        depth_data: DataFrame，列包含 price_level, bid_volume, ask_volume

    返回:
        estimated_impact: 估算的冲击成本（bps）
    """
    # 累计深度
    depth_data = depth_data.sort_values('price_level')
    depth_data['cumulative_ask_volume'] = depth_data['ask_volume'].cumsum()

    # 找到订单能吃到的深度
    filled_mask = depth_data['cumulative_ask_volume'] * depth_data['price_level'] < order_size_usd
    if filled_mask.sum() == 0:
        # 订单超过了订单簿可见深度
        return 50.0  # 保守估计 50bps 冲击

    # 计算加权平均成交价偏离
    last_filled_price = depth_data.loc[filled_mask, 'price_level'].iloc[-1]
    mid_price = depth_data['price_level'].median()
    impact_bps = abs(last_filled_price - mid_price) / mid_price * 10000

    return impact_bps
```

### 3.4 滑点对收益的敏感性分析

```python
def slippage_sensitivity(gross_return, turnover_annual, slippage_scenarios):
    """
    滑点敏感性分析：不同滑点假设下的净收益

    参数:
        gross_return: 毛收益（年化）
        turnover_annual: 年化换手率
        slippage_scenarios: 滑点假设列表（bps）

    返回:
        sensitivity_df: 各滑点假设下的净收益
    """
    results = []
    for slip_bps in slippage_scenarios:
        slip_cost = turnover_annual * slip_bps / 10000
        net_return = gross_return - slip_cost
        results.append({
            'slippage_bps': slip_bps,
            'annual_slip_cost': slip_cost * 100,
            'net_return_pct': net_return * 100,
            'return_reduction_pct': (slip_cost / gross_return * 100)
                                 if gross_return > 0 else float('inf')
        })

    return pd.DataFrame(results)
```

Formal Gate 要求：

```yaml
formal_gate_slippage:
  # 使用平方根模型作为默认
  default_model: "square_root"

  # 极端滑点（50bps）下净收益仍为正
  worst_case_positive: true

  # 滑点成本不超过毛收益的 40%
  slippage_to_gross_ratio: "< 40%"
```

---

## 4. 风险约束

### 4.1 单标的仓位上限

横截面策略需要防止过度集中于单一标的，否则回撤会远超可控范围。

```python
def apply_position_limit(target_weights, max_single_weight=0.10,
                          max_sector_weight=0.25):
    """
    施加仓位约束

    参数:
        target_weights: Series, 目标权重
        max_single_weight: 单标的最大权重
        max_sector_weight: 单行业最大权重

    返回:
        constrained_weights: 约束后的权重
    """
    weights = target_weights.copy()

    # 约束 1：单标的仓位上限
    weights = weights.clip(upper=max_single_weight)

    # 约束 2：权重归一化（多头部分和空头部分分别归一化）
    long_mask = weights > 0
    short_mask = weights < 0

    if long_mask.sum() > 0:
        weights[long_mask] = weights[long_mask] / weights[long_mask].sum()
    if short_mask.sum() > 0:
        weights[short_mask] = weights[short_mask] / weights[short_mask].abs().sum()

    return weights
```

注意：
- 单标的权重通常不超过 5%-10%，具体取决于标的数量
- 小市值标的的流动性可能不足以支撑大仓位
- 权重约束会降低策略收益，但这是真实约束，必须纳入

### 4.2 行业暴露限制

横截面因子可能对某些行业有系统性暴露，比如价值因子偏好金融、厌恶科技。

```python
def check_sector_exposure(weights, sector_map, max_sector_exposure=0.30):
    """
    检查并约束行业暴露

    参数:
        weights: Series, 标的权重
        sector_map: dict, 标的到行业的映射
        max_sector_exposure: 单行业最大暴露

    返回:
        sector_exposure: 各行业暴露
        violations: 违规行业列表
    """
    exposure = {}
    for symbol, weight in weights.items():
        sector = sector_map.get(symbol, 'Unknown')
        exposure[sector] = exposure.get(sector, 0) + abs(weight)

    # 多头侧归一化后检查
    total_long = weights[weights > 0].sum()
    if total_long > 0:
        exposure_pct = {s: e / total_long for s, e in exposure.items()}
    else:
        exposure_pct = exposure

    violations = [s for s, e in exposure_pct.items()
                  if e > max_sector_exposure]

    return exposure_pct, violations
```

### 4.3 Beta 中性约束

横截面策略通常要求市场中性，控制系统性风险暴露。

```python
def apply_beta_neutralization(weights, betas, target_beta=0.0):
    """
    Beta 中性化调整

    参数:
        weights: Series, 目标权重
        betas: Series, 各标的 Beta
        target_beta: 目标组合 Beta（通常为 0）

    返回:
        adjusted_weights: Beta 中性化后的权重
    """
    # 当前组合 Beta
    portfolio_beta = (weights * betas).sum()

    # 需要调整的量
    beta_adjustment = portfolio_beta - target_beta

    # 使用最小方差方法调整权重
    # 简化版本：按 Beta 比例调整
    adjusted_weights = weights.copy()
    for symbol in weights.index:
        adjusted_weights[symbol] -= (
            beta_adjustment * weights[symbol] * betas[symbol]
            / (weights * betas ** 2).sum()
        )

    return adjusted_weights
```

Formal Gate 要求：

```yaml
formal_gate_risk_constraints:
  # 必须施加单标的仓位上限
  position_limit_applied: true

  # 必须检查行业暴露
  sector_check_completed: true

  # 组合 Beta 绝对值 < 0.1
  portfolio_beta_abs: "< 0.1"

  # 最大回撤 < 25%
  max_drawdown: ">= -25%"
```

---

## 5. 策略容量评估

### 5.1 容量定义与重要性

**策略容量** 是在不显著影响收益的前提下所能管理的最大资金规模。横截面策略的容量受限于最不流动的那批标的。

### 5.2 容量估算方法

```python
def estimate_strategy_capacity(universe, daily_volumes, target_participation_rate=0.01,
                                target_weights=None):
    """
    策略容量估算

    参数:
        universe: 标的列表
        daily_volumes: DataFrame, 各标的日均成交量
        target_participation_rate: 目标市场参与率（默认 1%）
        target_weights: 目标权重（可选）

    返回:
        capacity_usd: 策略容量估算（USD）
        bottleneck_assets: 容量瓶颈标的
    """
    asset_capacities = {}

    for symbol in universe:
        if symbol in daily_volumes.columns:
            avg_vol = daily_volumes[symbol].mean()
            asset_cap = avg_vol * target_participation_rate * 252
            asset_capacities[symbol] = asset_cap

    # 如果有权重，按权重加权估算
    if target_weights is not None:
        # 容量受限于：资产容量 / 该资产在组合中的权重
        capacity_per_asset = {}
        for symbol, cap in asset_capacities.items():
            w = abs(target_weights.get(symbol, 0))
            if w > 0:
                capacity_per_asset[symbol] = cap / w

        overall_capacity = min(capacity_per_asset.values())
        bottleneck = min(capacity_per_asset, key=capacity_per_asset.get)
    else:
        # 等权假设
        overall_capacity = min(asset_capacities.values()) * len(asset_capacities)
        bottleneck = min(asset_capacities, key=asset_capacities.get)

    return overall_capacity, bottleneck
```

### 5.3 AUM 敏感性分析

```python
def aum_sensitivity_analysis(base_returns, turnover, slippage_model_fn,
                              aum_levels=None):
    """
    AUM 敏感性分析：不同资金规模下的策略表现

    参数:
        base_returns: 基准收益序列（不考虑冲击）
        turnover: 年化换手率
        slippage_model_fn: 滑点模型函数
        aum_levels: 资金规模列表
    """
    if aum_levels is None:
        aum_levels = [100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]

    results = []
    for aum in aum_levels:
        # 随 AUM 增大，滑点增大
        slip_bps = slippage_model_fn(aum)
        annual_slip_cost = turnover * slip_bps / 10000

        gross_annual = base_returns.mean() * 252
        net_annual = gross_annual - annual_slip_cost
        net_annual_vol = base_returns.std() * np.sqrt(252)
        net_sharpe = net_annual / net_annual_vol if net_annual_vol > 0 else 0

        results.append({
            'AUM': aum,
            'slippage_bps': slip_bps,
            'annual_cost': annual_slip_cost * 100,
            'gross_return': gross_annual * 100,
            'net_return': net_annual * 100,
            'return_decay': (annual_slip_cost / gross_annual * 100)
                           if gross_annual > 0 else float('inf'),
            'net_sharpe': net_sharpe,
        })

    return pd.DataFrame(results)
```

容量评估示例：

| AUM (USD) | 年化滑点 | 毛收益 | 净收益 | 收益衰减 | 净 Sharpe |
|-----------|----------|--------|--------|----------|-----------|
| 10 万 | 0.5% | 18% | 16.8% | 7% | 1.6 |
| 50 万 | 1.8% | 18% | 13.4% | 26% | 1.2 |
| 100 万 | 3.2% | 18% | 9.2% | 49% | 0.8 |
| 500 万 | 8.5% | 18% | -2.3% | 113% | -0.2 |

该策略容量上限约为 30-50 万美元。

### 5.4 流动性天花板

```python
def liquidity_ceilings(universe, daily_volumes, min_volume_threshold_usd=1_000_000):
    """
    识别流动性天花板标的：日均成交量低于阈值的标的

    参数:
        min_volume_threshold_usd: 最低流动性门槛
    """
    low_liquidity = []
    for symbol in universe:
        if symbol in daily_volumes.columns:
            avg_vol = daily_volumes[symbol].mean()
            if avg_vol < min_volume_threshold_usd:
                low_liquidity.append({
                    'symbol': symbol,
                    'avg_daily_volume': avg_vol,
                    'capacity_at_1pct': avg_vol * 0.01
                })

    return sorted(low_liquidity, key=lambda x: x['avg_daily_volume'])
```

Formal Gate 要求：

```yaml
formal_gate_capacity:
  # 必须完成容量评估
  capacity_estimated: true

  # 净 Sharpe > 0.5 时的最大 AUM 为策略容量
  min_viable_sharpe: ">= 0.5"

  # 容量瓶颈已识别并记录
  bottleneck_identified: true

  # 流动性天花板标的已标记
  liquidity_ceilings_flagged: true
```

---

## 6. 执行可实现性

### 6.1 再平衡窗口

再平衡频率直接影响成本和收益的平衡。

| 再平衡频率 | 优势 | 劣势 | 适用场景 |
|-----------|------|------|----------|
| 日频 | 信号跟踪最紧密 | 换手率极高，成本大 | 高频因子、大容量策略 |
| 周频 | 成本可控，信号时效性较好 | 每周仍有较高换手 | 多数横截面因子的默认选择 |
| 月频 | 成本最低 | 信号滞后严重 | 低频因子、大资金 |
| 信号触发制 | 只在信号变化显著时调仓 | 逻辑复杂，需额外参数 | 需要精细控制成本的场景 |

```python
def rebalance_schedule(factor_values, rebalance_freq='W',
                       signal_threshold=0.3):
    """
    生成再平衡时间表

    参数:
        factor_values: DataFrame, 因子值
        rebalance_freq: 再平衡频率（D/W/M）
        signal_threshold: 信号变化阈值（用于信号触发制）
    """
    if rebalance_freq == 'D':
        # 每日再平衡
        rebalance_dates = factor_values.index
    elif rebalance_freq == 'W':
        # 每周再平衡（每周一）
        rebalance_dates = factor_values.resample('W-MON').first().index
    elif rebalance_freq == 'M':
        # 每月再平衡（每月第一个交易日）
        rebalance_dates = factor_values.resample('MS').first().index
    elif rebalance_freq == 'signal':
        # 信号触发：因子排序变化超过阈值时再平衡
        rank_diff = factor_values.rank(axis=1).diff().abs()
        significant_change = (rank_diff > signal_threshold).any(axis=1)
        rebalance_dates = significant_change[significant_change].index
    else:
        raise ValueError(f"未知的再平衡频率: {rebalance_freq}")

    return rebalance_dates
```

### 6.2 限价单 vs 市价单

| 维度 | 限价单 | 市价单 |
|------|--------|--------|
| 成本 | 较低（约 30-50% 便宜） | 较高（Taker 费率 + 更大滑点） |
| 成交确定性 | 不保证成交（约 70-85%） | 保证成交 |
| 执行时间 | 可能延迟数分钟到数小时 | 即时成交 |
| 适用场景 | 再平衡窗口充裕、流动性好 | 需要快速执行的紧急调仓 |
| 回测假设 | 假设成交率，计算跟踪误差 | 假设 Taker 费率 + 固定滑点 |

注意：
- 默认使用限价单（再平衡窗口充裕）
- 回测中假设限价单成交率为 75%-85%
- 未成交部分次日继续挂单或转为市价单

### 6.3 分批执行

大规模调仓必须分批执行以降低市场冲击。

```python
def execution_plan(target_trades, max_single_trade_usd=50_000,
                   n_batches=None, execution_window_hours=4):
    """
    生成分批执行计划

    参数:
        target_trades: DataFrame, 目标交易列表（symbol, side, size_usd）
        max_single_trade_usd: 单笔最大交易额
        n_batches: 分批数量（若指定则忽略 max_single_trade_usd）
        execution_window_hours: 执行时间窗口

    返回:
        batches: 分批执行计划
    """
    if n_batches is None:
        # 自动计算分批数
        max_trade = target_trades['size_usd'].abs().max()
        n_batches = max(1, int(np.ceil(max_trade / max_single_trade_usd)))

    # TWAP（时间加权平均价格）分批
    batch_size = len(target_trades) // n_batches + 1
    batches = []

    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(target_trades))
        batch = target_trades.iloc[start:end].copy()
        batch['batch_id'] = i
        batch['execution_time'] = (
            pd.Timestamp.now() + pd.Timedelta(
                hours=i * execution_window_hours / n_batches
            )
        )
        batches.append(batch)

    return batches
```

### 6.4 执行质量回测评估

```python
def execution_quality_metrics(trades, market_prices):
    """
    评估执行质量

    参数:
        trades: DataFrame, 实际交易记录
        market_prices: DataFrame, 市场价格序列

    返回:
        metrics: 执行质量指标
    """
    # 实现差（Implementation Shortfall）
    # 信号发出时价格 vs 实际成交价格
    implementation_shortfall = []

    for _, trade in trades.iterrows():
        decision_price = market_prices.loc[trade['signal_time'], trade['symbol']]
        execution_price = trade['execution_price']

        if trade['side'] == 'buy':
            shortfall = (execution_price - decision_price) / decision_price
        else:
            shortfall = (decision_price - execution_price) / decision_price

        implementation_shortfall.append(shortfall)

    avg_shortfall = np.mean(implementation_shortfall)
    vs_vwap = ...  # 与 VWAP 的对比

    return {
        'avg_implementation_shortfall_bps': avg_shortfall * 10000,
        'fill_rate': len(trades[trades['filled'] == True]) / len(trades),
        'avg_execution_time_seconds': (
            trades['execution_time'] - trades['signal_time']
        ).dt.total_seconds().mean(),
    }
```

---

## 7. Formal Gate 总汇

### 7.1 门禁检查清单

```yaml
formal_gate_backtest_stage:

  FG-POST_COST_RETURN:
    requirement: "扣除全部成本后净收益 > 0"
    evidence: "backtest_results.parquet"
    status: "PASS / FAIL"

  FG-COST_RATIO:
    requirement: "总成本占毛收益比例 < 50%"
    evidence: "cost_breakdown.json"
    status: "PASS / FAIL"

  FG-SLIPPAGE_ROBUST:
    requirement: "极端滑点（50bps）下净收益仍为正"
    evidence: "slippage_sensitivity.csv"
    status: "PASS / FAIL"

  FG-POSITION_LIMIT:
    requirement: "已施加单标的仓位上限"
    evidence: "risk_constraints.json"
    status: "PASS / FAIL"

  FG-BETA_NEUTRAL:
    requirement: "组合 Beta 绝对值 < 0.1"
    evidence: "risk_constraints.json"
    status: "PASS / FAIL"

  FG-CAPACITY:
    requirement: "策略容量已评估，净 Sharpe > 0.5 时的 AUM 明确"
    evidence: "capacity_analysis.json"
    status: "PASS / FAIL"

  FG-EXECUTABLE:
    requirement: "执行方案可行（再平衡窗口、订单类型、分批方案）"
    evidence: "execution_plan.json"
    status: "PASS / FAIL"

  FG-DRAWDOWN:
    requirement: "最大回撤 >= -25%"
    evidence: "risk_metrics.json"
    status: "PASS / FAIL"
```

### 7.2 决策状态

| 状态 | 条件 | 后续动作 |
|------|------|----------|
| **PASS** | 所有 FG 通过 | 进入 Holdout 阶段 |
| **CONDITIONAL PASS** | 核心 FG 通过，容量或执行有瑕疵 | 记录约束条件，进入 Holdout |
| **RETRY** | 成本模型或执行逻辑有误 | 修正后重新运行，不修改因子结构 |
| **NO_GO** | 扣费后亏损或容量极小 | 终止研究线，归档 |

---

## 8. 常见错误与防范

### 8.1 错误清单

| 错误类型 | 描述 | 后果 | 防范措施 |
|----------|------|------|----------|
| **忽略资金费率** | 回测不计入永续合约资金费 | 净收益被高估 5-30% | 必须使用历史资金费率数据 |
| **假设零滑点** | 使用中间价作为成交价 | 严重高估收益 | 至少使用平方根滑点模型 |
| **忽略借贷成本** | 做空不计入借贷利率 | 空头收益被高估 | 使用实际借贷利率历史数据 |
| **不施加仓位约束** | 允许单标的过度集中 | 回撤远超实际可控范围 | 强制施加仓位上限 |
| **忽略 Beta 暴露** | 多空不配比导致市场敞口 | 回撤被系统性风险放大 | 强制 Beta 中性化 |
| **不评估容量** | 小资金回测直接推算大资金 | 实盘规模放大后策略失效 | 必须做 AUM 敏感性分析 |
| **假设 100% 限价单成交** | 限价单全部成交 | 低估跟踪误差 | 假设 75-85% 成交率 |
| **同日信号同日交易** | 信号计算和交易在同一天完成 | 前视偏差 | T 日信号、T+1 日执行 |

### 8.2 实际案例

案例：动量因子扣费后策略失效

某团队构建了一个 20 日动量因子，Test 阶段多空 Sharpe = 2.1，进入 Backtest 后：

```
毛收益: 25% 年化
交易手续费: -4.2%（周频再平衡，换手率高）
资金费率: -3.8%（做多热门标的费率高）
滑点成本: -5.1%（部分标的流动性不足）
借贷成本: -2.3%（做空冷门标的借贷利率高）
──────────────────────────
净收益: 9.6% 年化
净 Sharpe: 0.65
```

处理方式：
- 虽然净收益大幅下降，但仍为正且 Sharpe > 0.5
- 判定 CONDITIONAL PASS
- 在 Backtest 中优化再平衡频率：周频改为双周频
- 优化后净收益提升至 13.2%，Sharpe = 0.92
- 同时标记了策略容量上限为 30 万美元

---

## 9. 输出 Artifact

### 9.1 机器可读产物

```yaml
backtest_results.parquet:
  用途: 回测核心结果
  粒度: 日频
  字段: date, portfolio_return, gross_return, cost_breakdown, drawdown
  消费者: Holdout 参考

cost_breakdown.json:
  用途: 成本分解
  内容: fee, funding, borrow, slippage, total
  消费者: 成本分析

risk_metrics.json:
  用途: 风险指标汇总
  内容: sharpe, max_dd, calmar, beta, sector_exposure
  消费者: Formal Gate

capacity_analysis.json:
  用途: 容量评估
  内容: aum_sensitivity, bottleneck, liquidity_ceilings
  消费者: 资金规模决策

execution_plan.json:
  用途: 执行方案
  内容: rebalance_freq, order_type, batch_plan
  消费者: 实盘执行参考
```

### 9.2 人类可读产物

```yaml
backtest_report.md:
  用途: Backtest 阶段完整报告
  消费者: 团队评审、Holdout 参考

gate_decision.md:
  用途: 门禁决策文档
  必需字段: stage, status, decision_basis, frozen_scope, next_steps

cost_analysis_report.md:
  用途: 成本分析专项报告
  内容: 各成本项的占比、敏感性分析
  消费者: 成本优化决策
```

---

## 10. 与 Holdout 阶段的交接

### 10.1 冻结传递

```yaml
frozen_spec_handover:
  from_stage: "backtest"
  to_stage: "holdout"

  frozen_items:
    trading_rules: "完整的交易规则（入场、出场、仓位、风控）"
    cost_model: "成本模型（手续费、资金费率、借贷、滑点）"
    risk_constraints: "风险约束（仓位上限、行业暴露、Beta 中性）"
    execution_plan: "执行方案（再平衡频率、订单类型、分批方案）"
    capacity_limit: "策略容量上限"

  holdout_must_use:
    - "只能使用 Backtest 冻结的成本模型"
    - "必须施加相同的风险约束"
    - "不能修改执行方案"
    - "不能调整再平衡频率"
```

### 10.2 Holdout 阶段需要特别关注的事项

1. 成本一致性：Holdout 期间的市场结构可能导致成本模型参数变化（如波动率升高导致滑点增大）
2. 流动性变化：Holdout 期间可能有新的低流动性标的加入 Universe
3. 执行偏差：Holdout 应使用与 Backtest 相同的执行假设，不做任何优化

---

**文档版本**: v1.0
**最后更新**: 2026-04-02
**适用领域**: 横截面因子研究
