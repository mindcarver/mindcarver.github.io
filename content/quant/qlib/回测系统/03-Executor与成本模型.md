# Executor 与成本模型

## 1. Executor 机制

### 1.1 Executor 定义

**Executor**：负责执行交易信号的组件，模拟真实交易环境

```python
class Executor:
    """
    交易执行器
    
    功能：
    1. 接收交易信号
    2. 执行交易
    3. 计算成本
    4. 更新持仓
    5. 记录绩效
    """
    
    def __init__(self, config):
        self.account = config['initial_account']
        self.positions = {}
        self.cash = self.account
        self.trade_history = []
        
    def execute(self, signals, prices):
        """
        执行交易信号
        
        参数:
            signals: 交易信号
            prices: 价格数据
        
        返回:
            trades: 执行的交易
        """
        trades = []
        
        for stock, target_weight in signals.items():
            if target_weight == 0:
                # 卖出
                trade = self.sell(stock, prices[stock])
            else:
                # 买入
                trade = self.buy(stock, target_weight, prices[stock])
            
            if trade:
                trades.append(trade)
        
        return trades
    
    def buy(self, stock, target_weight, price):
        """
        买入股票
        
        参数:
            stock: 股票代码
            target_weight: 目标权重
            price: 价格
        
        返回:
            trade: 交易记录
        """
        # 计算目标持仓价值
        target_value = self.account * target_weight
        
        # 计算需要买入的股数
        shares = int(target_value / price)
        
        if shares <= 0:
            return None
        
        # 计算交易金额
        trade_value = shares * price
        
        # 检查现金是否足够
        if trade_value > self.cash:
            shares = int(self.cash / price)
            trade_value = shares * price
        
        # 更新持仓和现金
        if stock not in self.positions:
            self.positions[stock] = 0
        
        self.positions[stock] += shares
        self.cash -= trade_value
        
        # 记录交易
        trade = {
            'stock': stock,
            'action': 'buy',
            'shares': shares,
            'price': price,
            'value': trade_value
        }
        self.trade_history.append(trade)
        
        return trade
    
    def sell(self, stock, price):
        """
        卖出股票
        
        参数:
            stock: 股票代码
            price: 价格
        
        返回:
            trade: 交易记录
        """
        # 检查持仓
        if stock not in self.positions or self.positions[stock] <= 0:
            return None
        
        # 全部卖出
        shares = self.positions[stock]
        trade_value = shares * price
        
        # 更新持仓和现金
        self.positions[stock] = 0
        self.cash += trade_value
        
        # 记录交易
        trade = {
            'stock': stock,
            'action': 'sell',
            'shares': shares,
            'price': price,
            'value': trade_value
        }
        self.trade_history.append(trade)
        
        return trade
```

### 1.2 Qlib Executor 架构

```python
Executor {
    execute(signals) → trades           # 执行交易信号
    apply_costs(trades) → adj_returns  # 应用交易成本
    generate_portfolio_metrics() → metrics  # 生成组合指标
}
```

### 1.3 Executor 功能

**1. 信号执行**
- 接收交易信号
- 模拟订单提交
- 计算成交价格

**2. 成本应用**
- 计算手续费
- 计算滑点成本
- 计算市场冲击

**3. 持仓管理**
- 更新当前持仓
- 计算持仓价值
- 管理现金余额

**4. 绩效记录**
- 记录每笔交易
- 计算实时绩效
- 生成绩效报告

---

## 2. 交易成本模型

### 2.1 成本构成

**1. 手续费**
- 按交易金额的固定比例收取
- 买入费率和卖出费率可能不同
- 通常有最低费用限制

**2. 滑点**
- 实际成交价格与理想价格的差异
- 与交易规模和流动性相关
- 通常按价格百分比计算

**3. 市场冲击**
- 大额交易对价格的冲击
- 与交易金额和市场流动性相关
- 通常按交易量的平方根计算

### 2.2 手续费计算

**计算公式**：
```
手续费 = 交易金额 × 费率
手续费 = max(手续费, 最低费用)
```

**Python 实现**：
```python
def calculate_commission(trade_value, commission_rate, min_commission=5):
    """
    计算手续费
    
    参数:
        trade_value: 交易金额
        commission_rate: 手续费率
        min_commission: 最低手续费
    
    返回:
        commission: 手续费
    """
    commission = trade_value * commission_rate
    commission = max(commission, min_commission)
    
    return commission

# 示例
trade_value = 100000  # 交易10万元
commission_rate = 0.0003  # 万分之三
min_commission = 5  # 最低5元

commission = calculate_commission(trade_value, commission_rate, min_commission)
print(f"手续费 = {commission:.2f} 元")
```

### 2.3 滑点计算

**计算公式**：
```
滑点成本 = 交易金额 × 滑点率
成交价格 = 理想价格 × (1 ± 滑点率)
```

**Python 实现**：
```python
def calculate_slippage(trade_value, slippage_rate, is_buy=True):
    """
    计算滑点成本
    
    参数:
        trade_value: 交易金额
        slippage_rate: 滑点率
        is_buy: 是否买入
    
    返回:
        slippage_cost: 滑点成本
        execution_price: 执行价格
    """
    # 买入时滑点增加成本，卖出时滑点减少收益
    if is_buy:
        slippage_cost = trade_value * slippage_rate
        execution_price = 1 + slippage_rate
    else:
        slippage_cost = trade_value * slippage_rate
        execution_price = 1 - slippage_rate
    
    return slippage_cost, execution_price

# 示例
trade_value = 100000  # 交易10万元
slippage_rate = 0.001  # 千分之一

# 买入
slippage_buy, price_buy = calculate_slippage(trade_value, slippage_rate, is_buy=True)
print(f"买入滑点 = {slippage_buy:.2f} 元")

# 卖出
slippage_sell, price_sell = calculate_slippage(trade_value, slippage_rate, is_buy=False)
print(f"卖出滑点 = {slippage_sell:.2f} 元")
```

### 2.4 市场冲击计算

**计算公式**：
```
市场冲击 = k × sqrt(交易量 / 日均成交量)
```

**Python 实现**：
```python
def calculate_market_impact(trade_value, avg_daily_value, impact_factor=0.1):
    """
    计算市场冲击
    
    参数:
        trade_value: 交易金额
        avg_daily_value: 日均交易金额
        impact_factor: 冲击系数
    
    返回:
        market_impact: 市场冲击成本
    """
    if avg_daily_value == 0:
        return 0
    
    # 计算交易比例
    trade_ratio = trade_value / avg_daily_value
    
    # 计算市场冲击
    market_impact = trade_value * impact_factor * np.sqrt(trade_ratio)
    
    return market_impact

# 示例
trade_value = 1000000  # 交易100万元
avg_daily_value = 50000000  # 日均交易5000万元

market_impact = calculate_market_impact(trade_value, avg_daily_value)
print(f"市场冲击 = {market_impact:.2f} 元")
```

### 2.5 总成本模型

```python
def calculate_total_cost(trade_value, commission_rate, slippage_rate, 
                           min_commission=5, avg_daily_value=None, impact_factor=0.1):
    """
    计算总成本
    
    参数:
        trade_value: 交易金额
        commission_rate: 手续费率
        slippage_rate: 滑点率
        min_commission: 最低手续费
        avg_daily_value: 日均交易金额
        impact_factor: 冲击系数
    
    返回:
        total_cost: 总成本
        cost_breakdown: 成本明细
    """
    # 1. 手续费
    commission = calculate_commission(trade_value, commission_rate, min_commission)
    
    # 2. 滑点
    slippage = trade_value * slippage_rate
    
    # 3. 市场冲击
    if avg_daily_value and impact_factor:
        market_impact = calculate_market_impact(trade_value, avg_daily_value, impact_factor)
    else:
        market_impact = 0
    
    # 4. 总成本
    total_cost = commission + slippage + market_impact
    
    # 5. 成本明细
    cost_breakdown = {
        'commission': commission,
        'slippage': slippage,
        'market_impact': market_impact,
        'total': total_cost
    }
    
    return total_cost, cost_breakdown

# 示例
trade_value = 100000  # 交易10万元
commission_rate = 0.0003  # 万分之三
slippage_rate = 0.001  # 千分之一
avg_daily_value = 5000000  # 日均交易500万元
impact_factor = 0.1

total_cost, breakdown = calculate_total_cost(
    trade_value, commission_rate, slippage_rate,
    min_commission=5,
    avg_daily_value=avg_daily_value,
    impact_factor=impact_factor
)

print("成本明细:")
for key, value in breakdown.items():
    print(f"  {key}: {value:.2f} 元")
print(f"  总成本: {total_cost:.2f} 元")
print(f"  成本率: {total_cost / trade_value:.4f}")
```

---

## 3. 成本敏感性分析

### 3.1 成本参数范围

**手续费率**：
- 低：0.0001（万分之一）
- 中：0.001（千分之一）
- 高：0.003（千分之三）

**滑点率**：
- 低：0.0005（万分之一）
- 中：0.001（千分之一）
- 高：0.005（千分之五）

### 3.2 成本影响分析

**对收益率的影响**：
```
净值收益 = 原始收益 - 总成本
```

**对换手率的影响**：
```
换手成本 = 换手率 × 平均资产 × 成本率
```

**对策略选择的影响**：
- 高换手率策略对成本更敏感
- 低成本环境适合高频策略
- 高成本环境适合低换手率策略

### 3.3 成本敏感性分析

```python
def cost_sensitivity_analysis(trade_value, commission_rates, slippage_rates):
    """
    成本敏感性分析
    
    参数:
        trade_value: 交易金额
        commission_rates: 手续费率列表
        slippage_rates: 滑点率列表
    
    返回:
        sensitivity: 敏感性分析结果
    """
    sensitivity = []
    
    for comm_rate in commission_rates:
        for slip_rate in slippage_rates:
            total_cost, _ = calculate_total_cost(
                trade_value, comm_rate, slip_rate
            )
            
            sensitivity.append({
                'commission_rate': comm_rate,
                'slippage_rate': slip_rate,
                'total_cost': total_cost,
                'cost_rate': total_cost / trade_value
            })
    
    return pd.DataFrame(sensitivity)

# 示例
commission_rates = [0.0001, 0.0003, 0.001, 0.003]
slippage_rates = [0.0005, 0.001, 0.002, 0.005]

sensitivity_df = cost_sensitivity_analysis(100000, commission_rates, slippage_rates)

print("成本敏感性分析:")
print(sensitivity_df)
```

### 3.4 换手率与成本的关系

```python
def turnover_cost_analysis(avg_value, turnover_rates, cost_rate=0.003):
    """
    换手率与成本关系分析
    
    参数:
        avg_value: 平均资产
        turnover_rates: 换手率列表
        cost_rate: 成本率
    
    返回:
        analysis: 分析结果
    """
    analysis = []
    
    for turnover in turnover_rates:
        # 年化换手成本
        annual_cost = avg_value * turnover * cost_rate
        
        analysis.append({
            'turnover': turnover,
            'annual_cost': annual_cost,
            'cost_ratio': annual_cost / avg_value
        })
    
    return pd.DataFrame(analysis)

# 示例
avg_value = 1000000  # 平均资产100万
turnover_rates = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]  # 换手率

cost_df = turnover_cost_analysis(avg_value, turnover_rates, cost_rate=0.003)

print("换手率与成本关系:")
print(cost_df)
```

---

## 4. 实践建议

### 4.1 成本控制策略

**1. 降低换手率**
- 优化调仓频率
- 增加持仓周期
- 使用止损止盈

**2. 降低交易成本**
- 选择低佣金券商
- 优化下单时间
- 分批建仓

**3. 选择流动性好的股票**
- 避免小市值股票
- 关注日均成交量
- 避免停牌股票

### 4.2 成本参数配置

**Qlib 配置示例**：
```python
executor_config = {
    'class': 'SimulatorExecutor',
    'module_path': 'qlib.backtest.executor',
    'kwargs': {
        'time_per_step': 'day',
        'generate_portfolio_metrics': True
    }
}

exchange_config = {
    'freq': 'day',
    'limit_threshold': 0.095,
    'deal_price': 'close',
    'open_cost': 0.0005,   # 买入费率万分之五
    'close_cost': 0.0015,  # 卖出费率千分之1.5
    'min_cost': 5           # 最低手续费5元
}
```

### 4.3 成本敏感性测试

**测试流程**：
1. 使用低成本参数回测
2. 逐步增加成本参数
3. 观察策略表现变化
4. 确定策略的成本敏感性

**Python 实现**：
```python
def test_cost_sensitivity(strategy, returns, base_config, 
                         cost_multipliers=[1.0, 1.5, 2.0, 3.0]):
    """
    成本敏感性测试
    
    参数:
        strategy: 交易策略
        returns: 收益率数据
        base_config: 基础配置
        cost_multipliers: 成本倍数列表
    
    返回:
        results: 测试结果
    """
    results = []
    
    for multiplier in cost_multipliers:
        # 调整成本参数
        config = base_config.copy()
        config['open_cost'] *= multiplier
        config['close_cost'] *= multiplier
        
        # 回测
        performance = backtest(strategy, returns, config)
        
        results.append({
            'cost_multiplier': multiplier,
            'return': performance['return'],
            'sharpe': performance['sharpe'],
            'max_drawdown': performance['max_drawdown']
        })
    
    return pd.DataFrame(results)

# 示例
base_config = {
    'open_cost': 0.0005,
    'close_cost': 0.0015,
    'min_cost': 5
}

results = test_cost_sensitivity(strategy, returns, base_config)
print("成本敏感性测试:")
print(results)
```

---

## 总结

交易成本是量化投资中不可忽视的因素：

1. **成本构成**：手续费、滑点、市场冲击
2. **成本计算**：建立准确的成本模型
3. **成本敏感性**：分析成本对策略的影响
4. **成本控制**：优化策略降低成本

**建议**：
- 务必做成本敏感性分析
- 选择合理的成本参数
- 优化策略降低换手率
- 选择流动性好的股票
