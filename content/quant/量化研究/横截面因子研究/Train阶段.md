# Train 阶段规范 — 横截面因子研究

## 目录

1. [阶段概述](#1-阶段概述)
2. [核心任务](#2-核心任务)
3. [多因子组合方法](#3-多因子组合方法)
4. [权重学习算法](#4-权重学习算法)
5. [参数固定策略](#5-参数固定策略)
6. [组合构造规则初版](#6-组合构造规则初版)
7. [Formal Gate 要求](#7-formal-gate-要求)
8. [常见错误和反模式](#8-常见错误和反模式)
9. [实际案例：加密货币多因子组合](#9-实际案例加密货币多因子组合)
10. [输出 Artifact 规范](#10-输出-artifact-规范)
11. [与下一阶段的交接标准](#11-与下一阶段的交接标准)

---

## 1. 阶段概述

### 1.1 阶段定义

Train 是横截面因子研究流程的第四阶段，位于 Signal Ready 之后、Test 之前。

**核心目标**：将多个标准化因子组合成**预测能力更强的复合因子**，并学习最优权重。

### 1.2 横截面因子研究的 Train 特点

与时间序列策略不同，横截面因子研究的 Train 阶段：

| 维度 | 时间序列策略 | 横截面因子研究 |
|------|-------------|---------------|
| 训练目标 | 预测单标的收益方向 | 预测横截面收益排序 |
| 样本构成 | 时序样本为主 | 横截面+时序混合样本 |
| 权重含义 | 信号权重 | 因子权重 |
| 输出结果 | 阈值、分位点 | 因子权重系数 |

### 1.3 在流程中的位置

```
Mandate → Data Ready → Signal Ready → Train → Test → Backtest → Holdout
                                    ↑
                                  当前阶段
```

**前置依赖**：
- Signal Ready：已提供标准化因子库、单因子诊断报告

**后续影响**：
- Test 阶段将验证复合因子的预测能力
- Backtest 将基于复合因子构建交易组合

---

## 2. 核心任务

### 2.1 任务清单

| 任务 | 输出 | 优先级 |
|------|------|--------|
| 多因子组合建模 | composite_factor_model.pkl | P0 |
| 因子权重学习 | factor_weights.json | P0 |
| 参数固定 | frozen_train_params.json | P0 |
| 组合构造规则 | portfolio_construction_rules.md | P0 |
| Train 期回测分析 | train_performance_report.json | P1 |

### 2.2 核心原则

1. **样本外完整性**：所有权重学习只在 Train 窗口内进行
2. **可复现性**：权重学习结果必须可精确复现
3. **鲁棒性优先**：选择稳定而非最优的权重方案

---

## 3. 多因子组合方法

### 3.1 因子组合分类

#### 方法1：简单等权组合

```python
def equal_weight_combination(df, factor_cols):
    """
    等权组合：最简单的多因子方法
    
    优点：简单、鲁棒、过拟合风险低
    缺点：忽略因子质量差异
    """
    # 确保所有因子已标准化
    normalized_factors = df[factor_cols].fillna(0)
    
    # 等权求和
    composite = normalized_factors.mean(axis=1)
    
    # 再次标准化
    composite = (composite - composite.mean()) / composite.std()
    
    return composite
```

#### 方法2：IC加权组合

```python
def ic_weighted_combination(df, factor_cols, ic_dict):
    """
    IC加权：按因子IC绝对值加权
    
    逻辑：IC越大的因子对预测贡献越大
    """
    # 提取IC值
    weights = np.array([abs(ic_dict[col]) for col in factor_cols])
    weights = weights / weights.sum()  # 归一化
    
    # 加权求和
    composite = df[factor_cols].fillna(0).dot(weights)
    
    # 标准化
    composite = (composite - composite.mean()) / composite.std()
    
    return composite, weights
```

#### 方法3：正交化组合

```python
def orthogonal_combination(df, factor_cols, ic_ranking):
    """
    正交化组合：按IC排序依次正交化
    
    逻辑：先放入最强因子，后续因子与前序正交
    优点：消除因子间相关性
    """
    df = df.copy()
    residuals = []
    weights = []
    
    for i, factor in enumerate(ic_ranking):
        if i == 0:
            # 第一个因子直接使用
            residual = df[factor]
            weight = 1.0
        else:
            # 对前序因子回归，取残差
            previous_factors = ic_ranking[:i]
            X = df[previous_factors].values
            y = df[factor].values
            
            # OLS回归
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            residual = y - model.predict(X)
            
            # 残差标准化
            residual = (residual - residual.mean()) / residual.std()
            weight = 0.5  # 残差因子权重减半
        
        residuals.append(residual)
        weights.append(weight)
    
    # 合成
    composite = sum(w * r for w, r in zip(weights, residuals))
    composite = (composite - composite.mean()) / composite.std()
    
    return composite, dict(zip(ic_ranking, weights))
```

#### 方法4：机器学习组合

```python
def ml_combination(X_train, y_train, X_test, model_type='lightgbm'):
    """
    机器学习组合：让模型学习因子权重
    
    优点：自动处理非线性、交互效应
    缺点：过拟合风险高，需要严格验证
    """
    if model_type == 'lightgbm':
        import lightgbm as lgb
        model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            n_jobs=-1
        )
    elif model_type == 'ridge':
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)
    
    # 训练
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 获取特征重要性（作为权重参考）
    if hasattr(model, 'feature_importances_'):
        importance = dict(zip(X_train.columns, model.feature_importances_))
    elif hasattr(model, 'coef_'):
        importance = dict(zip(X_train.columns, np.abs(model.coef_)))
    else:
        importance = None
    
    return y_pred, importance
```

### 3.2 组合方法选择指南

| 场景 | 推荐方法 | 理由 |
|------|---------|------|
| 因子数量少（<5） | IC加权 | 简单有效 |
| 因子相关性高 | 正交化组合 | 消除冗余 |
| 因子数量多（>10） | 机器学习+正则化 | 自动选择 |
| 稳定性优先 | 等权组合 | 鲁棒性最强 |
| 样本量充足 | 机器学习 | 充分利用数据 |

---

## 4. 权重学习算法

### 4.1 权重学习的目标函数

横截面因子权重学习的核心目标：**最大化复合因子的预测能力**

#### 目标函数1：最大化IR

$$\max_w \quad IR = \frac{E[IC_t]}{\sigma(IC_t)}$$

其中 $IC_t = \text{corr}(w^T F_t, R_{t+1})$

```python
def optimize_ir(train_data, factor_cols, return_col):
    """
    最大化IR的权重优化
    
    使用滚动窗口IC计算
    """
    from scipy.optimize import minimize
    
    def calculate_ir(weights, data):
        # 计算复合因子
        composite = data[factor_cols].dot(weights)
        composite = (composite - composite.mean()) / composite.std()
        
        # 滚动IC
        ic_series = []
        for timestamp in data['timestamp'].unique():
            cross_section = data[data['timestamp'] == timestamp]
            ic = cross_section[[composite.name, return_col]].corr(method='spearman').iloc[0, 1]
            ic_series.append(ic)
        
        ic_series = pd.Series(ic_series)
        return -ic_series.mean() / ic_series.std()  # 负号因为是最小化
    
    # 约束：权重和为1，非负
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    ]
    bounds = [(0, 1) for _ in factor_cols]
    
    # 初始值：等权
    x0 = np.ones(len(factor_cols)) / len(factor_cols)
    
    result = minimize(
        calculate_ir,
        x0,
        args=(train_data,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return result.x, -result.fun
```

#### 目标函数2：最大化多空收益

$$\max_w \quad E[R_{long} - R_{short}]$$

```python
def optimize_long_short_return(train_data, factor_cols, return_col, n_groups=5):
    """
    最大化多空收益的权重优化
    """
    from scipy.optimize import minimize
    
    def calculate_long_short_return(weights, data):
        # 计算复合因子
        composite = data[factor_cols].dot(weights)
        data = data.copy()
        data['composite'] = (composite - composite.mean()) / composite.std()
        
        # 分组
        def分组时刻(group):
            group['group'] = pd.qcut(
                group['composite'],
                q=n_groups,
                labels=range(n_groups),
                duplicates='drop'
            )
            return group
        
        data = data.groupby('timestamp').apply(分组时刻).droplevel(0)
        
        # 计算多空收益
        group_returns = data.groupby('group')[return_col].mean()
        long_short = group_rewards.iloc[-1] - group_returns.iloc[0]
        
        return -long_short  # 负号因为是最小化
    
    # 约束
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    ]
    bounds = [(0, 1) for _ in factor_cols]
    x0 = np.ones(len(factor_cols)) / len(factor_cols)
    
    result = minimize(
        calculate_long_short_return,
        x0,
        args=(train_data,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return result.x, -result.fun
```

### 4.2 交叉验证

横截面因子研究的特殊性：**时间序列交叉验证**

```python
def time_series_cv(data, factor_cols, return_col, n_splits=5):
    """
    时间序列交叉验证
    
    不能用K-Fold（会引入前视）
    必须使用滚动窗口或扩展窗口
    """
    from sklearn.model_selection import TimeSeriesSplit
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    cv_results = []
    
    for train_idx, val_idx in tscv.split(data):
        train_data = data.iloc[train_idx]
        val_data = data.iloc[val_idx]
        
        # 训练权重
        weights, _ = optimize_ir(train_data, factor_cols, return_col)
        
        # 验证
        val_composite = val_data[factor_cols].dot(weights)
        val_ic = val_data[['composite_temp', return_col]].corr(
            val_composite, 
            val_data[return_col],
            method='spearman'
        )
        
        cv_results.append({
            'weights': weights,
            'val_ic': val_ic
        })
    
    return cv_results
```

### 4.3 权重稳定性检验

```python
def weight_stability_analysis(weight_history, factor_cols):
    """
    权重稳定性分析
    
    检查权重是否随时间剧烈波动
    """
    import matplotlib.pyplot as plt
    
    weight_df = pd.DataFrame(weight_history, columns=factor_cols)
    
    # 计算权重标准差
    weight_std = weight_df.std()
    
    # 计算权重变异系数
    weight_cv = weight_df.std() / weight_df.mean()
    
    # 可视化
    plt.figure(figsize=(12, 6))
    weight_df.plot(kind='box')
    plt.title('Factor Weight Distribution')
    plt.ylabel('Weight')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return {
        'weight_std': weight_std.to_dict(),
        'weight_cv': weight_cv.to_dict(),
        'max_std_factor': weight_std.idxmax(),
        'most_stable_factor': weight_std.idxmin()
    }
```

---

## 5. 参数固定策略

### 5.1 需要固定的参数

| 参数类别 | 参数名 | 固定方法 | 后续能否改 |
|---------|-------|---------|-----------|
| 因子权重 | $w_1, w_2, ..., w_n$ | Train期学习 | ❌ Test/Backtest复用 |
| 预处理参数 | MAD倍数、标准化方法 | Signal Ready学习 | ❌ 已冻结 |
| 分组参数 | 分组数量、分组方法 | Train期确定 | ❌ Test复用 |
| 组合参数 | 调仓频率、持仓数量 | Train期确定 | ⚠️ 可在Backtest微调 |

### 5.2 权重固定规范

```json
{
  "frozen_weights": {
    "version": "v1.0",
    "frozen_at": "2024-01-01T00:00:00Z",
    "train_period": {
      "start": "2021-01-01",
      "end": "2023-12-31"
    },
    "factor_weights": {
      "momentum_20d": 0.25,
      "reversal_5d": 0.15,
      "volatility_30d": 0.10,
      "turnover_20d": 0.20,
      "active_address_7d": 0.15,
      "rsi_14": 0.10,
      "macd": 0.05
    },
    "weight_summary": {
      "n_factors": 7,
      "max_weight": 0.25,
      "min_weight": 0.05,
      "weight_concentration": 0.18
    },
    "optimization_meta": {
      "objective": "maximize_ir",
      "method": "SLSQP",
      "train_ir": 0.52,
      "train_sharpe": 1.85
    }
  }
}
```

### 5.3 固定参数的验证

```python
def validate_frozen_weights(frozen_weights, test_data):
    """
    验证固定权重在Test期的表现
    
    不用于调整权重，仅用于验证稳定性
    """
    # 加载固定权重
    weights = np.array(list(frozen_weights['factor_weights'].values()))
    factor_cols = list(frozen_weights['factor_weights'].keys())
    
    # 计算Test期复合因子
    test_composite = test_data[factor_cols].dot(weights)
    
    # 计算Test期IC
    test_ic = test_data[['composite', 'forward_return']].corr(method='spearman').iloc[0, 1]
    
    # 比较Train vs Test
    train_ir = frozen_weights['optimization_meta']['train_ir']
    
    validation_result = {
        'test_ic_mean': test_ic.mean(),
        'test_ic_std': test_ic.std(),
        'test_ir': test_ic.mean() / test_ic.std(),
        'ir_decay': (train_ir - test_ir.mean() / test_ic.std()) / train_ir,
        'validation_verdict': 'PASS' if test_ic.mean() > 0 else 'WARNING'
    }
    
    return validation_result
```

---

## 6. 组合构造规则初版

### 6.1 组合构造决策树

```
复合因子得分
    ↓
筛选池（Filter Pool）
    ├─ 流动性过滤
    ├─ 新币过滤
    └─ 稳定币过滤
    ↓
分组（Grouping）
    ├─ 十分位
    └─ 5分组
    ↓
选币（Selection）
    ├─ Top N
    ├─ 多空配对
    └─ 分层抽样
    ↓
权重分配（Weighting）
    ├─ 等权
    ├─ 因子分值加权
    └─ 市值倒数加权
    ↓
最终组合
```

### 6.2 筛选池规则

```python
def define_universe_filter(df):
    """
    定义筛选池规则
    
    返回符合条件的标的
    """
    # 规则1：流动性过滤
    liquidity_filter = (
        (df['volume_24h'] > 1_000_000) &  # 24h成交额>100万U
        (df['spread_pct'] < 0.5)           # 买卖价差<0.5%
    )
    
    # 规则2：上市时间过滤
    listing_filter = df['days_listed'] > 30  # 上市超过30天
    
    # 规则3：稳定币过滤
    stablecoin_filter = ~df['is_stablecoin']
    
    # 规则4：价格过滤
    price_filter = df['price'] > 0.01  # 价格>1分
    
    # 合并所有过滤条件
    valid_universe = (
        liquidity_filter & 
        listing_filter & 
        stablecoin_filter & 
        price_filter
    )
    
    return valid_universe
```

### 6.3 选币规则

#### 规则1：Top N 多头

```python
def top_n_selection(df, composite_col, n=20):
    """
    选择复合因子最高的N个币种做多
    """
    # 每个时间点独立选择
    def select_top(group):
        return group.nlargest(n, composite_col)
    
    selected = df.groupby('timestamp').apply(select_top).droplevel(0)
    
    # 等权分配
    selected['weight'] = 1.0 / n
    
    return selected
```

#### 规则2：多空配对

```python
def long_short_selection(df, composite_col, n_long=10, n_short=10):
    """
    多空配对：做多Top N，做空Bottom N
    """
    def select_ls(group):
        long = group.nlargest(n_long, composite_col).copy()
        short = group.nsmallest(n_short, composite_col).copy()
        
        long['weight'] = 1.0 / n_long
        short['weight'] = -1.0 / n_short
        
        return pd.concat([long, short])
    
    selected = df.groupby('timestamp').apply(select_ls).droplevel(0)
    
    return selected
```

#### 规则3：分层抽样

```python
def stratified_selection(df, composite_col, n_strata=5, n_per_stratum=4):
    """
    分层抽样：每个因子十分位选择若干币种
    """
    def stratified_select(group):
        # 分层
        group['stratum'] = pd.qcut(
            group[composite_col],
            q=n_strata,
            labels=range(n_strata),
            duplicates='drop'
        )
        
        # 每层随机抽样
        selected = group.groupby('stratum').apply(
            lambda x: x.sample(n=min(n_per_stratum, len(x)), random_state=42)
        ).droplevel(0)
        
        # 权重：等权
        selected['weight'] = 1.0 / len(selected)
        
        return selected
    
    selected = df.groupby('timestamp').apply(stratified_select).droplevel(0)
    
    return selected
```

### 6.4 权重分配规则

```python
def portfolio_weighting(df, method='equal', **kwargs):
    """
    组合权重分配
    
    参数:
        df: 已选中的组合
        method: 权重方法 ('equal', 'factor_score', 'inv_market_cap')
    """
    if method == 'equal':
        # 等权
        df['final_weight'] = df.groupby('timestamp')['weight'].transform(
            lambda x: x / x.sum()
        )
    
    elif method == 'factor_score':
        # 因子分值加权
        def score_weighted(group):
            # 因子分值归一化到[0,1]
            scores = (group['composite_score'] - group['composite_score'].min())
            scores = scores / scores.sum()
            return scores
        
        df['final_weight'] = df.groupby('timestamp').apply(score_weighted).droplevel(0)
    
    elif method == 'inv_market_cap':
        # 市值倒数加权（小币权重更高）
        def inv_cap_weighted(group):
            inv_cap = 1 / group['market_cap']
            weights = inv_cap / inv_cap.sum()
            return weights
        
        df['final_weight'] = df.groupby('timestamp').apply(inv_cap_weighted).droplevel(0)
    
    return df
```

### 6.5 调仓规则

```python
def rebalance_rules(current_portfolio, new_signals, threshold=0.3):
    """
    调仓规则：决定何时调仓
    
    参数:
        current_portfolio: 当前持仓
        new_signals: 新信号
        threshold: 权重变化阈值
    """
    # 计算权重变化
    weight_change = abs(new_signals['final_weight'] - current_portfolio['final_weight'])
    
    # 调仓条件
    rebalance_signals = (
        (weight_change > threshold) |  # 权重变化超过阈值
        (new_signals['symbol'].isin(current_portfolio['symbol']) == False) |  # 新进标的
        (current_portfolio['symbol'].isin(new_signals['symbol']) == False)  # 被踢出标的
    )
    
    return rebalance_signals
```

---

## 7. Formal Gate 要求

### 7.1 Train Formal Gate

**通过标准**（必须全部满足）：

| Gate 项目 | 标准 | 验证方法 |
|-----------|------|----------|
| 权重学习完成 | 所有因子权重已确定 | 人工审查 |
| Train期IR达标 | Train IR > 0.5 | 自动验证 |
| 权重稳定性 | 权重变异系数 < 0.5 | 自动验证 |
| 交叉验证通过 | 各折IC方向一致 | 自动验证 |
| 组合规则完整 | 筛选-选币-权重-调仓完整 | 人工审查 |

### 7.2 Audit Gate（审计门禁）

**补充检查项**：

| 检查项 | 说明 | 处理方式 |
|--------|------|----------|
| 因子冗余度 | 权重集中于少数因子 | 记录，不阻断 |
| 样本敏感度 | 不同Train窗口权重大幅波动 | 记录，不阻断 |
| 交易成本预估 | 基于换手率预估成本 | 记录，不阻断 |

---

## 8. 常见错误和反模式

### 8.1 权重学习错误

| 错误 | 后果 | 正确做法 |
|------|------|----------|
| 用全样本学习权重 | 前视偏差 | 只用Train窗口 |
| 忽略时间序列特性 | 随机交叉验证 | 时间序列交叉验证 |
| 过度优化 | 权重不稳定 | 正则化+稳定性检验 |

### 8.2 组合构造错误

| 错误 | 后果 | 正确做法 |
|------|------|----------|
| 忽略流动性过滤 | 无法交易 | 必须设置流动性门槛 |
| 固定持仓数量 | 市场变化时适应性差 | 动态调整持仓数 |
| 忽略交易成本 | 回测收益虚高 | 考虑换手率和成本 |

### 8.3 参数固定错误

| 错误 | 后果 | 正确做法 |
|------|------|----------|
| Train期参数未冻结 | Test期偷看未来 | 严格记录冻结时点 |
| 参数泄露到Test | 样本外失效 | Train/Test完全隔离 |

---

## 9. 实际案例：加密货币多因子组合

### 9.1 案例背景

- **Universe**：Top 200 币种（不含稳定币）
- **Train期**：2021-2023
- **因子**：7个经过Signal Ready的因子

### 9.2 完整实现

```python
import pandas as pd
import numpy as np
from scipy.optimize import minimize

# 1. 加载数据
train_data = pd.read_parquet('factor_library_train.parquet')

# 2. 因子列表
factor_cols = [
    'momentum_20d',
    'reversal_5d',
    'volatility_30d',
    'turnover_20d',
    'active_address_7d',
    'rsi_14',
    'macd'
]

# 3. 权重学习
def optimize_weights(train_data, factor_cols):
    """最大化IR"""
    def negative_ir(weights):
        # 复合因子
        composite = train_data[factor_cols].dot(weights)
        composite = (composite - composite.mean()) / composite.std()
        
        # 滚动IC
        ic_series = train_data.groupby('timestamp').apply(
            lambda g: g[['composite', 'forward_return_5d']].corr('spearman').iloc[0, 1]
        )
        
        return -ic_series.mean() / ic_series.std()
    
    # 约束
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for _ in factor_cols]
    x0 = np.ones(len(factor_cols)) / len(factor_cols)
    
    result = minimize(negative_ir, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x

# 4. 学习权重
optimal_weights = optimize_weights(train_data, factor_cols)

# 5. 构造复合因子
train_data['composite'] = train_data[factor_cols].dot(optimal_weights)
train_data['composite'] = train_data.groupby('timestamp')['composite'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# 6. 组合构造
def construct_portfolio(df, composite_col, n_long=15, n_short=15):
    """多空组合构造"""
    portfolios = []
    
    for timestamp in df['timestamp'].unique():
        cross_section = df[df['timestamp'] == timestamp].copy()
        
        # 流动性过滤
        valid = cross_section[cross_section['volume_24h'] > 1_000_000]
        
        if len(valid) < n_long + n_short:
            continue
        
        # 选币
        long = valid.nlargest(n_long, composite_col)
        short = valid.nsmallest(n_short, composite_col)
        
        long['weight'] = 1.0 / n_long
        short['weight'] = -1.0 / n_short
        
        portfolio = pd.concat([long, short])
        portfolio['timestamp'] = timestamp
        portfolios.append(portfolio)
    
    return pd.concat(portfolios)

# 7. Train期回测
train_portfolios = construct_portfolio(train_data, 'composite')
train_returns = train_portfolios.groupby('timestamp').apply(
    lambda g: (g['weight'] * g['forward_return_5d']).sum()
)

# 8. 性能评估
train_sharpe = train_returns.mean() / train_returns.std() * np.sqrt(365)
train_max_dd = (train_returns.cumsum() - train_returns.cumsum().cummax()).min()

print(f"Train Sharpe: {train_sharpe:.2f}")
print(f"Train Max DD: {train_max_dd:.2%}")
```

### 9.3 权重结果

```json
{
  "factor_weights": {
    "momentum_20d": 0.28,
    "reversal_5d": 0.18,
    "volatility_30d": 0.08,
    "turnover_20d": 0.15,
    "active_address_7d": 0.12,
    "rsi_14": 0.10,
    "macd": 0.09
  },
  "train_performance": {
    "sharpe_ratio": 1.92,
    "max_drawdown": -0.15,
    "annual_return": 0.45,
    "win_rate": 0.56
  }
}
```

---

## 10. 输出 Artifact 规范

### 10.1 必需产出物

| 产物 | 格式 | 用途 | 消费者 |
|------|------|------|--------|
| frozen_weights.json | JSON | 冻结的因子权重 | Test/Backtest |
| composite_model.pkl | Pickle | 复合因子计算模型 | Test/Backtest |
| portfolio_rules.md | Markdown | 组合构造规则 | Backtest |
| train_performance.json | JSON | Train期性能报告 | 决策参考 |
| weight_analysis.csv | CSV | 权重稳定性分析 | 研究参考 |

### 10.2 Frozen Spec 示例

```yaml
# frozen_train_spec.yaml

version: "v1.0"
frozen_at: "2024-01-01T00:00:00Z"

train_config:
  period:
    start: "2021-01-01"
    end: "2023-12-31"
  n_obs: 1095

factor_weights:
  momentum_20d: 0.28
  reversal_5d: 0.18
  volatility_30d: 0.08
  turnover_20d: 0.15
  active_address_7d: 0.12
  rsi_14: 0.10
  macd: 0.09

portfolio_rules:
  universe_filter:
    min_volume_24h: 1000000
    min_days_listed: 30
    exclude_stablecoin: true
  
  selection:
    method: "long_short_pair"
    n_long: 15
    n_short: 15
  
  weighting:
    method: "equal"
  
  rebalance:
    frequency: "weekly"
    weight_change_threshold: 0.3

交接要求:
  - Test必须使用frozen_weights中的权重
  - Test不得修改portfolio_rules
  - 如需调整规则，创建Child Lineage
```

---

## 11. 与下一阶段的交接标准

### 11.1 交接内容

| 交接物 | 内容 | 用途 |
|--------|------|------|
| Frozen Spec | 权重+规则配置 | Test照单执行 |
| 复合因子模型 | 计算逻辑 | Test生成信号 |
| Train基准 | Train期性能 | Test对比基准 |

### 11.2 交接检查清单

**Train 提交前**：
- [ ] 权重已完全冻结
- [ ] 组合规则已完整定义
- [ ] Train性能已评估
- [ ] Frozen Spec已生成

**Test 接收时**：
- [ ] Frozen Spec可加载
- [ ] 权重和规则可执行
- [ ] 预期性能已明确

### 11.3 不允许的交接后修改

| 项目 | Train 冻结后 | Test 能否改 |
|------|-------------|-----------|
| 因子权重 | ❌ | ❌ 绝对禁止 |
| 组合规则 | ❌ | ❌ 需改则Child Lineage |
| 筛选条件 | ❌ | ⚠️ 仅流动性过滤可微调 |

---

**文档版本**: v1.0  
**最后更新**: 2026-04-02  
**维护者**: Quant Research Team
