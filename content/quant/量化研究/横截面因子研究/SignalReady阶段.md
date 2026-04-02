# Signal Ready 阶段规范 — 横截面因子研究

## 目录

1. [阶段概述](#1-阶段概述)
2. [核心任务](#2-核心任务)
3. [候选因子公式设计](#3-候选因子公式设计)
4. [因子预处理流程](#4-因子预处理流程)
5. [单因子Diagnostics](#5-单因子diagnostics)
6. [因子家族去重初步](#6-因子家族去重初步)
7. [Formal Gate 要求](#7-formal-gate-要求)
8. [常见错误和反模式](#8-常见错误和反模式)
9. [实际案例：加密货币横截面动量因子](#9-实际案例加密货币横截面动量因子)
10. [输出 Artifact 规范](#10-输出-artifact-规范)
11. [与下一阶段的交接标准](#11-与下一阶段的交接标准)

---

## 1. 阶段概述

### 1.1 阶段定义

Signal Ready 是横截面因子研究流程的第三阶段，位于 Data Ready 之后、Train 之前。

**核心目标**：将原始候选因子转化为**可分析、可组合的标准化因子库**，完成单因子诊断和初步去重。

### 1.2 横截面因子研究的特殊性

与传统时间序列策略不同，横截面因子研究在 Signal Ready 阶段有独特要求：

| 维度 | 时间序列策略 | 横截面因子研究 |
|------|-------------|---------------|
| 信号生成 | 单标的独立计算 | 所有标的统一计算 |
| 预处理要求 | 基本标准化 | 必须中性化（行业/市值） |
| 诊断重点 | 自相关性 | IC/IR、分组收益 |
| 去重需求 | 参数相关性检验 | 因子家族相关性检验 |

### 1.3 在流程中的位置

```
Mandate → Data Ready → Signal Ready → Train → Test → Backtest → Holdout
                        ↑
                      当前阶段
```

**前置依赖**：
- Mandate：已冻结 Universe（如 Top 200 流通市值币种）、时间切分
- Data Ready：价格、成交量、链上数据已完成对齐和清洗

**后续影响**：
- Train 阶段将基于标准化因子库构建多因子组合
- Test 阶段将验证因子组合的预测能力

---

## 2. 核心任务

### 2.1 任务清单

| 任务 | 输出 | 优先级 |
|------|------|--------|
| 候选因子公式设计 | Factor_Formula_Library.md | P0 |
| 去极值处理 | outlier_treated.parquet | P0 |
| 标准化处理 | standardized.parquet | P0 |
| 中性化处理 | neutralized.parquet | P0 |
| 单因子 Diagnostics | factor_diagnostics_report.json | P0 |
| 因子家族去重 | factor_correlation_matrix.csv | P1 |

### 2.2 核心原则

1. **可比较性优先**：所有因子必须经过统一的预处理才能比较
2. **横截面有效性**：每个时间点的因子分布必须有区分度
3. **样本外完整性**：预处理参数必须在 Train 内学习，Test/Backtest 复用

---

## 3. 候选因子公式设计

### 3.1 因子分类体系

在横截面因子研究中，候选因子通常分为以下类别：

| 类别 | 描述 | 加密货币示例 |
|------|------|-------------|
| **动量类** | 历史收益率的持续性 | 20日动量、12个月累积收益 |
| **反转类** | 短期过度反应的反转 | 5日反转、隔夜跳空 |
| **波动率类** | 风险补偿相关 | 历史波动率、GARCH波动率 |
| **流动性类** | 交易活跃度相关 | 换手率、成交额占比 |
| **链上数据类** | 加密货币特有 | 活跃地址数、大额转账 |
| **技术指标类** | 传统技术分析 | RSI、MACD、布林带 |
| **情绪类** | 市场情绪相关 | 社交媒体活跃度、恐慌指数 |

### 3.2 因子公式模板

每个候选因子需要按以下模板定义：

```markdown
## 因子名称：{Factor_Name}

### 基本信息
- **因子ID**：MOM_20D（动量_20日）
- **因子类型**：动量类
- **数据依赖**：收盘价
- **计算窗口**：20个交易日

### 计算公式
```
MOM_20D(t) = (Close(t) / Close(t-20) - 1) * 100
```

### 数据要求
- **最小历史长度**：至少21个交易日
- **缺失值处理**：历史不足时返回 NaN
- **新币处理**：上市不足20日返回 NaN

### 时间语义
- **信号时间标签**：close_time（T日收盘）
- **可用时间**：T+1 日开盘
- **前视边界**：仅使用 T-20 至 T 的历史数据

### 预期方向
- **正/负相关**：正相关（因子值越高，预期收益越高）
```

### 3.3 加密货币特殊因子示例

#### 链上活跃地址因子

```python
def active_address_factor(df, window=7):
    """
    链上活跃地址变化率
    
    参数:
        df: 包含 active_addresses 列的 DataFrame
        window: 计算窗口（天）
    
    返回:
        标准化后的因子值
    """
    # 计算环比变化率
    factor = (df['active_addresses'] / df['active_addresses'].shift(window) - 1)
    return factor
```

#### 流动性挖矿收益率因子

```python
def staking_apr_factor(df):
    """
    质押年化收益率因子
    
    逻辑：高APR可能吸引更多资金，推高价格
    """
    return df['staking_apr'] * df['total_staked'] / df['market_cap']
```

---

## 4. 因子预处理流程

### 4.1 预处理流程图

```
原始因子值
    ↓
去极值（Outlier Treatment）
    ↓
标准化（Standardization）
    ↓
中性化（Neutralization）
    ↓
标准化因子库
```

### 4.2 去极值处理

#### 为什么要去极值

极端值会：
- 扭曲因子分布
- 影响相关性分析
- 导致组合权重失调

#### 去极值方法

| 方法 | 公式 | 适用场景 | 加密货币注意 |
|------|------|---------|-------------|
| **MAD法** | $\pm 3 \times MAD$ | 非正态分布 | 推荐，币价分布肥尾 |
| **3σ法** | $\mu \pm 3\sigma$ | 近似正态分布 | 慎用，可能过度截断 |
| **百分位法** | [1%, 99%] | 任意分布 | 简单有效 |

#### MAD法实现

```python
def mad_winsorize(factor, n=3):
    """
    MAD去极值（Median Absolute Deviation）
    
    参数:
        factor: 横截面因子值（Series）
        n: MAD倍数，通常取3
    
    返回:
        处理后的因子值
    """
    median = factor.median()
    mad = (factor - median).abs().median()
    upper = median + n * mad
    lower = median - n * mad
    
    return factor.clip(lower, upper)
```

#### 横截面 vs 时序去极值

```python
# 错误：时序去极值（会破坏横截面关系）
df['factor'] = df.groupby('symbol')['factor'].transform(lambda x: x.clip(x.quantile(0.01), x.quantile(0.99)))

# 正确：横截面去极值（每个时间点独立处理）
df['factor'] = df.groupby('timestamp')['factor'].transform(lambda x: mad_winsorize(x))
```

### 4.3 标准化处理

#### 为什么要标准化

不同因子的量纲差异巨大：
- 动量因子：范围[-50%, +200%]
- RSI因子：范围[0, 100]
- 换手率：范围[0.01%, 50%]

标准化使因子可比较、可组合。

#### 标准化方法

| 方法 | 公式 | 特点 |
|------|------|------|
| **Z-Score** | $(x - \mu) / \sigma$ | 均值0，方差1 |
| **Rank** | $rank(x) / (N-1)$ | 严格[0,1]，无极值 |
| **Min-Max** | $(x - min) / (max - min)$ | 严格[0,1] |

#### 横截面Z-Score实现

```python
def cross_sectional_standardize(df, factor_col):
    """
    横截面Z-Score标准化
    
    每个时间点独立计算均值和标准差
    """
    def z_score(group):
        mean = group.mean()
        std = group.std()
        if std == 0 or std != std:  # 零方差或NaN
            return pd.Series(0, index=group.index)
        return (group - mean) / std
    
    df[f'{factor_col}_std'] = df.groupby('timestamp')[factor_col].transform(z_score)
    return df
```

#### 加密货币特殊情况处理

```python
def robust_standardize(df, factor_col):
    """
    鲁棒标准化：处理新币、流动性差币种
    
    特殊处理：
    1. 流动性过低币种不参与标准化计算
    2. 剔除 NaN 后再标准化
    3. 保持横截面结构
    """
    def z_score_robust(group):
        # 流动性过滤
        valid = group[(group['volume_24h'] > 1_000_000) &  # 24h成交额>100万U
                     (group[factor_col].notna())]
        
        if len(valid) < 10:  # 有效样本太少
            return pd.Series(np.nan, index=group.index)
        
        mean = valid[factor_col].mean()
        std = valid[factor_col].std()
        
        if std == 0:
            return pd.Series(0, index=group.index)
        
        result = (group[factor_col] - mean) / std
        return result
    
    return df.groupby('timestamp').apply(z_score_robust).droplevel(0)
```

### 4.4 中性化处理

#### 什么是中性化

**中性化**：剔除因子中对特定风险因子的暴露，使因子成为"纯alpha"。

常见中性化目标：
- **市值中性**：剔除大小盘偏好
- **行业中性**：剔除行业偏向（传统金融）
- **Stablecoin中性**：剔除稳定币暴露（加密货币）

#### 为什么要中性化

| 未中性化风险 | 后果 |
|-------------|------|
| 市值因子掺杂 | 大盘币涨跌影响因子表现 |
| 行业集中 | 特定板块（如DeFi）行情主导 |
| Stablecoin污染 | USDT/USDC表现扭曲结果 |

#### 横截面回归中性化

```python
def orthogonalize_factor(df, factor_col, risk_factors):
    """
    横截面回归中性化
    
    参数:
        df: 包含多标的单时刻数据
        factor_col: 待中性化因子
        risk_factors: 风险因子列表 ['log_market_cap', 'is_stablecoin']
    
    返回:
        中性化后的因子（残差）
    """
    # 准备回归数据
    X = df[risk_factors].values
    y = df[factor_col].values
    
    # 剔除NaN
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X_clean = X[mask]
    y_clean = y[mask]
    
    if len(X_clean) < 10 or np.linalg.matrix_rank(X_clean) < len(risk_factors):
        return df[factor_col]  # 样本不足或多重共线性，返回原值
    
    # OLS回归
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_clean, y_clean)
    
    # 计算残差（中性化后的因子）
    residuals = y - model.predict(X)
    return pd.Series(residuals, index=df.index)

# 应用到每个时间点
def neutralize_cross_section(df, factor_col, risk_cols):
    def neutralize(group):
        return orthogonalize_factor(group, factor_col, risk_cols)
    
    return df.groupby('timestamp').apply(neutralize).droplevel(0)
```

#### 加密货币中性化示例

```python
# 风险因子定义
crypto_risk_factors = [
    'log_market_cap',      # 对数市值
    'is_stablecoin',       # 是否稳定币
    'is_deFi',            # 是否DeFi代币
    'is_layer1',          # 是否Layer1公链
    'volatility_30d'      # 30日波动率
]

# 应用中性化
df['momentum_neutral'] = neutralize_cross_section(
    df, 
    'momentum_raw',
    crypto_risk_factors
)
```

#### 中性化的注意事项

| 问题 | 处理方式 |
|------|---------|
| 多重共线性 | 使用VIF检验，剔除高VIF因子 |
| 样本不足 | 少于30个标的时跳过中性化 |
| 新币处理 | 新币先标准化，暂不中性化 |
| 稳定币处理 | 单独处理稳定币组 |

---

## 5. 单因子Diagnostics

### 5.1 诊断指标体系

单因子诊断回答两个核心问题：
1. **因子有效吗？**（IC/IR分析）
2. **因子怎么用？**（分组收益分析）

#### 核心诊断指标

| 指标 | 英文 | 定义 | 判断标准 |
|------|------|------|---------|
| **IC系数** | Information Coefficient | 因子与未来收益的相关系数 | \|IC\| > 0.03 有效 |
| **IR比率** | Information Ratio | IC均值 / IC标准差 | IR > 0.5 稳定 |
| **IC胜率** | IC Win Rate | IC>0的占比 | >55% 方向正确 |
| **ICt统计量** | IC t-stat | IC均值/标准误差*sqrt(N) | \|t\| > 1.96 显著 |

#### 分组收益指标

| 指标 | 定义 | 判断标准 |
|------|------|---------|
| **多空收益** | Top组 - Bottom组 | 年化>5% |
| **单调性** | 分组收益的排序关系 | 严格单调更佳 |
| **最大回撤** | 多空组合回撤 | <20% |

### 5.2 IC分析实现

```python
def factor_ic_analysis(df, factor_col, return_col='forward_return_5d', periods=20):
    """
    因子IC分析
    
    参数:
        df: 包含因子和收益的DataFrame
        factor_col: 因子列名
        return_col: 未来收益列名
        periods: 分析期数
    
    返回:
        IC统计字典
    """
    ic_series = []
    
    for timestamp in df['timestamp'].unique()[-periods:]:
        cross_section = df[df['timestamp'] == timestamp]
        
        # 计算Spearman相关系数（对异常值更鲁棒）
        ic = cross_section[[factor_col, return_col]].corr(method='spearman').iloc[0, 1]
        ic_series.append(ic)
    
    ic_series = pd.Series(ic_series)
    
    stats = {
        'IC_mean': ic_series.mean(),
        'IC_std': ic_series.std(),
        'IR': ic_series.mean() / ic_series.std() if ic_series.std() > 0 else 0,
        'IC_win_rate': (ic_series > 0).mean(),
        'IC_skew': ic_series.skew(),
        'IC_kurtosis': ic_series.kurtosis(),
        'IC_series': ic_series.tolist()
    }
    
    return stats
```

### 5.3 分组收益分析

```python
def factor_group_returns(df, factor_col, return_col, n_groups=5):
    """
    因子分组收益分析
    
    参数:
        df: 包含因子和收益的DataFrame
        factor_col: 因子列名
        return_col: 未来收益列名
        n_groups: 分组数量
    
    返回:
        分组收益统计
    """
    # 每个时间点横截面分组
    def分组时刻(group):
        group['factor_group'] = pd.qcut(
            group[factor_col], 
            q=n_groups, 
            labels=range(1, n_groups+1),
            duplicates='drop'
        )
        return group
    
    df = df.groupby('timestamp').apply(分组时刻).droplevel(0)
    
    # 计算各组平均收益
    group_returns = df.groupby('factor_group')[return_col].mean()
    
    # 计算多空收益
    long_short = group_returns.iloc[-1] - group_returns.iloc[0]
    
    return {
        'group_returns': group_returns.to_dict(),
        'long_short_annualized': long_short * 365,  # 假设日收益
        'monotonicity': check_monotonicity(group_returns)
    }

def check_monotonicity(returns):
    """检查分组收益单调性"""
    returns_list = returns.tolist()
    increasing = all(returns_list[i] <= returns_list[i+1] for i in range(len(returns_list)-1))
    decreasing = all(returns_list[i] >= returns_list[i+1] for i in range(len(returns_list)-1))
    return 'increasing' if increasing else ('decreasing' if decreasing else 'none')
```

### 5.4 诊断报告模板

```json
{
  "factor_id": "MOM_20D",
  "factor_name": "20日动量因子",
  "analysis_period": {
    "start": "2023-01-01",
    "end": "2024-12-31",
    "n_observations": 520
  },
  "ic_analysis": {
    "IC_mean": 0.045,
    "IC_std": 0.12,
    "IR": 0.375,
    "IC_win_rate": 0.58,
    "IC_t_stat": 2.67,
    "IC_max": 0.35,
    "IC_min": -0.28
  },
  "group_analysis": {
    "n_groups": 5,
    "group_returns": {
      "G1": -0.002,
      "G2": -0.001,
      "G3": 0.000,
      "G4": 0.001,
      "G5": 0.003
    },
    "long_short_daily": 0.005,
    "long_short_annualized": 1.83,
    "monotonicity": "increasing",
    "max_drawdown": 0.15
  },
  "diagnostics_verdict": "PASS",
  "notes": "因子有效性显著，但IR偏低，需要与其他因子组合"
}
```

---

## 6. 因子家族去重初步

### 6.1 为什么要去重

同一因子家族内的因子往往高度相关：
- 5日、10日、20日动量高度相关
- RSI、KDJ、Stochastic都是超买超卖指标

保留所有因子会导致：
- 多重共线性
- 过拟合风险
- 组合权重不稳定

### 6.2 相关性分析

```python
def factor_correlation_analysis(df, factor_cols, method='spearman'):
    """
    因子相关性分析
    
    参数:
        df: 因子数据
        factor_cols: 因子列名列表
        method: 相关系数方法 ('pearson', 'spearman')
    
    返回:
        相关性矩阵
    """
    # 对每个时间点计算相关性，然后平均
    correlations = []
    
    for timestamp in df['timestamp'].unique():
        cross_section = df[df['timestamp'] == timestamp]
        corr = cross_section[factor_cols].corr(method=method)
        correlations.append(corr)
    
    # 平均相关性矩阵
    avg_corr = np.mean(correlations, axis=0)
    return pd.DataFrame(avg_corr, index=factor_cols, columns=factor_cols)
```

### 6.3 去重策略

#### 策略1：IC筛选

```python
def select_by_ic_rank(factor_diagnostics, top_n=20):
    """
    按IC绝对值排序，选择Top N
    """
    df = pd.DataFrame(factor_diagnostics)
    df['IC_abs'] = df['IC_mean'].abs()
    return df.nlargest(top_n, 'IC_abs')['factor_id'].tolist()
```

#### 策略2：聚类去重

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

def cluster_deduplicate(correlation_matrix, threshold=0.8):
    """
    基于相关性的聚类去重
    
    参数:
        correlation_matrix: 因子相关性矩阵
        threshold: 相关性阈值，高于此值的因子合并
    
    返回:
        保留的因子列表
    """
    # 转换为距离矩阵
    distance_matrix = 1 - np.abs(correlation_matrix)
    np.fill_diagonal(distance_matrix.values, 0)
    
    # 层次聚类
    clustering = AgglomerativeClustering(
        n_clusters=None,
        affinity='precomputed',
        linkage='average',
        distance_threshold=1-threshold
    )
    
    labels = clustering.fit_predict(distance_matrix)
    
    # 每个聚类选择IC最高的因子
    selected = []
    for cluster_id in np.unique(labels):
        cluster_factors = np.where(labels == cluster_id)[0]
        # 假设已有factor_diagnostics数据
        best = max(cluster_factors, key=lambda x: factor_diagnostics[x]['IC_abs'])
        selected.append(correlation_matrix.index[best])
    
    return selected
```

#### 策略3：因子合成

```python
def composite_factor(df, factor_cols, weights=None):
    """
    因子合成：将高相关因子合成一个复合因子
    
    参数:
        df: 因子数据
        factor_cols: 要合成的因子列表
        weights: 权重，None则等权
    
    返回:
        合成后的因子
    """
    if weights is None:
        weights = np.ones(len(factor_cols)) / len(factor_cols)
    
    # 先标准化各因子
    normalized_factors = []
    for col in factor_cols:
        normalized = (df[col] - df[col].mean()) / df[col].std()
        normalized_factors.append(normalized)
    
    # 加权求和
    composite = sum(w * f for w, f in zip(weights, normalized_factors))
    
    # 再次标准化
    return (composite - composite.mean()) / composite.std()
```

### 6.4 去重报告

```markdown
## 因子去重报告

### 分析概况
- 分析因子数量：45个
- 去重阈值：相关性 > 0.8
- 保留因子数量：18个

### 去重结果

| 因子家族 | 原始数量 | 保留数量 | 保留因子 |
|---------|---------|---------|---------|
| 动量类 | 12 | 3 | MOM_20D, MOM_60D, 12M_MOM |
| 反转类 | 8 | 2 | REV_5D, REV_Monthly |
| 波动率类 | 6 | 2 | VOL_20D, GARCH_VOL |
| 流动性类 | 10 | 4 | TURNOVER_20D, AMIHUD_ILR, Volume_VWAP, Bid_Ask_Spread |
| 技术指标类 | 9 | 7 | RSI_14, MACD, BB_Width, ATR, OBV, MFI, Stochastic |

### 相关性热图
[生成相关性矩阵热图]

### 下一步
保留的18个因子进入Train阶段的多因子组合学习
```

---

## 7. Formal Gate 要求

### 7.1 Signal Ready Formal Gate

**通过标准**（必须全部满足）：

| Gate 项目 | 标准 | 验证方法 |
|-----------|------|----------|
| 因子公式完整性 | 每个因子有完整定义 | 人工审查 |
| 预处理一致性 | 所有因子经过统一预处理 | 自动检查 |
| IC统计显著 | 至少50%因子 \|IC\| > 0.02 | 自动验证 |
| 无前视风险 | 预处理参数未用未来数据 | 代码审计 |
| 因子去重完成 | 高相关因子已处理 | 人工审查 |

### 7.2 Audit Gate（审计门禁）

**补充检查项**：

| 检查项 | 说明 | 处理方式 |
|--------|------|----------|
| 因子覆盖度 | 每个时间点的有效因子比例 | 记录，不阻断 |
| 稳定币处理 | 稳定币是否正确处理 | 记录，不阻断 |
| 新币处理 | 新上市币种处理策略 | 记录，不阻断 |
| 因子时效性 | 因子IC是否随时间衰减 | 记录，不阻断 |

---

## 8. 常见错误和反模式

### 8.1 预处理错误

| 错误 | 后果 | 正确做法 |
|------|------|----------|
| 时序标准化而非横截面 | 破坏横截面比较关系 | 每个时间点独立标准化 |
| 预处理参数用全样本 | 前视偏差 | 在Train窗口学习，Test/Backtest复用 |
| 忽略中性化 | 因子混杂风险暴露 | 必须对主要风险因子中性化 |

### 8.2 IC分析错误

| 错误 | 后果 | 正确做法 |
|------|------|----------|
| 用Pearson相关 | 异常值影响大 | 用Spearman秩相关 |
| 混合计算全时期IC | 掩盖时序变化 | 滚动窗口计算IC |
| 忽略IC显著性 | 虚假相关性 | 计算t统计量 |

### 8.3 去重错误

| 错误 | 后果 | 正确做法 |
|------|------|----------|
| 仅用Train数据相关性 | 过拟合去重 | 用全样本相关性去重 |
| 丢弃低IC高相关因子 | 可能丢失互补信息 | 考虑因子合成 |

---

## 9. 实际案例：加密货币横截面动量因子

### 9.1 案例背景

- **Universe**：Top 200 流通市值币种（不含稳定币）
- **时间范围**：2021-2024
- **因子**：20日动量因子

### 9.2 完整实现流程

```python
import pandas as pd
import numpy as np

# 1. 原始因子计算
def calculate_momentum_20d(df_prices):
    """计算20日动量"""
    df = df_prices.copy()
    df['momentum_raw'] = df.groupby('symbol')['close'].transform(
        lambda x: x / x.shift(20) - 1
    )
    return df

# 2. 去极值
def winsorize_mad(df, factor_col, n=3):
    """MAD去极值"""
    def mad_clip(series):
        median = series.median()
        mad = (series - median).abs().median()
        upper = median + n * mad
        lower = median - n * mad
        return series.clip(lower, upper)
    
    df[f'{factor_col}_winsor'] = df.groupby('timestamp')[factor_col].transform(mad_clip)
    return df

# 3. 标准化
def standardize_zscore(df, factor_col):
    """Z-Score标准化"""
    def z_score(series):
        return (series - series.mean()) / series.std()
    
    df[f'{factor_col}_std'] = df.groupby('timestamp')[factor_col].transform(z_score)
    return df

# 4. 中性化
def neutralize_market_cap(df, factor_col):
    """市值中性化"""
    def regress_out(group):
        X = group[['log_market_cap']].values
        y = group[factor_col].values
        
        mask = ~(np.isnan(X) | np.isnan(y))
        if sum(mask) < 10:
            return pd.Series(np.nan, index=group.index)
        
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X[mask], y[mask])
        
        return y - model.predict(X)
    
    df[f'{factor_col}_neutral'] = df.groupby('timestamp').apply(regress_out).droplevel(0)
    return df

# 完整流程
def process_momentum_factor(df_prices):
    # 1. 计算原始因子
    df = calculate_momentum_20d(df_prices)
    
    # 2. 去极值
    df = winsorize_mad(df, 'momentum_raw')
    
    # 3. 标准化
    df = standardize_zscore(df, 'momentum_raw_winsor')
    
    # 4. 中性化
    df = neutralize_market_cap(df, 'momentum_raw_winsor_std')
    
    return df[['timestamp', 'symbol', 'momentum_raw_winsor_std_neutral']]
```

### 9.3 诊断结果

```
20日动量因子诊断报告
====================

IC分析:
- IC均值: 0.048
- IC标准差: 0.152
- IR: 0.316
- IC胜率: 56.3%
- IC t统计量: 2.21 (p < 0.05)

分组收益 (5分组):
- G1 (最低): -0.15%/日
- G2: -0.08%/日
- G3: 0.02%/日
- G4: 0.11%/日
- G5 (最高): 0.19%/日
- 多空收益: 0.34%/日 → 年化 124%
- 最大回撤: 18%

结论: 因子有效，但稳定性一般，建议与其他因子组合使用
```

---

## 10. 输出 Artifact 规范

### 10.1 必需产出物

| 产物 | 格式 | 用途 | 消费者 |
|------|------|------|--------|
| Factor_Formula_Library.md | Markdown | 因子定义文档 | 所有阶段 |
| factor_library.parquet | Parquet | 标准化因子库 | Train/Test |
| factor_diagnostics.json | JSON | 单因子诊断报告 | Train/决策 |
| correlation_matrix.csv | CSV | 因子相关性矩阵 | 去重决策 |
| preprocessing_params.json | JSON | 预处理参数（Train学习） | Test/Backtest复用 |

### 10.2 因子库数据结构

```
factor_library.parquet 结构:
- timestamp: 统一时间戳
- symbol: 标的代码
- factor_1: 标准化因子1
- factor_2: 标准化因子2
- ...
- factor_N: 标准化因子N
- is_valid: 该记录是否有效（用于新币/流动性过滤）
```

### 10.3 预处理参数规范

```json
{
  "preprocessing_config": {
    "outlier_treatment": {
      "method": "MAD",
      "n_mad": 3
    },
    "standardization": {
      "method": "z_score",
      "min_samples": 10
    },
    "neutralization": {
      "risk_factors": ["log_market_cap", "is_stablecoin"],
      "method": "ols_residual"
    }
  },
  "frozen_at": "2024-01-01T00:00:00Z",
  "train_period": {
    "start": "2021-01-01",
    "end": "2023-12-31"
  }
}
```

---

## 11. 与下一阶段的交接标准

### 11.1 交接内容

| 交接物 | 内容 | 用途 |
|--------|------|------|
| Frozen Spec | 预处理参数配置 | Test/Backtest复用 |
| 因子库 | 标准化因子数据 | Train的输入 |
| 诊断报告 | 单因子有效性验证 | 因子选择参考 |
| 去重结果 | 保留因子列表 | Train的候选因子集 |

### 11.2 交接检查清单

**Signal Ready 提交前**：
- [ ] 所有因子已完成预处理
- [ ] 单因子IC诊断完成
- [ ] 高相关因子已去重
- [ ] 预处理参数已冻结

**Train 接收时**：
- [ ] 因子库可加载
- [ ] 因子数量符合预期
- [ ] 预处理参数可复用

### 11.3 不允许的交接后修改

| 项目 | Signal Ready 冻结后 | Train 能否改 |
|------|---------------------|-------------|
| 预处理方法 | ❌ | ❌ 绝对禁止 |
| 预处理参数 | ✅ 在Train学习 | ✅ 可以重新学习 |
| 因子公式 | ❌ | ❌ 需要则重走Signal Ready |
| 因子选择 | ✅ 初步筛选 | ✅ Train可以进一步选择 |

---

**文档版本**: v1.0  
**最后更新**: 2026-04-02  
**维护者**: Quant Research Team
