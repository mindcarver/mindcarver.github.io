# Qlib特征工程实践指南

## 引言

前四个文档我们从理论和数学层面深入探讨了Qlib特征工程的核心概念：全景概览、横截面标准化与中性化、Horizon对齐、相对强弱预测的量化思维。

本文将从实践角度出发，提供完整的Qlib特征工程实践指南，包括特征张量优化、Pipeline配置、因子质量评估、数据泄漏防护，以及链上数据集成的深度实践。

---

## 1. 特征张量优化

### 1.1 内存优化策略

**数据类型优化**

在量化因子计算中，数据的精度要求通常不高，可以通过降低数据类型来节省内存。

**优化示例**：

```python
# 优化前：float64（8字节）
import numpy as np
factor = np.random.randn(5000, 1000).astype('float64')  # 5000只股票 x 1000天
memory = factor.nbytes / 1024**2  # 38.15 MB

# 优化后：float32（4字节）
factor = factor.astype('float32')
memory = factor.nbytes / 1024**2  # 19.07 MB（节省50%）
```

**时间戳优化**：

```python
# 优化前：datetime64[ns]（8字节）
dates = pd.date_range('2020-01-01', '2023-12-31')
memory = dates.nbytes / 1024**2  # 0.01 MB（小数据不显著）

# 优化后：int64（时间戳）
timestamps = dates.astype('int64')  # Unix时间戳
memory = timestamps.nbytes / 1024**2  # 0.01 MB
```

**稀疏矩阵应用**

对于稀疏数据（如行业哑变量、指数衰减权重），使用稀疏矩阵可以大幅节省内存。

**示例：行业哑变量**

```python
from scipy.sparse import csr_matrix

# 优化前：密集矩阵
industry_dummy = np.zeros((5000, 30))  # 5000只股票 x 30个行业
# 每个股票只属于1个行业，稀疏度约96.7%
memory = industry_dummy.nbytes / 1024**2  # 1.14 MB

# 优化后：稀疏矩阵
row_indices = [0, 1, 2, ..., 4999]  # 股票索引
col_indices = [5, 10, 3, ..., 12]   # 行业索引
data = np.ones(5000)

industry_sparse = csr_matrix((data, (row_indices, col_indices)), shape=(5000, 30))
memory = industry_sparse.data.nbytes / 1024**2 + \
         industry_sparse.indptr.nbytes / 1024**2 + \
         industry_sparse.indices.nbytes / 1024**2  # 0.09 MB（节省92%）
```

**分片加载**

对于大规模因子库（如10,000个因子，5000只股票，10年数据），分片加载是必要的。

**策略1：按时间分片**

```python
# 将10年数据分成10个年度分片
years = ['2020', '2021', '2022', '2023']
for year in years:
    start_time = f"{year}-01-01"
    end_time = f"{year}-12-31"
    factors = D.features(instruments, factor_exprs, start_time, end_time)
    # 处理当前年度数据
```

**策略2：按资产分片**

```python
# 将5000只股票分成10个分片
chunks = list(range(0, 5000, 500))  # [0, 500, 1000, ..., 4500]
for i in range(len(chunks)-1):
    start_idx = chunks[i]
    end_idx = chunks[i+1]
    chunk_instruments = instruments[start_idx:end_idx]
    factors = D.features(chunk_instruments, factor_exprs, start_time, end_time)
    # 处理当前分片数据
```

### 1.2 计算优化

**向量化操作**

避免使用for循环，使用numpy/pandas的向量化操作。

**慢示例**：

```python
# 慢：for循环
def slow_rolling_mean(data, window):
    result = np.zeros_like(data)
    for i in range(window, len(data)):
        result[i] = data[i-window:i].mean()
    return result

%timeit slow_rolling_mean(price, 20)
# 输出：100 ms ± 5 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

**快示例**：

```python
# 快：向量化
def fast_rolling_mean(data, window):
    return data.rolling(window).mean()

%timeit fast_rolling_mean(price, 20)
# 输出：1 ms ± 0.1 ms per loop (mean ± std. dev. of 7 runs, 1000 loops each)
```

**并行化**

对于大规模因子计算，使用多进程并行化。

**示例**：

```python
from multiprocessing import Pool

def compute_factor(args):
    stock_id, start_time, end_time = args
    # 计算单个股票的因子
    factor = D.features([stock_id], [factor_expr], start_time, end_time)
    return factor

# 串行（慢）
for stock_id in instruments:
    factor = compute_factor((stock_id, start_time, end_time))

# 并行（快）
args = [(stock_id, start_time, end_time) for stock_id in instruments]
with Pool(8) as p:  # 8个进程
    results = p.map(compute_factor, args)
```

**滚动窗口优化**

对于滚动窗口计算，使用增量计算避免重复计算。

**慢示例**：

```python
# 慢：每次重新计算
def slow_rolling_incremental(data, window):
    result = np.zeros_like(data)
    for i in range(window, len(data)):
        result[i] = data[i-window:i].mean()  # 重复计算
    return result
```

**快示例**：

```python
# 快：增量计算
def fast_rolling_incremental(data, window):
    result = np.zeros_like(data)
    # 计算第一个窗口
    result[window-1] = data[:window].mean()

    # 增量更新
    for i in range(window, len(data)):
        # 减去离开窗口的值，加上进入窗口的值
        result[i] = result[i-1] + (data[i] - data[i-window]) / window

    return result

# 证明正确性
# result[i] = result[i-1] + (data[i] - data[i-window]) / window
# = mean(data[i-window:i-1]) + (data[i] - data[i-window]) / window
# = (sum(data[i-window:i-1]) + data[i] - data[i-window]) / window
# = sum(data[i-window+1:i]) / window
# = mean(data[i-window+1:i+1])
# = mean(data[i-window+1:i])  # 下一轮的窗口
```

### 1.3 大规模因子库管理

**存储方案**

**方案1：HDF5**

适合中小规模因子库（10K因子）。

```python
import h5py

# 创建HDF5文件
with h5py.File('factor_library.h5', 'w') as f:
    # 创建数据集
    f.create_dataset('factor1', data=factor1, compression='gzip')
    f.create_dataset('factor2', data=factor2, compression='gzip')

    # 创建属性
    f['factor1'].attrs['name'] = 'momentum_20'
    f['factor1'].attrs['ic_mean'] = 0.05
    f['factor1'].attrs['ir'] = 1.2

# 读取因子
with h5py.File('factor_library.h5', 'r') as f:
    factor1 = f['factor1'][:]
    ic_mean = f['factor1'].attrs['ic_mean']
```

**方案2：Parquet**

适合大规模因子库（100K+因子）。

```python
import pyarrow.parquet as pq
import pandas as pd

# 保存因子（Parquet格式）
factor_df = pd.DataFrame({
    'date': dates,
    'instrument': instruments,
    'factor1': factor1,
    'factor2': factor2
})
factor_df.to_parquet('factor_library.parquet', engine='pyarrow')

# 读取因子
factor_df = pd.read_parquet('factor_library.parquet')
```

**索引策略**

**复合索引**：

```python
# 设置复合索引
factor_df.set_index(['date', 'instrument'], inplace=True)

# 快速查询
factor = factor_df.loc['2023-01-01', '000001.SZ']
```

**B树索引**：

```python
# 对于HDF5
with h5py.File('factor_library.h5', 'a') as f:
    # 创建软链接作为索引
    f.create_dataset('date_index', data=dates)
    f.create_dataset('instrument_index', data=instruments)
```

---

## 2. Pipeline配置模板

### 2.1 基础模板：行情因子

```yaml
# config/baseline.yaml

# 数据源
data:
  provider: "qlib.data.LocalFileProvider"
  uri: "data/qlib/qlib_data/cn_data"
  region: "cn"

# 因子定义
factors:
  - name: "ma_5"
    expr: "Mean($close, 5)"
    description: "5日均线"

  - name: "ma_20"
    expr: "Mean($close, 20)"
    description: "20日均线"

  - name: "momentum_10"
    expr: "($close / Ref($close, 10)) - 1"
    description: "10日动量"

# 标签
label:
  expr: "Ref($close, -5) / $close - 1"
  horizon: 5
  description: "未来5日收益率"

# 标准化
normalization:
  type: "zscore"
  axis: "cross_section"

# 回测配置
backtest:
  start_time: "2020-01-01"
  end_time: "2023-12-31"
  rebalance_freq: "5d"
  top_k: 100  # 选Top 100只股票
```

### 2.2 进阶模板：多因子组合

```yaml
# config/advanced.yaml

# 数据源
data:
  provider: "qlib.data.LocalFileProvider"
  uri: "data/qlib/qlib_data/cn_data"

# 因子定义
factors:
  # 技术因子
  - name: "rsi_14"
    expr: "RSI($close, 14)"
    category: "technical"

  - name: "boll_upper"
    expr: "Mean($close, 20) + 2 * Std($close, 20)"
    category: "technical"

  - name: "boll_lower"
    expr: "Mean($close, 20) - 2 * Std($close, 20)"
    category: "technical"

  # 动量因子
  - name: "momentum_20"
    expr: "($close / Ref($close, 20)) - 1"
    category: "momentum"

  - name: "price_acceleration"
    expr: "Ref($close, -1) / $close - Ref($close, -20) / Ref($close, -21)"
    category: "momentum"

  # 波动因子
  - name: "volatility_20"
    expr: "Std($close, 20) / Mean($close, 20)"
    category: "volatility"

  - name: "atr_14"
    expr: "ATR($high, $low, $close, 14)"
    category: "volatility"

  # 流动性因子
  - name: "turnover_rate"
    expr: "$turnover"
    category: "liquidity"

  - name: "volume_ratio"
    expr: "$volume / Mean($volume, 20)"
    category: "liquidity"

# 因子组合
combination:
  method: "ic_weighted"  # IC加权
  weights: "auto"  # 自动学习
  min_ic: 0.03  # 最小IC阈值

# 中性化
neutralization:
  - "industry"  # 行业中性
  - "market_cap"  # 市值中性

# 标准化
normalization:
  type: "zscore"
  axis: "cross_section"
  method: "robust"  # 鲁棒标准化（对异常值不敏感）

# 回测配置
backtest:
  start_time: "2020-01-01"
  end_time: "2023-12-31"
  rebalance_freq: "5d"
  top_k: 50
  max_weight: 0.02  # 单只股票最大权重2%
```

---

## 3. 因子质量评估流程

### 3.1 IC/IR分析流程

```python
def evaluate_factor(factor_name, start_date, end_date, horizon=5):
    """
    评估因子的IC/IR
    """
    # 1. 加载数据
    factor = load_factor(factor_name, start_date, end_date)
    price = load_price(start_date, end_date)

    # 2. 计算收益率
    returns = price.pct_change(horizon).shift(-horizon)

    # 3. 对齐数据
    data = pd.DataFrame({
        'factor': factor,
        'return': returns
    }).dropna()

    # 4. 计算IC序列
    ic_series = []
    for date in data.index.get_level_values('date').unique():
        subset = data.loc[date]
        ic = subset['factor'].corr(subset['return'])
        ic_series.append(ic)

    ic_series = pd.Series(ic_series)

    # 5. 统计指标
    ic_mean = ic_series.mean()
    ic_std = ic_series.std()
    ir = ic_mean / ic_std if ic_std > 0 else 0

    # 6. IC显著性检验
    n = len(data.loc[date])  # 横截面股票数
    t_stat = ic_mean * np.sqrt(n / (1 - ic_mean**2))
    p_value = 2 * (1 - t.cdf(abs(t_stat), df=n-2))

    # 7. 分组回测
    quintile_returns = quintile_backtest(data, n_groups=5)

    return {
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'ir': ir,
        't_stat': t_stat,
        'p_value': p_value,
        'quintile_returns': quintile_returns
    }
```

### 3.2 衰减周期测试

```python
def decay_analysis(factor_name, max_horizon=60):
    """
    测试因子在不同Horizon下的IC
    """
    ic_by_horizon = []

    for h in range(1, max_horizon+1):
        ic = compute_ic(factor_name, horizon=h)
        ic_by_horizon.append(ic)

    # 找到最大IC的Horizon
    best_h = np.argmax(ic_by_horizon) + 1

    # 计算衰减率
    decay_rate = compute_decay_rate(ic_by_horizon)

    return {
        'ic_by_horizon': ic_by_horizon,
        'best_horizon': best_h,
        'decay_rate': decay_rate
    }

def compute_decay_rate(ic_series, threshold=0.5):
    """
    计算衰减率：IC从最大值衰减到50%所需的时间
    """
    max_ic = max(ic_series)
    half_max_ic = max_ic * threshold

    for i, ic in enumerate(ic_series):
        if ic < half_max_ic:
            return i + 1  # 返回衰减到50%所需的Horizon

    return len(ic_series)  # 如果没有衰减到50%，返回总长度
```

### 3.3 相关性矩阵与去重

```python
def factor_correlation_analysis(factor_list, threshold=0.9):
    """
    因子相关性分析与去重
    """
    # 计算相关系数矩阵
    corr_matrix = pd.DataFrame(index=factor_list, columns=factor_list)

    for i, f1 in enumerate(factor_list):
        for j, f2 in enumerate(factor_list):
            if i <= j:
                corr = compute_correlation(f1, f2)
                corr_matrix.loc[f1, f2] = corr
                corr_matrix.loc[f2, f1] = corr

    # 层次聚类去重
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    # 将相关系数转换为距离
    distance_matrix = 1 - corr_matrix.values
    distance_vector = squareform(distance_matrix)

    # 层次聚类
    Z = linkage(distance_vector, method='average')
    clusters = fcluster(Z, t=threshold, criterion='distance')

    # 每个聚类选择IC最高的因子
    cluster_dict = {}
    for i, cluster_id in enumerate(clusters):
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = []
        cluster_dict[cluster_id].append(factor_list[i])

    selected_factors = []
    for cluster_id, factors in cluster_dict.items():
        best_factor = max(factors, key=lambda f: get_ic(f))
        selected_factors.append(best_factor)

    return {
        'selected_factors': selected_factors,
        'corr_matrix': corr_matrix,
        'clusters': cluster_dict
    }
```

---

## 4. 数据泄漏防护

### 4.1 常见泄漏场景

**场景1：未来数据泄露**

```python
# 错误：计算t时刻的波动率，使用了未来数据
def wrong_volatility(price, window):
    result = np.zeros_like(price)
    for i in range(len(price)):
        result[i] = price[i:i+window].std()  # 包含i+1到i+window-1的数据
    return result

# 正确：只使用历史数据
def correct_volatility(price, window):
    result = np.zeros_like(price)
    for i in range(window-1, len(price)):
        result[i] = price[i-window+1:i+1].std()  # 只使用i-window+1到i的数据
    return result

# 更优：使用Qlib的Rolling算子
def qlib_volatility():
    return Expression(Std($close, 20))  # Qlib自动处理时序
```

**场景2：样本外信息混入**

```python
# 错误：在整个数据集上标准化
def wrong_standardization(X_train, X_test):
    X_all = np.concatenate([X_train, X_test], axis=0)
    scaler = StandardScaler().fit(X_all)  # 使用了测试集信息
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# 正确：只在训练集上标准化
def correct_standardization(X_train, X_test):
    scaler = StandardScaler().fit(X_train)  # 只使用训练集信息
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled
```

### 4.2 检测方法

```python
def detect_lookahead_bias(factor_df, price_df, horizon=5):
    """
    检测Look-ahead Bias
    """
    # 方法1：检查因子是否与未来价格相关
    lookahead_corr = []

    for lag in [-1, -2, -5, -10]:
        shifted_factor = factor_df.shift(lag)
        corr = shifted_factor.corrwith(price_df)
        lookahead_corr.append(corr)

    # 如果shift(-1)相关性显著 > IC，说明有泄漏
    ic_mean = compute_ic(factor_df, price_df, horizon=horizon)
    if lookahead_corr[0].mean() > 2 * ic_mean:
        warn("Potential lookahead bias detected!")
        return False

    # 方法2：检查因子分布突变
    factor_diff = factor_df.diff().abs()
    threshold = factor_diff.std() * 5
    if (factor_diff > threshold).any():
        warn("Factor has extreme jumps, check for data leaks!")
        return False

    return True
```

### 4.3 最佳实践清单

- [ ] 因子计算只用历史数据（$t$ 时刻的因子只用 $<t$ 的数据）
- [ ] 标签使用shift(-h)对齐（将未来收益对齐到当前时刻）
- [ ] 测试集严格在训练集之后（时间序列划分）
- [ ] 避免使用未来统计量（Rolling、Lag算子）
- [ ] 定期进行泄漏检测（detect_lookahead_bias）
- [ ] 审计日志记录每一步操作
- [ ] 样本外严格验证（不使用样本内信息调参）

---

## 5. 链上数据集成

### 5.1 链上数据特征工程挑战

与传统的金融数据相比，链上数据有以下独特挑战：

**挑战1：数据频率极高**

- 区块链交易是7×24小时实时发生的
- 数据频率可能达到秒级甚至毫秒级
- 传统Qlib框架主要用于日频数据，需要适配

**挑战2：数据维度复杂**

- Token级别（类似股票代码）
- 地址级别（用户行为）
- 协议级别（DeFi生态）
- 交易级别（逐笔分析）

**挑战3：数据质量参差不齐**

- 部分数据缺失（如隐私币、跨链桥）
- 数据清洗困难（Sybil攻击、女巫攻击）
- 数据来源多样（RPC、The Graph、Dune Analytics）

**挑战4：链上链下融合**

- 需要将链上数据与链下数据融合
- 时间戳对齐困难（区块链时间 vs 现实时间）
- 不同链的数据格式不统一

### 5.2 DeFi数据映射到Qlib（Token级别）

**Token级别的数据映射**

将Token视为"股票"，链上指标视为"因子"：

| Qlib概念 | 链上概念 | 示例 |
|---------|---------|------|
| 股票代码 | Token地址或Symbol | ETH, BTC, UNI, AAVE |
| 行业 | 链 | Ethereum, BSC, Polygon |
| 市值 | Token市值 | Market Cap |
| 成交量 | 交易量 | Volume |
| 价格 | DEX价格 | Price from Uniswap |

**DEX Swap行为因子**

**因子1：Swap Volume Momentum（交易量动量）**

```python
# Qlib Expression伪代码
swap_momentum = (
    Ref($swap_volume, -5) / $swap_volume - 1
)
```

**金融含义**：
- 如果交易量在5天内增长了50%，说明市场关注度高
- 动量因子通常为正，说明未来可能继续上涨

**因子2：Price Velocity（价格变化速率）**

```python
# Qlib Expression伪代码
price_velocity = (
    ($price / Ref($price, 1)) - 1
) / sqrt(1 + $gas_price)  # 归一化Gas成本
```

**金融含义**：
- 价格变化速率考虑了Gas成本（交易成本）
- 如果价格涨了10%，但Gas成本很高，实际净收益可能很低

**因子3：Liquidity Depth（流动性深度）**

```python
# Qlib Expression伪代码
liquidity_depth = (
    $liquidity_0side + $liquidity_1side
) / $price
```

**金融含义**：
- 流动性深度越大，滑点越小，交易成本越低
- 流动性深度是DeFi协议健康度的重要指标

**因子4：Slippage Ratio（滑点比率）**

```python
# Qlib Expression伪代码
slippage = (
    ($price - $twap_price) / $twap_price
) * $trade_size
```

**金融含义**：
- 滑点比率衡量大额交易对价格的影响
- 滑点大说明流动性差，不适合大额交易

### 5.3 链上指标集成（地址级别→聚合）

**地址级别的因子**

**因子1：Daily Active Addresses（DAA）动量**

```python
# Qlib Expression伪代码
daa_momentum = (
    Ref($daily_active_addresses, -7) /
    $daily_active_addresses - 1
)
```

**金融含义**：
- 活跃地址数7天内增长了50%，说明用户参与度高
- DAA是网络效应的重要指标

**因子2：Transaction Velocity（交易活跃度）**

```python
# Qlib Expression伪代码
tx_velocity = (
    $transaction_count / $total_addresses
)
```

**金融含义**：
- 每个地址的平均交易次数
- 交易活跃度高说明Token流通性好

**因子3：Whale Activity（大户活动）**

```python
# Qlib Expression伪代码
whale_activity = (
    sum($transaction_size where $transaction_size > 1000 ETH) /
    $total_volume
)
```

**金融含义**：
- 大户（Whale）交易占比
- 大户活动多可能是"聪明钱"进场

### 5.4 协议级别因子

**因子1：TVL（Total Value Locked）增长**

```python
# Qlib Expression伪代码
tvl_growth = (
    Ref($protocol_tvl, -30) / $protocol_tvl - 1
)
```

**金融含义**：
- TVL 30天内增长了50%，说明协议发展迅速
- TVL是DeFi协议健康度的核心指标

**因子2：Staking APY（质押收益率）**

```python
# Qlib Expression伪代码
apy_factor = (
    $staking_apy / $risk_free_rate
)
```

**金融含义**：
- 质押收益率相对于无风险收益率的倍数
- APY高说明协议激励性强，可能吸引更多用户

### 5.5 交易级别因子（高频分析）

**因子1：Order Book Imbalance（订单簿失衡）**

```python
# Qlib Expression伪代码
order_book_imbalance = (
    ($bid_volume - $ask_volume) /
    ($bid_volume + $ask_volume)
)
```

**金融含义**：
- 订单簿失衡=1表示全是买单，= -1表示全是卖单
- 订单簿失衡大说明市场情绪强（买方或卖方压倒性优势）

**因子2：Trade Size Distribution（交易规模分布）**

```python
# Qlib Expression伪代码
whale_ratio = (
    sum($trade_size where $trade_size > 1000 ETH) /
    $total_volume
)
```

**金融含义**：
- 大额交易占比
- 大额交易多可能是"聪明钱"或"鲸鱼"活动

### 5.6 链上-传统数据融合

**融合案例1：CEX-DEX套利因子**

```python
# Qlib Expression伪代码
# 交易所价差
exchange_spread = (
    abs($cex_price - $dex_price) / $dex_price
)

# 套利信号（考虑流动性）
arbitrage_signal = (
    exchange_spread *
    min($cex_liquidity, $dex_liquidity) /  # 流动性限制
    (1 + $gas_price)  # Gas成本
)
```

**金融含义**：
- CEX和DEX价差大，说明有套利机会
- 但需要考虑流动性限制和Gas成本
- 如果价差=5%，但Gas成本=2%，净套利收益=3%

**融合案例2：On-chain Momentum + Off-chain Volume**

```python
# Qlib Expression伪代码
# 混合动量因子
hybrid_momentum = (
    0.6 * ($on_chain_return) +
    0.4 * ($off_chain_volume / Ref($off_chain_volume, -1))
)
```

**金融含义**：
- 链上收益占60%，链下成交量占40%
- 结合链上和链下信息，提高因子稳定性

**融合案例3：Cross-chain Arbitrage（跨链套利）**

```python
# Qlib Expression伪代码
# 跨链价差
chain_spread = (
    abs($eth_chain_price - $bsc_chain_price) /
    $eth_chain_price
)

# 跨桥时间
bridge_time = ($block_time_target - $block_time_source)

# 跨桥费用
bridge_fee = $bridge_gas * $gas_price

# 套利收益
arbitrage_profit = (
    chain_spread -
    bridge_fee / $trade_size -  # 跨桥费用
    bridge_time * $time_decay  # 时间衰减
)
```

**金融含义**：
- 跨链价差大，说明有套利机会
- 但需要考虑跨桥时间、跨桥费用
- 如果价差=10%，但跨桥费用=5%，跨桥时间=10小时（时间衰减=2%），净套利收益=3%

### 5.7 链上数据集成实践总结

**链上数据集成的三个层次**：

**层次1：Token级别（类似股票）**
- 将Token映射为"股票代码"
- 链上指标映射为"因子"
- 适合传统量化框架

**层次2：地址级别（用户行为）**
- 将地址行为聚合为Token级别因子
- 关注大户、活跃地址、交易活跃度
- 适合用户行为分析

**层次3：协议级别（DeFi生态）**
- 将协议指标映射为"行业因子"
- 关注TVL、APY、使用率
- 适合DeFi生态分析

**链上数据集成的四个维度**：

**维度1：Token级别（资产维度）**
- 价格、市值、成交量
- DEX Swap行为
- 跨链套利机会

**维度2：地址级别（用户维度）**
- 活跃地址数
- 大户活动
- 用户留存率

**维度3：协议级别（生态维度）**
- TVL增长
- APY激励
- 使用率

**维度4：交易级别（微观维度）**
- 订单簿失衡
- 交易规模分布
- 滑点分析

**链上数据融合的三个方向**：

**方向1：链上-链下融合**
- CEX-DEX价差
- 链上收益 + 链下成交量
- 跨链套利

**方向2：多链融合**
- Ethereum + BSC + Polygon
- 跨链价差
- 跨链流动性

**方向3：链上-链上融合**
- Token价格 + 流动性
- DEX Swap + 链上指标
- 协议TVL + APY

---

## 6. 完整项目模板

### 6.1 目录结构

```
qlib-factor-project/
├── config/                    # 配置文件
│   ├── baseline.yaml         # 基础模板
│   ├── advanced.yaml         # 进阶模板
│   └── on_chain.yaml         # 链上数据模板
├── data/                      # 数据目录
│   ├── qlib_data/            # Qlib数据
│   └── on_chain_data/        # 链上数据
├── factors/                   # 因子定义
│   ├── expressions.py        # Qlib表达式
│   ├── on_chain_factors.py   # 链上因子
│   └── traditional_factors.py # 传统因子
├── evaluation/                # 评估模块
│   ├── ic_analysis.py        # IC分析
│   ├── backtest.py           # 回测
│   └── report.py             # 报告生成
├── utils/                     # 工具函数
│   ├── data_loader.py        # 数据加载
│   ├── normalization.py      # 标准化
│   └── leakage_detector.py   # 泄漏检测
├── logs/                      # 日志目录
└── main.py                   # 主程序
```

### 6.2 主程序模板

```python
# main.py

from qlib import init
from factors.expressions import load_traditional_factors
from factors.on_chain_factors import load_on_chain_factors
from evaluation.ic_analysis import evaluate_all_factors
from evaluation.report import generate_report

def main():
    # 1. 初始化Qlib
    init(provider_uri="data/qlib/qlib_data/cn_data")

    # 2. 加载传统因子
    traditional_factors = load_traditional_factors(
        config_path="config/advanced.yaml"
    )

    # 3. 加载链上因子
    on_chain_factors = load_on_chain_factors(
        tokens=["ETH", "BTC", "UNI", "AAVE"],
        config_path="config/on_chain.yaml"
    )

    # 4. 合并因子
    all_factors = {**traditional_factors, **on_chain_factors}

    # 5. 评估因子
    results = evaluate_all_factors(
        factors=all_factors,
        start_time="2020-01-01",
        end_time="2023-12-31"
    )

    # 6. 生成报告
    generate_report(results, output_path="logs/factor_report.html")

    print("Factor evaluation completed!")

if __name__ == "__main__":
    main()
```

### 6.3 配置文件模板

```yaml
# config/on_chain.yaml

# 数据源
data:
  on_chain:
    source: "dune_api"  # 或 "the_graph", "custom_rpc"
    chains: ["ethereum", "bsc", "polygon"]
    tokens: ["ETH", "BTC", "UNI", "AAVE", "LINK"]

  metrics:
    # Token级别
    - name: "swap_volume"
      type: "token_level"
      frequency: "1d"

    - name: "liquidity_depth"
      type: "token_level"
      frequency: "1d"

    # 地址级别
    - name: "daily_active_addresses"
      type: "address_level"
      frequency: "1d"

    - name: "whale_activity"
      type: "address_level"
      frequency: "1d"

    # 协议级别
    - name: "protocol_tvl"
      type: "protocol_level"
      frequency: "1d"

# 因子定义
factors:
  - name: "swap_volume_momentum"
    expr: "Ref($swap_volume, -5) / $swap_volume - 1"
    category: "momentum"

  - name: "liquidity_depth_ratio"
    expr: "$liquidity / $market_cap"
    category: "liquidity"

  - name: "daa_momentum"
    expr: "Ref($daily_active_addresses, -7) / $daily_active_addresses - 1"
    category: "on_chain"

  - name: "whale_activity_ratio"
    expr: "$whale_volume / $total_volume"
    category: "on_chain"

  - name: "tvl_growth"
    expr: "Ref($protocol_tvl, -30) / $protocol_tvl - 1"
    category: "protocol"

# 因子组合
combination:
  on_chain_weight: 0.4
  off_chain_weight: 0.6
  method: "ic_weighted"

# 回测配置
backtest:
  start_time: "2022-01-01"
  end_time: "2023-12-31"
  rebalance_freq: "1d"  # 链上数据变化快，日频调仓
  min_liquidity: 100000  # 最小流动性
  max_gas_price: 50  # 最大Gas价格（Gwei）

# 数据泄漏防护
leakage_protection:
  enable: true
  check_lookahead: true
  check_future_info: true
  audit_log: true
```

---

## 总结

本文从实践角度提供了完整的Qlib特征工程指南，涵盖了从数据优化到链上集成的各个层面。

### 核心要点回顾

1. **特征张量优化**：
   - 内存优化：数据类型、稀疏矩阵、分片加载
   - 计算优化：向量化、并行化、增量计算
   - 存储优化：HDF5、Parquet、索引策略

2. **Pipeline配置**：
   - 基础模板：行情因子
   - 进阶模板：多因子组合、中性化
   - 链上模板：多链、多协议、多维度

3. **因子质量评估**：
   - IC/IR分析：相关性、稳定性、显著性
   - 衰减周期测试：不同Horizon的IC表现
   - 相关性矩阵与去重：层次聚类、IC加权

4. **数据泄漏防护**：
   - 常见泄漏场景：未来数据、样本外信息
   - 检测方法：相关性检验、分布突变检测
   - 最佳实践清单：7项关键检查

5. **链上数据集成**：
   - Token级别：DEX Swap行为因子
   - 地址级别：活跃地址、大户活动
   - 协议级别：TVL、APY因子
   - 交易级别：订单簿、滑点分析
   - 链上-链下融合：CEX-DEX套利、混合因子

### 链上数据集成的核心洞察

链上数据集成不仅是技术问题，更是思维模式的转变：

**从单维度到多维度**：
- 传统：股票（单维度）
- 链上：Token × 地址 × 协议 × 交易（四维度）

**从静态到动态**：
- 传统：日频数据（静态）
- 链上：7×24小时实时数据（动态）

**从单一到融合**：
- 传统：传统金融数据（单一）
- 链上：链上 × 链下 × 跨链（融合）

**从被动到主动**：
- 传统：被动接受数据
- 链上：主动构建数据管道（RPC、The Graph、Dune）

### 实践建议

**对于传统量化**：
- 关注IC/IR，而不是绝对收益
- 横截面标准化和中性化是必须的
- 严格防护数据泄漏

**对于链上量化**：
- 四维度分析：Token × 地址 × 协议 × 交易
- 三个融合方向：链上-链下、多链、链上-链上
- 动态数据管道：实时数据流处理

**对于项目构建**：
- 模块化设计：配置、因子、评估、工具分离
- 日志审计：记录每一步操作
- 持续迭代：因子库动态更新

### 未来展望

随着Web3和DeFi的快速发展，链上数据量化将成为量化投资的新前沿。Qlib的特征工程框架不仅适用于传统金融，也可以扩展到链上数据分析，为量化投资提供系统化的工具和思维。

横截面标准化不是数学技巧，而是一种投资策略：
> 我们不赌国运涨跌，也不赌行业轮动，我们只赌"同一环境下，谁比谁强"。

链上数据集成不是技术问题，而是一种新范式：
> 我们不局限于传统金融数据，我们融合链上、链下、跨链的多维信息。

---

至此，Qlib特征工程的五个核心文档全部完成。从理论基础到实践指南，从传统金融到链上数据，形成了一个完整的知识体系。

希望这些文档能够帮助你在量化投资的道路上走得更远、更稳。
