# Mandate 阶段 - 横截面因子研究授权与边界冻结

## 目录
1. [阶段定义](#阶段定义)
2. [为什么需要 Mandate 阶段](#为什么需要-mandate-阶段)
3. [核心内容要素](#核心内容要素)
4. [横截面因子研究的特殊考虑](#横截面因子研究的特殊考虑)
5. [Formal Gate 要求](#formal-gate-要求)
6. [Audit Gate 检查项](#audit-gate-检查项)
7. [常见错误和反模式](#常见错误和反模式)
8. [实际案例：加密货币动量因子研究](#实际案例加密货币动量因子研究)
9. [输出 Artifact 规范](#输出-artifact-规范)
10. [与下一阶段的交接标准](#与下一阶段的交接标准)

---

## 阶段定义

### Mandate（研究授权）定义

**Mandate** 是横截面因子研究流程的第一个正式阶段，在做任何数据分析之前，先把研究问题的边界和范围冻结下来。

这一步要解决四件事：

1. **明确因子定义**：清楚定义要研究的具体因子是什么
2. **冻结标的池**：确定哪些资产进入横截面比较
3. **设定预测期**：明确因子预测未来收益的时间窗口
4. **建立可验证假设**：确保因子有效性可以被客观检验

### 核心目标

| 目标 | 横截面研究特有含义 | 价值 |
|------|------------------|------|
| **明确因子主问题** | 该因子是否能预测横截面收益差异？ | 避免方向错误 |
| **防止标的池漂移** | 研究中途不能悄悄添加/删除标的 | 避免选择偏差 |
| **冻结预测期** | 因子与未来收益的时间关系固定 | 防止前视偏差 |
| **建立检验标准** | 什么情况算因子"有效" | 客观评估 |

### 阶段定位

```
研究想法 → Mandate → DataReady → PanelReady → FactorReady → ...
           ↑
      在这里冻结
```

Mandate 是横截面因子研究的宪法——一旦通过，后续所有工作都必须在 Mandate 定义的框架内进行。

---

## 为什么需要 Mandate 阶段

### 横截面因子研究的特有风险

**风险场景 1：标的池悄悄膨胀**

```python
# Mandate 阶段定义
universe = top_50_by_volume()  # Top 50 币种

# 研究中途发现效果不好
universe = top_100_by_volume()  # 悄悄扩大到 Top 100

# 或者
universe = [c for c in top_50 if c.sharpe > 0]  # 按结果筛选
```

**后果**：无法确定因子有效性是来自因子本身还是标的筛选

**风险场景 2：预测期事后优化**

```
初始定义：1 小时预测期
发现效果不好 → 改为 4 小时
还是不好 → 改为 24 小时
终于找到显著结果 → 声称"早就知道24小时最好"
```

**后果**：过拟合历史数据，样本外失效

**风险场景 3：因子定义悄悄改变**

```
Mandate: 研究"简单动量因子" = 当前价格 / 20日前价格
实现: 加入各种过滤条件、波动率调整、
       自适应窗口、混合其他信号...
实际结果: 变成了"复杂混合因子"，
          但仍然报告为"动量因子"
```

**后果**：无法解释因子来源，失去可解释性

### Mandate 怎么解决这些问题

Mandate 阶段的做法是：

1. 提前冻结标的池，避免事后选择偏差
2. 固定预测期，防止挖掘到虚假的时间关系
3. 明确因子计算方式，确保不同实现得到相同结果
4. 建立检验标准，避免"直到找到显著结果为止"

---

## 核心内容要素

### 1. 定义目标市场

**目标市场**规定了横截面比较的范围——我们在哪些资产之间做比较。

#### 规范格式

```yaml
目标市场定义:
  市场类型: "加密货币现货市场"
  
  准入口径:
    - "CoinMarketCap 24小时交易量 Top N"
    - "排除稳定币（USDT, USDC, DAI, ...）"
    - "排除杠杆代币（UP/DOWN 系列）"
    - "至少有 X 天历史数据"
  
  基线数量: N 个交易对
  
  更新规则:
    - "Mandate 冻结后不再更新"
    - "新上市币种不纳入"
    - "退市币种保留历史数据"
```

#### 示例

```yaml
目标市场: 加密货币 Top 50 流动性池
  
  准入口径:
    - "Binance 现货市场 24h 交易量 Top 50"
    - "排除所有稳定币"
    - "排除杠杆 ETF 代币"
    - "至少 180 天历史数据"
    - "当前价格 > $0.01（排除垃圾币）"
  
  基线集合: 42 个交易对
  完整列表: 见 universe_manifest.csv
  
  明确排除:
    稳定币:
      - "USDT, USDC, DAI, FDUSD, ..."
    杠杆代币:
      - "BTCUP, ETHDOWN, ..."
    低流动性:
      - "24h 交易量 < $100万 的币种"
```

#### 质量检查

| 检查项 | 标准 | 状态 |
|--------|------|------|
| 准入口径清晰 | 5分钟能向新人解释清楚 | ✅ |
| 可复现 | 给定数据能重建相同集合 | ✅ |
| 无歧义 | 边界情况有明确规则 | ✅ |
| 已记录 | universe_manifest.csv 存在 | ✅ |

### 2. 定义横截面任务

横截面任务要回答的是：因子要预测什么？我们在比较什么？

#### 核心问题格式

```
在 [目标市场] 中，[因子名称] 是否能够预测
[未来 H 期] 的 [横截面收益排序]？
```

#### 示例

```yaml
研究主问题:
  文本: >
    在加密货币 Top 50 流动性池中，
    20日动量因子是否能够预测
    未来 24 小时的横截面收益排序？
  
  假设方向: >
    �量量高的资产未来收益也高
    （正动量溢价）
  
  主要评估指标: >
    - IC (Information Coefficient)
    - Rank IC
    - 分位数收益单调性
  
  对比基准: >
    - 等权重组合
    - 市值加权组合
```

#### 任务类型

横截面因子研究常见的任务类型：

```yaml
任务类型:
  
  收益预测:
    问题: "因子能否预测未来收益？"
    标签: "未来 N 期收益率"
    评估: "IC, Rank IC, 分位数收益"
    示例: "动量因子预测未来24h收益"
  
  风险预测:
    问题: "因子能否预测未来波动？"
    标签: "未来 N 期波动率"
    评估: "与实际波动的相关性"
    示例: "波动率因子预测未来风险"
  
  分类任务:
    问题: "因子能否区分赢家/输家？"
    标签: "二分类（涨/跌）"
    评估: "AUC, 准确率"
    示例: "情绪因子预测涨跌"
```

### 3. 定义 Label Horizon

**Label Horizon** 是因子与预测目标之间的时间间隔，也是横截面因子研究中最关键的参数之一。

#### Horizon 设计原则

```python
# Horizon 必须在 Mandate 中冻结
horizon_config = {
    'primary_horizon': '24h',      # 主要预测期
    'secondary_horizons': ['1h', '4h', '168h'],  # 次要探索期
    
    '对齐方式': '向前对齐到整点',
    '标签计算': 'log_return',
    '缺失处理': '排除当日观测'
}
```

#### Horizon 选择考虑

| Horizon | 适用场景 | 风险 |
|---------|---------|------|
| **1小时** | 高频交易因子 | 噪声大、交易成本高 |
| **24小时** | 日内策略 | 平衡信噪比和交易频率 |
| **7天** | 中期策略 | 样本量少、滞后性强 |
| **30天** | 长期策略 | 数据不足、过拟合风险 |

#### 实战建议

```yaml
Horizon 设计最佳实践:
  
  从假设出发:
    好的: "基于市场微观结构，认为24h合理"
    坏的: "尝试多个horizon，选最好的"
  
  考虑交易成本:
    短horizon: "换手率高，成本侵蚀收益"
    长horizon: "换手率低，但信号衰减"
  
  保持一致性:
    因子计算时间点 → 标签计算时间点
    必须清晰定义，避免前视偏差
```

#### Horizon 规范示例

```yaml
Label Horizon 定义:
  
  主要Horizon: 24小时
  
  计算规则:
    1. T时刻获取因子值（使用T时刻及之前数据）
    2. 计算(T+24h)的收益率
    3. 收益率 = log(price_T+24h) - log(price_T)
  
  边界条件:
    - 如果T+24h价格缺失: 排除该观测
    - 如果期间发生退市: 使用最后可用价格
    - 如果期间发生分叉: 按照处理规则计算
  
  对齐规范:
    - 时间戳对齐到整点
    - 使用UTC时间
    - 交易所交易时段统一处理
```

### 4. 定义成功标准

**成功标准**要回答一个问题：什么情况下我们认为因子"有效"？

#### 多层次评估体系

```yaml
成功标准层次:
  
  统计显著性:
    指标:
      - "IC t-stat > 2"
      - "Rank IC p-value < 0.05"
    阈值: >
      IC: > 0.03 (弱), > 0.05 (中), > 0.08 (强)
      Rank IC: > 0.02 (弱), > 0.04 (中), > 0.06 (强)
  
  经济显著性:
    指标:
      - "多空组合年化收益"
      - "多空组合夏普比率"
    阈值: >
      年化收益: > 10%
      夏普比率: > 1.0
  
  稳健性:
    指标:
      - "跨时期稳定性"
      - "分子样本稳定性"
    要求: >
      至少 2/3 的月份 IC 符号一致
      Top/Bottom 分位数收益差距稳定
  
  实用性:
    指标:
      - "换手率"
      - "容量估计"
    要求: >
      日换手率 < 50%
      可承载资金 > $100万
```

#### 通过/失败判定

```python
def evaluate_factor_success(ic_results, return_results, turnover):
    """
    因子成功判定逻辑
    """
    verdict = {
        'PASS': '所有核心标准满足',
        'CONDITIONAL_PASS': '核心标准满足，次要条件不满足',
        'RETRY': '统计显著但经济不显著',
        'NO_GO': '统计不显著或方向错误'
    }
    
    # 统计显著性（必须）
    stat_sig = (ic_results['mean_ic'] > 0.03 and 
                ic_results['t_stat'] > 2)
    
    # 经济显著性（必须）
    econ_sig = (return_results['long_short_sharpe'] > 1.0)
    
    # 稳健性（重要）
    robust = (return_results['pct_positive_months'] > 0.6)
    
    # 实用性（加分）
    practical = (turnover['daily'] < 0.5)
    
    # 判定
    if stat_sig and econ_sig:
        if robust and practical:
            return 'PASS'
        else:
            return 'CONDITIONAL_PASS'
    elif stat_sig:
        return 'RETRY'
    else:
        return 'NO_GO'
```

#### 成功标准配置示例

```yaml
成功标准配置:
  
  必须满足 (Formal Gate):
    统计检验:
      - "IC 均值 > 0.03"
      - "IC t-stat > 2"
      - "Rank IC p-value < 0.05"
    
    方向一致性:
      - "IC 符号与假设一致"
      - "至少 60% 时期 IC 符号一致"
  
  重要但非必须:
    经济显著性:
      - "Top-Bottom 分位数年化收益 > 10%"
      - "多空组合夏普 > 1.0"
    
    稳健性:
      - "滚动 3 月 IC 标准差 < 0.02"
      - "跨子时段稳定"
  
  加分项:
    实用性:
      - "日换手率 < 30%"
      - "估计容量 > $500万"
    
    可解释性:
      - "因子有明确经济逻辑"
      - "与已知因子相关性 < 0.7"
```

---

## 横截面因子研究的特殊考虑

### 横截面 vs 时间序列

```yaml
研究类型对比:
  
  时间序列策略:
    问题: "这个资产未来会涨吗？"
    比较: 同一资产不同时间点
    典型: "BTC 现在买入，未来会涨吗？"
    风险: 前视偏差、过拟合
  
  横截面因子:
    问题: "哪个资产未来表现更好？"
    比较: 同一时间点不同资产
    典型: "现在 BTC 和 ETH，哪个未来24h表现更好？"
    风险: 标的选择偏差、幸存者偏差
```

### Panel 数据的特殊性

横截面因子研究使用 Panel 数据（时间 x 标的），这带来了特殊的挑战：

```yaml
Panel 数据挑战:
  
  数据结构:
    维度: (时间点 × 标的)
    示例: "1000天 × 50币 = 50,000观测"
    
  缺失模式:
    时间缺失: "某天整个市场停牌"
    标的缺失: "新币上市前、退市后"
    随机缺失: "数据中断"
    
  处理原则:
    - "不能简单删除缺失"
    - "需要明确标记可用性"
    - "保持 Panel 结构完整"
```

### 因子计算的时间语义

横截面因子计算必须严格遵守"只用过去数据"的原则。

```python
# 正确的时间语义
def calculate_factor_at_t(panel_data, t):
    """
    计算 t 时刻的因子值
    
    关键: 只使用 t 及之前的数据
    """
    # ✅ 正确: 使用 t 时刻及之前的数据
    past_prices = panel_data['price'][:t]  # 到 t 为止
    factor = calculate_momentum(past_prices, window=20)
    
    # ❌ 错误: 使用了未来的数据
    # future_prices = panel_data['price'][t:t+24]
    # factor = calculate_with_future(past_prices, future_prices)
    
    return factor
```

---

## Formal Gate 要求

### FG-1: 目标市场清晰定义

```yaml
检查标准:
  - 准入口径无歧义
  - 基线集合完整列出
  - 明确排除项目说明
  - 提供可验证的标的清单

验收方式:
  - universe_manifest.csv
  - 准入口径的算法实现
  - 人工审查合理性
```

### FG-2: 横截面任务明确

```yaml
检查标准:
  - 研究问题符合规范格式
  - 明确预测目标（收益/风险/分类）
  - 包含因子、市场、Horizon 三要素
  - 可明确判断假设是否成立

验收方式:
  - 文档审查
  - 5分钟可理解性测试
```

### FG-3: Label Horizon 冻结

```yaml
检查标准:
  - 主要 Horizon 明确指定
  - Horizon 计算规则清晰
  - 边界条件处理完整
  - 没有预留"待定"或"可调整"

验收方式:
  - machine-readable Horizon 配置
  - 人工审查时间逻辑正确性
```

### FG-4: 成功标准可验证

```yaml
检查标准:
  - 统计显著性阈值明确
  - 经济显著性标准清晰
  - 稳健性要求可操作
  - 通过/失败判定规则完整

验收方式:
  - success_criteria.yaml
  - 可执行的评估脚本
```

### FG-5: 输出 Artifact 完整

```yaml
检查标准:
  - mandate.yml（机器可读配置）
  - mandate_summary.md（人类可读总结）
  - universe_manifest.csv（标的清单）
  - horizon_config.yaml（Horizon 配置）
  - success_criteria.yaml（成功标准）

验收方式:
  - 文件存在性检查
  - 配置文件格式验证
```

---

## Audit Gate 检查项

### AG-1: 因子经济逻辑

```yaml
目的: 记录为什么选择这个因子
记录内容:
  - 理论基础或市场观察
  - 预期作用机制
  - 与已知因子的关系

价值: 帮助理解因子来源
```

### AG-2: 数据可行性评估

```yaml
目的: 评估数据是否支持研究
评估内容:
  - 因子计算所需数据是否可得
  - 历史 Horizon 是否足够
  - 数据质量初步评估

价值: 提前识别数据风险
```

### AG-3: 容量初步估计

```yaml
目的: 粗略估计策略容量
评估内容:
  - 目标市场日均交易量
  - 预期资金占比
  - 流动性风险

价值: 判断研究是否值得投入
```

### AG-4: 文献/先例参考

```yaml
目的: 站在前人肩膀上
记录内容:
  - 相关学术论文
  - 团队历史研究
  - 业界最佳实践

价值: 避免重复已知失败的方向
```

---

## 常见错误和反模式

### 错误 1：目标市场模糊

**反模式**：
```
❌ "研究加密货币市场"
❌ "主要数字货币"
❌ "流动性好的币种"
```

**正确做法**：
```
✅ "Binance 现货 Top 50 by 24h 交易量"
✅ "排除稳定币和杠杆代币"
✅ "至少 180 天历史数据"
```

### 错误 2：Horizon 事后优化

**反模式**：
```python
# 尝试多个 Horizon
for horizon in [1, 4, 6, 12, 24, 48, 168]:
    ic = test_factor(horizon)
    if ic > best_ic:
        best_horizon = horizon
        best_ic = ic

# 声称"早就知道 48 小时最好"
```

**正确做法**：
```yaml
Mandate 阶段:
  基于: "市场微观结构和交易成本分析"
  决定: "主要 Horizon = 24 小时"
  冻结: "不允许事后修改"

次要分析:
  可以: "报告其他 Horizon 的结果"
  标记: "作为次要/探索性分析"
```

### 错误 3：成功标准太宽松

**反模式**：
```yaml
成功标准:
  - "IC > 0"（几乎总是满足）
  - "某一时期显著即可"
  - "调整后总能找到显著"
```

**正确做法**：
```yaml
成功标准:
  统计检验:
    - "IC t-stat > 2"（约 p < 0.05）
    - "全样本和各子样本都显著"
  
  稳健性:
    - "至少 60% 时期 IC 符号一致"
    - "不能只靠某一时期显著"
```

### 错误 4：忽略横截面结构

**反模式**：
```python
# 把横截面问题当作时间序列处理
for symbol in universe:
    # 对每个币单独分析
    test_time_series_factor(symbol)
```

**正确做法**：
```python
# 利用横截面结构
for t in time_points:
    # 每个时间点，比较所有币
    factor_values = calculate_factor_at_t(panel, t)
    future_returns = calculate_returns(panel, t, horizon)
    ic = spearmanr(factor_values, future_returns)
```

---

## 实际案例：加密货币动量因子研究

### Mandate 文档示例

```yaml
研究主问题:
  文本: >
    在 Binance 现货 Top 50 流动性池中，
    20日动量因子是否能够预测
    未来 24 小时的横截面收益排序？
  
  假设方向: "正动量溢价"
    预期: 过去 20 天表现好的资产，未来 24h 继续表现好
  
  主要评估指标:
    - IC (Information Coefficient)
    - Rank IC
    - Top/Bottom 分位数收益差

目标市场:
  市场类型: "加密货币现货市场"
  交易所: "Binance"
  
  准入口径:
    - "24 小时交易量 Top 50"
    - "排除所有稳定币"
    - "排除杠杆代币"
    - "至少 180 天历史数据"
    - "当前价格 > $0.01"
  
  基线数量: 42 个交易对
  完整列表: universe_manifest.csv
  
  明确排除:
    稳定币: "USDT, USDC, DAI, FDUSD, TUSD, ..."
    杠杆代币: "BTCUP, ETHDOWN, ..."

Label Horizon:
  主要 Horizon: "24 小时"
  
  计算规则:
    1. T 时刻获取因子值（使用 T 及之前数据）
    2. 计算 (T+24h) 的收益率
    3. 收益率 = log(P_T+24h) - log(P_T)
  
  对齐规范:
    - 时间戳对齐到整点
    - 使用 UTC 时间
    - 统一使用收盘价
  
  边界条件:
    - T+24h 价格缺失: 排除该观测
    - 期间退市: 使用最后可用价格
    - 期间分叉: 按比例调整

因子定义:
  因子名称: "20日动量"
  
  计算方法:
    momentum_t = (price_t - price_t-20) / price_t-20
  
  数据要求:
    - 至少 20 天历史价格
    - 使用日收盘价
  
  处理规则:
    - 历史不足 20 天: 排除当日该标的
    - 价格异常: 标记但不排除
    - 退市: 最后可用日后不再计算

时间窗:
  研究范围: 2021-01-01 至 2024-12-31
  
  时间切分:
    Train: 2021-01-01 至 2023-12-31 (3年)
    Test: 2024-01-01 至 2024-06-30 (6个月)
    Backtest: 2024-07-01 至 2024-09-30 (3个月)
    Holdout: 2024-10-01 至 2024-12-31 (3个月)
  
  切分原则:
    - 按日历时间顺序
    - 不允许重叠
    - Holdout 完全未参与设计

成功标准:
  必须满足 (Formal Gate):
    统计显著性:
      - "全样本 IC 均值 > 0.03"
      - "IC t-stat > 2.0"
      - "Rank IC p-value < 0.05"
    
    方向一致性:
      - "IC 符号与假设一致（正）"
      - "至少 60% 周 IC 符号一致"
  
  重要标准:
    经济显著性:
      - "Top-Bottom 分位数年化收益 > 10%"
      - "多空组合夏普 > 1.0"
    
    稳健性:
      - "Train/Test IC 符号一致"
      - "滚动 3 月 IC 标准差 < 0.03"

参数边界:
  可探索参数 (Train 阶段):
    - 回看窗口: [10, 20, 40, 60] 天
    - 平滑方法: [无, 5日MA, 10日MA]
  
  冻结参数:
    - 因子类型: 动量
    - 价格类型: 收盘价
    - Horizon: 24 小时
  
  禁止修改:
    - 研究主问题
    - 目标市场定义
    - Label Horizon
    - 成功标准

风险评估:
  数据风险:
    - "加密货币数据质量可能较差"
    - "交易所差异可能影响结果"
  
  过拟合风险:
    - "加密市场变化快，历史模式可能失效"
    - "样本外验证尤其重要"
  
  执行风险:
    - "24h 换手可能带来较高交易成本"
    - "流动性不足的币种难以执行"

Gate Decision: [待 DataReady 完成后评审]
```

---

## 输出 Artifact 规范

### 必需的输出文件

#### 1. mandate.yml（机器可读配置）

```yaml
mandate:
  version: "1.0"
  created_at: "2026-04-02"
  lineage_id: "momentum_crypto_xs_v1"
  research_type: "cross_sectional_factor"

research_question:
  text: "在 Binance Top 50 中，20日动量因子是否预测未来24h横截面收益？"
  hypothesis_direction: "positive"
  task_type: "return_prediction"
  primary_metric: "ic"
  secondary_metrics: ["rank_ic", "quantile_returns"]

target_market:
  exchange: "binance"
  market_type: "spot"
  eligibility_criteria:
    - "Top 50 by 24h volume"
    - "Exclude stablecoins"
    - "Exclude leveraged tokens"
    - "Min 180 days history"
    - "Price > $0.01"
  baseline_count: 42
  manifest_file: "universe_manifest.csv"

label_horizon:
  primary: "24h"
  calculation: "log_return"
  alignment: "hourly_utc"
  boundary_handling: "exclude_missing"

factor_definition:
  name: "momentum_20d"
  type: "momentum"
  calculation:
    lookback: 20
    price_type: "close"
  data_requirements:
    min_history: 20

time_window:
  study_start: "2021-01-01"
  study_end: "2024-12-31"
  splits:
    train: {start: "2021-01-01", end: "2023-12-31"}
    test: {start: "2024-01-01", end: "2024-06-30"}
    backtest: {start: "2024-07-01", end: "2024-09-30"}
    holdout: {start: "2024-10-01", end: "2024-12-31"}

success_criteria:
  statistical:
    min_ic: 0.03
    min_t_stat: 2.0
    max_p_value: 0.05
  economic:
    min_annual_return: 0.10
    min_sharpe: 1.0
  robustness:
    min_positive_weeks_pct: 0.60
```

#### 2. universe_manifest.csv（标的清单）

```csv
symbol,included,reason,listing_date,notes
BTC,TRUE,Baseline,2010-01-01,
ETH,TRUE,Baseline,2015-08-07,
BNB,TRUE,Baseline,2017-09-01,
SOL,TRUE,Baseline,2020-04-10,
...
USDT,FALSE,Stablecoin,2014-10-06,
USDC,FALSE,Stablecoin,2018-10-08,
BTCUP,FALSE,Leveraged token,2020-05-19,
```

#### 3. horizon_config.yaml（Horizon 配置）

```yaml
horizon:
  primary:
    hours: 24
    label: "return_24h"
  
  calculation:
    method: "log_return"
    formula: "log(price_T+24h) - log(price_T)"
  
  alignment:
    frequency: "1h"
    timezone: "UTC"
    point: "end_of_period"
  
  boundaries:
    missing_price: "exclude_observation"
    delisted: "use_last_available"
    forked: "proportional_adjustment"
```

#### 4. success_criteria.yaml（成功标准）

```yaml
success_criteria:
  version: "1.0"
  
  formal_gates:
    statistical_significance:
      mean_ic: {operator: ">", threshold: 0.03}
      ic_t_stat: {operator: ">", threshold: 2.0}
      rank_ic_p_value: {operator: "<", threshold: 0.05}
    
    direction_consistency:
      ic_sign_matches_hypothesis: true
      positive_weeks_pct: {operator: ">=", threshold: 0.60}
  
  important_criteria:
    economic_significance:
      top_bottom_annual_return: {operator: ">", threshold: 0.10}
      long_short_sharpe: {operator: ">", threshold: 1.0}
    
    robustness:
      train_test_ic_consistency: true
      rolling_ic_std: {operator: "<", threshold: 0.03}
  
  bonus_criteria:
    practicality:
      daily_turnover: {operator: "<", threshold: 0.30}
      estimated_capacity: {operator: ">", threshold: 500000}
    
    interpretability:
      economic_rationale_exists: true
      correlation_with_known_factors: {operator: "<", threshold: 0.7}
```

---

## 与下一阶段的交接标准

### DataReady 阶段的输入要求

| 项目 | 交付物 | 格式 | 用途 |
|------|--------|------|------|
| 研究范围 | mandate.yml | YAML | 指导数据需求 |
| 标的列表 | universe_manifest.csv | CSV | 数据提取范围 |
| Horizon | horizon_config.yaml | YAML | 标签计算 |
| 数据字段 | 因子定义 | 文本 | 确定需要的字段 |

### 交接验收标准

```yaml
DataReady 可以开始的条件:
  ✅ mandate.yml 通过格式验证
  ✅ universe_manifest.csv 包含所有必需字段
  ✅ horizon_config.yaml Horizon 合理
  ✅ 数据源明确且可访问
  ✅ 因子计算所需字段清晰

验收方式:
  - 自动化脚本验证配置文件格式
  - 人工审查研究范围合理性
  - 数据源可用性检查
```

---

**文档版本**: v1.0  
**最后更新**: 2026-04-02  
**维护者**: 量化研究团队
