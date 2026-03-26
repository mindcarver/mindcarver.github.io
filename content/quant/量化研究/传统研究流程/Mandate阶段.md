# Mandate 阶段 - 研究授权与边界冻结

## 目录
1. [阶段定义](#阶段定义)
2. [为什么需要 Mandate 阶段](#为什么需要-mandate-阶段)
3. [核心内容要素](#核心内容要素)
4. [必须冻结的内容清单](#必须冻结的内容清单)
5. [Formal Gate 要求](#formal-gate-要求)
6. [Audit Gate 检查项](#audit-gate-检查项)
7. [常见错误和反模式](#常见错误和反模式)
8. [实际案例](#实际案例)
9. [输出 Artifact 规范](#输出-artifact-规范)
10. [与下一阶段的交接标准](#与下一阶段的交接标准)

---

## 阶段定义

**Mandate（研究授权）**是量化研究流程的第一个正式阶段，也是最重要的阶段之一。它的核心目的是在开始任何数据分析之前，**冻结研究问题的边界和范围**。

### 核心目标

1. **明确研究主问题**：清楚定义"我们要回答什么问题"
2. **防止目标漂移**：避免研究过程中悄无声息地改变研究目标
3. **建立可验证的假设**：确保研究结论可以被客观验证
4. **划定研究边界**：明确什么在研究范围内，什么不在

### 阶段定位

```
研究想法 → Mandate → Data Ready → Signal Ready → ...
           ↑
      在这里冻结
```

Mandate 是所有后续阶段的**宪法**——一旦 Mandate 通过，后续阶段的所有工作都必须在 Mandate 定义的框架内进行。

---

## 为什么需要 Mandate 阶段

### 问题场景：研究目标悄无声息地改变

没有 Mandate 阶段时，常见的问题是：

**场景 1：目标漂移**
```
初始想法：研究动量信号在加密货币市场的有效性
中途发现：波动率信号看起来更有趣
最终结果：实际上研究的是波动率，但报告仍然说"动量研究"
```

**场景 2：范围膨胀**
```
初始定义：只研究主流币（BTC、ETH、主要稳定币）
中途发现：加入一些山寨币后收益更好
最终结果：无法确定收益是来自信号还是标的筛选
```

**场景 3：假设匹配**
```
先看到数据：发现某个时间段有很好表现
后定义假设：宣称"早就知道这个时间窗口有效"
最终结果：实际上是在过拟合历史数据
```

### Mandate 的解决方案

通过 Mandate 阶段，我们：

1. **提前写清楚**：在看到任何结果之前，把研究问题写下来
2. **冻结边界**：Universe、时间窗、信号机制在开始前就固定
3. **建立基线**：后续所有结果都相对于这个冻结的基线来评估
4. **防止作弊**：无法事后修改研究问题来匹配结果

### 量化研究中的类比

Mandate 就像临床试验中的**试验注册**：
- 你必须在开始试验前注册你要测试的假设
- 不能在看到结果后修改主要终点
- 所有次要分析都必须明确标记为探索性

---

## 核心内容要素

一个完整的 Mandate 必须包含以下五个核心要素：

### 1. 研究主问题（Research Question）

**定义**：研究要回答的单一、明确的问题。

**规范格式**：
```
在 [Universe] 中，使用 [信号机制]，
在 [时间窗] 上，是否能够获得 [预期方向] 的收益？
```

**示例**：
```
"在 Top 50 加密货币交易对中，
使用 20 日动量信号，
在 2020-2024 年期间，
是否能够获得正风险调整收益？"
```

**质量标准**：
- ✅ 具体可测：能够明确判断"是"或"否"
- ✅ 单一问题：一个 Mandate 只回答一个问题
- ✅ 可证伪：存在可能证明假设不成立的证据
- ❌ 模糊表达："研究各种信号的效果"
- ❌ 多重问题："动量和均值回归哪个更好"

### 2. 时间窗（Time Window）

**定义**：研究的起止时间，以及如何切分 Train/Test/Backtest/Holdout。

**必须指定**：
```yaml
整体研究窗:
  开始: 2020-01-01
  结束: 2024-12-31

时间切分:
  Train: 2020-01-01 → 2022-12-31
  Test: 2023-01-01 → 2023-12-31
  Backtest: 2024-01-01 → 2024-06-30
  Holdout: 2024-07-01 → 2024-12-31

切分原则:
  - 按日历时间切分，不允许随机切分
  - Train 不得少于 2 年数据
  - Test 必须是完整的日历年
  - Holdout 必须完全未参与任何设计
```

**为什么重要**：
- 时间序列数据不能随机切分
- 必须保证未来数据不会泄漏到过去
- Holdout 提供最终的真实性检验

### 3. Universe（标的集合）

**定义**：研究对象的具体范围。

**规范格式**：
```yaml
Universe 定义:
  准入口径:
    - CoinMarketCap Top 50 by 24h volume
    - 稳定币排除
    - 必须有至少 1 年历史数据

  更新频率:
    Mandate 冻结后不再更新

  基线集合:
    BTC, ETH, BNB, SOL, ADA, XRP, DOT, ...
    (完整列表见附件 universe_manifest.csv)

  明确排除:
    - 稳定币（USDT, USDC, ...）
    - 杠杆代币（BTCUP, ETHDOWN, ...）
    - 缺乏流动性的长尾币
```

**常见错误**：
- ❌ "主要加密货币"（太模糊）
- ❌ 研究中途悄悄添加新币
- ❌ 根据结果筛选标的

### 4. 信号机制（Signal Mechanism）

**定义**：要研究的具体信号或策略逻辑。

**规范描述**：
```yaml
信号名称: 20 日动量信号

计算方法:
  1. 获取过去 20 天的收盘价序列
  2. 计算 momentum = (今日价格 - 20 日前价格) / 20 日前价格
  3. 按 momentum 分为 5 个分位组

信号逻辑:
  - Long: momentum 最高分位
  - Short: momentum 最低分位（可选）
  - Rebalance: 每周调整

参数:
  - 回看窗口: 20 天
  - 分位组数: 5
  - 调整频率: 每周

边界条件:
  - 最少历史天数: 20 天
  - 缺失处理: 排除当日
```

**关键要求**：
- 完全可复现：不同人实现应得到相同结果
- 明确边界：处理异常情况的规则
- 参数明确：所有参数值在 Mandate 中固定

### 5. 参数边界（Parameter Boundaries）

**定义**：允许探索的参数范围，以及不允许触碰的边界。

**示例**：
```yaml
可探索参数:
  回看窗口: [10, 20, 40, 60] 天
  分位组数: [3, 5, 10]
  调整频率: [每日, 每周, 每月]

冻结参数:
  - 信号类型：动量（不允许改用其他信号）
  - 基础价格：收盘价（不允许用 OHLC 其他价格）
  - Universe：不允许事后筛选

禁止修改:
  - 研究主问题
  - Universe 定义
  - 时间切分
  - 信号的基本逻辑
```

**区别说明**：
- **可探索参数**：在 Train 阶段可以优化的参数
- **冻结参数**：Mandate 通过后不能修改的参数
- **禁止修改**：任何阶段都不能改的核心设定

---

## 必须冻结的内容清单

### 冻结原则

**什么是冻结**？冻结意味着在 Mandate 通过后，这些内容**不能以任何理由修改**，除非：
1. 发现了实现 bug（需要 Rollback 流程）
2. 开启新的 Child Lineage（新的研究方向）

### 完整冻结清单

| 类别 | 冻结内容 | 冻结后可以做什么 |
|------|----------|------------------|
| **研究问题** | 研究主问题的准确表述 | 可以添加次要分析，但不能改主问题 |
| **Universe** | 准入口径和基线集合 | 可以在 Test 阶段淘汰坏标的，不能添加新标的 |
| **时间窗** | 整体研究窗和切分点 | 不允许修改任何时间边界 |
| **信号逻辑** | 信号的基本定义和计算方法 | 可以优化参数，不能改变信号类型 |
| **假设方向** | 预期的收益方向（正/负） | 不能看到负收益后改说"我们早就知道" |
| **评估指标** | 主要评估指标（如 Sharpe） | 可以添加辅助指标，不能换主要指标 |
| **数据源** | 使用的数据源和版本 | 不能在研究中途换数据源 |

### 冻结时间点

```
Mandate 阶段开始 → 撰写 Mandate 文档 → 团队评审 → Mandate 通过
                                                           ↑
                                                     所有内容在此冻结
```

### 解冻的唯一途径

如果发现必须修改冻结内容，只有两条路径：

**路径 1：Rollback（回退）**
```yaml
适用场景: 发现实现 bug、数据 bug
流程:
  1. 记录 rollback_stage
  2. 明确 allowed_modifications
  3. 重新执行受影响的阶段
  4. 不能改变研究主问题
```

**路径 2：Child Lineage（子谱系）**
```yaml
适用场景: 研究方向发生实质变化
流程:
  1. 创建新的 lineage_id
  2. 从 Mandate 开始新研究线
  3. 两条谱系独立运行
  4. 不能自动替换主线
```

---

## Formal Gate 要求

Formal Gate 是**硬性要求**，不满足不能宣布 Mandate 完成。

### Gate 必需项

#### FG-1: 研究主问题清晰可测
```yaml
检查标准:
  - 问题表述符合规范格式
  - 可以明确判断假设是否成立
  - 单一问题，非多重问题
  - 包含 Universe、信号、时间窗三要素

验收方式:
  - 文档审查
  - 必须通过 5 分钟可理解性测试
```

#### FG-2: 时间窗完整冻结
```yaml
检查标准:
  - 明确指定整体研究窗起止时间
  - 完成 Train/Test/Backtest/Holdout 切分
  - 切分符合规范（时间顺序、长度要求）
  - 没有预留"待定"时间边界

验收方式:
  - machine-readable 时间窗配置文件
  - 人工审查切分合理性
```

#### FG-3: Universe 明确定义
```yaml
检查标准:
  - 准入口径清晰无歧义
  - 基线集合完整列出
  - 明确排除项目说明
  - 提供可验证的标的清单文件

验收方式:
  - universe_manifest.csv
  - 准入口径的算法实现（如脚本）
```

#### FG-4: 信号机制完整描述
```yaml
检查标准:
  - 计算方法分步骤描述
  - 所有参数明确指定或给出边界
  - 边界条件处理规则清晰
  - 不同人实现应得到相同结果

验收方式:
  - 信号逻辑的伪代码或公式
  - 参数边界表
```

#### FG-5: 输出 Artifact 完整
```yaml
检查标准:
  - mandate.yml（机器可读配置）
  - mandate_summary.md（人类可读总结）
  - universe_manifest.csv（标的清单）
  - time_window_config.json（时间窗配置）

验收方式:
  - 文件存在性检查
  - 配置文件格式验证
```

### Gate 决策术语

```yaml
PASS:
  所有 FG 满足，可以进入 Data Ready 阶段

CONDITIONAL PASS:
  核心要素满足，次要问题可以后续补充
  必须明确列出条件

RETRY:
  存在可以修复的问题，需要修改后重新提交

NO-GO:
  研究问题本身存在缺陷，不建议继续
```

---

## Audit Gate 检查项

Audit Gate 是**补充性检查**，用于记录和解释，但不直接阻断晋级。

### Audit 检查项

#### AG-1: 研究动机记录
```yaml
目的: 记录为什么选择这个研究问题
记录内容:
  - 研究背景和市场观察
  - 预期贡献或应用场景
  - 与现有研究的关系

价值: 帮助未来回顾研究动机
```

#### AG-2: 风险评估
```yaml
目的: 提前识别可能的研究风险
评估维度:
  - 数据可用性风险
  - 信号可计算性风险
  - 过拟合风险
  - 执行可行性风险

价值: 提前准备应对方案
```

#### AG-3: 资源需求评估
```yaml
目的: 估算研究所需资源
评估内容:
  - 计算资源需求
  - 数据存储需求
  - 预期耗时
  - 人力投入

价值: 帮助排期和优先级排序
```

#### AG-4: 文献/先例参考
```yaml
目的: 站在前人肩膀上
记录内容:
  - 相关学术论文
  - 团队历史研究
  - 业界最佳实践

价值: 避免重复已知失败的方向
```

### Audit 记录原则

```yaml
必需性:
  - 可以回答"不知道"，但不能不记录

用途:
  - 积累团队经验
  - 识别潜在问题
  - 不是晋级障碍

禁止:
  - 不能用 audit-only 发现偷换 formal gate
```

---

## 常见错误和反模式

### 错误 1：模糊的研究问题

**反模式**：
```
❌ "研究各种技术指标在加密货币市场的表现"
❌ "探索动量策略的优化空间"
❌ "寻找有效的交易信号"
```

**为什么错误**：
- 无法判断假设是否成立
- 范围太大，无法执行
- 容易在研究中途改变目标

**正确做法**：
```
✅ "在 Top 50 加密货币中，20 日动量信号
     在 2020-2024 年期间是否能够产生正风险调整收益？"
```

### 错误 2：Universe 悄悄改变

**反模式**：
```python
# Mandate 阶段
universe = top_50_by_volume()

# 研究中途发现
universe = top_100_by_volume()  # 俏俏扩大了

# 或者
universe = [c for c in top_50 if c.sharpe > 0]  # 按结果筛选
```

**后果**：
- 无法确定收益来自信号还是标的筛选
- 违反了 Mandate 冻结原则

**正确做法**：
- 在 Mandate 中明确 Universe
- 研究中途只可以淘汰坏标的，不能添加
- 如果确实需要改 Universe，开启 Child Lineage

### 错误 3：时间窗随意调整

**反模式**：
```
# Mandate 定义
Train: 2020-2021
Test: 2022

# 发现 Test 效果不好
Train: 2020-2022  # 悄悄把 Test 并入 Train
Test: 2023  # 重新选择 Test 窗口
```

**后果**：
- 样本外验证失效
- 本质上是在过拟合

**正确做法**：
- Mandate 冻结时间窗
- 如果时间窗确实有问题，Rollback 到 Mandate
- 重新定义后再继续

### 错误 4：信号逻辑中途改变

**反模式**：
```
Mandate: 研究"简单动量"信号
研究过程: 加入均值回归、波动率加权、
         自适应参数...
实际结果: 变成了"混合信号"，
          但仍然报告为"动量研究"
```

**后果**：
- 无法分离各种成分的贡献
- 失去了研究的清晰性

**正确做法**：
- Mandate 明确信号边界
- 如果发现需要新信号，创建新的 Mandate
- 可以在 Mandate 中预设"扩展方向"

### 错误 5：缺少参数边界

**反模式**：
```
Mandate: "研究动量信号"
实现: 随意尝试各种参数组合
      10天、20天、50天、100天...
      每日、每周、每月调整...
      加入各种过滤条件...
```

**后果**：
- 参数搜索空间无限
- 容易陷入过拟合
- 无法界定研究范围

**正确做法**：
```yaml
Mandate 明确:
  可探索参数:
    回看窗口: [10, 20, 40, 60]
    分位数: [3, 5, 10]
    调整频率: [每日, 每周]

  禁止:
    - 添加新的过滤条件
    - 引入其他信号
    - 改变信号基本逻辑
```

---

## 实际案例

### 案例 1：加密货币动量策略研究

#### Mandate 文档示例

```yaml
研究主问题:
  文本: 在 CoinMarketCap Top 50 加密货币中，
        基于 20 日动量信号构建的多空组合，
        在 2020-2024 年期间是否能够获得
        显著为正的风险调整收益？

  假设方向: 动量溢价为正
  主要评估指标: Sharpe Ratio

时间窗:
  研究范围: 2020-01-01 至 2024-12-31
  Train: 2020-01-01 至 2022-12-31 (3年)
  Test: 2023-01-01 至 2023-12-31 (1年)
  Backtest: 2024-01-01 至 2024-06-30 (6个月)
  Holdout: 2024-07-01 至 2024-12-31 (6个月)

Universe:
  准入口径:
    - CoinMarketCap 24h 交易量 Top 50
    - 排除稳定币（USDT, USDC, DAI, ...）
    - 排除杠杆代币（UP/DOWN 系列）
    - 至少 1 年历史数据

  基线数量: 42 个交易对
  完整列表: 见 universe_manifest.csv

  明确排除:
    - 所有稳定币
    - 杠杆代币
    - 2023 年后上市的新币

信号机制:
  信号名称: 20 日动量分位数

  计算步骤:
    1. 获取每个标的过去 20 天收盘价
    2. 计算动量: (P_t - P_{t-20}) / P_{t-20}
    3. 跨所有标的计算动量分位数（5 分位）
    4. Long 组: 分位数 Q5（动量最高）
    5. Short 组: 分位数 Q1（动量最低）

  构建规则:
    - 等权重组合
    - 每周 rebalance
    - 使用当日收盘价执行

  边界条件:
    - 数据不足 20 天: 排除当日该标的
    - 价格异常: 如果当日价格变动 > 50%，标记为异常
    - 退市处理: 最后一个交易日退出，不再计入

参数边界:
  可探索参数 (Train 阶段):
    - 回看窗口: [10, 20, 40, 60] 天
    - 分位组数: [3, 5, 10]
    - 调整频率: [每日, 每周, 每月]

  冻结参数:
    - 信号类型: 动量（不改用其他信号）
    - 基础价格: 收盘价
    - 组合构建: 等权重

  禁止修改:
    - 研究主问题
    - Universe 定义
    - 时间切分
    - 动量信号的基本逻辑

数据源:
  价格数据: Binance Spot OHLCV
  参考数据: CoinMarketCap (用于 Universe 构建)
  数据版本: v1.0 (2025-01-15 导出)

预期贡献:
  学术: 验证实证资产动量溢价在加密货币市场存在
  实践: 为多空策略提供基础信号

风险评估:
  - 动量崩溃风险: 市场反转时可能大幅回撤
  - 流动性风险: 部分 Top 50 标的流动性可能不足
  - 执行风险: 每周 rebalance 的滑点成本

资源需求:
  计算资源: 中等（单机可完成）
  预期耗时: 2-3 周
  人力: 1 名研究员

Gate Decision: [待 Data Ready 完成后评审]
```

#### Gate 评审记录

```yaml
评审人: 团队 Lead
评审时间: 2025-01-20
评审结论: CONDITIONAL PASS

通过项:
  ✅ FG-1: 研究主问题清晰可测
  ✅ FG-2: 时间窗完整冻结
  ✅ FG-3: Universe 明确定义
  ✅ FG-4: 信号机制完整描述
  ✅ FG-5: 输出 Artifact 完整

条件:
  ⚠️ 需要补充:
     1. 明确"风险调整收益"的具体计算方法
     2. 补充执行成本的初步估计
     3. 明确 Sharpe 计算的频率（日/周/月）

下次评审: Data Ready Gate
```

### 案例 2：失败的 Mandate（反面教材）

#### 问题分析

```yaml
原 Mandate:
  研究问题: "研究加密市场的各种技术指标"

问题:
  ❌ 太宽泛，无法判断假设是否成立
  ❌ "各种指标"没有具体化
  ❌ 缺少时间窗定义
  ❌ Universe 模糊（"加密市场"）

Gate 结果: NO-GO

原因:
  - 无法从研究结果判断假设是否成立
  - 范围太大，实际无法执行
  - 容易在研究中途改变目标

建议:
  缩小研究范围，选择一个具体信号
  重新撰写 Mandate
```

---

## 输出 Artifact 规范

### 必需的输出文件

#### 1. mandate.yml（机器可读配置）

```yaml
# Mandate 配置文件
mandate:
  version: "1.0"
  created_at: "2025-01-20"
  lineage_id: "momentum_top50_v1"

research_question:
  text: "在 CoinMarketCap Top 50 加密货币中，基于 20 日动量信号..."
  hypothesis_direction: "positive"
  primary_metric: "sharpe_ratio"

time_window:
  study_start: "2020-01-01"
  study_end: "2024-12-31"
  splits:
    train:
      start: "2020-01-01"
      end: "2022-12-31"
    test:
      start: "2023-01-01"
      end: "2023-12-31"
    backtest:
      start: "2024-01-01"
      end: "2024-06-30"
    holdout:
      start: "2024-07-01"
      end: "2024-12-31"

universe:
  eligibility_criteria:
    - "CMC Top 50 by 24h volume"
    - "Exclude stablecoins"
    - "Exclude leveraged tokens"
    - "Min 1 year history"
  baseline_count: 42
  manifest_file: "universe_manifest.csv"

signal:
  name: "momentum_20d_decile"
  type: "momentum"
  description: "20-day return cross-sectional ranking"
  calculation:
    lookback_window: 20
    quantiles: 5
    rebalance_frequency: "weekly"

parameters:
  exploratory:
    lookback_window: [10, 20, 40, 60]
    n_quantiles: [3, 5, 10]
    rebalance_frequency: ["daily", "weekly", "monthly"]

  frozen:
    signal_type: "momentum"
    price_type: "close"
    weighting: "equal"

data_source:
  price_data: "binance_spot_ohlcv"
  reference_data: "coinmarketcap"
  version: "v1.0"
  export_date: "2025-01-15"
```

#### 2. mandate_summary.md（人类可读总结）

```markdown
# Mandate: 加密货币动量策略研究

## 研究问题
在 Top 50 加密货币中，20 日动量信号在 2020-2024 年是否产生正风险调整收益？

## 核心假设
动量溢价在加密货币市场存在，即过去表现好的资产会继续表现好。

## 研究范围
- **Universe**: CMC Top 50（排除稳定币和杠杆代币）
- **时间**: 2020-2024，按 Train/Test/Backtest/Holdout 切分
- **信号**: 20 日动量，5 分位数，等权重组合

## 预期贡献
- 学术：验证加密市场动量溢价
- 实践：提供基础多空信号

## 风险评估
- 动量崩溃风险
- 流动性风险
- 执行成本风险

## Gate 决策
待 Data Ready 完成后评审
```

#### 3. universe_manifest.csv（标的清单）

```csv
symbol,included,reason,listing_date
BTC,TRUE,Baseline,2010-01-01
ETH,TRUE,Baseline,2015-08-07
BNB,TRUE,Baseline,2017-09-01
...
USDT,FALSE,Stablecoin,2014-10-06
BTCUP,FALSE,Leveraged token,2020-05-19
```

#### 4. time_window_config.json（时间窗配置）

```json
{
  "study_window": {
    "start": "2020-01-01",
    "end": "2024-12-31"
  },
  "splits": {
    "train": {
      "start": "2020-01-01",
      "end": "2022-12-31",
      "duration_days": 1096
    },
    "test": {
      "start": "2023-01-01",
      "end": "2023-12-31",
      "duration_days": 365
    },
    "backtest": {
      "start": "2024-01-01",
      "end": "2024-06-30",
      "duration_days": 182
    },
    "holdout": {
      "start": "2024-07-01",
      "end": "2024-12-31",
      "duration_days": 184
    }
  },
  "split_principle": "chronological_no_overlap",
  "holdout_purpose": "final_validation_no_design_leakage"
}
```

### Artifact 目录结构

```
project_root/
├── mandates/
│   └── momentum_top50_v1/
│       ├── mandate.yml              # 机器可读配置
│       ├── mandate_summary.md       # 人类可读总结
│       ├── universe_manifest.csv    # 标的清单
│       ├── time_window_config.json  # 时间窗配置
│       ├── field_dictionary.md      # 字段说明
│       └── artifact_catalog.md      # 产物目录
```

---

## 与下一阶段的交接标准

### Data Ready 阶段的输入要求

Mandate 完成后，Data Ready 阶段需要明确的输入：

#### 交接清单

| 项目 | 交付物 | 格式 | 用途 |
|------|--------|------|------|
| 研究范围 | mandate.yml | YAML | 指导数据需求 |
| 标的列表 | universe_manifest.csv | CSV | 数据提取范围 |
| 时间范围 | time_window_config.json | JSON | 数据提取时间 |
| 数据源 | mandate.yml 中定义 | 文本 | 数据源选择 |
| 字段需求 | 信号描述 | 文本 | 确定需要的字段 |

#### 交接验收标准

```yaml
Data Ready 可以开始的条件:
  ✅ mandate.yml 通过格式验证
  ✅ universe_manifest.csv 包含所有必需字段
  ✅ time_window_config.json 时间范围合理
  ✅ 数据源明确且可访问
  ✅ 字段需求清晰可理解

验收方式:
  - 自动化脚本验证配置文件格式
  - 人工审查研究范围合理性
  - 数据源可用性检查

Gate 决策:
  Mandate PASS → 可以开始 Data Ready
  Mandate CONDITIONAL PASS → 满足条件后开始
  Mandate RETRY → 修改后重新提交
  Mandate NO-GO → 终止当前研究方向
```

### Frozen Spec 的传递

```yaml
Mandate Frozen Spec:
  包含内容:
    - 研究主问题（不可修改）
    - Universe 定义（不可扩展）
    - 时间切分（不可调整）
    - 信号基本逻辑（不可改变）

  传递方式:
    - mandate.yml 作为 Data Ready 的输入
    - Data Ready 只能验证，不能修改

  违反后果:
    - 如果 Data Ready 修改了冻结内容
    - 整条研究线失效
    - 需要从 Mandate 重新开始
```

### 版本控制

```yaml
Mandate 版本:
  命名规范: {研究类型}_{主要特征}_v{版本号}
  示例: momentum_top50_v1, reversal_stablecoins_v2

版本锁定:
  - 一旦通过 Gate，版本锁定
  - 任何修改需要新版本号
  - 旧版本归档保留

Lineage 追踪:
  - 每个 Mandate 分配唯一 lineage_id
  - 后续所有阶段标记相同 lineage_id
  - Child Lineage 使用新的 lineage_id
```

---

## 总结：Mandate 的核心价值

### 防止什么

1. **目标漂移**：研究中途悄无声息地改变目标
2. **范围膨胀**：不断添加新的标的、信号、参数
3. **假设匹配**：根据结果修改假设
4. **样本污染**：未来数据泄漏到过去
5. **过拟合**：在历史数据上找规律

### 确保什么

1. **清晰的研究问题**：知道要回答什么
2. **固定的研究边界**：知道什么能做、什么不能做
3. **可验证的假设**：能够判断假设是否成立
4. **可复现的过程**：不同人能重现研究结果
5. **诚实的验证**：样本外验证真正有效

### 成功的 Mandate 特征

```yaml
一个成功的 Mandate 应该:
  清晰:
    - 5 分钟能向新人解释清楚
    - 研究问题可以用一句话说明

  可测:
    - 能够明确判断假设是否成立
    - 评估指标客观可计算

  冻结:
    - 核心要素不预留模糊空间
    - 后续阶段"照单执行"即可

  诚实:
    - 不预设结论
    - 允许假设被证伪

  可行:
    - 数据可获得
    - 计算可实现
    - 资源可满足
```

### 最后的提醒

```
Mandate 花费的时间：1-2 天
节省的时间：后续阶段避免走弯路

偷懒 Mandate 的代价：
  - 中途发现方向不对
  - 结果无法解释
  - 整条研究线推倒重来

好的 Mandate 是成功研究的一半
```

---

## 附录：Mandate 模板

### 快速开始模板

```yaml
# Mandate 模板
# 复制此文件并填写具体内容

mandate:
  version: "1.0"
  created_at: "YYYY-MM-DD"
  lineage_id: "{研究类型}_{主要特征}_v1"

research_question:
  text: "在 [Universe] 中，使用 [信号机制]，在 [时间窗] 上，是否能够获得 [预期方向] 的收益？"
  hypothesis_direction: "positive/negative"
  primary_metric: "sharpe_ratio/returns/drawdown/..."

time_window:
  study_start: "YYYY-MM-DD"
  study_end: "YYYY-MM-DD"
  splits:
    train: {start: "...", end: "..."}
    test: {start: "...", end: "..."}
    backtest: {start: "...", end: "..."}
    holdout: {start: "...", end: "..."}

universe:
  eligibility_criteria:
    - "准入口径 1"
    - "准入口径 2"
  baseline_count: N
  manifest_file: "universe_manifest.csv"
  explicit_exclusions:
    - "排除项目 1"
    - "排除项目 2"

signal:
  name: "{信号名称}"
  type: "{信号类型}"
  description: "{信号描述}"
  calculation:
    parameter_1: value_1
    parameter_2: value_2

parameters:
  exploratory:
    parameter_1: [value1, value2, ...]

  frozen:
    parameter_2: "固定值"

data_source:
  primary_data: "{数据源}"
  version: "{版本}"
```

---

**文档版本**: v1.0
**最后更新**: 2025-01-20
**维护者**: 量化研究团队
**反馈**: 请在团队会议中提出改进建议
