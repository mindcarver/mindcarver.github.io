# Mandate 阶段 - 横截面因子研究授权与边界冻结

> 详细版：`../展开讲/Mandate阶段.md`

## 1. 阶段定位

Mandate 是横截面因子研究的正式起点。它的职责不是证明因子有效，而是冻结研究合同，避免后续把研究边界、样本切分和执行假设写乱。

推荐主链如下：

```text
Mandate
  -> DataReady
  -> SignalReady
  -> TrainFreeze
  -> TestEvidence
  -> BacktestReady
  -> HoldoutValidation
```

Mandate 结束时，团队应当只回答一个问题：

> 我们到底在研究什么，允许怎么研究，不允许怎么改？

---

## 2. 本阶段只冻结什么

Mandate 只冻结上游研究合同，不冻结下游统计胜负。

必须冻结的内容：

- `research_intent`
- `research_route`
- `scope_contract`
- `target_market / universe 口径`
- `time_split`
- `data_contract`
- `execution_contract`
- `kill_criteria`

必须明确写清：

- 研究路线是否为 `cross_sectional_factor`
- 研究对象是“同一时点资产间比较”，不是单资产时序择时
- Universe 如何入选、如何排除、冻结后是否允许变化
- Train / Test / Backtest / Holdout 的时间边界
- 数据来源、bar size、时区、更新时间语义
- 允许的交易表达是 `long_short_market_neutral` 还是 `long_only_rank`

---

## 3. 本阶段不该做什么

Mandate 不允许提前宣布任何统计或交易层结论。

禁止事项：

- 用 `IC / Rank IC / Sharpe / 回撤 / 容量` 作为 Mandate formal gate
- 在 Mandate 阶段写“因子已经有效”或“预计通过 Test”
- 根据历史结果倒推最优 horizon
- 先试多个 horizon 再把最好那个写回合同
- 用“后面再看”替代明确的 time split 或 universe 口径

原因很简单：这些都属于下游证据层或执行层，不属于研究授权层。

---

## 4. 必需输入

进入 Mandate 前，至少需要有以下原始信息：

- 原始研究想法
- 目标市场与资产类别
- 预期数据来源
- 预期交易表达
- 研究路线判断

如果这些信息不全，Mandate 可以记录不确定项，但不能把不确定项伪装成已冻结结论。

---

## 5. 必需输出

建议至少产出以下 artifact：

- `mandate.md`
- `research_scope.md`
- `research_route.yaml`
- `time_split.json`
- `stage_completion_certificate.yaml`

其中必须机器可读的最少字段包括：

```yaml
research_route: cross_sectional_factor

target_market:
  market: crypto
  venue_scope: [binance_spot]
  universe_rule: "Top N by liquidity with explicit exclusions"
  exclusions:
    - stablecoins
    - leveraged_tokens

time_split:
  train:
    start: "2021-01-01"
    end: "2023-12-31"
  test:
    start: "2024-01-01"
    end: "2024-06-30"
  backtest:
    start: "2024-07-01"
    end: "2024-09-30"
  holdout:
    start: "2024-10-01"
    end: "2024-12-31"

execution_contract:
  portfolio_expression: long_short_market_neutral
  bar_size: 1h
  timezone: UTC
```

---

## 6. Formal Gate

Mandate 的 formal gate 只检查“合同是否清楚、可执行、可追溯”。

### FG-1: 研究路线明确

- 是否明确为 `cross_sectional_factor`
- 是否明确排除时间序列主线措辞

### FG-2: Universe 口径冻结

- 入选规则明确
- 排除规则明确
- 冻结后是否允许新增或剔除有明确治理

### FG-3: 时间切分冻结

- Train / Test / Backtest / Holdout 边界明确
- 没有“待定”“后续再调”这类留口子写法

### FG-4: 数据与执行合同明确

- 数据来源、bar size、时区、发布时间语义明确
- 交易表达明确

### FG-5: Kill Criteria 明确

- 明确什么情况下停止本条研究线
- 明确什么情况下必须重开 child lineage

---

## 7. Audit Gate

Mandate 的 audit gate 可以记录判断依据，但不阻断正式推进：

- 因子经济学逻辑
- 数据可得性与重建成本
- 可能的执行难点
- 已知文献或市场先例

这些内容可以写，但不能冒充正式验证结论。

---

## 8. 常见反模式

### 反模式 1：把下游门禁写进 Mandate

错误写法：

- `IC 均值 > 0.03`
- `Sharpe > 1.0`
- `容量 > 500 万美元`

这些不是授权条件，而是后续阶段才有资格判断的证据。

### 反模式 2：Horizon 事后优化

错误流程：

```python
best_h = None
best_ic = -999
for h in [1, 4, 24, 72]:
    ic = run_research(h)
    if ic > best_ic:
        best_ic = ic
        best_h = h
```

然后再把 `best_h` 回写到 Mandate。  
这会直接污染研究合同。

### 反模式 3：Universe 留口子

错误写法：

- “先用 Top 50，不行再看 Top 100”
- “必要时可按效果调整样本池”

这种写法会把选择偏差合法化。

### 反模式 4：把横截面研究写成单资产预测

错误写法：

- “预测 BTC 明天会不会涨”
- “提高单币命中率”

横截面因子研究关心的是同一时点的相对排序，不是单资产方向命中率。

---

## 9. 与下一阶段的交接

DataReady 只能消费 Mandate 已冻结的内容，不能静默修改：

- universe 口径
- 时间边界
- research route
- 执行表达

下一阶段需要接收的核心对象：

- `research_route.yaml`
- `time_split.json`
- `mandate.md`
- `research_scope.md`

---

## 10. 一句话标准

Mandate 写完后，如果团队还不能明确回答“研究的边界是什么、后面哪些改动必须开新 lineage”，那这个阶段就还没完成。
