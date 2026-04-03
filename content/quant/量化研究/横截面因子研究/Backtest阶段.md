# Backtest Ready 阶段 - 组合与执行合同冻结

## 1. 阶段定位

BacktestReady 的职责是把已经通过 TestEvidence 的候选，落成真正可复用的组合、成本、风险和执行合同。

它回答的问题是：

> 在不改信号定义、不改训练尺子、不重选因子的前提下，这个组合表达在交易约束下是否成立？

BacktestReady 是交易层合同冻结，不是重新研究信号。

---

## 2. 本阶段只冻结什么

BacktestReady 只冻结五组内容：

- `execution_policy`
- `portfolio_policy`
- `risk_overlay`
- `engine_contract`
- `delivery_contract`

这意味着本阶段新增冻结对象应主要包括：

- 组合权重面板
- 回测引擎口径
- 成本模型
- 风险约束
- 容量证据

---

## 3. 本阶段不该做什么

禁止事项：

- 重新选择 factor
- 重估 TrainFreeze 的尺子
- 回写 SignalReady 的因子定义
- 为了让 backtest 更好看，再改 bucket、rebalance 或 neutralization 规则
- 把 holdout 结果预先合并进 backtest 解释

如果 backtest 看完后才决定“把周频改成双周频更好”，那要先判断这是否其实是上游 `rebalance_contract` 的变更，而不是假装属于正常 backtest 优化。

---

## 4. 必需输入

建议至少有以下输入：

- `selected_factor_spec.json`
- `factor_selection.csv`
- `test_gate_decision.md`
- `stage_completion_certificate.yaml`

---

## 5. 必需输出

建议至少产出以下 artifact：

- `frozen_portfolio_spec.json`
- `portfolio_weight_panel.parquet`
- `portfolio_curve.parquet`
- `engine_compare.csv`
- `vectorbt/`
- `backtrader/`
- `strategy_combo_ledger.csv`
- `capacity_review.md`
- `backtest_gate_decision.md`
- `artifact_catalog.md`
- `field_dictionary.md`

其中至少应明确：

```yaml
execution_policy:
  signal_to_trade_lag: 1_bar
  order_type: maker_first
  rebalance_window: 4h

portfolio_policy:
  portfolio_expression: long_short_market_neutral
  weighting: equal_weight
  holding_count:
    long: 10
    short: 10

risk_overlay:
  max_single_weight: 0.1
  beta_neutral: true
  capacity_participation_limit: 0.01
```

---

## 6. Formal Gate

### FG-1: 组合规则明确

- `portfolio_expression` 与上游一致
- 组合权重生成规则明确
- 组合台账有选择理由

### FG-2: 执行合同明确

- 信号到交易的时滞明确
- 下单方式明确
- 再平衡窗口明确
- 不允许“实盘时再看”这种留口子写法

### FG-3: 风险覆盖明确

- 仓位上限明确
- 是否市场中性明确
- 暴露检查可追溯

### FG-4: 双引擎或多引擎结果可比

- 引擎口径一致性可检查
- 差异有解释

### FG-5: 成本与容量可追溯

- 成本假设来源明确
- 流动性代理明确
- 参与率边界明确

---

## 7. Audit Gate

BacktestReady 可以记录但不应越权的审计项：

- 扣费前后收益差
- 成本分解
- 容量瓶颈资产
- 引擎差异来源

这些项帮助判断交易可实现性，但不能成为改写上游研究合同的理由。

---

## 8. 常见反模式

### 反模式 1：回测结果不好就改上游尺子

错误流程：

- 看见扣费后收益下降
- 回去改 bucket 或 rebalance
- 然后把修改后的结果继续当同一条研究线

这会直接破坏阶段治理。

### 反模式 2：忽略成本与容量的可追溯性

错误写法：

- “假设滑点 5 bps”
- “容量大概没问题”

如果来源不可追溯，这些数字只是装饰。

### 反模式 3：只保留一条最好看的组合

如果 `strategy_combo_ledger.csv` 没有记录为什么选择该组合、为什么拒绝其他组合，团队会反复重跑同样的搜索。

### 反模式 4：把 Backtest 写成 Holdout 预演

Backtest 可以多次修正实现错误，但不能把未见窗口结果混进来，更不能提前消费 Holdout。

---

## 9. 与下一阶段的交接

HoldoutValidation 只能复用 BacktestReady 已冻结的方案，不得：

- 改参数
- 改 factor selection
- 改组合规则
- 改执行合同

下一阶段主要依赖：

- `frozen_portfolio_spec.json`
- `strategy_combo_ledger.csv`
- `portfolio_weight_panel.parquet`
- `portfolio_curve.parquet`
- `capacity_review.md`

---

## 10. 一句话标准

BacktestReady 完成后，团队应当拿到一套“现在就能原样带去未见窗口验证”的组合合同，而不是一份仍在偷偷研究信号本体的回测笔记。
