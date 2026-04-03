# Train Freeze 阶段 - 训练窗尺子冻结

> 详细版：`../展开讲/Train阶段.md`

## 1. 阶段定位

虽然文件名仍叫 `Train阶段`，但这里更准确的职责是 `TrainFreeze`。

这个阶段不是宣布哪个因子赢了，也不是正式做交易层回测；它的职责是把后续必须复用的训练窗尺子固定下来。

它回答的问题是：

> 在不偷看 Test / Backtest / Holdout 的前提下，我们训练窗里学到了哪些预处理、中性化、分桶和调仓尺子？

---

## 2. 本阶段只冻结什么

TrainFreeze 只冻结五组内容：

- `preprocess_contract`
- `neutralization_contract`
- `ranking_bucket_contract`
- `rebalance_contract`
- `delivery_contract`

这里的“尺子”指的是后续阶段必须复用的规则，而不是最终交易胜负。

典型冻结对象包括：

- winsorize / clipping 规则
- 标准化规则
- 中性化回归设定
- bucket 切点或排序规则
- rebalance 频率与触发语义

---

## 3. 本阶段不该做什么

禁止事项：

- 用 Test 或 Backtest 结果反推 train 尺子
- 在本阶段重新发明 SignalReady 已冻结的因子定义
- 在本阶段宣布“最终胜出因子”或“最佳组合”
- 在本阶段写交易层 `Top N 选币 / 组合持仓 / 执行方案`
- 在本阶段输出 `best_h`、单资产命中率等时序语义

特别强调：

如果为了让下游表现更好，回头改 `ranking_bucket` 或 `rebalance`，那不是正常迭代，而是污染研究线。

---

## 4. 必需输入

建议至少有以下输入：

- `factor_manifest.yaml`
- `factor_panel.parquet`
- `factor_coverage.parquet`
- `factor_contract.md`
- `stage_completion_certificate.yaml`

---

## 5. 必需输出

建议至少产出以下 artifact：

- `csf_train_freeze.yaml`
- `train_quality.parquet`
- `train_variant_ledger.csv`
- `train_rejects.csv`
- `train_gate_decision.md`
- `artifact_catalog.md`
- `field_dictionary.md`

`csf_train_freeze.yaml` 至少应显式区分：

```yaml
frozen_signal_contract_reference: MOM_20D_v1

preprocess_rules:
  winsorize: mad_3
  standardize: cross_sectional_zscore

neutralization_rules:
  policy: market_beta_neutral
  regressors:
    - beta_30d
    - log_market_cap

ranking_bucket_rules:
  bucket_mode: quintile
  min_cross_section_size: 20

rebalance_rules:
  frequency: 1d
  trigger: scheduled_only

auxiliary_conditions:
  listing_days_min: 30
```

---

## 6. Formal Gate

### FG-1: 预处理尺子明确

- 去极值规则明确
- 标准化规则明确
- 缺失处理规则明确

### FG-2: 中性化尺子明确

- 中性化是否启用明确
- 风险暴露来源明确
- 不允许事后因为下游结果去改回归条件

### FG-3: 分桶尺子明确

- bucket 规则明确
- 最小截面样本量明确
- ties 和缺失处理明确

### FG-4: 调仓尺子明确

- 频率明确
- 触发语义明确
- 不允许留“回测后再优化”这种口子

### FG-5: 参数台账完整

- `train_variant_ledger.csv` 可追溯
- reject 项有原因
- freeze 后不可治理轴明确

---

## 7. Audit Gate

可以记录但不应越界的审计项：

- 不同训练窗下尺子的稳定性
- coverage 与 breadth
- 尺子对小样本时段的脆弱性

这些项可以帮助判断风险，但不应把 Test 或 Backtest 的绩效结果回写进来。

---

## 8. 常见反模式

### 反模式 1：拿下游结果反推训练规则

错误流程：

- 先跑 Test
- 看到单调性不好
- 再回来改 bucket 切点

这会直接破坏独立样本验证。

### 反模式 2：把 TrainFreeze 写成投资组合设计

错误写法：

- Top N 选币
- Long / Short 持仓数
- 仓位分配方式
- 执行窗口

这些属于 BacktestReady 的组合与执行合同，不属于训练窗尺子。

### 反模式 3：重开因子定义轴

如果在本阶段改：

- raw factor fields
- derived factor fields
- score formula
- factor role

那说明问题应回到 SignalReady，而不是继续伪装成 Train 调参。

### 反模式 4：只留最终版本，不留 reject ledger

如果看不到候选尺子为何被拒绝，后续团队会重复探索同一死路。

---

## 9. 与下一阶段的交接

TestEvidence 只能消费 TrainFreeze 已冻结的尺子，不得：

- 在 test 窗重估 preprocess
- 在 test 窗重估 neutralization
- 在 test 窗重估 bucket 规则
- 在 test 窗回写 rebalance 规则

下一阶段主要依赖：

- `csf_train_freeze.yaml`
- `train_quality.parquet`
- `train_variant_ledger.csv`
- `train_rejects.csv`

---

## 10. 一句话标准

TrainFreeze 完成后，下游团队应当拿到一把固定尺子去量独立样本，而不是拿到一个“看到结果再调一调”的半成品。
