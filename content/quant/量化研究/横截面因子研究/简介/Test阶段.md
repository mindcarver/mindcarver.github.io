# Test Evidence 阶段 - 独立样本证据冻结

> 详细版：`../展开讲/Test阶段.md`

## 1. 阶段定位

虽然文件名仍叫 `Test阶段`，但这里更准确的职责是 `TestEvidence`。

这个阶段的职责不是宣布交易层胜利，而是冻结独立样本上的正式证据。

它回答的问题是：

> 在完全独立于训练尺子的样本上，这个横截面因子是否真的有排序能力，或者这个 filter / combo 是否真的改善了结果？

---

## 2. 本阶段只冻结什么

TestEvidence 只冻结五组内容：

- `window_contract`
- `formal_gate_contract`
- `admissibility_contract`
- `audit_contract`
- `delivery_contract`

对于不同因子角色，证据语义必须分流：

- `standalone_alpha`: 关注 `Rank IC`、`ICIR`、bucket returns、monotonicity、breadth / coverage
- `regime_filter` / `combo_filter`: 关注 gated vs ungated 的改善、尾部风险改善、覆盖率代价

---

## 3. 本阶段不该做什么

禁止事项：

- 把 Test 当 Backtest
- 在 Test 阶段宣布最终交易策略胜利
- 在 Test 窗重估 TrainFreeze 的 preprocess / neutralization / bucket / rebalance 尺子
- 输出 `best_horizon`
- 用 OOS 结果回写上游 factor 定义或训练尺子

特别强调：

如果文中写“不能在 OOS 重新估计分位切点”，代码也必须真正复用训练窗已冻结规则，不能一边声明一边重新 `qcut`。

---

## 4. 必需输入

建议至少有以下输入：

- `csf_train_freeze.yaml`
- `train_variant_ledger.csv`
- `factor_manifest.yaml`
- `asset_universe_membership.parquet`
- `stage_completion_certificate.yaml`

---

## 5. 必需输出

建议至少产出以下 artifact：

- `rank_ic_timeseries.parquet`
- `bucket_returns.parquet`
- `admissibility_report.parquet`
- `factor_selection.csv`
- `factor_selection.parquet`
- `selected_factor_spec.json`
- `test_gate_table.csv`
- `test_gate_decision.md`
- `artifact_catalog.md`
- `field_dictionary.md`

`selected_factor_spec.json` 至少应写清：

```yaml
selected_factor_id: MOM_20D
factor_role: standalone_alpha
factor_structure: single_factor
portfolio_expression: long_short_market_neutral
neutralization_policy: market_beta_neutral
```

---

## 6. Formal Gate

### 6.1 `standalone_alpha` 的正式 gate

建议核心关注：

- `Rank IC`
- `ICIR`
- bucket returns
- monotonicity
- breadth / coverage

这些指标回答的是“横截面排序能力是否存在”，而不是“净收益是否已经可交易”。

### 6.2 `regime_filter` / `combo_filter` 的正式 gate

建议核心关注：

- gated vs ungated 的对比
- 分布改善是否稳定
- 回撤改善是否明显
- 尾部风险改善是否明显
- 覆盖率损失是否仍可接受

### 6.3 不属于本阶段的 gate

以下项目可以记录，但不应作为本阶段主 gate：

- 净 Sharpe
- 成本后收益
- 执行质量
- 容量上限

这些属于 BacktestReady。

---

## 7. Audit Gate

TestEvidence 可以记录：

- OOS 覆盖率
- regime 分层结果
- 多重比较修正
- 负结果归档

但记录 regime 风险，不等于允许在本阶段直接改组合规则。

---

## 8. 常见反模式

### 反模式 1：OOS 重新估计训练尺子

错误写法：

```python
labels = pd.qcut(factor_in_oos, q=5)
```

如果 bucket 切点本该来自训练窗冻结规则，这种写法就是泄漏。

### 反模式 2：把 Test 写成回测绩效报告

错误写法：

- `Sharpe >= 0.8`
- `最大回撤 >= -25%`
- `容量预估`

这些都属于交易层，不是证据层。

### 反模式 3：输出 `best_horizon`

横截面路线里，这会把 Test 重新变成参数搜索器。  
如果需要重新定义 horizon，应回到上游合同，而不是在 Test 里偷偷选一个最好看的。

### 反模式 4：只保留通过结果

如果失败的 regime、不达标窗口、被拒绝因子没有归档，后续会系统性高估研究质量。

---

## 9. 与下一阶段的交接

BacktestReady 只允许消费：

- 已冻结的 `selected_factor_spec`
- 已冻结的 factor selection
- 已记录的 admissibility 结论

BacktestReady 不得：

- 重新选择因子
- 重新学习训练尺子
- 在交易层再解释上游信号身份

---

## 10. 一句话标准

TestEvidence 完成后，团队应当拿到“这个因子是否具备独立样本排序能力”的正式证据，而不是一份已经偷偷混入交易层优化的报告。
