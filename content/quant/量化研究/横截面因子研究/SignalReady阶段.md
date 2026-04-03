# Signal Ready 阶段 - 因子合同冻结

## 1. 阶段定位

SignalReady 的职责是冻结横截面因子的正式定义，而不是提前宣布它是赢家。

它回答的问题是：

> 这个因子到底是什么，以什么角色存在，用什么组合表达，采用什么中性化语义？

SignalReady 是“因子合同层”，不是“统计胜利层”。

推荐主链：

```text
Mandate
  -> DataReady
  -> SignalReady
  -> TrainFreeze
  -> TestEvidence
  -> BacktestReady
  -> HoldoutValidation
```

---

## 2. 本阶段只冻结什么

SignalReady 只冻结五组内容：

- `factor_identity`
- `factor_role_contract`
- `factor_structure_contract`
- `neutralization_policy`
- `delivery_contract`

其中必须显式给出：

- `factor_role`: `standalone_alpha | regime_filter | combo_filter`
- `factor_structure`: `single_factor | multi_factor_score`
- `portfolio_expression`: `long_short_market_neutral | long_only_rank`
- `neutralization_policy`: `none | market_beta_neutral | group_neutral`

如果是 `multi_factor_score`，第一版只能是确定性组合公式，不能在这里引入训练后学权重。

---

## 3. 本阶段不该做什么

禁止事项：

- 用 `IC / IR / Sharpe / 回撤` 作为 SignalReady formal gate
- 在本阶段用全样本结果做因子去重
- 在本阶段根据 Test 或 Holdout 表现回写因子定义
- 在本阶段偷偷重写 DataReady 的 panel key、eligibility 或时间边界
- 用单资产命中率、best horizon 之类时序主线措辞描述横截面因子

SignalReady 可以写“预期方向”和“交付字段”，但不能写“因子已经通过统计检验”。

---

## 4. 必需输入

建议至少有以下输入：

- `panel_manifest.json`
- `asset_universe_membership.parquet`
- `eligibility_base_mask.parquet`
- `csf_data_contract.md`
- `stage_completion_certificate.yaml`

---

## 5. 必需输出

建议至少产出以下 artifact：

- `factor_panel.parquet`
- `factor_manifest.yaml`
- `factor_coverage.parquet`
- `factor_contract.md`
- `factor_field_dictionary.md`
- `signal_gate_decision.md`
- `artifact_catalog.md`
- `field_dictionary.md`

`factor_manifest.yaml` 至少应包含：

```yaml
factor_id: MOM_20D
factor_role: standalone_alpha
factor_structure: single_factor
portfolio_expression: long_short_market_neutral
neutralization_policy: market_beta_neutral

inputs:
  raw_factor_fields:
    - close
  derived_factor_fields:
    - ret_20d

time_semantics:
  signal_timestamp: close_time
  available_after: next_bar_open
```

---

## 6. Formal Gate

### FG-1: 因子身份明确

- 因子 ID 唯一
- 输入字段明确
- 变换链明确
- 时间语义明确

### FG-2: 因子角色明确

- 明确是 `standalone_alpha`、`regime_filter` 还是 `combo_filter`
- 不允许“后面再决定角色”

### FG-3: 因子结构明确

- 明确是 `single_factor` 还是 `multi_factor_score`
- 若为多因子，组合公式必须确定性、可复现

### FG-4: 投组合表达与中性化策略明确

- `portfolio_expression` 明确
- `neutralization_policy` 明确
- 若为 `group_neutral`，所用 taxonomy 版本必须可追溯

### FG-5: 交付物真实存在

- `factor_panel.parquet` 真实生成
- `factor_manifest.yaml` 与 `factor_contract.md` 一致
- `factor_coverage.parquet` 能解释缺失与覆盖

---

## 7. Audit Gate

SignalReady 可以记录但不应越界的审计项：

- 因子覆盖度
- 因子缺失模式
- 输入字段稳定性
- 新币与稳定币处理语义

这些是合同层审计。  
它们不能替代独立样本上的统计证据。

---

## 8. 常见反模式

### 反模式 1：把 SignalReady 写成单因子绩效报告

错误写法：

- “至少 50% 因子 |IC| > 0.02”
- “多空收益年化 > 5% 才能进入下一阶段”

这些都属于 TestEvidence，不属于 SignalReady。

### 反模式 2：全样本去重

错误写法：

- 用全样本相关性矩阵决定保留哪些因子
- 用未来表现倒推“哪个因子家族该删”

这会把下游信息倒灌回因子定义。

### 反模式 3：在本阶段学权重

错误写法：

```python
weights = fit_model(X_train, y_train)
```

这已经是 TrainFreeze 之后的事情，不应在 SignalReady 里发生。

### 反模式 4：角色和表达不冻结

如果没有明确 `factor_role` 和 `portfolio_expression`，下游每个人都会按自己的理解消费同一个因子。

---

## 9. 与下一阶段的交接

TrainFreeze 只能在已冻结的 Signal 合同上学习训练窗尺子，不能重开以下轴：

- factor expression
- raw / derived fields
- factor role
- factor structure
- portfolio expression
- neutralization policy

下一阶段主要依赖：

- `factor_panel.parquet`
- `factor_manifest.yaml`
- `factor_coverage.parquet`
- `factor_contract.md`

---

## 10. 一句话标准

SignalReady 完成后，任何人看到 artifact 都应该能复现同一个因子，而不是各自脑补一版“他真正想表达的信号”。
