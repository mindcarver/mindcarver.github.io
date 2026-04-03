# Holdout Validation 阶段 - 最终未见窗口验证

## 1. 阶段定位

HoldoutValidation 是横截面因子研究链里的最终验证层。

它回答的问题是：

> 在完全未参与设计、调参、组合选择的窗口上，已冻结方案是否仍然成立？如果不成立，应该如何解释和处置？

Holdout 的核心纪律只有一句：

> 只能解释结果，不能回写规则。

---

## 2. 本阶段只冻结什么

HoldoutValidation 只冻结五组内容：

- `window_contract`
- `reuse_contract`
- `drift_audit`
- `failure_governance`
- `delivery_contract`

必须同时保留：

- 单窗口结果
- 合并窗口结果
- 与 Backtest 的对比结果

---

## 3. 本阶段不该做什么

禁止事项：

- 改参数
- 改 factor selection
- 改组合规则
- 改执行合同
- 把 Holdout 当额外优化窗口
- 多次运行后只挑最好的一次
- 用“延期观察”伪装成不是新的 Holdout

如果 Holdout 不理想，合法动作是：

- `CONDITIONAL PASS`
- `NO_GO`
- `CHILD LINEAGE`

而不是“再给它三个月看看能不能好起来”。

---

## 4. 必需输入

建议至少有以下输入：

- `frozen_portfolio_spec.json`
- `strategy_combo_ledger.csv`
- `stage_completion_certificate.yaml`
- `time_split.json`

---

## 5. 必需输出

建议至少产出以下 artifact：

- `holdout_run_manifest.json`
- `holdout_backtest_compare.csv`
- `window_results/`
- `holdout_gate_decision.md`
- `artifact_catalog.md`
- `field_dictionary.md`

其中至少要明确：

```yaml
window_contract:
  holdout_start: "2024-10-01"
  holdout_end: "2024-12-31"
  single_run_only: true

reuse_contract:
  modify_factor_selection: false
  modify_portfolio_rules: false
  modify_cost_model: false

failure_governance:
  direction_flip_threshold: explicit
  child_lineage_trigger: explicit
```

---

## 6. Formal Gate

### FG-1: 复用关系明确

- 明确复用的是哪一版 frozen portfolio spec
- 明确没有修改任何上游冻结对象

### FG-2: 单窗口与合并窗口结果齐全

- 单窗口结果存在
- 合并窗口结果存在
- 二者与 Backtest 可对比

### FG-3: Drift Audit 明确

- rolling OOS 一致性低时必须给出显式解释
- 解释要对应具体时间段与原因
- 不允许“结果不好可能只是运气”这种空话

### FG-4: 方向翻转定义明确

- 何为方向翻转必须预先定义
- 翻转后是 `CONDITIONAL PASS`、`NO_GO` 还是 `CHILD LINEAGE` 必须预先写清
- 不允许根据翻转是否有利来临时改判定口径

### FG-5: Holdout 与 Backtest 结果不混合

- 只能比较，不能拼接成一条统一绩效曲线

---

## 7. Audit Gate

HoldoutValidation 可以记录：

- 相对 Backtest 的退化幅度
- 市场环境差异
- 方向一致性变化
- 极端事件影响

这些记录用于解释结果，不是给上游规则开后门。

---

## 8. 常见反模式

### 反模式 1：根据 Holdout 结果回头调参

这会直接使 Holdout 失效。  
一旦发生，只能把该窗口视为已消费，后续若继续研究，应开新 lineage。

### 反模式 2：把延期观察当作“不是新的 Holdout”

错误写法：

- “先跑三个月 holdout，不好再延三个月”
- “延期观察不算新的 holdout”

这本质上仍然是在继续消费未见窗口。

### 反模式 3：只保留合并结果，不保留单窗口结果

这样会掩盖具体失败时段，也无法做像样的 drift audit。

### 反模式 4：把 Holdout 与 Backtest 拼成单曲线

这会让最终验证窗口失去独立性，属于典型治理错误。

---

## 9. 失败处理

Holdout 失败时，先区分三类问题：

- 实现错误
- 市场环境漂移
- 策略本身失效

只有实现错误允许按既定治理做修复后重跑。  
如果是环境漂移或策略失效，合法路径应是：

- 保留失败证据
- 明确 drift audit
- 触发 `CONDITIONAL PASS`、`NO_GO` 或 `CHILD LINEAGE`

而不是继续消费同一 holdout 窗口找更好看的答案。

---

## 10. 与 Shadow / 实盘的交接

只有在 HoldoutValidation 完成后，才有资格进入 Shadow 或实盘准备。

需要交接的不是“最好看的一段结果”，而是：

- 已冻结的完整方案
- Holdout 对比结论
- drift audit
- failure governance
- 明确的风险说明

如果结论是 `CONDITIONAL PASS`，必须额外附带：

- 初始资金限制
- 监控指标
- 降级或退出条件

---

## 11. 一句话标准

HoldoutValidation 完成后，如果团队还能随意改方案再试一次，那就说明这个阶段实际上还没成立。
