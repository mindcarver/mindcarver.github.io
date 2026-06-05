# TestEvidence 阶段 - 详细展开

> 第一次阅读建议先看：[`./英文术语表.md`](./英文术语表.md)

## 1. 先说人话：TestEvidence 为什么不是“回测前的小回测”

很多团队写 Test 文档时最容易犯的错误，就是把它写成一个“简化版 Backtest”：

- 看 Sharpe
- 看回撤
- 看净收益
- 顺手开始谈容量

这会直接把证据层和交易层混在一起。

TestEvidence 这个阶段真正要回答的是：

> 这个信号在独立样本上，是否真的具有横截面证据意义？

也就是说，这一步主要看的是：

- 排序能力
- 证据稳定性
- admissibility

而不是“交易后赚了多少钱”。

---

## 2. TestEvidence 回答的核心问题

## 2.1 `standalone_alpha` 有没有横截面排序能力

这类因子要回答的是：

- 因子值高的资产，后续在截面上是不是更容易表现更好
- 这种排序关系是否稳定
- 分组收益是否有单调性

这类问题天然对应：

- `Rank IC`
- `ICIR`
- bucket returns
- monotonicity
- breadth / coverage

## 2.2 `regime_filter` / `combo_filter` 有没有改善作用

这类因子不一定单独追求排序能力，而是要回答：

- 加了这个条件后，组合分布是否更好
- drawdown 是否改善
- tail risk 是否改善
- 覆盖率下降是否可接受

所以它和 standalone alpha 的 gate 逻辑不能混在一起。

## 2.3 进入 BacktestReady 的候选是谁

TestEvidence 不是在交易层宣布大胜，但它确实要冻结：

- 哪些 factor variant 具备继续进入 backtest 的资格
- 冻结后的 `csf_selected_variants_test` 是谁
- 哪些对象被拒绝
- 为什么被拒绝

在 CSF current skill 里，这一步不再只是泛泛写一份“挑中了哪个因子”的说明，
而是要把后续 BacktestReady 能直接消费的冻结对象落进：

- `csf_selected_variants_test.csv`
- `csf_test_gate_table.csv`
- `csf_test_contract.md`

也就是说，
TestEvidence 结束时不仅要回答“谁通过了证据门禁”，
还要回答“Backtest 到底允许消费哪一版横截面候选 variant”。

## 2.4 先把这些英文字段说清楚

TestEvidence 的英文字段都在区分“统计证据”和“交易胜利”。它们用来证明候选值得进入 BacktestReady，但不宣布策略已经能交易赚钱。

| 字段 | 人话含义 | 为什么要用它 |
| --- | --- | --- |
| `window_contract` | Test 独立样本窗口合同。 | 明确这段数据没有参与训练尺子估计，避免测试窗口被事后调整。 |
| `formal_gate_contract` | 正式证据门禁合同。 | 规定通过 TestEvidence 要看哪些证据，且必须与 factor role 匹配。 |
| `Rank IC` | 因子排名与后续收益排名的相关性。 | 衡量 standalone alpha 是否真的有横截面排序能力。 |
| `ICIR` | IC 的稳定性指标。 | 只看平均 IC 可能被少数窗口撑起来，ICIR 看稳定程度。 |
| `bucket returns` | 分桶后各组的后续收益。 | 检查高分组和低分组是否出现有序分层。 |
| `monotonicity` | 分桶收益是否大体单调。 | 如果高分不比低分好，排序信号就不可信。 |
| `breadth` | 证据覆盖的广度。 | 证明结果不是只靠少数资产或少数时点。 |
| `admissibility_contract` | 准入合同。 | 即使有证据，也要判断覆盖、稳定性和样本宽度是否足够进入回测。 |
| `audit_contract` | 审计合同。 | 记录 regime、负结果、多重比较和 crowding，不只留下好看的结果。 |
| `crowding_review.md` | 拥挤度审计。 | 它是风险披露，不应偷偷变成 formal gate 的阻断条件。 |
| `csf_selected_variants_test.csv` | 被准入进入 BacktestReady 的 CSF 候选清单。 | BacktestReady 只能消费这份冻结清单，不能重新挑候选。 |
| `csf_test_gate_table.csv` | TestEvidence 结构化门禁表。 | 让通过/拒绝的证据和理由可审计、可机器读取。 |
| `csf_test_contract.md` | TestEvidence 合同说明。 | 解释证据口径、冻结候选和下游消费边界。 |

---

## 3. 本阶段冻结的五组内容怎么理解

## 3.1 `window_contract`

这组解决的是：

> Test 的独立样本窗口到底是什么。

必须明确：

- 时间边界
- 该窗口是否完全独立于训练尺子估计
- 这一窗口内允许做什么，不允许做什么

如果 Test 窗口边界模糊，下游很容易把“测试结果不好”解释成“窗口取错了”。

## 3.2 `formal_gate_contract`

这组解决的是：

> 对不同角色的因子，正式证据门禁到底看哪些项。

关键点不是阈值写得多漂亮，而是：

- gate 是否与因子角色一致
- gate 是否属于证据层

例如：

- standalone alpha：看 `Rank IC / ICIR / bucket returns / monotonicity`
- regime filter：看 gated vs ungated 的改善

## 3.3 `admissibility_contract`

这组回答的是：

> 即便有一定证据，它是否仍然值得进入交易层回测？

例如可能考虑：

- 覆盖率是否太差
- breadth 是否太窄
- regime 依赖是否过强
- 结果是否仅在极少数窗口成立

它不是交易层胜利，但它决定“有没有资格继续往下走”。

## 3.4 `audit_contract`

这组负责记录：

- regime 分层表现
- 多重比较修正
- 负结果归档
- 边界情况说明
- crowding / distinctiveness 审计

其重点在于留下完整证据，而不是只留下好看的结果。

特别要补一句 current skill 里的硬边界：

> `crowding_review.md` 属于 `audit_contract`，
> 不是 `formal_gate_contract` 的直接阻断条件。

也就是说：

- 可以把 crowding 发现写进审计层
- 可以提示后续 Backtest / Holdout 注意 distinctiveness 风险
- 但不能把“拥挤度高”直接偷偷写进 formal gate，当成阻断证据门禁的理由

## 3.5 `delivery_contract`

这里解决的是：

- TestEvidence 结束后，哪些机器可读产物必须存在
- 哪些冻结对象要供 BacktestReady 直接消费

按照 current skill，至少应显式落下：

- `rank_ic_timeseries.parquet`
- `rank_ic_summary.json`
- `bucket_returns.parquet`
- `monotonicity_report.json`
- `breadth_coverage_report.parquet`
- `subperiod_stability_report.json`
- `filter_condition_panel.parquet`
- `target_strategy_condition_compare.parquet`
- `gated_vs_ungated_summary.json`
- `csf_test_gate_table.csv`
- `crowding_review.md`
- `csf_selected_variants_test.csv`
- `csf_test_contract.md`
- `csf_test_gate_decision.md`
- `run_manifest.json`
- `artifact_catalog.md`
- `field_dictionary.md`

---

## 4. 关于 OOS 分桶，最容易误解的地方

这是横截面研究里非常容易写错、也非常容易讲错的一点。

### 误区：OOS 里一律不能 `qcut`

这种说法不够精确。

如果 TrainFreeze 冻结的 bucket 规则本来就是：

> 每个时点按当期横截面排序分成五组

那么在 OOS 每个时点按同样的规则分桶，是**应用既定规则**，不是泄漏。

真正不允许的是：

- 原来冻结五组，后来在 OOS 改成三组
- 原来冻结 quintile，后来改成 decile
- 原来冻结 rank bucket，后来改成按经验阈值切
- 原来冻结缺失直接剔除，后来在 OOS 为了结果好看换处理方式

也就是说：

> 不允许的是改变规则，不是机械地禁止“在 OOS 做分桶运算”。

这点在详细文档里必须讲清，不然团队容易把“应用冻结规则”和“重估规则”混为一谈。

---

## 5. 为什么 TestEvidence 不能提前谈交易层胜利

### 5.1 因为 Test 解决的是证据问题，不是执行问题

你在 Test 里看到：

- Rank IC 好
- bucket 单调

这说明信号在统计上有意义。  
但还没回答：

- 扣费后是否为正
- 容量是否够
- 执行是否可实现
- 风险覆盖是否足够

这些属于 BacktestReady。

### 5.2 因为 Test 的主要职责是“筛资格”

更准确地说，TestEvidence 像是：

- 证明它值得进入交易层
- 不是证明它已经是交易层赢家

所以把：

- `Sharpe >= 0.8`
- `回撤 < 25%`
- `容量 > X`

写成 Test 主 gate，是错位。

---

## 6. 这一阶段应如何落地

## 6.1 先按角色分流

第一步不要急着算指标，而是先问：

> 当前这个对象是 standalone alpha，还是 filter / combo？

因为如果角色错了，后面整个证据逻辑都会错。

## 6.2 对 standalone alpha 看排序证据

常见核心对象：

- `rank_ic_timeseries.parquet`
- `bucket_returns.parquet`
- monotonicity 结果
- breadth / coverage 结果

你最终要回答的是：

- 是否存在稳定的排序能力
- 是否不是只靠少数窗口撑起来

## 6.3 对 filter / combo 看改善证据

这类对象应更关注：

- gated vs ungated 对比
- tail risk 是否下降
- drawdown 是否改善
- 覆盖率损失是否值得

如果它只是减少了样本，结果看起来“更干净”，不一定代表真的有改善价值。

## 6.4 冻结 `csf_selected_variants_test` 与 test contract

TestEvidence 结束时，应该明确：

- 哪些对象进入 backtest
- 哪些不进入
- 原因是什么
- `csf_selected_variants_test.csv` 是否已经冻结
- 这些冻结对象是否已经写入 `csf_test_contract.md` 和 `csf_test_gate_table.csv`

这里最忌讳的不是错选，而是“只留下当前一个版本，看不见背后的筛选过程”。

---

## 7. 输出物应该长什么样

典型输出物包括：

- `rank_ic_timeseries.parquet`
- `rank_ic_summary.json`
- `bucket_returns.parquet`
- `monotonicity_report.json`
- `breadth_coverage_report.parquet`
- `subperiod_stability_report.json`
- `csf_test_gate_table.csv`
- `crowding_review.md`
- `csf_selected_variants_test.csv`
- `csf_test_contract.md`
- `csf_test_gate_decision.md`
- `run_manifest.json`

其中最关键的通常是：

### `csf_selected_variants_test.csv`

它要保证 BacktestReady 消费的是一套**被正式冻结的横截面候选 variant**，而不是一段口头描述。

它要让团队看得到：

- 被选中的 variant 有哪些
- 没被选中的 variant 有哪些
- 为什么

### `csf_test_gate_table.csv`

它负责承接 formal gate 的结构化结论，包括排序证据、稳定性、覆盖率和准入判断。

### `csf_test_contract.md`

它负责说明 BacktestReady 只能消费哪一批冻结 variant，以及这些 variant 是按什么证据合同被选出的。

---

## 8. 最常见的错误

## 8.1 把 Test 写成回测绩效报告

错误表现：

- 主标题还是 Test，正文已经在谈净收益、成本后 Sharpe、容量

这会让下游 BacktestReady 失去清晰职责。

## 8.2 根据 OOS 结果改训练尺子

这是最严重的错误之一。

例如：

- OOS 单调性不好，就回去改 bucket
- OOS 覆盖率低，就回去改 eligibility
- OOS 结果差，就回去改 neutralization

这会直接破坏独立样本验证。

## 8.3 只报好的 regime

如果 regime 分层结果里：

- 牛市差
- 熊市强

却只报告熊市强，那团队会系统性高估信号鲁棒性。

## 8.4 输出 `best_horizon`

在 CSF 横截面因子路线里，这通常就是错误信号。

`qros-csf-test-evidence` 明确不应产出：

- `best_h`
- 预测 horizon 口径
- 单资产命中率语义

如果文档里出现这些词，通常说明研究线正在从“同一时点横向排序”滑向“单资产时序择时”。

CSF TestEvidence 应冻结的是通过证据门禁的 factor variants，而不是冻结“哪个持有期最好”。

---

## 9. 与下一阶段的边界

BacktestReady 只能消费：

- 已冻结的 `csf_selected_variants_test.csv`
- 已冻结的 `csf_test_contract.md`
- admissibility 结论

BacktestReady 不应再做：

- variant 重选
- horizon 重估
- 训练尺子重估
- 信号身份重解释

如果 BacktestReady 连“评的是谁”都不清楚，问题不在 backtest，而在 TestEvidence 没冻结干净。

---

## 10. 最后给一个判断标准

合格的 TestEvidence，应该满足：

> 即便先不跑任何交易层仿真，团队也能清楚知道：
> 这个对象是否拥有独立样本上的正式证据，
> 哪些 `csf_selected_variants_test` 已经冻结，
> 以及 BacktestReady 到底应该消费哪一版 `csf_test_contract.md`。

如果还在把这一步写成“半个回测”，说明阶段边界还没收紧。
