# BacktestReady 阶段 - 详细展开

> 第一次阅读建议先看：[`./英文术语表.md`](./英文术语表.md)

## 1. 先说人话：BacktestReady 锁的是“组合如何被交易”，不是“信号是什么”

到了 BacktestReady，很多团队最容易做的一件错事是：

- 发现交易后结果不好
- 然后回头改信号、改分桶、改训练尺子

这说明他们其实还没分清：

- 上游在锁信号和证据
- BacktestReady 在锁组合和执行

BacktestReady 真正要回答的是：

> 在不改变信号定义和训练尺子的前提下，这个候选进入交易世界后，组合表达、成本、风险和容量是否站得住？

所以它是**交易层合同冻结**。

---

## 2. BacktestReady 回答的核心问题

## 2.1 这个候选将被怎样组成组合

这里回答的是：

- long short 还是 long only
- 权重如何形成
- 持仓数量怎么定

这不是 SignalReady 的角色问题，而是交易层组合问题。

## 2.2 信号怎么映射成交易

例如：

- T 时发出信号，什么时候交易
- maker 优先还是 taker
- 再平衡窗口多大

这些属于执行政策，而不是信号本身。

## 2.3 风险覆盖是否足够

这里要看：

- 单标的权重上限
- beta 暴露控制
- 某些 group / sector 暴露控制

这一步的目标不是追求最好看收益，而是让结果具有交易可解释性。

## 2.4 成本和容量是否可追溯

BacktestReady 不是为了给出一个漂亮净值曲线，而是为了明确：

- 成本模型来自哪里
- 流动性代理是什么
- 容量上限是怎么估出来的

---

## 3. 本阶段冻结的五组内容怎么理解

## 3.1 `portfolio_contract`

这组解决的是“怎么把通过 TestEvidence 的 CSF variant 变成组合”。

典型内容包括：

- portfolio expression
- 权重规则
- long / short 数量
- 是否 market neutral

这一层必须显式使用上游已冻结的 `factor_role`、`factor_structure`、`portfolio_expression` 和 `neutralization_policy`。

## 3.2 `execution_contract`

这组解决的是“怎么下单”。

典型内容包括：

- signal to trade lag
- maker/taker 选择
- 再平衡窗口
- 是否分批执行

这里最容易犯的错是把实现习惯当成合同。  
例如“先按现在代码里默认配置跑一下”，这不叫冻结合同。

## 3.3 `risk_contract`

这组解决的是“组合被允许暴露成什么样”。

例如：

- 单标的上限
- beta neutral
- 某类资产暴露上限
- 参与率上限

如果没有 risk contract，回测结果往往只是“理论最好看”，不是可接受的交易方案。

## 3.4 `diagnostic_contract`

这组解决的是“结果是用什么回测引擎和什么口径跑出来的”。

除了引擎差异，还要保留：

- return accounting provenance
- 路径级风险诊断
- 成本与容量诊断
- signal/factor proxy return 与 formal PnL 的边界

原因不是为了炫技，而是为了避免把某个引擎特性、proxy return 或成本假设误当成策略收益。

## 3.5 `delivery_contract`

这一组定义：

- 哪些产物必须落盘
- 什么文件是后续 Holdout 要直接复用的冻结方案
- 哪些 ledger 必须保留

---

## 4. 为什么 BacktestReady 不能回写上游

## 4.1 因为它解决的是交易约束，不是信号定义

如果你在 BacktestReady 看见：

- 净收益差
- 滑点大
- 容量低

正确的问题应该是：

- 执行合同是不是太激进
- 组合规则是不是不现实
- 风险覆盖是不是不够

而不是：

- 那我把 bucket 改一下
- 那我把 signal 再中性化一下
- 那我把因子重新选一下

后者已经是上游阶段的内容。

## 4.2 因为一旦回写，就没法知道问题到底出在哪

如果回测不好，你一边改信号、一边改组合、一边改执行，最后即便结果变好了，也没人知道：

- 是信号变好了
- 是执行变宽松了
- 是成本假设变轻了

阶段治理的核心价值，就是把这些变化分层隔离。

---

## 5. 这一阶段应怎么落地

## 5.1 先固定已通过 TestEvidence 的 variant

BacktestReady 开始时，首先要明确：

- 当前消费的是谁
- 角色是什么
- 组合表达是什么
- 来自哪一版 `csf_selected_variants_test.csv`

如果这一层还含糊，后面所有 backtest 结果都不可靠。

## 5.2 再写 `portfolio_contract`

这里应该明确：

- 持仓数量
- 权重方式
- long / short 是否对称
- 是否 market neutral

如果团队在这里还说“先跑一下再决定”，说明组合合同还没冻结。

## 5.3 再写 `execution_contract`

尤其要明确：

- T 日信号，T 还是 T+1 交易
- maker 优先还是 taker
- 是否假设部分未成交
- 再平衡窗口有多宽

很多“高收益策略”，只是因为在这里默认了过于乐观的成交语义。

## 5.4 再写 `risk_contract`

例如：

- 单标的权重不能超过 10%
- 组合 beta 要控制在某个范围
- 某类 group 暴露不能失控

风险覆盖不是为了压低收益，而是为了让回测结果具有实盘可解释性。

## 5.5 再做成本和容量追溯

这一层不能只写数字，要写来源。

例如：

- 手续费假设来源
- 滑点代理来源
- 借贷 / 资金费率来源
- 容量参与率边界来源

如果这些来源不可追溯，净收益就没有多少解释价值。

---

## 6. 输出物为什么必须丰富

BacktestReady 结束后，通常至少应有：

- `portfolio_contract.yaml`
- `portfolio_weight_panel.parquet`
- `rebalance_ledger.csv`
- `turnover_capacity_report.parquet`
- `cost_assumption_report.md`
- `portfolio_summary.parquet`
- `portfolio_return_series.parquet`
- `equity_curve.parquet`
- `portfolio_pnl_ledger.parquet`
- `asset_pnl_ledger.parquet`
- `risk_adjusted_metrics.parquet`
- `name_level_metrics.parquet`
- `drawdown_report.json`
- `target_strategy_compare.parquet`
- `csf_backtest_gate_table.csv`
- `return_accounting_provenance.yaml`
- `csf_backtest_contract.md`
- `csf_backtest_gate_decision.md`
- `run_manifest.json`
- `artifact_catalog.md`
- `field_dictionary.md`

这几类文件各自解决不同问题：

### `portfolio_contract.yaml`

明确“冻结方案是什么”。

### `portfolio_weight_panel.parquet`

明确“每个时点究竟持了什么、权重是多少”。

### `return_accounting_provenance.yaml`

明确 formal PnL 和 return series 来自哪里，避免把 factor score 或 proxy return 当成正式回测收益。

### `csf_backtest_gate_table.csv`

明确本阶段的交易层 gate 结论。

### `turnover_capacity_report.parquet` / `cost_assumption_report.md`

明确“这个组合在多大规模、什么成本假设下仍成立”。

---

## 7. 本阶段最常见的错误

## 7.1 为了净值更好看就改上游规则

这是最常见也最危险的错误。

例如：

- 看见扣费后太差，就把 rebalance 频率改了
- 看见 drawdown 太大，就回去改 factor selection

如果这些规则本应属于上游冻结对象，那你其实已经在消费下游结果回写上游。

## 7.2 成本模型只有数字，没有来源

比如：

- “滑点 5 bps”
- “借贷成本 3%”

如果没有来源，这些数字只是装饰。

## 7.3 容量只写一句“问题不大”

容量不是态度问题，而是可追溯的边界问题。

至少应该明确：

- 用了什么流动性代理
- 参与率上限是多少
- 哪些资产是瓶颈

## 7.4 只保留最好看的组合，不保留筛选过程

这样后面团队只会不停重复同样搜索。

---

## 8. 与下一阶段怎么衔接

HoldoutValidation 的职责是：

- 复用这套 `portfolio_contract.yaml`
- 在未见窗口验证
- 做 drift audit

所以 Holdout 阶段不应再改：

- factor selection
- portfolio contract
- execution contract
- risk contract

如果 Holdout 结果不好，正确动作是解释或失败治理，而不是回头把 BacktestReady 重新雕一遍。

---

## 9. 最后给一个判断标准

合格的 BacktestReady，应该满足：

> 现在把这套冻结方案原样扔进未见窗口，团队也能清楚知道它交易的是什么、怎么交易、成本与容量边界在哪里。

如果还需要边跑边定，那它就还不是 Ready。
