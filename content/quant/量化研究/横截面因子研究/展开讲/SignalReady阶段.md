# SignalReady 阶段 - 详细展开

> 第一次阅读建议先看：[`./英文术语表.md`](./英文术语表.md)

## 1. 先说人话：SignalReady 锁的不是胜负，而是“信号身份”

很多研究文档一到信号阶段就忍不住开始写：

- 这个因子 IC 如何
- 哪个因子更强
- 哪些因子该删

这其实已经混到 Test 或 Train 之后去了。

SignalReady 真正要锁的是：

> 这个因子到底是什么，它扮演什么角色，它是怎么被组合表达消费的，它是否带中性化语义。

你可以把它理解成**因子的正式身份证和合同说明书**。

没有这层，后面每个人都可能在研究同一个名字、不同内容的信号。

---

## 2. SignalReady 回答的核心问题

## 2.1 这个因子是什么

即：

- 因子 ID 是什么
- 输入字段是什么
- 变换链是什么
- 时间语义是什么

如果这一层没写清，“动量因子”这个名字其实没有任何约束力。  
它可能是 20 日收益，也可能是 20 日收益再叠加流动性权重，也可能是先去极值再排名后的某种分数。

## 2.2 这个因子扮演什么角色

QROS 里至少要把它分成：

- `standalone_alpha`
- `regime_filter`
- `combo_filter`

这一步非常关键，因为后续证据逻辑完全不同。

例如：

- `standalone_alpha` 要证明自己有横截面排序能力
- `regime_filter` 要证明自己能改善已存在组合
- `combo_filter` 要证明自己能作为条件层改进分布

如果角色不明确，后面很容易拿错尺子去量它。

## 2.3 这个因子是单因子还是确定性多因子分数

必须明确：

- `single_factor`
- `multi_factor_score`

注意：

这里允许的多因子，只能是**确定性公式**。  
不能在 SignalReady 阶段引入“训练后学出来的权重”，因为那属于 TrainFreeze 阶段。

## 2.4 这个因子将被什么组合表达消费

常见表达有：

- `long_short_market_neutral`
- `long_only_rank`

这一步必须写明，因为同一个信号放在不同组合表达里，含义是不同的。

## 2.5 这个因子是否自带中性化语义

必须明确：

- `none`
- `market_beta_neutral`
- `group_neutral`

尤其是 `group_neutral`，如果不明确 taxonomy 版本，后面任何“中性化结果”都不可追溯。

---

## 3. 本阶段真正要冻结的五组内容

在 CSF 路线里，SignalReady 冻结的是横截面因子的正式合同和真实 factor panel。
本阶段的 freeze groups 是：

- `factor_identity`
- `panel_contract`
- `factor_expression`
- `context_contract`
- `delivery_contract`

也就是说，不能只在 Markdown 里写“这是某某因子”，而要真实物化：

> `factor identity -> factor expression -> factor_panel.parquet -> downstream CSF artifacts`

## 3.1 `factor_identity`

这组解决的是：

> 这个横截面因子到底是谁。

至少要写清：

- `factor_id`
- `factor_role`
- `factor_structure`
- 是否 `single_factor` 还是 `multi_factor_score`
- `portfolio_expression`
- `neutralization_policy`

这些语义必须从 mandate / route identity 继承，不能在 SignalReady 静默改成另一条路线。

## 3.2 `panel_contract`

这组解决的是：

> 因子值落在哪一张已冻结的 `date x asset` 面板上。

至少要写清：

- 继承哪个 DataReady panel
- 主键和时间键是什么
- 是否沿用 DataReady 的 eligibility 语义
- factor panel 的时间覆盖和资产覆盖

如果这里没写清，下游 Train/Test 就可能在另一张面板上消费“同名因子”。

## 3.3 `factor_expression`

这组解决的是：

> factor panel 到底按什么公式、什么字段、什么确定性变换被算出来。

至少要写清：

- 原始输入字段
- 派生字段
- 公式或变换链
- 如果是 `multi_factor_score`，组合方式是否 deterministic

例如：

```yaml
factor_expression:
  factor_id: MOM_20D
  structure: single_factor
  raw_fields:
    - close
  derived_fields:
    - ret_20d
  formula: ret_20d
```

这里最危险的错误，是只留下“20 日动量”这种自然语言标签，
却没把真正的计算表达式冻结下来。

## 3.4 `context_contract`

这组解决的是：

> 因子在横截面上下文里如何被解释。

至少包括：

- group context
- taxonomy 版本
- coverage 语义
- input field source
- route inheritance

尤其是 `group_neutral`，如果 taxonomy 版本不可追溯，后面任何分组中性化结果都不可复现。

## 3.5 `delivery_contract`

这组解决的是：

> 本阶段真实交付哪些 CSF formal artifacts。

按照 CSF current skill，本阶段至少应真实交付：

- `factor_panel.parquet`
- `factor_manifest.yaml`
- `component_factor_manifest.yaml`
- `factor_coverage_report.parquet`
- `factor_group_context.parquet`
- `route_inheritance_contract.yaml`
- `factor_contract.md`
- `factor_field_dictionary.md`
- `csf_signal_ready_gate_decision.md`
- `run_manifest.json`
- `artifact_catalog.md`
- `field_dictionary.md`

只有这样，TrainFreeze 才是在消费**同一个 factor 实体**，
而不是在消费“描述上差不多”的几个实现。

---

## 4. 为什么 SignalReady 不能提前做“因子好坏判断”

因为这会把合同层和证据层混在一起。

### 例子 1：在 SignalReady 写“至少 50% 因子 |IC| > 0.02”

这听起来很合理，但其实有两个问题：

1. 它依赖独立样本统计，不属于合同层
2. 它会刺激人去反向修改 signal 定义，让它更容易过这个门槛

### 例子 2：在 SignalReady 用全样本相关性做去重

这更危险。

因为“全样本”通常已经包含 Test / Backtest / Holdout 的信息。  
你一旦用它来决定保留哪些因子，本质上就是把下游信息倒灌回上游定义。

### 例子 3：在 SignalReady 学权重

这会直接把本阶段变成“半个训练阶段”。

SignalReady 可以定义确定性多因子公式，比如：

```yaml
formula: 0.5 * zscore(momentum) + 0.5 * zscore(turnover)
```

但不能写：

```python
weights = fit_model(X_train, y_train)
```

这一步必须留到 TrainFreeze 阶段，而不能发生在 SignalReady。

---

## 5. 实战里应怎么写一份合格的 Signal 合同

## 5.1 先固定 identity

别急着谈表现，先把下面这些写死：

- 输入字段
- 派生字段
- 时间语义
- 缺失处理语义

## 5.2 再固定 role

问自己一个问题：

> 后面我是打算把它当作单独 alpha 评估，还是当过滤器评估？

如果答案含糊，那说明合同还没写清。

## 5.3 再固定 structure

如果是多因子分数，必须写出：

- 子项有哪些
- 组合是线性、排序加和还是布尔 gating
- 是否 deterministic

## 5.4 再固定 portfolio_expression 与 neutralization_policy

因为这两者决定了下游如何正确消费信号。

特别是在加密横截面场景里，很多看似“信号本身”的表现，其实来自：

- beta 暴露
- 某类资产分组暴露
- 大小盘偏好

所以中性化政策必须明确，不然下游评估会混语义。

## 5.5 CSF current skill 里的 baseline factor 纪律

这是这套文档最值得补充的一条。

在 current `qros-csf-signal-ready-author/review` 口径里，
SignalReady 第一版应先冻结清楚 baseline factor：

- 可以是 1 个 baseline factor
- 也可以是极少数明确声明的 component factors
- 但不应把搜索网格、训练后权重或下游筛选结果提前塞进本阶段

换句话说：

> 先把“这个 baseline factor 是谁”冻结干净，
> 再让 TrainFreeze 去消费它，
> 而不是一上来就把整片参数搜索空间包装成 signal ready。

这条纪律的价值在于：

- 限制上游定义漂移
- 限制训练搜索空间提前膨胀
- 防止 Train 阶段第一次发现 signal schema 其实没冻结干净

## 5.6 未物化的 factor / component 不能静默消失

如果某些本次已经声明要物化的 baseline factor 或 component factor：

- 生成失败
- coverage 太差，导致无法作为正式交付物被下游消费
- 因 schema 或输入问题未能物化

就必须在 `factor_coverage_report.parquet`、`factor_contract.md` 或 gate decision 中单独列出来，并写清原因。

这里的 coverage 是交付完整性和可消费性检查，不是 alpha performance 证据。

它不能被写成“这个因子表现好 / 不好”的判断。

这些失败对象不能被静默省略，然后假装整批 factor 都已冻结完成。

这也不是允许把 full search batch 提前带入 SignalReady 的例外。

如果一批搜索参数还没有进入 TrainFreeze 的治理范围，就不应该靠“未物化清单”包装进本阶段。

---

## 6. 这一阶段常见的错误

## 6.1 名字很清楚，内容很模糊

比如写：

- 因子名：20 日动量

但不写：

- 用哪个价格字段
- 用 close 还是 VWAP
- 信号时点是什么
- 可用时点是什么

这会让同名因子在不同实现里变成不同东西。

## 6.2 把角色留空

一旦不写 `factor_role`，后面就会出现：

- A 把它当 standalone alpha
- B 把它当 combo filter
- C 又把它当 regime filter

最后大家拿同一个因子名在讨论三件不同的事情。

## 6.3 用全样本去重

这是文档中非常需要避免的一类写法。

正确的 SignalReady 里，可以写：

- 哪些因子在定义上高度相近
- 哪些因子家族可能存在重叠

但不要在这里做最终“保留 / 淘汰”的基于全样本表现的决定。

## 6.4 把时间序列术语带进来

比如：

- best horizon
- 单资产 hit rate
- 触发买卖点

这类词一进来，整条横截面研究线就会开始往时序策略滑。

---

## 7. 输出物为什么必须真实存在

SignalReady 不能只停留在文档语义层。

必须真实产出完整的 delivery contract：

- `factor_panel.parquet`
- `factor_manifest.yaml`
- `component_factor_manifest.yaml`
- `factor_coverage_report.parquet`
- `factor_group_context.parquet`
- `route_inheritance_contract.yaml`
- `factor_contract.md`
- `factor_field_dictionary.md`
- `csf_signal_ready_gate_decision.md`
- `run_manifest.json`
- `artifact_catalog.md`
- `field_dictionary.md`

原因很直接：

- TrainFreeze 需要直接消费这些 artifact
- 后面所有阶段都需要知道“到底评的是哪个 factor、哪套 schema、哪份 coverage”

如果只有 Markdown 描述，没有真实 factor artifacts，
那么下游每个人都会自己实现一版“我理解中的这个因子”。

---

## 8. 与下一阶段的边界

TrainFreeze 只能在已冻结的 signal 合同上学习训练窗尺子。

也就是说，进入下一阶段后不能再改：

- factor expression
- raw / derived fields
- factor role
- factor structure
- portfolio expression
- neutralization policy

如果这些轴在 SignalReady 尚未关闭前发现问题，可以回到本阶段修正后重新冻结。

如果 SignalReady 已经关闭，或者下游已经消费了这些 artifact，就不应原地改写旧合同，而应开新版本或新 lineage。

---

## 9. 最后给一个判断标准

合格的 SignalReady，应该满足这样一个条件：

> 一个不参与当前讨论的人，只看 `factor_manifest.yaml`、`factor_contract.md`、`factor_field_dictionary.md` 和 `factor_panel.parquet`，
> 就能知道这个 baseline factor 是什么、对应哪套 `date x asset` 面板、后面应怎样被评估。

如果做不到这一点，说明信号合同还没有真正冻结。
