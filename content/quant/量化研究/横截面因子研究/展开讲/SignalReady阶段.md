# SignalReady 阶段 - 详细展开

> 对应摘要版：`../简介/SignalReady阶段.md`
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
不能在 SignalReady 阶段引入“训练后学出来的权重”，因为那是 TrainFreeze 之后的事情。

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

## 3.1 `factor_identity`

这是因子的“身份信息”。

至少要包括：

- `factor_id`
- `factor_name`
- `raw_factor_fields`
- `derived_factor_fields`
- `signal_timestamp`
- `available_after`

例如：

```yaml
factor_id: MOM_20D
raw_factor_fields:
  - close
derived_factor_fields:
  - ret_20d
time_semantics:
  signal_timestamp: close_time
  available_after: next_bar_open
```

SignalReady 最危险的错误之一，就是在这一层写得太松，导致后面人各自脑补。

## 3.2 `factor_role_contract`

这组的核心是：

> 后续应该用什么证据逻辑去检这个因子？

比如：

- `standalone_alpha`：要看排序能力
- `regime_filter`：要看 gated vs ungated 是否改善
- `combo_filter`：要看组合后分布和风险是否改善

角色一旦冻结，下游不能再因为结果不好就改成另一个角色。

## 3.3 `factor_structure_contract`

这里决定这个因子的内部结构。

如果是 `single_factor`，应明确：

- 单一核心公式
- 不允许后续偷偷拼接多个条件轴

如果是 `multi_factor_score`，应明确：

- 每个组成字段
- 组合方式
- 是否先标准化再线性组合

但是不能写成：

- “后面训练再学权重”

那已经不是合同层，而是训练层。

## 3.4 `neutralization_policy`

这一层不只是“要不要中性化”，更是在定义：

> 这个因子的正式输出语义，到底是不是已经剔除了某些暴露。

例如：

- `none`
- `market_beta_neutral`
- `group_neutral`

如果写成 `group_neutral`，必须同时绑定：

- group taxonomy 从哪里来
- taxonomy 版本是什么

## 3.5 `delivery_contract`

这个阶段不应该只输出文字说明，还要输出可被机器消费的产物：

- `factor_panel.parquet`
- `factor_manifest.yaml`
- `factor_coverage.parquet`

这样后面进入 TrainFreeze 时，使用的是同一个信号实体，而不是“描述上的同一个信号”。

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
score = 0.5 * zscore(momentum) + 0.5 * zscore(turnover)
```

但不能写：

```python
weights = fit_model(X_train, y_train)
```

这一步必须留到 TrainFreeze 之后。

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

必须真实产出：

- `factor_panel.parquet`
- `factor_manifest.yaml`
- `factor_coverage.parquet`

原因很直接：

- TrainFreeze 需要直接消费这些 artifact
- 后面所有阶段都需要知道“到底评的是哪个版本的信号”

如果只有 Markdown 描述，没有真实 panel，那么下游每个人都会自己实现一版“我理解中的这个因子”。

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

如果这些轴必须改，说明问题不在 TrainFreeze，而在 SignalReady 本身，应该回到这里修正或开新 lineage。

---

## 9. 最后给一个判断标准

合格的 SignalReady，应该满足这样一个条件：

> 一个不参与当前讨论的人，只看 `factor_manifest`、`factor_contract` 和 `factor_panel`，就能知道这个因子是什么、扮演什么角色、后面应怎样被评估。

如果做不到这一点，说明信号合同还没有真正冻结。
