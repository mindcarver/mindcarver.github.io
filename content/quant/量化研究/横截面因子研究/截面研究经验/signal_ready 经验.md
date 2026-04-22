这组 `csf_signal_ready` 的 author / review，比 `csf_data_ready` 更接近你整套研究 OS 的核心。  
如果说 `data_ready` 冻结的是**研究母体**，那 `signal_ready` 冻结的就是：

> **你到底在研究什么因子，它在策略里扮演什么角色，它以什么确定性规则被表达成正式可比较的横截面分数。**

我重新想了一遍后，觉得这阶段最值得沉淀的，不是“怎么把 factor 算出来”，而是下面这些方法论和高频坑。

---

# 一、先给这个阶段一个准确定位

`csf_signal_ready` 不是普通的“因子开发”，也不是“先写个公式试试看”。

它真正冻结的是 5 件事：

### 1. 因子身份

- `factor_id`
    
- `factor_version`
    
- `factor_direction`
    

也就是：**这到底是不是同一个因子**。

---

### 2. 因子角色

- `standalone_alpha`
    
- `regime_filter`
    
- `combo_filter`
    

也就是：**这个因子在策略里到底干什么**。

---

### 3. 因子结构

- `single_factor`
    
- `multi_factor_score`
    

也就是：**它是单一因子，还是一个确定性组合分数**。

---

### 4. 组合表达与中性化

- `portfolio_expression`
    
- `neutralization_policy`
    

也就是：**这个因子准备被怎么消费，而不是留到 train/test 再解释。**

---

### 5. 正式机器产物

- `factor_panel.parquet`
    
- `factor_manifest.yaml`
    
- coverage / dictionary / contract / catalog / run_manifest
    

也就是：**它不是口头概念，而是一个真实可消费、可重建、可审查的 factor contract。**

---

所以一句话说：

> `csf_signal_ready` 冻结的不是“研究想法”，而是“正式因子合同”。

---

# 二、它和 `csf_data_ready` 的本质区别

这个特别值得沉淀，因为两阶段很容易混。

---

## `csf_data_ready` 冻结的是

- 研究母体
    
- 面板主键
    
- universe membership
    
- eligibility 底座
    
- taxonomy/shared feature base
    
- coverage 证据
    

也就是：

> **研究是在什么世界里进行。**

---

## `csf_signal_ready` 冻结的是

- 因子身份
    
- 因子角色
    
- 因子结构
    
- score 公式
    
- factor direction
    
- coverage / missing / eligibility 传递
    
- portfolio expression / neutralization
    

也就是：

> **在这个世界里，你定义了一个什么因子。**

---

所以最值得沉淀的一条原则是：

> **data_ready 冻结研究母体，signal_ready 冻结研究语义。**

如果这两层不分开，后面 train/test 很容易把：

- universe 变化
    
- eligibility 变化
    
- score 变化
    
- factor direction 变化
    

混成一团。

---

# 三、从 author / review 里提炼出的核心经验主线

我觉得可以沉淀成 7 条。

---

## 主线 1：signal_ready 最重要的是“禁止下游重新解释因子”

review gate 里最关键的几句其实是：

- `factor_direction` 必须冻结
    
- `factor_role` 必须冻结
    
- `factor_structure` 必须冻结
    
- `portfolio_expression` 必须冻结
    
- `neutralization_policy` 必须冻结
    
- 多因子组合公式必须是确定性的
    
- 不得把 test 才知道的 quantile / cutoff 偷写回 signal
    
- 不得依赖 train/test/backtest 结果回写
    

这背后说明 signal_ready 真正要防的是：

> **下游靠回测结果反过来“发明”因子定义。**

### 这是横截面研究里最危险的漂移源之一

比如：

- test 后才说“这个因子其实是反向的”
    
- backtest 后才说“其实做 group neutral 更合理”
    
- train 后才说“这个多因子权重应该学出来”
    
- 看到回测后再定义 cutoff / quantile / rank 区间
    

这些都不是 train 阶段行为，而是在**篡改 signal contract**。

### 经验沉淀

这个阶段必须强制回答：

> **如果 train/test 都还没开始，我能不能把这个因子的方向、角色、结构和消费方式完整讲清楚？**

如果不能，signal_ready 还没完成。

---

## 主线 2：factor_role 是一级语义，不是注释信息

这个 skill 最有价值的地方之一，是强制把 `factor_role` 显式化：

- `standalone_alpha`
    
- `regime_filter`
    
- `combo_filter`
    

这非常重要，因为很多研究失败，不是 factor 没效果，而是：

> **作者和 reviewer 根本没说清这个 factor 是拿来排序、拿来过滤，还是拿来做组合覆盖层的。**

### 典型错位

#### 错位 1：把 filter 伪装成 alpha

比如一个 volatility regime 条件，本来只是：

- 高波动不交易
    
- 趋势强时不用反转
    

却被包装成“独立 alpha 因子”。

结果 reviewer 会错误地拿：

- Rank IC
    
- bucket returns
    
- monotonicity
    

去评估它。

---

#### 错位 2：把 alpha 因子评成 filter

本来 residual 是横截面排序因子，却只看：

- gated vs ungated improvement
    

这样会把真正的 alpha 证据漏掉。

### 经验沉淀

要把这一条写成长期原则：

> **先定义因子角色，再定义证据标准，再定义 portfolio expression。**

不是反过来。

---

## 主线 3：factor_structure 的价值在于把“单因子”和“确定性多因子分数”严格分层

这阶段强制：

- `single_factor`
    
- `multi_factor_score`
    

并且规定：

- 第一版 `multi_factor_score` 只能是确定性公式
    
- 不允许 train-learned weights
    

这背后是在防一种非常常见的混乱：

> **把 signal contract 和 train contract 混在一起。**

### 常见坏味道

#### 1. 多因子写成一句“综合打分”

但没有明确：

- 子项是什么
    
- 哪些是 raw factor
    
- 哪些是 derived factor
    
- 最终 score field 是什么
    
- 组合公式是什么
    

---

#### 2. 用“之后训练学习权重”来定义 signal

这会导致 signal_ready 根本没有冻结因子，只冻结了一个愿望。

---

#### 3. 多因子结构实际上依赖后续样本表现

这等于把 signal_ready 延迟到 train/test。

### 经验沉淀

你这个阶段特别应该沉淀一条规则：

> **signal_ready 只允许冻结“静态可复现映射”，不允许冻结“未来再决定的组合过程”。**

也就是：

- 可以是固定线性组合
    
- 可以是 rank 加和
    
- 可以是布尔 gating
    
- 但不能是未来训练再学权重
    

---

## 主线 4：portfolio_expression 和 neutralization_policy 必须在 signal_ready 冻结，不要拖到 train/backtest

这个设计非常对，而且很值得强调。

因为很多团队会把：

- `long_short_market_neutral`
    
- `long_only_rank`
    
- `group_relative_long_short`
    
- `target_strategy_filter`
    
- `target_strategy_overlay`
    

留到后面再说。

但你这里明确规定：

- 角色决定合法表达集合
    
- neutralization 也必须显式冻结
    
- `group_neutral` 必须关联 group taxonomy version
    

这说明你已经抓住一个核心问题：

> **组合表达并不是纯交易层实现细节，它反过来定义了这个 factor 在研究语义上“到底押什么”。**

### 举例

同一个 factor：

- `long_only_rank`  
    押的是绝对涨跌
    
- `long_short_market_neutral`  
    押的是相对强弱
    
- `group_relative_long_short`  
    押的是组内相对关系
    

这三者不是同一个研究命题。

### 经验沉淀

signal_ready 阶段要形成固定心智：

> **factor definition = score semantics + consumption semantics**

不是只冻结一个数字列。

---

## 主线 5：factor_direction 必须被当成正式合同，而不是 downstream interpretation

review 里写得很硬：

- `factor_direction` 已冻结
    
- 不允许到 test/backtest 再解释
    
- `factor_direction` 不清楚是 blocking
    
- `factor_direction`、`final_score_field`、`score_combination_formula` 都不能空
    

这是非常关键的经验。

### 为什么这么重要

因为横截面研究最常见的“伪成功”之一就是：

> **先算一个东西，再到 test 阶段看它正向好还是反向好。**

这等于把 test 信息写回 signal。

### 常见坏味道

- “这个 residual 可能越高越好，也可能越低越好，先跑跑看”
    
- “这个 score 暂时先不写 direction，等 test 再决定”
    
- “先出 raw value，后面谁大谁好看结果再说”
    

### 经验沉淀

signal_ready 必须有一条长期纪律：

> **如果一个因子在进入 test 前还不能回答“高分到底意味着什么”，那它还不是正式因子。**

---

## 主线 6：eligibility / coverage / missing semantics 在 signal_ready 里必须继续传递，而不是默认继承

这点特别值得沉淀。

review 明确要求：

- 缺失值策略冻结
    
- coverage contract 冻结
    
- eligibility 传递规则冻结
    
- raw/derived/final score 字段写清
    
- factor_panel 非空且 `(date, asset)` 唯一
    

这说明 signal_ready 阶段虽然消费 data_ready，但不能偷懒地说：

> “eligibility 和缺失值在上游定义过了，这里默认沿用。”

因为 signal 层还会引入新的问题：

### 新问题包括

- 原始字段缺失如何传递到因子缺失
    
- 多因子组合时任一子项缺失怎么办
    
- coverage 是按 raw factor 还是 final score 统计
    
- factor 是否允许部分资产无值
    
- eligibility false 和 factor missing 是否区分
    
- group context 缺失时 group-neutral 是否还能做
    

### 经验沉淀

signal_ready 最好沉淀出一类固定模板：

- `input_field_missing_semantics`
    
- `derived_factor_missing_semantics`
    
- `eligibility_propagation_rule`
    
- `coverage_definition_for_final_score`
    

---

## 主线 7：signal_ready 里最值钱的产物不是 factor_panel，而是 factor_manifest

这点很容易被忽略。

很多人会把重点放在：

- 算出一个 `factor_panel.parquet`
    

但从你这套 skill 看，真正的一致性锚点其实是：

- `factor_manifest.yaml`
    
- `factor_contract.md`
    
- `factor_field_dictionary.md`
    
- `route_inheritance_contract.yaml`（review 里出现）
    
- `artifact_catalog.md`
    
- `field_dictionary.md`
    
- `run_manifest.json`
    

这背后说明一个很重要的方法论：

> **因子值本身不是因子合同，因子合同是“字段 + 角色 + 方向 + 结构 + 表达 + 继承关系 + 缺失/覆盖/传递语义”的整体。**

### 经验沉淀

要把 reviewer / author 都训练成先看 manifest，再看 panel，不要只看数值文件。

---

# 四、这个阶段最容易遇到的高频问题

我按类型给你归纳。

---

## A. 身份类问题

### 常见问题

- `factor_id` 命名不稳定
    
- 改了公式但没升 version
    
- `factor_direction` 空着
    
- `factor_direction` 与 `portfolio_expression` 不一致
    

### 风险

后面你根本分不清：

- 是同一个 factor 的版本更新
    
- 还是已经变成另一个 factor
    

### 经验沉淀

必须建立一套：

- id/version 规则
    
- 什么时候 bump version
    
- 什么时候开 child lineage
    

---

## B. 角色类问题

### 常见问题

- 把 `regime_filter` 写成 `standalone_alpha`
    
- `combo_filter` 没有 target strategy reference
    
- 非 standalone 但没有说明作用对象
    
- 一个 factor 同时想当 alpha 又想当 filter
    

### 风险

证据标准、gate、portfolio expression 全部错位。

### 经验沉淀

角色要先于评价指标冻结。

---

## C. 结构类问题

### 常见问题

- `multi_factor_score` 只写一句“综合打分”
    
- 没写 raw / derived / final score
    
- 子项来源不清
    
- 组合公式依赖未来训练
    

### 风险

signal_ready 名义上冻结了因子，实际上没冻结。

### 经验沉淀

多因子必须回答这几个问题：

- 子项有哪些
    
- 子项来自 data_ready 哪些字段
    
- 组合方式是什么
    
- 是 deterministic 吗
    
- final_score_field 是哪个
    

---

## D. 中性化与组上下文类问题

### 常见问题

- 写了 `group_neutral`，但没 group taxonomy reference
    
- 需要组内排序，但没冻结 group context
    
- taxonomy 版本和 data_ready 不一致
    
- neutralization_policy 和 portfolio_expression 冲突
    

### 风险

后面回测时 silently 补组信息，导致 signal contract 被改写。

### 经验沉淀

`group_neutral` 一旦出现，就必须把它当成依赖冻结资产，而不是一个配置选项。

---

## E. 传递语义类问题

### 常见问题

- eligibility 和 factor missing 混掉
    
- 组合后 coverage 下降但没说明
    
- 某些字段缺失时到底 drop 还是 NaN 不清楚
    
- input field 缺失语义无法追踪到 final score
    

### 风险

reviewer 不知道 coverage 变化来自：

- data_ready 母体
    
- signal 计算
    
- 组合逻辑
    
- group context 缺失
    

### 经验沉淀

必须把“信号层缺失”和“上游不可研究”分开。

---

## F. 伪完成类问题

### 常见问题

- 有 `factor_contract.md`，没真实 factor panel
    
- 有 panel，但只是 placeholder
    
- author skill 和 review skill 的输出名不一致
    
- 没有真实 provenance / run manifest
    

### 风险

signal_ready 形式完成，实质没有下游可消费资产。

### 经验沉淀

和 data_ready 一样，这阶段也要反复强调：

> **完成 = 可重建、可审计、可消费，不是“说清楚了”。**

---

# 五、我认为这套文档里有几处值得你主动统一/补强的地方

这个很重要，因为你要做 skill 体系，文档内一致性比局部规则更关键。

---

## 1. 输出文件名存在不一致

author Required Outputs 写的是：

- `factor_coverage.parquet`
    

但 mandatory discipline 又提到：

- `factor_coverage_report.parquet`
    
- `factor_group_context.parquet`
    

review Required Outputs 写的是：

- `factor_coverage_report.parquet`
    
- `route_inheritance_contract.yaml`
    

### 建议

你最好统一成一个明确集合，否则 reviewer 会遇到歧义：

- author 说自己交付了 `factor_coverage.parquet`
    
- reviewer 却按 `factor_coverage_report.parquet` 查 blocking
    

这类不一致非常容易制造假性 `FIX_REQUIRED`。

---

## 2. `route_inheritance_contract.yaml` 在 review 里要求，但 author 里没列 Required Outputs

这说明 review 在检查一个 author skill 没明确要求产出的东西。

### 建议

你要么：

- 把它补进 author Required Outputs  
    要么
    
- 从 review 必需输出里删掉  
    要么
    
- 写清它是否由系统自动生成、或是否包含在 `factor_manifest.yaml`
    

否则 author/reviewer contract 不闭环。

---

## 3. `portfolio_expression` 被要求来自 mandate 冻结，但 author Working Rules 看起来像在本阶段收敛确认

review checklist 写：

- `factor_role、factor_structure、portfolio_expression、neutralization_policy 均来自 mandate 冻结`
    

但 author skill 似乎允许在 signal_ready 阶段逐组确认这些内容。

### 这里要补一个清楚口径

到底是：

#### 方案 A

mandate 只冻结候选范围，signal_ready 正式冻结具体选择

还是

#### 方案 B

mandate 已冻结具体值，signal_ready 只能继承，不得重新选择

这两种流程含义完全不同。

我更建议用 A，但要写得明确：

> mandate 冻结允许集合与上层研究命题，signal_ready 冻结正式 factor contract 的具体实例。

---

# 六、如果沉淀经验，我建议重点做这几类“长期资产”

---

## 1. `csf_signal_ready` 的 anti-pattern 文档

可以专门列：

- 用 test/backtest 才知道的东西回写 signal
    
- role 不清，filter 伪装 alpha
    
- multi-factor 没有 deterministic formula
    
- direction 留到 downstream 再解释
    
- group_neutral 没有 taxonomy reference
    
- coverage / eligibility / missing semantics 混在一起
    
- factor panel 有值，但 contract 没冻结
    

这个会非常实用。

---

## 2. reviewer 的优先检查顺序

建议 reviewer 固定先查：

1. `factor_manifest.yaml`
    
2. `factor_contract.md`
    
3. `factor_panel.parquet`
    
4. coverage / missing / eligibility 传递
    
5. group context / taxonomy reference
    
6. run_manifest / provenance
    
7. catalog / dictionary
    

不要先看公式细节，否则容易陷入局部正确、整体失焦。

---

## 3. 一套 factor contract 模板

我建议长期固定这些字段：

- factor_id
    
- factor_version
    
- factor_role
    
- factor_structure
    
- factor_direction
    
- raw_factor_fields
    
- derived_factor_fields
    
- final_score_field
    
- score_combination_formula
    
- portfolio_expression
    
- neutralization_policy
    
- group_taxonomy_reference
    
- eligibility_propagation_rule
    
- missing_value_semantics
    
- coverage_definition
    
- route_inheritance
    

这会极大减少后续阶段漂移。

---

## 4. 一套 lineage 触发条件案例库

这个阶段特别值得沉淀：

### 哪些改动只是修文档

- 补说明
    
- 补 catalog
    
- 补 dictionary
    

### 哪些改动要 bump factor_version

- 方向改了
    
- 组合公式改了
    
- final score 字段改了
    
- role 语义改了
    

### 哪些改动必须 child lineage

- 因子结构从横截面改成时序
    
- 角色发生实质变化
    
- 因子不再研究同一命题
    

这个对研究 OS 会非常关键。

---

# 七、如果让我用一句话总结 signal_ready 最值得沉淀的经验

我会写成：

> **`csf_signal_ready` 最重要的经验沉淀，不是“如何算出一个 factor 值”，而是“如何把横截面因子的身份、角色、方向、结构、组合表达和中性化语义冻结成一个不依赖 train/test 回写、可重建、可审查、可下游直接消费的正式合同”。**

---

# 八、最后给你一个很实用的 reviewer / author 共用八问

1. **这个 factor 的高分到底代表什么，是否已冻结 direction？**
    
2. **它是 alpha、filter，还是 overlay，角色是否明确？**
    
3. **它是 single factor 还是 deterministic multi-factor score？**
    
4. **raw / derived / final score 三层字段是否清楚？**
    
5. **portfolio_expression 和 neutralization_policy 是否已冻结，而不是留给回测解释？**
    
6. **eligibility、coverage、missing semantics 是否从 data_ready 正式传递到 signal_ready？**
    
7. **如果启用 group_neutral，group context 和 taxonomy reference 是否已冻结？**
    
8. **下游 train/test 是否还能偷偷改变这个因子的定义？如果还能，这阶段就没完成。**
    

---
