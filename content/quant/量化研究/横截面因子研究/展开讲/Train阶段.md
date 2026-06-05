# TrainFreeze 阶段 - 详细展开

> 第一次阅读建议先看：[`./英文术语表.md`](./英文术语表.md)

## 1. 先说人话：TrainFreeze 为什么不是“训练赢家”

很多人一看到 `Train`，第一反应就是：

- 学权重
- 挑最好因子
- 拼最强组合

这在普通建模语境里没问题，但在有阶段治理的横截面研究里不够精确。

这里真正要冻结的是：

> 后续独立样本必须复用的训练窗尺子。

也就是：

- 预处理怎么做
- 中性化怎么做
- 分桶怎么做
- 调仓怎么做

这些是“尺子”，不是“最终胜利宣言”。

如果一句话概括：

> TrainFreeze 冻结的是评估与消费规则，不是因子最终排名。

---

## 2. TrainFreeze 回答的核心问题

## 2.1 预处理尺子是什么

比如：

- 去极值规则
- 标准化规则
- 缺失处理规则

这些看起来像小实现细节，但对横截面排序影响很大。

这里要分清一件事：

> 不是每个因子都必须做同一种预处理，但每个因子都必须明确“做什么 / 不做什么 / 为什么”。

也就是说，TrainFreeze 不能默认“代码里怎么写就怎么做”，而要把预处理规则正式冻结下来。

### 去极值规则是否必须做

去极值不是绝对必须，但大多数横截面因子都应该至少评估是否需要。

它解决的问题是：某个时点上少数资产的极端值会支配排序、z-score、回归中性化和分桶结果。

加密资产里尤其常见，例如新币、流动性很薄的币、异常成交或数据毛刺，都可能让某个因子值大到不合理。

常见做法有三类：

- `winsorize`：按分位数截尾，比如每个时点把因子值限制在 1% 到 99% 分位之间。
- `MAD clipping`：按中位数和 MAD 截断，比如超过 `median +/- 3 * MAD` 的值压回边界。
- `hard clipping`：按业务上限截断，比如某些比例类字段不允许超过预先定义的范围。

一般建议：

- 原始连续变量、成交量变化、收益率、波动率、估值类字段，通常要考虑去极值。
- 已经是 rank、bucket、布尔条件或天然有界的变量，可以明确写 `none`，但要说明原因。
- 去极值边界只能用训练窗规则确定，不能因为 Test / Backtest 结果不好再调。

### 标准化规则是否必须做

标准化也不是所有因子都必须做，但只要后面涉及多因子组合、中性化回归、跨期质量比较，通常就应该做。

它解决的问题是：不同因子或不同时点的数值尺度不一致。

例如动量可能是 `0.03`，成交额可能是几百万，如果直接相加或回归，尺度大的字段会主导结果。

常见做法有：

- `cross_sectional_zscore`：每个时点在资产截面内减均值、除标准差。
- `rank transform`：每个时点把因子值转成排名或百分位。
- `demean`：每个时点只减去截面均值，不除以标准差。
- `robust zscore`：用中位数和 MAD 替代均值和标准差。

一般建议：

- 单因子只看排序时，`rank transform` 往往更稳，因为它直接保留排序信息。
- 要做线性组合或回归中性化时，`cross_sectional_zscore` 或 robust z-score 更常见。
- 因子本身已经是横截面排名、分位数或标准分时，不要重复标准化，除非合同里明确说明。

### 缺失处理规则是否必须做

缺失处理必须做。这里的“必须”不是说必须填补缺失，而是必须明确缺失怎么处理。

缺失值如果不处理，下游会出现三类问题：

- 每个阶段默认行为不同，有的 drop，有的 fill，有的当 0。
- coverage 变化被误认为因子表现变化。
- 新上市资产或数据缺口被错误地当成低分 / 高分信号。

常见做法有：

- `leave_null_and_exclude`：保留空值，评估和分桶时排除。
- `leave_null_and_record_coverage`：保留空值，并在 coverage 报告里记录缺失模式。
- `cross_sectional_median_impute`：用同一时点截面中位数填补。
- `group_median_impute`：用同一 taxonomy group 内的中位数填补。
- `zero_fill`：填 0，只适合 0 有真实业务含义的字段。

一般建议：

- Signal 缺失通常优先 `leave_null_and_record_coverage`，不要随便填 0。
- 如果缺失来自“该资产当时不可研究”，应由 DataReady 的 eligibility 处理，而不是 TrainFreeze 硬填。
- 如果选择 impute，必须说明填补发生在什么截面、是否按 group、是否只用当时可见数据。

## 2.2 中性化尺子是什么

即：

- 是否中性化
- 对哪些暴露做中性化
- 用什么回归设定

这一步如果不冻结，后面很容易出现“结果不好就多加一个暴露再中性化一下”。

## 2.3 分桶尺子是什么

例如：

- 五分位还是十分位
- ties 怎么处理
- 最小截面样本量多少
- 小样本时是否跳过

这些都属于 train 里应固定下来的测量尺子。

## 2.4 调仓尺子是什么

注意，这里的 rebalance 不是完整执行方案，而是后面要复用的调仓语义。

比如：

- 频率是 1d、1w 还是信号触发
- 是否允许 schedule only
- 是否允许辅助触发条件

如果这一步不固定，后面很容易在看到 Test / Backtest 结果后反向改调仓规则。

## 2.5 先把这些英文字段说清楚

TrainFreeze 的英文字段都围绕“训练窗尺子”。这些字段不是证明因子赢了，而是规定后续独立样本必须用哪把尺子量。

| 字段 | 人话含义 | 为什么要用它 |
| --- | --- | --- |
| `preprocess_contract` | 预处理合同，规定因子值进入证据层前怎么整理。 | 去极值、标准化、缺失处理都会改变排序，必须在训练窗先冻结。 |
| `winsorize` | 去极值，把极端值压到某个边界。 | 防止少数异常值支配排序和分桶结果。 |
| `clipping` | 裁剪，把超过范围的数值截断。 | 明确极端值处理方式，避免后续看结果再换规则。 |
| `cross_sectional_zscore` | 每个时点在资产截面上做 z-score。 | 让同一时点不同资产的因子值可比较，不混入跨时间尺度差异。 |
| `rank transform` | 把原值变成排名或百分位。 | 减少量纲和极端值影响，直接服务横截面排序。 |
| `neutralization_contract` | 中性化合同。 | 规定是否剥离 beta、市值、group 等已知暴露，防止后续结果不好再加条件。 |
| `ranking_bucket_contract` | 分桶合同。 | 固定 quintile/decile、ties、小样本跳过等规则，保证 Test 不重估尺子。 |
| `quintile` / `decile` / `tertile` | 五分位、十分位、三分位分桶。 | 分桶数量不同会改变证据外观，必须先冻结。 |
| `ties` | 并列值处理规则。 | 因子值相同的资产如何排序会影响分桶边界。 |
| `min_cross_section_size` | 最小横截面样本数。 | 截面太小时 IC 和 bucket returns 不稳定，应按规则跳过。 |
| `rebalance_contract` | 调仓语义合同。 | 固定后续证据和回测怎么按时间消费信号，不能看成本后再改。 |
| `scheduled_only` | 只按预定频率调仓。 | 防止临时触发条件把训练尺子变成策略优化。 |
| `search_governance_contract` | 搜索治理合同。 | 说明哪些轴可在 TrainFreeze 比较，哪些 SignalReady 轴已锁死。 |
| `variant ledger` | 候选版本台账。 | 记录比较过哪些尺子，避免只保留赢家、丢掉审计路径。 |
| `reject ledger` | 拒绝台账。 | 说明哪些候选为什么被拒绝，防止团队重复探索同一失败方案。 |
| `csf_train_freeze.yaml` | 训练尺子的机器可读冻结文件。 | 下游 TestEvidence 直接消费它，不能再口头解释“我们大概这么做”。 |

---

## 3. 本阶段冻结的六组内容怎么理解

## 3.1 `preprocess_contract`

这一组解决的是：

> 因子值在进入任何正式证据层之前，应该怎样被“整理”。

典型包括：

- winsorize
- clipping
- cross-sectional zscore
- rank transform

重点不在“方法多高级”，而在“规则是否稳定且可复现”。

一个合格的冻结结果应该能写成类似这样：

```yaml
preprocess_contract:
  outlier_policy:
    method: mad_clipping
    scope: per_timestamp_cross_section
    threshold: 3.0
  standardization_policy:
    method: cross_sectional_zscore
    scope: per_timestamp_cross_section
    apply_after_outlier_policy: true
  missing_policy:
    method: leave_null_and_record_coverage
    exclude_from_bucket: true
    exclude_from_ic: true
```

这里每个字段都在锁一个关键问题：

- `outlier_policy`：极端值怎么处理。
- `scope`：规则是在每个时点的横截面内做，还是跨时间做。横截面因子通常应优先每个时点内处理。
- `standardization_policy`：是否标准化，以及用什么方法。
- `apply_after_outlier_policy`：先去极值再标准化，避免极端值污染均值和标准差。
- `missing_policy`：缺失值是保留、排除还是填补。
- `exclude_from_bucket` / `exclude_from_ic`：缺失样本是否参与分桶和 IC 计算。

## 3.2 `neutralization_contract`

这一组解决的是：

> 因子正式进入后续阶段前，是否以及如何剥离某些已知暴露。

例如：

- 对 `beta_30d` 做中性化
- 对 `log_market_cap` 做中性化
- 对某个 taxonomy group 做中性化

关键在于：

- 暴露来源是否明确
- 回归规则是否固定
- 下游不得因为表现不好就重写回归条件

## 3.3 `ranking_bucket_contract`

这一组解决的是：

> 后面正式比较横截面排序能力时，怎样把资产分桶。

这包含：

- quintile / decile / tertile
- 最小样本量
- ties 处理
- 缺失处理

这里需要特别澄清一个常见误解：

如果你冻结的规则是“每个时点按横截面分成五组”，  
那在 OOS 每期按同样的五分位规则分桶，是在**复用冻结规则**，不是在“偷看结果重估切点”。

真正不允许的是：

- 原本冻结五组，后来为了结果好看改成三组
- 原本冻结 rank quintile，后来改成自定义阈值
- 原本冻结等频分桶，后来改成不等宽分桶

## 3.4 `rebalance_contract`

这组是最容易被下游回写的部分之一。

因为一旦 Backtest 看起来成本高，大家就很容易说：

- 那把周频改成双周频
- 那把触发条件加一个门槛

这类动作一旦影响到训练窗定义的正式调仓尺子，本质上就不是“优化执行”，而是“改研究规则”。

所以这里必须明确：

- 频率
- 触发模式
- 辅助条件

## 3.5 `search_governance_contract`

这一组解决的是：

- 哪些轴在 TrainFreeze 里仍允许比较
- 哪些轴已经被 SignalReady 锁死
- 想改 signal expression / raw fields / score formula 时应如何拒绝
- variant ledger 和 reject ledger 如何记录

它的作用是防止把“训练候选”伪装成“重新定义因子”。

例如：

- 标准化方式、分桶方式、中性化设定可以是 train-governable axes
- `factor_expression`、`raw_factor_fields`、`derived_factor_fields`、`score_combination_formula` 通常不是

如果训练阶段发现必须改这些 signal 轴，正确动作不是继续调参，而是回到 SignalReady 或开新 lineage。

## 3.6 `delivery_contract`

这一组解决的是：

- 训练阶段真实产物长什么样
- 参数台账怎么记
- reject 项怎么保留

如果没有这层，后面只会看到一份“当前版本”，看不到此前尝试过什么、为什么被拒绝。

---

## 4. 为什么 TrainFreeze 不能偷看下游结果

这是本阶段最重要的纪律。

### 场景 1：先看 Test，再改 bucket

错误流程：

- 先在 Test 里看 bucket 单调性不好
- 再回 Train 把 quintile 改成 tercile

这会让 Test 不再是独立样本验证。

### 场景 2：先看 Backtest，再改 rebalance

错误流程：

- 先发现扣费后收益下降
- 再回来把调仓从日频改周频

如果 rebalance 是 train 尺子的一部分，这就是下游回写上游。

### 场景 3：先看下游结果，再加中性化条件

比如：

- 下游发现某段回撤大
- 再回 train 给因子加 size neutral

这等于把风险结果倒灌回训练尺子。

所以 TrainFreeze 的核心纪律是：

> Test 和 Backtest 只能消费 train 尺子，不能反过来雕刻 train 尺子。

---

## 5. 这一阶段应如何落地

## 5.1 在训练窗内估计尺子

这里的“估计”是可以发生的，但只能发生在训练窗。

例如：

- MAD 倍数如何取
- 标准化方案如何定
- 某些中性化回归设定如何固化
- bucket 规则是否满足最低截面稳定性

只要这些都是在训练窗内完成，并被明确记录，就属于本阶段职责。

## 5.2 把可治理轴与不可治理轴分开

这一点特别重要。

应该明确写出：

- 哪些轴在 TrainFreeze 仍可治理
- 哪些轴在 SignalReady 之后已经不可治理

例如：

- `factor_role`
- `factor_structure`
- `score formula`

这些通常已经是 SignalReady 冻结过的轴，不应再碰。

## 5.3 记录 variant ledger 与 reject ledger

训练阶段常常会有多个候选尺子。

比如：

- 两种标准化方案
- 两种 bucket 方案
- 两种中性化设定

关键不是不能比较，而是必须把：

- 每个候选的身份
- 是否被接受
- 被拒绝原因

都保留下来。

否则几周后团队会重复跑同一批无效探索。

---

## 6. 输出物应该长什么样

典型输出物包括：

- `csf_train_freeze.yaml`
- `train_factor_quality.parquet`
- `train_variant_ledger.csv`
- `train_variant_rejects.csv`
- `train_bucket_diagnostics.parquet`
- `train_neutralization_diagnostics.parquet`
- `csf_train_contract.md`
- `csf_train_freeze_gate_decision.md`
- `run_manifest.json`
- `artifact_catalog.md`
- `field_dictionary.md`

其中最关键的是 `csf_train_freeze.yaml`。

它不应该是一份模糊说明，而应显式区分：

```yaml
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

只要这份文件还在说“后续再定”，说明 freeze 还没发生。

---

## 7. 最常见的错误

## 7.1 把 TrainFreeze 写成组合设计阶段

错误写法：

- Top N 选币
- Long / Short 持仓数
- 仓位分配方式
- 执行窗口

这些已经属于组合与执行合同，更接近 BacktestReady。

## 7.2 只留最后一个版本

如果只有当前版本，没有 reject ledger，就会导致：

- 后面重复探索
- 审计时无法解释为何这样定
- 团队经验无法沉淀

## 7.3 没有写清不可治理轴

如果不显式写出哪些轴已经锁死，下游就很容易把上游变更包装成“训练调参”。

## 7.4 把训练窗结果写成正式胜利

训练窗里看到排序能力好，不代表正式胜利。  
这一步只能说明“我们把尺子定下来了”，不能说明“证据已经成立”。

---

## 8. 与下一阶段怎么衔接

进入 TestEvidence 后，只允许在独立样本上消费：

- preprocess 规则
- neutralization 规则
- bucket 规则
- rebalance 语义

不允许：

- 再学一遍这些尺子
- 再调一遍这些尺子
- 根据 Test 结果把这些尺子回写

如果 TestEvidence 发现这些尺子根本不合理，正确动作是：

- 回到 TrainFreeze 修正，并明确这会消费新的研究迭代
- 或按治理要求开新 lineage

而不是“边 test 边悄悄改”。

---

## 9. 最后给一个判断标准

TrainFreeze 合格与否，可以用一句话判断：

> 下游拿到这些 artifact 后，应该只能用它们去量独立样本，而不能继续把它们当成可随结果扭动的旋钮。

如果后面还能随便改，那就说明这个 freeze 其实没冻结住。
