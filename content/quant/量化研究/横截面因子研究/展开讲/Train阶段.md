# TrainFreeze 阶段 - 详细展开

> 对应摘要版：`../简介/Train阶段.md`

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

---

## 3. 本阶段冻结的五组内容怎么理解

## 3.1 `preprocess_contract`

这一组解决的是：

> 因子值在进入任何正式证据层之前，应该怎样被“整理”。

典型包括：

- winsorize
- clipping
- cross-sectional zscore
- rank transform

重点不在“方法多高级”，而在“规则是否稳定且可复现”。

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

## 3.5 `delivery_contract`

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
- `train_quality.parquet`
- `train_variant_ledger.csv`
- `train_rejects.csv`
- `train_gate_decision.md`

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
