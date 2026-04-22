这组 `csf_train_freeze`，我觉得是你整套流程里**最容易被做歪**、但也**最能拉开体系成熟度**的阶段。

如果说：

- `csf_data_ready` 冻结的是**研究母体**
    
- `csf_signal_ready` 冻结的是**正式因子合同**
    

那么：

> `csf_train_freeze` 冻结的就是  
> **“这个已定义好的因子，后续要用什么尺子去清洗、去中性化、去分桶、去再平衡，但又不能借此偷偷重定义因子本身。”**

这阶段最值得沉淀的不是参数表，而是**训练治理边界**。

---

# 一、先给这个阶段一个准确定位

这阶段不是：

- 找最终赢家
    
- 选最赚钱参数
    
- 用训练集把因子“调到最好看”
    

它真正做的是：

> **把 signal 已经冻结好的 factor contract，转换成 downstream test 只能复用、不能再争论的 train rules。**

也就是冻结五类尺子：

- `preprocess_contract`
    
- `neutralization_contract`
    
- `ranking_bucket_contract`
    
- `rebalance_contract`
    
- `delivery_contract`
    

所以一句话：

> `csf_train_freeze` 冻结的不是“结果最好的一版”，而是“后续允许怎么比较、怎么分组、怎么执行的正式训练尺子”。

---

# 二、它和前两个阶段的本质区别

这个特别值得沉淀，不然后面很容易串层。

## `data_ready`

回答：

> 研究母体是什么？

## `signal_ready`

回答：

> 因子是什么？

## `train_freeze`

回答：

> **在不改因子定义的前提下，这个因子后续用什么训练口径被消费？**

也就是：

- 怎么 winsorize
    
- 怎么 standardize
    
- 怎么 neutralize
    
- 怎么 bucket
    
- 怎么 rebalance
    
- 哪些变体允许比较
    
- 哪些变体根本不合法
    

所以这里不是“重新发明 signal”，而是：

> **对 signal 的可治理消费轴做冻结。**

---

# 三、我觉得这阶段最核心的经验主线，有 8 条

---

## 主线 1：这阶段最重要的不是“调参”，而是“划边界”

你这份文档里最强的一条纪律其实是：

- `signal_ready` 已冻结的 factor expression / transform 轴  
    **不得在本阶段重新作为候选搜索轴**
    
- 如果 inherited variant 想改：
    
    - `fragility_score_transform`
        
    - `raw_factor_fields`
        
    - `derived_factor_fields`
        
    - `score_combination_formula`
        

就必须 reject，并说明应该回到 `csf_signal_ready`

这条特别关键。因为 train 阶段最常见的污染就是：

> **名义上在调训练尺子，实际上在偷偷改 signal。**

### 典型伪装

- “只是换一个 transform”
    
- “只是换一下 raw factor field”
    
- “只是把组合公式改一版”
    
- “只是把 score 从 rank sum 改成 weighted sum”
    

这些很多都不是 train 轴，而是 **signal 轴**。

### 值得沉淀的经验

这阶段一定要形成一个固定心智：

> **train_freeze 只允许治理消费尺子，不允许重写因子表达。**

---

## 主线 2：train_freeze 的目标不是“宣布 winner”，而是“冻结可复用尺子”

你 author 里写得很对：

- 当前阶段只冻结 train 尺子，不替因子宣布赢家
    
- 不得借用 test/backtest 结果回写 train 尺子
    

review 里也对应得很硬：

- 不得根据 test/backtest 结果回写 train freeze
    
- 在 train 内直接用收益最大化选 final winner 不合法
    
- search governance 只用于排除荒谬区间或不可研究 variant，不得以收益最大化方式选胜者
    

这反映出一个非常成熟的区分：

> **train_freeze 是“治理层”，不是“冠军评选层”。**

### 这阶段真正能做什么

- 冻结合理的 winsorize 区间
    
- 冻结 standardize 规则
    
- 冻结 neutralization 大类、window、taxonomy reference
    
- 冻结 bucket schema / quantile count / tie-break / min names
    
- 冻结 rebalance frequency / lag / overlap policy
    
- 冻结 admissible variants 和 reject ledger
    

### 不能做什么

- 看哪个 variant 收益最高就留哪个
    
- 用下游结果选最终 quantile
    
- 用 test 表现反推 neutralization 条件
    
- 用 backtest 去定 rebalance/lag
    

### 值得沉淀的经验

一句话：

> **这阶段的“好”不是收益高，而是口径稳、边界清、可复用、可审计。**

---

## 主线 3：variant governance 是这阶段最值钱的资产之一

你这里强制要求：

- `train_variant_ledger.csv`
    
- `train_rejects.csv` / review 里叫 `train_variant_rejects.csv`
    
- 所有候选组合都必须有身份记录
    
- reject 不是静默丢弃，而是显式记账
    
- `candidate_variant_ids`、`kept_variant_ids`、`train_governable_axes` 都不能空
    

这说明你已经抓住了一个训练阶段的核心问题：

> **研究里最危险的不是选错，而是“选过什么、丢过什么、为什么丢”没人知道。**

### 常见坏味道

- 只保留最后一个赢家，没有候选轨迹
    
- 调过很多版，但没 ledger
    
- 某些 variant 被丢掉，没有拒绝原因
    
- 某些 variant 其实非法跨了 signal 边界，但没被显式打回
    

### 值得沉淀的经验

你这个阶段其实特别适合沉淀成：

> **variant governance protocol**

至少要固定这几个概念：

- candidate variants
    
- admissible variants
    
- kept variants
    
- rejected variants
    
- reject reasons
    
- signal-axis violation
    
- non-governable-axis breach
    

---

## 主线 4：preprocess / neutralization / bucket / rebalance 必须语义分离

这个也是你文档里非常好的点。

你要求 `csf_train_freeze.yaml` 显式区分：

- `preprocess_rules`
    
- `neutralization_rules`
    
- `ranking_bucket_rules`
    
- `rebalance_rules`
    
- `auxiliary_conditions`
    

这是非常必要的，因为训练阶段最容易出的问题就是**不同尺子混在一起**。

### 常见混法

#### 1. preprocess 和 coverage floor 混一起

例如：

- 缺失太多直接 drop
    
- 但又把 drop 写成 winsorize 的附属逻辑
    

#### 2. neutralization 和 factor transform 混一起

例如：

- 先 group demean 再当 signal transform 写回去
    

#### 3. bucket 规则和 rebalance 规则混一起

例如：

- 每 5 天 rebalance，但 bucket 划分又跟 overlap 策略耦合
    

#### 4. auxiliary conditions 到处漂

例如：

- 最小名称数
    
- coverage floor
    
- 最小流动性
    
- 最小组内样本数
    

写散在不同脚本里

### 值得沉淀的经验

这阶段一定要坚持：

> **每个尺子回答不同问题，不能靠一个“训练配置总表”模糊带过。**

---

## 主线 5：neutralization 在 train_freeze 里必须被当成独立合同，而不是一个开关

review 里写得很清楚：

- neutralization policy、beta estimation window、group taxonomy reference 已冻结
    
- neutralization 如存在，必须有独立合同和诊断
    
- neutralization 大类改变要 child lineage
    

这特别重要，因为很多研究里 neutralization 被当成：

```text
on / off
```

但实际上它是一个完整 contract：

- neutralization policy 是什么
    
- beta 怎么估
    
- 用什么窗口
    
- group taxonomy 从哪来
    
- market beta neutral 和 group neutral 分别依赖什么
    
- neutralization 后如何诊断
    

### 常见坏味道

- 写了 `group_neutral`，没 taxonomy reference
    
- 写了 `market_beta_neutral`，没 beta estimation window
    
- neutralization 改了，但没 bump lineage
    
- 做了 neutralization，却没有独立 diagnostics
    

### 值得沉淀的经验

neutralization 不是“预处理附加项”，而是：

> **训练尺子中的一级合同对象。**

---

## 主线 6：bucket 规则是正式研究语义，不是回测显示设置

这个阶段很值得强调 bucket 这一块。

review 明确要求冻结：

- bucket schema
    
- quantile count
    
- tie-break rule
    
- min names per bucket
    

这说明 bucket 不是“图表怎么画”，而是：

> **横截面比较方式的正式定义。**

### 为什么这很关键

因为很多人会在 test/backtest 后才去改：

- 分几桶
    
- top/bottom 用几分位
    
- ties 怎么打破
    
- 组内样本太少怎么办
    

这些都不应该留到后面。

### 常见问题

- quantile count 没定
    
- ties 没规则
    
- min names per bucket 没写
    
- coverage 太低时 bucket 如何退化没定义
    

### 值得沉淀的经验

bucket 规则必须被视为：

> **cross-sectional comparison contract**

不是展示层参数。

---

## 主线 7：rebalance / lag / overlap 是 train_freeze 的核心，不是 execution 小细节

review 明确写：

- rebalance frequency、signal lag 和 overlap policy 已冻结
    
- rebalance / lag / overlap 口径未冻结是 blocking
    

这点特别值钱，因为很多研究把这些留到回测层去“试”。

但实际上：

- `rebalance frequency`
    
- `signal lag`
    
- `overlap policy`
    

会直接改变你在研究什么。

### 举例

同一个 factor：

- T+0 消费 vs T+1 消费  
    不是同一研究命题
    
- 每天 rebalance vs 每周 rebalance  
    也不是同一个消费尺子
    
- overlapping holds vs non-overlapping  
    直接改变收益分布和比较语义
    

### 值得沉淀的经验

这阶段需要固定一句话：

> **rebalance / lag / overlap 不是 execution tuning，而是 train contract 的组成部分。**

---

## 主线 8：真正的 train governance，不是“搜索最优”，而是“排除不合法和荒谬区间”

review 里有一句很成熟：

> search governance 只用于排除荒谬区间或不可研究 variant，不得以收益最大化方式选胜者

我觉得这句话特别值得单独沉淀。

因为它把 train 阶段的 search 角色定义得很清楚：

### train search 可以做

- 排除明显不合法的 quantile count
    
- 排除 coverage 太差的 bucket schema
    
- 排除 min names 不满足的分组规则
    
- 排除违反 signal boundary 的 inherited variants
    
- 排除 neutralization 无法成立的配置
    

### train search 不能做

- 用收益最大化挑最终规则
    
- 用下游 performance 反写训练尺子
    
- 用 test/backtest 结果重新界定 governable axes
    

### 值得沉淀的经验

train governance 的核心是：

> **约束搜索空间，而不是用下游结果定义搜索空间。**

---

# 四、这个阶段最容易遇到的高频问题

---

## A. 边界漂移类问题

### 常见问题

- 训练时偷偷改了 `score_combination_formula`
    
- 把 `raw_factor_fields` 当 train axis
    
- inherited variant 变更 `derived_factor_fields`
    
- 改了 transform 还自称 train variant
    

### 风险

signal_ready 形同虚设。

### 经验沉淀

所有非 governable 轴，一旦被碰，必须：

- reject
    
- 记账
    
- 说明需重开 `csf_signal_ready`
    

---

## B. 参数治理类问题

### 常见问题

- 只有保留者，没有 reject ledger
    
- candidate ids 没记录
    
- variant id 不唯一
    
- 同一个 variant 改了配置但没换身份
    
- rejects 无原因
    

### 风险

训练轨迹不可审计，后面很容易“合理化赢家”。

### 经验沉淀

variant ledger 是正式产物，不是实验笔记。

---

## C. neutralization 类问题

### 常见问题

- neutralization policy 有，但没 window
    
- group neutral 有，但没 taxonomy reference
    
- neutralization 后没有 diagnostics
    
- neutralization 改大类却没 child lineage
    

### 风险

后续 test 才知道 neutralization 实际在干什么。

### 经验沉淀

neutralization 需要单独诊断、单独合同、单独 lineage 规则。

---

## D. bucket / ranking 类问题

### 常见问题

- quantile count 未冻结
    
- ties 处理未定义
    
- min names per bucket 未定义
    
- 组内排序还是全市场排序没说清
    
- ranking rules 和 eligibility floor 混在一起
    

### 风险

test 时 silently 变桶、变切点、变比较方式。

### 经验沉淀

bucket schema 是正式训练尺子的一部分。

---

## E. rebalance 类问题

### 常见问题

- rebalance frequency 没冻结
    
- lag 没定义清楚
    
- overlap policy 没定义
    
- downstream 才决定 hold overlap
    

### 风险

test/backtest 实际上在重新定义 train contract。

### 经验沉淀

rebalance 规则必须在 train_freeze 结束前讲清。

---

## F. 伪完成类问题

### 常见问题

- 有 YAML，没有真实质量证据
    
- 只有 kept variants，没有 rejects
    
- 有 ledger，但 stage-local program 是 thin wrapper
    
- replay 入口不真实
    
- 当前 repo 里并没有实际生成产物的代码
    

### 风险

形式完成，实质不可审计。

### 经验沉淀

这阶段和前两个阶段一样，完成定义必须是：

- 可重建
    
- 可追溯
    
- 可解释
    
- 不靠口头补充
    

---

# 五、我觉得这组文档里有几处很值得你主动统一

这个很重要，因为现在 author / review 之间已经出现了几处不一致。

---

## 1. 输出文件名不一致

author Required Outputs：

- `train_quality.parquet`
    
- `train_variant_ledger.csv`
    
- `train_rejects.csv`
    
- `train_gate_decision.md`
    

review Required Outputs：

- `train_factor_quality.parquet`
    
- `train_variant_ledger.csv`
    
- `train_variant_rejects.csv`
    
- `train_bucket_diagnostics.parquet`
    
- `csf_train_contract.md`
    

这里已经明显不闭环。

### 建议统一

至少统一这几项：

- `train_quality.parquet` vs `train_factor_quality.parquet`
    
- `train_rejects.csv` vs `train_variant_rejects.csv`
    
- author 没写 `train_bucket_diagnostics.parquet`
    
- author 没写 `csf_train_contract.md`
    
- author 有 `train_gate_decision.md`，review 没列为必需输出
    

这类不一致很容易让 reviewer 给出假性 blocking。

---

## 2. author 没显式要求 `run_manifest.json`，但你前两个阶段都很强调 provenance

这一阶段 author 特别强调：

- stage-local stage program
    
- 不能 thin wrapper
    
- 必须 replay 入口真实
    

但 Required Outputs 没列 `run_manifest.json`。

review 的共用输入里又默认有 `run_manifest.json`。

### 建议

这一阶段也应显式加入 `run_manifest.json`，不然 provenance contract 不完整。

---

## 3. author 用 `delivery_contract`，但 review 里其实在审程序真实性，不只是 delivery

这一阶段最特别的是：

- 明确要求 stage-local 程序
    
- 不能 thin wrapper
    
- 必须是真实产生产物的程序
    

这比前两个阶段更强。

### 建议

你可以考虑把 `delivery_contract` 明确扩成：

- `delivery_and_provenance_contract`  
    或者在其下明确一块：
    
- `stage_program_contract`
    

因为这阶段程序真实性本身就是 formal gate 的重要组成部分。

---

# 六、如果要沉淀长期经验资产，我建议优先做这几类

---

## 1. `csf_train_freeze` anti-pattern 文档

非常值得固定下来：

- 用 test/backtest 结果回写 train 尺子
    
- signal 轴伪装成 train 轴
    
- 只有 kept，没有 rejects
    
- neutralization 有开关，没有合同
    
- bucket 规则未冻结
    
- rebalance / lag / overlap 留给下游
    
- 用收益最大化选 final winner
    
- thin wrapper 冒充 stage program
    

---

## 2. reviewer 优先检查顺序

建议 reviewer 固定先看：

1. `csf_train_freeze.yaml`
    
2. `train_variant_ledger.csv`
    
3. `train_variant_rejects.csv`
    
4. `train_factor_quality.parquet`
    
5. `train_bucket_diagnostics.parquet`
    
6. neutralization contract / diagnostics
    
7. rebalance / lag / overlap rules
    
8. provenance / stage-local program
    

这样 reviewer 不会先被表面参数表带偏。

---

## 3. 一套 train governance 模板字段

强烈建议长期固定：

- `frozen_signal_contract_reference`
    
- `train_governable_axes`
    
- `non_governable_axes_after_signal`
    
- `non_governable_axis_reject_rule`
    
- `preprocess_rules`
    
- `neutralization_rules`
    
- `ranking_bucket_rules`
    
- `rebalance_rules`
    
- `auxiliary_conditions`
    
- `candidate_variant_ids`
    
- `kept_variant_ids`
    
- `reject_variant_ids`
    
- `reject_reasons`
    

---

## 4. 一套 child-lineage 触发案例库

这个阶段尤其重要：

### 必须 child lineage 的例子

- 用 test/backtest 回写 train 尺子
    
- 预处理语义改变
    
- neutralization 大类改变
    

### 只需本 stage rollback / 修正的例子

- ledger 补充不完整
    
- reject reason 补写
    
- bucket diagnostics 补产出
    
- 文档和 YAML 对齐
    

这会极大减少 reviewer / author 的争议。

---

# 七、如果让我用一句话总结这个阶段最该沉淀什么

我会写成：

> **`csf_train_freeze` 最重要的经验沉淀，不是“怎么在训练集里把因子调到最好”，而是“如何把 signal 已冻结的因子在不被重新发明的前提下，冻结成一套可审计、可复用、可拒绝非法变体、且不能被 test/backtest 回写的正式训练尺子治理合同”。**

---

# 八、最后给你一个 author / reviewer 共用的 8 问

1. **这次改的是 train 尺子，还是其实已经改到 signal 轴了？**
    
2. **哪些轴允许治理，哪些轴一碰就必须 reject / 回到 signal_ready？**
    
3. **所有 candidate variants 是否都有身份，所有 rejects 是否都有原因？**
    
4. **preprocess / neutralization / bucket / rebalance 是否真正分层冻结？**
    
5. **neutralization 是否有独立合同、window、taxonomy reference 和 diagnostics？**
    
6. **bucket schema、tie-break、min names、coverage floor 是否都已冻结？**
    
7. **rebalance frequency、lag、overlap 是否已冻结，而不是留给 test/backtest？**
    
8. **下游 test 能不能再偷偷改这些训练尺子？如果还能，这阶段就没完成。**
    

---

你把 `csf_test_evidence` 也贴出来的话，我可以继续按同样方式，把四个阶段串成一条完整的方法论主线，顺便帮你找出整个 skill 体系里哪些字段名、artifact 名和 gate 语义还需要统一。