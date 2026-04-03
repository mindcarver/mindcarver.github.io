# Mandate 阶段 - 详细展开

> 对应摘要版：`../简介/Mandate阶段.md`

## 1. 先说人话：Mandate 到底在干什么

Mandate 不是“开始研究”的同义词，也不是“先随便写点想法”。  
它的真正作用是：**把这条研究线的边界先锁死，再允许后续的人去做数据、信号、训练和验证。**

横截面因子研究里，最常见的失败不是代码写错，而是研究边界在过程中不断漂移：

- 看到结果不好，就换 universe
- 看到相关性弱，就改 horizon
- 看到某个时期效果差，就把切分窗口挪一挪
- 看到某些资产拖后腿，就把它们悄悄排除

Mandate 的存在，就是为了把这些“研究过程中看起来很合理、事后看起来全是污染”的动作拦在最前面。

如果一句话概括：

> Mandate 阶段只决定“研究合同”，不决定“研究胜负”。

---

## 2. Mandate 回答的核心问题

一篇合格的 Mandate，至少要能回答下面这几类问题。

### 2.1 研究对象是什么

- 这是哪一类市场
- 比较的是哪些资产
- 研究的对象是相对排序，还是单资产方向

横截面因子研究里，研究对象通常不是“BTC 明天涨不涨”，而是：

> 在同一个时点，哪些资产相对更强，哪些相对更弱？

所以这里必须明确写出：

- `research_route = cross_sectional_factor`
- 比较口径是 `date x asset`
- 后续评价的是排序能力，而不是单资产命中率

### 2.2 研究边界是什么

边界主要有四类：

- Universe 边界
- 时间边界
- 数据边界
- 执行表达边界

只要这四类边界有一个写不清，后面就一定会出现“看结果再修边界”的冲动。

### 2.3 允许改什么，不允许改什么

Mandate 的价值不只在“写了什么”，还在“哪些东西一旦写下去就不能再偷偷改”。

比如：

- Universe 口径改了，通常不是正常迭代，而是研究线变了
- Holdout 时间窗改了，通常不是优化，而是污染最终验证
- 交易表达从 `long_only_rank` 改成 `long_short_market_neutral`，不是小修，是研究对象变了

这些都应在 Mandate 里预先定义为：

- 正常推进
- 需要 review
- 必须开 child lineage

---

## 3. Mandate 里真正该冻结的东西

下面逐项展开。

## 3.1 `research_intent`

这是“为什么研究这条线”的正式描述。

不要写成空话，比如：

- “看看这个因子能不能赚钱”
- “探索一下有没有 alpha”

这种写法没有约束力，也没有研究可复用性。

更好的写法应包含：

- 市场观察
- 假设机制
- 预期受益对象
- 可能失效的场景

例如：

```yaml
research_intent:
  observation: "流动性改善阶段，小市值高换手资产更容易在横截面上跑赢"
  hypothesis: "20 日成交活跃度变化能解释后续 1d 排序差异"
  counter_hypothesis: "该现象只是 beta 或小市值暴露，并不构成独立 alpha"
```

这里的重点不是文采，而是后续能不能明确反驳自己。

## 3.2 `research_route`

这一项必须非常明确。

在横截面因子研究里，最容易发生的混乱是：

- 文档标题说自己在做横截面
- 内容却开始写单资产 hit rate、best horizon、择时命中率

所以 `research_route` 的作用，是把这条线从一开始就钉死。

对于这组文档，应明确为：

```yaml
research_route: cross_sectional_factor
```

一旦这项写清，下游就不该再引入：

- 单资产触发语义
- 单资产开平仓胜率
- 时序主线的 best_h 叙事

## 3.3 `scope_contract`

这项解决的是“研究范围到底有多大”。

需要写清：

- 只做单因子，还是允许多因子确定性打分
- 只研究某一市场，还是允许跨 venue
- 是否允许长短组合
- 是否允许 regime filter

这里不要为了显得灵活而写成：

- “后续可按结果决定”
- “必要时扩大范围”

这类写法会让后面的每一步都失去治理约束。

## 3.4 `target_market / universe 口径`

这是 Mandate 里最关键的部分之一。

需要明确：

- 市场：现货、合约、股票还是期货
- 交易 venue：一个交易所还是多个
- 入选口径：按市值、成交额、流动性还是其他
- 排除口径：稳定币、杠杆代币、新上市资产、低价垃圾币等

这一项必须写到“别人按文档也能重建同一 universe”为止。

一个合格的 universe 定义，通常至少要包括：

```yaml
target_market:
  market: crypto
  venue_scope:
    - binance_spot
  universe_rule: "Top 50 by 24h liquidity after exclusions"
  exclusions:
    - stablecoins
    - leveraged_tokens
    - listing_days < 30
```

### 为什么 universe 必须先冻结

因为横截面结果极度依赖比较对象。

你把样本池从 Top 50 改成 Top 200，看起来只是“覆盖更全面”，但实际上会同步改变：

- 排序难度
- 流动性结构
- 横截面离散度
- 因子覆盖度
- 下游容量和成本

所以 Universe 变化不是小修，而是研究对象变化。

## 3.5 `time_split`

很多量化文档写到这里就开始模糊，这是大坑。

你必须明确：

- Train 窗口
- Test 窗口
- Backtest 窗口
- Holdout 窗口

并且要写成具体时间边界，而不是：

- “训练期较长”
- “后面留一些数据做验证”

因为只要时间边界不死，后面就一定会有人在结果差的时候想“再挪一点”。

建议写法：

```yaml
time_split:
  train:
    start: "2021-01-01"
    end: "2023-12-31"
  test:
    start: "2024-01-01"
    end: "2024-06-30"
  backtest:
    start: "2024-07-01"
    end: "2024-09-30"
  holdout:
    start: "2024-10-01"
    end: "2024-12-31"
```

### 为什么 Holdout 必须在 Mandate 就定掉

因为如果 Holdout 不是一开始就定死，后面任何“延长一下”“换个更代表性的窗口”的动作，都会变成把最终验证窗口当成可消费资源。

## 3.6 `data_contract`

DataReady 阶段会把数据真正落成 panel，但 Mandate 阶段必须先写清数据合同。

至少应写出：

- 数据源
- bar size
- 时区
- 对齐语义
- 数据发布时间语义

例如：

```yaml
data_contract:
  bar_size: 1h
  timezone: UTC
  price_source: spot_klines
  liquidity_source: rolling_volume_24h
  signal_timestamp: close_time
  available_after: next_bar_open
```

这一步的重点是：

> DataReady 以后只能实现这个合同，不能重新发明一个版本。

## 3.7 `execution_contract`

这一项不要求你在 Mandate 就把所有交易细节写死，但至少要冻结“研究打算如何被消费”。

比如：

- `long_short_market_neutral`
- `long_only_rank`

这两种表达差异非常大。

如果这一步不写，后面很容易出现这样的偷换：

- 先用多空表达验证
- 结果一般
- 再换成长多
- 然后说“本质上还是一个信号”

这会导致研究对象漂移。

## 3.8 `kill_criteria`

很多文档写到这里最敷衍，但其实这是 Mandate 的治理核心。

Kill criteria 要解决的是：

- 什么情况说明这条线不值得继续
- 什么情况说明不是结束，而是该开 child lineage

例如：

- Universe 无法稳定重建
- 数据来源无法回放
- 交易表达与原始假设冲突
- 关键定义轴必须被修改

只要这些条件成立，就不能把后续动作包装成“正常优化”。

---

## 4. Mandate 阶段最容易写错的地方

## 4.1 把统计门槛写进 Mandate

最常见错误就是把：

- `IC > 0.03`
- `Sharpe > 1.0`
- `容量 > 500 万`

写成 Mandate 的 formal gate。

这类内容看起来很“量化”，但其实是错位的。

原因不是这些指标没用，而是：

- 这些指标要依赖下游证据才有资格判断
- 一旦写在 Mandate，人就会倾向于倒推研究过程去满足它

正确做法是：

- Mandate 只定义后面要检什么
- 不在 Mandate 决定后面会不会通过

## 4.2 先试几个 horizon 再冻结一个

这是另一个非常高频的污染源。

如果你在 Mandate 阶段写：

```python
for h in [1, 4, 24, 72]:
    ...
```

然后选一个“最好”的 horizon 写回合同，本质上就是把研究结果倒灌回研究授权。

正确思路是：

- horizon 来自假设与执行表达
- 不是来自先跑结果再回填

## 4.3 Universe 留口子

比如写：

- “先用 Top 50，不行再扩到 Top 100”
- “必要时调整样本池”

这种话等于没冻结。

一篇看起来完整但 universe 可随结果变动的 Mandate，实际上没有治理价值。

## 4.4 把横截面问题写成时间序列问题

错误表述包括：

- “预测 BTC 明天方向”
- “提高单币 hit rate”
- “best horizon 是多少”

横截面研究关心的是同一时点的相对强弱排序。  
只要正文老是滑回单资产叙事，后面所有阶段都会一起跑偏。

---

## 5. Mandate 的输出应该长什么样

最少应该有：

- `mandate.md`
- `research_scope.md`
- `research_route.yaml`
- `time_split.json`
- `stage_completion_certificate.yaml`

这里最重要的不是文件名，而是文件是否真的能约束后续行为。

比如 `research_route.yaml` 必须是机器可读的，不应只在 Markdown 里口头说“这是横截面研究”。

---

## 6. Mandate 结束后，下游应该怎么消费

DataReady 的职责不是“再理解一遍 Mandate”，而是**严格实现 Mandate**。

所以从 Mandate 到 DataReady，最关键的交接要求是：

- 不能静默改 universe
- 不能静默改时间边界
- 不能静默改研究路线
- 不能静默改执行表达

如果 DataReady 发现 Mandate 本身不可执行，正确动作应是：

- 回到 Mandate 修正
- 或者开新 lineage

而不是让 DataReady 悄悄替 Mandate 做决定。

---

## 7. 最后给一个判断标准

怎么判断一篇 Mandate 到底合不合格？

最简单的方法是问三遍：

1. 我们在研究什么？
2. 哪些东西后面不能改？
3. 如果非改不可，是正常迭代还是另起一条线？

如果这三问答不清，Mandate 就还没写完。
