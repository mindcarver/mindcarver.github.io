# DataReady 阶段 - 详细展开

> 对应摘要版：`../简介/DataReady阶段.md`

## 1. 先说人话：DataReady 在锁什么

DataReady 不是“把数据下载下来”这么简单。  
它锁的是后续整个横截面研究世界里的**公共底座**。

如果 Mandate 锁的是研究合同，那么 DataReady 锁的是：

> 后续所有因子、所有训练、所有验证，究竟建立在哪一张 `date x asset` 的面板世界上。

一旦这个底座不统一，后面就会出现这些问题：

- 因子 A 用的是一套 universe
- 因子 B 用的是另一套 universe
- 训练时按一个 eligibility，测试时又按另一个 eligibility
- 某些分类字段每个人自己算一遍

最后研究线表面上在比较因子，实际上在比较不同数据世界。

---

## 2. DataReady 回答的核心问题

## 2.1 面板主键是什么

这是最基础也最容易被忽视的问题。

横截面研究的底座不是普通时间序列表，而是 `date x asset` 面板。

所以你必须明确：

- `date_key`
- `asset_key`
- `panel_frequency`
- `timezone`

如果这些主键是隐式的，下游每个人都会按自己的理解做对齐。

## 2.2 哪些资产在什么时点属于研究 universe

这里不是问“资产有没有数据”，而是问：

> 在这个时点，它是不是本来就属于研究对象？

这就是 `asset_universe_membership` 的意义。

## 2.3 哪些观测是基础可研究的

这一步和 universe membership 不是一回事。

例如：

- 某资产属于研究 universe
- 但某个时间点缺关键字段
- 或者上市天数不足
- 或者某些基础约束不满足

这时它仍然属于 universe，但该观测不一定 `eligible`。

所以要把：

- 成员关系
- 基础可研究性

分开记录。

## 2.4 哪些共享派生层后续要复用

很多字段不是因子本身，但后面会被广泛复用，比如：

- market cap
- rolling volume
- listing days
- taxonomy 标签

如果这些共享字段没有在 DataReady 统一生成，后面大家就会各自派生一版。

---

## 3. DataReady 真正该冻结的五组内容

## 3.1 `panel_contract`

这组解决的是“数据面板长什么样”。

至少要明确：

- 每一行 / 每一列的主键语义
- 频率是 `1h`、`4h` 还是 `1d`
- 时间戳到底表示 bar close 还是 bar open
- 时区是否统一为 UTC

如果这一步没讲清，SignalReady 里时间语义必然会乱。

典型字段：

```yaml
panel_manifest:
  date_key: timestamp
  asset_key: symbol
  panel_frequency: 1h
  timezone: UTC
  timestamp_semantics: close_time
  coverage_rule: "after eligibility_base_mask"
```

## 3.2 `taxonomy_contract`

这组解决的是资产分组和分类底座。

它不是因子定义的一部分，但后面很多事情会用到，比如：

- `group_neutral`
- 行业 / 板块中性
- category exposure 检查

在加密场景里，taxonomy 可能包括：

- stablecoin
- layer1
- defi
- meme
- exchange_token

关键点不是分类是否完美，而是：

- 分类是否版本化
- 分类是否可重现
- 分类变化是否可追溯

## 3.3 `eligibility_contract`

这组最容易被滥用。

它只负责记录“基础可研究性”，不能偷偷混进具体因子逻辑。

例如，下面这些可以属于 base eligibility：

- listing days >= 30
- 有基础价格与成交量数据
- 非稳定币
- 非杠杆代币

但下面这些不该属于 base eligibility：

- 动量值非空
- RSI 可计算
- 某因子值大于某阈值

后者已经是信号层逻辑。

## 3.4 `shared_feature_base`

这一组的目标是统一共享派生字段。

例如：

- `log_market_cap`
- `volume_24h`
- `days_listed`
- `beta_30d`
- `taxonomy flags`

注意：

这些字段不是为了“提前做因子”，而是为了避免每个阶段都各算一遍不一致的基础变量。

## 3.5 `delivery_contract`

这组决定：

- 后续如何重建 DataReady
- 当前产物从哪里来
- 运行程序快照存在哪里
- 输入目录和 replay command 是什么

如果没有这层，DataReady 就只是“当前机器上的一份结果”，而不是正式阶段产物。

---

## 4. DataReady 最容易混淆的几个概念

## 4.1 universe membership 不等于 eligibility

这是横截面研究中一个特别常见的误区。

例子：

- 某币属于 Top 50 universe
- 但某天链上字段缺失

那它仍然可能在 `asset_universe_membership` 中为真，  
但在 `eligibility_base_mask` 中该时点为假。

如果把两者混成一个表，后面很难区分：

- 是资产本来不在研究范围
- 还是这个时点基础数据不可研究

## 4.2 shared feature 不等于 factor

例如 `log_market_cap` 是很多中性化与风控要用的共享字段，  
但它不是某个研究线自动拥有的“正式因子”。

DataReady 要做的是把共享底座准备好，而不是在这里提前认定哪些字段就是候选 alpha。

## 4.3 数据对齐不等于 lookahead 验证

DataReady 可以检查：

- 时区是否统一
- 时间戳是否对齐
- 缺失是否可解释

但不要在这里用“某因子的 IC 很高所以怀疑 leakage”来做判断。  
那已经依赖了因子定义，属于后面阶段。

---

## 5. 这一阶段该怎么落地

## 5.1 先从 Mandate 消费边界

DataReady 开始时，不应重新发明问题，而应先逐条消费：

- route
- universe rule
- time split
- bar size
- execution 语义

如果 Mandate 写的是 `1h`、`UTC`、`close_time`，DataReady 就必须沿用，不应自己换成另一个版本。

## 5.2 构建 `asset_universe_membership`

这一步记录的是：

> 每个时点，哪些资产按规则属于研究 universe。

关键不是“最终名单”，而是**时点级关系**。

如果 universe 是静态冻结，那 membership 可能大体不变。  
如果 mandate 允许时点级成员关系记录，也必须按真实规则落盘，而不是只给一份最终静态清单。

## 5.3 构建 `eligibility_base_mask`

这一步的思路是：

> 不问这个因子能不能算，只问这个观测是否具备基础研究资格。

典型 base eligibility 条件可以是：

- 非稳定币
- 上市天数足够
- 基础价格和成交量字段齐全
- 不违反 mandate 的基础排除规则

## 5.4 产出 `cross_section_coverage`

这一步帮助后续阶段理解：

- 每个时点有多少资产可研究
- 哪些资产覆盖度差
- 某些时段是不是几乎没有截面

这类信息非常关键，因为横截面研究不是时间序列研究。  
如果某些时点截面几乎塌了，后面做 IC 或 bucket returns 会非常不稳。

## 5.5 产出 `shared_feature_base`

这一步的重点不是多，而是统一。

应优先放那些：

- 多个下游阶段都会用到
- 口径必须统一
- 自己重复计算容易不一致

的字段。

---

## 6. 输出物应该长什么样

最重要的文件通常有这些：

- `panel_manifest.json`
- `asset_universe_membership.parquet`
- `eligibility_base_mask.parquet`
- `cross_section_coverage.parquet`
- `shared_feature_base/`
- `asset_taxonomy_snapshot.parquet`
- `run_manifest.json`
- `rebuild_csf_data_ready.py`

其中最容易被忽略的是后两项。

### 为什么 `run_manifest.json` 很重要

因为没有它，别人只能看到“结果”，看不到：

- 用了哪个程序
- 哪个 runtime 版本
- 从哪里读入
- 如何重放

这会让 DataReady 失去工程可复现性。

### 为什么要保存 stage-local 程序快照

因为 repo 里后续代码可能变化。  
如果不把当前阶段真实使用的生成程序快照保存下来，过几周后你很可能已经无法确定：

- 当时到底用的是哪一版逻辑
- 结果差异来自数据变了还是程序变了

---

## 7. 本阶段最常见的错误

## 7.1 把 DataReady 写成因子探索

典型错误：

- 一边构数据，一边看哪个因子相关性高
- 用因子效果倒推 base mask

这样做会让数据底座不再中立。

## 7.2 把 eligibility 当成“我要保留的资产”

一旦 eligibility 里混入太多主观条件，后续所有研究都会站在一套已经偏向某类结果的样本之上。

## 7.3 只有 parquet，没有解释

如果只有结果文件，没有：

- field dictionary
- artifact catalog
- run manifest
- 程序快照

那别人很难安全复用这一步。

## 7.4 静默纠正 Mandate

例如：

- “这个 universe 太难做了，我帮它缩一下”
- “这个时间切分不太合理，我先改一下”

这类行为在工程上看像救火，在治理上看就是越权。

---

## 8. 到下一阶段时，应该怎样交接

DataReady 结束后，SignalReady 应该在同一套底座上定义因子合同。

SignalReady 不应再改：

- panel key
- timezone
- eligibility 语义
- taxonomy 版本
- shared feature 的来源口径

如果 SignalReady 发现这些东西不够用，正确动作是回到 DataReady 修正，而不是自己偷偷派生一版。

---

## 9. 最后给一个判断标准

合格的 DataReady，应该满足这样一个条件：

> 后面任何一个因子作者进来，不需要自己重新理解“世界长什么样”，只需要在这套既定的 `date x asset` 面板世界上定义信号。

如果还需要“每个人再搭一版数据底座”，说明 DataReady 还没真正完成。
