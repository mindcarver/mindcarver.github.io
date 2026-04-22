
# 阶段常见问题

## csf_data_ready

【A 面板合同类问题】
- 只有单资产时序表，没有显式 panel
- `date_key` / `asset_key` 只是默认存在，没写进 manifest
- `coverage_rule` 模糊，例如只写“按有效数据覆盖”
- panel frequency 没明说，日频/小时频靠文件名猜

风险： 后面 signal/train 会对“同一天同一资产的一行”产生不同理解。


【B universe membership 类问题】
- membership 只存静态资产列表，不按日期记录
- 某些资产何时加入/退出 universe 无法重建
- 研究期内资产存续变化没记录
- mandate 冻结 universe 后，author 在 data-ready 静默调整

风险： IC、coverage、backtest 都会失去同一研究母体。

【C Eligibility 类问题】
- eligibility 和 signal 缺失混用
- eligibility 规则写在特定因子脚本里
- 基础研究资格与交易过滤混在一起
- false / NaN / missing 没有区分语义

风险：后面 reviewer 不能判断样本变化来自 data-ready 还是 signal-ready。


经验沉淀：
eligibility 必须强制回答这几个问题：

- 它是 base research eligibility 吗
- 它是否与具体 signal 无关
- 它的 false / missing 各自代表什么
- 它是否会改变 lineage


【D coverage 类问题】
- 只给一个全局覆盖率数字
- 不按日期审计 coverage 波动
- 覆盖率突然掉下来没有解释
- 没法对应到 membership 和 eligibility

风险：研究结果可能只在高覆盖窗口成立，但被 overall summary 掩盖。

经验沉淀：
coverage 不应只是统计摘要，而应是：

> **panel completeness evidence over date x asset space**

特别要沉淀：

- coverage by date
- coverage by asset bucket
- pre-eligibility vs post-eligibility coverage
- anomaly windows


## E. Shared feature base 类问题

### 常见问题

- shared features 只是“未来可能会用到”的预想字段
- 没有时间语义
- 没有缺失语义
- 混入具体 signal 定义
- 与 taxonomy / eligibility 边界不清

### 风险

shared feature base 会变成一个杂物间，后面任何字段都能往里塞。

### 经验沉淀

shared feature base 要求回答：

- 为什么这是“shared”而不是某个 signal 私有字段
- 这个字段的 time semantics 是什么
- 缺失值语义是什么
- 是否允许下游再加工
- 是否依赖 taxonomy / eligibility


## F. Delivery / provenance 类问题

### 常见问题

- `run_manifest.json` 只是形式存在
- replay_command 无法定位真实构建脚本
- 脚本保存在别处，不在 stage-local
- artifact catalog / field dictionary 与实际文件不同步

### 风险

过几周后就无法判断当前产物是否真的对应当前 contract。

### 经验沉淀

delivery contract 里最关键的不是“文件打包好”，而是：

> **author claim 与 machine-readable provenance 是否闭环。**