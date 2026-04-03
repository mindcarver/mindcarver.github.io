# DataReady 阶段 - 横截面面板底座冻结

## 1. 阶段定位

DataReady 的职责是把 Mandate 中冻结的研究边界，落成真正可复用的横截面面板底座。

它回答的问题是：

> 我们后续所有因子都建立在什么 `date x asset` 面板、准入语义和共享派生层之上？

它不回答“某个因子是否有效”，也不回答“策略能不能赚钱”。

推荐主链：

```text
Mandate
  -> DataReady
  -> SignalReady
  -> TrainFreeze
  -> TestEvidence
  -> BacktestReady
  -> HoldoutValidation
```

---

## 2. 本阶段只冻结什么

DataReady 只冻结横截面研究的底座合同：

- `panel_contract`
- `taxonomy_contract`
- `eligibility_contract`
- `shared_feature_base`
- `delivery_contract`

对应的核心问题分别是：

- 面板主键是什么
- 资产分类和分组底座是什么
- 哪些观测是“基础可研究”的
- 哪些共享派生字段后续可复用
- 这些产物如何重建和交付

---

## 3. 本阶段不该做什么

禁止事项：

- 在 DataReady 阶段做具体因子定义
- 在 DataReady 阶段用因子 IC 检查 lookahead
- 在 DataReady 阶段挑因子、去重、筛优
- 把 eligibility mask 写成某个因子的专用过滤器
- 静默改动 Mandate 冻结的 universe、时间边界或 route

如果一个检查必须依赖 `factor_values` 才成立，那它通常已经不是 DataReady，而是 SignalReady 之后的事情。

---

## 4. 必需输入

建议至少有以下输入：

- `mandate.md`
- `research_scope.md`
- `research_route.yaml`
- `time_split.json`
- `stage_completion_certificate.yaml`

如果 `research_route` 不是 `cross_sectional_factor`，这篇规范就不适用。

---

## 5. 必需输出

建议至少产出以下 artifact：

- `panel_manifest.json`
- `asset_universe_membership.parquet`
- `eligibility_base_mask.parquet`
- `cross_section_coverage.parquet`
- `shared_feature_base/`
- `asset_taxonomy_snapshot.parquet`（若后续允许 `group_neutral`）
- `csf_data_contract.md`
- `run_manifest.json`
- `rebuild_csf_data_ready.py` 或等价程序快照
- `artifact_catalog.md`
- `field_dictionary.md`

建议最少机器可读字段：

```yaml
panel_manifest:
  date_key: timestamp
  asset_key: symbol
  panel_frequency: 1h
  timezone: UTC
  coverage_rule: "cross-sectional coverage after eligibility_base_mask"

eligibility_contract:
  base_only: true
  contains_factor_logic: false

shared_feature_base:
  includes:
    - market_cap
    - volume_24h
    - listing_days
  taxonomy_version: v1
```

---

## 6. Formal Gate

### FG-1: 面板主键明确

- `date_key` 明确
- `asset_key` 明确
- `panel_frequency` 明确
- `coverage_rule` 明确

### FG-2: Universe 成员关系可追溯

- 每个时点的成员关系可回放
- 没有把“最终留下的资产”误写成“历史一直存在的资产”

### FG-3: Eligibility 与因子逻辑分离

- `eligibility_base_mask` 只记录基础可研究掩码
- 不混入某个因子的特定条件

### FG-4: 共享派生层可复用

- `shared_feature_base/` 的字段来源明确
- 若后续允许 `group_neutral`，taxonomy 必须版本化

### FG-5: 可重建

- 有 `run_manifest.json`
- 有 stage-local 程序快照
- 能说明输入根目录与 replay 命令

---

## 7. Audit Gate

可以记录但不应越界的审计项：

- 覆盖率
- 缺失模式
- 时间对齐质量
- 坏价与极端值分布

这些都属于数据底座质量。  
但“某个因子的 IC 是否异常高”不属于这里。

---

## 8. 常见反模式

### 反模式 1：把 DataReady 写成因子研究

错误写法：

```python
ic = spearmanr(factor_values, future_return)
if ic > 0.3:
    warn("lookahead")
```

这依赖具体因子，已经越界。

### 反模式 2：把 eligibility 做成因子过滤器

错误写法：

- “只有动量值可算时才算 eligible”
- “RSI 小于 30 的资产进入 base mask”

base mask 只能表达基础可研究性，不能偷带因子定义。

### 反模式 3：静默重写 Universe

错误写法：

- 因为历史覆盖率差，就直接缩窄资产池
- 因为某阶段效果差，就改 universe 口径

任何这类动作都应回到 Mandate 或开 child lineage。

### 反模式 4：只有文档，没有可重建程序

如果只有说明文档，没有实际生成面板的程序快照和运行记录，DataReady 不算完成。

---

## 9. 与下一阶段的交接

SignalReady 只能消费 DataReady 已冻结的底座，不能静默修改：

- 面板主键
- eligibility 语义
- taxonomy 版本
- 时间边界

下一阶段主要依赖：

- `panel_manifest.json`
- `asset_universe_membership.parquet`
- `eligibility_base_mask.parquet`
- `cross_section_coverage.parquet`
- `shared_feature_base/`

---

## 10. 一句话标准

DataReady 完成后，后续任何因子作者都应当基于同一份 `date x asset` 面板底座开工，而不是各自偷偷重建一版数据世界。
