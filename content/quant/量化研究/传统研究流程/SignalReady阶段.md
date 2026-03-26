# Signal Ready 阶段规范

---

doc_id: QSP-SR-v1.0

title: Signal Ready 阶段规范 — 信号就绪标准

subtitle: 将研究对象定义为统一、可复现的信号字段合同

date: 2026-03-26

status: v1.0

owner: Quant Research Team

audience:
- Quant Researcher
- Quant Dev
- PM

---

## 目录

1. [阶段概述](#1-阶段概述)
2. [核心任务](#2-核心任务)
3. [Signal Field Contract 规范](#3-signal-field-contract-规范)
4. [参数清单 (Param Manifest)](#4-参数清单-param-manifest)
5. [信号时序生成规范](#5-信号时序生成规范)
6. [冻结原则](#6-冻结原则)
7. [Formal Gate 要求](#7-formal-gate-要求)
8. [常见错误](#8-常见错误)
9. [输出 Artifact](#9-输出-artifact)
10. [与 Train Calibration 的交接](#10-与-train-calibration-的交接)

---

## 1. 阶段概述

### 1.1 阶段定义

Signal Ready 是量化研究流程的第三阶段，位于 Data Ready 之后、Train Calibration 之前。

**核心目标**：将研究对象从"想法"或"数据"转化为**统一、可复现的信号字段合同**。

### 1.2 为什么要独立这个阶段

| 问题 | 如果跳过这个阶段 |
|------|-----------------|
| 信号定义不一致 | Train/Test/Backtest 各自理解不同，结果无法比较 |
| 参数身份模糊 | 无法追溯是哪组参数产生的结果 |
| 时间语义混乱 | 出现前视偏差而不自知 |
| 可复现性缺失 | 换个人或换台机器无法重现相同结果 |

### 1.3 在流程中的位置

```
Mandate → Data Ready → Signal Ready → Train Calibration → Test Evidence → ...
                        ↑
                      当前阶段
```

**前置依赖**：
- Mandate 阶段已冻结研究主问题、Universe、时间切分
- Data Ready 阶段已确认数据基础层可用

**后续影响**：
- Train Calibration 将基于信号字段合同"定尺子"
- Test Evidence 将验证冻结后的信号结构
- Backtest 将复用已定义的信号时序

---

## 2. 核心任务

### 2.1 任务清单

| 任务 | 输出 | 优先级 |
|------|------|--------|
| 定义信号字段合同 | Signal_Field_Contract.md | P0 |
| 生成参数清单 | param_manifest.json | P0 |
| 生成信号时序 | signal_timeseries.pkl | P0 |
| 完成字段字典 | field_dictionary.md | P0 |
| 完成产物目录 | artifact_catalog.md | P1 |
| 通过时间语义检查 | lookahead_audit.log | P0 |

### 2.2 核心原则

1. **Hypothesis before implementation**
   - 先明确定义信号是什么，再写代码实现
   - 禁止"边写边改定义"

2. **Contract over convention**
   - 所有信号定义必须在合同中显式声明
   - 禁止靠"约定俗成"或"大家都知道"

3. **Machine-readable first**
   - 参数清单、时序数据必须是机器可读格式
   - 人类可读文档作为配套说明

---

## 3. Signal Field Contract 规范

### 3.1 什么是信号字段合同

Signal Field Contract 是信号字段的**正式定义规范**，相当于信号的"身份证"。

**类比**：就像 API 接口文档，定义了输入、输出、类型、约束。

### 3.2 字段合同模板

```markdown
# Signal Field Contract: {signal_name}

## 基本信息

| 字段 | 值 |
|------|-----|
| signal_name | 信号名称（英文，snake_case） |
| signal_id | 唯一标识符（如 RSRS_001） |
| version | v1.0 |
| author | 创建者 |
| created_date | YYYY-MM-DD |

## 字段定义

### 主字段

| 字段名 | 数据类型 | 含义 | 单位 | 可空 | 空值语义 |
|--------|---------|------|------|------|----------|
| rsrs_score | float64 | RSRS 标准化得分 | 无 | 否 | N/A |
| rsrs_r2 | float64 | RSRS R² 值 | 无 | 是 | NaN 表示回归失败 |

### 时间字段

| 字段名 | 数据类型 | 含义 | 时区 |
|--------|---------|------|------|
| timestamp | datetime64[ns] | 信号产生时间 | UTC |

### 标识字段

| 字段名 | 数据类型 | 含义 | 取值范围 |
|--------|---------|------|----------|
| symbol | str | 标的代码 | 如 "BTCUSDT" |
| param_id | str | 参数组合标识 | 如 "p1_w20_n10" |

## 时间语义

### 信号时间标签
- **定义**：信号值代表哪个时间点的判断
- **取值**：`{open_time | close_time | decision_time}`
- **本信号**：close_time（使用收盘后数据计算）

### 可用时间
- **定义**：信号何时可被交易系统消费
- **计算**：timestamp + 1 个时间单位
- **示例**：T 日收盘信号 → T+1 日开盘可用

### 前视边界
- **禁止使用的未来信息**：
  - 当日收盘价（如果信号用于盘中决策）
  - 未来成交量、波动率
- **允许使用的历史信息**：
  - 过去 N 期的 OHLCV
  - 过去计算的技术指标

## 参数空间

### 参数列表

| 参数名 | 类型 | 含义 | 默认值 | 搜索范围 |
|--------|------|------|--------|----------|
| window | int | 回归窗口 | 20 | [10, 30] |
| normalize | bool | 是否标准化 | True | [True, False] |

## 计算逻辑

### 算法描述
1. 取过去 window 期的收盘价和最高价
2. 进行线性回归：high = α + β × close
3. 计算 R² 和标准化 residuals
4. rsrs_score = (residuals - mean) / std

### 伪代码
```python
def calculate_rsrs(close_series, high_series, window):
    beta, r2 = rolling_regression(close_series, high_series, window)
    residuals = high_series - (alpha + beta * close_series)
    normalized = (residuals - rolling_mean(residuals, window)) / rolling_std(residuals, window)
    return {
        'rsrs_score': normalized.iloc[-1],
        'rsrs_r2': r2.iloc[-1]
    }
```

## 验证检查

- [ ] 时间语义已明确定义
- [ ] 前视边界已明确标注
- [ ] 参数空间已完全声明
- [ ] 字段类型已锁定
- [ ] 空值语义已说明

## 变更记录

| 版本 | 日期 | 变更内容 | 变更人 |
|------|------|----------|--------|
| v1.0 | YYYY-MM-DD | 初始版本 | XXX |
```

### 3.3 字段合同示例

**示例：动量信号**

| 字段名 | 类型 | 含义 | 时间语义 |
|--------|------|------|----------|
| momentum_20d | float64 | 20 日收益率 | close_time (T 日收盘) |
| mom_rank | float64 | 横截面排名 | close_time (T 日收盘) |
| mom_decile | int8 | 十分位分组 | close_time (T 日收盘) |

---

## 4. 参数清单 (Param Manifest)

### 4.1 什么是参数清单

Param Manifest 是参数组合的**唯一标识和完整记录**。

**核心作用**：
- 不允许靠文件名猜
- 不允许靠人脑记
- 每组参数有唯一 ID

### 4.2 参数清单格式

#### JSON 格式（推荐）

```json
{
  "manifest_id": "RSRS_20260326_P1",
  "signal_name": "rsrs_signal",
  "signal_version": "v1.0",
  "created_date": "2026-03-26T10:00:00Z",
  "created_by": "researcher_a",
  "lineage_id": "lineage_rsrs_001",

  "param_space": {
    "window": {"type": "int", "range": [10, 30], "default": 20},
    "normalize": {"type": "bool", "options": [true, false], "default": true}
  },

  "param_combinations": [
    {
      "param_id": "p1_w20_n_true",
      "window": 20,
      "normalize": true,
      "source": "grid_search",
      "status": "active"
    },
    {
      "param_id": "p2_w18_n_true",
      "window": 18,
      "normalize": true,
      "source": "grid_search",
      "status": "candidate"
    }
  ],

  "metadata": {
    "search_strategy": "grid_search",
    "total_combinations": 12,
    "selected_count": 2,
    "selection_criteria": "train_sharpe > 1.5"
  }
}
```

#### CSV 格式（轻量级）

```csv
param_id,window,normalize,source,status,train_sharpe
p1_w20_n_true,20,True,grid_search,active,1.73
p2_w18_n_true,18,True,grid_search,candidate,1.65
p3_w22_n_false,22,False,grid_search,rejected,1.12
```

### 4.3 参数 ID 生成规则

**命名规范**：`p{序号}_{参数摘要}`

**示例**：
- `p1_w20_n_true` → 参数组1，窗口=20，标准化=true
- `p2_mom10d_voladj` → 参数组2，动量期=10日，波动率调整
- `p3_w30_nFalse_ma5` → 参数组3，窗口=30，标准化=False，MA平滑=5

**禁止**：
- ❌ `best_params` → 无法识别是哪组
- ❌ `params_v2` → 版本不明确
- ❌ 依赖文件名 → 必须在 manifest 中显式声明

### 4.4 参数清单与字段合同的关系

```
Signal Field Contract
    ↓ 定义参数空间（参数名、类型、范围）
Param Manifest
    ↓ 列举具体参数组合及其唯一 ID
信号时序数据
    ↓ 每条记录携带 param_id
Train / Test / Backtest
    ↓ 通过 param_id 追溯和复现
```

---

## 5. 信号时序生成规范

### 5.1 时序数据结构

**最小字段集**：

| 字段名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| timestamp | datetime64 | ✅ | 统一时间戳 |
| symbol | str | ✅ | 标的标识 |
| param_id | str | ✅ | 参数组合 ID |
| {signal_value} | float64 | ✅ | 信号值 |
| {auxiliary_fields} | mixed | ❌ | 辅助字段（如 R²） |

### 5.2 时序数据格式

**Parquet 格式**（推荐）：

```python
# 示例：pandas DataFrame → Parquet
df.to_parquet("signal_rsrs_p1_w20_n_true.parquet", engine="pyarrow")

# 文件命名规范
signal_{signal_name}_{param_id}.parquet
```

**HDF5 格式**（兼容）：

```python
df.to_hdf("signal_rsrs_p1_w20_n_true.h5", key="data")
```

### 5.3 时序数据质量检查

**Formal Gate 要求**：

| 检查项 | 标准 | 检查方法 |
|--------|------|----------|
| 时间单调性 | timestamp 严格递增 | `df.timestamp.is_monotonic_increasing` |
| 符号覆盖 | 所有 symbol 都有数据 | `df.symbol.nunique() == universe_size` |
| 缺失率 | 单字段缺失率 < 5% | `df.isnull().mean() < 0.05` |
| 前视检查 | signal[t] 只用 t 及以前数据 | 人工审查 + 单元测试 |
| 类型一致性 | 字段类型与合同一致 | `df.dtypes == contract_dtypes` |

### 5.4 多参数组合管理

**目录结构**：

```
signals/
├── rsrs_signal/
│   ├── p1_w20_n_true.parquet
│   ├── p2_w18_n_true.parquet
│   ├── p3_w22_n_false.parquet
│   └── manifest.json
└── momentum_signal/
    ├── p1_10d.parquet
    ├── p2_20d.parquet
    └── manifest.json
```

---

## 6. 冻结原则

### 6.1 必须冻结的内容

在 Signal Ready 阶段结束时，以下内容**必须冻结**：

| 冻结项 | 内容 | 后续能否修改 |
|--------|------|--------------|
| 信号字段定义 | 字段名、类型、含义 | ❌ Train/Test/Backtest 不得修改 |
| 时间语义 | 信号时间标签、可用时间 | ❌ 修改意味着新信号 |
| 参数空间 | 参数名、类型、搜索范围 | ⚠️ 修改需创建 Child Lineage |
| 参数 ID 规则 | 命名规范、生成规则 | ❌ 后续必须沿用 |
| 时序数据格式 | 文件格式、字段顺序 | ❌ 确保兼容性 |

### 6.2 冻结的时机

1. **Signal Field Contract 签署后** → 字段定义、时间语义冻结
2. **Param Manifest 生成后** → 参数空间、ID 规则冻结
3. **时序数据 QC 通过后** → 数据格式、质量标准冻结

### 6.3 冻结后的变更流程

```
发现需要修改
    ↓
评估变更影响
    ↓
├─ 小改动（如字段文档） → 允许，记录在变更日志
├─ 中等改动（如参数范围） → 创建 Child Lineage
└─ 大改动（如信号机制） → 重新从 Mandate 开始
```

---

## 7. Formal Gate 要求

### 7.1 Signal Ready Formal Gate

**通过标准**（必须全部满足）：

| Gate 项目 | 标准 | 验证方法 |
|-----------|------|----------|
| 字段合同完整性 | 所有字段按模板填写 | 人工审查 |
| 参数清单有效性 | 每个 param_id 唯一且可解析 | 自动检查 |
| 时序数据质量 | 通过所有 QC 检查 | 自动验证 |
| 前视边界清零 | 无未声明的前视风险 | 代码审计 + 单元测试 |
| 配套文档完整 | field_dictionary + artifact_catalog | 人工审查 |

### 7.2 Audit Gate（审计门禁）

**补充检查项**：

| 检查项 | 说明 | 处理方式 |
|--------|------|----------|
| 参数覆盖合理性 | 搜索范围是否覆盖有效区间 | 记录，不阻断 |
| 计算效率 | 信号生成耗时是否可接受 | 记录，不阻断 |
| 代码可读性 | 实现代码是否有注释 | 记录，不阻断 |
| 文档质量 | 文档是否易于理解 | 记录，不阻断 |

### 7.3 Gate Decision 模板

```markdown
# Signal Ready Gate Decision

## 基本信息

| 字段 | 值 |
|------|-----|
| stage | Signal Ready |
| lineage_id | {lineage_id} |
| decision_date | YYYY-MM-DD |
| reviewer | {reviewer_name} |

## 决策结果

**状态**：PASS / CONDITIONAL PASS / RETRY / NO-GO

## Formal Gate 检查

- [ ] 字段合同完整性
- [ ] 参数清单有效性
- [ ] 时序数据质量
- [ ] 前视边界清零
- [ ] 配套文档完整

## 冻结内容

本阶段冻结以下内容，后续阶段不得修改：
1. 信号字段定义（见 Signal_Field_Contract.md）
2. 参数空间和 ID 规则（见 param_manifest.json）
3. 时间语义（可用时间 = timestamp + 1）

## 下一步

- 进入 Train Calibration 阶段
- 交接文件：{frozen_spec_path}

## 备注

{conditional_pass 的条件或 retry 的原因}
```

---

## 8. 常见错误

### 8.1 字段定义错误

| 错误 | 后果 | 正确做法 |
|------|------|----------|
| 字段名含中文 | 跨平台兼容问题 | 使用英文 snake_case |
| 时间语义未声明 | 前视风险 | 明确是 open_time 还是 close_time |
| 空值语义未说明 | 数据处理错误 | 声明 NaN 的具体含义 |

### 8.2 参数管理错误

| 错误 | 后果 | 正确做法 |
|------|------|----------|
| 靠文件名区分参数 | 无法追溯 | 使用 param_manifest.json |
| 参数 ID 无规律 | 无法快速识别 | 使用 `p{序号}_{摘要}` 格式 |
| 未记录被拒绝参数 | 幸存者偏差 | 在 manifest 中保留所有参数 |

### 8.3 时序数据错误

| 错误 | 后果 | 正确做法 |
|------|------|----------|
| 缺失静默填充 | 虚假信号 | 显式保留 NaN |
| 时间戳不统一 | 错位信号 | 强制统一时区 |
| 字段类型不一致 | 后续处理失败 | 在合同中锁定类型 |

### 8.4 前视风险

| 危险操作 | 为什么危险 | 替代方案 |
|----------|-----------|----------|
| 用当日收盘价做盘中决策 | 前视偏差 | 用前一收盘或实时价 |
| 用未来波动率标准化 | 前视偏差 | 用历史滚动窗口 |
| 用未来统计量 | 前视偏差 | 用 expanding window |

---

## 9. 输出 Artifact

### 9.1 必需产出物

| 产物 | 格式 | 用途 | 消费者 |
|------|------|------|--------|
| Signal_Field_Contract.md | Markdown | 信号定义规范 | 所有阶段 |
| param_manifest.json | JSON | 参数清单 | Train/Test/Backtest |
| signal_timeseries.parquet | Parquet | 信号时序数据 | Train Calibration |
| field_dictionary.md | Markdown | 字段说明文档 | 人类读者 |
| artifact_catalog.md | Markdown | 产物目录 | 项目管理 |

### 9.2 可选产出物

| 产物 | 格式 | 用途 |
|------|------|------|
| signal_generation.py | Python | 信号生成代码 |
| signal_qc_report.html | HTML | 质量检查报告 |
| lookahead_audit.log | 文本 | 前视审计日志 |

### 9.3 Artifact Catalog 示例

```markdown
# Signal Ready Artifact Catalog

## 产物清单

| 文件名 | 用途 | 粒度 | 主键 | 消费者 | 机器可读 |
|--------|------|------|------|--------|----------|
| Signal_Field_Contract.md | 信号定义 | 信号级 | signal_id | 所有阶段 | ❌ |
| param_manifest.json | 参数清单 | 参数组级 | param_id | Train/Test | ✅ |
| signal_timeseries.parquet | 时序数据 | 行级 | (timestamp, symbol, param_id) | Train Calibration | ✅ |
| field_dictionary.md | 字段说明 | 字段级 | field_name | 人类读者 | ❌ |

## 字段说明映射

| 文件 | 字段名 | 说明 |
|------|--------|------|
| signal_timeseries.parquet | rsrs_score | 见 Signal_Field_Contract.md → 主字段 |
| signal_timeseries.parquet | rsrs_r2 | 见 Signal_Field_Contract.md → 主字段 |
| param_manifest.json | param_id | 见本文件 → 参数 ID 生成规则 |

## 版本信息

- 生成时间：YYYY-MM-DD HH:MM:SS UTC
- 生成工具：signal_generator v1.2.0
- 数据源：data_ready_layer v1.0
```

---

## 10. 与 Train Calibration 的交接

### 10.1 交接内容

**从 Signal Ready 交接给 Train Calibration**：

| 交接物 | 内容 | 用途 |
|--------|------|------|
| Frozen Spec | 冻结的信号定义 + 参数清单 | Train 照单执行 |
| 信号时序数据 | 可直接用于训练的数据 | Train 的输入 |
| 时间语义说明 | 信号的可用时间 | Train 的对齐基准 |

### 10.2 Frozen Spec 格式

```yaml
# frozen_spec_signal_rsrs_v1.yaml

signal_metadata:
  signal_name: rsrs_signal
  signal_id: RSRS_001
  version: v1.0
  frozen_at: "2026-03-26T10:00:00Z"

frozen_fields:
  - field_name: rsrs_score
    data_type: float64
    time_semantic: close_time
    available_at: "timestamp + 1d"
  - field_name: rsrs_r2
    data_type: float64
    nullable: true
    na_semantic: "regression failed"

frozen_params:
  default_param_id: p1_w20_n_true
  param_space:
    window: {type: int, frozen_range: [10, 30]}
    normalize: {type: bool, frozen_options: [true, false]}

交接要求:
  - Train 不得修改信号定义
  - Train 不得重估参数范围
  - Train 必须使用 frozen_spec 中的时序数据
```

### 10.3 交接检查清单

**Signal Ready 提交前**：

- [ ] 所有 Formal Gate 已通过
- [ ] Frozen Spec 已生成
- [ ] 时序数据已通过 QC
- [ ] 配套文档已更新

**Train Calibration 接收时**：

- [ ] Frozen Spec 已审阅
- [ ] 时序数据可加载
- [ ] 字段类型匹配
- [ ] 时间语义已理解

### 10.4 不允许的交接后修改

| 项目 | Signal Ready 冻结后 | Train Calibration 能否改 |
|------|---------------------|------------------------|
| 信号字段定义 | ❌ | ❌ 绝对禁止 |
| 参数空间 | ❌ | ❌ 只能选，不能扩 |
| 时间语义 | ❌ | ❌ 改了就不是同一信号 |
| 参数选择 | ✅ 初步选择 | ✅ 可以基于 Train 结果选择 |
| 时序数据 | ✅ 数据层 | ❌ 不能重新生成 |

---

## 附录

### A. 快速检查清单

**完成 Signal Ready 前，确认以下事项**：

- [ ] Signal Field Contract 已完整填写
- [ ] 参数清单已生成，所有 param_id 唯一
- [ ] 信号时序数据已通过 QC 检查
- [ ] 前视边界已明确，无未声明风险
- [ ] field_dictionary.md 已完成
- [ ] artifact_catalog.md 已完成
- [ ] Frozen Spec 已生成
- [ ] Formal Gate 已通过

### B. 相关文档

- [Data Ready 阶段规范](./DataReady阶段.md)
- [Train Calibration 阶段规范](./TrainCalibration阶段.md)
- [专业术语表](./传统量化流程涉及的专业术语.md)
- [标准化流程总结](./标准化流程总结.md)

### C. 变更记录

| 版本 | 日期 | 变更内容 | 变更人 |
|------|------|----------|--------|
| v1.0 | 2026-03-26 | 初始版本 | Quant Research Team |

---

**文档状态**：v1.0 正式版

**最后更新**：2026-03-26

**维护者**：Quant Research Team
