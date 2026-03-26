# Data Ready 阶段详细文档

**文档ID**: QSP-DR-v1.0
**阶段编号**: 01
**阶段名称**: Data Ready (数据就绪)
**日期**: 2026-03-26
**状态**: v1.0
**负责角色**: Data Scientist + Quant Researcher

---

## 目录

1. [阶段定义与核心目的](#1-阶段定义与核心目的)
2. [为什么需要 Data Ready](#2-为什么需要-data-ready)
3. [数据需求分析](#3-数据需求分析)
4. [数据获取与验证](#4-数据获取与验证)
5. [时间对齐处理](#5-时间对齐处理)
6. [缺失值处理](#6-缺失值处理)
7. [质量控制 (QC)](#7-质量控制-qc)
8. [覆盖审计](#8-覆盖审计)
9. [数据分层设计](#9-数据分层设计)
10. [Formal Gate 要求](#10-formal-gate-要求)
11. [常见错误与防范](#11-常见错误与防范)
12. [输出 Artifact](#12-输出-artifact)
13. [与 Signal Ready 的交接](#13-与-signal-ready-的交接)
14. [数据质量报告模板](#14-数据质量报告模板)

---

## 1. 阶段定义与核心目的

### 1.1 阶段定义

**Data Ready (数据就绪)** 是确认原始数据可以被转换为可研究的数据基础层的阶段。

**核心特征**：
- **数据验证层**: 确保数据可用、可靠、可理解
- **问题隔离**: 防止把数据问题误认为是 Alpha 信号
- **基础建设**: 为后续阶段提供稳定的数据基础

### 1.2 核心目的

**为什么需要 Data Ready**：

| 目的 | 说明 | 价值 |
|------|------|------|
| **防止数据污染** | 识别并处理数据质量问题 | 避免虚假信号 |
| **确保可复现** | 标准化数据处理流程 | 结果可重复验证 |
| **提高效率** | 提前发现数据问题 | 减少后期返工 |
| **建立信任** | 系统化的数据质量报告 | 增强研究结果可信度 |
| **知识沉淀** | 记录数据特性和处理方法 | 团队知识积累 |

### 1.3 与 Mandate 的关系

**Mandate → Data Ready 交接**：
```yaml
mandate_to_data_ready:

  inputs:
    - "Universe 定义 → 数据范围"
    - "时间窗设计 → 数据期间"
    - "信号机制假设 → 数据类型需求"
    - "成功标准 → 数据质量标准"

  outputs:
    - "可用数据集"
    - "数据质量报告"
    - "数据处理规范"
    - "字段字典"

  constraints:
    - "不能修改 Mandate 冻结的内容"
    - "只能按 Mandate 定义的范围准备数据"
```

---

## 2. 为什么需要 Data Ready

### 2.1 数据问题的隐蔽性

**常见数据问题**：
```yaml
common_data_issues:

  time_alignment:
    issue: "不同标的时间戳不对齐"
    consequence: "产生虚假的领先-滞后关系"
    example: >
      BTC 使用 UTC 时间，ETH 使用本地时间，
      导致看似 ETH 领先 BTC 的虚假信号

  missing_data:
    issue: "数据缺失"
    consequence: "样本偏差、计算错误"
    example: >
      新币交易初期数据缺失，
      导致只用老数据计算，产生选择偏差

  stale_data:
    issue: "数据不更新"
    consequence: "使用过期信息做决策"
    example: >
      价格长时间不更新（流动性差），
      但仍然用最新价格计算信号

  outliers:
    issue: "异常值"
    consequence: "扭曲统计结果"
    example: >
      闪电崩盘或技术故障导致极端价格，
      影响波动率和相关性计算

  survivorship_bias:
    issue: "只保留当前存在的标的"
    consequence: "高估历史表现"
    example: >
      只用当前存在的币做回测，
      忽略已经退市或下架的币
```

### 2.2 Data Ready 的价值

**正例**：
```yaml
before_data_ready:
  problem: "信号在 IS 上显著，OOS 上失效"
  investigation: "花了2周排查信号逻辑"
  root_cause: "测试数据有大量缺失，样本量不足"

after_data_ready:
  qc_report: >
    Data Ready 阶段发现测试期间
    数据缺失率 35%，提前扩充时间窗
  outcome: "节省2周排查时间，结果更可靠"
```

---

## 3. 数据需求分析

### 3.1 基于 Mandate 的需求分析

**数据范围**：
```python
def derive_data_requirements(mandate_spec):
    """
    从 Mandate 推导数据需求
    """
    requirements = {
        'universe': {
            'symbols': mandate_spec['universe']['initial_symbols'],
            'inclusion_criteria': mandate_spec['universe']['inclusion_criteria'],
            'exclusion_criteria': mandate_spec['universe']['exclusion_criteria']
        },
        'time_range': {
            'start': min([
                mandate_spec['time_window']['train']['start'],
                mandate_spec['time_window']['test']['start'],
                mandate_spec['time_window']['backtest']['start'],
                mandate_spec['time_window']['holdout']['start']
            ]),
            'end': max([
                mandate_spec['time_window']['train']['end'],
                mandate_spec['time_window']['test']['end'],
                mandate_spec['time_window']['backtest']['end'],
                mandate_spec['time_window']['holdout']['end']
            ])
        },
        'data_types': infer_data_types(mandate_spec['signal_mechanism']),
        'quality_standards': mandate_spec['success_criteria']['data_quality']
    }

    return requirements
```

### 3.2 数据类型需求

**根据信号机制确定数据类型**：
```yaml
data_type_requirements:

  price_data:
    required_for:
      - "动量信号"
      - "均值回归"
      - "波动率建模"
    fields:
      - "open"
      - "high"
      - "low"
      - "close"
      - "volume"

  fundamental_data:
    required_for:
      - "价值因子"
      - "质量因子"
    fields:
      - "market_cap"
      - "pe_ratio"
      - "revenue"
      - "earnings"

  alternative_data:
    required_for:
      - "情绪信号"
      - "链上指标"
    fields:
      - "social_mentions"
      - "on_chain_volume"
      - "active_addresses"
```

### 3.3 数据频率需求

**根据研究问题确定频率**：
```yaml
frequency_requirements:

  intraday:
    frequency: "1分钟 - 1小时"
    use_cases:
      - "高频交易"
      - "日内动量"
    considerations:
      - "数据量大"
      - "计算成本高"
      - "微结构效应"

  daily:
    frequency: "1天"
    use_cases:
      - "短期动量"
      - "技术指标"
    considerations:
      - "平衡数据量和信噪比"

  weekly:
    frequency: "1周"
    use_cases:
      - "中期动量"
      - "均值回归"
    considerations:
      - "更稳定的信号"

  monthly:
    frequency: "1月"
    use_cases:
      - "长期趋势"
      - "基本面因子"
    considerations:
      - "数据量少"
      - "滞后性强"
```

---

## 4. 数据获取与验证

### 4.1 数据源评估

**数据源选择标准**：
```yaml
data_source_evaluation:

  reliability:
    criteria:
      - "数据提供商声誉"
      - "历史准确性记录"
      - "数据验证机制"
    weight: "高"

  completeness:
    criteria:
      - "历史数据覆盖"
      - "字段完整性"
      - "缺失数据处理"
    weight: "高"

  timeliness:
    criteria:
      - "数据延迟"
      - "更新频率"
      - "实时性要求"
    weight: "中"

  cost:
    criteria:
      - "数据订阅费用"
      - "存储成本"
      - "计算成本"
    weight: "中"

  ease_of_use:
    criteria:
      - "API 质量"
      - "文档完整性"
      - "技术支持"
    weight: "低"
```

### 4.2 数据获取检查清单

**获取前检查**：
```yaml
pre_acquisition_checks:

  coverage_check:
    - [ ] 时间范围覆盖完整需求
    - [ ] Universe 所有标的都有数据
    - [ ] 所需字段都可用

  quality_check:
    - [ ] 数据准确性可验证
    - [ ] 缺失数据可识别
    - [ ] 异常值可检测

  technical_check:
    - [ ] API 访问可用
    - [ ] 数据格式可解析
    - [ ] 存储方案可行

  legal_check:
    - [ ] 使用权限明确
    - [ ] 数据许可合规
    - [ ] 隐私问题考虑
```

**获取后验证**：
```python
def validate_acquired_data(raw_data, requirements):
    """
    验证获取的数据是否符合需求
    """
    validation_results = {}

    # 覆盖范围验证
    validation_results['time_coverage'] = {
        'required_start': requirements['time_range']['start'],
        'required_end': requirements['time_range']['end'],
        'actual_start': raw_data['timestamp'].min(),
        'actual_end': raw_data['timestamp'].max(),
        'coverage_ratio': calculate_coverage_ratio(
            raw_data, requirements['time_range']
        )
    }

    # Universe 验证
    required_symbols = set(requirements['universe']['symbols'])
    actual_symbols = set(raw_data['symbol'].unique())
    validation_results['universe_coverage'] = {
        'required_count': len(required_symbols),
        'actual_count': len(actual_symbols),
        'missing_symbols': list(required_symbols - actual_symbols),
        'extra_symbols': list(actual_symbols - required_symbols)
    }

    # 字段验证
    required_fields = requirements['data_types']['required_fields']
    actual_fields = raw_data.columns.tolist()
    validation_results['field_coverage'] = {
        'required_fields': required_fields,
        'missing_fields': set(required_fields) - set(actual_fields),
        'all_present': set(required_fields).issubset(set(actual_fields))
    }

    return validation_results
```

---

## 5. 时间对齐处理

### 5.1 时间对齐的重要性

**为什么需要时间对齐**：
```yaml
time_alignment_importance:

  problem_scenarios:
    timezone_mismatch:
      description: "不同市场使用不同时区"
      example: "美股 EST vs 加密货币 UTC"
      consequence: "错误的时间顺序"

    exchange_asynchrony:
      description: "不同交易所交易时间不同"
      example: "亚洲盘 vs 美洲盘"
      consequence: "流动性时间不一致"

    daylight_saving:
      description: "夏令时切换"
      example: "美东时间夏令时"
      consequence: "时间偏移1小时"

  consequences:
    - "虚假的领先-滞后关系"
    - "错误的收益计算"
    - "扭曲的相关性分析"
```

### 5.2 时间对齐方法

**标准化时间戳**：
```python
def standardize_timestamps(df, target_timezone='UTC'):
    """
    标准化时间戳到统一时区
    """
    # 转换到目标时区
    if df['timestamp'].dt.tz is not None:
        df['timestamp_utc'] = df['timestamp'].dt.tz_convert(target_timezone)
    else:
        # 假设原始数据是 UTC
        df['timestamp_utc'] = df['timestamp'].dt.tz_localize(target_timezone)

    return df

def align_to_frequency(df, frequency='1H'):
    """
    对齐到指定频率
    """
    # 向下对齐到整点
    df['timestamp_aligned'] = df['timestamp_utc'].dt.floor(frequency)

    return df
```

**缺失时间点处理**：
```python
def handle_missing_timestamps(df, frequency='1H'):
    """
    处理缺失的时间点
    """
    # 创建完整的时间范围
    full_range = pd.date_range(
        start=df['timestamp_aligned'].min(),
        end=df['timestamp_aligned'].max(),
        freq=frequency
    )

    # 对每个标的创建完整索引
    aligned_dfs = []
    for symbol in df['symbol'].unique():
        symbol_data = df[df['symbol'] == symbol].set_index('timestamp_aligned')

        # 重新索引到完整时间范围
        symbol_data = symbol_data.reindex(full_range)

        # 标记缺失
        symbol_data['_is_missing'] = symbol_data['close'].isna()

        aligned_dfs.append(symbol_data.reset_index().assign(symbol=symbol))

    return pd.concat(aligned_dfs, ignore_index=True)
```

### 5.3 时间对齐验证

**验证检查**：
```python
def validate_time_alignment(df):
    """
    验证时间对齐质量
    """
    validation_results = {}

    # 检查时间戳一致性
    validation_results['timezone_consistent'] = df['timestamp_utc'].dt.tz.zone == 'UTC'

    # 检查频率一致性
    time_diffs = df['timestamp_aligned'].diff().dropna()
    expected_diff = pd.Timedelta('1H')
    validation_results['frequency_consistent'] = (time_diffs == expected_diff).all()

    # 检查缺失率
    validation_results['missing_rate'] = df['_is_missing'].mean()

    return validation_results
```

---

## 6. 缺失值处理

### 6.1 缺失值类型

**缺失值分类**：
```yaml
missing_value_types:

  completely_missing:
    description: "标的完全不存在"
    example: "新币上市前"
    handling: "从 Universe 中排除该时段"

  partially_missing:
    description: "部分字段缺失"
    example: "只有价格没有成交量"
    handling: "明确标记，不使用该字段"

  temporarily_missing:
    description: "暂时性缺失"
    example: "交易暂停、网络中断"
    handling: "保留 NaN，不填充"

  structurally_missing:
    description: "结构性缺失"
    example: "股票没有分红"
    handling: "用特殊值标记"
```

### 6.2 缺失值处理原则

**核心原则**：
```yaml
missing_value_principles:

  transparency:
    principle: "缺失必须显式标记"
    implementation: "使用 _is_missing 字段"
    forbidden: "静默 forward-fill"

  traceability:
    principle: "可追溯到原始状态"
    implementation: "保留原始数据列"
    example: "保留原始 close，添加 close_filled"

  conservatism:
    principle: "宁可缺也不要假"
    implementation: "不确定时保留 NaN"
    rationale: "避免虚假信号"

  documentation:
    principle: "所有处理必须记录"
    implementation: "数据处理日志"
    requirement: "可复现"
```

### 6.3 缺失值处理方法

**处理方法**：
```python
def handle_missing_values(df, handling_config):
    """
    处理缺失值
    """
    df_processed = df.copy()

    # 1. 标记缺失
    for field in ['open', 'high', 'low', 'close', 'volume']:
        if field in df_processed.columns:
            df_processed[f'{field}_is_missing'] = df_processed[field].isna()

    # 2. 完全缺失的时段
    # 标记为不可用，不填充
    df_processed['_period_available'] = ~(
        df_processed['close'].isna() &
        df_processed['volume'].isna()
    )

    # 3. 价格缺失但成交量存在 (异常情况)
    # 保留 NaN，用于后续 QC
    df_processed['_price_missing_with_volume'] = (
        df_processed['close'].isna() &
        df_processed['volume'].notna()
    )

    # 4. 成交量缺失但价格存在 (可能合理)
    # 标记但保留
    df_processed['_volume_missing_with_price'] = (
        df_processed['volume'].isna() &
        df_processed['close'].notna()
    )

    return df_processed
```

### 6.4 缺失值容忍度

**设定缺失值容忍度**：
```yaml
missing_value_tolerance:

  symbol_level:
    max_missing_rate: 0.20  # 20%
    action: "超过阈值从 Universe 中排除"

  field_level:
    price_fields:
      max_missing_rate: 0.05  # 5%
      action: "超过阈值标记为低质量"

    volume_fields:
      max_missing_rate: 0.10  # 10%
      action: "超过阈值标记为低质量"

  time_level:
    max_consecutive_missing: 24  # 小时
    action: "超过阈值标记为数据断档"
```

---

## 7. 质量控制 (QC)

### 7.1 QC 指标体系

**核心 QC 指标**：
```yaml
qc_metrics:

  completeness:
    metrics:
      - name: "missing_rate"
        formula: "缺失值 / 总观测数"
        threshold: "< 0.10"

      - name: "coverage_ratio"
        formula: "实际观测数 / 预期观测数"
        threshold: "> 0.90"

  accuracy:
    metrics:
      - name: "bad_price_rate"
        formula: "异常价格数 / 总观测数"
        threshold: "< 0.01"

      - name: "zero_volume_rate"
        formula: "零成交量数 / 总观测数"
        threshold: "< 0.05"

  consistency:
    metrics:
      - name: "ohlc_consistency"
        formula: "low <= open, close <= high"
        threshold: "100%"

      - name: "volume_positive"
        formula: "成交量 >= 0"
        threshold: "100%"

  timeliness:
    metrics:
      - name: "staleness_rate"
        formula: "价格未更新小时数 / 总小时数"
        threshold: "< 0.10"

      - name: "delay"
        formula: "数据时间戳 - 当前时间"
        threshold: "< 5 分钟"
```

### 7.2 QC 实现

**QC 检查函数**：
```python
def perform_data_qc(df):
    """
    执行数据质量检查
    """
    qc_results = {}

    # 1. 完整性检查
    qc_results['completeness'] = {
        'missing_rate': {
            'close': df['close'].isna().mean(),
            'volume': df['volume'].isna().mean(),
        },
        'coverage_ratio': len(df) / len(df)  # 实际/预期
    }

    # 2. 准确性检查
    # 异常价格：负值或零值
    qc_results['accuracy'] = {
        'bad_price_rate': (
            (df['close'] <= 0) | (df['high'] <= 0) | (df['low'] <= 0)
        ).mean(),
        'zero_volume_rate': (df['volume'] == 0).mean()
    }

    # 3. 一致性检查
    # OHLC 一致性：low <= open, close <= high
    ohlc_consistent = (
        (df['low'] <= df['open']) &
        (df['low'] <= df['close']) &
        (df['open'] <= df['high']) &
        (df['close'] <= df['high'])
    )
    qc_results['consistency'] = {
        'ohlc_consistency': ohlc_consistent.mean(),
        'volume_positive': (df['volume'] >= 0).mean()
    }

    # 4. 时效性检查
    # Staleness: 价格未更新
    df_sorted = df.sort_values('timestamp_aligned')
    price_unchanged = df_sorted.groupby('symbol')['close'].diff() == 0
    qc_results['timeliness'] = {
        'staleness_rate': price_unchanged.mean()
    }

    # 5. 异常值检查
    # 价格跳变：单日涨跌幅超过 50%
    price_change = df_sorted.groupby('symbol')['close'].pct_change()
    qc_results['outliers'] = {
        'extreme_move_rate': (abs(price_change) > 0.5).mean()
    }

    return qc_results
```

### 7.3 QC 报告

**QC 报告模板**：
```yaml
qc_report:
  symbol: "BTC_USDT"
  period: "2020-01-01 to 2024-12-31"

  completeness:
    missing_rate:
      close: 0.002  # 0.2%
      volume: 0.005  # 0.5%
    status: "PASS"
    threshold: "< 0.10"

  accuracy:
    bad_price_rate: 0.000  # 0%
    zero_volume_rate: 0.087  # 8.7%
    status: "WARNING"
    notes: "零成交量较多，可能在低流动性时段"

  consistency:
    ohlc_consistency: 1.000  # 100%
    volume_positive: 1.000  # 100%
    status: "PASS"

  timeliness:
    staleness_rate: 0.023  # 2.3%
    status: "PASS"

  outliers:
    extreme_move_rate: 0.001  # 0.1%
    status: "PASS"

  overall_status: "PASS"
  recommendations:
    - "注意低流动性时段的零成交量"
```

---

## 8. 覆盖审计

### 8.1 覆盖审计的目的

**为什么需要覆盖审计**：
```yaml
coverage_audit_purpose:

  sample_size_adequacy:
    question: "数据量是否足够统计检验？"
    consequence: "数据不足导致功效低、结果不可靠"

  temporal_representation:
    question: "各时期是否有足够数据？"
    consequence: "某些时期数据不足导致偏差"

  symbol_representation:
    question: "各标的是否有足够数据？"
    consequence: "某些标的被排除导致选择偏差"

  completeness_verification:
    question: "是否有系统性缺失？"
    consequence: "系统性缺失导致虚假模式"
```

### 8.2 覆盖审计方法

**审计维度**：
```python
def perform_coverage_audit(df, mandate_spec):
    """
    执行覆盖审计
    """
    audit_results = {}

    # 1. 时间覆盖
    required_periods = {
        'train': (mandate_spec['time_window']['train']['start'],
                  mandate_spec['time_window']['train']['end']),
        'test': (mandate_spec['time_window']['test']['start'],
                 mandate_spec['time_window']['test']['end']),
        'backtest': (mandate_spec['time_window']['backtest']['start'],
                     mandate_spec['time_window']['backtest']['end']),
        'holdout': (mandate_spec['time_window']['holdout']['start'],
                    mandate_spec['time_window']['holdout']['end'])
    }

    audit_results['time_coverage'] = {}
    for period, (start, end) in required_periods.items():
        period_data = df[
            (df['timestamp_aligned'] >= start) &
            (df['timestamp_aligned'] <= end)
        ]
        total_expected = len(required_periods) * 24 * len(df['symbol'].unique())
        audit_results['time_coverage'][period] = {
            'actual_observations': len(period_data),
            'expected_observations': total_expected,
            'coverage_ratio': len(period_data) / total_expected if total_expected > 0 else 0
        }

    # 2. 标的覆盖
    audit_results['symbol_coverage'] = {}
    for symbol in df['symbol'].unique():
        symbol_data = df[df['symbol'] == symbol]
        audit_results['symbol_coverage'][symbol] = {
            'total_observations': len(symbol_data),
            'missing_observations': symbol_data['close'].isna().sum(),
            'coverage_ratio': 1 - symbol_data['close'].isna().mean()
        }

    # 3. 字段覆盖
    audit_results['field_coverage'] = {}
    for field in ['open', 'high', 'low', 'close', 'volume']:
        if field in df.columns:
            audit_results['field_coverage'][field] = {
                'total_observations': len(df),
                'missing_observations': df[field].isna().sum(),
                'coverage_ratio': 1 - df[field].isna().mean()
            }

    return audit_results
```

### 8.3 覆盖审计报告

**审计报告模板**：
```markdown
## 覆盖审计报告

### 时间覆盖
| 阶段 | 预期观测数 | 实际观测数 | 覆盖率 | 状态 |
|------|-----------|-----------|--------|------|
| Train | 100,000 | 98,500 | 98.5% | ✅ PASS |
| Test | 20,000 | 19,200 | 96.0% | ✅ PASS |
| Backtest | 10,000 | 9,800 | 98.0% | ✅ PASS |
| Holdout | 5,000 | 4,900 | 98.0% | ✅ PASS |

### 标的覆盖
| 标的 | 总观测数 | 缺失数 | 覆盖率 | 状态 |
|------|---------|--------|--------|------|
| BTC_USDT | 35,040 | 70 | 99.8% | ✅ PASS |
| ETH_USDT | 35,040 | 175 | 99.5% | ✅ PASS |
| ... | ... | ... | ... | ... |

### 字段覆盖
| 字段 | 总观测数 | 缺失数 | 覆盖率 | 状态 |
|------|---------|--------|--------|------|
| close | 175,200 | 876 | 99.5% | ✅ PASS |
| volume | 175,200 | 3,504 | 98.0% | ✅ PASS |
| ... | ... | ... | ... | ... |

### 总体评估
- **时间覆盖**: ✅ 所有阶段覆盖率 > 95%
- **标的覆盖**: ✅ 所有标的覆盖率 > 95%
- **字段覆盖**: ✅ 所有关键字段覆盖率 > 95%

### 建议
- 数据覆盖充分，可以进入下一阶段
```

---

## 9. 数据分层设计

### 9.1 数据分层架构

**为什么要分层**：
```yaml
layering_rationale:

  separation_of_concerns:
    raw_layer: "原始数据，不做修改"
    cleaned_layer: "清洗后数据，可复现"
    feature_layer: "特征工程，复用性"

  traceability:
    benefit: "每一层都能追溯到上一层"
    requirement: "保留处理日志"

  reproducibility:
    benefit: "从原始数据重新生成"
    requirement: "代码 + 参数 = 结果"

  efficiency:
    benefit: "中间结果缓存"
    requirement: "版本控制"
```

### 9.2 数据分层定义

**三层架构**：
```yaml
data_layers:

  raw_layer:
    name: "原始数据层"
    purpose: "存储原始获取的数据"
    characteristics:
      - "不做任何修改"
      - "保留原始格式"
      - "只读"
    schema:
      - "timestamp_raw"
      - "symbol"
      - "open_raw"
      - "high_raw"
      - "low_raw"
      - "close_raw"
      - "volume_raw"

  cleaned_layer:
    name: "清洗数据层"
    purpose: "时间对齐、QC 后的数据"
    characteristics:
      - "时间标准化"
      - "QC 标记"
      - "缺失值标记"
    schema:
      - "timestamp_utc"
      - "timestamp_aligned"
      - "symbol"
      - "open"
      - "high"
      - "low"
      - "close"
      - "volume"
      - "_is_missing"
      - "_period_available"
      - "qc_*"

  feature_layer:
    name: "特征数据层"
    purpose: "信号计算用的特征"
    characteristics:
      - "派生特征"
      - "标准化特征"
      - "可复用"
    schema:
      - "timestamp_utc"
      - "symbol"
      - "returns_1h"
      - "returns_24h"
      - "volatility_24h"
      - "volume_ma_24h"
      - "rsi_14"
      - "..."
```

### 9.3 数据处理流程

**处理流程**：
```python
def create_cleaned_layer(raw_df):
    """
    创建清洗数据层
    """
    df = raw_df.copy()

    # 1. 时间标准化
    df['timestamp_utc'] = standardize_timestamps(df)
    df['timestamp_aligned'] = align_to_frequency(df)

    # 2. 时间对齐
    df = handle_missing_timestamps(df)

    # 3. QC 检查
    qc_results = perform_data_qc(df)

    # 4. 添加 QC 标记
    for metric, value in qc_results.items():
        df[f'_qc_{metric}'] = value

    # 5. 缺失值标记
    df = handle_missing_values(df, {})

    return df

def create_feature_layer(cleaned_df):
    """
    创建特征数据层
    """
    df = cleaned_df.copy()

    # 1. 收益率特征
    df['returns_1h'] = df.groupby('symbol')['close'].pct_change()
    df['returns_24h'] = df.groupby('symbol')['close'].pct_change(24)

    # 2. 波动率特征
    df['volatility_24h'] = df.groupby('symbol')['returns_1h'].rolling(24).std()

    # 3. 成交量特征
    df['volume_ma_24h'] = df.groupby('symbol')['volume'].rolling(24).mean()

    # 4. 技术指标
    df['rsi_14'] = calculate_rsi(df['close'], 14)

    return df
```

---

## 10. Formal Gate 要求

### 10.1 Data Ready 阶段 Formal Gate

**必需检查项**：
```yaml
data_ready_formal_gates:

  data_acquired:
    requirement: "数据已获取并验证"
    criteria:
      - "时间范围覆盖 Mandate 需求"
      - "Universe 所有标的有数据"
      - "所需字段都可用"
    evidence: "data_acquisition_report.md"
    status: "PASS | FAIL"

  time_aligned:
    requirement: "时间戳已标准化和对齐"
    criteria:
      - "统一时区 (UTC)"
      - "统一频率"
      - "缺失时间点已处理"
    evidence: "time_alignment_report.md"
    status: "PASS | FAIL"

  quality_checked:
    requirement: "数据质量检查已完成"
    criteria:
      - "QC 指标计算完成"
      - "所有标的 QC 报告生成"
      - "低质量数据已标记"
    evidence: "qc_report.yaml"
    status: "PASS | FAIL"

  coverage_adequate:
    requirement: "数据覆盖充分"
    criteria:
      - "时间覆盖率 > 95%"
      - "标的覆盖率 > 95%"
      - "字段覆盖率 > 95%"
    evidence: "coverage_audit_report.md"
    status: "PASS | FAIL"

  layers_created:
    requirement: "数据分层已创建"
    criteria:
      - "原始数据层存在"
      - "清洗数据层存在"
      - "特征数据层存在 (如需要)"
    evidence: "data_layer_manifest.yaml"
    status: "PASS | FAIL"

  documentation_complete:
    requirement: "文档和字段字典完整"
    criteria:
      - "数据处理日志完整"
      - "字段字典完整"
      - "产物目录完整"
    evidence: "field_dictionary.md + artifact_catalog.md"
    status: "PASS | FAIL"
```

### 10.2 决策状态

**可能的状态**：
```yaml
verdict_states:

  PASS:
    description: "所有 Formal Gate 通过，可进入 Signal Ready"
    frozen_items:
      - "数据范围"
      - "时间对齐规范"
      - "QC 标准"
      - "数据分层架构"

  RETRY:
    description: "数据获取或处理有问题，修复后重试"
    scope:
      - "重新获取数据"
      - "修复处理逻辑"
      - "调整 QC 标准"
    forbidden:
      - "不能改变 Mandate 定义的范围"

  RESEARCH_AGAIN:
    description: "数据问题无法解决，需要回到 Mandate"
    rollback_stage: "00_mandate"
    reasons:
      - "数据不可得"
      - "数据质量太差"
      - "覆盖严重不足"

  NO_GO:
    description: "数据问题导致研究不可行，终止"
    reasons:
      - "关键数据永久缺失"
      - "数据成本过高"
      - "法律合规问题"
```

---

## 11. 常见错误与防范

### 11.1 错误清单

| 错误类型 | 描述 | 后果 | 防范措施 |
|---------|------|------|---------|
| **静默填充** | 用 forward-fill 掩盖缺失 | 虚假信号 | 显式标记缺失 |
| **时间错位** | 时间戳不统一 | 虚假因果关系 | 严格时间对齐 |
| **选择偏差** | 只保留数据好的标的 | 幸存者偏差 | 覆盖审计 |
| **前视偏差** | 使用未来信息 | 虚假良好表现 | 严格时间顺序 |
| **单位混淆** | 不同标的单位不同 | 计算错误 | 标准化单位 |
| **忽略 QC** | 不做数据质量检查 | 结果不可靠 | 强制 QC 检查 |
| **过度清洗** | 删除太多数据 | 样本不足 | 保守处理原则 |

### 11.2 防范机制

**代码层面**：
```python
# 强制显式标记
def safe_forward_fill(series, max_fill=1):
    """
    安全的前向填充，限制连续填充数
    """
    filled = series.copy()
    count = 0

    for i in range(1, len(filled)):
        if pd.isna(filled.iloc[i]):
            if count < max_fill and not pd.isna(filled.iloc[i-1]):
                filled.iloc[i] = filled.iloc[i-1]
                count += 1
            else:
                count = 0

    return filled

# 防止前视偏差
def ensure_no_lookahead(df):
    """
    确保没有使用未来信息
    """
    # 检查是否有未来函数
    for col in df.columns:
        if 'future' in col.lower() or 'lead' in col.lower():
            warnings.warn(f"可能的前视偏差: {col}")

    # 检查时间排序
    if not df['timestamp'].is_monotonic_increasing:
        raise ValueError("数据未按时间排序，可能存在前视偏差")

    return df
```

**流程层面**：
```yaml
safety_checks:

  code_review:
    - "检查数据处理逻辑"
    - "验证时间顺序"
    - "确认无未来函数"

  automated_tests:
    - "时间单调性测试"
    - "覆盖率测试"
    - "QC 阈值测试"

  peer_review:
    - "独立审查数据处理"
    - "验证 QC 结果"
    - "确认数据可用性"
```

---

## 12. 输出 Artifact

### 12.1 机器可读产物

**必需文件**：
```yaml
machine_readable_artifacts:

  raw_data.parquet:
    description: "原始数据层"
    schema:
      - "timestamp_raw"
      - "symbol"
      - "open_raw"
      - "high_raw"
      - "low_raw"
      - "close_raw"
      - "volume_raw"

  cleaned_data.parquet:
    description: "清洗数据层"
    schema:
      - "timestamp_utc"
      - "timestamp_aligned"
      - "symbol"
      - "open"
      - "high"
      - "low"
      - "close"
      - "volume"
      - "_is_missing"
      - "_period_available"
      - "_qc_*"

  qc_report.yaml:
    description: "QC 报告"
    content:
      - "symbol_level_qc"
      - "field_level_qc"
      - "overall_qc_summary"

  coverage_audit.yaml:
    description: "覆盖审计报告"
    content:
      - "time_coverage"
      - "symbol_coverage"
      - "field_coverage"

  data_layer_manifest.yaml:
    description: "数据分层清单"
    content:
      - "layer_names"
      - "layer_paths"
      - "layer_schemas"
```

### 12.2 人类可读产物

**必需文档**：
```yaml
human_readable_artifacts:

  data_ready_report.md:
    description: "Data Ready 阶段总结报告"
    sections:
      - "执行摘要"
      - "数据获取情况"
      - "数据质量评估"
      - "覆盖审计结果"
      - "数据处理方法"
      - "发现的问题"
      - "下一步计划"

  field_dictionary.md:
    description: "字段字典"
    content:
      - "所有字段的说明"
      - "字段类型"
      - "字段含义"
      - "单位"
      - "是否可空"
      - "空值语义"

  artifact_catalog.md:
    description: "产物目录"
    content:
      - "产物列表"
      - "用途说明"
      - "消费者说明"

  data_processing_log.md:
    description: "数据处理日志"
    content:
      - "处理步骤"
      - "参数设置"
      - "中间结果"
      - "问题记录"
```

---

## 13. 与 Signal Ready 的交接

### 13.1 交接内容

**Data Ready → Signal Ready 交接清单**：
```yaml
data_ready_handover:

  frozen_spec:
    file: "data_ready_spec.yaml"
    content:
      - "数据范围"
      - "时间对齐规范"
      - "QC 标准"
      - "数据分层架构"

  data_layers:
    cleaned_layer:
      file: "cleaned_data.parquet"
      description: "可用于信号计算的数据"
      schema: {...}

  quality_assurance:
    qc_report:
      file: "qc_report.yaml"
      description: "所有数据质量检查结果"

    coverage_audit:
      file: "coverage_audit.yaml"
      description: "覆盖审计结果"

  documentation:
    field_dictionary: "field_dictionary.md"
    processing_log: "data_processing_log.md"
```

### 13.2 Signal Ready 阶段职责

**基于 Data Ready 的信号定义**：
1. **使用清洗数据**: 从 cleaned_layer 读取数据
2. **遵循字段定义**: 按 field_dictionary 使用字段
3. **理解数据特性**: 参考 QC 报告了解数据质量
4. **复用特征**: 使用 feature_layer 的特征

### 13.3 禁止事项

**Signal Ready 不能做的事**：
```yaml
forbidden_in_signal_ready:
  data_reacquisition: "禁止重新获取数据"
  time_realignment: "禁止重新对齐时间"
  qc_standard_change: "禁止改变 QC 标准"
  layer_recreation: "禁止重新创建数据层"
```

---

## 14. 数据质量报告模板

### 14.1 QC 报告模板

```markdown
---
doc_id: DR-QC-{lineage_id}-{run_id}
title: Data Quality Report — {项目名称}
date: YYYY-MM-DD
stage: 01_data_ready
lineage_id: {lineage_id}
run_id: {run_id}
---

## 执行摘要

**数据来源**: {数据源}
**数据期间**: {开始日期} 至 {结束日期}
**标的数量**: {N} 个

**总体评估**: {PASS / WARNING / FAIL}

**关键发现**:
- 数据覆盖率: {XX.XX}%
- QC 通过率: {XX.XX}%
- 主要问题: {问题描述}

## 数据覆盖

### 时间覆盖
| 阶段 | 预期观测数 | 实际观测数 | 覆盖率 | 状态 |
|------|-----------|-----------|--------|------|
| Train | {N} | {N} | {XX%} | {状态} |
| Test | {N} | {N} | {XX%} | {状态} |
| Backtest | {N} | {N} | {XX%} | {状态} |
| Holdout | {N} | {N} | {XX%} | {状态} |

### 标的覆盖
| 标的 | 总观测数 | 缺失数 | 覆盖率 | 状态 |
|------|---------|--------|--------|------|
| {标的1} | {N} | {N} | {XX%} | {状态} |
| {标的2} | {N} | {N} | {XX%} | {状态} |

### 字段覆盖
| 字段 | 总观测数 | 缺失数 | 覆盖率 | 状态 |
|------|---------|--------|--------|------|
| close | {N} | {N} | {XX%} | {状态} |
| volume | {N} | {N} | {XX%} | {状态} |

## 数据质量

### 完整性
| 指标 | 值 | 阈值 | 状态 |
|------|-----|------|------|
| 缺失率 (close) | {X.XX%} | < 10% | {状态} |
| 缺失率 (volume) | {X.XX%} | < 10% | {状态} |

### 准确性
| 指标 | 值 | 阈值 | 状态 |
|------|-----|------|------|
| 坏价率 | {X.XX%} | < 1% | {状态} |
| 零成交量率 | {X.XX%} | < 5% | {状态} |

### 一致性
| 指标 | 值 | 阈值 | 状态 |
|------|-----|------|------|
| OHLC 一致性 | {XX%} | 100% | {状态} |
| 成交量非负 | {XX%} | 100% | {状态} |

### 时效性
| 指标 | 值 | 阈值 | 状态 |
|------|-----|------|------|
| 停滞率 | {X.XX%} | < 10% | {状态} |

### 异常值
| 指标 | 值 | 阈值 | 状态 |
|------|-----|------|------|
| 极端变动率 | {X.XX%} | < 1% | {状态} |

## 数据处理

### 时间对齐
- **原始时区**: {时区}
- **目标时区**: UTC
- **对齐频率**: {频率}
- **处理方法**: {方法描述}

### 缺失值处理
- **完全缺失**: {处理方法}
- **部分缺失**: {处理方法}
- **暂时性缺失**: {处理方法}

### 数据分层
- **原始数据层**: {路径}
- **清洗数据层**: {路径}
- **特征数据层**: {路径}

## 发现的问题

### 严重问题
{如果有严重问题，列出并说明影响}

### 警告问题
{如果有警告问题，列出并说明影响}

### 建议优化
{优化建议}

## 总体评估

### 可用性评估
- **统计研究**: {是/否}
- **信号计算**: {是/否}
- **回测使用**: {是/否}

### 限制条件
{数据使用的限制条件}

### 下一步建议
{对 Signal Ready 阶段的建议}

## 附录

### A. 标的详细 QC
{每个标的的详细 QC 结果}

### B. 时间序列图
{数据覆盖率时间序列图}

### C. 处理代码
{关键处理代码片段}
```

---

**文档版本**: v1.0
**最后更新**: 2026-03-26
**下次评审**: 2026-06-26
