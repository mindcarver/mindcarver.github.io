# DataReady 阶段 - 横截面因子数据就绪

## 目录
1. [阶段定义](#阶段定义)
2. [为什么需要 DataReady 阶段](#为什么需要-dataready-阶段)
3. [Panel 数据结构](#panel-数据结构)
4. [Universe 过滤](#universe-过滤)
5. [时间对齐 (Alignment)](#时间对齐-alignment)
6. [Leakage 检查](#leakage-检查)
7. [数据质量验证](#数据质量验证)
8. [Formal Gate 要求](#formal-gate-要求)
9. [常见错误和反模式](#常见错误和反模式)
10. [实际案例：加密货币动量因子数据准备](#实际案例加密货币动量因子数据准备)
11. [输出 Artifact 规范](#输出-artifact-规范)
12. [与下一阶段的交接标准](#与下一阶段的交接标准)

---

## 阶段定义

### DataReady 定义

**DataReady（数据就绪）** 确认原始数据可以被转换为可用于横截面因子研究的 Panel 数据结构。

### 核心目标

| 目标 | 横截面研究特有含义 | 价值 |
|------|------------------|------|
| **构建 Panel 结构** | 时间 × 标的二维数据矩阵 | 奠定分析基础 |
| **确保 Universe 稳定** | 研究期间标的池一致性 | 避免选择偏差 |
| **验证时间对齐** | 所有标的时间戳一致 | 防止虚假信号 |
| **排除 Leakage** | 因子计算不使用未来信息 | 确保结果有效 |

### 阶段定位

```
Mandate → DataReady → PanelReady → FactorReady → ...
         ↑
    数据准备层
```

DataReady 是横截面因子研究的数据基础——所有因子计算和检验都依赖这个阶段构建的 Panel 数据。

---

## 为什么需要 DataReady 阶段

### 横截面因子数据的特殊挑战

**挑战 1：Panel 数据缺失模式复杂**

```python
# Panel 数据的多种缺失模式
panel_missing = {
    '时间缺失': '某天整个市场停牌',
    '标的缺失': '新币上市前、退市后',
    '随机缺失': '数据中断',
    '结构性缺失': '某币种没有某字段'
}
```

**挑战 2：Universe 动态变化**

```python
# 加密货币市场的动态性
universe_changes = {
    '新币上市': '每月新增多个币种',
    '退市': '币种可能下架',
    '流动性变化': 'Top 50 成分不断变化'
}

# 如果不处理，会导致选择偏差
```

**挑战 3：时间对齐问题**

```python
# 不同数据源的时间戳不一致
time_issues = {
    '时区问题': '交易所使用不同时区',
    '对齐问题': 'K线时间戳定义不同',
    '夏令时': '某些地区夏令时切换'
}
```

### DataReady 的价值

有个实际教训：
```yaml
before_data_ready:
  problem: "因子在 Train 上显著，Test 上失效"
  investigation: "花了1周排查因子逻辑"
  root_cause: "Test 期间多个新币上市，改变了 Universe 构成"

after_data_ready:
  qc_report: >
    DataReady 阶段发现 Test 期间
    Universe 成分变化超过 30%，
    提前调整处理策略
  outcome: "节省1周排查时间，结果更可靠"
```

---

## Panel 数据结构

### Panel 数据定义

**Panel 数据**是横截面因子研究的核心数据结构，有两个维度：时间（T）和标的（N）。

```python
import pandas as pd

# Panel 数据结构示例
panel_structure = {
    'shape': '(T, N)',  # T 个时间点，N 个标的
    'index': '时间戳',
    'columns': '标的代码',
    'values': '观测值（价格、收益率等）'
}

# 实际示例
# 假设有 1000 天，50 个币种
panel_shape = (1000, 50)  # 50,000 个观测
```

### Panel 数据类型

```yaml
Panel 数据类型:
  
  价格 Panel:
    维度: (时间 × 标的)
    值: 价格或收益率
    示例: "每个时间点每个币的收盘价"
    
  因子 Panel:
    维度: (时间 × 标的)
    值: 计算得到的因子值
    示例: "每个时间点每个币的动量值"
    
  标签 Panel:
    维度: (时间 × 标的)
    值: 未来收益
    示例: "每个时间点每个币的未来24h收益"
    
  可用性 Panel:
    维度: (时间 × 标的)
    值: 布尔值，表示该观测是否可用
    示例: "标记哪些时间点哪些币可以用于分析"
```

### Panel 数据示例

```python
# 价格 Panel 示例
price_panel = pd.DataFrame(
    index=pd.date_range('2021-01-01', '2024-12-31', freq='D'),
    columns=['BTC', 'ETH', 'BNB', 'SOL', 'ADA', ...],
    data=...  # 价格数据
)

# 展示结构
print(price_panel.shape)  # (1461, 42) - 1461天，42个币
print(price_panel.head())
"""
            BTC      ETH      BNB      SOL      ADA
2021-01-01  35000    1200     40       5        0.5
2021-01-02  36000    1250     42       5.5      0.52
...
"""
```

### Panel 操作基础

```python
# 计算 Panel 收益率
def calculate_panel_returns(price_panel):
    """
    计算 Panel 收益率
    
    沿时间轴计算收益率，保持 Panel 结构
    """
    returns_panel = price_panel.pct_change()
    return returns_panel

# 横截面操作
def cross_sectional_rank(panel_at_t):
    """
    计算某时间点的横截面排序
    """
    ranks = panel_at_t.rank(pct=True)
    return ranks

# 应用到整个 Panel
rank_panel = price_panel.apply(
    lambda row: row.rank(pct=True),
    axis=1  # 沿横截面（标的）方向
)
```

---

## Universe 过滤

### Universe 过滤的目的

Universe 过滤确保研究期间使用的标的池符合 Mandate 定义，同时处理动态变化。

### 过滤步骤

#### 步骤 1：应用 Mandate 准入口径

```python
def apply_eligibility_criteria(raw_data, mandate_config):
    """
    应用 Mandate 定义的准入口径
    """
    filtered_data = raw_data.copy()
    
    # 按交易量过滤
    volume_threshold = calculate_top_n_volume(
        filtered_data, 
        n=mandate_config['target_market']['baseline_count']
    )
    filtered_data = filtered_data[
        filtered_data['volume_24h'] >= volume_threshold
    ]
    
    # 排除稳定币
    stablecoins = mandate_config['target_market']['excluded_stablecoins']
    filtered_data = filtered_data[
        ~filtered_data['symbol'].isin(stablecoins)
    ]
    
    # 排除杠杆代币
    leveraged = ['UP', 'DOWN', 'BULL', 'BEAR']
    filtered_data = filtered_data[
        ~filtered_data['symbol'].str.contains('|'.join(leveraged))
    ]
    
    # 最小历史天数
    min_days = mandate_config['factor_definition']['data_requirements']['min_history']
    filtered_data = filter_by_history_length(filtered_data, min_days)
    
    return filtered_data
```

#### 步骤 2：处理 Universe 动态变化

```python
def handle_universe_changes(panel_data, mandate_config):
    """
    处理研究期间 Universe 的动态变化
    """
    # 策略 1：冻结初始 Universe（推荐）
    if mandate_config['universe_strategy'] == 'freeze_initial':
        initial_universe = get_universe_at_start(panel_data)
        filtered_panel = panel_data[initial_universe]
        
        # 新上市币种不纳入
        # 退市币种保留历史数据，之后标记为不可用
        
    # 策略 2：动态 Universe（谨慎使用）
    elif mandate_config['universe_strategy'] == 'dynamic':
        # 每个时间点使用当时的 Top N
        # 需要注意：可能引入前视偏差
        filtered_panel = apply_dynamic_universe(panel_data)
        
    return filtered_panel
```

#### 步骤 3：生成可用性矩阵

```python
def create_availability_matrix(panel_data, universe_config):
    """
    生成可用性矩阵
    
    标记每个时间点每个标的是否可用于分析
    """
    availability = pd.DataFrame(
        index=panel_data.index,
        columns=panel_data.columns,
        dtype=bool
    )
    
    for symbol in panel_data.columns:
        # 检查数据是否存在
        data_exists = panel_data[symbol].notna()
        
        # 检查是否在 Universe 中
        in_universe = is_in_universe_at_t(
            symbol, 
            panel_data.index,
            universe_config
        )
        
        # 检查历史数据是否足够
        sufficient_history = has_sufficient_history(
            panel_data[symbol],
            min_days=universe_config['min_history']
        )
        
        availability[symbol] = data_exists & in_universe & sufficient_history
    
    return availability
```

### Universe 过滤配置

```yaml
Universe 过滤配置:
  
  过滤策略: "freeze_initial"
    理由: "避免选择偏差"
    实现: "使用研究开始时的 Universe"
  
  新币处理:
    策略: "不纳入"
    理由: "避免事后选择"
  
  退市处理:
    策略: "保留历史，标记不可用"
    理由: "避免幸存者偏差"
  
  最小历史:
    要求: "至少 20 天数据"
    应用: "计算因子前检查"
```

---

## 时间对齐 (Alignment)

### 时间对齐的重要性

横截面因子研究要求所有标的时间戳严格对齐。时间不对齐会直接产生虚假的横截面关系。

### 对齐步骤

#### 步骤 1：时区标准化

```python
def standardize_timezone(raw_data, target_timezone='UTC'):
    """
    标准化所有时间戳到统一时区
    """
    # 加密货币通常使用 UTC
    # 确保所有数据都在同一时区
    
    if raw_data.index.tz is None:
        raw_data.index = raw_data.index.tz_localize(target_timezone)
    else:
        raw_data.index = raw_data.index.tz_convert(target_timezone)
    
    return raw_data
```

#### 步骤 2：频率对齐

```python
def align_frequency(panel_data, frequency='1h'):
    """
    对齐到指定频率
    """
    # 向下对齐到整点
    aligned_index = panel_data.index.floor(frequency)
    
    # 重新采样
    aligned_data = panel_data.copy()
    aligned_data.index = aligned_index
    
    # 去除重复时间戳
    aligned_data = aligned_data[~aligned_data.index.duplicated(keep='last')]
    
    return aligned_data
```

#### 步骤 3：处理缺失时间点

```python
def handle_missing_timestamps(panel_data, frequency='1h'):
    """
    处理缺失的时间点
    
    关键：保留缺失标记，不填充
    """
    # 创建完整的时间范围
    full_range = pd.date_range(
        start=panel_data.index.min(),
        end=panel_data.index.max(),
        freq=frequency
    )
    
    # 重新索引
    aligned_panel = panel_data.reindex(full_range)
    
    # 标记缺失
    for col in aligned_panel.columns:
        aligned_panel[f'{col}_is_missing'] = aligned_panel[col].isna()
    
    return aligned_panel
```

### 时间对齐验证

```python
def validate_time_alignment(panel_data):
    """
    验证时间对齐质量
    """
    validation_results = {}
    
    # 检查时区一致性
    validation_results['timezone_consistent'] = (
        panel_data.index.tz.zone == 'UTC'
    )
    
    # 检查频率一致性
    time_diffs = panel_data.index.to_series().diff()
    expected_diff = pd.Timedelta('1h')
    validation_results['frequency_consistent'] = (
        time_diffs.mode()[0] == expected_diff
    )
    
    # 检查缺失率
    validation_results['missing_rate'] = (
        panel_data.isna().mean().mean()
    )
    
    return validation_results
```

---

## Leakage 检查

### Leakage 的定义和危害

**Leakage（信息泄漏）** 指计算因子或标签时使用了当时不可获得的未来信息。这是横截面因子研究中最严重的错误，没有之一。

### Leakage 类型

```yaml
Leakage 类型:
  
  时间 Leakage:
    描述: "使用了未来时刻的数据"
    示例: "计算 T 时刻因子时使用了 T+1 的价格"
    危害: "造成虚假的预测能力"
  
  数据 Leakage:
    描述: "使用了当时未发布的数据"
    示例: "使用了盘后才发布的财报数据"
    危害: "实盘无法复现"
  
  标的 Leakage:
    描述: "基于未来信息选择标的"
    示例: "只选择后来表现好的币"
    危害: "选择偏差"
```

### Leakage 检查方法

#### 方法 1：时间顺序检查

```python
def check_temporal_order(panel_data, factor_col, label_col, horizon):
    """
    检查时间顺序是否正确
    
    因子只能使用过去的数据，标签只能用未来的数据
    """
    issues = []
    
    for t in panel_data.index:
        # 因子计算：只能用 t 及之前的数据
        factor_data = panel_data.loc[:t, factor_col]
        
        # 标签计算：应该用 t+horizon 的数据
        label_time = t + pd.Timedelta(hours=horizon)
        
        if label_time in panel_data.index:
            label_data = panel_data.loc[label_time, label_col]
            
            # 检查：因子数据不能包含未来信息
            # 这通常需要在因子计算层面保证
        else:
            issues.append(f"Label time {label_time} not in data")
    
    return issues
```

#### 方法 2：前视偏差检测

```python
def detect_lookahead_bias(factor_values, price_data):
    """
    检测因子是否包含前视偏差
    
    原理：如果因子包含未来信息，IC 会异常高
    """
    # 计算因子与未来收益的相关性
    ics = []
    
    for t in range(len(factor_values) - 24):
        factor_t = factor_values.iloc[t]
        future_return = price_data.iloc[t + 24]
        
        ic = spearmanr(factor_t, future_return)[0]
        ics.append(ic)
    
    # 如果 IC 异常高（> 0.3），可能存在前视偏差
    if np.mean(ics) > 0.3:
        warnings.warn("Abnormally high IC detected. Check for lookahead bias!")
    
    return ics
```

#### 方法 3：完整性检查

```python
def check_data_leakage(panel_data, mandate_config):
    """
    检查数据泄漏
    """
    leakage_report = {
        'temporal_leakage': [],
        'data_leakage': [],
        'universe_leakage': []
    }
    
    # 检查 1：因子计算时间
    # 确保因子计算只用过去数据
    for t in panel_data.index:
        # 检查是否有未来函数调用
        # 这通常需要代码审计
        pass
    
    # 检查 2：数据可用性
    # 确保使用的数据在当时是可获得的
    for col in panel_data.columns:
        # 检查数据发布时间 vs 计算时间
        pass
    
    # 检查 3：Universe 选择
    # 确保 Universe 选择没有使用未来信息
    # 例如：不能根据未来收益选择标的
    pass
    
    return leakage_report
```

### Leakage 防范措施

```yaml
Leakage 防范:
  
  代码层面:
    - "使用 `shift()` 确保只用过去数据"
    - "禁止在因子计算中使用未来函数"
    - "时间戳严格审查"
  
  流程层面:
    - "Mandate 阶段明确数据可用性"
    - "DataReady 阶段验证时间对齐"
    - "代码审查重点检查时间逻辑"
  
  验证层面:
    - "计算 Forward IC"
    - "检查异常高的 IC"
    - "实盘前 paper trading"
```

---

## 数据质量验证

### 质量指标

```yaml
Panel 数据质量指标:
  
  完整性:
    缺失率: "< 10%"
    覆盖率: "> 90%"
    
  准确性:
    坏价率: "< 1%"
    异常值率: "< 5%"
    
  一致性:
    时间一致性: "100%"
    标的一致性: "100%"
    
  及时性:
    延迟: "< 数据发布间隔"
```

### QC 实现

```python
def perform_panel_qc(panel_data, availability_matrix):
    """
    执行 Panel 数据质量检查
    """
    qc_results = {}
    
    # 1. 覆盖率检查
    qc_results['coverage'] = {
        'overall_coverage': availability_matrix.mean().mean(),
        'time_coverage': availability_matrix.mean(axis=1).describe(),
        'symbol_coverage': availability_matrix.mean(axis=0).describe()
    }
    
    # 2. 缺失模式检查
    qc_results['missing_patterns'] = {
        'consecutive_missing': check_consecutive_missing(availability_matrix),
        'random_missing': check_random_missing(availability_matrix),
        'block_missing': check_block_missing(availability_matrix)
    }
    
    # 3. 异常值检查
    qc_results['outliers'] = {
        'price_spikes': detect_price_spikes(panel_data),
        'zero_prices': detect_zero_prices(panel_data),
        'extreme_returns': detect_extreme_returns(panel_data)
    }
    
    # 4. 横截面检查
    qc_results['cross_sectional'] = {
        'min_symbols_per_period': availability_matrix.sum(axis=1).min(),
        'min_periods_per_symbol': availability_matrix.sum(axis=0).min()
    }
    
    return qc_results
```

---

## Formal Gate 要求

### FG-1: Panel 结构完整

```yaml
检查标准:
  - Panel 数据维度正确 (T × N)
  - 时间索引连续且对齐
  - 标的列完整
  - 数据类型正确

验收方式:
  - Panel 结构验证脚本
  - 数据形状报告
```

### FG-2: Universe 过滤正确

```yaml
检查标准:
  - 准入口径正确应用
  - 排除规则正确执行
  - Universe 变化记录完整
  - 可用性矩阵生成

验收方式:
  - universe_manifest.csv 更新
  - availability_matrix.parquet
  - 过滤日志
```

### FG-3: 时间对齐验证

```yaml
检查标准:
  - 所有数据统一时区 (UTC)
  - 时间戳对齐到指定频率
  - 缺失时间点正确处理
  - 对齐质量报告通过

验收方式:
  - time_alignment_report.yaml
  - 对齐验证脚本
```

### FG-4: Leakage 检查通过

```yaml
检查标准:
  - 无时间 Leakage
  - 无数据 Leakage
  - 无标的 Leakage
  - Leakage 报告无严重问题

验收方式:
  - leakage_check_report.yaml
  - 代码审计记录
```

### FG-5: 数据质量达标

```yaml
检查标准:
  - 覆盖率 > 90%
  - 缺失率 < 10%
  - 无严重数据问题
  - QC 报告完整

验收方式:
  - qc_report.yaml
  - data_quality_summary.md
```

---

## 常见错误和反模式

### 错误 1：Universe 悄悄改变

**反模式**：
```python
# 研究中途悄悄改变 Universe
if period == 'test':
    universe = top_50_by_volume()  # 每天重新选
else:
    universe = initial_universe
```

**正确做法**：
```python
# Mandate 阶段冻结 Universe
universe = get_universe_at_start(mandate_config)
# 整个研究期间保持一致
```

### 错误 2：时间不对齐导致虚假信号

**反模式**：
```python
# 不同币种使用不同时区
btc_price = get_price('BTC', timezone='UTC')
eth_price = get_price('ETH', timezone='EST')  # 错误！
```

**正确做法**：
```python
# 统一时区
btc_price = get_price('BTC', timezone='UTC').tz_convert('UTC')
eth_price = get_price('ETH', timezone='EST').tz_convert('UTC')
```

### 错误 3：Leakage 导致虚假结果

**反模式**：
```python
# 计算 T 时刻的因子，但用了 T+1 的数据
def calculate_factor_at_t(data, t):
    return data[t:t+20].mean()  # 包含未来数据！
```

**正确做法**：
```python
# 只用过去的数据
def calculate_factor_at_t(data, t):
    return data[t-20:t].mean()  # 只用过去数据
```

---

## 实际案例：加密货币动量因子数据准备

### 数据准备流程

```python
# 完整的 DataReady 流程示例
class CryptoFactorDataReady:
    def __init__(self, mandate_config):
        self.mandate = mandate_config
        self.raw_data = None
        self.panel_data = None
        self.availability_matrix = None
    
    def run(self):
        """执行完整 DataReady 流程"""
        # 1. 加载原始数据
        self.load_raw_data()
        
        # 2. 构建 Panel
        self.build_panel()
        
        # 3. 应用 Universe 过滤
        self.apply_universe_filter()
        
        # 4. 时间对齐
        self.align_time()
        
        # 5. 生成可用性矩阵
        self.create_availability_matrix()
        
        # 6. Leakage 检查
        self.check_leakage()
        
        # 7. 数据质量验证
        self.run_qc()
        
        # 8. 输出结果
        self.save_results()
    
    def load_raw_data(self):
        """加载原始数据"""
        # 从 Binance API 或数据库加载
        symbols = self.mandate['universe']['baseline_symbols']
        
        data_list = []
        for symbol in symbols:
            symbol_data = load_binance_data(
                symbol,
                start=self.mandate['time_window']['study_start'],
                end=self.mandate['time_window']['study_end']
            )
            data_list.append(symbol_data)
        
        self.raw_data = pd.concat(data_list)
    
    def build_panel(self):
        """构建 Panel 数据"""
        # 透视为 Panel 结构
        self.panel_data = self.raw_data.pivot(
            index='timestamp',
            columns='symbol',
            values='close'
        )
    
    def apply_universe_filter(self):
        """应用 Universe 过滤"""
        # 获取初始 Universe
        initial_universe = self.get_initial_universe()
        
        # 过滤到初始 Universe
        self.panel_data = self.panel_data[initial_universe]
        
        # 记录 Universe 变化
        self.universe_changes = self.track_universe_changes()
    
    def align_time(self):
        """时间对齐"""
        # 标准化时区
        self.panel_data.index = self.panel_data.index.tz_convert('UTC')
        
        # 对齐到小时
        self.panel_data.index = self.panel_data.index.floor('1h')
        
        # 处理重复时间戳
        self.panel_data = self.panel_data[~self.panel_data.index.duplicated(keep='last')]
    
    def create_availability_matrix(self):
        """生成可用性矩阵"""
        self.availability_matrix = pd.DataFrame(
            index=self.panel_data.index,
            columns=self.panel_data.columns,
            dtype=bool
        )
        
        for symbol in self.panel_data.columns:
            # 检查数据是否存在
            exists = self.panel_data[symbol].notna()
            
            # 检查历史数据是否足够
            sufficient = self.check_sufficient_history(symbol)
            
            self.availability_matrix[symbol] = exists & sufficient
    
    def check_leakage(self):
        """Leakage 检查"""
        # 检查时间顺序
        self.temporal_check = self.check_temporal_order()
        
        # 检查数据可用性
        self.data_leakage_check = self.check_data_availability()
        
        # 生成报告
        self.leakage_report = {
            'temporal_leakage': self.temporal_check,
            'data_leakage': self.data_leakage_check,
            'overall_status': 'PASS' if all([
                self.temporal_check['status'] == 'OK',
                self.data_leakage_check['status'] == 'OK'
            ]) else 'FAIL'
        }
    
    def run_qc(self):
        """数据质量检查"""
        self.qc_results = {
            'coverage': self.calculate_coverage(),
            'missing_patterns': self.analyze_missing(),
            'outliers': self.detect_outliers(),
            'cross_sectional': self.cross_sectional_stats()
        }
        
        # 生成 QC 报告
        self.qc_report = self.generate_qc_report()
    
    def save_results(self):
        """保存结果"""
        # 保存 Panel 数据
        self.panel_data.to_parquet('panel_data.parquet')
        
        # 保存可用性矩阵
        self.availability_matrix.to_parquet('availability_matrix.parquet')
        
        # 保存报告
        with open('data_ready_report.yaml', 'w') as f:
            yaml.dump({
                'leakage_check': self.leakage_report,
                'qc_results': self.qc_results,
                'panel_shape': self.panel_data.shape,
                'universe_size': len(self.panel_data.columns)
            }, f)
```

### Gate 决策示例

```yaml
DataReady Gate 决策:
  
  评审时间: "2026-04-02"
  评审结论: "PASS"
  
  通过项:
    ✅ FG-1: "Panel 结构完整 (1461 × 42)"
    ✅ FG-2: "Universe 过滤正确，冻结初始 42 个币"
    ✅ FG-3: "时间对齐到 UTC 小时，无重复"
    ✅ FG-4: "Leakage 检查通过，无前视偏差"
    ✅ FG-5: "覆盖率 95.2%，质量达标"
  
  冻结内容:
    - "Panel 数据结构"
    - "Universe 成分（42 个币）"
    - "时间对齐规范（UTC 小时）"
    - "可用性矩阵"
  
  下一步: "进入 PanelReady 阶段"
```

---

## 输出 Artifact 规范

### 必需的输出文件

#### 1. panel_data.parquet（Panel 数据）

```yaml
描述: "横截面因子研究的 Panel 数据"
格式: "Parquet"
维度: "(时间 × 标的)"
内容: "价格、收益率等基础数据"
```

#### 2. availability_matrix.parquet（可用性矩阵）

```yaml
描述: "标记每个观测是否可用"
格式: "Parquet"
维度: "(时间 × 标的)"
值: "布尔值"
```

#### 3. data_ready_report.yaml（数据就绪报告）

```yaml
version: "1.0"
stage: "data_ready"
lineage_id: "momentum_crypto_xs_v1"
timestamp: "2026-04-02"

panel_structure:
  shape: [1461, 42]
  time_range: ["2021-01-01", "2024-12-31"]
  symbols: ["BTC", "ETH", "BNB", ...]

universe_filter:
  initial_size: 42
  excluded_stablecoins: 8
  excluded_leveraged: 5
  excluded_insufficient_history: 0

time_alignment:
  timezone: "UTC"
  frequency: "1h"
  alignment_method: "floor"

leakage_check:
  temporal_leakage: "NONE"
  data_leakage: "NONE"
  universe_leakage: "NONE"
  status: "PASS"

qc_results:
  coverage:
    overall: 0.952
    by_time: {min: 0.88, max: 0.98}
    by_symbol: {min: 0.85, max: 0.99}
  
  missing_patterns:
    consecutive_max: 12
    block_missing: 0
  
  outliers:
    price_spikes: 0.001
    extreme_returns: 0.005

gate_decision:
  status: "PASS"
  frozen_items:
    - "panel_data.parquet"
    - "availability_matrix.parquet"
    - "universe_manifest.csv"
  next_stage: "panel_ready"
```

---

## 与下一阶段的交接标准

### PanelReady 阶段的输入要求

| 项目 | 交付物 | 格式 | 用途 |
|------|--------|------|------|
| Panel 数据 | panel_data.parquet | Parquet | 因子计算基础 |
| 可用性矩阵 | availability_matrix.parquet | Parquet | 样本选择 |
| Universe 列表 | universe_manifest.csv | CSV | 标的过滤 |
| 时间配置 | time_alignment.yaml | YAML | 时间对齐 |

### 交接验收标准

```yaml
PanelReady 可以开始的条件:
  ✅ panel_data.parquet 存在且可读
  ✅ availability_matrix.parquet 存在且可读
  ✅ Panel 结构符合要求 (T × N)
  ✅ 覆盖率 > 90%
  ✅ Leakage 检查通过

验收方式:
  - 自动化脚本验证文件格式
  - Panel 结构检查
  - 数据质量报告审查
```

---

**文档版本**: v1.0  
**最后更新**: 2026-04-02  
**维护者**: 量化研究团队
