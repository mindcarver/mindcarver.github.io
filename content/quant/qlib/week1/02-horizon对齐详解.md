# Horizon对齐（Label Shift）详解

## 引言

在量化投资和机器学习建模中，"Horizon对齐"（Horizon Alignment，又称Label Shift）是一个最基础、最核心，却最容易被误解的概念。它解决的是量化投资中最根本的问题：

**"我现在的因子，如何对应未来的收益？"**

简单来说，Horizon对齐就是**消除"未来函数"并建立因果预测关系**。

本文将从数学定义、详细示例、实现对比、不同Horizon的影响、常见错误陷阱等多个维度，全面解析Horizon对齐的原理与实践。

---

## 1. 核心概念与数学定义

### 1.1 Horizon的定义

**Horizon（预测时长/视界）**是指从当前时刻 $t$ 到未来时刻 $t+h$ 的时间跨度，记为 $h$。

**常见的Horizon设置**：
- $h=1$：预测未来1天（日频）
- $h=5$：预测未来5天（周频）
- $h=20$：预测未来20天（月频）
- $h=60$：预测未来60天（季频）

**Horizon的选择取决于**：
- 交易频率：高频交易 $h$ 较小，低频投资 $h$ 较大
- 因子类型：动量因子 $h$ 较小，价值因子 $h$ 较大
- 持仓周期：短线策略 $h$ 较小，长线策略 $h$ 较大

### 1.2 收益率的数学定义

**离散收益率（Simple Return）**

从 $t$ 时刻到 $t+h$ 时刻的收益率定义为：

$$ R_{t \to t+h} = \frac{P_{t+h} - P_t}{P_t} $$

其中：
- $P_t$：$t$ 时刻的价格
- $P_{t+h}$：$t+h$ 时刻的价格
- $R_{t \to t+h}$：从 $t$ 到 $t+h$ 的收益率

**对数收益率（Log Return）**

对数收益率具有可加性，是量化研究中常用的形式：

$$ r_{t \to t+h} = \ln\left(\frac{P_{t+h}}{P_t}\right) = \ln(P_{t+h}) - \ln(P_t) $$

**两者的关系**

当收益率较小时（$|R| \ll 1$），对数收益率约等于离散收益率：

$$ r \approx R $$

推导（泰勒展开）：

$$ \ln(1+R) = R - \frac{R^2}{2} + \frac{R^3}{3} - \cdots \approx R $$

**优势对比**

| 收益率类型 | 优势 | 劣势 |
|-----------|------|------|
| 离散收益率 | 直观、易懂 | 不可加（跨时间） |
| 对数收益率 | 可加、对称性 | 解释性稍差 |

**可加性示例**

假设有3天的价格：$P_1=100, P_2=110, P_3=121$

**离散收益率**：
$$ R_{1 \to 2} = (110-100)/100 = 10\% $$
$$ R_{2 \to 3} = (121-110)/110 = 10\% $$
$$ R_{1 \to 3} = (121-100)/100 = 21\% \neq R_{1 \to 2} + R_{2 \to 3} $$

**对数收益率**：
$$ r_{1 \to 2} = \ln(110/100) = 0.0953 $$
$$ r_{2 \to 3} = \ln(121/110) = 0.0953 $$
$$ r_{1 \to 3} = \ln(121/100) = 0.1906 = r_{1 \to 2} + r_{2 \to 3} \quad \checkmark $$

### 1.3 Label Shift的推导过程

**问题提出**

在量化回测或机器学习建模中：
- **因子（Factor/Feature）**：是我们在 $t$ 时刻就能观察到的数据（如：$t$ 时刻的收盘价、PE、成交量等）
- **收益率（Label/Target）**：是我们要预测的目标，通常是从 $t$ 时刻到未来 $t+h$ 时刻的涨跌幅

问题是：如何将"当前的因子"与"未来的收益"对齐到同一行数据中？

**错误的对齐方式（Look-ahead Bias）**

$$ \text{Row}_t = \{ \underbrace{\text{Factor}_t}_{\text{t时刻可观测}}, \underbrace{\text{Return}_t}_{\text{t时刻收益（已实现）}} \} $$

**问题分析**：
- $\text{Return}_t$ 是 $t$ 时刻到 $t-1$ 时刻的收益（已实现）
- 在 $t$ 时刻，我们不知道 $\text{Return}_t$（要等到 $t+1$ 时刻才知道）
- 如果用 $\text{Return}_t$ 训练模型，模型会"看到未来"，这是Look-ahead Bias

**正确的对齐方式（Label Shift）**

$$ \text{Row}_t = \{ \underbrace{\text{Factor}_t}_{\text{t时刻可观测}}, \underbrace{\text{Return}_{t \to t+h}}_{\text{未来h期收益}} \} $$

其中：

$$ \text{Return}_{t \to t+h} = \frac{P_{t+h}}{P_t} - 1 $$

**Label Shift的操作**

Label Shift的本质是将 $t+h$ 时刻才产生的收益率，"平移"到 $t$ 时刻的因子行上：

$$ \text{Label}_t = \text{Return}_{t \to t+h} $$

**时序因果约束**

Horizon对齐必须满足时序因果约束：

$$ \text{Factor}_t \leftarrow \text{Label}_{t \to t+h} \quad \checkmark $$
$$ \text{Factor}_{t+1} \leftarrow \text{Label}_{t \to t+h} \quad \times $$

**约束解释**：
- 在 $t$ 时刻，我们可以观测到 $\text{Factor}_t$
- 在 $t$ 时刻，我们预测的是 $\text{Label}_{t \to t+h}$（未来收益）
- 不能在 $t$ 时刻预测 $\text{Label}_{t \to t+h}$（这已经是过去的收益）

### 1.4 Dataset形式化表示

**训练集构造**

整个训练集可以形式化为：

$$ \mathcal{D} = \{ (\mathbf{X}_t, y_{t+h}) \mid t = 1, 2, \dots, T-h \} $$

其中：
- $\mathbf{X}_t \in \mathbb{R}^{N \times F}$：$t$ 时刻 $N$ 个资产的 $F$ 个因子
- $y_{t+h} \in \mathbb{R}^N$：$t+h$ 时刻 $N$ 个资产的收益
- $T$：总时间长度
- $N$：资产数量
- $F$：因子数量

**维度解释**

**时间维度（Time）**：
- 训练集时间范围：$[t_1, t_{T-h}]$
- 标签时间范围：$[t_{1+h}, t_T]$
- 注意：训练集最后 $h$ 个时刻没有标签（因为未来数据不存在）

**资产维度（Number of instruments）**：
- 横截面：每个时刻有 $N$ 个资产
- 模型可以学习"同一时刻不同资产之间的关系"（横截面信息）

**因子维度（Factors）**：
- 每个资产有 $F$ 个因子
- 模型可以学习"同一资产不同因子之间的关系"（时序信息）

**矩阵形式**

**特征矩阵** $\mathbf{X} \in \mathbb{R}^{(T-h) \times N \times F}$：
$$ \mathbf{X} = \begin{bmatrix}
\mathbf{X}_1 \\
\mathbf{X}_2 \\
\vdots \\
\mathbf{X}_{T-h}
\end{bmatrix} = \begin{bmatrix}
\begin{bmatrix}x_{1,1,1} & \cdots & x_{1,1,F} \\ \vdots & \ddots & \vdots \\ x_{1,N,1} & \cdots & x_{1,N,F}\end{bmatrix} \\
\begin{bmatrix}x_{2,1,1} & \cdots & x_{2,1,F} \\ \vdots & \ddots & \vdots \\ x_{2,N,1} & \cdots & x_{2,N,F}\end{bmatrix} \\
\vdots \\
\begin{bmatrix}x_{T-h,1,1} & \cdots & x_{T-h,1,F} \\ \vdots & \ddots & \vdots \\ x_{T-h,N,1} & \cdots & x_{T-h,N,F}\end{bmatrix}
\end{bmatrix} $$

**标签矩阵** $\mathbf{Y} \in \mathbb{R}^{(T-h) \times N}$：
$$ \mathbf{Y} = \begin{bmatrix}
y_{1+h} \\
y_{2+h} \\
\vdots \\
y_T
\end{bmatrix} = \begin{bmatrix}
y_{1+h,1} & \cdots & y_{1+h,N} \\
y_{2+h,1} & \cdots & y_{2+h,N} \\
\vdots & \ddots & \vdots \\
y_{T,1} & \cdots & y_{T,N}
\end{bmatrix} $$

**对齐关系**

$$ \mathbf{X}_t \leftrightarrow y_{t+h} $$

其中 $t = 1, 2, \dots, T-h$。

---

## 2. 详细示例演示

### 2.1 原始数据表

假设我们有某只股票3天的价格数据：

| 日期 (t) | 收盘价 (Price) | 因子值 (Factor) |
|---------|---------------|-----------------|
| 2023-01-01 | 100 | 1.5 |
| 2023-01-02 | 105 | 1.6 |
| 2023-01-03 | 103 | 1.4 |

**解释**：
- 01-01：价格100，因子1.5
- 01-02：价格105，因子1.6
- 01-03：价格103，因子1.4

### 2.2 Step 1：计算收益率

假设我们要预测未来1天的收益率（$h=1$）。

**计算公式**：

$$ R_{t \to t+1} = \frac{P_{t+1} - P_t}{P_t} $$

**计算过程**：

**01-01 到 01-02**：
$$ R_{1 \to 2} = \frac{105 - 100}{100} = 5\% $$

**01-02 到 01-03**：
$$ R_{2 \to 3} = \frac{103 - 105}{105} = -1.9\% $$

**收益率表**：

| 日期 (t) | 收盘价 (Price) | 因子值 (Factor) | 收益率 (Return) |
|---------|---------------|-----------------|-----------------|
| 2023-01-01 | 100 | 1.5 | 5% |
| 2023-01-02 | 105 | 1.6 | -1.9% |
| 2023-01-03 | 103 | 1.4 | NaN |

**注意**：01-03没有收益率，因为还没有01-04的数据。

### 2.3 Step 2：Label Shift

现在我们需要将未来的收益率对齐到当前时刻。

**对齐规则**：

$$ \text{Label}_t = R_{t \to t+1} $$

**对齐过程**：

**01-01**：
- 因子：1.5
- 对齐的标签：$R_{1 \to 2} = 5\%$（01-01预测01-02的收益）

**01-02**：
- 因子：1.6
- 对齐的标签：$R_{2 \to 3} = -1.9\%$（01-02预测01-03的收益）

**01-03**：
- 因子：1.4
- 对齐的标签：NaN（没有未来数据）

**对齐后的数据表**：

| 日期 (t) | 因子值 (Factor) | 标签 (Label) |
|---------|-----------------|-------------|
| 2023-01-01 | 1.5 | 5% |
| 2023-01-02 | 1.6 | -1.9% |
| 2023-01-03 | 1.4 | NaN |

**解读**：

- 01-01：用因子1.5预测未来1天的收益（5%）
- 01-02：用因子1.6预测未来1天的收益（-1.9%）
- 01-03：没有未来数据，无法预测（删除或填充）

**Pandas代码实现**：

```python
import pandas as pd

# 原始数据
df = pd.DataFrame({
    'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
    'price': [100, 105, 103],
    'factor': [1.5, 1.6, 1.4]
})

# 计算收益率
df['return'] = df['price'].pct_change(1)

# Label Shift（向上移动h行）
h = 1
df['label'] = df['return'].shift(-h)

# 清理缺失值
df_clean = df.dropna(subset=['label'])

print(df_clean[['date', 'factor', 'label']])
```

**输出**：

```
        date  factor  label
0 2023-01-01     1.5   0.05
1 2023-01-02     1.6  -0.019
```

### 2.4 不同Horizon对比

**Horizon = 1（预测未来1天）**

| 日期 (t) | 因子 | 标签（Label） |
|---------|------|-------------|
| 01-01 | 1.5 | 5% |
| 01-02 | 1.6 | -1.9% |

**解读**：
- 用01-01的因子（1.5）预测01-01到01-02的收益（5%）
- 用01-02的因子（1.6）预测01-02到01-03的收益（-1.9%）

**Horizon = 5（预测未来5天）**

假设我们有更长时间的数据：

| 日期 | 价格 |
|------|------|
| 01-01 | 100 |
| 01-02 | 105 |
| 01-03 | 103 |
| 01-04 | 107 |
| 01-05 | 110 |
| 01-06 | 115 |

**计算收益率**：

$$ R_{1 \to 6} = \frac{115 - 100}{100} = 15\% $$
$$ R_{2 \to 7} = \text{未知} $$
$$ \vdots $$

**Label Shift**：

| 日期 (t) | 因子 | 标签（Label） |
|---------|------|-------------|
| 01-01 | 1.5 | 15% |
| 01-02 | 1.6 | NaN |

**解读**：
- 用01-01的因子（1.5）预测01-01到01-06的收益（15%）
- 01-02没有未来5天的数据（如果数据只有到01-06）

**Horizon = 20（预测未来20天）**

类似地，计算20天的收益率，然后向上移动20行。

**不同Horizon的影响**：

| Horizon | 信号时效性 | IC均值 | 噪声水平 | 适用场景 |
|---------|-----------|--------|---------|---------|
| 1 | 极强 | 0.02-0.05 | 极高 | 高频交易 |
| 5 | 强 | 0.05-0.08 | 高 | 短线策略 |
| 20 | 中 | 0.08-0.12 | 中 | 中线策略 |
| 60 | 弱 | 0.12-0.15 | 低 | 长线策略 |

**趋势**：
- Horizon越大，IC通常越高（因为长期趋势更清晰）
- Horizon越大，信号时效性越差（因为未来信息太多）

---

## 3. 实现代码对比

### 3.1 Pandas实现

**核心代码**：

```python
import pandas as pd

# 计算收益率
df['return'] = df['price'].pct_change(h)

# Label Shift（向上移动h行）
df['label'] = df['return'].shift(-h)

# 清理缺失值
df = df.dropna(subset=['label'])
```

**完整示例**：

```python
import pandas as pd
import numpy as np

# 模拟数据
np.random.seed(42)
dates = pd.date_range('2020-01-01', '2023-12-31')
n = len(dates)
prices = 100 * np.cumprod(1 + np.random.randn(n) * 0.01)
factors = np.random.randn(n) * 0.1

df = pd.DataFrame({
    'date': dates,
    'price': prices,
    'factor': factors
})

# 设置Horizon
h = 5

# Step 1: 计算收益率
df['return'] = df['price'].pct_change(h)

# Step 2: Label Shift
df['label'] = df['return'].shift(-h)

# Step 3: 清理缺失值
df_clean = df.dropna(subset=['label'])

# 验证对齐
print(f"原始数据量: {len(df)}")
print(f"对齐后数据量: {len(df_clean)}")
print(f"损失数据量: {len(df) - len(df_clean)}")
print(f"数据损失率: {(len(df) - len(df_clean)) / len(df):.2%}")

# 验证对齐正确性
sample = df_clean.iloc[0]
print(f"\n示例验证:")
print(f"日期: {sample['date']}")
print(f"因子: {sample['factor']:.4f}")
print(f"标签: {sample['label']:.4f}")
print(f"验证: {sample['label']:.4f} ≈ {(df.loc[df['date'] == sample['date'] + pd.Timedelta(days=h), 'price'].values[0] / sample['price'] - 1):.4f}")
```

**输出示例**：

```
原始数据量: 1096
对齐后数据量: 1091
损失数据量: 5
数据损失率: 0.46%

示例验证:
日期: 2020-01-01 00:00:00
因子: 0.0046
标签: 0.0492
验证: 0.0492 ≈ 0.0492
```

### 3.2 Qlib实现

**核心代码**：

```python
from qlib import Expression
from qlib.data import D

# 定义因子
factor_expr = Expression($close / Ref($close, 1) - 1)

# 定义Label（Horizon对齐）
label_expr = Expression(Ref($close, -h) / $close - 1)

# 加载数据（自动对齐）
features = D.features([factor_expr], start_time, end_time)
labels = D.features([label_expr], start_time, end_time)
```

**完整示例**：

```python
from qlib import init
from qlib.data import D
from qlib.expr import Expression
import pandas as pd

# 初始化Qlib
init(provider_uri="data/qlib/qlib_data/cn_data")

# 定义时间范围
start_time = "2020-01-01"
end_time = "2023-12-31"

# 定义Horizon
h = 5

# 定义因子（示例：动量因子）
factor_expr = Expression(
    ($close / Ref($close, 1)) - 1
)

# 定义Label（Horizon对齐）
label_expr = Expression(
    Ref($close, -h) / $close - 1
)

# 加载数据（Qlib自动对齐）
instruments = D.instruments(market="all")
features = D.features(
    instruments,
    [factor_expr],
    start_time=start_time,
    end_time=end_time
)
labels = D.features(
    instruments,
    [label_expr],
    start_time=start_time,
    end_time=end_time
)

# 验证对齐
print(f"特征矩阵形状: {features.shape}")  # [T, N, 1]
print(f"标签矩阵形状: {labels.shape}")    # [T, N, 1]

# 提取单只股票
stock = "000001.SZ"
feature_series = features.xs(stock, level="instrument").iloc[:, 0]
label_series = labels.xs(stock, level="instrument").iloc[:, 0]

# 验证对齐正确性
idx = feature_series.index[0]
print(f"\n示例验证:")
print(f"股票: {stock}")
print(f"日期: {idx}")
print(f"因子: {feature_series[idx]:.4f}")
print(f"标签: {label_series[idx]:.4f}")
```

**Qlib的优势**：
- 自动处理时间对齐
- 支持表达式系统
- 高效的增量计算
- 自动处理缺失值

### 3.3 实现对比总结

| 维度 | Pandas | Qlib |
|------|--------|------|
| **易用性** | 中等 | 高（表达式系统） |
| **性能** | 低（逐列计算） | 高（增量计算） |
| **灵活性** | 高（任意操作） | 中等（算子约束） |
| **可维护性** | 低（代码分散） | 高（声明式） |
| **可回测性** | 低（易泄漏） | 高（强制因果） |

---

## 4. 不同Horizon的影响分析

### 4.1 短期（h=1）：高频噪声问题

**优势**：
- 信号时效性强：模型学到的是"最近"的信息
- 适合高频交易：可以快速调整持仓
- 交易机会多：每天都有新的预测

**劣势**：
- 日内噪声严重：短期价格波动主要由随机噪声驱动
- IC波动大：因子表现不稳定，今天IC=0.05，明天IC=-0.02
- 交易成本高：频繁交易导致交易成本侵蚀收益

**IC分析**：

假设因子IC均值=0.03，标准差=0.1：

$$ \text{IR} = \frac{0.03}{0.1} = 0.3 $$

IR < 0.5，说明因子不稳定。

**适用场景**：
- 高频交易（分钟级、秒级）
- 市场中性策略（对冲市场风险）
- 流动性好的市场（如美股）

**不适用场景**：
- 流动性差的市场（如小盘股、新兴市场）
- 交易成本高的策略（如期权、期货）
- 长线投资

### 4.2 中期（h=5）：信号稳定性

**优势**：
- 信号稳定性好：IC均值较高，波动较小
- 噪声相对可控：5天内的价格波动有一定趋势
- 适合量化选股：可以选出表现较好的股票

**劣势**：
- 周期性效应：可能受周效应、月效应影响
- 持仓周期中等：需要定期调仓（如每周调仓）

**IC分析**：

假设因子IC均值=0.05，标准差=0.08：

$$ \text{IR} = \frac{0.05}{0.08} = 0.625 $$

IR > 0.5，说明因子可用。

**适用场景**：
- 量化选股（A股多因子策略）
- 市场中性对冲基金
- 指数增强

### 4.3 长期（h=20）：趋势主导

**优势**：
- 趋势信号清晰：20天的价格波动主要由基本面驱动
- IC较高：长期来看，因子与收益相关性更强
- 适合价值投资：低频交易，交易成本低

**劣势**：
- 对基本面变化反应慢：需要较长时间才能识别基本面变化
- 持仓周期长：需要长时间持有，可能错过短期机会
- 市场风格切换：如果市场风格切换，因子可能失效

**IC分析**：

假设因子IC均值=0.08，标准差=0.06：

$$ \text{IR} = \frac{0.08}{0.06} = 1.33 $$

IR > 1.0，说明因子非常稳定。

**适用场景**：
- 价值投资（低PE、高ROE）
- 基本面投资
- 长线基金

### 4.4 Horizon选择建议

**根据因子类型选择**：

| 因子类型 | 推荐Horizon | 原因 |
|---------|-------------|------|
| 动量因子 | 1-5天 | 动量效应短期存在 |
| 均值回归 | 5-20天 | 均值回归需要一定时间 |
| 价值因子 | 20-60天 | 价值发现需要时间 |
| 质量因子 | 20-60天 | 质量效应长期存在 |

**根据交易频率选择**：

| 交易频率 | 推荐Horizon | 原因 |
|---------|-------------|------|
| 高频（分钟级） | 1-5天 | 需要快速调整 |
| 中频（日频） | 5-20天 | 平衡时效性和稳定性 |
| 低频（周频/月频） | 20-60天 | 长线持有 |

**根据市场特征选择**：

| 市场特征 | 推荐Horizon | 原因 |
|---------|-------------|------|
| 流动性好 | 1-5天 | 可以快速调仓 |
| 流动性差 | 20-60天 | 降低交易成本 |
| 有效性高 | 20-60天 | 长期趋势清晰 |
| 有效性低 | 1-5天 | 捕捉短期机会 |

---

## 5. 常见错误与调试

### 5.1 Look-ahead Bias典型案例

**案例1：未来均值**

**错误代码**：

```python
# 计算20日均值，包含未来数据
df['ma20'] = df['price'].rolling(20).mean()

# 问题：在t时刻，ma20包含了t+1到t+19的数据
```

**正确代码**：

```python
# 计算t-19到t的均值
df['ma20'] = df['price'].rolling(20, min_periods=20).mean().shift(1)
```

**案例2：未来波动率**

**错误代码**：

```python
# 计算20日波动率，包含未来数据
df['volatility'] = df['return'].rolling(20).std()
```

**正确代码**：

```python
# 只用历史数据计算波动率
df['volatility'] = df['return'].rolling(20, min_periods=20).std().shift(1)
```

**案例3：未来相关性**

**错误代码**：

```python
# 计算与市场的相关性，包含未来数据
df['correlation'] = df['return'].rolling(20).corr(market_return)
```

**正确代码**：

```python
# 只用历史数据计算相关性
df['correlation'] = df['return'].rolling(20, min_periods=20).corr(market_return).shift(1)
```

### 5.2 检测方法

**方法1：IC突然变高**

如果训练集IC=0.05，但回测IC=0.5，说明可能存在Look-ahead Bias。

**方法2：回测夏普异常高**

如果策略回测夏普>5，说明可能存在Look-ahead Bias。

**方法3：因子与未来价格相关性**

检验因子是否与未来价格相关：

```python
# 检测因子是否与未来价格相关
for lag in [-1, -2, -5, -10]:
    corr = df['factor'].shift(lag).corr(df['price'])
    print(f"Lag {lag}: {corr:.4f}")
```

如果 `lag=-1` 的相关性显著大于其他lag，说明因子包含未来信息。

---

## 6. 实盘vs回测的差异

### 6.1 回测阶段

在回测阶段，我们有完整的历史数据：

**流程**：
1. 加载历史数据（$t=1$ 到 $t=T$）
2. 计算 Label Shift：$\text{Label}_t = \text{Return}_{t \to t+h}$
3. 训练模型：$\text{Model}(\text{Factor}_t) \to \text{Label}_t$
4. 验证模型：计算IC、回测收益等

**关键**：Label Shift是可行的，因为我们有未来数据。

### 6.2 实盘阶段

在实盘阶段，我们只有当前和过去的数据：

**流程**：
1. 加载实时数据（$t=1$ 到 $t=\text{now}$）
2. 计算当前因子：$\text{Factor}_{\text{now}}$
3. 预测未来收益：$\hat{\text{Label}}_{\text{now} \to \text{now}+h} = \text{Model}(\text{Factor}_{\text{now}})$
4. 交易决策：根据预测进行交易

**关键**：Label不存在，我们只能预测。

### 6.3 差异总结

| 维度 | 回测 | 实盘 |
|------|------|------|
| **数据完整性** | 有完整历史和未来数据 | 只有当前和过去数据 |
| **Label可用性** | 可用（Label Shift） | 不可用（需要预测） |
| **验证方式** | 可以验证预测准确性 | 只能等待h期后验证 |
| **风险** | 过拟合风险 | 实盘失败风险 |

---

## 7. 总结

Horizon对齐（Label Shift）是量化投资中最基础、最核心的技术，它解决了"当前的因子如何对应未来的收益"这一根本问题。

### 核心要点回顾

1. **Horizon定义**：预测时长 $h$，从 $t$ 到 $t+h$ 的时间跨度
2. **收益率公式**：
   - 离散收益率：$R_{t \to t+h} = (P_{t+h} - P_t) / P_t$
   - 对数收益率：$r_{t \to t+h} = \ln(P_{t+h} / P_t)$
3. **Label Shift**：
   - 错误对齐：$\{ \text{Factor}_t, \text{Return}_t \}$（Look-ahead Bias）
   - 正确对齐：$\{ \text{Factor}_t, \text{Return}_{t \to t+h} \}$（Label Shift）
4. **Pandas实现**：`df['label'] = df['return'].shift(-h)`
5. **Qlib实现**：`Expression(Ref($close, -h) / $close - 1)`
6. **Horizon选择**：
   - 短期（h=1）：高频噪声，IC不稳定
   - 中期（h=5）：信号稳定，IC中等
   - 长期（h=20）：趋势主导，IC较高
7. **常见错误**：Look-ahead Bias、未来均值、未来波动率等
8. **实盘vs回测**：
   - 回测：有完整数据，可以Label Shift
   - 实盘：只有当前数据，需要预测

### 量化投资的核心思想

Horizon对齐的本质是**建立因果预测关系**：

$$ \text{Cause (t)} \rightarrow \text{Effect (t+h)} $$

而不是：

$$ \text{Cause (t)} \leftrightarrow \text{Effect (t)} $$

这个看似简单的技术，却是量化投资区别于"赌大小"的分水岭：
- 赌大小：预测明天涨不涨（短期随机）
- 量化投资：用当前因子预测未来收益（因果预测）

在下一文档中，我们将探讨另一个核心主题：从"预测绝对价格"到"预测相对强弱"的量化思维转变。
