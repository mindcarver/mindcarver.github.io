# LSTM模型构建系列 - 架构与实现

## 📚 系列概述
本系列文档涵盖LSTM模型的各种架构、模型定义、超参数配置和变体。

---

## 📖 文档列表

1. [LSTM模型架构](#lstm模型架构)
2. [单层LSTM](#单层lstm)
3. [多层LSTM](#多层lstm)
4. [双向LSTM](#双向lstm)
5. [LSTM变体](#lstm变体)
6. [超参数选择](#超参数选择)

---

## LSTM模型架构

### 完整LSTM模型结构

```
输入 (batch, seq_len, input_size)
    ↓
LSTM层 (多层)
    ↓
Dropout层 (防止过拟合)
    ↓
全连接层
    ↓
输出 (batch, output_size)
```

### 参数说明

#### 输入参数
- `input_size`: 特征维度
- `hidden_size`: LSTM隐藏单元数
- `num_layers`: LSTM层数
- `dropout`: Dropout比例
- `output_size`: 输出维度

#### 模型参数

| 参数 | 说明 | 推荐值 | 影响 |
|------|------|--------|------|
| **input_size** | 输入特征维度 | 由数据决定 | 不可改变 |
| **hidden_size** | 隐藏单元数 | 32-128 | 模型容量 |
| **num_layers** | LSTM层数 | 1-3 | 模型深度 |
| **dropout** | Dropout比例 | 0.1-0.3 | 防止过拟合 |
| **batch_first** | batch是否在前 | True | 数据格式 |

---

## 单层LSTM

### 模型定义

```python
import torch.nn as nn

class SingleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        # Dropout层
        self.dropout = nn.Dropout(dropout)

        # 全连接层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        输入: (batch, seq_len, input_size)
        输出: (batch, output_size)
        """
        # 前向传播
        lstm_out, _ = self.lstm(x)

        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]

        # Dropout
        last_output = self.dropout(last_output)

        # 全连接层
        output = self.fc(last_output)

        return output
```

### 使用示例

```python
# 创建模型
model = SingleLSTM(
    input_size=10,
    hidden_size=64,
    output_size=1,
    dropout=0.2
)

# 输入数据
x = torch.randn(32, 20, 10)  # (batch=32, seq_len=20, input_size=10)

# 前向传播
output = model(x)

print(output.shape)  # torch.Size([32, 1])
```

### 适用场景
- 简单时序预测任务
- 数据量有限
- 快速原型开发

---

## 多层LSTM

### 模型定义

```python
class MultiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 多层LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 全连接层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        输入: (batch, seq_len, input_size)
        输出: (batch, output_size)
        """
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        output = self.fc(last_output)
        return output
```

### 使用示例

```python
# 创建模型
model = MultiLSTM(
    input_size=10,
    hidden_size=64,
    num_layers=3,
    dropout=0.2,
    output_size=1
)

# 输入数据
x = torch.randn(32, 20, 10)

# 前向传播
output = model(x)

print(output.shape)  # torch.Size([32, 1])
```

### 多层LSTM的特点

**优势**:
- 增强模型表达能力
- 学习更复杂的特征
- 提升模型性能

**劣势**:
- 参数量增加
- 训练时间增加
- 过拟合风险增加

### 适用场景
- 复杂时序模式
- 数据量充足
- 需要更强的表达能力

---

## 双向LSTM（Bi-LSTM）

### 模型定义

```python
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 双向LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,  # 双向
            dropout=dropout if num_layers > 1 else 0
        )

        # 双向输出维度是hidden_size * 2
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        """
        输入: (batch, seq_len, input_size)
        输出: (batch, output_size)
        """
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        output = self.fc(last_output)
        return output
```

### 使用示例

```python
# 创建模型
model = BiLSTM(
    input_size=10,
    hidden_size=64,
    num_layers=2,
    dropout=0.2,
    output_size=1
)

# 输入数据
x = torch.randn(32, 20, 10)

# 前向传播
output = model(x)

print(output.shape)  # torch.Size([32, 1])
```

### 双向LSTM的特点

**优势**:
- 同时利用过去和未来信息
- 适合需要上下文的任务
- 性能通常更好

**劣势**:
- 参数量增加一倍
- 不能用于实时预测（需要未来数据）
- 训练时间增加

### 适用场景
- 文本分类、情感分析
- 机器翻译
- 需要上下文信息的任务
- ❌ **不推荐**: 实时股票预测

---

## LSTM变体

### 1. 堆叠LSTM（Stacked LSTM）

#### 定义
- 多层LSTM堆叠
- 每层学习不同层次的抽象

#### 架构
```
输入 → LSTM层1 → LSTM层2 → ... → LSTM层N → 输出
```

#### 实现
```python
class StackedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output
```

### 2. 编码器-解码器LSTM

#### 定义
- 编码器：将序列编码为固定长度向量
- 解码器：从向量生成输出序列

#### 架构
```
输入序列 → 编码器LSTM → 上下文向量 → 解码器LSTM → 输出序列
```

#### 实现
```python
class EncoderDecoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        # 编码器
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)

        # 解码器
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        # 输出层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 编码
        _, (h_n, c_n) = self.encoder(x)

        # 解码
        decoder_input = h_n[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        decoder_output, _ = self.decoder(decoder_input, (h_n, c_n))

        # 输出
        output = self.fc(decoder_output)

        return output
```

### 3. 注意力LSTM（Attention LSTM）

#### 定义
- 添加注意力机制
- 动态关注重要时间步

#### 实现
```python
class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        # 注意力层
        self.attention = nn.Linear(hidden_size, 1)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM输出
        lstm_out, _ = self.lstm(x)

        # 计算注意力权重
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)

        # 加权求和
        context = torch.sum(attention_weights * lstm_out, dim=1)

        # 输出
        output = self.fc(context)

        return output
```

---

## 超参数选择

### 关键超参数

#### 1. hidden_size（隐藏单元数）

| 任务规模 | hidden_size | 说明 |
|---------|-------------|------|
| **小规模** | 32-64 | 简单任务，数据量小 |
| **中等规模** | 64-128 | 一般任务 |
| **大规模** | 128-256 | 复杂任务，数据量大 |

**选择原则**:
- 从小开始，逐步增加
- 监控过拟合
- 考虑计算资源

#### 2. num_layers（LSTM层数）

| 模式 | num_layers | 说明 |
|------|-----------|------|
| **简单** | 1 | 简单任务 |
| **中等** | 2-3 | 一般任务 |
| **复杂** | 3-5 | 复杂任务 |

**选择原则**:
- 不要过度堆叠
- 2-3层通常足够
- 超过5层收益递减

#### 3. dropout（Dropout比例）

| 场景 | dropout | 说明 |
|------|---------|------|
| **无过拟合** | 0.0-0.1 | 训练集表现好 |
| **轻微过拟合** | 0.1-0.3 | 轻微过拟合 |
| **严重过拟合** | 0.3-0.5 | 严重过拟合 |

**选择原则**:
- 从0.1开始
- 根据验证集调整
- 不要超过0.5

#### 4. learning_rate（学习率）

| 优化器 | learning_range | 说明 |
|--------|---------------|------|
| **Adam** | 0.0001-0.001 | 推荐默认值 |
| **SGD** | 0.01-0.1 | 需要momentum |
| **RMSprop** | 0.001-0.01 | RNN专用 |

**学习率调度**:
```python
# 初始学习率
learning_rate = 0.001

# 学习率衰减
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,  # 每10个epoch
    gamma=0.1      # 学习率乘0.1
)

# 余弦退火
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=50  # 总epoch数
)
```

#### 5. batch_size（批大小）

| 硬件 | batch_size | 说明 |
|------|-----------|------|
| **CPU** | 16-32 | 内存有限 |
| **单GPU** | 32-128 | GPU内存 |
| **多GPU** | 64-256 | 并行计算 |

**选择原则**:
- 2的幂次方（32, 64, 128）
- 根据GPU内存调整
- 越大越稳定，但越慢

#### 6. seq_len（序列长度）

| 预测目标 | seq_len | 说明 |
|---------|---------|------|
| **短期** | 5-10 | 日内交易 |
| **中期** | 20-60 | 几天到几周 |
| **长期** | 60-120 | 几个月 |

**选择原则**:
- 基于业务逻辑
- 通过实验确定
- 考虑计算成本

### 超参数搜索

#### 网格搜索

```python
from itertools import product

# 参数网格
param_grid = {
    'hidden_size': [32, 64, 128],
    'num_layers': [1, 2, 3],
    'dropout': [0.1, 0.2, 0.3],
    'learning_rate': [0.0001, 0.001, 0.01],
    'batch_size': [16, 32, 64]
}

# 生成所有组合
param_combinations = list(product(
    param_grid['hidden_size'],
    param_grid['num_layers'],
    param_grid['dropout'],
    param_grid['learning_rate'],
    param_grid['batch_size']
))
```

#### 随机搜索

```python
import random

# 随机搜索n次
n_trials = 20

for _ in range(n_trials):
    # 随机选择参数
    hidden_size = random.choice([32, 64, 128])
    num_layers = random.choice([1, 2, 3])
    dropout = random.uniform(0.1, 0.3)
    learning_rate = random.choice([0.0001, 0.001, 0.01])
    batch_size = random.choice([16, 32, 64])

    # 训练和评估
    # ...
```

#### 贝叶斯优化

```python
from skopt import gp_minimize

# 定义搜索空间
space = [
    (32, 256),           # hidden_size
    (1, 4),              # num_layers
    (0.1, 0.5),          # dropout
    (0.0001, 0.01, 'log'),  # learning_rate
    (16, 128)            # batch_size
]

# 定义目标函数
def objective(params):
    hidden_size, num_layers, dropout, learning_rate, batch_size = params

    # 训练模型
    model = MultiLSTM(
        input_size=10,
        hidden_size=hidden_size,
        num_layers=int(num_layers),
        dropout=dropout,
        output_size=1
    )

    # 返回验证损失
    return val_loss

# 优化
result = gp_minimize(objective, space, n_calls=50)
```

---

## 核心知识点总结

### LSTM模型架构
- ✅ 完整LSTM结构
- ✅ 参数说明
- ✅ 数据流

### 单层LSTM
- ✅ 模型定义
- ✅ 使用示例
- ✅ 适用场景

### 多层LSTM
- ✅ 模型定义
- ✅ 优劣势分析
- ✅ 适用场景

### 双向LSTM
- ✅ 模型定义
- ✅ 优劣势分析
- ✅ 适用场景

### LSTM变体
- ✅ 堆叠LSTM
- ✅ 编码器-解码器LSTM
- ✅ 注意力LSTM

### 超参数选择
- ✅ 关键超参数
- ✅ 推荐值
- ✅ 超参数搜索方法

---

## 下一步

继续学习: [时序数据处理系列](../04_时序数据处理系列/README.md)
