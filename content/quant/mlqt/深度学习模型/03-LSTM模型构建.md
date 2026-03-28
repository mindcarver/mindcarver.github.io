# 03-LSTM模型构建

> 本节详细介绍如何用 PyTorch 构建各种 LSTM 模型，包括单层 LSTM、多层 LSTM、双向 LSTM 和 LSTM+Attention。

## 单层 LSTM

### 结构和输入输出

```
单层 LSTM 结构：

输入 (batch, seq_len, input_size)
         │
         ▼
    ┌─────────┐
    │  LSTM   │
    │ (hidden) │
    └─────────┘
         │
         ├──→ output: (batch, seq_len, hidden_size)
         │         (每个时间步的隐藏状态)
         │
         └──→ h_n: (num_layers, batch, hidden_size)
                 (最后时间步的隐藏状态)

             c_n: (num_layers, batch, hidden_size)
                 (最后时间步的细胞状态)
```

### nn.LSTM 输入输出维度

```python
import torch
import torch.nn as nn

# 创建 LSTM
lstm = nn.LSTM(
    input_size=10,      # 输入特征维度
    hidden_size=20,     # 隐藏状态维度
    batch_first=True    # 输入形状 (batch, seq, feature)
)

# 输入
x = torch.randn(32, 50, 10)  # (batch=32, seq_len=50, input_size=10)

# 前向传播
output, (h_n, c_n) = lstm(x)

print(f"输入形状: {x.shape}")        # (32, 50, 10)
print(f"输出形状: {output.shape}")   # (32, 50, 20)
print(f"h_n 形状: {h_n.shape}")      # (1, 32, 20)  # num_layers=1
print(f"c_n 形状: {c_n.shape}")      # (1, 32, 20)
```

维度说明：

| 变量 | 形状 | 说明 |
|------|------|------|
| `input` | `(batch, seq_len, input_size)` | 输入序列 |
| `output` | `(batch, seq_len, hidden_size)` | 每个时间步的隐藏状态 |
| `h_n` | `(num_layers, batch, hidden_size)` | 最后时间步的隐藏状态 |
| `c_n` | `(num_layers, batch, hidden_size)` | 最后时间步的细胞状态 |

### 完整的单层 LSTM 预测模型

```python
import torch
import torch.nn as nn

class SingleLSTM(nn.Module):
    """单层 LSTM 预测模型"""
    def __init__(self, input_size, hidden_size, output_size=1, dropout=0.0):
        super(SingleLSTM, self).__init__()
        self.hidden_size = hidden_size

        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0
        )

        # 输出层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            output: (batch, output_size)
        """
        # LSTM 前向传播
        # lstm_out: (batch, seq_len, hidden_size)
        # h_n: (1, batch, hidden_size)
        # c_n: (1, batch, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]  # (batch, hidden_size)

        # 通过全连接层得到预测
        output = self.fc(last_output)  # (batch, output_size)

        return output

# 使用示例
model = SingleLSTM(input_size=10, hidden_size=32, output_size=1)
x = torch.randn(16, 20, 10)  # (batch=16, seq_len=20, input_size=10)
output = model(x)
print(f"输出形状: {output.shape}")  # (16, 1)

# 查看模型参数
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数: {total_params:,}")
```

模型结构：

```
SingleLSTM 数据流：

输入 (batch, seq_len, input_size=10)
         │
         ▼
    ┌─────────────────────┐
    │      LSTM           │
    │  hidden_size=32     │
    │  output: (batch,    │
    │    seq_len, 32)     │
    └─────────────────────┘
         │
         │ 取最后时间步 [:, -1, :]
         ▼
    ┌─────────────────────┐
    │      Linear         │
    │  32 → 1             │
    └─────────────────────┘
         │
         ▼
输出 (batch, 1)
```

## 多层 LSTM（Stacked LSTM）

### 结构

```
多层 LSTM 结构（num_layers=2）：

输入 (batch, seq_len, input_size)
         │
         ▼
    ┌─────────────────┐
    │   LSTM Layer 1  │
    │   (hidden=64)   │
    └────────┬────────┘
             │ (中间输出，可能 dropout)
             ▼
    ┌─────────────────┐
    │   LSTM Layer 2  │
    │   (hidden=32)   │
    └────────┬────────┘
             │
             ├──→ output: (batch, seq_len, 32)
             │
             └──→ h_n: (2, batch, 各层hidden_size)
                  c_n: (2, batch, 各层hidden_size)
```

### 实现代码

```python
import torch
import torch.nn as nn

class StackedLSTM(nn.Module):
    """多层 LSTM 预测模型"""
    def __init__(self, input_size, hidden_sizes, output_size=1, dropout=0.2):
        """
        Args:
            input_size: 输入特征维度
            hidden_sizes: 列表，每层 LSTM 的隐藏维度，如 [64, 32]
            output_size: 输出维度
            dropout: 层间 dropout 比例
        """
        super(StackedLSTM, self).__init__()
        self.hidden_sizes = hidden_sizes

        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_sizes[-1],
            num_layers=len(hidden_sizes),
            batch_first=True,
            dropout=dropout if len(hidden_sizes) > 1 else 0
        )

        # 输出层
        self.fc = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        # LSTM 前向传播
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 取最后一层的最后时间步
        last_output = lstm_out[:, -1, :]

        # 预测
        output = self.fc(last_output)

        return output

# 使用示例
model = StackedLSTM(
    input_size=10,
    hidden_sizes=[64, 32],  # 第一层 64 维，第二层 32 维
    output_size=1,
    dropout=0.2
)

x = torch.randn(16, 20, 10)
output = model(x)
print(f"输出形状: {output.shape}")  # (16, 1)

# 参数量对比
single_params = sum(p.numel() for p in SingleLSTM(10, 32, 1).parameters())
stacked_params = sum(p.numel() for p in model.parameters())
print(f"单层参数: {single_params:,}")
print(f"多层参数: {stacked_params:,}")
```

### 层间 Dropout

```python
import torch
import torch.nn as nn

# Dropout 在 LSTM 层间的作用
lstm_with_dropout = nn.LSTM(
    input_size=10,
    hidden_size=32,
    num_layers=3,
    dropout=0.3,  # 除最后一层外，每层后都应用 30% dropout
    batch_first=True
)

# 等价于手动实现
class ManualDropoutLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        # 第一层
        self.layers.append(nn.LSTM(input_size, hidden_size, batch_first=True))

        # 后续层
        for _ in range(num_layers - 1):
            self.layers.append(nn.LSTM(hidden_size, hidden_size, batch_first=True))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x, _ = layer(x)
            if i < len(self.layers) - 1:  # 最后一层不 dropout
                x = self.dropout(x)
        return x, None
```

## 双向 LSTM（Bidirectional LSTM）

### 原理

```
双向 LSTM 结构：

输入序列:  t=1    t=2    t=3    t=4    t=5
           │      │      │      │      │
           ▼      ▼      ▼      ▼      ▼
    ┌─────────────────────────────────────┐
    │         前向 LSTM (→)               │
    │    h₁ → h₂ → h₃ → h₄ → h₅          │
    └─────────────────────────────────────┘
    ┌─────────────────────────────────────┐
    │         后向 LSTM (←)               │
    │    h₁ ← h₂ ← h₃ ← h₄ ← h₅          │
    └─────────────────────────────────────┘
           │      │      │      │      │
           ▼      ▼      ▼      ▼      ▼
        拼接: [h₁→+←h₁] [h₂→+←h₂] ...
           (hidden_size * 2)
```

### 代码实现

```python
import torch
import torch.nn as nn

class BidirectionalLSTM(nn.Module):
    """双向 LSTM 预测模型"""
    def __init__(self, input_size, hidden_size, output_size=1, dropout=0.0):
        super(BidirectionalLSTM, self).__init__()
        self.hidden_size = hidden_size

        # 双向 LSTM
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # 输出层：输入是 hidden_size * 2（双向拼接）
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        # bilstm_out: (batch, seq_len, hidden_size * 2)
        # h_n: (2, batch, hidden_size)  # 2 = 前向+后向
        # c_n: (2, batch, hidden_size)
        bilstm_out, (h_n, c_n) = self.bilstm(x)

        # 取最后时间步
        last_output = bilstm_out[:, -1, :]  # (batch, hidden_size * 2)

        # 预测
        output = self.fc(last_output)

        return output

# 使用示例
model = BidirectionalLSTM(input_size=10, hidden_size=32, output_size=1)
x = torch.randn(16, 20, 10)
output = model(x)
print(f"输出形状: {output.shape}")  # (16, 1)
```

### ⚠️ 双向 LSTM 在量化中的陷阱

**问题**：双向 LSTM 会"看见未来"，导致回测虚高！

```
双向 LSTM 的数据泄漏问题：

训练时:  使用 t=1~20 预测 t=21
         双向 LSTM 在 t=1 时刻能看到 t=20 的信息！

         正常:  t=1 → t=2 → ... → t=20 → t=21
         双向:  t=1 ←→ t=2 ←→ ... ←→ t=20 (全部可见)

实盘时:  只有 t=1~20 的历史，无法使用双向
         训练和实际条件不一致！
```

**结论**：
- ✅ 适合：分类任务（如情感分析）、序列标注
- ❌ 不适合：时序预测（会引入未来信息）

## LSTM + Attention

### 注意力机制原理

```
Attention 权重计算：

LSTM 输出序列:
    h₁    h₂    h₃    h₄    h₅
    │     │     │     │     │
    ▼     ▼     ▼     ▼     ▼
   [ ]   [ ]   [ ]   [ ]   [ ]   ← context vectors
    │     │     │     │     │
    └─────┴─────┴─────┴─────┘
            │
            ▼
    [Attention Layer]
            │
    ────────┴────────
    权重:  0.1   0.1   0.3   0.4   0.1
                 ↑        ↑
            重点关注这些时间步

            │
            ▼
    加权求和: 0.1*h₁ + 0.1*h₂ + 0.3*h₃ + 0.4*h₄ + 0.1*h₅
            │
            ▼
        预测输出
```

### Bahdanau Attention 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    """Bahdanau Attention (Additive Attention)"""
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.hidden_size = hidden_size

        # Attention 权重计算层
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        """
        Args:
            hidden: (batch, hidden_size) 最后时间步的隐藏状态
            encoder_outputs: (batch, seq_len, hidden_size) 所有时间步的输出
        Returns:
            context: (batch, hidden_size) 上下文向量
            attn_weights: (batch, seq_len) 注意力权重
        """
        batch_size, seq_len, hidden_size = encoder_outputs.size()

        # 扩展 hidden 以匹配 seq_len
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, hidden_size)

        # 计算注意力能量
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
        # energy: (batch, seq_len, hidden_size)

        attention = self.v(energy).squeeze(2)  # (batch, seq_len)

        # 计算 softmax 权重
        attn_weights = F.softmax(attention, dim=1)  # (batch, seq_len)

        # 加权求和得到上下文向量
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        # context: (batch, 1, hidden_size)
        context = context.squeeze(1)  # (batch, hidden_size)

        return context, attn_weights

# 使用示例
attention = BahdanauAttention(hidden_size=32)
hidden = torch.randn(16, 32)  # (batch, hidden_size)
encoder_outputs = torch.randn(16, 20, 32)  # (batch, seq_len, hidden_size)

context, attn_weights = attention(hidden, encoder_outputs)
print(f"上下文向量形状: {context.shape}")  # (16, 32)
print(f"注意力权重形状: {attn_weights.shape}")  # (16, 20)

# 可视化注意力权重（第一个样本，前10个时间步）
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 3))
plt.bar(range(10), attn_weights[0, :10].detach().numpy())
plt.xlabel('Time Step')
plt.ylabel('Attention Weight')
plt.title('Attention Weights Visualization')
plt.show()
```

### LSTM + Attention 完整模型

```python
import torch
import torch.nn as nn

class LSTMWithAttention(nn.Module):
    """LSTM + Attention 预测模型"""
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=1, dropout=0.0):
        super(LSTMWithAttention, self).__init__()
        self.hidden_size = hidden_size

        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention 层
        self.attention = BahdanauAttention(hidden_size)

        # 输出层
        self.fc = nn.Linear(hidden_size * 2, output_size)  # LSTM输出 + Attention上下文

    def forward(self, x):
        # LSTM 前向传播
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (batch, seq_len, hidden_size)

        # 取最后时间步的隐藏状态
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)

        # 计算 Attention
        context, attn_weights = self.attention(last_hidden, lstm_out)
        # context: (batch, hidden_size)

        # 拼接最后隐藏状态和上下文向量
        combined = torch.cat([last_hidden, context], dim=1)  # (batch, hidden_size * 2)

        # 预测
        output = self.fc(combined)

        return output, attn_weights

# 使用示例
model = LSTMWithAttention(input_size=10, hidden_size=32, output_size=1, num_layers=2, dropout=0.2)
x = torch.randn(16, 20, 10)
output, attn_weights = model(x)
print(f"输出形状: {output.shape}")  # (16, 1)
print(f"注意力权重形状: {attn_weights.shape}")  # (16, 20)
```

## LSTM 变体对比

| 模型 | 特点 | 参数量 | 适用场景 | 量化推荐度 |
|------|------|--------|----------|------------|
| **单层 LSTM** | 简单高效 | 少 | 简单时序模式 | ⭐⭐⭐⭐ |
| **多层 LSTM** | 更强表达能力 | 中 | 复杂模式 | ⭐⭐⭐⭐⭐ |
| **双向 LSTM** | 双向上下文 | 中×2 | 分类、标注 | ⭐⭐（预测慎用） |
| **LSTM+Attention** | 关注重要时间步 | 中+ | 长序列、重要点识别 | ⭐⭐⭐⭐⭐ |
| **GRU** | 简化 LSTM | 少 | 资源受限 | ⭐⭐⭐ |

## 超参数选择指南

### 常用超参数范围

| 超参数 | 常用范围 | 推荐起点 | 说明 |
|--------|----------|----------|------|
| `hidden_size` | 32-512 | 64-128 | 影响模型容量 |
| `num_layers` | 1-4 | 2 | 过深容易过拟合 |
| `dropout` | 0.0-0.5 | 0.1-0.3 | 层间 dropout |
| `learning_rate` | 1e-4 - 1e-3 | 1e-3 | Adam 优化器 |
| `batch_size` | 16-256 | 32-64 | 取决于数据大小 |
| `seq_len` | 10-60 | 20 | 预测用的历史长度 |
| `horizon` | 1-20 | 1 | 预测未来多少步 |

### 超参数搜索策略

```python
# 常用的超参数组合
hyperparameter_configs = [
    # 小模型快速实验
    {'hidden_size': 32, 'num_layers': 1, 'dropout': 0.1, 'lr': 1e-3},

    # 中等模型
    {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.2, 'lr': 1e-3},

    # 大模型（需要更多数据）
    {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.3, 'lr': 5e-4},

    # 深层模型（谨慎使用）
    {'hidden_size': 64, 'num_layers': 3, 'dropout': 0.3, 'lr': 5e-4},
]

# Grid Search 示例
for config in hyperparameter_configs:
    model = StackedLSTM(
        input_size=10,
        hidden_sizes=[config['hidden_size']] * config['num_layers'],
        dropout=config['dropout']
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    # 训练和评估...
```

## 模型构建最佳实践

### 1. 输入维度检查

```python
def check_input_dimensions(model, input_size, seq_len, batch_size):
    """检查模型输入输出维度"""
    x = torch.randn(batch_size, seq_len, input_size)
    model.eval()
    with torch.no_grad():
        output = model(x)
    print(f"输入: {x.shape} → 输出: {output.shape}")
    return output.shape

# 使用
check_input_dimensions(model, input_size=10, seq_len=20, batch_size=32)
```

### 2. 模型摘要

```python
def model_summary(model, input_size, seq_len):
    """打印模型摘要"""
    x = torch.randn(1, seq_len, input_size)

    print("=" * 50)
    print(f"模型: {model.__class__.__name__}")
    print("=" * 50)

    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
        print(f"{name:30s} {list(param.shape)}")

    print("=" * 50)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"不可训练参数: {total_params - trainable_params:,}")
    print("=" * 50)

    # 测试前向传播
    model.eval()
    with torch.no_grad():
        output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print("=" * 50)

# 使用
model_summary(model, input_size=10, seq_len=20)
```

### 3. 模型保存和加载

```python
# 保存模型
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss,
}, 'lstm_model.pth')

# 加载模型
checkpoint = torch.load('lstm_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

## 核心知识点总结

### 1. LSTM 输入输出
```python
# 输入: (batch, seq_len, input_size)
# 输出: (batch, seq_len, hidden_size)
# h_n: (num_layers, batch, hidden_size)
# c_n: (num_layers, batch, hidden_size)
```

### 2. 多层 LSTM
```python
nn.LSTM(
    input_size=10,
    hidden_size=32,
    num_layers=2,      # 层数
    dropout=0.2        # 层间 dropout
)
```

### 3. 双向 LSTM（慎用！）
```python
# 输出维度是 hidden_size * 2
# 量化预测中慎用，会引入未来信息
nn.LSTM(..., bidirectional=True)
```

### 4. LSTM + Attention
```python
# Bahdanau Attention
# 1. 计算注意力能量
# 2. Softmax 得到权重
# 3. 加权求和得到上下文
# 4. 拼接上下文和最后隐藏状态
```

## 练习建议

1. **对比不同架构**：用相同数据训练单层、多层、LSTM+Attention
2. **可视化注意力**：观察模型关注哪些时间步
3. **调试维度**：手动计算各层输出维度，验证理解

## 下一节

在 **[04-时序数据处理.md](./04-时序数据处理.md)** 中，我们将学习如何处理量化时序数据。
