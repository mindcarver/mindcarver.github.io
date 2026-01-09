# PyTorch框架系列 - Tensor与模型构建

## 📚 系列概述
本系列文档涵盖PyTorch的核心概念、Tensor操作、自动微分、模型定义和常用层。

---

## 📖 文档列表

1. [PyTorch简介](#pytorch简介)
2. [Tensor基础](#tensor基础)
3. [Autograd自动微分](#autograd自动微分)
4. [nn.Module模型定义](#nnmodule模型定义)
5. [常用层](#常用层)

---

## PyTorch简介

### 什么是PyTorch
- 基于Python的深度学习框架
- Facebook（Meta）开发
- 学术研究首选框架

### 核心特点

#### 1. 动态计算图
- 运行时构建计算图
- 灵活性高
- 适合研究

#### 2. GPU加速
- 自动利用GPU
- 显著提升训练速度
- 支持CUDA

#### 3. 自动微分
- 自动计算梯度
- 简化反向传播
- 支持复杂计算图

#### 4. 丰富的API
- 预定义层和模型
- 优化器和损失函数
- 数据处理工具

### 为什么量化投资用PyTorch

**1. 灵活性强**
- 可以自定义复杂的模型结构
- 容易实现研究想法

**2. 社区活跃**
- 大量教程和示例
- 问题容易解决

**3. 易于部署**
- 支持导出为多种格式
- 生产环境友好

---

## Tensor基础

### Tensor定义
- PyTorch的核心数据结构
- 类似于NumPy数组
- 可以运行在GPU上

### 创建Tensor

```python
import torch

# 从Python列表创建
t1 = torch.tensor([1, 2, 3, 4])

# 从NumPy创建
import numpy as np
np_array = np.array([[1, 2], [3, 4]])
t2 = torch.from_numpy(np_array)

# 创建特殊Tensor
zeros = torch.zeros(2, 3)        # 全零
ones = torch.ones(2, 3)          # 全一
random = torch.randn(2, 3)       # 标准正态分布

# 创建序列
arange = torch.arange(0, 10)      # 0-9
linspace = torch.linspace(0, 10, 5)  # 0到10，5个点
```

### Tensor操作

#### 基本运算

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# 加法、减法、乘法、除法
c = a + b
d = a - b
e = a * b
f = a / b

# 点积
dot = torch.dot(a, b)

# 矩阵乘法
mat_a = torch.randn(2, 3)
mat_b = torch.randn(3, 4)
mat_c = torch.mm(mat_a, mat_b)  # 或 mat_a @ mat_b
```

#### 统计运算

```python
x = torch.randn(10)

# 均值、标准差、方差
mean = x.mean()
std = x.std()
var = x.var()

# 最大值、最小值
max_val = x.max()
min_val = x.min()

# 求和
sum_val = x.sum()
```

#### 形状操作

```python
x = torch.arange(12)

# reshape
x_reshaped = x.view(3, 4)  # 或 x.reshape(3, 4)

# 转置
x_transposed = x_reshaped.t()

# squeeze: 去除维度为1的
x_squeezed = torch.randn(1, 10, 1).squeeze()

# unsqueeze: 增加维度
x_unsqueezed = x.unsqueeze(0)
```

### Tensor vs NumPy

| 特性 | NumPy | Tensor |
|------|-------|--------|
| **GPU支持** | ❌ | ✅ |
| **自动微分** | ❌ | ✅ |
| **性能** | CPU | CPU/GPU |
| **API** | 类似 | 类似 |
| **互操作** | 易 | 易 |

#### 互操作

```python
# NumPy → Tensor
np_array = np.array([1, 2, 3])
tensor = torch.from_numpy(np_array)

# Tensor → NumPy
array = tensor.numpy()
```

---

## Autograd自动微分

### Autograd概述
- PyTorch的自动微分引擎
- 自动计算梯度
- 支持复杂的计算图

### 关键概念

#### 1. requires_grad
- 标记需要计算梯度的Tensor
- 默认为False
- 通常是模型参数

#### 2. backward()
- 反向传播，计算梯度
- 从loss开始
- 沿计算图传播梯度

#### 3. grad
- 存储梯度值
- 在backward()后填充
- 用于参数更新

### 简单示例

#### 单变量梯度

```python
import torch

# 创建需要梯度的Tensor
x = torch.tensor(2.0, requires_grad=True)

# 定义函数: y = x^2 + 3x + 1
y = x**2 + 3 * x + 1

# 反向传播
y.backward()

# 梯度: dy/dx = 2x + 3 = 2*2 + 3 = 7
print(x.grad)  # tensor(7.)
```

#### 多变量梯度

```python
x1 = torch.tensor(2.0, requires_grad=True)
x2 = torch.tensor(3.0, requires_grad=True)

# y = x1^2 + x2^2
y = x1**2 + x2**2

y.backward()

print(f"∂y/∂x1 = {x1.grad}")  # 4.0
print(f"∂y/∂x2 = {x2.grad}")  # 6.0
```

### 训练循环中的梯度

```python
# 模型参数
w = torch.tensor([1.0], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)

# 输入和目标
x = torch.tensor([2.0])
y_true = torch.tensor([5.0])

# 前向传播
y_pred = w * x + b

# 计算损失
loss = (y_pred - y_true) ** 2

# 反向传播
loss.backward()

print(f"∂loss/∂w = {w.grad}")  # -4.0
print(f"∂loss/∂b = {b.grad}")  # -2.0

# 参数更新
learning_rate = 0.1
with torch.no_grad():
    w -= learning_rate * w.grad
    b -= learning_rate * b.grad

print(f"w = {w.data}")  # 1.4
print(f"b = {b.data}")  # 0.2

# 重要: 更新后清零梯度
w.grad.zero_()
b.grad.zero_()
```

### 梯度计算注意事项

#### 1. 清零梯度
```python
# 每次backward()前需要清零梯度
optimizer.zero_grad()
# 或手动清零
model.zero_grad()
```

#### 2. 禁用梯度计算
```python
# 评估时禁用梯度
with torch.no_grad():
    predictions = model(X)

# 推理模式
model.eval()
```

#### 3. 梯度裁剪
```python
# 防止梯度爆炸
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## nn.Module模型定义

### nn.Module概述
- PyTorch中所有神经网络模型的基类
- 提供模型管理和自动微分功能
- 必须实现__init__和forward方法

### 模型定义模板

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义层

    def forward(self, x):
        # 前向传播
        return output
```

### 线性模型示例

```python
class SimpleLinearModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # 定义层
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 前向传播
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 创建模型
model = SimpleLinearModel(input_size=10, hidden_size=20, output_size=1)

# 查看模型
print(model)

# 参数数量
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数: {total_params}")
```

### 查看模型参数

```python
# 遍历参数
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# 访问特定层
fc1_weights = model.fc1.weight
fc1_bias = model.fc1.bias

print(fc1_weights.shape)
print(fc1_bias.shape)
```

### 模型方法

```python
# 训练模式
model.train()

# 评估模式
model.eval()

# 移动到GPU
model.to('cuda')

# 保存模型参数
torch.save(model.state_dict(), 'model.pth')

# 加载模型参数
model.load_state_dict(torch.load('model.pth'))
```

---

## 常用层

### 全连接层（nn.Linear）

#### 公式
```
y = xW^T + b
```

#### 示例

```python
# 输入维度10，输出维度5
linear = nn.Linear(10, 5)

# 输入 (batch_size=3, input_size=10)
x = torch.randn(3, 10)

# 输出 (batch_size=3, output_size=5)
y = linear(x)

print(y.shape)  # torch.Size([3, 5])
```

### LSTM层（nn.LSTM）

#### 参数说明
- `input_size`: 输入特征维度
- `hidden_size`: 隐藏状态维度
- `num_layers`: LSTM层数
- `batch_first`: batch是否在第一维
- `bidirectional`: 是否双向
- `dropout`: Dropout比例

#### 输入格式
- `batch_first=False`: (seq_len, batch, input_size)
- `batch_first=True`: (batch, seq_len, input_size)

#### 输出格式
- `output`: (batch, seq_len, hidden_size)
- `h_n`: (num_layers, batch, hidden_size)
- `c_n`: (num_layers, batch, hidden_size)

#### 示例

```python
# 创建LSTM
lstm = nn.LSTM(
    input_size=10,
    hidden_size=20,
    num_layers=2,
    batch_first=True,
    dropout=0.2
)

# 输入 (batch=5, seq_len=8, input_size=10)
x = torch.randn(5, 8, 10)

# 前向传播
output, (h_n, c_n) = lstm(x)

print(f"output: {output.shape}")  # (5, 8, 20)
print(f"h_n: {h_n.shape}")        # (2, 5, 20)
print(f"c_n: {c_n.shape}")        # (2, 5, 20)
```

### 激活函数

```python
import torch.nn as nn

x = torch.randn(5)

# ReLU
relu = nn.ReLU()
y_relu = relu(x)  # max(0, x)

# Sigmoid
sigmoid = nn.Sigmoid()
y_sigmoid = sigmoid(x)  # 1 / (1 + exp(-x))

# Tanh
tanh = nn.Tanh()
y_tanh = tanh(x)  # (exp(x) - exp(-x)) / (exp(x) + exp(-x))

# Leaky ReLU
leaky_relu = nn.LeakyReLU(0.01)
y_leaky = leaky_relu(x)  # max(0.01x, x)
```

### Dropout

#### 作用
- 防止过拟合
- 随机丢弃部分神经元

#### 示例

```python
dropout = nn.Dropout(p=0.5)  # 50%的神经元被置零

x = torch.ones(10)
y = dropout(x)

print(x)  # tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
print(y)  # tensor([2., 0., 2., 0., 2., 0., 0., 2., 0., 2.])  # 约50%为0
```

### BatchNorm

#### 作用
- 加速训练
- 提高稳定性

#### 1D BatchNorm

```python
batchnorm1d = nn.BatchNorm1d(num_features=10)

# 输入 (batch=5, features=10)
x = torch.randn(5, 10)
y = batchnorm1d(x)
```

#### 2D BatchNorm（用于CNN）

```python
batchnorm2d = nn.BatchNorm2d(num_features=10)

# 输入 (batch=5, channels=10, height=20, width=20)
x = torch.randn(5, 10, 20, 20)
y = batchnorm2d(x)
```

### Embedding层

#### 作用
- 将离散值映射为稠密向量
- 用于自然语言处理

#### 示例

```python
# 词汇表大小10000，嵌入维度100
embedding = nn.Embedding(num_embeddings=10000, embedding_dim=100)

# 输入 (batch=5, seq_len=10)
x = torch.randint(0, 10000, (5, 10))

# 输出 (batch=5, seq_len=10, embedding_dim=100)
y = embedding(x)

print(y.shape)  # torch.Size([5, 10, 100])
```

---

## 核心知识点总结

### PyTorch简介
- ✅ 动态计算图
- ✅ GPU加速
- ✅ 自动微分
- ✅ 丰富的API

### Tensor基础
- ✅ Tensor创建
- ✅ Tensor操作
- ✅ Tensor vs NumPy
- ✅ 互操作

### Autograd
- ✅ requires_grad
- ✅ backward()
- ✅ 梯度计算
- ✅ 梯度清零和裁剪

### nn.Module
- ✅ 模型定义模板
- ✅ forward方法
- ✅ 模型参数管理
- ✅ 训练/评估模式

### 常用层
- ✅ nn.Linear
- ✅ nn.LSTM
- ✅ 激活函数
- ✅ Dropout
- ✅ BatchNorm

---

## 下一步

继续学习: [LSTM模型构建系列](../03_LSTM模型构建系列/README.md)
