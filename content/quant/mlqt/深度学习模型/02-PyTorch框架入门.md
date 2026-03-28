# 02-PyTorch框架入门

> 本节介绍 PyTorch 框架的核心概念，包括 Tensor 操作、自动微分、模型定义和数据加载，为后续构建深度学习模型打下基础。

## PyTorch 简介和安装

### 什么是 PyTorch？

PyTorch 是 Meta 开发的深度学习框架，以其**动态计算图**和**Python 风格**而广受欢迎。

```
PyTorch vs TensorFlow (早期对比):

                    PyTorch              TensorFlow (1.x)
                    ───────              ─────────────────
计算图               动态                 静态
调试                 容易（Pythonic）     困难
部署                 较弱                 强
学习曲线             平缓                 陡峭
社区                学术界/研究           工业界
```

### 安装

```bash
# CPU 版本
pip install torch

# GPU 版本（CUDA 11.8）
pip install torch --index-url https://download.pytorch.org/whl/cu118

# GPU 版本（CUDA 12.1）
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 检查安装
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Tensor 基础

Tensor 是 PyTorch 的核心数据结构，类似于 NumPy 的 ndarray，但可以在 GPU 上运行。

### 创建 Tensor

```python
import torch
import numpy as np

# 从列表创建
t1 = torch.tensor([1, 2, 3])
print(f"从列表: {t1}")  # tensor([1, 2, 3])

# 从 NumPy 创建
arr = np.array([1, 2, 3])
t2 = torch.from_numpy(arr)
print(f"从 NumPy: {t2}")  # tensor([1, 2, 3], dtype=torch.int32)

# 创建全零 Tensor
t3 = torch.zeros(2, 3)
print(f"全零:\n{t3}")
# tensor([[0., 0., 0.],
#         [0., 0., 0.]])

# 创建全一 Tensor
t4 = torch.ones(2, 3)
print(f"全一:\n{t4}")

# 创建随机 Tensor（正态分布）
t5 = torch.randn(2, 3)
print(f"随机:\n{t5}")

# 创建指定范围的 Tensor
t6 = torch.arange(0, 10, 2)
print(f"范围: {t6}")  # tensor([0, 2, 4, 6, 8])

# 创建等间隔 Tensor
t7 = torch.linspace(0, 1, 5)
print(f"等间隔: {t7}")  # tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])
```

### Tensor 操作

```python
import torch

# 创建 Tensor
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"原始 Tensor:\n{x}")
# tensor([[1, 2, 3],
#         [4, 5, 6]])

# 形状
print(f"形状: {x.shape}")  # torch.Size([2, 3])
print(f"维度: {x.ndim}")   # 2

# 索引和切片
print(f"第一行: {x[0]}")           # tensor([1, 2, 3])
print(f"第一列: {x[:, 0]}")        # tensor([1, 4])
print(f"右下角: {x[1:, 1:]}")      # tensor([[5, 6]])

# Reshape
y = x.reshape(3, 2)
print(f"Reshape (3, 2):\n{y}")
# tensor([[1, 2],
#         [3, 4],
#         [5, 6]])

# 展平
z = x.flatten()
print(f"展平: {z}")  # tensor([1, 2, 3, 4, 5, 6])

# 拼接
a = torch.tensor([1, 2])
b = torch.tensor([3, 4])
print(f"按行拼接: {torch.cat([a, b], dim=0)}")  # tensor([1, 2, 3, 4])

# 堆叠
print(f"堆叠:\n{torch.stack([a, b], dim=0)}")
# tensor([[1, 2],
#         [3, 4]])

# 数学运算
x1 = torch.tensor([1, 2, 3])
x2 = torch.tensor([4, 5, 6])
print(f"加法: {x1 + x2}")      # tensor([5, 7, 9])
print(f"乘法: {x1 * x2}")      # tensor([ 4, 10, 18])
print(f"点积: {x1 @ x2}")      # 32
print(f"矩阵乘法: {torch.mm(x1.unsqueeze(0), x2.unsqueeze(1))}")  # tensor([[32]])

# 广播
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
y = torch.tensor([1, 2, 3])
print(f"广播加法:\n{x + y}")
# tensor([[2, 4, 6],
#         [5, 7, 9]])

# 统计
x = torch.tensor([1, 2, 3, 4, 5])
print(f"均值: {x.mean()}")        # 3.0
print(f"标准差: {x.std()}")       # 1.4142
print(f"最大值: {x.max()}")       # 5
print(f"最大值索引: {x.argmax()}")  # 4
```

### GPU 加速

```python
import torch

# 检查 CUDA 是否可用
print(f"CUDA 可用: {torch.cuda.is_available()}")

# 检查 GPU 数量
print(f"GPU 数量: {torch.cuda.device_count()}")

# 获取当前设备
print(f"当前设备: {torch.cuda.current_device()}")

# 获取设备名称
if torch.cuda.is_available():
    print(f"设备名称: {torch.cuda.get_device_name(0)}")

# 创建 Tensor 并放到 GPU
x = torch.randn(3, 3)
x_gpu = x.to('cuda')  # 或 x.cuda()
print(f"设备: {x_gpu.device}")  # cuda:0

# 从 GPU 移回 CPU
x_cpu = x_gpu.cpu()
print(f"设备: {x_cpu.device}")  # cpu

# 自动选择设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.randn(3, 3, device=device)
print(f"自动选择设备: {x.device}")

# 模型也可以放到 GPU
model = MyModel().to(device)

# 性能对比
import time

x = torch.randn(1000, 1000)

# CPU
start = time.time()
for _ in range(100):
    _ = x @ x
print(f"CPU 时间: {time.time() - start:.4f}秒")

if torch.cuda.is_available():
    x_gpu = x.cuda()
    # 预热
    for _ in range(10):
        _ = x_gpu @ x_gpu

    start = time.time()
    for _ in range(100):
        _ = x_gpu @ x_gpu
    torch.cuda.synchronize()  # 等待 GPU 完成
    print(f"GPU 时间: {time.time() - start:.4f}秒")
```

### 与 NumPy 互转

```python
import torch
import numpy as np

# NumPy → Tensor
arr = np.array([1, 2, 3])
t = torch.from_numpy(arr)
print(f"NumPy → Tensor: {t}")

# Tensor → NumPy
t = torch.tensor([1, 2, 3])
arr = t.numpy()
print(f"Tensor → NumPy: {arr}")

# 注意：共享内存
arr = np.array([1, 2, 3])
t = torch.from_numpy(arr)
t[0] = 999
print(f"修改 Tensor 后 NumPy 也变: {arr}")  # [999 2 3]

# 避免共享内存
t = torch.tensor(arr)  # 复制数据
# 或
t = torch.from_numpy(arr).clone()
```

## Autograd 自动微分

PyTorch 的 Autograd 提供了自动微分功能，是训练神经网络的核心。

### 基本概念

```python
import torch

# 创建 Tensor 并设置 requires_grad=True
x = torch.tensor([2.0, 3.0], requires_grad=True)
print(f"x: {x}, requires_grad: {x.requires_grad}")

# 计算过程
y = x + 2
z = y * y * 3
out = z.mean()

print(f"y = x + 2: {y}")
print(f"z = y² × 3: {z}")
print(f"out = z.mean(): {out}")

# 反向传播
out.backward()

# 查看梯度
print(f"x.grad: {x.grad}")  # tensor([4.5000, 6.0000])

# 手动验证梯度
# out = (3 * (x+2)²).mean()
# dout/dx₁ = 6*(x₁+2)/2 = 3*(x₁+2) = 3*(2+2) = 12/2 = 6
# dout/dx₂ = 6*(x₂+2)/2 = 3*(x₂+2) = 3*(3+2) = 15/2 = 7.5
```

### 计算图

```python
import torch

# 构建计算图
a = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([3.0], requires_grad=True)
c = torch.tensor([4.0], requires_grad=True)

# y = a + b
y = a + b
# z = y * c
z = y * c

# 反向传播
z.backward()

print(f"dz/da: {a.grad}")  # 4.0 (= c)
print(f"dz/db: {b.grad}")  # 4.0 (= c)
print(f"dz/dc: {c.grad}")  # 5.0 (= a+b)
```

计算图可视化：

```
计算图结构：

    a ───┐
         │ (+) ─── y ───┐
    b ───┘             │ (×) ─── z ───→ backward()
                      │
    c ─────────────────┘

梯度反向传播：
    z 对 a 的梯度：∂z/∂a = ∂z/∂y × ∂y/∂a = c × 1 = 4
    z 对 b 的梯度：∂z/∂b = ∂z/∂y × ∂y/∂b = c × 1 = 4
    z 对 c 的梯度：∂z/∂c = y = a + b = 5
```

### 梯度控制

```python
import torch

# 禁用梯度计算
with torch.no_grad():
    x = torch.tensor([1.0], requires_grad=True)
    y = x + 1
    print(f"禁用梯度: y.requires_grad = {y.requires_grad}")  # False

# 或使用装饰器
@torch.no_grad()
def add(x, y):
    return x + y

x = torch.tensor([1.0], requires_grad=True)
y = add(x, 2)
print(f"装饰器禁用: y.requires_grad = {y.requires_grad}")  # False

# 清空梯度
x = torch.tensor([2.0], requires_grad=True)
y = x * x
y.backward()
print(f"第一次梯度: {x.grad}")  # tensor([4.])

# 再次计算需要清空梯度
y = x * x
y.backward()
print(f"不清空梯度会累加: {x.grad}")  # tensor([8.])

x.grad.zero_()
y = x * x
y.backward()
print(f"清空后梯度: {x.grad}")  # tensor([4.])
```

### 梯度裁剪

```python
import torch

# 创建参数
w = torch.randn(3, requires_grad=True)
x = torch.randn(3)
y = w * x
loss = y.sum()

loss.backward()
print(f"裁剪前梯度范数: {w.grad.norm()}")  # 例如: 5.2

# 裁剪梯度
torch.nn.utils.clip_grad_norm_([w], max_norm=1.0)
print(f"裁剪后梯度范数: {w.grad.norm()}")  # <= 1.0
```

## nn.Module 模型定义

`nn.Module` 是 PyTorch 中所有神经网络模块的基类。

### 自定义模型

```python
import torch
import torch.nn as nn

# 定义一个简单的 MLP
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        # 定义层
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 定义前向传播
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# 创建模型
model = SimpleMLP(input_size=10, hidden_size=20, output_size=1)

# 前向传播
x = torch.randn(5, 10)  # batch_size=5, input_size=10
output = model(x)
print(f"输出形状: {output.shape}")  # torch.Size([5, 1])
```

模型结构可视化：

```
SimpleMLP 结构：

输入层 (10)
    │
    ▼
┌─────────┐
│ fc1     │ Linear(10 → 20)
│ ReLU    │
└─────────┘
    │
    ▼
┌─────────┐
│ fc2     │ Linear(20 → 20)
│ ReLU    │
└─────────┘
    │
    ▼
┌─────────┐
│ fc3     │ Linear(20 → 1)
└─────────┘
    │
    ▼
输出 (1)
```

### 参数管理

```python
import torch.nn as nn

model = SimpleMLP(10, 20, 1)

# 查看所有参数
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# 输出:
# fc1.weight: torch.Size([20, 10])
# fc1.bias: torch.Size([20])
# fc2.weight: torch.Size([20, 20])
# fc2.bias: torch.Size([20])
# fc3.weight: torch.Size([1, 20])
# fc3.bias: torch.Size([1])

# 查看参数总数
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数数: {total_params}")

# 可训练参数
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"可训练参数: {trainable_params}")

# 访问特定参数
print(f"fc1 权重:\n{model.fc1.weight[:3, :3]}")  # 前3行3列

# 冻结参数
model.fc1.weight.requires_grad = False
print(f"fc1 可训练: {model.fc1.weight.requires_grad}")  # False
```

## 常用层

### nn.Linear（全连接层）

```python
import torch.nn as nn

# 创建全连接层
fc = nn.Linear(in_features=10, out_features=5, bias=True)

# 输入
x = torch.randn(3, 10)  # (batch_size=3, in_features=10)

# 前向传播
output = fc(x)
print(f"输出形状: {output.shape}")  # (3, 5)

# 查看权重和偏置
print(f"权重形状: {fc.weight.shape}")  # (5, 10)
print(f"偏置形状: {fc.bias.shape}")    # (5)
```

### nn.LSTM

```python
import torch.nn as nn

# 单层单向 LSTM
lstm = nn.LSTM(
    input_size=10,      # 输入特征维度
    hidden_size=20,     # 隐藏状态维度
    num_layers=1,       # LSTM 层数
    batch_first=True,   # 输入形状 (batch, seq, feature)
    bidirectional=False # 是否双向
)

# 输入: (batch_size=3, seq_len=5, input_size=10)
x = torch.randn(3, 5, 10)

# 前向传播
output, (h_n, c_n) = lstm(x)

print(f"输出形状: {output.shape}")   # (3, 5, 20)
print(f"h_n 形状: {h_n.shape}")       # (1, 3, 20)
print(f"c_n 形状: {c_n.shape}")       # (1, 3, 20)
```

LSTM 参数详解：

| 参数 | 说明 | 常用值 |
|------|------|--------|
| `input_size` | 输入特征维度 | 特征数量 |
| `hidden_size` | 隐藏状态维度 | 32, 64, 128, 256 |
| `num_layers` | LSTM 层数 | 1-4 |
| `batch_first` | 输入维度顺序 | True（推荐） |
| `bidirectional` | 是否双向 | False（预测）, True（分类） |
| `dropout` | 层间 dropout | 0.0-0.5 |

### nn.Dropout

```python
import torch.nn as nn

# Dropout 层
dropout = nn.Dropout(p=0.5)  # 50% 概率丢弃

x = torch.randn(5, 10)
output = dropout(x)

# 注意：只在训练时生效
dropout.train()  # 训练模式（有 dropout）
output_train = dropout(x)
print(f"训练模式（部分元素为0）:\n{output_train}")

dropout.eval()  # 评估模式（无 dropout，按比例缩放）
output_eval = dropout(x)
print(f"评估模式:\n{output_eval}")
```

### nn.BatchNorm1d

```python
import torch.nn as nn

# 1D BatchNorm
bn = nn.BatchNorm1d(num_features=10)

# 输入: (batch_size=5, num_features=10)
x = torch.randn(5, 10)

output = bn(x)
print(f"输出形状: {output.shape}")  # (5, 10)

# 查看运行的统计量
print(f"运行均值: {bn.running_mean[:3]}")  # 前3个特征
print(f"运行方差: {bn.running_var[:3]}")
```

### nn.Embedding

```python
import torch.nn as nn

# Embedding 层（用于类别变量）
embedding = nn.Embedding(num_embeddings=100, embedding_dim=32)

# 输入: 索引
x = torch.tensor([1, 5, 10, 50])

# 前向传播
output = embedding(x)
print(f"输出形状: {output.shape}")  # (4, 32)
print(f"第一个嵌入向量: {output[0]}")
```

## Dataset 和 DataLoader

### 自定义 Dataset

```python
import torch
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    """时序数据集"""
    def __init__(self, data, seq_len, horizon=1):
        """
        Args:
            data: numpy array 或 tensor, shape (n_samples, n_features)
            seq_len: 序列长度（用多少个历史步预测）
            horizon: 预测步数
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len
        self.horizon = horizon

    def __len__(self):
        # 返回样本数量
        return len(self.data) - self.seq_len - self.horizon + 1

    def __getitem__(self, idx):
        # 返回单个样本
        x = self.data[idx:idx + self.seq_len]      # 输入序列
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.horizon]  # 标签
        return x, y

# 使用示例
import numpy as np

# 创建模拟数据（100个样本，5个特征）
data = np.random.randn(100, 5)

# 创建数据集
dataset = TimeSeriesDataset(data, seq_len=10, horizon=1)

print(f"数据集大小: {len(dataset)}")

# 获取第一个样本
x, y = dataset[0]
print(f"输入形状: {x.shape}")  # (10, 5)
print(f"标签形状: {y.shape}")  # (1, 5)
```

### DataLoader

```python
from torch.utils.data import DataLoader

# 创建 DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=16,      # 批次大小
    shuffle=False,      # 时序数据不 shuffle
    num_workers=0,      # 数据加载进程数（0=主进程）
    drop_last=False,    # 是否丢弃最后不完整的批次
    pin_memory=False    # 是否锁页内存（GPU 加速）
)

# 迭代
for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
    print(f"批次 {batch_idx}: x={x_batch.shape}, y={y_batch.shape}")
    if batch_idx >= 2:  # 只显示前3个批次
        break
```

DataLoader 参数说明：

| 参数 | 说明 | 量化场景建议 |
|------|------|-------------|
| `batch_size` | 批次大小 | 32-256 |
| `shuffle` | 是否打乱 | **False**（时序数据不打乱） |
| `num_workers` | 加载进程数 | 0（调试）/ 4（生产） |
| `drop_last` | 丢弃末批次 | True（训练）/ False（验证） |
| `pin_memory` | 锁页内存 | True（GPU 训练） |

## 设备管理

```python
import torch

# 自动选择设备
def get_device():
    """获取最佳设备"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Apple Silicon GPU
        return torch.device('mps')
    else:
        return torch.device('cpu')

device = get_device()
print(f"使用设备: {device}")

# 将模型和数据移到设备
model = SimpleMLP(10, 20, 1).to(device)
x = torch.randn(5, 10).to(device)

# 前向传播
output = model(x)

# 确保数据在同一设备
def predict(model, x):
    model.eval()
    with torch.no_grad():
        x = x.to(device)  # 确保输入在正确设备
        output = model(x)
    return output.cpu()   # 将结果移回 CPU
```

## 完整训练流程示例

下面是一个极简的端到端训练流程：

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

# ========== 1. 定义模型 ==========
class SimpleModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.net(x)

# ========== 2. 定义数据集 ==========
class SimpleDataset(Dataset):
    def __init__(self, n_samples=1000, n_features=10):
        self.X = torch.randn(n_samples, n_features)
        self.y = torch.randn(n_samples, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ========== 3. 训练函数 ==========
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for X_batch, y_batch in dataloader:
        # 移到设备
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # 前向传播
        predictions = model(X_batch).squeeze()
        loss = criterion(predictions, y_batch.squeeze())

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# ========== 4. 主程序 ==========
def main():
    # 超参数
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 10

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据
    dataset = SimpleDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 模型
    model = SimpleModel().to(device)

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 训练循环
    for epoch in range(EPOCHS):
        loss = train(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}")

    # 预测
    model.eval()
    with torch.no_grad():
        X_test = torch.randn(1, 10).to(device)
        prediction = model(X_test)
        print(f"预测值: {prediction.item():.4f}")

if __name__ == "__main__":
    main()
```

## 核心知识点总结

### 1. Tensor 操作
```python
# 创建
x = torch.randn(3, 4)
y = torch.zeros(3, 4)

# 操作
z = x + y           # 加法
z = torch.cat([x, y], dim=0)  # 拼接
z = x.reshape(4, 3)  # 重塑

# GPU
x_gpu = x.to('cuda')
```

### 2. Autograd
```python
x = torch.tensor([2.0], requires_grad=True)
y = x * x
y.backward()
print(x.grad)  # 4.0
```

### 3. nn.Module
```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)
```

### 4. DataLoader
```python
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,  # 时序数据不打乱
    num_workers=4
)
```

## 练习建议

1. **熟悉 Tensor 操作**：练习各种 reshape、索引、切片
2. **手写梯度计算**：用简单函数验证自动微分
3. **构建自定义 Dataset**：实现自己的时序数据集
4. **完整训练流程**：从数据到模型到训练的完整代码

## 下一节

在 **[03-LSTM模型构建.md](./03-LSTM模型构建.md)** 中，我们将学习如何用 PyTorch 构建各种 LSTM 模型。
