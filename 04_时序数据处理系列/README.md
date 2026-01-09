# 时序数据处理系列 - 预处理与Dataset

## 📚 系列概述
本系列文档涵盖时序数据的构造、标准化、数据划分、PyTorch Dataset和DataLoader。

---

## 📖 文档列表

1. [滑动窗口方法](#滑动窗口方法)
2. [数据划分](#数据划分)
3. [特征标准化](#特征标准化)
4. [PyTorch Dataset](#pytorch-dataset)
5. [DataLoader](#dataloader)

---

## 滑动窗口方法

### 原理
- 使用过去N天的数据预测下一天
- 窗口滑动，生成多个样本
- 是构造时序训练数据的标准方法

### 示例

#### 原始数据
```
Day 1: [0.1, 0.2, 0.3]
Day 2: [0.2, 0.3, 0.4]
Day 3: [0.3, 0.4, 0.5]
Day 4: [0.4, 0.5, 0.6]
Day 5: [0.5, 0.6, 0.7]
```

#### 序列长度 = 3

| 样本 | 输入 (X) | 目标 (y) |
|------|---------|---------|
| 1 | Day 1-3 | Day 4 |
| 2 | Day 2-4 | Day 5 |

### Python实现

```python
import numpy as np

def create_sequences(data, seq_len, target_idx=0):
    """
    滑动窗口构造时序序列

    参数:
        data: 原始数据 (n_samples, n_features)
        seq_len: 序列长度
        target_idx: 目标特征索引

    返回:
        X: 输入序列 (n_samples-seq_len, seq_len, n_features)
        y: 目标值 (n_samples-seq_len,)
    """
    X, y = [], []

    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, target_idx])

    return np.array(X), np.array(y)

# 示例
data = np.random.randn(100, 10)  # 100天，10个特征
X, y = create_sequences(data, seq_len=20, target_idx=0)

print(X.shape)  # (80, 20, 10)
print(y.shape)  # (80,)
```

### 高级滑动窗口

```python
def create_sequences_multi_step(data, seq_len, target_len, target_idx=0):
    """
    多步预测滑动窗口

    参数:
        data: 原始数据
        seq_len: 输入序列长度
        target_len: 预测步数
        target_idx: 目标特征索引

    返回:
        X: (n_samples-seq_len-target_len, seq_len, n_features)
        y: (n_samples-seq_len-target_len, target_len)
    """
    X, y = [], []

    for i in range(len(data) - seq_len - target_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+target_len, target_idx])

    return np.array(X), np.array(y)

# 示例：预测未来5天
X, y = create_sequences_multi_step(data, seq_len=20, target_len=5)
print(X.shape)  # (75, 20, 10)
print(y.shape)  # (75, 5)
```

### 滑动窗口选择建议

| 预测目标 | seq_len | 说明 |
|---------|---------|------|
| **短期预测** | 5-10 | 日内交易 |
| **中期预测** | 20-60 | 几天到几周 |
| **长期预测** | 60-120 | 几个月 |

---

## 数据划分

### 时间序列划分原则
- 按时间顺序划分
- 不能随机划分
- 训练集 < 验证集 < 测试集

### 划分方法

```python
def time_series_split(data, train_ratio=0.7, val_ratio=0.15):
    """
    时间序列数据划分

    参数:
        data: 完整数据
        train_ratio: 训练集比例
        val_ratio: 验证集比例

    返回:
        train, val, test
    """
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]

    return train, val, test

# 示例
data = np.random.randn(1000, 10)
train, val, test = time_series_split(data)

print(f"Train: {train.shape}")  # (700, 10)
print(f"Val: {val.shape}")      # (150, 10)
print(f"Test: {test.shape}")     # (150, 10)
```

### 滚动窗口验证

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for fold, (train_index, test_index) in enumerate(tscv.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    print(f"Fold {fold+1}:")
    print(f"  Train: {X_train.shape}")
    print(f"  Test: {X_test.shape}")
```

### Walk-Forward验证

```python
def walk_forward_validation(data, initial_train_size, step_size):
    """
    Walk-Forward验证

    参数:
        data: 完整数据
        initial_train_size: 初始训练集大小
        step_size: 每次前进步数
    """
    splits = []
    n = len(data)

    train_end = initial_train_size

    while train_end + step_size < n:
        test_start = train_end
        test_end = min(test_start + step_size, n)

        splits.append((
            data[:train_end],
            data[test_start:test_end]
        ))

        train_end += step_size

    return splits

# 示例
splits = walk_forward_validation(data, initial_train_size=500, step_size=100)

for i, (train, test) in enumerate(splits):
    print(f"Fold {i+1}: Train {train.shape}, Test {test.shape}")
```

---

## 特征标准化

### 标准化方法

#### 1. Z-score标准化（StandardScaler）

```python
from sklearn.preprocessing import StandardScaler

# 拟合训练集
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1]))

# 转换验证集和测试集
X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1]))
X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1]))

# 恢复形状
X_train_scaled = X_train_scaled.reshape(X_train.shape)
X_val_scaled = X_val_scaled.reshape(X_val.shape)
X_test_scaled = X_test_scaled.reshape(X_test.shape)
```

#### 2. Min-Max标准化

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1]))

X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1]))
X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1]))

# 恢复形状
X_train_scaled = X_train_scaled.reshape(X_train.shape)
X_val_scaled = X_val_scaled.reshape(X_val.shape)
X_test_scaled = X_test_scaled.reshape(X_test.shape)
```

#### 3. RobustScaler

```python
from sklearn.preprocessing import RobustScaler

# 对异常值更鲁棒
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1]))

X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1]))
X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1]))

# 恢复形状
X_train_scaled = X_train_scaled.reshape(X_train.shape)
X_val_scaled = X_val_scaled.reshape(X_val.shape)
X_test_scaled = X_test_scaled.reshape(X_test.shape)
```

### 滚动标准化

```python
def rolling_standardize(data, window=20):
    """
    滚动窗口标准化

    参数:
        data: 原始数据 (n_samples, n_features)
        window: 滚动窗口大小

    返回:
        scaled: 标准化后的数据
    """
    scaled = np.zeros_like(data)

    for i in range(len(data)):
        if i < window:
            # 初期使用累积数据
            mean = data[:i+1].mean(axis=0)
            std = data[:i+1].std(axis=0)
        else:
            # 使用滚动窗口
            mean = data[i-window:i].mean(axis=0)
            std = data[i-window:i].std(axis=0)

        scaled[i] = (data[i] - mean) / (std + 1e-8)

    return scaled

# 示例
X_train_scaled = rolling_standardize(X_train, window=20)
```

### 标准化注意事项

#### 1. 只用训练集拟合
```python
# ❌ 错误：用所有数据拟合
scaler = StandardScaler()
X_scaled = scaler.fit_transform(all_data)

# ✅ 正确：只用训练集拟合
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```

#### 2. 反标准化预测结果
```python
# 预测
predictions = model(X_test_scaled)

# 反标准化
predictions_original = scaler.inverse_transform(predictions)
```

---

## PyTorch Dataset

### 自定义Dataset

```python
from torch.utils.data import Dataset
import torch

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        """
        参数:
            X: 输入数据 (n_samples, seq_len, n_features)
            y: 目标值 (n_samples,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        """返回样本数量"""
        return len(self.X)

    def __getitem__(self, idx):
        """
        获取单个样本

        返回:
            X: (seq_len, n_features)
            y: scalar
        """
        return self.X[idx], self.y[idx]
```

### 创建Dataset

```python
# 创建Dataset
train_dataset = TimeSeriesDataset(X_train, y_train)
val_dataset = TimeSeriesDataset(X_val, y_val)
test_dataset = TimeSeriesDataset(X_test, y_test)

# 查看大小
print(f"Train dataset size: {len(train_dataset)}")
print(f"Val dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# 获取单个样本
X_sample, y_sample = train_dataset[0]
print(f"Sample X shape: {X_sample.shape}")
print(f"Sample y: {y_sample}")
```

### 高级Dataset

```python
class AdvancedTimeSeriesDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X, y = self.X[idx], self.y[idx]

        # 应用变换
        if self.transform:
            X = self.transform(X)

        return X, y

# 定义变换
def add_noise(x, noise_level=0.01):
    """添加噪声"""
    noise = torch.randn_like(x) * noise_level
    return x + noise

# 创建Dataset
train_dataset = AdvancedTimeSeriesDataset(
    X_train,
    y_train,
    transform=lambda x: add_noise(x, 0.01)
)
```

---

## DataLoader

### 创建DataLoader

```python
from torch.utils.data import DataLoader

# 创建DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,  # 训练集打乱
    num_workers=4,
    pin_memory=True  # 加速GPU传输
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,  # 验证集不打乱
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)
```

### 使用DataLoader

```python
# 训练循环
for epoch in range(num_epochs):
    model.train()

    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        # 前向传播
        predictions = model(X_batch)
        loss = criterion(predictions.squeeze(), y_batch)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印进度
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    # 验证
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            predictions = model(X_batch)
            loss = criterion(predictions.squeeze(), y_batch)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f"Epoch {epoch}, Val Loss: {val_loss:.4f}")
```

### DataLoader参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| **batch_size** | 批大小 | 32, 64, 128 |
| **shuffle** | 是否打乱 | 训练集=True，验证/测试=False |
| **num_workers** | 加载进程数 | 4-8 |
| **pin_memory** | 锁页内存 | True（GPU训练） |
| **drop_last** | 丢弃不完整batch | False |

### 动态批大小

```python
# 根据序列长度动态调整
class DynamicBatchSampler:
    def __init__(self, dataset, max_batch_size=32):
        self.dataset = dataset
        self.max_batch_size = max_batch_size

    def __iter__(self):
        batch = []
        for idx in range(len(self.dataset)):
            batch.append(idx)
            if len(batch) >= self.max_batch_size:
                yield batch
                batch = []

        if batch:
            yield batch

# 使用
train_loader = DataLoader(
    train_dataset,
    batch_sampler=DynamicBatchSampler(train_dataset, max_batch_size=32),
    collate_fn=lambda batch: default_collate([train_dataset[i] for i in batch])
)
```

---

## 数据处理流程

### 完整流程

```python
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# 1. 加载数据
data = np.load('stock_data.npy')  # (n_days, n_features)

# 2. 构造序列
def create_sequences(data, seq_len, target_idx=0):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, target_idx])
    return np.array(X), np.array(y)

X, y = create_sequences(data, seq_len=20)

# 3. 划分数据
train_size = int(0.7 * len(X))
val_size = int(0.15 * len(X))

X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]

# 4. 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

# 5. 创建Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TimeSeriesDataset(X_train_scaled, y_train)
val_dataset = TimeSeriesDataset(X_val_scaled, y_val)
test_dataset = TimeSeriesDataset(X_test_scaled, y_test)

# 6. 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Train: {len(train_dataset)} samples")
print(f"Val: {len(val_dataset)} samples")
print(f"Test: {len(test_dataset)} samples")
```

---

## 核心知识点总结

### 滑动窗口
- ✅ 基本原理
- ✅ 单步预测
- ✅ 多步预测
- ✅ 序列长度选择

### 数据划分
- ✅ 时间序列划分原则
- ✅ 滚动窗口验证
- ✅ Walk-Forward验证

### 特征标准化
- ✅ Z-score标准化
- ✅ Min-Max标准化
- ✅ RobustScaler
- ✅ 滚动标准化

### Dataset
- ✅ 自定义Dataset
- ✅ 高级Dataset
- ✅ 数据变换

### DataLoader
- ✅ 创建DataLoader
- ✅ 参数配置
- ✅ 使用示例

---

## 下一步

继续学习: [模型训练优化系列](../05_模型训练优化系列/README.md)
