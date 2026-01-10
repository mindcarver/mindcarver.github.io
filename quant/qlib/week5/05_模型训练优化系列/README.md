# 模型训练优化系列 - 训练与优化策略

## 📚 系列概述
本系列文档涵盖损失函数、优化器、训练循环、早停策略、正则化方法和超参数优化。

---

## 📖 文档列表

1. [损失函数](#损失函数)
2. [优化器](#优化器)
3. [训练循环](#训练循环)
4. [早停策略](#早停策略)
5. [正则化](#正则化)
6. [学习率调度](#学习率调度)

---

## 损失函数

### 回归任务损失函数

#### 1. MSE（Mean Squared Error）

```python
import torch.nn as nn

criterion = nn.MSELoss()

predictions = torch.randn(10, 1)
targets = torch.randn(10, 1)

loss = criterion(predictions, targets)
```

**特点**:
- 对大误差敏感
- 计算简单
- 常用

**适用场景**:
- 回归任务
- 异常值较少的数据

#### 2. MAE（Mean Absolute Error）

```python
criterion = nn.L1Loss()

loss = criterion(predictions, targets)
```

**特点**:
- 对异常值鲁棒
- 梯度恒定
- 不常用

**适用场景**:
- 异常值较多
- 需要鲁棒性

#### 3. Smooth L1 Loss

```python
criterion = nn.SmoothL1Loss()

loss = criterion(predictions, targets)
```

**特点**:
- MSE和MAE的折中
- 对异常值鲁棒
- 推荐使用

**适用场景**:
- 通用回归任务
- 平衡鲁棒性和稳定性

### 自定义损失函数

```python
def custom_loss(predictions, targets, model, lambda_l1=0.01, lambda_l2=0.01):
    """
    自定义损失函数

    参数:
        predictions: 模型预测
        targets: 真实值
        model: 模型
        lambda_l1: L1正则化系数
        lambda_l2: L2正则化系数
    """
    # MSE
    mse = torch.mean((predictions - targets) ** 2)

    # L1正则化
    l1_reg = sum(p.abs().sum() for p in model.parameters())

    # L2正则化
    l2_reg = sum(p.pow(2).sum() for p in model.parameters())

    # 总损失
    total_loss = mse + lambda_l1 * l1_reg + lambda_l2 * l2_reg

    return total_loss
```

### 损失函数选择建议

| 场景 | 推荐损失函数 |
|------|-------------|
| **通用回归** | MSE |
| **异常值多** | MAE |
| **平衡选择** | Smooth L1 |
| **需要正则化** | 自定义损失 |

---

## 优化器

### 常用优化器

#### 1. SGD（随机梯度下降）

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4  # L2正则化
)
```

**特点**:
- 简单稳定
- 收敛慢
- 需要调参

**适用场景**:
- 数据量大
- 需要稳定训练

#### 2. Adam（推荐）

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    weight_decay=1e-4
)
```

**特点**:
- 自适应学习率
- 收敛快
- 默认选择

**适用场景**:
- 通用场景
- 快速迭代

#### 3. RMSprop

```python
optimizer = torch.optim.RMSprop(
    model.parameters(),
    lr=0.001,
    alpha=0.99
)
```

**特点**:
- 适合RNN
- 自适应学习率

**适用场景**:
- RNN/LSTM
- 时序预测

### 优化器对比

| 优化器 | 适用场景 | 优点 | 缺点 | 推荐学习率 |
|--------|---------|------|------|-----------|
| **SGD** | 数据量大 | 简单稳定 | 收敛慢 | 0.01-0.1 |
| **Adam** | 通用 | 收敛快 | 可能过拟合 | 0.0001-0.001 |
| **RMSprop** | RNN | 适合序列 | 参数调整 | 0.001-0.01 |

### 优化器选择建议

**选择Adam**:
- 默认选择
- 快速迭代
- 通用场景

**选择SGD**:
- 数据量大
- 需要稳定
- 研究论文

**选择RMSprop**:
- RNN/LSTM
- 时序预测

---

## 训练循环

### 完整训练循环

```python
import copy
from tqdm import tqdm

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50):
    """
    训练模型

    参数:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        num_epochs: 训练轮数

    返回:
        model: 最佳模型
        train_losses: 训练损失
        val_losses: 验证损失
    """

    # 记录损失
    train_losses = []
    val_losses = []

    # 最佳模型
    best_model = None
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0

        # 使用进度条
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')

        for X_batch, y_batch in pbar:
            # 前向传播
            predictions = model(X_batch)
            loss = criterion(predictions.squeeze(), y_batch)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 参数更新
            optimizer.step()

            train_loss += loss.item()

            # 更新进度条
            pbar.set_postfix({'loss': loss.item()})

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                predictions = model(X_batch)
                loss = criterion(predictions.squeeze(), y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())

        # 打印进度
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")

    # 加载最佳模型
    model.load_state_dict(best_model)

    return model, train_losses, val_losses
```

### 训练循环示例

```python
# 定义模型
model = LSTMModel(
    input_size=10,
    hidden_size=64,
    num_layers=2,
    output_size=1
)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练
model, train_losses, val_losses = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=50
)
```

---

## 早停策略（Early Stopping）

### 原理
- 监控验证集损失
- 连续N个epoch不改善则停止
- 防止过拟合

### 实现

```python
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=True):
        """
        早停策略

        参数:
            patience: 容忍不改善的epoch数
            min_delta: 最小改善幅度
            verbose: 是否打印信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """保存最佳模型"""
        if self.verbose:
            print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}). Saving model...')
        self.best_model = copy.deepcopy(model.state_dict())

    def load_best_model(self, model):
        """加载最佳模型"""
        if self.best_model is not None:
            model.load_state_dict(self.best_model)
```

### 使用早停

```python
# 创建早停
early_stopping = EarlyStopping(patience=10, min_delta=0.0001, verbose=True)

# 训练循环
for epoch in range(num_epochs):
    # 训练
    train_loss = train(model, train_loader, criterion, optimizer)

    # 验证
    val_loss = validate(model, val_loader, criterion)

    # 检查早停
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping!")
        break

# 加载最佳模型
early_stopping.load_best_model(model)
```

---

## 正则化

### 1. L1/L2正则化

```python
def l1_regularization(model, lambda_l1=0.01):
    """
    L1正则化

    参数:
        model: 模型
        lambda_l1: L1系数

    返回:
        L1损失
    """
    l1_loss = 0
    for param in model.parameters():
        l1_loss += torch.sum(torch.abs(param))
    return lambda_l1 * l1_loss

def l2_regularization(model, lambda_l2=0.01):
    """
    L2正则化

    参数:
        model: 模型
        lambda_l2: L2系数

    返回:
        L2损失
    """
    l2_loss = 0
    for param in model.parameters():
        l2_loss += torch.sum(param ** 2)
    return lambda_l2 * l2_loss

# 使用
loss = criterion(predictions, targets)
loss += l1_regularization(model, lambda_l1=0.01)
loss += l2_regularization(model, lambda_l2=0.01)
```

### 2. Dropout

```python
class LSTMModelWithDropout(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)  # Dropout层
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out[:, -1, :])
        return output
```

### 3. 批量归一化（BatchNorm）

```python
class LSTMModelWithBN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_size)  # BatchNorm
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.bn(lstm_out[:, -1, :])
        output = self.fc(lstm_out)
        return output
```

### 4. 梯度裁剪

```python
# 反向传播后
loss.backward()

# 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 或梯度裁剪
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

---

## 学习率调度

### 学习率衰减

#### 1. StepLR

```python
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,  # 每10个epoch
    gamma=0.1      # 学习率乘0.1
)

# 训练循环
for epoch in range(num_epochs):
    train(model, train_loader, optimizer, criterion)
    scheduler.step()  # 更新学习率
```

#### 2. ReduceLROnPlateau

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',      # 监控指标越小越好
    factor=0.1,      # 学习率乘0.1
    patience=5,      # 容忍5个epoch不改善
    verbose=True
)

# 训练循环
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion)
    val_loss = validate(model, val_loader, criterion)
    scheduler.step(val_loss)  # 基于验证损失调整
```

#### 3. CosineAnnealingLR

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=50,        # 总epoch数
    eta_min=1e-6     # 最小学习率
)

# 训练循环
for epoch in range(num_epochs):
    train(model, train_loader, optimizer, criterion)
    scheduler.step()
```

#### 4. Warmup

```python
from torch.optim.lr_scheduler import LambdaLR

def warmup_lambda(epoch, warmup_epochs=10):
    """Warmup学习率调度"""
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    return 1.0

scheduler = LambdaLR(
    optimizer,
    lr_lambda=lambda epoch: warmup_lambda(epoch, warmup_epochs=10)
)
```

### 学习率可视化

```python
import matplotlib.pyplot as plt

# 记录学习率
learning_rates = []

for epoch in range(num_epochs):
    # 训练
    train(model, train_loader, optimizer, criterion)

    # 记录学习率
    learning_rates.append(optimizer.param_groups[0]['lr'])

    # 更新学习率
    scheduler.step()

# 绘制学习率曲线
plt.figure(figsize=(10, 6))
plt.plot(range(num_epochs), learning_rates)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.show()
```

---

## 训练技巧

### 1. 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for X_batch, y_batch in train_loader:
    optimizer.zero_grad()

    with autocast():
        predictions = model(X_batch)
        loss = criterion(predictions.squeeze(), y_batch)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 2. 梯度累积

```python
accumulation_steps = 4

for i, (X_batch, y_batch) in enumerate(train_loader):
    predictions = model(X_batch)
    loss = criterion(predictions.squeeze(), y_batch)

    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 3. 模型并行

```python
# 多GPU训练
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

model.to('cuda')
```

---

## 核心知识点总结

### 损失函数
- ✅ MSE、MAE、Smooth L1
- ✅ 自定义损失函数
- ✅ 选择建议

### 优化器
- ✅ SGD、Adam、RMSprop
- ✅ 优化器对比
- ✅ 选择建议

### 训练循环
- ✅ 完整训练循环
- ✅ 进度条
- ✅ 梯度裁剪

### 早停策略
- ✅ EarlyStopping实现
- ✅ 使用方法
- ✅ 模型保存

### 正则化
- ✅ L1/L2正则化
- ✅ Dropout
- ✅ BatchNorm
- ✅ 梯度裁剪

### 学习率调度
- ✅ StepLR
- ✅ ReduceLROnPlateau
- ✅ CosineAnnealingLR
- ✅ Warmup

---

## 下一步

继续学习: [模型评估与对比系列](../06_模型评估与对比系列/README.md)
