# 实战应用系列 - 案例与最佳实践

## 📚 系列概述
本系列文档涵盖完整的LSTM预测流程、超参数调优、模型保存与加载、评估指标和最佳实践。

---

## 📖 文档列表

1. [完整预测流程](#完整预测流程)
2. [超参数调优](#超参数调优)
3. [模型保存与加载](#模型保存与加载)
4. [评估指标](#评估指标)
5. [LSTM vs LightGBM](#lstm-vs-lightgbm)
6. [最佳实践](#最佳实践)

---

## 完整预测流程

### 步骤1: 数据准备

```python
import pandas as pd
import numpy as np

# 加载数据
data = pd.read_csv('stock_prices.csv')

# 特征工程
features = ['close', 'volume', 'ma5', 'ma20', 'rsi']
data = data[features].values

# 标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# 构造序列
def create_sequences(data, seq_len, target_idx=0):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, target_idx])
    return np.array(X), np.array(y)

seq_len = 20
X, y = create_sequences(data_scaled, seq_len)

# 划分数据
train_size = int(0.7 * len(X))
val_size = int(0.15 * len(X))

X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]

print(f"Train: {X_train.shape}")
print(f"Val: {X_val.shape}")
print(f"Test: {X_test.shape}")
```

### 步骤2: 模型定义

```python
import torch
import torch.nn as nn

class StockPredictionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        output = self.fc(last_output)
        return output

# 创建模型
model = StockPredictionLSTM(
    input_size=X_train.shape[2],
    hidden_size=64,
    num_layers=2,
    dropout=0.2
)
```

### 步骤3: Dataset和DataLoader

```python
from torch.utils.data import Dataset, DataLoader

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 创建Dataset和DataLoader
train_dataset = StockDataset(X_train, y_train)
val_dataset = StockDataset(X_val, y_val)
test_dataset = StockDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

### 步骤4: 训练

```python
import copy

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练
num_epochs = 50
best_val_loss = float('inf')
best_model = None

for epoch in range(num_epochs):
    # 训练
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions.squeeze(), y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # 验证
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            predictions = model(X_batch)
            loss = criterion(predictions.squeeze(), y_batch)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model.state_dict())

# 加载最佳模型
model.load_state_dict(best_model)
```

### 步骤5: 评估

```python
# 测试
model.eval()
predictions = []
actuals = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        pred = model(X_batch)
        predictions.extend(pred.squeeze().numpy())
        actuals.extend(y_batch.numpy())

# 反标准化
predictions = scaler.inverse_transform(
    np.column_stack([predictions, np.zeros((len(predictions), X_train.shape[2]-1))])
)[:, 0]

actuals = scaler.inverse_transform(
    np.column_stack([actuals, np.zeros((len(actuals), X_train.shape[2]-1))])
)[:, 0]

# 计算指标
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(actuals, predictions)
mae = mean_absolute_error(actuals, predictions)
r2 = r2_score(actuals, predictions)

print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")
```

---

## 超参数调优

### 网格搜索

```python
from itertools import product

# 定义参数网格
param_grid = {
    'hidden_size': [32, 64, 128],
    'num_layers': [1, 2, 3],
    'dropout': [0.1, 0.2, 0.3],
    'learning_rate': [0.0001, 0.001, 0.01],
    'batch_size': [16, 32, 64]
}

# 生成所有参数组合
param_combinations = list(product(
    param_grid['hidden_size'],
    param_grid['num_layers'],
    param_grid['dropout'],
    param_grid['learning_rate'],
    param_grid['batch_size']
))

best_params = None
best_val_loss = float('inf')

for hidden_size, num_layers, dropout, lr, batch_size in param_combinations:
    print(f"Testing: hidden_size={hidden_size}, num_layers={num_layers}, dropout={dropout}, lr={lr}, batch_size={batch_size}")

    # 创建模型
    model = StockPredictionLSTM(
        input_size=X_train.shape[2],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 定义优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 训练（快速验证）
    for epoch in range(10):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions.squeeze(), y_batch)
            loss.backward()
            optimizer.step()

    # 验证
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            predictions = model(X_batch)
            loss = criterion(predictions.squeeze(), y_batch)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(f"  Val Loss: {val_loss:.4f}")

    # 更新最佳参数
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_params = {
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout,
            'learning_rate': lr,
            'batch_size': batch_size
        }

print(f"\nBest parameters: {best_params}")
print(f"Best validation loss: {best_val_loss:.4f}")
```

### 随机搜索

```python
import random

n_trials = 50

for trial in range(n_trials):
    # 随机选择参数
    hidden_size = random.choice([32, 64, 128, 256])
    num_layers = random.choice([1, 2, 3, 4])
    dropout = random.uniform(0.1, 0.5)
    learning_rate = random.choice([0.0001, 0.001, 0.01])
    batch_size = random.choice([16, 32, 64, 128])

    print(f"Trial {trial+1}/{n_trials}")

    # 训练和验证
    # ...
```

---

## 模型保存与加载

### 保存模型

```python
# 1. 保存整个模型（包含结构和参数）
torch.save(model, 'lstm_model.pth')

# 2. 只保存模型参数（推荐）
torch.save(model.state_dict(), 'lstm_model_state.pth')

# 3. 保存检查点（包含优化器状态）
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'best_val_loss': best_val_loss
}
torch.save(checkpoint, 'checkpoint.pth')
```

### 加载模型

```python
# 1. 加载整个模型
model = torch.load('lstm_model.pth')
model.eval()  # 设置为评估模式

# 2. 加载模型参数（需要先定义模型）
model = StockPredictionLSTM(
    input_size=X_train.shape[2],
    hidden_size=64,
    num_layers=2,
    dropout=0.2
)
model.load_state_dict(torch.load('lstm_model_state.pth'))
model.eval()

# 3. 加载检查点
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
best_val_loss = checkpoint['best_val_loss']
```

### 最佳实践

```python
# 训练时保存最佳模型
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # 训练和验证
    val_loss = validate(model, val_loader)

    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"Saved best model with val_loss: {val_loss:.4f}")

    # 定期保存检查点
    if (epoch + 1) % 10 == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss
        }
        torch.save(checkpoint, f'checkpoint_epoch_{epoch+1}.pth')
```

---

## 评估指标

### 回归指标

#### 1. MSE（均方误差）

```python
mse = torch.mean((predictions - targets) ** 2)
print(f"MSE: {mse.item():.4f}")
```

#### 2. MAE（平均绝对误差）

```python
mae = torch.mean(torch.abs(predictions - targets))
print(f"MAE: {mae.item():.4f}")
```

#### 3. RMSE（均方根误差）

```python
rmse = torch.sqrt(torch.mean((predictions - targets) ** 2))
print(f"RMSE: {rmse.item():.4f}")
```

#### 4. R²（决定系数）

```python
ss_res = torch.sum((targets - predictions) ** 2)
ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
r2 = 1 - ss_res / ss_tot
print(f"R²: {r2.item():.4f}")
```

### 量化投资专用指标

#### 1. IC（信息系数）

```python
from scipy.stats import spearmanr

ic, _ = spearmanr(predictions.numpy(), targets.numpy())
print(f"IC: {ic:.4f}")
```

#### 2. ICIR（信息系数信息比率）

```python
# 计算多期IC
ic_values = []
for i in range(n_periods):
    ic, _ = spearmanr(preds[i], targets[i])
    ic_values.append(ic)

icir = np.mean(ic_values) / np.std(ic_values)
print(f"ICIR: {icir:.4f}")
```

#### 3. MAPE（平均绝对百分比误差）

```python
mape = torch.mean(torch.abs((targets - predictions) / targets)) * 100
print(f"MAPE: {mape.item():.2f}%")
```

### 完整评估

```python
def evaluate_model(model, test_loader, scaler):
    """
    评估模型

    参数:
        model: 模型
        test_loader: 测试数据加载器
        scaler: 标准化器

    返回:
        评估结果字典
    """
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            pred = model(X_batch)
            predictions.extend(pred.squeeze().numpy())
            actuals.extend(y_batch.numpy())

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # 反标准化
    predictions_original = scaler.inverse_transform(
        np.column_stack([predictions, np.zeros((len(predictions), X_train.shape[2]-1))])
    )[:, 0]

    actuals_original = scaler.inverse_transform(
        np.column_stack([actuals, np.zeros((len(actuals), X_train.shape[2]-1))])
    )[:, 0]

    # 计算指标
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from scipy.stats import spearmanr

    results = {
        'MSE': mean_squared_error(actuals_original, predictions_original),
        'MAE': mean_absolute_error(actuals_original, predictions_original),
        'RMSE': np.sqrt(mean_squared_error(actuals_original, predictions_original)),
        'R²': r2_score(actuals_original, predictions_original),
        'IC': spearmanr(predictions_original, actuals_original)[0],
        'MAPE': np.mean(np.abs((actuals_original - predictions_original) / actuals_original)) * 100
    }

    return results, predictions_original, actuals_original

# 评估
results, preds, actuals = evaluate_model(model, test_loader, scaler)

for metric, value in results.items():
    print(f"{metric}: {value:.4f}")
```

---

## LSTM vs LightGBM

### 对比维度

| 维度 | LSTM | LightGBM |
|------|------|----------|
| **数据需求** | 大量 | 中等 |
| **训练时间** | 长 | 短 |
| **特征工程** | 少 | 多 |
| **可解释性** | 低 | 高 |
| **长期依赖** | 优秀 | 差 |
| **过拟合风险** | 高 | 中 |
| **推理速度** | 中 | 快 |
| **GPU支持** | ✅ | ❌ |
| **学习曲线** | 慢 | 快 |

### 性能对比实验

```python
# LSTM模型
lstm_model = StockPredictionLSTM(
    input_size=X_train.shape[2],
    hidden_size=64,
    num_layers=2,
    dropout=0.2
)

# 训练LSTM
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)

for epoch in range(50):
    lstm_model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predictions = lstm_model(X_batch)
        loss = criterion(predictions.squeeze(), y_batch)
        loss.backward()
        optimizer.step()

# LightGBM模型
import lightgbm as lgb

# 展平数据
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# 训练
lgb_model = lgb.LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6
)
lgb_model.fit(X_train_flat, y_train)

# 预测
lstm_preds = lstm_model(torch.FloatTensor(X_test)).squeeze().numpy()
lgb_preds = lgb_model.predict(X_test_flat)

# 对比
lstm_mse = np.mean((lstm_preds - y_test) ** 2)
lgb_mse = np.mean((lgb_preds - y_test) ** 2)

print(f"LSTM MSE: {lstm_mse:.4f}")
print(f"LightGBM MSE: {lgb_mse:.4f}")
```

### 选择建议

**选择LSTM**:
- 数据量大
- 长期依赖重要
- 特征工程少
- 需要捕捉复杂模式

**选择LightGBM**:
- 数据量中等
- 需要快速迭代
- 可解释性重要
- 计算资源有限

---

## 最佳实践

### 数据准备

#### 1. 数据质量检查
```python
# 检查缺失值
print(data.isnull().sum())

# 检查异常值
print(data.describe())

# 处理缺失值
data = data.fillna(method='ffill')
```

#### 2. 特征选择
```python
# 选择与目标相关的特征
from sklearn.feature_selection import SelectKBest, f_regression

selector = SelectKBest(f_regression, k=10)
X_selected = selector.fit_transform(X, y)
```

#### 3. 时间序列划分
```python
# 严格按时间顺序划分
# 避免未来函数
# 保留样本外数据
```

### 模型设计

#### 1. 模型复杂度
```python
# 从简单模型开始
# 逐步增加复杂度
# 避免过度复杂
```

#### 2. 超参数选择
```python
# hidden_size: 32-128
# num_layers: 1-3
# dropout: 0.1-0.3
# learning_rate: 0.0001-0.01
```

#### 3. 正则化
```python
# 使用Dropout防止过拟合
# 使用BatchNorm加速训练
# 使用L1/L2正则化
```

### 训练策略

#### 1. 早停
```python
# 监控验证集损失
# 防止过拟合
# 节省训练时间
```

#### 2. 学习率调度
```python
# 初始学习率较大
# 逐渐降低学习率
# 使用学习率衰减
```

#### 3. 梯度裁剪
```python
# 防止梯度爆炸
# 稳定训练过程
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 评估与验证

#### 1. 多指标评估
```python
# 不要只看一个指标
# 综合考虑多个指标
# 关注风险调整收益
```

#### 2. 样本外验证
```python
# 保留一部分数据不参与训练
# 严格验证泛化能力
# 避免过拟合
```

#### 3. 稳定性测试
```python
# 在不同时间段测试
# 检查参数稳定性
# 验证鲁棒性
```

### 常见问题

#### Q1: LSTM过拟合怎么办？

**A**: 多种方法结合:
- 增加Dropout比例
- 减少模型复杂度
- 增加训练数据
- 使用更强的正则化

#### Q2: LSTM训练很慢怎么办？

**A**: 优化训练速度:
- 使用GPU
- 减小batch_size
- 减少模型复杂度
- 使用更快的优化器

#### Q3: LSTM vs LightGBM如何选择？

**A**: 根据实际情况:
- 数据量大、长期依赖重要：LSTM
- 需要快速迭代、可解释性重要：LightGBM
- 两者都试，选择更好的

#### Q4: 如何确定序列长度？

**A**: 通过实验确定:
- 尝试不同的序列长度（10, 20, 30, 60）
- 使用验证集选择最优长度
- 考虑业务逻辑（如一周、一月）

---

## 核心知识点总结

### 完整预测流程
- ✅ 数据准备
- ✅ 模型定义
- ✅ Dataset和DataLoader
- ✅ 训练
- ✅ 评估

### 超参数调优
- ✅ 网格搜索
- ✅ 随机搜索
- ✅ 贝叶斯优化

### 模型保存与加载
- ✅ 保存模型
- ✅ 加载模型
- ✅ 最佳实践

### 评估指标
- ✅ 回归指标
- ✅ 量化投资指标
- ✅ 完整评估

### LSTM vs LightGBM
- ✅ 对比维度
- ✅ 性能对比
- ✅ 选择建议

### 最佳实践
- ✅ 数据准备
- ✅ 模型设计
- ✅ 训练策略
- ✅ 评估验证

---

## 附录

### 专业术语表

| 术语 | 英文 | 解释 |
|------|------|------|
| RNN | Recurrent Neural Network | 循环神经网络 |
| LSTM | Long Short-Term Memory | 长短期记忆网络 |
| GRU | Gated Recurrent Unit | 门控循环单元 |
| IC | Information Coefficient | 信息系数 |
| ICIR | IC Information Ratio | 信息系数信息比率 |
| Early Stopping | 早停 | 防止过拟合的技术 |
| Dropout | 丢弃 | 随机丢弃神经元 |

### 推荐学习资源

**书籍**:
- 《深度学习》（Goodfellow等）
- 《动手学深度学习》
- 《Python深度学习》

**在线课程**:
- Coursera: Deep Learning Specialization
- Fast.ai: Practical Deep Learning for Coders

**文档和教程**:
- PyTorch官方文档：https://pytorch.org/docs/
- PyTorch教程：https://pytorch.org/tutorials/

---

**系列文档结束**

祝学习顺利！🎓
