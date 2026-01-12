# Qlib 量化投资学习路径

> 系统学习量化投资，从特征工程到深度学习模型

---

## 📚 学习概览

本模块提供系统化的Qlib量化投资学习路径，涵盖从基础特征工程到高级深度学习模型的完整知识体系。

---

## 🎯 学习路径

### 推荐学习顺序

```
Week 1: 特征工程基础
    ↓
Week 2: LightGBM模型
    ↓
Week 3: 回测系统
    ↓
Week 5: LSTM深度学习
```

---

## 📖 课程内容

### Week 1 - 特征工程 📊

系统讲解Qlib特征工程的核心概念与实践方法。

**文档列表**:
- [01-qlib特征工程全景概览](week1/01-qlib特征工程全景概览.md)
- [02-horizon对齐详解](week1/02-horizon对齐详解.md)
- [03-横截面标准化与中性化](week1/03-横截面标准化与中性化.md)
- [04-相对强弱预测的量化思维](week1/04-相对强弱预测的量化思维.md)
- [05-qlib特征工程实践指南](week1/05-qlib特征工程实践指南.md)

**学习目标**:
- ✅ 理解Qlib特征工程的核心概念
- ✅ 掌握horizon对齐方法
- ✅ 学会横截面标准化与中性化
- ✅ 培养相对强弱预测的量化思维
- ✅ 能够独立完成特征工程实践

**预计时间**: 5-6小时（4-5天）

---

### Week 2 - LightGBM ⚡

深入学习LightGBM在量化投资中的应用。

**文档列表**:
- [01-Gradient Boosting原理](week2/01-Gradient-Boosting原理.md) - GOSS、EFB、Leaf-wise三大创新
- [02-时序数据划分](week2/02-时序数据划分.md) - 因果性约束、交叉验证、滚动窗口
- [03-模型训练](week2/03-模型训练.md) - IC优化、在线学习、分布式训练
- [04-IC-Rank-IC评估指标](week2/04-IC-Rank-IC评估指标.md) - 统计检验、时序分析、IR指标
- [05-特征重要性分析](week2/05-特征重要性分析.md) - Permutation、SHAP、稳定性分析
- [06-学习检查清单](week2/06-学习检查清单.md) - 学习目标与实践建议

**学习目标**:
- ✅ 理解Gradient Boosting原理
- ✅ 掌握时序数据划分方法
- ✅ 学会模型训练与优化
- ✅ 能够使用IC/Rank-IC评估模型
- ✅ 掌握特征重要性分析方法

**预计时间**: 6-8小时（5-7天）

---

### Week 3 - 回测系统 📈

完整讲解策略回测、投资组合构建、绩效评估。

**文档列表**:
- [01-交易策略理论](week3/01-交易策略理论.md) - Top-K、IC权重、MV优化
- [02-投资组合构建方法](week3/02-投资组合构建方法.md) - 三种组合构建方法对比
- [03-Executor与成本模型](week3/03-Executor与成本模型.md) - 交易成本与执行机制
- [04-绩效评估指标](week3/04-绩效评估指标.md) - 收益、风险、绩效指标
- [05-实验分析方法](week3/05-实验分析方法.md) - 参数敏感性、样本外验证
- [06-回测流程与实践](week3/06-回测流程与实践.md) - Qlib回测框架
- [07-学习检查清单](week3/07-学习检查清单.md) - 学习目标与实践建议

**学习目标**:
- ✅ 理解交易策略理论
- ✅ 掌握投资组合构建方法
- ✅ 了解交易成本模型
- ✅ 能够计算绩效评估指标
- ✅ 学会实验分析方法
- ✅ 能够完成完整的回测流程

**预计时间**: 7-9小时（6-8天）

---

### Week 5 - LSTM深度学习 🧠

深入学习LSTM神经网络，掌握时序数据的深度学习方法。

**文档系列**:

#### 1. 基础理论系列
- [基础理论系列](week5/01_基础理论系列/)
  - 深度学习基础、RNN原理、LSTM原理
  - LSTM vs RNN vs GRU对比

#### 2. PyTorch框架系列
- [PyTorch框架系列](week5/02_PyTorch框架系列/)
  - Tensor操作、Autograd、nn.Module
  - 常用层（LSTM、Linear、Dropout）

#### 3. LSTM模型构建系列
- [LSTM模型构建系列](week5/03_LSTM模型构建系列/)
  - 单层、多层、双向LSTM
  - LSTM变体与超参数选择

#### 4. 时序数据处理系列
- [时序数据处理系列](week5/04_时序数据处理系列/)
  - 滑动窗口、数据划分、特征标准化
  - PyTorch Dataset与DataLoader

#### 5. 模型训练优化系列
- [模型训练优化系列](week5/05_模型训练优化系列/)
  - 损失函数、优化器、训练循环
  - 早停策略、正则化、学习率调度

#### 6. 实战应用系列
- [实战应用系列](week5/06_实战应用系列/)
  - 完整预测流程、超参数调优
  - 模型保存加载、评估指标
  - LSTM vs LightGBM对比、最佳实践

**学习目标**:
- ✅ 理解深度学习和LSTM原理
- ✅ 掌握PyTorch框架
- ✅ 能够构建LSTM模型
- ✅ 学会时序数据处理
- ✅ 掌握模型训练与优化
- ✅ 能够完成完整的实战应用

**预计时间**: 8-10小时（5-6天）

---

## 🛠️ 技术栈

- **Python**: 主要编程语言
- **Qlib**: 量化投资平台
- **PyTorch**: 深度学习框架
- **LightGBM**: 梯度提升树
- **Pandas/NumPy**: 数据处理
- **Matplotlib**: 数据可视化

---

## 💡 学习建议

### 循序渐进
按照推荐的学习顺序逐步学习，不要跳过基础概念。

### 动手实践
每个模块都包含代码示例，建议动手运行和修改。

### 理解原理
不仅要会用，还要理解背后的原理。

### 多维度思考
从不同角度理解问题，如风险、收益、成本等。

### 持续优化
量化投资是一个持续优化的过程，不要满足于一次性结果。

---

## 📚 扩展阅读

### 推荐书籍

1. 《量化投资：策略与技术》
2. 《Python金融大数据分析》
3. 《统计套利》
4. 《深度学习》（Ian Goodfellow）
5. 《动手学深度学习》

### 在线课程

- Coursera: Deep Learning Specialization
- Fast.ai: Practical Deep Learning for Coders
- edX: Machine Learning

---

## 🤝 交流与反馈

如有问题或建议，欢迎通过以下方式联系：

- 📧 Email: [待添加]
- 💻 GitHub: [待添加]
- 💬 微信: [待添加]

---

<div align="center" style="margin-top: 2rem; padding: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 8px;">
  <h3 style="margin: 0;">开始学习之旅</h3>
  <p style="margin: 0.5rem 0 1.5rem 0; opacity: 0.9;">从特征工程开始，系统学习量化投资</p>
  <a href="week1/" style="background: white; color: #667eea; padding: 0.8rem 1.5rem; border-radius: 5px; text-decoration: none; font-weight: 500;">开始学习 Week 1 →</a>
</div>

---

<div align="center" style="margin-top: 2rem; padding-top: 2rem; border-top: 1px solid #eee; color: #999; font-size: 0.9rem;">
  <p style="margin: 0;">持续更新中，欢迎收藏和分享！</p>
</div>
