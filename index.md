---
layout: default
---

# 量化投资技术文档

欢迎来到量化投资技术文档站，专注于特征工程、LightGBM模型和回测引擎的实践与探索。

---

<div class="card">
  <h2 style="margin-top: 0;">📊 快速导航</h2>
  <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; margin-top: 1rem;">
    <div style="padding: 1rem; background: white; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
      <strong style="color: #667eea; font-size: 1.2rem;">📚 特征工程</strong>
      <p style="margin: 0.5rem 0;">Qlib特征工程的系统讲解</p>
      <a href="特征工程/" style="color: #667eea;">进入模块 →</a>
    </div>
    <div style="padding: 1rem; background: white; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
      <strong style="color: #667eea; font-size: 1.2rem;">⚡ LightGBM</strong>
      <p style="margin: 0.5rem 0;">机器学习在量化中的应用</p>
      <a href="LightGBM/" style="color: #667eea;">进入模块 →</a>
    </div>
    <div style="padding: 1rem; background: white; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
      <strong style="color: #667eea; font-size: 1.2rem;">📊 回测引擎</strong>
      <p style="margin: 0.5rem 0;">策略回测与绩效评估</p>
      <a href="backtest/" style="color: #667eea;">进入模块 →</a>
    </div>
    <div style="padding: 1rem; background: white; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
      <strong style="color: #667eea; font-size: 1.2rem;">🚀 快速开始</strong>
      <p style="margin: 0.5rem 0;">新手入门指南</p>
      <a href="快速开始.md" style="color: #667eea;">开始学习 →</a>
    </div>
  </div>
</div>

---

## 📚 文档导航

### 特征工程模块

[→ 进入特征工程文档](特征工程/)

系统讲解Qlib特征工程的核心概念与实践方法：

- [01-qlib特征工程全景概览](特征工程/01-qlib特征工程全景概览.md)
- [02-horizon对齐详解](特征工程/02-horizon对齐详解.md)
- [03-横截面标准化与中性化](特征工程/03-横截面标准化与中性化.md)
- [04-相对强弱预测的量化思维](特征工程/04-相对强弱预测的量化思维.md)
- [05-qlib特征工程实践指南](特征工程/05-qlib特征工程实践指南.md)

---

### LightGBM模块

[→ 进入LightGBM文档](LightGBM/)

深入学习LightGBM在量化投资中的应用：

- [01-Gradient Boosting原理](LightGBM/01-Gradient-Boosting原理.md) - GOSS、EFB、Leaf-wise三大创新
- [02-时序数据划分](LightGBM/02-时序数据划分.md) - 因果性约束、交叉验证、滚动窗口
- [03-模型训练](LightGBM/03-模型训练.md) - IC优化、在线学习、分布式训练
- [04-IC-Rank-IC评估指标](LightGBM/04-IC-Rank-IC评估指标.md) - 统计检验、时序分析、IR指标
- [05-特征重要性分析](LightGBM/05-特征重要性分析.md) - Permutation、SHAP、稳定性分析
- [06-学习检查清单](LightGBM/06-学习检查清单.md) - 学习目标与实践建议

---

### 回测引擎模块

[→ 进入回测引擎文档](backtest/)

完整讲解策略回测、投资组合构建、绩效评估：

- [01-交易策略理论](backtest/01-交易策略理论.md) - Top-K、IC权重、MV优化
- [02-投资组合构建方法](backtest/02-投资组合构建方法.md) - 三种组合构建方法对比
- [03-Executor与成本模型](backtest/03-Executor与成本模型.md) - 交易成本与执行机制
- [04-绩效评估指标](backtest/04-绩效评估指标.md) - 收益、风险、绩效指标
- [05-实验分析方法](backtest/05-实验分析方法.md) - 参数敏感性、样本外验证
- [06-回测流程与实践](backtest/06-回测流程与实践.md) - Qlib回测框架
- [07-学习检查清单](backtest/07-学习检查清单.md) - 学习目标与实践建议

---

### AI辅助策略研究

- [AI辅助策略研究](zlyq/AI 辅助策略研究.md)
- [ResearchAgent](zlyq/ResearchAgent.md)
- [提示词](zlyq/提示词.md)

---

## 🎯 学习路径

### 新手入门

```
特征工程基础 → LightGBM原理 → 模型训练 → 投资组合构建 → 策略回测
```

适合初学者，从基础概念开始，逐步学习完整的量化投资流程。

### 进阶提升

```
时序数据划分 → IC优化训练 → 特征重要性分析 → 成本模型 → 绩效评估
```

适合有一定基础的学习者，深入理解和优化各个环节。

### 实战应用

```
从实际项目出发 → 遇到问题查文档 → 理论学习 → 实践应用
```

适合有经验的开发者，通过解决实际问题提升技能。

---

## 💡 学习建议

1. **循序渐进**：按照建议的学习路径逐步学习，不要跳过基础概念
2. **动手实践**：每个模块都包含代码示例，建议动手运行和修改
3. **理解原理**：不仅要会用，还要理解背后的原理
4. **多维度思考**：从不同角度理解问题，如风险、收益、成本等
5. **持续优化**：量化投资是一个持续优化的过程，不要满足于一次性结果

---

<div class="card" style="text-align: center; margin-top: 2rem;">
  <h3 style="margin-top: 0;">💡 技术支持</h3>
  <p style="margin-bottom: 1rem;">文档支持数学公式渲染（使用 KaTeX）和代码高亮，适合在线阅读和本地开发参考</p>
  <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
    <a href="快速开始.md" style="background: #667eea; color: white; padding: 0.8rem 1.5rem; border-radius: 5px; text-decoration: none;">快速开始</a>
    <a href="站点导航.md" style="background: #764ba2; color: white; padding: 0.8rem 1.5rem; border-radius: 5px; text-decoration: none;">查看导航</a>
    <a href="关于.md" style="background: #667eea; color: white; padding: 0.8rem 1.5rem; border-radius: 5px; text-decoration: none;">关于我们</a>
  </div>
</div>
