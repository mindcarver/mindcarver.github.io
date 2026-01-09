---
layout: default
---

# 量化投资技术文档

欢迎来到量化投资技术文档站，专注于特征工程和LightGBM模型的实践与探索。

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

---

## 🎯 文档特色

- **系统性强**：从基础原理到实践应用的完整知识体系
- **实战导向**：包含大量代码示例和最佳实践
- **量化特色**：针对量化投资场景专门设计的内容
- **持续更新**：跟随最新技术发展不断补充完善
- **易于阅读**：优美的页面设计和清晰的文档结构

---

## 📖 快速开始

1. 如果你是新手，先阅读 [快速开始指南](快速开始.md)
2. 如果想了解所有文档，查看 [站点导航](站点导航.md)
3. 如果刚接触量化，建议从**特征工程**模块开始
4. 如果你想了解机器学习在量化中的应用，可以从**LightGBM**模块入手
5. 每个文档都包含理论讲解和代码示例，便于实践

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
