---
illustration_id: 01
type: framework
style: notion
---

完整量化回测流水线 - 概念框架图

Minimalist hand-drawn line art framework diagram. Soft rounded rectangles for each module, thin connecting arrows, neutral color palette on clean white background. Doodled icons inside each module.

STRUCTURE: horizontal pipeline, 6 nodes left to right

NODES:
1. 数据生成 — 200只股票 × 3年日频 × 8行业 — icon: table/grid
2. Alpha模型 — LightGBM训练 → 预测信号 — icon: brain/neural
3. 组合构建 — Top-K等权 / IC加权 — icon: pie chart
4. 执行模拟 — T+1限制 + 交易成本 — icon: gear/settings
5. 绩效评估 — Sharpe / 回撤 / 胜率 — icon: chart/metrics
6. 对比分析 — 成本影响 / 调仓频率 — icon: split compare

RELATIONSHIPS:
- Sequential flow: 1 → 2 → 3 → 4 → 5 → 6
- Feedback loop from 6 back to 3 (优化迭代)

LABELS: "200只股票", "3年日频", "8行业", "LightGBM", "Top-K等权", "IC加权", "T+1", "Sharpe", "最大回撤", "胜率"
STYLE: Hand-drawn line art, soft rounded shapes, thin black strokes, muted pastel fills (light blue, light mint, light peach), clean white background. Minimal decoration. Simple hand-drawn arrows connecting nodes.
Watermark: "链上无名" positioned at bottom-right.
Clean composition with generous white space. Simple background. Main elements centered horizontally.
ASPECT: 16:9
