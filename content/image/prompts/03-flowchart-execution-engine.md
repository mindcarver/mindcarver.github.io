---
illustration_id: 03
type: flowchart
style: notion
---

回测引擎执行流程 — T+1 延迟与成本扣减

Minimalist hand-drawn line art flowchart. Top-down vertical layout. Clear step indicators with soft rounded rectangles, simple arrow connections, minimal decoration. Focus on process clarity. Thin black strokes, muted pastel fills on clean white background.

STEPS:
1. 当日信号生成 — Alpha模型输出预测值 → 目标权重向量
2. T+1 延迟执行 — 今日信号 → 明日开盘执行 (dashed border, highlighted as key constraint)
3. 计算目标持仓 — 目标权重 × 当前组合价值 / 当前价格 → 目标股数
4. 计算调仓量 — 目标股数 - 当前持仓 = 需交易股数
5. 分类交易方向 — 买入 (trade_shares > 0) / 卖出 (trade_shares < 0)
6. 扣减交易成本:
   - 买入: 佣金(万3) + 滑点(0.1%)
   - 卖出: 佣金(万3) + 印花税(千1) + 滑点(0.1%)
   - 双边费率合计: ~0.35%
7. 更新状态 — 持仓 ← 目标持仓, 现金 ← 余额 - 成本
8. 记录组合价值 — 每日净值 → 收益序列

CONNECTIONS: Bold downward arrows between steps. Step 2 has a "⚠️" annotation callout. Step 6 branches into two sub-paths (买/卖) that merge at step 7.

LABELS: "T+1", "万3", "千1", "0.1%", "0.35%", "目标持仓", "当前持仓", "组合净值"
STYLE: Hand-drawn line art, soft rounded rectangles, thin black strokes. Step 2 highlighted with dashed border and light yellow fill. Cost step 6 highlighted with light red fill. Other steps in light blue/mint fills. Simple hand-drawn arrows. Clean white background.
Watermark: "链上无名" positioned at bottom-right.
Clean composition with generous white space. Simple background. Vertically centered flow.
ASPECT: 9:16
