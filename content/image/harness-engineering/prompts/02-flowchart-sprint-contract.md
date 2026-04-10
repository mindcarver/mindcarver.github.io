---
illustration_id: 02
type: flowchart
style: notion
---

Sprint Contract 机制 — 从模糊意图到可验证目标

Minimalist hand-drawn line art flowchart. Horizontal left-to-right flow with emphasis on the central transformation step. Soft rounded shapes, thin black strokes, muted pastel fills on clean white background.

STRUCTURE: 4 nodes left to right with one emphasized center node

NODE 1 (left) — 高层 Spec:
- Rounded rectangle, wider and taller
- Label: "Planner 输出"
- Content lines: "模糊的产品意图", "目标与范围", "产品深度定义"
- Fill: light mint (#B5E5CF)
- Icon: cloud/abstract thought bubble

NODE 2 (center, EMPHASIZED) — Sprint Contract:
- Large dashed-border rounded rectangle, visually prominent
- Label: "Sprint Contract (中间工件)"
- Content lines: "明确的交付物定义", "可测试的验收标准", "实现范围收束"
- Annotation below: "把创造空间收束成可验证目标"
- Fill: light yellow (#FFF3B0) with thick dashed border
- Icon: contract/document with checkmark

NODE 3 (right-top) — Generator 实现:
- Rounded rectangle
- Label: "Generator"
- Content: "按 Contract 实现", "局部闭环", "任务切片"
- Fill: light blue (#A8D8EA)
- Icon: code brackets

NODE 4 (right-bottom) — Evaluator 验收:
- Rounded rectangle
- Label: "Evaluator"
- Content: "按 Contract 验收", "Playwright 交互测试", "阈值检查"
- Fill: light coral (#FFD5C2)
- Icon: magnifying glass

ARROWS:
- Node 1 → Node 2: solid arrow labeled "扩展"
- Node 2 → Node 3: solid arrow labeled "明确交付物"
- Node 2 → Node 4: dashed arrow labeled "验收标准" (evaluator references contract directly)
- Node 3 → Node 4: solid arrow labeled "提交"
- Node 4 → Node 3: curved dashed arrow labeled "不通过"

BOTTOM NOTE: Small text box — "没有 Sprint Contract，Generator 和 Evaluator 各说各话"

LABELS: "高层 Spec", "Sprint Contract", "交付物", "验收标准", "Playwright", "返工", "模糊意图", "可验证目标"
STYLE: Hand-drawn line art, soft rounded rectangles, thin black strokes, muted pastel fills. Center node (Sprint Contract) visually emphasized with larger size, dashed border, and yellow fill. Doodled icons. Clean white background. Minimal decoration.
Watermark: "链上无名" positioned at bottom-right.
Clean composition with generous white space. Simple background. Left-to-right flow with center emphasis.
ASPECT: 16:9
