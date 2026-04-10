---
illustration_id: 01
type: framework
style: notion
---

Harness Engineering 三层职责分离架构

Minimalist hand-drawn line art framework diagram. Vertical layered layout with three main zones connected by arrows. Soft rounded rectangles, thin black strokes, muted pastel fills on clean white background.

LAYOUT: Top-to-bottom flow

ZONE 1 (top) — Planner:
- Label: "Planner — 扩写产品意图"
- Sub-labels: "高层 Spec", "留出创造空间", "不过度规定实现细节"
- Icon: lightbulb/compass
- Fill: light mint (#B5E5CF)

MIDDLE BRIDGE — Sprint Contract:
- Prominent dashed-line rounded rectangle spanning width
- Label: "Sprint Contract"
- Sub-labels: "这轮交什么", "怎么验证完成", "可测试的验收标准"
- Icon: handshake/document
- Fill: light yellow (#FFF3B0) with dashed border

ZONE 2 (bottom-left) — Generator:
- Label: "Generator — 实现"
- Sub-labels: "按 Sprint 推进", "局部闭环", "任务切片控制 coherence"
- Icon: gear/code brackets
- Fill: light blue (#A8D8EA)

ZONE 3 (bottom-right) — Evaluator:
- Label: "Evaluator — 验收"
- Sub-labels: "Playwright 真实交互", "UI + API + DB 检查", "低于阈值则返工"
- Icon: magnifying glass/checkmark
- Fill: light coral (#FFD5C2)

ARROWS:
- Planner → Sprint Contract (solid, "扩展意图 → 收束目标")
- Sprint Contract → Generator (solid, "明确交付物")
- Sprint Contract → Evaluator (solid, "验收标准")
- Generator → Evaluator (solid, "提交成果")
- Evaluator → Generator (dashed, "不通过 → 返工", curved arrow back)

LABELS: "Planner", "Generator", "Evaluator", "Sprint Contract", "高层 Spec", "可测试目标", "返工", "Playwright", "coherence"
STYLE: Hand-drawn line art, soft rounded shapes, thin black strokes, muted pastel fills. Doodled icons in each zone. Clean white background. Sprint contract zone emphasized with dashed border. Minimal decoration.
Watermark: "链上无名" positioned at bottom-right.
Clean composition with generous white space. Simple background. Centered vertical layout.
ASPECT: 9:16
