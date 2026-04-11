---
illustration_id: 01
type: flowchart
style: notion
---

Monitor 工作流程 — 从用户指令到自动排查

Minimalist hand-drawn line art flowchart. Horizontal left-to-right flow with 5 nodes, emphasizing the event-driven wake-up moment. Soft rounded shapes, thin black strokes, muted pastel fills on clean white background.

STRUCTURE: 5 nodes left to right with one emphasized trigger node

NODE 1 (left) — User Prompt:
- Rounded rectangle
- Label: "用户"
- Content: "帮我盯着日志"
- Fill: light mint (#B5E5CF)
- Icon: speech bubble

NODE 2 (center-left) — Claude Spawns Monitor:
- Rounded rectangle
- Label: "Claude Code"
- Content: "收到，派 Monitor 去盯着" / "自己接着干别的"
- Fill: light blue (#A8D8EA)
- Icon: robot/agent with arrow splitting into two paths

NODE 3 (center, EMPHASIZED) — Background Monitor:
- Large dashed-border rounded rectangle, visually prominent
- Label: "Monitor (后台)"
- Content: "tail -f app.log | grep ERROR" / "安安静静跑着"
- Annotation: "持久后台任务，不影响对话"
- Fill: light yellow (#FFF3B0) with thick dashed border
- Icon: eye/watchful

NODE 4 (center-right, TRIGGER) — Error Detected:
- Rounded rectangle with red accent border
- Label: "事件触发"
- Content: "ERROR db: connection refused" / "Monitor event → 唤醒 Claude"
- Fill: light coral (#FFD5C2) with red border
- Icon: alert/exclamation mark
- Star/burst decoration to emphasize this is the key moment

NODE 5 (right) — Claude Investigates:
- Rounded rectangle
- Label: "Claude 自动排查"
- Content: "检测到错误，正在调查" / "Postgres 连接失败原因"
- Fill: light blue (#A8D8EA)
- Icon: magnifying glass with gear

ARROWS:
- Node 1 → Node 2: solid arrow labeled "指令"
- Node 2 → Node 3: solid arrow labeled "派生"
- Node 2 also has a dashed arrow going down labeled "继续对话" (showing Claude keeps working)
- Node 3 → Node 4: solid arrow, labeled "检测到错误" (this is the key event arrow, slightly thicker)
- Node 4 → Node 5: solid arrow with burst effect, labeled "唤醒"

TOP ANNOTATION: Small text box — "polling → event-driven：不轮询，有事才醒"

BOTTOM NOTE: "人类全程不需要碰键盘"

LABELS: "用户", "Claude Code", "Monitor", "后台", "事件触发", "ERROR", "唤醒", "自动排查", "tail -f", "grep", "Postgres", "polling", "event-driven", "继续对话"
STYLE: Hand-drawn line art, soft rounded rectangles, thin black strokes, muted pastel fills. Node 3 (Monitor) visually emphasized with dashed border and yellow fill. Node 4 (trigger) has red accent. Doodled icons. Clean white background.
Watermark: "链上无名" positioned at bottom-right.
Clean composition with generous white space. Left-to-right flow with emphasis on the trigger moment.
ASPECT: 16:9
