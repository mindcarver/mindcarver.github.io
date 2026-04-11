---
illustration_id: 02
type: comparison
style: notion
---

/loop vs /schedule vs Monitor — 三兄弟对比

Minimalist hand-drawn line art comparison diagram. Three-column layout with character cards. Soft rounded shapes, thin black strokes, muted pastel fills on clean white background.

LAYOUT: Three vertical cards side by side with a comparison table below

CARD 1 (left) — /loop 闹钟:
- Title: "/loop"
- Subtitle: "闹钟"
- Icon: alarm clock (hand-drawn)
- Trigger: "定时轮询" — "每 5 分钟醒一次，不管有没有事"
- Runtime: "本地终端，关了就没"
- Token cost: "高" — "1 小时 12 次调用，没事也跑"
- Best for: "短期高频巡检"
- Fill: light blue (#A8D8EA)
- Metaphor drawing: small doodle of a security guard patrolling

CARD 2 (center) — /schedule 管家:
- Title: "/schedule"
- Subtitle: "管家"
- Icon: butler/housekeeper (hand-drawn)
- Trigger: "定时任务" — "每天/每周到点干活"
- Runtime: "Anthropic 云端，关电脑也跑"
- Token cost: "中" — "频率低但每次跑完整流程"
- Best for: "周期性维护任务"
- Fill: light mint (#B5E5CF)
- Metaphor drawing: small doodle of a cleaner with schedule

CARD 3 (right, EMPHASIZED) — Monitor 报警器:
- Title: "Monitor"
- Subtitle: "报警器"
- Icon: smoke alarm (hand-drawn)
- Trigger: "事件驱动" — "平时不醒，出事了才叫"
- Runtime: "本地后台，持续运行"
- Token cost: "低" — "一天可能不触发，触发即关键"
- Best for: "出错需要处理"
- Fill: light coral (#FFD5C2)
- Metaphor drawing: small doodle of smoke alarm with alert waves
- Slightly larger or with subtle glow to emphasize

BOTTOM COMPARISON ROW:
- Row label: "触发方式"
- /loop: "定时"
- /schedule: "定时"
- Monitor: "事件" (bold/distinct)

- Row label: "空闲消耗"
- /loop: "浪费"
- /schedule: "中等"
- Monitor: "接近零"

BOTTOM NOTE: "问题类型决定工具选择：定时做什么 → /loop 或 /schedule；出事了怎么办 → Monitor"

LABELS: "/loop", "/schedule", "Monitor", "闹钟", "管家", "报警器", "定时轮询", "定时任务", "事件驱动", "本地", "云端", "后台", "Token", "高", "中", "低", "浪费", "接近零", "巡检", "维护", "出错"
STYLE: Hand-drawn line art, soft rounded rectangle cards in row layout, thin black strokes, muted pastel fills per card. Right card (Monitor) slightly emphasized. Doodled icons and metaphor drawings. Clean white background.
Watermark: "链上无名" positioned at bottom-right.
Clean composition with generous white space. Three-column centered layout.
ASPECT: 16:9
