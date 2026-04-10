---
type: mixed
density: balanced
style: notion
palette: default
image_count: 4
article: content/agent/claudecode/claude code harness engineering 实战.md
---

## Illustration 1
**Position**: Section 三 (三层职责分离, after "这不是为了形式好看..." paragraph)
**Purpose**: 将抽象的三 agent 架构具象化，展示 planner/generator/evaluator 的职责边界和 sprint contract 的桥梁作用
**Visual Content**: 框架图，三层结构：顶部 planner（扩写产品意图）→ 中间 sprint contract（可验证目标对齐）→ 底部分为两条路径：generator（实现）和 evaluator（验收），中间用箭头连接，evaluator 不通过时回到 generator
**Type**: framework
**Filename**: 01-framework-harness-architecture.png

## Illustration 2
**Position**: Section 四 (sprint contract, after "这其实是一种很漂亮的分层" paragraph)
**Purpose**: 可视化 sprint contract 的核心机制——从高层 spec 到可测试目标的收束过程
**Visual Content**: 流程图，从左到右：高层 Spec（模糊的产品意图）→ Sprint Contract（明确的交付物+验收标准）→ 实现 → 验证，contract 用醒目的虚线框标注，标注"这轮交什么、怎么验"
**Type**: flowchart
**Filename**: 02-flowchart-sprint-contract.png

## Illustration 3
**Position**: Section 六 (成本对比, after "evaluator 的价值不是绝对的" paragraph)
**Purpose**: 用数据可视化展示 harness 的成本代价和质量收益，强化"值不值"的工程判断
**Visual Content**: 信息图对比，solo vs full harness vs 优化后的 harness，三个指标卡片：时间（20分钟 vs 6小时 vs 4小时）、成本（$9 vs $200 vs $124）、质量提升幅度
**Type**: infographic
**Filename**: 03-infographic-cost-benefit.png

## Illustration 4
**Position**: Section 八 (Anthropic vs OpenAI, after "两者并不冲突，反而是互补的" paragraph)
**Purpose**: 直观对比两种 harness 设计哲学的差异和互补关系
**Visual Content**: 左右对比图，左侧 Anthropic（运行时编排：失效边界、临时支架、随模型做减法），右侧 OpenAI（系统工程：repo 记忆、lint 约束、反熵清理），中间用"互补"连接
**Type**: comparison
**Filename**: 04-comparison-harness-philosophy.png
