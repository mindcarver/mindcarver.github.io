---
type: mixed
density: balanced
style: notion
palette: default
image_count: 4
article: content/quant/mlqt/回测方法论/06-实战案例.md
---

## Illustration 1
**Position**: 项目概述 (after section overview, before section 1)
**Purpose**: 给读者一个完整的回测流程全局视图，理解 6 个模块之间的关系
**Visual Content**: 横向流程框架图，6 个模块从左到右排列：数据生成 → Alpha模型 → 组合构建 → 执行模拟 → 绩效评估 → 对比分析，箭头连接，每个模块下方标注关键产出
**Type**: framework
**Filename**: 01-framework-backtest-pipeline.png

## Illustration 2
**Position**: Section 3 组合构建 (after 3.2 IC加权组合)
**Purpose**: 直观对比两种组合构建方法的逻辑差异和权重分布特征
**Visual Content**: 左右对比图，左侧 Top-K 等权（条形图，30只股票等高条），右侧 IC 加权（条形图，权重按预测值幂次递减），下方标注各自的优缺点关键词
**Type**: comparison
**Filename**: 02-comparison-portfolio-methods.png

## Illustration 3
**Position**: Section 4 执行模拟 (after 4.2 回测引擎)
**Purpose**: 展示回测引擎的核心执行逻辑，特别是 T+1 限制和成本扣减流程
**Visual Content**: 纵向流程图：信号生成 → T+1 延迟执行 → 计算目标持仓 → 计算调仓量 → 扣减交易成本（佣金+印花税+滑点）→ 更新持仓和现金 → 记录组合价值
**Type**: flowchart
**Filename**: 03-flowchart-execution-engine.png

## Illustration 4
**Position**: Section 5 绩效评估 (after 5.2 完整绩效报告)
**Purpose**: 将分散的绩效指标整合为一个可视化仪表盘概念图
**Visual Content**: 信息图仪表盘布局，4 个区域：左上收益指标（总收益率、年化收益率）、右上风险指标（年化波动率、最大回撤）、左下风险调整（Sharpe比率）、右下交易统计（胜率、日均收益、偏度、峰度）
**Type**: infographic
**Filename**: 04-infographic-performance-dashboard.png
