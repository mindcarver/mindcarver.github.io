# Oh My OpenCode 使用指南

> 把单一的 AI 编程助手变成一个自律的 AI 开发军团

---

## 📚 目录

1. [核心概念](#核心概念)
2. [两种工作模式](#两种工作模式)
3. [命令详解](#命令详解)
4. [Agent 系统](#agent-系统)
5. [技能系统](#技能系统)
6. [实用技巧](#实用技巧)
7. [配置文件](#配置文件)
8. [常见场景](#常见场景)

---

## 核心概念

### 什么是 Oh My OpenCode？

Oh My OpenCode 是 OpenCode 的超级插件，通过多 Agent 协作系统，将单一的 AI 编程助手升级为具备高度自主性的 AI 开发军团。

### 核心魔法词：`ultrawork` / `ulw`

在提示词中加入 `ultrawork` 或 `ulw`，即可触发全自动工作模式：

| 能力 | 说明 |
|------|------|
| 🔍 自动探索 | 深度分析代码库结构和依赖关系 |
| 📋 自动计划 | 智能拆解任务，制定执行步骤 |
| 🤖 自动分配 | 调用专家 Agent 处理各自专长 |
| ✅ 自动验证 | 运行测试和诊断确保质量 |
| 🔄 持续迭代 | 任务未完成则自动优化直到 100% |

---

## 两种工作模式

### 模式一：Ultrawork 模式 (快速工作)

**适用场景**：快速任务、明确目标、信任 AI 自主决策

#### 基本语法

在提示词中加入 `"ulw"` 或 `"ultrawork"`：

```bash
# 功能开发
ulw 添加用户认证功能到我的 Next.js 应用

# Bug 修复
ulw 修复所有测试失败

# 代码重构
ulw 重构登录页面，添加更好的 UI

# 性能优化
ulw 优化数据库查询性能
```

#### 工作流程

```
用户输入 ulw 任务
       ↓
Sisyphus 接收，分析意图
       ↓
自动探索代码库模式
       ↓
并行调用专家 Agent
  (Hephaestus, Oracle, Librarian...)
       ↓
验证结果（测试、诊断）
       ↓
   未完成？继续迭代
       ↓
  100% 完成 → 结束
```

---

### 模式二：Prometheus 模式 (精密工作)

**适用场景**：复杂项目、关键生产变更、需要文档化决策

#### 执行步骤

**步骤 1：进入 Prometheus 模式**

```bash
# 在 OpenCode TUI 中按 Tab 键
# 进入 Prometheus 规划师模式
```

**步骤 2：描述工作**

```bash
我要为电商系统添加完整的支付流程，支持多种支付方式
```

**步骤 3：接受采访**

Prometheus 会通过一系列问题来明确需求：

```
Prometheus: 需要支持哪些支付方式？
你: 支付宝、微信支付、信用卡

Prometheus: 是否需要支付失败重试机制？
你: 是的，最多重试 3 次

Prometheus: 订单状态如何流转？
你: 待支付 → 已支付 → 发货中 → 已完成
```

**步骤 4：确认计划**

Prometheus 生成详细计划并保存到：

```
.sisyphus/plans/支付流程计划.md
```

计划包含：
- ✓ 任务列表
- ✓ 验收标准
- ✓ 风险评估
- ✓ 技术选型建议

**步骤 5：执行计划**

```bash
# 执行 Prometheus 生成的计划
/start-work

# 或指定特定计划
/start-work 支付流程计划
```

---

## 命令详解

### 核心命令列表

| 命令 | 功能 | 示例 |
|------|------|------|
| `ulw` / `ultrawork` | 全自动工作模式 | `ulw 添加认证` |
| `/start-work` | 执行 Prometheus 计划 | `/start-work` |
| `/init-deep` | 生成项目上下文文件 | `/init-deep` |
| `/ulw-loop` | Ultrawork 持续循环 | `/ulw-loop "构建 API"` |
| `/ralph-loop` | 自引用开发循环 | `/ralph-loop "重构模块"` |
| `/cancel-ralph` | 取消 Ralph Loop | `/cancel-ralph` |

---

### `/init-deep` - 深度初始化

在项目中生成层级化的 `AGENTS.md` 文件系统。

#### 用法

```bash
# 基础用法
/init-deep

# 强制创建新文件
/init-deep --create-new

# 限制深度
/init-deep --max-depth=3
```

#### 生成的文件结构

```
project/
├── AGENTS.md              # 全局架构
├── src/
│   ├── AGENTS.md          # src 级别规范
│   └── components/
│       └── AGENTS.md      # 组件级详情
```

---

### `/ulw-loop` - 持续工作循环

让 AI 一直工作直到任务 100% 完成。

#### 用法

```bash
# 基础用法
/ulw-loop "实现完整的用户管理系统"

# 带最大迭代次数
/ulw-loop "重构所有旧代码" --max-iterations=50
```

---

### `/ralph-loop` - 自引用循环

更智能的循环，AI 会自我评估和改进。

#### 用法

```bash
# 开始循环
/ralph-loop "构建一个博客系统"

# 设置最大迭代次数
/ralph-loop "优化性能" --max-iterations=100

# 停止循环
/cancel-ralph
```

---

## Agent 系统

### 专家 Agent 介绍

#### Sisyphus (指挥官)

| 属性 | 说明 |
|------|------|
| 模型 | `glm-5` (你已配置) |
| 职责 | 总指挥，制定计划，分配任务 |
| 特点 | 极度自律，任务不完成绝不罢休 |

#### Hephaestus (工匠)

| 属性 | 说明 |
|------|------|
| 模型 | `gpt-5.3-codex` / `glm-5` |
| 职责 | 深度自主执行代码任务 |
| 特点 | 独立探索模式，不需要保姆式指导 |

#### Prometheus (规划师)

| 属性 | 说明 |
|------|------|
| 模型 | `glm-5` |
| 职责 | 战略规划，通过访谈明确需求 |
| 特点 | 动手前先做好完整计划 |

#### Oracle (分析者)

| 属性 | 说明 |
|------|------|
| 模型 | `glm-5` |
| 职责 | 只读代码审查，架构分析 |
| 特点 | 高度推理，不修改代码 |

#### Librarian (图书管理员)

| 属性 | 说明 |
|------|------|
| 模型 | `glm-5` |
| 职责 | 多仓库分析，文档模式查找 |
| 特点 | 找相似的代码实现 |

#### Explore (探索者)

| 属性 | 说明 |
|------|------|
| 模型 | `glm-5` |
| 职责 | 快速代码库搜索 |
| 特点 | 定位特定功能的文件 |

---

### Agent 调用语法

使用 `@agent-name` 直接调用专家 Agent：

```bash
# 让 Oracle 审查架构
@oracle 审查这个设计并提议架构方案

# 让 Librarian 查找实现模式
@librarian 这个功能是如何实现的？找找类似模式

# 让 Explore 搜索文件
@explore 找所有支付相关的文件
```

---

## 技能系统

### 内置技能

| 技能 | 功能 | 使用场景 |
|------|------|----------|
| `playwright` | 浏览器自动化 | 网页测试、截图、交互 |
| `agent-browser` | Agent 专用浏览器 | 爬虫、表单填写 |
| `git-master` | Git 专家 | 原子提交、rebase 操作 |
| `frontend-ui-ux` | UI/UX 设计 | 前端界面设计 |

---

### `agent-browser` 使用示例

```bash
# 打开网页
agent-browser open https://example.com

# 获取页面快照（带元素引用）
agent-browser snapshot -i
# 输出: textbox "Email" [ref=e1], button "Submit" [ref=e3]

# 填写表单
agent-browser fill @e1 "user@example.com"
agent-browser fill @e2 "password123"

# 点击按钮
agent-browser click @e3

# 等待加载完成
agent-browser wait --load networkidle

# 截图
agent-browser screenshot result.png

# 关闭浏览器
agent-browser close
```

---

### 禁用特定技能

编辑配置文件 `~/.config/opencode/oh-my-opencode.json`：

```json
{
  "disabled_skills": ["playwright", "agent-browser"]
}
```

---

## 实用技巧

### 1. 组合使用技巧

```bash
# 精密 + 快速：先用 Prometheus 规划，再用 ulw 执行
[按 Tab] → Prometheus 计划 → /start-work

# 快速迭代：直接 ulw
ulw 实现登录功能

# 长期项目：ralph-loop
/ralph-loop "这个周末构建一个完整的 SaaS"
```

---

### 2. 项目初始化工作流

```bash
# 1. 进入新项目
cd my-project

# 2. 生成上下文文件
/init-deep

# 3. 开始工作
ulw 添加基础功能
```

---

### 3. 处理复杂任务

```bash
# 使用 Prometheus 模式
[按 Tab] 进入 Prometheus
→ 描述复杂需求
→ 回答采访问题
→ 查看生成的计划 .sisyphus/plans/*.md
→ /start-work 执行
```

---

### 4. 快速修复 Bug

```bash
# 直接用 ultrawork
ulw 修复登录页面的验证问题
```

---

### 5. 重构代码

```bash
# 使用 ralph-loop 持续优化
/ralph-loop "重构整个用户模块，提升性能"
```

---

## 配置文件

### 文件位置

| 文件 | 位置 | 作用 |
|------|------|------|
| 主配置 | `~/.config/opencode/opencode.json` | MCP、插件配置 |
| Agent配置 | `~/.config/opencode/oh-my-opencode.json` | 模型映射 |
| 项目配置 | `.opencode/oh-my-opencode.json` | 项目级覆盖 |
| 计划文件 | `.sisyphus/plans/*.md` | Prometheus 生成的计划 |

---

## 常见场景

### 场景 1：新功能开发

```bash
ulw 添加用户评论功能，支持回复和点赞
```

---

### 场景 2：Bug 修复

```bash
ulw 修复支付页面在 Safari 上的显示问题
```

---

### 场景 3：性能优化

```bash
ulw 优化首页加载速度，目标 LCP < 2s
```

---

### 场景 4：代码重构

```bash
/ralph-loop "把所有 callback 改成 async/await"
```

---

### 场景 5：复杂项目

```bash
[Tab] → Prometheus 模式
→ "构建一个完整的电商后台"
→ 回答问题
→ /start-work
```

---

## 快速参考

### 命令速查表

| 模式 | 触发方式 | 适用场景 |
|------|----------|----------|
| Ultrawork | `ulw <任务>` | 快速开发、明确目标 |
| Prometheus | `[Tab]` 键 | 复杂项目、需要规划 |
| 持续循环 | `/ulw-loop "<任务>"` | 长期任务、需要迭代 |
| 自引用循环 | `/ralph-loop "<任务>"` | 自我优化、智能改进 |

### Agent 速查表

| Agent | 擅长 | 调用方式 |
|-------|------|----------|
| Sisyphus | 总指挥 | 自动触发 |
| Hephaestus | 代码实现 | 自动调用 |
| Prometheus | 需求规划 | `[Tab]` 键 |
| Oracle | 架构审查 | `@oracle` |
| Librarian | 模式查找 | `@librarian` |
| Explore | 文件搜索 | `@explore` |
